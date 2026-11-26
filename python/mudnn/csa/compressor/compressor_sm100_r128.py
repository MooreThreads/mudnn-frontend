# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Dedicated ratio-128 CSA/HCA compressor kernels for MUSA TileLang.

Forward selects between the tilewarp and rowgroup APE-TMA kernels for C1/C2
row buckets. The rowgroup path has one warp per output row, while tilewarp
shares an APE tile across four warps and merges their online-softmax partials.
Backward selects rowgroup or tilewarp by row capacity and reduces sharded dAPE.
"""

import threading

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

import torch

import tilelang
import tilelang.language as T


from ._jit import COMPILER_COMPILE_FLAGS as _COMPILE_FLAGS
from ._jit import COMPILER_PASS_CONFIGS as _PASS_CONFIGS

_LOG2E = 1.4426950408889634
_RCP4_SOURCE = """
#include <tl_templates/musa/common.h>
TL_DEVICE void csa_rcp4_ptr(float* dst, const float* src) {
  const tl_f4 x = *reinterpret_cast<const tl_f4*>(src);
  const tl_f4 one{1.0f, 1.0f, 1.0f, 1.0f};
  *reinterpret_cast<tl_f4*>(dst) = one / x;
}
"""

SUPPORTED_HEAD_DIMS = (128, 512)

_COMPILED = {}
_COMPILE_LOCK = threading.Lock()

# Row counts include the output capacity. Small measured buckets use tilewarp;
# the remaining buckets use the rowgroup APE-TMA builder. Segment count only
# selects the address-calculation path.
_FORWARD_TMA_ROW_BUCKETS = {
    (1, 128): ((64, (1, 16)), (576, (4, 16)),
               (4095, (16, 64)), (None, (8, 32))),
    (1, 512): ((160, (4, 16)), (4095, (8, 32)),
               (8191, (4, 16)), (None, (16, 64))),
    (2, 128): ((64, (1, 16)), (576, (2, 16)), (None, (8, 32))),
    (2, 512): ((160, (4, 16)), (None, (16, 64))),
}
_FORWARD_TILEWARP_ROW_BUCKETS = {
    # Measured small-row cutovers. D512 returns to rowgroup TMA at R65;
    # D128 keeps tilewarp through the useful R512/R1024 bucket.
    (1, 128): (512, (1, 16, 4, 4)),
    (2, 128): (1024, (1, 16, 4, 4)),
    (1, 512): (64, (1, 32, 4, 8)),
    (2, 512): (64, (1, 32, 4, 8)),
}

# Forward


def _build_forward_program_r128_rowgroup_ape_tma(
    head_dim: int,
    ape_tile_tokens: int,
    row_tile: int = 8,
    single_segment: bool = True,
    coff: int = 1,
):
    """R128 C1/C2 row-tile forward with a double-buffered APE tile.

    This follows the R4 rowgroup layout: one warp owns one row and the CTA
    contains ``row_tile`` warps.  There is deliberately no second row-group
    dimension.  The softmax scratch is reused for every chunk and the only
    state that survives a chunk is the current row's max, denominator, and
    accumulator. C2 first consumes the previous block's first half when that
    block exists, then consumes the current block's second half.
    """

    chunk_tokens = 4
    if head_dim not in (128, 512):
        raise ValueError(f"unsupported head_dim={head_dim}")
    if coff not in (1, 2):
        raise ValueError(f"unsupported coff={coff}")
    if row_tile not in (1, 2, 4, 8, 16):
        raise ValueError(f"unsupported row_tile={row_tile}")
    if ape_tile_tokens not in (8, 16, 32, 64, 128):
        raise ValueError(f"unsupported ape_tile_tokens={ape_tile_tokens}")
    if ape_tile_tokens % chunk_tokens:
        raise ValueError("ape_tile_tokens must be divisible by chunk_tokens")

    ratio = 128
    vector = 4
    threads_x = 32
    columns_per_cta = threads_x * vector
    column_ctas = head_dim // columns_per_cta
    ape_tiles = ratio // ape_tile_tokens
    total_ape_tiles = coff * ape_tiles
    width = coff * head_dim
    chunks_per_tile = ape_tile_tokens // chunk_tokens
    total_tokens = T.dynamic("total_tokens")
    total_comp = T.dynamic("total_comp")
    num_offsets = T.dynamic("num_offsets")

    @T.prim_func
    def main(
        kv: T.Tensor((total_tokens, width), T.bfloat16),
        score: T.Tensor((total_tokens, width), T.bfloat16),
        ape: T.Tensor((ratio, width), T.float32),
        cu_seqlens: T.Tensor((num_offsets,), T.int32),
        cu_seqlens_comp: T.Tensor((num_offsets,), T.int32),
        out: T.Tensor((total_comp, head_dim), T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(total_comp, row_tile),
            column_ctas,
            threads=row_tile * threads_x,
        ) as (row_cta, col_cta):
            T.import_source(_RCP4_SOURCE)
            tx = T.get_thread_binding()
            warp = tx // threads_x
            lane = tx % threads_x
            column_base = (col_cta * threads_x + lane) * vector

            ape_shared = T.alloc_shared(
                (2, ape_tile_tokens, columns_per_cta), T.float32
            )
            T.annotate_layout(
                {ape_shared: T.Layout(ape_shared.shape, lambda *idx: idx)}
            )

            row = row_cta * row_tile + warp
            valid_row = row < total_comp
            if single_segment:
                row_in_pack = valid_row & (row < cu_seqlens_comp[1])
                active_row = T.if_then_else(
                    row_in_pack, row, 0
                )
                block = active_row - cu_seqlens_comp[0]
                row_token = cu_seqlens[0] + block * ratio
            else:
                num_segments = num_offsets - 1
                nb_valid = cu_seqlens_comp[num_segments]
                active_row = T.if_then_else(valid_row, row, 0)
                row_in_pack = active_row < nb_valid
                segment = T.alloc_var(dtype=T.int32)
                segment = 0
                for s in T.serial(num_segments):
                    segment = T.if_then_else(
                        row_in_pack
                        & (active_row >= cu_seqlens_comp[s])
                        & (active_row < cu_seqlens_comp[s + 1]),
                        s,
                        segment,
                    )
                block = T.if_then_else(
                    row_in_pack, active_row - cu_seqlens_comp[segment], 0
                )
                row_token = cu_seqlens[segment] + block * ratio

            max_score = T.alloc_local((vector,), T.float32)
            denom = T.alloc_local((vector,), T.float32)
            accum = T.alloc_local((vector,), T.float32)
            old_scale = T.alloc_local((vector,), T.float32)
            logits = T.alloc_local((chunk_tokens, vector), T.float32)
            values = T.alloc_local((chunk_tokens, vector), T.bfloat16)
            probabilities = T.alloc_local((chunk_tokens, vector), T.float32)
            ape_local = T.alloc_local((vector,), T.float32)
            score_local = T.alloc_local((vector,), T.bfloat16)

            for i in T.unroll(vector):
                max_score[i] = -T.infinity(T.float32)
                denom[i] = 0.0
                accum[i] = 0.0

            # Warp 0 issues APE TME. All row warps wait for the same tile.
            ape_ready = T.alloc_barrier([threads_x, threads_x])
            if warp == 0:
                T.tma_copy(
                    ape[0:ape_tile_tokens,
                        col_cta * columns_per_cta:(col_cta + 1) * columns_per_cta],
                    ape_shared[0, :, :],
                    barrier=ape_ready[0],
                    leader_scope_threads=threads_x,
                )
                T.barrier_arrive(ape_ready[0])
            T.mbarrier_wait_parity(ape_ready[0], 0)
            T.sched_boundary()
            T.sync_threads()

            for ape_tile in T.unroll(
                total_ape_tiles, explicit=False, unroll_factor=1
            ):
                stage = ape_tile % 2
                half = ape_tile // ape_tiles
                tile_in_half = ape_tile % ape_tiles
                next_tile = T.min(ape_tile + 1, total_ape_tiles - 1)
                next_half = next_tile // ape_tiles
                next_tile_in_half = next_tile % ape_tiles
                if ape_tile < total_ape_tiles - 1:
                    if warp == 0:
                        T.tma_copy(
                            ape[
                                next_tile_in_half * ape_tile_tokens : (next_tile_in_half + 1) * ape_tile_tokens,
                                next_half * head_dim + col_cta * columns_per_cta :
                                next_half * head_dim + (col_cta + 1) * columns_per_cta,
                            ],
                            ape_shared[1 - stage, :, :],
                            barrier=ape_ready[1 - stage],
                            leader_scope_threads=threads_x,
                        )
                        T.barrier_arrive(ape_ready[1 - stage])

                # Capacity rows beyond the packed segments use block zero of
                # segment zero, matching the eager gather reference.
                if valid_row & ((coff == 1) | (block > 0) | (half == 1)):
                    for chunk in T.unroll(
                        chunks_per_tile, explicit=False, unroll_factor=1
                    ):
                        for token_in_chunk in T.unroll(
                            chunk_tokens, unroll_factor=chunk_tokens
                        ):
                            token_in_tile = chunk * chunk_tokens + token_in_chunk
                            token_offset = tile_in_half * ape_tile_tokens + token_in_tile
                            for i in T.vectorized(vector):
                                ape_local[i] = ape_shared[
                                    stage, token_in_tile, lane * vector + i
                                ] * _LOG2E
                            if coff == 2:
                                token = row_token + token_offset - T.if_then_else(
                                    half == 0, ratio, 0
                                )
                            else:
                                token = row_token + token_offset
                            input_column_base = column_base + half * head_dim
                            input_column_end = input_column_base + vector
                            score_local[0:4] = T.reinterpret(
                                T.ldg64(score[token, input_column_base:input_column_end]),
                                "bfloat16x4",
                            )
                            values[token_in_chunk, 0:4] = T.reinterpret(
                                T.ldg64(kv[token, input_column_base:input_column_end]),
                                "bfloat16x4",
                            )
                            for i in T.vectorized(vector):
                                logits[token_in_chunk, i] = (
                                    score_local[i].astype(T.float32) * _LOG2E
                                    + ape_local[i]
                                )

                        for i in T.vectorized(vector):
                            max01 = T.max(logits[0, i], logits[1, i])
                            max23 = T.max(logits[2, i], logits[3, i])
                            new_max = T.max(max_score[i], T.max(max01, max23))
                            old_scale[i] = T.exp2(max_score[i] - new_max)
                            max_score[i] = new_max
                        for i in T.vectorized(vector):
                            denom[i] *= old_scale[i]
                            accum[i] *= old_scale[i]
                        for token_in_chunk in T.unroll(
                            chunk_tokens, unroll_factor=chunk_tokens
                        ):
                            for i in T.vectorized(vector):
                                probabilities[token_in_chunk, i] = T.exp2(
                                    logits[token_in_chunk, i] - max_score[i]
                                )
                            for i in T.vectorized(vector):
                                prob = probabilities[token_in_chunk, i]
                                denom[i] += prob
                                accum[i] = values[token_in_chunk, i].astype(T.float32) * prob + accum[i]

                if ape_tile < total_ape_tiles - 1:
                    T.mbarrier_wait_parity(
                        ape_ready[1 - stage], ((ape_tile + 1) // 2) & 1
                    )
                    T.sched_boundary()
                    T.sync_threads()

            if valid_row:
                inv_denom = T.alloc_local((vector,), T.float32)
                T.evaluate(T.call_extern(
                    "void", "csa_rcp4_ptr",
                    T.access_ptr(inv_denom, "w"), T.access_ptr(denom, "r"),
                ))
                for i in T.vectorized(vector):
                    out[row, column_base + i] = (
                        accum[i] * inv_denom[i]
                    ).astype(T.bfloat16)

    return main



def _build_forward_program_r128_tilewarp(
    head_dim: int,
    rows_per_cta: int = 2,
    single_segment: bool = True,
    coff: int = 1,
    ape_tile_tokens: int = 32,
    warps_per_cta: int = 4,
    chunk_tokens: int = 4,
    vector: int = 4,
):
    if head_dim not in (128, 512) or coff not in (1, 2):
        raise ValueError((head_dim, coff))
    if rows_per_cta not in (1, 2, 4, 8):
        raise ValueError(rows_per_cta)
    if ape_tile_tokens not in (8, 16, 32, 64, 128):
        raise ValueError(ape_tile_tokens)
    if warps_per_cta not in (1, 2, 4, 8):
        raise ValueError(warps_per_cta)
    if chunk_tokens not in (4, 8):
        raise ValueError(chunk_tokens)
    if vector != 4:
        raise ValueError(
            "only vector=4 is supported: vector=2 APE lowering is invalid"
        )

    ratio = 128
    if ratio % ape_tile_tokens or ape_tile_tokens % warps_per_cta:
        raise ValueError("ape_tile_tokens must divide ratio and warps")
    tokens_per_warp = ape_tile_tokens // warps_per_cta
    if tokens_per_warp % chunk_tokens:
        raise ValueError("tokens_per_warp must be divisible by chunk_tokens")
    threads_x = 32
    columns_per_cta = threads_x * vector
    if head_dim % columns_per_cta:
        raise ValueError("head_dim must be divisible by columns_per_cta")
    column_ctas = head_dim // columns_per_cta
    ape_tiles = ratio // ape_tile_tokens
    total_ape_tiles = coff * ape_tiles
    width = coff * head_dim
    load_dtype = "bfloat16x4"
    total_tokens = T.dynamic("total_tokens")
    total_comp = T.dynamic("total_comp")
    num_offsets = T.dynamic("num_offsets")

    @T.prim_func
    def main(
        kv: T.Tensor((total_tokens, width), T.bfloat16),
        score: T.Tensor((total_tokens, width), T.bfloat16),
        ape: T.Tensor((ratio, width), T.float32),
        cu_seqlens: T.Tensor((num_offsets,), T.int32),
        cu_seqlens_comp: T.Tensor((num_offsets,), T.int32),
        out: T.Tensor((total_comp, head_dim), T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(total_comp, rows_per_cta),
            column_ctas,
            threads=warps_per_cta * threads_x,
        ) as (row_cta, col_cta):
            T.import_source(_RCP4_SOURCE)
            tx = T.get_thread_binding()
            warp = tx // threads_x
            lane = tx % threads_x
            column_base = (col_cta * threads_x + lane) * vector
            column = lane * vector

            ape_shared = T.alloc_shared(
                (2, ape_tile_tokens, columns_per_cta), T.float32
            )
            T.annotate_layout(
                {ape_shared: T.Layout(ape_shared.shape, lambda *idx: idx)}
            )
            partial_max = T.alloc_shared((warps_per_cta, columns_per_cta), T.float32)
            partial_denom = T.alloc_shared((warps_per_cta, columns_per_cta), T.float32)
            partial_accum = T.alloc_shared((warps_per_cta, columns_per_cta), T.float32)

            max_score = T.alloc_local((rows_per_cta, vector), T.float32)
            denom = T.alloc_local((rows_per_cta, vector), T.float32)
            accum = T.alloc_local((rows_per_cta, vector), T.float32)
            old_scale = T.alloc_local((vector,), T.float32)
            logits = T.alloc_local((chunk_tokens, vector), T.float32)
            values = T.alloc_local((chunk_tokens, vector), T.bfloat16)
            probabilities = T.alloc_local((chunk_tokens, vector), T.float32)
            ape_local = T.alloc_local((vector,), T.float32)
            score_local = T.alloc_local((vector,), T.bfloat16)
            merged_max = T.alloc_local((vector,), T.float32)
            merged_denom = T.alloc_local((vector,), T.float32)
            merged_accum = T.alloc_local((vector,), T.float32)
            merge_scale = T.alloc_local((vector,), T.float32)
            inv_denom = T.alloc_local((vector,), T.float32)

            for row_iter in T.unroll(rows_per_cta):
                for i in T.vectorized(vector):
                    max_score[row_iter, i] = -T.infinity(T.float32)
                    denom[row_iter, i] = 0.0
                    accum[row_iter, i] = 0.0

            ape_ready = T.alloc_barrier([threads_x, threads_x])
            if warp == 0:
                T.tma_copy(
                    ape[0:ape_tile_tokens,
                        col_cta * columns_per_cta:(col_cta + 1) * columns_per_cta],
                    ape_shared[0, :, :],
                    barrier=ape_ready[0],
                    leader_scope_threads=threads_x,
                )
                T.barrier_arrive(ape_ready[0])
            T.mbarrier_wait_parity(ape_ready[0], 0)
            T.sched_boundary()
            T.sync_threads()

            for ape_tile in T.unroll(
                total_ape_tiles, explicit=False, unroll_factor=1
            ):
                stage = ape_tile % 2
                half = ape_tile // ape_tiles
                tile_in_half = ape_tile % ape_tiles
                next_tile = T.min(ape_tile + 1, total_ape_tiles - 1)
                next_half = next_tile // ape_tiles
                next_tile_in_half = next_tile % ape_tiles
                if ape_tile < total_ape_tiles - 1:
                    if warp == 0:
                        T.tma_copy(
                            ape[
                                next_tile_in_half * ape_tile_tokens : (next_tile_in_half + 1) * ape_tile_tokens,
                                next_half * head_dim + col_cta * columns_per_cta :
                                next_half * head_dim + (col_cta + 1) * columns_per_cta,
                            ],
                            ape_shared[1 - stage, :, :],
                            barrier=ape_ready[1 - stage],
                            leader_scope_threads=threads_x,
                        )
                        T.barrier_arrive(ape_ready[1 - stage])

                # Every warp consumes its own eight tokens, and the same APE
                # tile is reused for each output row before changing stages.
                for row_iter in T.unroll(rows_per_cta, explicit=False, unroll_factor=1):
                    row = row_cta * rows_per_cta + row_iter
                    valid_row = row < total_comp
                    if single_segment:
                        row_in_pack = valid_row & (row < cu_seqlens_comp[1])
                        active_row = T.if_then_else(row_in_pack, row, 0)
                        block = active_row - cu_seqlens_comp[0]
                        row_token = cu_seqlens[0] + block * ratio
                    else:
                        num_segments = num_offsets - 1
                        nb_valid = cu_seqlens_comp[num_segments]
                        active_row = T.if_then_else(valid_row, row, 0)
                        row_in_pack = active_row < nb_valid
                        segment = T.alloc_var(dtype=T.int32)
                        segment = 0
                        for s in T.serial(num_segments):
                            segment = T.if_then_else(
                                row_in_pack
                                & (active_row >= cu_seqlens_comp[s])
                                & (active_row < cu_seqlens_comp[s + 1]),
                                s, segment,
                            )
                        block = T.if_then_else(
                            row_in_pack, active_row - cu_seqlens_comp[segment], 0
                        )
                        row_token = cu_seqlens[segment] + block * ratio

                    if valid_row & ((coff == 1) | (block > 0) | (half == 1)):
                        for chunk in T.unroll(tokens_per_warp // chunk_tokens, explicit=False, unroll_factor=1):
                            for k in T.unroll(chunk_tokens):
                                token_in_tile = warp * tokens_per_warp + chunk * chunk_tokens + k
                                token_offset = tile_in_half * ape_tile_tokens + token_in_tile
                                for i in T.vectorized(vector):
                                    ape_local[i] = ape_shared[
                                        stage, token_in_tile, column + i
                                    ] * _LOG2E
                                if coff == 2:
                                    token = row_token + token_offset - T.if_then_else(
                                        half == 0, ratio, 0
                                    )
                                else:
                                    token = row_token + token_offset
                                input_column_base = column_base + half * head_dim
                                input_column_end = input_column_base + vector
                                score_local[0:vector] = T.reinterpret(
                                    T.ldg64(
                                        score[token, input_column_base:input_column_end]
                                    ),
                                    load_dtype,
                                )
                                values[k, 0:vector] = T.reinterpret(
                                    T.ldg64(
                                        kv[token, input_column_base:input_column_end]
                                    ),
                                    load_dtype,
                                )
                                for i in T.vectorized(vector):
                                    logits[k, i] = (
                                        score_local[i].astype(T.float32) * _LOG2E
                                        + ape_local[i]
                                    )
                            for i in T.vectorized(vector):
                                if chunk_tokens == 4:
                                    chunk_max = T.max(
                                        T.max(logits[0, i], logits[1, i]),
                                        T.max(logits[2, i], logits[3, i]),
                                    )
                                else:
                                    chunk_max = T.max(
                                        T.max(
                                            T.max(logits[0, i], logits[1, i]),
                                            T.max(logits[2, i], logits[3, i]),
                                        ),
                                        T.max(
                                            T.max(logits[4, i], logits[5, i]),
                                            T.max(logits[6, i], logits[7, i]),
                                        ),
                                    )
                                new_max = T.max(max_score[row_iter, i], chunk_max)
                                old_scale[i] = T.exp2(max_score[row_iter, i] - new_max)
                                max_score[row_iter, i] = new_max
                            for i in T.vectorized(vector):
                                denom[row_iter, i] *= old_scale[i]
                                accum[row_iter, i] *= old_scale[i]
                            for k in T.unroll(chunk_tokens):
                                for i in T.vectorized(vector):
                                    probabilities[k, i] = T.exp2(
                                        logits[k, i] - max_score[row_iter, i]
                                    )
                                for i in T.vectorized(vector):
                                    prob = probabilities[k, i]
                                    denom[row_iter, i] += prob
                                    accum[row_iter, i] = (
                                        values[k, i].astype(T.float32) * prob
                                        + accum[row_iter, i]
                                    )

                if ape_tile < total_ape_tiles - 1:
                    T.mbarrier_wait_parity(
                        ape_ready[1 - stage], ((ape_tile + 1) // 2) & 1
                    )
                    T.sched_boundary()
                    T.sync_threads()

            # Keep the final shared-partial merge loop runtime-serial so the
            # per-row partial buffers are reused instead of being expanded.
            for row_iter in T.unroll(rows_per_cta, explicit=False, unroll_factor=1):
                row = row_cta * rows_per_cta + row_iter
                for i in T.vectorized(vector):
                    partial_max[warp, column + i] = max_score[row_iter, i]
                    partial_denom[warp, column + i] = denom[row_iter, i]
                    partial_accum[warp, column + i] = accum[row_iter, i]
                T.sync_threads()

                if warp == 0:
                    for i in T.vectorized(vector):
                        merged_max[i] = -T.infinity(T.float32)
                        merged_denom[i] = 0.0
                        merged_accum[i] = 0.0
                    for w in T.unroll(warps_per_cta):
                        for i in T.vectorized(vector):
                            merged_max[i] = T.max(
                                merged_max[i], partial_max[w, column + i]
                            )
                    for w in T.unroll(warps_per_cta):
                        for i in T.vectorized(vector):
                            merge_scale[i] = T.exp2(
                                partial_max[w, column + i] - merged_max[i]
                            )
                        for i in T.vectorized(vector):
                            merged_denom[i] += partial_denom[w, column + i] * merge_scale[i]
                            merged_accum[i] += partial_accum[w, column + i] * merge_scale[i]
                    if row < total_comp:
                        T.evaluate(T.call_extern(
                            "void", "csa_rcp4_ptr",
                            T.access_ptr(inv_denom, "w"),
                            T.access_ptr(merged_denom, "r"),
                        ))
                        for i in T.vectorized(vector):
                            out[row, column_base + i] = (
                                merged_accum[i] * inv_denom[i]
                            ).astype(T.bfloat16)
                T.sync_threads()

    return main


def _forward_kernel_schedule_r128(ratio, d, coff, nb_total, num_offsets):
    """Select the tilewarp or rowgroup-TMA builder for a row bucket.

    Tilewarp is used only for measured small-row buckets. Its extra
    cross-warp partial merge is amortized at low row counts, while the
    rowgroup-TMA builder wins once enough rows are available to reuse the APE
    tile efficiently.
    """

    if ratio != 128 or coff not in (1, 2) or d not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"unsupported R128 forward shape: ratio={ratio}, d={d}, coff={coff}")
    single_segment = num_offsets == 2
    tilewarp_max, tilewarp_config = _FORWARD_TILEWARP_ROW_BUCKETS[(coff, d)]
    if nb_total is not None and nb_total <= tilewarp_max:
        return ("tilewarp", *tilewarp_config, single_segment)
    for max_rows, tma_tile in _FORWARD_TMA_ROW_BUCKETS[(coff, d)]:
        if max_rows is None or (nb_total is not None and nb_total <= max_rows):
            return ("tma", *tma_tile, single_segment)
    raise AssertionError("forward TMA schedule requires a final open bucket")


def _compile_fwd_r128(key, ratio, d, coff, schedule):
    """JIT-compile the forward launch entry for ``key`` (capture-guarded)."""

    with _COMPILE_LOCK:
        kernel = _COMPILED.get(key)
        if kernel is None:
            query = getattr(torch.musa, "is_current_stream_capturing", None)
            if query is not None and query():
                raise RuntimeError(
                    f"CSA compressor (r128): first call for config {key} happened under "
                    "MUSA graph capture (JIT compilation is not capture-safe); compile() "
                    "or run one eager step for this configuration before capturing."
                )
            if schedule[0] == "tma":
                _, row_tile, ape_tile_tokens, single_segment = schedule
                program = _build_forward_program_r128_rowgroup_ape_tma(
                    int(d), ape_tile_tokens, row_tile=row_tile,
                    coff=int(coff), single_segment=single_segment,
                )
            elif schedule[0] == "tilewarp":
                _, rows_per_cta, ape_tile_tokens, warps_per_cta, chunk_tokens, single_segment = schedule
                program = _build_forward_program_r128_tilewarp(
                    int(d), rows_per_cta=rows_per_cta,
                    ape_tile_tokens=ape_tile_tokens,
                    warps_per_cta=warps_per_cta,
                    chunk_tokens=chunk_tokens,
                    coff=int(coff), single_segment=single_segment,
                )
            else:
                raise ValueError(f"unsupported R128 forward schedule: {schedule}")
            kernel = tilelang.compile(
                program,
                target="musa",
                pass_configs=_PASS_CONFIGS,
                compile_flags=_COMPILE_FLAGS,
            )
            _COMPILED[key] = kernel
    return kernel


def precompile_fwd_r128(ratio, d, coff, device, nb_total=None, num_offsets=None):
    """Ensure the ratio=128 forward kernel(s) for this configuration are JIT-compiled.

    With ``nb_total`` given, compiles exactly the schedule bucket that shape will use;
    without it, compiles every row and segment bucket this config can select at
    runtime, so a subsequent MUSA-graph capture cannot hit a cold bucket.
    """

    if nb_total is not None:
        candidates = [nb_total]
    else:
        # Compile one representative from each measured row bucket.
        candidates = [1] + [
            max_rows + 1
            for max_rows, _ in _FORWARD_TMA_ROW_BUCKETS[(coff, d)]
            if max_rows is not None
        ]
        # Also cover the tilewarp-to-TMA cutovers so graph capture does not
        # encounter a cold JIT for a newly selected builder.
        candidates += [65]
        if d == 128:
            candidates += [513 if coff == 1 else 1025]
        candidates = sorted(set(candidates))
    offsets = (2, 3) if num_offsets is None else (num_offsets,)
    schedules = {
        _forward_kernel_schedule_r128(ratio, d, coff, nb, n_offsets)
        for nb in candidates for n_offsets in offsets
    }

    if device is None:
        raise RuntimeError("CSA TileLang compilation needs a concrete MUSA device")
    with torch.musa.device(device):
        for schedule in schedules:
            key = ("r128fwd", ratio, d, coff, schedule, device.index)
            if key in _COMPILED:
                continue
            _compile_fwd_r128(key, ratio, d, coff, schedule)

def run_fwd_r128(
    kv,
    score,
    ape,
    cu_seqlens,
    cu_seqlens_comp,
    out,
    nb_total,
    ratio,
    d,
    coff,
    stream_handle=None,
):
    """Launch the cached ratio-128 forward kernel on the current MUSA stream."""

    del stream_handle
    if int(nb_total) == 0:
        return
    dev = kv.device.index
    schedule = _forward_kernel_schedule_r128(
        ratio, d, coff, int(nb_total), int(cu_seqlens_comp.numel())
    )
    key = ("r128fwd", ratio, d, coff, schedule, dev)
    with torch.musa.device(dev):
        kernel = _COMPILED.get(key)
        if kernel is None:
            kernel = _compile_fwd_r128(key, ratio, d, coff, schedule)
        kernel(kv, score, ape, cu_seqlens, cu_seqlens_comp, out)


# Backward


def _build_backward_program_r128_fused_ablation(
    ratio: int,
    head_dim: int,
    coff: int,
    dape_shards: int,
    warps_per_cta: int = 8,
    ape_tile_tokens: int = 32,
):
    """Each warp owns one row; chunk-online softmax uses double-buffered APE TMA."""
    if ratio != 128 or coff not in (1, 2):
        raise ValueError("rowgroup backward requires ratio=128, coff=1/2")
    if head_dim not in (128, 512):
        raise ValueError("rowgroup backward supports head_dim 128 or 512")
    vector = 4
    chunk_tokens = 4
    threads_x = 32
    columns_per_cta = threads_x * vector
    if ape_tile_tokens not in (8, 16, 32, 64, 128):
        raise ValueError("ape_tile_tokens must be 8, 16, 32, 64, or 128")
    if warps_per_cta not in (4, 8):
        raise ValueError("rowgroup backward requires 4 or 8 warps per CTA")
    if dape_shards < 2 or dape_shards & (dape_shards - 1):
        raise ValueError("dape_shards must be a power of two greater than one")
    block_threads = warps_per_cta * threads_x
    column_ctas = head_dim // columns_per_cta
    ape_tiles = ratio // ape_tile_tokens
    total_ape_tiles = coff * ape_tiles
    width = coff * head_dim
    # A C2/D512 tensor reaches 4 GiB at 16384 rows. Promote the token index
    # before the 1024-column stride is applied to global addresses.
    address_index_dtype = T.int64 if width == 1024 else T.int32
    chunks_per_tile = ape_tile_tokens // chunk_tokens
    total_tokens = T.dynamic("total_tokens")
    total_comp = T.dynamic("total_comp")
    num_offsets = T.dynamic("num_offsets")

    @T.prim_func
    def main(
        kv: T.Tensor((total_tokens, width), T.bfloat16),
        score: T.Tensor((total_tokens, width), T.bfloat16),
        ape: T.Tensor((ratio, width), T.float32),
        cu_seqlens: T.Tensor((num_offsets,), T.int32),
        cu_seqlens_comp: T.Tensor((num_offsets,), T.int32),
        grad_out: T.Tensor((total_comp, head_dim), T.bfloat16),
        grad_kv: T.Tensor((total_tokens, width), T.bfloat16),
        grad_score: T.Tensor((total_tokens, width), T.bfloat16),
        grad_ape: T.Tensor((ratio * dape_shards, width), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(total_comp, warps_per_cta), column_ctas, threads=block_threads
        ) as (row_cta, col_cta):
            tx = T.get_thread_binding()
            warp = tx // threads_x
            row = row_cta * warps_per_cta + warp
            lane = tx % threads_x
            column = lane * vector
            col_start = col_cta * columns_per_cta + column
            col_end = col_start + vector
            num_segments = num_offsets - 1
            nb_valid = cu_seqlens_comp[num_segments]
            ape_shared = T.alloc_shared(
                (2, ape_tile_tokens, columns_per_cta), T.float32
            )
            ape_ready = T.alloc_barrier([threads_x, threads_x])
            T.annotate_layout({ape_shared: T.Layout(ape_shared.shape, lambda *idx: idx)})
            score_local = T.alloc_local((vector,), T.bfloat16)
            value_local = T.alloc_local((vector,), T.bfloat16)
            score_chunk = T.alloc_local((chunk_tokens, vector), T.bfloat16)
            value_chunk = T.alloc_local((chunk_tokens, vector), T.bfloat16)
            ape_local = T.alloc_local((vector,), T.float32)
            logits = T.alloc_local((chunk_tokens, vector), T.float32)
            probabilities = T.alloc_local((chunk_tokens, vector), T.float32)
            local_denom = T.alloc_local((vector,), T.float32)
            local_accum = T.alloc_local((vector,), T.float32)
            local_max = T.alloc_local((vector,), T.float32)
            first_denom = T.alloc_local((vector,), T.float32)
            first_accum = T.alloc_local((vector,), T.float32)
            first_max = T.alloc_local((vector,), T.float32)
            old_scale = T.alloc_local((vector,), T.float32)
            pooled_local = T.alloc_local((vector,), T.float32)
            go_local = T.alloc_local((vector,), T.float32)
            inv_go_local = T.alloc_local((vector,), T.float32)
            pg_local = T.alloc_local((vector,), T.float32)
            dape_vec = T.alloc_local((vector,), T.float32)
            grad_kv_vec = T.alloc_local((vector,), T.bfloat16)
            grad_score_vec = T.alloc_local((vector,), T.bfloat16)
            valid_row = row < nb_valid
            active_row = T.if_then_else(valid_row, row, 0)
            segment = T.alloc_var(dtype=T.int32)
            segment = 0
            for s in T.serial(num_segments):
                segment = T.if_then_else(
                    valid_row
                    & (row >= cu_seqlens_comp[s])
                    & (row < cu_seqlens_comp[s + 1]),
                    s,
                    segment,
                )
            token_start = T.cast(
                cu_seqlens[segment]
                + (active_row - cu_seqlens_comp[segment]) * ratio,
                address_index_dtype,
            )
            block = active_row - cu_seqlens_comp[segment]
            for i in T.unroll(vector):
                local_denom[i] = 0.0
                local_accum[i] = 0.0
                local_max[i] = -T.infinity(T.float32)

            if warp == 0:
                T.tma_copy(
                    ape[0:ape_tile_tokens,
                        col_cta * columns_per_cta:(col_cta + 1) * columns_per_cta],
                    ape_shared[0, :, :],
                    barrier=ape_ready[0],
                    leader_scope_threads=threads_x,
                )
                T.barrier_arrive(ape_ready[0])
            T.mbarrier_wait_parity(ape_ready[0], 0)
            T.sched_boundary()
            T.sync_threads()

            # First sweep: online softmax. The last prefetch primes tile 0
            # for the second sweep before this loop exits.
            for tile in T.unroll(total_ape_tiles, explicit=False, unroll_factor=1):
                if coff == 2 and tile == ape_tiles:
                    for i in T.vectorized(vector):
                        first_denom[i] = local_denom[i]
                        first_accum[i] = local_accum[i]
                        first_max[i] = local_max[i]
                        local_denom[i] = 0.0
                        local_accum[i] = 0.0
                        local_max[i] = -T.infinity(T.float32)
                stage = tile % 2
                half = tile // ape_tiles
                tile_in_half = tile % ape_tiles
                next_tile = (tile + 1) % total_ape_tiles
                next_half = next_tile // ape_tiles
                next_tile_in_half = next_tile % ape_tiles
                if warp == 0:
                    T.tma_copy(
                        ape[next_tile_in_half * ape_tile_tokens:
                            (next_tile_in_half + 1) * ape_tile_tokens,
                            next_half * head_dim + col_cta * columns_per_cta:
                            next_half * head_dim + (col_cta + 1) * columns_per_cta],
                        ape_shared[1 - stage, :, :],
                        barrier=ape_ready[1 - stage],
                        leader_scope_threads=threads_x,
                    )
                    T.barrier_arrive(ape_ready[1 - stage])

                if valid_row & ((coff == 1) | (block > 0) | (half == 1)):
                    for chunk in T.unroll(chunks_per_tile, explicit=False, unroll_factor=1):
                        for token_in_chunk in T.unroll(
                            chunk_tokens, unroll_factor=chunk_tokens
                        ):
                            token_in_tile = chunk * chunk_tokens + token_in_chunk
                            token_offset = tile_in_half * ape_tile_tokens + token_in_tile
                            token = token_start + token_offset - (coff - 1) * (1 - half) * ratio
                            input_col_start = col_start + half * head_dim
                            input_col_end = input_col_start + vector
                            for i in T.vectorized(vector):
                                ape_local[i] = ape_shared[
                                    stage, token_in_tile, column + i
                                ] * _LOG2E
                            score_local[0:4] = T.reinterpret(
                                T.ldg64(score[token, input_col_start:input_col_end]),
                                "bfloat16x4",
                            )
                            for i in T.vectorized(vector):
                                logits[token_in_chunk, i] = (
                                    score_local[i].astype(T.float32) * _LOG2E + ape_local[i]
                                )
                        for i in T.vectorized(vector):
                            max01 = T.max(logits[0, i], logits[1, i])
                            max23 = T.max(logits[2, i], logits[3, i])
                            chunk_max = T.max(max01, max23)
                            new_max = T.max(local_max[i], chunk_max)
                            old_scale[i] = T.exp2(local_max[i] - new_max)
                            local_max[i] = new_max
                        for i in T.vectorized(vector):
                            local_denom[i] *= old_scale[i]
                            local_accum[i] *= old_scale[i]
                        for token_in_chunk in T.unroll(
                            chunk_tokens, unroll_factor=chunk_tokens
                        ):
                            token_in_tile = chunk * chunk_tokens + token_in_chunk
                            token_offset = tile_in_half * ape_tile_tokens + token_in_tile
                            token = token_start + token_offset - (coff - 1) * (1 - half) * ratio
                            input_col_start = col_start + half * head_dim
                            input_col_end = input_col_start + vector
                            value_local[0:4] = T.reinterpret(
                                T.ldg64(kv[token, input_col_start:input_col_end]),
                                "bfloat16x4",
                            )
                            for i in T.vectorized(vector):
                                probabilities[token_in_chunk, i] = T.exp2(
                                    logits[token_in_chunk, i] - local_max[i]
                                )
                            for i in T.vectorized(vector):
                                prob = probabilities[token_in_chunk, i]
                                local_denom[i] += prob
                                local_accum[i] = (
                                    value_local[i].astype(T.float32)
                                    * prob
                                    + local_accum[i]
                                )

                T.mbarrier_wait_parity(
                    ape_ready[1 - stage], ((tile + 1) // 2) & 1
                )
                T.sched_boundary()
                T.sync_threads()

            if coff == 2 and valid_row:
                for i in T.vectorized(vector):
                    merged_max = T.max(first_max[i], local_max[i])
                    scale_first = T.exp2(first_max[i] - merged_max)
                    scale_second = T.exp2(local_max[i] - merged_max)
                    local_denom[i] = (
                        first_denom[i] * scale_first
                        + local_denom[i] * scale_second
                    )
                    local_accum[i] = (
                        first_accum[i] * scale_first
                        + local_accum[i] * scale_second
                    )
                    local_max[i] = merged_max

            if valid_row:
                go_local[0:4] = T.cast(
                    T.reinterpret(
                        T.ldg64(grad_out[row, col_start:col_end]),
                        "bfloat16x4",
                    ),
                    T.float32x4,
                )
                for i in T.vectorized(vector):
                    denom_inv = 1.0 / local_denom[i]
                    pooled_local[i] = local_accum[i] * denom_inv
                    inv_go_local[i] = denom_inv * go_local[i]

            # Second sweep: recompute probabilities and write gradients.
            for tile in T.unroll(total_ape_tiles, explicit=False, unroll_factor=1):
                all_tile = total_ape_tiles + tile
                stage = all_tile % 2
                half = tile // ape_tiles
                tile_in_half = tile % ape_tiles
                if tile < total_ape_tiles - 1:
                    next_tile = tile + 1
                    next_half = next_tile // ape_tiles
                    next_tile_in_half = next_tile % ape_tiles
                    if warp == 0:
                        T.tma_copy(
                            ape[next_tile_in_half * ape_tile_tokens:
                                (next_tile_in_half + 1) * ape_tile_tokens,
                                next_half * head_dim + col_cta * columns_per_cta:
                                next_half * head_dim + (col_cta + 1) * columns_per_cta],
                            ape_shared[1 - stage, :, :],
                            barrier=ape_ready[1 - stage],
                            leader_scope_threads=threads_x,
                        )
                        T.barrier_arrive(ape_ready[1 - stage])

                if valid_row & ((coff == 1) | (block > 0) | (half == 1)):
                    # Recompute in the same four-token chunks as pass 1.
                    for chunk in T.unroll(
                        chunks_per_tile, explicit=False, unroll_factor=1
                    ):
                        for token_in_chunk in T.unroll(
                            chunk_tokens, unroll_factor=chunk_tokens
                        ):
                            token_in_tile = chunk * chunk_tokens + token_in_chunk
                            token_offset = tile_in_half * ape_tile_tokens + token_in_tile
                            token = token_start + token_offset - (coff - 1) * (1 - half) * ratio
                            input_col_start = col_start + half * head_dim
                            input_col_end = input_col_start + vector
                            score_chunk[token_in_chunk, 0:4] = T.reinterpret(
                                T.ldg64(score[token, input_col_start:input_col_end]),
                                "bfloat16x4",
                            )
                            value_chunk[token_in_chunk, 0:4] = T.reinterpret(
                                T.ldg64(kv[token, input_col_start:input_col_end]),
                                "bfloat16x4",
                            )
                        for token_in_chunk in T.unroll(
                            chunk_tokens, unroll_factor=chunk_tokens
                        ):
                            token_in_tile = chunk * chunk_tokens + token_in_chunk
                            token_offset = tile_in_half * ape_tile_tokens + token_in_tile
                            token = token_start + token_offset - (coff - 1) * (1 - half) * ratio
                            input_col_start = col_start + half * head_dim
                            for i in T.vectorized(vector):
                                ape_local[i] = ape_shared[
                                    stage, token_in_tile, column + i
                                ] * _LOG2E
                            for i in T.vectorized(vector):
                                probability = T.exp2(
                                    (score_chunk[token_in_chunk, i].astype(T.float32) * _LOG2E
                                     + ape_local[i])
                                    - local_max[i]
                                )
                                pg_local[i] = probability * inv_go_local[i]
                                dape_vec[i] = pg_local[i] * (
                                    value_chunk[token_in_chunk, i].astype(T.float32)
                                    - pooled_local[i]
                                )
                                grad_kv_vec[i] = pg_local[i].astype(T.bfloat16)
                                grad_score_vec[i] = dape_vec[i].astype(T.bfloat16)
                            for i in T.vectorized(vector):
                                grad_kv[token, input_col_start + i] = (
                                    grad_kv_vec[i]
                                )
                                grad_score[token, input_col_start + i] = (
                                    grad_score_vec[i]
                                )
                            T.evaluate(
                                T.call_extern(
                                    "void",
                                    "__builtin_assume",
                                    T.reinterpret(
                                        T.address_of(
                                            grad_ape[
                                                row % dape_shards * ratio + token_offset,
                                                input_col_start,
                                            ]
                                        ),
                                        dtype="uint64",
                                    ) % 16 == 0,
                                )
                            )
                            T.atomic_addx4(
                                grad_ape[row % dape_shards * ratio + token_offset, input_col_start],
                                dape_vec[0],
                            )

                if tile < total_ape_tiles - 1:
                    T.mbarrier_wait_parity(
                        ape_ready[1 - stage], ((all_tile + 1) // 2) & 1
                    )
                    T.sched_boundary()
                    T.sync_threads()

    return main


def _build_backward_program_r128_fused_tilewarp(
    ratio: int,
    head_dim: int,
    coff: int,
    rows_per_cta: int = 1,
    warps_per_cta: int = 4,
    ape_tile_tokens: int = 16,
    chunk_tokens: int = 4,
    dape_shards: int = 16,
):
    """Build the tilewarp variant of the fused R128 backward.

    A CTA shares one APE tile across several warps.  Each warp owns a
    disjoint token slice, computes a partial online softmax, and contributes
    its partial max/denominator/value sum to shared memory.  Warp 0 merges
    those partials; all warps then recompute their own token slice and write
    dKV/dScore while atomically accumulating dAPE.  ``rows_per_cta`` lets the
    same tile be reused by multiple rows, matching the forward tilewarp
    layout.
    """
    if ratio != 128 or coff not in (1, 2):
        raise ValueError("tilewarp backward requires ratio=128, coff=1/2")
    if head_dim not in (128, 512):
        raise ValueError("tilewarp backward supports head_dim 128 or 512")
    if rows_per_cta not in (1, 2, 4):
        raise ValueError("rows_per_cta must be 1, 2, or 4")
    if warps_per_cta not in (1, 2, 4, 8):
        raise ValueError("tilewarp backward requires 1, 2, 4, or 8 warps per CTA")
    if ape_tile_tokens not in (8, 16, 32, 64, 128):
        raise ValueError("ape_tile_tokens must be 8, 16, 32, 64, or 128")
    if chunk_tokens not in (4, 8):
        raise ValueError("chunk_tokens must be 4 or 8")
    if ape_tile_tokens % warps_per_cta:
        raise ValueError("ape_tile_tokens must be divisible by warps_per_cta")
    tokens_per_warp = ape_tile_tokens // warps_per_cta
    if tokens_per_warp % chunk_tokens:
        raise ValueError("tokens_per_warp must be divisible by chunk_tokens")
    if dape_shards < 1 or dape_shards & (dape_shards - 1):
        raise ValueError("dape_shards must be a positive power of two")

    vector = 4
    threads_x = 32
    columns_per_cta = threads_x * vector
    if head_dim % columns_per_cta:
        raise ValueError("head_dim must be divisible by columns_per_cta")
    block_threads = warps_per_cta * threads_x
    column_ctas = head_dim // columns_per_cta
    ape_tiles = ratio // ape_tile_tokens
    total_ape_tiles = coff * ape_tiles
    width = coff * head_dim
    address_index_dtype = T.int64 if width == 1024 else T.int32
    chunks_per_tile = tokens_per_warp // chunk_tokens
    total_tokens = T.dynamic("total_tokens")
    total_comp = T.dynamic("total_comp")
    num_offsets = T.dynamic("num_offsets")

    @T.prim_func
    def main(
        kv: T.Tensor((total_tokens, width), T.bfloat16),
        score: T.Tensor((total_tokens, width), T.bfloat16),
        ape: T.Tensor((ratio, width), T.float32),
        cu_seqlens: T.Tensor((num_offsets,), T.int32),
        cu_seqlens_comp: T.Tensor((num_offsets,), T.int32),
        grad_out: T.Tensor((total_comp, head_dim), T.bfloat16),
        grad_kv: T.Tensor((total_tokens, width), T.bfloat16),
        grad_score: T.Tensor((total_tokens, width), T.bfloat16),
        grad_ape: T.Tensor((ratio * dape_shards, width), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(total_comp, rows_per_cta),
            column_ctas,
            threads=block_threads,
        ) as (row_cta, col_cta):
            tx = T.get_thread_binding()
            warp = tx // threads_x
            lane = tx % threads_x
            column = lane * vector
            col_start = col_cta * columns_per_cta + column
            col_end = col_start + vector
            num_segments = num_offsets - 1
            nb_valid = cu_seqlens_comp[num_segments]

            ape_shared = T.alloc_shared(
                (2, ape_tile_tokens, columns_per_cta), T.float32
            )
            T.annotate_layout(
                {ape_shared: T.Layout(ape_shared.shape, lambda *idx: idx)}
            )
            partial_max = T.alloc_shared(
                (rows_per_cta, warps_per_cta, columns_per_cta), T.float32
            )
            partial_denom = T.alloc_shared(
                (rows_per_cta, warps_per_cta, columns_per_cta), T.float32
            )
            partial_accum = T.alloc_shared(
                (rows_per_cta, warps_per_cta, columns_per_cta), T.float32
            )
            first_max = T.alloc_shared(
                (rows_per_cta, warps_per_cta, columns_per_cta), T.float32
            )
            first_denom = T.alloc_shared(
                (rows_per_cta, warps_per_cta, columns_per_cta), T.float32
            )
            first_accum = T.alloc_shared(
                (rows_per_cta, warps_per_cta, columns_per_cta), T.float32
            )
            merged_max = T.alloc_shared(
                (rows_per_cta, columns_per_cta), T.float32
            )
            merged_denom = T.alloc_shared(
                (rows_per_cta, columns_per_cta), T.float32
            )
            pooled = T.alloc_shared(
                (rows_per_cta, columns_per_cta), T.float32
            )
            inv_go = T.alloc_shared(
                (rows_per_cta, columns_per_cta), T.float32
            )
            T.annotate_layout({
                partial_max: T.Layout(partial_max.shape, lambda *idx: idx),
                partial_denom: T.Layout(partial_denom.shape, lambda *idx: idx),
                partial_accum: T.Layout(partial_accum.shape, lambda *idx: idx),
                first_max: T.Layout(first_max.shape, lambda *idx: idx),
                first_denom: T.Layout(first_denom.shape, lambda *idx: idx),
                first_accum: T.Layout(first_accum.shape, lambda *idx: idx),
                merged_max: T.Layout(merged_max.shape, lambda *idx: idx),
                merged_denom: T.Layout(merged_denom.shape, lambda *idx: idx),
                pooled: T.Layout(pooled.shape, lambda *idx: idx),
                inv_go: T.Layout(inv_go.shape, lambda *idx: idx),
            })

            score_chunk = T.alloc_local((chunk_tokens, vector), T.bfloat16)
            value_chunk = T.alloc_local((chunk_tokens, vector), T.bfloat16)
            value_local = T.alloc_local((vector,), T.bfloat16)
            ape_local = T.alloc_local((vector,), T.float32)
            logits = T.alloc_local((chunk_tokens, vector), T.float32)
            probabilities = T.alloc_local((chunk_tokens, vector), T.float32)
            local_denom = T.alloc_local((rows_per_cta, vector), T.float32)
            local_accum = T.alloc_local((rows_per_cta, vector), T.float32)
            local_max = T.alloc_local((rows_per_cta, vector), T.float32)
            token_start = T.alloc_local((rows_per_cta,), address_index_dtype)
            block_local = T.alloc_local((rows_per_cta,), T.int32)
            valid_local = T.alloc_local((rows_per_cta,), T.bool)
            go_local = T.alloc_local((vector,), T.float32)
            dape_vec = T.alloc_local((vector,), T.float32)
            grad_kv_vec = T.alloc_local((vector,), T.bfloat16)
            grad_score_vec = T.alloc_local((vector,), T.bfloat16)

            for row_iter in T.unroll(rows_per_cta):
                row = row_cta * rows_per_cta + row_iter
                valid_row = row < nb_valid
                active_row = T.if_then_else(valid_row, row, 0)
                segment = T.alloc_var(dtype=T.int32)
                segment = 0
                for s in T.serial(num_segments):
                    segment = T.if_then_else(
                        valid_row
                        & (row >= cu_seqlens_comp[s])
                        & (row < cu_seqlens_comp[s + 1]),
                        s,
                        segment,
                    )
                token_start[row_iter] = T.cast(
                    cu_seqlens[segment]
                    + (active_row - cu_seqlens_comp[segment]) * ratio,
                    address_index_dtype,
                )
                block_local[row_iter] = (
                    active_row - cu_seqlens_comp[segment]
                )
                valid_local[row_iter] = valid_row
                for i in T.vectorized(vector):
                    local_denom[row_iter, i] = 0.0
                    local_accum[row_iter, i] = 0.0
                    local_max[row_iter, i] = -T.infinity(T.float32)

            ape_ready = T.alloc_barrier([threads_x, threads_x])
            if warp == 0:
                T.tma_copy(
                    ape[0:ape_tile_tokens,
                        col_cta * columns_per_cta:(col_cta + 1) * columns_per_cta],
                    ape_shared[0, :, :],
                    barrier=ape_ready[0],
                    leader_scope_threads=threads_x,
                )
                T.barrier_arrive(ape_ready[0])
            T.mbarrier_wait_parity(ape_ready[0], 0)
            T.sched_boundary()
            T.sync_threads()

            # First sweep: every warp owns a disjoint token slice and builds
            # a partial online softmax for every row in this CTA.
            for tile in T.unroll(
                total_ape_tiles, explicit=False, unroll_factor=1
            ):
                if coff == 2 and tile == ape_tiles:
                    for row_iter in T.unroll(rows_per_cta):
                        for i in T.vectorized(vector):
                            first_max[row_iter, warp, column + i] = local_max[
                                row_iter, i
                            ]
                            first_denom[row_iter, warp, column + i] = local_denom[
                                row_iter, i
                            ]
                            first_accum[row_iter, warp, column + i] = local_accum[
                                row_iter, i
                            ]
                            local_denom[row_iter, i] = 0.0
                            local_accum[row_iter, i] = 0.0
                            local_max[row_iter, i] = -T.infinity(T.float32)

                stage = tile % 2
                half = tile // ape_tiles
                tile_in_half = tile % ape_tiles
                next_tile = (tile + 1) % total_ape_tiles
                next_half = next_tile // ape_tiles
                next_tile_in_half = next_tile % ape_tiles
                if warp == 0:
                    T.tma_copy(
                        ape[next_tile_in_half * ape_tile_tokens:
                            (next_tile_in_half + 1) * ape_tile_tokens,
                            next_half * head_dim + col_cta * columns_per_cta:
                            next_half * head_dim + (col_cta + 1) * columns_per_cta],
                        ape_shared[1 - stage, :, :],
                        barrier=ape_ready[1 - stage],
                        leader_scope_threads=threads_x,
                    )
                    T.barrier_arrive(ape_ready[1 - stage])

                for row_iter in T.unroll(rows_per_cta):
                    if valid_local[row_iter] & (
                        (coff == 1) | (block_local[row_iter] > 0) | (half == 1)
                    ):
                        for chunk in T.unroll(
                            chunks_per_tile, explicit=False, unroll_factor=1
                        ):
                            for token_in_chunk in T.unroll(
                                chunk_tokens, unroll_factor=chunk_tokens
                            ):
                                token_in_tile = (
                                    warp * tokens_per_warp
                                    + chunk * chunk_tokens
                                    + token_in_chunk
                                )
                                token_offset = (
                                    tile_in_half * ape_tile_tokens + token_in_tile
                                )
                                token = (
                                    token_start[row_iter]
                                    + token_offset
                                    - (coff - 1) * (1 - half) * ratio
                                )
                                input_col_start = col_start + half * head_dim
                                input_col_end = input_col_start + vector
                                for i in T.vectorized(vector):
                                    ape_local[i] = ape_shared[
                                        stage, token_in_tile, column + i
                                    ] * _LOG2E
                                score_chunk[token_in_chunk, 0:4] = T.reinterpret(
                                    T.ldg64(score[token, input_col_start:input_col_end]),
                                    "bfloat16x4",
                                )
                                for i in T.vectorized(vector):
                                    logits[token_in_chunk, i] = (
                                        score_chunk[token_in_chunk, i].astype(T.float32)
                                        * _LOG2E
                                        + ape_local[i]
                                    )
                            for i in T.vectorized(vector):
                                max01 = T.max(logits[0, i], logits[1, i])
                                max23 = T.max(logits[2, i], logits[3, i])
                                chunk_max = T.max(max01, max23)
                                new_max = T.max(local_max[row_iter, i], chunk_max)
                                old_scale = T.exp2(
                                    local_max[row_iter, i] - new_max
                                )
                                local_max[row_iter, i] = new_max
                                local_denom[row_iter, i] *= old_scale
                                local_accum[row_iter, i] *= old_scale
                            for token_in_chunk in T.unroll(
                                chunk_tokens, unroll_factor=chunk_tokens
                            ):
                                token_in_tile = (
                                    warp * tokens_per_warp
                                    + chunk * chunk_tokens
                                    + token_in_chunk
                                )
                                token_offset = (
                                    tile_in_half * ape_tile_tokens + token_in_tile
                                )
                                token = (
                                    token_start[row_iter]
                                    + token_offset
                                    - (coff - 1) * (1 - half) * ratio
                                )
                                input_col_start = col_start + half * head_dim
                                input_col_end = input_col_start + vector
                                value_local[0:4] = T.reinterpret(
                                    T.ldg64(kv[token, input_col_start:input_col_end]),
                                    "bfloat16x4",
                                )
                                for i in T.vectorized(vector):
                                    probabilities[token_in_chunk, i] = T.exp2(
                                        logits[token_in_chunk, i]
                                        - local_max[row_iter, i]
                                    )
                                    local_denom[row_iter, i] += probabilities[
                                        token_in_chunk, i
                                    ]
                                    local_accum[row_iter, i] += (
                                        value_local[i].astype(T.float32)
                                        * probabilities[token_in_chunk, i]
                                    )

                T.mbarrier_wait_parity(
                    ape_ready[1 - stage], ((tile + 1) // 2) & 1
                )
                T.sched_boundary()
                T.sync_threads()

            for row_iter in T.unroll(rows_per_cta):
                for i in T.vectorized(vector):
                    partial_max[row_iter, warp, column + i] = local_max[
                        row_iter, i
                    ]
                    partial_denom[row_iter, warp, column + i] = local_denom[
                        row_iter, i
                    ]
                    partial_accum[row_iter, warp, column + i] = local_accum[
                        row_iter, i
                    ]
            T.sync_threads()

            # Warp 0 merges all token slices (and both C2 halves), then makes
            # the row-wise denominator/output factors visible to every warp.
            if warp == 0:
                for row_iter in T.unroll(rows_per_cta):
                    for i in T.vectorized(vector):
                        merged_max[row_iter, column + i] = -T.infinity(T.float32)
                        merged_denom[row_iter, column + i] = 0.0
                        pooled[row_iter, column + i] = 0.0
                        inv_go[row_iter, column + i] = 0.0
                    if coff == 2:
                        for w in T.unroll(warps_per_cta):
                            for i in T.vectorized(vector):
                                merged_max[row_iter, column + i] = T.max(
                                    merged_max[row_iter, column + i],
                                    first_max[row_iter, w, column + i],
                                )
                    for w in T.unroll(warps_per_cta):
                        for i in T.vectorized(vector):
                            merged_max[row_iter, column + i] = T.max(
                                merged_max[row_iter, column + i],
                                partial_max[row_iter, w, column + i],
                            )
                    if coff == 2:
                        for w in T.unroll(warps_per_cta):
                            for i in T.vectorized(vector):
                                scale = T.exp2(
                                    first_max[row_iter, w, column + i]
                                    - merged_max[row_iter, column + i]
                                )
                                merged_denom[row_iter, column + i] += (
                                    first_denom[row_iter, w, column + i] * scale
                                )
                                pooled[row_iter, column + i] += (
                                    first_accum[row_iter, w, column + i] * scale
                                )
                    for w in T.unroll(warps_per_cta):
                        for i in T.vectorized(vector):
                            scale = T.exp2(
                                partial_max[row_iter, w, column + i]
                                - merged_max[row_iter, column + i]
                            )
                            merged_denom[row_iter, column + i] += (
                                partial_denom[row_iter, w, column + i] * scale
                            )
                            pooled[row_iter, column + i] += (
                                partial_accum[row_iter, w, column + i] * scale
                            )
                    row = row_cta * rows_per_cta + row_iter
                    if row < nb_valid:
                        go_local[0:4] = T.cast(
                            T.reinterpret(
                                T.ldg64(grad_out[row, col_start:col_end]),
                                "bfloat16x4",
                            ),
                            T.float32x4,
                        )
                        for i in T.vectorized(vector):
                            denom_inv = 1.0 / merged_denom[row_iter, column + i]
                            pooled[row_iter, column + i] *= denom_inv
                            inv_go[row_iter, column + i] = (
                                denom_inv * go_local[i]
                            )
            T.sync_threads()

            # Second sweep: recompute the same token slices and emit gradients.
            for tile in T.unroll(
                total_ape_tiles, explicit=False, unroll_factor=1
            ):
                all_tile = total_ape_tiles + tile
                stage = all_tile % 2
                half = tile // ape_tiles
                tile_in_half = tile % ape_tiles
                if tile < total_ape_tiles - 1:
                    next_tile = tile + 1
                    next_half = next_tile // ape_tiles
                    next_tile_in_half = next_tile % ape_tiles
                    if warp == 0:
                        T.tma_copy(
                            ape[next_tile_in_half * ape_tile_tokens:
                                (next_tile_in_half + 1) * ape_tile_tokens,
                                next_half * head_dim + col_cta * columns_per_cta:
                                next_half * head_dim + (col_cta + 1) * columns_per_cta],
                            ape_shared[1 - stage, :, :],
                            barrier=ape_ready[1 - stage],
                            leader_scope_threads=threads_x,
                        )
                        T.barrier_arrive(ape_ready[1 - stage])

                for row_iter in T.unroll(rows_per_cta):
                    if valid_local[row_iter] & (
                        (coff == 1) | (block_local[row_iter] > 0) | (half == 1)
                    ):
                        for chunk in T.unroll(
                            chunks_per_tile, explicit=False, unroll_factor=1
                        ):
                            for token_in_chunk in T.unroll(
                                chunk_tokens, unroll_factor=chunk_tokens
                            ):
                                token_in_tile = (
                                    warp * tokens_per_warp
                                    + chunk * chunk_tokens
                                    + token_in_chunk
                                )
                                token_offset = (
                                    tile_in_half * ape_tile_tokens + token_in_tile
                                )
                                token = (
                                    token_start[row_iter]
                                    + token_offset
                                    - (coff - 1) * (1 - half) * ratio
                                )
                                input_col_start = col_start + half * head_dim
                                input_col_end = input_col_start + vector
                                score_chunk[token_in_chunk, 0:4] = T.reinterpret(
                                    T.ldg64(score[token, input_col_start:input_col_end]),
                                    "bfloat16x4",
                                )
                                value_chunk[token_in_chunk, 0:4] = T.reinterpret(
                                    T.ldg64(kv[token, input_col_start:input_col_end]),
                                    "bfloat16x4",
                                )
                            for token_in_chunk in T.unroll(
                                chunk_tokens, unroll_factor=chunk_tokens
                            ):
                                token_in_tile = (
                                    warp * tokens_per_warp
                                    + chunk * chunk_tokens
                                    + token_in_chunk
                                )
                                token_offset = (
                                    tile_in_half * ape_tile_tokens + token_in_tile
                                )
                                token = (
                                    token_start[row_iter]
                                    + token_offset
                                    - (coff - 1) * (1 - half) * ratio
                                )
                                input_col_start = col_start + half * head_dim
                                for i in T.vectorized(vector):
                                    ape_local[i] = ape_shared[
                                        stage, token_in_tile, column + i
                                    ] * _LOG2E
                                for i in T.vectorized(vector):
                                    probability = T.exp2(
                                        (
                                            score_chunk[token_in_chunk, i].astype(
                                                T.float32
                                            )
                                            * _LOG2E
                                            + ape_local[i]
                                        )
                                        - merged_max[row_iter, column + i]
                                    )
                                    dape_vec[i] = (
                                        probability * inv_go[row_iter, column + i]
                                        * (
                                            value_chunk[token_in_chunk, i].astype(
                                                T.float32
                                            )
                                            - pooled[row_iter, column + i]
                                        )
                                    )
                                    grad_kv_vec[i] = (
                                        probability * inv_go[row_iter, column + i]
                                    ).astype(T.bfloat16)
                                    grad_score_vec[i] = dape_vec[i].astype(T.bfloat16)
                                for i in T.vectorized(vector):
                                    grad_kv[token, input_col_start + i] = grad_kv_vec[i]
                                    grad_score[token, input_col_start + i] = grad_score_vec[i]
                                T.evaluate(
                                    T.call_extern(
                                        "void",
                                        "__builtin_assume",
                                        T.reinterpret(
                                            T.address_of(
                                            grad_ape[
                                                    ((row_cta * rows_per_cta + row_iter)
                                                     % dape_shards)
                                                    * ratio
                                                    + token_offset,
                                                    input_col_start,
                                                ]
                                            ),
                                            dtype="uint64",
                                        )
                                        % 16
                                        == 0,
                                    )
                                )
                                T.atomic_addx4(
                                    grad_ape[
                                        ((row_cta * rows_per_cta + row_iter)
                                         % dape_shards)
                                        * ratio
                                        + token_offset,
                                        input_col_start,
                                    ],
                                    dape_vec[0],
                                )

                if tile < total_ape_tiles - 1:
                    T.mbarrier_wait_parity(
                        ape_ready[1 - stage], ((all_tile + 1) // 2) & 1
                    )
                    T.sched_boundary()
                    T.sync_threads()

    return main


def _build_backward_ape_reduce_shards_r128(head_dim: int, shards: int):
    ratio = 128

    @T.prim_func
    def main(
        partial: T.Tensor((ratio * shards, head_dim), T.float32),
        grad_ape: T.Tensor((ratio, head_dim), T.float32),
    ):
        with T.Kernel(ratio, T.ceildiv(head_dim, 128), threads=128) as (
            token, col_cta
        ):
            col = col_cta * 128 + T.get_thread_binding()
            total = T.alloc_var(dtype=T.float32)
            total = 0.0
            for shard in T.unroll(shards):
                total = total + partial[shard * ratio + token, col]
            grad_ape[token, col] = grad_ape[token, col] + total

    return main


# MP32 rowgroup candidates. Entries are
# (maximum output rows, APE tile tokens, warps per CTA, dAPE shards).
_ROWGROUP_BWD_ROW_BUCKETS = {
    (1, 128): ((256, 64, 4, 16), (2048, 32, 4, 16),
               (4096, 16, 8, 16), (None, 32, 8, 16)),
    (1, 512): ((256, 64, 4, 16), (768, 16, 8, 16),
               (None, 32, 8, 16)),
    (2, 128): ((768, 64, 4, 16), (2048, 32, 8, 16),
               (None, 32, 4, 16)),
    (2, 512): ((256, 64, 4, 16), (768, 32, 8, 16),
               (2048, 32, 4, 16), (4096, 16, 4, 16),
               (None, 16, 4, 8)),
}

# Small-row MP32 tilewarp buckets. Entries are (maximum output rows,
# rows per CTA, APE tile, warps, chunk, dAPE shards).
_TILEWARP_BWD_ROW_BUCKETS = {
    (1, 128): ((512, 1, 32, 4, 4, 16), (768, 1, 16, 4, 4, 16)),
    (1, 512): ((128, 1, 32, 4, 4, 16), (256, 1, 16, 4, 4, 16)),
    (2, 128): ((128, 1, 32, 4, 4, 16), (512, 1, 16, 4, 4, 16)),
    (2, 512): ((256, 1, 16, 4, 4, 16),),
}


def fused_backward_schedule_r128(ratio, d, coff, nb_total):
    """Return ``(builder, rows_per_cta, tile, warps, chunk, shards)``."""
    if int(ratio) != 128 or (int(coff), int(d)) not in _ROWGROUP_BWD_ROW_BUCKETS:
        raise ValueError("fused R128 backward supports ratio=128, coff=1/2, d=128/512")
    if nb_total is None:
        raise ValueError("fused R128 backward needs the output row capacity")
    shape = int(coff), int(d)
    for tilewarp in _TILEWARP_BWD_ROW_BUCKETS[shape]:
        if int(nb_total) <= tilewarp[0]:
            return ("tilewarp", *tilewarp[1:])
    for limit, tile, warps, shards in _ROWGROUP_BWD_ROW_BUCKETS[shape]:
        if limit is None or int(nb_total) <= limit:
            return "rowgroup", warps, tile, warps, 4, shards
    raise AssertionError("unreachable R128 backward schedule")


def _compile_bwd_r128_fused(key, ratio, d, coff, schedule):
    """Compile the fused main kernel and its dAPE shard reduction once."""
    with _COMPILE_LOCK:
        cached = _COMPILED.get(key)
        if cached is not None:
            return cached
        query = getattr(torch.musa, "is_current_stream_capturing", None)
        if query is not None and query():
            raise RuntimeError(
                f"CSA compressor (r128): first call for config {key} happened under "
                "MUSA graph capture (JIT compilation is not capture-safe); compile() "
                "or run one eager step for this configuration before capturing."
            )
        kind, rows_per_cta, tile, warps, chunk, shards = schedule
        if kind == "rowgroup":
            program = _build_backward_program_r128_fused_ablation(
                int(ratio), int(d), int(coff),
                warps_per_cta=warps,
                ape_tile_tokens=tile,
                dape_shards=shards,
            )
        elif kind == "tilewarp":
            program = _build_backward_program_r128_fused_tilewarp(
                int(ratio), int(d), int(coff),
                rows_per_cta=rows_per_cta,
                warps_per_cta=warps,
                ape_tile_tokens=tile,
                chunk_tokens=chunk,
                dape_shards=shards,
            )
        else:
            raise ValueError(f"unknown R128 backward builder: {kind}")
        main = tilelang.compile(
            program,
            target="musa",
            pass_configs=_PASS_CONFIGS,
            compile_flags=_COMPILE_FLAGS,
        )
        reduce = tilelang.compile(
            _build_backward_ape_reduce_shards_r128(int(coff) * int(d), shards),
            target="musa",
            pass_configs=_PASS_CONFIGS,
            compile_flags=_COMPILE_FLAGS,
        )
        _COMPILED[key] = (main, reduce)
        return main, reduce


def precompile_bwd_r128(ratio, d, coff, device, nb_total=None):
    """Compile every schedule or the one selected by the row capacity."""
    if device is None:
        raise RuntimeError("CSA TileLang compilation needs a concrete MUSA device")
    if int(ratio) != 128 or (int(coff), int(d)) not in _ROWGROUP_BWD_ROW_BUCKETS:
        raise ValueError("fused R128 backward supports ratio=128, coff=1/2, d=128/512")
    schedules = (
        {fused_backward_schedule_r128(ratio, d, coff, nb_total)}
        if nb_total is not None else
        {("rowgroup", warps, tile, warps, 4, shards)
         for _, tile, warps, shards
         in _ROWGROUP_BWD_ROW_BUCKETS[int(coff), int(d)]}
        | {("tilewarp", *bucket[1:]) for bucket
           in _TILEWARP_BWD_ROW_BUCKETS[int(coff), int(d)]}
    )
    with torch.musa.device(device):
        for schedule in schedules:
            key = ("r128bwd_fused", int(ratio), int(d), int(coff),
                   schedule, device.index)
            if key not in _COMPILED:
                _compile_bwd_r128_fused(key, ratio, d, coff, schedule)


def run_bwd_r128(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out,
    grad_kv, grad_score, grad_ape, nb_total, ratio, d, coff,
    stream_handle=None,
):
    """Launch the fused R128 backward and reduce its dAPE shards."""
    del stream_handle
    if int(nb_total) == 0:
        return
    dev = kv.device.index
    schedule = fused_backward_schedule_r128(ratio, d, coff, int(nb_total))
    key = ("r128bwd_fused", int(ratio), int(d), int(coff), schedule, dev)
    with torch.musa.device(dev):
        main, reduce = _compile_bwd_r128_fused(key, ratio, d, coff, schedule)
        shards = schedule[-1]
        partial = torch.zeros(
            (shards * 128, int(coff) * int(d)),
            dtype=torch.float32,
            device=kv.device,
        )
        main(
            kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out,
            grad_kv, grad_score, partial,
        )
        reduce(partial, grad_ape)
