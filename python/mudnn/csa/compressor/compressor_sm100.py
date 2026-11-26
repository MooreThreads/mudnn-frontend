# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Fused short-window CSA/HCA compressor kernels for MUSA TileLang.

This module is the TileLang counterpart of the competitor's
``compressor_sm100.py``.  It keeps the same fused dataflow and tensor contract:

``gather -> + APE -> overlap-window transform -> fp32 softmax -> gated sum -> bf16``

For ``coff == 2`` the first half of the window reads the previous block's first
projection and is invalid for the first block of every segment; the second half
reads the current block's second projection.  ``coff == 1`` uses the current
block only.  Segment tails are dropped and static-capacity rows gather from the
first segment block, matching the eager reference and the CUDA implementation.

R2 and R4 forward use the rowgroup kernel: one warp owns one compressed row
at a time, with four adjacent columns per thread.
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

# Keep the public pointer contract aligned with the CUDA/CuTe implementation.
# TileLang/MUSA does not require this value for every scalar access, but using
# one fixed contract keeps all BF16/FP32 runtime tensors interchangeable across
# the forward and backward APIs.
PTR_ALIGN_BYTES = 16
CU_ALIGN_BYTES = 4
SUPPORTED_HEAD_DIMS = (128, 512)
_LOG2E = 1.4426950408889634


_RCP4_SOURCE = """
#include <tl_templates/musa/common.h>
TL_DEVICE void csa_rcp4_ptr(float* dst, const float* src) {
  const tl_f4 x = *reinterpret_cast<const tl_f4*>(src);
  const tl_f4 one{1.0f, 1.0f, 1.0f, 1.0f};
  *reinterpret_cast<tl_f4*>(dst) = one / x;
}
"""

_COMPILED = {}
_COMPILE_LOCK = threading.Lock()


def _build_forward_program_r4_rowgroup(
    head_dim: int,
    row_groups: int,
    ape_shared: bool = False,
    single_segment: bool = False,
    warps_per_cta: int = 4,
    coff: int = 2,
    ratio: int = 4,
):
    """Each warp completes ``row_groups`` rows, one row at a time."""

    if head_dim not in (128, 512):
        raise ValueError(f"unsupported head_dim={head_dim}")
    if row_groups not in (1, 2, 4, 8, 16):
        raise ValueError(f"unsupported row_groups={row_groups}")
    if warps_per_cta not in (1, 2, 4, 8):
        raise ValueError(f"unsupported warps_per_cta={warps_per_cta}")
    if coff not in (1, 2):
        raise ValueError(f"unsupported coff={coff}")
    if ratio not in (2, 4):
        raise ValueError(f"unsupported ratio={ratio}")
    vector = 4
    chunk_tokens = ratio
    threads_x = 32
    row_tile = warps_per_cta
    rows_per_cta = row_tile * row_groups
    column_ctas = head_dim // (threads_x * vector)
    total_tokens = T.dynamic("total_tokens")
    total_comp = T.dynamic("total_comp")
    num_offsets = T.dynamic("num_offsets")

    @T.prim_func
    def main(
        kv: T.Tensor((total_tokens, coff * head_dim), T.bfloat16),
        score: T.Tensor((total_tokens, coff * head_dim), T.bfloat16),
        ape: T.Tensor((ratio, coff * head_dim), T.float32),
        cu_seqlens: T.Tensor((num_offsets,), T.int32),
        cu_seqlens_comp: T.Tensor((num_offsets,), T.int32),
        out: T.Tensor((total_comp, head_dim), T.bfloat16),
    ):
        with T.Kernel(
            T.ceildiv(total_comp, rows_per_cta),
            column_ctas,
            threads=row_tile * threads_x,
        ) as (row_cta, col_cta):
            T.import_source(_RCP4_SOURCE)
            tx = T.get_thread_binding()
            warp = tx // threads_x
            lane = tx % threads_x
            column_base = (col_cta * threads_x + lane) * vector
            column_end = column_base + vector

            if ape_shared:
                ape_tile = T.alloc_shared((coff, ratio, threads_x * vector), T.float32)
                ape_load = T.alloc_local((vector,), T.float32)
                for load_round in T.serial(T.ceildiv(coff * ratio, row_tile)):
                    load_item = tx + load_round * row_tile * threads_x
                    if load_item < coff * ratio * threads_x:
                        load_slot = load_item // threads_x
                        load_lane = load_item % threads_x
                        load_half = load_slot // ratio
                        load_k = load_slot % ratio
                        load_col = (
                            load_half * head_dim
                            + col_cta * threads_x * vector
                            + load_lane * vector
                        )
                        ape_load[0:4] = T.reinterpret(
                            T.ldg128(ape[load_k, load_col : load_col + 4]), T.float32x4
                        )
                        for i in T.vectorized(vector):
                            ape_tile[load_half, load_k, load_lane * vector + i] = (
                                ape_load[i] * _LOG2E
                            )
                T.sync_threads()

            score_local = T.alloc_local((vector,), T.float32)
            ape_local = T.alloc_local((vector,), T.float32)
            old_scale = T.alloc_local((vector,), T.float32)
            max_score = T.alloc_local((vector,), T.float32)
            denom = T.alloc_local((vector,), T.float32)
            accum = T.alloc_local((vector,), T.float32)

            logits_pair = T.alloc_local((chunk_tokens, vector), T.float32)
            values_pair = T.alloc_local((chunk_tokens, vector), T.float32)
            probs_pair = T.alloc_local((chunk_tokens, vector), T.float32)

            if not single_segment:
                num_segments = num_offsets - 1
                nb_valid = cu_seqlens_comp[num_segments]
            for row_group in T.serial(row_groups):
                row = row_cta * rows_per_cta + row_group * row_tile + warp
                valid_row = row < total_comp
                if single_segment:
                    active_row = T.if_then_else(
                        valid_row & (row < cu_seqlens_comp[1]), row, 0
                    )
                    block = active_row - cu_seqlens_comp[0]
                    tok0 = cu_seqlens[0] + block * ratio
                else:
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
                    tok0 = cu_seqlens[segment] + block * ratio

                for lane_id in T.unroll(vector):
                    max_score[lane_id] = -T.infinity(T.float32)
                    denom[lane_id] = 0.0
                    accum[lane_id] = 0.0

                if valid_row:
                    for half in T.serial(coff):
                        if coff == 1:
                            token_start = tok0
                            half_valid = True
                        else:
                            previous_half = half == 0
                            token_start = T.if_then_else(
                                previous_half, tok0 - ratio, tok0
                            )
                            half_valid = T.if_then_else(
                                previous_half, block > 0, True
                            )
                        half_column = half * head_dim + column_base
                        half_column_end = half_column + vector
                        if half_valid:
                            for chunk in T.serial(ratio // chunk_tokens):
                                for token_in_chunk in T.unroll(chunk_tokens, unroll_factor=chunk_tokens):
                                    k = chunk * chunk_tokens + token_in_chunk
                                    token = token_start + k
                                    score_local[0:4] = T.cast(T.reinterpret(
                                        T.ldg64(score[token, half_column:half_column_end]),
                                        "bfloat16x4"), T.float32x4)
                                    values_pair[token_in_chunk, 0:4] = T.cast(T.reinterpret(
                                        T.ldg64(kv[token, half_column:half_column_end]),
                                        "bfloat16x4"), T.float32x4)
                                    if ape_shared:
                                        ape_local[0:vector] = ape_tile[
                                            half, k, lane * vector : (lane + 1) * vector
                                        ]
                                    else:
                                        ape_local[0:4] = T.reinterpret(
                                            T.ldg128(ape[k, half_column:half_column_end]), T.float32x4)
                                    for lane_id in T.vectorized(vector):
                                        if ape_shared:
                                            logits_pair[token_in_chunk, lane_id] = (
                                                score_local[lane_id] * _LOG2E
                                                + ape_local[lane_id]
                                            )
                                        else:
                                            logits_pair[token_in_chunk, lane_id] = (
                                                score_local[lane_id]
                                                + ape_local[lane_id]
                                            ) * _LOG2E

                                for lane_id in T.vectorized(vector):
                                    max01 = T.max(logits_pair[0, lane_id], logits_pair[1, lane_id])
                                    if chunk_tokens == 4:
                                        max23 = T.max(logits_pair[2, lane_id], logits_pair[3, lane_id])
                                        new_max = T.max(max_score[lane_id], T.max(max01, max23))
                                    else:
                                        new_max = T.max(max_score[lane_id], max01)
                                    old_scale[lane_id] = T.exp2(
                                        max_score[lane_id] - new_max
                                    )
                                    max_score[lane_id] = new_max
                                for lane_id in T.unroll(vector):
                                    denom[lane_id] *= old_scale[lane_id]
                                    accum[lane_id] *= old_scale[lane_id]
                                for token_in_chunk in T.unroll(chunk_tokens, unroll_factor=chunk_tokens):
                                    for lane_id in T.vectorized(vector):
                                        probs_pair[token_in_chunk, lane_id] = T.exp2(
                                            logits_pair[token_in_chunk, lane_id]
                                            - max_score[lane_id]
                                        )
                                    for lane_id in T.unroll(vector):
                                        prob = probs_pair[token_in_chunk, lane_id]
                                        denom[lane_id] += prob
                                        accum[lane_id] = (
                                            values_pair[token_in_chunk, lane_id] * prob
                                            + accum[lane_id]
                                        )


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


def _forward_strategy(ratio, d, coff, nb_total, num_offsets=None):
    """Choose the rowgroup layout using ratio, capacity and segment count."""

    if ratio not in (2, 4) or d not in SUPPORTED_HEAD_DIMS or coff not in (1, 2):
        raise ValueError(f"unsupported forward configuration: {ratio=}, {d=}, {coff=}")
    rows = 0 if nb_total is None else int(nb_total)
    single_segment = num_offsets == 2
    if ratio == 2:
        if single_segment:
            if rows < 2048:
                return "rowgroup_f1"
            if d == 128:
                return "rowgroup_x4" if rows >= 32768 else "rowgroup_x1"
            if coff == 1 or rows >= 8192:
                return "rowgroup_x4"
            return "rowgroup_f1"
        if d == 512 and rows >= 8192:
            return "rowgroup_y4"
        if d == 128 and coff == 2 and rows >= 32768:
            return "rowgroup_y4"
        return "rowgroup_g1" if rows < 2048 else "rowgroup_y1"

    # Preserve the other measured R4 layouts.
    if coff == 1:
        if single_segment:
            if d == 128:
                return "rowgroup_f1" if rows < 2048 else "rowgroup_x1"
            return "rowgroup_f1" if rows < 2048 else "rowgroup_x4"
        if d == 128:
            return "rowgroup_g1" if rows < 8192 else "rowgroup_y1"
        return "rowgroup_g1" if rows < 2048 else "rowgroup_y1"
    if single_segment:
        if d == 128:
            if rows < 4096:
                return "rowgroup_f1"
            return "rowgroup_x8" if rows >= 262144 else "rowgroup_x2"
        return "rowgroup_f1" if rows < 768 else "rowgroup_x2"
    if d == 128:
        return "rowgroup_g1" if rows < 8192 else "rowgroup_y1"
    return "rowgroup_g1" if rows < 2048 else "rowgroup_y1"


def _forward_kernel_key(ratio, d, coff, device_index, strategy):
    return ("fwd", int(ratio), int(d), int(coff), device_index, strategy)


def _compile_fwd(key, ratio, d, coff, strategy):
    """JIT-compile the selected rowgroup forward kernel (capture-guarded)."""

    with _COMPILE_LOCK:
        kernel = _COMPILED.get(key)
        if kernel is None:
            query = getattr(torch.musa, "is_current_stream_capturing", None)
            if query is not None and query():
                raise RuntimeError(
                    f"CSA compressor: first call for config {key} happened under MUSA "
                    "graph capture (JIT compilation is not capture-safe); compile() or "
                    "run one eager step for this configuration before capturing."
                )
            option = strategy.removeprefix("rowgroup_")
            ape_mode, row_groups = option[0], int(option[1:])
            program = _build_forward_program_r4_rowgroup(
                int(d),
                row_groups=row_groups,
                ape_shared=ape_mode in ("x", "y"),
                single_segment=ape_mode in ("f", "x"),
                warps_per_cta=4,
                coff=int(coff),
                ratio=int(ratio),
            )
            kernel = tilelang.compile(
                program,
                target="musa",
                pass_configs=_PASS_CONFIGS,
                compile_flags=_COMPILE_FLAGS,
            )
            _COMPILED[key] = kernel
    return kernel


def precompile_fwd(ratio, d, coff, device, nb_total=None, num_offsets=None):
    """Ensure the forward kernel for this configuration is JIT-compiled."""

    if device is None:
        raise RuntimeError("CSA TileLang compilation needs a concrete MUSA device")
    strategy = _forward_strategy(ratio, d, coff, nb_total, num_offsets)
    key = _forward_kernel_key(ratio, d, coff, device.index, strategy)
    with torch.musa.device(device):
        if key not in _COMPILED:
            _compile_fwd(key, ratio, d, coff, strategy)


def run_fwd(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, nb_total, ratio, d, coff, stream_handle=None):
    """Launch the cached rowgroup forward kernel on the current MUSA stream."""

    del stream_handle
    if int(nb_total) == 0:
        return
    dev = kv.device.index
    strategy = _forward_strategy(
        ratio, d, coff, int(nb_total), int(cu_seqlens.shape[0])
    )
    key = _forward_kernel_key(ratio, d, coff, dev, strategy)
    with torch.musa.device(dev):
        kernel = _COMPILED.get(key)
        if kernel is None:
            kernel = _compile_fwd(key, ratio, d, coff, strategy)
        kernel(kv, score, ape, cu_seqlens, cu_seqlens_comp, out)


def _backward_r4_vec4_schedule(ratio, d, coff, nb_total, num_offsets):
    """Return ``(rows_per_cta, rows_per_warp, direct_atomic)`` for R2/R4.

    Direct FP32 dAPE atomics avoid the scratch/reduce launch at small capacities.
    Larger capacities use 256 dAPE shards.  The buckets come from the MP32 device
    measurements of R4 with both head dimensions; R2 reuses these buckets.
    """

    if (
        nb_total is None
        or int(ratio) not in (2, 4)
        or int(d) not in SUPPORTED_HEAD_DIMS
        or int(coff) not in (1, 2)
    ):
        return None
    rows = int(nb_total)
    if rows <= 16:
        return 4, 1, True
    if int(coff) == 1:
        if rows <= 512:
            if rows <= 128:
                return (4, 2, True) if int(d) == 128 else (2, 2, True)
            return (4, 4, True) if int(d) == 128 else (8, 4, True)
        return (8, 1, False) if int(d) == 128 and rows <= 8192 else (8, 2, False)
    if rows <= 256:
        if int(d) == 128:
            return 2, 2, True
        return (4, 2, True) if rows <= 128 else (8, 2, True)
    if rows <= 1024:
        return (4, 4, True) if int(d) == 128 else (8, 4, True)
    if rows == 2048 and num_offsets == 2:
        return (4, 1, False) if int(d) == 128 else (8, 2, False)
    if rows == 6144 and num_offsets == 4:
        return (2, 1, False) if int(d) == 128 else (8, 4, False)
    if rows < 32768:
        return (16, 4, False) if int(d) == 128 else (8, 2, False)
    if num_offsets == 2:
        return (48, 12, False) if int(d) == 128 else (64, 16, False)
    return (64, 16, False) if int(d) == 128 else (48, 12, False)


def _build_backward_program_vec4_r4(
    head_dim, coff, rows_per_cta=8, rows_per_warp=2,
    dape_shards=256, single_segment=False, direct_atomic=False, ratio=4,
):
    """Build the row-warp, four-column backward for R2/R4 and C1/C2.

    Each warp owns one compressed row and walks ``rows_per_warp`` rows
    serially. C1 uses ``ratio`` tokens; C2 merges both halves into a two-ratio
    softmax, masking the previous-block half at each segment's first row.
    dKV/dScore stay disjoint; dAPE is accumulated per ``(half, token)`` and
    published through FP32 vector atomics, directly for tiny rows or through
    dAPE shards when contention makes a reduction worthwhile.
    """
    vector, threads_x = 4, 32
    if ratio not in (2, 4):
        raise ValueError("vec4 backward requires ratio=2 or ratio=4")
    if coff not in (1, 2):
        raise ValueError("vec4 backward requires coff=1 or coff=2")
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"vec4 backward requires head_dim in {SUPPORTED_HEAD_DIMS}")
    if dape_shards < 1:
        raise ValueError("dape_shards must be positive")
    columns_per_cta = threads_x * vector
    if head_dim % columns_per_cta:
        raise ValueError(f"head_dim={head_dim} must be divisible by {columns_per_cta}")
    column_ctas = head_dim // columns_per_cta
    rows_per_cta = int(rows_per_cta)
    rows_per_warp = int(rows_per_warp)
    if rows_per_warp < 1 or rows_per_cta % rows_per_warp:
        raise ValueError(
            f"rows_per_cta={rows_per_cta} must be divisible by rows_per_warp={rows_per_warp}"
        )
    block_warps = rows_per_cta // rows_per_warp
    block_threads = block_warps * threads_x
    if rows_per_cta < 1 or block_threads > 1024:
        raise ValueError(
            f"rows_per_cta={rows_per_cta}, rows_per_warp={rows_per_warp} give invalid CTA size {block_threads}"
        )
    width = coff * head_dim
    window = coff * ratio
    total_tokens = T.dynamic("total_tokens")
    total_comp = T.dynamic("total_comp")
    num_offsets = T.dynamic("num_offsets")
    num_blocks = T.dynamic("num_blocks")

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
        grad_ape: T.Tensor((ratio, width), T.float32),
        dape_partial: T.Tensor((num_blocks, ratio, width), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(total_comp, rows_per_cta), column_ctas, threads=block_threads
        ) as (row_cta, col_cta):
            tx = T.get_thread_binding()
            row_lane = tx // threads_x
            lane_id = tx % threads_x
            row_base = row_cta * rows_per_cta
            col0 = col_cta * columns_per_cta + lane_id * vector
            col1 = col0 + vector
            num_segments = num_offsets - 1
            nb_valid = cu_seqlens_comp[num_segments]
            segment = T.alloc_var(dtype=T.int32)
            scratch = T.alloc_shared((window, 1, columns_per_cta), T.float32)
            ape_load = T.alloc_local((vector,), T.float32)
            ape_col_base = col_cta * columns_per_cta
            for load_round in T.serial(T.ceildiv(window * threads_x, block_threads)):
                load_item = tx + load_round * block_threads
                if load_item < window * threads_x:
                    load_slot = load_item // threads_x
                    load_vec = load_item % threads_x
                    load_half = load_slot // ratio
                    load_k = load_slot % ratio
                    load_col = load_half * head_dim + ape_col_base + load_vec * vector
                    ape_load[0:4] = T.reinterpret(
                        T.ldg128(ape[load_k, load_col : load_col + vector]), T.float32x4
                    )
                    for lane in T.vectorized(vector):
                        scratch[load_slot, 0, load_vec * vector + lane] = (
                            ape_load[lane] * _LOG2E
                        )
            T.sync_threads()
            go_local = T.alloc_local((vector,), T.float32)
            score_local = T.alloc_local((window * vector,), T.float32)
            kv_local = T.alloc_local((window * vector,), T.float32)
            max_local = T.alloc_local((vector,), T.float32)
            denom_local = T.alloc_local((vector,), T.float32)
            inv_denom_local = T.alloc_local((vector,), T.float32)
            dot_local = T.alloc_local((vector,), T.float32)
            out_local = T.alloc_local((vector,), T.float32)
            if direct_atomic and rows_per_warp > 1:
                dape_local = T.alloc_local((window * vector,), T.float32)
                for lane in T.unroll(window * vector):
                    dape_local[lane] = 0.0
            for row_iter in T.serial(rows_per_warp):
                row = row_base + row_iter * block_warps + row_lane
                valid_row = row < nb_valid
                if single_segment:
                    seg_tok0 = cu_seqlens[0]
                    block = row - cu_seqlens_comp[0]
                else:
                    segment = 0
                    for s in T.serial(num_segments):
                        segment = T.if_then_else(
                            valid_row
                            & (row >= cu_seqlens_comp[s])
                            & (row < cu_seqlens_comp[s + 1]),
                            s,
                            segment,
                        )
                    seg_tok0 = cu_seqlens[segment]
                    block = T.if_then_else(valid_row, row - cu_seqlens_comp[segment], 0)
                tok0 = seg_tok0 + block * ratio
                if valid_row:
                    go_local[0:4] = T.cast(
                        T.reinterpret(T.ldg64(grad_out[row, col0:col1]), "bfloat16x4"),
                        T.float32x4,
                    )
                    for lane in T.unroll(vector):
                        max_local[lane] = -T.infinity(T.float32)
                    for half in T.unroll(coff):
                        slot_base = half * ratio
                        if coff == 1:
                            half_token_base = tok0
                            half_column_base = col0
                            half_valid = valid_row
                        else:
                            half_token_base = T.if_then_else(half == 0, tok0 - ratio, tok0)
                            half_column_base = T.if_then_else(
                                half == 0, col0, head_dim + col0
                            )
                            half_valid = T.if_then_else(half == 0, block > 0, valid_row)
                        half_column_end = half_column_base + vector
                        if half_valid:
                            for k in T.unroll(ratio):
                                slot = slot_base + k
                                token = half_token_base + k
                                score_local[
                                    slot * vector : slot * vector + vector
                                ] = T.cast(
                                    T.reinterpret(
                                        T.ldg64(
                                            score[
                                                token, half_column_base:half_column_end
                                            ]
                                        ),
                                        "bfloat16x4",
                                    ),
                                    T.float32x4,
                                )
                                kv_local[
                                    slot * vector : slot * vector + vector
                                ] = T.cast(
                                    T.reinterpret(
                                        T.ldg64(
                                            kv[token, half_column_base:half_column_end]
                                        ),
                                        "bfloat16x4",
                                    ),
                                    T.float32x4,
                                )
                                for lane in T.vectorized(vector):
                                    ape_value = scratch[
                                        slot, 0, lane_id * vector + lane
                                    ]
                                    logit = T.ieee_fmaf(
                                        score_local[slot * vector + lane],
                                        _LOG2E,
                                        ape_value,
                                        "rn",
                                    )
                                    score_local[slot * vector + lane] = logit
                                    max_local[lane] = T.max(max_local[lane], logit)
                        else:
                            for k in T.unroll(ratio):
                                slot = slot_base + k
                                for lane in T.unroll(vector):
                                    score_local[slot * vector + lane] = -T.infinity(
                                        T.float32
                                    )
                                    kv_local[slot * vector + lane] = 0.0
                    for lane in T.unroll(vector):
                        denom_local[lane] = 0.0
                    for slot in T.unroll(window):
                        for lane in T.vectorized(vector):
                            probability = T.exp2(
                                score_local[slot * vector + lane] - max_local[lane]
                            )
                            score_local[slot * vector + lane] = probability
                            denom_local[lane] = denom_local[lane] + probability
                    for lane in T.unroll(vector):
                        inv_denom_local[lane] = 1.0 / denom_local[lane]
                    for lane in T.unroll(vector):
                        dot_local[lane] = 0.0
                    for half in T.unroll(coff):
                        if coff == 1:
                            half_token_base = tok0
                            half_column_base = col0
                            half_valid = valid_row
                        else:
                            half_token_base = T.if_then_else(half == 0, tok0 - ratio, tok0)
                            half_column_base = T.if_then_else(
                                half == 0, col0, head_dim + col0
                            )
                            half_valid = T.if_then_else(half == 0, block > 0, valid_row)
                        half_column_end = half_column_base + vector
                        if half_valid:
                            for k in T.unroll(ratio):
                                slot = half * ratio + k
                                token = half_token_base + k
                                for lane in T.vectorized(vector):
                                    probability = (
                                        score_local[slot * vector + lane]
                                        * inv_denom_local[lane]
                                    )
                                    score_local[slot * vector + lane] = probability
                                    out_local[lane] = T.ieee_mul(
                                        go_local[lane], probability, "rn"
                                    )
                                    weighted_dp = T.ieee_mul(
                                        out_local[lane],
                                        kv_local[slot * vector + lane].astype(
                                            T.float32
                                        ),
                                        "rn",
                                    )
                                    dot_local[lane] = dot_local[lane] + weighted_dp
                                    kv_local[slot * vector + lane] = weighted_dp
                                T.stg64(
                                    grad_kv[token, half_column_base:half_column_end],
                                    T.reinterpret(
                                        T.cast(out_local[0:4], "bfloat16x4"), "uint32x2"
                                    ),
                                )
                    for half in T.unroll(coff):
                        if coff == 1:
                            half_token_base = tok0
                            half_column_base = col0
                            half_valid = valid_row
                        else:
                            half_token_base = T.if_then_else(half == 0, tok0 - ratio, tok0)
                            half_column_base = T.if_then_else(
                                half == 0, col0, head_dim + col0
                            )
                            half_valid = T.if_then_else(half == 0, block > 0, valid_row)
                        half_column_end = half_column_base + vector
                        if half_valid:
                            for k in T.unroll(ratio):
                                slot = half * ratio + k
                                token = half_token_base + k
                                for lane in T.vectorized(vector):
                                    probability = score_local[slot * vector + lane]
                                    weighted_dp = kv_local[slot * vector + lane]
                                    ds = T.ieee_fmaf(
                                        probability, -dot_local[lane], weighted_dp, "rn"
                                    )
                                    out_local[lane] = ds
                                T.stg64(
                                    grad_score[token, half_column_base:half_column_end],
                                    T.reinterpret(
                                        T.cast(out_local[0:4], "bfloat16x4"), "uint32x2"
                                    ),
                                )
                                if direct_atomic:
                                    if rows_per_warp > 1:
                                        for lane in T.vectorized(vector):
                                            dape_local[slot * vector + lane] = (
                                                dape_local[slot * vector + lane]
                                                + out_local[lane]
                                            )
                                    else:
                                        T.evaluate(
                                            T.call_extern(
                                                "void", "__builtin_assume",
                                                T.reinterpret(
                                                    T.address_of(
                                                        grad_ape[k, half * head_dim + col0]
                                                    ),
                                                    dtype="uint64",
                                                ) % 16 == 0,
                                            )
                                        )
                                        T.atomic_addx4(
                                            grad_ape[k, half * head_dim + col0],
                                            out_local[0],
                                        )
                                else:
                                    T.evaluate(
                                        T.call_extern(
                                            "void", "__builtin_assume",
                                            T.reinterpret(
                                                T.address_of(
                                                    dape_partial[
                                                        row % dape_shards,
                                                        k,
                                                        half * head_dim + col0,
                                                    ]
                                                ),
                                                dtype="uint64",
                                            ) % 16 == 0,
                                        )
                                    )
                                    T.atomic_addx4(
                                        dape_partial[
                                            row % dape_shards, k, half * head_dim + col0
                                        ],
                                        out_local[0],
                                    )
            if direct_atomic and rows_per_warp > 1:
                if row_base + row_lane < nb_valid:
                    for half in T.unroll(coff):
                        for k in T.unroll(ratio):
                            slot = half * ratio + k
                            apex_col = half * head_dim + col0
                            T.evaluate(
                                T.call_extern(
                                    "void", "__builtin_assume",
                                    T.reinterpret(
                                        T.address_of(grad_ape[k, apex_col]),
                                        dtype="uint64",
                                    ) % 16 == 0,
                                )
                            )
                            T.atomic_addx4(
                                grad_ape[k, apex_col], dape_local[slot * vector]
                            )

    return main


def _build_backward_ape_reduce_program_chunked(ratio, width, threads=256, chunks=16):
    """Reduce CTA-partial dAPE with enough parallelism to saturate DRAM.

    The serial reduction has only ``ceil(ratio*width/threads)`` CTAs, so on a
    large output it leaves most MPs idle.  Each chunk here owns a strided
    subset of row blocks and publishes one atomic contribution per output.
    The main kernel remains atomic-free, while ``grad_ape`` retains its
    accumulate-in-place contract.
    """

    chunks = int(chunks)
    if chunks < 1:
        raise ValueError("chunked dAPE reduction requires chunks >= 1")
    num_blocks = T.dynamic("num_blocks")

    @T.prim_func
    def main(
        dape_partial: T.Tensor((num_blocks, ratio, width), T.float32),
        grad_ape: T.Tensor((ratio, width), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(ratio * width, threads), chunks, threads=threads
        ) as (output_block, chunk):
            tx = T.get_thread_binding()
            item = output_block * threads + tx
            if item < ratio * width:
                k = item // width
                column = item % width
                total = T.alloc_var(dtype=T.float32)
                total = 0.0
                for row_iter in T.serial(T.ceildiv(num_blocks, chunks)):
                    row_block = chunk + row_iter * chunks
                    if row_block < num_blocks:
                        total = total + dape_partial[row_block, k, column]
                T.atomic_add(grad_ape[k, column], total)

    return main


def _compile_bwd_r4_vec4(
    key, d, coff, rows_per_cta, rows_per_warp, single_segment, direct_atomic,
    device_index, ratio=4,
):
    """Compile the shared R2/R4 backward main and optional dAPE reducer."""

    with _COMPILE_LOCK:
        cached = _COMPILED.get(key)
        if cached is not None:
            return cached
        query = getattr(torch.musa, "is_current_stream_capturing", None)
        if query is not None and query():
            raise RuntimeError(
                f"CSA compressor: first call for config {key} happened under MUSA "
                "graph capture (JIT compilation is not capture-safe); compile() or "
                "run one eager step for this configuration before capturing."
            )
        main_kernel = tilelang.compile(
            _build_backward_program_vec4_r4(
                int(d), int(coff),
                rows_per_cta=rows_per_cta,
                rows_per_warp=rows_per_warp,
                dape_shards=256,
                single_segment=single_segment,
                direct_atomic=direct_atomic,
                ratio=int(ratio),
            ),
            target="musa",
            pass_configs=_PASS_CONFIGS,
            compile_flags=_COMPILE_FLAGS,
        )
        reduce_kernel = None
        if not direct_atomic:
            reduce_key = (
                "bwd_short_ape_reduce", int(ratio), int(coff * d), 256, 32, device_index,
            )
            reduce_kernel = _COMPILED.get(reduce_key)
            if reduce_kernel is None:
                reduce_kernel = tilelang.compile(
                    _build_backward_ape_reduce_program_chunked(
                        int(ratio), int(coff * d), 256, 32
                    ),
                    target="musa",
                    pass_configs=_PASS_CONFIGS,
                    compile_flags=_COMPILE_FLAGS,
                )
                _COMPILED[reduce_key] = reduce_kernel
        _COMPILED[key] = (main_kernel, reduce_kernel)
    return _COMPILED[key]


def precompile_bwd(ratio, d, coff, device, nb_total=None, num_offsets=None):
    """Compile the R2/R4 vec4 backward for a known capacity and segment count."""

    if device is None:
        raise RuntimeError("CSA TileLang compilation needs a concrete MUSA device")
    if nb_total is None or num_offsets is None:
        raise ValueError("short backward precompile requires nb_total and num_offsets")
    device_index = (
        device.index if device.index is not None else torch.musa.current_device()
    )
    short_schedule = _backward_r4_vec4_schedule(
        ratio, d, coff, nb_total, num_offsets
    )
    if short_schedule is None:
        raise ValueError(f"unsupported short backward configuration: ratio={ratio}, d={d}, coff={coff}")
    rows_per_cta, rows_per_warp, direct_atomic = short_schedule
    single_segment = num_offsets == 2
    key = (
        "bwd_short_vec4", int(ratio), int(d), int(coff), rows_per_cta, rows_per_warp,
        single_segment, direct_atomic, device_index,
    )
    with torch.musa.device(device):
        _compile_bwd_r4_vec4(
            key, d, coff, rows_per_cta, rows_per_warp,
            single_segment, direct_atomic, device_index, ratio=ratio,
        )


def run_bwd(
    kv,
    score,
    ape,
    cu_seqlens,
    cu_seqlens_comp,
    grad_out,
    grad_kv,
    grad_score,
    grad_ape,
    nb_total,
    ratio,
    d,
    coff,
    stream_handle=None,
):
    """Launch the cached backward kernel on the current MUSA stream."""

    del stream_handle
    if int(nb_total) == 0:
        return
    dev = kv.device.index
    short_schedule = _backward_r4_vec4_schedule(
        ratio, d, coff, int(nb_total), cu_seqlens.shape[0]
    )
    if short_schedule is None:
        raise ValueError(f"unsupported short backward configuration: ratio={ratio}, d={d}, coff={coff}")
    rows_per_cta, rows_per_warp, direct_atomic = short_schedule
    single_segment = cu_seqlens.shape[0] == 2
    key = (
        "bwd_short_vec4", int(ratio), int(d), int(coff), rows_per_cta, rows_per_warp,
        single_segment, direct_atomic, dev,
    )
    with torch.musa.device(dev):
        main_kernel, reduce_kernel = _compile_bwd_r4_vec4(
            key, d, coff, rows_per_cta, rows_per_warp,
            single_segment, direct_atomic, dev, ratio=ratio,
        )
        kernel_args = (
            kv,
            score,
            ape,
            cu_seqlens,
            cu_seqlens_comp,
            grad_out,
            grad_kv,
            grad_score,
            grad_ape,
        )
        if direct_atomic:
            # The last argument is unused in the direct specialization.
            main_kernel(*kernel_args, grad_ape.unsqueeze(0))
        else:
            scratch = torch.zeros(
                (256, int(ratio), int(coff * d)),
                dtype=torch.float32,
                device=kv.device,
            )
            main_kernel(*kernel_args, scratch)
            reduce_kernel(scratch, grad_ape)
