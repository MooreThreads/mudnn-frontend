"""MP31  TileLang kernel for the DSA indexer forward (BF16 Q/K).

Port of :mod:`indexer_forward_mp32` with two changes that adapt the kernel to
the ``mp_31`` target on tilelang-musa 0.1.12:

1. Every ``T.tma_copy`` is a plain ``T.copy`` (``barrier=`` dropped; the
   producer's ``T.barrier_arrive`` after the copy keeps the pipeline
   ordering). The TMA lowering of this tilelang build is broken on mp_31:
   the untouched mp32 source dies at runtime with ``MUSA error: cluster
   read fail`` / ``MUSALaunch Error: unknown (code=784)``. Plain copies are
   numerically correct and 5.6x faster than the eager fallback on MP31
   (23.4 ms vs 130.5 ms at B=1, S_q=6144, S_k=12288, H=64, D=128, ratio=4).
2. Pass configs and compile flags come from the ``SPARSE_PREFILL`` recipe
   (:mod:`..utils.prefill`) that the mp31 sparse-attention kernels already
   use, instead of the mp32 literals.

Tuned launch defaults live in :mod:`api_mp31` (``tile_qh=128, tile_kv=256,
stage_kv=2`` measured fastest at the shape above). The mxfp8 variant is
MP32-only and intentionally not ported.

Unlike a TMA descriptor load, a plain ``T.copy`` of the last (unaligned) KV
tile reads out-of-bounds rows within the padded extent; call sites that pass
S_k not divisible by the tile should keep the output padded exactly as the
mp32 path already does.
"""
import tilelang
import tilelang.language as T

from ..utils.prefill import (
    SPARSE_PREFILL_COMPILE_FLAGS,
    SPARSE_PREFILL_PASS_CONFIGS,
)

INDEXER_FORWARD_MP31_TARGET = {"kind": "musa", "arch": "mp_31"}

INDEXER_FORWARD_MP31_PASS_CONFIGS = {
    **SPARSE_PREFILL_PASS_CONFIGS,
}

INDEXER_FORWARD_MP31_COMPILE_FLAGS = list(SPARSE_PREFILL_COMPILE_FLAGS)


@tilelang.jit(
    target=INDEXER_FORWARD_MP31_TARGET,
    pass_configs=INDEXER_FORWARD_MP31_PASS_CONFIGS,
    compile_flags=INDEXER_FORWARD_MP31_COMPILE_FLAGS,
)
def indexer_forward_mp31(
    headdim=128,
    head_q=64,
    tile_qh=128,
    tile_kv=128,
    stage_q=1,
    stage_kv=1,
    dtype="bfloat16",
    dtype_weight="float32",
    is_varlen=False,
    ratio: int = 1,
    sm_scale: float = 1.0,
    has_q_causal_offsets: bool = True,
):
    assert stage_q >= 1 and stage_kv >= 1
    assert ratio >= 1
    assert tile_qh % head_q == 0, "tile_qh must pack whole q-heads"
    assert tile_kv % 128 == 0
    assert tile_kv in (128, 256, 512), "tile_kv 384 hangs at runtime"

    tile_q = tile_qh // head_q
    consumer_threads = (tile_kv // 128) * 128
    producer_threads = 128
    total_threads = producer_threads + consumer_threads

    dtype_bytes = {"float8_e4m3fn": 1, "float4_e2m1fn_x2": 1, "bfloat16": 2}[dtype]
    weight_bytes = {"float32": 4, "bfloat16": 2}[dtype_weight]
    qkv_align = 16 // dtype_bytes
    w_align = 16 // weight_bytes

    smem_capacity_bytes = 256 * 1024
    smem_bytes = (
        (stage_kv * tile_kv * headdim + tile_q * head_q * headdim) * dtype_bytes
        + tile_q * head_q * weight_bytes
    )
    assert smem_bytes <= smem_capacity_bytes, f"smem budget exceeded: {smem_bytes} bytes"

    if is_varlen:
        total_q = T.dynamic("total_q")
        total_k = T.dynamic("total_k")
        batch_p1 = T.dynamic("batch_p1")
        max_seqlen_k = T.dynamic("max_seqlen_k")
        batch = batch_p1 - 1

        q_stride_s = T.dynamic("q_stride_s")
        q_stride_h = T.dynamic("q_stride_h")
        kv_stride_s = T.dynamic("kv_stride_s")
        w_stride_s = T.dynamic("w_stride_s")
        o_stride_s = T.dynamic("o_stride_s")

        q_type = T.StridedTensor(
            [total_q, head_q, headdim], (q_stride_s, q_stride_h, 1), dtype
        )
        kv_type = T.StridedTensor([total_k, headdim], (kv_stride_s, 1), dtype)
        weight_type = T.StridedTensor(
            [total_q, head_q], (w_stride_s, 1), dtype_weight
        )
        out_type = T.StridedTensor(
            [total_q, max_seqlen_k], (o_stride_s, 1), "float32"
        )
        cu_q_type = T.Tensor([batch_p1], "int32")
        cu_k_type = T.Tensor([batch_p1], "int32")
        q_causal_offsets_type = T.Tensor([batch], "int32")
    else:
        batch = T.dynamic("batch")
        seqlen_q = T.dynamic("seqlen_q")
        seqlen_k = T.dynamic("seqlen_k")

        q_stride_b = T.dynamic("q_stride_b")
        q_stride_s = T.dynamic("q_stride_s")
        q_stride_h = T.dynamic("q_stride_h")
        kv_stride_b = T.dynamic("kv_stride_b")
        kv_stride_s = T.dynamic("kv_stride_s")
        w_stride_b = T.dynamic("w_stride_b")
        w_stride_s = T.dynamic("w_stride_s")
        o_stride_b = T.dynamic("o_stride_b")
        o_stride_s = T.dynamic("o_stride_s")

        q_type = T.StridedTensor(
            [batch, seqlen_q, head_q, headdim],
            (q_stride_b, q_stride_s, q_stride_h, 1),
            dtype,
        )
        kv_type = T.StridedTensor(
            [batch, seqlen_k, headdim], (kv_stride_b, kv_stride_s, 1), dtype
        )
        weight_type = T.StridedTensor(
            [batch, seqlen_q, head_q], (w_stride_b, w_stride_s, 1), dtype_weight
        )
        out_type = T.StridedTensor(
            [batch, seqlen_q, seqlen_k], (o_stride_b, o_stride_s, 1), "float32"
        )
        q_causal_offsets_type = T.Tensor([batch], "int32")

    @T.macro
    def load(
        q,
        kv,
        weight,
        q_shared,
        kv_shared,
        weight_shared,
        bar_q_full,
        bar_kv_full,
        bar_kv_empty,
        producer_kv_phase,
        num_tile_kv,
        batch_idx,
        tile_q_idx,
    ):
        T.copy(
            q[batch_idx, tile_q_idx * tile_q : (tile_q_idx + 1) * tile_q, 0:head_q, 0:headdim],
            q_shared)
        T.copy(
            weight[batch_idx, tile_q_idx * tile_q : (tile_q_idx + 1) * tile_q, 0:head_q],
            weight_shared)
        T.barrier_arrive(bar_q_full)

        for it in T.serial(num_tile_kv):
            stage_idx = it % stage_kv
            T.barrier_wait(bar_kv_empty[stage_idx], producer_kv_phase[0] & 1)
            T.copy(
                kv[batch_idx, it * tile_kv : (it + 1) * tile_kv, 0:headdim],
                kv_shared[stage_idx, :, :])
            T.barrier_arrive(bar_kv_full[stage_idx])
            if stage_idx == stage_kv - 1:
                producer_kv_phase[0] = producer_kv_phase[0] ^ 1

    @T.macro
    def load_varlen(
        q,
        kv,
        weight,
        q_shared,
        kv_shared,
        weight_shared,
        bar_q_full,
        bar_kv_full,
        bar_kv_empty,
        producer_kv_phase,
        num_tile_kv,
        q_row_base,
        kv_row_base,
    ):
        T.copy(
            q[q_row_base : q_row_base + tile_q, 0:head_q, 0:headdim],
            q_shared)
        T.copy(
            weight[q_row_base : q_row_base + tile_q, 0:head_q],
            weight_shared)
        T.barrier_arrive(bar_q_full)

        for it in T.serial(num_tile_kv):
            stage_idx = it % stage_kv
            T.barrier_wait(bar_kv_empty[stage_idx], producer_kv_phase[0] & 1)
            T.copy(
                kv[kv_row_base + it * tile_kv : kv_row_base + (it + 1) * tile_kv, 0:headdim],
                kv_shared[stage_idx, :, :])
            T.barrier_arrive(bar_kv_full[stage_idx])
            if stage_idx == stage_kv - 1:
                producer_kv_phase[0] = producer_kv_phase[0] ^ 1

    @T.macro
    def apply_relu(acc3):
        for kv, qi, h in T.Parallel(tile_kv, tile_q, head_q):
            acc3[kv, qi, h] = T.max(acc3[kv, qi, h], 0.0)

    @T.macro
    def weighted_reduce(acc3, w_reg, reduced):
        for kv, qi, h in T.Parallel(tile_kv, tile_q, head_q):
            acc3[kv, qi, h] = acc3[kv, qi, h] * w_reg[qi, h]
        T.reduce_sum(acc3, reduced, dim=2, clear=True)

    @T.macro
    def epilogue(
        reduced,
        out,
        out_row_base,
        q_tile_base,
        q_len,
        k_len,
        q_causal_offset,
        it,
    ):
        for kv, qi in T.Parallel(tile_kv, tile_q):
            q_local = q_tile_base + qi
            k_local = it * tile_kv + kv
            if (
                q_local < q_len
                and k_local < k_len
                and k_local < (q_causal_offset + q_local + 1) // ratio
            ):
                if sm_scale == 1.0:
                    out[out_row_base + qi, k_local] = reduced[kv, qi]
                else:
                    out[out_row_base + qi, k_local] = reduced[kv, qi] * sm_scale

    @T.macro
    def compute(
        kv_shared,
        q_shared_2d,
        acc_s,
        reduced,
        weight_shared,
        out,
        out_row_base,
        q_tile_base,
        q_len,
        k_len,
        q_causal_offset,
        bar_q_full,
        bar_kv_full,
        bar_kv_empty,
        consumer_kv_phase,
        num_tile_kv,
    ):
        acc3 = T.reshape(acc_s, [tile_kv, tile_q, head_q])
        w_reg = T.alloc_fragment([tile_q, head_q], dtype_weight)

        T.barrier_wait(bar_q_full, 0)
        T.copy(weight_shared, w_reg)

        for it in T.serial(num_tile_kv):
            stage_idx = it % stage_kv
            T.barrier_wait(bar_kv_full[stage_idx], consumer_kv_phase[0] & 1)

            T.clear(acc_s)
            T.gemm(
                kv_shared[stage_idx, :, :],
                q_shared_2d,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )

            T.barrier_arrive(bar_kv_empty[stage_idx])

            apply_relu(acc3)
            weighted_reduce(acc3, w_reg, reduced)
            epilogue(reduced, out, out_row_base, q_tile_base, q_len, k_len, q_causal_offset, it)

            if stage_idx == stage_kv - 1:
                consumer_kv_phase[0] = consumer_kv_phase[0] ^ 1

    @T.macro
    def fwd(q, kv, weight, q_causal_offsets, out, tile_m_idx, batch_idx):
        T.assume(q_stride_b % qkv_align == 0)
        T.assume(q_stride_s % qkv_align == 0)
        T.assume(q_stride_h % qkv_align == 0)
        T.assume(kv_stride_b % qkv_align == 0)
        T.assume(kv_stride_s % qkv_align == 0)
        T.assume(w_stride_b % w_align == 0)
        T.assume(w_stride_s % w_align == 0)

        if has_q_causal_offsets:
            q_causal_offset = q_causal_offsets[batch_idx]
        else:
            q_causal_offset = T.int32(0)

        q_last = T.min((tile_m_idx + 1) * tile_q, seqlen_q) - 1
        num_tile_kv = T.ceildiv(
            T.min((q_causal_offset + q_last + 1) // ratio, seqlen_k),
            tile_kv,
        )

        q_shared = T.alloc_shared([tile_q, head_q, headdim], dtype)
        q_shared_2d = T.view(q_shared, [tile_qh, headdim])
        kv_shared = T.alloc_shared([stage_kv, tile_kv, headdim], dtype)
        weight_shared = T.alloc_shared([tile_q, head_q], dtype_weight)

        bar_q_full = T.alloc_barrier(producer_threads)
        bar_kv_full = T.alloc_barrier([producer_threads] * stage_kv)
        bar_kv_empty = T.alloc_barrier([consumer_threads] * stage_kv)

        T.sync_threads()

        tid = T.get_thread_binding()

        if tid >= consumer_threads:
            producer_kv_phase = T.alloc_local([1], T.int32)
            T.fill(producer_kv_phase, 1)
            load(
                q,
                kv,
                weight,
                q_shared,
                kv_shared,
                weight_shared,
                bar_q_full,
                bar_kv_full,
                bar_kv_empty,
                producer_kv_phase,
                num_tile_kv,
                batch_idx,
                tile_m_idx,
            )

        if tid < consumer_threads:
            consumer_kv_phase = T.alloc_local([1], T.int32)
            T.fill(consumer_kv_phase, 0)

            acc_s = T.alloc_fragment([tile_kv, tile_qh], "float32")
            reduced = T.alloc_fragment([tile_kv, tile_q], "float32")
            out_b = T.alias_buffer(
                out, [seqlen_q, seqlen_k], [o_stride_s, 1], batch_idx * o_stride_b
            )
            compute(
                kv_shared,
                q_shared_2d,
                acc_s,
                reduced,
                weight_shared,
                out_b,
                tile_m_idx * tile_q,
                tile_m_idx * tile_q,
                seqlen_q,
                seqlen_k,
                q_causal_offset,
                bar_q_full,
                bar_kv_full,
                bar_kv_empty,
                consumer_kv_phase,
                num_tile_kv,
            )

    @T.macro
    def fwd_varlen(q, kv, weight, q_causal_offsets, out, cu_q, cu_k, tile_m_idx, batch_idx):
        T.assume(q_stride_s % qkv_align == 0)
        T.assume(q_stride_h % qkv_align == 0)
        T.assume(kv_stride_s % qkv_align == 0)
        T.assume(w_stride_s % w_align == 0)

        q_start = cu_q[batch_idx]
        q_len = cu_q[batch_idx + 1] - q_start
        k_start = cu_k[batch_idx]
        k_len = cu_k[batch_idx + 1] - k_start

        num_tiles = T.ceildiv(q_len, tile_q)
        q_tile_base = tile_m_idx * tile_q
        q_row_base = q_start + q_tile_base

        if has_q_causal_offsets:
            q_causal_offset = q_causal_offsets[batch_idx]
        else:
            q_causal_offset = T.int32(0)

        q_last = T.min(q_tile_base + tile_q, q_len) - 1
        num_tile_kv = T.ceildiv(
            T.min((q_causal_offset + q_last + 1) // ratio, k_len),
            tile_kv,
        )

        q_shared = T.alloc_shared([tile_q, head_q, headdim], dtype)
        q_shared_2d = T.view(q_shared, [tile_qh, headdim])
        kv_shared = T.alloc_shared([stage_kv, tile_kv, headdim], dtype)
        weight_shared = T.alloc_shared([tile_q, head_q], dtype_weight)

        bar_q_full = T.alloc_barrier(producer_threads)
        bar_kv_full = T.alloc_barrier([producer_threads] * stage_kv)
        bar_kv_empty = T.alloc_barrier([consumer_threads] * stage_kv)

        T.sync_threads()

        tid = T.get_thread_binding()

        if tile_m_idx < num_tiles:
            if tid >= consumer_threads:
                producer_kv_phase = T.alloc_local([1], T.int32)
                T.fill(producer_kv_phase, 1)
                load_varlen(
                    q,
                    kv,
                    weight,
                    q_shared,
                    kv_shared,
                    weight_shared,
                    bar_q_full,
                    bar_kv_full,
                    bar_kv_empty,
                    producer_kv_phase,
                    num_tile_kv,
                    q_row_base,
                    k_start,
                )

            if tid < consumer_threads:
                consumer_kv_phase = T.alloc_local([1], T.int32)
                T.fill(consumer_kv_phase, 0)

                acc_s = T.alloc_fragment([tile_kv, tile_qh], "float32")
                reduced = T.alloc_fragment([tile_kv, tile_q], "float32")
                compute(
                    kv_shared,
                    q_shared_2d,
                    acc_s,
                    reduced,
                    weight_shared,
                    out,
                    q_row_base,
                    q_tile_base,
                    q_len,
                    k_len,
                    q_causal_offset,
                    bar_q_full,
                    bar_kv_full,
                    bar_kv_empty,
                    consumer_kv_phase,
                    num_tile_kv,
                )

    if is_varlen:
        if has_q_causal_offsets:
            @T.prim_func
            def run(
                q: q_type,
                kv: kv_type,
                weight: weight_type,
                q_causal_offsets: q_causal_offsets_type,
                out: out_type,
                cu_q: cu_q_type,
                cu_k: cu_k_type,
                max_seqlen_q: T.int32,
            ):
                with T.Kernel(
                    T.ceildiv(max_seqlen_q, tile_q), batch, threads=total_threads
                ) as (tile_m_idx, batch_idx):
                    fwd_varlen(q, kv, weight, q_causal_offsets, out, cu_q, cu_k, tile_m_idx, batch_idx)
        else:
            @T.prim_func
            def run(
                q: q_type,
                kv: kv_type,
                weight: weight_type,
                out: out_type,
                cu_q: cu_q_type,
                cu_k: cu_k_type,
                max_seqlen_q: T.int32,
            ):
                with T.Kernel(
                    T.ceildiv(max_seqlen_q, tile_q), batch, threads=total_threads
                ) as (tile_m_idx, batch_idx):
                    fwd_varlen(q, kv, weight, None, out, cu_q, cu_k, tile_m_idx, batch_idx)
    elif has_q_causal_offsets:
        @T.prim_func
        def run(
            q: q_type,
            kv: kv_type,
            weight: weight_type,
            q_causal_offsets: q_causal_offsets_type,
            out: out_type,
        ):
            with T.Kernel(
                T.ceildiv(seqlen_q, tile_q), batch, threads=total_threads
            ) as (tile_m_idx, batch_idx):
                fwd(q, kv, weight, q_causal_offsets, out, tile_m_idx, batch_idx)
    else:
        @T.prim_func
        def run(q: q_type, kv: kv_type, weight: weight_type, out: out_type):
            with T.Kernel(
                T.ceildiv(seqlen_q, tile_q), batch, threads=total_threads
            ) as (tile_m_idx, batch_idx):
                fwd(q, kv, weight, None, out, tile_m_idx, batch_idx)

    return run
