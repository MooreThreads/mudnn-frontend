"""Warp-specialized GDN forward chunk scan and the handoff it saves for backward.

The kernel and its wrapper are a verbatim extract of the upstream bundle;
``gdn_chunk_fwd`` at the bottom is the port's orchestration.
"""

import torch
import tilelang
import tilelang.language as T

from ._jit import FWD_COMPILE_FLAGS, FWD_PASS_CONFIGS
from .gdn_kkt import kkt_solve_train
from .gdn_shapes import gdn_head_dim_error, head_dim_supported
from .gdn_varlen import dense_cu, varlen_maps

__all__ = ["fused_chunk_gdn_prefill_train", "gdn_chunk_fwd"]


def _tl_contiguous(x):
    """Give ``x`` the exact row-major strides a TileLang buffer parameter wants.

    ``Tensor.is_contiguous()`` ignores the stride of size-1 axes, so torch can hand
    out a stride tuple that a ``T.Tensor`` host stub rejects even though torch calls
    the tensor contiguous -- and ``.contiguous()`` will not fix it. Re-striding is
    free in that case; a genuinely strided tensor still gets a copy.

    Call this inside the autograd Function: ``as_strided`` is differentiable and its
    backward would materialize a zeroed buffer for nothing.
    """
    stride, canonical, acc = x.stride(), [], 1
    for size in reversed(x.shape):
        canonical.append(acc)
        acc *= size
    canonical = tuple(reversed(canonical))
    if stride == canonical:
        return x
    if x.is_contiguous():
        return x.as_strided(x.shape, canonical)
    return x.contiguous()


UNROLL_MIN_CHUNKS = 64
UNROLL_MAX_CTAS = 128


@tilelang.jit(pass_configs=FWD_PASS_CONFIGS, compile_flags=FWD_COMPILE_FLAGS)
def tilelang_fused_chunk_gdn_prefill_train(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    unroll_chunk_loop=False,
    padded_store=False,
    chunk_padded=False,
):
    padded_store = padded_store or chunk_padded
    num_tokens = T.dynamic("num_tokens")
    raw_batch_size = T.dynamic("raw_batch_size")
    num_chunks = T.dynamic("num_chunks")
    block_S = chunk_size
    if not head_dim_supported(DK, DV, chunk_size):
        raise ValueError(gdn_head_dim_error("gdn_prefill", DK, DV, chunk_size))

    q_shape = (1, num_tokens, Hg, DK)
    k_shape = (1, num_tokens, Hg, DK)
    v_shape = (1, num_tokens, H, DV)
    o_shape = (1, num_tokens, H, DV)
    a_shape = (1, num_tokens, H, chunk_size)
    g_shape = (1, num_tokens, H)
    b_shape = (1, num_tokens, H)
    store_tokens = T.dynamic("store_tokens") if padded_store else num_tokens
    d_shape = o_shape[:1] + (store_tokens,) + o_shape[2:]
    gp_shape = (H,) + g_shape[:-2] + (store_tokens,)
    b_shape = (H,) + b_shape[:-2] + (store_tokens,)
    a_shape = a_shape[:1] + (store_tokens,) + a_shape[2:]
    h0_shape = (raw_batch_size, H, DK, DV)
    ht_shape = (raw_batch_size, H, DK, DV)
    hs_shape = ((1, H, num_chunks, DK, DV) if chunk_padded
                else (raw_batch_size, H, num_chunks, DK, DV))
    seqlens_shape = ((2 * (raw_batch_size + 1),) if chunk_padded
                     else (raw_batch_size + 1,))

    q_stride_b = T.dynamic("q_stride_b")
    q_stride_t = T.dynamic("q_stride_t")
    q_stride_h = T.dynamic("q_stride_h")
    k_stride_b = T.dynamic("k_stride_b")
    k_stride_t = T.dynamic("k_stride_t")
    k_stride_h = T.dynamic("k_stride_h")
    v_stride_b = T.dynamic("v_stride_b")
    v_stride_t = T.dynamic("v_stride_t")
    v_stride_h = T.dynamic("v_stride_h")
    q_strides = (q_stride_b, q_stride_t, q_stride_h, 1)
    k_strides = (k_stride_b, k_stride_t, k_stride_h, 1)
    v_strides = (v_stride_b, v_stride_t, v_stride_h, 1)

    @T.macro
    def producer_phase(i):
        return T.bitwise_xor((i // 2) % 2, 1)

    @T.macro
    def consumer_phase(i):
        return (i // 2) % 2

    @T.macro
    def inv_perm_dv(i):
        stride = DV // 8
        return (i % stride) * 8 + (i // stride)

    # The MUSA TileLang release used by the CI image resolves prim-func
    # annotations with ``typing.get_type_hints`` but does not pass closure
    # locals such as ``q_shape`` and ``q_strides``.  Keep this module free of
    # postponed annotations so these symbolic shapes are evaluated while the
    # inner function is defined.
    @T.prim_func
    def tilelang_fused_chunk_gdn_prefill_train_kernel(
        q: T.StridedTensor(q_shape, q_strides, qkva_dtype),
        k: T.StridedTensor(k_shape, k_strides, qkva_dtype),
        v: T.StridedTensor(v_shape, v_strides, qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor(seqlens_shape, dtype=seqlen_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
        hs: T.Tensor(hs_shape, dtype=qkva_dtype),
        delta: T.Tensor(d_shape, dtype=qkva_dtype),
        g_prefix_out: T.Tensor(gp_shape, dtype=accum_dtype),
    ):
        launch_batch_size = raw_batch_size
        cta_threads = 1024
        warpgroup_threads = cta_threads // 8
        # Must be DERIVED: too high only wastes threads, too low means nobody
        # drains and the Hs race comes back silently.
        hs_drain_tx_limit = 2 * warpgroup_threads + 32
        with T.Kernel(launch_batch_size * H, threads=cta_threads) as (bbh,):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")

            batch_idx = 0
            seq_start_idx = cu_seqlens[bb]
            seq_end_idx = cu_seqlens[bb + 1]

            seq_len = T.alloc_var("int32")
            seq_len = seq_end_idx - seq_start_idx
            num_iters = T.alloc_var("int32")
            num_iters = T.ceildiv(seq_len, block_S)
            if chunk_padded:
                chunk_base = T.alloc_var("int32")
                chunk_base = cu_seqlens[(raw_batch_size + 1) + bb]

            q_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            k_shared = T.alloc_shared((2, block_S, DK), dtype=qkva_dtype)
            scaled_k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            pa_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")
            b_shared = T.alloc_shared((2, block_S), dtype=accum_dtype, scope="shared")

            h_shared = T.alloc_shared((2, DK, DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, DV), dtype=qkva_dtype)
            g_exp_shared = T.alloc_shared(
                (2, block_S), dtype=accum_dtype, scope="shared"
            )
            g_exp_rev_shared = T.alloc_shared(
                (2, block_S), dtype=accum_dtype, scope="shared"
            )
            h_fragment = T.alloc_fragment((DK, DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            uv_fragment = T.alloc_fragment((block_S, DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            q_is_ready = T.alloc_barrier(arrive_count=[32] * 2)
            q_is_free = T.alloc_barrier(arrive_count=[640] * 2)
            k_is_ready = T.alloc_barrier(arrive_count=[32] * 2)
            k_is_free = T.alloc_barrier(arrive_count=[640] * 2)
            gb_is_ready = T.alloc_barrier(arrive_count=[64] * 2)
            gb_is_free = T.alloc_barrier(arrive_count=[768] * 2)
            v_is_ready = T.alloc_barrier(arrive_count=32)
            v_is_free = T.alloc_barrier(arrive_count=128)

            b_is_ready = T.alloc_barrier(arrive_count=64)
            a_is_ready = T.alloc_barrier(arrive_count=32)
            a_is_free = T.alloc_barrier(arrive_count=128)
            inv_is_ready = T.alloc_barrier(arrive_count=512)
            inv_is_free = T.alloc_barrier(arrive_count=128)
            p_is_ready = T.alloc_barrier(arrive_count=512)
            scaled_k_is_ready = T.alloc_barrier(arrive_count=512)
            vd_is_ready = T.alloc_barrier(arrive_count=128)
            vd_is_free = T.alloc_barrier(arrive_count=512)
            h_is_ready = T.alloc_barrier(arrive_count=[512] * 2)
            h_is_free = T.alloc_barrier(arrive_count=[256] * 2)

            tx = T.get_thread_binding()

            with T.ws(0):
                w_local = T.alloc_local([8], accum_dtype)
                for i_s in T.serial(num_iters):
                    store_left = ((chunk_base + i_s) * block_S if chunk_padded
                                  else seq_start_idx + i_s * block_S)
                    T.barrier_wait(h_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(k_is_ready[i_s % 2], consumer_phase(i_s))
                    T.gemm(
                        k_shared[i_s % 2, :, :],
                        h_shared[i_s % 2, :, :],
                        uv_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(v_is_ready, i_s % 2)
                    T.warpgroup_wait(0)
                    T.barrier_arrive(k_is_free[i_s % 2])
                    T.barrier_arrive(h_is_free[i_s % 2])

                    ldg_tx = tx % 8
                    ldg_ty = tx // 8
                    for j_r in T.unroll(block_S // 16):
                        for j_p in T.unroll(max(1, DV // 64)):
                            for j_u in T.vectorized(8):
                                j_s = j_r * 16 + ldg_ty
                                w_local[j_u] = v_shared[
                                    j_s, ldg_tx * 8 + j_p * (DV // 2) + j_u
                                ]
                            for j_u in T.vectorized(8):
                                j_s = j_r * 16 + ldg_ty
                                j_v = (j_u * (DV // 64) + j_p) * 8 + ldg_tx
                                w_local[j_u] -= (
                                    g_exp_shared[i_s % 2, j_s]
                                    * uv_fragment[j_s, j_v]
                                )
                            for j_u in T.vectorized(8):
                                j_s = j_r * 16 + ldg_ty
                                j_v = ldg_tx * 8 + j_p * (DV // 2) + j_u
                                v_shared[j_s, j_v] = w_local[j_u]
                    T.lma_wait()
                    T.barrier_arrive(gb_is_free[i_s % 2])

                    T.barrier_wait(inv_is_ready, i_s % 2)
                    T.lma_wait()
                    T.gemm(
                        pa_shared,
                        v_shared,
                        uv_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    T.warpgroup_wait(0)
                    T.barrier_arrive(inv_is_free)
                    T.barrier_wait(vd_is_free, T.bitwise_xor(i_s % 2, 1))
                    pub_local = T.alloc_local([8], qkva_dtype)
                    for j_r in T.unroll(block_S // 16):
                        for j_h in T.unroll(max(1, DV // 64)):
                            for j_e in T.vectorized(8):
                                pub_local[j_e] = uv_fragment[
                                    j_r * 16 + ldg_ty,
                                    (((j_h * 8 + j_e) // 2) * 16
                                     + ((j_h * 8 + j_e) % 2) * 8
                                     + ldg_tx) if DV == 128 else
                                    ((j_h * 8 + j_e) * 8 + ldg_tx),
                                ]
                            for j_e in T.vectorized(8):
                                v_shared[
                                    j_r * 16 + ldg_ty,
                                    ldg_tx * (DV // 8) + j_h * 8 + j_e
                                ] = pub_local[j_e]
                            for j_e in T.vectorized(8):
                                vd_shared[
                                    j_r * 16 + ldg_ty,
                                    ldg_tx * (DV // 8) + j_h * 8 + j_e
                                ] = pub_local[j_e]
                    T.lma_wait()
                    T.barrier_arrive(vd_is_ready)
                    T.copy(
                        uv_fragment,
                        delta[
                            batch_idx,
                            store_left : store_left + block_S,
                            bh,
                            0:DV,
                        ],
                    )

            with T.ws(1):
                for i_s in T.serial(num_iters):
                    store_left = seq_start_idx + i_s * block_S
                    valid_seqs = T.min(seq_end_idx - store_left, block_S)

                    T.barrier_wait(q_is_ready[i_s % 2], consumer_phase(i_s))
                    T.barrier_wait(h_is_ready[i_s % 2], consumer_phase(i_s))
                    T.gemm(
                        q_shared[i_s % 2, :, :],
                        h_shared[i_s % 2, :, :],
                        o_fragment,
                        clear_accum=True,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()

                    T.warpgroup_wait(0)
                    T.barrier_arrive(q_is_free[i_s % 2])
                    T.barrier_arrive(h_is_free[i_s % 2])

                    T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                    for j_s, j_v in T.Parallel(block_S, DV):
                        o_fragment[j_s, j_v] *= g_exp_shared[i_s % 2, j_s] * scale
                    T.barrier_arrive(gb_is_free[i_s % 2])

                    T.barrier_wait(p_is_ready, i_s % 2)
                    T.barrier_wait(vd_is_ready, i_s % 2)
                    T.gemm(
                        pa_shared,
                        v_shared,
                        o_fragment,
                        clear_accum=False,
                        wg_wait=-1,
                    )
                    T.warpgroup_commit_batch()
                    T.warpgroup_wait(0)
                    T.barrier_arrive(a_is_free)
                    T.barrier_arrive(v_is_free)
                    o_local = T.alloc_local([8], o_dtype)
                    o_tx = tx % 8
                    o_ty = (tx // 8) % 16
                    for j_r in T.unroll(block_S // 16):
                        for j_p in T.unroll(max(1, DV // 64)):
                            for j_d in T.vectorized(8):
                                o_local[j_d] = o_fragment[
                                    j_r * 16 + o_ty,
                                    (j_d * (DV // 64) + j_p) * 8 + o_tx
                                ]
                            if j_r * 16 + o_ty < valid_seqs:
                                for j_d in T.vectorized(8):
                                    o[
                                        batch_idx,
                                        store_left + j_r * 16 + o_ty,
                                        bh,
                                        j_p * (DV // 2) + o_tx * 8 + j_d,
                                    ] = o_local[j_d]

            with T.ws(2, 3, 4, 5):
                if use_initial_state:
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_fragment[j_k, j_v] = h0[bb, bh, inv_perm_dv(j_v), j_k]
                else:
                    for j_k, j_v in T.Parallel(DK, DV):
                        h_fragment[j_k, j_v] = 0.0
                if unroll_chunk_loop:
                    for i_o in T.serial(num_iters // 2):
                        u_slot_e = T.alloc_var("int32")
                        u_slot_e = 0
                        valid_seqs_e = T.min(
                            seq_end_idx - seq_start_idx - (i_o * 2) * block_S, block_S
                        )
                        T.barrier_wait(h_is_free[u_slot_e], T.bitwise_xor(i_o % 2, 1))
                        for j_k, j_v in T.Parallel(DK, DV):
                            h_shared[0, j_k, j_v] = h_fragment[j_k, j_v]
                        T.lma_wait()
                        T.sync_threads(0, 512)
                        T.barrier_arrive(h_is_ready[u_slot_e])
                        T.tma_copy(
                            h_shared[0, :, :],
                            hs[0 if chunk_padded else bb, bh, (chunk_base + (i_o * 2)) if chunk_padded else (i_o * 2), 0:DK, 0:DV],
                        )

                        T.barrier_wait(q_is_ready[u_slot_e], i_o % 2)
                        T.barrier_wait(k_is_ready[u_slot_e], i_o % 2)
                        T.gemm(
                            q_shared[0, :, :],
                            k_shared[0, :, :],
                            p_fragment,
                            transpose_B=True,
                            clear_accum=True,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(gb_is_ready[u_slot_e], i_o % 2)
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            g_fragment[j_s, j_t] = (
                                g_shared[0, j_s] - g_shared[0, j_t]
                            )
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            g_fragment[j_s, j_t] = T.if_then_else(
                                g_fragment[j_s, j_t] <= 0.0,
                                g_fragment[j_s, j_t],
                                -float("inf"),
                            )
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            g_fragment[j_s, j_t] = T.exp2(
                                g_fragment[j_s, j_t] * 1.4426950408889634
                            )

                        T.barrier_wait(a_is_ready, 0)
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            pa_shared[j_s, j_t] = (
                                g_fragment[j_s, j_t]
                                * pa_shared[j_s, j_t]
                                * b_shared[0, j_t]
                            )
                        T.lma_wait()
                        T.barrier_arrive(inv_is_ready)
                        for j_s, j_k in T.Parallel(block_S, DK):
                            scaled_k_shared[j_s, j_k] = (
                                k_shared[0, j_s, j_k]
                                * g_exp_rev_shared[0, j_s]
                            )

                        T.warpgroup_wait(0)
                        T.barrier_arrive(q_is_free[u_slot_e])
                        T.barrier_arrive(k_is_free[u_slot_e])
                        g_last_local_e = g_exp_shared[0, valid_seqs_e - 1]
                        for j_k, j_v in T.Parallel(DK, DV):
                            h_fragment[j_k, j_v] *= g_last_local_e
                        T.barrier_arrive(gb_is_free[u_slot_e])

                        T.barrier_wait(inv_is_free, 0)
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            pa_shared[j_s, j_t] = (
                                g_fragment[j_s, j_t] * p_fragment[j_s, j_t] * scale
                            )
                        T.lma_wait()
                        T.barrier_arrive(p_is_ready)

                        if tx < hs_drain_tx_limit:
                            T.tma_store_wait()
                        T.barrier_wait(vd_is_ready, 0)
                        T.lma_wait()
                        T.sync_threads(0, 512)
                        T.gemm(
                            scaled_k_shared,
                            vd_shared,
                            h_fragment,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            clear_accum=False,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(vd_is_free)
                        u_slot_o = T.alloc_var("int32")
                        u_slot_o = 1
                        valid_seqs_o = T.min(
                            seq_end_idx - seq_start_idx - (i_o * 2 + 1) * block_S, block_S
                        )
                        T.barrier_wait(h_is_free[u_slot_o], T.bitwise_xor(i_o % 2, 1))
                        for j_k, j_v in T.Parallel(DK, DV):
                            h_shared[1, j_k, j_v] = h_fragment[j_k, j_v]
                        T.lma_wait()
                        T.sync_threads(0, 512)
                        T.barrier_arrive(h_is_ready[u_slot_o])
                        T.tma_copy(
                            h_shared[1, :, :],
                            hs[0 if chunk_padded else bb, bh, (chunk_base + (i_o * 2 + 1)) if chunk_padded else (i_o * 2 + 1), 0:DK, 0:DV],
                        )

                        T.barrier_wait(q_is_ready[u_slot_o], i_o % 2)
                        T.barrier_wait(k_is_ready[u_slot_o], i_o % 2)
                        T.gemm(
                            q_shared[1, :, :],
                            k_shared[1, :, :],
                            p_fragment,
                            transpose_B=True,
                            clear_accum=True,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(gb_is_ready[u_slot_o], i_o % 2)
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            g_fragment[j_s, j_t] = (
                                g_shared[1, j_s] - g_shared[1, j_t]
                            )
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            g_fragment[j_s, j_t] = T.if_then_else(
                                g_fragment[j_s, j_t] <= 0.0,
                                g_fragment[j_s, j_t],
                                -float("inf"),
                            )
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            g_fragment[j_s, j_t] = T.exp2(
                                g_fragment[j_s, j_t] * 1.4426950408889634
                            )

                        T.barrier_wait(a_is_ready, 1)
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            pa_shared[j_s, j_t] = (
                                g_fragment[j_s, j_t]
                                * pa_shared[j_s, j_t]
                                * b_shared[1, j_t]
                            )
                        T.lma_wait()
                        T.barrier_arrive(inv_is_ready)
                        for j_s, j_k in T.Parallel(block_S, DK):
                            scaled_k_shared[j_s, j_k] = (
                                k_shared[1, j_s, j_k]
                                * g_exp_rev_shared[1, j_s]
                            )

                        T.warpgroup_wait(0)
                        T.barrier_arrive(q_is_free[u_slot_o])
                        T.barrier_arrive(k_is_free[u_slot_o])
                        g_last_local_o = g_exp_shared[1, valid_seqs_o - 1]
                        for j_k, j_v in T.Parallel(DK, DV):
                            h_fragment[j_k, j_v] *= g_last_local_o
                        T.barrier_arrive(gb_is_free[u_slot_o])

                        T.barrier_wait(inv_is_free, 1)
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            pa_shared[j_s, j_t] = (
                                g_fragment[j_s, j_t] * p_fragment[j_s, j_t] * scale
                            )
                        T.lma_wait()
                        T.barrier_arrive(p_is_ready)

                        if tx < hs_drain_tx_limit:
                            T.tma_store_wait()
                        T.barrier_wait(vd_is_ready, 1)
                        T.lma_wait()
                        T.sync_threads(0, 512)
                        T.gemm(
                            scaled_k_shared,
                            vd_shared,
                            h_fragment,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            clear_accum=False,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(vd_is_free)
                    if num_iters % 2 == 1:
                            u_slot_t = T.alloc_var("int32")
                            u_slot_t = 0
                            valid_seqs_t = T.min(
                                seq_end_idx - seq_start_idx - (num_iters - 1) * block_S, block_S
                            )
                            T.barrier_wait(h_is_free[u_slot_t], T.bitwise_xor((num_iters // 2) % 2, 1))
                            for j_k, j_v in T.Parallel(DK, DV):
                                h_shared[0, j_k, j_v] = h_fragment[j_k, j_v]
                            T.lma_wait()
                            T.sync_threads(0, 512)
                            T.barrier_arrive(h_is_ready[u_slot_t])
                            T.tma_copy(
                                h_shared[0, :, :],
                                hs[0 if chunk_padded else bb, bh, (chunk_base + (num_iters - 1)) if chunk_padded else (num_iters - 1), 0:DK, 0:DV],
                            )

                            T.barrier_wait(q_is_ready[u_slot_t], (num_iters // 2) % 2)
                            T.barrier_wait(k_is_ready[u_slot_t], (num_iters // 2) % 2)
                            T.gemm(
                                q_shared[0, :, :],
                                k_shared[0, :, :],
                                p_fragment,
                                transpose_B=True,
                                clear_accum=True,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()

                            T.barrier_wait(gb_is_ready[u_slot_t], (num_iters // 2) % 2)
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                g_fragment[j_s, j_t] = (
                                    g_shared[0, j_s] - g_shared[0, j_t]
                                )
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                g_fragment[j_s, j_t] = T.if_then_else(
                                    g_fragment[j_s, j_t] <= 0.0,
                                    g_fragment[j_s, j_t],
                                    -float("inf"),
                                )
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                g_fragment[j_s, j_t] = T.exp2(
                                    g_fragment[j_s, j_t] * 1.4426950408889634
                                )

                            T.barrier_wait(a_is_ready, 0)
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                pa_shared[j_s, j_t] = (
                                    g_fragment[j_s, j_t]
                                    * pa_shared[j_s, j_t]
                                    * b_shared[0, j_t]
                                )
                            T.lma_wait()
                            T.barrier_arrive(inv_is_ready)
                            for j_s, j_k in T.Parallel(block_S, DK):
                                scaled_k_shared[j_s, j_k] = (
                                    k_shared[0, j_s, j_k]
                                    * g_exp_rev_shared[0, j_s]
                                )

                            T.warpgroup_wait(0)
                            T.barrier_arrive(q_is_free[u_slot_t])
                            T.barrier_arrive(k_is_free[u_slot_t])
                            g_last_local_t = g_exp_shared[0, valid_seqs_t - 1]
                            for j_k, j_v in T.Parallel(DK, DV):
                                h_fragment[j_k, j_v] *= g_last_local_t
                            T.barrier_arrive(gb_is_free[u_slot_t])

                            T.barrier_wait(inv_is_free, 0)
                            for j_s, j_t in T.Parallel(block_S, block_S):
                                pa_shared[j_s, j_t] = (
                                    g_fragment[j_s, j_t] * p_fragment[j_s, j_t] * scale
                                )
                            T.lma_wait()
                            T.barrier_arrive(p_is_ready)

                            if tx < hs_drain_tx_limit:
                                T.tma_store_wait()
                            T.barrier_wait(vd_is_ready, 0)
                            T.lma_wait()
                            T.sync_threads(0, 512)
                            T.gemm(
                                scaled_k_shared,
                                vd_shared,
                                h_fragment,
                                transpose_A=True,
                                policy=T.GemmWarpPolicy.FullCol,
                                clear_accum=False,
                                wg_wait=-1,
                            )
                            T.warpgroup_commit_batch()
                            T.warpgroup_wait(0)
                            T.barrier_arrive(vd_is_free)
                else:
                    for i_s in T.serial(num_iters):
                        valid_seqs = T.min(
                            seq_end_idx - seq_start_idx - i_s * block_S, block_S
                        )
                        T.barrier_wait(h_is_free[i_s % 2], producer_phase(i_s))
                        for j_k, j_v in T.Parallel(DK, DV):
                            h_shared[i_s % 2, j_k, j_v] = h_fragment[j_k, j_v]
                        T.lma_wait()
                        T.sync_threads(0, 512)
                        T.barrier_arrive(h_is_ready[i_s % 2])
                        T.tma_copy(
                            h_shared[i_s % 2, :, :],
                            hs[0 if chunk_padded else bb, bh, (chunk_base + i_s) if chunk_padded else i_s, 0:DK, 0:DV],
                        )

                        T.barrier_wait(q_is_ready[i_s % 2], consumer_phase(i_s))
                        T.barrier_wait(k_is_ready[i_s % 2], consumer_phase(i_s))
                        T.gemm(
                            q_shared[i_s % 2, :, :],
                            k_shared[i_s % 2, :, :],
                            p_fragment,
                            transpose_B=True,
                            clear_accum=True,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()

                        T.barrier_wait(gb_is_ready[i_s % 2], consumer_phase(i_s))
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            g_fragment[j_s, j_t] = (
                                g_shared[i_s % 2, j_s] - g_shared[i_s % 2, j_t]
                            )
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            g_fragment[j_s, j_t] = T.if_then_else(
                                g_fragment[j_s, j_t] <= 0.0,
                                g_fragment[j_s, j_t],
                                -float("inf"),
                            )
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            g_fragment[j_s, j_t] = T.exp2(
                                g_fragment[j_s, j_t] * 1.4426950408889634
                            )

                        T.barrier_wait(a_is_ready, i_s % 2)
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            pa_shared[j_s, j_t] = (
                                g_fragment[j_s, j_t]
                                * pa_shared[j_s, j_t]
                                * b_shared[i_s % 2, j_t]
                            )
                        T.lma_wait()
                        T.barrier_arrive(inv_is_ready)
                        for j_s, j_k in T.Parallel(block_S, DK):
                            scaled_k_shared[j_s, j_k] = (
                                k_shared[i_s % 2, j_s, j_k]
                                * g_exp_rev_shared[i_s % 2, j_s]
                            )

                        T.warpgroup_wait(0)
                        T.barrier_arrive(q_is_free[i_s % 2])
                        T.barrier_arrive(k_is_free[i_s % 2])
                        g_last_local = g_exp_shared[i_s % 2, valid_seqs - 1]
                        for j_k, j_v in T.Parallel(DK, DV):
                            h_fragment[j_k, j_v] *= g_last_local
                        T.barrier_arrive(gb_is_free[i_s % 2])

                        T.barrier_wait(inv_is_free, i_s % 2)
                        for j_s, j_t in T.Parallel(block_S, block_S):
                            pa_shared[j_s, j_t] = (
                                g_fragment[j_s, j_t] * p_fragment[j_s, j_t] * scale
                            )
                        T.lma_wait()
                        T.barrier_arrive(p_is_ready)

                        if tx < hs_drain_tx_limit:
                            T.tma_store_wait()
                        T.barrier_wait(vd_is_ready, i_s % 2)
                        T.lma_wait()
                        T.sync_threads(0, 512)
                        T.gemm(
                            scaled_k_shared,
                            vd_shared,
                            h_fragment,
                            transpose_A=True,
                            policy=T.GemmWarpPolicy.FullCol,
                            clear_accum=False,
                            wg_wait=-1,
                        )
                        T.warpgroup_commit_batch()
                        T.warpgroup_wait(0)
                        T.barrier_arrive(vd_is_free)
                if store_final_state:
                    for j_k, j_v in T.Parallel(DK, DV):
                        ht[bb, bh, inv_perm_dv(j_v), j_k] = h_fragment[j_k, j_v]

            with T.ws(6):
                if tx < 800:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(q_is_free[i_s % 2], producer_phase(i_s))
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        if tx == 768:
                            T.tma_copy(
                                q[batch_idx, load_left:load_right, bhg, 0:DK],
                                q_shared[i_s % 2, :, :],
                                barrier=q_is_ready[i_s % 2],
                            )
                        T.barrier_arrive(q_is_ready[i_s % 2])
                elif tx < 832:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(k_is_free[i_s % 2], producer_phase(i_s))
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        if tx == 800:
                            T.tma_copy(
                                k[batch_idx, load_left:load_right, bhg, 0:DK],
                                k_shared[i_s % 2, :, :],
                                barrier=k_is_ready[i_s % 2],
                            )
                        T.barrier_arrive(k_is_ready[i_s % 2])
                elif tx < 864:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(a_is_free, T.bitwise_xor(i_s % 2, 1))
                        load_left = ((chunk_base + i_s) * block_S if chunk_padded
                                     else seq_start_idx + i_s * block_S)
                        load_right = load_left + block_S
                        if tx == 832:
                            T.tma_copy(
                                a[batch_idx, load_left:load_right, bh, 0:block_S],
                                pa_shared,
                                barrier=a_is_ready,
                            )
                        T.barrier_arrive(a_is_ready)
                elif tx < 896:
                    for i_s in T.serial(num_iters):
                        load_left = seq_start_idx + i_s * block_S
                        load_right = load_left + block_S
                        T.barrier_wait(v_is_free, T.bitwise_xor(i_s % 2, 1))
                        if tx == 864:
                            T.tma_copy(
                                v[
                                    batch_idx,
                                    load_left:load_right,
                                    bh,
                                    :,
                                ],
                                v_shared,
                                barrier=v_is_ready,
                            )
                        T.barrier_arrive(v_is_ready)
            with T.ws(7):
                if tx < 960:
                    for i_s in T.serial(num_iters):
                        T.barrier_wait(gb_is_free[i_s % 2], producer_phase(i_s))
                        load_left = seq_start_idx + i_s * block_S
                        if chunk_padded:
                            pad_left = (chunk_base + i_s) * block_S
                            pad_right = pad_left + block_S
                        load_right = load_left + block_S
                        valid_seqs = T.min(
                            seq_end_idx - seq_start_idx - i_s * block_S, block_S
                        )
                        if tx == 896:
                            T.tma_copy(
                                b[bh, batch_idx,
                                  pad_left:pad_right] if chunk_padded else
                                b[bh, batch_idx, load_left:load_right],
                                b_shared[i_s % 2, :],
                                barrier=b_is_ready,
                            )
                        T.barrier_arrive(b_is_ready)
                        T.sync_threads(1, 64)
                        gb_lane = tx - 896
                        gb_lane_in_warp = gb_lane % 32
                        g_prefix = T.alloc_var(accum_dtype)
                        g_peer = T.alloc_var(accum_dtype)
                        g_prefix = T.if_then_else(
                            gb_lane < valid_seqs,
                            g[batch_idx, load_left + gb_lane, bh],
                            -float("inf"),
                        )
                        for j_shift in T.unroll(5):
                            j_offset = 1 << j_shift
                            g_peer = T.shfl_up(g_prefix, j_offset)
                            if gb_lane_in_warp >= j_offset:
                                g_prefix += g_peer
                        g_shared[i_s % 2, gb_lane] = g_prefix
                        T.lma_wait()
                        T.sync_threads(1, 64)
                        if gb_lane >= 32:
                            g_prefix += g_shared[i_s % 2, 31]
                            g_shared[i_s % 2, gb_lane] = g_prefix
                        T.lma_wait()
                        T.sync_threads(1, 64)
                        for j_s in T.Parallel(block_S):
                            g_exp_rev_shared[i_s % 2, j_s] = (
                                g_shared[i_s % 2, valid_seqs - 1]
                                - g_shared[i_s % 2, j_s]
                            )
                        for j_s in T.Parallel(block_S):
                            g_exp_rev_shared[i_s % 2, j_s] = T.if_then_else(
                                g_exp_rev_shared[i_s % 2, j_s] <= 0,
                                g_exp_rev_shared[i_s % 2, j_s],
                                float("-inf"),
                            )

                        for j_s in T.Parallel(block_S):
                            g_exp_rev_shared[i_s % 2, j_s] = T.exp2(
                                g_exp_rev_shared[i_s % 2, j_s] * 1.4426950408889634
                            )
                            g_exp_shared[i_s % 2, j_s] = T.exp2(
                                g_shared[i_s % 2, j_s] * 1.4426950408889634
                            )
                        T.lma_wait()
                        T.barrier_wait(b_is_ready, i_s % 2)
                        T.barrier_arrive(gb_is_ready[i_s % 2])
                        for j_s in T.Parallel(block_S):
                            g_prefix_out[
                                bh,
                                batch_idx,
                                (pad_left if chunk_padded else load_left) + j_s,
                            ] = T.if_then_else(
                                j_s < valid_seqs,
                                g_shared[i_s % 2, j_s],
                                g_shared[i_s % 2, valid_seqs - 1],
                            )

    def _symbol_part(value):
        return (
            str(value)
            .replace("torch.", "")
            .replace(".", "p")
            .replace("-", "m")
            .replace(" ", "_")
        )

    symbol = (
        "tilelang_fused_chunk_gdn_prefill_train_kernel"
        f"_h{H}_hg{Hg}_dk{DK}_dv{DV}_cs{chunk_size}"
        f"_s{_symbol_part(scale)}"
        f"_q{_symbol_part(qkva_dtype)}"
        f"_g{_symbol_part(g_dtype)}"
        f"_b{_symbol_part(b_dtype)}"
        f"_h0{_symbol_part(h0_dtype)}"
        f"_ht{_symbol_part(ht_dtype)}"
        f"_o{_symbol_part(o_dtype)}"
        f"_seq{_symbol_part(seqlen_dtype)}"
        f"_init{int(use_initial_state)}_final{int(store_final_state)}"
        "_var1"
        "_log1"
        "_vmp0"
        f"_u2{int(unroll_chunk_loop)}"
        "_logp1"
        + ("_cp1" if chunk_padded else "")
    )
    return tilelang_fused_chunk_gdn_prefill_train_kernel.with_attr("global_symbol", symbol)


def fused_chunk_gdn_prefill_train(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor | None = None,
    output_state: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    output_o: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunks_per_sequence: int | None = None,
    pad_rows: int = 0,
    chunk_padded: bool = False,
    total_chunks: int | None = None,
):
    _data_batch, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    scale = scale or K ** (-0.5)
    if not head_dim_supported(K, V, chunk_size):
        raise ValueError(gdn_head_dim_error("gdn_prefill", K, V, chunk_size))
    assert chunk_size == 64
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1

    if cu_seqlens is None:
        raise ValueError(
            "the prefill ABI is varlen: pass cu_seqlens. A dense batch enters "
            "as one packed sequence per row -- see gdn_chunk_fwd, which builds "
            "the equal-length vector for it.")
    real_batch_size = (len(cu_seqlens) // 2 - 1 if chunk_padded
                       else len(cu_seqlens) - 1)
    seqlen_dtype = cu_seqlens.dtype

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty(
            (real_batch_size, H, K, V), dtype=torch.float32, device=k.device
        )
    if output is None:
        o = torch.empty_like(v)
    else:
        if output.shape != v.shape:
            raise ValueError(f"Expected output shape {v.shape}, got {output.shape}.")
        if output.dtype != v.dtype:
            raise ValueError(f"Expected output dtype {v.dtype}, got {output.dtype}.")
        if output.device != v.device:
            raise ValueError(f"Expected output device {v.device}, got {output.device}.")
        o = output

    final_state_shape = (real_batch_size, H, K, V)
    if output_state is None:
        final_state = torch.empty(
            final_state_shape, dtype=torch.float32, device=k.device
        )
    else:
        if output_state.shape != final_state_shape:
            raise ValueError(
                f"Expected output_state shape {final_state_shape}, got {output_state.shape}."
            )
        if output_state.dtype != torch.float32:
            raise ValueError(
                f"Expected output_state dtype torch.float32, got {output_state.dtype}."
            )
        if output_state.device != k.device:
            raise ValueError(
                f"Expected output_state device {k.device}, got {output_state.device}."
            )
        final_state = output_state

    if chunks_per_sequence is None:
        sequence_length = num_tokens // real_batch_size
        chunks_per_sequence = (sequence_length + chunk_size - 1) // chunk_size
    if chunk_padded:
        if total_chunks is None:
            raise ValueError("chunk_padded requires total_chunks")
        hs = torch.empty((1, H, total_chunks, K, V),
                         dtype=q.dtype, device=q.device)
    else:
        hs = torch.empty(
            (real_batch_size, H, chunks_per_sequence, K, V),
            dtype=q.dtype,
            device=q.device,
        )
    unroll_chunk_loop = (
        chunks_per_sequence >= UNROLL_MIN_CHUNKS
        and real_batch_size * H <= UNROLL_MAX_CTAS
    )
    delta = torch.empty(
        (v.shape[0], num_tokens + pad_rows) + tuple(v.shape[2:]),
        dtype=v.dtype, device=v.device,
    )
    g_prefix_out = torch.empty(
        (g.shape[2], g.shape[0], g.shape[1] + pad_rows),
        dtype=torch.float32, device=g.device,
    )

    tilelang_fused_chunk_gdn_prefill_train_kernel = tilelang_fused_chunk_gdn_prefill_train(
        H,
        Hg,
        K,
        V,
        chunk_size,
        scale,
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=b.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        unroll_chunk_loop=unroll_chunk_loop,
        padded_store=pad_rows > 0,
        chunk_padded=chunk_padded,
    )
    tilelang_fused_chunk_gdn_prefill_train_kernel(
        q,
        k,
        v,
        a,
        g,
        b,
        initial_state,
        cu_seqlens,
        o,
        final_state,
        hs,
        delta,
        g_prefix_out,
    )

    if not output_final_state:
        final_state = None
    if not output_o:
        o = None

    return o, final_state, hs, delta, g_prefix_out


def gdn_chunk_fwd(q, k, v, g, beta, h0=None, scale=None, chunk_size=64,
                  lengths=None,
                  need_final_state=True, cu_seqlens=None):
    """Run the forward kernels and return the backward handoff.

    Args:
        q, k, v: [B, S, H, 128] BF16, q/k already L2-normalized.
        g: [B, S, H] FP32 log-space decay; beta: [B, S, H] BF16. All at their TRUE
            length -- nothing here is padded.
        need_final_state: skip the state transpose (a launch and a 4 MiB round trip)
            when the caller drops ``ht``.

    Returns:
        ``(o, ht, A, Hs, Delta, GPrefix, beta_hbs)``. ``o`` keeps the true length;
        every handoff is reported at the chunk-aligned length because the backward
        walks whole 64-row chunks. ``A`` holds the symmetric KKT values in the strict
        upper triangle and a decay value on the diagonal; ``GPrefix`` is cumulative
        LOG-decay; ``GPrefix``/``beta_hbs`` are head-outermost [H, B, S_pad].
    """
    B, S, H, D = q.shape
    if not head_dim_supported(D, v.shape[-1], chunk_size):
        raise ValueError(
            gdn_head_dim_error("kernels", D, v.shape[-1], chunk_size))
    assert chunk_size == 64, "kernels require chunk_size=64"
    maps = None
    if cu_seqlens is not None:
        assert B == 1, "packed varlen requires B == 1"
        maps = varlen_maps(cu_seqlens, q.device, lengths=lengths)
        if maps.num_tokens != S:
            raise ValueError(
                f"cu_seqlens covers {maps.num_tokens} tokens, q has {S}")
        NT = maps.NT
    else:
        NT = -(-S // chunk_size)
    S_pad = NT * chunk_size
    pad_rows = S_pad - S
    if pad_rows and maps is None:
        if B != 1:
            raise ValueError("a non-chunk-aligned S requires B == 1")
        if need_final_state:
            raise ValueError(
                "a partial last chunk cannot produce the final state; pad S to a "
                "multiple of the chunk or pass need_final_state=False")
    if scale is None:
        scale = D ** -0.5

    total_tokens = B * S
    cu = (maps.cu2 if maps is not None else dense_cu(B, S, q.device))
    cu_kkt = maps.cu2[:maps.smap.shape[0]] if maps is not None else cu
    qf = _tl_contiguous(q.reshape(1, total_tokens, H, D)).to(v.dtype)
    kf = _tl_contiguous(k.reshape(1, total_tokens, H, D)).to(v.dtype)
    vf = _tl_contiguous(v.reshape(1, total_tokens, H, D))
    gf = _tl_contiguous(g.reshape(1, total_tokens, H).float())
    beta_bf = _tl_contiguous(beta.reshape(1, total_tokens, H))

    initial_state = None
    if h0 is not None:
        initial_state = h0.transpose(-1, -2).contiguous().float()

    A, beta_hbs = kkt_solve_train(
        kf, beta_bf, cu_kkt, None if maps is not None else NT,
        chunk_size=chunk_size, pad_rows=pad_rows,
        chunk_padded=maps is not None,
    )
    o, final_state, Hs, Delta, GPrefix = fused_chunk_gdn_prefill_train(
        q=qf,
        k=kf,
        v=vf,
        a=A,
        g=gf,
        b=beta_hbs,
        cu_seqlens=cu,
        chunks_per_sequence=None if maps is not None else NT,
        scale=scale,
        initial_state=initial_state,
        chunk_size=chunk_size,
        pad_rows=pad_rows,
        chunk_padded=maps is not None,
        total_chunks=NT if maps is not None else None,
    )

    return (
        o.reshape(B, S, H, D),
        final_state.transpose(-1, -2).contiguous() if need_final_state else None,
        A.reshape(B, S_pad, H, chunk_size),
        Hs,
        Delta.reshape(B, S_pad, H, D),
        GPrefix.reshape(H, B, S_pad),
        beta_hbs.view(H, B, S_pad),
    )
