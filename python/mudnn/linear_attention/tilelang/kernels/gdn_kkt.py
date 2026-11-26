"""Per-chunk KKT solve for the GDN forward.

Solves ``(I - tril(beta * K K^T, -1))^-1`` per 64-token chunk and republishes
the betas it already reads as the FP32 ``[H, B, tokens]`` handoff the prefill
kernel and the backward both consume.
"""

import torch
import tilelang
import tilelang.language as T

from ._jit import KKT_COMPILE_FLAGS, KKT_PASS_CONFIGS

from .gdn_shapes import gdn_head_dim_error, head_dim_supported

__all__ = ["kkt_solve_train"]


@tilelang.jit(pass_configs=KKT_PASS_CONFIGS, compile_flags=KKT_COMPILE_FLAGS)
def tilelang_kkt_solve_train(
    H,
    Hg,
    DK,
    chunk_size,
    accum_dtype,
    qkva_dtype,
    b_dtype,
    seqlen_dtype,
    padded_store=False,
    chunk_padded=False,
):
    padded_store = padded_store or chunk_padded
    data_batch_size = T.dynamic("data_batch_size")
    real_batch_size = T.dynamic("real_batch_size")
    num_tokens = T.dynamic("num_tokens")
    block_S = chunk_size

    a_tokens = T.dynamic("a_tokens") if padded_store else num_tokens
    k_shape = (data_batch_size, num_tokens, Hg, DK)
    a_shape = (data_batch_size, a_tokens, H, chunk_size)
    b_shape = (data_batch_size, num_tokens, H)
    bft_shape = (H, data_batch_size, a_tokens)
    heads_per_k = H // Hg

    k_stride_b = T.dynamic("k_stride_b")
    k_stride_t = T.dynamic("k_stride_t")
    k_stride_h = T.dynamic("k_stride_h")
    k_strides = (k_stride_b, k_stride_t, k_stride_h, 1)

    @T.macro
    def kernel_body(
        chunk_ord,
        bh,
        batch_idx,
        chunk_idx,
        body_iter,
        seq_start_idx,
        seq_end_idx,
        k,
        b,
        a,
        bft,
    ):
        left = seq_start_idx + chunk_idx * block_S
        right = left + block_S
        store_left = chunk_ord * block_S if chunk_padded else left
        data_batch_idx = 0
        bhg = bh // heads_per_k
        barrier_phase = body_iter % 2

        k_shared = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
        b_shared = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
        a64_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
        valid_seqs = T.min(seq_end_idx - left, block_S)
        store_rows = (
            block_S if chunk_padded
            else (T.if_then_else(seq_end_idx == num_tokens, block_S, valid_seqs)
                  if padded_store else valid_seqs))

        a32d_shared = T.alloc_shared((2, 32, 36), dtype=accum_dtype)
        a32i0_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32i1_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32o1_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32o2_shared = T.alloc_shared((32, 32), dtype=qkva_dtype)
        a32o_fragment = T.alloc_fragment((32, 32), dtype=accum_dtype)

        T.annotate_layout(
            {a32d_shared: tilelang.layout.make_linear_layout(a32d_shared)}
        )

        k_load_is_ready = T.alloc_barrier(arrive_count=128)
        diag_input_is_ready = T.alloc_barrier(arrive_count=128)
        diag_inverse_is_ready = T.alloc_barrier(arrive_count=128)
        tx = T.get_thread_binding()
        T.tma_copy(
            k[data_batch_idx, left:right, bhg, 0:DK],
            k_shared,
            barrier=k_load_is_ready,
        )
        T.barrier_arrive(k_load_is_ready)
        T.barrier_wait(k_load_is_ready, barrier_phase)
        T.gemm(k_shared, k_shared, a64_fragment, transpose_B=True, clear_accum=True)
        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s < j_t:
                a64_fragment[j_s, j_t] = 0.0

        if right <= seq_end_idx:
            for j_s in T.Parallel(block_S):
                b_shared[j_s] = b[data_batch_idx, left + j_s, bh]
        else:
            for j_s in T.Parallel(block_S):
                if left + j_s < seq_end_idx:
                    b_shared[j_s] = b[data_batch_idx, left + j_s, bh]
                else:
                    b_shared[j_s] = 0
        T.sync_threads()
        for j_s in T.Parallel(block_S):
            if j_s < store_rows:
                bft[bh, data_batch_idx, store_left + j_s] = b_shared[j_s]
        for j_s, j_t in T.Parallel(block_S, block_S):
            if (j_s // 32) == (j_t // 32):
                a32d_shared[j_s // 32, j_s % 32, j_t % 32] = T.if_then_else(
                    j_s == j_t,
                    1.0,
                    -a64_fragment[j_s, j_t] * b_shared[j_s],
                )
        T.barrier_arrive(diag_input_is_ready)
        T.barrier_wait(diag_input_is_ready, barrier_phase)

        if tx < 64:
            diag_block = tx // 32
            diag_row_idx = tx % 32
            diag_row = T.alloc_local([32], accum_dtype)

            for k_t in T.vectorized(32):
                diag_row[k_t] = a32d_shared[diag_block, diag_row_idx, k_t]

            for src_row in T.unroll(31):
                row_scale = T.if_then_else(
                    diag_row_idx > src_row, diag_row[src_row], 0.0
                )
                for k_t in T.unroll(src_row):
                    src_row_value = T.shfl_sync(0xFFFFFFFF, diag_row[k_t], src_row, 32)
                    diag_row[k_t] += row_scale * src_row_value

            if diag_block == 0:
                for k_t in T.vectorized(32):
                    a32i0_shared[diag_row_idx, k_t] = diag_row[k_t]
            else:
                for k_t in T.vectorized(32):
                    a32i1_shared[diag_row_idx, k_t] = diag_row[k_t]

        T.barrier_arrive(diag_inverse_is_ready)
        T.barrier_wait(diag_inverse_is_ready, barrier_phase)

        for j_s, j_t in T.Parallel(block_S, block_S):
            if j_s >= 32 and j_t < 32:
                a32o1_shared[j_s - 32, j_t] = -a64_fragment[j_s, j_t] * b_shared[j_s]
        T.sync_threads()

        T.gemm(a32i1_shared, a32o1_shared, a32o_fragment, clear_accum=True, wg_wait=-1)
        T.warpgroup_commit_batch()
        T.warpgroup_wait(0)
        T.copy(a32o_fragment, a32o2_shared)
        T.sync_threads()
        T.gemm(a32o2_shared, a32i0_shared, a32o_fragment, clear_accum=True, wg_wait=-1)
        T.warpgroup_commit_batch()
        for k_s, k_t in T.Parallel(32, 32):
            if k_s < k_t:
                a32i1_shared[k_s, k_t] = a64_fragment[32 + k_t, 32 + k_s]
        for k_s, k_t in T.Parallel(32, 32):
            a32o1_shared[k_s, k_t] = a64_fragment[32 + k_t, k_s]
        T.sync_threads()
        for k_s, k_t in T.Parallel(32, 32):
            if 32 + k_s < store_rows:
                a[data_batch_idx, store_left + 32 + k_s, bh, 32 + k_t] = (
                    T.if_then_else(
                        32 + k_s < valid_seqs,
                        a32i1_shared[k_s, k_t],
                        T.if_then_else(k_s == k_t, 1.0, 0.0).astype(qkva_dtype),
                    )
                    if padded_store else a32i1_shared[k_s, k_t]
                )
        for k_s, k_t in T.Parallel(32, 32):
            if k_s < store_rows:
                a[data_batch_idx, store_left + k_s, bh, 32 + k_t] = (
                    T.if_then_else(
                        k_s < valid_seqs,
                        a32o1_shared[k_s, k_t],
                        T.Cast(qkva_dtype, 0.0),
                    )
                    if padded_store else a32o1_shared[k_s, k_t]
                )
        T.warpgroup_wait(0)
        for k_s, k_t in T.Parallel(32, 32):
            if k_s < k_t:
                a32i0_shared[k_s, k_t] = a64_fragment[k_t, k_s]
        for k_s, k_t in T.Parallel(32, 32):
            a32i1_shared[k_s, k_t] = a32o_fragment[k_s, k_t]
        T.sync_threads()
        for k_s, k_t in T.Parallel(32, 32):
            if k_s < store_rows:
                a[data_batch_idx, store_left + k_s, bh, k_t] = (
                    T.if_then_else(
                        k_s < valid_seqs,
                        a32i0_shared[k_s, k_t],
                        T.if_then_else(k_s == k_t, 1.0, 0.0).astype(qkva_dtype),
                    )
                    if padded_store else a32i0_shared[k_s, k_t]
                )
        for k_s, k_t in T.Parallel(32, 32):
            if 32 + k_s < store_rows:
                a[data_batch_idx, store_left + 32 + k_s, bh, k_t] = (
                    T.if_then_else(
                        32 + k_s < valid_seqs,
                        a32i1_shared[k_s, k_t],
                        T.Cast(qkva_dtype, 0.0),
                    )
                    if padded_store else a32i1_shared[k_s, k_t]
                )

    if chunk_padded:
        @T.prim_func
        def tilelang_kkt_ragged_kernel(
            k: T.StridedTensor(k_shape, k_strides, qkva_dtype),
            b: T.Tensor(b_shape, dtype=b_dtype),
            cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
            a: T.Tensor(a_shape, dtype=qkva_dtype),
            bft: T.Tensor(bft_shape, dtype=accum_dtype),
        ):
            launch_num_chunks = T.ceildiv(num_tokens, block_S) + real_batch_size
            with T.Kernel(launch_num_chunks * H, threads=128) as (bch,):
                bc, bh = bch // H, bch % H

                seq_idx = T.alloc_var("int32")
                acc_chunks = T.alloc_var("int32")
                seq_chunks = T.alloc_var("int32")
                seq_start_idx = T.alloc_var("int32")
                seq_end_idx = T.alloc_var("int32")
                chunk_ord = T.alloc_var("int32")

                chunk_ord = bc
                seq_idx = 0
                acc_chunks = 0
                seq_start_idx = cu_seqlens[0]
                seq_end_idx = cu_seqlens[1]
                seq_chunks = T.ceildiv(seq_end_idx - seq_start_idx, block_S)
                while seq_idx < real_batch_size and acc_chunks + seq_chunks <= bc:
                    acc_chunks += seq_chunks
                    seq_idx += 1
                    seq_start_idx = seq_end_idx
                    seq_end_idx = cu_seqlens[T.min(seq_idx + 1, real_batch_size)]
                    seq_chunks = T.ceildiv(seq_end_idx - seq_start_idx, block_S)

                if seq_idx >= real_batch_size:
                    seq_idx = 0
                    acc_chunks = bc
                    chunk_ord = 0
                    seq_start_idx = cu_seqlens[0]
                    seq_end_idx = cu_seqlens[1]

                kernel_body(chunk_ord, bh, seq_idx, bc - acc_chunks, 0,
                            seq_start_idx, seq_end_idx, k, b, a, bft)

        return tilelang_kkt_ragged_kernel

    @T.prim_func
    def tilelang_kkt_solve_kernel(
        k: T.StridedTensor(k_shape, k_strides, qkva_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        cu_seqlens: T.Tensor([real_batch_size + 1], dtype=seqlen_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        bft: T.Tensor(bft_shape, dtype=accum_dtype),
        chunks_per_sequence: T.int32,
    ):
        launch_num_chunks = T.ceildiv(num_tokens, block_S) + real_batch_size
        with T.Kernel(launch_num_chunks * H, threads=128) as (bch,):
            bc, bh = bch // H, bch % H

            seq_idx = T.alloc_var("int32")
            acc_chunks = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")

            seq_idx = bc // chunks_per_sequence
            acc_chunks = seq_idx * chunks_per_sequence
            seq_start_idx = cu_seqlens[T.min(seq_idx, real_batch_size)]
            seq_end_idx = cu_seqlens[T.min(seq_idx + 1, real_batch_size)]

            # Surplus CTAs re-derive chunk 0 and republish identical bytes (0.0 us).
            # Retiring them instead silently drops 9 of the 13 CTA barriers.
            if seq_idx >= real_batch_size:
                seq_idx = 0
                acc_chunks = bc
                seq_start_idx = cu_seqlens[0]
                seq_end_idx = cu_seqlens[1]

            kernel_body(
                bc,
                bh,
                seq_idx,
                bc - acc_chunks,
                0,
                seq_start_idx,
                seq_end_idx,
                k,
                b,
                a,
                bft,
            )

    return tilelang_kkt_solve_kernel


def kkt_solve_train(
    k: torch.Tensor,
    b: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunks_per_sequence: int | None,
    chunk_size: int = 64,
    pad_rows: int = 0,
    chunk_padded: bool = False,
):
    """Solve the per-chunk KKT system and republish beta head-outermost.

    Args:
        k: [data_batch, tokens, Hg, 128] over the flattened token axis.
        b: raw [data_batch, tokens, H] beta activation.
        chunks_per_sequence: chunks in every packed sequence, or None together with
            ``chunk_padded`` for a ragged pack. A ragged pack with a fixed count
            silently maps CTAs to the wrong chunks.

    Returns:
        ``a`` [data_batch, tokens + pad_rows, H, chunk_size] and ``beta_hbs``
        [H, data_batch, tokens + pad_rows] FP32. ``pad_rows`` over-allocates both so
        a non-chunk-aligned ``S`` still yields whole ``chunk_size`` tiles; the kernel
        fills those rows with the identity for ``a`` and 0 for ``beta_hbs``.
    """
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H = b.shape
    if not head_dim_supported(K, K, chunk_size):
        raise ValueError(gdn_head_dim_error("KKT solve", K, K, chunk_size))
    assert chunk_size == 64
    assert k.stride(-1) == 1

    seqlen_dtype = cu_seqlens.dtype
    if chunk_padded:
        if chunks_per_sequence is not None:
            raise ValueError("chunk_padded requires the ragged chunk map; do "
                             "not pass chunks_per_sequence with it")
    else:
        chunks_per_sequence = int(chunks_per_sequence)
        if chunks_per_sequence < 1:
            raise ValueError(f"chunks_per_sequence must be >= 1, got {chunks_per_sequence}")
        if (cu_seqlens.numel() - 1) * chunks_per_sequence * chunk_size < num_tokens:
            raise ValueError(
                f"the uniform chunk map holds {(cu_seqlens.numel() - 1) * chunks_per_sequence}"
                f" chunks, too few for {num_tokens} tokens")

    a = torch.empty(
        (batch_size, num_tokens + pad_rows, H, chunk_size),
        dtype=k.dtype, device=k.device,
    )
    beta_hbs = torch.empty(
        (H, batch_size, num_tokens + pad_rows),
        dtype=torch.float32, device=k.device,
    )

    tilelang_kkt_solve_kernel = tilelang_kkt_solve_train(
        H,
        Hg,
        K,
        chunk_size,
        qkva_dtype=k.dtype,
        b_dtype=b.dtype,
        seqlen_dtype=seqlen_dtype,
        accum_dtype="float32",
        padded_store=pad_rows > 0,
        chunk_padded=chunk_padded,
    )
    if chunk_padded:
        tilelang_kkt_solve_kernel(k, b, cu_seqlens, a, beta_hbs)
    else:
        tilelang_kkt_solve_kernel(k, b, cu_seqlens, a, beta_hbs, chunks_per_sequence)

    return a, beta_hbs
