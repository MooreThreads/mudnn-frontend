"""Dense (full-KV) indexer backward -- architecture-generic TileLang MUSA path.

The generic variant of this family; :mod:`dense_indexer_backward_mp32` is the
mp_32-tuned one, and :class:`~.api.DenseIndexerBackward` selects between them from
the device capability. The public entry point is always
``dense_indexer_backward``.

Semantics (BSHD; batch ``b``, query row ``q``, indexer head ``h``, KV index
``k``; the dense path visits every KV position, so there is no top-K index):

.. code-block:: text

    S[q, h, k]     = Q[b, q, h, :] . K[b, k, :]           (fp32 accumulator)
    R[q, h, k]     = relu(sm_scale * S[q, h, k])
    dW[q, h]       = sum_k grad_signal[q, k] * R[q, h, k]
    dS[q, h, k]    = (S[q, h, k] > 0) ? grad_signal[q, k] * w[q, h] : 0
    dQ[b, q, h, :] = sum_k sm_scale * dS[q, h, k] * K[b, k, :]
    dK[b, k, :]   += sum_q sum_h sm_scale * dS[q, h, k] * Q[b, q, h, :]

``grad_signal`` is the clipped-log KL score gradient produced by the score-grad
stage, exactly as in ``cudnn-frontend``'s dense backward:

.. code-block:: text

    lse     = index_lse[b, q]
    denom   = attn_l1norm[b, q] + 1e-10
    sml     = index_score_raw[b, q, k] - lse
    predict = exp2(sml * log2(e))
    target  = attn_score_raw[b, q, k] / denom
    g       = -max(target, exp(-100)) * (sml >= -100) * grad_scale * grad_loss
    grad_signal[q, k] = g - predict * sum_k g           (0 outside the mask)

The ``ratio`` causal bound is ``col_limit = clamp((q + 1) // ratio, 0, S_k)``;
positions at or beyond it hold ``grad_signal = 0`` and contribute nothing to
``dQ`` / ``dK`` / ``dW``.

Call signature of the returned runner
------------------------------------
::

    (IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK,
     AttnScore, AttnL1Norm, IdxScoreRaw, IdxLSE, GradLoss, grad_scale,
     CuSeqlensQ=None, CuSeqlensK=None, QCausalOffsets=None,
     current_stream=None, sm_scale=None)

Tensor contract
---------------
``IndexQ`` / ``Weights`` / ``IndexK`` and the ``dIndexQ`` / ``dWeights`` outputs
are bf16 (or fp16), all sharing one dtype.

``AttnScore`` and ``IdxScoreRaw`` are the **raw fp32 dense scores** with
``AttnL1Norm`` / ``IdxLSE`` their per-row denominators. ``IdxScoreRaw`` is
**overwritten in place** with ``grad_signal``, as upstream; ``AttnScore`` stays
read-only.

``GradLoss`` may be a single-element fp32 device tensor, a Python float, or
``None`` (treated as ``1.0``). ``grad_scale`` is the host scalar
``loss_coeff / normalization_tokens``.

``dIndexK`` must be an **fp32** buffer: the kernel atomic-accumulates fp32 and
leaves the cast to the caller, and the runner zeroes it first.

``sm_scale`` is the one non-upstream addition to the signature: ``None`` uses the
value the runner was built with, a float overrides it for that launch only. It is
a runtime operand, never a compile key, and must be positive because it scales
the ReLU argument.

Layout: BSHD only
-----------------
This variant implements BSHD ``(B, S_q, ...)`` tensors. THD / packed tensors
(``cu_seqlens_*``) and ``q_causal_offsets`` raise ``NotImplementedError`` rather
than being silently ignored; the mp_32 variant implements both.

Compile key and limits
----------------------
One kernel per ``(heads, dim, block_I, num_stages, ratio)``; ``batch``, ``S_q``,
``S_k`` and ``sm_scale`` are runtime.

* ``heads`` a multiple of 16 and ``dim`` a multiple of 8 (SQMMA tiles).
* ``block_I`` must divide ``max_seqlen_k``: the K tiles are issued as whole
  unpredicated loads.
* ``ratio >= 1`` and ``sm_scale > 0``.
* Dynamic shared memory is capped at 192 KB here (the mp_32 variant has 256 KB),
  which is why ``block_I`` defaults to 64.
* ``num_stages`` and ``q_rows`` are coupled by shared memory; the shipped pair is
  ``num_stages=1, q_rows=2``, with ``q_rows=1`` going with ``num_stages=3``.

Accuracy
--------
bf16-precision against an fp32 reference rather than bit-exact.
"""
import math

try:
    import torch
except ImportError:  # pragma: no cover - `import mudnn` must work without torch
    torch = None

try:
    import tilelang
    import tilelang.language as T
except ImportError:  # pragma: no cover - reported when a kernel is built
    tilelang = None
    T = None

__all__ = ["dense_indexer_backward"]

_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16 if torch is not None else None,
    "float16": torch.float16 if torch is not None else None,
}

def _causal_col_limit(q_index, ratio, seqlen_k):
    return T.min((q_index + 1) // ratio, seqlen_k)

CLIP_LOG_MIN = -100.0
CLIP_PROB_MIN = math.exp(CLIP_LOG_MIN)
DENOM_EPS = 1e-10
LOG2E = 1.4426950408889634

_MAX_DYNAMIC_SHARED_MEMORY = 196608

_SMEM_OVERHEAD = 8192

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2}

THREADS = 512
PRODUCER_THREADS = THREADS // 2
CONSUMER_THREADS = THREADS - PRODUCER_THREADS
PRODUCER_TID_RANGE = "tid < PRODUCER_THREADS"

_GEMM_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
} if tilelang is not None else {}

_SCORE_GRAD_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
} if tilelang is not None else {}

def _require_tilelang():
    if tilelang is None or T is None:
        raise NotImplementedError(
            "dense_indexer_backward requires tilelang_musa: install the "
            "tilelang_musa wheel for this checkout before building the kernel."
        )
    return tilelang

def _dyn(*names):
    return tuple(T.dynamic(n) for n in names)

_RUNTIME_AXES = ("batch", "seqlen_q", "seqlen_k")

def _require_torch():
    if torch is None:
        raise NotImplementedError("dense_indexer_backward requires torch")
    return torch

def _score_grad_kernel(chunk, ratio, threads):
    tl = _require_tilelang()

    @tl.jit(target="musa", pass_configs=_SCORE_GRAD_PASS_CONFIGS)
    def _jit_kernel(chunk, ratio, threads):
        batch, seqlen_q, seqlen_k = _dyn(*_RUNTIME_AXES)
        a0, a1 = _dyn("a0", "a1")
        i0, i1 = _dyn("i0", "i1")
        (n0,) = _dyn("n0")
        (e0,) = _dyn("e0")

        @T.prim_func
        def kernel(
            AttnScore: T.StridedTensor(
                (batch, seqlen_q, seqlen_k), strides=(a0, a1, 1),
                dtype="float32"),
            AttnL1Norm: T.StridedTensor(
                (batch, seqlen_q), strides=(n0, 1), dtype="float32"),
            IdxScoreRaw: T.StridedTensor(
                (batch, seqlen_q, seqlen_k), strides=(i0, i1, 1),
                dtype="float32"),
            IdxLSE: T.StridedTensor(
                (batch, seqlen_q), strides=(e0, 1), dtype="float32"),
            GradLoss: T.Tensor((1,), "float32"),
            Scale: T.Tensor((1,), "float32"),
        ):
            with T.Kernel(seqlen_q, batch, threads=threads) as (bq, bb):
                g_part = T.alloc_fragment([1, chunk], "float32")
                sum_f = T.alloc_fragment([1], "float32")
                s_meta = T.alloc_shared([3], "float32")
                s_sum = T.alloc_shared([1], "float32")

                tid = T.get_thread_binding()
                if tid == 0:
                    s_meta[0] = IdxLSE[bb, bq]
                    s_meta[1] = AttnL1Norm[bb, bq]
                    s_meta[2] = Scale[0] * GradLoss[0]
                T.sync_threads()

                T.clear(g_part)
                for ko in T.serial((seqlen_k + chunk - 1) // chunk):
                    for _r, kk in T.Parallel(1, chunk):
                        k_idx = ko * chunk + kk
                        if k_idx < _causal_col_limit(
                            bq, ratio, seqlen_k
                        ):
                            sml = IdxScoreRaw[bb, bq, k_idx] - s_meta[0]
                            target = AttnScore[bb, bq, k_idx] / (
                                s_meta[1] + DENOM_EPS
                            )
                            target_eff = T.max(target, CLIP_PROB_MIN)
                            mask = T.if_then_else(sml >= CLIP_LOG_MIN, 1.0, 0.0)
                            g_part[0, kk] += -target_eff * mask * s_meta[2]

                T.reduce_sum(g_part, sum_f, dim=1)
                T.copy(sum_f, s_sum)

                for ko in T.serial((seqlen_k + chunk - 1) // chunk):
                    for _r, kk in T.Parallel(1, chunk):
                        k_idx = ko * chunk + kk
                        if k_idx < seqlen_k:
                            if k_idx < _causal_col_limit(
                                bq, ratio, seqlen_k
                            ):
                                sml = IdxScoreRaw[bb, bq, k_idx] - s_meta[0]
                                predict = T.exp2(sml * LOG2E)
                                target = AttnScore[bb, bq, k_idx] / (
                                    s_meta[1] + DENOM_EPS
                                )
                                target_eff = T.max(target, CLIP_PROB_MIN)
                                mask = T.if_then_else(
                                    sml >= CLIP_LOG_MIN, 1.0, 0.0
                                )
                                IdxScoreRaw[bb, bq, k_idx] = (
                                    -target_eff * mask * s_meta[2]
                                    - predict * s_sum[0]
                                )
                            else:
                                IdxScoreRaw[bb, bq, k_idx] = 0.0

        return kernel

    return _jit_kernel(chunk, ratio, threads)

def _dense_indexer_backward_gemm(
    heads,
    dim,
    block_I,
    num_stages,
    ratio,
    dtype,
    accum_dtype,
    q_rows=1,
):
    tl = _require_tilelang()

    M = int(q_rows) * int(heads)

    @tl.jit(target="musa", pass_configs=_GEMM_PASS_CONFIGS)
    def _jit_kernel(
        heads, dim, block_I, num_stages, ratio, dtype, accum_dtype, q_rows,
    ):
        batch, seqlen_q, seqlen_k = _dyn(*_RUNTIME_AXES)
        q0, q1, q2 = _dyn("q0", "q1", "q2")
        k0, k1 = _dyn("k0", "k1")
        w0, w1 = _dyn("w0", "w1")
        g0, g1 = _dyn("g0", "g1")
        dq0, dq1, dq2 = _dyn("dq0", "dq1", "dq2")
        dw0, dw1 = _dyn("dw0", "dw1")
        dk0, dk1 = _dyn("dk0", "dk1")

        @T.prim_func
        def kernel(
            Q: T.StridedTensor((batch, seqlen_q, heads, dim),
                               strides=(q0, q1, q2, 1), dtype=dtype),
            K: T.StridedTensor((batch, seqlen_k, dim),
                               strides=(k0, k1, 1), dtype=dtype),
            W: T.StridedTensor((batch, seqlen_q, heads),
                               strides=(w0, w1, 1), dtype=dtype),
            GS: T.StridedTensor((batch, seqlen_q, seqlen_k),
                                strides=(g0, g1, 1), dtype=accum_dtype),
            dQ: T.StridedTensor((batch, seqlen_q, heads, dim),
                                strides=(dq0, dq1, dq2, 1), dtype=dtype),
            dW: T.StridedTensor((batch, seqlen_q, heads),
                                strides=(dw0, dw1, 1), dtype=dtype),
            dK: T.StridedTensor((batch, seqlen_k, dim),
                                strides=(dk0, dk1, 1), dtype=accum_dtype),
            sm_scale: T.float32,
        ):
            q_blocks = (seqlen_q + q_rows - 1) // q_rows
            with T.Kernel(q_blocks, batch, threads=THREADS) as (bq, bb):
                q_base = bq * q_rows
                Q_s = T.alloc_shared([M, dim], dtype)
                if q_rows == 1:
                    Q_b = T.alloc_shared([heads, dim], dtype)
                else:
                    Q_b0 = T.alloc_shared([heads, dim], dtype)
                    Q_b1 = T.alloc_shared([heads, dim], dtype)
                W_shared = T.alloc_shared([M], dtype)
                K_s = T.alloc_shared([num_stages, block_I, dim], dtype)
                K_b = T.alloc_shared([num_stages, block_I, dim], dtype)
                GS_shared = T.alloc_shared(
                    [num_stages, q_rows, block_I], accum_dtype
                )
                dS_q = T.alloc_shared([M, block_I], dtype)
                if q_rows == 1:
                    dS_k = T.alloc_shared([heads, block_I], dtype)
                else:
                    dS_k0 = T.alloc_shared([heads, block_I], dtype)
                    dS_k1 = T.alloc_shared([heads, block_I], dtype)
                q_ready = T.alloc_barrier(PRODUCER_THREADS)
                w_ready = T.alloc_barrier(PRODUCER_THREADS)
                ds_ready = T.alloc_barrier(CONSUMER_THREADS)
                ds_free = T.alloc_barrier(PRODUCER_THREADS)
                ds_written = T.alloc_barrier(CONSUMER_THREADS)
                k_ready = T.alloc_barrier([PRODUCER_THREADS] * num_stages)
                gs_ready = T.alloc_barrier([PRODUCER_THREADS] * num_stages)
                k_free = T.alloc_barrier([CONSUMER_THREADS] * num_stages)

                tid = T.get_thread_binding()

                T.sync_threads()

                q_last = T.min(q_base + q_rows, seqlen_q)
                col_limit = _causal_col_limit(
                q_last - 1, ratio, seqlen_k
                )
                num_k_blocks = (col_limit + block_I - 1) // block_I
                if tid < CONSUMER_THREADS:
                    S_frag = T.alloc_fragment([M, block_I], accum_dtype)
                    dS_frag = T.alloc_fragment([M, block_I], accum_dtype)
                    dQ_frag = T.alloc_fragment([M, dim], accum_dtype)
                    dW_partial = T.alloc_fragment([M, block_I], accum_dtype)
                    dW_frag = T.alloc_fragment([M], accum_dtype)

                    T.clear(dQ_frag)
                    T.clear(dW_frag)
                    T.barrier_wait(q_ready, 0)
                    T.barrier_wait(w_ready, 0)

                    for ko in T.serial(num_k_blocks):
                        s = ko % num_stages
                        T.barrier_wait(k_ready[s], (ko // num_stages) % 2)
                        T.barrier_wait(gs_ready[s], (ko // num_stages) % 2)

                        T.gemm(
                            Q_s,
                            K_s[s, :, :],
                            S_frag,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for m, kk in T.Parallel(M, block_I):
                            S_frag[m, kk] = T.max(
                                S_frag[m, kk] * sm_scale, 0.0
                            )

                        if ko >= 1:
                            T.barrier_wait(ds_free, (ko - 1) % 2)

                        for m, kk in T.Parallel(M, block_I):
                            if q_rows == 1:
                                gs = GS_shared[s, 0, kk]
                                dS_frag[m, kk] = T.if_then_else(
                                    S_frag[m, kk] > 0.0,
                                    gs * T.Cast(accum_dtype, W_shared[m])
                                    * sm_scale,
                                    0.0,
                                )
                            else:
                                gs = GS_shared[s, m // heads, kk]
                                dS_frag[m, kk] = T.if_then_else(
                                    S_frag[m, kk] > 0.0,
                                    T.if_then_else(
                                        q_base + m // heads < seqlen_q,
                                        gs * T.Cast(accum_dtype, W_shared[m])
                                        * sm_scale,
                                        0.0,
                                    ),
                                    0.0,
                                )
                            dW_partial[m, kk] = S_frag[m, kk] * gs
                        T.reduce_sum(dW_partial, dW_frag, dim=1, clear=False)

                        T.copy(dS_frag, dS_q)
                        if q_rows == 1:
                            T.copy(dS_frag, dS_k)
                        else:
                            for m, kk in T.Parallel(M, block_I):
                                if m < heads:
                                    dS_k0[m, kk] = T.Cast(dtype, dS_frag[m, kk])
                                else:
                                    dS_k1[m - heads, kk] = T.Cast(
                                        dtype, dS_frag[m, kk])

                        T.fence_proxy_async()
                        T.barrier_arrive(ds_written)
                        T.barrier_wait(ds_written, ko % 2)
                        T.barrier_arrive(ds_ready)

                        T.gemm(dS_q, K_b[s, :, :], dQ_frag)

                        T.barrier_arrive(k_free[s])

                    if q_rows == 1:
                        T.copy(dQ_frag, dQ[bb, bq, 0:heads, 0:dim])
                        T.copy(dW_frag, dW[bb, bq, 0:heads])
                    else:
                        for m, dd in T.Parallel(M, dim):
                            if q_base + m // heads < seqlen_q:
                                dQ[bb, q_base + m // heads, m % heads, dd] = (
                                    T.Cast(dtype, dQ_frag[m, dd])
                                )
                        for m in T.Parallel(M):
                            if q_base + m // heads < seqlen_q:
                                dW[bb, q_base + m // heads, m % heads] = (
                                    T.Cast(dtype, dW_frag[m])
                                )

                else:
                    dK_frag = T.alloc_fragment([block_I, dim], accum_dtype)

                    if q_rows == 1:
                        T.tma_copy(Q[bb, bq, 0:heads, 0:dim], Q_s,
                                   barrier=q_ready)
                        T.tma_copy(Q[bb, bq, 0:heads, 0:dim], Q_b,
                                   barrier=q_ready)
                        T.tma_copy(W[bb, bq, 0:heads], W_shared,
                                   barrier=w_ready)
                    else:
                        T.tma_copy(
                            Q[bb, q_base, 0:heads, 0:dim],
                            Q_s[0:heads, :], barrier=q_ready)
                        T.tma_copy(
                            Q[bb, q_base, 0:heads, 0:dim],
                            Q_b0, barrier=q_ready)
                        T.tma_copy(
                            W[bb, q_base, 0:heads],
                            W_shared[0:heads], barrier=w_ready)
                        T.tma_copy(
                            Q[bb, T.min(q_base + 1, seqlen_q - 1),
                              0:heads, 0:dim],
                            Q_s[heads:M, :], barrier=q_ready)
                        T.tma_copy(
                            Q[bb, T.min(q_base + 1, seqlen_q - 1),
                              0:heads, 0:dim],
                            Q_b1, barrier=q_ready)
                        T.tma_copy(
                            W[bb, T.min(q_base + 1, seqlen_q - 1), 0:heads],
                            W_shared[heads:M], barrier=w_ready)
                    T.barrier_arrive(q_ready)
                    T.barrier_arrive(w_ready)

                    for p in range(num_stages - 1):
                        if p < num_k_blocks:
                            T.tma_copy(
                                K[bb, p * block_I : (p + 1) * block_I, 0:dim],
                                K_s[p, :, :],
                                barrier=k_ready[p],
                            )
                            T.tma_copy(
                                K[bb, p * block_I : (p + 1) * block_I, 0:dim],
                                K_b[p, :, :],
                                barrier=k_ready[p],
                            )
                            if q_rows == 1:
                                T.tma_copy(
                                    GS[bb, bq,
                                       p * block_I : (p + 1) * block_I],
                                    GS_shared[p, 0, :],
                                    barrier=gs_ready[p],
                                )
                            else:
                                T.tma_copy(
                                    GS[bb, q_base,
                                       p * block_I : (p + 1) * block_I],
                                    GS_shared[p, 0, :],
                                    barrier=gs_ready[p],
                                )
                                T.tma_copy(
                                    GS[bb, T.min(q_base + 1, seqlen_q - 1),
                                       p * block_I : (p + 1) * block_I],
                                    GS_shared[p, 1, :],
                                    barrier=gs_ready[p],
                                )
                            T.barrier_arrive(k_ready[p])
                            T.barrier_arrive(gs_ready[p])

                    for ko in T.serial(num_k_blocks):
                        nxt = ko + num_stages - 1
                        nxt_slot = nxt % num_stages
                        if nxt < num_k_blocks:
                            if nxt >= num_stages:
                                T.barrier_wait(
                                    k_free[nxt_slot],
                                    (nxt // num_stages - 1) % 2,
                                )
                            T.tma_copy(
                                K[
                                    bb,
                                    nxt * block_I : (nxt + 1) * block_I,
                                    0:dim,
                                ],
                                K_s[nxt_slot, :, :],
                                barrier=k_ready[nxt_slot],
                            )
                            T.tma_copy(
                                K[
                                    bb,
                                    nxt * block_I : (nxt + 1) * block_I,
                                    0:dim,
                                ],
                                K_b[nxt_slot, :, :],
                                barrier=k_ready[nxt_slot],
                            )
                            T.tma_copy(
                                GS[bb, q_base,
                                   nxt * block_I : (nxt + 1) * block_I],
                                GS_shared[nxt_slot, 0, :],
                                barrier=gs_ready[nxt_slot],
                            )
                            if q_rows == 2:
                                T.tma_copy(
                                    GS[bb, T.min(q_base + 1, seqlen_q - 1),
                                       nxt * block_I : (nxt + 1) * block_I],
                                    GS_shared[nxt_slot, 1, :],
                                    barrier=gs_ready[nxt_slot],
                                )
                            T.barrier_arrive(k_ready[nxt_slot])
                            T.barrier_arrive(gs_ready[nxt_slot])

                        T.barrier_wait(ds_ready, ko % 2)
                        T.clear(dK_frag)
                        if q_rows == 1:
                            T.gemm(dS_k, Q_b, dK_frag, transpose_A=True)
                        else:
                            T.gemm(dS_k0, Q_b0, dK_frag, transpose_A=True)
                            T.gemm(dS_k1, Q_b1, dK_frag, transpose_A=True)
                        T.atomic_add(dK[bb, ko * block_I, 0], dK_frag)
                        T.barrier_arrive(ds_free)

        return kernel

    return _jit_kernel(
        heads,
        dim,
        block_I,
        num_stages,
        ratio,
        dtype,
        accum_dtype,
        q_rows,
    )

def _stream_context(current_stream, device):
    """Run work on *current_stream*, honouring ``torch.musa`` streams.

    Delegates to :func:`mudnn._torch_stream.stream_context`, which understands
    MUSA handles, the default stream and per-thread streams. The previous local
    copy tested ``isinstance(current_stream, torch.cuda.Stream)`` -- and
    ``torch.musa.Stream`` is *not* a ``torch.cuda.Stream`` -- so every MUSA
    stream fell through to ``nullcontext()`` and the launch silently ran on the
    ambient stream instead. A raw handle was worse: ``torch.cuda.ExternalStream``
    raises ``ValueError`` for a MUSA device, which the local ``except`` clause
    did not catch.

    Reusing the shared helper also keeps default-stream ordering consistent with
    the rest of the package, which is the reason it exists.
    """
    from mudnn._torch_stream import stream_context

    return stream_context(current_stream, device)

def dense_indexer_backward(
    batch,
    max_seqlen_q,
    max_seqlen_k,
    heads,
    dim,
    sm_scale=1.0,
    block_I=64,
    ratio=1,
    is_varlen=False,
    has_q_causal_offsets=False,
    threads=THREADS,
    num_stages=1,
    q_rows=2,
    score_grad_chunk=256,
    score_grad_threads=256,
    dtype="bfloat16",
    accum_dtype="float32",
):
    """Build the dense indexer-backward runner.

    Mirrors ``cudnn-frontend``'s dense factory signature, including the
    keyword-only envelope (``block_I``, ``ratio``, ``is_varlen``,
    ``has_q_causal_offsets``). Returns a callable with the signature, tensor
    contract, layout rules and limits documented in the module docstring.

    This generic variant accepts **BSHD only**: ``is_varlen`` and
    ``has_q_causal_offsets`` raise ``NotImplementedError``. Use the mp_32 variant
    (selected automatically on that capability) for THD or explicit causal
    offsets.

    Raises ``ValueError`` for a configuration outside the envelope, naming the
    constraint.
    """
    _require_tilelang()
    torch_mod = _require_torch()

    if is_varlen:
        raise NotImplementedError(
            "dense_indexer_backward is_varlen (THD/packed) is not implemented "
            "in the TileLang MUSA kernel; build with BSHD tensors."
        )
    if has_q_causal_offsets:
        raise NotImplementedError(
            "dense_indexer_backward has_q_causal_offsets is not implemented in "
            "the TileLang MUSA kernel: the causal bound is derived from the "
            "absolute query index (ratio only)."
        )
    if float(sm_scale) <= 0.0:
        raise ValueError(
            "dense_indexer_backward requires sm_scale > 0: the ReLU gate is "
            "applied to the scaled scores, which only matches the unscaled "
            f"formulation for positive scales (got sm_scale={sm_scale})."
        )
    if int(ratio) < 1:
        raise ValueError(f"dense_indexer_backward requires ratio >= 1, got {ratio}")
    if int(heads) % 16 != 0 or int(dim) % 8 != 0:
        raise ValueError(
            "dense_indexer_backward requires SQMMA-friendly tiles: heads a "
            f"multiple of 16 and dim a multiple of 8 (got heads={heads}, "
            f"dim={dim})"
        )
    if int(block_I) <= 0:
        raise ValueError(
            f"dense_indexer_backward requires block_I > 0, got {block_I}"
        )
    for _name, _val in (("batch", batch), ("max_seqlen_q", max_seqlen_q)):
        if int(_val) < 1:
            raise ValueError(
                f"dense_indexer_backward requires {_name} >= 1, got {_val}"
            )
    if int(max_seqlen_k) % int(block_I) != 0:
        raise ValueError(
            "dense_indexer_backward requires block_I to divide max_seqlen_k "
            f"(the TMA K-tile loads are unpredicated), got block_I={block_I}, "
            f"max_seqlen_k={max_seqlen_k}"
        )
    # Only 1 and 2 are implemented: every branch below is written for those two,
    # and the dS handoff indexes ``dS_k1[m - heads]``, which is only meaningful
    # when the stacked extent is one or two head-groups. Without this check a
    # larger value reaches TileLang and fails there with an internal shape
    # assertion (``Q[2] is illegal``) that says nothing about the real cause.
    # The mp_32 factory does implement 3 and 4; this one does not.
    if int(q_rows) not in (1, 2):
        raise ValueError(
            "dense_indexer_backward (architecture-generic) implements "
            f"q_rows in (1, 2), got q_rows={q_rows}. Use the mp_32 variant for "
            "q_rows 3 or 4."
        )
    m_extent = int(q_rows) * int(heads)
    smem_bytes = _SMEM_OVERHEAD + (
        2 * m_extent * dim * _DTYPE_BYTES[dtype]
        + 2 * num_stages * block_I * dim * _DTYPE_BYTES[dtype]
        + 2 * m_extent * block_I * _DTYPE_BYTES[dtype]
        + 2 * num_stages * q_rows * block_I * 4
        + 2 * m_extent * _DTYPE_BYTES[dtype]
    )
    if smem_bytes > _MAX_DYNAMIC_SHARED_MEMORY:
        raise ValueError(
            "dense_indexer_backward plan needs "
            f"{smem_bytes} B of dynamic shared memory but the target allows "
            f"{_MAX_DYNAMIC_SHARED_MEMORY} B; reduce block_I, num_stages or "
            f"q_rows (block_I={block_I}, num_stages={num_stages}, "
            f"heads={heads}, dim={dim}, q_rows={q_rows})"
        )
    if int(threads) != THREADS:
        raise ValueError(
            f"dense_indexer_backward requires threads={THREADS} (a 256/256 "
            f"TMA-producer / compute split, expressed as "
            f"tid < PRODUCER_THREADS); got threads={threads}"
        )

    plan = (int(heads), int(dim), int(block_I), int(num_stages), int(ratio),
            int(q_rows))
    plan_dtype = _TORCH_DTYPES[dtype]
    declared = (int(batch), int(max_seqlen_q), int(max_seqlen_k))

    score_grad = _score_grad_kernel(
        chunk=int(score_grad_chunk),
        ratio=int(ratio),
        threads=int(score_grad_threads),
    )
    gemm = _dense_indexer_backward_gemm(
        heads=int(heads),
        dim=int(dim),
        block_I=int(block_I),
        num_stages=int(num_stages),
        ratio=int(ratio),
        dtype=dtype,
        accum_dtype=accum_dtype,
        q_rows=int(q_rows),
    )
    default_sm_scale = float(sm_scale)

    def _grad_loss_tensor(grad_loss, device):
        if grad_loss is None:
            return torch_mod.ones((1,), dtype=torch_mod.float32, device=device)
        if torch_mod.is_tensor(grad_loss):
            if grad_loss.numel() != 1:
                raise ValueError(
                    "grad_loss must be a single-element tensor, got numel="
                    f"{grad_loss.numel()}"
                )
            return grad_loss.reshape(1).to(
                device=device, dtype=torch_mod.float32
            )
        return torch_mod.full(
            (1,), float(grad_loss), dtype=torch_mod.float32, device=device
        )

    def _check_tensor(name, tensor, shape, expected_dtype):
        if tensor is None:
            raise ValueError(f"dense_indexer_backward requires {name}")
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"dense_indexer_backward {name} shape mismatch: this plan was "
                f"built for {tuple(shape)}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"dense_indexer_backward {name} dtype mismatch: this plan was "
                f"built for {expected_dtype}, got {tensor.dtype}"
            )

    def _run(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK,
        AttnScore,
        AttnL1Norm,
        IdxScoreRaw,
        IdxLSE,
        GradLoss,
        grad_scale,
        CuSeqlensQ=None,
        CuSeqlensK=None,
        QCausalOffsets=None,
        current_stream=None,
        sm_scale=None,
    ):
        if CuSeqlensQ is not None or CuSeqlensK is not None:
            raise NotImplementedError(
                "dense_indexer_backward THD execution is not implemented in the "
                "TileLang MUSA kernel"
            )
        if QCausalOffsets is not None:
            raise NotImplementedError(
                "dense_indexer_backward q_causal_offsets execution is not "
                "implemented in the TileLang MUSA kernel"
            )

        if IndexQ.ndim != 4 or IndexK.ndim != 3:
            raise ValueError(
                "dense_indexer_backward expects BSHD index_q (B, Sq, H, D) and "
                f"index_k (B, Sk, D); got index_q.ndim={IndexQ.ndim}, "
                f"index_k.ndim={IndexK.ndim}"
            )
        b, sq, h, d = (int(IndexQ.shape[0]), int(IndexQ.shape[1]),
                       int(IndexQ.shape[2]), int(IndexQ.shape[3]))
        sk = int(IndexK.shape[1])
        if (h, d) != (plan[0], plan[1]):
            raise ValueError(
                "dense_indexer_backward heads/dim are compile-time tile axes "
                f"and cannot vary at runtime: this runner was built for "
                f"heads={plan[0]}, dim={plan[1]}, got heads={h}, dim={d}"
            )
        if sk % plan[2] != 0:
            raise ValueError(
                "dense_indexer_backward requires block_I to divide seqlen_k "
                f"(the TMA K-tile loads are unpredicated), got "
                f"block_I={plan[2]}, seqlen_k={sk}"
            )
        _check_tensor("index_q", IndexQ, (b, sq, h, d), plan_dtype)
        _check_tensor("weights", Weights, (b, sq, h), plan_dtype)
        _check_tensor("index_k", IndexK, (b, sk, d), plan_dtype)
        _check_tensor("d_index_q", dIndexQ, (b, sq, h, d), plan_dtype)
        _check_tensor("d_weights", dWeights, (b, sq, h), plan_dtype)
        _check_tensor("attn_score", AttnScore, (b, sq, sk), torch_mod.float32)
        _check_tensor("attn_l1norm", AttnL1Norm, (b, sq), torch_mod.float32)
        _check_tensor("index_score", IdxScoreRaw, (b, sq, sk), torch_mod.float32)
        _check_tensor("index_lse", IdxLSE, (b, sq), torch_mod.float32)
        if dIndexK.dtype != torch_mod.float32:
            raise ValueError(
                "dense_indexer_backward requires an fp32 dIndexK buffer (the "
                "kernel atomic-accumulates fp32); cast afterwards when the "
                f"output dtype is not fp32, got {dIndexK.dtype}"
            )
        _check_tensor("d_index_k", dIndexK, (b, sk, d), torch_mod.float32)

        for _name, _t in (
            ("index_q", IndexQ), ("weights", Weights), ("index_k", IndexK),
            ("d_index_q", dIndexQ), ("d_weights", dWeights),
            ("attn_score", AttnScore), ("attn_l1norm", AttnL1Norm),
            ("index_score", IdxScoreRaw), ("index_lse", IdxLSE),
            ("d_index_k", dIndexK),
        ):
            if _t.stride(-1) != 1:
                raise ValueError(
                    f"dense_indexer_backward {_name} must have a unit "
                    f"innermost stride (TMA requirement); got strides "
                    f"{tuple(_t.stride())}"
                )

        eff_sm_scale = default_sm_scale if sm_scale is None else float(sm_scale)
        if eff_sm_scale <= 0.0:
            raise ValueError(
                "dense_indexer_backward requires sm_scale > 0: the ReLU gate "
                "is applied to the scaled scores, which only matches the "
                "unscaled formulation for positive scales (got "
                f"sm_scale={eff_sm_scale})."
            )

        # These two are device tensors the kernel reads, so they are created on
        # the stream the kernel launches on; left outside, a non-default stream
        # would race their fill against the first read.
        with _stream_context(current_stream, IndexQ.device):
            grad_loss = _grad_loss_tensor(GradLoss, IndexQ.device)
            scale = torch_mod.full(
                (1,), float(grad_scale), dtype=torch_mod.float32,
                device=IndexQ.device,
            )
            dIndexK.zero_()
            score_grad(AttnScore, AttnL1Norm, IdxScoreRaw, IdxLSE, grad_loss, scale)
            gemm(IndexQ, IndexK, Weights, IdxScoreRaw, dIndexQ, dWeights,
                 dIndexK, eff_sm_scale)

    _run.sm_scale = default_sm_scale
    _run.ratio = int(ratio)
    _run.block_I = int(block_I)
    _run.plan = plan
    _run.declared_shape = declared
    _run.producer_threads = PRODUCER_THREADS
    _run.consumer_threads = CONSUMER_THREADS
    return _run
