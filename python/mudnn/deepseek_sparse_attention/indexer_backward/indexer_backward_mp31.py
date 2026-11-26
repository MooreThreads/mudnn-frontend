"""Sparse (top-K) indexer backward for the MUSA ``mp_31`` TileLang path.

The sparse sibling of :mod:`dense_indexer_backward_mp32`, and a port of
``cudnn-frontend``'s ``indexer_backward_sm100``. The factory and the callable it
returns take the same arguments in the same order as upstream::

    kernel = indexer_backward_mp31(batch, seqlen, seqlen_k, heads, dim, topk,
                                   sm_scale=1.0, block_I=64,
                                   topk_indices_global=True)

MP31 port notes
-----------------------
Derived from :mod:`indexer_backward_mp32`; the kernels stay arch-generic
(``target="musa"``). The differences:

* ``block_I`` defaults to 64. The MP31 driver rejects opt-in dynamic
  shared memory near the 256 KB device cap (a 201728-byte request aborts
  with ``musa_module.cc:240: Fatal``), and ``block_I=128`` with the extra
  ``Q_s2`` buffer needs 206336 B at heads=64, dim=128 -- over the ceiling.
  ``block_I=64`` needs 123648 B.
* The dk GEMM reads Q from a dedicated second shared buffer ``Q_s2`` (plain
  ``T.copy`` load). One buffer cannot serve both the S GEMM (Q under
  ``_annotate_sqmma(..., True)``, transposed-B) and the dk GEMM (plain
  operand, opposite layout): reannotating a single buffer for both trips
  either a hard ``k_major mismatch`` compiler error or -- keeping the S
  layout -- silently corrupts dIndexK (cosine ~0.05 while dIndexQ and
  dWeights stay correct).
    kernel(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK,
           AttnScore, IndexScore, TopkIndices, GradLoss, grad_scale,
           current_stream=None)

plus ``.score_grad`` / ``.score_grad_zero`` / ``.gemm_only`` /
``.gemm_after_score`` for callers that stage the two halves themselves.

Tensor contract
---------------
Shapes (BSHD; ``B`` may be any value)::

    IndexQ      (B, S_q, H, D)   bfloat16
    Weights     (B, S_q, H)      bfloat16
    IndexK      (B, S_k, D)      bfloat16
    dIndexQ     (B, S_q, H, D)   bfloat16
    dWeights    (B, S_q, H)      bfloat16
    dIndexK     (B, S_k, D)      float32 or bfloat16
    AttnScore   (B, S_q, topk)   float32
    IndexScore  (B, S_q, topk)   float32
    TopkIndices (B, S_q, topk)   int32

``bfloat16`` is not a preference: the SQMMA operand type is fixed when the kernel
is built, so any other input dtype is reinterpreted rather than converted and is
rejected by ``check_support``.

``AttnScore`` and ``IndexScore`` are **probabilities**, not logits -- the
already-normalised target and predict distributions, with ``IndexScore`` aligned
slot-for-slot with ``TopkIndices``. The score-grad stage **overwrites
``AttnScore`` in place** with ``grad_signal`` and leaves ``IndexScore``
read-only, as upstream does; a caller that reuses the buffer across two launches
therefore feeds the first launch's gradient in as the second one's target.

``dIndexK`` is fp32 (atomic-accumulated directly) or bfloat16 (internal fp32
scratch plus a trailing cast). Either way it is zeroed internally, so a caller's
stale contents do not matter.

``grad_scale`` is a runtime launch argument, never a compile key: the loss scale
may change between calls without recompiling.

Compile key and limits
----------------------
One kernel per ``(heads, dim, block_I, topk, dtype, accum_dtype,
topk_indices_global)``. ``batch``, ``seqlen`` and ``seqlen_k`` are runtime
extents, so a packed training loop that changes ``(B, S_q, S_k)`` every step
reuses one compiled artifact.

Rejected at build time, with a message naming the constraint:

* ``topk > 2048``
* ``block_I`` not dividing ``topk``
* ``heads % 8 != 0``
* a configuration whose shared-memory budget exceeds the 192 KB cap

``topk_indices_global`` selects the id convention and is part of the key, so the
two conventions are two kernels:

* ``True`` -- ``b * S_k + local``, indexing the flat ``(B * S_k, D)`` K view
  directly. Trusted as given, as upstream does.
* ``False`` -- local, ``0 <= id < S_k``, range-checked and offset in registers.
  An out-of-range or negative id contributes nothing.

Not implemented
---------------
``use_persistent`` (upstream's persistent grid-stride CTA), TMA ``Gather4``, the
``sm100_v2`` two-term bf16-expansion backend, and the ``use_ds_half`` split dS
publication. ``dWeights`` is written in the input dtype, not fp32.
"""
import math

import torch

from ._common import (
    _GEMM_PASS_CONFIGS,
    _SCORE_GRAD_PASS_CONFIGS,
    _SMEM_OVERHEAD,
    _annotate_sqmma,
    _dyn,
    _require_torch,
    _require_tilelang,
)

# MP31 (mp_31) measured opt-in ceiling: the driver rejects a 201728-byte
# request ("musa_module.cc:240: Fatal: Failed to set the allowed dynamic
# shared memory size") while ~184 KB requests launch, so budget 192 KB.
# The mp_32 sibling keeps the 256 KB device cap.
_MAX_DYNAMIC_SHARED_MEMORY = 196608

tilelang = None
T = None
try:
    import tilelang
    import tilelang.language as T
except Exception:
    tilelang = None
    T = None

CLIP_LOG_MIN = -100.0
CLIP_PROB_MIN = math.exp(CLIP_LOG_MIN)

GEMM_THREADS = 256

SCORE_GRAD_THREADS = 256

SCORE_GRAD_CHUNK = 256

MAX_TOPK = 2048

def _score_grad_kernel_mp31(topk, threads, chunk):
    tl = _require_tilelang()
    n_chunks = (int(topk) + int(chunk) - 1) // int(chunk)

    @tl.jit(target="musa", pass_configs=_SCORE_GRAD_PASS_CONFIGS)
    def _jit_kernel(topk, threads, chunk):
        batch, seqlen = _dyn("batch", "seqlen")
        rows_q = _dyn("rows_q")[0]

        @T.prim_func
        def kernel(
            AttnScore: T.Tensor((rows_q, topk), "float32"),
            IndexScore: T.Tensor((rows_q, topk), "float32"),
            GradLoss: T.Tensor((1,), "float32"),
            Scale: T.Tensor((1,), "float32"),
            max_seqlen: T.int32,
            n_batch: T.int32,
        ):
            with T.Kernel(max_seqlen, n_batch, threads=threads) as (bq, bb):
                g_part = T.alloc_fragment([1, chunk], "float32")
                sum_f = T.alloc_fragment([1], "float32")
                f_meta2 = T.alloc_fragment([1, chunk], "float32")
                s_sum = T.alloc_shared([1], "float32")

                q_row = bb * max_seqlen + bq
                tid = T.get_thread_binding()
                s_scale = T.alloc_shared([1], "float32")
                if tid == 0:
                    s_scale[0] = Scale[0] * GradLoss[0]
                T.sync_threads()

                for _r, kk in T.Parallel(1, chunk):
                    f_meta2[0, kk] = s_scale[0]

                T.clear(s_sum)
                for ko in T.serial(n_chunks):
                    T.clear(g_part)
                    for _r, kk in T.Parallel(1, chunk):
                        k_idx = ko * chunk + kk
                        if k_idx < topk:
                            sml_p = IndexScore[q_row, k_idx]
                            target = T.max(AttnScore[q_row, k_idx],
                                           CLIP_PROB_MIN)
                            mask = T.if_then_else(
                                sml_p >= CLIP_PROB_MIN, 1.0, 0.0)
                            g_part[0, kk] = -target * mask * f_meta2[0, kk]
                    T.reduce_sum(g_part, sum_f, dim=1)
                    if tid == 0:
                        s_sum[0] = s_sum[0] + sum_f[0]
                    T.sync_threads()

                for ko in T.serial(n_chunks):
                    for _r, kk in T.Parallel(1, chunk):
                        k_idx = ko * chunk + kk
                        if k_idx < topk:
                            sml_p = IndexScore[q_row, k_idx]
                            target = T.max(AttnScore[q_row, k_idx],
                                           CLIP_PROB_MIN)
                            mask = T.if_then_else(
                                sml_p >= CLIP_PROB_MIN, 1.0, 0.0)
                            g = -target * mask * f_meta2[0, kk]
                            AttnScore[q_row, k_idx] = g - sml_p * s_sum[0]

        return kernel

    return _jit_kernel(int(topk), int(threads), int(chunk))

def _gemm_kernel_mp31(heads, dim, block_I, topk, dtype, accum_dtype,
                      topk_indices_global):
    tl = _require_tilelang()
    num_blocks = int(topk) // int(block_I)
    if num_blocks * int(block_I) != int(topk):
        raise ValueError(
            f"indexer_backward needs block_I to divide topk (got topk={topk}, "
            f"block_I={block_I}); the top-K axis is walked in whole blocks")

    @tl.jit(target="musa", pass_configs=_GEMM_PASS_CONFIGS)
    def _jit_kernel(heads, dim, block_I, topk, dtype, accum_dtype,
                    topk_indices_global):
        batch, seqlen = _dyn("batch", "seqlen")
        rows_q = _dyn("rows_q")[0]
        rows_k = _dyn("rows_k")[0]

        @T.prim_func
        def kernel(
            Q: T.Tensor((rows_q, heads, dim), dtype),
            K: T.Tensor((rows_k, dim), dtype),
            W: T.Tensor((rows_q, heads), dtype),
            GS: T.Tensor((rows_q, topk), accum_dtype),
            dQ: T.Tensor((rows_q, heads, dim), dtype),
            dW: T.Tensor((rows_q, heads), dtype),
            dK: T.Tensor((rows_k, dim), "float32"),
            TopkIdx: T.Tensor((rows_q, topk), "int32"),
            max_seqlen: T.int32,
            max_seqlen_k: T.int32,
            n_batch: T.int32,
            sm_scale: T.float32,
        ):
            with T.Kernel(max_seqlen, n_batch, threads=GEMM_THREADS) as (bq, bb):
                M = heads
                q_row = bb * max_seqlen + bq

                Q_s = T.alloc_shared([M, dim], dtype)
                Q_s2 = T.alloc_shared([M, dim], dtype)
                K_s = T.alloc_shared([block_I, dim], dtype)
                K_b = T.alloc_shared([block_I, dim], dtype)
                dS_q = T.alloc_shared([M, block_I], dtype)
                dS_kT = T.alloc_shared([block_I, M], dtype)
                GS_s = T.alloc_shared([block_I], accum_dtype)
                idx_s = T.alloc_shared([block_I], "int32")
                valid_s = T.alloc_shared([block_I], "int32")
                dK_sh = T.alloc_shared([block_I, dim], accum_dtype)

                S_frag = T.alloc_fragment([M, block_I], accum_dtype)
                dS_sign = T.alloc_fragment([M, block_I], accum_dtype)
                dQ_frag = T.alloc_fragment([M, dim], accum_dtype)
                dW_part = T.alloc_fragment([M, block_I], accum_dtype)
                dW_frag = T.alloc_fragment([M], accum_dtype)
                dK_frag = T.alloc_fragment([block_I, dim], accum_dtype)
                W_frag = T.alloc_fragment([M], dtype)

                tid = T.get_thread_binding()
                T.copy(W[q_row, 0:heads], W_frag)
                q_ready = T.alloc_barrier(GEMM_THREADS)
                T.tma_copy(Q[q_row, 0:heads, 0:dim], Q_s, barrier=q_ready)
                T.barrier_arrive(q_ready)
                T.barrier_wait(q_ready, 0)
                T.copy(Q[q_row, 0:heads, 0:dim], Q_s2)

                T.clear(dQ_frag)
                T.clear(dW_part)

                for p in range(num_blocks):
                    for r in T.Parallel(block_I):
                        raw = TopkIdx[q_row, p * block_I + r]
                        if topk_indices_global:
                            idx_s[r] = raw
                            valid_s[r] = 1
                        else:
                            ok = (raw >= 0) & (raw < max_seqlen_k)
                            idx_s[r] = T.if_then_else(
                                ok, bb * max_seqlen_k + raw,
                                bb * max_seqlen_k)
                            valid_s[r] = T.if_then_else(ok, 1, 0)
                    T.sync_threads()

                    for r, dd in T.Parallel(block_I, dim):
                        v = K[idx_s[r], dd]
                        K_s[r, dd] = v
                        K_b[r, dd] = v
                    for r in T.Parallel(block_I):
                        GS_s[r] = GS[q_row, p * block_I + r]
                    T.sync_threads()

                    _annotate_sqmma(Q_s, True, dim, sg=16)
                    T.gemm(Q_s, K_s, S_frag, transpose_B=True,
                           clear_accum=True, policy=T.GemmWarpPolicy.FullRow)

                    for m, kk in T.Parallel(M, block_I):
                        s_val = S_frag[m, kk]
                        gs = GS_s[kk]
                        keep = valid_s[kk] != 0
                        dS_sign[m, kk] = T.if_then_else(
                            T.if_then_else(keep, s_val > 0.0, False),
                            sm_scale * gs * W_frag[m], 0.0)
                        dW_part[m, kk] += T.if_then_else(
                            keep, gs * T.max(sm_scale * s_val, 0.0), 0.0)

                    T.copy(dS_sign, dS_q)
                    T.sync_threads()
                    T.gemm(dS_q, K_b, dQ_frag,
                           policy=T.GemmWarpPolicy.FullRow)

                    for kk, m in T.Parallel(block_I, M):
                        dS_kT[kk, m] = dS_q[m, kk]
                    T.sync_threads()
                    T.gemm(dS_kT, Q_s2, dK_frag,
                           clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
                    T.sync_threads()

                    T.copy(dK_frag, dK_sh)
                    T.sync_threads()
                    for r, dd in T.Parallel(block_I, dim):
                        T.atomic_add(dK[idx_s[r], dd], dK_sh[r, dd])
                    T.sync_threads()

                T.reduce_sum(dW_part, dW_frag, dim=1)
                T.copy(dQ_frag, dQ[q_row, 0:heads, 0:dim])
                T.copy(dW_frag, dW[q_row, 0:heads])

        return kernel

    return _jit_kernel(int(heads), int(dim), int(block_I), int(topk), dtype,
                       accum_dtype, bool(topk_indices_global))

def _smem_bytes(heads, dim, block_I, dtype):
    """Dynamic shared memory the sparse kernel allocates, in bytes.

    Mirrors the ``T.alloc_shared`` list in ``_gemm_kernel_mp31`` one entry at a
    time, because this number is the *only* guard against a plan that overflows
    the 192 KB budget. Counting a buffer twice is harmless; counting one too few
    lets an unbuildable plan through, which is what happened to ``dS_k``: the
    kernel allocates ``dS_q`` *and* ``dS_k`` (both ``[M, block_I]``), and the
    formula used to charge only one of them. At ``heads=192, dim=128,
    block_I=128`` that undercounted by 48 KB and passed a plan needing ~281 KB.
    """
    elem = 2 if dtype in ("bfloat16", "float16") else 4
    acc = 4
    total = (
        2 * heads * dim * elem      # Q_s + Q_s2 (separate sqmma layouts)
        + 2 * block_I * dim * elem  # K_s + K_b
        + 2 * heads * block_I * elem  # dS_q + dS_k
        + block_I * acc             # GS_s
        + 2 * block_I * 4           # idx_s + valid_s
        + block_I * dim * acc       # dK_sh
    )
    return _SMEM_OVERHEAD + total

def indexer_backward_mp31(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    topk,
    sm_scale=1.0,
    block_I=64,
    topk_indices_global: bool = True,
):
    """Build the sparse indexer-backward runner for ``mp_32``.

    Signatures match ``cudnn-frontend``'s ``indexer_backward_sm100``. ``batch`` /
    ``seqlen`` / ``seqlen_k`` are runtime extents, so one compiled artifact serves
    every packed shape; ``heads`` / ``dim`` / ``topk`` / ``block_I`` /
    ``topk_indices_global`` are compile-time.

    Returns a callable carrying ``.score_grad``, ``.score_grad_zero``,
    ``.gemm_only`` and ``.gemm_after_score``, plus ``.sm_scale``, ``.block_I``,
    ``.topk`` and ``.plan``. Raises ``ValueError`` for a configuration outside the
    envelope (see the module docstring).

    See the module docstring for the tensor contract, the in-place
    ``AttnScore`` overwrite, and the two ``topk_indices_global`` conventions.
    """
    torch_mod = _require_torch()
    _require_tilelang()

    batch = int(batch)
    seqlen = int(seqlen)
    seqlen_k = int(seqlen_k)
    heads = int(heads)
    dim = int(dim)
    topk = int(topk)
    block_I = int(block_I)
    if topk > MAX_TOPK:
        raise ValueError(
            f"indexer_backward_mp31 supports topk <= {MAX_TOPK} (got {topk}); "
            "larger values need the persistent reschedule that is not ported")
    if topk % block_I != 0:
        raise ValueError(
            f"indexer_backward_mp31 needs block_I to divide topk "
            f"(got topk={topk}, block_I={block_I})")
    if heads % 8 != 0:
        raise ValueError(
            f"indexer_backward_mp31 needs heads to be a multiple of 8 "
            f"(got heads={heads}); the FullRow warp split is by 8 rows")
    _elem = 2
    need = _smem_bytes(heads, dim, block_I, "bfloat16")
    if need > _MAX_DYNAMIC_SHARED_MEMORY:
        raise ValueError(
            f"indexer_backward_mp31 needs {need} B of dynamic shared memory for "
            f"heads={heads} dim={dim} block_I={block_I}, over the "
            f"{_MAX_DYNAMIC_SHARED_MEMORY} B cap")

    dtype = "bfloat16"
    accum_dtype = "float32"
    score_grad = _score_grad_kernel_mp31(topk, SCORE_GRAD_THREADS,
                                         SCORE_GRAD_CHUNK)
    gemm = _gemm_kernel_mp31(heads, dim, block_I, topk, dtype, accum_dtype,
                             topk_indices_global)
    default_sm_scale = float(sm_scale)

    def _scalar_tensor(value, device):
        if torch_mod.is_tensor(value):
            return value
        return torch_mod.full((1,), float(value), dtype=torch_mod.float32,
                              device=device)

    def _grad_loss_tensor(value, device):
        if value is None:
            return torch_mod.ones((1,), dtype=torch_mod.float32, device=device)
        if torch_mod.is_tensor(value):
            return value.reshape(1) if value.ndim == 0 else value
        return torch_mod.full((1,), float(value), dtype=torch_mod.float32,
                              device=device)

    def _flat(t, shape):
        if tuple(t.shape) == tuple(shape):
            return t
        if not t.is_contiguous():
            raise ValueError(
                "indexer_backward_mp31 requires contiguous tensors; got a "
                f"non-contiguous tensor of shape {tuple(t.shape)}")
        return t.view(*shape)

    def _run(
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK,
        AttnScore,
        IndexScore,
        TopkIndices,
        GradLoss,
        grad_scale,
        current_stream=None,
    ):
        rows_q = batch * seqlen
        rows_k = batch * seqlen_k
        Q = _flat(IndexQ, (rows_q, heads, dim))
        W = _flat(Weights, (rows_q, heads))
        K = _flat(IndexK, (rows_k, dim))
        dQ = _flat(dIndexQ, (rows_q, heads, dim))
        dW = _flat(dWeights, (rows_q, heads))
        GS = _flat(AttnScore, (rows_q, topk))
        IS = _flat(IndexScore, (rows_q, topk))
        Tk = _flat(TopkIndices, (rows_q, topk))
        # Everything that touches device memory happens inside the stream
        # context: the fp32 scratch, the scalar operands and the trailing cast.
        # Allocating the scratch outside would let the caching allocator hand it
        # back for reuse while this launch is still writing it, and the cast
        # would otherwise race the kernel when ``current_stream`` is not the
        # ambient stream.
        with _stream_context(current_stream, IndexQ.device):
            if dIndexK.dtype == torch_mod.float32:
                dK = _flat(dIndexK, (rows_k, dim))
                dK_f32 = dK
                out_dtype = None
            else:
                dK_f32 = torch_mod.empty((rows_k, dim),
                                         dtype=torch_mod.float32,
                                         device=IndexQ.device)
                out_dtype = dIndexK.dtype

            scale = _scalar_tensor(grad_scale, IndexQ.device)
            loss = _grad_loss_tensor(GradLoss, IndexQ.device)
            dK_f32.zero_()
            score_grad(GS, IS, loss, scale, seqlen, batch)
            gemm(Q, K, W, GS, dQ, dW, dK_f32, Tk,
                 seqlen, seqlen_k, batch, default_sm_scale)
            if out_dtype is not None:
                dIndexK.copy_(dK_f32.view_as(dIndexK))

    def _score_grad_only(IndexQ, AttnScore, IndexScore, GradLoss, grad_scale,
                         current_stream=None):
        rows_q = batch * seqlen
        GS = _flat(AttnScore, (rows_q, topk))
        IS = _flat(IndexScore, (rows_q, topk))
        loss = _grad_loss_tensor(GradLoss, IndexQ.device)
        scale = _scalar_tensor(grad_scale, IndexQ.device)
        with _stream_context(current_stream, IndexQ.device):
            score_grad(GS, IS, loss, scale, seqlen, batch)

    def _gemm_only(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK,
                   GradSignal, TopkIndices, current_stream=None):
        rows_q = batch * seqlen
        rows_k = batch * seqlen_k
        Q = _flat(IndexQ, (rows_q, heads, dim))
        W = _flat(Weights, (rows_q, heads))
        K = _flat(IndexK, (rows_k, dim))
        dQ = _flat(dIndexQ, (rows_q, heads, dim))
        dW = _flat(dWeights, (rows_q, heads))
        G = _flat(GradSignal, (rows_q, topk))
        Tk = _flat(TopkIndices, (rows_q, topk))
        with _stream_context(current_stream, IndexQ.device):
            if dIndexK.dtype == torch_mod.float32:
                dK_f32 = _flat(dIndexK, (rows_k, dim))
                out_dtype = None
            else:
                dK_f32 = torch_mod.empty((rows_k, dim),
                                         dtype=torch_mod.float32,
                                         device=IndexQ.device)
                out_dtype = dIndexK.dtype
            dK_f32.zero_()
            gemm(Q, K, W, G, dQ, dW, dK_f32, Tk,
                 seqlen, seqlen_k, batch, default_sm_scale)
            if out_dtype is not None:
                dIndexK.copy_(dK_f32.view_as(dIndexK))

    _run.score_grad = _score_grad_only
    _run.score_grad_zero = _score_grad_only
    _run.gemm_only = _gemm_only
    _run.gemm_after_score = _gemm_only
    _run.sm_scale = default_sm_scale
    _run.block_I = block_I
    _run.topk = topk
    _run.plan = {
        "batch": batch, "seqlen": seqlen, "seqlen_k": seqlen_k,
        "heads": heads, "dim": dim, "topk": topk, "block_I": block_I,
        "topk_indices_global": bool(topk_indices_global),
    }
    return _run

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

