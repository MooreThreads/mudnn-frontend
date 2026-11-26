"""GDN backward orchestration, autograd wiring, and the kernel-layer entry points."""

import torch
import torch.nn.functional as F

from .gdn_chunk_bwd_grad import gdn_chunk_bwd_grad
from .gdn_chunk_bwd_scan import k2_fused
from .gdn_chunk_fwd import _tl_contiguous, gdn_chunk_fwd
from .gdn_relayout import scalar_untranspose
from .gdn_shapes import (gdn_dense_error, gdn_dense_supported,
                         gdn_shape_error, gdn_shape_supported,
                         gdn_unpadded_supported, gdn_varlen_error,
                         gdn_varlen_supported)
from .gdn_varlen import varlen_lengths, varlen_maps

_UNPADDED_ENABLED = True

__all__ = ["gdn_chunk", "gdn_chunk_fwd_only"]


def _pad_to_chunk(q, k, v, g, beta, pad):
    qkv = tuple(F.pad(x, (0, 0, 0, 0, 0, pad)) for x in (q, k, v))
    return qkv + tuple(F.pad(x, (0, 0, 0, pad), value=0.0) for x in (g, beta))


def _check_inputs(q, k, v, g, beta, scale=None):
    if scale is not None and not scale > 0:
        raise ValueError(f"scale must be positive or None, got {scale}")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, beta)):
        raise TypeError("the TileLang GDN kernels require BF16 q, k, v and beta, got "
                        f"{q.dtype}, {k.dtype}, {v.dtype}, {beta.dtype}")
    if g.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"g must be FP32 log-space decay, got {g.dtype}")
    if any(x.device.type != "musa" for x in (q, k, v, g, beta)):
        raise ValueError("the TileLang GDN kernels are MUSA-only")


def _prepare_dense(q, k, v, g, beta, initial_state, output_final_state, *,
                   for_backward):
    """Gate the shape and pad to the chunk unless the unpadded path is open.

    Returns the (possibly padded) tensors plus the true and the chunk-aligned
    length. ``for_backward`` is what adds the ``initial_state`` clause: dh0
    comes from the one K2 kernel with no ragged split, so under a carry-in the
    handoffs have to be the chunk-aligned length. A forward-only call produces
    no dh0, which is why it keeps the unpadded path open there.
    """
    original_length = q.shape[1]
    pad = (-original_length) % 64
    padded_length = original_length + pad
    B, H, DK, DV = q.shape[0], q.shape[2], q.shape[3], v.shape[-1]
    if not gdn_dense_supported(B, original_length, H, DK, DV, 64):
        raise ValueError(
            gdn_dense_error("GDN", B, original_length, H, DK, DV, 64))

    unpadded = (
        _UNPADDED_ENABLED
        and bool(pad)
        and not output_final_state
        and not (for_backward and initial_state is not None)
        and gdn_unpadded_supported(B, original_length, H, DK, DV, 64)
    )
    if pad and not unpadded:
        q, k, v, g, beta = _pad_to_chunk(q, k, v, g, beta, pad)
    return q, k, v, g, beta, original_length, padded_length


def _save_ctx(ctx, saved, *, dtypes, scale, initial_state, original_length,
              padded_length, cu_seqlens, lengths=None):
    """Everything ``backward`` reads back, written from one place."""
    ctx.save_for_backward(*saved)
    ctx.scale = scale
    ctx.initial_dtype = initial_state.dtype if initial_state is not None else None
    ctx.input_dtypes = dtypes
    ctx.original_length = original_length
    ctx.padded_length = padded_length
    ctx.cu_seqlens = cu_seqlens
    ctx.lengths = lengths


def _run_gdn_chunk_bwd(q, k, beta_hbs, do, A, Hs, Delta, GPrefix, dht=None,
                       scale=None, chunk_size=64, need_dh0=True, dbeta_dtype=None,
                       vmaps=None):
    """Backward for one sequence. Returns the six gradients.

    ``q``/``k``/``do`` carry the TRUE length and the kernels predicate their row
    reads against it; every handoff and every output is chunk-aligned, which is
    the length all the shape predicates and allocations below use.
    """
    B, _, H, DK = q.shape
    S = Delta.shape[1]
    DV = do.shape[-1]
    if not gdn_shape_supported(B, S, H, DK, DV, chunk_size):
        raise ValueError(gdn_shape_error("GDN backward", B, S, H, DK, DV, chunk_size))
    if scale is None:
        scale = DK ** -0.5

    DRhs, DKst, Ds, DdLast, dh0 = k2_fused(
        q, k, beta_hbs, do, A, GPrefix, Delta, Hs, dht=dht, scale=scale,
        chunk_size=chunk_size, need_dh0=need_dh0, vmaps=vmaps,
    )
    dv = DRhs
    dq, dk, dg, dbeta = gdn_chunk_bwd_grad(
        q, k, beta_hbs, do, A, GPrefix, Delta, Hs, DRhs, DKst, Ds,
        DdLast, scale=scale, chunk_size=chunk_size, vmaps=vmaps,
    )
    dbeta_dtype = torch.float32 if dbeta_dtype is None else dbeta_dtype
    dg, dbeta = scalar_untranspose(dg, dbeta, dbeta_dtype=dbeta_dtype)
    return dq, dk, dv, dg, dbeta, dh0


class _GatedDeltaNet(torch.autograd.Function):
    """Padding, handoff retention, and the forward/backward pairing."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state, scale, output_final_state,
                cu_seqlens=None, lengths=None):
        if cu_seqlens is not None:
            return _forward_varlen(ctx, q, k, v, g, beta, scale, cu_seqlens,
                                   lengths)
        q, k, v, g, beta, original_length, padded_length = _prepare_dense(
            q, k, v, g, beta, initial_state, output_final_state,
            for_backward=True)

        o, ht, A, Hs, Delta, GPrefix, beta_hbs = gdn_chunk_fwd(
            q, k, v, g, beta, h0=initial_state, scale=scale,
            need_final_state=output_final_state)
        _save_ctx(ctx, (q, k, beta_hbs, A, Hs, Delta, GPrefix),
                  dtypes=(q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype),
                  scale=scale, initial_state=initial_state,
                  original_length=original_length, padded_length=padded_length,
                  cu_seqlens=None)
        return o[:, :original_length], ht if output_final_state else None

    @staticmethod
    def backward(ctx, do, dht):
        q, k, beta_hbs, A, Hs, Delta, GPrefix = ctx.saved_tensors
        cu_seqlens = ctx.cu_seqlens
        has_init = ctx.initial_dtype is not None
        if cu_seqlens is None and q.shape[1] != ctx.original_length:
            do = F.pad(do, (0, 0, 0, 0, 0, ctx.padded_length - ctx.original_length))
        do = _tl_contiguous(do)
        if dht is not None:
            dht = _tl_contiguous(dht)

        vmaps = (varlen_maps(cu_seqlens, q.device, lengths=ctx.lengths)
                 if cu_seqlens is not None else None)
        dq, dk, dv, dg, dbeta, dh0 = _run_gdn_chunk_bwd(
            q, k, beta_hbs, do, A, Hs, Delta, GPrefix, dht=dht, scale=ctx.scale,
            need_dh0=has_init, dbeta_dtype=ctx.input_dtypes[4], vmaps=vmaps,
        )
        if vmaps is not None:
            dv, dg, dbeta = (x.index_select(1, vmaps.rows) for x in (dv, dg, dbeta))
        grads = tuple(x[:, :ctx.original_length].to(dt)
                      for x, dt in zip((dq, dk, dv, dg, dbeta), ctx.input_dtypes))
        dh0 = dh0.to(ctx.initial_dtype) if has_init else None
        return grads + (dh0, None, None, None, None)


def _forward_varlen(ctx, q, k, v, g, beta, scale, cu_seqlens, lengths=None):
    """Packed varlen forward: nothing is padded and no state is carried."""
    o, _ht, A, Hs, Delta, GPrefix, beta_hbs = gdn_chunk_fwd(
        q, k, v, g, beta, h0=None, scale=scale, need_final_state=False,
        cu_seqlens=cu_seqlens, lengths=lengths)
    _save_ctx(ctx, (q, k, beta_hbs, A, Hs, Delta, GPrefix),
              dtypes=(q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype),
              scale=scale, initial_state=None,
              original_length=q.shape[1], padded_length=q.shape[1],
              cu_seqlens=cu_seqlens, lengths=lengths)
    return o, None


def _check_varlen(q, v, cu_seqlens, initial_state, output_final_state):
    """Refuse what the packed path does not serve, before any kernel runs.

    Returns the per-sequence token counts. Reading them is a blocking D2H sync,
    so this is the ONE place that pays it; everything downstream takes the
    tuple.
    """
    if q.shape[0] != 1:
        raise ValueError("cu_seqlens packs sequences on the token axis, so it "
                         f"requires B == 1; got B={q.shape[0]}")
    if initial_state is not None or output_final_state:
        raise NotImplementedError(
            "packed varlen does not carry per-sequence state yet; pass "
            "initial_state=None and output_final_state=False")
    seqs = varlen_lengths(cu_seqlens)
    if sum(seqs) != q.shape[1]:
        raise ValueError(f"cu_seqlens covers {sum(seqs)} tokens, q has {q.shape[1]}")
    H, DK, DV = v.shape[2], q.shape[3], v.shape[-1]
    if not gdn_varlen_supported(seqs, H, DK, DV, 64):
        raise ValueError(gdn_varlen_error("GDN", seqs, H, DK, DV, 64))
    return seqs


def gdn_chunk(q, k, v, g, beta, *, scale=None, initial_state=None,
              output_final_state=True, cu_seqlens=None):
    """Differentiable chunked gated delta rule.

    ``q, k, v`` are [B, S, H, 128] BF16 with q/k already L2-normalized, ``g`` is
    [B, S, H] FP32 log-space decay and ``beta`` is [B, S, H] BF16. Returns
    ``(o [B, S, H, 128], final_state [B, H, 128, 128] or None)``. A
    non-chunk-aligned ``S`` runs without a padded copy of the inputs unless the
    final state is requested; see ``gdn_shapes.gdn_unpadded_supported``.
    """
    _check_inputs(q, k, v, g, beta, scale)
    lengths = None
    if cu_seqlens is not None:
        lengths = _check_varlen(q, v, cu_seqlens, initial_state,
                                output_final_state)
    q, k, v, g, beta = (_tl_contiguous(x) for x in (q, k, v, g, beta))
    return _GatedDeltaNet.apply(q, k, v, g, beta, initial_state, scale,
                                output_final_state, cu_seqlens, lengths)


def gdn_chunk_fwd_only(q, k, v, g, beta, *, scale=None, initial_state=None,
                       output_final_state=True, cu_seqlens=None):
    """Forward without retaining the backward handoff. Same returns as above."""
    _check_inputs(q, k, v, g, beta, scale)
    if cu_seqlens is not None:
        lengths = _check_varlen(q, v, cu_seqlens, initial_state,
                                output_final_state)
        with torch.no_grad():
            o, *_ = gdn_chunk_fwd(q, k, v, g, beta, h0=None, scale=scale,
                                  need_final_state=False, cu_seqlens=cu_seqlens,
                                  lengths=lengths)
        return o, None
    q, k, v, g, beta, original_length, _ = _prepare_dense(
        q, k, v, g, beta, initial_state, output_final_state, for_backward=False)
    with torch.no_grad():
        o, ht, *_ = gdn_chunk_fwd(q, k, v, g, beta, h0=initial_state, scale=scale,
                                  need_final_state=output_final_state)
    return o[:, :original_length], ht if output_final_state else None
