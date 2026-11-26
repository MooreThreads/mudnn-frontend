# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Independent references for the DSA indexer-backward tests.

Ported from ``cudnn-frontend``'s
``test/python/deepseek_sparse_attention/cutedsl/dsa_reference.py``, keeping the
construction and the *reasoning* behind it rather than the file: the sparse and
dense predict distributions, the ratio-causal masks, the two autograd
references, and the cosine/RMS comparison the upstream tests assert on.

Why an autograd reference rather than hand-written gradient formulas
--------------------------------------------------------------------
The kernels run three bf16 GEMMs with fp32 accumulators. Building the same chain
in torch and letting ``autograd`` differentiate it gives a check that shares no
code and no algebra with the kernel: a sign error, a missing ``w[h]`` factor or a
transposed contraction all show up, where a hand-derived formula written by the
same author as the kernel would happily repeat the same mistake.

Two subtleties are carried over verbatim because they are load-bearing:

* The inputs stay in their **original dtype** through the recomputed score chain,
  so the vector-Jacobian product models the kernel's bf16 GEMMs with an fp32
  accumulator. Casting to fp32 first would compare
  ``fp32(bf16(x)) * fp32(bf16(y))`` against ``fp32(x) * fp32(y)``.
* The KL gradient stage is evaluated **analytically**, not by backpropagating
  through ``log(predict.clamp(min=eps))``. At ``eps = exp(-100)`` the log
  backward's ``1 / predict`` overflows fp32 for subnormal ``predict``, and
  clipping at ``float32.tiny`` instead moves the clip mask. Upstream measured
  ~0.55 rms relative error for the autograd-through-log form against <= 0.005
  for the analytic one, i.e. the difference would be in the *reference*.

Device neutrality
-----------------
Every helper takes its device from the tensors it is given, so the same module
serves the MUSA suite here and would serve a CUDA checkout unchanged.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch


#: The kernels' clip constant (``CLIP_LOG_MIN``), i.e. the smallest probability
#: the log-space KL formulation can represent.
CLIP_LOG_MIN = -100.0
CLIP_PROB_MIN = math.exp(CLIP_LOG_MIN)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def ratio_causal_mask(
    s_q: int,
    s_k: int,
    ratio: int,
    device,
    q_causal_offset: int = 0,
) -> torch.Tensor:
    """Ratio-causal validity mask for one local Q segment, ``(S_q, S_k)``.

    ``col_limit = floor((q_causal_offset + q + 1) / ratio)``, clamped to
    ``[0, S_k]`` -- the same expression the kernels use, with the offset added
    *before* the division, so under ``ratio > 1`` an offset is worth
    ``offset // ratio`` columns rather than ``offset``.
    """
    q = q_causal_offset + torch.arange(s_q, device=device, dtype=torch.int64)
    k_pos = torch.arange(s_k, device=device, dtype=torch.int64)
    col_limit = torch.div(q + 1, ratio, rounding_mode="floor").clamp(0, s_k)
    return k_pos.view(1, s_k) < col_limit.view(s_q, 1)


def batched_ratio_causal_mask(
    s_q: int,
    s_k: int,
    ratio: int,
    device,
    batch: int,
    q_causal_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``(B, S_q, S_k)`` ratio-causal masks, with optional per-batch offsets."""
    if q_causal_offsets is None:
        return (
            ratio_causal_mask(s_q, s_k, ratio, device)
            .unsqueeze(0)
            .expand(batch, -1, -1)
        )
    offsets = q_causal_offsets.to(device=device, dtype=torch.int64)
    q = torch.arange(s_q, device=device, dtype=torch.int64).view(1, s_q)
    k_pos = torch.arange(s_k, device=device, dtype=torch.int64).view(1, 1, s_k)
    col_limit = torch.div(
        offsets.view(batch, 1) + q + 1, ratio, rounding_mode="floor"
    ).clamp(0, s_k)
    return k_pos < col_limit.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Predict distributions
# ---------------------------------------------------------------------------


def indexer_predict_distribution(
    q_indexer: torch.Tensor,  # (B, S_q, H, D)
    k_indexer: torch.Tensor,  # (B, S_k, D)
    weights: torch.Tensor,  # (B, S_q, H)
    topk_indices: torch.Tensor,  # (B, S_q, topk) int32
    sm_scale: float,
    return_scores: bool = False,
    topk_indices_global: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Differentiable sparse predict distribution, ``(B, S_q, topk)``.

    Mirrors the kernel's score math as a plain op graph::

        score = sum_h relu(Q_h . K_{topk}^T) * (W_h * sm_scale)

    with ``-1`` indices masked to ``-inf`` before the softmax (the kernels treat
    a negative id as "no contribution"). Differentiable in ``q``, ``w`` and
    ``k``, which is what :func:`ref_indexer_backward` differentiates.

    ``topk_indices_global`` says which id convention the tensor carries:

    * ``False`` (default) -- local, ``0 <= id < S_k``, the convention a
      per-batch gather needs.
    * ``True`` -- flat, ``b * S_k + local``, the convention the kernel's flat
      ``(B * S_k, D)`` K view takes.

    The *reference* gathers inside each batch (``gather`` on dimension 2 of an
    expanded ``(B, S_q, S_k, D)``), so a flat id has to be converted back to
    local here. Getting this wrong is silent rather than loud: batch 0's flat ids
    equal its local ones, so only batch 1+ is misread, and only when ``B > 1``.
    """
    b, s_q, h, d = q_indexer.shape
    _, s_k, _ = k_indexer.shape
    topk = topk_indices.shape[-1]

    idx = topk_indices.long()
    if topk_indices_global:
        batch_off = (
            torch.arange(b, device=idx.device, dtype=torch.long).view(b, 1, 1) * s_k
        )
        idx = idx - batch_off
    # ``clamp`` must bound *both* ends. Only ``min=0`` leaves an
    # above-range or masked id as a gather index past ``S_k``, and such a gather
    # faults or hangs instead of comparing. ``valid`` below already zeroes those
    # positions' contribution, so which row is gathered does not matter.
    idx = idx.clamp(min=0, max=max(s_k - 1, 0))
    k_gather = torch.gather(
        k_indexer.unsqueeze(1).expand(b, s_q, s_k, d),
        2,
        idx.unsqueeze(-1).expand(b, s_q, topk, d),
    )  # (B, S_q, topk, D)

    scores_per_head = torch.einsum("bqhd,bqtd->bqht", q_indexer, k_gather)
    scores_per_head = torch.relu(scores_per_head)
    weights_scaled = weights * sm_scale
    scores = (scores_per_head * weights_scaled.unsqueeze(-1)).sum(dim=2)

    valid = topk_indices >= 0
    scores = scores.masked_fill(~valid, float("-inf"))
    predict = torch.softmax(scores, dim=-1)
    return (predict, scores) if return_scores else predict


def dense_indexer_predict_distribution(
    q_indexer: torch.Tensor,  # (B, S_q, H, D)
    k_indexer: torch.Tensor,  # (B, S_k, D)
    weights: torch.Tensor,  # (B, S_q, H)
    sm_scale: float,
    ratio: int,
    q_causal_offsets: Optional[torch.Tensor] = None,
    return_scores: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Differentiable dense (full-KV) predict distribution, ``(B, S_q, S_k)``."""
    b, s_q, h, d = q_indexer.shape
    _, s_k, _ = k_indexer.shape

    scores_per_head = torch.einsum(
        "bqhd,bkd->bqhk",
        q_indexer.to(torch.float32),
        k_indexer.to(torch.float32),
    )
    scores_per_head = torch.relu(scores_per_head)
    scores = (scores_per_head * (weights * sm_scale).unsqueeze(-1)).sum(dim=2)

    valid = batched_ratio_causal_mask(
        s_q, s_k, ratio, q_indexer.device, b, q_causal_offsets
    )
    scores = scores.masked_fill(~valid, float("-inf"))
    predict = torch.softmax(scores, dim=-1)
    return (predict, scores) if return_scores else predict


# ---------------------------------------------------------------------------
# Autograd references
# ---------------------------------------------------------------------------


def _kl_grad_signal(
    attn_score: torch.Tensor,
    predict_f32: torch.Tensor,
    grad_scale: torch.Tensor,
) -> torch.Tensor:
    """``grad_signal = g - predict * sum_k g`` with ``g = -target * mask * scale``.

    This is ``dL/dlogits`` for ``kl_div(log_target=True)`` with both operands
    clipped at :data:`CLIP_LOG_MIN`, evaluated analytically -- see the module
    docstring for why it must not be autograd through ``log``.
    """
    target_eff = attn_score.to(torch.float32).clamp(min=CLIP_PROB_MIN)
    log_clip_mask = (predict_f32 >= CLIP_PROB_MIN).to(torch.float32)
    g = -target_eff * log_clip_mask * grad_scale
    return g - predict_f32 * g.sum(dim=-1, keepdim=True)


def ref_indexer_backward(
    index_q: torch.Tensor,  # (B, S_q, H, D) bf16
    weights: torch.Tensor,  # (B, S_q, H)   bf16
    index_k: torch.Tensor,  # (B, S_k, D)   bf16
    attn_score: torch.Tensor,  # (B, S_q, topk) fp32 target
    index_score: torch.Tensor,  # (B, S_q, topk) fp32 predict operand
    topk_indices: torch.Tensor,  # (B, S_q, topk) int32
    sm_scale: float = 1.0,
    grad_scale: float = 1.0,
    topk_indices_global: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd reference for the sparse indexer backward.

    Forms the clipped-log KL grad signal from the supplied fp32 ``index_score``
    -- the exact tensor the kernel reads, so a subnormal ``predict`` is not
    flushed to zero and the clip mask does not move -- then backpropagates it
    through scores recomputed from ``(index_q, weights, index_k, topk_indices)``
    to ``index_q``, ``weights`` and ``index_k``. Returns gradients in the inputs'
    dtypes.
    """
    q = index_q.detach().clone().requires_grad_(True)
    w = weights.detach().clone().requires_grad_(True)
    k = index_k.detach().clone().requires_grad_(True)

    _, scores = indexer_predict_distribution(
        q, k, w, topk_indices, sm_scale, return_scores=True,
        topk_indices_global=topk_indices_global,
    )

    grad_signal = _kl_grad_signal(
        attn_score, index_score.detach().to(torch.float32), grad_scale
    )
    grad_signal = grad_signal.masked_fill(topk_indices < 0, 0.0)

    dq, dw, dk = torch.autograd.grad(
        scores, (q, w, k), grad_outputs=grad_signal.to(scores.dtype)
    )
    return dq.to(index_q.dtype), dw.to(weights.dtype), dk.to(index_k.dtype)


def ref_dense_indexer_backward(
    index_q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    attn_score_raw: torch.Tensor,
    attn_l1norm: torch.Tensor,
    sm_scale: float = 1.0,
    ratio: int = 1,
    grad_scale: float = 1.0,
    q_causal_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd reference for the dense indexer backward.

    The dense path consumes *raw* attention scores plus their per-row L1 norm
    rather than a normalised target, so the target is formed here and zeroed
    outside the ratio-causal mask before the same analytic KL stage.
    """
    q = index_q.detach().clone().requires_grad_(True)
    w = weights.detach().clone().requires_grad_(True)
    k = index_k.detach().clone().requires_grad_(True)

    predict, scores = dense_indexer_predict_distribution(
        q, k, w, sm_scale, ratio, q_causal_offsets=q_causal_offsets,
        return_scores=True,
    )

    b, s_q, _, _ = index_q.shape
    _, s_k, _ = index_k.shape
    valid = batched_ratio_causal_mask(
        s_q, s_k, ratio, index_q.device, b, q_causal_offsets
    )
    target = attn_score_raw.to(torch.float32).masked_fill(~valid, 0.0)
    target = target / attn_l1norm.to(torch.float32).unsqueeze(-1).clamp(min=1e-10)

    grad_signal = _kl_grad_signal(target, predict.to(torch.float32), grad_scale)
    grad_signal = grad_signal.masked_fill(~valid, 0.0)

    dq, dw, dk = torch.autograd.grad(
        scores, (q, w, k), grad_outputs=grad_signal.to(scores.dtype)
    )
    return dq.to(index_q.dtype), dw.to(weights.dtype), dk.to(index_k.dtype)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.to(torch.float32).flatten()
    bf = b.to(torch.float32).flatten()
    denom = af.norm() * bf.norm()
    if denom.item() == 0.0:
        return 1.0
    return (af @ bf / denom).item()


def _compare(name: str, actual, ref, cos_min: float, rms_rel_max: float) -> str:
    cos = cosine_similarity(actual, ref)
    err = actual.to(torch.float32) - ref.to(torch.float32)
    rms_err = err.pow(2).mean().sqrt().item()
    rms_ref = ref.to(torch.float32).pow(2).mean().sqrt().item()
    rms_rel = rms_err / max(rms_ref, 1e-12)
    assert cos >= cos_min, (
        f"{name}: cosine similarity {cos:.4f} < {cos_min} -- direction mismatch"
    )
    assert rms_rel <= rms_rel_max, (
        f"{name}: RMS relative error {rms_rel:.4f} > {rms_rel_max} -- magnitude "
        f"mismatch (rms_err={rms_err:.4g}, rms_ref={rms_ref:.4g})"
    )
    return f"{name} cos={cos:.6f} rms_rel={rms_rel:.4g}"


def check_ref_indexer_backward(
    index_q,
    weights,
    index_k,
    attn_score,
    index_score,
    topk_indices,
    d_index_q_actual,
    d_weights_actual,
    d_index_k_actual,
    sm_scale: float = 1.0,
    grad_scale: float = 1.0,
    topk_indices_global: bool = False,
    cos_min: float = 0.97,
    rms_rel_max: float = 0.55,
) -> str:
    """Compare the sparse gradients against :func:`ref_indexer_backward`.

    Two aggregate criteria rather than an elementwise bound: at bf16 input
    precision the worst elementwise error is dominated by near-zero entries,
    where the *relative* error blows up while the gradient is still correct. A
    systematic error (wrong direction or wrong magnitude) fails both criteria,
    and a quantisation-only difference passes both. The defaults are upstream's,
    calibrated at ``s_q=128, heads=64, topk=128``; larger problems measure
    ``cos > 0.99`` and callers may pass ``cos_min=0.99``.
    """
    dq_ref, dw_ref, dk_ref = ref_indexer_backward(
        index_q, weights, index_k, attn_score, index_score, topk_indices,
        sm_scale=sm_scale, grad_scale=grad_scale,
        topk_indices_global=topk_indices_global,
    )
    parts = []
    for name, actual, ref in (
        ("d_index_q", d_index_q_actual, dq_ref),
        ("d_weights", d_weights_actual, dw_ref),
        ("d_index_k", d_index_k_actual, dk_ref),
    ):
        parts.append(_compare(name, actual, ref, cos_min, rms_rel_max))
    return "; ".join(parts)


def check_ref_dense_indexer_backward(
    index_q,
    weights,
    index_k,
    attn_score_raw,
    attn_l1norm,
    d_index_q_actual,
    d_weights_actual,
    d_index_k_actual,
    sm_scale: float = 1.0,
    ratio: int = 1,
    grad_scale: float = 1.0,
    q_causal_offsets: Optional[torch.Tensor] = None,
    cos_min: float = 0.97,
    rms_rel_max: float = 0.55,
) -> str:
    """Dense counterpart of :func:`check_ref_indexer_backward`."""
    dq_ref, dw_ref, dk_ref = ref_dense_indexer_backward(
        index_q, weights, index_k, attn_score_raw, attn_l1norm,
        sm_scale=sm_scale, ratio=ratio, grad_scale=grad_scale,
        q_causal_offsets=q_causal_offsets,
    )
    parts = []
    for name, actual, ref in (
        ("d_index_q", d_index_q_actual, dq_ref),
        ("d_weights", d_weights_actual, dw_ref),
        ("d_index_k", d_index_k_actual, dk_ref),
    ):
        parts.append(_compare(name, actual, ref, cos_min, rms_rel_max))
    return "; ".join(parts)


def ref_dense_indexer_score_recompute(
    q_indexer: torch.Tensor,  # (B, S_q, H, D)
    k_indexer: torch.Tensor,  # (B, S_k, D)
    weights: torch.Tensor,  # (B, S_q, H)
    ratio: int = 1,
    q_causal_offsets: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dense score and log-denominator, as the forward recompute produces them.

    Returns ``(score, lse)`` with ``score[b, q, k] = sum_h relu(Q_h . K_k) * W_h``
    zeroed outside the mask, and ``lse = logsumexp`` over the valid columns.
    The tests need this to build a *consistent* ``index_score`` / ``index_lse``
    pair; an inconsistent pair would make the kernel and the reference disagree
    for a reason that has nothing to do with either.
    """
    b, s_q, h, d = q_indexer.shape
    _, s_k, _ = k_indexer.shape
    scores_per_head = torch.einsum(
        "bqhd,bkd->bqhk",
        q_indexer.to(torch.float32),
        k_indexer.to(torch.float32),
    )
    scores_per_head = torch.relu(scores_per_head)
    scores = (scores_per_head * weights.to(torch.float32).unsqueeze(-1)).sum(dim=2)
    valid = batched_ratio_causal_mask(
        s_q, s_k, ratio, q_indexer.device, b, q_causal_offsets
    )
    scores = scores.masked_fill(~valid, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    return scores.masked_fill(~valid, 0.0), lse
