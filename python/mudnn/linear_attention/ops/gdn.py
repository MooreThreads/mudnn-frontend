"""Gated DeltaNet (GDN) linear attention as a PyTorch custom operator.

Provides the ``gated_delta_net`` entry point and the ``mudnn::gated_delta_net_fwd``
and ``mudnn::gated_delta_net_bwd`` custom ops.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from functools import lru_cache
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

import mudnn
from mudnn.engines import decline_types, engines_for

from ..graph_analyzer import analyze

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

fprop_cache: dict = {}
bprop_cache: dict = {}

STATE_DTYPE = torch.float32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan_key(facts, plan_name):
    return (facts, plan_name)


#: MP32 D128 training prefers v2; unsupported configurations fall back to the
#: original training pair. ``plan_name`` pins one engine outright.
GDN_ENGINE_PREFERENCE = ("gdn_tilelang_v2", "gdn_tilelang", "gdn_prefill_tilelang",
                         "gdn_decode_tilelang")


def select_plan(facts, plan_name, cache):
    """The first engine that serves ``facts``, compiled and cached in ``cache``.

    ``plan_name`` pins one engine by name. Raises ``cudnnGraphNotSupportedError``
    when no engine serves the configuration.
    """
    key = _plan_key(facts, plan_name)
    plan = cache.get(key)
    if plan is not None:
        return plan
    engines = [e for e in engines_for([facts.op]) if e.name in GDN_ENGINE_PREFERENCE]
    engines.sort(key=lambda e: GDN_ENGINE_PREFERENCE.index(e.name))
    if plan_name is not None:
        engines = [e for e in engines if e.name == plan_name]
        if not engines:
            raise mudnn.cudnnGraphNotSupportedError(f"no {plan_name!r} engine for the GDN family")
    declined = []
    for engine in engines:
        try:
            engine.check_support(facts)
            plan = engine.build_plan(facts)
        except decline_types() as exc:
            declined.append(f"{engine.name}: {exc}")
            continue
        cache[key] = plan
        return plan
    raise mudnn.cudnnGraphNotSupportedError(
        "no engine serves this gated_delta_net call"
        + (" (" + "; ".join(declined) + ")" if declined else " (no engine is registered)"))


def _state_buffer(q, n, h_o, d_qk, d_v, wanted):
    """The state buffer, empty when the caller did not ask for one."""
    return q.new_empty((n, h_o, d_qk, d_v) if wanted else (0,), dtype=STATE_DTYPE)


def _input_device_context(device):
    # On some runtimes exchanging even the current device is expensive.
    # The ordinary same-device call needs no exchange; another input GPU does.
    if torch.musa.current_device() == device.index:
        return nullcontext()
    return torch.musa.device(device)


@lru_cache(maxsize=256)
def _backward_facts(facts):
    """Reuse immutable forward metadata; saved tensors enforce version checks."""
    return replace(facts, op="GDN_BWD", checkpoint_every_n_tokens=0)


def _run_forward(q, k, v, g, beta, cu_seqlens, scale, initial_state,
                 output_final_state, plan_name, checkpoint_every_n_tokens=0,
                 facts=None):
    """Analyze, pick the plan, run it. Returns ``(o, final, checkpoints, *handoff)``."""
    if facts is None:
        facts = analyze("GDN", q, k, v, g, beta, cu_seqlens,
                        initial_state=initial_state,
                        output_final_state=output_final_state,
                        checkpoint_every_n_tokens=checkpoint_every_n_tokens)
    plan = select_plan(facts, plan_name, fprop_cache)
    # TileLang compilation also consults the current device. Keep it aligned
    # with the input even when the caller last used another GPU.
    with _input_device_context(q.device):
        return plan.execute(
            facts, None, q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
            g=g.contiguous(), beta=beta.contiguous(), cu_seqlens=cu_seqlens,
            initial_state=None if initial_state is None else initial_state.contiguous(),
            scale=scale, output_final_state=output_final_state)


def _run_backward(dO, q, k, v, g, beta, cu_seqlens, handoff, scale, d_final_state,
                  initial_state, output_final_state, plan_name, checkpoints=None,
                  facts=None):
    """Analyze, pick the plan, run it. Returns the six gradients, ``dh0`` possibly None.

    ``handoff`` is opaque here: its arity and meaning belong to the engine that
    produced it, so a plan with a different forward contract cannot collide with
    the legacy five-tensor one.
    """
    if facts is None:
        facts = analyze("GDN_BWD", q, k, v, g, beta, cu_seqlens,
                        initial_state=initial_state,
                        output_final_state=output_final_state)
    plan = select_plan(facts, plan_name, bprop_cache)
    with _input_device_context(q.device):
        return plan.execute(
            facts, None, dO=dO.contiguous(), d_final_state=d_final_state,
            q=q.contiguous(), k=k.contiguous(), v=v.contiguous(),
            g=g.contiguous(), beta=beta.contiguous(), cu_seqlens=cu_seqlens,
            handoff=tuple(handoff), checkpoints=checkpoints,
            initial_state=initial_state, scale=scale,
            output_final_state=output_final_state)


# ---------------------------------------------------------------------------
# Forward (inference)
# ---------------------------------------------------------------------------


@torch.library.custom_op("mudnn::gated_delta_net_fwd", mutates_args=())
def gdn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    plan_name: Optional[str] = None,
    checkpoint_every_n_tokens: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward without the backward handoff.

    Returns ``(o, final_state, checkpoints)``, where ``checkpoints`` is empty
    unless ``checkpoint_every_n_tokens`` asked for the state series.
    """
    out = _run_forward(q, k, v, g, beta, cu_seqlens, scale, initial_state,
                       output_final_state, plan_name,
                       checkpoint_every_n_tokens)
    o, final_state, checkpoints = out[0], out[1], out[2]
    if final_state is None:
        final_state = q.new_empty((0,), dtype=STATE_DTYPE)
    return o, final_state, checkpoints


@gdn_fwd.register_fake
def gdn_fwd_fake(q, k, v, g, beta, cu_seqlens, scale, initial_state=None,
                 output_final_state=False, plan_name=None,
                 checkpoint_every_n_tokens=0):
    total, h_q, d_qk = q.shape
    h_v, d_v = v.shape[1], v.shape[2]
    h_o = max(h_q, h_v)
    n = cu_seqlens.shape[0] - 1
    checkpoints = (q.new_empty((max(total // checkpoint_every_n_tokens + n, 1),
                                h_o, d_v, d_qk))
                  if checkpoint_every_n_tokens > 0 else q.new_empty((0,)))
    return (q.new_empty((total, h_o, d_v)),
            _state_buffer(q, n, h_o, d_qk, d_v, output_final_state),
            checkpoints)


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


@torch.library.custom_op("mudnn::gated_delta_net_bwd", mutates_args=())
def gdn_bwd(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    handoff: List[torch.Tensor],
    scale: float,
    d_final_state: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients for one GDN call. ``handoff`` is the forward's ``(A, Hs, Delta,
    GPrefix, beta_hbs)``; ``v``/``g``/``beta`` are read for shape only."""
    dq, dk, dv, dg, dbeta, dh0 = _run_backward(
        dO, q, k, v, g, beta, cu_seqlens, handoff, scale, d_final_state,
        initial_state, output_final_state, plan_name)
    if dh0 is None:
        dh0 = q.new_empty((0,), dtype=STATE_DTYPE)
    return dq, dk, dv, dg, dbeta, dh0


@gdn_bwd.register_fake
def gdn_bwd_fake(dO, q, k, v, g, beta, cu_seqlens, handoff, scale,
                 d_final_state=None, initial_state=None, output_final_state=False,
                 plan_name=None):
    n = cu_seqlens.shape[0] - 1
    h_o, d_qk, d_v = max(q.shape[1], v.shape[1]), q.shape[-1], v.shape[-1]
    return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v),
            torch.empty_like(g), torch.empty_like(beta),
            _state_buffer(q, n, h_o, d_qk, d_v, initial_state is not None))


# ---------------------------------------------------------------------------
# Autograd
# ---------------------------------------------------------------------------


# Not a custom op: the handoff's token extent depends on the cu_seqlens values,
# which ``register_fake`` cannot compute from shapes alone.
class _GatedDeltaNet(torch.autograd.Function):
    """Training path: run the forward, keep its handoff, feed it to the backward."""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu_seqlens, scale, initial_state,
                output_final_state, plan_name, checkpoint_every_n_tokens=0):
        ctx.facts = analyze("GDN", q, k, v, g, beta, cu_seqlens,
                            initial_state=initial_state,
                            output_final_state=output_final_state,
                            checkpoint_every_n_tokens=checkpoint_every_n_tokens)
        o, final_state, checkpoints, *handoff = _run_forward(
            q, k, v, g, beta, cu_seqlens, scale, initial_state,
            output_final_state, plan_name, checkpoint_every_n_tokens,
            facts=ctx.facts)
        ctx.save_for_backward(q, k, v, g, beta, cu_seqlens, initial_state,
                              checkpoints, *handoff)
        ctx.scale = scale
        ctx.output_final_state = output_final_state
        ctx.plan_name = plan_name
        ctx.set_materialize_grads(False)
        if final_state is None:
            final_state = q.new_empty((0,), dtype=STATE_DTYPE)
        ctx.mark_non_differentiable(checkpoints)
        return o, final_state, checkpoints

    @staticmethod
    def backward(ctx, dO, d_final_state, _dcheckpoints):
        (q, k, v, g, beta, cu_seqlens, initial_state, checkpoints,
         *handoff) = ctx.saved_tensors
        if dO is None:
            dO = torch.zeros(q.shape[0], v.shape[1], v.shape[2],
                             dtype=v.dtype, device=v.device)
        if d_final_state is not None and d_final_state.numel() == 0:
            d_final_state = None
        dq, dk, dv, dg, dbeta, dh0 = _run_backward(
            dO, q, k, v, g, beta, cu_seqlens, handoff, ctx.scale, d_final_state,
            initial_state, ctx.output_final_state, ctx.plan_name, checkpoints,
            facts=_backward_facts(ctx.facts))
        return (dq, dk, dv, dg, dbeta, None, None,
                dh0 if initial_state is not None else None, None, None, None)


# ---------------------------------------------------------------------------
# Input adaptation
# ---------------------------------------------------------------------------


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf / (xf.square().sum(dim=-1, keepdim=True) + eps).sqrt()).to(x.dtype)


def _expand_heads(q, k, h_q, h_k, h_v):
    """Broadcast the query-side heads out to the value heads (grouped-query)."""
    if h_v == h_q and h_k == h_q:
        return q, k
    if h_v % h_q:
        raise ValueError(f"v heads ({h_v}) must be a multiple of q heads ({h_q})")

    def grow(t):
        heads = t.shape[1]
        return (t.unsqueeze(2).expand(t.shape[0], heads, h_v // heads, t.shape[-1])
                 .reshape(t.shape[0], h_v, t.shape[-1]))

    if h_k == h_q:
        return grow(q), grow(k)
    if h_k == h_v:
        return grow(q), k
    raise ValueError(f"k head count ({h_k}) must match q's ({h_q}) or v's ({h_v})")


def _apply_gate(g, safe_gate, a_log, dt_bias, h_o):
    """``g = -exp(a_log) * softplus(g + dt_bias)``, the safe-gate transform."""
    if not safe_gate:
        if a_log is not None or dt_bias is not None:
            raise ValueError("gated_delta_net: a_log/dt_bias require safe_gate=True")
        return g
    for name, t in (("a_log", a_log), ("dt_bias", dt_bias)):
        if t is not None and tuple(t.shape) != (h_o,):
            raise ValueError(f"gated_delta_net: {name!r} must be [{h_o}], got {tuple(t.shape)}")
    alpha = g.float() if dt_bias is None else g.float() + dt_bias.float()
    amplitude = 1.0 if a_log is None else a_log.float().exp()
    return -amplitude * F.softplus(alpha)


def _apply_beta(beta, use_sigmoid, allow_neg_eigval, out_dtype=None):
    if allow_neg_eigval and not use_sigmoid:
        raise ValueError("gated_delta_net: allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if not use_sigmoid:
        return beta
    activated = torch.sigmoid(beta.float())
    # Raw logits may arrive in fp32; the kernels read the activated value in the
    # io dtype, so narrow to the caller's dtype (bf16 for the training pair, and
    # fp16 when the port pairing is pinned).
    return (2.0 * activated if allow_neg_eigval else activated).to(
        beta.dtype if out_dtype is None else out_dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _gated_delta_net_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
):
    """Run validated training inputs; engines adapt the canonical K-last state."""

    # The kernels fuse none of these, so they run in torch and autograd owns
    # their gradients.
    h_o = max(q.shape[1], v.shape[1])
    q, k = _expand_heads(q, k, q.shape[1], k.shape[1], v.shape[1])
    if use_qk_l2norm_in_kernel:
        q, k = _l2_normalize(q), _l2_normalize(k)
    g = _apply_gate(g, safe_gate, a_log, dt_bias, h_o)
    beta = _apply_beta(beta, use_beta_sigmoid_in_kernel, allow_neg_eigval, q.dtype)

    if torch.is_grad_enabled() and any(
            t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state)):
        o, final, checkpoints = _GatedDeltaNet.apply(
            q, k, v, g, beta, cu_seqlens, float(scale), initial_state,
            bool(output_final_state), plan_name, int(checkpoint_every_n_tokens))
    else:
        o, final, checkpoints = torch.ops.mudnn.gated_delta_net_fwd(
            q, k, v, g, beta, cu_seqlens, float(scale), initial_state=initial_state,
            output_final_state=bool(output_final_state), plan_name=plan_name,
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens))
    if checkpoint_every_n_tokens > 0:
        return o, final, checkpoints
    return o, final


from .gdn_inference import gated_delta_net

__all__ = ["gated_delta_net"]
