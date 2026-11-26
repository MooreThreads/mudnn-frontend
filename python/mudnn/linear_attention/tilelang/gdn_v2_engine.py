"""GDN on the fused pairing: inference prefill forward + chunked backward.

The forward is the chunked prefill kernel (the same one the inference engine
uses); the backward is the chunked backward, which rebuilds the chunk factors
itself instead of reading a forward handoff. The pair therefore shares no
intermediate tensors, which is why its handoff is the empty tuple.

The one optional exception is the checkpoint series, mirroring cuDNN frontend's
``checkpoint_every_n_tokens``: when the caller sets a cadence the forward also
returns the state series and the backward reuses it instead of recomputing it.

Default for unpinned training calls: ``plan_name="gdn_tilelang_v2"``. It leads
GDN_ENGINE_PREFERENCE and declines MP31 and non-128 head dims, so those calls fall
back to ``gdn_tilelang``.
"""

from __future__ import annotations

import torch

import mudnn
from mudnn.engines.base import BaseEngine, CompiledPlan

from ..graph_analyzer import Facts
from .engine import check_contiguous

HEAD_DIM = 128


class GdnV2Plan(CompiledPlan):
    """Prefill forward, chunked backward; the handoff between them is empty."""

    plan_name = "gdn_tilelang_v2"

    def __init__(self, facts: Facts):
        self.is_bwd = facts.is_bwd

    def execute(self, facts: Facts, ctx=None, **buffers):
        if facts.is_bwd != self.is_bwd:
            raise ValueError(f"{self.plan_name}: plan was built for "
                             f"{'GDN_BWD' if self.is_bwd else 'GDN'}")
        if self.is_bwd:
            return self._backward(**buffers)
        return self._forward(facts=facts, **buffers)

    def _forward(self, *, q, k, v, g, beta, cu_seqlens, initial_state, scale,
                 output_final_state, facts=None):
        """Returns ``(o, final_state, checkpoints)``; the series is empty unless asked."""
        from .kernels.inference.mp32.gdn_prefill import fused_chunk_gdn_prefill

        if initial_state is not None:
            initial_state = initial_state.contiguous()
        check_contiguous(self.plan_name, q=q, k=k, v=v, g=g, beta=beta,
                         initial_state=initial_state)
        cadence = int(getattr(facts, "checkpoint_every_n_tokens", 0) or 0)
        o, checkpoints, final_state = fused_chunk_gdn_prefill(
            q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
            g=g.float().unsqueeze(0), b=beta.unsqueeze(0),
            a=None, cu_seqlens=cu_seqlens, scale=scale,
            initial_state=initial_state, output_state=None,
            output_final_state=bool(output_final_state),
            is_log_space=True, l2norm_eps=None,
            checkpoint_every_n_tokens=cadence,
        )
        if checkpoints is None:
            checkpoints = o.new_empty((0,))
        return o.squeeze(0), final_state, checkpoints

    def _backward(self, *, dO, d_final_state, q, k, v, g, beta, cu_seqlens,
                  handoff, initial_state, scale, output_final_state,
                  checkpoints=None):
        """Returns ``(dq, dk, dv, dg, dbeta, d_initial_state)``.

        ``handoff`` is empty: this backward rebuilds the chunk factors itself. The
        forward's ``checkpoints``, when the caller asked for them, stand in for the
        rebuild of the state series only.
        """
        from .gdn_backward import prepare_gdn_bwd

        check_contiguous(self.plan_name, dO=dO, q=q, k=k, v=v, g=g, beta=beta)
        if checkpoints is not None and not checkpoints.numel():
            checkpoints = None
        prepared = prepare_gdn_bwd(
            dO.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(),
            g.float().contiguous(), beta.contiguous(), cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            d_final_state=d_final_state,
            state_checkpoints=checkpoints,
            checkpoint_every_n_tokens=64 if checkpoints is not None else 0,
            use_qk_l2norm_in_kernel=False,
            safe_gate=False,
            gate_domain="log",
            use_beta_sigmoid_in_kernel=False,
        )
        dq, dk, dv, dg, dbeta, dh0 = prepared.run()[:6]
        return (dq, dk, dv, dg.to(g.dtype), dbeta.to(beta.dtype), dh0)


class GdnV2Engine(BaseEngine):
    """fused pairing: MP32, K = V = 128, fp16/bf16 io, fp32 state."""

    name = "gdn_tilelang_v2"

    def check_support(self, facts: Facts) -> None:
        if facts.invalid:
            raise NotImplementedError(f"{self.name}: {facts.invalid}")
        if facts.device_type != "musa" or not facts.uniform_io:
            raise NotImplementedError(f"{self.name}: matching MUSA q/k/v required")
        if facts.arch != (3, 2):
            raise NotImplementedError(
                f"{self.name}: the chunked backward is MP32-only, got arch {facts.arch}")
        if facts.checkpoint_every_n_tokens not in (0, 64):
            raise NotImplementedError(f"{self.name}: training checkpoint cadence must be 0 or 64")
        if facts.d_qk != HEAD_DIM or facts.d_v != HEAD_DIM:
            raise NotImplementedError(
                f"{self.name}: K = V = {HEAD_DIM} only, got K={facts.d_qk} V={facts.d_v}")
        if facts.io_dtype not in (mudnn.data_type.BFLOAT16, mudnn.data_type.HALF):
            raise NotImplementedError(
                f"{self.name}: fp16/bf16 io only, got {facts.io_dtype}")
        if facts.g_dtype is not mudnn.data_type.FLOAT:
            raise NotImplementedError(
                f"{self.name}: 'g' must be fp32 log-space decay, got {facts.g_dtype}")
        if facts.state_dtype not in (mudnn.data_type.FLOAT, None):
            raise NotImplementedError(
                f"{self.name}: fp32 state only, got {facts.state_dtype}")

    def build_plan(self, facts: Facts, ctx=None) -> GdnV2Plan:
        return GdnV2Plan(facts)


__all__ = ["GdnV2Engine", "GdnV2Plan"]
