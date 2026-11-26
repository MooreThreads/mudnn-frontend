"""TileLang GDN engine: the GDN and GDN_BWD operations on chunked MUSA kernels."""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

import mudnn
from mudnn.engines.base import BaseEngine, CompiledPlan, ExecutionContext

from ..graph_analyzer import Facts
from .engine import check_contiguous, tilelang_la_gate

# Kernel dtypes: bfloat16 q/k/v/beta, float32 g and state.
IO_DTYPE = torch.bfloat16
GATE_DTYPE = torch.float32
STATE_DTYPE = torch.float32


class GdnTileLangPlan(CompiledPlan):
    """Route (packed-varlen or dense) and batch/sequence view resolved once per
    configuration from ``facts``."""

    plan_name = "gdn_tilelang"

    def __init__(self, facts: Facts):
        from .kernels import engine_api as kernels

        self.kernels = kernels
        self.is_bwd = facts.is_bwd
        self.seqs = facts.seqs
        # A ragged batch is packed on the token axis (B == 1, cu_seqlens) and
        # carries no state; only the dense [B, S, H, D] route does.
        self.packed = facts.uniform_len is None
        self.batch = 1 if self.packed else facts.num_sequences
        self.seq = facts.total_tokens if self.packed else facts.uniform_len
        self.heads = facts.h_o
        self.d_qk = facts.d_qk
        self.d_v = facts.d_v

    def _bshd(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the kernels' [B, S, ...] view of a THD buffer ``t``."""
        return t.view(self.batch, self.seq, *t.shape[1:])

    def _pad_rows(self, initial_state, output_final_state: bool) -> int:
        if self.packed:
            return 0
        return self.kernels.dense_pad_rows(
            self.batch, self.seq, self.heads, self.d_qk, self.d_v,
            initial_state, output_final_state)

    def execute(self, facts: Facts, ctx: ExecutionContext, **buffers):
        # Raised here, not from check_support: the documented contract for a cadence
        # this engine cannot serve is NotImplementedError, and planning would turn a
        # check_support refusal into a plan-name error instead.
        if not facts.is_bwd and facts.checkpoint_every_n_tokens > 0:
            raise NotImplementedError(
                f"{self.plan_name}: the training kernels produce no per-chunk `state_checkpoints`, so "
                "this call needs gdn_tilelang_v2 or a cadence of 0")
        if facts.is_bwd != self.is_bwd:
            raise ValueError(f"{self.plan_name}: plan was built for {'GDN_BWD' if self.is_bwd else 'GDN'}")
        return self._backward(**buffers) if self.is_bwd else self._forward(**buffers)

    def _forward(self, *, q, k, v, g, beta, cu_seqlens, initial_state, scale,
                 output_final_state):
        """Returns ``(o, final_state, A, Hs, Delta, GPrefix, beta_hbs)``; every
        entry after ``final_state`` is the backward handoff.

        The driver hands the canonical cuDNN V-major ``[N, HO, V, K]`` state; these
        kernels read and report the K-major ``[N, HO, K, V]`` one, so the
        conversion lives here rather than in the router.
        """
        kern = self.kernels
        if initial_state is not None:
            initial_state = initial_state.transpose(-1, -2).contiguous()
        check_contiguous(self.plan_name, q=q, k=k, v=v, g=g, beta=beta,
                         initial_state=initial_state)
        q, k, v, g, beta = (self._bshd(t) for t in (q, k, v, g, beta))
        if self.packed:
            o, ht, *handoff = kern.chunk_forward(
                q, k, v, g, beta, h0=None, scale=scale, need_final_state=False,
                cu_seqlens=cu_seqlens, lengths=self.seqs)
            true_len = self.seq
        else:
            pad = self._pad_rows(initial_state, output_final_state)
            qp, kp, vp, gp, bp, true_len, _aligned = kern.prepare_dense(
                q, k, v, g, beta, initial_state, output_final_state,
                for_backward=True)
            assert qp.shape[1] == self.seq + pad, (
                "dense_pad_rows disagrees with prepare_dense; the backward would "
                "pad q/k to a different length than the forward did")
            o, ht, *handoff = kern.chunk_forward(
                qp, kp, vp, gp, bp, h0=initial_state, scale=scale,
                need_final_state=output_final_state)
            o = o[:, :true_len]
        if ht is not None:
            ht = ht.transpose(-1, -2).contiguous()
        return (o.reshape(-1, *o.shape[2:]), ht, o.new_empty((0,)), *handoff)

    def _backward(self, *, dO, d_final_state, q, k, cu_seqlens, handoff,
                  initial_state, scale, output_final_state, v=None, g=None,
                  beta=None, checkpoints=None):
        """Returns ``(dq, dk, dv, dg, dbeta, d_initial_state)``.

        ``handoff`` is this engine's own forward contract,
        ``(A, Hs, Delta, GPrefix, beta_hbs)``; the driver passes it through
        untouched so another engine can define a different one. ``v``/``g``/
        ``beta`` are accepted because a recomputing engine needs them; this one
        reads every activation out of the handoff instead.
        """
        A, Hs, Delta, GPrefix, beta_hbs = handoff
        kern = self.kernels
        check_contiguous(self.plan_name, dO=dO, q=q, k=k)
        q, k, dO = (self._bshd(t) for t in (q, k, dO))
        true_len = self.seq
        pad = self._pad_rows(initial_state, output_final_state)
        if pad:
            # The handoff is chunk-aligned, so the rows read here must be too;
            # forward and backward both pad with zeros.
            q, k, dO = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, dO))
        dO = kern.tl_contiguous(dO)
        if d_final_state is not None:
            # canonical V-major in, K-major to the kernels
            d_final_state = kern.tl_contiguous(
                d_final_state.transpose(-1, -2))

        vmaps = (kern.varlen_maps(cu_seqlens, q.device, lengths=self.seqs)
                 if self.packed else None)
        dq, dk, dv, dg, dbeta, dh0 = kern.run_backward(
            q, k, beta_hbs, dO, A, Hs, Delta, GPrefix, dht=d_final_state,
            scale=scale, need_dh0=initial_state is not None,
            dbeta_dtype=IO_DTYPE, vmaps=vmaps)
        if vmaps is not None:
            # The packed kernels emit v-side gradients in chunk order; restore
            # the caller's token axis.
            dv, dg, dbeta = (t.index_select(1, vmaps.rows) for t in (dv, dg, dbeta))
        grads = tuple(
            t[:, :true_len].to(dtype).reshape(-1, *t.shape[2:])
            for t, dtype in zip((dq, dk, dv, dg, dbeta),
                                (IO_DTYPE, IO_DTYPE, IO_DTYPE, GATE_DTYPE, IO_DTYPE)))
        if dh0 is None:
            return (*grads, None)
        return (*grads, dh0.to(STATE_DTYPE).transpose(-1, -2).contiguous())


def _warn_if_unsound(facts: Facts) -> None:
    """Warn once when the training pair is chosen on MP32.

    The chunked kernels are self-consistent there (the fusion-knob equivalence holds)
    but agree with an FP64 recurrence only to relL2 ~1.4 at D = 128, and D = 64 fails to
    compile. The default routing avoids them on MP32, so this is reachable only by a
    deliberate pin; warn rather than decline so a future TileLang fix can be observed.
    """
    if facts.is_bwd:
        return
    if facts.arch != (3, 2):
        return
    warnings.warn(
        "gdn_tilelang: the chunked GDN training kernels are numerically unsound on MP32 "
        "with the current TileLang (fp32-recurrence relL2 ~1.4 at D=128, D=64 fails to "
        "compile); gdn_tilelang_v2 is the default there",
        RuntimeWarning,
        stacklevel=2,
    )


class GdnTileLangEngine(BaseEngine):
    """The GDN family on TileLang MUSA: bf16 io, DK == DV in {64, 128}, chunk 64."""

    name = "gdn_tilelang"

    def check_support(self, facts: Facts) -> None:
        kern = _kernels()
        tilelang_la_gate(self.name, facts, "GDN")
        if facts.g_dtype is not mudnn.data_type.FLOAT:
            raise NotImplementedError(f"{self.name}: 'g' must be fp32 log-space decay, got {facts.g_dtype}")
        if facts.beta_dtype not in (mudnn.data_type.BFLOAT16, None):
            raise NotImplementedError(f"{self.name}: 'beta' must be bfloat16, got {facts.beta_dtype}")
        if facts.state_dtype not in (mudnn.data_type.FLOAT, None):
            raise NotImplementedError(f"{self.name}: 'initial_state' must be fp32, got {facts.state_dtype}")

        heads, d_qk, d_v = facts.h_o, facts.d_qk, facts.d_v
        if facts.uniform_len is None:
            if facts.has_initial_state or facts.wants_final_state:
                raise NotImplementedError(
                    f"{self.name}: a ragged batch carries no per-sequence state; pass "
                    "initial_state=None and output_final_state=False")
            if not kern.gdn_varlen_supported(facts.seqs, heads, d_qk, d_v, kern.CHUNK):
                raise NotImplementedError(
                    kern.gdn_varlen_error(self.name, facts.seqs, heads, d_qk, d_v, kern.CHUNK))
        elif not kern.gdn_dense_supported(facts.num_sequences, facts.uniform_len,
                                          heads, d_qk, d_v, kern.CHUNK):
            raise NotImplementedError(
                kern.gdn_dense_error(self.name, facts.num_sequences, facts.uniform_len,
                                     heads, d_qk, d_v, kern.CHUNK))
    def build_plan(self, facts: Facts, ctx: ExecutionContext = None) -> GdnTileLangPlan:
        _warn_if_unsound(facts)
        return GdnTileLangPlan(facts)


def _kernels():
    """Returns the kernel entry-point package, imported on first use."""
    from .kernels import engine_api

    return engine_api


__all__ = ["GdnTileLangEngine", "GdnTileLangPlan"]
