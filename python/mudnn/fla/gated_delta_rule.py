# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Accelerated drop-in for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``.

Maps the flash-linear-attention signature onto ``mudnn.linear_attention.
gated_delta_net`` and calls the wrapped FLA function for anything the engines do
not serve, so a configuration outside the support surface still runs.

FLA is batch-first ``[B, T, H, ...]`` with ``g``/``beta`` indexed by the value
heads; the native op is token-packed THD, so the reshape is a view. The state is
exchanged in FLA's layout: K-major ``[N, HV, K, V]`` is transposed at the
boundary; ``state_v_first`` already matches the native K-last layout.
"""

from __future__ import annotations

from typing import Optional

import torch

import mudnn
from mudnn.linear_attention import gated_delta_net

_DECLINE = (mudnn.cudnnGraphNotSupportedError, NotImplementedError)
_IO_DTYPE = torch.bfloat16

_LAST = {"path": None}

#: Dense boundary vectors, reused so the op's length cache hits instead of
#: paying a device read per call.
_CU_CACHE: dict = {}


def last_path() -> Optional[str]:
    """The route the most recent shimmed call took, for tests and telemetry."""
    return _LAST["path"]


class _Decline(Exception):
    """Raised when a call cannot be adapted to the native op."""


def _dense_cu(batch: int, seq: int, device) -> torch.Tensor:
    """The ``[N + 1]`` boundaries of a dense ``[batch, seq]`` batch."""
    key = (batch, seq, device)
    cu = _CU_CACHE.get(key)
    if cu is None:
        cu = torch.arange(0, (batch + 1) * seq, seq, dtype=torch.int32, device=device)
        _CU_CACHE[key] = cu
    return cu


def _to_native(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens,
               use_qk_l2norm_in_kernel, use_beta_sigmoid_in_kernel, allow_neg_eigval,
               state_v_first, use_gate_in_kernel, A_log, dt_bias):
    """Run one FLA-shaped call on the native op. Returns ``(o, final_state)``."""
    if q.dim() != 4:
        raise _Decline("expected [B, T, H, K]")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise _Decline("allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    if any(t.dtype is not _IO_DTYPE for t in (q, k, v)):
        raise _Decline("q/k/v must be bfloat16")
    if not use_beta_sigmoid_in_kernel and beta.dtype is not _IO_DTYPE:
        raise _Decline("beta must be bfloat16 unless the sigmoid is fused")

    batch, seq, heads = q.shape[0], q.shape[1], q.shape[2]
    value_heads = v.shape[2]
    h_o = max(heads, value_heads)
    if g.shape[-1] != h_o or beta.shape[-1] != h_o:
        raise _Decline("g/beta head count does not match max(H, HV)")
    if cu_seqlens is None:
        cu = _dense_cu(batch, seq, q.device)
    else:
        if batch != 1:
            raise _Decline("varlen requires B == 1")
        cu = cu_seqlens

    a_log = dt_bias_t = None
    if use_gate_in_kernel:
        if A_log is None:
            raise _Decline("use_gate_in_kernel requires A_log")
        a_log = A_log.float().reshape(-1)
        dt_bias_t = dt_bias.float().reshape(-1) if dt_bias is not None else None
        if a_log.shape[0] != h_o or (dt_bias_t is not None and dt_bias_t.shape[0] != h_o):
            raise _Decline("a_log/dt_bias head count does not match max(H, HV)")
    elif A_log is not None or dt_bias is not None:
        raise _Decline("A_log/dt_bias require use_gate_in_kernel")

    def thd(t):
        return t.reshape(-1, *t.shape[2:])

    state = initial_state
    if state is not None:
        state = state.float()
        # Keep a view: the selected engine materializes its own state layout.
        state = state if state_v_first else state.transpose(-1, -2)
    out, final_state = gated_delta_net(
        thd(q), thd(k), thd(v), thd(g).float(), thd(beta), cu, scale=scale,
        initial_state=state, output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval, safe_gate=use_gate_in_kernel,
        a_log=a_log, dt_bias=dt_bias_t, plan_name="gdn_tilelang")
    out = out.reshape(batch, seq, *out.shape[1:])
    if not output_final_state:
        return out, None
    return out, final_state if state_v_first else final_state.transpose(-1, -2).contiguous()


def make_chunk_gated_delta_rule(real_fn):
    """Wrap FLA's ``chunk_gated_delta_rule`` with a native fast path and its fallback."""

    def chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None,
                               output_final_state=False, use_qk_l2norm_in_kernel=False,
                               use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
                               state_v_first=False, cu_seqlens=None, cu_seqlens_cpu=None,
                               cp_context=None, **kwargs):
        A_log = kwargs.get("A_log")
        dt_bias = kwargs.get("dt_bias")
        use_gate_in_kernel = kwargs.get("use_gate_in_kernel", False)

        def fallback(reason):
            _LAST["path"] = f"fallback:{reason}"
            return real_fn(q, k, v, g, beta, scale=scale, initial_state=initial_state,
                           output_final_state=output_final_state,
                           use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                           use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
                           allow_neg_eigval=allow_neg_eigval, state_v_first=state_v_first,
                           cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu,
                           cp_context=cp_context, **kwargs)

        if cp_context is not None:
            return fallback("cp_context")
        if q.device.type != "musa":
            return fallback(f"device:{q.device.type}")
        try:
            out = _to_native(q, k, v, g, beta, scale, initial_state, output_final_state,
                             cu_seqlens, use_qk_l2norm_in_kernel,
                             use_beta_sigmoid_in_kernel, allow_neg_eigval, state_v_first,
                             use_gate_in_kernel, A_log, dt_bias)
        except (_Decline, *_DECLINE) as exc:
            return fallback(str(exc) or type(exc).__name__)
        _LAST["path"] = "native"
        return out

    chunk_gated_delta_rule.__wrapped__ = real_fn
    return chunk_gated_delta_rule


__all__ = ["last_path", "make_chunk_gated_delta_rule"]
