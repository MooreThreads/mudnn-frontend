# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KDA contracts matching cuDNN frontend, backed by native MUSA kernels.

Public gates use natural logs or linear decays. Optional safe-gate parameters,
FP32 gate storage and registered autograd retain cuDNN's dtype/return rules.
Backend engines report unsupported head dimensions explicitly. The historical
six-gradient explicit backward extension keeps its native defaults.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional, Tuple

import torch

import mudnn
from mudnn.engines import decline_types, engines_for

from ..graph_analyzer import analyze_kda

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

fprop_cache: dict = {}

IO_DTYPES = (torch.float16, torch.bfloat16)
GATE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
STATE_DTYPES = (torch.float32, torch.bfloat16)
CU_DTYPES = (torch.int32, torch.int64)


def _plan_key(facts, plan_name):
    """The cache key is the whole facts record.

    Listing fields by hand means a field the engines read but the list omits
    can be varied without invalidating the cache, so a previously accepted call
    would satisfy a later one that should have been declined.  ``KdaFacts`` is
    frozen and hashable, so the record itself is the safe key.
    """
    return (facts, plan_name)


def select_plan(facts, plan_name, cache):
    """The first engine that serves ``facts``, compiled and cached in ``cache``.

    ``plan_name`` pins one engine by name. Raises
    ``cudnnGraphNotSupportedError`` when no engine serves the configuration,
    listing every engine's reason.
    """
    key = _plan_key(facts, plan_name)
    plan = cache.get(key)
    if plan is not None:
        return plan
    engines = engines_for([facts.op])
    if plan_name is not None:
        engines = [e for e in engines if e.name == plan_name]
        if not engines:
            raise mudnn.cudnnGraphNotSupportedError(
                f"no {plan_name!r} engine for the KDA family"
            )
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
        "no engine serves this kimi_delta_attention call"
        + (
            " (" + "; ".join(declined) + ")"
            if declined
            else " (no engine is registered)"
        )
    )


def _check_dtype(name, tensor, want) -> None:
    if tensor is None:
        return
    wants = want if isinstance(want, tuple) else (want,)
    if tensor.dtype not in wants:
        names = " or ".join(str(w) for w in wants)
        raise TypeError(
            f"kimi_delta_attention: {name} must be {names} "
            f"(kernel-native; callers convert), got {tensor.dtype}"
        )


def _reduce_head_grad(gradient, original):
    if gradient.shape == original.shape:
        return gradient
    return (
        gradient.reshape(original.shape[0], original.shape[1], -1, original.shape[2])
        .sum(2, dtype=torch.float32)
        .to(original.dtype)
    )


def _native_gate_params(q, g, a_log, dt_bias):
    a = (
        torch.zeros(g.shape[1], device=q.device)
        if a_log is None
        else a_log.float().contiguous()
    )
    dt = (
        torch.zeros(g.shape[1:], device=q.device)
        if dt_bias is None
        else dt_bias.float().contiguous()
    )
    return a, dt


def _validate_call(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    initial_state,
    safe_gate,
    gate_domain,
    a_log,
    dt_bias,
    checkpoint,
    allow_neg_eigval,
    use_beta_sigmoid,
    state_indices=None,
):
    if any(t.dim() != 3 for t in (q, k, v)):
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if g.dim() != 3:
        raise ValueError("g must be a per-key-channel [total_tokens, HO, K] tensor")
    if beta.dim() != 2:
        raise ValueError("beta must be [total_tokens, HO]")
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a [N + 1] boundary vector")
    if cu_seqlens.dtype not in CU_DTYPES:
        raise ValueError("kimi_delta_attention: cu_seqlens must be int32 or int64")
    if k.shape[1] not in (q.shape[1], v.shape[1]):
        raise ValueError(
            f"k head count ({k.shape[1]}) must match q's ({q.shape[1]}) or v's ({v.shape[1]})"
        )
    ho = max(q.shape[1], v.shape[1])
    if any(ho % t.shape[1] for t in (q, k, v)):
        raise ValueError("q/k/v head counts must divide the output head count")
    if any(t.shape[0] != q.shape[0] for t in (k, v, g, beta)):
        raise ValueError("q/k/v/g/beta must share the total_tokens dimension")
    if k.shape[2] != q.shape[2]:
        raise ValueError("q and k must have the same head dimension")
    if tuple(g.shape[1:]) != (ho, q.shape[2]):
        raise ValueError("g must be per-key-channel [total_tokens, HO, K]")
    if beta.shape[1] != ho:
        raise ValueError("beta must be [total_tokens, HO]")
    for name, t in (("q", q), ("k", k), ("v", v)):
        _check_dtype(name, t, IO_DTYPES)
    _check_dtype("g", g, GATE_DTYPES)
    _check_dtype("beta", beta, (torch.float32, q.dtype))
    _check_dtype("initial_state", initial_state, STATE_DTYPES)
    _check_dtype("a_log", a_log, GATE_DTYPES)
    _check_dtype("dt_bias", dt_bias, GATE_DTYPES)
    if not safe_gate and (a_log is not None or dt_bias is not None):
        raise ValueError("kimi_delta_attention: a_log/dt_bias require safe_gate=True")
    if gate_domain not in ("log", "linear"):
        raise ValueError(
            f"kimi_delta_attention: gate_domain must be 'log' or 'linear', got {gate_domain!r}"
        )
    if gate_domain == "linear" and safe_gate:
        raise ValueError(
            "kimi_delta_attention: gate_domain='linear' cannot combine with safe_gate=True"
        )
    if allow_neg_eigval and not use_beta_sigmoid:
        raise ValueError(
            "kimi_delta_attention: allow_neg_eigval requires use_beta_sigmoid_in_kernel"
        )
    if checkpoint < 0:
        raise ValueError("checkpoint_every_n_tokens must be >= 0")
    n, ho = cu_seqlens.shape[0] - 1, max(q.shape[1], v.shape[1])
    if initial_state is not None:
        if initial_state.dim() != 4 or tuple(initial_state.shape[1:]) != (
            ho,
            v.shape[2],
            q.shape[2],
        ):
            raise ValueError("initial_state must have shape [N, HO, V, K]")
        if state_indices is None and initial_state.shape[0] != n:
            raise ValueError(
                f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {n} sequences"
            )
    if a_log is not None and tuple(a_log.shape) != (ho,):
        raise ValueError(f"a_log must have shape [{ho}]")
    if dt_bias is not None and tuple(dt_bias.shape) != (ho, q.shape[2]):
        raise ValueError(f"dt_bias must have shape [{ho}, {q.shape[2]}]")
    if state_indices is not None:
        _check_dtype("state_indices", state_indices, torch.int32)
        if initial_state is None:
            raise ValueError("state_indices requires initial_state")
        if state_indices.dim() != 1 or state_indices.shape[0] != n:
            raise ValueError(f"state_indices must be [N] = [{n}]")
        if checkpoint:
            raise ValueError(
                "state_indices cannot be combined with checkpoint_every_n_tokens"
            )
    for name, t in (
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("cu_seqlens", cu_seqlens),
        ("initial_state", initial_state),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
        ("state_indices", state_indices),
    ):
        if t is not None and t.device != q.device:
            raise ValueError(
                f"kimi_delta_attention: {name} must be on q's device ({q.device}); got {t.device}"
            )


def _run_forward(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale,
    initial_state,
    output_final_state,
    a_log,
    dt_bias,
    checkpoint_every_n_tokens,
    safe_gate,
    gate_domain,
    lower_bound,
    use_qk_l2norm_in_kernel,
    use_beta_sigmoid_in_kernel,
    state_checkpoints,
    plan_name,
    batch_invariant,
    allow_neg_eigval,
    overwrite_initial_state,
    uses_state_indices,
    state_indices=None,
    final_state_out=None,
):
    with torch.musa.device(q.device) if q.device.type == "musa" else nullcontext():
        if q.shape[0] == 0:
            if cu_seqlens.count_nonzero().item():
                raise ValueError("empty inputs require zero cu_seqlens boundaries")
            shape = (
                cu_seqlens.shape[0] - 1,
                max(q.shape[1], v.shape[1]),
                v.shape[2],
                q.shape[2],
            )
            state = (
                q.new_zeros(shape, dtype=torch.float32)
                if initial_state is None
                else initial_state.clone()
            )
            checkpoints = (
                state.to(q.dtype).clone() if checkpoint_every_n_tokens else None
            )
            final = (
                (final_state_out if final_state_out is not None else state)
                if output_final_state
                else None
            )
            return q.new_empty((0, shape[1], shape[2])), final, checkpoints
        facts = analyze_kda(
            "KDA",
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            initial_state=initial_state,
            output_final_state=output_final_state,
            a_log=a_log,
            dt_bias=dt_bias,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            safe_gate=safe_gate,
            gate_domain=gate_domain,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            lower_bound=lower_bound,
            batch_invariant=batch_invariant,
            allow_neg_eigval=allow_neg_eigval,
            overwrite_initial_state=overwrite_initial_state,
            uses_state_indices=uses_state_indices,
        )
        plan = select_plan(facts, plan_name, fprop_cache)
        if safe_gate:
            a_log, dt_bias = _native_gate_params(q, g, a_log, dt_bias)
        return plan.execute(
            facts,
            None,
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            cu_seqlens=cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            a_log=a_log,
            dt_bias=dt_bias,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            lower_bound=lower_bound,
            state_checkpoints=state_checkpoints,
            state_indices=state_indices,
            final_state_out=final_state_out,
        )


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


@torch.library.custom_op("mudnn::kimi_delta_attention_fwd", mutates_args=())
def kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_call(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        initial_state,
        safe_gate,
        gate_domain,
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        allow_neg_eigval,
        use_beta_sigmoid_in_kernel,
    )
    out = _run_forward(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale,
        initial_state,
        output_final_state,
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        safe_gate,
        gate_domain,
        -5.0 if gate_lower_bound is None else gate_lower_bound,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
        None,
        plan_name,
        batch_invariant,
        allow_neg_eigval,
        False,
        False,
    )
    final = (
        out[1]
        if out[1] is not None
        else q.new_empty(
            0, dtype=torch.float32 if initial_state is None else initial_state.dtype
        )
    )
    return out[0], final, out[2] if out[2] is not None else q.new_empty(0)


@kda_fwd.register_fake
def kda_fwd_fake(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    checkpoint_every_n_tokens=0,
    plan_name: Optional[str] = None,
):
    """Return FakeTensor-compatible output metadata for KDA forward."""
    total, H, K = q.shape
    HK = k.shape[1]
    HV, V = v.shape[1], v.shape[2]
    if HK not in (H, HV):
        raise ValueError(f"k head count ({HK}) must match q's ({H}) or v's ({HV})")
    HO = max(H, HV)
    N = cu_seqlens.shape[0] - 1
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"kimi_delta_attention: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}"
        )
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError(
            "kimi_delta_attention: allow_neg_eigval requires use_beta_sigmoid_in_kernel"
        )
    if gate_domain not in ("log", "linear"):
        raise ValueError(
            f"kimi_delta_attention: gate_domain must be 'log' or 'linear', got {gate_domain!r}"
        )
    if gate_domain == "linear" and safe_gate:
        raise ValueError(
            "kimi_delta_attention: gate_domain='linear' cannot combine with safe_gate=True"
        )
    if initial_state is not None and initial_state.shape[0] != N:
        raise ValueError(
            f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences"
        )
    o = q.new_empty(total, HO, V)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    final = q.new_empty(
        (N, HO, V, K) if output_final_state else (0,), dtype=state_dtype
    )
    if checkpoint_every_n_tokens > 0:
        total_checkpoints = max(total // int(checkpoint_every_n_tokens) + N, 1)
        state_checkpoints = q.new_empty(total_checkpoints, HO, V, K)
    else:
        state_checkpoints = q.new_empty(0)
    return o, final, state_checkpoints


@torch.library.custom_op(
    "mudnn::kimi_delta_attention_fwd_overwrite_state", mutates_args=("initial_state",)
)
def kda_fwd_overwrite_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
    state_indices: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_call(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        initial_state,
        safe_gate,
        gate_domain,
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        allow_neg_eigval,
        use_beta_sigmoid_in_kernel,
        state_indices,
    )
    out = _run_forward(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale,
        initial_state,
        output_final_state,
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        safe_gate,
        gate_domain,
        -5.0 if gate_lower_bound is None else gate_lower_bound,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
        None,
        plan_name,
        batch_invariant,
        allow_neg_eigval,
        True,
        state_indices is not None,
        state_indices=state_indices,
        final_state_out=initial_state if output_final_state else None,
    )
    return out[0], out[2] if out[2] is not None else q.new_empty(0)


@kda_fwd_overwrite_state.register_fake
def kda_fwd_overwrite_state_fake(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale,
    initial_state,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    checkpoint_every_n_tokens=0,
    plan_name=None,
    state_indices=None,
    output_final_state=True,
):
    """Return fake outputs for the in-place KDA forward custom op."""
    o, _, state_checkpoints = kda_fwd_fake(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        scale=scale,
        initial_state=initial_state if state_indices is None else None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        batch_invariant=batch_invariant,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        gate_domain=gate_domain,
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        plan_name=plan_name,
        output_final_state=bool(output_final_state),
    )
    return o, state_checkpoints


# ---------------------------------------------------------------------------
# Backward (MP31 chunk KDA)
# ---------------------------------------------------------------------------


@torch.library.custom_op("mudnn::_kimi_delta_attention_bwd_raw", mutates_args=())
def _kda_bwd_raw(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    d_final_state: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    lower_bound: float = -5.0,
    use_qk_l2norm_in_kernel: bool = True,
    plan_name: Optional[str] = None,
    safe_gate: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    gate_domain: str = "log2",
    precise_state_scan: bool = False,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    with torch.musa.device(q.device) if q.device.type == "musa" else nullcontext():
        """Run the vendored MP31 chunk-KDA backward pipeline."""
        from ..mutlass.kda_backward import chunk_kda_bwd

        return chunk_kda_bwd(
            dO,
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            d_final_state=d_final_state,
            cu_seqlens=cu_seqlens,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            plan_name=plan_name,
            safe_gate=safe_gate,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            gate_domain=gate_domain,
            precise_state_scan=precise_state_scan,
        )


@_kda_bwd_raw.register_fake
def _kda_bwd_raw_fake(
    dO,
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale,
    d_final_state=None,
    initial_state=None,
    a_log=None,
    dt_bias=None,
    lower_bound=-5.0,
    use_qk_l2norm_in_kernel=True,
    plan_name=None,
    safe_gate=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    gate_domain="log2",
    precise_state_scan=False,
):
    n = cu_seqlens.shape[0] - 1
    return (
        q.new_empty(q.shape),
        k.new_empty(k.shape),
        v.new_empty(v.shape),
        g.new_empty(g.shape, dtype=torch.float32),
        beta.new_empty(beta.shape, dtype=torch.float32),
        (
            q.new_empty((n, v.shape[-2], v.shape[-1], q.shape[-1]), dtype=torch.float32)
            if initial_state is not None
            else q.new_empty(0, dtype=torch.float32)
        ),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@torch.library.custom_op("mudnn::kimi_delta_attention_bwd", mutates_args=())
def kda_bwd(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    d_final_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    plan_name: Optional[str] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    _validate_call(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        initial_state,
        safe_gate,
        gate_domain,
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        allow_neg_eigval,
        use_beta_sigmoid_in_kernel,
    )
    _check_dtype("dO", dO, q.dtype)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    if d_final_state is not None and d_final_state.dtype != state_dtype:
        raise TypeError(
            f"kimi_delta_attention: d_final_state must be {state_dtype} "
            "(one state dtype per kernel)"
        )
    if state_checkpoints is not None:
        _check_dtype("state_checkpoints", state_checkpoints, q.dtype)
    if q.shape[0] == 0:
        if cu_seqlens.count_nonzero().item():
            raise ValueError("empty inputs require zero cu_seqlens boundaries")
        ds = (
            q.new_empty(0, dtype=torch.float32)
            if initial_state is None
            else (
                torch.zeros_like(initial_state)
                if d_final_state is None
                else d_final_state.clone()
            )
        )
        return (
            *(torch.zeros_like(t) for t in (q, k, v, g, beta)),
            ds,
            q.new_empty(0, dtype=torch.float32)
            if a_log is None
            else torch.zeros_like(a_log),
            q.new_empty(0, dtype=torch.float32)
            if dt_bias is None
            else torch.zeros_like(dt_bias),
        )
    native_a, native_dt = (
        _native_gate_params(q, g, a_log, dt_bias) if safe_gate else (None, None)
    )
    # shortcut: backward rebuilds checkpoints; reuse them when the native scan accepts a cadence.
    heads = g.shape[1]
    nq, nk, nv = (t.repeat_interleave(heads // t.shape[1], dim=1) for t in (q, k, v))
    dq, dk, dv, dg, dbeta, ds = torch.ops.mudnn._kimi_delta_attention_bwd_raw(
        dO.contiguous(),
        nq.contiguous(),
        nk.contiguous(),
        nv.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        cu_seqlens.contiguous(),
        scale,
        initial_state=None if initial_state is None else initial_state.contiguous(),
        d_final_state=None
        if d_final_state is None
        else d_final_state.float().contiguous(),
        a_log=native_a,
        dt_bias=native_dt,
        lower_bound=-5.0 if gate_lower_bound is None else gate_lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        plan_name=plan_name,
        safe_gate=safe_gate,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        gate_domain=gate_domain,
        precise_state_scan=True,
    )
    dq, dk, dv = (_reduce_head_grad(t, x) for t, x in zip((dq, dk, dv), (q, k, v)))
    da = (
        q.new_empty(0, dtype=torch.float32)
        if a_log is None
        else (dg * (g.float() + native_dt)).sum((0, 2)).to(a_log.dtype)
    )
    ddt = (
        q.new_empty(0, dtype=torch.float32)
        if dt_bias is None
        else dg.sum(0).to(dt_bias.dtype)
    )
    ds = (
        q.new_empty(0, dtype=torch.float32)
        if initial_state is None
        else ds.to(initial_state.dtype)
    )
    return dq, dk, dv, dg.to(g.dtype), dbeta.to(beta.dtype), ds, da, ddt


@kda_bwd.register_fake
def kda_bwd_fake(
    dO,
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale,
    initial_state=None,
    d_final_state=None,
    state_checkpoints=None,
    checkpoint_every_n_tokens=0,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    plan_name=None,
):
    """Return FakeTensor-compatible output metadata for KDA backward."""
    dstate_dtype = initial_state.dtype if initial_state is not None else torch.float32
    if d_final_state is not None and d_final_state.dtype != dstate_dtype:
        raise TypeError(
            f"kimi_delta_attention: d_final_state must be {dstate_dtype} (one state dtype per kernel)"
        )
    dstate0 = (
        initial_state.new_empty(initial_state.shape, dtype=dstate_dtype)
        if initial_state is not None
        else q.new_empty(0, dtype=dstate_dtype)
    )
    d_a_log = (
        torch.empty_like(a_log)
        if a_log is not None
        else q.new_empty(0, dtype=torch.float32)
    )
    d_dt_bias = (
        torch.empty_like(dt_bias)
        if dt_bias is not None
        else q.new_empty(0, dtype=torch.float32)
    )
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(beta),
        dstate0,
        d_a_log,
        d_dt_bias,
    )


def kda_setup_context(ctx, inputs, output):
    """Save KDA forward inputs and static options needed by autograd."""
    (
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_lower_bound,
        gate_domain,
        a_log,
        dt_bias,
        checkpoint_every_n_tokens,
        plan_name,
    ) = inputs
    saved = [q, k, v, g, beta, cu_seqlens]
    ctx.checkpoint_reuse = (
        checkpoint_every_n_tokens > 0
        and checkpoint_every_n_tokens % 16 == 0
        and output[2].numel() > 0
    )
    ctx.checkpoint_every_n_tokens = (
        checkpoint_every_n_tokens if ctx.checkpoint_reuse else 0
    )
    if ctx.checkpoint_reuse:
        saved.append(output[2])
    ctx.has_a_log = a_log is not None
    ctx.has_dt_bias = dt_bias is not None
    if ctx.has_a_log:
        saved.append(a_log)
    if ctx.has_dt_bias:
        saved.append(dt_bias)
    ctx.save_for_backward(*saved)
    ctx.initial_state = initial_state
    ctx.scale = scale
    ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
    ctx.batch_invariant = batch_invariant
    ctx.plan_name = plan_name
    ctx.use_beta_sigmoid_in_kernel = bool(use_beta_sigmoid_in_kernel)
    ctx.allow_neg_eigval = bool(allow_neg_eigval)
    ctx.safe_gate = bool(safe_gate)
    ctx.gate_lower_bound = gate_lower_bound
    ctx.gate_domain = str(gate_domain)
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(output[2])


def kda_backward(ctx, dO, dFinal, dstate_checkpoints):
    """Apply the registered first-order KDA backward formula."""
    # Non-reentrant activation checkpointing allows one saved-tensor unpack.
    saved_tensors = ctx.saved_tensors
    if ctx.checkpoint_reuse:
        q, k, v, g, beta, cu_seqlens, state_checkpoints = saved_tensors[:7]
        gate_params = list(saved_tensors[7:])
    else:
        q, k, v, g, beta, cu_seqlens = saved_tensors[:6]
        state_checkpoints = None
        gate_params = list(saved_tensors[6:])
    a_log = gate_params.pop(0) if ctx.has_a_log else None
    dt_bias = gate_params.pop(0) if ctx.has_dt_bias else None
    initial_state = ctx.initial_state

    if dO is None:
        dO = torch.zeros(
            q.shape[0],
            max(q.shape[1], v.shape[1]),
            v.shape[2],
            dtype=q.dtype,
            device=q.device,
        )
    dstate_in = dFinal if (dFinal is not None and dFinal.numel() > 0) else None
    (
        dq,
        dk,
        dv,
        dg,
        dbeta,
        dstate0,
        d_a_log,
        d_dt_bias,
    ) = torch.ops.mudnn.kimi_delta_attention_bwd(
        dO,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        ctx.scale,
        initial_state=initial_state,
        d_final_state=dstate_in,
        state_checkpoints=state_checkpoints,
        checkpoint_every_n_tokens=ctx.checkpoint_every_n_tokens,
        use_qk_l2norm_in_kernel=ctx.use_qk_l2norm_in_kernel,
        batch_invariant=ctx.batch_invariant,
        use_beta_sigmoid_in_kernel=ctx.use_beta_sigmoid_in_kernel,
        allow_neg_eigval=ctx.allow_neg_eigval,
        safe_gate=ctx.safe_gate,
        gate_lower_bound=ctx.gate_lower_bound,
        gate_domain=ctx.gate_domain,
        a_log=a_log,
        dt_bias=dt_bias,
        plan_name=ctx.plan_name,
    )
    return (
        dq,
        dk,
        dv,
        dg,
        dbeta,
        None,
        None,
        dstate0 if initial_state is not None else None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        d_a_log if ctx.has_a_log else None,
        d_dt_bias if ctx.has_dt_bias else None,
        None,
        None,
    )


torch.library.register_autograd(
    "mudnn::kimi_delta_attention_fwd", kda_backward, setup_context=kda_setup_context
)


def kimi_delta_attention_bwd(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    d_final_state: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    gate_lower_bound: float = -5.0,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, ...]:
    """Gradients of KDA for the MP31 chunk backend.

    The backward mirrors the frontend's packed THD layout and returns
    ``(dq, dk, dv, dg, dbeta, d_initial_state)``.  The migrated kernel currently
    requires the safe-gate ``a_log`` and ``dt_bias`` tensors.
    """
    tensors = (dO, q, k, v, g)
    if any(t.dim() != 3 for t in tensors):
        raise ValueError("dO, q, k, v and g must be packed THD tensors")
    if beta.dim() != 2:
        raise ValueError("beta must be [total_tokens, HO]")
    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a [N + 1] boundary vector")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if not math.isfinite(float(scale)):
        raise ValueError(f"scale must be finite, got {scale}")
    if (a_log is None) != (dt_bias is None):
        raise ValueError("a_log and dt_bias must be provided together")
    if a_log is None:
        raise NotImplementedError(
            "kimi_delta_attention_bwd currently supports the safe-gate path only; "
            "provide a_log and dt_bias"
        )
    return torch.ops.mudnn._kimi_delta_attention_bwd_raw(
        dO.contiguous(),
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        cu_seqlens.contiguous(),
        float(scale),
        d_final_state=d_final_state,
        initial_state=initial_state,
        a_log=a_log,
        dt_bias=dt_bias,
        lower_bound=float(gate_lower_bound),
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        plan_name=plan_name,
    )


def kimi_delta_attention(
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
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
    overwrite_initial_state: bool = False,
    state_indices: Optional[torch.Tensor] = None,
):
    """Kimi Delta Attention (KDA) linear attention.

    THD layout (matches the graph-API KDA node):

        q: ``[total_tokens, H, K]``; k: ``[total_tokens, HK, K]`` (HK = H or
        HK = HV); v: ``[total_tokens, HV, V]``
        g: ``[total_tokens, HO, K]`` (per-key-channel log decay);
        beta: ``[total_tokens, HO]`` (scalar); cu_seqlens: ``[N+1]`` int32
        initial_state / final_state: ``[N, HO, V, K]``
        (``HO = max(H, HV)``, the gates, output, and state heads)

    A dense batch of N equal-length sequences is expressed as
    ``cu_seqlens = [0, T, 2T, ...]`` over the flattened tokens.

    Dtypes are kernel-native and strict (callers convert).  ``g`` is float32,
    bfloat16 or float16 and ``dG`` comes back in the same dtype.  ``beta`` and
    ``dBeta`` are float32 or the io dtype (raw logits of either dtype under
    ``use_beta_sigmoid_in_kernel``).  ``initial_state`` is float32 or bfloat16
    and ``final_state`` and the state gradients (``d_final_state``,
    ``d_initial_state``) follow it.

    Args:
        g: per-key-channel decay: log-space by default (``alpha = exp(g) in
            (0, 1]^K``), the linear decay ``alpha`` itself when
            ``gate_domain="linear"``, or raw pre-activation logits when
            ``safe_gate=True``.
        beta: per-token scalar write strength (float32 or the io dtype,
            post-sigmoid), or float32/io-dtype logits when
            ``use_beta_sigmoid_in_kernel=True``.
        cu_seqlens: ``[N+1]`` int32/int64 sequence boundaries over the packed tokens.
        scale: attention scale applied to ``q``. Defaults to ``1 / sqrt(K)``.
        initial_state: optional recurrent state (otherwise zero); float32 or
            bfloat16, and ``final_state`` comes back in the same dtype.
        output_final_state: if ``True``, also return the per-sequence state
            after the last token.
        use_qk_l2norm_in_kernel: if ``True``, L2-normalize the Q/K rows inside
            the kernel (the KDA model's feature map); if ``False``, pass Q/K
            as given (the caller owns their conditioning).
        batch_invariant: if ``True``, each sequence's results are bitwise
            independent of the batch composition (whole-sequence scheduling;
            disables split-K load balancing).
        overwrite_initial_state: if ``True``, ``final_state`` is written
            into the ``initial_state`` buffer, which is required and
            returned as the final state; the engine runs the chain or the
            uncut schedule and never the split-K cut, so the aliasing is
            race-free.  Inference only: the in-place op has no autograd
            formula.
        use_beta_sigmoid_in_kernel: apply ``sigmoid(beta)`` inside the kernel;
            the backward returns the raw-logit beta gradient.
        allow_neg_eigval: scale the fused beta sigmoid by 2, so the delta-rule
            operator ``I - beta k k^T`` can reach negative eigenvalues
            (a reflection rather than a projection). Requires
            ``use_beta_sigmoid_in_kernel``.
        safe_gate: interpret ``g`` through the safe-gate transform
            ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``.
            ``a_log`` and ``dt_bias`` are optional; an absent ``a_log`` is
            unit amplitude (``exp(a_log) = 1``), an absent ``dt_bias`` is zero
            bias, and an absent parameter gets no gradient. The backward
            returns the raw-logit ``g`` gradient plus the gradients of the
            parameters that were given. Passing either without
            ``safe_gate=True`` is an error.
        gate_lower_bound: safe-gate lower bound in log space (default -5.0).
        gate_domain: ``"log"`` (default): ``g`` holds ``ln(alpha)``;
            ``"linear"``: ``g`` holds ``alpha in (0, 1]^K`` (a 1e-10 floor is
            applied in-kernel) and ``dG`` comes back with respect to ``alpha``.
            Cannot combine with ``safe_gate``.
        a_log: ``[HO]`` safe-gate per-head log-amplitude (float32, bfloat16 or
            float16; ``d_a_log`` comes back in the same dtype), or ``None``
            for unit amplitude.
        dt_bias: ``[HO, K]`` safe-gate channel bias (float32, bfloat16 or
            float16; ``d_dt_bias`` comes back in the same dtype), or ``None``
            for zero bias.
        checkpoint_every_n_tokens: if ``> 0``, also return the per-chunk
            recurrent state series ``state_checkpoints`` (``[total_checkpoints, HO, V, K]`` io dtype,
            one entry per N tokens strictly before each sequence end; the
            mutlass engine requires a positive multiple of the kernel chunk size, 16). The series is
            a non-differentiable dump.

        plan_name: optionally pin one execution plan by name (the plan
            API's ``get_plan_name_at_index`` names, e.g. ``kda_frost``); a
            graph offering no such plan raises ``cudnnGraphNotSupportedError``.
        state_indices: optional ``[N]`` int32 table making ``initial_state`` a
            pool ``[N_pool, HO, V, K]``: sequence ``i`` reads row
            ``state_indices[i]`` and, under ``output_final_state``, writes its
            final state back to that row in place, so a serving stack keeps
            its paged state pool without a gather and scatter around the
            call.  Implies ``overwrite_initial_state``.  Forward only; cannot
            combine with ``checkpoint_every_n_tokens``.

    Returns:
        ``(o, final_state)`` with ``o`` shaped like ``v``, or
        ``(o, final_state, state_checkpoints)`` when ``checkpoint_every_n_tokens > 0``.
        ``final_state`` is empty unless ``output_final_state=True``; under
        ``overwrite_initial_state`` or ``state_indices`` it is the caller's buffer.
    """
    if q.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if state_indices is not None and initial_state is None:
        raise ValueError(
            "kimi_delta_attention: state_indices selects rows of the initial_state pool, so initial_state is required"
        )
    if overwrite_initial_state or state_indices is not None:
        if initial_state is None:
            raise ValueError(
                "kimi_delta_attention: overwrite_initial_state requires initial_state"
            )
        o, state_checkpoints = torch.ops.mudnn.kimi_delta_attention_fwd_overwrite_state(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            float(scale),
            initial_state=initial_state,
            use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
            batch_invariant=bool(batch_invariant),
            use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
            allow_neg_eigval=bool(allow_neg_eigval),
            safe_gate=bool(safe_gate),
            gate_lower_bound=float(gate_lower_bound)
            if gate_lower_bound is not None
            else None,
            gate_domain=str(gate_domain),
            a_log=a_log,
            dt_bias=dt_bias,
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
            plan_name=plan_name,
            state_indices=state_indices,
            output_final_state=bool(output_final_state)
            or bool(overwrite_initial_state),
        )
        final_state = (
            initial_state
            if (output_final_state or overwrite_initial_state)
            else initial_state.new_empty(0)
        )
    else:
        o, final_state, state_checkpoints = torch.ops.mudnn.kimi_delta_attention_fwd(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            float(scale),
            initial_state=initial_state,
            output_final_state=bool(output_final_state),
            use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
            batch_invariant=bool(batch_invariant),
            use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
            allow_neg_eigval=bool(allow_neg_eigval),
            safe_gate=bool(safe_gate),
            gate_lower_bound=float(gate_lower_bound)
            if gate_lower_bound is not None
            else None,
            gate_domain=str(gate_domain),
            a_log=a_log,
            dt_bias=dt_bias,
            checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
            plan_name=plan_name,
        )
    if checkpoint_every_n_tokens > 0:
        return o, final_state, state_checkpoints
    return o, final_state


def _summary_transition(k, g, beta, cu_seqlens, **opts):
    # An identity seed and zero values give the recurrence's linear map.
    heads, dim = g.shape[1], k.shape[2]
    identity = (
        torch.eye(dim, device=k.device)
        .expand(cu_seqlens.shape[0] - 1, heads, dim, dim)
        .contiguous()
    )
    values = k.new_zeros(k.shape[0], heads, dim)
    _, transition, _ = torch.ops.mudnn.kimi_delta_attention_fwd(
        torch.zeros_like(k),
        k,
        values,
        g,
        beta,
        cu_seqlens,
        1.0,
        initial_state=identity,
        output_final_state=True,
        **opts,
    )
    return transition


def _summary_setup_context(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(*output)


def _summary_backward(ctx, *grad_outputs):
    raise RuntimeError("summaries are non-differentiable")


@torch.library.custom_op("mudnn::kimi_delta_attention_summary", mutates_args=())
def kda_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_transition: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_dtype("initial_state", initial_state, torch.float32)
    if k.shape[1] != g.shape[1]:
        k = k.repeat_interleave(g.shape[1] // k.shape[1], dim=1)
    opts = dict(
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        gate_domain=gate_domain,
        a_log=a_log,
        dt_bias=dt_bias,
        batch_invariant=batch_invariant,
        plan_name=plan_name,
    )
    _, state, _ = torch.ops.mudnn.kimi_delta_attention_fwd(
        torch.zeros_like(k),
        k,
        v,
        g,
        beta,
        cu_seqlens,
        1.0,
        initial_state=initial_state,
        output_final_state=True,
        **opts,
    )
    transition = (
        _summary_transition(k, g, beta, cu_seqlens, **opts)
        if output_transition
        else k.new_empty(0, dtype=torch.float32)
    )
    return state, transition


@kda_summary.register_fake
def kda_summary_fake(
    k,
    v,
    g,
    beta,
    cu_seqlens,
    initial_state=None,
    output_transition=False,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    batch_invariant=False,
    plan_name: Optional[str] = None,
):
    """Return FakeTensor-compatible outputs for a KDA forward summary."""
    total, HK, K = k.shape
    HV, V = v.shape[1], v.shape[2]
    HO = g.shape[1]
    N = cu_seqlens.shape[0] - 1
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"kimi_delta_attention_summary: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}"
        )
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError(
            "kimi_delta_attention_summary: allow_neg_eigval requires use_beta_sigmoid_in_kernel"
        )
    if gate_domain not in ("log", "linear"):
        raise ValueError(
            f"kimi_delta_attention_summary: gate_domain must be 'log' or 'linear', got {gate_domain!r}"
        )
    if gate_domain == "linear" and safe_gate:
        raise ValueError(
            "kimi_delta_attention_summary: gate_domain='linear' cannot combine with safe_gate=True"
        )
    if initial_state is not None and initial_state.shape[0] != N:
        raise ValueError(
            f"initial_state must carry one state per sequence: got {initial_state.shape[0]} for {N} sequences"
        )
    final_state = k.new_empty((N, HO, V, K), dtype=torch.float32)
    transition = k.new_empty(
        (N, HO, K, K) if output_transition else (0,), dtype=torch.float32
    )
    return final_state, transition


@torch.library.custom_op("mudnn::kimi_delta_attention_summary_bwd", mutates_args=())
def kda_summary_bwd(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    d_final_state: Optional[torch.Tensor] = None,
    output_transition: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.musa.device(q.device) if q.device.type == "musa" else nullcontext():
        _validate_call(
            q,
            k,
            dO,
            g,
            beta,
            cu_seqlens,
            None,
            safe_gate,
            gate_domain,
            a_log,
            dt_bias,
            0,
            allow_neg_eigval,
            use_beta_sigmoid_in_kernel,
        )
        _check_dtype("d_final_state", d_final_state, torch.float32)
        if q.shape[0] == 0:
            if cu_seqlens.count_nonzero().item():
                raise ValueError("empty inputs require zero cu_seqlens boundaries")
            shape = (cu_seqlens.shape[0] - 1, g.shape[1], dO.shape[2], k.shape[2])
            state = (
                q.new_zeros(shape, dtype=torch.float32)
                if d_final_state is None
                else d_final_state.clone()
            )
            transition = (
                torch.eye(k.shape[2], device=q.device)
                .expand(shape[0], shape[1], -1, -1)
                .clone()
                if output_transition
                else q.new_empty(0, dtype=torch.float32)
            )
            return state, transition
        facts = analyze_kda(
            "KDA",
            q,
            k,
            dO,
            g,
            beta,
            cu_seqlens,
            safe_gate=safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        select_plan(facts, plan_name, fprop_cache)
        native_a, native_dt = (
            _native_gate_params(q, g, a_log, dt_bias) if safe_gate else (None, None)
        )
        heads = g.shape[1]
        q, k, dO = (t.repeat_interleave(heads // t.shape[1], dim=1) for t in (q, k, dO))
        values = torch.zeros_like(dO)
        zero_state = q.new_zeros(
            cu_seqlens.shape[0] - 1, g.shape[1], dO.shape[2], k.shape[2]
        )
        result = torch.ops.mudnn._kimi_delta_attention_bwd_raw(
            dO.contiguous(),
            q.contiguous(),
            k.contiguous(),
            values,
            g.contiguous(),
            beta.contiguous(),
            cu_seqlens.contiguous(),
            scale,
            initial_state=zero_state,
            d_final_state=d_final_state,
            a_log=native_a,
            dt_bias=native_dt,
            lower_bound=-5.0 if gate_lower_bound is None else gate_lower_bound,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            plan_name=plan_name,
            safe_gate=safe_gate,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            gate_domain=gate_domain,
        )
        opts = dict(
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            gate_lower_bound=gate_lower_bound,
            gate_domain=gate_domain,
            a_log=a_log,
            dt_bias=dt_bias,
            batch_invariant=batch_invariant,
            plan_name=plan_name,
        )
        transition = (
            _summary_transition(k, g, beta, cu_seqlens, **opts)
            .transpose(-1, -2)
            .contiguous()
            if output_transition
            else k.new_empty(0, dtype=torch.float32)
        )
        return result[5], transition


@kda_summary_bwd.register_fake
def kda_summary_bwd_fake(
    dO,
    q,
    k,
    g,
    beta,
    cu_seqlens,
    scale,
    d_final_state=None,
    output_transition=False,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_lower_bound=None,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    batch_invariant=False,
    plan_name: Optional[str] = None,
):
    """Return FakeTensor-compatible outputs for a KDA backward summary."""
    HV, V = dO.shape[1], dO.shape[2]
    K = k.shape[2]
    HO = g.shape[1]
    N = cu_seqlens.shape[0] - 1
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"kimi_delta_attention_summary_bwd: cu_seqlens must be int32 or int64; got {cu_seqlens.dtype}"
        )
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError(
            "kimi_delta_attention_summary_bwd: allow_neg_eigval requires use_beta_sigmoid_in_kernel"
        )
    if gate_domain not in ("log", "linear"):
        raise ValueError(
            f"kimi_delta_attention_summary_bwd: gate_domain must be 'log' or 'linear', got {gate_domain!r}"
        )
    if gate_domain == "linear" and safe_gate:
        raise ValueError(
            "kimi_delta_attention_summary_bwd: gate_domain='linear' cannot combine with safe_gate=True"
        )
    if d_final_state is not None and tuple(d_final_state.shape) != (N, HO, V, K):
        raise ValueError(
            f"kimi_delta_attention_summary_bwd: d_final_state must be [N, HO, V, K] = [{N}, {HO}, {V}, {K}]; got {tuple(d_final_state.shape)}"
        )
    d_initial_state = dO.new_empty((N, HO, V, K), dtype=torch.float32)
    transition = dO.new_empty(
        (N, HO, K, K) if output_transition else (0,), dtype=torch.float32
    )
    return d_initial_state, transition


def kimi_delta_attention_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
):
    """Kimi Delta Attention (KDA) per-span summary for context parallelism: the state-only pass over each sequence, returning
    ``(H, M)`` float32 with ``H`` ``[N, HO, V, K]`` the final state under ``initial_state`` (zero when ``None``) and ``M``
    ``[N, HO, K, K]`` the span transition in stored domain (``M_buf = M^T``), so that across spans ``X_final = X_init @
    M_buf + X_H`` and ``X_dh0 = X_dht @ M_buf^T + X_G``.  Both outputs are non-differentiable.  THD layout and gate
    semantics match :func:`kimi_delta_attention` (k ``[total_tokens, HK, K]``, v ``[total_tokens, HV, V]``, g
    ``[total_tokens, HO, K]`` per-key-channel log decay, beta ``[total_tokens, HO]``, cu_seqlens ``[N+1]``); ``initial_state`` is float32.

    Args:
        use_beta_sigmoid_in_kernel: apply ``sigmoid(beta)`` in the kernel; ``beta`` then holds raw logits.
        safe_gate: ``g`` is read as ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``; ``a_log`` ``[HO]`` and
            ``dt_bias`` ``[HO, K]`` (fp32/bf16/fp16) default to unit amplitude / zero bias and require ``safe_gate=True``.
        gate_lower_bound: safe-gate lower bound in log space (default -5.0).
        gate_domain: ``"log"`` (``g = ln(alpha)``) or ``"linear"`` (``g = alpha in (0, 1]^K``, 1e-10 floor, ``dG`` with respect
            to ``alpha``); not with ``safe_gate``.
        plan_name: pin one execution plan by name (e.g. ``kda_mutlass_mp32``).
    """
    if k.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    final_state, transition = torch.ops.mudnn.kimi_delta_attention_summary(
        k,
        v,
        g,
        beta,
        cu_seqlens,
        initial_state=initial_state,
        output_transition=True,
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
        allow_neg_eigval=bool(allow_neg_eigval),
        safe_gate=bool(safe_gate),
        gate_lower_bound=float(gate_lower_bound)
        if gate_lower_bound is not None
        else None,
        gate_domain=str(gate_domain),
        a_log=a_log,
        dt_bias=dt_bias,
        batch_invariant=bool(batch_invariant),
        plan_name=plan_name,
    )
    return final_state, transition


def kimi_delta_attention_summary_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dO: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: Optional[float] = None,
    d_final_state: Optional[torch.Tensor] = None,
    output_transition: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    batch_invariant: bool = False,
    plan_name: Optional[str] = None,
):
    """Kimi Delta Attention (KDA) per-span backward summary for context parallelism: the reverse state-gradient recurrence
    over each sequence, returning ``G`` ``[N, HO, V, K]`` float32, the incoming state gradient ``d_initial_state`` under
    ``d_final_state`` (the pure summary term with ``None``, the full ``dh0`` otherwise), and with ``output_transition=True``
    also ``transition`` ``[N, HO, K, K]`` float32, the span transition in the backward's stored-domain orientation
    (``M_buf^T``) so that ``X_dh0 = X_dht @ transition + X_G``.  The outputs are non-differentiable.  THD layout and gate
    semantics match :func:`kimi_delta_attention` (q ``[total_tokens, HQ, K]``, k ``[total_tokens, HK, K]``, g
    ``[total_tokens, HO, K]``, beta ``[total_tokens, HO]``, dO ``[total_tokens, HO, V]`` in the io dtype, cu_seqlens
    ``[N+1]``); ``d_final_state`` is float32 ``[N, HO, V, K]``.

    Args:
        scale: attention scale applied to ``q``; defaults to ``1 / sqrt(K)``.
        use_beta_sigmoid_in_kernel: apply ``sigmoid(beta)`` in the kernel; ``beta`` then holds raw logits.
        safe_gate: ``g`` is read as ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``; ``a_log`` ``[HO]`` and
            ``dt_bias`` ``[HO, K]`` (fp32/bf16/fp16) default to unit amplitude / zero bias and require ``safe_gate=True``.
        gate_lower_bound: safe-gate lower bound in log space (default -5.0).
        gate_domain: ``"log"`` (``g = ln(alpha)``) or ``"linear"`` (``g = alpha in (0, 1]^K``, 1e-10 floor, ``dG`` with respect
            to ``alpha``); not with ``safe_gate``.
        plan_name: pin one execution plan by name (e.g. ``kda_mutlass_mp32``).
    """
    if q.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    d_initial_state, transition = torch.ops.mudnn.kimi_delta_attention_summary_bwd(
        dO,
        q,
        k,
        g,
        beta,
        cu_seqlens,
        float(scale),
        d_final_state=d_final_state,
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
        allow_neg_eigval=bool(allow_neg_eigval),
        safe_gate=bool(safe_gate),
        gate_lower_bound=float(gate_lower_bound)
        if gate_lower_bound is not None
        else None,
        gate_domain=str(gate_domain),
        a_log=a_log,
        dt_bias=dt_bias,
        batch_invariant=bool(batch_invariant),
        output_transition=bool(output_transition),
        plan_name=plan_name,
    )
    return (d_initial_state, transition) if output_transition else d_initial_state


torch.library.register_autograd(
    "mudnn::kimi_delta_attention_summary",
    _summary_backward,
    setup_context=_summary_setup_context,
)
torch.library.register_autograd(
    "mudnn::kimi_delta_attention_summary_bwd",
    _summary_backward,
    setup_context=_summary_setup_context,
)

__all__ = [
    "kimi_delta_attention",
    "kimi_delta_attention_summary",
    "kimi_delta_attention_summary_bwd",
    "kimi_delta_attention_bwd",
    "select_plan",
]
