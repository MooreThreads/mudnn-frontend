"""cuDNN-frontend-shaped GDN API and inference-only custom operators.

Contract baseline: NVIDIA/cudnn-frontend 33c6fcb54d11.
"""

import math
import threading
from typing import Optional, Tuple

import torch

import mudnn
from mudnn.engines import decline_types, engines_for

from ..tilelang.gdn_inference_common import InferenceFacts
from ..tilelang.kernels.inference.helpers import get_arch

_plans = {}
_plan_lock = threading.Lock()


def _validate(q, k, v, g, beta, cu, initial_state, indices, a_log, dt_bias, flags):
    if any(t.dim() != 3 for t in (q, k, v)):
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("q/k/v must agree on tokens and q/k head dimensions")
    h, hv, hk = q.shape[1], v.shape[1], k.shape[1]
    ho = max(h, hv)
    if min(h, hk, hv, q.shape[-1], v.shape[-1]) <= 0:
        raise ValueError("head counts and head dimensions must be positive")
    if hk not in (h, hv) or ho % h or ho % hv:
        raise ValueError("incompatible q/k/v head counts")
    if q.dtype not in (torch.float16, torch.bfloat16) or not (q.dtype == k.dtype == v.dtype):
        raise TypeError("gated_delta_net: q/k/v must share a fp16 or bf16 dtype")
    for name, t, dtypes in (
        ("g", g, (torch.float32, torch.float16, torch.bfloat16)),
        ("beta", beta, (torch.float32, q.dtype)),
    ):
        if t.shape != (q.shape[0], ho):
            raise ValueError(f"gated_delta_net: {name} must be [total_tokens, HO]")
        if t.dtype not in dtypes:
            raise TypeError(f"gated_delta_net: {name} has unsupported dtype {t.dtype}")
    if cu.dim() != 1 or cu.numel() < 2 or cu.dtype not in (torch.int32, torch.int64):
        raise ValueError("gated_delta_net: cu_seqlens must be int32/int64 [N+1]")
    if flags["gate_domain"] not in ("log", "linear"):
        raise ValueError("gate_domain must be 'log' or 'linear'")
    if flags["safe_gate"] and flags["gate_domain"] == "linear":
        raise ValueError("gate_domain='linear' cannot combine with safe_gate")
    if flags["allow_neg_eigval"] and not flags["use_beta_sigmoid_in_kernel"]:
        raise ValueError("allow_neg_eigval requires use_beta_sigmoid_in_kernel")
    for name, t in (("a_log", a_log), ("dt_bias", dt_bias)):
        if t is not None:
            if not flags["safe_gate"]:
                raise ValueError("a_log/dt_bias require safe_gate=True")
            if t.shape != (ho,):
                raise ValueError(f"{name} must be [HO]")
            if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise TypeError(f"{name} must be fp32/bf16/fp16")
            if t.stride(0) != 1:
                raise NotImplementedError(f"{name} requires stride 1")
    checkpoint = flags["checkpoint_every_n_tokens"]
    if checkpoint < 0 or checkpoint % 64:
        raise ValueError("checkpoint_every_n_tokens must be 0 or a positive multiple of 64")
    n = cu.numel() - 1
    if initial_state is not None:
        expected_tail = (ho, v.shape[-1], q.shape[-1])
        if initial_state.dim() != 4 or tuple(initial_state.shape[1:]) != expected_tail:
            raise ValueError("initial_state must be [N_or_pool, HO, V, K] (K-last)")
        if initial_state.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("initial_state must be fp32 or bf16")
        if indices is None and initial_state.shape[0] != n:
            raise ValueError("initial_state must carry one state per sequence")
    if indices is not None:
        if initial_state is None:
            raise ValueError("state_indices selects rows of initial_state, which is required")
        if indices.dtype != torch.int32:
            raise TypeError("state_indices must be int32")
        if indices.shape != (n,):
            raise ValueError("state_indices must be [N]")
        if checkpoint:
            raise ValueError("state_indices cannot combine with checkpoint_every_n_tokens")
    if flags["overwrite_initial_state"] and initial_state is None:
        raise ValueError("overwrite_initial_state requires initial_state")
    for name, t in (
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("cu_seqlens", cu),
        ("initial_state", initial_state),
        ("state_indices", indices),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
    ):
        if t is not None and t.device != q.device:
            raise ValueError(f"{name} must be on q's device")
    if cu.stride(0) != 1:
        raise NotImplementedError("cu_seqlens requires stride 1")


def _flags(
    scale,
    output_final_state,
    use_qk_l2norm_in_kernel,
    batch_invariant,
    use_beta_sigmoid_in_kernel,
    allow_neg_eigval,
    safe_gate,
    gate_domain,
    checkpoint_every_n_tokens,
    overwrite_initial_state=False,
):
    return dict(
        scale=scale,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        batch_invariant=batch_invariant,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_domain=gate_domain,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        overwrite_initial_state=overwrite_initial_state,
    )


def _run(q, k, v, g, beta, cu, initial_state, indices, a_log, dt_bias, flags, plan_name, final_out=None):
    _validate(q, k, v, g, beta, cu, initial_state, indices, a_log, dt_bias, flags)
    if q.device.type != "musa":
        raise mudnn.cudnnGraphNotSupportedError("GDN inference requires a MUSA device")
    tensors = (q, k, v, g, beta, cu, initial_state, indices, a_log, dt_bias)
    facts = InferenceFacts(
        get_arch(q.device),
        q.device,
        tuple(None if t is None else tuple(t.shape) for t in tensors),
        tuple(None if t is None else tuple(t.stride()) for t in tensors),
        tuple(None if t is None else t.dtype for t in tensors),
        tuple(sorted(flags.items())),
    )
    key = (facts, plan_name)
    with _plan_lock:
        plan = _plans.get(key)
        if plan is None:
            candidates = [e for e in engines_for(["GDN"]) if e.name in ("gdn_prefill_tilelang", "gdn_decode_tilelang")]
            if plan_name is not None:
                candidates = [e for e in candidates if e.name == plan_name]
            else:
                # Decode is explicitly pinned until automatic length-safe ranking is tuned.
                candidates = [e for e in candidates if e.name == "gdn_prefill_tilelang"]
            reasons = []
            for engine in candidates:
                try:
                    engine.check_support(facts)
                    plan = engine.build_plan(facts)
                except decline_types() as exc:
                    reasons.append(f"{engine.name}: {exc}")
                    continue
                _plans[key] = plan
                break
            if plan is None:
                raise mudnn.cudnnGraphNotSupportedError(f"no {plan_name!r} GDN inference plan: " + "; ".join(reasons))
    dtype = torch.float32 if initial_state is None else initial_state.dtype
    final = final_out
    if final is None and flags["output_final_state"]:
        final = torch.empty(cu.numel() - 1, max(q.shape[1], v.shape[1]), v.shape[-1], q.shape[-1], dtype=dtype, device=q.device)
    o, checkpoints = plan.execute(
        facts,
        None,
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu,
        initial_state=initial_state,
        state_indices=indices,
        a_log=a_log,
        dt_bias=dt_bias,
        final_state=final,
    )
    if final is None:
        final = torch.empty(0, dtype=dtype, device=q.device)
    if checkpoints is None:
        checkpoints = q.new_empty(0)
    return o, final, checkpoints


@torch.library.custom_op("mudnn::gdn_inference_fwd", mutates_args=())
def inference_fwd(
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
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flags = _flags(
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_domain,
        checkpoint_every_n_tokens,
    )
    return _run(q, k, v, g, beta, cu_seqlens, initial_state, None, a_log, dt_bias, flags, plan_name)


@inference_fwd.register_fake
def inference_fwd_fake(
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
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    checkpoint_every_n_tokens=0,
    plan_name=None,
):
    flags = _flags(
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_domain,
        checkpoint_every_n_tokens,
    )
    _validate(q, k, v, g, beta, cu_seqlens, initial_state, None, a_log, dt_bias, flags)
    total, _, dim = q.shape
    ho, n = max(q.shape[1], v.shape[1]), cu_seqlens.numel() - 1
    dtype = torch.float32 if initial_state is None else initial_state.dtype
    final = q.new_empty((n, ho, v.shape[-1], dim) if output_final_state else (0,), dtype=dtype)
    rows = max(total // checkpoint_every_n_tokens + n, 1) if checkpoint_every_n_tokens else 0
    checkpoints = q.new_empty((rows, ho, v.shape[-1], dim) if rows else (0,))
    return q.new_empty(total, ho, v.shape[-1]), final, checkpoints


@torch.library.custom_op("mudnn::gdn_inference_overwrite", mutates_args=("initial_state",))
def inference_overwrite(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool = True,
    use_qk_l2norm_in_kernel: bool = False,
    batch_invariant: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
    state_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flags = _flags(
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_domain,
        checkpoint_every_n_tokens,
        True,
    )
    o, _, checkpoints = _run(
        q, k, v, g, beta, cu_seqlens, initial_state, state_indices, a_log, dt_bias, flags, plan_name, initial_state if output_final_state else None
    )
    return o, checkpoints


@inference_overwrite.register_fake
def inference_overwrite_fake(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale,
    initial_state,
    output_final_state=True,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    checkpoint_every_n_tokens=0,
    plan_name=None,
    state_indices=None,
):
    flags = _flags(
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_domain,
        checkpoint_every_n_tokens,
        True,
    )
    _validate(q, k, v, g, beta, cu_seqlens, initial_state, state_indices, a_log, dt_bias, flags)
    rows = max(q.shape[0] // checkpoint_every_n_tokens + cu_seqlens.numel() - 1, 1) if checkpoint_every_n_tokens else 0
    checkpoints = q.new_empty((rows, max(q.shape[1], v.shape[1]), v.shape[-1], q.shape[-1]) if rows else (0,))
    return q.new_empty(q.shape[0], max(q.shape[1], v.shape[1]), v.shape[-1]), checkpoints


def gated_delta_net(
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
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    plan_name: Optional[str] = None,
    overwrite_initial_state: bool = False,
    state_indices: Optional[torch.Tensor] = None,
):
    """GDN over THD tensors; states are K-last ``[N_or_pool, HO, V, K]``.

    Returns ``(o, final_state)`` (empty final when not requested), plus an
    io-dtype pre-token checkpoint series when checkpoint_every_n_tokens > 0.
    Defaults to prefill inference; pin ``gdn_decode_tilelang`` for decode.
    Training prefers ``gdn_tilelang_v2`` on MP32 D128 and otherwise falls back
    to ``gdn_tilelang``. Unsupported engine capabilities raise.
    """
    if q.dim() != 3:
        raise ValueError("expected THD [total_tokens, heads, dim] tensors")
    if q.shape[-1] <= 0:
        raise ValueError("head dimensions must be positive")
    scale = 1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale)
    flags = _flags(
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_domain,
        checkpoint_every_n_tokens,
        overwrite_initial_state,
    )
    _validate(q, k, v, g, beta, cu_seqlens, initial_state, state_indices, a_log, dt_bias, flags)
    needs_backward = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (q, k, v, g, beta, initial_state, a_log, dt_bias))
    v2 = plan_name == "gdn_tilelang_v2"
    # Both training engines are training-only plans, so a pin to either one belongs on the
    # training branch even without gradients; there it reports the contract error instead of
    # falling through to the inference engines and failing with a plan-name miss.
    training_pin = plan_name in ("gdn_tilelang", "gdn_tilelang_v2")
    use_training_engine = needs_backward or training_pin or (plan_name is None and q.shape[-1] == v.shape[-1] == 64)
    if use_training_engine:
        from .gdn import _gated_delta_net_dispatch

        if plan_name not in (None, "gdn_tilelang", "gdn_tilelang_v2"):
            raise mudnn.cudnnGraphNotSupportedError(f"{plan_name}: inference only; no backward handoff")
        if v2 and not needs_backward:
            raise mudnn.cudnnGraphNotSupportedError(
                "gdn_tilelang_v2: this pairing is a training plan; "
                "call it from a differentiable path or pin an inference plan")
        if overwrite_initial_state or state_indices is not None:
            raise NotImplementedError("training engine does not support state mutation or pools")
        if batch_invariant:
            raise NotImplementedError("gated_delta_net: batch_invariant is not supported")
        if plan_name == "gdn_tilelang" and q.dtype != torch.bfloat16:
            raise TypeError("gdn_tilelang requires bfloat16 IO")
        if initial_state is not None and initial_state.dtype != torch.float32:
            raise TypeError("initial_state must be torch.float32 for training")
        if needs_backward and checkpoint_every_n_tokens not in (0, 64):
            raise NotImplementedError("GDN training supports checkpoint cadence 0 or 64 only")
        if g.dtype != torch.float32 or (beta.dtype != q.dtype and not use_beta_sigmoid_in_kernel):
            raise NotImplementedError(
                "training engine requires fp32 g and an activated beta in the io dtype")
        if gate_domain == "linear":
            g = g.clamp_min(1e-10).log()
        # Engines own their layout: the driver always hands the canonical cuDNN
        # V-major [N, HO, V, K] state and expects it back in that layout, so a
        # K-major engine converts at its own boundary.
        out = _gated_delta_net_dispatch(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            allow_neg_eigval=allow_neg_eigval,
            safe_gate=safe_gate,
            a_log=a_log,
            dt_bias=dt_bias,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            plan_name=plan_name,
        )
        if checkpoint_every_n_tokens > 0:
            return out[0], out[1], out[2]
        return out[0], out[1]
    kwargs = dict(
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        batch_invariant=batch_invariant,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        safe_gate=safe_gate,
        gate_domain=gate_domain,
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        plan_name=plan_name,
    )
    if overwrite_initial_state or state_indices is not None:
        kwargs["output_final_state"] = bool(output_final_state or overwrite_initial_state)
        o, checkpoints = inference_overwrite(q, k, v, g, beta, cu_seqlens, scale, initial_state, state_indices=state_indices, **kwargs)
        final = initial_state if kwargs["output_final_state"] else initial_state.new_empty(0)
    else:
        o, final, checkpoints = inference_fwd(q, k, v, g, beta, cu_seqlens, scale, initial_state, **kwargs)
    if checkpoint_every_n_tokens:
        return o, final, checkpoints
    return o, final
