# SPDX-License-Identifier: Apache-2.0
# Algorithm adapted from NVIDIA/cudnn-frontend 33c6fcb54d110f2c4fbeaf7c0330067492142189.
# Upstream: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared chunk-64 GDN backward implementation.

The public standalone API is re-exported by linear_attention.experimental.
"""

from dataclasses import dataclass
import math

import torch

from .kernels.training.gdn_bwd_kernels import factors, recompute
from .kernels.training.gdn_bwd_chunked import backward, fold_heads


def _grow(tensor, heads):
    return tensor.repeat_interleave(heads // tensor.shape[1], dim=1).contiguous() if tensor.shape[1] != heads else tensor.contiguous()


def _fold(tensor, heads):
    if tensor.shape[1] == heads:
        return tensor
    # Accumulate in fp32 straight from the low-precision view (no materialised fp32 copy).
    return tensor.view(tensor.shape[0], heads, tensor.shape[1] // heads, tensor.shape[2]).sum(2, dtype=torch.float32).to(tensor.dtype)


# ---------------------------------------------------------------------------
# Preprocessing VJPs. The production path runs l2norm / safe-gate / beta-sigmoid
# as torch ops and lets autograd own their gradients; this standalone port runs
# the same transforms outside autograd, so these helpers must stay the exact VJP
# of the gdn.py prologue (_l2_normalize / _apply_gate / _apply_beta).
# The transform-VJP checks in test_gdn_bwd.py pin that equivalence.
# ---------------------------------------------------------------------------


def _l2norm_vjp(source, normalized, grad, eps=1e-6):
    """VJP of row-wise L2 normalization ``x / sqrt(sum(x^2) + eps)``."""
    inv = torch.rsqrt(source.float().square().sum(-1, keepdim=True) + eps)
    grad = grad.float()
    return (inv * (grad - normalized * (grad * normalized).sum(-1, keepdim=True))).to(source.dtype)


def _safe_gate_dg(dg, amplitude, raw):
    """VJP of ``g`` through the safe gate ``-amplitude * softplus(raw)``."""
    return -dg * amplitude * torch.sigmoid(raw)


def _beta_sigmoid_db(db, bx, factor):
    """VJP of ``beta`` through ``factor * sigmoid(beta)``, given the activated ``bx``."""
    return db * bx * (1 - bx / factor)


@dataclass
class PreparedBackward:
    inputs: tuple
    transformed: tuple
    flags: dict
    factors: torch.Tensor | None
    prefixes: torch.Tensor | None
    checkpoints: torch.Tensor | None
    state_gradient: torch.Tensor | None
    kernel: object

    @torch.no_grad()
    def run(self):
        q, k, v, g, beta, cu, initial, dfs, a_log, dt_bias = self.inputs
        if self.kernel is None:
            empty = torch.empty(0, device=q.device)
            ds0 = empty if initial is None else (torch.zeros_like(initial) if dfs is None else dfs.clone())
            return tuple(torch.zeros_like(t) for t in (q, k, v, g, beta)) + (
                ds0,
                empty if a_log is None else torch.zeros_like(a_log),
                empty if dt_bias is None else torch.zeros_like(dt_bias),
            )
        qn, kn, qx, kx, vx, gx, bx, do = self.transformed
        h = qx.shape[1]
        out = [torch.empty_like(t) for t in (qx, kx, vx)]
        out += [torch.empty_like(gx), torch.empty_like(bx), torch.empty_like(self.state_gradient)]
        self.kernel(qx, kx, vx, bx, do, cu, self.factors, self.prefixes, self.checkpoints, self.state_gradient, *out)
        if q.shape[1] == k.shape[1] and q.shape[1] != h:
            # Grouped value heads: one fused fp32 fold for dQ and dK.
            dq, dk = (torch.empty(t.shape[0], q.shape[1], t.shape[2], device=t.device, dtype=t.dtype) for t in out[:2])
            fold_heads(q.shape[1], h // q.shape[1], qx.shape[2], qx.dtype)(out[0], out[1], dq, dk)
        else:
            dq, dk = (_fold(t, source.shape[1]) for t, source in zip(out[:2], (q, k)))
        dv = _fold(out[2], v.shape[1])
        dg, db, ds0 = out[3:]
        if self.flags["l2norm"]:
            dq, dk = _l2norm_vjp(q, qn, dq), _l2norm_vjp(k, kn, dk)
        da = torch.empty(0, device=q.device)
        ddt = torch.empty(0, device=q.device)
        if self.flags["safe_gate"]:
            raw = g.float() + (dt_bias.float() if dt_bias is not None else 0)
            amplitude = a_log.float().exp() if a_log is not None else 1
            if a_log is not None:
                da = (dg * gx).sum(0).to(a_log.dtype)
            dg = _safe_gate_dg(dg, amplitude, raw)
            if dt_bias is not None:
                ddt = dg.sum(0).to(dt_bias.dtype)
        elif self.flags["linear_gate"]:
            dg = torch.where(g.float() >= 1e-10, dg / g.float().clamp_min(1e-10), 0)
        if self.flags["beta_sigmoid"]:
            factor = 2 if self.flags["negative"] else 1
            db = _beta_sigmoid_db(db, bx, factor)
        if initial is None:
            ds0 = torch.empty(0, device=q.device)
        else:
            ds0 = ds0.to(initial.dtype)
        return dq, dk, dv, dg.to(g.dtype), db.to(beta.dtype), ds0, da, ddt


@torch.no_grad()
def prepare_gdn_bwd(
    dO,
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale=None,
    initial_state=None,
    d_final_state=None,
    state_checkpoints=None,
    checkpoint_every_n_tokens=0,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    plan_name=None,
):
    """Prepare independent chunk factors and recompute states; run() performs bwd.

    Supplied checkpoints are K-last io-dtype states at cadence 64, with the
    frontend's conservative ``total // 64 + num_sequences`` row capacity.
    """
    if plan_name not in (None, "gdn_bwd_kernels"):
        raise ValueError(f"experimental GDN port does not serve plan {plan_name!r}")
    if any(t.ndim != 3 for t in (q, k, v, dO)):
        raise ValueError("q/k/v/dO must be THD tensors")
    total, hq, dk = q.shape
    hv, dv, hk = v.shape[1], v.shape[2], k.shape[1]
    h = max(hq, hv)
    if min(hq, hk, hv) <= 0 or dk not in (64, 128) or dv not in (64, 128):
        raise NotImplementedError("experimental GDN port requires positive heads and K/V each in {64,128}")
    if hk not in (hq, hv) or h % hq or h % hv:
        raise ValueError("incompatible grouped q/k/v heads")
    if k.shape != (total, hk, dk) or v.shape[0] != total or dO.shape != (total, h, dv):
        raise ValueError("incompatible q/k/v/dO shapes")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != q.dtype for t in (k, v, dO)):
        raise TypeError("q/k/v/dO must share fp16 or bf16 dtype")
    if q.device.type != "musa":
        raise ValueError("experimental GDN port requires a MUSA device")
    for name, tensor in (("g", g), ("beta", beta)):
        if tensor.shape != (total, h):
            raise ValueError(f"{name} must be [total, HO]")
    if g.dtype not in (torch.float32, torch.float16, torch.bfloat16) or beta.dtype not in (torch.float32, q.dtype):
        raise TypeError("g must be fp32/fp16/bf16; beta must be fp32 or io dtype")
    cu = cu_seqlens
    if cu.ndim != 1 or cu.numel() < 2 or cu.dtype not in (torch.int32, torch.int64):
        raise ValueError("cu_seqlens must be int32/int64 [N+1]")
    if gate_domain not in ("log", "linear") or (safe_gate and gate_domain == "linear"):
        raise ValueError("gate_domain must be log/linear; linear cannot combine with safe_gate")
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("allow_neg_eigval requires beta sigmoid")
    if checkpoint_every_n_tokens not in (0, 64):
        raise NotImplementedError("first experimental port supports checkpoint cadence 64 only")
    if batch_invariant:
        raise NotImplementedError("batch-invariant bitwise behavior has not been validated for this experimental port")
    if state_checkpoints is not None and checkpoint_every_n_tokens != 64:
        raise ValueError("supplied state_checkpoints require checkpoint_every_n_tokens=64")
    n = cu.numel() - 1
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    for name, tensor in (("initial_state", initial_state), ("d_final_state", d_final_state)):
        if tensor is not None:
            if tensor.shape != (n, h, dv, dk):
                raise ValueError(f"{name} must be [N,HO,V,K]")
            if tensor.dtype not in (torch.float32, torch.bfloat16) or tensor.dtype != state_dtype:
                raise TypeError(f"{name} must match state dtype {state_dtype}")
    for name, tensor in (("a_log", a_log), ("dt_bias", dt_bias)):
        if tensor is not None:
            if not safe_gate or tensor.shape != (h,):
                raise ValueError(f"{name} requires safe_gate and shape [HO]")
            if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise TypeError(f"{name} must be fp32/fp16/bf16")
    for tensor in (k, v, g, beta, dO, cu, initial_state, d_final_state, state_checkpoints, a_log, dt_bias):
        if tensor is not None and tensor.device != q.device:
            raise ValueError("all tensors must be on q's device")
    scale = dk**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("scale must be finite")
    boundaries = cu.detach().cpu().tolist()
    if boundaries[0] != 0 or boundaries[-1] != total or any(a > b for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must be nondecreasing, start at zero, and end at total")
    capacity = max(total // 64 + n, 1)
    if state_checkpoints is not None:
        if state_checkpoints.shape != (capacity, h, dv, dk) or state_checkpoints.dtype != q.dtype:
            raise ValueError("checkpoint shape/dtype must match [capacity,HO,V,K] and io dtype")
    capability = torch.musa.get_device_capability(q.device)
    if capability != (3, 2):
        raise NotImplementedError(f"experimental GDN port currently targets MP32, got capability {capability}")
    if total == 0:
        return PreparedBackward((q, k, v, g, beta, cu, initial_state, d_final_state, a_log, dt_bias), (), {}, None, None, None, None, None)
    if use_qk_l2norm_in_kernel:
        qn, kn = q.float(), k.float()
        qn = qn * torch.rsqrt(qn.square().sum(-1, keepdim=True) + 1e-6)
        kn = kn * torch.rsqrt(kn.square().sum(-1, keepdim=True) + 1e-6)
        qx, kx = _grow(qn.to(q.dtype), h), _grow(kn.to(k.dtype), h)
    else:
        # No fp32 round trip: q.float().to(q.dtype) is the identity but costs four full-size cast kernels.
        qn = kn = None
        qx, kx = _grow(q, h), _grow(k, h)
    vx = _grow(v, h)
    gx = g.float()
    if safe_gate:
        gx = -(a_log.float().exp() if a_log is not None else 1) * torch.nn.functional.softplus(gx + (dt_bias.float() if dt_bias is not None else 0))
    elif gate_domain == "linear":
        gx = gx.clamp_min(1e-10).log()
    bx = beta.float()
    if use_beta_sigmoid_in_kernel:
        bx = bx.sigmoid() * (2 if allow_neg_eigval else 1)
    gx, bx, cu = gx.contiguous(), bx.contiguous(), cu.contiguous()
    w = torch.empty(capacity, h, 64, 64, dtype=q.dtype, device=q.device)
    prefix = torch.empty(capacity, h, 64, dtype=torch.float32, device=q.device)
    factors(h, dk, q.dtype, cu.dtype)(kx, gx, bx, cu, w, prefix)
    kernel = backward(
        h, dk, dv, q.dtype, cu.dtype, d_final_state is not None, scale, initial_state is not None, single_sequence=n == 1
    )
    if state_checkpoints is None:
        checkpoints = torch.empty(capacity, h, dv, dk, dtype=q.dtype, device=q.device)
        state = initial_state.float().contiguous() if initial_state is not None else torch.empty(n, h, dv, dk, device=q.device, dtype=torch.float32)
        if not kernel.recompute_forward(kx, vx, bx, cu, w, prefix, state, checkpoints):
            recompute(h, dk, dv, q.dtype, cu.dtype, initial_state is not None)(kx, vx, bx, cu, w, prefix, state, checkpoints)
    else:
        checkpoints = state_checkpoints.contiguous()
    state_gradient = d_final_state.float().contiguous() if d_final_state is not None else torch.empty(n, h, dv, dk, device=q.device, dtype=torch.float32)
    kernel.bind_forward(qx, kx, vx, bx, cu, w, prefix, checkpoints, boundaries, initial_state is not None)
    return PreparedBackward(
        (q, k, v, g, beta, cu, initial_state, d_final_state, a_log, dt_bias),
        (qn, kn, qx, kx, vx, gx, bx, dO.contiguous()),
        dict(
            l2norm=use_qk_l2norm_in_kernel,
            safe_gate=safe_gate,
            linear_gate=gate_domain == "linear",
            beta_sigmoid=use_beta_sigmoid_in_kernel,
            negative=allow_neg_eigval,
        ),
        w,
        prefix,
        checkpoints,
        state_gradient,
        kernel,
    )


@torch.no_grad()
def gdn_bwd(
    dO,
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    scale=None,
    initial_state=None,
    d_final_state=None,
    state_checkpoints=None,
    checkpoint_every_n_tokens=0,
    use_qk_l2norm_in_kernel=False,
    batch_invariant=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    gate_domain="log",
    a_log=None,
    dt_bias=None,
    plan_name=None,
):
    """Return (dQ,dK,dV,dG,dBeta,d_initial_state,d_a_log,d_dt_bias).

    This first-order-only standalone prototype is not an autograd registration,
    a graph node, or a replacement for the existing training engine.
    """
    prepared = prepare_gdn_bwd(
        dO,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        scale,
        initial_state,
        d_final_state,
        state_checkpoints,
        checkpoint_every_n_tokens,
        use_qk_l2norm_in_kernel,
        batch_invariant,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        safe_gate,
        gate_domain,
        a_log,
        dt_bias,
        plan_name,
    )
    return prepared.run()
