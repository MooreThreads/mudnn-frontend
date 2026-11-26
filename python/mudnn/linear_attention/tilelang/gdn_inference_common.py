"""Metadata-only planning shared by the GDN inference engines."""

from dataclasses import dataclass

import torch

from .kernels.inference.helpers import Arch, MP31_ARCH, MP32_ARCH


@dataclass(frozen=True)
class InferenceFacts:
    arch: Arch
    device: torch.device
    shapes: tuple
    strides: tuple
    dtypes: tuple
    options: tuple

    @property
    def flags(self):
        return dict(self.options)


def check_common(name, facts):
    if not isinstance(facts, InferenceFacts):
        raise NotImplementedError(f"{name}: inference only; no backward handoff")
    if facts.arch not in (MP31_ARCH, MP32_ARCH):
        raise NotImplementedError(f"{name}: requires MP31 or MP32, got {facts.arch}")
    q, k, v = facts.shapes[:3]
    if q != k or v[1] % q[1]:
        raise NotImplementedError(f"{name}: requires Hq == Hk and Hv divisible by Hq")
    if q[-1] != 128 or v[-1] != 128:
        raise NotImplementedError(f"{name}: first migration supports K=V=128")
    if facts.dtypes[6] not in (None, torch.float32):
        raise NotImplementedError(f"{name}: initial_state must be float32")
    for shape, strides in zip(facts.shapes, facts.strides):
        if shape is not None and any(stride < 0 for stride in strides):
            raise NotImplementedError(f"{name}: negative strides are unsupported")


def prepare_gates(g, beta, flags, a_log=None, dt_bias=None):
    """Declared prefill prologue; the decode engine instead fuses these operations."""
    if flags["safe_gate"]:
        x = g.float()
        if dt_bias is not None:
            x = x + dt_bias.float()
        amplitude = 1.0 if a_log is None else a_log.float().exp()
        g = -amplitude * torch.nn.functional.softplus(x)
    elif flags["gate_domain"] == "linear":
        g = g.float().clamp_min(1e-10).log()
    if flags["use_beta_sigmoid_in_kernel"]:
        beta = beta.float().sigmoid()
        if flags["allow_neg_eigval"]:
            beta = beta * 2.0
    return g.contiguous(), beta.contiguous()
