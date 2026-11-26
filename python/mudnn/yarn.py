"""YARN RoPE frequency helpers for the cuDNN compatibility package."""

from __future__ import annotations

import math

import torch


def _resolve_torch_device(device: torch.device | str) -> torch.device | str:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return resolved
    try:
        torch.empty((), device=resolved)
    except Exception:
        return torch.device("cpu")
    return resolved


def _correction_dim(
    rotations: float, head_dim: int, base: float, max_position: int
) -> float:
    return (
        head_dim
        * math.log(max_position / (rotations * 2.0 * math.pi))
        / (2.0 * math.log(base))
    )


def _correction_range(
    beta_fast: float, beta_slow: float, head_dim: int, base: float, max_position: int
) -> tuple[int, int]:
    low = math.floor(_correction_dim(beta_fast, head_dim, base, max_position))
    high = math.ceil(_correction_dim(beta_slow, head_dim, base, max_position))
    return max(low, 0), min(high, head_dim - 1)


def _linear_ramp(low: int, high: int, count: int, *, device, dtype):
    if low == high:
        high = high + 0.001
    index = torch.arange(count, device=device, dtype=dtype)
    return torch.clamp((index - low) / (high - low), 0.0, 1.0)


def yarn_get_mscale(scaling_factor: float, mscale_factor: float = 1.0) -> float:
    if scaling_factor <= 1.0:
        return 1.0
    return 0.1 * float(mscale_factor) * math.log(float(scaling_factor)) + 1.0


def compute_yarn_inv_freq(
    head_dim: int,
    *,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_position: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if head_dim % 2 != 0:
        raise ValueError(f"YARN/RoPE require even head_dim, got {head_dim}")

    device = _resolve_torch_device(device)
    half = head_dim // 2
    inv_extra = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    if scaling_factor <= 1.0:
        return inv_extra

    inv_inter = inv_extra / float(scaling_factor)
    low, high = _correction_range(
        float(beta_fast),
        float(beta_slow),
        int(head_dim),
        float(base),
        int(original_max_position),
    )
    mask = 1.0 - _linear_ramp(low, high, half, device=device, dtype=dtype)
    return inv_inter * (1.0 - mask) + inv_extra * mask


def compute_yarn_freqs(
    max_seq_len: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_position: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    mscale_factor: float = 1.0,
    mscale_all_dim_factor: float = 0.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, float]:
    device = _resolve_torch_device(device)
    inv_freq = compute_yarn_inv_freq(
        head_dim,
        base=base,
        scaling_factor=scaling_factor,
        original_max_position=original_max_position,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        device=device,
        dtype=dtype,
    )
    half = int(head_dim) // 2
    positions = torch.arange(max_seq_len, device=device, dtype=dtype)
    angles = torch.outer(positions, inv_freq)
    freqs = torch.zeros(max_seq_len, 1, 1, head_dim, device=device, dtype=dtype)
    freqs[:, 0, 0, :half] = angles

    numerator = yarn_get_mscale(scaling_factor, mscale_factor)
    denominator = (
        yarn_get_mscale(scaling_factor, mscale_all_dim_factor)
        if mscale_all_dim_factor != 0.0
        else 1.0
    )
    return freqs, numerator / denominator


__all__ = ["compute_yarn_freqs", "compute_yarn_inv_freq", "yarn_get_mscale"]
