# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Input contract shared by the native sparse attention forward and backward."""

import math

import torch

from .runtime import MP31_ARCH, MP32_ARCH, device_capability


def validate_inputs(q, kv, indices, sink, lengths, scale, *, backward=False, empty=False) -> float:
    """Validate Q/KV/indices/sink/lengths (tensors or descriptors) and return the scale."""
    if q.ndim != 3 or kv.ndim != 2 or indices.ndim != 2:
        raise ValueError("expected Q [S_q,H,D], KV [S_kv,D], topk_idxs [S_q,K]")
    sq, heads, dim = q.shape
    if dim not in (512, 576):
        raise ValueError(f"head_dim must be 512 or 576, got {dim}")
    if dim != 512:
        raise NotImplementedError("MUSA sparse attention supports QK=V=512 only")
    if kv.shape[1] != dim or indices.shape[0] != sq:
        raise ValueError("KV dimension and topk_idxs rows must match Q")
    if heads <= 0 or (heads > 64 and heads % 64):
        raise NotImplementedError("MUSA sparse attention requires H<=64 or H%64==0")
    if not empty and (sq <= 0 or kv.shape[0] <= 0 or indices.shape[1] <= 0):
        raise ValueError("sequence extents and physical Top-K must be positive")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Q must be FP16 or BF16, got {q.dtype}")
    if q.dtype == torch.float16:
        raise NotImplementedError("MUSA sparse attention supports BF16 only")
    named = [("KV", kv, torch.bfloat16), ("topk_idxs", indices, torch.int32)]
    for name, tensor, shape, dtype in (("attn_sink", sink, (heads,), torch.float32), ("topk_length", lengths, (sq,), torch.int32)):
        if tensor is not None:
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            named.append((name, tensor, dtype))
    for name, tensor, dtype in named:
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on Q's device")
    scale = dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if q.device.type != "musa":
        raise ValueError(f"Q must live on MUSA, got {q.device}")
    arch = device_capability(q.device)
    if arch not in (MP31_ARCH, MP32_ARCH):
        raise NotImplementedError(f"sparse attention requires MP31 or MP32, found MP{arch[0]}{arch[1]}")
    if backward and torch.are_deterministic_algorithms_enabled():
        raise NotImplementedError("sparse attention backward uses non-deterministic FP32 atomics")
    return scale


def pad_heads(tensor: torch.Tensor, heads: int, value: float = 0) -> torch.Tensor:
    """Pad the head axis (dim 1, or dim 0 for 1-D sinks) to 64."""
    if heads >= 64:
        return tensor
    shape = list(tensor.shape)
    axis = 0 if tensor.ndim == 1 else 1
    shape[axis] = 64 - heads
    return torch.cat((tensor, tensor.new_full(shape, value)), dim=axis)
