# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers shared by DSA Python wrappers."""

from contextvars import ContextVar
from dataclasses import replace
from functools import lru_cache
from typing import Optional

import torch

from mudnn._torch_stream import as_torch_stream, stream_context

MP31_ARCH = (3, 1)
MP32_ARCH = (3, 2)
_COMPILE_ONLY = ContextVar("dsa_compile_only", default=False)
_ARCH_VARIANTS: dict = {}


def jit_for_arch(jit_impl, arch: tuple[int, int]):
    """Return the TileLang JIT factory retargeted to ``arch``; factories are declared for mp_32."""
    if arch == MP32_ARCH:
        return jit_impl
    key = (id(jit_impl), arch)
    variant = _ARCH_VARIANTS.get(key)
    if variant is None:
        variant = replace(jit_impl, target={"kind": "musa", "arch": f"mp_{arch[0]}{arch[1]}"})
        _ARCH_VARIANTS[key] = variant
    return variant


def compile_only() -> bool:
    """Whether native kernels should be built without launching."""
    return _COMPILE_ONLY.get()


@lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple[int, int]:
    return tuple(torch.musa.get_device_capability(device_index))


def device_capability(device=None) -> tuple[int, int]:
    if device is None:
        index = torch.musa.current_device()
    elif isinstance(device, int):
        index = device
    else:
        index = torch.device(device).index
        if index is None:
            index = torch.musa.current_device()
    return _device_capability(index)


def get_musa_arch(device) -> tuple[int, int]:
    return device_capability(device)


def is_musa(tensor: Optional[torch.Tensor]) -> bool:
    return tensor is not None and tensor.device.type == "musa"


def resolve_num_mps(device, num_mps: Optional[int] = None) -> int:
    if num_mps is not None and num_mps > 0:
        return num_mps
    return torch.musa.get_device_properties(device).multi_processor_count


def cosize(shape, stride) -> int:
    return 1 + sum((size - 1) * step for size, step in zip(shape, stride))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def maybe_contiguous(x: Optional[torch.Tensor], stream=None) -> Optional[torch.Tensor]:
    if x is None or x.stride(-1) == 1:
        return x
    with torch_stream_context(stream, x.device):
        return x.contiguous()


def validate_q_causal_offsets(
    q_causal_offsets: Optional[torch.Tensor],
    batch: int,
    device: torch.device,
    stream=None,
) -> Optional[torch.Tensor]:
    if q_causal_offsets is None:
        return None
    if q_causal_offsets.dtype != torch.int32:
        raise ValueError("q_causal_offsets must be int32")
    if q_causal_offsets.ndim != 1 or q_causal_offsets.shape[0] != batch:
        raise ValueError(f"q_causal_offsets must have shape ({batch},), got {tuple(q_causal_offsets.shape)}")
    if q_causal_offsets.device != device:
        raise ValueError("q_causal_offsets must be on the same device as q")
    if q_causal_offsets.is_contiguous():
        return q_causal_offsets
    with torch_stream_context(stream, device):
        return q_causal_offsets.contiguous()


def resolve_stream(device, stream=None):
    """Return the ``torch.musa.Stream`` for a stream object, raw handle, or None."""
    if device.type != "musa":
        return None
    if stream is None:
        return torch.musa.current_stream(device)
    return as_torch_stream(stream, device)


def torch_stream_context(stream=None, device=None):
    """Run torch work on ``stream``; CPU tensors run eagerly."""
    if device is not None and torch.device(device).type != "musa":
        return stream_context(None)
    return stream_context(stream, device)


def record_stream(stream, *tensors) -> None:
    if stream is None:
        return
    for tensor in tensors:
        if tensor is not None and tensor.device.type == "musa":
            tensor.record_stream(stream)


def contiguous(tensor: Optional[torch.Tensor], alignment: int = 1) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    result = tensor.contiguous()
    if result.data_ptr() % alignment:
        result = result.clone(memory_format=torch.contiguous_format)
    return result


def output_buffer(tensor, shape, dtype, device, name, *, required=False) -> None:
    if tensor is None and not required:
        return
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be preallocated")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype or tensor.device != device:
        raise ValueError(f"{name} shape, dtype, or device mismatch")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def no_overlap(named) -> None:
    intervals = []
    for name, tensor in named:
        if tensor is not None and tensor.numel():
            begin = tensor.data_ptr()
            intervals.append((begin, begin + tensor.numel() * tensor.element_size(), name))
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise ValueError(f"{left[2]} and {right[2]} must not overlap")


def copy_result(dst: Optional[torch.Tensor], src: torch.Tensor) -> torch.Tensor:
    if dst is None:
        return src
    dst.copy_(src.to(dtype=dst.dtype))
    return dst
