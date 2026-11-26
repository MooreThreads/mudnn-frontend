# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Adapters between MUSA stream handles and ``torch.musa`` stream objects.

The CUDA frontend keeps this conversion in one helper because passing a raw
driver handle through ``torch.cuda.ExternalStream`` can break default-stream
ordering.  TileLang MUSA launches on the current MUSA stream, so this module
provides the corresponding handle conversion and context in one place.  NVTX
is intentionally not wrapped here: the MUSA runtime used by this repository
does not expose a stable ``torch.musa.nvtx.range`` equivalent.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional


# MUSA's null stream handle is the default stream.  Do not pass it to
# torch.musa.ExternalStream(0), which creates a separate external stream.
DEFAULT_STREAM_HANDLES = frozenset({0})


def _device_index(torch, device) -> int:
    if device is None:
        return int(torch.musa.current_device())
    if isinstance(device, int):
        return int(device)
    return int(device.index if device.index is not None else torch.musa.current_device())


def _stream_handle(torch, stream) -> int:
    if isinstance(stream, torch.musa.Stream):
        return int(stream.musa_stream)
    return int(stream)


def _raw_current_stream(torch, device) -> Optional[int]:
    """Return the current raw MUSA stream when torch_musa exposes the getter."""

    try:
        import torch_musa
    except ImportError:  # pragma: no cover - torch_musa is required on MUSA
        torch_musa = None

    getter = getattr(getattr(torch_musa, "_MUSAC", None), "_musa_getCurrentRawStream", None)
    if getter is not None:
        return int(getter(_device_index(torch, device)))
    return int(torch.musa.current_stream(device).musa_stream)


def as_torch_stream(stream, device=None):
    """Convert a MUSA stream object or raw handle to ``torch.musa.Stream``.

    The null/default stream is mapped to ``torch.musa.default_stream`` instead
    of ``torch.musa.ExternalStream(0)``.  A genuine side-stream handle is
    wrapped with ``ExternalStream``; an existing torch stream is returned after
    validating its device.
    """

    import torch

    if isinstance(stream, torch.musa.Stream):
        if device is not None:
            expected = torch.device("musa", _device_index(torch, device))
            if stream.device != expected:
                raise ValueError(f"stream must be on {expected}, got {stream.device}")
        return stream

    handle = int(stream)
    default = torch.musa.default_stream(device)
    if handle in DEFAULT_STREAM_HANDLES or handle == int(default.musa_stream):
        return default

    current = torch.musa.current_stream(device)
    if handle == int(current.musa_stream):
        return current
    return torch.musa.ExternalStream(handle, device=device)


@contextmanager
def stream_context(stream, device=None, *, verify_current: bool = False) -> Iterator[None]:
    """Run torch and TileLang work on ``stream`` when one is supplied."""

    if stream is None:
        yield
        return

    import torch

    handle = _stream_handle(torch, stream)
    if not verify_current and handle not in DEFAULT_STREAM_HANDLES:
        current = _raw_current_stream(torch, device)
        if current is not None and handle == current:
            yield
            return

    with torch.musa.stream(as_torch_stream(stream, device)):
        yield


__all__ = [
    "DEFAULT_STREAM_HANDLES",
    "as_torch_stream",
    "stream_context",
]
