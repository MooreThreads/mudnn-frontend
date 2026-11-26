"""Packed-varlen index tables and the packed-row read helper.

Two axes are in play and keeping them straight IS the contract:

* the PACKED TOKEN axis holds the caller's q/k/v/g/beta/do/o/dq/dk -- sequence
  ``i`` occupies ``[cu_seqlens[i], cu_seqlens[i+1])``;
* the CHUNK-PADDED ROW axis holds every handoff we allocate -- global chunk
  ``g`` owns ``[g*C, (g+1)*C)`` unconditionally.

The backward indexes whole chunks, and on the packed axis a sequence's last
partial chunk would run into the next sequence's tokens.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch
import tilelang.language as T

from .gdn_shapes import _CHUNK


def _rdp(fn, row, shift, end):
    """Read a PACKED row for a chunk-padded ``row``; zero past THIS sequence's end.

    Takes ``shift``/``end`` as arguments rather than closing over them: the
    eager builder turns a read helper into a traced closure and cannot capture
    a factory-scope cell -- it emits a ``make_closure`` whose parameter list
    omits it, and every path breaks, dense included. Select between ``rd`` and
    this with a trace-time conditional at the call site.
    """
    prow = row + shift
    return T.if_then_else(prow < end, fn(T.max(T.min(prow, end - 1), 0)), 0)


class VarlenMaps(NamedTuple):
    """Everything the packed path derives from the sequence lengths."""

    cmap: torch.Tensor
    smap: torch.Tensor
    cu2: torch.Tensor
    rows: torch.Tensor
    NT: int
    num_tokens: int


@functools.lru_cache(maxsize=64)
def _varlen_maps_cached(lens: tuple[int, ...], device_str: str, C: int = _CHUNK):
    starts, cstarts = [0], [0]
    for length in lens:
        starts.append(starts[-1] + length)
        cstarts.append(cstarts[-1] + (length + C - 1) // C)

    seq, shift, send, flags = [], [], [], []
    for i, _ in enumerate(lens):
        n = cstarts[i + 1] - cstarts[i]
        untestable = 2 if n <= 2 else 0
        seq += [i] * n
        shift += [starts[i] - cstarts[i] * C] * n
        send += [starts[i + 1]] * n
        flags += [untestable] * (n - 1) + [1 + untestable]
    cmap = torch.tensor([seq, shift, send, flags], dtype=torch.int32).t().contiguous()
    smap = torch.tensor([[starts[i], cstarts[i]] for i in range(len(lens) + 1)],
                        dtype=torch.int32)
    cu2 = torch.tensor(starts + cstarts, dtype=torch.int32)
    rows = (torch.cat([torch.arange(length, dtype=torch.int64) + cstarts[i] * C
                       for i, length in enumerate(lens)])
            if lens else torch.zeros(0, dtype=torch.int64))
    dev = torch.device(device_str)
    return VarlenMaps(cmap.to(dev), smap.to(dev), cu2.to(dev), rows.to(dev),
                      cstarts[-1], starts[-1])


def varlen_lengths(cu_seqlens) -> tuple[int, ...]:
    """The per-sequence token counts behind a cumulative-offset vector."""
    cu = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
    if len(cu) < 2 or cu[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0 and hold NS+1 entries, got {cu}")
    lengths = tuple(b - a for a, b in zip(cu, cu[1:]))
    if any(length <= 0 for length in lengths):
        raise ValueError(f"cu_seqlens must be strictly increasing, got {cu}")
    return lengths


def varlen_maps(cu_seqlens, device, lengths=None) -> VarlenMaps:
    """Index tables for one length composition, cached on the lengths.

    Reading the lengths costs one BLOCKING device-to-host sync, which drains
    the queue and stalls the CPU on every kernel already enqueued -- the worst
    thing to do repeatedly on an operator this launch-bound. The maps are
    lru_cached, but the ``.tolist()`` that builds their key is not, so callers
    that already hold the lengths pass them in: ``gdn_chunk`` reads them once
    per call and threads them through the autograd node. The host needs NT to
    size every handoff anyway, so no version of this stays on device.
    """
    if lengths is None:
        lengths = varlen_lengths(cu_seqlens)
    return _varlen_maps_cached(tuple(lengths), str(device))


@functools.lru_cache(maxsize=16)
def dense_cu(batch: int, seq: int, device) -> torch.Tensor:
    """``cu_seqlens`` for a dense [B, S] batch: one equal-length sequence a row.

    Constant for a given launch shape, and the kernels only read it, so it is
    built once instead of once per forward.
    """
    return torch.arange(0, batch * seq + 1, seq, dtype=torch.int32, device=device)


@functools.lru_cache(maxsize=8)
def map_dummy(device, cols: int) -> torch.Tensor:
    """One-row stand-in for a CMap/SMap the flag-off build never reads.

    An unread tensor parameter is pruned from the device signature, but tvm_ffi
    still validates its declared shape on the host.
    """
    return torch.zeros(1, cols, dtype=torch.int32, device=device)
