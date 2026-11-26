"""Shared engine layer for the TileLang linear-attention families: the
check_support core and the execute-time buffer gate."""

from __future__ import annotations

import mudnn

#: The kernels are compiled for MUSA and load torch_musa.
DEVICE_TYPE = "musa"


def tilelang_la_gate(engine: str, facts, op: str) -> None:
    """Raise NotImplementedError unless the analyzer record passes the gates
    common to every TileLang LA family (op, device, dtypes, head counts).

    Args:
        engine: engine name used in the error messages.
        facts: the analyzer record, or None.
        op: the forward op name; ``{op}_BWD`` is also accepted.
    """
    if facts is None or facts.op not in (op, f"{op}_BWD"):
        raise NotImplementedError(f"{engine} supports exactly one {op}/{op}_BWD call")
    if facts.invalid:
        raise NotImplementedError(f"{engine}: {facts.invalid}")
    if facts.device_type != DEVICE_TYPE:
        raise NotImplementedError(f"{engine}: the TileLang kernels are {DEVICE_TYPE}-only, got {facts.device_type!r}")
    if not facts.uniform_io:
        raise NotImplementedError(f"{engine}: q/k/v dtypes must match")
    if facts.io_dtype not in (mudnn.data_type.BFLOAT16, None):
        raise NotImplementedError(f"{engine}: q/k/v must be bfloat16, got {facts.io_dtype}")
    if facts.h_q != facts.h_k or facts.h_q != facts.h_v:
        raise NotImplementedError(
            f"{engine}: q/k/v head counts must be equal after grouped-query expansion "
            f"(got {facts.h_q}/{facts.h_k}/{facts.h_v})")
    if facts.cu_dtype not in (mudnn.data_type.INT32, mudnn.data_type.INT64, None):
        raise NotImplementedError(f"{engine}: cu_seqlens must be int32 or int64, got {facts.cu_dtype}")


def check_contiguous(plan_name: str, **buffers) -> None:
    """Raise ValueError if any bound buffer is not contiguous.

    Args:
        plan_name: plan name used in the error message.
        buffers: name -> buffer mapping; None entries are skipped.
    """
    # The kernels' host stubs assert exact strides.
    for name, buffer in buffers.items():
        if buffer is None:
            continue
        if not buffer.is_contiguous():
            raise ValueError(
                f"{plan_name}: buffer for {name!r} must be contiguous; got shape "
                f"{tuple(buffer.shape)} strides {tuple(buffer.stride())}")


__all__ = ["DEVICE_TYPE", "check_contiguous", "tilelang_la_gate"]
