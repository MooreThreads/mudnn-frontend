"""Helpers shared by the per-architecture dense indexer-backward modules.

Each architecture module used to import these from the ``mp_32`` module, which
made the ``mp_31`` path depend on a file named for another architecture. They
live here instead so neither module imports the other.
"""

import math

try:
    import torch
except ImportError:  # pragma: no cover - `import mudnn` must work without torch
    torch = None

try:
    import tilelang
    import tilelang.language as T
except ImportError:  # pragma: no cover - reported when a kernel is built
    tilelang = None
    T = None

__all__ = [
    "T",
    "_GEMM_PASS_CONFIGS",
    "_SCORE_GRAD_PASS_CONFIGS",
    "_SMEM_OVERHEAD",
    "_annotate_sqmma",
    "_dyn",
    "_require_tilelang",
    "_require_torch",
    "tilelang",
    "torch",
]

_SMEM_OVERHEAD = 8192

LOG2E = 1.4426950408889634


_GEMM_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
} if tilelang is not None else {}

_SCORE_GRAD_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
} if tilelang is not None else {}


def _require_tilelang():
    if tilelang is None or T is None:
        raise NotImplementedError(
            "dense_indexer_backward requires tilelang_musa: install the "
            "tilelang_musa wheel for this checkout before building the kernel."
        )
    return tilelang


def _dyn(*names):
    return tuple(T.dynamic(n) for n in names)


def _require_torch():
    if torch is None:
        raise NotImplementedError("dense_indexer_backward requires torch")
    return torch


def _annotate_sqmma(buffer, k_major, continuity=None, sg=None):
    swizzle_granularity = sg if sg is not None else 16
    if continuity is None:
        layout = tilelang.layout.make_sqmma_swizzled_layout(
            buffer, k_major=k_major, swizzle_granularity=swizzle_granularity
        )
    else:
        layout = tilelang.layout.make_sqmma_swizzled_layout(
            buffer, k_major=k_major, continuity=continuity, swizzle_granularity=swizzle_granularity
        )
    T.annotate_layout(
        {buffer: layout},
        allow_reannotation=True,
        allow_buffer_region=True,
    )
