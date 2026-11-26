"""Engine ids: one flat integer space, split into segments by provider.

    [0, 10_000)       muDNN backend engines, numbered by the backend as indices
                      into its heuristics result (0..N-1, N small today).
    [10_000, 20_000)  reserved for C++-side engines; not populated.
    [20_000, ...)     python engines, one FAMILY_BLOCK-wide block per family.
"""

from __future__ import annotations

BACKEND_ENGINE_ID_BASE = 0
CPP_ENGINE_ID_BASE = 10_000
PYTHON_ENGINE_ID_BASE = 20_000

# A family never opens a second block, so engine_id alone identifies the family
# and the two cannot disagree.
FAMILY_BLOCK = 100
GDN_ID_BASE = PYTHON_ENGINE_ID_BASE + 100  # 20_100..20_199
KDA_ID_BASE = PYTHON_ENGINE_ID_BASE + 200  # 20_200..20_299


def is_python_engine(engine_id: int) -> bool:
    """Whether ``engine_id`` names a python engine rather than a backend one."""
    return int(engine_id) >= PYTHON_ENGINE_ID_BASE


def is_backend_engine(engine_id: int) -> bool:
    """Whether ``engine_id`` names a muDNN backend engine."""
    return BACKEND_ENGINE_ID_BASE <= int(engine_id) < CPP_ENGINE_ID_BASE


__all__ = [
    "BACKEND_ENGINE_ID_BASE",
    "CPP_ENGINE_ID_BASE",
    "FAMILY_BLOCK",
    "GDN_ID_BASE",
    "KDA_ID_BASE",
    "PYTHON_ENGINE_ID_BASE",
    "is_backend_engine",
    "is_python_engine",
]
