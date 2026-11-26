"""Static table of the python engines this build ships, and their id encoding.

Classification (this file) names the one family that serves a graph; that
family's engines then answer ``check_support()``. Stage one is a necessary
condition, never the verdict, so this file describes no engine's coverage and
imports no engine code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .engine_ids import FAMILY_BLOCK, GDN_ID_BASE, KDA_ID_BASE

_LOG = logging.getLogger("mudnn.engines.manifest")

#: Read live so a test can toggle it per process.
_ENABLE_ENV = "MUDNN_FRONTEND_ENABLE_OPTIN_ENGINES"


def opt_in_engines_enabled() -> bool:
    """Whether engines marked ``EngineSlot.opt_in`` are offered."""
    return os.environ.get(_ENABLE_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


# A shipped slot is fixed forever: append the next free one, never reorder or
# reuse. ``opt_in`` must be answerable without importing the engine.
@dataclass(frozen=True)
class EngineSlot:
    """One engine's place in its family's id block: slot index and opt-in gate."""

    slot: int
    opt_in: bool = False


@dataclass(frozen=True)
class EngineFamily:
    """One family of in-tree engines, described without importing them.

    ``engine_id`` is the first id of the block the family owns, ``module`` and
    ``factory`` name its collector, and ``slots`` is the single source of every
    python engine id.
    """

    engine_id: int
    name: str
    module: str
    factory: str
    slots: Mapping[str, EngineSlot] = field(default_factory=dict)

    @property
    def id_end(self) -> int:
        """One past the last id in this family's block."""
        return self.engine_id + FAMILY_BLOCK

    def owns(self, engine_id: int) -> bool:
        """Whether ``engine_id`` falls in this family's block."""
        return self.engine_id <= engine_id < self.id_end

    def offered_ids(self) -> Dict[str, int]:
        """``{engine name: engine id}`` for the engines on offer in this process.

        Maturity only. Whether an engine suits a device is its own
        ``check_support()``; a coarser copy here would be a second place to lie.
        """
        enabled = opt_in_engines_enabled()
        return {name: self.engine_id + s.slot for name, s in self.slots.items() if enabled or not s.opt_in}


# Node types that NAME a family -- not a list of what a family can serve. Names,
# not enum members, so this file imports no engine code.
_ANCHOR_NODE_TO_FAMILY = {
    "GDN": "gdn",
    "GDN_BWD": "gdn",
    "KDA": "kda",
    "KDA_BWD": "kda",
}

MANIFEST: Tuple[EngineFamily, ...] = (
    EngineFamily(
        GDN_ID_BASE,
        "gdn",
        "mudnn.linear_attention",
        "GdnEngines",
        slots={
            "gdn_tilelang": EngineSlot(0),
            "gdn_prefill_tilelang": EngineSlot(1),
            "gdn_decode_tilelang": EngineSlot(2),
            "gdn_tilelang_v2": EngineSlot(3),
        },
    ),
    EngineFamily(
        KDA_ID_BASE,
        "kda",
        "mudnn.linear_attention",
        "KdaEngines",
        slots={
            "kda_mutlass_mp31": EngineSlot(0),
            "kda_mutlass_mp32": EngineSlot(1),
        },
    ),
)

_INSTANCES: Dict[Any, Any] = {}


def family_for(node_types: Iterable[str]) -> Optional[EngineFamily]:
    """The one family these node-type names belong to, or None.

    A pure property of the names: no arch, no environment. Naming two families
    means neither. An unlisted name is ignored rather than refused, so coverage
    stays the engines' decision.
    """
    named = {_ANCHOR_NODE_TO_FAMILY[n] for n in node_types if n in _ANCHOR_NODE_TO_FAMILY}
    if len(named) != 1:
        return None
    name = named.pop()
    return next(f for f in MANIFEST if f.name == name)


def instantiate(family: EngineFamily, ids: Dict[str, int]):
    """Import ``family``'s module and build the engines named in ``ids``.

    The factory is handed the ids its engines are to use, so an engine cannot
    claim a number the manifest did not give it and a gated engine is absent
    rather than built and filtered. Returns the built engines, cached per
    (family, ids).
    """
    key = (family.engine_id, tuple(sorted(ids.items())))
    cached = _INSTANCES.get(key)
    if cached is not None:
        return cached
    import importlib

    try:
        made = getattr(importlib.import_module(family.module), family.factory)(dict(ids))
    except ImportError as exc:
        _LOG.info("family %s is unavailable in this environment: %s", family.name, exc)
        made = []
    except Exception:  # noqa: BLE001
        # An engine that vanishes from every plan list without a word is the
        # failure mode this table exists to remove.
        _LOG.warning("family %s failed to load and will not be offered", family.name, exc_info=True)
        made = []
    engines = list(made) if isinstance(made, (list, tuple)) else [made]
    for engine in engines:
        if engine.engine_id != ids.get(engine.name):
            raise ValueError(
                f"engine {engine.name!r} answers for id {engine.engine_id}, "
                f"not the {ids.get(engine.name)} the manifest assigned it")
    _INSTANCES[key] = engines
    return engines


def engines_for(node_types: Iterable[str]):
    """Every in-tree engine of these node types' family, in candidate order.

    Candidate order is not a ranking: ``check_support()`` still decides.
    """
    family = family_for(node_types)
    if family is None:
        return []
    ids = family.offered_ids()
    return list(instantiate(family, ids)) if ids else []


def engine_for_id(engine_id: int):
    """The engine that owns ``engine_id``, or None when no slot or gate matches.

    An id is fully decodable from this table, so a recorded plan can be replayed
    on a fresh call without anything having been registered first.
    """
    for family in MANIFEST:
        if not family.owns(engine_id):
            continue
        ids = family.offered_ids()
        if engine_id not in ids.values():
            return None
        return next((e for e in instantiate(family, ids) if e.engine_id == engine_id), None)
    return None


__all__ = [
    "MANIFEST",
    "EngineFamily",
    "EngineSlot",
    "engine_for_id",
    "engines_for",
    "family_for",
    "instantiate",
    "opt_in_engines_enabled",
]
