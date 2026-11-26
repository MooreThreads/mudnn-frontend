# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Drop-in acceleration for flash-linear-attention (FLA).

``accelerate_fla()`` monkeypatches the FLA entry points this build can serve and
calls the original implementation for everything else::

    import mudnn.fla

    mudnn.fla.accelerate_fla()
    mudnn.fla.restore_fla()
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .gated_delta_rule import last_path, make_chunk_gated_delta_rule

__all__ = ["accelerate_fla", "is_accelerated", "last_path", "restore_fla"]


@dataclass(frozen=True)
class _PatchSpec:
    """Where one target lives and how its replacement is built."""

    module_path: str
    attribute: str
    make_replacement: Callable
    owner_attribute: Optional[str] = None
    default: bool = True


@dataclass(frozen=True)
class _AppliedPatch:
    """One patch in force: its spec, owner, and the two callables involved."""

    spec: _PatchSpec
    owner: object
    original: object
    replacement: object


def _function_replacement(factory):
    def make_replacement(module, owner, original):
        del module, owner
        return factory(original)

    return make_replacement


_TARGETS = {
    "gated_delta_rule": _PatchSpec(
        "fla.ops.gated_delta_rule",
        "chunk_gated_delta_rule",
        _function_replacement(make_chunk_gated_delta_rule),
    ),
}
_ALIASES = {"gdn": "gated_delta_rule"}
_DEFAULT_TARGETS = tuple(name for name, spec in _TARGETS.items() if spec.default)
_ORIGINALS: dict = {}


def _canonical_target(target: str) -> str:
    """Resolve an alias and reject an unknown target name."""
    if not isinstance(target, str):
        raise TypeError(f"FLA acceleration target must be a string, got {type(target).__name__}")
    target = _ALIASES.get(target, target)
    if target not in _TARGETS:
        raise ValueError(f"unknown FLA acceleration target {target!r}; expected one of: {', '.join(_TARGETS)}")
    return target


def _normalize_targets(targets, *, default: Iterable[str]) -> tuple:
    """Canonical target names in registry order, so restoration is deterministic."""
    if targets is None:
        requested = tuple(default)
    elif isinstance(targets, str):
        requested = (targets,)
    else:
        requested = tuple(targets)
    if not requested:
        raise ValueError("at least one FLA acceleration target is required")
    canonical = {_canonical_target(target) for target in requested}
    return tuple(target for target in _TARGETS if target in canonical)


def is_accelerated(target: Optional[str] = None) -> bool:
    """Whether any target, or one named target, is currently patched."""
    if target is None:
        return any(getattr(a.owner, a.spec.attribute, None) is a.replacement for a in _ORIGINALS.values())
    applied = _ORIGINALS.get(_canonical_target(target))
    return applied is not None and getattr(applied.owner, applied.spec.attribute, None) is applied.replacement


def _drop_displaced_patch(target: str) -> None:
    """Forget a patch whose owner was replaced, without clobbering the new owner."""
    applied = _ORIGINALS.pop(target, None)
    if applied is not None and applied.spec.owner_attribute is None:
        _rebind_everywhere(applied.spec.attribute, applied.replacement, applied.original)


def _rebind_everywhere(fn_name: str, original, replacement) -> None:
    """Rebind imports that captured a patched module-level function by reference.

    FLA's own layers do ``from ... import chunk_gated_delta_rule`` at import
    time, so patching the defining module alone would not reach them.
    """
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            if getattr(module, fn_name, None) is original:
                setattr(module, fn_name, replacement)
        except Exception:
            continue


def accelerate_fla(verbose: bool = True, *, targets=None) -> None:
    """Patch the selected FLA targets, incrementally and idempotently.

    ``targets`` is a name or an iterable of names; ``None`` enables the
    defaults. Raises ImportError when an explicitly named target is
    unavailable.
    """
    requested = _normalize_targets(targets, default=_DEFAULT_TARGETS)
    available = {target for target in requested if is_accelerated(target)}
    resolved, missing, reasons = [], [], {}

    for target in requested:
        if is_accelerated(target):
            continue
        _drop_displaced_patch(target)
        spec = _TARGETS[target]
        try:
            module = importlib.import_module(spec.module_path)
        except ImportError as error:
            missing.append(target)
            reasons[target] = str(error) or f"cannot import {spec.module_path}"
            continue
        owner = module if spec.owner_attribute is None else getattr(module, spec.owner_attribute, None)
        if owner is None:
            missing.append(target)
            reasons[target] = f"{spec.module_path} has no {spec.owner_attribute} owner"
            continue
        original = getattr(owner, spec.attribute, None)
        if original is None:
            missing.append(target)
            reasons[target] = f"the target owner has no {spec.attribute} attribute"
            continue
        try:
            replacement = spec.make_replacement(module, owner, original)
        except ImportError as error:
            missing.append(target)
            reasons[target] = str(error) or "the installed target does not match the supported contract"
            continue
        resolved.append((target, spec, owner, original, replacement))

    # An explicit selection is a contract: never silently apply only a subset.
    if targets is not None and missing:
        details = ", ".join(f"{t} ({reasons[t]})" if t in reasons else t for t in missing)
        raise ImportError(f"accelerate_fla() could not enable FLA target(s): {details}")

    newly_patched = []
    for target, spec, owner, original, replacement in resolved:
        if spec.owner_attribute is None:
            _rebind_everywhere(spec.attribute, original, replacement)
        else:
            setattr(owner, spec.attribute, replacement)
        _ORIGINALS[target] = _AppliedPatch(spec, owner, original, replacement)
        newly_patched.append(target)
        available.add(target)

    if not available and not newly_patched:
        raise ImportError(f"accelerate_fla() could not find supported FLA target(s): {', '.join(requested)}")
    if verbose and newly_patched:
        print(f"[mudnn.fla] accelerated FLA {', '.join(newly_patched)} on MUSA; "
              "unsupported configs fall back to FLA.")


def restore_fla(*, targets=None) -> None:
    """Undo all active patches, or only the selected targets."""
    if targets is None:
        requested = tuple(target for target in _TARGETS if target in _ORIGINALS)
    else:
        requested = _normalize_targets(targets, default=())
    for target in requested:
        applied = _ORIGINALS.pop(target, None)
        if applied is None:
            continue
        if applied.spec.owner_attribute is None:
            _rebind_everywhere(applied.spec.attribute, applied.replacement, applied.original)
        if getattr(applied.owner, applied.spec.attribute, None) is applied.replacement:
            setattr(applied.owner, applied.spec.attribute, applied.original)
