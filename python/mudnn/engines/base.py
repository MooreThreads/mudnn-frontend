"""Engine contract: ``check_support`` declines a call, ``build_plan`` resolves it
once, and ``CompiledPlan.execute`` runs it against a ``Facts`` record."""

from __future__ import annotations

from abc import ABC
from typing import Any, NamedTuple


def decline_types():
    """Exception types that mean an engine does not serve a call.

    Includes ``ImportError``: an engine whose optional dependency is absent
    cannot serve the call.
    """
    import mudnn

    return (NotImplementedError, mudnn.cudnnGraphNotSupportedError, ImportError)


class ExecutionContext(NamedTuple):
    """Runtime context handed to a compiled plan: target device and stream."""

    device: Any = None
    stream: Any = None


class CompiledPlan:
    """A compiled (configuration, plan) artifact. Subclass for a real engine."""

    plan_name: str = "base"

    def get_workspace_size(self) -> int:
        """Workspace bytes this plan needs at execute time (default 0)."""
        return 0

    def execute(self, facts, ctx: ExecutionContext, **buffers):
        raise NotImplementedError


class _EagerPlan(CompiledPlan):
    """Default plan for engines that only implement ``execute``."""

    def __init__(self, engine: "BaseEngine"):
        self.engine = engine
        self.plan_name = engine.name

    def execute(self, facts, ctx: ExecutionContext, **buffers):
        return self.engine.execute(facts, ctx, **buffers)


class BaseEngine(ABC):
    """Base class for python engines.

    Attributes:
        name: the plan name callers pin with ``plan_name=``.
        engine_id: stable id in the python region, assigned by the manifest.
    """

    name: str = "base"
    engine_id: Any = None

    def check_support(self, facts) -> None:
        """Raise to decline ``facts``. Default accepts; subclasses narrow it."""

    def build_plan(self, facts, ctx: ExecutionContext = None) -> CompiledPlan:
        """Resolve ``facts`` into a reusable plan. Default wraps ``execute``."""
        return _EagerPlan(self)

    def execute(self, facts, ctx: ExecutionContext, **buffers):
        raise NotImplementedError(f"engine {self.name!r} must implement execute() or build_plan()")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, engine_id={self.engine_id})"


__all__ = ["BaseEngine", "CompiledPlan", "ExecutionContext", "decline_types"]
