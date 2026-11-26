"""Staging rules of ``build_backend.py``, the packaging step the JIT depends on.

The backend copies the kernel sources into ``python/mudnn/data`` at build time.
Two failure modes are worth locking down, because both used to install a broken
package silently rather than fail the build:

* a checkout whose ``3rdparty/mutlass`` submodule was never initialised, which
  staged no mutlass headers at all;
* an sdist, which carries the staged tree but not the sources it came from.

These tests do not build a wheel: they run ``_stage`` against simulated trees,
which is exactly the decision the real build makes.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

_FRONTEND_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_SOURCE = _FRONTEND_ROOT / "build_backend.py"

_STAGED = ("kernels", "include", "csrc", "mutlass/include")


def _load_backend(root: Path):
    pytest.importorskip("setuptools", reason="the build backend imports setuptools")
    shutil.copy(_BACKEND_SOURCE, root / "build_backend.py")
    spec = importlib.util.spec_from_file_location(f"build_backend_{root.name}", root / "build_backend.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_checkout(root: Path, *, mutlass: bool) -> None:
    (root / "kernels" / "include" / "mate").mkdir(parents=True)
    (root / "kernels" / "include" / "mate" / "kda.hpp").write_text("// kernel\n")
    (root / "kernels" / "csrc").mkdir(parents=True)
    (root / "kernels" / "csrc" / "kda.cpp").write_text("// csrc\n")
    if mutlass:
        (root / "3rdparty" / "mutlass" / "include" / "mutlass").mkdir(parents=True)
        (root / "3rdparty" / "mutlass" / "include" / "mutlass" / "gemm.hpp").write_text("// mutlass\n")


def _pre_stage(root: Path) -> None:
    data = root / "python" / "mudnn" / "data"
    for destination in _STAGED:
        (data / destination).mkdir(parents=True)
        (data / destination / "staged.marker").write_text("staged\n")
    (data / "__init__.py").write_text("")


def test_stage_refuses_a_checkout_without_the_mutlass_submodule(tmp_path):
    """A plain ``git clone`` leaves the submodule empty; that must fail the build.

    The JIT cannot compile without the mutlass headers, so shipping a wheel
    without them would only move the error to the first KDA call.
    """
    _make_checkout(tmp_path, mutlass=False)

    with pytest.raises(RuntimeError) as excinfo:
        _load_backend(tmp_path)._stage(use_symlinks=False)

    message = str(excinfo.value)
    assert "3rdparty/mutlass/include" in message, "the error must name the missing source"
    assert "git submodule update --init --recursive" in message, "the error must say what to run"
    assert not (tmp_path / "python" / "mudnn" / "data").exists(), "nothing may be staged"


def test_stage_initialises_the_submodule_when_it_can(tmp_path, monkeypatch):
    """In a git checkout the backend tries the submodule itself first."""
    _make_checkout(tmp_path, mutlass=False)
    (tmp_path / ".git").mkdir()
    calls: list[list[str]] = []

    class _Completed:
        returncode = 0
        stderr = ""

    def _fake_run(command, **kwargs):
        calls.append(list(command))
        return _Completed()

    backend = _load_backend(tmp_path)
    monkeypatch.setattr(backend.subprocess, "run", _fake_run)
    with pytest.raises(RuntimeError):
        # the fake ``git`` reports success but stages nothing, so the missing
        # source is still missing
        backend._stage(use_symlinks=False)

    assert calls == [["git", "submodule", "update", "--init", "--recursive", "--", "3rdparty/mutlass"]]


def test_stage_reuses_a_complete_pre_staged_tree(tmp_path):
    """An sdist ships ``mudnn/data`` and not the sources it was staged from."""
    _pre_stage(tmp_path)
    marker = tmp_path / "python" / "mudnn" / "data" / "kernels" / "staged.marker"

    _load_backend(tmp_path)._stage(use_symlinks=False)

    assert marker.is_file(), "the pre-staged tree must be left alone"


def test_stage_mirrors_every_source_of_a_complete_checkout(tmp_path):
    _make_checkout(tmp_path, mutlass=True)

    _load_backend(tmp_path)._stage(use_symlinks=False)

    data = tmp_path / "python" / "mudnn" / "data"
    for relative in (
        "kernels/include/mate/kda.hpp",
        "include/mate/kda.hpp",
        "csrc/kda.cpp",
        "mutlass/include/mutlass/gemm.hpp",
        "__init__.py",
    ):
        assert (data / relative).is_file(), f"missing staged {relative}"


def test_the_sdist_carries_the_build_backend():
    """``backend-path = ["."]`` means the sdist has to contain the backend."""
    manifest = (_FRONTEND_ROOT / "MANIFEST.in").read_text()

    assert "build_backend.py" in manifest
