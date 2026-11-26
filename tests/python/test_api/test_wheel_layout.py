# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Wheel layout: what ``pip install`` puts at the top of the install prefix.

The wheel mirrors cuDNN Frontend's layout, where the Python packages and the C++
header tree both sit at the prefix root, so a consumer can point ``-I`` at
site-packages without a separate ``cmake --install``.  These tests assert the
declaration (pyproject.toml, MANIFEST.in) and the tree it refers to; building a
wheel is exercised by the packaging step in the build instructions instead.

Three properties are easy to lose silently and are checked explicitly:

* the header tree staying a shipped top-level entry,
* no shipped header being named ``cudnn*`` -- that namespace belongs to
  ``nvidia-cudnn-frontend``, and a same-named file makes one wheel silently
  overwrite the other when both are installed into one prefix, and
* ``looseversion`` staying out of the wheel, because vendoring it shadows a real
  installation of that package.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_FRONTEND_ROOT = Path(__file__).resolve().parents[3]

try:  # Python 3.11+
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 here
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ModuleNotFoundError:
        _toml = None


def _pyproject() -> dict:
    if _toml is None:
        try:
            from pip._vendor import tomli as vendored  # type: ignore[import-not-found]
        except ModuleNotFoundError:  # pragma: no cover - no TOML reader at all
            pytest.skip("no TOML reader available to inspect pyproject.toml")
        return vendored.loads((_FRONTEND_ROOT / "pyproject.toml").read_text())
    return _toml.loads((_FRONTEND_ROOT / "pyproject.toml").read_text())


def test_the_wheel_ships_the_cpp_header_tree():
    """``include/`` is a top-level entry, the way cuDNN Frontend ships it."""
    setuptools = _pyproject()["tool"]["setuptools"]

    assert setuptools["package-dir"]["include"] == "include"
    assert "." in setuptools["packages"]["find"]["where"]
    assert "include" in setuptools["packages"]["find"]["include"]
    assert setuptools["packages"]["find"]["namespaces"] is True
    assert "**/*" in setuptools["package-data"]["include"]


def test_the_header_tree_is_the_one_the_frontend_installs():
    """Package data is only useful if the files are where CMake installs them."""
    for relative in (
        "mudnn_frontend.h",
        "mudnn_frontend_version.h",
        "mudnn_frontend/graph.h",
        "mudnn_frontend/knobs.h",
        "mudnn_frontend/backend/backend_descriptor.h",
    ):
        assert (_FRONTEND_ROOT / "include" / relative).is_file(), f"missing include/{relative}"


def test_no_shipped_header_is_named_after_cudnn():
    """``cudnn*`` header names belong to nvidia-cudnn-frontend, not to this wheel.

    ``include/cudnn_backend_base.h`` used to be shipped as a shim forwarding to
    ``mudnn_frontend.h``.  The two packages install that path with different
    contents, so whichever installed last silently broke the other.  The C++
    API here is ``mudnn_frontend``-named only; keep it that way.
    """
    include_root = _FRONTEND_ROOT / "include"
    offenders = sorted(p.relative_to(include_root).as_posix() for p in include_root.rglob("*") if p.is_file() and p.name.startswith("cudnn"))

    assert not offenders, f"headers must not start with 'cudnn': {offenders}"
    removed = include_root / "cudnn_backend_base.h"
    assert not removed.exists(), "upstream's cudnn_backend_base.h holds cuDNN types that do not exist here"


def test_the_sdist_lists_the_header_tree():
    """A stripped sdist must not produce a header-less wheel."""
    manifest = (_FRONTEND_ROOT / "MANIFEST.in").read_text()

    assert re.search(r"^recursive-include\s+include\b", manifest, re.MULTILINE), manifest


def test_looseversion_is_not_shipped():
    """Vendoring it shadowed real installs; it is a dev dependency now."""
    assert not (_FRONTEND_ROOT / "python" / "looseversion").exists()
    dev = _pyproject().get("dependency-groups", {}).get("dev", [])
    assert any(entry.split(">")[0].split("=")[0].strip() == "looseversion" for entry in dev), dev


def test_the_version_has_one_source():
    """pyproject reads the version instead of restating it."""
    project = _pyproject()["project"]
    setuptools = _pyproject()["tool"]["setuptools"]

    assert "version" not in project, "declare the version once, in _version.py"
    assert project["dynamic"] == ["version"]
    assert setuptools["dynamic"]["version"] == {"attr": "mudnn_frontend._version.__version__"}

    version_py = _FRONTEND_ROOT / "python" / "mudnn_frontend" / "_version.py"
    match = re.search(r'^__version__ = "([^"]+)"$', version_py.read_text(), re.MULTILINE)
    assert match, version_py.read_text()

    import mudnn
    import mudnn_frontend

    assert mudnn_frontend.__version__ == match.group(1)
    assert mudnn.__version__ == match.group(1)


def test_the_serialized_json_version_comes_from_that_source():
    """The graph JSON used to hard-code a version that could drift."""
    graph = _FRONTEND_ROOT / "python" / "mudnn_frontend" / "graph.py"
    text = graph.read_text()

    assert '"cudnn_frontend_version": __version__,' in text
    assert not re.search(r'"cudnn_frontend_version":\s*"\d', text)
