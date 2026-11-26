"""PEP 517 build backend for mudnn-frontend.

The JIT compiles MUSA C++/mutlass kernel sources *at runtime*, so those sources
have to be present both in a checkout (where they are edited) and inside an
installed wheel (where the user runs from).  They cannot simply live under
``python/`` -- the kernel tree is third-party source, not Python -- so the
sources stay in their own top-level directories and this backend stages the
subset the JIT needs into ``mudnn/data/`` at build time.

    kernels/                     -> python/mudnn/data/kernels
    kernels/include/             -> python/mudnn/data/include
    kernels/csrc/                -> python/mudnn/data/csrc
    3rdparty/mutlass/include/    -> python/mudnn/data/mutlass/include

Those targets mirror exactly what ``mudnn/_jit/env.py`` looks for through
``_resolve_repo_or_packaged``: the checkout path while developing, the packaged
path once installed.

Editable installs get symlinks, so editing a kernel or a header is picked up
without reinstalling.  Wheels and sdists get real copies, because a symlink to
a developer's disk is meaningless once the archive has moved.

``python/mudnn/data`` is generated, and is listed in ``.gitignore``.

Two things this backend refuses to do quietly:

* build a wheel whose ``mudnn/data`` is incomplete.  ``3rdparty/mutlass`` is a
  git submodule, so a plain ``git clone`` leaves it empty; the JIT would then
  fail much later, inside ninja, with a missing-header error.  Instead the
  checkout's submodules are initialised first, and if that is not possible the
  build fails naming the path.
* read sources that are not there.  An sdist carries the staged ``mudnn/data``
  tree rather than the sources it came from, so a complete pre-staged tree is
  used as is.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import build_meta as _orig

_root = Path(__file__).parent.resolve()
_data_dir = _root / "python" / "mudnn" / "data"

#: ``(source relative to the repo root, destination under mudnn/data)``.
_STAGED_SOURCES: tuple[tuple[str, str], ...] = (
    ("kernels", "kernels"),
    ("kernels/include", "include"),
    ("kernels/csrc", "csrc"),
    ("3rdparty/mutlass/include", "mutlass/include"),
)


def _remove(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


#: Staged sources that come from a git submodule, and the submodule carrying
#: them.  A plain ``git clone`` leaves these empty.
_SUBMODULE_SOURCES: dict[str, str] = {"3rdparty/mutlass/include": "3rdparty/mutlass"}

_MISSING_SOURCES_HINT = """\
cannot stage the JIT kernel sources: they are missing from the build tree

{missing}
"""

_SUBMODULE_HINT = """\
Those paths are populated by a git submodule that a plain `git clone` leaves
empty:

{paths}

From this checkout, run

    git submodule update --init --recursive

(or re-clone with `--recursive`) and install again.  A wheel or an sdist needs
none of this: both already carry the staged mudnn/data tree."""


def _missing_sources() -> list[str]:
    return [rel for rel, _ in _STAGED_SOURCES if not (_root / rel).exists()]


def _pre_staged() -> bool:
    """True when ``mudnn/data`` already holds every staged tree."""
    return all((_data_dir / dest).exists() for _, dest in _STAGED_SOURCES)


def _missing_sources_error(missing: list[str]) -> RuntimeError:
    message = _MISSING_SOURCES_HINT.format(missing="\n".join(f"    {rel}" for rel in missing))
    submodules = sorted({_SUBMODULE_SOURCES[rel] for rel in missing if rel in _SUBMODULE_SOURCES})
    if submodules:
        message += _SUBMODULE_HINT.format(paths="\n".join(f"    {path}" for path in submodules))
    return RuntimeError(message)


def _init_submodule(submodule: str) -> bool:
    """Populate one empty submodule so a plain ``git clone`` can still build.

    Only attempted for a checkout: an unpacked sdist has no ``.git``.  The
    submodule may need the network, and its URL may be an SSH one, so the
    prompts are disabled and a failure is reported, never fatal by itself.
    """
    if not (_root / ".git").exists() or shutil.which("git") is None:
        return False
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    command = f"git submodule update --init --recursive -- {submodule}"
    print(f"[mudnn-frontend] {submodule} is empty; running `{command}`", flush=True)
    try:
        completed = subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive", "--", submodule],
            cwd=_root,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env
        print(f"[mudnn-frontend] {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    if completed.returncode:
        print(completed.stderr.strip(), file=sys.stderr)
    return completed.returncode == 0


def _stage(use_symlinks: bool) -> None:
    missing = _missing_sources()

    if missing:
        if _pre_staged():
            # An sdist ships the staged tree (python/mudnn/data) and not the
            # sources it was staged from, so there is nothing left to do.
            return
        for rel in missing:
            submodule = _SUBMODULE_SOURCES.get(rel)
            if submodule is not None:
                _init_submodule(submodule)
        missing = _missing_sources()

    if missing:
        raise _missing_sources_error(missing)

    _remove(_data_dir)
    _data_dir.mkdir(parents=True, exist_ok=True)

    # ``mudnn.data`` is a package so setuptools' package-data can ship the tree;
    # the marker is generated here because the directory itself is generated.
    (_data_dir / "__init__.py").write_text(
        '"""Packaged JIT kernel sources and third-party headers (generated).\n\n'
        'Staged by build_backend.py; see mudnn/_jit/env.py for how it is found.\n"""\n'
    )

    for source_rel, dest_rel in _STAGED_SOURCES:
        source = _root / source_rel
        destination = _data_dir / dest_rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if use_symlinks:
            destination.symlink_to(source, target_is_directory=True)
        else:
            shutil.copytree(source, destination, symlinks=False, dirs_exist_ok=True)


# The staging has to happen before setuptools scans for packages and data, and
# pip may call any of the metadata hooks first, so every entry point stages.
def get_requires_for_build_wheel(config_settings=None):
    _stage(use_symlinks=False)
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    _stage(use_symlinks=False)
    return _orig.get_requires_for_build_sdist(config_settings)


def get_requires_for_build_editable(config_settings=None):
    _stage(use_symlinks=True)
    return _orig.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _stage(use_symlinks=False)
    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _stage(use_symlinks=True)
    return _orig.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _stage(use_symlinks=False)
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _stage(use_symlinks=True)
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    _stage(use_symlinks=False)
    return _orig.build_sdist(sdist_directory, config_settings)
