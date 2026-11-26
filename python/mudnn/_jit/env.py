"""JIT environment: version, MUSA arch resolution, cache/work directories,
and the repo-or-packaged path resolution the JIT sources are found through.

Ported from the upstream JIT environment module.  The layout is the only thing that
changed: mudnn-frontend keeps its Python packages under ``python/``, so the
repository root is discovered by marker rather than by ``package_root.parent``,
and the kernel sources live under ``kernels/`` because ``include/`` is reserved
for the cuDNN-compatible header-only frontend.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

_package_root: pathlib.Path = pathlib.Path(__file__).resolve().parents[1]


def _looks_like_repo_root(candidate: pathlib.Path) -> bool:
    """Whether ``candidate`` is *this* project's checkout root.

    The check is deliberately project-specific.  A bare ``.git`` or
    ``CMakeLists.txt`` is not enough: an installed package can sit under some
    unrelated directory that happens to carry one (a virtualenv created inside
    another repository, a shared ``/tmp``), and treating that as the checkout
    would send every source lookup to the wrong tree.  Requiring the package
    layout ``python/mudnn`` pins the match to this project.
    """
    if not (candidate / "python" / "mudnn").is_dir():
        return False
    return (candidate / ".git").exists() or (candidate / "CMakeLists.txt").exists()


def _find_repo_root() -> pathlib.Path:
    """The checkout root, found by marker instead of by a fixed depth.

    A mudnn-frontend checkout has ``CMakeLists.txt`` at its root; a git
    checkout also has ``.git`` there.  Neither exists in an installed wheel,
    which is exactly when the packaged ``data/`` tree is used instead.
    """
    for parent in _package_root.parents:
        if _looks_like_repo_root(parent):
            return parent
    return _package_root.parent


_repo_root: pathlib.Path = _find_repo_root()


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
def _load_build_meta_version() -> str | None:
    build_meta_path = _package_root / "_build_meta.py"
    if not build_meta_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "mudnn._build_meta_runtime", build_meta_path
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) and version else None


def _load_package_version() -> str | None:
    """``mudnn.__version__``, the version the frontend package declares."""
    try:
        import mudnn

        version = getattr(mudnn, "__version__", None)
        return version if isinstance(version, str) and version else None
    except Exception:
        return None


def _load_installed_package_version() -> str | None:
    try:
        from importlib.metadata import version as package_version

        resolved = package_version("mudnn-frontend")
        return resolved if resolved else None
    except Exception:
        return None


def _load_repo_version() -> str | None:
    version_file = _repo_root / "version.txt"
    if not version_file.exists():
        return None
    version = version_file.read_text().strip()
    return version if version else None


def _prefer_repo_tree() -> bool:
    return (_repo_root / ".git").exists()


def resolve_runtime_version_string() -> str:
    candidates = (
        (
            _load_repo_version(),
            _load_build_meta_version(),
            _load_package_version(),
            _load_installed_package_version(),
        )
        if _prefer_repo_tree()
        else (
            _load_build_meta_version(),
            _load_package_version(),
            _load_installed_package_version(),
            _load_repo_version(),
        )
    )

    for candidate in candidates:
        if not candidate:
            continue
        try:
            Version(candidate)
        except InvalidVersion:
            continue
        return candidate
    return "0.0.0"


def _resolve_base_version(version_string: str | None = None) -> str:
    return Version(version_string or resolve_runtime_version_string()).base_version


# ---------------------------------------------------------------------------
# MUSA architectures
# ---------------------------------------------------------------------------
def _parse_arch_token(arch: str) -> tuple[int, int]:
    parts = arch.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid MUSA architecture: {arch}")
    major, minor = (int(part) for part in parts)
    return major, minor


@dataclass(frozen=True, order=True)
class MusaArch:
    major: int
    minor: int

    @classmethod
    def parse(cls, value: str) -> "MusaArch":
        major, minor = _parse_arch_token(value.strip())
        return cls(major=major, minor=minor)

    @property
    def cc(self) -> int:
        return self.major * 10 + self.minor

    @property
    def name(self) -> str:
        return f"mp{self.major}{self.minor}"

    @property
    def target_flag(self) -> str:
        return f"--offload-arch=mp_{self.major}{self.minor}"


def _normalize_arch_pairs(arch_pairs: list[tuple[int, int]]) -> tuple[str, ...]:
    unique_arch_pairs = sorted(set(arch_pairs))
    if not unique_arch_pairs:
        raise ValueError("No MUSA architectures resolved for JIT cache namespace")
    return tuple(f"mp{major}{minor}" for major, minor in unique_arch_pairs)


def _normalize_musa_arch_list(arch_list: str) -> tuple[str, ...]:
    arch_pairs = [_parse_arch_token(arch) for arch in arch_list.split()]
    return _normalize_arch_pairs(arch_pairs)


def _visible_musa_modules():
    """The torch module that owns MUSA, whichever one this environment has.

    The frontend's own kernels use ``torch.musa``; a standalone ``torch_musa``
    install also exists. Both expose the same ``is_available`` /
    ``device_count`` / ``get_device_properties`` trio.
    """
    try:
        import torch

        musa = getattr(torch, "musa", None)
        if musa is not None and musa.is_available():
            return musa
    except Exception:
        pass
    try:
        import torch_musa

        if torch_musa.is_available():
            return torch_musa
    except Exception:
        pass
    return None


def _detect_visible_musa_arch_list() -> tuple[str, ...]:
    musa = _visible_musa_modules()
    if musa is None:
        raise RuntimeError(
            "Unable to auto-detect visible MUSA architectures because no MUSA "
            "device is available. Set MUDNN_JIT_MUSA_ARCH_LIST explicitly, for "
            "example 'MUDNN_JIT_MUSA_ARCH_LIST=3.1'."
        )

    arch_pairs: list[tuple[int, int]] = []
    for device_idx in range(musa.device_count()):
        props = musa.get_device_properties(device_idx)
        arch_pairs.append((int(props.major), int(props.minor)))
    return _normalize_arch_pairs(arch_pairs)


def resolve_musa_arch_list() -> tuple[str, ...]:
    """The one place MUSA architectures are decided for JIT.

    An explicit ``MUDNN_JIT_MUSA_ARCH_LIST`` wins because it is the only way to
    build for a target other than the locally visible device: cross building,
    or validating an MP32 build from an MP31 host. It is also the same variable
    that drives the ``--offload-arch`` flags, so honouring it keeps the two in
    step by construction.
    """
    arch_list = os.environ.get("MUDNN_JIT_MUSA_ARCH_LIST")
    if arch_list is not None:
        stripped = arch_list.strip()
        if not stripped:
            raise ValueError("MUDNN_JIT_MUSA_ARCH_LIST is set but empty.")
        return _normalize_musa_arch_list(stripped)
    return _detect_visible_musa_arch_list()


def resolve_musa_target_flags(arch: MusaArch | None = None) -> tuple[str, ...]:
    if arch is not None:
        return (arch.target_flag,)
    return tuple(f"--offload-arch=mp_{item[2:]}" for item in resolve_musa_arch_list())


def resolve_musa_arch_key() -> str:
    return "_".join(resolve_musa_arch_list())


MUDNN_JIT_VERSION = Version(resolve_runtime_version_string())
MUDNN_JIT_BASE_VERSION = MUDNN_JIT_VERSION.base_version

MUDNN_JIT_BASE_DIR: pathlib.Path = pathlib.Path(
    os.getenv("MUDNN_JIT_WORKSPACE_BASE", pathlib.Path.home().as_posix())
)

MUDNN_JIT_CACHE_DIR: pathlib.Path = MUDNN_JIT_BASE_DIR / ".cache" / "mudnn" / "jit"


# ---------------------------------------------------------------------------
# Source locations
# ---------------------------------------------------------------------------
def _resolve_repo_or_packaged(
    repo_relative: str, packaged_relative: str
) -> pathlib.Path:
    """A source directory, from the checkout when developing and from the
    installed ``mudnn/data`` tree otherwise.

    The JIT compiles C++/mutlass sources at runtime, so the headers and
    templates have to ship inside the wheel as well as exist in the checkout.
    The build backend stages them into ``mudnn/data``; this picks whichever is
    present, preferring the checkout so edits are picked up immediately.
    """
    repo_path = _repo_root / repo_relative
    packaged_path = _package_root / "data" / packaged_relative
    if _prefer_repo_tree() and repo_path.exists():
        return repo_path
    if packaged_path.exists():
        return packaged_path
    return repo_path


def resolve_csrc_dir() -> pathlib.Path:
    return _resolve_repo_or_packaged("kernels/csrc", "csrc")


def resolve_kernels_dir() -> pathlib.Path:
    """Root of the JIT-compiled kernel sources (``kernels/`` in a checkout).

    Resolved repo-or-packaged like the include and csrc roots, so an installed
    wheel finds the sources the build backend staged under ``mudnn/data``.
    """
    return _resolve_repo_or_packaged("kernels", "kernels")


def resolve_include_dir() -> pathlib.Path:
    return _resolve_repo_or_packaged("kernels/include", "include")


def resolve_mutlass_include_dir() -> pathlib.Path:
    return _resolve_repo_or_packaged("3rdparty/mutlass/include", "mutlass/include")


MUDNN_JIT_DATA: pathlib.Path = _package_root / "data"
MUDNN_JIT_CSRC_DIR: pathlib.Path = resolve_csrc_dir()
MUDNN_JIT_TEMPLATE_DIR: pathlib.Path = MUDNN_JIT_CSRC_DIR / "templates"
MUDNN_JIT_INCLUDE_DIR: pathlib.Path = resolve_include_dir()
MUTLASS_INCLUDE_DIR: pathlib.Path = resolve_mutlass_include_dir()


def resolve_musa_workspace_dir(arch: MusaArch) -> pathlib.Path:
    return MUDNN_JIT_CACHE_DIR / MUDNN_JIT_BASE_VERSION / arch.name


def resolve_musa_generated_dir(arch: MusaArch) -> pathlib.Path:
    return resolve_musa_workspace_dir(arch) / "generated"


def resolve_musa_jit_dir(arch: MusaArch) -> pathlib.Path:
    return resolve_musa_workspace_dir(arch) / "cached_ops"


# ---------------------------------------------------------------------------
# Arch-dependent names, resolved on first use
# ---------------------------------------------------------------------------
# Resolving the architecture needs either a visible MUSA device or an explicit
# ``MUDNN_JIT_MUSA_ARCH_LIST``.  Doing that at import time would make
# ``import mudnn._jit`` fail on a machine that only ever compiles (a build host,
# a CPU-only lint run) or that builds for another target.  These names are
# therefore computed the first time they are read, so the failure -- and the
# error message naming the variable to set -- lands at the point of use.
_ARCH_DEPENDENT_NAMES = frozenset(
    {
        "MUDNN_JIT_MUSA_ARCH_LIST",
        "MUDNN_JIT_MUSA_ARCH_KEY",
        "MUDNN_JIT_MUSA_TARGET_FLAGS",
        "MUDNN_JIT_WORKSPACE_DIR",
        "MUDNN_JIT_GEN_SRC_DIR",
        "MUDNN_JIT_DIR",
    }
)


def _compute_arch_dependent() -> dict[str, object]:
    arch_list = resolve_musa_arch_list()
    workspace_dir = MUDNN_JIT_CACHE_DIR / MUDNN_JIT_BASE_VERSION / "_".join(arch_list)
    return {
        "MUDNN_JIT_MUSA_ARCH_LIST": arch_list,
        "MUDNN_JIT_MUSA_ARCH_KEY": "_".join(arch_list),
        "MUDNN_JIT_MUSA_TARGET_FLAGS": tuple(
            f"--offload-arch=mp_{arch[2:]}" for arch in arch_list
        ),
        "MUDNN_JIT_WORKSPACE_DIR": workspace_dir,
        "MUDNN_JIT_GEN_SRC_DIR": workspace_dir / "generated",
        "MUDNN_JIT_DIR": workspace_dir / "cached_ops",
    }


def __getattr__(name: str):
    """PEP 562 hook: materialize the arch-dependent constants on first access."""
    if name in _ARCH_DEPENDENT_NAMES:
        computed = _compute_arch_dependent()
        globals().update(computed)
        return computed[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MUDNN_JIT_BASE_DIR",
    "MUDNN_JIT_BASE_VERSION",
    "MUDNN_JIT_CACHE_DIR",
    "MUDNN_JIT_CSRC_DIR",
    "MUDNN_JIT_DATA",
    "MUDNN_JIT_INCLUDE_DIR",
    "MUDNN_JIT_TEMPLATE_DIR",
    "MUDNN_JIT_VERSION",
    "MUTLASS_INCLUDE_DIR",
    "MusaArch",
    "resolve_csrc_dir",
    "resolve_include_dir",
    "resolve_kernels_dir",
    "resolve_musa_arch_key",
    "resolve_musa_arch_list",
    "resolve_musa_generated_dir",
    "resolve_musa_jit_dir",
    "resolve_musa_target_flags",
    "resolve_musa_workspace_dir",
    "resolve_mutlass_include_dir",
    "resolve_runtime_version_string",
]

