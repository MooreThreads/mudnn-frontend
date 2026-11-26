"""mudnn._jit: JIT compilation of MUSA C++/mutlass kernels.

The facility compiles kernel sources with ``mcc`` through ninja at runtime and
loads the result with ``tvm_ffi``.  Callers describe a kernel once as a
:class:`JitSpec` (via :func:`gen_jit_spec`) and call ``build_and_load()``; the
result is cached in-process and on disk, so the first call compiles and later
calls are cheap.  No pre-build step is required.

Ported from the upstream JIT.  The generic half is kept whole; the
operator-specific modules (gemm / flash-attention / msa / moe / ...) are not
part of this package.

Quick start::

    from pathlib import Path
    from mudnn._jit import gen_jit_spec, build_jit_specs, clear_cache_dir

    spec = gen_jit_spec(
        "my_kernel",
        [Path("kernels/my_kernel.mu")],
        extra_include_paths=[Path("kernels/include")],
    )
    module = spec.build_and_load()
    module.my_kernel(...)

Environment variables (all optional):

    MUDNN_JIT_MUSA_ARCH_LIST  explicit arch list, e.g. ``3.1`` or ``3.1 3.2``;
                              required when no MUSA device is visible
    MUDNN_JIT_MUSA_HOME       optional MUSA SDK prefix (``MUSA_HOME`` is also
                              accepted); otherwise infer it from ``mcc``
    MUDNN_JIT_MCC             path to mcc
    MUDNN_JIT_NINJA           path to ninja
    MUDNN_JIT_VERBOSE         ``1`` streams the ninja output
    MUDNN_JIT_WORKSPACE_BASE  base directory for the JIT cache
    MUDNN_JIT_EXTRA_CFLAGS / _MUSAFLAGS / _LDFLAGS   extra flags
    MAX_JOBS                  ninja parallelism
"""

from . import env as env
from ._dependencies import (
    JitDependencySpec as JitDependencySpec,
)
from ._dependencies import (
    JitSourceTemplate as JitSourceTemplate,
)
from ._dependencies import (
    ResolvedJitDependencies as ResolvedJitDependencies,
)
from ._dependencies import (
    resolve_jit_dependencies as resolve_jit_dependencies,
)
from .core import (
    GeneratedSource as GeneratedSource,
)
from .core import (
    JitSpec as JitSpec,
)
from .core import (
    JitSpecRegistry as JitSpecRegistry,
)
from .core import (
    JitSpecStatus as JitSpecStatus,
)
from .core import (
    build_jit_specs as build_jit_specs,
)
from .core import (
    clear_cache_dir as clear_cache_dir,
)
from .core import (
    gen_jit_spec as gen_jit_spec,
)
from .core import (
    get_tmpdir as get_tmpdir,
)
from .core import (
    jit_spec_registry as jit_spec_registry,
)
from .core import (
    temporary_max_jobs as temporary_max_jobs,
)
from .utils import (
    EXPORT_FUNC as EXPORT_FUNC,
)
from .utils import (
    TVM_HEADER as TVM_HEADER,
)
from .utils import (
    dtype_torch2mutlass_map as dtype_torch2mutlass_map,
)
from .utils import (
    maybe_contiguous as maybe_contiguous,
)
from .utils import (
    mutlass_element as mutlass_element,
)

__version__ = env.MUDNN_JIT_VERSION.base_version


def spec_statuses():
    """Per-spec compile status for everything generated in this process."""
    return jit_spec_registry.get_all_statuses()


def spec_stats():
    """Counts of JIT-compiled / not-compiled specs in this process."""
    return jit_spec_registry.get_stats()


def prewarm(specs=None, *, jobs=None, verbose=False):
    """Build the given specs (default: every registered one) ahead of use.

    Purely an accelerator: nothing in this package requires a pre-build, and a
    kernel compiled on first use behaves identically.
    """
    if specs is None:
        specs = list(jit_spec_registry.get_all_specs().values())
    build_jit_specs(list(specs), jobs=jobs, verbose=verbose)
    return specs


__all__ = [
    "EXPORT_FUNC",
    "GeneratedSource",
    "JitDependencySpec",
    "JitSourceTemplate",
    "JitSpec",
    "JitSpecRegistry",
    "JitSpecStatus",
    "ResolvedJitDependencies",
    "TVM_HEADER",
    "build_jit_specs",
    "clear_cache_dir",
    "dtype_torch2mutlass_map",
    "env",
    "gen_jit_spec",
    "get_tmpdir",
    "jit_spec_registry",
    "maybe_contiguous",
    "mutlass_element",
    "prewarm",
    "resolve_jit_dependencies",
    "spec_stats",
    "spec_statuses",
    "temporary_max_jobs",
]
