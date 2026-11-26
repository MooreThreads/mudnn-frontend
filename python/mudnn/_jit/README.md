# `mudnn._jit` — runtime JIT for MUSA C++/mutlass kernels

Compiles MUSA kernel sources with `mcc` (through ninja) at **runtime** and loads
the result with `tvm_ffi`. No pre-build step, no CMake target, nothing to wire
into the wheel build beyond the package itself.

Ported from the upstream JIT.

## Why you would use it

Writing a MUSA kernel as a header/template plus a small `JitSpec` means the
kernel ships as source and is compiled on the machine that runs it. That is the
same model TileLang kernels in this repo already use, applied to hand-written
mutlass/C++ kernels.

## Minimal use

```python
from pathlib import Path
from mudnn._jit import gen_jit_spec

spec = gen_jit_spec(
    "my_kernel",                       # module + build-directory name
    [Path("kernels/my_kernel.mu")],    # .mu / .cu / .cpp sources
    extra_include_paths=[Path("kernels/include")],
    extra_cuda_cflags=["-O3"],
)

module = spec.build_and_load()         # compiles on first call, then cached
module.my_kernel(a, b, c, scale)       # TVM-FFI typed call
```

A kernel exports itself with the usual TVM-FFI macro:

```cpp
void my_kernel(ffi::TensorView a, ffi::TensorView c, double scale) { /* ... */ }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(my_kernel, my_kernel);
```

See `kernels/smoke/vec_add.mu` for a complete, dependency-free example.

## Templated kernels

`gen_jit_spec` accepts `generated_sources` — a mapping from an output path to
either a source string or a `JitSourceTemplate` (a Jinja template plus its
context). This is how a kernel is specialized per dtype, arch or feature set;
the callers in the tree build the spec name from the same axes so two
specializations cannot collide.

## Where things live

| | checkout | installed wheel |
|---|---|---|
| kernel sources | `kernels/` | `mudnn/data/kernels/` |
| kernel headers | `kernels/include/` | `mudnn/data/include/` |
| templates | `kernels/csrc/` | `mudnn/data/csrc/` |
| build cache | `~/.cache/mudnn/jit/<version>/<mpXX>/` | same |

Both layouts resolve through `mudnn._jit.env._resolve_repo_or_packaged`, which
prefers the checkout while developing and the packaged copy when installed.

## Caching

The first `build_and_load()` compiles; the result is cached

* **on disk** at `~/.cache/mudnn/jit/<version>/<mpXX>/cached_ops/<name>/`, keyed
  by architecture, so ninja re-links only when a source, template, header or
  flag actually changed;
* **in process** for callers that hold on to the module.

`mudnn._jit.prewarm(...)` / `build_jit_specs(...)` compile a batch of specs ahead
of use, but that is an accelerator only: nothing here requires a pre-build, and a
kernel compiled on first use behaves identically. Runtime JIT is the only
execution path -- there is no ahead-of-time mode.

## Environment variables

| Variable | Purpose |
|---|---|
| `MUDNN_JIT_MUSA_ARCH_LIST` | e.g. `3.1`, or `3.1 3.2`. Required when no MUSA device is visible; also the only way to cross-build |
| `MUDNN_JIT_MUSA_HOME` / `MUSA_HOME` | MUSA SDK prefix when it is not discoverable from `mcc` (e.g. `/usr/local/musa`) |
| `MUDNN_JIT_MCC` | path to `mcc` |
| `MUDNN_JIT_NINJA` | path to `ninja` |
| `MUDNN_JIT_VERBOSE` | `1` streams ninja output |
| `MUDNN_JIT_WORKSPACE_BASE` | base directory of the build cache |
| `MUDNN_JIT_EXTRA_CFLAGS` / `MUDNN_JIT_EXTRA_MUSAFLAGS` / `MUDNN_JIT_EXTRA_LDFLAGS` | extra flags |
| `MAX_JOBS` | ninja parallelism |

## Requirements

* a MUSA SDK providing `mcc` **>= 5.1.0** (runtime toolchain, not a Python dep)
* `ninja` on `PATH`
* `tvm_ffi` — the MUSA build is not on PyPI; install the vendor wheel
  (`apache_tvm_ffi-*.whl`) alongside the `jit` extra:
  `pip install -e ".[jit]"`

## Architecture handling

The architecture is resolved once, from `MUDNN_JIT_MUSA_ARCH_LIST` when set and
from the visible device otherwise, and it feeds three things that **must agree**:
the rendered `ArchTag`, the `--offload-arch` flag, and the cache namespace. A
mismatch between the first two makes `mcc` compile the architecture-specific
instruction bodies out and produce an empty commit group, so they are derived
from one call rather than passed in separately.

Arch-dependent names (`MUDNN_JIT_MUSA_ARCH_LIST`, `MUDNN_JIT_DIR`, …) are
resolved lazily on first access, so importing the package never requires a
device.

## Deliberate non-features

* No automatic dependency tracking beyond the include/template scan in
  `_dependencies.py`; a changed header is picked up because ninja's depfile
  lists it.
* No cross-architecture single module: one build directory per architecture.
