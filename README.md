# muDNN Frontend

## cuDNN Frontend compatibility

The public C++ headers and Python `mudnn` package target the cuDNN Frontend
0.0.1 source contract.  This includes the legacy `_v8` builders, Modern Graph
overloads and return types, backend C attribute aliases, and the structured
`context`/`nodes`/`tensors` JSON protocol.  The Python package provides the
same `_compiled_module` import surface while dispatching to muDNN through its
ctypes backend.

Compatibility does not imply that every CUDA/cuDNN kernel has a muDNN backend
implementation.  In particular, FP8/MXFP8 SDPA and execution-plan timing for
autotune are not currently exposed by muDNN.  Their public methods raise or
return an explicit `NOT_SUPPORTED` result immediately; they never create a
placeholder graph or report a successful no-op.  Kernel-cache attachment to a
legacy execution plan is rejected for the same reason.

`mudnn-frontend` is a compatibility layer for building muDNN backend graph
descriptors with a cuDNN-frontend-like API.

`mudnn-frontend` is intended to be used as a standalone repository on top of an
existing muDNN backend installation or staged developer build. Source
`env.sh.example` before building frontend test modules:

```bash
source env.sh.example
./build_frontend_tests.sh
```

Run pytest explicitly after the pybind test modules have been built:

```bash
export MUDNN_FRONTEND_FRAMEWORK_TEST_BUILD_DIR=$PWD/build/tests/cpp
export MUDNN_FRONTEND_CPP_TEST_BUILD_DIR=$PWD/build/tests/cpp
pytest -q tests/python/framework/test_framework_graph.py tests/python/cpp
```

By default the environment file uses the installed backend at
`/usr/local/musa`. Override these variables for another backend:

```bash
export MUDNN_BACKEND_ROOT=/usr/local/musa
export MUDNN_BACKEND_INCLUDE=$MUDNN_BACKEND_ROOT/include
export MUDNN_BACKEND_LIB=$MUDNN_BACKEND_ROOT/lib
```

### Installing the Python package

`pip install .` installs `mudnn` with everything the JIT needs at runtime: the
KDA kernel sources and mutlass headers are staged into `mudnn/data/` by
`build_backend.py`. Those sources include the `3rdparty/mutlass` submodule, so
clone with it:

```bash
git clone --recursive <repo> && pip install .
```

A plain `git clone` leaves `3rdparty/mutlass` empty; the build initialises it on
the fly, and if it cannot (no git, no credentials, or a tree without `.git`) the
build fails naming the missing path instead of installing a package whose
kernels can never compile. A wheel or an sdist needs no submodule, because both
carry the staged `mudnn/data/` tree:

```bash
pip install dist/mudnn_frontend-0.1.0-py3-none-any.whl
pip install dist/mudnn_frontend-0.1.0.tar.gz
```

The first KDA call then compiles the kernels for the resolved architecture with
`mcc` through ninja and caches them under `~/.cache/mudnn/jit/`.

#### What the wheel installs

The wheel layout matches cuDNN Frontend: the Python packages and the C++ header
tree, both rooted at the install prefix.

| Top level | Contents |
|---|---|
| `mudnn/`, `mudnn_frontend/`, `cudnn_frontend/` | Python API, plus the staged `mudnn/data/` JIT sources |
| `include/` | `mudnn_frontend.h`, `mudnn_frontend_*.h` and `mudnn_frontend/` headers |

So a C++ consumer needs no `cmake --install` step — point the compiler at the
site-packages root:

```bash
g++ -std=c++17 -I"$(python -c 'import mudnn, pathlib; print(pathlib.Path(mudnn.__file__).parent.parent)')" ...
```

The backend (`mudnn.h` and the MUSA runtime) still comes from the muDNN install,
so add `-I$MUDNN_BACKEND_INCLUDE` and the MUSA toolkit include as usual. A
mismatched backend is a compile error here, by design: the frontend headers name
backend symbols from the compatibility contract, not from any old `mudnn.h`.

No shipped header starts with `cudnn`: the C++ API is named `mudnn_frontend`
exclusively, so installing this wheel can never overwrite a header owned by
`nvidia-cudnn-frontend` -- or be overwritten by it. The cuDNN-style *C API*
aliases (`cudnnHandle_t`, `CUDNN_*` constants) still live in
`mudnn_frontend.h`; only the filenames are muDNN-named.

The version lives in one place, `python/mudnn_frontend/_version.py`, read by
`pyproject.toml` through `[tool.setuptools.dynamic]`. `mudnn.__version__`,
`mudnn_frontend.__version__` and the serialized `cudnn_frontend_version` all
read it. `looseversion` is a dev-only dependency of the test suite; it is not
shipped. Install the test extras with `pip install --group dev`.

The frontend has two API surfaces and keeps separate test paths for them:

- Python API tests live under `tests/python/test_api`,
  `tests/python/test_graph`, and `tests/python/test_ops`; these import
  `mudnn_frontend` or `cudnn_frontend` directly.
- C++ API tests live under `tests/cpp` and are exposed to pytest through
  pybind modules. Pytest wrappers for C++-driven cases live under
  `tests/python/framework` or future `tests/python/cpp` suites.

Current coverage is intentionally small and focused on the backend graph
foundation:

- C++ header-only builders for tensors, pointwise/matmul/convolution/reduction
  descriptors, convolution forward/backward operations, resample/pooling
  forward/backward operations, concat/reshape operations, RNG operations, signal
  operations, gen-stats operations, BN finalize statistics operations,
  BN backward weights operations, batch-norm inference/training forward and
  backward operations, SDPA operations, paged-cache-load operations,
  block-scale quantize/dequantize
  operations, operation graphs, engines, engine configs, execution plans, and
  variant packs.
- Header-only C++ API under `include/mudnn_frontend.h`,
  `include/mudnn_frontend_*.h`, and `include/mudnn_frontend/`, mirroring the
  upstream cudnn-frontend include layout under mudnn names. No shipped header is
  named `cudnn*`: upstream's `cudnn_backend_base.h` and `cudnn_interface.h` are
  deliberately not provided, because their contents are cuDNN types that do not
  exist here and the filenames would collide with `nvidia-cudnn-frontend` in a
  shared prefix. The cuDNN-style C API
  aliases (`cudnnHandle_t`, `CUDNN_*` constants, `cudnnBackend*` wrappers) live
  in `mudnn_frontend.h` itself.
- Python builder objects for lightweight graph construction tests and scripting,
  including pointwise, matmul, convolution forward/backward, reduction,
  resample/pooling, concat, reshape, RNG, signal, gen-stats, BN finalize,
  BN backward weights, and norm-forward operation descriptors, norm-backward operation descriptors,
  block-scale operation descriptors, paged-cache-load operation descriptors,
  plus SDPA operation descriptors.
- A Python `pygraph()` surface for cudnn-frontend-style graph construction,
  including tensor creation, tensor-like import, pointwise, matmul, convolution
  forward/backward-data/backward-filter, avg/max pooling forward, avg-pool
  backward, max-pool backward with generated index tensors, concat, reshape,
  RNG, batch-norm inference/training forward and
  backward, signal, gen-stats, BN finalize inference/training statistics,
  BN backward weights, paged-cache-load, block-scale
  quantize/dequantize, reduction, and SDPA nodes. The Python package exposes both
  `DataType`/`HeurMode`/`PointwiseMode` and cudnn-frontend-style lowercase
  aliases `data_type`/`heur_mode`/`pointwise_mode`, plus convenience graph
  methods such as `add`, `bias`, `sub`, `mul`, `relu`,
  `scaled_dot_product_attention`, chainable data-type setters such as
  `set_io_data_type`, `set_intermediate_data_type`, and `set_compute_data_type`,
  `get_execution_plan_count`, and `get_workspace_size`.

The C++ layer forwards to `mudnnBackendCreateDescriptor`,
`mudnnBackendSetAttribute`, `mudnnBackendFinalize`, and
`mudnnBackendExecute`. Extend it by adding builder methods that map directly to
the corresponding `MUDNN_ATTR_*` backend graph attributes.

The CMake install exports frontend headers under `include/` and Python packages
under `python/`. Add `${mudnn_frontend_install_prefix}/python` to `PYTHONPATH`
to import `mudnn`, `mudnn_frontend`, or the compatibility alias
`cudnn_frontend`; the installed Python loader resolves the configured muDNN
backend `libmudnn.so` and its sibling muDNN component libraries automatically.
The header-only `mudnn_frontend` CMake interface target is installed by the
frontend repository's own CMake install path.

The Python `Graph` layer uses `ctypes` to lower pointwise, matmul, convolution
forward/backward-data/backward-filter, resample/pooling forward, avg-pool
backward, max-pool backward with generated index tensors, concat, reshape, RNG,
signal, gen-stats, batch-norm finalize inference/training statistics, BN
backward weights, batch-norm inference/training forward, batch-norm backward,
paged-cache-load, block-scale quantize/dequantize, reduction, and SDPA
forward/backward subsets into native muDNN backend graph descriptors, engine
heuristics, and execution plans when
`build_operation_graph()` / `build_plans()` are called. `execute()` accepts a
mapping from `GraphTensor`, tensor name, or tensor UID to device pointer,
creates a native variant pack, and calls `mudnnBackendExecute`.

SDPA backend graph coverage is intentionally narrow but executable. Inference
supports Q/K/V/O fixed-shape graphs through the muDNN SDPA math forward path.
The backend descriptor accepts additive bias and additive/boolean mask tensors,
with 2D/3D/4D attention-mask shapes and 2D padding-mask shape. The Python graph
frontend supports additive bias+mask fusion by inserting a virtual pointwise
add before SDPA, and backend graph execution supports boolean-mask+bias fusion
by materializing a temporary additive mask before calling math SDPA. The Python
graph API accepts cuDNN-style top-left diagonal causal masking
(`diagonal_band_right_bound=0`) as an alias for the current causal-mask flag,
and accepts per-operation `compute_data_type` for SDPA forward and backward.
Other cuDNN SDPA keywords such as ALiBi, explicit sequence-length tensors,
ragged max lengths, non-AUTO implementation modes, unfused-FMA requests,
block masks, RNG dumps, paged-attention table descriptors, and nonzero dropout
are accepted by the Python signature but reject with `NotImplementedError`
until backend graph execution supports them.
SDPA backward also accepts `rng_dump` and rejects non-`None` values explicitly
until dropout backward execution is implemented.
Training forward and backward use the muDNN flash-attention APIs for a
conservative dense subset: HALF/BFLOAT16 rank-4 Q/K/V/O tensors with
last-dimension stride 1, FP32 contiguous stats shaped `[batch, heads, seqlen_q, 1]`, no bias/mask/dropout/padding, and flash-supported head dimensions. Varlen,
general diagonal bands, training masks/bias/dropout, and broader flash shape
coverage remain future work.

## Linear attention (`mudnn.linear_attention`)

`mudnn.linear_attention.gated_delta_net` is a Gated DeltaNet (GDN) operation in
THD layout: token-packed `[total_tokens, heads, dim]` q/k/v, `[total_tokens, HO]` g/beta, and a `[N + 1]` `cu_seqlens` boundary vector. A dense batch of
equal-length sequences is `cu_seqlens = [0, T, 2T, ...]` and is the only form
that carries `initial_state` / `output_final_state`; a ragged batch rejects
both. The recurrent state is K-major `[N, HO, K, V]`.

The layers:

- `linear_attention/ops/gdn.py` — the backend-independent operation contract:
  `torch.library` custom ops for the inference forward and the backward, the
  autograd wiring, and the public `gated_delta_net`.
- `linear_attention/graph_analyzer.py` — one call normalized into the `Facts`
  record every engine judges from.
- `linear_attention/tilelang/` — the `gdn_tilelang` engine, with the TileLang
  MUSA kernels under `kernels/` and `kernels/engine_api.py` as the engine's
  entry points into them.
- `engines/manifest.py` — the static engine table. A call classifies to exactly
  one family, and that family's engines answer `check_support()` in turn;
  `plan_name=` pins one.

Support surface of the `gdn_tilelang` engine: MUSA only, bfloat16 q/k/v/beta,
float32 g and state, `DK == DV` in `{64, 128}`, chunk 64, at least two chunks
per sequence. `use_qk_l2norm_in_kernel`, `use_beta_sigmoid_in_kernel`,
`allow_neg_eigval`, `safe_gate` and grouped-query expansion are applied in torch
ahead of the op rather than fused into the kernel, so their gradients come from
autograd; `batch_invariant` and `checkpoint_every_n_tokens` are rejected.

### Kimi Delta Attention (KDA)

`mudnn.linear_attention.kimi_delta_attention` is a Kimi Delta Attention
operation in the same THD layout, mirroring cuDNN frontend's
`cudnn.linear_attention.kimi_delta_attention` argument for argument. It differs
from GDN in the two places the shapes show it: `g` is per-key-channel
`[total_tokens, HO, K]`, and the recurrent state is V-major `[N, HO, V, K]`.

The layers:

- `linear_attention/ops/kda.py` — the backend-independent contract: the
  `mudnn::kimi_delta_attention_fwd` custom op and its fake, and the public
  `kimi_delta_attention`. Forward only.
- `linear_attention/mutlass/kda_ops.py` — the JIT specs: it renders the kernel
  source, compiles it with `mcc` through ninja, and caches the module per
  architecture.
- `linear_attention/mutlass/kda_run.py` — the driver: the chunk-major workspace
  ABI and the two kernel launches.
- `linear_attention/mutlass/kda_engine.py` — the `kda_mutlass_mp31` and
  `kda_mutlass_mp32` engines.
- `kernels/include/mate/` and `kernels/csrc/` — the MUSA kernels for the
  JIT engines. They include `mutlass/`, which is the `3rdparty/mutlass` submodule.

Nothing is pre-built. The first call compiles both kernels for the resolved
architecture (`mp31` or `mp32`) and caches them under
`~/.cache/mudnn/jit/<version>/<arch>/`; `build_backend.py` is what puts the
kernel sources inside an installed wheel.

Support surface: MUSA only, `K == V == 128`, chunk 16, bf16/fp16 q/k/v with `g`
and `beta` in the same dtype, fp32 or bf16 state, int32/int64 `cu_seqlens`,
dense and ragged THD, grouped-query heads, `use_qk_l2norm_in_kernel`,
`initial_state` / `output_final_state`, safe gate (`a_log` / `dt_bias`, fp32)
and `checkpoint_every_n_tokens`. `use_beta_sigmoid_in_kernel=True` is
*required*, because the kernel sigmoids `beta` unconditionally and so does not
implement cuDNN frontend's default of `false`. `gate_domain="linear"`,
`batch_invariant`, `allow_neg_eigval`, `overwrite_initial_state` and
`state_indices` are declined by name rather than approximated.

`checkpoint_every_n_tokens` returns cuDNN's third output,
`state_checkpoints`: the recurrent state entering token `j * cadence` for each
`j`, packed per sequence, in the activation dtype. The cadence must be a
positive multiple of the kernel chunk (16), and the tensor is allocated to
cuDNN's capacity bound `max(total // cadence + N, 1)` of which only
`sum_b ((len_b - 1) // cadence + 1)` rows are written -- the slack is left
uninitialised, as cuDNN leaves it. Checkpointing only adds stores, so the
output and final state are bit-identical with and without it.

Tests live in `tests/python/test_ops/`; GDN/FLA/KDA device tests skip themselves
when `torch_musa` is unavailable (GDN and FLA also need `tilelang` / `fla`),
while CSA also has CPU contract tests:

```bash
pytest -q tests/python/test_ops/test_gdn.py tests/python/test_ops/test_kda.py \
  tests/python/test_ops/test_csa_compressor.py tests/python/test_ops/test_fla_compat.py
```

## CSA compressor (`mudnn.csa.compressor`)

The frontend also contains a TileLang MUSA implementation of
the gated pooling region used by CSA/HCA Compressor.  It accepts packed THD
inputs `kv`/`score` shaped `[T, coff * D]` (BF16), `ape` shaped
`[ratio, coff * D]` (FP32), INT32 `cu_seqlens` and `cu_seqlens_comp`, and
returns `TupleDict(out=...)` with shape `[total_comp, D]` (BF16):

```python
from mudnn.csa.compressor import (
    csa_compressor_backward_wrapper,
    csa_compressor_forward_wrapper,
)

result = csa_compressor_forward_wrapper(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio=4, coff=2
)
out = result["out"]

grads = csa_compressor_backward_wrapper(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out,
    ratio=4, coff=2,
)
d_kv, d_score, d_ape = (
    grads["grad_kv"], grads["grad_score"], grads["grad_ape"]
)
```

The v1 TileLang envelope is MUSA mp_31 or newer, `ratio in {2, 4, 128}`,
`coff in {1, 2}`, and contiguous tensors. Ratio 128 uses a separate
chunked-softmax forward family and is validated for
`head_dim in {128, 512}`. The implementation keeps the CuTe reference semantics:
`coff=2` uses the previous block's first projection half plus the current
block's second half, with the first block's previous half masked out; all
softmax and weighted accumulation is FP32 before one BF16 store.  The eager
oracle is self-contained in `tests/python/test_ops/test_csa_compressor.py`, as
in the competing suite, and is not part of the runtime package.

For a static output capacity, pass `total_comp=capacity` to the forward wrapper
and allocate `grad_out` with the same first dimension. Rows after the device-side
`cu_seqlens_comp[-1]` reproduce eager's row-0 gather in forward and are ignored
by backward. This avoids a host `.item()` and allows a warmed forward/backward
pair to be captured and replayed with `torch.musa.MUSAGraph`; callers must ensure
the capacity covers the true row count. A new TileLang configuration must be
compiled before capture. The backward class API accumulates into caller-owned
FP32 `grad_ape`, while the wrapper returns a fresh zero-based result. Because
that reduction uses atomics, strict deterministic mode rejects backward and
warn-only mode reports one warning.

The package is lazy: importing `mudnn` does not import TileLang.  JIT compilation
starts only from `CSACompressorForward.compile()` or the first wrapper call.
The short-ratio kernel uses register-resident windows, vec2 lane ownership,
coalesced global reads, vectorized output stores, and a 64-thread column
schedule without shared-memory barriers.  The ratio-128 forward maps
32-thread column groups across parallel window chunks.  Its backward follows
the competitor's staged chunked dataflow: score/APE and KV are staged in
shared memory, chunk partials are merged in fixed order, and dAPE is reduced
with one FP32 atomic per owned slot and CTA.  MP31 uses scalar lanes, a
16-chunk wide bucket for the larger windows, and small row aggregation for
D512 long packs; the static schedule rejects tiles above its 56 KiB shared
memory budget.  The standalone benchmark in
`benchmark/csa/bench_csa_compressor.py` compares this path with the streaming
fallback, reports class/API and MUSA graph timings, and provides the
performance regression gate. `gate_csa_compressor_r128.py` is the
numerical/dispatch gate; `reg_probe_csa_compressor_r128.py` checks generated
MUSA register, local-memory, spill, and shared-memory resources.
The kernel modules follow the competitor's family split:
`compressor_sm100.py` handles short ratios and `compressor_sm100_r128.py` handles
the dedicated ratio-128 schedule.

The test-case mapping against the competing CuTe DSL implementation is
tracked in [`doc/csa_compressor_test_coverage.md`](doc/csa_compressor_test_coverage.md),
including the full forward/backward shape matrix, static capacity, graph replay,
runtime hazards, ratio-128 schedule differences, and the benchmark/resource gates.
