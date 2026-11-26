# Experimental cuDNN-algorithm GDN backward port

This standalone first-order backward is a MUSA/TileLang translation of the
chunk-64 WY inverse-adjoint algorithm in NVIDIA cuDNN frontend commit
`33c6fcb54d110f2c4fbeaf7c0330067492142189`. It is not a literal CUDA/CuTe
kernel copy and does not preserve NVIDIA's scheduling. The standalone backward
API is not registered as its own engine or graph node. Its implementation is
also used by the `gdn_tilelang_v2` training plan behind the canonical GDN API.

```python
from mudnn.linear_attention.experimental import (
    gdn_bwd,
    prepare_gdn_bwd,
)

grads = gdn_bwd(dO, q, k, v, g, beta, cu_seqlens)
# dQ, dK, dV, dG, dBeta, d_initial_state, d_a_log, d_dt_bias

prepared = prepare_gdn_bwd(dO, q, k, v, g, beta, cu_seqlens)
grads = prepared.run()

reused = gdn_bwd(
    dO, q, k, v, g, beta, cu_seqlens,
    state_checkpoints=prepared.checkpoints,
    checkpoint_every_n_tokens=64,
)
```

Preparation transforms inputs, computes chunk inverse factors, recomputes
state checkpoints if none are supplied, and caches forward-only adjoint maps
and FP32 within-chunk pair decays.
`run()` allocates outputs and scratch space, executes all dO-dependent work,
and applies input-transform chain rules. Neither entry point constructs an
autograd graph. Missing optional gradients are empty device tensors.
Prepared objects retain their inputs and forward caches; do not mutate those
inputs or reuse a prepared object for a different forward evaluation.

## Supported prototype inputs

- MP32 MUSA devices only; unsupported architectures are explicitly rejected.
- Packed THD Q/K/V/dO in a common FP16 or BF16 dtype. K and V dimensions are
  independently 64 or 128.
- Either direction of compatible integer head grouping; K heads must match Q
  or V heads, and gradients are reduced back to each input's head count.
- Log-space gates in FP32/FP16/BF16, beta in FP32 or the IO dtype.
- Int32/int64 device CU boundaries, including empty and sub-chunk sequences.
  Validation copies CU boundaries to the host and synchronizes; this is not a
  device-only scheduling or graph-capture implementation.
- Initial/final-state gradients in matching FP32 or BF16, with state shape
  `[N, HO, V, K]`; packed batches may carry state.
- Q/K L2 normalization, beta sigmoid, negative-eigenvalue beta scaling,
  safe-gate parameter gradients, and linear-domain gates.
- Cadence-64 IO-dtype checkpoints with shape
  `[max(total // 64 + N, 1), HO, V, K]`. Unused capacity rows are padding.
  Self-generated checkpoint reuse is covered by bitwise-equality checks.
  Exchange with a checkpoint serialized by the NVIDIA kernel has not been
  validated.

Cadences other than 0/64, `batch_invariant=True`, other plan names, invalid
shapes/devices/dtypes, and invalid CU boundaries are explicitly rejected.
Cadence 0 means recompute when no checkpoints are supplied. The prototype does
not implement higher-order differentiation, a standalone forward replacement,
native graph integration, or full frontend parity.

## Execution and production boundary

The shared implementation lives in `tilelang/gdn_backward.py` and
`tilelang/kernels/training/`. The v2 training engine imports that implementation
directly. This experimental package retains its standalone public API as a
compatibility facade; it does not maintain a second set of math kernels.

The kernels use the validated fixed configuration. Import-time tuning
experiments, arbitrary compiler flag replacements, timing instrumentation and
calculation-skipping ablations have been removed. The persistent body keeps
512 threads, a 244-register cap, and post-RA scheduling above 32 heads.

Preparation builds inverse/prefix factors, recomputes checkpoints when needed,
and prepares the forward-dependent maps and pair coefficients. Backward uses
the cut path when state-transfer bounds permit it, otherwise a TME-pipelined
state scan; dimension-specific non-persistent kernels remain available for the
standalone API's supported K/V combinations. These choices follow input data
and supported dimensions rather than process-wide experimental switches.

GEMM accumulators, scanned states and scalar gradient accumulation remain FP32;
cached maps and pair coefficients retain the original IO dtype. Device-side
prefix scheduling and graph-capture training support are unchanged.

## Numerical scope and reproduction

The independent validation harness compares all applicable gradients against
CPU FP64 recurrence gradients with a relative-L2 threshold of 0.02. Its 13
numeric cases cover all four K/V dimension pairs, both IO dtypes, grouped heads,
ragged/empty/short sequences, FP32/BF16 states, both CU integer dtypes, and
transform gradients. It also checks eight rejection cases, checkpoint reuse,
and the absence of autograd graphs. This is bounded evidence, not exhaustive
cuDNN conformance or training-convergence certification.

The current implementation passed all 13 numerical cases and eight rejection
cases again on 2026-10-21 (21/21). The first optimized version passed them, and
three focused repository regressions, on 2026-10-14. Its largest
observed relative-L2 error was 0.00577012 (0.5770%), in a safe-gate dt-bias
gradient. The initial unoptimized implementation's maximum was 0.0054083.
Optimization measurements and candidate-specific validation are preserved in
`gdn-bwd-opt-results-20261014` beside this worktree; use its final
`validation.json` and `benchmark.json` rather than treating an intermediate
profile's two-input accuracy check as full validation.

Focused regression command, inside a GPU reservation:

```bash
python -m pytest -q tests/python/test_ops/test_gdn_bwd.py
```

The legacy backward has independent-gradient numerical failures on the fixed
benchmark inputs. Its latency is a scheduling reference, not an equal-correctness
speedup baseline. The historical measurements above do not qualify the default
training route; current routing and its validation are documented in the repository README and covered by
`test_gdn.py`.
