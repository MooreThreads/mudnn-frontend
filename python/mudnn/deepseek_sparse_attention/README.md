## DSA module

Layout, class and wrapper signatures follow cuDNN frontend
`python/cudnn/deepseek_sparse_attention`. MP31 and MP32 take the
places of SM90 and SM100.

- **Sparse Attention Forward** (`SparseAttentionForward`,
  `sparse_attention_forward_wrapper`): native TileLang sparse prefill for
  BF16 QK=V=512, H64/H128, arbitrary logical K, invalid/OOB
  and duplicate indices, per-query lengths, attention sinks and indexer LSE
  (`indexer_topk` in 0/512/1024/2048 for H64, 0/512/1024 for H128). MP31 runs
  H128 in one launch with two 64-head CTA groups per query,
  writing directly into the final output. The requested indexer LSE is captured
  during the same forward pass, including the MP32 H64/B128 direct-ring path.
- **Sparse Attention Backward** (`SparseAttentionBackward`,
  `sparse_attention_backward_wrapper`): consumes the forward `out` and KV-only
  `lse`; returns BF16 `dq`/`dkv` and FP32 `d_sink`. Top-K is padded to 64 and
  limited to 8192. dKV and d_sink use FP32 atomics, so `deterministic=True`,
  `block_tile!=64` and `q_cluster_mode='adaptive_pair'` raise
  `NotImplementedError`.

- **Everything else** (indexer forward, Top-K family, score recompute, both
  indexer backwards, the compact score+TopK bundle and the two candidate-buffer
  sizing helpers) is a torch fallback in `_torch_fallback.py`. Those entry points
  exist with the upstream argument list and documented semantics, but compute the
  dense score tile instead of running a MUSA kernel; they are the reference path,
  not an optimized one.

### Upstream argument parity

Every public wrapper accepts the upstream positional and keyword arguments in
the upstream order, including the ones that only tune the CUDA kernels
(`m_block_size`, `n_block_size`, `q_stage`, `kv_stage`, `sf_vec_size`,
`num_copy_bits`, `packed_window_plan`). Purely performance-shaped arguments are
accepted and ignored; they never change results. The upstream call sites in
`megatron-lm-musa-patch` therefore work unchanged:

| Caller probe | What it needs | Result here |
|---|---|---|
| `msa_patch/fused_compressor.py::_get_frontend` | `cudnn.csa.compressor.csa_compressor_{forward,backward}_wrapper` | available as `mudnn.csa.compressor.*` / `mudnn.CSA.*` |
| `csa_utils/fused_sparse_attention.py::_compact_indexer_available` | callable `DSA.indexer_forward_top_k_wrapper` | present |
| `prepare_{bshd,thd}_compact_indexer_workspace` | `DSA.compress_topk_cand_buffer_size{,_thd}` | present, upstream-exact sizing |
| `musa_patch/topk_window_patch.py` | `packed_window_plan` in the compact wrapper signature | present (hint accepted, results unchanged) |

FP8/MXFP8 is the one thing the fallbacks refuse: `precision='fp8'`/`'mxfp8'` or
any `q_scale`/`k_scale` raises `NotImplementedError` rather than silently
returning BF16 numbers.

### Ratio-causal alignment

`ratio` masking follows the upstream top-left contract: row `q` of segment `b`
sees the first `clamp((q_causal_offsets[b] + q + 1) // ratio, 0, S_k)` compressed
KV positions. A decode-shaped query (`S_q < S_k`) or a context-parallel segment
must pass `q_causal_offsets`; without it the window starts at position 0. There
is no implicit bottom-right alignment.

The MP31 forward kernel in `dsa_fwd_mp31.py` ports the original MP31
generic schedule. Its current validation uses the 2026-10-09 TileLang
`0.1.12+musa.3.gitcd37c616` and matching mcc/runtime; results from other
toolchain combinations are recorded separately.

### Prepared execution and wrappers

The public classes follow `check_support()` → `compile()` → `execute()`.
`compile()` prepares the device kernels from the sample tensor metadata before
execution. `execute()` does not compile or allocate tensor storage. An empty
forward plan fills the caller's buffers without launching the attention kernel.

Use the wrappers for automatic preparation. They stage strided inputs,
allocate omitted outputs, and cache plans by tensor metadata. The forward
wrapper also aligns input pointers and pads the INDEX width to a multiple of
64 with `-1`. Calls with
different shapes select matching plans; `softmax_scale` remains a runtime
argument. Allocation and staging run on the selected stream.

For direct `SparseAttentionForward` use, prepare contiguous inputs and pad the
INDEX width before constructing the plan. Runtime shapes, strides, dtypes,
devices and optional-input presence must match the samples. Q/KV require
16-byte aligned pointers, INDEX requires 32-byte alignment, and sink/lengths
require 4-byte alignment. Pass these contiguous output buffers on Q's device
to `execute()`:

| Buffer | Shape | Type |
|---|---|---|
| `out` | `(S_q, H, 512)` | BF16, 16-byte aligned |
| `max_logits`, `lse` | `(S_q, H)` | FP32 |
| `lse_indexer` | `(S_q, H)` | FP32; required when `indexer_topk > 0`, otherwise `None` |

Outputs must not overlap inputs or each other. Read-only forward inputs may
share storage. Concurrent calls need separate output buffers. No forward
scratch buffer is required by the public interface.

For direct backward execution, preallocate `dq`, `dkv`, keyword-only `d_sink`,
and a contiguous one-dimensional uint8 `workspace` of at least
`scratch_workspace_bytes()` bytes with 256-byte alignment. Inputs must match
the plan and be contiguous; inputs, outputs and workspace must not overlap.
The backward wrapper handles strided inputs/outputs and omitted output or
workspace allocation. Represent an absent backward sink with an FP32 `(H,)`
tensor filled with `-inf`.

The internal backward stages are named `preprocess`, `gradient` and
`postprocess`. Both architectures use row preprocessing for Delta,
sink-adjusted LSE and atomic `d_sink`; MP31 reuses the row kernel compiled for
MP31. MP32 `postprocess` converts FP32 dKV to BF16 and reverses its internal
4×4 column permutation. MP31 converts its ordinary dKV layout with `copy_`.
These internal names do not change the public API.

MP32 backward uses one persistent CTA per MP, with 256 C0 threads, 256 C1
threads and 128 producer threads. Two KV32 slots overlap gather with compute;
early dS publication and per-chunk Q/dO release overlap work across queries.
Empty queries execute one fully masked pair so every role completes the
release protocol. The kernel supports the same variable lengths and padded
Top-K values through 8192, and retains BF16 operands with FP32 accumulation.

Full 64-head groups use the caller's Q/out/dout/dQ storage directly. The
workspace no longer reserves four unused BF16 staging copies for those groups:
at SQ32768/D512 this removes 8 GiB for H64 or 16 GiB for H128 from the previous
layout. Always query `scratch_workspace_bytes()` for the actual plan size;
smaller head groups still need padding storage.

### Statistics and masking

`lse` and `lse_indexer` use natural logarithms and exclude the attention sink;
the sink participates in the output denominator. `max_logits` is the maximum
valid scaled KV logit. Invalid/OOB slots contribute nothing, while repeated
valid indices remain repeated attention entries. Causality is encoded by the
caller in INDEX.

`topk_length` counts slots in the row prefix, not valid indices. Negative
lengths act as zero; lengths beyond the row width are bounded by the kernel's
prepared INDEX width, whose padded slots are invalid. An all-invalid row
returns zero output, `max_logits=-inf` and `lse=+inf`. The indexer statistic
is `+inf` if its prefix has no valid entries or `topk_length < indexer_topk`.

### DSv4.1 sparse + sliding-window attention

There is no separate SWA API. Put both KV pools in one `kv` tensor, e.g.
`kv = [kv_compressed; kv_window]`, and build each index row as

```text
[ indexer top-K ids (512) | window ids + kv_compressed.shape[0] (128) ]
```

with `-1` for unused slots. Both parts share one softmax and the sink.
`indexer_topk=512` returns `lse_indexer` over the sparse prefix only, so do not
compact these rows; if `topk_length` is supplied it must be at least
`indexer_topk`. Split `dkv` at `kv_compressed.shape[0]` to recover the two pool
gradients.

```python
from mudnn import DSA

fwd = DSA.sparse_attention_forward_wrapper(q, kv, indices, attn_sink=sink, indexer_topk=512)
bwd = DSA.sparse_attention_backward_wrapper(q, kv, fwd["out"], dout, fwd["lse"], sink, indices)
dkv_compressed, dkv_window = bwd["dkv"].split([n_compressed, kv.shape[0] - n_compressed])
```
