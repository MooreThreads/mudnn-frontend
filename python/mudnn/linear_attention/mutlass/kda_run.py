"""Functional driver for the chunked KDA forward on MUSA.

Builds and calls the two JIT-compiled mutlass kernels ported from the upstream implementation --
``prepare`` (K1) and ``recurrence`` (K2) -- through :mod:`mudnn._jit`.

Nothing is pre-compiled: the first call compiles for the resolved architecture
and caches the module, exactly like the TileLang kernels in this repository.

This module is the kernel-facing half of the operator.  Argument validation,
the torch.library entry point and the engine wrapper live above it; keeping the
driver free of them is what lets it be exercised directly from a test.

The workspace layout mirrors ``kernels/include/mate/flat/kda/chunk_kda_workspace.hpp``
and must be kept in sync with it; the assertions below are the same ones that implementation
makes, so a divergence fails loudly instead of corrupting memory.
"""

from __future__ import annotations

import torch

from ..._jit import env as jit_env  # noqa: F401  (documents where kernels come from)
from . import kda_ops

__all__ = [
    "CHUNK",
    "HEAD_DIM",
    "RECORD_STRIDE",
    "allocate_kda_workspace",
    "kda_checkpoint_rows",
    "kda_chunk_state_rows",
    "run_chunk_kda",
]

# --- the chunk-major workspace ABI ------------------------------------------
# Mirrors ChunkKdaWorkspace in chunk_kda_workspace.hpp.
CHUNK = kda_ops.KDA_PREFILL_CHUNK_SIZE
HEAD_DIM = kda_ops.KDA_PREFILL_HEAD_DIM

_DECAYED_ELEMENTS = 2 * CHUNK * HEAD_DIM
_KRESTORED_ELEMENTS = CHUNK * HEAD_DIM
_P_AND_INVERSE_ELEMENTS = CHUNK * CHUNK
_TOTAL_ELEMENTS = HEAD_DIM

_DECAYED_OFFSET = 0
_KRESTORED_OFFSET = _DECAYED_OFFSET + _DECAYED_ELEMENTS
_P_OFFSET = _KRESTORED_OFFSET + _KRESTORED_ELEMENTS
_INVERSE_OFFSET = _P_OFFSET + _P_AND_INVERSE_ELEMENTS
_TOTAL_OFFSET = _INVERSE_OFFSET + _P_AND_INVERSE_ELEMENTS
# Backward-facing copies of the unfolded P and k_restored, appended after
# `total` (see chunk_kda_workspace.hpp).  The forward's fold overwrites the
# `p`/`k_restored` fields with A_rec and T, which K3/K4 cannot consume.
_P_BACKWARD_OFFSET = _TOTAL_OFFSET + _TOTAL_ELEMENTS
_KRESTORED_BACKWARD_OFFSET = _P_BACKWARD_OFFSET + _P_AND_INVERSE_ELEMENTS
# FP32 twin of the chunk decay product, in Element slots (two per float).  The
# bf16 `total` row rounds `exp2(sum(g))` to ~0.4 %, which is coarser than the
# per-chunk decay itself, so decaying the running state with it accumulates a
# linear error in the chunk count; the recurrence reads this row instead.
_DECAY_ELEMENTS = 2 * _TOTAL_ELEMENTS
_DECAY_OFFSET = _KRESTORED_BACKWARD_OFFSET + _KRESTORED_ELEMENTS
RECORD_STRIDE = _DECAY_OFFSET + _DECAY_ELEMENTS

assert RECORD_STRIDE == 9344, RECORD_STRIDE
assert _P_BACKWARD_OFFSET == 6784, _P_BACKWARD_OFFSET
assert _TOTAL_OFFSET == 6656, _TOTAL_OFFSET
assert _DECAY_OFFSET == 9088, _DECAY_OFFSET


def allocate_kda_workspace(records: int, heads: int, device, dtype):
    """Allocate the single chunk-major workspace and return its five views.

    Every field of every record lands on a 256 B boundary, which the kernels'
    descriptors require; the offsets and the record stride are all multiples of
    256 B and the allocation base is too, so this holds by construction -- and
    is asserted rather than trusted.
    """
    records, heads = int(records), int(heads)
    buffer = torch.empty((records, heads, RECORD_STRIDE), device=device, dtype=dtype)
    if buffer.element_size() != 2:
        raise ValueError(
            "the chunk-major workspace layout assumes 2-byte activation elements, "
            f"got {buffer.element_size()}"
        )
    row_stride = heads * RECORD_STRIDE

    def view(offset, shape, inner_stride):
        return torch.as_strided(
            buffer,
            size=shape,
            stride=(row_stride, RECORD_STRIDE) + inner_stride,
            storage_offset=offset,
        )

    decayed = view(_DECAYED_OFFSET, (records, heads, 2 * CHUNK, HEAD_DIM), (HEAD_DIM, 1))
    k_restored = view(_KRESTORED_OFFSET, (records, heads, CHUNK, HEAD_DIM), (HEAD_DIM, 1))
    inverse = view(_INVERSE_OFFSET, (records, heads, CHUNK, CHUNK), (CHUNK, 1))
    p = view(_P_OFFSET, (records, heads, CHUNK, CHUNK), (CHUNK, 1))
    total = view(_TOTAL_OFFSET, (records, heads, HEAD_DIM), (1,))

    if buffer.data_ptr() % 256:
        raise AssertionError(
            f"workspace base {buffer.data_ptr():#x} is not 256 B aligned"
        )
    for name, tensor in (
        ("decayed", decayed),
        ("k_restored", k_restored),
        ("inverse", inverse),
        ("p", p),
        ("total", total),
    ):
        if tensor.data_ptr() % 256:
            raise AssertionError(
                f"{name} view at {tensor.data_ptr():#x} is not 256 B aligned"
            )
    return decayed, k_restored, inverse, p, total


def kda_chunk_state_rows(
    *, batch: int, num_tokens: int, cu_seqlens: torch.Tensor | None = None
) -> int:
    """Rows needed for the packed chunk-slot ``chunk_states`` pool."""
    chunks = -(-num_tokens // CHUNK)
    if cu_seqlens is None:
        return batch * chunks
    return chunks + int(cu_seqlens.numel() - 1)


def kda_checkpoint_rows(
    *,
    batch: int,
    num_tokens: int,
    cu_seqlens: torch.Tensor | None = None,
    every_n_tokens: int,
) -> int:
    """Rows needed for the compact checkpoint pool.

    ``max(total_tokens // C + N, 1)``, the same bound cudnn-frontend allocates;
    it covers the exact packed count ``sum_b ((len_b - 1) // C + 1)``.
    """
    if every_n_tokens <= 0:
        raise ValueError("every_n_tokens must be positive.")
    if every_n_tokens % CHUNK:
        raise ValueError(
            f"every_n_tokens must be a multiple of the kernel chunk ({CHUNK}), "
            f"got {every_n_tokens}."
        )
    sequences = batch if cu_seqlens is None else int(cu_seqlens.numel() - 1)
    total_tokens = batch * num_tokens if cu_seqlens is None else num_tokens
    return max(total_tokens // every_n_tokens + sequences, 1)


def _config(
    q,
    *,
    initial_state,
    final_state,
    A_log,
    dt_bias,
    normalize_qk,
    cu_seqlens,
    checkpoint_every_n_tokens=0,
    gate_dtype=None,
    beta_dtype=None,
    use_beta_sigmoid=True,
    allow_neg_eigval=False,
    gate_domain="log2",
):
    """The JIT specialization this call needs.

    Every option here changes the generated source or its function name, so the
    module cache key follows the same axes.
    """
    state = initial_state if initial_state is not None else final_state
    state_dtype = state.dtype if state is not None else q.dtype
    cadence = int(checkpoint_every_n_tokens or 0)
    return kda_ops.make_kda_ops_config(
        q.dtype,
        gate_dtype=gate_dtype,
        beta_dtype=beta_dtype,
        use_beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        gate_domain=gate_domain,
        state_dtype=state_dtype,
        cu_seqlens_dtype=cu_seqlens.dtype if cu_seqlens is not None else None,
        has_state_in=initial_state is not None,
        has_state_out=final_state is not None,
        # The checkpoint pool travels through the kernel's `chunk_states`
        # argument; a cadence is the only thing that reaches it from the
        # operator, so the two axes move together.
        has_chunk_states_out=cadence > 0,
        checkpoint_every_n_tokens=cadence,
        state_fp32=state_dtype == torch.float32,
        has_gate_params=A_log is not None,
        is_varlen=cu_seqlens is not None,
        normalize_qk=bool(normalize_qk),
    )


def _as_4d(x, *, is_varlen):
    """The kernels' ``[B, T, H, D]`` view of a ``[S, H, D]`` varlen input."""
    if not is_varlen:
        return x
    return x if x.dim() == 4 else x.unsqueeze(0)


@torch.no_grad()
def run_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    use_qk_l2norm_in_kernel: bool = True,
    output: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    chunk_states: torch.Tensor | None = None,
    state_checkpoints: torch.Tensor | None = None,
    checkpoint_every_n_tokens: int = 0,
    use_beta_sigmoid=True,
    allow_neg_eigval=False,
    gate_domain="log2",
):
    """Chunk KDA forward.

    Accepts either dense ``[B, T, H, D]`` tensors or the packed-varlen form
    ``[S, H, D]`` with ``cu_seqlens``; the kernels always see ``[B, T, H, D]``
    with ``B == 1`` in the varlen case, matching the upstream driver.

    Returns ``output``, or ``(output, final_state)`` when
    ``output_final_state`` is set; with ``checkpoint_every_n_tokens`` it returns
    ``(output, final_state, state_checkpoints)`` instead, where the checkpoint
    tensor is the caller's ``state_checkpoints`` or one this allocates.

    ``initial_state_indices`` selects pool rows for both state reads and writes.
    The legacy ``chunk_states`` path is not exposed by the operator contract.
    """
    if chunk_states is not None:
        # Only the checkpoint cadence reaches the pool from the operator; the
        # packed chunk-slot layout has no counterpart in its contract.
        raise NotImplementedError(
            "run_chunk_kda: the legacy chunk_states pool is not exposed by the "
            "kimi_delta_attention contract; use checkpoint_every_n_tokens"
        )
    cadence = int(checkpoint_every_n_tokens or 0)
    if q.shape[-1] != HEAD_DIM or v.shape[-1] != HEAD_DIM:
        raise ValueError(f"head dimension must be {HEAD_DIM}")

    is_varlen = cu_seqlens is not None
    if is_varlen:
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must be int32 or int64")
        cu_seqlens = cu_seqlens.contiguous()
    q, k, v, g = (_as_4d(t, is_varlen=is_varlen) for t in (q, k, v, g))
    if is_varlen and beta.dim() == 2:
        beta = beta.unsqueeze(0)
    beta = beta.contiguous()
    if not is_varlen and q.dim() != 4:
        raise ValueError("expected dense [B, T, H, D] tensors")

    if scale is None:
        scale = HEAD_DIM**-0.5
    squeeze_varlen = is_varlen and output is None and q.shape[0] == 1
    if output is None:
        output = torch.empty_like(v)
    if output_final_state and final_state is None:
        final_state = torch.empty(
            (
                q.shape[0] if not is_varlen else int(cu_seqlens.numel() - 1),
                v.shape[2],
                HEAD_DIM,
                HEAD_DIM,
            ),
            device=q.device,
            dtype=initial_state.dtype if initial_state is not None else q.dtype,
        )
    state_out = final_state if output_final_state else None

    config = _config(
        q,
        initial_state=initial_state,
        final_state=state_out,
        A_log=A_log,
        dt_bias=dt_bias,
        normalize_qk=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        checkpoint_every_n_tokens=cadence,
        gate_dtype=g.dtype,
        beta_dtype=beta.dtype,
        use_beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        gate_domain=gate_domain,
    )

    batch, tokens, value_heads = q.shape[0], q.shape[1], v.shape[2]
    chunks = -(-tokens // CHUNK)
    nseq = int(cu_seqlens.numel() - 1) if is_varlen else batch
    # Varlen packs each sequence's chunks contiguously and leaves one spare
    # record per sequence, so the prepare kernel can be partitioned per CTA.
    records = chunks + nseq if is_varlen else nseq * chunks

    workspace = allocate_kda_workspace(records, value_heads, q.device, q.dtype)
    state_indices = (
        torch.arange(nseq, device=q.device, dtype=torch.int32)
        if initial_state_indices is None
        else initial_state_indices
    )
    # One int4 range descriptor per launched prepare partition; only a
    # multi-sequence varlen launch has more than one.
    metadata_capacity = (
        kda_ops.resolve_kda_prepare_ctas_per_mp() * _physical_num_mps(q.device)
        if is_varlen and nseq > 1
        else 1
    )
    metadata = torch.empty((metadata_capacity, 4), device=q.device, dtype=torch.int32)

    prepare = kda_ops.get_kda_prepare_ops_module(config).get_function(
        kda_ops.get_kda_prepare_ops_function_name(config)
    )
    prepare(
        q,
        k,
        g,
        beta,
        *workspace,
        metadata,
        cu_seqlens,
        A_log,
        dt_bias,
        float(scale),
        float(lower_bound),
        bool(use_qk_l2norm_in_kernel),
    )

    # The cadence is rendered into the kernel, so the pool only has to be big
    # enough (the kernel checks that) and to carry the activation dtype, which
    # is also what the contract hands back.  Its row count is cudnn's bound.
    if cadence:
        rows = kda_checkpoint_rows(
            batch=int(q.shape[0]),
            num_tokens=tokens,
            cu_seqlens=cu_seqlens if is_varlen else None,
            every_n_tokens=cadence,
        )
        if state_checkpoints is None:
            state_checkpoints = torch.empty(
                (rows, value_heads, HEAD_DIM, HEAD_DIM),
                device=q.device,
                dtype=q.dtype,
            )
        elif state_checkpoints.dtype != q.dtype:
            raise TypeError(
                "run_chunk_kda: state_checkpoints must carry the activation dtype "
                f"({q.dtype}), got {state_checkpoints.dtype}"
            )
        elif state_checkpoints.shape[0] < rows:
            raise ValueError(
                f"run_chunk_kda: state_checkpoints needs at least {rows} rows, "
                f"got {state_checkpoints.shape[0]}"
            )

    recurrence = kda_ops.get_kda_recurrence_ops_module(config).get_function(
        kda_ops.get_kda_recurrence_ops_function_name(config)
    )
    recurrence(
        v,
        output,
        *workspace,
        state_indices,
        initial_state,
        state_out,
        cu_seqlens,
        state_checkpoints if cadence else None,
    )

    if squeeze_varlen:
        output = output.squeeze(0)
    if cadence:
        return output, (final_state if output_final_state else None), state_checkpoints
    if output_final_state:
        return output, final_state
    return output


def _physical_num_mps(device) -> int:
    """The device's physical MUSA MP count, which sizes the metadata region."""
    index = device.index if getattr(device, "index", None) is not None else torch.musa.current_device()
    return torch.musa.get_device_properties(index).multi_processor_count
