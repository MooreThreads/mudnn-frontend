"""Normalize one linear-attention call into the :class:`Facts` record engines judge."""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch

from ..datatypes import _torch_to_cudnn_data_type

#: Kernels walk whole chunks: a sequence's token count is charged to the next
#: multiple of this.
CHUNK = 64


@lru_cache(maxsize=None)
def _musa_arch(device):
    """Hardware capability is immutable for a tensor's indexed device."""
    return tuple(torch.musa.get_device_capability(device))


@dataclass(frozen=True)
class Facts:
    """Facts describing one call. When ``invalid`` is set the call is malformed
    and the other fields must not be trusted."""

    op: str
    invalid: Optional[str]
    device_type: str
    total_tokens: int
    h_q: int
    h_k: int
    h_v: int
    h_o: int
    d_qk: int
    d_v: int
    seqs: Tuple[int, ...]
    uniform_len: Optional[int]
    io_dtype: object
    uniform_io: bool
    g_dtype: object
    beta_dtype: object
    state_dtype: object
    cu_dtype: object
    has_initial_state: bool
    wants_final_state: bool
    #: Cadence of the forward's state series (cuDNN's
    #: ``checkpoint_every_n_tokens``); the backward reuses a non-zero series.
    checkpoint_every_n_tokens: int = 0
    #: Architecture and index belong to the input tensor, not current_device().
    device_index: Optional[int] = None
    arch: Optional[Tuple[int, int]] = None

    @property
    def is_bwd(self) -> bool:
        return self.op.endswith("_BWD")

    @property
    def num_sequences(self) -> int:
        return len(self.seqs)

    @property
    def padded_tokens(self) -> int:
        """Tokens the kernels actually walk, each sequence charged to a chunk."""
        return sum(-(-length // CHUNK) * CHUNK for length in self.seqs)


#: ``id(cu_seqlens) -> (weakref, version, lengths)``. Keyed by identity because
#: a tensor's ``==`` is elementwise.
_LENGTHS_CACHE: Dict[int, tuple] = {}


def _split(cu_seqlens: torch.Tensor) -> Tuple[int, ...]:
    bounds = cu_seqlens.detach().to("cpu", torch.int64).tolist()
    return tuple(b - a for a, b in zip(bounds, bounds[1:]))


def sequence_lengths(cu_seqlens: torch.Tensor) -> Tuple[int, ...]:
    """Per-sequence token counts from a ``[N + 1]`` boundary vector.

    Reading a device ``cu_seqlens`` requires a blocking copy to host, so the
    result is cached against the tensor's identity and version.
    """
    if cu_seqlens.device.type == "cpu":
        return _split(cu_seqlens)
    key = id(cu_seqlens)
    entry = _LENGTHS_CACHE.get(key)
    if entry is not None and entry[0]() is cu_seqlens:
        if entry[1] == cu_seqlens._version:
            return entry[2]
    else:
        weakref.finalize(cu_seqlens, _LENGTHS_CACHE.pop, key, None)
    lengths = _split(cu_seqlens)
    _LENGTHS_CACHE[key] = (weakref.ref(cu_seqlens), cu_seqlens._version, lengths)
    return lengths


def analyze(
    op: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    checkpoint_every_n_tokens: int = 0,
) -> Facts:
    """Describe a THD linear-attention call as a :class:`Facts` record.

    ``q``/``k``/``v`` are ``[total_tokens, heads, dim]``, ``g``/``beta`` are
    ``[total_tokens, HO]`` with ``HO = max(h_q, h_v)``, and ``cu_seqlens`` is
    the ``[N + 1]`` boundary vector. Structural problems are reported in
    ``Facts.invalid`` instead of raising.
    """
    invalid = None
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        invalid = "q, k and v must be THD [total_tokens, heads, dim]"
    elif g.dim() != 2 or beta.dim() != 2:
        invalid = "g and beta must be [total_tokens, HO]"
    elif cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        invalid = "cu_seqlens must be a [N + 1] boundary vector"
    elif not (q.shape[0] == k.shape[0] == v.shape[0] == g.shape[0] == beta.shape[0]):
        invalid = "q, k, v, g and beta must agree on the token axis"
    elif checkpoint_every_n_tokens < 0:
        invalid = "checkpoint_every_n_tokens must be >= 0"
    elif q.shape[-1] != k.shape[-1]:
        invalid = "q and k must share their head dimension"

    h_q, h_k, h_v = (int(t.shape[1]) if t.dim() == 3 else 0 for t in (q, k, v))
    h_o = max(h_q, h_v)
    seqs = () if invalid else sequence_lengths(cu_seqlens)
    if invalid is None:
        if sum(seqs) != int(q.shape[0]):
            invalid = f"cu_seqlens covers {sum(seqs)} tokens, q has {int(q.shape[0])}"
        elif g.shape[1] != h_o or beta.shape[1] != h_o:
            invalid = f"g and beta must carry HO = max(h_q, h_v) = {h_o} heads"

    uniform = seqs[0] if seqs and len(set(seqs)) == 1 else None
    io = _torch_to_cudnn_data_type(q.dtype)
    return Facts(
        op=op,
        invalid=invalid,
        device_type=q.device.type,
        total_tokens=int(q.shape[0]),
        h_q=h_q,
        h_k=h_k,
        h_v=h_v,
        h_o=h_o,
        d_qk=int(q.shape[-1]),
        d_v=int(v.shape[-1]),
        seqs=seqs,
        uniform_len=uniform,
        io_dtype=io,
        uniform_io=q.dtype == k.dtype == v.dtype,
        g_dtype=_torch_to_cudnn_data_type(g.dtype),
        beta_dtype=_torch_to_cudnn_data_type(beta.dtype),
        state_dtype=None if initial_state is None else _torch_to_cudnn_data_type(initial_state.dtype),
        cu_dtype=_torch_to_cudnn_data_type(cu_seqlens.dtype),
        has_initial_state=initial_state is not None,
        wants_final_state=bool(output_final_state),
        checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
        device_index=q.device.index,
        arch=_musa_arch(q.device) if q.device.type == "musa" else None,
    )


__all__ = ["CHUNK", "Facts", "KdaFacts", "analyze", "analyze_kda",
           "sequence_lengths"]


# ---------------------------------------------------------------------------
# KDA (Kimi Delta Attention)
# ---------------------------------------------------------------------------
# KDA needs its own record rather than a flag on Facts above: its defining
# feature -- a per-key-channel gate -- changes the shape of `g`, and its
# recurrent state is V-major where GDN's is K-major.  Folding that into the
# shared record would make every GDN engine re-check layout it cannot vary.


@dataclass(frozen=True)
class KdaFacts:
    """One ``kimi_delta_attention`` call, as the engines see it.

    ``invalid`` reports a malformed call (one no engine could serve); support
    judgement stays per-engine, matched in ``check_support``.
    """

    op: str
    invalid: Optional[str]
    device_type: str
    arch: Optional[str]

    total_tokens: int
    h_q: int
    h_k: int
    h_v: int
    h_o: int
    d_qk: int
    d_v: int

    seqs: Tuple[int, ...]
    uniform_len: Optional[int]

    io_dtype: object
    uniform_io: bool
    g_dtype: object
    beta_dtype: object
    state_dtype: object
    gate_dtype: object
    cu_dtype: object

    #: ``g``'s channel dimension.  KDA's gate is per-key-channel, so a conforming
    #: call has ``gate_channels == d_qk``; 0 means g was not 3-D.
    gate_channels: int

    has_initial_state: bool
    wants_final_state: bool
    checkpoint_every_n_tokens: int
    safe_gate: bool
    gate_domain: str
    lower_bound: float
    use_qk_l2norm_in_kernel: bool
    use_beta_sigmoid_in_kernel: bool
    batch_invariant: bool
    allow_neg_eigval: bool
    overwrite_initial_state: bool
    uses_state_indices: bool

    @property
    def wants_checkpoints(self) -> bool:
        return self.checkpoint_every_n_tokens > 0

    @property
    def is_bwd(self) -> bool:
        return self.op.endswith("_BWD")

    @property
    def num_sequences(self) -> int:
        return len(self.seqs)

    @property
    def padded_tokens(self) -> int:
        return sum(-(-length // CHUNK) * CHUNK for length in self.seqs)


def _current_arch() -> Optional[str]:
    """The MUSA architecture to record on a KDA call.

    This is part of the plan cache key, so a process that re-resolves the
    architecture cannot reuse a plan built for another one.  The authoritative
    build target is :func:`kda_ops.resolve_kda_arch`, which the engines check
    against; this only has to follow it.

    An explicit ``MUDNN_JIT_MUSA_ARCH_LIST`` wins, because that is what drives
    ``--offload-arch``; a device that disagrees with it must not silently pick
    the device's architecture.
    """
    import os

    configured = os.environ.get("MUDNN_JIT_MUSA_ARCH_LIST", "").strip()
    if configured:
        return "mp" + configured.split()[0].replace(".", "")
    try:
        musa = getattr(torch, "musa", None)
        if musa is None or not musa.is_available():
            return None
        props = musa.get_device_properties(musa.current_device())
        return f"mp{int(props.major)}{int(props.minor)}"
    except Exception:
        return None


def analyze_kda(
    op: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    safe_gate: bool = False,
    gate_domain: str = "log",
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    lower_bound: float = -5.0,
    batch_invariant: bool = False,
    allow_neg_eigval: bool = False,
    overwrite_initial_state: bool = False,
    uses_state_indices: bool = False,
) -> KdaFacts:
    """Describe a THD KDA call as a :class:`KdaFacts` record.

    ``q``/``k``/``v`` are ``[total_tokens, heads, dim]``; ``g`` is
    ``[total_tokens, HO, K]`` (per-key-channel) and ``beta`` is
    ``[total_tokens, HO]``.  Structural problems land in ``invalid`` instead of
    raising, so every engine can report its own reason for declining.
    """
    invalid = None
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        invalid = "q, k and v must be THD [total_tokens, heads, dim]"
    elif g.dim() != 3:
        invalid = "g must be [total_tokens, HO, K] (per-key-channel)"
    elif beta.dim() != 2:
        invalid = "beta must be [total_tokens, HO]"
    elif cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        invalid = "cu_seqlens must be a [N + 1] boundary vector"
    elif not (q.shape[0] == k.shape[0] == v.shape[0] == g.shape[0] == beta.shape[0]):
        invalid = "q, k, v, g and beta must agree on the token axis"
    elif q.shape[-1] != k.shape[-1]:
        invalid = "q and k must share their head dimension"

    h_q, h_k, h_v = (int(t.shape[1]) if t.dim() == 3 else 0 for t in (q, k, v))
    h_o = max(h_q, h_v)
    gate_channels = int(g.shape[2]) if g.dim() == 3 else 0
    seqs = () if invalid else sequence_lengths(cu_seqlens)

    if invalid is None:
        if sum(seqs) != int(q.shape[0]):
            invalid = f"cu_seqlens covers {sum(seqs)} tokens, q has {int(q.shape[0])}"
        elif g.shape[1] != h_o or beta.shape[1] != h_o:
            invalid = f"g and beta must carry HO = max(h_q, h_v) = {h_o} heads"
        elif gate_channels != int(q.shape[-1]):
            invalid = (
                f"g must be per-key-channel with K = {int(q.shape[-1])} channels, "
                f"got {gate_channels}"
            )
        elif gate_domain not in ("log", "linear"):
            invalid = f"gate_domain must be 'log' or 'linear', got {gate_domain!r}"
        elif gate_domain == "linear" and safe_gate:
            invalid = "gate_domain='linear' cannot combine with safe_gate=True"
        elif not safe_gate and (a_log is not None or dt_bias is not None):
            invalid = "a_log/dt_bias require safe_gate=True"
        elif checkpoint_every_n_tokens < 0:
            invalid = "checkpoint_every_n_tokens must be >= 0"

    uniform = seqs[0] if seqs and len(set(seqs)) == 1 else None
    return KdaFacts(
        op=op,
        invalid=invalid,
        device_type=q.device.type,
        arch=_current_arch(),
        total_tokens=int(q.shape[0]),
        h_q=h_q,
        h_k=h_k,
        h_v=h_v,
        h_o=h_o,
        d_qk=int(q.shape[-1]),
        d_v=int(v.shape[-1]),
        seqs=seqs,
        uniform_len=uniform,
        io_dtype=_torch_to_cudnn_data_type(q.dtype),
        uniform_io=q.dtype == k.dtype == v.dtype,
        g_dtype=_torch_to_cudnn_data_type(g.dtype),
        beta_dtype=_torch_to_cudnn_data_type(beta.dtype),
        state_dtype=None if initial_state is None else _torch_to_cudnn_data_type(initial_state.dtype),
        gate_dtype=None,
        cu_dtype=_torch_to_cudnn_data_type(cu_seqlens.dtype),
        gate_channels=gate_channels,
        has_initial_state=initial_state is not None,
        wants_final_state=bool(output_final_state),
        checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
        safe_gate=bool(safe_gate),
        gate_domain=gate_domain,
        lower_bound=float(lower_bound),
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
        batch_invariant=bool(batch_invariant),
        allow_neg_eigval=bool(allow_neg_eigval),
        overwrite_initial_state=bool(overwrite_initial_state),
        uses_state_indices=bool(uses_state_indices),
    )
