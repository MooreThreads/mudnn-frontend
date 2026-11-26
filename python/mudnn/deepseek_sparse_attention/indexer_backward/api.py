"""cuDNN-aligned indexer-backward surface for the muDNN frontend.

Mirrors ``cudnn-frontend``'s ``indexer_backward/api.py`` public surface:

* ``IndexerBackward`` / ``indexer_backward_wrapper`` -- sparse top-K backward.
* ``DenseIndexerBackward`` / ``dense_indexer_backward_wrapper`` -- full-KV dense
  backward.

Both families select their implementation from the device capability inside
``compile()``; the choice never appears in a public name and there is no caller
flag for it. The entry points are always ``indexer_backward`` and
``dense_indexer_backward``.

Sparse (``IndexerBackward``)
---------------------------
* ``mp_32`` -- the TileLang kernel in :mod:`indexer_backward_mp32`. It runs
  upstream's three stages, consumes ``index_score`` as the predict operand and
  **overwrites ``attn_score``** with ``grad_signal``.
* ``mp_31`` -- the MP31 port in :mod:`indexer_backward_mp31`: same stages and
  contract (score-buffer semantics included), with ``block_I=64`` as the arch
  default -- the MP31 opt-in shared-memory ceiling does not fit the mp_32
  default of 128 at the training shapes.
* otherwise -- the pure-torch implementation in ``mudnn.DSA``. It recomputes the
  predict distribution from ``(index_q, index_k, weights, topk_indices)``, so
  ``index_score`` is advisory and neither score buffer is mutated.

That difference is observable: reusing an ``attn_score`` buffer across two
launches is correct only on the kernel path. ``topk_indices_global=True`` is
available on ``mp_32`` and rejected on the fallback, which gathers local ids
only.

Dense (``DenseIndexerBackward``)
-------------------------------
Implemented in TileLang MUSA in both variants: ``dense_indexer_backward.py`` is
the generic one and ``dense_indexer_backward_mp32.py`` the mp_32-tuned one.
``dIndexK`` is always an fp32 accumulation buffer at the kernel boundary; a
narrower output dtype is served by an internal scratch plus a cast.

Documented deviations from upstream
-----------------------------------
* ``grad_loss`` defaults to ``None`` (meaning the scalar ``1.0``) so existing
  muDNN call sites keep working; upstream requires the single-element fp32
  device tensor positionally.
* ``execute()`` returns ``None``, but ``compile()`` returns ``self`` so
  ``compile().execute(...)`` chaining works.
* ``backend="v2"`` is device-neutral here (upstream suffixes it with an
  architecture) and is request-or-fail, since no kernel implements it.
* ``check_support()`` makes no compute-capability claim: public API alignment
  does not assert hardware support. The narrow envelopes (bf16 operands,
  ``heads % 8 == 0``, ``block_I | topk``, ``topk <= 2048``) belong to the kernels
  and are reported by them, not by the surface.
"""
from __future__ import annotations

import warnings
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - `import mudnn` must work without torch
    torch = None

from mudnn._torch_stream import stream_context
from mudnn.api_base import APIBase, TupleDict

INDEXER_BACKWARD_BACKENDS = ("default", "v2")

__all__ = [
    "DenseIndexerBackward",
    "IndexerBackward",
    "dense_indexer_backward_wrapper",
    "indexer_backward_wrapper",
]

def _require_torch():
    if torch is None:
        raise NotImplementedError(
            "mudnn DSA indexer backward requires torch"
        )
    return torch

# Upstream spells the opt-in backend after its target architecture
# (``"sm100_v2"``). This frontend's canonical value is the *device-neutral*
# ``"v2"`` -- no architecture name may reach the public surface -- but the
# upstream spelling is accepted on input and normalised, so code written against
# cuDNN reaches the "recognised but unimplemented" path instead of failing with a
# ValueError about the enum it was told to use.
_BACKEND_ALIASES = {"sm100_v2": "v2"}

def _validate_backend(backend: str) -> str:
    """Return the canonical spelling of *backend*, or raise ``ValueError``."""
    canonical = _BACKEND_ALIASES.get(backend, backend)
    if canonical not in INDEXER_BACKWARD_BACKENDS:
        raise ValueError(
            f"backend must be one of {INDEXER_BACKWARD_BACKENDS} "
            f"(or the upstream spelling {tuple(_BACKEND_ALIASES)}), "
            f"got {backend!r}"
        )
    return canonical

def _reject_v2_backend(backend: str) -> None:
    if backend == "v2":
        raise NotImplementedError(
            "IndexerBackward backend='v2' (upstream spelling "
            "'sm100_v2') is not implemented in muDNN frontend: the v2 "
            "(two-term bf16-expansion) GEMM stage has no muDNN/MUSA kernel "
            "yet. Use backend='default'."
        )

def _reject_global_topk_indices(topk_indices_global: bool) -> None:
    if topk_indices_global:
        raise NotImplementedError(
            "IndexerBackward topk_indices_global=True is not implemented in "
            "muDNN frontend: the torch fallback gathers index_k with "
            "per-batch local ids only. Pass local ids (topk_indices_global="
            "False) or convert them beforehand."
        )

def _validate_float_tensor(name: str, tensor):
    _require_torch()
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
        )
    if not tensor.is_floating_point():
        raise ValueError(
            f"{name} must be a floating-point tensor, got {tensor.dtype}"
        )
    return tensor

def _validate_fp32_tensor(name: str, tensor):
    _require_torch()
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
        )
    if tensor.dtype != torch.float32:
        raise ValueError(
            f"{name} must be float32 (the fallback consumes raw fp32 "
            f"scores), got {tensor.dtype}"
        )
    return tensor

def _validate_int32_tensor(name: str, tensor):
    _require_torch()
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
        )
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    return tensor

def _as_contiguous(tensor):
    """Return a layout the kernel can read, and a way to restore the caller's.

    The TileLang kernels index with ``.view(...)``, which needs a compact
    row-major layout, so a strided tensor cannot be passed through. Upstream
    handles this by copying at the wrapper boundary; this does the same.

    Returns ``(contiguous, restore_or_None)``. ``restore`` is a callable that
    writes the contiguous copy back into the caller's tensor, and is `None` when
    no copy was made. Call it for buffers the kernel mutates (``index_score`` /
    ``attn_score``) and for outputs; skip it for pure inputs.
    """
    if tensor is None or tensor.is_contiguous():
        return tensor, None

    contiguous = tensor.contiguous()

    def restore():
        tensor.copy_(contiguous)

    return contiguous, restore

def _grad_loss_operand(grad_loss, device):
    """Validate ``grad_loss`` and hand it to the kernel unchanged.

    A **tensor is returned as-is**, never converted to a Python float. The
    conversion used to happen here and it was wrong twice over: ``float(t)`` is a
    device-to-host copy, so every ``execute`` synchronised the device, and under
    CUDA-graph capture that copy is illegal -- and even where capture succeeded,
    replay would only reuse the value captured the first time instead of letting
    the caller change ``grad_loss`` between replays. Upstream validates the
    tensor and passes it through for exactly this reason.

    ``None`` means "no per-call scaling" (the kernel substitutes 1.0), and a
    plain Python number is still accepted for convenience; that path does have to
    materialise a device tensor, so pass a tensor when capturing a graph.
    """
    if grad_loss is None:
        return None
    if torch is not None and torch.is_tensor(grad_loss):
        if grad_loss.numel() != 1:
            raise ValueError(
                f"grad_loss must be a single-element tensor, got numel="
                f"{grad_loss.numel()}"
            )
        if grad_loss.dtype != torch.float32:
            raise ValueError(
                f"grad_loss must be float32, got {grad_loss.dtype}"
            )
        if device is not None and grad_loss.device != device:
            raise ValueError(
                f"grad_loss must live on {device}, got {grad_loss.device}"
            )
        return grad_loss
    if isinstance(grad_loss, bool) or not isinstance(grad_loss, (int, float)):
        raise TypeError(
            "grad_loss must be None, a single-element float32 tensor, or a "
            f"Python number, got {type(grad_loss).__name__}"
        )
    return grad_loss

def _dtype_is_floating(dtype) -> bool:
    return bool(getattr(dtype, "is_floating_point", False))

_TILELANG_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
}

def _tilelang_dtype(dtype) -> str:
    try:
        return _TILELANG_DTYPES[dtype]
    except KeyError as exc:
        raise ValueError(
            "DenseIndexerBackward requires a 16-bit tensor-core operand dtype "
            f"(bfloat16 or float16) for the TileLang kernel, got {dtype}"
        ) from exc

_MP32_CAPABILITY = 32
_MP31_CAPABILITY = 31


def _default_block_I(device) -> int:
    """Arch default for ``block_I`` when the caller leaves it unset.

    MP31 (``mp_31``) has a ~192 KB opt-in shared-memory ceiling, which rules
    out the mp_32 default of 128 at the training shapes (heads=64, dim=128
    needs 206336 B); 64 fits (123648 B). Explicit values are respected as-is.
    """
    return 64 if _device_capability(device) == _MP31_CAPABILITY else 128


def _musa_device_index(device) -> Optional[int]:
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        text = device.strip()
        if text == "musa":
            return 0
        if text.startswith("musa:"):
            try:
                return int(text.split(":", 1)[1])
            except ValueError:
                return None
        return None
    if getattr(device, "type", None) != "musa":
        return None
    index = getattr(device, "index", None)
    if index is None:
        return 0
    try:
        return int(index)
    except (TypeError, ValueError):
        return None

def _device_capability(device) -> Optional[int]:
    """Compute capability as ``major * 10 + minor``, or ``None``.

    ``None`` means "cannot tell", and it is load-bearing: the callers fall back to
    the architecture-generic implementation. Returning it silently would let a
    transient driver failure turn an ``mp_32`` device into a slower generic one
    with no explanation, so a *query that fails* is warned about. The cases that
    legitimately have no capability -- no MUSA runtime, or a non-MUSA device --
    stay silent, because those are the normal torch-only paths.
    """
    _require_torch()
    get_props = getattr(
        getattr(torch, "musa", None), "get_device_properties", None
    )
    if get_props is None:
        return None
    index = _musa_device_index(device)
    if index is None:
        # Either not a MUSA device at all, or an index we cannot resolve; both are
        # expected on the torch-only paths.
        return None
    try:
        props = get_props(index)
        major = int(getattr(props, "major", 0))
        minor = int(getattr(props, "minor", 0))
    except Exception as exc:
        warnings.warn(
            "could not query the compute capability of MUSA device "
            f"{index} ({type(exc).__name__}: {exc}); falling back to the "
            "architecture-generic implementation, which is slower than the "
            "device-specific one on mp_32. Pass an explicit capability check "
            "or fix the driver if this is unexpected.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if major <= 0:
        warnings.warn(
            f"MUSA device {index} reported a non-positive compute capability "
            f"(major={major}, minor={minor}); falling back to the "
            "architecture-generic implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return major * 10 + minor

def _tilelang_dense_envelope(heads, dim, block_I, max_seqlen_k, ratio):
    """Why a shape cannot use the TileLang dense kernel, or ``None`` if it can.

    The kernels have hard structural requirements that a ``ValueError`` from the
    factory reports as a *failure*. For a caller that only wants a working
    backward pass they are a reason to pick a different implementation, not to
    abort: ``compile()`` therefore asks this first and falls back to the torch
    implementation when it returns a reason.

    Kept in one place so the dispatch decision and the error message cannot
    drift. These are the conditions the SQMMA tile layout imposes:

    * ``heads % 16 == 0`` and ``dim % 8 == 0`` -- the SQMMA operand tile shapes.
    * ``block_I`` divides ``max_seqlen_k`` -- the K tiles are issued as whole
      unpredicated TMA loads, so a partial tail tile cannot be expressed.
    * ``ratio >= 1``.

    Legacy fixtures use ``heads=2, dim=2``; those were the shapes that failed
    with ``requires SQMMA-friendly tiles`` on a MUSA lane while passing on a
    CPU-only one, which is why this is decided before the build rather than
    after.
    """
    if int(heads) % 16 != 0:
        return f"heads={heads} is not a multiple of 16"
    if int(dim) % 8 != 0:
        return f"dim={dim} is not a multiple of 8"
    if int(block_I) <= 0 or int(max_seqlen_k) % int(block_I) != 0:
        return (
            f"block_I={block_I} does not divide max_seqlen_k={max_seqlen_k}"
        )
    if int(ratio) < 1:
        return f"ratio={ratio} is not >= 1"
    return None

def _max_from_cu_seqlens(cu_seqlens) -> int:
    """Largest per-batch length encoded by a ``(B + 1,)`` prefix-sum tensor."""
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must hold at least one sequence")
    deltas = cu_seqlens[1:] - cu_seqlens[:-1]
    return int(deltas.max().item())

def _dense_kernel_capability(device, is_thd, has_q_causal_offsets):
    """``(supports_thd, supports_offsets)`` for the implementation on *device*.

    The two dense factories are not interchangeable: only the mp_32 variant
    implements THD and ``q_causal_offsets``, and the generic one rejects both.
    ``compile()`` therefore asks what the selected implementation can express
    instead of assuming the two are equivalent.
    """
    is_mp32 = _device_capability(device) == _MP32_CAPABILITY
    # The generic kernel is BSHD-only and has no per-row causal offset.
    return is_mp32, is_mp32

def _tilelang_sparse_envelope(heads, topk, block_I):
    """Why a shape cannot use the TileLang sparse kernel, or ``None`` if it can.

    Same role as :func:`_tilelang_dense_envelope`. The sparse kernel additionally
    needs ``heads % 8 == 0`` (its M extent is ``heads``) and ``block_I | topk``.
    """
    if int(heads) % 8 != 0:
        return f"heads={heads} is not a multiple of 8"
    if int(block_I) <= 0 or int(topk) % int(block_I) != 0:
        return f"block_I={block_I} does not divide topk={topk}"
    return None

class _DenseIndexerBackwardFallback:
    """Torch implementation used when the TileLang envelope does not apply.

    Mirrors :class:`_SparseIndexerBackwardFallback`, but for the dense family.
    Without it a shape outside the SQMMA envelope (``heads`` not a multiple of
    16, ``dim`` not a multiple of 8, ``block_I`` not dividing ``max_seqlen_k``)
    failed at plan-build time on a MUSA lane while working on a CPU-only one --
    the same input producing two different verdicts depending on hardware.
    """

    __slots__ = ("sm_scale", "ratio", "block_I", "reason")

    def __init__(self, sm_scale, ratio, block_I, reason):
        self.sm_scale = float(sm_scale)
        self.ratio = int(ratio)
        self.block_I = int(block_I)
        self.reason = str(reason)

    def __call__(
        self,
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK,
        AttnScore,
        AttnL1Norm,
        IdxScoreRaw,
        IdxLSE,
        GradLoss,
        grad_scale,
        CuSeqlensQ=None,
        CuSeqlensK=None,
        QCausalOffsets=None,
        current_stream=None,
        sm_scale=None,
    ):
        from .._torch_fallback import _dense_indexer_backward_wrapper_naive as _wrapper

        # The torch implementation divides ``loss_coeff`` by the normalization
        # itself, while ``execute`` hands us the already-divided ``grad_scale``.
        # Undo the division so the two agree instead of silently scaling by
        # ``batch * S_q``.
        if IndexQ.ndim == 4:
            _norm = int(IndexQ.shape[0]) * int(IndexQ.shape[1])
        else:
            _norm = int(IndexQ.shape[0])
        _loss_coeff = float(grad_scale) * max(_norm, 1)

        # The torch path is not graph-capturable anyway, so resolving the scalar
        # here is fine; ``None`` means "no per-call scaling".
        _require_torch()
        if GradLoss is None:
            _grad_loss = 1.0
        elif torch.is_tensor(GradLoss):
            _grad_loss = float(GradLoss)
        else:
            _grad_loss = float(GradLoss)

        with stream_context(current_stream, IndexQ.device):
            outputs = _wrapper(
                IndexQ,
                Weights,
                IndexK,
                AttnScore,
                AttnL1Norm,
                IdxScoreRaw,
                IdxLSE,
                sm_scale=self.sm_scale if sm_scale is None else float(sm_scale),
                loss_coeff=_loss_coeff,
                grad_loss=_grad_loss,
                block_I=self.block_I,
                ratio=self.ratio,
                cu_seqlens_q=CuSeqlensQ,
                cu_seqlens_k=CuSeqlensK,
            )
            dIndexQ.copy_(outputs["d_index_q"].to(dIndexQ.dtype))
            dWeights.copy_(outputs["d_weights"].to(dWeights.dtype))
            dIndexK.copy_(outputs["d_index_k"].to(dIndexK.dtype))

def _dense_indexer_backward_factory(device):
    if _device_capability(device) == _MP32_CAPABILITY:
        from .dense_indexer_backward_mp32 import (
            dense_indexer_backward_mp32 as factory,
        )
    else:
        from .dense_indexer_backward import dense_indexer_backward as factory
    return factory

def _sparse_indexer_backward_factory(device):
    capability = _device_capability(device)
    if capability == _MP32_CAPABILITY:
        from .indexer_backward_mp32 import indexer_backward_mp32 as factory

        return factory
    if capability == _MP31_CAPABILITY:
        from .indexer_backward_mp31 import indexer_backward_mp31 as factory

        return factory
    return None

class _SparseIndexerBackwardFallback:

    __slots__ = ("sm_scale", "block_I", "topk_indices_global")

    def __init__(self, sm_scale, block_I, topk_indices_global):
        self.sm_scale = float(sm_scale)
        self.block_I = int(block_I)
        self.topk_indices_global = bool(topk_indices_global)

    def __call__(
        self,
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK,
        AttnScore,
        IndexScore,
        TopkIndices,
        GradLoss,
        loss_coeff,
        current_stream=None,
    ):
        from .._torch_fallback import _indexer_backward_wrapper_naive as _wrapper

        # ``None`` is this API's spelling of "no per-call scaling"; the torch
        # implementation multiplies by ``float(grad_loss)`` and would raise on it.
        # Normalise here rather than in the terminal wrapper, which has to stay
        # usable on its own with its documented ``grad_loss=1.0`` default.
        _require_torch()
        if GradLoss is None:
            _grad_loss = 1.0
        elif torch.is_tensor(GradLoss):
            _grad_loss = float(GradLoss)
        else:
            _grad_loss = float(GradLoss)

        # The copies into the caller's buffers run on ``current_stream`` too, so
        # that a non-default stream sees the fallback's results in order with
        # whatever the caller does next.
        with stream_context(current_stream, IndexQ.device):
            outputs = _wrapper(
                IndexQ,
                Weights,
                IndexK,
                AttnScore,
                IndexScore,
                TopkIndices,
                sm_scale=self.sm_scale,
                loss_coeff=loss_coeff,
                grad_loss=_grad_loss,
                block_I=self.block_I,
                stream=current_stream,
            )
            dIndexQ.copy_(outputs["d_index_q"])
            dWeights.copy_(outputs["d_weights"])
            dIndexK.copy_(outputs["d_index_k"])

class _Mp32SparseIndexerBackward:

    __slots__ = ("_kernel",)

    def __init__(self, kernel):
        self._kernel = kernel

    def __call__(
        self,
        IndexQ,
        Weights,
        IndexK,
        dIndexQ,
        dWeights,
        dIndexK,
        AttnScore,
        IndexScore,
        TopkIndices,
        GradLoss,
        loss_coeff,
        current_stream=None,
    ):
        batch = int(IndexQ.shape[0])
        seqlen = int(IndexQ.shape[1])
        grad_scale = float(loss_coeff) / max(batch * seqlen, 1)
        self._kernel(
            IndexQ,
            Weights,
            IndexK,
            dIndexQ,
            dWeights,
            dIndexK,
            AttnScore,
            IndexScore,
            TopkIndices,
            GradLoss,
            grad_scale,
            current_stream,
        )

class IndexerBackward(APIBase):
    """Sparse top-K indexer backward.

    Consumes the forward-computed target distribution ``attn_score``, the predict
    distribution ``index_score`` and the sparse top-K indices, and produces
    ``d_index_q``, ``d_weights`` and ``d_index_k``.

    The constructor takes sample tensors that fix the plan (shapes, dtypes,
    strides); ``execute`` may then be called repeatedly with tensors of the same
    signature. ``compile()`` selects the implementation from the device
    capability -- see the module docstring for the one observable difference
    between them (whether ``attn_score`` is overwritten in place).

    ``sm_scale``, ``block_I`` and ``topk_indices_global`` are plan parameters,
    not per-call ones. On ``mp_32`` the kernel additionally requires bf16
    operands and ``block_I | topk``, ``heads % 8 == 0``, ``topk <= 2048``; those
    are reported at build time by name.
    """

    def __init__(
        self,
        sample_index_q: "torch.Tensor",
        sample_weights: "torch.Tensor",
        sample_index_k: "torch.Tensor",
        sample_d_index_q: "torch.Tensor",
        sample_d_weights: "torch.Tensor",
        sample_d_index_k: "torch.Tensor",
        sample_attn_score: "torch.Tensor",
        sample_index_score: "torch.Tensor",
        sample_topk_indices: "torch.Tensor",
        sm_scale: float = 1.0,
        block_I: Optional[int] = None,
        topk_indices_global: bool = False,
        backend: str = "default",
    ):
        super().__init__()
        backend = _validate_backend(backend)

        self.iq_desc = self._make_tensor_desc(
            sample_index_q, name="sample_index_q"
        )
        self.w_desc = self._make_tensor_desc(
            sample_weights, name="sample_weights"
        )
        self.ik_desc = self._make_tensor_desc(
            sample_index_k, name="sample_index_k"
        )
        self.diq_desc = self._make_tensor_desc(
            sample_d_index_q, name="sample_d_index_q"
        )
        self.dw_desc = self._make_tensor_desc(
            sample_d_weights, name="sample_d_weights"
        )
        self.dik_desc = self._make_tensor_desc(
            sample_d_index_k, name="sample_d_index_k"
        )
        self.attn_desc = self._make_tensor_desc(
            sample_attn_score, name="sample_attn_score"
        )
        self.idx_score_desc = self._make_tensor_desc(
            sample_index_score, name="sample_index_score"
        )
        self.topk_desc = self._make_tensor_desc(
            sample_topk_indices, name="sample_topk_indices"
        )

        if sample_index_q.ndim != 4:
            raise ValueError(
                "IndexerBackward sample_index_q must be 4D (B, S_q, H, D), "
                f"got {sample_index_q.ndim}D shape "
                f"{tuple(sample_index_q.shape)}"
            )
        if sample_index_k.ndim != 3:
            raise ValueError(
                "IndexerBackward sample_index_k must be 3D (B, S_k, D), "
                f"got {sample_index_k.ndim}D shape "
                f"{tuple(sample_index_k.shape)}"
            )
        if sample_topk_indices.ndim != 3:
            raise ValueError(
                "IndexerBackward sample_topk_indices must be 3D "
                f"(B, S_q, topk), got {sample_topk_indices.ndim}D shape "
                f"{tuple(sample_topk_indices.shape)}"
            )

        batch, seqlen, heads, head_dim = sample_index_q.shape
        self.batch = int(batch)
        self.seqlen = int(seqlen)
        self.seqlen_k = int(sample_index_k.shape[1])
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.topk = int(sample_topk_indices.shape[-1])
        self.sm_scale = float(sm_scale)
        if block_I is None:
            block_I = _default_block_I(sample_index_q.device)
        self.block_I = int(block_I)
        self.topk_indices_global = bool(topk_indices_global)
        self.backend = backend
        # Holds the mp_32 *or* mp_31 factory (see _sparse_indexer_backward_factory);
        # None means the torch fallback.
        self._mp32_factory = _sparse_indexer_backward_factory(
            sample_index_q.device
        )
        self._plan_layout_validated = False

    def _validate_plan_shapes_and_layout(self) -> None:
        batch, seqlen, heads, head_dim = (
            self.batch,
            self.seqlen,
            self.heads,
            self.head_dim,
        )
        seqlen_k, topk = self.seqlen_k, self.topk
        q_shape = (batch, seqlen, heads, head_dim)
        w_shape = (batch, seqlen, heads)
        k_shape = (batch, seqlen_k, head_dim)
        score_shape = (batch, seqlen, topk)

        checks = (
            (self.iq_desc, "index_q", q_shape),
            (self.w_desc, "weights", w_shape),
            (self.ik_desc, "index_k", k_shape),
            (self.diq_desc, "d_index_q", q_shape),
            (self.dw_desc, "d_weights", w_shape),
            (self.dik_desc, "d_index_k", k_shape),
            (self.attn_desc, "attn_score", score_shape),
            (self.idx_score_desc, "index_score", score_shape),
            (self.topk_desc, "topk_indices", score_shape),
        )
        for desc, name, expected in checks:
            if desc is None:
                continue
            actual = tuple(desc.shape)
            if actual != tuple(expected):
                raise ValueError(
                    f"IndexerBackward {name} shape mismatch: expected "
                    f"{tuple(expected)}, got {actual}"
                )
        for desc, name, _ in checks:
            if desc is None:
                continue
            if not desc.is_contiguous():
                raise ValueError(
                    f"IndexerBackward requires contiguous (compact "
                    f"row-major) tensors; {name} has shape {tuple(desc.shape)} "
                    f"with non-compact stride {tuple(desc.stride)}."
                )
        self._plan_layout_validated = True

    def check_support(self) -> bool:
        _reject_v2_backend(self.backend)
        if self._mp32_factory is None:
            _reject_global_topk_indices(self.topk_indices_global)
        self._value_error_if(
            self.block_I <= 0, f"block_I must be positive, got {self.block_I}"
        )
        for desc, name in (
            (self.iq_desc, "index_q"),
            (self.w_desc, "weights"),
            (self.ik_desc, "index_k"),
        ):
            if desc is None:
                continue
            if not _dtype_is_floating(desc.dtype):
                raise ValueError(
                    f"IndexerBackward {name} must be a floating-point tensor, "
                    f"got {desc.dtype}"
                )
        for desc, name in (
            (self.attn_desc, "attn_score"),
            (self.idx_score_desc, "index_score"),
        ):
            if desc is None:
                continue
            if desc.dtype != torch.float32:
                raise ValueError(
                    f"IndexerBackward {name} must be float32, got {desc.dtype}"
                )
        if self.topk_desc is not None and self.topk_desc.dtype != torch.int32:
            raise ValueError(
                "IndexerBackward topk_indices must be int32, got "
                f"{self.topk_desc.dtype}"
            )
        for desc, name in (
            (self.diq_desc, "d_index_q"),
            (self.dw_desc, "d_weights"),
            (self.dik_desc, "d_index_k"),
        ):
            if desc is None:
                continue
            if not _dtype_is_floating(desc.dtype):
                raise ValueError(
                    f"IndexerBackward {name} must be a floating-point tensor, "
                    f"got {desc.dtype}"
                )
        if self._mp32_factory is not None:
            self._validate_mp32_operand_dtypes()
        self._validate_plan_shapes_and_layout()
        self._is_supported = True
        return True

    def _validate_mp32_operand_dtypes(self) -> None:
        for desc, name in (
            (self.iq_desc, "index_q"),
            (self.w_desc, "weights"),
            (self.ik_desc, "index_k"),
        ):
            if desc is None:
                continue
            if desc.dtype != torch.bfloat16:
                raise ValueError(
                    f"IndexerBackward {name} must be bfloat16 on the TileLang "
                    f"kernel path (mp_31/mp_32): the SQMMA kernel's operand "
                    f"type is fixed at build time, so a "
                    f"{desc.dtype} tensor would be reinterpreted rather than "
                    f"converted. Cast the operand, or use a non-MUSA device "
                    f"where the torch fallback accepts other dtypes."
                )
        for desc, name in (
            (self.diq_desc, "d_index_q"),
            (self.dw_desc, "d_weights"),
        ):
            if desc is None:
                continue
            if desc.dtype != torch.bfloat16:
                raise ValueError(
                    f"IndexerBackward {name} must be bfloat16 on the TileLang "
                    f"kernel path (the kernel stores the gradient with the "
                    f"operand dtype), got {desc.dtype}"
                )
    def compile(self):
        """Build the plan and return ``self`` for chaining.

        The counterpart of upstream's architecture-specific factory selection
        (``indexer_backward`` / ``indexer_backward_v2``):

        * ``mp_32`` -- build the TileLang sparse kernel. Every argument that
          varies its codegen (``sm_scale`` is runtime; ``block_I``,
          ``topk`` and the id convention are compile keys) is passed here, and
          the shapes go in too, because they are runtime extents of the
          prim_func rather than part of the key.
        * otherwise -- bind the ``mudnn.DSA`` torch fallback.

        The two objects present the same positional contract, so ``execute``
        does not branch.
        """
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            # A shape outside the kernel's structural envelope is a reason to use
            # the torch implementation, not to raise: the same inputs must behave
            # the same way on every device. See ``_tilelang_sparse_envelope``.
            _outside = (
                None if self._mp32_factory is None
                else _tilelang_sparse_envelope(
                    self.heads, self.topk, self.block_I)
            )
            if self._mp32_factory is not None and _outside is None:
                self._compiled_kernel = _Mp32SparseIndexerBackward(
                    self._mp32_factory(
                        batch=self.batch,
                        seqlen=self.seqlen,
                        seqlen_k=self.seqlen_k,
                        heads=self.heads,
                        dim=self.head_dim,
                        topk=self.topk,
                        sm_scale=self.sm_scale,
                        block_I=self.block_I,
                        topk_indices_global=self.topk_indices_global,
                    )
                )
            else:
                self._compiled_kernel = _SparseIndexerBackwardFallback(
                    sm_scale=self.sm_scale,
                    block_I=self.block_I,
                    topk_indices_global=self.topk_indices_global,
                )
        return self

    def _check_execute_signature(self, *entries) -> None:
        for tensor, desc, name in entries:
            if desc is None:
                continue
            if tensor.dtype != desc.dtype:
                raise ValueError(
                    f"{name} dtype mismatch: this plan was compiled for "
                    f"{desc.dtype}, got {tensor.dtype}. Build a new plan (or "
                    "use indexer_backward_wrapper) for a different signature."
                )
            if tuple(tensor.shape) != tuple(desc.shape):
                raise ValueError(
                    f"{name} shape mismatch: this plan was compiled for "
                    f"{tuple(desc.shape)}, got {tuple(tensor.shape)}."
                )
            if desc.is_contiguous() and not tensor.is_contiguous():
                raise ValueError(
                    f"{name} stride/layout mismatch: this plan was compiled "
                    f"for stride {tuple(desc.stride)}, got "
                    f"{tuple(tensor.stride())}."
                )

    def execute(
        self,
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k,
        attn_score,
        index_score,
        topk_indices,
        grad_loss=None,
        loss_coeff: float = 1.0,
        current_stream=None,
    ) -> None:
        if self._compiled_kernel is None:
            self.compile()
        if not self._plan_layout_validated:
            self._validate_plan_shapes_and_layout()
        self._check_execute_signature(
            (index_q, self.iq_desc, "index_q"),
            (weights, self.w_desc, "weights"),
            (index_k, self.ik_desc, "index_k"),
            (d_index_q, self.diq_desc, "d_index_q"),
            (d_weights, self.dw_desc, "d_weights"),
            (d_index_k, self.dik_desc, "d_index_k"),
            (attn_score, self.attn_desc, "attn_score"),
            (index_score, self.idx_score_desc, "index_score"),
            (topk_indices, self.topk_desc, "topk_indices"),
        )
        grad_loss_operand = _grad_loss_operand(grad_loss, index_q.device)
        self._compiled_kernel(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn_score,
            index_score,
            topk_indices,
            grad_loss_operand,
            loss_coeff,
            current_stream,
        )

class DenseIndexerBackward(APIBase):
    """Dense full-KV indexer backward.

    Full-KV counterpart to :class:`IndexerBackward`. It consumes the raw dense
    score tensors and denominators produced by the dense indexer / dense
    attention score recompute wrappers, normalizes them inside the score-grad
    stage, and produces ``d_index_q``, ``d_weights`` and ``d_index_k``.

    Supports BSHD ``(B, S_q, ...)`` and -- where the selected implementation can
    express it -- THD ``(T_q, ...)`` packed tensors with ``cu_seqlens_q`` /
    ``cu_seqlens_k``.

    What is available depends on **where the plan runs**, and ``compile()``
    picks the implementation rather than asking the caller to:

    ==================  ==================  =====================
    feature             kernel capability   torch fallback
    ==================  ==================  =====================
    BSHD                yes                 yes
    THD / packed        yes (mp_32 only)    yes
    ``q_causal_offsets``  yes (mp_32 only)  **no**
    ==================  ==================  =====================

    A shape outside the kernel's tile envelope falls back to torch (see
    :func:`_tilelang_dense_envelope`). THD does the same off ``mp_32``, because
    only the mp_32 variant implements it. ``q_causal_offsets`` has no torch
    implementation, so requesting it without a kernel that supports it raises
    ``NotImplementedError`` at ``compile()`` time rather than producing a
    silently different result.

    ``q_causal_offsets``, where supported: build the plan with
    ``has_q_causal_offsets=True`` and pass a ``(batch,)`` int32 tensor to
    ``execute``. The offset is added to the query row index *before* the ratio
    division -- ``clamp((offset + q + 1) // ratio, 0, seqlen_k)`` -- matching
    cudnn-frontend, so under ``ratio > 1`` an offset of one column is worth
    ``offset // ratio`` columns of bound, not ``offset``. A negative offset is
    allowed and clamps the bound at zero. The plan flag and the runtime tensor
    must agree: passing offsets to a plan built without the flag is an error,
    and so is omitting them from a plan built with it.

    ``max_seqlen_q`` / ``max_seqlen_k`` are required for THD and are *derived*
    from the tensor shapes for BSHD -- a value passed for BSHD is checked
    against those shapes instead of being silently dropped.

    The fallback makes no ``heads >= 64`` / compute-capability claim.
    """

    def __init__(
        self,
        sample_index_q: "torch.Tensor",
        sample_weights: "torch.Tensor",
        sample_index_k: "torch.Tensor",
        sample_d_index_q: "torch.Tensor",
        sample_d_weights: "torch.Tensor",
        sample_d_index_k: "torch.Tensor",
        sample_attn_score: "torch.Tensor",
        sample_attn_l1norm: "torch.Tensor",
        sample_index_score: "torch.Tensor",
        sample_index_lse: "torch.Tensor",
        sm_scale: float = 1.0,
        # ``block_I=64``, not upstream's 128. The generic kernel is capped at
        # 192 KB of dynamic shared memory and 128 needs 207,360 B at
        # ``heads=64, dim=128, num_stages=1, q_rows=2`` (the defaults this plan
        # passes), so 128 made the *default* configuration fail to build on any
        # non-mp_32 device. 64 needs 140,800 B and both factories already default
        # to 64 anyway; this just stops the API layer from overriding them with
        # a value that only fits the mp_32 budget (256 KB).
        block_I: int = 64,
        ratio: int = 1,
        is_thd: bool = False,
        batch: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        has_q_causal_offsets: bool = False,
    ):
        super().__init__()
        self.iq_desc = self._make_tensor_desc(
            sample_index_q, name="sample_index_q"
        )
        self.w_desc = self._make_tensor_desc(
            sample_weights, name="sample_weights"
        )
        self.ik_desc = self._make_tensor_desc(
            sample_index_k, name="sample_index_k"
        )
        self.diq_desc = self._make_tensor_desc(
            sample_d_index_q, name="sample_d_index_q"
        )
        self.dw_desc = self._make_tensor_desc(
            sample_d_weights, name="sample_d_weights"
        )
        self.dik_desc = self._make_tensor_desc(
            sample_d_index_k, name="sample_d_index_k"
        )
        self.attn_desc = self._make_tensor_desc(
            sample_attn_score, name="sample_attn_score"
        )
        self.attn_denom_desc = self._make_tensor_desc(
            sample_attn_l1norm, name="sample_attn_l1norm"
        )
        self.idx_score_desc = self._make_tensor_desc(
            sample_index_score, name="sample_index_score"
        )
        self.idx_lse_desc = self._make_tensor_desc(
            sample_index_lse, name="sample_index_lse"
        )

        self.sm_scale = float(sm_scale)
        self.block_I = int(block_I)
        self.ratio = int(ratio)
        self.is_thd = bool(is_thd)
        self.has_q_causal_offsets = bool(has_q_causal_offsets)

        if self.is_thd:
            if sample_index_q.ndim != 3 or sample_index_k.ndim != 2:
                raise ValueError(
                    "THD dense indexer backward expects q=(T_q, H, D) and "
                    "k=(T_k, D); got "
                    f"{tuple(sample_index_q.shape)} and "
                    f"{tuple(sample_index_k.shape)}"
                )
            if batch is None or max_seqlen_q is None or max_seqlen_k is None:
                raise ValueError(
                    "THD dense indexer backward requires batch and "
                    "max_seqlen_q/k"
                )
            total_q, heads, head_dim = sample_index_q.shape
            self.batch = int(batch)
            self.normalization_tokens = int(total_q)
            self.total_k = int(sample_index_k.shape[0])
            self.max_seqlen_q = int(max_seqlen_q)
            self.max_seqlen_k = int(max_seqlen_k)
        else:
            if sample_index_q.ndim != 4 or sample_index_k.ndim != 3:
                raise ValueError(
                    "BSHD dense indexer backward expects q=(B, S_q, H, D) and "
                    "k=(B, S_k, D); got "
                    f"{tuple(sample_index_q.shape)} and "
                    f"{tuple(sample_index_k.shape)}"
                )
            b, seqlen_q, heads, head_dim = sample_index_q.shape
            self.batch = int(b)
            self.normalization_tokens = int(b * seqlen_q)
            self.total_k = int(b * sample_index_k.shape[1])
            self.max_seqlen_q = int(seqlen_q)
            self.max_seqlen_k = int(sample_index_k.shape[1])

            # Upstream ignores ``max_seqlen_q/k`` for BSHD and derives both from
            # the tensor shapes, which is also what the kernel needs. Rather than
            # dropping a caller's value silently -- which makes the parameter
            # look effective when it is not -- a value that disagrees with the
            # shapes is rejected. Agreeing values are accepted as documentation.
            for _name, _given, _actual in (
                ("max_seqlen_q", max_seqlen_q, self.max_seqlen_q),
                ("max_seqlen_k", max_seqlen_k, self.max_seqlen_k),
            ):
                if _given is not None and int(_given) != _actual:
                    raise ValueError(
                        f"BSHD dense indexer backward derives {_name} from the "
                        f"tensor shapes ({_actual}); the value passed "
                        f"({int(_given)}) disagrees. Omit it for BSHD, or pass "
                        "a matching value. THD (cu_seqlens) is where an "
                        "explicit max_seqlen is used."
                    )
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self._plan_layout_validated = False

    def _validate_plan_shapes_and_layout(self) -> None:
        heads, head_dim = self.heads, self.head_dim
        if self.is_thd:
            t_q, t_k, max_k = (
                self.normalization_tokens,
                self.total_k,
                self.max_seqlen_k,
            )
            shape_checks = (
                (self.iq_desc, "index_q", (t_q, heads, head_dim)),
                (self.w_desc, "weights", (t_q, heads)),
                (self.ik_desc, "index_k", (t_k, head_dim)),
                (self.diq_desc, "d_index_q", (t_q, heads, head_dim)),
                (self.dw_desc, "d_weights", (t_q, heads)),
                (self.dik_desc, "d_index_k", (t_k, head_dim)),
                (self.attn_desc, "attn_score", (t_q, max_k)),
                (self.idx_score_desc, "index_score", (t_q, max_k)),
                (self.attn_denom_desc, "attn_l1norm", (t_q,)),
                (self.idx_lse_desc, "index_lse", (t_q,)),
            )
        else:
            seqlen_q, seqlen_k = self.max_seqlen_q, self.max_seqlen_k
            shape_checks = (
                (self.iq_desc, "index_q", (self.batch, seqlen_q, heads, head_dim)),
                (self.w_desc, "weights", (self.batch, seqlen_q, heads)),
                (self.ik_desc, "index_k", (self.batch, seqlen_k, head_dim)),
                (self.diq_desc, "d_index_q", (self.batch, seqlen_q, heads, head_dim)),
                (self.dw_desc, "d_weights", (self.batch, seqlen_q, heads)),
                (self.dik_desc, "d_index_k", (self.batch, seqlen_k, head_dim)),
                (self.attn_desc, "attn_score", (self.batch, seqlen_q, seqlen_k)),
                (self.idx_score_desc, "index_score", (self.batch, seqlen_q, seqlen_k)),
                (self.attn_denom_desc, "attn_l1norm", (self.batch, seqlen_q)),
                (self.idx_lse_desc, "index_lse", (self.batch, seqlen_q)),
            )
        for desc, name, expected in shape_checks:
            if desc is None:
                continue
            actual = tuple(desc.shape)
            if actual != tuple(expected):
                raise ValueError(
                    f"DenseIndexerBackward {name} shape mismatch: expected "
                    f"{tuple(expected)}, got {actual}"
                )
        for desc, name, _ in shape_checks:
            if desc is None:
                continue
            if not desc.is_contiguous():
                raise ValueError(
                    "DenseIndexerBackward requires contiguous (compact "
                    f"row-major) tensors; {name} has shape {tuple(desc.shape)} "
                    f"with non-compact stride {tuple(desc.stride)}."
                )
        self._plan_layout_validated = True

    def _validate_q_causal_offsets(self, q_causal_offsets) -> None:
        expected = (self.batch,)
        shape = tuple(q_causal_offsets.shape)
        if shape != expected:
            raise ValueError(
                "DenseIndexerBackward q_causal_offsets must have shape "
                f"{expected} (one offset per batch), got {shape}."
            )
        if q_causal_offsets.dtype != torch.int32:
            raise ValueError(
                "DenseIndexerBackward q_causal_offsets must be int32, got "
                f"{q_causal_offsets.dtype}."
            )
        if not q_causal_offsets.is_contiguous():
            raise ValueError(
                "DenseIndexerBackward q_causal_offsets must be contiguous, "
                f"got stride {tuple(q_causal_offsets.stride())}."
            )

    def check_support(self) -> bool:
        self._value_error_if(
            self.block_I <= 0, f"block_I must be positive, got {self.block_I}"
        )
        self._value_error_if(
            self.ratio < 1, f"ratio must be >= 1, got {self.ratio}"
        )
        # The 16-bit operand restriction belongs to the SQMMA *kernel*, not to
        # this surface: the torch implementation handles any floating dtype. So
        # only enforce it when the plan will actually build a kernel -- otherwise
        # a perfectly serviceable fp32 call would be rejected just because the
        # device happens to have a kernel that cannot take it.
        _uses_kernel = (
            _tilelang_dense_envelope(
                self.heads, self.head_dim, self.block_I, self.max_seqlen_k,
                self.ratio,
            )
            is None
        )
        _kernel_thd, _kernel_offsets = _dense_kernel_capability(
            self.iq_desc.device, self.is_thd, self.has_q_causal_offsets
        )
        if self.is_thd and not _kernel_thd:
            _uses_kernel = False
        if self.has_q_causal_offsets and not _kernel_offsets:
            _uses_kernel = False
        if _uses_kernel:
            for desc, name in (
                (self.iq_desc, "index_q"),
                (self.w_desc, "weights"),
                (self.ik_desc, "index_k"),
            ):
                if desc is None:
                    continue
                if desc.dtype not in (torch.bfloat16, torch.float16):
                    raise ValueError(
                        f"DenseIndexerBackward {name} must be bfloat16 or "
                        f"float16 (the SQMMA kernel operands), got {desc.dtype}"
                    )
        if not (self.iq_desc.dtype == self.w_desc.dtype == self.ik_desc.dtype):
            raise ValueError(
                "DenseIndexerBackward requires index_q, weights and index_k to "
                "share one dtype, got "
                f"{self.iq_desc.dtype}, {self.w_desc.dtype}, {self.ik_desc.dtype}"
            )
        for desc, name in (
            (self.attn_desc, "attn_score"),
            (self.attn_denom_desc, "attn_l1norm"),
            (self.idx_score_desc, "index_score"),
            (self.idx_lse_desc, "index_lse"),
        ):
            if desc is None:
                continue
            if desc.dtype != torch.float32:
                raise ValueError(
                    f"DenseIndexerBackward {name} must be float32, got "
                    f"{desc.dtype}"
                )
        for desc, name in (
            (self.diq_desc, "d_index_q"),
            (self.dw_desc, "d_weights"),
        ):
            if desc is None:
                continue
            if desc.dtype != self.iq_desc.dtype:
                raise ValueError(
                    f"DenseIndexerBackward {name} must match the index_q dtype "
                    f"({self.iq_desc.dtype}); the kernel stores the gradient "
                    f"with the input dtype, got {desc.dtype}"
                )
        if self.dik_desc is not None and self.dik_desc.dtype not in (
            torch.float32,
            self.iq_desc.dtype,
        ):
            raise ValueError(
                "DenseIndexerBackward d_index_k must be float32 or the index_q "
                f"dtype ({self.iq_desc.dtype}), got {self.dik_desc.dtype}"
            )
        self._validate_plan_shapes_and_layout()
        # Report a capability gap at ``check_support`` time rather than letting
        # ``compile()`` be the first to notice. The plan is otherwise valid, so
        # this is a property of the selected implementation, not of the tensors.
        if self.has_q_causal_offsets:
            _thd_ok, _offsets_ok = _dense_kernel_capability(
                self.iq_desc.device, self.is_thd, True
            )
            if not _offsets_ok:
                raise NotImplementedError(
                    "DenseIndexerBackward has_q_causal_offsets=True requires the "
                    "mp_32 dense kernel; no implementation available on this "
                    "device applies per-row causal offsets"
                )
        self._is_supported = True
        return True

    def compile(self):
        """Build the plan and return ``self`` for chaining.

        Dispatches to the TileLang MUSA ``dense_indexer_backward`` factory, the
        counterpart of upstream's architecture-specific dense kernel factory.
        The factory is picked from the device capability (see
        :func:`_dense_indexer_backward_factory`); every variant presents the
        same public ``dense_indexer_backward`` contract.
        """
        self._ensure_support_checked()
        if self._compiled_kernel is None:
            # Two independent reasons to use the torch implementation: a tile
            # shape the SQMMA envelope cannot express, and a layout the selected
            # kernel variant does not implement. Both are decided here so the
            # same input behaves the same way on every device.
            _outside = _tilelang_dense_envelope(
                self.heads,
                self.head_dim,
                self.block_I,
                self.max_seqlen_k,
                self.ratio,
            )
            _kernel_thd, _kernel_offsets = _dense_kernel_capability(
                self.iq_desc.device, self.is_thd, self.has_q_causal_offsets
            )
            if _outside is None and self.is_thd and not _kernel_thd:
                _outside = (
                    "THD/packed tensors are implemented by the mp_32 dense "
                    "kernel only; this device runs the generic one"
                )
            # ``has_q_causal_offsets`` is not an envelope question -- the shape can
            # be perfectly legal -- but the generic kernel rejects it outright, so
            # it has to force the decision here rather than being forwarded to a
            # factory that will raise with a message about the *kernel* instead of
            # about the missing capability.
            if _outside is None and self.has_q_causal_offsets and not _kernel_offsets:
                _outside = (
                    "q_causal_offsets is implemented by the mp_32 dense kernel "
                    "only, and the torch fallback has no per-row causal offset"
                )
            if _outside is not None:
                if self.has_q_causal_offsets and not _kernel_offsets:
                    # Nor does the torch implementation support offsets, so a
                    # fallback here would silently ignore them. Say so instead.
                    raise NotImplementedError(
                        "DenseIndexerBackward has_q_causal_offsets=True requires "
                        "the mp_32 dense kernel: the implementation selected for "
                        f"this device cannot apply per-row offsets ({_outside}). "
                        "Run on mp_32 or build the plan without "
                        "has_q_causal_offsets."
                    )
                self._compiled_kernel = _DenseIndexerBackwardFallback(
                    sm_scale=self.sm_scale,
                    ratio=self.ratio,
                    block_I=self.block_I,
                    reason=_outside or "kernel variant does not support THD",
                )
                return self
            _factory = _dense_indexer_backward_factory(self.iq_desc.device)

            self._compiled_kernel = _factory(
                batch=self.batch,
                max_seqlen_q=self.max_seqlen_q,
                max_seqlen_k=self.max_seqlen_k,
                heads=self.heads,
                dim=self.head_dim,
                sm_scale=self.sm_scale,
                block_I=self.block_I,
                ratio=self.ratio,
                is_varlen=self.is_thd,
                has_q_causal_offsets=self.has_q_causal_offsets,
                dtype=_tilelang_dtype(self.iq_desc.dtype),
            )
        return self

    def _check_execute_signature(self, *entries) -> None:
        for tensor, desc, name in entries:
            if desc is None:
                continue
            if tensor.dtype != desc.dtype:
                raise ValueError(
                    f"{name} dtype mismatch: this plan was compiled for "
                    f"{desc.dtype}, got {tensor.dtype}. Build a new plan (or "
                    "use dense_indexer_backward_wrapper) for a different "
                    "signature."
                )
            if tuple(tensor.shape) != tuple(desc.shape):
                raise ValueError(
                    f"{name} shape mismatch: this plan was compiled for "
                    f"{tuple(desc.shape)}, got {tuple(tensor.shape)}."
                )
            if desc.is_contiguous() and not tensor.is_contiguous():
                raise ValueError(
                    f"{name} stride/layout mismatch: this plan was compiled "
                    f"for stride {tuple(desc.stride)}, got "
                    f"{tuple(tensor.stride())}."
                )

    def execute(
        self,
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        grad_loss=None,
        loss_coeff: float = 1.0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        q_causal_offsets=None,
        current_stream=None,
    ) -> None:
        if q_causal_offsets is not None and not self.has_q_causal_offsets:
            raise ValueError(
                "DenseIndexerBackward received q_causal_offsets but the plan was "
                "built with has_q_causal_offsets=False; rebuild the plan with "
                "has_q_causal_offsets=True (the flag is part of the plan, not a "
                "per-call choice)"
            )
        if q_causal_offsets is None and self.has_q_causal_offsets:
            raise ValueError(
                "DenseIndexerBackward was built with has_q_causal_offsets=True "
                "but received q_causal_offsets=None; pass the per-batch offsets "
                "tensor"
            )
        if q_causal_offsets is not None:
            self._validate_q_causal_offsets(q_causal_offsets)
        if self._compiled_kernel is None:
            self.compile()
        if not self._plan_layout_validated:
            self._validate_plan_shapes_and_layout()
        if self.is_thd and (cu_seqlens_q is None or cu_seqlens_k is None):
            raise ValueError(
                "THD dense indexer backward requires both cu_seqlens_q and "
                "cu_seqlens_k (the plan was built with is_thd=True); got "
                f"cu_seqlens_q={'set' if cu_seqlens_q is not None else 'None'}, "
                f"cu_seqlens_k={'set' if cu_seqlens_k is not None else 'None'}")
        if not self.is_thd and (
            cu_seqlens_q is not None or cu_seqlens_k is not None
        ):
            raise ValueError(
                "BSHD dense indexer backward must not receive cu_seqlens_q/k; "
                "build the plan with is_thd=True for packed THD tensors"
            )
        self._check_execute_signature(
            (index_q, self.iq_desc, "index_q"),
            (weights, self.w_desc, "weights"),
            (index_k, self.ik_desc, "index_k"),
            (d_index_q, self.diq_desc, "d_index_q"),
            (d_weights, self.dw_desc, "d_weights"),
            (d_index_k, self.dik_desc, "d_index_k"),
            (attn_score, self.attn_desc, "attn_score"),
            (attn_l1norm, self.attn_denom_desc, "attn_l1norm"),
            (index_score, self.idx_score_desc, "index_score"),
            (index_lse, self.idx_lse_desc, "index_lse"),
        )
        grad_loss_operand = _grad_loss_operand(grad_loss, index_q.device)
        grad_scale = float(loss_coeff) / max(int(self.normalization_tokens), 1)

        # The scratch allocation, the launch and the cast back must all be
        # ordered against ``current_stream``. The kernel launches on that stream,
        # so an allocation or a ``copy_`` left on the ambient stream would race:
        # with a non-default stream the cast could read the scratch before the
        # kernel filled it, or the allocation could be handed back while the
        # kernel still writes it.
        with stream_context(current_stream, index_q.device):
            d_index_k_target = d_index_k
            if d_index_k.dtype == torch.float32:
                d_index_k_f32 = d_index_k
            else:
                d_index_k_f32 = torch.empty(
                    tuple(d_index_k.shape),
                    dtype=torch.float32,
                    device=d_index_k.device,
                )

            self._compiled_kernel(
                index_q,
                weights,
                index_k,
                d_index_q,
                d_weights,
                d_index_k_f32,
                attn_score,
                attn_l1norm,
                index_score,
                index_lse,
                grad_loss_operand,
                grad_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                q_causal_offsets,
                current_stream,
            )

            if d_index_k_f32 is not d_index_k_target:
                d_index_k_target.copy_(d_index_k_f32)

_CACHE_OF_INDEXER_BACKWARD_OBJECTS: dict = {}
_CACHE_OF_DENSE_INDEXER_BACKWARD_OBJECTS: dict = {}

def indexer_backward_wrapper(
    index_q,
    weights,
    index_k,
    attn_score,
    index_score,
    topk_indices,
    grad_loss=None,
    sm_scale: float = 1.0,
    loss_coeff: float = 1.0,
    block_I: Optional[int] = None,
    topk_indices_global: bool = False,
    d_index_q=None,
    d_weights=None,
    d_index_k=None,
    stream=None,
    *,
    backend: str = "default",
) -> TupleDict:
    """Sparse indexer backward. Returns ``{'d_index_q', 'd_weights', 'd_index_k'}``.

    ``index_score`` holds the predict distribution aligned slot-for-slot with
    ``topk_indices``; the fused softmax returned by compressed indexer forward
    can be passed directly. Upstream overwrites ``attn_score`` with
    ``grad_signal`` during the score-grad precompute and keeps ``index_score``
    read-only; the muDNN fallback recomputes the predict distribution instead
    and therefore leaves both score buffers untouched.

    Missing output buffers are allocated with ``torch.empty_like`` on the
    corresponding input. ``grad_loss`` accepts ``None`` (treated as the scalar
    ``1.0``), a Python float, or the single-element fp32 tensor upstream
    requires.

    ``backend`` is keyword-only and appended after the historical parameter
    list, exactly as upstream (with the device-neutral ``"v2"`` value).
    ``"v2"`` is rejected with ``NotImplementedError`` because no muDNN/MUSA
    kernel implements it.
    """
    backend = _validate_backend(backend)
    _reject_v2_backend(backend)
    _require_torch()

    _validate_float_tensor("index_q", index_q)
    _validate_float_tensor("weights", weights)
    _validate_float_tensor("index_k", index_k)
    _validate_fp32_tensor("attn_score", attn_score)
    _validate_fp32_tensor("index_score", index_score)
    _validate_int32_tensor("topk_indices", topk_indices)

    if index_q.ndim != 4:
        raise ValueError(
            "indexer_backward index_q must be 4D (B, S_q, H, D), got "
            f"{index_q.ndim}D shape {tuple(index_q.shape)}"
        )
    if index_k.ndim != 3:
        raise ValueError(
            "indexer_backward index_k must be 3D (B, S_k, D), got "
            f"{index_k.ndim}D shape {tuple(index_k.shape)}"
        )
    if topk_indices.ndim != 3:
        raise ValueError(
            "indexer_backward topk_indices must be 3D (B, S_q, topk), got "
            f"{topk_indices.ndim}D shape {tuple(topk_indices.shape)}"
        )

    # See the dense wrapper: strided tensors are legal to the caller and illegal
    # to the kernel's ``.view(...)`` indexing, so they are copied compact here.
    # ``attn_score`` is overwritten with ``grad_signal`` on a kernel-capable
    # device, so it and the outputs are copied back afterwards.
    index_q, _ = _as_contiguous(index_q)
    weights, _ = _as_contiguous(weights)
    index_k, _ = _as_contiguous(index_k)
    attn_score, _attn_score_restore = _as_contiguous(attn_score)
    index_score, _index_score_restore = _as_contiguous(index_score)
    topk_indices, _ = _as_contiguous(topk_indices)

    # ``_caller`` is what the caller passed (and what is returned); ``_work`` is
    # the compact buffer the kernel writes. They differ only for a strided
    # destination, where the result is copied back through ``_restore``.
    _d_index_q_caller = d_index_q
    _d_weights_caller = d_weights
    _d_index_k_caller = d_index_k
    _d_index_q_work, _d_index_q_restore = (
        (None, None) if d_index_q is None else _as_contiguous(d_index_q)
    )
    _d_weights_work, _d_weights_restore = (
        (None, None) if d_weights is None else _as_contiguous(d_weights)
    )
    _d_index_k_work, _d_index_k_restore = (
        (None, None) if d_index_k is None else _as_contiguous(d_index_k)
    )
    d_index_q = (torch.empty_like(index_q) if _d_index_q_work is None
                 else _d_index_q_work)
    d_weights = (torch.empty_like(weights) if _d_weights_work is None
                 else _d_weights_work)
    d_index_k = (torch.empty_like(index_k) if _d_index_k_work is None
                 else _d_index_k_work)

    batch, seqlen, heads, head_dim = index_q.shape
    seqlen_k = index_k.shape[1]
    topk = topk_indices.shape[-1]
    if block_I is None:
        # Resolve before the cache key so the key (and the plan) always carry
        # the concrete arch default rather than the ``None`` sentinel.
        block_I = _default_block_I(index_q.device)

    key = (
        index_q.device,
        index_q.dtype,
        weights.dtype,
        index_k.dtype,
        d_index_q.dtype,
        d_weights.dtype,
        d_index_k.dtype,
        int(batch),
        int(seqlen),
        int(seqlen_k),
        int(heads),
        int(head_dim),
        int(topk),
        float(sm_scale),
        int(block_I),
        bool(topk_indices_global),
        backend,
    )
    obj = _CACHE_OF_INDEXER_BACKWARD_OBJECTS.get(key)
    if obj is None:
        obj = IndexerBackward(
            sample_index_q=index_q,
            sample_weights=weights,
            sample_index_k=index_k,
            sample_d_index_q=d_index_q,
            sample_d_weights=d_weights,
            sample_d_index_k=d_index_k,
            sample_attn_score=attn_score,
            sample_index_score=index_score,
            sample_topk_indices=topk_indices,
            sm_scale=sm_scale,
            block_I=block_I,
            topk_indices_global=topk_indices_global,
            backend=backend,
        )
        # Not an ``assert``: under ``python -O`` assertions vanish and the plan
        # would be used without ever being validated.
        if not obj.check_support():
            raise ValueError(
                f"{obj.__class__.__name__}.check_support() rejected the plan "
                "built from these tensors"
            )
        obj.compile()
        _CACHE_OF_INDEXER_BACKWARD_OBJECTS[key] = obj

    obj.execute(
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k,
        attn_score,
        index_score,
        topk_indices,
        grad_loss=grad_loss,
        loss_coeff=loss_coeff,
        current_stream=stream,
    )

    # ``attn_score`` now holds grad_signal on a kernel-capable device, and the
    # outputs hold the gradients; write all four back through the caller's views.
    for _restore in (
        _attn_score_restore,
        _index_score_restore,
        _d_index_q_restore,
        _d_weights_restore,
        _d_index_k_restore,
    ):
        if _restore is not None:
            _restore()

    return TupleDict(
        d_index_q=d_index_q if _d_index_q_caller is None else _d_index_q_caller,
        d_weights=d_weights if _d_weights_caller is None else _d_weights_caller,
        d_index_k=d_index_k if _d_index_k_caller is None else _d_index_k_caller,
    )

def dense_indexer_backward_wrapper(
    index_q,
    weights,
    index_k,
    attn_score,
    attn_l1norm,
    index_score,
    index_lse,
    grad_loss=None,
    sm_scale: float = 1.0,
    loss_coeff: float = 1.0,
    # 64 rather than upstream's 128 -- see ``DenseIndexerBackward``: 128 exceeds
    # the generic kernel's 192 KB shared-memory budget at the default
    # ``num_stages``/``q_rows``, which made the default call fail on every
    # non-mp_32 device. Pass 128 explicitly on mp_32 if you want it.
    block_I: int = 64,
    ratio: int = 1,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets=None,
    d_index_q=None,
    d_weights=None,
    d_index_k=None,
    stream=None,
) -> TupleDict:
    """Dense indexer backward. Returns ``{'d_index_q', 'd_weights', 'd_index_k'}``.

    Dispatches to the TileLang MUSA ``dense_indexer_backward`` kernel plan.
    ``attn_score`` and ``index_score`` are the raw dense scores produced by
    ``dense_attn_score_recompute_wrapper`` and
    ``dense_indexer_score_recompute_wrapper``; ``attn_l1norm`` and
    ``index_lse`` are their denominators. The score-grad stage overwrites
    ``index_score`` in place with ``grad_signal`` (upstream's contract);
    ``attn_score`` stays read-only.

    Keyword-only, like upstream: ``sm_scale`` (must be > 0), ``loss_coeff``,
    ``block_I``, ``ratio``, ``max_seqlen_q``, ``max_seqlen_k``, optional
    pre-allocated outputs, and ``stream``. A bf16 ``d_index_k`` output is served
    by an internal fp32 scratch plus a cast, matching upstream's fp32
    accumulation.

    Both layouts are accepted, as upstream: BSHD ``(B, S_q, ...)``, and THD
    ``(T_q, ...)`` when ``cu_seqlens_q`` / ``cu_seqlens_k`` are passed together.
    ``q_causal_offsets`` is accepted where a supporting implementation is
    available; see :class:`DenseIndexerBackward` for the capability matrix. A
    feature the selected implementation cannot express is reported instead of
    being ignored.
    """
    _require_torch()

    _validate_float_tensor("index_q", index_q)
    _validate_float_tensor("weights", weights)
    _validate_float_tensor("index_k", index_k)
    _validate_fp32_tensor("attn_score", attn_score)
    _validate_fp32_tensor("attn_l1norm", attn_l1norm)
    _validate_fp32_tensor("index_score", index_score)
    _validate_fp32_tensor("index_lse", index_lse)
    if int(ratio) < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio}")
    if int(block_I) <= 0:
        raise ValueError(f"block_I must be positive, got {block_I}")

    # Strided tensors are legal for the caller and illegal for the kernel, which
    # indexes with ``.view(...)``. Copy them compact rather than rejecting the
    # call: upstream does the same, and a rejected call is indistinguishable from
    # an unsupported one. ``index_score`` is mutated in place by the score-grad
    # stage, so it and the three outputs are copied back.
    index_q, _ = _as_contiguous(index_q)
    weights, _ = _as_contiguous(weights)
    index_k, _ = _as_contiguous(index_k)
    attn_score, _ = _as_contiguous(attn_score)
    attn_l1norm, _ = _as_contiguous(attn_l1norm)
    index_lse, _ = _as_contiguous(index_lse)
    index_score, _index_score_restore = _as_contiguous(index_score)

    is_thd = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_thd:
        # THD is implemented by the mp_32 dense kernel and by the torch
        # fallback; ``compile()`` picks between them. Both need the packed
        # layout and the two bounds, so they are resolved here.
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError(
                "THD dense indexer backward needs both cu_seqlens_q and "
                "cu_seqlens_k; got only one of them"
            )
        _validate_int32_tensor("cu_seqlens_q", cu_seqlens_q)
        _validate_int32_tensor("cu_seqlens_k", cu_seqlens_k)
        if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
            raise ValueError(
                "cu_seqlens_q / cu_seqlens_k must be 1-D (B + 1,) int32"
            )
        if int(cu_seqlens_q.shape[0]) != int(cu_seqlens_k.shape[0]):
            raise ValueError(
                "cu_seqlens_q and cu_seqlens_k must describe the same batch, "
                f"got {int(cu_seqlens_q.shape[0])} and "
                f"{int(cu_seqlens_k.shape[0])} offsets"
            )
        if index_q.ndim != 3 or weights.ndim != 2 or index_k.ndim != 2:
            raise ValueError(
                "THD dense indexer backward expects q=(T_q, H, D), "
                "w=(T_q, H), k=(T_k, D); got "
                f"{tuple(index_q.shape)}, {tuple(weights.shape)}, "
                f"{tuple(index_k.shape)}"
            )
        batch = int(cu_seqlens_q.shape[0]) - 1
        total_q = int(index_q.shape[0])
        total_k = int(index_k.shape[0])
        # ``max_seqlen_*`` is a real input for THD: the score tensor is
        # ``(T_q, max_seqlen_k)``, so it cannot be derived from ``T_k`` (which is
        # the packed total). Falling back to the per-batch maximum keeps the
        # upstream "optional" contract.
        max_seqlen_q = (
            int(max_seqlen_q) if max_seqlen_q is not None
            else _max_from_cu_seqlens(cu_seqlens_q)
        )
        max_seqlen_k = (
            int(max_seqlen_k) if max_seqlen_k is not None
            else _max_from_cu_seqlens(cu_seqlens_k)
        )
    else:
        if index_q.ndim != 4 or weights.ndim != 3 or index_k.ndim != 3:
            raise ValueError(
                "BSHD dense indexer backward expects q=(B, S_q, H, D), "
                "w=(B, S_q, H), k=(B, S_k, D); got "
                f"{tuple(index_q.shape)}, {tuple(weights.shape)}, "
                f"{tuple(index_k.shape)}"
            )
        batch = int(index_q.shape[0])
        total_q = batch * int(index_q.shape[1])
        total_k = batch * int(index_k.shape[1])
        # Derived from the shapes, as upstream does; a disagreeing explicit value
        # is rejected by the plan rather than dropped.
        max_seqlen_q = int(index_q.shape[1])
        max_seqlen_k = int(index_k.shape[1])

    # ``_caller`` is what the caller passed (and what is returned); ``_work`` is
    # the compact buffer the kernel writes. They differ only for a strided
    # destination, where the result is copied back through ``_restore``.
    _d_index_q_caller = d_index_q
    _d_weights_caller = d_weights
    _d_index_k_caller = d_index_k
    _d_index_q_work, _d_index_q_restore = (
        (None, None) if d_index_q is None else _as_contiguous(d_index_q)
    )
    _d_weights_work, _d_weights_restore = (
        (None, None) if d_weights is None else _as_contiguous(d_weights)
    )
    _d_index_k_work, _d_index_k_restore = (
        (None, None) if d_index_k is None else _as_contiguous(d_index_k)
    )
    d_index_q = (torch.empty_like(index_q) if _d_index_q_work is None
                 else _d_index_q_work)
    d_weights = (torch.empty_like(weights) if _d_weights_work is None
                 else _d_weights_work)
    d_index_k = (torch.empty_like(index_k) if _d_index_k_work is None
                 else _d_index_k_work)

    key = (
        index_q.device,
        index_q.dtype,
        weights.dtype,
        index_k.dtype,
        d_index_k.dtype,
        bool(is_thd),
        batch,
        int(index_q.shape[-1]),
        max_seqlen_q,
        max_seqlen_k,
        total_q if is_thd else None,
        total_k if is_thd else None,
        float(sm_scale),
        int(block_I),
        int(ratio),
        bool(is_thd),
        q_causal_offsets is not None,
    )
    obj = _CACHE_OF_DENSE_INDEXER_BACKWARD_OBJECTS.get(key)
    if obj is None:
        obj = DenseIndexerBackward(
            sample_index_q=index_q,
            sample_weights=weights,
            sample_index_k=index_k,
            sample_d_index_q=d_index_q,
            sample_d_weights=d_weights,
            sample_d_index_k=d_index_k,
            sample_attn_score=attn_score,
            sample_attn_l1norm=attn_l1norm,
            sample_index_score=index_score,
            sample_index_lse=index_lse,
            sm_scale=sm_scale,
            block_I=block_I,
            ratio=ratio,
            is_thd=is_thd,
            batch=batch,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            has_q_causal_offsets=q_causal_offsets is not None,
        )
        # Not an ``assert``: under ``python -O`` assertions vanish and the plan
        # would be used without ever being validated.
        if not obj.check_support():
            raise ValueError(
                f"{obj.__class__.__name__}.check_support() rejected the plan "
                "built from these tensors"
            )
        obj.compile()
        _CACHE_OF_DENSE_INDEXER_BACKWARD_OBJECTS[key] = obj

    obj.execute(
        index_q,
        weights,
        index_k,
        d_index_q,
        d_weights,
        d_index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        grad_loss=grad_loss,
        loss_coeff=loss_coeff,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_causal_offsets=q_causal_offsets,
        current_stream=stream,
    )

    # Restore the caller's layout. ``index_score`` holds grad_signal now, and the
    # three outputs hold the gradients, so all four have to be written back
    # through the strided view the caller passed in.
    for _restore in (
        _index_score_restore,
        _d_index_q_restore,
        _d_weights_restore,
        _d_index_k_restore,
    ):
        if _restore is not None:
            _restore()

    return TupleDict(
        d_index_q=d_index_q if _d_index_q_caller is None else _d_index_q_caller,
        d_weights=d_weights if _d_weights_caller is None else _d_weights_caller,
        d_index_k=d_index_k if _d_index_k_caller is None else _d_index_k_caller,
    )
