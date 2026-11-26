# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrappers for the fused CSA/HCA compressor TileLang kernels.

``CSACompressorForward`` and ``CSACompressorBackward`` wrap the TileLang
kernels in ``compressor_sm100.py`` and ``compressor_sm100_r128.py`` for the
THD-packed layout::

    out[b, j] = sum_k kv[w(b, k), c(k, j)] *
                softmax_k(score[w(b, k), c(k, j)] + ape[k % ratio, c(k, j)])

The per-block window has ``ratio`` entries for ``coff == 1`` and ``2 * ratio``
entries for ``coff == 2``; the latter's previous-half window is invalid for the
first block of every segment.  Autograd wiring remains in the caller.  These
APIs only provide descriptor validation, lazy TileLang compilation, and kernel
execution.

The validated MUSA envelope is BF16 ``kv``/``score``/``out``, FP32 ``ape``,
INT32 cumulative offsets, int32-safe flat offsets, ``ratio`` in ``{2, 4, 128}``,
and ``coff`` in ``{1, 2}``.  Ratio 128 is currently validated for
``head_dim`` in ``{128, 512}``.  Unsupported configurations raise instead of
falling back to an eager implementation.

Forward, dKV, and dScore use FP32 intermediates before the final BF16 store.
dAPE is accumulated with FP32 atomics and is therefore not bitwise deterministic;
backward refuses strict deterministic mode.  The MUSA TileLang adapter launches
on the current MUSA stream, so explicit streams are selected with
``torch.musa.stream`` around the launch; no CUDA/NVTX dependency is required.
"""

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator, Optional

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

import torch

from mudnn._torch_stream import as_torch_stream, stream_context
from mudnn.api_base import APIBase, TupleDict

from .compressor_sm100 import (
    CU_ALIGN_BYTES,
    PTR_ALIGN_BYTES,
    precompile_bwd,
    precompile_fwd,
    run_bwd,
    run_fwd,
)
from .compressor_sm100_r128 import (
    precompile_bwd_r128,
    precompile_fwd_r128,
    run_bwd_r128,
    run_fwd_r128,
)

SUPPORTED_RATIOS = (2, 4, 128)
SUPPORTED_COFF = (1, 2)
R128_HEAD_DIMS = (128, 512)
MINIMUM_MUSA_ARCH_MAJOR = 3
_MAX_HEAD_DIM = 128 * 65535
_INT32_LIMIT = 2**31
_API_CACHE_MAXSIZE = 256


class _LruDict:
    """Bounded thread-safe LRU mapping for compiled API instances."""

    def __init__(self, maxsize: int = _API_CACHE_MAXSIZE):
        self._data: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get(self, key, default=None):
        with self._lock:
            if key not in self._data:
                return default
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)


def _is_musa_available(torch) -> bool:
    musa = getattr(torch, "musa", None)
    return musa is not None and bool(musa.is_available())


@contextmanager
def _device_context(device):
    """Anchor compilation, allocation and launch in the tensor's MUSA device."""

    with torch.musa.device(device):
        yield


def _reject_deterministic_backward() -> None:
    """Reject strict deterministic mode because dAPE uses FP32 atomics."""

    if not torch.are_deterministic_algorithms_enabled():
        return
    message = (
        "CSA compressor backward accumulates dAPE with fp32 atomics and is not "
        "deterministic; torch.use_deterministic_algorithms(True) is set. Use a "
        "deterministic (eager) implementation instead."
    )
    if torch.is_deterministic_algorithms_warn_only_enabled():
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    else:
        raise RuntimeError(message)


def _resolve_stream_handle(current_stream) -> Optional[int]:
    """Integer MUSA stream handle for the TileLang launch path."""

    if current_stream is None:
        return None
    if isinstance(current_stream, int):
        return int(current_stream)
    return int(current_stream.musa_stream)


@contextmanager
def _torch_stream_context(current_stream, device: torch.device) -> Iterator[None]:
    """Run torch work on ``current_stream`` when one is given."""

    with stream_context(current_stream, device):
        yield


class _CSACompressorBase(APIBase):
    """Shared descriptor plumbing and ``check_support`` for forward and backward."""

    def __init__(
        self,
        sample_kv: torch.Tensor,  # (total_tokens, coff * head_dim) BF16
        sample_score: torch.Tensor,  # (total_tokens, coff * head_dim) BF16
        sample_ape: torch.Tensor,  # (ratio, coff * head_dim) FP32
        sample_cu_seqlens: torch.Tensor,  # (B + 1,) INT32 token offsets
        sample_cu_seqlens_comp: torch.Tensor,  # (B + 1,) INT32 compressed offsets
        sample_out: torch.Tensor,  # (total_comp, head_dim) BF16
        ratio: int = 4,
        coff: int = 2,
    ):
        """Capture tensor metadata and the ``(ratio, coff)`` configuration.

        The sample tensors provide shape, dtype, stride, and device metadata;
        meta tensors are accepted.  Validation runs in ``check_support`` and no
        device data is read until ``execute``.
        """
        super().__init__()
        self._warn_experimental_api()
        self.kv_desc = self._make_tensor_desc(sample_kv, name="sample_kv")
        self.score_desc = self._make_tensor_desc(sample_score, name="sample_score")
        self.ape_desc = self._make_tensor_desc(sample_ape, name="sample_ape")
        self.cu_desc = self._make_tensor_desc(sample_cu_seqlens, name="sample_cu_seqlens")
        self.cuc_desc = self._make_tensor_desc(sample_cu_seqlens_comp, name="sample_cu_seqlens_comp")
        self.out_desc = self._make_tensor_desc(sample_out, name="sample_out")
        self.ratio = int(ratio)
        self.coff = int(coff)
        self.total_tokens = None
        self.total_comp = None
        self.head_dim = None
        self.n_seg = None
        self.target_device: Optional[torch.device] = None

    def check_support(self) -> bool:
        """Validate the configuration against the kernels' validated envelope."""

        self._logger.debug("Entering check_support")
        if self.ratio in (2, 4):
            self._value_error_if(
                self.coff not in (1, 2),
                f"CSA compressor at ratio={self.ratio} is validated for coff in " f"{{1, 2}}, got coff={self.coff}",
            )
        elif self.ratio == 128:
            self._value_error_if(
                self.coff not in (1, 2),
                f"CSA compressor at ratio=128 supports coff in {{1, 2}}, " f"got coff={self.coff}",
            )
        else:
            self._value_error_if(
                True,
                "CSA compressor is validated for ratio in {2, 4, 128} only, " f"got ratio={self.ratio}, coff={self.coff}",
            )

        self._value_error_if(
            self.kv_desc.ndim != 2,
            f"kv must be 2-D (total_tokens, coff * head_dim), got " f"{self.kv_desc.shape}",
        )
        self._value_error_if(
            self.out_desc.ndim != 2,
            f"out/grad_out must be 2-D (total_comp, head_dim), got " f"{self.out_desc.shape}",
        )
        total_tokens, width = self.kv_desc.shape
        total_comp, head_dim = self.out_desc.shape
        self._value_error_if(
            head_dim < 1 or width != self.coff * head_dim,
            f"kv width must equal coff * head_dim = {self.coff} * {head_dim}, " f"got {width}",
        )
        self._value_error_if(
            self.score_desc.shape != self.kv_desc.shape,
            f"score shape {self.score_desc.shape} != kv shape " f"{self.kv_desc.shape}",
        )
        self._value_error_if(
            self.ape_desc.shape != (self.ratio, width),
            f"ape must be (ratio, coff * head_dim) = ({self.ratio}, {width}), " f"got {self.ape_desc.shape}",
        )
        self._value_error_if(
            self.cu_desc.ndim != 1 or self.cuc_desc.ndim != 1,
            "cu_seqlens and cu_seqlens_comp must be 1-D",
        )
        self._value_error_if(
            self.cu_desc.shape != self.cuc_desc.shape or self.cu_desc.shape[0] < 2,
            "cu_seqlens and cu_seqlens_comp must both have B + 1 >= 2 " f"entries, got {self.cu_desc.shape} and {self.cuc_desc.shape}",
        )

        self._check_dtype(self.kv_desc, torch.bfloat16, name="kv")
        self._check_dtype(self.score_desc, torch.bfloat16, name="score")
        self._check_dtype(self.ape_desc, torch.float32, name="ape")
        self._check_dtype(self.cu_desc, torch.int32, name="cu_seqlens")
        self._check_dtype(self.cuc_desc, torch.int32, name="cu_seqlens_comp")
        self._check_dtype(self.out_desc, torch.bfloat16, name="out/grad_out")

        self._value_error_if(
            total_tokens * width >= _INT32_LIMIT,
            "total_tokens * coff * head_dim must be less than 2**31 for " f"int32 flat offsets, got {total_tokens} * {width}",
        )
        self._value_error_if(
            total_comp * head_dim >= _INT32_LIMIT,
            "total_comp * head_dim must be less than 2**31 for int32 flat " f"offsets, got {total_comp} * {head_dim}",
        )
        self._value_error_if(
            self.ratio * width >= _INT32_LIMIT,
            "ratio * coff * head_dim must be less than 2**31 for int32 APE " f"offsets, got {self.ratio} * {width}",
        )
        self._value_error_if(
            head_dim > _MAX_HEAD_DIM,
            f"head_dim must be <= {_MAX_HEAD_DIM} (forward launch gridDim.y " f"bound), got {head_dim}",
        )
        if self.ratio == 128:
            self._value_error_if(
                head_dim not in R128_HEAD_DIMS,
                "CSA compressor ratio=128 is validated for head_dim in " f"{R128_HEAD_DIMS}, got {head_dim}",
            )
        self._value_error_if(
            total_comp > 0 and total_tokens < self.ratio,
            f"total_comp={total_comp} > 0 requires at least ratio={self.ratio} " f"tokens, got {total_tokens}",
        )
        all_descs = (
            self.kv_desc,
            self.score_desc,
            self.ape_desc,
            self.cu_desc,
            self.cuc_desc,
            self.out_desc,
        )
        for desc, name in zip(
            all_descs,
            ("kv", "score", "ape", "cu_seqlens", "cu_seqlens_comp", "out/grad_out"),
        ):
            self._value_error_if(not desc.is_contiguous(), f"{name} must be contiguous")

        devices = {desc.device for desc in all_descs}
        self._value_error_if(
            any(dev.type not in ("musa", "meta") for dev in devices),
            "all tensors must be MUSA tensors, got devices " f"{sorted(str(dev) for dev in devices)}",
        )
        musa_devices = {dev for dev in devices if dev.type == "musa"}
        self._value_error_if(
            len(musa_devices) > 1,
            "all tensors must share one MUSA device, got " f"{sorted(str(dev) for dev in musa_devices)}",
        )
        if musa_devices:
            self._runtime_error_if(
                not _is_musa_available(torch),
                "CSA compressor requires an available MUSA device",
            )
            target = next(iter(musa_devices))
            if target.index is None:
                target = torch.device("musa", torch.musa.current_device())
        elif _is_musa_available(torch):
            target = torch.device("musa", torch.musa.current_device())
        else:
            target = None

        if target is not None:
            capability = torch.musa.get_device_capability(target)
            self._runtime_error_if(
                capability[0] < MINIMUM_MUSA_ARCH_MAJOR,
                "CSA compressor requires MUSA architecture major >= " f"{MINIMUM_MUSA_ARCH_MAJOR}, found " f"mp_{capability[0]}{capability[1]} on {target}",
            )

        self.total_tokens = total_tokens
        self.total_comp = total_comp
        self.head_dim = head_dim
        self.n_seg = self.cu_desc.shape[0] - 1
        self.target_device = target
        self._is_supported = True
        return True

    def _validate_runtime_tensor(self, tensor, name, shape, dtype, device, align):
        """Cheap per-call validation of one runtime tensor."""

        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.device.type != "musa" or tensor.device != device:
            raise ValueError(f"{name} must be a MUSA tensor on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride {tuple(tensor.stride())}")
        if tensor.data_ptr() % align:
            raise ValueError(f"{name} base pointer must be {align}-byte aligned, " f"got 0x{tensor.data_ptr():x}")

    @staticmethod
    def _record_streams(tensors, current_stream, device: torch.device) -> None:
        """Keep tensor storage alive for work enqueued on an explicit stream."""

        if current_stream is None:
            return
        consumer = as_torch_stream(current_stream, device)
        for tensor in tensors:
            tensor.record_stream(consumer)


class CSACompressorForward(_CSACompressorBase):
    """Fused CSA compressor forward: one kernel over the whole THD pack.

    Rows in ``[cu_seqlens_comp[-1], total_comp)`` are static-capacity padding
    (for MUSA-graph static shapes) and are computed with first-in-segment
    semantics from token 0, exactly like the eager gather.
    """

    def check_support(self) -> bool:
        super().check_support()
        if self.head_dim not in (128, 512):
            self._is_supported = False
            raise ValueError(
                f"CSA compressor forward is validated for head_dim in (128, 512), got {self.head_dim}"
            )
        return True

    def compile(self) -> None:
        """JIT-compile the forward kernel for this ``(ratio, head_dim, coff, device)`` (idempotent)."""

        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        # Route by ratio, as in the CUDA frontend.  The MUSA precompile
        # helpers own device selection, capture validation and JIT caching.
        if self.ratio == 128:
            precompile_fwd_r128(
                self.ratio, self.head_dim, self.coff, self.target_device,
                nb_total=self.total_comp, num_offsets=self.n_seg + 1,
            )
            run = run_fwd_r128
        else:
            precompile_fwd(
                self.ratio, self.head_dim, self.coff, self.target_device,
                nb_total=self.total_comp, num_offsets=self.n_seg + 1,
            )
            run = run_fwd

        ratio, head_dim, coff = self.ratio, self.head_dim, self.coff

        def tensor_api(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, stream_handle):
            """Per-config closure: invoke the routed MUSA forward kernel."""

            # The MUSA TVM-FFI adapter does not expose a ``stream`` keyword;
            # the caller selects the launch stream with ``torch.musa.stream``.
            run(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, out.shape[0], ratio, head_dim, coff, stream_handle=stream_handle)

        self._compiled_kernel = tensor_api
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_comp: torch.Tensor,
        out: torch.Tensor,
        current_stream: Optional[torch.musa.Stream] = None,
    ) -> None:
        """Run the compiled forward kernel; ``out`` is fully overwritten."""

        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise ValueError("CSACompressorForward kernel not compiled")
        device = self.target_device
        width = self.coff * self.head_dim
        self._validate_runtime_tensor(kv, "kv", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(score, "score", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(ape, "ape", (self.ratio, width), torch.float32, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(cu_seqlens, "cu_seqlens", (self.n_seg + 1,), torch.int32, device, CU_ALIGN_BYTES)
        self._validate_runtime_tensor(cu_seqlens_comp, "cu_seqlens_comp", (self.n_seg + 1,), torch.int32, device, CU_ALIGN_BYTES)
        self._validate_runtime_tensor(out, "out", (self.total_comp, self.head_dim), torch.bfloat16, device, PTR_ALIGN_BYTES)
        if out.numel() == 0:
            return
        with _device_context(device), _torch_stream_context(current_stream, device):
            self._compiled_kernel(kv, score, ape, cu_seqlens, cu_seqlens_comp, out, _resolve_stream_handle(current_stream))
        self._record_streams((kv, score, ape, cu_seqlens, cu_seqlens_comp, out), current_stream, device)


class CSACompressorBackward(_CSACompressorBase):
    """Fused CSA compressor backward: recompute window probs and write gradients.

    ``grad_kv`` and ``grad_score`` are fully overwritten, including exact zeros for
    unused slots.  ``grad_ape`` is accumulated in place with FP32 atomics; class API
    callers must zero it when they want fresh rather than accumulated gradients.
    """

    def compile(self) -> None:
        """JIT-compile the backward kernel for this ``(ratio, head_dim, coff, device)`` (idempotent)."""

        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        if self._compiled_kernel is not None:
            return
        try:
            # Route by ratio, as in the forward path.
            if self.ratio == 128:
                precompile_bwd_r128(self.ratio, self.head_dim, self.coff, self.target_device, nb_total=self.total_comp)
                run = run_bwd_r128
            else:
                precompile_bwd(
                    self.ratio, self.head_dim, self.coff, self.target_device,
                    nb_total=self.total_comp, num_offsets=self.n_seg + 1,
                )
                run = run_bwd
        except ImportError as exc:
            raise NotImplementedError("CSA TileLang execution requires the tilelang_musa runtime") from exc

        ratio, head_dim, coff = self.ratio, self.head_dim, self.coff

        def tensor_api(
            kv,
            score,
            ape,
            cu_seqlens,
            cu_seqlens_comp,
            grad_out,
            grad_kv,
            grad_score,
            grad_ape,
            stream_handle,
        ):
            """Per-config closure: invoke the routed MUSA backward kernel."""

            # The MUSA TVM-FFI adapter launches on the current MUSA stream.
            run(
                kv,
                score,
                ape,
                cu_seqlens,
                cu_seqlens_comp,
                grad_out,
                grad_kv,
                grad_score,
                grad_ape,
                grad_out.shape[0],
                ratio,
                head_dim,
                coff,
                stream_handle=stream_handle,
            )

        self._compiled_kernel = tensor_api
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        ape: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_comp: torch.Tensor,
        grad_out: torch.Tensor,
        grad_kv: torch.Tensor,
        grad_score: torch.Tensor,
        grad_ape: torch.Tensor,
        current_stream: Optional[torch.musa.Stream] = None,
    ) -> None:
        """Run the compiled backward kernel into the gradient buffers."""

        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise ValueError("CSACompressorBackward kernel not compiled")
        _reject_deterministic_backward()
        device = self.target_device
        width = self.coff * self.head_dim
        self._validate_runtime_tensor(kv, "kv", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(score, "score", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(ape, "ape", (self.ratio, width), torch.float32, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(cu_seqlens, "cu_seqlens", (self.n_seg + 1,), torch.int32, device, CU_ALIGN_BYTES)
        self._validate_runtime_tensor(cu_seqlens_comp, "cu_seqlens_comp", (self.n_seg + 1,), torch.int32, device, CU_ALIGN_BYTES)
        self._validate_runtime_tensor(grad_out, "grad_out", (self.total_comp, self.head_dim), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(grad_kv, "grad_kv", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(grad_score, "grad_score", (self.total_tokens, width), torch.bfloat16, device, PTR_ALIGN_BYTES)
        self._validate_runtime_tensor(grad_ape, "grad_ape", (self.ratio, width), torch.float32, device, PTR_ALIGN_BYTES)
        if grad_out.numel() == 0:
            return
        with _device_context(device), _torch_stream_context(current_stream, device):
            # The MUSA TileLang backward kernel writes consumed positions only.
            # Zeroing these buffers preserves the competitor's full-overwrite
            # contract for segment tails, overlap-only columns and static padding.
            grad_kv.zero_()
            grad_score.zero_()
            self._compiled_kernel(
                kv,
                score,
                ape,
                cu_seqlens,
                cu_seqlens_comp,
                grad_out,
                grad_kv,
                grad_score,
                grad_ape,
                _resolve_stream_handle(current_stream),
            )
        self._record_streams((kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out, grad_kv, grad_score, grad_ape), current_stream, device)


_api_cache = _LruDict()
_api_build_lock = threading.Lock()


def _get_api(kind, kv, score, ape, cu_seqlens, cu_seqlens_comp, out_shape, ratio, coff):
    """Build (or fetch) a compiled forward/backward API instance for these tensors."""
    key = (
        kind,
        int(ratio),
        int(coff),
        out_shape[1],
        tuple(kv.shape),
        out_shape[0],
        cu_seqlens.shape[0],
        kv.device.index,
    )
    api = _api_cache.get(key)
    if api is not None:
        return api
    with _api_build_lock:
        api = _api_cache.get(key)
        if api is not None:
            return api
        sample_out = torch.empty(out_shape, dtype=torch.bfloat16, device="meta")
        cls = CSACompressorForward if kind == "fwd" else CSACompressorBackward
        api = cls(
            sample_kv=kv,
            sample_score=score,
            sample_ape=ape,
            sample_cu_seqlens=cu_seqlens,
            sample_cu_seqlens_comp=cu_seqlens_comp,
            sample_out=sample_out,
            ratio=ratio,
            coff=coff,
        )
        api.check_support()
        api.compile()
        _api_cache.put(key, api)
        return api


def _infer_head_dim(kv: torch.Tensor, head_dim: Optional[int], coff: int) -> int:
    """Infer ``head_dim`` from the packed kv width when not given explicitly."""

    if kv.ndim != 2:
        raise ValueError("kv must be 2-D (total_tokens, coff * head_dim), " f"got {tuple(kv.shape)}")
    if head_dim is not None:
        return int(head_dim)
    width = kv.shape[1]
    if coff < 1 or width % coff != 0:
        raise ValueError(f"cannot infer head_dim from kv width {width} and coff {coff}")
    return width // coff


def csa_compressor_forward_wrapper(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    ratio: int = 4,
    head_dim: Optional[int] = None,
    coff: int = 2,
    total_comp: Optional[int] = None,
    stream: Optional[torch.musa.Stream] = None,
) -> TupleDict:
    """High-level forward wrapper. Allocates and returns the pooled output.

    Args:
        kv: ``(total_tokens, coff * head_dim)`` BF16 gate values (THD packed).
        score: ``(total_tokens, coff * head_dim)`` BF16 gate scores.
        ape: ``(ratio, coff * head_dim)`` FP32 additive position embedding.
        cu_seqlens: ``(B + 1,)`` int32 cumulative token counts per segment.
        cu_seqlens_comp: ``(B + 1,)`` int32 cumulative compressed-block counts,
            ``cu_seqlens_comp[b + 1] - cu_seqlens_comp[b] ==
            (cu_seqlens[b + 1] - cu_seqlens[b]) // ratio``.
        ratio: Compression ratio (tokens per output block); validated envelope:
            ``{2, 4, 128}`` (the wrappers route to the matching kernel family by ratio).
        head_dim: Output feature dimension; inferred from ``kv`` width when omitted.
        coff: ``1`` for the own-block window form (window = ``ratio`` tokens, no overlap) or
            ``2`` for the overlapping-window form (window = ``2 * ratio``); validated
            envelope: ``{1, 2}`` at all ratios (ratio=128 additionally requires ``head_dim``
            in ``{128, 512}``).
        total_comp: Output row count. Defaults to ``cu_seqlens_comp[-1]`` (synchronizes);
            pass it explicitly (e.g. a static MUSA-graph capacity, which must be
            ``>= cu_seqlens_comp[-1]``) to stay capture-safe.
        stream: MUSA stream for allocation and kernel launch (None -> current stream).

    Returns:
        ``{'out': (total_comp, head_dim) BF16}`` pooled output (pre-RMSNorm).
    """
    head_dim = _infer_head_dim(kv, head_dim, coff)
    if total_comp is None:
        if cu_seqlens_comp.numel() < 2:
            raise ValueError("cu_seqlens_comp must have B + 1 >= 2 entries")
        total_comp = int(cu_seqlens_comp[-1].item())
    api = _get_api(
        "fwd",
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        (int(total_comp), head_dim),
        ratio,
        coff,
    )
    with torch.musa.device(kv.device), _torch_stream_context(stream, kv.device):
        out = torch.empty(int(total_comp), head_dim, dtype=torch.bfloat16, device=kv.device)
    api.execute(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        out,
        current_stream=stream,
    )
    return TupleDict(out=out)


def csa_compressor_backward_wrapper(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    grad_out: torch.Tensor,
    ratio: int = 4,
    head_dim: Optional[int] = None,
    coff: int = 2,
    stream: Optional[torch.musa.Stream] = None,
) -> TupleDict:
    """High-level backward wrapper. Allocates the grad buffers and fills them.

    ``grad_out`` is ``(total_comp, head_dim)`` BF16 (the incoming gradient of the
    forward wrapper's ``out``); gradients on static-capacity padding rows are ignored.
    ``grad_kv``/``grad_score`` are allocated zeroed before launch because the TileLang
    kernel writes only consumed positions; unconsumed token/column positions therefore
    retain exact zeros.  ``grad_ape`` is allocated zeroed (fp32 atomic accumulation).
    When ``total_comp == 0`` the kernel is not launched and all three grads remain exact
    zeros.  The backward API raises under
    ``torch.use_deterministic_algorithms(True)`` because ``grad_ape`` uses FP32 atomics.

    Args:
        kv: ``(total_tokens, coff * head_dim)`` BF16 gate values (THD packed).
        score: ``(total_tokens, coff * head_dim)`` BF16 gate scores.
        ape: ``(ratio, coff * head_dim)`` FP32 additive position embedding.
        cu_seqlens: ``(B + 1,)`` int32 cumulative token counts per segment.
        cu_seqlens_comp: ``(B + 1,)`` int32 cumulative compressed-block counts,
            ``cu_seqlens_comp[b + 1] - cu_seqlens_comp[b] ==
            (cu_seqlens[b + 1] - cu_seqlens[b]) // ratio``.
        grad_out: ``(total_comp, head_dim)`` BF16 incoming gradient.
        ratio: Compression ratio; validated envelope ``{2, 4, 128}``.
        head_dim: Output feature dimension; inferred from ``kv`` width when omitted.
        coff: Window form ``{1, 2}``; ratio=128 additionally requires ``head_dim``
            in ``{128, 512}``.
        stream: MUSA stream for allocation and kernel launch (None -> current stream).

    Returns:
        ``{'grad_kv': (total_tokens, coff * head_dim) BF16,
           'grad_score': (total_tokens, coff * head_dim) BF16,
           'grad_ape': (ratio, coff * head_dim) FP32}``
    """
    head_dim = _infer_head_dim(kv, head_dim, coff)
    if grad_out.ndim != 2:
        raise ValueError("grad_out must be 2-D (total_comp, head_dim), " f"got {tuple(grad_out.shape)}")
    api = _get_api(
        "bwd",
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        tuple(grad_out.shape),
        ratio,
        coff,
    )
    with torch.musa.device(kv.device), _torch_stream_context(stream, kv.device):
        if grad_out.shape[0] == 0:
            # No kernel launch below -> the buffers must carry autograd's exact zeros.
            grad_kv = torch.zeros_like(kv)
            grad_score = torch.zeros_like(score)
        else:
            grad_kv = torch.empty_like(kv)
            grad_score = torch.empty_like(score)
        grad_ape = torch.zeros_like(ape, dtype=torch.float32)
    api.execute(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        grad_out,
        grad_kv,
        grad_score,
        grad_ape,
        current_stream=stream,
    )
    return TupleDict(grad_kv=grad_kv, grad_score=grad_score, grad_ape=grad_ape)
