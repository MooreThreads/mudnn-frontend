# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""APIBase integration for the MP31/MP32 DSA sparse-prefill forward kernel (QK=V=512)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from mudnn.api_base import APIBase, TupleDict
from ..utils.runtime import MP31_ARCH, MP32_ARCH, contiguous, device_capability, no_overlap, output_buffer, record_stream, resolve_stream
from ..utils.sparse_attention import validate_inputs

# (H, D_qk) -> supported indexer_topk.
_SUPPORTED_VARIANTS = {
    (64, 512): (0, 512, 1024, 2048),
    (128, 512): (0, 512, 1024),
}


def _compile(q, kv, topk_idxs, attn_sink, topk_length, indexer_topk):
    arch = device_capability(q.device)
    if arch == MP31_ARCH:
        from ._interface_mp31 import compile_mp31 as compile_kernel, launch_mp31 as launch
    elif arch == MP32_ARCH:
        from ._interface_mp32 import compile_mp32 as compile_kernel, launch_mp32 as launch
    else:
        raise NotImplementedError(f"sparse attention forward requires MP31 or MP32, found MP{arch[0]}{arch[1]}")
    with torch.musa.device(q.device):
        return compile_kernel(q, kv, topk_idxs, attn_sink, topk_length, indexer_topk), launch


class SparseAttentionForward(APIBase):
    def __init__(
        self,
        sample_q: torch.Tensor,  # (total_S_q, H, D_qk) BF16
        sample_kv: torch.Tensor,  # (total_S_kv, D_qk) BF16, K=V latent
        sample_topk_idxs: torch.Tensor,  # (total_S_q, logical_K) INT32
        sample_attn_sink: Optional[torch.Tensor] = None,  # (H,) FP32
        sample_topk_length: Optional[torch.Tensor] = None,  # (total_S_q,) INT32
        softmax_scale: Optional[float] = None,
        indexer_topk: int = 0,
    ):
        super().__init__()
        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.kv_desc = self._make_tensor_desc(sample_kv, name="sample_kv")
        self.topk_idxs_desc = self._make_tensor_desc(sample_topk_idxs, name="sample_topk_idxs")
        self.attn_sink_desc = self._make_tensor_desc(sample_attn_sink, name="sample_attn_sink")
        self.topk_length_desc = self._make_tensor_desc(sample_topk_length, name="sample_topk_length")
        self.softmax_scale = None if softmax_scale is None else float(softmax_scale)
        self.indexer_topk = int(indexer_topk)
        self.num_heads: Optional[int] = None
        self.head_dim: Optional[int] = None
        self.head_dim_v: Optional[int] = None
        self.logical_topk: Optional[int] = None

    def check_support(self) -> bool:
        validate_inputs(
            self.q_desc,
            self.kv_desc,
            self.topk_idxs_desc,
            self.attn_sink_desc,
            self.topk_length_desc,
            self.softmax_scale,
            empty=True,
        )
        _, num_heads, head_dim = self.q_desc.shape
        logical_topk = self.topk_idxs_desc.shape[1]
        variant = (num_heads, head_dim)
        self._value_error_if(
            variant not in _SUPPORTED_VARIANTS,
            f"SparseAttentionForward supports only (H, D_qk) in {tuple(_SUPPORTED_VARIANTS)}, got {variant}",
        )
        supported_indexer_topk = _SUPPORTED_VARIANTS[variant]
        self._value_error_if(
            self.indexer_topk not in supported_indexer_topk,
            f"indexer_topk={self.indexer_topk} is unsupported for (H, D_qk)={variant}; expected one of {supported_indexer_topk}",
        )
        self._value_error_if(
            self.indexer_topk > logical_topk,
            f"indexer_topk ({self.indexer_topk}) must not exceed logical K ({logical_topk})",
        )
        for name, desc in (
            ("q", self.q_desc), ("kv", self.kv_desc), ("topk_idxs", self.topk_idxs_desc),
            ("attn_sink", self.attn_sink_desc), ("topk_length", self.topk_length_desc),
        ):
            self._value_error_if(desc is not None and not desc.is_contiguous(), f"{name} must be contiguous; use the wrapper to normalize inputs")
        self._value_error_if(logical_topk % 64 != 0, "topk_idxs width must be a multiple of 64; use the wrapper to pad indices")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.head_dim_v = 512
        self.logical_topk = logical_topk
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        self._launcher = None
        if not self.q_desc.shape[0] or not self.kv_desc.shape[0] or not self.logical_topk:
            self._compiled_kernel = True
        else:
            self._compiled_kernel, self._launcher = _compile(
                self.q_desc, self.kv_desc, self.topk_idxs_desc, self.attn_sink_desc,
                self.topk_length_desc, self.indexer_topk,
            )

    def execute(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_idxs: torch.Tensor,
        *,
        attn_sink: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
        max_logits: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        lse_indexer: Optional[torch.Tensor] = None,
        current_stream: Optional[torch.musa.Stream] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self._compiled_kernel is None:
            raise ValueError("SparseAttentionForward kernel not compiled")
        scale = self.softmax_scale if softmax_scale is None else softmax_scale
        scale = validate_inputs(q, kv, topk_idxs, attn_sink, topk_length, scale, empty=True)
        if (q.shape[1], q.shape[2]) != (self.num_heads, self.head_dim):
            raise ValueError("Q heads and head_dim must match the compiled plan")
        if self.indexer_topk > topk_idxs.shape[1]:
            raise ValueError("indexer_topk must not exceed logical K")
        if self.indexer_topk == 0 and lse_indexer is not None:
            raise ValueError("lse_indexer must be None when indexer_topk == 0")
        shape = tuple(q.shape[:2])
        outputs = (out, max_logits, lse, lse_indexer)
        for name, tensor, dims, dtype in (
            ("out", out, (*shape, 512), q.dtype),
            ("max_logits", max_logits, shape, torch.float32),
            ("lse", lse, shape, torch.float32),
            ("lse_indexer", lse_indexer, shape, torch.float32),
        ):
            output_buffer(tensor, dims, dtype, q.device, name, required=name != "lse_indexer" or bool(self.indexer_topk))
            if tensor is not None and dtype == torch.bfloat16 and tensor.data_ptr() % 16:
                raise ValueError(f"{name} base pointer must be 16-byte aligned")
        inputs = (("q", q), ("kv", kv), ("topk_idxs", topk_idxs), ("attn_sink", attn_sink), ("topk_length", topk_length))
        for name, tensor in inputs:
            desc = getattr(self, name + "_desc")
            if (desc is None) != (tensor is None):
                raise ValueError(f"{name} presence must match the compiled plan")
            if desc is not None and (
                tuple(tensor.shape) != desc.shape or tuple(tensor.stride()) != desc.stride
                or tensor.dtype != desc.dtype or tensor.device != desc.device or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} shape, dtype, device, and layout must match the compiled plan")
            alignment = 16 if name in ("q", "kv") else (32 if name == "topk_idxs" else 4)
            if tensor is not None and tensor.data_ptr() % alignment:
                raise ValueError(f"{name} base pointer must be {alignment}-byte aligned; use the wrapper to normalize inputs")
        named_outputs = list(zip(("out", "max_logits", "lse", "lse_indexer"), outputs))
        no_overlap(named_outputs)
        for item in inputs:
            no_overlap([item, *named_outputs])
        stream = resolve_stream(q.device, current_stream)
        with torch.musa.device(q.device), torch.musa.stream(stream), torch.no_grad():
            record_stream(stream, q, kv, topk_idxs, attn_sink, topk_length, *outputs)
            if self._launcher is None:
                out.zero_()
                max_logits.fill_(-float("inf"))
                lse.fill_(float("inf"))
                if lse_indexer is not None:
                    lse_indexer.fill_(float("inf"))
            else:
                self._launcher(
                    self._compiled_kernel, q, kv, topk_idxs, scale, attn_sink, topk_length,
                    out, max_logits, lse, lse_indexer,
                )
        return outputs


_cache_of_sparse_attention_forward_objects: dict = {}


def sparse_attention_forward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    *,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    indexer_topk: int = 0,
    stream: Optional[torch.musa.Stream] = None,
) -> TupleDict:
    """Return ``out``, ``max_logits``, ``lse`` and ``lse_indexer`` (None when ``indexer_topk == 0``)."""
    scale = validate_inputs(q, kv, topk_idxs, attn_sink, topk_length, softmax_scale, empty=True)
    if int(indexer_topk) > topk_idxs.shape[1]:
        raise ValueError("indexer_topk must not exceed logical K")
    launch_stream = resolve_stream(q.device, stream)
    with torch.musa.device(q.device), torch.musa.stream(launch_stream), torch.no_grad():
        record_stream(launch_stream, q, kv, topk_idxs, attn_sink, topk_length)
        q, kv = contiguous(q, 16), contiguous(kv, 16)
        topk_idxs = contiguous(topk_idxs, 32)
        if topk_idxs.shape[1] % 64:
            topk_idxs = torch.nn.functional.pad(topk_idxs, (0, 64 - topk_idxs.shape[1] % 64), value=-1)
        attn_sink, topk_length = contiguous(attn_sink, 4), contiguous(topk_length, 4)
        tensors = (q, kv, topk_idxs, attn_sink, topk_length)
        key = (
            tuple(None if t is None else (t.device, t.dtype, tuple(t.shape), tuple(t.stride())) for t in tensors),
            int(indexer_topk),
        )
        obj = _cache_of_sparse_attention_forward_objects.get(key)
        if obj is None:
            obj = SparseAttentionForward(
                sample_q=q, sample_kv=kv, sample_topk_idxs=topk_idxs,
                sample_attn_sink=attn_sink, sample_topk_length=topk_length, indexer_topk=indexer_topk,
            )
            obj.check_support()
            obj.compile()
            _cache_of_sparse_attention_forward_objects[key] = obj
        shape = tuple(q.shape[:2])
        out = torch.empty((*shape, 512), dtype=q.dtype, device=q.device)
        max_logits = torch.empty(shape, dtype=torch.float32, device=q.device)
        lse = torch.empty_like(max_logits)
        lse_indexer = torch.empty_like(max_logits) if indexer_topk else None
        out, max_logits, lse, lse_indexer = obj.execute(
            q, kv, topk_idxs, attn_sink=attn_sink, topk_length=topk_length, softmax_scale=scale,
            out=out, max_logits=max_logits, lse=lse, lse_indexer=lse_indexer, current_stream=launch_stream,
        )
    return TupleDict(out=out, max_logits=max_logits, lse=lse, lse_indexer=lse_indexer)


__all__ = ["SparseAttentionForward", "sparse_attention_forward_wrapper"]
