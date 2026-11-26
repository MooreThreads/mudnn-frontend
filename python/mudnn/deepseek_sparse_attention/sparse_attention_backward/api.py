# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper for DeepSeek Sparse Attention backward (QK=V=512) on MP31 and MP32.

It consumes the ``out`` and ``lse`` tensors produced by the DSA sparse-attention forward path.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from mudnn.api_base import APIBase, TupleDict
from ..utils.runtime import MP32_ARCH, contiguous, device_capability, no_overlap, output_buffer, record_stream, resolve_stream
from ..utils.sparse_attention import validate_inputs
from ._workspace import layout, views

# The MP32 kernel keeps per-tile validity masks in shared memory; both chips share this limit.
MAX_TOPK = 8192


def _is_mp32(device) -> bool:
    return device.type == "musa" and device_capability(device) == MP32_ARCH


class SparseAttentionBackward(APIBase):
    def __init__(
        self,
        sample_q: torch.Tensor,  # (total_S_q, H, D) BF16
        sample_kv: torch.Tensor,  # (total_S_kv, D) BF16 (K=V)
        sample_out: torch.Tensor,  # (total_S_q, H, D_v)
        sample_dout: torch.Tensor,  # (total_S_q, H, D_v)
        sample_lse: torch.Tensor,  # (total_S_q, H) FP32, KV-only LSE
        sample_attn_sink: torch.Tensor,  # (H,) FP32
        sample_topk_idxs: torch.Tensor,  # (total_S_q, topk_max) INT32
        sample_dq: Optional[torch.Tensor] = None,
        sample_dkv: Optional[torch.Tensor] = None,
        sample_topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        block_tile: int = 64,
        deterministic: bool = False,
        q_cluster_mode: str = "off",
    ):
        super().__init__()
        self._descs = {}
        for name, tensor in (
            ("q", sample_q),
            ("kv", sample_kv),
            ("out", sample_out),
            ("dout", sample_dout),
            ("lse", sample_lse),
            ("attn_sink", sample_attn_sink),
            ("topk_idxs", sample_topk_idxs),
            ("topk_length", sample_topk_length),
        ):
            desc = self._make_tensor_desc(tensor, name="sample_" + name)
            self._descs[name] = desc
            setattr(self, name + "_desc", desc)
        self._output_descs = (self._make_tensor_desc(sample_dq), self._make_tensor_desc(sample_dkv))
        self.softmax_scale = softmax_scale
        self.block_tile = int(block_tile)
        self.deterministic = bool(deterministic)
        self.q_cluster_mode = q_cluster_mode
        self._layout = None
        self._workspace_bytes = 0
        self._cast = None

    def check_support(self) -> bool:
        self._value_error_if(
            self.q_cluster_mode not in ("off", "adaptive_pair"),
            f"q_cluster_mode must be 'off' or 'adaptive_pair', got {self.q_cluster_mode!r}",
        )
        self._not_implemented_error_if(self.q_cluster_mode != "off", "MUSA q_cluster_mode='adaptive_pair' is not supported")
        self._not_implemented_error_if(self.deterministic, "MUSA deterministic backward is unsupported; FP32 atomics are used")
        self._not_implemented_error_if(self.block_tile != 64, "MUSA sparse attention requires block_tile=64")
        self._value_error_if(self.attn_sink_desc is None, "attn_sink must be an FP32 tensor; use -inf for absent sinks")
        validate_inputs(
            self.q_desc,
            self.kv_desc,
            self.topk_idxs_desc,
            self.attn_sink_desc,
            self.topk_length_desc,
            self.softmax_scale,
            backward=True,
        )
        sq, heads, _ = self.q_desc.shape
        for name, desc, shape, dtype in (
            ("out", self.out_desc, (sq, heads, 512), self.q_desc.dtype),
            ("dout", self.dout_desc, (sq, heads, 512), self.q_desc.dtype),
            ("lse", self.lse_desc, (sq, heads), torch.float32),
        ):
            self._value_error_if(
                desc is None or desc.shape != shape or desc.dtype != dtype or desc.device != self.q_desc.device,
                f"{name} must be {dtype} with shape {shape} on Q's device",
            )
        for name, desc in self._descs.items():
            self._not_implemented_error_if(
                desc is not None and not desc.is_contiguous(),
                f"SparseAttentionBackward addresses only contiguous {name}; use the wrapper for staging",
            )
        for name, desc, expected in zip(("dq", "dkv"), self._output_descs, (self.q_desc, self.kv_desc)):
            self._value_error_if(
                desc is not None
                and (desc.shape != expected.shape or desc.dtype != expected.dtype or desc.device != expected.device or not desc.is_contiguous()),
                f"sample_{name} shape, dtype, device, or layout mismatch",
            )
        topk = self.topk_idxs_desc.shape[1]
        self._not_implemented_error_if(
            (topk + 63) // 64 * 64 > MAX_TOPK, f"MUSA sparse attention backward supports topk_max <= {MAX_TOPK}, got {topk}"
        )
        self._layout, self._workspace_bytes = layout(sq, self.kv_desc.shape[0], heads, topk)
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._ensure_support_checked()
        heads = max(64, self.q_desc.shape[1])
        topk = (self.topk_idxs_desc.shape[1] + 63) // 64 * 64
        with torch.musa.device(self.q_desc.device):
            if _is_mp32(self.q_desc.device):
                from ._interface_mp32 import compile_mp32, compile_postprocess

                self._cast = compile_postprocess()
                self._compiled_kernel = compile_mp32(heads, topk)
            else:
                from ._interface_mp31 import compile_mp31

                self._cast = None
                self._compiled_kernel = compile_mp31(heads, topk)

    def scratch_workspace_bytes(self) -> int:
        self._ensure_support_checked()
        return self._workspace_bytes

    def execute(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        out: torch.Tensor,
        dout: torch.Tensor,
        lse: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        dq: torch.Tensor,
        dkv: torch.Tensor,
        topk_length: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        current_stream: Optional[torch.musa.Stream] = None,
        workspace: Optional[torch.Tensor] = None,
        *,
        d_sink: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for name, tensor in (("dq", dq), ("dkv", dkv), ("d_sink", d_sink)):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be preallocated; use sparse_attention_backward_wrapper for automatic output allocation")
        if self._compiled_kernel is None:
            raise RuntimeError("call compile() before execute()")
        tensors = dict(q=q, kv=kv, out=out, dout=dout, lse=lse, attn_sink=attn_sink, topk_idxs=topk_idxs, topk_length=topk_length)
        for name, tensor in tensors.items():
            desc = self._descs[name]
            if (tensor is None) != (desc is None):
                raise ValueError(f"{name} presence must match the compiled plan")
            if desc is not None and (
                tuple(tensor.shape) != desc.shape or tensor.dtype != desc.dtype or tensor.device != desc.device or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} shape, dtype, device, and layout must match the compiled plan")
        scale = self.softmax_scale if softmax_scale is None else softmax_scale
        scale = validate_inputs(q, kv, topk_idxs, attn_sink, topk_length, scale, backward=True)
        for name, tensor, shape, dtype in (
            ("dq", dq, q.shape, q.dtype),
            ("dkv", dkv, kv.shape, kv.dtype),
            ("d_sink", d_sink, attn_sink.shape, torch.float32),
        ):
            output_buffer(tensor, shape, dtype, q.device, name, required=True)
        if not isinstance(workspace, torch.Tensor) or workspace.ndim != 1:
            raise ValueError("workspace must be a preallocated 1-D uint8 tensor")
        output_buffer(workspace, workspace.shape, torch.uint8, q.device, "workspace", required=True)
        if workspace.numel() < self._workspace_bytes or workspace.data_ptr() % 256:
            raise ValueError("workspace is too small or not 256-byte aligned")
        no_overlap([*tensors.items(), ("dq", dq), ("dkv", dkv), ("d_sink", d_sink), ("workspace", workspace)])
        stream = resolve_stream(q.device, current_stream)
        with torch.musa.device(q.device), torch.musa.stream(stream), torch.no_grad():
            record_stream(stream, *tensors.values(), dq, dkv, d_sink, workspace)
            w = views(workspace, self._layout)
            heads, topk = q.shape[1], topk_idxs.shape[1]
            # Inputs that already match the padded kernel extents are read and written in
            # place; only padded heads or Top-K go through the workspace.
            src = dict(w)
            if heads == w["q"].shape[1]:
                src.update(q=q, out=out, dout=dout, lse=lse, sink=attn_sink, dq=dq, d_sink=d_sink)
            else:
                for name, tensor in (("q", q), ("out", out), ("dout", dout)):
                    w[name].zero_()
                    w[name][:, :heads].copy_(tensor)
                w["lse"].fill_(float("inf"))
                w["lse"][:, :heads].copy_(lse)
                w["sink"].fill_(-float("inf"))
                w["sink"][:heads].copy_(attn_sink)
            if topk == w["indices"].shape[1]:
                src["indices"] = topk_idxs
            else:
                w["indices"].fill_(-1)
                w["indices"][:, :topk].copy_(topk_idxs)
            if topk_length is None:
                w["lengths"].fill_(topk)
            else:
                w["lengths"].copy_(topk_length)
                w["lengths"].clamp_(0, topk)
            w["dkv"].zero_()
            if _is_mp32(q.device):
                from ._interface_mp32 import launch_mp32 as launch
            else:
                from ._interface_mp31 import launch_mp31 as launch
            launch(self._compiled_kernel, src, kv, scale)
            if src["dq"] is not dq:
                dq.copy_(w["dq"][:, :heads])
                d_sink.copy_(w["d_sink"][:heads])
            if self._cast is not None:
                self._cast(w["dkv"], dkv)
            else:
                dkv.copy_(w["dkv"])
        return dq, dkv, d_sink


_cache_of_SparseAttentionBackwardObjects: dict = {}


def sparse_attention_backward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    dq: Optional[torch.Tensor] = None,
    dkv: Optional[torch.Tensor] = None,
    block_tile: int = 64,
    deterministic: bool = False,
    stream: Optional[torch.musa.Stream] = None,
    workspace: Optional[torch.Tensor] = None,
    q_cluster_mode: str = "off",
) -> TupleDict:
    """Return ``{'dq', 'dkv', 'd_sink'}``; stages strided inputs and allocates omitted outputs."""
    if q_cluster_mode not in ("off", "adaptive_pair"):
        raise ValueError(f"q_cluster_mode must be 'off' or 'adaptive_pair', got {q_cluster_mode!r}")
    if attn_sink is None:
        raise ValueError("attn_sink must be an FP32 tensor; use -inf for absent sinks")
    validate_inputs(q, kv, topk_idxs, attn_sink, topk_length, softmax_scale, backward=True)
    launch_stream = resolve_stream(q.device, stream)
    with torch.musa.device(q.device), torch.musa.stream(launch_stream), torch.no_grad():
        inputs = (q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length)
        record_stream(launch_stream, *inputs, dq, dkv, workspace)
        q, kv, out, dout, lse, attn_sink, topk_idxs, topk_length = map(contiguous, inputs)
        key = (
            q.device,
            q.dtype,
            q.shape,
            kv.shape,
            out.shape,
            dout.shape,
            lse.shape,
            attn_sink.shape,
            topk_idxs.shape,
            topk_length is not None,
            int(block_tile),
            softmax_scale,
            bool(deterministic),
            q_cluster_mode,
        )
        obj = _cache_of_SparseAttentionBackwardObjects.get(key)
        if obj is None:
            obj = SparseAttentionBackward(
                sample_q=q,
                sample_kv=kv,
                sample_out=out,
                sample_dout=dout,
                sample_lse=lse,
                sample_attn_sink=attn_sink,
                sample_topk_idxs=topk_idxs,
                sample_topk_length=topk_length,
                softmax_scale=softmax_scale,
                block_tile=block_tile,
                deterministic=deterministic,
                q_cluster_mode=q_cluster_mode,
            )
            obj.check_support()
            obj.compile()
            _cache_of_SparseAttentionBackwardObjects[key] = obj
        dq_user, dkv_user = dq, dkv
        for name, tensor, reference in (("dq", dq, q), ("dkv", dkv, kv)):
            if tensor is not None and (tensor.shape != reference.shape or tensor.dtype != reference.dtype or tensor.device != reference.device):
                raise ValueError(f"{name} shape, dtype, or device mismatch")
        if dq is None or not dq.is_contiguous():
            dq = torch.empty_like(q)
        if dkv is None or not dkv.is_contiguous():
            dkv = torch.empty_like(kv)
        d_sink = torch.empty_like(attn_sink)
        if workspace is None:
            workspace = torch.empty(obj.scratch_workspace_bytes(), dtype=torch.uint8, device=q.device)
        dq_out, dkv_out, d_sink_out = obj.execute(
            q,
            kv,
            out,
            dout,
            lse,
            attn_sink,
            topk_idxs,
            dq,
            dkv,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            workspace=workspace,
            current_stream=launch_stream,
            d_sink=d_sink,
        )
        if dq_user is not None and dq_user is not dq_out:
            dq_user.copy_(dq_out)
            dq_out = dq_user
        if dkv_user is not None and dkv_user is not dkv_out:
            dkv_user.copy_(dkv_out)
            dkv_out = dkv_user
        record_stream(launch_stream, dq_out, dkv_out, d_sink_out)
    return TupleDict(dq=dq_out, dkv=dkv_out, d_sink=d_sink_out)


__all__ = ["SparseAttentionBackward", "sparse_attention_backward_wrapper"]
