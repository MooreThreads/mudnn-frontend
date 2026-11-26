"""Optional SM100 SDPA API compatibility names.

The real cuDNN frontend package backs these APIs with SM100 CUTE kernels.  The
muDNN compatibility package provides a narrow torch fallback for ordinary
FLOAT/HALF/BFLOAT16 tensors so upstream API tests can exercise the same tensor
layout and cache paths on hosts without those CUDA kernels.
"""

from __future__ import annotations

import math

from mudnn.api_base import APIBase, TupleDict


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - upstream tests require torch
        raise NotImplementedError("mudnn.sdpa fallback requires torch") from exc
    return torch


def _validate_common(
    q_tensor,
    k_tensor,
    v_tensor,
    *,
    cum_seqlen_q_tensor=None,
    cum_seqlen_k_tensor=None,
):
    torch = _torch()
    if not all(torch.is_tensor(tensor) for tensor in (q_tensor, k_tensor, v_tensor)):
        raise ValueError("q_tensor, k_tensor, and v_tensor must be torch tensors")
    if q_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"unsupported q dtype {q_tensor.dtype}")
    if k_tensor.dtype != q_tensor.dtype or v_tensor.dtype != q_tensor.dtype:
        raise ValueError("q_tensor, k_tensor, and v_tensor must have matching dtype")
    if cum_seqlen_q_tensor is None and cum_seqlen_k_tensor is None:
        if q_tensor.ndim != 4 or k_tensor.ndim != 4 or v_tensor.ndim != 4:
            raise ValueError("B,H,S,D SDPA tensors must be rank-4")
        b, h_q, s_q, d = q_tensor.shape
        b_k, h_k, s_k, d_k = k_tensor.shape
        b_v, h_v, s_v, d_v = v_tensor.shape
        if b != b_k or b != b_v or h_k != h_v or s_k != s_v or d != d_k or d != d_v:
            raise ValueError("B,H,S,D SDPA tensor dimensions are incompatible")
        if h_k <= 0 or h_q % h_k != 0:
            raise ValueError("query heads must be divisible by key/value heads")
        return "bhsd", b, s_q, s_k, h_q, h_k, d
    if cum_seqlen_q_tensor is None or cum_seqlen_k_tensor is None:
        raise ValueError("cum_seqlen_q_tensor and cum_seqlen_k_tensor must be paired")
    if q_tensor.ndim != 3 or k_tensor.ndim != 3 or v_tensor.ndim != 3:
        raise ValueError("T,H,D SDPA tensors must be rank-3")
    total_q, h_q, d = q_tensor.shape
    total_k, h_k, d_k = k_tensor.shape
    total_v, h_v, d_v = v_tensor.shape
    if total_k != total_v or h_k != h_v or d != d_k or d != d_v:
        raise ValueError("T,H,D SDPA tensor dimensions are incompatible")
    if h_k <= 0 or h_q % h_k != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if cum_seqlen_q_tensor.ndim != 1 or cum_seqlen_k_tensor.ndim != 1:
        raise ValueError("cumulative sequence tensors must be rank-1")
    if (
        cum_seqlen_q_tensor.dtype != torch.int32
        or cum_seqlen_k_tensor.dtype != torch.int32
    ):
        raise ValueError("cumulative sequence tensors must be int32")
    if cum_seqlen_q_tensor.shape != cum_seqlen_k_tensor.shape:
        raise ValueError("cumulative sequence tensors must have the same shape")
    batch = int(cum_seqlen_q_tensor.numel()) - 1
    if batch <= 0:
        raise ValueError("cumulative sequence tensors must contain at least 2 entries")
    return "thd", batch, total_q, total_k, h_q, h_k, d


def _as_reference_layout(q_tensor, k_tensor, v_tensor, layout: str, h_k: int):
    if layout == "bhsd":
        b, h_q, s_q, d = q_tensor.shape
        h_r = h_q // h_k
        q = q_tensor.reshape(b, h_k, h_r, s_q, d).permute(0, 3, 1, 2, 4)
        k = k_tensor.permute(0, 2, 1, 3).unsqueeze(3)
        v = v_tensor.permute(0, 2, 1, 3).unsqueeze(3)
        return q.contiguous(), k.contiguous(), v.contiguous()
    total_q, h_q, d = q_tensor.shape
    h_r = h_q // h_k
    q = q_tensor.reshape(total_q, h_k, h_r, d).unsqueeze(0)
    k = k_tensor.unsqueeze(0).unsqueeze(3)
    v = v_tensor.unsqueeze(0).unsqueeze(3)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _apply_window_mask(scores, *, is_causal: bool, window_size):
    torch = _torch()
    s_q, s_k = scores.shape
    left, right = window_size
    if is_causal:
        right = 0
    if left < 0 and right < 0:
        return scores
    q_coords = torch.arange(s_q, device=scores.device).view(-1, 1)
    k_coords = torch.arange(s_k, device=scores.device).view(1, -1)
    if left < 0:
        mask = k_coords > q_coords + s_k - s_q + right
    else:
        mask = (k_coords > q_coords + s_k - s_q + right) | (
            k_coords < q_coords + s_k - s_q - left
        )
    return scores.masked_fill(mask, -torch.inf)


def _sdpa_forward_reference(
    q_tensor,
    k_tensor,
    v_tensor,
    *,
    cum_seqlen_q_tensor=None,
    cum_seqlen_k_tensor=None,
    is_causal=False,
    window_size=(-1, -1),
    scale_softmax=None,
    scale_output=1.0,
):
    torch = _torch()
    layout, batch, s_q_max, s_k_max, h_q, h_k, d = _validate_common(
        q_tensor,
        k_tensor,
        v_tensor,
        cum_seqlen_q_tensor=cum_seqlen_q_tensor,
        cum_seqlen_k_tensor=cum_seqlen_k_tensor,
    )
    q, k, v = _as_reference_layout(q_tensor, k_tensor, v_tensor, layout, h_k)
    h_r = h_q // h_k
    scale = (
        (1.0 / math.sqrt(d)) if scale_softmax in (None, 0.0) else float(scale_softmax)
    )
    o = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.zeros(
        (1 if layout == "thd" else batch, h_k, h_r, q.shape[1]),
        dtype=torch.float32,
        device=q_tensor.device,
    )

    for batch_idx in range(batch):
        q_offset = int(cum_seqlen_q_tensor[batch_idx].item()) if layout == "thd" else 0
        k_offset = int(cum_seqlen_k_tensor[batch_idx].item()) if layout == "thd" else 0
        s_q = (
            int(
                (
                    cum_seqlen_q_tensor[batch_idx + 1] - cum_seqlen_q_tensor[batch_idx]
                ).item()
            )
            if layout == "thd"
            else s_q_max
        )
        s_k = (
            int(
                (
                    cum_seqlen_k_tensor[batch_idx + 1] - cum_seqlen_k_tensor[batch_idx]
                ).item()
            )
            if layout == "thd"
            else s_k_max
        )
        physical_batch = 0 if layout == "thd" else batch_idx
        for h_k_idx in range(h_k):
            cur_k = k[physical_batch, k_offset : k_offset + s_k, h_k_idx, 0, :].to(
                torch.float32
            )
            cur_v = v[physical_batch, k_offset : k_offset + s_k, h_k_idx, 0, :].to(
                torch.float32
            )
            for h_r_idx in range(h_r):
                cur_q = q[
                    physical_batch, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :
                ].to(torch.float32)
                scores = torch.einsum("qd,kd->qk", cur_q, cur_k) * scale
                scores = _apply_window_mask(
                    scores, is_causal=bool(is_causal), window_size=window_size
                )
                cur_lse = torch.logsumexp(scores, dim=-1)
                weights = torch.softmax(scores, dim=-1)
                cur_o = torch.einsum("qk,kd->qd", weights, cur_v) * float(scale_output)
                o[physical_batch, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :] = (
                    cur_o
                )
                lse[physical_batch, h_k_idx, h_r_idx, q_offset : q_offset + s_q] = (
                    cur_lse
                )

    if layout == "bhsd":
        o = o.reshape(batch, s_q_max, h_q, d).permute(0, 2, 1, 3).contiguous()
        lse = lse.reshape(batch, h_q, s_q_max).contiguous()
    else:
        o = o.reshape(q_tensor.shape[0], h_q, d).contiguous()
        lse = lse.reshape(q_tensor.shape[0], h_q).contiguous()
    return o.to(q_tensor.dtype), lse


def _sdpa_backward_reference(
    q_tensor,
    k_tensor,
    v_tensor,
    o_tensor,
    do_tensor,
    lse_tensor,
    *,
    cum_seqlen_q_tensor=None,
    cum_seqlen_k_tensor=None,
    is_causal=False,
    window_size=(-1, -1),
    scale_softmax=None,
):
    torch = _torch()
    layout, batch, s_q_max, s_k_max, h_q, h_k, d = _validate_common(
        q_tensor,
        k_tensor,
        v_tensor,
        cum_seqlen_q_tensor=cum_seqlen_q_tensor,
        cum_seqlen_k_tensor=cum_seqlen_k_tensor,
    )
    if o_tensor.shape != q_tensor.shape or do_tensor.shape != q_tensor.shape:
        raise ValueError("o_tensor and do_tensor must match q_tensor shape")
    q, k, v = _as_reference_layout(q_tensor, k_tensor, v_tensor, layout, h_k)
    o, _, _ = _as_reference_layout(o_tensor, k_tensor, v_tensor, layout, h_k)
    do, _, _ = _as_reference_layout(do_tensor, k_tensor, v_tensor, layout, h_k)
    h_r = h_q // h_k
    scale = (
        (1.0 / math.sqrt(d)) if scale_softmax in (None, 0.0) else float(scale_softmax)
    )
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    lse_ref = (
        lse_tensor.reshape(batch, h_k, h_r, s_q_max)
        if layout == "bhsd"
        else lse_tensor.reshape(q_tensor.shape[0], h_q)
        .transpose(0, 1)
        .reshape(h_k, h_r, q_tensor.shape[0])
        .unsqueeze(0)
    ).to(torch.float32)

    for batch_idx in range(batch):
        q_offset = int(cum_seqlen_q_tensor[batch_idx].item()) if layout == "thd" else 0
        k_offset = int(cum_seqlen_k_tensor[batch_idx].item()) if layout == "thd" else 0
        s_q = (
            int(
                (
                    cum_seqlen_q_tensor[batch_idx + 1] - cum_seqlen_q_tensor[batch_idx]
                ).item()
            )
            if layout == "thd"
            else s_q_max
        )
        s_k = (
            int(
                (
                    cum_seqlen_k_tensor[batch_idx + 1] - cum_seqlen_k_tensor[batch_idx]
                ).item()
            )
            if layout == "thd"
            else s_k_max
        )
        physical_batch = 0 if layout == "thd" else batch_idx
        for h_k_idx in range(h_k):
            cur_k = k[physical_batch, k_offset : k_offset + s_k, h_k_idx, 0, :].to(
                torch.float32
            )
            cur_v = v[physical_batch, k_offset : k_offset + s_k, h_k_idx, 0, :].to(
                torch.float32
            )
            for h_r_idx in range(h_r):
                cur_q = q[
                    physical_batch, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :
                ].to(torch.float32)
                cur_o = o[
                    physical_batch, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :
                ].to(torch.float32)
                cur_do = do[
                    physical_batch, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :
                ].to(torch.float32)
                cur_lse = lse_ref[
                    physical_batch, h_k_idx, h_r_idx, q_offset : q_offset + s_q
                ]
                scores = torch.einsum("qd,kd->qk", cur_q, cur_k) * scale
                scores = _apply_window_mask(
                    scores, is_causal=bool(is_causal), window_size=window_size
                )
                p = torch.exp(scores - cur_lse.reshape(-1, 1))
                cur_dv = torch.einsum("kq,qd->kd", p.transpose(1, 0), cur_do)
                dp = torch.einsum("qd,kd->qk", cur_do, cur_v)
                dsum = torch.einsum("qd,qd->q", cur_o, cur_do).reshape(-1, 1)
                ds = p * (dp - dsum) * scale
                cur_dk = torch.einsum("kq,qd->kd", ds.transpose(1, 0), cur_q)
                cur_dq = torch.einsum("qk,kd->qd", ds, cur_k)
                dq[physical_batch, q_offset : q_offset + s_q, h_k_idx, h_r_idx, :] = (
                    cur_dq
                )
                dk[physical_batch, k_offset : k_offset + s_k, h_k_idx, 0, :] += cur_dk
                dv[physical_batch, k_offset : k_offset + s_k, h_k_idx, 0, :] += cur_dv

    if layout == "bhsd":
        dq = dq.permute(0, 2, 3, 1, 4).reshape_as(q_tensor).contiguous()
        dk = dk.squeeze(3).permute(0, 2, 1, 3).reshape_as(k_tensor).contiguous()
        dv = dv.squeeze(3).permute(0, 2, 1, 3).reshape_as(v_tensor).contiguous()
    else:
        dq = dq.squeeze(0).reshape_as(q_tensor).contiguous()
        dk = dk.squeeze(0).squeeze(2).reshape_as(k_tensor).contiguous()
        dv = dv.squeeze(0).squeeze(2).reshape_as(v_tensor).contiguous()
    return dq.to(q_tensor.dtype), dk.to(k_tensor.dtype), dv.to(v_tensor.dtype)


class SdpafwdSm100D256(APIBase):
    def __init__(
        self,
        sample_q,
        sample_k,
        sample_v,
        sample_o,
        sample_lse,
        sample_cum_seqlen_q=None,
        sample_cum_seqlen_k=None,
        max_s_q=None,
        max_s_k=None,
        qk_acc_dtype=None,
        pv_acc_dtype=None,
        mma_tiler_mn=(128, 128),
        is_causal=False,
        window_size=(-1, -1),
        scale_softmax=None,
        scale_output=1.0,
    ):
        super().__init__()
        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_v = sample_v
        self.sample_o = sample_o
        self.sample_lse = sample_lse
        self.sample_cum_seqlen_q = sample_cum_seqlen_q
        self.sample_cum_seqlen_k = sample_cum_seqlen_k
        self.max_s_q = max_s_q
        self.max_s_k = max_s_k
        self.qk_acc_dtype = qk_acc_dtype
        self.pv_acc_dtype = pv_acc_dtype
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.is_causal = bool(is_causal)
        self.window_size = tuple(window_size)
        self.scale_softmax = scale_softmax
        self.scale_output = float(scale_output)
        self._compiled_kernel = None

    def check_support(self) -> bool:
        _validate_common(
            self.sample_q,
            self.sample_k,
            self.sample_v,
            cum_seqlen_q_tensor=self.sample_cum_seqlen_q,
            cum_seqlen_k_tensor=self.sample_cum_seqlen_k,
        )
        if self.sample_o.shape != self.sample_q.shape:
            raise ValueError("sample_o must match sample_q shape")
        if self.mma_tiler_mn != (128, 128):
            raise ValueError(
                f"mma_tiler_mn must be (128, 128), got {self.mma_tiler_mn}"
            )
        self._is_supported = True
        return True

    def compile(self):
        self.check_support()
        self._compiled_kernel = True
        return self

    def execute(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        cum_seqlen_q_tensor=None,
        cum_seqlen_k_tensor=None,
        current_stream=None,
    ):
        del current_stream
        if self._compiled_kernel is None:
            self.compile()
        o, lse = _sdpa_forward_reference(
            q_tensor,
            k_tensor,
            v_tensor,
            cum_seqlen_q_tensor=cum_seqlen_q_tensor,
            cum_seqlen_k_tensor=cum_seqlen_k_tensor,
            is_causal=self.is_causal,
            window_size=self.window_size,
            scale_softmax=self.scale_softmax,
            scale_output=self.scale_output,
        )
        o_tensor.copy_(o)
        lse_tensor.copy_(lse)
        return None


class SdpabwdSm100D256(APIBase):
    def __init__(
        self,
        sample_q,
        sample_k,
        sample_v,
        sample_o,
        sample_do,
        sample_lse,
        sample_dq,
        sample_dk,
        sample_dv,
        sample_cum_seqlen_q=None,
        sample_cum_seqlen_k=None,
        max_s_q=None,
        max_s_k=None,
        acc_dtype=None,
        mma_tiler_mn=(128, 128),
        dkdv_mma_tiler_mn=(128, 64),
        is_causal=False,
        window_size=(-1, -1),
        scale_softmax=None,
    ):
        super().__init__()
        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_v = sample_v
        self.sample_o = sample_o
        self.sample_do = sample_do
        self.sample_lse = sample_lse
        self.sample_dq = sample_dq
        self.sample_dk = sample_dk
        self.sample_dv = sample_dv
        self.sample_cum_seqlen_q = sample_cum_seqlen_q
        self.sample_cum_seqlen_k = sample_cum_seqlen_k
        self.max_s_q = max_s_q
        self.max_s_k = max_s_k
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = tuple(mma_tiler_mn)
        self.dkdv_mma_tiler_mn = tuple(dkdv_mma_tiler_mn)
        self.is_causal = bool(is_causal)
        self.window_size = tuple(window_size)
        self.scale_softmax = scale_softmax
        self._compiled_kernel = None

    def check_support(self) -> bool:
        _validate_common(
            self.sample_q,
            self.sample_k,
            self.sample_v,
            cum_seqlen_q_tensor=self.sample_cum_seqlen_q,
            cum_seqlen_k_tensor=self.sample_cum_seqlen_k,
        )
        if (
            self.sample_o.shape != self.sample_q.shape
            or self.sample_do.shape != self.sample_q.shape
        ):
            raise ValueError("sample_o and sample_do must match sample_q shape")
        if self.sample_dq.shape != self.sample_q.shape:
            raise ValueError("sample_dq must match sample_q shape")
        if (
            self.sample_dk.shape != self.sample_k.shape
            or self.sample_dv.shape != self.sample_v.shape
        ):
            raise ValueError("sample_dk/sample_dv must match sample_k/sample_v shapes")
        if self.mma_tiler_mn != (128, 128):
            raise ValueError(
                f"mma_tiler_mn must be (128, 128), got {self.mma_tiler_mn}"
            )
        if self.dkdv_mma_tiler_mn[0] != 128:
            raise ValueError(
                f"dkdv_mma_tiler_mn first dim must be 128, got {self.dkdv_mma_tiler_mn}"
            )
        self._is_supported = True
        return True

    def compile(self):
        self.check_support()
        self._compiled_kernel = True
        return self

    def execute(
        self,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        do_tensor,
        lse_tensor,
        dq_tensor,
        dk_tensor,
        dv_tensor,
        cum_seqlen_q_tensor=None,
        cum_seqlen_k_tensor=None,
        current_stream=None,
    ):
        del current_stream
        if self._compiled_kernel is None:
            self.compile()
        dq, dk, dv = _sdpa_backward_reference(
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            do_tensor,
            lse_tensor,
            cum_seqlen_q_tensor=cum_seqlen_q_tensor,
            cum_seqlen_k_tensor=cum_seqlen_k_tensor,
            is_causal=self.is_causal,
            window_size=self.window_size,
            scale_softmax=self.scale_softmax,
        )
        dq_tensor.copy_(dq)
        dk_tensor.copy_(dk)
        dv_tensor.copy_(dv)
        return None


_cache_of_SdpafwdSm100D256Objects = {}
_cache_of_SdpabwdSm100D256Objects = {}


def sdpa_fwd_wrapper_sm100_d256(
    q_tensor,
    k_tensor,
    v_tensor,
    cum_seqlen_q_tensor=None,
    cum_seqlen_k_tensor=None,
    max_s_q=None,
    max_s_k=None,
    qk_acc_dtype=None,
    pv_acc_dtype=None,
    mma_tiler_mn=(128, 128),
    is_causal=False,
    window_size=(-1, -1),
    scale_softmax=None,
    scale_output=1.0,
    current_stream=None,
) -> TupleDict:
    del max_s_q, max_s_k, qk_acc_dtype, pv_acc_dtype, current_stream
    o_tensor, lse_tensor = _sdpa_forward_reference(
        q_tensor,
        k_tensor,
        v_tensor,
        cum_seqlen_q_tensor=cum_seqlen_q_tensor,
        cum_seqlen_k_tensor=cum_seqlen_k_tensor,
        is_causal=is_causal,
        window_size=tuple(window_size),
        scale_softmax=scale_softmax,
        scale_output=scale_output,
    )
    if tuple(mma_tiler_mn) != (128, 128):
        raise ValueError(f"mma_tiler_mn must be (128, 128), got {mma_tiler_mn}")
    return TupleDict(o_tensor=o_tensor, lse_tensor=lse_tensor)


def sdpa_bwd_wrapper_sm100_d256(
    q_tensor,
    k_tensor,
    v_tensor,
    o_tensor,
    do_tensor,
    lse_tensor,
    cum_seqlen_q_tensor=None,
    cum_seqlen_k_tensor=None,
    max_s_q=None,
    max_s_k=None,
    acc_dtype=None,
    mma_tiler_mn=(128, 128),
    dkdv_mma_tiler_mn=(128, 64),
    is_causal=False,
    window_size=(-1, -1),
    scale_softmax=None,
    current_stream=None,
) -> TupleDict:
    del max_s_q, max_s_k, acc_dtype, current_stream
    if tuple(mma_tiler_mn) != (128, 128):
        raise ValueError(f"mma_tiler_mn must be (128, 128), got {mma_tiler_mn}")
    if tuple(dkdv_mma_tiler_mn)[0] != 128:
        raise ValueError(
            f"dkdv_mma_tiler_mn first dim must be 128, got {dkdv_mma_tiler_mn}"
        )
    dq_tensor, dk_tensor, dv_tensor = _sdpa_backward_reference(
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        do_tensor,
        lse_tensor,
        cum_seqlen_q_tensor=cum_seqlen_q_tensor,
        cum_seqlen_k_tensor=cum_seqlen_k_tensor,
        is_causal=is_causal,
        window_size=tuple(window_size),
        scale_softmax=scale_softmax,
    )
    return TupleDict(dq_tensor=dq_tensor, dk_tensor=dk_tensor, dv_tensor=dv_tensor)


__all__ = [
    "SdpafwdSm100D256",
    "sdpa_fwd_wrapper_sm100_d256",
    "SdpabwdSm100D256",
    "sdpa_bwd_wrapper_sm100_d256",
]
