"""Experimental SDPA op compatibility wrapper."""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in frontend tests.
    torch = None

_fprop_cache = {}
_bprop_cache = {}


def _shape_of(tensor):
    return tuple(getattr(tensor, "shape", ()))


def _dtype_of(tensor):
    return getattr(tensor, "dtype", None)


def _device_of(tensor):
    return getattr(tensor, "device", None)


def _normalize_seq_len(name: str, tensor, batch: int, limit: int, device):
    if tensor is None:
        return None
    if not hasattr(tensor, "reshape"):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as query")
    if tensor.numel() != batch:
        raise ValueError(f"{name} must contain one length per batch item")
    lengths = tensor.reshape(batch).to(dtype=torch.long)
    if torch.any(lengths < 0) or torch.any(lengths > int(limit)):
        raise ValueError(f"{name} values must be in [0, {int(limit)}]")
    return lengths


def _normalize_cumulative_seq_len(name: str, tensor, limit: int, device):
    if tensor is None:
        return None
    if not hasattr(tensor, "reshape"):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same device as query")
    if tensor.numel() < 2:
        raise ValueError(f"{name} must contain at least two offsets")
    offsets = tensor.reshape(-1).to(dtype=torch.long)
    if int(offsets[0]) != 0:
        raise ValueError(f"{name} must start at 0")
    if torch.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"{name} must be non-decreasing")
    if int(offsets[-1]) > int(limit):
        raise ValueError(f"{name} values must be in [0, {int(limit)}]")
    return offsets


def _combine_sdpa_mask(
    query,
    key,
    attn_mask,
    is_causal,
    diagonal_alignment,
    left_bound,
    right_bound,
    seq_len_q,
    seq_len_kv,
):
    mask = attn_mask
    bool_mask = None
    batch = query.shape[0]
    seq_q = query.shape[2]
    seq_k = key.shape[2]

    if is_causal and right_bound < 0:
        right_bound = 0
    if diagonal_alignment not in (0, 1):
        raise ValueError(
            "SDPA diagonal_alignment must be TOP_LEFT(0) or BOTTOM_RIGHT(1)"
        )

    if right_bound >= 0 or left_bound >= 0:
        q_index = torch.arange(seq_q, device=query.device).reshape(1, 1, seq_q, 1)
        k_index = torch.arange(seq_k, device=query.device).reshape(1, 1, 1, seq_k)
        if diagonal_alignment == 0:
            diagonal_offset = 0
        elif seq_len_q is not None and seq_len_kv is not None:
            diagonal_offset = (seq_len_kv - seq_len_q).reshape(batch, 1, 1, 1)
        else:
            diagonal_offset = seq_k - seq_q
        bool_mask = torch.ones(
            (1, 1, seq_q, seq_k), device=query.device, dtype=torch.bool
        )
        if right_bound >= 0:
            bool_mask = bool_mask & (
                k_index <= q_index + diagonal_offset + int(right_bound)
            )
        if left_bound >= 0:
            bool_mask = bool_mask & (
                k_index > q_index + diagonal_offset - int(left_bound)
            )

    query_valid = None
    if seq_len_q is not None:
        q_index = torch.arange(seq_q, device=query.device).reshape(1, seq_q)
        query_valid = q_index < seq_len_q.reshape(batch, 1)
    if seq_len_kv is not None:
        k_index = torch.arange(seq_k, device=query.device).reshape(1, 1, 1, seq_k)
        key_valid = k_index < seq_len_kv.reshape(batch, 1, 1, 1)
        bool_mask = key_valid if bool_mask is None else bool_mask & key_valid

    if bool_mask is None:
        return mask, is_causal, query_valid

    if mask is None:
        return bool_mask, False, query_valid
    if mask.dtype is torch.bool:
        return mask & bool_mask, False, query_valid

    padding_mask = torch.zeros(
        bool_mask.shape, device=query.device, dtype=query.dtype
    ).masked_fill(~bool_mask, float("-inf"))
    return mask + padding_mask, False, query_valid


def _validate_tensor(name: str, tensor):
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    if tensor is None or not hasattr(tensor, "shape"):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != 4:
        raise ValueError(f"{name} must use BHSD rank-4 layout")
    if tensor.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(f"{name} dtype must be float32, float16, or bfloat16")


def _cache_key(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    seq_len_q,
    seq_len_kv,
    cumulative_seq_len_q,
    cumulative_seq_len_kv,
    diagonal_alignment,
    left_bound,
    right_bound,
    scale,
    is_causal,
    enable_gqa,
):
    return (
        _shape_of(query),
        _shape_of(key),
        _shape_of(value),
        _dtype_of(query),
        _dtype_of(key),
        _dtype_of(value),
        _device_of(query),
        _shape_of(attn_mask),
        _dtype_of(attn_mask),
        _device_of(attn_mask),
        float(dropout_p),
        _shape_of(seq_len_q),
        _dtype_of(seq_len_q),
        _device_of(seq_len_q),
        _shape_of(seq_len_kv),
        _dtype_of(seq_len_kv),
        _device_of(seq_len_kv),
        _shape_of(cumulative_seq_len_q),
        _dtype_of(cumulative_seq_len_q),
        _device_of(cumulative_seq_len_q),
        _shape_of(cumulative_seq_len_kv),
        _dtype_of(cumulative_seq_len_kv),
        _device_of(cumulative_seq_len_kv),
        diagonal_alignment,
        left_bound,
        right_bound,
        scale,
        is_causal,
        enable_gqa,
    )


def _scaled_dot_product_attention_cumulative(
    query,
    key,
    value,
    dropout_p,
    is_causal,
    attn_scale,
    enable_gqa,
    diagonal_alignment,
    left_bound,
    right_bound,
    cum_q,
    cum_kv,
):
    if query.shape[0] != 1 or key.shape[0] != 1 or value.shape[0] != 1:
        raise ValueError(
            "cumulative sequence length fallback expects packed tensors with "
            "batch dimension 1"
        )
    if cum_q.numel() != cum_kv.numel():
        raise ValueError(
            "cumulative_seq_len_q and cumulative_seq_len_kv must describe the "
            "same number of batch segments"
        )

    output = torch.zeros(
        (1, query.shape[1], query.shape[2], value.shape[-1]),
        dtype=query.dtype,
        device=query.device,
    )
    for batch_idx in range(cum_q.numel() - 1):
        q_start = int(cum_q[batch_idx])
        q_end = int(cum_q[batch_idx + 1])
        kv_start = int(cum_kv[batch_idx])
        kv_end = int(cum_kv[batch_idx + 1])
        if q_start == q_end:
            continue
        if kv_start == kv_end:
            raise ValueError("cumulative_seq_len_kv contains an empty segment")

        q_segment = query[:, :, q_start:q_end, :]
        k_segment = key[:, :, kv_start:kv_end, :]
        v_segment = value[:, :, kv_start:kv_end, :]
        if enable_gqa and q_segment.shape[1] != k_segment.shape[1]:
            repeat = q_segment.shape[1] // k_segment.shape[1]
            k_segment = k_segment.repeat_interleave(repeat, dim=1)
            v_segment = v_segment.repeat_interleave(repeat, dim=1)

        segment_mask, segment_causal, _query_valid = _combine_sdpa_mask(
            q_segment,
            k_segment,
            None,
            is_causal,
            diagonal_alignment,
            left_bound,
            right_bound,
            None,
            None,
        )
        output[:, :, q_start:q_end, :] = (
            torch.nn.functional.scaled_dot_product_attention(
                q_segment,
                k_segment,
                v_segment,
                attn_mask=segment_mask,
                dropout_p=dropout_p,
                is_causal=segment_causal,
                scale=attn_scale,
            )
        )
    return output


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask: Optional[object] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    *,
    diagonal_alignment: int = 0,
    left_bound: int = -1,
    right_bound: int = -1,
    seq_len_q: Optional[object] = None,
    seq_len_kv: Optional[object] = None,
    cumulative_seq_len_q: Optional[object] = None,
    cumulative_seq_len_kv: Optional[object] = None,
):
    _validate_tensor("query", query)
    _validate_tensor("key", key)
    _validate_tensor("value", value)
    if attn_mask is not None:
        raise NotImplementedError(
            "mudnn.experimental.ops.sdpa does not support attn_mask"
        )
    if float(dropout_p) != 0.0:
        raise NotImplementedError(
            "mudnn.experimental.ops.sdpa does not support dropout"
        )
    has_cumulative = (
        cumulative_seq_len_q is not None or cumulative_seq_len_kv is not None
    )
    if has_cumulative and (
        cumulative_seq_len_q is None or cumulative_seq_len_kv is None
    ):
        raise ValueError(
            "cumulative_seq_len_q and cumulative_seq_len_kv must be provided together"
        )
    if has_cumulative and (seq_len_q is not None or seq_len_kv is not None):
        raise NotImplementedError(
            "SDPA cumulative sequence length fallback does not combine with "
            "seq_len_q or seq_len_kv"
        )
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("query, key, and value must have the same dtype")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value batch dimensions must match")
    if key.shape[1] != value.shape[1]:
        raise ValueError("key and value head dimensions must match")
    if key.shape[2] != value.shape[2]:
        raise ValueError("key and value sequence dimensions must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    seq_len_q = _normalize_seq_len(
        "seq_len_q", seq_len_q, query.shape[0], query.shape[2], query.device
    )
    seq_len_kv = _normalize_seq_len(
        "seq_len_kv", seq_len_kv, query.shape[0], key.shape[2], query.device
    )
    cumulative_seq_len_q = _normalize_cumulative_seq_len(
        "cumulative_seq_len_q",
        cumulative_seq_len_q,
        query.shape[2],
        query.device,
    )
    cumulative_seq_len_kv = _normalize_cumulative_seq_len(
        "cumulative_seq_len_kv",
        cumulative_seq_len_kv,
        key.shape[2],
        query.device,
    )
    if not enable_gqa and query.shape[1] != key.shape[1]:
        raise ValueError(
            f"query has {query.shape[1]} heads but key has {key.shape[1]} heads. "
            "Set enable_gqa=True for grouped-query attention."
        )
    if enable_gqa and query.shape[1] != key.shape[1]:
        if query.shape[1] % key.shape[1] != 0:
            raise ValueError("query heads must be a multiple of key heads for GQA")
        repeat = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)

    attn_scale = scale if scale is not None else (1.0 / math.sqrt(query.shape[-1]))
    requested_is_causal = is_causal
    attn_mask, is_causal, query_valid = _combine_sdpa_mask(
        query,
        key,
        attn_mask,
        is_causal,
        int(diagonal_alignment),
        int(left_bound),
        int(right_bound),
        seq_len_q,
        seq_len_kv,
    )
    if query.shape[-1] != 256:
        _fprop_cache.setdefault(
            _cache_key(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                seq_len_q,
                seq_len_kv,
                cumulative_seq_len_q,
                cumulative_seq_len_kv,
                int(diagonal_alignment),
                int(left_bound),
                int(right_bound),
                attn_scale,
                requested_is_causal,
                enable_gqa,
            ),
            True,
        )
    if has_cumulative:
        return _scaled_dot_product_attention_cumulative(
            query,
            key,
            value,
            dropout_p,
            requested_is_causal,
            attn_scale,
            enable_gqa,
            int(diagonal_alignment),
            int(left_bound),
            int(right_bound),
            cumulative_seq_len_q,
            cumulative_seq_len_kv,
        )

    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=attn_scale,
    )
    if query_valid is not None:
        output = output.masked_fill(
            ~query_valid.reshape(query.shape[0], 1, query.shape[2], 1), 0
        )
    return output
