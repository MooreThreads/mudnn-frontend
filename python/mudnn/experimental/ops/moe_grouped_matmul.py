"""Experimental MoE grouped matmul compatibility wrapper."""

from __future__ import annotations

from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in frontend tests.
    torch = None

_graph_cache = {}


def _shape_of(tensor):
    return tuple(getattr(tensor, "shape", ()))


def _stride_of(tensor):
    stride = getattr(tensor, "stride", None)
    if callable(stride):
        return tuple(stride())
    return tuple(stride) if stride is not None else ()


def _dtype_of(tensor):
    return getattr(tensor, "dtype", None)


def _device_of(tensor):
    return getattr(tensor, "device", None)


def _validate_tensor(name: str, tensor, rank: int, dtypes):
    if torch is None:
        raise RuntimeError("PyTorch is not available")
    if tensor is None or not hasattr(tensor, "shape"):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() != rank:
        raise ValueError(f"{name} must be rank-{rank}")
    if tensor.dtype not in dtypes:
        raise ValueError(f"{name} dtype is not supported")


def _cache_key(
    token,
    weight,
    first_token_offset,
    token_index,
    token_ks,
    mode,
    top_k,
):
    return (
        _shape_of(token),
        _stride_of(token),
        _dtype_of(token),
        _shape_of(weight),
        _stride_of(weight),
        _dtype_of(weight),
        _shape_of(first_token_offset),
        _stride_of(first_token_offset),
        _dtype_of(first_token_offset),
        (
            (
                _shape_of(token_index),
                _stride_of(token_index),
                _dtype_of(token_index),
            )
            if token_index is not None
            else None
        ),
        (
            (
                _shape_of(token_ks),
                _stride_of(token_ks),
                _dtype_of(token_ks),
            )
            if token_ks is not None
            else None
        ),
        mode,
        top_k,
        _device_of(token),
    )


def _flatten_offsets(first_token_offset, num_experts: int, total_tokens: int):
    flat = first_token_offset.detach().cpu().reshape(-1).tolist()
    if len(flat) < num_experts:
        raise ValueError("first_token_offset must provide one start per expert")
    offsets = [int(value) for value in flat[:num_experts]]
    if offsets != sorted(offsets):
        raise ValueError("first_token_offset must be sorted")
    if offsets and (offsets[0] < 0 or offsets[-1] > total_tokens):
        raise ValueError("first_token_offset values are out of token range")
    return offsets


def _validate_index_tensor(name: str, tensor):
    if tensor is None or not hasattr(tensor, "shape"):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() not in {2, 3}:
        raise ValueError(f"{name} must be rank-2 or rank-3")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} dtype is not supported")


def _flatten_token_index(token_index, total_tokens: int):
    flat = token_index.detach().cpu().reshape(-1).tolist()
    indices = [int(value) for value in flat]
    if any(index < 0 or index >= total_tokens for index in indices):
        raise ValueError("token_index values are out of token range")
    return indices


def moe_grouped_matmul(
    token,
    weight,
    first_token_offset,
    token_index: Optional[object] = None,
    token_ks: Optional[object] = None,
    mode: str = "none",
    top_k: int = 1,
):
    """Torch fallback for cuDNN experimental MoE grouped matmul.

    The implemented subset matches upstream's direct routed mode: tokens are
    already grouped by expert, or GATHER mode where ``token_index`` first
    creates that routed token stream. ``first_token_offset`` stores each
    expert's start in the routed stream; the stream may already include
    top-k-expanded tokens. SCATTER routing needs accumulation semantics that
    are intentionally left unsupported here.
    """
    mode = str(mode).lower()
    if mode not in {"none", "gather", "scatter"}:
        raise ValueError("Invalid mode")
    if mode == "gather" and token_index is None:
        raise ValueError("GATHER mode requires token_index")
    if mode == "scatter":
        if token_index is None or token_ks is None:
            raise ValueError("SCATTER mode requires both token_index and token_ks")
        raise NotImplementedError(
            "mudnn.experimental.ops.moe_grouped_matmul torch fallback supports "
            "only mode='none' and mode='gather'"
        )
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if mode == "none" and (token_index is not None or token_ks is not None):
        raise NotImplementedError(
            "mudnn.experimental.ops.moe_grouped_matmul torch fallback does not "
            "support token_index or token_ks"
        )
    if token_ks is not None:
        raise NotImplementedError(
            "mudnn.experimental.ops.moe_grouped_matmul torch fallback does not "
            "support token_ks"
        )

    float_dtypes = {torch.float32, torch.float16, torch.bfloat16}
    _validate_tensor("token", token, 3, float_dtypes)
    _validate_tensor("weight", weight, 3, float_dtypes)
    _validate_tensor("first_token_offset", first_token_offset, 3, {torch.int32})
    if token_index is not None:
        _validate_index_tensor("token_index", token_index)
    if token.shape[0] != 1:
        raise ValueError("token batch dimension must be 1")
    if token.dtype != weight.dtype:
        raise TypeError("token and weight must have the same dtype")
    if (
        token.device != weight.device
        or token.device != first_token_offset.device
        or (token_index is not None and token.device != token_index.device)
    ):
        raise ValueError(
            "token, weight, first_token_offset, and token_index must share a device"
        )
    _batch, total_tokens, k = tuple(int(dim) for dim in token.shape)
    num_experts, weight_k, n = tuple(int(dim) for dim in weight.shape)
    if k != weight_k:
        raise ValueError("token and weight K dimensions are incompatible")
    key = _cache_key(
        token,
        weight,
        first_token_offset,
        token_index,
        token_ks,
        mode,
        top_k,
    )
    if mode == "gather":
        indices = _flatten_token_index(token_index, total_tokens)
        token = token.index_select(
            1, torch.tensor(indices, dtype=torch.long, device=token.device)
        )
        total_tokens = len(indices)

    _graph_cache.setdefault(key, True)

    offsets = _flatten_offsets(first_token_offset, num_experts, total_tokens)
    output = torch.zeros((1, total_tokens, n), dtype=token.dtype, device=token.device)
    for expert, start in enumerate(offsets):
        end = offsets[expert + 1] if expert + 1 < num_experts else total_tokens
        if start < end:
            output[0, start:end] = (
                token[0, start:end].to(torch.float32) @ weight[expert].to(torch.float32)
            ).to(token.dtype)
    return output
