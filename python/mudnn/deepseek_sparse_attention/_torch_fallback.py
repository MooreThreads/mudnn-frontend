"""DeepSeek Sparse Attention compatibility helpers.

This module provides small torch fallbacks for DSA APIs whose semantics are
simple enough to express without native MUSA kernels. Larger sparse-attention
kernels remain explicitly unsupported until a backend implementation exists.
"""

try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("DSA torch fallbacks require torch") from exc
    return torch


def _validate_indexer_top_k(input_values, seq_lens, top_k: int, next_n: int):
    torch = _torch()
    if input_values is None or seq_lens is None:
        raise ValueError("IndexerTopK requires input_values and seq_lens")
    if len(input_values.shape) != 2:
        raise ValueError("IndexerTopK input_values must be rank-2")
    if len(seq_lens.shape) != 1:
        raise ValueError("IndexerTopK seq_lens must be rank-1")
    if seq_lens.dtype not in {torch.int32, torch.int64}:
        raise ValueError("IndexerTopK seq_lens must be INT32 or INT64")
    top_k = int(top_k)
    next_n = int(next_n)
    if top_k <= 0:
        raise ValueError("IndexerTopK top_k must be positive")
    if next_n <= 0:
        raise ValueError("IndexerTopK next_n must be positive")
    n_rows = int(input_values.shape[0])
    batch = int(seq_lens.shape[0])
    if batch <= 0 or n_rows % batch != 0:
        raise ValueError("IndexerTopK rows must be divisible by seq_lens batch")
    if n_rows // batch != next_n:
        raise ValueError("IndexerTopK next_n must match rows per batch")
    return top_k, next_n


def _indexer_top_k(input_values, seq_lens, top_k: int, next_n: int):
    torch = _torch()
    top_k, _ = _validate_indexer_top_k(input_values, seq_lens, top_k, next_n)
    n_rows, num_cols = (int(dim) for dim in input_values.shape)
    batch = int(seq_lens.shape[0])
    rows_per_batch = n_rows // batch
    indices = torch.zeros(
        (n_rows, top_k), dtype=torch.int32, device=input_values.device
    )
    values = torch.zeros(
        (n_rows, top_k), dtype=input_values.dtype, device=input_values.device
    )
    for batch_idx in range(batch):
        length = int(seq_lens[batch_idx].item())
        length = max(0, min(length, num_cols))
        if length == 0:
            continue
        row_start = batch_idx * rows_per_batch
        row_end = row_start + rows_per_batch
        effective_k = min(top_k, length)
        top_values, top_indices = torch.topk(
            input_values[row_start:row_end, :length], effective_k, dim=1
        )
        indices[row_start:row_end, :effective_k] = top_indices.to(torch.int32)
        values[row_start:row_end, :effective_k] = top_values
    return indices, values


def _ratio_causal_mask(s_q: int, s_k: int, ratio: int, device, q_causal_offsets=None):
    """Ratio-causal validity mask with the upstream (cuDNN frontend) contract.

    Position ``(q, k)`` is valid when ``k < floor((offset + q + 1) / ratio)``,
    clamped to ``[0, s_k]``, where ``offset`` is ``q_causal_offsets[i]`` for
    batch ``i`` (the global uncompressed token index of that segment's local
    ``q[0]``) and ``0`` when no offsets are supplied.  The upstream kernels use
    exactly this top-left convention (see ``dsa_reference._ratio_causal_mask``
    and the mp_32 TileLang kernels' ``_causal_col_limit``); there is no implicit
    bottom-right alignment, so decoding or context-parallel segments must pass
    ``q_causal_offsets``.

    Returns ``(s_q, s_k)`` when ``q_causal_offsets`` is ``None`` and
    ``(batch, s_q, s_k)`` otherwise.
    """
    torch = _torch()
    ratio = int(ratio)
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    q = torch.arange(s_q, device=device, dtype=torch.int64).view(1, -1)
    k_pos = torch.arange(s_k, device=device, dtype=torch.int64).view(1, 1, -1)
    if q_causal_offsets is None:
        offset = torch.zeros((1, 1), dtype=torch.int64, device=device)
    else:
        offset = q_causal_offsets.to(device=device, dtype=torch.int64).reshape(-1, 1)
    col_limit = torch.div(offset + q + 1, ratio, rounding_mode="floor").clamp(0, s_k)
    valid = k_pos < col_limit.unsqueeze(-1)
    if q_causal_offsets is None:
        return valid.reshape(s_q, s_k)
    return valid


#: Backwards-compatible name for the pre-existing helper.  It now follows the
#: upstream top-left ratio-causal contract; callers that relied on the old
#: implicit ``s_k * ratio - s_q`` bottom-right alignment must pass
#: ``q_causal_offsets=s_k * ratio - s_q`` explicitly.
def _bottom_right_causal_mask(s_q: int, s_k: int, ratio: int, device):
    return _ratio_causal_mask(s_q, s_k, ratio, device)


def _is_finite(tensor):
    """Elementwise finiteness, without ``torch.isfinite``.

    MUSA's ``torch.isfinite`` on a float32 tensor runs at roughly 8 ns/element
    (13.1 ms for 2M elements) while the equivalent comparison sequence runs at
    ~0.02 ns/element; on a ``(4096, 1024)`` score tile it was the single largest
    cost in ``indexer_forward_top_k_wrapper``, larger than the sort and both
    GEMMs together.  ``NaN`` fails every comparison, so ``x == x`` rules it out,
    and the two ``inf`` bounds complete the definition.
    """
    torch = _torch()
    return (tensor == tensor) & torch.gt(tensor, float("-inf")) & torch.lt(
        tensor, float("inf")
    )


def _logsumexp_last(scores):
    """``logsumexp`` over the last axis with torch's CUDA empty-row semantics.

    ``logsumexp`` of an all-``-inf`` row is ``-inf`` (the log of an empty sum),
    which is what torch/CUDA returns and what the upstream kernels emit.  MUSA's
    ``logsumexp`` returns ``nan`` for that input instead, and the case is real in
    DSV4: with ``ratio``-causal masking the first row of every segment has
    ``floor((0 + 0 + 1) / ratio) == 0`` valid KV positions.  Normalising here
    keeps the fallback faithful on both backends rather than leaking a backend
    quirk into the returned denom.
    """
    torch = _torch()
    denom = torch.logsumexp(scores, dim=-1)
    empty = ~_is_finite(scores).any(dim=-1)
    if bool(empty.any()):
        denom = torch.where(empty, torch.full_like(denom, float("-inf")), denom)
    return denom


def _safe_softmax_last(scores):
    """``softmax`` over the last axis, returning exact zeros for all-``-inf`` rows.

    Same motivation as :func:`_logsumexp_last`: a row with no valid slot has no
    distribution, and the upstream contract is the exact value ``0`` for every
    slot rather than the ``nan`` a plain softmax produces.
    """
    torch = _torch()
    result = torch.softmax(scores, dim=-1)
    empty = ~_is_finite(scores).any(dim=-1, keepdim=True)
    return torch.where(empty, torch.zeros_like(result), result)


def _indexer_forward(q, k, w, ratio: int, sm_scale: float = 1.0, q_causal_offsets=None):
    if len(q.shape) != 4 or len(k.shape) != 4 or len(w.shape) != 3:
        raise ValueError("indexer_forward expects q/k rank-4 and w rank-3")
    b, s_q, h_q, d = (int(dim) for dim in q.shape)
    kb, s_k, h_kv, kd = (int(dim) for dim in k.shape)
    if kb != b or kd != d or tuple(w.shape) != (b, s_q, h_q):
        raise ValueError("indexer_forward tensor dimensions are incompatible")
    if h_kv <= 0 or h_q % h_kv != 0:
        raise ValueError("indexer_forward query heads must be divisible by kv heads")
    q_f = q.to(dtype=_torch().float32)
    k_f = k.to(dtype=_torch().float32)
    w_f = w.to(dtype=_torch().float32)
    k_expanded = k_f.repeat_interleave(h_q // h_kv, dim=2)
    scores = _torch().einsum("bqhd,bkhd->bqhk", q_f, k_expanded)
    scores = _torch().relu(scores)
    scores = float(sm_scale) * (scores * w_f.unsqueeze(-1)).sum(dim=2)
    valid = _ratio_causal_mask(s_q, s_k, int(ratio), q.device, q_causal_offsets)
    return scores.masked_fill(~valid, float("-inf"))


def _dense_indexer_score_recompute(
    q, k, w, ratio: int, sm_scale: float = 1.0, q_causal_offsets=None
):
    scores = _indexer_forward(
        q, k, w, ratio, sm_scale=float(sm_scale), q_causal_offsets=q_causal_offsets
    )
    valid = _is_finite(scores)
    denom = _logsumexp_last(scores)
    return scores.masked_fill(~valid, 0.0), denom


def _dense_attn_score_recompute(
    q, k, lse, softmax_scale: float, ratio: int, q_causal_offsets=None
):
    torch = _torch()
    if len(q.shape) != 4 or len(k.shape) != 4 or len(lse.shape) != 3:
        raise ValueError("dense_attn_score_recompute expects q/k rank-4 and lse rank-3")
    b, s_q, h_q, d = (int(dim) for dim in q.shape)
    kb, s_k, h_kv, kd = (int(dim) for dim in k.shape)
    if kb != b or kd != d or tuple(lse.shape) != (b, s_q, h_q):
        raise ValueError(
            "dense_attn_score_recompute tensor dimensions are incompatible"
        )
    if h_kv <= 0 or h_q % h_kv != 0:
        raise ValueError("dense_attn_score_recompute heads are incompatible")
    k_expanded = k.to(torch.float32).repeat_interleave(h_q // h_kv, dim=2)
    qk = torch.einsum("bqhd,bkhd->bqhk", q.to(torch.float32), k_expanded)
    probabilities = torch.exp(
        qk * float(softmax_scale) - lse.to(torch.float32).unsqueeze(-1)
    )
    scores = probabilities.sum(dim=2)
    valid = _ratio_causal_mask(s_q, s_k, int(ratio), q.device, q_causal_offsets)
    scores = scores.masked_fill(~valid, 0.0)
    denom = scores.sum(dim=-1)
    return scores, denom


def _cu_seqlens_to_lengths(cu_seqlens, name: str):
    torch = _torch()
    if cu_seqlens is None:
        raise ValueError(f"{name} is required")
    if len(cu_seqlens.shape) != 1:
        raise ValueError(f"{name} must be rank-1")
    if cu_seqlens.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"{name} must be INT32 or INT64")
    if int(cu_seqlens.numel()) <= 0:
        raise ValueError(f"{name} must not be empty")
    values = [int(v) for v in cu_seqlens.detach().cpu().tolist()]
    if values[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if any(values[i] > values[i + 1] for i in range(len(values) - 1)):
        raise ValueError(f"{name} must be nondecreasing")
    return [values[i + 1] - values[i] for i in range(len(values) - 1)], values


def _validate_thd_dense_score_inputs(q, k, aux, cu_seqlens_q, cu_seqlens_k):
    if len(q.shape) != 3 or len(k.shape) != 3 or len(aux.shape) != 2:
        raise ValueError("THD dense score expects q/k rank-3 and aux rank-2")
    q_lengths, q_offsets = _cu_seqlens_to_lengths(cu_seqlens_q, "cu_seqlens_q")
    k_lengths, k_offsets = _cu_seqlens_to_lengths(cu_seqlens_k, "cu_seqlens_k")
    if len(q_lengths) != len(k_lengths):
        raise ValueError("cu_seqlens_q and cu_seqlens_k batch sizes differ")
    total_q, h_q, d = (int(dim) for dim in q.shape)
    total_k, h_kv, kd = (int(dim) for dim in k.shape)
    if q_offsets[-1] != total_q or k_offsets[-1] != total_k:
        raise ValueError("THD cumulative sequence lengths do not match q/k totals")
    if kd != d or tuple(aux.shape) != (total_q, h_q):
        raise ValueError("THD dense score tensor dimensions are incompatible")
    if h_kv <= 0 or h_q % h_kv != 0:
        raise ValueError("THD dense score heads are incompatible")
    return q_lengths, q_offsets, k_lengths, k_offsets


def _validate_thd_dense_score_shapes(q, k, aux, out, denom):
    if len(q.shape) != 3 or len(k.shape) != 3 or len(aux.shape) != 2:
        raise ValueError("THD dense score expects q/k rank-3 and aux rank-2")
    total_q, h_q, d = (int(dim) for dim in q.shape)
    _, h_kv, kd = (int(dim) for dim in k.shape)
    if kd != d or tuple(aux.shape) != (total_q, h_q):
        raise ValueError("THD dense score tensor dimensions are incompatible")
    if h_kv <= 0 or h_q % h_kv != 0:
        raise ValueError("THD dense score heads are incompatible")
    if len(out.shape) != 2 or int(out.shape[0]) != total_q:
        raise ValueError("THD dense score output shape is incompatible")
    if tuple(denom.shape) != (total_q,):
        raise ValueError("THD dense score denominator shape is incompatible")


def _max_seqlen(lengths, explicit):
    return int(explicit) if explicit is not None else (max(lengths) if lengths else 0)


def _thd_segment_offsets(q_causal_offsets, batch_idx: int):
    """One-element int32 view of the per-segment causal offset (``None`` passes through)."""
    if q_causal_offsets is None:
        return None
    torch = _torch()
    offsets = q_causal_offsets.reshape(-1)
    index = min(int(batch_idx), int(offsets.numel()) - 1)
    return offsets[index : index + 1].to(torch.int32)


def _dense_indexer_score_recompute_thd(
    q,
    k,
    w,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    max_seqlen_k=None,
    ratio: int,
    sm_scale: float = 1.0,
    q_causal_offsets=None,
):
    torch = _torch()
    q_lengths, q_offsets, k_lengths, k_offsets = _validate_thd_dense_score_inputs(
        q, k, w, cu_seqlens_q, cu_seqlens_k
    )
    max_k = _max_seqlen(k_lengths, max_seqlen_k)
    out = torch.zeros((int(q.shape[0]), max_k), dtype=torch.float32, device=q.device)
    denom = torch.empty((int(q.shape[0]),), dtype=torch.float32, device=q.device)
    for batch_idx, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
        q_start, q_end = q_offsets[batch_idx], q_offsets[batch_idx + 1]
        k_start, k_end = k_offsets[batch_idx], k_offsets[batch_idx + 1]
        if sk > max_k:
            raise ValueError("max_seqlen_k is smaller than a KV sequence length")
        if sq == 0:
            continue
        segment_out, segment_denom = _dense_indexer_score_recompute(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            w[q_start:q_end].unsqueeze(0),
            ratio,
            sm_scale=sm_scale,
            q_causal_offsets=_thd_segment_offsets(q_causal_offsets, batch_idx),
        )
        out[q_start:q_end, :sk] = segment_out[0]
        denom[q_start:q_end] = segment_denom[0]
    return out, denom


def _dense_attn_score_recompute_thd(
    q,
    k,
    lse,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    max_seqlen_k=None,
    softmax_scale: float,
    ratio: int,
    q_causal_offsets=None,
):
    torch = _torch()
    q_lengths, q_offsets, k_lengths, k_offsets = _validate_thd_dense_score_inputs(
        q, k, lse, cu_seqlens_q, cu_seqlens_k
    )
    max_k = _max_seqlen(k_lengths, max_seqlen_k)
    out = torch.zeros((int(q.shape[0]), max_k), dtype=torch.float32, device=q.device)
    denom = torch.empty((int(q.shape[0]),), dtype=torch.float32, device=q.device)
    for batch_idx, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
        q_start, q_end = q_offsets[batch_idx], q_offsets[batch_idx + 1]
        k_start, k_end = k_offsets[batch_idx], k_offsets[batch_idx + 1]
        if sk > max_k:
            raise ValueError("max_seqlen_k is smaller than a KV sequence length")
        if sq == 0:
            continue
        segment_out, segment_denom = _dense_attn_score_recompute(
            q[q_start:q_end].unsqueeze(0),
            k[k_start:k_end].unsqueeze(0),
            lse[q_start:q_end].unsqueeze(0),
            softmax_scale,
            ratio,
            q_causal_offsets=_thd_segment_offsets(q_causal_offsets, batch_idx),
        )
        out[q_start:q_end, :sk] = segment_out[0]
        denom[q_start:q_end] = segment_denom[0]
    return out, denom


def _local_topk_indices(topk_indices, seqlen_k: int, topk_indices_global: bool):
    if not topk_indices_global:
        return topk_indices
    torch = _torch()
    batch_offsets = (
        torch.arange(
            topk_indices.shape[0], device=topk_indices.device, dtype=topk_indices.dtype
        )
        .view(-1, 1, 1)
        .mul(int(seqlen_k))
    )
    return torch.where(topk_indices >= 0, topk_indices - batch_offsets, topk_indices)


def _sparse_indexer_score_recompute(
    q, k, w, topk_indices, topk_length=None, *, topk_indices_global=False
):
    torch = _torch()
    if len(q.shape) != 4 or len(k.shape) != 3 or len(w.shape) != 3:
        raise ValueError(
            "sparse_indexer_score_recompute expects q rank-4, k rank-3, w rank-3"
        )
    b, s_q, h_q, d = (int(dim) for dim in q.shape)
    kb, s_k, kd = (int(dim) for dim in k.shape)
    if kb != b or kd != d or tuple(w.shape) != (b, s_q, h_q):
        raise ValueError(
            "sparse_indexer_score_recompute tensor dimensions are incompatible"
        )
    topk_indices = _local_topk_indices(topk_indices, s_k, bool(topk_indices_global))
    topk = int(topk_indices.shape[-1])
    idx = topk_indices.clamp(min=0, max=s_k - 1).long()
    k_gather = torch.gather(
        k.to(torch.float32).unsqueeze(1).expand(b, s_q, s_k, d),
        2,
        idx.unsqueeze(-1).expand(b, s_q, topk, d),
    )
    scores = torch.einsum("bqhd,bqtd->bqht", q.to(torch.float32), k_gather)
    scores = torch.relu(scores)
    scores = (scores * w.to(torch.float32).unsqueeze(-1)).sum(dim=2)
    valid = topk_indices >= 0
    if topk_length is not None:
        positions = torch.arange(topk, device=q.device).view(1, 1, topk)
        valid = valid & (positions < topk_length.unsqueeze(-1))
    scores = scores.masked_fill(~valid, float("-inf"))
    return _safe_softmax_last(scores)


def _sparse_attn_score_recompute(
    q,
    k,
    lse,
    topk_indices,
    softmax_scale: float,
    topk_length=None,
    *,
    topk_indices_global=False,
):
    torch = _torch()
    if len(q.shape) != 4 or len(k.shape) != 3 or len(lse.shape) != 3:
        raise ValueError(
            "sparse_attn_score_recompute expects q rank-4, k rank-3, lse rank-3"
        )
    b, s_q, h_q, d = (int(dim) for dim in q.shape)
    kb, s_k, kd = (int(dim) for dim in k.shape)
    if kb != b or kd != d or tuple(lse.shape) != (b, s_q, h_q):
        raise ValueError(
            "sparse_attn_score_recompute tensor dimensions are incompatible"
        )
    topk_indices = _local_topk_indices(topk_indices, s_k, bool(topk_indices_global))
    topk = int(topk_indices.shape[-1])
    idx = topk_indices.clamp(min=0, max=s_k - 1).long()
    k_gather = torch.gather(
        k.to(torch.float32).unsqueeze(1).expand(b, s_q, s_k, d),
        2,
        idx.unsqueeze(-1).expand(b, s_q, topk, d),
    )
    qk = torch.einsum("bqhd,bqtd->bqht", q.to(torch.float32), k_gather)
    target = torch.exp(
        qk * float(softmax_scale) - lse.to(torch.float32).unsqueeze(-1)
    ).sum(dim=2)
    valid = topk_indices >= 0
    if topk_length is not None:
        positions = torch.arange(topk, device=q.device).view(1, 1, topk)
        valid = valid & (positions < topk_length.unsqueeze(-1))
    target = target.masked_fill(~valid, 0.0)
    denom = target.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return target / denom


def _indexer_predict_distribution(q, k, w, topk_indices, sm_scale: float):
    torch = _torch()
    if len(q.shape) != 4 or len(k.shape) != 3 or len(w.shape) != 3:
        raise ValueError("indexer_backward expects q rank-4, k rank-3, w rank-3")
    b, s_q, h, d = (int(dim) for dim in q.shape)
    kb, s_k, kd = (int(dim) for dim in k.shape)
    if kb != b or kd != d or tuple(w.shape) != (b, s_q, h):
        raise ValueError("indexer_backward tensor dimensions are incompatible")
    topk = int(topk_indices.shape[-1])
    idx = topk_indices.clamp(min=0).long()
    k_gather = torch.gather(
        k.unsqueeze(1).expand(b, s_q, s_k, d),
        2,
        idx.unsqueeze(-1).expand(b, s_q, topk, d),
    )
    scores_per_head = torch.einsum("bqhd,bqtd->bqht", q, k_gather)
    scores_per_head = torch.relu(scores_per_head)
    scores = (scores_per_head * (w * float(sm_scale)).unsqueeze(-1)).sum(dim=2)
    scores = scores.masked_fill(topk_indices < 0, float("-inf"))
    return _safe_softmax_last(scores)


def _indexer_backward(
    index_q,
    weights,
    index_k,
    attn_score,
    topk_indices,
    *,
    sm_scale: float,
    grad_scale: float,
):
    torch = _torch()
    q = index_q.detach().clone().requires_grad_(True)
    w = weights.detach().clone().requires_grad_(True)
    k = index_k.detach().clone().requires_grad_(True)
    predict = _indexer_predict_distribution(q, k, w, topk_indices, sm_scale)
    eps = torch.finfo(torch.float32).tiny
    loss = (
        -float(grad_scale)
        * (
            attn_score.to(torch.float32).clamp(min=eps)
            * torch.log(predict.to(torch.float32).clamp(min=eps))
        ).sum()
    )
    dq, dw, dk = torch.autograd.grad(loss, (q, w, k))
    return dq.to(index_q.dtype), dw.to(weights.dtype), dk.to(index_k.dtype)


def _dense_indexer_predict_distribution(q, k, w, sm_scale: float, ratio: int):
    torch = _torch()
    if len(q.shape) != 4 or len(k.shape) != 3 or len(w.shape) != 3:
        raise ValueError("dense_indexer_backward expects q rank-4, k rank-3, w rank-3")
    b, s_q, h, d = (int(dim) for dim in q.shape)
    kb, s_k, kd = (int(dim) for dim in k.shape)
    if kb != b or kd != d or tuple(w.shape) != (b, s_q, h):
        raise ValueError("dense_indexer_backward tensor dimensions are incompatible")
    scores_per_head = torch.einsum(
        "bqhd,bkd->bqhk", q.to(torch.float32), k.to(torch.float32)
    )
    scores_per_head = torch.relu(scores_per_head)
    scores = (scores_per_head * (w * float(sm_scale)).unsqueeze(-1)).sum(dim=2)
    valid = _bottom_right_causal_mask(s_q, s_k, int(ratio), q.device)
    scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
    return _safe_softmax_last(scores), scores


def _dense_indexer_backward(
    index_q,
    weights,
    index_k,
    attn_score,
    attn_l1norm,
    *,
    sm_scale: float,
    ratio: int,
    grad_scale: float,
):
    torch = _torch()
    q = index_q.detach().clone().requires_grad_(True)
    w = weights.detach().clone().requires_grad_(True)
    k = index_k.detach().clone().requires_grad_(True)
    predict, scores = _dense_indexer_predict_distribution(q, k, w, sm_scale, ratio)
    _, s_q, _, _ = index_q.shape
    _, s_k, _ = index_k.shape
    valid = _bottom_right_causal_mask(s_q, s_k, int(ratio), index_q.device)
    target = attn_score.to(torch.float32).masked_fill(~valid.unsqueeze(0), 0.0)
    target = target / attn_l1norm.to(torch.float32).unsqueeze(-1).clamp(min=1e-10)
    eps = torch.exp(torch.tensor(-100.0, device=index_q.device, dtype=torch.float32))
    target_eff = target.clamp(min=eps)
    predict = predict.to(torch.float32)
    log_clip_mask = (predict >= eps).to(torch.float32)
    g = -target_eff * log_clip_mask * float(grad_scale)
    grad_signal = (g - predict * g.sum(dim=-1, keepdim=True)).masked_fill(
        ~valid.unsqueeze(0), 0.0
    )
    dq, dw, dk = torch.autograd.grad(
        scores, (q, w, k), grad_outputs=grad_signal.to(scores.dtype)
    )
    return dq.to(index_q.dtype), dw.to(weights.dtype), dk.to(index_k.dtype)


def _dense_indexer_backward_thd(
    index_q,
    weights,
    index_k,
    attn_score,
    attn_l1norm,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_k=None,
    sm_scale: float,
    ratio: int,
    grad_scale: float,
):
    torch = _torch()
    q_lengths, q_offsets, k_lengths, k_offsets = _validate_thd_dense_score_inputs(
        index_q, index_k.unsqueeze(1), weights, cu_seqlens_q, cu_seqlens_k
    )
    max_k = _max_seqlen(k_lengths, max_seqlen_k)
    expected_score_shape = (int(index_q.shape[0]), max_k)
    if tuple(attn_score.shape) != expected_score_shape:
        raise ValueError(
            f"THD attn_score must have shape {expected_score_shape}, got {tuple(attn_score.shape)}"
        )
    if tuple(attn_l1norm.shape) != (int(index_q.shape[0]),):
        raise ValueError("THD attn_l1norm shape is incompatible")
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.zeros_like(index_k)
    for batch_idx, (sq, sk) in enumerate(zip(q_lengths, k_lengths)):
        q_start, q_end = q_offsets[batch_idx], q_offsets[batch_idx + 1]
        k_start, k_end = k_offsets[batch_idx], k_offsets[batch_idx + 1]
        if sk > max_k:
            raise ValueError("max_seqlen_k is smaller than a KV sequence length")
        if sq == 0:
            continue
        dq, dw, dk = _dense_indexer_backward(
            index_q[q_start:q_end].unsqueeze(0),
            weights[q_start:q_end].unsqueeze(0),
            index_k[k_start:k_end].unsqueeze(0),
            attn_score[q_start:q_end, :sk].unsqueeze(0),
            attn_l1norm[q_start:q_end].unsqueeze(0),
            sm_scale=sm_scale,
            ratio=ratio,
            grad_scale=grad_scale,
        )
        d_index_q[q_start:q_end] = dq[0]
        d_weights[q_start:q_end] = dw[0]
        d_index_k[k_start:k_end] += dk[0]
    return d_index_q, d_weights, d_index_k


class IndexerTopK:
    def __init__(
        self,
        sample_input_values=None,
        sample_seq_lens=None,
        top_k=None,
        next_n=1,
        return_val=True,
        **kwargs,
    ):
        if top_k is None:
            raise TypeError("IndexerTopK missing required argument: 'top_k'")
        self.sample_input_values = sample_input_values
        self.sample_seq_lens = sample_seq_lens
        self.top_k = int(top_k)
        self.next_n = int(next_n)
        self.return_val = bool(return_val)
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        _validate_indexer_top_k(
            self.sample_input_values,
            self.sample_seq_lens,
            self.top_k,
            self.next_n,
        )
        return True

    def compile(self):
        self.check_support()
        self._compiled = True
        return self

    def execute(self, input_values, seq_lens, current_stream=None, **kwargs):
        del current_stream, kwargs
        if not self._compiled:
            self.compile()
        indices, values = _indexer_top_k(
            input_values, seq_lens, self.top_k, self.next_n
        )
        return indices, values if self.return_val else None


def indexer_top_k_wrapper(
    input_values,
    seq_lens,
    top_k,
    next_n=1,
    return_val=True,
    num_copy_bits=0,
    stream=None,
):
    """Per-batch top-``top_k`` over the first ``seq_lens[b]`` columns of each row.

    ``num_copy_bits`` selects the upstream radix path's copy granularity (a
    performance knob only, no semantic effect) and is validated but unused
    here.  Returns ``TupleDict(indices=int32, values=...)``; ``values`` is
    ``None`` when ``return_val=False``.
    """
    del stream
    if int(num_copy_bits) < 0:
        raise ValueError("num_copy_bits must be non-negative")
    indices, values = _indexer_top_k(input_values, seq_lens, top_k, next_n)
    return TupleDict(indices=indices, values=values if return_val else None)


def indexer_forward_wrapper(
    q,
    k,
    w,
    ratio=1,
    qhead_per_kv_head=None,
    m_block_size=128,
    n_block_size=128,
    q_stage=2,
    kv_stage=4,
    sm_scale=1.0,
    out=None,
    stream=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    q_causal_offsets=None,
    precision="bf16",
    q_scale=None,
    k_scale=None,
    cu_seqlens_q_scale_padded=None,
    cu_seqlens_k_scale_padded=None,
    sf_vec_size=32,
    return_lse=False,
    lse_out=None,
):
    """Dense indexer scores ``sm_scale * sum_h relu(Q.K^T) * W``.

    Argument list matches upstream ``indexer_forward_wrapper``.  Two layouts:

    * BSHD -- ``q (B, S_q, H_q, D)``, ``k (B, S_k, H_kv, D)``, ``w (B, S_q, H_q)``,
      returns ``(B, S_q, S_k)`` fp32 with ``-inf`` outside the ratio-causal window.
    * THD  -- ``q (total_q, H_q, D)``, ``k (total_k, H_kv, D)``, ``w (total_q, H_q)``
      plus ``cu_seqlens_q``/``cu_seqlens_k``; returns
      ``(total_q, max_seqlen_k)`` fp32, one row per packed query row, padded
      with ``-inf`` past each segment's KV extent.

    ``m_block_size``/``n_block_size``/``q_stage``/``kv_stage`` are upstream kernel
    tiling knobs with no semantic effect, and are accepted and ignored here for
    call-site compatibility.  ``precision``/``q_scale``/``k_scale`` select the
    upstream MXFP8 path, which this reference implementation cannot reproduce,
    so an FP8 request raises instead of returning BF16 numbers.
    """
    del m_block_size, n_block_size, q_stage, kv_stage, max_seqlen_q, sf_vec_size, stream
    _reject_fp8_precision(
        precision,
        q_scale,
        k_scale,
        "indexer_forward_wrapper",
        cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded,
    )
    _resolve_qhead_per_kv_head(qhead_per_kv_head, q, k)
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        scores = _indexer_forward_thd(
            q,
            k,
            w,
            int(ratio),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            sm_scale=float(sm_scale),
            q_causal_offsets=q_causal_offsets,
        )
    else:
        scores = _indexer_forward(
            q,
            k,
            w,
            int(ratio),
            sm_scale=float(sm_scale),
            q_causal_offsets=q_causal_offsets,
        )
    if out is not None:
        if tuple(out.shape) != tuple(scores.shape) or out.dtype != scores.dtype:
            raise ValueError(
                f"out must be {scores.dtype} {tuple(scores.shape)}, got {out.dtype} {tuple(out.shape)}"
            )
        out.copy_(scores)
        scores = out
    if return_lse or lse_out is not None:
        lse = _logsumexp_last(scores)
        if lse_out is not None:
            lse_out.copy_(lse)
            lse = lse_out
        return TupleDict(scores=scores, lse=lse)
    return TupleDict(scores=scores)


# ---------------------------------------------------------------------------
# Compressed (candidate-buffer) indexer forward + Top-K
# ---------------------------------------------------------------------------


def _indexer_scores(q, k, w, ratio, q_causal_offsets, sm_scale):
    """Dense indexer scores in the layout-agnostic 4-D/4-D form used below."""
    return _indexer_forward(
        q, k, w, int(ratio), sm_scale=float(sm_scale), q_causal_offsets=q_causal_offsets
    )


def _indexer_forward_thd(
    q,
    k,
    w,
    ratio,
    cu_seqlens_q,
    cu_seqlens_k,
    *,
    max_seqlen_k=None,
    sm_scale=1.0,
    q_causal_offsets=None,
):
    """Packed-THD dense indexer scores, one row per packed query row.

    ``q``/``w`` are ``(total_q, H, D)``/``(total_q, H)`` and ``k`` is
    ``(total_k, H_kv, D)``.  Each segment is evaluated independently with its
    own ratio-causal offset, exactly like the upstream readme's per-segment
    fallback; the result is ``(total_q, max_seqlen_k)`` with ``-inf`` padding.
    """
    torch = _torch()
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("THD indexer forward requires both cu_seqlens_q and cu_seqlens_k")
    if q.dim() != 3 or k.dim() != 3 or w.dim() != 2:
        raise ValueError("THD indexer forward expects q/k rank-3 and w rank-2")
    if q.shape[0] != w.shape[0]:
        raise ValueError("THD indexer forward q/w row counts must match")
    qcu = cu_seqlens_q.to(torch.int64).tolist()
    kcu = cu_seqlens_k.to(torch.int64).tolist()
    if len(qcu) != len(kcu):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same length")
    if max_seqlen_k is None:
        max_seqlen_k = max(
            [int(kcu[i + 1]) - int(kcu[i]) for i in range(len(kcu) - 1)] + [0]
        )
    total_q = int(q.shape[0])
    scores = torch.full(
        (total_q, int(max_seqlen_k)), float("-inf"), dtype=torch.float32, device=q.device
    )
    for batch in range(len(qcu) - 1):
        q0, q1 = int(qcu[batch]), int(qcu[batch + 1])
        k0, k1 = int(kcu[batch]), int(kcu[batch + 1])
        if q1 <= q0 or k1 <= k0:
            continue
        offset = None
        if q_causal_offsets is not None:
            offset = q_causal_offsets.reshape(-1)[
                min(batch, q_causal_offsets.numel() - 1)
            : min(batch, q_causal_offsets.numel() - 1) + 1].to(torch.int32)
        segment = _indexer_scores(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            offset,
            sm_scale,
        )
        scores[q0:q1, : k1 - k0] = segment[0]
    return scores


def _compact_topk(scores, top_k: int):
    """Top-``top_k`` (indices, logits) per row of a dense ``(..., S_k)`` score tensor.

    Masked (``-inf``) positions are never selected: they sort last, so the
    resulting slot count is ``min(top_k, finite columns)`` and the remainder is
    padded with ``index = -1`` / ``logit = -inf``.

    Slot order follows the upstream contract, which leaves the order of equally
    scoring slots unspecified.  ``torch.sort(..., stable=True)`` is deliberately
    *not* used: on MUSA it takes a slow path costing a flat ~1.5 ms regardless of
    row width (a ``(4096, 32)`` tile: 1584 us stable vs 47 us plain), which was
    over half of this whole call.  The selected *set* is unaffected either way --
    every ``-inf`` slot is invalid whatever its position -- so only the arbitrary
    order of exact ties changes.
    """
    torch = _torch()
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    s_k = int(scores.shape[-1])
    width = min(top_k, s_k)
    sorted_logits, sorted_indices = torch.sort(scores, dim=-1, descending=True)
    logits = sorted_logits[..., :width]
    indices = sorted_indices[..., :width]
    valid = _is_finite(logits)
    indices = torch.where(valid, indices, torch.full_like(indices, -1)).to(torch.int32)
    logits = torch.where(valid, logits, torch.full_like(logits, float("-inf")))
    if width < top_k:
        pad_shape = (*logits.shape[:-1], top_k - width)
        indices = torch.cat([indices, torch.full(pad_shape, -1, dtype=torch.int32, device=scores.device)], dim=-1)
        logits = torch.cat([logits, torch.full(pad_shape, float("-inf"), dtype=logits.dtype, device=scores.device)], dim=-1)
    return indices, logits


def _compact_softmax(indices, logits):
    """Row softmax over the selected logits; all-invalid rows become exact zeros."""
    torch = _torch()
    valid = indices >= 0
    scores = logits.masked_fill(~valid, float("-inf"))
    softmax = torch.softmax(scores, dim=-1)
    softmax = torch.nan_to_num(softmax, nan=0.0)
    return torch.where(valid, softmax, torch.zeros_like(softmax))


def _check_out_buffer(name, buffer, shape, dtype, device):
    torch = _torch()
    if buffer is None:
        return
    if not isinstance(buffer, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(buffer.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(buffer.shape)}")
    if buffer.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {buffer.dtype}")
    if buffer.device != device:
        raise ValueError(f"{name} must be on {device}, got {buffer.device}")
    if not buffer.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def compress_topk_cand_buffer_size(
    bs: int,
    seqlen_q: int,
    seqlen_k: int,
    ratio: int,
    microbatch_rows: int = -1,
    return_lse: bool = False,
    q_causal_offsets=None,
) -> int:
    """Float32 elements a ``cand_buffer`` must hold for the compact Top-K path.

    Mirrors upstream ``compress_topk_cand_buffer_size`` exactly: the compact
    candidate layout for one batch is ``G(offset + s_q) - G(offset)`` entries with
    ``G(n) = ratio*Q*(Q-1)/2 + Q*(S+1)``, ``Q = n // ratio``, ``S = n % ratio``,
    so the buffer is ``bs * per_batch_floats`` (the largest micro-batch window
    when ``microbatch_rows`` splits the launch).

    The muDNN reference implementation does not materialize this scratch buffer
    (it keeps the dense score tile instead), but the sizing contract is
    implemented so caller code and workspace planners written against upstream
    keep working and allocate the same amount.
    """
    if seqlen_q > seqlen_k * ratio:
        raise ValueError(
            f"seqlen_q ({seqlen_q}) must be <= seqlen_k * ratio ({seqlen_k * ratio})"
        )
    if int(bs) <= 0:
        raise ValueError("bs must be positive")
    if q_causal_offsets is not None:
        if not isinstance(q_causal_offsets, _torch().Tensor):
            raise TypeError("q_causal_offsets must be a torch.Tensor")
        if q_causal_offsets.ndim != 1 or int(q_causal_offsets.numel()) != int(bs):
            raise ValueError(
                f"q_causal_offsets must be a 1D tensor of shape ({int(bs)},)"
            )
        return int(
            sum(
                _compress_per_batch_floats(
                    int(seqlen_q), int(ratio), int(offset)
                )
                for offset in q_causal_offsets.to("cpu", _torch().int64).tolist()
            )
        )
    if return_lse:
        if int(microbatch_rows) > 0:
            raise NotImplementedError(
                "microbatch_rows > 0 is not supported with LSE output "
                "(return_lse / lse_out); pass microbatch_rows=0"
            )
        return int(bs) * _compress_per_batch_floats(int(seqlen_q), int(ratio), 0)
    rows = int(microbatch_rows)
    if rows < 0:
        rows = _compress_microbatch_rows(int(seqlen_q), int(bs), int(ratio))
    if 0 < rows < int(seqlen_q):
        largest, start = 0, 0
        while start < int(seqlen_q):
            window = min(rows, int(seqlen_q) - start)
            largest = max(
                largest, _compress_per_batch_floats(window, int(ratio), start)
            )
            start += window
        return int(bs) * largest
    return int(bs) * _compress_per_batch_floats(int(seqlen_q), int(ratio), 0)


def _compress_G(n: int, ratio: int) -> int:
    """``sum_{m=0}^{n} floor(m / ratio)`` in closed form (upstream ``G``)."""
    quotient = n // ratio
    remainder = n - quotient * ratio
    return ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


def _compress_per_batch_floats(seqlen_q: int, ratio: int, q_causal_offset: int = 0) -> int:
    if q_causal_offset < 0:
        raise ValueError(f"q_causal_offset must be >= 0, got {q_causal_offset}")
    return _compress_G(q_causal_offset + seqlen_q, ratio) - _compress_G(
        q_causal_offset, ratio
    )


def _compress_microbatch_rows(seqlen_q: int, bs: int, ratio: int) -> int:
    """Upstream long-sequence auto micro-batch window (``0`` means single launch)."""
    if seqlen_q < 32768 or seqlen_q % ratio != 0:
        return 0
    if bs >= 8:
        rows = 2048
    elif bs >= 4:
        rows = 4096
    else:
        rows = 8192
    rows = (rows // ratio) * ratio
    if rows <= 0 or rows >= seqlen_q:
        return 0
    return rows


def compress_topk_cand_buffer_size_thd(
    cu_seqlens_q, cu_seqlens_k, ratio: int, q_causal_offsets=None
):
    """THD sizing helper: ``(cand_batch_offsets, total_floats)``.

    ``cand_batch_offsets`` is an int64 ``(bs + 1,)`` exclusive prefix sum of the
    per-batch compact slab sizes on ``cu_seqlens_q.device``; ``total_floats`` is
    the float32 element count the ``cand_buffer`` must hold.  Both match
    upstream ``compress_topk_cand_buffer_size_thd``, including the
    per-batch ``q_causal_offsets`` contract.
    """
    torch = _torch()
    if (
        cu_seqlens_q.ndim != 1
        or cu_seqlens_k.ndim != 1
        or cu_seqlens_q.numel() != cu_seqlens_k.numel()
        or cu_seqlens_q.numel() < 2
    ):
        raise ValueError("cu_seqlens_q/k must be 1D of equal length (bs+1) >= 2")
    bs = int(cu_seqlens_q.numel()) - 1
    if q_causal_offsets is not None:
        if (
            not getattr(q_causal_offsets, "is_musa", False)
            and not q_causal_offsets.is_cuda
        ):
            raise ValueError("q_causal_offsets must be a device (MUSA/CUDA) tensor")
        if (
            q_causal_offsets.dtype != torch.int32
            or q_causal_offsets.ndim != 1
            or q_causal_offsets.numel() != bs
            or q_causal_offsets.device != cu_seqlens_q.device
        ):
            raise ValueError(
                "q_causal_offsets must be a 1D int32 tensor of shape (bs,) "
                "on the same device as cu_seqlens_q"
            )
    cu_q = cu_seqlens_q.to(torch.int64)
    lengths = cu_q[1:] - cu_q[:-1]
    if q_causal_offsets is None:
        offsets_q = torch.zeros_like(lengths)
    else:
        offsets_q = q_causal_offsets.to(torch.int64)
    ratio = int(ratio)

    def _G(n):
        quotient = n // ratio
        remainder = n - quotient * ratio
        return ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)

    per_batch = _G(offsets_q + lengths) - _G(offsets_q)
    offsets = torch.zeros(lengths.numel() + 1, dtype=torch.int64, device=cu_q.device)
    torch.cumsum(per_batch, dim=0, out=offsets[1:])
    return offsets, int(offsets[-1].item())


def indexer_forward_top_k_wrapper(
    q,
    k,
    w,
    top_k,
    *,
    ratio=4,
    qhead_per_kv_head=None,
    m_block_size=128,
    n_block_size=128,
    q_stage=2,
    kv_stage=4,
    sm_scale=1.0,
    stream=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    q_causal_offsets=None,
    precision="bf16",
    q_scale=None,
    k_scale=None,
    cu_seqlens_q_scale_padded=None,
    cu_seqlens_k_scale_padded=None,
    sf_vec_size=32,
    return_lse=False,
    lse_out=None,
    return_softmax=None,
    softmax_out=None,
    microbatch_rows=-1,
    topk_indices_global=True,
    cand_buffer=None,
    out_indices=None,
    out_logits=None,
    cand_batch_offsets=None,
    deterministic=False,
    packed_window_plan=None,
):
    """Combined dense indexer score + Top-K selection (upstream compact API).

    Returns ``TupleDict(indices=int32, logits=fp32, softmax=fp32)`` plus
    ``lse`` when requested.  BSHD outputs are ``(B, S_q, top_k)`` and THD
    outputs ``(total_q, top_k)``.

    Contract notes (all inherited from the upstream wrapper):

    * ``indices`` are global KV ids by default (``b * S_k + local`` for BSHD,
      ``cu_seqlens_k[b] + local`` for THD); pass ``topk_indices_global=False``
      for local ids.
    * Invalid slots carry ``index = -1`` and ``logit = -inf``; the fused
      ``softmax`` of an all-invalid row is exactly ``0`` (never NaN).
    * ``out_indices``/``out_logits``/``softmax_out``/``lse_out`` are copied in
      place and returned, so callers that preallocate them observe aliasing.
    * ``cand_buffer``/``cand_batch_offsets``/``microbatch_rows`` are
      performance-only hints that select the upstream two-stage candidate
      layout.  This reference implementation computes the dense score tile and
      an exact Top-K, so it consumes no scratch buffer and does not window the
      launch; the documented results are identical either way.  Buffer arguments
      are still validated so a mis-sized allocation fails at the call site
      instead of being hidden.
    * ``packed_window_plan`` is an upstream performance hint for the packed
      GEMM and is accepted/ignored here; it never changes the selected set.
    """
    torch = _torch()
    _reject_fp8_precision(
        precision,
        q_scale,
        k_scale,
        "indexer_forward_top_k_wrapper",
        cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded,
    )
    # Upstream tiling / launch-granularity knobs: performance-only here.
    del (
        m_block_size,
        n_block_size,
        q_stage,
        kv_stage,
        sf_vec_size,
        stream,
        deterministic,
        packed_window_plan,
    )
    is_thd = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_thd:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError(
                "THD combined indexer+TopK requires both cu_seqlens_q and cu_seqlens_k"
            )
        if q.dim() != 3 or k.dim() != 3 or w.dim() != 2:
            raise ValueError("THD combined indexer+TopK expects q/k=(T,H,D), w=(T,H)")
    elif q.dim() != 4 or k.dim() != 4 or w.dim() != 3:
        raise ValueError(
            "BSHD combined indexer+TopK expects q=(B,S,H,D), k=(B,S,H_kv,D), w=(B,S,H)"
        )
    _resolve_qhead_per_kv_head(qhead_per_kv_head, q, k)
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    ratio = int(ratio)
    if cand_buffer is not None:
        if not isinstance(cand_buffer, torch.Tensor):
            raise TypeError("cand_buffer must be a torch.Tensor")
        if cand_buffer.dtype != torch.float32:
            raise ValueError("cand_buffer must be float32")
        if not cand_buffer.is_contiguous():
            raise ValueError("cand_buffer must be contiguous (it is written in place)")
        if cand_buffer.device != q.device:
            raise ValueError(
                f"cand_buffer device {cand_buffer.device} != input device {q.device}"
            )
        if is_thd and cand_batch_offsets is not None:
            if cand_buffer.numel() < int(cand_batch_offsets[-1].item()):
                raise ValueError(
                    "cand_buffer too small for cand_batch_offsets "
                    "(use compress_topk_cand_buffer_size_thd(...))"
                )
        elif not is_thd:
            required = compress_topk_cand_buffer_size(
                int(q.shape[0]),
                int(q.shape[1]),
                int(k.shape[1]),
                ratio,
                microbatch_rows=microbatch_rows,
                return_lse=bool(return_lse or lse_out is not None),
            )
            if cand_buffer.numel() < required:
                raise ValueError(
                    "cand_buffer too small: need >= "
                    f"{required} float32 elements "
                    "(use compress_topk_cand_buffer_size(...)), got "
                    f"{cand_buffer.numel()}"
                )

    if is_thd:
        total_q = int(q.shape[0])
        if max_seqlen_k is None:
            lengths = (cu_seqlens_k.to(torch.int64)[1:] - cu_seqlens_k.to(torch.int64)[:-1])
            max_seqlen_k = int(lengths.max().item()) if lengths.numel() else 0
        indices = torch.full(
            (total_q, top_k), -1, dtype=torch.int32, device=q.device
        )
        logits = torch.full(
            (total_q, top_k), float("-inf"), dtype=torch.float32, device=q.device
        )
        qcu = cu_seqlens_q.to(torch.int64).tolist()
        kcu = cu_seqlens_k.to(torch.int64).tolist()
        for batch in range(len(qcu) - 1):
            q0, q1 = int(qcu[batch]), int(qcu[batch + 1])
            k0, k1 = int(kcu[batch]), int(kcu[batch + 1])
            if q1 <= q0 or k1 <= k0:
                continue
            offset = None
            if q_causal_offsets is not None:
                start = min(batch, int(q_causal_offsets.numel()) - 1)
                offset = q_causal_offsets.reshape(-1)[start : start + 1].to(torch.int32)
            segment = _indexer_scores(
                q[q0:q1].unsqueeze(0),
                k[k0:k1].unsqueeze(0),
                w[q0:q1].unsqueeze(0),
                ratio,
                offset,
                sm_scale,
            )
            seg_idx, seg_logits = _compact_topk(segment, top_k)
            indices[q0:q1] = seg_idx[0]
            logits[q0:q1] = seg_logits[0]
            if topk_indices_global:
                indices[q0:q1] = torch.where(
                    indices[q0:q1] >= 0,
                    indices[q0:q1] + k0,
                    indices[q0:q1],
                )
    else:
        scores = _indexer_scores(q, k, w, ratio, q_causal_offsets, sm_scale)
        local_indices, logits = _compact_topk(scores, top_k)
        if topk_indices_global:
            batch_offsets = (
                torch.arange(
                    int(q.shape[0]), device=q.device, dtype=torch.int32
                )
                .view(-1, 1, 1)
                .mul(int(k.shape[1]))
            )
            indices = torch.where(
                local_indices >= 0, local_indices + batch_offsets, local_indices
            )
        else:
            indices = local_indices

    result = {}
    if out_indices is not None:
        _check_out_buffer("out_indices", out_indices, indices.shape, torch.int32, q.device)
        out_indices.copy_(indices)
        indices = out_indices
    result["indices"] = indices
    if out_logits is not None:
        _check_out_buffer("out_logits", out_logits, logits.shape, torch.float32, q.device)
        out_logits.copy_(logits)
        logits = out_logits
    result["logits"] = logits
    if return_softmax is not False or softmax_out is not None:
        softmax = _compact_softmax(indices, logits)
        if softmax_out is not None:
            _check_out_buffer(
                "softmax_out", softmax_out, softmax.shape, torch.float32, q.device
            )
            softmax_out.copy_(softmax)
            softmax = softmax_out
        result["softmax"] = softmax
    if return_lse or lse_out is not None:
        lse = torch.logsumexp(
            logits.masked_fill(indices < 0, float("-inf")), dim=-1
        )
        if lse_out is not None:
            _check_out_buffer("lse_out", lse_out, lse.shape, torch.float32, q.device)
            lse_out.copy_(lse)
            lse = lse_out
        result["lse"] = lse
    return TupleDict(**result)


def _reject_fp8_precision(precision, q_scale, k_scale, name, *extra_scales):
    """Refuse FP8/MXFP8 requests instead of silently returning BF16 numbers."""
    lowered = str(precision).lower()
    scales = (q_scale, k_scale) + tuple(extra_scales)
    if lowered not in ("bf16", "bfloat16", "fp32", "float32") or any(
        scale is not None for scale in scales
    ):
        raise NotImplementedError(
            f"{name}: FP8/MXFP8 quantization is not implemented in the muDNN "
            "torch fallback; pass precision='bf16' without scales"
        )


def _resolve_qhead_per_kv_head(qhead_per_kv_head, q, k):
    """Validate an explicit ``qhead_per_kv_head`` against ``q``/``k`` head counts."""
    if qhead_per_kv_head is None:
        return None
    heads_q = int(q.shape[-2])
    heads_kv = int(k.shape[-2] if k.dim() in (3, 4) else k.shape[-1])
    if int(qhead_per_kv_head) * heads_kv != heads_q:
        raise ValueError(
            f"qhead_per_kv_head={int(qhead_per_kv_head)} does not match "
            f"{heads_q} query heads over {heads_kv} kv heads"
        )
    return int(qhead_per_kv_head)


def dense_indexer_score_recompute_wrapper(
    q,
    k,
    weights,
    qhead_per_kv_head=None,
    out=None,
    denom_out=None,
    sm_scale=1.0,
    ratio=1,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    q_causal_offsets=None,
    precision="bf16",
    q_scale=None,
    k_scale=None,
    cu_seqlens_q_scale_padded=None,
    cu_seqlens_k_scale_padded=None,
    sf_vec_size=32,
    stream=None,
):
    """Dense indexer scores and their log-sum-exp over the ratio-causal window.

    Returns ``TupleDict(out=(B,S_q,S_k)|(T_q,max_k) fp32 with 0 on masked
    positions, denom=(B,S_q)|(T_q,) fp32 log-sum-exp)``.  ``sm_scale`` and
    ``q_causal_offsets`` follow the upstream contract; the offsets are applied
    per packed segment on the THD path.
    """
    del max_seqlen_q, sf_vec_size, stream
    _reject_fp8_precision(
        precision,
        q_scale,
        k_scale,
        "dense_indexer_score_recompute_wrapper",
        cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded,
    )
    _resolve_qhead_per_kv_head(qhead_per_kv_head, q, k)
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        result, denom = _dense_indexer_score_recompute_thd(
            q,
            k,
            weights,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            ratio=int(ratio),
            sm_scale=float(sm_scale),
            q_causal_offsets=q_causal_offsets,
        )
    else:
        result, denom = _dense_indexer_score_recompute(
            q,
            k,
            weights,
            int(ratio),
            sm_scale=float(sm_scale),
            q_causal_offsets=q_causal_offsets,
        )
    return TupleDict(
        out=_copy_result(out, result), denom=_copy_result(denom_out, denom)
    )


def dense_attn_score_recompute_wrapper(
    q,
    k,
    lse,
    softmax_scale,
    qhead_per_kv_head=None,
    out=None,
    denom_out=None,
    ratio=1,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    q_causal_offsets=None,
    precision="bf16",
    q_scale=None,
    k_scale=None,
    cu_seqlens_q_scale_padded=None,
    cu_seqlens_k_scale_padded=None,
    sf_vec_size=32,
    stream=None,
):
    """Dense attention probabilities summed over heads (``exp(qk*s - lse)``)."""
    del max_seqlen_q, sf_vec_size, stream
    _reject_fp8_precision(
        precision,
        q_scale,
        k_scale,
        "dense_attn_score_recompute_wrapper",
        cu_seqlens_q_scale_padded,
        cu_seqlens_k_scale_padded,
    )
    _resolve_qhead_per_kv_head(qhead_per_kv_head, q, k)
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        result, denom = _dense_attn_score_recompute_thd(
            q,
            k,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=float(softmax_scale),
            ratio=int(ratio),
            q_causal_offsets=q_causal_offsets,
        )
    else:
        result, denom = _dense_attn_score_recompute(
            q,
            k,
            lse,
            float(softmax_scale),
            int(ratio),
            q_causal_offsets=q_causal_offsets,
        )
    return TupleDict(
        out=_copy_result(out, result), denom=_copy_result(denom_out, denom)
    )


def sparse_indexer_score_recompute_wrapper(
    q_indexer,
    k_indexer,
    weights,
    topk_indices,
    qhead_per_kv_head=None,
    topk_length=None,
    topk_indices_global=False,
    out=None,
    stream=None,
):
    """Sparse indexer ``predict`` distribution over the selected Top-K slots.

    Returns ``TupleDict(predict=...)`` of shape ``(B, S_q, top_k)`` fp32; slots
    outside ``topk_length`` / carrying ``-1`` are excluded from the softmax.
    ``out`` is written in place and returned as ``predict``.
    """
    del stream
    _resolve_qhead_per_kv_head(qhead_per_kv_head, q_indexer, k_indexer)
    predict = _sparse_indexer_score_recompute(
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        topk_length,
        topk_indices_global=topk_indices_global,
    )
    return TupleDict(predict=_copy_result(out, predict))


def sparse_attn_score_recompute_wrapper(
    q_attn,
    k_attn,
    lse,
    topk_indices,
    softmax_scale,
    qhead_per_kv_head=None,
    topk_length=None,
    topk_indices_global=False,
    out=None,
    stream=None,
):
    """Sparse attention ``target`` distribution (L1-normalised, over Top-K)."""
    del stream
    _resolve_qhead_per_kv_head(qhead_per_kv_head, q_attn, k_attn)
    target = _sparse_attn_score_recompute(
        q_attn,
        k_attn,
        lse,
        topk_indices,
        float(softmax_scale),
        topk_length,
        topk_indices_global=topk_indices_global,
    )
    return TupleDict(target=_copy_result(out, target))


def _grad_scale(loss_coeff, grad_loss, batch: int, seq_q: int) -> float:
    return (float(loss_coeff) / (int(batch) * int(seq_q))) * float(grad_loss)


def _indexer_backward_wrapper_naive(
    index_q,
    weights,
    index_k,
    attn_score,
    index_score,
    topk_indices,
    *,
    sm_scale=1.0,
    loss_coeff=1.0,
    grad_loss=1.0,
    block_I=None,
    stream=None,
    **kwargs,
):
    """The pure-torch implementation, and the **terminal** of the dispatch chain.

    Deliberately not exported and deliberately ignorant of the architecture: it
    is what :class:`~mudnn.deepseek_sparse_attention.indexer_backward.api.
    IndexerBackward` calls when no TileLang kernel applies. If that class called
    :func:`indexer_backward_wrapper` instead, the two would call each other
    forever on a non-``mp_32`` device -- which is why the dispatching wrapper and
    this one are separate symbols rather than one function with a branch.

    ``index_score`` is accepted for signature parity and ignored: this
    implementation recomputes the predict distribution from
    ``(index_q, index_k, weights, topk_indices)``. ``block_I``, ``stream`` and
    ``**kwargs`` are likewise accepted and ignored.
    """
    del index_score, block_I, stream, kwargs
    batch = int(index_q.shape[0])
    seq_q = int(index_q.shape[1])
    dq, dw, dk = _indexer_backward(
        index_q,
        weights,
        index_k,
        attn_score,
        topk_indices,
        sm_scale=float(sm_scale),
        grad_scale=_grad_scale(loss_coeff, grad_loss, batch, seq_q),
    )
    return TupleDict(d_index_q=dq, d_weights=dw, d_index_k=dk)


def indexer_backward_wrapper(
    index_q,
    weights,
    index_k,
    attn_score,
    index_score,
    topk_indices,
    grad_loss=1.0,
    sm_scale=1.0,
    loss_coeff=1.0,
    block_I=None,
    topk_indices_global=False,
    d_index_q=None,
    d_weights=None,
    d_index_k=None,
    stream=None,
    *,
    backend="default",
):
    """Sparse indexer backward, dispatched by device capability.

    Routes through
    :class:`~mudnn.deepseek_sparse_attention.indexer_backward.api.
    IndexerBackward`, which selects the implementation:

    * ``mp_32`` -- the TileLang sparse kernel. It runs upstream's three stages,
      so it consumes ``index_score`` as the predict operand and **overwrites**
      ``attn_score`` with ``grad_signal``.
    * other devices -- :func:`_indexer_backward_wrapper_naive`, the pure-torch
      path, which recomputes the predict distribution and mutates neither score
      buffer.

    That difference is observable, so a caller that reuses an ``attn_score``
    buffer across two launches gets different behaviour per architecture. The
    choice is made from the hardware rather than a caller flag because it is a
    property of the available kernel, not a preference.

    ``topk_indices_global`` was previously unreachable on this surface; it is
    forwarded now that both implementations can express it (the kernel compiles
    a variant per convention, and the torch path rejects it rather than
    mis-gathering).

    The pre-allocated outputs ``d_index_q`` / ``d_weights`` / ``d_index_k`` and
    the keyword-only ``backend`` are accepted here for parity with upstream and
    with the api module's wrapper. This function used to end with
    ``del kwargs``, which meant an upstream-style call either raised ``TypeError``
    or -- worse for anything spelled positionally -- was accepted and silently
    dropped. There is no ``**kwargs`` any more, so an unknown argument is an
    error at the call site.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.api import (
        indexer_backward_wrapper as _dispatch,
    )

    return _dispatch(
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk_indices,
        grad_loss=grad_loss,
        sm_scale=float(sm_scale),
        loss_coeff=loss_coeff,
        block_I=128 if block_I is None else int(block_I),
        topk_indices_global=bool(topk_indices_global),
        d_index_q=d_index_q,
        d_weights=d_weights,
        d_index_k=d_index_k,
        stream=stream,
        backend=backend,
    )


def _dense_indexer_backward_wrapper_naive(
    index_q,
    weights,
    index_k,
    attn_score,
    attn_l1norm,
    index_score,
    index_lse,
    grad_loss=1.0,
    sm_scale=1.0,
    loss_coeff=1.0,
    block_I=None,
    ratio=1,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    stream=None,
    q_causal_offsets=None,
):
    """The pure-torch dense implementation, and the **terminal** of the chain.

    ``index_score`` / ``index_lse`` are accepted for signature parity and
    ignored: this implementation *recomputes* the predict distribution from
    ``(index_q, index_k, weights)``. ``q_causal_offsets`` has no torch
    implementation, so passing it is an error rather than a silent no-op --
    a dropped offset changes the causal bound and therefore the result.
    """
    if q_causal_offsets is not None:
        raise NotImplementedError(
            "dense indexer backward q_causal_offsets has no torch "
            "implementation; it requires the mp_32 dense kernel. Drop the "
            "argument or run on mp_32."
        )
    del index_score, index_lse, block_I, max_seqlen_q, stream
    if cu_seqlens_q is not None or cu_seqlens_k is not None:
        total_q = int(index_q.shape[0])
        dq, dw, dk = _dense_indexer_backward_thd(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
            sm_scale=float(sm_scale),
            ratio=int(ratio),
            grad_scale=(float(loss_coeff) / max(total_q, 1)) * float(grad_loss),
        )
    else:
        batch = int(index_q.shape[0])
        seq_q = int(index_q.shape[1])
        dq, dw, dk = _dense_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            sm_scale=float(sm_scale),
            ratio=int(ratio),
            grad_scale=_grad_scale(loss_coeff, grad_loss, batch, seq_q),
        )
    return TupleDict(d_index_q=dq, d_weights=dw, d_index_k=dk)


def dense_indexer_backward_wrapper(
    index_q,
    weights,
    index_k,
    attn_score,
    attn_l1norm,
    index_score,
    index_lse,
    *,
    sm_scale=1.0,
    loss_coeff=1.0,
    grad_loss=1.0,
    block_I=None,
    ratio=1,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    q_causal_offsets=None,
    d_index_q=None,
    d_weights=None,
    d_index_k=None,
    stream=None,
):
    """Dense indexer backward, dispatched by device capability.

    Routes through
    :class:`~mudnn.deepseek_sparse_attention.indexer_backward.api.
    DenseIndexerBackward` so that this name and the api module's wrapper are the
    *same* behaviour, matching how ``cudnn.DSA`` mirrors the api package.

    This name used to *be* the torch implementation and quietly dropped
    ``index_score``, ``index_lse``, ``q_causal_offsets`` and anything else it did
    not recognise. That made the same package expose two same-named wrappers with
    different semantics, and made an unsupported argument look accepted. Unknown
    arguments now raise ``TypeError``, and ``q_causal_offsets`` is honoured where
    a supporting implementation exists.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.api import (
        dense_indexer_backward_wrapper as _dispatch,
    )

    return _dispatch(
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
        grad_loss=grad_loss,
        sm_scale=sm_scale,
        loss_coeff=loss_coeff,
        # ``None`` here means "use the api default", which is the device-aware
        # value; the old signature used ``None`` to mean "unspecified" too.
        block_I=64 if block_I is None else int(block_I),
        ratio=ratio,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        q_causal_offsets=q_causal_offsets,
        d_index_q=d_index_q,
        d_weights=d_weights,
        d_index_k=d_index_k,
        stream=stream,
    )


def sparse_attention_backward_wrapper(
    q,
    kv,
    out,
    dout,
    lse,
    attn_sink,
    topk_indices,
    *,
    softmax_scale=None,
    topk_length=None,
    stream=None,
    **kwargs,
):
    del out, lse, stream, kwargs
    if softmax_scale is None:
        softmax_scale = 1.0 / (float(q.shape[-1]) ** 0.5)
    dq, dkv, d_sink = _sparse_attention_backward(
        q,
        kv,
        dout,
        attn_sink,
        topk_indices,
        softmax_scale=float(softmax_scale),
        topk_length=topk_length,
    )
    return TupleDict(dq=dq, dkv=dkv, d_sink=d_sink)


def _copy_result(dst, src):
    if dst is None:
        return src
    dst.copy_(src.to(dtype=dst.dtype))
    return dst


class _CompiledFallback:
    def __init__(self):
        self._compiled = False

    def check_support(self):
        return True

    def compile(self):
        self.check_support()
        self._compiled = True
        return self

    def _ensure_compiled(self):
        if not self._compiled:
            self.compile()


class IndexerForward(_CompiledFallback):
    def __init__(
        self,
        sample_q,
        sample_k,
        sample_w,
        sample_out,
        ratio=4,
        qhead_per_kv_head=None,
        m_block_size=128,
        n_block_size=128,
        q_stage=2,
        kv_stage=4,
        sm_scale=1.0,
    ):
        super().__init__()
        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_w = sample_w
        self.sample_out = sample_out
        self.ratio = int(ratio)
        self.qhead_per_kv_head = qhead_per_kv_head
        self.m_block_size = int(m_block_size)
        self.n_block_size = int(n_block_size)
        self.q_stage = int(q_stage)
        self.kv_stage = int(kv_stage)
        self.sm_scale = float(sm_scale)

    def check_support(self):
        scores = _indexer_forward(
            self.sample_q, self.sample_k, self.sample_w, self.ratio
        )
        if tuple(self.sample_out.shape) != tuple(scores.shape):
            raise ValueError(
                "IndexerForward sample_out shape must match computed scores"
            )
        return True

    def execute(self, q, k, w, out, current_stream=None):
        del current_stream
        self._ensure_compiled()
        outputs = indexer_forward_wrapper(
            q,
            k,
            w,
            ratio=self.ratio,
            qhead_per_kv_head=self.qhead_per_kv_head,
        )
        _copy_result(out, outputs["scores"])


class SparseIndexerScoreRecompute(_CompiledFallback):
    def __init__(
        self,
        sample_q_indexer,
        sample_k_indexer,
        sample_weights,
        sample_topk_indices,
        sample_out,
        sample_topk_length=None,
        qhead_per_kv_head=None,
        topk_indices_global=False,
    ):
        super().__init__()
        self.sample_q_indexer = sample_q_indexer
        self.sample_k_indexer = sample_k_indexer
        self.sample_weights = sample_weights
        self.sample_topk_indices = sample_topk_indices
        self.sample_out = sample_out
        self.sample_topk_length = sample_topk_length
        self.qhead_per_kv_head = qhead_per_kv_head
        self.topk_indices_global = bool(topk_indices_global)

    def check_support(self):
        predict = _sparse_indexer_score_recompute(
            self.sample_q_indexer,
            self.sample_k_indexer,
            self.sample_weights,
            self.sample_topk_indices,
            self.sample_topk_length,
            topk_indices_global=self.topk_indices_global,
        )
        if tuple(self.sample_out.shape) != tuple(predict.shape):
            raise ValueError(
                "SparseIndexerScoreRecompute sample_out shape must match topk output"
            )
        return True

    def execute(
        self,
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        out=None,
        topk_length=None,
        current_stream=None,
    ):
        del current_stream
        self._ensure_compiled()
        outputs = sparse_indexer_score_recompute_wrapper(
            q_indexer,
            k_indexer,
            weights,
            topk_indices,
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_length=topk_length,
            topk_indices_global=self.topk_indices_global,
        )
        return _copy_result(out, outputs["predict"])


class SparseAttnScoreRecompute(_CompiledFallback):
    def __init__(
        self,
        sample_q_attn,
        sample_k_attn,
        sample_lse,
        sample_topk_indices,
        sample_out,
        softmax_scale,
        sample_topk_length=None,
        qhead_per_kv_head=None,
        topk_indices_global=False,
    ):
        super().__init__()
        self.sample_q_attn = sample_q_attn
        self.sample_k_attn = sample_k_attn
        self.sample_lse = sample_lse
        self.sample_topk_indices = sample_topk_indices
        self.sample_out = sample_out
        self.softmax_scale = float(softmax_scale)
        self.sample_topk_length = sample_topk_length
        self.qhead_per_kv_head = qhead_per_kv_head
        self.topk_indices_global = bool(topk_indices_global)

    def check_support(self):
        target = _sparse_attn_score_recompute(
            self.sample_q_attn,
            self.sample_k_attn,
            self.sample_lse,
            self.sample_topk_indices,
            self.softmax_scale,
            self.sample_topk_length,
            topk_indices_global=self.topk_indices_global,
        )
        if tuple(self.sample_out.shape) != tuple(target.shape):
            raise ValueError(
                "SparseAttnScoreRecompute sample_out shape must match topk output"
            )
        return True

    def execute(
        self,
        q_attn,
        k_attn,
        lse,
        topk_indices,
        out=None,
        topk_length=None,
        softmax_scale=None,
        current_stream=None,
    ):
        del current_stream
        self._ensure_compiled()
        outputs = sparse_attn_score_recompute_wrapper(
            q_attn,
            k_attn,
            lse,
            topk_indices,
            self.softmax_scale if softmax_scale is None else float(softmax_scale),
            qhead_per_kv_head=self.qhead_per_kv_head,
            topk_length=topk_length,
            topk_indices_global=self.topk_indices_global,
        )
        return _copy_result(out, outputs["target"])


class DenseIndexerScoreRecompute(_CompiledFallback):
    def __init__(
        self,
        sample_q,
        sample_k,
        sample_weights,
        sample_out,
        sample_denom_out,
        qhead_per_kv_head=None,
        sm_scale=1.0,
        ratio=1,
        is_thd=False,
    ):
        super().__init__()
        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_weights = sample_weights
        self.sample_out = sample_out
        self.sample_denom_out = sample_denom_out
        self.qhead_per_kv_head = qhead_per_kv_head
        self.sm_scale = float(sm_scale)
        self.ratio = int(ratio)
        self.is_thd = bool(is_thd)

    def check_support(self):
        if self.is_thd:
            _validate_thd_dense_score_shapes(
                self.sample_q,
                self.sample_k,
                self.sample_weights,
                self.sample_out,
                self.sample_denom_out,
            )
            return True
        else:
            out, denom = _dense_indexer_score_recompute(
                self.sample_q, self.sample_k, self.sample_weights, self.ratio
            )
        if tuple(self.sample_out.shape) != tuple(out.shape):
            raise ValueError("DenseIndexerScoreRecompute sample_out shape mismatch")
        if tuple(self.sample_denom_out.shape) != tuple(denom.shape):
            raise ValueError(
                "DenseIndexerScoreRecompute sample_denom_out shape mismatch"
            )
        return True

    def execute(
        self,
        q,
        k,
        weights,
        out=None,
        denom_out=None,
        sm_scale=None,
        ratio=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        current_stream=None,
    ):
        del sm_scale, max_seqlen_q, current_stream
        self._ensure_compiled()
        outputs = dense_indexer_score_recompute_wrapper(
            q,
            k,
            weights,
            qhead_per_kv_head=self.qhead_per_kv_head,
            ratio=self.ratio if ratio is None else int(ratio),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )
        return _copy_result(out, outputs["out"]), _copy_result(
            denom_out, outputs["denom"]
        )


class DenseAttnScoreRecompute(_CompiledFallback):
    def __init__(
        self,
        sample_q,
        sample_k,
        sample_lse,
        sample_out,
        sample_denom_out,
        softmax_scale,
        qhead_per_kv_head=None,
        ratio=1,
        is_thd=False,
    ):
        super().__init__()
        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_lse = sample_lse
        self.sample_out = sample_out
        self.sample_denom_out = sample_denom_out
        self.softmax_scale = float(softmax_scale)
        self.qhead_per_kv_head = qhead_per_kv_head
        self.ratio = int(ratio)
        self.is_thd = bool(is_thd)

    def check_support(self):
        if self.is_thd:
            _validate_thd_dense_score_shapes(
                self.sample_q,
                self.sample_k,
                self.sample_lse,
                self.sample_out,
                self.sample_denom_out,
            )
            return True
        else:
            out, denom = _dense_attn_score_recompute(
                self.sample_q,
                self.sample_k,
                self.sample_lse,
                self.softmax_scale,
                self.ratio,
            )
        if tuple(self.sample_out.shape) != tuple(out.shape):
            raise ValueError("DenseAttnScoreRecompute sample_out shape mismatch")
        if tuple(self.sample_denom_out.shape) != tuple(denom.shape):
            raise ValueError("DenseAttnScoreRecompute sample_denom_out shape mismatch")
        return True

    def execute(
        self,
        q,
        k,
        lse,
        out=None,
        denom_out=None,
        softmax_scale=None,
        ratio=None,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        current_stream=None,
    ):
        del max_seqlen_q, current_stream
        self._ensure_compiled()
        outputs = dense_attn_score_recompute_wrapper(
            q,
            k,
            lse,
            self.softmax_scale if softmax_scale is None else float(softmax_scale),
            qhead_per_kv_head=self.qhead_per_kv_head,
            ratio=self.ratio if ratio is None else int(ratio),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=max_seqlen_k,
        )
        return _copy_result(out, outputs["out"]), _copy_result(
            denom_out, outputs["denom"]
        )


class IndexerBackward(_CompiledFallback):
    def __init__(
        self,
        sample_index_q,
        sample_weights,
        sample_index_k,
        sample_d_index_q,
        sample_d_weights,
        sample_d_index_k,
        sample_attn_score,
        sample_index_score,
        sample_topk_indices,
        sm_scale=1.0,
        block_I=128,
        topk_indices_global=False,
    ):
        super().__init__()
        self.sample_index_q = sample_index_q
        self.sample_weights = sample_weights
        self.sample_index_k = sample_index_k
        self.sample_d_index_q = sample_d_index_q
        self.sample_d_weights = sample_d_weights
        self.sample_d_index_k = sample_d_index_k
        self.sample_attn_score = sample_attn_score
        self.sample_index_score = sample_index_score
        self.sample_topk_indices = sample_topk_indices
        self.sm_scale = float(sm_scale)
        self.block_I = int(block_I)
        self.topk_indices_global = bool(topk_indices_global)

    def check_support(self):
        """Validate the sample tensors **without executing anything**.

        This used to call ``indexer_backward_wrapper`` and compare the shapes of
        the tensors it returned. That ran a full backward pass at plan-build
        time, which was wrong three ways:

        * on ``mp_32`` it launched the real TileLang kernel, whose contract is to
          overwrite ``attn_score`` with ``grad_signal`` -- and in the usual
          pattern the sample *is* the buffer ``execute`` later consumes, so the
          caller silently got results computed against a gradient instead of the
          distribution;
        * on any other device it paid a whole ``B*S_q*S_k*D`` torch autograd
          graph just to learn three shapes;
        * either way ``check_support`` is documented as a cheap pre-flight, not a
          numerical operation.

        The output shapes are derivable from the inputs, so no execution is
        needed: ``d_index_q`` mirrors ``index_q``, ``d_weights`` mirrors
        ``weights`` and ``d_index_k`` mirrors ``index_k`` (the naive
        implementation casts each gradient back to its input's dtype).
        """
        q, w, k = self.sample_index_q, self.sample_weights, self.sample_index_k
        if q.ndim != 4:
            raise ValueError(
                "IndexerBackward sample_index_q must be (B, S_q, H, D), got "
                f"{tuple(q.shape)}"
            )
        batch, seq_q, heads, head_dim = (int(x) for x in q.shape)
        if tuple(w.shape) != (batch, seq_q, heads):
            raise ValueError(
                "IndexerBackward sample_weights must be "
                f"{(batch, seq_q, heads)}, got {tuple(w.shape)}"
            )
        if k.ndim != 3 or int(k.shape[0]) != batch or int(k.shape[2]) != head_dim:
            raise ValueError(
                f"IndexerBackward sample_index_k must be (B, S_k, {head_dim}) "
                f"with B={batch}, got {tuple(k.shape)}"
            )
        top_k = int(self.sample_topk_indices.shape[-1])
        for tensor, name in (
            (self.sample_topk_indices, "sample_topk_indices"),
            (self.sample_attn_score, "sample_attn_score"),
            (self.sample_index_score, "sample_index_score"),
        ):
            if tuple(tensor.shape) != (batch, seq_q, top_k):
                raise ValueError(
                    f"IndexerBackward {name} must be {(batch, seq_q, top_k)}, "
                    f"got {tuple(tensor.shape)}"
                )
        for tensor, expected, name in (
            (self.sample_d_index_q, q.shape, "sample_d_index_q"),
            (self.sample_d_weights, w.shape, "sample_d_weights"),
            (self.sample_d_index_k, k.shape, "sample_d_index_k"),
        ):
            if tuple(tensor.shape) != tuple(expected):
                raise ValueError(
                    f"IndexerBackward {name} must be {tuple(expected)}, got "
                    f"{tuple(tensor.shape)}"
                )
        return True

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
        grad_loss=1.0,
        loss_coeff=1.0,
        current_stream=None,
    ):
        """Run the backward. Parameter order matches upstream: ``grad_loss``
        precedes ``loss_coeff``.

        They used to be the other way round. Both are scalars, so a positional
        call silently swapped them -- ``loss_coeff`` became the gradient loss and
        vice versa -- producing a wrong result with no error. This is a
        deliberate break: pass both by keyword if you are unsure.
        """
        del current_stream
        self._ensure_compiled()
        outputs = indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            index_score,
            topk_indices,
            sm_scale=self.sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=self.block_I,
            topk_indices_global=self.topk_indices_global,
        )
        _copy_result(d_index_q, outputs["d_index_q"])
        _copy_result(d_weights, outputs["d_weights"])
        _copy_result(d_index_k, outputs["d_index_k"])


class DenseIndexerBackward(_CompiledFallback):
    def __init__(
        self,
        sample_index_q,
        sample_weights,
        sample_index_k,
        sample_d_index_q,
        sample_d_weights,
        sample_d_index_k,
        sample_attn_score,
        sample_attn_l1norm,
        sample_index_score,
        sample_index_lse,
        sm_scale=1.0,
        block_I=128,
        ratio=1,
        is_thd=False,
        batch=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
    ):
        super().__init__()
        self.sample_index_q = sample_index_q
        self.sample_weights = sample_weights
        self.sample_index_k = sample_index_k
        self.sample_d_index_q = sample_d_index_q
        self.sample_d_weights = sample_d_weights
        self.sample_d_index_k = sample_d_index_k
        self.sample_attn_score = sample_attn_score
        self.sample_attn_l1norm = sample_attn_l1norm
        self.sample_index_score = sample_index_score
        self.sample_index_lse = sample_index_lse
        self.sm_scale = float(sm_scale)
        self.block_I = int(block_I)
        self.ratio = int(ratio)
        self.is_thd = bool(is_thd)
        self.batch = batch
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k

    def check_support(self):
        """Validate the sample tensors **without executing the backward**.

        Same defect as :meth:`IndexerBackward.check_support`: the batch branch
        used to call ``dense_indexer_backward_wrapper`` and compare the returned
        shapes, which ran a full backward pass (a real kernel launch on MUSA, a
        whole autograd graph elsewhere) just to learn shapes that follow from the
        inputs. ``d_index_q`` mirrors ``index_q``, ``d_weights`` mirrors
        ``weights``, and ``d_index_k`` mirrors ``index_k``.
        """
        q, w, k = self.sample_index_q, self.sample_weights, self.sample_index_k
        if self.is_thd:
            _validate_thd_dense_score_shapes(
                q,
                k.unsqueeze(1),
                w,
                self.sample_attn_score,
                self.sample_attn_l1norm,
            )
            expected = (q.shape, w.shape, k.shape)
        else:
            if q.ndim != 4:
                raise ValueError(
                    "DenseIndexerBackward sample_index_q must be "
                    f"(B, S_q, H, D), got {tuple(q.shape)}"
                )
            batch, seq_q, heads, head_dim = (int(x) for x in q.shape)
            if tuple(w.shape) != (batch, seq_q, heads):
                raise ValueError(
                    "DenseIndexerBackward sample_weights must be "
                    f"{(batch, seq_q, heads)}, got {tuple(w.shape)}"
                )
            if (k.ndim != 3 or int(k.shape[0]) != batch
                    or int(k.shape[2]) != head_dim):
                raise ValueError(
                    "DenseIndexerBackward sample_index_k must be "
                    f"(B, S_k, {head_dim}) with B={batch}, got {tuple(k.shape)}"
                )
            seq_k = int(k.shape[1])
            for tensor, expected_shape, name in (
                (self.sample_attn_score, (batch, seq_q, seq_k),
                 "sample_attn_score"),
                (self.sample_index_score, (batch, seq_q, seq_k),
                 "sample_index_score"),
                (self.sample_attn_l1norm, (batch, seq_q),
                 "sample_attn_l1norm"),
                (self.sample_index_lse, (batch, seq_q), "sample_index_lse"),
            ):
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"DenseIndexerBackward {name} must be {expected_shape}, "
                        f"got {tuple(tensor.shape)}"
                    )
            expected = (q.shape, w.shape, k.shape)
        for tensor, expected_shape, name in (
            (self.sample_d_index_q, expected[0], "sample_d_index_q"),
            (self.sample_d_weights, expected[1], "sample_d_weights"),
            (self.sample_d_index_k, expected[2], "sample_d_index_k"),
        ):
            if tuple(tensor.shape) != tuple(expected_shape):
                raise ValueError(
                    f"DenseIndexerBackward {name} must be "
                    f"{tuple(expected_shape)}, got {tuple(tensor.shape)}"
                )
        return True

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
        grad_loss=1.0,
        loss_coeff=1.0,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        current_stream=None,
    ):
        """Run the backward. Parameter order matches upstream: ``grad_loss``
        precedes ``loss_coeff`` (see ``IndexerBackward.execute``)."""
        del current_stream
        self._ensure_compiled()
        outputs = dense_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            sm_scale=self.sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=self.block_I,
            ratio=self.ratio,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=self.max_seqlen_k,
        )
        _copy_result(d_index_q, outputs["d_index_q"])
        _copy_result(d_weights, outputs["d_weights"])
        _copy_result(d_index_k, outputs["d_index_k"])


__all__ = [
    "IndexerTopK",
    "indexer_top_k_wrapper",
    "IndexerForward",
    "IndexerBackward",
    "DenseIndexerBackward",
    "SparseIndexerScoreRecompute",
    "SparseAttnScoreRecompute",
    "DenseIndexerScoreRecompute",
    "DenseAttnScoreRecompute",
    "indexer_forward_wrapper",
    "indexer_backward_wrapper",
    # The terminal pure-torch implementation. Exported (though not public) so
    # ``IndexerBackward``'s fallback can call it directly instead of recursing
    # through the dispatching wrapper.
    "_indexer_backward_wrapper_naive",
    "dense_indexer_backward_wrapper",
    # Terminal torch implementation of the dense family; the fallback in
    # ``indexer_backward.api`` calls it directly.
    "_dense_indexer_backward_wrapper_naive",
    "sparse_indexer_score_recompute_wrapper",
    "sparse_attn_score_recompute_wrapper",
    "dense_indexer_score_recompute_wrapper",
    "dense_attn_score_recompute_wrapper",
]
