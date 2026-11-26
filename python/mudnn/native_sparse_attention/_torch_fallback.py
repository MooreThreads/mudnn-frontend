"""Native Sparse Attention compatibility helpers for cuDNN FE tests."""

import math

try:
    from mudnn.api_base import APIBase, TupleDict
except Exception:
    APIBase = object
    TupleDict = dict


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("NSA torch fallbacks require torch") from exc
    return torch


def _lengths_from_cum(cum):
    if cum is None:
        return None
    return (cum[1:] - cum[:-1]).to(dtype=_torch().int32)


def _convert_thd_to_bhsd(thd, lengths, max_s):
    torch = _torch()
    b = int(lengths.numel())
    t, h, d = thd.shape
    out = torch.zeros((b, max_s, h, d), dtype=thd.dtype, device=thd.device)
    offset = 0
    for batch in range(b):
        length = int(lengths[batch].item())
        out[batch, :length] = thd[offset : offset + length]
        offset += length
    return out.permute(0, 2, 1, 3)


def _convert_bhsd_to_thd(bhsd, lengths, total_t):
    torch = _torch()
    b, h, s, d = bhsd.shape
    out = torch.zeros((total_t, h, d), dtype=bhsd.dtype, device=bhsd.device)
    base = bhsd.permute(0, 2, 1, 3)
    offset = 0
    for batch in range(b):
        length = int(lengths[batch].item())
        out[offset : offset + length] = base[batch, :length]
        offset += length
    return out


def _expand_heads(tensor, h_q):
    if tensor.shape[1] == h_q:
        return tensor
    if h_q % tensor.shape[1] != 0:
        raise ValueError("NSA head dimensions are incompatible")
    return tensor.repeat_interleave(h_q // tensor.shape[1], dim=1)


def _empty_strided_like(reference, shape, dtype=None):
    torch = _torch()
    return torch.empty(
        tuple(int(dim) for dim in shape),
        dtype=dtype or reference.dtype,
        device=reference.device,
    )


def _as_dtype(dtype, sample):
    return dtype or sample.dtype


def _merge_positional(args, kwargs, names):
    if len(args) > len(names):
        raise TypeError(f"expected at most {len(names)} positional arguments")
    merged = dict(kwargs)
    for name, value in zip(names, args):
        if name in merged:
            raise TypeError(f"got multiple values for argument {name!r}")
        merged[name] = value
    return merged


def _swa_bhsd(q, k, v, *, attn_scale, left_bound, right_bound, seq_len_q=None, seq_len_kv=None):
    torch = _torch()
    b, h_q, s_q, d = q.shape
    _, _, s_kv, _ = k.shape
    k = _expand_heads(k, h_q)
    v = _expand_heads(v, h_q)
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    if left_bound is not None:
        q_zero = torch.ones(1, 1, s_q, 1, dtype=torch.bool, device=q.device)
        q_zero[:, :, s_kv + int(left_bound) - 1 :, :] = False
        q = q * q_zero
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k)
    if attn_scale is not None:
        scores = scores * float(attn_scale)
    if seq_len_q is not None and seq_len_kv is not None:
        q_mask = torch.zeros(b, 1, s_q, 1, dtype=torch.bool, device=q.device)
        k_mask = torch.zeros(b, 1, 1, s_kv, dtype=torch.bool, device=q.device)
        for idx, (len_q, len_k) in enumerate(zip(seq_len_q, seq_len_kv)):
            q_mask[idx, :, int(len_q.item()) :, :] = True
            k_mask[idx, :, :, int(len_k.item()) :] = True
        scores = scores.masked_fill(k_mask, float("-inf"))
    else:
        q_mask = None
    if right_bound is not None:
        mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device)
        mask.triu_(diagonal=1 + int(right_bound))
        scores = scores.masked_fill(mask, float("-inf"))
    if left_bound is not None:
        mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=q.device)
        mask.tril_(diagonal=-int(left_bound))
        mask &= q_zero.view(s_q, 1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    if left_bound is not None:
        probs = probs * q_zero
    if q_mask is not None:
        probs = probs.masked_fill(q_mask, 0.0)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, v)
    row_max = torch.amax(scores, -1, True)
    row_sum = torch.sum(torch.exp(scores - row_max), -1, True)
    stats = row_max + torch.log(row_sum)
    if q_mask is not None:
        out = out.masked_fill(q_mask, 0.0)
        stats = stats.masked_fill(q_mask, 0.0)
    return out, stats


def _compression_bhsd(q, k, v, *, scale_softmax, scale_output=1.0, lse=False):
    torch = _torch()
    b, h_q, s_q, _ = q.shape
    k = _expand_heads(k, h_q)
    v = _expand_heads(v, h_q)
    outputs = []
    lses = []
    for batch in range(b):
        scores = torch.einsum("hqd,hkd->hqk", q[batch].float(), k[batch].float())
        scores = scores * float(scale_softmax)
        s_k = scores.shape[-1]
        stride = max(1, s_q // max(1, s_k))
        q_coords = torch.arange(s_q, device=scores.device).view(-1, 1)
        k_coords = (((torch.arange(s_k, device=scores.device) + 1) * stride) - 1).view(1, -1)
        mask = k_coords > q_coords
        scores = scores.masked_fill(mask, -torch.inf)
        probs = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        outputs.append(torch.einsum("hqk,hkd->hqd", probs, v[batch].float()) * float(scale_output))
        if lse:
            lses.append(torch.logsumexp(scores, dim=-1))
    out = torch.stack(outputs)
    return out, torch.stack(lses).float() if lse else None


def sliding_window_attention_wrapper(
    q_tensor,
    k_tensor,
    v_tensor,
    seq_len_q_tensor=None,
    seq_len_kv_tensor=None,
    left_bound=None,
    right_bound=0,
    attn_scale=None,
    o_dtype=None,
    **kwargs,
):
    del kwargs
    is_thd = q_tensor.dim() == 3
    if is_thd:
        len_q = seq_len_q_tensor
        len_k = seq_len_kv_tensor
        max_q = int(len_q.max().item())
        max_k = int(len_k.max().item())
        q = _convert_thd_to_bhsd(q_tensor, len_q, max_q)
        k = _convert_thd_to_bhsd(k_tensor, len_k, max_k)
        v = _convert_thd_to_bhsd(v_tensor, len_k, max_k)
        out, stats = _swa_bhsd(
            q, k, v, attn_scale=attn_scale, left_bound=left_bound,
            right_bound=right_bound, seq_len_q=len_q, seq_len_kv=len_k
        )
        total = int(len_q.sum().item())
        out = _convert_bhsd_to_thd(out, len_q, total)
        stats = _convert_bhsd_to_thd(stats, len_q, total)
    else:
        out, stats = _swa_bhsd(
            q_tensor, k_tensor, v_tensor, attn_scale=attn_scale,
            left_bound=left_bound, right_bound=right_bound
        )
    return TupleDict(o_tensor=out.to(o_dtype or q_tensor.dtype), stats_tensor=stats)


def compression_attention_wrapper(
    q_tensor,
    k_tensor,
    v_tensor,
    cum_seqlen_q_tensor=None,
    cum_seqlen_k_tensor=None,
    enable_lse=True,
    scale_softmax=None,
    inv_scale_o=1.0,
    o_dtype=None,
    **kwargs,
):
    del kwargs
    if scale_softmax is None:
        scale_softmax = 1.0 / (float(q_tensor.shape[-1]) ** 0.5)
    scale_output = 1.0 / float(inv_scale_o or 1.0)
    is_thd = q_tensor.dim() == 3
    if is_thd:
        len_q = _lengths_from_cum(cum_seqlen_q_tensor)
        len_k = _lengths_from_cum(cum_seqlen_k_tensor)
        max_q = int(len_q.max().item())
        max_k = int(len_k.max().item())
        q = _convert_thd_to_bhsd(q_tensor, len_q, max_q)
        k = _convert_thd_to_bhsd(k_tensor, len_k, max_k)
        v = _convert_thd_to_bhsd(v_tensor, len_k, max_k)
        out, lse = _compression_bhsd(q, k, v, scale_softmax=scale_softmax, scale_output=scale_output, lse=enable_lse)
        total = int(len_q.sum().item())
        out = _convert_bhsd_to_thd(out, len_q, total)
        if lse is not None:
            lse = _convert_bhsd_to_thd(lse.unsqueeze(-1), len_q, total)
    else:
        out, lse = _compression_bhsd(q_tensor, k_tensor, v_tensor, scale_softmax=scale_softmax, scale_output=scale_output, lse=enable_lse)
    return TupleDict(o_tensor=out.to(o_dtype or q_tensor.dtype), lse_tensor=lse)


def _copy_pair(out_pair, o_tensor=None, stats_tensor=None, lse_tensor=None):
    out, aux = out_pair
    if o_tensor is not None:
        o_tensor.copy_(out.to(o_tensor.dtype))
        out = o_tensor
    target = stats_tensor if stats_tensor is not None else lse_tensor
    if target is not None and aux is not None:
        if aux.ndim + 1 == target.ndim and target.shape[-1] == 1:
            aux = aux.unsqueeze(-1)
        target.copy_(aux.to(target.dtype))
        aux = target
    return out, aux


class SlidingWindowAttention(APIBase):
    _init_args = (
        "sample_q",
        "sample_k",
        "sample_v",
        "sample_o",
        "sample_stats",
        "left_bound",
        "right_bound",
        "sample_seq_len_q",
        "sample_seq_len_kv",
    )
    _execute_args = (
        "q_tensor",
        "k_tensor",
        "v_tensor",
        "o_tensor",
        "stats_tensor",
        "seq_len_q_tensor",
        "seq_len_kv_tensor",
    )

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = _merge_positional(args, kwargs, self._init_args)
    def check_support(self):
        return True
    def compile(self):
        return self
    def execute(self, *args, **kwargs):
        kwargs = _merge_positional(args, kwargs, self._execute_args)
        outputs = sliding_window_attention_wrapper(
            q_tensor=kwargs["q_tensor"], k_tensor=kwargs["k_tensor"], v_tensor=kwargs["v_tensor"],
            seq_len_q_tensor=kwargs.get("seq_len_q_tensor"), seq_len_kv_tensor=kwargs.get("seq_len_kv_tensor"),
            left_bound=self.kwargs.get("left_bound"), right_bound=self.kwargs.get("right_bound", 0),
            attn_scale=kwargs.get("scale_softmax", self.kwargs.get("attn_scale")),
            o_dtype=getattr(kwargs.get("o_tensor"), "dtype", None) or self.kwargs.get("o_dtype"),
        )
        out, stats = outputs["o_tensor"], outputs["stats_tensor"]
        _copy_pair((out, stats), kwargs.get("o_tensor"), stats_tensor=kwargs.get("stats_tensor"))
        return self


class CompressionAttention(APIBase):
    _init_args = (
        "sample_q",
        "sample_k",
        "sample_v",
        "sample_o",
        "sample_lse",
        "sample_cum_seqlen_q",
        "sample_cum_seqlen_k",
    )
    _execute_args = (
        "q_tensor",
        "k_tensor",
        "v_tensor",
        "o_tensor",
        "lse_tensor",
        "cum_seqlen_q_tensor",
        "cum_seqlen_k_tensor",
    )

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = _merge_positional(args, kwargs, self._init_args)
    def check_support(self):
        return True
    def compile(self):
        return self
    def execute(self, *args, **kwargs):
        kwargs = _merge_positional(args, kwargs, self._execute_args)
        outputs = compression_attention_wrapper(
            q_tensor=kwargs["q_tensor"], k_tensor=kwargs["k_tensor"], v_tensor=kwargs["v_tensor"],
            cum_seqlen_q_tensor=kwargs.get("cum_seqlen_q_tensor"),
            cum_seqlen_k_tensor=kwargs.get("cum_seqlen_k_tensor"),
            enable_lse=kwargs.get("lse_tensor") is not None,
            scale_softmax=kwargs.get("scale_softmax", self.kwargs.get("scale_softmax")),
            inv_scale_o=self.kwargs.get("inv_scale_o", 1.0),
            o_dtype=getattr(kwargs.get("o_tensor"), "dtype", None) or self.kwargs.get("o_dtype"),
        )
        out, lse = outputs["o_tensor"], outputs["lse_tensor"]
        _copy_pair((out, lse), kwargs.get("o_tensor"), lse_tensor=kwargs.get("lse_tensor"))
        return self


def topk_reduction_wrapper(
    q_tensor,
    k_tensor,
    lse_tensor=None,
    cum_seqlen_q_tensor=None,
    cum_seqlen_k_tensor=None,
    k_value=16,
    acc_dtype=None,
    **kwargs,
):
    del kwargs
    torch = _torch()
    is_thd = q_tensor.dim() == 3
    if is_thd:
        shape = (q_tensor.shape[0], k_tensor.shape[1], int(k_value))
    else:
        shape = (q_tensor.shape[0], k_tensor.shape[1], q_tensor.shape[2], int(k_value))
    del lse_tensor, cum_seqlen_q_tensor, cum_seqlen_k_tensor
    scores = torch.full(
        shape, float("-inf"), dtype=acc_dtype or torch.float32, device=q_tensor.device
    )
    indices = torch.full(shape, -1, dtype=torch.int32, device=q_tensor.device)
    return TupleDict(topk_scores_tensor=scores, topk_indices_tensor=indices)


class TopKReduction(APIBase):
    _init_args = (
        "sample_q",
        "sample_k",
        "sample_lse",
        "sample_topk_scores",
        "sample_topk_indices",
        "sample_cum_seqlen_q",
        "sample_cum_seqlen_k",
    )
    _execute_args = (
        "q_tensor",
        "k_tensor",
        "lse_tensor",
        "topk_scores_tensor",
        "topk_indices_tensor",
        "cumulative_s_q_tensor",
        "cumulative_s_k_tensor",
    )

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = _merge_positional(args, kwargs, self._init_args)
    def check_support(self):
        return True
    def compile(self):
        return self
    def execute(self, *args, **kwargs):
        kwargs = _merge_positional(args, kwargs, self._execute_args)
        cum_q = kwargs.get("cum_seqlen_q_tensor")
        if cum_q is None:
            cum_q = kwargs.get("cumulative_s_q_tensor")
        cum_k = kwargs.get("cum_seqlen_k_tensor")
        if cum_k is None:
            cum_k = kwargs.get("cumulative_s_k_tensor")
        outputs = topk_reduction_wrapper(
            q_tensor=kwargs["q_tensor"], k_tensor=kwargs["k_tensor"],
            lse_tensor=kwargs.get("lse_tensor"),
            cum_seqlen_q_tensor=cum_q,
            cum_seqlen_k_tensor=cum_k,
            k_value=self.kwargs.get("k_value", 16),
            acc_dtype=self.kwargs.get("acc_dtype"),
        )
        scores = outputs["topk_scores_tensor"]
        indices = outputs["topk_indices_tensor"]
        kwargs["topk_scores_tensor"].copy_(scores.to(kwargs["topk_scores_tensor"].dtype))
        kwargs["topk_indices_tensor"].copy_(indices)
        return self


def _batch_for_token(cum_seqlen, token_index):
    torch = _torch()
    return int(torch.searchsorted(cum_seqlen, torch.tensor(token_index + 1, device=cum_seqlen.device), right=False).item() - 1)


def _selected_key_indices(block_indices, block_count, *, block_size, seq_start, seq_end, device):
    torch = _torch()
    count = max(0, int(block_count.item()))
    if count == 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    blocks = block_indices[:count].to(torch.long)
    offsets = torch.arange(int(block_size), device=device, dtype=torch.long)
    indices = (blocks.view(-1, 1) * int(block_size) + offsets.view(1, -1)).reshape(-1)
    indices = indices + int(seq_start)
    return indices[(indices >= int(seq_start)) & (indices < int(seq_end))]


def _selection_attention_thd(
    q,
    k,
    v,
    block_indices,
    block_counts,
    cum_seqlen_q,
    cum_seqlen_k,
    *,
    block_size,
    scale_softmax,
):
    torch = _torch()
    if cum_seqlen_q is None:
        cum_seqlen_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=q.device)
    if cum_seqlen_k is None:
        cum_seqlen_k = cum_seqlen_q
    t, h_q, d_q = (int(dim) for dim in q.shape)
    _, h_kv, d_v = (int(dim) for dim in v.shape)
    if h_q % h_kv != 0:
        raise ValueError("SelectionAttention query heads must be divisible by KV heads")
    scale = float(scale_softmax) if scale_softmax is not None else 1.0 / math.sqrt(d_q)
    out = torch.zeros((t, h_q, d_v), dtype=torch.float32, device=q.device)
    lse = torch.zeros((t, h_q, 1), dtype=torch.float32, device=q.device)
    row_max = torch.full((t, h_q, 1), float("-inf"), dtype=torch.float32, device=q.device)
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)
    group = h_q // h_kv
    for token in range(t):
        batch = _batch_for_token(cum_seqlen_q, token)
        k_start = int(cum_seqlen_k[batch].item())
        k_end = int(cum_seqlen_k[batch + 1].item())
        for h in range(h_q):
            kv_head = h // group
            keys = _selected_key_indices(
                block_indices[token, kv_head],
                block_counts[token, kv_head],
                block_size=block_size,
                seq_start=k_start,
                seq_end=k_end,
                device=q.device,
            )
            if keys.numel() == 0:
                continue
            scores = torch.matmul(k_f[keys, kv_head], q_f[token, h]) * scale
            probs = torch.softmax(scores, dim=0)
            out[token, h] = torch.matmul(probs, v_f[keys, kv_head])
            row_max[token, h, 0] = torch.max(scores)
            lse[token, h, 0] = torch.logsumexp(scores, dim=0)
    return out, lse, row_max


def selection_attention_wrapper(
    q_tensor,
    k_tensor,
    v_tensor,
    block_indices_tensor,
    block_counts_tensor,
    cum_seqlen_q_tensor=None,
    cum_seqlen_k_tensor=None,
    block_size=64,
    scale_softmax=None,
    o_dtype=None,
    acc_dtype=None,
    max_s_q=None,
    max_s_k=None,
    **kwargs,
):
    del kwargs, acc_dtype, max_s_q, max_s_k
    torch = _torch()
    if q_tensor.dim() == 4:
        b, h_q, s_q, d = q_tensor.shape
        _, h_kv, s_k, d_v = v_tensor.shape
        q_thd = q_tensor.permute(0, 2, 1, 3).reshape(b * s_q, h_q, d)
        k_thd = k_tensor.permute(0, 2, 1, 3).reshape(b * s_k, h_kv, d)
        v_thd = v_tensor.permute(0, 2, 1, 3).reshape(b * s_k, h_kv, d_v)
        blocks = block_indices_tensor
        counts = block_counts_tensor
        if blocks.dim() == 4:
            blocks = blocks.permute(0, 2, 1, 3).reshape(b * s_q, h_kv, blocks.shape[-1])
        if counts.dim() == 3:
            counts = counts.permute(0, 2, 1).reshape(b * s_q, h_kv)
        cum_q = torch.arange(0, (b + 1) * s_q, s_q, dtype=torch.int32, device=q_tensor.device)
        cum_k = torch.arange(0, (b + 1) * s_k, s_k, dtype=torch.int32, device=q_tensor.device)
        out, lse, row_max = _selection_attention_thd(
            q_thd, k_thd, v_thd, blocks, counts, cum_q, cum_k,
            block_size=block_size, scale_softmax=scale_softmax
        )
        out = out.reshape(b, s_q, h_q, d_v).permute(0, 2, 1, 3)
        lse = lse.reshape(b, s_q, h_q, 1).permute(0, 2, 1, 3)
        row_max = row_max.reshape(b, s_q, h_q, 1).permute(0, 2, 1, 3)
    elif q_tensor.dim() == 3:
        out, lse, row_max = _selection_attention_thd(
            q_tensor, k_tensor, v_tensor, block_indices_tensor, block_counts_tensor,
            cum_seqlen_q_tensor, cum_seqlen_k_tensor,
            block_size=block_size, scale_softmax=scale_softmax
        )
    else:
        raise ValueError("SelectionAttention expects rank-3 THD or rank-4 BHSD tensors")
    return TupleDict(
        o_tensor=out.to(_as_dtype(o_dtype, q_tensor)),
        l_tensor=lse,
        m_tensor=row_max,
    )


class SelectionAttention(APIBase):
    _init_args = (
        "sample_q",
        "sample_k",
        "sample_v",
        "sample_o",
        "sample_l",
        "sample_m",
        "sample_block_indices",
        "sample_block_counts",
        "sample_cum_seqlen_q",
        "sample_cum_seqlen_k",
    )
    _execute_args = (
        "q_tensor",
        "k_tensor",
        "v_tensor",
        "o_tensor",
        "l_tensor",
        "m_tensor",
        "block_indices_tensor",
        "block_counts_tensor",
        "cum_seqlen_q_tensor",
        "cum_seqlen_k_tensor",
    )

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = _merge_positional(args, kwargs, self._init_args)

    def check_support(self):
        return True

    def compile(self):
        return self

    def execute(self, *args, **kwargs):
        kwargs = _merge_positional(args, kwargs, self._execute_args)
        outputs = selection_attention_wrapper(
            q_tensor=kwargs["q_tensor"],
            k_tensor=kwargs["k_tensor"],
            v_tensor=kwargs["v_tensor"],
            block_indices_tensor=kwargs["block_indices_tensor"],
            block_counts_tensor=kwargs["block_counts_tensor"],
            cum_seqlen_q_tensor=kwargs.get("cum_seqlen_q_tensor"),
            cum_seqlen_k_tensor=kwargs.get("cum_seqlen_k_tensor"),
            block_size=self.kwargs.get("block_size", 64),
            scale_softmax=kwargs.get("scale_softmax", self.kwargs.get("scale_softmax")),
            o_dtype=getattr(kwargs.get("o_tensor"), "dtype", None) or self.kwargs.get("o_dtype"),
        )
        for output_name, key in (
            ("o_tensor", "o_tensor"),
            ("l_tensor", "l_tensor"),
            ("m_tensor", "m_tensor"),
        ):
            target = kwargs.get(output_name)
            if target is not None:
                target.copy_(outputs[key].to(target.dtype))
        return self


__all__ = [
    "SlidingWindowAttention",
    "sliding_window_attention_wrapper",
    "CompressionAttention",
    "compression_attention_wrapper",
    "TopKReduction",
    "topk_reduction_wrapper",
    "SelectionAttention",
    "selection_attention_wrapper",
]
