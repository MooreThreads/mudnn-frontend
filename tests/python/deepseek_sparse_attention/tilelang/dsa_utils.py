# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared inputs, FP64 references and metrics for the native sparse attention tests."""

import json

import pytest
import torch

from deepseek_sparse_attention.tilelang.dsa_reference import ref_sparse_attention_forward


def on_cpu(fn):
    """Evaluate a reference on CPU (MUSA lacks FP64 matmul and gather-backward is unreliable) and return it on the input device."""

    def wrapped(*args, **kwargs):
        device = next(t.device for t in (*args, *kwargs.values()) if isinstance(t, torch.Tensor))
        to_cpu = lambda t: t.cpu() if isinstance(t, torch.Tensor) else t
        result = fn(*map(to_cpu, args), **{k: to_cpu(v) for k, v in kwargs.items()})
        if isinstance(result, tuple):
            return tuple(r.to(device) if isinstance(r, torch.Tensor) else r for r in result)
        return result.to(device) if isinstance(result, torch.Tensor) else result

    return wrapped


def native_device():
    """Skip unless a MP31/MP32 device with TileLang is available; return its capability."""
    pytest.importorskip("torch_musa")
    pytest.importorskip("tilelang")
    if not torch.musa.is_available():
        pytest.skip("MUSA device unavailable")
    return tuple(torch.musa.get_device_capability())


def native_inputs(sq, skv, heads, topk, features=True, seed=2026, dim=512):
    rng = torch.Generator().manual_seed(seed)
    q = torch.randn(sq, heads, dim, generator=rng).bfloat16()
    kv = torch.randn(skv, dim, generator=rng).bfloat16()
    idx = torch.randint(skv, (sq, topk), generator=rng, dtype=torch.int32)
    dout = torch.randn(sq, heads, 512, generator=rng).bfloat16()
    sink = lengths = None
    if features:
        # Empty row, out-of-range and negative sentinels, duplicates, ragged lengths, +-inf sinks.
        idx[0] = -1
        if topk > 3:
            idx[:, 3] = skv
        if topk > 7:
            idx[:, 7] = -7
        if topk > 1:
            idx[:, -1] = idx[:, 1]
        if topk > 64:
            idx[:, 64] = idx[:, 1]
        lengths = torch.tensor(([0, 1, 63, 64, 65, topk] * ((sq + 5) // 6))[:sq], dtype=torch.int32)
        sink = torch.linspace(-1, 1, heads, dtype=torch.float32)
        sink[0], sink[-1] = -float("inf"), float("inf")
    return q, kv, idx, sink, lengths, dout


def native_reference(q, kv, indices, sink=None, lengths=None, scale=None):
    """FP64 slot-preserving reference with differentiable +-inf sinks; returns (out, max_logits, lse)."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    valid = (indices >= 0) & (indices < kv.shape[0])
    if lengths is not None:
        valid &= torch.arange(indices.shape[1])[None, :] < lengths[:, None]
    selected = kv[indices.long().masked_fill(~valid, 0)]
    selected = selected.masked_fill(~valid[..., None], 0)
    logits = torch.einsum("shd,std->sht", q, selected) * scale
    logits = logits.masked_fill(~valid[:, None, :], -float("inf"))
    empty = ~valid.any(-1)
    # Give empty rows a finite normalization before explicitly zeroing their contribution.
    safe_logits = torch.where(empty[:, None, None], torch.zeros_like(logits), logits)
    finite_lse = safe_logits.logsumexp(-1)
    probs = safe_logits.softmax(-1).masked_fill(~valid[:, None, :], 0)
    out = torch.einsum("sht,std->shd", probs, selected[..., :512])
    if sink is not None:
        safe_sink = torch.where(sink.isfinite(), sink, torch.zeros_like(sink))
        gate = torch.sigmoid(finite_lse - safe_sink[None, :])
        gate = torch.where(sink.isneginf()[None, :], torch.ones_like(gate), gate)
        gate = torch.where(sink.isposinf()[None, :], torch.zeros_like(gate), gate)
        out = out * gate[..., None]
    return out, logits.amax(-1), finite_lse.masked_fill(empty[:, None], float("inf"))


def upstream_check(outputs, q, kv, indices, sink=None, lengths=None, scale=None, indexer_topk=0):
    """Cross-check against the ported cuDNN frontend gather reference (FP32 on CPU)."""
    tensors = [t.detach().cpu() if t is not None else None for t in (q, kv, indices, sink, lengths)]
    q, kv, indices, sink, lengths = tensors
    golden = ref_sparse_attention_forward(
        q, kv, sink, indices, topk_length=lengths, softmax_scale=scale, indexer_topk=indexer_topk, return_full=True
    )
    for name, expected in zip(("out", "max_logits", "lse", "lse_indexer"), golden):
        if expected is None:
            assert outputs[name] is None
        else:
            metrics("upstream/" + name, outputs[name], expected)


def metrics(name, actual, expected, gradients=False):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected)), name
    finite = torch.isfinite(expected)
    assert torch.equal(actual[~finite], expected[~finite]), name
    x, y = actual[finite], expected[finite]
    error = x - y
    max_abs = error.abs().max().item() if x.numel() else 0.0
    rel_l2 = (error.norm() / y.norm().clamp_min(1e-30)).item()
    sim = (error.square().sum() / (x.square().sum() + y.square().sum()).clamp_min(1e-30)).item()
    print(json.dumps(dict(tensor=name, max_abs=max_abs, rel_l2=rel_l2, similarity_error=sim)))
    if gradients:
        assert rel_l2 < 2e-2 and sim < 1e-4, (name, rel_l2, sim)
    else:
        stats = name.split("/")[-1] in ("lse", "max_logits", "lse_indexer")
        torch.testing.assert_close(actual, expected, rtol=0.002 if stats else 0.02, atol=0.02 if stats else 0.01)


def dsv41_inputs(positions, heads, ratio=4, sparse_topk=512, window=128, seed=2027, seq_lens=None, sink=True, duplicates=False):
    """DSv4.1 layout: KV = [compressed | raw tokens]; each index row is [indexer top-K | causal window].

    ``positions`` index the packed raw tokens; ``seq_lens`` splits them into sequences whose compressed
    blocks and windows never cross a sequence start. ``sink=False`` uses -inf sinks; ``duplicates``
    repeats the first indexer slot of each row.
    """
    rng = torch.Generator().manual_seed(seed)
    positions = torch.as_tensor(positions, dtype=torch.int64)
    seq_lens = [int(positions.max()) + 1] if seq_lens is None else list(seq_lens)
    tokens = sum(seq_lens)
    starts = torch.tensor([0] + seq_lens[:-1]).cumsum(0).tolist()
    block_starts = torch.tensor([0] + [n // ratio for n in seq_lens[:-1]]).cumsum(0).tolist()
    n_sparse = sum(n // ratio for n in seq_lens)
    q = (torch.randint(-7, 8, (len(positions), heads, 512), generator=rng).float() / 16).bfloat16()
    kv = (torch.randint(-7, 8, (n_sparse + tokens, 512), generator=rng).float() / 16).bfloat16()
    dout = torch.randn(len(positions), heads, 512, generator=rng).bfloat16()
    idx = torch.full((len(positions), sparse_topk + window), -1, dtype=torch.int32)
    for row, t in enumerate(positions.tolist()):
        seq = max(i for i, start in enumerate(starts) if start <= t)
        local = t - starts[seq]
        visible = (local + 1) // ratio  # only completed compressed blocks are causal
        idx[row, : min(sparse_topk, visible)] = torch.randperm(visible, generator=rng)[:sparse_topk].int() + block_starts[seq]
        if duplicates and visible > 1:
            idx[row, 1] = idx[row, 0]
        start = max(starts[seq], t - window + 1)
        idx[row, sparse_topk : sparse_topk + t + 1 - start] = torch.arange(start, t + 1, dtype=torch.int32) + n_sparse
    sinks = torch.linspace(-1, 1, heads, dtype=torch.float32) if sink else torch.full((heads,), -float("inf"))
    return q, kv, idx, sinks, dout, n_sparse


# Early rows have partial pools (no compressed block, short window); late rows fill all 640 slots.
DSV41_POSITIONS = [0, 2, 3, 7, 126, 127, 128, 900, 2047, 2048, 2200, 2299, 2300, 2301, 2302, 2303]
# Two packed sequences: sequence starts (one valid window slot, empty prefix), duplicated indexer slots.
DSV41_PACKED_SEQ_LENS = [700, 1700]
DSV41_PACKED_POSITIONS = [0, 1, 3, 127, 128, 699, 700, 701, 703, 827, 828, 1500, 2398, 2399]
