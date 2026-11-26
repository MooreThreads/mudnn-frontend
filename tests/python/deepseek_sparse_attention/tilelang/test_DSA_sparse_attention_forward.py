# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Sparse attention forward: contract checks on CPU and MP31/MP32 kernel numerics."""

import pytest
import torch

from mudnn import deepseek_sparse_attention as DSA
from mudnn.deepseek_sparse_attention.sparse_attention_forward import api as forward_api
from deepseek_sparse_attention.tilelang.dsa_utils import (
    DSV41_PACKED_POSITIONS,
    DSV41_PACKED_SEQ_LENS,
    DSV41_POSITIONS,
    dsv41_inputs,
    metrics,
    native_device,
    native_inputs,
    native_reference,
    upstream_check,
)


def test_DSA_sparse_attention_forward_rejects_cpu_and_invalid_contracts(monkeypatch):
    q, kv, idx, _, _, _ = native_inputs(6, 300, 64, 64)
    with pytest.raises(ValueError, match="MUSA"):
        DSA.sparse_attention_forward_wrapper(q, kv, idx[:, :63])
    with pytest.raises(ValueError, match="Q must"):
        DSA.sparse_attention_forward_wrapper(q.float(), kv, idx)
    with pytest.raises(NotImplementedError, match="BF16"):
        DSA.sparse_attention_forward_wrapper(q.half(), kv.half(), idx)
    with pytest.raises(ValueError, match="head_dim"):
        DSA.sparse_attention_forward_wrapper(q[..., :256], kv[:, :256], idx)
    q576, kv576 = (torch.zeros(*t.shape[:-1], 576, dtype=t.dtype) for t in (q, kv))
    with pytest.raises(NotImplementedError, match="QK=V=512"):
        DSA.sparse_attention_forward_wrapper(q576, kv576, idx)
    op = DSA.SparseAttentionForward(q, kv, idx)
    with pytest.raises(ValueError, match="not compiled"):
        op.execute(q, kv, idx)
    with pytest.raises(ValueError, match="MUSA"):
        op.check_support()
    monkeypatch.setattr(forward_api, "validate_inputs", lambda *args, **kwargs: 512**-0.5)
    assert op.check_support()
    monkeypatch.setattr(forward_api, "_compile", lambda *args: (object(), None))
    assert op.compile() is None
    assert op._is_supported and op._compiled_kernel is not None
    for heads, dim, indexer_topk in ((16, 512, 0), (64, 512, 4096), (128, 512, 2048)):
        sample = torch.empty(1, heads, dim, dtype=torch.bfloat16)
        bad = DSA.SparseAttentionForward(sample, kv[:, :dim], idx, indexer_topk=indexer_topk)
        with pytest.raises(ValueError, match="indexer_topk|supports only"):
            bad.check_support()
    with pytest.raises(ValueError, match="exceed"):
        DSA.SparseAttentionForward(q, kv, idx[:, :8], indexer_topk=512).check_support()
    with pytest.raises(ValueError, match="multiple of 64"):
        DSA.SparseAttentionForward(q, kv, idx[:, :63].contiguous()).check_support()
    strided_q = torch.empty(*q.shape, 2, dtype=q.dtype)[..., 0]
    with pytest.raises(ValueError, match="contiguous"):
        DSA.SparseAttentionForward(strided_q, kv, idx).check_support()


@pytest.mark.parametrize(
    "sq,skv,heads,topk,features",
    [(6, 300, h, k, f) for h in (64, 128) for k in (64, 128, 2048) for f in (False, True)]
    + [
        (1, 128, 64, 128, False),
        (62, 95, 128, 128, True),
        (213, 1592, 128, 384, True),
        (64, 8192, 128, 2048, True),
        (64, 8192, 64, 4096, True),
        (37, 8192, 128, 4096, True),
        (1024, 257, 64, 128, True),
        (3, 95, 64, 65, True),
        (2, 95, 64, 1, False),
        (2, 95, 128, 1, False),
    ],
)
def test_DSA_sparse_attention_forward_matches_reference(sq, skv, heads, topk, features):
    native_device()
    q, kv, idx, sink, lengths, _ = native_inputs(sq, skv, heads, topk, features)
    if sq >= 1024:
        # Vary tile counts across persistent queries, including partial last tiles.
        rng = torch.Generator().manual_seed(20261027)
        q = (torch.randint(-7, 8, q.shape, generator=rng).float() / 16).bfloat16()
        kv = (torch.randint(-7, 8, kv.shape, generator=rng).float() / 16).bfloat16()
        lengths = torch.arange(1, sq + 1, dtype=torch.int32).clamp(max=min(topk, skv))
        lengths[0] = 0
        idx.fill_(-1)
        for row in range(sq):
            count = int(lengths[row])
            idx[row, :count] = torch.randperm(min(row + 1, skv), generator=rng)[:count]
    reference = native_reference(q.double(), kv.double(), idx, None if sink is None else sink.double(), lengths)
    qg, kvg, idxg, sinkg, lensg = (t.musa() if t is not None else None for t in (q, kv, idx, sink, lengths))
    outputs = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lensg)
    torch.musa.synchronize()
    for name, expected in zip(("out", "max_logits", "lse"), reference):
        metrics(name, outputs[name], expected)
    assert list(outputs.keys()) == ["out", "max_logits", "lse", "lse_indexer"]
    assert outputs["lse_indexer"] is None
    assert outputs["out"].dtype == torch.bfloat16 and outputs["out"].shape == (sq, heads, 512)
    assert all(t.is_contiguous() for t in outputs.values() if t is not None)
    upstream_check(outputs, q, kv, idx, sink, lengths)
    if sq >= 1024:
        # Persistent CTAs must preserve results when reusing shared storage across queries.
        saved = {name: outputs[name].cpu() for name in ("out", "max_logits", "lse")}
        for _ in range(2):
            repeated = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lensg)
            torch.musa.synchronize()
            for name, expected in saved.items():
                torch.testing.assert_close(repeated[name].cpu(), expected, rtol=0, atol=0)
    if topk >= 512:
        indexed = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lensg, indexer_topk=512)
        prefix_lse = native_reference(q.double(), kv.double(), idx[:, :512], None, lengths)[2]
        if lengths is not None:
            prefix_lse = prefix_lse.masked_fill(lengths[:, None] < 512, float("inf"))
        metrics("lse_indexer", indexed["lse_indexer"], prefix_lse)
        upstream_check(indexed, q, kv, idx, sink, lengths, indexer_topk=512)
    if features:
        assert torch.count_nonzero(outputs["out"][0]).item() == 0
        assert torch.count_nonzero(outputs["out"][:, -1]).item() == 0


@pytest.mark.parametrize("layout", ["strided", "offset", "singleton"])
def test_DSA_sparse_attention_forward_layouts_streams_and_outputs(layout, monkeypatch):
    native_device()
    q, kv, idx, sink, lengths, _ = native_inputs(3, 95, 64, 65, True)
    idx[0] = -1
    lengths[:] = torch.tensor([65, -2, 99], dtype=torch.int32)
    if layout == "singleton":
        q, idx, lengths = q[2:], idx[2:], lengths[2:]

    def materialize(tensor):
        if layout == "strided":
            storage = torch.empty((*tensor.shape, 2), dtype=tensor.dtype, device="musa")
            view = storage[..., 0]
        elif layout == "offset":
            storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device="musa")
            view = storage[1:].view(tensor.shape)
        else:
            view = tensor.musa()
            if tensor.ndim > 1 and tensor.shape[0] == 1:
                view = view.as_strided(view.shape, (0, *view.stride()[1:]))
        view.copy_(tensor)
        return view

    qg, kg, ig, sg, lg = [materialize(t) for t in (q, kv, idx, sink, lengths)]
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    wrapped = DSA.sparse_attention_forward_wrapper(qg, kg, ig, attn_sink=sg, topk_length=lg, softmax_scale=0.025, stream=stream)
    stream.synchronize()
    upstream_check(wrapped, q, kv, idx, sink, lengths, 0.025)
    # Direct execute consumes normalized inputs; staging belongs to the wrapper.
    qc, kc = forward_api.contiguous(qg, 16), forward_api.contiguous(kg, 16)
    ic = torch.nn.functional.pad(forward_api.contiguous(ig, 32), (0, 63), value=-1)
    sc, lc = forward_api.contiguous(sg, 4), forward_api.contiguous(lg, 4)
    op = DSA.SparseAttentionForward(qc, kc, ic, sc, lc, softmax_scale=0.1)
    with pytest.raises(ValueError, match="not compiled"):
        op.execute(qc, kc, ic)
    assert op.check_support() is True and op.compile() is None
    buffers = [torch.empty((*q.shape[:2], 512), dtype=torch.bfloat16, device="musa")]
    buffers += [torch.empty(q.shape[:2], dtype=torch.float32, device="musa") for _ in range(2)]

    def reject_allocation(*args, **kwargs):
        raise AssertionError("execute must use caller-provided output storage")

    stream.wait_stream(torch.musa.current_stream())
    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "zeros_like", "full", "full_like", "cat"):
            patch.setattr(torch, name, reject_allocation)
        actual = op.execute(
            qc, kc, ic, attn_sink=sc, topk_length=lc, softmax_scale=0.025,
            out=buffers[0], max_logits=buffers[1], lse=buffers[2], current_stream=stream.musa_stream,
        )
    stream.synchronize()
    assert isinstance(actual, tuple) and len(actual) == 4 and actual[3] is None
    assert all(value is buffer for value, buffer in zip(actual, buffers))
    expected = native_reference(q.double(), kv.double(), idx, sink.double(), lengths, 0.025)
    for name, value, golden in zip(("out", "max_logits", "lse"), actual, expected):
        metrics(name, value, golden)
    upstream_check(dict(zip(("out", "max_logits", "lse", "lse_indexer"), actual)), q, kv, idx, sink, lengths, 0.025)
    storage = torch.empty(buffers[0].numel() + 1, dtype=q.dtype, device="musa")
    with pytest.raises(ValueError, match="16-byte aligned"):
        op.execute(qc, kc, ic, out=storage[1:].view_as(buffers[0]))
    with pytest.raises(ValueError, match="lse_indexer"):
        op.execute(qc, kc, ic, lse_indexer=buffers[1])
    with pytest.raises(ValueError, match="softmax_scale"):
        op.execute(qc, kc, ic, softmax_scale=-1)
    with pytest.raises(ValueError, match="compiled plan"):
        op.execute(qc[:1, :16], kc, ic[:1])
    with pytest.raises(ValueError, match="out must be preallocated"):
        op.execute(qc, kc, ic)
    with pytest.raises(ValueError, match="overlap"):
        op.execute(qc, kc, ic, attn_sink=sc, topk_length=lc, out=qc, max_logits=buffers[1], lse=buffers[2])
    # Q and KV may alias because both are read-only.
    alias_q = qc[:1]
    alias_kv = alias_q.reshape(64, 512)
    alias_idx = torch.arange(64, dtype=torch.int32, device="musa").view(1, 64)
    aliased = DSA.sparse_attention_forward_wrapper(alias_q, alias_kv, alias_idx)
    upstream_check(aliased, q[:1], q[0], torch.arange(64, dtype=torch.int32).view(1, 64))
    # Scale stays dynamic; shape and layout select compiled plans.
    first = DSA.sparse_attention_forward_wrapper(qg, kg, ig, softmax_scale=0.025)
    second = DSA.sparse_attention_forward_wrapper(qg[:1], kg, ig[:1], softmax_scale=0.1)
    cached = dict(forward_api._cache_of_sparse_attention_forward_objects)
    DSA.sparse_attention_forward_wrapper(qg[:1], kg, ig[:1], softmax_scale=0.025)
    assert cached == forward_api._cache_of_sparse_attention_forward_objects
    for result, qq, ii, scale in ((first, q, idx, 0.025), (second, q[:1], idx[:1], 0.1)):
        upstream_check(result, qq, kv, ii, scale=scale)
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        for eq, ekv, ei in ((qg[:0], kg, ig[:0]), (qg, kg[:0], ig), (qg, kg, ig[:, :0])):
            result = DSA.sparse_attention_forward_wrapper(eq, ekv, ei, stream=stream.musa_stream)
            stream.synchronize()
            assert result["out"].dtype == torch.bfloat16
            assert result["max_logits"].dtype == result["lse"].dtype == torch.float32
            upstream_check(result, eq, ekv, ei)
            assert torch.count_nonzero(result["out"]).item() == 0
            assert result["max_logits"].isneginf().all().item()
            assert result["lse"].isposinf().all().item()
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.parametrize("heads,topk,indexer_topk", [(64, 2051, k) for k in (512, 1024, 2048)] + [(64, 4096, 512), (64, 4097, 512), (64, 8192, 2048), (128, 2048, 1024)])
@pytest.mark.parametrize("has_sink,has_lengths", [(True, True), (False, False), (True, False), (False, True)])
def test_DSA_sparse_attention_forward_indexer_lse(heads, topk, indexer_topk, has_sink, has_lengths, monkeypatch):
    native_device()
    q, kv, idx, sink, lengths, _ = native_inputs(6, 300, heads, topk, True)
    if not has_sink:
        sink = None
    if not has_lengths:
        lengths = None
    if lengths is not None:
        lengths[:] = torch.tensor([topk, topk, indexer_topk - 1, indexer_topk, 9000, -1], dtype=torch.int32)
    idx[1, :indexer_topk] = -1
    qg, kg, ig, sg, lg = (None if t is None else t.musa() for t in (q, kv, idx, sink, lengths))
    result = DSA.sparse_attention_forward_wrapper(qg, kg, ig, attn_sink=sg, topk_length=lg, indexer_topk=indexer_topk)
    upstream_check(result, q, kv, idx, sink, lengths, indexer_topk=indexer_topk)
    prefix = native_reference(q.double(), kv.double(), idx[:, :indexer_topk], lengths=lengths)[2]
    if lengths is not None:
        prefix.masked_fill_(lengths[:, None] < indexer_topk, float("inf"))
    metrics("lse_indexer", result["lse_indexer"], prefix)
    assert result["lse_indexer"].dtype == torch.float32
    assert result["lse_indexer"].is_contiguous()
    if topk % 64 == 0:
        op = DSA.SparseAttentionForward(qg, kg, ig, sg, lg, indexer_topk=indexer_topk)
        op.check_support()
        op.compile()
        buffers = {name: torch.empty_like(value) for name, value in result.items()}
        kernel, persistent_blocks = op._compiled_kernel
        calls = []

        def counted_kernel(*args):
            calls.append(None)
            return kernel(*args)

        def reject_allocation(*args, **kwargs):
            raise AssertionError("execute must not allocate")

        op._compiled_kernel = (counted_kernel, persistent_blocks)
        with monkeypatch.context() as patch:
            for name in ("empty", "empty_like", "zeros", "zeros_like", "full", "full_like", "cat"):
                patch.setattr(torch, name, reject_allocation)
            actual = op.execute(qg, kg, ig, attn_sink=sg, topk_length=lg, **buffers)
        assert len(calls) == 1
        for name, value in zip(buffers, actual):
            assert value is buffers[name]
            torch.testing.assert_close(value, result[name], rtol=0, atol=0)


@pytest.mark.parametrize("case", ["tiny_scale", "negative_logits", "empty_prefix"])
def test_DSA_sparse_attention_forward_mask_sentinels(case):
    native_device()
    q, kv, _, _, _, _ = native_inputs(1, 3, 64, 64, False)
    idx = torch.tensor([[0, 0, 1, -1, 3]], dtype=torch.int32)
    scale = 1e-10
    if case == "negative_logits":
        q.fill_(1024)
        kv.fill_(-1024)
        # Keep scaled logits moderate so the FP32 upstream reference retains log(count).
        scale = 1e-8
    elif case == "empty_prefix":
        idx = torch.full((1, 128), -1, dtype=torch.int32)
        idx[0, 64:67] = torch.tensor([0, 0, 1], dtype=torch.int32)
        scale = 512**-0.5
    result = DSA.sparse_attention_forward_wrapper(q.musa(), kv.musa(), idx.musa(), softmax_scale=scale)
    expected = native_reference(q.double(), kv.double(), idx, scale=scale)
    for name, golden in zip(("out", "max_logits", "lse"), expected):
        metrics(name, result[name], golden)
    upstream_check(result, q, kv, idx, scale=scale)


@pytest.mark.parametrize("heads", [64, 128])
def test_DSA_sparse_attention_forward_dsv41_sparse512_window128(heads):
    native_device()
    q, kv, idx, sink, _, _ = dsv41_inputs(DSV41_POSITIONS, heads)
    reference = native_reference(q.double(), kv.double(), idx, sink.double())
    prefix_lse = native_reference(q.double(), kv.double(), idx[:, :512])[2]
    qg, kvg, idxg, sinkg = (t.musa() for t in (q, kv, idx, sink))
    outputs = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, indexer_topk=512)
    torch.musa.synchronize()
    for name, expected in zip(("out", "max_logits", "lse", "lse_indexer"), (*reference, prefix_lse)):
        metrics(name, outputs[name], expected)
    upstream_check(outputs, q, kv, idx, sink, indexer_topk=512)


@pytest.mark.parametrize("sink", [True, False])
@pytest.mark.parametrize("heads", [64, 128])
def test_DSA_sparse_attention_forward_dsv41_packed_sequences(heads, sink):
    native_device()
    q, kv, idx, sinks, _, _ = dsv41_inputs(
        DSV41_PACKED_POSITIONS, heads, seq_lens=DSV41_PACKED_SEQ_LENS, sink=sink, duplicates=True
    )
    reference = native_reference(q.double(), kv.double(), idx, sinks.double())
    prefix_lse = native_reference(q.double(), kv.double(), idx[:, :512])[2]
    qg, kvg, idxg, sinkg = (t.musa() for t in (q, kv, idx, sinks))
    outputs = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, indexer_topk=512)
    for name, expected in zip(("out", "max_logits", "lse", "lse_indexer"), (*reference, prefix_lse)):
        metrics(name, outputs[name], expected)
    upstream_check(outputs, q, kv, idx, sinks, indexer_topk=512)
    # An explicit full-width topk_length is the same contract as omitting it.
    lengths = torch.full((idx.shape[0],), idx.shape[1], dtype=torch.int32, device="musa")
    explicit = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lengths, indexer_topk=512)
    for name in ("out", "max_logits", "lse", "lse_indexer"):
        assert torch.equal(explicit[name], outputs[name]), name


@pytest.mark.parametrize("heads", [64, 128])
def test_DSA_sparse_attention_forward_dsv41_repeat_bitwise(heads):
    native_device()
    # 1024 consecutive strict-causal queries; the tail rows fill all 640 slots.
    q, kv, idx, sinks, _, _ = dsv41_inputs(list(range(1280, 2304)), heads)
    qg, kvg, idxg, sinkg = (t.musa() for t in (q, kv, idx, sinks))
    first = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, indexer_topk=512)
    first = {name: first[name].clone() for name in ("out", "max_logits", "lse", "lse_indexer")}
    for _ in range(5):
        again = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, indexer_topk=512)
        for name, expected in first.items():
            assert torch.equal(again[name], expected), name
    rows = torch.tensor([0, 511, 1023])
    reference = native_reference(q[rows].double(), kv.double(), idx[rows], sinks.double())
    for name, expected in zip(("out", "max_logits", "lse"), reference):
        metrics("sampled/" + name, first[name][rows.musa()], expected)
