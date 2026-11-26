# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Sparse attention backward: contract checks on CPU and MP31/MP32 gradients against FP64 autograd."""

import inspect

import pytest
import torch

from mudnn import deepseek_sparse_attention as DSA
from deepseek_sparse_attention.tilelang.dsa_utils import (
    DSV41_PACKED_POSITIONS,
    DSV41_PACKED_SEQ_LENS,
    DSV41_POSITIONS,
    dsv41_inputs,
    metrics,
    native_device,
    native_inputs,
    native_reference,
)


def test_DSA_sparse_attention_backward_contract_on_cpu():
    execute = inspect.signature(DSA.SparseAttentionBackward.execute).parameters
    assert execute["dq"].default is inspect.Parameter.empty
    assert execute["dkv"].default is inspect.Parameter.empty
    assert execute["d_sink"].kind == inspect.Parameter.KEYWORD_ONLY
    q, kv, idx, sink, _, dout = native_inputs(6, 300, 16, 64)
    lse = torch.zeros(6, 16)
    with pytest.raises(ValueError, match="MUSA"):
        DSA.sparse_attention_backward_wrapper(q, kv, dout, dout, lse, sink, idx)
    bwd = DSA.SparseAttentionBackward(q, kv, dout, dout, lse, sink, idx)
    with pytest.raises(ValueError, match="MUSA"):
        bwd.compile()
    with pytest.raises(ValueError, match="dq must be preallocated"):
        bwd.execute(q, kv, dout, dout, lse, sink, idx, None, None, d_sink=None)
    with pytest.raises(ValueError, match="q_cluster_mode"):
        DSA.sparse_attention_backward_wrapper(q, kv, dout, dout, lse, sink, idx, q_cluster_mode="unknown")
    for kwargs, match in (({"deterministic": True}, "deterministic"), ({"block_tile": 128}, "block_tile"), ({"q_cluster_mode": "adaptive_pair"}, "adaptive_pair")):
        with pytest.raises(NotImplementedError, match=match):
            DSA.SparseAttentionBackward(q, kv, dout, dout, lse, sink, idx, **kwargs).check_support()
    with pytest.raises(NotImplementedError, match="BF16"):
        DSA.SparseAttentionBackward(q.half(), kv.half(), dout.half(), dout.half(), lse, sink, idx).check_support()
    q576, kv576 = (torch.zeros(*t.shape[:-1], 576, dtype=t.dtype) for t in (q, kv))
    with pytest.raises(NotImplementedError, match="QK=V=512"):
        DSA.SparseAttentionBackward(q576, kv576, dout, dout, lse, sink, idx).check_support()


_BACKWARD_CASES = [(6, 300, h, k, f) for h in (16, 64, 128) for k in (64, 128, 2048) for f in (False, True)] + [
    (17, 8192, 64, 2048, True),
    (13, 1592, 128, 384, True),
    (3, 95, 32, 65, True),
    (2, 95, 16, 1, False),
    (2, 95, 16, 1, True),
]


@pytest.mark.parametrize("sq,skv,heads,topk,features", [
    pytest.param(3, 95, 64, 8192, True, id="topk-limit-8192")
] + _BACKWARD_CASES + [(6, 300, 32, 2051, True)])
def test_DSA_sparse_attention_backward_matches_autograd(sq, skv, heads, topk, features):
    native_device()
    q, kv, idx, sink, lengths, dout = native_inputs(sq, skv, heads, topk, features)
    qg, kvg, idxg, sinkg, lensg, dog = (t.musa() if t is not None else None for t in (q, kv, idx, sink, lengths, dout))
    native_sink = sinkg if sinkg is not None else torch.full((heads,), -float("inf"), dtype=torch.float32, device=qg.device)
    qr, kr = q.double().requires_grad_(), kv.double().requires_grad_()
    sr = None if sink is None else sink.double().requires_grad_()
    ref_out, _, ref_lse = native_reference(qr, kr, idx, sr, lengths)
    ref_out.backward(dout.double())
    expected = (qr.grad, kr.grad, None if sr is None else sr.grad)
    sources = ["reference"]
    if heads in (64, 128):
        sources.append("native")
        outputs = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lensg)
    for source in sources:
        out = ref_out.detach().bfloat16().musa() if source == "reference" else outputs["out"]
        lse = ref_lse.detach().float().musa() if source == "reference" else outputs["lse"]
        result = DSA.sparse_attention_backward_wrapper(qg, kvg, out, dog, lse, native_sink, idxg, topk_length=lensg)
        torch.musa.synchronize()
        for name, golden in zip(("dq", "dkv", "d_sink"), expected):
            assert result[name].shape == (q.shape if name == "dq" else kv.shape if name == "dkv" else (heads,))
            if golden is None:
                assert torch.count_nonzero(result[name]).item() == 0
            else:
                metrics(f"{source}/{name}", result[name], golden, gradients=True)
        if features:
            assert torch.count_nonzero(result["dq"][0]).item() == 0


def test_DSA_sparse_attention_backward_strides_streams_and_workspace(monkeypatch):
    native_device()
    q, kv, idx, sink, lengths, dout = native_inputs(6, 300, 64, 128, True)
    lengths[0], lengths[-1] = -1, 200
    idx[1] = -1
    scale = 0.1

    def strided(tensor):
        shape = list(tensor.shape)
        shape[0] *= 2
        storage = torch.empty(shape, dtype=tensor.dtype, device="musa")
        view = storage[::2]
        view.copy_(tensor)
        return view

    qg, kvg, idxg = (strided(t) for t in (q, kv, idx))
    sinkg, lensg, dog = (t.musa() for t in (sink, lengths, dout))
    stream = torch.musa.Stream()
    forward = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lensg, softmax_scale=scale)
    out, lse = forward["out"], forward["lse"]
    qc, kc, ic = qg.contiguous(), kvg.contiguous(), idxg.contiguous()
    with pytest.raises(NotImplementedError, match="contiguous"):
        DSA.SparseAttentionBackward(qg, kvg, out, dog, lse, sinkg, idxg).check_support()
    with pytest.raises(ValueError, match="lse"):
        DSA.SparseAttentionBackward(qc, kc, out, dog, lse.bfloat16(), sinkg, ic).check_support()
    bwd = DSA.SparseAttentionBackward(qc, kc, out, dog, lse, sinkg, ic, sample_topk_length=lensg, softmax_scale=scale)
    assert bwd.check_support()
    assert bwd.compile() is None
    dq, dkv, dsink = torch.empty_like(qc), torch.empty_like(kc), torch.empty_like(sinkg)
    workspace = torch.empty(bwd.scratch_workspace_bytes(), dtype=torch.uint8, device=qc.device)
    args = (qc, kc, out, dog, lse, sinkg, ic, dq, dkv)
    with pytest.raises(ValueError, match="workspace"):
        bwd.execute(*args, topk_length=lensg, d_sink=dsink)
    with pytest.raises(ValueError, match="too small"):
        bwd.execute(*args, topk_length=lensg, workspace=workspace[:1], d_sink=dsink)
    with pytest.raises(ValueError, match="overlap"):
        bwd.execute(qc, kc, out, dog, lse, sinkg, ic, qc, dkv, topk_length=lensg, workspace=workspace, d_sink=dsink)
    stream.wait_stream(torch.musa.current_stream())

    def reject_allocation(*args, **kwargs):
        raise AssertionError("execute must use caller-provided output/workspace storage")

    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "zeros_like", "full", "full_like", "cat"):
            patch.setattr(torch, name, reject_allocation)
        result = bwd.execute(*args, topk_length=lensg, current_stream=stream, workspace=workspace, d_sink=dsink)
    stream.synchronize()
    assert result[0] is dq and result[1] is dkv and result[2] is dsink
    qr, kr, sr = q.double().requires_grad_(), kv.double().requires_grad_(), sink.double().requires_grad_()
    native_reference(qr, kr, idx, sr, lengths, scale)[0].backward(dout.double())
    for name, value, golden in zip(("dq", "dkv", "d_sink"), result, (qr.grad, kr.grad, sr.grad)):
        metrics(name, value, golden, gradients=True)
    assert torch.count_nonzero(dq[1]).item() == 0
    sdq, sdkv = strided(torch.zeros_like(q)), strided(torch.zeros_like(kv))
    stream.wait_stream(torch.musa.current_stream())
    staged = DSA.sparse_attention_backward_wrapper(qg, kvg, out, dog, lse, sinkg, idxg, scale, lensg, sdq, sdkv, stream=stream, workspace=workspace)
    stream.synchronize()
    assert staged["dq"] is sdq and staged["dkv"] is sdkv
    metrics("staged/dq", sdq, qr.grad, gradients=True)
    metrics("staged/dkv", sdkv, kr.grad, gradients=True)
    for bad in (None, lensg[:1]):
        with pytest.raises(ValueError, match="topk_length"):
            bwd.execute(*args, topk_length=bad, workspace=workspace, d_sink=dsink)
    with pytest.raises(ValueError, match="aligned"):
        bwd.execute(*args, topk_length=lensg, workspace=workspace[1:], d_sink=dsink)
    # Reuse poisoned storage and override the compiled default scale on a raw stream.
    for runtime_scale in (0.025, 0.1):
        fresh = DSA.sparse_attention_forward_wrapper(qc, kc, ic, attn_sink=sinkg, topk_length=lensg, softmax_scale=runtime_scale)
        workspace.fill_(255)
        dq.fill_(float("nan"))
        dkv.fill_(float("nan"))
        dsink.fill_(float("nan"))
        stream.wait_stream(torch.musa.current_stream())
        reused = bwd.execute(
            qc,
            kc,
            fresh["out"],
            dog,
            fresh["lse"],
            sinkg,
            ic,
            dq,
            dkv,
            topk_length=lensg,
            softmax_scale=runtime_scale,
            current_stream=stream.musa_stream,
            workspace=workspace,
            d_sink=dsink,
        )
        stream.synchronize()
        qr.grad = kr.grad = sr.grad = None
        native_reference(qr, kr, idx, sr, lengths, runtime_scale)[0].backward(dout.double())
        for name, value, golden in zip(("dq", "dkv", "d_sink"), reused, (qr.grad, kr.grad, sr.grad)):
            metrics(f"reused/{runtime_scale}/{name}", value, golden, gradients=True)
    no_sink = torch.full_like(sinkg, -float("inf"))
    empty = DSA.sparse_attention_forward_wrapper(qc[:1], kc, ic[:1])
    empty_grad = DSA.sparse_attention_backward_wrapper(qc[:1], kc, empty["out"], dog[:1], empty["lse"], no_sink, ic[:1])
    for value in empty_grad.values():
        assert torch.count_nonzero(value).item() == 0
    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(NotImplementedError, match="atomics"):
            bwd.execute(*args, topk_length=lensg, workspace=workspace, d_sink=dsink)
    finally:
        torch.use_deterministic_algorithms(old)


def test_DSA_sparse_attention_backward_atomic_repeated_and_unused_kv():
    native_device()
    q, kv, idx, sink, lengths, dout = native_inputs(6, 300, 128, 128, True, seed=1)
    idx[1:] = idx[1:].remainder(8)
    qg, kvg, idxg, sinkg, lensg, dog = (t.musa() for t in (q, kv, idx, sink, lengths, dout))
    qr, kr, sr = q.double().requires_grad_(), kv.double().requires_grad_(), sink.double().requires_grad_()
    native_reference(qr, kr, idx, sr, lengths)[0].backward(dout.double())
    outputs = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lensg)
    for iteration in range(10):
        result = DSA.sparse_attention_backward_wrapper(qg, kvg, outputs["out"], dog, outputs["lse"], sinkg, idxg, topk_length=lensg)
        for name, golden in zip(("dq", "dkv", "d_sink"), (qr.grad, kr.grad, sr.grad)):
            metrics(f"{iteration}/{name}", result[name], golden, gradients=True)
        assert torch.count_nonzero(result["dkv"][8:]).item() == 0
        assert torch.count_nonzero(result["dq"][0]).item() == 0
        assert torch.count_nonzero(result["dq"][:, -1]).item() == 0


@pytest.mark.parametrize("heads,topk", [(16, 65), (64, 65), (64, 640), (128, 65), (128, 640)])
def test_DSA_sparse_attention_backward_persistent_empty_and_ragged_queries(heads, topk):
    if native_device() != (3, 2):
        pytest.skip("Persistent backward requires MP32")
    mp_count = torch.musa.get_device_properties(torch.musa.current_device()).multi_processor_count
    sq = 2 * mp_count + 5
    q, kv, idx, sink, lengths, dout = native_inputs(sq, 95, heads, topk)
    pattern = torch.tensor([0, 1, 31, 32, 33, 63, 64, 65, topk, -1, topk + 1], dtype=torch.int32)
    lengths = pattern[torch.arange(sq) % pattern.numel()]
    # Each resident CTA crosses empty/nonempty queries and the grid has a partial final wave.
    lengths[:mp_count] = 0
    qr, kr, sr = q.double().requires_grad_(), kv.double().requires_grad_(), sink.double().requires_grad_()
    ref_out, _, ref_lse = native_reference(qr, kr, idx, sr, lengths)
    ref_out.backward(dout.double())
    expected = (qr.grad, kr.grad, sr.grad)
    qg, kvg, idxg, sinkg, lensg, dog = (t.musa() for t in (q, kv, idx, sink, lengths, dout))
    out, lse = ref_out.detach().bfloat16().musa(), ref_lse.detach().float().musa()
    bwd = DSA.SparseAttentionBackward(qg, kvg, out, dog, lse, sinkg, idxg, sample_topk_length=lensg)
    bwd.compile()
    dq, dkv, dsink = torch.empty_like(qg), torch.empty_like(kvg), torch.empty_like(sinkg)
    workspace = torch.empty(bwd.scratch_workspace_bytes(), dtype=torch.uint8, device=qg.device)
    for empty in (False, True, False):
        lensg.copy_(torch.zeros_like(lengths) if empty else lengths)
        if empty:
            out.zero_()
            lse.fill_(float("inf"))
        else:
            out.copy_(ref_out.detach().bfloat16())
            lse.copy_(ref_lse.detach().float())
        workspace.fill_(255)
        for value in (dq, dkv, dsink):
            value.fill_(float("nan"))
        result = bwd.execute(qg, kvg, out, dog, lse, sinkg, idxg, dq, dkv,
                             topk_length=lensg, workspace=workspace, d_sink=dsink)
        torch.musa.synchronize()
        for name, value, golden in zip(("dq", "dkv", "d_sink"), result, expected):
            if empty:
                assert torch.count_nonzero(value).item() == 0, name
            else:
                metrics(name, value, golden, gradients=True)
        assert torch.count_nonzero(dq[:mp_count]).item() == 0


def test_DSA_sparse_attention_backward_rejects_topk_beyond_limit():
    native_device()
    q, kv, idx, _, _, dout = native_inputs(3, 95, 64, 8193, False)
    q, kv, idx, dout = (t.musa() for t in (q, kv, idx, dout))
    sink = torch.full((64,), -float("inf"), dtype=torch.float32, device="musa")
    lse = torch.zeros(3, 64, dtype=torch.float32, device="musa")
    with pytest.raises(NotImplementedError, match="8192"):
        DSA.SparseAttentionBackward(q, kv, dout, dout, lse, sink, idx).check_support()
    with pytest.raises(NotImplementedError, match="8192"):
        DSA.sparse_attention_backward_wrapper(q, kv, dout, dout, lse, sink, idx)


def test_DSA_sparse_attention_backward_cast_preserves_columns_and_row_tail():
    if native_device() != (3, 2):
        pytest.skip("MP32 cast kernel")
    from mudnn.deepseek_sparse_attention.sparse_attention_backward.dsa_bwd_mp32 import postprocess

    dim = 512
    source = torch.randn(19, dim, generator=torch.Generator().manual_seed(81))
    physical = source.reshape(19, dim // 16, 4, 4).transpose(-1, -2).reshape(19, dim)
    target = torch.full((19, dim), float("nan"), dtype=torch.bfloat16, device="musa")
    postprocess(dim)(physical.musa(), target)
    torch.testing.assert_close(target.cpu(), source.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("heads", [64, 128])
def test_DSA_sparse_attention_backward_dsv41_sparse512_window128(heads):
    native_device()
    q, kv, idx, sink, dout, n_sparse = dsv41_inputs(DSV41_POSITIONS, heads)
    qr, kr, sr = (t.double().requires_grad_() for t in (q, kv, sink))
    ref_out, _, _ = native_reference(qr, kr, idx, sr)
    ref_out.backward(dout.double())
    qg, kvg, idxg, sinkg, dog = (t.musa() for t in (q, kv, idx, sink, dout))
    forward = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, indexer_topk=512)
    result = DSA.sparse_attention_backward_wrapper(qg, kvg, forward["out"], dog, forward["lse"], sinkg, idxg)
    torch.musa.synchronize()
    metrics("dq", result["dq"], qr.grad, gradients=True)
    # The caller splits the shared KV gradient back into the compressed and window pools.
    metrics("dkv_sparse", result["dkv"][:n_sparse], kr.grad[:n_sparse], gradients=True)
    metrics("dkv_swa", result["dkv"][n_sparse:], kr.grad[n_sparse:], gradients=True)
    metrics("d_sink", result["d_sink"], sr.grad, gradients=True)


@pytest.mark.parametrize("sink", [True, False])
@pytest.mark.parametrize("heads", [64, 128])
def test_DSA_sparse_attention_backward_dsv41_packed_sequences(heads, sink):
    native_device()
    q, kv, idx, sinks, dout, n_sparse = dsv41_inputs(
        DSV41_PACKED_POSITIONS, heads, seq_lens=DSV41_PACKED_SEQ_LENS, sink=sink, duplicates=True
    )
    qr, kr, sr = (t.double().requires_grad_() for t in (q, kv, sinks))
    native_reference(qr, kr, idx, sr)[0].backward(dout.double())
    qg, kvg, idxg, sinkg, dog = (t.musa() for t in (q, kv, idx, sinks, dout))
    forward = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, indexer_topk=512)
    result = DSA.sparse_attention_backward_wrapper(qg, kvg, forward["out"], dog, forward["lse"], sinkg, idxg)
    metrics("dq", result["dq"], qr.grad, gradients=True)
    metrics("dkv_sparse", result["dkv"][:n_sparse], kr.grad[:n_sparse], gradients=True)
    metrics("dkv_swa", result["dkv"][n_sparse:], kr.grad[n_sparse:], gradients=True)
    if sink:
        metrics("d_sink", result["d_sink"], sr.grad, gradients=True)
    else:
        assert torch.count_nonzero(result["d_sink"]).item() == 0
    referenced = torch.zeros(kv.shape[0], dtype=torch.bool)
    referenced[idx[idx >= 0].long()] = True
    assert torch.count_nonzero(result["dkv"][~referenced.musa()]).item() == 0
    # Sequence starts see exactly one window slot; with -inf sinks the score gradient vanishes.
    single = ((idx >= 0).sum(-1) == 1).nonzero().flatten()
    assert single.numel() >= 2
    single_dq = result["dq"][single.musa()].float().abs().max().item()
    print("single_slot_dq_max_abs", single_dq, "dq_max_abs", result["dq"].float().abs().max().item())
    if not sink:
        assert single_dq <= 1e-3 * result["dq"].float().abs().max().item()
    # dQ is written once per query (no atomics); it must be reproducible.
    again = DSA.sparse_attention_backward_wrapper(qg, kvg, forward["out"], dog, forward["lse"], sinkg, idxg)
    assert torch.equal(again["dq"], result["dq"])
    metrics("repeat/dkv", again["dkv"], kr.grad, gradients=True)
