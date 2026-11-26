# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Torch fallbacks for the DSA indexer, top-K and score-recompute APIs."""

import pytest


def _dense_indexer_backward_kernel_available():
    """Whether the TileLang MUSA dense indexer backward kernel can be built.

    ``DenseIndexerBackward`` dispatches to
    ``mudnn.deepseek_sparse_attention.indexer_backward.dense_indexer_backward``,
    which needs both an importable ``tilelang_musa`` and a MUSA device. Where
    either is missing the factory declines with ``NotImplementedError`` instead
    of falling back to a torch reference, so the numeric assertions only run on
    a MUSA lane.
    """
    import importlib.util

    if importlib.util.find_spec("tilelang") is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    musa = getattr(torch, "musa", None)
    return bool(musa is not None and musa.is_available())


def test_mudnn_frontend_cudnn_dsa_indexer_top_k_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn
    from mudnn import DSA

    assert "DSA" in mudnn.__all__
    input_values = torch.tensor(
        [
            [0.5, 2.0, -1.0, 1.5],
            [3.0, 1.0, 2.5, -0.5],
            [4.0, -1.0, 0.25, 0.5],
            [0.0, 5.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    seq_lens = torch.tensor([3, 4], dtype=torch.int32)

    op = DSA.IndexerTopK(
        sample_input_values=input_values,
        sample_seq_lens=seq_lens,
        top_k=2,
        next_n=2,
        return_val=True,
    )
    assert op.check_support()
    indices, values = op.compile().execute(input_values, seq_lens)

    expected_values, expected_indices = torch.topk(input_values[:2, :3], 2, dim=1)
    torch.testing.assert_close(indices[:2], expected_indices.to(torch.int32))
    torch.testing.assert_close(values[:2], expected_values)
    expected_values, expected_indices = torch.topk(input_values[2:, :4], 2, dim=1)
    torch.testing.assert_close(indices[2:], expected_indices.to(torch.int32))
    torch.testing.assert_close(values[2:], expected_values)

    wrapped = DSA.indexer_top_k_wrapper(
        input_values, seq_lens, top_k=2, next_n=2, return_val=False
    )
    assert wrapped["values"] is None
    torch.testing.assert_close(wrapped["indices"], indices)

    with pytest.raises(ValueError, match="next_n"):
        DSA.indexer_top_k_wrapper(input_values, seq_lens, top_k=2, next_n=1)


def test_mudnn_frontend_cudnn_dsa_upstream_package_path_aliases():
    torch = pytest.importorskip("torch")
    from mudnn import DSA
    from mudnn.deepseek_sparse_attention import (
        DSA as deepseek_dsa,
        indexer_top_k_wrapper as package_topk_wrapper,
    )
    from mudnn.deepseek_sparse_attention.indexer_forward.api import (
        IndexerForward,
        indexer_forward_wrapper,
    )
    from mudnn.deepseek_sparse_attention.indexer_top_k.api import IndexerTopK
    from mudnn.deepseek_sparse_attention.score_recompute.api import (
        DenseIndexerScoreRecompute,
        dense_indexer_score_recompute_wrapper,
    )

    input_values = torch.tensor(
        [[0.5, 2.0, -1.0], [3.0, 1.0, 2.5]], dtype=torch.float32
    )
    seq_lens = torch.tensor([3, 3], dtype=torch.int32)

    op = IndexerTopK(input_values, seq_lens, 2)
    indices, values = op.compile().execute(input_values, seq_lens)
    wrapped = package_topk_wrapper(input_values, seq_lens, top_k=2)
    torch.testing.assert_close(indices, wrapped["indices"])
    torch.testing.assert_close(values, wrapped["values"])
    torch.testing.assert_close(
        deepseek_dsa.indexer_top_k_wrapper(input_values, seq_lens, top_k=2)["indices"],
        DSA.indexer_top_k_wrapper(input_values, seq_lens, top_k=2)["indices"],
    )

    local_indices = torch.tensor(
        [[[0, 2, -1], [1, -1, 3]], [[2, -1, 0], [-1, -1, 1]]], dtype=torch.int32
    )
    expected_global = torch.tensor(
        [[[0, 2, -1], [1, -1, 3]], [[6, -1, 4], [-1, -1, 5]]], dtype=torch.int32
    )
    torch.testing.assert_close(
        deepseek_dsa.local_to_global_wrapper(local_indices, seqlen_k=4)["indices"],
        expected_global,
    )

    thd_local = torch.tensor([[0, 2, -1], [1, -1, 3], [0, -1, 2]], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 5, 9], dtype=torch.int32)
    expected_thd = torch.tensor([[0, 2, -1], [1, -1, 3], [5, -1, 7]], dtype=torch.int32)
    torch.testing.assert_close(
        deepseek_dsa.local_to_global_wrapper(
            thd_local,
            seqlen_k=0,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
        )["indices"],
        expected_thd,
    )

    compact = deepseek_dsa.compactify_wrapper(expected_global)
    torch.testing.assert_close(
        compact["indices"],
        torch.tensor(
            [[[0, 2, -1], [1, 3, -1]], [[6, 4, -1], [5, -1, -1]]],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        compact["topk_length"], torch.tensor([2, 2, 2, 1], dtype=torch.int32)
    )

    q = torch.tensor([[[[0.5, -0.25]], [[-0.5, 0.75]]]], dtype=torch.float32)
    k = torch.tensor([[[[1.0, 0.5]], [[-0.25, 0.75]]]], dtype=torch.float32)
    w = torch.tensor([[[0.5], [1.5]]], dtype=torch.float32)
    torch.testing.assert_close(
        indexer_forward_wrapper(q, k, w, ratio=1)["scores"],
        DSA.indexer_forward_wrapper(q, k, w, ratio=1)["scores"],
    )
    torch.testing.assert_close(
        dense_indexer_score_recompute_wrapper(q, k, w, ratio=1)["out"],
        DSA.dense_indexer_score_recompute_wrapper(q, k, w, ratio=1)["out"],
    )
    assert IndexerForward is DSA.IndexerForward
    assert DenseIndexerScoreRecompute is DSA.DenseIndexerScoreRecompute


def test_mudnn_frontend_cudnn_dsa_class_api_torch_fallbacks():
    torch = pytest.importorskip("torch")
    from mudnn import DSA
    from mudnn.deepseek_sparse_attention.indexer_backward.api import (
        DenseIndexerBackward,
        IndexerBackward,
    )
    from mudnn.deepseek_sparse_attention.score_recompute.api import (
        DenseAttnScoreRecompute,
        DenseIndexerScoreRecompute,
        SparseAttnScoreRecompute,
        SparseIndexerScoreRecompute,
    )

    q = torch.tensor(
        [
            [
                [[0.5, -0.25], [1.0, 0.25]],
                [[-0.5, 0.75], [0.25, 1.5]],
            ]
        ],
        dtype=torch.float32,
    )
    k_dense = torch.tensor(
        [[[[1.0, 0.5]], [[-0.25, 0.75]], [[0.5, -1.0]]]], dtype=torch.float32
    )
    k_sparse = k_dense[:, :, 0, :]
    weights = torch.tensor([[[0.5, 1.25], [1.5, -0.5]]], dtype=torch.float32)
    lse = torch.tensor([[[0.25, -0.5], [0.75, 0.125]]], dtype=torch.float32)
    topk = torch.tensor([[[0, 2], [1, -1]]], dtype=torch.int32)

    scores = torch.empty(q.shape[0], q.shape[1], k_dense.shape[1])
    op = DSA.IndexerForward(q, k_dense, weights, scores, ratio=1)
    assert op.check_support()
    op.compile().execute(q, k_dense, weights, scores)
    torch.testing.assert_close(
        scores, DSA.indexer_forward_wrapper(q, k_dense, weights, ratio=1)["scores"]
    )

    sparse_predict = torch.empty_like(topk, dtype=torch.float32)
    sparse_op = SparseIndexerScoreRecompute(q, k_sparse, weights, topk, sparse_predict)
    assert sparse_op.check_support()
    result = sparse_op.compile().execute(q, k_sparse, weights, topk, out=sparse_predict)
    assert result is sparse_predict
    torch.testing.assert_close(
        sparse_predict,
        DSA.sparse_indexer_score_recompute_wrapper(q, k_sparse, weights, topk)[
            "predict"
        ],
    )

    sparse_target = torch.empty_like(sparse_predict)
    sparse_attn_op = SparseAttnScoreRecompute(
        q, k_sparse, lse, topk, sparse_target, 0.5
    )
    sparse_attn_op.compile().execute(q, k_sparse, lse, topk, out=sparse_target)
    torch.testing.assert_close(
        sparse_target,
        DSA.sparse_attn_score_recompute_wrapper(q, k_sparse, lse, topk, 0.5)["target"],
    )

    dense_out = torch.empty_like(scores)
    dense_denom = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32)
    dense_op = DenseIndexerScoreRecompute(
        q, k_dense, weights, dense_out, dense_denom, ratio=1
    )
    actual_out, actual_denom = dense_op.compile().execute(
        q, k_dense, weights, out=dense_out, denom_out=dense_denom
    )
    assert actual_out is dense_out
    assert actual_denom is dense_denom
    expected_dense = DSA.dense_indexer_score_recompute_wrapper(
        q, k_dense, weights, ratio=1
    )
    torch.testing.assert_close(dense_out, expected_dense["out"])
    torch.testing.assert_close(dense_denom, expected_dense["denom"])

    dense_attn_out = torch.empty_like(scores)
    dense_attn_denom = torch.empty_like(dense_denom)
    dense_attn_op = DenseAttnScoreRecompute(
        q, k_dense, lse, dense_attn_out, dense_attn_denom, 0.5, ratio=1
    )
    dense_attn_op.compile().execute(
        q, k_dense, lse, out=dense_attn_out, denom_out=dense_attn_denom
    )
    expected_dense_attn = DSA.dense_attn_score_recompute_wrapper(
        q, k_dense, lse, 0.5, ratio=1
    )
    torch.testing.assert_close(dense_attn_out, expected_dense_attn["out"])
    torch.testing.assert_close(dense_attn_denom, expected_dense_attn["denom"])

    index_score = DSA.sparse_indexer_score_recompute_wrapper(
        q, k_sparse, weights, topk
    )["predict"]
    attn_score = torch.tensor([[[0.25, 0.75], [1.0, 0.0]]], dtype=torch.float32)
    d_q = torch.empty_like(q)
    d_w = torch.empty_like(weights)
    d_k = torch.empty_like(k_sparse)
    backward_op = IndexerBackward(
        q, weights, k_sparse, d_q, d_w, d_k, attn_score, index_score, topk
    )
    backward_op.compile().execute(
        q, weights, k_sparse, d_q, d_w, d_k, attn_score, index_score, topk
    )
    expected_backward = DSA.indexer_backward_wrapper(
        q, weights, k_sparse, attn_score, index_score, topk
    )
    torch.testing.assert_close(d_q, expected_backward["d_index_q"])
    torch.testing.assert_close(d_w, expected_backward["d_weights"])
    torch.testing.assert_close(d_k, expected_backward["d_index_k"])

    dense_back_out = DSA.dense_indexer_score_recompute_wrapper(
        q, k_dense, weights, ratio=1
    )
    dense_d_q = torch.empty_like(q)
    dense_d_w = torch.empty_like(weights)
    dense_d_k = torch.empty_like(k_sparse)
    if _dense_indexer_backward_kernel_available():
        # The TileLang MUSA kernel takes 16-bit SQMMA operands and an fp32 dK
        # accumulator.
        dense_q = q.to(torch.bfloat16)
        dense_k = k_sparse.to(torch.bfloat16)
        dense_w = weights.to(torch.bfloat16)
        dense_d_q = torch.empty_like(dense_q)
        dense_d_w = torch.empty_like(dense_w)
        dense_d_k = torch.empty(
            tuple(dense_k.shape), dtype=torch.float32, device=dense_k.device
        )
        dense_back_op = DenseIndexerBackward(
            dense_q,
            dense_w,
            dense_k,
            dense_d_q,
            dense_d_w,
            dense_d_k,
            dense_back_out["out"],
            dense_back_out["denom"],
            dense_back_out["out"],
            dense_back_out["denom"],
            ratio=1,
        )
        dense_back_op.compile().execute(
            dense_q,
            dense_w,
            dense_k,
            dense_d_q,
            dense_d_w,
            dense_d_k,
            dense_back_out["out"],
            dense_back_out["denom"],
            dense_back_out["out"],
            dense_back_out["denom"],
        )
        assert torch.isfinite(dense_d_q).all()
        assert torch.isfinite(dense_d_w).all()
        assert torch.isfinite(dense_d_k).all()
    else:
        # ``DenseIndexerBackward`` dispatches to the TileLang MUSA kernel
        # factory instead of silently falling back to a torch reference. Off a
        # MUSA lane the plan declines: either the factory reports the missing
        # tilelang_musa install, or the fp32 fixture below is already rejected
        # by check_support() because the kernel takes 16-bit SQMMA operands.
        # The numeric comparison belongs to the MUSA lane.
        dense_back_op = DenseIndexerBackward(
            q,
            weights,
            k_sparse,
            dense_d_q,
            dense_d_w,
            dense_d_k,
            dense_back_out["out"],
            dense_back_out["denom"],
            dense_back_out["out"],
            dense_back_out["denom"],
            ratio=1,
        )
        with pytest.raises((NotImplementedError, ValueError)):
            dense_back_op.compile()


def test_mudnn_frontend_cudnn_dsa_score_recompute_torch_fallbacks():
    torch = pytest.importorskip("torch")
    from mudnn import DSA

    q = torch.tensor(
        [
            [
                [[0.5, -0.25], [1.0, 0.25]],
                [[-0.5, 0.75], [0.25, 1.5]],
            ]
        ],
        dtype=torch.float32,
    )
    k_dense = torch.tensor(
        [[[[1.0, 0.5]], [[-0.25, 0.75]], [[0.5, -1.0]]]], dtype=torch.float32
    )
    weights = torch.tensor([[[0.5, 1.25], [1.5, -0.5]]], dtype=torch.float32)
    lse = torch.tensor([[[0.25, -0.5], [0.75, 0.125]]], dtype=torch.float32)

    # Ratio-causal alignment is the upstream top-left contract: row ``q`` sees
    # the first ``floor((q_causal_offset + q + 1) / ratio)`` compressed KV
    # positions.  A decode-shaped query (S_q < S_k) therefore has to pass the
    # global start of its first row explicitly, exactly like the upstream
    # kernels and the training caller do.
    q_causal_offsets = torch.tensor(
        [k_dense.shape[1] - q.shape[1]], dtype=torch.int32
    )
    indexer = DSA.indexer_forward_wrapper(
        q, k_dense, weights, ratio=1, q_causal_offsets=q_causal_offsets
    )["scores"]
    k_expanded = k_dense.repeat_interleave(2, dim=2)
    expected_indexer = torch.einsum("bqhd,bkhd->bqhk", q, k_expanded)
    expected_indexer = torch.relu(expected_indexer)
    expected_indexer = (expected_indexer * weights.unsqueeze(-1)).sum(dim=2)
    q_global_start = int(q_causal_offsets[0])
    q_pos = torch.arange(q.shape[1]).view(-1, 1)
    k_pos = torch.arange(k_dense.shape[1]).view(1, -1)
    valid = k_pos < (q_global_start + q_pos + 1)
    expected_indexer = expected_indexer.masked_fill(~valid.unsqueeze(0), -torch.inf)
    torch.testing.assert_close(indexer, expected_indexer)
    # The default (no offsets) is the top-left origin, so the ratio-causal
    # window of local row q ends at KV index q (ratio=1).
    default_indexer = DSA.indexer_forward_wrapper(
        q, k_dense, weights, ratio=1
    )["scores"]
    unmasked = (torch.relu(torch.einsum("bqhd,bkhd->bqhk", q, k_expanded)) * weights.unsqueeze(-1)).sum(dim=2)
    expected_default = unmasked.masked_fill(k_pos >= (q_pos + 1), -torch.inf)
    torch.testing.assert_close(default_indexer, expected_default)

    dense = DSA.dense_indexer_score_recompute_wrapper(
        q, k_dense, weights, ratio=1, q_causal_offsets=q_causal_offsets
    )
    torch.testing.assert_close(
        dense["out"],
        expected_indexer.masked_fill(~torch.isfinite(expected_indexer), 0.0),
    )
    torch.testing.assert_close(
        dense["denom"], torch.logsumexp(expected_indexer, dim=-1)
    )

    dense_attn = DSA.dense_attn_score_recompute_wrapper(
        q, k_dense, lse, 0.5, ratio=1, q_causal_offsets=q_causal_offsets
    )
    expected_prob = torch.exp(
        torch.einsum("bqhd,bkhd->bqhk", q, k_expanded) * 0.5 - lse.unsqueeze(-1)
    ).sum(dim=2)
    expected_prob = expected_prob.masked_fill(~valid.unsqueeze(0), 0.0)
    torch.testing.assert_close(dense_attn["out"], expected_prob)
    torch.testing.assert_close(dense_attn["denom"], expected_prob.sum(dim=-1))

    k_sparse = k_dense[:, :, 0, :]
    topk = torch.tensor([[[0, 2], [1, -1]]], dtype=torch.int32)
    sparse_indexer = DSA.sparse_indexer_score_recompute_wrapper(
        q, k_sparse, weights, topk
    )["predict"]
    idx = topk.clamp(min=0).long()
    gathered = torch.gather(
        k_sparse.unsqueeze(1).expand(1, 2, 3, 2),
        2,
        idx.unsqueeze(-1).expand(1, 2, 2, 2),
    )
    sparse_scores = torch.einsum("bqhd,bqtd->bqht", q, gathered)
    sparse_scores = (torch.relu(sparse_scores) * weights.unsqueeze(-1)).sum(dim=2)
    sparse_scores = sparse_scores.masked_fill(topk < 0, -torch.inf)
    torch.testing.assert_close(sparse_indexer, torch.softmax(sparse_scores, dim=-1))

    offsets = torch.tensor([0, 3], dtype=torch.int32).view(2, 1, 1)
    global_topk = torch.where(topk.expand(2, -1, -1) >= 0, topk + offsets, topk)
    q2 = q.expand(2, -1, -1, -1).clone()
    k2 = k_sparse.expand(2, -1, -1).clone()
    weights2 = weights.expand(2, -1, -1).clone()
    local_result = DSA.sparse_indexer_score_recompute_wrapper(
        q2, k2, weights2, topk.expand(2, -1, -1).clone()
    )["predict"]
    global_result = DSA.sparse_indexer_score_recompute_wrapper(
        q2, k2, weights2, global_topk, topk_indices_global=True
    )["predict"]
    torch.testing.assert_close(global_result, local_result)

    sparse_attn = DSA.sparse_attn_score_recompute_wrapper(q, k_sparse, lse, topk, 0.5)[
        "target"
    ]
    sparse_prob = torch.exp(
        torch.einsum("bqhd,bqtd->bqht", q, gathered) * 0.5 - lse.unsqueeze(-1)
    ).sum(dim=2)
    sparse_prob = sparse_prob.masked_fill(topk < 0, 0.0)
    sparse_prob = sparse_prob / sparse_prob.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    torch.testing.assert_close(sparse_attn, sparse_prob)


def test_mudnn_frontend_cudnn_dsa_indexer_backward_torch_fallbacks():
    torch = pytest.importorskip("torch")
    from mudnn import DSA

    index_q = torch.tensor(
        [[[[0.5, -0.25], [1.0, 0.25]], [[-0.5, 0.75], [0.25, 1.5]]]],
        dtype=torch.float32,
    )
    weights = torch.tensor([[[0.5, 1.25], [1.5, -0.5]]], dtype=torch.float32)
    index_k = torch.tensor(
        [[[1.0, 0.5], [-0.25, 0.75], [0.5, -1.0]]], dtype=torch.float32
    )
    topk = torch.tensor([[[0, 2], [1, -1]]], dtype=torch.int32)
    attn_score = torch.tensor([[[0.25, 0.75], [1.0, 0.0]]], dtype=torch.float32)
    index_score = DSA.sparse_indexer_score_recompute_wrapper(
        index_q, index_k, weights, topk
    )["predict"]

    actual = DSA.indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        attn_score,
        index_score,
        topk,
        loss_coeff=2.0,
        grad_loss=1.0,
    )

    q_ref = index_q.detach().clone().requires_grad_(True)
    w_ref = weights.detach().clone().requires_grad_(True)
    k_ref = index_k.detach().clone().requires_grad_(True)
    predict = DSA.sparse_indexer_score_recompute_wrapper(q_ref, k_ref, w_ref, topk)[
        "predict"
    ]
    loss = -(
        attn_score * torch.log(predict.clamp(min=torch.finfo(torch.float32).tiny))
    ).sum()
    expected = torch.autograd.grad(loss, (q_ref, w_ref, k_ref))
    torch.testing.assert_close(actual["d_index_q"], expected[0])
    torch.testing.assert_close(actual["d_weights"], expected[1])
    torch.testing.assert_close(actual["d_index_k"], expected[2])

    dense_attn = torch.tensor([[[0.3, 0.7, 0.0], [0.2, 0.6, 0.0]]], dtype=torch.float32)
    dense_l1 = dense_attn.sum(dim=-1)
    dense = DSA.dense_indexer_backward_wrapper(
        index_q,
        weights,
        index_k,
        dense_attn,
        dense_l1,
        dense_attn,
        dense_l1,
        loss_coeff=2.0,
        grad_loss=1.0,
    )
    assert dense["d_index_q"].shape == index_q.shape
    assert dense["d_weights"].shape == weights.shape
    assert dense["d_index_k"].shape == index_k.shape
    assert torch.isfinite(dense["d_index_q"]).all()
    assert torch.isfinite(dense["d_weights"]).all()
    assert torch.isfinite(dense["d_index_k"]).all()
