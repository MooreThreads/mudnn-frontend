# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Torch fallbacks for the NSA APIs."""

import pytest


def test_mudnn_frontend_cudnn_nsa_torch_fallbacks():
    torch = pytest.importorskip("torch")
    import mudnn
    from mudnn import NSA
    from mudnn.native_sparse_attention import NSA as native_nsa
    from mudnn.native_sparse_attention.top_k.api import topk_reduction_wrapper

    assert "NSA" in mudnn.__all__
    assert native_nsa.TopKReduction is NSA.TopKReduction

    q = torch.tensor(
        [[[[0.5, -0.25], [1.0, 0.25]], [[-0.5, 0.75], [0.25, 1.5]]]],
        dtype=torch.float32,
    )
    k = torch.tensor([[[[1.0, 0.5], [-0.25, 0.75]]]], dtype=torch.float32)
    v = torch.tensor([[[[0.5, -1.0], [1.25, 0.25]]]], dtype=torch.float32)

    swa = NSA.sliding_window_attention_wrapper(
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        left_bound=1,
        right_bound=0,
        attn_scale=0.5,
    )
    assert swa["o_tensor"].shape == q.shape
    assert swa["stats_tensor"].shape == (1, 2, 2, 1)

    comp = NSA.compression_attention_wrapper(
        q_tensor=q,
        k_tensor=k,
        v_tensor=v,
        enable_lse=True,
        scale_softmax=0.5,
    )
    assert comp["o_tensor"].shape == q.shape
    assert comp["lse_tensor"].shape == (1, 2, 2)

    q_thd = q.permute(0, 2, 1, 3).reshape(2, 2, 2)
    k_thd = k.permute(0, 2, 1, 3).reshape(2, 1, 2)
    v_thd = v.permute(0, 2, 1, 3).reshape(2, 1, 2)
    block_indices = torch.zeros((2, 1, 1), dtype=torch.int32)
    block_counts = torch.ones((2, 1), dtype=torch.int32)
    seqlens = torch.tensor([0, 2], dtype=torch.int32)
    selection = NSA.selection_attention_wrapper(
        q_tensor=q_thd,
        k_tensor=k_thd,
        v_tensor=v_thd,
        block_indices_tensor=block_indices,
        block_counts_tensor=block_counts,
        cum_seqlen_q_tensor=seqlens,
        cum_seqlen_k_tensor=seqlens,
        block_size=2,
        scale_softmax=0.5,
    )
    assert selection["o_tensor"].shape == q_thd.shape
    assert selection["l_tensor"].shape == (2, 2, 1)
    assert selection["m_tensor"].shape == (2, 2, 1)

    topk = topk_reduction_wrapper(
        q_tensor=q,
        k_tensor=k,
        lse_tensor=comp["lse_tensor"],
        k_value=3,
    )
    assert topk["topk_scores_tensor"].shape == (1, 1, 2, 3)
    assert topk["topk_indices_tensor"].shape == (1, 1, 2, 3)
    assert torch.isneginf(topk["topk_scores_tensor"]).all()
    torch.testing.assert_close(
        topk["topk_indices_tensor"],
        torch.full((1, 1, 2, 3), -1, dtype=torch.int32),
    )
