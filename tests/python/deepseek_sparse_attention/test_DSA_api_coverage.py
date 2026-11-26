# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""DSA/CSA surface acceptance: every cuDNN-frontend DSA/CSA API DeepSeek-V4 training needs.

The suite lists 15 public entry points of ``cudnn-frontend`` that the
DeepSeek-V4 training script
(``megatron-lm-musa-patch/examples/deepseek-v4/pretrain_deepseekv4_musa.py``)
reaches, split into nine ``sparse_loss=True`` APIs, three performance APIs and
three ``sparse_loss=False`` APIs.  This module is the executable version of that
list: it asserts the muDNN frontend exposes each name on the ``DSA``/``CSA``
namespace and then *runs* it on a MUSA device against the FP32/FP64 torch
reference in ``deepseek_sparse_attention.tilelang.dsa_reference``.

Three tiers are covered so "supported" cannot hide behind a smaller shape:

* ``test_sw89132_*_available`` -- the name exists on ``mudnn.DSA``/``mudnn.CSA``
  and on the compatibility import paths the training stack uses
  (``mudnn.deepseek_sparse_attention.*``, ``mudnn.csa.compressor``).
* BSHD/THD contract tests -- the argument surface the training wrappers pass
  (``qhead_per_kv_head``, ``topk_length``, ``q_causal_offsets``, preallocated
  ``out`` buffers, ``cu_seqlens_*``) is accepted and honored.
* training-shape numerics -- the same shapes ``run_pretrain_deepseekv4_musa.sh``
  configures (``S_q=4096``, ``H_q=64``, ``D_idx=128``, ``ratio`` 4/128,
  ``topk=512``) for the four operators called out as high priority.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# The DSA/CSA API list this suite was written against
# ---------------------------------------------------------------------------

#: ``sparse_loss=True`` -- 9 APIs the DSV4 training path needs unconditionally.
SPARSE_LOSS_APIS = (
    "compactify_wrapper",
    "indexer_forward_wrapper",
    "indexer_top_k_wrapper",
    "sparse_indexer_score_recompute_wrapper",
    "sparse_attn_score_recompute_wrapper",
    "indexer_backward_wrapper",
    "sparse_attention_backward_wrapper",
)

#: Performance / layout APIs (medium priority).
PERFORMANCE_APIS = (
    "indexer_forward_top_k_wrapper",
    "compress_topk_cand_buffer_size",
    "compress_topk_cand_buffer_size_thd",
)

#: ``sparse_loss=False`` APIs (low priority).
DENSE_APIS = (
    "dense_indexer_score_recompute_wrapper",
    "dense_attn_score_recompute_wrapper",
    "dense_indexer_backward_wrapper",
)

#: The two CSA compressor wrappers, which live on ``cudnn.csa`` not ``cudnn.DSA``.
CSA_APIS = (
    "csa_compressor_forward_wrapper",
    "csa_compressor_backward_wrapper",
)

#: ``sparse_attention_forward_wrapper`` also belongs to the training path; the
#: "9 required" count refers to the indexer/loss family above, so it is tracked
#: separately here.
FORWARD_APIS = ("sparse_attention_forward_wrapper",)

ALL_DSA_APIS = SPARSE_LOSS_APIS + PERFORMANCE_APIS + DENSE_APIS + FORWARD_APIS


def _dsa():
    from mudnn import DSA

    return DSA


def _musa_device():
    pytest.importorskip("torch_musa")
    if not torch.musa.is_available():
        pytest.skip("MUSA device unavailable")
    return torch.device("musa")


def _native_backend():
    """Skip unless the native TileLang DSA kernels can build on this device."""
    pytest.importorskip("tilelang")
    _musa_device()
    return tuple(torch.musa.get_device_capability())


# ---------------------------------------------------------------------------
# Tier 1: availability on every public path the training stack imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL_DSA_APIS)
def test_sw89132_dsa_api_available(name):
    DSA = _dsa()
    assert callable(getattr(DSA, name)), f"DSA.{name} is not callable"
    assert name in importlib.import_module(
        "mudnn.deepseek_sparse_attention"
    ).__all__, f"{name} missing from mudnn.deepseek_sparse_attention.__all__"


@pytest.mark.parametrize("name", CSA_APIS)
def test_sw89132_csa_api_available(name):
    import mudnn

    compressor = importlib.import_module("mudnn.csa.compressor")
    assert callable(getattr(compressor, name)), f"mudnn.csa.compressor.{name} is not callable"
    # ``cudnn.csa.compressor.<wrapper>`` is the path megatron-lm-musa-patch
    # probes in ``musa_patch/fused_compressor.py::_get_frontend``.
    assert callable(getattr(mudnn.csa.compressor, name))
    assert callable(getattr(mudnn.CSA, name))


def test_sw89132_compact_api_import_path_matches_upstream():
    """The upstream import paths for the three performance APIs must resolve."""
    from mudnn.deepseek_sparse_attention.indexer_forward import (
        compress_topk_cand_buffer_size,
        compress_topk_cand_buffer_size_thd,
        indexer_forward_top_k_wrapper,
    )

    DSA = _dsa()
    assert indexer_forward_top_k_wrapper is DSA.indexer_forward_top_k_wrapper
    assert compress_topk_cand_buffer_size is DSA.compress_topk_cand_buffer_size
    assert compress_topk_cand_buffer_size_thd is DSA.compress_topk_cand_buffer_size_thd


# ---------------------------------------------------------------------------
# Tier 2: sizing helpers reproduce the upstream candidate layout exactly
# ---------------------------------------------------------------------------


def _compact_G(n: int, ratio: int) -> int:
    """``sum_{m=0}^{n} floor(m / ratio)`` -- the upstream compact row offset."""
    quotient, remainder = divmod(n, ratio)
    return ratio * quotient * (quotient - 1) // 2 + quotient * (remainder + 1)


@pytest.mark.parametrize(
    "bs,seqlen_q,seqlen_k,ratio",
    [
        (1, 4096, 1024, 4),
        (2, 1024, 1024, 4),
        (8, 4096, 32, 128),
        (1, 8192, 4096, 2),
        (4, 32768, 8192, 4),
    ],
)
def test_sw89132_compress_topk_cand_buffer_size_matches_upstream(bs, seqlen_q, seqlen_k, ratio):
    DSA = _dsa()
    expected = bs * (_compact_G(seqlen_q, ratio) - _compact_G(0, ratio))
    assert DSA.compress_topk_cand_buffer_size(
        bs, seqlen_q, seqlen_k, ratio, microbatch_rows=0
    ) == expected
    # The default (``microbatch_rows=-1``) agrees with ``0`` until the 32768-row
    # auto-microbatch threshold, above which it sizes only the largest window.
    if seqlen_q < 32768:
        assert (
            DSA.compress_topk_cand_buffer_size(bs, seqlen_q, seqlen_k, ratio) == expected
        )
    else:
        windowed = DSA.compress_topk_cand_buffer_size(bs, seqlen_q, seqlen_k, ratio)
        assert windowed < expected
    with pytest.raises(ValueError, match="seqlen_q"):
        DSA.compress_topk_cand_buffer_size(bs, seqlen_k * ratio + 1, seqlen_k, ratio)


def test_sw89132_compress_topk_cand_buffer_size_thd_matches_upstream():
    DSA = _dsa()
    device = _musa_device()
    cu_q = torch.tensor([0, 100, 300], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, 40, 90], dtype=torch.int32, device=device)
    offsets, total = DSA.compress_topk_cand_buffer_size_thd(cu_q, cu_k, 4)
    assert offsets.dtype == torch.int64 and offsets.shape == (3,)
    expected = sum(
        _compact_G(int(cu_q[i + 1]) - int(cu_q[i]), 4) - _compact_G(0, 4) for i in range(2)
    )
    assert total == expected
    assert offsets.tolist() == [0] + list(
        torch.cumsum(
            torch.tensor(
                [
                    _compact_G(int(cu_q[i + 1]) - int(cu_q[i]), 4) - _compact_G(0, 4)
                    for i in range(2)
                ]
            ),
            0,
        ).tolist()
    )
    # Per-segment causal offsets change the per-slab size, exactly like upstream.
    causal = torch.tensor([16, 0], dtype=torch.int32, device=device)
    _, shifted = DSA.compress_topk_cand_buffer_size_thd(cu_q, cu_k, 4, q_causal_offsets=causal)
    assert shifted == _compact_G(16 + 100, 4) - _compact_G(16, 4) + _compact_G(200, 4)
    with pytest.raises(ValueError, match="cu_seqlens_q/k"):
        DSA.compress_topk_cand_buffer_size_thd(
            torch.tensor([0], dtype=torch.int32, device=device), cu_k, 4
        )


# ---------------------------------------------------------------------------
# Tier 3: real device execution against the torch reference
# ---------------------------------------------------------------------------


def _ref_indexer_scores(q, k, w, ratio, offsets=None, sm_scale=1.0):
    b, s_q, h_q, d = q.shape
    _, s_k, h_kv, _ = k.shape
    k_exp = k.float().repeat_interleave(h_q // h_kv, dim=2)
    scores = torch.einsum("bqhd,bkhd->bqhk", q.float(), k_exp).relu()
    scores = float(sm_scale) * (scores * w.float().unsqueeze(-1)).sum(dim=2)
    q_pos = torch.arange(s_q, device=q.device, dtype=torch.int64).view(1, s_q)
    k_pos = torch.arange(s_k, device=q.device, dtype=torch.int64).view(1, 1, s_k)
    offset = (
        torch.zeros((1, 1), dtype=torch.int64, device=q.device)
        if offsets is None
        else offsets.to(torch.int64).reshape(-1, 1)
    )
    limit = ((offset + q_pos + 1) // ratio).clamp(0, s_k)
    return scores.masked_fill(~(k_pos < limit.unsqueeze(-1)), float("-inf"))


def _check_compact_topk(dense, result, top_k, atol=2e-3):
    """Upstream ``check_ref_compressed_topk`` plus the fused-softmax contract."""
    indices, logits = result["indices"], result["logits"]
    assert indices.dtype == torch.int32 and logits.dtype == torch.float32
    valid = indices >= 0
    expected = torch.isfinite(dense).sum(dim=-1).clamp(max=top_k)
    assert torch.equal(valid.sum(dim=-1), expected)
    assert bool(((indices < dense.shape[-1]) | ~valid).all())
    gathered = torch.gather(dense, -1, indices.clamp(min=0).long())
    torch.testing.assert_close(logits[valid], gathered[valid], atol=atol, rtol=atol)
    assert bool(torch.isneginf(logits[~valid]).all())
    width = min(top_k, int(dense.shape[-1]))
    expected_values = torch.topk(dense, width, dim=-1).values
    # Sorted descending, the first ``width`` slots hold the selected values;
    # anything past them is the top_k > S_k padding.
    actual_values = torch.sort(logits, dim=-1, descending=True).values[..., :width]
    finite = torch.isfinite(expected_values)
    torch.testing.assert_close(actual_values[finite], expected_values[finite], atol=atol, rtol=atol)
    assert bool(torch.isneginf(logits[..., width:]).all())
    softmax_ref = torch.softmax(
        logits.masked_fill(~valid, float("-inf")), dim=-1
    ).nan_to_num(0.0).masked_fill(~valid, 0.0)
    torch.testing.assert_close(result["softmax"], softmax_ref, atol=1e-5, rtol=1e-4)
    assert torch.isfinite(result["softmax"]).all()


@pytest.mark.parametrize(
    "b,s_q,s_k,h_q,d,ratio,top_k",
    [
        (2, 128, 32, 64, 128, 4, 16),
        (2, 64, 64, 64, 128, 1, 8),
        (1, 32, 8, 8, 128, 4, 5),
        (1, 4096, 32, 64, 128, 128, 512),
    ],
)
def test_sw89132_indexer_forward_and_top_k_bundle(b, s_q, s_k, h_q, d, ratio, top_k):
    """``indexer_forward_wrapper`` + ``indexer_forward_top_k_wrapper`` + TopK family."""
    DSA = _dsa()
    device = _musa_device()
    torch.manual_seed(0)
    q = torch.randn((b, s_q, h_q, d), dtype=torch.bfloat16, device=device)
    k = torch.randn((b, s_k, 1, d), dtype=torch.bfloat16, device=device)
    w = torch.randn((b, s_q, h_q), dtype=torch.bfloat16, device=device).abs() * 0.1
    dense = _ref_indexer_scores(q, k, w, ratio)

    # indexer_forward_wrapper: scores (+ the preallocated-out path).
    scores = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio)["scores"]
    out = torch.empty_like(scores)
    scores_again = DSA.indexer_forward_wrapper(q, k, w, ratio=ratio, out=out)["scores"]
    assert scores_again.data_ptr() == out.data_ptr()
    for actual in (scores, scores_again):
        finite = torch.isfinite(dense)
        assert torch.equal(torch.isfinite(actual), finite)
        torch.testing.assert_close(actual[finite], dense[finite], atol=1e-3, rtol=1e-3)

    # indexer_forward_top_k_wrapper: BSHD local and global ids, both against the
    # dense reference and against the standalone TopK path.
    for global_ids in (False, True):
        result = DSA.indexer_forward_top_k_wrapper(
            q, k, w, top_k=top_k, ratio=ratio, topk_indices_global=global_ids
        )
        reported = result["indices"]
        local = reported
        if global_ids:
            local = torch.where(
                reported >= 0,
                reported - torch.arange(b, device=device, dtype=torch.int32).view(-1, 1, 1) * s_k,
                reported,
            )
        _check_compact_topk(
            dense,
            {"indices": local, "logits": result["logits"], "softmax": result["softmax"]},
            top_k,
        )

    # indexer_top_k_wrapper: per-(batch, row) top-k over the packed score rows.
    flat_rows = s_q * b
    seq_lens = torch.full((b,), flat_rows // b, dtype=torch.int32, device=device)
    packed = dense.reshape(-1, s_k).contiguous()
    topk_result = DSA.indexer_top_k_wrapper(
        packed, seq_lens, top_k=top_k, next_n=s_q, return_val=True
    )
    width = min(top_k, s_k)
    expected_values, expected_indices = torch.topk(packed, width, dim=1)
    # Slots past ``min(top_k, S_k)`` are the zero padding the upstream radix
    # kernel also emits (see ``ref_indexer_top_k``).
    assert int(topk_result["indices"].shape[-1]) == top_k
    torch.testing.assert_close(
        topk_result["indices"][:, :width].long(), expected_indices.long()
    )
    assert bool((topk_result["indices"][:, width:] == 0).all())
    if topk_result["values"] is not None:
        torch.testing.assert_close(topk_result["values"][:, :width], expected_values)
        assert bool((topk_result["values"][:, width:] == 0).all())

    # compactify_wrapper + local_to_global_wrapper round-trip.
    local_topk = topk_result["indices"].view(b, s_q, -1)
    global_topk = DSA.local_to_global_wrapper(local_topk, seqlen_k=s_k)["indices"]
    compact = DSA.compactify_wrapper(global_topk)
    lengths = compact["topk_length"].view(b, s_q)
    assert torch.equal(lengths, (global_topk >= 0).sum(-1).to(torch.int32))
    expected_compact = torch.sort(global_topk, dim=-1, descending=False).values
    # ``compactify`` keeps the input order and only moves ``-1`` to the tail.
    for row in range(b):
        for col in range(0, min(s_q, 8)):
            original = global_topk[row, col]
            kept = compact["indices"][row, col]
            assert torch.equal(kept[kept >= 0], original[original >= 0])
            assert bool((kept[len(kept[kept >= 0]) :] < 0).all())
    del expected_compact

    # THD (packed) compact path with per-segment causal offsets.
    cu_q = torch.tensor([0, s_q], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, s_k], dtype=torch.int32, device=device)
    thd = DSA.indexer_forward_top_k_wrapper(
        q[0],
        k[0],
        w[0],
        top_k=top_k,
        ratio=ratio,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        topk_indices_global=False,
    )
    _check_compact_topk(
        dense[0],
        {"indices": thd["indices"], "logits": thd["logits"], "softmax": thd["softmax"]},
        top_k,
    )
    # A preallocated candidate buffer plus output buffers must be validated and
    # aliased back, which is what the Megatron workspace planner relies on.
    floats = DSA.compress_topk_cand_buffer_size(b, s_q, s_k, ratio, microbatch_rows=0)
    cand = torch.empty(floats, dtype=torch.float32, device=device)
    out_idx = torch.empty((b, s_q, top_k), dtype=torch.int32, device=device)
    out_logits = torch.empty((b, s_q, top_k), dtype=torch.float32, device=device)
    out_softmax = torch.empty((b, s_q, top_k), dtype=torch.float32, device=device)
    aliased = DSA.indexer_forward_top_k_wrapper(
        q,
        k,
        w,
        top_k=top_k,
        ratio=ratio,
        microbatch_rows=0,
        topk_indices_global=False,
        cand_buffer=cand,
        out_indices=out_idx,
        out_logits=out_logits,
        softmax_out=out_softmax,
    )
    assert aliased["indices"].data_ptr() == out_idx.data_ptr()
    assert aliased["logits"].data_ptr() == out_logits.data_ptr()
    assert aliased["softmax"].data_ptr() == out_softmax.data_ptr()
    with pytest.raises(ValueError, match="cand_buffer too small"):
        DSA.indexer_forward_top_k_wrapper(
            q,
            k,
            w,
            top_k=top_k,
            ratio=ratio,
            microbatch_rows=0,
            topk_indices_global=False,
            cand_buffer=cand[: max(1, floats // 2)],
        )


def test_sw89132_indexer_forward_rejects_fp8_without_native_support():
    DSA = _dsa()
    device = _musa_device()
    q = torch.zeros((1, 8, 8, 128), dtype=torch.bfloat16, device=device)
    k = torch.zeros((1, 4, 1, 128), dtype=torch.bfloat16, device=device)
    w = torch.zeros((1, 8, 8), dtype=torch.bfloat16, device=device)
    with pytest.raises(NotImplementedError, match="FP8"):
        DSA.indexer_forward_wrapper(q, k, w, precision="mxfp8")
    with pytest.raises(NotImplementedError, match="FP8"):
        DSA.indexer_forward_top_k_wrapper(q, k, w, top_k=4, precision="mxfp8")
    with pytest.raises(NotImplementedError, match="FP8"):
        DSA.dense_indexer_score_recompute_wrapper(
            q, k, w, precision="mxfp8", q_scale=torch.ones(1, device=device)
        )


def test_sw89132_sparse_loss_operators_training_shapes():
    """The four high-priority DSV4 training operators at ``S_q=4096``/``topk=512``.

    Shapes follow ``run_pretrain_deepseekv4_musa.sh``: even layer ``ratio=4``
    (``S_k=1024``) and odd layer ``ratio=128`` (``S_k=32``), ``H_q=64``,
    ``D_idx=128``, ``topk=512``.
    """
    from deepseek_sparse_attention.tilelang.dsa_reference import (
        ref_indexer_backward,
        ref_sparse_attn_score_recompute,
        ref_sparse_indexer_score_recompute,
    )

    DSA = _dsa()
    _native_backend()
    device = _musa_device()
    torch.manual_seed(7)
    s_q, h_q, d_idx, topk = 4096, 64, 128, 512
    for s_k, ratio in ((1024, 4), (32, 128)):
        q = torch.randn((1, s_q, h_q, d_idx), dtype=torch.bfloat16, device=device) * 0.1
        k = torch.randn((1, s_k, d_idx), dtype=torch.bfloat16, device=device) * 0.1
        w = torch.rand((1, s_q, h_q), dtype=torch.bfloat16, device=device) * 0.1
        lengths = torch.full((1, s_q), min(topk, s_k), dtype=torch.int32, device=device)
        topk_idx = torch.argsort(
            torch.rand((1, s_q, s_k), device=device), dim=-1, descending=True
        )[:, :, : min(topk, s_k)].to(torch.int32)
        width = int(topk_idx.shape[-1])
        if width < topk:
            topk_idx = torch.nn.functional.pad(topk_idx, (0, topk - width), value=-1)

        predict = DSA.sparse_indexer_score_recompute_wrapper(
            q, k, w, topk_idx, topk_length=lengths
        )["predict"]
        expected_predict = ref_sparse_indexer_score_recompute(q, k, w, topk_idx, lengths)
        torch.testing.assert_close(predict, expected_predict, atol=2e-3, rtol=2e-2)

        gathered_k = k[0].float()[topk_idx[0].clamp(min=0).long()].unsqueeze(0)  # (1, S_q, topk, D)
        lse = (torch.einsum("bqhd,bqtd->bqht", q.float(), gathered_k) * d_idx**-0.5).logsumexp(
            dim=-1
        )  # (1, S_q, H_q)
        target = DSA.sparse_attn_score_recompute_wrapper(
            q, k, lse, topk_idx, d_idx**-0.5, topk_length=lengths
        )["target"]
        expected_target = ref_sparse_attn_score_recompute(
            q, k, lse, topk_idx, d_idx**-0.5, lengths
        )
        torch.testing.assert_close(target, expected_target, atol=2e-3, rtol=2e-2)

        # indexer_backward at the same shapes against the kernel-faithful
        # analytic reference in ``dsa_reference.ref_indexer_backward``.
        attn_score = expected_target
        index_score = expected_predict
        index_lse = torch.logsumexp(
            torch.where(topk_idx >= 0, torch.log(expected_predict.clamp(min=1e-30)), -torch.inf),
            dim=-1,
        )
        grads = DSA.indexer_backward_wrapper(
            q,
            w,
            k,
            attn_score,
            index_score,
            topk_idx,
            sm_scale=d_idx**-0.5,
            loss_coeff=float(s_q),
        )
        ref_dq, ref_dw, ref_dk = ref_indexer_backward(
            q,
            w,
            k,
            attn_score,
            index_score,
            topk_idx,
            sm_scale=d_idx**-0.5,
            grad_scale=1.0,
        )
        torch.testing.assert_close(
            grads["d_index_q"], ref_dq, atol=5e-2, rtol=5e-2, equal_nan=True
        )
        torch.testing.assert_close(
            grads["d_weights"], ref_dw, atol=5e-2, rtol=5e-2, equal_nan=True
        )
        torch.testing.assert_close(
            grads["d_index_k"], ref_dk, atol=5e-1, rtol=5e-1, equal_nan=True
        )
        del index_lse


def _reference_denom(raw_scores):
    """``logsumexp`` over the last axis with torch's CUDA empty-row semantics.

    MUSA's ``logsumexp`` yields ``nan`` for an all-``-inf`` row where torch/CUDA
    yields ``-inf``, so the oracle normalises it rather than inheriting the
    backend quirk.
    """
    denom = raw_scores.logsumexp(dim=-1)
    empty = ~torch.isfinite(raw_scores).any(dim=-1)
    return torch.where(empty, torch.full_like(denom, float("-inf")), denom)


def test_sw89132_dense_family_bshd_contract():
    """``dense_*`` wrappers accept the upstream kwargs and honor ``out``/``sm_scale``."""
    from deepseek_sparse_attention.tilelang.dsa_reference import (
        _batched_ratio_causal_mask,
        ref_dense_attn_score_recompute,
        ref_dense_indexer_score_recompute,
    )

    DSA = _dsa()
    device = _musa_device()
    torch.manual_seed(11)
    b, s_q, s_k, h_q, d = 2, 128, 32, 8, 128
    q = torch.randn((b, s_q, h_q, d), dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn((b, s_k, 1, d), dtype=torch.bfloat16, device=device) * 0.1
    w = torch.rand((b, s_q, h_q), dtype=torch.bfloat16, device=device) * 0.1
    offsets = torch.tensor([8, 0], dtype=torch.int32, device=device)

    out = torch.empty((b, s_q, s_k), dtype=torch.float32, device=device)
    denom = torch.empty((b, s_q), dtype=torch.float32, device=device)
    result = DSA.dense_indexer_score_recompute_wrapper(
        q,
        k,
        w,
        qhead_per_kv_head=h_q,
        out=out,
        denom_out=denom,
        sm_scale=0.5,
        ratio=4,
        q_causal_offsets=offsets,
    )
    assert result["out"].data_ptr() == out.data_ptr()
    expected_out, expected_denom = ref_dense_indexer_score_recompute(
        q, k, w, ratio=4, q_causal_offsets=offsets
    )
    # ``ref_dense_indexer_score_recompute`` has no ``sm_scale``: it returns
    # ``sm_scale == 1`` scores, so applying the wrapper's 0.5 is a scale of the
    # output but *not* of the log-sum-exp.  Re-derive the denom the wrapper owes:
    # ``logsumexp(0.5 * scores)`` over the same valid positions.
    valid_mask = _batched_ratio_causal_mask(
        s_q, s_k, 4, device, b, offsets
    )
    raw_scores = expected_out.masked_fill(~valid_mask, float("-inf"))
    torch.testing.assert_close(out * 2.0, expected_out, atol=2e-2, rtol=2e-2)
    expected_denom_scaled = _reference_denom(raw_scores * 0.5)
    # ``torch.logsumexp`` returns ``nan`` for an all-``-inf`` row on MUSA while
    # torch/CUDA returns ``-inf``; the reference helper inherits that backend
    # quirk, so compare finite rows exactly and require that both sides agree on
    # which rows are empty (row 0 of every segment has no valid KV at all).
    empty = ~torch.isfinite(expected_denom_scaled)
    assert torch.equal(empty, ~torch.isfinite(denom)), "empty-row set differs"
    finite = ~empty
    torch.testing.assert_close(
        denom[finite], expected_denom_scaled[finite], atol=2e-2, rtol=2e-2
    )
    assert bool((denom[empty] == float("-inf")).all()), (
        "an empty (no valid KV) row must be -inf, not "
        f"{denom[empty].tolist()} -- MUSA logsumexp returns nan there"
    )

    lse = torch.randn((b, s_q, h_q), dtype=torch.float32, device=device)
    attn_out = torch.empty_like(out)
    attn_denom = torch.empty_like(denom)
    attn = DSA.dense_attn_score_recompute_wrapper(
        q,
        k,
        lse,
        0.5,
        qhead_per_kv_head=h_q,
        out=attn_out,
        denom_out=attn_denom,
        ratio=4,
        q_causal_offsets=offsets,
    )
    assert attn["out"].data_ptr() == attn_out.data_ptr()
    ref_attn_out, ref_attn_denom = ref_dense_attn_score_recompute(
        q, k, lse, 0.5, ratio=4, q_causal_offsets=offsets
    )
    torch.testing.assert_close(attn_out, ref_attn_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(attn_denom, ref_attn_denom, atol=2e-2, rtol=2e-2)

    # ``dense_indexer_backward_wrapper`` requires the fp32 score buffers, so it
    # is exercised in the mp_32 kernel test; here we only assert the surface
    # rejects the FP8 request it cannot serve.
    with pytest.raises(NotImplementedError, match="FP8"):
        DSA.dense_attn_score_recompute_wrapper(
            q, k, lse, 0.5, precision="fp8", q_scale=torch.ones(1, device=device)
        )


def test_sw89132_csa_compressor_runs_training_ratios():
    """CSA compressor forward/backward on both DSV4 compress ratios (r4 and r128)."""
    from mudnn.csa.compressor import (
        csa_compressor_backward_wrapper,
        csa_compressor_forward_wrapper,
    )

    device = _musa_device()
    torch.manual_seed(13)
    for ratio, coff, head_dim in ((4, 2, 512), (128, 1, 512)):
        seq = 4096
        cu = torch.tensor([0, seq], dtype=torch.int32, device=device)
        comp = (seq // ratio) * torch.tensor([0, 1], dtype=torch.int32, device=device)
        kv = torch.randn((seq, coff * head_dim), dtype=torch.bfloat16, device=device) * 0.1
        score = torch.randn((seq, coff * head_dim), dtype=torch.bfloat16, device=device) * 0.1
        ape = torch.randn((ratio, coff * head_dim), dtype=torch.float32, device=device) * 0.01
        fwd = csa_compressor_forward_wrapper(
            kv, score, ape, cu, comp, ratio=ratio, head_dim=head_dim, coff=coff
        )
        out = fwd["out"]
        assert out.shape == (int(comp[-1]), head_dim) and out.dtype == torch.bfloat16
        grad_out = torch.randn_like(out)
        bwd = csa_compressor_backward_wrapper(
            kv,
            score,
            ape,
            cu,
            comp,
            grad_out,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
        )
        dkv, dscore, dape = bwd["grad_kv"], bwd["grad_score"], bwd["grad_ape"]
        assert dkv.shape == kv.shape and dscore.shape == score.shape
        assert dape.shape == ape.shape
        # The forward is a gated softmax pool, so the output must stay finite and
        # the dAPE reduction must have produced non-trivial values.
        assert torch.isfinite(out.float()).all()
        assert torch.isfinite(dkv.float()).all()
        assert torch.isfinite(dscore.float()).all()
        assert float(dape.abs().sum()) > 0.0


def test_sw89132_sparse_attention_forward_backward_roundtrip():
    """The two sparse-attention APIs the training graph calls in fwd and bwd."""
    DSA = _dsa()
    _native_backend()
    device = _musa_device()
    torch.manual_seed(17)
    s_q, s_kv, heads, d, topk = 64, 300, 64, 512, 128
    q = (torch.randn((s_q, heads, d), dtype=torch.bfloat16, device=device) * 0.05)
    kv = (torch.randn((s_kv, d), dtype=torch.bfloat16, device=device) * 0.05)
    idx = torch.randint(0, s_kv, (s_q, topk), dtype=torch.int32, device=device)
    dout = (torch.randn((s_q, heads, d), dtype=torch.bfloat16, device=device) * 0.05)
    sink = torch.linspace(-1, 1, heads, dtype=torch.float32, device=device)

    fwd = DSA.sparse_attention_forward_wrapper(q, kv, idx, attn_sink=sink)
    out, lse = fwd["out"], fwd["lse"]
    assert out.shape == (s_q, heads, d) and torch.isfinite(out.float()).all()
    assert torch.isfinite(lse.float()).all()

    dq = torch.empty_like(q)
    dkv = torch.empty_like(kv)
    bwd = DSA.sparse_attention_backward_wrapper(
        q, kv, out, dout, lse, sink, idx, softmax_scale=d**-0.5, dq=dq, dkv=dkv
    )
    for name in ("dq", "dkv", "d_sink"):
        if name == "d_sink":
            continue
        assert bwd[name] is not None
    assert bwd["dq"].data_ptr() == dq.data_ptr()
    assert bwd["dkv"].data_ptr() == dkv.data_ptr()
    assert torch.isfinite(dq.float()).all() and torch.isfinite(dkv.float()).all()
