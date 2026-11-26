# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""BF16 tests for the MP31  indexer forward kernel."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")
pytest.importorskip("tilelang")

from mudnn.deepseek_sparse_attention.indexer_forward.api_mp31 import (  # noqa: E402
    indexer_forward_wrapper,
)
from test_ops.dsa_mp31_arch import mp31_skip_reason  # noqa: E402

_MP31_SKIP_REASON = mp31_skip_reason()

pytestmark = [
    pytest.mark.skipif(
        not torch.musa.is_available(),
        reason="the MP31 indexer-forward TileLang kernel is MUSA-only",
    ),
    pytest.mark.skipif(
        _MP31_SKIP_REASON is not None,
        reason=_MP31_SKIP_REASON or "",
    ),
]

DEVICE = "musa"
HEADDIM = 128

def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor

def _ref_indexer_forward(q, k, w, ratio, q_causal_offsets, sm_scale=1.0):
    batch, seqlen_q, heads, _ = q.shape
    seqlen_k = k.shape[1]
    qf = q.float()
    kf = k.float()[:, :, 0, :]
    scores = torch.einsum("bqhd,bkd->bqhk", qf, kf)
    out = (scores.relu() * w.float()[..., None]).sum(2) * sm_scale

    if q_causal_offsets is None:
        q_causal_offsets = torch.zeros(batch, dtype=torch.int32, device=q.device)
    limit = (q_causal_offsets[:, None] + torch.arange(seqlen_q, device=q.device)[None, :] + 1) // ratio
    limit = limit.clamp(max=seqlen_k)
    mask = torch.arange(seqlen_k, device=q.device)[None, :] < limit[:, :, None]
    return torch.where(mask, out, torch.full_like(out, float("-inf")))

def _ref_indexer_forward_varlen(
    q,
    k,
    w,
    seqlen_qkv,
    cu_q_values,
    cu_k_values,
    ratio,
    offsets,
    sm_scale=1.0,
):
    ref = torch.full(
        (q.shape[0], max(seqlen_k for _, seqlen_k in seqlen_qkv)),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    for batch, (_, seqlen_k) in enumerate(seqlen_qkv):
        q0, q1 = cu_q_values[batch], cu_q_values[batch + 1]
        k0, k1 = cu_k_values[batch], cu_k_values[batch + 1]
        segment = _ref_indexer_forward(
            q[q0:q1].unsqueeze(0),
            k[k0:k1].unsqueeze(0),
            w[q0:q1].unsqueeze(0),
            ratio,
            None if offsets is None else offsets[batch : batch + 1],
            sm_scale=sm_scale,
        )
        ref[q0:q1, :seqlen_k] = segment[0, :, :seqlen_k]
    return ref

@pytest.mark.parametrize("seqlen_qkv", [(1024, 1234), (1024, 1236), (1234, 2468)])
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("q_causal_offset", [None, 0, 1, 4, 16, 252])
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("sm_scale", [0.5])
@pytest.mark.parametrize(
    "m_block,n_block,kv_stage",
    [
        (128, 128, 2),
        (128, 128, 4),
        (64, 256, 2),
        (64, 256, 4),
        (128, 512, 2),
    ],
)
def test_indexer_forward(
    seqlen_qkv,
    num_heads,
    ratio,
    q_causal_offset,
    weights_dtype,
    sm_scale,
    m_block,
    n_block,
    kv_stage,
):
    torch.manual_seed(666)
    seqlen_q, seqlen_k = seqlen_qkv
    q = torch.randn(1, seqlen_q, num_heads, HEADDIM, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    k = torch.randn(1, seqlen_k, 1, HEADDIM, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    w = torch.randn(1, seqlen_q, num_heads, dtype=torch.float32).to(weights_dtype).to(DEVICE)
    if q_causal_offset is None:
        q_causal_offsets = None
    else:
        q_causal_offsets = torch.full(
            (1,), q_causal_offset, dtype=torch.int32, device=DEVICE
        )

    ref = _ref_indexer_forward(q, k, w, ratio, q_causal_offsets, sm_scale=sm_scale)

    out = indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=num_heads,
        m_block_size=m_block,
        n_block_size=n_block,
        kv_stage=kv_stage,
        sm_scale=sm_scale,
        q_causal_offsets=q_causal_offsets,
    )["scores"]
    torch.musa.synchronize()

    finite = torch.isfinite(ref)
    assert torch.equal(torch.isneginf(out), torch.isneginf(ref))
    torch.testing.assert_close(out[finite], ref[finite], rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize(
    "seqlen_qkv",
    [
        [(512, 512), (512, 512), (512, 512)],
        [(1234, 1234), (1234, 1234)],
        [(1, 1), (3, 7), (7, 67), (11, 70)],
    ],
)
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("ratio", [1, 2, 4])
@pytest.mark.parametrize("q_causal_offset", [None, 0, 1, 4, 16, 252])
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("sm_scale", [0.5])
@pytest.mark.parametrize(
    "m_block,n_block,kv_stage",
    [
        (128, 128, 2),
        (128, 128, 4),
        (64, 256, 2),
        (64, 256, 4),
        (128, 512, 2),
    ],
)
def test_indexer_forward_varlen(
    seqlen_qkv,
    num_heads,
    ratio,
    q_causal_offset,
    weights_dtype,
    sm_scale,
    m_block,
    n_block,
    kv_stage,
):
    torch.manual_seed(666)
    q_lengths = [seqlen_q for seqlen_q, _ in seqlen_qkv]
    k_lengths = [seqlen_k for _, seqlen_k in seqlen_qkv]
    cu_q_values = [0]
    for length in q_lengths:
        cu_q_values.append(cu_q_values[-1] + length)
    cu_k_values = [0]
    for length in k_lengths:
        cu_k_values.append(cu_k_values[-1] + length)
    total_q = cu_q_values[-1]
    total_k = cu_k_values[-1]

    q = torch.randn(total_q, num_heads, HEADDIM, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    k = torch.randn(total_k, 1, HEADDIM, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    w = torch.randn(total_q, num_heads, dtype=torch.float32).to(weights_dtype).to(DEVICE)
    cu_q = torch.tensor(cu_q_values, dtype=torch.int32, device=DEVICE)
    cu_k = torch.tensor(cu_k_values, dtype=torch.int32, device=DEVICE)

    if q_causal_offset is None:
        offsets = None
    else:
        offsets = torch.full(
            (len(seqlen_qkv),), q_causal_offset, dtype=torch.int32, device=DEVICE
        )

    ref = _ref_indexer_forward_varlen(
        q,
        k,
        w,
        seqlen_qkv,
        cu_q_values,
        cu_k_values,
        ratio,
        offsets,
        sm_scale=sm_scale,
    )

    out = indexer_forward_wrapper(
        q,
        k,
        w,
        ratio=ratio,
        qhead_per_kv_head=num_heads,
        m_block_size=m_block,
        n_block_size=n_block,
        kv_stage=kv_stage,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        q_causal_offsets=offsets,
    )["scores"]
    torch.musa.synchronize()

    for batch, (_, seqlen_k) in enumerate(seqlen_qkv):
        q0, q1 = cu_q_values[batch], cu_q_values[batch + 1]
        assert bool(torch.isneginf(out[q0:q1, seqlen_k:]).all().item())

    finite = torch.isfinite(ref)
    assert torch.equal(torch.isneginf(out), torch.isneginf(ref))
    torch.testing.assert_close(out[finite], ref[finite], rtol=1e-4, atol=1e-4)

