# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Sparse (top-K) DSA indexer-backward tests.

Ported from ``cudnn-frontend``'s
``test/python/deepseek_sparse_attention/cutedsl/test_DSA_indexer_backward.py``.
This file is the MP31  twin of ``test_dsa_indexer_backward_mp32.py``:
it drives ``indexer_backward_mp31`` with ``block_I=64`` -- the MP31
default under the ~192 KB opt-in shared-memory ceiling.

The upstream file is 2591 lines because it covers four architectures, a
persistent-CTA launch policy, ``Gather4`` capability gating, the ``sm100_v2``
bf16-expansion backend, PDL, multi-device and multi-threaded stream semantics.
None of those exist on this side, so the *behaviours* are ported and the
architecture-specific scaffolding is not.

What is covered here, one test per behaviour:

===========================  ==================================================
values vs an autograd ref   ``test_indexer_backward_matches_reference``
packed-shape reuse           ``test_indexer_backward_packed_shapes_reuse_plan``
grad_scale is runtime        ``test_indexer_backward_grad_scale_is_runtime``
fp32 dK output               ``test_indexer_backward_fp32_dindexk_zeroed``
dtype/shape/rank rejection   ``test_indexer_backward_rejects_*``
topk / block_I envelope      ``test_indexer_backward_envelope``
topk id conventions          ``test_indexer_backward_local_vs_global_ids``
out-of-range local ids       ``test_indexer_backward_local_id_oob``
staged entry points          ``test_indexer_backward_staged_entry_points``
===========================  ==================================================

The upstream tests reach the kernel through ``cudnn.DSA``'s wrapper. Here the
wrapper delegates to a torch fallback that *recomputes* the predict distribution
and ignores ``index_score`` (see the class docstring), while the TileLang kernel
this work added consumes ``index_score`` directly. Testing the wrapper would
therefore exercise the fallback, not the port. These tests drive
``IndexerBackward`` with an explicitly supplied mp_31 kernel so the numerics of
the TileLang stage are what gets checked; the wrapper's own contract has
separate coverage in ``test_dsa_indexer_backward_api.py``.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")
pytest.importorskip("tilelang")

from test_ops.dsa_indexer_backward_reference import (  # noqa: E402
    CLIP_PROB_MIN,
    check_ref_indexer_backward,
    indexer_predict_distribution,
)
from test_ops.dsa_mp31_arch import mp31_skip_reason  # noqa: E402

DEVICE = "musa"

# Two independent gates. The arch one has to be a marker rather than a check
# inside each test: once a factory call has been made, TileLang has already been
# asked to compile; this suite validates the MP31 configuration (block_I=64
# under the ~192 KB opt-in budget), other capabilities get the mp_32 suite.
_MP31_SKIP_REASON = mp31_skip_reason()

pytestmark = [
    pytest.mark.skipif(
        not torch.musa.is_available(),
        reason="the DSA indexer-backward TileLang kernels are MUSA-only",
    ),
    pytest.mark.skipif(
        _MP31_SKIP_REASON is not None,
        reason=_MP31_SKIP_REASON or "",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kernel(batch, s_q, s_k, heads, dim, topk, **kwargs):
    from mudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_mp31 import (
        indexer_backward_mp31,
    )

    return indexer_backward_mp31(
        batch=batch,
        seqlen=s_q,
        seqlen_k=s_k,
        heads=heads,
        dim=dim,
        topk=topk,
        sm_scale=kwargs.pop("sm_scale", 1.0),
        block_I=kwargs.pop("block_I", 64),
        topk_indices_global=kwargs.pop("topk_indices_global", True),
    )


def _allocate(batch, s_q, s_k, heads, dim, topk, sm_scale=1.0, seed=0,
              global_ids=True):
    """Inputs matching what the kernel consumes.

    ``index_score`` must be the predict distribution *consistent* with
    ``(index_q, weights, index_k, topk_indices)``: the kernel reads it as the
    predict operand of the KL stage while the reference recomputes the scores
    from the same operands, so an inconsistent pair would make the two disagree
    for a reason that is neither's fault.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    index_q = torch.randn(batch, s_q, heads, dim, device=DEVICE, dtype=torch.bfloat16,
                          generator=g)
    weights = torch.randn(batch, s_q, heads, device=DEVICE, dtype=torch.bfloat16,
                          generator=g)
    index_k = torch.randn(batch, s_k, dim, device=DEVICE, dtype=torch.bfloat16,
                          generator=g)

    topk_k = min(topk, s_k)
    topk_indices = torch.stack([
        torch.stack([
            torch.randperm(s_k, device=DEVICE, generator=g)[:topk_k]
            for _ in range(s_q)
        ])
        for _ in range(batch)
    ]).to(torch.int32)
    if topk_k < topk:
        pad = torch.full((batch, s_q, topk - topk_k), -1, dtype=torch.int32,
                         device=DEVICE)
        topk_indices = torch.cat([topk_indices, pad], dim=-1)
    if global_ids:
        topk_indices = topk_indices + (
            torch.arange(batch, device=DEVICE, dtype=torch.int32)
            .view(batch, 1, 1) * s_k
        )

    with torch.no_grad():
        index_score = indexer_predict_distribution(
            index_q.float(), index_k.float(), weights.float(), topk_indices,
            sm_scale, topk_indices_global=global_ids,
        ).contiguous()

    attn_score = torch.softmax(
        torch.randn(batch, s_q, topk, device=DEVICE, generator=g), dim=-1
    ).float().contiguous()
    return index_q, weights, index_k, attn_score, index_score, topk_indices


def _run(kernel, index_q, weights, index_k, attn_score, index_score,
         topk_indices, grad_loss=1.0, grad_scale=1.0, d_index_k=None):
    """Launch the fused pipeline into fresh (or supplied) output buffers."""
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    if d_index_k is None:
        d_index_k = torch.zeros(index_k.shape, device=DEVICE, dtype=torch.float32)
    if isinstance(grad_loss, float):
        grad_loss = torch.tensor([grad_loss], device=DEVICE)
    kernel(
        index_q, weights, index_k,
        d_index_q, d_weights, d_index_k,
        attn_score, index_score, topk_indices,
        grad_loss, grad_scale,
    )
    torch.musa.synchronize()
    return d_index_q, d_weights, d_index_k


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topk", [128, 256, 512])
@pytest.mark.parametrize("global_ids", [True, False])
def test_indexer_backward_matches_reference(topk, global_ids):
    """Gradients match an independent autograd reference.

    ``index_score`` is the predict operand, so ``attn_score`` alone is random
    and the KL stage stays non-trivial. Both id conventions are covered because
    the kernel ``const_expr``-branches on ``topk_indices_global``, i.e. they are
    two different compiled kernels, not two spellings of one.
    """
    batch, s_q, s_k, heads, dim = 1, 128, 512, 64, 128
    index_q, weights, index_k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=topk, global_ids=global_ids
    )
    attn_ref, score_ref = attn.clone(), score.clone()
    kernel = _kernel(batch, s_q, s_k, heads, dim, topk,
                     topk_indices_global=global_ids)
    dq, dw, dk = _run(kernel, index_q, weights, index_k, attn, score, ids)

    for name, tensor in (("d_index_q", dq), ("d_weights", dw),
                         ("d_index_k", dk)):
        assert tensor.shape == {"d_index_q": index_q, "d_weights": weights,
                                "d_index_k": index_k}[name].shape
        assert torch.isfinite(tensor.float()).all(), f"nonfinite {name}"

    check_ref_indexer_backward(
        index_q, weights, index_k, attn_ref, score_ref, ids,
        dq, dw, dk, sm_scale=1.0, grad_scale=1.0,
        topk_indices_global=global_ids,
    )


def test_indexer_backward_attn_score_overwritten_in_place():
    """Kernel 1 leaves ``grad_signal`` in ``attn_score``; ``index_score`` is
    read-only. Upstream's contract, and the reason a mismatched plan reuse has
    to be rejected *before* the launch rather than after."""
    batch, s_q, s_k, heads, dim, topk = 1, 32, 256, 64, 128, 128
    index_q, weights, index_k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=5
    )
    attn_before, score_before = attn.clone(), score.clone()
    kernel = _kernel(batch, s_q, s_k, heads, dim, topk)
    _run(kernel, index_q, weights, index_k, attn, score, ids)

    assert not torch.equal(attn, attn_before), "attn_score was not overwritten"
    assert torch.equal(score, score_before), "index_score must stay read-only"
    # grad_signal is the KL gradient: rows sum to ~0 by construction
    row_sum = attn.float().sum(dim=-1)
    assert row_sum.abs().max().item() < 1e-2, (
        f"grad_signal rows should sum to ~0, got max {row_sum.abs().max().item():.3g}"
    )


# ---------------------------------------------------------------------------
# Compilation behaviour
# ---------------------------------------------------------------------------


def test_indexer_backward_packed_shapes_reuse_plan(monkeypatch):
    """Packed training changes ``(B, S_q, S_k)`` every step; a shape change must
    not recompile.

    ``batch`` / ``seqlen`` / ``seqlen_k`` are runtime extents of the prim_func, so
    the cache key is the tile configuration only and one compiled artifact serves
    every packed shape. GitHub #1188 is the upstream issue this guards.

    The counter is installed at **TileLang's compile entry point**, not at the
    kernel factory: the factory runs once per plan build by design (it is a thin
    wrapper that returns a ``@tl.jit`` function), so counting it would fail for a
    reason that is not a recompile. Upstream's equivalent test does the same
    thing one level down, monkeypatching ``cute.compile``.
    """
    import tilelang

    real_compile = tilelang.compile
    compiles = []

    def counting_compile(*args, **kwargs):
        compiles.append(1)
        return real_compile(*args, **kwargs)

    monkeypatch.setattr(tilelang, "compile", counting_compile)

    for i, (s_q, s_k) in enumerate([(64, 1024), (160, 3072), (96, 2048)]):
        batch, heads, dim, topk = 1, 64, 128, 128
        index_q, weights, index_k, attn, score, ids = _allocate(
            batch, s_q, s_k, heads, dim, topk, seed=100 + i
        )
        attn_ref, score_ref = attn.clone(), score.clone()
        kernel = _kernel(batch, s_q, s_k, heads, dim, topk)
        dq, dw, dk = _run(kernel, index_q, weights, index_k, attn, score, ids)
        check_ref_indexer_backward(
            index_q, weights, index_k, attn_ref, score_ref, ids,
            dq, dw, dk, topk_indices_global=True, cos_min=0.99,
        )

    # One compile for the sparse GEMM, one for score-grad: two kernels, first
    # shape only. A third+ compile means the runtime extent leaked into the key.
    assert len(compiles) <= 2, (
        f"expected at most 2 compiles (GEMM + score-grad) across three packed "
        f"shapes, got {len(compiles)} -- a runtime shape leaked into the "
        f"compile key")


def test_indexer_backward_grad_scale_is_runtime():
    """``grad_scale`` is a launch argument, not a compile key.

    Because kernel 1 forms ``g = -target * mask * (grad_loss * grad_scale)``, the
    whole gradient chain scales linearly with it. Checking two scales against the
    reference catches both a stale cached scale and a wrong operand order
    (``grad_scale`` versus ``sm_scale``) -- the latter is a real bug this port
    had, and it showed up as a *uniform* factor ``grad_scale / sm_scale`` on
    every output.
    """
    batch, s_q, s_k, heads, dim, topk = 1, 32, 512, 64, 128, 128
    q, w, k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=7
    )
    attn_ref, score_ref = attn.clone(), score.clone()
    kernel = _kernel(batch, s_q, s_k, heads, dim, topk)

    results = {}
    for scale in (1.0, 0.5, 2.0):
        dq, dw, dk = _run(kernel, q, w, k, attn.clone(), score, ids,
                          grad_scale=scale)
        results[scale] = (dq, dw, dk)
        check_ref_indexer_backward(
            index_q=q, weights=w, index_k=k, attn_score=attn_ref,
            index_score=score_ref, topk_indices=ids,
            d_index_q_actual=dq, d_weights_actual=dw, d_index_k_actual=dk,
            grad_scale=scale, topk_indices_global=True,
        )

    # linearity: doubling the scale doubles every gradient
    for a, b in zip(results[1.0], results[2.0]):
        doubled = b.float() / 2.0
        err = (a.float() - doubled).abs().max().item()
        denom = max(a.float().abs().max().item(), 1e-6)
        assert err / denom < 5e-2, (
            f"grad_scale should scale the result linearly, rel={err / denom:.3g}")


# ---------------------------------------------------------------------------
# Output dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dk_dtype", [torch.float32, torch.bfloat16])
def test_indexer_backward_dindexk_dtype_paths(dk_dtype):
    """fp32 ``d_index_k`` is atomic-accumulated directly; any other dtype goes
    through an internal fp32 scratch plus a trailing cast.

    The fp32 buffer must also be **zeroed internally**: the epilogue atomic-adds
    into it, so a caller-owned buffer full of garbage must still produce the same
    answer. Upstream makes that promise explicitly and this checks it by filling
    the buffer with a large sentinel first.
    """
    batch, s_q, s_k, heads, dim, topk = 1, 32, 512, 64, 128, 128
    index_q, weights, index_k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=11
    )
    attn_ref, score_ref = attn.clone(), score.clone()
    kernel = _kernel(batch, s_q, s_k, heads, dim, topk)

    dirty = torch.full(index_k.shape, 1e4, device=DEVICE, dtype=dk_dtype)
    dq, dw, dk = _run(kernel, index_q, weights, index_k, attn, score, ids,
                      d_index_k=dirty)
    assert dk.dtype == dk_dtype
    if dk_dtype is torch.float32:
        assert dk.data_ptr() == dirty.data_ptr(), "fp32 dK must be written in place"

    check_ref_indexer_backward(
        index_q, weights, index_k, attn_ref, score_ref, ids,
        dq, dw, dk.float(), topk_indices_global=True,
        cos_min=0.9 if dk_dtype is torch.bfloat16 else 0.97,
    )


# ---------------------------------------------------------------------------
# Negative behaviour / envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(topk=160), "divide topk"),
        (dict(topk=4096), "topk <= 2048"),
        (dict(heads=63), "multiple of 8"),
    ],
)
def test_indexer_backward_envelope(kwargs, match):
    """Out-of-envelope configurations are rejected at build time, by name.

    ``block_I`` must divide ``topk`` (the K axis is walked in whole blocks) and
    ``heads`` must be a multiple of 8 (the ``FullRow`` warp split). Rejecting in
    the factory rather than at launch means a caller cannot get a kernel that
    faults or silently reads past a tile.
    """
    base = dict(batch=1, s_q=16, s_k=256, heads=64, dim=128, topk=128,
                block_I=64)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        _kernel(**base)


def test_indexer_backward_rejects_wrong_rank():
    """Rank is checked at plan build with a message naming the tensor.

    Driven through ``IndexerBackward`` rather than ``indexer_backward_wrapper``:
    the wrapper's fallback path imports ``mudnn.DSA`` before it validates, so a
    missing DSA package would raise ``ModuleNotFoundError`` and hide whether the
    rank check exists at all. The class validates in ``__init__``, which is
    exactly the "reject the plan, do not fail dirty" behaviour upstream promises
    -- kernel 1 overwrites ``attn_score``, so a bad plan must not reach it.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.api import IndexerBackward

    dev = DEVICE
    q = torch.zeros(2, 8, 64, 128, device=dev, dtype=torch.bfloat16)
    w = torch.zeros(2, 8, 64, device=dev, dtype=torch.bfloat16)
    k = torch.zeros(2, 16, 128, device=dev, dtype=torch.bfloat16)
    dq = torch.zeros_like(q)
    dw = torch.zeros_like(w)
    dk = torch.zeros_like(k)
    attn = torch.zeros(2, 8, 8, device=dev, dtype=torch.float32)
    score = torch.zeros(2, 8, 8, device=dev, dtype=torch.float32)
    ids = torch.zeros(2, 8, 8, device=dev, dtype=torch.int32)

    # ``k`` is already 3D, so the wrong-rank case has to be built by dropping a
    # genuine axis: squeezing ``dim=0`` of a 3D tensor is a no-op and the plan
    # then (correctly) accepts it, which is how this assertion first failed.
    with pytest.raises(ValueError, match="index_k must be 3D"):
        IndexerBackward(q, w, k[0], dq, dw, dk, attn, score, ids)
    with pytest.raises(ValueError, match="topk_indices must be 3D"):
        IndexerBackward(q, w, k, dq, dw, dk, attn, score, ids[..., 0])
    with pytest.raises(ValueError, match="index_q must be 4D"):
        IndexerBackward(q[0], w, k, dq, dw, dk, attn, score, ids)


# ---------------------------------------------------------------------------
# Top-K id conventions
# ---------------------------------------------------------------------------


def test_indexer_backward_local_vs_global_ids_agree():
    """The two id conventions are different kernels that must agree.

    ``topk_indices_global=True`` takes ``b * S_k + local`` ids directly;
    ``False`` validates a local id and adds the per-batch offset in registers.
    With ``batch > 1`` the offset arithmetic is the only thing that differs, so a
    cross-batch mix-up shows up here and nowhere else.

    Each run gets an ``index_score`` recomputed from the ids *it* will gather.
    ``index_score`` is the predict operand the kernel reads while the scores are
    recomputed from the gathered K, so handing both runs one shared tensor would
    compare two inconsistent predict distributions rather than two addressing
    conventions -- which is exactly how the first version of this test failed
    with rel=52.
    """
    batch, s_q, s_k, heads, dim, topk = 2, 16, 512, 64, 128, 128
    q, w, k, attn, _, local_ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=21, global_ids=False
    )
    global_ids = local_ids + (
        torch.arange(batch, device=DEVICE, dtype=torch.int32).view(batch, 1, 1) * s_k
    )

    def score_for(ids, is_global):
        with torch.no_grad():
            return indexer_predict_distribution(
                q.float(), k.float(), w.float(), ids, 1.0,
                topk_indices_global=is_global,
            ).contiguous()

    dq_l, dw_l, dk_l = _run(
        _kernel(batch, s_q, s_k, heads, dim, topk, topk_indices_global=False),
        q, w, k, attn.clone(), score_for(local_ids, False), local_ids,
    )
    dq_g, dw_g, dk_g = _run(
        _kernel(batch, s_q, s_k, heads, dim, topk, topk_indices_global=True),
        q, w, k, attn.clone(), score_for(global_ids, True), global_ids,
    )

    for name, a, b in (("dQ", dq_l, dq_g), ("dW", dw_l, dw_g),
                       ("dK", dk_l, dk_g)):
        err = (a.float() - b.float()).abs().max().item()
        denom = max(b.float().abs().max().item(), 1e-6)
        assert err / denom < 2e-2, (
            f"{name}: local and global id conventions disagree, "
            f"rel={err / denom:.3g}")


def test_indexer_backward_local_id_out_of_range_contributes_nothing():
    """An out-of-range local id must contribute exactly zero, not read garbage.

    Building this comparison is easy to get wrong, and the naive version *is*
    wrong: zeroing the sentinel slots' ``attn_score`` / ``index_score`` and
    expecting an unchanged result ignores that
    ``grad_signal = g - predict * sum(g)`` depends on the *row* sum of ``g``, so
    removing one slot changes every other slot's signal. That produced a real
    false failure here (rel=0.0695).

    The correct statement is simply "the kernel agrees with a reference that
    masks those positions", so the test asserts exactly that: the masked run is
    compared against :func:`ref_indexer_backward`, whose ``masked_fill`` drops
    negative ids from the contraction. If the kernel clamped the id without
    masking the contribution, it would accumulate an extra gathered row and the
    comparison would fail.

    The global-id path deliberately does *not* range-check (it trusts the
    caller's ``b * S_k + local`` contract), so only the local path is covered.
    """
    batch, s_q, s_k, heads, dim, topk = 1, 16, 512, 64, 128, 128
    q, w, k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=31, global_ids=False
    )
    attn_ref, score_ref = attn.clone(), score.clone()
    kernel = _kernel(batch, s_q, s_k, heads, dim, topk, topk_indices_global=False)

    masked = ids.clone()
    masked[:, :, :4] = -1
    # A few ids past this batch's K extent: also no contribution.
    masked[:, :, 4:6] = s_k + 100

    dq, dw, dk = _run(kernel, q, w, k, attn, score, masked)
    for name, t in (("dQ", dq), ("dW", dw), ("dK", dk)):
        assert torch.isfinite(t.float()).all(), (
            f"nonfinite {name}: an out-of-range id must not read out of bounds")

    check_ref_indexer_backward(
        q, w, k, attn_ref, score_ref, masked, dq, dw, dk,
        topk_indices_global=False, cos_min=0.95, rms_rel_max=0.2,
    )


# ---------------------------------------------------------------------------
# Plan dispatch (IndexerBackward)
# ---------------------------------------------------------------------------


def test_indexer_backward_plan_dispatches_to_mp31_kernel():
    """On ``mp_31`` the plan binds the TileLang kernel, not the torch fallback.

    This is the dispatch itself: ``compile()`` must pick the mp31 implementation
    from the device capability, and the adapter must present the same positional
    contract as the fallback so ``execute`` does not branch.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward import api as A
    from mudnn.deepseek_sparse_attention.indexer_backward.api import IndexerBackward

    capability = A._device_capability(torch.device(DEVICE + ":0"))
    if capability != A._MP31_CAPABILITY:
        pytest.skip(f"requires mp_31, got capability {capability}")
    # An unknown / non-MUSA device must fall through to the torch fallback.
    assert A._sparse_indexer_backward_factory(torch.device("cpu")) is None

    batch, s_q, s_k, heads, dim, topk = 1, 16, 256, 64, 128, 128
    q, w, k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=61, global_ids=False
    )
    plan = IndexerBackward(
        q, w, k, torch.empty_like(q), torch.empty_like(w),
        torch.zeros(k.shape, device=DEVICE, dtype=torch.float32),
        attn, score, ids, sm_scale=1.0, block_I=64,
    )
    assert plan.check_support()
    plan.compile()
    assert type(plan._compiled_kernel).__name__ == "_Mp32SparseIndexerBackward"  # shared adapter


def test_indexer_backward_plan_matches_reference_through_execute():
    """The full plan path (``check_support`` -> ``compile`` -> ``execute``).

    The kernel-level tests call the factory's callable directly, which skips the
    plan's descriptor validation, the adapter's ``loss_coeff`` division and the
    in-place ``attn_score`` contract. This covers all three, on both id
    conventions, and pins the two behaviours that differ from the torch
    fallback: kernel 1 overwrites ``attn_score`` with ``grad_signal``, and
    ``index_score`` stays read-only.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.api import IndexerBackward

    for global_ids in (False, True):
        batch, s_q, s_k, heads, dim, topk = 2, 16, 512, 64, 128, 128
        q, w, k, attn, score, ids = _allocate(
            batch, s_q, s_k, heads, dim, topk, seed=71, global_ids=global_ids
        )
        attn_ref, score_ref = attn.clone(), score.clone()
        dq = torch.empty_like(q)
        dw = torch.empty_like(w)
        dk = torch.zeros(k.shape, device=DEVICE, dtype=torch.float32)

        plan = IndexerBackward(
            q, w, k, dq, dw, dk, attn, score, ids,
            sm_scale=1.0, block_I=64, topk_indices_global=global_ids,
        )
        assert plan.check_support()
        plan.compile()
        plan.execute(
            q, w, k, dq, dw, dk, attn, score, ids,
            grad_loss=torch.tensor([1.0], device=DEVICE),
            loss_coeff=float(batch * s_q),
        )
        torch.musa.synchronize()

        assert not torch.equal(attn, attn_ref), (
            "the mp31 path must overwrite attn_score with grad_signal")
        assert torch.equal(score, score_ref), (
            "index_score must stay read-only (it is the predict operand)")
        check_ref_indexer_backward(
            q, w, k, attn_ref, score_ref, ids, dq, dw, dk,
            grad_scale=1.0, topk_indices_global=global_ids, cos_min=0.99,
        )


def test_indexer_backward_plan_rejects_non_bf16_operands_on_mp31():
    """The mp31 kernel's operand dtype is baked in, so it must be checked.

    The factory never reads the caller's dtype -- it always builds with
    ``bfloat16`` -- so an fp16 or fp32 tensor would be *reinterpreted* rather
    than converted. ``check_support`` therefore has to refuse it; letting it
    through would produce plausible-looking garbage.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.api import IndexerBackward

    batch, s_q, s_k, heads, dim, topk = 1, 8, 128, 64, 128, 128
    q, w, k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=81, global_ids=False
    )
    plan = IndexerBackward(
        q.half(), w.half(), k.half(), torch.empty_like(q).half(),
        torch.empty_like(w).half(),
        torch.zeros(k.shape, device=DEVICE, dtype=torch.float32),
        attn, score, ids, sm_scale=1.0, block_I=64,
    )
    with pytest.raises(ValueError, match="must be bfloat16 on the TileLang"):
        plan.check_support()


def test_indexer_backward_plan_loss_coeff_scales_linearly():
    """``loss_coeff`` must reach the gradient linearly through the plan.

    The plan hands ``loss_coeff`` to the callable and the mp31 adapter divides by
    ``B * S_q`` before launching; the fallback divides inside ``mudnn.DSA``. Both
    therefore apply ``(loss_coeff / (B*S_q)) * grad_loss``. A missing or doubled
    division shows up here as a factor of ``B * S_q`` -- which is exactly what a
    silently-skipped adapter would produce.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.api import IndexerBackward

    batch, s_q, s_k, heads, dim, topk = 1, 32, 512, 64, 128, 128
    q, w, k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=91, global_ids=True
    )

    def once(loss_coeff):
        dq = torch.empty_like(q)
        dw = torch.empty_like(w)
        dk = torch.zeros(k.shape, device=DEVICE, dtype=torch.float32)
        plan = IndexerBackward(
            q, w, k, dq, dw, dk, attn.clone(), score, ids,
            sm_scale=1.0, block_I=64, topk_indices_global=True,
        )
        plan.check_support()
        plan.compile()
        plan.execute(q, w, k, dq, dw, dk, attn.clone(), score, ids,
                     grad_loss=1.0, loss_coeff=float(loss_coeff))
        torch.musa.synchronize()
        return dq, dw, dk

    base = once(batch * s_q)          # effective grad_scale = 1.0
    doubled = once(2 * batch * s_q)   # effective grad_scale = 2.0
    for name, a, b in zip(("dQ", "dW", "dK"), base, doubled):
        residual = (b.float() - 2.0 * a.float()).abs().max().item()
        scale = max(a.float().abs().max().item(), 1e-6)
        assert residual / scale < 5e-2, (
            f"{name}: doubling loss_coeff must double the gradient, "
            f"rel={residual / scale:.3g}")


def test_indexer_backward_wrapper_dispatches_to_mp31(monkeypatch):
    """``mudnn.DSA.indexer_backward_wrapper`` routes by device capability.

    The DSA-level wrapper is the surface users actually call, and it used to run
    the pure-torch implementation unconditionally. It now goes through
    ``IndexerBackward``, so on ``mp_31`` it must reach the TileLang kernel and
    inherit its in-place ``attn_score`` contract.
    """
    from mudnn import DSA

    from mudnn.deepseek_sparse_attention.indexer_backward import api as A

    capability = A._device_capability(torch.device(DEVICE + ":0"))
    if capability != A._MP31_CAPABILITY:
        pytest.skip(f"requires mp_31, got capability {capability}")

    batch, s_q, s_k, heads, dim, topk = 1, 32, 512, 64, 128, 128
    q, w, k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=101, global_ids=False
    )
    attn_ref, score_ref = attn.clone(), score.clone()

    out = DSA.indexer_backward_wrapper(
        q, w, k, attn, score, ids,
        sm_scale=1.0, loss_coeff=float(batch * s_q), grad_loss=1.0,
    )
    torch.musa.synchronize()

    plan = list(A._CACHE_OF_INDEXER_BACKWARD_OBJECTS.values())[-1]
    assert type(plan).__name__ == "IndexerBackward"
    assert type(plan._compiled_kernel).__name__ == "_Mp32SparseIndexerBackward"  # shared adapter
    assert not torch.equal(attn, attn_ref), (
        "the mp31 path must overwrite attn_score with grad_signal")
    assert torch.equal(score, score_ref), "index_score must stay read-only"
    check_ref_indexer_backward(
        q, w, k, attn_ref, score_ref, ids,
        out["d_index_q"], out["d_weights"], out["d_index_k"],
        grad_scale=1.0, cos_min=0.99,
    )


def test_indexer_backward_wrapper_fallback_does_not_recurse(monkeypatch):
    """With no TileLang factory, the wrapper falls back without recursing.

    ``IndexerBackward``'s fallback calls ``mudnn.DSA``'s torch implementation.
    If the DSA *wrapper* were used for that call, and the wrapper routes back
    through ``IndexerBackward``, the two would call each other forever on a
    non-TileLang device. Hence the terminal ``_indexer_backward_wrapper_naive``
    symbol; this pins the property by forcing the factory to ``None`` and
    checking the result, which a recursion would never reach.

    It also pins the *behavioural* difference between the paths: the torch
    fallback leaves ``attn_score`` intact, because it recomputes the predict
    distribution instead of running an in-place score-grad stage.
    """
    from mudnn import DSA

    from mudnn.deepseek_sparse_attention.indexer_backward import api as A

    monkeypatch.setattr(A, "_sparse_indexer_backward_factory", lambda device: None)
    monkeypatch.setattr(A, "_CACHE_OF_INDEXER_BACKWARD_OBJECTS", {})

    batch, s_q, s_k, heads, dim, topk = 1, 32, 512, 64, 128, 128
    q, w, k, attn, score, ids = _allocate(
        batch, s_q, s_k, heads, dim, topk, seed=111, global_ids=False
    )
    attn_ref, score_ref = attn.clone(), score.clone()

    out = DSA.indexer_backward_wrapper(
        q, w, k, attn, score, ids,
        sm_scale=1.0, loss_coeff=float(batch * s_q), grad_loss=1.0,
    )
    torch.musa.synchronize()

    plan = list(A._CACHE_OF_INDEXER_BACKWARD_OBJECTS.values())[-1]
    assert type(plan._compiled_kernel).__name__ == "_SparseIndexerBackwardFallback"
    assert torch.equal(attn, attn_ref), (
        "the torch fallback must not mutate attn_score")
    check_ref_indexer_backward(
        q, w, k, attn_ref, score_ref, ids,
        out["d_index_q"], out["d_weights"], out["d_index_k"],
        grad_scale=1.0, cos_min=0.99,
    )

    # A global-id request cannot be served by the fallback; it must say so
    # rather than gather the wrong rows.
    with pytest.raises(NotImplementedError, match="topk_indices_global"):
        DSA.indexer_backward_wrapper(
            q, w, k, attn.clone(), score, ids, topk_indices_global=True
        )


# ---------------------------------------------------------------------------
# Staged entry points
# ---------------------------------------------------------------------------


def test_indexer_backward_staged_entry_points():
    """``score_grad`` + ``gemm_only`` must reproduce the fused launch.

    Upstream exposes the two stages separately so a caller can overlap them with
    other work; if the split leaves a different result, the barrier/ordering
    between them is wrong. Also checks that ``score_grad_zero`` exists as an
    alias, which is part of the surface upstream publishes.
    """
    batch, s_q, s_k, heads, dim, topk = 1, 32, 512, 64, 128, 128
    q, w, k, attn, score, ids = _allocate(batch, s_q, s_k, heads, dim, topk,
                                          seed=41)
    kernel = _kernel(batch, s_q, s_k, heads, dim, topk)
    grad_loss = torch.tensor([1.0], device=DEVICE)

    dq_f, dw_f, dk_f = _run(kernel, q, w, k, attn.clone(), score, ids,
                            grad_loss=grad_loss, grad_scale=0.75)

    gs = attn.clone()
    dq_s = torch.empty_like(q)
    dw_s = torch.empty_like(w)
    dk_s = torch.zeros(k.shape, device=DEVICE, dtype=torch.float32)
    kernel.score_grad(q, gs, score, grad_loss, 0.75)
    kernel.gemm_only(q, w, k, dq_s, dw_s, dk_s, gs, ids)
    torch.musa.synchronize()

    for name, a, b in (("dQ", dq_f, dq_s), ("dW", dw_f, dw_s),
                       ("dK", dk_f, dk_s)):
        err = (a.float() - b.float()).abs().max().item()
        assert err < 1e-5, f"{name}: staged run differs from fused, {err:.3g}"

    assert hasattr(kernel, "score_grad_zero")
    assert hasattr(kernel, "gemm_after_score")
