"""Standalone GDN backward: numerical gradients, checkpoints and transform VJPs."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from mudnn.linear_attention.experimental import gdn_bwd, prepare_gdn_bwd
from mudnn.linear_attention.experimental.gdn_bwd import (
    _beta_sigmoid_db,
    _l2norm_vjp,
    _safe_gate_dg,
)
from mudnn.linear_attention.ops.gdn import _apply_beta, _apply_gate, _l2_normalize
from test_ops.reference_gdn import (
    chunk_gdn_diff,
    relative_l2,
    recurrent_gdn_fp64,
    mp32_skip_reason,
)

GPU = pytest.mark.skipif(not getattr(torch, "musa", None) or not torch.musa.is_available(), reason="requires MUSA")
_mp32_reason = mp32_skip_reason()
MP32_ONLY = pytest.mark.skipif(_mp32_reason is not None, reason=_mp32_reason or "")


@GPU
@MP32_ONLY
@pytest.mark.parametrize("with_state", (False, True))
def test_port_gradients_and_checkpoint_reuse(with_state):
    torch.manual_seed(1515)
    q, k = [torch.nn.functional.normalize(torch.randn(256, 4, 128), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn_like(q)
    g, beta = -torch.rand(256, 4) * 0.8, torch.rand(256, 4) * 0.8
    inputs = (q, k, v, g, beta)
    h0 = torch.randn(2, 4, 128, 128) * 0.1 if with_state else None
    dfs = torch.randn_like(h0) if with_state else None
    do = torch.randn_like(v)
    cu = torch.tensor([0, 128, 256], dtype=torch.int32)
    leaves = [t.float().requires_grad_() for t in inputs]
    initial = h0.transpose(-1, -2).contiguous().requires_grad_() if with_state else None
    output, final = chunk_gdn_diff(*leaves, cu, initial_state=initial, output_final_state=with_state)
    loss = (output * do.float()).sum()
    if with_state:
        loss = loss + (final * dfs.transpose(-1, -2)).sum()
    expected = list(torch.autograd.grad(loss, leaves + ([initial] if with_state else [])))
    if with_state:
        expected[-1] = expected[-1].transpose(-1, -2)
    arguments = dict(
        dO=do.to("musa"),
        q=q.to("musa"),
        k=k.to("musa"),
        v=v.to("musa"),
        g=g.to("musa"),
        beta=beta.to("musa"),
        cu_seqlens=cu.to("musa"),
        initial_state=h0.to("musa") if with_state else None,
        d_final_state=dfs.to("musa") if with_state else None,
    )
    prepared = prepare_gdn_bwd(**arguments)
    got = prepared.run()
    reused = gdn_bwd(**arguments, state_checkpoints=prepared.checkpoints, checkpoint_every_n_tokens=64)
    for actual, ref in zip(got, expected):
        assert torch.isfinite(actual).all()
        assert relative_l2(ref, actual.cpu()) < 0.02
        assert not actual.requires_grad
    assert all(torch.equal(a, b) for a, b in zip(got, reused))


@GPU
@MP32_ONLY
def test_single_sequence_cut_tail_with_int64_prefix():
    # The first chunk can load both chunks via TMA. The final token uses masked
    # loads and stores, and the int64 prefix must survive lowering.
    torch.manual_seed(42)
    q, k = [torch.nn.functional.normalize(torch.randn(129, 4, 128), dim=-1).bfloat16() for _ in range(2)]
    v, do = torch.randn_like(q), torch.randn_like(q)
    g = torch.full((129, 4), -0.75)
    g[-1] = -32  # Keep the final one-token chunk below the state-transfer bound.
    beta = torch.rand(129, 4).bfloat16()
    cu = torch.tensor([0, 129], dtype=torch.int64)
    leaves = [t.double().requires_grad_() for t in (q, k, v, g, beta)]
    output, _ = recurrent_gdn_fp64(*leaves, cu, 128**-0.5)
    expected = torch.autograd.grad((output * do.double()).sum(), leaves)

    args = [t.to("musa") for t in (do, q, k, v, g, beta, cu)]
    prepared = prepare_gdn_bwd(*args)
    assert prepared.kernel.cut and prepared.kernel.fused_persistent
    actual = prepared.run()
    for want, got in zip(expected, actual):
        assert relative_l2(want, got.cpu()) < 0.02
    reused = gdn_bwd(*args, state_checkpoints=prepared.checkpoints, checkpoint_every_n_tokens=64)
    assert all(torch.equal(a, b) for a, b in zip(actual, reused))


@GPU
@MP32_ONLY
def test_empty_inputs_validate_and_propagate_state_gradient():
    q = torch.empty(0, 4, 128, dtype=torch.bfloat16, device="musa")
    g = torch.empty(0, 4, device="musa")
    cu = torch.zeros(3, dtype=torch.int32, device="musa")
    initial = torch.randn(2, 4, 128, 128, device="musa")
    final_gradient = torch.randn_like(initial)
    args = dict(dO=q, q=q, k=q, v=q, g=g, beta=g, cu_seqlens=cu, initial_state=initial, d_final_state=final_gradient)
    got = prepare_gdn_bwd(**args).run()
    assert torch.equal(got[5], final_gradient)
    with pytest.raises(ValueError, match="nondecreasing"):
        gdn_bwd(**{**args, "cu_seqlens": torch.tensor([0, 1, 0], dtype=torch.int32, device="musa")})
    with pytest.raises(ValueError, match="g must"):
        gdn_bwd(**{**args, "g": torch.empty(0, 1, device="musa")})
    with pytest.raises(NotImplementedError, match="batch-invariant"):
        gdn_bwd(**args, batch_invariant=True)


@GPU
@MP32_ONLY
@pytest.mark.parametrize(
    "dtype,dim_k,dim_v",
    [
        (torch.bfloat16, 64, 64),
        (torch.bfloat16, 64, 128),
        (torch.bfloat16, 128, 64),
        (torch.bfloat16, 128, 128),
        (torch.float16, 128, 128),
    ],
)
def test_standalone_dimension_pairs_match_fp64(dtype, dim_k, dim_v):
    """Preserve the standalone API's non-persistent dimension combinations."""
    torch.manual_seed(713)
    q, k = [torch.nn.functional.normalize(torch.randn(128, 1, dim_k), dim=-1).to(dtype) for _ in range(2)]
    v = (torch.randn(128, 1, dim_v) * 0.3).to(dtype)
    g = -torch.rand(128, 1) * 0.03
    beta = torch.rand(128, 1).to(dtype)
    state = torch.randn(1, 1, dim_v, dim_k) * 0.03
    do = torch.randn_like(v)
    dfinal = torch.randn_like(state) * 0.1
    cu = torch.tensor([0, 128], dtype=torch.int32)
    refs = [t.double().requires_grad_() for t in (q, k, v, g, beta)]
    h0 = state.double().transpose(-1, -2).contiguous().requires_grad_()
    out, final = recurrent_gdn_fp64(*refs, cu, dim_k**-0.5, h0)
    loss = (out * do.double()).sum() + (final * dfinal.double().transpose(-1, -2)).sum()
    expected = list(torch.autograd.grad(loss, (*refs, h0)))
    expected[-1] = expected[-1].transpose(-1, -2)
    actual = gdn_bwd(
        do.to("musa"),
        *(t.to("musa") for t in (q, k, v, g, beta)),
        cu.to("musa"),
        initial_state=state.to("musa"),
        d_final_state=dfinal.to("musa"),
    )
    for name, want, got in zip(("q", "k", "v", "g", "beta", "state"), expected, actual):
        assert torch.isfinite(got).all()
        assert relative_l2(want, got.cpu()) < 0.02, name


# Input-transform VJPs against PyTorch autograd (CPU)


def _close(a, b):
    return torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_l2norm_vjp_matches_autograd():
    torch.manual_seed(0)
    x = torch.randn(64, 4, 128, dtype=torch.float32, requires_grad=True)
    normalized = _l2_normalize(x)
    gout = torch.randn_like(normalized)
    (ref,) = torch.autograd.grad(normalized, x, gout)
    got = _l2norm_vjp(x.detach(), normalized.detach(), gout)
    assert got.shape == ref.shape
    assert _close(ref, got)


def test_safe_gate_vjp_matches_autograd():
    torch.manual_seed(1)
    total, h = 48, 4
    g = torch.randn(total, h, dtype=torch.float32, requires_grad=True)
    a_log = torch.randn(h, dtype=torch.float32, requires_grad=True)
    dt_bias = torch.randn(h, dtype=torch.float32, requires_grad=True)
    out = _apply_gate(g, True, a_log, dt_bias, h)  # -exp(a_log) * softplus(g + dt_bias)
    dg_out = torch.randn_like(out)
    ref_dg, ref_da, ref_ddt = torch.autograd.grad(out, (g, a_log, dt_bias), dg_out)

    amplitude = a_log.detach().float().exp()
    raw = g.detach().float() + dt_bias.detach().float()
    got_dg = _safe_gate_dg(dg_out, amplitude, raw)
    # run() computes da from the *incoming* gradient and gx == the gate output,
    # and ddt from the *propagated* dg; mirror that order here.
    got_da = (dg_out * out.detach()).sum(0)
    got_ddt = got_dg.sum(0)
    assert _close(ref_dg, got_dg)
    assert _close(ref_da, got_da)
    assert _close(ref_ddt, got_ddt)


@pytest.mark.parametrize("allow_neg_eigval", (False, True))
def test_beta_sigmoid_vjp_matches_autograd(allow_neg_eigval):
    torch.manual_seed(2)
    beta = torch.randn(48, 4, dtype=torch.float32, requires_grad=True)
    out = _apply_beta(beta, True, allow_neg_eigval, torch.float32)
    db_out = torch.randn_like(out)
    (ref,) = torch.autograd.grad(out, beta, db_out)

    factor = 2 if allow_neg_eigval else 1
    bx = out.detach()  # == factor * sigmoid(beta), the activated value run() holds
    got = _beta_sigmoid_db(db_out, bx, factor)
    assert _close(ref, got)
