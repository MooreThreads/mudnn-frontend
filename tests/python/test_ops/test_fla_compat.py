# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Tests for the ``mudnn.fla`` drop-in shim over flash-linear-attention."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")
pytest.importorskip("tilelang")
pytest.importorskip("fla")

import mudnn.fla  # noqa: E402
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as REAL  # noqa: E402

DEVICE = "musa"
DIM = 128
#: bf16 against a different bf16 kernel; the same gate the GDN suite uses.
REL_TOL = 2e-2

pytestmark = pytest.mark.skipif(
    not torch.musa.is_available(), reason="the TileLang GDN kernels are MUSA-only")


@pytest.fixture(autouse=True)
def _restore():
    yield
    mudnn.fla.restore_fla()


def _relative_l2(reference, actual):
    ref = reference.float()
    return ((ref - actual.float()).norm() / ref.norm().clamp(min=1e-12)).item()


def _inputs(batch=2, seq=512, heads=4, value_heads=None, requires_grad=False,
            dtype=torch.bfloat16, seed=3):
    torch.manual_seed(seed)
    hv = heads if value_heads is None else value_heads
    make = lambda h: torch.randn(batch, seq, h, DIM, device=DEVICE, dtype=dtype) * 0.5  # noqa: E731
    tensors = [make(heads), make(heads), make(hv),
               torch.randn(batch, seq, hv, device=DEVICE, dtype=dtype) * 0.5,
               torch.randn(batch, seq, hv, device=DEVICE, dtype=dtype)]
    if requires_grad:
        tensors = [t.detach().requires_grad_(True) for t in tensors]
    a_log = torch.rand(hv, device=DEVICE, dtype=torch.float32).log()
    dt_bias = torch.rand(hv, device=DEVICE, dtype=torch.float32)
    return (*tensors, a_log, dt_bias)


def _call(fn, q, k, v, g, beta, a_log, dt_bias, **kwargs):
    """One call in the shape FLA's GatedDeltaNet layer makes."""
    return fn(q=q, k=k, v=v, g=g, beta=beta, A_log=a_log, dt_bias=dt_bias,
              use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
              use_beta_sigmoid_in_kernel=True, state_v_first=True, **kwargs)


def _patched():
    mudnn.fla.accelerate_fla(verbose=False)
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


# ---------------------------------------------------------------------------
# Patch lifecycle
# ---------------------------------------------------------------------------


def test_patch_reaches_the_fla_layer_and_restores():
    import fla.layers.gated_deltanet as layer

    assert not mudnn.fla.is_accelerated()
    patched = _patched()
    assert mudnn.fla.is_accelerated("gated_delta_rule")
    assert mudnn.fla.is_accelerated("gdn")
    assert layer.chunk_gated_delta_rule is patched
    mudnn.fla.accelerate_fla(verbose=False)  # idempotent
    mudnn.fla.restore_fla()
    assert not mudnn.fla.is_accelerated()
    assert layer.chunk_gated_delta_rule is REAL


def test_unknown_target_is_refused():
    with pytest.raises(ValueError, match="unknown FLA acceleration target"):
        mudnn.fla.accelerate_fla(targets="nope")
    with pytest.raises(TypeError, match="must be a string"):
        mudnn.fla.accelerate_fla(targets=[1])


# ---------------------------------------------------------------------------
# Native path
# ---------------------------------------------------------------------------


def test_forward_matches_fla():
    args = _inputs()
    with torch.no_grad():
        reference, _ = _call(REAL, *args)
        actual, _ = _call(_patched(), *args)
    assert mudnn.fla.last_path() == "native"
    assert _relative_l2(reference, actual) < REL_TOL


def test_backward_matches_fla():
    args = _inputs(requires_grad=True)
    reference, _ = _call(REAL, *args)
    reference.float().sum().backward()
    grads = [t.grad.clone() for t in args[:5]]
    for tensor in args[:5]:
        tensor.grad = None

    actual, _ = _call(_patched(), *args)
    actual.float().sum().backward()
    assert mudnn.fla.last_path() == "native"
    assert _relative_l2(reference, actual) < REL_TOL
    for name, want, tensor in zip("qkvgb", grads, args[:5]):
        assert _relative_l2(want, tensor.grad) < REL_TOL, f"d{name}"


def test_state_round_trips_in_fla_layout():
    """``state_v_first`` states cross the boundary and come back V-major."""
    args = _inputs(batch=2, seq=256)
    state = torch.randn(2, 4, DIM, DIM, device=DEVICE, dtype=torch.float32) * 0.1
    with torch.no_grad():
        reference, ref_state = _call(REAL, *args, initial_state=state,
                                     output_final_state=True)
        actual, got_state = _call(_patched(), *args, initial_state=state,
                                  output_final_state=True)
    assert mudnn.fla.last_path() == "native"
    assert got_state.shape == ref_state.shape
    assert _relative_l2(reference, actual) < REL_TOL
    assert _relative_l2(ref_state, got_state) < REL_TOL


def test_state_k_major_passes_through():
    """``state_v_first=False`` is FLA's K-major layout, which the kernels take as is."""
    args = _inputs(batch=2, seq=256)
    state = torch.randn(2, 4, DIM, DIM, device=DEVICE, dtype=torch.float32) * 0.1
    with torch.no_grad():
        reference, ref_state = REAL(q=args[0], k=args[1], v=args[2], g=args[3],
                                    beta=args[4], A_log=args[5], dt_bias=args[6],
                                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                                    use_beta_sigmoid_in_kernel=True, state_v_first=False,
                                    initial_state=state, output_final_state=True)
        patched = _patched()
        actual, got_state = patched(q=args[0], k=args[1], v=args[2], g=args[3],
                                    beta=args[4], A_log=args[5], dt_bias=args[6],
                                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                                    use_beta_sigmoid_in_kernel=True, state_v_first=False,
                                    initial_state=state, output_final_state=True)
    assert mudnn.fla.last_path() == "native"
    assert _relative_l2(reference, actual) < REL_TOL
    assert _relative_l2(ref_state, got_state) < REL_TOL


def test_head_count_mismatch_falls_back():
    q, k, v, g, beta, a_log, dt_bias = _inputs(batch=1, seq=256, heads=4)
    with torch.no_grad():
        _call(_patched(), q, k, v, g[..., :2], beta[..., :2], a_log[:2], dt_bias[:2])
    assert mudnn.fla.last_path().startswith("fallback:")


def test_grouped_query_heads():
    args = _inputs(heads=2, value_heads=4)
    with torch.no_grad():
        reference, _ = _call(REAL, *args)
        actual, _ = _call(_patched(), *args)
    assert mudnn.fla.last_path() == "native"
    assert _relative_l2(reference, actual) < REL_TOL


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_float16_falls_back_bitwise():
    args = _inputs(dtype=torch.float16)
    with torch.no_grad():
        actual, _ = _call(_patched(), *args)
        reference, _ = _call(REAL, *args)
    assert mudnn.fla.last_path() == "fallback:q/k/v must be bfloat16"
    assert torch.equal(reference, actual)


def test_short_sequence_falls_back_bitwise():
    """One chunk is below the kernels' two-chunk minimum."""
    args = _inputs(batch=1, seq=64)
    with torch.no_grad():
        actual, _ = _call(_patched(), *args)
        reference, _ = _call(REAL, *args)
    assert mudnn.fla.last_path().startswith("fallback:")
    assert torch.equal(reference, actual)


def test_context_parallel_falls_back():
    args = _inputs(batch=1, seq=256)
    patched = _patched()
    with torch.no_grad(), pytest.raises(Exception):
        _call(patched, *args, cp_context=object())
    assert mudnn.fla.last_path() == "fallback:cp_context"
