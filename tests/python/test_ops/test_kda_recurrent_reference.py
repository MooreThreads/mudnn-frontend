"""CPU checks for the imported, independent FP64 recurrence oracle."""

import pytest

torch = pytest.importorskip("torch")
from .reference_kda_recurrent import kda_reference


def test_single_token_decays_before_predicting_the_delta():
    q = torch.tensor([[[[0.3, -0.4]]]], dtype=torch.float64)
    k = torch.tensor([[[[0.6, 0.8]]]], dtype=torch.float64)
    v = torch.tensor([[[[0.7]]]], dtype=torch.float64)
    g = torch.tensor([[[[0.5, 0.25]]]], dtype=torch.float64).log()
    beta = torch.tensor([[[0.75]]], dtype=torch.float64)
    initial = torch.tensor([[[[0.2, -0.3]]]], dtype=torch.float64)
    decayed = initial * g.exp()
    expected = (
        decayed + ((v - (decayed * k).sum(-1, keepdim=True)) * beta[..., None]) * k
    )
    out, state = kda_reference(q, k, v, g, beta, scale=1, initial_state=initial)
    torch.testing.assert_close(state, expected)
    torch.testing.assert_close(out, (expected * q).sum(-1, keepdim=True))
    torch.testing.assert_close(
        initial, torch.tensor([[[[0.2, -0.3]]]], dtype=torch.float64)
    )


def test_empty_sequence_preserves_its_state():
    q = torch.ones(1, 1, 1, 2, dtype=torch.float64)
    g = torch.full_like(q, -0.5)
    v = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    initial = torch.arange(6, dtype=torch.float64).reshape(3, 1, 1, 2)
    _, state = kda_reference(
        q,
        q,
        v,
        g,
        v[..., 0],
        cu_seqlens=torch.tensor([0, 0, 1, 1]),
        initial_state=initial,
    )
    torch.testing.assert_close(state[[0, 2]], initial[[0, 2]])


def test_safe_gate_and_parameter_gradients_are_fp64_differentiable():
    torch.manual_seed(17)
    tensors = [
        torch.randn(1, 2, 1, d, dtype=torch.float64, requires_grad=True)
        for d in (2, 2, 1, 2)
    ]
    beta = torch.randn(1, 2, 1, dtype=torch.float64, requires_grad=True)
    a_log = torch.randn(1, dtype=torch.float64, requires_grad=True)
    dt_bias = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)

    def run(q, k, v, g, beta, a_log, dt_bias):
        return kda_reference(
            q,
            k,
            v,
            g,
            beta,
            safe_gate=True,
            a_log=a_log,
            dt_bias=dt_bias,
            use_beta_sigmoid=True,
        )

    assert torch.autograd.gradcheck(run, (*tensors, beta, a_log, dt_bias))
