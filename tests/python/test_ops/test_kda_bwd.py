"""Regression coverage for the MP31 and MP32 chunk-KDA backward."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mudnn.linear_attention import kimi_delta_attention_bwd  # noqa: E402
from .reference_kda_bwd import oracle_backward  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="MUSA is not available",
)


def _rel(a, b):
    a = a.detach().float().cpu().double()
    b = b.detach().float().cpu().double()
    return float(
        ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30))
    )


def _mp32_benchmark():
    """The MP32 K4 benchmark module, or a skip when it is not published."""
    import runpy

    path = Path(__file__).resolve().parents[3] / "benchmark/bench_kda_k4_mp32.py"
    if not path.is_file():
        pytest.skip("the MP32 K4 benchmark is not published in this snapshot")
    return runpy.run_path(str(path))


@pytest.mark.parametrize("lengths", [(1,), (16,), (17,), (33,), (3, 17)])
def test_kda_backward_matches_reference(
    lengths, heads=2, gate_lower_bound=-5.0, gate_bias=0.0
):
    torch.manual_seed(23)
    device = torch.device("musa")
    dim = 128
    tokens = sum(lengths)
    ends = [0]
    for length in lengths:
        ends.append(ends[-1] + length)

    def rand(*shape, scale=1.0):
        return (torch.rand(*shape, device=device) - 0.5) * scale

    q0, k0, v0 = (rand(heads, tokens, dim, scale=0.5) for _ in range(3))
    g0 = rand(heads, tokens, dim, scale=2.0) + gate_bias
    beta0 = rand(heads, tokens, scale=4.0)
    dO0 = rand(heads, tokens, dim)
    state0 = rand(len(lengths), heads, dim, dim, scale=0.2)
    d_final0 = rand(len(lengths), heads, dim, dim, scale=0.1)
    a_log = torch.randn(heads, device=device) * 0.3
    dt_bias = torch.zeros(heads, dim, device=device)

    to_thd = lambda x: x.transpose(0, 1).contiguous().to(torch.bfloat16)
    q, k, v, g = (to_thd(x) for x in (q0, k0, v0, g0))
    beta, dO = beta0.transpose(0, 1).contiguous().to(torch.bfloat16), to_thd(dO0)
    cu = torch.tensor(ends, device=device, dtype=torch.int32)
    got = kimi_delta_attention_bwd(
        dO,
        q,
        k,
        v,
        g,
        beta,
        cu,
        scale=dim**-0.5,
        initial_state=state0.to(torch.bfloat16),
        d_final_state=d_final0.float(),
        a_log=a_log,
        dt_bias=dt_bias,
        gate_lower_bound=gate_lower_bound,
    )

    references = []
    for seq, (begin, end) in enumerate(zip(ends, ends[1:])):
        references.append(oracle_backward(
            *[x[:, begin:end].float() for x in (q0, k0, v0, g0, beta0, dO0)],
            state0[seq].float(), d_final_state=d_final0[seq].float(),
            scale=dim**-0.5, A_log=a_log.float(), dt_bias=dt_bias.float(),
            lower_bound=gate_lower_bound, use_qk_l2norm=True, chunk=16,
        ))
    ref = {name: torch.cat([r[name] for r in references], dim=1)
           for name in ("dq", "dk", "dv", "dg", "dbeta")}
    ref["dstate0"] = torch.stack([r["dstate0"] for r in references])
    got_views = (
        got[0].transpose(0, 1),
        got[1].transpose(0, 1),
        got[2].transpose(0, 1),
        got[3].transpose(0, 1),
        got[4].transpose(0, 1),
        got[5],
    )
    errors = {name: _rel(actual, ref[name]) for name, actual in
              zip(("dq", "dk", "dv", "dg", "dbeta", "dstate0"), got_views)}
    assert max(errors.values()) < 3.0e-2, errors


@pytest.mark.parametrize("lower_bound", [-5.01, -5.0, -1.0, 0.0])
@pytest.mark.parametrize("gate_bias", [-3.0, 3.0])
def test_mp32_k4_reciprocal_domain(lower_bound, gate_bias):
    if torch.musa.get_device_capability() != (3, 2):
        pytest.skip("MP32 reciprocal dispatch")
    test_kda_backward_matches_reference(
        (17,), gate_lower_bound=lower_bound, gate_bias=gate_bias
    )


@pytest.mark.parametrize("normalize_qk", [True, False])
@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 15, 16])
@pytest.mark.parametrize("seed", [0, 107])
@pytest.mark.parametrize("misaligned", [False, True])
@pytest.mark.parametrize("reuse_prepare", [False, True])
def test_mp32_k4_workspace_abi(tokens, normalize_qk, seed, misaligned, reuse_prepare):
    if torch.musa.get_device_capability() != (3, 2):
        pytest.skip("MP32 workspace layout")
    import runpy
    from pathlib import Path

    bench = _mp32_benchmark()
    case = bench["prepare_case"](tokens=tokens, seed=seed, normalize_qk=normalize_qk)
    if reuse_prepare:
        bench["produce_prepare_cache"](case)
    if misaligned:
        for index in (0, 1, 2, 3, 5):
            original = case.inputs[index]
            storage = torch.empty(original.numel() + 1, device=original.device, dtype=original.dtype)
            view = storage[1:].view(original.shape)
            view.copy_(original)
            assert view.data_ptr() % 16 == 2
            case.inputs[index] = view
    kernel = bench["load_kernel"](case)
    kernel(*case.args())
    bench["check"](case)


@pytest.mark.parametrize("reuse_prepare", [False, True])
def test_mp32_k4_repeated_launches(reuse_prepare):
    if torch.musa.get_device_capability() != (3, 2):
        pytest.skip("MP32 synchronization regression")
    import runpy
    from pathlib import Path

    bench = _mp32_benchmark()
    fixture = bench["prepare_case"](tokens=16, seed=41)
    case = bench["repeat_case"](fixture, tokens=256, heads=8)
    if reuse_prepare:
        bench["produce_prepare_cache"](case)
    kernel = bench["load_kernel"](case)
    for iteration in range(20):
        for tensor in case.outputs:
            tensor.fill_(float("nan"))
        kernel(*case.args())
        if iteration % 5 == 0 or iteration == 19:
            for name, actual in zip(("dq", "dk", "dv", "dg", "dbeta"), case.outputs):
                ref = case.reference[name].transpose(0, 1).unsqueeze(0).to(actual.device)
                multiples = (1, 16, 4, 1) if actual.ndim == 4 else (1, 16, 4)
                expected = ref.repeat(multiples).float()
                assert torch.isfinite(actual).all(), name
                assert _rel(actual, expected) < 0.03, name


@pytest.mark.parametrize("chunks", [1, 2, 3, 4])
@pytest.mark.parametrize("heads", [2, 8, 16, 24, 64])
def test_mp32_k4_chunk_head_schedule(chunks, heads):
    if torch.musa.get_device_capability() != (3, 2):
        pytest.skip("MP32 chunk/head scheduling")
    import runpy
    from pathlib import Path

    bench = _mp32_benchmark()
    case = bench["repeat_case"](bench["prepare_case"](seed=107),
                                tokens=16 * chunks, heads=heads)
    scales = torch.exp2((torch.arange(chunks * heads, device="musa") % 3).float())
    scales = scales.reshape(chunks, heads)
    case.inputs[5].view(1, chunks, 16, heads, 128).mul_(scales[None, :, None, :, None])
    case.dstate.mul_(scales[:, :, None, None])
    for output in case.outputs:
        output.fill_(float("nan"))
    bench["load_kernel"](case)(*case.args())
    token_scales = scales[:, None, :].expand(chunks, 16, heads).reshape(1, 16 * chunks, heads)
    for name, actual in zip(("dq", "dk", "dv", "dg", "dbeta"), case.outputs):
        ref = case.reference[name].transpose(0, 1).unsqueeze(0).to("musa")
        multiples = (1, chunks, heads // 2, 1) if actual.ndim == 4 else (1, chunks, heads // 2)
        expected = ref.repeat(multiples).float()
        expected *= token_scales.unsqueeze(-1) if actual.ndim == 4 else token_scales
        assert torch.isfinite(actual).all(), (name, chunks, heads)
        assert _rel(actual, expected) < 0.03, (name, chunks, heads)


@pytest.mark.parametrize("lengths", [(17,), (3, 17)])
def test_mp32_kda_backward_scheduled_heads(lengths):
    if torch.musa.get_device_capability() != (3, 2):
        pytest.skip("MP32 chunk/head scheduling")
    test_kda_backward_matches_reference(lengths, heads=16)


def test_mp32_prepare_cache_rejects_wrong_extent():
    if torch.musa.get_device_capability() != (3, 2):
        pytest.skip("MP32 prepare cache")
    import runpy
    from pathlib import Path

    bench = _mp32_benchmark()
    case = bench["prepare_case"](seed=23)
    bench["produce_prepare_cache"](case)
    kernel = bench["load_kernel"](case)
    case.prepare_cache = torch.empty(
        (1, 2, case.prepare_cache.shape[-1] - 1), device="musa", dtype=torch.float32
    )
    with pytest.raises(RuntimeError):
        kernel(*case.args())
