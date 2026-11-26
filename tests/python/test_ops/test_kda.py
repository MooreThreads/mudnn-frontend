"""Unit tests for ``kimi_delta_attention`` (KDA forward).

The operator mirrors cuDNN frontend's ``cudnn.linear_attention.kimi_delta_attention``
argument for argument, so the surface tests here are written against that
contract: THD tensors, an explicit ``cu_seqlens``, and cuDNN's option names.

Correctness is checked against ``reference_kda.blockwise_kda_reference``, the
fp32 oracle the source kernels were validated with upstream.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")

from mudnn.engines import engine_for_id, engines_for  # noqa: E402
from mudnn.linear_attention import kimi_delta_attention  # noqa: E402
from mudnn.linear_attention.graph_analyzer import analyze_kda  # noqa: E402
from mudnn.linear_attention.mutlass.kda_engine import (  # noqa: E402
    ARCHITECTURES,
    KdaMutlassMp31Engine,
    KdaMutlassMp32Engine,
)

from .reference_kda import HEAD_SIZE, blockwise_kda_reference  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "musa") and torch.musa.is_available()),
    reason="MUSA is not available",
)

D = HEAD_SIZE
ENGINES = [f"kda_mutlass_{arch}" for arch in ARCHITECTURES]
REL_TOL = 2e-2
STATE_TOL = 8e-2


def _kernels_cover_this_device() -> bool:
    """Whether this process resolves to an architecture these kernels exist for.

    ``check_support`` gates on the build target *before* it looks at dtypes and
    options, so on a device no engine was built for the refusal is about the
    architecture and the per-option messages these tests assert never appear.
    A device without a backend is a missing feature rather than a failure, so
    those cases skip; what the gate does is asserted by
    ``test_engines_decline_a_build_target_that_is_not_theirs``, which patches
    the target and therefore needs no particular device.
    """
    from mudnn.linear_attention.mutlass import kda_ops

    try:
        return kda_ops.resolve_kda_arch() in ARCHITECTURES
    except Exception:
        return False


requires_kda_kernels = pytest.mark.skipif(
    not _kernels_cover_this_device(),
    reason="no mutlass KDA backend for this device's architecture",
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _inputs(seq_lens, h_qk, h_v, *, dtype=torch.bfloat16, seed=0):
    """THD inputs for a packed batch, in the operator's layout."""
    torch.manual_seed(seed)
    total = sum(seq_lens)
    h_o = max(h_qk, h_v)

    def rand(*shape, lo=0.0, hi=1.0):
        t = torch.empty(shape, device="musa", dtype=torch.float32)
        return t.uniform_(lo, hi).to(dtype)

    q = rand(total, h_qk, D, lo=-0.25, hi=0.25)
    k = rand(total, h_qk, D, lo=-0.25, hi=0.25)
    v = rand(total, h_v, D, lo=-0.25, hi=0.25)
    # g is per-key-channel: same shape as v, one decay per key channel.
    g = rand(total, h_v, D, lo=-1.0, hi=1.0)
    beta = rand(total, h_v, lo=-1.0, hi=1.0)
    cu = torch.tensor(
        [0, *[sum(seq_lens[: i + 1]) for i in range(len(seq_lens))]],
        device="musa",
        dtype=torch.int32,
    )
    return q, k, v, g, beta, cu, h_o


def _gate_params(h_v, *, seed=1):
    """Safe-gate parameters, which the oracle always expects.

    The reference models the gate transform unconditionally, so a call that
    omits ``a_log``/``dt_bias`` (and therefore uses ``g`` directly) has no
    oracle here; the correctness tests exercise the safe-gate path, which is
    also the one the upstream implementation validates.
    """
    torch.manual_seed(seed)
    a_log = torch.empty(h_v, device="musa").uniform_(-1.0, 0.5).to(torch.float32)
    dt_bias = (torch.empty(h_v, D, device="musa").uniform_(-0.5, 0.5)
               .to(torch.float32))
    return a_log, dt_bias


def _reference(q, k, v, g, beta, cu, *, scale, use_qk_l2norm_in_kernel,
               initial_state=None, a_log=None, dt_bias=None, lower_bound=-5.0):
    n = int(cu.numel() - 1)
    if initial_state is None:
        initial_state = torch.zeros(n, v.shape[1], D, D, device="musa",
                                   dtype=torch.float32)
    assert a_log is not None and dt_bias is not None, "oracle needs gate params"
    o, fs = blockwise_kda_reference(
        q, k, v, g, beta,
        scale=scale, initial_state=initial_state,
        A_log=a_log, dt_bias=dt_bias, lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu,
    )
    return o, fs


def _assert_close(actual, expected, *, tol, what):
    a = actual.detach().float()
    e = expected.detach().float()
    assert a.shape == e.shape, f"{what}: shape {tuple(a.shape)} != {tuple(e.shape)}"
    diff = (a - e).abs()
    bad = int((diff > tol + tol * e.abs()).sum().item())
    assert bad == 0, (
        f"{what}: {bad}/{diff.numel()} elements outside {tol} "
        f"(max abs {float(diff.max()):.4g})"
    )


# ---------------------------------------------------------------------------
# Registration surface
# ---------------------------------------------------------------------------


def test_manifest_offers_the_mutlass_engine():
    engines = engines_for(["KDA"])
    assert [e.name for e in engines] == ENGINES
    assert engine_for_id(engines[0].engine_id) is engines[0]
    # The KDA family owns its own id block, so it cannot collide with GDN's.
    assert engines[0].engine_id // 100 == 202
    assert engines_for(["NOT_AN_OP"]) == []


def test_every_architecture_has_exactly_one_engine_and_the_block_is_open():
    """One engine per architecture, each in its own slot of the family's block.

    Asserted against ``ARCHITECTURES`` rather than a literal list, so adding an
    architecture to the driver without registering an engine for it fails here
    instead of silently leaving those devices unserved.
    """
    engines = engines_for(["KDA"])
    assert [e.arch for e in engines] == list(ARCHITECTURES)
    assert len({e.engine_id for e in engines}) == len(engines)
    assert engine_for_id(engines[-1].engine_id + 1) is None


def test_unknown_plan_name_is_refused():
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    with pytest.raises(Exception, match="no 'nope' engine"):
        kimi_delta_attention(
            q, k, v, g, beta, cu,
            use_beta_sigmoid_in_kernel=True, plan_name="nope",
        )


# ---------------------------------------------------------------------------
# The operator's own validation
# ---------------------------------------------------------------------------


def test_thd_rank_is_required():
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    with pytest.raises(ValueError, match="THD"):
        kimi_delta_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0),
            beta.unsqueeze(0), cu, use_beta_sigmoid_in_kernel=True,
        )


def test_gate_must_be_per_key_channel():
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    with pytest.raises(ValueError, match="per-key-channel"):
        kimi_delta_attention(q, k, v, g.reshape(-1, 1), beta, cu,
                             use_beta_sigmoid_in_kernel=True)


@requires_kda_kernels
def test_dtype_gate_is_a_type_error():
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    # beta may be fp32 or the io dtype, but not a third dtype.
    with pytest.raises(TypeError, match="beta"):
        kimi_delta_attention(q, k, v, g, beta.to(torch.float16), cu,
                             use_beta_sigmoid_in_kernel=True)
    a_log, dt_bias = _gate_params(1)
    out, _ = kimi_delta_attention(q, k, v, g.float(), beta, cu,
                                  safe_gate=True, a_log=a_log, dt_bias=dt_bias,
                                  use_beta_sigmoid_in_kernel=True)
    assert torch.isfinite(out).all()


@requires_kda_kernels
@pytest.mark.parametrize(
    "include_a,include_dt", [(False, False), (True, False), (False, True)]
)
def test_safe_gate_parameters_default_independently(include_a, include_dt):
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    a_log = torch.zeros(v.shape[1], device="musa", dtype=torch.float32)
    dt_bias = torch.zeros(v.shape[1], D, device="musa")
    options = dict(safe_gate=True, use_beta_sigmoid_in_kernel=True)
    actual = kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
        cu,
        a_log=a_log if include_a else None,
        dt_bias=dt_bias if include_dt else None,
        **options,
    )
    expected = kimi_delta_attention(
        q, k, v, g, beta, cu, a_log=a_log, dt_bias=dt_bias, **options
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="safe_gate"):
        kimi_delta_attention(
            q,
            k,
            v,
            g,
            beta,
            cu,
            a_log=a_log,
            dt_bias=torch.zeros(v.shape[1], D, device="musa", dtype=torch.float32),
            use_beta_sigmoid_in_kernel=True,
        )


def test_gate_domain_linear_cannot_combine_with_safe_gate():
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    with pytest.raises(ValueError, match="gate_domain"):
        kimi_delta_attention(q, k, v, g, beta, cu, gate_domain="linear",
                             safe_gate=True, use_beta_sigmoid_in_kernel=True)


def test_allow_neg_eigval_requires_the_fused_sigmoid():
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    with pytest.raises(ValueError, match="allow_neg_eigval"):
        kimi_delta_attention(q, k, v, g, beta, cu, allow_neg_eigval=True,
                             use_beta_sigmoid_in_kernel=False)


def test_state_indices_cannot_combine_with_checkpoints():
    """cuDNN frontend rejects the pair, so this contract has to as well."""
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    n = int(cu.numel() - 1)
    s0 = torch.zeros(n, v.shape[1], D, D, device="musa", dtype=torch.float32)
    idx = torch.zeros(n, device="musa", dtype=torch.int32)
    with pytest.raises(ValueError, match="cannot be combined"):
        kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0,
                             state_indices=idx, checkpoint_every_n_tokens=D,
                             use_beta_sigmoid_in_kernel=True)


# ---------------------------------------------------------------------------
# Interface parity with cuDNN frontend
# ---------------------------------------------------------------------------


def test_signature_matches_the_cudnn_frontend_contract():
    """Argument names, order and defaults, frozen against the reference.

    ``cudnn.linear_attention.kimi_delta_attention`` is the interface this
    operator copies; a caller must be able to switch the import and keep the
    call, so a name, a position or a default drifting here is a break.
    """
    import inspect

    expected = [
        ("q", inspect.Parameter.empty),
        ("k", inspect.Parameter.empty),
        ("v", inspect.Parameter.empty),
        ("g", inspect.Parameter.empty),
        ("beta", inspect.Parameter.empty),
        ("cu_seqlens", inspect.Parameter.empty),
        ("scale", None),
        ("initial_state", None),
        ("output_final_state", False),
        ("use_qk_l2norm_in_kernel", False),
        ("batch_invariant", False),
        ("use_beta_sigmoid_in_kernel", False),
        ("allow_neg_eigval", False),
        ("safe_gate", False),
        ("gate_lower_bound", None),
        ("gate_domain", "log"),
        ("a_log", None),
        ("dt_bias", None),
        ("checkpoint_every_n_tokens", 0),
        ("plan_name", None),
        ("overwrite_initial_state", False),
        ("state_indices", None),
    ]
    got = [
        (p.name, p.default)
        for p in inspect.signature(kimi_delta_attention).parameters.values()
    ]
    assert got == expected


# ---------------------------------------------------------------------------
# Additional public options and pooled state updates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_beta_sigmoid_in_kernel": False},
        {"gate_domain": "linear"},
        {"batch_invariant": True},
        {"allow_neg_eigval": True},
    ],
    ids=["beta-not-logits", "linear-domain", "batch-invariant", "neg-eigval"],
)
@requires_kda_kernels
def test_public_options_are_supported(kwargs):
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    call = dict(use_beta_sigmoid_in_kernel=True)
    call.update(kwargs)
    g.fill_(0.8 if kwargs.get("gate_domain") == "linear" else -0.2)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(q.dtype)
    if not call["use_beta_sigmoid_in_kernel"]:
        beta = beta.sigmoid()
    out, _ = kimi_delta_attention(q, k, v, g, beta, cu, **call)
    assert out.shape == v.shape and bool(torch.isfinite(out).all())


@requires_kda_kernels
def test_overwrite_initial_state_updates_the_selected_pool_rows():
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    n = int(cu.numel() - 1)
    s0 = torch.zeros(n, v.shape[1], D, D, device="musa", dtype=torch.float32)
    call = dict(use_beta_sigmoid_in_kernel=True)

    expected_o, expected_state = kimi_delta_attention(
        q, k, v, g, beta, cu, initial_state=s0, output_final_state=True, **call
    )
    out, state = kimi_delta_attention(
        q, k, v, g, beta, cu, initial_state=s0, overwrite_initial_state=True,
        output_final_state=True, **call
    )
    assert state is s0
    torch.testing.assert_close(out, expected_o, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    pool = torch.randn(n + 2, v.shape[1], D, D, device="musa")
    untouched = pool.clone()
    idx = torch.arange(1, n + 1, device="musa", dtype=torch.int32)
    expected_o, expected_state = kimi_delta_attention(
        q,
        k,
        v,
        g,
        beta,
        cu,
        initial_state=pool[idx].contiguous(),
        output_final_state=True,
        **call,
    )
    out, state = kimi_delta_attention(
        q, k, v, g, beta, cu, initial_state=pool, state_indices=idx,
        output_final_state=True, **call
    )
    assert state is pool
    torch.testing.assert_close(out, expected_o, rtol=0, atol=0)
    torch.testing.assert_close(pool[idx], expected_state, rtol=0, atol=0)
    torch.testing.assert_close(pool[[0, n + 1]], untouched[[0, n + 1]], rtol=0, atol=0)


def test_engines_decline_a_build_target_that_is_not_theirs(monkeypatch):
    """The ArchTag is part of the kernel ABI, so the gate is on the build target.

    ``check_support`` asks :mod:`kda_ops` what it will compile for instead of
    reading the device, so patching that one decision has to move the choice:
    each engine accepts its own architecture and refuses the other's.  Driving
    it from the resolver keeps the test off the device's own architecture.
    """
    from mudnn.linear_attention.mutlass import kda_ops

    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    facts = analyze_kda("KDA", q, k, v, g, beta, cu,
                        use_beta_sigmoid_in_kernel=True)
    for engine, other in (
        (KdaMutlassMp31Engine(), "mp32"),
        (KdaMutlassMp32Engine(), "mp31"),
    ):
        monkeypatch.setattr(kda_ops, "resolve_kda_arch", lambda a=engine.arch: a)
        engine.check_support(facts)
        monkeypatch.setattr(kda_ops, "resolve_kda_arch", lambda a=other: a)
        with pytest.raises(Exception, match=f"built for {engine.arch}"):
            engine.check_support(facts)


# ---------------------------------------------------------------------------
# Correctness against the fp32 oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_qk_l2norm_in_kernel", [False, True],
                         ids=["pre-norm", "kernel-l2norm"])
@requires_kda_kernels
def test_single_sequence_matches_the_reference(use_qk_l2norm_in_kernel):
    seq_lens = [64]
    q, k, v, g, beta, cu, _ = _inputs(seq_lens, 1, 1)
    a_log, dt_bias = _gate_params(v.shape[1])
    o, fs = kimi_delta_attention(
        q, k, v, g, beta, cu,
        output_final_state=True,
        a_log=a_log, dt_bias=dt_bias, safe_gate=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=True,
    )
    ref_o, ref_fs = _reference(
        q, k, v, g, beta, cu, a_log=a_log, dt_bias=dt_bias,
        scale=D ** -0.5, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    _assert_close(o, ref_o, tol=REL_TOL, what="output")
    _assert_close(fs, ref_fs.to(fs.dtype), tol=STATE_TOL, what="final_state")


@pytest.mark.parametrize(
    "seq_lens", [[17], [31, 63], [19, 32, 57]], ids=["tail", "two", "three"]
)
@requires_kda_kernels
def test_varlen_matches_the_reference(seq_lens):
    q, k, v, g, beta, cu, _ = _inputs(seq_lens, 1, 2)
    a_log, dt_bias = _gate_params(v.shape[1])
    o, fs = kimi_delta_attention(
        q, k, v, g, beta, cu, output_final_state=True,
        a_log=a_log, dt_bias=dt_bias, safe_gate=True,
        use_beta_sigmoid_in_kernel=True,
    )
    ref_o, ref_fs = _reference(q, k, v, g, beta, cu, a_log=a_log, dt_bias=dt_bias,
                               scale=D ** -0.5, use_qk_l2norm_in_kernel=False)
    _assert_close(o, ref_o, tol=REL_TOL, what="output")
    _assert_close(fs, ref_fs.to(fs.dtype), tol=STATE_TOL, what="final_state")


@requires_kda_kernels
def test_grouped_query_heads_match_the_reference():
    """``H_v`` a multiple of ``H_qk``: the gate and state carry H_v heads."""
    q, k, v, g, beta, cu, _ = _inputs([32, 32], 2, 4)
    a_log, dt_bias = _gate_params(v.shape[1])
    o, fs = kimi_delta_attention(q, k, v, g, beta, cu, output_final_state=True,
                                 a_log=a_log, dt_bias=dt_bias, safe_gate=True,
                                 use_beta_sigmoid_in_kernel=True)
    ref_o, ref_fs = _reference(q, k, v, g, beta, cu, a_log=a_log, dt_bias=dt_bias,
                               scale=D ** -0.5, use_qk_l2norm_in_kernel=False)
    _assert_close(o, ref_o, tol=REL_TOL, what="output")
    _assert_close(fs, ref_fs.to(fs.dtype), tol=STATE_TOL, what="final_state")


@requires_kda_kernels
def test_initial_state_is_honoured():
    seq_lens = [48]
    q, k, v, g, beta, cu, _ = _inputs(seq_lens, 1, 1)
    n = len(seq_lens)
    s0 = torch.zeros(n, v.shape[1], D, D, device="musa", dtype=torch.float32)
    s0.uniform_(-0.05, 0.05)

    a_log, dt_bias = _gate_params(v.shape[1])
    o, fs = kimi_delta_attention(q, k, v, g, beta, cu, initial_state=s0,
                                 output_final_state=True,
                                 a_log=a_log, dt_bias=dt_bias, safe_gate=True,
                                 use_beta_sigmoid_in_kernel=True)
    ref_o, ref_fs = _reference(q, k, v, g, beta, cu, a_log=a_log, dt_bias=dt_bias,
                               scale=D ** -0.5, use_qk_l2norm_in_kernel=False,
                               initial_state=s0)
    _assert_close(o, ref_o, tol=REL_TOL, what="output")
    _assert_close(fs, ref_fs.to(fs.dtype), tol=STATE_TOL, what="final_state")


@requires_kda_kernels
def test_final_state_is_optional():
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    result = kimi_delta_attention(q, k, v, g, beta, cu, use_beta_sigmoid_in_kernel=True)
    assert isinstance(result, tuple) and len(result) == 2
    o, fs = result
    assert o.shape == v.shape
    assert fs.numel() == 0


@requires_kda_kernels
def test_fake_shapes_match_the_real_op():
    """The registered fake must agree with what the kernel returns."""
    q, k, v, g, beta, cu, _ = _inputs([32, 32], 1, 2)
    o, fs = kimi_delta_attention(q, k, v, g, beta, cu,
                                 output_final_state=True,
                                 use_beta_sigmoid_in_kernel=True)
    fake_o, fake_fs, fake_ck = torch.ops.mudnn.kimi_delta_attention_fwd(
        q, k, v, g, beta, cu, D ** -0.5,
        output_final_state=True, use_beta_sigmoid_in_kernel=True,
        plan_name=None,
    )
    assert o.shape == fake_o.shape
    assert fs.shape == fake_fs.shape
    assert fake_ck.numel() == 0


# ---------------------------------------------------------------------------
# State checkpoints (``checkpoint_every_n_tokens``)
# ---------------------------------------------------------------------------


def _cadence_config(cadence):
    """A recurrence config for one cadence, for the name/sharing tests."""
    from mudnn.linear_attention.mutlass import kda_ops

    return kda_ops.make_kda_ops_config(
        torch.bfloat16,
        state_dtype=torch.bfloat16,
        cu_seqlens_dtype=torch.int32,
        has_state_in=False,
        has_state_out=True,
        has_chunk_states_out=cadence > 0,
        checkpoint_every_n_tokens=cadence,
        state_fp32=False,
        has_gate_params=True,
        is_varlen=True,
        normalize_qk=False,
    )


def _checkpoint_rows(seq_lens, cadence):
    """cuDNN's capacity bound, ``max(total // C + N, 1)``."""
    return max(sum(seq_lens) // cadence + len(seq_lens), 1)


@requires_kda_kernels
def test_checkpoint_series_has_the_contract_shape():
    seq_lens = [64, 96]
    q, k, v, g, beta, cu, _ = _inputs(seq_lens, 1, 1)
    a_log, dt_bias = _gate_params(v.shape[1])
    o, fs, ck = kimi_delta_attention(
        q, k, v, g, beta, cu, output_final_state=True,
        a_log=a_log, dt_bias=dt_bias, safe_gate=True,
        use_beta_sigmoid_in_kernel=True, checkpoint_every_n_tokens=D,
    )
    rows = _checkpoint_rows(seq_lens, D)
    written = sum((n - 1) // D + 1 for n in seq_lens)
    assert o.shape == v.shape
    assert fs.shape == (len(seq_lens), v.shape[1], D, D)
    assert ck.shape == (rows, v.shape[1], D, D)
    # cudnn's `state_checkpoints` carries the activation dtype, not the state's.
    assert ck.dtype == q.dtype
    # The pool is sized to the capacity bound but only the packed series is
    # written (`count_b = (len_b - 1) // C + 1` per sequence, prefix-summed), so
    # the slack is uninitialised -- cuDNN allocates it with `torch.empty` too.
    assert 0 < written <= rows
    assert bool(torch.isfinite(ck[:written].float()).all())


@requires_kda_kernels
def test_the_first_checkpoint_is_the_initial_state():
    """Row 0 of a sequence is the state entering its first token.

    cuDNN stores "the chunk whose state enters token ``j * cadence``", so the
    ``j == 0`` row is the state before any token has been consumed -- the
    initial state.  That pins both the row mapping and the store: a pool is the
    activation dtype, so it is a rounded copy of the fp32 state.
    """
    seq_lens = [64]
    q, k, v, g, beta, cu, _ = _inputs(seq_lens, 1, 1)
    a_log, dt_bias = _gate_params(v.shape[1])
    s0 = torch.zeros(len(seq_lens), v.shape[1], D, D, device="musa",
                     dtype=torch.float32).uniform_(-0.05, 0.05)
    _, _, ck = kimi_delta_attention(
        q, k, v, g, beta, cu, initial_state=s0, output_final_state=True,
        a_log=a_log, dt_bias=dt_bias, safe_gate=True,
        use_beta_sigmoid_in_kernel=True, checkpoint_every_n_tokens=D,
    )
    assert ck.dtype == q.dtype
    _assert_close(ck[0:1], s0.to(ck.dtype), tol=STATE_TOL,
                  what="first checkpoint")


@requires_kda_kernels
def test_checkpointing_does_not_change_the_output():
    """The cadence is a store-only specialization, so the math must not move.

    The cadence is rendered into the kernel rather than passed to it, which is
    exactly the kind of change that can silently alter the schedule.
    """
    seq_lens = [64, 96]
    q, k, v, g, beta, cu, _ = _inputs(seq_lens, 1, 1)
    a_log, dt_bias = _gate_params(v.shape[1])
    call = dict(a_log=a_log, dt_bias=dt_bias, safe_gate=True,
                use_beta_sigmoid_in_kernel=True, output_final_state=True)
    o0, fs0 = kimi_delta_attention(q, k, v, g, beta, cu, **call)
    o1, fs1, ck = kimi_delta_attention(q, k, v, g, beta, cu,
                                       checkpoint_every_n_tokens=D, **call)
    _assert_close(o1, o0, tol=2e-2, what="checkpoint output")
    _assert_close(fs1, fs0, tol=2e-2, what="checkpoint final state")
    assert ck.numel() > 0


@requires_kda_kernels
def test_checkpoints_ride_an_fp32_state_in_the_activation_dtype():
    """Checkpoint storage follows the activation dtype independently of the seed."""
    seq_lens = [64]
    q, k, v, g, beta, cu, _ = _inputs(seq_lens, 1, 1)
    a_log, dt_bias = _gate_params(v.shape[1])
    s0 = torch.zeros(len(seq_lens), v.shape[1], D, D, device="musa",
                     dtype=torch.float32).uniform_(-0.05, 0.05)
    o, fs, ck = kimi_delta_attention(
        q, k, v, g, beta, cu, initial_state=s0, output_final_state=True,
        a_log=a_log, dt_bias=dt_bias, safe_gate=True,
        use_beta_sigmoid_in_kernel=True, checkpoint_every_n_tokens=D,
    )
    assert fs.dtype == torch.float32
    assert ck.dtype == q.dtype
    ref_o, ref_fs = _reference(q, k, v, g, beta, cu, a_log=a_log,
                               dt_bias=dt_bias, scale=D ** -0.5,
                               use_qk_l2norm_in_kernel=False, initial_state=s0)
    _assert_close(o, ref_o, tol=REL_TOL, what="output")
    _assert_close(fs, ref_fs.to(fs.dtype), tol=STATE_TOL, what="final_state")


@requires_kda_kernels
def test_cadence_is_a_specialization_axis():
    """Two cadences must not share one rendered kernel.

    The cadence is baked into the source, and the function name is what the
    generated ``.mu`` path, the spec registry key and the module cache all
    derive from, so a shared name would hand back one specialization for the
    other.
    """
    from mudnn.linear_attention.mutlass import kda_ops

    names = {
        kda_ops.get_kda_recurrence_ops_function_name(_cadence_config(c))
        for c in (0, D, 2 * D)
    }
    assert len(names) == 3
    assert any("_ckpt_0" in name for name in names)


@requires_kda_kernels
def test_the_prepare_kernel_is_shared_across_cadences():
    """Cadences share prepare when their MP32 restored-key frame agrees."""
    from mudnn.linear_attention.mutlass import kda_ops

    assert (kda_ops.get_kda_prepare_ops_module(_cadence_config(D))
            is kda_ops.get_kda_prepare_ops_module(_cadence_config(2 * D)))
    no_states = kda_ops.get_kda_prepare_ops_module(_cadence_config(0))
    with_states = kda_ops.get_kda_prepare_ops_module(_cadence_config(D))
    assert (no_states is with_states) == (kda_ops.resolve_kda_arch() != "mp32")


def test_a_cadence_that_is_not_a_chunk_multiple_is_refused():
    q, k, v, g, beta, cu, _ = _inputs([32], 1, 1)
    with pytest.raises(Exception, match="multiple of the kernel chunk"):
        kimi_delta_attention(q, k, v, g, beta, cu,
                             checkpoint_every_n_tokens=D + 1,
                             use_beta_sigmoid_in_kernel=True)
    with pytest.raises(ValueError, match="checkpoint_every_n_tokens"):
        kimi_delta_attention(q, k, v, g, beta, cu,
                             checkpoint_every_n_tokens=-D,
                             use_beta_sigmoid_in_kernel=True)
