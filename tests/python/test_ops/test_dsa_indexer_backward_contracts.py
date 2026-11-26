"""DSA indexer-backward tests that hold on every architecture.

The ``_mp32`` suites gate themselves on the device capability, because the
TileLang kernels they exercise do not compile elsewhere. The contracts checked
here are different in kind: they are about *which implementation gets chosen* and
about what a plan does **without** running a kernel, so they must pass on every
lane -- including the ``mp_31`` CI runner, where a regression here is exactly the
kind of thing that would otherwise only surface on a developer's machine.

Putting them in a gated module would have been the easy option and the wrong
one: they were originally appended to the mp_32 suites and reported as skipped,
which is indistinguishable from "not covered".
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")

from mudnn.deepseek_sparse_attention.indexer_backward.api import (  # noqa: E402
    DenseIndexerBackward,
    _grad_loss_operand,
    _tilelang_dense_envelope,
    _tilelang_sparse_envelope,
)

DEVICE = "musa"

pytestmark = pytest.mark.skipif(
    not torch.musa.is_available(),
    reason="the DSA indexer-backward plans need a MUSA device",
)


def _require_mp32(module: str, *names: str):
    """Resolve names on an MP32 module, or skip when MP32 is withheld.

    ``mudnn`` turns a missing optional submodule into a stub that raises
    ``NotImplementedError`` on attribute access *and* on every call, so both are
    funnelled through one skip. The returned callables re-raise as a skip too.
    """
    import importlib

    def _skip():
        pytest.skip(f"{module} is not published in this snapshot")

    def _guard(value):
        if not callable(value):
            return value

        def wrapped(*args, **kwargs):
            try:
                return value(*args, **kwargs)
            except NotImplementedError:
                return _skip()

        return wrapped

    try:
        imported = importlib.import_module(module)
        values = [_guard(getattr(imported, name)) for name in names]
    except (ModuleNotFoundError, NotImplementedError):
        return _skip()
    return values if len(values) > 1 else values[0]


def test_tilelang_envelope_predicates():
    """The dispatch predicates name the structural limits they guard.

    These conditions decide between the TileLang kernel and the torch
    implementation, so they must be checked without a device and without
    building anything.
    """
    # Legal shapes.
    assert _tilelang_dense_envelope(64, 128, 64, 512, 1) is None
    assert _tilelang_sparse_envelope(64, 512, 64) is None

    # Each rejection names its cause, so a caller can act on it.
    assert "multiple of 16" in _tilelang_dense_envelope(2, 128, 64, 512, 1)
    assert "multiple of 8" in _tilelang_dense_envelope(64, 4, 64, 512, 1)
    assert "does not divide max_seqlen_k" in _tilelang_dense_envelope(
        64, 128, 64, 500, 1
    )
    assert "ratio" in _tilelang_dense_envelope(64, 128, 64, 512, 0)
    assert "multiple of 8" in _tilelang_sparse_envelope(12, 512, 64)
    assert "does not divide topk" in _tilelang_sparse_envelope(64, 500, 64)


def test_grad_loss_tensor_is_passed_through_not_converted():
    """``grad_loss`` must reach the kernel as a tensor (review 3.3).

    Converting it with ``float()`` costs a device-to-host copy on every
    ``execute``, which also makes the call impossible to capture into a graph and
    freezes the value across replays. Identity is the property that matters: a
    copy would still be a tensor but would defeat the purpose.
    """
    tensor = torch.tensor([2.0], device=DEVICE)
    assert _grad_loss_operand(tensor, torch.device(DEVICE + ":0")) is tensor

    # ``None`` means "no per-call scaling"; a Python number is still accepted for
    # convenience and is left as-is for the caller to materialise.
    assert _grad_loss_operand(None, None) is None
    assert _grad_loss_operand(2.5, None) == 2.5

    with pytest.raises(ValueError, match="single-element"):
        _grad_loss_operand(torch.ones(2, device=DEVICE), None)
    with pytest.raises(ValueError, match="float32"):
        _grad_loss_operand(torch.ones(1, device=DEVICE, dtype=torch.bfloat16), None)


def test_dense_plan_outside_tilelang_envelope_falls_back_to_torch():
    """A shape the SQMMA tiles cannot express uses torch, not a raise (3.6).

    ``heads=2, dim=2`` is the legacy fixture shape. It used to fail with
    ``requires SQMMA-friendly tiles`` on a MUSA lane while passing on a CPU-only
    one -- the same input producing two verdicts depending on hardware. The plan
    now detects the envelope up front and binds the torch implementation, so the
    public contract is identical everywhere.
    """
    q = torch.randn(1, 4, 2, 2, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(1, 4, 2, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 2, device=DEVICE, dtype=torch.bfloat16)
    attn = torch.rand(1, 4, 8, device=DEVICE, dtype=torch.float32)
    l1 = attn.sum(-1).contiguous()
    score = torch.rand(1, 4, 8, device=DEVICE, dtype=torch.float32)
    lse = torch.rand(1, 4, device=DEVICE, dtype=torch.float32)
    dq, dw = torch.empty_like(q), torch.empty_like(w)
    dk = torch.empty_like(k)

    plan = DenseIndexerBackward(
        q, w, k, dq, dw, dk, attn, l1, score, lse,
        max_seqlen_q=4, max_seqlen_k=8,
    )
    plan.compile()
    assert type(plan._compiled_kernel).__name__ == "_DenseIndexerBackwardFallback"

    plan.execute(q, w, k, dq, dw, dk, attn.clone(), l1.clone(), score.clone(), lse)
    torch.musa.synchronize()
    assert torch.isfinite(dq).all() and torch.isfinite(dk).all()


def test_legacy_check_support_does_not_execute_the_kernel():
    """``DSA.IndexerBackward.check_support()`` must stay a shape pre-flight.

    It used to call the wrapper and compare the shapes of what came back, which
    ran a full backward pass at plan-build time. On a kernel-capable device that
    launched the real kernel, whose contract overwrites ``attn_score`` with
    ``grad_signal`` -- and the usual calling pattern makes the sample the very
    buffer ``execute`` later consumes, so the caller silently got results
    computed against a gradient instead of the distribution.
    """
    from mudnn import DSA

    batch, s_q, s_k, heads, dim, topk = 1, 32, 512, 64, 128, 128
    g = torch.Generator(device=DEVICE).manual_seed(131)
    q = torch.randn(batch, s_q, heads, dim, device=DEVICE,
                    dtype=torch.bfloat16, generator=g)
    w = torch.randn(batch, s_q, heads, device=DEVICE,
                    dtype=torch.bfloat16, generator=g)
    k = torch.randn(batch, s_k, dim, device=DEVICE,
                    dtype=torch.bfloat16, generator=g)
    ids = torch.stack([
        torch.stack([torch.randperm(s_k, device=DEVICE, generator=g)[:topk]
                     for _ in range(s_q)])
        for _ in range(batch)
    ]).to(torch.int32)
    attn = torch.softmax(
        torch.randn(batch, s_q, topk, device=DEVICE, generator=g), dim=-1
    ).float()
    score = attn.clone()
    dq = torch.empty_like(q)
    dw = torch.empty_like(w)
    dk = torch.zeros_like(k)

    legacy = DSA.IndexerBackward(
        q, w, k, dq, dw, dk, attn, score, ids, block_I=64
    )
    attn_before, score_before = attn.clone(), score.clone()
    assert legacy.check_support() is True
    torch.musa.synchronize()

    assert torch.equal(attn, attn_before), (
        "check_support must not overwrite attn_score with grad_signal")
    assert torch.equal(score, score_before)

    # A genuinely wrong sample shape is still rejected. ``mudnn.DSA`` names the
    # plan from the api module, which reports the offending argument as
    # ``d_index_k``; the torch fallback spelled it ``sample_d_index_k``.
    bad = DSA.IndexerBackward(
        q, w, k, dq, dw, torch.zeros_like(q), attn, score, ids, block_I=64
    )
    with pytest.raises(ValueError, match="d_index_k"):
        bad.check_support()


def test_backend_accepts_upstream_spelling():
    """``backend="sm100_v2"`` is recognised and normalised (review A1).

    This frontend's canonical value is the device-neutral ``"v2"``, but code
    written against cuDNN passes ``"sm100_v2"``. That used to fail with a
    ValueError listing an enum the caller had not used; it must instead reach the
    "recognised but unimplemented" path.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.api import (
        _reject_v2_backend,
        _validate_backend,
    )

    assert _validate_backend("sm100_v2") == "v2"
    assert _validate_backend("v2") == "v2"
    assert _validate_backend("default") == "default"
    with pytest.raises(ValueError, match="backend must be one of"):
        _validate_backend("not_a_backend")
    with pytest.raises(NotImplementedError, match="sm100_v2"):
        _reject_v2_backend(_validate_backend("sm100_v2"))


def test_bshd_rejects_a_max_seqlen_that_disagrees_with_the_shapes():
    """``max_seqlen_*`` is honoured, not silently dropped (review A4).

    Upstream derives both for BSHD; dropping a caller's value makes the parameter
    look effective when it is not. Agreeing values are accepted, disagreeing ones
    are rejected.
    """
    q = torch.randn(1, 64, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(1, 64, 64, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, 256, 128, device=DEVICE, dtype=torch.bfloat16)
    attn = torch.rand(1, 64, 256, device=DEVICE, dtype=torch.float32)
    l1 = attn.sum(-1).contiguous()
    score = torch.rand(1, 64, 256, device=DEVICE, dtype=torch.float32)
    lse = torch.rand(1, 64, device=DEVICE, dtype=torch.float32)
    dq, dw = torch.empty_like(q), torch.empty_like(w)
    dk = torch.zeros(1, 256, 128, device=DEVICE, dtype=torch.float32)

    def build(**kw):
        return DenseIndexerBackward(
            q, w, k, dq, dw, dk, attn, l1, score, lse, **kw
        )

    with pytest.raises(ValueError, match="derives max_seqlen_q"):
        build(max_seqlen_q=128, max_seqlen_k=256)

    ok = build(max_seqlen_q=64, max_seqlen_k=256)
    assert (ok.max_seqlen_q, ok.max_seqlen_k) == (64, 256)


def test_offsets_are_rejected_when_no_implementation_supports_them():
    """``q_causal_offsets`` must fail loudly off mp_32 (review A3).

    The generic kernel has no per-row causal offset and the torch fallback has
    none either, so accepting the flag would silently compute a different bound.
    The rejection belongs at ``check_support``, before any compile work.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.api import (
        _MP32_CAPABILITY,
        _device_capability,
    )

    if _device_capability(torch.device(DEVICE + ":0")) == _MP32_CAPABILITY:
        pytest.skip("this device's kernel implements q_causal_offsets")

    q = torch.randn(1, 64, 64, 128, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(1, 64, 64, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, 256, 128, device=DEVICE, dtype=torch.bfloat16)
    attn = torch.rand(1, 64, 256, device=DEVICE, dtype=torch.float32)
    l1 = attn.sum(-1).contiguous()
    score = torch.rand(1, 64, 256, device=DEVICE, dtype=torch.float32)
    lse = torch.rand(1, 64, device=DEVICE, dtype=torch.float32)

    plan = DenseIndexerBackward(
        q, w, k, torch.empty_like(q), torch.empty_like(w),
        torch.zeros(1, 256, 128, device=DEVICE, dtype=torch.float32),
        attn, l1, score, lse, has_q_causal_offsets=True,
    )
    with pytest.raises(NotImplementedError, match="mp_32"):
        plan.check_support()


def test_sparse_smem_charges_exactly_the_allocated_buffers():
    """The sparse budget matches the kernel's ``T.alloc_shared`` list.

    This number is the only guard a plan has against overflowing the 256 KB
    dynamic shared memory budget, so over-counting is not harmless -- it rejects
    plans that fit. Review 3.7 #1 found an under-count here: the kernel used to
    allocate ``dS_q`` *and* ``dS_k`` while the formula charged one, letting a plan
    needing ~281 KB pass. ``dS_q`` has since been folded away, so the formula now
    charges one ``dS_k``; charging two would over-count 8 KB.

    The ``heads`` coefficient is ``dim*2`` (Q_s) ``+ block_I*2`` (dS_k); a formula
    that still charged a second dS tile would report exactly ``step*block_I*2``
    more per head step.
    """
    _MAX_DYNAMIC_SHARED_MEMORY, _SMEM_OVERHEAD, _smem_bytes = _require_mp32(
        "mudnn.deepseek_sparse_attention.indexer_backward.indexer_backward_mp32",
        "_MAX_DYNAMIC_SHARED_MEMORY",
        "_SMEM_OVERHEAD",
        "_smem_bytes",
    )

    # Per-head coefficient pins the dS term: doubling only ``block_I`` moves every
    # block_I-proportional term, so compare two head counts instead.
    base, step = 64, 16
    per_head = _smem_bytes(base + step, 128, 64, "bfloat16") - _smem_bytes(
        base, 128, 64, "bfloat16"
    )
    expected = step * (128 * 2 + 64 * 2)
    assert per_head == expected, (
        f"per-head shared memory is {per_head} B, expected {expected} B "
        f"(128*2 for Q_s + 64*2 for the single dS_k tile); charging a second dS "
        f"tile would give {step * (128 * 2 + 2 * 64 * 2)} B")

    # And the whole figure, term by term, against the allocations in
    # _gemm_kernel_mp32 at the default ring depth.
    heads, block_I, dim, ns = 192, 128, 128, 2
    e, acc = 2, 4
    terms = (
        heads * dim * e,
        ns * block_I * dim * e,
        ns * block_I * dim * e,
        ns * block_I * acc,
        2 * ns * block_I * 4,
        block_I * 4,
        heads * block_I * e,
    )
    assert _smem_bytes(heads, dim, block_I, "bfloat16") == _SMEM_OVERHEAD + sum(
        terms
    ), "shared memory formula no longer matches the kernel's allocations"

    # This shape fits, which is why the review's rejection of it is gone: the
    # under-count it exposed is fixed, and the over-count it left behind is not.
    assert _smem_bytes(heads, dim, block_I, "bfloat16") <= _MAX_DYNAMIC_SHARED_MEMORY


def test_dsa_wrappers_no_longer_swallow_arguments():
    """``mudnn.DSA``'s wrappers forward instead of discarding (review A5/A6).

    They used to end with ``del kwargs``, so an upstream-style argument was either
    a ``TypeError`` or -- for anything spelled positionally -- accepted and
    silently dropped. ``q_causal_offsets`` was the dangerous case: dropping it
    changes the causal bound, so the result looks fine and is wrong.
    """
    import inspect

    from mudnn import DSA

    for fn in (DSA.indexer_backward_wrapper, DSA.dense_indexer_backward_wrapper):
        params = inspect.signature(fn).parameters
        assert not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ), f"{fn.__name__} must not accept **kwargs"

    sparse_params = inspect.signature(DSA.indexer_backward_wrapper).parameters
    for name in ("d_index_q", "d_weights", "d_index_k", "backend"):
        assert name in sparse_params, f"sparse wrapper missing {name}"

    dense_params = inspect.signature(DSA.dense_indexer_backward_wrapper).parameters
    for name in ("d_index_q", "d_weights", "d_index_k", "q_causal_offsets"):
        assert name in dense_params, f"dense wrapper missing {name}"

    # The dense wrapper must be a dispatcher now, not an alias of the torch
    # implementation -- otherwise it does not share the api module's semantics.
    # ``mudnn.DSA`` is a namespace over the package, so the private torch
    # implementation is addressed through the module that defines it.
    from mudnn.deepseek_sparse_attention import _torch_fallback

    assert (
        DSA.dense_indexer_backward_wrapper
        is not _torch_fallback._dense_indexer_backward_wrapper_naive
    )


def test_legacy_execute_takes_grad_loss_before_loss_coeff():
    """Parameter order matches upstream (review A7).

    Both arguments are scalars, so a positional call with the old order swapped
    them and produced a wrong scale with no error. This pins the order so it
    cannot silently regress.
    """
    import inspect

    from mudnn import DSA

    for cls in (DSA.IndexerBackward, DSA.DenseIndexerBackward):
        names = list(inspect.signature(cls.execute).parameters)
        assert names.index("grad_loss") < names.index("loss_coeff"), (
            f"{cls.__name__}.execute must take grad_loss before loss_coeff, "
            f"got {names}")


def test_strided_inputs_are_copied_not_rejected():
    """Non-compact tensors are accepted via a compact copy (review A8).

    The kernels index with ``.view(...)``, which needs a compact layout, so a
    strided tensor used to be rejected outright while upstream copies it. A
    rejected call is indistinguishable from an unsupported one, so the wrapper
    normalises instead.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.api import (
        _as_contiguous,
    )

    base = torch.zeros(2, 8, device=DEVICE)
    strided = base[:, ::2]
    assert not strided.is_contiguous()

    compact, restore = _as_contiguous(strided)
    assert compact.is_contiguous()
    assert compact is not strided, "a real copy is required"
    assert restore is not None

    # The restore hook writes the working copy back through the strided view.
    compact.fill_(3.0)
    restore()
    assert torch.all(base[:, ::2] == 3.0)

    # An already-compact tensor is passed through untouched.
    same, no_restore = _as_contiguous(base)
    assert same is base and no_restore is None


def test_dsa_dense_wrapper_matches_api_wrapper():
    """The two same-named dense wrappers now share one implementation (A5)."""
    from mudnn.deepseek_sparse_attention.indexer_backward.api import (
        dense_indexer_backward_wrapper as api_wrapper,
    )
    from mudnn import DSA

    batch, s_q, s_k, heads, dim = 1, 32, 128, 64, 128
    g = torch.Generator(device=DEVICE).manual_seed(77)
    q = torch.randn(batch, s_q, heads, dim, device=DEVICE,
                    dtype=torch.bfloat16, generator=g)
    w = torch.randn(batch, s_q, heads, device=DEVICE,
                    dtype=torch.bfloat16, generator=g)
    k = torch.randn(batch, s_k, dim, device=DEVICE,
                    dtype=torch.bfloat16, generator=g)
    attn = torch.rand(batch, s_q, s_k, device=DEVICE, dtype=torch.float32)
    l1 = attn.sum(-1).contiguous()
    score = torch.rand(batch, s_q, s_k, device=DEVICE, dtype=torch.float32)
    lse = torch.rand(batch, s_q, device=DEVICE, dtype=torch.float32)
    coeff = float(batch * s_q)

    dsa_out = DSA.dense_indexer_backward_wrapper(
        q, w, k, attn.clone(), l1.clone(), score.clone(), lse, loss_coeff=coeff
    )
    api_out = api_wrapper(
        q, w, k, attn.clone(), l1.clone(), score.clone(), lse, loss_coeff=coeff
    )
    torch.musa.synchronize()
    torch.testing.assert_close(
        dsa_out["d_index_q"].float(), api_out["d_index_q"].float()
    )


def test_packed_lengths_are_checked_against_the_declared_max():
    """THD per-batch lengths must not exceed ``max_seqlen`` (review 3.7 #3).

    The launch grid covers only ``max_seqlen`` rows per batch, so an over-long
    sequence would be silently truncated: the excess rows never get computed and
    the caller receives a complete-looking tensor with zeros or stale values. The
    torch implementation raises for this, so the kernel must agree.
    """
    _check_packed_lengths = _require_mp32(
        "mudnn.deepseek_sparse_attention.indexer_backward.dense_indexer_backward_mp32",
        "_check_packed_lengths",
    )

    with pytest.raises(ValueError, match="exceeds the declared max_seqlen"):
        _check_packed_lengths(
            "cu_seqlens_k", [64, 128], 64, True, 64
        )
    with pytest.raises(ValueError, match="exceeds the declared max_seqlen"):
        _check_packed_lengths(
            "cu_seqlens_q", [32, 64], 64, False, 32
        )

    # The pre-existing rules still hold.
    with pytest.raises(ValueError, match="not a multiple of"):
        _check_packed_lengths("cu_seqlens_k", [48], 64, True, 64)
    with pytest.raises(ValueError, match="non-decreasing"):
        _check_packed_lengths("cu_seqlens_q", [-1, 5], 64, False, 64)

    # Legal input passes: Q lengths need no alignment, K lengths do.
    _check_packed_lengths("cu_seqlens_k", [64, 128], 64, True, 192)
    _check_packed_lengths("cu_seqlens_q", [3, 5], 64, False, 5)


def test_plan_validation_is_not_an_assert():
    """Plan validation must survive ``python -O`` (review 3.7 #6).

    ``assert obj.check_support()`` disappears under ``-O``, leaving the plan
    unvalidated. The check is a plain branch now.
    """
    import inspect

    import mudnn.deepseek_sparse_attention.indexer_backward.api as api

    source = inspect.getsource(api)
    assert "assert obj.check_support()" not in source


def test_capability_query_failure_warns_instead_of_silently_degrading():
    """A failed capability query warns; a non-MUSA device stays quiet (3.7 #5).

    ``None`` makes the callers pick the architecture-generic implementation. A
    transient driver failure must not silently downgrade an ``mp_32`` device.
    """
    from unittest import mock

    import mudnn.deepseek_sparse_attention.indexer_backward.api as api

    def boom(_index):
        raise RuntimeError("simulated driver failure")

    with mock.patch.object(torch.musa, "get_device_properties", boom):
        with pytest.warns(RuntimeWarning, match="could not query"):
            assert api._device_capability(torch.device(DEVICE + ":0")) is None


def test_generic_factory_rejects_unimplemented_q_rows():
    """The generic kernel implements ``q_rows`` 1 and 2 only (review 3.7 #4).

    Larger values used to reach TileLang and fail there with an internal shape
    assertion that said nothing about the cause.
    """
    from mudnn.deepseek_sparse_attention.indexer_backward.dense_indexer_backward import (
        dense_indexer_backward,
    )

    with pytest.raises(ValueError, match="q_rows in \\(1, 2\\)"):
        dense_indexer_backward(
            batch=1, max_seqlen_q=64, max_seqlen_k=128,
            heads=64, dim=128, block_I=64, q_rows=3,
        )
