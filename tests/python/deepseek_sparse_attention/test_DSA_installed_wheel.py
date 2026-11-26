# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run DSA sparse-attention forward checks against the built wheel.

``tests/python/conftest.py`` puts ``<checkout>/python`` at ``sys.path[0]``, so
everything else in this tree imports the checkout and never the artifact.  That
is right for development and blind for release validation: a wheel that omits a
file, ships a stale one, or disagrees with its own metadata stays green.

This module closes that gap.  The checks are written *here*, as the
``_WHEEL_UT`` source below, and are written out to a temporary directory and run
in a subprocess whose only ``mudnn`` is the installed wheel -- no checkout on
``sys.path``, no repository ``conftest.py``, ``cwd`` outside the tree.  Nothing
is imported from ``tests/python/deepseek_sparse_attention/``: the wheel consumer
is a separate audience, and its checks must not depend on the source tree
layout.

Build the artifact, then point the test at it (from the checkout root)::

    rm -rf build python/*.egg-info dist
    python -m pip wheel . --no-deps -w dist
    MUDNN_FRONTEND_WHEEL=dist/mudnn_frontend-*.whl \\
        python -m pytest tests/python/deepseek_sparse_attention/test_DSA_installed_wheel.py -q

``MUDNN_FRONTEND_WHEEL`` may also name an already-installed prefix, which skips
the ``pip install`` step.  Unset, every test here skips: a wheel is a release
artifact and the ordinary developer run must not depend on one.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_FRONTEND_ROOT = Path(__file__).resolve().parents[3]

#: The contract this port targets, one parametrized check per entry.
_CONTRACT_NAMES = ("SparseAttentionForward.__init__", "SparseAttentionForward.execute", "sparse_attention_forward_wrapper")

#: Behavioural checks that must pass on any machine, device or not.  Naming them
#: individually is what stops an all-skipped run from reading as success.
_DEVICE_FREE_CHECKS = (
    "test_forward_rejects_inputs_it_cannot_run",
    "test_forward_declines_unsupported_variants",
    "test_execute_refuses_to_run_before_compile",
)

#: Checks that run only when the machine has a MP31/MP32 device plus TileLang.
_DEVICE_CHECKS = (
    "test_forward_matches_the_reference_on_device",
    "test_forward_is_repeatable_on_device",
)

#: The wheel consumer's checks.  Self-contained on purpose: only ``mudnn``,
#: ``torch`` and ``pytest`` are imported, so the file exercises the artifact and
#: nothing else.  ``MUDNN_WHEEL_PREFIX`` / ``MUDNN_CHECKOUT_ROOT`` are exported
#: by the caller so the first check can prove which ``mudnn`` won.
_WHEEL_UT = r'''
"""Checks for the installed mudnn-frontend wheel's DSA sparse-attention forward."""

import inspect
import math
import os
import pathlib
import re

import pytest
import torch

from mudnn import deepseek_sparse_attention as DSA
from mudnn.deepseek_sparse_attention.sparse_attention_forward import api as forward_api


PREFIX = pathlib.Path(os.environ["MUDNN_WHEEL_PREFIX"]).resolve()
CHECKOUT = pathlib.Path(os.environ["MUDNN_CHECKOUT_ROOT"]).resolve()

#: (name, kind, default) of the cuDNN frontend contract this port targets,
#: captured so a wheel built from a drifted tree fails here.
FORWARD_CONTRACT = {
    "SparseAttentionForward.__init__": [
        ("sample_q", "P", None), ("sample_kv", "P", None), ("sample_topk_idxs", "P", None),
        ("sample_attn_sink", "P", "None"), ("sample_topk_length", "P", "None"),
        ("softmax_scale", "P", "None"), ("indexer_topk", "P", "0"),
    ],
    "SparseAttentionForward.execute": [
        ("q", "P", None), ("kv", "P", None), ("topk_idxs", "P", None),
        ("attn_sink", "K", "None"), ("topk_length", "K", "None"), ("softmax_scale", "K", "None"),
        ("out", "K", "None"), ("max_logits", "K", "None"), ("lse", "K", "None"),
        ("lse_indexer", "K", "None"), ("current_stream", "K", "None"),
    ],
    "sparse_attention_forward_wrapper": [
        ("q", "P", None), ("kv", "P", None), ("topk_idxs", "P", None),
        ("attn_sink", "K", "None"), ("topk_length", "K", "None"), ("softmax_scale", "K", "None"),
        ("indexer_topk", "K", "0"), ("stream", "K", "None"),
    ],
}

_KINDS = {inspect.Parameter.POSITIONAL_OR_KEYWORD: "P", inspect.Parameter.KEYWORD_ONLY: "K"}


def _rejecting_validate(*args, **kwargs):
    """Stand in for the device-touching validator so plan logic can run on CPU."""
    return 512**-0.5


def _inputs(sq, heads, topk, dim=512, skv=300, features=True, seed=2026):
    """CPU inputs in the shapes the API takes, before any device staging."""
    rng = torch.Generator().manual_seed(seed)
    q = torch.randn(sq, heads, dim, generator=rng).bfloat16()
    kv = torch.randn(skv, dim, generator=rng).bfloat16()
    idx = torch.randint(skv, (sq, topk), generator=rng, dtype=torch.int32)
    sink = lengths = None
    if features:
        # An empty row, an out-of-range and a negative sentinel, duplicates, ragged lengths, +-inf sinks.
        idx[0] = -1
        if topk > 3:
            idx[:, 3] = skv
        if topk > 7:
            idx[:, 7] = -7
        if topk > 1:
            idx[:, -1] = idx[:, 1]
        lengths = torch.tensor(([0, 1, 63, 64, 65, topk] * ((sq + 5) // 6))[:sq], dtype=torch.int32)
        sink = torch.linspace(-1, 1, heads, dtype=torch.float32)
        sink[0], sink[-1] = -float("inf"), float("inf")
    return q, kv, idx, sink, lengths


def _reference(q, kv, indices, sink, lengths, scale, head_dim_v=512):
    """FP64 slot-preserving reference, written from the documented semantics.

    Duplicate indices stay distinct softmax slots; negative, out-of-range and
    past-``lengths`` slots score ``-inf``; ``topk_length`` selects a prefix;
    the sink gates the output by ``sigmoid(lse - sink)`` with ``+-inf`` sinks
    collapsing to 1 or 0; an empty row yields ``out=0``, ``max_logits=-inf``,
    ``lse=+inf``; ``lse`` excludes the sink.
    """
    depth = kv.shape[0]
    slot = torch.arange(indices.shape[1])
    valid = (indices >= 0) & (indices < depth)
    if lengths is not None:
        valid = valid & (slot[None, :] < lengths[:, None])
    gathered = kv[indices.clamp(0, depth - 1)] * valid[..., None]
    logits = torch.einsum("shd,skd->shk", q, gathered) * scale
    logits = logits.masked_fill(~valid[:, None, :], -math.inf)
    empty = ~valid.any(-1)
    pooled = torch.where(empty[:, None, None], torch.zeros_like(logits), logits)
    lse = pooled.logsumexp(-1)
    probs = pooled.softmax(-1) * valid[:, None, :]
    out = torch.einsum("shk,skd->shd", probs, gathered[..., :head_dim_v])
    if sink is not None:
        finite = torch.where(torch.isfinite(sink), sink, torch.zeros_like(sink))
        gate = torch.sigmoid(lse - finite[None, :])
        gate = torch.where(torch.isneginf(sink)[None, :], torch.ones_like(gate), gate)
        gate = torch.where(torch.isposinf(sink)[None, :], torch.zeros_like(gate), gate)
        out = out * gate[..., None]
    return out, logits.amax(-1), lse.masked_fill(empty[:, None], math.inf)


def _device():
    """Skip unless a MP31/MP32 device with TileLang is present."""
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device unavailable")
    pytest.importorskip("tilelang")
    capability = tuple(torch.musa.get_device_capability())
    if capability not in ((3, 1), (3, 2)):
        pytest.skip(f"requires MP31 or MP32, found MP{capability[0]}{capability[1]}")
    return capability


def test_imports_the_wheel_not_a_checkout():
    """The whole point: `import mudnn` must resolve inside the installed wheel."""
    import mudnn
    import mudnn_frontend

    for module in (mudnn, mudnn_frontend):
        where = pathlib.Path(module.__file__).resolve()
        assert where.is_relative_to(PREFIX), f"{module.__name__} came from {where}, outside {PREFIX}"
        assert not where.is_relative_to(CHECKOUT), f"{module.__name__} came from the checkout ({where}); the wheel was not under test"

    dist_info = sorted(PREFIX.glob("mudnn_frontend-*.dist-info"))
    assert dist_info, f"{PREFIX} carries no dist-info, so it is not an installed wheel"
    declared = re.search(r"^Version: (.+)$", (dist_info[0] / "METADATA").read_text(), re.MULTILINE).group(1).strip()
    assert mudnn.__version__ == declared, f"mudnn.__version__ is {mudnn.__version__} but the wheel declares {declared}"

    source_version = re.search(r'^__version__ = "([^"]+)"$', (CHECKOUT / "python" / "mudnn_frontend" / "_version.py").read_text(), re.MULTILINE).group(1)
    assert mudnn.__version__ == source_version, f"the wheel is {mudnn.__version__} but this checkout is {source_version}; rebuild it from this tree"


@pytest.mark.parametrize("name", sorted(FORWARD_CONTRACT))
def test_forward_signatures_match_the_cudnn_frontend_contract(name):
    """Parameter names, kinds and defaults keep the upstream call sites working."""
    owner, _, method = name.partition(".")
    target = getattr(DSA, owner)
    if method:
        target = getattr(target, method)
    actual = [
        (p.name, _KINDS.get(p.kind, str(p.kind)), None if p.default is inspect.Parameter.empty else repr(p.default))
        for p in inspect.signature(target).parameters.values()
        if p.name != "self"
    ]
    assert actual == FORWARD_CONTRACT[name]


def test_forward_rejects_inputs_it_cannot_run():
    """Validate the cheap rejections a wheel-consumer hits first."""
    q, kv, idx, _, _ = _inputs(6, 64, 64)
    with pytest.raises(ValueError, match="MUSA"):
        DSA.sparse_attention_forward_wrapper(q, kv, idx)
    with pytest.raises(ValueError, match="Q must"):
        DSA.sparse_attention_forward_wrapper(q.float(), kv, idx)
    with pytest.raises(NotImplementedError, match="BF16"):
        DSA.sparse_attention_forward_wrapper(q.half(), kv.half(), idx)
    with pytest.raises(ValueError, match="head_dim"):
        DSA.sparse_attention_forward_wrapper(q[..., :256], kv[:, :256], idx)
    q576, kv576 = (torch.zeros(*t.shape[:-1], 576, dtype=t.dtype) for t in (q, kv))
    with pytest.raises(NotImplementedError, match="QK=V=512"):
        DSA.sparse_attention_forward_wrapper(q576, kv576, idx)
    with pytest.raises(ValueError, match="softmax_scale"):
        DSA.sparse_attention_forward_wrapper(q, kv, idx, softmax_scale=-1.0)
    with pytest.raises(ValueError, match="softmax_scale"):
        DSA.sparse_attention_forward_wrapper(q, kv, idx, softmax_scale=float("nan"))


def test_forward_declines_unsupported_variants(monkeypatch):
    """Plan-level rejections, reached by standing in for the device-touching validator."""
    q, kv, idx, _, _ = _inputs(6, 64, 64)
    monkeypatch.setattr(forward_api, "validate_inputs", _rejecting_validate)

    op = DSA.SparseAttentionForward(q, kv, idx)
    assert op.check_support() is True

    for heads, dim, indexer_topk in ((16, 512, 0), (64, 512, 4096), (128, 512, 2048)):
        sample = torch.empty(1, heads, dim, dtype=torch.bfloat16)
        bad = DSA.SparseAttentionForward(sample, kv[:, :dim], idx, indexer_topk=indexer_topk)
        with pytest.raises(ValueError, match="indexer_topk|supports only"):
            bad.check_support()
    with pytest.raises(ValueError, match="exceed"):
        DSA.SparseAttentionForward(q, kv, idx[:, :8], indexer_topk=512).check_support()
    with pytest.raises(ValueError, match="multiple of 64"):
        DSA.SparseAttentionForward(q, kv, idx[:, :63].contiguous()).check_support()
    strided_q = torch.empty(*q.shape, 2, dtype=q.dtype)[..., 0]
    with pytest.raises(ValueError, match="contiguous"):
        DSA.SparseAttentionForward(strided_q, kv, idx).check_support()


def test_execute_refuses_to_run_before_compile(monkeypatch):
    """`execute` must never compile or allocate; the plan has to say so."""
    q, kv, idx, _, _ = _inputs(6, 64, 64)
    monkeypatch.setattr(forward_api, "validate_inputs", _rejecting_validate)
    op = DSA.SparseAttentionForward(q, kv, idx)
    with pytest.raises(ValueError, match="not compiled"):
        op.execute(q, kv, idx)
    monkeypatch.setattr(forward_api, "_compile", lambda *args: (object(), None))
    assert op.check_support() is True
    assert op.compile() is None
    assert op._is_supported and op._compiled_kernel is not None


def test_forward_matches_the_reference_on_device():
    """The kernel itself: numerics against the FP64 reference, including the sentinels."""
    _device()
    q, kv, idx, sink, lengths = _inputs(6, 64, 64)
    expected = _reference(q.double(), kv.double(), idx, None if sink is None else sink.double(), lengths, 512**-0.5)
    qg, kvg, idxg, sinkg, lensg = (None if t is None else t.musa() for t in (q, kv, idx, sink, lengths))
    outputs = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lensg)
    torch.musa.synchronize()

    assert list(outputs.keys()) == ["out", "max_logits", "lse", "lse_indexer"]
    assert outputs["lse_indexer"] is None
    assert outputs["out"].dtype == torch.bfloat16 and outputs["out"].shape == (6, 64, 512)
    for name, golden in zip(("out", "max_logits", "lse"), expected):
        actual = outputs[name].detach().cpu().double()
        golden = golden.detach().cpu().double()
        assert torch.equal(torch.isfinite(actual), torch.isfinite(golden)), name
        finite = torch.isfinite(golden)
        assert torch.equal(actual[~finite], golden[~finite]), name
        stats = name in ("max_logits", "lse")
        torch.testing.assert_close(actual[finite], golden[finite], rtol=0.002 if stats else 0.02, atol=0.02 if stats else 0.01)
    # Row 0 has no valid slot at all, so it must come back exactly zero.
    assert torch.count_nonzero(outputs["out"][0]).item() == 0
    # The +inf sink on the last head zeroes that head's output.
    assert torch.count_nonzero(outputs["out"][:, -1]).item() == 0


def test_forward_is_repeatable_on_device():
    """A cached plan must not leak state between calls."""
    _device()
    q, kv, idx, sink, lengths = _inputs(6, 64, 64)
    qg, kvg, idxg, sinkg, lensg = (t.musa() for t in (q, kv, idx, sink, lengths))
    first = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lensg)
    torch.musa.synchronize()
    saved = {name: first[name].cpu().clone() for name in ("out", "max_logits", "lse")}
    for _ in range(2):
        again = DSA.sparse_attention_forward_wrapper(qg, kvg, idxg, attn_sink=sinkg, topk_length=lensg)
        torch.musa.synchronize()
        for name, golden in saved.items():
            torch.testing.assert_close(again[name].cpu(), golden, rtol=0, atol=0)
'''

_WHEEL_UT_MODULE = "test_wheel_dsa_forward.py"


def _clean_env(prefix: Path, workdir: Path) -> dict:
    """Environment for the subprocess: the artifact first, the checkout nowhere."""
    dropped = ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_CURRENT_TEST", "PYTEST_XDIST_TESTRUNUID")
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    env["PYTHONPATH"] = os.pathsep.join([str(prefix), str(workdir)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MUDNN_WHEEL_PREFIX"] = str(prefix)
    env["MUDNN_CHECKOUT_ROOT"] = str(_FRONTEND_ROOT)
    return env


def _run(argv, *, env, cwd, timeout=3600):
    return subprocess.run(argv, env=env, cwd=cwd, text=True, capture_output=True, timeout=timeout)


@pytest.fixture(scope="module")
def wheel_artifact():
    """Path to the wheel (or an installed prefix) named by ``MUDNN_FRONTEND_WHEEL``."""
    raw = os.environ.get("MUDNN_FRONTEND_WHEEL", "").strip()
    if not raw:
        pytest.skip("set MUDNN_FRONTEND_WHEEL=<path to mudnn_frontend-*.whl> or an install prefix to run the wheel-consumption checks")
    path = Path(raw).expanduser()
    if path.is_file() and path.suffix == ".whl":
        return path
    if path.is_dir() and (path / "mudnn" / "__init__.py").is_file():
        return path
    pytest.fail(f"MUDNN_FRONTEND_WHEEL={raw!r} is neither a .whl nor an install prefix containing mudnn/")


@pytest.fixture(scope="module")
def installed_wheel(wheel_artifact, tmp_path_factory):
    """Install the artifact into a private prefix and return that prefix."""
    if wheel_artifact.is_dir():
        return wheel_artifact
    prefix = tmp_path_factory.mktemp("wheel_prefix")
    result = _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-warn-script-location",
            "--disable-pip-version-check",
            "--target",
            str(prefix),
            str(wheel_artifact),
        ],
        env=_clean_env(prefix, prefix),
        cwd=prefix,
        timeout=1800,
    )
    assert result.returncode == 0, f"pip install {wheel_artifact} failed:\n{result.stdout}\n{result.stderr}"
    assert (prefix / "mudnn" / "__init__.py").is_file(), f"the wheel installed no mudnn package into {prefix}"
    return prefix


@pytest.fixture(scope="module")
def wheel_ut_report(installed_wheel, tmp_path_factory):
    """Write the embedded checks next to the prefix and run them against it.

    Returns pytest's ``{test name: outcome}`` plus the raw output, so each test
    below can make its own claim about the run.
    """
    workdir = tmp_path_factory.mktemp("wheel_ut")
    module = workdir / _WHEEL_UT_MODULE
    module.write_text(_WHEEL_UT, encoding="utf-8")
    result = _run(
        [sys.executable, "-m", "pytest", "-v", "--no-header", "-p", "no:cacheprovider", "--tb=short", module.name],
        env=_clean_env(installed_wheel, workdir),
        cwd=workdir,
        timeout=7200,
    )
    output = result.stdout + result.stderr
    outcomes = dict(re.findall(r"::(test_\w+(?:\[[^\]]*\])?)\s+(PASSED|FAILED|SKIPPED|ERROR)", output))
    return {"returncode": result.returncode, "output": output, "outcomes": outcomes}


def test_the_installed_wheel_imports_and_keeps_the_forward_contract(wheel_ut_report):
    """Identity, metadata agreement and the upstream signatures, checked in the artifact."""
    report = wheel_ut_report
    assert report["returncode"] == 0, f"the wheel checks failed:\n{report['output']}"
    outcomes = report["outcomes"]
    assert outcomes.get("test_imports_the_wheel_not_a_checkout") == "PASSED", f"the artifact was not the thing under test:\n{report['output']}"
    assert all(
        outcomes.get(f"test_forward_signatures_match_the_cudnn_frontend_contract[{name}]") == "PASSED" for name in _CONTRACT_NAMES
    ), f"the wheel's forward signatures drifted from the contract:\n{report['output']}"


def test_the_wheel_declines_what_it_cannot_run(wheel_ut_report):
    """Every device-free behavioural check must actually have run and passed."""
    report = wheel_ut_report
    assert report["returncode"] == 0, f"the wheel checks failed:\n{report['output']}"
    outcomes = report["outcomes"]
    not_passed = {name: outcomes.get(name) for name in _DEVICE_FREE_CHECKS if outcomes.get(name) != "PASSED"}
    assert not not_passed, f"these device-free checks did not pass against the wheel: {not_passed}\n{report['output']}"


def test_the_wheel_runs_the_forward_kernels_on_a_ph1_device(wheel_ut_report):
    """Device numerics, when this machine has the hardware; a skip is not a pass."""
    report = wheel_ut_report
    assert report["returncode"] == 0, f"the wheel checks failed:\n{report['output']}"
    outcomes = report["outcomes"]
    missing = [name for name in _DEVICE_CHECKS if name not in outcomes]
    assert not missing, f"the device checks were never collected: {missing}\n{report['output']}"
    skipped = [name for name in _DEVICE_CHECKS if outcomes[name] == "SKIPPED"]
    if skipped:
        pytest.skip(f"no MP31/MP32 device on this machine, so {skipped} did not run; run this on a device lane for the kernel numbers")
    assert all(outcomes[name] == "PASSED" for name in _DEVICE_CHECKS), f"the device numerics failed:\n{report['output']}"
