"""Complete ratio=128 numerics and dispatch gate for TileLang/MUSA.

The gate mirrors the competitor's contract categories while using the public
TileLang class API: finite unit-scale cases, an exact exponent-shift case,
static row/token-capacity padding, an fp32-overflow NaN-pattern case, and a
ratio=4 reference cross-check.  The eager reference and input helpers are
loaded from the repository's pytest module so benchmark and pytest cannot
silently use different numerics.

Examples::

  PYTHONPATH=python python benchmark/csa/gate_csa_compressor_r128.py --quick
  PYTHONPATH=python python benchmark/csa/gate_csa_compressor_r128.py --json gate.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch_musa  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from mudnn.csa.compressor import CSACompressorBackward, CSACompressorForward  # noqa: E402
from mudnn.csa.compressor.compressor_sm100_r128 import (  # noqa: E402
    _FORWARD_TMA_ROW_BUCKETS,
    _FORWARD_TILEWARP_ROW_BUCKETS,
    _ROWGROUP_BWD_ROW_BUCKETS,
    _TILEWARP_BWD_ROW_BUCKETS,
    fused_backward_schedule_r128,
    _forward_kernel_schedule_r128,
)


RATIO = 128
ABS_TOL = 1.6e-2
DIFF_FRAC = 1.0e-3
DAPE_TOL = 1.0e-3
CONFIGS = {
    "c1d128": (1, 128),
    "c2d128": (2, 128),
    "c1d512": (1, 512),
    "c2d512": (2, 512),
}
EDGE_PACK = [127, 8192, 0, 129, 128, 3, 515, 1024]
CASES = [
    ("c1d128", "1x8192", [8192]),
    ("c1d128", "3x8192", [8192, 8192, 8192]),
    ("c1d128", "ragged3", [1023, 2048, 509]),
    ("c1d128", "edgepack", EDGE_PACK),
    ("c1d128", "1x65536", [65536]),
    ("c1d128", "1x131072", [131072]),
    ("c2d128", "1x8192", [8192]),
    ("c2d128", "edgepack", EDGE_PACK),
    ("c2d128", "1x65536", [65536]),
    ("c1d512", "1x8192", [8192]),
    ("c1d512", "1x65536", [65536]),
    ("c1d512", "1x131072", [131072]),
    ("c2d512", "1x8192", [8192]),
    ("c2d512", "1x65536", [65536]),
    ("c2d512", "1x131072", [131072]),
    # MUSA-specific bucket selectors not present in the CUDA competitor.
    ("c1d512", "1x256", [256]),
    ("c2d128", "1x16384", [16384]),
]
SCALED_CASES = [("c2d128", "1x8192*64", [8192], 64.0)]
PADDING_CASES = [
    ("c1d512", 8, 37),
    ("c2d512", 8, 21),
    ("c1d128", 5, 37),
    ("c2d128", 5, 21),
]


def _load_test_helpers():
    path = ROOT / "tests/python/test_ops/test_csa_compressor.py"
    spec = importlib.util.spec_from_file_location("csa_compressor_test_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CSA test helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schedule_case(coff: int, head_dim: int, rows: int):
    fwd = _forward_kernel_schedule_r128(RATIO, head_dim, coff, rows, 2)
    bwd = fused_backward_schedule_r128(RATIO, head_dim, coff, rows)
    return fwd, bwd, None


def _shipped_schedules():
    """Enumerate every schedule bucket represented by the dispatch constants."""

    boundaries = {1, 64, 128, 512, 1024, 8192}
    for shape in CONFIGS.values():
        for limit, _ in _FORWARD_TMA_ROW_BUCKETS[shape]:
            if limit is not None:
                boundaries.update((limit, limit + 1))
        tilewarp_limit, _ = _FORWARD_TILEWARP_ROW_BUCKETS[shape]
        boundaries.update((tilewarp_limit, tilewarp_limit + 1))
        for limit, *_ in _ROWGROUP_BWD_ROW_BUCKETS[shape]:
            if limit is not None:
                boundaries.update((limit, limit + 1))
        for limit, *_ in _TILEWARP_BWD_ROW_BUCKETS[shape]:
            boundaries.update((limit, limit + 1))
    fwd = set()
    bwd = set()
    for coff, head_dim in CONFIGS.values():
        for rows in boundaries:
            fs, bs, _ = _schedule_case(coff, head_dim, rows)
            fwd.add(((coff, head_dim), fs))
            bwd.add(((coff, head_dim), bs))
    return fwd, bwd


def _make_apis(test, kv, score, ape, cu, cuc, total_comp, head_dim, coff):
    sample_out = torch.empty((total_comp, head_dim), dtype=torch.bfloat16, device="meta")
    fwd = CSACompressorForward(kv, score, ape, cu, cuc, sample_out, ratio=RATIO, coff=coff)
    bwd = CSACompressorBackward(kv, score, ape, cu, cuc, sample_out, ratio=RATIO, coff=coff)
    fwd.check_support()
    bwd.check_support()
    fwd.compile()
    bwd.compile()
    return fwd, bwd


def _run_direct(fwd, bwd, kv, score, ape, cu, cuc, grad_out, poison=False):
    total_comp, head_dim = (int(x) for x in grad_out.shape)
    out = torch.empty((total_comp, head_dim), device="musa", dtype=torch.bfloat16)
    grad_kv = torch.empty_like(kv)
    grad_score = torch.empty_like(score)
    if poison:
        out.fill_(float("nan"))
        grad_kv.fill_(float("nan"))
        grad_score.fill_(float("nan"))
    else:
        out.zero_()
        grad_kv.zero_()
        grad_score.zero_()
    grad_ape = torch.zeros_like(ape, dtype=torch.float32)
    fwd.execute(kv, score, ape, cu, cuc, out)
    bwd.execute(kv, score, ape, cu, cuc, grad_out, grad_kv, grad_score, grad_ape)
    torch.musa.synchronize()
    return out, grad_kv, grad_score, grad_ape


def _tol_check(actual, reference, tolerance):
    diff = (actual.float() - reference.float()).abs()
    n_diff = int((actual != reference).sum().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    ok = n_diff <= max(1, int(DIFF_FRAC * actual.numel())) and max_abs <= tolerance
    return bool(ok), n_diff, max_abs


def _forward_tol_check(actual, reference, scale=1.0):
    """Check BF16 forward output against the FP32 online-softmax oracle."""
    diff = (actual.float() - reference.float()).abs()
    n_diff = int((actual != reference).sum().item())
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    ok = torch.isclose(
        actual.float(), reference.float(), rtol=0.01, atol=0.001 * scale
    ).all().item()
    return bool(ok), n_diff, max_abs


def _oracle_check(actual, eager, oracle):
    fused_error = float((actual.double().cpu() - oracle.double().cpu()).abs().max().item())
    eager_error = float((eager.double().cpu() - oracle.double().cpu()).abs().max().item())
    return (
        fused_error <= eager_error * (1 + 1e-6) + 1e-4,
        fused_error,
        eager_error,
    )


def _run_finite_case(test, config, shape_name, lens, covered, scale=1.0):
    coff, head_dim = CONFIGS[config]
    kv, score, ape, cu, cuc, total_comp = test._make_inputs(lens, head_dim, RATIO, coff, device="musa")
    grad_out = test._make_grad_out(total_comp, head_dim, device="musa")
    if scale != 1.0:
        if not math.log2(scale).is_integer():
            raise ValueError("scaled gate inputs must use an exact power-of-two scale")
        kv = (kv.float() * scale).bfloat16()
        grad_out = (grad_out.float() * scale).bfloat16()

    fwd_schedule, bwd_schedule, shared = _schedule_case(coff, head_dim, total_comp)
    covered["fwd"].add(((coff, head_dim), fwd_schedule))
    covered["bwd"].add(((coff, head_dim), bwd_schedule))
    fwd_api, bwd_api = _make_apis(test, kv, score, ape, cu, cuc, total_comp, head_dim, coff)
    runs = [
        _run_direct(
            fwd_api,
            bwd_api,
            kv,
            score,
            ape,
            cu,
            cuc,
            grad_out,
            poison=index != 0,
        )
        for index in range(3)
    ]
    fused = runs[0]
    stable = all(torch.equal(fused[index], other[index]) for other in runs[1:] for index in range(3))
    dape_replay_delta = max(float((fused[3] - other[3]).abs().max().item()) for other in runs[1:])
    dape_stable = all(torch.allclose(fused[3], other[3], rtol=0, atol=1e-3) for other in runs[1:])
    nan_ok = not any(torch.isnan(t).any().item() for run in runs for t in run[:3])

    fp32 = test._run_eager_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        total_comp,
        RATIO,
        head_dim,
        coff,
        grad_out,
        "fp32",
    )
    fp64 = test._run_eager_fwd_bwd(
        kv.cpu(),
        score.cpu(),
        ape.cpu(),
        cu.cpu(),
        cuc.cpu(),
        total_comp,
        RATIO,
        head_dim,
        coff,
        grad_out.cpu(),
        "fp64",
    )
    fwd_tol, fwd_nd, fwd_ma = _forward_tol_check(fused[0], fp32[0], scale)
    dkv_tol, dkv_nd, dkv_ma = _tol_check(fused[1], fp32[1], ABS_TOL * scale)
    dscore_tol, dscore_nd, dscore_ma = _tol_check(fused[2], fp32[2], ABS_TOL * scale * scale)
    dape_ma = float((fused[3] - fp32[3]).abs().max().item())
    dape_tol = dape_ma <= DAPE_TOL * scale * scale
    fwd_oracle, fwd_err, fwd_eager_err = _oracle_check(fused[0], fp32[0], fp64[0])
    dkv_oracle, dkv_err, dkv_eager_err = _oracle_check(fused[1], fp32[1], fp64[1])
    dscore_oracle, dscore_err, dscore_eager_err = _oracle_check(fused[2], fp32[2], fp64[2])
    dape_oracle, dape_err, dape_eager_err = _oracle_check(fused[3], fp32[3], fp64[3])
    # dAPE uses FP32 atomics.  Its replay delta is recorded, but scaled inputs
    # deliberately do not gate that absolute delta (the competitor contract
    # only gates deterministic output tensors after exponent shifts).
    g1 = stable and nan_ok and (dape_stable or scale != 1.0)
    g2 = fwd_tol and dkv_tol and dscore_tol and dape_tol
    # dAPE atomic reduction order is not gated for the scaled case, matching the
    # competitor contract; deterministic output tensors remain oracle-gated.
    g3 = fwd_oracle and dkv_oracle and dscore_oracle and (dape_oracle or scale != 1.0)
    ok = bool(g1 and g2 and g3)
    record = {
        "config": config,
        "shape": shape_name,
        "scale": scale,
        "ok": ok,
        "g1_deterministic": g1,
        "g2_tolerance": g2,
        "g3_oracle": g3,
        "stable": stable,
        "dape_replay_delta": dape_replay_delta,
        "nan_ok": nan_ok,
        "forward_schedule": list(fwd_schedule),
        "backward_schedule": list(bwd_schedule),
        "shared_bytes": shared,
        "fwd": {
            "n_diff": fwd_nd,
            "max_abs": fwd_ma,
            "err_fused": fwd_err,
            "err_eager": fwd_eager_err,
            "oracle_ok": fwd_oracle,
        },
        "dkv": {
            "n_diff": dkv_nd,
            "max_abs": dkv_ma,
            "err_fused": dkv_err,
            "err_eager": dkv_eager_err,
            "oracle_ok": dkv_oracle,
        },
        "dscore": {
            "n_diff": dscore_nd,
            "max_abs": dscore_ma,
            "err_fused": dscore_err,
            "err_eager": dscore_eager_err,
            "oracle_ok": dscore_oracle,
        },
        "dape": {
            "max_abs": dape_ma,
            "oracle_ok": dape_oracle,
            "parity_gated": scale == 1.0,
            "err_fused": dape_err,
            "err_eager": dape_eager_err,
        },
    }
    print(
        f"{config:7s} {shape_name:14s} {'PASS' if ok else 'FAIL'} "
        f"g1={'Y' if g1 else 'N'} g2={'Y' if g2 else 'N'} "
        f"g3={'Y' if g3 else 'N'} fwd_ma={fwd_ma:.2e} "
        f"dKV_ma={dkv_ma:.2e} dS_ma={dscore_ma:.2e} dAPE={dape_ma:.2e}",
        flush=True,
    )
    return record


def _run_padding_case(test, config, pad, token_pad, covered):
    coff, head_dim = CONFIGS[config]
    lens = [1023, 2048, 509]
    kv, score, ape, cu, cuc, total_true = test._make_inputs(lens, head_dim, RATIO, coff, device="musa")
    width = coff * head_dim
    generator = torch.Generator(device="cpu").manual_seed(13)
    extra_kv = torch.randn(token_pad, width, generator=generator).bfloat16().to("musa")
    extra_score = torch.randn(token_pad, width, generator=generator).bfloat16().to("musa")
    kv_full = torch.cat((kv, extra_kv))
    score_full = torch.cat((score, extra_score))
    capacity = total_true + pad
    grad_out = test._make_grad_out(capacity, head_dim, seed=11, device="musa")
    grad_out_ref = grad_out.clone()
    grad_out_ref[total_true:] = 0

    fwd_schedule, bwd_schedule, shared = _schedule_case(coff, head_dim, capacity)
    covered["fwd"].add(((coff, head_dim), fwd_schedule))
    covered["bwd"].add(((coff, head_dim), bwd_schedule))
    fwd_api, bwd_api = _make_apis(test, kv_full, score_full, ape, cu, cuc, capacity, head_dim, coff)
    runs = [
        _run_direct(
            fwd_api,
            bwd_api,
            kv_full,
            score_full,
            ape,
            cu,
            cuc,
            grad_out,
            poison=index != 0,
        )
        for index in range(3)
    ]
    fused = runs[0]
    stable = all(torch.equal(fused[index], other[index]) for other in runs[1:] for index in range(3))
    stable = stable and all(torch.allclose(fused[3], other[3], rtol=0, atol=1e-3) for other in runs[1:])
    nan_ok = not any(torch.isnan(t).any().item() for run in runs for t in run[:3])

    # Model token-capacity padding as a separate short segment.  Appending to
    # the final ragged segment could cross its next ratio boundary (the local
    # reference validates compressed counts), while a zero-compressed segment
    # preserves the original valid-row count and keeps the padding unconsumed.
    cu_values = [int(value) for value in cu.cpu().tolist()]
    cuc_values = [int(value) for value in cuc.cpu().tolist()]
    cu_ref = torch.tensor(cu_values + [cu_values[-1] + token_pad], dtype=torch.int32)
    cuc_ref = torch.tensor(cuc_values + [cuc_values[-1]], dtype=torch.int32)
    ref32 = test._run_eager_fwd_bwd(
        kv_full,
        score_full,
        ape,
        cu_ref.to("musa"),
        cuc_ref.to("musa"),
        capacity,
        RATIO,
        head_dim,
        coff,
        grad_out_ref,
        "fp32",
    )
    ref64 = test._run_eager_fwd_bwd(
        kv_full.cpu(),
        score_full.cpu(),
        ape.cpu(),
        cu_ref,
        cuc_ref,
        capacity,
        RATIO,
        head_dim,
        coff,
        grad_out_ref.cpu(),
        "fp64",
    )
    fwd_tol, fwd_nd, fwd_ma = _forward_tol_check(fused[0], ref32[0])
    dkv_tol, dkv_nd, dkv_ma = _tol_check(fused[1], ref32[1], ABS_TOL)
    dscore_tol, dscore_nd, dscore_ma = _tol_check(fused[2], ref32[2], ABS_TOL)
    dape_ma = float((fused[3] - ref32[3]).abs().max().item())
    fwd_oracle, _, _ = _oracle_check(fused[0], ref32[0], ref64[0])
    dkv_oracle, _, _ = _oracle_check(fused[1], ref32[1], ref64[1])
    dscore_oracle, _, _ = _oracle_check(fused[2], ref32[2], ref64[2])
    dape_oracle, _, _ = _oracle_check(fused[3], ref32[3], ref64[3])
    token_zero = bool((fused[1][kv.shape[0] :] == 0).all().item() and (fused[2][kv.shape[0] :] == 0).all().item())
    g1 = stable and nan_ok
    g2 = fwd_tol and dkv_tol and dscore_tol and dape_ma <= DAPE_TOL
    g3 = fwd_oracle and dkv_oracle and dscore_oracle and dape_oracle
    ok = bool(g1 and g2 and g3 and token_zero)
    record = {
        "kind": "padding",
        "config": config,
        "row_pad": pad,
        "token_pad": token_pad,
        "ok": ok,
        "g1_deterministic": g1,
        "g2_tolerance": g2,
        "g3_oracle": g3,
        "token_padding_zero": token_zero,
        "forward_schedule": list(fwd_schedule),
        "backward_schedule": list(bwd_schedule),
        "shared_bytes": shared,
        "fwd_n_diff": fwd_nd,
        "fwd_max_abs": fwd_ma,
        "dkv_n_diff": dkv_nd,
        "dkv_max_abs": dkv_ma,
        "dscore_n_diff": dscore_nd,
        "dscore_max_abs": dscore_ma,
        "dape_max_abs": dape_ma,
    }
    print(
        f"{config:7s} padding rows+{pad} tokens+{token_pad}: "
        f"{'PASS' if ok else 'FAIL'} g1={'Y' if g1 else 'N'} "
        f"g2={'Y' if g2 else 'N'} g3={'Y' if g3 else 'N'} "
        f"token_zero={'Y' if token_zero else 'N'}",
        flush=True,
    )
    return record


def _run_overflow_case(test, covered):
    config = "c1d128"
    coff, head_dim = CONFIGS[config]
    total_tokens = RATIO
    total_comp = 1
    width = coff * head_dim
    kv = torch.ones((total_tokens, width), device="musa", dtype=torch.bfloat16)
    score = torch.full(
        (total_tokens, width),
        torch.finfo(torch.bfloat16).max,
        device="musa",
        dtype=torch.bfloat16,
    )
    ape = torch.full(
        (RATIO, width),
        torch.finfo(torch.float32).max,
        device="musa",
        dtype=torch.float32,
    )
    cu = torch.tensor([0, total_tokens], device="musa", dtype=torch.int32)
    cuc = torch.tensor([0, total_comp], device="musa", dtype=torch.int32)
    grad_out = torch.ones((total_comp, head_dim), device="musa", dtype=torch.bfloat16)
    fwd_schedule, bwd_schedule, shared = _schedule_case(coff, head_dim, total_comp)
    covered["fwd"].add(((coff, head_dim), fwd_schedule))
    covered["bwd"].add(((coff, head_dim), bwd_schedule))
    fwd_api, bwd_api = _make_apis(test, kv, score, ape, cu, cuc, total_comp, head_dim, coff)
    runs = [
        _run_direct(
            fwd_api,
            bwd_api,
            kv,
            score,
            ape,
            cu,
            cuc,
            grad_out,
            poison=index != 0,
        )
        for index in range(3)
    ]
    fused = runs[0]
    ref32 = test._run_eager_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        total_comp,
        RATIO,
        head_dim,
        coff,
        grad_out,
        "fp32",
    )
    ref64 = test._run_eager_fwd_bwd(
        kv.cpu(),
        score.cpu(),
        ape.cpu(),
        cu.cpu(),
        cuc.cpu(),
        total_comp,
        RATIO,
        head_dim,
        coff,
        grad_out.cpu(),
        "fp64",
    )
    replay_stable = all(torch.equal(fused[index].cpu().view(torch.uint8), other[index].cpu().view(torch.uint8)) for other in runs[1:] for index in range(4))
    mask_equal = all(torch.equal(torch.isnan(fused[index]).cpu(), torch.isnan(ref32[index]).cpu()) for index in range(4))
    oracle_finite = all(bool(torch.isfinite(t).all().item()) for t in ref64)
    out_oracle_finite = bool(torch.allclose(ref64[0], torch.ones_like(ref64[0])))
    ok = bool(replay_stable and mask_equal and oracle_finite and out_oracle_finite)
    record = {
        "kind": "overflow",
        "config": config,
        "shape": "fp32_score_plus_ape_inf",
        "ok": ok,
        "replay_stable": replay_stable,
        "nan_mask_equal_to_fp32": mask_equal,
        "fp64_oracle_finite": oracle_finite,
        "fp64_out_is_one": out_oracle_finite,
        "forward_schedule": list(fwd_schedule),
        "backward_schedule": list(bwd_schedule),
        "shared_bytes": shared,
    }
    print(
        f"{config:7s} overflow {'PASS' if ok else 'FAIL'} "
        f"replay={'Y' if replay_stable else 'N'} "
        f"nan_mask={'Y' if mask_equal else 'N'} "
        f"fp64_finite={'Y' if oracle_finite else 'N'}",
        flush=True,
    )
    return record


def _validate_reference(test):
    """Validate the shared eager reference against the production short-ratio path."""

    coff, head_dim, ratio = 2, 128, 4
    kv, score, ape, cu, cuc, total_comp = test._make_inputs([2048], head_dim, ratio, coff, device="musa")
    grad_out = test._make_grad_out(total_comp, head_dim, device="musa")
    fused = test._run_fused_fwd_bwd(kv, score, ape, cu, cuc, total_comp, ratio, head_dim, coff, grad_out)
    fp32 = test._run_eager_fwd_bwd(kv, score, ape, cu, cuc, total_comp, ratio, head_dim, coff, grad_out, "fp32")
    upstream = test._run_eager_fwd_bwd(kv, score, ape, cu, cuc, total_comp, ratio, head_dim, coff, grad_out, "upstream")
    test._assert_grads_vs_fp32(fused[1], fused[2], fp32[1], fp32[2], ratio)
    fwd_ok = _forward_tol_check(fused[0], fp32[0])[0]
    dape_ok = bool((fused[3] - fp32[3]).abs().max().item() <= DAPE_TOL)
    upstream_ok = all(torch.allclose(a.float(), b.float(), rtol=0, atol=0.1) for a, b in zip(fused, upstream))
    ok = bool(fwd_ok and dape_ok and upstream_ok)
    record = {
        "ok": ok,
        "ratio": ratio,
        "coff": coff,
        "head_dim": head_dim,
        "forward_fp32_ok": fwd_ok,
        "dape_fp32_ok": dape_ok,
        "upstream_tolerance_ok": upstream_ok,
    }
    print(
        f"reference r{ratio}/c{coff}: {'PASS' if ok else 'FAIL'} "
        f"fp32={'Y' if fwd_ok else 'N'} dAPE={'Y' if dape_ok else 'N'} "
        f"upstream={'Y' if upstream_ok else 'N'}",
        flush=True,
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if not torch.musa.is_available():
        parser.error("MUSA device is required")
    test = _load_test_helpers()
    covered = {"fwd": set(), "bwd": set()}
    reference = _validate_reference(test)
    selected = CASES[:2] if args.quick else CASES
    records = [_run_finite_case(test, config, shape, lens, covered) for config, shape, lens in selected]
    missing = []
    if not args.quick:
        records.extend(_run_finite_case(test, config, shape, lens, covered, scale=scale) for config, shape, lens, scale in SCALED_CASES)
        records.extend(_run_padding_case(test, config, pad, token_pad, covered) for config, pad, token_pad in PADDING_CASES)
        records.append(_run_overflow_case(test, covered))
        shipped_fwd, shipped_bwd = _shipped_schedules()
        missing = [("fwd", *item) for item in sorted(shipped_fwd - covered["fwd"], key=str)] + [
            ("bwd", *item) for item in sorted(shipped_bwd - covered["bwd"], key=str)
        ]
    all_pass = bool(reference["ok"] and all(item["ok"] for item in records) and not missing)
    print(
        f"GATE {'PASS' if all_pass else 'FAIL'} "
        f"{sum(item['ok'] for item in records)}/{len(records)} "
        f"reference={'PASS' if reference['ok'] else 'FAIL'} "
        f"schedule_coverage={len(covered['fwd'])}/"
        f"{len(_shipped_schedules()[0]) if not args.quick else '-'}" + (f" missing={missing}" if missing else ""),
        flush=True,
    )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "all_pass": all_pass,
                    "reference": reference,
                    "cases": records,
                    "missing": missing,
                },
                indent=2,
            )
        )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
