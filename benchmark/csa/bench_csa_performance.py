"""Measure production CSA R4/R128 forward and backward cases on MP32.

The case list mirrors ``doc/csa_compressor_performance_comparison.md``.  Forward and
backward are launched through the production R4/R128 dispatchers.  Bandwidth
uses the same logical-byte convention as the competitor table:

* forward: ``4 * tokens * (coff*d) + 2 * rows * d``;
* backward: ``8 * tokens * (coff*d)``.

These are normalized logical bandwidths, not hardware DRAM counters.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

import tilelang  # noqa: E402
import tilelang.contrib.mcc as mcc_backend  # noqa: E402
import torch  # noqa: E402
import torch_musa  # noqa: F401,E402

from mudnn.csa.compressor import compressor_sm100 as r4  # noqa: E402
from mudnn.csa.compressor import compressor_sm100_r128 as r128  # noqa: E402

# name: (ratio, coff, head_dim, segment token counts)
CASES = {
    "r4_c2_1x8192_d128": (4, 2, 128, (8192,)),
    "r4_c2_3x8192_d128": (4, 2, 128, (8192, 8192, 8192)),
    "r4_c2_1x8192_d512": (4, 2, 512, (8192,)),
    "r4_c2_3x8192_d512": (4, 2, 512, (8192, 8192, 8192)),
    "r128_c1_131072_d128": (128, 1, 128, (131072,)),
    "r128_c2_65536_d128": (128, 2, 128, (65536,)),
    "r128_c1_65536_d512": (128, 1, 512, (65536,)),
    "r128_c2_65536_d512": (128, 2, 512, (65536,)),
    "r128_c2_131072_d512": (128, 2, 512, (131072,)),
}
DEFAULT_CASES = tuple(CASES)

# Large-row production cases are selected explicitly with --cases.  Keep the
# default nine-case comparison unchanged because these inputs are much larger.
CASES.update({
    "r4_c2_rows344064_d128": (4, 2, 128, (344064 * 4,)),
    "r4_c2_rows86016_d512": (4, 2, 512, (86016 * 4,)),
    "r128_c1_rows65536_d128": (128, 1, 128, (65536 * 128,)),
    "r128_c1_rows4096_d512": (128, 1, 512, (4096 * 128,)),
})


def _inputs(name, ratio, coff, d, segments):
    seed = 20261030 + list(CASES).index(name)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = sum(segments)
    rows_per_segment = tuple(tokens_i // ratio for tokens_i in segments)
    rows = sum(rows_per_segment)
    width = coff * d
    kv = torch.randn((tokens, width), generator=generator).to(torch.bfloat16).to("musa")
    score = torch.randn((tokens, width), generator=generator).to(torch.bfloat16).to("musa")
    ape = torch.randn((ratio, width), generator=generator).to("musa")
    cu = [0]
    cuc = [0]
    for token_count, row_count in zip(segments, rows_per_segment):
        cu.append(cu[-1] + token_count)
        cuc.append(cuc[-1] + row_count)
    cu = torch.tensor(cu, dtype=torch.int32, device="musa")
    cuc = torch.tensor(cuc, dtype=torch.int32, device="musa")
    out = torch.empty((rows, d), dtype=torch.bfloat16, device="musa")
    grad_out = torch.randn((rows, d), generator=generator).to(torch.bfloat16).to("musa")
    grad_kv = torch.empty_like(kv)
    grad_score = torch.empty_like(score)
    grad_ape = torch.empty((ratio, width), dtype=torch.float32, device="musa")
    return kv, score, ape, cu, cuc, out, grad_out, grad_kv, grad_score, grad_ape, rows


def _sample_rows(cuc, rows):
    boundaries = [int(x) for x in cuc.cpu().tolist()]
    result = set()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        result.update((start, min(start + 1, end - 1), (start + end - 1) // 2, end - 1))
    return sorted(x for x in result if 0 <= x < rows)


def _reference(kv, score, ape, cu, cuc, ratio, coff, d, rows, sample_rows):
    expected = []
    cu_host = [int(x) for x in cu.cpu().tolist()]
    cuc_host = [int(x) for x in cuc.cpu().tolist()]
    for row in sample_rows:
        segment = next(i for i in range(len(cuc_host) - 1) if cuc_host[i] <= row < cuc_host[i + 1])
        block = row - cuc_host[segment]
        tok0 = cu_host[segment] + block * ratio
        if coff == 1:
            values = kv[tok0 : tok0 + ratio, :d].float()
            logits = score[tok0 : tok0 + ratio, :d].float() + ape[:, :d]
        else:
            right = slice(tok0, tok0 + ratio)
            values = kv[right, d:].float()
            logits = score[right, d:].float() + ape[:, d:]
            if block > 0:
                left = slice(tok0 - ratio, tok0)
                values = torch.cat((kv[left, :d].float(), values), dim=0)
                logits = torch.cat((score[left, :d].float() + ape[:, :d], logits), dim=0)
        expected.append((torch.softmax(logits, dim=0) * values).sum(dim=0).bfloat16())
    return torch.stack(expected)


def _time_production(launch, num_tests, rounds, compute_kernels_per_launch=None):
    samples = []
    event_breakdown = []
    active_steps = 2 * num_tests - 1
    for _ in range(rounds):
        launch()
        torch.musa.synchronize()
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=active_steps, repeat=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.MUSA], schedule=schedule) as profiler:
            for _ in range(2 * num_tests):
                launch()
                profiler.step()
            torch.musa.synchronize()
        events = [event for event in profiler.key_averages() if event.device_time_total > 0]
        if not events:
            raise RuntimeError("Kineto returned no MUSA kernel events")
        event_breakdown.append({
            event.key: {
                "time_us_per_launch": event.device_time_total / active_steps,
                "count_per_launch": event.count / active_steps,
            }
            for event in events
        })
        if compute_kernels_per_launch is None:
            selected = events
        else:
            # TileLang emits both the backward main and dAPE reduce as
            # `main_kernel`; MUSA fill kernels have distinct names. Validate
            # the event count so a compiler naming change cannot skew results.
            selected = [event for event in events if event.key == "main_kernel"]
            count = sum(event.count for event in selected)
            expected = active_steps * compute_kernels_per_launch
            if count != expected:
                raise RuntimeError(
                    f"expected {expected} TileLang backward kernel events, got {count}; "
                    f"MUSA events: {event_breakdown[-1]}"
                )
        samples.append(sum(event.device_time_total for event in selected) / active_steps)
    return samples, event_breakdown


def _run_case(name, num_tests, rounds, mode):
    ratio, coff, d, segments = CASES[name]
    tensors = _inputs(name, ratio, coff, d, segments)
    kv, score, ape, cu, cuc, out, grad_out, grad_kv, grad_score, grad_ape, rows = tensors
    width = coff * d
    sample_rows = _sample_rows(cuc, rows)

    if mode == "forward":
        if ratio == 4:
            schedule = r4._forward_strategy(ratio, d, coff, rows, int(cu.shape[0]))
            r4.precompile_fwd(ratio, d, coff, kv.device, nb_total=rows, num_offsets=int(cu.shape[0]))

            def launch():
                r4.run_fwd(kv, score, ape, cu, cuc, out, rows, ratio, d, coff)

        else:
            schedule = r128._forward_kernel_schedule_r128(ratio, d, coff, rows, int(cu.shape[0]))
            r128.precompile_fwd_r128(ratio, d, coff, kv.device, nb_total=rows, num_offsets=int(cu.shape[0]))

            def launch():
                r128.run_fwd_r128(kv, score, ape, cu, cuc, out, rows, ratio, d, coff)

        launch()
        torch.musa.synchronize()
        expected = _reference(kv, score, ape, cu, cuc, ratio, coff, d, rows, sample_rows)
        delta = (out[sample_rows].float() - expected.float()).abs()
        finite = bool(torch.isfinite(out).all().item())
        max_abs = float(delta.max().item())
        if ratio == 128:
            accurate = bool(torch.isclose(out[sample_rows].float(), expected.float(), rtol=0.01, atol=0.001).all().item())
        else:
            accurate = max_abs <= 0.015625
        if not finite or not accurate:
            raise AssertionError(f"{name}: forward reference mismatch, max_abs={max_abs}")
        logical_bytes = 4 * (sum(segments)) * width + 2 * rows * d
    else:
        if ratio == 4:
            schedule = r4._backward_r4_vec4_schedule(
                ratio, d, coff, rows, int(cu.shape[0])
            )
            r4.precompile_bwd(
                ratio, d, coff, kv.device,
                nb_total=rows, num_offsets=int(cu.shape[0]),
            )

            def launch():
                grad_kv.zero_()
                grad_score.zero_()
                grad_ape.zero_()
                r4.run_bwd(
                    kv,
                    score,
                    ape,
                    cu,
                    cuc,
                    grad_out,
                    grad_kv,
                    grad_score,
                    grad_ape,
                    rows,
                    ratio,
                    d,
                    coff,
                )

        else:
            schedule = r128.fused_backward_schedule_r128(ratio, d, coff, rows)
            r128.precompile_bwd_r128(ratio, d, coff, kv.device, nb_total=rows)

            def launch():
                grad_kv.zero_()
                grad_score.zero_()
                grad_ape.zero_()
                r128.run_bwd_r128(
                    kv,
                    score,
                    ape,
                    cu,
                    cuc,
                    grad_out,
                    grad_kv,
                    grad_score,
                    grad_ape,
                    rows,
                    ratio,
                    d,
                    coff,
                )

        launch()
        torch.musa.synchronize()
        finite = all(bool(torch.isfinite(x).all().item()) for x in (grad_kv, grad_score, grad_ape))
        if not finite:
            raise AssertionError(f"{name}: nonfinite backward gradient")
        max_abs = None
        logical_bytes = 8 * sum(segments) * width

    compute_kernels_per_launch = None
    if mode == "backward":
        compute_kernels_per_launch = 1 if ratio == 4 and schedule[2] else 2
    samples, event_breakdown = _time_production(
        launch, num_tests, rounds, compute_kernels_per_launch
    )
    median_us = statistics.median(samples)
    result = {
        "case": name,
        "mode": mode,
        "ratio": ratio,
        "coff": coff,
        "head_dim": d,
        "segments_tokens": list(segments),
        "tokens": sum(segments),
        "rows": rows,
        "schedule": schedule,
        "sample_rows": sample_rows,
        "finite": finite,
        "sample_fp32_max_abs": max_abs,
        "samples_us": samples,
        "event_breakdown_by_round": event_breakdown,
        "compute_kernels_per_launch": compute_kernels_per_launch,
        "median_us": median_us,
        "logical_bytes": logical_bytes,
        "logical_tbps": logical_bytes / (median_us * 1e6),
        "timing_scope": (
            "Kineto TileLang main_kernel events (backward main + dAPE reduce); "
            "gradient and scratch zero kernels, JIT, and allocation excluded"
            if mode == "backward" else
            "Kineto MUSA kernel events; JIT and allocation excluded"
        ),
    }
    del tensors
    torch.musa.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--modes", nargs="+", choices=("forward", "backward"), default=("forward", "backward"))
    parser.add_argument("--num-tests", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mcc", type=Path, required=True)
    args = parser.parse_args()
    if not torch.musa.is_available() or not args.mcc.is_file():
        parser.error("MUSA device and MCC compiler are required")
    mcc_backend.get_mcc_compiler = lambda: str(args.mcc)
    torch.musa.set_device(0)
    payload = {
        "device": torch.musa.get_device_name(),
        "mcc": str(args.mcc),
        "mcc_version": subprocess.check_output([str(args.mcc), "--version"], text=True).strip(),
        "tilelang": str(Path(tilelang.__file__).resolve()),
        "block_schedule_mode": os.environ.get("MUSA_BLOCK_SCHEDULE_MODE"),
        "source_sha256": {
            "r4": hashlib.sha256(Path(r4.__file__).read_bytes()).hexdigest(),
            "r128": hashlib.sha256(Path(r128.__file__).read_bytes()).hexdigest(),
        },
        "timing": {
            "num_tests": args.num_tests, "rounds": args.rounds, "flush_l2": False,
            "backward_scope": "TileLang main + dAPE reduce device events only; all zero kernels excluded",
        },
        "bandwidth": "competitor-compatible unique logical tensor bytes; not physical DRAM traffic",
        "results": [],
    }
    for mode in args.modes:
        for name in args.cases:
            result = _run_case(name, args.num_tests, args.rounds, mode)
            payload["results"].append(result)
            print(json.dumps(result), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
