"""Resource probe for every published TileLang ratio=128 kernel variant.

This is the MUSA counterpart of the competitor's ptxas probe.  It derives the
forward/backward schedule union from the dispatch functions, JIT-compiles each
unique variant with ``TILELANG_DUMP_KERNEL_SOURCE``, and invokes ``mcc
-resource-usage -c`` on each generated source.  The probe fails on register,
local-memory, or spill regressions.

Example::

  PYTHONPATH=python python benchmark/csa/reg_probe_csa_compressor_r128.py
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch_musa  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from mudnn.csa.compressor.compressor_sm100_r128 import (  # noqa: E402
    _FORWARD_TMA_ROW_BUCKETS,
    _FORWARD_TILEWARP_ROW_BUCKETS,
    _ROWGROUP_BWD_ROW_BUCKETS,
    _TILEWARP_BWD_ROW_BUCKETS,
    fused_backward_schedule_r128,
    _forward_kernel_schedule_r128,
    precompile_bwd_r128,
    precompile_fwd_r128,
)
from mudnn.csa.compressor._jit import COMPILER_COMPILE_FLAGS  # noqa: E402


CONFIGS = ((1, 128), (2, 128), (1, 512), (2, 512))
REGISTER_RE = re.compile(r"Register usage.*?Temp\s+(\d+)")
LOCAL_RE = re.compile(r"Local memory:\s*(\d+)\s*Bytes")
SPILL_RE = re.compile(r"Spill (?:load|store):\s*(\d+)\s*Bytes")


def _unique_schedules():
    seen = {}
    for coff, head_dim in CONFIGS:
        shape = coff, head_dim
        boundaries = {1, 64, 128, 512, 1024, 8192}
        for limit, _ in _FORWARD_TMA_ROW_BUCKETS[shape]:
            if limit is not None:
                boundaries.update((limit, limit + 1))
        forward_tilewarp_limit, _ = _FORWARD_TILEWARP_ROW_BUCKETS[shape]
        boundaries.update((forward_tilewarp_limit, forward_tilewarp_limit + 1))
        for limit, *_ in _ROWGROUP_BWD_ROW_BUCKETS[shape]:
            if limit is not None:
                boundaries.update((limit, limit + 1))
        for limit, *_ in _TILEWARP_BWD_ROW_BUCKETS[shape]:
            boundaries.update((limit, limit + 1))
        for rows in sorted(boundaries):
            fwd = _forward_kernel_schedule_r128(128, head_dim, coff, rows, 2)
            bwd = fused_backward_schedule_r128(128, head_dim, coff, rows)
            seen[("fwd", coff, head_dim, fwd)] = rows
            seen[("bwd", coff, head_dim, bwd)] = rows
    return sorted(seen.items(), key=lambda item: str(item[0]))


def _resource_usage(source: Path, include_root: Path):
    mcc = shutil.which("mcc")
    if mcc is None:
        raise RuntimeError("mcc is not available; run in the TileLang/MUSA environment")
    package = include_root
    command = [
        mcc,
        "--cuda-device-only",
        "--cuda-gpu-arch=mp_32",
        "-resource-usage",
        "-std=c++17",
        "-Wno-error=address-of-temporary",
        "-Wno-error=c++11-narrowing",
        "-c",
        f"-I{package / '3rdparty' / '..' / 'src'}",
        f"-I{package / '3rdparty' / 'mutlass' / 'include'}",
        "-D__MUSA_ARCH_LIST__=320",
        "-mllvm",
        "-mtgpu-alloc-shared-memory-from-zero=1",
        *COMPILER_COMPILE_FLAGS,
        str(source),
        "-o",
        str(source.with_suffix(".resource.o")),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout
    registers = [int(value) for value in REGISTER_RE.findall(output)]
    locals_ = [int(value) for value in LOCAL_RE.findall(output)]
    spills = [int(value) for value in SPILL_RE.findall(output)]
    return result.returncode, output, registers, locals_, spills


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-temp-registers", type=int, default=128)
    parser.add_argument("--keep-sources", type=Path, default=None)
    args = parser.parse_args()
    if not torch.musa.is_available():
        parser.error("MUSA device is required for TileLang JIT")
    device = torch.device("musa", torch.musa.current_device())

    try:
        import tilelang
    except ImportError as exc:
        parser.error(f"TileLang is required: {exc}")

    package = Path(tilelang.__file__).resolve().parent
    work = args.keep_sources or Path(tempfile.mkdtemp(prefix="csa_r128_mu_"))
    work.mkdir(parents=True, exist_ok=True)
    old_dump = os.environ.get("TILELANG_DUMP_KERNEL_SOURCE")
    old_cache = os.environ.get("TILELANG_CACHE_DIR")
    os.environ["TILELANG_DUMP_KERNEL_SOURCE"] = str(work / "kernel.mu")
    os.environ["TILELANG_CACHE_DIR"] = str(work / "tilelang-cache")

    results = []
    try:
        for (kind, coff, head_dim, schedule), rows in _unique_schedules():
            before = set(work.glob("*.mu"))
            if kind == "fwd":
                precompile_fwd_r128(128, head_dim, coff, device, nb_total=rows, num_offsets=2)
            else:
                precompile_bwd_r128(128, head_dim, coff, device, nb_total=rows)
            generated = sorted(set(work.glob("*.mu")) - before)
            if not generated:
                raise RuntimeError(f"TileLang did not dump source for {kind} {schedule}")
            # Backward precompile also emits a small dAPE reducer. Probe the
            # main kernel, which is the largest generated source.
            source = max(generated, key=lambda path: path.stat().st_size)
            rc, output, registers, locals_, spills = _resource_usage(source, package)
            clean = (
                rc == 0
                and bool(registers)
                and max(registers) <= args.max_temp_registers
                and (not locals_ or max(locals_) == 0)
                and (not spills or max(spills) == 0)
            )
            tag = f"{kind}_c{coff}d{head_dim}_{schedule}"
            result = {
                "tag": tag,
                "source": str(source),
                "register_temp": max(registers) if registers else None,
                "local_bytes": max(locals_) if locals_ else None,
                "spill_bytes": max(spills) if spills else None,
                "shared_bytes": None,
                "ok": clean,
            }
            results.append(result)
            print(
                f"{tag:52s} regs={result['register_temp']!s:>4} "
                f"local={result['local_bytes']!s:>6} spill={result['spill_bytes']!s:>6} "
                f"{'PASS' if clean else 'FAIL'}",
                flush=True,
            )
            if not clean:
                print(output, file=sys.stderr)
    finally:
        if old_dump is None:
            os.environ.pop("TILELANG_DUMP_KERNEL_SOURCE", None)
        else:
            os.environ["TILELANG_DUMP_KERNEL_SOURCE"] = old_dump
        if old_cache is None:
            os.environ.pop("TILELANG_CACHE_DIR", None)
        else:
            os.environ["TILELANG_CACHE_DIR"] = old_cache
        if args.keep_sources is None:
            shutil.rmtree(work, ignore_errors=True)

    ok = bool(results) and all(item["ok"] for item in results)
    print(f"RESOURCE {'PASS' if ok else 'FAIL'} {sum(item['ok'] for item in results)}/{len(results)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
