# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List


_THIS_DIR = Path(__file__).resolve().parent
_FRONTEND_DIR = _THIS_DIR.parents[2]
_PARENT_DIR = _FRONTEND_DIR.parent

_KERNEL_RE = re.compile(r"muModuleGetFunction\([^,]+,\s*[^,]+,\s*([^)]+)\)")
_PERF_RE = re.compile(r"PERF_MS\s+([0-9.eE+-]+)")
_ARCH_BY_COMPUTE_CAPABILITY = {
    (2, 2): "mp_22",
    (3, 1): "mp_31",
}


CASE_SPECS = {
    "concat_reshape": {
        "expected_kernel_count": 2,
    },
    "conv_relu": {
        "expected_kernel_count": 1,
    },
    "matmul_bias": {
        "expected_kernel_count": 1,
    },
    "matmul_bias_vector": {
        "expected_kernel_count": 1,
    },
    "matmul_bias_relu": {
        "expected_kernel_count": {
            "mp_22": 2,
            "mp_31": 1,
        },
    },
    "matmul_bias_relu_vector": {
        "expected_kernel_count": {
            "mp_22": 2,
            "mp_31": 1,
        },
    },
    "matmul_bias_relu_bf16": {
        "expected_kernel_count": {
            "mp_22": 2,
            "mp_31": 1,
        },
    },
    "matmul_reduce": {
        "expected_kernel_count": 2,
    },
    "pointwise_residual_gelu": {
        "expected_kernel_count": 2,
    },
    "pointwise_residual_gelu_bf16": {
        "expected_kernel_count": 2,
    },
    "reduction_sum": {
        "expected_kernel_count": 1,
    },
    "reduction_sum_bf16": {
        "expected_kernel_count": 1,
    },
    "softmax_last_dim": {
        "expected_kernel_count": 3,
    },
    "sdpa_causal": {
        "expected_kernel_count": 4,
    },
    "sdpa_causal_flash_f16": {
        "expected_kernel_count": 1,
    },
    "sdpa_causal_flash_bf16": {
        "expected_kernel_count": 1,
    },
}


def _current_compute_capability() -> tuple[int, int]:
    source = r"""
#include <iostream>
#include <musa_runtime_api.h>

int main() {
  int device = 0;
  auto status = musaGetDevice(&device);
  if (status != musaSuccess) {
    std::cerr << "musaGetDevice failed: " << static_cast<int>(status) << "\n";
    return 1;
  }
  musaDeviceProp prop{};
  status = musaGetDeviceProperties(&prop, device);
  if (status != musaSuccess) {
    std::cerr << "musaGetDeviceProperties failed: "
              << static_cast<int>(status) << "\n";
    return 2;
  }
  std::cout << prop.major << " " << prop.minor << "\n";
  return 0;
}
"""
    with tempfile.TemporaryDirectory(prefix="mudnn_frontend_arch_") as tmp:
        source_path = Path(tmp) / "detect_musa_arch.cc"
        binary_path = Path(tmp) / "detect_musa_arch"
        source_path.write_text(source, encoding="utf-8")
        compile_cmd = [
            os.getenv("CXX", "g++"),
            str(source_path),
            "-I/usr/local/musa/include",
            "-L/usr/local/musa/lib",
            "-lmusart",
            "-Wl,-rpath,/usr/local/musa/lib",
            "-o",
            str(binary_path),
        ]
        compile_proc = subprocess.run(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if compile_proc.returncode != 0:
            raise RuntimeError(
                "failed to build MUSA architecture detector:\n"
                + compile_proc.stderr[-4000:]
            )
        run_proc = subprocess.run(
            [str(binary_path)],
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if run_proc.returncode != 0:
            raise RuntimeError(
                "MUSA architecture detector failed:\n" + run_proc.stderr[-4000:]
            )
    fields = run_proc.stdout.strip().split()
    if len(fields) != 2:
        raise RuntimeError(
            "MUSA architecture detector produced unexpected output: "
            + run_proc.stdout.strip()
        )
    return int(fields[0]), int(fields[1])


@lru_cache(maxsize=1)
def kernel_golden_arch() -> str:
    capability = _current_compute_capability()
    arch = _ARCH_BY_COMPUTE_CAPABILITY.get(capability)
    if arch:
        return arch
    raise RuntimeError(
        "Unable to map MUSA compute capability "
        f"{capability[0]}.{capability[1]} to a kernel golden architecture."
    )


def expected_kernel_count(spec: Dict[str, object], arch: str) -> int:
    expected = spec["expected_kernel_count"]
    if isinstance(expected, dict):
        if arch not in expected:
            raise AssertionError(f"no kernel golden for architecture {arch}")
        return int(expected[arch])
    return int(expected)


def _prepend_env_path(
    env: Dict[str, str], name: str, paths: Iterable[Path | str]
) -> None:
    entries = [str(path) for path in paths if path and Path(path).exists()]
    current = env.get(name, "")
    if current:
        entries.append(current)
    env[name] = os.pathsep.join(entries)


def _framework_module_dirs() -> List[Path]:
    return [
        Path(os.environ["MUDNN_FRONTEND_FRAMEWORK_TEST_BUILD_DIR"])
        if os.environ.get("MUDNN_FRONTEND_FRAMEWORK_TEST_BUILD_DIR")
        else None,
        _FRONTEND_DIR / "build" / "tests" / "cpp",
        Path.cwd() / "build" / "tests" / "cpp",
        Path.cwd(),
    ]


def _backend_lib_dirs() -> List[Path]:
    return [
        Path(os.environ["MUDNN_BACKEND_LIB"])
        if os.environ.get("MUDNN_BACKEND_LIB")
        else None,
        _PARENT_DIR / "build" / "export" / "mudnn" / "lib",
        _PARENT_DIR / "build" / "mp_31" / "mudnn" / "lib",
        _PARENT_DIR / "build" / "mp_31" / "lib",
        Path("/usr/local/musa/lib"),
    ]


def framework_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["MUSA_LOG"] = "0xffff"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    _prepend_env_path(
        env,
        "PYTHONPATH",
        [
            *_framework_module_dirs(),
            _FRONTEND_DIR / "python",
            _FRONTEND_DIR / "tests" / "python",
        ],
    )
    _prepend_env_path(
        env, "LD_LIBRARY_PATH", [*_backend_lib_dirs(), *_framework_module_dirs()]
    )
    return env


_BF16_HELPER_CODE = """
def bf16(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    rounded = u + np.uint32(0x7fff) + lsb.astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)
"""


def case_code(case_name: str) -> str:
    cases = {
        "matmul_bias": """
import numpy as np
import mudnn_frontend_framework_test as f
a = np.linspace(-0.25, 0.25, 64 * 128, dtype=np.float32).reshape(64, 128)
b = np.linspace(0.15, -0.15, 128 * 96, dtype=np.float32).reshape(128, 96)
bias = np.linspace(-0.1, 0.1, 64 * 96, dtype=np.float32).reshape(64, 96)
r = f.matmul_bias(a, b, bias)
print("PERF_MS", r["average_ms"])
""",
        "matmul_bias_vector": """
import numpy as np
import mudnn_frontend_framework_test as f
a = np.linspace(-0.25, 0.25, 64 * 128, dtype=np.float32).reshape(64, 128)
b = np.linspace(0.15, -0.15, 128 * 96, dtype=np.float32).reshape(128, 96)
bias = np.linspace(-0.1, 0.1, 96, dtype=np.float32)
r = f.matmul_bias_vector(a, b, bias)
print("PERF_MS", r["average_ms"])
""",
        "matmul_bias_relu": """
import numpy as np
import mudnn_frontend_framework_test as f
a = np.linspace(-0.5, 0.5, 48 * 96, dtype=np.float32).reshape(48, 96)
b = np.linspace(0.2, -0.2, 96 * 64, dtype=np.float32).reshape(96, 64)
bias = np.linspace(-0.15, 0.15, 48 * 64, dtype=np.float32).reshape(48, 64)
r = f.matmul_bias_relu(a, b, bias)
print("PERF_MS", r["average_ms"])
""",
        "matmul_bias_relu_vector": """
import numpy as np
import mudnn_frontend_framework_test as f
a = np.linspace(-0.5, 0.5, 48 * 96, dtype=np.float32).reshape(48, 96)
b = np.linspace(0.2, -0.2, 96 * 64, dtype=np.float32).reshape(96, 64)
bias = np.linspace(-0.15, 0.15, 64, dtype=np.float32)
r = f.matmul_bias_relu_vector(a, b, bias)
print("PERF_MS", r["average_ms"])
""",
        "matmul_bias_relu_bf16": (
            """
import numpy as np
import mudnn_frontend_framework_test as f
"""
            + _BF16_HELPER_CODE
            + """
a = bf16(np.linspace(-0.5, 0.5, 48 * 96, dtype=np.float32).reshape(48, 96))
b = bf16(np.linspace(0.2, -0.2, 96 * 64, dtype=np.float32).reshape(96, 64))
bias = bf16(np.linspace(-0.15, 0.15, 48 * 64, dtype=np.float32).reshape(48, 64))
r = f.matmul_bias_relu_bf16(a, b, bias)
print("PERF_MS", r["average_ms"])
"""
        ),
        "matmul_reduce": """
import numpy as np
import mudnn_frontend_framework_test as f
a = np.linspace(-0.25, 0.25, 64 * 128, dtype=np.float32).reshape(64, 128)
b = np.linspace(0.15, -0.15, 128 * 96, dtype=np.float32).reshape(128, 96)
r = f.matmul_reduce(a, b, 1)
print("PERF_MS", r["average_ms"])
""",
        "pointwise_residual_gelu": """
import numpy as np
import mudnn_frontend_framework_test as f
x = np.linspace(-1.5, 1.5, 8 * 64, dtype=np.float32).reshape(8, 64)
residual = np.linspace(0.2, -0.2, 8 * 64, dtype=np.float32).reshape(8, 64)
r = f.pointwise_residual_gelu(x, residual)
print("PERF_MS", r["average_ms"])
""",
        "pointwise_residual_gelu_bf16": (
            """
import numpy as np
import mudnn_frontend_framework_test as f
"""
            + _BF16_HELPER_CODE
            + """
x = bf16(np.linspace(-1.5, 1.5, 8 * 64, dtype=np.float32).reshape(8, 64))
residual = bf16(np.linspace(0.2, -0.2, 8 * 64, dtype=np.float32).reshape(8, 64))
r = f.pointwise_residual_gelu_bf16(x, residual)
print("PERF_MS", r["average_ms"])
"""
        ),
        "reduction_sum": """
import numpy as np
import mudnn_frontend_framework_test as f
x = np.linspace(-0.75, 0.75, 4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
r = f.reduction_sum(x, 1)
print("PERF_MS", r["average_ms"])
""",
        "reduction_sum_bf16": (
            """
import numpy as np
import mudnn_frontend_framework_test as f
"""
            + _BF16_HELPER_CODE
            + """
x = bf16(np.linspace(-0.75, 0.75, 4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6))
r = f.reduction_sum_bf16(x, 1)
print("PERF_MS", r["average_ms"])
"""
        ),
        "concat_reshape": """
import numpy as np
import mudnn_frontend_framework_test as f
x = np.linspace(-0.4, 0.2, 3 * 4, dtype=np.float32).reshape(3, 4)
y = np.linspace(0.3, -0.3, 5 * 4, dtype=np.float32).reshape(5, 4)
r = f.concat_reshape(x, y)
print("PERF_MS", r["average_ms"])
""",
        "softmax_last_dim": """
import numpy as np
import mudnn_frontend_framework_test as f
x = np.linspace(-1.0, 1.0, 4 * 8, dtype=np.float32).reshape(4, 8)
r = f.softmax_last_dim(x)
print("PERF_MS", r["average_ms"])
""",
        "conv_relu": """
import numpy as np
import mudnn_frontend_framework_test as f
x = np.linspace(-0.5, 0.5, 1 * 3 * 8 * 8, dtype=np.float32).reshape(1, 3, 8, 8)
w = np.linspace(0.2, -0.2, 4 * 3 * 3 * 3, dtype=np.float32).reshape(4, 3, 3, 3)
r = f.conv_relu(x, w)
print("PERF_MS", r["average_ms"])
""",
        "sdpa_causal": """
import math
import numpy as np
import mudnn_frontend_framework_test as f
dims = (1, 2, 8, 16)
q = np.linspace(-0.25, 0.25, math.prod(dims), dtype=np.float32).reshape(dims)
k = np.linspace(0.2, -0.2, math.prod(dims), dtype=np.float32).reshape(dims)
v = np.linspace(-0.1, 0.3, math.prod(dims), dtype=np.float32).reshape(dims)
r = f.sdpa_causal(q, k, v, 1.0 / math.sqrt(dims[-1]))
print("PERF_MS", r["average_ms"])
""",
        "sdpa_causal_flash_f16": """
import math
import numpy as np
import mudnn_frontend_framework_test as f
dims = (1, 2, 8, 64)
q = np.linspace(-0.25, 0.25, np.prod(dims), dtype=np.float32).reshape(dims).astype(np.float16).view(np.uint16)
k = np.linspace(0.2, -0.2, np.prod(dims), dtype=np.float32).reshape(dims).astype(np.float16).view(np.uint16)
v = np.linspace(-0.1, 0.3, np.prod(dims), dtype=np.float32).reshape(dims).astype(np.float16).view(np.uint16)
r = f.sdpa_causal_flash_f16(q, k, v, 1.0 / math.sqrt(dims[-1]))
print("PERF_MS", r["average_ms"])
""",
        "sdpa_causal_flash_bf16": (
            """
import math
import numpy as np
import mudnn_frontend_framework_test as f
"""
            + _BF16_HELPER_CODE
            + """
dims = (1, 2, 8, 64)
q = bf16(np.linspace(-0.25, 0.25, np.prod(dims), dtype=np.float32).reshape(dims))
k = bf16(np.linspace(0.2, -0.2, np.prod(dims), dtype=np.float32).reshape(dims))
v = bf16(np.linspace(-0.1, 0.3, np.prod(dims), dtype=np.float32).reshape(dims))
r = f.sdpa_causal_flash_bf16(q, k, v, 1.0 / math.sqrt(dims[-1]))
print("PERF_MS", r["average_ms"])
"""
        ),
    }
    try:
        return cases[case_name]
    except KeyError as exc:
        raise ValueError(f"unknown framework graph case: {case_name}") from exc


def extract_kernel_names(log_text: str) -> List[str]:
    names = []
    for match in _KERNEL_RE.finditer(log_text):
        name = match.group(1).strip()
        if name not in names:
            names.append(name)
    return names


def demangle_kernel_names(kernel_names: List[str]) -> List[str]:
    if not kernel_names:
        return []
    proc = subprocess.run(
        ["c++filt"],
        input="\n".join(kernel_names) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"c++filt failed: {proc.stderr}")
    return [line for line in proc.stdout.splitlines() if line]


def run_case(case_name: str, timeout: int = 300) -> Dict[str, object]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", case_code(case_name)],
            env=framework_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.output or b""
        log_text = output.decode("utf-8", errors="ignore")
        kernel_names = extract_kernel_names(log_text)
        return {
            "case": case_name,
            "returncode": "timeout",
            "average_ms": None,
            "kernel_count": len(kernel_names),
            "kernel_names": kernel_names,
            "kernel_labels": demangle_kernel_names(kernel_names),
            "log_tail": log_text[-4000:],
        }
    log_text = proc.stdout.decode("utf-8", errors="ignore")
    perf_match = _PERF_RE.search(log_text)
    kernel_names = extract_kernel_names(log_text)
    kernel_labels = demangle_kernel_names(kernel_names)
    return {
        "case": case_name,
        "returncode": proc.returncode,
        "average_ms": float(perf_match.group(1)) if perf_match else None,
        "kernel_count": len(kernel_names),
        "kernel_names": kernel_names,
        "kernel_labels": kernel_labels,
        "log_tail": log_text[-4000:],
    }


def assert_case_matches_golden(result: Dict[str, object]) -> None:
    case_name = str(result["case"])
    spec = CASE_SPECS[case_name]
    arch = kernel_golden_arch()
    if result["returncode"] != 0:
        raise AssertionError(
            f"{case_name} subprocess failed with return code "
            f"{result['returncode']}:\n{result['log_tail']}"
        )
    if result["average_ms"] is None or result["average_ms"] <= 0.0:
        raise AssertionError(f"{case_name} did not report a positive PERF_MS")
    expected_count = expected_kernel_count(spec, arch)
    if result["kernel_count"] != expected_count:
        raise AssertionError(
            f"{case_name} kernel count mismatch on {arch}: "
            f"expected {expected_count}, got {result['kernel_count']}\n"
            + "\n".join(str(name) for name in result["kernel_labels"])
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect mudnn-frontend framework graph kernels via MUSA_LOG."
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=sorted(CASE_SPECS),
        help="Case to run. Defaults to all cases.",
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    arch = kernel_golden_arch()
    case_names = args.case or sorted(CASE_SPECS)
    results = [run_case(case_name, timeout=args.timeout) for case_name in case_names]
    for result in results:
        assert_case_matches_golden(result)

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        for result in results:
            print(
                f"{result['case']}: arch={arch} kernels={result['kernel_count']} "
                f"average_ms={result['average_ms']:.6f}"
            )
            for label in result["kernel_labels"]:
                print(f"  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
