# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")

_THIS_DIR = Path(__file__).resolve().parent
_FRONTEND_DIR = _THIS_DIR.parents[2]
_MODULE_DIRS = [
    os.getenv("MUDNN_FRONTEND_FRAMEWORK_TEST_BUILD_DIR"),
    _FRONTEND_DIR / "build" / "tests" / "cpp",
    Path.cwd() / "build" / "tests" / "cpp",
    Path.cwd(),
]
for module_dir in _MODULE_DIRS:
    if module_dir and Path(module_dir).exists():
        sys.path.insert(0, str(module_dir))

import mudnn_frontend_framework_test as framework_test  # noqa: E402


_MATMUL_M, _MATMUL_K, _MATMUL_N = 4096, 8192, 3072


def _as_numpy(tensor):
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32))


def _as_bf16_bits(tensor):
    return np.ascontiguousarray(
        tensor.detach().to(torch.bfloat16).view(torch.uint16).cpu().numpy()
    )


def _as_fp16_bits(tensor):
    return np.ascontiguousarray(
        tensor.detach().to(torch.float16).view(torch.uint16).cpu().numpy()
    )


def _from_bf16_bits(array):
    return torch.from_numpy(array).view(torch.bfloat16).to(torch.float32)


def _from_fp16_bits(array):
    return torch.from_numpy(array).view(torch.float16).to(torch.float32)


def _bf16_bits_from_float32(array):
    values = np.asarray(array, dtype=np.float32).view(np.uint32)
    lsb = (values >> 16) & 1
    rounded = values + np.uint32(0x7FFF) + lsb.astype(np.uint32)
    return (rounded >> 16).astype(np.uint16)


def _float32_from_bf16_bits(array):
    values = np.asarray(array, dtype=np.uint16).astype(np.uint32) << 16
    return values.view(np.float32)


def _assert_perf(result):
    assert result["average_ms"] > 0.0
    assert result["warmup_iters"] == 3
    assert result["perf_iters"] == 20


def _sample_indices(size):
    return np.array(sorted({0, 1, size // 2, size - 2, size - 1}), dtype=np.int64)


def _assert_matmul_bias_sample(
    result, a, b, bias, *, relu=False, rtol=5e-2, atol=5e-2
):
    row_idx = _sample_indices(a.shape[0])
    col_idx = _sample_indices(b.shape[1])
    expected = a[row_idx] @ b[:, col_idx]
    if bias.ndim == 1:
        expected = expected + bias[col_idx]
    else:
        expected = expected + bias[np.ix_(row_idx, col_idx)]
    if relu:
        expected = np.maximum(expected, 0.0)
    actual = result["output"][np.ix_(row_idx, col_idx)]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_matmul_bias_graph():
    # PyTorch, ONNX Runtime, XLA, TensorFlow, and cudnn_frontend use matmul
    # epilogue graphs: graph.matmul(A, B) followed by graph.add(..., bias).
    a = torch.linspace(
        -0.25,
        0.25,
        _MATMUL_M * _MATMUL_K,
        dtype=torch.float32,
    ).reshape(_MATMUL_M, _MATMUL_K)
    b = torch.linspace(
        0.15,
        -0.15,
        _MATMUL_K * _MATMUL_N,
        dtype=torch.float32,
    ).reshape(_MATMUL_K, _MATMUL_N)
    bias = torch.linspace(
        -0.1,
        0.1,
        _MATMUL_M * _MATMUL_N,
        dtype=torch.float32,
    ).reshape(_MATMUL_M, _MATMUL_N)
    result = framework_test.matmul_bias(_as_numpy(a), _as_numpy(b), _as_numpy(bias))
    _assert_perf(result)

    _assert_matmul_bias_sample(result, _as_numpy(a), _as_numpy(b), _as_numpy(bias))


def test_matmul_bias_vector_graph():
    a = torch.linspace(
        -0.25,
        0.25,
        _MATMUL_M * _MATMUL_K,
        dtype=torch.float32,
    ).reshape(_MATMUL_M, _MATMUL_K)
    b = torch.linspace(
        0.15,
        -0.15,
        _MATMUL_K * _MATMUL_N,
        dtype=torch.float32,
    ).reshape(_MATMUL_K, _MATMUL_N)
    bias = torch.linspace(-0.1, 0.1, _MATMUL_N, dtype=torch.float32)
    result = framework_test.matmul_bias_vector(
        _as_numpy(a), _as_numpy(b), _as_numpy(bias)
    )
    _assert_perf(result)

    _assert_matmul_bias_sample(result, _as_numpy(a), _as_numpy(b), _as_numpy(bias))


def test_matmul_bias_relu_graph():
    # PyTorch/ONNX Runtime/TensorFlow graph lowerings commonly fuse matmul,
    # expanded bias, and ReLU-style pointwise epilogues.
    a = torch.linspace(
        -0.25,
        0.25,
        _MATMUL_M * _MATMUL_K,
        dtype=torch.float32,
    ).reshape(_MATMUL_M, _MATMUL_K)
    b = torch.linspace(
        0.15,
        -0.15,
        _MATMUL_K * _MATMUL_N,
        dtype=torch.float32,
    ).reshape(_MATMUL_K, _MATMUL_N)
    bias = torch.linspace(
        -0.15,
        0.15,
        _MATMUL_M * _MATMUL_N,
        dtype=torch.float32,
    ).reshape(_MATMUL_M, _MATMUL_N)
    result = framework_test.matmul_bias_relu(
        _as_numpy(a), _as_numpy(b), _as_numpy(bias)
    )
    _assert_perf(result)

    _assert_matmul_bias_sample(
        result,
        _as_numpy(a),
        _as_numpy(b),
        _as_numpy(bias),
        relu=True,
    )


def test_matmul_bias_relu_vector_graph():
    a = torch.linspace(
        -0.25,
        0.25,
        _MATMUL_M * _MATMUL_K,
        dtype=torch.float32,
    ).reshape(_MATMUL_M, _MATMUL_K)
    b = torch.linspace(
        0.15,
        -0.15,
        _MATMUL_K * _MATMUL_N,
        dtype=torch.float32,
    ).reshape(_MATMUL_K, _MATMUL_N)
    bias = torch.linspace(-0.15, 0.15, _MATMUL_N, dtype=torch.float32)
    result = framework_test.matmul_bias_relu_vector(
        _as_numpy(a), _as_numpy(b), _as_numpy(bias)
    )
    _assert_perf(result)

    _assert_matmul_bias_sample(
        result,
        _as_numpy(a),
        _as_numpy(b),
        _as_numpy(bias),
        relu=True,
    )


def test_matmul_bias_relu_bf16_graph():
    # PyTorch/ONNX Runtime/TensorFlow matmul epilogue graph with BF16 IO and
    # FLOAT compute, represented through uint16 BF16 bit patterns in pybind.
    a = torch.linspace(
        -0.5,
        0.5,
        _MATMUL_M * _MATMUL_K,
        dtype=torch.float32,
    ).reshape(_MATMUL_M, _MATMUL_K)
    b = torch.linspace(
        0.2,
        -0.2,
        _MATMUL_K * _MATMUL_N,
        dtype=torch.float32,
    ).reshape(_MATMUL_K, _MATMUL_N)
    bias = torch.linspace(
        -0.15,
        0.15,
        _MATMUL_M * _MATMUL_N,
        dtype=torch.float32,
    ).reshape(_MATMUL_M, _MATMUL_N)
    a_np = _as_numpy(a)
    b_np = _as_numpy(b)
    bias_np = _as_numpy(bias)
    a_bits = _bf16_bits_from_float32(a_np)
    b_bits = _bf16_bits_from_float32(b_np)
    bias_bits = _bf16_bits_from_float32(bias_np)

    result = framework_test.matmul_bias_relu_bf16(
        a_bits, b_bits, bias_bits
    )
    _assert_perf(result)
    assert result["output_dtype"] == "bfloat16_bits"

    sampled_result = dict(result)
    sampled_result["output"] = _float32_from_bf16_bits(result["output"])
    _assert_matmul_bias_sample(
        sampled_result,
        _float32_from_bf16_bits(a_bits),
        _float32_from_bf16_bits(b_bits),
        _float32_from_bf16_bits(bias_bits),
        relu=True,
        atol=6e-2,
        rtol=6e-2,
    )


def test_matmul_reduce_graph():
    a = np.linspace(
        0.01,
        0.02,
        _MATMUL_M * _MATMUL_K,
        dtype=np.float32,
    ).reshape(_MATMUL_M, _MATMUL_K)
    b = np.linspace(
        0.02,
        0.03,
        _MATMUL_K * _MATMUL_N,
        dtype=np.float32,
    ).reshape(_MATMUL_K, _MATMUL_N)

    result = framework_test.matmul_reduce(a, b, 1)
    _assert_perf(result)
    assert result["output"].shape == (_MATMUL_M, 1)

    row_idx = _sample_indices(_MATMUL_M)
    expected_sample = (a[row_idx] @ b).sum(axis=1, keepdims=True)
    actual_sample = result["output"][row_idx]
    np.testing.assert_allclose(
        actual_sample, expected_sample, rtol=5e-2, atol=5e-2
    )


def test_pointwise_residual_gelu_graph():
    # TransformerEngine, PyTorch, XLA, TensorFlow, and ONNX Runtime use residual
    # add plus GELU-family pointwise chains in MLP blocks.
    x = torch.linspace(-1.5, 1.5, 8 * 64, dtype=torch.float32).reshape(8, 64)
    residual = torch.linspace(0.2, -0.2, 8 * 64, dtype=torch.float32).reshape(8, 64)
    expected = torch.nn.functional.gelu(x + residual, approximate="tanh")

    result = framework_test.pointwise_residual_gelu(_as_numpy(x), _as_numpy(residual))
    _assert_perf(result)

    torch.testing.assert_close(
        torch.from_numpy(result["output"]), expected, atol=2e-3, rtol=2e-3
    )


def test_pointwise_residual_gelu_bf16_graph():
    # TransformerEngine, PyTorch, XLA, TensorFlow, and ONNX Runtime use BF16
    # residual add plus GELU chains in MLP blocks.
    x = torch.linspace(-1.5, 1.5, 8 * 64, dtype=torch.float32).reshape(8, 64)
    residual = torch.linspace(0.2, -0.2, 8 * 64, dtype=torch.float32).reshape(8, 64)
    x_bf16 = x.to(torch.bfloat16)
    residual_bf16 = residual.to(torch.bfloat16)
    expected = torch.nn.functional.gelu(
        x_bf16.to(torch.float32) + residual_bf16.to(torch.float32),
        approximate="tanh",
    ).to(torch.bfloat16)

    result = framework_test.pointwise_residual_gelu_bf16(
        _as_bf16_bits(x), _as_bf16_bits(residual)
    )
    _assert_perf(result)
    assert result["output_dtype"] == "bfloat16_bits"

    torch.testing.assert_close(
        _from_bf16_bits(result["output"]),
        expected.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )


def test_reduction_sum_graph():
    # PyTorch, ONNX Runtime, TensorFlow, XLA, and cudnn_frontend use reduction
    # graph nodes for normalization/statistics and shape-preserving sum paths.
    x = torch.linspace(-0.75, 0.75, 4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6)
    expected = x.sum(dim=1, keepdim=True)

    result = framework_test.reduction_sum(_as_numpy(x), 1)
    _assert_perf(result)

    torch.testing.assert_close(
        torch.from_numpy(result["output"]), expected, atol=2e-3, rtol=2e-3
    )


def test_reduction_sum_bf16_graph():
    # PyTorch, ONNX Runtime, TensorFlow, XLA, and cudnn_frontend use BF16
    # reduction graph nodes for normalization/statistics paths.
    x = torch.linspace(-0.75, 0.75, 4 * 5 * 6, dtype=torch.float32).reshape(4, 5, 6)
    x_bf16 = x.to(torch.bfloat16)
    expected = x_bf16.to(torch.float32).sum(dim=1, keepdim=True).to(torch.bfloat16)

    result = framework_test.reduction_sum_bf16(_as_bf16_bits(x), 1)
    _assert_perf(result)
    assert result["output_dtype"] == "bfloat16_bits"

    torch.testing.assert_close(
        _from_bf16_bits(result["output"]),
        expected.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )


def test_concat_reshape_graph():
    # ONNX Runtime, XLA, TensorFlow, and cudnn_frontend use concat/reshape graph
    # nodes for layout/view-style tensor assembly.
    x = torch.linspace(-0.4, 0.2, 3 * 4, dtype=torch.float32).reshape(3, 4)
    y = torch.linspace(0.3, -0.3, 5 * 4, dtype=torch.float32).reshape(5, 4)
    expected = torch.cat([x, y], dim=0).reshape(2, 16)

    result = framework_test.concat_reshape(_as_numpy(x), _as_numpy(y))
    _assert_perf(result)

    torch.testing.assert_close(
        torch.from_numpy(result["output"]), expected, atol=1e-5, rtol=1e-5
    )


def test_softmax_last_dim_graph():
    # PyTorch/TransformerEngine/XLA/TensorFlow attention and classification
    # paths lower softmax-like patterns into exp, reduction, and division nodes.
    x = torch.linspace(-1.0, 1.0, 4 * 8, dtype=torch.float32).reshape(4, 8)
    expected = torch.softmax(x, dim=1)

    result = framework_test.softmax_last_dim(_as_numpy(x))
    _assert_perf(result)

    torch.testing.assert_close(
        torch.from_numpy(result["output"]), expected, atol=2e-3, rtol=2e-3
    )


def test_conv_relu_graph():
    # PyTorch Conv_v8 and ONNX Runtime frontend graph paths use conv_fprop
    # followed by a pointwise ReLU activation node.
    x = torch.linspace(-0.5, 0.5, 1 * 3 * 8 * 8, dtype=torch.float32).reshape(
        1, 3, 8, 8
    )
    w = torch.linspace(0.2, -0.2, 4 * 3 * 3 * 3, dtype=torch.float32).reshape(
        4, 3, 3, 3
    )
    expected = torch.nn.functional.relu(torch.nn.functional.conv2d(x, w, padding=1))

    result = framework_test.conv_relu(_as_numpy(x), _as_numpy(w))
    _assert_perf(result)

    torch.testing.assert_close(
        torch.from_numpy(result["output"]), expected, atol=2e-3, rtol=2e-3
    )


def test_sdpa_causal_graph():
    # TransformerEngine, XLA, TensorFlow, PyTorch, and cudnn_frontend use SDPA
    # graphs: graph.sdpa(Q, K, V, causal_mask=True, is_inference=True).
    dims = (1, 2, 8, 16)
    q = torch.linspace(-0.25, 0.25, math.prod(dims), dtype=torch.float32).reshape(dims)
    k = torch.linspace(0.2, -0.2, math.prod(dims), dtype=torch.float32).reshape(dims)
    v = torch.linspace(-0.1, 0.3, math.prod(dims), dtype=torch.float32).reshape(dims)
    scale = 1.0 / math.sqrt(dims[-1])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
    )

    result = framework_test.sdpa_causal(_as_numpy(q), _as_numpy(k), _as_numpy(v), scale)
    _assert_perf(result)

    torch.testing.assert_close(
        torch.from_numpy(result["output"]), expected, atol=3e-3, rtol=3e-3
    )


def test_sdpa_causal_flash_f16_graph():
    dims = (1, 2, 8, 64)
    q = torch.linspace(-0.25, 0.25, math.prod(dims), dtype=torch.float32).reshape(dims)
    k = torch.linspace(0.2, -0.2, math.prod(dims), dtype=torch.float32).reshape(dims)
    v = torch.linspace(-0.1, 0.3, math.prod(dims), dtype=torch.float32).reshape(dims)
    q_half = q.to(torch.float16)
    k_half = k.to(torch.float16)
    v_half = v.to(torch.float16)
    scale = 1.0 / math.sqrt(dims[-1])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_half.to(torch.float32),
        k_half.to(torch.float32),
        v_half.to(torch.float32),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
    ).to(torch.float16)

    result = framework_test.sdpa_causal_flash_f16(
        _as_fp16_bits(q), _as_fp16_bits(k), _as_fp16_bits(v), scale
    )
    _assert_perf(result)
    assert result["output_dtype"] == "float16_bits"

    torch.testing.assert_close(
        _from_fp16_bits(result["output"]),
        expected.to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )


def test_sdpa_causal_flash_bf16_graph():
    dims = (1, 2, 8, 64)
    q = torch.linspace(-0.25, 0.25, math.prod(dims), dtype=torch.float32).reshape(dims)
    k = torch.linspace(0.2, -0.2, math.prod(dims), dtype=torch.float32).reshape(dims)
    v = torch.linspace(-0.1, 0.3, math.prod(dims), dtype=torch.float32).reshape(dims)
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)
    v_bf16 = v.to(torch.bfloat16)
    scale = 1.0 / math.sqrt(dims[-1])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_bf16.to(torch.float32),
        k_bf16.to(torch.float32),
        v_bf16.to(torch.float32),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=scale,
    ).to(torch.bfloat16)

    result = framework_test.sdpa_causal_flash_bf16(
        _as_bf16_bits(q), _as_bf16_bits(k), _as_bf16_bits(v), scale
    )
    _assert_perf(result)
    assert result["output_dtype"] == "bfloat16_bits"

    torch.testing.assert_close(
        _from_bf16_bits(result["output"]),
        expected.to(torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )
