# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

import sys
import os
import re
import json
import math
import subprocess
import ctypes
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common import device_memory

try:
    from mudnn_tests import get_c_api_test_symbol
except ModuleNotFoundError:

    def get_c_api_test_symbol(_):
        return None


sys.dont_write_bytecode = True
_FRONTEND_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT = _FRONTEND_ROOT.parent
_FRONTEND_PYTHON = _FRONTEND_ROOT / "python"


def _installed_mudnn_prefix():
    override = os.environ.get("MUDNN_FRONTEND_INSTALL_PREFIX")
    if override:
        return Path(override)
    backend_root = os.environ.get("MUDNN_BACKEND_ROOT")
    if backend_root:
        return Path(backend_root)
    build_dir = Path(
        os.environ.get("MUDNN_FRONTEND_TEST_BUILD_DIR", _REPO_ROOT / "build" / "mp_31")
    )
    return build_dir / "mudnn"


def _backend_include_flags():
    dirs = []
    override = os.environ.get("MUDNN_BACKEND_INCLUDE")
    if override:
        dirs.append(override)
    else:
        backend_root = os.environ.get("MUDNN_BACKEND_ROOT")
        if backend_root:
            dirs.append(str(Path(backend_root) / "include"))
    dirs.append("/usr/local/musa/include")
    flags = []
    for directory in dirs:
        flags += ["-I", directory]
    return flags


sys.path.insert(0, str(_FRONTEND_PYTHON))


class _FakeDataPtr:
    def __init__(self, pointer):
        self._pointer = pointer

    def data_ptr(self):
        return self._pointer


class _FakeCudaArray:
    def __init__(self, pointer):
        self.__cuda_array_interface__ = {
            "shape": (1,),
            "typestr": "<f4",
            "data": (pointer, False),
            "version": 3,
        }


def _conv2d_nchw(x, w, pad=(0, 0), stride=(1, 1), dilation=(1, 1), groups=1):
    n, c, h, width = x.shape
    k, wc, r, s = w.shape
    assert groups > 0
    assert c == wc * groups
    assert k % groups == 0
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    out_h = (h + 2 * pad_h - dilation_h * (r - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (s - 1) - 1) // stride_w + 1
    out = np.zeros((n, k, out_h, out_w), dtype=np.float32)
    out_channels_per_group = k // groups
    for ni in range(n):
        for ki in range(k):
            group = ki // out_channels_per_group
            for oi in range(out_h):
                for oj in range(out_w):
                    total = 0.0
                    for ci in range(wc):
                        for ri in range(r):
                            for sj in range(s):
                                xi = oi * stride_h + ri * dilation_h - pad_h
                                xj = oj * stride_w + sj * dilation_w - pad_w
                                if 0 <= xi < h and 0 <= xj < width:
                                    x_channel = group * wc + ci
                                    total += (
                                        x[ni, x_channel, xi, xj] * w[ki, ci, ri, sj]
                                    )
                    out[ni, ki, oi, oj] = total
    return out


def _conv3d_ncdhw(x, w, pad=(0, 0, 0), stride=(1, 1, 1), dilation=(1, 1, 1)):
    n, c, depth, height, width = x.shape
    k, wc, r, s, t = w.shape
    assert c == wc
    pad_d, pad_h, pad_w = pad
    stride_d, stride_h, stride_w = stride
    dilation_d, dilation_h, dilation_w = dilation
    out_d = (depth + 2 * pad_d - dilation_d * (r - 1) - 1) // stride_d + 1
    out_h = (height + 2 * pad_h - dilation_h * (s - 1) - 1) // stride_h + 1
    out_w = (width + 2 * pad_w - dilation_w * (t - 1) - 1) // stride_w + 1
    out = np.zeros((n, k, out_d, out_h, out_w), dtype=np.float32)
    for ni in range(n):
        for ki in range(k):
            for od in range(out_d):
                for oh in range(out_h):
                    for ow in range(out_w):
                        total = 0.0
                        for ci in range(c):
                            for ri in range(r):
                                xd = od * stride_d + ri * dilation_d - pad_d
                                if xd < 0 or xd >= depth:
                                    continue
                                for si in range(s):
                                    xh = oh * stride_h + si * dilation_h - pad_h
                                    if xh < 0 or xh >= height:
                                        continue
                                    for ti in range(t):
                                        xw = ow * stride_w + ti * dilation_w - pad_w
                                        if 0 <= xw < width:
                                            total += (
                                                x[ni, ci, xd, xh, xw]
                                                * w[ki, ci, ri, si, ti]
                                            )
                        out[ni, ki, od, oh, ow] = total
    return out


def _conv2d_bwd_data_nchw(
    dy, w, x_shape, pad=(0, 0), stride=(1, 1), dilation=(1, 1), groups=1
):
    _, c, h, width = x_shape
    n, k, out_h, out_w = dy.shape
    wk, wc, r, s = w.shape
    assert groups > 0
    assert k == wk and c == wc * groups and k % groups == 0
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    dx = np.zeros(x_shape, dtype=np.float32)
    out_channels_per_group = k // groups
    for ni in range(n):
        for ci in range(c):
            group = ci // wc
            ci_local = ci - group * wc
            for ki in range(
                group * out_channels_per_group, (group + 1) * out_channels_per_group
            ):
                for oi in range(out_h):
                    for oj in range(out_w):
                        for ri in range(r):
                            for sj in range(s):
                                xi = oi * stride_h + ri * dilation_h - pad_h
                                xj = oj * stride_w + sj * dilation_w - pad_w
                                if 0 <= xi < h and 0 <= xj < width:
                                    dx[ni, ci, xi, xj] += (
                                        dy[ni, ki, oi, oj] * w[ki, ci_local, ri, sj]
                                    )
    return dx


def _conv2d_bwd_filter_nchw(
    x, dy, w_shape, pad=(0, 0), stride=(1, 1), dilation=(1, 1), groups=1
):
    n, c, h, width = x.shape
    k, wc, r, s = w_shape
    assert groups > 0
    assert c == wc * groups and dy.shape[0] == n and dy.shape[1] == k
    assert k % groups == 0
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    dw = np.zeros(w_shape, dtype=np.float32)
    out_channels_per_group = k // groups
    for ni in range(n):
        for ki in range(k):
            group = ki // out_channels_per_group
            for oi in range(dy.shape[2]):
                for oj in range(dy.shape[3]):
                    for ci in range(wc):
                        for ri in range(r):
                            for sj in range(s):
                                xi = oi * stride_h + ri * dilation_h - pad_h
                                xj = oj * stride_w + sj * dilation_w - pad_w
                                if 0 <= xi < h and 0 <= xj < width:
                                    dw[ki, ci, ri, sj] += (
                                        dy[ni, ki, oi, oj]
                                        * x[ni, group * wc + ci, xi, xj]
                                    )
    return dw


def _block_scale_quantize_reference(x, block_size):
    q = np.zeros_like(x)
    scale_shape = x.shape[:-1] + ((x.shape[-1] + block_size - 1) // block_size,)
    scale = np.zeros(scale_shape, dtype=x.dtype)
    for outer_index in np.ndindex(x.shape[:-1]):
        for block in range(scale_shape[-1]):
            start = block * block_size
            end = min(start + block_size, x.shape[-1])
            block_values = x[outer_index + (slice(start, end),)]
            block_scale = np.max(np.abs(block_values))
            if block_scale == 0:
                block_scale = np.array(1.0, dtype=x.dtype)
            scale[outer_index + (block,)] = block_scale
            q[outer_index + (slice(start, end),)] = block_values / block_scale
    y = q * np.repeat(scale, block_size, axis=-1)[..., : x.shape[-1]]
    return q, scale, y


def _sdpa_reference(q, k, v, scale, causal_mask=False, bias=None, mask=None):
    logits = np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(scale)
    if bias is not None:
        logits = logits + bias
    if mask is not None:
        if mask.dtype == np.bool_:
            logits = np.where(mask, logits, -np.inf)
        else:
            logits = logits + mask
    if causal_mask:
        seq_q = logits.shape[-2]
        seq_k = logits.shape[-1]
        causal = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)
        logits = np.where(causal, -np.inf, logits)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.matmul(weights, v)


def _sdpa_backward_reference(q, k, v, d_o, scale, causal_mask=False):
    logits = np.matmul(q, np.swapaxes(k, -1, -2)) * np.float32(scale)
    if causal_mask:
        seq_q = logits.shape[-2]
        seq_k = logits.shape[-1]
        causal = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)
        logits = np.where(causal, -np.inf, logits)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    d_v = np.matmul(np.swapaxes(weights, -1, -2), d_o)
    d_weights = np.matmul(d_o, np.swapaxes(v, -1, -2))
    d_logits = weights * (
        d_weights - np.sum(d_weights * weights, axis=-1, keepdims=True)
    )
    d_q = np.matmul(d_logits, k) * np.float32(scale)
    d_k = np.matmul(np.swapaxes(d_logits, -1, -2), q) * np.float32(scale)
    return d_q, d_k, d_v


def _bn_finalize_training_reference(
    y_sum, y_sq_sum, scale, bias, prev_mean, prev_var, accum_count, momentum, epsilon
):
    count = np.float32(accum_count)
    mean = y_sum / count
    var = y_sq_sum / count - mean * mean
    inv_std = np.float32(1.0) / np.sqrt(var + np.float32(epsilon))
    updated_mean = np.float32(1.0 - momentum) * prev_mean + np.float32(momentum) * mean
    updated_var = (
        np.float32(1.0 - momentum) * prev_var
        + np.float32(momentum) * np.float32(accum_count / (accum_count - 1)) * var
    )
    eq_scale = scale * inv_std
    eq_bias = bias - mean * eq_scale
    return mean, inv_std, updated_mean, updated_var, eq_scale, eq_bias


def _bn_bwd_weights_reference(x, dy, mean, invstd, scale):
    reduce_axes = (0, 2, 3)
    reduced_count = np.float32(np.prod([x.shape[axis] for axis in reduce_axes]))
    normalized = (x - mean) * invstd
    dbn_scale = (dy * normalized).sum(axis=reduce_axes, keepdims=True)
    dbn_bias = dy.sum(axis=reduce_axes, keepdims=True)
    eq_dy_scale = scale * invstd
    param = eq_dy_scale * invstd * dbn_scale
    eq_x_scale = -param / reduced_count
    eq_bias = (mean * param - eq_dy_scale * dbn_bias) / reduced_count
    return dbn_scale, dbn_bias, eq_dy_scale, eq_x_scale, eq_bias


def _avg_pool2d_nchw_include_padding(x, window, pad, stride):
    n, c, h, width = x.shape
    win_h, win_w = window
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    out_h = (h + 2 * pad_h - win_h) // stride_h + 1
    out_w = (width + 2 * pad_w - win_w) // stride_w + 1
    out = np.zeros((n, c, out_h, out_w), dtype=np.float32)
    scale = float(win_h * win_w)
    for ni in range(n):
        for ci in range(c):
            for oi in range(out_h):
                for oj in range(out_w):
                    total = 0.0
                    for wi in range(win_h):
                        for wj in range(win_w):
                            xi = oi * stride_h + wi - pad_h
                            xj = oj * stride_w + wj - pad_w
                            if 0 <= xi < h and 0 <= xj < width:
                                total += x[ni, ci, xi, xj]
                    out[ni, ci, oi, oj] = total / scale
    return out


def _avg_pool2d_nchw_exclude_padding(x, window, pad, stride):
    n, c, h, width = x.shape
    win_h, win_w = window
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    out_h = (h + 2 * pad_h - win_h) // stride_h + 1
    out_w = (width + 2 * pad_w - win_w) // stride_w + 1
    out = np.zeros((n, c, out_h, out_w), dtype=np.float32)
    for ni in range(n):
        for ci in range(c):
            for oi in range(out_h):
                for oj in range(out_w):
                    total = 0.0
                    valid = 0
                    for wi in range(win_h):
                        for wj in range(win_w):
                            xi = oi * stride_h + wi - pad_h
                            xj = oj * stride_w + wj - pad_w
                            if 0 <= xi < h and 0 <= xj < width:
                                total += x[ni, ci, xi, xj]
                                valid += 1
                    out[ni, ci, oi, oj] = total / valid
    return out


def _max_pool2d_nchw(x, window, pad, stride):
    n, c, h, width = x.shape
    win_h, win_w = window
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    out_h = (h + 2 * pad_h - win_h) // stride_h + 1
    out_w = (width + 2 * pad_w - win_w) // stride_w + 1
    out = np.empty((n, c, out_h, out_w), dtype=np.float32)
    for ni in range(n):
        for ci in range(c):
            for oi in range(out_h):
                for oj in range(out_w):
                    values = []
                    for wi in range(win_h):
                        for wj in range(win_w):
                            xi = oi * stride_h + wi - pad_h
                            xj = oj * stride_w + wj - pad_w
                            if 0 <= xi < h and 0 <= xj < width:
                                values.append(x[ni, ci, xi, xj])
                    out[ni, ci, oi, oj] = max(values)
    return out


def _max_pool2d_bwd_nchw(x, dy, window, pad, stride):
    n, c, h, width = x.shape
    win_h, win_w = window
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    dx = np.zeros_like(x)
    for ni in range(n):
        for ci in range(c):
            for oi in range(dy.shape[2]):
                for oj in range(dy.shape[3]):
                    best_value = None
                    best_index = None
                    for wi in range(win_h):
                        for wj in range(win_w):
                            xi = oi * stride_h + wi - pad_h
                            xj = oj * stride_w + wj - pad_w
                            if 0 <= xi < h and 0 <= xj < width:
                                value = x[ni, ci, xi, xj]
                                if best_value is None or value > best_value:
                                    best_value = value
                                    best_index = (ni, ci, xi, xj)
                    dx[best_index] += dy[ni, ci, oi, oj]
    return dx


def _avg_pool2d_bwd_nchw_include_padding(dy, x_shape, window, pad, stride):
    n, c, h, width = x_shape
    win_h, win_w = window
    pad_h, pad_w = pad
    stride_h, stride_w = stride
    dx = np.zeros(x_shape, dtype=np.float32)
    scale = float(win_h * win_w)
    for ni in range(n):
        for ci in range(c):
            for oi in range(dy.shape[2]):
                for oj in range(dy.shape[3]):
                    grad = dy[ni, ci, oi, oj] / scale
                    for wi in range(win_h):
                        for wj in range(win_w):
                            xi = oi * stride_h + wi - pad_h
                            xj = oj * stride_w + wj - pad_w
                            if 0 <= xi < h and 0 <= xj < width:
                                dx[ni, ci, xi, xj] += grad
    return dx


def test_mudnn_frontend_python_builders():
    import cudnn_frontend as fe

    dims = (1, 2, 3)
    strides = (6, 3, 1)
    x = fe.TensorBuilder().setDim(dims).setStrides(strides).setId(1).build()
    b = fe.TensorBuilder().setDim(dims).setStrides(strides).setId(2).build()
    y = fe.TensorBuilder().setDim(dims).setStrides(strides).setId(3).build()
    pw_desc = (
        fe.PointWiseDescBuilder()
        .setMode(fe.PointwiseMode.ADD)
        .setMathPrecision(fe.DataType.FLOAT)
        .build()
    )
    op = (
        fe.OperationBuilder("pointwise")
        .setXDesc(x)
        .setBDesc(b)
        .setYDesc(y)
        .setPwDesc(pw_desc)
        .setAlpha(1.0)
        .setAlpha2(1.0)
        .build()
    )
    select_op = (
        fe.OperationBuilder("pointwise")
        .setXDesc(x)
        .setBDesc(b)
        .setTDesc(b)
        .setYDesc(y)
        .setPwDesc(fe.CUDNN_POINTWISE_BINARY_SELECT)
        .build()
    )
    backward_like_op = (
        fe.OperationBuilder("pointwise")
        .setDYDesc(y)
        .setDXDesc(x)
        .setPwDesc(fe.CUDNN_POINTWISE_RELU_FWD)
        .build()
    )
    graph = (
        fe.OperationGraphBuilder().setHandle(object()).setOperationGraph([op]).build()
    )
    heur = (
        fe.EngineHeuristicsBuilder()
        .setOperationGraph(graph)
        .setHeurMode(fe.HeurMode.INSTANT)
        .build()
    )
    configs = heur.getEngineConfig()
    plan = (
        fe.ExecutionPlanBuilder()
        .setHandle(graph.handle)
        .setEngineConfig(configs[0])
        .build()
    )
    variant = (
        fe.VariantPackBuilder().setUids([1, 2, 3]).setDataPointers([11, 22, 33]).build()
    )

    assert graph.operations == (op,)
    assert op.attributes["pointwise_desc"] == pw_desc
    assert op.attributes["pointwise_mode"] == fe.PointwiseMode.ADD
    assert select_op.inputs == {"x": x, "b": b, "t": b}
    assert select_op.outputs == {"y": y}
    assert select_op.attributes["pointwise_mode"] == fe.PointwiseMode.BINARY_SELECT
    assert backward_like_op.inputs == {"dy": y}
    assert backward_like_op.outputs == {"dx": x}
    assert pw_desc.math_precision == fe.DataType.FLOAT
    assert len(configs) == 1
    assert plan.getWorkspaceSize() == 0
    assert variant.uids == (1, 2, 3)
    assert variant.data_pointers == (11, 22, 33)


def test_mudnn_frontend_python_counted_builder_overloads():
    import cudnn_frontend as fe

    dims = (1, 2, 3)
    strides = (6, 3, 1)
    x = fe.TensorBuilder().setDim(3, dims).setStrides(3, strides).setId(1).build()
    y = fe.TensorBuilder().setDim(3, dims).setStrides(3, strides).setId(2).build()
    op = (
        fe.OperationBuilder("pointwise")
        .setXDesc(x)
        .setYDesc(y)
        .setPwDesc(fe.CUDNN_POINTWISE_IDENTITY)
        .build()
    )
    graph = (
        fe.OperationGraphBuilder()
        .setHandle(object())
        .setOperationGraph(1, (op,))
        .build()
    )
    variant = (
        fe.VariantPackBuilder().setUids(2, (1, 2)).setDataPointers(2, (11, 22)).build()
    )

    assert x.dims == dims
    assert x.strides == strides
    assert graph.operations == (op,)
    assert variant.uids == (1, 2)
    assert variant.data_pointers == (11, 22)

    with pytest.raises(ValueError, match="count does not match"):
        fe.TensorBuilder().setDim(2, dims)
    with pytest.raises(ValueError, match="count does not match"):
        fe.OperationGraphBuilder().setOperationGraph(2, (op,))
    with pytest.raises(ValueError, match="count does not match"):
        fe.VariantPackBuilder().setUids(3, (1, 2))


def test_mudnn_frontend_python_cudnn_root_symbols_match_upstream():
    import ast

    import mudnn

    repo_root = _REPO_ROOT
    upstream_init = (
        repo_root.parent / "cudnn-frontend" / "python" / "cudnn" / "__init__.py"
    )
    if not upstream_init.exists():
        pytest.skip("upstream cudnn Python package is unavailable")

    tree = ast.parse(upstream_init.read_text(encoding="utf-8"))
    upstream_symbols = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "symbols_to_import":
                upstream_symbols = ast.literal_eval(node.value)
                break
        if upstream_symbols is not None:
            break

    assert upstream_symbols is not None
    assert all(hasattr(mudnn, name) for name in upstream_symbols)
    assert set(upstream_symbols) <= set(mudnn.__all__)
    assert set(upstream_symbols) <= set(mudnn.symbols_to_import)


def test_mudnn_frontend_python_private_backend_attribute_alias_constants():
    import cudnn_frontend as fe

    # Exact ABI assertions prevent ctypes lowering from silently sending a
    # valid value for the wrong attribute after backend header changes.
    assert fe.CUDNN_ATTR_CONVOLUTION_GROUP_COUNT == 9000
    assert fe.CUDNN_ATTR_OPERATION_SDPA_FWD_USE_PADDING_MASK == 9001
    assert fe.CUDNN_ATTR_OPERATION_SDPA_FWD_BIAS_DESC == 9002
    assert fe.CUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK == 9003
    assert fe.CUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK == 9004


def test_mudnn_frontend_python_cudnn_alias_constants():
    import logging
    from typing import Any as TypingAny

    import mudnn
    import cudnn_frontend as fe
    from mudnn._experimental_warnings import (
        _reset_experimental_api_warning_registry,
    )
    from mudnn.api_base import (
        APIBase,
        TensorDesc,
        TupleDict,
        ceil_div,
        is_power_of_2,
        warn_experimental_api_once,
    )
    from mudnn.datatypes import (
        _convert_to_cutlass_data_type,
        _cudnn_to_torch_data_type,
        _is_torch_tensor,
        _library_type,
        _torch_to_cudnn_data_type,
    )
    from mudnn.experimental.ops.sdpa import (
        _bprop_cache,
        _fprop_cache,
        scaled_dot_product_attention,
    )
    import mudnn.datatypes as cudnn_datatypes
    from mudnn.graph import graph as compat_graph_context
    from mudnn.ops import causal_conv1d
    from mudnn.sdpa import (
        SdpabwdSm100D256,
        SdpafwdSm100D256,
        sdpa_bwd_wrapper_sm100_d256,
        sdpa_fwd_wrapper_sm100_d256,
    )
    import mudnn.wrapper as cudnn_wrapper
    from mudnn.wrapper import Graph as WrapperGraph
    from mudnn.yarn import compute_yarn_freqs, compute_yarn_inv_freq, yarn_get_mscale
    from mudnn.gemm_amax.api import GemmAmaxSm100, gemm_amax
    from mudnn.grouped_gemm import (
        GroupedGemmGluSm100 as PackageGroupedGemmGluSm100,
        grouped_gemm_glu_wrapper_sm100 as package_grouped_gemm_glu_wrapper_sm100,
    )
    from mudnn.grouped_gemm.grouped_gemm_wgrad.api import GroupedGemmWgradSm100
    from mudnn.discrete_grouped_gemm import (
        DiscreteGroupedGemmSwigluSm100 as PackageDiscreteGroupedGemmSwigluSm100,
        discrete_grouped_gemm_swiglu_wrapper_sm100 as package_discrete_grouped_gemm_swiglu_wrapper_sm100,
    )
    from mudnn.deepseek_sparse_attention.indexer_forward.api import TMA_ALIGN_ELEMS
    from mudnn.rmsnorm_rht_amax.api import (
        DEFAULT_NUM_THREADS_BY_N,
        RPC_CANDIDATES,
        TARGET_MIN_CTAS,
        best_num_threads,
        pick_rows_per_cta,
    )
    from mudnn.rmsnorm_rht_amax import (
        best_num_threads as package_best_num_threads,
        pick_rows_per_cta as package_pick_rows_per_cta,
    )
    from mudnn.sdpa.fwd.api import SdpafwdSm100D256 as OptionalSdpafwdSm100D256

    assert mudnn.backend_version() == fe.backend_version()
    assert mudnn.backend_version_string() == fe.backend_version_string()
    assert mudnn.backend_version() == 92300
    assert mudnn.backend_version_string() == "9.23.0"
    assert mudnn.pygraph is fe.pygraph
    assert mudnn.CUDNN_DATA_FLOAT == fe.CUDNN_DATA_FLOAT
    assert mudnn.CUDNN_DATA_UINT8 == fe.DataType.UINT8
    assert mudnn.data_type.INT8x4 == fe.CUDNN_DATA_INT8x4
    assert mudnn.data_type.UINT8x4 == fe.CUDNN_DATA_UINT8x4
    assert mudnn.data_type.INT8x32 == fe.CUDNN_DATA_INT8x32
    assert fe.data_type("int8x4") == mudnn.data_type.INT8x4
    assert fe.data_type("uint8x4") == mudnn.data_type.UINT8x4
    assert fe.data_type("int8x32") == mudnn.data_type.INT8x32
    assert mudnn.tensor_reordering.INT8x32 == fe.CUDNN_TENSOR_REORDERING_INT8x32
    assert mudnn.tensor_reordering.F16x16 == fe.CUDNN_TENSOR_REORDERING_F16x16
    assert mudnn.tensor_reordering.F8_128x4 == fe.CUDNN_TENSOR_REORDERING_F8_128x4
    assert mudnn.reshape_mode.VIEW_ONLY == fe.CUDNN_RESHAPE_VIEW_ONLY
    assert mudnn.reshape_mode.LOGICAL == fe.CUDNN_RESHAPE_LOGICAL
    assert mudnn.reduction_mode.NOT_SET == fe.CUDNN_REDUCE_TENSOR_NOT_SET
    assert mudnn.norm_forward_phase.NOT_SET == fe.CUDNN_NORM_FWD_NOT_SET
    upstream_root_symbols = {
        "backend_version",
        "backend_version_string",
        "get_last_error_string",
        "destroy_handle",
        "norm_forward_phase",
        "reduction_mode",
        "behavior_note",
        "knob_type",
        "create_handle",
        "create_kernel_cache",
        "create_device_properties",
        "kernel_cache",
        "device_properties",
        "get_stream",
        "numerical_note",
        "set_stream",
        "build_plan_policy",
        "data_type",
        "tensor_reordering",
        "heur_mode",
        "pygraph",
        "tensor",
        "knob",
        "cudnnGraphNotSupportedError",
        "diagonal_alignment",
        "attention_implementation",
        "moe_grouped_matmul_mode",
        "scalar_type",
        "reshape_mode",
        "is_windows",
        "load_cudnn",
        "module_name",
        "DiscreteGroupedGemmDswigluSm100",
        "DiscreteGroupedGemmSwigluSm100",
        "GemmAmaxSm100",
        "GemmDsreluSm100",
        "GemmSreluSm100",
        "GemmSwigluSm100",
        "GroupedGemmDsreluSm100",
        "GroupedGemmDgluSm100",
        "GroupedGemmDswigluSm100",
        "GroupedGemmGluSm100",
        "GroupedGemmGluHadamardSm100",
        "GroupedGemmQuantSm100",
        "GroupedGemmSreluSm100",
        "GroupedGemmSwigluSm100",
        "GroupedGemmWgradSm100",
        "RmsNormRhtAmaxSm100",
        "SdpabwdSm100D256",
        "SdpafwdSm100D256",
        "discrete_grouped_gemm",
        "discrete_grouped_gemm_dswiglu_wrapper_sm100",
        "discrete_grouped_gemm_swiglu_wrapper_sm100",
        "gemm_amax_wrapper_sm100",
        "gemm_dsrelu_wrapper_sm100",
        "gemm_srelu_wrapper_sm100",
        "gemm_swiglu_wrapper_sm100",
        "grouped_gemm",
        "grouped_gemm_dsrelu_wrapper_sm100",
        "grouped_gemm_dglu_wrapper_sm100",
        "grouped_gemm_dswiglu_wrapper_sm100",
        "grouped_gemm_glu_wrapper_sm100",
        "grouped_gemm_glu_hadamard_wrapper_sm100",
        "grouped_gemm_quant_wrapper_sm100",
        "grouped_gemm_srelu_wrapper_sm100",
        "grouped_gemm_swiglu_wrapper_sm100",
        "grouped_gemm_wgrad_wrapper_sm100",
        "rmsnorm_rht_amax_wrapper_sm100",
        "sdpa_bwd_wrapper_sm100_d256",
        "sdpa_fwd_wrapper_sm100_d256",
    }
    assert upstream_root_symbols <= set(mudnn.__all__)
    assert all(hasattr(mudnn, name) for name in upstream_root_symbols)
    assert mudnn.SdpafwdSm100D256 is SdpafwdSm100D256
    assert mudnn.SdpabwdSm100D256 is SdpabwdSm100D256
    assert mudnn.sdpa_fwd_wrapper_sm100_d256 is sdpa_fwd_wrapper_sm100_d256
    assert mudnn.sdpa_bwd_wrapper_sm100_d256 is sdpa_bwd_wrapper_sm100_d256
    assert mudnn.grouped_gemm.GroupedGemmGluSm100 is PackageGroupedGemmGluSm100
    assert (
        mudnn.grouped_gemm.grouped_gemm_glu_wrapper_sm100
        is package_grouped_gemm_glu_wrapper_sm100
    )
    assert (
        mudnn.discrete_grouped_gemm.DiscreteGroupedGemmSwigluSm100
        is PackageDiscreteGroupedGemmSwigluSm100
    )
    assert (
        mudnn.discrete_grouped_gemm.discrete_grouped_gemm_swiglu_wrapper_sm100
        is package_discrete_grouped_gemm_swiglu_wrapper_sm100
    )
    assert mudnn.is_windows() == sys.platform.startswith("win")
    assert mudnn.load_cudnn() is None
    assert cudnn_datatypes.is_windows() == sys.platform.startswith("win")
    assert cudnn_datatypes.module_name == (
        ".Release._compiled_module"
        if sys.platform.startswith("win")
        else "._compiled_module"
    )
    assert mudnn.module_name == cudnn_datatypes.module_name
    assert mudnn.Any is TypingAny
    assert mudnn._library_type is _library_type
    assert mudnn._is_torch_tensor is _is_torch_tensor
    assert "backend_version" in mudnn.symbols_to_import
    assert "reshape_mode" in mudnn.symbols_to_import
    assert cudnn_wrapper.CudnnHandle is int
    assert cudnn_wrapper.logger.name == "mudnn.wrapper"
    assert "CudnnHandle" in cudnn_wrapper.__all__
    assert "logger" in cudnn_wrapper.__all__
    assert mudnn.tensor is mudnn.GraphTensor
    assert mudnn.data_type.NOT_SET == mudnn.CUDNN_DATA_NOT_SET
    assert mudnn.scalar_type.RUNTIME_PARAM == 0
    assert mudnn.scalar_type.COMPILE_TIME_CONST == 1
    # The backend version is a dotted numeric string ("9.23.0").  Parse it here
    # instead of comparing with LooseVersion: that package used to be vendored
    # in this tree only so this line could run, and comparing against "0.0.0"
    # accepted any string at all.  Parsing is dependency-free and strictly
    # stronger -- a malformed version string now fails.
    backend_version = mudnn.backend_version_string()
    assert re.fullmatch(r"\d+(\.\d+)+", backend_version), backend_version
    default_dtype_graph = mudnn.pygraph(io_data_type=mudnn.data_type.HALF)
    default_dtype_tensor = default_dtype_graph.tensor(
        dim=(1, 2), data_type=mudnn.data_type.NOT_SET
    )
    assert default_dtype_tensor.get_dim() == (1, 2)
    assert default_dtype_tensor.get_stride() == (2, 1)
    assert default_dtype_tensor.get_data_type() == mudnn.data_type.HALF
    assert isinstance(mudnn.get_last_error_string(), str)
    kernel_cache = mudnn.create_kernel_cache()
    assert isinstance(kernel_cache, mudnn.kernel_cache)
    assert "kernel_cache" in repr(kernel_cache)
    kernel_cache_json = kernel_cache.serialize()
    restored_kernel_cache = mudnn.create_kernel_cache()
    assert restored_kernel_cache.deserialize(kernel_cache_json) is restored_kernel_cache
    assert restored_kernel_cache.serialize() == kernel_cache_json
    with pytest.raises(RuntimeError, match="already been initialized"):
        restored_kernel_cache.deserialize(kernel_cache_json)
    assert mudnn.create_kernel_cache(kernel_cache_json).serialize() == kernel_cache_json

    device_props = mudnn.create_device_properties(0)
    assert isinstance(device_props, mudnn.device_properties)
    assert "device_properties" in repr(device_props)
    device_props_json = device_props.serialize()
    restored_device_props = mudnn.create_device_properties(device_props_json)
    assert restored_device_props.serialize() == device_props_json

    metadata_graph = mudnn.pygraph(
        io_data_type=mudnn.data_type.FLOAT,
        kernel_cache=kernel_cache,
        device_property=device_props,
        sm_count=120,
        sm_version=100,
        is_dynamic_shape_enabled=True,
        is_override_shape_enabled=True,
    )
    metadata_payload = json.loads(metadata_graph.serialize().decode("utf-8"))
    assert metadata_payload["kernel_cache"] == kernel_cache_json
    assert metadata_payload["device_property"] == device_props_json
    assert metadata_payload["sm_count"] == 120
    assert metadata_payload["sm_version"] == 100
    assert metadata_payload["is_dynamic_shape_enabled"] is True
    assert metadata_payload["is_override_shape_enabled"] is True
    metadata_restored = mudnn.pygraph(
        kernel_cache=restored_kernel_cache,
        device_properties=restored_device_props,
    ).deserialize(metadata_graph.serialize())
    assert metadata_restored.kernel_cache is restored_kernel_cache
    assert metadata_restored.device_property is restored_device_props
    assert metadata_restored._serialized_kernel_cache == kernel_cache_json
    assert metadata_restored._serialized_device_property == device_props_json
    assert metadata_restored.sm_count == 120
    assert metadata_restored.sm_version == 100
    assert metadata_restored.is_dynamic_shape_enabled is True
    assert metadata_restored.is_override_shape_enabled is True
    with pytest.raises(ValueError, match="device_property"):
        mudnn.pygraph(
            device_property=device_props,
            device_properties=restored_device_props,
        )
    knob = mudnn.knob(mudnn.knob_type.SPLIT_K, max_value=8, min_value=1, stride=2)
    assert knob.type == mudnn.knob_type.SPLIT_K
    assert knob.max_value == 8
    assert knob.min_value == 1
    assert knob.stride == 2
    upstream_knob_names = {
        "NOT_SET",
        "SWIZZLE",
        "TILE_SIZE",
        "EDGE",
        "MULTIPLY",
        "SPLIT_K_BUF",
        "TILEK",
        "STAGES",
        "REDUCTION_MODE",
        "SPLIT_K_SLC",
        "IDX_MODE",
        "SPECFILT",
        "KERNEL_CFG",
        "WORKSPACE",
        "TILE_CGA_M",
        "TILE_CGA_N",
        "BLOCK_SIZE",
        "OCCUPANCY",
        "ARRAY_SIZE_PER_THREAD",
        "SPLIT_COLS",
        "TILE_ROWS",
        "TILE_COLS",
        "LOAD_SIZE",
        "CTA_COUNT",
        "STREAM_K",
        "SPLIT_P_SLC",
        "TILE_M",
        "TILE_N",
        "WARP_SPEC_CFG",
        "SWAP_AB",
        "INPUT_TMA_ENABLE",
        "OUTPUT_TMA_ENABLE",
    }
    assert upstream_knob_names <= set(mudnn.knob_type.__members__)
    kernel_cfg_knob = mudnn.knob(mudnn.knob_type.KERNEL_CFG, 31, 0, 1)
    assert kernel_cfg_knob.type == mudnn.knob_type.KERNEL_CFG
    assert kernel_cfg_knob.max_value == 31
    assert kernel_cfg_knob.min_value == 0
    assert kernel_cfg_knob.stride == 1
    assert mudnn.behavior_note.RUNTIME_COMPILATION.value >= 0
    assert mudnn.numerical_note.TENSOR_CORE.value >= 0
    assert (
        mudnn.build_plan_policy.HEURISTICS_CHOICE
        == fe.build_plan_policy.HEURISTICS_CHOICE
    )
    assert mudnn.tensor_reordering.NONE == fe.tensor_reordering.NONE
    assert issubclass(mudnn.cudnnGraphNotSupportedError, fe.BackendError)
    assert WrapperGraph is cudnn_wrapper.Graph
    assert WrapperGraph is not mudnn.Graph
    assert TMA_ALIGN_ELEMS == 4
    assert DEFAULT_NUM_THREADS_BY_N[4096] == 256
    assert RPC_CANDIDATES == (2, 4, 8)
    assert TARGET_MIN_CTAS == 148
    assert best_num_threads(4096) == 512
    assert best_num_threads(96) is None
    assert pick_rows_per_cta(1184) == 8
    assert pick_rows_per_cta(16) == 2
    assert package_best_num_threads is best_num_threads
    assert package_pick_rows_per_cta is pick_rows_per_cta
    assert ceil_div(5, 2) == 3
    assert is_power_of_2(8)
    assert not is_power_of_2(6)
    api_base = APIBase()
    stream_sentinel = object()
    assert api_base._get_default_stream(stream_sentinel) is stream_sentinel
    with pytest.raises(NotImplementedError, match="cute tensor metadata"):
        api_base._make_fake_cute_compact_tensor(
            dtype=mudnn.data_type.FLOAT,
            shape=(1, 2),
            stride_order=(1, 0),
        )
    tuple_dict = TupleDict(first=1, second=2)
    assert tuple(tuple_dict) == (1, 2)
    assert tuple_dict[0] == 1
    with pytest.raises(IndexError, match="out of range"):
        tuple_dict[2]
    desc = TensorDesc(
        dtype=mudnn.data_type.FLOAT,
        shape=(2, 3),
        stride=(3, 1),
        stride_order=(1, 0),
        device="cpu",
    )
    assert desc.ndim == 2
    assert desc.size(1) == 3
    assert tuple(desc.size()) == (2, 3)
    assert desc.is_contiguous()
    assert desc.transpose(0, 1).shape == (3, 2)
    assert desc.transpose(0, 1).stride == (1, 3)
    assert desc.permute(1, 0).shape == (3, 2)
    assert desc.unsqueeze(0).shape == (1, 2, 3)
    assert desc.unsqueeze(0).stride == (6, 3, 1)
    assert desc.unsqueeze(0).squeeze(0) == desc
    assert desc.view(3, 2).shape == (3, 2)
    assert desc.view(-1).shape == (6,)
    assert desc.as_strided((3, 2), (1, 3)).stride == (1, 3)
    transposed_desc = desc.transpose(0, 1)
    assert not transposed_desc.is_contiguous()
    assert transposed_desc.contiguous().stride == (2, 1)
    sliced_desc = desc.as_strided((2, 3, 2), (12, 4, 2))
    assert sliced_desc.view(12).stride == (2,)
    torch = pytest.importorskip("torch")
    base = APIBase()
    tensor = torch.empty_strided((2, 3), (3, 1), dtype=torch.float16)
    tensor_desc = base._make_tensor_desc(tensor, name="tensor")
    assert tensor_desc.shape == (2, 3)
    assert tensor_desc.stride == (3, 1)
    assert tensor_desc.dtype == torch.float16
    assert tensor_desc.name == "tensor"
    assert base._pad_tensor_to_ndim(tensor_desc, 4, "tensor").shape == (2, 3, 1, 1)
    assert (
        base._unpad_tensor_to_ndim(
            base._pad_tensor_to_ndim(tensor_desc, 4, "tensor"), 2, "tensor"
        )
        == tensor_desc
    )
    assert base._is_f16(tensor)
    assert base._check_tensor_shape(tensor_desc, (2, 3), name="tensor") == (2, 3)
    assert base._check_tensor_stride(tensor_desc, stride=(3, 1), name="tensor") == (
        (3, 1),
        (1, 0),
    )
    assert base._check_dtype(tensor, torch.float16, name="tensor") == torch.float16
    base._value_error_if(False, "unused")
    with pytest.raises(ValueError, match="shape mismatch"):
        base._check_tensor_shape(tensor_desc, (3, 2), name="tensor")
    with pytest.raises(ValueError, match="stride mismatch"):
        base._check_tensor_stride(tensor_desc, stride=(1, 3), name="tensor")
    with pytest.raises(ValueError, match="dtype mismatch"):
        base._check_dtype(tensor, torch.float32, name="tensor")
    with pytest.raises(NotImplementedError, match="cute tensor metadata"):
        base._make_fake_cute_tensor_like(tensor)
    uint8_tensor = torch.empty_strided((2, 4), (4, 1), dtype=torch.uint8)
    fp4_desc = base._make_tensor_desc(
        uint8_tensor, name="fp4", interpret_uint8_as_fp4x2=True
    )
    assert fp4_desc.shape == (2, 8)
    assert fp4_desc.stride == (8, 1)
    assert base._is_fp4x2(fp4_desc)
    with pytest.raises(RuntimeError, match="unique"):
        desc.permute(0, 0)
    with pytest.raises(RuntimeError, match="input size"):
        desc.view(4, 2)
    with pytest.raises(RuntimeError, match="size and stride"):
        transposed_desc.view(6)
    with pytest.raises(RuntimeError, match="storage_offset"):
        desc.as_strided((2, 3), (3, 1), storage_offset=1)
    with pytest.raises(NotImplementedError, match="APIBase"):
        APIBase().check_support()

    cached = mudnn.graph_cache(lambda value: value)(lambda value: value + 1)
    assert cached(1) == 2
    assert cached(1) == 2

    with compat_graph_context(
        None,
        io_data_type=mudnn.data_type.FLOAT,
        intermediate_data_type=mudnn.data_type.FLOAT,
        compute_data_type=mudnn.data_type.FLOAT,
    ) as (compat_graph, tensors):
        assert isinstance(compat_graph, mudnn.Graph)
        assert tensors == []
    inv_freq = compute_yarn_inv_freq(8, scaling_factor=1.0, device="cpu")
    freqs, mscale = compute_yarn_freqs(4, 8, scaling_factor=1.0, device="cpu")
    assert inv_freq.shape == (4,)
    assert freqs.shape == (4, 1, 1, 8)
    assert mscale == 1.0
    default_freqs, _ = compute_yarn_freqs(2, 8, scaling_factor=1.0)
    assert default_freqs.shape == (2, 1, 1, 8)
    explicit_cuda_inv_freq = compute_yarn_inv_freq(8, scaling_factor=1.0, device="cuda")
    assert explicit_cuda_inv_freq.shape == (4,)
    assert yarn_get_mscale(40.0) > 1.0
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        assert _torch_to_cudnn_data_type(torch.float32) == mudnn.data_type.FLOAT
        assert _torch_to_cudnn_data_type(torch.uint8) == mudnn.data_type.UINT8
        assert _library_type(torch.float16) == mudnn.data_type.HALF
        assert _cudnn_to_torch_data_type(mudnn.data_type.INT64) == torch.int64
    assert _convert_to_cutlass_data_type(None) is None
    assert _fprop_cache == {}
    assert _bprop_cache == {}
    with pytest.raises(TypeError, match="query"):
        scaled_dot_product_attention(None, None, None)
    assert OptionalSdpafwdSm100D256 is SdpafwdSm100D256
    with pytest.raises(TypeError):
        SdpafwdSm100D256()
    with pytest.raises(TypeError):
        SdpabwdSm100D256()
    with pytest.raises(TypeError):
        sdpa_fwd_wrapper_sm100_d256()
    with pytest.raises(TypeError):
        sdpa_bwd_wrapper_sm100_d256()
    if torch is not None:
        x = torch.empty(1, 2, 3, dtype=torch.float32)
        weight = torch.empty(2, 1, dtype=torch.float32)
        assert causal_conv1d(x, weight).shape == x.shape

        q = torch.randn(3, 2, 4, dtype=torch.bfloat16)
        k = torch.randn(5, 1, 4, dtype=torch.bfloat16)
        w = torch.randn(3, 2, dtype=torch.bfloat16)
        cu_q = torch.tensor([0, 2, 3], dtype=torch.int32)
        cu_k = torch.tensor([0, 3, 5], dtype=torch.int32)
        thd_scores = mudnn.DSA.dense_indexer_score_recompute_wrapper(
            q,
            k,
            w,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_k=3,
        )
        expected_scores = torch.zeros(3, 3, dtype=torch.float32)
        expected_denom = torch.empty(3, dtype=torch.float32)
        for q0, q1, k0, k1 in ((0, 2, 0, 3), (2, 3, 3, 5)):
            segment = mudnn.DSA.dense_indexer_score_recompute_wrapper(
                q[q0:q1].unsqueeze(0),
                k[k0:k1].unsqueeze(0),
                w[q0:q1].unsqueeze(0),
            )
            expected_scores[q0:q1, : k1 - k0] = segment["out"][0]
            expected_denom[q0:q1] = segment["denom"][0]
        torch.testing.assert_close(thd_scores["out"], expected_scores)
        torch.testing.assert_close(thd_scores["denom"], expected_denom)
        score_op = mudnn.DSA.DenseIndexerScoreRecompute(
            q,
            k,
            w,
            torch.empty_like(expected_scores),
            torch.empty_like(expected_denom),
            is_thd=True,
        ).compile()
        class_scores, class_denom = score_op.execute(
            q, k, w, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_k=3
        )
        torch.testing.assert_close(class_scores, expected_scores)
        torch.testing.assert_close(class_denom, expected_denom)

        lse = torch.randn(3, 2, dtype=torch.float32)
        thd_attn = mudnn.DSA.dense_attn_score_recompute_wrapper(
            q,
            k,
            lse,
            0.5,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_k=3,
        )
        expected_attn = torch.zeros(3, 3, dtype=torch.float32)
        expected_attn_denom = torch.empty(3, dtype=torch.float32)
        for q0, q1, k0, k1 in ((0, 2, 0, 3), (2, 3, 3, 5)):
            segment = mudnn.DSA.dense_attn_score_recompute_wrapper(
                q[q0:q1].unsqueeze(0),
                k[k0:k1].unsqueeze(0),
                lse[q0:q1].unsqueeze(0),
                0.5,
            )
            expected_attn[q0:q1, : k1 - k0] = segment["out"][0]
            expected_attn_denom[q0:q1] = segment["denom"][0]
        torch.testing.assert_close(thd_attn["out"], expected_attn)
        torch.testing.assert_close(thd_attn["denom"], expected_attn_denom)
        attn_op = mudnn.DSA.DenseAttnScoreRecompute(
            q,
            k,
            lse,
            torch.empty_like(expected_attn),
            torch.empty_like(expected_attn_denom),
            0.5,
            is_thd=True,
        ).compile()
        class_attn, class_attn_denom = attn_op.execute(
            q, k, lse, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_k=3
        )
        torch.testing.assert_close(class_attn, expected_attn)
        torch.testing.assert_close(class_attn_denom, expected_attn_denom)

        index_q = torch.randn(3, 2, 4, dtype=torch.bfloat16)
        weights = torch.randn(3, 2, dtype=torch.bfloat16)
        index_k = torch.randn(5, 4, dtype=torch.bfloat16)
        attn_score = torch.rand(3, 3, dtype=torch.float32)
        attn_score[2, 2] = 0.0
        attn_l1norm = attn_score.sum(dim=-1)
        index_score = torch.zeros_like(attn_score)
        index_lse = torch.zeros(3, dtype=torch.float32)
        thd_backward = mudnn.DSA.dense_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            loss_coeff=float(index_q.shape[0]),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_k=3,
        )
        expected_dq = torch.empty_like(index_q)
        expected_dw = torch.empty_like(weights)
        expected_dk = torch.zeros_like(index_k)
        for q0, q1, k0, k1 in ((0, 2, 0, 3), (2, 3, 3, 5)):
            segment = mudnn.DSA.dense_indexer_backward_wrapper(
                index_q[q0:q1].unsqueeze(0),
                weights[q0:q1].unsqueeze(0),
                index_k[k0:k1].unsqueeze(0),
                attn_score[q0:q1, : k1 - k0].unsqueeze(0),
                attn_l1norm[q0:q1].unsqueeze(0),
                index_score[q0:q1, : k1 - k0].unsqueeze(0),
                index_lse[q0:q1].unsqueeze(0),
                loss_coeff=float(q1 - q0),
            )
            expected_dq[q0:q1] = segment["d_index_q"][0]
            expected_dw[q0:q1] = segment["d_weights"][0]
            expected_dk[k0:k1] += segment["d_index_k"][0]
        torch.testing.assert_close(thd_backward["d_index_q"], expected_dq)
        torch.testing.assert_close(thd_backward["d_weights"], expected_dw)
        torch.testing.assert_close(thd_backward["d_index_k"], expected_dk)
        d_index_q = torch.empty_like(index_q)
        d_weights = torch.empty_like(weights)
        d_index_k = torch.empty_like(index_k)
        backward_op = mudnn.DSA.DenseIndexerBackward(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            is_thd=True,
            batch=2,
            max_seqlen_q=2,
            max_seqlen_k=3,
        ).compile()
        backward_op.execute(
            index_q,
            weights,
            index_k,
            d_index_q,
            d_weights,
            d_index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            loss_coeff=float(index_q.shape[0]),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
        )
        torch.testing.assert_close(d_index_q, expected_dq)
        torch.testing.assert_close(d_weights, expected_dw)
        torch.testing.assert_close(d_index_k, expected_dk)
    with pytest.raises(NotImplementedError, match="GemmAmaxSm100"):
        GemmAmaxSm100().check_support()
    with pytest.raises(NotImplementedError, match="gemm_amax"):
        gemm_amax()
    with pytest.raises(NotImplementedError, match="GroupedGemmWgradSm100"):
        GroupedGemmWgradSm100().compile()
    for api_name in (
        "DiscreteGroupedGemmDswigluSm100",
        "DiscreteGroupedGemmSwigluSm100",
        "GemmAmaxSm100",
        "GemmDsreluSm100",
        "GemmSreluSm100",
        "GemmSwigluSm100",
        "GroupedGemmDsreluSm100",
        "GroupedGemmDgluSm100",
        "GroupedGemmDswigluSm100",
        "GroupedGemmGluSm100",
        "GroupedGemmGluHadamardSm100",
        "GroupedGemmQuantSm100",
        "GroupedGemmSreluSm100",
        "GroupedGemmSwigluSm100",
        "GroupedGemmWgradSm100",
        "RmsNormRhtAmaxSm100",
    ):
        with pytest.raises(NotImplementedError, match=api_name):
            getattr(mudnn, api_name)().check_support()
    for wrapper_name in (
        "discrete_grouped_gemm_dswiglu_wrapper_sm100",
        "discrete_grouped_gemm_swiglu_wrapper_sm100",
        "gemm_amax_wrapper_sm100",
        "gemm_dsrelu_wrapper_sm100",
        "gemm_srelu_wrapper_sm100",
        "gemm_swiglu_wrapper_sm100",
        "grouped_gemm_dsrelu_wrapper_sm100",
        "grouped_gemm_dglu_wrapper_sm100",
        "grouped_gemm_dswiglu_wrapper_sm100",
        "grouped_gemm_glu_wrapper_sm100",
        "grouped_gemm_glu_hadamard_wrapper_sm100",
        "grouped_gemm_quant_wrapper_sm100",
        "grouped_gemm_srelu_wrapper_sm100",
        "grouped_gemm_swiglu_wrapper_sm100",
        "grouped_gemm_wgrad_wrapper_sm100",
        "rmsnorm_rht_amax_wrapper_sm100",
    ):
        with pytest.raises(NotImplementedError, match=wrapper_name):
            getattr(mudnn, wrapper_name)()
    logger = logging.getLogger("mudnn_frontend_test.experimental")
    _reset_experimental_api_warning_registry()
    warn_experimental_api_once(logger, "ExampleApi")
    warn_experimental_api_once(logger, "ExampleApi")
    _reset_experimental_api_warning_registry()

    dims = (1, 2, 3)
    strides = (6, 3, 1)
    x = (
        fe.TensorBuilder()
        .setDim(dims)
        .setStrides(strides)
        .setId(1)
        .setDataType(fe.CUDNN_DATA_FLOAT)
        .build()
    )
    b = (
        fe.TensorBuilder()
        .setDim(dims)
        .setStrides(strides)
        .setId(2)
        .setDataType(fe.CUDNN_DATA_FLOAT)
        .build()
    )
    y = (
        fe.TensorBuilder()
        .setDim(dims)
        .setStrides(strides)
        .setId(3)
        .setDataType(fe.CUDNN_DATA_FLOAT)
        .build()
    )

    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
        .setxDesc(x)
        .setbDesc(b)
        .setyDesc(y)
        .setpwDesc(fe.CUDNN_POINTWISE_ADD)
        .setAlpha(1.0)
        .setAlpha2(1.0)
        .build()
    )

    assert fe.CUDNN_STATUS_SUCCESS == 0
    assert isinstance(fe.backend_version(), int)
    assert fe.backend_version() == 92300
    assert isinstance(fe.backend_version_string(), str)
    assert fe.backend_version_string() == "9.23.0"
    assert callable(fe.causal_conv1d_forward)
    assert callable(fe.causal_conv1d_backward)
    assert fe.CUDNN_HEUR_MODE_INSTANT == fe.HeurMode.INSTANT
    assert fe.CUDNN_POINTWISE_SQRT == fe.PointwiseMode.SQRT
    assert fe.CUDNN_POINTWISE_ADD_SQUARE == fe.PointwiseMode.ADD_SQUARE
    assert fe.CUDNN_POINTWISE_MOD == fe.PointwiseMode.MOD
    assert fe.CUDNN_POINTWISE_POW == fe.PointwiseMode.POW
    assert fe.CUDNN_POINTWISE_ABS == fe.PointwiseMode.ABS
    assert fe.CUDNN_POINTWISE_CEIL == fe.PointwiseMode.CEIL
    assert fe.CUDNN_POINTWISE_EXP == fe.PointwiseMode.EXP
    assert fe.CUDNN_POINTWISE_FLOOR == fe.PointwiseMode.FLOOR
    assert fe.CUDNN_POINTWISE_LOG == fe.PointwiseMode.LOG
    assert fe.CUDNN_POINTWISE_TAN == fe.PointwiseMode.TAN
    assert fe.CUDNN_POINTWISE_ERF == fe.PointwiseMode.ERF
    assert fe.CUDNN_POINTWISE_RECIPROCAL == fe.PointwiseMode.RECIPROCAL
    assert fe.CUDNN_POINTWISE_ELU_FWD == fe.PointwiseMode.ELU_FWD
    assert fe.CUDNN_POINTWISE_SOFTPLUS_FWD == fe.PointwiseMode.SOFTPLUS_FWD
    assert fe.CUDNN_POINTWISE_GELU_APPROX_TANH_FWD == (
        fe.PointwiseMode.GELU_APPROX_TANH_FWD
    )
    assert fe.CUDNN_POINTWISE_SWISH_FWD == fe.PointwiseMode.SWISH_FWD
    assert fe.CUDNN_POINTWISE_RELU_BWD == fe.PointwiseMode.RELU_BWD
    assert fe.CUDNN_POINTWISE_TANH_BWD == fe.PointwiseMode.TANH_BWD
    assert fe.CUDNN_POINTWISE_SIGMOID_BWD == fe.PointwiseMode.SIGMOID_BWD
    assert fe.CUDNN_POINTWISE_ELU_BWD == fe.PointwiseMode.ELU_BWD
    assert fe.CUDNN_POINTWISE_GELU_BWD == fe.PointwiseMode.GELU_BWD
    assert fe.CUDNN_POINTWISE_SOFTPLUS_BWD == fe.PointwiseMode.SOFTPLUS_BWD
    assert fe.CUDNN_POINTWISE_SWISH_BWD == fe.PointwiseMode.SWISH_BWD
    assert fe.CUDNN_POINTWISE_GELU_APPROX_TANH_BWD == (
        fe.PointwiseMode.GELU_APPROX_TANH_BWD
    )
    assert fe.CUDNN_POINTWISE_CMP_NEQ == fe.PointwiseMode.CMP_NEQ
    assert fe.CUDNN_POINTWISE_CMP_GT == fe.PointwiseMode.CMP_GT
    assert fe.CUDNN_POINTWISE_GEN_INDEX == fe.PointwiseMode.GEN_INDEX
    assert fe.CUDNN_POINTWISE_BINARY_SELECT == fe.PointwiseMode.BINARY_SELECT
    assert fe.CUDNN_ATTR_POINTWISE_ELU_ALPHA > 0
    assert fe.CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA > 0
    assert fe.CUDNN_ATTR_OPERATION_POINTWISE_XDESC > 0
    assert fe.CUDNN_ATTR_OPERATION_POINTWISE_TDESC > 0
    assert op.op_type == "pointwise"
    assert op.attributes["pointwise_mode"] == fe.PointwiseMode.ADD

    pw_desc = (
        fe.PointWiseDescBuilder()
        .setMode(fe.CUDNN_POINTWISE_SOFTPLUS_FWD)
        .setEluAlpha(0.75)
        .setSoftplusBeta(1.5)
        .build()
    )
    assert pw_desc.elu_alpha == 0.75
    assert pw_desc.softplus_beta == 1.5


def test_mudnn_frontend_python_cudnn_optional_gemm_wrapper_caches(monkeypatch):
    import mudnn
    from mudnn.discrete_grouped_gemm.discrete_grouped_gemm_dswiglu import (
        api as discrete_grouped_gemm_dswiglu_api,
    )
    from mudnn.discrete_grouped_gemm.discrete_grouped_gemm_swiglu import (
        api as discrete_grouped_gemm_swiglu_api,
    )
    from mudnn.gemm_amax import api as gemm_amax_api
    from mudnn.gemm_dsrelu import api as gemm_dsrelu_api
    from mudnn.gemm_srelu import api as gemm_srelu_api
    from mudnn.gemm_swiglu import api as gemm_swiglu_api
    from mudnn.grouped_gemm.grouped_gemm_dsrelu import api as grouped_gemm_dsrelu_api
    from mudnn.grouped_gemm.grouped_gemm_dglu import api as grouped_gemm_dglu_api
    from mudnn.grouped_gemm.grouped_gemm_dswiglu import (
        api as grouped_gemm_dswiglu_api,
    )
    from mudnn.grouped_gemm.grouped_gemm_glu import api as grouped_gemm_glu_api
    from mudnn.grouped_gemm.grouped_gemm_glu_hadamard import (
        api as grouped_gemm_glu_hadamard_api,
    )
    from mudnn.grouped_gemm.grouped_gemm_quant import api as grouped_gemm_quant_api
    from mudnn.grouped_gemm.grouped_gemm_srelu import api as grouped_gemm_srelu_api
    from mudnn.grouped_gemm.grouped_gemm_swiglu import api as grouped_gemm_swiglu_api
    from mudnn.grouped_gemm.grouped_gemm_wgrad import api as grouped_gemm_wgrad_api
    from mudnn.rmsnorm_rht_amax import api as rmsnorm_rht_amax_api

    def patch_optional_api(api_class):
        compile_count = {"value": 0}

        def counted_compile(self):
            compile_count["value"] += 1

        monkeypatch.setattr(api_class, "check_support", lambda self: True)
        monkeypatch.setattr(api_class, "compile", counted_compile)
        monkeypatch.setattr(api_class, "execute", lambda self, **kwargs: None)
        return compile_count

    def tensor(*shape):
        return SimpleNamespace(shape=shape)

    common_gemm_kwargs = {
        "b_tensor": tensor(512, 256, 2),
        "sfa_tensor": tensor(128, 16),
        "sfb_tensor": tensor(128, 16),
        "prob_tensor": tensor(256, 256, 2),
        "alpha": 0.5,
        "acc_dtype": "float32",
        "mma_tiler_mn": (256, 256),
        "cluster_shape_mn": (2, 1),
        "norm_const_tensor": None,
        "sf_vec_size": 16,
        "vector_f32": False,
    }

    gemm_amax_api._cache_of_GemmAmaxSm100Objects.clear()
    gemm_amax_compile_count = patch_optional_api(gemm_amax_api.GemmAmaxSm100)
    for _ in range(2):
        c_tensor, amax_tensor = mudnn.gemm_amax_wrapper_sm100(
            a_tensor=tensor(256, 512, 2),
            b_tensor=tensor(512, 256, 2),
            sfa_tensor=tensor(128, 16),
            sfb_tensor=tensor(128, 16),
            c_tensor=tensor(256, 256, 2),
            amax_tensor=tensor(1),
            c_major="n",
            c_dtype="bfloat16",
            acc_dtype="float32",
            mma_tiler_mn=(256, 256),
            cluster_shape_mn=(2, 1),
            sf_vec_size=16,
        )
    assert c_tensor.shape == (256, 256, 2)
    assert amax_tensor.shape == (1,)
    assert gemm_amax_compile_count["value"] == 1
    assert len(gemm_amax_api._cache_of_GemmAmaxSm100Objects) == 1

    rmsnorm_rht_amax_api._cache_of_RmsNormRhtAmaxSm100Objects.clear()
    rmsnorm_compile_count = patch_optional_api(rmsnorm_rht_amax_api.RmsNormRhtAmaxSm100)
    for _ in range(2):
        outputs = mudnn.rmsnorm_rht_amax_wrapper_sm100(
            x_tensor=tensor(256, 2048),
            w_tensor=tensor(2048),
            o_tensor=tensor(256, 2048),
            amax_tensor=tensor(128),
            eps=1e-5,
            num_threads=128,
            rows_per_cta=2,
        )
    assert outputs["o_tensor"].shape == (256, 2048)
    assert outputs["amax_tensor"].shape == (128,)
    assert rmsnorm_compile_count["value"] == 1
    assert len(rmsnorm_rht_amax_api._cache_of_RmsNormRhtAmaxSm100Objects) == 1

    gemm_srelu_api._cache_of_GemmSreluSm100Objects.clear()
    srelu_compile_count = patch_optional_api(gemm_srelu_api.GemmSreluSm100)
    monkeypatch.delenv("CUDNN_FE_GEMM_DYNAMIC_M", raising=False)
    monkeypatch.delenv("CUDNN_FE_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (256, 384):
        mudnn.gemm_srelu_wrapper_sm100(
            **common_gemm_kwargs,
            a_tensor=tensor(m, 512, 2),
            c_major="n",
            c_dtype="bfloat16",
            d_dtype="bfloat16",
        )
    assert srelu_compile_count["value"] == 2
    assert len(gemm_srelu_api._cache_of_GemmSreluSm100Objects) == 2

    gemm_srelu_api._cache_of_GemmSreluSm100Objects.clear()
    srelu_compile_count = patch_optional_api(gemm_srelu_api.GemmSreluSm100)
    monkeypatch.setenv("CUDNN_FE_GEMM_DYNAMIC_M", "1")
    for m in (256, 384):
        mudnn.gemm_srelu_wrapper_sm100(
            **common_gemm_kwargs,
            a_tensor=tensor(m, 512, 2),
            c_major="n",
            c_dtype="bfloat16",
            d_dtype="bfloat16",
        )
    assert srelu_compile_count["value"] == 1
    assert len(gemm_srelu_api._cache_of_GemmSreluSm100Objects) == 1

    gemm_srelu_api._cache_of_GemmSreluSm100Objects.clear()
    srelu_compile_count = patch_optional_api(gemm_srelu_api.GemmSreluSm100)
    monkeypatch.setenv("CUDNN_FE_GEMM_DYNAMIC_MNKL", "1")
    monkeypatch.delenv("CUDNN_FE_GEMM_DYNAMIC_M", raising=False)
    for shape in ((256, 512, 2, 512, 256, 2), (384, 640, 3, 640, 384, 3)):
        m, k, l, bk, n, bl = shape
        mudnn.gemm_srelu_wrapper_sm100(
            **{
                **common_gemm_kwargs,
                "a_tensor": tensor(m, k, l),
                "b_tensor": tensor(bk, n, bl),
                "c_major": "n",
                "c_dtype": "bfloat16",
                "d_dtype": "bfloat16",
            }
        )
    assert srelu_compile_count["value"] == 1
    assert len(gemm_srelu_api._cache_of_GemmSreluSm100Objects) == 1

    gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()
    dsrelu_compile_count = patch_optional_api(gemm_dsrelu_api.GemmDsreluSm100)
    monkeypatch.delenv("CUDNN_FE_GEMM_DYNAMIC_M", raising=False)
    monkeypatch.delenv("CUDNN_FE_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (256, 384):
        mudnn.gemm_dsrelu_wrapper_sm100(
            **common_gemm_kwargs,
            a_tensor=tensor(m, 512, 2),
            c_tensor=tensor(m, 256, 2),
            d_major="n",
            d_dtype="bfloat16",
        )
    assert dsrelu_compile_count["value"] == 2
    assert len(gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects) == 2

    gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()
    dsrelu_compile_count = patch_optional_api(gemm_dsrelu_api.GemmDsreluSm100)
    monkeypatch.setenv("CUDNN_FE_GEMM_DYNAMIC_M", "1")
    for m in (256, 384):
        mudnn.gemm_dsrelu_wrapper_sm100(
            **common_gemm_kwargs,
            a_tensor=tensor(m, 512, 2),
            c_tensor=tensor(m, 256, 2),
            d_major="n",
            d_dtype="bfloat16",
        )
    assert dsrelu_compile_count["value"] == 1
    assert len(gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects) == 1

    gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()
    dsrelu_compile_count = patch_optional_api(gemm_dsrelu_api.GemmDsreluSm100)
    monkeypatch.setenv("CUDNN_FE_GEMM_DYNAMIC_MNKL", "1")
    monkeypatch.delenv("CUDNN_FE_GEMM_DYNAMIC_M", raising=False)
    for shape in ((256, 512, 2, 512, 256, 2), (384, 640, 3, 640, 384, 3)):
        m, k, l, bk, n, bl = shape
        mudnn.gemm_dsrelu_wrapper_sm100(
            **{
                **common_gemm_kwargs,
                "a_tensor": tensor(m, k, l),
                "b_tensor": tensor(bk, n, bl),
                "c_tensor": tensor(m, n, l),
                "d_major": "n",
                "d_dtype": "bfloat16",
            }
        )
    assert dsrelu_compile_count["value"] == 1
    assert len(gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects) == 1

    gemm_swiglu_api._cache_of_GemmSwigluSm100Objects.clear()
    swiglu_compile_count = patch_optional_api(gemm_swiglu_api.GemmSwigluSm100)
    for _ in range(2):
        mudnn.gemm_swiglu_wrapper_sm100(
            a_tensor=tensor(256, 512, 2),
            b_tensor=tensor(256, 512, 2),
            alpha=1.0,
            c_major="n",
            ab12_dtype="float32",
            c_dtype="bfloat16",
            acc_dtype="float32",
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            sf_vec_size=16,
            vector_f32=False,
            ab12_stages=4,
        )
    assert swiglu_compile_count["value"] == 1
    assert len(gemm_swiglu_api._cache_of_GemmSwigluSm100Objects) == 1

    gemm_swiglu_api._cache_of_GemmSwigluSm100Objects.clear()
    swiglu_compile_count = patch_optional_api(gemm_swiglu_api.GemmSwigluSm100)
    mudnn.gemm_swiglu_wrapper_sm100(
        a_tensor=tensor(256, 512, 2),
        b_tensor=tensor(256, 512, 2),
        alpha=1.0,
        c_major="n",
        ab12_dtype="float32",
        c_dtype="bfloat16",
        acc_dtype="float32",
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sfa_tensor=tensor(1, 2, 4, 32, 4, 4),
        sfb_tensor=tensor(1, 2, 4, 32, 4, 4),
        norm_const_tensor=tensor(1),
        sf_vec_size=16,
        vector_f32=False,
        ab12_stages=4,
    )
    assert swiglu_compile_count["value"] == 1
    assert len(gemm_swiglu_api._cache_of_GemmSwigluSm100Objects) == 1

    common_grouped_kwargs = {
        "b_tensor": tensor(256, 512, 2),
        "sfa_tensor": tensor(1, 2, 4, 32, 4, 4),
        "sfb_tensor": tensor(1, 2, 4, 32, 4, 4),
        "padded_offsets": tensor(2),
        "alpha_tensor": tensor(2),
        "norm_const_tensor": None,
        "prob_tensor": tensor(512, 1, 1),
        "acc_dtype": "float32",
        "c_dtype": "bfloat16",
        "d_dtype": "bfloat16",
        "cd_major": "n",
        "mma_tiler_mn": (256, 256),
        "cluster_shape_mn": (2, 1),
        "sf_vec_size": 32,
        "vector_f32": False,
        "m_aligned": 256,
        "discrete_col_sfd": False,
        "use_dynamic_sched": False,
    }

    grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects.clear()
    grouped_srelu_compile_count = patch_optional_api(
        grouped_gemm_srelu_api.GroupedGemmSreluSm100
    )
    monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (512, 768):
        mudnn.grouped_gemm_srelu_wrapper_sm100(
            **common_grouped_kwargs,
            a_tensor=tensor(m, 512, 1),
        )
    assert grouped_srelu_compile_count["value"] == 1
    assert len(grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects) == 1

    grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects.clear()
    grouped_srelu_compile_count = patch_optional_api(
        grouped_gemm_srelu_api.GroupedGemmSreluSm100
    )
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    for shape in ((512, 512, 256), (768, 640, 384)):
        m, k, n = shape
        mudnn.grouped_gemm_srelu_wrapper_sm100(
            **{
                **common_grouped_kwargs,
                "a_tensor": tensor(m, k, 1),
                "b_tensor": tensor(n, k, 2),
            }
        )
    assert grouped_srelu_compile_count["value"] == 1
    assert len(grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects) == 1

    grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects.clear()
    grouped_srelu_compile_count = patch_optional_api(
        grouped_gemm_srelu_api.GroupedGemmSreluSm100
    )
    mudnn.grouped_gemm_srelu_wrapper_sm100(
        **common_grouped_kwargs,
        a_tensor=tensor(0, 512, 1),
    )
    mudnn.grouped_gemm_srelu_wrapper_sm100(
        **common_grouped_kwargs,
        a_tensor=tensor(512, 512, 1),
    )
    assert grouped_srelu_compile_count["value"] == 1
    assert len(grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects) == 1

    common_grouped_dsrelu_kwargs = {
        **common_grouped_kwargs,
        "c_tensor": tensor(512, 256, 1),
        "dprob_tensor": tensor(512, 1, 1),
        "d_srelu_tensor": tensor(512, 256, 1),
        "use_dsrelu_reuse": False,
    }
    grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects.clear()
    grouped_dsrelu_compile_count = patch_optional_api(
        grouped_gemm_dsrelu_api.GroupedGemmDsreluSm100
    )
    monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (512, 768):
        mudnn.grouped_gemm_dsrelu_wrapper_sm100(
            **{
                **common_grouped_dsrelu_kwargs,
                "a_tensor": tensor(m, 512, 1),
                "c_tensor": tensor(m, 256, 1),
                "d_srelu_tensor": tensor(m, 256, 1),
            }
        )
    assert grouped_dsrelu_compile_count["value"] == 1
    assert len(grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects) == 1

    grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects.clear()
    grouped_dsrelu_compile_count = patch_optional_api(
        grouped_gemm_dsrelu_api.GroupedGemmDsreluSm100
    )
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    for shape in ((512, 512, 256), (768, 640, 384)):
        m, k, n = shape
        mudnn.grouped_gemm_dsrelu_wrapper_sm100(
            **{
                **common_grouped_dsrelu_kwargs,
                "a_tensor": tensor(m, k, 1),
                "b_tensor": tensor(n, k, 2),
                "c_tensor": tensor(m, n, 1),
                "d_srelu_tensor": tensor(m, n, 1),
            }
        )
    assert grouped_dsrelu_compile_count["value"] == 1
    assert len(grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects) == 1

    grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects.clear()
    grouped_dsrelu_compile_count = patch_optional_api(
        grouped_gemm_dsrelu_api.GroupedGemmDsreluSm100
    )
    mudnn.grouped_gemm_dsrelu_wrapper_sm100(
        **{
            **common_grouped_dsrelu_kwargs,
            "a_tensor": tensor(0, 512, 1),
            "c_tensor": tensor(0, 256, 1),
            "d_srelu_tensor": tensor(0, 256, 1),
        }
    )
    mudnn.grouped_gemm_dsrelu_wrapper_sm100(
        **{
            **common_grouped_dsrelu_kwargs,
            "a_tensor": tensor(512, 512, 1),
            "c_tensor": tensor(512, 256, 1),
            "d_srelu_tensor": tensor(512, 256, 1),
        }
    )
    assert grouped_dsrelu_compile_count["value"] == 1
    assert len(grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects) == 1

    common_grouped_swiglu_kwargs = {
        **common_grouped_kwargs,
        "c_tensor": tensor(512, 512, 1),
    }
    grouped_gemm_swiglu_api._cache_of_GroupedGemmSwigluSm100Objects.clear()
    grouped_swiglu_compile_count = patch_optional_api(
        grouped_gemm_swiglu_api.GroupedGemmSwigluSm100
    )
    monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (512, 768):
        mudnn.grouped_gemm_swiglu_wrapper_sm100(
            **{
                **common_grouped_swiglu_kwargs,
                "a_tensor": tensor(m, 512, 1),
                "c_tensor": tensor(m, 512, 1),
            }
        )
    assert grouped_swiglu_compile_count["value"] == 1
    assert len(grouped_gemm_swiglu_api._cache_of_GroupedGemmSwigluSm100Objects) == 1

    grouped_gemm_swiglu_api._cache_of_GroupedGemmSwigluSm100Objects.clear()
    grouped_swiglu_compile_count = patch_optional_api(
        grouped_gemm_swiglu_api.GroupedGemmSwigluSm100
    )
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    for shape in ((512, 512, 256), (768, 640, 384)):
        m, k, n = shape
        mudnn.grouped_gemm_swiglu_wrapper_sm100(
            **{
                **common_grouped_swiglu_kwargs,
                "a_tensor": tensor(m, k, 1),
                "b_tensor": tensor(n, k, 2),
                "c_tensor": tensor(m, n * 2, 1),
            }
        )
    assert grouped_swiglu_compile_count["value"] == 1
    assert len(grouped_gemm_swiglu_api._cache_of_GroupedGemmSwigluSm100Objects) == 1

    grouped_gemm_swiglu_api._cache_of_GroupedGemmSwigluSm100Objects.clear()
    grouped_swiglu_compile_count = patch_optional_api(
        grouped_gemm_swiglu_api.GroupedGemmSwigluSm100
    )
    mudnn.grouped_gemm_swiglu_wrapper_sm100(
        **{
            **common_grouped_swiglu_kwargs,
            "a_tensor": tensor(512, 512, 1),
            "prob_tensor": None,
        }
    )
    mudnn.grouped_gemm_swiglu_wrapper_sm100(
        **common_grouped_swiglu_kwargs,
        a_tensor=tensor(512, 512, 1),
    )
    assert grouped_swiglu_compile_count["value"] == 2
    assert len(grouped_gemm_swiglu_api._cache_of_GroupedGemmSwigluSm100Objects) == 2

    grouped_gemm_swiglu_api._cache_of_GroupedGemmSwigluSm100Objects.clear()
    grouped_swiglu_compile_count = patch_optional_api(
        grouped_gemm_swiglu_api.GroupedGemmSwigluSm100
    )
    mudnn.grouped_gemm_swiglu_wrapper_sm100(
        **common_grouped_swiglu_kwargs,
        a_tensor=tensor(0, 512, 1),
    )
    mudnn.grouped_gemm_swiglu_wrapper_sm100(
        **common_grouped_swiglu_kwargs,
        a_tensor=tensor(512, 512, 1),
    )
    assert grouped_swiglu_compile_count["value"] == 1
    assert len(grouped_gemm_swiglu_api._cache_of_GroupedGemmSwigluSm100Objects) == 1

    common_grouped_dswiglu_kwargs = {
        **common_grouped_swiglu_kwargs,
        "beta_tensor": tensor(2),
    }
    grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects.clear()
    grouped_dswiglu_compile_count = patch_optional_api(
        grouped_gemm_dswiglu_api.GroupedGemmDswigluSm100
    )
    monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (512, 768):
        mudnn.grouped_gemm_dswiglu_wrapper_sm100(
            **{
                **common_grouped_dswiglu_kwargs,
                "a_tensor": tensor(m, 512, 1),
                "c_tensor": tensor(m, 512, 1),
            }
        )
    assert grouped_dswiglu_compile_count["value"] == 1
    assert len(grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects) == 1

    grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects.clear()
    grouped_dswiglu_compile_count = patch_optional_api(
        grouped_gemm_dswiglu_api.GroupedGemmDswigluSm100
    )
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    for shape in ((512, 512, 256), (768, 640, 384)):
        m, k, n = shape
        mudnn.grouped_gemm_dswiglu_wrapper_sm100(
            **{
                **common_grouped_dswiglu_kwargs,
                "a_tensor": tensor(m, k, 1),
                "b_tensor": tensor(n, k, 2),
                "c_tensor": tensor(m, n * 2, 1),
            }
        )
    assert grouped_dswiglu_compile_count["value"] == 1
    assert len(grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects) == 1

    grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects.clear()
    grouped_dswiglu_compile_count = patch_optional_api(
        grouped_gemm_dswiglu_api.GroupedGemmDswigluSm100
    )
    mudnn.grouped_gemm_dswiglu_wrapper_sm100(
        **{
            **common_grouped_dswiglu_kwargs,
            "a_tensor": tensor(0, 512, 1),
            "c_tensor": tensor(0, 512, 1),
        }
    )
    mudnn.grouped_gemm_dswiglu_wrapper_sm100(
        **{
            **common_grouped_dswiglu_kwargs,
            "a_tensor": tensor(512, 512, 1),
            "c_tensor": tensor(512, 512, 1),
        }
    )
    assert grouped_dswiglu_compile_count["value"] == 1
    assert len(grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects) == 1

    common_discrete_swiglu_kwargs = {
        **common_grouped_swiglu_kwargs,
        "b_tensor": None,
        "sfb_tensor": None,
        "b_ptrs": tensor(2),
        "sfb_ptrs": tensor(2),
        "n": 256,
        "b_dtype": "float4_e2m1fn_x2",
        "b_major": "k",
    }
    discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects.clear()
    discrete_swiglu_compile_count = patch_optional_api(
        discrete_grouped_gemm_swiglu_api.DiscreteGroupedGemmSwigluSm100
    )
    monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (512, 768):
        mudnn.discrete_grouped_gemm_swiglu_wrapper_sm100(
            **{
                **common_discrete_swiglu_kwargs,
                "a_tensor": tensor(m, 512, 1),
                "c_tensor": tensor(m, 512, 1),
            }
        )
    assert discrete_swiglu_compile_count["value"] == 1
    assert (
        len(
            discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects
        )
        == 1
    )

    discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects.clear()
    discrete_swiglu_compile_count = patch_optional_api(
        discrete_grouped_gemm_swiglu_api.DiscreteGroupedGemmSwigluSm100
    )
    mudnn.discrete_grouped_gemm_swiglu_wrapper_sm100(
        **{
            **common_discrete_swiglu_kwargs,
            "a_tensor": tensor(512, 512, 1),
            "prob_tensor": None,
        }
    )
    mudnn.discrete_grouped_gemm_swiglu_wrapper_sm100(
        **common_discrete_swiglu_kwargs,
        a_tensor=tensor(512, 512, 1),
    )
    assert discrete_swiglu_compile_count["value"] == 2
    assert (
        len(
            discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects
        )
        == 2
    )

    discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects.clear()
    discrete_swiglu_compile_count = patch_optional_api(
        discrete_grouped_gemm_swiglu_api.DiscreteGroupedGemmSwigluSm100
    )
    mudnn.discrete_grouped_gemm_swiglu_wrapper_sm100(
        **common_discrete_swiglu_kwargs,
        a_tensor=tensor(0, 512, 1),
    )
    mudnn.discrete_grouped_gemm_swiglu_wrapper_sm100(
        **common_discrete_swiglu_kwargs,
        a_tensor=tensor(512, 512, 1),
    )
    assert discrete_swiglu_compile_count["value"] == 1
    assert (
        len(
            discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects
        )
        == 1
    )

    common_discrete_dswiglu_kwargs = {
        **common_grouped_dswiglu_kwargs,
        "b_tensor": None,
        "sfb_tensor": None,
        "b_ptrs": tensor(2),
        "sfb_ptrs": tensor(2),
        "n": 256,
        "b_dtype": "float4_e2m1fn_x2",
        "b_major": "k",
    }
    discrete_grouped_gemm_dswiglu_api._cache_of_DiscreteGroupedGemmDswigluSm100Objects.clear()
    discrete_dswiglu_compile_count = patch_optional_api(
        discrete_grouped_gemm_dswiglu_api.DiscreteGroupedGemmDswigluSm100
    )
    monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (512, 768):
        mudnn.discrete_grouped_gemm_dswiglu_wrapper_sm100(
            **{
                **common_discrete_dswiglu_kwargs,
                "a_tensor": tensor(m, 512, 1),
                "c_tensor": tensor(m, 512, 1),
            }
        )
    assert discrete_dswiglu_compile_count["value"] == 1
    assert (
        len(
            discrete_grouped_gemm_dswiglu_api._cache_of_DiscreteGroupedGemmDswigluSm100Objects
        )
        == 1
    )

    discrete_grouped_gemm_dswiglu_api._cache_of_DiscreteGroupedGemmDswigluSm100Objects.clear()
    discrete_dswiglu_compile_count = patch_optional_api(
        discrete_grouped_gemm_dswiglu_api.DiscreteGroupedGemmDswigluSm100
    )
    mudnn.discrete_grouped_gemm_dswiglu_wrapper_sm100(
        **{
            **common_discrete_dswiglu_kwargs,
            "a_tensor": tensor(0, 512, 1),
            "c_tensor": tensor(0, 512, 1),
        }
    )
    mudnn.discrete_grouped_gemm_dswiglu_wrapper_sm100(
        **{
            **common_discrete_dswiglu_kwargs,
            "a_tensor": tensor(512, 512, 1),
            "c_tensor": tensor(512, 512, 1),
        }
    )
    assert discrete_dswiglu_compile_count["value"] == 1
    assert (
        len(
            discrete_grouped_gemm_dswiglu_api._cache_of_DiscreteGroupedGemmDswigluSm100Objects
        )
        == 1
    )

    common_grouped_glu_kwargs = {
        **common_grouped_kwargs,
        "act_func": "swiglu",
    }
    grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()
    grouped_glu_compile_count = patch_optional_api(
        grouped_gemm_glu_api.GroupedGemmGluSm100
    )
    monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (512, 768):
        mudnn.grouped_gemm_glu_wrapper_sm100(
            **common_grouped_glu_kwargs,
            a_tensor=tensor(m, 512, 1),
        )
    assert grouped_glu_compile_count["value"] == 1
    assert len(grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects) == 1

    grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()
    grouped_glu_compile_count = patch_optional_api(
        grouped_gemm_glu_api.GroupedGemmGluSm100
    )
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    for shape in ((512, 512, 256), (768, 640, 384)):
        m, k, n = shape
        mudnn.grouped_gemm_glu_wrapper_sm100(
            **{
                **common_grouped_glu_kwargs,
                "a_tensor": tensor(m, k, 1),
                "b_tensor": tensor(n, k, 2),
            }
        )
    assert grouped_glu_compile_count["value"] == 1
    assert len(grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects) == 1

    grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()
    grouped_glu_compile_count = patch_optional_api(
        grouped_gemm_glu_api.GroupedGemmGluSm100
    )
    common_discrete_glu_kwargs = {
        **common_grouped_glu_kwargs,
        "b_tensor": None,
        "sfb_tensor": None,
        "b_ptrs": tensor(2),
        "sfb_ptrs": tensor(2),
        "n": 256,
        "b_dtype": "float4_e2m1fn_x2",
        "b_major": "k",
    }
    for m in (512, 768):
        mudnn.grouped_gemm_glu_wrapper_sm100(
            **common_discrete_glu_kwargs,
            a_tensor=tensor(m, 512, 1),
        )
    assert grouped_glu_compile_count["value"] == 1
    assert len(grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects) == 1

    common_grouped_dglu_kwargs = {
        **common_grouped_glu_kwargs,
        "c_tensor": tensor(512, 512, 1),
        "beta_tensor": tensor(2),
        "dprob_tensor": tensor(512, 1, 1),
        "act_func": "dswiglu",
    }
    grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()
    grouped_dglu_compile_count = patch_optional_api(
        grouped_gemm_dglu_api.GroupedGemmDgluSm100
    )
    monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (512, 768):
        mudnn.grouped_gemm_dglu_wrapper_sm100(
            **{
                **common_grouped_dglu_kwargs,
                "a_tensor": tensor(m, 512, 1),
                "c_tensor": tensor(m, 512, 1),
            }
        )
    assert grouped_dglu_compile_count["value"] == 1
    assert len(grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects) == 1

    grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()
    grouped_dglu_compile_count = patch_optional_api(
        grouped_gemm_dglu_api.GroupedGemmDgluSm100
    )
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    for shape in ((512, 512, 256), (768, 640, 384)):
        m, k, n = shape
        mudnn.grouped_gemm_dglu_wrapper_sm100(
            **{
                **common_grouped_dglu_kwargs,
                "a_tensor": tensor(m, k, 1),
                "b_tensor": tensor(n, k, 2),
                "c_tensor": tensor(m, n * 2, 1),
            }
        )
    assert grouped_dglu_compile_count["value"] == 1
    assert len(grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects) == 1

    grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()
    grouped_dglu_compile_count = patch_optional_api(
        grouped_gemm_dglu_api.GroupedGemmDgluSm100
    )
    common_discrete_dglu_kwargs = {
        **common_grouped_dglu_kwargs,
        "b_tensor": None,
        "sfb_tensor": None,
        "b_ptrs": tensor(2),
        "sfb_ptrs": tensor(2),
        "n": 256,
        "b_dtype": "float4_e2m1fn_x2",
        "b_major": "k",
    }
    for m in (512, 768):
        mudnn.grouped_gemm_dglu_wrapper_sm100(
            **{
                **common_discrete_dglu_kwargs,
                "a_tensor": tensor(m, 512, 1),
                "c_tensor": tensor(m, 512, 1),
            }
        )
    assert grouped_dglu_compile_count["value"] == 1
    assert len(grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects) == 1

    common_grouped_quant_kwargs = {
        **common_grouped_kwargs,
        "c_dtype": None,
    }
    grouped_gemm_quant_api._cache_of_GroupedGemmQuantSm100Objects.clear()
    grouped_quant_compile_count = patch_optional_api(
        grouped_gemm_quant_api.GroupedGemmQuantSm100
    )
    monkeypatch.delenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", raising=False)
    for m in (512, 768):
        mudnn.grouped_gemm_quant_wrapper_sm100(
            **common_grouped_quant_kwargs,
            a_tensor=tensor(m, 512, 1),
        )
    assert grouped_quant_compile_count["value"] == 1
    assert len(grouped_gemm_quant_api._cache_of_GroupedGemmQuantSm100Objects) == 1

    grouped_gemm_quant_api._cache_of_GroupedGemmQuantSm100Objects.clear()
    grouped_quant_compile_count = patch_optional_api(
        grouped_gemm_quant_api.GroupedGemmQuantSm100
    )
    monkeypatch.setenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")
    for shape in ((512, 512, 256), (768, 640, 384)):
        m, k, n = shape
        mudnn.grouped_gemm_quant_wrapper_sm100(
            **{
                **common_grouped_quant_kwargs,
                "a_tensor": tensor(m, k, 1),
                "b_tensor": tensor(n, k, 2),
            }
        )
    assert grouped_quant_compile_count["value"] == 1
    assert len(grouped_gemm_quant_api._cache_of_GroupedGemmQuantSm100Objects) == 1

    grouped_gemm_quant_api._cache_of_GroupedGemmQuantSm100Objects.clear()
    grouped_quant_compile_count = patch_optional_api(
        grouped_gemm_quant_api.GroupedGemmQuantSm100
    )
    common_discrete_quant_kwargs = {
        **common_grouped_quant_kwargs,
        "b_tensor": None,
        "sfb_tensor": None,
        "b_ptrs": tensor(2),
        "sfb_ptrs": tensor(2),
        "n": 256,
        "b_dtype": "float4_e2m1fn_x2",
        "b_major": "k",
    }
    for m in (512, 768):
        mudnn.grouped_gemm_quant_wrapper_sm100(
            **common_discrete_quant_kwargs,
            a_tensor=tensor(m, 512, 1),
        )
    assert grouped_quant_compile_count["value"] == 1
    assert len(grouped_gemm_quant_api._cache_of_GroupedGemmQuantSm100Objects) == 1

    common_grouped_glu_hadamard_kwargs = {
        **common_grouped_glu_kwargs,
        "c_tensor": tensor(512, 512, 1),
        "d_tensor": tensor(512, 256, 1),
    }
    grouped_gemm_glu_hadamard_api._cache_of_GroupedGemmGluHadamardSm100Objects.clear()
    grouped_glu_hadamard_compile_count = patch_optional_api(
        grouped_gemm_glu_hadamard_api.GroupedGemmGluHadamardSm100
    )
    for _ in range(2):
        mudnn.grouped_gemm_glu_hadamard_wrapper_sm100(
            **common_grouped_glu_hadamard_kwargs,
            a_tensor=tensor(512, 512, 1),
        )
    assert grouped_glu_hadamard_compile_count["value"] == 1
    assert (
        len(grouped_gemm_glu_hadamard_api._cache_of_GroupedGemmGluHadamardSm100Objects)
        == 1
    )

    grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()
    wgrad_compile_count = patch_optional_api(
        grouped_gemm_wgrad_api.GroupedGemmWgradSm100
    )
    common_wgrad_kwargs = {
        "a_tensor": tensor(512, 20),
        "b_tensor": tensor(20, 256),
        "sfa_tensor": tensor(128, 4),
        "sfb_tensor": tensor(128, 4),
        "offsets_tensor": tensor(2),
        "global_scale_a": tensor(1),
        "global_scale_b": tensor(1),
        "output_mode": "dense",
        "acc_dtype": "float32",
        "wgrad_dtype": "bfloat16",
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
        "sf_vec_size": 16,
    }
    mudnn.grouped_gemm_wgrad_wrapper_sm100(**common_wgrad_kwargs)
    mudnn.grouped_gemm_wgrad_wrapper_sm100(
        **{
            **common_wgrad_kwargs,
            "a_tensor": tensor(512, 160),
            "b_tensor": tensor(160, 256),
        }
    )
    assert wgrad_compile_count["value"] == 1
    assert len(grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects) == 1

    grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()
    wgrad_compile_count = patch_optional_api(
        grouped_gemm_wgrad_api.GroupedGemmWgradSm100
    )
    for input_order in ("tensor2d", "tensor_ragged"):
        mudnn.grouped_gemm_wgrad_wrapper_sm100(
            **common_wgrad_kwargs,
            input_order=input_order,
        )
    assert wgrad_compile_count["value"] == 2
    assert len(grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects) == 2

    gemm_srelu_api._cache_of_GemmSreluSm100Objects.clear()
    gemm_amax_api._cache_of_GemmAmaxSm100Objects.clear()
    gemm_dsrelu_api._cache_of_GemmDsreluSm100Objects.clear()
    gemm_swiglu_api._cache_of_GemmSwigluSm100Objects.clear()
    rmsnorm_rht_amax_api._cache_of_RmsNormRhtAmaxSm100Objects.clear()
    grouped_gemm_srelu_api._cache_of_GroupedGemmSreluSm100Objects.clear()
    grouped_gemm_dsrelu_api._cache_of_GroupedGemmDsreluSm100Objects.clear()
    grouped_gemm_glu_api._cache_of_GroupedGemmGluSm100Objects.clear()
    grouped_gemm_dglu_api._cache_of_GroupedGemmDgluSm100Objects.clear()
    grouped_gemm_quant_api._cache_of_GroupedGemmQuantSm100Objects.clear()
    grouped_gemm_glu_hadamard_api._cache_of_GroupedGemmGluHadamardSm100Objects.clear()
    grouped_gemm_swiglu_api._cache_of_GroupedGemmSwigluSm100Objects.clear()
    grouped_gemm_dswiglu_api._cache_of_GroupedGemmDswigluSm100Objects.clear()
    discrete_grouped_gemm_swiglu_api._cache_of_DiscreteGroupedGemmSwigluSm100Objects.clear()
    discrete_grouped_gemm_dswiglu_api._cache_of_DiscreteGroupedGemmDswigluSm100Objects.clear()
    grouped_gemm_wgrad_api._cache_of_GroupedGemmWgradSm100Objects.clear()


def test_mudnn_frontend_python_upstream_experimental_warning_once(caplog):
    import logging

    from mudnn.api_base import (
        _reset_experimental_api_warning_registry as api_base_reset_warnings,
        warn_experimental_api_once as api_base_warn_once,
    )
    from mudnn._experimental_warnings import (
        _reset_experimental_api_warning_registry,
        warn_experimental_api_once,
    )

    logger = logging.getLogger("mudnn.test.experimental")
    _reset_experimental_api_warning_registry()
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            warn_experimental_api_once(logger, "FirstExperimentalApi")
            warn_experimental_api_once(logger, "FirstExperimentalApi")
            warn_experimental_api_once(logger, "SecondExperimentalApi")

        assert [record.getMessage() for record in caplog.records] == [
            "FirstExperimentalApi is an experimental API",
            "SecondExperimentalApi is an experimental API",
        ]
        api_base_reset_warnings()
        caplog.clear()
        api_base_warn_once(logger, "ApiBaseExperimentalApi")
        api_base_warn_once(logger, "ApiBaseExperimentalApi")
        assert [record.getMessage() for record in caplog.records] == [
            "ApiBaseExperimentalApi is an experimental API"
        ]
    finally:
        _reset_experimental_api_warning_registry()


def test_mudnn_frontend_wrapper_positional_call_compatibility():
    torch = pytest.importorskip("torch")
    import mudnn
    from mudnn.wrapper import Graph as WrapperGraph

    class FakeTensor:
        def __init__(self, uid, dim=(2, 3), stride=(3, 1)):
            self._uid = uid
            self._dim = dim
            self._stride = stride

        def get_uid(self):
            return self._uid

        def get_dim(self):
            return self._dim

        def get_stride(self):
            return self._stride

        def get_data_type(self):
            return mudnn.data_type.FLOAT

    class FakeGraph:
        def __init__(self):
            self.executed = None

        def get_execution_plan_count(self):
            return 1

        def execute(self, variant_pack, **kwargs):
            self.executed = (dict(variant_pack), dict(kwargs))

    graph = WrapperGraph.__new__(WrapperGraph)
    fake_graph = FakeGraph()
    input_tensor = FakeTensor(11)
    output_tensor = FakeTensor(22)
    setattr(graph, "_Graph__graph", fake_graph)
    setattr(graph, "_Graph__input_tuples", (input_tensor,))
    setattr(graph, "_Graph__output_tuples", (output_tensor,))
    setattr(graph, "_Graph__handle", None)
    setattr(graph, "_Graph__workspace", torch.empty(0, dtype=torch.uint8))

    x = torch.ones((2, 3), dtype=torch.float32)
    y = graph(x)

    assert hasattr(WrapperGraph, "_Graph__call_with_positional_args")
    assert hasattr(WrapperGraph, "__del__")
    assert y.shape == x.shape
    assert y.stride() == x.stride()
    assert y.dtype == x.dtype
    assert fake_graph.executed is not None
    variant_pack, kwargs = fake_graph.executed
    assert variant_pack[11] is x
    assert variant_pack[22] is y
    assert kwargs["workspace"].dtype == torch.uint8


def test_mudnn_frontend_wrapper_set_io_tuples_validates_non_virtual_coverage():
    from mudnn.wrapper import Graph as WrapperGraph

    class FakeTensor:
        def __init__(self, uid, name, virtual=False):
            self._uid = uid
            self._name = name
            self._virtual = virtual

        def get_uid(self):
            return self._uid

        def get_name(self):
            return self._name

        def get_is_virtual(self):
            return self._virtual

    class FakeGraph:
        def __init__(self):
            self.io_tuples = None

        def get_execution_plan_count(self):
            return 1

        def set_io_tuples(self, inputs, outputs):
            self.io_tuples = (tuple(inputs), tuple(outputs))

    def make_wrapper(input_tensors, output_tensors):
        graph = WrapperGraph.__new__(WrapperGraph)
        fake_graph = FakeGraph()
        setattr(graph, "_Graph__graph", fake_graph)
        setattr(graph, "_Graph__tensor_map", {})
        setattr(graph, "_Graph__tensor_unknown", [])
        setattr(graph, "_Graph__tensor_in", dict(input_tensors))
        setattr(graph, "_Graph__tensor_out", dict(output_tensors))
        setattr(graph, "_Graph__inputs", [])
        setattr(graph, "_Graph__outputs", [])
        return graph, fake_graph

    x = FakeTensor(1, "x")
    bias = FakeTensor(2, "bias")
    y = FakeTensor(3, "y")
    aux = FakeTensor(4, "aux")
    virtual = FakeTensor(5, "virtual", virtual=True)

    graph, fake_graph = make_wrapper({"add::x": x, "add::bias": bias}, {"add::y": y})
    graph.set_io_tuples(["add::x", "add::bias"], ["add::y"])
    assert fake_graph.io_tuples == ((x, bias), (y,))

    graph, _ = make_wrapper({"op::virtual": virtual}, {"op::y": y})
    with pytest.raises(ValueError, match="virtual tensor"):
        graph.set_io_tuples(["op::virtual"], ["op::y"])

    graph, _ = make_wrapper({"add::x": x, "add::bias": bias}, {"add::y": y})
    with pytest.raises(ValueError, match="Node input add::bias"):
        graph.set_io_tuples(["add::x"], ["add::y"])

    graph, _ = make_wrapper({"add::x": x}, {"add::y": y, "add::aux": aux})
    with pytest.raises(ValueError, match="Node output add::aux"):
        graph.set_io_tuples(["add::x"], ["add::y"])


def _mcore_yarn_inv_freq_reference(
    head_dim, base, scaling_factor, original_max_position, beta_fast, beta_slow
):
    import torch

    inv_extra = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    if scaling_factor <= 1.0:
        return inv_extra
    inv_inter = inv_extra / scaling_factor
    low = math.floor(
        (head_dim * math.log(original_max_position / (beta_fast * 2 * math.pi)))
        / (2 * math.log(base))
    )
    high = math.ceil(
        (head_dim * math.log(original_max_position / (beta_slow * 2 * math.pi)))
        / (2 * math.log(base))
    )
    low = max(low, 0)
    high = min(high, head_dim - 1)
    if low == high:
        high = high + 0.001
    idx = torch.arange(head_dim // 2, dtype=torch.float32)
    ramp = torch.clamp((idx - low) / (high - low), 0.0, 1.0)
    mask = 1.0 - ramp
    return inv_inter * (1.0 - mask) + inv_extra * mask


def test_mudnn_frontend_python_upstream_yarn_reference_cases():
    import torch

    from mudnn.yarn import (
        compute_yarn_freqs,
        compute_yarn_inv_freq,
        yarn_get_mscale,
    )

    cases = [
        (64, 40.0, 4096, 32, 1),
        (128, 4.0, 32768, 32, 1),
        (128, 1.0, 4096, 32, 1),
        (192, 8.0, 8192, 32, 1),
    ]
    for head_dim, factor, original_max_position, beta_fast, beta_slow in cases:
        inv_freq = compute_yarn_inv_freq(
            head_dim,
            base=10000.0,
            scaling_factor=factor,
            original_max_position=original_max_position,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            device="cpu",
        )
        reference = _mcore_yarn_inv_freq_reference(
            head_dim,
            10000.0,
            factor,
            original_max_position,
            beta_fast,
            beta_slow,
        )
        torch.testing.assert_close(inv_freq, reference, rtol=0, atol=0)

    freqs, mscale = compute_yarn_freqs(
        256,
        128,
        scaling_factor=1.0,
        original_max_position=4096,
        device="cpu",
    )
    inv_std = 1.0 / (10000.0 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
    expected = torch.zeros(256, 1, 1, 128, dtype=torch.float32)
    expected[:, 0, 0, :64] = torch.outer(
        torch.arange(256, dtype=torch.float32), inv_std
    )
    torch.testing.assert_close(freqs, expected, rtol=0, atol=0)
    assert mscale == 1.0
    assert abs(yarn_get_mscale(40.0, 1.0) - (0.1 * math.log(40.0) + 1.0)) < 1e-12


def test_mudnn_frontend_python_cudnn_jit_graph_cache_builds_once():
    import mudnn
    from mudnn.graph import graph as compat_graph

    class FakeTensor:
        def __init__(self, shape, stride, dtype):
            self.shape = tuple(shape)
            self._stride = tuple(stride)
            self.dtype = dtype

        def stride(self):
            return self._stride

    calls = {"count": 0}

    def cache_key(handle, x, y):
        return (
            tuple(x.shape),
            tuple(x.stride()),
            tuple(y.shape),
            tuple(y.stride()),
            x.dtype,
            y.dtype,
        )

    @mudnn.jit(heur_modes=[mudnn.heur_mode.A, mudnn.heur_mode.B])
    @mudnn.graph_cache(key_fn=cache_key)
    def make_graph(handle, x, y):
        calls["count"] += 1
        with compat_graph(handle, io_data_type=mudnn.data_type.FLOAT) as (graph, _):
            x_desc = graph.tensor_like(x, name="x")
            y_desc = graph.tensor_like(y, name="y")
            out_desc = graph.add(x_desc, y_desc, name="out")
            out_desc.set_output(True)
            return graph, [x_desc, y_desc, out_desc]

    x = FakeTensor((2, 3), (3, 1), mudnn.data_type.FLOAT)
    y = FakeTensor((2, 3), (3, 1), mudnn.data_type.FLOAT)

    graph1, uids1 = make_graph(None, x, y)
    graph2, uids2 = make_graph(None, x, y)

    assert calls["count"] == 1
    assert graph1 is graph2
    assert uids1 == uids2 == [1, 2, 3]
    assert graph1.execution_plan_modes == (mudnn.heur_mode.A, mudnn.heur_mode.B)
    assert graph1.get_execution_plan_count() >= 1


def test_mudnn_frontend_python_cudnn_alias_constants_match_header():
    import cudnn_frontend as fe

    header = _FRONTEND_ROOT / "include"
    header = header / "mudnn_frontend.h"
    names = sorted(
        set(
            re.findall(
                r"static constexpr auto\s+(CUDNN_[A-Za-z0-9_]+)\s*=", header.read_text()
            )
        )
    )
    missing = [name for name in names if not hasattr(fe, name)]

    assert names
    assert not missing


def test_mudnn_frontend_python_graph_tensor_upstream_kwargs():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.HALF)
    x = graph.tensor(
        name="x",
        uid=42,
        dim=(2, 3),
        stride=(3, 1),
        dtype=fe.data_type.FLOAT,
        device="cuda",
        reordering_type=fe.tensor_reordering.F8_128x4,
        alignment=32,
        vector_count=4,
        vector_dimension=1,
    )
    ragged = graph.tensor(
        name="ragged",
        uid=7,
        dim=(3, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=fe.data_type.INT32,
    )
    x.set_ragged_offset(ragged).set_ragged_offset_multiplier(8)
    made = graph._make_tensor(
        (1,),
        (1,),
        fe.data_type.FLOAT,
        name="made",
        uid=99,
        reordering_type=fe.tensor_reordering.F8_128x4,
    )
    made.set_output(True).set_data_type(fe.data_type.HALF).set_reordering_type(
        fe.tensor_reordering.NONE
    )
    made.set_alignment(64).set_vector_count_and_dimension(2, 0)

    assert x.get_uid() == 42
    assert x.get_data_type() == fe.data_type.FLOAT
    assert x._set_data_type(fe.data_type.FLOAT) is x
    assert x.device == "cuda"
    assert x.reordering_type == fe.tensor_reordering.F8_128x4
    assert x.get_reordering_type() == fe.tensor_reordering.F8_128x4
    assert x.get_alignment() == 32
    assert x.get_vector_count() == 4
    assert x.get_vector_dimension() == 1
    assert x.get_is_pass_by_value() is False
    assert x.get_has_compile_time_constant() is False
    assert x.ragged_offset is ragged
    assert x.ragged_offset_multiplier == 8
    assert made.get_uid() == 99
    assert made.get_data_type() == fe.data_type.HALF
    assert made.reordering_type == fe.tensor_reordering.NONE
    assert made.get_alignment() == 64
    assert made.get_vector_count() == 2
    assert made.get_vector_dimension() == 0
    assert graph.tensor(name="next", dim=(1,)).get_uid() == 100

    restored = fe.pygraph().deserialize(graph.serialize())
    restored_x = restored.query_tensor_attributes_of_uid(42)
    restored_ragged = restored.query_tensor_attributes_of_uid(7)
    restored_made = restored.query_tensor_attributes_of_uid(99)
    assert restored_x.get_data_type() == fe.data_type.FLOAT
    assert restored_x.device == "cuda"
    assert restored_x.reordering_type == int(fe.tensor_reordering.F8_128x4)
    assert restored_x.get_alignment() == 32
    assert restored_x.get_vector_count() == 4
    assert restored_x.get_vector_dimension() == 1
    assert restored_x.ragged_offset is restored_ragged
    assert restored_x.ragged_offset_multiplier == 8
    assert restored_made.reordering_type == int(fe.tensor_reordering.NONE)
    assert restored_made.get_data_type() == fe.data_type.HALF
    assert restored_made.get_alignment() == 64
    assert restored_made.get_vector_count() == 2
    assert restored_made.get_vector_dimension() == 0

    like = graph.tensor_like(
        x,
        name="x_like",
        uid=123,
        compute_data_type=fe.data_type.HALF,
        ragged_offset_multiplier=16,
    )
    assert like.get_uid() == 123
    assert like.get_name() == "x_like"
    assert like.get_data_type() == fe.data_type.HALF
    assert like.get_dim() == x.get_dim()
    assert like.get_stride() == x.get_stride()
    assert like.device == x.device
    assert like.get_reordering_type() == x.get_reordering_type()
    assert like.get_alignment() == x.get_alignment()
    assert like.get_vector_count() == x.get_vector_count()
    assert like.get_vector_dimension() == x.get_vector_dimension()
    assert like.ragged_offset is ragged
    assert like.ragged_offset_multiplier == 16

    with pytest.raises(ValueError, match="alignment"):
        x.set_alignment(0)
    with pytest.raises(ValueError, match="vector count"):
        x.set_vector_count_and_dimension(0, 0)

    with pytest.raises(ValueError, match="data_type and dtype"):
        graph.tensor(
            dim=(1,),
            data_type=fe.data_type.FLOAT,
            dtype=fe.data_type.HALF,
        )
    with pytest.raises(ValueError, match="multiplier"):
        x.set_ragged_offset_multiplier(0)


def test_mudnn_frontend_python_graph_make_tensor_ragged_offset_build():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    ragged = graph.tensor(
        name="ragged",
        uid=17,
        dim=(3,),
        stride=(1,),
        data_type=fe.data_type.INT64,
    )
    x = graph._make_tensor(
        (2, 3),
        (3, 1),
        fe.data_type.FLOAT,
        name="x",
        uid=18,
        ragged_offset=ragged,
        ragged_offset_multiplier=4,
    )
    y = graph.identity(input=x, name="identity")
    y.set_output(True)

    assert x.ragged_offset is ragged
    assert x.ragged_offset_multiplier == 4
    restored = fe.pygraph().deserialize(graph.serialize())
    restored_x = restored.query_tensor_attributes_of_uid(18)
    restored_ragged = restored.query_tensor_attributes_of_uid(17)
    assert restored_x.ragged_offset is restored_ragged
    assert restored_x.ragged_offset_multiplier == 4

    graph.build()
    assert graph._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_ragged_offset_can_follow_tensor():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = graph.tensor(
        name="x",
        uid=18,
        dim=(2, 3),
        stride=(3, 1),
        data_type=fe.data_type.FLOAT,
    )
    ragged = graph.tensor(
        name="ragged",
        uid=17,
        dim=(3,),
        stride=(1,),
        data_type=fe.data_type.INT64,
    )
    x.set_ragged_offset(ragged)
    y = graph.identity(input=x, name="identity")
    y.set_output(True)

    graph.build()
    assert graph._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_ragged_dynamic_override_physical_stride():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(
        name="x",
        uid=18,
        dim=(2, 4, 3, 5),
        stride=(20, 5, 20, 1),
        data_type=fe.data_type.FLOAT,
    )
    ragged = graph.tensor(
        name="ragged",
        uid=17,
        dim=(3, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=fe.data_type.INT64,
    )
    x.set_ragged_offset(ragged)

    graph._validate_dynamic_shape_overrides(
        override_uids=[x.uid, ragged.uid],
        override_shapes=[x.dims, ragged.dims],
        override_strides=[(20, 5, 1), ragged.strides],
    )

    plain = graph.tensor(
        name="plain",
        uid=19,
        dim=x.dims,
        stride=x.strides,
        data_type=fe.data_type.FLOAT,
    )
    with pytest.raises(ValueError, match="rank"):
        graph._validate_dynamic_shape_overrides(
            override_uids=[plain.uid],
            override_shapes=[plain.dims],
            override_strides=[(20, 5, 1)],
        )


def test_mudnn_frontend_python_graph_sdpa_thd_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        intermediate_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    q = graph.tensor(
        name="q",
        uid=1,
        dim=(2, 2, 2, 2),
        stride=(4, 2, 4, 1),
        data_type=fe.data_type.FLOAT,
    )
    q_ragged = graph.tensor(
        name="q_ragged",
        uid=5,
        dim=(3, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=fe.data_type.INT64,
    )
    q.set_ragged_offset(q_ragged)
    k = graph.tensor(
        name="k",
        uid=2,
        dim=(2, 1, 3, 2),
        stride=(6, 2, 2, 1),
        data_type=fe.data_type.FLOAT,
    )
    v = graph.tensor(
        name="v",
        uid=3,
        dim=(2, 1, 3, 2),
        stride=(6, 2, 2, 1),
        data_type=fe.data_type.FLOAT,
    )
    seq_q = graph.tensor(
        name="seq_q",
        uid=7,
        dim=(2, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=fe.data_type.INT32,
    )
    seq_kv = graph.tensor(
        name="seq_kv",
        uid=8,
        dim=(2, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=fe.data_type.INT32,
    )
    o, _ = graph.sdpa(
        name="sdpa_thd",
        q=q,
        k=k,
        v=v,
        attn_scale=1.0,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
        use_padding_mask=True,
        generate_stats=False,
    )
    o.set_output(True).set_uid(4).set_dim(q.dims).set_stride(q.strides)
    o_ragged = graph.tensor(
        name="o_ragged",
        uid=6,
        dim=(3, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=fe.data_type.INT64,
    )
    o.set_ragged_offset(o_ragged)

    graph.build_operation_graph()
    assert type(graph._native).__name__ == "_CpuFallbackNativeGraph"

    q_data = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5]],
            [[0.0, 1.0], [1.0, -0.5]],
            [[1.0, 1.0], [-0.5, 1.0]],
        ],
        dtype=torch.float32,
    )
    k_data = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
            [[[1.0, -1.0], [0.5, 0.5], [9.0, 9.0]]],
        ],
        dtype=torch.float32,
    )
    v_data = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]],
            [[[2.0, -1.0], [4.0, 1.0], [99.0, 99.0]]],
        ],
        dtype=torch.float32,
    )
    seq_q_data = torch.tensor([[[[2]]], [[[1]]]], dtype=torch.int32)
    seq_kv_data = torch.tensor([[[[3]]], [[[2]]]], dtype=torch.int32)
    ragged_data = torch.tensor([[[[0]]], [[[8]]], [[[12]]]], dtype=torch.int64)
    out = torch.empty_like(q_data)

    graph.execute(
        {
            q.uid: q_data,
            q_ragged.uid: ragged_data,
            k.uid: k_data,
            v.uid: v_data,
            seq_q.uid: seq_q_data,
            seq_kv.uid: seq_kv_data,
            o.uid: out,
            o_ragged.uid: ragged_data,
        },
        override_uids=[
            q.uid,
            q_ragged.uid,
            k.uid,
            v.uid,
            seq_q.uid,
            seq_kv.uid,
            o.uid,
            o_ragged.uid,
        ],
        override_shapes=[
            q.dims,
            q_ragged.dims,
            k.dims,
            v.dims,
            seq_q.dims,
            seq_kv.dims,
            o.dims,
            o_ragged.dims,
        ],
        override_strides=[
            q_data.stride(),
            ragged_data.stride(),
            k_data.stride(),
            v_data.stride(),
            seq_q_data.stride(),
            seq_kv_data.stride(),
            out.stride(),
            ragged_data.stride(),
        ],
    )

    expected_rows = []
    for batch, q_slice in enumerate((q_data[:2], q_data[2:])):
        q_batch = q_slice.permute(1, 0, 2)
        k_batch = k_data[batch, :, : int(seq_kv_data[batch].item()), :].repeat(
            q_batch.shape[0], 1, 1
        )
        v_batch = v_data[batch, :, : int(seq_kv_data[batch].item()), :].repeat(
            q_batch.shape[0], 1, 1
        )
        scores = torch.matmul(
            q_batch.to(torch.float32),
            k_batch.to(torch.float32).transpose(-2, -1),
        )
        probs = torch.softmax(scores, dim=-1)
        expected_rows.append(torch.matmul(probs, v_batch).permute(1, 0, 2))
    expected = torch.cat(expected_rows, dim=0)
    torch.testing.assert_close(out, expected)


def test_mudnn_frontend_python_graph_accepts_torch_dtype_values():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    graph = fe.pygraph(
        io_data_type=torch.float32,
        intermediate_data_type=torch.float16,
        compute_data_type=torch.float32,
    )
    x = graph.tensor(name="x", dim=(2, 3), data_type=torch.float16)
    y = graph.tensor(name="y", dim=(2, 3), dtype=torch.bfloat16)

    assert graph.io_data_type == fe.data_type.FLOAT
    assert graph.intermediate_data_type == fe.data_type.HALF
    assert graph.compute_data_type == fe.data_type.FLOAT
    assert x.get_data_type() == fe.data_type.HALF
    assert y.get_data_type() == fe.data_type.BFLOAT16

    y.set_data_type(torch.uint8)
    graph.set_io_data_type(torch.bool)
    graph.set_intermediate_data_type(torch.int32)
    graph.set_compute_data_type(torch.float64)

    assert y.get_data_type() == fe.data_type.UINT8
    assert graph.io_data_type == fe.data_type.BOOLEAN
    assert graph.intermediate_data_type == fe.data_type.INT32
    assert graph.compute_data_type == fe.data_type.DOUBLE

    noncontiguous = torch.empty(2, 3, dtype=torch.float16).t()
    like = graph.tensor_like(noncontiguous, name="noncontiguous_like")
    overridden = graph.tensor_like(
        noncontiguous,
        name="overridden_like",
        data_type=fe.data_type.FLOAT,
    )

    assert like.get_dim() == (3, 2)
    assert like.get_stride() == (1, 3)
    assert like.get_data_type() == fe.data_type.HALF
    assert overridden.get_data_type() == fe.data_type.FLOAT


def test_mudnn_frontend_python_operation_builder_methods_match_header():
    import cudnn_frontend as fe

    include_dir = _FRONTEND_ROOT / "include"
    header_text = "\n".join(
        path.read_text()
        for path in sorted((include_dir / "mudnn_frontend").glob("*.h"))
    )
    cpp_methods = sorted(
        set(
            re.findall(
                r"OperationBuilder&\s+(set[A-Za-z0-9_]+)\s*\(",
                header_text,
            )
        )
    )
    python_methods = sorted(
        name
        for name in dir(fe.OperationBuilder)
        if name.startswith("set") and callable(getattr(fe.OperationBuilder, name))
    )

    assert cpp_methods
    assert not sorted(set(cpp_methods) - set(python_methods))
    assert not sorted(set(python_methods) - set(cpp_methods))


def test_mudnn_frontend_python_builder_methods_match_header():
    import cudnn_frontend as fe

    include_dir = _FRONTEND_ROOT / "include"
    header_texts = {
        path: path.read_text()
        for path in sorted(
            [include_dir / "mudnn_frontend.h"]
            + list((include_dir / "mudnn_frontend").glob("*.h"))
        )
    }
    builder_classes = [
        "TensorBuilder",
        "PointWiseDescBuilder",
        "MatMulDescBuilder",
        "ConvDescBuilder",
        "ResampleDescBuilder",
        "RngDescBuilder",
        "ReductionDescBuilder",
        "OperationGraphBuilder",
        "EngineBuilder",
        "EngineConfigBuilder",
        "EngineHeuristicsBuilder",
        "ExecutionPlanBuilder",
        "VariantPackBuilder",
    ]

    missing = {}
    extra = {}
    for class_name in builder_classes:
        cpp_body = None
        for header_text in header_texts.values():
            cpp_body = re.search(
                rf"class\s+{class_name}\s*\{{(?P<body>.*?)(?:\n\}};|\nclass\s+|\Z)",
                header_text,
                re.DOTALL,
            )
            if cpp_body is not None:
                break
        assert cpp_body is not None
        cpp_methods = set(
            re.findall(
                rf"\b{class_name}&\s+(set[A-Za-z0-9_]+)\s*\(",
                cpp_body.group("body"),
            )
        )
        python_class = getattr(fe, class_name)
        python_methods = {
            name
            for name in dir(python_class)
            if name.startswith("set") and callable(getattr(python_class, name))
        }
        if class_name == "ConvDescBuilder":
            python_methods |= {
                name
                for name in dir(fe.ConvolutionDescBuilder)
                if name.startswith("set")
                and callable(getattr(fe.ConvolutionDescBuilder, name))
            }
        missing[class_name] = sorted(cpp_methods - python_methods)
        extra[class_name] = sorted(python_methods - cpp_methods)

    assert not {key: value for key, value in missing.items() if value}
    assert not {key: value for key, value in extra.items() if value}


def test_mudnn_frontend_python_pointwise_desc_builder_alias():
    import cudnn_frontend as fe

    desc = (
        fe.PointwiseDescBuilder()
        .setMode(fe.CUDNN_POINTWISE_RELU_FWD)
        .setMathPrecision(fe.CUDNN_DATA_FLOAT)
        .setReluLowerClip(0.0)
        .setReluUpperClip(6.0)
        .setReluLowerClipSlope(0.25)
        .setSwishBeta(1.5)
        .setAxis(1)
        .build()
    )

    assert isinstance(desc, fe.PointwiseDesc)
    assert desc.mode == fe.PointwiseMode.RELU_FWD
    assert desc.math_precision == fe.DataType.FLOAT
    assert desc.relu_lower_clip == 0.0
    assert desc.relu_upper_clip == 6.0
    assert desc.relu_lower_clip_slope == 0.25
    assert desc.swish_beta == 1.5
    assert desc.axis == 1


def test_mudnn_frontend_python_matmul_builders():
    import cudnn_frontend as fe

    a = (
        fe.TensorBuilder()
        .setDim((2, 3))
        .setStrides((3, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    b = (
        fe.TensorBuilder()
        .setDim((3, 4))
        .setStrides((4, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    c = (
        fe.TensorBuilder()
        .setDim((2, 4))
        .setStrides((4, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    matmul_desc = (
        fe.MatMulDescBuilder()
        .setMathPrecision(fe.DataType.FLOAT)
        .setPaddingValue(0.0)
        .build()
    )
    op = (
        fe.OperationBuilder("matmul")
        .setMatmulDesc(matmul_desc)
        .setAMatDesc(a)
        .setBMatDesc(b)
        .setCMatDesc(c)
        .build()
    )

    assert op.inputs == {"a": a, "b": b}
    assert op.outputs == {"c": c}
    assert op.attributes["matmul_desc"] == matmul_desc
    assert matmul_desc.math_precision == fe.DataType.FLOAT
    assert matmul_desc.padding_value == 0.0


def test_mudnn_frontend_python_reduction_builders():
    import cudnn_frontend as fe

    x = (
        fe.TensorBuilder()
        .setDim((2, 3))
        .setStrides((3, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    y = (
        fe.TensorBuilder()
        .setDim((2, 1))
        .setStrides((1, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    group_offset = (
        fe.TensorBuilder()
        .setDim((2,))
        .setStrides((1,))
        .setId(3)
        .setDataType(fe.DataType.INT32)
        .build()
    )
    reduction_desc = (
        fe.ReductionDescBuilder()
        .setReductionOp(fe.CUDNN_REDUCE_TENSOR_ADD)
        .setMathPrecision(fe.CUDNN_DATA_FLOAT)
        .build()
    )
    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
        .setReductionDesc(reduction_desc)
        .setReductionXDesc(x)
        .setReductionYDesc(y)
        .setGroupOffsetDesc(group_offset)
        .build()
    )

    assert fe.CUDNN_TYPE_REDUCTION_OPERATOR_TYPE > 0
    assert fe.CUDNN_BACKEND_REDUCTION_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_REDUCTION_OPERATOR > 0
    assert isinstance(reduction_desc, fe.ReductionDesc)
    assert reduction_desc.mode == fe.ReductionMode.ADD
    assert reduction_desc.math_precision == fe.DataType.FLOAT
    assert op.op_type == "reduction"
    assert op.inputs == {"x": x, "group_offset": group_offset}
    assert op.outputs == {"y": y}
    assert op.attributes["reduction_desc"] == reduction_desc
    assert op.attributes["reduction_mode"] == fe.ReductionMode.ADD


def test_mudnn_frontend_python_graph_reduction_upstream_signature():
    import mudnn

    graph = mudnn.pygraph(
        io_data_type=mudnn.data_type.FLOAT,
        compute_data_type=mudnn.data_type.FLOAT,
    )
    x = graph.tensor(
        name="x",
        dim=(2, 3, 4),
        stride=(12, 4, 1),
        data_type=mudnn.data_type.FLOAT,
    )

    upstream = graph.reduction(x, mode=mudnn.reduction_mode.MUL, name="mul_reduce")
    legacy = graph.reduction(mudnn.reduction_mode.ADD, x, axis=2, name="add_reduce")

    assert upstream.dims == (1, 1, 1)
    assert legacy.dims == (2, 3, 1)
    assert graph.nodes[-2].attributes["mode"] == mudnn.reduction_mode.MUL
    assert graph.nodes[-1].attributes["mode"] == mudnn.reduction_mode.ADD

    grouped_graph = mudnn.pygraph(
        io_data_type=mudnn.data_type.FLOAT,
        compute_data_type=mudnn.data_type.FLOAT,
    )
    grouped_x = grouped_graph.tensor(
        name="grouped_x",
        dim=(2, 3, 4),
        stride=(12, 4, 1),
        data_type=mudnn.data_type.FLOAT,
    )
    group_offset = grouped_graph.tensor(
        name="group_offset",
        dim=(2,),
        stride=(1,),
        data_type=mudnn.data_type.INT32,
    )
    grouped = grouped_graph.reduction(
        grouped_x,
        mode=mudnn.reduction_mode.ADD,
        name="grouped_reduce",
        group_offset=group_offset,
    )

    assert grouped.dims == (1, 1, 1)
    assert grouped_graph.nodes[-1].inputs["group_offset"] is group_offset
    with pytest.raises(NotImplementedError, match="grouped reduction"):
        grouped_graph.validate().build_operation_graph()


def test_mudnn_frontend_python_resample_builders():
    import cudnn_frontend as fe

    x = (
        fe.TensorBuilder()
        .setDim((1, 1, 4, 4))
        .setStrides((16, 16, 4, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    y = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 2))
        .setStrides((4, 4, 2, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    dx = (
        fe.TensorBuilder()
        .setDim((1, 1, 4, 4))
        .setStrides((16, 16, 4, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    idx = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 2))
        .setStrides((4, 4, 2, 1))
        .setId(4)
        .setDataType(fe.DataType.INT64)
        .build()
    )
    desc = (
        fe.ResampleDescBuilder()
        .setMode(fe.CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING)
        .setMathPrecision(fe.CUDNN_DATA_FLOAT)
        .setSpatialDimCount(2)
        .setWindowDim((2, 2))
        .setPrePadding((0, 0))
        .setPostPadding((0, 0))
        .setStride((2, 2))
        .build()
    )
    fwd = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
        .setResampleDesc(desc)
        .setResampleXDesc(x)
        .setResampleYDesc(y)
        .setResampleIndexDesc(idx)
        .setAlpha(1.0)
        .setBeta(0.0)
        .build()
    )
    bwd = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
        .setPoolingDesc(desc)
        .setPoolingDxDesc(dx)
        .setPoolingDyDesc(y)
        .setAlpha(1.0)
        .setBeta(0.0)
        .build()
    )

    assert fe.CUDNN_TYPE_RESAMPLE_MODE > 0
    assert fe.CUDNN_BACKEND_RESAMPLE_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC > 0
    assert isinstance(desc, fe.ResampleDesc)
    assert isinstance(desc, fe.PoolingDesc)
    assert desc.mode == fe.ResampleMode.AVGPOOL_INCLUDE_PADDING
    assert desc.window_dims == (2, 2)
    assert fwd.op_type == "resample_fwd"
    assert fwd.inputs == {"x": x}
    assert fwd.outputs == {"y": y, "idx": idx}
    assert fwd.attributes["resample_desc"] == desc
    assert bwd.op_type == "resample_bwd"
    assert bwd.inputs == {"dy": y}
    assert bwd.outputs == {"dx": dx}


def test_mudnn_frontend_python_concat_reshape_builders():
    import cudnn_frontend as fe

    a = (
        fe.TensorBuilder()
        .setDim((2, 2))
        .setStrides((2, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    b = (
        fe.TensorBuilder()
        .setDim((2, 3))
        .setStrides((3, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    y = (
        fe.TensorBuilder()
        .setDim((2, 5))
        .setStrides((5, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    z = (
        fe.TensorBuilder()
        .setDim((10,))
        .setStrides((1,))
        .setId(4)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )

    concat = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR)
        .setConcatAxis(1)
        .setConcatInputDescs((a, b))
        .setConcatOutputDesc(y)
        .setConcatInplaceIndex(0)
        .build()
    )
    reshape = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
        .setReshapeXDesc(y)
        .setReshapeYDesc(z)
        .setReshapeMode(fe.reshape_mode.LOGICAL)
        .build()
    )
    slice_op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR)
        .setSliceXDesc(y)
        .setSliceYDesc(b)
        .setSliceStartIndices((0, 1))
        .setSliceLimitIndices((2, 4))
        .setSliceStrides((1, 1))
        .build()
    )
    transpose = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR)
        .setTransposeXDesc(a)
        .setTransposeYDesc(b)
        .setTransposePermutation((1, 0))
        .build()
    )

    assert fe.CUDNN_ATTR_OPERATION_CONCAT_AXIS > 0
    assert fe.CUDNN_ATTR_OPERATION_RESHAPE_XDESC > 0
    assert fe.CUDNN_ATTR_OPERATION_RESHAPE_MODE > 0
    assert fe.CUDNN_ATTR_OPERATION_SLICE_XDESC > 0
    assert fe.CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC > 0
    assert concat.op_type == "concat"
    assert concat.inputs == {"x0": a, "x1": b}
    assert concat.outputs == {"y": y}
    assert reshape.attributes["reshape_mode"] == fe.reshape_mode.LOGICAL
    assert concat.attributes["axis"] == 1
    assert concat.attributes["inplace_index"] == 0
    assert reshape.op_type == "reshape"
    assert reshape.inputs == {"x": y}
    assert reshape.outputs == {"y": z}
    assert slice_op.op_type == "slice"
    assert slice_op.inputs == {"x": y}
    assert slice_op.outputs == {"y": b}
    assert slice_op.attributes["start_indices"] == (0, 1)
    assert slice_op.attributes["limit_indices"] == (2, 4)
    assert slice_op.attributes["strides"] == (1, 1)
    assert transpose.op_type == "transpose"
    assert transpose.inputs == {"x": a}
    assert transpose.outputs == {"y": b}
    assert transpose.attributes["permutation"] == (1, 0)


def test_mudnn_frontend_python_rng_builders():
    import cudnn_frontend as fe

    y = (
        fe.TensorBuilder()
        .setDim((2, 3))
        .setStrides((3, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    offset = (
        fe.TensorBuilder()
        .setDim((1,))
        .setStrides((1,))
        .setId(2)
        .setDataType(fe.DataType.INT64)
        .build()
    )
    desc = (
        fe.RngDescBuilder()
        .setDistribution(fe.CUDNN_RNG_DISTRIBUTION_UNIFORM)
        .setUniformDistMinimum(-0.5)
        .setUniformDistMaximum(0.5)
        .setNormalDistMean(1.0)
        .setNormalDistStandardDeviation(2.0)
        .setBernoulliDistProbability(0.25)
        .build()
    )
    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
        .setRngDesc(desc)
        .setRngYDesc(y)
        .setRngSeed(123)
        .setRngOffsetDesc(offset)
        .build()
    )

    assert fe.CUDNN_TYPE_RNG_DISTRIBUTION > 0
    assert fe.CUDNN_BACKEND_RNG_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_RNG_DESC > 0
    assert desc.distribution == fe.RngDistribution.UNIFORM
    assert desc.uniform_minimum == -0.5
    assert desc.uniform_maximum == 0.5
    assert op.op_type == "rng"
    assert op.inputs == {"offset": offset}
    assert op.outputs == {"y": y}
    assert op.attributes["rng_desc"] == desc
    assert op.attributes["seed"] == 123


def test_mudnn_frontend_python_genstats_builders():
    import cudnn_frontend as fe

    x = (
        fe.TensorBuilder()
        .setDim((2, 3, 4))
        .setStrides((12, 4, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    sum_tensor = (
        fe.TensorBuilder()
        .setDim((1, 3, 1))
        .setStrides((3, 1, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    sqsum_tensor = (
        fe.TensorBuilder()
        .setDim((1, 3, 1))
        .setStrides((3, 1, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR)
        .setGenStatsMode(fe.CUDNN_GENSTATS_SUM_SQSUM)
        .setGenStatsMathPrecision(fe.DataType.FLOAT)
        .setGenStatsXDesc(x)
        .setGenStatsSumDesc(sum_tensor)
        .setGenStatsSqSumDesc(sqsum_tensor)
        .build()
    )

    assert fe.CUDNN_TYPE_GENSTATS_MODE > 0
    assert fe.CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC > 0
    assert fe.CUDNN_GENSTATS_SUM_SQSUM == fe.GenStatsMode.SUM_SQSUM
    assert op.op_type == "genstats"
    assert op.inputs == {"x": x}
    assert op.outputs == {"sum": sum_tensor, "sqsum": sqsum_tensor}
    assert op.attributes["mode"] == fe.GenStatsMode.SUM_SQSUM
    assert op.attributes["math_precision"] == fe.DataType.FLOAT


def test_mudnn_frontend_python_bn_finalize_builders():
    import cudnn_frontend as fe

    param = (
        fe.TensorBuilder()
        .setDim((1, 2, 1, 1))
        .setStrides((2, 1, 1, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    accum = (
        fe.TensorBuilder()
        .setDim((1,))
        .setStrides((1,))
        .setId(2)
        .setDataType(fe.DataType.INT64)
        .build()
    )
    inference = (
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR
        )
        .setBnFinalizeStatsMode(fe.CUDNN_BN_FINALIZE_STATISTICS_INFERENCE)
        .setBnFinalizeMathPrecision(fe.DataType.FLOAT)
        .setBnFinalizeScaleDesc(param)
        .setBnFinalizeBiasDesc(param)
        .setBnFinalizePrevRunningMeanDesc(param)
        .setBnFinalizePrevRunningVarDesc(param)
        .setBnFinalizeEqScaleDesc(param)
        .setBnFinalizeEqBiasDesc(param)
        .setBnFinalizeEpsilon(1.0e-5)
        .build()
    )
    training = (
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR
        )
        .setBnFinalizeStatsMode(fe.CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
        .setBnFinalizeMathPrecision(fe.DataType.FLOAT)
        .setBnFinalizeYSumDesc(param)
        .setBnFinalizeYSqSumDesc(param)
        .setBnFinalizeScaleDesc(param)
        .setBnFinalizeBiasDesc(param)
        .setBnFinalizePrevRunningMeanDesc(param)
        .setBnFinalizePrevRunningVarDesc(param)
        .setBnFinalizeUpdatedRunningMeanDesc(param)
        .setBnFinalizeUpdatedRunningVarDesc(param)
        .setBnFinalizeSavedMeanDesc(param)
        .setBnFinalizeSavedInvStdDesc(param)
        .setBnFinalizeEqScaleDesc(param)
        .setBnFinalizeEqBiasDesc(param)
        .setBnFinalizeAccumCountDesc(accum)
        .setBnFinalizeExpAvgFactor(0.25)
        .setBnFinalizeEpsilon(1.0e-5)
        .build()
    )

    assert fe.CUDNN_TYPE_BN_FINALIZE_STATS_MODE > 0
    assert fe.CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC > 0
    assert fe.CUDNN_BN_FINALIZE_STATISTICS_INFERENCE == fe.BnFinalizeStatsMode.INFERENCE
    assert training.op_type == "bn_finalize"
    assert inference.attributes["mode"] == fe.BnFinalizeStatsMode.INFERENCE
    assert training.attributes["mode"] == fe.BnFinalizeStatsMode.TRAINING
    assert training.inputs["accum_count"] == accum
    assert training.outputs["saved_inv_std"] == param
    assert training.attributes["momentum"] == 0.25


def test_mudnn_frontend_python_bn_bwd_weights_builders():
    import cudnn_frontend as fe

    x = (
        fe.TensorBuilder()
        .setDim((2, 2, 2, 2))
        .setStrides((8, 4, 2, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    param = (
        fe.TensorBuilder()
        .setDim((1, 2, 1, 1))
        .setStrides((2, 1, 1, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    out = (
        fe.TensorBuilder()
        .setDim((1, 2, 1, 1))
        .setStrides((2, 1, 1, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )

    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)
        .setBnBwdWeightsMathPrecision(fe.DataType.FLOAT)
        .setBnBwdWeightsMeanDesc(param)
        .setBnBwdWeightsInvstdDesc(param)
        .setBnBwdWeightsBnScaleDesc(param)
        .setBnBwdWeightsXDesc(x)
        .setBnBwdWeightsDyDesc(x)
        .setBnBwdWeightsDBnScaleDesc(out)
        .setBnBwdWeightsDBnBiasDesc(out)
        .setBnBwdWeightsEqDyScaleDesc(out)
        .setBnBwdWeightsEqXScaleDesc(out)
        .setBnBwdWeightsEqBiasDesc(out)
        .build()
    )

    assert fe.CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC > 0
    assert fe.CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS > 0
    assert op.op_type == "bn_bwd_weights"
    assert op.inputs["x"] == x
    assert op.inputs["dy"] == x
    assert op.inputs["scale"] == param
    assert op.outputs["dbn_scale"] == out
    assert op.outputs["eq_bias"] == out
    assert op.attributes["math_precision"] == fe.DataType.FLOAT


def test_mudnn_frontend_python_block_scale_builders():
    import cudnn_frontend as fe

    x = (
        fe.TensorBuilder()
        .setDim((2, 5))
        .setStrides((5, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    scale = (
        fe.TensorBuilder()
        .setDim((2, 3))
        .setStrides((3, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    y = (
        fe.TensorBuilder()
        .setDim((2, 5))
        .setStrides((5, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    quant = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR)
        .setBlockScaleQuantizeXDesc(x)
        .setBlockScaleQuantizeYDesc(y)
        .setBlockScaleQuantizeScaleDesc(scale)
        .setBlockScaleQuantizeMathPrecision(fe.DataType.FLOAT)
        .setBlockScaleQuantizeBlockSize(2)
        .build()
    )
    dequant = (
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR
        )
        .setBlockScaleDequantizeXDesc(y)
        .setBlockScaleDequantizeScaleDesc(scale)
        .setBlockScaleDequantizeYDesc(x)
        .setBlockScaleDequantizeMathPrecision(fe.DataType.FLOAT)
        .setBlockScaleDequantizeBlockSize(2)
        .setBlockScaleDequantizeNegativeScale(True)
        .build()
    )

    assert fe.CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR > 0
    assert fe.CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC > 0
    assert fe.CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE > 0
    assert fe.CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE > 0
    assert quant.op_type == "block_scale_quantize"
    assert quant.inputs == {"x": x}
    assert quant.outputs == {"y": y, "scale": scale}
    assert quant.attributes["math_precision"] == fe.DataType.FLOAT
    assert quant.attributes["block_size"] == 2
    assert dequant.op_type == "block_scale_dequantize"
    assert dequant.inputs == {"x": y, "scale": scale}
    assert dequant.outputs == {"y": x}
    assert dequant.attributes["block_size"] == 2
    assert dequant.attributes["is_negative_scale"] is True


def test_mudnn_frontend_python_moe_grouped_matmul_builders():
    import cudnn_frontend as fe

    token = (
        fe.TensorBuilder()
        .setDim((1, 6, 4))
        .setStrides((24, 4, 1))
        .setId(1)
        .setDataType(fe.DataType.BFLOAT16)
        .build()
    )
    weight = (
        fe.TensorBuilder()
        .setDim((2, 4, 8))
        .setStrides((32, 8, 1))
        .setId(2)
        .setDataType(fe.DataType.BFLOAT16)
        .build()
    )
    first_token_offset = (
        fe.TensorBuilder()
        .setDim((2,))
        .setStrides((1,))
        .setId(3)
        .setDataType(fe.DataType.INT64)
        .build()
    )
    token_index = (
        fe.TensorBuilder()
        .setDim((2, 3))
        .setStrides((3, 1))
        .setId(4)
        .setDataType(fe.DataType.INT64)
        .build()
    )
    token_ks = (
        fe.TensorBuilder()
        .setDim((6,))
        .setStrides((1,))
        .setId(5)
        .setDataType(fe.DataType.INT64)
        .build()
    )
    output = (
        fe.TensorBuilder()
        .setDim((1, 6, 8))
        .setStrides((48, 8, 1))
        .setId(6)
        .setDataType(fe.DataType.BFLOAT16)
        .build()
    )
    dweight = (
        fe.TensorBuilder()
        .setDim((2, 4, 8))
        .setStrides((32, 8, 1))
        .setId(7)
        .setDataType(fe.DataType.BFLOAT16)
        .build()
    )

    fwd = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR)
        .setMoeGroupedMatmulTokenDesc(token)
        .setMoeGroupedMatmulWeightDesc(weight)
        .setMoeGroupedMatmulFirstTokenOffsetDesc(first_token_offset)
        .setMoeGroupedMatmulOutputDesc(output)
        .setMoeGroupedMatmulTokenIndexDesc(token_index)
        .setMoeGroupedMatmulTokenKsDesc(token_ks)
        .setMoeGroupedMatmulMode(fe.moe_grouped_matmul_mode.SCATTER)
        .setMoeGroupedMatmulMathPrecision(fe.DataType.FLOAT)
        .setMoeGroupedMatmulTopK(2)
        .build()
    )
    bwd = (
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR
        )
        .setMoeGroupedMatmulBwdDOutputDesc(output)
        .setMoeGroupedMatmulBwdTokenDesc(token)
        .setMoeGroupedMatmulBwdFirstTokenOffsetDesc(first_token_offset)
        .setMoeGroupedMatmulBwdDWeightDesc(dweight)
        .setMoeGroupedMatmulBwdMathPrecision(fe.DataType.FLOAT)
        .build()
    )

    assert fe.CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR > 0
    assert fe.CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K > 0
    assert fe.CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC > 0
    assert fwd.op_type == "moe_grouped_matmul"
    assert fwd.inputs["token"] is token
    assert fwd.inputs["weight"] is weight
    assert fwd.inputs["first_token_offset"] is first_token_offset
    assert fwd.inputs["token_index"] is token_index
    assert fwd.inputs["token_ks"] is token_ks
    assert fwd.outputs == {"output": output}
    assert fwd.attributes["mode"] == fe.moe_grouped_matmul_mode.SCATTER
    assert fwd.attributes["compute_data_type"] == fe.DataType.FLOAT
    assert fwd.attributes["top_k"] == 2
    assert bwd.op_type == "moe_grouped_matmul_bwd"
    assert bwd.inputs == {
        "doutput": output,
        "token": token,
        "first_token_offset": first_token_offset,
    }
    assert bwd.outputs == {"dweight": dweight}


def test_mudnn_frontend_python_paged_cache_load_builders():
    import cudnn_frontend as fe

    container = (
        fe.TensorBuilder()
        .setDim((3, 2, 2))
        .setStrides((4, 2, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    y = (
        fe.TensorBuilder()
        .setDim((2, 2, 2))
        .setStrides((4, 2, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    sequence = (
        fe.TensorBuilder()
        .setDim((2,))
        .setStrides((1,))
        .setId(3)
        .setDataType(fe.DataType.INT64)
        .build()
    )
    page_table = (
        fe.TensorBuilder()
        .setDim((3,))
        .setStrides((1,))
        .setId(4)
        .setDataType(fe.DataType.INT64)
        .build()
    )
    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR)
        .setPagedCacheLoadContainerDesc(container)
        .setPagedCacheLoadYDesc(y)
        .setPagedCacheLoadSequenceDesc(sequence)
        .setPagedCacheLoadPageTableDesc(page_table)
        .build()
    )

    assert fe.CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC > 0
    assert fe.CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC > 0
    assert op.op_type == "paged_cache_load"
    assert op.inputs == {
        "container": container,
        "sequence": sequence,
        "page_table": page_table,
    }
    assert op.outputs == {"y": y}


def test_mudnn_frontend_python_signal_builders():
    import cudnn_frontend as fe

    x = (
        fe.TensorBuilder()
        .setDim((2, 3))
        .setStrides((3, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    y = (
        fe.TensorBuilder()
        .setDim((2, 3))
        .setStrides((3, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    flag = (
        fe.TensorBuilder()
        .setDim((1,))
        .setStrides((1,))
        .setId(3)
        .setDataType(fe.DataType.INT64)
        .build()
    )
    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR)
        .setSignalMode(fe.CUDNN_SIGNAL_SET)
        .setSignalValue(17)
        .setSignalFlagDesc(flag)
        .setSignalXDesc(x)
        .setSignalYDesc(y)
        .build()
    )

    assert fe.CUDNN_TYPE_SIGNAL_MODE > 0
    assert fe.CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC > 0
    assert fe.CUDNN_SIGNAL_SET == fe.SignalMode.SET
    assert fe.CUDNN_SIGNAL_WAIT == fe.SignalMode.WAIT
    assert op.op_type == "signal"
    assert op.inputs == {"flag": flag, "x": x}
    assert op.outputs == {"y": y}
    assert op.attributes["mode"] == fe.SignalMode.SET
    assert op.attributes["value"] == 17


def test_mudnn_frontend_python_norm_forward_builders():
    import cudnn_frontend as fe

    x = (
        fe.TensorBuilder()
        .setDim((2, 2, 2, 2))
        .setStrides((8, 4, 2, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    param = (
        fe.TensorBuilder()
        .setDim((1, 2, 1, 1))
        .setStrides((2, 1, 1, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    y = (
        fe.TensorBuilder()
        .setDim((2, 2, 2, 2))
        .setStrides((8, 4, 2, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
        .setNormMode(fe.CUDNN_BATCH_NORM)
        .setNormFwdPhase(fe.CUDNN_NORM_FWD_INFERENCE)
        .setNormFwdXDesc(x)
        .setNormFwdMeanDesc(param)
        .setNormFwdInvVarianceDesc(param)
        .setNormFwdScaleDesc(param)
        .setNormFwdBiasDesc(param)
        .setNormFwdYDesc(y)
        .setNormFwdEpsilon(1.0e-5)
        .build()
    )

    assert fe.CUDNN_TYPE_NORM_MODE > 0
    assert fe.CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_NORM_FWD_MODE > 0
    assert fe.CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC > 0
    assert fe.CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC > 0
    assert fe.CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC > 0
    assert op.op_type == "norm_fwd"
    assert op.inputs["x"] == x
    assert op.inputs["mean"] == param
    assert op.outputs == {"y": y}
    assert op.attributes["mode"] == fe.NormMode.BATCH_NORM
    assert op.attributes["phase"] == fe.NormFwdPhase.INFERENCE
    assert op.attributes["epsilon"] == 1.0e-5

    training_op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
        .setNormMode(fe.CUDNN_BATCH_NORM)
        .setNormFwdPhase(fe.CUDNN_NORM_FWD_TRAINING)
        .setNormFwdXDesc(x)
        .setNormFwdMeanDesc(param)
        .setNormFwdInvVarianceDesc(param)
        .setNormFwdScaleDesc(param)
        .setNormFwdBiasDesc(param)
        .setNormFwdInputRunningMeanDesc(param)
        .setNormFwdInputRunningVarDesc(param)
        .setNormFwdOutputRunningMeanDesc(param)
        .setNormFwdOutputRunningVarDesc(param)
        .setNormFwdYDesc(y)
        .setNormFwdExpAvgFactor(0.25)
        .setNormFwdEpsilon(1.0e-4)
        .build()
    )
    assert training_op.inputs["input_running_mean"] == param
    assert training_op.outputs["output_running_variance"] == param
    assert training_op.attributes["phase"] == fe.NormFwdPhase.TRAINING
    assert training_op.attributes["momentum"] == 0.25
    assert training_op.attributes["epsilon"] == 1.0e-4

    backward_op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
        .setNormBwdMode(fe.CUDNN_BATCH_NORM)
        .setNormBwdXDesc(x)
        .setNormBwdMeanDesc(param)
        .setNormBwdInvVarianceDesc(param)
        .setNormBwdDyDesc(x)
        .setNormBwdScaleDesc(param)
        .setNormBwdDxDesc(y)
        .setNormBwdDScaleDesc(param)
        .setNormBwdDBiasDesc(param)
        .setNormBwdEpsilon(1.0e-5)
        .build()
    )
    assert fe.CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_NORM_BWD_MODE > 0
    assert backward_op.op_type == "norm_bwd"
    assert backward_op.inputs["dy"] == x
    assert backward_op.outputs["dx"] == y
    assert backward_op.attributes["mode"] == fe.NormMode.BATCH_NORM
    assert backward_op.attributes["epsilon"] == 1.0e-5

    tensor_epsilon_op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
        .setNormMode(fe.CUDNN_BATCH_NORM)
        .setNormFwdPhase(fe.CUDNN_NORM_FWD_TRAINING)
        .setNormFwdXDesc(x)
        .setNormFwdMeanDesc(param)
        .setNormFwdInvVarianceDesc(param)
        .setNormFwdScaleDesc(param)
        .setNormFwdBiasDesc(param)
        .setNormFwdInputRunningMeanDesc(param)
        .setNormFwdInputRunningVarDesc(param)
        .setNormFwdOutputRunningMeanDesc(param)
        .setNormFwdOutputRunningVarDesc(param)
        .setNormFwdYDesc(y)
        .setNormFwdEpsilon(param)
        .setNormFwdExpAvgFactor(param)
        .build()
    )
    assert tensor_epsilon_op.inputs["epsilon"] == param
    assert tensor_epsilon_op.inputs["momentum"] == param
    assert "epsilon" not in tensor_epsilon_op.attributes
    assert "momentum" not in tensor_epsilon_op.attributes


def test_mudnn_frontend_python_convolution_builders():
    import cudnn_frontend as fe

    x = (
        fe.TensorBuilder()
        .setDim((1, 1, 3, 3))
        .setStrides((9, 9, 3, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    w = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 2))
        .setStrides((4, 4, 2, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    y = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 2))
        .setStrides((4, 4, 2, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    conv_desc = (
        fe.ConvDescBuilder()
        .setMathPrecision(fe.CUDNN_DATA_FLOAT)
        .setConvolutionMode(fe.CUDNN_CROSS_CORRELATION)
        .setSpatialDimCount(2)
        .setPrePadding((0, 0))
        .setPostPadding((0, 0))
        .setStride((1, 1))
        .setDilation((1, 1))
        .setGroupCount(1)
        .build()
    )
    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
        .setConvDesc(conv_desc)
        .setConvXDesc(x)
        .setConvWDesc(w)
        .setConvYDesc(y)
        .setAlpha(1.0)
        .setBeta(0.0)
        .build()
    )

    assert fe.CUDNN_TYPE_CONVOLUTION_MODE > 0
    assert fe.CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_CONVOLUTION_GROUP_COUNT > 0
    assert fe.CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC > 0
    assert isinstance(conv_desc, fe.ConvDesc)
    assert isinstance(conv_desc, fe.ConvolutionDesc)
    assert conv_desc.math_precision == fe.DataType.FLOAT
    assert conv_desc.mode == fe.ConvolutionMode.CROSS_CORRELATION
    assert conv_desc.pre_paddings == (0, 0)
    assert conv_desc.post_paddings == (0, 0)
    assert conv_desc.filter_strides == (1, 1)
    assert conv_desc.dilations == (1, 1)
    assert conv_desc.group_count == 1
    assert op.op_type == "conv_fprop"
    assert op.inputs == {"x": x, "w": w}
    assert op.outputs == {"y": y}
    assert op.attributes["conv_desc"] == conv_desc
    assert op.attributes["alpha1"] == 1.0
    assert op.attributes["beta"] == 0.0


def test_mudnn_frontend_python_convolution_backward_builders():
    import cudnn_frontend as fe

    dy = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 2))
        .setStrides((4, 4, 2, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    w = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 2))
        .setStrides((4, 4, 2, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    dx = (
        fe.TensorBuilder()
        .setDim((1, 1, 3, 3))
        .setStrides((9, 9, 3, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    x = (
        fe.TensorBuilder()
        .setDim((1, 1, 3, 3))
        .setStrides((9, 9, 3, 1))
        .setId(4)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    dw = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 2))
        .setStrides((4, 4, 2, 1))
        .setId(5)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    conv_desc = (
        fe.ConvolutionDescBuilder()
        .setMathPrecision(fe.CUDNN_DATA_FLOAT)
        .setConvolutionMode(fe.CUDNN_CROSS_CORRELATION)
        .setPrePadding((0, 0))
        .setPostPadding((0, 0))
        .setStride((1, 1))
        .setDilation((1, 1))
        .build()
    )
    dgrad = (
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
        )
        .setConvDesc(conv_desc)
        .setConvWDesc(w)
        .setConvDxDesc(dx)
        .setConvDyDesc(dy)
        .setAlpha(1.0)
        .setBeta(0.0)
        .build()
    )
    wgrad = (
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR
        )
        .setConvDesc(conv_desc)
        .setConvDWDesc(dw)
        .setConvXDesc(x)
        .setConvDYDesc(dy)
        .setAlpha(1.0)
        .setBeta(0.0)
        .build()
    )

    assert fe.CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR > 0
    assert fe.CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR > 0
    assert fe.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC > 0
    assert fe.CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC > 0
    assert dgrad.op_type == "conv_bwd_data"
    assert dgrad.inputs == {"w": w, "dy": dy}
    assert dgrad.outputs == {"dx": dx}
    assert dgrad.attributes["conv_desc"] == conv_desc
    assert wgrad.op_type == "conv_bwd_filter"
    assert wgrad.inputs == {"x": x, "dy": dy}
    assert wgrad.outputs == {"dw": dw}
    assert wgrad.attributes["conv_desc"] == conv_desc


def test_mudnn_frontend_python_sdpa_builders():
    import cudnn_frontend as fe

    q = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 2))
        .setStrides((4, 4, 2, 1))
        .setId(1)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    k = (
        fe.TensorBuilder()
        .setDim((1, 1, 3, 2))
        .setStrides((6, 6, 2, 1))
        .setId(2)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    v = (
        fe.TensorBuilder()
        .setDim((1, 1, 3, 2))
        .setStrides((6, 6, 2, 1))
        .setId(3)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    o = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 2))
        .setStrides((4, 4, 2, 1))
        .setId(4)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )
    attn_probs = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 3))
        .setStrides((6, 6, 3, 1))
        .setId(100)
        .setDataType(fe.DataType.FLOAT)
        .setVirtual(True)
        .build()
    )
    dropout_mask = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 3))
        .setStrides((6, 6, 3, 1))
        .setId(101)
        .setDataType(fe.DataType.BOOLEAN)
        .setVirtual(True)
        .build()
    )
    stats = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 1))
        .setStrides((2, 2, 1, 1))
        .setId(102)
        .setDataType(fe.DataType.FLOAT)
        .setVirtual(True)
        .build()
    )
    bias = (
        fe.TensorBuilder()
        .setDim((1, 1, 2, 3))
        .setStrides((6, 6, 3, 1))
        .setId(5)
        .setDataType(fe.DataType.FLOAT)
        .build()
    )

    op = (
        fe.OperationBuilder("sdpa")
        .setQDesc(q)
        .setKDesc(k)
        .setVDesc(v)
        .setBiasDesc(bias)
        .setODesc(o)
        .setAttnProbsDesc(attn_probs)
        .setDropoutMaskDesc(dropout_mask)
        .setSdpaStatsDesc(stats)
        .setSdpaScale(0.5)
        .setSdpaGenerateStats(False)
        .setSdpaCausalMask(False)
        .setSdpaUsePaddingMask(False)
        .setSdpaUseAlibiMask(False)
        .setSdpaUseCausalMaskBottomRight(False)
        .setSdpaDiagonalAlignment(fe.diagonal_alignment.TOP_LEFT)
        .setSdpaDiagonalBandRightBound(0)
        .setSdpaDiagonalBandLeftBound(1)
        .setSdpaDropout(0.0)
        .setSdpaSeqLenQDesc(stats)
        .setSdpaSeqLenKVDesc(stats)
        .build()
    )

    assert op.inputs == {
        "q": q,
        "k": k,
        "v": v,
        "bias": bias,
        "seq_len_q": stats,
        "seq_len_kv": stats,
    }
    assert op.outputs == {
        "o": o,
        "attn_probs": attn_probs,
        "dropout_mask": dropout_mask,
        "stats": stats,
    }
    assert op.attributes == {
        "sdpa_scale": 0.5,
        "sdpa_is_inference": True,
        "sdpa_generate_stats": False,
        "sdpa_causal_mask": True,
        "sdpa_use_padding_mask": False,
        "sdpa_use_causal_mask_bottom_right": False,
        "sdpa_diagonal_alignment": fe.diagonal_alignment.TOP_LEFT,
        "sdpa_diagonal_band_right_bound": 0,
        "sdpa_diagonal_band_left_bound": 1,
    }
    diagonal_op = (
        fe.OperationBuilder("sdpa")
        .setSdpaUseCausalMaskBottomRight(True)
        .setSdpaDiagonalAlignment(fe.diagonal_alignment.BOTTOM_RIGHT)
        .setSdpaDiagonalBandLeftBound(2)
        .setSdpaDiagonalBandRightBound(3)
        .build()
    )
    assert diagonal_op.attributes == {
        "sdpa_use_causal_mask_bottom_right": True,
        "sdpa_diagonal_alignment": fe.diagonal_alignment.BOTTOM_RIGHT,
        "sdpa_diagonal_band_left_bound": 2,
        "sdpa_diagonal_band_right_bound": 3,
    }
    with pytest.raises(ValueError, match="ALiBi"):
        fe.OperationBuilder("sdpa").setSdpaUseAlibiMask(True)
    with pytest.raises(ValueError, match="dropout"):
        fe.OperationBuilder("sdpa").setSdpaDropout(0.25)
    with pytest.raises(ValueError, match="sliding_window_length"):
        fe.OperationBuilder("sdpa").setSdpaSlidingWindowLength(2)
    with pytest.raises(ValueError, match="ragged"):
        fe.OperationBuilder("sdpa").setSdpaMaxTotalSeqLenKV(3)
    with pytest.raises(ValueError, match="score modifiers"):
        fe.OperationBuilder("sdpa").setSdpaScoreMod(lambda *args: args[0])
    with pytest.raises(ValueError, match="score_sum_exp"):
        fe.OperationBuilder("sdpa").setSdpaScoreSumExpDesc(stats)
    with pytest.raises(ValueError, match="score_max"):
        fe.OperationBuilder("sdpa").setSdpaScoreMaxDesc(stats)
    with pytest.raises(ValueError, match="sink-token"):
        fe.OperationBuilder("sdpa").setSdpaSinkTokenDesc(stats)
    for method_name, error_name in (
        ("setSdpaDescaleQDesc", "descale_q"),
        ("setSdpaDescaleKDesc", "descale_k"),
        ("setSdpaDescaleVDesc", "descale_v"),
        ("setSdpaDescaleSDesc", "descale_s"),
        ("setSdpaScaleSDesc", "scale_s"),
        ("setSdpaScaleODesc", "scale_o"),
    ):
        with pytest.raises(ValueError, match=error_name):
            getattr(fe.OperationBuilder("sdpa"), method_name)(stats)


def test_mudnn_frontend_python_sdpa_backward_builders():
    import cudnn_frontend as fe

    def contiguous_strides(dims):
        strides = []
        stride = 1
        for dim in reversed(dims):
            strides.insert(0, stride)
            stride *= dim
        return tuple(strides)

    def tensor(uid, dims, dtype=fe.DataType.FLOAT):
        return (
            fe.TensorBuilder()
            .setDim(dims)
            .setStrides(contiguous_strides(dims))
            .setId(uid)
            .setDataType(dtype)
            .build()
        )

    q = tensor(1, (1, 1, 2, 2))
    k = tensor(2, (1, 1, 3, 2))
    v = tensor(3, (1, 1, 3, 4))
    o = tensor(4, (1, 1, 2, 4))
    d_o = tensor(5, (1, 1, 2, 4))
    stats = tensor(6, (1, 1, 2, 1))
    d_q = tensor(7, (1, 1, 2, 2))
    d_k = tensor(8, (1, 1, 3, 2))
    d_v = tensor(9, (1, 1, 3, 4))
    bias = tensor(10, (1, 1, 2, 3))
    d_bias = tensor(11, (1, 1, 2, 3))
    mask = tensor(12, (1, 1, 2, 3), fe.DataType.BOOLEAN)
    seq_q = tensor(13, (1,), fe.DataType.INT32)
    seq_kv = tensor(14, (1, 1, 1, 1), fe.DataType.INT64)

    op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)
        .setQDesc(q)
        .setKDesc(k)
        .setVDesc(v)
        .setODesc(o)
        .setSdpaBwdDODesc(d_o)
        .setSdpaBwdStatsDesc(stats)
        .setSdpaBwdDQDesc(d_q)
        .setSdpaBwdDKDesc(d_k)
        .setSdpaBwdDVDesc(d_v)
        .setBiasDesc(bias)
        .setSdpaBwdDBiasDesc(d_bias)
        .setMaskDesc(mask)
        .setSdpaBwdScale(0.5)
        .setSdpaBwdCausalMask(True)
        .setSdpaBwdDiagonalAlignment(fe.diagonal_alignment.TOP_LEFT)
        .setSdpaBwdDiagonalBandRightBound(0)
        .setSdpaBwdDiagonalBandLeftBound(1)
        .setSdpaBwdUsePaddingMask(False)
        .setSdpaBwdUseAlibiMask(False)
        .setSdpaBwdUseCausalMaskBottomRight(False)
        .setSdpaBwdDropout(0.0)
        .setSdpaBwdUseDeterministicAlgorithm(True)
        .setSdpaBwdSeqLenQDesc(seq_q)
        .setSdpaBwdSeqLenKVDesc(seq_kv)
        .build()
    )

    assert op.op_type == "sdpa_bwd"
    assert op.inputs == {
        "q": q,
        "k": k,
        "v": v,
        "o": o,
        "d_o": d_o,
        "stats": stats,
        "bias": bias,
        "mask": mask,
        "seq_len_q": seq_q,
        "seq_len_kv": seq_kv,
    }
    assert op.outputs == {"d_q": d_q, "d_k": d_k, "d_v": d_v, "d_bias": d_bias}
    assert op.attributes == {
        "sdpa_scale": 0.5,
        "sdpa_causal_mask": True,
        "sdpa_diagonal_alignment": fe.diagonal_alignment.TOP_LEFT,
        "sdpa_diagonal_band_right_bound": 0,
        "sdpa_diagonal_band_left_bound": 1,
        "sdpa_use_padding_mask": False,
        "sdpa_use_causal_mask_bottom_right": False,
        "sdpa_bwd_deterministic": True,
    }
    diagonal_bwd = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)
        .setSdpaBwdUseCausalMaskBottomRight(True)
        .setSdpaBwdDiagonalAlignment(fe.diagonal_alignment.BOTTOM_RIGHT)
        .setSdpaBwdDiagonalBandLeftBound(2)
        .setSdpaBwdDiagonalBandRightBound(3)
        .build()
    )
    assert diagonal_bwd.attributes == {
        "sdpa_use_causal_mask_bottom_right": True,
        "sdpa_diagonal_alignment": fe.diagonal_alignment.BOTTOM_RIGHT,
        "sdpa_diagonal_band_left_bound": 2,
        "sdpa_diagonal_band_right_bound": 3,
    }
    with pytest.raises(ValueError, match="ALiBi"):
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR
        ).setSdpaBwdUseAlibiMask(True)
    with pytest.raises(ValueError, match="dropout"):
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR
        ).setSdpaBwdDropout(0.5)
    with pytest.raises(ValueError, match="sliding_window_length"):
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR
        ).setSdpaBwdSlidingWindowLength(2)
    seq_op = (
        fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)
        .setSdpaBwdSeqLenQDesc(seq_q)
        .setSdpaBwdSeqLenKvDesc(seq_kv)
        .build()
    )
    assert seq_op.inputs == {"seq_len_q": seq_q, "seq_len_kv": seq_kv}
    with pytest.raises(ValueError, match="ragged"):
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR
        ).setSdpaBwdMaxTotalSeqLenQ(2)
    with pytest.raises(ValueError, match="score modifiers"):
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR
        ).setSdpaBwdScoreMod(lambda *args: args[0])
    with pytest.raises(ValueError, match="score modifiers"):
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR
        ).setSdpaBwdScoreModBprop(lambda *args: args[0])
    with pytest.raises(ValueError, match="sink-token"):
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR
        ).setSdpaBwdSinkTokenDesc(stats)
    with pytest.raises(ValueError, match="dSink-token"):
        fe.OperationBuilder(
            fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR
        ).setSdpaBwdDSinkTokenDesc(stats)
    for method_name, error_name in (
        ("setSdpaBwdDescaleQDesc", "descale_q"),
        ("setSdpaBwdDescaleKDesc", "descale_k"),
        ("setSdpaBwdDescaleVDesc", "descale_v"),
        ("setSdpaBwdDescaleODesc", "descale_o"),
        ("setSdpaBwdDescaleDODesc", "descale_dO"),
        ("setSdpaBwdDescaleSDesc", "descale_s"),
        ("setSdpaBwdDescaleDPDesc", "descale_dP"),
        ("setSdpaBwdScaleSDesc", "scale_s"),
        ("setSdpaBwdScaleDQDesc", "scale_dQ"),
        ("setSdpaBwdScaleDKDesc", "scale_dK"),
        ("setSdpaBwdScaleDVDesc", "scale_dV"),
        ("setSdpaBwdScaleDPDesc", "scale_dP"),
    ):
        with pytest.raises(ValueError, match=error_name):
            getattr(
                fe.OperationBuilder(fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR),
                method_name,
            )(stats)


def test_mudnn_frontend_python_graph_api_sdpa_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=(2, 4, 8, 16), data_type=fe.DataType.HALF)
    k = graph.tensor(name="k", dim=(2, 4, 8, 16), data_type=fe.DataType.HALF)
    v = graph.tensor(name="v", dim=(2, 4, 8, 32), data_type=fe.DataType.HALF)
    o, stats = graph.sdpa(
        q,
        k,
        v,
        name="attention",
        attn_scale=0.25,
        causal_mask=True,
        is_inference=True,
        compute_data_type=fe.DataType.FLOAT,
    )

    assert o.dims == (2, 4, 8, 32)
    assert stats is None
    assert graph.nodes[-1].op_type == "sdpa"
    assert graph.nodes[-1].attributes["causal_mask"] is True
    assert graph.nodes[-1].attributes["compute_data_type"] == fe.DataType.FLOAT
    graph.validate()
    graph.build_operation_graph()


def test_mudnn_frontend_python_graph_api_scaled_dot_product_attention_alias():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=(1, 2, 4, 8), data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=(1, 2, 4, 8), data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=(1, 2, 4, 16), data_type=fe.DataType.FLOAT)
    bias = graph.tensor(name="bias", dim=(1, 2, 4, 4), data_type=fe.DataType.FLOAT)

    o, stats = graph.scaled_dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        name="attention_alias",
        attn_scale=0.125,
        use_causal_mask=True,
        generate_stats=False,
        compute_data_type=fe.DataType.FLOAT,
    )

    assert o.dims == (1, 2, 4, 16)
    assert stats is None
    assert graph.nodes[-1].op_type == "sdpa"
    assert graph.nodes[-1].name == "attention_alias"
    assert graph.nodes[-1].inputs["bias"] is bias
    assert graph.nodes[-1].attributes["attn_scale"] == 0.125
    assert graph.nodes[-1].attributes["causal_mask"] is True
    assert graph.nodes[-1].attributes["compute_data_type"] == fe.DataType.FLOAT
    assert graph.nodes[-1].attributes["generate_stats"] is False
    graph.validate().build_operation_graph()


def test_mudnn_frontend_python_graph_api_sdpa_tensor_attn_scale():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF)
    k = graph.tensor(name="k", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF)
    v = graph.tensor(name="v", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF)
    scale = graph.tensor(name="scale", dim=(1,), data_type=fe.DataType.DOUBLE)

    o, stats = graph.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_scale=scale,
        generate_stats=True,
    )
    assert stats is not None
    assert graph.nodes[-1].inputs["scale"] is scale
    assert graph.nodes[-1].attributes["attn_scale"] is None
    graph.validate().build_operation_graph()

    d_o = graph.tensor(name="do", dim=o.dims, data_type=fe.DataType.HALF)
    d_q, d_k, d_v = graph.sdpa_backward(
        q,
        k,
        v,
        o,
        d_o,
        stats,
        attn_scale=scale,
        deterministic=True,
    )
    assert d_q.dims == q.dims
    assert d_k.dims == k.dims
    assert d_v.dims == v.dims
    assert graph.nodes[-1].inputs["scale"] is scale
    assert graph.nodes[-1].attributes["attn_scale"] is None
    graph.validate().build_operation_graph()

    restored = fe.pygraph().deserialize(graph.serialize())
    assert restored.nodes[-2].inputs["scale"].name == "scale"
    assert restored.nodes[-1].inputs["scale"].name == "scale"


def test_mudnn_frontend_python_graph_api_sdpa_diagonal_causal_alias():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    q = graph.tensor(name="q", dim=(1, 1, 3, 4), data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=(1, 1, 3, 4), data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=(1, 1, 3, 4), data_type=fe.DataType.FLOAT)

    graph.sdpa(
        q,
        k,
        v,
        generate_stats=False,
        diagonal_alignment=fe.diagonal_alignment.TOP_LEFT,
        diagonal_band_right_bound=0,
    )

    assert graph.nodes[-1].attributes["causal_mask"] is True
    graph.scaled_dot_product_attention(
        q,
        k,
        v,
        generate_stats=False,
        diagonal_alignment=fe.diagonal_alignment.TOP_LEFT,
        diagonal_band_right_bound=0,
        use_alibi_mask=False,
        dropout=0.0,
    )
    assert graph.nodes[-1].attributes["causal_mask"] is True

    graph.sdpa(
        q,
        k,
        v,
        generate_stats=False,
        diagonal_alignment=fe.diagonal_alignment.TOP_LEFT,
    )
    assert graph.nodes[-1].attributes["causal_mask"] is False
    graph.sdpa(
        q,
        k,
        v,
        generate_stats=False,
        diagonal_alignment=fe.diagonal_alignment.BOTTOM_RIGHT,
        diagonal_band_right_bound=0,
    )
    assert (
        graph.nodes[-1].attributes["diagonal_alignment"]
        == fe.diagonal_alignment.BOTTOM_RIGHT
    )
    graph.sdpa(
        q,
        k,
        v,
        generate_stats=False,
        diagonal_band_left_bound=0,
        diagonal_band_right_bound=0,
    )
    assert graph.nodes[-1].attributes["diagonal_band_left_bound"] == 0


def test_mudnn_frontend_python_graph_api_sdpa_unsupported_cudnn_keywords():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    q = graph.tensor(name="q", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT)
    seq = graph.tensor(name="seq", dim=(1,), data_type=fe.DataType.INT32)
    aux = graph.tensor(name="aux", dim=(1, 1, 2, 2), data_type=fe.DataType.FLOAT)

    graph.sdpa(
        q,
        k,
        v,
        generate_stats=False,
        implementation=fe.attention_implementation.AUTO,
        unfuse_fma=False,
        use_causal_mask_bottom_right=False,
        dropout=(0.0, aux, aux),
    )
    assert graph.nodes[-1].op_type == "sdpa"
    graph.sdpa(
        q,
        k,
        v,
        generate_stats=False,
        implementation=fe.attention_implementation.UNIFIED,
    )
    assert (
        graph.nodes[-1].attributes["implementation"]
        == fe.attention_implementation.UNIFIED
    )

    graph.sdpa(q, k, v, generate_stats=False, use_alibi_mask=True)
    assert graph.nodes[-1].attributes["use_alibi_mask"] is True
    with pytest.raises(NotImplementedError, match="use_causal_mask_bottom_right"):
        graph.sdpa(q, k, v, generate_stats=False, use_causal_mask_bottom_right=True)
    graph.sdpa(q, k, v, generate_stats=False, seq_len_q=seq)
    assert graph.nodes[-1].inputs["seq_len_q"] is seq
    graph.sdpa(q, k, v, generate_stats=False, cu_seq_len_q=seq)
    assert graph.nodes[-1].inputs["seq_len_q"] is seq
    with pytest.raises(ValueError, match="seq_len_q and cu_seq_len_q disagree"):
        graph.sdpa(
            q,
            k,
            v,
            generate_stats=False,
            seq_len_q=seq,
            cu_seq_len_q=aux,
        )
    graph.scaled_dot_product_attention(
        q, k, v, generate_stats=False, max_total_seq_len_q=2
    )
    assert graph.nodes[-1].attributes["max_total_seq_len_q"] == 2
    with pytest.raises(NotImplementedError, match="sliding_window_length"):
        graph.sdpa(q, k, v, generate_stats=False, sliding_window_length=2)
    with pytest.raises(NotImplementedError, match="sliding_window_length"):
        graph.scaled_dot_product_attention(
            q, k, v, generate_stats=False, sliding_window_length=2
        )
    with pytest.raises(NotImplementedError, match="implementation"):
        graph.sdpa(
            q,
            k,
            v,
            generate_stats=False,
            implementation=fe.attention_implementation.COMPOSITE,
        )
    graph.sdpa(q, k, v, generate_stats=False, unfuse_fma=True)
    assert graph.nodes[-1].attributes["unfuse_fma"] is True
    block_mask = graph.tensor(
        name="block_mask", dim=(1, 1, 1, 1), data_type=fe.DataType.UINT8
    )
    graph.sdpa(q, k, v, generate_stats=False, block_mask=block_mask)
    assert graph.nodes[-1].inputs["block_mask"] is block_mask
    graph.sdpa(q, k, v, generate_stats=False, rng_dump=aux)
    assert graph.nodes[-1].outputs["rng_dump"] is aux
    page_table_aux = graph.tensor(
        name="page_table_aux", dim=(1, 1), data_type=fe.DataType.INT32
    )
    graph.sdpa(
        q,
        k,
        v,
        generate_stats=False,
        paged_attention_k_table=page_table_aux,
        paged_attention_v_table=page_table_aux,
        paged_attention_max_seq_len_kv=2,
    )
    assert graph.nodes[-1].inputs["paged_attention_k_table"] is page_table_aux
    assert graph.nodes[-1].inputs["paged_attention_v_table"] is page_table_aux
    assert graph.nodes[-1].attributes["paged_attention_max_seq_len_kv"] == 2
    with pytest.raises(ValueError, match="dropout requires rng_dump"):
        graph.sdpa(q, k, v, generate_stats=False, dropout=0.25)
    graph.sdpa(q, k, v, generate_stats=False, dropout=(0.25, seq, seq), rng_dump=aux)
    assert graph.nodes[-1].attributes["dropout_probability"] == 0.25
    score_graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    sq = score_graph.tensor(name="q", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT)
    sk = score_graph.tensor(name="k", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT)
    sv = score_graph.tensor(name="v", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT)
    score_graph.sdpa(
        sq, sk, sv, generate_stats=False, score_mod=lambda graph, score: score
    )
    assert score_graph.nodes[-1].attributes["score_mod"] is True
    score_graph.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.A]
    )
    assert score_graph.check_support() is score_graph
    paged_graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    pq = paged_graph.tensor(name="q", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT)
    pk = paged_graph.tensor(name="k", dim=(4, 1, 1, 4), data_type=fe.DataType.FLOAT)
    pv = paged_graph.tensor(name="v", dim=(4, 1, 1, 4), data_type=fe.DataType.FLOAT)
    page_table = paged_graph.tensor(
        name="page_table", dim=(1, 1), data_type=fe.DataType.INT32
    )
    paged_graph.sdpa(
        pq,
        pk,
        pv,
        generate_stats=False,
        paged_attention_k_table=page_table,
        paged_attention_v_table=page_table,
        paged_attention_max_seq_len_kv=2,
    )
    paged_graph.validate().build_operation_graph()
    paged_graph.create_execution_plans([fe.heur_mode.A])
    with pytest.raises(fe.cudnnGraphNotSupportedError, match="paged_attention"):
        paged_graph.check_support()
    score_sum_exp = graph.tensor(
        name="score_sum_exp", dim=(1, 1, 2, 1), data_type=fe.DataType.FLOAT
    )
    score_max = graph.tensor(
        name="score_max", dim=(1, 1, 2, 1), data_type=fe.DataType.FLOAT
    )
    graph.sdpa(
        q,
        k,
        v,
        generate_stats=False,
        score_sum_exp=score_sum_exp,
        score_max=score_max,
    )
    assert graph.nodes[-1].outputs["score_sum_exp"] is score_sum_exp
    assert graph.nodes[-1].outputs["score_max"] is score_max
    with pytest.raises(ValueError, match="score_sum_exp dimensions"):
        graph.sdpa(q, k, v, generate_stats=False, score_sum_exp=aux)
    bad_score_dtype = graph.tensor(
        name="bad_score_dtype", dim=(1, 1, 2, 1), data_type=fe.DataType.HALF
    )
    with pytest.raises(ValueError, match="score_max data type"):
        graph.sdpa(q, k, v, generate_stats=False, score_max=bad_score_dtype)
    with pytest.raises(ValueError, match="sink_token dimensions"):
        graph.sdpa(q, k, v, generate_stats=False, sink_token=aux)


def test_mudnn_frontend_python_graph_api_sdpa_fp8_public_names():
    import cudnn_frontend as fe

    assert fe.DataType.FP8_E4M3 == 12
    assert fe.DataType.FP8_E5M2 == 13
    assert fe.DataType.FAST_FLOAT_FOR_FP8 == 14
    assert fe.DataType.FP8_E8M0 == 15
    assert fe.DataType.FP4_E2M1 == 16
    assert fe.DataType.INT4 == 17
    assert fe.DataType.UINT4 == 18
    assert fe.DataType.UINT32 == 19
    assert fe.DataType.COMPLEX_FP32 == 20
    assert fe.DataType.COMPLEX_FP64 == 21
    assert fe.CUDNN_DATA_FP8_E4M3 is fe.DataType.FP8_E4M3
    assert fe.CUDNN_DATA_FP8_E5M2 is fe.DataType.FP8_E5M2
    assert fe.CUDNN_DATA_FAST_FLOAT_FOR_FP8 is fe.DataType.FAST_FLOAT_FOR_FP8
    assert fe.CUDNN_DATA_FP8_E8M0 is fe.DataType.FP8_E8M0
    assert fe.CUDNN_DATA_FP4_E2M1 is fe.DataType.FP4_E2M1
    assert fe.CUDNN_DATA_INT4 is fe.DataType.INT4
    assert fe.CUDNN_DATA_UINT4 is fe.DataType.UINT4
    assert fe.CUDNN_DATA_UINT32 is fe.DataType.UINT32
    assert fe.CUDNN_DATA_COMPLEX_FP32 is fe.DataType.COMPLEX_FP32
    assert fe.CUDNN_DATA_COMPLEX_FP64 is fe.DataType.COMPLEX_FP64

    graph = fe.pygraph(io_data_type=fe.DataType.FP8_E4M3)
    q = graph.tensor(name="q", dim=(1, 1, 2, 4), data_type=fe.DataType.FP8_E4M3)
    k = graph.tensor(name="k", dim=(1, 1, 2, 4), data_type=fe.DataType.FP8_E4M3)
    v = graph.tensor(name="v", dim=(1, 1, 2, 4), data_type=fe.DataType.FP8_E4M3)
    scale = graph.tensor(name="scale", dim=(1,), data_type=fe.DataType.FLOAT)

    for keyword in (
        "descale_q",
        "descale_k",
        "descale_v",
        "descale_s",
        "scale_s",
        "scale_o",
    ):
        with pytest.raises(NotImplementedError, match=keyword):
            graph.sdpa(q, k, v, generate_stats=False, **{keyword: scale})


def test_mudnn_frontend_python_graph_api_sdpa_mxfp8_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.DataType.FP8_E4M3,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=(1, 2, 4, 8), data_type=fe.DataType.FP8_E4M3)
    k = graph.tensor(name="k", dim=(1, 2, 4, 8), data_type=fe.DataType.FP8_E4M3)
    v = graph.tensor(name="v", dim=(1, 2, 4, 16), data_type=fe.DataType.FP8_E4M3)
    descale_q = graph.tensor(name="descale_q", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_k = graph.tensor(name="descale_k", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_v = graph.tensor(name="descale_v", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_s = graph.tensor(name="descale_s", dim=(1,), data_type=fe.DataType.FLOAT)
    scale_s = graph.tensor(name="scale_s", dim=(1,), data_type=fe.DataType.FLOAT)
    scale_o = graph.tensor(name="scale_o", dim=(1,), data_type=fe.DataType.FP8_E8M0)

    o, stats, amax_o = graph.sdpa_mxfp8(
        q,
        k,
        v,
        descale_q=descale_q,
        descale_k=descale_k,
        descale_v=descale_v,
        descale_s=descale_s,
        scale_s=scale_s,
        scale_o=scale_o,
        name="mxfp8_attention",
        attn_scale=0.25,
        generate_stats=False,
        compute_data_type=fe.DataType.FAST_FLOAT_FOR_FP8,
    )

    assert o.dims == (1, 2, 4, 16)
    assert stats is None
    assert amax_o.dims == (1,)
    assert amax_o.data_type == fe.DataType.FLOAT
    node = graph.nodes[-1]
    assert node.op_type == "sdpa"
    assert node.name == "mxfp8_attention"
    assert node.attributes["mxfp8"] is True
    assert node.attributes["compute_data_type"] == fe.DataType.FAST_FLOAT_FOR_FP8
    assert node.inputs["descale_q"] is descale_q
    assert node.inputs["descale_k"] is descale_k
    assert node.inputs["descale_v"] is descale_v
    assert node.inputs["descale_s"] is descale_s
    assert node.inputs["scale_s"] is scale_s
    assert node.inputs["scale_o"] is scale_o

    restored = fe.pygraph().deserialize(graph.serialize())
    restored_node = restored.nodes[-1]
    assert restored_node.attributes["mxfp8"] is True
    assert restored_node.inputs["scale_o"].data_type == fe.DataType.FP8_E8M0
    assert restored_node.outputs["o"].dims == (1, 2, 4, 16)
    assert restored_node.outputs["amax_o"].data_type == fe.DataType.FLOAT

    lower_graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    lq = lower_graph.tensor(name="q", dim=(1, 2, 4, 8), data_type=fe.DataType.FLOAT)
    lk = lower_graph.tensor(name="k", dim=(1, 2, 4, 8), data_type=fe.DataType.FLOAT)
    lv = lower_graph.tensor(name="v", dim=(1, 2, 4, 16), data_type=fe.DataType.FLOAT)
    lower_scale = lower_graph.tensor(
        name="scale", dim=(1,), data_type=fe.DataType.FLOAT
    )
    lower_graph.sdpa_mxfp8(
        lq,
        lk,
        lv,
        descale_q=lower_scale,
        descale_k=lower_scale,
        descale_v=lower_scale,
        descale_s=lower_scale,
        scale_s=lower_scale,
        scale_o=lower_scale,
        generate_stats=False,
    )
    with pytest.raises(NotImplementedError, match="sdpa_mxfp8 native lowering"):
        lower_graph.validate().build_operation_graph()


def test_mudnn_frontend_python_graph_api_sdpa_mxfp8_old_cudnn_version_unsupported(
    monkeypatch,
):
    import mudnn
    import cudnn_frontend as fe

    monkeypatch.setattr(mudnn, "backend_version_string", lambda: "9.1.0")
    graph = fe.pygraph(
        io_data_type=fe.DataType.FP8_E4M3,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=(1, 1, 2, 8), data_type=fe.DataType.FP8_E4M3)
    k = graph.tensor(name="k", dim=(1, 1, 2, 8), data_type=fe.DataType.FP8_E4M3)
    v = graph.tensor(name="v", dim=(1, 1, 2, 8), data_type=fe.DataType.FP8_E4M3)
    scale = graph.tensor(name="scale", dim=(1,), data_type=fe.DataType.FP8_E8M0)

    graph.sdpa_mxfp8(
        q,
        k,
        v,
        descale_q=scale,
        descale_k=scale,
        descale_v=scale,
        generate_stats=False,
    )
    with pytest.raises(fe.cudnnGraphNotSupportedError, match="9.21.0"):
        graph.validate()


def test_mudnn_frontend_python_graph_api_sdpa_fp8_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.DataType.FP8_E4M3,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=(1, 2, 4, 8), data_type=fe.DataType.FP8_E4M3)
    k = graph.tensor(name="k", dim=(1, 2, 4, 8), data_type=fe.DataType.FP8_E4M3)
    v = graph.tensor(name="v", dim=(1, 2, 4, 16), data_type=fe.DataType.FP8_E4M3)
    descale_q = graph.tensor(name="descale_q", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_k = graph.tensor(name="descale_k", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_v = graph.tensor(name="descale_v", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_s = graph.tensor(name="descale_s", dim=(1,), data_type=fe.DataType.FLOAT)
    scale_s = graph.tensor(name="scale_s", dim=(1,), data_type=fe.DataType.FLOAT)
    scale_o = graph.tensor(name="scale_o", dim=(1,), data_type=fe.DataType.FLOAT)

    o, stats, amax_s, amax_o = graph.sdpa_fp8(
        q,
        k,
        v,
        descale_q=descale_q,
        descale_k=descale_k,
        descale_v=descale_v,
        descale_s=descale_s,
        scale_s=scale_s,
        scale_o=scale_o,
        name="fp8_attention",
        generate_stats=True,
        compute_data_type=fe.DataType.FAST_FLOAT_FOR_FP8,
        paged_attention_max_seq_len_kv=4,
    )

    assert o.dims == (1, 2, 4, 16)
    assert stats.dims == (1, 2, 4, 1)
    assert amax_s.dims == (1,)
    assert amax_o.dims == (1,)
    assert amax_s.data_type == fe.DataType.FLOAT
    assert amax_o.data_type == fe.DataType.FLOAT
    node = graph.nodes[-1]
    assert node.op_type == "sdpa"
    assert node.name == "fp8_attention"
    assert node.attributes["fp8"] is True
    assert node.attributes["mxfp8"] is True
    assert node.attributes["paged_attention_max_seq_len_kv"] == 4
    assert node.outputs["amax_s"] is amax_s
    assert node.outputs["amax_o"] is amax_o

    restored = fe.pygraph().deserialize(graph.serialize())
    restored_node = restored.nodes[-1]
    assert restored_node.attributes["fp8"] is True
    assert restored_node.outputs["amax_s"].data_type == fe.DataType.FLOAT
    assert restored_node.outputs["amax_o"].dims == (1,)

    lower_graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    lq = lower_graph.tensor(name="q", dim=(1, 2, 4, 8), data_type=fe.DataType.FLOAT)
    lk = lower_graph.tensor(name="k", dim=(1, 2, 4, 8), data_type=fe.DataType.FLOAT)
    lv = lower_graph.tensor(name="v", dim=(1, 2, 4, 16), data_type=fe.DataType.FLOAT)
    lower_scale = lower_graph.tensor(
        name="scale", dim=(1,), data_type=fe.DataType.FLOAT
    )
    lower_graph.sdpa_fp8(
        lq,
        lk,
        lv,
        descale_q=lower_scale,
        descale_k=lower_scale,
        descale_v=lower_scale,
        descale_s=lower_scale,
        scale_s=lower_scale,
        scale_o=lower_scale,
        generate_stats=False,
    )
    with pytest.raises(NotImplementedError, match="sdpa_fp8 native lowering"):
        lower_graph.validate().build_operation_graph()


def test_mudnn_frontend_python_graph_api_sdpa_bias_mask_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    q = graph.tensor(name="q", dim=(1, 1, 2, 2), data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=(1, 1, 3, 2), data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=(1, 1, 3, 2), data_type=fe.DataType.FLOAT)
    bias = graph.tensor(name="bias", dim=(1, 1, 2, 3), data_type=fe.DataType.FLOAT)
    additive_mask = graph.tensor(
        name="additive_mask", dim=(1, 1, 2, 3), data_type=fe.DataType.FLOAT
    )
    mask = graph.tensor(name="mask", dim=(1, 1, 2, 3), data_type=fe.DataType.BOOLEAN)
    pad_mask = graph.tensor(name="pad_mask", dim=(1, 3), data_type=fe.DataType.BOOLEAN)

    o_bias, _ = graph.sdpa(q, k, v, bias=bias, generate_stats=False)
    assert graph.nodes[-1].inputs["bias"] is bias
    assert o_bias.dims == (1, 1, 2, 2)
    o_mask, _ = graph.sdpa(q, k, v, mask=mask, generate_stats=False)
    assert graph.nodes[-1].inputs["mask"] is mask
    assert o_mask.dims == (1, 1, 2, 2)
    graph.sdpa(q, k, v, mask=pad_mask, generate_stats=False)
    assert graph.nodes[-1].attributes["use_padding_mask"] is True
    fused, _ = graph.sdpa(q, k, v, bias=bias, mask=additive_mask, generate_stats=False)
    assert fused.dims == (1, 1, 2, 2)
    assert graph.nodes[-2].op_type == "pointwise"
    assert graph.nodes[-2].inputs == {"x": bias, "b": additive_mask}
    assert graph.nodes[-1].inputs["bias"] is graph.nodes[-2].outputs["y"]
    bool_fused, _ = graph.sdpa(q, k, v, bias=bias, mask=mask, generate_stats=False)
    assert bool_fused.dims == (1, 1, 2, 2)
    assert graph.nodes[-1].op_type == "sdpa"
    assert graph.nodes[-1].inputs["bias"] is bias
    assert graph.nodes[-1].inputs["mask"] is mask

    train_graph = fe.pygraph(io_data_type=fe.DataType.HALF)
    train_q = train_graph.tensor(
        name="q", dim=(2, 4, 8, 16), data_type=fe.DataType.HALF
    )
    train_k = train_graph.tensor(
        name="k", dim=(2, 4, 8, 16), data_type=fe.DataType.HALF
    )
    train_v = train_graph.tensor(
        name="v", dim=(2, 4, 8, 32), data_type=fe.DataType.HALF
    )
    train_o, train_stats = train_graph.scaled_dot_product_attention(
        train_q,
        train_k,
        train_v,
        name="train_attention",
        generate_stats=True,
    )
    assert train_o.dims == (2, 4, 8, 32)
    assert train_stats is not None
    assert train_stats.dims == (2, 4, 8, 1)
    assert train_graph.nodes[-1].attributes["is_inference"] is False
    assert train_graph.nodes[-1].attributes["generate_stats"] is True
    train_graph.validate()
    train_graph.build_operation_graph().create_execution_plans([fe.HeurMode.INSTANT])
    assert train_graph._native.operation_graph.ptr.value is not None
    with pytest.raises(ValueError, match="generate_stats and is_inference disagree"):
        graph.sdpa(q, k, v, generate_stats=True, is_inference=True)
    with pytest.raises(ValueError, match="requires generate_stats or is_inference"):
        graph.sdpa(q, k, v)
    with pytest.raises(ValueError, match="causal_mask and use_causal_mask disagree"):
        graph.sdpa(
            q, k, v, causal_mask=True, use_causal_mask=False, generate_stats=False
        )


def test_mudnn_frontend_python_graph_api_sdpa_seq_len_symbolic_and_lowering():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=(2, 1, 3, 4), data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=(2, 1, 5, 4), data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=(2, 1, 5, 4), data_type=fe.DataType.FLOAT)
    seq_q = graph.tensor(name="seq_q", dim=(2, 1, 1, 1), data_type=fe.DataType.INT32)
    seq_kv = graph.tensor(name="seq_kv", dim=(2, 1, 1, 1), data_type=fe.DataType.INT32)

    o, stats = graph.sdpa(
        q,
        k,
        v,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
        use_padding_mask=True,
        generate_stats=False,
        attn_scale=0.5,
        name="ragged_attention",
    )

    assert stats is None
    assert o.dims == (2, 1, 3, 4)
    o.set_output(True)
    node = graph.nodes[-1]
    assert node.inputs["seq_len_q"] is seq_q
    assert node.inputs["seq_len_kv"] is seq_kv
    assert node.attributes["use_padding_mask"] is True

    restored = fe.pygraph().deserialize(graph.serialize())
    restored_node = restored.nodes[-1]
    assert restored_node.inputs["seq_len_q"].name == "seq_q"
    assert restored_node.inputs["seq_len_kv"].name == "seq_kv"
    assert restored_node.attributes["use_padding_mask"] is True
    restored.validate().build_operation_graph()

    bad_shape = graph.tensor(name="bad_shape", dim=(3,), data_type=fe.DataType.INT32)
    with pytest.raises(ValueError, match="seq_len_q dimensions"):
        graph.sdpa(q, k, v, seq_len_q=bad_shape, generate_stats=False)

    bad_dtype = graph.tensor(name="bad_dtype", dim=(2,), data_type=fe.DataType.FLOAT)
    with pytest.raises(ValueError, match="seq_len_kv data type"):
        graph.sdpa(q, k, v, seq_len_kv=bad_dtype, generate_stats=False)


def test_mudnn_frontend_python_graph_api_sdpa_backward_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    q = graph.tensor(name="q", dim=(1, 1, 2, 2), data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=(1, 1, 3, 2), data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=(1, 1, 3, 4), data_type=fe.DataType.FLOAT)
    o = graph.tensor(name="o", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT)
    d_o = graph.tensor(name="do", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT)
    stats = graph.tensor(name="stats", dim=(1, 1, 2, 1), data_type=fe.DataType.FLOAT)
    bias = graph.tensor(name="bias", dim=(1, 1, 2, 3), data_type=fe.DataType.FLOAT)
    d_bias = graph.tensor(name="dbias", dim=(1, 1, 2, 3), data_type=fe.DataType.FLOAT)
    mask = graph.tensor(name="mask", dim=(1, 1, 2, 3), data_type=fe.DataType.BOOLEAN)
    seq_q = graph.tensor(name="seq_q", dim=(1,), data_type=fe.DataType.INT32)
    seq_kv = graph.tensor(name="seq_kv", dim=(1, 1, 1, 1), data_type=fe.DataType.INT64)

    d_q, d_k, d_v = graph.sdpa_backward(
        q,
        k,
        v,
        o,
        d_o,
        stats,
        name="attention_bwd",
        attn_scale=0.5,
        bias=bias,
        d_bias=d_bias,
        mask=mask,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
        use_causal_mask=True,
        deterministic=True,
        compute_data_type=fe.DataType.FLOAT,
    )

    assert d_q.dims == q.dims
    assert d_k.dims == k.dims
    assert d_v.dims == v.dims
    assert graph.nodes[-1].op_type == "sdpa_bwd"
    assert graph.nodes[-1].inputs["stats"] is stats
    assert graph.nodes[-1].inputs["seq_len_q"] is seq_q
    assert graph.nodes[-1].inputs["seq_len_kv"] is seq_kv
    assert graph.nodes[-1].outputs["d_bias"] is d_bias
    assert graph.nodes[-1].attributes["attn_scale"] == 0.5
    assert graph.nodes[-1].attributes["causal_mask"] is True
    assert graph.nodes[-1].attributes["deterministic"] is True
    assert graph.nodes[-1].attributes["compute_data_type"] == fe.DataType.FLOAT
    graph.validate()
    with pytest.raises(NotImplementedError, match="bias/mask"):
        graph.build_operation_graph()

    restored = fe.pygraph().deserialize(graph.serialize())
    assert restored.nodes[-1].inputs["seq_len_q"].name == "seq_q"
    assert restored.nodes[-1].inputs["seq_len_kv"].name == "seq_kv"

    causal_graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    causal_q = causal_graph.tensor(
        name="q", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    causal_k = causal_graph.tensor(
        name="k", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    causal_v = causal_graph.tensor(
        name="v", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    causal_o = causal_graph.tensor(
        name="o", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    causal_do = causal_graph.tensor(
        name="do", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    causal_stats = causal_graph.tensor(
        name="stats", dim=(1, 1, 2, 1), data_type=fe.DataType.FLOAT
    )
    causal_graph.sdpa_backward(
        causal_q,
        causal_k,
        causal_v,
        causal_o,
        causal_do,
        causal_stats,
        attn_scale=0.5,
        use_causal_mask=True,
        deterministic=True,
    )
    assert causal_graph.nodes[-1].attributes["causal_mask"] is True
    causal_graph.validate().build_operation_graph()

    ragged_graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    ragged_q = ragged_graph.tensor(
        name="q", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    ragged_k = ragged_graph.tensor(
        name="k", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    ragged_v = ragged_graph.tensor(
        name="v", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    ragged_o = ragged_graph.tensor(
        name="o", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    ragged_do = ragged_graph.tensor(
        name="do", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF
    )
    ragged_stats = ragged_graph.tensor(
        name="stats", dim=(1, 1, 2, 1), data_type=fe.DataType.FLOAT
    )
    ragged_seq = ragged_graph.tensor(name="seq", dim=(1,), data_type=fe.DataType.INT32)
    ragged_graph.sdpa_backward(
        ragged_q,
        ragged_k,
        ragged_v,
        ragged_o,
        ragged_do,
        ragged_stats,
        seq_len_q=ragged_seq,
    )
    ragged_graph.validate()
    with pytest.raises(NotImplementedError, match="sequence-length"):
        ragged_graph.build_operation_graph()

    with pytest.raises(ValueError, match="stats data type must be FLOAT"):
        bad_graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        bad_q = bad_graph.tensor(
            name="q", dim=(1, 1, 2, 2), data_type=fe.DataType.FLOAT
        )
        bad_k = bad_graph.tensor(
            name="k", dim=(1, 1, 3, 2), data_type=fe.DataType.FLOAT
        )
        bad_v = bad_graph.tensor(
            name="v", dim=(1, 1, 3, 4), data_type=fe.DataType.FLOAT
        )
        bad_o = bad_graph.tensor(
            name="o", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT
        )
        bad_do = bad_graph.tensor(
            name="do", dim=(1, 1, 2, 4), data_type=fe.DataType.FLOAT
        )
        bad_stats = bad_graph.tensor(
            name="stats", dim=(1, 1, 2, 1), data_type=fe.DataType.HALF
        )
        bad_graph.sdpa_backward(bad_q, bad_k, bad_v, bad_o, bad_do, bad_stats)


def test_mudnn_frontend_python_graph_api_sdpa_backward_keyword_aliases():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    q = graph.tensor(name="q", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF)
    k = graph.tensor(name="k", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF)
    v = graph.tensor(name="v", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF)
    o = graph.tensor(name="o", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF)
    d_o = graph.tensor(name="dO", dim=(1, 1, 2, 64), data_type=fe.DataType.HALF)
    stats = graph.tensor(name="stats", dim=(1, 1, 2, 1), data_type=fe.DataType.FLOAT)
    d_bias = graph.tensor(name="dBias", dim=(1, 1, 2, 2), data_type=fe.DataType.HALF)
    seq = graph.tensor(name="seq", dim=(1,), data_type=fe.DataType.INT32)

    d_q, d_k, d_v = graph.sdpa_backward(
        q,
        k,
        v,
        o,
        dO=d_o,
        stats=stats,
        dBias=d_bias,
        name="attention_bwd_alias",
        diagonal_alignment=fe.diagonal_alignment.TOP_LEFT,
        diagonal_band_right_bound=0,
        use_deterministic_algorithm=True,
        use_alibi_mask=False,
        dropout=0.0,
    )

    assert d_q.dims == q.dims
    assert d_k.dims == k.dims
    assert d_v.dims == v.dims
    assert graph.nodes[-1].inputs["d_o"] is d_o
    assert graph.nodes[-1].outputs["d_bias"] is d_bias
    assert graph.nodes[-1].attributes["causal_mask"] is True
    assert graph.nodes[-1].attributes["deterministic"] is True

    with pytest.raises(ValueError, match="requires d_o or dO"):
        graph.sdpa_backward(q, k, v, o, stats=stats)
    with pytest.raises(ValueError, match="d_o and dO disagree"):
        graph.sdpa_backward(q, k, v, o, d_o, stats, dO=o)
    with pytest.raises(ValueError, match="d_bias and dBias disagree"):
        graph.sdpa_backward(q, k, v, o, d_o, stats, d_bias=d_bias, dBias=o)
    with pytest.raises(ValueError, match="use_deterministic_algorithm disagree"):
        graph.sdpa_backward(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            deterministic=True,
            use_deterministic_algorithm=False,
        )
    with pytest.raises(NotImplementedError, match="top-left causal"):
        graph.sdpa_backward(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            diagonal_alignment=fe.diagonal_alignment.BOTTOM_RIGHT,
            diagonal_band_right_bound=0,
        )
    graph.sdpa_backward(q, k, v, o, d_o, stats, seq_len_kv=seq)
    assert graph.nodes[-1].inputs["seq_len_kv"] is seq
    graph.sdpa_backward(q, k, v, o, d_o, stats, cu_seq_len_kv=seq)
    assert graph.nodes[-1].inputs["seq_len_kv"] is seq
    with pytest.raises(ValueError, match="seq_len_kv and cu_seq_len_kv disagree"):
        graph.sdpa_backward(
            q,
            k,
            v,
            o,
            d_o,
            stats,
            seq_len_kv=stats,
            cu_seq_len_kv=o,
        )
    with pytest.raises(NotImplementedError, match="rng_dump"):
        graph.sdpa_backward(q, k, v, o, d_o, stats, rng_dump=stats)
    with pytest.raises(NotImplementedError, match="dropout"):
        graph.sdpa_backward(q, k, v, o, d_o, stats, dropout=0.1)
    with pytest.raises(NotImplementedError, match="sliding_window_length"):
        graph.sdpa_backward(q, k, v, o, d_o, stats, sliding_window_length=2)
    graph.sdpa_backward(
        q,
        k,
        v,
        o,
        d_o,
        stats,
        score_mod=lambda _graph, score: score,
        score_mod_bprop=lambda _graph, d_p: d_p,
    )
    assert graph.nodes[-1].attributes["score_mod"] is True
    assert graph.nodes[-1].attributes["score_mod_bprop"] is True
    assert graph.nodes[-1].attributes["score_mod_input_uid"] is not None
    assert graph.nodes[-1].attributes["score_mod_bprop_input_uid"] is not None
    with pytest.raises(NotImplementedError, match="sink_token"):
        graph.sdpa_backward(q, k, v, o, d_o, stats, sink_token=stats)
    with pytest.raises(NotImplementedError, match="dSink_token"):
        graph.sdpa_backward(q, k, v, o, d_o, stats, dSink_token=stats)


def test_mudnn_frontend_python_graph_api_sdpa_backward_fp8_public_names():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FP8_E4M3)
    q = graph.tensor(name="q", dim=(1, 1, 2, 64), data_type=fe.DataType.FP8_E4M3)
    k = graph.tensor(name="k", dim=(1, 1, 2, 64), data_type=fe.DataType.FP8_E4M3)
    v = graph.tensor(name="v", dim=(1, 1, 2, 64), data_type=fe.DataType.FP8_E4M3)
    o = graph.tensor(name="o", dim=(1, 1, 2, 64), data_type=fe.DataType.FP8_E4M3)
    d_o = graph.tensor(name="dO", dim=(1, 1, 2, 64), data_type=fe.DataType.FP8_E4M3)
    stats = graph.tensor(name="stats", dim=(1, 1, 2, 1), data_type=fe.DataType.FLOAT)
    scale = graph.tensor(name="scale", dim=(1,), data_type=fe.DataType.FLOAT)

    for keyword in (
        "descale_q",
        "descale_k",
        "descale_v",
        "descale_o",
        "descale_dO",
        "descale_s",
        "descale_dP",
        "scale_s",
        "scale_dQ",
        "scale_dK",
        "scale_dV",
        "scale_dP",
    ):
        with pytest.raises(NotImplementedError, match=keyword):
            graph.sdpa_backward(q, k, v, o, dO=d_o, stats=stats, **{keyword: scale})

    descale_q = graph.tensor(name="descale_q", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_k = graph.tensor(name="descale_k", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_v = graph.tensor(name="descale_v", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_do = graph.tensor(name="descale_dO", dim=(1,), data_type=fe.DataType.FLOAT)
    scale_dq = graph.tensor(name="scale_dQ", dim=(1,), data_type=fe.DataType.FLOAT)
    scale_dk = graph.tensor(name="scale_dK", dim=(1,), data_type=fe.DataType.FLOAT)
    scale_dv = graph.tensor(name="scale_dV", dim=(1,), data_type=fe.DataType.FLOAT)
    scale_dp = graph.tensor(name="scale_dP", dim=(1,), data_type=fe.DataType.FLOAT)
    scale_s = graph.tensor(name="scale_s", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_s = graph.tensor(name="descale_s", dim=(1,), data_type=fe.DataType.FLOAT)
    descale_dp = graph.tensor(name="descale_dP", dim=(1,), data_type=fe.DataType.FLOAT)

    d_q, d_k, d_v, amax_dq, amax_dk, amax_dv, amax_dp = graph.sdpa_fp8_backward(
        q=q,
        k=k,
        v=v,
        o=o,
        dO=d_o,
        stats=stats,
        descale_q=descale_q,
        descale_k=descale_k,
        descale_v=descale_v,
        descale_dO=descale_do,
        descale_s=descale_s,
        descale_dP=descale_dp,
        scale_s=scale_s,
        scale_dQ=scale_dq,
        scale_dK=scale_dk,
        scale_dV=scale_dv,
        scale_dP=scale_dp,
        compute_data_type=fe.DataType.FAST_FLOAT_FOR_FP8,
    )
    assert d_q.dims == q.dims
    assert d_k.dims == k.dims
    assert d_v.dims == v.dims
    assert amax_dq.data_type == fe.DataType.FLOAT
    assert amax_dk.data_type == fe.DataType.FLOAT
    assert amax_dv.data_type == fe.DataType.FLOAT
    assert amax_dp.data_type == fe.DataType.FLOAT
    fp8_node = graph.nodes[-1]
    assert fp8_node.attributes["fp8_bwd"] is True
    assert fp8_node.attributes["compute_data_type"] == fe.DataType.FAST_FLOAT_FOR_FP8
    assert fp8_node.inputs["scale_dQ"] is scale_dq
    assert fp8_node.outputs["amax_dP"].data_type == fe.DataType.FLOAT

    restored = fe.pygraph().deserialize(graph.serialize())
    restored_node = restored.nodes[-1]
    assert restored_node.attributes["fp8_bwd"] is True
    assert restored_node.outputs["amax_dP"].data_type == fe.DataType.FLOAT

    lower_graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    lq = lower_graph.tensor(name="q", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    lk = lower_graph.tensor(name="k", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    lv = lower_graph.tensor(name="v", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    lo = lower_graph.tensor(name="o", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    ldo = lower_graph.tensor(name="dO", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    lstats = lower_graph.tensor(
        name="stats", dim=(1, 1, 2, 1), data_type=fe.DataType.FLOAT
    )
    lscale = lower_graph.tensor(name="scale", dim=(1,), data_type=fe.DataType.FLOAT)
    lower_graph.sdpa_fp8_backward(
        lq,
        lk,
        lv,
        lo,
        dO=ldo,
        stats=lstats,
        descale_q=lscale,
        descale_k=lscale,
        descale_v=lscale,
        descale_dO=lscale,
        descale_s=lscale,
        descale_dP=lscale,
        scale_s=lscale,
        scale_dQ=lscale,
        scale_dK=lscale,
        scale_dV=lscale,
        scale_dP=lscale,
    )
    with pytest.raises(NotImplementedError, match="sdpa_fp8_backward native lowering"):
        lower_graph.validate().build_operation_graph()

    mxfp8_graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    mq = mxfp8_graph.tensor(name="q", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    mk = mxfp8_graph.tensor(name="k", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    mv = mxfp8_graph.tensor(name="v", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    mo = mxfp8_graph.tensor(name="o", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    mdo = mxfp8_graph.tensor(name="dO", dim=(1, 1, 2, 64), data_type=fe.DataType.FLOAT)
    mstats = mxfp8_graph.tensor(
        name="stats", dim=(1, 1, 2, 1), data_type=fe.DataType.FLOAT
    )
    mscale = mxfp8_graph.tensor(name="scale", dim=(1,), data_type=fe.DataType.FLOAT)
    d_q, d_k, d_v, amax_dq, amax_dk, amax_dv = mxfp8_graph.sdpa_mxfp8_backward(
        q=mq,
        k=mk,
        v=mv,
        o_f16=mo,
        dO=mdo,
        stats=mstats,
        descale_q=mscale,
        descale_k=mscale,
        descale_v=mscale,
        descale_dO=mscale,
    )
    assert d_q.dims == mq.dims
    assert d_k.dims == mk.dims
    assert d_v.dims == mv.dims
    assert amax_dq.data_type == fe.DataType.FLOAT
    assert amax_dk.data_type == fe.DataType.FLOAT
    assert amax_dv.data_type == fe.DataType.FLOAT
    assert mxfp8_graph.nodes[-1].attributes["mxfp8_bwd"] is True
    with pytest.raises(
        NotImplementedError, match="sdpa_mxfp8_backward native lowering"
    ):
        mxfp8_graph.validate().build_operation_graph()


def test_mudnn_frontend_python_graph_api_execute_sdpa():
    import cudnn_frontend as fe

    q_np = np.array(
        [[[[0.2, -0.5], [1.0, 0.25]]]],
        dtype=np.float32,
    )
    k_np = np.array(
        [[[[0.0, 0.75], [-0.25, 0.5], [1.0, -1.0]]]],
        dtype=np.float32,
    )
    v_np = np.array(
        [[[[1.0, -1.0], [0.5, 2.0], [-0.25, 0.75]]]],
        dtype=np.float32,
    )
    scale = 0.5
    expected = _sdpa_reference(q_np, k_np, v_np, scale)

    tensors = []
    try:
        q = device_memory.create_tensor(
            q_np.__array_interface__["data"][0],
            q_np.nbytes,
            q_np.shape,
            "float32",
        )
        k = device_memory.create_tensor(
            k_np.__array_interface__["data"][0],
            k_np.nbytes,
            k_np.shape,
            "float32",
        )
        v = device_memory.create_tensor(
            v_np.__array_interface__["data"][0],
            v_np.nbytes,
            v_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [q, k, v, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q_desc = graph.tensor(name="q", dim=q_np.shape, data_type=fe.DataType.FLOAT)
        k_desc = graph.tensor(name="k", dim=k_np.shape, data_type=fe.DataType.FLOAT)
        v_desc = graph.tensor(name="v", dim=v_np.shape, data_type=fe.DataType.FLOAT)
        o_desc, stats = graph.sdpa(
            q_desc,
            k_desc,
            v_desc,
            name="attention",
            attn_scale=scale,
            is_inference=True,
        )
        assert stats is None

        graph.execute(
            {
                q_desc: device_memory.get_tensor_addr(q)[0],
                k_desc: device_memory.get_tensor_addr(k)[0],
                v_desc: device_memory.get_tensor_addr(v)[0],
                o_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=5.0e-4)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_sdpa_score_outputs_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    batch, seq_q, seq_kv, heads_q, heads_kv, dim = 1, 3, 3, 2, 1, 4
    q_shape = (batch, heads_q, seq_q, dim)
    kv_shape = (batch, heads_kv, seq_kv, dim)
    q_stride = (heads_q * seq_q * dim, dim, heads_q * dim, 1)
    kv_stride = (heads_kv * seq_kv * dim, dim, heads_kv * dim, 1)
    score_shape = (batch, heads_q, seq_q, 1)
    score_stride = (heads_q * seq_q, seq_q, 1, 1)

    q_value = torch.randn(batch, seq_q, heads_q, dim, dtype=torch.float32)
    k_value = torch.randn(batch, seq_kv, heads_kv, dim, dtype=torch.float32)
    v_value = torch.randn(batch, seq_kv, heads_kv, dim, dtype=torch.float32)
    o_value = torch.empty(batch, seq_q, heads_q, dim, dtype=torch.float32)
    score_max_value = torch.empty(batch, heads_q, seq_q, dtype=torch.float32)
    score_sum_exp_value = torch.empty(batch, heads_q, seq_q, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(
        name="q", dim=q_shape, stride=q_stride, data_type=fe.DataType.FLOAT
    )
    k = graph.tensor(
        name="k", dim=kv_shape, stride=kv_stride, data_type=fe.DataType.FLOAT
    )
    v = graph.tensor(
        name="v", dim=kv_shape, stride=kv_stride, data_type=fe.DataType.FLOAT
    )
    score_max = graph.tensor(
        name="max",
        dim=score_shape,
        stride=score_stride,
        data_type=fe.DataType.FLOAT,
    )
    score_sum_exp = graph.tensor(
        name="sum_exp",
        dim=score_shape,
        stride=score_stride,
        data_type=fe.DataType.FLOAT,
    )
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=0.5,
        use_causal_mask=True,
        generate_stats=False,
        score_max=score_max,
        score_sum_exp=score_sum_exp,
    )
    assert stats is None
    o.set_output(True).set_dim(q_shape).set_stride(q_stride)

    graph.validate().build_operation_graph()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            o: o_value,
            score_max: score_max_value,
            score_sum_exp: score_sum_exp_value,
        }
    )

    q_view = torch.as_strided(q_value.reshape(-1), q_shape, q_stride)
    k_view = torch.as_strided(k_value.reshape(-1), kv_shape, kv_stride)
    v_view = torch.as_strided(v_value.reshape(-1), kv_shape, kv_stride)
    k_view = k_view.repeat_interleave(heads_q // heads_kv, dim=1)
    v_view = v_view.repeat_interleave(heads_q // heads_kv, dim=1)
    scores = torch.matmul(q_view, k_view.transpose(-2, -1)) * 0.5
    causal = torch.ones(seq_q, seq_kv, dtype=torch.bool).triu(1)
    scores = torch.where(causal, torch.full_like(scores, -torch.inf), scores)
    expected_max = scores.max(dim=-1, keepdim=True).values
    expected_sum_exp = torch.exp(scores - expected_max).sum(dim=-1, keepdim=True)
    expected_o = torch.matmul(torch.softmax(scores, dim=-1), v_view)

    torch.testing.assert_close(
        torch.as_strided(o_value.reshape(-1), q_shape, q_stride),
        expected_o,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        score_max_value,
        expected_max.squeeze(-1),
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        score_sum_exp_value,
        expected_sum_exp.squeeze(-1),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_mudnn_frontend_python_graph_api_execute_sdpa_seq_len_score_max_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.arange(2 * 1 * 3 * 2, dtype=torch.float32).reshape(2, 1, 3, 2)
    k_value = torch.arange(2 * 1 * 4 * 2, dtype=torch.float32).reshape(2, 1, 4, 2)
    v_value = torch.arange(2 * 1 * 4 * 2, dtype=torch.float32).reshape(2, 1, 4, 2)
    q_value = q_value / 11.0 - 0.25
    k_value = k_value / 13.0 - 0.5
    v_value = v_value / 17.0 - 0.75
    seq_len_q_value = torch.tensor([2, 0], dtype=torch.int32)
    seq_len_kv_value = torch.tensor([3, 2], dtype=torch.int32)
    o_value = torch.empty_like(q_value)
    score_max_value = torch.empty(2, 1, 3, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    seq_len_q = graph.tensor(
        name="seq_q", dim=seq_len_q_value.shape, data_type=fe.DataType.INT32
    )
    seq_len_kv = graph.tensor(
        name="seq_kv", dim=seq_len_kv_value.shape, data_type=fe.DataType.INT32
    )
    score_max = graph.tensor(name="max", dim=(2, 1, 3, 1), data_type=fe.DataType.FLOAT)
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=0.5,
        generate_stats=False,
        use_padding_mask=True,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        score_max=score_max,
    )
    assert stats is None
    o.set_output(True)
    graph.validate().build_operation_graph()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            seq_len_q: seq_len_q_value,
            seq_len_kv: seq_len_kv_value,
            o: o_value,
            score_max: score_max_value,
        }
    )

    scores = torch.matmul(q_value, k_value.transpose(-2, -1)) * 0.5
    q_index = torch.arange(q_value.shape[-2]).reshape(1, 1, -1, 1)
    k_index = torch.arange(k_value.shape[-2]).reshape(1, 1, 1, -1)
    valid_q = q_index < seq_len_q_value.reshape(-1, 1, 1, 1)
    valid_k = k_index < seq_len_kv_value.reshape(-1, 1, 1, 1)
    scores = torch.where(valid_q & valid_k, scores, torch.full_like(scores, -torch.inf))
    row_has_scores = torch.isfinite(scores).any(dim=-1, keepdim=True)
    scores_for_softmax = torch.where(row_has_scores, scores, torch.zeros_like(scores))
    expected_max = scores_for_softmax.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scores_for_softmax, dim=-1) * row_has_scores.to(
        dtype=scores.dtype
    )
    expected_o = torch.matmul(probs, v_value)

    torch.testing.assert_close(o_value, expected_o, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(
        score_max_value, expected_max.squeeze(-1), rtol=1.0e-5, atol=1.0e-5
    )


def test_mudnn_frontend_python_graph_api_execute_sdpa_sink_token_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.arange(1 * 2 * 3 * 2, dtype=torch.float32).reshape(1, 2, 3, 2)
    k_value = torch.arange(1 * 1 * 4 * 2, dtype=torch.float32).reshape(1, 1, 4, 2)
    v_value = torch.arange(1 * 1 * 4 * 2, dtype=torch.float32).reshape(1, 1, 4, 2)
    q_value = q_value / 7.0 - 0.5
    k_value = k_value / 11.0 - 0.25
    v_value = v_value / 13.0 - 0.75
    sink_value = torch.tensor([[[[0.25]], [[-0.5]]]], dtype=torch.float32)
    o_value = torch.empty(1, 2, 3, 2, dtype=torch.float32)
    score_max_value = torch.empty(1, 2, 3, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    sink = graph.tensor(name="sink", dim=sink_value.shape, data_type=fe.DataType.FLOAT)
    score_max = graph.tensor(name="max", dim=(1, 2, 3, 1), data_type=fe.DataType.FLOAT)
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=0.5,
        generate_stats=False,
        sink_token=sink,
        score_max=score_max,
    )
    assert stats is None
    o.set_output(True)
    graph.validate().build_operation_graph()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            sink: sink_value,
            o: o_value,
            score_max: score_max_value,
        }
    )

    k_expanded = k_value.repeat_interleave(2, dim=1)
    v_expanded = v_value.repeat_interleave(2, dim=1)
    scores = torch.matmul(q_value, k_expanded.transpose(-2, -1)) * 0.5
    extended_scores = torch.cat(
        [sink_value.expand(tuple(scores.shape[:-1]) + (1,)), scores], dim=-1
    )
    expected_max = extended_scores.max(dim=-1, keepdim=True).values
    probs = torch.softmax(extended_scores, dim=-1)[..., 1:]
    expected_o = torch.matmul(probs, v_expanded)

    torch.testing.assert_close(o_value, expected_o, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(
        score_max_value, expected_max.squeeze(-1), rtol=1.0e-5, atol=1.0e-5
    )


def test_mudnn_frontend_python_graph_api_execute_sdpa_alibi_cpu_fallback():
    import math
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.arange(1 * 3 * 3 * 2, dtype=torch.float32).reshape(1, 3, 3, 2)
    k_value = torch.arange(1 * 3 * 4 * 2, dtype=torch.float32).reshape(1, 3, 4, 2)
    v_value = torch.arange(1 * 3 * 4 * 2, dtype=torch.float32).reshape(1, 3, 4, 2)
    q_value = q_value / 17.0 - 0.5
    k_value = k_value / 19.0 - 0.25
    v_value = v_value / 23.0 - 0.75
    o_value = torch.empty(1, 3, 3, 2, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=0.5,
        generate_stats=False,
        use_alibi_mask=True,
    )
    assert stats is None
    o.set_output(True)
    graph.validate().build_operation_graph()
    graph.execute({q: q_value, k: k_value, v: v_value, o: o_value})

    scores = torch.matmul(q_value, k_value.transpose(-2, -1)) * 0.5
    distance = torch.arange(4, dtype=torch.float32) - torch.arange(
        3, dtype=torch.float32
    ).reshape(3, 1)
    n = 2 ** math.floor(math.log2(3))
    slopes = torch.pow(torch.tensor(2.0 ** (-8.0 / n)), torch.arange(1, 1 + n))
    extra = torch.pow(
        torch.tensor(2.0 ** (-4.0 / n)), torch.arange(1, 1 + 2 * (3 - n), 2)
    )
    slopes = torch.cat([slopes, extra]).reshape(1, 3, 1, 1)
    scores = scores + distance.reshape(1, 1, 3, 4) * slopes
    expected = torch.matmul(torch.softmax(scores, dim=-1), v_value)
    torch.testing.assert_close(o_value, expected, rtol=1.0e-5, atol=1.0e-5)


def test_mudnn_frontend_python_graph_api_execute_sdpa_block_mask_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.tensor([[[[0.25, -0.5], [0.75, 0.125]]]], dtype=torch.float32)
    k_value = torch.arange(1 * 1 * 129 * 2, dtype=torch.float32).reshape(1, 1, 129, 2)
    v_value = torch.arange(1 * 1 * 129 * 2, dtype=torch.float32).reshape(1, 1, 129, 2)
    k_value = k_value / 101.0 - 0.5
    v_value = v_value / 97.0 - 1.0
    block_mask_value = torch.tensor([[[[0x01]]]], dtype=torch.uint8)
    o_value = torch.empty(1, 1, 2, 2, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    block_mask = graph.tensor(
        name="block_mask", dim=block_mask_value.shape, data_type=fe.DataType.UINT8
    )
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=1.0,
        generate_stats=False,
        block_mask=block_mask,
    )
    assert stats is None
    o.set_output(True)
    graph.validate().build_operation_graph()
    graph.execute(
        {q: q_value, k: k_value, v: v_value, block_mask: block_mask_value, o: o_value}
    )

    scores = torch.matmul(q_value, k_value.transpose(-2, -1))
    scores[..., 128:] = -torch.inf
    expected = torch.matmul(torch.softmax(scores, dim=-1), v_value)
    torch.testing.assert_close(o_value, expected, rtol=1.0e-5, atol=1.0e-5)


def test_mudnn_frontend_python_graph_api_execute_sdpa_seq_len_bottom_right_sink_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.arange(1 * 2 * 4 * 2, dtype=torch.float32).reshape(1, 2, 4, 2)
    k_value = torch.arange(1 * 2 * 4 * 2, dtype=torch.float32).reshape(1, 2, 4, 2)
    v_value = torch.arange(1 * 2 * 4 * 2, dtype=torch.float32).reshape(1, 2, 4, 2)
    q_value = q_value / 17.0 - 0.5
    k_value = k_value / 19.0 - 0.25
    v_value = v_value / 23.0 - 0.75
    seq_len_q_value = torch.tensor([4], dtype=torch.int32)
    seq_len_kv_value = torch.tensor([2], dtype=torch.int32)
    sink_value = torch.tensor([[[[0.125]], [[-0.25]]]], dtype=torch.float32)
    o_value = torch.empty_like(q_value)
    score_max_value = torch.empty(1, 2, 4, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    seq_len_q = graph.tensor(
        name="seq_q", dim=seq_len_q_value.shape, data_type=fe.DataType.INT32
    )
    seq_len_kv = graph.tensor(
        name="seq_kv", dim=seq_len_kv_value.shape, data_type=fe.DataType.INT32
    )
    sink = graph.tensor(name="sink", dim=sink_value.shape, data_type=fe.DataType.FLOAT)
    score_max = graph.tensor(name="max", dim=(1, 2, 4, 1), data_type=fe.DataType.FLOAT)
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=0.5,
        generate_stats=False,
        use_padding_mask=True,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        diagonal_alignment=fe.DiagonalAlignment.BOTTOM_RIGHT,
        diagonal_band_left_bound=2,
        diagonal_band_right_bound=4,
        sink_token=sink,
        score_max=score_max,
    )
    assert stats is None
    o.set_output(True)
    graph.validate().build_operation_graph()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            seq_len_q: seq_len_q_value,
            seq_len_kv: seq_len_kv_value,
            sink: sink_value,
            o: o_value,
            score_max: score_max_value,
        }
    )

    scores = torch.matmul(q_value, k_value.transpose(-2, -1)) * 0.5
    q_index = torch.arange(q_value.shape[-2]).reshape(1, 1, -1, 1)
    k_index = torch.arange(k_value.shape[-2]).reshape(1, 1, 1, -1)
    valid_q = q_index < seq_len_q_value.reshape(-1, 1, 1, 1)
    valid_k = k_index < seq_len_kv_value.reshape(-1, 1, 1, 1)
    offset = (seq_len_kv_value - seq_len_q_value).reshape(-1, 1, 1, 1)
    diagonal_keep = (k_index > q_index + offset - 2) & (k_index <= q_index + offset + 4)
    scores = torch.where(
        valid_q & valid_k & diagonal_keep,
        scores,
        torch.full_like(scores, -torch.inf),
    )
    sink_scores = torch.where(
        valid_q,
        sink_value.expand(tuple(scores.shape[:-1]) + (1,)),
        torch.full(tuple(scores.shape[:-1]) + (1,), -torch.inf),
    )
    extended_scores = torch.cat([sink_scores, scores], dim=-1)
    row_has_scores = torch.isfinite(extended_scores).any(dim=-1, keepdim=True)
    scores_for_softmax = torch.where(
        row_has_scores, extended_scores, torch.zeros_like(extended_scores)
    )
    expected_max = scores_for_softmax.max(dim=-1, keepdim=True).values
    probs = torch.softmax(scores_for_softmax, dim=-1)[..., 1:]
    probs = probs * row_has_scores.to(dtype=probs.dtype)
    expected_o = torch.matmul(probs, v_value)

    torch.testing.assert_close(o_value, expected_o, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(
        score_max_value, expected_max.squeeze(-1), rtol=1.0e-5, atol=1.0e-5
    )


def test_mudnn_frontend_python_graph_api_execute_sdpa_seq_len_left_bound_score_sum_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.arange(1 * 1 * 5 * 2, dtype=torch.float32).reshape(1, 1, 5, 2)
    k_value = torch.arange(1 * 1 * 5 * 2, dtype=torch.float32).reshape(1, 1, 5, 2)
    v_value = torch.arange(1 * 1 * 5 * 2, dtype=torch.float32).reshape(1, 1, 5, 2)
    q_value = q_value / 17.0 - 0.5
    k_value = k_value / 19.0 - 0.25
    v_value = v_value / 23.0 - 0.75
    seq_len_q_value = torch.tensor([5], dtype=torch.int32)
    seq_len_kv_value = torch.tensor([2], dtype=torch.int32)
    o_value = torch.empty_like(q_value)
    score_sum_exp_value = torch.empty(1, 1, 5, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    seq_len_q = graph.tensor(
        name="seq_q", dim=seq_len_q_value.shape, data_type=fe.DataType.INT32
    )
    seq_len_kv = graph.tensor(
        name="seq_kv", dim=seq_len_kv_value.shape, data_type=fe.DataType.INT32
    )
    score_sum_exp = graph.tensor(
        name="sum_exp", dim=(1, 1, 5, 1), data_type=fe.DataType.FLOAT
    )
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=0.5,
        generate_stats=False,
        use_padding_mask=True,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        diagonal_alignment=fe.DiagonalAlignment.TOP_LEFT,
        diagonal_band_left_bound=2,
        diagonal_band_right_bound=0,
        score_sum_exp=score_sum_exp,
    )
    assert stats is None
    o.set_output(True)
    graph.validate().build_operation_graph()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            seq_len_q: seq_len_q_value,
            seq_len_kv: seq_len_kv_value,
            o: o_value,
            score_sum_exp: score_sum_exp_value,
        }
    )

    scores = torch.matmul(q_value, k_value.transpose(-2, -1)) * 0.5
    q_index = torch.arange(q_value.shape[-2]).reshape(1, 1, -1, 1)
    k_index = torch.arange(k_value.shape[-2]).reshape(1, 1, 1, -1)
    valid_q = q_index < seq_len_q_value.reshape(-1, 1, 1, 1)
    valid_k = k_index < seq_len_kv_value.reshape(-1, 1, 1, 1)
    window_keep = (k_index <= q_index) & (k_index > q_index - 2)
    scores = torch.where(
        valid_q & valid_k & window_keep,
        scores,
        torch.full_like(scores, -torch.inf),
    )
    row_has_scores = torch.isfinite(scores).any(dim=-1, keepdim=True)
    scores_for_softmax = torch.where(row_has_scores, scores, torch.zeros_like(scores))
    score_max = scores_for_softmax.max(dim=-1, keepdim=True).values
    expected_sum_exp = torch.exp(scores_for_softmax - score_max).sum(
        dim=-1, keepdim=True
    )
    expected_sum_exp = torch.where(
        row_has_scores, expected_sum_exp, torch.zeros_like(expected_sum_exp)
    )
    probs = torch.softmax(scores_for_softmax, dim=-1) * row_has_scores.to(
        dtype=scores.dtype
    )
    expected_o = torch.matmul(probs, v_value)

    torch.testing.assert_close(o_value, expected_o, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(
        score_sum_exp_value,
        expected_sum_exp.squeeze(-1),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_mudnn_frontend_python_graph_api_execute_sdpa_diagonal_band_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    def run_case(q_value, k_value, v_value, alignment, left_bound, right_bound):
        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
        k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
        v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
        o, stats = graph.sdpa(
            q,
            k,
            v,
            attn_scale=1.0,
            generate_stats=False,
            diagonal_alignment=alignment,
            diagonal_band_left_bound=left_bound,
            diagonal_band_right_bound=right_bound,
        )
        assert stats is None
        o.set_output(True)
        graph.validate().build_operation_graph()
        out = torch.empty_like(q_value)
        graph.execute(
            {q: q_value, k: k_value, v: v_value, o: out, None: None, 999_999: out}
        )

        q_index = torch.arange(q_value.shape[-2]).reshape(1, 1, q_value.shape[-2], 1)
        k_index = torch.arange(k_value.shape[-2]).reshape(1, 1, 1, k_value.shape[-2])
        offset = 0
        if alignment == fe.diagonal_alignment.BOTTOM_RIGHT:
            offset = k_value.shape[-2] - q_value.shape[-2]
        keep_rows = None
        mask = torch.ones(1, 1, q_value.shape[-2], k_value.shape[-2], dtype=torch.bool)
        if right_bound is not None:
            mask &= k_index <= q_index + offset + int(right_bound)
        if left_bound is not None:
            keep_rows = q_index < k_value.shape[-2] + int(left_bound) - 1
            mask &= (k_index > q_index + offset - int(left_bound)) | ~keep_rows
        expected = torch.nn.functional.scaled_dot_product_attention(
            q_value, k_value, v_value, attn_mask=mask, scale=1.0
        )
        if keep_rows is not None:
            expected = expected * keep_rows.to(dtype=expected.dtype)
        expected = torch.nan_to_num(expected, nan=0.0)
        torch.testing.assert_close(out, expected, rtol=5.0e-4, atol=5.0e-4)

    q = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8) / 17.0
    k = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8) / 19.0 - 0.5
    v = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8) / 23.0 - 1.0
    run_case(q, k, v, fe.diagonal_alignment.TOP_LEFT, 2, 0)

    q_br = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 8) / 29.0
    k_br = torch.arange(32, dtype=torch.float32).reshape(1, 1, 4, 8) / 31.0 - 0.25
    v_br = torch.arange(32, dtype=torch.float32).reshape(1, 1, 4, 8) / 37.0 - 0.5
    run_case(q_br, k_br, v_br, fe.diagonal_alignment.BOTTOM_RIGHT, 2, 0)

    q_long = torch.arange(40, dtype=torch.float32).reshape(1, 1, 5, 8) / 41.0
    k_short = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8) / 43.0 - 0.5
    v_short = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8) / 47.0 - 1.0
    run_case(q_long, k_short, v_short, fe.diagonal_alignment.TOP_LEFT, 2, 0)
    run_case(q_long, k_short, v_short, fe.diagonal_alignment.BOTTOM_RIGHT, None, 1)


def test_mudnn_frontend_python_graph_api_execute_ragged_sdpa_diagonal_offset_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4) / 11.0
    k_value = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4) / 13.0 - 0.5
    v_value = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4) / 17.0 - 1.0
    seq_len_q = torch.tensor([3], dtype=torch.int32).reshape(1, 1, 1, 1)
    seq_len_kv = torch.tensor([6], dtype=torch.int32).reshape(1, 1, 1, 1)
    q_offset = torch.tensor([0, q_value.numel()], dtype=torch.int64).reshape(2, 1, 1, 1)
    o_offset = torch.tensor([0, q_value.numel()], dtype=torch.int64).reshape(2, 1, 1, 1)
    stats_offset = torch.tensor([0, 3], dtype=torch.int64).reshape(2, 1, 1, 1)
    out = torch.empty_like(q_value)
    score_max_value = torch.empty(3, 1, 1, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(
        name="q",
        dim=(1, 1, 3, 4),
        stride=(12, 4, 4, 1),
        data_type=fe.DataType.FLOAT,
    )
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    q_ragged = graph.tensor(
        name="q_ragged", dim=q_offset.shape, data_type=fe.DataType.INT64
    )
    q.set_ragged_offset(q_ragged)
    seq_q = graph.tensor(name="seq_q", dim=seq_len_q.shape, data_type=fe.DataType.INT32)
    seq_kv = graph.tensor(
        name="seq_kv", dim=seq_len_kv.shape, data_type=fe.DataType.INT32
    )
    score_max = graph.tensor(
        name="score_max", dim=(1, 1, 3, 1), data_type=fe.DataType.FLOAT
    )
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=1.0,
        generate_stats=False,
        use_padding_mask=True,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
        diagonal_band_right_bound=2,
        score_max=score_max,
    )
    assert stats is None
    o.set_output(True).set_dim((1, 1, 3, 4)).set_stride((12, 4, 4, 1))
    score_max.set_output(True).set_dim((1, 1, 3, 1)).set_stride((3, 1, 1, 1))
    o_ragged = graph.tensor(
        name="o_ragged", dim=o_offset.shape, data_type=fe.DataType.INT64
    )
    stats_ragged = graph.tensor(
        name="stats_ragged", dim=stats_offset.shape, data_type=fe.DataType.INT64
    )
    o.set_ragged_offset(o_ragged)
    score_max.set_ragged_offset(stats_ragged)
    graph.validate().build_operation_graph()
    graph.execute(
        {
            q: q_value,
            q_ragged: q_offset,
            k: k_value,
            v: v_value,
            seq_q: seq_len_q,
            seq_kv: seq_len_kv,
            o: out,
            o_ragged: o_offset,
            score_max: score_max_value,
            stats_ragged: stats_offset,
        }
    )

    q_batch = q_value.permute(1, 0, 2)
    scores = torch.matmul(q_batch, k_value[0].transpose(-2, -1))
    q_index = torch.arange(3).reshape(3, 1)
    k_index = torch.arange(6).reshape(1, 6)
    scores = torch.where(
        k_index > q_index + 2,
        torch.full_like(scores, -torch.inf),
        scores,
    )
    expected = torch.matmul(torch.softmax(scores, dim=-1), v_value[0]).permute(1, 0, 2)
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(
        score_max_value,
        scores.max(dim=-1, keepdim=True).values.permute(1, 0, 2),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_mudnn_frontend_python_graph_api_execute_ragged_sdpa_sink_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2) / 11.0
    k_value = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2) / 13.0
    v_value = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3) / 17.0
    q_value = q_value - 0.5
    k_value = k_value - 0.25
    v_value = v_value - 0.75
    sink_value = torch.tensor([[[[0.125]], [[-0.25]]]], dtype=torch.float32)
    seq_len_q = torch.tensor([3], dtype=torch.int32)
    seq_len_kv = torch.tensor([4], dtype=torch.int32)
    q_offset = torch.tensor([0, q_value.numel()], dtype=torch.int64)
    k_offset = torch.tensor([0, k_value.numel()], dtype=torch.int64)
    v_offset = torch.tensor([0, v_value.numel()], dtype=torch.int64)
    o_offset = torch.tensor([0, 3 * 2 * 3], dtype=torch.int64)
    seed_value = torch.tensor([123], dtype=torch.int64)
    offset_value = torch.tensor([789], dtype=torch.int64)
    rng_dump_value = torch.zeros(1, 2, 3, 4, dtype=torch.float32)
    out = torch.empty(3, 2, 3, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=(1, 2, 3, 2), data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=(1, 2, 4, 2), data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=(1, 2, 4, 3), data_type=fe.DataType.FLOAT)
    seq_q = graph.tensor(name="seq_q", dim=(1,), data_type=fe.DataType.INT32)
    seq_kv = graph.tensor(name="seq_kv", dim=(1,), data_type=fe.DataType.INT32)
    sink = graph.tensor(name="sink", dim=sink_value.shape, data_type=fe.DataType.FLOAT)
    seed = graph.tensor(name="seed", dim=(1,), data_type=fe.DataType.INT64)
    offset = graph.tensor(name="offset", dim=(1,), data_type=fe.DataType.INT64)
    rng_dump = graph.tensor(
        name="rng_dump", dim=rng_dump_value.shape, data_type=fe.DataType.FLOAT
    )
    q_ragged = graph.tensor(name="q_ragged", dim=(2,), data_type=fe.DataType.INT64)
    k_ragged = graph.tensor(name="k_ragged", dim=(2,), data_type=fe.DataType.INT64)
    v_ragged = graph.tensor(name="v_ragged", dim=(2,), data_type=fe.DataType.INT64)
    o_ragged = graph.tensor(name="o_ragged", dim=(2,), data_type=fe.DataType.INT64)
    q.set_ragged_offset(q_ragged)
    k.set_ragged_offset(k_ragged)
    v.set_ragged_offset(v_ragged)
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=0.5,
        generate_stats=False,
        use_padding_mask=True,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
        diagonal_alignment=fe.DiagonalAlignment.TOP_LEFT,
        diagonal_band_left_bound=2,
        diagonal_band_right_bound=1,
        sink_token=sink,
        dropout=(0.25, seed, offset),
        rng_dump=rng_dump,
    )
    assert stats is None
    o.set_output(True).set_ragged_offset(o_ragged)
    graph.validate().build_operation_graph()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            seq_q: seq_len_q,
            seq_kv: seq_len_kv,
            sink: sink_value,
            seed: seed_value,
            offset: offset_value,
            rng_dump: rng_dump_value,
            q_ragged: q_offset,
            k_ragged: k_offset,
            v_ragged: v_offset,
            o_ragged: o_offset,
            o: out,
        }
    )

    q_batch = q_value.permute(1, 0, 2)
    k_batch = k_value.permute(1, 0, 2)
    v_batch = v_value.permute(1, 0, 2)
    scores = torch.matmul(q_batch, k_batch.transpose(-2, -1)) * 0.5
    q_index = torch.arange(3).reshape(3, 1)
    k_index = torch.arange(4).reshape(1, 4)
    masked = (k_index > q_index + 1) | (k_index <= q_index - 2)
    scores = torch.where(
        masked.unsqueeze(0), torch.full_like(scores, -torch.inf), scores
    )
    extended_scores = torch.cat(
        [sink_value[0].expand(2, 3, 1), scores],
        dim=-1,
    )
    probs = torch.softmax(extended_scores, dim=-1)[..., 1:]
    expected = torch.matmul(probs / 0.75, v_batch).permute(1, 0, 2)
    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(rng_dump_value, torch.ones_like(rng_dump_value))


def test_mudnn_frontend_python_graph_api_execute_sdpa_dropout_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8) / 17.0
    k_value = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8) / 19.0 - 0.5
    v_value = torch.arange(24, dtype=torch.float32).reshape(1, 1, 3, 8) / 23.0 - 1.0
    seed_value = torch.tensor([123], dtype=torch.int64)
    offset_value = torch.tensor([789], dtype=torch.int64)
    rng_dump_value = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    out = torch.empty_like(q_value)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    seed = graph.tensor(name="seed", dim=(1,), data_type=fe.DataType.INT64)
    offset = graph.tensor(name="offset", dim=(1,), data_type=fe.DataType.INT64)
    rng_dump = graph.tensor(
        name="rng_dump", dim=rng_dump_value.shape, data_type=fe.DataType.FLOAT
    )
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=1.0,
        generate_stats=False,
        dropout=(0.25, seed, offset),
        rng_dump=rng_dump,
    )
    assert stats is None
    o.set_output(True)
    graph.validate().build_operation_graph()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            seed: seed_value,
            offset: offset_value,
            rng_dump: rng_dump_value,
            o: out,
        }
    )

    expected = (
        torch.nn.functional.scaled_dot_product_attention(
            q_value, k_value, v_value, scale=1.0
        )
        / 0.75
    )
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(rng_dump_value, torch.ones_like(rng_dump_value))


def test_mudnn_frontend_python_graph_api_execute_sdpa_left_bound_dropout_stats_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    q_value = torch.arange(1 * 2 * 5 * 4, dtype=torch.float32).reshape(1, 2, 5, 4)
    k_value = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    v_value = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    q_value = q_value / 17.0 - 0.5
    k_value = k_value / 19.0 - 0.25
    v_value = v_value / 23.0 - 0.75
    seed_value = torch.tensor([123], dtype=torch.int64)
    offset_value = torch.tensor([789], dtype=torch.int64)
    rng_dump_value = torch.zeros(1, 2, 5, 3, dtype=torch.float32)
    out = torch.empty_like(q_value)
    stats_value = torch.empty(1, 2, 5, 1, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    seed = graph.tensor(name="seed", dim=(1,), data_type=fe.DataType.INT64)
    offset = graph.tensor(name="offset", dim=(1,), data_type=fe.DataType.INT64)
    rng_dump = graph.tensor(
        name="rng_dump", dim=rng_dump_value.shape, data_type=fe.DataType.FLOAT
    )
    o, stats = graph.sdpa(
        q,
        k,
        v,
        attn_scale=0.5,
        generate_stats=True,
        diagonal_alignment=fe.DiagonalAlignment.TOP_LEFT,
        diagonal_band_left_bound=1,
        diagonal_band_right_bound=0,
        dropout=(0.25, seed, offset),
        rng_dump=rng_dump,
    )
    o.set_output(True)
    stats.set_output(True)
    graph.validate().build_operation_graph()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            seed: seed_value,
            offset: offset_value,
            rng_dump: rng_dump_value,
            o: out,
            stats: stats_value,
        }
    )

    q_index = torch.arange(q_value.shape[-2]).reshape(1, 1, -1, 1)
    k_index = torch.arange(k_value.shape[-2]).reshape(1, 1, 1, -1)
    keep_query = q_index < k_value.shape[-2]
    scores = (
        torch.matmul(
            (q_value * keep_query.to(dtype=q_value.dtype)).to(dtype=torch.float32),
            k_value.to(dtype=torch.float32).transpose(-2, -1),
        )
        * 0.5
    )
    mask = k_index > q_index
    left_mask = (k_index <= q_index - 1) & keep_query
    scores = torch.where(mask | left_mask, torch.full_like(scores, -torch.inf), scores)
    probs = torch.softmax(scores, dim=-1) * keep_query.to(dtype=scores.dtype)
    expected = torch.matmul(probs / 0.75, v_value.to(dtype=torch.float32))
    expected_stats = scores.max(dim=-1, keepdim=True).values + torch.log(
        torch.exp(scores - scores.max(dim=-1, keepdim=True).values).sum(
            dim=-1, keepdim=True
        )
    )

    torch.testing.assert_close(out, expected, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(stats_value, expected_stats, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(rng_dump_value, torch.ones_like(rng_dump_value))


def test_mudnn_frontend_python_graph_api_execute_sdpa_generate_stats():
    import cudnn_frontend as fe

    # MUSA Flash Attention requires a head dimension divisible by eight when
    # the fused forward also materializes the LSE/Stats output.
    q_np = np.array(
        [[[[
            0.2, -0.5, 0.1, 0.4, -0.3, 0.6, -0.1, 0.7,
        ], [
            1.0, 0.25, -0.3, 0.2, 0.4, -0.6, 0.8, -0.15,
        ]]]],
        dtype=np.float16,
    )
    k_np = np.array(
        [[[[
            0.0, 0.75, -0.4, 0.1, 0.2, -0.35, 0.45, -0.15,
        ], [
            -0.25, 0.5, 0.3, -0.2, 0.6, -0.1, -0.45, 0.35,
        ]]]],
        dtype=np.float16,
    )
    v_np = np.array(
        [[[[
            1.0, -1.0, 0.25, 0.5, -0.75, 0.125, 0.3, -0.4,
        ], [
            0.5, 2.0, -0.25, 0.75, -0.2, 0.6, -1.25, 0.9,
        ]]]],
        dtype=np.float16,
    )
    scale = 0.5
    expected = _sdpa_reference(
        q_np.astype(np.float32),
        k_np.astype(np.float32),
        v_np.astype(np.float32),
        scale,
    ).astype(np.float16)
    stats_np = np.zeros((1, 1, 2, 1), dtype=np.float32)

    tensors = []
    try:
        q = device_memory.create_tensor(
            q_np.__array_interface__["data"][0], q_np.nbytes, q_np.shape, "float16"
        )
        k = device_memory.create_tensor(
            k_np.__array_interface__["data"][0], k_np.nbytes, k_np.shape, "float16"
        )
        v = device_memory.create_tensor(
            v_np.__array_interface__["data"][0], v_np.nbytes, v_np.shape, "float16"
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float16"
        )
        stats = device_memory.create_tensor(
            stats_np.__array_interface__["data"][0],
            stats_np.nbytes,
            stats_np.shape,
            "float32",
        )
        tensors = [q, k, v, out, stats]

        graph = fe.pygraph(
            io_data_type=fe.DataType.HALF,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q_desc = graph.tensor(name="q", dim=q_np.shape, data_type=fe.DataType.HALF)
        k_desc = graph.tensor(name="k", dim=k_np.shape, data_type=fe.DataType.HALF)
        v_desc = graph.tensor(name="v", dim=v_np.shape, data_type=fe.DataType.HALF)
        o_desc, stats_desc = graph.sdpa(
            q_desc,
            k_desc,
            v_desc,
            name="attention_train",
            attn_scale=scale,
            generate_stats=True,
        )

        graph.execute(
            {
                q_desc: device_memory.get_tensor_addr(q)[0],
                k_desc: device_memory.get_tensor_addr(k)[0],
                v_desc: device_memory.get_tensor_addr(v)[0],
                o_desc: device_memory.get_tensor_addr(out)[0],
                stats_desc: device_memory.get_tensor_addr(stats)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        actual_stats = device_memory.copy_to_ndarray(stats, stats_np.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-2, atol=2.0e-2)
        assert np.all(np.isfinite(actual_stats))
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


@pytest.mark.parametrize("causal_mask", [False, True])
def test_mudnn_frontend_python_graph_api_execute_sdpa_backward(causal_mask):
    import cudnn_frontend as fe

    q_np = np.linspace(-0.3, 0.4, num=1 * 1 * 2 * 64, dtype=np.float32).reshape(
        1, 1, 2, 64
    )
    k_np = np.linspace(0.2, -0.25, num=1 * 1 * 2 * 64, dtype=np.float32).reshape(
        1, 1, 2, 64
    )
    v_np = np.linspace(-0.5, 0.6, num=1 * 1 * 2 * 64, dtype=np.float32).reshape(
        1, 1, 2, 64
    )
    d_o_np = np.linspace(0.35, -0.4, num=1 * 1 * 2 * 64, dtype=np.float32).reshape(
        1, 1, 2, 64
    )
    q_np = q_np.astype(np.float16)
    k_np = k_np.astype(np.float16)
    v_np = v_np.astype(np.float16)
    d_o_np = d_o_np.astype(np.float16)
    scale = 0.5
    expected_dq, expected_dk, expected_dv = _sdpa_backward_reference(
        q_np.astype(np.float32),
        k_np.astype(np.float32),
        v_np.astype(np.float32),
        d_o_np.astype(np.float32),
        scale,
        causal_mask=causal_mask,
    )
    expected_dq = expected_dq.astype(np.float16)
    expected_dk = expected_dk.astype(np.float16)
    expected_dv = expected_dv.astype(np.float16)

    tensors = []
    try:
        q = device_memory.create_tensor(
            q_np.__array_interface__["data"][0], q_np.nbytes, q_np.shape, "float16"
        )
        k = device_memory.create_tensor(
            k_np.__array_interface__["data"][0], k_np.nbytes, k_np.shape, "float16"
        )
        v = device_memory.create_tensor(
            v_np.__array_interface__["data"][0], v_np.nbytes, v_np.shape, "float16"
        )
        d_o = device_memory.create_tensor(
            d_o_np.__array_interface__["data"][0],
            d_o_np.nbytes,
            d_o_np.shape,
            "float16",
        )
        d_q = device_memory.create_tensor(
            0, expected_dq.nbytes, expected_dq.shape, "float16"
        )
        d_k = device_memory.create_tensor(
            0, expected_dk.nbytes, expected_dk.shape, "float16"
        )
        d_v = device_memory.create_tensor(
            0, expected_dv.nbytes, expected_dv.shape, "float16"
        )
        tensors = [q, k, v, d_o, d_q, d_k, d_v]

        graph = fe.pygraph(
            io_data_type=fe.DataType.HALF,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q_desc = graph.tensor(name="q", dim=q_np.shape, data_type=fe.DataType.HALF)
        k_desc = graph.tensor(name="k", dim=k_np.shape, data_type=fe.DataType.HALF)
        v_desc = graph.tensor(name="v", dim=v_np.shape, data_type=fe.DataType.HALF)
        d_o_desc = graph.tensor(name="do", dim=d_o_np.shape, data_type=fe.DataType.HALF)
        o_desc, stats_desc = graph.sdpa(
            q_desc,
            k_desc,
            v_desc,
            name="attention_train",
            attn_scale=scale,
            generate_stats=True,
            causal_mask=causal_mask,
        )
        d_q_desc, d_k_desc, d_v_desc = graph.sdpa_backward(
            q_desc,
            k_desc,
            v_desc,
            o_desc,
            d_o_desc,
            stats_desc,
            name="attention_bwd",
            attn_scale=scale,
            use_causal_mask=causal_mask,
        )
        d_q_desc.set_output()
        d_k_desc.set_output()
        d_v_desc.set_output()
        workspace_size = graph.build().get_workspace_size()
        workspace = device_memory.create_tensor(
            0, workspace_size, (workspace_size,), "uint8"
        )
        tensors.append(workspace)

        graph.execute(
            {
                q_desc: device_memory.get_tensor_addr(q)[0],
                k_desc: device_memory.get_tensor_addr(k)[0],
                v_desc: device_memory.get_tensor_addr(v)[0],
                d_o_desc: device_memory.get_tensor_addr(d_o)[0],
                d_q_desc: device_memory.get_tensor_addr(d_q)[0],
                d_k_desc: device_memory.get_tensor_addr(d_k)[0],
                d_v_desc: device_memory.get_tensor_addr(d_v)[0],
            },
            workspace_ptr=device_memory.get_tensor_addr(workspace)[0],
        )

        actual_dq = device_memory.copy_to_ndarray(d_q, expected_dq.nbytes)
        actual_dk = device_memory.copy_to_ndarray(d_k, expected_dk.nbytes)
        actual_dv = device_memory.copy_to_ndarray(d_v, expected_dv.nbytes)
        assert np.all(np.isfinite(actual_dq))
        assert np.all(np.isfinite(actual_dk))
        assert np.all(np.isfinite(actual_dv))
        assert np.any(actual_dq != 0.0)
        assert np.any(actual_dk != 0.0)
        assert np.any(actual_dv != 0.0)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_sdpa_backward_seq_len_window_sink_cpu_fallback(
    monkeypatch,
):
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *args, **kwargs: "Fake CPU CUDA"
    )

    q_value = torch.arange(1 * 2 * 4 * 3, dtype=torch.float32).reshape(1, 2, 4, 3)
    k_value = torch.arange(1 * 1 * 4 * 3, dtype=torch.float32).reshape(1, 1, 4, 3)
    v_value = torch.arange(1 * 1 * 4 * 2, dtype=torch.float32).reshape(1, 1, 4, 2)
    d_o_value = torch.arange(1 * 2 * 4 * 2, dtype=torch.float32).reshape(1, 2, 4, 2)
    q_value = q_value / 17.0 - 0.5
    k_value = k_value / 19.0 - 0.25
    v_value = v_value / 23.0 - 0.75
    d_o_value = d_o_value / 29.0 - 0.5
    o_value = torch.zeros_like(d_o_value)
    stats_value = torch.zeros(1, 2, 4, 1, dtype=torch.float32)
    seq_len_q_value = torch.tensor([4], dtype=torch.int32)
    seq_len_kv_value = torch.tensor([2], dtype=torch.int32)
    sink_value = torch.tensor([[[[0.125]], [[-0.25]]]], dtype=torch.float32)
    d_q_value = torch.empty_like(q_value)
    d_k_value = torch.empty_like(k_value)
    d_v_value = torch.empty_like(v_value)
    d_sink_value = torch.empty_like(sink_value)

    def reference():
        q_req = q_value.detach().clone().requires_grad_(True)
        k_req = k_value.detach().clone().requires_grad_(True)
        v_req = v_value.detach().clone().requires_grad_(True)
        sink_req = sink_value.detach().clone().requires_grad_(True)
        k_expanded = k_req.repeat_interleave(2, dim=1)
        v_expanded = v_req.repeat_interleave(2, dim=1)
        scores = torch.matmul(q_req, k_expanded.transpose(-2, -1)) * 0.5
        q_index = torch.arange(q_value.shape[-2]).reshape(1, 1, -1, 1)
        k_index = torch.arange(k_value.shape[-2]).reshape(1, 1, 1, -1)
        valid_q = q_index < seq_len_q_value.reshape(-1, 1, 1, 1)
        valid_k = k_index < seq_len_kv_value.reshape(-1, 1, 1, 1)
        window_keep = (k_index <= q_index + 1) & (k_index > q_index - 2)
        scores = torch.where(
            valid_q & valid_k & window_keep,
            scores,
            torch.full_like(scores, -torch.inf),
        )
        sink_scores = torch.where(
            valid_q,
            sink_req.expand(tuple(scores.shape[:-1]) + (1,)),
            torch.full(tuple(scores.shape[:-1]) + (1,), -torch.inf),
        )
        extended_scores = torch.cat([sink_scores, scores], dim=-1)
        row_has_scores = torch.isfinite(extended_scores).any(dim=-1, keepdim=True)
        scores_for_softmax = torch.where(
            row_has_scores, extended_scores, torch.zeros_like(extended_scores)
        )
        probs = torch.softmax(scores_for_softmax, dim=-1)[..., 1:]
        probs = probs * row_has_scores.to(dtype=probs.dtype)
        result = torch.matmul(probs, v_expanded)
        return torch.autograd.grad(result, (q_req, k_req, v_req, sink_req), d_o_value)

    expected_dq, expected_dk, expected_dv, expected_dsink = reference()

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    o = graph.tensor(name="o", dim=o_value.shape, data_type=fe.DataType.FLOAT)
    d_o = graph.tensor(name="do", dim=d_o_value.shape, data_type=fe.DataType.FLOAT)
    stats = graph.tensor(
        name="stats", dim=stats_value.shape, data_type=fe.DataType.FLOAT
    )
    seq_len_q = graph.tensor(
        name="seq_q", dim=seq_len_q_value.shape, data_type=fe.DataType.INT32
    )
    seq_len_kv = graph.tensor(
        name="seq_kv", dim=seq_len_kv_value.shape, data_type=fe.DataType.INT32
    )
    sink = graph.tensor(name="sink", dim=sink_value.shape, data_type=fe.DataType.FLOAT)
    d_sink = graph.tensor(
        name="d_sink", dim=d_sink_value.shape, data_type=fe.DataType.FLOAT
    )
    d_q, d_k, d_v = graph.sdpa_backward(
        q,
        k,
        v,
        o,
        d_o,
        stats,
        attn_scale=0.5,
        use_padding_mask=True,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        diagonal_alignment=fe.DiagonalAlignment.TOP_LEFT,
        diagonal_band_left_bound=2,
        diagonal_band_right_bound=1,
        sink_token=sink,
        dSink_token=d_sink,
    )
    d_q.set_output(True)
    d_k.set_output(True)
    d_v.set_output(True)
    graph.validate().build_operation_graph().create_execution_plans([fe.heur_mode.A])
    graph.check_support().build_plans()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            o: o_value,
            d_o: d_o_value,
            stats: stats_value,
            seq_len_q: seq_len_q_value,
            seq_len_kv: seq_len_kv_value,
            sink: sink_value,
            d_sink: d_sink_value,
            d_q: d_q_value,
            d_k: d_k_value,
            d_v: d_v_value,
        }
    )

    torch.testing.assert_close(d_q_value, expected_dq, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(d_k_value, expected_dk, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(d_v_value, expected_dv, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(d_sink_value, expected_dsink, rtol=1.0e-5, atol=1.0e-5)


def test_mudnn_frontend_python_graph_api_execute_sdpa_backward_ragged_cpu_fallback(
    monkeypatch,
):
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *args, **kwargs: "Fake CPU CUDA"
    )

    seq_len_q_value = torch.tensor([2, 1], dtype=torch.int32)
    seq_len_kv_value = torch.tensor([1, 2], dtype=torch.int32)
    q_offsets = torch.tensor([0, 8, 12], dtype=torch.int64)
    k_offsets = torch.tensor([0, 2, 6], dtype=torch.int64)
    v_offsets = torch.tensor([0, 3, 9], dtype=torch.int64)
    o_offsets = torch.tensor([0, 12, 18], dtype=torch.int64)
    stats_offsets = torch.tensor([0, 4, 6], dtype=torch.int64)

    q_value = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2) / 11.0
    k_value = torch.arange(3 * 1 * 2, dtype=torch.float32).reshape(3, 1, 2) / 13.0
    v_value = torch.arange(3 * 1 * 3, dtype=torch.float32).reshape(3, 1, 3) / 17.0
    d_o_value = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3) / 19.0
    q_value = q_value - 0.5
    k_value = k_value - 0.25
    v_value = v_value - 0.75
    d_o_value = d_o_value - 0.5
    o_value = torch.zeros_like(d_o_value)
    stats_value = torch.zeros(3, 2, 1, dtype=torch.float32)
    d_q_value = torch.empty_like(q_value)
    d_k_value = torch.empty_like(k_value)
    d_v_value = torch.empty_like(v_value)

    def reference():
        expected_dq = torch.zeros_like(q_value)
        expected_dk = torch.zeros_like(k_value)
        expected_dv = torch.zeros_like(v_value)
        for batch, (q_len, kv_len) in enumerate(
            zip(seq_len_q_value.tolist(), seq_len_kv_value.tolist())
        ):
            q_start = int(q_offsets[batch].item()) // (2 * 2)
            k_start = int(k_offsets[batch].item()) // (1 * 2)
            v_start = int(v_offsets[batch].item()) // (1 * 3)
            o_start = int(o_offsets[batch].item()) // (2 * 3)
            q_req = (
                q_value[q_start : q_start + q_len]
                .permute(1, 0, 2)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            k_req = (
                k_value[k_start : k_start + kv_len]
                .permute(1, 0, 2)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            v_req = (
                v_value[v_start : v_start + kv_len]
                .permute(1, 0, 2)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            d_o_batch = d_o_value[o_start : o_start + q_len].permute(1, 0, 2)
            k_expanded = k_req.repeat_interleave(2, dim=0)
            v_expanded = v_req.repeat_interleave(2, dim=0)
            scores = torch.matmul(q_req, k_expanded.transpose(-2, -1)) * 0.5
            q_index = torch.arange(q_len).reshape(q_len, 1)
            k_index = torch.arange(kv_len).reshape(1, kv_len)
            keep = k_index <= q_index + kv_len - q_len
            scores = torch.where(
                keep.unsqueeze(0), scores, torch.full_like(scores, -torch.inf)
            )
            row_has_scores = torch.isfinite(scores).any(dim=-1, keepdim=True)
            probs = torch.softmax(
                torch.where(row_has_scores, scores, torch.zeros_like(scores)), dim=-1
            )
            probs = probs * row_has_scores.to(dtype=probs.dtype)
            result = torch.matmul(probs, v_expanded)
            o_value[o_start : o_start + q_len] = result.detach().permute(1, 0, 2)
            d_q, d_k, d_v = torch.autograd.grad(
                result, (q_req, k_req, v_req), d_o_batch
            )
            expected_dq[q_start : q_start + q_len] = d_q.permute(1, 0, 2)
            expected_dk[k_start : k_start + kv_len] += d_k.permute(1, 0, 2)
            expected_dv[v_start : v_start + kv_len] += d_v.permute(1, 0, 2)
        return expected_dq, expected_dk, expected_dv

    expected_dq, expected_dk, expected_dv = reference()

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=(2, 2, 2, 2), data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=(2, 1, 2, 2), data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=(2, 1, 2, 3), data_type=fe.DataType.FLOAT)
    o = graph.tensor(name="o", dim=(2, 2, 2, 3), data_type=fe.DataType.FLOAT)
    d_o = graph.tensor(name="do", dim=(2, 2, 2, 3), data_type=fe.DataType.FLOAT)
    stats = graph.tensor(name="stats", dim=(2, 2, 2, 1), data_type=fe.DataType.FLOAT)
    seq_len_q = graph.tensor(name="seq_q", dim=(2,), data_type=fe.DataType.INT32)
    seq_len_kv = graph.tensor(name="seq_kv", dim=(2,), data_type=fe.DataType.INT32)
    q_ragged = graph.tensor(name="q_ragged", dim=(3,), data_type=fe.DataType.INT64)
    k_ragged = graph.tensor(name="k_ragged", dim=(3,), data_type=fe.DataType.INT64)
    v_ragged = graph.tensor(name="v_ragged", dim=(3,), data_type=fe.DataType.INT64)
    o_ragged = graph.tensor(name="o_ragged", dim=(3,), data_type=fe.DataType.INT64)
    stats_ragged = graph.tensor(
        name="stats_ragged", dim=(3,), data_type=fe.DataType.INT64
    )
    q.set_ragged_offset(q_ragged)
    k.set_ragged_offset(k_ragged)
    v.set_ragged_offset(v_ragged)
    o.set_ragged_offset(o_ragged)
    d_o.set_ragged_offset(o_ragged)
    stats.set_ragged_offset(stats_ragged)
    d_q, d_k, d_v = graph.sdpa_backward(
        q,
        k,
        v,
        o,
        d_o,
        stats,
        attn_scale=0.5,
        use_padding_mask=True,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        max_total_seq_len_q=3,
        max_total_seq_len_kv=3,
        diagonal_alignment=fe.DiagonalAlignment.BOTTOM_RIGHT,
        diagonal_band_right_bound=0,
        use_deterministic_algorithm=True,
    )
    d_q.set_output(True).set_ragged_offset(q_ragged)
    d_k.set_output(True).set_ragged_offset(k_ragged)
    d_v.set_output(True).set_ragged_offset(v_ragged)
    graph.validate().build_operation_graph().create_execution_plans([fe.heur_mode.A])
    graph.check_support().build_plans()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            o: o_value,
            d_o: d_o_value,
            stats: stats_value,
            seq_len_q: seq_len_q_value,
            seq_len_kv: seq_len_kv_value,
            q_ragged: q_offsets,
            k_ragged: k_offsets,
            v_ragged: v_offsets,
            o_ragged: o_offsets,
            stats_ragged: stats_offsets,
            d_q: d_q_value,
            d_k: d_k_value,
            d_v: d_v_value,
        }
    )

    torch.testing.assert_close(d_q_value, expected_dq, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(d_k_value, expected_dk, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(d_v_value, expected_dv, rtol=1.0e-5, atol=1.0e-5)


def test_mudnn_frontend_python_graph_api_execute_sdpa_backward_bias_cpu_fallback(
    monkeypatch,
):
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *args, **kwargs: "Fake CPU CUDA"
    )

    q_value = torch.arange(1 * 2 * 3 * 2, dtype=torch.float32).reshape(1, 2, 3, 2)
    k_value = torch.arange(1 * 1 * 4 * 2, dtype=torch.float32).reshape(1, 1, 4, 2)
    v_value = torch.arange(1 * 1 * 4 * 2, dtype=torch.float32).reshape(1, 1, 4, 2)
    d_o_value = torch.arange(1 * 2 * 3 * 2, dtype=torch.float32).reshape(1, 2, 3, 2)
    bias_value = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    q_value = q_value / 11.0 - 0.25
    k_value = k_value / 13.0 - 0.5
    v_value = v_value / 17.0 - 0.75
    d_o_value = d_o_value / 19.0 - 0.5
    bias_value = bias_value / 23.0 - 0.25
    o_value = torch.zeros_like(d_o_value)
    stats_value = torch.zeros(1, 2, 3, 1, dtype=torch.float32)
    d_q_value = torch.empty_like(q_value)
    d_k_value = torch.empty_like(k_value)
    d_v_value = torch.empty_like(v_value)
    d_bias_value = torch.empty_like(bias_value)

    def reference():
        q_req = q_value.detach().clone().requires_grad_(True)
        k_req = k_value.detach().clone().requires_grad_(True)
        v_req = v_value.detach().clone().requires_grad_(True)
        bias_req = bias_value.detach().clone().requires_grad_(True)
        k_expanded = k_req.repeat_interleave(2, dim=1)
        v_expanded = v_req.repeat_interleave(2, dim=1)
        scores = torch.matmul(q_req, k_expanded.transpose(-2, -1)) * 0.5
        scores = scores + bias_req
        probs = torch.softmax(scores, dim=-1)
        result = torch.matmul(probs, v_expanded)
        return torch.autograd.grad(result, (q_req, k_req, v_req, bias_req), d_o_value)

    expected_dq, expected_dk, expected_dv, expected_dbias = reference()

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor(name="q", dim=q_value.shape, data_type=fe.DataType.FLOAT)
    k = graph.tensor(name="k", dim=k_value.shape, data_type=fe.DataType.FLOAT)
    v = graph.tensor(name="v", dim=v_value.shape, data_type=fe.DataType.FLOAT)
    o = graph.tensor(name="o", dim=o_value.shape, data_type=fe.DataType.FLOAT)
    d_o = graph.tensor(name="do", dim=d_o_value.shape, data_type=fe.DataType.FLOAT)
    stats = graph.tensor(
        name="stats", dim=stats_value.shape, data_type=fe.DataType.FLOAT
    )
    bias = graph.tensor(name="bias", dim=bias_value.shape, data_type=fe.DataType.FLOAT)
    d_bias = graph.tensor(
        name="dbias", dim=d_bias_value.shape, data_type=fe.DataType.FLOAT
    )
    d_q, d_k, d_v = graph.sdpa_backward(
        q,
        k,
        v,
        o,
        d_o,
        stats,
        attn_scale=0.5,
        bias=bias,
        dBias=d_bias,
    )
    d_q.set_output(True)
    d_k.set_output(True)
    d_v.set_output(True)
    graph.validate().build_operation_graph().create_execution_plans([fe.heur_mode.A])
    graph.check_support().build_plans()
    graph.execute(
        {
            q: q_value,
            k: k_value,
            v: v_value,
            o: o_value,
            d_o: d_o_value,
            stats: stats_value,
            bias: bias_value,
            d_bias: d_bias_value,
            d_q: d_q_value,
            d_k: d_k_value,
            d_v: d_v_value,
        }
    )

    torch.testing.assert_close(d_q_value, expected_dq, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(d_k_value, expected_dk, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(d_v_value, expected_dv, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(d_bias_value, expected_dbias, rtol=1.0e-5, atol=1.0e-5)


def test_mudnn_frontend_python_graph_api_execute_sdpa_causal_mask():
    import cudnn_frontend as fe

    q_np = np.array(
        [[[[0.2, -0.4], [1.0, 0.5], [-0.3, 0.7]]]],
        dtype=np.float32,
    )
    k_np = np.array(
        [[[[0.6, -0.1], [-0.2, 0.9], [0.4, 0.3]]]],
        dtype=np.float32,
    )
    v_np = np.array(
        [[[[1.0, -0.5], [0.25, 0.75], [-1.0, 0.5]]]],
        dtype=np.float32,
    )
    scale = 0.75
    expected = _sdpa_reference(q_np, k_np, v_np, scale, causal_mask=True)
    noncausal_expected = _sdpa_reference(q_np, k_np, v_np, scale)
    assert not np.allclose(expected, noncausal_expected, rtol=1.0e-5, atol=1.0e-5)

    tensors = []
    try:
        q = device_memory.create_tensor(
            q_np.__array_interface__["data"][0], q_np.nbytes, q_np.shape, "float32"
        )
        k = device_memory.create_tensor(
            k_np.__array_interface__["data"][0], k_np.nbytes, k_np.shape, "float32"
        )
        v = device_memory.create_tensor(
            v_np.__array_interface__["data"][0], v_np.nbytes, v_np.shape, "float32"
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [q, k, v, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q_desc = graph.tensor(name="q", dim=q_np.shape, data_type=fe.DataType.FLOAT)
        k_desc = graph.tensor(name="k", dim=k_np.shape, data_type=fe.DataType.FLOAT)
        v_desc = graph.tensor(name="v", dim=v_np.shape, data_type=fe.DataType.FLOAT)
        o_desc, stats = graph.sdpa(
            q_desc,
            k_desc,
            v_desc,
            name="attention_causal",
            attn_scale=scale,
            causal_mask=True,
            is_inference=True,
        )
        assert stats is None

        graph.execute(
            {
                q_desc: device_memory.get_tensor_addr(q)[0],
                k_desc: device_memory.get_tensor_addr(k)[0],
                v_desc: device_memory.get_tensor_addr(v)[0],
                o_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=5.0e-4)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_sdpa_with_bias():
    import cudnn_frontend as fe

    q_np = np.array([[[[0.2, -0.5], [1.0, 0.25]]]], dtype=np.float32)
    k_np = np.array([[[[0.0, 0.75], [-0.25, 0.5], [1.0, -1.0]]]], dtype=np.float32)
    v_np = np.array([[[[1.0, -1.0], [0.5, 2.0], [-0.25, 0.75]]]], dtype=np.float32)
    bias_np = np.array([[[[0.0, -1.0, 0.25], [0.5, 0.0, -0.75]]]], dtype=np.float32)
    scale = 0.5
    expected = _sdpa_reference(q_np, k_np, v_np, scale, bias=bias_np)

    tensors = []
    try:
        q = device_memory.create_tensor(
            q_np.__array_interface__["data"][0], q_np.nbytes, q_np.shape, "float32"
        )
        k = device_memory.create_tensor(
            k_np.__array_interface__["data"][0], k_np.nbytes, k_np.shape, "float32"
        )
        v = device_memory.create_tensor(
            v_np.__array_interface__["data"][0], v_np.nbytes, v_np.shape, "float32"
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [q, k, v, bias, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q_desc = graph.tensor(name="q", dim=q_np.shape, data_type=fe.DataType.FLOAT)
        k_desc = graph.tensor(name="k", dim=k_np.shape, data_type=fe.DataType.FLOAT)
        v_desc = graph.tensor(name="v", dim=v_np.shape, data_type=fe.DataType.FLOAT)
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        o_desc, _ = graph.sdpa(
            q_desc,
            k_desc,
            v_desc,
            bias=bias_desc,
            name="attention",
            attn_scale=scale,
            is_inference=True,
        )

        graph.execute(
            {
                q_desc: device_memory.get_tensor_addr(q)[0],
                k_desc: device_memory.get_tensor_addr(k)[0],
                v_desc: device_memory.get_tensor_addr(v)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                o_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=5.0e-4)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_sdpa_with_bool_mask():
    import cudnn_frontend as fe

    q_np = np.array([[[[0.2, -0.5], [1.0, 0.25]]]], dtype=np.float32)
    k_np = np.array([[[[0.0, 0.75], [-0.25, 0.5], [1.0, -1.0]]]], dtype=np.float32)
    v_np = np.array([[[[1.0, -1.0], [0.5, 2.0], [-0.25, 0.75]]]], dtype=np.float32)
    mask_np = np.array([[[[True, False, True], [True, True, False]]]], dtype=np.bool_)
    scale = 0.5
    logits = np.matmul(q_np, np.swapaxes(k_np, -1, -2)) * scale
    logits = np.where(mask_np, logits, np.float32(-1.0e9))
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    expected = np.matmul(weights, v_np)

    tensors = []
    try:
        q = device_memory.create_tensor(
            q_np.__array_interface__["data"][0], q_np.nbytes, q_np.shape, "float32"
        )
        k = device_memory.create_tensor(
            k_np.__array_interface__["data"][0], k_np.nbytes, k_np.shape, "float32"
        )
        v = device_memory.create_tensor(
            v_np.__array_interface__["data"][0], v_np.nbytes, v_np.shape, "float32"
        )
        mask = device_memory.create_tensor(
            mask_np.__array_interface__["data"][0],
            mask_np.nbytes,
            mask_np.shape,
            "bool",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [q, k, v, mask, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q_desc = graph.tensor(name="q", dim=q_np.shape, data_type=fe.DataType.FLOAT)
        k_desc = graph.tensor(name="k", dim=k_np.shape, data_type=fe.DataType.FLOAT)
        v_desc = graph.tensor(name="v", dim=v_np.shape, data_type=fe.DataType.FLOAT)
        mask_desc = graph.tensor(
            name="mask", dim=mask_np.shape, data_type=fe.DataType.BOOLEAN
        )
        o_desc, _ = graph.sdpa(
            q_desc,
            k_desc,
            v_desc,
            mask=mask_desc,
            name="attention",
            attn_scale=scale,
            is_inference=True,
        )

        graph.execute(
            {
                q_desc: device_memory.get_tensor_addr(q)[0],
                k_desc: device_memory.get_tensor_addr(k)[0],
                v_desc: device_memory.get_tensor_addr(v)[0],
                mask_desc: device_memory.get_tensor_addr(mask)[0],
                o_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=5.0e-4)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_sdpa_with_bias_and_bool_mask():
    import cudnn_frontend as fe

    q_np = np.array([[[[0.2, -0.5], [1.0, 0.25]]]], dtype=np.float32)
    k_np = np.array([[[[0.0, 0.75], [-0.25, 0.5], [1.0, -1.0]]]], dtype=np.float32)
    v_np = np.array([[[[1.0, -1.0], [0.5, 2.0], [-0.25, 0.75]]]], dtype=np.float32)
    bias_np = np.array([[[[0.0, -1.0, 0.25], [0.5, 0.0, -0.75]]]], dtype=np.float32)
    mask_np = np.array([[[[True, False, True], [True, True, False]]]], dtype=np.bool_)
    scale = 0.5
    expected = _sdpa_reference(q_np, k_np, v_np, scale, bias=bias_np, mask=mask_np)

    tensors = []
    try:
        q = device_memory.create_tensor(
            q_np.__array_interface__["data"][0], q_np.nbytes, q_np.shape, "float32"
        )
        k = device_memory.create_tensor(
            k_np.__array_interface__["data"][0], k_np.nbytes, k_np.shape, "float32"
        )
        v = device_memory.create_tensor(
            v_np.__array_interface__["data"][0], v_np.nbytes, v_np.shape, "float32"
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        mask = device_memory.create_tensor(
            mask_np.__array_interface__["data"][0],
            mask_np.nbytes,
            mask_np.shape,
            "bool",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [q, k, v, bias, mask, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q_desc = graph.tensor(name="q", dim=q_np.shape, data_type=fe.DataType.FLOAT)
        k_desc = graph.tensor(name="k", dim=k_np.shape, data_type=fe.DataType.FLOAT)
        v_desc = graph.tensor(name="v", dim=v_np.shape, data_type=fe.DataType.FLOAT)
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        mask_desc = graph.tensor(
            name="mask", dim=mask_np.shape, data_type=fe.DataType.BOOLEAN
        )
        o_desc, _ = graph.sdpa(
            q_desc,
            k_desc,
            v_desc,
            bias=bias_desc,
            mask=mask_desc,
            name="attention",
            attn_scale=scale,
            is_inference=True,
        )

        graph.execute(
            {
                q_desc: device_memory.get_tensor_addr(q)[0],
                k_desc: device_memory.get_tensor_addr(k)[0],
                v_desc: device_memory.get_tensor_addr(v)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                mask_desc: device_memory.get_tensor_addr(mask)[0],
                o_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=5.0e-4)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_sdpa_score_mod_document_mask_cpu():
    from functools import partial

    torch = pytest.importorskip("torch")
    import cudnn_frontend as fe

    def document_mask(graph, score, document_tensor, document_tensor_t, neg_inf):
        col_index = graph.gen_index(
            input=score,
            axis=3,
            name="col_index",
            compute_data_type=fe.DataType.INT32,
        )
        col_index.set_data_type(fe.DataType.INT32)
        all_ones = graph.cmp_le(
            input=col_index,
            comparison=col_index,
            compute_data_type=fe.DataType.FLOAT,
            name="all_ones",
        )
        all_ones.set_data_type(fe.DataType.INT32)
        row_bcast = graph.scale(
            input=all_ones,
            scale=document_tensor,
            compute_data_type=fe.DataType.FLOAT,
            name="row_bcast",
        )
        row_bcast.set_data_type(fe.DataType.INT32)
        col_bcast = graph.scale(
            input=all_ones,
            scale=document_tensor_t,
            compute_data_type=fe.DataType.FLOAT,
            name="col_bcast",
        )
        col_bcast.set_data_type(fe.DataType.INT32)
        same_document = graph.cmp_eq(
            input=row_bcast,
            comparison=col_bcast,
            compute_data_type=fe.DataType.FLOAT,
            name="same_document",
        )
        same_document.set_data_type(fe.DataType.INT32)
        return graph.binary_select(
            input0=score, input1=neg_inf, mask=same_document, name="masked_score"
        )

    q_data = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) / 11.0
    k_data = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) / 13.0 - 0.5
    v_data = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) / 17.0 - 1.0
    documents = torch.tensor([[[[0], [0], [1]]]], dtype=torch.int32)
    documents_t = documents.reshape(1, 1, 1, 3)
    neg_inf = torch.full((1, 1, 1, 1), -1.0e9, dtype=torch.float32)
    actual = torch.empty_like(q_data)
    scale = 0.5

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    q = graph.tensor_like(q_data)
    k = graph.tensor_like(k_data)
    v = graph.tensor_like(v_data)
    document_tensor = graph.tensor_like(documents)
    document_tensor_t = graph.tensor_like(documents_t)
    neg_inf_tensor = graph.tensor(
        name="neg_inf",
        dim=neg_inf.shape,
        stride=neg_inf.stride(),
        data_type=fe.DataType.FLOAT,
        is_pass_by_value=True,
    )
    o, _ = graph.sdpa(
        q,
        k,
        v,
        generate_stats=False,
        attn_scale=scale,
        score_mod=partial(
            document_mask,
            document_tensor=document_tensor,
            document_tensor_t=document_tensor_t,
            neg_inf=neg_inf_tensor,
        ),
        name="document_sdpa",
    )
    o.set_output(True).set_dim(q_data.shape).set_stride(q_data.stride())

    graph.validate().build_operation_graph().create_execution_plans([fe.heur_mode.A])
    graph.check_support().build_plans()
    graph.execute(
        {
            q: q_data,
            k: k_data,
            v: v_data,
            o: actual,
            document_tensor: documents,
            document_tensor_t: documents_t,
            neg_inf_tensor: neg_inf,
        }
    )

    scores = (q_data @ k_data.transpose(-2, -1)) * scale
    scores = torch.where(documents == documents_t, scores, neg_inf)
    expected = torch.softmax(scores, dim=-1) @ v_data
    torch.testing.assert_close(actual, expected)


def test_mudnn_frontend_python_graph_api_execute_sdpa_with_padding_mask():
    import cudnn_frontend as fe

    q_np = np.array([[[[0.2, -0.5], [1.0, 0.25]]]], dtype=np.float32)
    k_np = np.array([[[[0.0, 0.75], [-0.25, 0.5], [1.0, -1.0]]]], dtype=np.float32)
    v_np = np.array([[[[1.0, -1.0], [0.5, 2.0], [-0.25, 0.75]]]], dtype=np.float32)
    pad_mask_np = np.array([[True, False, True]], dtype=np.bool_)
    scale = 0.5
    logits = np.matmul(q_np, np.swapaxes(k_np, -1, -2)) * scale
    logits = np.where(pad_mask_np[:, None, None, :], logits, np.float32(-1.0e9))
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    expected = np.matmul(weights, v_np)

    tensors = []
    try:
        q = device_memory.create_tensor(
            q_np.__array_interface__["data"][0], q_np.nbytes, q_np.shape, "float32"
        )
        k = device_memory.create_tensor(
            k_np.__array_interface__["data"][0], k_np.nbytes, k_np.shape, "float32"
        )
        v = device_memory.create_tensor(
            v_np.__array_interface__["data"][0], v_np.nbytes, v_np.shape, "float32"
        )
        pad_mask = device_memory.create_tensor(
            pad_mask_np.__array_interface__["data"][0],
            pad_mask_np.nbytes,
            pad_mask_np.shape,
            "bool",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [q, k, v, pad_mask, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q_desc = graph.tensor(name="q", dim=q_np.shape, data_type=fe.DataType.FLOAT)
        k_desc = graph.tensor(name="k", dim=k_np.shape, data_type=fe.DataType.FLOAT)
        v_desc = graph.tensor(name="v", dim=v_np.shape, data_type=fe.DataType.FLOAT)
        pad_mask_desc = graph.tensor(
            name="pad_mask", dim=pad_mask_np.shape, data_type=fe.DataType.BOOLEAN
        )
        o_desc, _ = graph.sdpa(
            q_desc,
            k_desc,
            v_desc,
            mask=pad_mask_desc,
            name="attention",
            attn_scale=scale,
            is_inference=True,
        )

        graph.execute(
            {
                q_desc: device_memory.get_tensor_addr(q)[0],
                k_desc: device_memory.get_tensor_addr(k)[0],
                v_desc: device_memory.get_tensor_addr(v)[0],
                pad_mask_desc: device_memory.get_tensor_addr(pad_mask)[0],
                o_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=5.0e-4)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_sdpa_with_bias_and_additive_mask():
    import cudnn_frontend as fe

    q_np = np.array([[[[0.2, -0.5], [1.0, 0.25]]]], dtype=np.float32)
    k_np = np.array([[[[0.0, 0.75], [-0.25, 0.5], [1.0, -1.0]]]], dtype=np.float32)
    v_np = np.array([[[[1.0, -1.0], [0.5, 2.0], [-0.25, 0.75]]]], dtype=np.float32)
    bias_np = np.array([[[[0.0, -1.0, 0.25], [0.5, 0.0, -0.75]]]], dtype=np.float32)
    mask_np = np.array([[[[0.0, -0.5, 0.0], [-1.0, 0.0, -0.25]]]], dtype=np.float32)
    scale = 0.5
    logits = np.matmul(q_np, np.swapaxes(k_np, -1, -2)) * scale + bias_np + mask_np
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    expected = np.matmul(weights, v_np)

    tensors = []
    try:
        q = device_memory.create_tensor(
            q_np.__array_interface__["data"][0], q_np.nbytes, q_np.shape, "float32"
        )
        k = device_memory.create_tensor(
            k_np.__array_interface__["data"][0], k_np.nbytes, k_np.shape, "float32"
        )
        v = device_memory.create_tensor(
            v_np.__array_interface__["data"][0], v_np.nbytes, v_np.shape, "float32"
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        mask = device_memory.create_tensor(
            mask_np.__array_interface__["data"][0],
            mask_np.nbytes,
            mask_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [q, k, v, bias, mask, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        q_desc = graph.tensor(name="q", dim=q_np.shape, data_type=fe.DataType.FLOAT)
        k_desc = graph.tensor(name="k", dim=k_np.shape, data_type=fe.DataType.FLOAT)
        v_desc = graph.tensor(name="v", dim=v_np.shape, data_type=fe.DataType.FLOAT)
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        mask_desc = graph.tensor(
            name="mask", dim=mask_np.shape, data_type=fe.DataType.FLOAT
        )
        o_desc, _ = graph.sdpa(
            q_desc,
            k_desc,
            v_desc,
            bias=bias_desc,
            mask=mask_desc,
            name="attention",
            attn_scale=scale,
            is_inference=True,
        )
        workspace_size = graph.get_workspace_size()
        workspace = (
            device_memory.create_tensor(0, workspace_size, (workspace_size,), "uint8")
            if workspace_size
            else None
        )
        if workspace is not None:
            tensors.append(workspace)
        graph.execute(
            {
                q_desc: device_memory.get_tensor_addr(q)[0],
                k_desc: device_memory.get_tensor_addr(k)[0],
                v_desc: device_memory.get_tensor_addr(v)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                mask_desc: device_memory.get_tensor_addr(mask)[0],
                o_desc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=(
                device_memory.get_tensor_addr(workspace)[0]
                if workspace is not None
                else None
            ),
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=5.0e-4, atol=5.0e-4)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_native_matmul_relu_lowering():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        intermediate_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    a = graph.tensor(name="a", dim=(2, 8, 16), data_type=fe.data_type.FLOAT)
    b = graph.tensor(name="b", dim=(2, 16, 32), data_type=fe.data_type.FLOAT)
    bias = graph.tensor(name="bias", dim=(2, 8, 32), data_type=fe.data_type.FLOAT)
    c = graph.matmul(a, b, name="gemm")
    biased = graph.bias(c, bias, name="gemm_bias")
    relu = graph.relu(biased, name="gemm_relu")
    graph.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    ).check_support().build_plans()

    assert relu.dims == (2, 8, 32)
    assert graph.execution_plan_modes == (fe.heur_mode.INSTANT,)
    assert graph._native.operation_graph.ptr.value is not None
    assert len(graph._native.plans) == 1
    assert graph.get_execution_plan_count() >= 1
    assert graph.get_workspace_size() >= 0
    with pytest.raises(ValueError, match="missing data pointers"):
        graph.execute({})


def test_mudnn_frontend_python_graph_matmul_broadcast_batch_dims():
    import mudnn

    graph = mudnn.pygraph(
        name="cudnn_graph_0",
        intermediate_data_type=mudnn.data_type.FLOAT,
        compute_data_type=mudnn.data_type.FLOAT,
    )
    x = graph.tensor(
        name="X",
        dim=(1, 64, 64),
        stride=(64 * 64, 64, 1),
        data_type=mudnn.data_type.FP8_E4M3,
    )
    w = graph.tensor(
        name="W",
        dim=(2, 64, 64),
        stride=(64 * 64, 64, 1),
        data_type=mudnn.data_type.FP8_E4M3,
    )
    c = graph.matmul(x, w)

    assert c.get_dim() == (2, 64, 64)
    assert c.get_stride() == (64 * 64, 64, 1)
    mask = graph.tensor(
        name="mask",
        dim=(2, 1, 1),
        stride=(1, 1, 1),
        data_type=mudnn.data_type.INT32,
    )
    selected = graph.binary_select(c, c, mask)
    assert selected.get_dim() == (2, 64, 64)

    bad = graph.tensor(
        name="bad",
        dim=(3, 64, 64),
        stride=(64 * 64, 64, 1),
        data_type=mudnn.data_type.FP8_E4M3,
    )
    with pytest.raises(ValueError, match="not broadcastable"):
        graph.matmul(bad, w)


def test_mudnn_frontend_python_graph_fp8_silu_mul_reduction_build_unsupported():
    import mudnn

    graph = mudnn.pygraph(
        intermediate_data_type=mudnn.data_type.FLOAT,
        compute_data_type=mudnn.data_type.FLOAT,
    )
    x = graph.tensor(
        name="X",
        dim=(1, 64, 64),
        stride=(64 * 64, 64, 1),
        data_type=mudnn.data_type.FP8_E4M3,
    )
    w = graph.tensor(
        name="W",
        dim=(2, 64, 64),
        stride=(64 * 64, 64, 1),
        data_type=mudnn.data_type.FP8_E4M3,
    )
    c0 = graph.matmul(x, w)
    x_scale = graph.tensor_scalar(0.5, data_type=mudnn.data_type.FLOAT, name="X_DQ")
    w_scale = graph.tensor_scalar(0.5, data_type=mudnn.data_type.FLOAT, name="W_DQ")
    c2 = graph.mul(graph.mul(c0, x_scale), w_scale)
    silu = graph.mul(graph.sigmoid(c2), c2)
    mask = graph.tensor(
        name="mask",
        dim=(2, 1, 1),
        stride=(1, 1, 1),
        data_type=mudnn.data_type.INT32,
    )
    selected = graph.binary_select(c2, silu, mask)
    out = graph.reduction(selected, mode=mudnn.reduction_mode.MUL)
    out.set_dim([1, 64, 64]).set_stride([64 * 64, 64, 1]).set_output(True)
    out.set_data_type(mudnn.data_type.FLOAT)

    with pytest.raises(mudnn.cudnnGraphNotSupportedError, match="NOT_SUPPORTED"):
        graph.build([mudnn.heur_mode.A, mudnn.heur_mode.FALLBACK])


def test_mudnn_frontend_python_graph_matmul_float_accumulator_bias_build():
    import mudnn

    batch, seq, hidden = 2, 8, 16
    graph = mudnn.pygraph(
        intermediate_data_type=mudnn.data_type.FLOAT,
        compute_data_type=mudnn.data_type.FLOAT,
    )
    x = graph.tensor(
        name="X",
        dim=(batch, seq, hidden),
        stride=(seq * hidden, hidden, 1),
        data_type=mudnn.data_type.HALF,
    )
    weight = graph.tensor(
        name="W",
        dim=(1, hidden, hidden * 4),
        stride=(hidden * hidden * 4, hidden * 4, 1),
        data_type=mudnn.data_type.HALF,
    )
    bias = graph.tensor(
        name="B",
        dim=(1, 1, hidden * 4),
        stride=(hidden * 4, hidden * 4, 1),
        data_type=mudnn.data_type.HALF,
    )

    matmul = graph.matmul(name="matmul", A=x, B=weight)
    y = graph.bias(name="bias", input=matmul, bias=bias)
    y.set_output(True).set_data_type(mudnn.data_type.HALF)

    assert matmul.get_dim() == (batch, seq, hidden * 4)
    assert matmul.get_data_type() == mudnn.data_type.FLOAT
    graph.validate().build_operation_graph().create_execution_plans(
        [mudnn.heur_mode.A, mudnn.heur_mode.FALLBACK]
    )
    assert graph.get_execution_plan_count() >= 1
    assert mudnn.behavior_note.RUNTIME_COMPILATION in (
        graph.get_behavior_notes_for_plan_at_index(0)
    )
    graph.check_support().build_plans()
    assert mudnn.behavior_note.RUNTIME_COMPILATION in graph.get_behavior_notes()


def test_mudnn_frontend_python_graph_api_build_and_execute_aliases():
    import cudnn_frontend as fe

    a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    b_np = np.full((2, 3), 4.0, dtype=np.float32)
    expected = a_np + b_np

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [a, b, out]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.add(a_desc, b_desc, name="y").set_output()
        graph.build([fe.HeurMode.INSTANT])
        assert graph._prepare_variant_pack_template() is graph
        assert graph._get_variant_pack_uids_sorted() == [
            a_desc.uid,
            b_desc.uid,
            y_desc.uid,
        ]
        assert graph.query_tensor_attributes_of_uid(a_desc.uid) is a_desc
        assert fe._get_data_ptr(_FakeDataPtr(0x1234)) == 0x1234

        graph._execute(
            {
                "a": device_memory.get_tensor_addr(a)[0],
                "b": device_memory.get_tensor_addr(b)[0],
                "y": device_memory.get_tensor_addr(out)[0],
            },
            workspace=None,
            handle=None,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)

        graph.execute(
            {
                a_desc: device_memory.get_tensor_addr(a)[0],
                b_desc: device_memory.get_tensor_addr(b)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            },
            None,
            None,
        )
        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_plan_index_methods():
    import mudnn
    import cudnn_frontend as fe

    x_np = np.array([[1.0, -2.0, 3.0], [0.25, 4.0, -5.0]], dtype=np.float32)
    expected = x_np + x_np

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.add(x_desc, x_desc, name="y")
        graph.validate().build_operation_graph().create_execution_plans(
            [fe.HeurMode.INSTANT]
        ).check_support()

        assert graph.get_execution_plan_count() >= 1
        assert graph.get_engine_count() == graph.get_execution_plan_count()
        assert graph.get_knobs_for_engine(0) == []
        assert list(graph.get_knobs_for_engine(0)) == []
        assert graph.get_engine_and_knobs_at_index(0) == (0, {})
        assert graph.get_plan_name_at_index(0) == "mudnn_engine_config_0"
        assert graph.get_behavior_notes() == []
        assert graph.get_behavior_notes_for_plan_at_index(0) == []
        assert graph.select_numeric_notes([]) is graph
        assert graph.select_behavior_notes([]) is graph
        assert graph.deselect_numeric_notes([]) is graph
        assert graph.deselect_behavior_notes([]) is graph
        assert graph.deselect_engines([]) is graph
        assert graph.deselect_workspace_greater_than(1 << 30) is graph
        assert graph.create_execution_plan(0) is graph
        assert graph.build_plan_at_index(0) is graph
        assert graph.get_workspace_size_plan_at_index(0) == graph.get_workspace_size()
        assert graph._get_variant_pack_uids_sorted() == [x_desc.uid, y_desc.uid]
        assert graph.query_tensor_attributes_of_uid(y_desc.uid) is y_desc
        assert isinstance(graph.key(), bytes)
        assert '"op_type"' in repr(graph)
        assert (
            graph.execute_plan_at_index(
                {
                    x_desc: device_memory.get_tensor_addr(x)[0],
                    y_desc: device_memory.get_tensor_addr(out)[0],
                },
                index=0,
            )
            is graph
        )
        ptrs = [
            device_memory.get_tensor_addr(x)[0],
            device_memory.get_tensor_addr(out)[0],
        ]
        assert graph._execute_with_ptrs(ptrs, workspace=None, handle=None) is graph
        raw_ptrs = (ctypes.c_void_p * len(ptrs))(*ptrs)
        assert (
            graph._execute_with_raw_ptrs(
                ctypes.addressof(raw_ptrs), len(ptrs), workspace=None, handle=None
            )
            is graph
        )
        assert (
            graph._execute_plan_at_index(
                {
                    x_desc: device_memory.get_tensor_addr(x)[0],
                    y_desc: device_memory.get_tensor_addr(out)[0],
                },
                workspace=None,
                index=0,
                handle=None,
            )
            is graph
        )
        with pytest.raises(IndexError, match="out of range"):
            graph.build_plan_at_index(graph.get_execution_plan_count())
        with pytest.raises(IndexError, match="out of range"):
            graph.get_plan_name_at_index(graph.get_execution_plan_count())
        with pytest.raises(IndexError, match="out of range"):
            graph.get_knobs_for_engine(graph.get_execution_plan_count())
        with pytest.raises(KeyError, match="tensor uid"):
            graph.query_tensor_attributes_of_uid(999_999)
        with pytest.raises(NotImplementedError, match="custom execution-plan knobs"):
            graph.create_execution_plan(0, {mudnn.knob_type.KERNEL_CFG: 0})
        override_uids = [x_desc.uid, y_desc.uid]
        override_shapes = [x_np.shape, expected.shape]
        override_strides = [x_np.strides, expected.strides]
        assert (
            graph.get_workspace_size_plan_at_index(
                0,
                override_uids=override_uids,
                override_shapes=override_shapes,
                override_strides=override_strides,
            )
            == graph.get_workspace_size()
        )
        assert (
            graph.execute(
                {
                    x_desc: device_memory.get_tensor_addr(x)[0],
                    y_desc: device_memory.get_tensor_addr(out)[0],
                },
                override_uids=override_uids,
                override_shapes=override_shapes,
                override_strides=override_strides,
            )
            is graph
        )
        assert (
            graph.execute_plan_at_index(
                {
                    x_desc: device_memory.get_tensor_addr(x)[0],
                    y_desc: device_memory.get_tensor_addr(out)[0],
                },
                index=0,
                override_uids=override_uids,
                override_shapes=override_shapes,
                override_strides=override_strides,
            )
            is graph
        )
        with pytest.raises(ValueError, match="require override_uids"):
            graph.get_workspace_size_plan_at_index(
                0,
                override_uids=[x_desc.uid],
            )
        with pytest.raises(ValueError, match="lengths"):
            graph.execute(
                {
                    x_desc: device_memory.get_tensor_addr(x)[0],
                    y_desc: device_memory.get_tensor_addr(out)[0],
                },
                override_uids=[x_desc.uid],
                override_shapes=[x_np.shape],
                override_strides=[],
            )
        with pytest.raises(KeyError, match="tensor uid"):
            graph.execute_plan_at_index(
                {
                    x_desc: device_memory.get_tensor_addr(x)[0],
                    y_desc: device_memory.get_tensor_addr(out)[0],
                },
                index=0,
                override_uids=[999_999],
                override_shapes=[x_np.shape],
                override_strides=[x_np.strides],
            )
        with pytest.raises(ValueError, match="rank"):
            graph.execute(
                {
                    x_desc: device_memory.get_tensor_addr(x)[0],
                    y_desc: device_memory.get_tensor_addr(out)[0],
                },
                override_uids=[x_desc.uid],
                override_shapes=[(x_np.size,)],
                override_strides=[(1,)],
            )
        with pytest.raises(NotImplementedError, match="CUDA graph"):
            graph.populate_cuda_graph(None, {}, None, None)
        with pytest.raises(NotImplementedError, match="CUDA graph"):
            graph.update_cuda_graph(None, {}, None, None)

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_context_io_tuples():
    import cudnn_frontend as fe

    with fe.Graph(
        io_data_type=fe.DataType.FLOAT,
        inputs=["x", "bias"],
        outputs=["y"],
    ) as graph:
        x = graph.tensor(name="x", dim=(2, 3), data_type=fe.DataType.FLOAT)
        bias = graph.tensor(name="bias", dim=(2, 3), data_type=fe.DataType.FLOAT)
        y = graph.add(x, bias, name="add")
        y.set_name("y").set_output(True)

    assert graph._plans_built is True
    assert [tensor.name for tensor in graph._input_tuples] == ["x", "bias"]
    assert [tensor.name for tensor in graph._output_tuples] == ["y"]


def test_mudnn_frontend_python_cudnn_wrapper_graph_context_and_call(monkeypatch):
    torch = pytest.importorskip("torch")
    import mudnn
    from mudnn.wrapper import Graph as WrapperGraph

    monkeypatch.setattr(mudnn.Graph, "build_operation_graph", lambda self: self)
    monkeypatch.setattr(mudnn.Graph, "check_support", lambda self: self)
    monkeypatch.setattr(mudnn.Graph, "build_plans", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(mudnn.Graph, "get_workspace_size", lambda self: 0)
    monkeypatch.setattr(mudnn.Graph, "get_execution_plan_count", lambda self: 1)

    captured = {}

    def fake_execute(self, variant_pack, **kwargs):
        captured["variant_pack"] = dict(variant_pack)
        captured["kwargs"] = dict(kwargs)
        return self

    monkeypatch.setattr(mudnn.Graph, "execute", fake_execute)

    x_tensor = torch.empty((2, 3), dtype=torch.float32)
    bias_tensor = torch.empty((2, 3), dtype=torch.float32)
    with WrapperGraph(io_data_type=torch.float32, workspace_alloc=False) as graph:
        y = graph.add(x_tensor, bias_tensor, name="add")
        y.set_output(True)

    graph.set_io_tuples([x_tensor, bias_tensor], [y])
    output = graph(x_tensor, bias_tensor, workspace=0)
    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (2, 3)
    assert output.dtype == torch.float32
    assert set(captured["variant_pack"].values()) == {x_tensor, bias_tensor, output}
    assert captured["kwargs"]["workspace"] == 0

    tensor_dict = {"add::0": x_tensor, "add::1": bias_tensor}
    returned = graph(tensor_dict, workspace=0)
    assert returned is tensor_dict
    assert "add" in returned
    assert isinstance(returned["add"], torch.Tensor)


def test_mudnn_frontend_python_graph_api_set_io_tuples_and_positional_call():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    x = graph.tensor(name="x", dim=(2, 3), data_type=fe.DataType.FLOAT)
    bias = graph.tensor(name="bias", dim=(2, 3), data_type=fe.DataType.FLOAT)
    y = graph.add(x, bias, name="add")
    y.set_name("y").set_output(True)

    graph.set_io_tuples(["add::x", bias], ["add::y"])
    assert [tensor.uid for tensor in graph._input_tuples] == [x.uid, bias.uid]
    assert [tensor.uid for tensor in graph._output_tuples] == [y.uid]
    assert x.get_uid() == x.uid
    assert x.get_name() == "x"
    assert x.get_is_virtual() is False

    captured = {}

    def fake_execute(variant_pack, **kwargs):
        captured["variant_pack"] = dict(variant_pack)
        captured["kwargs"] = dict(kwargs)
        return graph

    graph.execute = fake_execute
    assert graph(101, 202, output_ptrs=303, workspace=404) is graph
    assert captured["variant_pack"] == {x.uid: 101, bias.uid: 202, y.uid: 303}
    assert captured["kwargs"] == {"workspace": 404}

    with pytest.raises(ValueError, match="Number of arguments"):
        graph(101, output_ptrs=303)
    with pytest.raises(ValueError, match="requires tensor inputs"):
        graph(101, 202)

    torch = pytest.importorskip("torch")
    captured.clear()
    x_tensor = torch.empty((2, 3), dtype=torch.float32)
    bias_tensor = torch.empty((2, 3), dtype=torch.float32)
    auto_output = graph(x_tensor, bias_tensor)
    assert isinstance(auto_output, torch.Tensor)
    assert tuple(auto_output.shape) == (2, 3)
    assert auto_output.dtype == torch.float32
    assert auto_output.device == x_tensor.device
    assert captured["variant_pack"] == {
        x.uid: x_tensor,
        bias.uid: bias_tensor,
        y.uid: auto_output,
    }

    with pytest.raises(ValueError, match="duplicate"):
        graph.set_io_tuples(["x", x], ["y"])
    with pytest.raises(ValueError, match="not found"):
        graph.set_io_tuples(["missing"], ["y"])


def test_mudnn_frontend_python_graph_api_explicit_intermediate_output_is_external():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    a = graph.tensor(name="a", dim=(2, 3), data_type=fe.DataType.FLOAT)
    b = graph.scale(a, 2.0, name="b").set_output()
    c = graph.relu(b, name="c")

    with pytest.raises(ValueError, match=str(b.uid)):
        graph._normalize_data_ptrs({a: 1, c: 3})


def test_mudnn_frontend_python_graph_api_unconsumed_aux_output_is_internal():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    x = graph.tensor(name="x", dim=(2, 3, 4), data_type=fe.DataType.FLOAT)
    sum_tensor, sqsum_tensor = graph.genstats(x, axis=(0, 2), name="stats")
    sum_tensor.set_name("sum").set_output(True)

    graph.set_io_tuples(["x"], ["sum"])
    assert graph._get_variant_pack_uids_sorted() == [x.uid, sum_tensor.uid]
    assert sqsum_tensor.uid not in graph._external_tensor_uids()


def test_mudnn_frontend_python_graph_api_variant_pack_pointer_objects():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
    a = graph.tensor(name="a", dim=(2, 3), data_type=fe.DataType.FLOAT)
    b = graph.tensor(name="b", dim=(2, 3), data_type=fe.DataType.FLOAT)
    y = graph.add(a, b, name="y").set_output()

    host = np.zeros((2, 3), dtype=np.float32)
    normalized = graph._normalize_data_ptrs(
        {
            a: _FakeDataPtr(0x1000),
            "b": _FakeCudaArray(0x2000),
            y.uid: ctypes.c_void_p(host.__array_interface__["data"][0]),
        }
    )

    assert normalized[a.uid] == 0x1000
    assert normalized[b.uid] == 0x2000
    assert normalized[y.uid] == host.__array_interface__["data"][0]
    with pytest.raises(ValueError, match="null data pointer"):
        graph._normalize_data_ptrs({a: ctypes.c_void_p(), b: 0x2, y: 0x3})


def test_mudnn_frontend_python_graph_api_pointwise_convenience_methods():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    a = graph.tensor(name="a", dim=(2, 3), data_type=fe.data_type.FLOAT)
    b = graph.tensor(name="b", dim=(2, 3), data_type=fe.data_type.FLOAT)

    add = graph.add(a, b, name="add")
    sub = graph.sub(add, b, name="sub")
    mul = graph.mul(sub, b, name="mul")
    div = graph.div(mul, b, rounding_mode=None, name="div")
    sqrt = graph.sqrt(div, name="sqrt")
    rsqrt = graph.rsqrt(sqrt, name="rsqrt")
    exp = graph.exp(rsqrt, name="exp")
    log = graph.log(exp, name="log")
    tanh = graph.tanh(log, name="tanh")
    sigmoid = graph.sigmoid(tanh, name="sigmoid")
    gelu = graph.gelu(sigmoid, name="gelu")
    gelu_approx = graph.gelu_approx_tanh(gelu, name="gelu_approx")
    scaled = graph.scale(gelu_approx, 2.5, name="scaled")
    gt = graph.gt(a, b, name="gt")
    le = graph.le(a, b, name="le")
    pred = graph.logical_or(gt, le, name="pred")
    not_pred = graph.logical_not(pred, name="not_pred")
    selected = graph.select(not_pred, a, b, name="selected")
    identity = graph.identity(input=selected, compute_data_type=fe.data_type.FLOAT)
    cmp_gt = graph.cmp_gt(input=a, comparison=b, name="cmp_gt")
    binary_selected = graph.binary_select(
        input0=identity, input1=b, mask=cmp_gt, name="binary_selected"
    )
    swish = graph.swish(input=binary_selected, swish_beta=1.25, name="swish")
    relu_bwd = graph.relu_backward(loss=swish, input=binary_selected, name="relu_bwd")
    swish_bwd = graph.swish_backward(loss=relu_bwd, input=swish, name="swish_bwd")

    assert [node.op_type for node in graph.nodes] == [
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
    ]
    assert graph.nodes[0].attributes["mode"] == fe.pointwise_mode.ADD
    assert graph.nodes[1].attributes["mode"] == fe.pointwise_mode.SUB
    assert graph.nodes[2].attributes["mode"] == fe.pointwise_mode.MUL
    assert graph.nodes[3].attributes["mode"] == fe.pointwise_mode.DIV
    floor_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    floor_a = floor_graph.tensor(name="a", dim=(2, 3), data_type=fe.data_type.FLOAT)
    floor_b = floor_graph.tensor(name="b", dim=(2, 3), data_type=fe.data_type.FLOAT)
    floor_div = floor_graph.div(
        floor_a, floor_b, rounding_mode="floor", name="floor_div"
    )
    assert floor_div.name == "floor_div"
    assert [node.attributes["mode"] for node in floor_graph.nodes] == [
        fe.pointwise_mode.DIV,
        fe.pointwise_mode.FLOOR,
    ]
    assert floor_graph.nodes[0].outputs["y"].name == "floor_div_quotient"
    trunc_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    trunc_a = trunc_graph.tensor(name="a", dim=(2, 3), data_type=fe.data_type.FLOAT)
    trunc_b = trunc_graph.tensor(name="b", dim=(2, 3), data_type=fe.data_type.FLOAT)
    trunc_div = trunc_graph.div(
        trunc_a, trunc_b, rounding_mode="trunc", name="trunc_div"
    )
    assert trunc_div.name == "trunc_div"
    assert [node.attributes["mode"] for node in trunc_graph.nodes] == [
        fe.pointwise_mode.DIV,
        fe.pointwise_mode.FLOOR,
        fe.pointwise_mode.CEIL,
        fe.pointwise_mode.CMP_GE,
        fe.pointwise_mode.BINARY_SELECT,
    ]
    with pytest.raises(NotImplementedError, match="rounding_mode"):
        trunc_graph.div(trunc_a, trunc_b, rounding_mode="round")
    assert graph.nodes[4].attributes["mode"] == fe.pointwise_mode.SQRT
    assert graph.nodes[5].attributes["mode"] == fe.pointwise_mode.RSQRT
    assert graph.nodes[6].attributes["mode"] == fe.pointwise_mode.EXP
    assert graph.nodes[7].attributes["mode"] == fe.pointwise_mode.LOG
    assert graph.nodes[8].attributes["mode"] == fe.pointwise_mode.TANH_FWD
    assert graph.nodes[9].attributes["mode"] == fe.pointwise_mode.SIGMOID_FWD
    assert graph.nodes[10].attributes["mode"] == fe.pointwise_mode.GELU_FWD
    assert graph.nodes[11].attributes["mode"] == fe.pointwise_mode.GELU_APPROX_TANH_FWD
    assert graph.nodes[12].attributes["mode"] == fe.pointwise_mode.IDENTITY
    assert graph.nodes[12].attributes["alpha1"] == 2.5
    assert graph.nodes[13].attributes["mode"] == fe.pointwise_mode.CMP_GT
    assert graph.nodes[14].attributes["mode"] == fe.pointwise_mode.CMP_LE
    assert graph.nodes[15].attributes["mode"] == fe.pointwise_mode.LOGICAL_OR
    assert graph.nodes[16].attributes["mode"] == fe.pointwise_mode.LOGICAL_NOT
    assert graph.nodes[17].attributes["mode"] == fe.pointwise_mode.BINARY_SELECT
    assert graph.nodes[18].attributes["mode"] == fe.pointwise_mode.IDENTITY
    assert graph.nodes[19].attributes["mode"] == fe.pointwise_mode.CMP_GT
    assert graph.nodes[20].attributes["mode"] == fe.pointwise_mode.BINARY_SELECT
    assert graph.nodes[21].attributes["mode"] == fe.pointwise_mode.SWISH_FWD
    assert graph.nodes[21].attributes["swish_beta"] == 1.25
    assert graph.nodes[22].attributes["mode"] == fe.pointwise_mode.RELU_BWD
    assert graph.nodes[23].attributes["mode"] == fe.pointwise_mode.SWISH_BWD
    assert gt.data_type == fe.data_type.BOOLEAN
    assert cmp_gt.data_type == fe.data_type.BOOLEAN
    assert selected.data_type == fe.data_type.FLOAT
    assert binary_selected.data_type == fe.data_type.FLOAT
    assert swish.data_type == fe.data_type.FLOAT
    assert relu_bwd.data_type == fe.data_type.FLOAT
    assert swish_bwd.data_type == fe.data_type.FLOAT
    assert scaled.dims == (2, 3)
    assert swish_bwd.dims == (2, 3)


def test_mudnn_frontend_python_graph_api_cudnn_pointwise_aliases():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT, handle=object())
    x = graph.tensor(name="x", dim=(2, 3, 4), data_type=fe.data_type.FLOAT)
    y = graph.tensor(name="y", dim=(2, 3, 4), data_type=fe.data_type.FLOAT)

    index = graph.gen_index(input=x, axis=1, name="row_index")
    cmp_eq = graph.cmp_eq(input=index, comparison=index, name="cmp_eq")
    cmp_neq = graph.cmp_neq(input=x, comparison=y, name="cmp_neq")
    cmp_ge = graph.cmp_ge(
        input=x, comparison=y, compute_data_type=fe.data_type.BOOLEAN, name="cmp_ge"
    )
    cmp_lt = graph.cmp_lt(input=x, comparison=y, name="cmp_lt")
    cmp_le = graph.cmp_le(input=x, comparison=y, name="cmp_le")
    pointwise_outputs = [
        graph.max(input0=x, input1=y, compute_data_type=fe.data_type.FLOAT),
        graph.min(input0=x, input1=y),
        graph.mod(input0=x, input1=y),
        graph.pow(input0=x, input1=y),
        graph.add_square(x, y),
        graph.abs(input=x),
        graph.ceil(input=x),
        graph.floor(input=x),
        graph.tan(input=x),
        graph.erf(input=x),
        graph.reciprocal(input=x),
    ]
    activation_outputs = [
        graph.leaky_relu(input=x, negative_slope=0.125, name="leaky_relu"),
        graph.elu(input=x, elu_alpha=0.75, name="elu"),
        graph.softplus(input=x, softplus_beta=1.5, name="softplus"),
        graph.tanh_backward(y, x, name="tanh_backward"),
        graph.sigmoid_backward(y, x, name="sigmoid_backward"),
        graph.elu_backward(y, x, elu_alpha=0.75, name="elu_backward"),
        graph.gelu_backward(y, x, name="gelu_backward"),
        graph.softplus_backward(y, x, softplus_beta=1.5, name="softplus_backward"),
        graph.gelu_approx_tanh_backward(y, x, name="gelu_approx_tanh_backward"),
        graph.leaky_relu_backward(y, x, 0.125, name="leaky_relu_backward"),
    ]

    assert graph.handle is not None
    assert index.dims == x.dims
    assert index.data_type == fe.data_type.INT32
    assert graph.nodes[0].attributes["mode"] == fe.pointwise_mode.GEN_INDEX
    assert graph.nodes[0].attributes["axis"] == 1
    assert graph.nodes[1].attributes["mode"] == fe.pointwise_mode.CMP_EQ
    assert graph.nodes[2].attributes["mode"] == fe.pointwise_mode.CMP_NEQ
    assert graph.nodes[3].attributes["mode"] == fe.pointwise_mode.CMP_GE
    assert graph.nodes[4].attributes["mode"] == fe.pointwise_mode.CMP_LT
    assert graph.nodes[5].attributes["mode"] == fe.pointwise_mode.CMP_LE
    assert [node.attributes["mode"] for node in graph.nodes[6:17]] == [
        fe.pointwise_mode.MAX,
        fe.pointwise_mode.MIN,
        fe.pointwise_mode.MOD,
        fe.pointwise_mode.POW,
        fe.pointwise_mode.ADD_SQUARE,
        fe.pointwise_mode.ABS,
        fe.pointwise_mode.CEIL,
        fe.pointwise_mode.FLOOR,
        fe.pointwise_mode.TAN,
        fe.pointwise_mode.ERF,
        fe.pointwise_mode.RECIPROCAL,
    ]
    assert [node.attributes["mode"] for node in graph.nodes[17:]] == [
        fe.pointwise_mode.RELU_FWD,
        fe.pointwise_mode.ELU_FWD,
        fe.pointwise_mode.SOFTPLUS_FWD,
        fe.pointwise_mode.TANH_BWD,
        fe.pointwise_mode.SIGMOID_BWD,
        fe.pointwise_mode.ELU_BWD,
        fe.pointwise_mode.GELU_BWD,
        fe.pointwise_mode.SOFTPLUS_BWD,
        fe.pointwise_mode.GELU_APPROX_TANH_BWD,
        fe.pointwise_mode.RELU_BWD,
    ]
    assert graph.nodes[17].attributes["relu_lower_clip_slope"] == 0.125
    assert graph.nodes[18].attributes["elu_alpha"] == 0.75
    assert graph.nodes[19].attributes["softplus_beta"] == 1.5
    assert graph.nodes[22].attributes["elu_alpha"] == 0.75
    assert graph.nodes[24].attributes["softplus_beta"] == 1.5
    assert graph.nodes[26].attributes["relu_lower_clip_slope"] == 0.125
    assert cmp_eq.data_type == fe.data_type.BOOLEAN
    assert cmp_neq.data_type == fe.data_type.BOOLEAN
    assert cmp_ge.data_type == fe.data_type.BOOLEAN
    assert cmp_lt.data_type == fe.data_type.BOOLEAN
    assert cmp_le.data_type == fe.data_type.BOOLEAN
    assert all(tensor.dims == x.dims for tensor in pointwise_outputs)
    assert all(tensor.dims == x.dims for tensor in activation_outputs)


def test_mudnn_frontend_python_graph_api_softmax_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    p = graph.tensor(name="p", dim=(2, 3, 4), data_type=fe.data_type.FLOAT)
    sink = graph.tensor(name="sink", dim=(2, 3, 1), data_type=fe.data_type.FLOAT)

    s, stats, max_output, sum_exp = graph.softmax(
        p,
        dim=-1,
        sink=sink,
        name="softmax",
        generate_stats=True,
        return_max_sum_exp=True,
    )

    assert s.dims == p.dims
    assert stats.dims == (2, 3, 1)
    assert max_output.dims == (2, 3, 1)
    assert sum_exp.dims == (2, 3, 1)
    assert [node.op_type for node in graph.nodes] == [
        "reduction",
        "pointwise",
        "pointwise",
        "pointwise",
        "reduction",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
        "pointwise",
    ]
    assert graph.nodes[0].attributes["mode"] == fe.reduction_mode.MAX
    assert graph.nodes[1].attributes["mode"] == fe.pointwise_mode.MAX
    assert graph.nodes[3].attributes["mode"] == fe.pointwise_mode.EXP
    assert graph.nodes[4].attributes["mode"] == fe.reduction_mode.ADD
    assert graph.nodes[-2].attributes["mode"] == fe.pointwise_mode.LOG
    assert graph.nodes[-1].attributes["mode"] == fe.pointwise_mode.ADD

    restored = fe.pygraph().deserialize(graph.serialize())
    assert restored.nodes[-1].name == "softmax_stats"
    assert restored.nodes[-1].outputs["y"].dims == (2, 3, 1)
    with pytest.raises(ValueError, match="conflict"):
        graph.softmax(p, axis=1, dim=-1)


def test_mudnn_frontend_python_graph_api_execute_gen_index():
    import cudnn_frontend as fe

    x_np = np.zeros((2, 3, 4), dtype=np.float32)
    expected = np.broadcast_to(
        np.arange(x_np.shape[1], dtype=np.int32).reshape(1, 3, 1), x_np.shape
    )

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(0, expected.nbytes, expected.shape, "int32")
        tensors = [x, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.gen_index(input=x_desc, axis=1, name="idx")
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_array_equal(actual, expected)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_pointwise_broadcast_shape():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 3), data_type=fe.data_type.FLOAT)
    b = graph.tensor(name="b", dim=(1, 3), data_type=fe.data_type.FLOAT)
    y = graph.add(x, b, name="broadcast_add")

    assert y.dims == (2, 3)
    assert y.strides == (3, 1)
    with pytest.raises(ValueError, match="not broadcastable"):
        graph.add(x, graph.tensor(name="bad", dim=(2, 4), data_type=fe.data_type.FLOAT))


def test_mudnn_frontend_python_graph_api_reduction_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = graph.tensor(name="x", dim=(2, 3, 4), data_type=fe.data_type.FLOAT)
    summed = graph.sum(x, dim=(1, -1), name="sum")
    mean = graph.mean(x, dim=1, name="mean")
    amax = graph.amax(x, dim=-1, name="amax")

    assert summed.dims == (2, 1, 1)
    assert mean.dims == (2, 1, 4)
    assert amax.dims == (2, 3, 1)
    assert graph.nodes[0].op_type == "reduction"
    assert graph.nodes[0].attributes["mode"] == fe.reduction_mode.ADD
    assert graph.nodes[0].attributes["axis"] == (1, 2)
    assert graph.nodes[1].attributes["mode"] == fe.reduction_mode.AVG
    assert graph.nodes[2].attributes["mode"] == fe.reduction_mode.AMAX
    with pytest.raises(ValueError, match="duplicates"):
        graph.sum(x, axis=(1, -2))
    with pytest.raises(ValueError, match="out of range"):
        graph.sum(x, axis=3)
    with pytest.raises(ValueError, match="conflict"):
        graph.mean(x, axis=1, dim=2)


def test_mudnn_frontend_python_graph_api_genstats_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = graph.tensor(name="x", dim=(2, 3, 4), data_type=fe.data_type.FLOAT)
    sum_tensor, sqsum_tensor = graph.genstats(x, axis=(0, 2), name="stats")

    assert sum_tensor.dims == (1, 3, 1)
    assert sqsum_tensor.dims == (1, 3, 1)
    assert graph.nodes[0].op_type == "genstats"
    assert graph.nodes[0].inputs == {"x": x}
    assert graph.nodes[0].outputs == {"sum": sum_tensor, "sqsum": sqsum_tensor}
    assert graph.nodes[0].attributes["mode"] == fe.GenStatsMode.SUM_SQSUM
    assert graph.nodes[0].attributes["axis"] == (0, 2)
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    with pytest.raises(ValueError, match="duplicates"):
        graph.genstats(x, axis=(1, -2))
    with pytest.raises(ValueError, match="out of range"):
        graph.genstats(x, axis=3)
    with pytest.raises(ValueError, match="output data type"):
        graph.genstats(x, axis=1, data_type=fe.data_type.DOUBLE)


def test_mudnn_frontend_python_graph_api_genstats_default_compact_float_stats_build():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.HALF,
        intermediate_data_type=fe.data_type.HALF,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = graph.tensor(name="x", dim=(2, 3, 4, 5), data_type=fe.data_type.HALF)
    sum_tensor, sqsum_tensor = graph.genstats(input=x, name="stats")

    assert sum_tensor.dims == (3,)
    assert sqsum_tensor.dims == (3,)
    assert graph.nodes[0].attributes["axis"] == (0, 2, 3)

    sum_tensor.set_output(True).set_data_type(fe.data_type.FLOAT)
    sqsum_tensor.set_output(True).set_data_type(fe.data_type.FLOAT)
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_api_genstats_fake_cuda_cpu_pointer_fallback(
    monkeypatch,
):
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *args, **kwargs: "Fake CPU CUDA"
    )

    x = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    expected_sum = x.sum(dim=(0, 2, 3))
    expected_sqsum = torch.square(x).sum(dim=(0, 2, 3))
    sum_out = torch.zeros(3, dtype=torch.float32)
    sqsum_out = torch.zeros(3, dtype=torch.float32)

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x_desc = graph.tensor(name="x", dim=x.shape, stride=x.stride())
    sum_desc, sqsum_desc = graph.genstats(input=x_desc, name="stats")
    sum_desc.set_output(True).set_data_type(fe.data_type.FLOAT)
    sqsum_desc.set_output(True).set_data_type(fe.data_type.FLOAT)
    graph.build()

    graph.execute(
        {
            x_desc: x.data_ptr(),
            sum_desc: sum_out.data_ptr(),
            sqsum_desc: sqsum_out,
        }
    )

    torch.testing.assert_close(sum_out, expected_sum)
    torch.testing.assert_close(sqsum_out, expected_sqsum)


def test_mudnn_frontend_python_graph_api_bn_finalize_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    param = graph.tensor(name="param", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    eq_scale, eq_bias = graph.bn_finalize_inference(
        param, param, param, param, epsilon=1.0e-5, name="bn_final_inf"
    )

    assert eq_scale.dims == param.dims
    assert eq_bias.dims == param.dims
    assert graph.nodes[0].op_type == "bn_finalize"
    assert graph.nodes[0].attributes["mode"] == fe.BnFinalizeStatsMode.INFERENCE
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    training_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    training_param = training_graph.tensor(
        name="param", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    training_accum_count = training_graph.tensor(
        name="accum_count", dim=(1,), data_type=fe.data_type.INT64
    )
    training_outputs = training_graph.bn_finalize_training(
        training_param,
        training_param,
        training_param,
        training_param,
        training_param,
        training_param,
        training_accum_count,
        momentum=0.25,
        epsilon=1.0e-5,
        name="bn_final_train",
    )
    assert training_graph.nodes[0].attributes["mode"] == fe.BnFinalizeStatsMode.TRAINING
    assert len(training_outputs) == 6
    training_graph.build()
    assert training_graph._native.operation_graph.ptr.value is not None

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    good = err_graph.tensor(name="good", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    bad_shape = err_graph.tensor(
        name="bad_shape", dim=(1, 3, 1, 1), data_type=fe.data_type.FLOAT
    )
    bad_dtype = err_graph.tensor(
        name="bad_dtype", dim=(1, 2, 1, 1), data_type=fe.data_type.DOUBLE
    )
    bad_accum = err_graph.tensor(
        name="bad_accum", dim=(2,), data_type=fe.data_type.INT64
    )
    good_accum = err_graph.tensor(
        name="good_accum", dim=(1,), data_type=fe.data_type.INT64
    )
    with pytest.raises(ValueError, match="dimensions"):
        err_graph.bn_finalize_inference(good, bad_shape, good, good)
    with pytest.raises(ValueError, match="data type"):
        err_graph.bn_finalize_inference(good, bad_dtype, good, good)
    with pytest.raises(ValueError, match="epsilon"):
        err_graph.bn_finalize_inference(good, good, good, good, epsilon=-1.0)
    with pytest.raises(ValueError, match="one element"):
        err_graph.bn_finalize_training(good, good, good, good, good, good, bad_accum)
    with pytest.raises(ValueError, match="output data type"):
        err_graph.bn_finalize_training(
            good,
            good,
            good,
            good,
            good,
            good,
            good_accum,
            data_type=fe.data_type.DOUBLE,
        )


def test_mudnn_frontend_python_graph_api_bn_bwd_weights_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    dy = graph.tensor(name="dy", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    param = graph.tensor(name="param", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    outputs = graph.bn_backward_weights(x, dy, param, param, param, name="bn_bwd_w")

    assert len(outputs) == 5
    for output in outputs:
        assert output.dims == param.dims
    assert graph.nodes[0].op_type == "bn_bwd_weights"
    assert graph.nodes[0].inputs["scale"] == param
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    err_x = err_graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    bad_dy = err_graph.tensor(
        name="bad_dy", dim=(2, 2, 2, 1), data_type=fe.data_type.FLOAT
    )
    bad_dtype_dy = err_graph.tensor(
        name="bad_dtype_dy", dim=(2, 2, 2, 2), data_type=fe.data_type.DOUBLE
    )
    bad_param = err_graph.tensor(
        name="bad_param", dim=(1, 3, 1, 1), data_type=fe.data_type.FLOAT
    )
    bad_dtype_param = err_graph.tensor(
        name="bad_dtype_param", dim=(1, 2, 1, 1), data_type=fe.data_type.DOUBLE
    )
    good_param = err_graph.tensor(
        name="good_param", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )

    with pytest.raises(ValueError, match="shape and data type"):
        err_graph.bn_backward_weights(err_x, bad_dy, good_param, good_param, good_param)
    with pytest.raises(ValueError, match="shape and data type"):
        err_graph.bn_backward_weights(
            err_x, bad_dtype_dy, good_param, good_param, good_param
        )
    with pytest.raises(ValueError, match="dimensions"):
        err_graph.bn_backward_weights(err_x, err_x, bad_param, good_param, good_param)
    with pytest.raises(ValueError, match="data type"):
        err_graph.bn_backward_weights(
            err_x, err_x, bad_dtype_param, good_param, good_param
        )
    with pytest.raises(ValueError, match="output data type"):
        err_graph.bn_backward_weights(
            err_x,
            err_x,
            good_param,
            good_param,
            good_param,
            data_type=fe.data_type.DOUBLE,
        )


def test_mudnn_frontend_python_graph_api_resample_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = graph.tensor(name="x", dim=(1, 1, 4, 4), data_type=fe.data_type.FLOAT)
    avg = graph.avg_pool(x, window=(2, 2), stride=(2, 2), name="avg")
    max_pool = graph.max_pool(x, window=(2, 2), stride=(1, 2), name="max")
    dy = graph.tensor(name="dy", dim=(1, 1, 2, 2), data_type=fe.data_type.FLOAT)
    dx = graph.avg_pool_bwd(
        dy, x_shape=(1, 1, 4, 4), window=(2, 2), stride=(2, 2), name="avg_bwd"
    )
    max_dx = graph.max_pool_bwd(x, dy, window=(2, 2), stride=(2, 2), name="max_bwd")

    assert avg.dims == (1, 1, 2, 2)
    assert max_pool.dims == (1, 1, 3, 2)
    assert dx.dims == (1, 1, 4, 4)
    assert max_dx.dims == (1, 1, 4, 4)
    assert graph.nodes[0].op_type == "resample_fwd"
    assert graph.nodes[0].attributes["mode"] == fe.resample_mode.AVGPOOL_INCLUDE_PADDING
    assert graph.nodes[1].attributes["mode"] == fe.resample_mode.MAXPOOL
    assert graph.nodes[2].op_type == "resample_bwd"
    assert graph.nodes[3].op_type == "resample_fwd"
    assert "idx" in graph.nodes[3].outputs
    assert graph.nodes[4].op_type == "resample_bwd"
    assert graph.nodes[4].inputs["idx"] == graph.nodes[3].outputs["idx"]

    single = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    single_x = single.tensor(name="x", dim=(1, 1, 4, 4), data_type=fe.data_type.FLOAT)
    single.avg_pool(single_x, window=(2, 2), stride=(2, 2), name="avg")
    single.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert single._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_api_resample_rejects_invalid_shapes():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(1, 1, 4, 4), data_type=fe.data_type.FLOAT)
    dy = graph.tensor(name="dy", dim=(1, 1, 2, 2), data_type=fe.data_type.FLOAT)
    bad_rank = graph.tensor(
        name="bad_rank", dim=(1, 4, 4), data_type=fe.data_type.FLOAT
    )
    bad_dy = graph.tensor(name="bad_dy", dim=(1, 1, 3, 3), data_type=fe.data_type.FLOAT)

    with pytest.raises(ValueError, match="rank-4"):
        graph.avg_pool(bad_rank, window=(2, 2))
    with pytest.raises(ValueError, match="rank-2"):
        graph.avg_pool(x, window=(2,))
    with pytest.raises(ValueError, match="positive"):
        graph.avg_pool(x, window=(0, 2))
    with pytest.raises(ValueError, match="positive"):
        graph.avg_pool(x, window=(5, 5))
    with pytest.raises(ValueError, match="rank-4"):
        graph.avg_pool_bwd(dy, x_shape=(1, 1, 4), window=(2, 2))
    with pytest.raises(ValueError, match="dy dimensions"):
        graph.avg_pool_bwd(bad_dy, x_shape=x.dims, window=(2, 2), stride=(2, 2))
    with pytest.raises(ValueError, match="requires index"):
        graph.resample_bwd(
            dy,
            x_shape=x.dims,
            mode=fe.resample_mode.MAXPOOL,
            window=(2, 2),
            stride=(2, 2),
        )
    with pytest.raises(ValueError, match="dy dimensions"):
        graph.max_pool_bwd(x, bad_dy, window=(2, 2), stride=(2, 2))


def test_mudnn_frontend_python_graph_api_conv_fprop_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = graph.tensor(name="x", dim=(1, 1, 5, 5), data_type=fe.data_type.FLOAT)
    w = graph.tensor(name="w", dim=(2, 1, 3, 3), data_type=fe.data_type.FLOAT)
    y = graph.conv_fprop(
        x,
        w,
        padding=(1, 1),
        stride=(2, 2),
        dilation=(1, 1),
        name="conv",
    )

    assert y.dims == (1, 2, 3, 3)
    assert y.strides == (18, 9, 3, 1)
    assert graph.nodes[-1].op_type == "conv_fprop"
    assert graph.nodes[-1].attributes["mode"] == fe.convolution_mode.CROSS_CORRELATION
    assert graph.nodes[-1].attributes["padding"] == (1, 1)
    assert graph.nodes[-1].attributes["stride"] == (2, 2)
    assert graph.nodes[-1].attributes["dilation"] == (1, 1)
    graph.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert graph._native.operation_graph.ptr.value is not None

    channels_last_graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    cl_x = channels_last_graph.tensor(
        name="cl_x",
        dim=(4, 16, 32, 32),
        stride=(16384, 1, 512, 16),
        data_type=fe.data_type.FLOAT,
    )
    cl_w = channels_last_graph.tensor(
        name="cl_w",
        dim=(32, 16, 3, 3),
        stride=(144, 1, 48, 16),
        data_type=fe.data_type.FLOAT,
    )
    cl_y = channels_last_graph.conv_fprop(
        cl_x, cl_w, padding=(1, 1), stride=(1, 1), dilation=(1, 1)
    )
    assert cl_y.dims == (4, 32, 32, 32)
    assert cl_y.strides == (32768, 1, 1024, 32)

    alias_graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    alias_x = alias_graph.tensor(
        name="x", dim=(1, 1, 5, 5), data_type=fe.data_type.FLOAT
    )
    alias_w = alias_graph.tensor(
        name="w", dim=(2, 1, 3, 3), data_type=fe.data_type.FLOAT
    )
    alias_y = alias_graph.conv_fprop(
        image=alias_x,
        weight=alias_w,
        pre_padding=(1, 1),
        post_padding=(1, 1),
        stride=(2, 2),
        dilation=(1, 1),
        convolution_mode=fe.convolution_mode.CROSS_CORRELATION,
        name="conv_alias",
    )
    assert alias_y.dims == (1, 2, 3, 3)
    assert alias_graph.nodes[-1].attributes["padding"] == (1, 1)
    assert (
        alias_graph.nodes[-1].attributes["mode"]
        == fe.convolution_mode.CROSS_CORRELATION
    )
    alias_graph.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert alias_graph._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_api_conv_fprop_grouped_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = graph.tensor(name="x", dim=(1, 4, 4, 4), data_type=fe.data_type.FLOAT)
    w = graph.tensor(name="w", dim=(4, 2, 3, 3), data_type=fe.data_type.FLOAT)
    y = graph.conv_fprop(x, w, padding=(0, 0), name="grouped_conv")

    assert y.dims == (1, 4, 2, 2)
    assert graph.nodes[-1].attributes["group_count"] == 2
    graph.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert graph._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_api_conv3d_bias_leaky_relu_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.HALF,
        intermediate_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = graph.tensor(name="x", dim=(4, 16, 52, 54, 56), data_type=fe.data_type.HALF)
    w = graph.tensor(name="w", dim=(32, 16, 3, 3, 3), data_type=fe.data_type.HALF)
    bias = graph.tensor(name="bias", dim=(1, 32, 1, 1, 1), data_type=fe.data_type.HALF)

    conv = graph.conv_fprop(
        image=x,
        weight=w,
        padding=[0, 1, 2],
        stride=[2, 3, 4],
        dilation=[1, 1, 1],
        name="conv",
    )
    biased = graph.bias(name="bias_add", input=conv, bias=bias)
    y = graph.leaky_relu(name="relu", input=biased, negative_slope=0.01)
    y.set_output(True)

    assert conv.dims == (4, 32, 25, 18, 15)
    assert biased.dims == conv.dims
    assert y.dims == conv.dims
    assert graph.nodes[0].op_type == "conv_fprop"
    assert graph.nodes[0].attributes["padding"] == (0, 1, 2)
    assert graph.nodes[0].attributes["stride"] == (2, 3, 4)
    assert graph.nodes[0].attributes["dilation"] == (1, 1, 1)

    graph.validate().build_operation_graph()
    assert graph._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_api_conv_fprop_rejects_invalid_shapes():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(1, 1, 3, 3), data_type=fe.data_type.FLOAT)
    w = graph.tensor(name="w", dim=(1, 1, 2, 2), data_type=fe.data_type.FLOAT)
    bad_rank = graph.tensor(
        name="bad_rank", dim=(1, 3, 3), data_type=fe.data_type.FLOAT
    )
    bad_channels = graph.tensor(
        name="bad_channels", dim=(1, 2, 2, 2), data_type=fe.data_type.FLOAT
    )
    too_large = graph.tensor(
        name="too_large", dim=(1, 1, 5, 5), data_type=fe.data_type.FLOAT
    )

    with pytest.raises(ValueError, match="rank-3 to rank-5"):
        graph.conv_fprop(bad_rank, w)
    with pytest.raises(ValueError, match="channel"):
        graph.conv_fprop(x, bad_channels)
    with pytest.raises(ValueError, match="rank-2"):
        graph.conv_fprop(x, w, padding=(0,))
    with pytest.raises(ValueError, match="nonnegative"):
        graph.conv_fprop(x, w, padding=(-1, 0))
    with pytest.raises(ValueError, match="positive"):
        graph.conv_fprop(x, w, stride=(0, 1))
    with pytest.raises(ValueError, match="output dimensions"):
        graph.conv_fprop(x, too_large)
    asym = graph.conv_fprop(x, w, pre_padding=(1, 0), post_padding=(0, 0))
    assert asym.dims == (1, 1, 3, 2)
    assert graph.nodes[-1].attributes["pre_padding"] == (1, 0)
    assert graph.nodes[-1].attributes["post_padding"] == (0, 0)
    with pytest.raises(ValueError, match="conflicts"):
        graph.conv_fprop(
            x,
            w,
            padding=(0, 0),
            pre_padding=(1, 1),
            post_padding=(1, 1),
        )


def test_mudnn_frontend_python_graph_api_conv_backward_symbolic():
    import cudnn_frontend as fe

    dgrad_graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    dy = dgrad_graph.tensor(name="dy", dim=(1, 1, 3, 3), data_type=fe.data_type.FLOAT)
    w = dgrad_graph.tensor(name="w", dim=(1, 1, 2, 2), data_type=fe.data_type.FLOAT)
    dx = dgrad_graph.conv_bwd_data(
        dy,
        w,
        x_shape=(1, 1, 5, 5),
        padding=(1, 1),
        stride=(2, 2),
        dilation=(1, 1),
        name="dgrad",
    )
    assert dx.dims == (1, 1, 5, 5)
    assert dgrad_graph.nodes[0].op_type == "conv_bwd_data"
    assert dgrad_graph.nodes[0].outputs == {"dx": dx}
    dgrad_graph.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert dgrad_graph._native.operation_graph.ptr.value is not None

    wgrad_graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = wgrad_graph.tensor(name="x", dim=(1, 1, 5, 5), data_type=fe.data_type.FLOAT)
    dy = wgrad_graph.tensor(name="dy", dim=(1, 1, 3, 3), data_type=fe.data_type.FLOAT)
    dw = wgrad_graph.conv_bwd_filter(
        x,
        dy,
        w_shape=(1, 1, 2, 2),
        padding=(1, 1),
        stride=(2, 2),
        dilation=(1, 1),
        name="wgrad",
    )
    assert dw.dims == (1, 1, 2, 2)
    assert wgrad_graph.nodes[0].op_type == "conv_bwd_filter"
    assert wgrad_graph.nodes[0].outputs == {"dw": dw}
    wgrad_graph.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert wgrad_graph._native.operation_graph.ptr.value is not None

    inferred_wgrad = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = inferred_wgrad.tensor(name="x", dim=(1, 4, 4, 4), data_type=fe.data_type.FLOAT)
    dy = inferred_wgrad.tensor(
        name="dy", dim=(1, 4, 2, 2), data_type=fe.data_type.FLOAT
    )
    dw = inferred_wgrad.conv_wgrad(
        name="wgrad",
        image=x,
        loss=dy,
        padding=(0, 0),
        stride=(1, 1),
        dilation=(1, 1),
    )
    dw.set_output(True).set_dim((4, 4, 3, 3))
    assert dw.dims == (4, 4, 3, 3)
    assert inferred_wgrad.nodes[0].op_type == "conv_bwd_filter"
    assert inferred_wgrad.nodes[0].attributes["group_count"] == 1
    inferred_wgrad.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert inferred_wgrad._native.operation_graph.ptr.value is not None

    inferred_grouped_wgrad = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = inferred_grouped_wgrad.tensor(
        name="x", dim=(1, 8, 16, 64), data_type=fe.data_type.FLOAT
    )
    dy = inferred_grouped_wgrad.tensor(
        name="dy", dim=(1, 144, 4, 32), data_type=fe.data_type.FLOAT
    )
    dw = inferred_grouped_wgrad.conv_wgrad(
        name="inferred_grouped_wgrad",
        image=x,
        loss=dy,
        padding=(0, 2),
        stride=(2, 2),
        dilation=(2, 1),
    )
    dw.set_output(True).set_dim((144, 4, 5, 5))
    # Upstream mudnn-frontend sets the final filter shape after conv_wgrad()
    # creation; group_count must follow that late output shape.
    assert inferred_grouped_wgrad.nodes[0].attributes["group_count"] == 2
    inferred_grouped_wgrad.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert inferred_grouped_wgrad._native.operation_graph.ptr.value is not None

    wgrad_3d = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = wgrad_3d.tensor(name="x", dim=(1, 16, 40, 8, 64), data_type=fe.data_type.FLOAT)
    dy = wgrad_3d.tensor(
        name="dy", dim=(1, 104, 40, 8, 62), data_type=fe.data_type.FLOAT
    )
    dw = wgrad_3d.conv_wgrad(
        name="wgrad_3d",
        image=x,
        loss=dy,
        padding=(0, 1, 1),
        stride=(1, 1, 1),
        dilation=(2, 1, 1),
    )
    dw.set_output(True).set_dim((104, 4, 1, 3, 5))
    wgrad_3d.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert wgrad_3d._native.operation_graph.ptr.value is not None

    grouped_dgrad = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    dy = grouped_dgrad.tensor(name="dy", dim=(1, 4, 2, 2), data_type=fe.data_type.FLOAT)
    w = grouped_dgrad.tensor(name="w", dim=(4, 2, 3, 3), data_type=fe.data_type.FLOAT)
    dx = grouped_dgrad.conv_bwd_data(
        dy,
        w,
        x_shape=(1, 4, 4, 4),
        convolution_mode=fe.convolution_mode.CROSS_CORRELATION,
        name="grouped_dgrad",
    )
    assert dx.dims == (1, 4, 4, 4)
    assert grouped_dgrad.nodes[0].attributes["group_count"] == 2
    grouped_dgrad.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert grouped_dgrad._native.operation_graph.ptr.value is not None

    grouped_wgrad = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = grouped_wgrad.tensor(name="x", dim=(1, 4, 4, 4), data_type=fe.data_type.FLOAT)
    dy = grouped_wgrad.tensor(name="dy", dim=(1, 4, 2, 2), data_type=fe.data_type.FLOAT)
    dw = grouped_wgrad.conv_bwd_filter(
        x,
        dy,
        w_shape=(4, 2, 3, 3),
        convolution_mode=fe.convolution_mode.CROSS_CORRELATION,
        name="grouped_wgrad",
    )
    assert dw.dims == (4, 2, 3, 3)
    assert grouped_wgrad.nodes[0].attributes["group_count"] == 2
    grouped_wgrad.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.INSTANT]
    )
    assert grouped_wgrad._native.operation_graph.ptr.value is not None

    alias_dgrad = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    dy = alias_dgrad.tensor(name="dy", dim=(1, 1, 3, 3), data_type=fe.data_type.FLOAT)
    w = alias_dgrad.tensor(name="w", dim=(1, 1, 2, 2), data_type=fe.data_type.FLOAT)
    dx = alias_dgrad.conv_dgrad(
        loss=dy,
        filter=w,
        pre_padding=(1, 1),
        post_padding=(1, 1),
        stride=(2, 2),
        dilation=(1, 1),
        convolution_mode=fe.convolution_mode.CROSS_CORRELATION,
        name="dgrad_alias",
    )
    assert dx.dims == (1, 1, 4, 4)
    assert alias_dgrad.nodes[0].attributes["padding"] == (1, 1)
    assert (
        alias_dgrad.nodes[0].attributes["mode"] == fe.convolution_mode.CROSS_CORRELATION
    )

    alias_wgrad = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    x = alias_wgrad.tensor(name="x", dim=(1, 1, 5, 5), data_type=fe.data_type.FLOAT)
    dy = alias_wgrad.tensor(name="dy", dim=(1, 1, 3, 3), data_type=fe.data_type.FLOAT)
    dw = alias_wgrad.conv_wgrad(
        image=x,
        loss=dy,
        pre_padding=(1, 1),
        post_padding=(1, 1),
        stride=(2, 2),
        dilation=(1, 1),
        convolution_mode=fe.convolution_mode.CROSS_CORRELATION,
        name="wgrad_alias",
    )
    assert dw.dims == (1, 1, 3, 3)
    assert alias_wgrad.nodes[0].attributes["padding"] == (1, 1)
    assert (
        alias_wgrad.nodes[0].attributes["mode"] == fe.convolution_mode.CROSS_CORRELATION
    )


def test_mudnn_frontend_python_graph_api_conv_backward_rejects_invalid_shapes():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    dy = graph.tensor(name="dy", dim=(1, 1, 2, 2), data_type=fe.data_type.FLOAT)
    w = graph.tensor(name="w", dim=(1, 1, 2, 2), data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(1, 1, 3, 3), data_type=fe.data_type.FLOAT)
    bad_rank = graph.tensor(
        name="bad_rank", dim=(1, 2, 2), data_type=fe.data_type.FLOAT
    )
    bad_channels = graph.tensor(
        name="bad_channels", dim=(1, 2, 2, 2), data_type=fe.data_type.FLOAT
    )
    bad_dy = graph.tensor(name="bad_dy", dim=(1, 1, 3, 3), data_type=fe.data_type.FLOAT)

    with pytest.raises(ValueError, match="rank-3 to rank-5"):
        graph.conv_bwd_data(bad_rank, w, x_shape=x.dims)
    with pytest.raises(ValueError, match="x_shape rank"):
        graph.conv_bwd_data(dy, w, x_shape=(1, 1, 3))
    with pytest.raises(ValueError, match="channel"):
        graph.conv_bwd_data(dy, bad_channels, x_shape=x.dims)
    with pytest.raises(ValueError, match="dy dimensions"):
        graph.conv_bwd_data(bad_dy, w, x_shape=x.dims)
    asym_dx = graph.conv_dgrad(
        loss=dy,
        filter=w,
        pre_padding=(1, 0),
        post_padding=(0, 0),
    )
    assert asym_dx.dims == (1, 1, 2, 3)
    assert graph.nodes[-1].attributes["pre_padding"] == (1, 0)
    assert graph.nodes[-1].attributes["post_padding"] == (0, 0)
    with pytest.raises(ValueError, match="rank-3 to rank-5"):
        graph.conv_bwd_filter(bad_rank, dy, w_shape=w.dims)
    with pytest.raises(ValueError, match="w_shape rank"):
        graph.conv_bwd_filter(x, dy, w_shape=(1, 1, 2))
    with pytest.raises(ValueError, match="channel"):
        graph.conv_bwd_filter(bad_channels, dy, w_shape=w.dims)
    with pytest.raises(ValueError, match="dy dimensions"):
        graph.conv_bwd_filter(x, bad_dy, w_shape=w.dims)
    asym_dw = graph.conv_wgrad(
        image=x,
        loss=dy,
        pre_padding=(1, 0),
        post_padding=(0, 0),
    )
    assert asym_dw.dims == (1, 1, 3, 2)
    assert graph.nodes[-1].attributes["pre_padding"] == (1, 0)
    assert graph.nodes[-1].attributes["post_padding"] == (0, 0)


def test_mudnn_frontend_python_graph_api_concat_reshape_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    a = graph.tensor(name="a", dim=(2, 2), data_type=fe.data_type.FLOAT)
    b = graph.tensor(name="b", dim=(2, 3), data_type=fe.data_type.FLOAT)
    y = graph.concat((a, b), axis=-1, name="concat")
    z = graph.reshape(y, shape=(10,), name="reshape")

    assert y.dims == (2, 5)
    assert z.dims == (10,)
    assert graph.nodes[0].op_type == "concat"
    assert graph.nodes[0].attributes["axis"] == 1
    assert graph.nodes[1].op_type == "reshape"
    assert graph.nodes[1].attributes["reshape_mode"] == fe.reshape_mode.VIEW_ONLY
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    logical = graph.reshape(
        y, shape=(10,), name="logical_reshape", reshape_mode=fe.reshape_mode.LOGICAL
    )
    assert logical.dims == (10,)
    assert graph.nodes[-1].attributes["reshape_mode"] == fe.reshape_mode.LOGICAL


def test_mudnn_frontend_python_graph_api_slice_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(4, 6, 8), data_type=fe.data_type.FLOAT)
    y = graph.slice(
        x,
        (slice(1, None), slice(None, 5, 2), slice(2, 8, 3)),
        name="slice",
    )

    assert y.dims == (3, 3, 2)
    assert y.strides == (48, 16, 3)
    assert graph.nodes[0].op_type == "slice"
    assert graph.nodes[0].attributes["start_indices"] == (1, 0, 2)
    assert graph.nodes[0].attributes["limit_indices"] == (4, 5, 8)
    assert graph.nodes[0].attributes["strides"] == (1, 2, 3)

    restored = fe.pygraph().deserialize(graph.serialize())
    restored_node = restored.nodes[0]
    assert restored_node.op_type == "slice"
    assert restored_node.outputs["y"].dims == (3, 3, 2)
    assert restored_node.attributes["strides"] == (1, 2, 3)
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_api_transpose_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 3, 4), data_type=fe.data_type.FLOAT)
    y = graph.transpose(x, permutation=(1, 0, 2), name="transpose")

    assert y.dims == (3, 2, 4)
    assert y.strides == (4, 12, 1)
    assert graph.nodes[0].op_type == "transpose"
    assert graph.nodes[0].attributes["permutation"] == (1, 0, 2)

    restored = fe.pygraph().deserialize(graph.serialize())
    restored_node = restored.nodes[0]
    assert restored_node.op_type == "transpose"
    assert restored_node.outputs["y"].dims == (3, 2, 4)
    assert restored_node.outputs["y"].strides == (4, 12, 1)
    assert restored_node.attributes["permutation"] == (1, 0, 2)
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_api_execute_slice_transpose():
    import cudnn_frontend as fe

    x_np = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    expected_slice = x_np[1:4, :5:2, 2:8:3].copy()
    trans_x_np = x_np[:2, :3, :4].copy()
    expected_transpose = np.transpose(trans_x_np, (1, 0, 2)).copy()

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        slice_out = device_memory.create_tensor(
            0, expected_slice.nbytes, expected_slice.shape, "float32"
        )
        trans_x = device_memory.create_tensor(
            trans_x_np.__array_interface__["data"][0],
            trans_x_np.nbytes,
            trans_x_np.shape,
            "float32",
        )
        transpose_out = device_memory.create_tensor(
            0, expected_transpose.nbytes, expected_transpose.shape, "float32"
        )
        tensors = [x, slice_out, trans_x, transpose_out]

        slice_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
        x_desc = slice_graph.tensor(
            name="x", dim=x_np.shape, data_type=fe.data_type.FLOAT
        )
        slice_desc = slice_graph.slice(
            x_desc,
            (slice(1, None), slice(None, 5, 2), slice(2, 8, 3)),
            name="slice",
        )
        slice_desc.set_output(True).set_stride((6, 2, 1))
        slice_graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                slice_desc: device_memory.get_tensor_addr(slice_out)[0],
            }
        )
        actual_slice = device_memory.copy_to_ndarray(slice_out, expected_slice.nbytes)
        np.testing.assert_allclose(actual_slice, expected_slice)

        transpose_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
        trans_x_desc = transpose_graph.tensor(
            name="x", dim=trans_x_np.shape, data_type=fe.data_type.FLOAT
        )
        transpose_desc = transpose_graph.transpose(
            trans_x_desc, permutation=(1, 0, 2), name="transpose"
        )
        transpose_desc.set_output(True).set_stride((8, 4, 1))
        transpose_graph.execute(
            {
                trans_x_desc: device_memory.get_tensor_addr(trans_x)[0],
                transpose_desc: device_memory.get_tensor_addr(transpose_out)[0],
            }
        )
        actual_transpose = device_memory.copy_to_ndarray(
            transpose_out, expected_transpose.nbytes
        )
        np.testing.assert_allclose(actual_transpose, expected_transpose)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_slice_matmul_torch_cpu_fallback():
    torch = pytest.importorskip("torch")
    import cudnn_frontend as fe

    a = torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5).to(torch.bfloat16)
    b = torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 6).to(torch.bfloat16)
    a_slice = a[1:, 2:, :]
    b_slice = b[1:, :, 3:]
    expected = torch.matmul(a_slice, b_slice)

    graph = fe.pygraph(io_data_type=fe.data_type.BFLOAT16)
    a_desc = graph.tensor_like(a)
    b_desc = graph.tensor_like(b)
    a_slice_desc = graph.slice(a_desc, (slice(1, None), slice(2, None), slice(None)))
    b_slice_desc = graph.slice(b_desc, (slice(1, None), slice(None), slice(3, None)))
    c_desc = graph.matmul(
        A=a_slice_desc,
        B=b_slice_desc,
        compute_data_type=fe.data_type.FLOAT,
        name="matmul",
    )
    c_desc.set_output(True).set_data_type(fe.data_type.BFLOAT16)

    actual = torch.zeros_like(expected)
    graph.execute({a_desc: a, b_desc: b, c_desc: actual}, workspace=torch.empty(0))
    torch.testing.assert_close(actual, expected)


def test_mudnn_frontend_python_graph_api_rope_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 4, 3, 10), data_type=fe.data_type.FLOAT)
    freqs = graph.tensor(name="freqs", dim=(4, 6), data_type=fe.data_type.FLOAT)

    y = graph.rope(
        input=x,
        freqs_cis=freqs,
        batch_first=True,
        name="rope",
    )

    assert y.dims == x.dims
    assert y.name == "rope"
    assert graph.nodes[0].op_type == "slice"
    assert graph.nodes[0].outputs["y"].dims == (2, 4, 3, 6)
    assert graph.nodes[1].outputs["y"].dims == (2, 4, 3, 4)
    assert graph.nodes[-6].outputs["y"].dims == (1, 4, 1, 6)
    pointwise_modes = [
        node.attributes["mode"] for node in graph.nodes if node.op_type == "pointwise"
    ]
    assert fe.pointwise_mode.COS in pointwise_modes
    assert fe.pointwise_mode.SIN in pointwise_modes
    assert fe.pointwise_mode.NEG in pointwise_modes

    restored = fe.pygraph().deserialize(graph.serialize())
    assert restored.nodes[-1].op_type == "concat"
    assert restored.nodes[-1].outputs["y"].dims == x.dims
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None


def test_mudnn_frontend_python_graph_api_rope_interleaved_mla_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(8, 2, 4, 12), data_type=fe.data_type.FLOAT)
    freqs = graph.tensor(name="freqs", dim=(8, 8), data_type=fe.data_type.FLOAT)

    y = graph.rope(
        x=x,
        freqs=freqs,
        rotary_interleaved=True,
        multi_latent_attention=True,
        name="rope_interleaved",
    )

    assert y.dims == x.dims
    assert y.name == "rope_interleaved"
    assert any(node.name == "rope_interleaved_mla_reordered" for node in graph.nodes)
    assert any(
        node.name == "rope_interleaved_interleaved_stack" for node in graph.nodes
    )
    assert any(
        node.op_type == "reshape" and node.outputs["y"].dims == (8, 1, 1, 8)
        for node in graph.nodes
    )
    restored = fe.pygraph().deserialize(graph.serialize())
    assert restored.nodes[-1].outputs["y"].dims == x.dims


def test_mudnn_frontend_python_graph_api_rope_partial_scaled_backward_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 4, 3, 12), data_type=fe.data_type.FLOAT)
    freqs = graph.tensor(name="freqs", dim=(4, 1, 1, 8), data_type=fe.data_type.FLOAT)

    y = graph.rope(
        input=x,
        freqs=freqs,
        output_scale=1.25,
        rope_dim=8,
        batch_first=True,
        name="rope_partial",
    )

    assert y.dims == x.dims
    assert y.name == "rope_partial"
    assert graph.nodes[0].op_type == "slice"
    assert graph.nodes[0].outputs["y"].dims == (2, 4, 3, 8)
    assert graph.nodes[0].attributes["start_indices"] == (0, 0, 0, 4)
    assert graph.nodes[1].outputs["y"].dims == (2, 4, 3, 4)
    assert graph.nodes[1].attributes["start_indices"] == (0, 0, 0, 0)
    assert graph.nodes[-2].op_type == "concat"
    assert graph.nodes[-2].outputs["y"].dims == x.dims
    assert graph.nodes[-1].op_type == "pointwise"
    assert graph.nodes[-1].attributes["mode"] == fe.pointwise_mode.IDENTITY
    assert graph.nodes[-1].attributes["alpha1"] == 1.25

    d_x = graph.rope_backward(
        dY=y,
        freqs=freqs,
        output_scale=1.25,
        rope_dim=8,
        batch_first=True,
        name="rope_backward_partial",
    )

    assert d_x.dims == x.dims
    assert d_x.name == "rope_backward_partial"
    assert any(node.name == "rope_backward_partial_neg_freqs" for node in graph.nodes)
    assert graph.nodes[-1].attributes["alpha1"] == 1.25

    restored = fe.pygraph().deserialize(graph.serialize())
    assert restored.nodes[-1].outputs["y"].dims == x.dims
    assert restored.nodes[-1].attributes["alpha1"] == 1.25

    with pytest.raises(ValueError, match="last dimension must match rope_dim"):
        graph.rope(input=x, freqs=freqs, rope_dim=6)
    with pytest.raises(ValueError, match="requires dY and freqs_cis"):
        graph.rope_backward(freqs=freqs)


def test_mudnn_frontend_python_graph_api_execute_rope_composite():
    import cudnn_frontend as fe

    x_np = np.arange(2 * 3 * 2 * 8, dtype=np.float32).reshape(2, 3, 2, 8) / 17.0
    freqs_np = np.linspace(-0.6, 0.7, 3 * 6, dtype=np.float32).reshape(3, 6)
    x_rot = x_np[..., :6]
    rotated_half = np.concatenate((-x_rot[..., 3:6], x_rot[..., :3]), axis=-1)
    freqs = freqs_np.reshape(1, 3, 1, 6)
    expected = np.concatenate(
        (x_rot * np.cos(freqs) + rotated_half * np.sin(freqs), x_np[..., 6:]),
        axis=-1,
    ).astype(np.float32)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        freqs_tensor = device_memory.create_tensor(
            freqs_np.__array_interface__["data"][0],
            freqs_np.nbytes,
            freqs_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, freqs_tensor, out]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        freqs_desc = graph.tensor(
            name="freqs", dim=freqs_np.shape, data_type=fe.DataType.FLOAT
        )
        y_desc = graph.rope(
            input=x_desc,
            freqs_cis=freqs_desc,
            batch_first=True,
            name="rope",
        )
        workspace_size = graph.get_workspace_size()
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, (workspace_size,), "int8"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                freqs_desc: device_memory.get_tensor_addr(freqs_tensor)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=1.0e-3)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_rope_partial_scaled():
    import cudnn_frontend as fe

    x_np = np.arange(2 * 3 * 2 * 10, dtype=np.float32).reshape(2, 3, 2, 10) / 23.0
    freqs_np = np.linspace(-0.4, 0.5, 3 * 6, dtype=np.float32).reshape(3, 6)
    output_scale = 1.25
    pass_prefix = x_np[..., :4]
    x_rot = x_np[..., 4:10]
    rotated_half = np.concatenate((-x_rot[..., 3:6], x_rot[..., :3]), axis=-1)
    freqs = freqs_np.reshape(1, 3, 1, 6)
    rotated = x_rot * np.cos(freqs) + rotated_half * np.sin(freqs)
    expected = (np.concatenate((pass_prefix, rotated), axis=-1) * output_scale).astype(
        np.float32
    )

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        freqs_tensor = device_memory.create_tensor(
            freqs_np.__array_interface__["data"][0],
            freqs_np.nbytes,
            freqs_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, freqs_tensor, out]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        freqs_desc = graph.tensor(
            name="freqs", dim=freqs_np.shape, data_type=fe.DataType.FLOAT
        )
        y_desc = graph.rope(
            input=x_desc,
            freqs_cis=freqs_desc,
            output_scale=output_scale,
            rope_dim=6,
            batch_first=True,
            name="rope_partial",
        )
        workspace_size = graph.get_workspace_size()
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, (workspace_size,), "int8"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                freqs_desc: device_memory.get_tensor_addr(freqs_tensor)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=1.0e-3)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_concat_reshape_rejects_invalid_shapes():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    a = graph.tensor(name="a", dim=(2, 2), data_type=fe.data_type.FLOAT)
    b = graph.tensor(name="b", dim=(3, 2), data_type=fe.data_type.FLOAT)
    c = graph.tensor(name="c", dim=(2, 2), data_type=fe.data_type.INT32)

    with pytest.raises(ValueError, match="at least one"):
        graph.concat((), axis=0)
    with pytest.raises(ValueError, match="out of range"):
        graph.concat((a, b), axis=3)
    with pytest.raises(ValueError, match="non-axis"):
        graph.concat((a, b), axis=1)
    with pytest.raises(ValueError, match="data type"):
        graph.concat((a, c), axis=0)
    with pytest.raises(NotImplementedError, match="inplace"):
        graph.concat((a, b), axis=1, inplace_index=0)
    with pytest.raises(ValueError, match="preserve element count"):
        graph.reshape(a, shape=(5,))
    with pytest.raises(ValueError, match="rank"):
        graph.slice(a, (slice(None),))
    with pytest.raises(ValueError, match="positive"):
        graph.slice(a, (slice(None), slice(None, None, -1)))
    with pytest.raises(ValueError, match="rank"):
        graph.transpose(a, permutation=(0,))
    with pytest.raises(ValueError, match="duplicates"):
        graph.transpose(a, permutation=(0, 0))
    with pytest.raises(ValueError, match="out of range"):
        graph.transpose(a, permutation=(0, 2))


def test_mudnn_frontend_python_graph_api_rng_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    offset = graph.tensor(name="offset", dim=(1,), data_type=fe.data_type.INT64)
    bad_offset = graph.tensor(name="bad_offset", dim=(2,), data_type=fe.data_type.INT64)
    y = graph.rng(
        shape=(2, 3),
        seed=7,
        distribution=fe.RngDistribution.UNIFORM,
        minimum=-1.0,
        maximum=1.0,
        offset=offset,
        name="rng",
    )

    assert y.dims == (2, 3)
    assert graph.nodes[0].op_type == "rng"
    assert graph.nodes[0].inputs == {"offset": offset}
    assert graph.nodes[0].attributes["seed"] == 7
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    with pytest.raises(ValueError, match="minimum"):
        err_graph.rng(shape=(2,), seed=1, minimum=1.0, maximum=1.0)
    with pytest.raises(ValueError, match="standard deviation"):
        err_graph.rng(
            shape=(2,),
            seed=1,
            distribution=fe.RngDistribution.NORMAL,
            stddev=0.0,
        )
    with pytest.raises(ValueError, match="probability"):
        err_graph.rng(
            shape=(2,),
            seed=1,
            distribution=fe.RngDistribution.BERNOULLI,
            probability=1.5,
        )
    with pytest.raises(ValueError, match="offset"):
        err_graph.rng(shape=(2,), seed=1, offset=bad_offset)


def test_mudnn_frontend_python_graph_api_moe_grouped_matmul_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.BFLOAT16,
        compute_data_type=fe.data_type.FLOAT,
    )
    token = graph.tensor(name="token", dim=(1, 6, 4), data_type=fe.data_type.BFLOAT16)
    weight = graph.tensor(name="weight", dim=(2, 4, 8), data_type=fe.data_type.BFLOAT16)
    first_token_offset = graph.tensor(
        name="first_token_offset", dim=(2,), data_type=fe.data_type.INT64
    )
    first_token_offset_3d = graph.tensor(
        name="first_token_offset_3d",
        dim=(2, 1, 1),
        stride=(1, 1, 1),
        data_type=fe.data_type.INT32,
    )
    token_index = graph.tensor(
        name="token_index", dim=(2, 3), data_type=fe.data_type.INT64
    )
    token_ks = graph.tensor(name="token_ks", dim=(6,), data_type=fe.data_type.INT64)

    none_output = graph.moe_grouped_matmul(
        token,
        weight,
        first_token_offset,
        name="moe_none",
    )
    gather_output = graph.moe_grouped_matmul(
        token,
        weight,
        first_token_offset,
        token_index=token_index,
        mode=fe.moe_grouped_matmul_mode.GATHER,
        compute_data_type=fe.data_type.FLOAT,
        name="moe_gather",
    )
    scatter_output = graph.moe_grouped_matmul(
        token,
        weight,
        first_token_offset,
        token_index=token_index,
        token_ks=token_ks,
        mode=fe.moe_grouped_matmul_mode.SCATTER,
        top_k=2,
        name="moe_scatter",
    )
    singleton_tail_output = graph.moe_grouped_matmul(
        token,
        weight,
        first_token_offset_3d,
        name="moe_singleton_tail_offset",
    )

    assert none_output.dims == (1, 6, 8)
    assert gather_output.dims == (1, 3, 8)
    assert scatter_output.dims == (1, 6, 8)
    assert singleton_tail_output.dims == (1, 6, 8)
    assert graph.nodes[0].op_type == "moe_grouped_matmul"
    assert graph.nodes[0].attributes["mode"] == fe.moe_grouped_matmul_mode.NONE
    assert graph.nodes[1].attributes["mode"] == fe.moe_grouped_matmul_mode.GATHER
    assert graph.nodes[2].inputs["token_ks"] is token_ks
    assert graph.nodes[2].attributes["top_k"] == 2

    restored = fe.pygraph().deserialize(graph.serialize())
    assert restored.nodes[2].op_type == "moe_grouped_matmul"
    assert restored.nodes[2].outputs["output"].dims == (1, 6, 8)
    assert restored.nodes[2].attributes["mode"] == fe.moe_grouped_matmul_mode.SCATTER
    assert restored.nodes[3].inputs["first_token_offset"].dims == (2, 1, 1)

    err_graph = fe.pygraph(io_data_type=fe.data_type.BFLOAT16)
    with pytest.raises(ValueError, match="K dimensions"):
        err_graph.moe_grouped_matmul(
            token,
            err_graph.tensor(
                name="bad_weight",
                dim=(2, 5, 8),
                data_type=fe.data_type.BFLOAT16,
            ),
            first_token_offset,
        )
    with pytest.raises(ValueError, match="token_index"):
        err_graph.moe_grouped_matmul(
            token,
            weight,
            first_token_offset,
            mode=fe.moe_grouped_matmul_mode.GATHER,
        )
    with pytest.raises(ValueError, match="top_k"):
        err_graph.moe_grouped_matmul(
            token,
            weight,
            first_token_offset,
            token_index=token_index,
            token_ks=token_ks,
            mode=fe.moe_grouped_matmul_mode.SCATTER,
            top_k=0,
        )


def test_mudnn_frontend_python_graph_api_moe_grouped_matmul_bwd_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.BFLOAT16,
        compute_data_type=fe.data_type.FLOAT,
    )
    doutput = graph.tensor(
        name="doutput", dim=(1, 6, 8), data_type=fe.data_type.BFLOAT16
    )
    token = graph.tensor(name="token", dim=(1, 6, 4), data_type=fe.data_type.BFLOAT16)
    first_token_offset = graph.tensor(
        name="first_token_offset", dim=(2,), data_type=fe.data_type.INT64
    )
    first_token_offset_3d = graph.tensor(
        name="first_token_offset_3d",
        dim=(2, 1, 1),
        stride=(1, 1, 1),
        data_type=fe.data_type.INT32,
    )

    dweight = graph.moe_grouped_matmul_bwd(
        doutput,
        token,
        first_token_offset,
        compute_data_type=fe.data_type.FLOAT,
        name="moe_bwd",
    )

    assert dweight.dims == (2, 4, 8)
    assert graph.moe_grouped_matmul_bwd(
        doutput,
        token,
        first_token_offset_3d,
        name="moe_bwd_singleton_tail_offset",
    ).dims == (2, 4, 8)
    assert graph.nodes[0].op_type == "moe_grouped_matmul_bwd"
    assert graph.nodes[0].outputs == {"dweight": dweight}
    assert graph.nodes[0].attributes["compute_data_type"] == fe.data_type.FLOAT
    restored = fe.pygraph().deserialize(graph.serialize())
    assert restored.nodes[0].outputs["dweight"].dims == (2, 4, 8)

    with pytest.raises(ValueError, match="token counts"):
        graph.moe_grouped_matmul_bwd(
            graph.tensor(
                name="bad_doutput",
                dim=(1, 5, 8),
                data_type=fe.data_type.BFLOAT16,
            ),
            token,
            first_token_offset,
        )


def test_mudnn_frontend_python_graph_api_execute_moe_grouped_matmul_cpu_fallback():
    torch = pytest.importorskip("torch")
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    token = graph.tensor(name="token", dim=(1, 5, 3), data_type=fe.data_type.FLOAT)
    weight = graph.tensor(
        name="weight",
        dim=(2, 3, 2),
        stride=(6, 1, 3),
        data_type=fe.data_type.FLOAT,
    )
    first_token_offset = graph.tensor(
        name="first_token_offset",
        dim=(2, 1, 1),
        stride=(1, 1, 1),
        data_type=fe.data_type.INT32,
    )
    output = graph.moe_grouped_matmul(
        token,
        weight,
        first_token_offset,
        mode=fe.moe_grouped_matmul_mode.NONE,
        compute_data_type=fe.data_type.FLOAT,
    )
    output.set_output(True)

    token_data = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3) / 10.0
    weight_storage = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            -1.0,
            0.5,
            0.25,
            2.0,
            -0.5,
            1.5,
        ],
        dtype=torch.float32,
    )
    weight_view = torch.as_strided(weight_storage, (2, 3, 2), (6, 1, 3))
    offsets = torch.tensor([0, 2], dtype=torch.int32).reshape(2, 1, 1)
    actual = torch.empty(1, 5, 2, dtype=torch.float32)

    graph.validate().build_operation_graph().create_execution_plans([fe.heur_mode.A])
    graph.check_support().build_plans()
    assert graph.get_workspace_size() == 0
    graph.execute(
        {
            token: token_data,
            weight: weight_view,
            first_token_offset: offsets,
            output: actual,
        }
    )

    expected = torch.zeros_like(actual)
    expected[0, 0:2] = token_data[0, 0:2] @ weight_view[0]
    expected[0, 2:5] = token_data[0, 2:5] @ weight_view[1]
    torch.testing.assert_close(actual, expected)

    bwd_graph = fe.pygraph(
        io_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    doutput = bwd_graph.tensor(
        name="doutput", dim=(1, 5, 2), data_type=fe.data_type.FLOAT
    )
    bwd_token = bwd_graph.tensor(
        name="token", dim=(1, 5, 3), data_type=fe.data_type.FLOAT
    )
    bwd_offsets = bwd_graph.tensor(
        name="first_token_offset",
        dim=(2, 1, 1),
        stride=(1, 1, 1),
        data_type=fe.data_type.INT32,
    )
    dweight = bwd_graph.moe_grouped_matmul_bwd(
        doutput,
        bwd_token,
        bwd_offsets,
        compute_data_type=fe.data_type.FLOAT,
    )
    dweight.set_output(True)

    doutput_data = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2) / 7.0
    actual_dweight = torch.empty(2, 3, 2, dtype=torch.float32)
    bwd_graph.validate().build_operation_graph().create_execution_plans(
        [fe.heur_mode.A]
    )
    bwd_graph.check_support().build_plans()
    bwd_graph.execute(
        {
            doutput: doutput_data,
            bwd_token: token_data,
            bwd_offsets: offsets,
            dweight: actual_dweight,
        }
    )

    expected_dweight = torch.zeros_like(actual_dweight)
    expected_dweight[0] = token_data[0, 0:2].transpose(0, 1) @ doutput_data[0, 0:2]
    expected_dweight[1] = token_data[0, 2:5].transpose(0, 1) @ doutput_data[0, 2:5]
    torch.testing.assert_close(actual_dweight, expected_dweight)


def test_mudnn_frontend_python_graph_api_block_scale_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 5), data_type=fe.data_type.FLOAT)
    q, scale = graph.block_scale_quantize(x, block_size=2, name="quant")
    y = graph.block_scale_dequantize(q, scale, block_size=2, name="dequant")

    assert q.dims == x.dims
    assert scale.dims == (2, 3)
    assert y.dims == x.dims
    assert graph.nodes[0].op_type == "block_scale_quantize"
    assert graph.nodes[1].op_type == "block_scale_dequantize"
    assert graph.nodes[0].attributes["block_size"] == 2
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    list_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    list_x = list_graph.tensor(name="x", dim=(2, 5), data_type=fe.data_type.FLOAT)
    q_list, scale_list = list_graph.block_scale_quantize(
        list_x, block_size=[1, 2], name="quant_list"
    )
    y_list = list_graph.block_scale_dequantize(
        q_list, scale_list, block_size=[1, 2], name="dequant_list"
    )
    assert q_list.dims == list_x.dims
    assert scale_list.dims == (2, 3)
    assert y_list.dims == list_x.dims
    assert list_graph.nodes[0].attributes["block_size"] == 2
    list_graph.build()
    assert list_graph._native.operation_graph.ptr.value is not None

    neg_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    neg_x = neg_graph.tensor(name="x", dim=(2, 5), data_type=fe.data_type.FLOAT)
    neg_scale = neg_graph.tensor(name="scale", dim=(2, 3), data_type=fe.data_type.FLOAT)
    neg_y = neg_graph.block_scale_dequantize(
        neg_x,
        neg_scale,
        block_size=[1, 2],
        is_negative_scale=True,
        name="dequant_negative",
    )
    assert neg_y.dims == neg_x.dims
    assert neg_graph.nodes[0].attributes["is_negative_scale"] is True
    with pytest.raises(NotImplementedError, match="negative block scales"):
        neg_graph.build()

    int8_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    int8_x = int8_graph.tensor(name="x", dim=(1, 32, 32), data_type=fe.data_type.INT8)
    float_scale = int8_graph.tensor(
        name="scale", dim=(1, 32, 1), data_type=fe.data_type.FLOAT
    )
    int8_y = int8_graph.block_scale_dequantize(
        int8_x,
        float_scale,
        block_size=[1, 32],
        data_type=fe.data_type.FLOAT,
        name="dequant_int8",
    )
    assert int8_y.dims == int8_x.dims
    assert int8_y.data_type == fe.data_type.FLOAT

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    err_x = err_graph.tensor(name="x", dim=(2, 5), data_type=fe.data_type.FLOAT)
    bad_scale = err_graph.tensor(
        name="bad_scale", dim=(2, 2), data_type=fe.data_type.FLOAT
    )
    non_last_scale = err_graph.tensor(
        name="non_last_scale", dim=(1, 5), data_type=fe.data_type.FLOAT
    )
    bad_dtype = err_graph.tensor(
        name="bad_dtype", dim=(2, 3), data_type=fe.data_type.DOUBLE
    )
    with pytest.raises(ValueError, match="block_size"):
        err_graph.block_scale_quantize(err_x, block_size=0)
    with pytest.raises(ValueError, match="block_size"):
        err_graph.block_scale_quantize(err_x, block_size=[1, 0])
    non_last = err_graph.block_scale_dequantize(
        err_x, non_last_scale, block_size=[2, 1]
    )
    assert non_last.dims == err_x.dims
    with pytest.raises(NotImplementedError, match="last-dimension"):
        err_graph.build()
    with pytest.raises(NotImplementedError, match="multi-axis"):
        err_graph.block_scale_quantize(err_x, block_size=[2, 2])
    with pytest.raises(ValueError, match="dimensions"):
        err_graph.block_scale_dequantize(err_x, bad_scale, block_size=2)
    with pytest.raises(ValueError, match="data type"):
        err_graph.block_scale_dequantize(err_x, bad_dtype, block_size=2)


def test_mudnn_frontend_python_graph_api_block_scale_blackwell_symbolic():
    import mudnn

    batch, m, n, k, block_size = 1, 128, 128, 128, 32
    block_scale_dim_m = 128
    block_scale_dim_n = 128
    block_scale_dim_k = 4

    graph = mudnn.pygraph(
        io_data_type=mudnn.data_type.HALF,
        intermediate_data_type=mudnn.data_type.FLOAT,
        compute_data_type=mudnn.data_type.FLOAT,
    )
    tensor_a = graph.tensor(
        name="tensor_a",
        dim=(batch, m, k),
        stride=(m * k, k, 1),
        data_type=mudnn.data_type.FP8_E4M3,
    )
    tensor_b = graph.tensor(
        name="tensor_b",
        dim=(batch, k, n),
        stride=(k * n, 1, k),
        data_type=mudnn.data_type.FP8_E4M3,
    )
    block_descale_a = graph.tensor(
        name="block_descale_a",
        dim=(batch, block_scale_dim_m, block_scale_dim_k),
        stride=(block_scale_dim_m * block_scale_dim_k, block_scale_dim_k, 1),
        data_type=mudnn.data_type.FP8_E8M0,
        reordering_type=mudnn.tensor_reordering.F8_128x4,
    )
    block_descale_b = graph.tensor(
        name="block_descale_b",
        dim=(batch, block_scale_dim_k, block_scale_dim_n),
        stride=(block_scale_dim_n * block_scale_dim_k, 1, block_scale_dim_k),
        data_type=mudnn.data_type.FP8_E8M0,
        reordering_type=mudnn.tensor_reordering.F8_128x4,
    )

    dequant_a = graph.block_scale_dequantize(
        tensor_a, block_descale_a, block_size=[1, block_size], name="dequantize_a"
    )
    dequant_b = graph.block_scale_dequantize(
        tensor_b, block_descale_b, block_size=[block_size, 1], name="dequantize_b"
    )
    tensor_c = graph.matmul(
        dequant_a,
        dequant_b,
        compute_data_type=mudnn.data_type.FLOAT,
        name="matmul",
    )
    tensor_d, block_scale = graph.block_scale_quantize(
        tensor_c,
        block_size=block_size,
        axis=2,
        transpose=False,
        name="quantize_output",
    )
    tensor_d.set_output(True).set_data_type(mudnn.data_type.FP8_E4M3)
    block_scale.set_output(True).set_data_type(mudnn.data_type.FP8_E8M0)

    assert dequant_a.get_data_type() == mudnn.data_type.HALF
    assert dequant_b.get_data_type() == mudnn.data_type.HALF
    assert tensor_c.get_dim() == (batch, m, n)
    assert tensor_d.get_dim() == tensor_c.get_dim()
    assert block_scale.get_dim() == (batch, m, 4)
    assert graph.nodes[1].attributes["block_axis"] == 1
    graph.build_operation_graph().create_execution_plans(
        [mudnn.heur_mode.A, mudnn.heur_mode.FALLBACK]
    ).check_support().build_plans()
    assert graph._native.operation_graph is None


def test_mudnn_frontend_python_graph_api_paged_cache_load_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    container = graph.tensor(
        name="container", dim=(3, 2, 2), data_type=fe.data_type.FLOAT
    )
    sequence = graph.tensor(name="sequence", dim=(2,), data_type=fe.data_type.INT64)
    page_table = graph.tensor(name="page_table", dim=(3,), data_type=fe.data_type.INT64)
    y = graph.paged_cache_load(container, sequence, page_table, name="load")

    assert y.dims == (2, 2, 2)
    assert graph.nodes[0].op_type == "paged_cache_load"
    assert graph.nodes[0].inputs == {
        "container": container,
        "sequence": sequence,
        "page_table": page_table,
    }
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    err_container = err_graph.tensor(
        name="container", dim=(3, 2, 2), data_type=fe.data_type.FLOAT
    )
    bad_sequence_rank = err_graph.tensor(
        name="bad_sequence_rank", dim=(2, 1), data_type=fe.data_type.INT64
    )
    bad_sequence_dtype = err_graph.tensor(
        name="bad_sequence_dtype", dim=(2,), data_type=fe.data_type.FLOAT
    )
    bad_page_rank = err_graph.tensor(
        name="bad_page_rank", dim=(3, 1), data_type=fe.data_type.INT64
    )
    good_sequence = err_graph.tensor(
        name="good_sequence", dim=(2,), data_type=fe.data_type.INT64
    )
    good_page_table = err_graph.tensor(
        name="good_page_table", dim=(3,), data_type=fe.data_type.INT64
    )

    with pytest.raises(ValueError, match="sequence"):
        err_graph.paged_cache_load(err_container, bad_sequence_rank, good_page_table)
    with pytest.raises(ValueError, match="int64"):
        err_graph.paged_cache_load(err_container, bad_sequence_dtype, good_page_table)
    with pytest.raises(ValueError, match="page_table"):
        err_graph.paged_cache_load(err_container, good_sequence, bad_page_rank)
    with pytest.raises(ValueError, match="output data type"):
        err_graph.paged_cache_load(
            err_container,
            good_sequence,
            good_page_table,
            data_type=fe.data_type.DOUBLE,
        )


def test_mudnn_frontend_python_graph_api_signal_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 3), data_type=fe.data_type.FLOAT)
    flag = graph.tensor(name="flag", dim=(1,), data_type=fe.data_type.INT64)
    y = graph.signal(fe.SignalMode.SET, flag, x, value=11, name="set")

    assert y.dims == x.dims
    assert graph.nodes[0].op_type == "signal"
    assert graph.nodes[0].inputs == {"x": x}
    assert graph.nodes[0].outputs == {"y": y, "flag": flag}
    assert graph.nodes[0].attributes["mode"] == fe.SignalMode.SET
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    wait_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    wait_x = wait_graph.tensor(name="x", dim=(2, 3), data_type=fe.data_type.FLOAT)
    wait_flag = wait_graph.tensor(name="flag", dim=(1,), data_type=fe.data_type.INT64)
    wait_y = wait_graph.signal(fe.CUDNN_SIGNAL_WAIT, wait_flag, wait_x, value=11)
    assert wait_graph.nodes[0].inputs == {"x": wait_x, "flag": wait_flag}
    assert wait_graph.nodes[0].outputs == {"y": wait_y}

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    err_x = err_graph.tensor(name="x", dim=(2, 3), data_type=fe.data_type.FLOAT)
    bad_flag_dtype = err_graph.tensor(
        name="bad_flag_dtype", dim=(1,), data_type=fe.data_type.FLOAT
    )
    bad_flag_shape = err_graph.tensor(
        name="bad_flag_shape", dim=(2,), data_type=fe.data_type.INT64
    )
    good_flag = err_graph.tensor(
        name="good_flag", dim=(1,), data_type=fe.data_type.INT64
    )
    with pytest.raises(ValueError, match="int64"):
        err_graph.signal(fe.SignalMode.SET, bad_flag_dtype, err_x, value=1)
    with pytest.raises(ValueError, match="one element"):
        err_graph.signal(fe.SignalMode.SET, bad_flag_shape, err_x, value=1)
    with pytest.raises(ValueError, match="output data type"):
        err_graph.signal(
            fe.SignalMode.SET, good_flag, err_x, value=1, data_type=fe.DataType.DOUBLE
        )


def test_mudnn_frontend_python_graph_api_batch_norm_inference_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    mean = graph.tensor(name="mean", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    var = graph.tensor(name="var", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    scale = graph.tensor(name="scale", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    bias = graph.tensor(name="bias", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    y = graph.batch_norm_inference(x, mean, var, scale, bias, epsilon=1.0e-5, name="bn")

    assert y.dims == x.dims
    assert graph.nodes[0].op_type == "norm_fwd"
    assert graph.nodes[0].attributes["mode"] == fe.NormMode.BATCH_NORM
    assert graph.nodes[0].attributes["phase"] == fe.NormFwdPhase.INFERENCE
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    bad_x = err_graph.tensor(name="bad_x", dim=(2, 2, 2), data_type=fe.data_type.FLOAT)
    err_x = err_graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    bad_param = err_graph.tensor(
        name="bad_param", dim=(1, 3, 1, 1), data_type=fe.data_type.FLOAT
    )
    bad_dtype = err_graph.tensor(
        name="bad_dtype", dim=(1, 2, 1, 1), data_type=fe.data_type.DOUBLE
    )
    param = err_graph.tensor(
        name="param", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )

    with pytest.raises(ValueError, match="rank-4"):
        err_graph.batch_norm_inference(bad_x, param, param, param, param)
    with pytest.raises(ValueError, match="dimensions"):
        err_graph.batch_norm_inference(err_x, bad_param, param, param, param)
    with pytest.raises(ValueError, match="data type"):
        err_graph.batch_norm_inference(err_x, bad_dtype, param, param, param)
    with pytest.raises(ValueError, match="epsilon"):
        err_graph.batch_norm_inference(err_x, param, param, param, param, epsilon=-1.0)


def test_mudnn_frontend_python_graph_api_batch_norm_training_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    running_mean = graph.tensor(
        name="running_mean", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    running_var = graph.tensor(
        name="running_var", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    scale = graph.tensor(name="scale", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    bias = graph.tensor(name="bias", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)

    y, out_mean, out_var, saved_mean, saved_inv_var = graph.batch_norm_training(
        x, running_mean, running_var, scale, bias, momentum=0.25, epsilon=1.0e-5
    )

    assert y.dims == x.dims
    assert out_mean.dims == running_mean.dims
    assert out_var.dims == running_var.dims
    assert saved_mean.dims == running_mean.dims
    assert saved_inv_var.dims == running_var.dims
    assert graph.nodes[0].op_type == "norm_fwd"
    assert graph.nodes[0].attributes["phase"] == fe.NormFwdPhase.TRAINING
    assert graph.nodes[0].attributes["momentum"] == 0.25
    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    mixed_graph = fe.pygraph(
        io_data_type=fe.data_type.HALF,
        intermediate_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    mixed_x = mixed_graph.tensor(
        name="mixed_x", dim=(2, 2, 2, 2), data_type=fe.data_type.HALF
    )
    mixed_param = mixed_graph.tensor(
        name="mixed_param", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    mixed_y, mixed_saved_mean, mixed_saved_inv_var, _, _ = mixed_graph.batchnorm(
        input=mixed_x,
        scale=mixed_param,
        bias=mixed_param,
        in_running_mean=mixed_param,
        in_running_var=mixed_param,
        epsilon=mixed_graph.tensor_scalar(1.0e-5),
        momentum=mixed_graph.tensor_scalar(0.1),
    )
    assert mixed_y.data_type == fe.data_type.HALF
    mixed_saved_mean.set_data_type(fe.data_type.FLOAT)
    mixed_saved_inv_var.set_data_type(fe.data_type.FLOAT)
    mixed_graph.validate()

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    bad_x = err_graph.tensor(name="bad_x", dim=(2, 2, 2), data_type=fe.data_type.FLOAT)
    err_x = err_graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    bad_param = err_graph.tensor(
        name="bad_param", dim=(1, 3, 1, 1), data_type=fe.data_type.FLOAT
    )
    bad_dtype = err_graph.tensor(
        name="bad_dtype", dim=(1, 2, 1, 1), data_type=fe.data_type.DOUBLE
    )
    param = err_graph.tensor(
        name="param", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )

    with pytest.raises(ValueError, match="rank-4"):
        err_graph.batch_norm_training(bad_x, param, param, param, param)
    with pytest.raises(ValueError, match="dimensions"):
        err_graph.batch_norm_training(err_x, bad_param, param, param, param)
    with pytest.raises(ValueError, match="data type"):
        err_graph.batch_norm_training(err_x, bad_dtype, param, param, param)
    with pytest.raises(ValueError, match="epsilon"):
        err_graph.batch_norm_training(err_x, param, param, param, param, epsilon=-1.0)


def test_mudnn_frontend_python_graph_api_batch_norm_backward_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    dy = graph.tensor(name="dy", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    mean = graph.tensor(name="mean", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    var = graph.tensor(name="var", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    scale = graph.tensor(name="scale", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    dx, dscale, dbias = graph.batch_norm_backward(
        x, dy, mean, var, scale, epsilon=1.0e-5, name="bn_bwd"
    )

    assert dx.dims == x.dims
    assert dscale.dims == mean.dims
    assert dbias.dims == mean.dims
    assert graph.nodes[0].op_type == "norm_bwd"
    assert graph.nodes[0].attributes["mode"] == fe.NormMode.BATCH_NORM

    # The fourth operand is the *saved inverse standard deviation* of the
    # training forward pass, so it must be nameable as ``inv_variance`` -- the
    # name cuDNN uses for MUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC and
    # the name the C++ builder already uses. Naming it ``variance`` invited
    # callers to feed raw variance and silently get wrong gradients, because
    # the backend does not apply epsilon again on this path.
    kwargs_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    kwargs_x = kwargs_graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    kwargs_dy = kwargs_graph.tensor(name="dy", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    kwargs_mean = kwargs_graph.tensor(
        name="mean", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    kwargs_inv_variance = kwargs_graph.tensor(
        name="inv_variance", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    kwargs_scale = kwargs_graph.tensor(
        name="scale", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    kwargs_graph.batch_norm_backward(
        x=kwargs_x,
        dy=kwargs_dy,
        mean=kwargs_mean,
        inv_variance=kwargs_inv_variance,
        scale=kwargs_scale,
    )
    assert kwargs_graph.nodes[0].inputs["variance"] is kwargs_inv_variance

    graph.build()
    assert graph._native.operation_graph.ptr.value is not None

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    err_x = err_graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    bad_dy = err_graph.tensor(
        name="bad_dy", dim=(2, 2, 2, 1), data_type=fe.data_type.FLOAT
    )
    bad_dtype_dy = err_graph.tensor(
        name="bad_dtype_dy", dim=(2, 2, 2, 2), data_type=fe.data_type.DOUBLE
    )
    bad_param = err_graph.tensor(
        name="bad_param", dim=(1, 3, 1, 1), data_type=fe.data_type.FLOAT
    )
    param = err_graph.tensor(
        name="param", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )

    with pytest.raises(ValueError, match="dy"):
        err_graph.batch_norm_backward(err_x, bad_dy, param, param, param)
    with pytest.raises(ValueError, match="dy"):
        err_graph.batch_norm_backward(err_x, bad_dtype_dy, param, param, param)
    with pytest.raises(ValueError, match="dimensions"):
        err_graph.batch_norm_backward(err_x, err_x, bad_param, param, param)
    with pytest.raises(ValueError, match="epsilon"):
        err_graph.batch_norm_backward(err_x, err_x, param, param, param, epsilon=-1.0)


def test_mudnn_frontend_python_graph_api_cudnn_batchnorm_aliases():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT)
    running_mean = graph.tensor(
        name="running_mean", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    running_var = graph.tensor(
        name="running_var", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    scale = graph.tensor(name="scale", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    bias = graph.tensor(name="bias", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT)
    dy = graph.tensor(name="dy", dim=x.dims, data_type=fe.data_type.FLOAT)

    y, saved_mean, saved_inv_var, out_running_mean, out_running_var = graph.batchnorm(
        name="BN",
        input=x,
        scale=scale,
        bias=bias,
        in_running_mean=running_mean,
        in_running_var=running_var,
        epsilon=1.0e-5,
        momentum=0.25,
    )
    inf_y = graph.batchnorm_inference(
        input=x,
        mean=running_mean,
        inv_variance=running_var,
        scale=scale,
        bias=bias,
    )
    dx, dscale, dbias = graph.batchnorm_backward(
        name="DBN",
        grad=dy,
        input=x,
        scale=scale,
        mean=running_mean,
        inv_variance=running_var,
    )

    assert y.dims == x.dims
    assert saved_mean.dims == running_mean.dims
    assert saved_inv_var.dims == running_var.dims
    assert out_running_mean.dims == running_mean.dims
    assert out_running_var.dims == running_var.dims
    assert inf_y.dims == x.dims
    assert dx.dims == x.dims
    assert dscale.dims == running_mean.dims
    assert dbias.dims == running_mean.dims
    assert graph.nodes[0].attributes["phase"] == fe.NormFwdPhase.TRAINING
    assert graph.nodes[0].outputs["saved_mean"] is saved_mean
    assert graph.nodes[0].outputs["saved_inv_variance"] is saved_inv_var
    assert graph.nodes[1].attributes["phase"] == fe.NormFwdPhase.INFERENCE
    assert graph.nodes[2].op_type == "norm_bwd"

    scalar_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    scalar_x = scalar_graph.tensor(
        name="x", dim=(2, 2, 2, 2), data_type=fe.data_type.FLOAT
    )
    scalar_param = scalar_graph.tensor(
        name="param", dim=(1, 2, 1, 1), data_type=fe.data_type.FLOAT
    )
    epsilon_tensor = scalar_graph.tensor(
        name="epsilon", dim=(1,), data_type=fe.data_type.FLOAT
    )
    momentum_tensor = scalar_graph.tensor(
        name="momentum", dim=(1,), data_type=fe.data_type.FLOAT
    )
    scalar_graph.batchnorm_inference(
        input=scalar_x,
        mean=scalar_param,
        inv_variance=scalar_param,
        scale=scalar_param,
        bias=scalar_param,
        epsilon=epsilon_tensor,
    )
    scalar_graph.batchnorm(
        input=scalar_x,
        scale=scalar_param,
        bias=scalar_param,
        in_running_mean=scalar_param,
        in_running_var=scalar_param,
        epsilon=epsilon_tensor,
        momentum=momentum_tensor,
    )
    scalar_graph.batchnorm_backward(
        grad=scalar_x,
        input=scalar_x,
        scale=scalar_param,
        mean=scalar_param,
        inv_variance=scalar_param,
        epsilon=epsilon_tensor,
    )
    assert scalar_graph.nodes[0].inputs["epsilon"] is epsilon_tensor
    assert "epsilon" not in scalar_graph.nodes[0].attributes
    assert scalar_graph.nodes[1].inputs["epsilon"] is epsilon_tensor
    assert scalar_graph.nodes[1].inputs["momentum"] is momentum_tensor
    assert "epsilon" not in scalar_graph.nodes[1].attributes
    assert "momentum" not in scalar_graph.nodes[1].attributes
    assert scalar_graph.nodes[2].inputs["epsilon"] is epsilon_tensor
    assert "epsilon" not in scalar_graph.nodes[2].attributes


def test_mudnn_frontend_python_graph_api_cudnn_norm_aliases_symbolic_and_errors():
    import cudnn_frontend as fe

    graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    x = graph.tensor(name="x", dim=(4, 8, 1, 1), data_type=fe.data_type.FLOAT)
    scale = graph.tensor(name="scale", dim=(1, 8, 1, 1), data_type=fe.data_type.FLOAT)
    bias = graph.tensor(name="bias", dim=(1, 8, 1, 1), data_type=fe.data_type.FLOAT)
    dy = graph.tensor(name="dy", dim=x.dims, data_type=fe.data_type.FLOAT)

    y, mean, inv_var = graph.layernorm(
        name="LN",
        norm_forward_phase=fe.norm_fwd_phase.TRAINING,
        input=x,
        scale=scale,
        bias=bias,
        epsilon=1.0e-4,
    )
    dx, dscale, dbias = graph.layernorm_backward(
        name="DLN",
        grad=dy,
        input=x,
        scale=scale,
        mean=mean,
        inv_variance=inv_var,
    )
    ada_y, ada_mean, ada_inv_var = graph.adalayernorm(
        name="AdaLN",
        norm_forward_phase=fe.norm_fwd_phase.TRAINING,
        input=x,
        scale=scale,
        bias=bias,
        epsilon=1.0e-4,
    )
    ada_dx, ada_dscale, ada_dbias = graph.adalayernorm_backward(
        name="DAdaLN",
        grad=dy,
        input=x,
        scale=scale,
        mean=ada_mean,
        inv_variance=ada_inv_var,
    )
    rms_y, rms_inv_var = graph.rmsnorm(
        name="RMS",
        norm_forward_phase=fe.norm_fwd_phase.TRAINING,
        input=x,
        scale=scale,
        bias=None,
        epsilon=1.0e-5,
    )
    rms_dx, rms_dscale, rms_dbias = graph.rmsnorm_backward(
        name="DRMS",
        grad=dy,
        input=x,
        scale=scale,
        inv_variance=rms_inv_var,
        has_dbias=False,
    )
    in_y, in_mean, in_inv_var = graph.instancenorm(
        name="IN",
        norm_forward_phase=fe.norm_fwd_phase.TRAINING,
        input=x,
        scale=scale,
        bias=bias,
    )
    in_dx, in_dscale, in_dbias = graph.instancenorm_backward(
        name="DIN",
        grad=dy,
        input=x,
        scale=scale,
        mean=in_mean,
        inv_variance=in_inv_var,
    )

    assert y.dims == x.dims
    assert mean.dims == (4, 1, 1, 1)
    assert inv_var.dims == (4, 1, 1, 1)
    assert dx.dims == x.dims
    assert dscale.dims == scale.dims
    assert dbias.dims == scale.dims
    assert ada_y.dims == x.dims
    assert ada_mean.dims == (4, 1, 1, 1)
    assert ada_inv_var.dims == (4, 1, 1, 1)
    assert ada_dx.dims == x.dims
    assert ada_dscale.dims == scale.dims
    assert ada_dbias.dims == scale.dims
    assert rms_y.dims == x.dims
    assert rms_inv_var.dims == (4, 1, 1, 1)
    assert rms_dx.dims == x.dims
    assert rms_dscale.dims == scale.dims
    assert rms_dbias is None
    assert in_y.dims == x.dims
    assert in_mean.dims == (4, 8, 1, 1)
    assert in_inv_var.dims == (4, 8, 1, 1)
    assert in_dx.dims == x.dims
    assert in_dscale.dims == scale.dims
    assert in_dbias.dims == scale.dims
    assert graph.nodes[0].attributes["mode"] == fe.NormMode.LAYER_NORM
    assert graph.nodes[2].attributes["mode"] == fe.NormMode.ADA_LAYER_NORM
    assert graph.nodes[4].attributes["mode"] == fe.NormMode.RMS_NORM
    assert "bias" not in graph.nodes[4].inputs
    assert "mean" not in graph.nodes[5].inputs
    assert "dbias" not in graph.nodes[5].outputs
    assert graph.nodes[6].attributes["mode"] == fe.NormMode.INSTANCE_NORM

    rank2_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    rank2_x = rank2_graph.tensor(name="x", dim=(16, 32), data_type=fe.data_type.FLOAT)
    rank2_scale = rank2_graph.tensor(
        name="scale", dim=(1, 32), data_type=fe.data_type.FLOAT
    )
    rank2_y, rank2_inv_var = rank2_graph.rmsnorm(
        input=rank2_x,
        scale=rank2_scale,
    )
    assert rank2_y.dims == rank2_x.dims
    assert rank2_inv_var.dims == (16, 1)

    err_graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
    err_x = err_graph.tensor(name="x", dim=(4, 8, 1, 1), data_type=fe.data_type.FLOAT)
    err_scale = err_graph.tensor(
        name="scale", dim=(1, 8, 1, 1), data_type=fe.data_type.FLOAT
    )
    bad_scale = err_graph.tensor(
        name="bad_scale", dim=(1, 7, 1, 1), data_type=fe.data_type.FLOAT
    )
    bad_dy = err_graph.tensor(
        name="bad_dy", dim=(4, 8, 1, 2), data_type=fe.data_type.FLOAT
    )
    epsilon_tensor = err_graph.tensor(
        name="epsilon", dim=(1,), data_type=fe.data_type.FLOAT
    )
    eps_y, eps_mean, eps_inv_var = err_graph.layernorm(
        input=err_x,
        scale=err_scale,
        bias=err_scale,
        epsilon=epsilon_tensor,
    )
    assert eps_y.dims == err_x.dims
    assert eps_mean.dims == (4, 1, 1, 1)
    assert eps_inv_var.dims == (4, 1, 1, 1)
    assert err_graph.nodes[-1].inputs["epsilon"] is epsilon_tensor
    assert "epsilon" not in err_graph.nodes[-1].attributes
    ada_eps_y, ada_eps_mean, ada_eps_inv_var = err_graph.adalayernorm(
        input=err_x,
        scale=err_scale,
        bias=None,
        epsilon=epsilon_tensor,
    )
    assert ada_eps_y.dims == err_x.dims
    assert ada_eps_mean.dims == (4, 1, 1, 1)
    assert ada_eps_inv_var.dims == (4, 1, 1, 1)
    assert err_graph.nodes[-1].inputs["epsilon"] is epsilon_tensor
    assert err_graph.nodes[-1].attributes["mode"] == fe.NormMode.ADA_LAYER_NORM
    bad_epsilon = err_graph.tensor(
        name="bad_epsilon", dim=(2,), data_type=fe.data_type.FLOAT
    )
    with pytest.raises(ValueError, match="one element"):
        err_graph.layernorm(
            input=err_x,
            scale=err_scale,
            bias=err_scale,
            epsilon=bad_epsilon,
        )
    with pytest.raises(ValueError, match="dimensions"):
        err_graph.rmsnorm(input=err_x, scale=bad_scale)
    with pytest.raises(ValueError, match="grad"):
        err_graph.rmsnorm_backward(
            grad=bad_dy,
            input=err_x,
            scale=err_scale,
            inv_variance=err_graph.tensor(
                name="inv_var", dim=(4, 1, 1, 1), data_type=fe.data_type.FLOAT
            ),
        )
    with pytest.raises(ValueError, match="rank-4"):
        err_graph.instancenorm(
            input=rank2_x,
            scale=rank2_scale,
            bias=rank2_scale,
        )


def test_mudnn_frontend_python_graph_api_serialize_deserialize_symbolic():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.data_type.HALF,
        intermediate_data_type=fe.data_type.FLOAT,
        compute_data_type=fe.data_type.FLOAT,
    )
    a = graph.tensor(name="a", dim=(2, 3), data_type=fe.data_type.HALF).set_uid(10)
    b = graph.tensor(name="b", dim=(2, 3), data_type=fe.data_type.HALF).set_uid(11)
    epsilon = graph.tensor(
        name="epsilon",
        dim=(1,),
        data_type=fe.data_type.FLOAT,
        is_pass_by_value=True,
    ).set_uid(12)
    y = graph.pointwise(fe.pointwise_mode.ADD, a, b, name="y").set_uid(13)
    y.set_output(True)
    graph.create_execution_plans([fe.heur_mode.A, fe.heur_mode.FALLBACK])

    data = graph.serialize()
    assert isinstance(data, bytes)

    restored = fe.pygraph()
    restored.deserialize(data)

    assert restored.io_data_type == fe.data_type.HALF
    assert restored.intermediate_data_type == fe.data_type.FLOAT
    assert restored.compute_data_type == fe.data_type.FLOAT
    assert restored.execution_plan_modes == (fe.heur_mode.A, fe.heur_mode.FALLBACK)
    assert [tensor.uid for tensor in restored.tensors] == [10, 11, 12, 13]
    assert restored.tensors[2].is_by_value is True
    assert restored.tensors[3].is_output is True
    assert restored.nodes[0].op_type == "pointwise"
    assert restored.nodes[0].inputs["x"].uid == 10
    assert restored.nodes[0].outputs["y"].uid == 13
    assert restored.nodes[0].attributes["mode"] == fe.pointwise_mode.ADD

    restored_with_handle = fe.pygraph()
    handle = object()
    restored_with_handle.deserialize(handle, list(data))
    assert restored_with_handle.handle is handle
    assert restored_with_handle.nodes[0].inputs["b"].name == "b"


def test_mudnn_frontend_python_graph_api_execute_serialized_pointwise_add():
    import cudnn_frontend as fe

    a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    b_np = np.full((2, 3), 4.0, dtype=np.float32)
    y_np = np.zeros((2, 3), dtype=np.float32)

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        y = device_memory.create_tensor(0, y_np.nbytes, y_np.shape, "float32")
        tensors = [a, b, y]
        a_ptr = device_memory.get_tensor_addr(a)[0]
        b_ptr = device_memory.get_tensor_addr(b)[0]
        y_ptr = device_memory.get_tensor_addr(y)[0]

        graph = fe.pygraph(io_data_type=fe.data_type.FLOAT)
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.data_type.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.data_type.FLOAT)
        y_desc = graph.pointwise(fe.pointwise_mode.ADD, a_desc, b_desc, name="y")
        y_desc.set_output(True)
        graph.build([fe.heur_mode.INSTANT])

        restored = fe.pygraph()
        restored.deserialize(graph.serialize())
        restored.execute({"a": a_ptr, "b": b_ptr, "y": y_ptr})

        out = device_memory.copy_to_ndarray(y, y_np.nbytes)
        np.testing.assert_allclose(out, a_np + b_np)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_pointwise_add():
    import cudnn_frontend as fe

    a_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    b_np = np.full((2, 3), 2.0, dtype=np.float32)
    y_np = np.zeros((2, 3), dtype=np.float32)

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        y = device_memory.create_tensor(0, y_np.nbytes, y_np.shape, "float32")
        tensors = [a, b, y]
        a_ptr = device_memory.get_tensor_addr(a)[0]
        b_ptr = device_memory.get_tensor_addr(b)[0]
        y_ptr = device_memory.get_tensor_addr(y)[0]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.pointwise(fe.PointwiseMode.ADD, a_desc, b_desc, name="y")
        graph.execute({a_desc: a_ptr, b_desc: b_ptr, y_desc: y_ptr})

        out = device_memory.copy_to_ndarray(y, y_np.nbytes)
        np.testing.assert_allclose(out, a_np + b_np)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_disconnected_pointwise_outputs():
    import cudnn_frontend as fe

    x1_np = np.arange(12, dtype=np.float32).reshape(3, 4) / 17.0
    x2_np = np.arange(12, dtype=np.float32).reshape(3, 4) / 19.0 - 0.25
    cos1_np = np.full((3, 4), 0.75, dtype=np.float32)
    cos2_np = np.full((3, 4), 0.5, dtype=np.float32)
    sin1_np = np.full((3, 4), 0.125, dtype=np.float32)
    sin2_np = np.full((3, 4), 0.25, dtype=np.float32)
    y1_np = x1_np * cos1_np - x2_np * sin1_np
    y2_np = x2_np * cos2_np + x1_np * sin2_np

    arrays = [x1_np, x2_np, cos1_np, cos2_np, sin1_np, sin2_np]
    tensors = []
    try:
        input_tensors = [
            device_memory.create_tensor(
                array.__array_interface__["data"][0],
                array.nbytes,
                array.shape,
                "float32",
            )
            for array in arrays
        ]
        y1 = device_memory.create_tensor(0, y1_np.nbytes, y1_np.shape, "float32")
        y2 = device_memory.create_tensor(0, y2_np.nbytes, y2_np.shape, "float32")
        tensors = input_tensors + [y1, y2]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x1, x2, cos1, cos2, sin1, sin2 = [
            graph.tensor(name=name, dim=(3, 4), data_type=fe.DataType.FLOAT)
            for name in ("x1", "x2", "cos1", "cos2", "sin1", "sin2")
        ]
        y1_desc = graph.sub(graph.mul(x1, cos1), graph.mul(x2, sin1), name="y1")
        y1_desc.set_output(True)
        y2_desc = graph.add(graph.mul(x2, cos2), graph.mul(x1, sin2), name="y2")
        y2_desc.set_output(True)

        workspace_size = graph.get_workspace_size()
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, (workspace_size,), "int8"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]
        graph.execute(
            {
                x1: device_memory.get_tensor_addr(input_tensors[0])[0],
                x2: device_memory.get_tensor_addr(input_tensors[1])[0],
                cos1: device_memory.get_tensor_addr(input_tensors[2])[0],
                cos2: device_memory.get_tensor_addr(input_tensors[3])[0],
                sin1: device_memory.get_tensor_addr(input_tensors[4])[0],
                sin2: device_memory.get_tensor_addr(input_tensors[5])[0],
                y1_desc: device_memory.get_tensor_addr(y1)[0],
                y2_desc: device_memory.get_tensor_addr(y2)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual_y1 = device_memory.copy_to_ndarray(y1, y1_np.nbytes)
        actual_y2 = device_memory.copy_to_ndarray(y2, y2_np.nbytes)
        np.testing.assert_allclose(actual_y1, y1_np, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(actual_y2, y2_np, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_pass_by_value_numpy_input():
    import cudnn_frontend as fe

    x_np = np.arange(6, dtype=np.float32).reshape(2, 3)
    scale_np = np.array([[1.75]], dtype=np.float32)
    expected = x_np * scale_np

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        scale_desc = graph.tensor(
            name="scale",
            dim=scale_np.shape,
            stride=(1, 1),
            data_type=fe.DataType.FLOAT,
            is_pass_by_value=True,
        )
        y_desc = graph.mul(x_desc, scale_desc, name="y")
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                scale_desc: scale_np,
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_compile_time_scalar_alias():
    import mudnn

    graph = mudnn.pygraph(
        io_data_type=mudnn.data_type.FLOAT,
        compute_data_type=mudnn.data_type.FLOAT,
    )
    x = graph.tensor(
        name="x", dim=(1, 4), stride=(4, 1), data_type=mudnn.data_type.FLOAT
    )
    scale = graph.tensor_scalar(2.5, mudnn.scalar_type.COMPILE_TIME_CONST)
    scale.set_name("scale").set_dim((1, 1)).set_stride((1, 1))
    y = graph.mul(x, scale, name="mul")
    y.set_name("y").set_output(True).set_data_type(mudnn.data_type.FLOAT)

    assert scale.is_by_value
    assert scale.scalar_type == "COMPILE_TIME_CONST"
    assert scale.get_dim() == (1, 1)
    graph.validate()

    restored = mudnn.pygraph()
    restored.deserialize(graph.serialize())
    restored_scale = restored.query_tensor_attributes_of_uid(scale.uid)
    assert restored_scale.scalar_type == "COMPILE_TIME_CONST"
    assert restored_scale.get_data_type() == mudnn.data_type.FLOAT


def test_mudnn_frontend_python_graph_api_execute_extended_pointwise_aliases():
    import cudnn_frontend as fe

    a_np = np.array([[-1.2, 2.3, -3.7], [4.2, -5.5, 6.1]], dtype=np.float32)
    b_np = np.array([[2.0, 1.5, 0.5], [3.0, 2.0, 1.25]], dtype=np.float32)
    expected = np.maximum(a_np, b_np) + np.minimum(a_np, b_np) ** 2
    expected = np.remainder(expected, b_np)
    expected = np.power(expected + 1.25, b_np)
    expected = 1.0 / np.ceil(np.abs(expected))

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [a, b, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        max_desc = graph.max(input0=a_desc, input1=b_desc, name="max")
        min_desc = graph.min(input0=a_desc, input1=b_desc, name="min")
        add_square_desc = graph.add_square(max_desc, min_desc, name="add_square")
        mod_desc = graph.mod(input0=add_square_desc, input1=b_desc, name="mod")
        shifted_desc = graph.add(
            mod_desc,
            graph.tensor_scalar(1.25, data_type=fe.DataType.FLOAT),
            name="shift",
        )
        pow_desc = graph.pow(input0=shifted_desc, input1=b_desc, name="pow")
        y_desc = graph.reciprocal(
            graph.ceil(graph.abs(input=pow_desc, name="abs"), name="ceil"),
            name="reciprocal",
        )
        graph.build()
        workspace_ptr = None
        workspace_size = graph.get_workspace_size()
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, (workspace_size,), "int8"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]
        graph.execute(
            {
                a_desc: device_memory.get_tensor_addr(a)[0],
                b_desc: device_memory.get_tensor_addr(b)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=1.0e-3)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_activation_pointwise_aliases():
    import cudnn_frontend as fe

    x_np = np.array([[-2.0, -0.5, 0.0], [0.25, 1.0, 2.0]], dtype=np.float32)
    leaky = np.where(x_np >= 0.0, x_np, 0.2 * x_np)
    expected = np.log1p(np.exp(leaky)).astype(np.float32)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        leaky_desc = graph.leaky_relu(input=x_desc, negative_slope=0.2, name="leaky")
        y_desc = graph.softplus(input=leaky_desc, name="softplus")

        graph.build()
        workspace_ptr = None
        workspace_size = graph.get_workspace_size()
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, (workspace_size,), "int8"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=1.0e-3)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_reduction_sum():
    import cudnn_frontend as fe

    x_np = np.array([[1.0, -2.0, 3.0], [4.5, 0.5, -1.0]], dtype=np.float32)
    expected = x_np.sum(axis=1, keepdims=True)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.sum(x_desc, axis=1, name="sum")
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_reduction_mul_upstream_shape_edit():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [[1.0, -2.0, 3.0], [4.5, 0.5, -1.0]],
            [[2.0, 3.0, -0.5], [0.25, -4.0, 2.0]],
        ],
        dtype=np.float32,
    )
    expected = np.prod(x_np, axis=0, keepdims=True)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.reduction(x_desc, mode=fe.reduction_mode.MUL, name="mul_reduce")
        y_desc.set_dim(expected.shape).set_stride((6, 3, 1)).set_output(True)

        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_softmax_composite():
    import cudnn_frontend as fe

    x_np = np.array(
        [[1.0, -2.0, 3.0], [4.5, 0.5, -1.0]],
        dtype=np.float32,
    )
    x_max = x_np.max(axis=1, keepdims=True)
    exp = np.exp(x_np - x_max)
    sum_exp = exp.sum(axis=1, keepdims=True)
    expected_y = exp / sum_exp
    expected_stats = x_max + np.log(sum_exp)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        y = device_memory.create_tensor(
            0, expected_y.nbytes, expected_y.shape, "float32"
        )
        stats = device_memory.create_tensor(
            0, expected_stats.nbytes, expected_stats.shape, "float32"
        )
        tensors = [x, y, stats]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc, stats_desc = graph.softmax(
            x_desc,
            axis=1,
            name="softmax",
            generate_stats=True,
        )
        workspace_size = graph.get_workspace_size()
        workspace_ptr = None
        if workspace_size > 0:
            workspace = device_memory.create_tensor(
                0, workspace_size, (workspace_size,), "uint8"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(y)[0],
                stats_desc: device_memory.get_tensor_addr(stats)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual_y = device_memory.copy_to_ndarray(y, expected_y.nbytes)
        actual_stats = device_memory.copy_to_ndarray(stats, expected_stats.nbytes)
        np.testing.assert_allclose(actual_y, expected_y, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            actual_stats, expected_stats, rtol=1.0e-5, atol=1.0e-5
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_genstats():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
            ],
            [
                [[2.5, 1.0], [-3.0, 4.0]],
                [[0.75, -2.5], [1.5, 3.0]],
            ],
        ],
        dtype=np.float32,
    )
    expected_sum = x_np.sum(axis=(0, 2, 3), keepdims=True)
    expected_sqsum = np.square(x_np).sum(axis=(0, 2, 3), keepdims=True)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        sum_tensor = device_memory.create_tensor(
            0, expected_sum.nbytes, expected_sum.shape, "float32"
        )
        sqsum_tensor = device_memory.create_tensor(
            0, expected_sqsum.nbytes, expected_sqsum.shape, "float32"
        )
        tensors = [x, sum_tensor, sqsum_tensor]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        sum_desc, sqsum_desc = graph.genstats(x_desc, axis=(0, 2, 3), name="stats")
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                sum_desc: device_memory.get_tensor_addr(sum_tensor)[0],
                sqsum_desc: device_memory.get_tensor_addr(sqsum_tensor)[0],
            }
        )

        actual_sum = device_memory.copy_to_ndarray(sum_tensor, expected_sum.nbytes)
        actual_sqsum = device_memory.copy_to_ndarray(
            sqsum_tensor, expected_sqsum.nbytes
        )
        np.testing.assert_allclose(actual_sum, expected_sum, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            actual_sqsum, expected_sqsum, rtol=1.0e-5, atol=1.0e-5
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_bn_finalize_inference():
    import cudnn_frontend as fe

    scale_np = np.array([[[[1.5]], [[0.75]]]], dtype=np.float32)
    bias_np = np.array([[[[-0.25]], [[0.5]]]], dtype=np.float32)
    mean_np = np.array([[[[0.5]], [[-1.0]]]], dtype=np.float32)
    var_np = np.array([[[[4.0]], [[0.25]]]], dtype=np.float32)
    epsilon = 1.0e-3
    expected_eq_scale = scale_np / np.sqrt(var_np + np.float32(epsilon))
    expected_eq_bias = bias_np - mean_np * expected_eq_scale

    tensors = []
    try:
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float32",
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        mean = device_memory.create_tensor(
            mean_np.__array_interface__["data"][0],
            mean_np.nbytes,
            mean_np.shape,
            "float32",
        )
        var = device_memory.create_tensor(
            var_np.__array_interface__["data"][0],
            var_np.nbytes,
            var_np.shape,
            "float32",
        )
        eq_scale = device_memory.create_tensor(
            0, expected_eq_scale.nbytes, expected_eq_scale.shape, "float32"
        )
        eq_bias = device_memory.create_tensor(
            0, expected_eq_bias.nbytes, expected_eq_bias.shape, "float32"
        )
        tensors = [scale, bias, mean, var, eq_scale, eq_bias]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        mean_desc = graph.tensor(
            name="mean", dim=mean_np.shape, data_type=fe.DataType.FLOAT
        )
        var_desc = graph.tensor(
            name="var", dim=var_np.shape, data_type=fe.DataType.FLOAT
        )
        eq_scale_desc, eq_bias_desc = graph.bn_finalize_inference(
            scale_desc, bias_desc, mean_desc, var_desc, epsilon=epsilon
        )
        graph.execute(
            {
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                mean_desc: device_memory.get_tensor_addr(mean)[0],
                var_desc: device_memory.get_tensor_addr(var)[0],
                eq_scale_desc: device_memory.get_tensor_addr(eq_scale)[0],
                eq_bias_desc: device_memory.get_tensor_addr(eq_bias)[0],
            }
        )

        actual_eq_scale = device_memory.copy_to_ndarray(
            eq_scale, expected_eq_scale.nbytes
        )
        actual_eq_bias = device_memory.copy_to_ndarray(
            eq_bias, expected_eq_bias.nbytes
        )
        np.testing.assert_allclose(
            actual_eq_scale, expected_eq_scale, rtol=1.0e-5, atol=1.0e-5
        )
        np.testing.assert_allclose(
            actual_eq_bias, expected_eq_bias, rtol=1.0e-5, atol=1.0e-5
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_bn_finalize_training():
    import cudnn_frontend as fe

    accum_count_np = np.array([8], dtype=np.int64)
    momentum = 0.25
    epsilon = 1.0e-3
    y_sum_np = np.array([[[[4.0]], [[-8.0]]]], dtype=np.float32)
    y_sq_sum_np = np.array([[[[22.0]], [[18.0]]]], dtype=np.float32)
    scale_np = np.array([[[[1.5]], [[0.75]]]], dtype=np.float32)
    bias_np = np.array([[[[-0.25]], [[0.5]]]], dtype=np.float32)
    prev_mean_np = np.array([[[[1.0]], [[-2.0]]]], dtype=np.float32)
    prev_var_np = np.array([[[[3.0]], [[1.5]]]], dtype=np.float32)
    expected = _bn_finalize_training_reference(
        y_sum_np,
        y_sq_sum_np,
        scale_np,
        bias_np,
        prev_mean_np,
        prev_var_np,
        accum_count_np.item(),
        momentum,
        epsilon,
    )

    tensors = []
    try:
        inputs = []
        for array in (
            y_sum_np,
            y_sq_sum_np,
            scale_np,
            bias_np,
            prev_mean_np,
            prev_var_np,
            accum_count_np,
        ):
            dtype = "int64" if array.dtype == np.int64 else "float32"
            inputs.append(
                device_memory.create_tensor(
                    array.__array_interface__["data"][0],
                    array.nbytes,
                    array.shape,
                    dtype,
                )
            )
        outputs = [
            device_memory.create_tensor(0, value.nbytes, value.shape, "float32")
            for value in expected
        ]
        tensors = inputs + outputs

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        y_sum_desc = graph.tensor(
            name="y_sum", dim=y_sum_np.shape, data_type=fe.DataType.FLOAT
        )
        y_sq_sum_desc = graph.tensor(
            name="y_sq_sum", dim=y_sq_sum_np.shape, data_type=fe.DataType.FLOAT
        )
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        prev_mean_desc = graph.tensor(
            name="prev_mean", dim=prev_mean_np.shape, data_type=fe.DataType.FLOAT
        )
        prev_var_desc = graph.tensor(
            name="prev_var", dim=prev_var_np.shape, data_type=fe.DataType.FLOAT
        )
        accum_count_desc = graph.tensor(
            name="accum_count",
            dim=accum_count_np.shape,
            data_type=fe.DataType.INT64,
        )
        output_descs = graph.bn_finalize_training(
            y_sum_desc,
            y_sq_sum_desc,
            scale_desc,
            bias_desc,
            prev_mean_desc,
            prev_var_desc,
            accum_count_desc,
            momentum=momentum,
            epsilon=epsilon,
        )
        uid_to_ptr = {
            y_sum_desc: device_memory.get_tensor_addr(inputs[0])[0],
            y_sq_sum_desc: device_memory.get_tensor_addr(inputs[1])[0],
            scale_desc: device_memory.get_tensor_addr(inputs[2])[0],
            bias_desc: device_memory.get_tensor_addr(inputs[3])[0],
            prev_mean_desc: device_memory.get_tensor_addr(inputs[4])[0],
            prev_var_desc: device_memory.get_tensor_addr(inputs[5])[0],
            accum_count_desc: device_memory.get_tensor_addr(inputs[6])[0],
        }
        uid_to_ptr.update(
            {
                desc: device_memory.get_tensor_addr(tensor)[0]
                for desc, tensor in zip(output_descs, outputs)
            }
        )
        graph.execute(uid_to_ptr)

        for tensor, expected_value in zip(outputs, expected):
            actual = device_memory.copy_to_ndarray(tensor, expected_value.nbytes)
            np.testing.assert_allclose(actual, expected_value, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_bn_bwd_weights():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
            ],
            [
                [[2.5, 1.0], [-3.0, 4.0]],
                [[0.75, -2.5], [1.5, 3.0]],
            ],
        ],
        dtype=np.float32,
    )
    dy_np = np.array(
        [
            [
                [[0.5, -1.0], [2.0, -0.25]],
                [[1.25, -0.5], [0.75, 2.0]],
            ],
            [
                [[-1.5, 0.25], [1.0, 0.5]],
                [[-0.75, 1.5], [-2.0, 0.25]],
            ],
        ],
        dtype=np.float32,
    )
    mean_np = np.array([[[[0.5]], [[-1.0]]]], dtype=np.float32)
    invstd_np = np.array([[[[0.75]], [[1.25]]]], dtype=np.float32)
    scale_np = np.array([[[[1.5]], [[0.75]]]], dtype=np.float32)
    expected = _bn_bwd_weights_reference(x_np, dy_np, mean_np, invstd_np, scale_np)

    tensors = []
    try:
        inputs = []
        for array in (x_np, dy_np, mean_np, invstd_np, scale_np):
            inputs.append(
                device_memory.create_tensor(
                    array.__array_interface__["data"][0],
                    array.nbytes,
                    array.shape,
                    "float32",
                )
            )
        outputs = [
            device_memory.create_tensor(0, value.nbytes, value.shape, "float32")
            for value in expected
        ]
        tensors = inputs + outputs

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        mean_desc = graph.tensor(
            name="mean", dim=mean_np.shape, data_type=fe.DataType.FLOAT
        )
        invstd_desc = graph.tensor(
            name="invstd", dim=invstd_np.shape, data_type=fe.DataType.FLOAT
        )
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        output_descs = graph.bn_backward_weights(
            x_desc, dy_desc, mean_desc, invstd_desc, scale_desc, name="bn_bwd_w"
        )
        uid_to_ptr = {
            x_desc: device_memory.get_tensor_addr(inputs[0])[0],
            dy_desc: device_memory.get_tensor_addr(inputs[1])[0],
            mean_desc: device_memory.get_tensor_addr(inputs[2])[0],
            invstd_desc: device_memory.get_tensor_addr(inputs[3])[0],
            scale_desc: device_memory.get_tensor_addr(inputs[4])[0],
        }
        uid_to_ptr.update(
            {
                desc: device_memory.get_tensor_addr(tensor)[0]
                for desc, tensor in zip(output_descs, outputs)
            }
        )
        graph.execute(uid_to_ptr)

        for tensor, expected_value in zip(outputs, expected):
            actual = device_memory.copy_to_ndarray(tensor, expected_value.nbytes)
            np.testing.assert_allclose(actual, expected_value, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_concat():
    import cudnn_frontend as fe

    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np.float32)
    expected = np.concatenate((a_np, b_np), axis=1)

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [a, b, out]

        meta_graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        scratch = meta_graph._make_tensor(
            (1,),
            (1,),
            fe.DataType.FLOAT,
            False,
            False,
            name="scratch",
            uid=777,
        )
        assert meta_graph.query_tensor_attributes_of_uid(777) is scratch

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.concatenate((a_desc, b_desc), axis=1, name="concat")
        graph.execute(
            {
                a_desc: device_memory.get_tensor_addr(a)[0],
                b_desc: device_memory.get_tensor_addr(b)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_reshape():
    import cudnn_frontend as fe

    x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = x_np.reshape(2, 6)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.reshape(x_desc, shape=expected.shape, name="reshape")
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_transpose():
    import cudnn_frontend as fe

    x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    permutation = (1, 0, 2)
    expected = np.transpose(x_np, permutation)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.transpose(x_desc, permutation=permutation, name="transpose")
        device_memory.set_tensor_stride(out, y_desc.strides)
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_rng_uniform_with_offset():
    import cudnn_frontend as fe

    expected_shape = (2, 4)
    offset_np = np.array([3], dtype=np.int64)

    tensors = []
    try:
        out = device_memory.create_tensor(
            0,
            np.zeros(expected_shape, dtype=np.float32).nbytes,
            expected_shape,
            "float32",
        )
        offset = device_memory.create_tensor(
            offset_np.__array_interface__["data"][0],
            offset_np.nbytes,
            offset_np.shape,
            "int64",
        )
        tensors = [out, offset]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        offset_desc = graph.tensor(
            name="offset", dim=offset_np.shape, data_type=fe.DataType.INT64
        )
        y_desc = graph.rng(
            shape=expected_shape,
            seed=1234,
            distribution=fe.RngDistribution.UNIFORM,
            minimum=-0.25,
            maximum=0.75,
            offset=offset_desc,
            name="rng",
        )
        graph.execute(
            {
                offset_desc: device_memory.get_tensor_addr(offset)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(
            out, np.zeros(expected_shape, dtype=np.float32).nbytes
        )
        assert actual.shape == expected_shape
        assert np.all(actual >= -0.25)
        assert np.all(actual < 0.75)
        assert np.unique(actual).size > 1
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_signal_set():
    import cudnn_frontend as fe

    x_np = np.array([[1.0, -2.0, 3.5], [4.0, 5.0, -6.0]], dtype=np.float32)
    y_np = np.zeros_like(x_np)
    flag_np = np.array([0], dtype=np.int64)
    signal_value = 41

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        y = device_memory.create_tensor(0, y_np.nbytes, y_np.shape, "float32")
        flag = device_memory.create_tensor(
            flag_np.__array_interface__["data"][0],
            flag_np.nbytes,
            flag_np.shape,
            "int64",
        )
        tensors = [x, y, flag]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        flag_desc = graph.tensor(
            name="flag", dim=flag_np.shape, data_type=fe.DataType.INT64
        )
        y_desc = graph.signal(
            fe.SignalMode.SET, flag_desc, x_desc, value=signal_value, name="signal"
        )
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(y)[0],
                flag_desc: device_memory.get_tensor_addr(flag)[0],
            }
        )

        actual_y = device_memory.copy_to_ndarray(y, y_np.nbytes)
        actual_flag = device_memory.copy_to_ndarray(flag, flag_np.nbytes)
        np.testing.assert_allclose(actual_y, x_np, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_array_equal(
            actual_flag, np.array([signal_value], dtype=np.int64)
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_block_scale_quantize_dequantize():
    import cudnn_frontend as fe
    import mudnn_frontend._backend as backend

    x_np = np.array(
        [[1.0, -2.0, 0.5, 4.0, -5.0], [0.0, 0.0, 6.0, -3.0, 2.0]],
        dtype=np.float32,
    )
    block_size = 2
    expected_q, expected_scale, expected_y = _block_scale_quantize_reference(
        x_np, block_size
    )

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        q = device_memory.create_tensor(
            0, expected_q.nbytes, expected_q.shape, "float32"
        )
        scale = device_memory.create_tensor(
            0, expected_scale.nbytes, expected_scale.shape, "float32"
        )
        y = device_memory.create_tensor(
            0, expected_y.nbytes, expected_y.shape, "float32"
        )
        tensors = [x, q, scale, y]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        q_desc, scale_desc = graph.block_scale_quantize(
            x_desc, block_size=block_size, name="quant"
        )
        q_desc.set_output()
        scale_desc.set_output()
        y_desc = graph.block_scale_dequantize(
            q_desc, scale_desc, block_size=block_size, name="dequant"
        )
        graph.validate().build_operation_graph().create_execution_plans(
            [fe.HeurMode.INSTANT]
        ).check_support().build_plans()

        (workspace_size,) = graph._native.plans[0].get_int64(
            backend.MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE
        )
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, ((workspace_size + 3) // 4,), "float32"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]

        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                q_desc: device_memory.get_tensor_addr(q)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                y_desc: device_memory.get_tensor_addr(y)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual_q = device_memory.copy_to_ndarray(q, expected_q.nbytes)
        actual_scale = device_memory.copy_to_ndarray(scale, expected_scale.nbytes)
        actual_y = device_memory.copy_to_ndarray(y, expected_y.nbytes)
        np.testing.assert_allclose(actual_q, expected_q, rtol=1.0e-6, atol=1.0e-6)
        np.testing.assert_allclose(
            actual_scale, expected_scale, rtol=1.0e-6, atol=1.0e-6
        )
        np.testing.assert_allclose(actual_y, expected_y, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_paged_cache_load():
    import cudnn_frontend as fe

    container_np = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    sequence_np = np.array([2, 0], dtype=np.int64)
    page_table_np = np.array([1, 2, 0], dtype=np.int64)
    expected = container_np[page_table_np[sequence_np]]

    tensors = []
    try:
        container = device_memory.create_tensor(
            container_np.__array_interface__["data"][0],
            container_np.nbytes,
            container_np.shape,
            "float32",
        )
        sequence = device_memory.create_tensor(
            sequence_np.__array_interface__["data"][0],
            sequence_np.nbytes,
            sequence_np.shape,
            "int64",
        )
        page_table = device_memory.create_tensor(
            page_table_np.__array_interface__["data"][0],
            page_table_np.nbytes,
            page_table_np.shape,
            "int64",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [container, sequence, page_table, out]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        container_desc = graph.tensor(
            name="container", dim=container_np.shape, data_type=fe.DataType.FLOAT
        )
        sequence_desc = graph.tensor(
            name="sequence", dim=sequence_np.shape, data_type=fe.DataType.INT64
        )
        page_table_desc = graph.tensor(
            name="page_table", dim=page_table_np.shape, data_type=fe.DataType.INT64
        )
        y_desc = graph.paged_cache_load(
            container_desc, sequence_desc, page_table_desc, name="load"
        )
        graph.execute(
            {
                container_desc: device_memory.get_tensor_addr(container)[0],
                sequence_desc: device_memory.get_tensor_addr(sequence)[0],
                page_table_desc: device_memory.get_tensor_addr(page_table)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_batch_norm_inference():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
            ],
            [
                [[2.5, 1.0], [-3.0, 4.0]],
                [[0.75, -2.5], [1.5, 3.0]],
            ],
        ],
        dtype=np.float32,
    )
    mean_np = np.array([[[[0.5]], [[-1.0]]]], dtype=np.float32)
    var_np = np.array([[[[4.0]], [[0.25]]]], dtype=np.float32)
    scale_np = np.array([[[[1.5]], [[0.75]]]], dtype=np.float32)
    bias_np = np.array([[[[-0.25]], [[0.5]]]], dtype=np.float32)
    expected = (x_np - mean_np) / np.sqrt(var_np + np.float32(1.0e-5))
    expected = expected * scale_np + bias_np

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        mean = device_memory.create_tensor(
            mean_np.__array_interface__["data"][0],
            mean_np.nbytes,
            mean_np.shape,
            "float32",
        )
        var = device_memory.create_tensor(
            var_np.__array_interface__["data"][0],
            var_np.nbytes,
            var_np.shape,
            "float32",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float32",
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, mean, var, scale, bias, out]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        mean_desc = graph.tensor(
            name="mean", dim=mean_np.shape, data_type=fe.DataType.FLOAT
        )
        var_desc = graph.tensor(
            name="var", dim=var_np.shape, data_type=fe.DataType.FLOAT
        )
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        y_desc = graph.batch_norm_inference(
            x_desc,
            mean_desc,
            var_desc,
            scale_desc,
            bias_desc,
            epsilon=1.0e-5,
            name="bn",
        )
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                mean_desc: device_memory.get_tensor_addr(mean)[0],
                var_desc: device_memory.get_tensor_addr(var)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_batch_norm_training():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
            ],
            [
                [[2.5, 1.0], [-3.0, 4.0]],
                [[0.75, -2.5], [1.5, 3.0]],
            ],
        ],
        dtype=np.float32,
    )
    running_mean_np = np.array([[[[0.25]], [[-0.5]]]], dtype=np.float32)
    running_var_np = np.array([[[[1.5]], [[2.0]]]], dtype=np.float32)
    scale_np = np.array([[[[1.5]], [[0.75]]]], dtype=np.float32)
    bias_np = np.array([[[[-0.25]], [[0.5]]]], dtype=np.float32)
    momentum = np.float32(0.25)
    epsilon = np.float32(1.0e-5)

    mean_np = np.mean(x_np, axis=(0, 2, 3), keepdims=True).astype(np.float32)
    var_np = np.mean((x_np - mean_np) ** 2, axis=(0, 2, 3), keepdims=True).astype(
        np.float32
    )
    saved_inv_var_np = (1.0 / np.sqrt(var_np + epsilon)).astype(np.float32)
    expected_y = ((x_np - mean_np) * saved_inv_var_np * scale_np + bias_np).astype(
        np.float32
    )
    reduced_count = np.float32(x_np.shape[0] * x_np.shape[2] * x_np.shape[3])
    unbiased_var_np = (var_np * reduced_count / (reduced_count - 1.0)).astype(
        np.float32
    )
    expected_running_mean = (
        (np.float32(1.0) - momentum) * running_mean_np + momentum * mean_np
    ).astype(np.float32)
    expected_running_var = (
        (np.float32(1.0) - momentum) * running_var_np + momentum * unbiased_var_np
    ).astype(np.float32)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        running_mean = device_memory.create_tensor(
            running_mean_np.__array_interface__["data"][0],
            running_mean_np.nbytes,
            running_mean_np.shape,
            "float32",
        )
        running_var = device_memory.create_tensor(
            running_var_np.__array_interface__["data"][0],
            running_var_np.nbytes,
            running_var_np.shape,
            "float32",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float32",
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected_y.nbytes, expected_y.shape, "float32"
        )
        out_running_mean = device_memory.create_tensor(
            0,
            expected_running_mean.nbytes,
            expected_running_mean.shape,
            "float32",
        )
        out_running_var = device_memory.create_tensor(
            0,
            expected_running_var.nbytes,
            expected_running_var.shape,
            "float32",
        )
        saved_mean = device_memory.create_tensor(
            0, mean_np.nbytes, mean_np.shape, "float32"
        )
        saved_inv_var = device_memory.create_tensor(
            0, saved_inv_var_np.nbytes, saved_inv_var_np.shape, "float32"
        )
        tensors = [
            x,
            running_mean,
            running_var,
            scale,
            bias,
            out,
            out_running_mean,
            out_running_var,
            saved_mean,
            saved_inv_var,
        ]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        running_mean_desc = graph.tensor(
            name="running_mean",
            dim=running_mean_np.shape,
            data_type=fe.DataType.FLOAT,
        )
        running_var_desc = graph.tensor(
            name="running_var", dim=running_var_np.shape, data_type=fe.DataType.FLOAT
        )
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        (
            y_desc,
            out_running_mean_desc,
            out_running_var_desc,
            saved_mean_desc,
            saved_inv_var_desc,
        ) = graph.batch_norm_training(
            x_desc,
            running_mean_desc,
            running_var_desc,
            scale_desc,
            bias_desc,
            momentum=float(momentum),
            epsilon=float(epsilon),
            name="bn_train",
        )
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                running_mean_desc: device_memory.get_tensor_addr(running_mean)[0],
                running_var_desc: device_memory.get_tensor_addr(running_var)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
                out_running_mean_desc: device_memory.get_tensor_addr(
                    out_running_mean
                )[0],
                out_running_var_desc: device_memory.get_tensor_addr(out_running_var)[
                    0
                ],
                saved_mean_desc: device_memory.get_tensor_addr(saved_mean)[0],
                saved_inv_var_desc: device_memory.get_tensor_addr(saved_inv_var)[0],
            }
        )

        actual_y = device_memory.copy_to_ndarray(out, expected_y.nbytes)
        actual_running_mean = device_memory.copy_to_ndarray(
            out_running_mean, expected_running_mean.nbytes
        )
        actual_running_var = device_memory.copy_to_ndarray(
            out_running_var, expected_running_var.nbytes
        )
        actual_saved_mean = device_memory.copy_to_ndarray(saved_mean, mean_np.nbytes)
        actual_saved_inv_var = device_memory.copy_to_ndarray(
            saved_inv_var, saved_inv_var_np.nbytes
        )
        np.testing.assert_allclose(actual_y, expected_y, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            actual_running_mean, expected_running_mean, rtol=1.0e-5, atol=1.0e-5
        )
        np.testing.assert_allclose(
            actual_running_var, expected_running_var, rtol=1.0e-5, atol=1.0e-5
        )
        np.testing.assert_allclose(actual_saved_mean, mean_np, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            actual_saved_inv_var, saved_inv_var_np, rtol=1.0e-5, atol=1.0e-5
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_batch_norm_backward():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
            ],
            [
                [[2.5, 1.0], [-3.0, 4.0]],
                [[0.75, -2.5], [1.5, 3.0]],
            ],
        ],
        dtype=np.float32,
    )
    dy_np = np.array(
        [
            [
                [[0.5, -1.0], [2.0, -0.25]],
                [[1.25, -0.5], [0.75, 2.0]],
            ],
            [
                [[-1.5, 0.25], [1.0, 0.5]],
                [[-0.75, 1.5], [-2.0, 0.25]],
            ],
        ],
        dtype=np.float32,
    )
    mean_np = np.array([[[[0.5]], [[-1.0]]]], dtype=np.float32)
    # ``batch_norm_backward`` consumes the *saved inverse standard deviation*
    # from the training forward pass, so epsilon is already folded into it and
    # must not be applied a second time here.
    inv_std_np = np.array([[[[4.0]], [[0.25]]]], dtype=np.float32)
    scale_np = np.array([[[[1.5]], [[0.75]]]], dtype=np.float32)
    normalized_np = (x_np - mean_np) * inv_std_np
    reduce_axes = (0, 2, 3)
    sample_count = x_np.shape[0] * x_np.shape[2] * x_np.shape[3]
    expected_dscale = (dy_np * normalized_np).sum(axis=reduce_axes, keepdims=True)
    expected_dbias = dy_np.sum(axis=reduce_axes, keepdims=True)
    expected_dx = (
        scale_np
        * inv_std_np
        * (
            dy_np
            - expected_dbias / sample_count
            - normalized_np * expected_dscale / sample_count
        )
    ).astype(np.float32)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        mean = device_memory.create_tensor(
            mean_np.__array_interface__["data"][0],
            mean_np.nbytes,
            mean_np.shape,
            "float32",
        )
        inv_std = device_memory.create_tensor(
            inv_std_np.__array_interface__["data"][0],
            inv_std_np.nbytes,
            inv_std_np.shape,
            "float32",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float32",
        )
        dx = device_memory.create_tensor(
            0, expected_dx.nbytes, expected_dx.shape, "float32"
        )
        dscale = device_memory.create_tensor(
            0, expected_dscale.nbytes, expected_dscale.shape, "float32"
        )
        dbias = device_memory.create_tensor(
            0, expected_dbias.nbytes, expected_dbias.shape, "float32"
        )
        tensors = [x, dy, mean, inv_std, scale, dx, dscale, dbias]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        mean_desc = graph.tensor(
            name="mean", dim=mean_np.shape, data_type=fe.DataType.FLOAT
        )
        inv_std_desc = graph.tensor(
            name="inv_variance", dim=inv_std_np.shape, data_type=fe.DataType.FLOAT
        )
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        dx_desc, dscale_desc, dbias_desc = graph.batch_norm_backward(
            x_desc,
            dy_desc,
            mean_desc,
            inv_std_desc,
            scale_desc,
            epsilon=1.0e-5,
            name="bn_bwd",
        )
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                mean_desc: device_memory.get_tensor_addr(mean)[0],
                inv_std_desc: device_memory.get_tensor_addr(inv_std)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                dx_desc: device_memory.get_tensor_addr(dx)[0],
                dscale_desc: device_memory.get_tensor_addr(dscale)[0],
                dbias_desc: device_memory.get_tensor_addr(dbias)[0],
            }
        )

        actual_dx = device_memory.copy_to_ndarray(dx, expected_dx.nbytes)
        actual_dscale = device_memory.copy_to_ndarray(dscale, expected_dscale.nbytes)
        actual_dbias = device_memory.copy_to_ndarray(dbias, expected_dbias.nbytes)
        np.testing.assert_allclose(actual_dx, expected_dx, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            actual_dscale, expected_dscale, rtol=1.0e-5, atol=1.0e-5
        )
        np.testing.assert_allclose(
            actual_dbias, expected_dbias, rtol=1.0e-5, atol=1.0e-5
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_layernorm_forward():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
                [[0.75, -2.5], [1.5, 3.0]],
            ],
            [
                [[-0.5, 1.25], [2.5, -3.0]],
                [[4.0, -1.0], [0.25, 1.5]],
                [[-2.0, 0.5], [3.5, -0.75]],
            ],
        ],
        dtype=np.float32,
    )
    scale_np = np.linspace(0.5, 1.6, num=12, dtype=np.float32).reshape(1, 3, 2, 2)
    bias_np = np.linspace(-0.4, 0.7, num=12, dtype=np.float32).reshape(1, 3, 2, 2)
    epsilon_np = np.array([[[[1.0e-5]]]], dtype=np.float32)
    mean_np = x_np.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32)
    var_np = ((x_np - mean_np) ** 2).mean(axis=(1, 2, 3), keepdims=True)
    inv_std_np = (1.0 / np.sqrt(var_np + epsilon_np)).astype(np.float32)
    expected_y = ((x_np - mean_np) * inv_std_np * scale_np + bias_np).astype(np.float32)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float32",
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected_y.nbytes, expected_y.shape, "float32"
        )
        mean = device_memory.create_tensor(
            0, mean_np.nbytes, mean_np.shape, "float32"
        )
        inv_std = device_memory.create_tensor(
            0, inv_std_np.nbytes, inv_std_np.shape, "float32"
        )
        tensors = [x, scale, bias, out, mean, inv_std]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        epsilon_desc = graph.tensor(
            name="epsilon",
            dim=epsilon_np.shape,
            is_pass_by_value=True,
            data_type=fe.DataType.FLOAT,
        )
        y_desc, mean_desc, inv_std_desc = graph.layernorm(
            name="ln",
            norm_forward_phase=fe.norm_forward_phase.TRAINING,
            input=x_desc,
            scale=scale_desc,
            bias=bias_desc,
            epsilon=epsilon_desc,
        )
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                epsilon_desc: epsilon_np,
                y_desc: device_memory.get_tensor_addr(out)[0],
                mean_desc: device_memory.get_tensor_addr(mean)[0],
                inv_std_desc: device_memory.get_tensor_addr(inv_std)[0],
            }
        )

        actual_y = device_memory.copy_to_ndarray(out, expected_y.nbytes)
        actual_mean = device_memory.copy_to_ndarray(mean, mean_np.nbytes)
        actual_inv_std = device_memory.copy_to_ndarray(inv_std, inv_std_np.nbytes)
        np.testing.assert_allclose(actual_y, expected_y, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(actual_mean, mean_np, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(actual_inv_std, inv_std_np, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_layernorm_backward():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
                [[0.75, -2.5], [1.5, 3.0]],
            ],
            [
                [[-0.5, 1.25], [2.5, -3.0]],
                [[4.0, -1.0], [0.25, 1.5]],
                [[-2.0, 0.5], [3.5, -0.75]],
            ],
        ],
        dtype=np.float32,
    )
    dy_np = np.linspace(-0.8, 0.9, num=x_np.size, dtype=np.float32).reshape(x_np.shape)
    scale_np = np.linspace(0.5, 1.6, num=12, dtype=np.float32).reshape(1, 3, 2, 2)
    epsilon = np.float32(1.0e-5)
    mean_np = x_np.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32)
    var_np = ((x_np - mean_np) ** 2).mean(axis=(1, 2, 3), keepdims=True)
    inv_std_np = (1.0 / np.sqrt(var_np + epsilon)).astype(np.float32)
    xhat_np = (x_np - mean_np) * inv_std_np
    dy_scale_np = dy_np * scale_np
    reduced_count = np.float32(np.prod(x_np.shape[1:]))
    expected_dx = (
        inv_std_np
        * (
            dy_scale_np
            - dy_scale_np.sum(axis=(1, 2, 3), keepdims=True) / reduced_count
            - xhat_np
            * (dy_scale_np * xhat_np).sum(axis=(1, 2, 3), keepdims=True)
            / reduced_count
        )
    ).astype(np.float32)
    expected_dscale = (dy_np * xhat_np).sum(axis=0, keepdims=True).astype(np.float32)
    expected_dbias = dy_np.sum(axis=0, keepdims=True).astype(np.float32)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        mean = device_memory.create_tensor(
            mean_np.__array_interface__["data"][0],
            mean_np.nbytes,
            mean_np.shape,
            "float32",
        )
        inv_std = device_memory.create_tensor(
            inv_std_np.__array_interface__["data"][0],
            inv_std_np.nbytes,
            inv_std_np.shape,
            "float32",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float32",
        )
        dx = device_memory.create_tensor(
            0, expected_dx.nbytes, expected_dx.shape, "float32"
        )
        dscale = device_memory.create_tensor(
            0, expected_dscale.nbytes, expected_dscale.shape, "float32"
        )
        dbias = device_memory.create_tensor(
            0, expected_dbias.nbytes, expected_dbias.shape, "float32"
        )
        tensors = [x, dy, mean, inv_std, scale, dx, dscale, dbias]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        mean_desc = graph.tensor(
            name="mean", dim=mean_np.shape, data_type=fe.DataType.FLOAT
        )
        inv_std_desc = graph.tensor(
            name="inv_std", dim=inv_std_np.shape, data_type=fe.DataType.FLOAT
        )
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        dx_desc, dscale_desc, dbias_desc = graph.layernorm_backward(
            name="ln_bwd",
            grad=dy_desc,
            input=x_desc,
            scale=scale_desc,
            mean=mean_desc,
            inv_variance=inv_std_desc,
            epsilon=float(epsilon),
        )
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                mean_desc: device_memory.get_tensor_addr(mean)[0],
                inv_std_desc: device_memory.get_tensor_addr(inv_std)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                dx_desc: device_memory.get_tensor_addr(dx)[0],
                dscale_desc: device_memory.get_tensor_addr(dscale)[0],
                dbias_desc: device_memory.get_tensor_addr(dbias)[0],
            }
        )

        actual_dx = device_memory.copy_to_ndarray(dx, expected_dx.nbytes)
        actual_dscale = device_memory.copy_to_ndarray(dscale, expected_dscale.nbytes)
        actual_dbias = device_memory.copy_to_ndarray(dbias, expected_dbias.nbytes)
        np.testing.assert_allclose(actual_dx, expected_dx, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            actual_dscale, expected_dscale, rtol=1.0e-5, atol=1.0e-5
        )
        np.testing.assert_allclose(
            actual_dbias, expected_dbias, rtol=1.0e-5, atol=1.0e-5
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_rmsnorm_forward_backward():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
                [[0.75, -2.5], [1.5, 3.0]],
            ],
            [
                [[-0.5, 1.25], [2.5, -3.0]],
                [[4.0, -1.0], [0.25, 1.5]],
                [[-2.0, 0.5], [3.5, -0.75]],
            ],
        ],
        dtype=np.float32,
    )
    dy_np = np.linspace(-0.8, 0.9, num=x_np.size, dtype=np.float32).reshape(x_np.shape)
    scale_np = np.linspace(0.5, 1.6, num=12, dtype=np.float32).reshape(1, 3, 2, 2)
    epsilon = np.float32(1.0e-5)
    inv_rms_np = (
        1.0 / np.sqrt((x_np * x_np).mean(axis=(1, 2, 3), keepdims=True) + epsilon)
    ).astype(np.float32)
    expected_y = (x_np * inv_rms_np * scale_np).astype(np.float32)
    dy_scale_x = (dy_np * scale_np * x_np).sum(axis=(1, 2, 3), keepdims=True)
    reduced_count = np.float32(np.prod(x_np.shape[1:]))
    expected_dx = (
        dy_np * scale_np * inv_rms_np
        - x_np * (inv_rms_np**3) * dy_scale_x / reduced_count
    ).astype(np.float32)
    expected_dscale = (
        (dy_np * x_np * inv_rms_np).sum(axis=0, keepdims=True).astype(np.float32)
    )

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float32",
        )
        y = device_memory.create_tensor(
            0, expected_y.nbytes, expected_y.shape, "float32"
        )
        inv_rms = device_memory.create_tensor(
            0, inv_rms_np.nbytes, inv_rms_np.shape, "float32"
        )
        dx = device_memory.create_tensor(
            0, expected_dx.nbytes, expected_dx.shape, "float32"
        )
        dscale = device_memory.create_tensor(
            0, expected_dscale.nbytes, expected_dscale.shape, "float32"
        )
        tensors = [x, dy, scale, y, inv_rms, dx, dscale]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        y_desc, inv_rms_desc = graph.rmsnorm(
            name="rms",
            norm_forward_phase=fe.norm_forward_phase.TRAINING,
            input=x_desc,
            scale=scale_desc,
            epsilon=float(epsilon),
        )
        dx_desc, dscale_desc, dbias_desc = graph.rmsnorm_backward(
            name="rms_bwd",
            grad=dy_desc,
            input=x_desc,
            scale=scale_desc,
            inv_variance=inv_rms_desc,
            epsilon=float(epsilon),
            has_dbias=False,
        )
        assert dbias_desc is None
        workspace_size = graph.get_workspace_size()
        workspace = (
            device_memory.create_tensor(0, workspace_size, (workspace_size,), "uint8")
            if workspace_size
            else None
        )
        if workspace is not None:
            tensors.append(workspace)
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                y_desc: device_memory.get_tensor_addr(y)[0],
                inv_rms_desc: device_memory.get_tensor_addr(inv_rms)[0],
                dx_desc: device_memory.get_tensor_addr(dx)[0],
                dscale_desc: device_memory.get_tensor_addr(dscale)[0],
            },
            workspace_ptr=(
                device_memory.get_tensor_addr(workspace)[0]
                if workspace is not None
                else None
            ),
        )

        actual_y = device_memory.copy_to_ndarray(y, expected_y.nbytes)
        actual_dx = device_memory.copy_to_ndarray(dx, expected_dx.nbytes)
        actual_dscale = device_memory.copy_to_ndarray(dscale, expected_dscale.nbytes)
        np.testing.assert_allclose(actual_y, expected_y, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(actual_dx, expected_dx, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            actual_dscale, expected_dscale, rtol=1.0e-5, atol=1.0e-5
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_instancenorm_forward_backward():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
                [[0.75, -2.5], [1.5, 3.0]],
            ],
            [
                [[-0.5, 1.25], [2.5, -3.0]],
                [[4.0, -1.0], [0.25, 1.5]],
                [[-2.0, 0.5], [3.5, -0.75]],
            ],
        ],
        dtype=np.float32,
    )
    dy_np = np.linspace(-0.8, 0.9, num=x_np.size, dtype=np.float32).reshape(x_np.shape)
    scale_np = np.array([[[[0.5]], [[1.25]], [[1.75]]]], dtype=np.float32)
    bias_np = np.array([[[[-0.4]], [[0.1]], [[0.7]]]], dtype=np.float32)
    epsilon = np.float32(1.0e-5)
    mean_np = x_np.mean(axis=(2, 3), keepdims=True).astype(np.float32)
    var_np = ((x_np - mean_np) ** 2).mean(axis=(2, 3), keepdims=True)
    inv_std_np = (1.0 / np.sqrt(var_np + epsilon)).astype(np.float32)
    xhat_np = (x_np - mean_np) * inv_std_np
    expected_y = (xhat_np * scale_np + bias_np).astype(np.float32)
    dy_scale_np = dy_np * scale_np
    reduced_count = np.float32(np.prod(x_np.shape[2:]))
    expected_dx = (
        inv_std_np
        * (
            dy_scale_np
            - dy_scale_np.sum(axis=(2, 3), keepdims=True) / reduced_count
            - xhat_np
            * (dy_scale_np * xhat_np).sum(axis=(2, 3), keepdims=True)
            / reduced_count
        )
    ).astype(np.float32)
    expected_dscale = (
        (dy_np * xhat_np).sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    )
    expected_dbias = dy_np.sum(axis=(0, 2, 3), keepdims=True).astype(np.float32)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float32",
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        y = device_memory.create_tensor(
            0, expected_y.nbytes, expected_y.shape, "float32"
        )
        mean = device_memory.create_tensor(
            0, mean_np.nbytes, mean_np.shape, "float32"
        )
        inv_std = device_memory.create_tensor(
            0, inv_std_np.nbytes, inv_std_np.shape, "float32"
        )
        dx = device_memory.create_tensor(
            0, expected_dx.nbytes, expected_dx.shape, "float32"
        )
        dscale = device_memory.create_tensor(
            0, expected_dscale.nbytes, expected_dscale.shape, "float32"
        )
        dbias = device_memory.create_tensor(
            0, expected_dbias.nbytes, expected_dbias.shape, "float32"
        )
        tensors = [x, dy, scale, bias, y, mean, inv_std, dx, dscale, dbias]

        fwd_graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = fwd_graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        scale_desc = fwd_graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        bias_desc = fwd_graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        y_desc, mean_desc, inv_std_desc = fwd_graph.instancenorm(
            name="in",
            norm_forward_phase=fe.norm_forward_phase.TRAINING,
            input=x_desc,
            scale=scale_desc,
            bias=bias_desc,
            epsilon=float(epsilon),
        )
        fwd_graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                y_desc: device_memory.get_tensor_addr(y)[0],
                mean_desc: device_memory.get_tensor_addr(mean)[0],
                inv_std_desc: device_memory.get_tensor_addr(inv_std)[0],
            }
        )

        bwd_graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        bwd_x_desc = bwd_graph.tensor(
            name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT
        )
        dy_desc = bwd_graph.tensor(
            name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT
        )
        bwd_scale_desc = bwd_graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        bwd_mean_desc = bwd_graph.tensor(
            name="mean", dim=mean_np.shape, data_type=fe.DataType.FLOAT
        )
        bwd_inv_std_desc = bwd_graph.tensor(
            name="inv_std", dim=inv_std_np.shape, data_type=fe.DataType.FLOAT
        )
        dx_desc, dscale_desc, dbias_desc = bwd_graph.instancenorm_backward(
            name="in_bwd",
            grad=dy_desc,
            input=bwd_x_desc,
            scale=bwd_scale_desc,
            mean=bwd_mean_desc,
            inv_variance=bwd_inv_std_desc,
            epsilon=float(epsilon),
        )
        bwd_graph.execute(
            {
                bwd_x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                bwd_scale_desc: device_memory.get_tensor_addr(scale)[0],
                bwd_mean_desc: device_memory.get_tensor_addr(mean)[0],
                bwd_inv_std_desc: device_memory.get_tensor_addr(inv_std)[0],
                dx_desc: device_memory.get_tensor_addr(dx)[0],
                dscale_desc: device_memory.get_tensor_addr(dscale)[0],
                dbias_desc: device_memory.get_tensor_addr(dbias)[0],
            }
        )

        actual_y = device_memory.copy_to_ndarray(y, expected_y.nbytes)
        actual_mean = device_memory.copy_to_ndarray(mean, mean_np.nbytes)
        actual_inv_std = device_memory.copy_to_ndarray(inv_std, inv_std_np.nbytes)
        actual_dx = device_memory.copy_to_ndarray(dx, expected_dx.nbytes)
        actual_dscale = device_memory.copy_to_ndarray(dscale, expected_dscale.nbytes)
        actual_dbias = device_memory.copy_to_ndarray(dbias, expected_dbias.nbytes)
        np.testing.assert_allclose(actual_y, expected_y, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(actual_mean, mean_np, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(actual_inv_std, inv_std_np, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(actual_dx, expected_dx, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            actual_dscale, expected_dscale, rtol=1.0e-5, atol=1.0e-5
        )
        np.testing.assert_allclose(
            actual_dbias, expected_dbias, rtol=1.0e-5, atol=1.0e-5
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_instancenorm_half_with_float_stats():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [
                [[1.0, -2.0], [3.0, 0.5]],
                [[-1.5, 2.0], [4.0, -0.25]],
            ],
            [
                [[-0.5, 1.25], [2.5, -3.0]],
                [[4.0, -1.0], [0.25, 1.5]],
            ],
        ],
        dtype=np.float16,
    )
    dy_np = np.array(
        [
            [
                [[0.25, -0.5], [1.25, 0.75]],
                [[-1.0, 0.5], [0.125, -0.25]],
            ],
            [
                [[-0.75, 0.875], [0.375, -1.125]],
                [[1.5, -0.625], [0.25, 0.5]],
            ],
        ],
        dtype=np.float16,
    )
    scale_np = np.array([[[[0.5]], [[1.25]]]], dtype=np.float16)
    bias_np = np.array([[[[-0.4]], [[0.1]]]], dtype=np.float16)
    epsilon_np = np.array([[[[1.0e-5]]]], dtype=np.float32)

    x_f32 = x_np.astype(np.float32)
    dy_f32 = dy_np.astype(np.float32)
    scale_f32 = scale_np.astype(np.float32)
    bias_f32 = bias_np.astype(np.float32)
    mean_np = x_f32.mean(axis=(2, 3), keepdims=True).astype(np.float32)
    var_np = ((x_f32 - mean_np) ** 2).mean(axis=(2, 3), keepdims=True)
    inv_std_np = (1.0 / np.sqrt(var_np + epsilon_np)).astype(np.float32)
    xhat_np = (x_f32 - mean_np) * inv_std_np
    expected_y = (xhat_np * scale_f32 + bias_f32).astype(np.float16)
    dy_scale_np = dy_f32 * scale_f32
    reduced_count = np.float32(np.prod(x_np.shape[2:]))
    expected_dx = (
        inv_std_np
        * (
            dy_scale_np
            - dy_scale_np.sum(axis=(2, 3), keepdims=True) / reduced_count
            - xhat_np
            * (dy_scale_np * xhat_np).sum(axis=(2, 3), keepdims=True)
            / reduced_count
        )
    ).astype(np.float16)
    expected_dscale = (
        (dy_f32 * xhat_np).sum(axis=(0, 2, 3), keepdims=True).astype(np.float16)
    )
    expected_dbias = dy_f32.sum(axis=(0, 2, 3), keepdims=True).astype(np.float16)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float16",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float16",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float16",
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float16",
        )
        y = device_memory.create_tensor(
            0, expected_y.nbytes, expected_y.shape, "float16"
        )
        mean = device_memory.create_tensor(
            0, mean_np.nbytes, mean_np.shape, "float32"
        )
        inv_std = device_memory.create_tensor(
            0, inv_std_np.nbytes, inv_std_np.shape, "float32"
        )
        dx = device_memory.create_tensor(
            0, expected_dx.nbytes, expected_dx.shape, "float16"
        )
        dscale = device_memory.create_tensor(
            0, expected_dscale.nbytes, expected_dscale.shape, "float16"
        )
        dbias = device_memory.create_tensor(
            0, expected_dbias.nbytes, expected_dbias.shape, "float16"
        )
        tensors = [x, dy, scale, bias, y, mean, inv_std, dx, dscale, dbias]

        fwd_graph = fe.pygraph(
            io_data_type=fe.DataType.HALF,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = fwd_graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.HALF)
        scale_desc = fwd_graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.HALF
        )
        bias_desc = fwd_graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.HALF
        )
        epsilon_desc = fwd_graph.tensor(
            name="epsilon",
            dim=epsilon_np.shape,
            stride=(1, 1, 1, 1),
            data_type=fe.DataType.FLOAT,
            is_pass_by_value=True,
        )
        y_desc, mean_desc, inv_std_desc = fwd_graph.instancenorm(
            name="in_half",
            norm_forward_phase=fe.norm_forward_phase.TRAINING,
            input=x_desc,
            scale=scale_desc,
            bias=bias_desc,
            epsilon=epsilon_desc,
        )
        fwd_graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                epsilon_desc: epsilon_np,
                y_desc: device_memory.get_tensor_addr(y)[0],
                mean_desc: device_memory.get_tensor_addr(mean)[0],
                inv_std_desc: device_memory.get_tensor_addr(inv_std)[0],
            }
        )

        actual_y = device_memory.copy_to_ndarray(y, expected_y.nbytes)
        actual_mean = device_memory.copy_to_ndarray(mean, mean_np.nbytes)
        actual_inv_std = device_memory.copy_to_ndarray(inv_std, inv_std_np.nbytes)
        np.testing.assert_allclose(actual_y, expected_y, rtol=1.0e-3, atol=1.0e-3)
        np.testing.assert_allclose(actual_mean, mean_np, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(actual_inv_std, inv_std_np, rtol=1.0e-5, atol=1.0e-5)

        bwd_graph = fe.pygraph(
            io_data_type=fe.DataType.HALF,
            compute_data_type=fe.DataType.FLOAT,
        )
        bwd_x_desc = bwd_graph.tensor(
            name="x", dim=x_np.shape, data_type=fe.DataType.HALF
        )
        dy_desc = bwd_graph.tensor(
            name="dy", dim=dy_np.shape, data_type=fe.DataType.HALF
        )
        bwd_scale_desc = bwd_graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.HALF
        )
        bwd_mean_desc = bwd_graph.tensor(
            name="mean", dim=mean_np.shape, data_type=fe.DataType.FLOAT
        )
        bwd_inv_std_desc = bwd_graph.tensor(
            name="inv_std", dim=inv_std_np.shape, data_type=fe.DataType.FLOAT
        )
        bwd_epsilon_desc = bwd_graph.tensor(
            name="epsilon",
            dim=epsilon_np.shape,
            stride=(1, 1, 1, 1),
            data_type=fe.DataType.FLOAT,
            is_pass_by_value=True,
        )
        dx_desc, dscale_desc, dbias_desc = bwd_graph.instancenorm_backward(
            name="in_half_bwd",
            grad=dy_desc,
            input=bwd_x_desc,
            scale=bwd_scale_desc,
            mean=bwd_mean_desc,
            inv_variance=bwd_inv_std_desc,
            epsilon=bwd_epsilon_desc,
        )
        bwd_graph.execute(
            {
                bwd_x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                bwd_scale_desc: device_memory.get_tensor_addr(scale)[0],
                bwd_mean_desc: device_memory.get_tensor_addr(mean)[0],
                bwd_inv_std_desc: device_memory.get_tensor_addr(inv_std)[0],
                bwd_epsilon_desc: epsilon_np,
                dx_desc: device_memory.get_tensor_addr(dx)[0],
                dscale_desc: device_memory.get_tensor_addr(dscale)[0],
                dbias_desc: device_memory.get_tensor_addr(dbias)[0],
            }
        )

        actual_dx = device_memory.copy_to_ndarray(dx, expected_dx.nbytes)
        actual_dscale = device_memory.copy_to_ndarray(dscale, expected_dscale.nbytes)
        actual_dbias = device_memory.copy_to_ndarray(dbias, expected_dbias.nbytes)
        np.testing.assert_allclose(actual_dx, expected_dx, rtol=1.0e-3, atol=1.0e-3)
        np.testing.assert_allclose(
            actual_dscale, expected_dscale, rtol=1.0e-3, atol=1.0e-3
        )
        np.testing.assert_allclose(
            actual_dbias, expected_dbias, rtol=1.0e-3, atol=1.0e-3
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_rmsnorm_swish_composite():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [[[1.0]], [[-2.0]], [[3.0]], [[0.5]]],
            [[[-1.5]], [[2.0]], [[4.0]], [[-0.25]]],
            [[[0.75]], [[-2.5]], [[1.5]], [[3.0]]],
        ],
        dtype=np.float32,
    )
    scale_np = np.array([[[[0.5]], [[1.25]], [[0.75]], [[1.5]]]], dtype=np.float32)
    epsilon_np = np.array([[[[1.0e-5]]]], dtype=np.float32)
    inv_rms_np = (
        1.0 / np.sqrt((x_np * x_np).mean(axis=(1, 2, 3), keepdims=True) + epsilon_np)
    ).astype(np.float32)
    rms_np = (x_np * inv_rms_np * scale_np).astype(np.float32)
    expected = (rms_np / (1.0 + np.exp(-rms_np))).astype(np.float32)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, scale, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.FLOAT
        )
        epsilon_desc = graph.tensor(
            name="epsilon",
            dim=epsilon_np.shape,
            stride=(1, 1, 1, 1),
            data_type=fe.DataType.FLOAT,
            is_pass_by_value=True,
        )
        rms_desc, _ = graph.rmsnorm(
            name="RmsNorm",
            norm_forward_phase=fe.norm_forward_phase.INFERENCE,
            input=x_desc,
            scale=scale_desc,
            epsilon=epsilon_desc,
        )
        z_desc = graph.swish(input=rms_desc, swish_beta=1.0, name="SiLU")
        z_desc.set_output(True)

        workspace_size = graph.get_workspace_size()
        workspace = (
            device_memory.create_tensor(0, workspace_size, (workspace_size,), "uint8")
            if workspace_size
            else None
        )
        if workspace is not None:
            tensors.append(workspace)
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                epsilon_desc: epsilon_np,
                z_desc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=(
                device_memory.get_tensor_addr(workspace)[0]
                if workspace is not None
                else None
            ),
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_rmsnorm_swish_fake_cuda_cpu_fallback(
    monkeypatch,
):
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *args, **kwargs: "Fake CPU CUDA"
    )

    x_value = torch.tensor(
        [[1.0, -2.0, 3.0, 0.5], [-1.5, 2.0, 4.0, -0.25]],
        dtype=torch.bfloat16,
    )
    scale_value = torch.tensor([[0.5, 1.25, 0.75, 1.5]], dtype=torch.bfloat16)
    out_value = torch.empty_like(x_value)
    epsilon = 1.0e-5

    graph = fe.pygraph(
        io_data_type=fe.DataType.BFLOAT16,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor(
        name="x",
        dim=x_value.shape,
        stride=x_value.stride(),
        data_type=fe.DataType.BFLOAT16,
    )
    scale_desc = graph.tensor(
        name="scale",
        dim=scale_value.shape,
        stride=scale_value.stride(),
        data_type=fe.DataType.BFLOAT16,
    )
    rms_desc, _ = graph.rmsnorm(
        name="RmsNorm",
        norm_forward_phase=fe.norm_forward_phase.INFERENCE,
        input=x_desc,
        scale=scale_desc,
        epsilon=epsilon,
    )
    z_desc = graph.swish(input=rms_desc, swish_beta=1.0, name="SiLU")
    z_desc.set_output(True)

    graph.build()
    assert type(graph._native).__name__ == "_CpuFallbackNativeGraph"
    graph.execute({x_desc: x_value, scale_desc: scale_value, z_desc: out_value})

    x_float = x_value.to(dtype=torch.float32)
    inv_rms = torch.rsqrt((x_float * x_float).mean(dim=1, keepdim=True) + epsilon)
    rms = (scale_value * (x_value * inv_rms.to(dtype=x_value.dtype))).to(
        dtype=torch.bfloat16
    )
    expected = (rms * torch.sigmoid(rms)).to(dtype=torch.bfloat16)
    torch.testing.assert_close(out_value, expected, rtol=0.0, atol=0.0)


def test_mudnn_frontend_python_graph_api_execute_rmsnorm_swish_fp8_fake_cuda_cpu_fallback(
    monkeypatch,
):
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch does not expose float8_e4m3fn")
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *args, **kwargs: "Fake CPU CUDA"
    )

    tokens, channels = 1560, 64
    base = torch.linspace(-2.0, 2.0, tokens * channels, dtype=torch.float32)
    x_value = base.reshape(tokens, channels, 1, 1).to(dtype=torch.bfloat16)
    scale_value = torch.linspace(0.5, 1.5, channels, dtype=torch.float32).reshape(
        1, channels, 1, 1
    )
    scale_value = scale_value.to(dtype=torch.bfloat16)
    out_value = torch.empty(x_value.shape, dtype=torch.float8_e4m3fn)
    epsilon = 1.0e-6 / channels

    graph = fe.pygraph(
        io_data_type=fe.DataType.BFLOAT16,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor(
        name="x",
        dim=x_value.shape,
        stride=x_value.stride(),
        data_type=fe.DataType.BFLOAT16,
    )
    scale_desc = graph.tensor(
        name="scale",
        dim=scale_value.shape,
        stride=scale_value.stride(),
        data_type=fe.DataType.BFLOAT16,
    )
    rms_desc, _ = graph.rmsnorm(
        name="RmsNorm",
        norm_forward_phase=fe.norm_forward_phase.INFERENCE,
        input=x_desc,
        scale=scale_desc,
        epsilon=epsilon,
    )
    z_desc = graph.swish(input=rms_desc, swish_beta=1.0, name="SiLU")
    z_desc.set_output(True).set_data_type(fe.DataType.FP8_E4M3)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([fe.heur_mode.OPENSOURCE])
    graph.check_support().build_plans(fe.build_plan_policy.HEURISTICS_CHOICE)
    assert type(graph._native).__name__ == "_CpuFallbackNativeGraph"
    graph.execute({x_desc: x_value, scale_desc: scale_value, z_desc: out_value})

    x_float = x_value.to(dtype=torch.float32)
    inv_rms = torch.rsqrt(
        (x_float * x_float).mean(dim=(1, 2, 3), keepdim=True) + epsilon
    )
    rms = x_float * inv_rms * scale_value.to(dtype=torch.float32)
    expected = (rms * torch.sigmoid(rms)).to(dtype=torch.float8_e4m3fn)
    torch.testing.assert_close(out_value.float(), expected.float(), rtol=0.0, atol=0.0)


def test_mudnn_frontend_python_graph_api_execute_rmsnorm_swish_nvfp4_fake_cuda_cpu_fallback(
    monkeypatch,
):
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch does not expose float8_e4m3fn")
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *args, **kwargs: "Fake CPU CUDA"
    )

    tokens, channels = 1560, 64
    x_value = torch.linspace(-2.0, 2.0, tokens * channels, dtype=torch.float32).reshape(
        tokens, channels, 1, 1
    )
    x_value = x_value.to(dtype=torch.bfloat16)
    scale_value = torch.linspace(0.5, 1.5, channels, dtype=torch.float32).reshape(
        1, channels, 1, 1
    )
    scale_value = scale_value.to(dtype=torch.bfloat16)
    out_value = torch.empty(tokens * channels // 2, dtype=torch.uint8)
    epsilon = 1.0e-6 / channels

    graph = fe.pygraph(
        io_data_type=fe.DataType.BFLOAT16,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor(
        name="x",
        dim=x_value.shape,
        stride=x_value.stride(),
        data_type=fe.DataType.BFLOAT16,
    )
    scale_desc = graph.tensor(
        name="scale",
        dim=scale_value.shape,
        stride=scale_value.stride(),
        data_type=fe.DataType.BFLOAT16,
    )
    rms_desc, _ = graph.rmsnorm(
        name="RmsNorm",
        norm_forward_phase=fe.norm_forward_phase.INFERENCE,
        input=x_desc,
        scale=scale_desc,
        epsilon=epsilon,
    )
    z_desc = graph.swish(input=rms_desc, swish_beta=1.0, name="SiLU")
    z_desc.set_output(True).set_data_type(fe.DataType.FP4_E2M1)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([fe.heur_mode.OPENSOURCE])
    graph.check_support().build_plans(fe.build_plan_policy.HEURISTICS_CHOICE)
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8)
    assert workspace.numel() >= tokens * (channels // 16)
    graph.execute(
        {x_desc: x_value, scale_desc: scale_value, z_desc: out_value},
        workspace,
    )

    x_float = x_value.to(dtype=torch.float32)
    inv_rms = torch.rsqrt(
        (x_float * x_float).mean(dim=(1, 2, 3), keepdim=True) + epsilon
    )
    silu = x_float * inv_rms * scale_value.to(dtype=torch.float32)
    silu = silu * torch.sigmoid(silu)
    block_values = silu.reshape(tokens, channels // 16, 16)
    block_scale = (
        torch.clamp(
            block_values.abs().amax(dim=2, keepdim=True),
            min=torch.finfo(torch.float32).tiny,
        )
        / 6.0
    )
    scaled = block_values / block_scale
    fp4_positive = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    nibbles = (
        (scaled.abs().unsqueeze(-1) - fp4_positive).abs().argmin(dim=-1).to(torch.uint8)
        + (scaled < 0).to(torch.uint8) * 8
    ).reshape(tokens, channels)
    expected = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).reshape(-1)
    torch.testing.assert_close(out_value, expected)


def test_mudnn_frontend_python_graph_api_opensource_rmsnorm_swish_unsupported_shape():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.DataType.BFLOAT16,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor(
        name="x",
        dim=(1560, 48, 1, 1),
        stride=(48, 1, 1, 1),
        data_type=fe.DataType.BFLOAT16,
    )
    scale_desc = graph.tensor(
        name="scale",
        dim=(1, 48, 1, 1),
        stride=(48, 1, 1, 1),
        data_type=fe.DataType.BFLOAT16,
    )
    rms_desc, _ = graph.rmsnorm(
        name="RmsNorm",
        norm_forward_phase=fe.norm_forward_phase.INFERENCE,
        input=x_desc,
        scale=scale_desc,
        epsilon=1.0e-6 / 48,
    )
    z_desc = graph.swish(input=rms_desc, swish_beta=1.0, name="SiLU")
    z_desc.set_output(True)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([fe.heur_mode.OPENSOURCE])
    with pytest.raises(fe.cudnnGraphNotSupportedError, match="channels=48"):
        graph.check_support()


def test_mudnn_frontend_python_graph_api_execute_layernorm_half_with_float_stats():
    import cudnn_frontend as fe

    x_np = np.array(
        [
            [[[-0.75, 1.25]], [[2.5, -1.5]]],
            [[[0.5, -2.0]], [[1.0, 3.0]]],
        ],
        dtype=np.float16,
    )
    dy_np = np.array(
        [
            [[[0.25, -0.5]], [[1.25, 0.75]]],
            [[[-1.0, 0.5]], [[0.125, -0.25]]],
        ],
        dtype=np.float16,
    )
    scale_np = np.array([[[[0.75, 1.25]], [[1.5, 0.5]]]], dtype=np.float16)
    bias_np = np.array([[[[-0.125, 0.25]], [[0.5, -0.75]]]], dtype=np.float16)
    epsilon = np.float32(1.0e-5)

    x_f32 = x_np.astype(np.float32)
    dy_f32 = dy_np.astype(np.float32)
    scale_f32 = scale_np.astype(np.float32)
    bias_f32 = bias_np.astype(np.float32)
    mean_np = x_f32.mean(axis=(1, 2, 3), keepdims=True).astype(np.float32)
    var_np = ((x_f32 - mean_np) ** 2).mean(axis=(1, 2, 3), keepdims=True)
    inv_std_np = (1.0 / np.sqrt(var_np + epsilon)).astype(np.float32)
    xhat_np = (x_f32 - mean_np) * inv_std_np
    expected_y = (xhat_np * scale_f32 + bias_f32).astype(np.float16)
    dy_scale_np = dy_f32 * scale_f32
    reduced_count = np.float32(np.prod(x_np.shape[1:]))
    expected_dx = (
        inv_std_np
        * (
            dy_scale_np
            - dy_scale_np.sum(axis=(1, 2, 3), keepdims=True) / reduced_count
            - xhat_np
            * (dy_scale_np * xhat_np).sum(axis=(1, 2, 3), keepdims=True)
            / reduced_count
        )
    ).astype(np.float16)
    expected_dscale = (dy_f32 * xhat_np).sum(axis=0, keepdims=True).astype(np.float16)
    expected_dbias = dy_f32.sum(axis=0, keepdims=True).astype(np.float16)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float16",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float16",
        )
        scale = device_memory.create_tensor(
            scale_np.__array_interface__["data"][0],
            scale_np.nbytes,
            scale_np.shape,
            "float16",
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float16",
        )
        out = device_memory.create_tensor(
            0, expected_y.nbytes, expected_y.shape, "float16"
        )
        mean = device_memory.create_tensor(
            0, mean_np.nbytes, mean_np.shape, "float32"
        )
        inv_std = device_memory.create_tensor(
            0, inv_std_np.nbytes, inv_std_np.shape, "float32"
        )
        dx = device_memory.create_tensor(
            0, expected_dx.nbytes, expected_dx.shape, "float16"
        )
        dscale = device_memory.create_tensor(
            0, expected_dscale.nbytes, expected_dscale.shape, "float16"
        )
        dbias = device_memory.create_tensor(
            0, expected_dbias.nbytes, expected_dbias.shape, "float16"
        )
        tensors = [x, dy, scale, bias, out, mean, inv_std, dx, dscale, dbias]

        graph = fe.pygraph(
            io_data_type=fe.DataType.HALF,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.HALF)
        scale_desc = graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.HALF
        )
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.HALF
        )
        y_desc, mean_desc, inv_std_desc = graph.layernorm(
            name="ln",
            norm_forward_phase=fe.norm_forward_phase.TRAINING,
            input=x_desc,
            scale=scale_desc,
            bias=bias_desc,
            epsilon=float(epsilon),
        )
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                scale_desc: device_memory.get_tensor_addr(scale)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
                mean_desc: device_memory.get_tensor_addr(mean)[0],
                inv_std_desc: device_memory.get_tensor_addr(inv_std)[0],
            }
        )
        actual_y = device_memory.copy_to_ndarray(out, expected_y.nbytes)
        actual_mean = device_memory.copy_to_ndarray(mean, mean_np.nbytes)
        actual_inv_std = device_memory.copy_to_ndarray(inv_std, inv_std_np.nbytes)
        np.testing.assert_allclose(actual_y, expected_y, rtol=1.0e-3, atol=1.0e-3)
        np.testing.assert_allclose(actual_mean, mean_np, rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(actual_inv_std, inv_std_np, rtol=1.0e-5, atol=1.0e-5)

        bwd_graph = fe.pygraph(
            io_data_type=fe.DataType.HALF,
            compute_data_type=fe.DataType.FLOAT,
        )
        bwd_x_desc = bwd_graph.tensor(
            name="x", dim=x_np.shape, data_type=fe.DataType.HALF
        )
        dy_desc = bwd_graph.tensor(
            name="dy", dim=dy_np.shape, data_type=fe.DataType.HALF
        )
        bwd_scale_desc = bwd_graph.tensor(
            name="scale", dim=scale_np.shape, data_type=fe.DataType.HALF
        )
        bwd_mean_desc = bwd_graph.tensor(
            name="mean", dim=mean_np.shape, data_type=fe.DataType.FLOAT
        )
        bwd_inv_std_desc = bwd_graph.tensor(
            name="inv_std", dim=inv_std_np.shape, data_type=fe.DataType.FLOAT
        )
        dx_desc, dscale_desc, dbias_desc = bwd_graph.layernorm_backward(
            name="ln_bwd",
            grad=dy_desc,
            input=bwd_x_desc,
            scale=bwd_scale_desc,
            mean=bwd_mean_desc,
            inv_variance=bwd_inv_std_desc,
            epsilon=float(epsilon),
        )
        bwd_graph.execute(
            {
                bwd_x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                bwd_scale_desc: device_memory.get_tensor_addr(scale)[0],
                bwd_mean_desc: device_memory.get_tensor_addr(mean)[0],
                bwd_inv_std_desc: device_memory.get_tensor_addr(inv_std)[0],
                dx_desc: device_memory.get_tensor_addr(dx)[0],
                dscale_desc: device_memory.get_tensor_addr(dscale)[0],
                dbias_desc: device_memory.get_tensor_addr(dbias)[0],
            }
        )
        actual_dx = device_memory.copy_to_ndarray(dx, expected_dx.nbytes)
        actual_dscale = device_memory.copy_to_ndarray(dscale, expected_dscale.nbytes)
        actual_dbias = device_memory.copy_to_ndarray(dbias, expected_dbias.nbytes)
        np.testing.assert_allclose(actual_dx, expected_dx, rtol=1.0e-3, atol=1.0e-3)
        np.testing.assert_allclose(
            actual_dscale, expected_dscale, rtol=1.0e-3, atol=1.0e-3
        )
        np.testing.assert_allclose(
            actual_dbias, expected_dbias, rtol=1.0e-3, atol=1.0e-3
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_concat_then_reshape():
    import cudnn_frontend as fe

    a_np = np.arange(4, dtype=np.float32).reshape(2, 2)
    b_np = np.arange(4, 10, dtype=np.float32).reshape(2, 3)
    expected = np.concatenate((a_np, b_np), axis=1).reshape(10)

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [a, b, out]

        graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        concat_desc = graph.concat((a_desc, b_desc), axis=1, name="concat")
        y_desc = graph.reshape(concat_desc, shape=expected.shape, name="reshape")
        graph.build()
        workspace_size = graph.get_workspace_size()
        workspace = None
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, (workspace_size,), "int8"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]
        graph.execute(
            {
                a_desc: device_memory.get_tensor_addr(a)[0],
                b_desc: device_memory.get_tensor_addr(b)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_avg_pool():
    import cudnn_frontend as fe

    x_np = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    expected = _avg_pool2d_nchw_include_padding(x_np, (2, 2), (0, 0), (2, 2))

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.avg_pool(x_desc, window=(2, 2), stride=(2, 2), name="avg")
        graph.build()
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_avg_pool_exclude_padding():
    import cudnn_frontend as fe

    x_np = np.array([[[[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]]]], dtype=np.float32)
    expected = _avg_pool2d_nchw_exclude_padding(x_np, (2, 2), (1, 1), (1, 1))

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.avg_pool(
            x_desc,
            window=(2, 2),
            padding=(1, 1),
            stride=(1, 1),
            include_padding=False,
            name="avg_exclude",
        )
        graph.build()
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_max_pool():
    import cudnn_frontend as fe

    x_np = np.array(
        [[[[1.0, -2.0, 3.0, 4.0], [5.0, 6.0, -7.0, 8.0], [9.0, -1.0, 2.0, 0.0]]]],
        dtype=np.float32,
    )
    expected = _max_pool2d_nchw(x_np, (2, 2), (0, 0), (1, 2))

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.max_pool(x_desc, window=(2, 2), stride=(1, 2), name="max")
        graph.build()
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_avg_pool_bwd():
    import cudnn_frontend as fe

    dy_np = np.array([[[[1.0, -2.0], [0.5, 3.0]]]], dtype=np.float32)
    expected = _avg_pool2d_bwd_nchw_include_padding(
        dy_np, (1, 1, 4, 4), (2, 2), (0, 0), (2, 2)
    )

    tensors = []
    try:
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [dy, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        dx_desc = graph.avg_pool_bwd(
            dy_desc,
            x_shape=expected.shape,
            window=(2, 2),
            stride=(2, 2),
            name="avg_bwd",
        )
        graph.build()
        graph.execute(
            {
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                dx_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_max_pool_bwd():
    import cudnn_frontend as fe

    x_np = np.array(
        [[[[1.0, -2.0, 3.0, 4.0], [5.0, 6.0, -7.0, 8.0], [9.0, -1.0, 2.0, 0.0]]]],
        dtype=np.float32,
    )
    dy_np = np.array([[[[1.0, -2.0], [0.5, 3.0]]]], dtype=np.float32)
    expected = _max_pool2d_bwd_nchw(x_np, dy_np, (2, 2), (0, 0), (1, 2))

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, dy, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        dx_desc = graph.max_pool_bwd(
            x_desc,
            dy_desc,
            window=(2, 2),
            stride=(1, 2),
            name="max_bwd",
        )
        graph.build()
        workspace_size = graph.get_workspace_size()
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, ((workspace_size + 3) // 4,), "float32"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                dx_desc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_conv_fprop():
    import cudnn_frontend as fe

    x_np = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [2.5, 1.5, -2.0]]]],
        dtype=np.float32,
    )
    w_np = np.array([[[[0.5, -1.0], [2.0, 0.25]]]], dtype=np.float32)
    expected = _conv2d_nchw(x_np, w_np)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        w = device_memory.create_tensor(
            w_np.__array_interface__["data"][0],
            w_np.nbytes,
            w_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, w, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        w_desc = graph.tensor(name="w", dim=w_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.conv_fprop(x_desc, w_desc, name="conv")
        graph.build()

        workspace_ptr = None
        workspace_size = graph.get_workspace_size()
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, ((workspace_size + 3) // 4,), "float32"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]

        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                w_desc: device_memory.get_tensor_addr(w)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_conv_fprop_asymmetric_padding():
    import cudnn_frontend as fe

    x_np = np.array(
        [[[[1.0, 2.0, -1.0], [0.5, -0.25, 3.0], [2.5, 1.5, -2.0]]]],
        dtype=np.float32,
    )
    w_np = np.array([[[[0.5, -1.0], [2.0, 0.25]]]], dtype=np.float32)
    padded = np.pad(x_np, ((0, 0), (0, 0), (1, 0), (0, 1)))
    expected = _conv2d_nchw(padded, w_np)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        w = device_memory.create_tensor(
            w_np.__array_interface__["data"][0],
            w_np.nbytes,
            w_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, w, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        w_desc = graph.tensor(name="w", dim=w_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.conv_fprop(
            x_desc,
            w_desc,
            pre_padding=(1, 0),
            post_padding=(0, 1),
            name="conv",
        )
        assert y_desc.dims == expected.shape
        graph.build()

        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                w_desc: device_memory.get_tensor_addr(w)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_conv_fprop_half_host_fallback():
    import cudnn_frontend as fe

    x_np = np.array(
        [[[[1.0, -0.5, 2.0], [0.25, 1.5, -1.0], [2.5, -2.0, 0.75]]]],
        dtype=np.float16,
    )
    w_np = np.array([[[[0.5, -1.0], [1.25, 0.25]]]], dtype=np.float16)
    expected = _conv2d_nchw(x_np.astype(np.float32), w_np.astype(np.float32)).astype(
        np.float16
    )

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0], x_np.nbytes, x_np.shape, "float16"
        )
        w = device_memory.create_tensor(
            w_np.__array_interface__["data"][0], w_np.nbytes, w_np.shape, "float16"
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float16"
        )
        tensors = [x, w, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.HALF,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.HALF)
        w_desc = graph.tensor(name="w", dim=w_np.shape, data_type=fe.DataType.HALF)
        y_desc = graph.conv_fprop(x_desc, w_desc, name="conv")
        graph.build()

        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                w_desc: device_memory.get_tensor_addr(w)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=1.0e-3)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_clipped_relu_half_cpu_fallback():
    import cudnn_frontend as fe

    x_np = np.array([[-2.0, 0.25, 0.75], [1.0, -0.5, 0.5]], dtype=np.float16)
    expected = np.clip(x_np, 0.5, 0.75).astype(np.float16)
    out_np = np.zeros_like(x_np)

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.HALF)
    y_desc = graph.relu(x_desc, lower_clip=0.5, upper_clip=0.75, name="relu")
    y_desc.set_output(True)
    graph.build()

    graph.execute({x_desc: x_np, y_desc: out_np})

    np.testing.assert_allclose(out_np, expected, rtol=1.0e-3, atol=1.0e-3)


def test_mudnn_frontend_python_graph_api_execute_conv_clipped_relu_torch_cpu_fallback():
    import cudnn_frontend as fe
    import torch

    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 6, dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    w = torch.randn(4, 3, 3, 2, dtype=torch.float16)
    expected = torch.clamp(
        torch.nn.functional.conv2d(
            x, w, padding=(1, 0), stride=(2, 2), dilation=(1, 1)
        ),
        min=0.5,
        max=0.75,
    )
    out = torch.zeros_like(expected)

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    w_desc = graph.tensor_like(w)
    conv = graph.conv_fprop(
        image=x_desc,
        weight=w_desc,
        padding=[1, 0],
        stride=[2, 2],
        dilation=[1, 1],
    )
    y_desc = graph.relu(conv, lower_clip=0.5, upper_clip=0.75, name="relu")
    y_desc.set_output(True)
    graph.build()

    graph.execute({x_desc: x, w_desc: w, y_desc: out})

    torch.testing.assert_close(out, expected, rtol=1.0e-3, atol=1.0e-3)


def test_mudnn_frontend_python_graph_api_execute_conv3d_bias_leaky_relu_torch_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    x = torch.randn(1, 4, 5, 6, 3, dtype=torch.float16).permute(0, 4, 1, 2, 3)
    w = torch.randn(2, 2, 3, 2, 3, dtype=torch.float16).permute(0, 4, 1, 2, 3)
    bias = torch.randn(1, 1, 1, 1, 2, dtype=torch.float16).permute(0, 4, 1, 2, 3)
    padding = (0, 1, 2)
    stride = (1, 2, 1)
    dilation = (1, 1, 1)
    negative_slope = 0.01
    expected = torch.nn.functional.leaky_relu(
        torch.nn.functional.conv3d(
            x,
            w,
            bias=bias.reshape(-1),
            padding=padding,
            stride=stride,
            dilation=dilation,
        ),
        negative_slope=negative_slope,
    )
    out = torch.zeros_like(expected)

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    w_desc = graph.tensor_like(w)
    bias_desc = graph.tensor_like(bias)
    conv = graph.conv_fprop(
        image=x_desc,
        weight=w_desc,
        padding=padding,
        stride=stride,
        dilation=dilation,
    )
    biased = graph.bias(name="bias", input=conv, bias=bias_desc)
    y_desc = graph.leaky_relu(
        name="leaky_relu", input=biased, negative_slope=negative_slope
    )
    y_desc.set_output(True)
    graph.build()

    graph.execute({x_desc: x, w_desc: w, bias_desc: bias, y_desc: out})

    torch.testing.assert_close(out, expected, rtol=1.0e-2, atol=1.0e-2)


def test_mudnn_frontend_python_graph_api_execute_int8_conv_identity_torch_cpu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    x = torch.randint(-8, 8, (1, 8, 5, 5), dtype=torch.int8).to(
        memory_format=torch.channels_last
    )
    w = torch.randint(-4, 4, (3, 8, 3, 3), dtype=torch.int8).to(
        memory_format=torch.channels_last
    )
    expected = torch.nn.functional.conv2d(x, w, padding=(1, 1)).to(dtype=torch.int32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.INT8,
        intermediate_data_type=fe.DataType.INT32,
        compute_data_type=fe.DataType.INT32,
    )
    x_desc = graph.tensor_like(x)
    w_desc = graph.tensor_like(w)
    conv = graph.conv_fprop(
        image=x_desc,
        weight=w_desc,
        padding=(1, 1),
        stride=(1, 1),
        dilation=(1, 1),
    )
    y_desc = graph.identity(name="identity", input=conv)
    y_desc.set_output(True).set_data_type(fe.DataType.INT32)
    graph.build()

    actual = torch.empty_like(expected)
    graph.execute({x_desc: x, w_desc: w, y_desc: actual})

    torch.testing.assert_close(actual, expected)


def test_mudnn_frontend_python_graph_api_torch_cpu_layernorm_backward_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    epsilon = 1.0e-3
    x = (
        3
        * torch.randn(8, 16, 1, 1, dtype=torch.bfloat16, requires_grad=True).to(
            memory_format=torch.channels_last
        )
        - 0.5
    )
    scale = (
        5
        * torch.randn(1, 16, 1, 1, dtype=torch.bfloat16, requires_grad=True).to(
            memory_format=torch.channels_last
        )
        - 1.0
    )
    bias = (
        7
        * torch.randn(1, 16, 1, 1, dtype=torch.bfloat16, requires_grad=True).to(
            memory_format=torch.channels_last
        )
        - 2.0
    )
    expected_y = torch.nn.functional.layer_norm(
        x,
        [16, 1, 1],
        weight=scale.squeeze(0),
        bias=bias.squeeze(0),
        eps=epsilon,
    )
    target = torch.randn_like(expected_y)
    loss = torch.square(expected_y.float() - target.float()).mean()
    expected_y.retain_grad()
    x.retain_grad()
    scale.retain_grad()
    bias.retain_grad()
    loss.backward()

    graph = fe.pygraph(
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    scale_desc = graph.tensor_like(scale)
    bias_desc = graph.tensor_like(bias)
    epsilon_desc = graph.tensor(
        name="epsilon",
        dim=(1, 1, 1, 1),
        is_pass_by_value=True,
        data_type=fe.DataType.FLOAT,
    )
    y_desc, mean_desc, inv_var_desc = graph.layernorm(
        name="ln",
        norm_forward_phase=fe.norm_forward_phase.TRAINING,
        input=x_desc,
        scale=scale_desc,
        bias=bias_desc,
        epsilon=epsilon_desc,
    )
    y_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    mean_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    inv_var_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    graph.build()

    y_actual = torch.empty_like(expected_y)
    mean_actual = torch.empty((8, 1, 1, 1), dtype=torch.float32)
    inv_var_actual = torch.empty_like(mean_actual)
    graph.execute(
        {
            x_desc: x.detach(),
            scale_desc: scale.detach(),
            bias_desc: bias.detach(),
            epsilon_desc: torch.tensor([[[[epsilon]]]], dtype=torch.float32),
            y_desc: y_actual,
            mean_desc: mean_actual,
            inv_var_desc: inv_var_actual,
        }
    )

    expected_mean = x.float().mean(dim=(1, 2, 3), keepdim=True)
    expected_inv_var = torch.rsqrt(
        torch.square(x.float() - expected_mean).mean(dim=(1, 2, 3), keepdim=True)
        + epsilon
    )
    torch.testing.assert_close(y_actual, expected_y, rtol=0.0, atol=0.0)
    torch.testing.assert_close(mean_actual, expected_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(inv_var_actual, expected_inv_var, rtol=0.0, atol=0.0)

    bwd_graph = fe.pygraph(
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    bwd_x_desc = bwd_graph.tensor_like(x, name="x")
    dy_desc = bwd_graph.tensor_like(expected_y.grad, name="dy")
    bwd_scale_desc = bwd_graph.tensor_like(scale, name="scale")
    mean_bwd_desc = bwd_graph.tensor_like(mean_desc, name="mean")
    inv_var_bwd_desc = bwd_graph.tensor_like(inv_var_desc, name="inv_var")
    dx_desc, dscale_desc, dbias_desc = bwd_graph.layernorm_backward(
        name="ln_bwd",
        grad=dy_desc,
        input=bwd_x_desc,
        scale=bwd_scale_desc,
        mean=mean_bwd_desc,
        inv_variance=inv_var_bwd_desc,
        epsilon=epsilon,
    )
    dx_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    dscale_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    dbias_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    bwd_graph.build()

    dx_actual = torch.empty_like(x)
    dscale_actual = torch.empty_like(scale)
    dbias_actual = torch.empty_like(bias)
    bwd_graph.execute(
        {
            bwd_x_desc: x.detach(),
            dy_desc: expected_y.grad.detach(),
            bwd_scale_desc: scale.detach(),
            mean_bwd_desc: mean_actual.detach(),
            inv_var_bwd_desc: inv_var_actual.detach(),
            dx_desc: dx_actual,
            dscale_desc: dscale_actual,
            dbias_desc: dbias_actual,
        }
    )

    torch.testing.assert_close(dx_actual, x.grad, rtol=2.0e-4, atol=2.0e-4)
    torch.testing.assert_close(dscale_actual, scale.grad, rtol=2.0e-4, atol=2.0e-4)
    torch.testing.assert_close(dbias_actual, bias.grad, rtol=2.0e-4, atol=2.0e-4)


def test_mudnn_frontend_python_graph_api_torch_cpu_rmsnorm_backward_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    epsilon = 1.0e-3
    x = 2 * torch.randn(8, 16, 1, 1, dtype=torch.bfloat16, requires_grad=True) - 1.25
    scale = (
        3 * torch.randn(1, 16, 1, 1, dtype=torch.bfloat16, requires_grad=True) - 2.75
    )
    bias = torch.randn(1, 16, 1, 1, dtype=torch.bfloat16, requires_grad=True)

    norm_x = torch.mean(x * x, dim=(1, 2, 3), keepdim=True)
    expected_inv_var = torch.rsqrt(norm_x.float() + epsilon)
    expected_y = scale * (x * expected_inv_var.to(dtype=x.dtype)) + bias
    target = torch.randn_like(expected_y)
    loss = torch.square(expected_y.float() - target.float()).mean()
    expected_y.retain_grad()
    x.retain_grad()
    scale.retain_grad()
    bias.retain_grad()
    loss.backward()

    graph = fe.pygraph(
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    scale_desc = graph.tensor_like(scale)
    bias_desc = graph.tensor_like(bias)
    epsilon_desc = graph.tensor(
        name="epsilon",
        dim=(1, 1, 1, 1),
        is_pass_by_value=True,
        data_type=fe.DataType.FLOAT,
    )
    y_desc, inv_var_desc = graph.rmsnorm(
        name="rms",
        norm_forward_phase=fe.norm_forward_phase.TRAINING,
        input=x_desc,
        scale=scale_desc,
        bias=bias_desc,
        epsilon=epsilon_desc,
    )
    y_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    inv_var_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    graph.build()

    y_actual = torch.empty_like(expected_y)
    inv_var_actual = torch.empty((8, 1, 1, 1), dtype=torch.float32)
    graph.execute(
        {
            x_desc: x.detach(),
            scale_desc: scale.detach(),
            bias_desc: bias.detach(),
            epsilon_desc: torch.tensor([[[[epsilon]]]], dtype=torch.float32),
            y_desc: y_actual,
            inv_var_desc: inv_var_actual,
        }
    )

    torch.testing.assert_close(y_actual, expected_y, rtol=0.0, atol=0.0)
    torch.testing.assert_close(inv_var_actual, expected_inv_var, rtol=0.0, atol=0.0)

    bwd_graph = fe.pygraph(
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    bwd_x_desc = bwd_graph.tensor_like(x, name="x")
    dy_desc = bwd_graph.tensor_like(expected_y.grad, name="dy")
    bwd_scale_desc = bwd_graph.tensor_like(scale, name="scale")
    inv_var_bwd_desc = bwd_graph.tensor_like(inv_var_desc, name="inv_var")
    dx_desc, dscale_desc, dbias_desc = bwd_graph.rmsnorm_backward(
        name="rms_bwd",
        grad=dy_desc,
        input=bwd_x_desc,
        scale=bwd_scale_desc,
        inv_variance=inv_var_bwd_desc,
        has_dbias=True,
    )
    dx_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    dscale_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    dbias_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    bwd_graph.build()

    dx_actual = torch.empty_like(x)
    dscale_actual = torch.empty_like(scale)
    dbias_actual = torch.empty_like(bias)
    bwd_graph.execute(
        {
            bwd_x_desc: x.detach(),
            dy_desc: expected_y.grad.detach(),
            bwd_scale_desc: scale.detach(),
            inv_var_bwd_desc: inv_var_actual.detach(),
            dx_desc: dx_actual,
            dscale_desc: dscale_actual,
            dbias_desc: dbias_actual,
        }
    )

    torch.testing.assert_close(dx_actual, x.grad, rtol=2.0e-4, atol=2.0e-4)
    torch.testing.assert_close(dscale_actual, scale.grad, rtol=5.0e-4, atol=5.0e-4)
    torch.testing.assert_close(dbias_actual, bias.grad, rtol=5.0e-4, atol=5.0e-4)


def test_mudnn_frontend_python_graph_api_torch_cpu_batchnorm_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    epsilon = 1.0e-3
    momentum = 0.1
    x = torch.randn(2, 3, 4, 5, dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    scale = torch.randn(1, 3, 1, 1, dtype=torch.float32)
    bias = torch.randn(1, 3, 1, 1, dtype=torch.float32)
    running_mean = torch.randn(1, 3, 1, 1, dtype=torch.float32)
    running_var = torch.rand(1, 3, 1, 1, dtype=torch.float32) + 0.5

    x_float = x.float()
    expected_running_mean = running_mean.reshape(-1).clone()
    expected_running_var = running_var.reshape(-1).clone()
    expected_y = torch.nn.functional.batch_norm(
        x_float,
        expected_running_mean,
        expected_running_var,
        weight=scale.reshape(-1),
        bias=bias.reshape(-1),
        training=True,
        momentum=momentum,
        eps=epsilon,
    ).to(dtype=torch.float16)
    expected_mean = x_float.mean(dim=(0, 2, 3), keepdim=True)
    expected_inv_var = torch.rsqrt(
        torch.var(x_float, dim=(0, 2, 3), keepdim=True) + epsilon
    )

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    scale_desc = graph.tensor_like(scale)
    bias_desc = graph.tensor_like(bias)
    running_mean_desc = graph.tensor_like(running_mean)
    running_var_desc = graph.tensor_like(running_var)
    (
        y_desc,
        saved_mean_desc,
        saved_inv_desc,
        out_mean_desc,
        out_var_desc,
    ) = graph.batchnorm(
        input=x_desc,
        scale=scale_desc,
        bias=bias_desc,
        in_running_mean=running_mean_desc,
        in_running_var=running_var_desc,
        epsilon=epsilon,
        momentum=momentum,
    )
    y_desc.set_output(True).set_data_type(fe.DataType.HALF)
    saved_mean_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    saved_inv_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    out_mean_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    out_var_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    graph.build()

    y_actual = torch.empty_like(x)
    saved_mean_actual = torch.empty_like(running_mean)
    saved_inv_actual = torch.empty_like(running_var)
    out_mean_actual = torch.empty_like(running_mean)
    out_var_actual = torch.empty_like(running_var)
    graph.execute(
        {
            x_desc: x,
            scale_desc: scale,
            bias_desc: bias,
            running_mean_desc: running_mean,
            running_var_desc: running_var,
            y_desc: y_actual,
            saved_mean_desc: saved_mean_actual,
            saved_inv_desc: saved_inv_actual,
            out_mean_desc: out_mean_actual,
            out_var_desc: out_var_actual,
        }
    )

    torch.testing.assert_close(y_actual, expected_y, rtol=1.0e-3, atol=1.0e-3)
    torch.testing.assert_close(saved_mean_actual, expected_mean, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        saved_inv_actual, expected_inv_var, rtol=1.0e-6, atol=1.0e-6
    )
    torch.testing.assert_close(
        out_mean_actual.reshape(-1), expected_running_mean, rtol=1.0e-6, atol=1.0e-6
    )
    torch.testing.assert_close(
        out_var_actual.reshape(-1), expected_running_var, rtol=1.0e-6, atol=1.0e-6
    )

    dy = torch.randn_like(x)
    inv = saved_inv_actual.float()
    xhat = (x_float - expected_mean) * inv
    dy_scale = dy.float() * scale
    reduce_axes = (0, 2, 3)
    reduce_count = x.shape[0] * x.shape[2] * x.shape[3]
    expected_dx = inv * (
        dy_scale
        - dy_scale.sum(dim=reduce_axes, keepdim=True) / reduce_count
        - xhat * (dy_scale * xhat).sum(dim=reduce_axes, keepdim=True) / reduce_count
    )
    expected_dscale = (dy.float() * xhat).sum(dim=reduce_axes, keepdim=True)
    expected_dbias = dy.float().sum(dim=reduce_axes, keepdim=True)

    bwd_graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    bwd_x_desc = bwd_graph.tensor_like(x, name="x")
    dy_desc = bwd_graph.tensor_like(dy, name="dy")
    bwd_scale_desc = bwd_graph.tensor_like(scale, name="scale")
    mean_desc = bwd_graph.tensor_like(saved_mean_actual, name="mean")
    inv_desc = bwd_graph.tensor_like(saved_inv_actual, name="inv")
    dx_desc, dscale_desc, dbias_desc = bwd_graph.batchnorm_backward(
        grad=dy_desc,
        input=bwd_x_desc,
        scale=bwd_scale_desc,
        mean=mean_desc,
        inv_variance=inv_desc,
    )
    dx_desc.set_output(True).set_data_type(fe.DataType.HALF)
    dscale_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    dbias_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    bwd_graph.build()

    dx_actual = torch.empty_like(x)
    dscale_actual = torch.empty_like(scale)
    dbias_actual = torch.empty_like(scale)
    bwd_graph.execute(
        {
            bwd_x_desc: x,
            dy_desc: dy,
            bwd_scale_desc: scale,
            mean_desc: saved_mean_actual,
            inv_desc: saved_inv_actual,
            dx_desc: dx_actual,
            dscale_desc: dscale_actual,
            dbias_desc: dbias_actual,
        }
    )

    torch.testing.assert_close(
        dx_actual, expected_dx.to(dtype=torch.float16), rtol=1.0e-3, atol=1.0e-3
    )
    torch.testing.assert_close(dscale_actual, expected_dscale, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(dbias_actual, expected_dbias, rtol=1.0e-5, atol=1.0e-5)


def test_mudnn_frontend_python_graph_api_execute_leaky_relu_backward_half_cpu_fallback():
    import cudnn_frontend as fe

    x_np = np.array([[-2.0, 0.0, 0.75], [1.0, -0.5, 0.5]], dtype=np.float16)
    dy_np = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float16)
    negative_slope = 0.125
    expected = np.where(x_np > 0.0, dy_np, dy_np * negative_slope).astype(np.float16)
    out_np = np.zeros_like(x_np)

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.HALF)
    dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.HALF)
    dx_desc = graph.leaky_relu_backward(dy_desc, x_desc, negative_slope)
    dx_desc.set_output(True)
    graph.build()

    graph.execute({x_desc: x_np, dy_desc: dy_np, dx_desc: out_np})

    np.testing.assert_allclose(out_np, expected, rtol=1.0e-3, atol=1.0e-3)


def test_mudnn_frontend_python_graph_api_execute_reduction_half_to_float_cpu_fallback():
    import cudnn_frontend as fe

    x_np = np.arange(2 * 3 * 2, dtype=np.float16).reshape(2, 3, 2) / 10.0
    expected = x_np.astype(np.float32).sum(axis=1, keepdims=True)
    out_np = np.zeros_like(expected, dtype=np.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.HALF)
    y_desc = graph.reduction(x_desc, mode=fe.reduction_mode.ADD, axis=1)
    y_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    graph.build()

    graph.execute({x_desc: x_np, y_desc: out_np})

    np.testing.assert_allclose(out_np, expected, rtol=1.0e-6, atol=1.0e-6)


def test_mudnn_frontend_python_graph_api_execute_torch_cpu_conv_reduction_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    x = torch.arange(1 * 2 * 4 * 4, dtype=torch.float16).reshape(1, 2, 4, 4) / 10.0
    w = torch.arange(3 * 2 * 3 * 3, dtype=torch.float16).reshape(3, 2, 3, 3) / 20.0
    expected = torch.nn.functional.conv2d(x, w, padding=(1, 1)).sum(
        dim=1, keepdim=False
    )
    out = torch.zeros_like(expected, dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    w_desc = graph.tensor_like(w)
    conv = graph.conv_fprop(x_desc, w_desc, padding=[1, 1])
    y_desc = graph.reduction(conv, mode=fe.reduction_mode.ADD)
    y_desc.set_output(True).set_dim([1, 1, 4, 4]).set_data_type(fe.DataType.FLOAT)
    graph.build()

    graph.execute({x_desc: x, w_desc: w, y_desc: out})

    torch.testing.assert_close(expected.to(torch.float32), out)


def test_mudnn_frontend_python_graph_api_fp8_pointwise_build_unsupported_check():
    import cudnn_frontend as fe

    graph = fe.pygraph(
        io_data_type=fe.DataType.FP8_E4M3,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x = graph.tensor(name="x", dim=(1, 4, 4), data_type=fe.DataType.FP8_E4M3)
    scale = graph.tensor_scalar(0.5, data_type=fe.DataType.FLOAT, name="scale")
    y = graph.mul(graph.sigmoid(graph.mul(x, scale)), x)
    y.set_output(True).set_data_type(fe.DataType.FP8_E4M3)

    graph.build_operation_graph()
    graph.create_execution_plans([fe.heur_mode.A, fe.heur_mode.FALLBACK])
    with pytest.raises(fe.cudnnGraphNotSupportedError, match="FP8_E4M3"):
        graph.check_support()


def test_mudnn_frontend_python_graph_api_torch_cpu_matmul_bias_bfloat16_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    x = torch.randn(1, 5, 7, dtype=torch.bfloat16)
    w = torch.randn(1, 7, 11, dtype=torch.bfloat16)
    bias = torch.randn(1, 1, 11, dtype=torch.bfloat16)
    expected = (torch.matmul(x.float(), w.float()) + bias.float()).to(torch.bfloat16)
    out = torch.zeros_like(expected)

    graph = fe.pygraph(
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    w_desc = graph.tensor_like(w)
    bias_desc = graph.tensor_like(bias)
    y_desc = graph.bias(graph.matmul(x_desc, w_desc), bias_desc)
    y_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    graph.build()

    graph.execute({x_desc: x, w_desc: w, bias_desc: bias, y_desc: out})

    torch.testing.assert_close(expected, out, atol=0.0, rtol=0.0)


def test_mudnn_frontend_python_graph_api_torch_cpu_int8_bfloat16_matmul_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    a = torch.randint(-2, 2, (2, 3, 5), dtype=torch.int8)
    b = torch.randn(2, 5, 4, dtype=torch.bfloat16)
    expected = torch.matmul(a.to(torch.bfloat16), b).to(torch.bfloat16)
    out = torch.zeros_like(expected)

    graph = fe.pygraph(
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    a_desc = graph.tensor_like(a)
    b_desc = graph.tensor_like(b)
    a_casted = graph.identity(input=a_desc, compute_data_type=fe.DataType.FLOAT)
    a_casted.set_data_type(fe.DataType.BFLOAT16)
    c_desc = graph.matmul(
        name="matmul",
        A=a_casted,
        B=b_desc,
        compute_data_type=fe.DataType.FLOAT,
    )
    c_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    graph.build()

    graph.execute({a_desc: a, b_desc: b, c_desc: out})

    torch.testing.assert_close(expected, out, rtol=1.0e-2, atol=1.0e-1)

    half_a = torch.randn(2, 3, 5, dtype=torch.float16)
    half_expected = torch.matmul(half_a.to(torch.bfloat16), b).to(torch.bfloat16)
    half_out = torch.zeros_like(half_expected)

    half_graph = fe.pygraph(
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    half_a_desc = half_graph.tensor_like(half_a)
    half_b_desc = half_graph.tensor_like(b)
    half_a_casted = half_graph.identity(
        input=half_a_desc, compute_data_type=fe.DataType.FLOAT
    )
    half_a_casted.set_data_type(fe.DataType.BFLOAT16)
    half_c_desc = half_graph.matmul(
        name="matmul",
        A=half_a_casted,
        B=half_b_desc,
        compute_data_type=fe.DataType.FLOAT,
    )
    half_c_desc.set_output(True).set_data_type(fe.DataType.BFLOAT16)
    half_graph.build()

    half_graph.execute({half_a_desc: half_a, half_b_desc: b, half_c_desc: half_out})

    torch.testing.assert_close(half_expected, half_out, rtol=1.0e-2, atol=1.0e-1)

    bare_a = torch.randn(2, 3, 5, dtype=torch.float16)
    bare_b = torch.randn(2, 5, 4, dtype=torch.float16)
    bare_expected = torch.matmul(bare_a, bare_b).to(torch.float16)
    bare_out = torch.zeros_like(bare_expected)

    bare_graph = fe.pygraph(
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    bare_a_desc = bare_graph.tensor_like(bare_a)
    bare_b_desc = bare_graph.tensor_like(bare_b)
    bare_c_desc = bare_graph.matmul(
        name="matmul",
        A=bare_a_desc,
        B=bare_b_desc,
        compute_data_type=fe.DataType.FLOAT,
    )
    bare_c_desc.set_output(True).set_data_type(fe.DataType.HALF)
    bare_graph.build()

    bare_graph.execute(
        {bare_a_desc: bare_a, bare_b_desc: bare_b, bare_c_desc: bare_out}
    )

    torch.testing.assert_close(bare_expected, bare_out, rtol=1.0e-3, atol=1.0e-3)


def test_mudnn_frontend_python_graph_api_torch_cpu_matmul_bias_gelu_fallback():
    import cudnn_frontend as fe

    torch = pytest.importorskip("torch")

    x = torch.randn(3, 5, 7, dtype=torch.float32)
    w = torch.randn(3, 7, 11, dtype=torch.float32)
    bias = torch.randn(3, 1, 1, dtype=torch.float32)
    expected = torch.nn.functional.gelu(torch.matmul(x, w) + bias)
    out = torch.zeros_like(expected)

    graph = fe.pygraph(
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    w_desc = graph.tensor_like(w)
    bias_desc = graph.tensor_like(bias)
    y_desc = graph.gelu(graph.bias(graph.matmul(x_desc, w_desc), bias_desc))
    y_desc.set_output(True).set_data_type(fe.DataType.FLOAT)
    graph.build()

    graph.execute({x_desc: x, w_desc: w, bias_desc: bias, y_desc: out})

    torch.testing.assert_close(expected, out, atol=0.0, rtol=0.0)


def test_mudnn_frontend_python_graph_api_execute_grouped_conv_fprop():
    import cudnn_frontend as fe

    x_np = np.arange(1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4) / 10.0
    w_np = np.arange(4 * 2 * 3 * 3, dtype=np.float32).reshape(4, 2, 3, 3) / 20.0
    expected = _conv2d_nchw(x_np, w_np, groups=2)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        w = device_memory.create_tensor(
            w_np.__array_interface__["data"][0],
            w_np.nbytes,
            w_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, w, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        w_desc = graph.tensor(name="w", dim=w_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.conv_fprop(x_desc, w_desc, name="grouped_conv")
        graph.build()
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                w_desc: device_memory.get_tensor_addr(w)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_conv3d_fprop():
    import cudnn_frontend as fe

    x_np = np.arange(27, dtype=np.float32).reshape(1, 1, 3, 3, 3)
    w_np = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    expected = _conv3d_ncdhw(x_np, w_np)

    tensors = []
    try:
        x_flat = np.ascontiguousarray(x_np.reshape(-1))
        w_flat = np.ascontiguousarray(w_np.reshape(-1))
        x = device_memory.create_tensor(
            x_flat.__array_interface__["data"][0],
            x_flat.nbytes,
            x_flat.shape,
            "float32",
        )
        w = device_memory.create_tensor(
            w_flat.__array_interface__["data"][0],
            w_flat.nbytes,
            w_flat.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, (expected.size,), "float32"
        )
        tensors = [x, w, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        w_desc = graph.tensor(name="w", dim=w_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.conv_fprop(
            x_desc,
            w_desc,
            padding=(0, 0, 0),
            stride=(1, 1, 1),
            dilation=(1, 1, 1),
            name="conv3d",
        )
        graph.build()
        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                w_desc: device_memory.get_tensor_addr(w)[0],
                y_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes).reshape(
            expected.shape
        )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_conv_bwd_data():
    import cudnn_frontend as fe

    dy_np = np.array([[[[1.0, -2.0], [0.5, 3.0]]]], dtype=np.float32)
    w_np = np.array([[[[0.5, -1.0], [2.0, 0.25]]]], dtype=np.float32)
    expected = _conv2d_bwd_data_nchw(dy_np, w_np, (1, 1, 3, 3))

    tensors = []
    try:
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        w = device_memory.create_tensor(
            w_np.__array_interface__["data"][0],
            w_np.nbytes,
            w_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [dy, w, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        w_desc = graph.tensor(name="w", dim=w_np.shape, data_type=fe.DataType.FLOAT)
        dx_desc = graph.conv_bwd_data(
            dy_desc,
            w_desc,
            x_shape=expected.shape,
            name="dgrad",
        )
        graph.build()

        graph.execute(
            {
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                w_desc: device_memory.get_tensor_addr(w)[0],
                dx_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_conv_bwd_filter():
    import cudnn_frontend as fe

    x_np = np.array(
        [[[[1.0, 2.0, 3.0], [4.0, -1.0, 0.5], [2.5, 1.5, -2.0]]]],
        dtype=np.float32,
    )
    dy_np = np.array([[[[1.0, -2.0], [0.5, 3.0]]]], dtype=np.float32)
    expected = _conv2d_bwd_filter_nchw(x_np, dy_np, (1, 1, 2, 2))

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, dy, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        dw_desc = graph.conv_bwd_filter(
            x_desc,
            dy_desc,
            w_shape=expected.shape,
            name="wgrad",
        )
        graph.build()

        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                dw_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_conv_wgrad_infers_filter_shape():
    import cudnn_frontend as fe

    x_np = np.arange(1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4) / 10.0
    dy_np = np.arange(1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2) / 5.0
    expected = _conv2d_bwd_filter_nchw(x_np, dy_np, (4, 4, 3, 3))

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, dy, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        dw_desc = graph.conv_wgrad(
            name="wgrad",
            image=x_desc,
            loss=dy_desc,
            padding=(0, 0),
            stride=(1, 1),
            dilation=(1, 1),
        )
        assert dw_desc.dims == expected.shape
        graph.build()

        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                dw_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_conv_wgrad_set_dim_updates_groups():
    torch = pytest.importorskip("torch")
    import cudnn_frontend as fe

    x = torch.randn(1, 8, 16, 16, dtype=torch.float32)
    dy = torch.randn(1, 40, 14, 14, dtype=torch.float32)
    expected_shape = (40, 2, 3, 3)
    expected = torch.nn.grad.conv2d_weight(
        x,
        expected_shape,
        dy,
        padding=(0, 1),
        stride=(1, 1),
        dilation=(1, 2),
        groups=4,
    )
    actual = torch.empty_strided(expected_shape, expected.stride(), dtype=torch.float32)

    graph = fe.pygraph(
        io_data_type=fe.DataType.FLOAT,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor_like(x)
    dy_desc = graph.tensor_like(dy)
    dw_desc = graph.conv_wgrad(
        image=x_desc,
        loss=dy_desc,
        padding=(0, 1),
        stride=(1, 1),
        dilation=(1, 2),
    )
    assert graph.nodes[-1].attributes["group_count"] == 1
    dw_desc.set_output(True).set_dim(expected_shape).set_stride(expected.stride())
    assert graph.nodes[-1].attributes["group_count"] == 4

    graph.build()
    graph.execute({x_desc: x, dy_desc: dy, dw_desc: actual})
    torch.testing.assert_close(actual, expected)


def test_mudnn_frontend_python_graph_api_execute_grouped_conv_bwd_data():
    import cudnn_frontend as fe

    dy_np = np.arange(1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2) / 10.0
    w_np = np.arange(4 * 2 * 3 * 3, dtype=np.float32).reshape(4, 2, 3, 3) / 20.0
    expected = _conv2d_bwd_data_nchw(dy_np, w_np, (1, 4, 4, 4), groups=2)

    tensors = []
    try:
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        w = device_memory.create_tensor(
            w_np.__array_interface__["data"][0],
            w_np.nbytes,
            w_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [dy, w, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        w_desc = graph.tensor(name="w", dim=w_np.shape, data_type=fe.DataType.FLOAT)
        dx_desc = graph.conv_bwd_data(
            dy_desc,
            w_desc,
            x_shape=expected.shape,
            name="grouped_dgrad",
        )
        assert graph.nodes[-1].attributes["group_count"] == 2
        graph.build()

        graph.execute(
            {
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                w_desc: device_memory.get_tensor_addr(w)[0],
                dx_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_grouped_conv_bwd_data_half():
    import cudnn_frontend as fe

    dy_np = (
        np.arange(1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2) / 11.0
    ).astype(np.float16)
    w_np = (
        np.arange(4 * 2 * 3 * 3, dtype=np.float32).reshape(4, 2, 3, 3) / 17.0
    ).astype(np.float16)
    expected = _conv2d_bwd_data_nchw(
        dy_np.astype(np.float32),
        w_np.astype(np.float32),
        (1, 4, 4, 4),
        groups=2,
    ).astype(np.float16)

    tensors = []
    try:
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float16",
        )
        w = device_memory.create_tensor(
            w_np.__array_interface__["data"][0],
            w_np.nbytes,
            w_np.shape,
            "float16",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float16"
        )
        tensors = [dy, w, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.HALF,
            compute_data_type=fe.DataType.FLOAT,
        )
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.HALF)
        w_desc = graph.tensor(name="w", dim=w_np.shape, data_type=fe.DataType.HALF)
        dx_desc = graph.conv_bwd_data(
            dy_desc,
            w_desc,
            x_shape=expected.shape,
            name="grouped_dgrad_half",
        )
        assert graph.nodes[-1].attributes["group_count"] == 2
        graph.build()

        graph.execute(
            {
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                w_desc: device_memory.get_tensor_addr(w)[0],
                dx_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        # The host fallback reads HALF storage into double and rounds once at
        # output; the NumPy helper accumulates through float32 reference arrays.
        np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=2.0e-2)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_grouped_conv_bwd_filter():
    import cudnn_frontend as fe

    x_np = np.arange(1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4) / 10.0
    dy_np = np.arange(1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2) / 5.0
    expected = _conv2d_bwd_filter_nchw(x_np, dy_np, (4, 2, 3, 3), groups=2)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, dy, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT)
        dw_desc = graph.conv_bwd_filter(
            x_desc,
            dy_desc,
            w_shape=expected.shape,
            name="grouped_wgrad",
        )
        assert graph.nodes[-1].attributes["group_count"] == 2
        graph.build()

        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                dw_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_grouped_conv_bwd_filter_half():
    import cudnn_frontend as fe

    x_np = (
        np.arange(1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4) / 17.0
    ).astype(np.float16)
    dy_np = (
        np.arange(1 * 4 * 2 * 2, dtype=np.float32).reshape(1, 4, 2, 2) / 11.0
    ).astype(np.float16)
    expected = _conv2d_bwd_filter_nchw(
        x_np.astype(np.float32),
        dy_np.astype(np.float32),
        (4, 2, 3, 3),
        groups=2,
    ).astype(np.float16)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float16",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float16",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float16"
        )
        tensors = [x, dy, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.HALF,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.HALF)
        dy_desc = graph.tensor(name="dy", dim=dy_np.shape, data_type=fe.DataType.HALF)
        dw_desc = graph.conv_bwd_filter(
            x_desc,
            dy_desc,
            w_shape=expected.shape,
            name="grouped_wgrad_half",
        )
        assert graph.nodes[-1].attributes["group_count"] == 2
        graph.build()

        graph.execute(
            {
                x_desc: device_memory.get_tensor_addr(x)[0],
                dy_desc: device_memory.get_tensor_addr(dy)[0],
                dw_desc: device_memory.get_tensor_addr(out)[0],
            }
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        # The host fallback accumulates descriptor-read HALF values in double
        # and rounds once at output, while the NumPy helper accumulates through
        # float32 storage; allow the resulting sub-ULP/ULP-scale HALF drift.
        np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=2.0e-2)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_scale_bias_relu_wgrad_torch_cpu_fallback():
    torch = pytest.importorskip("torch")
    import cudnn_frontend as fe

    x = torch.randn(2, 8, 8, 8, dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    dy = torch.randn(2, 4, 8, 8, dtype=torch.float16).to(
        memory_format=torch.channels_last
    )
    scale = (torch.randn(1, 8, 1, 1, dtype=torch.float16) * 0.01).to(
        memory_format=torch.channels_last
    )
    bias = (torch.randn(1, 8, 1, 1, dtype=torch.float16) * 0.01).to(
        memory_format=torch.channels_last
    )
    expected = torch.nn.grad.conv2d_weight(
        torch.relu(x.float() * scale.float() + bias.float()),
        (4, 8, 3, 3),
        dy.float(),
        padding=(1, 1),
    ).to(dtype=torch.float16)

    graph = fe.pygraph(
        io_data_type=fe.DataType.HALF,
        intermediate_data_type=fe.DataType.FLOAT,
        compute_data_type=fe.DataType.FLOAT,
    )
    x_desc = graph.tensor(
        name="x", dim=x.shape, stride=x.stride(), data_type=fe.DataType.HALF
    )
    dy_desc = graph.tensor(
        name="dy", dim=dy.shape, stride=dy.stride(), data_type=fe.DataType.HALF
    )
    scale_desc = graph.tensor(
        name="scale",
        dim=scale.shape,
        stride=scale.stride(),
        data_type=fe.DataType.HALF,
    )
    bias_desc = graph.tensor(
        name="bias", dim=bias.shape, stride=bias.stride(), data_type=fe.DataType.HALF
    )

    scaled = graph.scale(name="scale_node", input=x_desc, scale=scale_desc)
    biased = graph.bias(name="bias_node", input=scaled, bias=bias_desc)
    activated = graph.relu(name="relu_node", input=biased)
    dw_desc = graph.conv_wgrad(
        name="wgrad",
        image=activated,
        loss=dy_desc,
        padding=(1, 1),
        stride=(1, 1),
        dilation=(1, 1),
    )
    dw_desc.set_output(True).set_dim(expected.shape)

    actual = torch.empty_like(expected)
    graph.execute(
        {
            x_desc: x,
            dy_desc: dy,
            scale_desc: scale,
            bias_desc: bias,
            dw_desc: actual,
        }
    )
    torch.testing.assert_close(actual, expected, rtol=1.0e-3, atol=2.0e-2)

    oversized = torch.zeros_like(x)
    graph.execute(
        {
            x_desc: x,
            dy_desc: dy,
            scale_desc: scale,
            bias_desc: bias,
            dw_desc: oversized,
        }
    )
    oversized_storage = oversized.as_strided(
        (oversized.untyped_storage().nbytes() // oversized.element_size(),), (1,)
    )
    torch.testing.assert_close(
        oversized_storage[: expected.numel()].reshape(expected.shape),
        expected,
        rtol=1.0e-3,
        atol=2.0e-2,
    )


def test_mudnn_frontend_python_graph_api_execute_div_sigmoid():
    import cudnn_frontend as fe
    import mudnn_frontend._backend as backend

    a_np = np.array([[1.0, 2.0, 3.0], [0.25, 4.0, 5.0]], dtype=np.float32)
    b_np = np.array([[2.0, 4.0, 1.5], [0.5, 8.0, 2.0]], dtype=np.float32)
    expected = 1.0 / (1.0 + np.exp(-np.floor(a_np / b_np)))

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [a, b, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        quotient = graph.div(a_desc, b_desc, rounding_mode="floor", name="quotient")
        sigmoid = graph.sigmoid(quotient, name="sigmoid")
        graph.validate().build_operation_graph().create_execution_plans(
            [fe.HeurMode.INSTANT]
        ).check_support().build_plans()

        (workspace_size,) = graph._native.plans[0].get_int64(
            backend.MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE
        )
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, ((workspace_size + 3) // 4,), "float32"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]

        graph.execute(
            {
                a_desc: device_memory.get_tensor_addr(a)[0],
                b_desc: device_memory.get_tensor_addr(b)[0],
                sigmoid: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_div_trunc():
    import cudnn_frontend as fe
    import mudnn_frontend._backend as backend

    a_np = np.array([[3.7, -3.7, 5.0], [-5.0, 0.25, -0.25]], dtype=np.float32)
    b_np = np.array([[2.0, 2.0, -2.0], [-2.0, 0.5, 0.5]], dtype=np.float32)
    expected = np.trunc(a_np / b_np).astype(np.float32)

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [a, b, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        trunc = graph.div(a_desc, b_desc, rounding_mode="trunc", name="trunc")
        graph.validate().build_operation_graph().create_execution_plans(
            [fe.HeurMode.INSTANT]
        ).check_support().build_plans()

        (workspace_size,) = graph._native.plans[0].get_int64(
            backend.MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE
        )
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, ((workspace_size + 3) // 4,), "float32"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]

        graph.execute(
            {
                a_desc: device_memory.get_tensor_addr(a)[0],
                b_desc: device_memory.get_tensor_addr(b)[0],
                trunc: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_scale():
    import cudnn_frontend as fe

    x_np = np.array([[1.0, -2.0, 3.0], [0.25, 4.0, -5.0]], dtype=np.float32)
    expected = x_np * np.float32(-1.5)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [x, out]
        x_ptr = device_memory.get_tensor_addr(x)[0]
        out_ptr = device_memory.get_tensor_addr(out)[0]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        x_desc = graph.tensor(name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT)
        y_desc = graph.scale(x_desc, -1.5, name="scaled")
        graph.execute({x_desc: x_ptr, y_desc: out_ptr})

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_swish_and_relu_backward():
    import cudnn_frontend as fe

    x_np = np.array([[-2.0, -0.5, 0.0], [0.25, 1.0, 3.0]], dtype=np.float32)
    dy_np = np.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=np.float32)
    beta = np.float32(1.25)
    swish_expected = x_np / (np.float32(1.0) + np.exp(-beta * x_np))
    relu_bwd_expected = np.where(x_np > 0.0, dy_np, 0.0).astype(np.float32)

    tensors = []
    try:
        x = device_memory.create_tensor(
            x_np.__array_interface__["data"][0],
            x_np.nbytes,
            x_np.shape,
            "float32",
        )
        dy = device_memory.create_tensor(
            dy_np.__array_interface__["data"][0],
            dy_np.nbytes,
            dy_np.shape,
            "float32",
        )
        swish_out = device_memory.create_tensor(
            0, swish_expected.nbytes, swish_expected.shape, "float32"
        )
        relu_bwd_out = device_memory.create_tensor(
            0, relu_bwd_expected.nbytes, relu_bwd_expected.shape, "float32"
        )
        tensors = [x, dy, swish_out, relu_bwd_out]

        swish_graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        swish_x = swish_graph.tensor(
            name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT
        )
        swish_y = swish_graph.swish(input=swish_x, swish_beta=float(beta))
        swish_graph.execute(
            {
                swish_x: device_memory.get_tensor_addr(x)[0],
                swish_y: device_memory.get_tensor_addr(swish_out)[0],
            }
        )

        relu_graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        relu_x = relu_graph.tensor(
            name="x", dim=x_np.shape, data_type=fe.DataType.FLOAT
        )
        relu_dy = relu_graph.tensor(
            name="dy", dim=dy_np.shape, data_type=fe.DataType.FLOAT
        )
        relu_dx = relu_graph.relu_backward(loss=relu_dy, input=relu_x)
        relu_graph.execute(
            {
                relu_x: device_memory.get_tensor_addr(x)[0],
                relu_dy: device_memory.get_tensor_addr(dy)[0],
                relu_dx: device_memory.get_tensor_addr(relu_bwd_out)[0],
            }
        )

        swish_actual = device_memory.copy_to_ndarray(swish_out, swish_expected.nbytes)
        relu_bwd_actual = device_memory.copy_to_ndarray(
            relu_bwd_out, relu_bwd_expected.nbytes
        )
        np.testing.assert_allclose(
            swish_actual, swish_expected, rtol=2.0e-5, atol=2.0e-5
        )
        np.testing.assert_allclose(
            relu_bwd_actual, relu_bwd_expected, rtol=1.0e-6, atol=1.0e-6
        )
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_compare_select():
    import cudnn_frontend as fe
    import mudnn_frontend._backend as backend

    a_np = np.array([[1.0, -2.0, 3.0], [0.25, 4.0, -5.0]], dtype=np.float32)
    b_np = np.array([[0.5, -3.0, 4.0], [0.5, 2.0, -6.0]], dtype=np.float32)
    true_np = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    false_np = np.array(
        [[-10.0, -20.0, -30.0], [-40.0, -50.0, -60.0]], dtype=np.float32
    )
    expected = np.where(a_np > b_np, true_np, false_np).astype(np.float32)

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        true_value = device_memory.create_tensor(
            true_np.__array_interface__["data"][0],
            true_np.nbytes,
            true_np.shape,
            "float32",
        )
        false_value = device_memory.create_tensor(
            false_np.__array_interface__["data"][0],
            false_np.nbytes,
            false_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [a, b, true_value, false_value, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        true_desc = graph.tensor(
            name="true", dim=true_np.shape, data_type=fe.DataType.FLOAT
        )
        false_desc = graph.tensor(
            name="false", dim=false_np.shape, data_type=fe.DataType.FLOAT
        )
        pred = graph.gt(a_desc, b_desc, name="pred")
        selected = graph.select(pred, true_desc, false_desc, name="selected")
        graph.validate().build_operation_graph().create_execution_plans(
            [fe.HeurMode.INSTANT]
        ).check_support().build_plans()

        (workspace_size,) = graph._native.plans[0].get_int64(
            backend.MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE
        )
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, ((workspace_size + 3) // 4,), "float32"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]

        graph.execute(
            {
                a_desc: device_memory.get_tensor_addr(a)[0],
                b_desc: device_memory.get_tensor_addr(b)[0],
                true_desc: device_memory.get_tensor_addr(true_value)[0],
                false_desc: device_memory.get_tensor_addr(false_value)[0],
                selected: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_matmul_broadcast_bias():
    import cudnn_frontend as fe
    import mudnn_frontend._backend as backend

    m, k, n = 16, 32, 24
    a_np = np.linspace(-0.5, 0.5, m * k, dtype=np.float32).reshape(m, k)
    b_np = np.linspace(0.35, -0.35, k * n, dtype=np.float32).reshape(k, n)
    bias_np = np.linspace(-0.2, 0.2, n, dtype=np.float32).reshape(1, n)
    expected = a_np @ b_np + bias_np

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [a, b, bias, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        matmul = graph.matmul(a_desc, b_desc, name="matmul")
        biased = graph.bias(matmul, bias_desc, name="bias")
        graph.validate().build_operation_graph().create_execution_plans(
            [fe.HeurMode.INSTANT]
        ).check_support().build_plans()

        (workspace_size,) = graph._native.plans[0].get_int64(
            backend.MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE
        )
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, ((workspace_size + 3) // 4,), "float32"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]

        graph.execute(
            {
                a_desc: device_memory.get_tensor_addr(a)[0],
                b_desc: device_memory.get_tensor_addr(b)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                biased: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=1.0e-3)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_execute_matmul_bias_relu():
    import cudnn_frontend as fe
    import mudnn_frontend._backend as backend

    m, k, n = 16, 32, 24
    a_np = np.linspace(-0.65, 0.65, m * k, dtype=np.float32).reshape(m, k)
    b_np = np.linspace(0.3, -0.3, k * n, dtype=np.float32).reshape(k, n)
    bias_np = np.linspace(-0.4, 0.4, m * n, dtype=np.float32).reshape(m, n)
    expected = np.maximum(a_np @ b_np + bias_np, 0.0)

    tensors = []
    try:
        a = device_memory.create_tensor(
            a_np.__array_interface__["data"][0],
            a_np.nbytes,
            a_np.shape,
            "float32",
        )
        b = device_memory.create_tensor(
            b_np.__array_interface__["data"][0],
            b_np.nbytes,
            b_np.shape,
            "float32",
        )
        bias = device_memory.create_tensor(
            bias_np.__array_interface__["data"][0],
            bias_np.nbytes,
            bias_np.shape,
            "float32",
        )
        out = device_memory.create_tensor(
            0, expected.nbytes, expected.shape, "float32"
        )
        tensors = [a, b, bias, out]

        graph = fe.pygraph(
            io_data_type=fe.DataType.FLOAT,
            intermediate_data_type=fe.DataType.FLOAT,
            compute_data_type=fe.DataType.FLOAT,
        )
        a_desc = graph.tensor(name="a", dim=a_np.shape, data_type=fe.DataType.FLOAT)
        b_desc = graph.tensor(name="b", dim=b_np.shape, data_type=fe.DataType.FLOAT)
        bias_desc = graph.tensor(
            name="bias", dim=bias_np.shape, data_type=fe.DataType.FLOAT
        )
        matmul = graph.matmul(a_desc, b_desc, name="matmul")
        biased = graph.pointwise(fe.PointwiseMode.ADD, matmul, bias_desc, name="bias")
        relu = graph.relu(biased, name="relu")
        graph.validate().build_operation_graph().create_execution_plans(
            [fe.HeurMode.INSTANT]
        ).check_support().build_plans()

        (workspace_size,) = graph._native.plans[0].get_int64(
            backend.MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE
        )
        workspace_ptr = None
        if workspace_size:
            workspace = device_memory.create_tensor(
                0, workspace_size, ((workspace_size + 3) // 4,), "float32"
            )
            tensors.append(workspace)
            workspace_ptr = device_memory.get_tensor_addr(workspace)[0]

        graph.execute(
            {
                a_desc: device_memory.get_tensor_addr(a)[0],
                b_desc: device_memory.get_tensor_addr(b)[0],
                bias_desc: device_memory.get_tensor_addr(bias)[0],
                relu: device_memory.get_tensor_addr(out)[0],
            },
            workspace_ptr=workspace_ptr,
        )

        actual = device_memory.copy_to_ndarray(out, expected.nbytes)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-3, atol=1.0e-3)
    finally:
        for tensor in tensors:
            device_memory.release_device_data(tensor)


def test_mudnn_frontend_python_graph_api_tensor_like():
    import cudnn_frontend as fe

    graph = fe.pygraph()
    array = np.zeros((2, 3, 4), dtype=np.float32)
    tensor = (
        graph.tensor_like(array, name="array")
        .set_data_type(fe.data_type.HALF)
        .set_stride((12, 4, 1))
        .set_output()
        .set_name("renamed")
    )

    assert tensor.name == "renamed"
    assert tensor.dims == (2, 3, 4)
    assert tensor.strides == (12, 4, 1)
    assert tensor.data_type == fe.data_type.HALF
    assert tensor.is_output is True
    assert tensor.is_virtual is False


def test_mudnn_frontend_python_graph_api_data_type_setters():
    import cudnn_frontend as fe

    graph = (
        fe.pygraph()
        .set_io_data_type(fe.data_type.HALF)
        .set_intermediate_data_type(fe.data_type.FLOAT)
        .set_compute_data_type(fe.data_type.FLOAT)
    )
    x = graph.tensor(name="x", dim=(2, 3))
    w = graph.tensor(name="w", dim=(3, 4))
    y = graph.matmul(x, w)

    assert graph.io_data_type == fe.DataType.HALF
    assert graph.intermediate_data_type == fe.DataType.FLOAT
    assert graph.compute_data_type == fe.DataType.FLOAT
    assert x.data_type == fe.DataType.HALF
    assert y.data_type == fe.DataType.HALF
    assert graph.nodes[-1].attributes["compute_data_type"] == fe.DataType.FLOAT


def test_mudnn_frontend_cpp_builds_pointwise_add_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_builds_pointwise_add_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_executes_matmul_bias_relu_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_executes_matmul_bias_relu_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_matmul_bias_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_executes_matmul_bias_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_pointwise_aliases_graph():
    fn = get_c_api_test_symbol(
        "mudnn_frontend_graph_api_executes_pointwise_aliases_graph"
    )
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_gen_index_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_executes_gen_index_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_serialized_scale_graph():
    fn = get_c_api_test_symbol(
        "mudnn_frontend_graph_api_executes_serialized_scale_graph"
    )
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_concat_reshape_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_executes_concat_reshape_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_reduction_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_executes_reduction_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_genstats_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_executes_genstats_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_rng_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_executes_rng_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_builds_batchnorm_graphs():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_builds_batchnorm_graphs")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_builds_conv_resample_graphs():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_builds_conv_resample_graphs")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_builds_batchnorm_forward_backward_graphs():
    fn = get_c_api_test_symbol(
        "mudnn_frontend_graph_api_builds_batchnorm_forward_backward_graphs"
    )
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_builds_norm_alias_graphs():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_builds_norm_alias_graphs")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_builds_block_scale_graphs():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_builds_block_scale_graphs")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_builds_paged_cache_signal_graphs():
    fn = get_c_api_test_symbol(
        "mudnn_frontend_graph_api_builds_paged_cache_signal_graphs"
    )
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_builds_sdpa_backward_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_builds_sdpa_backward_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_sdpa_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_graph_api_executes_sdpa_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_sdpa_bias_bool_mask_graph():
    fn = get_c_api_test_symbol(
        "mudnn_frontend_graph_api_executes_sdpa_bias_bool_mask_graph"
    )
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_graph_api_executes_sdpa_generate_stats_graph():
    fn = get_c_api_test_symbol(
        "mudnn_frontend_graph_api_executes_sdpa_generate_stats_graph"
    )
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_executes_matmul_broadcast_bias_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_executes_matmul_broadcast_bias_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_executes_compare_select_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_executes_compare_select_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_executes_scale_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_executes_scale_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_builds_sdpa_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_builds_sdpa_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_executes_sdpa_bias_bool_mask_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_executes_sdpa_bias_bool_mask_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_cpp_executes_sdpa_backward_graph():
    fn = get_c_api_test_symbol("mudnn_frontend_executes_sdpa_backward_graph")
    if fn is None:
        pytest.skip("mudnn frontend C++ wrapper test is a C API test")
    assert fn()


def test_mudnn_frontend_installed_cmake_consumer(tmp_path):
    install_prefix = _installed_mudnn_prefix()
    config_file = install_prefix / "lib" / "cmake" / "mudnn" / "mudnnConfig.cmake"
    if not config_file.exists():
        pytest.skip("installed mudnn CMake package is not available")

    source_dir = tmp_path / "consumer"
    build_dir = tmp_path / "build"
    source_dir.mkdir()
    (source_dir / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.12)",
                "project(mudnn_frontend_consumer LANGUAGES CXX)",
                "find_package(mudnn CONFIG REQUIRED)",
                "add_executable(consumer main.cc)",
                "target_link_libraries(consumer PRIVATE mudnn_frontend)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "main.cc").write_text(
        "\n".join(
            [
                '#include "mudnn_frontend.h"',
                "#include <vector>",
                "",
                "int main() {",
                "  if (cudnnGetVersion() == 0) return 21;",
                "  (void)cudnnGetCudartVersion();",
                "  constexpr int64_t dims[] = {1, 2, 3};",
                "  constexpr int64_t strides[] = {6, 3, 1};",
                "  constexpr int64_t attn_dims[] = {1, 1, 2, 3};",
                "  constexpr int64_t attn_strides[] = {6, 6, 3, 1};",
                "  auto tensor = mudnn_frontend::TensorBuilder()",
                "                    .setDim(3, dims)",
                "                    .setStrides(3, strides)",
                "                    .setId(7)",
                "                    .setDataType(CUDNN_DATA_FLOAT)",
                "                    .build();",
                "  auto mask = mudnn_frontend::TensorBuilder()",
                "                  .setDim(4, attn_dims)",
                "                  .setStrides(4, attn_strides)",
                "                  .setId(8)",
                "                  .setDataType(CUDNN_DATA_BOOLEAN)",
                "                  .build();",
                "  auto bias = mudnn_frontend::TensorBuilder()",
                "                  .setDim(4, attn_dims)",
                "                  .setStrides(4, attn_strides)",
                "                  .setId(9)",
                "                  .setDataType(CUDNN_DATA_FLOAT)",
                "                  .build();",
                "  auto stats = mudnn_frontend::TensorBuilder()",
                "                   .setDim(4, attn_dims)",
                "                   .setStrides(4, attn_strides)",
                "                   .setId(10)",
                "                   .setDataType(CUDNN_DATA_FLOAT)",
                "                   .setVirtual(true)",
                "                   .build();",
                "  static_assert(CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC ==",
                "                MUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC);",
                "  static_assert(CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC ==",
                "                MUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC);",
                "  static_assert(CUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK ==",
                "                MUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK);",
                "  static_assert(CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC ==",
                "                MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC);",
                "  static_assert(CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC ==",
                "                MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC);",
                "  static_assert(CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC ==",
                "                MUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC);",
                "  static_assert(CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC ==",
                "                MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC);",
                "  static_assert(CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC ==",
                "                MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC);",
                "  static_assert(CUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK ==",
                "                MUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK);",
                "  static_assert(CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR ==",
                "                MUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR);",
                "  static_assert(CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR ==",
                "                MUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR);",
                "  static_assert(CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION ==",
                "                MUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION);",
                "  static_assert(CUDNN_DATA_FP8_E4M3 == MUDNN_DATA_FP8_E4M3);",
                "  static_assert(CUDNN_DATA_FP8_E5M2 == MUDNN_DATA_FP8_E5M2);",
                "  static_assert(CUDNN_DATA_FAST_FLOAT_FOR_FP8 ==",
                "                MUDNN_DATA_FAST_FLOAT_FOR_FP8);",
                "  static_assert(CUDNN_DATA_FP8_E8M0 == MUDNN_DATA_FP8_E8M0);",
                "  static_assert(CUDNN_DATA_FP4_E2M1 == MUDNN_DATA_FP4_E2M1);",
                "  static_assert(CUDNN_DATA_INT4 == MUDNN_DATA_INT4);",
                "  static_assert(CUDNN_DATA_UINT4 == MUDNN_DATA_UINT4);",
                "  static_assert(CUDNN_DATA_UINT32 == MUDNN_DATA_UINT32);",
                "  static_assert(CUDNN_DATA_COMPLEX_FP32 ==",
                "                MUDNN_DATA_COMPLEX_FP32);",
                "  static_assert(CUDNN_DATA_COMPLEX_FP64 ==",
                "                MUDNN_DATA_COMPLEX_FP64);",
                "  static_assert(CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE ==",
                "                MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE);",
                "  static_assert(mudnn_frontend::DataType_t::FP8_E4M3 ==",
                "                static_cast<mudnn_frontend::DataType_t>(",
                "                    MUDNN_DATA_FP8_E4M3));",
                "  static_assert(mudnn_frontend::KnobType_t::KERNEL_CFG ==",
                "                static_cast<mudnn_frontend::KnobType_t>(12));",
                "  static_assert(mudnn_frontend::DiagonalAlignment_t::TOP_LEFT ==",
                "                mudnn_frontend::DiagonalAlignment::TOP_LEFT);",
                "  auto op = mudnn_frontend::OperationBuilder(",
                "                CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)",
                "                .setQDesc(tensor)",
                "                .setKDesc(tensor)",
                "                .setVDesc(tensor)",
                "                .setODesc(tensor)",
                "                .setBiasDesc(bias)",
                "                .setMaskDesc(mask)",
                "                .setSdpaFwdStatsDesc(stats)",
                "                .setSdpaScale(0.5)",
                "                .setSdpaDiagonalBandRightBound(0)",
                "                .setSdpaUseAlibiMask(false)",
                "                .setSdpaUseCausalMaskBottomRight(false)",
                "                .setSdpaDropout(0.0)",
                "                .setSdpaSeqLenQDesc(stats)",
                "                .setSdpaSeqLenKVDesc(stats)",
                "                .setSdpaGenerateStats(false);",
                "  auto bwd = mudnn_frontend::OperationBuilder(",
                "                 CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)",
                "                 .setQDesc(tensor)",
                "                 .setKDesc(tensor)",
                "                 .setVDesc(tensor)",
                "                 .setODesc(tensor)",
                "                 .setSdpaBwdDODesc(tensor)",
                "                 .setSdpaBwdStatsDesc(stats)",
                "                 .setSdpaBwdDQDesc(tensor)",
                "                 .setSdpaBwdDKDesc(tensor)",
                "                 .setSdpaBwdDVDesc(tensor)",
                "                 .setSdpaBwdScale(0.5)",
                "                 .setSdpaBwdDiagonalBandRightBound(0)",
                "                 .setSdpaBwdUseAlibiMask(false)",
                "                 .setSdpaBwdUseCausalMaskBottomRight(false)",
                "                 .setSdpaBwdDropout(0.0)",
                "                 .setSdpaBwdSeqLenQDesc(stats)",
                "                 .setSdpaBwdSeqLenKVDesc(stats)",
                "                 .setSdpaBwdUseDeterministicAlgorithm(true);",
                "  auto transpose = mudnn_frontend::OperationBuilder(",
                "                       CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR)",
                "                       .setTransposeXDesc(tensor)",
                "                       .setTransposeYDesc(tensor)",
                "                       .setTransposePermutation({0, 1, 2});",
                "  auto block_dequant = mudnn_frontend::OperationBuilder(",
                "                         CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR)",
                "                         .setBlockScaleDequantizeXDesc(tensor)",
                "                         .setBlockScaleDequantizeScaleDesc(tensor)",
                "                         .setBlockScaleDequantizeYDesc(tensor)",
                "                         .setBlockScaleDequantizeMathPrecision(CUDNN_DATA_FLOAT)",
                "                         .setBlockScaleDequantizeBlockSize(1)",
                "                         .setBlockScaleDequantizeNegativeScale(false)",
                "                         .setBlockScaleDequantizeIsNegativeScale(false);",
                "  mudnn_frontend::graph::Graph graph;",
                "  auto compile_scalar = graph.tensor(",
                "      2.5f, mudnn_frontend::graph::ScalarType::COMPILE_TIME_CONST);",
                '  compile_scalar->set_name("compile_scalar").set_dim({1}).set_stride({1});',
                "  auto ragged_offset = graph.tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("offsets")',
                "          .set_dim({3, 1, 1, 1})",
                "          .set_stride({1, 1, 1, 1})",
                "          .set_data_type(CUDNN_DATA_INT64)",
                "          .set_uid(70));",
                "  auto ragged = graph.tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("ragged")',
                "          .set_dim({2, 4, 8, 16})",
                "          .set_stride({512, 128, 16, 1})",
                "          .set_data_type(CUDNN_DATA_HALF)",
                "          .set_reordering_type(mudnn_frontend::TensorReordering_t::F16x16)",
                "          .set_uid(71)",
                "          .set_ragged_offset(ragged_offset)",
                "          .set_ragged_offset_multiplier(4));",
                '  auto ragged_clone = graph.tensor_like(ragged, "ragged_clone");',
                "  (void)ragged_clone;",
                "  auto score_graph = std::make_shared<mudnn_frontend::graph::Graph>();",
                "  auto score = score_graph->tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("score")',
                "          .set_dim({2, 4, 8, 8})",
                "          .set_stride({256, 64, 8, 1}));",
                "  auto score_bias = score_graph->tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("score_bias")',
                "          .set_dim({2, 1, 8, 8})",
                "          .set_stride({64, 64, 8, 1}));",
                "  auto biased_score = mudnn_frontend::graph::attn::score_modifiers::bias(",
                "      score_graph, score, score_bias);",
                "  (void)biased_score;",
                "  mudnn_frontend::graph::Tensor_attributes queried;",
                "  if (!graph.query_tensor_attributes_of_uid(71, queried).is_good()) return 12;",
                "  if (queried.get_ragged_offset_multiplier() != 4) return 13;",
                "  if (!queried.validate().is_good()) return 14;",
                "  mudnn_frontend::graph::Graph bad_graph;",
                "  auto bad_x = bad_graph.tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("missing_dims")',
                "          .set_stride({1, 1}));",
                "  (void)bad_x;",
                "  auto bad_status = bad_graph.validate();",
                "  if (bad_status.get_code() !=",
                "      mudnn_frontend::graph::error_code_t::ATTRIBUTE_NOT_SET) return 15;",
                "  mudnn_frontend::graph::Graph dup_graph;",
                "  dup_graph.tensor(mudnn_frontend::graph::Tensor_attributes()",
                '                       .set_name("a")',
                "                       .set_dim({1})",
                "                       .set_stride({1})",
                "                       .set_uid(1));",
                "  dup_graph.tensor(mudnn_frontend::graph::Tensor_attributes()",
                '                       .set_name("b")',
                "                       .set_dim({1})",
                "                       .set_stride({1})",
                "                       .set_uid(1));",
                "  auto dup_status = dup_graph.validate();",
                "  if (dup_status.get_code() !=",
                "      mudnn_frontend::graph::error_code_t::INVALID_VALUE) return 16;",
                "  if (dup_status.get_code() !=",
                "      mudnn_frontend::error_code_t::INVALID_VALUE) return 17;",
                "  std::string invalid_plan_name;",
                "  if (!graph.get_plan_name_at_index(-1, invalid_plan_name).is_bad())",
                "    return 18;",
                "  int64_t engine_id = -1;",
                "  std::unordered_map<mudnn_frontend::KnobType_t, int64_t> knobs;",
                "  if (!graph.get_engine_and_knobs_at_index(-1, engine_id, knobs).is_bad())",
                "    return 19;",
                "  if (!graph.create_execution_plan(-1, knobs).is_bad()) return 20;",
                "  mudnn_frontend::graph::SDPA_attributes sdpa_attrs;",
                "  mudnn_frontend::graph::SDPA_backward_attributes sdpa_bwd_attrs;",
                "  mudnn_frontend::graph::SDPA_fp8_attributes sdpa_fp8_attrs;",
                "  mudnn_frontend::graph::SDPA_fp8_backward_attributes sdpa_fp8_bwd_attrs;",
                "  sdpa_attrs.set_attn_scale(ragged_offset)",
                "            .set_bias(ragged)",
                "            .set_block_mask(ragged)",
                "            .set_logit_max(ragged)",
                "            .set_score_sum_exp(ragged)",
                "            .set_alibi_mask(true)",
                "            .set_diagonal_alignment(",
                "                mudnn_frontend::DiagonalAlignment_t::TOP_LEFT)",
                "            .set_diagonal_band_left_bound(2)",
                "            .set_diagonal_band_right_bound(4)",
                "            .set_dropout(0.1, ragged_offset, ragged_offset)",
                "            .set_dropout(ragged_offset, ragged)",
                "            .set_padding_mask(true)",
                "            .set_seq_len_q(ragged_offset)",
                "            .set_seq_len_kv(ragged_offset)",
                "            .set_cu_seq_len_q(ragged_offset)",
                "            .set_cu_seq_len_kv(ragged_offset)",
                "            .set_implementation(",
                "                mudnn_frontend::AttentionImplementation_t::UNIFIED);",
                "  sdpa_bwd_attrs.set_use_deterministic_algorithm(true)",
                "                .set_deterministic_algorithm(true)",
                "                .set_dbias(ragged)",
                "                .set_sink_token(ragged)",
                "                .set_dsink_token(ragged)",
                "                .set_paged_attention_k_table(ragged)",
                "                .set_paged_attention_v_table(ragged)",
                "                .set_paged_attention_max_seq_len_kv(8)",
                "                .set_score_mod([](auto, auto score) { return score; })",
                "                .set_score_mod_bprop([](auto, auto score) { return score; })",
                "                .set_seq_len_q(ragged_offset)",
                "                .set_seq_len_kv(ragged_offset);",
                "  mudnn_frontend::graph::Graph sdpa_band_graph;",
                "  sdpa_band_graph.set_io_data_type(",
                "      mudnn_frontend::DataType_t::FLOAT)",
                "      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)",
                "      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);",
                "  auto q = sdpa_band_graph.tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("q")',
                "          .set_dim({1, 1, 4, 2})",
                "          .set_stride({8, 8, 2, 1})",
                "          .set_data_type(mudnn_frontend::DataType_t::FLOAT));",
                "  auto k = sdpa_band_graph.tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("k")',
                "          .set_dim({1, 1, 4, 2})",
                "          .set_stride({8, 8, 2, 1})",
                "          .set_data_type(mudnn_frontend::DataType_t::FLOAT));",
                "  auto v = sdpa_band_graph.tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("v")',
                "          .set_dim({1, 1, 4, 2})",
                "          .set_stride({8, 8, 2, 1})",
                "          .set_data_type(mudnn_frontend::DataType_t::FLOAT));",
                "  mudnn_frontend::graph::SDPA_attributes band_attrs;",
                "  band_attrs.set_generate_stats(true)",
                "            .set_diagonal_alignment(",
                "                mudnn_frontend::DiagonalAlignment_t::BOTTOM_RIGHT)",
                "            .set_diagonal_band_right_bound(1);",
                "  auto band_outputs = sdpa_band_graph.sdpa(q, k, v, band_attrs);",
                "  (void)band_outputs;",
                "  auto band_status = sdpa_band_graph.validate();",
                "  if (band_status.get_code() !=",
                "      mudnn_frontend::graph::error_code_t::GRAPH_NOT_SUPPORTED)",
                "    return 21;",
                '  sdpa_fp8_attrs.set_name("fp8")',
                "                .set_generate_stats(true)",
                "                .set_causal_mask(true)",
                "                .set_causal_mask_bottom_right(true)",
                "                .set_attn_scale(0.5)",
                "                .set_compute_data_type(",
                "                    mudnn_frontend::DataType_t::FLOAT);",
                "  auto fp8_outputs = graph.sdpa_fp8(",
                "      ragged, ragged, ragged, ragged, ragged, ragged, ragged,",
                "      ragged, ragged, sdpa_fp8_attrs);",
                "  (void)fp8_outputs;",
                "  auto mxfp8_outputs = graph.sdpa_fp8(",
                "      ragged, ragged, ragged, ragged, ragged, ragged, sdpa_fp8_attrs);",
                "  (void)mxfp8_outputs;",
                '  sdpa_fp8_bwd_attrs.set_name("fp8_bwd")',
                "                    .set_causal_mask(true)",
                "                    .set_causal_mask_bottom_right(true)",
                "                    .set_deterministic_algorithm(true)",
                "                    .set_attn_scale(0.5)",
                "                    .set_compute_data_type(",
                "                        mudnn_frontend::DataType_t::FLOAT);",
                "  auto fp8_bwd_outputs = graph.sdpa_fp8_backward(",
                "      ragged, ragged, ragged, ragged, ragged, ragged, ragged, ragged,",
                "      ragged, ragged, ragged, ragged, ragged, ragged, ragged, ragged,",
                "      ragged, ragged, sdpa_fp8_bwd_attrs);",
                "  (void)fp8_bwd_outputs;",
                "  auto mxfp8_bwd_outputs = graph.sdpa_fp8_backward(",
                "      ragged, ragged, ragged, ragged, ragged, ragged, ragged, ragged,",
                "      ragged, ragged, ragged, ragged, ragged, ragged, ragged, ragged,",
                "      ragged, sdpa_fp8_bwd_attrs);",
                "  (void)mxfp8_bwd_outputs;",
                "  mudnn_frontend::graph::Reduction_attributes red_attrs;",
                "  auto reduced = graph.reduction(ragged, red_attrs);",
                "  (void)reduced;",
                "  const auto graph_key = graph.key();",
                "  (void)graph_key;",
                "  std::vector<uint8_t> serialized;",
                "  if (!graph.serialize(serialized).is_good()) return 22;",
                "  mudnn_frontend::graph::Graph restored;",
                "  if (!restored.deserialize(nullptr, serialized).is_good()) return 23;",
                "  auto sorted_uids = restored.get_variant_pack_uids_sorted();",
                "  if (sorted_uids.empty()) return 24;",
                "  mudnn_frontend::graph::Graph empty_build_graph;",
                "  auto empty_build_status = empty_build_graph.build(",
                "      nullptr, {mudnn_frontend::HeurMode_t::INSTANT});",
                "  if (!empty_build_status.is_bad()) return 25;",
                "  auto graph_x = graph.tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("quantized_input")',
                "          .set_dim({1, 32, 32})",
                "          .set_stride({1024, 32, 1})",
                "          .set_data_type(CUDNN_DATA_INT8));",
                "  auto graph_scale = graph.tensor(",
                "      mudnn_frontend::graph::Tensor_attributes()",
                '          .set_name("scale")',
                "          .set_dim({1, 32, 1})",
                "          .set_stride({32, 1, 1})",
                "          .set_data_type(CUDNN_DATA_FLOAT));",
                "  auto dq_attributes =",
                "      mudnn_frontend::graph::Block_scale_dequantize_attributes()",
                '          .set_name("dq_negative")',
                "          .set_block_size(std::vector<int32_t>{1, 32})",
                "          .set_is_negative_scale(true)",
                "          .set_compute_data_type(CUDNN_DATA_FLOAT);",
                "  if (!dq_attributes.get_is_negative_scale()) return 8;",
                "  auto graph_y = graph.block_scale_dequantize(",
                "      graph_x, graph_scale, dq_attributes);",
                '  if (graph_y->get_name() != "dq_negative::Y") return 9;',
                "  graph_y->set_output(true).set_data_type(CUDNN_DATA_FLOAT);",
                "  bool rejected = false;",
                "  try {",
                "    mudnn_frontend::OperationBuilder(",
                "        CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)",
                "        .setSdpaUseAlibiMask(true);",
                "  } catch (const std::invalid_argument&) {",
                "    rejected = true;",
                "  }",
                "  if (!rejected) return 2;",
                "  rejected = false;",
                "  try {",
                "    mudnn_frontend::OperationBuilder(",
                "        CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)",
                "        .setSdpaBwdDropout(0.5);",
                "  } catch (const std::invalid_argument&) {",
                "    rejected = true;",
                "  }",
                "  if (!rejected) return 3;",
                "  rejected = false;",
                "  try {",
                "    mudnn_frontend::OperationBuilder(",
                "        CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)",
                "        .setSdpaScoreMod([](){});",
                "  } catch (const std::invalid_argument&) {",
                "    rejected = true;",
                "  }",
                "  if (!rejected) return 4;",
                "  rejected = false;",
                "  try {",
                "    mudnn_frontend::OperationBuilder(",
                "        CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)",
                "        .setSdpaBwdDSinkTokenDesc(stats);",
                "  } catch (const std::invalid_argument&) {",
                "    rejected = true;",
                "  }",
                "  if (!rejected) return 5;",
                "  rejected = false;",
                "  try {",
                "    mudnn_frontend::OperationBuilder(",
                "        CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)",
                "        .setSdpaDescaleQDesc(stats);",
                "  } catch (const std::invalid_argument&) {",
                "    rejected = true;",
                "  }",
                "  if (!rejected) return 6;",
                "  rejected = false;",
                "  try {",
                "    mudnn_frontend::OperationBuilder(",
                "        CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)",
                "        .setSdpaBwdScaleDQDesc(stats);",
                "  } catch (const std::invalid_argument&) {",
                "    rejected = true;",
                "  }",
                "  if (!rejected) return 7;",
                "  (void)op;",
                "  (void)bwd;",
                "  (void)transpose;",
                "  (void)block_dequant;",
                "  (void)graph_y;",
                "  return tensor.get_backend_descriptor() != nullptr &&",
                "         mask.get_backend_descriptor() != nullptr &&",
                "         bias.get_backend_descriptor() != nullptr ? 0 : 1;",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={install_prefix}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir)], check=True)
    subprocess.run([str(build_dir / "consumer")], check=True)


def test_mudnn_frontend_cpp_json_roundtrip_runtime(tmp_path):
    repo_root = _REPO_ROOT
    install_prefix = _installed_mudnn_prefix()
    header = install_prefix / "include" / "mudnn_frontend.h"
    thirdparty = (
        repo_root.parent
        / "mudnn-frontend"
        / "include"
        / "cudnn_frontend"
        / "thirdparty"
    )
    if not header.exists() or not (thirdparty / "nlohmann" / "json.hpp").exists():
        pytest.skip(
            "installed frontend header or upstream nlohmann header is unavailable"
        )

    source = tmp_path / "json_roundtrip.cc"
    binary = tmp_path / "json_roundtrip"
    source.write_text(
        "\n".join(
            [
                '#include "mudnn_frontend.h"',
                "#include <iostream>",
                "#include <vector>",
                "",
                "int main() {",
                "  namespace fe = mudnn_frontend;",
                "  auto tensor = fe::graph::Tensor_attributes()",
                '                    .set_name("image")',
                "                    .set_dim({4, 32, 16, 16})",
                "                    .set_stride({8192, 1, 512, 32})",
                "                    .set_is_virtual(true)",
                "                    .set_is_pass_by_value(true)",
                "                    .set_uid(12312)",
                "                    .set_reordering_type(fe::TensorReordering_t::F16x16)",
                "                    .set_data_type(fe::DataType_t::HALF);",
                "  json tensor_json = tensor;",
                "  auto tensor2 = tensor_json.get<fe::graph::Tensor_attributes>();",
                "  if (!(tensor2 == tensor)) return 1;",
                "",
                "  auto conv = fe::graph::Conv_fprop_attributes()",
                '                  .set_name("conv_fprop")',
                "                  .set_pre_padding({2, 0})",
                "                  .set_post_padding({0, 2})",
                "                  .set_stride({1, 1})",
                "                  .set_dilation({1, 1})",
                "                  .set_compute_data_type(fe::DataType_t::FLOAT);",
                "  json conv_json = conv;",
                "  auto conv2 = conv_json.get<fe::graph::Conv_fprop_attributes>();",
                "  if (!(conv2 == conv)) return 2;",
                "  if (conv2.get_padding() != std::vector<int64_t>({2, 0})) return 12;",
                "  if (conv2.get_pre_padding() != std::vector<int64_t>({2, 0})) return 13;",
                "  if (conv2.get_post_padding() != std::vector<int64_t>({0, 2})) return 14;",
                "",
                "  fe::graph::Graph graph;",
                "  graph.set_io_data_type(fe::DataType_t::HALF)",
                "       .set_intermediate_data_type(fe::DataType_t::FLOAT)",
                "       .set_compute_data_type(fe::DataType_t::FLOAT);",
                "  auto a = graph.tensor(fe::graph::Tensor_attributes()",
                '                          .set_name("A")',
                "                          .set_dim({2, 3})",
                "                          .set_stride({3, 1})",
                "                          .set_uid(1));",
                "  auto b = graph.tensor(fe::graph::Tensor_attributes()",
                '                          .set_name("B")',
                "                          .set_dim({2, 3})",
                "                          .set_stride({3, 1})",
                "                          .set_uid(2));",
                "  auto y = graph.add(",
                '      a, b, fe::graph::Pointwise_attributes().set_name("add"));',
                "  y->set_output(true).set_uid(3);",
                "  json graph_json = graph;",
                "  fe::graph::Graph restored;",
                "  auto status = restored.deserialize(graph_json);",
                "  if (!status.is_good()) return 3;",
                "  json graph_json2 = restored;",
                "  if (graph_json != graph_json2) return 4;",
                "  auto uids = restored.get_variant_pack_uids_sorted();",
                "  if (uids != std::vector<int64_t>({1, 2, 3})) return 5;",
                "  return 0;",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-I",
            str(install_prefix / "include"),
            *_backend_include_flags(),
            "-I",
            str(thirdparty),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


def test_mudnn_frontend_cpp_json_fallback_syntax(tmp_path):
    repo_root = _REPO_ROOT
    frontend_include = _FRONTEND_ROOT / "include"
    if not (frontend_include / "mudnn_frontend.h").exists():
        pytest.skip("source frontend header is unavailable")

    source = tmp_path / "json_fallback_syntax.cc"
    source.write_text(
        "\n".join(
            [
                "#include <mudnn_frontend.h>",
                "#include <iostream>",
                "#include <memory>",
                "#include <unordered_map>",
                "",
                "int main() {",
                "  namespace fe = mudnn_frontend;",
                "  cudnnHandle_t handle = nullptr;",
                "  if (cudnnGetVersion() == 0) return 11;",
                "  auto conv = fe::graph::Conv_fprop_attributes()",
                '                  .set_name("conv")',
                "                  .set_pre_padding({2, 0})",
                "                  .set_post_padding({0, 2})",
                "                  .set_stride({1, 1})",
                "                  .set_dilation({1, 1});",
                "  if (conv.get_padding() != std::vector<int64_t>({2, 0})) return 12;",
                "  if (conv.get_pre_padding() != std::vector<int64_t>({2, 0})) return 13;",
                "  if (conv.get_post_padding() != std::vector<int64_t>({0, 2})) return 14;",
                "  json conv_json = conv;",
                "  auto conv_copy = conv_json.get<fe::graph::Conv_fprop_attributes>();",
                "  if (!(conv_copy == conv)) return 15;",
                "  auto tensor = fe::graph::Tensor_attributes()",
                '                    .set_name("image")',
                "                    .set_dim({4, 32, 16, 16})",
                "                    .set_stride({8192, 1, 512, 32})",
                "                    .set_is_virtual(true)",
                "                    .set_is_pass_by_value(true)",
                "                    .set_uid(12312)",
                "                    .set_reordering_type(fe::TensorReordering_t::F16x16)",
                "                    .set_alignment(32)",
                "                    .set_vector_count_and_dimension(4, 1)",
                "                    .set_data_type(fe::DataType_t::HALF);",
                "  if (tensor.get_alignment() != 32) return 5;",
                "  if (tensor.get_vector_count() != 4) return 6;",
                "  if (tensor.get_vector_dimension() != 1) return 7;",
                "  if (!tensor.get_is_virtual()) return 8;",
                "  if (!tensor.get_is_pass_by_value()) return 9;",
                "  if (tensor.get_has_compile_time_constant()) return 10;",
                "  json tensor_json = tensor;",
                "  auto tensor_copy = tensor_json;",
                "  if (!(tensor_copy == tensor)) return 1;",
                "",
                "  fe::graph::Graph graph;",
                "  graph.set_io_data_type(fe::DataType_t::HALF)",
                "       .set_intermediate_data_type(fe::DataType_t::FLOAT)",
                "       .set_compute_data_type(fe::DataType_t::FLOAT);",
                "  graph.set_override_shape_enabled(true);",
                "  auto dy = graph.tensor(fe::graph::Tensor_attributes()",
                '                           .set_name("DY")',
                "                           .set_dim({4, 64, 16, 16})",
                "                           .set_stride({64 * 16 * 16, 1, 64 * 16, 64})",
                "                           .set_uid(21));",
                "  auto w = graph.tensor(fe::graph::Tensor_attributes()",
                '                          .set_name("W")',
                "                          .set_dim({64, 32, 3, 3})",
                "                          .set_stride({32 * 3 * 3, 1, 32 * 3, 32})",
                "                          .set_uid(22));",
                "  auto dx = graph.conv_dgrad(dy, w, conv);",
                "  dx->set_dim({4, 32, 16, 16}).set_output(true).set_uid(23);",
                "  auto dw = graph.conv_wgrad(dy, dx, conv);",
                "  dw->set_output(true).set_uid(28);",
                "  auto dy_like = graph.tensor_like(dy);",
                '  dy_like->set_name("dy_like").set_uid(29);',
                "  auto scalar = graph.tensor(1.0f);",
                "  (void)scalar;",
                "  auto param = graph.tensor(fe::graph::Tensor_attributes()",
                '                              .set_name("param")',
                "                              .set_dim({1, 32, 1, 1})",
                "                              .set_stride({32, 1, 32, 32})",
                "                              .set_data_type(fe::DataType_t::FLOAT)",
                "                              .set_uid(24));",
                "  auto dbn_options = fe::graph::DBN_weight_attributes();",
                "  auto dbn_outputs = graph.dbn_weight(dx, dy, param, param, param, dbn_options);",
                "  std::get<0>(dbn_outputs)->set_output(true).set_uid(25);",
                "  auto bn_options = fe::graph::Batchnorm_attributes()",
                "                    .set_epsilon(scalar)",
                "                    .set_previous_running_stats(param, param, scalar)",
                "                    .set_peer_stats({param});",
                "  auto bn_outputs = graph.batchnorm(dx, param, param, bn_options);",
                "  (void)bn_outputs;",
                "  fe::graph::Tensor_attributes scalar_attr(1.0f);",
                '  auto epsilon = graph.tensor(scalar_attr.set_name("epsilon"));',
                "  auto accum_count = graph.tensor(static_cast<int64_t>(64));",
                "  auto bn_finalize_attr = fe::graph::BN_finalize_attributes()",
                "                              .set_previous_running_stats(param, param, scalar);",
                "  auto bn_finalize_outputs = graph.bn_finalize(",
                "      param, param, param, param, epsilon, accum_count, bn_finalize_attr);",
                "  (void)bn_finalize_outputs;",
                "  auto bn_infer_attr = fe::graph::Batchnorm_inference_attributes();",
                "  auto bn_infer = graph.batchnorm_inference(",
                "      dx, param, param, param, param, bn_infer_attr);",
                "  (void)bn_infer;",
                "  auto dbn_attr = fe::graph::Batchnorm_backward_attributes()",
                "                      .set_saved_mean_and_inv_variance(param, param)",
                "                      .set_peer_stats({param});",
                "  auto dbn_saved_outputs = graph.batchnorm_backward(dy, dx, param, dbn_attr);",
                "  (void)dbn_saved_outputs;",
                "  auto ln_attr = fe::graph::Layernorm_backward_attributes()",
                "                     .set_saved_mean_and_inv_variance(param, param);",
                "  auto ln_bwd_outputs = graph.layernorm_backward(dy, dx, param, ln_attr);",
                "  (void)ln_bwd_outputs;",
                "  auto ada_attr = fe::graph::AdaLayernorm_attributes()",
                "                      .set_forward_phase(fe::NormFwdPhase_t::TRAINING)",
                "                      .set_epsilon(epsilon);",
                "  auto ada_outputs = graph.adalayernorm(dx, param, param, ada_attr);",
                "  (void)ada_outputs;",
                "  auto ada_bwd_attr = fe::graph::AdaLayernorm_backward_attributes()",
                "                          .set_saved_mean_and_inv_variance(param, param);",
                "  auto ada_bwd_outputs = graph.adalayernorm_backward(dy, dx, param, ada_bwd_attr);",
                "  (void)ada_bwd_outputs;",
                "  auto rms_attr = fe::graph::Rmsnorm_attributes()",
                "                      .set_forward_phase(fe::NormFwdPhase_t::INFERENCE)",
                "                      .set_bias(param)",
                "                      .set_epsilon(epsilon);",
                "  auto rms_outputs = graph.rmsnorm(dx, param, rms_attr);",
                "  (void)rms_outputs;",
                "  auto rms_bwd_attr = fe::graph::Rmsnorm_backward_attributes()",
                "                         .has_dbias(false);",
                "  auto rms_bwd_outputs = graph.rmsnorm_backward(",
                "      dy, dx, param, param, rms_bwd_attr);",
                "  (void)rms_bwd_outputs;",
                "  auto clipped_relu_attr = fe::graph::Pointwise_attributes()",
                "                               .set_mode(fe::PointwiseMode_t::RELU_FWD)",
                "                               .set_relu_lower_clip(0.0)",
                "                               .set_relu_upper_clip(6.0)",
                "                               .set_relu_lower_clip_slope(0.1);",
                "  auto clipped_relu = graph.pointwise(dx, clipped_relu_attr);",
                "  (void)clipped_relu;",
                "  auto concat_options = fe::graph::Concatenate_attributes()",
                "                            .set_axis(1)",
                "                            .set_in_place_index(0);",
                "  auto concat = graph.concatenate({dx, dx}, concat_options);",
                "  concat->set_output(true).set_uid(26);",
                "  auto reshaped = graph.reshape(",
                "      dx, fe::graph::Reshape_attributes()",
                '              .set_name("reshape")',
                "              .set_reshape_mode(fe::ReshapeMode_t::LOGICAL));",
                "  reshaped->set_dim({4, 16, 32}).set_stride({512, 32, 1});",
                "  auto transposed = graph.transpose(",
                "      reshaped, fe::graph::Transpose_attributes()",
                '                    .set_name("transpose")',
                "                    .set_permutation({0, 2, 1})",
                "                    .set_compute_data_type(fe::DataType_t::FLOAT));",
                "  (void)transposed;",
                "  auto sliced = graph.slice(",
                "      dx, fe::graph::Slice_attributes()",
                '              .set_name("slice")',
                "              .set_slices({{0, 4}, {0, 32}, {0, 16}, {0, 16}})",
                "              .set_strides({1, 1, 1, 1}));",
                "  (void)sliced;",
                "  auto resample_attr = fe::graph::Resample_attributes()",
                "                         .set_generate_index(true)",
                "                         .set_resampling_mode(fe::ResampleMode_t::MAXPOOL)",
                "                         .set_window({2, 2})",
                "                         .set_pre_padding({1, 1})",
                "                         .set_post_padding({2, 2})",
                "                         .set_stride({2, 2});",
                "  (void)resample_attr;",
                "  auto genstats = graph.genstats(dx, fe::graph::Genstats_attributes());",
                "  std::get<0>(genstats)->set_output(true).set_uid(27);",
                "  auto bq_attr = fe::graph::Block_scale_quantize_attributes()",
                "                   .set_block_size({1, 32})",
                "                   .set_axis(3)",
                "                   .set_transpose(false);",
                "  (void)bq_attr;",
                "  auto bdq_attr = fe::graph::Block_scale_dequantize_attributes()",
                "                    .set_block_size({32, 1});",
                "  (void)bdq_attr;",
                "  auto moe_token = graph.tensor(fe::graph::Tensor_attributes()",
                '                              .set_name("moe_token")',
                "                              .set_dim({1, 8, 16})",
                "                              .set_stride({128, 16, 1})",
                "                              .set_data_type(fe::DataType_t::HALF)",
                "                              .set_uid(30));",
                "  auto moe_weight = graph.tensor(fe::graph::Tensor_attributes()",
                '                               .set_name("moe_weight")',
                "                               .set_dim({2, 16, 32})",
                "                               .set_stride({512, 32, 1})",
                "                               .set_data_type(fe::DataType_t::HALF)",
                "                               .set_uid(31));",
                "  auto moe_offsets = graph.tensor(fe::graph::Tensor_attributes()",
                '                                .set_name("moe_offsets")',
                "                                .set_dim({2, 1, 1})",
                "                                .set_stride({1, 1, 1})",
                "                                .set_data_type(fe::DataType_t::INT32)",
                "                                .set_uid(32));",
                "  auto moe_index = graph.tensor(fe::graph::Tensor_attributes()",
                '                              .set_name("moe_index")',
                "                              .set_dim({2, 8})",
                "                              .set_stride({8, 1})",
                "                              .set_data_type(fe::DataType_t::INT32)",
                "                              .set_uid(33));",
                "  auto moe_ks = graph.tensor(fe::graph::Tensor_attributes()",
                '                           .set_name("moe_ks")',
                "                           .set_dim({8})",
                "                           .set_stride({1})",
                "                           .set_data_type(fe::DataType_t::INT32)",
                "                           .set_uid(34));",
                "  auto moe_attr = fe::graph::Moe_grouped_matmul_attributes()",
                '                    .set_name("moe")',
                "                    .set_mode(fe::MoeGroupedMatmulMode_t::SCATTER)",
                "                    .set_compute_data_type(fe::DataType_t::FLOAT)",
                "                    .set_top_k(2);",
                "  auto moe_output = graph.moe_grouped_matmul(",
                "      moe_token, moe_weight, moe_offsets, moe_index, moe_ks, moe_attr);",
                "  moe_output->set_output(true).set_uid(35);",
                "  auto moe_bwd_attr = fe::graph::Moe_grouped_matmul_bwd_attributes()",
                '                        .set_name("moe_bwd")',
                "                        .set_compute_data_type(fe::DataType_t::FLOAT);",
                "  auto moe_dweight = graph.moe_grouped_matmul_bwd(",
                "      moe_output, moe_token, moe_offsets, moe_bwd_attr);",
                "  moe_dweight->set_output(true).set_uid(36);",
                "  json graph_json = graph;",
                "  std::cout << graph_json << std::endl;",
                "  std::cout << graph << std::endl;",
                "  (void)graph.get_workspace_size();",
                "  (void)graph.get_workspace_size_plan_at_index(0);",
                "  (void)graph.get_autotune_workspace_size();",
                "  std::vector<int64_t> override_uids = {21, 22, 23};",
                "  std::vector<std::vector<int64_t>> override_shapes = {",
                "      {4, 64, 16, 16}, {64, 32, 3, 3}, {4, 32, 16, 16}};",
                "  std::vector<std::vector<int64_t>> override_strides = override_shapes;",
                "  int64_t workspace_size = 0;",
                "  (void)graph.get_workspace_size(handle, workspace_size, override_uids,",
                "                                 override_shapes, override_strides);",
                "  (void)graph.get_workspace_size_plan_at_index(",
                "      handle, 0, workspace_size, override_uids, override_shapes,",
                "      override_strides);",
                "  int64_t engine_count = 0;",
                "  (void)graph.get_engine_count(engine_count);",
                "  std::vector<fe::Knob> knobs;",
                "  (void)graph.get_knobs_for_engine(0, knobs);",
                "  (void)graph.build({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});",
                "  (void)graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);",
                "  (void)graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false);",
                "  (void)graph.warmup(handle);",
                '  graph.deselect_engines({"eng4_"});',
                "  graph.deselect_shared_mem_greater_than(256 * 1024);",
                "  graph.deselect_workspace_greater_than(1024 * 1024);",
                "  (void)graph.print();",
                "  std::unordered_map<int64_t, void*> variant_pack;",
                "  (void)graph.execute(handle, variant_pack, nullptr);",
                "  (void)graph.execute(handle, variant_pack, nullptr, override_uids,",
                "                      override_shapes, override_strides);",
                "  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> tensor_pack = {",
                "      {dy, nullptr}, {w, nullptr}, {dx, nullptr}};",
                "  (void)graph.autotune(handle, tensor_pack, nullptr);",
                "  (void)graph.execute(handle, tensor_pack, nullptr);",
                "  (void)graph.populate_cuda_graph(handle, tensor_pack, nullptr, nullptr);",
                "  (void)graph.update_cuda_graph(handle, tensor_pack, nullptr, nullptr);",
                "  (void)graph.populate_cuda_graph(handle, variant_pack, nullptr, nullptr);",
                "  (void)graph.update_cuda_graph(handle, variant_pack, nullptr, nullptr);",
                "  struct cudaGraphOpaque;",
                "  using cudaGraph_t = cudaGraphOpaque*;",
                "  cudaGraph_t cuda_graph = nullptr;",
                "  (void)graph.populate_cuda_graph(handle, tensor_pack, nullptr, cuda_graph);",
                "  (void)graph.update_cuda_graph(handle, tensor_pack, nullptr, cuda_graph);",
                "  (void)graph.populate_cuda_graph(handle, variant_pack, nullptr, cuda_graph);",
                "  (void)graph.update_cuda_graph(handle, variant_pack, nullptr, cuda_graph);",
                "  fe::graph::Graph restored;",
                "  (void)restored.deserialize(graph_json);",
                "  json graph_json2 = restored;",
                "  if (graph_json != graph_json2) return 2;",
                "  return 0;",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-fsyntax-only",
            "-DMUDNN_FRONTEND_FORCE_FALLBACK_JSON",
            "-I",
            str(frontend_include),
            "-I",
            str(repo_root / "include" / "mudnn" / "export" / "c"),
            "-I",
            str(repo_root / "include" / "mudnn" / "export" / "cc"),
            "-I",
            str(repo_root / "include"),
            *_backend_include_flags(),
            str(source),
        ],
        check=True,
    )

    upstream_serialize = (
        repo_root.parent / "mudnn-frontend" / "test" / "cpp" / "serialize.cpp"
    )
    if upstream_serialize.exists():
        catch_dir = tmp_path / "catch2"
        catch_dir.mkdir()
        (catch_dir / "catch_test_macros.hpp").write_text(
            "\n".join(
                [
                    "#pragma once",
                    "#define MUDNN_FRONTEND_JOIN2(a, b) a##b",
                    "#define MUDNN_FRONTEND_JOIN(a, b) MUDNN_FRONTEND_JOIN2(a, b)",
                    "#define TEST_CASE(...) \\",
                    "  static void MUDNN_FRONTEND_JOIN(mudnn_frontend_fake_test_, __LINE__)()",
                    "#define REQUIRE(...) do { (void)sizeof(__VA_ARGS__); } while (0)",
                    "#define SKIP(...) do { return; } while (0)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-fsyntax-only",
                "-DMUDNN_FRONTEND_FORCE_FALLBACK_JSON",
                "-I",
                str(tmp_path),
                "-I",
                str(frontend_include),
                "-I",
                str(repo_root / "include" / "mudnn" / "export" / "c"),
                "-I",
                str(repo_root / "include" / "mudnn" / "export" / "cc"),
                "-I",
                str(repo_root / "include"),
                *_backend_include_flags(),
                str(upstream_serialize),
            ],
            check=True,
        )


def _rewrite_upstream_cpp_tree(upstream_root, out_root):
    """Copy an upstream mudnn-frontend C++ tree applying the mudnn rename.

    The installed frontend tree only provides mudnn_frontend names, so upstream
    sources are compiled after the same mechanical rename a downstream consumer
    would apply when migrating.
    """
    for path in sorted(upstream_root.rglob("*")):
        if path.suffix not in (".cpp", ".h", ".hpp", ".cuh"):
            continue
        rel = path.relative_to(upstream_root)
        text = path.read_text(encoding="utf-8")
        text = text.replace("cudnn_frontend", "mudnn_frontend").replace(
            "CUDNN_FRONTEND", "MUDNN_FRONTEND"
        )
        out = out_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


def test_mudnn_frontend_upstream_cpp_tests_syntax(tmp_path):
    repo_root = _REPO_ROOT
    install_prefix = _installed_mudnn_prefix()
    header = install_prefix / "include" / "mudnn_frontend.h"
    upstream_cpp = repo_root.parent / "mudnn-frontend" / "test" / "cpp"
    thirdparty = (
        repo_root.parent
        / "mudnn-frontend"
        / "include"
        / "cudnn_frontend"
        / "thirdparty"
    )
    if not header.exists() or not upstream_cpp.exists():
        pytest.skip("installed frontend header or upstream C++ tests are unavailable")

    catch_dir = tmp_path / "catch2"
    catch_dir.mkdir()
    (catch_dir / "catch_test_macros.hpp").write_text(
        "\n".join(
            [
                "#pragma once",
                "#define MUDNN_FRONTEND_JOIN2(a, b) a##b",
                "#define MUDNN_FRONTEND_JOIN(a, b) MUDNN_FRONTEND_JOIN2(a, b)",
                "#define TEST_CASE(...) \\",
                "  static void MUDNN_FRONTEND_JOIN(mudnn_frontend_fake_test_, __LINE__)()",
                "#define REQUIRE(...) do { (void)sizeof(__VA_ARGS__); } while (0)",
                "#define SKIP(...) do { return; } while (0)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    sources = sorted(upstream_cpp.glob("*.cpp"))
    assert sources
    rewritten_root = tmp_path / "upstream_rewritten"
    _rewrite_upstream_cpp_tree(upstream_cpp, rewritten_root)
    for source in sources:
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-fsyntax-only",
                "-I",
                str(tmp_path),
                "-I",
                str(install_prefix / "include"),
                *_backend_include_flags(),
                "-I",
                str(thirdparty),
                str(rewritten_root / source.name),
            ],
            check=True,
        )


def _write_upstream_cpp_sample_syntax_shims(tmp_path: Path):
    catch_dir = tmp_path / "catch2"
    (catch_dir / "benchmark").mkdir(parents=True)
    (catch_dir / "generators").mkdir(parents=True)
    catch_dir.mkdir(exist_ok=True)
    (catch_dir / "catch_test_macros.hpp").write_text(
        textwrap.dedent(
            """
            #pragma once
            #include <stdexcept>
            #include <utility>
            #define MUDNN_FRONTEND_JOIN2(a,b) a##b
            #define MUDNN_FRONTEND_JOIN(a,b) MUDNN_FRONTEND_JOIN2(a,b)
            #define TEST_CASE(...) static void MUDNN_FRONTEND_JOIN(mudnn_frontend_fake_test_, __LINE__)()
            #define SECTION(...) if (true)
            #define REQUIRE(...) do { (void)sizeof(__VA_ARGS__); } while (0)
            #define REQUIRE_FALSE(...) do { (void)sizeof(!(__VA_ARGS__)); } while (0)
            #define CHECK(...) do { (void)sizeof(__VA_ARGS__); } while (0)
            #define FAIL(...) do { throw std::runtime_error("fail"); } while (0)
            #define SKIP(...) do { throw std::runtime_error("skip"); } while (0)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (catch_dir / "benchmark" / "catch_benchmark.hpp").write_text(
        textwrap.dedent(
            """
            #pragma once
            namespace Catch { namespace Benchmark {
            struct Chronometer {
              int runs() const { return 1; }
              template <typename F> void measure(F&&) {}
            };
            } }
            #define BENCHMARK_ADVANCED(...) if (false) {} else [&]
            #define BENCHMARK(...) [&]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    generators = textwrap.dedent(
        """
        #pragma once
        #include <initializer_list>
        namespace Catch { namespace Generators {
        template <typename T> T values(std::initializer_list<T> v) { return *v.begin(); }
        template <typename T> T range(T first, T) { return first; }
        } }
        using Catch::Generators::range;
        #define GENERATE(...) (__VA_ARGS__)
        """
    ).lstrip()
    (catch_dir / "generators" / "catch_generators.hpp").write_text(
        generators, encoding="utf-8"
    )
    (catch_dir / "generators" / "catch_generators_range.hpp").write_text(
        generators, encoding="utf-8"
    )
    (tmp_path / "cuda_fp16.h").write_text(
        textwrap.dedent(
            """
            #pragma once
            #include <cstdint>
            struct __half_raw { unsigned short x; };
            struct half {
              unsigned short x;
              half() = default;
              explicit half(unsigned short v) : x(v) {}
            };
            using __half = half;
            inline bool operator==(half a, half b) { return a.x == b.x; }
            inline bool operator!=(half a, half b) { return !(a == b); }
            inline float __half2float(half) { return 0.0f; }
            inline half __float2half(float) { return half{}; }
            inline half __float2half_rn(float) { return half{}; }
            """
        ).lstrip(),
        encoding="utf-8",
    )
    cuda_runtime = textwrap.dedent(
        """
        #pragma once
        #include <cstddef>
        #include <cstdint>
        #include "cuda_fp16.h"
        typedef int cudaError_t;
        static constexpr cudaError_t cudaSuccess = 0;
        typedef void* cudaStream_t;
        typedef void* cudaGraph_t;
        typedef void* cudaGraphExec_t;
        typedef void* cudaEvent_t;
        struct cudaGraphExecUpdateResultInfo { int result = 0; };
        enum cudaMemcpyKind {
          cudaMemcpyHostToHost = 0,
          cudaMemcpyHostToDevice = 1,
          cudaMemcpyDeviceToHost = 2,
          cudaMemcpyDeviceToDevice = 3,
          cudaMemcpyDefault = 4
        };
        struct cudaDeviceProp { int major = 10; int minor = 0; char name[256] = {}; };
        inline const char* cudaGetErrorString(cudaError_t) { return "cudaSuccess"; }
        inline cudaError_t cudaGetDevice(int* d) { if (d) *d = 0; return cudaSuccess; }
        inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
          if (p) { p->major = 10; p->minor = 0; }
          return cudaSuccess;
        }
        template <typename T> inline cudaError_t cudaMalloc(T** p, size_t) {
          if (p) *p = nullptr;
          return cudaSuccess;
        }
        template <typename T> inline cudaError_t cudaMallocAsync(T** p, size_t, cudaStream_t) {
          if (p) *p = nullptr;
          return cudaSuccess;
        }
        inline cudaError_t cudaFree(void*) { return cudaSuccess; }
        inline cudaError_t cudaFreeAsync(void*, cudaStream_t) { return cudaSuccess; }
        inline cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) {
          return cudaSuccess;
        }
        inline cudaError_t cudaMemcpyAsync(
            void*, const void*, size_t, cudaMemcpyKind, cudaStream_t = 0) {
          return cudaSuccess;
        }
        inline cudaError_t cudaMemset(void*, int, size_t) { return cudaSuccess; }
        inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t = 0) {
          return cudaSuccess;
        }
        inline cudaError_t cudaStreamCreate(cudaStream_t* s) {
          if (s) *s = nullptr;
          return cudaSuccess;
        }
        inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
        inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
        inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
        inline cudaError_t cudaEventCreate(cudaEvent_t* e) {
          if (e) *e = nullptr;
          return cudaSuccess;
        }
        inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
        inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = 0) {
          return cudaSuccess;
        }
        inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
        inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
          if (ms) *ms = 0.0f;
          return cudaSuccess;
        }
        inline cudaError_t cudaGraphCreate(cudaGraph_t* g, unsigned int) {
          if (g) *g = nullptr;
          return cudaSuccess;
        }
        inline cudaError_t cudaStreamBeginCapture(cudaStream_t, int) { return cudaSuccess; }
        inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t*) {
          return cudaSuccess;
        }
        inline cudaError_t cudaGraphInstantiate(
            cudaGraphExec_t*, cudaGraph_t, void*, void*, unsigned long long) {
          return cudaSuccess;
        }
        inline cudaError_t cudaGraphExecUpdate(
            cudaGraphExec_t, cudaGraph_t, cudaGraphExecUpdateResultInfo*) {
          return cudaSuccess;
        }
        inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) {
          return cudaSuccess;
        }
        inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return cudaSuccess; }
        inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }
        static constexpr int cudaStreamCaptureModeGlobal = 0;
        """
    ).lstrip()
    (tmp_path / "cuda_runtime.h").write_text(cuda_runtime, encoding="utf-8")
    (tmp_path / "cuda_runtime_api.h").write_text(cuda_runtime, encoding="utf-8")
    (tmp_path / "cublasLt.h").write_text(
        textwrap.dedent(
            """
            #pragma once
            typedef void* cublasLtHandle_t;
            typedef int cublasStatus_t;
            static constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS = 0;
            inline int cublasLtGetVersion() { return 130500; }
            inline cublasStatus_t cublasLtCreate(cublasLtHandle_t* h) {
              if (h) *h = nullptr;
              return CUBLAS_STATUS_SUCCESS;
            }
            inline cublasStatus_t cublasLtDestroy(cublasLtHandle_t) {
              return CUBLAS_STATUS_SUCCESS;
            }
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "cudnn_subquadratic_ops.h").write_text(
        textwrap.dedent(
            """
            #pragma once
            #include <mudnn_frontend.h>
            enum cudnnCausalConv1dActivation_t {
              CUDNN_CAUSAL_CONV1D_ACTIVATION_NONE = 0,
              CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU = 1
            };
            inline cudnnStatus_t cudnnCausalConv1dForward(...) {
              return CUDNN_STATUS_NOT_SUPPORTED;
            }
            inline cudnnStatus_t cudnnCausalConv1dBackward(...) {
              return CUDNN_STATUS_NOT_SUPPORTED;
            }
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_mudnn_frontend_upstream_cpp_samples_syntax(tmp_path):
    repo_root = _REPO_ROOT
    install_prefix = _installed_mudnn_prefix()
    upstream_samples = repo_root.parent / "mudnn-frontend" / "samples" / "cpp"
    thirdparty = (
        repo_root.parent
        / "mudnn-frontend"
        / "include"
        / "cudnn_frontend"
        / "thirdparty"
    )
    if not upstream_samples.exists():
        pytest.skip("upstream mudnn-frontend C++ samples are unavailable")
    if not (install_prefix / "include" / "mudnn_frontend.h").exists():
        pytest.skip("installed mudnn frontend headers are unavailable")

    _write_upstream_cpp_sample_syntax_shims(tmp_path)
    sources = sorted(upstream_samples.glob("**/*.cpp"))
    assert sources
    rewritten_root = tmp_path / "upstream_samples_rewritten"
    _rewrite_upstream_cpp_tree(upstream_samples, rewritten_root)
    for source in sources:
        rewritten_source = rewritten_root / source.relative_to(upstream_samples)
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-fsyntax-only",
                "-include",
                "memory",
                "-include",
                "mudnn_frontend.h",
                "-I",
                str(tmp_path),
                "-I",
                str(install_prefix / "include"),
                *_backend_include_flags(),
                "-I",
                str(rewritten_root),
                "-I",
                str(thirdparty),
                str(rewritten_source),
            ],
            check=True,
        )


def test_mudnn_frontend_legacy_split_headers_syntax(tmp_path):
    repo_root = _REPO_ROOT
    install_prefix = _installed_mudnn_prefix()
    header = install_prefix / "include" / "mudnn_frontend.h"
    split_header = install_prefix / "include" / "mudnn_frontend_find_plan.h"
    if not header.exists() or not split_header.exists():
        pytest.skip("installed frontend split headers are unavailable")

    source = tmp_path / "legacy_split_headers.cc"
    source.write_text(
        "\n".join(
            [
                "#include <mudnn_frontend.h>",
                "#include <mudnn_frontend_Filters.h>",
                "#include <mudnn_frontend_get_plan.h>",
                "#include <mudnn_frontend_find_plan.h>",
                "#include <mudnn_frontend_utils.h>",
                "",
                "int main() {",
                "  int64_t dims[4] = {1, 2, 3, 4};",
                "  int64_t strides[4] = {24, 12, 4, 1};",
                "  int64_t spatial[2] = {1, 1};",
                "  auto x = mudnn_frontend::TensorBuilder()",
                "      .setDim(4, dims)",
                "      .setStride(4, strides)",
                "      .setVectorCountAndDimension(1, 0)",
                "      .setReorderType(mudnn_frontend::TensorReordering_t::INT8x32)",
                "      .setDataType(CUDNN_DATA_FLOAT)",
                "      .setId(1)",
                "      .build();",
                "  auto conv = mudnn_frontend::ConvDescBuilder()",
                "      .setComputeType(CUDNN_DATA_FLOAT)",
                "      .setMathMode(CUDNN_CROSS_CORRELATION)",
                "      .setSpatialDimCount(2)",
                "      .setPrePadding(2, spatial)",
                "      .setPostPadding(2, spatial)",
                "      .setSpatialStride(2, spatial)",
                "      .setDilation(2, spatial)",
                "      .build();",
                "  mudnn_frontend::EngineConfigList configs;",
                "  mudnn_frontend::EngineConfigList filtered;",
                "  mudnn_frontend::filter(",
                "      configs, filtered,",
                "      [](cudnnBackendDescriptor_t) { return false; });",
                "  (void)mudnn_frontend::hasNumericalNote<",
                "      CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(nullptr);",
                "  auto device_props = std::make_shared<mudnn_frontend::DeviceProperties>();",
                "  std::vector<uint8_t> device_props_data;",
                "  if (!device_props->set_device_id(0).build().is_good()) return 3;",
                "  if (!device_props->serialize(device_props_data).is_good()) return 4;",
                "  auto restored_props = std::make_shared<mudnn_frontend::DeviceProperties>();",
                "  if (!restored_props->deserialize(device_props_data).is_good()) return 5;",
                "  mudnn_frontend::graph::Graph deviceless_graph;",
                "  deviceless_graph.set_device_properties(restored_props)",
                "      .set_io_data_type(mudnn_frontend::DataType_t::HALF)",
                "      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);",
                "  auto kernel_cache = std::make_shared<mudnn_frontend::KernelCache>();",
                "  deviceless_graph.set_dynamic_shape_enabled(true)",
                "      .set_kernel_cache(kernel_cache)",
                "      .select_behavior_notes({",
                "          mudnn_frontend::BehaviorNote_t::RUNTIME_COMPILATION});",
                "  auto notes = deviceless_graph.get_behavior_notes();",
                "  if (notes.size() != 1) return 6;",
                "  auto resample = mudnn_frontend::graph::Resample_attributes()",
                "      .set_padding_mode(mudnn_frontend::PaddingMode_t::NEG_INF_PAD);",
                "  if (mudnn_frontend::detail::get_backend_version() != CUDNN_VERSION) return 7;",
                "  if (mudnn_frontend::detail::get_element_size_in_bits(",
                "          mudnn_frontend::DataType_t::FP4_E2M1) != 4) return 8;",
                "  if (mudnn_frontend::detail::get_element_size_in_bits(",
                "          mudnn_frontend::DataType_t::UINT8) != 8) return 9;",
                "  auto unsupported =",
                "      mudnn_frontend::error_code_t::GRAPH_NOT_SUPPORTED;",
                "  (void)x;",
                "  (void)conv;",
                "  (void)resample;",
                "  (void)unsupported;",
                "  return 0;",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-fsyntax-only",
            "-I",
            str(install_prefix / "include"),
            *_backend_include_flags(),
            str(source),
        ],
        check=True,
    )


def _installed_header_name_for_upstream(upstream_rel):
    """Map an upstream mudnn-frontend include path to the installed mudnn name.

    Returns ``None`` for an upstream header this package deliberately does not
    ship.  The rule and the rename rule are the same one: after mapping, a name
    that still starts with ``cudnn`` is a cuDNN-owned filename with no muDNN
    counterpart, so it is not installed.  ``cudnn_backend_base.h`` and
    ``cudnn_frontend/cudnn_interface.h`` are the two such files; both hold cuDNN
    types that do not exist here, and shipping their names would collide with
    ``nvidia-cudnn-frontend`` in a shared prefix.
    """
    if upstream_rel == "cudnn_frontend.h":
        installed = "mudnn_frontend.h"
    elif upstream_rel.startswith("cudnn_frontend/"):
        installed = "mudnn_frontend/" + upstream_rel[len("cudnn_frontend/") :]
    elif upstream_rel.startswith("cudnn_frontend_"):
        installed = "mudnn_frontend_" + upstream_rel[len("cudnn_frontend_") :]
    else:
        installed = upstream_rel
    return None if Path(installed).name.startswith("cudnn") else installed


def test_mudnn_frontend_upstream_include_tree_headers_syntax(tmp_path):
    repo_root = _REPO_ROOT
    install_prefix = _installed_mudnn_prefix()
    upstream_include = repo_root.parent / "mudnn-frontend" / "include"
    installed_include = install_prefix / "include"
    if not upstream_include.exists():
        pytest.skip("upstream mudnn-frontend headers are unavailable")
    if not (installed_include / "mudnn_frontend.h").exists():
        pytest.skip("installed mudnn frontend headers are unavailable")

    installed_headers = []
    for path in sorted(upstream_include.rglob("*.h")):
        rel = path.relative_to(upstream_include)
        if rel.parts[0] == "cudnn_frontend" and rel.parts[1:2] in (
            ("generated",),
            ("thirdparty",),
        ):
            continue
        installed = _installed_header_name_for_upstream(rel.as_posix())
        if installed is not None:
            installed_headers.append(installed)
    assert installed_headers
    missing = [
        header
        for header in installed_headers
        if not (installed_include / header).exists()
    ]
    assert not missing

    source = tmp_path / "include_tree_headers.cc"
    source.write_text(
        "\n".join([f"#include <{header}>" for header in installed_headers])
        + "\n"
        + "\n".join(
            [
                "int main() {",
                '  static_assert(MUDNN_FRONTEND_MAJOR_VERSION >= 1, "major");',
                '  static_assert(MUDNN_FRONTEND_VERSION >= 10000, "version");',
                "  return mudnn_frontend::detail::get_backend_version() == CUDNN_VERSION ? 0 : 1;",
                "}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-fsyntax-only",
            "-I",
            str(installed_include),
            *_backend_include_flags(),
            str(source),
        ],
        check=True,
    )


def test_mudnn_frontend_upstream_cpp_tests_header_syntax(tmp_path):
    repo_root = _REPO_ROOT
    install_prefix = _installed_mudnn_prefix()
    upstream_cpp = repo_root.parent / "mudnn-frontend" / "test" / "cpp"
    if not upstream_cpp.exists():
        pytest.skip("upstream mudnn-frontend C++ tests are unavailable")
    if not (install_prefix / "include" / "mudnn_frontend.h").exists():
        pytest.skip("installed mudnn frontend headers are unavailable")

    catch_dir = tmp_path / "catch2"
    catch_dir.mkdir()
    (catch_dir / "catch_test_macros.hpp").write_text(
        "\n".join(
            [
                "#pragma once",
                "#define CATCH_CONCAT_INNER(a,b) a##b",
                "#define CATCH_CONCAT(a,b) CATCH_CONCAT_INNER(a,b)",
                "#define TEST_CASE(...) static void CATCH_CONCAT(catch_test_, __LINE__)()",
                "#define REQUIRE(...) do { (void)sizeof((__VA_ARGS__) ? 1 : 0); } while (0)",
                "#define CHECK(...) do { (void)sizeof((__VA_ARGS__) ? 1 : 0); } while (0)",
                "#define SKIP(...) do {} while (0)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cpp_files = sorted(upstream_cpp.glob("*.cpp"))
    assert cpp_files
    rewritten_dir = tmp_path / "upstream_rewritten"
    _rewrite_upstream_cpp_tree(upstream_cpp, rewritten_dir)
    for source in cpp_files:
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-fsyntax-only",
                "-I",
                str(tmp_path),
                "-I",
                str(install_prefix / "include"),
                *_backend_include_flags(),
                str(rewritten_dir / source.name),
            ],
            check=True,
        )


def test_mudnn_frontend_upstream_python_test_modules_import(tmp_path):
    repo_root = _REPO_ROOT
    frontend_python = _FRONTEND_ROOT / "python"
    upstream_tests = repo_root.parent / "mudnn-frontend" / "test" / "python"
    upstream_fe_api = upstream_tests / "fe_api"
    if not upstream_tests.exists():
        pytest.skip("upstream mudnn-frontend Python tests are unavailable")

    script = tmp_path / "import_upstream_tests.py"
    script.write_text(
        "\n".join(
            [
                "import importlib.util",
                "import pathlib",
                "import sys",
                "import traceback",
                "import types",
                "",
                "import torch",
                "",
                "root = pathlib.Path(sys.argv[1])",
                "props = types.SimpleNamespace(",
                "    major=10, minor=0, name='Fake SM100 for import sweep')",
                "torch.cuda.is_available = lambda: True",
                "torch.cuda.get_device_capability = lambda *args, **kwargs: (10, 0)",
                "torch.cuda.get_device_name = lambda *args, **kwargs: props.name",
                "torch.cuda.current_device = lambda: 0",
                "torch.cuda.get_device_properties = lambda *args, **kwargs: props",
                "",
                "files = sorted(",
                "    path for path in root.rglob('test_*.py')",
                "    if path.name != 'conftest.py')",
                "ok = 0",
                "skips = []",
                "failures = []",
                "for index, path in enumerate(files):",
                "    name = (",
                "        'upstream_import_sweep_' + str(index) + '_'",
                "        + str(path.relative_to(root))",
                "            .replace('/', '_')",
                "            .replace('.', '_')",
                "    )",
                "    try:",
                "        spec = importlib.util.spec_from_file_location(name, path)",
                "        module = importlib.util.module_from_spec(spec)",
                "        sys.modules[name] = module",
                "        spec.loader.exec_module(module)",
                "        ok += 1",
                "    except BaseException as exc:",
                "        message = ''.join(",
                "            traceback.format_exception_only(type(exc), exc)",
                "        ).strip()",
                "        if type(exc).__name__ == 'Skipped' or 'Skipped' in type(exc).__name__:",
                "            skips.append((str(path.relative_to(root)), message))",
                "        else:",
                "            failures.append((str(path.relative_to(root)), message))",
                "",
                "print(",
                "    f'UPSTREAM_IMPORT_SWEEP files={len(files)} ok={ok} '",
                "    f'skips={len(skips)} failures={len(failures)}'",
                ")",
                "if failures:",
                "    for path, message in failures[:40]:",
                "        print('FAIL', path, message)",
                "    raise SystemExit(1)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(frontend_python), str(upstream_tests), str(upstream_fe_api)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, str(script), str(upstream_tests)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    assert "failures=0" in result.stdout


def test_mudnn_frontend_upstream_cudnn_package_tree_imports(tmp_path):
    repo_root = _REPO_ROOT
    frontend_python = _FRONTEND_ROOT / "python"
    upstream_cudnn = repo_root.parent / "mudnn-frontend" / "python" / "mudnn"
    if not upstream_cudnn.exists():
        pytest.skip("upstream mudnn Python package tree is unavailable")

    script = tmp_path / "import_upstream_cudnn_tree.py"
    script.write_text(
        "\n".join(
            [
                "import importlib",
                "import pathlib",
                "import sys",
                "",
                "root = pathlib.Path(sys.argv[1])",
                "modules = []",
                "for path in sorted(root.rglob('*.py')):",
                "    rel = path.relative_to(root.parent).with_suffix('')",
                "    parts = rel.parts",
                "    if parts[-1] == '__init__':",
                "        parts = parts[:-1]",
                "    modules.append('.'.join(parts))",
                "",
                "failures = []",
                "for module_name in modules:",
                "    try:",
                "        importlib.import_module(module_name)",
                "    except Exception as exc:",
                "        failures.append(",
                "            (module_name, type(exc).__name__, str(exc)))",
                "",
                "print(",
                "    f'UPSTREAM_CUDNN_PACKAGE_SWEEP modules={len(modules)} '",
                "    f'failures={len(failures)}'",
                ")",
                "if failures:",
                "    for module_name, exc_type, message in failures[:40]:",
                "        print('FAIL', module_name, exc_type, message)",
                "    raise SystemExit(1)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(frontend_python)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, str(script), str(upstream_cudnn)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    assert "failures=0" in result.stdout


def test_mudnn_frontend_upstream_pytest_collection_with_fake_cuda(tmp_path):
    repo_root = _REPO_ROOT
    frontend_python = _FRONTEND_ROOT / "python"
    upstream_tests = repo_root.parent / "mudnn-frontend" / "test" / "python"
    upstream_fe_api = upstream_tests / "fe_api"
    if not upstream_tests.exists():
        pytest.skip("upstream mudnn-frontend Python tests are unavailable")

    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            [
                "import types",
                "try:",
                "    import torch",
                "    props = types.SimpleNamespace(",
                "        major=10, minor=0, name='Fake SM100 for collection')",
                "    class FakeStream:",
                "        def __init__(self, *args, **kwargs):",
                "            self.cuda_stream = 0",
                "        def __enter__(self):",
                "            return self",
                "        def __exit__(self, *args):",
                "            return False",
                "        def wait_stream(self, *args, **kwargs):",
                "            return None",
                "        def synchronize(self):",
                "            return None",
                "    torch.cuda.is_available = lambda: True",
                "    torch.cuda.get_device_capability = lambda *args, **kwargs: (10, 0)",
                "    torch.cuda.get_device_name = lambda *args, **kwargs: props.name",
                "    torch.cuda.current_device = lambda: 0",
                "    torch.cuda.get_device_properties = lambda *args, **kwargs: props",
                "    torch.cuda.device_count = lambda: 1",
                "    torch.cuda.synchronize = lambda *args, **kwargs: None",
                "    torch.cuda.Stream = FakeStream",
                "    torch.cuda.current_stream = lambda *args, **kwargs: FakeStream()",
                "except Exception:",
                "    pass",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(frontend_python), str(upstream_tests), str(upstream_fe_api)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", str(upstream_tests)],
        text=True,
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests collected" in result.stdout


def test_mudnn_frontend_upstream_python_graph_api_signatures():
    import ast
    import inspect

    import mudnn
    from mudnn_frontend.graph import Graph

    repo_root = _REPO_ROOT
    upstream_tests = repo_root.parent / "mudnn-frontend" / "test" / "python"
    if not upstream_tests.exists():
        pytest.skip("upstream mudnn-frontend Python tests are unavailable")

    graph_sigs = {
        name: inspect.signature(obj)
        for name, obj in inspect.getmembers(Graph)
        if callable(obj) and not name.startswith("_")
    }
    cudnn_attrs = set(dir(mudnn))
    graph_names = {"graph", "g", "lower_graph", "mha_graph"}
    missing = []
    bad_keywords = []

    def check_keywords(owner, method_name, signature, keywords, filename):
        params = signature.parameters
        has_varkw = any(param.kind == param.VAR_KEYWORD for param in params.values())
        accepted = {
            name
            for name, param in params.items()
            if name != "self"
            and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
        if has_varkw:
            return
        for keyword in keywords:
            if keyword not in accepted:
                bad_keywords.append(
                    f"{filename}: {owner}.{method_name} got unsupported keyword {keyword}"
                )

    for path in sorted(upstream_tests.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            value = node.func.value
            keywords = [kw.arg for kw in node.keywords if kw.arg is not None]
            if isinstance(value, ast.Name) and value.id in graph_names:
                signature = graph_sigs.get(node.func.attr)
                if signature is None:
                    missing.append(f"{path.name}: Graph.{node.func.attr}")
                else:
                    check_keywords(
                        "Graph", node.func.attr, signature, keywords, path.name
                    )
            elif isinstance(value, ast.Name) and value.id == "mudnn":
                if node.func.attr not in cudnn_attrs:
                    missing.append(f"{path.name}: mudnn.{node.func.attr}")
                    continue
                target = getattr(mudnn, node.func.attr)
                if callable(target):
                    try:
                        signature = inspect.signature(target)
                    except (TypeError, ValueError):
                        continue
                    check_keywords(
                        "mudnn", node.func.attr, signature, keywords, path.name
                    )

    assert not missing
    assert not bad_keywords


def _write_upstream_cpu_remap_sitecustomize(path: Path):
    path.write_text(
        textwrap.dedent(
            """
            import torch
            import torch.nn.functional as F
            import numpy as _np

            if not hasattr(torch, "float8_e8m0fnu"):
                torch.float8_e8m0fnu = torch.uint8

            class FakeStream:
                cuda_stream = 0

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def wait_stream(self, *args, **kwargs):
                    return None

                def synchronize(self):
                    return None

            class FakeCUDAGraph:
                def replay(self):
                    return None

            class FakeCudaGraphContext:
                def __init__(self, graph):
                    self.graph = graph

                def __enter__(self):
                    return self.graph

                def __exit__(self, *args):
                    return False

            class FakeNvtx:
                @staticmethod
                def range_push(*args, **kwargs):
                    return None

                @staticmethod
                def range_pop(*args, **kwargs):
                    return None

            def _cpu_device(kwargs):
                device = kwargs.get("device")
                if device == "cuda" or str(device).startswith("cuda:"):
                    kwargs = dict(kwargs)
                    kwargs["device"] = "cpu"
                return kwargs

            def _wrap_factory(fn):
                def inner(*args, **kwargs):
                    return fn(*args, **_cpu_device(kwargs))

                return inner

            for _name in [
                "randn",
                "randn_like",
                "empty",
                "empty_like",
                "zeros",
                "zeros_like",
                "ones",
                "ones_like",
                "full",
                "rand",
                "randint",
                "randperm",
                "tensor",
                "arange",
                "linspace",
                "normal",
            ]:
                if hasattr(torch, _name):
                    setattr(torch, _name, _wrap_factory(getattr(torch, _name)))

            _orig_generator = torch.Generator

            def _generator(*args, **kwargs):
                return _orig_generator(*args, **_cpu_device(kwargs))

            torch.Generator = _generator

            _orig_tensor_to = torch.Tensor.to

            def _tensor_cuda(self, *args, **kwargs):
                return self

            def _tensor_to(self, *args, **kwargs):
                args = tuple(
                    "cpu" if arg == "cuda" or str(arg).startswith("cuda:") else arg
                    for arg in args
                )
                return _orig_tensor_to(self, *args, **_cpu_device(kwargs))

            def _module_to(self, *args, **kwargs):
                return self

            torch.Tensor.cuda = _tensor_cuda
            torch.Tensor.to = _tensor_to
            torch.nn.Module.to = _module_to

            F._mudnn_cpu_remap_orig_conv2d = F.conv2d

            def _conv2d(input, weight, *args, **kwargs):
                if (
                    getattr(input, "device", None) is not None
                    and input.device.type == "cpu"
                    and input.dtype == torch.int8
                ):
                    raise RuntimeError("Fake CPU CUDA does not expose INT8 conv2d")
                return F._mudnn_cpu_remap_orig_conv2d(input, weight, *args, **kwargs)

            F.conv2d = _conv2d

            _orig_mse_loss = F.mse_loss

            def _mse_loss(input, target, *args, **kwargs):
                if (
                    getattr(input, "device", None) is not None
                    and input.device.type == "cpu"
                    and input.dtype in (torch.float16, torch.bfloat16)
                ):
                    input = input.float()
                if (
                    getattr(target, "device", None) is not None
                    and target.device.type == "cpu"
                    and target.dtype in (torch.float16, torch.bfloat16)
                ):
                    target = target.float()
                return _orig_mse_loss(input, target, *args, **kwargs)

            F.mse_loss = _mse_loss

            _orig_batch_norm = F.batch_norm

            def _bn_1d(tensor):
                if (
                    getattr(tensor, "device", None) is not None
                    and tensor.device.type == "cpu"
                    and tensor.dim() == 4
                    and tensor.shape[0] == 1
                    and tensor.shape[2] == 1
                    and tensor.shape[3] == 1
                ):
                    return tensor.reshape(tensor.shape[1])
                return tensor

            def _copy_bn_stat_back(original, reshaped):
                if original is not reshaped and original is not None:
                    original.reshape(-1).copy_(reshaped.reshape(-1))

            def _batch_norm(
                input,
                running_mean,
                running_var,
                weight=None,
                bias=None,
                training=False,
                momentum=0.1,
                eps=1e-5,
            ):
                running_mean_1d = _bn_1d(running_mean)
                running_var_1d = _bn_1d(running_var)
                result = _orig_batch_norm(
                    input,
                    running_mean_1d,
                    running_var_1d,
                    weight=_bn_1d(weight),
                    bias=_bn_1d(bias),
                    training=training,
                    momentum=momentum,
                    eps=eps,
                )
                if training:
                    _copy_bn_stat_back(running_mean, running_mean_1d)
                    _copy_bn_stat_back(running_var, running_var_1d)
                return result

            F.batch_norm = _batch_norm

            if not hasattr(torch, "hash_tensor"):

                def _hash_tensor(tensor):
                    host = tensor.detach().cpu().contiguous()
                    if host.dtype == torch.bfloat16:
                        host = host.view(torch.int16)
                    data = host.numpy().view(_np.uint8).reshape(-1)
                    hash_value = int(data.sum(dtype=_np.uint64) % _np.uint64(2**31 - 1))
                    return torch.tensor(hash_value, dtype=torch.int64)

                torch.hash_tensor = _hash_tensor

            props = type(
                "Props",
                (),
                {
                    "major": 10,
                    "minor": 0,
                    "name": "Fake CPU CUDA",
                    "multi_processor_count": 1,
                },
            )()
            torch.cuda.is_available = lambda: True
            torch.cuda.device_count = lambda: 1
            torch.cuda.current_device = lambda: 0
            torch.cuda.get_device_capability = lambda *args, **kwargs: (10, 0)
            torch.cuda.get_device_name = lambda *args, **kwargs: props.name
            torch.cuda.get_device_properties = lambda *args, **kwargs: props
            torch.cuda.current_stream = lambda *args, **kwargs: FakeStream()
            torch.cuda.Stream = FakeStream
            torch.cuda.stream = lambda stream: stream
            torch.cuda.CUDAGraph = FakeCUDAGraph
            torch.cuda.graph = lambda graph, *args, **kwargs: FakeCudaGraphContext(graph)
            torch.cuda.nvtx = FakeNvtx
            torch.cuda.synchronize = lambda *args, **kwargs: None
            torch.cuda.empty_cache = lambda *args, **kwargs: None
            torch.cuda.mem_get_info = lambda *args, **kwargs: (8 << 30, 8 << 30)
            torch.cuda.get_rng_state = lambda *args, **kwargs: torch.get_rng_state()
            torch.cuda.set_rng_state = (
                lambda state, *args, **kwargs: torch.set_rng_state(state)
            )
            torch.cuda.manual_seed = lambda seed: None
            torch.cuda.manual_seed_all = lambda seed: None

            import sys as _sys
            import types as _types

            _cuda_module = _types.ModuleType("cuda")
            _bindings_module = _types.ModuleType("cuda.bindings")
            _driver_module = _types.ModuleType("cuda.bindings.driver")

            class CUstream(int):
                def __new__(cls, value=0):
                    return int.__new__(
                        cls, int(value) if value is not None else 0
                    )

            _driver_module.CUstream = CUstream
            _bindings_module.driver = _driver_module
            _cuda_module.bindings = _bindings_module
            _sys.modules.setdefault("cuda", _cuda_module)
            _sys.modules.setdefault("cuda.bindings", _bindings_module)
            _sys.modules.setdefault("cuda.bindings.driver", _driver_module)
            """
        ).lstrip(),
        encoding="utf-8",
    )


def test_mudnn_frontend_upstream_python_cpu_remap_execution_smoke(tmp_path):
    repo_root = _REPO_ROOT
    frontend_python = _FRONTEND_ROOT / "python"
    upstream_tests = repo_root.parent / "mudnn-frontend" / "test" / "python"
    upstream_fe_api = upstream_tests / "fe_api"
    if not upstream_tests.exists():
        pytest.skip("upstream mudnn-frontend Python tests are unavailable")

    _write_upstream_cpu_remap_sitecustomize(tmp_path / "sitecustomize.py")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDNN_FE_NORM_FUZZ_L0_TESTS"] = "1"
    env["CUDNN_FE_NORM_FUZZ_L0_SEED"] = "31415"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(frontend_python), str(upstream_tests), str(upstream_fe_api)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    targets = [
        str(upstream_tests / "test_apply_rope.py"),
        str(upstream_tests / "test_batchnorm.py"),
        str(upstream_tests / "test_block_scale_quantize.py"),
        str(upstream_tests / "test_block_scale_quantize_dynamic_shape.py"),
        str(upstream_tests / "test_slice.py"),
        str(upstream_tests / "test_conv_fprop.py"),
        str(upstream_tests / "test_conv_bias.py"),
        str(upstream_tests / "test_conv_fuzzer.py")
        + "::test_conv_random_L0[t21_N1_C16x32_40x64_f32_2d_fp_none]",
        str(upstream_tests / "test_conv_fuzzer.py")
        + "::test_conv_random_L0[t426_N1_C16x104_16x8x16_f32_3d_fp_none]",
        str(upstream_tests / "test_conv_fuzzer.py")
        + "::test_conv_random_L0[t99_N1_C8x16_g8_8x40_f16_2d_fp_none]",
        str(upstream_tests / "test_conv_fuzzer.py")
        + "::test_conv_random_L0[t6_N1_C8x144_g2_16x64_f16_2d_wg_none]",
        str(upstream_tests / "test_conv_fuzzer.py")
        + "::test_conv_random_L0[t160_N1_C8x16_40x88_f32_2d_dg_none]",
        str(upstream_tests / "test_conv_fuzzer.py")
        + "::test_conv_random_L0[t84_N1_C88x8_32x8_f32_2d_wg_none]",
        str(upstream_tests / "test_conv_fuzzer.py")
        + "::test_conv_depthwise_L0[dw13_N1_C8x8_g8_16x56_f16_2d_fp_none]",
        str(upstream_tests / "test_conv_genstats.py"),
        str(upstream_tests / "test_conv_reduction.py"),
        str(upstream_tests / "test_deviceless_aot_compilation.py"),
        str(upstream_tests / "test_cudnn_sdpa_op.py"),
        str(upstream_tests / "test_flexible_sdpa_bprop.py"),
        str(upstream_tests / "test_flexible_sdpa.py"),
        str(upstream_tests / "test_sdpa_thd.py"),
        str(upstream_tests / "test_sdpa_chunked_prefill.py"),
        str(upstream_tests / "test_sdpa_with_caching.py"),
        str(upstream_tests / "test_mhas.py")
        + "::test_sdpa[torch.float16-bshd_bshd_bshd-multi_head-paged0-bias1-alibi0-padding0-diagonal_alignment_0-left_bound0-right_bound0-dropout0-ragged0-infer0]",
        str(upstream_tests / "test_mhas.py")
        + "::test_sdpa[torch.float16-bshd_bshd_bshd-multi_head-paged0-bias0-alibi1-padding0-diagonal_alignment_0-left_bound0-right_bound0-dropout0-ragged0-infer0]",
        str(upstream_tests / "test_mhas.py")
        + "::test_sdpa[torch.float16-bshd_bshd_bshd-multi_head-paged0-bias0-alibi0-padding1-diagonal_alignment_0-left_bound0-right_bound0-dropout0-ragged0-infer0]",
        str(upstream_tests / "test_mhas.py")
        + "::test_sdpa[torch.float16-bshd_bshd_bshd-multi_head-paged0-bias1-alibi1-padding1-diagonal_alignment_0-left_bound0-right_bound0-dropout0-ragged0-infer0]",
        str(upstream_tests / "test_mhas_v2.py") + "::test_sdpa_random_fwd_L0[test1]",
        str(upstream_tests / "test_mhas_v2.py") + "::test_sdpa_random_bwd_L0[test3]",
        str(upstream_tests / "test_mhas_v2.py") + "::test_sdpa_random_bwd_L0[test4]",
        str(upstream_tests / "test_mhas_v2.py") + "::test_sdpa_random_bwd_L0[test6]",
        str(upstream_tests / "test_instancenorm.py"),
        str(upstream_tests / "test_kernel_cache.py"),
        str(upstream_tests / "test_layernorm.py"),
        str(upstream_tests / "test_low_precision_matmul.py"),
        str(upstream_tests / "test_matmul_bias_relu.py"),
        str(upstream_tests / "test_matmul_fuzzer.py")
        + "::test_matmul_fuzz[t99_b1_M296xN8xK104_f32_none]",
        str(upstream_tests / "test_moe_grouped_matmul.py"),
        str(upstream_tests / "test_moe_grouped_matmul_op.py"),
        str(upstream_tests / "test_norm_fuzzer.py"),
        str(upstream_tests / "test_oss_rope.py"),
        str(upstream_tests / "test_rmsnorm.py"),
        str(upstream_tests / "test_silu_and_mul.py"),
        str(upstream_tests / "test_wgrads.py") + "::test_scale_bias_relu_wgrad",
        str(upstream_tests / "test_yarn_rope.py"),
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *targets,
            "-k",
            "not TorchCompile",
            "--tb=short",
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout


def test_mudnn_frontend_upstream_python_sm100_non_l0_cpu_remap_smoke(tmp_path):
    repo_root = _REPO_ROOT
    frontend_python = _FRONTEND_ROOT / "python"
    upstream_tests = repo_root.parent / "mudnn-frontend" / "test" / "python"
    upstream_fe_api = upstream_tests / "fe_api"
    if not upstream_tests.exists():
        pytest.skip("upstream mudnn-frontend Python tests are unavailable")

    _write_upstream_cpu_remap_sitecustomize(tmp_path / "sitecustomize.py")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(frontend_python), str(upstream_tests), str(upstream_fe_api)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    targets = [
        str(upstream_tests / "test_sm100_prefill_oss_engine.py")
        + "::test_sm100_prefill_oss_engine_vs_cudnn_graph",
        str(upstream_tests / "test_sm100_rms_norm_silu_graph_api.py")
        + "::test_bf16_full_sweep[1560-64]",
        str(upstream_tests / "test_sm100_rms_norm_silu_graph_api.py")
        + "::test_bf16_smoke",
        str(upstream_tests / "test_sm100_rms_norm_silu_graph_api.py")
        + "::test_unsupported_problem_sizes[48-1560]",
        str(upstream_tests / "test_sm100_rms_norm_silu_graph_api.py")
        + "::test_epsilon_equivalence[64]",
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *targets,
            "-m",
            "",
            "--tb=short",
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout


def test_mudnn_frontend_upstream_fe_api_cpu_remap_execution_smoke(tmp_path):
    repo_root = _REPO_ROOT
    frontend_python = _FRONTEND_ROOT / "python"
    upstream_tests = repo_root.parent / "mudnn-frontend" / "test" / "python"
    upstream_fe_api = upstream_tests / "fe_api"
    if not upstream_fe_api.exists():
        pytest.skip("upstream mudnn-frontend fe_api tests are unavailable")

    _write_upstream_cpu_remap_sitecustomize(tmp_path / "sitecustomize.py")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(frontend_python), str(upstream_tests), str(upstream_fe_api)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(upstream_fe_api),
            "--tb=short",
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout


def test_mudnn_frontend_cudnn_gemm_srelu_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    a = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 8.0 - 1.0
    b = torch.arange(40, dtype=torch.float32).reshape(5, 4, 2) / 10.0 - 1.5
    sfa = torch.full_like(a, 0.5)
    sfb = torch.full_like(b, 0.25)
    prob = torch.linspace(0.25, 1.0, steps=6, dtype=torch.float32).reshape(3, 1, 2)

    expected_c = torch.einsum("mkl,nkl->mnl", a * sfa, b * sfb)
    expected_d = torch.relu(expected_c) ** 2 * prob.expand_as(expected_c)

    outputs = mudnn.gemm_srelu_wrapper_sm100(
        a,
        b,
        sfa,
        sfb,
        prob,
        c_dtype=torch.float32,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    c_tensor, d_tensor, amax_tensor, sfd_tensor, norm_const_tensor = outputs

    assert outputs["c_tensor"] is c_tensor
    assert outputs["d_tensor"] is d_tensor
    assert outputs["amax_tensor"] is amax_tensor
    assert outputs["sfd_tensor"] is sfd_tensor
    assert outputs["norm_const_tensor"] is norm_const_tensor
    assert outputs[0] is c_tensor
    assert outputs[1] is d_tensor
    assert c_tensor.stride() == (5, 1, 15)
    assert d_tensor.stride() == (5, 1, 15)
    torch.testing.assert_close(c_tensor, expected_c)
    torch.testing.assert_close(d_tensor, expected_d)
    torch.testing.assert_close(
        amax_tensor, torch.amax(torch.abs(expected_d)).reshape(1)
    )

    m_major = mudnn.gemm_srelu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        prob_tensor=prob,
        c_major="m",
        c_dtype=torch.float32,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    assert m_major["c_tensor"].stride() == (1, 3, 15)
    assert m_major["d_tensor"].stride() == (1, 3, 15)
    torch.testing.assert_close(m_major["c_tensor"], expected_c)
    torch.testing.assert_close(m_major["d_tensor"], expected_d)
    with pytest.raises(ValueError, match="c_major"):
        mudnn.gemm_srelu_wrapper_sm100(
            a,
            b,
            sfa,
            sfb,
            prob,
            c_major="bad",
        )

    c = torch.empty_like(expected_c, dtype=torch.bfloat16)
    d = torch.empty_like(expected_d, dtype=torch.bfloat16)
    amax = torch.empty((1,), dtype=torch.float32)
    op = mudnn.GemmSreluSm100(sample_a=a, sample_b=b)
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        prob_tensor=prob,
        c_tensor=c,
        d_tensor=d,
        amax_tensor=amax,
    )
    torch.testing.assert_close(c.float(), expected_c.to(torch.bfloat16).float())
    torch.testing.assert_close(d.float(), expected_d.to(torch.bfloat16).float())
    torch.testing.assert_close(amax, torch.amax(torch.abs(expected_d)).reshape(1))


def test_mudnn_frontend_cudnn_grouped_gemm_srelu_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    m, k, n, experts = 5, 3, 4, 2
    a = torch.arange(m * k, dtype=torch.float32).reshape(m, k, 1) / 7.0 - 1.0
    b = torch.arange(n * k * experts, dtype=torch.float32).reshape(n, k, experts)
    b = b / 11.0 - 1.25
    sfa = torch.linspace(0.25, 0.75, steps=m * k, dtype=torch.float32).reshape(m, k, 1)
    sfb = torch.linspace(0.5, 1.0, steps=n * k * experts, dtype=torch.float32).reshape(
        n, k, experts
    )
    alpha = torch.tensor([0.5, 1.25], dtype=torch.float32)
    bias = torch.linspace(-0.2, 0.3, steps=n * experts, dtype=torch.float32).reshape(
        n, experts
    )
    prob = torch.linspace(0.25, 1.0, steps=m, dtype=torch.float32).reshape(m, 1, 1)
    padded_offsets = torch.tensor([2, 5], dtype=torch.int32)

    expected_c = torch.empty((m, n, 1), dtype=torch.float32)
    start = 0
    for expert, end in enumerate(padded_offsets.tolist()):
        expected_c[start:end, :, 0] = torch.einsum(
            "mk,nk->mn",
            a[start:end, :, 0] * sfa[start:end, :, 0],
            b[:, :, expert] * sfb[:, :, expert],
        )
        expected_c[start:end, :, 0] *= alpha[expert]
        expected_c[start:end, :, 0] += bias[:, expert].unsqueeze(0)
        start = end
    expected_d = torch.relu(expected_c) ** 2 * prob.expand_as(expected_c)
    expected_amax = torch.stack(
        [
            torch.amax(torch.abs(expected_d[:2, :, 0])),
            torch.amax(torch.abs(expected_d[2:, :, 0])),
        ]
    ).reshape(experts, 1)

    outputs = mudnn.grouped_gemm_srelu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        bias_tensor=bias,
        prob_tensor=prob,
        c_dtype=torch.float32,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
    )

    torch.testing.assert_close(outputs["c_tensor"], expected_c)
    torch.testing.assert_close(outputs["d_tensor"], expected_d)
    torch.testing.assert_close(outputs["amax_tensor"], expected_amax)

    c = torch.empty_like(expected_c, dtype=torch.bfloat16)
    d = torch.empty_like(expected_d, dtype=torch.bfloat16)
    amax = torch.empty((experts, 1), dtype=torch.float32)
    op = mudnn.GroupedGemmSreluSm100(sample_a=a, sample_b=b)
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        bias_tensor=bias,
        prob_tensor=prob,
        c_tensor=c,
        d_tensor=d,
        amax_tensor=amax,
    )
    torch.testing.assert_close(c.float(), expected_c.to(torch.bfloat16).float())
    torch.testing.assert_close(d.float(), expected_d.to(torch.bfloat16).float())
    torch.testing.assert_close(amax, expected_amax)


def test_mudnn_frontend_cudnn_grouped_gemm_swiglu_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    m, k, n, experts = 5, 3, 64, 2
    a = torch.arange(m * k, dtype=torch.float32).reshape(m, k, 1) / 17.0 - 0.5
    b = torch.arange(n * k * experts, dtype=torch.float32).reshape(n, k, experts)
    b = b / 97.0 - 1.25
    sfa = torch.linspace(0.25, 0.85, steps=m * k, dtype=torch.float32).reshape(m, k, 1)
    sfb = torch.linspace(0.5, 1.0, steps=n * k * experts, dtype=torch.float32).reshape(
        n, k, experts
    )
    alpha = torch.tensor([0.75, 1.25], dtype=torch.float32)
    prob = torch.linspace(0.2, 0.9, steps=m, dtype=torch.float32).reshape(m, 1, 1)
    padded_offsets = torch.tensor([2, 5], dtype=torch.int32)

    expected_c = torch.empty((m, n, 1), dtype=torch.float32)
    start = 0
    for expert, end in enumerate(padded_offsets.tolist()):
        expected_c[start:end, :, 0] = torch.einsum(
            "mk,nk->mn",
            a[start:end, :, 0] * sfa[start:end, :, 0],
            b[:, :, expert] * sfb[:, :, expert],
        )
        expected_c[start:end, :, 0] *= alpha[expert]
        start = end

    cols = torch.arange(n, dtype=torch.long)
    block_cols = cols.view(n // 32, 32)
    gate_idx = block_cols[0::2].reshape(-1)
    up_idx = block_cols[1::2].reshape(-1)
    gate = expected_c.index_select(1, gate_idx)
    up = expected_c.index_select(1, up_idx)
    expected_d = up * (gate * torch.sigmoid(gate)) * prob.expand(m, n // 2, 1)
    expected_amax = torch.stack(
        [
            torch.amax(torch.abs(expected_d[:2, :, 0])),
            torch.amax(torch.abs(expected_d[2:, :, 0])),
        ]
    ).reshape(experts, 1)

    outputs = mudnn.grouped_gemm_swiglu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        prob_tensor=prob,
        c_dtype=torch.float32,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
    )

    torch.testing.assert_close(outputs["c_tensor"], expected_c)
    torch.testing.assert_close(outputs["d_tensor"], expected_d)
    torch.testing.assert_close(outputs["amax_tensor"], expected_amax)

    c = torch.empty_like(expected_c, dtype=torch.bfloat16)
    d = torch.empty_like(expected_d, dtype=torch.bfloat16)
    amax = torch.empty((experts, 1), dtype=torch.float32)
    op = mudnn.GroupedGemmSwigluSm100(sample_a=a, sample_b=b)
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        prob_tensor=prob,
        c_tensor=c,
        d_tensor=d,
        amax_tensor=amax,
    )
    torch.testing.assert_close(c.float(), expected_c.to(torch.bfloat16).float())
    torch.testing.assert_close(d.float(), expected_d.to(torch.bfloat16).float())
    torch.testing.assert_close(amax, expected_amax)


def test_mudnn_frontend_cudnn_grouped_gemm_glu_geglu_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    m, k, n, experts = 5, 3, 64, 2
    a = torch.arange(m * k, dtype=torch.float32).reshape(m, k, 1) / 23.0 - 0.5
    b = torch.arange(n * k * experts, dtype=torch.float32).reshape(n, k, experts)
    b = b / 109.0 - 1.0
    sfa = torch.linspace(0.2, 0.7, steps=m * k, dtype=torch.float32).reshape(m, k, 1)
    sfb = torch.linspace(
        0.35, 1.05, steps=n * k * experts, dtype=torch.float32
    ).reshape(n, k, experts)
    alpha = torch.tensor([0.8, 1.3], dtype=torch.float32)
    bias = torch.linspace(-0.2, 0.3, steps=n * experts, dtype=torch.float32).reshape(
        n, experts
    )
    prob = torch.linspace(0.25, 0.95, steps=m, dtype=torch.float32).reshape(m, 1, 1)
    padded_offsets = torch.tensor([2, 5], dtype=torch.int32)
    linear_offset = 0.5
    geglu_alpha = 2.0
    clamp_max = 0.75
    clamp_min = -0.5

    expected_c = torch.empty((m, n, 1), dtype=torch.float32)
    start = 0
    for expert, end in enumerate(padded_offsets.tolist()):
        expected_c[start:end, :, 0] = torch.einsum(
            "mk,nk->mn",
            a[start:end, :, 0] * sfa[start:end, :, 0],
            b[:, :, expert] * sfb[:, :, expert],
        )
        expected_c[start:end, :, 0] *= alpha[expert]
        expected_c[start:end, :, 0] += bias[:, expert].unsqueeze(0)
        start = end

    cols = torch.arange(n, dtype=torch.long)
    block_cols = cols.view(n // 32, 32)
    gate_idx = block_cols[0::2].reshape(-1)
    up_idx = block_cols[1::2].reshape(-1)
    gate = expected_c.index_select(1, gate_idx)
    up = expected_c.index_select(1, up_idx)
    gate_clamped = torch.clamp(gate, max=clamp_max)
    up_clamped = torch.clamp(up, min=clamp_min, max=clamp_max)
    expected_d = (up_clamped + linear_offset) * (
        gate_clamped * torch.sigmoid(geglu_alpha * gate_clamped)
    )
    expected_d = expected_d * prob.expand(m, n // 2, 1)
    expected_amax = torch.stack(
        [
            torch.amax(torch.abs(expected_d[:2, :, 0])),
            torch.amax(torch.abs(expected_d[2:, :, 0])),
        ]
    ).reshape(experts, 1)

    outputs = mudnn.grouped_gemm_glu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        bias_tensor=bias,
        prob_tensor=prob,
        c_dtype=torch.float32,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
        act_func="geglu",
        linear_offset=linear_offset,
        geglu_alpha=geglu_alpha,
        glu_clamp_max=clamp_max,
        glu_clamp_min=clamp_min,
    )

    torch.testing.assert_close(outputs["c_tensor"], expected_c)
    torch.testing.assert_close(outputs["d_tensor"], expected_d)
    torch.testing.assert_close(outputs["amax_tensor"], expected_amax)

    c = torch.empty_like(expected_c, dtype=torch.bfloat16)
    d = torch.empty_like(expected_d, dtype=torch.bfloat16)
    amax = torch.empty((experts, 1), dtype=torch.float32)
    op = mudnn.GroupedGemmGluSm100(sample_a=a, sample_b=b, act_func="geglu")
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        bias_tensor=bias,
        prob_tensor=prob,
        c_tensor=c,
        d_tensor=d,
        amax_tensor=amax,
        act_func="geglu",
        linear_offset=linear_offset,
        geglu_alpha=geglu_alpha,
        glu_clamp_max=clamp_max,
        glu_clamp_min=clamp_min,
    )
    torch.testing.assert_close(c.float(), expected_c.to(torch.bfloat16).float())
    torch.testing.assert_close(d.float(), expected_d.to(torch.bfloat16).float())
    torch.testing.assert_close(amax, expected_amax)


def test_mudnn_frontend_cudnn_grouped_gemm_glu_hadamard_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn
    from mudnn.grouped_gemm.grouped_gemm_glu_hadamard.hadamard_utils import (
        HADAMARD_SIZE,
        hadamard_matrix,
    )

    assert HADAMARD_SIZE == 16
    h16 = hadamard_matrix(HADAMARD_SIZE, dtype=torch.float32)
    assert h16.shape == (16, 16)
    torch.testing.assert_close(h16 @ h16.T, torch.eye(16) * 16)
    with pytest.raises(ValueError, match="positive"):
        hadamard_matrix(0)
    with pytest.raises(ValueError, match="power of 2"):
        hadamard_matrix(12)

    m, k, n, experts = 32, 3, 64, 2
    a = torch.arange(m * k, dtype=torch.float32).reshape(m, k, 1) / 37.0 - 0.5
    b = torch.arange(n * k * experts, dtype=torch.float32).reshape(n, k, experts)
    b = b / 131.0 - 1.0
    sfa = torch.linspace(0.4, 0.9, steps=m * k, dtype=torch.float32).reshape(m, k, 1)
    sfb = torch.linspace(0.5, 1.1, steps=n * k * experts, dtype=torch.float32).reshape(
        n, k, experts
    )
    alpha = torch.tensor([0.75, 1.25], dtype=torch.float32)
    bias = torch.linspace(-0.15, 0.25, steps=n * experts, dtype=torch.float32).reshape(
        n, experts
    )
    prob = torch.linspace(0.35, 0.95, steps=m, dtype=torch.float32).reshape(m, 1, 1)
    padded_offsets = torch.tensor([16, 32], dtype=torch.int32)

    expected_c = torch.empty((m, n, 1), dtype=torch.float32)
    start = 0
    for expert, end in enumerate(padded_offsets.tolist()):
        expected_c[start:end, :, 0] = torch.einsum(
            "mk,nk->mn",
            a[start:end, :, 0] * sfa[start:end, :, 0],
            b[:, :, expert] * sfb[:, :, expert],
        )
        expected_c[start:end, :, 0] *= alpha[expert]
        expected_c[start:end, :, 0] += bias[:, expert].unsqueeze(0)
        start = end

    cols = torch.arange(n, dtype=torch.long)
    block_cols = cols.view(n // 32, 32)
    gate_idx = block_cols[0::2].reshape(-1)
    up_idx = block_cols[1::2].reshape(-1)
    gate = expected_c.index_select(1, gate_idx)
    up = expected_c.index_select(1, up_idx)
    expected_d = up * (gate * torch.sigmoid(gate)) * prob.expand(m, n // 2, 1)
    expected_amax = torch.stack(
        [
            torch.amax(torch.abs(expected_d[:16, :, 0])),
            torch.amax(torch.abs(expected_d[16:, :, 0])),
        ]
    ).reshape(experts, 1)

    expected_post_amax = torch.empty((experts, 1), dtype=torch.float32)
    start = 0
    for expert, end in enumerate(padded_offsets.tolist()):
        group_m = end - start
        group_t = expected_d[start:end, :, 0].to(torch.bfloat16).float().T.contiguous()
        group_t = group_t.view(n // 2, group_m // 16, 16)
        transformed = (group_t @ h16) * (16**-0.5)
        expected_post_amax[expert, 0] = torch.amax(torch.abs(transformed))
        start = end

    outputs = mudnn.grouped_gemm_glu_hadamard_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        bias_tensor=bias,
        prob_tensor=prob,
        c_dtype=torch.float32,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
        act_func="swiglu",
        use_tmem_post_rht_amax=True,
    )

    torch.testing.assert_close(outputs["c_tensor"], expected_c)
    torch.testing.assert_close(outputs["d_tensor"], expected_d)
    torch.testing.assert_close(outputs["amax_tensor"], expected_amax)
    torch.testing.assert_close(outputs["post_rht_amax_tensor"], expected_post_amax)

    c = torch.empty_like(expected_c, dtype=torch.bfloat16)
    d = torch.empty_like(expected_d, dtype=torch.bfloat16)
    amax = torch.empty((experts, 1), dtype=torch.float32)
    post_amax = torch.empty((experts, 1), dtype=torch.float32)
    op = mudnn.GroupedGemmGluHadamardSm100(
        sample_a=a,
        sample_b=b,
        sample_c=c,
        sample_d=d,
        act_func="swiglu",
    )
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        bias_tensor=bias,
        prob_tensor=prob,
        c_tensor=c,
        d_tensor=d,
        amax_tensor=amax,
        post_rht_amax_tensor=post_amax,
        act_func="swiglu",
    )
    torch.testing.assert_close(c.float(), expected_c.to(torch.bfloat16).float())
    torch.testing.assert_close(d.float(), expected_d.to(torch.bfloat16).float())
    torch.testing.assert_close(amax, expected_amax)
    torch.testing.assert_close(post_amax, expected_post_amax)


def test_mudnn_frontend_cudnn_grouped_gemm_quant_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    m, k, n, experts = 5, 3, 4, 2
    a = torch.arange(m * k, dtype=torch.float32).reshape(m, k, 1) / 11.0 - 0.5
    b = torch.arange(n * k * experts, dtype=torch.float32).reshape(n, k, experts)
    b = b / 17.0 - 0.75
    sfa = torch.linspace(0.5, 1.0, steps=m * k, dtype=torch.float32).reshape(m, k, 1)
    sfb = torch.linspace(
        0.75, 1.25, steps=n * k * experts, dtype=torch.float32
    ).reshape(n, k, experts)
    alpha = torch.tensor([0.75, 1.25], dtype=torch.float32)
    row_scale = torch.linspace(0.5, 1.5, steps=m, dtype=torch.float32)
    bias = torch.linspace(-0.2, 0.3, steps=n * experts, dtype=torch.float32).reshape(
        n, experts
    )
    prob = torch.linspace(0.25, 0.95, steps=m, dtype=torch.float32).reshape(m, 1, 1)
    padded_offsets = torch.tensor([2, 5], dtype=torch.int32)

    expected = torch.empty((m, n, 1), dtype=torch.float32)
    start = 0
    for expert, end in enumerate(padded_offsets.tolist()):
        expected[start:end, :, 0] = torch.einsum(
            "mk,nk->mn",
            a[start:end, :, 0] * sfa[start:end, :, 0],
            b[:, :, expert] * sfb[:, :, expert],
        )
        expected[start:end, :, 0] *= alpha[expert]
        expected[start:end, :, 0] *= row_scale[start:end].reshape(-1, 1)
        expected[start:end, :, 0] += prob[start:end, 0, 0].reshape(-1, 1) * bias[
            :, expert
        ].reshape(1, -1)
        start = end
    expected_amax = torch.stack(
        [
            torch.amax(torch.abs(expected[:2, :, 0])),
            torch.amax(torch.abs(expected[2:, :, 0])),
        ]
    ).reshape(experts, 1)

    outputs = mudnn.grouped_gemm_quant_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        row_scale_tensor=row_scale,
        bias_tensor=bias,
        prob_tensor=prob,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
    )

    torch.testing.assert_close(outputs["d_tensor"], expected)
    torch.testing.assert_close(outputs["amax_tensor"], expected_amax)
    assert outputs["sfd_row_tensor"] is None
    assert outputs["sfd_col_tensor"] is None

    d = torch.empty_like(expected, dtype=torch.bfloat16)
    d_col = torch.empty_like(expected, dtype=torch.bfloat16)
    amax = torch.empty((experts, 1), dtype=torch.float32)
    op = mudnn.GroupedGemmQuantSm100(sample_a=a, sample_b=b)
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        row_scale_tensor=row_scale,
        bias_tensor=bias,
        prob_tensor=prob,
        d_tensor=d,
        d_col_tensor=d_col,
        amax_tensor=amax,
    )
    torch.testing.assert_close(d.float(), expected.to(torch.bfloat16).float())
    torch.testing.assert_close(d_col.float(), expected.to(torch.bfloat16).float())
    torch.testing.assert_close(amax, expected_amax)

    with pytest.raises(ValueError, match="prob_tensor is required"):
        mudnn.grouped_gemm_quant_wrapper_sm100(
            a_tensor=a,
            b_tensor=b,
            sfa_tensor=sfa,
            sfb_tensor=sfb,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha,
            prob_tensor=None,
            d_dtype=torch.float32,
            acc_dtype=torch.float32,
        )


def test_mudnn_frontend_cudnn_grouped_gemm_dswiglu_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    m, k, n, experts = 5, 3, 64, 2
    a = torch.arange(m * k, dtype=torch.float32).reshape(m, k, 1) / 19.0 - 0.5
    b = torch.arange(n * k * experts, dtype=torch.float32).reshape(n, k, experts)
    b = b / 101.0 - 1.0
    c = torch.linspace(-1.0, 1.5, steps=m * n * 2, dtype=torch.float32).reshape(
        m, n * 2, 1
    )
    sfa = torch.linspace(0.25, 0.8, steps=m * k, dtype=torch.float32).reshape(m, k, 1)
    sfb = torch.linspace(0.4, 1.1, steps=n * k * experts, dtype=torch.float32).reshape(
        n, k, experts
    )
    alpha = torch.tensor([0.75, 1.25], dtype=torch.float32)
    beta = torch.tensor([0.5, 1.5], dtype=torch.float32)
    prob = torch.linspace(0.3, 0.9, steps=m, dtype=torch.float32).reshape(m, 1, 1)
    padded_offsets = torch.tensor([2, 5], dtype=torch.int32)

    ref = torch.empty((m, n, 1), dtype=torch.float32)
    c_full = c.clone()
    start = 0
    for expert, end in enumerate(padded_offsets.tolist()):
        ref[start:end, :, 0] = torch.einsum(
            "mk,nk->mn",
            a[start:end, :, 0] * sfa[start:end, :, 0],
            b[:, :, expert] * sfb[:, :, expert],
        )
        ref[start:end, :, 0] *= alpha[expert] * alpha[expert]
        c_full[start:end, :, 0] *= beta[expert]
        start = end

    cols_2n = torch.arange(n * 2, dtype=torch.long)
    block_cols_2n = cols_2n.view((n * 2) // 32, 32)
    dest_idx_glu = block_cols_2n[0::2].reshape(-1)
    dest_idx_ab = block_cols_2n[1::2].reshape(-1)
    c_input = c_full.index_select(1, dest_idx_ab)
    c_gate = c_full.index_select(1, dest_idx_glu)
    sig = torch.sigmoid(c_gate)
    swish = c_gate * sig
    ab = ref * prob.expand_as(ref) * swish
    dswiglu = ref * prob.expand_as(ref) * c_input * sig * (1 + c_gate * (1 - sig))
    expected_d = torch.empty_like(c_full)
    expected_d.index_copy_(dim=1, index=dest_idx_ab, source=ab)
    expected_d.index_copy_(dim=1, index=dest_idx_glu, source=dswiglu)
    expected_dprob = torch.sum(swish * c_input * ref, dim=1, keepdim=True)
    expected_dbias = torch.stack(
        [
            expected_d[:2, :, 0].sum(dim=0),
            expected_d[2:, :, 0].sum(dim=0),
        ]
    ).reshape(experts, n * 2, 1)
    expected_amax = torch.empty((experts, 2, 1), dtype=torch.float32)
    expected_amax[0, 0, 0] = torch.amax(torch.abs(dswiglu[:2, :, 0]))
    expected_amax[0, 1, 0] = torch.amax(torch.abs(ab[:2, :, 0]))
    expected_amax[1, 0, 0] = torch.amax(torch.abs(dswiglu[2:, :, 0]))
    expected_amax[1, 1, 0] = torch.amax(torch.abs(ab[2:, :, 0]))

    dbias = torch.empty((experts, n * 2, 1), dtype=torch.bfloat16)
    outputs = mudnn.grouped_gemm_dswiglu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        c_tensor=c,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        beta_tensor=beta,
        prob_tensor=prob,
        dbias_tensor=dbias,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
    )

    torch.testing.assert_close(outputs["d_row_tensor"], expected_d)
    torch.testing.assert_close(outputs["dprob_tensor"], expected_dprob)
    torch.testing.assert_close(
        outputs["dbias_tensor"].float(), expected_dbias.to(torch.bfloat16).float()
    )
    torch.testing.assert_close(outputs["amax_tensor"], expected_amax)

    d = torch.empty_like(expected_d, dtype=torch.bfloat16)
    dprob = torch.empty_like(expected_dprob)
    amax = torch.empty((experts, 2, 1), dtype=torch.float32)
    op = mudnn.GroupedGemmDswigluSm100(sample_a=a, sample_b=b, sample_c=c)
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        c_tensor=c,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        beta_tensor=beta,
        prob_tensor=prob,
        d_row_tensor=d,
        dprob_tensor=dprob,
        amax_tensor=amax,
    )
    torch.testing.assert_close(d.float(), expected_d.to(torch.bfloat16).float())
    torch.testing.assert_close(dprob, expected_dprob)
    torch.testing.assert_close(amax, expected_amax)


def test_mudnn_frontend_cudnn_grouped_gemm_dglu_dswiglu_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    m, k, n, experts = 5, 3, 64, 2
    a = torch.arange(m * k, dtype=torch.float32).reshape(m, k, 1) / 29.0 - 0.5
    b = torch.arange(n * k * experts, dtype=torch.float32).reshape(n, k, experts)
    b = b / 113.0 - 1.0
    c = torch.linspace(-0.75, 1.25, steps=m * n * 2, dtype=torch.float32).reshape(
        m, n * 2, 1
    )
    sfa = torch.linspace(0.3, 0.85, steps=m * k, dtype=torch.float32).reshape(m, k, 1)
    sfb = torch.linspace(
        0.45, 1.15, steps=n * k * experts, dtype=torch.float32
    ).reshape(n, k, experts)
    alpha = torch.tensor([0.6, 1.4], dtype=torch.float32)
    beta = torch.tensor([0.75, 1.25], dtype=torch.float32)
    prob = torch.linspace(0.35, 0.95, steps=m, dtype=torch.float32).reshape(m, 1, 1)
    padded_offsets = torch.tensor([2, 5], dtype=torch.int32)

    ref = torch.empty((m, n, 1), dtype=torch.float32)
    c_full = c.clone()
    start = 0
    for expert, end in enumerate(padded_offsets.tolist()):
        ref[start:end, :, 0] = torch.einsum(
            "mk,nk->mn",
            a[start:end, :, 0] * sfa[start:end, :, 0],
            b[:, :, expert] * sfb[:, :, expert],
        )
        ref[start:end, :, 0] *= alpha[expert] * alpha[expert]
        c_full[start:end, :, 0] *= beta[expert]
        start = end

    cols_2n = torch.arange(n * 2, dtype=torch.long)
    block_cols_2n = cols_2n.view((n * 2) // 32, 32)
    dest_idx_glu = block_cols_2n[0::2].reshape(-1)
    dest_idx_ab = block_cols_2n[1::2].reshape(-1)
    c_input = c_full.index_select(1, dest_idx_ab)
    c_gate = c_full.index_select(1, dest_idx_glu)
    sig = torch.sigmoid(c_gate)
    swish = c_gate * sig
    ab = ref * prob.expand_as(ref) * swish
    dswiglu = ref * prob.expand_as(ref) * c_input * sig * (1 + c_gate * (1 - sig))
    expected_d = torch.empty_like(c_full)
    expected_d.index_copy_(dim=1, index=dest_idx_ab, source=ab)
    expected_d.index_copy_(dim=1, index=dest_idx_glu, source=dswiglu)
    expected_dprob = torch.sum(swish * c_input * ref, dim=1, keepdim=True)
    expected_dbias = torch.stack(
        [
            expected_d[:2, :, 0].sum(dim=0),
            expected_d[2:, :, 0].sum(dim=0),
        ]
    ).reshape(experts, n * 2, 1)
    expected_amax = torch.empty((experts, 2, 1), dtype=torch.float32)
    expected_amax[0, 0, 0] = torch.amax(torch.abs(dswiglu[:2, :, 0]))
    expected_amax[0, 1, 0] = torch.amax(torch.abs(ab[:2, :, 0]))
    expected_amax[1, 0, 0] = torch.amax(torch.abs(dswiglu[2:, :, 0]))
    expected_amax[1, 1, 0] = torch.amax(torch.abs(ab[2:, :, 0]))

    dbias = torch.empty((experts, n * 2, 1), dtype=torch.bfloat16)
    outputs = mudnn.grouped_gemm_dglu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        c_tensor=c,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        beta_tensor=beta,
        prob_tensor=prob,
        dbias_tensor=dbias,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
        act_func="dswiglu",
    )

    torch.testing.assert_close(outputs["d_row_tensor"], expected_d)
    torch.testing.assert_close(outputs["dprob_tensor"], expected_dprob)
    torch.testing.assert_close(
        outputs["dbias_tensor"].float(), expected_dbias.to(torch.bfloat16).float()
    )
    torch.testing.assert_close(outputs["amax_tensor"], expected_amax)

    d = torch.empty_like(expected_d, dtype=torch.bfloat16)
    dprob = torch.empty_like(expected_dprob)
    amax = torch.empty((experts, 2, 1), dtype=torch.float32)
    op = mudnn.GroupedGemmDgluSm100(
        sample_a=a, sample_b=b, sample_c=c, act_func="dswiglu"
    )
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        c_tensor=c,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        beta_tensor=beta,
        prob_tensor=prob,
        d_row_tensor=d,
        dprob_tensor=dprob,
        amax_tensor=amax,
        act_func="dswiglu",
    )
    torch.testing.assert_close(d.float(), expected_d.to(torch.bfloat16).float())
    torch.testing.assert_close(dprob, expected_dprob)
    torch.testing.assert_close(amax, expected_amax)


def test_mudnn_frontend_cudnn_grouped_gemm_wgrad_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    m, tokens, n, experts = 3, 5, 4, 2
    a = torch.arange(m * tokens, dtype=torch.float32).reshape(m, tokens) / 9.0 - 0.5
    b = torch.arange(tokens * n, dtype=torch.float32).reshape(tokens, n) / 7.0 - 0.75
    sfa = torch.linspace(0.5, 1.0, steps=m * tokens, dtype=torch.float32).reshape(
        m, tokens
    )
    sfb = torch.linspace(0.75, 1.25, steps=tokens * n, dtype=torch.float32).reshape(
        tokens, n
    )
    offsets = torch.tensor([2, 5], dtype=torch.int32)
    global_scale_a = torch.tensor([0.5, 1.25], dtype=torch.float32)
    global_scale_b = torch.tensor([1.5, 0.75], dtype=torch.float32)

    expected = torch.empty((experts, m, n), dtype=torch.float32)
    start = 0
    for expert, end in enumerate(offsets.tolist()):
        expected[expert] = torch.matmul(
            a[:, start:end] * sfa[:, start:end],
            b[start:end, :] * sfb[start:end, :],
        )
        expected[expert] *= global_scale_a[expert] * global_scale_b[expert]
        start = end

    outputs = mudnn.grouped_gemm_wgrad_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        offsets_tensor=offsets,
        global_scale_a=global_scale_a,
        global_scale_b=global_scale_b,
        output_mode="dense",
        input_order="tensor2d",
        acc_dtype=torch.float32,
        wgrad_dtype=torch.float32,
    )

    torch.testing.assert_close(outputs["wgrad_tensor"], expected)

    wgrad = torch.empty_like(expected, dtype=torch.bfloat16)
    op = mudnn.GroupedGemmWgradSm100(
        sample_a=a,
        sample_b=b,
        sample_sfa=sfa,
        sample_sfb=sfb,
        sample_offsets=offsets,
        sample_wgrad=wgrad,
        sample_global_scale_a=global_scale_a,
        sample_global_scale_b=global_scale_b,
        output_mode="dense",
        input_order="tensor2d",
        acc_dtype=torch.float32,
        wgrad_dtype=torch.bfloat16,
    )
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        offsets_tensor=offsets,
        wgrad_tensor=wgrad,
        global_scale_a=global_scale_a,
        global_scale_b=global_scale_b,
        output_mode="dense",
        input_order="tensor2d",
    )
    torch.testing.assert_close(wgrad.float(), expected.to(torch.bfloat16).float())


def test_mudnn_frontend_cudnn_grouped_gemm_dsrelu_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    m, k, n, experts = 5, 3, 4, 2
    a = torch.arange(m * k, dtype=torch.float32).reshape(m, k, 1) / 9.0 - 0.75
    b = torch.arange(n * k * experts, dtype=torch.float32).reshape(n, k, experts)
    b = b / 13.0 - 1.0
    c = torch.linspace(-0.6, 1.4, steps=m * n, dtype=torch.float32).reshape(m, n, 1)
    sfa = torch.linspace(0.4, 0.9, steps=m * k, dtype=torch.float32).reshape(m, k, 1)
    sfb = torch.linspace(0.3, 1.1, steps=n * k * experts, dtype=torch.float32).reshape(
        n, k, experts
    )
    alpha = torch.tensor([0.75, 1.5], dtype=torch.float32)
    prob = torch.linspace(0.2, 0.8, steps=m, dtype=torch.float32).reshape(m, 1, 1)
    padded_offsets = torch.tensor([2, 5], dtype=torch.int32)

    x = torch.empty((m, n, 1), dtype=torch.float32)
    start = 0
    for expert, end in enumerate(padded_offsets.tolist()):
        x[start:end, :, 0] = torch.einsum(
            "mk,nk->mn",
            a[start:end, :, 0] * sfa[start:end, :, 0],
            b[:, :, expert] * sfb[:, :, expert],
        )
        x[start:end, :, 0] *= alpha[expert]
        start = end
    c_relu = torch.relu(c)
    expected_d = 2 * c_relu * x * prob.expand_as(c)
    expected_d_srelu = c_relu**2 * prob.expand_as(c)
    expected_dprob = torch.sum(expected_d_srelu * x, dim=1, keepdim=True)
    expected_dbias = torch.stack(
        [
            expected_d[:2, :, 0].sum(dim=0),
            expected_d[2:, :, 0].sum(dim=0),
        ]
    ).reshape(experts, n, 1)
    expected_amax = torch.stack(
        [
            torch.amax(torch.abs(expected_d[:2, :, 0])),
            torch.amax(torch.abs(expected_d[2:, :, 0])),
        ]
    ).reshape(experts, 1)

    dbias = torch.empty((experts, n, 1), dtype=torch.bfloat16)
    outputs = mudnn.grouped_gemm_dsrelu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        c_tensor=c,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        prob_tensor=prob,
        dbias_tensor=dbias,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
    )

    torch.testing.assert_close(outputs["d_row_tensor"], expected_d)
    torch.testing.assert_close(outputs["d_srelu_tensor"], expected_d_srelu)
    torch.testing.assert_close(outputs["dprob_tensor"], expected_dprob)
    torch.testing.assert_close(
        outputs["dbias_tensor"].float(), expected_dbias.to(torch.bfloat16).float()
    )
    torch.testing.assert_close(outputs["amax_tensor"], expected_amax)

    d = torch.empty_like(expected_d, dtype=torch.bfloat16)
    d_srelu = torch.empty_like(expected_d_srelu, dtype=torch.bfloat16)
    dprob = torch.empty_like(expected_dprob)
    amax = torch.empty((experts, 1), dtype=torch.float32)
    op = mudnn.GroupedGemmDsreluSm100(sample_a=a, sample_b=b, sample_c=c)
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        c_tensor=c,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        padded_offsets=padded_offsets,
        alpha_tensor=alpha,
        prob_tensor=prob,
        d_row_tensor=d,
        d_srelu_tensor=d_srelu,
        dprob_tensor=dprob,
        amax_tensor=amax,
    )
    torch.testing.assert_close(d.float(), expected_d.to(torch.bfloat16).float())
    torch.testing.assert_close(
        d_srelu.float(), expected_d_srelu.to(torch.bfloat16).float()
    )
    torch.testing.assert_close(dprob, expected_dprob)
    torch.testing.assert_close(amax, expected_amax)


def test_mudnn_frontend_cudnn_gemm_dsrelu_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    a = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 7.0 - 1.0
    b = torch.arange(40, dtype=torch.float32).reshape(5, 4, 2) / 11.0 - 1.5
    c = torch.arange(30, dtype=torch.float32).reshape(3, 5, 2) / 13.0 - 0.5
    sfa = torch.full_like(a, 0.25)
    sfb = torch.full_like(b, 0.5)
    prob = torch.linspace(0.5, 1.25, steps=6, dtype=torch.float32).reshape(3, 1, 2)

    x = torch.einsum("mkl,nkl->mnl", a * sfa, b * sfb)
    relu_x = torch.relu(x)
    expected_d = c * prob.expand_as(c) * 2 * relu_x
    expected_dprob = torch.sum(relu_x**2 * c, dim=1, keepdim=True)

    outputs = mudnn.gemm_dsrelu_wrapper_sm100(
        a,
        b,
        c,
        sfa,
        sfb,
        prob,
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    d_tensor, dprob_tensor, amax_tensor, sfd_tensor = outputs

    assert outputs["d_tensor"] is d_tensor
    assert outputs["dprob_tensor"] is dprob_tensor
    assert outputs["amax_tensor"] is amax_tensor
    assert outputs["sfd_tensor"] is sfd_tensor
    assert outputs[0] is d_tensor
    assert outputs[1] is dprob_tensor
    assert d_tensor.stride() == (5, 1, 15)
    torch.testing.assert_close(d_tensor, expected_d)
    torch.testing.assert_close(dprob_tensor, expected_dprob)
    torch.testing.assert_close(
        amax_tensor, torch.amax(torch.abs(expected_d)).reshape(1)
    )

    m_major = mudnn.gemm_dsrelu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        c_tensor=c,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        prob_tensor=prob,
        d_major="m",
        d_dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    assert m_major["d_tensor"].stride() == (1, 3, 15)
    torch.testing.assert_close(m_major["d_tensor"], expected_d)
    with pytest.raises(ValueError, match="d_major"):
        mudnn.gemm_dsrelu_wrapper_sm100(a, b, c, sfa, sfb, prob, d_major="bad")

    d = torch.empty_like(expected_d, dtype=torch.bfloat16)
    dprob = torch.empty_like(expected_dprob)
    amax = torch.empty((1,), dtype=torch.float32)
    op = mudnn.GemmDsreluSm100(sample_a=a, sample_b=b, sample_c=c)
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        c_tensor=c,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        prob_tensor=prob,
        d_tensor=d,
        dprob_tensor=dprob,
        amax_tensor=amax,
    )
    torch.testing.assert_close(d.float(), expected_d.to(torch.bfloat16).float())
    torch.testing.assert_close(dprob, expected_dprob)
    torch.testing.assert_close(amax, torch.amax(torch.abs(expected_d)).reshape(1))


def test_mudnn_frontend_cudnn_gemm_swiglu_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    a = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 5.0 - 1.0
    b = torch.arange(384, dtype=torch.float32).reshape(64, 3, 2) / 29.0 - 3.0
    sfa = torch.full_like(a, 0.5)
    sfb = torch.full_like(b, 0.25)
    alpha = 0.75

    expected_ab12 = alpha * torch.einsum("mkl,nkl->mnl", a * sfa, b * sfb)
    cols = torch.arange(expected_ab12.shape[1], dtype=torch.long)
    block_cols = cols.view(expected_ab12.shape[1] // 32, 32)
    input_idx = block_cols[0::2].reshape(-1)
    gate_idx = block_cols[1::2].reshape(-1)
    input_tensor = expected_ab12.index_select(1, input_idx)
    gate_tensor = expected_ab12.index_select(1, gate_idx)
    expected_c = input_tensor * (gate_tensor * torch.sigmoid(gate_tensor))

    outputs = mudnn.gemm_swiglu_wrapper_sm100(
        a,
        b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        alpha=alpha,
        ab12_dtype=torch.float32,
        c_dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    ab12_tensor, c_tensor, sfc_tensor, amax_tensor = outputs

    assert outputs["ab12_tensor"] is ab12_tensor
    assert outputs["c_tensor"] is c_tensor
    assert outputs["sfc_tensor"] is sfc_tensor
    assert outputs["amax_tensor"] is amax_tensor
    assert outputs[0] is ab12_tensor
    assert outputs[1] is c_tensor
    assert ab12_tensor.stride() == (64, 1, 128)
    assert c_tensor.stride() == (32, 1, 64)
    torch.testing.assert_close(ab12_tensor, expected_ab12)
    torch.testing.assert_close(c_tensor, expected_c)
    assert sfc_tensor is None
    assert amax_tensor is None

    m_major = mudnn.gemm_swiglu_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        alpha=alpha,
        c_major="m",
        ab12_dtype=torch.float32,
        c_dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    assert m_major["ab12_tensor"].stride() == (1, 2, 128)
    assert m_major["c_tensor"].stride() == (1, 2, 64)
    torch.testing.assert_close(m_major["ab12_tensor"], expected_ab12)
    torch.testing.assert_close(m_major["c_tensor"], expected_c)
    with pytest.raises(ValueError, match="c_major"):
        mudnn.gemm_swiglu_wrapper_sm100(a, b, c_major="bad")

    ab12 = torch.empty_like(expected_ab12, dtype=torch.bfloat16)
    c = torch.empty_like(expected_c, dtype=torch.bfloat16)
    amax = torch.empty((1,), dtype=torch.float32)
    op = mudnn.GemmSwigluSm100(sample_a=a, sample_b=b)
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        alpha=alpha,
        ab12_tensor=ab12,
        c_tensor=c,
        amax_tensor=amax,
    )
    torch.testing.assert_close(ab12.float(), expected_ab12.to(torch.bfloat16).float())
    torch.testing.assert_close(c.float(), expected_c.to(torch.bfloat16).float())
    torch.testing.assert_close(amax, torch.amax(torch.abs(expected_c)).reshape(1))


def test_mudnn_frontend_cudnn_gemm_amax_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn
    from mudnn.gemm_amax import gemm_amax

    a = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 7.0 - 1.0
    b = torch.arange(40, dtype=torch.float32).reshape(5, 4, 2) / 11.0 - 1.5
    sfa = torch.full_like(a, 0.5)
    sfb = torch.full_like(b, 0.25)

    expected_c = torch.einsum("mkl,nkl->mnl", a * sfa, b * sfb)
    expected_amax = torch.amax(torch.abs(expected_c)).reshape(1, 1, 1)

    result = mudnn.gemm_amax_wrapper_sm100(
        a,
        b,
        sfa,
        sfb,
        c_dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    c_tensor, amax_tensor = result

    assert result["c_tensor"] is c_tensor
    assert result["amax_tensor"] is amax_tensor
    assert result[0] is c_tensor
    assert result[1] is amax_tensor
    assert c_tensor.stride() == (5, 1, 15)
    torch.testing.assert_close(c_tensor, expected_c)
    torch.testing.assert_close(amax_tensor, expected_amax)

    direct = gemm_amax(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        c_dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    assert direct["c_tensor"].stride() == (5, 1, 15)
    torch.testing.assert_close(direct["c_tensor"], expected_c)
    torch.testing.assert_close(direct["amax_tensor"], expected_amax)

    m_major = mudnn.gemm_amax_wrapper_sm100(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        c_major="m",
        c_dtype=torch.float32,
        acc_dtype=torch.float32,
    )
    assert m_major["c_tensor"].stride() == (1, 3, 15)
    torch.testing.assert_close(m_major["c_tensor"], expected_c)
    torch.testing.assert_close(m_major["amax_tensor"], expected_amax)
    with pytest.raises(ValueError, match="c_major"):
        mudnn.gemm_amax_wrapper_sm100(
            a_tensor=a,
            b_tensor=b,
            sfa_tensor=sfa,
            sfb_tensor=sfb,
            c_major="bad",
        )

    c = torch.empty_like(expected_c, dtype=torch.bfloat16)
    amax = torch.empty((1, 1, 1), dtype=torch.float32)
    op = mudnn.GemmAmaxSm100(sample_a=a, sample_b=b)
    assert op.check_support()
    op.compile().execute(
        a_tensor=a,
        b_tensor=b,
        sfa_tensor=sfa,
        sfb_tensor=sfb,
        c_tensor=c,
        amax_tensor=amax,
    )
    torch.testing.assert_close(c.float(), expected_c.to(torch.bfloat16).float())
    torch.testing.assert_close(amax, expected_amax)


def test_mudnn_frontend_cudnn_rmsnorm_rht_amax_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn

    def hadamard16():
        matrix = torch.tensor([[1.0]], dtype=torch.float32)
        while matrix.shape[0] < 16:
            top = torch.cat((matrix, matrix), dim=1)
            bottom = torch.cat((matrix, -matrix), dim=1)
            matrix = torch.cat((top, bottom), dim=0)
        return matrix / 4.0

    x = torch.arange(64, dtype=torch.float32).reshape(4, 16) / 17.0 - 1.0
    w = torch.linspace(0.5, 1.25, steps=16, dtype=torch.float32)
    eps = 1e-5
    rows_per_cta = 2

    rms = torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    normalized = x / rms * w.unsqueeze(0)
    expected_o = torch.matmul(normalized.view(4, 1, 16), hadamard16()).view(4, 16)
    expected_amax = expected_o.abs().view(2, rows_per_cta, 16).amax(dim=(1, 2))

    outputs = mudnn.rmsnorm_rht_amax_wrapper_sm100(
        x_tensor=x,
        w_tensor=w,
        eps=eps,
        num_threads=128,
        rows_per_cta=rows_per_cta,
    )

    torch.testing.assert_close(outputs["o_tensor"], expected_o)
    torch.testing.assert_close(outputs["amax_tensor"], expected_amax)

    o = torch.empty_like(x, dtype=torch.bfloat16)
    amax = torch.empty((2,), dtype=torch.float32)
    op = mudnn.RmsNormRhtAmaxSm100(
        sample_x=x,
        sample_w=w,
        sample_o=o,
        sample_amax=amax,
        eps=eps,
        num_threads=128,
        rows_per_cta=rows_per_cta,
    )
    assert op.check_support()
    op.compile().execute(
        x_tensor=x,
        w_tensor=w,
        o_tensor=o,
        amax_tensor=amax,
    )
    torch.testing.assert_close(o.float(), expected_o.to(torch.bfloat16).float())
    torch.testing.assert_close(amax, expected_amax)


def test_mudnn_frontend_cudnn_ops_causal_conv1d_torch_fallback():
    torch = pytest.importorskip("torch")
    import mudnn
    import mudnn_frontend as mfe

    x = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0).requires_grad_()
    weight = (
        torch.tensor(
            [[0.5, -0.25, 0.125], [1.0, 0.0, -0.5], [-0.75, 0.25, 0.5]],
            dtype=torch.float32,
        )
        .clone()
        .requires_grad_()
    )
    bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32, requires_grad=True)

    expected = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x, (weight.shape[1] - 1, 0)),
        weight.unsqueeze(1),
        bias=bias,
        groups=x.shape[1],
    )
    actual = mudnn.ops.causal_conv1d(x, weight, bias)
    torch.testing.assert_close(actual, expected)

    silu_actual = mudnn.ops.causal_conv1d(x, weight, bias, activation="silu")
    torch.testing.assert_close(silu_actual, torch.nn.functional.silu(expected))

    root_out = torch.empty_like(x)
    mudnn.causal_conv1d_forward(
        0,
        x,
        weight,
        bias,
        root_out,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        weight.shape[1],
        int(mudnn.data_type.FLOAT),
        0,
    )
    torch.testing.assert_close(root_out, expected)

    mfe_root_out = torch.empty_like(x)
    mfe.causal_conv1d_forward(
        0,
        x,
        weight,
        bias,
        mfe_root_out,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        weight.shape[1],
        int(mudnn.data_type.FLOAT),
        0,
    )
    torch.testing.assert_close(mfe_root_out, expected)

    raw_out = torch.empty_like(x)
    mudnn.causal_conv1d_forward(
        0,
        x.detach().data_ptr(),
        weight.detach().data_ptr(),
        bias.detach().data_ptr(),
        raw_out.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        weight.shape[1],
        int(mudnn.data_type.FLOAT),
        1,
    )
    torch.testing.assert_close(raw_out, torch.nn.functional.silu(expected))

    dy = torch.linspace(-0.5, 0.5, steps=x.numel(), dtype=torch.float32).reshape_as(x)
    expected_dx, expected_dw, expected_db = torch.autograd.grad(
        expected, (x, weight, bias), dy
    )
    dx = torch.empty_like(x)
    dw = torch.empty_like(weight)
    db = torch.empty_like(bias)
    mfe.causal_conv1d_backward(
        0,
        x,
        weight,
        bias,
        dy,
        dx,
        dw,
        db,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        weight.shape[1],
        int(mudnn.data_type.FLOAT),
        int(mudnn.data_type.FLOAT),
        0,
    )
    torch.testing.assert_close(dx, expected_dx)
    torch.testing.assert_close(dw, expected_dw)
    torch.testing.assert_close(db, expected_db)

    loss = silu_actual.sum()
    loss.backward()
    assert x.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None

    dy = torch.linspace(-0.5, 0.75, steps=x.numel(), dtype=torch.float32).reshape_as(x)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    ref = torch.nn.functional.silu(
        torch.nn.functional.conv1d(
            torch.nn.functional.pad(x_ref, (weight_ref.shape[1] - 1, 0)),
            weight_ref.unsqueeze(1),
            bias=bias_ref,
            groups=x_ref.shape[1],
        )
    )
    ref.backward(dy)

    dx = torch.empty_like(x)
    dweight = torch.empty_like(weight)
    dbias = torch.empty_like(bias)
    mudnn.causal_conv1d_backward(
        0,
        x.detach(),
        weight.detach(),
        bias.detach(),
        dy,
        dx,
        dweight,
        dbias,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        weight.shape[1],
        int(mudnn.data_type.FLOAT),
        int(mudnn.data_type.FLOAT),
        1,
    )
    torch.testing.assert_close(dx, x_ref.grad)
    torch.testing.assert_close(dweight, weight_ref.grad)
    torch.testing.assert_close(dbias, bias_ref.grad)

    raw_dx = torch.empty_like(x)
    raw_dweight = torch.empty_like(weight)
    raw_dbias = torch.empty_like(bias)
    mudnn.causal_conv1d_backward(
        0,
        x.detach().data_ptr(),
        weight.detach().data_ptr(),
        bias.detach().data_ptr(),
        dy.data_ptr(),
        raw_dx.data_ptr(),
        raw_dweight.data_ptr(),
        raw_dbias.data_ptr(),
        x.shape[0],
        x.shape[1],
        x.shape[2],
        weight.shape[1],
        int(mudnn.data_type.FLOAT),
        int(mudnn.data_type.FLOAT),
        1,
    )
    torch.testing.assert_close(raw_dx, x_ref.grad)
    torch.testing.assert_close(raw_dweight, weight_ref.grad)
    torch.testing.assert_close(raw_dbias, bias_ref.grad)

    x_bf16 = x.detach().to(torch.bfloat16).requires_grad_(True)
    weight_bf16 = weight.detach().to(torch.bfloat16).requires_grad_(True)
    bias_bf16 = bias.detach().to(torch.bfloat16).requires_grad_(True)
    expected_bf16 = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x_bf16, (weight_bf16.shape[1] - 1, 0)),
        weight_bf16.unsqueeze(1),
        bias=bias_bf16,
        groups=x_bf16.shape[1],
    )
    raw_bf16 = torch.empty_like(x_bf16)
    mudnn.causal_conv1d_forward(
        0,
        x_bf16.detach().data_ptr(),
        weight_bf16.detach().data_ptr(),
        bias_bf16.detach().data_ptr(),
        raw_bf16.data_ptr(),
        x_bf16.shape[0],
        x_bf16.shape[1],
        x_bf16.shape[2],
        weight_bf16.shape[1],
        int(mudnn.data_type.BFLOAT16),
        1,
    )
    torch.testing.assert_close(raw_bf16, torch.nn.functional.silu(expected_bf16))

    dy_bf16 = dy.to(torch.bfloat16)
    ref_bf16 = torch.nn.functional.silu(expected_bf16)
    ref_bf16.backward(dy_bf16)
    raw_dx_bf16 = torch.empty_like(x_bf16)
    raw_dweight_bf16 = torch.empty_like(weight_bf16)
    raw_dbias_bf16 = torch.empty_like(bias_bf16)
    mudnn.causal_conv1d_backward(
        0,
        x_bf16.detach().data_ptr(),
        weight_bf16.detach().data_ptr(),
        bias_bf16.detach().data_ptr(),
        dy_bf16.data_ptr(),
        raw_dx_bf16.data_ptr(),
        raw_dweight_bf16.data_ptr(),
        raw_dbias_bf16.data_ptr(),
        x_bf16.shape[0],
        x_bf16.shape[1],
        x_bf16.shape[2],
        weight_bf16.shape[1],
        int(mudnn.data_type.BFLOAT16),
        int(mudnn.data_type.BFLOAT16),
        1,
    )
    torch.testing.assert_close(raw_dx_bf16, x_bf16.grad)
    torch.testing.assert_close(raw_dweight_bf16, weight_bf16.grad)
    torch.testing.assert_close(raw_dbias_bf16, bias_bf16.grad)

    with pytest.raises(NotImplementedError, match="activation must be None"):
        mudnn.ops.causal_conv1d(x.detach(), weight.detach(), bias.detach(), "relu")


def test_mudnn_frontend_cudnn_ops_b2b_causal_conv1d_torch_fallback():
    """Host raw-pointer B2B path matches the float64 depthwise reference.

    The pointer entrypoints do not call libmudnn. Device tensors are rejected
    here only by shape and dtype; a valid CPU call must still ask for MUSA.
    """
    torch = pytest.importorskip("torch")
    import mudnn
    import mudnn_frontend as mfe

    def reference(x, weights_proj, weights_mixer, skip_bias):
        compute = torch.float64

        def conv(inp, weight):
            return torch.nn.functional.conv1d(
                torch.nn.functional.pad(inp, (weight.shape[-1] - 1, 0)),
                weight.contiguous().unsqueeze(1),
                groups=inp.shape[1],
            )

        proj = conv(x.to(compute), weights_proj.to(compute))
        gated = proj[:, 1::3, :] * proj[:, 2::3, :]
        mixed = conv(gated, weights_mixer.to(compute))
        dim = x.shape[1] // 3
        y = mixed + skip_bias.to(compute).view(1, dim, 1) * gated
        return y, y * proj[:, 0::3, :]

    def check_raw(dtype, data_type, batch, dim, seq_len, kernel_proj, kernel_mixer):
        gen = torch.Generator().manual_seed(
            batch * 10007 + dim * 97 + seq_len + kernel_proj * 17 + kernel_mixer
        )
        x = torch.randn(
            batch, 3 * dim, seq_len, generator=gen, dtype=torch.float32
        ).to(dtype)
        weights_proj = torch.randn(
            3 * dim, kernel_proj, generator=gen, dtype=torch.float32
        ).to(dtype)
        weights_mixer = torch.randn(
            dim, kernel_mixer, generator=gen, dtype=torch.float32
        ).to(dtype)
        skip_bias = torch.randn(dim, generator=gen, dtype=torch.float32).to(dtype)
        y_ref, y_gated_ref = reference(x, weights_proj, weights_mixer, skip_bias)

        def run_forward(entry):
            y = torch.empty(batch, dim, seq_len, dtype=dtype)
            y_gated = torch.empty_like(y)
            entry(
                0,
                x.data_ptr(),
                weights_proj.data_ptr(),
                weights_mixer.data_ptr(),
                skip_bias.data_ptr(),
                y.data_ptr(),
                y_gated.data_ptr(),
                batch,
                dim,
                seq_len,
                kernel_proj,
                kernel_mixer,
                data_type,
            )
            torch.testing.assert_close(y, y_ref.to(dtype), rtol=0, atol=0)
            torch.testing.assert_close(y_gated, y_gated_ref.to(dtype), rtol=0, atol=0)
            return y

        y = run_forward(mudnn.b2b_causal_conv1d_forward)
        run_forward(mfe.b2b_causal_conv1d_forward)

        dy = torch.randn(
            batch, dim, seq_len, generator=gen, dtype=torch.float32
        ).to(dtype)
        x_ref = x.detach().to(torch.float64).requires_grad_(True)
        proj_ref = weights_proj.detach().to(torch.float64).requires_grad_(True)
        mixer_ref = weights_mixer.detach().to(torch.float64).requires_grad_(True)
        bias_ref = skip_bias.detach().to(torch.float64).requires_grad_(True)
        _y_ref, y_gated_grad_ref = reference(x_ref, proj_ref, mixer_ref, bias_ref)
        grads = torch.autograd.grad(
            y_gated_grad_ref,
            (x_ref, proj_ref, mixer_ref, bias_ref),
            dy.to(torch.float64),
        )
        dx = torch.full_like(x, 7)
        dweights_proj = torch.full(weights_proj.shape, 7, dtype=torch.float32)
        dweights_mixer = torch.full(weights_mixer.shape, 7, dtype=torch.float32)
        dskip_bias = torch.full(skip_bias.shape, 7, dtype=torch.float32)
        mudnn.b2b_causal_conv1d_backward(
            0,
            x.data_ptr(),
            weights_proj.data_ptr(),
            weights_mixer.data_ptr(),
            skip_bias.data_ptr(),
            y.data_ptr(),
            dy.data_ptr(),
            dx.data_ptr(),
            dweights_proj.data_ptr(),
            dweights_mixer.data_ptr(),
            dskip_bias.data_ptr(),
            batch,
            dim,
            seq_len,
            kernel_proj,
            kernel_mixer,
            data_type,
            int(mudnn.data_type.FLOAT),
        )
        torch.testing.assert_close(dx, grads[0].to(dtype), rtol=0, atol=0)
        torch.testing.assert_close(dweights_proj, grads[1].to(torch.float32), rtol=0, atol=0)
        torch.testing.assert_close(dweights_mixer, grads[2].to(torch.float32), rtol=0, atol=0)
        torch.testing.assert_close(dskip_bias, grads[3].to(torch.float32), rtol=0, atol=0)

        mfe_dx = torch.empty_like(x)
        mfe_dweights_proj = torch.empty(weights_proj.shape, dtype=torch.float32)
        mfe_dweights_mixer = torch.empty(weights_mixer.shape, dtype=torch.float32)
        mfe_dskip_bias = torch.empty(skip_bias.shape, dtype=torch.float32)
        mfe.b2b_causal_conv1d_backward(
            0,
            x.data_ptr(),
            weights_proj.data_ptr(),
            weights_mixer.data_ptr(),
            skip_bias.data_ptr(),
            y.data_ptr(),
            dy.data_ptr(),
            mfe_dx.data_ptr(),
            mfe_dweights_proj.data_ptr(),
            mfe_dweights_mixer.data_ptr(),
            mfe_dskip_bias.data_ptr(),
            batch,
            dim,
            seq_len,
            kernel_proj,
            kernel_mixer,
            data_type,
            int(mudnn.data_type.FLOAT),
        )
        torch.testing.assert_close(mfe_dx, dx, rtol=0, atol=0)
        torch.testing.assert_close(mfe_dweights_proj, dweights_proj, rtol=0, atol=0)
        torch.testing.assert_close(mfe_dweights_mixer, dweights_mixer, rtol=0, atol=0)
        torch.testing.assert_close(mfe_dskip_bias, dskip_bias, rtol=0, atol=0)

    cases = (
        (torch.float32, int(mudnn.data_type.FLOAT), 1, 1, 4, 2, 2),
        (torch.float32, int(mudnn.data_type.FLOAT), 2, 3, 17, 4, 7),
        (torch.float16, int(mudnn.data_type.HALF), 2, 2, 8, 3, 5),
        (torch.bfloat16, int(mudnn.data_type.BFLOAT16), 1, 2, 6, 2, 4),
    )
    for dtype, data_type, batch, dim, seq_len, kernel_proj, kernel_mixer in cases:
        check_raw(dtype, data_type, batch, dim, seq_len, kernel_proj, kernel_mixer)

    x = torch.randn(1, 3, 4)
    weights_mixer = torch.randn(1, 2)
    skip_bias = torch.randn(1)
    with pytest.raises(ValueError, match="kernel_size_proj"):
        mudnn.ops.b2b_causal_conv1d(x, torch.randn(3, 1), weights_mixer, skip_bias)
    with pytest.raises(ValueError, match="kernel_size_proj"):
        mudnn.ops.b2b_causal_conv1d(x, torch.randn(3, 33), weights_mixer, skip_bias)
    with pytest.raises(ValueError, match="kernel_size_mixer"):
        mudnn.ops.b2b_causal_conv1d(x, torch.randn(3, 2), torch.randn(1, 257), skip_bias)
    with pytest.raises(ValueError, match="divisible by 3"):
        mudnn.ops.b2b_causal_conv1d(
            torch.randn(1, 4, 4), torch.randn(4, 2), weights_mixer, skip_bias
        )
    with pytest.raises(ValueError, match="Unsupported dtype"):
        mudnn.ops.b2b_causal_conv1d(
            x.to(torch.float64),
            torch.randn(3, 2, dtype=torch.float64),
            weights_mixer.to(torch.float64),
            skip_bias.to(torch.float64),
        )
    with pytest.raises(ValueError, match="MUSA"):
        mudnn.ops.b2b_causal_conv1d(x, torch.randn(3, 2), weights_mixer, skip_bias)

    # Explicit sizes are what libmudnn indexes. A mismatched tensor must be
    # rejected before the device launch, including on CPU.
    weights_proj = torch.randn(3, 2)
    y = torch.empty(1, 1, 4)
    y_gated = torch.empty_like(y)
    with pytest.raises(ValueError, match="Expected x shape"):
        mudnn.b2b_causal_conv1d_forward(
            0, x, weights_proj, weights_mixer, skip_bias, y, y_gated,
            2, 1, 4, 2, 2, int(mudnn.data_type.FLOAT),
        )
    with pytest.raises(ValueError, match="Expected weights_proj shape"):
        mudnn.b2b_causal_conv1d_forward(
            0, x, weights_proj, weights_mixer, skip_bias, y, y_gated,
            1, 1, 4, 4, 2, int(mudnn.data_type.FLOAT),
        )
    with pytest.raises(ValueError, match="does not match tensor dtype"):
        mudnn.b2b_causal_conv1d_forward(
            0, x, weights_proj, weights_mixer, skip_bias, y, y_gated,
            1, 1, 4, 2, 2, int(mudnn.data_type.HALF),
        )
    strided_y = torch.empty(1, 1, 8)[:, :, ::2]
    with pytest.raises(ValueError, match="y must be contiguous"):
        mudnn.b2b_causal_conv1d_forward(
            0, x, weights_proj, weights_mixer, skip_bias, strided_y, y_gated,
            1, 1, 4, 2, 2, int(mudnn.data_type.FLOAT),
        )


def test_mudnn_frontend_cudnn_experimental_sdpa_torch_fallback():
    torch = pytest.importorskip("torch")
    from mudnn.experimental.ops.sdpa import (
        _fprop_cache,
        scaled_dot_product_attention,
    )

    _fprop_cache.clear()
    q = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 8) / 17.0
    k = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 8) / 19.0 - 0.5
    v = torch.arange(48, dtype=torch.float32).reshape(1, 2, 3, 8) / 23.0 - 1.0

    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    actual = scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual, expected)
    assert len(_fprop_cache) == 1

    actual_again = scaled_dot_product_attention(q.clone(), k.clone(), v.clone())
    torch.testing.assert_close(actual_again, expected)
    assert len(_fprop_cache) == 1

    causal = scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.25)
    causal_expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=0.25
    )
    torch.testing.assert_close(causal, causal_expected)

    q_gqa = torch.arange(96, dtype=torch.float32).reshape(1, 4, 3, 8) / 29.0
    gqa = scaled_dot_product_attention(q_gqa, k, v, enable_gqa=True)
    repeated_k = k.repeat_interleave(2, dim=1)
    repeated_v = v.repeat_interleave(2, dim=1)
    gqa_expected = torch.nn.functional.scaled_dot_product_attention(
        q_gqa, repeated_k, repeated_v
    )
    torch.testing.assert_close(gqa, gqa_expected)

    bool_mask = torch.tensor(
        [[[[True, False, True], [True, True, False], [False, True, True]]]]
    )
    with pytest.raises(NotImplementedError, match="attn_mask"):
        scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)

    additive_mask = torch.zeros(1, 1, 3, 3, dtype=torch.float32)
    additive_mask[..., 0, 1] = -1000.0
    additive_mask[..., 2, 0] = -1000.0
    with pytest.raises(NotImplementedError, match="attn_mask"):
        scaled_dot_product_attention(q, k, v, attn_mask=additive_mask)

    with pytest.raises(NotImplementedError, match="dropout"):
        scaled_dot_product_attention(q, k, v, dropout_p=1.0)

    _fprop_cache.clear()
    q_d256 = torch.arange(512, dtype=torch.float32).reshape(1, 1, 2, 256) / 59.0
    k_d256 = torch.arange(512, dtype=torch.float32).reshape(1, 1, 2, 256) / 61.0
    v_d256 = torch.arange(512, dtype=torch.float32).reshape(1, 1, 2, 256) / 67.0
    d256 = scaled_dot_product_attention(q_d256, k_d256, v_d256)
    d256_expected = torch.nn.functional.scaled_dot_product_attention(
        q_d256, k_d256, v_d256
    )
    torch.testing.assert_close(d256, d256_expected)
    assert len(_fprop_cache) == 0

    q_seq = torch.arange(64, dtype=torch.float32).reshape(2, 1, 4, 8) / 31.0
    k_seq = torch.arange(64, dtype=torch.float32).reshape(2, 1, 4, 8) / 37.0 - 0.5
    v_seq = torch.arange(64, dtype=torch.float32).reshape(2, 1, 4, 8) / 41.0 - 1.0
    seq_len_q = torch.tensor([2, 3], dtype=torch.int32).reshape(2, 1, 1, 1)
    seq_len_kv = torch.tensor([3, 1], dtype=torch.int32).reshape(2, 1, 1, 1)
    seq_mask = torch.arange(4).reshape(1, 1, 1, 4) < seq_len_kv.reshape(2, 1, 1, 1)
    seq_actual = scaled_dot_product_attention(
        q_seq,
        k_seq,
        v_seq,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
    )
    seq_expected = torch.nn.functional.scaled_dot_product_attention(
        q_seq, k_seq, v_seq, attn_mask=seq_mask
    )
    seq_expected = seq_expected.masked_fill(
        ~(torch.arange(4).reshape(1, 1, 4, 1) < seq_len_q.reshape(2, 1, 1, 1)),
        0,
    )
    torch.testing.assert_close(seq_actual, seq_expected)

    causal_seq_actual = scaled_dot_product_attention(
        q_seq,
        k_seq,
        v_seq,
        is_causal=True,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
    )
    causal_mask = torch.arange(4).reshape(1, 1, 4, 1) >= torch.arange(4).reshape(
        1, 1, 1, 4
    )
    causal_seq_expected = torch.nn.functional.scaled_dot_product_attention(
        q_seq, k_seq, v_seq, attn_mask=causal_mask & seq_mask
    )
    causal_seq_expected = causal_seq_expected.masked_fill(
        ~(torch.arange(4).reshape(1, 1, 4, 1) < seq_len_q.reshape(2, 1, 1, 1)),
        0,
    )
    torch.testing.assert_close(causal_seq_actual, causal_seq_expected)

    before_window_cache = len(_fprop_cache)
    q_index = torch.arange(3).reshape(1, 1, 3, 1)
    k_index = torch.arange(3).reshape(1, 1, 1, 3)
    window_mask = (k_index <= q_index) & (k_index > q_index - 2)
    windowed = scaled_dot_product_attention(
        q, k, v, diagonal_alignment=0, left_bound=2, right_bound=0
    )
    windowed_expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=window_mask
    )
    torch.testing.assert_close(windowed, windowed_expected)
    wider_windowed = scaled_dot_product_attention(
        q, k, v, diagonal_alignment=0, left_bound=3, right_bound=0
    )
    assert wider_windowed.shape == q.shape
    assert len(_fprop_cache) == before_window_cache + 2

    q_br = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4) / 11.0
    k_br = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 13.0 - 0.25
    v_br = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 17.0 - 0.5
    q_br_index = torch.arange(2).reshape(1, 1, 2, 1)
    k_br_index = torch.arange(4).reshape(1, 1, 1, 4)
    bottom_right_mask = (k_br_index <= q_br_index + 2) & (k_br_index > q_br_index)
    bottom_right = scaled_dot_product_attention(
        q_br, k_br, v_br, diagonal_alignment=1, left_bound=2, right_bound=0
    )
    bottom_right_expected = torch.nn.functional.scaled_dot_product_attention(
        q_br, k_br, v_br, attn_mask=bottom_right_mask
    )
    torch.testing.assert_close(bottom_right, bottom_right_expected)

    br_seq = scaled_dot_product_attention(
        q_seq,
        k_seq,
        v_seq,
        is_causal=True,
        diagonal_alignment=1,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
    )
    br_offset = (seq_len_kv - seq_len_q).reshape(2, 1, 1, 1)
    br_seq_mask = torch.arange(4).reshape(1, 1, 1, 4) <= (
        torch.arange(4).reshape(1, 1, 4, 1) + br_offset
    )
    br_seq_mask = br_seq_mask & seq_mask
    br_seq_expected = torch.nn.functional.scaled_dot_product_attention(
        q_seq, k_seq, v_seq, attn_mask=br_seq_mask
    )
    br_seq_expected = br_seq_expected.masked_fill(
        ~(torch.arange(4).reshape(1, 1, 4, 1) < seq_len_q.reshape(2, 1, 1, 1)),
        0,
    )
    torch.testing.assert_close(br_seq, br_seq_expected, equal_nan=True)

    q_cum = torch.arange(40, dtype=torch.float32).reshape(1, 2, 5, 4) / 43.0
    k_cum = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4) / 47.0 - 0.5
    v_cum = torch.arange(24, dtype=torch.float32).reshape(1, 1, 6, 4) / 53.0 - 1.0
    cum_q = torch.tensor([0, 2, 5], dtype=torch.int32).reshape(3, 1, 1, 1)
    cum_kv = torch.tensor([0, 3, 6], dtype=torch.int32).reshape(3, 1, 1, 1)
    cum_actual = scaled_dot_product_attention(
        q_cum,
        k_cum,
        v_cum,
        is_causal=True,
        enable_gqa=True,
        diagonal_alignment=1,
        cumulative_seq_len_q=cum_q,
        cumulative_seq_len_kv=cum_kv,
    )
    cum_expected = torch.zeros_like(cum_actual)
    for q_start, q_end, kv_start, kv_end in [(0, 2, 0, 3), (2, 5, 3, 6)]:
        q_segment = q_cum[:, :, q_start:q_end, :]
        k_segment = k_cum[:, :, kv_start:kv_end, :].repeat_interleave(2, dim=1)
        v_segment = v_cum[:, :, kv_start:kv_end, :].repeat_interleave(2, dim=1)
        q_idx = torch.arange(q_end - q_start).reshape(1, 1, q_end - q_start, 1)
        k_idx = torch.arange(kv_end - kv_start).reshape(1, 1, 1, kv_end - kv_start)
        br_mask = k_idx <= q_idx + ((kv_end - kv_start) - (q_end - q_start))
        cum_expected[
            :, :, q_start:q_end, :
        ] = torch.nn.functional.scaled_dot_product_attention(
            q_segment, k_segment, v_segment, attn_mask=br_mask
        )
    torch.testing.assert_close(cum_actual, cum_expected)
    before_cum_cache = len(_fprop_cache)
    torch.testing.assert_close(
        scaled_dot_product_attention(
            q_cum.clone(),
            k_cum.clone(),
            v_cum.clone(),
            is_causal=True,
            enable_gqa=True,
            diagonal_alignment=1,
            cumulative_seq_len_q=cum_q.clone(),
            cumulative_seq_len_kv=cum_kv.clone(),
        ),
        cum_expected,
    )
    assert len(_fprop_cache) == before_cum_cache

    with pytest.raises(ValueError, match="enable_gqa"):
        scaled_dot_product_attention(q_gqa, k, v)
    with pytest.raises(NotImplementedError, match="attn_mask"):
        scaled_dot_product_attention(
            q, k, v, attn_mask=torch.ones(1, 1, 3, 3, dtype=torch.int32)
        )
    with pytest.raises(ValueError, match="provided together"):
        scaled_dot_product_attention(
            q,
            k,
            v,
            cumulative_seq_len_q=torch.tensor([0, 3], dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="non-decreasing"):
        scaled_dot_product_attention(
            q,
            k,
            v,
            cumulative_seq_len_q=torch.tensor([0, 3, 2], dtype=torch.int32),
            cumulative_seq_len_kv=torch.tensor([0, 1, 3], dtype=torch.int32),
        )


def test_mudnn_frontend_cudnn_sdpa_sm100_forward_backward_torch_fallback():
    torch = pytest.importorskip("torch")
    from mudnn.sdpa import (
        SdpabwdSm100D256,
        SdpafwdSm100D256,
        sdpa_bwd_wrapper_sm100_d256,
        sdpa_fwd_wrapper_sm100_d256,
    )
    from mudnn.sdpa.bwd.api import SdpabwdSm100D256 as BwdAlias
    from mudnn.sdpa.fwd.api import SdpafwdSm100D256 as FwdAlias

    assert FwdAlias is SdpafwdSm100D256
    assert BwdAlias is SdpabwdSm100D256

    q = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) / 17.0
    k = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) / 19.0 - 0.5
    v = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4) / 23.0 - 1.0
    scale = 0.5

    result = sdpa_fwd_wrapper_sm100_d256(
        q,
        k,
        v,
        is_causal=True,
        scale_softmax=scale,
    )
    k_repeated = k.repeat_interleave(2, dim=1)
    v_repeated = v.repeat_interleave(2, dim=1)
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k_repeated) * scale
    causal = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal.view(1, 1, 3, 3), -torch.inf)
    expected_weights = torch.softmax(scores, dim=-1)
    expected_o = torch.einsum("bhqk,bhkd->bhqd", expected_weights, v_repeated)
    expected_lse = torch.logsumexp(scores, dim=-1)
    torch.testing.assert_close(result["o_tensor"], expected_o)
    torch.testing.assert_close(result["lse_tensor"], expected_lse)

    out = torch.empty_like(q)
    lse = torch.empty((1, 2, 3), dtype=torch.float32)
    fwd = SdpafwdSm100D256(q, k, v, out, lse, is_causal=True, scale_softmax=scale)
    assert fwd.check_support()
    fwd.compile().execute(q, k, v, out, lse)
    torch.testing.assert_close(out, expected_o)
    torch.testing.assert_close(lse, expected_lse)

    do = torch.arange(24, dtype=torch.float32).reshape_as(q) / 29.0 - 0.25
    bwd_result = sdpa_bwd_wrapper_sm100_d256(
        q,
        k,
        v,
        result["o_tensor"],
        do,
        result["lse_tensor"],
        is_causal=True,
        scale_softmax=scale,
    )
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q_ref,
        k_ref.repeat_interleave(2, dim=1),
        v_ref.repeat_interleave(2, dim=1),
        is_causal=True,
        scale=scale,
    )
    expected.backward(do)
    torch.testing.assert_close(bwd_result["dq_tensor"], q_ref.grad)
    torch.testing.assert_close(bwd_result["dk_tensor"], k_ref.grad)
    torch.testing.assert_close(bwd_result["dv_tensor"], v_ref.grad)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    bwd = SdpabwdSm100D256(
        q,
        k,
        v,
        result["o_tensor"],
        do,
        result["lse_tensor"],
        dq,
        dk,
        dv,
        is_causal=True,
        scale_softmax=scale,
    )
    assert bwd.check_support()
    bwd.compile().execute(
        q, k, v, result["o_tensor"], do, result["lse_tensor"], dq, dk, dv
    )
    torch.testing.assert_close(dq, q_ref.grad)
    torch.testing.assert_close(dk, k_ref.grad)
    torch.testing.assert_close(dv, v_ref.grad)


def test_mudnn_frontend_cudnn_sdpa_sm100_thd_forward_torch_fallback():
    torch = pytest.importorskip("torch")
    from mudnn.sdpa import sdpa_fwd_wrapper_sm100_d256

    q = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4) / 17.0
    k = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4) / 19.0 - 0.5
    v = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4) / 23.0 - 1.0
    cum = torch.tensor([0, 2, 3], dtype=torch.int32)
    result = sdpa_fwd_wrapper_sm100_d256(
        q,
        k,
        v,
        cum_seqlen_q_tensor=cum,
        cum_seqlen_k_tensor=cum,
        scale_softmax=0.5,
    )
    assert result["o_tensor"].shape == q.shape
    assert result["lse_tensor"].shape == (3, 2)
    assert torch.isfinite(result["o_tensor"]).all()
    assert torch.isfinite(result["lse_tensor"]).all()


def test_mudnn_frontend_cudnn_experimental_moe_grouped_matmul_torch_fallback():
    torch = pytest.importorskip("torch")
    from mudnn.experimental.ops import moe_grouped_matmul
    from mudnn.experimental.ops.moe_grouped_matmul import _graph_cache

    _graph_cache.clear()
    experts, tokens_per_expert, k, n = 3, 2, 4, 5
    tokens = experts * tokens_per_expert
    token = torch.arange(tokens * k, dtype=torch.float32).reshape(1, tokens, k) / 13.0
    weight = (
        torch.arange(experts * k * n, dtype=torch.float32).reshape(experts, k, n) / 17.0
        - 1.0
    )
    first_token_offset = (
        torch.arange(experts, dtype=torch.int32).reshape(experts, 1, 1)
        * tokens_per_expert
    )

    expected = torch.zeros((1, tokens, n), dtype=torch.float32)
    for expert in range(experts):
        start = expert * tokens_per_expert
        end = start + tokens_per_expert
        expected[0, start:end] = token[0, start:end] @ weight[expert]

    actual = moe_grouped_matmul(
        token,
        weight,
        first_token_offset,
        mode="none",
        top_k=1,
    )
    torch.testing.assert_close(actual, expected)
    assert len(_graph_cache) == 1

    actual_again = moe_grouped_matmul(
        token.clone(),
        weight.clone(),
        first_token_offset.clone(),
        mode="none",
        top_k=1,
    )
    torch.testing.assert_close(actual_again, expected)
    assert len(_graph_cache) == 1

    topk_actual = moe_grouped_matmul(
        token,
        weight,
        first_token_offset,
        mode="none",
        top_k=2,
    )
    torch.testing.assert_close(topk_actual, expected)
    assert len(_graph_cache) == 2

    gather_index = torch.tensor([[[4], [1], [5], [0]]], dtype=torch.int32)
    gather_first_token_offset = torch.tensor([0, 1, 3], dtype=torch.int32).reshape(
        experts, 1, 1
    )
    gathered = token.index_select(1, gather_index.reshape(-1).long())
    expected_gather = torch.zeros((1, 4, n), dtype=torch.float32)
    for expert, (start, end) in enumerate(((0, 1), (1, 3), (3, 4))):
        expected_gather[0, start:end] = gathered[0, start:end] @ weight[expert]
    actual_gather = moe_grouped_matmul(
        token,
        weight,
        gather_first_token_offset,
        token_index=gather_index,
        mode="gather",
        top_k=2,
    )
    torch.testing.assert_close(actual_gather, expected_gather)
    assert actual_gather.shape == (1, 4, n)
    assert len(_graph_cache) == 3

    with pytest.raises(ValueError, match="GATHER mode requires token_index"):
        moe_grouped_matmul(token, weight, first_token_offset, mode="gather")
    with pytest.raises(ValueError, match="out of token range"):
        moe_grouped_matmul(
            token,
            weight,
            gather_first_token_offset,
            token_index=torch.tensor([[[tokens]]], dtype=torch.int32),
            mode="gather",
        )
    with pytest.raises(ValueError, match="SCATTER mode requires"):
        moe_grouped_matmul(token, weight, first_token_offset, mode="scatter")
    with pytest.raises(ValueError, match="top_k"):
        moe_grouped_matmul(token, weight, first_token_offset, mode="none", top_k=0)
    with pytest.raises(ValueError, match="K dimensions"):
        moe_grouped_matmul(token, weight[:, :3, :], first_token_offset)


def test_mudnn_frontend_installed_python_import_loads_backend():
    install_prefix = _installed_mudnn_prefix()
    package_dir = install_prefix / "python" / "mudnn_frontend"
    lib_file = install_prefix / "lib" / "libmudnn.so"
    if not package_dir.exists() or not lib_file.exists():
        pytest.skip("installed mudnn Python package is not available")

    python_install_dir = install_prefix / "python"
    cache_files = [
        path
        for path in python_install_dir.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    ]
    assert not cache_files

    env = os.environ.copy()
    env["PYTHONPATH"] = str(python_install_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("MUDNN_LIB", None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import mudnn_frontend as fe",
                    "import mudnn_frontend as mfe",
                    "from mudnn_frontend import _backend",
                    "handle = _backend.Handle()",
                    "handle.close()",
                    "assert fe.__version__ == mfe.__version__",
                    "assert fe.__all__ == mfe.__all__",
                    "assert 'CUDNN_DATA_FP8_E4M3' in fe.__all__",
                    "assert 'CUDNN_DATA_INT4' in fe.__all__",
                    "assert fe.data_type.FLOAT == fe.DataType.FLOAT",
                    "assert fe.CUDNN_DATA_FLOAT == fe.DataType.FLOAT",
                    "assert fe.CUDNN_DATA_FP8_E4M3 == fe.DataType.FP8_E4M3",
                    "assert fe.CUDNN_DATA_INT4 == fe.DataType.INT4",
                    "assert fe.CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR",
                    "assert fe.CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC",
                    "assert fe.CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC",
                    "assert fe.CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC",
                    "assert fe.CUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK",
                    "assert fe.CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR",
                    "assert fe.CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC",
                    "assert fe.CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC",
                    "assert fe.CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC",
                    "assert fe.CUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK",
                    "assert hasattr(fe.OperationBuilder, 'setSdpaDescaleQDesc')",
                    "assert hasattr(fe.OperationBuilder, 'setSdpaBwdScaleDQDesc')",
                    "graph = fe.pygraph(io_data_type=fe.DataType.FLOAT)",
                    "q = graph.tensor(name='q', dim=(1, 1, 2, 2), data_type=fe.DataType.FLOAT)",
                    "k = graph.tensor(name='k', dim=(1, 1, 3, 2), data_type=fe.DataType.FLOAT)",
                    "v = graph.tensor(name='v', dim=(1, 1, 3, 2), data_type=fe.DataType.FLOAT)",
                    "bias = graph.tensor(name='bias', dim=(1, 1, 2, 3), data_type=fe.DataType.FLOAT)",
                    "mask = graph.tensor(name='mask', dim=(1, 1, 2, 3), data_type=fe.DataType.BOOLEAN)",
                    "o, stats = graph.scaled_dot_product_attention(q, k, v, bias=bias, mask=mask, generate_stats=False)",
                    "assert o.dims == (1, 1, 2, 2)",
                    "assert stats is None",
                ]
            ),
        ],
        check=True,
        env=env,
    )
    cache_files = [
        path
        for path in python_install_dir.rglob("*")
        if path.name == "__pycache__" or path.suffix == ".pyc"
    ]
    assert not cache_files
