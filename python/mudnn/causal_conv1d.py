"""Raw causal Conv1D compatibility entrypoints.

MUSA tensors call ``mudnnCausalConv1d*`` in libmudnn.  CPU tensors and raw
host pointers keep the PyTorch reference so the API stays usable without a
device.  A MUSA call never falls back to that reference: a missing symbol or
an unsupported kernel raises instead.
"""

from __future__ import annotations

import ctypes
from typing import Any


_ACTIVATION_IDENTITY = 0
_ACTIVATION_SILU = 1

# cuDNN 9.25.1 accepts every integer in the layout range.  The backend keeps
# register kernels for a few widths and runs the rest on the runtime-K kernel.
_NCH_MAX_KERNEL_SIZE = 256
_NWH_MAX_KERNEL_SIZE = 128
_DATA_FLOAT = 0


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("mudnn.causal_conv1d requires torch") from exc
    return torch


def _dtype_maps():
    torch = _torch()
    return {
        0: (torch.float32, ctypes.c_float),
        2: (torch.float16, ctypes.c_uint16),
        9: (torch.bfloat16, ctypes.c_uint16),
    }


def _normalize_activation(activation: int | str) -> int:
    if isinstance(activation, str):
        if activation == "identity":
            return _ACTIVATION_IDENTITY
        if activation == "silu":
            return _ACTIVATION_SILU
    activation = int(activation)
    if activation not in {_ACTIVATION_IDENTITY, _ACTIVATION_SILU}:
        raise ValueError("Unsupported causal_conv1d activation; expected identity/0 or silu/1")
    return activation


def _kernel_size_message(api_name: str, kernel_size: int, max_kernel_size: int) -> str:
    return f"{api_name} kernel_size must be between 2 and {max_kernel_size}, " f"inclusive; got {kernel_size}"


def _check_kernel_size(api_name: str, kernel_size: int, max_kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size < 2 or kernel_size > max_kernel_size:
        raise ValueError(_kernel_size_message(api_name, kernel_size, max_kernel_size))
    return kernel_size


def _is_musa_tensor(value: Any) -> bool:
    if not _is_torch_tensor(value):
        return False
    return getattr(value, "device", None) is not None and value.device.type == "musa"


def _musa_stream(device) -> int:
    torch = _torch()
    from mudnn._torch_stream import _raw_current_stream

    stream = _raw_current_stream(torch, device)
    if stream is None:
        raise RuntimeError("the current MUSA stream is not available")
    return int(stream)


def _causal_symbol(name: str):
    try:
        from mudnn_frontend._backend import get_library
    except ImportError as exc:
        raise RuntimeError(f"{name} requires libmudnn; the MUSA path does not fall back to " "the host reference") from exc
    try:
        library = get_library()
    except Exception as exc:
        raise RuntimeError(f"{name} requires libmudnn; the MUSA path does not fall back to " "the host reference") from exc
    symbol = getattr(library.lib, name, None)
    if symbol is None:
        raise RuntimeError(f"{name} is not available in libmudnn; the MUSA path does not " "fall back to the host reference")
    return library, symbol


def _launch_forward(symbol_name, api_name, max_kernel_size, stream, x, weight, bias, out, batch, dim, seq_len, kernel_size, data_type, activation):
    kernel_size = _check_kernel_size(api_name, kernel_size, max_kernel_size)
    activation = _normalize_activation(activation)
    library, symbol = _causal_symbol(symbol_name)
    status = symbol(
        ctypes.c_void_p(int(stream)),
        ctypes.c_void_p(int(x)),
        ctypes.c_void_p(int(weight)),
        ctypes.c_void_p(int(bias)),
        ctypes.c_void_p(int(out)),
        int(batch),
        int(dim),
        int(seq_len),
        kernel_size,
        int(data_type),
        activation,
    )
    library.check(status, symbol_name)


def _launch_backward(
    symbol_name,
    api_name,
    max_kernel_size,
    stream,
    x,
    weight,
    bias,
    dy,
    dx,
    dweight,
    dbias,
    batch,
    dim,
    seq_len,
    kernel_size,
    data_type,
    dw_data_type,
    activation,
):
    kernel_size = _check_kernel_size(api_name, kernel_size, max_kernel_size)
    activation = _normalize_activation(activation)
    if int(dw_data_type) != _DATA_FLOAT:
        raise ValueError(f"{api_name} dw_data_type must be FLOAT (0); got {int(dw_data_type)}")
    library, symbol = _causal_symbol(symbol_name)
    status = symbol(
        ctypes.c_void_p(int(stream)),
        ctypes.c_void_p(int(x)),
        ctypes.c_void_p(int(weight)),
        ctypes.c_void_p(int(bias)),
        ctypes.c_void_p(int(dy)),
        ctypes.c_void_p(int(dx)),
        ctypes.c_void_p(int(dweight)),
        ctypes.c_void_p(int(dbias)),
        int(batch),
        int(dim),
        int(seq_len),
        kernel_size,
        int(data_type),
        int(dw_data_type),
        activation,
    )
    library.check(status, symbol_name)


def _validate_tensors(x, weight, bias, dy=None):
    if x.dim() != 3 or weight.dim() != 2 or bias.dim() != 1:
        raise ValueError(f"Expected x(3D), weight(2D), bias(1D); got " f"{x.shape}, {weight.shape}, {bias.shape}")
    if weight.shape[0] != x.shape[1]:
        raise ValueError(f"Channel mismatch: x has dim={x.shape[1]} but weight has shape " f"{weight.shape}")
    if bias.shape[0] != x.shape[1]:
        raise ValueError(f"Bias mismatch: x has dim={x.shape[1]} but bias has shape {bias.shape}")
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise TypeError("x, weight, and bias must have the same dtype")
    if dy is not None and (dy.shape != x.shape or dy.dtype != x.dtype):
        raise ValueError("dy must have the same shape and dtype as x")


def _causal_conv1d_compute(x, weight, bias, activation: int | str = 0):
    torch = _torch()
    activation = _normalize_activation(activation)
    _validate_tensors(x, weight, bias)
    y = torch.nn.functional.conv1d(
        torch.nn.functional.pad(x.contiguous(), (weight.shape[1] - 1, 0)),
        weight.contiguous().unsqueeze(1),
        bias=bias.contiguous(),
        groups=x.shape[1],
    )
    if activation == _ACTIVATION_SILU:
        y = torch.nn.functional.silu(y)
    return y


def _causal_conv1d_grads(x, weight, bias, dy, activation: int | str = 0):
    torch = _torch()
    activation = _normalize_activation(activation)
    _validate_tensors(x, weight, bias, dy)
    with torch.enable_grad():
        x_ref = x.detach().clone().requires_grad_(True)
        weight_ref = weight.detach().clone().requires_grad_(True)
        bias_ref = bias.detach().clone().requires_grad_(True)
        y = _causal_conv1d_compute(x_ref, weight_ref, bias_ref, activation)
        grads = torch.autograd.grad(y, (x_ref, weight_ref, bias_ref), dy.contiguous())
    return grads


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def _numpy_array_from_ptr(ptr, shape, data_type):
    try:
        import numpy as np
    except ImportError as exc:
        raise NotImplementedError("raw pointer causal_conv1d fallback requires numpy") from exc
    dtype_map = _dtype_maps()
    data_type = int(data_type)
    if data_type not in dtype_map:
        raise ValueError(f"Unsupported causal_conv1d data_type {data_type}")
    _torch_dtype, ctypes_type = dtype_map[data_type]
    size = 1
    for dim in shape:
        size *= int(dim)
    array_type = ctypes_type * size
    raw = array_type.from_address(int(ptr))
    if data_type == 0:
        array = np.ctypeslib.as_array(raw).reshape(shape)
    elif data_type == 2:
        array = np.ctypeslib.as_array(raw).view(np.float16).reshape(shape)
    else:
        array = np.ctypeslib.as_array(raw).reshape(shape)
    return array


def _tensor_from_ptr(ptr, shape, data_type):
    torch = _torch()
    tensor = torch.from_numpy(_numpy_array_from_ptr(ptr, shape, data_type))
    if int(data_type) == 9:
        return tensor.view(torch.bfloat16)
    return tensor


def causal_conv1d_forward(
    stream,
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch,
    dim,
    seq_len,
    kernel_size,
    data_type,
    activation,
):
    if _is_musa_tensor(x_ptr):
        _launch_forward(
            "mudnnCausalConv1dForward",
            "causal_conv1d",
            _NCH_MAX_KERNEL_SIZE,
            0 if stream is None else stream,
            x_ptr.data_ptr(),
            weight_ptr.data_ptr(),
            bias_ptr.data_ptr(),
            out_ptr.data_ptr(),
            batch,
            dim,
            seq_len,
            kernel_size,
            data_type,
            activation,
        )
        return None
    if _is_torch_tensor(x_ptr):
        y = _causal_conv1d_compute(x_ptr, weight_ptr, bias_ptr, activation)
        out_ptr.copy_(y.to(dtype=out_ptr.dtype))
        return None

    _check_kernel_size("causal_conv1d", kernel_size, _NCH_MAX_KERNEL_SIZE)
    x = _tensor_from_ptr(x_ptr, (batch, dim, seq_len), data_type)
    weight = _tensor_from_ptr(weight_ptr, (dim, kernel_size), data_type)
    bias = _tensor_from_ptr(bias_ptr, (dim,), data_type)
    out = _tensor_from_ptr(out_ptr, (batch, dim, seq_len), data_type)
    out.copy_(_causal_conv1d_compute(x, weight, bias, activation).to(dtype=out.dtype))
    return None


def causal_conv1d_nwh_forward(
    stream,
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch,
    dim,
    seq_len,
    kernel_size,
    data_type,
    activation,
):
    """NWH forward.  ``x``/``y`` are (batch, seq_len, dim), weight (kernel_size, dim)."""

    if _is_musa_tensor(x_ptr):
        _launch_forward(
            "mudnnCausalConv1dNwhForward",
            "causal_conv1d_nwh",
            _NWH_MAX_KERNEL_SIZE,
            0 if stream is None else stream,
            x_ptr.data_ptr(),
            weight_ptr.data_ptr(),
            bias_ptr.data_ptr(),
            out_ptr.data_ptr(),
            batch,
            dim,
            seq_len,
            kernel_size,
            data_type,
            activation,
        )
        return None
    # No host NWH reference: a raw pointer or CPU tensor is a caller error
    # rather than a silent layout transpose.
    _check_kernel_size("causal_conv1d_nwh", kernel_size, _NWH_MAX_KERNEL_SIZE)
    raise RuntimeError("causal_conv1d_nwh_forward requires MUSA tensors; the host reference " "only implements the NCH layout")


def causal_conv1d_backward(
    stream,
    x_ptr,
    weight_ptr,
    bias_ptr,
    dy_ptr,
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
    batch,
    dim,
    seq_len,
    kernel_size,
    data_type,
    dw_data_type,
    activation,
):
    if _is_musa_tensor(x_ptr):
        _launch_backward(
            "mudnnCausalConv1dBackward",
            "causal_conv1d",
            _NCH_MAX_KERNEL_SIZE,
            0 if stream is None else stream,
            x_ptr.data_ptr(),
            weight_ptr.data_ptr(),
            bias_ptr.data_ptr(),
            dy_ptr.data_ptr(),
            dx_ptr.data_ptr(),
            dweight_ptr.data_ptr(),
            dbias_ptr.data_ptr(),
            batch,
            dim,
            seq_len,
            kernel_size,
            data_type,
            dw_data_type,
            activation,
        )
        return None
    if _is_torch_tensor(x_ptr):
        dx, dweight, dbias = _causal_conv1d_grads(x_ptr, weight_ptr, bias_ptr, dy_ptr, activation)
        dx_ptr.copy_(dx.to(dtype=dx_ptr.dtype))
        dweight_ptr.copy_(dweight.to(dtype=dweight_ptr.dtype))
        dbias_ptr.copy_(dbias.to(dtype=dbias_ptr.dtype))
        return None

    _check_kernel_size("causal_conv1d", kernel_size, _NCH_MAX_KERNEL_SIZE)
    x = _tensor_from_ptr(x_ptr, (batch, dim, seq_len), data_type)
    weight = _tensor_from_ptr(weight_ptr, (dim, kernel_size), data_type)
    bias = _tensor_from_ptr(bias_ptr, (dim,), data_type)
    dy = _tensor_from_ptr(dy_ptr, (batch, dim, seq_len), data_type)
    dx = _tensor_from_ptr(dx_ptr, (batch, dim, seq_len), data_type)
    dweight = _tensor_from_ptr(dweight_ptr, (dim, kernel_size), dw_data_type)
    dbias = _tensor_from_ptr(dbias_ptr, (dim,), dw_data_type)
    grad_x, grad_weight, grad_bias = _causal_conv1d_grads(x, weight, bias, dy, activation)
    dx.copy_(grad_x.to(dtype=dx.dtype))
    dweight.copy_(grad_weight.to(dtype=dweight.dtype))
    dbias.copy_(grad_bias.to(dtype=dbias.dtype))
    return None


def causal_conv1d_nwh_backward(
    stream,
    x_ptr,
    weight_ptr,
    bias_ptr,
    dy_ptr,
    dx_ptr,
    dweight_ptr,
    dbias_ptr,
    batch,
    dim,
    seq_len,
    kernel_size,
    data_type,
    dw_data_type,
    activation,
):
    """NWH backward.  ``x``/``dx``/``dy`` are (batch, seq_len, dim)."""

    if _is_musa_tensor(x_ptr):
        _launch_backward(
            "mudnnCausalConv1dNwhBackward",
            "causal_conv1d_nwh",
            _NWH_MAX_KERNEL_SIZE,
            0 if stream is None else stream,
            x_ptr.data_ptr(),
            weight_ptr.data_ptr(),
            bias_ptr.data_ptr(),
            dy_ptr.data_ptr(),
            dx_ptr.data_ptr(),
            dweight_ptr.data_ptr(),
            dbias_ptr.data_ptr(),
            batch,
            dim,
            seq_len,
            kernel_size,
            data_type,
            dw_data_type,
            activation,
        )
        return None
    _check_kernel_size("causal_conv1d_nwh", kernel_size, _NWH_MAX_KERNEL_SIZE)
    raise RuntimeError("causal_conv1d_nwh_backward requires MUSA tensors; the host reference " "only implements the NCH layout")


__all__ = [
    "_causal_conv1d_compute",
    "_causal_conv1d_grads",
    "causal_conv1d_backward",
    "causal_conv1d_forward",
    "causal_conv1d_nwh_backward",
    "causal_conv1d_nwh_forward",
]
