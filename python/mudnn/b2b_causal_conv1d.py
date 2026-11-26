"""Raw back-to-back causal Conv1D entrypoints.

These call ``mudnnB2BCausalConv1dForward`` / ``Backward``.  The public tensor
layout matches the cuDNN 9.24+ SubquadraticOps surface and is the same bytes as
the 4-D view documented on ``mudnn_subquadratic_ops.h``:

    x, dx              (batch, 3 * dim, seq_len)
    weights_proj       (3 * dim, kernel_size_proj)
    weights_mixer      (dim, kernel_size_mixer)
    skip_bias          (dim,)
    y, y_gated, dy     (batch, dim, seq_len)

``kernel_size_proj`` is 2-32 and ``kernel_size_mixer`` is 2-256.  Activations
are float32, float16, or bfloat16.  Weight and skip gradients are accumulated
into caller-owned float32 buffers; the caller zeros them.
"""

from __future__ import annotations

import ctypes
from typing import Any

from mudnn_frontend import _backend


_KP_MIN, _KP_MAX = 2, 32
_KM_MIN, _KM_MAX = 2, 256
_GRAD_DTYPE = 0  # MUDNN_DATA_FLOAT


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("mudnn.b2b_causal_conv1d requires torch") from exc
    return torch


def _dtype_maps():
    torch = _torch()
    return {
        0: (torch.float32, ctypes.c_float),
        2: (torch.float16, ctypes.c_uint16),
        9: (torch.bfloat16, ctypes.c_uint16),
    }


def _dtype_to_int(dtype) -> int:
    torch = _torch()
    mapping = {torch.float32: 0, torch.float16: 2, torch.bfloat16: 9}
    if dtype not in mapping:
        raise ValueError(
            f"Unsupported dtype {dtype}. Supported: float32, float16, bfloat16."
        )
    return mapping[dtype]


def _gradient_dtype(dtype):
    torch = _torch()
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    raise ValueError(
        f"Unsupported dtype {dtype}. Supported: float32, float16, bfloat16."
    )


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


def _validate_shapes(x, weights_proj, weights_mixer, skip_bias, y=None, dy=None):
    if x.dim() != 3:
        raise ValueError(
            f"Expected x(3D) with shape (batch, 3*dim, seq_len); got {tuple(x.shape)}"
        )
    if x.shape[1] % 3 != 0:
        raise ValueError(f"Expected x.shape[1] divisible by 3; got {x.shape[1]}")
    dim = x.shape[1] // 3
    if weights_proj.dim() != 2 or weights_mixer.dim() != 2 or skip_bias.dim() != 1:
        raise ValueError(
            "Expected weights_proj(2D: 3*dim,K_proj), weights_mixer(2D: dim,K_mixer), "
            f"skip_bias(1D: dim); got {tuple(weights_proj.shape)}, "
            f"{tuple(weights_mixer.shape)}, {tuple(skip_bias.shape)}"
        )
    if weights_proj.shape[0] != 3 * dim:
        raise ValueError(
            f"Channel mismatch: x has 3*dim={3 * dim} but weights_proj has shape "
            f"{tuple(weights_proj.shape)}"
        )
    if weights_mixer.shape[0] != dim:
        raise ValueError(
            f"Channel mismatch: x has dim={dim} but weights_mixer has shape "
            f"{tuple(weights_mixer.shape)}"
        )
    if skip_bias.shape[0] != dim:
        raise ValueError(
            f"Channel mismatch: x has dim={dim} but skip_bias has shape "
            f"{tuple(skip_bias.shape)}"
        )
    kernel_size_proj = int(weights_proj.shape[1])
    kernel_size_mixer = int(weights_mixer.shape[1])
    if not _KP_MIN <= kernel_size_proj <= _KP_MAX:
        raise ValueError(
            "b2b_causal_conv1d kernel_size_proj must be between "
            f"{_KP_MIN} and {_KP_MAX}, inclusive; got {kernel_size_proj}"
        )
    if not _KM_MIN <= kernel_size_mixer <= _KM_MAX:
        raise ValueError(
            "b2b_causal_conv1d kernel_size_mixer must be between "
            f"{_KM_MIN} and {_KM_MAX}, inclusive; got {kernel_size_mixer}"
        )
    expected_y = (x.shape[0], dim, x.shape[2])
    if y is not None and tuple(y.shape) != expected_y:
        raise ValueError(f"Expected y shape {expected_y}; got {tuple(y.shape)}")
    if dy is not None and tuple(dy.shape) != expected_y:
        raise ValueError(f"Expected dy shape {expected_y}; got {tuple(dy.shape)}")
    return dim, kernel_size_proj, kernel_size_mixer


def _require_same(tensors, names):
    dtype = tensors[0].dtype
    device = tensors[0].device
    for tensor, name in zip(tensors, names):
        if tensor.dtype != dtype:
            raise TypeError(
                f"Dtype mismatch: {names[0]} is {dtype} but {name} is {tensor.dtype}"
            )
        if tensor.device != device:
            raise ValueError("All tensors must be on the same device")


def _expect_shape(tensor, shape, name):
    expected = tuple(shape)
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"Expected {name} shape {expected}; got {actual}")


def _require_contiguous(tensors, names):
    for tensor, name in zip(tensors, names):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def _device_stream(tensor) -> int:
    device_type = getattr(tensor.device, "type", None)
    if device_type == "musa":
        import torch

        return int(torch.musa.current_stream(tensor.device).musa_stream)
    if device_type == "cuda":
        import torch

        return int(torch.cuda.current_stream(tensor.device).cuda_stream)
    raise ValueError(
        "b2b_causal_conv1d tensors must be on a MUSA (or CUDA) device; "
        f"got {tensor.device}"
    )


def _numpy_array_from_ptr(ptr, shape, data_type):
    try:
        import numpy as np
    except ImportError as exc:
        raise NotImplementedError(
            "raw pointer b2b_causal_conv1d fallback requires numpy"
        ) from exc
    data_type = int(data_type)
    dtype_map = _dtype_maps()
    if data_type not in dtype_map:
        raise ValueError(f"Unsupported b2b_causal_conv1d data_type {data_type}")
    _torch_dtype, ctypes_type = dtype_map[data_type]
    size = 1
    for dim in shape:
        size *= int(dim)
    raw = (ctypes_type * size).from_address(int(ptr))
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


def _reference_forward(x, weights_proj, weights_mixer, skip_bias):
    """Float64 left-padded depthwise reference.  Returns ``(y, y_gated)``."""
    torch = _torch()
    dim, kernel_size_proj, kernel_size_mixer = _validate_shapes(
        x, weights_proj, weights_mixer, skip_bias
    )
    compute = torch.float64

    def conv(inp, weight):
        # weight is (channels, K).  Depthwise, so each channel has its own filter.
        filters = weight.contiguous().unsqueeze(1)
        return torch.nn.functional.conv1d(
            torch.nn.functional.pad(inp, (weight.shape[-1] - 1, 0)),
            filters,
            groups=inp.shape[1],
        )

    x64 = x.to(dtype=compute)
    proj = conv(x64, weights_proj.to(dtype=compute).contiguous())
    gated = proj[:, 1::3, :] * proj[:, 2::3, :]
    del kernel_size_proj, kernel_size_mixer
    mixed = conv(gated, weights_mixer.to(dtype=compute))
    y = mixed + skip_bias.to(dtype=compute).view(1, dim, 1) * gated
    return y, y * proj[:, 0::3, :]


def _call_forward(
    stream, x, weights_proj, weights_mixer, skip_bias, y, y_gated, data_type
):
    lib = _backend.get_library()
    forward = getattr(lib.lib, "mudnnB2BCausalConv1dForward", None)
    if forward is None:
        raise _backend.BackendError(
            "libmudnn does not export mudnnB2BCausalConv1dForward",
            status=_backend.MUDNN_STATUS_NOT_SUPPORTED,
        )
    batch = int(x.shape[0])
    dim = int(x.shape[1] // 3)
    seq_len = int(x.shape[2])
    lib.check(
        forward(
            ctypes.c_void_p(int(stream)),
            ctypes.c_void_p(int(x.data_ptr())),
            ctypes.c_void_p(int(weights_proj.data_ptr())),
            ctypes.c_void_p(int(weights_mixer.data_ptr())),
            ctypes.c_void_p(int(skip_bias.data_ptr())),
            ctypes.c_void_p(int(y.data_ptr())),
            ctypes.c_void_p(int(y_gated.data_ptr())),
            batch,
            dim,
            seq_len,
            int(weights_proj.shape[1]),
            int(weights_mixer.shape[1]),
            int(data_type),
        ),
        "mudnnB2BCausalConv1dForward",
    )


def _call_backward(
    stream,
    x,
    weights_proj,
    weights_mixer,
    skip_bias,
    y,
    dy,
    dx,
    dweights_proj,
    dweights_mixer,
    dskip_bias,
    data_type,
    dw_data_type,
):
    lib = _backend.get_library()
    backward = getattr(lib.lib, "mudnnB2BCausalConv1dBackward", None)
    if backward is None:
        raise _backend.BackendError(
            "libmudnn does not export mudnnB2BCausalConv1dBackward",
            status=_backend.MUDNN_STATUS_NOT_SUPPORTED,
        )
    batch = int(x.shape[0])
    dim = int(x.shape[1] // 3)
    seq_len = int(x.shape[2])
    lib.check(
        backward(
            ctypes.c_void_p(int(stream)),
            ctypes.c_void_p(int(x.data_ptr())),
            ctypes.c_void_p(int(weights_proj.data_ptr())),
            ctypes.c_void_p(int(weights_mixer.data_ptr())),
            ctypes.c_void_p(int(skip_bias.data_ptr())),
            ctypes.c_void_p(int(y.data_ptr())),
            ctypes.c_void_p(int(dy.data_ptr())),
            ctypes.c_void_p(int(dx.data_ptr())),
            ctypes.c_void_p(int(dweights_proj.data_ptr())),
            ctypes.c_void_p(int(dweights_mixer.data_ptr())),
            ctypes.c_void_p(int(dskip_bias.data_ptr())),
            batch,
            dim,
            seq_len,
            int(weights_proj.shape[1]),
            int(weights_mixer.shape[1]),
            int(data_type),
            int(dw_data_type),
        ),
        "mudnnB2BCausalConv1dBackward",
    )


def b2b_forward_primitive(x, weights_proj, weights_mixer, skip_bias):
    """Run the fused forward and return ``(y, y_gated)``."""
    _validate_shapes(x, weights_proj, weights_mixer, skip_bias)
    _require_same(
        (x, weights_proj, weights_mixer, skip_bias),
        ("x", "weights_proj", "weights_mixer", "skip_bias"),
    )
    _dtype_to_int(x.dtype)
    x = x.contiguous()
    weights_proj = weights_proj.contiguous()
    weights_mixer = weights_mixer.contiguous()
    skip_bias = skip_bias.contiguous()
    dim = x.shape[1] // 3
    y = _torch().empty(x.shape[0], dim, x.shape[2], device=x.device, dtype=x.dtype)
    y_gated = _torch().empty_like(y)
    _call_forward(
        _device_stream(x),
        x,
        weights_proj,
        weights_mixer,
        skip_bias,
        y,
        y_gated,
        _dtype_to_int(x.dtype),
    )
    return y, y_gated


def b2b_backward_primitive(dy, x, weights_proj, weights_mixer, skip_bias, y):
    """Return ``(dx, dweights_proj, dweights_mixer, dskip_bias)`` in ``x.dtype``.

    The backend accumulates float32 parameter gradients.  This helper zeros
    those buffers, so repeated calls do not accumulate.
    """
    _validate_shapes(x, weights_proj, weights_mixer, skip_bias, y=y, dy=dy)
    _require_same(
        (dy, x, weights_proj, weights_mixer, skip_bias, y),
        ("dy", "x", "weights_proj", "weights_mixer", "skip_bias", "y"),
    )
    torch = _torch()
    x = x.contiguous()
    weights_proj = weights_proj.contiguous()
    weights_mixer = weights_mixer.contiguous()
    skip_bias = skip_bias.contiguous()
    y = y.contiguous()
    dy = dy.contiguous()
    grad_dtype = _gradient_dtype(x.dtype)
    dx = torch.empty_like(x)
    dweights_proj = torch.zeros(weights_proj.shape, device=x.device, dtype=grad_dtype)
    dweights_mixer = torch.zeros(weights_mixer.shape, device=x.device, dtype=grad_dtype)
    dskip_bias = torch.zeros(skip_bias.shape, device=x.device, dtype=grad_dtype)
    _call_backward(
        _device_stream(x),
        x,
        weights_proj,
        weights_mixer,
        skip_bias,
        y,
        dy,
        dx,
        dweights_proj,
        dweights_mixer,
        dskip_bias,
        _dtype_to_int(x.dtype),
        _GRAD_DTYPE,
    )
    return (
        dx,
        dweights_proj.to(dtype=x.dtype),
        dweights_mixer.to(dtype=x.dtype),
        dskip_bias.to(dtype=x.dtype),
    )


class _B2BFunction:
    """Autograd wrapper.  Built lazily so importing this module needs no torch."""

    _cls = None

    @classmethod
    def apply(cls, x, weights_proj, weights_mixer, skip_bias):
        impl = cls._cls
        if impl is None:
            torch = _torch()

            class _Impl(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, weights_proj, weights_mixer, skip_bias):
                    y, y_gated = b2b_forward_primitive(
                        x, weights_proj, weights_mixer, skip_bias
                    )
                    ctx.save_for_backward(x, weights_proj, weights_mixer, skip_bias, y)
                    return y_gated

                @staticmethod
                def backward(ctx, grad_y_gated):
                    x, weights_proj, weights_mixer, skip_bias, y = ctx.saved_tensors
                    if grad_y_gated is None:
                        raise RuntimeError(
                            "b2b_causal_conv1d backward requires a gradient for y_gated"
                        )
                    dx, dwp, dwm, dsb = b2b_backward_primitive(
                        grad_y_gated, x, weights_proj, weights_mixer, skip_bias, y
                    )
                    return dx, dwp, dwm, dsb

            impl = _Impl
            cls._cls = impl
        return impl.apply(x, weights_proj, weights_mixer, skip_bias)


def b2b_causal_conv1d(x, weights_proj, weights_mixer, skip_bias):
    """Fused projection, gating, mixer, skip, and post-gating.

    Returns ``y_gated`` of shape ``(batch, dim, seq_len)``.  The intermediate
    mixer-plus-skip tensor is retained for backward and is not returned.
    """
    return _B2BFunction.apply(x, weights_proj, weights_mixer, skip_bias)


def b2b_causal_conv1d_forward(
    stream,
    x_ptr,
    weights_proj_ptr,
    weights_mixer_ptr,
    skip_bias_ptr,
    y_ptr,
    y_gated_ptr,
    batch,
    dim,
    seq_len,
    kernel_size_proj,
    kernel_size_mixer,
    data_type,
):
    """Low-level forward.  Tensor arguments write ``y`` and ``y_gated`` in place.

    Raw pointers use the host-memory torch reference so the single-op pointer
    tests and CPU callers keep working.  Device tensors must already be
    contiguous and must match ``batch``, ``dim``, ``seq_len``, and both kernel
    sizes.  A zero ``stream`` uses the tensor's current MUSA or CUDA stream;
    any other value is the ``musaStream_t`` passed to libmudnn.
    """
    if _is_torch_tensor(x_ptr):
        _validate_shapes(x_ptr, weights_proj_ptr, weights_mixer_ptr, skip_bias_ptr)
        _require_same(
            (
                x_ptr,
                weights_proj_ptr,
                weights_mixer_ptr,
                skip_bias_ptr,
                y_ptr,
                y_gated_ptr,
            ),
            ("x", "weights_proj", "weights_mixer", "skip_bias", "y", "y_gated"),
        )
        _expect_shape(x_ptr, (batch, 3 * dim, seq_len), "x")
        _expect_shape(weights_proj_ptr, (3 * dim, kernel_size_proj), "weights_proj")
        _expect_shape(weights_mixer_ptr, (dim, kernel_size_mixer), "weights_mixer")
        _expect_shape(skip_bias_ptr, (dim,), "skip_bias")
        _expect_shape(y_ptr, (batch, dim, seq_len), "y")
        _expect_shape(y_gated_ptr, (batch, dim, seq_len), "y_gated")
        if _dtype_to_int(x_ptr.dtype) != int(data_type):
            raise ValueError(
                f"data_type {data_type} does not match tensor dtype {x_ptr.dtype}"
            )
        _require_contiguous((y_ptr, y_gated_ptr), ("y", "y_gated"))
        _call_forward(
            _device_stream(x_ptr) if not stream else stream,
            x_ptr.contiguous(),
            weights_proj_ptr.contiguous(),
            weights_mixer_ptr.contiguous(),
            skip_bias_ptr.contiguous(),
            y_ptr,
            y_gated_ptr,
            int(data_type),
        )
        return None

    x = _tensor_from_ptr(x_ptr, (batch, 3 * dim, seq_len), data_type)
    weights_proj = _tensor_from_ptr(
        weights_proj_ptr, (3 * dim, kernel_size_proj), data_type
    )
    weights_mixer = _tensor_from_ptr(
        weights_mixer_ptr, (dim, kernel_size_mixer), data_type
    )
    skip_bias = _tensor_from_ptr(skip_bias_ptr, (dim,), data_type)
    y = _tensor_from_ptr(y_ptr, (batch, dim, seq_len), data_type)
    y_gated = _tensor_from_ptr(y_gated_ptr, (batch, dim, seq_len), data_type)
    y_ref, y_gated_ref = _reference_forward(x, weights_proj, weights_mixer, skip_bias)
    y.copy_(y_ref.to(dtype=y.dtype))
    y_gated.copy_(y_gated_ref.to(dtype=y_gated.dtype))
    del stream
    return None


def b2b_causal_conv1d_backward(
    stream,
    x_ptr,
    weights_proj_ptr,
    weights_mixer_ptr,
    skip_bias_ptr,
    y_ptr,
    dy_ptr,
    dx_ptr,
    dweights_proj_ptr,
    dweights_mixer_ptr,
    dskip_bias_ptr,
    batch,
    dim,
    seq_len,
    kernel_size_proj,
    kernel_size_mixer,
    data_type,
    dw_data_type,
):
    """Low-level backward.  Tensor grads are accumulated, matching the C API.

    Parameter gradients must already be contiguous float32.  ``dx`` is
    overwritten and must match ``x``.  A zero ``stream`` uses the tensor's
    current MUSA or CUDA stream.  Raw pointers use the host reference and
    overwrite the gradient buffers.
    """
    if _is_torch_tensor(x_ptr):
        if int(dw_data_type) != _GRAD_DTYPE:
            raise ValueError("dw_data_type must be float32")
        tensors = (
            x_ptr,
            weights_proj_ptr,
            weights_mixer_ptr,
            skip_bias_ptr,
            y_ptr,
            dy_ptr,
            dx_ptr,
        )
        _validate_shapes(
            x_ptr,
            weights_proj_ptr,
            weights_mixer_ptr,
            skip_bias_ptr,
            y=y_ptr,
            dy=dy_ptr,
        )
        _require_same(
            tensors,
            ("x", "weights_proj", "weights_mixer", "skip_bias", "y", "dy", "dx"),
        )
        _expect_shape(x_ptr, (batch, 3 * dim, seq_len), "x")
        _expect_shape(dx_ptr, (batch, 3 * dim, seq_len), "dx")
        _expect_shape(weights_proj_ptr, (3 * dim, kernel_size_proj), "weights_proj")
        _expect_shape(weights_mixer_ptr, (dim, kernel_size_mixer), "weights_mixer")
        _expect_shape(skip_bias_ptr, (dim,), "skip_bias")
        _expect_shape(y_ptr, (batch, dim, seq_len), "y")
        _expect_shape(dy_ptr, (batch, dim, seq_len), "dy")
        _expect_shape(dweights_proj_ptr, (3 * dim, kernel_size_proj), "dweights_proj")
        _expect_shape(dweights_mixer_ptr, (dim, kernel_size_mixer), "dweights_mixer")
        _expect_shape(dskip_bias_ptr, (dim,), "dskip_bias")
        if _dtype_to_int(x_ptr.dtype) != int(data_type):
            raise ValueError(
                f"data_type {data_type} does not match tensor dtype {x_ptr.dtype}"
            )
        grad_dtype = _torch().float32
        if (
            dweights_proj_ptr.dtype != grad_dtype
            or dweights_mixer_ptr.dtype != grad_dtype
            or dskip_bias_ptr.dtype != grad_dtype
        ):
            raise TypeError(
                "dweights_proj, dweights_mixer, and dskip_bias must be float32"
            )
        _require_contiguous(
            (
                y_ptr,
                dy_ptr,
                dx_ptr,
                dweights_proj_ptr,
                dweights_mixer_ptr,
                dskip_bias_ptr,
            ),
            ("y", "dy", "dx", "dweights_proj", "dweights_mixer", "dskip_bias"),
        )
        _call_backward(
            _device_stream(x_ptr) if not stream else stream,
            x_ptr.contiguous(),
            weights_proj_ptr.contiguous(),
            weights_mixer_ptr.contiguous(),
            skip_bias_ptr.contiguous(),
            y_ptr,
            dy_ptr,
            dx_ptr,
            dweights_proj_ptr,
            dweights_mixer_ptr,
            dskip_bias_ptr,
            int(data_type),
            _GRAD_DTYPE,
        )
        return None

    torch = _torch()
    x = _tensor_from_ptr(x_ptr, (batch, 3 * dim, seq_len), data_type)
    weights_proj = _tensor_from_ptr(
        weights_proj_ptr, (3 * dim, kernel_size_proj), data_type
    )
    weights_mixer = _tensor_from_ptr(
        weights_mixer_ptr, (dim, kernel_size_mixer), data_type
    )
    skip_bias = _tensor_from_ptr(skip_bias_ptr, (dim,), data_type)
    y = _tensor_from_ptr(y_ptr, (batch, dim, seq_len), data_type)
    dy = _tensor_from_ptr(dy_ptr, (batch, dim, seq_len), data_type)
    x_ref = x.detach().to(dtype=torch.float64).requires_grad_(True)
    wp_ref = weights_proj.detach().to(dtype=torch.float64).requires_grad_(True)
    wm_ref = weights_mixer.detach().to(dtype=torch.float64).requires_grad_(True)
    sb_ref = skip_bias.detach().to(dtype=torch.float64).requires_grad_(True)
    _y_ref, y_gated_ref = _reference_forward(x_ref, wp_ref, wm_ref, sb_ref)
    grads = torch.autograd.grad(
        y_gated_ref, (x_ref, wp_ref, wm_ref, sb_ref), dy.to(dtype=torch.float64)
    )
    dx = _tensor_from_ptr(dx_ptr, (batch, 3 * dim, seq_len), data_type)
    dweights_proj = _tensor_from_ptr(
        dweights_proj_ptr, (3 * dim, kernel_size_proj), dw_data_type
    )
    dweights_mixer = _tensor_from_ptr(
        dweights_mixer_ptr, (dim, kernel_size_mixer), dw_data_type
    )
    dskip_bias = _tensor_from_ptr(dskip_bias_ptr, (dim,), dw_data_type)
    dx.copy_(grads[0].to(dtype=dx.dtype))
    dweights_proj.copy_(grads[1].to(dtype=dweights_proj.dtype))
    dweights_mixer.copy_(grads[2].to(dtype=dweights_mixer.dtype))
    dskip_bias.copy_(grads[3].to(dtype=dskip_bias.dtype))
    del stream, y
    return None


__all__ = [
    "b2b_backward_primitive",
    "b2b_causal_conv1d",
    "b2b_causal_conv1d_backward",
    "b2b_causal_conv1d_forward",
    "b2b_forward_primitive",
]
