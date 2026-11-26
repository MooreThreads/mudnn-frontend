"""Causal Conv1D public API, aligned with the cuDNN frontend generic path.

MUSA tensors run through ``mudnnCausalConv1d*``.  CPU tensors keep the host
reference, so the API can be exercised without a device.  A MUSA call that
the backend cannot execute raises; it is never computed by the reference.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import mudnn
from mudnn.causal_conv1d import (
    _NCH_MAX_KERNEL_SIZE,
    _NWH_MAX_KERNEL_SIZE,
    _causal_conv1d_compute,
    _check_kernel_size,
    causal_conv1d_backward,
    causal_conv1d_forward,
    causal_conv1d_nwh_backward,
    causal_conv1d_nwh_forward,
)

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - exercised only without torch installed.
    torch = None
    Tensor = object

_TORCH_DTYPE_TO_MUDNN = {}
if torch is not None:
    _TORCH_DTYPE_TO_MUDNN = {
        torch.float32: mudnn.data_type.FLOAT,
        torch.float16: mudnn.data_type.HALF,
        torch.bfloat16: mudnn.data_type.BFLOAT16,
    }

_ACTIVATION_TO_INT = {"identity": 0, "silu": 1}


def _dtype_to_int(dtype) -> int:
    if dtype not in _TORCH_DTYPE_TO_MUDNN:
        raise ValueError(f"Unsupported dtype {dtype}. Supported: float32, float16, bfloat16.")
    return int(_TORCH_DTYPE_TO_MUDNN[dtype])


def _gradient_dtype(dtype):
    # FP16/BF16 parameter gradients accumulate in FP32, matching cuDNN.
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _normalize_causal_conv1d_activation(activation: Optional[str]) -> str:
    if activation is None or activation == "identity":
        return "identity"
    if activation in ("silu", "swish"):
        return "silu"
    raise NotImplementedError("activation must be None, 'identity', 'silu', or 'swish'")


def _activation_to_int(activation: str) -> int:
    if activation not in _ACTIVATION_TO_INT:
        raise ValueError("Unsupported activation '%s'. Supported: 'identity', 'silu'." % activation)
    return _ACTIVATION_TO_INT[activation]


def _is_musa(tensor) -> bool:
    return getattr(getattr(tensor, "device", None), "type", None) == "musa"


def _same_device(*tensors) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        listed = ", ".join(str(device) for device in devices)
        raise ValueError(f"All tensors must be on the same device: {listed}")


def _match_output_layout(output, public_x):
    """Keep a dense transpose view's stride when the backend wrote contiguous."""

    if output.stride() == public_x.stride():
        return output
    result = torch.empty_like(public_x)
    result.copy_(output)
    return result


def _validate_sequence_contract(
    x,
    weight,
    bias,
    seq_idx,
    cu_seqlens,
    initial_states,
    return_final_states,
    final_states_out,
) -> None:
    if not isinstance(return_final_states, bool):
        raise TypeError(f"return_final_states must be bool, got {type(return_final_states).__name__}")
    if final_states_out is not None and not return_final_states:
        raise ValueError("final_states_out requires return_final_states=True")
    if seq_idx is not None and cu_seqlens is not None:
        raise ValueError("seq_idx and cu_seqlens are mutually exclusive")
    if seq_idx is not None and return_final_states:
        raise ValueError("seq_idx and return_final_states are mutually exclusive")
    for name, tensor in (("x", x), ("weight", weight)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    for name, tensor in (
        ("bias", bias),
        ("seq_idx", seq_idx),
        ("cu_seqlens", cu_seqlens),
        ("initial_states", initial_states),
        ("final_states_out", final_states_out),
    ):
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor).__name__}")

    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, D, T], got {tuple(x.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [D, W], got {tuple(weight.shape)}")
    batch, channels, tokens = x.shape
    if weight.shape[0] != channels:
        raise ValueError(f"weight must have shape [D, W] with D={channels}, got {tuple(weight.shape)}")
    width = weight.shape[1]
    if width < 2:
        raise ValueError(f"weight width must be at least two, got W={width}")
    if bias is not None and tuple(bias.shape) != (channels,):
        raise ValueError(f"bias must have shape {(channels,)}, got {tuple(bias.shape)}")
    if seq_idx is not None and tuple(seq_idx.shape) != (batch, tokens):
        raise ValueError(f"seq_idx must have shape {(batch, tokens)}, got {tuple(seq_idx.shape)}")
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError(f"packed x must have B=1, got B={batch}")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [N + 1] with N >= 1")
        state_batch = cu_seqlens.shape[0] - 1
    else:
        state_batch = batch

    state_shape = (state_batch, channels, width - 1)
    for name, state in (
        ("initial_states", initial_states),
        ("final_states_out", final_states_out),
    ):
        if state is not None and tuple(state.shape) != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {tuple(state.shape)}")
        if state is not None and state.dtype != x.dtype:
            raise TypeError(f"{name} dtype must match x dtype {x.dtype}, got {state.dtype}")
        if state is not None and state.device != x.device:
            raise ValueError(f"{name} device must match x device {x.device}, got {state.device}")


def _reject_unsupported_state(seq_idx, cu_seqlens, initial_states, return_final_states, final_states_out) -> None:
    if seq_idx is not None:
        raise NotImplementedError("the current backend does not yet implement seq_idx")
    if cu_seqlens is not None or initial_states is not None or return_final_states or final_states_out is not None:
        raise NotImplementedError("the current backend cannot execute this stateful causal_conv1d call")


def _fwd_musa(x, weight, bias, activation: str):
    public_x = x
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    batch, dim, seq_len = x.shape
    kernel_size = weight.shape[1]
    if weight.shape[0] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[0]={dim})")
    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")
    _check_kernel_size("causal_conv1d", kernel_size, _NCH_MAX_KERNEL_SIZE)
    y = torch.empty_like(x)
    causal_conv1d_forward(
        _current_stream(x),
        x,
        weight,
        bias,
        y,
        int(batch),
        int(dim),
        int(seq_len),
        int(kernel_size),
        _dtype_to_int(x.dtype),
        _activation_to_int(activation),
    )
    return _match_output_layout(y, public_x)


def _bwd_musa(grad_out, x, weight, bias, activation: str) -> List:
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    grad_out = grad_out.contiguous()
    batch, dim, seq_len = x.shape
    kernel_size = weight.shape[1]
    dx = torch.empty_like(x)
    grad_dtype = _gradient_dtype(x.dtype)
    # dweight/dbias accumulate through atomicAdd, so the caller zeroes them.
    dweight = torch.zeros(weight.shape, device=x.device, dtype=grad_dtype)
    dbias = torch.zeros(bias.shape, device=x.device, dtype=grad_dtype)
    causal_conv1d_backward(
        _current_stream(x),
        x,
        weight,
        bias,
        grad_out,
        dx,
        dweight,
        dbias,
        int(batch),
        int(dim),
        int(seq_len),
        int(kernel_size),
        _dtype_to_int(x.dtype),
        _dtype_to_int(grad_dtype),
        _activation_to_int(activation),
    )
    return [dx, dweight.to(dtype=x.dtype), dbias.to(dtype=x.dtype)]


def _nwh_fwd_musa(x, weight, bias, activation: str):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    batch, seq_len, dim = x.shape
    kernel_size = weight.shape[0]
    if weight.shape[1] != dim:
        raise ValueError(f"Channel mismatch: x has dim={dim} but weight has shape {weight.shape} " f"(expected weight.shape[1]={dim})")
    if bias.shape[0] != dim:
        raise ValueError(f"Bias mismatch: x has dim={dim} but bias has shape {bias.shape} " f"(expected bias.shape[0]={dim})")
    _check_kernel_size("causal_conv1d_nwh", kernel_size, _NWH_MAX_KERNEL_SIZE)
    y = torch.empty_like(x)
    causal_conv1d_nwh_forward(
        _current_stream(x),
        x,
        weight,
        bias,
        y,
        int(batch),
        int(dim),
        int(seq_len),
        int(kernel_size),
        _dtype_to_int(x.dtype),
        _activation_to_int(activation),
    )
    return y


def _nwh_bwd_musa(grad_out, x, weight, bias, activation: str) -> List:
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    grad_out = grad_out.contiguous()
    batch, seq_len, dim = x.shape
    kernel_size = weight.shape[0]
    dx = torch.empty_like(x)
    grad_dtype = _gradient_dtype(x.dtype)
    dweight = torch.zeros(weight.shape, device=x.device, dtype=grad_dtype)
    dbias = torch.zeros(bias.shape, device=x.device, dtype=grad_dtype)
    causal_conv1d_nwh_backward(
        _current_stream(x),
        x,
        weight,
        bias,
        grad_out,
        dx,
        dweight,
        dbias,
        int(batch),
        int(dim),
        int(seq_len),
        int(kernel_size),
        _dtype_to_int(x.dtype),
        _dtype_to_int(grad_dtype),
        _activation_to_int(activation),
    )
    return [dx, dweight.to(dtype=x.dtype), dbias.to(dtype=x.dtype)]


def _register_ops() -> None:
    """Register the MUSA primitives and their autograd.

    ``torch.library.custom_op`` is used when present.  Older torch (the 2.3
    line) has neither it nor ``register_autograd``, so the same schemas are
    registered through ``Library`` and differentiated by an autograd.Function.
    Neither registration sets ``device_types``.
    """

    if torch is None or getattr(_register_ops, "done", False):
        return
    _register_ops.done = True

    custom_op = getattr(torch.library, "custom_op", None)
    register_autograd = getattr(torch.library, "register_autograd", None)
    if custom_op is None or register_autograd is None:
        _register_legacy_ops()
        return

    @custom_op("mudnn::causal_conv1d_fwd_primitive", mutates_args=())
    def _fwd_primitive(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
        return _fwd_musa(x, weight, bias, activation)

    @_fwd_primitive.register_fake
    def _fwd_fake(x, weight, bias, activation):
        return torch.empty_like(x)

    @custom_op("mudnn::causal_conv1d_bwd_primitive", mutates_args=())
    def _bwd_primitive(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
        return _bwd_musa(grad_out, x, weight, bias, activation)

    @_bwd_primitive.register_fake
    def _bwd_fake(grad_out, x, weight, bias, activation):
        return [torch.empty_like(x), torch.empty_like(weight), torch.empty_like(bias)]

    def _setup_context(ctx, inputs, output):
        saved_x, saved_weight, saved_bias, saved_activation = inputs
        ctx.save_for_backward(saved_x, saved_weight, saved_bias)
        ctx.activation = saved_activation

    def _autograd_bwd(ctx, grad_out):
        saved_x, saved_weight, saved_bias = ctx.saved_tensors
        dx, dweight, dbias = torch.ops.mudnn.causal_conv1d_bwd_primitive(grad_out, saved_x, saved_weight, saved_bias, ctx.activation)
        return dx, dweight, dbias, None

    register_autograd(
        "mudnn::causal_conv1d_fwd_primitive",
        _autograd_bwd,
        setup_context=_setup_context,
    )

    @custom_op("mudnn::causal_conv1d_nwh_fwd_primitive", mutates_args=())
    def _nwh_fwd_primitive(x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> Tensor:
        return _nwh_fwd_musa(x, weight, bias, activation)

    @_nwh_fwd_primitive.register_fake
    def _nwh_fwd_fake(x, weight, bias, activation):
        return torch.empty_like(x)

    @custom_op("mudnn::causal_conv1d_nwh_bwd_primitive", mutates_args=())
    def _nwh_bwd_primitive(grad_out: Tensor, x: Tensor, weight: Tensor, bias: Tensor, activation: str) -> List[Tensor]:
        return _nwh_bwd_musa(grad_out, x, weight, bias, activation)

    @_nwh_bwd_primitive.register_fake
    def _nwh_bwd_fake(grad_out, x, weight, bias, activation):
        return [torch.empty_like(x), torch.empty_like(weight), torch.empty_like(bias)]

    def _nwh_setup_context(ctx, inputs, output):
        saved_x, saved_weight, saved_bias, saved_activation = inputs
        ctx.save_for_backward(saved_x, saved_weight, saved_bias)
        ctx.activation = saved_activation

    def _nwh_autograd_bwd(ctx, grad_out):
        saved_x, saved_weight, saved_bias = ctx.saved_tensors
        dx, dweight, dbias = torch.ops.mudnn.causal_conv1d_nwh_bwd_primitive(grad_out, saved_x, saved_weight, saved_bias, ctx.activation)
        return dx, dweight, dbias, None

    register_autograd(
        "mudnn::causal_conv1d_nwh_fwd_primitive",
        _nwh_autograd_bwd,
        setup_context=_nwh_setup_context,
    )


def _current_stream(tensor: Tensor) -> int:
    from mudnn.causal_conv1d import _musa_stream

    return _musa_stream(tensor.device)


def _register_legacy_ops() -> None:
    """torch 2.3 registration: schemas plus a CompositeImplicitAutograd kernel."""

    library = torch.library.Library("mudnn", "FRAGMENT")
    schemas = (
        "causal_conv1d_fwd_primitive(Tensor x, Tensor weight, Tensor bias, str activation) -> Tensor",
        "causal_conv1d_bwd_primitive(Tensor grad_out, Tensor x, Tensor weight, Tensor bias, str activation) -> Tensor[]",
        "causal_conv1d_nwh_fwd_primitive(Tensor x, Tensor weight, Tensor bias, str activation) -> Tensor",
        "causal_conv1d_nwh_bwd_primitive(Tensor grad_out, Tensor x, Tensor weight, Tensor bias, str activation) -> Tensor[]",
    )
    for schema in schemas:
        try:
            library.define(schema)
        except RuntimeError:
            # A second import, or another module, already defined this op.
            pass

    def _define_impl(name, fn):
        if torch._C._dispatch_has_kernel_for_dispatch_key(f"mudnn::{name}", "CompositeImplicitAutograd"):
            return
        library.impl(name, fn, "CompositeImplicitAutograd")

    _define_impl("causal_conv1d_fwd_primitive", _fwd_musa)
    _define_impl("causal_conv1d_bwd_primitive", _bwd_musa)
    _define_impl("causal_conv1d_nwh_fwd_primitive", _nwh_fwd_musa)
    _define_impl("causal_conv1d_nwh_bwd_primitive", _nwh_bwd_musa)
    # The fragment must outlive this function or the kernels are dropped.
    _register_legacy_ops.library = library


_register_ops()


def _musa_forward(x, weight, bias, activation: str):
    return torch.ops.mudnn.causal_conv1d_fwd_primitive(x, weight, bias, activation)


def _musa_nwh_forward(x, weight, bias, activation: str):
    return torch.ops.mudnn.causal_conv1d_nwh_fwd_primitive(x, weight, bias, activation)


if torch is not None and getattr(torch.library, "register_autograd", None) is None:

    class _CausalConv1d(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, activation):
            ctx.save_for_backward(x, weight, bias)
            ctx.activation = activation
            return _fwd_musa(x, weight, bias, activation)

        @staticmethod
        def backward(ctx, grad_out):
            x, weight, bias = ctx.saved_tensors
            dx, dweight, dbias = _bwd_musa(grad_out, x, weight, bias, ctx.activation)
            return dx, dweight, dbias, None

    class _CausalConv1dNwh(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, activation):
            ctx.save_for_backward(x, weight, bias)
            ctx.activation = activation
            return _nwh_fwd_musa(x, weight, bias, activation)

        @staticmethod
        def backward(ctx, grad_out):
            x, weight, bias = ctx.saved_tensors
            dx, dweight, dbias = _nwh_bwd_musa(grad_out, x, weight, bias, ctx.activation)
            return dx, dweight, dbias, None

    def _musa_forward(x, weight, bias, activation: str):
        return _CausalConv1d.apply(x, weight, bias, activation)

    def _musa_nwh_forward(x, weight, bias, activation: str):
        return _CausalConv1dNwh.apply(x, weight, bias, activation)


def causal_conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: Optional[str] = None,
    *,
    seq_idx: Optional[Tensor] = None,
    cu_seqlens: Optional[Tensor] = None,
    initial_states: Optional[Tensor] = None,
    return_final_states: bool = False,
    final_states_out: Optional[Tensor] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Depthwise causal 1D convolution of ``x[B, D, T]``.

    ``seq_idx``, ``cu_seqlens`` and the state arguments are part of the public
    signature but the muDNN backend cannot execute them; passing any of them
    raises ``NotImplementedError`` without changing the state shape.
    """

    if torch is None:
        raise RuntimeError("PyTorch is not available")
    normalized = _normalize_causal_conv1d_activation(activation)
    if x.dim() != 3 or weight.dim() != 2:
        raise ValueError(f"Expected x(3D), weight(2D); got {x.shape}, {weight.shape}")
    if x.dtype == torch.bfloat16 and weight.dtype == torch.float32:
        raise NotImplementedError("BF16 activation with FP32 weight requires the native channel-last " "width-four SiLU route")
    stateful = seq_idx is not None or cu_seqlens is not None or initial_states is not None or return_final_states or final_states_out is not None
    if stateful:
        _validate_sequence_contract(
            x,
            weight,
            bias,
            seq_idx,
            cu_seqlens,
            initial_states,
            return_final_states,
            final_states_out,
        )
        if cu_seqlens is not None and seq_idx is None and not return_final_states:
            raise NotImplementedError("the current backend cannot execute this cu_seqlens-packed " "causal_conv1d call")
        _reject_unsupported_state(seq_idx, cu_seqlens, initial_states, return_final_states, final_states_out)

    if _is_musa(x):
        if bias is None:
            bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
        _same_device(x, weight, bias)
        return _musa_forward(x, weight, bias, normalized)

    if bias is None:
        bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
    if bias.dim() != 1 or bias.shape[0] != x.shape[1]:
        raise ValueError(f"Bias mismatch: x has dim={x.shape[1]} but bias has shape {bias.shape}")
    if x.dtype != weight.dtype or x.dtype != bias.dtype:
        raise TypeError("x, weight, and bias must have the same dtype")
    _dtype_to_int(x.dtype)
    return _causal_conv1d_compute(x, weight, bias, normalized)


def causal_conv1d_nwh(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    activation: str = "identity",
) -> Tensor:
    """Depthwise causal 1D convolution with NWH layout.

    ``x`` is ``(batch, seq_len, dim)`` and ``weight`` is ``(kernel_size, dim)``.
    ``kernel_size`` must be between 2 and 128, inclusive.  ``activation`` is
    ``"identity"`` or ``"silu"``.
    """

    if torch is None:
        raise RuntimeError("PyTorch is not available")
    if activation not in _ACTIVATION_TO_INT:
        raise ValueError("Unsupported activation '%s'. Supported: 'identity', 'silu'." % activation)
    if x.dim() != 3 or weight.dim() != 2:
        raise ValueError(f"Expected x(3D), weight(2D); got {x.shape}, {weight.shape}")
    if bias is None:
        bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)
    if not _is_musa(x):
        raise RuntimeError("causal_conv1d_nwh requires MUSA tensors; the host reference only " "implements the NCH layout")
    _same_device(x, weight, bias)
    return _musa_nwh_forward(x, weight, bias, activation)


__all__ = ["causal_conv1d", "causal_conv1d_nwh"]
