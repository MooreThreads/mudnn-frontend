# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.
#
# This software ("this software and its documentations" or "the software") is
# protected by Copyright and the information contained herein is confidential.

"""Generic-path tests for ``mudnn.ops.causal_conv1d`` and ``causal_conv1d_nwh``.

Mirrors the cuDNN frontend generic coverage: autograd against an fp64
``F.conv1d`` reference, plus the kernel-size and state rejections.  Float64
inputs are not part of the muDNN contract and are not tested.  Skips when
``torch.musa`` is missing or the backend symbols cannot be loaded.
"""

import math

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

import mudnn  # noqa: E402


def _musa_ready():
    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available():
        return False
    try:
        import mudnn_frontend._backend as backend
    except ImportError:
        return False
    try:
        library = backend.get_library()
    except Exception:
        return False
    return all(
        hasattr(library.lib, name)
        for name in (
            "mudnnCausalConv1dForward",
            "mudnnCausalConv1dBackward",
            "mudnnCausalConv1dNwhForward",
            "mudnnCausalConv1dNwhBackward",
        )
    )


pytestmark = pytest.mark.skipif(
    not _musa_ready(),
    reason="causal conv1d MUSA tests need torch.musa and libmudnn symbols",
)

_TOLERANCES = {
    torch.float32: (5e-6, 5e-6),
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (1e-2, 1e-2),
}

_CASES = [
    pytest.param(torch.float32, 4, "silu", id="fp32-k4-silu"),
    pytest.param(torch.float16, 4, "silu", id="fp16-k4-silu"),
    pytest.param(torch.bfloat16, 4, "silu", id="bf16-k4-silu"),
    pytest.param(torch.bfloat16, 3, "silu", id="bf16-k3-silu"),
    pytest.param(torch.float16, 7, "silu", id="fp16-k7-silu"),
    pytest.param(torch.bfloat16, 4, "identity", id="bf16-k4-identity"),
]

DEVICE = "musa"


def _make_tensor(shape, dtype, scale=1.0):
    return (scale * torch.randn(*shape, device=DEVICE)).to(dtype)


def _cpu_double(tensor):
    # torch.musa conv1d rejects float64, so the reference always runs on CPU.
    return tensor.detach().to(device="cpu", dtype=torch.float64).requires_grad_(True)


def _nch_reference(x, weight, bias=None, activation="identity"):
    dim, kernel_size = weight.shape
    y = F.conv1d(F.pad(x, (kernel_size - 1, 0)), weight.unsqueeze(1), bias=bias, groups=dim)
    return F.silu(y) if activation == "silu" else y


def _nwh_reference(x, weight, bias=None, activation="identity"):
    return _nch_reference(x.transpose(1, 2), weight.transpose(0, 1), bias, activation).transpose(1, 2)


def _assert_close(actual, expected, dtype):
    atol, rtol = _TOLERANCES[dtype]
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().to(dtype).cpu(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,kernel_size,activation", _CASES)
def test_causal_conv1d_autograd(dtype, kernel_size, activation):
    torch.manual_seed(42)
    batch, dim, seq_len = 2, 16, 128
    x_data = _make_tensor((batch, dim, seq_len), dtype, scale=0.1)
    weight_data = _make_tensor((dim, kernel_size), dtype, scale=1.0 / math.sqrt(kernel_size))
    bias_data = _make_tensor((dim,), dtype, scale=0.1)
    grad_out = _make_tensor((batch, dim, seq_len), dtype, scale=0.1)

    x = x_data.detach().requires_grad_(True)
    weight = weight_data.detach().requires_grad_(True)
    bias = bias_data.detach().requires_grad_(True)
    actual = mudnn.ops.causal_conv1d(x, weight, bias, activation=activation)
    actual.backward(grad_out)

    x_ref = _cpu_double(x_data)
    weight_ref = _cpu_double(weight_data)
    bias_ref = _cpu_double(bias_data)
    expected = _nch_reference(x_ref, weight_ref, bias_ref, activation)
    expected.backward(_cpu_double(grad_out).detach())

    _assert_close(actual, expected, dtype)
    _assert_close(x.grad, x_ref.grad, dtype)
    _assert_close(weight.grad, weight_ref.grad, dtype)
    _assert_close(bias.grad, bias_ref.grad, dtype)


@pytest.mark.parametrize("dtype,kernel_size,activation", _CASES)
def test_causal_conv1d_nwh_autograd(dtype, kernel_size, activation):
    torch.manual_seed(42)
    batch, dim, seq_len = 2, 16, 128
    x_data = _make_tensor((batch, seq_len, dim), dtype, scale=0.1)
    weight_data = _make_tensor((kernel_size, dim), dtype, scale=1.0 / math.sqrt(kernel_size))
    bias_data = _make_tensor((dim,), dtype, scale=0.1)
    grad_out = _make_tensor((batch, seq_len, dim), dtype, scale=0.1)

    x = x_data.detach().requires_grad_(True)
    weight = weight_data.detach().requires_grad_(True)
    bias = bias_data.detach().requires_grad_(True)
    actual = mudnn.ops.causal_conv1d_nwh(x, weight, bias, activation=activation)
    actual.backward(grad_out)

    x_ref = _cpu_double(x_data)
    weight_ref = _cpu_double(weight_data)
    bias_ref = _cpu_double(bias_data)
    expected = _nwh_reference(x_ref, weight_ref, bias_ref, activation)
    expected.backward(_cpu_double(grad_out).detach())

    _assert_close(actual, expected, dtype)
    _assert_close(x.grad, x_ref.grad, dtype)
    _assert_close(weight.grad, weight_ref.grad, dtype)
    _assert_close(bias.grad, bias_ref.grad, dtype)


def test_causal_conv1d_kernel_16_and_implicit_bias():
    torch.manual_seed(7)
    batch, dim, seq_len, kernel_size = 2, 8, 64, 16
    x = _make_tensor((batch, dim, seq_len), torch.float32, scale=0.1).requires_grad_(True)
    weight = _make_tensor((dim, kernel_size), torch.float32, scale=1.0 / math.sqrt(kernel_size)).requires_grad_(True)
    grad_out = _make_tensor((batch, dim, seq_len), torch.float32, scale=0.1)

    actual = mudnn.ops.causal_conv1d(x, weight, bias=None, activation="silu")
    actual.backward(grad_out)

    x_ref = _cpu_double(x)
    weight_ref = _cpu_double(weight)
    bias_ref = torch.zeros(dim, dtype=torch.float64, requires_grad=True)
    expected = _nch_reference(x_ref, weight_ref, bias_ref, "silu")
    expected.backward(_cpu_double(grad_out).detach())

    _assert_close(actual, expected, torch.float32)
    _assert_close(x.grad, x_ref.grad, torch.float32)
    _assert_close(weight.grad, weight_ref.grad, torch.float32)


@pytest.mark.parametrize(
    "binding_name,api_name,kernel_size",
    [
        ("causal_conv1d_forward", "causal_conv1d", 1),
        ("causal_conv1d_forward", "causal_conv1d", 257),
        ("causal_conv1d_nwh_forward", "causal_conv1d_nwh", 1),
        ("causal_conv1d_nwh_forward", "causal_conv1d_nwh", 129),
    ],
)
def test_rejects_kernel_size_out_of_range(binding_name, api_name, kernel_size):
    binding = getattr(mudnn, binding_name)
    # Zero pointers never reach the backend: the range check happens first.
    args = [0, 0, 0, 0, 0, 1, 1, 1, kernel_size, 0, 0]
    maximum = 256 if api_name == "causal_conv1d" else 128
    message = rf"{api_name} kernel_size must be between 2 and {maximum}, inclusive; got {kernel_size}"
    with pytest.raises(ValueError, match=message):
        binding(*args)


@pytest.mark.parametrize(
    "op,x_shape,weight_shape",
    [
        (mudnn.ops.causal_conv1d, (1, 4, 16), (4, 9)),
        (mudnn.ops.causal_conv1d_nwh, (1, 16, 4), (9, 4)),
    ],
)
def test_kernel_size_nine_autograd(op, x_shape, weight_shape):
    torch.manual_seed(9)
    x = _make_tensor(x_shape, torch.float32, scale=0.1).requires_grad_(True)
    weight = _make_tensor(weight_shape, torch.float32, scale=1.0 / 3.0).requires_grad_(True)
    bias = _make_tensor((x_shape[-1] if op is mudnn.ops.causal_conv1d_nwh else x_shape[1],), torch.float32, scale=0.1).requires_grad_(True)
    grad_out = _make_tensor(x_shape, torch.float32, scale=0.1)

    actual = op(x, weight, bias, activation="silu")
    actual.backward(grad_out)

    reference = _nwh_reference if op is mudnn.ops.causal_conv1d_nwh else _nch_reference
    x_ref = _cpu_double(x)
    weight_ref = _cpu_double(weight)
    bias_ref = _cpu_double(bias)
    expected = reference(x_ref, weight_ref, bias_ref, "silu")
    expected.backward(_cpu_double(grad_out).detach())

    _assert_close(actual, expected, torch.float32)
    _assert_close(x.grad, x_ref.grad, torch.float32)
    _assert_close(weight.grad, weight_ref.grad, torch.float32)
    _assert_close(bias.grad, bias_ref.grad, torch.float32)


def test_rejects_state_arguments():
    x = torch.zeros(2, 4, 16, device=DEVICE)
    weight = torch.zeros(4, 4, device=DEVICE)
    seq_idx = torch.zeros(2, 16, dtype=torch.int32, device=DEVICE)
    with pytest.raises(NotImplementedError, match="does not yet implement seq_idx"):
        mudnn.ops.causal_conv1d(x, weight, seq_idx=seq_idx)

    cu_seqlens = torch.tensor([0, 16], dtype=torch.int32, device=DEVICE)
    packed = torch.zeros(1, 4, 16, device=DEVICE)
    with pytest.raises(NotImplementedError, match="cu_seqlens-packed"):
        mudnn.ops.causal_conv1d(packed, weight, cu_seqlens=cu_seqlens)

    initial = torch.zeros(2, 4, 3, device=DEVICE)
    with pytest.raises(NotImplementedError, match="cannot execute this stateful"):
        mudnn.ops.causal_conv1d(x, weight, initial_states=initial)
