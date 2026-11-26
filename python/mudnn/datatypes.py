"""Datatype conversion helpers for the cuDNN compatibility package."""

from __future__ import annotations

import sys

import mudnn

cudnn_data_type = mudnn.data_type
module_name = (
    ".Release._compiled_module"
    if sys.platform.startswith("win")
    else "._compiled_module"
)

torch_available = None
_torch_to_cudnn_data_type_dict = None
cutlass_available = None
_torch_to_cutlass_data_type_dict = None


def is_windows():
    return sys.platform.startswith("win")


def is_torch_available():
    global torch_available, _torch_to_cudnn_data_type_dict
    if torch_available is None:
        try:
            import torch

            torch_available = True
            mapping = {
                torch.half: cudnn_data_type.HALF,
                torch.float16: cudnn_data_type.HALF,
                torch.bfloat16: cudnn_data_type.BFLOAT16,
                torch.float: cudnn_data_type.FLOAT,
                torch.float32: cudnn_data_type.FLOAT,
                torch.double: cudnn_data_type.DOUBLE,
                torch.float64: cudnn_data_type.DOUBLE,
                torch.int8: cudnn_data_type.INT8,
                torch.int32: cudnn_data_type.INT32,
                torch.int64: cudnn_data_type.INT64,
                torch.uint8: cudnn_data_type.UINT8,
                torch.bool: cudnn_data_type.BOOLEAN,
            }
            optional = {
                "float8_e4m3fn": cudnn_data_type.FP8_E4M3,
                "float8_e5m2": cudnn_data_type.FP8_E5M2,
                "float8_e8m0fnu": cudnn_data_type.FP8_E8M0,
                "float4_e2m1fn_x2": cudnn_data_type.FP4_E2M1,
            }
            for name, dtype in optional.items():
                if hasattr(torch, name):
                    mapping[getattr(torch, name)] = dtype
            _torch_to_cudnn_data_type_dict = mapping
        except ImportError:
            torch_available = False
            _torch_to_cudnn_data_type_dict = {}
    return torch_available


def is_cutlass_available():
    global cutlass_available, _torch_to_cutlass_data_type_dict
    if cutlass_available is None:
        try:
            import cutlass
            import torch

            cutlass_available = True
            mapping = {
                torch.half: getattr(cutlass, "Float16", None),
                torch.float16: getattr(cutlass, "Float16", None),
                torch.bfloat16: getattr(cutlass, "BFloat16", None),
                torch.float: getattr(cutlass, "Float32", None),
                torch.float32: getattr(cutlass, "Float32", None),
                torch.double: getattr(cutlass, "Float64", None),
                torch.float64: getattr(cutlass, "Float64", None),
                torch.int8: getattr(cutlass, "Int8", None),
                torch.int32: getattr(cutlass, "Int32", None),
                torch.int64: getattr(cutlass, "Int64", None),
                torch.uint8: getattr(cutlass, "Uint8", None),
                torch.bool: getattr(cutlass, "Boolean", None),
            }
            optional = {
                "float8_e4m3fn": "Float8E4M3FN",
                "float8_e5m2": "Float8E5M2",
                "float8_e8m0fnu": "Float8E8M0FNU",
                "float4_e2m1fn_x2": "Float4E2M1FN",
            }
            for torch_name, cutlass_name in optional.items():
                if hasattr(torch, torch_name):
                    mapping[getattr(torch, torch_name)] = getattr(
                        cutlass, cutlass_name, None
                    )
            _torch_to_cutlass_data_type_dict = {
                torch_type: cutlass_type
                for torch_type, cutlass_type in mapping.items()
                if cutlass_type is not None
            }
        except ImportError:
            cutlass_available = False
            _torch_to_cutlass_data_type_dict = {}
    return cutlass_available


def _torch_to_cudnn_data_type(torch_data_type):
    if is_torch_available():
        return _torch_to_cudnn_data_type_dict.get(torch_data_type)
    return None


def _torch_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2: bool = False):
    if is_cutlass_available() and is_torch_available():
        if interpret_uint8_as_fp4x2:
            import torch
            import cutlass

            if data_type == torch.uint8:
                return getattr(cutlass, "Float4E2M1FN", None)
        return _torch_to_cutlass_data_type_dict.get(data_type)
    return None


def _convert_to_cutlass_data_type(data_type, interpret_uint8_as_fp4x2: bool = False):
    if is_cutlass_available():
        import cutlass

        if isinstance(data_type, type) and issubclass(data_type, cutlass.Numeric):
            return data_type
        cutlass_type = _torch_to_cutlass_data_type(
            data_type, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2
        )
        if cutlass_type is None:
            raise ValueError("Unsupported tensor data type.")
        return cutlass_type
    return None


def _cudnn_to_torch_data_type(data_type):
    if is_torch_available():
        for torch_type, cudnn_type in _torch_to_cudnn_data_type_dict.items():
            if cudnn_type == data_type:
                return torch_type
    return None


def _library_type(input_type):
    if type(input_type) is cudnn_data_type:
        return input_type
    converted = _torch_to_cudnn_data_type(input_type)
    if converted is not None:
        return converted
    raise Exception(
        f"No available conversion from type {input_type} to a library type."
    )


def _is_torch_tensor(input_tensor) -> bool:
    if is_torch_available():
        import torch

        return isinstance(input_tensor, torch.Tensor)
    return False
