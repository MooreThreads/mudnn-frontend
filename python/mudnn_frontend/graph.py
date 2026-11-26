from __future__ import annotations

import ctypes
import json
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Mapping, Sequence

from . import _backend as _backend_constants
from ._version import __version__


class DataType(IntEnum):
    NOT_SET = -1
    FLOAT = 0
    DOUBLE = 1
    HALF = 2
    INT8 = 3
    INT32 = 4
    INT8x4 = 5
    UINT8 = 6
    UINT8x4 = 7
    INT8x32 = 8
    BFLOAT16 = 9
    INT64 = 10
    BOOLEAN = 11
    FP8_E4M3 = 12
    FP8_E5M2 = 13
    FAST_FLOAT_FOR_FP8 = 14
    FP8_E8M0 = 15
    FP4_E2M1 = 16
    INT4 = 17
    UINT4 = 18
    UINT32 = 19
    COMPLEX_FP32 = 20
    COMPLEX_FP64 = 21

    @classmethod
    def _missing_(cls, value):
        aliases = {
            "torch.float": cls.FLOAT,
            "torch.float32": cls.FLOAT,
            "torch.double": cls.DOUBLE,
            "torch.float64": cls.DOUBLE,
            "torch.half": cls.HALF,
            "torch.float16": cls.HALF,
            "torch.bfloat16": cls.BFLOAT16,
            "torch.int8": cls.INT8,
            "torch.int32": cls.INT32,
            "torch.int64": cls.INT64,
            "torch.uint8": cls.UINT8,
            "torch.bool": cls.BOOLEAN,
            "float32": cls.FLOAT,
            "float64": cls.DOUBLE,
            "float16": cls.HALF,
            "bfloat16": cls.BFLOAT16,
            "int8": cls.INT8,
            "int8x4": cls.INT8x4,
            "int8x32": cls.INT8x32,
            "int32": cls.INT32,
            "int64": cls.INT64,
            "uint8": cls.UINT8,
            "uint8x4": cls.UINT8x4,
            "bool": cls.BOOLEAN,
        }
        return aliases.get(str(value))


class PointwiseMode(IntEnum):
    ADD = 0
    MUL = 1
    MIN = 2
    MAX = 3
    SQRT = 4
    ADD_SQUARE = 5
    DIV = 6
    MOD = 7
    POW = 8
    SUB = 9
    ABS = 10
    CEIL = 11
    COS = 12
    EXP = 13
    FLOOR = 14
    LOG = 15
    NEG = 16
    SIN = 18
    RSQRT = 17
    TAN = 19
    ERF = 20
    IDENTITY = 21
    RECIPROCAL = 22
    RELU_FWD = 100
    TANH_FWD = 101
    SIGMOID_FWD = 102
    ELU_FWD = 103
    GELU_FWD = 104
    SOFTPLUS_FWD = 105
    SWISH_FWD = 106
    GELU_APPROX_TANH_FWD = 107
    RELU_BWD = 200
    TANH_BWD = 201
    SIGMOID_BWD = 202
    ELU_BWD = 203
    GELU_BWD = 204
    SOFTPLUS_BWD = 205
    SWISH_BWD = 206
    GELU_APPROX_TANH_BWD = 207
    CMP_EQ = 300
    CMP_NEQ = 301
    CMP_GT = 302
    CMP_GE = 303
    CMP_LT = 304
    CMP_LE = 305
    LOGICAL_AND = 400
    LOGICAL_OR = 401
    LOGICAL_NOT = 402
    GEN_INDEX = 501
    BINARY_SELECT = 601


def _maybe_data_type(value):
    try:
        return DataType(value)
    except (TypeError, ValueError):
        return None


_UNSUPPORTED_LOW_PRECISION_EXECUTION_DATA_TYPES = {
    DataType.FP8_E4M3,
    DataType.FP8_E5M2,
    DataType.FP8_E8M0,
    DataType.FP4_E2M1,
}

_SM100_RMSNORM_SILU_SUPPORTED_CHANNELS = {
    64,
    128,
    160,
    256,
    320,
    512,
    640,
    1024,
}
_SM100_RMSNORM_SILU_SUPPORTED_TOKENS = {
    1560,
    6240,
    24960,
    99840,
    399360,
}


def _public_backend_version_tuple() -> tuple[int, int, int]:
    version_fn = None
    cudnn_module = sys.modules.get("mudnn")
    if cudnn_module is not None:
        version_fn = getattr(cudnn_module, "backend_version_string", None)
    if version_fn is None:
        try:
            from . import backend_version_string as version_fn
        except Exception:
            version_fn = None
    if version_fn is None:
        return (0, 0, 0)
    try:
        raw_parts = str(version_fn()).split(".")
        parts = [int(part) for part in raw_parts[:3]]
        while len(parts) < 3:
            parts.append(0)
        return parts[0], parts[1], parts[2]
    except Exception:
        return (0, 0, 0)


class ReductionMode(IntEnum):
    NOT_SET = -1
    ADD = 0
    MUL = 1
    MIN = 2
    MAX = 3
    AMAX = 4
    AVG = 5
    NORM1 = 6
    NORM2 = 7
    MUL_NO_ZEROS = 8


class ConvolutionMode(IntEnum):
    CONVOLUTION = 0
    CROSS_CORRELATION = 1


class ReshapeMode(IntEnum):
    VIEW_ONLY = 0
    LOGICAL = 1


class ResampleMode(IntEnum):
    NEAREST = 0
    BILINEAR = 1
    AVGPOOL = 2
    AVGPOOL_INCLUDE_PADDING = 2
    MAXPOOL = 3
    AVGPOOL_EXCLUDE_PADDING = 4


class RngDistribution(IntEnum):
    BERNOULLI = 0
    UNIFORM = 1
    NORMAL = 2


class GenStatsMode(IntEnum):
    SUM_SQSUM = 0


class BnFinalizeStatsMode(IntEnum):
    TRAINING = 0
    INFERENCE = 1


class SignalMode(IntEnum):
    SET = 0
    WAIT = 1


class NormMode(IntEnum):
    LAYER_NORM = 0
    INSTANCE_NORM = 1
    BATCH_NORM = 2
    GROUP_NORM = 3
    RMS_NORM = 4
    ADA_LAYER_NORM = 5


class NormFwdPhase(IntEnum):
    NOT_SET = -1
    INFERENCE = 0
    TRAINING = 1


class MoeGroupedMatmulMode(IntEnum):
    NONE = 0
    GATHER = 1
    SCATTER = 2


class HeurMode(IntEnum):
    INSTANT = 0
    B = 1
    FALLBACK = 2
    A = 3
    OPENSOURCE = 2


class DiagonalAlignment(IntEnum):
    TOP_LEFT = 0
    BOTTOM_RIGHT = 1


class AttentionImplementation(IntEnum):
    AUTO = 0
    COMPOSITE = 1
    UNIFIED = 2


_ENUM_TYPES = {
    enum_type.__name__: enum_type
    for enum_type in (
        DataType,
        PointwiseMode,
        ReductionMode,
        ConvolutionMode,
        ReshapeMode,
        ResampleMode,
        RngDistribution,
        GenStatsMode,
        BnFinalizeStatsMode,
        SignalMode,
        NormMode,
        NormFwdPhase,
        MoeGroupedMatmulMode,
        HeurMode,
        DiagonalAlignment,
        AttentionImplementation,
    )
}


def _row_major_strides(dims: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for dim in reversed(dims):
        strides.append(stride)
        stride *= int(dim)
    return tuple(reversed(strides))


def _channels_last_strides(dims: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in dims)
    if len(dims) < 4:
        return _row_major_strides(dims)
    strides = [0] * len(dims)
    strides[1] = 1
    stride = dims[1]
    for dim in range(len(dims) - 1, 1, -1):
        strides[dim] = stride
        stride *= dims[dim]
    strides[0] = stride
    return tuple(strides)


def _is_channels_last_ncx(tensor: "GraphTensor") -> bool:
    if len(tensor.dims) < 4 or len(tensor.strides) != len(tensor.dims):
        return False
    return int(tensor.strides[1]) == 1 and tuple(tensor.strides) != _row_major_strides(
        tensor.dims
    )


def _broadcast_dims(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    result = []
    a_rev = tuple(reversed(a))
    b_rev = tuple(reversed(b))
    for i in range(max(len(a_rev), len(b_rev))):
        a_dim = a_rev[i] if i < len(a_rev) else 1
        b_dim = b_rev[i] if i < len(b_rev) else 1
        if a_dim <= 0 or b_dim <= 0:
            raise ValueError("broadcast dimensions must be positive")
        if a_dim != b_dim and a_dim != 1 and b_dim != 1:
            raise ValueError("pointwise inputs are not broadcastable")
        result.append(max(a_dim, b_dim))
    return tuple(reversed(result))


def _normalize_axis(axis: int, rank: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += int(rank)
    if normalized < 0 or normalized >= int(rank):
        raise ValueError("axis is out of range")
    return normalized


def _normalize_axes(axis: int | Sequence[int], rank: int, name: str) -> tuple[int, ...]:
    axes = (int(axis),) if isinstance(axis, int) else tuple(int(x) for x in axis)
    if not axes:
        raise ValueError(f"{name} axis must not be empty")
    normalized_axes = []
    for dim in axes:
        if dim < 0:
            dim += rank
        if dim < 0 or dim >= rank:
            raise ValueError(f"{name} axis is out of range")
        normalized_axes.append(dim)
    if len(set(normalized_axes)) != len(normalized_axes):
        raise ValueError(f"{name} axis contains duplicates")
    return tuple(normalized_axes)


def _resolve_axis_dim(
    axis: int | Sequence[int] | None,
    dim: int | Sequence[int] | None,
    name: str,
) -> int | Sequence[int] | None:
    def key(value: int | Sequence[int]) -> tuple[int, ...]:
        return (int(value),) if isinstance(value, int) else tuple(int(x) for x in value)

    if axis is not None and dim is not None and key(axis) != key(dim):
        raise ValueError(f"{name} axis and dim arguments conflict")
    return axis if axis is not None else dim


def _normalize_diagonal_alignment(
    value: DiagonalAlignment | int | str | None,
) -> DiagonalAlignment:
    if value is None:
        return DiagonalAlignment.TOP_LEFT
    if isinstance(value, str):
        name = value.upper()
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        try:
            return DiagonalAlignment[name]
        except KeyError as exc:
            raise ValueError(f"unsupported diagonal alignment: {value}") from exc
    try:
        return DiagonalAlignment(value)
    except ValueError as exc:
        raise ValueError(f"unsupported diagonal alignment: {value}") from exc


def _sdpa_diagonal_band_mask(
    score_shape: Sequence[int],
    device,
    alignment: DiagonalAlignment,
    left_bound: int | None,
    right_bound: int | None,
    diagonal_offsets=None,
):
    import torch

    seq_q = int(score_shape[-2])
    seq_k = int(score_shape[-1])
    batch = int(score_shape[0]) if len(score_shape) == 4 else 1
    base_shape = (seq_q, seq_k) if batch == 1 else (batch, 1, seq_q, seq_k)

    def scalar_offset(batch_index: int = 0) -> int:
        if alignment == DiagonalAlignment.TOP_LEFT:
            return 0
        if diagonal_offsets is None:
            return seq_k - seq_q
        if hasattr(diagonal_offsets, "reshape"):
            return int(diagonal_offsets.reshape(-1)[batch_index].item())
        return int(diagonal_offsets)

    masked = torch.zeros(base_shape, dtype=torch.bool, device=device)
    if right_bound is not None:
        if batch == 1:
            masked |= torch.ones(
                (seq_q, seq_k), dtype=torch.bool, device=device
            ).triu(diagonal=scalar_offset() + 1 + int(right_bound))
        else:
            for batch_index in range(batch):
                masked[batch_index, 0] |= torch.ones(
                    (seq_q, seq_k), dtype=torch.bool, device=device
                ).triu(
                    diagonal=scalar_offset(batch_index) + 1 + int(right_bound)
                )
    if left_bound is not None:
        left_row_keep = (
            torch.arange(seq_q, dtype=torch.int64, device=device)
            < seq_k + int(left_bound) - 1
        )
        if batch == 1:
            left_mask = torch.ones(
                (seq_q, seq_k), dtype=torch.bool, device=device
            ).tril(diagonal=scalar_offset() - int(left_bound))
            masked |= left_mask & left_row_keep.reshape(seq_q, 1)
        else:
            for batch_index in range(batch):
                left_mask = torch.ones(
                    (seq_q, seq_k), dtype=torch.bool, device=device
                ).tril(diagonal=scalar_offset(batch_index) - int(left_bound))
                masked[batch_index, 0] |= left_mask & left_row_keep.reshape(seq_q, 1)
    return masked


def _resolve_sdpa_causal_mask(
    op_name: str,
    causal_mask: bool,
    use_causal_mask: bool | None,
    diagonal_alignment: DiagonalAlignment | int | str | None,
    diagonal_band_left_bound: int | None,
    diagonal_band_right_bound: int | None,
) -> bool:
    if use_causal_mask is not None:
        if causal_mask and bool(use_causal_mask) != bool(causal_mask):
            raise ValueError(f"{op_name} causal_mask and use_causal_mask disagree")
        causal_mask = bool(use_causal_mask)

    if (
        diagonal_alignment is None
        and diagonal_band_left_bound is None
        and diagonal_band_right_bound is None
    ):
        return bool(causal_mask)

    alignment = _normalize_diagonal_alignment(diagonal_alignment)
    right_bound = (
        None if diagonal_band_right_bound is None else int(diagonal_band_right_bound)
    )
    band_requested = diagonal_band_left_bound is not None or right_bound is not None
    if not band_requested:
        return bool(causal_mask) and alignment == DiagonalAlignment.TOP_LEFT
    return (
        alignment == DiagonalAlignment.TOP_LEFT
        and diagonal_band_left_bound is None
        and right_bound == 0
    )


def _reject_unsupported_sdpa_options(op_name: str, **options):
    for name, value in options.items():
        if value is None:
            continue
        if name in ("use_causal_mask_bottom_right",) and bool(value) is False:
            continue
        if name in ("use_alibi_mask",) and (bool(value) is False or op_name == "sdpa"):
            continue
        if name == "unfuse_fma":
            continue
        if op_name in ("sdpa", "sdpa_backward") and name in (
            "seq_len_q",
            "seq_len_kv",
            "max_total_seq_len_q",
            "max_total_seq_len_kv",
        ):
            continue
        if name == "sliding_window_length" and value is None:
            continue
        if name == "dropout":
            if op_name == "sdpa":
                continue
            probability = value[0] if isinstance(value, tuple) else value
            if float(probability) == 0.0:
                continue
        if name == "implementation" and AttentionImplementation(value) in (
            AttentionImplementation.AUTO,
            AttentionImplementation.UNIFIED,
        ):
            continue
        if op_name == "sdpa_backward" and name in ("score_mod", "score_mod_bprop"):
            continue
        if op_name == "sdpa" and name == "block_mask":
            continue
        raise NotImplementedError(f"{op_name} does not support {name}")


def _resolve_tensor_alias(
    op_name: str,
    canonical_name: str,
    canonical_value,
    alias_name: str,
    alias_value,
):
    if canonical_value is not None and alias_value is not None:
        if canonical_value is not alias_value:
            raise ValueError(f"{op_name} {canonical_name} and {alias_name} disagree")
    return canonical_value if canonical_value is not None else alias_value


def _resolve_scalar_attr(op_name: str, attr_name: str, value) -> float:
    if isinstance(value, GraphTensor):
        raise NotImplementedError(
            f"{op_name} does not support tensor-valued {attr_name}"
        )
    return float(value)


def _check_scalar_float_tensor(op_name: str, attr_name: str, tensor: "GraphTensor"):
    if _num_elements(tensor.dims) != 1:
        raise ValueError(f"{op_name} {attr_name} tensor must contain one element")
    if tensor.data_type not in (DataType.FLOAT, DataType.DOUBLE):
        raise ValueError(f"{op_name} {attr_name} tensor data type must be FLOAT")


def _add_sdpa_scale_input(
    graph: "Graph",
    op_name: str,
    inputs: dict[str, "GraphTensor"],
    scale,
    scalar_name: str,
):
    if scale is None:
        return None
    if isinstance(scale, GraphTensor):
        _check_scalar_float_tensor(op_name, "attn_scale", scale)
        inputs["scale"] = scale
        return None
    inputs["scale"] = graph._new_scalar_by_value(scalar_name, float(scale))
    return float(scale)


def _check_sdpa_seq_len(name: str, tensor: "GraphTensor", batch: int):
    if tensor.data_type not in (DataType.INT32, DataType.INT64):
        raise ValueError(f"{name} data type must be INT32 or INT64")
    dims = tuple(int(dim) for dim in tensor.dims)
    batch = int(batch)
    if dims not in ((batch,), (batch, 1), (batch, 1, 1, 1)):
        raise ValueError(f"{name} dimensions must be batch-shaped")


def _num_elements(dims: Sequence[int]) -> int:
    total = 1
    for dim in dims:
        value = int(dim)
        if value <= 0:
            raise ValueError("tensor dimensions must be positive")
        total *= value
    return total


def _make_scalar_storage(value, data_type: DataType):
    data_type = DataType(data_type)
    if data_type == DataType.FLOAT:
        return ctypes.c_float(float(value))
    if data_type == DataType.DOUBLE:
        return ctypes.c_double(float(value))
    if data_type == DataType.INT64:
        return ctypes.c_int64(int(value))
    if data_type == DataType.INT32:
        return ctypes.c_int32(int(value))
    raise ValueError("tensor_scalar supports FLOAT, DOUBLE, INT32, and INT64")


def _scalar_type_name(value) -> str:
    if value is None:
        return "RUNTIME_PARAM"
    name = getattr(value, "name", None)
    if name in {"RUNTIME_PARAM", "COMPILE_TIME_CONST"}:
        return name
    if isinstance(value, str):
        normalized = value.upper()
        if normalized in {"RUNTIME_PARAM", "COMPILE_TIME_CONST"}:
            return normalized
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        as_int = None
    if as_int == 0:
        return "RUNTIME_PARAM"
    if as_int == 1:
        return "COMPILE_TIME_CONST"
    raise ValueError("scalar_type must be RUNTIME_PARAM or COMPILE_TIME_CONST")


def _scalar_data_type_from_value(value) -> DataType:
    if isinstance(value, float):
        return DataType.FLOAT
    if isinstance(value, int) and not isinstance(value, bool):
        return DataType.INT64
    return DataType.FLOAT


def _sequence_from_counted_args(
    args: tuple[object, ...], name: str
) -> tuple[object, ...]:
    if len(args) == 1:
        return tuple(args[0])
    if len(args) == 2:
        count = int(args[0])
        values = tuple(args[1])
        if count != len(values):
            raise ValueError(f"{name} count does not match values")
        return values
    raise TypeError(f"{name} expects values or count, values")


def _batch_norm_param_dims(x: "GraphTensor") -> tuple[int, int, int, int]:
    if len(x.dims) != 4:
        raise ValueError("batch_norm expects rank-4 NCHW tensors")
    return (1, x.dims[1], 1, 1)


def _check_batch_norm_param(name: str, tensor: "GraphTensor", x: "GraphTensor"):
    if tensor.dims != _batch_norm_param_dims(x):
        raise ValueError(f"batch_norm {name} dimensions must be (1, C, 1, 1)")
    if tensor.data_type not in (x.data_type, DataType.FLOAT):
        raise ValueError(f"batch_norm {name} data type must match x or be FLOAT")


def _norm_stat_dims(mode: NormMode, x: "GraphTensor") -> tuple[int, ...]:
    if not x.dims:
        raise ValueError("norm expects a non-scalar input tensor")
    if mode == NormMode.INSTANCE_NORM:
        if len(x.dims) != 4:
            raise ValueError("instancenorm expects rank-4 NCHW tensors")
        return (x.dims[0], x.dims[1], 1, 1)
    return (x.dims[0], *([1] * (len(x.dims) - 1)))


def _norm_param_dims(mode: NormMode, x: "GraphTensor") -> tuple[int, ...]:
    if not x.dims:
        raise ValueError("norm expects a non-scalar input tensor")
    if mode == NormMode.INSTANCE_NORM:
        if len(x.dims) != 4:
            raise ValueError("instancenorm expects rank-4 NCHW tensors")
        return (1, x.dims[1], 1, 1)
    return (1, *x.dims[1:])


def _check_norm_param(
    op_name: str, name: str, tensor: "GraphTensor", x: "GraphTensor", mode: NormMode
):
    expected = _norm_param_dims(mode, x)
    if tensor.dims != expected:
        raise ValueError(f"{op_name} {name} dimensions must be {expected}")
    if tensor.data_type != x.data_type:
        raise ValueError(f"{op_name} {name} data type must match input")


def _normalize_block_scale_block_size(
    x: "GraphTensor",
    block_size: int | Sequence[int],
    *,
    axis: int | None = None,
) -> int:
    return _normalize_block_scale_block_size_and_axis(x, block_size, axis=axis)[0]


def _normalize_block_scale_block_size_and_axis(
    x: "GraphTensor",
    block_size: int | Sequence[int],
    *,
    axis: int | None = None,
) -> tuple[int, int]:
    if isinstance(block_size, Sequence) and not isinstance(block_size, (str, bytes)):
        block_shape = tuple(int(value) for value in block_size)
        if not block_shape:
            raise ValueError("block_scale block_size must be non-empty")
        if any(value <= 0 for value in block_shape):
            raise ValueError("block_scale block_size must be positive")
        if len(block_shape) > len(x.dims):
            raise ValueError("block_scale block_size rank exceeds tensor rank")
        non_unit_axes = [
            len(x.dims) - len(block_shape) + index
            for index, value in enumerate(block_shape)
            if value != 1
        ]
        if not non_unit_axes:
            normalized_axis = (
                _normalize_axis(axis, len(x.dims))
                if axis is not None
                else len(x.dims) - 1
            )
            return 1, normalized_axis
        if len(non_unit_axes) != 1:
            raise NotImplementedError(
                "block_scale multi-axis block_size is not supported"
            )
        normalized_axis = non_unit_axes[0]
        if axis is not None and _normalize_axis(axis, len(x.dims)) != normalized_axis:
            raise ValueError("block_scale axis does not match block_size")
        return (
            block_shape[normalized_axis - (len(x.dims) - len(block_shape))],
            normalized_axis,
        )

    normalized = int(block_size)
    if normalized <= 0:
        raise ValueError("block_scale block_size must be positive")
    normalized_axis = (
        _normalize_axis(axis, len(x.dims)) if axis is not None else len(x.dims) - 1
    )
    return normalized, normalized_axis


def _block_scale_scale_dims(
    x: "GraphTensor", block_size: int | Sequence[int], *, axis: int | None = None
) -> tuple[int, ...]:
    block_size, normalized_axis = _normalize_block_scale_block_size_and_axis(
        x, block_size, axis=axis
    )
    if block_size <= 0:
        raise ValueError("block_scale block_size must be positive")
    if not x.dims:
        raise ValueError("block_scale expects a non-scalar tensor")
    dims = list(x.dims)
    dims[normalized_axis] = (dims[normalized_axis] + block_size - 1) // block_size
    return tuple(dims)


def _is_f8_128x4_reordered_scale(scale: "GraphTensor") -> bool:
    value = scale.reordering_type
    if isinstance(value, IntEnum):
        value = int(value)
    return value == 3


def _is_block_scale_dim_compatible(actual: int, expected: int) -> bool:
    if actual == expected:
        return True
    return actual >= expected and actual % 4 == 0


def _block_scale_scale_dims_match(
    scale: "GraphTensor",
    x: "GraphTensor",
    block_size: int | Sequence[int],
    *,
    axis: int | None = None,
) -> bool:
    expected = _block_scale_scale_dims(x, block_size, axis=axis)
    if scale.dims == expected:
        return True
    if not _is_f8_128x4_reordered_scale(scale):
        return False
    if len(scale.dims) != len(expected):
        return False
    return all(
        _is_block_scale_dim_compatible(actual, expect)
        for actual, expect in zip(scale.dims, expected)
    )


def _is_supported_block_scale_scale_data_type(data_type: DataType) -> bool:
    return DataType(data_type) in {
        DataType.FLOAT,
        DataType.FP8_E4M3,
        DataType.FP8_E5M2,
        DataType.FP8_E8M0,
    }


def _check_block_scale_scale(
    name: str,
    scale: "GraphTensor",
    x: "GraphTensor",
    block_size: int | Sequence[int],
    *,
    axis: int | None = None,
):
    if not _block_scale_scale_dims_match(scale, x, block_size, axis=axis):
        raise ValueError(f"block_scale {name} dimensions do not match block_size")
    if not _is_supported_block_scale_scale_data_type(scale.data_type):
        raise ValueError(f"block_scale {name} data type is not supported")


def _paged_cache_load_output_dims(
    container: "GraphTensor", sequence: "GraphTensor", page_table: "GraphTensor"
) -> tuple[int, ...]:
    if not container.dims:
        raise ValueError("paged_cache_load container must be non-scalar")
    if len(sequence.dims) != 1:
        raise ValueError("paged_cache_load sequence must be rank-1")
    if len(page_table.dims) != 1:
        raise ValueError("paged_cache_load page_table must be rank-1")
    if sequence.data_type != DataType.INT64 or page_table.data_type != DataType.INT64:
        raise ValueError("paged_cache_load sequence and page_table must be int64")
    return (sequence.dims[0], *container.dims[1:])


def _check_signal_flag(flag: "GraphTensor"):
    if flag.data_type != DataType.INT64:
        raise ValueError("signal flag data type must be int64")
    if _num_elements(flag.dims) != 1:
        raise ValueError("signal flag must contain exactly one element")


def _check_same_shape_type(name: str, tensor: "GraphTensor", reference: "GraphTensor"):
    if tensor.dims != reference.dims:
        raise ValueError(f"{name} dimensions must match reference")
    if tensor.data_type != reference.data_type:
        raise ValueError(f"{name} data type must match reference")


def _check_scalar_int64(name: str, tensor: "GraphTensor"):
    if tensor.data_type != DataType.INT64:
        raise ValueError(f"{name} data type must be int64")
    if _num_elements(tensor.dims) != 1:
        raise ValueError(f"{name} must contain exactly one element")


def _conv_output_spatial(
    input_spatial: Sequence[int],
    filter_spatial: Sequence[int],
    pre_padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    post_padding: Sequence[int] | None = None,
) -> tuple[int, ...]:
    post = pre_padding if post_padding is None else post_padding
    return tuple(
        (
            int(input_spatial[index])
            + int(pre_padding[index])
            + int(post[index])
            - int(dilation[index]) * (int(filter_spatial[index]) - 1)
            - 1
        )
        // int(stride[index])
        + 1
        for index in range(len(input_spatial))
    )


def _conv_input_spatial_from_output(
    output_spatial: Sequence[int],
    filter_spatial: Sequence[int],
    pre_padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    post_padding: Sequence[int] | None = None,
) -> tuple[int, ...]:
    post = pre_padding if post_padding is None else post_padding
    return tuple(
        (out_dim - 1) * stride_dim
        - pre_pad_dim
        - post_pad_dim
        + dilation_dim * (filter_dim - 1)
        + 1
        for out_dim, filter_dim, pre_pad_dim, post_pad_dim, stride_dim, dilation_dim in zip(
            output_spatial, filter_spatial, pre_padding, post, stride, dilation
        )
    )


def _infer_conv_filter_spatial(
    input_spatial: Sequence[int],
    output_spatial: Sequence[int],
    pre_padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
    post_padding: Sequence[int] | None = None,
) -> tuple[int, ...]:
    post = pre_padding if post_padding is None else post_padding
    values = []
    for index in range(len(input_spatial)):
        input_dim = int(input_spatial[index])
        output_dim = int(output_spatial[index])
        pre_pad_dim = int(pre_padding[index])
        post_pad_dim = int(post[index])
        stride_dim = int(stride[index])
        dilation_dim = int(dilation[index])
        if input_dim <= 0 or output_dim <= 0 or stride_dim <= 0 or dilation_dim <= 0:
            raise ValueError("conv_wgrad could not infer filter dimensions")
        kernel_extent = (
            input_dim + pre_pad_dim + post_pad_dim - (output_dim - 1) * stride_dim - 1
        )
        if kernel_extent >= 0 and kernel_extent % dilation_dim == 0:
            kernel = kernel_extent // dilation_dim + 1
            if kernel > 0:
                values.append(kernel)
                continue
        max_kernel = (input_dim + pre_pad_dim + post_pad_dim - 1) // dilation_dim + 1
        matched = None
        for kernel in range(1, max_kernel + 1):
            actual = (
                input_dim + pre_pad_dim + post_pad_dim - dilation_dim * (kernel - 1) - 1
            ) // stride_dim + 1
            if actual == output_dim:
                matched = kernel
                break
        if matched is None:
            raise ValueError("conv_wgrad could not infer filter dimensions")
        values.append(matched)
    return tuple(values)


def _normalize_conv_params(
    spatial_rank: int,
    padding: Sequence[int],
    stride: Sequence[int],
    dilation: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    pad = tuple(int(v) for v in padding)
    strides = tuple(int(v) for v in stride)
    dilations = tuple(int(v) for v in dilation)
    if (
        len(pad) != spatial_rank
        or len(strides) != spatial_rank
        or len(dilations) != spatial_rank
    ):
        raise ValueError(
            "convolution padding, stride, and dilation must be " f"rank-{spatial_rank}"
        )
    if any(v < 0 for v in pad) or any(v <= 0 for v in strides + dilations):
        raise ValueError(
            "convolution padding must be nonnegative; stride/dilation positive"
        )
    return pad, strides, dilations


def _resolve_conv_padding(
    spatial_rank: int,
    padding: Sequence[int] | None,
    pre_padding: Sequence[int] | None,
    post_padding: Sequence[int] | None,
) -> Sequence[int]:
    default = tuple(0 for _ in range(spatial_rank))
    if pre_padding is None and post_padding is None:
        return default if padding is None else padding
    base = default if padding is None else tuple(int(v) for v in padding)
    pre = base if pre_padding is None else tuple(int(v) for v in pre_padding)
    post = base if post_padding is None else tuple(int(v) for v in post_padding)
    if padding is not None and (pre != base or post != base):
        raise ValueError("convolution padding conflicts with pre/post padding")
    if pre != post:
        raise NotImplementedError(
            "asymmetric convolution pre/post padding is not supported"
        )
    return pre


def _resolve_conv_paddings(
    spatial_rank: int,
    padding: Sequence[int] | None,
    pre_padding: Sequence[int] | None,
    post_padding: Sequence[int] | None,
) -> tuple[Sequence[int], Sequence[int]]:
    default = tuple(0 for _ in range(spatial_rank))
    if pre_padding is None and post_padding is None:
        resolved = default if padding is None else padding
        return resolved, resolved
    base = default if padding is None else tuple(int(v) for v in padding)
    pre = base if pre_padding is None else tuple(int(v) for v in pre_padding)
    post = base if post_padding is None else tuple(int(v) for v in post_padding)
    if padding is not None and (pre != base or post != base):
        raise ValueError("convolution padding conflicts with pre/post padding")
    return pre, post


def _normalize_resample2d_params(
    window: Sequence[int], padding: Sequence[int], stride: Sequence[int]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    windows = tuple(int(v) for v in window)
    pad = tuple(int(v) for v in padding)
    strides = tuple(int(v) for v in stride)
    if len(windows) != 2 or len(pad) != 2 or len(strides) != 2:
        raise ValueError("resample window, padding, and stride must be rank-2")
    if any(v <= 0 for v in windows + strides) or any(v < 0 for v in pad):
        raise ValueError("resample window/stride must be positive; padding nonnegative")
    return windows, pad, strides


def _moe_first_token_offset_experts(
    first_token_offset: "GraphTensor", op_name: str
) -> int:
    if len(first_token_offset.dims) == 1:
        return int(first_token_offset.dims[0])
    if len(first_token_offset.dims) == 3 and tuple(first_token_offset.dims[1:]) == (
        1,
        1,
    ):
        return int(first_token_offset.dims[0])
    raise ValueError(f"{op_name} first_token_offset must be rank-1 or [experts, 1, 1]")


def _resample2d_output_spatial(
    input_spatial: Sequence[int],
    window: Sequence[int],
    padding: Sequence[int],
    stride: Sequence[int],
) -> tuple[int, int]:
    out_h = (int(input_spatial[0]) + 2 * int(padding[0]) - int(window[0])) // int(
        stride[0]
    ) + 1
    out_w = (int(input_spatial[1]) + 2 * int(padding[1]) - int(window[1])) // int(
        stride[1]
    ) + 1
    return out_h, out_w


def _pointer_value(value) -> int:
    if isinstance(value, ctypes.c_void_p):
        if value.value is None:
            raise ValueError("null data pointer")
        return int(value.value)
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "data_ptr"):
        return int(value.data_ptr())
    cuda_array_interface = getattr(value, "__cuda_array_interface__", None)
    if cuda_array_interface is not None:
        return int(cuda_array_interface["data"][0])
    array_interface = getattr(value, "__array_interface__", None)
    if array_interface is not None:
        return int(array_interface["data"][0])
    return int(value)


def _torch_dtype_for_data_type(data_type: DataType):
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("automatic output allocation requires torch") from exc

    mapping = {
        DataType.FLOAT: torch.float32,
        DataType.DOUBLE: torch.float64,
        DataType.HALF: torch.float16,
        DataType.BFLOAT16: torch.bfloat16,
        DataType.INT8: torch.int8,
        DataType.INT32: torch.int32,
        DataType.INT64: torch.int64,
        DataType.UINT8: torch.uint8,
        DataType.BOOLEAN: torch.bool,
    }
    for data_type_name, torch_name in (
        (DataType.FP8_E4M3, "float8_e4m3fn"),
        (DataType.FP8_E5M2, "float8_e5m2"),
        (DataType.FP8_E8M0, "float8_e8m0fnu"),
        (DataType.FP4_E2M1, "float4_e2m1fn_x2"),
    ):
        if hasattr(torch, torch_name):
            mapping[data_type_name] = getattr(torch, torch_name)
    mapping.setdefault(DataType.FP8_E8M0, torch.uint8)
    mapping.setdefault(DataType.FP4_E2M1, torch.uint8)
    data_type = DataType(data_type)
    if data_type not in mapping:
        raise NotImplementedError(
            f"automatic output allocation does not support {data_type.name}"
        )
    return mapping[data_type]


def _data_type_size(data_type: DataType) -> int:
    sizes = {
        DataType.FLOAT: 4,
        DataType.DOUBLE: 8,
        DataType.HALF: 2,
        DataType.BFLOAT16: 2,
        DataType.INT8: 1,
        DataType.UINT8: 1,
        DataType.INT32: 4,
        DataType.INT64: 8,
        DataType.BOOLEAN: 1,
        DataType.FP8_E4M3: 1,
        DataType.FP8_E5M2: 1,
        DataType.FP8_E8M0: 1,
        DataType.FP4_E2M1: 1,
    }
    return sizes[DataType(data_type)]


def _strided_storage_elements(dims: Sequence[int], strides: Sequence[int]) -> int:
    if not dims:
        return 0
    max_offset = 0
    for dim, stride in zip(dims, strides):
        if int(dim) <= 0:
            return 0
        max_offset += (int(dim) - 1) * int(stride)
    return int(max_offset) + 1


def _infer_torch_output_device(inputs: Sequence[object], output_tensor: "GraphTensor"):
    if output_tensor.device is not None:
        return output_tensor.device
    for value in inputs:
        device = getattr(value, "device", None)
        if device is not None:
            return device
    raise ValueError(
        "automatic output allocation requires tensor inputs or output tensor device"
    )


def get_data_ptr(value) -> int:
    return _pointer_value(value)


def _normalize_heur_modes(modes) -> tuple["HeurMode", ...]:
    if modes is None:
        return (HeurMode.INSTANT,)
    if isinstance(modes, (HeurMode, int)):
        return (HeurMode(modes),)
    return tuple(HeurMode(mode) for mode in modes)


def _serialize_value(value):
    if isinstance(value, IntEnum):
        return {
            "__mudnn_frontend_enum__": value.__class__.__name__,
            "value": int(value),
        }
    if isinstance(value, tuple):
        return {"__mudnn_frontend_tuple__": [_serialize_value(item) for item in value]}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    return value


def _deserialize_value(value):
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    if isinstance(value, dict):
        enum_name = value.get("__mudnn_frontend_enum__")
        if enum_name is not None:
            enum_type = _ENUM_TYPES.get(enum_name)
            if enum_type is None:
                raise ValueError(f"unknown serialized enum type: {enum_name}")
            return enum_type(value["value"])
        if "__mudnn_frontend_tuple__" in value:
            return tuple(
                _deserialize_value(item) for item in value["__mudnn_frontend_tuple__"]
            )
        return {key: _deserialize_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Tensor:
    uid: int
    dims: tuple[int, ...]
    strides: tuple[int, ...]
    data_type: DataType = DataType.FLOAT
    alignment: int = 16
    is_virtual: bool = False
    is_by_value: bool = False


class TensorBuilder:
    def __init__(self):
        self._uid = 0
        self._dims: tuple[int, ...] = ()
        self._strides: tuple[int, ...] = ()
        self._data_type = DataType.FLOAT
        self._alignment = 16
        self._is_virtual = False
        self._is_by_value = False

    def setId(self, uid: int):
        self._uid = int(uid)
        return self

    def setDim(self, *args):
        self._dims = tuple(int(x) for x in _sequence_from_counted_args(args, "dim"))
        return self

    def setStrides(self, *args):
        self._strides = tuple(
            int(x) for x in _sequence_from_counted_args(args, "stride")
        )
        return self

    def setStride(self, *args):
        return self.setStrides(*args)

    def setVectorCountAndDimension(self, vector_count: int, vector_dimension: int):
        return self

    def setReorderType(self, reordering_type):
        return self

    def setDataType(self, data_type: DataType):
        self._data_type = DataType(data_type)
        return self

    def setAlignment(self, alignment: int):
        self._alignment = int(alignment)
        return self

    def setVirtual(self, is_virtual: bool):
        self._is_virtual = bool(is_virtual)
        return self

    def setByValue(self, is_by_value: bool):
        self._is_by_value = bool(is_by_value)
        return self

    def build(self) -> Tensor:
        if len(self._dims) != len(self._strides):
            raise ValueError("dims and strides must have the same rank")
        return Tensor(
            uid=self._uid,
            dims=self._dims,
            strides=self._strides,
            data_type=self._data_type,
            alignment=self._alignment,
            is_virtual=self._is_virtual,
            is_by_value=self._is_by_value,
        )


@dataclass
class GraphTensor:
    uid: int
    name: str
    dims: tuple[int, ...]
    strides: tuple[int, ...]
    data_type: DataType
    is_virtual: bool = False
    is_output: bool = False
    is_by_value: bool = False
    device: object | None = None
    reordering_type: object | None = None
    alignment: int = 16
    vector_count: int = 1
    vector_dimension: int = -1
    ragged_offset: "GraphTensor | None" = None
    ragged_offset_multiplier: int = 1
    scalar_type: str | None = None

    def __hash__(self):
        return hash(self.uid)

    def set_data_type(self, data_type: DataType):
        self.data_type = DataType(data_type)
        return self

    def _set_data_type(self, data_type: DataType):
        return self.set_data_type(data_type)

    def set_dim(self, dim: Sequence[int]):
        self.dims = tuple(int(x) for x in dim)
        if len(self.strides) != len(self.dims):
            self.strides = _row_major_strides(self.dims)
        graph = getattr(self, "_graph", None)
        if graph is not None:
            graph._on_tensor_dim_changed(self)
        return self

    def set_stride(self, stride: Sequence[int]):
        strides = tuple(int(x) for x in stride)
        if len(strides) != len(self.dims):
            raise ValueError("tensor dim and stride must have the same rank")
        self.strides = strides
        return self

    def set_output(self, is_output: bool = True):
        self.is_output = bool(is_output)
        if self.is_output:
            self.is_virtual = False
        return self

    def set_name(self, name: str):
        self.name = str(name)
        return self

    def set_uid(self, uid: int):
        self.uid = int(uid)
        graph = getattr(self, "_graph", None)
        if graph is not None:
            graph._next_uid = max(graph._next_uid, self.uid + 1)
        return self

    def set_is_pass_by_value(self, is_by_value: bool = True):
        self.is_by_value = bool(is_by_value)
        return self

    def set_ragged_offset(
        self,
        ragged_offset: "GraphTensor | None",
        ragged_offset_multiplier: int = 1,
    ):
        self.ragged_offset = ragged_offset
        self.ragged_offset_multiplier = int(ragged_offset_multiplier)
        return self

    def set_ragged_offset_multiplier(self, ragged_offset_multiplier: int):
        multiplier = int(ragged_offset_multiplier)
        if multiplier <= 0:
            raise ValueError("ragged offset multiplier must be positive")
        self.ragged_offset_multiplier = multiplier
        return self

    def set_reordering_type(self, reordering_type):
        self.reordering_type = reordering_type
        return self

    def set_alignment(self, alignment: int):
        alignment = int(alignment)
        if alignment <= 0:
            raise ValueError("tensor alignment must be positive")
        self.alignment = alignment
        return self

    def set_vector_count_and_dimension(self, vector_count: int, vector_dimension: int):
        vector_count = int(vector_count)
        vector_dimension = int(vector_dimension)
        if vector_count <= 0:
            raise ValueError("tensor vector count must be positive")
        self.vector_count = vector_count
        self.vector_dimension = vector_dimension
        return self

    def get_uid(self) -> int:
        return int(self.uid)

    def get_name(self) -> str:
        return self.name

    def get_dim(self) -> tuple[int, ...]:
        return self.dims

    def get_stride(self) -> tuple[int, ...]:
        return self.strides

    def get_data_type(self) -> DataType:
        return self.data_type

    def get_is_virtual(self) -> bool:
        return bool(self.is_virtual)

    def get_is_pass_by_value(self) -> bool:
        return bool(self.is_by_value)

    def get_has_compile_time_constant(self) -> bool:
        return self.scalar_type == "compile_time_const"

    def get_reordering_type(self):
        return self.reordering_type

    def get_alignment(self) -> int:
        return int(self.alignment)

    def get_vector_count(self) -> int:
        return int(self.vector_count)

    def get_vector_dimension(self) -> int:
        return int(self.vector_dimension)


@dataclass(frozen=True)
class GraphNode:
    op_type: str
    name: str
    inputs: Mapping[str, GraphTensor]
    outputs: Mapping[str, GraphTensor]
    attributes: Mapping[str, object] = field(default_factory=dict)


class Graph:
    """Small mudnn-frontend-style symbolic graph API for muDNN frontend.

    This mirrors the Python graph-construction surface used by mudnn-frontend.
    Execution is intentionally left to the backend binding layer.
    """

    def __init__(
        self,
        name: str = "test_graph",
        io_data_type: DataType = DataType.NOT_SET,
        intermediate_data_type: DataType = DataType.NOT_SET,
        compute_data_type: DataType = DataType.NOT_SET,
        handle=None,
        sm_count=None,
        sm_version=None,
        kernel_cache=None,
        device_property=None,
        is_dynamic_shape_enabled: bool = False,
        is_override_shape_enabled: bool = False,
        *,
        device_properties=None,
        inputs: Sequence[object] | None = None,
        outputs: Sequence[object] | None = None,
        heuristics: Iterable[HeurMode] | None = None,
        workspace_alloc: bool = True,
    ):
        if inputs is not None and not isinstance(inputs, (list, tuple)):
            raise ValueError("inputs must be a list or tuple")
        if outputs is not None and not isinstance(outputs, (list, tuple)):
            raise ValueError("outputs must be a list or tuple")
        if device_property is not None and device_properties is not None:
            raise ValueError(
                "pass either device_property or device_properties, not both"
            )
        self.io_data_type = DataType(io_data_type)
        self.intermediate_data_type = DataType(intermediate_data_type)
        self.compute_data_type = DataType(compute_data_type)
        self.tensors: list[GraphTensor] = []
        self.nodes: list[GraphNode] = []
        self.execution_plan_modes: tuple[HeurMode, ...] = ()
        self._next_uid = 1
        self._validated = False
        self._operation_graph_built = False
        self._plans_built = False
        self._native = None
        self._runtime_external_uids: set[int] | None = None
        self._lowered_external_uids: set[int] | None = None
        self.handle = handle
        self._by_value_data: dict[int, object] = {}
        self.name = str(name)
        self._context_heuristics = _normalize_heur_modes(heuristics)
        self._workspace_alloc = bool(workspace_alloc)
        self._pending_io_inputs = tuple(inputs or ())
        self._pending_io_outputs = tuple(outputs or ())
        self._input_tuples: tuple[GraphTensor, ...] | None = None
        self._output_tuples: tuple[GraphTensor, ...] | None = None
        self.kernel_cache = kernel_cache
        self.device_property = (
            device_property if device_property is not None else device_properties
        )
        self.sm_count = None if sm_count is None else int(sm_count)
        self.sm_version = None if sm_version is None else int(sm_version)
        self.is_dynamic_shape_enabled = bool(is_dynamic_shape_enabled)
        self.is_override_shape_enabled = bool(is_override_shape_enabled)
        self._deselected_engine_names: tuple[str, ...] = ()
        self._max_workspace = (1 << 63) - 1
        self._max_shared_memory = (1 << 63) - 1
        self._selected_numeric_notes: tuple[int, ...] = ()
        self._deselected_numeric_notes: tuple[int, ...] = ()
        self._selected_behavior_notes: tuple[int, ...] = ()
        self._deselected_behavior_notes: tuple[int, ...] = ()

    def _effective_io_data_type(self) -> DataType:
        # A graph whose io data type was left at the cuDNN 0.0.1 NOT_SET
        # default behaves like the historical FLOAT default once tensors are
        # created through it.
        if self.io_data_type == DataType.NOT_SET:
            return DataType.FLOAT
        return self.io_data_type

    def _invalidate_lowering(self):
        self._operation_graph_built = False
        self._plans_built = False
        self._native = None
        self._lowered_external_uids = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return False
        if self._pending_io_inputs or self._pending_io_outputs:
            self.set_io_tuples(self._pending_io_inputs, self._pending_io_outputs)
        if self.nodes and not self._plans_built:
            self.build(self._context_heuristics)
        return False

    def _io_tensor_map(
        self,
    ) -> tuple[dict[object, GraphTensor], dict[object, GraphTensor]]:
        inputs: dict[object, GraphTensor] = {}
        outputs: dict[object, GraphTensor] = {}
        for tensor in self.tensors:
            inputs[tensor] = tensor
            outputs[tensor] = tensor
            inputs[tensor.uid] = tensor
            outputs[tensor.uid] = tensor
            inputs[str(tensor.uid)] = tensor
            outputs[str(tensor.uid)] = tensor
            inputs[tensor.name] = tensor
            outputs[tensor.name] = tensor
        for node in self.nodes:
            for index, (key, tensor) in enumerate(node.inputs.items()):
                inputs[f"{node.name}::{key}"] = tensor
                inputs[f"{node.name}::{index}"] = tensor
            for index, (key, tensor) in enumerate(node.outputs.items()):
                outputs[f"{node.name}::{key}"] = tensor
                outputs[f"{node.name}::{index}"] = tensor
        return inputs, outputs

    def _resolve_io_tensor(
        self,
        value: object,
        mapping: Mapping[object, GraphTensor],
        kind: str,
        index: int,
    ) -> GraphTensor:
        if isinstance(value, GraphTensor):
            return value
        if value in mapping:
            return mapping[value]
        raise ValueError(f"{kind} at index {index} ({value}) not found in tensor map")

    def set_io_tuples(
        self,
        inputs: Sequence[object],
        outputs: Sequence[object],
    ):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("inputs must be a list or tuple")
        if not isinstance(outputs, (list, tuple)):
            raise ValueError("outputs must be a list or tuple")
        if not self.tensors:
            self._pending_io_inputs = tuple(inputs)
            self._pending_io_outputs = tuple(outputs)
            return self

        input_map, output_map = self._io_tensor_map()
        input_tensors = tuple(
            self._resolve_io_tensor(value, input_map, "Input", index)
            for index, value in enumerate(inputs)
        )
        output_tensors = tuple(
            self._resolve_io_tensor(value, output_map, "Output", index)
            for index, value in enumerate(outputs)
        )

        seen: set[int] = set()
        for index, tensor in enumerate(input_tensors):
            if tensor.uid in seen:
                raise ValueError(
                    f"Input at index {index} ({tensor.name}) is a duplicate"
                )
            seen.add(tensor.uid)
            if tensor.is_virtual:
                raise ValueError(f"Input at index {index} is a virtual tensor")
        for index, tensor in enumerate(output_tensors):
            if tensor.uid in seen:
                raise ValueError(
                    f"Output at index {index} ({tensor.name}) is a duplicate"
                )
            seen.add(tensor.uid)

        explicit_output_uids = {tensor.uid for tensor in output_tensors}
        produced_info = self._produced_tensor_info()
        internal_uids = {
            uid
            for uid, info in produced_info.items()
            if uid not in explicit_output_uids and not info["tensor"].is_output
        }
        for tensor in self.tensors:
            if tensor.is_virtual or tensor.uid in internal_uids:
                continue
            if tensor.uid not in seen and not tensor.is_by_value:
                raise ValueError(
                    f"Tensor {tensor.name} is non-virtual but not specified as input or output"
                )

        self._pending_io_inputs = tuple(inputs)
        self._pending_io_outputs = tuple(outputs)
        self._input_tuples = input_tensors
        self._output_tuples = output_tensors
        self._invalidate_lowering()
        return self

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Mapping):
            return self.execute(args[0], **kwargs)
        if self._input_tuples is None:
            raise ValueError(
                "positional graph execution requires set_io_tuples() first"
            )
        if len(args) != len(self._input_tuples):
            raise ValueError(
                f"Number of arguments ({len(args)}) does not match number of inputs "
                f"({len(self._input_tuples)})"
            )
        output_ptrs = kwargs.pop("output_ptrs", None)
        if output_ptrs is None:
            output_ptrs = kwargs.pop("outputs", None)
        allocated_outputs = None
        if self._output_tuples and output_ptrs is None:
            try:
                import torch
            except ImportError as exc:
                raise NotImplementedError(
                    "automatic output allocation requires torch"
                ) from exc
            allocated_outputs = tuple(
                torch.empty_strided(
                    tuple(tensor.dims),
                    tuple(tensor.strides),
                    dtype=_torch_dtype_for_data_type(tensor.data_type),
                    device=_infer_torch_output_device(args, tensor),
                )
                for tensor in self._output_tuples
            )
            output_ptrs = allocated_outputs
        if output_ptrs is None:
            output_ptrs = ()
        elif len(self._output_tuples) == 1 and not isinstance(
            output_ptrs, (list, tuple)
        ):
            output_ptrs = (output_ptrs,)
        else:
            output_ptrs = tuple(output_ptrs)
        if len(output_ptrs) != len(self._output_tuples or ()):
            raise ValueError(
                f"Number of output pointers ({len(output_ptrs)}) does not match "
                f"number of outputs ({len(self._output_tuples or ())})"
            )
        variant_pack = {
            tensor.uid: pointer for tensor, pointer in zip(self._input_tuples, args)
        }
        variant_pack.update(
            {
                tensor.uid: pointer
                for tensor, pointer in zip(self._output_tuples or (), output_ptrs)
            }
        )
        self.execute(variant_pack, **kwargs)
        if allocated_outputs is not None:
            if len(allocated_outputs) == 1:
                return allocated_outputs[0]
            return allocated_outputs
        return self

    def set_io_data_type(self, data_type: DataType):
        self.io_data_type = DataType(data_type)
        return self

    def set_intermediate_data_type(self, data_type: DataType):
        self.intermediate_data_type = DataType(data_type)
        return self

    def set_compute_data_type(self, data_type: DataType):
        self.compute_data_type = DataType(data_type)
        return self

    def tensor(
        self,
        *,
        name: str | None = None,
        dim: Sequence[int] | None = None,
        stride: Sequence[int] | None = None,
        data_type: DataType | None = None,
        dtype: DataType | None = None,
        is_virtual: bool = False,
        is_pass_by_value: bool = False,
        uid: int | None = None,
        device=None,
        reordering_type=None,
        alignment: int = 16,
        vector_count: int = 1,
        vector_dimension: int = -1,
    ) -> GraphTensor:
        if dim is None:
            raise ValueError("tensor dim is required")
        if data_type is not None and dtype is not None:
            if DataType(data_type) != DataType(dtype):
                raise ValueError("tensor data_type and dtype must match")
        resolved_data_type = data_type if data_type is not None else dtype
        dims = tuple(int(x) for x in dim)
        strides = (
            tuple(int(x) for x in stride)
            if stride is not None
            else _row_major_strides(dims)
        )
        if len(dims) != len(strides):
            raise ValueError("tensor dim and stride must have the same rank")
        tensor_uid = int(uid) if uid is not None else self._next_uid
        tensor = GraphTensor(
            uid=tensor_uid,
            name=name or f"tensor_{tensor_uid}",
            dims=dims,
            strides=strides,
            data_type=(
                self._effective_io_data_type()
                if resolved_data_type is None
                or DataType(resolved_data_type) == DataType.NOT_SET
                else DataType(resolved_data_type)
            ),
            is_virtual=bool(is_virtual),
            is_by_value=bool(is_pass_by_value),
            device=device,
            reordering_type=reordering_type,
            alignment=int(alignment),
            vector_count=int(vector_count),
            vector_dimension=int(vector_dimension),
        )
        tensor._graph = self
        self._next_uid = max(self._next_uid, tensor_uid + 1)
        self.tensors.append(tensor)
        return tensor

    def _on_tensor_dim_changed(self, tensor: GraphTensor):
        for node in self.nodes:
            if node.op_type != "conv_bwd_filter":
                continue
            if node.outputs.get("dw") is not tensor:
                continue
            x = node.inputs["x"]
            dy = node.inputs["dy"]
            w_dims = tuple(int(v) for v in tensor.dims)
            if len(w_dims) != len(x.dims):
                continue
            if w_dims[1] <= 0 or x.dims[1] % w_dims[1] != 0:
                continue
            group_count = x.dims[1] // w_dims[1]
            if group_count <= 0 or dy.dims[1] != w_dims[0]:
                continue
            if w_dims[0] % group_count != 0:
                continue
            spatial_rank = len(x.dims) - 2
            pre_pad, strides, dilations = _normalize_conv_params(
                spatial_rank,
                node.attributes["pre_padding"],
                node.attributes["stride"],
                node.attributes["dilation"],
            )
            post_pad, _, _ = _normalize_conv_params(
                spatial_rank,
                node.attributes["post_padding"],
                strides,
                dilations,
            )
            expected_spatial = _conv_output_spatial(
                x.dims[2:], w_dims[2:], pre_pad, strides, dilations, post_pad
            )
            if expected_spatial != dy.dims[2:]:
                continue
            node.attributes["group_count"] = group_count

    def _make_tensor(
        self,
        dim: Sequence[int],
        stride: Sequence[int],
        data_type: DataType | None = None,
        is_virtual: bool = False,
        is_pass_by_value: bool = False,
        ragged_offset=None,
        reordering_type=None,
        alignment: int = 16,
        vector_count: int = 1,
        vector_dimension: int = -1,
        name: str | None = None,
        uid: int = -1,
        ragged_offset_multiplier: int = 1,
    ) -> GraphTensor:
        tensor = self.tensor(
            name=name or None,
            dim=dim,
            stride=stride,
            data_type=data_type,
            is_virtual=is_virtual,
            is_pass_by_value=is_pass_by_value,
            uid=uid if int(uid) >= 0 else None,
            reordering_type=reordering_type,
            alignment=alignment,
            vector_count=vector_count,
            vector_dimension=vector_dimension,
        )
        if ragged_offset is not None:
            tensor.set_ragged_offset(ragged_offset, ragged_offset_multiplier)
        elif int(ragged_offset_multiplier) != 1:
            tensor.set_ragged_offset_multiplier(ragged_offset_multiplier)
        return tensor

    def tensor_like(
        self,
        value=None,
        *,
        input=None,
        name: str | None = None,
        data_type: DataType | None = None,
        is_virtual: bool = False,
        is_pass_by_value: bool = False,
        uid: int | None = None,
        ragged_offset_multiplier: int | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        value = value if value is not None else input
        if data_type is None and compute_data_type is not None:
            data_type = compute_data_type
        if isinstance(value, GraphTensor):
            tensor = self.tensor(
                name=name,
                dim=value.dims,
                stride=value.strides,
                data_type=data_type or value.data_type,
                is_virtual=is_virtual,
                is_pass_by_value=is_pass_by_value,
                uid=uid,
                device=value.device,
                reordering_type=value.reordering_type,
                alignment=value.alignment,
                vector_count=value.vector_count,
                vector_dimension=value.vector_dimension,
            )
            if value.ragged_offset is not None:
                tensor.set_ragged_offset(
                    value.ragged_offset,
                    (
                        value.ragged_offset_multiplier
                        if ragged_offset_multiplier is None
                        else ragged_offset_multiplier
                    ),
                )
            elif ragged_offset_multiplier is not None:
                tensor.set_ragged_offset_multiplier(ragged_offset_multiplier)
            return tensor
        if not hasattr(value, "shape"):
            raise ValueError("tensor_like value must expose shape")
        dims = tuple(int(x) for x in value.shape)
        if hasattr(value, "stride") and callable(value.stride):
            strides = tuple(int(x) for x in value.stride())
        else:
            itemsize = int(getattr(getattr(value, "dtype", None), "itemsize", 1))
            if hasattr(value, "strides") and value.strides is not None and itemsize > 0:
                strides = tuple(int(x) // itemsize for x in value.strides)
            else:
                strides = _row_major_strides(dims)
        resolved_data_type = (
            data_type
            if data_type is not None
            else _maybe_data_type(getattr(value, "dtype", None))
        )
        return self.tensor(
            name=name,
            dim=dims,
            stride=strides,
            data_type=resolved_data_type,
            is_virtual=is_virtual,
            is_pass_by_value=is_pass_by_value,
            uid=uid,
            device=getattr(value, "device", None),
        )

    def tensor_scalar(
        self,
        value,
        scalar_type=None,
        *,
        data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        legacy_name = getattr(scalar_type, "name", None)
        if legacy_name in {"INT32", "INT64", "FLOAT", "DOUBLE"}:
            scalar_kind = "RUNTIME_PARAM"
        else:
            scalar_kind = _scalar_type_name(scalar_type)
        if data_type is None:
            if legacy_name in {"INT32", "INT64", "FLOAT", "DOUBLE"}:
                data_type = getattr(DataType, legacy_name)
            else:
                data_type = _scalar_data_type_from_value(value)
        data_type = DataType(data_type)
        tensor = self.tensor(
            name=name or f"scalar_{self._next_uid}",
            dim=(1,),
            data_type=data_type,
            is_pass_by_value=True,
        )
        tensor.scalar_type = scalar_kind
        self._by_value_data[tensor.uid] = _make_scalar_storage(value, data_type)
        return tensor

    def _new_virtual(
        self,
        name: str,
        like: GraphTensor,
        dims: Sequence[int] | None = None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        output_dims = tuple(int(dim) for dim in dims) if dims is not None else like.dims
        return self.tensor(
            name=name,
            dim=output_dims,
            stride=like.strides if output_dims == like.dims else None,
            data_type=data_type or like.data_type,
            is_virtual=True,
        )

    def _new_scalar_by_value(self, name: str, value: float) -> GraphTensor:
        return self.tensor_scalar(value, name=name, data_type=DataType.FLOAT)

    def pointwise(
        self,
        mode: PointwiseMode,
        x: GraphTensor,
        b: GraphTensor | None = None,
        t: GraphTensor | None = None,
        *,
        name: str | None = None,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        output_dims = x.dims if b is None else _broadcast_dims(x.dims, b.dims)
        if t is not None:
            output_dims = _broadcast_dims(output_dims, t.dims)
        output = self._new_virtual(
            name or f"pointwise_{len(self.nodes)}",
            x,
            output_dims,
            data_type=DataType(data_type) if data_type is not None else None,
        )
        inputs = {"x": x}
        if b is not None:
            inputs["b"] = b
        if t is not None:
            inputs["t"] = t
        self.nodes.append(
            GraphNode(
                op_type="pointwise",
                name=output.name,
                inputs=inputs,
                outputs={"y": output},
                attributes={
                    "mode": PointwiseMode(mode),
                    "alpha1": float(alpha1),
                    "alpha2": float(alpha2),
                },
            )
        )
        return output

    def add(
        self,
        a: GraphTensor,
        b: GraphTensor,
        *,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.ADD, a, b, name=name)

    def bias(
        self,
        x: GraphTensor | None = None,
        b: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        bias: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        tensor = x if x is not None else input
        bias_tensor = b if b is not None else bias
        if tensor is None or bias_tensor is None:
            raise ValueError("bias requires input and bias tensors")
        return self.add(
            tensor, bias_tensor, compute_data_type=compute_data_type, name=name
        )

    def sub(
        self,
        a: GraphTensor,
        b: GraphTensor,
        *,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.SUB, a, b, name=name)

    def mul(
        self,
        a: GraphTensor,
        b: GraphTensor,
        *,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.MUL, a, b, name=name)

    def div(
        self,
        a: GraphTensor,
        b: GraphTensor,
        *,
        compute_data_type: DataType | None = None,
        rounding_mode: str | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        quotient = self.pointwise(PointwiseMode.DIV, a, b, name=name)
        if rounding_mode is None:
            return quotient
        if rounding_mode == "floor":
            quotient.name = f"{name or quotient.name}_quotient"
            return self.floor(quotient, name=name)
        if rounding_mode == "trunc":
            quotient.name = f"{name or quotient.name}_quotient"
            floor_value = self.floor(quotient, name=f"{name or quotient.name}_floor")
            ceil_value = self.ceil(quotient, name=f"{name or quotient.name}_ceil")
            zero = self.tensor_scalar(
                0,
                data_type=quotient.data_type,
                name=f"{name or quotient.name}_zero",
            )
            non_negative = self.ge(
                quotient, zero, name=f"{name or quotient.name}_non_negative"
            )
            return self.select(non_negative, floor_value, ceil_value, name=name)
        raise NotImplementedError("div rounding_mode is not supported")

    def mod(
        self,
        input0: GraphTensor,
        input1: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.MOD, input0, input1, name=name)

    def pow(
        self,
        input0: GraphTensor,
        input1: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.POW, input0, input1, name=name)

    def add_square(
        self,
        a: GraphTensor,
        b: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.ADD_SQUARE, a, b, name=name)

    def min(
        self,
        input0: GraphTensor,
        input1: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.MIN, input0, input1, name=name)

    def max(
        self,
        input0: GraphTensor,
        input1: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.MAX, input0, input1, name=name)

    def neg(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        tensor = x if x is not None else input
        if tensor is None:
            raise ValueError("neg requires an input tensor")
        return self.pointwise(PointwiseMode.NEG, tensor, name=name)

    def abs(
        self,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.ABS, input, name=name)

    def ceil(
        self,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.CEIL, input, name=name)

    def floor(
        self,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.FLOOR, input, name=name)

    def tan(
        self,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.TAN, input, name=name)

    def erf(
        self,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.ERF, input, name=name)

    def reciprocal(
        self,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.pointwise(PointwiseMode.RECIPROCAL, input, name=name)

    def _compare(
        self, mode: PointwiseMode, a: GraphTensor, b: GraphTensor, *, name=None
    ) -> GraphTensor:
        return self.pointwise(mode, a, b, name=name, data_type=DataType.BOOLEAN)

    def _compare_alias(
        self,
        mode: PointwiseMode,
        input: GraphTensor | None = None,
        comparison: GraphTensor | None = None,
        *,
        a: GraphTensor | None = None,
        b: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        lhs = input if input is not None else a
        rhs = comparison if comparison is not None else b
        if lhs is None or rhs is None:
            raise ValueError("comparison requires input/comparison tensors")
        return self._compare(mode, lhs, rhs, name=name)

    def eq(
        self, a: GraphTensor, b: GraphTensor, *, name: str | None = None
    ) -> GraphTensor:
        return self._compare(PointwiseMode.CMP_EQ, a, b, name=name)

    def ne(
        self, a: GraphTensor, b: GraphTensor, *, name: str | None = None
    ) -> GraphTensor:
        return self._compare(PointwiseMode.CMP_NEQ, a, b, name=name)

    def gt(
        self, a: GraphTensor, b: GraphTensor, *, name: str | None = None
    ) -> GraphTensor:
        return self._compare(PointwiseMode.CMP_GT, a, b, name=name)

    def cmp_gt(
        self,
        input: GraphTensor | None = None,
        comparison: GraphTensor | None = None,
        *,
        a: GraphTensor | None = None,
        b: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._compare_alias(
            PointwiseMode.CMP_GT,
            input,
            comparison,
            a=a,
            b=b,
            name=name,
            compute_data_type=compute_data_type,
        )

    def cmp_eq(
        self,
        input: GraphTensor | None = None,
        comparison: GraphTensor | None = None,
        *,
        a: GraphTensor | None = None,
        b: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._compare_alias(
            PointwiseMode.CMP_EQ,
            input,
            comparison,
            a=a,
            b=b,
            name=name,
            compute_data_type=compute_data_type,
        )

    def cmp_neq(
        self,
        input: GraphTensor | None = None,
        comparison: GraphTensor | None = None,
        *,
        a: GraphTensor | None = None,
        b: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._compare_alias(
            PointwiseMode.CMP_NEQ,
            input,
            comparison,
            a=a,
            b=b,
            name=name,
            compute_data_type=compute_data_type,
        )

    def cmp_ge(
        self,
        input: GraphTensor | None = None,
        comparison: GraphTensor | None = None,
        *,
        a: GraphTensor | None = None,
        b: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._compare_alias(
            PointwiseMode.CMP_GE,
            input,
            comparison,
            a=a,
            b=b,
            name=name,
            compute_data_type=compute_data_type,
        )

    def cmp_lt(
        self,
        input: GraphTensor | None = None,
        comparison: GraphTensor | None = None,
        *,
        a: GraphTensor | None = None,
        b: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._compare_alias(
            PointwiseMode.CMP_LT,
            input,
            comparison,
            a=a,
            b=b,
            name=name,
            compute_data_type=compute_data_type,
        )

    def cmp_le(
        self,
        input: GraphTensor | None = None,
        comparison: GraphTensor | None = None,
        *,
        a: GraphTensor | None = None,
        b: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._compare_alias(
            PointwiseMode.CMP_LE,
            input,
            comparison,
            a=a,
            b=b,
            name=name,
            compute_data_type=compute_data_type,
        )

    def ge(
        self, a: GraphTensor, b: GraphTensor, *, name: str | None = None
    ) -> GraphTensor:
        return self._compare(PointwiseMode.CMP_GE, a, b, name=name)

    def lt(
        self, a: GraphTensor, b: GraphTensor, *, name: str | None = None
    ) -> GraphTensor:
        return self._compare(PointwiseMode.CMP_LT, a, b, name=name)

    def le(
        self, a: GraphTensor, b: GraphTensor, *, name: str | None = None
    ) -> GraphTensor:
        return self._compare(PointwiseMode.CMP_LE, a, b, name=name)

    def gen_index(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        axis: int,
        name: str | None = None,
        data_type: DataType | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("gen_index requires an input tensor")
        normalized_axis = _normalize_axis(axis, len(tensor.dims))
        output = self.pointwise(
            PointwiseMode.GEN_INDEX,
            tensor,
            name=name,
            data_type=DataType(data_type) if data_type is not None else DataType.INT32,
        )
        self.nodes[-1].attributes["axis"] = normalized_axis
        return output

    def logical_and(
        self,
        a: GraphTensor,
        b: GraphTensor,
        *,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self.pointwise(
            PointwiseMode.LOGICAL_AND, a, b, name=name, data_type=DataType.BOOLEAN
        )

    def logical_or(
        self,
        a: GraphTensor,
        b: GraphTensor,
        *,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self.pointwise(
            PointwiseMode.LOGICAL_OR, a, b, name=name, data_type=DataType.BOOLEAN
        )

    def logical_not(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        tensor = x if x is not None else input
        if tensor is None:
            raise ValueError("logical_not requires an input tensor")
        return self.pointwise(
            PointwiseMode.LOGICAL_NOT, tensor, name=name, data_type=DataType.BOOLEAN
        )

    def select(
        self,
        pred: GraphTensor,
        true_value: GraphTensor,
        false_value: GraphTensor,
        *,
        name: str | None = None,
    ) -> GraphTensor:
        if pred.data_type not in (DataType.BOOLEAN, DataType.INT32):
            raise ValueError("select predicate must be boolean or int32")
        _broadcast_dims(_broadcast_dims(pred.dims, true_value.dims), false_value.dims)
        return self.pointwise(
            PointwiseMode.BINARY_SELECT,
            true_value,
            false_value,
            pred,
            name=name,
        )

    def binary_select(
        self,
        input0: GraphTensor | None = None,
        input1: GraphTensor | None = None,
        mask: GraphTensor | None = None,
        *,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        if input0 is None or input1 is None or mask is None:
            raise ValueError("binary_select requires input0, input1, and mask")
        return self.select(mask, input0, input1, name=name)

    def _unary_alias(
        self,
        mode: PointwiseMode,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        tensor = x if x is not None else input
        if tensor is None:
            raise ValueError(f"{mode.name.lower()} requires an input tensor")
        return self.pointwise(mode, tensor, name=name)

    def sqrt(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self._unary_alias(
            PointwiseMode.SQRT,
            x,
            input=input,
            compute_data_type=compute_data_type,
            name=name,
        )

    def rsqrt(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self._unary_alias(
            PointwiseMode.RSQRT,
            x,
            input=input,
            compute_data_type=compute_data_type,
            name=name,
        )

    def cos(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self._unary_alias(
            PointwiseMode.COS,
            x,
            input=input,
            compute_data_type=compute_data_type,
            name=name,
        )

    def sin(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self._unary_alias(
            PointwiseMode.SIN,
            x,
            input=input,
            compute_data_type=compute_data_type,
            name=name,
        )

    def exp(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self._unary_alias(
            PointwiseMode.EXP,
            x,
            input=input,
            compute_data_type=compute_data_type,
            name=name,
        )

    def log(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self._unary_alias(
            PointwiseMode.LOG,
            x,
            input=input,
            compute_data_type=compute_data_type,
            name=name,
        )

    def tanh(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("tanh requires an input tensor")
        return self.pointwise(PointwiseMode.TANH_FWD, tensor, name=name)

    def sigmoid(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("sigmoid requires an input tensor")
        return self.pointwise(PointwiseMode.SIGMOID_FWD, tensor, name=name)

    def elu(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        elu_alpha: float = 1.0,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("elu requires an input tensor")
        output = self.pointwise(PointwiseMode.ELU_FWD, tensor, name=name)
        self.nodes[-1].attributes["elu_alpha"] = float(elu_alpha)
        return output

    def gelu(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("gelu requires an input tensor")
        return self.pointwise(PointwiseMode.GELU_FWD, tensor, name=name)

    def gelu_approx_tanh(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("gelu_approx_tanh requires an input tensor")
        return self.pointwise(PointwiseMode.GELU_APPROX_TANH_FWD, tensor, name=name)

    def softplus(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        softplus_beta: float = 1.0,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("softplus requires an input tensor")
        output = self.pointwise(PointwiseMode.SOFTPLUS_FWD, tensor, name=name)
        self.nodes[-1].attributes["softplus_beta"] = float(softplus_beta)
        return output

    def scale(
        self,
        x: GraphTensor | None = None,
        scalar: float | GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        scale: float | GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        tensor = x if x is not None else input
        scale_value = scalar if scalar is not None else scale
        if tensor is None or scale_value is None:
            raise ValueError("scale requires input and scale")
        if isinstance(scale_value, GraphTensor):
            return self.pointwise(PointwiseMode.MUL, tensor, scale_value, name=name)
        return self.pointwise(
            PointwiseMode.IDENTITY, tensor, name=name, alpha1=float(scale_value)
        )

    def identity(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("identity requires an input tensor")
        return self.pointwise(
            PointwiseMode.IDENTITY,
            tensor,
            name=name,
            data_type=data_type,
        )

    def relu(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        negative_slope: float | None = None,
        lower_clip: float | None = None,
        upper_clip: float | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("relu requires an input tensor")
        output = self.pointwise(PointwiseMode.RELU_FWD, tensor, name=name)
        if negative_slope is not None:
            self.nodes[-1].attributes["relu_lower_clip_slope"] = float(negative_slope)
        if lower_clip is not None:
            self.nodes[-1].attributes["relu_lower_clip"] = float(lower_clip)
        if upper_clip is not None:
            self.nodes[-1].attributes["relu_upper_clip"] = float(upper_clip)
        return output

    def leaky_relu(
        self,
        input: GraphTensor | None = None,
        negative_slope: float = 0.01,
        *,
        x: GraphTensor | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.relu(
            input,
            x=x,
            negative_slope=negative_slope,
            name=name,
            compute_data_type=compute_data_type,
        )

    def swish(
        self,
        input: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        swish_beta: float = 1.0,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        if tensor is None:
            raise ValueError("swish requires an input tensor")
        output = self.pointwise(PointwiseMode.SWISH_FWD, tensor, name=name)
        self.nodes[-1].attributes["swish_beta"] = float(swish_beta)
        return output

    def relu_backward(
        self,
        *,
        loss: GraphTensor,
        input: GraphTensor,
        negative_slope: float | None = None,
        lower_clip: float | None = None,
        upper_clip: float | None = None,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._activation_backward(
            PointwiseMode.RELU_BWD,
            loss=loss,
            input=input,
            name=name,
            relu_lower_clip_slope=negative_slope,
            relu_lower_clip=lower_clip,
            relu_upper_clip=upper_clip,
        )

    def leaky_relu_backward(
        self,
        loss: GraphTensor,
        input: GraphTensor,
        negative_slope: float,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self.relu_backward(
            loss=loss,
            input=input,
            negative_slope=negative_slope,
            name=name,
            compute_data_type=compute_data_type,
        )

    def tanh_backward(
        self,
        loss: GraphTensor,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._activation_backward(
            PointwiseMode.TANH_BWD, loss=loss, input=input, name=name
        )

    def sigmoid_backward(
        self,
        loss: GraphTensor,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._activation_backward(
            PointwiseMode.SIGMOID_BWD, loss=loss, input=input, name=name
        )

    def elu_backward(
        self,
        loss: GraphTensor,
        input: GraphTensor,
        *,
        elu_alpha: float = 1.0,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._activation_backward(
            PointwiseMode.ELU_BWD,
            loss=loss,
            input=input,
            name=name,
            elu_alpha=float(elu_alpha),
        )

    def gelu_backward(
        self,
        loss: GraphTensor,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._activation_backward(
            PointwiseMode.GELU_BWD, loss=loss, input=input, name=name
        )

    def softplus_backward(
        self,
        loss: GraphTensor,
        input: GraphTensor,
        *,
        softplus_beta: float = 1.0,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._activation_backward(
            PointwiseMode.SOFTPLUS_BWD,
            loss=loss,
            input=input,
            name=name,
            softplus_beta=float(softplus_beta),
        )

    def gelu_approx_tanh_backward(
        self,
        loss: GraphTensor,
        input: GraphTensor,
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        return self._activation_backward(
            PointwiseMode.GELU_APPROX_TANH_BWD, loss=loss, input=input, name=name
        )

    def swish_backward(
        self,
        *,
        loss: GraphTensor,
        input: GraphTensor,
        swish_beta: float = 1.0,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self._activation_backward(
            PointwiseMode.SWISH_BWD,
            loss=loss,
            input=input,
            name=name,
            swish_beta=float(swish_beta),
        )

    def _activation_backward(
        self,
        mode: PointwiseMode,
        *,
        loss: GraphTensor,
        input: GraphTensor,
        name: str | None = None,
        swish_beta: float | None = None,
        relu_lower_clip: float | None = None,
        relu_upper_clip: float | None = None,
        relu_lower_clip_slope: float | None = None,
        elu_alpha: float | None = None,
        softplus_beta: float | None = None,
    ) -> GraphTensor:
        if loss.dims != input.dims:
            raise ValueError("activation backward loss and input dimensions must match")
        if loss.data_type != input.data_type:
            raise ValueError("activation backward loss and input data types must match")
        output = self._new_virtual(name or f"pointwise_bwd_{len(self.nodes)}", input)
        attributes = {
            "mode": PointwiseMode(mode),
            "alpha1": 1.0,
            "alpha2": 1.0,
        }
        if swish_beta is not None:
            attributes["swish_beta"] = swish_beta
        if relu_lower_clip is not None:
            attributes["relu_lower_clip"] = relu_lower_clip
        if relu_upper_clip is not None:
            attributes["relu_upper_clip"] = relu_upper_clip
        if relu_lower_clip_slope is not None:
            attributes["relu_lower_clip_slope"] = relu_lower_clip_slope
        if elu_alpha is not None:
            attributes["elu_alpha"] = elu_alpha
        if softplus_beta is not None:
            attributes["softplus_beta"] = softplus_beta
        self.nodes.append(
            GraphNode(
                op_type="pointwise",
                name=output.name,
                inputs={"x": input, "dy": loss},
                outputs={"dx": output},
                attributes=attributes,
            )
        )
        return output

    @staticmethod
    def _broadcast_batch_dims(
        lhs: Sequence[int], rhs: Sequence[int], context: str
    ) -> tuple[int, ...]:
        result = []
        lhs_rev = tuple(lhs)[::-1]
        rhs_rev = tuple(rhs)[::-1]
        for index in range(max(len(lhs_rev), len(rhs_rev))):
            lhs_dim = lhs_rev[index] if index < len(lhs_rev) else 1
            rhs_dim = rhs_rev[index] if index < len(rhs_rev) else 1
            if lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1:
                raise ValueError(f"{context} batch dimensions are not broadcastable")
            result.append(max(lhs_dim, rhs_dim))
        return tuple(reversed(result))

    def matmul(
        self,
        a: GraphTensor | None = None,
        b: GraphTensor | None = None,
        *,
        A: GraphTensor | None = None,
        B: GraphTensor | None = None,
        compute_data_type: DataType | None = None,
        padding: float = 0.0,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        a = a if a is not None else A
        b = b if b is not None else B
        if a is None or b is None:
            raise ValueError("matmul requires A and B tensors")
        if len(a.dims) < 2 or len(b.dims) < 2:
            raise ValueError("matmul inputs must be at least rank-2")
        if a.dims[-1] != b.dims[-2]:
            raise ValueError("matmul K dimensions do not match")
        batch = self._broadcast_batch_dims(a.dims[:-2], b.dims[:-2], "matmul")
        output_dims = batch + (a.dims[-2], b.dims[-1])
        output = self.tensor(
            name=name or f"matmul_{len(self.nodes)}",
            dim=output_dims,
            data_type=data_type
            or (compute_data_type if compute_data_type is not None else None)
            or self.io_data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="matmul",
                name=output.name,
                inputs={"a": a, "b": b},
                outputs={"c": output},
                attributes={
                    "compute_data_type": (
                        DataType(compute_data_type)
                        if compute_data_type is not None
                        else self.compute_data_type
                    ),
                    "padding": float(padding),
                },
            )
        )
        return output

    def reduction(
        self,
        *args,
        mode: ReductionMode | None = None,
        x: GraphTensor | None = None,
        input: GraphTensor | None = None,
        axis: int | Sequence[int] | None = None,
        dim: int | Sequence[int] | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
        compute_data_type: DataType | None = None,
        group_offset: GraphTensor | None = None,
    ) -> GraphTensor:
        if len(args) > 2:
            raise TypeError("reduction expects input and optional mode")
        positional = list(args)
        tensor = input if input is not None else x
        reduce_mode = mode
        if positional:
            first = positional.pop(0)
            if isinstance(first, GraphTensor):
                tensor = first
                if positional:
                    reduce_mode = positional.pop(0)
            else:
                reduce_mode = first
                if positional:
                    tensor = positional.pop(0)
        if reduce_mode is None:
            raise ValueError("reduction requires a mode")
        reduce_mode = ReductionMode(reduce_mode)
        if not isinstance(tensor, GraphTensor):
            raise ValueError("reduction requires an input tensor")
        axis = _resolve_axis_dim(axis, dim, "reduction")
        axes = tuple(range(len(tensor.dims))) if axis is None else axis
        normalized_axes = _normalize_axes(axes, len(tensor.dims), "reduction")
        output_dims = tuple(
            1 if index in normalized_axes else dim
            for index, dim in enumerate(tensor.dims)
        )
        output = self.tensor(
            name=name or f"reduction_{len(self.nodes)}",
            dim=output_dims,
            data_type=data_type or tensor.data_type,
            is_virtual=True,
        )
        inputs = {"x": tensor}
        if group_offset is not None:
            inputs["group_offset"] = group_offset
        self.nodes.append(
            GraphNode(
                op_type="reduction",
                name=output.name,
                inputs=inputs,
                outputs={"y": output},
                attributes={
                    "mode": reduce_mode,
                    "compute_data_type": (
                        DataType(compute_data_type)
                        if compute_data_type is not None
                        else self.compute_data_type
                    ),
                    "axis": tuple(normalized_axes),
                },
            )
        )
        return output

    def genstats(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        axis: int | Sequence[int] | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
        compute_data_type: DataType | None = None,
        math_precision: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor]:
        x = x if x is not None else input
        if x is None:
            raise ValueError("genstats requires an input tensor")
        if axis is None:
            if len(x.dims) < 2:
                raise ValueError("genstats default axis requires rank >= 2")
            normalized_axes = tuple(index for index in range(len(x.dims)) if index != 1)
            output_dims = (x.dims[1],)
        else:
            normalized_axes = _normalize_axes(axis, len(x.dims), "genstats")
            output_dims = tuple(
                1 if index in normalized_axes else dim
                for index, dim in enumerate(x.dims)
            )
        out_type = data_type or x.data_type
        base_name = name or f"genstats_{len(self.nodes)}"
        sum_tensor = self.tensor(
            name=f"{base_name}_sum",
            dim=output_dims,
            data_type=out_type,
            is_virtual=True,
        )
        sqsum_tensor = self.tensor(
            name=f"{base_name}_sqsum",
            dim=output_dims,
            data_type=out_type,
            is_virtual=True,
        )
        if sum_tensor.data_type != x.data_type:
            raise ValueError("genstats output data type must match x")
        self.nodes.append(
            GraphNode(
                op_type="genstats",
                name=base_name,
                inputs={"x": x},
                outputs={"sum": sum_tensor, "sqsum": sqsum_tensor},
                attributes={
                    "mode": GenStatsMode.SUM_SQSUM,
                    "math_precision": math_precision or self.compute_data_type,
                    "axis": tuple(normalized_axes),
                },
            )
        )
        return sum_tensor, sqsum_tensor

    def bn_finalize_inference(
        self,
        scale: GraphTensor,
        bias: GraphTensor,
        running_mean: GraphTensor,
        running_var: GraphTensor,
        *,
        epsilon: float = 1.0e-5,
        name: str | None = None,
        data_type: DataType | None = None,
        math_precision: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor]:
        if epsilon < 0.0:
            raise ValueError("bn_finalize epsilon must be nonnegative")
        for param_name, tensor in (
            ("bias", bias),
            ("running_mean", running_mean),
            ("running_var", running_var),
        ):
            _check_same_shape_type(f"bn_finalize {param_name}", tensor, scale)
        out_type = data_type or scale.data_type
        if out_type != scale.data_type:
            raise ValueError("bn_finalize output data type must match scale")
        base_name = name or f"bn_finalize_inference_{len(self.nodes)}"
        eq_scale = self.tensor(
            name=f"{base_name}_eq_scale",
            dim=scale.dims,
            data_type=out_type,
            is_virtual=True,
        )
        eq_bias = self.tensor(
            name=f"{base_name}_eq_bias",
            dim=scale.dims,
            data_type=out_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="bn_finalize",
                name=base_name,
                inputs={
                    "scale": scale,
                    "bias": bias,
                    "prev_running_mean": running_mean,
                    "prev_running_var": running_var,
                },
                outputs={"eq_scale": eq_scale, "eq_bias": eq_bias},
                attributes={
                    "mode": BnFinalizeStatsMode.INFERENCE,
                    "epsilon": float(epsilon),
                    "math_precision": math_precision or self.compute_data_type,
                },
            )
        )
        return eq_scale, eq_bias

    def bn_finalize_training(
        self,
        y_sum: GraphTensor,
        y_sq_sum: GraphTensor,
        scale: GraphTensor,
        bias: GraphTensor,
        prev_running_mean: GraphTensor,
        prev_running_var: GraphTensor,
        accum_count: GraphTensor,
        *,
        momentum: float = 0.1,
        epsilon: float = 1.0e-5,
        name: str | None = None,
        data_type: DataType | None = None,
        math_precision: DataType | None = None,
    ) -> tuple[
        GraphTensor, GraphTensor, GraphTensor, GraphTensor, GraphTensor, GraphTensor
    ]:
        if epsilon < 0.0:
            raise ValueError("bn_finalize epsilon must be nonnegative")
        _check_scalar_int64("bn_finalize accum_count", accum_count)
        for param_name, tensor in (
            ("y_sum", y_sum),
            ("y_sq_sum", y_sq_sum),
            ("bias", bias),
            ("prev_running_mean", prev_running_mean),
            ("prev_running_var", prev_running_var),
        ):
            _check_same_shape_type(f"bn_finalize {param_name}", tensor, scale)
        out_type = data_type or scale.data_type
        if out_type != scale.data_type:
            raise ValueError("bn_finalize output data type must match scale")
        base_name = name or f"bn_finalize_training_{len(self.nodes)}"
        saved_mean = self.tensor(
            name=f"{base_name}_saved_mean",
            dim=scale.dims,
            data_type=out_type,
            is_virtual=True,
        )
        saved_inv_std = self.tensor(
            name=f"{base_name}_saved_inv_std",
            dim=scale.dims,
            data_type=out_type,
            is_virtual=True,
        )
        updated_running_mean = self.tensor(
            name=f"{base_name}_updated_running_mean",
            dim=scale.dims,
            data_type=out_type,
            is_virtual=True,
        )
        updated_running_var = self.tensor(
            name=f"{base_name}_updated_running_var",
            dim=scale.dims,
            data_type=out_type,
            is_virtual=True,
        )
        eq_scale = self.tensor(
            name=f"{base_name}_eq_scale",
            dim=scale.dims,
            data_type=out_type,
            is_virtual=True,
        )
        eq_bias = self.tensor(
            name=f"{base_name}_eq_bias",
            dim=scale.dims,
            data_type=out_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="bn_finalize",
                name=base_name,
                inputs={
                    "y_sum": y_sum,
                    "y_sq_sum": y_sq_sum,
                    "scale": scale,
                    "bias": bias,
                    "prev_running_mean": prev_running_mean,
                    "prev_running_var": prev_running_var,
                    "accum_count": accum_count,
                },
                outputs={
                    "saved_mean": saved_mean,
                    "saved_inv_std": saved_inv_std,
                    "updated_running_mean": updated_running_mean,
                    "updated_running_var": updated_running_var,
                    "eq_scale": eq_scale,
                    "eq_bias": eq_bias,
                },
                attributes={
                    "mode": BnFinalizeStatsMode.TRAINING,
                    "momentum": float(momentum),
                    "epsilon": float(epsilon),
                    "math_precision": math_precision or self.compute_data_type,
                },
            )
        )
        return (
            saved_mean,
            saved_inv_std,
            updated_running_mean,
            updated_running_var,
            eq_scale,
            eq_bias,
        )

    def sum(
        self,
        x: GraphTensor,
        *,
        axis: int | Sequence[int] | None = None,
        dim: int | Sequence[int] | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        axis = _resolve_axis_dim(axis, dim, "sum")
        if axis is None:
            raise ValueError("sum requires an axis or dim")
        return self.reduction(ReductionMode.ADD, x, axis=axis, name=name)

    def mean(
        self,
        x: GraphTensor,
        *,
        axis: int | Sequence[int] | None = None,
        dim: int | Sequence[int] | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        axis = _resolve_axis_dim(axis, dim, "mean")
        if axis is None:
            raise ValueError("mean requires an axis or dim")
        return self.reduction(ReductionMode.AVG, x, axis=axis, name=name)

    def amax(
        self,
        x: GraphTensor,
        *,
        axis: int | Sequence[int] | None = None,
        dim: int | Sequence[int] | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        axis = _resolve_axis_dim(axis, dim, "amax")
        if axis is None:
            raise ValueError("amax requires an axis or dim")
        return self.reduction(ReductionMode.AMAX, x, axis=axis, name=name)

    def softmax(
        self,
        p: GraphTensor,
        *,
        axis: int | None = None,
        dim: int | None = None,
        name: str | None = None,
        sink: GraphTensor | None = None,
        generate_stats: bool = False,
        return_stats: bool = False,
        return_max_sum_exp: bool = False,
        compute_data_type: DataType | None = None,
    ):
        axis = _resolve_axis_dim(axis, dim, "softmax")
        if axis is None:
            axis = -1
        normalized_axis = _normalize_axis(axis, len(p.dims))
        base_name = name or f"softmax_{len(self.nodes)}"
        max_output = self.reduction(
            ReductionMode.MAX,
            p,
            axis=normalized_axis,
            name=f"{base_name}_max",
            data_type=compute_data_type or self.compute_data_type,
        )
        if sink is not None:
            max_output = self.pointwise(
                PointwiseMode.MAX,
                max_output,
                sink,
                name=f"{base_name}_max_sink",
                data_type=compute_data_type or self.compute_data_type,
            )
        shifted = self.sub(p, max_output, name=f"{base_name}_shifted")
        exp_output = self.exp(shifted, name=f"{base_name}_exp")
        sum_exp = self.reduction(
            ReductionMode.ADD,
            exp_output,
            axis=normalized_axis,
            name=f"{base_name}_sum_exp",
            data_type=compute_data_type or self.compute_data_type,
        )
        if sink is not None:
            sink_shifted = self.sub(sink, max_output, name=f"{base_name}_sink_shifted")
            sink_exp = self.exp(sink_shifted, name=f"{base_name}_sink_exp")
            sum_exp = self.add(sum_exp, sink_exp, name=f"{base_name}_sum_exp_sink")
        output = self.div(exp_output, sum_exp, name=base_name)

        returns = [output]
        if generate_stats or return_stats:
            log_sum_exp = self.log(sum_exp, name=f"{base_name}_log_sum_exp")
            stats = self.add(max_output, log_sum_exp, name=f"{base_name}_stats")
            returns.append(stats)
        if return_max_sum_exp:
            returns.extend([max_output, sum_exp])
        return output if len(returns) == 1 else tuple(returns)

    def concat(
        self,
        tensors: Sequence[GraphTensor],
        *,
        axis: int,
        name: str | None = None,
        data_type: DataType | None = None,
        inplace_index: int | None = None,
    ) -> GraphTensor:
        inputs = tuple(tensors)
        if not inputs:
            raise ValueError("concat requires at least one input tensor")
        if inplace_index is not None:
            raise NotImplementedError("concat inplace output is not supported")
        rank = len(inputs[0].dims)
        if rank == 0:
            raise ValueError("concat inputs must have rank greater than zero")
        normalized_axis = _normalize_axis(axis, rank)
        output_dims = list(inputs[0].dims)
        output_dims[normalized_axis] = 0
        for tensor in inputs:
            if len(tensor.dims) != rank:
                raise ValueError("concat inputs must have the same rank")
            if tensor.data_type != inputs[0].data_type:
                raise ValueError("concat inputs must have the same data type")
            for index, dim in enumerate(tensor.dims):
                if index == normalized_axis:
                    output_dims[index] += int(dim)
                elif dim != inputs[0].dims[index]:
                    raise ValueError("concat non-axis dimensions must match")
        output = self.tensor(
            name=name or f"concat_{len(self.nodes)}",
            dim=tuple(output_dims),
            data_type=data_type or inputs[0].data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="concat",
                name=output.name,
                inputs={f"x{index}": tensor for index, tensor in enumerate(inputs)},
                outputs={"y": output},
                attributes={"axis": normalized_axis},
            )
        )
        return output

    def concatenate(
        self,
        inputs: Sequence[GraphTensor],
        axis: int,
        in_place_index: int | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        return self.concat(
            inputs,
            axis=axis,
            name=name,
            inplace_index=in_place_index,
        )

    def slice(
        self,
        input: GraphTensor,
        slices: Sequence[slice],
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        x = input
        if len(slices) != len(x.dims):
            raise ValueError("slice rank must match input rank")
        start_indices = []
        limit_indices = []
        slice_strides = []
        output_dims = []
        output_strides = []
        for dim, stride, spec in zip(x.dims, x.strides, slices):
            if not isinstance(spec, slice):
                raise TypeError("slice entries must be Python slice objects")
            start, stop, step = spec.indices(int(dim))
            if step <= 0:
                raise ValueError("slice step must be positive")
            length = max(0, (stop - start + step - 1) // step)
            if length <= 0:
                raise ValueError("slice output dimensions must be positive")
            start_indices.append(int(start))
            limit_indices.append(int(stop))
            slice_strides.append(int(step))
            output_dims.append(int(length))
            output_strides.append(int(stride) * int(step))
        output = self.tensor(
            name=name or f"slice_{len(self.nodes)}",
            dim=tuple(output_dims),
            stride=tuple(output_strides),
            data_type=x.data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="slice",
                name=output.name,
                inputs={"x": x},
                outputs={"y": output},
                attributes={
                    "start_indices": tuple(start_indices),
                    "limit_indices": tuple(limit_indices),
                    "strides": tuple(slice_strides),
                    "compute_data_type": compute_data_type or self.compute_data_type,
                },
            )
        )
        return output

    def reshape(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        shape: Sequence[int] | None = None,
        name: str | None = None,
        reshape_mode=None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        x = x if x is not None else input
        if x is None:
            raise ValueError("reshape requires an input tensor")
        if reshape_mode is None:
            normalized_mode = ReshapeMode.VIEW_ONLY
        else:
            normalized_mode = ReshapeMode(reshape_mode)
        output_dims = tuple(
            int(dim) for dim in (shape if shape is not None else x.dims)
        )
        if _num_elements(x.dims) != _num_elements(output_dims):
            raise ValueError("reshape output shape must preserve element count")
        output = self.tensor(
            name=name or f"reshape_{len(self.nodes)}",
            dim=output_dims,
            data_type=data_type or x.data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="reshape",
                name=output.name,
                inputs={"x": x},
                outputs={"y": output},
                attributes={"reshape_mode": normalized_mode},
            )
        )
        return output

    def transpose(
        self,
        input: GraphTensor,
        permutation: Sequence[int],
        *,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        x = input
        rank = len(x.dims)
        perm = tuple(int(dim) for dim in permutation)
        if rank == 0 or len(perm) != rank:
            raise ValueError("transpose permutation rank must match input rank")
        if len(set(perm)) != len(perm):
            raise ValueError("transpose permutation contains duplicates")
        if any(dim < 0 or dim >= rank for dim in perm):
            raise ValueError("transpose permutation is out of range")
        output = self.tensor(
            name=name or f"transpose_{len(self.nodes)}",
            dim=tuple(x.dims[dim] for dim in perm),
            stride=tuple(x.strides[dim] for dim in perm),
            data_type=x.data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="transpose",
                name=output.name,
                inputs={"x": x},
                outputs={"y": output},
                attributes={
                    "permutation": perm,
                    "compute_data_type": compute_data_type or self.compute_data_type,
                },
            )
        )
        return output

    def rope(
        self,
        input: GraphTensor | None = None,
        freqs_cis: GraphTensor | None = None,
        *,
        x: GraphTensor | None = None,
        freqs: GraphTensor | None = None,
        output_scale: float | None = None,
        rope_dim: int | None = None,
        rotary_interleaved: bool = False,
        batch_first: bool = False,
        multi_latent_attention: bool = False,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        tensor = input if input is not None else x
        freq_tensor = freqs_cis if freqs_cis is not None else freqs
        if tensor is None or freq_tensor is None:
            raise ValueError("rope requires input and freqs_cis tensors")
        if len(tensor.dims) < 2:
            raise ValueError("rope input rank must be at least 2")
        if len(freq_tensor.dims) == 0:
            raise ValueError("rope freqs_cis must be non-scalar")

        rot_dim = int(rope_dim) if rope_dim is not None else int(freq_tensor.dims[-1])
        input_dim = int(tensor.dims[-1])
        if rot_dim <= 0 or rot_dim > input_dim:
            raise ValueError(
                "rope rotary dimension must be within input last dimension"
            )
        if rot_dim % 2 != 0:
            raise ValueError("rope rotary dimension must be even")
        if int(freq_tensor.dims[-1]) != rot_dim:
            raise ValueError("rope freqs_cis last dimension must match rope_dim")

        rank = len(tensor.dims)
        prefix = (slice(None),) * (rank - 1)
        rotate_start = input_dim - rot_dim if rope_dim is not None else 0
        t_rot = self.slice(
            tensor,
            prefix + (slice(rotate_start, rotate_start + rot_dim),),
            name=f"{name or 'rope'}_rotary_input",
            compute_data_type=compute_data_type,
        )
        t_pass_prefix = None
        t_pass_suffix = None
        if rotate_start > 0:
            t_pass_prefix = self.slice(
                tensor,
                prefix + (slice(0, rotate_start),),
                name=f"{name or 'rope'}_pass_prefix",
                compute_data_type=compute_data_type,
            )
        if rotate_start + rot_dim < input_dim:
            t_pass_suffix = self.slice(
                tensor,
                prefix + (slice(rotate_start + rot_dim, input_dim),),
                name=f"{name or 'rope'}_pass_suffix",
                compute_data_type=compute_data_type,
            )

        if multi_latent_attention:
            even = self.slice(
                t_rot,
                prefix + (slice(0, rot_dim, 2),),
                name=f"{name or 'rope'}_mla_even",
                compute_data_type=compute_data_type,
            )
            odd = self.slice(
                t_rot,
                prefix + (slice(1, rot_dim, 2),),
                name=f"{name or 'rope'}_mla_odd",
                compute_data_type=compute_data_type,
            )
            t_rot = self.concat(
                (even, odd),
                axis=-1,
                name=f"{name or 'rope'}_mla_reordered",
                data_type=tensor.data_type,
            )

        half_dim = rot_dim // 2
        if rotary_interleaved:
            even = self.slice(
                t_rot,
                prefix + (slice(0, rot_dim, 2),),
                name=f"{name or 'rope'}_even",
                compute_data_type=compute_data_type,
            )
            odd = self.slice(
                t_rot,
                prefix + (slice(1, rot_dim, 2),),
                name=f"{name or 'rope'}_odd",
                compute_data_type=compute_data_type,
            )
            neg_odd = self.neg(odd, name=f"{name or 'rope'}_neg_odd")
            stacked = self.concat(
                (
                    self.reshape(
                        neg_odd,
                        shape=neg_odd.dims + (1,),
                        name=f"{name or 'rope'}_neg_odd_stack",
                    ),
                    self.reshape(
                        even,
                        shape=even.dims + (1,),
                        name=f"{name or 'rope'}_even_stack",
                    ),
                ),
                axis=-1,
                name=f"{name or 'rope'}_interleaved_stack",
                data_type=tensor.data_type,
            )
            rotated_half = self.reshape(
                stacked,
                shape=t_rot.dims,
                name=f"{name or 'rope'}_rotated_half",
                data_type=tensor.data_type,
            )
        else:
            first = self.slice(
                t_rot,
                prefix + (slice(0, half_dim),),
                name=f"{name or 'rope'}_first_half",
                compute_data_type=compute_data_type,
            )
            second = self.slice(
                t_rot,
                prefix + (slice(half_dim, rot_dim),),
                name=f"{name or 'rope'}_second_half",
                compute_data_type=compute_data_type,
            )
            rotated_half = self.concat(
                (
                    self.neg(second, name=f"{name or 'rope'}_neg_second_half"),
                    first,
                ),
                axis=-1,
                name=f"{name or 'rope'}_rotated_half",
                data_type=tensor.data_type,
            )

        freqs_rank = len(freq_tensor.dims)
        if freqs_rank == rank:
            try:
                _broadcast_dims(t_rot.dims, freq_tensor.dims)
                broadcast_freqs = freq_tensor
            except ValueError:
                seq_candidates = [rank - 2, 1 if batch_first else 0]
                freq_dims = [1] * rank
                for seq_axis in dict.fromkeys(seq_candidates):
                    if 0 <= seq_axis < rank - 1 and int(freq_tensor.dims[0]) == int(
                        t_rot.dims[seq_axis]
                    ):
                        freq_dims[seq_axis] = int(freq_tensor.dims[0])
                        break
                else:
                    raise ValueError("rope freqs_cis rank is not broadcastable")
                freq_dims[-1] = rot_dim
                broadcast_freqs = self.reshape(
                    freq_tensor,
                    shape=tuple(freq_dims),
                    name=f"{name or 'rope'}_freqs_broadcast",
                    data_type=freq_tensor.data_type,
                )
        else:
            freq_dims = [1] * rank
            if freqs_rank == 1:
                pass
            elif freqs_rank == 2:
                seq_axis = 1 if batch_first else 0
                if seq_axis >= rank - 1:
                    raise ValueError("rope input rank is too small for batch_first")
                freq_dims[seq_axis] = int(freq_tensor.dims[0])
            else:
                leading = tuple(int(dim) for dim in freq_tensor.dims[:-1])
                if len(leading) > rank - 1:
                    raise ValueError("rope freqs_cis rank is not broadcastable")
                start = rank - 1 - len(leading)
                freq_dims[start : rank - 1] = leading
            freq_dims[-1] = rot_dim
            broadcast_freqs = self.reshape(
                freq_tensor,
                shape=tuple(freq_dims),
                name=f"{name or 'rope'}_freqs_broadcast",
                data_type=freq_tensor.data_type,
            )

        cos = self.cos(broadcast_freqs, name=f"{name or 'rope'}_cos")
        sin = self.sin(broadcast_freqs, name=f"{name or 'rope'}_sin")
        rotated = self.add(
            self.mul(t_rot, cos, name=f"{name or 'rope'}_cos_term"),
            self.mul(rotated_half, sin, name=f"{name or 'rope'}_sin_term"),
            name=f"{name or 'rope'}_rotated",
        )
        if t_pass_prefix is not None or t_pass_suffix is not None:
            parts = tuple(
                tensor
                for tensor in (t_pass_prefix, rotated, t_pass_suffix)
                if tensor is not None
            )
            rotated = self.concat(
                parts,
                axis=-1,
                name=(f"{name}_unscaled" if output_scale is not None and name else name)
                or f"rope_{len(self.nodes)}",
                data_type=tensor.data_type,
            )
        elif name is not None and output_scale is None:
            rotated.set_name(name)
        if output_scale is not None:
            rotated = self.scale(
                rotated,
                float(output_scale),
                name=name or f"rope_{len(self.nodes)}",
            )
        return rotated

    def rope_backward(
        self,
        dY: GraphTensor | None = None,
        freqs_cis: GraphTensor | None = None,
        *,
        dy: GraphTensor | None = None,
        freqs: GraphTensor | None = None,
        output_scale: float | None = None,
        rope_dim: int | None = None,
        rotary_interleaved: bool = False,
        batch_first: bool = False,
        multi_latent_attention: bool = False,
        name: str | None = None,
        compute_data_type: DataType | None = None,
    ) -> GraphTensor:
        grad = dY if dY is not None else dy
        freq_tensor = freqs_cis if freqs_cis is not None else freqs
        if grad is None or freq_tensor is None:
            raise ValueError("rope_backward requires dY and freqs_cis tensors")
        neg_freqs = self.neg(freq_tensor, name=f"{name or 'rope_backward'}_neg_freqs")
        return self.rope(
            input=grad,
            freqs_cis=neg_freqs,
            output_scale=output_scale,
            rope_dim=rope_dim,
            rotary_interleaved=rotary_interleaved,
            batch_first=batch_first,
            multi_latent_attention=multi_latent_attention,
            name=name,
            compute_data_type=compute_data_type,
        )

    def moe_grouped_matmul(
        self,
        token: GraphTensor,
        weight: GraphTensor,
        first_token_offset: GraphTensor,
        token_index: GraphTensor | None = None,
        token_ks: GraphTensor | None = None,
        mode: MoeGroupedMatmulMode = MoeGroupedMatmulMode.NONE,
        compute_data_type: DataType | None = None,
        top_k: int = 0,
        name: str | None = None,
    ) -> GraphTensor:
        mode = MoeGroupedMatmulMode(mode)
        if len(token.dims) != 3 or len(weight.dims) != 3:
            raise ValueError("moe_grouped_matmul token and weight must be rank-3")
        expert_count = _moe_first_token_offset_experts(
            first_token_offset, "moe_grouped_matmul"
        )
        if token.dims[2] != weight.dims[1]:
            raise ValueError("moe_grouped_matmul K dimensions must match")
        if expert_count != weight.dims[0]:
            raise ValueError(
                "moe_grouped_matmul first_token_offset length must match experts"
            )
        if mode in (MoeGroupedMatmulMode.GATHER, MoeGroupedMatmulMode.SCATTER):
            if token_index is None:
                raise ValueError("moe_grouped_matmul token_index is required")
            if len(token_index.dims) != 2:
                raise ValueError("moe_grouped_matmul token_index must be rank-2")
        if mode == MoeGroupedMatmulMode.SCATTER:
            if token_ks is None:
                raise ValueError("moe_grouped_matmul token_ks is required")
            if len(token_ks.dims) != 1:
                raise ValueError("moe_grouped_matmul token_ks must be rank-1")
            if int(top_k) <= 0:
                raise ValueError("moe_grouped_matmul top_k must be positive")

        token_count = (
            token_index.dims[1]
            if mode == MoeGroupedMatmulMode.GATHER
            else token.dims[1]
        )
        output = self.tensor(
            name=name or f"moe_grouped_matmul_{len(self.nodes)}",
            dim=(1, int(token_count), int(weight.dims[2])),
            data_type=token.data_type,
            is_virtual=True,
        )
        inputs = {
            "token": token,
            "weight": weight,
            "first_token_offset": first_token_offset,
        }
        if token_index is not None:
            inputs["token_index"] = token_index
        if token_ks is not None:
            inputs["token_ks"] = token_ks
        self.nodes.append(
            GraphNode(
                op_type="moe_grouped_matmul",
                name=output.name,
                inputs=inputs,
                outputs={"output": output},
                attributes={
                    "mode": mode,
                    "compute_data_type": compute_data_type or self.compute_data_type,
                    "top_k": int(top_k),
                },
            )
        )
        return output

    def moe_grouped_matmul_bwd(
        self,
        doutput: GraphTensor,
        token: GraphTensor,
        first_token_offset: GraphTensor,
        compute_data_type: DataType | None = None,
        name: str | None = None,
    ) -> GraphTensor:
        if len(doutput.dims) != 3 or len(token.dims) != 3:
            raise ValueError("moe_grouped_matmul_bwd doutput and token must be rank-3")
        expert_count = _moe_first_token_offset_experts(
            first_token_offset, "moe_grouped_matmul_bwd"
        )
        if doutput.dims[1] != token.dims[1]:
            raise ValueError("moe_grouped_matmul_bwd token counts must match")
        dweight = self.tensor(
            name=name or f"moe_grouped_matmul_bwd_{len(self.nodes)}",
            dim=(
                expert_count,
                int(token.dims[2]),
                int(doutput.dims[2]),
            ),
            data_type=token.data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="moe_grouped_matmul_bwd",
                name=dweight.name,
                inputs={
                    "doutput": doutput,
                    "token": token,
                    "first_token_offset": first_token_offset,
                },
                outputs={"dweight": dweight},
                attributes={
                    "compute_data_type": compute_data_type or self.compute_data_type,
                },
            )
        )
        return dweight

    def rng(
        self,
        *,
        shape: Sequence[int],
        seed: int,
        distribution: RngDistribution = RngDistribution.UNIFORM,
        minimum: float = 0.0,
        maximum: float = 1.0,
        mean: float = 0.0,
        stddev: float = 1.0,
        probability: float = 0.5,
        offset: GraphTensor | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        distribution = RngDistribution(distribution)
        output_dims = tuple(int(dim) for dim in shape)
        _num_elements(output_dims)
        if distribution == RngDistribution.UNIFORM and not minimum < maximum:
            raise ValueError("uniform RNG minimum must be less than maximum")
        if distribution == RngDistribution.NORMAL and stddev <= 0.0:
            raise ValueError("normal RNG standard deviation must be positive")
        if distribution == RngDistribution.BERNOULLI and not 0.0 <= probability <= 1.0:
            raise ValueError("bernoulli RNG probability must be in [0, 1]")
        if offset is not None:
            if offset.data_type != DataType.INT64 or offset.dims != (1,):
                raise ValueError("rng offset must be an int64 scalar tensor")
        output = self.tensor(
            name=name or f"rng_{len(self.nodes)}",
            dim=output_dims,
            data_type=data_type or self.io_data_type,
            is_virtual=True,
        )
        inputs = {}
        if offset is not None:
            inputs["offset"] = offset
        self.nodes.append(
            GraphNode(
                op_type="rng",
                name=output.name,
                inputs=inputs,
                outputs={"y": output},
                attributes={
                    "distribution": distribution,
                    "seed": int(seed),
                    "minimum": float(minimum),
                    "maximum": float(maximum),
                    "mean": float(mean),
                    "stddev": float(stddev),
                    "probability": float(probability),
                },
            )
        )
        return output

    def batch_norm_inference(
        self,
        x: GraphTensor,
        mean: GraphTensor,
        variance: GraphTensor,
        scale: GraphTensor,
        bias: GraphTensor,
        *,
        epsilon: float | GraphTensor = 1.0e-5,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        epsilon_value = None
        if isinstance(epsilon, GraphTensor):
            _check_scalar_float_tensor("batch_norm_inference", "epsilon", epsilon)
        else:
            epsilon_value = _resolve_scalar_attr(
                "batch_norm_inference", "epsilon", epsilon
            )
            if epsilon_value < 0.0:
                raise ValueError("batch_norm_inference epsilon must be nonnegative")
        for param_name, tensor in (
            ("mean", mean),
            ("variance", variance),
            ("scale", scale),
            ("bias", bias),
        ):
            _check_batch_norm_param(param_name, tensor, x)
        output = self.tensor(
            name=name or f"batch_norm_inference_{len(self.nodes)}",
            dim=x.dims,
            data_type=data_type or x.data_type,
            is_virtual=True,
        )
        inputs = {
            "x": x,
            "mean": mean,
            "variance": variance,
            "scale": scale,
            "bias": bias,
        }
        if isinstance(epsilon, GraphTensor):
            inputs["epsilon"] = epsilon
        attributes = {
            "mode": NormMode.BATCH_NORM,
            "phase": NormFwdPhase.INFERENCE,
        }
        if epsilon_value is not None:
            attributes["epsilon"] = epsilon_value
        self.nodes.append(
            GraphNode(
                op_type="norm_fwd",
                name=output.name,
                inputs=inputs,
                outputs={"y": output},
                attributes=attributes,
            )
        )
        return output

    def batch_norm_training(
        self,
        x: GraphTensor,
        input_running_mean: GraphTensor,
        input_running_variance: GraphTensor,
        scale: GraphTensor,
        bias: GraphTensor,
        *,
        momentum: float | GraphTensor = 0.1,
        epsilon: float | GraphTensor = 1.0e-5,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor, GraphTensor, GraphTensor]:
        epsilon_value = None
        if isinstance(epsilon, GraphTensor):
            _check_scalar_float_tensor("batch_norm_training", "epsilon", epsilon)
        else:
            epsilon_value = _resolve_scalar_attr(
                "batch_norm_training", "epsilon", epsilon
            )
            if epsilon_value < 0.0:
                raise ValueError("batch_norm_training epsilon must be nonnegative")
        momentum_value = None
        if isinstance(momentum, GraphTensor):
            _check_scalar_float_tensor("batch_norm_training", "momentum", momentum)
        else:
            momentum_value = _resolve_scalar_attr(
                "batch_norm_training", "momentum", momentum
            )
        param_pairs = (
            ("input_running_mean", input_running_mean),
            ("input_running_variance", input_running_variance),
            ("scale", scale),
            ("bias", bias),
        )
        for param_name, tensor in param_pairs:
            _check_batch_norm_param(param_name, tensor, x)
        base_name = name or f"batch_norm_training_{len(self.nodes)}"
        output = self.tensor(
            name=base_name,
            dim=x.dims,
            data_type=data_type or x.data_type,
            is_virtual=True,
        )
        output_running_mean = self.tensor(
            name=f"{base_name}_running_mean",
            dim=_batch_norm_param_dims(x),
            data_type=x.data_type,
            is_virtual=True,
        )
        output_running_variance = self.tensor(
            name=f"{base_name}_running_variance",
            dim=_batch_norm_param_dims(x),
            data_type=x.data_type,
            is_virtual=True,
        )
        saved_mean = self.tensor(
            name=f"{base_name}_saved_mean",
            dim=_batch_norm_param_dims(x),
            data_type=x.data_type,
            is_virtual=True,
        )
        saved_inv_variance = self.tensor(
            name=f"{base_name}_saved_inv_variance",
            dim=_batch_norm_param_dims(x),
            data_type=x.data_type,
            is_virtual=True,
        )
        inputs = {
            "x": x,
            "input_running_mean": input_running_mean,
            "input_running_variance": input_running_variance,
            "scale": scale,
            "bias": bias,
        }
        if isinstance(epsilon, GraphTensor):
            inputs["epsilon"] = epsilon
        if isinstance(momentum, GraphTensor):
            inputs["momentum"] = momentum
        attributes = {
            "mode": NormMode.BATCH_NORM,
            "phase": NormFwdPhase.TRAINING,
        }
        if momentum_value is not None:
            attributes["momentum"] = momentum_value
        if epsilon_value is not None:
            attributes["epsilon"] = epsilon_value
        self.nodes.append(
            GraphNode(
                op_type="norm_fwd",
                name=output.name,
                inputs=inputs,
                outputs={
                    "y": output,
                    "output_running_mean": output_running_mean,
                    "output_running_variance": output_running_variance,
                    "saved_mean": saved_mean,
                    "saved_inv_variance": saved_inv_variance,
                },
                attributes=attributes,
            )
        )
        return (
            output,
            output_running_mean,
            output_running_variance,
            saved_mean,
            saved_inv_variance,
        )

    def batch_norm_backward(
        self,
        x: GraphTensor,
        dy: GraphTensor,
        mean: GraphTensor,
        inv_variance: GraphTensor,
        scale: GraphTensor,
        *,
        epsilon: float | GraphTensor = 1.0e-5,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor]:
        """Batch-norm backward over the saved training statistics.

        ``inv_variance`` is the *saved inverse standard deviation* produced by
        the training forward pass (``MUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC``),
        not the raw variance: the epsilon is already folded into it by the
        forward pass. ``epsilon`` is accepted for signature compatibility with
        the forward-side graph builders and is not applied again here.
        """
        epsilon_value = None
        if isinstance(epsilon, GraphTensor):
            _check_scalar_float_tensor("batch_norm_backward", "epsilon", epsilon)
        else:
            epsilon_value = _resolve_scalar_attr(
                "batch_norm_backward", "epsilon", epsilon
            )
            if epsilon_value < 0.0:
                raise ValueError("batch_norm_backward epsilon must be nonnegative")
        if dy.dims != x.dims or dy.data_type != x.data_type:
            raise ValueError("batch_norm_backward dy shape and data type must match x")
        for param_name, tensor in (
            ("mean", mean),
            ("inv_variance", inv_variance),
            ("scale", scale),
        ):
            _check_batch_norm_param(param_name, tensor, x)
        base_name = name or f"batch_norm_backward_{len(self.nodes)}"
        dx = self.tensor(
            name=base_name,
            dim=x.dims,
            data_type=data_type or x.data_type,
            is_virtual=True,
        )
        dscale = self.tensor(
            name=f"{base_name}_dscale",
            dim=_batch_norm_param_dims(x),
            data_type=x.data_type,
            is_virtual=True,
        )
        dbias = self.tensor(
            name=f"{base_name}_dbias",
            dim=_batch_norm_param_dims(x),
            data_type=x.data_type,
            is_virtual=True,
        )
        inputs = {
            "x": x,
            "dy": dy,
            "mean": mean,
            "variance": inv_variance,
            "scale": scale,
        }
        if isinstance(epsilon, GraphTensor):
            inputs["epsilon"] = epsilon
        attributes = {"mode": NormMode.BATCH_NORM}
        if epsilon_value is not None:
            attributes["epsilon"] = epsilon_value
        self.nodes.append(
            GraphNode(
                op_type="norm_bwd",
                name=dx.name,
                inputs=inputs,
                outputs={"dx": dx, "dscale": dscale, "dbias": dbias},
                attributes=attributes,
            )
        )
        return dx, dscale, dbias

    def batchnorm(
        self,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        bias: GraphTensor | None = None,
        in_running_mean: GraphTensor | None = None,
        in_running_var: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        momentum: float | GraphTensor = 0.1,
        peer_stats: Sequence[GraphTensor] | None = None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor, GraphTensor, GraphTensor]:
        if peer_stats:
            raise NotImplementedError("batchnorm peer_stats is not supported")
        if (
            input is None
            or scale is None
            or bias is None
            or in_running_mean is None
            or in_running_var is None
        ):
            raise ValueError("batchnorm requires input, scale, bias, and running stats")
        y, out_mean, out_var, saved_mean, saved_inv_var = self.batch_norm_training(
            input,
            in_running_mean,
            in_running_var,
            scale,
            bias,
            epsilon=epsilon,
            momentum=momentum,
            name=name,
            data_type=data_type,
        )
        return y, saved_mean, saved_inv_var, out_mean, out_var

    def batchnorm_inference(
        self,
        input: GraphTensor | None = None,
        mean: GraphTensor | None = None,
        inv_variance: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        bias: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        if (
            input is None
            or mean is None
            or inv_variance is None
            or scale is None
            or bias is None
        ):
            raise ValueError(
                "batchnorm_inference requires input, stats, scale, and bias"
            )
        return self.batch_norm_inference(
            input,
            mean,
            inv_variance,
            scale,
            bias,
            epsilon=epsilon,
            name=name,
            data_type=data_type,
        )

    def batchnorm_backward(
        self,
        grad: GraphTensor | None = None,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        mean: GraphTensor | None = None,
        inv_variance: GraphTensor | None = None,
        peer_stats: Sequence[GraphTensor] | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor]:
        if peer_stats:
            raise NotImplementedError("batchnorm_backward peer_stats is not supported")
        if (
            grad is None
            or input is None
            or scale is None
            or mean is None
            or inv_variance is None
        ):
            raise ValueError(
                "batchnorm_backward requires grad, input, scale, and stats"
            )
        return self.batch_norm_backward(
            input,
            grad,
            mean,
            inv_variance,
            scale,
            epsilon=epsilon,
            name=name,
            data_type=data_type,
        )

    def _norm_forward(
        self,
        op_name: str,
        mode: NormMode,
        *,
        input: GraphTensor,
        scale: GraphTensor,
        bias: GraphTensor | None,
        epsilon: float | GraphTensor,
        norm_forward_phase: NormFwdPhase = NormFwdPhase.TRAINING,
        name: str | None = None,
        data_type: DataType | None = None,
        has_mean: bool = True,
    ):
        epsilon_value = None
        if isinstance(epsilon, GraphTensor):
            _check_scalar_float_tensor(op_name, "epsilon", epsilon)
        else:
            epsilon_value = _resolve_scalar_attr(op_name, "epsilon", epsilon)
            if epsilon_value < 0.0:
                raise ValueError(f"{op_name} epsilon must be nonnegative")
        phase = NormFwdPhase(norm_forward_phase)
        _check_norm_param(op_name, "scale", scale, input, mode)
        if bias is not None:
            _check_norm_param(op_name, "bias", bias, input, mode)
        base_name = name or f"{op_name}_{len(self.nodes)}"
        y = self.tensor(
            name=base_name,
            dim=input.dims,
            data_type=data_type or input.data_type,
            is_virtual=True,
        )
        stat_dims = _norm_stat_dims(mode, input)
        outputs = {"y": y}
        if phase == NormFwdPhase.TRAINING and has_mean:
            outputs["saved_mean"] = self.tensor(
                name=f"{base_name}_mean",
                dim=stat_dims,
                data_type=self.compute_data_type,
                is_virtual=True,
            )
        if phase == NormFwdPhase.TRAINING or mode == NormMode.RMS_NORM:
            outputs["saved_inv_variance"] = self.tensor(
                name=f"{base_name}_inv_variance",
                dim=stat_dims,
                data_type=self.compute_data_type,
                is_virtual=True,
            )
        inputs = {"x": input, "scale": scale}
        if bias is not None:
            inputs["bias"] = bias
        if isinstance(epsilon, GraphTensor):
            inputs["epsilon"] = epsilon
        attributes = {
            "mode": mode,
            "phase": phase,
        }
        if epsilon_value is not None:
            attributes["epsilon"] = epsilon_value
        self.nodes.append(
            GraphNode(
                op_type="norm_fwd",
                name=y.name,
                inputs=inputs,
                outputs=outputs,
                attributes=attributes,
            )
        )
        if has_mean:
            return (
                y,
                outputs.get("saved_mean"),
                outputs.get("saved_inv_variance"),
            )
        return y, outputs.get("saved_inv_variance")

    def _norm_backward(
        self,
        op_name: str,
        mode: NormMode,
        *,
        grad: GraphTensor,
        input: GraphTensor,
        scale: GraphTensor,
        mean: GraphTensor | None,
        inv_variance: GraphTensor,
        epsilon: float | GraphTensor = 1.0e-5,
        name: str | None = None,
        data_type: DataType | None = None,
        has_dbias: bool = True,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor | None]:
        epsilon_value = None
        if isinstance(epsilon, GraphTensor):
            _check_scalar_float_tensor(op_name, "epsilon", epsilon)
        else:
            epsilon_value = _resolve_scalar_attr(op_name, "epsilon", epsilon)
            if epsilon_value < 0.0:
                raise ValueError(f"{op_name} epsilon must be nonnegative")
        if grad.dims != input.dims or grad.data_type != input.data_type:
            raise ValueError(f"{op_name} grad shape and data type must match input")
        _check_norm_param(op_name, "scale", scale, input, mode)
        stat_dims = _norm_stat_dims(mode, input)
        if mean is not None and mean.dims != stat_dims:
            raise ValueError(f"{op_name} mean dimensions must be {stat_dims}")
        if inv_variance.dims != stat_dims:
            raise ValueError(f"{op_name} inv_variance dimensions must be {stat_dims}")
        base_name = name or f"{op_name}_{len(self.nodes)}"
        dx = self.tensor(
            name=base_name,
            dim=input.dims,
            data_type=data_type or input.data_type,
            is_virtual=True,
        )
        dscale = self.tensor(
            name=f"{base_name}_dscale",
            dim=scale.dims,
            data_type=scale.data_type,
            is_virtual=True,
        )
        outputs: dict[str, GraphTensor] = {"dx": dx, "dscale": dscale}
        dbias = None
        if has_dbias:
            dbias = self.tensor(
                name=f"{base_name}_dbias",
                dim=scale.dims,
                data_type=scale.data_type,
                is_virtual=True,
            )
            outputs["dbias"] = dbias
        inputs = {
            "x": input,
            "dy": grad,
            "variance": inv_variance,
            "scale": scale,
        }
        if mean is not None:
            inputs["mean"] = mean
        if isinstance(epsilon, GraphTensor):
            inputs["epsilon"] = epsilon
        attributes = {"mode": mode}
        if epsilon_value is not None:
            attributes["epsilon"] = epsilon_value
        self.nodes.append(
            GraphNode(
                op_type="norm_bwd",
                name=dx.name,
                inputs=inputs,
                outputs=outputs,
                attributes=attributes,
            )
        )
        return dx, dscale, dbias

    def layernorm(
        self,
        norm_forward_phase: NormFwdPhase = NormFwdPhase.TRAINING,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        bias: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor]:
        if input is None or scale is None or bias is None:
            raise ValueError("layernorm requires input, scale, and bias")
        return self._norm_forward(
            "layernorm",
            NormMode.LAYER_NORM,
            input=input,
            scale=scale,
            bias=bias,
            epsilon=epsilon,
            norm_forward_phase=norm_forward_phase,
            name=name,
            data_type=data_type,
            has_mean=True,
        )

    def layernorm_backward(
        self,
        grad: GraphTensor | None = None,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        mean: GraphTensor | None = None,
        inv_variance: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor | None]:
        if (
            grad is None
            or input is None
            or scale is None
            or mean is None
            or inv_variance is None
        ):
            raise ValueError(
                "layernorm_backward requires grad, input, scale, and stats"
            )
        return self._norm_backward(
            "layernorm_backward",
            NormMode.LAYER_NORM,
            grad=grad,
            input=input,
            scale=scale,
            mean=mean,
            inv_variance=inv_variance,
            epsilon=epsilon,
            name=name,
            data_type=data_type,
            has_dbias=True,
        )

    def adalayernorm(
        self,
        norm_forward_phase: NormFwdPhase = NormFwdPhase.TRAINING,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        bias: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor]:
        if input is None or scale is None:
            raise ValueError("adalayernorm requires input and scale")
        return self._norm_forward(
            "adalayernorm",
            NormMode.ADA_LAYER_NORM,
            input=input,
            scale=scale,
            bias=bias,
            epsilon=epsilon,
            norm_forward_phase=norm_forward_phase,
            name=name,
            data_type=data_type,
            has_mean=True,
        )

    def adalayernorm_backward(
        self,
        grad: GraphTensor | None = None,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        mean: GraphTensor | None = None,
        inv_variance: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor | None]:
        if (
            grad is None
            or input is None
            or scale is None
            or mean is None
            or inv_variance is None
        ):
            raise ValueError(
                "adalayernorm_backward requires grad, input, scale, and stats"
            )
        return self._norm_backward(
            "adalayernorm_backward",
            NormMode.ADA_LAYER_NORM,
            grad=grad,
            input=input,
            scale=scale,
            mean=mean,
            inv_variance=inv_variance,
            epsilon=epsilon,
            name=name,
            data_type=data_type,
            has_dbias=True,
        )

    def rmsnorm(
        self,
        norm_forward_phase: NormFwdPhase = NormFwdPhase.TRAINING,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        bias: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor]:
        if input is None or scale is None:
            raise ValueError("rmsnorm requires input and scale")
        return self._norm_forward(
            "rmsnorm",
            NormMode.RMS_NORM,
            input=input,
            scale=scale,
            bias=bias,
            epsilon=epsilon,
            norm_forward_phase=norm_forward_phase,
            name=name,
            data_type=data_type,
            has_mean=False,
        )

    def rmsnorm_backward(
        self,
        grad: GraphTensor | None = None,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        inv_variance: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        has_dbias: bool = True,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor | None]:
        if grad is None or input is None or scale is None or inv_variance is None:
            raise ValueError(
                "rmsnorm_backward requires grad, input, scale, and inv_variance"
            )
        return self._norm_backward(
            "rmsnorm_backward",
            NormMode.RMS_NORM,
            grad=grad,
            input=input,
            scale=scale,
            mean=None,
            inv_variance=inv_variance,
            epsilon=epsilon,
            name=name,
            data_type=data_type,
            has_dbias=has_dbias,
        )

    def instancenorm(
        self,
        norm_forward_phase: NormFwdPhase = NormFwdPhase.TRAINING,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        bias: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor]:
        if input is None or scale is None or bias is None:
            raise ValueError("instancenorm requires input, scale, and bias")
        return self._norm_forward(
            "instancenorm",
            NormMode.INSTANCE_NORM,
            input=input,
            scale=scale,
            bias=bias,
            epsilon=epsilon,
            norm_forward_phase=norm_forward_phase,
            name=name,
            data_type=data_type,
            has_mean=True,
        )

    def instancenorm_backward(
        self,
        grad: GraphTensor | None = None,
        input: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        mean: GraphTensor | None = None,
        inv_variance: GraphTensor | None = None,
        epsilon: float | GraphTensor = 1.0e-5,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor | None]:
        if grad is None or input is None or scale is None or inv_variance is None:
            raise ValueError(
                "instancenorm_backward requires grad, input, scale, and inv_variance"
            )
        return self._norm_backward(
            "instancenorm_backward",
            NormMode.INSTANCE_NORM,
            grad=grad,
            input=input,
            scale=scale,
            mean=mean,
            inv_variance=inv_variance,
            epsilon=epsilon,
            name=name,
            data_type=data_type,
            has_dbias=True,
        )

    def bn_backward_weights(
        self,
        x: GraphTensor,
        dy: GraphTensor,
        mean: GraphTensor,
        invstd: GraphTensor,
        scale: GraphTensor,
        *,
        name: str | None = None,
        data_type: DataType | None = None,
        math_precision: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor, GraphTensor, GraphTensor]:
        if dy.dims != x.dims or dy.data_type != x.data_type:
            raise ValueError("bn_backward_weights dy shape and data type must match x")
        for param_name, tensor in (
            ("mean", mean),
            ("invstd", invstd),
            ("scale", scale),
        ):
            _check_batch_norm_param(param_name, tensor, x)
        out_type = data_type or scale.data_type
        if out_type != scale.data_type:
            raise ValueError("bn_backward_weights output data type must match scale")
        base_name = name or f"bn_backward_weights_{len(self.nodes)}"
        dbn_scale = self.tensor(
            name=f"{base_name}_dbn_scale",
            dim=_batch_norm_param_dims(x),
            data_type=out_type,
            is_virtual=True,
        )
        dbn_bias = self.tensor(
            name=f"{base_name}_dbn_bias",
            dim=_batch_norm_param_dims(x),
            data_type=out_type,
            is_virtual=True,
        )
        eq_dy_scale = self.tensor(
            name=f"{base_name}_eq_dy_scale",
            dim=_batch_norm_param_dims(x),
            data_type=out_type,
            is_virtual=True,
        )
        eq_x_scale = self.tensor(
            name=f"{base_name}_eq_x_scale",
            dim=_batch_norm_param_dims(x),
            data_type=out_type,
            is_virtual=True,
        )
        eq_bias = self.tensor(
            name=f"{base_name}_eq_bias",
            dim=_batch_norm_param_dims(x),
            data_type=out_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="bn_bwd_weights",
                name=base_name,
                inputs={
                    "x": x,
                    "dy": dy,
                    "mean": mean,
                    "invstd": invstd,
                    "scale": scale,
                },
                outputs={
                    "dbn_scale": dbn_scale,
                    "dbn_bias": dbn_bias,
                    "eq_dy_scale": eq_dy_scale,
                    "eq_x_scale": eq_x_scale,
                    "eq_bias": eq_bias,
                },
                attributes={
                    "math_precision": math_precision or self.compute_data_type,
                },
            )
        )
        return dbn_scale, dbn_bias, eq_dy_scale, eq_x_scale, eq_bias

    def block_scale_quantize(
        self,
        x: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        block_size: int | Sequence[int],
        axis: int | None = None,
        transpose: bool = False,
        name: str | None = None,
        data_type: DataType | None = None,
        scale_data_type: DataType | None = None,
        compute_data_type: DataType | None = None,
        math_precision: DataType | None = None,
    ) -> tuple[GraphTensor, GraphTensor]:
        x = x if x is not None else input
        if x is None:
            raise ValueError("block_scale_quantize requires an input tensor")
        if transpose:
            raise NotImplementedError("block_scale_quantize transpose is not supported")
        block_size, block_axis = _normalize_block_scale_block_size_and_axis(
            x, block_size, axis=axis
        )
        scale_dims = _block_scale_scale_dims(x, block_size, axis=block_axis)
        base_name = name or f"block_scale_quantize_{len(self.nodes)}"
        y = self.tensor(
            name=base_name,
            dim=x.dims,
            data_type=data_type if data_type is not None else x.data_type,
            is_virtual=True,
        )
        scale = self.tensor(
            name=f"{base_name}_scale",
            dim=scale_dims,
            data_type=scale_data_type if scale_data_type is not None else x.data_type,
            is_virtual=True,
        )
        precision = math_precision
        if precision is None:
            precision = compute_data_type
        if precision is None:
            precision = self.compute_data_type
        self.nodes.append(
            GraphNode(
                op_type="block_scale_quantize",
                name=y.name,
                inputs={"x": x},
                outputs={"y": y, "scale": scale},
                attributes={
                    "block_size": block_size,
                    "block_axis": block_axis,
                    "math_precision": precision,
                },
            )
        )
        return y, scale

    def block_scale_dequantize(
        self,
        x: GraphTensor | None = None,
        scale: GraphTensor | None = None,
        *,
        input: GraphTensor | None = None,
        descale: GraphTensor | None = None,
        block_size: int | Sequence[int],
        is_negative_scale: bool = False,
        name: str | None = None,
        data_type: DataType | None = None,
        compute_data_type: DataType | None = None,
        math_precision: DataType | None = None,
    ) -> GraphTensor:
        x = x if x is not None else input
        scale = scale if scale is not None else descale
        if x is None or scale is None:
            raise ValueError("block_scale_dequantize requires input and descale")
        block_size, block_axis = _normalize_block_scale_block_size_and_axis(
            x, block_size
        )
        _check_block_scale_scale("scale", scale, x, block_size, axis=block_axis)
        y = self.tensor(
            name=name or f"block_scale_dequantize_{len(self.nodes)}",
            dim=x.dims,
            data_type=data_type if data_type is not None else self.io_data_type,
            is_virtual=True,
        )
        precision = math_precision
        if precision is None:
            precision = compute_data_type
        if precision is None:
            precision = self.compute_data_type
        self.nodes.append(
            GraphNode(
                op_type="block_scale_dequantize",
                name=y.name,
                inputs={"x": x, "scale": scale},
                outputs={"y": y},
                attributes={
                    "block_size": block_size,
                    "block_axis": block_axis,
                    "math_precision": precision,
                    "is_negative_scale": bool(is_negative_scale),
                },
            )
        )
        return y

    def paged_cache_load(
        self,
        container: GraphTensor,
        sequence: GraphTensor,
        page_table: GraphTensor,
        *,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        output_dims = _paged_cache_load_output_dims(container, sequence, page_table)
        y = self.tensor(
            name=name or f"paged_cache_load_{len(self.nodes)}",
            dim=output_dims,
            data_type=data_type or container.data_type,
            is_virtual=True,
        )
        if y.data_type != container.data_type:
            raise ValueError("paged_cache_load output data type must match container")
        self.nodes.append(
            GraphNode(
                op_type="paged_cache_load",
                name=y.name,
                inputs={
                    "container": container,
                    "sequence": sequence,
                    "page_table": page_table,
                },
                outputs={"y": y},
                attributes={},
            )
        )
        return y

    def signal(
        self,
        mode: SignalMode,
        flag: GraphTensor,
        x: GraphTensor,
        *,
        value: int,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        mode = SignalMode(mode)
        _check_signal_flag(flag)
        y = self.tensor(
            name=name or f"signal_{len(self.nodes)}",
            dim=x.dims,
            data_type=data_type or x.data_type,
            is_virtual=True,
        )
        if y.data_type != x.data_type:
            raise ValueError("signal output data type must match x")
        inputs = {"x": x}
        outputs = {"y": y}
        if mode == SignalMode.SET:
            outputs["flag"] = flag
        else:
            inputs["flag"] = flag
        self.nodes.append(
            GraphNode(
                op_type="signal",
                name=y.name,
                inputs=inputs,
                outputs=outputs,
                attributes={"mode": mode, "value": int(value)},
            )
        )
        return y

    def conv_fprop(
        self,
        x: GraphTensor | None = None,
        w: GraphTensor | None = None,
        *,
        image: GraphTensor | None = None,
        weight: GraphTensor | None = None,
        padding: Sequence[int] | None = None,
        pre_padding: Sequence[int] | None = None,
        post_padding: Sequence[int] | None = None,
        stride: Sequence[int] = (1, 1),
        dilation: Sequence[int] = (1, 1),
        convolution_mode: ConvolutionMode = ConvolutionMode.CROSS_CORRELATION,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> GraphTensor:
        x = x if x is not None else image
        w = w if w is not None else weight
        if x is None or w is None:
            raise ValueError("conv_fprop requires image and weight tensors")
        if len(x.dims) != len(w.dims) or len(x.dims) < 3 or len(x.dims) > 5:
            raise ValueError("conv_fprop expects rank-3 to rank-5 NC... tensors")
        if w.dims[1] <= 0 or x.dims[1] % w.dims[1] != 0:
            raise ValueError("conv_fprop input and weight channel dimensions differ")
        group_count = x.dims[1] // w.dims[1]
        if w.dims[0] % group_count != 0:
            raise ValueError("conv_fprop output channels must be divisible by groups")
        spatial_rank = len(x.dims) - 2
        pre_padding, post_padding = _resolve_conv_paddings(
            spatial_rank, padding, pre_padding, post_padding
        )
        pre_pad, strides, dilations = _normalize_conv_params(
            spatial_rank, pre_padding, stride, dilation
        )
        post_pad, _, _ = _normalize_conv_params(
            spatial_rank, post_padding, strides, dilations
        )
        output_spatial = _conv_output_spatial(
            x.dims[2:], w.dims[2:], pre_pad, strides, dilations, post_pad
        )
        if any(dim <= 0 for dim in output_spatial):
            raise ValueError("conv_fprop output dimensions must be positive")
        output = self.tensor(
            name=name or f"conv_fprop_{len(self.nodes)}",
            dim=(x.dims[0], w.dims[0], *output_spatial),
            stride=(
                _channels_last_strides((x.dims[0], w.dims[0], *output_spatial))
                if _is_channels_last_ncx(x)
                else None
            ),
            data_type=data_type or self.io_data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="conv_fprop",
                name=output.name,
                inputs={"x": x, "w": w},
                outputs={"y": output},
                attributes={
                    "padding": pre_pad,
                    "pre_padding": pre_pad,
                    "post_padding": post_pad,
                    "stride": strides,
                    "dilation": dilations,
                    "group_count": group_count,
                    "compute_data_type": (
                        DataType(compute_data_type)
                        if compute_data_type is not None
                        else self.compute_data_type
                    ),
                    "mode": ConvolutionMode(convolution_mode),
                    "alpha": float(alpha),
                    "beta": float(beta),
                },
            )
        )
        return output

    def conv_bwd_data(
        self,
        dy: GraphTensor,
        w: GraphTensor,
        *,
        x_shape: Sequence[int],
        padding: Sequence[int] | None = None,
        pre_padding: Sequence[int] | None = None,
        post_padding: Sequence[int] | None = None,
        stride: Sequence[int] = (1, 1),
        dilation: Sequence[int] = (1, 1),
        convolution_mode: ConvolutionMode = ConvolutionMode.CROSS_CORRELATION,
        name: str | None = None,
        data_type: DataType | None = None,
        compute_data_type: DataType | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> GraphTensor:
        if len(dy.dims) != len(w.dims) or len(dy.dims) < 3 or len(dy.dims) > 5:
            raise ValueError("conv_bwd_data expects rank-3 to rank-5 NC... tensors")
        x_dims = tuple(int(v) for v in x_shape)
        if len(x_dims) != len(dy.dims):
            raise ValueError("conv_bwd_data x_shape rank must match dy")
        if dy.dims[0] != x_dims[0]:
            raise ValueError("conv_bwd_data batch dimensions differ")
        if w.dims[1] <= 0 or x_dims[1] % w.dims[1] != 0:
            raise ValueError("conv_bwd_data channel dimensions differ")
        group_count = x_dims[1] // w.dims[1]
        if dy.dims[1] != w.dims[0] or w.dims[0] % group_count != 0:
            raise ValueError("conv_bwd_data channel dimensions differ")
        spatial_rank = len(dy.dims) - 2
        pre_padding, post_padding = _resolve_conv_paddings(
            spatial_rank, padding, pre_padding, post_padding
        )
        pre_pad, strides, dilations = _normalize_conv_params(
            spatial_rank, pre_padding, stride, dilation
        )
        post_pad, _, _ = _normalize_conv_params(
            spatial_rank, post_padding, strides, dilations
        )
        expected_spatial = _conv_output_spatial(
            x_dims[2:], w.dims[2:], pre_pad, strides, dilations, post_pad
        )
        if expected_spatial != dy.dims[2:]:
            raise ValueError("conv_bwd_data dy dimensions do not match convolution")
        output = self.tensor(
            name=name or f"conv_bwd_data_{len(self.nodes)}",
            dim=x_dims,
            data_type=data_type or self.io_data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="conv_bwd_data",
                name=output.name,
                inputs={"dy": dy, "w": w},
                outputs={"dx": output},
                attributes={
                    "padding": pre_pad,
                    "pre_padding": pre_pad,
                    "post_padding": post_pad,
                    "stride": strides,
                    "dilation": dilations,
                    "group_count": group_count,
                    "compute_data_type": (
                        DataType(compute_data_type)
                        if compute_data_type is not None
                        else self.compute_data_type
                    ),
                    "mode": ConvolutionMode(convolution_mode),
                    "alpha": float(alpha),
                    "beta": float(beta),
                },
            )
        )
        return output

    def conv_dgrad(
        self,
        loss: GraphTensor,
        filter: GraphTensor,
        *,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        **kwargs,
    ) -> GraphTensor:
        padding = kwargs.get("padding")
        pre_padding = kwargs.get("pre_padding")
        post_padding = kwargs.get("post_padding")
        stride = kwargs.get("stride", (1, 1))
        dilation = kwargs.get("dilation", (1, 1))
        spatial_rank = len(loss.dims) - 2
        pre_padding, post_padding = _resolve_conv_paddings(
            spatial_rank, padding, pre_padding, post_padding
        )
        pre_pad, strides, dilations = _normalize_conv_params(
            spatial_rank, pre_padding, stride, dilation
        )
        post_pad, _, _ = _normalize_conv_params(
            spatial_rank, post_padding, strides, dilations
        )
        inferred_spatial = _conv_input_spatial_from_output(
            loss.dims[2:], filter.dims[2:], pre_pad, strides, dilations, post_pad
        )
        return self.conv_bwd_data(
            loss,
            filter,
            x_shape=kwargs.pop(
                "x_shape",
                (loss.dims[0], filter.dims[1], *inferred_spatial),
            ),
            name=name,
            compute_data_type=compute_data_type,
            **kwargs,
        )

    def conv_bwd_filter(
        self,
        x: GraphTensor,
        dy: GraphTensor,
        *,
        w_shape: Sequence[int],
        padding: Sequence[int] | None = None,
        pre_padding: Sequence[int] | None = None,
        post_padding: Sequence[int] | None = None,
        stride: Sequence[int] = (1, 1),
        dilation: Sequence[int] = (1, 1),
        convolution_mode: ConvolutionMode = ConvolutionMode.CROSS_CORRELATION,
        name: str | None = None,
        data_type: DataType | None = None,
        compute_data_type: DataType | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> GraphTensor:
        if len(x.dims) != len(dy.dims) or len(x.dims) < 3 or len(x.dims) > 5:
            raise ValueError("conv_bwd_filter expects rank-3 to rank-5 NC... tensors")
        w_dims = tuple(int(v) for v in w_shape)
        if len(w_dims) != len(x.dims):
            raise ValueError("conv_bwd_filter w_shape rank must match x")
        if x.dims[0] != dy.dims[0]:
            raise ValueError("conv_bwd_filter batch dimensions differ")
        if w_dims[1] <= 0 or x.dims[1] % w_dims[1] != 0:
            raise ValueError("conv_bwd_filter channel dimensions differ")
        group_count = x.dims[1] // w_dims[1]
        if dy.dims[1] != w_dims[0] or w_dims[0] % group_count != 0:
            raise ValueError("conv_bwd_filter channel dimensions differ")
        spatial_rank = len(x.dims) - 2
        pre_padding, post_padding = _resolve_conv_paddings(
            spatial_rank, padding, pre_padding, post_padding
        )
        pre_pad, strides, dilations = _normalize_conv_params(
            spatial_rank, pre_padding, stride, dilation
        )
        post_pad, _, _ = _normalize_conv_params(
            spatial_rank, post_padding, strides, dilations
        )
        expected_spatial = _conv_output_spatial(
            x.dims[2:], w_dims[2:], pre_pad, strides, dilations, post_pad
        )
        if expected_spatial != dy.dims[2:]:
            raise ValueError("conv_bwd_filter dy dimensions do not match convolution")
        output = self.tensor(
            name=name or f"conv_bwd_filter_{len(self.nodes)}",
            dim=w_dims,
            data_type=data_type or self.io_data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="conv_bwd_filter",
                name=output.name,
                inputs={"x": x, "dy": dy},
                outputs={"dw": output},
                attributes={
                    "padding": pre_pad,
                    "pre_padding": pre_pad,
                    "post_padding": post_pad,
                    "stride": strides,
                    "dilation": dilations,
                    "group_count": group_count,
                    "compute_data_type": (
                        DataType(compute_data_type)
                        if compute_data_type is not None
                        else self.compute_data_type
                    ),
                    "mode": ConvolutionMode(convolution_mode),
                    "alpha": float(alpha),
                    "beta": float(beta),
                },
            )
        )
        return output

    def conv_wgrad(
        self,
        image: GraphTensor,
        loss: GraphTensor,
        *,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        **kwargs,
    ) -> GraphTensor:
        w_shape = kwargs.pop("w_shape", None)
        if w_shape is None:
            padding = kwargs.get("padding")
            pre_padding = kwargs.get("pre_padding")
            post_padding = kwargs.get("post_padding")
            spatial_rank = len(image.dims) - 2
            pre_padding, post_padding = _resolve_conv_paddings(
                spatial_rank, padding, pre_padding, post_padding
            )
            pre_pad, strides, dilations = _normalize_conv_params(
                spatial_rank,
                pre_padding,
                kwargs.get("stride", (1, 1)),
                kwargs.get("dilation", (1, 1)),
            )
            post_pad, _, _ = _normalize_conv_params(
                spatial_rank, post_padding, strides, dilations
            )
            filter_spatial = _infer_conv_filter_spatial(
                image.dims[2:], loss.dims[2:], pre_pad, strides, dilations, post_pad
            )
            w_shape = (loss.dims[1], image.dims[1], *filter_spatial)
        return self.conv_bwd_filter(
            image,
            loss,
            w_shape=w_shape,
            name=name,
            compute_data_type=compute_data_type,
            **kwargs,
        )

    def resample(
        self,
        x: GraphTensor,
        *,
        mode: ResampleMode,
        window: Sequence[int],
        padding: Sequence[int] = (0, 0),
        stride: Sequence[int] | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        return_indices: bool = False,
    ) -> GraphTensor:
        if len(x.dims) != 4:
            raise ValueError("resample expects rank-4 NCHW tensors")
        windows, pad, strides = _normalize_resample2d_params(
            window, padding, stride if stride is not None else window
        )
        out_h, out_w = _resample2d_output_spatial(x.dims[2:], windows, pad, strides)
        if out_h <= 0 or out_w <= 0:
            raise ValueError("resample output dimensions must be positive")
        output = self.tensor(
            name=name or f"resample_{len(self.nodes)}",
            dim=(x.dims[0], x.dims[1], out_h, out_w),
            data_type=data_type or self.io_data_type,
            is_virtual=True,
        )
        outputs = {"y": output}
        if return_indices:
            if ResampleMode(mode) != ResampleMode.MAXPOOL:
                raise ValueError("resample indices are only supported for max-pool")
            outputs["idx"] = self.tensor(
                name=f"{output.name}_idx",
                dim=output.dims,
                data_type=DataType.INT64,
                is_virtual=True,
            )
        self.nodes.append(
            GraphNode(
                op_type="resample_fwd",
                name=output.name,
                inputs={"x": x},
                outputs=outputs,
                attributes={
                    "mode": ResampleMode(mode),
                    "window": windows,
                    "padding": pad,
                    "stride": strides,
                    "compute_data_type": self.compute_data_type,
                    "alpha": float(alpha),
                    "beta": float(beta),
                },
            )
        )
        return output

    def avg_pool(
        self,
        x: GraphTensor,
        *,
        window: Sequence[int],
        padding: Sequence[int] = (0, 0),
        stride: Sequence[int] | None = None,
        include_padding: bool = True,
        name: str | None = None,
    ) -> GraphTensor:
        mode = (
            ResampleMode.AVGPOOL_INCLUDE_PADDING
            if include_padding
            else ResampleMode.AVGPOOL_EXCLUDE_PADDING
        )
        return self.resample(
            x, mode=mode, window=window, padding=padding, stride=stride, name=name
        )

    def max_pool(
        self,
        x: GraphTensor,
        *,
        window: Sequence[int],
        padding: Sequence[int] = (0, 0),
        stride: Sequence[int] | None = None,
        name: str | None = None,
        return_indices: bool = False,
    ) -> GraphTensor:
        return self.resample(
            x,
            mode=ResampleMode.MAXPOOL,
            window=window,
            padding=padding,
            stride=stride,
            name=name,
            return_indices=return_indices,
        )

    def resample_bwd(
        self,
        dy: GraphTensor,
        *,
        x_shape: Sequence[int],
        mode: ResampleMode,
        window: Sequence[int],
        padding: Sequence[int] = (0, 0),
        stride: Sequence[int] | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        x: GraphTensor | None = None,
        y: GraphTensor | None = None,
        indices: GraphTensor | None = None,
    ) -> GraphTensor:
        if len(dy.dims) != 4:
            raise ValueError("resample_bwd expects rank-4 NCHW tensors")
        x_dims = tuple(int(v) for v in x_shape)
        if len(x_dims) != 4:
            raise ValueError("resample_bwd x_shape must be rank-4")
        windows, pad, strides = _normalize_resample2d_params(
            window, padding, stride if stride is not None else window
        )
        expected_spatial = _resample2d_output_spatial(x_dims[2:], windows, pad, strides)
        if dy.dims[:2] != x_dims[:2] or dy.dims[2:] != expected_spatial:
            raise ValueError("resample_bwd dy dimensions do not match resample")
        inputs = {"dy": dy}
        if indices is not None:
            if indices.dims != dy.dims or indices.data_type != DataType.INT64:
                raise ValueError(
                    "resample_bwd indices must be int64 with dy dimensions"
                )
            inputs["idx"] = indices
        if x is not None:
            if x.dims != x_dims or x.data_type != dy.data_type:
                raise ValueError("resample_bwd x dimensions and data type must match")
            inputs["x"] = x
        if y is not None:
            if y.dims != dy.dims or y.data_type != dy.data_type:
                raise ValueError(
                    "resample_bwd y dimensions and data type must match dy"
                )
            inputs["y"] = y
        if mode == ResampleMode.MAXPOOL and indices is None:
            raise ValueError("max-pool backward requires index tensor")
        output = self.tensor(
            name=name or f"resample_bwd_{len(self.nodes)}",
            dim=x_dims,
            data_type=data_type or self.io_data_type,
            is_virtual=True,
        )
        self.nodes.append(
            GraphNode(
                op_type="resample_bwd",
                name=output.name,
                inputs=inputs,
                outputs={"dx": output},
                attributes={
                    "mode": ResampleMode(mode),
                    "window": windows,
                    "padding": pad,
                    "stride": strides,
                    "compute_data_type": self.compute_data_type,
                    "alpha": float(alpha),
                    "beta": float(beta),
                },
            )
        )
        return output

    def avg_pool_bwd(
        self,
        dy: GraphTensor,
        *,
        x_shape: Sequence[int],
        window: Sequence[int],
        padding: Sequence[int] = (0, 0),
        stride: Sequence[int] | None = None,
        include_padding: bool = True,
        name: str | None = None,
    ) -> GraphTensor:
        mode = (
            ResampleMode.AVGPOOL_INCLUDE_PADDING
            if include_padding
            else ResampleMode.AVGPOOL_EXCLUDE_PADDING
        )
        return self.resample_bwd(
            dy,
            x_shape=x_shape,
            mode=mode,
            window=window,
            padding=padding,
            stride=stride,
            name=name,
        )

    def max_pool_bwd(
        self,
        x: GraphTensor,
        dy: GraphTensor,
        *,
        window: Sequence[int],
        padding: Sequence[int] = (0, 0),
        stride: Sequence[int] | None = None,
        name: str | None = None,
        data_type: DataType | None = None,
    ) -> GraphTensor:
        if len(x.dims) != 4 or len(dy.dims) != 4:
            raise ValueError("max_pool_bwd expects rank-4 NCHW tensors")
        windows, pad, strides = _normalize_resample2d_params(
            window, padding, stride if stride is not None else window
        )
        expected_spatial = _resample2d_output_spatial(x.dims[2:], windows, pad, strides)
        if dy.dims[:2] != x.dims[:2] or dy.dims[2:] != expected_spatial:
            raise ValueError("max_pool_bwd dy dimensions do not match max-pool")
        if dy.data_type != x.data_type:
            raise ValueError("max_pool_bwd dy data type must match x")
        base_name = name or f"max_pool_bwd_{len(self.nodes)}"
        y = self.resample(
            x,
            mode=ResampleMode.MAXPOOL,
            window=windows,
            padding=pad,
            stride=strides,
            name=f"{base_name}_forward",
            data_type=dy.data_type,
            return_indices=True,
        )
        idx = self.nodes[-1].outputs["idx"]
        return self.resample_bwd(
            dy,
            x_shape=x.dims,
            mode=ResampleMode.MAXPOOL,
            window=windows,
            padding=pad,
            stride=strides,
            name=base_name,
            data_type=data_type or x.data_type,
            x=x,
            y=y,
            indices=idx,
        )

    def sdpa(
        self,
        q: GraphTensor,
        k: GraphTensor,
        v: GraphTensor,
        *,
        name: str | None = None,
        attn_scale: float | GraphTensor | None = None,
        bias: GraphTensor | None = None,
        mask: GraphTensor | None = None,
        causal_mask: bool = False,
        use_causal_mask: bool | None = None,
        diagonal_alignment: DiagonalAlignment | int | str | None = None,
        diagonal_band_left_bound: int | None = None,
        diagonal_band_right_bound: int | None = None,
        use_padding_mask: bool | None = None,
        use_alibi_mask: bool = False,
        use_causal_mask_bottom_right: bool | None = None,
        seq_len_q: GraphTensor | None = None,
        seq_len_kv: GraphTensor | None = None,
        cu_seq_len_q: GraphTensor | None = None,
        cu_seq_len_kv: GraphTensor | None = None,
        max_total_seq_len_q: int | None = None,
        max_total_seq_len_kv: int | None = None,
        dropout: float | None = None,
        sliding_window_length: int | None = None,
        generate_stats: bool | None = None,
        is_inference: bool | None = None,
        compute_data_type: DataType | None = None,
        implementation: AttentionImplementation | int | None = None,
        unfuse_fma: bool = False,
        block_mask: GraphTensor | None = None,
        rng_dump: GraphTensor | None = None,
        paged_attention_k_table: GraphTensor | None = None,
        paged_attention_v_table: GraphTensor | None = None,
        paged_attention_max_seq_len_kv: int | None = None,
        descale_q: GraphTensor | None = None,
        descale_k: GraphTensor | None = None,
        descale_v: GraphTensor | None = None,
        descale_s: GraphTensor | None = None,
        scale_s: GraphTensor | None = None,
        scale_o: GraphTensor | None = None,
        score_mod=None,
        score_sum_exp: GraphTensor | None = None,
        score_max: GraphTensor | None = None,
        sink_token: GraphTensor | None = None,
    ) -> tuple[GraphTensor, GraphTensor | None, GraphTensor]:
        seq_len_q = _resolve_tensor_alias(
            "sdpa", "seq_len_q", seq_len_q, "cu_seq_len_q", cu_seq_len_q
        )
        seq_len_kv = _resolve_tensor_alias(
            "sdpa", "seq_len_kv", seq_len_kv, "cu_seq_len_kv", cu_seq_len_kv
        )
        requested_causal_mask = bool(
            use_causal_mask if use_causal_mask is not None else causal_mask
        )
        causal_mask = _resolve_sdpa_causal_mask(
            "sdpa",
            causal_mask,
            use_causal_mask,
            diagonal_alignment,
            diagonal_band_left_bound,
            diagonal_band_right_bound,
        )
        normalized_diagonal_alignment = (
            None
            if diagonal_alignment is None
            else _normalize_diagonal_alignment(diagonal_alignment)
        )
        diagonal_left_bound = (
            None if diagonal_band_left_bound is None else int(diagonal_band_left_bound)
        )
        diagonal_right_bound = (
            None
            if diagonal_band_right_bound is None
            else int(diagonal_band_right_bound)
        )
        if normalized_diagonal_alignment is None and (
            diagonal_left_bound is not None or diagonal_right_bound is not None
        ):
            normalized_diagonal_alignment = DiagonalAlignment.TOP_LEFT
        if (
            requested_causal_mask
            and normalized_diagonal_alignment == DiagonalAlignment.BOTTOM_RIGHT
            and diagonal_right_bound is None
        ):
            diagonal_right_bound = 0
        paged_attention = (
            paged_attention_k_table is not None
            or paged_attention_v_table is not None
            or paged_attention_max_seq_len_kv is not None
        )
        _reject_unsupported_sdpa_options(
            "sdpa",
            use_alibi_mask=use_alibi_mask,
            use_causal_mask_bottom_right=use_causal_mask_bottom_right,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            max_total_seq_len_q=max_total_seq_len_q,
            max_total_seq_len_kv=max_total_seq_len_kv,
            dropout=dropout,
            sliding_window_length=sliding_window_length,
            implementation=implementation,
            unfuse_fma=unfuse_fma,
            block_mask=block_mask,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_s=descale_s,
            scale_s=scale_s,
            scale_o=scale_o,
        )
        dropout_probability = 0.0
        dropout_seed = None
        dropout_offset = None
        if dropout is not None:
            if isinstance(dropout, tuple):
                if len(dropout) != 3:
                    raise ValueError(
                        "sdpa dropout tuple must be (probability, seed, offset)"
                    )
                dropout_probability = float(dropout[0])
                dropout_seed = dropout[1]
                dropout_offset = dropout[2]
            else:
                dropout_probability = float(dropout)
            if dropout_probability < 0.0 or dropout_probability >= 1.0:
                raise ValueError("sdpa dropout probability must be in [0, 1)")
            if dropout_probability != 0.0 and rng_dump is None:
                raise ValueError("sdpa dropout requires rng_dump")
        if generate_stats is None and is_inference is None:
            raise ValueError("sdpa requires generate_stats or is_inference")
        if generate_stats is not None and is_inference is not None:
            if bool(is_inference) == bool(generate_stats):
                raise ValueError("sdpa generate_stats and is_inference disagree")
        effective_is_inference = (
            not bool(generate_stats)
            if generate_stats is not None
            else bool(is_inference)
        )
        if bias is not None and mask is not None:
            if mask.data_type == DataType.BOOLEAN:
                if bias.data_type != q.data_type:
                    raise ValueError("sdpa bias data type must match q")
            else:
                if bias.data_type != q.data_type or mask.data_type != q.data_type:
                    raise ValueError(
                        "sdpa bias and additive mask data types must match q"
                    )
                bias = self.add(bias, mask, name=f"{name or 'sdpa'}_bias_mask")
                mask = None
        if len(q.dims) != 4 or len(k.dims) != 4 or len(v.dims) != 4:
            raise ValueError("sdpa expects q, k, and v to be rank-4 tensors")
        if not paged_attention:
            if q.dims[0] != k.dims[0] or q.dims[0] != v.dims[0]:
                raise ValueError("sdpa batch dimensions must match")
            if k.dims[2] != v.dims[2]:
                raise ValueError("sdpa key/value sequence dimensions must match")
        if q.dims[3] != k.dims[3]:
            raise ValueError("sdpa q/k head dimensions must match")
        if paged_attention:
            if paged_attention_k_table is None or paged_attention_v_table is None:
                raise ValueError("sdpa paged_attention requires K and V page tables")
            if paged_attention_k_table.data_type not in (
                DataType.INT32,
                DataType.INT64,
            ):
                raise ValueError("sdpa paged_attention_k_table must be INT32 or INT64")
            if paged_attention_v_table.data_type not in (
                DataType.INT32,
                DataType.INT64,
            ):
                raise ValueError("sdpa paged_attention_v_table must be INT32 or INT64")
            if (
                paged_attention_max_seq_len_kv is not None
                and int(paged_attention_max_seq_len_kv) <= 0
            ):
                raise ValueError("sdpa paged_attention_max_seq_len_kv must be positive")
        if bias is not None and bias.data_type != q.data_type:
            raise ValueError("sdpa bias data type must match q")
        if mask is not None and mask.data_type not in (q.data_type, DataType.BOOLEAN):
            raise ValueError("sdpa mask data type must be boolean or match q")
        if sink_token is not None:
            if sink_token.data_type != DataType.FLOAT:
                raise ValueError("sdpa sink_token data type must be FLOAT")
            if tuple(sink_token.dims) != (1, q.dims[1], 1, 1):
                raise ValueError("sdpa sink_token dimensions must be (1, heads, 1, 1)")
        for score_name, score_tensor in (
            ("score_sum_exp", score_sum_exp),
            ("score_max", score_max),
        ):
            if score_tensor is None:
                continue
            if score_tensor.data_type != DataType.FLOAT:
                raise ValueError(f"sdpa {score_name} data type must be FLOAT")
            if tuple(score_tensor.dims) != (q.dims[0], q.dims[1], q.dims[2], 1):
                raise ValueError(
                    f"sdpa {score_name} dimensions must be "
                    "(batch, heads, query_sequence, 1)"
                )
        if seq_len_q is not None:
            _check_sdpa_seq_len("seq_len_q", seq_len_q, q.dims[0])
        if seq_len_kv is not None:
            _check_sdpa_seq_len("seq_len_kv", seq_len_kv, q.dims[0])
        for mask_tensor in (bias, mask):
            if mask_tensor is None:
                continue
            valid_shapes = {
                (q.dims[2], k.dims[2]),
                (q.dims[0] * q.dims[1], q.dims[2], k.dims[2]),
                (q.dims[0], q.dims[2], k.dims[2]),
                (1, q.dims[2], k.dims[2]),
                (q.dims[0], q.dims[1], q.dims[2], k.dims[2]),
                (q.dims[0], 1, q.dims[2], k.dims[2]),
                (1, q.dims[1], q.dims[2], k.dims[2]),
                (1, 1, q.dims[2], k.dims[2]),
                (q.dims[0], k.dims[2]),
            }
            if tuple(mask_tensor.dims) not in valid_shapes:
                raise ValueError("sdpa mask/bias dimensions are not supported")
        output = self.tensor(
            name=name or f"sdpa_{len(self.nodes)}",
            dim=(q.dims[0], q.dims[1], q.dims[2], v.dims[3]),
            data_type=self.io_data_type,
            is_virtual=True,
        )
        stats = None
        if not effective_is_inference:
            stats = self.tensor(
                name=f"{output.name}_stats",
                dim=(q.dims[0], q.dims[1], q.dims[2], 1),
                data_type=DataType.FLOAT,
                is_virtual=True,
            )
        score_mod_output = None
        if score_mod is not None:
            score_tensor = self.tensor(
                name=f"{output.name}_score",
                dim=(q.dims[0], q.dims[1], q.dims[2], k.dims[2]),
                data_type=DataType.FLOAT,
                is_virtual=True,
            )
            score_mod_start = len(self.nodes)
            score_mod_output = score_mod(self, score_tensor)
            if not isinstance(score_mod_output, GraphTensor):
                raise ValueError("sdpa score_mod must return a graph tensor")
            for score_node in self.nodes[score_mod_start:]:
                score_node.attributes["_score_mod_node"] = True
        inputs = {"q": q, "k": k, "v": v}
        scalar_attn_scale = _add_sdpa_scale_input(
            self, "sdpa", inputs, attn_scale, f"{output.name}_scale"
        )
        if bias is not None:
            inputs["bias"] = bias
        if mask is not None:
            inputs["mask"] = mask
        if block_mask is not None:
            inputs["block_mask"] = block_mask
        if seq_len_q is not None:
            inputs["seq_len_q"] = seq_len_q
        if seq_len_kv is not None:
            inputs["seq_len_kv"] = seq_len_kv
        if paged_attention_k_table is not None:
            inputs["paged_attention_k_table"] = paged_attention_k_table
        if paged_attention_v_table is not None:
            inputs["paged_attention_v_table"] = paged_attention_v_table
        if sink_token is not None:
            inputs["sink_token"] = sink_token
        if dropout_seed is not None:
            inputs["dropout_seed"] = dropout_seed
        if dropout_offset is not None:
            inputs["dropout_offset"] = dropout_offset
        outputs = {"o": output}
        if stats is not None:
            outputs["stats"] = stats
        if score_sum_exp is not None:
            outputs["score_sum_exp"] = score_sum_exp
        if score_max is not None:
            outputs["score_max"] = score_max
        if rng_dump is not None:
            outputs["rng_dump"] = rng_dump
        self.nodes.append(
            GraphNode(
                op_type="sdpa",
                name=output.name,
                inputs=inputs,
                outputs=outputs,
                attributes={
                    "attn_scale": scalar_attn_scale,
                    "causal_mask": bool(causal_mask),
                    "is_inference": bool(effective_is_inference),
                    "generate_stats": not bool(effective_is_inference),
                    "compute_data_type": (
                        DataType(compute_data_type)
                        if compute_data_type is not None
                        else self.compute_data_type
                    ),
                    "use_padding_mask": bool(
                        use_padding_mask
                        if use_padding_mask is not None
                        else (
                            mask is not None
                            and tuple(mask.dims) == (q.dims[0], k.dims[2])
                        )
                    ),
                    "use_alibi_mask": bool(use_alibi_mask),
                    "score_mod": score_mod is not None,
                    "score_mod_input_uid": (
                        None if score_mod_output is None else int(score_tensor.uid)
                    ),
                    "score_mod_output_uid": (
                        None if score_mod_output is None else int(score_mod_output.uid)
                    ),
                    "diagonal_alignment": normalized_diagonal_alignment,
                    "diagonal_band_left_bound": diagonal_left_bound,
                    "diagonal_band_right_bound": diagonal_right_bound,
                    "dropout_probability": dropout_probability,
                    "implementation": (
                        None
                        if implementation is None
                        else AttentionImplementation(implementation)
                    ),
                    "max_total_seq_len_q": max_total_seq_len_q,
                    "max_total_seq_len_kv": max_total_seq_len_kv,
                    "unfuse_fma": bool(unfuse_fma),
                    "paged_attention_max_seq_len_kv": (
                        None
                        if paged_attention_max_seq_len_kv is None
                        else int(paged_attention_max_seq_len_kv)
                    ),
                },
            )
        )
        return output, stats

    def scaled_dot_product_attention(
        self,
        q: GraphTensor,
        k: GraphTensor,
        v: GraphTensor,
        *,
        name: str | None = None,
        attn_scale: float | GraphTensor | None = None,
        bias: GraphTensor | None = None,
        mask: GraphTensor | None = None,
        causal_mask: bool = False,
        use_causal_mask: bool | None = None,
        diagonal_alignment: DiagonalAlignment | int | str | None = None,
        diagonal_band_left_bound: int | None = None,
        diagonal_band_right_bound: int | None = None,
        use_padding_mask: bool | None = None,
        use_alibi_mask: bool = False,
        use_causal_mask_bottom_right: bool | None = None,
        seq_len_q: GraphTensor | None = None,
        seq_len_kv: GraphTensor | None = None,
        cu_seq_len_q: GraphTensor | None = None,
        cu_seq_len_kv: GraphTensor | None = None,
        max_total_seq_len_q: int | None = None,
        max_total_seq_len_kv: int | None = None,
        dropout: float | None = None,
        sliding_window_length: int | None = None,
        generate_stats: bool | None = None,
        is_inference: bool | None = None,
        compute_data_type: DataType | None = None,
        implementation: AttentionImplementation | int | None = None,
        unfuse_fma: bool = False,
        block_mask: GraphTensor | None = None,
        rng_dump: GraphTensor | None = None,
        paged_attention_k_table: GraphTensor | None = None,
        paged_attention_v_table: GraphTensor | None = None,
        paged_attention_max_seq_len_kv: int | None = None,
        descale_q: GraphTensor | None = None,
        descale_k: GraphTensor | None = None,
        descale_v: GraphTensor | None = None,
        descale_s: GraphTensor | None = None,
        scale_s: GraphTensor | None = None,
        scale_o: GraphTensor | None = None,
        score_mod=None,
        score_sum_exp: GraphTensor | None = None,
        score_max: GraphTensor | None = None,
        sink_token: GraphTensor | None = None,
    ) -> tuple[GraphTensor, GraphTensor | None]:
        return self.sdpa(
            q,
            k,
            v,
            name=name,
            attn_scale=attn_scale,
            bias=bias,
            mask=mask,
            causal_mask=causal_mask,
            use_causal_mask=use_causal_mask,
            diagonal_alignment=diagonal_alignment,
            diagonal_band_left_bound=diagonal_band_left_bound,
            diagonal_band_right_bound=diagonal_band_right_bound,
            use_padding_mask=use_padding_mask,
            use_alibi_mask=use_alibi_mask,
            use_causal_mask_bottom_right=use_causal_mask_bottom_right,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            cu_seq_len_q=cu_seq_len_q,
            cu_seq_len_kv=cu_seq_len_kv,
            max_total_seq_len_q=max_total_seq_len_q,
            max_total_seq_len_kv=max_total_seq_len_kv,
            dropout=dropout,
            sliding_window_length=sliding_window_length,
            generate_stats=generate_stats,
            is_inference=is_inference,
            compute_data_type=compute_data_type,
            implementation=implementation,
            unfuse_fma=unfuse_fma,
            block_mask=block_mask,
            rng_dump=rng_dump,
            paged_attention_k_table=paged_attention_k_table,
            paged_attention_v_table=paged_attention_v_table,
            paged_attention_max_seq_len_kv=paged_attention_max_seq_len_kv,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_s=descale_s,
            scale_s=scale_s,
            scale_o=scale_o,
            score_mod=score_mod,
            score_sum_exp=score_sum_exp,
            score_max=score_max,
            sink_token=sink_token,
        )

    def sdpa_mxfp8(
        self,
        q: GraphTensor,
        k: GraphTensor,
        v: GraphTensor,
        descale_q: GraphTensor,
        descale_k: GraphTensor,
        descale_v: GraphTensor,
        *,
        descale_s: GraphTensor | None = None,
        scale_s: GraphTensor | None = None,
        scale_o: GraphTensor | None = None,
        name: str | None = None,
        attn_scale: float | GraphTensor | None = None,
        bias: GraphTensor | None = None,
        mask: GraphTensor | None = None,
        causal_mask: bool = False,
        use_causal_mask: bool | None = None,
        use_causal_mask_bottom_right: bool = False,
        diagonal_alignment: DiagonalAlignment | int | str | None = None,
        diagonal_band_left_bound: int | None = None,
        diagonal_band_right_bound: int | None = None,
        use_padding_mask: bool | None = None,
        generate_stats: bool | None = None,
        is_inference: bool | None = None,
        compute_data_type: DataType | None = None,
        sink_token: GraphTensor | None = None,
        unfuse_fma: bool = False,
    ) -> tuple[GraphTensor, GraphTensor | None]:
        for tensor_name, tensor in (
            ("descale_q", descale_q),
            ("descale_k", descale_k),
            ("descale_v", descale_v),
        ):
            if not isinstance(tensor, GraphTensor):
                raise ValueError(f"sdpa_mxfp8 requires {tensor_name}")
        if unfuse_fma:
            raise NotImplementedError("sdpa_mxfp8 unfuse_fma is not supported")
        output, stats = self.sdpa(
            q,
            k,
            v,
            name=name,
            attn_scale=attn_scale,
            bias=bias,
            mask=mask,
            causal_mask=causal_mask,
            use_causal_mask=use_causal_mask,
            use_causal_mask_bottom_right=use_causal_mask_bottom_right,
            diagonal_alignment=diagonal_alignment,
            diagonal_band_left_bound=diagonal_band_left_bound,
            diagonal_band_right_bound=diagonal_band_right_bound,
            use_padding_mask=use_padding_mask,
            generate_stats=generate_stats,
            is_inference=is_inference,
            compute_data_type=compute_data_type,
            sink_token=sink_token,
        )
        node = self.nodes[-1]
        node.inputs.update(
            {
                "descale_q": descale_q,
                "descale_k": descale_k,
                "descale_v": descale_v,
            }
        )
        for tensor_name, tensor in (
            ("descale_s", descale_s),
            ("scale_s", scale_s),
            ("scale_o", scale_o),
        ):
            if tensor is not None:
                node.inputs[tensor_name] = tensor
        amax_o = self.tensor(
            name=f"{output.name}_amax_o",
            dim=(1,),
            data_type=DataType.FLOAT,
            is_virtual=True,
        )
        node.outputs["amax_o"] = amax_o
        node.attributes["mxfp8"] = True
        return output, stats, amax_o

    def sdpa_fp8(
        self,
        q: GraphTensor,
        k: GraphTensor,
        v: GraphTensor,
        *,
        descale_q: GraphTensor,
        descale_k: GraphTensor,
        descale_v: GraphTensor,
        descale_s: GraphTensor,
        scale_s: GraphTensor,
        scale_o: GraphTensor,
        name: str | None = None,
        attn_scale: float | GraphTensor | None = None,
        bias: GraphTensor | None = None,
        mask: GraphTensor | None = None,
        causal_mask: bool = False,
        use_causal_mask: bool | None = None,
        use_causal_mask_bottom_right: bool = False,
        use_alibi_mask: bool = False,
        use_padding_mask: bool | None = None,
        seq_len_q: GraphTensor | None = None,
        seq_len_kv: GraphTensor | None = None,
        sliding_window: int | None = None,
        diagonal_alignment: DiagonalAlignment | int | str | None = None,
        left_bound: int | None = None,
        right_bound: int | None = None,
        dropout=None,
        rng_dump: GraphTensor | None = None,
        paged_attention_k_table: GraphTensor | None = None,
        paged_attention_v_table: GraphTensor | None = None,
        paged_attention_max_seq_len_kv: int | None = None,
        generate_stats: bool | None = None,
        is_inference: bool | None = None,
        compute_data_type: DataType | None = None,
        fn=None,
        score_max: GraphTensor | None = None,
        score_sum_exp: GraphTensor | None = None,
        sink_token: GraphTensor | None = None,
    ) -> tuple[GraphTensor, GraphTensor | None, GraphTensor, GraphTensor]:
        output, stats, amax_o = self.sdpa_mxfp8(
            q,
            k,
            v,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_s=descale_s,
            scale_s=scale_s,
            scale_o=scale_o,
            name=name,
            attn_scale=attn_scale,
            bias=bias,
            mask=mask,
            causal_mask=causal_mask,
            use_causal_mask=use_causal_mask,
            use_causal_mask_bottom_right=use_causal_mask_bottom_right,
            diagonal_alignment=diagonal_alignment,
            diagonal_band_left_bound=left_bound,
            diagonal_band_right_bound=right_bound,
            use_padding_mask=use_padding_mask,
            generate_stats=generate_stats,
            is_inference=is_inference,
            compute_data_type=compute_data_type,
            sink_token=sink_token,
        )
        unsupported = {
            "use_alibi_mask": use_alibi_mask,
            "seq_len_q": seq_len_q,
            "seq_len_kv": seq_len_kv,
            "sliding_window": sliding_window,
            "dropout": dropout,
            "rng_dump": rng_dump,
            "paged_attention_k_table": paged_attention_k_table,
            "paged_attention_v_table": paged_attention_v_table,
            "fn": fn,
            "score_max": score_max,
            "score_sum_exp": score_sum_exp,
        }
        for option, value in unsupported.items():
            if value not in (None, False, 0):
                raise NotImplementedError(f"sdpa_fp8 {option} is not supported")
        amax_s = self.tensor(
            name=f"{output.name}_amax_s",
            dim=(1,),
            data_type=DataType.FLOAT,
            is_virtual=True,
        )
        node = self.nodes[-1]
        node.outputs["amax_s"] = amax_s
        node.attributes["fp8"] = True
        node.attributes["paged_attention_max_seq_len_kv"] = (
            None
            if paged_attention_max_seq_len_kv is None
            else int(paged_attention_max_seq_len_kv)
        )
        return output, stats, amax_s, amax_o

    def sdpa_mxfp8_backward(
        self,
        q=None,
        q_T=None,
        k=None,
        k_T=None,
        v=None,
        o_f16=None,
        dO_f16=None,
        dO=None,
        dO_T=None,
        stats=None,
        descale_q=None,
        descale_q_T=None,
        descale_k=None,
        descale_k_T=None,
        descale_v=None,
        descale_dO=None,
        descale_dO_T=None,
        *,
        attn_scale=None,
        use_padding_mask: bool = False,
        seq_len_q=None,
        seq_len_kv=None,
        use_causal_mask: bool = False,
        use_causal_mask_bottom_right: bool = False,
        diagonal_alignment: DiagonalAlignment | int | str | None = None,
        left_bound=None,
        right_bound=None,
        use_deterministic_algorithm: bool = False,
        dropout=None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        sink_token=None,
        dSink_token=None,
    ):
        if v is None and all(
            isinstance(tensor, GraphTensor) for tensor in (q_T, k, k_T)
        ):
            # Compatibility for positional calls:
            # sdpa_mxfp8_backward(q, k, v, o, ...)
            q_T, k, k_T, v, o_f16 = None, q_T, None, k, k_T
        d_o = dO_f16 if dO_f16 is not None else dO
        d_q, d_k, d_v = self._sdpa_fp8_backward_symbolic(
            "sdpa_mxfp8_backward",
            q=q,
            k=k,
            v=v,
            o=o_f16,
            d_o=d_o,
            stats=stats,
            name=name,
            attn_scale=attn_scale,
            use_padding_mask=use_padding_mask,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            use_causal_mask=use_causal_mask,
            use_causal_mask_bottom_right=use_causal_mask_bottom_right,
            diagonal_alignment=diagonal_alignment,
            left_bound=left_bound,
            right_bound=right_bound,
            use_deterministic_algorithm=use_deterministic_algorithm,
            dropout=dropout,
            compute_data_type=compute_data_type,
            extra_inputs={
                "q_T": q_T,
                "k_T": k_T,
                "dO_T": dO_T,
                "descale_q": descale_q,
                "descale_q_T": descale_q_T,
                "descale_k": descale_k,
                "descale_k_T": descale_k_T,
                "descale_v": descale_v,
                "descale_dO": descale_dO,
                "descale_dO_T": descale_dO_T,
                "sink_token": sink_token,
                "dSink_token": dSink_token,
            },
            attributes={"mxfp8_bwd": True},
        )
        node = self.nodes[-1]
        amax_dq = self.tensor(
            name=f"{node.name}_amax_dq",
            dim=(1,),
            data_type=DataType.FLOAT,
            is_virtual=True,
        )
        amax_dk = self.tensor(
            name=f"{node.name}_amax_dk",
            dim=(1,),
            data_type=DataType.FLOAT,
            is_virtual=True,
        )
        amax_dv = self.tensor(
            name=f"{node.name}_amax_dv",
            dim=(1,),
            data_type=DataType.FLOAT,
            is_virtual=True,
        )
        node.outputs.update(
            {"amax_dQ": amax_dq, "amax_dK": amax_dk, "amax_dV": amax_dv}
        )
        return d_q, d_k, d_v, amax_dq, amax_dk, amax_dv

    def sdpa_fp8_backward(
        self,
        q=None,
        k=None,
        v=None,
        o=None,
        dO=None,
        stats=None,
        descale_q=None,
        descale_k=None,
        descale_v=None,
        descale_o=None,
        descale_dO=None,
        descale_s=None,
        descale_dP=None,
        scale_s=None,
        scale_dQ=None,
        scale_dK=None,
        scale_dV=None,
        scale_dP=None,
        *,
        attn_scale=None,
        use_padding_mask: bool = False,
        seq_len_q=None,
        seq_len_kv=None,
        use_causal_mask: bool = False,
        use_causal_mask_bottom_right: bool = False,
        diagonal_alignment: DiagonalAlignment | int | str | None = None,
        left_bound=None,
        right_bound=None,
        use_deterministic_algorithm: bool = False,
        dropout=None,
        compute_data_type: DataType | None = None,
        name: str | None = None,
        sink_token=None,
        dSink_token=None,
    ):
        d_q, d_k, d_v = self._sdpa_fp8_backward_symbolic(
            "sdpa_fp8_backward",
            q=q,
            k=k,
            v=v,
            o=o,
            d_o=dO,
            stats=stats,
            name=name,
            attn_scale=attn_scale,
            use_padding_mask=use_padding_mask,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            use_causal_mask=use_causal_mask,
            use_causal_mask_bottom_right=use_causal_mask_bottom_right,
            diagonal_alignment=diagonal_alignment,
            left_bound=left_bound,
            right_bound=right_bound,
            use_deterministic_algorithm=use_deterministic_algorithm,
            dropout=dropout,
            compute_data_type=compute_data_type,
            extra_inputs={
                "descale_q": descale_q,
                "descale_k": descale_k,
                "descale_v": descale_v,
                "descale_o": descale_o,
                "descale_dO": descale_dO,
                "descale_s": descale_s,
                "descale_dP": descale_dP,
                "scale_s": scale_s,
                "scale_dQ": scale_dQ,
                "scale_dK": scale_dK,
                "scale_dV": scale_dV,
                "scale_dP": scale_dP,
                "sink_token": sink_token,
                "dSink_token": dSink_token,
            },
            attributes={"fp8_bwd": True},
        )
        node = self.nodes[-1]
        amax_dq = self.tensor(
            name=f"{node.name}_amax_dq",
            dim=(1,),
            data_type=DataType.FLOAT,
            is_virtual=True,
        )
        amax_dk = self.tensor(
            name=f"{node.name}_amax_dk",
            dim=(1,),
            data_type=DataType.FLOAT,
            is_virtual=True,
        )
        amax_dv = self.tensor(
            name=f"{node.name}_amax_dv",
            dim=(1,),
            data_type=DataType.FLOAT,
            is_virtual=True,
        )
        amax_dp = self.tensor(
            name=f"{node.name}_amax_dp",
            dim=(1,),
            data_type=DataType.FLOAT,
            is_virtual=True,
        )
        node.outputs.update(
            {
                "amax_dQ": amax_dq,
                "amax_dK": amax_dk,
                "amax_dV": amax_dv,
                "amax_dP": amax_dp,
            }
        )
        return d_q, d_k, d_v, amax_dq, amax_dk, amax_dv, amax_dp

    def _sdpa_fp8_backward_symbolic(
        self,
        api_name: str,
        *,
        q: GraphTensor | None,
        k: GraphTensor | None,
        v: GraphTensor | None,
        o: GraphTensor | None,
        d_o: GraphTensor | None,
        stats: GraphTensor | None,
        name: str | None,
        attn_scale: float | GraphTensor | None,
        use_padding_mask: bool | None,
        seq_len_q: GraphTensor | None,
        seq_len_kv: GraphTensor | None,
        use_causal_mask: bool,
        use_causal_mask_bottom_right: bool,
        diagonal_alignment: DiagonalAlignment | int | str | None,
        left_bound,
        right_bound,
        use_deterministic_algorithm: bool,
        dropout,
        compute_data_type: DataType | None,
        extra_inputs: Mapping[str, GraphTensor | None],
        attributes: Mapping[str, object],
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor]:
        if dropout not in (None, False, 0, 0.0):
            raise NotImplementedError(f"{api_name} dropout is not supported")
        if use_causal_mask_bottom_right:
            raise NotImplementedError(
                f"{api_name} bottom-right causal mask is not supported"
            )
        causal_mask = _resolve_sdpa_causal_mask(
            api_name,
            use_causal_mask,
            None,
            diagonal_alignment,
            left_bound,
            right_bound,
        )
        for tensor, label in (
            (q, "q"),
            (k, "k"),
            (v, "v"),
            (o, "o"),
            (d_o, "dO"),
            (stats, "stats"),
        ):
            if not isinstance(tensor, GraphTensor):
                raise ValueError(f"{api_name} requires {label}")
        for tensor, label in (
            (q, "q"),
            (k, "k"),
            (v, "v"),
            (o, "o"),
            (d_o, "dO"),
            (stats, "stats"),
        ):
            if len(tensor.dims) != 4:
                raise ValueError(f"{api_name} expects {label} to be rank-4")
        if q.dims[0] != k.dims[0] or q.dims[0] != v.dims[0]:
            raise ValueError(f"{api_name} batch dimensions must match")
        if q.dims[3] != k.dims[3]:
            raise ValueError(f"{api_name} q/k head dimensions must match")
        if k.dims[2] != v.dims[2]:
            raise ValueError(f"{api_name} key/value sequence dimensions must match")
        if o.dims != (q.dims[0], q.dims[1], q.dims[2], v.dims[3]):
            raise ValueError(f"{api_name} output tensor dimensions are invalid")
        if d_o.dims != o.dims:
            raise ValueError(f"{api_name} dO dimensions must match o")
        if stats.dims != (q.dims[0], q.dims[1], q.dims[2], 1):
            raise ValueError(f"{api_name} stats dimensions are invalid")
        if seq_len_q is not None:
            _check_sdpa_seq_len("seq_len_q", seq_len_q, q.dims[0])
        if seq_len_kv is not None:
            _check_sdpa_seq_len("seq_len_kv", seq_len_kv, q.dims[0])
        base = name or f"{api_name}_{len(self.nodes)}"
        d_q = self.tensor(
            name=f"{base}_dq", dim=q.dims, data_type=q.data_type, is_virtual=True
        )
        d_k = self.tensor(
            name=f"{base}_dk", dim=k.dims, data_type=k.data_type, is_virtual=True
        )
        d_v = self.tensor(
            name=f"{base}_dv", dim=v.dims, data_type=v.data_type, is_virtual=True
        )
        inputs = {"q": q, "k": k, "v": v, "o": o, "d_o": d_o, "stats": stats}
        scalar_attn_scale = _add_sdpa_scale_input(
            self, api_name, inputs, attn_scale, f"{base}_scale"
        )
        if seq_len_q is not None:
            inputs["seq_len_q"] = seq_len_q
        if seq_len_kv is not None:
            inputs["seq_len_kv"] = seq_len_kv
        for input_name, tensor in extra_inputs.items():
            if tensor is not None:
                inputs[input_name] = tensor
        node_attributes = {
            "attn_scale": scalar_attn_scale,
            "causal_mask": bool(causal_mask),
            "use_padding_mask": bool(use_padding_mask),
            "deterministic": bool(use_deterministic_algorithm),
            "compute_data_type": (
                DataType(compute_data_type)
                if compute_data_type is not None
                else self.compute_data_type
            ),
        }
        node_attributes.update(attributes)
        self.nodes.append(
            GraphNode(
                op_type="sdpa_bwd",
                name=base,
                inputs=inputs,
                outputs={"d_q": d_q, "d_k": d_k, "d_v": d_v},
                attributes=node_attributes,
            )
        )
        return d_q, d_k, d_v

    def sdpa_backward(
        self,
        q: GraphTensor,
        k: GraphTensor,
        v: GraphTensor,
        o: GraphTensor,
        d_o: GraphTensor | None = None,
        stats: GraphTensor | None = None,
        *,
        dO: GraphTensor | None = None,
        name: str | None = None,
        attn_scale: float | GraphTensor | None = None,
        bias: GraphTensor | None = None,
        d_bias: GraphTensor | None = None,
        dBias: GraphTensor | None = None,
        dbias: GraphTensor | None = None,
        mask: GraphTensor | None = None,
        causal_mask: bool = False,
        use_causal_mask: bool | None = None,
        diagonal_alignment: DiagonalAlignment | int | str | None = None,
        diagonal_band_left_bound: int | None = None,
        diagonal_band_right_bound: int | None = None,
        use_padding_mask: bool | None = None,
        use_alibi_mask: bool = False,
        use_causal_mask_bottom_right: bool | None = None,
        seq_len_q: GraphTensor | None = None,
        seq_len_kv: GraphTensor | None = None,
        cu_seq_len_q: GraphTensor | None = None,
        cu_seq_len_kv: GraphTensor | None = None,
        max_total_seq_len_q: int | None = None,
        max_total_seq_len_kv: int | None = None,
        dropout: float | None = None,
        sliding_window_length: int | None = None,
        deterministic: bool = False,
        use_deterministic_algorithm: bool | None = None,
        compute_data_type: DataType | None = None,
        rng_dump: GraphTensor | None = None,
        descale_q: GraphTensor | None = None,
        descale_k: GraphTensor | None = None,
        descale_v: GraphTensor | None = None,
        descale_o: GraphTensor | None = None,
        descale_dO: GraphTensor | None = None,
        descale_s: GraphTensor | None = None,
        descale_dP: GraphTensor | None = None,
        scale_s: GraphTensor | None = None,
        scale_dQ: GraphTensor | None = None,
        scale_dK: GraphTensor | None = None,
        scale_dV: GraphTensor | None = None,
        scale_dP: GraphTensor | None = None,
        score_mod=None,
        score_mod_bprop=None,
        sink_token: GraphTensor | None = None,
        dSink_token: GraphTensor | None = None,
    ) -> tuple[GraphTensor, GraphTensor, GraphTensor]:
        seq_len_q = _resolve_tensor_alias(
            "sdpa_backward", "seq_len_q", seq_len_q, "cu_seq_len_q", cu_seq_len_q
        )
        seq_len_kv = _resolve_tensor_alias(
            "sdpa_backward",
            "seq_len_kv",
            seq_len_kv,
            "cu_seq_len_kv",
            cu_seq_len_kv,
        )
        causal_mask = _resolve_sdpa_causal_mask(
            "sdpa_backward",
            causal_mask,
            use_causal_mask,
            diagonal_alignment,
            diagonal_band_left_bound,
            diagonal_band_right_bound,
        )
        fake_cuda_cpu_remap = False
        try:
            import torch

            fake_cuda_cpu_remap = "Fake CPU CUDA" in str(torch.cuda.get_device_name())
        except Exception:
            pass
        if (
            diagonal_alignment is not None
            or diagonal_band_left_bound is not None
            or diagonal_band_right_bound is not None
        ):
            alignment = _normalize_diagonal_alignment(diagonal_alignment)
            right_bound = (
                None
                if diagonal_band_right_bound is None
                else int(diagonal_band_right_bound)
            )
            if (
                alignment != DiagonalAlignment.TOP_LEFT
                or diagonal_band_left_bound is not None
                or right_bound not in (None, 0)
            ) and not fake_cuda_cpu_remap:
                raise NotImplementedError(
                    "sdpa_backward only supports top-left causal diagonal masking "
                    "through diagonal_band_right_bound=0"
                )
        unsupported_options = {
            "use_alibi_mask": use_alibi_mask,
            "use_causal_mask_bottom_right": use_causal_mask_bottom_right,
            "seq_len_q": seq_len_q,
            "seq_len_kv": seq_len_kv,
            "max_total_seq_len_q": max_total_seq_len_q,
            "max_total_seq_len_kv": max_total_seq_len_kv,
            "dropout": dropout,
            "sliding_window_length": sliding_window_length,
            "rng_dump": rng_dump,
            "descale_q": descale_q,
            "descale_k": descale_k,
            "descale_v": descale_v,
            "descale_o": descale_o,
            "descale_dO": descale_dO,
            "descale_s": descale_s,
            "descale_dP": descale_dP,
            "scale_s": scale_s,
            "scale_dQ": scale_dQ,
            "scale_dK": scale_dK,
            "scale_dV": scale_dV,
            "scale_dP": scale_dP,
            "score_mod": score_mod,
            "score_mod_bprop": score_mod_bprop,
        }
        if not fake_cuda_cpu_remap:
            unsupported_options["sink_token"] = sink_token
            unsupported_options["dSink_token"] = dSink_token
        _reject_unsupported_sdpa_options("sdpa_backward", **unsupported_options)
        if use_deterministic_algorithm is not None:
            if deterministic and bool(use_deterministic_algorithm) != bool(
                deterministic
            ):
                raise ValueError(
                    "sdpa_backward deterministic and "
                    "use_deterministic_algorithm disagree"
                )
            deterministic = bool(use_deterministic_algorithm)
        if d_o is not None and dO is not None and d_o is not dO:
            raise ValueError("sdpa_backward d_o and dO disagree")
        d_o = d_o if d_o is not None else dO
        if d_o is None:
            raise ValueError("sdpa_backward requires d_o or dO")
        if stats is None:
            raise ValueError("sdpa_backward requires stats")
        if d_bias is not None and dbias is not None and d_bias is not dbias:
            raise ValueError("sdpa_backward d_bias and dbias disagree")
        if d_bias is not None and dBias is not None and d_bias is not dBias:
            raise ValueError("sdpa_backward d_bias and dBias disagree")
        if dbias is not None and dBias is not None and dbias is not dBias:
            raise ValueError("sdpa_backward dbias and dBias disagree")
        d_bias = d_bias if d_bias is not None else dbias
        d_bias = d_bias if d_bias is not None else dBias
        for tensor, label in (
            (q, "q"),
            (k, "k"),
            (v, "v"),
            (o, "o"),
            (d_o, "d_o"),
        ):
            if len(tensor.dims) != 4:
                raise ValueError(f"sdpa_backward expects {label} to be rank-4")
        if len(stats.dims) != 4:
            raise ValueError("sdpa_backward expects stats to be rank-4")
        if q.dims[0] != k.dims[0] or q.dims[0] != v.dims[0]:
            raise ValueError("sdpa_backward batch dimensions must match")
        if k.dims[1] != v.dims[1] or q.dims[1] % k.dims[1] != 0:
            raise ValueError("sdpa_backward head dimensions must match or divide q")
        if k.dims[2] != v.dims[2]:
            raise ValueError("sdpa_backward key/value sequence dimensions must match")
        if q.dims[3] != k.dims[3]:
            raise ValueError("sdpa_backward q/k head dimensions must match")
        if o.dims != (q.dims[0], q.dims[1], q.dims[2], v.dims[3]):
            raise ValueError("sdpa_backward output tensor dimensions are invalid")
        if d_o.dims != o.dims:
            raise ValueError("sdpa_backward d_o dimensions must match o")
        if stats.dims != (q.dims[0], q.dims[1], q.dims[2], 1):
            raise ValueError("sdpa_backward stats dimensions are invalid")
        if stats.data_type != DataType.FLOAT:
            raise ValueError("sdpa_backward stats data type must be FLOAT")
        for tensor, label in ((k, "k"), (v, "v"), (o, "o"), (d_o, "d_o")):
            if tensor.data_type != q.data_type:
                raise ValueError(f"sdpa_backward {label} data type must match q")
        if bias is not None and bias.data_type != q.data_type:
            raise ValueError("sdpa_backward bias data type must match q")
        if d_bias is not None and d_bias.data_type != q.data_type:
            raise ValueError("sdpa_backward d_bias data type must match q")
        if mask is not None and mask.data_type not in (q.data_type, DataType.BOOLEAN):
            raise ValueError("sdpa_backward mask data type must be boolean or match q")
        if sink_token is not None:
            if sink_token.data_type != DataType.FLOAT:
                raise ValueError("sdpa_backward sink_token data type must be FLOAT")
            if tuple(sink_token.dims) != (1, q.dims[1], 1, 1):
                raise ValueError(
                    "sdpa_backward sink_token dimensions must be (1, heads, 1, 1)"
                )
        if dSink_token is not None:
            if dSink_token.data_type != DataType.FLOAT:
                raise ValueError("sdpa_backward dSink_token data type must be FLOAT")
            if tuple(dSink_token.dims) != (1, q.dims[1], 1, 1):
                raise ValueError(
                    "sdpa_backward dSink_token dimensions must be (1, heads, 1, 1)"
                )
        if seq_len_q is not None:
            _check_sdpa_seq_len("seq_len_q", seq_len_q, q.dims[0])
        if seq_len_kv is not None:
            _check_sdpa_seq_len("seq_len_kv", seq_len_kv, q.dims[0])

        def check_attention_shape(tensor: GraphTensor, label: str):
            valid_shapes = {
                (q.dims[2], k.dims[2]),
                (q.dims[0] * q.dims[1], q.dims[2], k.dims[2]),
                (q.dims[0], q.dims[2], k.dims[2]),
                (1, q.dims[2], k.dims[2]),
                (q.dims[0], q.dims[1], q.dims[2], k.dims[2]),
                (q.dims[0], 1, q.dims[2], k.dims[2]),
                (1, q.dims[1], q.dims[2], k.dims[2]),
                (1, 1, q.dims[2], k.dims[2]),
                (q.dims[0], k.dims[2]),
            }
            if tuple(tensor.dims) not in valid_shapes:
                raise ValueError(f"sdpa_backward {label} dimensions are not supported")

        for tensor, label in ((bias, "bias"), (d_bias, "d_bias"), (mask, "mask")):
            if tensor is not None:
                check_attention_shape(tensor, label)

        base = name or f"sdpa_bwd_{len(self.nodes)}"
        score_mod_output = None
        if score_mod is not None:
            score_tensor = self.tensor(
                name=f"{base}_score",
                dim=(q.dims[0], q.dims[1], q.dims[2], k.dims[2]),
                data_type=DataType.FLOAT,
                is_virtual=True,
            )
            score_mod_start = len(self.nodes)
            score_mod_output = score_mod(self, score_tensor)
            if not isinstance(score_mod_output, GraphTensor):
                raise ValueError("sdpa_backward score_mod must return a graph tensor")
            for score_node in self.nodes[score_mod_start:]:
                score_node.attributes["_score_mod_node"] = True
        score_mod_bprop_output = None
        if score_mod_bprop is not None:
            d_p_tensor = self.tensor(
                name=f"{base}_dp",
                dim=(q.dims[0], q.dims[1], q.dims[2], k.dims[2]),
                data_type=DataType.FLOAT,
                is_virtual=True,
            )
            score_mod_bprop_start = len(self.nodes)
            score_mod_bprop_output = score_mod_bprop(self, d_p_tensor)
            if not isinstance(score_mod_bprop_output, GraphTensor):
                raise ValueError(
                    "sdpa_backward score_mod_bprop must return a graph tensor"
                )
            for score_node in self.nodes[score_mod_bprop_start:]:
                score_node.attributes["_score_mod_bprop_node"] = True
        d_q = self.tensor(
            name=f"{base}_dq", dim=q.dims, data_type=q.data_type, is_virtual=True
        )
        d_k = self.tensor(
            name=f"{base}_dk", dim=k.dims, data_type=k.data_type, is_virtual=True
        )
        d_v = self.tensor(
            name=f"{base}_dv", dim=v.dims, data_type=v.data_type, is_virtual=True
        )
        inputs = {"q": q, "k": k, "v": v, "o": o, "d_o": d_o, "stats": stats}
        scalar_attn_scale = _add_sdpa_scale_input(
            self, "sdpa_backward", inputs, attn_scale, f"{base}_scale"
        )
        if bias is not None:
            inputs["bias"] = bias
        if mask is not None:
            inputs["mask"] = mask
        if seq_len_q is not None:
            inputs["seq_len_q"] = seq_len_q
        if seq_len_kv is not None:
            inputs["seq_len_kv"] = seq_len_kv
        if sink_token is not None:
            inputs["sink_token"] = sink_token
        outputs = {"d_q": d_q, "d_k": d_k, "d_v": d_v}
        if d_bias is not None:
            outputs["d_bias"] = d_bias
        if dSink_token is not None:
            outputs["dSink_token"] = dSink_token
        self.nodes.append(
            GraphNode(
                op_type="sdpa_bwd",
                name=base,
                inputs=inputs,
                outputs=outputs,
                attributes={
                    "attn_scale": scalar_attn_scale,
                    "causal_mask": bool(causal_mask),
                    "use_padding_mask": bool(
                        use_padding_mask
                        if use_padding_mask is not None
                        else (
                            mask is not None
                            and tuple(mask.dims) == (q.dims[0], k.dims[2])
                        )
                    ),
                    "deterministic": bool(deterministic),
                    "compute_data_type": (
                        DataType(compute_data_type)
                        if compute_data_type is not None
                        else self.compute_data_type
                    ),
                    "score_mod": score_mod is not None,
                    "score_mod_input_uid": (
                        None if score_mod is None else score_tensor.uid
                    ),
                    "score_mod_output_uid": (
                        None if score_mod_output is None else score_mod_output.uid
                    ),
                    "score_mod_bprop": score_mod_bprop is not None,
                    "score_mod_bprop_input_uid": (
                        None if score_mod_bprop is None else d_p_tensor.uid
                    ),
                    "score_mod_bprop_output_uid": (
                        None
                        if score_mod_bprop_output is None
                        else score_mod_bprop_output.uid
                    ),
                    "diagonal_alignment": (
                        None
                        if diagonal_alignment is None
                        else _normalize_diagonal_alignment(diagonal_alignment)
                    ),
                    "diagonal_band_left_bound": (
                        None
                        if diagonal_band_left_bound is None
                        else int(diagonal_band_left_bound)
                    ),
                    "diagonal_band_right_bound": (
                        None
                        if diagonal_band_right_bound is None
                        else int(diagonal_band_right_bound)
                    ),
                },
            )
        )
        return d_q, d_k, d_v

    def validate(self):
        if not self.nodes:
            raise ValueError("graph contains no operations")
        produced = {
            tensor.uid for node in self.nodes for tensor in node.outputs.values()
        }
        if not produced:
            raise ValueError("graph contains no outputs")
        self._preflight_sdpa_fp8_version_support()
        self._validated = True
        return self

    def _preflight_sdpa_fp8_version_support(self):
        has_fp8_sdpa = any(
            node.op_type in ("sdpa", "sdpa_bwd")
            and (
                node.attributes.get("fp8")
                or node.attributes.get("mxfp8")
                or node.attributes.get("fp8_bwd")
                or node.attributes.get("mxfp8_bwd")
            )
            for node in self.nodes
        )
        if not has_fp8_sdpa:
            return
        version = _public_backend_version_tuple()
        if version[0] == 9 and version < (9, 21, 0):
            from . import cudnnGraphNotSupportedError

            raise cudnnGraphNotSupportedError(
                "FP8/MXFP8 SDPA requires cuDNN backend version 9.21.0 or newer"
            )

    def build_operation_graph(self, handle=None):
        if not self._validated:
            self.validate()
        if handle is None:
            handle = self.handle
        try:
            self._native = _NativeGraphLowerer(self).lower(handle)
        except Exception as exc:
            from . import _backend as backend
            from . import cudnnGraphNotSupportedError

            if isinstance(exc, backend.BackendError) and exc.status in (
                backend.MUDNN_STATUS_NOT_SUPPORTED,
                backend.MUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE,
            ):
                raise cudnnGraphNotSupportedError(str(exc), status=exc.status) from exc
            raise
        self._operation_graph_built = True
        self._lowered_external_uids = self._external_tensor_uids()
        self.handle = self._native.handle
        return self

    def create_execution_plans(self, modes: Iterable[HeurMode]):
        self.execution_plan_modes = tuple(HeurMode(mode) for mode in modes)
        return self

    def create_execution_plan(self, engine_index: int, knobs: Mapping | None = None):
        if knobs:
            raise NotImplementedError(
                "mudnn_frontend does not support custom execution-plan knobs"
            )
        return self.build_plan_at_index(int(engine_index))

    def key(self) -> bytes:
        return self.serialize()

    def _unsupported_sm100_rmsnorm_silu_reasons(self) -> list[str]:
        if HeurMode.OPENSOURCE not in self.execution_plan_modes:
            return []
        producers = {
            tensor.uid: node for node in self.nodes for tensor in node.outputs.values()
        }
        reasons: list[str] = []
        for node in self.nodes:
            if node.op_type != "pointwise":
                continue
            if PointwiseMode(node.attributes["mode"]) != PointwiseMode.SWISH_FWD:
                continue
            swish_input = node.inputs.get("x")
            if swish_input is None:
                continue
            producer = producers.get(swish_input.uid)
            if producer is None or producer.op_type != "norm_fwd":
                continue
            if NormMode(producer.attributes["mode"]) != NormMode.RMS_NORM:
                continue
            x = producer.inputs.get("x")
            if x is None or len(x.dims) < 2:
                continue
            tokens = int(x.dims[0])
            channels = int(x.dims[1])
            if (
                channels not in _SM100_RMSNORM_SILU_SUPPORTED_CHANNELS
                or tokens not in _SM100_RMSNORM_SILU_SUPPORTED_TOKENS
            ):
                reasons.append(
                    "SM100 OPENSOURCE RMSNorm+SiLU supports channels "
                    f"{sorted(_SM100_RMSNORM_SILU_SUPPORTED_CHANNELS)} and tokens "
                    f"{sorted(_SM100_RMSNORM_SILU_SUPPORTED_TOKENS)}, got "
                    f"channels={channels}, tokens={tokens}"
                )
        return reasons

    def _has_sm100_rmsnorm_silu_output_type(self, data_type: DataType) -> bool:
        if HeurMode.OPENSOURCE not in self.execution_plan_modes:
            return False
        producers = {
            tensor.uid: node for node in self.nodes for tensor in node.outputs.values()
        }
        for node in self.nodes:
            if node.op_type != "pointwise":
                continue
            if PointwiseMode(node.attributes["mode"]) != PointwiseMode.SWISH_FWD:
                continue
            swish_input = node.inputs.get("x")
            output = node.outputs.get("y")
            if swish_input is None or output is None:
                continue
            producer = producers.get(swish_input.uid)
            if producer is None or producer.op_type != "norm_fwd":
                continue
            if NormMode(producer.attributes["mode"]) != NormMode.RMS_NORM:
                continue
            if output.data_type == DataType(data_type):
                return True
        return False

    def check_support(self, handle=None):
        if not self._operation_graph_built:
            self.build_operation_graph(handle)
        if any(
            node.op_type == "sdpa"
            and (
                "paged_attention_k_table" in node.inputs
                or "paged_attention_v_table" in node.inputs
                or node.attributes.get("paged_attention_max_seq_len_kv") is not None
            )
            for node in self.nodes
        ):
            from . import cudnnGraphNotSupportedError

            raise cudnnGraphNotSupportedError(
                "SDPA paged_attention execution is not supported"
            )
        sm100_rmsnorm_silu_reasons = self._unsupported_sm100_rmsnorm_silu_reasons()
        if sm100_rmsnorm_silu_reasons:
            from . import cudnnGraphNotSupportedError

            raise cudnnGraphNotSupportedError(sm100_rmsnorm_silu_reasons[0])
        unsupported = sorted(
            {
                tensor.data_type.name
                for tensor in self.tensors
                if tensor.data_type in _UNSUPPORTED_LOW_PRECISION_EXECUTION_DATA_TYPES
            }
        )
        low_precision_cpu_fallback = isinstance(
            self._native, _CpuFallbackNativeGraph
        ) and any(
            node.op_type in ("block_scale_quantize", "block_scale_dequantize")
            for node in self.nodes
        )
        sm100_rmsnorm_silu_low_precision_cpu_fallback = (
            isinstance(self._native, _CpuFallbackNativeGraph)
            and (
                (
                    unsupported == [DataType.FP8_E4M3.name]
                    and self._has_sm100_rmsnorm_silu_output_type(DataType.FP8_E4M3)
                )
                or (
                    unsupported == [DataType.FP4_E2M1.name]
                    and self._has_sm100_rmsnorm_silu_output_type(DataType.FP4_E2M1)
                )
            )
        )
        if (
            unsupported
            and not low_precision_cpu_fallback
            and not sm100_rmsnorm_silu_low_precision_cpu_fallback
        ):
            from . import cudnnGraphNotSupportedError

            raise cudnnGraphNotSupportedError(
                "low-precision execution is not supported for data types: "
                + ", ".join(unsupported)
            )
        self._native.ensure_engine_configs()
        if not any(
            self._plan_is_selected(index)
            for index in range(len(self._native.engine_configs))
        ):
            from . import cudnnGraphNotSupportedError

            raise cudnnGraphNotSupportedError(
                "all execution plans were deselected by graph filters"
            )
        return self

    def get_execution_plan_count(self) -> int:
        if not self._operation_graph_built:
            self.build_operation_graph()
        self._native.ensure_engine_configs()
        return len(self._native.engine_configs)

    def get_engine_count(self) -> int:
        return self.get_execution_plan_count()

    def get_knobs_for_engine(self, engine_index: int):
        if not self._operation_graph_built:
            self.build_operation_graph()
        self._native.validate_engine_index(int(engine_index))
        return self._engine_knobs(int(engine_index))

    def get_engine_and_knobs_at_index(self, index: int):
        self._ensure_plan_index_available(int(index))
        engine = self._engine_descriptor(int(index))
        from . import _backend as backend

        engine_id = engine.get_int64(backend.MUDNN_ATTR_ENGINE_GLOBAL_INDEX)[0]
        return engine_id, {
            int(item.type): int(item.min_value)
            for item in self._engine_knobs(int(index))
        }

    def build_plans(self, policy=None, handle=None):
        if policy is not None:
            if policy.__class__.__name__ == "BuildPlanPolicy":
                pass
            else:
                try:
                    self.execution_plan_modes = _normalize_heur_modes(policy)
                except TypeError:
                    if handle is None:
                        handle = policy
        self.check_support(handle)
        if not self.execution_plan_modes:
            self.execution_plan_modes = (HeurMode.INSTANT,)
        for index in range(len(self._native.engine_configs)):
            if self._plan_is_selected(index):
                self._native.build_plan_at_index(index)
        self._plans_built = True
        return self

    def build_plan_at_index(self, index: int, handle=None):
        self.check_support(handle)
        self._ensure_plan_index_available(int(index))
        self._native.build_plan_at_index(int(index))
        self._plans_built = True
        return self

    def build(self, modes: Iterable[HeurMode] | None = None, handle=None):
        self.validate().build_operation_graph(handle)
        self.create_execution_plans(modes or (HeurMode.INSTANT,))
        self.check_support(handle).build_plans(handle)
        return self

    def _serialize_mudnn_payload(self) -> dict:
        payload = {
            "format": "mudnn_frontend.graph",
            "version": 1,
            "io_data_type": int(self.io_data_type),
            "intermediate_data_type": int(self.intermediate_data_type),
            "compute_data_type": int(self.compute_data_type),
            "name": self.name,
            "execution_plan_modes": [int(mode) for mode in self.execution_plan_modes],
            "next_uid": int(self._next_uid),
            "kernel_cache": (
                self.kernel_cache.serialize()
                if hasattr(self.kernel_cache, "serialize")
                else None
            ),
            "device_property": (
                self.device_property.serialize()
                if hasattr(self.device_property, "serialize")
                else None
            ),
            "sm_count": self.sm_count,
            "sm_version": self.sm_version,
            "is_dynamic_shape_enabled": bool(self.is_dynamic_shape_enabled),
            "is_override_shape_enabled": bool(self.is_override_shape_enabled),
            "tensors": [
                {
                    "uid": int(tensor.uid),
                    "name": tensor.name,
                    "dims": list(tensor.dims),
                    "strides": list(tensor.strides),
                    "data_type": int(tensor.data_type),
                    "is_virtual": bool(tensor.is_virtual),
                    "is_output": bool(tensor.is_output),
                    "is_by_value": bool(tensor.is_by_value),
                    "device": tensor.device,
                    "reordering_type": (
                        int(tensor.reordering_type)
                        if isinstance(tensor.reordering_type, IntEnum)
                        else tensor.reordering_type
                    ),
                    "alignment": int(tensor.alignment),
                    "vector_count": int(tensor.vector_count),
                    "vector_dimension": int(tensor.vector_dimension),
                    "ragged_offset_uid": (
                        int(tensor.ragged_offset.uid)
                        if tensor.ragged_offset is not None
                        else None
                    ),
                    "ragged_offset_multiplier": int(tensor.ragged_offset_multiplier),
                    "scalar_type": tensor.scalar_type,
                    "by_value": (
                        self._by_value_data[tensor.uid].value
                        if tensor.uid in self._by_value_data
                        else None
                    ),
                }
                for tensor in self.tensors
            ],
            "nodes": [
                {
                    "op_type": node.op_type,
                    "name": node.name,
                    "inputs": {
                        name: int(tensor.uid) for name, tensor in node.inputs.items()
                    },
                    "outputs": {
                        name: int(tensor.uid) for name, tensor in node.outputs.items()
                    },
                    "attributes": {
                        name: _serialize_value(value)
                        for name, value in node.attributes.items()
                    },
                }
                for node in self.nodes
            ],
        }
        return payload

    @staticmethod
    def _upstream_node_tag(op_type: str) -> str:
        return {
            "pointwise": "POINTWISE",
            "matmul": "MATMUL",
            "conv_fprop": "CONV_FPROP",
            "conv_bwd_data": "CONV_DGRAD",
            "conv_bwd_filter": "CONV_WGRAD",
            "reduction": "REDUCTION",
            "genstats": "GENSTATS",
            "resample": "RESAMPLE",
            "resample_bwd": "RESAMPLE",
            "sdpa": "SDPA",
            "sdpa_bwd": "SDPA_BWD",
            "slice": "SLICE",
            "moe_grouped_matmul": "MOE_GROUPED_MATMUL",
            "moe_grouped_matmul_bwd": "MOE_GROUPED_MATMUL_BWD",
        }.get(op_type, op_type.upper())

    @staticmethod
    def _upstream_port_name(op_type: str, port: str, output: bool) -> str:
        mappings = {
            "pointwise": {
                "x": "IN_0", "b": "IN_1", "t": "IN_2", "y": "OUT_0"
            },
            "matmul": {"a": "A", "b": "B", "c": "C"},
            "conv_fprop": {"x": "X", "w": "W", "y": "Y"},
            "conv_bwd_data": {"dy": "DY", "w": "W", "dx": "DX"},
            "conv_bwd_filter": {"dy": "DY", "x": "X", "dw": "DW"},
            "reduction": {"x": "X", "y": "Y"},
            "genstats": {"x": "X", "sum": "SUM", "sq_sum": "SQ_SUM"},
            "resample": {"x": "X", "y": "Y", "index": "Index"},
            "slice": {"x": "X", "y": "Y"},
        }
        return mappings.get(op_type, {}).get(port, port)

    @staticmethod
    def _upstream_attribute_value(name: str, value):
        enum_names = {
            "mode", "compute_data_type", "math_mode", "resample_mode",
            "padding_mode", "diagonal_alignment", "implementation",
            "forward_phase",
        }
        if name in enum_names and isinstance(value, IntEnum):
            return value.name
        if isinstance(value, IntEnum):
            return value.name
        return _serialize_value(value)

    def serialize(self) -> bytes:
        from . import backend_version_string

        mudnn_payload = self._serialize_mudnn_payload()
        tensors = {}
        for tensor in self.tensors:
            tensor_json = {
                "name": tensor.name,
                "data_type": tensor.data_type.name,
                "dim": list(tensor.dims),
                "stride": list(tensor.strides),
                "is_virtual": bool(tensor.is_virtual),
                "pass_by_value": (
                    self._by_value_data[tensor.uid].value
                    if tensor.uid in self._by_value_data
                    else None
                ),
                "is_pass_by_value": bool(tensor.is_by_value),
                "reordering_type": (
                    tensor.reordering_type.name
                    if isinstance(tensor.reordering_type, IntEnum)
                    else "NONE"
                ),
                "uid": int(tensor.uid),
                "uid_assigned": True,
            }
            if tensor.ragged_offset is not None:
                tensor_json["ragged_offset_uid"] = int(tensor.ragged_offset.uid)
                tensor_json["ragged_offset_name"] = tensor.ragged_offset.name
            tensors[str(tensor.uid)] = tensor_json

        nodes = []
        for node in self.nodes:
            attributes = {
                name: self._upstream_attribute_value(name, value)
                for name, value in node.attributes.items()
            }
            attributes.update(
                {
                    "tag": self._upstream_node_tag(node.op_type),
                    "name": node.name,
                    "inputs": {
                        self._upstream_port_name(node.op_type, name, False): int(tensor.uid)
                        for name, tensor in node.inputs.items()
                        if tensor is not None
                    },
                    "outputs": {
                        self._upstream_port_name(node.op_type, name, True): int(tensor.uid)
                        for name, tensor in node.outputs.items()
                        if tensor is not None
                    },
                }
            )
            if "compute_data_type" not in attributes:
                attributes["compute_data_type"] = self.compute_data_type.name
            nodes.append(attributes)

        payload = {
            "context": {
                "name": self.name,
                "compute_data_type": self.compute_data_type.name,
                "intermediate_data_type": self.intermediate_data_type.name,
                "io_data_type": self.io_data_type.name,
                "sm_count": -1 if self.sm_count is None else int(self.sm_count),
                "is_dynamic_shape_enabled": bool(self.is_dynamic_shape_enabled),
                "is_override_shape_enabled": bool(self.is_override_shape_enabled),
            },
            "json_version": "1.0",
            "cudnn_backend_version": backend_version_string(),
            "cudnn_frontend_version": __version__,
            "nodes": nodes,
            "tensors": tensors,
            "mudnn_payload": mudnn_payload,
            "kernel_cache": mudnn_payload["kernel_cache"],
            "device_property": mudnn_payload["device_property"],
            "sm_count": mudnn_payload["sm_count"],
            "sm_version": mudnn_payload["sm_version"],
            "is_dynamic_shape_enabled": mudnn_payload[
                "is_dynamic_shape_enabled"
            ],
            "is_override_shape_enabled": mudnn_payload[
                "is_override_shape_enabled"
            ],
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )

    @classmethod
    def _convert_upstream_payload(cls, upstream: dict) -> dict:
        context = upstream.get("context", {})

        def data_type(value):
            if value is None:
                return int(DataType.NOT_SET)
            if isinstance(value, str):
                return int(DataType[value])
            return int(value)

        tensors = []
        tensor_key_to_uid = {}
        output_uids = {
            int(ref)
            for node in upstream.get("nodes", ())
            for ref in node.get("outputs", {}).values()
            if isinstance(ref, int) or str(ref).lstrip("-").isdigit()
        }
        next_uid = 1
        for key, info in upstream.get("tensors", {}).items():
            uid = int(info.get("uid", key if str(key).lstrip("-").isdigit() else next_uid))
            next_uid = max(next_uid, uid + 1)
            tensor_key_to_uid[str(key)] = uid
            tensor_key_to_uid[str(info.get("name", key))] = uid
            tensors.append(
                {
                    "uid": uid,
                    "name": str(info.get("name", key)),
                    "dims": list(info.get("dim", ())),
                    "strides": list(info.get("stride", ())),
                    "data_type": data_type(info.get("data_type")),
                    "is_virtual": bool(info.get("is_virtual", False)),
                    "is_output": uid in output_uids and not bool(info.get("is_virtual", False)),
                    "is_by_value": bool(info.get("is_pass_by_value", False)),
                    "device": None,
                    "reordering_type": info.get("reordering_type", "NONE"),
                    "alignment": 16,
                    "vector_count": 1,
                    "vector_dimension": -1,
                    "ragged_offset_uid": info.get("ragged_offset_uid"),
                    "ragged_offset_multiplier": 1,
                    "scalar_type": None,
                    "by_value": info.get("pass_by_value"),
                }
            )

        tag_to_op = {
            "POINTWISE": "pointwise",
            "MATMUL": "matmul",
            "CONV_FPROP": "conv_fprop",
            "CONV_DGRAD": "conv_bwd_data",
            "CONV_WGRAD": "conv_bwd_filter",
            "REDUCTION": "reduction",
            "GENSTATS": "genstats",
            "RESAMPLE": "resample",
            "SDPA": "sdpa",
            "SDPA_BWD": "sdpa_bwd",
            "SLICE": "slice",
            "MOE_GROUPED_MATMUL": "moe_grouped_matmul",
            "MOE_GROUPED_MATMUL_BWD": "moe_grouped_matmul_bwd",
        }
        reverse_ports = {
            "pointwise": {"IN_0": "x", "IN_1": "b", "IN_2": "t", "OUT_0": "y"},
            "matmul": {"A": "a", "B": "b", "C": "c"},
            "conv_fprop": {"X": "x", "W": "w", "Y": "y"},
            "conv_bwd_data": {"DY": "dy", "W": "w", "DX": "dx"},
            "conv_bwd_filter": {"DY": "dy", "X": "x", "DW": "dw"},
            "reduction": {"X": "x", "Y": "y"},
            "genstats": {"X": "x", "SUM": "sum", "SQ_SUM": "sq_sum"},
            "resample": {"X": "x", "Y": "y", "Index": "index"},
            "slice": {"X": "x", "Y": "y"},
        }

        def resolve(ref):
            text = str(ref)
            return int(ref) if text.lstrip("-").isdigit() else tensor_key_to_uid[text]

        nodes = []
        for info in upstream.get("nodes", ()):
            tag = str(info.get("tag", ""))
            op_type = tag_to_op.get(tag, tag.lower())
            ports = reverse_ports.get(op_type, {})
            attrs = {
                name: value
                for name, value in info.items()
                if name not in {"tag", "name", "inputs", "outputs"}
            }
            if isinstance(attrs.get("compute_data_type"), str):
                attrs["compute_data_type"] = data_type(attrs["compute_data_type"])
            if isinstance(attrs.get("mode"), str):
                enum = PointwiseMode if op_type == "pointwise" else ReductionMode
                attrs["mode"] = int(enum[attrs["mode"]])
            if isinstance(attrs.get("math_mode"), str):
                attrs["math_mode"] = int(ConvolutionMode[attrs["math_mode"]])
            if isinstance(attrs.get("resample_mode"), str):
                attrs["resample_mode"] = int(ResampleMode[attrs["resample_mode"]])
            nodes.append(
                {
                    "op_type": op_type,
                    "name": str(info.get("name", "")),
                    "inputs": {
                        ports.get(name, name): resolve(ref)
                        for name, ref in info.get("inputs", {}).items()
                    },
                    "outputs": {
                        ports.get(name, name): resolve(ref)
                        for name, ref in info.get("outputs", {}).items()
                    },
                    "attributes": {
                        name: _serialize_value(value) for name, value in attrs.items()
                    },
                }
            )

        return {
            "format": "mudnn_frontend.graph",
            "version": 1,
            "io_data_type": data_type(context.get("io_data_type")),
            "intermediate_data_type": data_type(context.get("intermediate_data_type")),
            "compute_data_type": data_type(context.get("compute_data_type")),
            "execution_plan_modes": [],
            "next_uid": next_uid,
            "kernel_cache": None,
            "device_property": None,
            "sm_count": context.get("sm_count"),
            "sm_version": None,
            "is_dynamic_shape_enabled": bool(context.get("is_dynamic_shape_enabled", False)),
            "is_override_shape_enabled": bool(context.get("is_override_shape_enabled", False)),
            "tensors": tensors,
            "nodes": nodes,
            "name": str(context.get("name", "test_graph")),
        }

    def deserialize(self, *args, handle_=None, pyobj=None):
        if handle_ is not None or pyobj is not None:
            if args:
                raise TypeError(
                    "deserialize keyword form does not accept positional args"
                )
            args = (handle_, pyobj) if handle_ is not None else (pyobj,)
        if len(args) == 1:
            handle = self.handle
            data = args[0]
        elif len(args) == 2:
            handle, data = args
        else:
            raise TypeError("deserialize expects data or handle, data")
        if isinstance(data, dict):
            payload = data
        else:
            if isinstance(data, str):
                raw = data.encode("utf-8")
            elif isinstance(data, (bytes, bytearray)):
                raw = bytes(data)
            else:
                raw = bytes(int(item) for item in data)
            payload = json.loads(raw.decode("utf-8"))
        if "mudnn_payload" in payload:
            payload = payload["mudnn_payload"]
        elif payload.get("json_version") == "1.0":
            payload = self._convert_upstream_payload(payload)
        if payload.get("format") != "mudnn_frontend.graph":
            raise ValueError("unsupported serialized graph format")
        if int(payload.get("version", 0)) != 1:
            raise ValueError("unsupported serialized graph version")

        self.io_data_type = DataType(payload["io_data_type"])
        self.intermediate_data_type = DataType(payload["intermediate_data_type"])
        self.compute_data_type = DataType(payload["compute_data_type"])
        self.execution_plan_modes = tuple(
            HeurMode(mode) for mode in payload.get("execution_plan_modes", ())
        )
        self.tensors = []
        self._by_value_data = {}
        uid_to_tensor = {}
        for info in payload.get("tensors", ()):
            data_type = DataType(info["data_type"])
            tensor = GraphTensor(
                uid=int(info["uid"]),
                name=str(info["name"]),
                dims=tuple(int(dim) for dim in info["dims"]),
                strides=tuple(int(stride) for stride in info["strides"]),
                data_type=data_type,
                is_virtual=bool(info.get("is_virtual", False)),
                is_output=bool(info.get("is_output", False)),
                is_by_value=bool(info.get("is_by_value", False)),
                device=info.get("device"),
                reordering_type=info.get("reordering_type"),
                alignment=int(info.get("alignment", 16)),
                vector_count=int(info.get("vector_count", 1)),
                vector_dimension=int(info.get("vector_dimension", -1)),
                ragged_offset_multiplier=int(info.get("ragged_offset_multiplier", 1)),
                scalar_type=info.get("scalar_type"),
            )
            tensor._graph = self
            self.tensors.append(tensor)
            uid_to_tensor[tensor.uid] = tensor
            if tensor.is_by_value and info.get("by_value") is not None:
                self._by_value_data[tensor.uid] = _make_scalar_storage(
                    info["by_value"], data_type
                )
        for info in payload.get("tensors", ()):
            ragged_offset_uid = info.get("ragged_offset_uid")
            if ragged_offset_uid is not None:
                uid_to_tensor[int(info["uid"])].ragged_offset = uid_to_tensor[
                    int(ragged_offset_uid)
                ]

        self.nodes = []
        for info in payload.get("nodes", ()):
            try:
                inputs = {
                    name: uid_to_tensor[int(uid)]
                    for name, uid in info.get("inputs", {}).items()
                }
                outputs = {
                    name: uid_to_tensor[int(uid)]
                    for name, uid in info.get("outputs", {}).items()
                }
            except KeyError as exc:
                raise ValueError(f"serialized graph references unknown tensor {exc}")
            self.nodes.append(
                GraphNode(
                    op_type=str(info["op_type"]),
                    name=str(info["name"]),
                    inputs=inputs,
                    outputs=outputs,
                    attributes={
                        name: _deserialize_value(value)
                        for name, value in info.get("attributes", {}).items()
                    },
                )
            )
        self._next_uid = max(
            int(payload.get("next_uid", 1)),
            max(uid_to_tensor.keys(), default=0) + 1,
        )
        self._validated = False
        self._operation_graph_built = False
        self._plans_built = False
        self._native = None
        self.handle = handle
        self._serialized_kernel_cache = payload.get("kernel_cache")
        self._serialized_device_property = payload.get("device_property")
        self.sm_count = payload.get("sm_count")
        self.sm_version = payload.get("sm_version")
        self.name = str(payload.get("name", self.name))
        self.is_dynamic_shape_enabled = bool(
            payload.get("is_dynamic_shape_enabled", False)
        )
        self.is_override_shape_enabled = bool(
            payload.get("is_override_shape_enabled", False)
        )
        return self

    def get_workspace_size(
        self,
        handle=None,
        override_uids=None,
        override_shapes=None,
        override_strides=None,
    ) -> int:
        self._validate_dynamic_shape_overrides(
            override_uids, override_shapes, override_strides
        )
        if not self._plans_built:
            self.build_plans(handle=handle)
        sizes = [
            plan.get_int64(_NativeGraphLowerer.backend_attr_workspace_size())[0]
            for plan in self._native.plans
            if plan is not None
        ]
        return max(sizes, default=0)

    def get_workspace_size_plan_at_index(
        self,
        index: int,
        handle=None,
        override_uids=None,
        override_shapes=None,
        override_strides=None,
    ) -> int:
        self._validate_dynamic_shape_overrides(
            override_uids, override_shapes, override_strides
        )
        self.build_plan_at_index(int(index), handle)
        plan = self._native.plans[int(index)]
        return plan.get_int64(_NativeGraphLowerer.backend_attr_workspace_size())[0]

    def get_plan_name_at_index(self, index: int) -> str:
        self._ensure_plan_index_available(int(index))
        return f"mudnn_engine_config_{int(index)}"

    def _behavior_notes(self) -> list[int]:
        if any(node.op_type == "matmul" for node in self.nodes):
            return [0]
        return []

    def get_behavior_notes(self) -> list[int]:
        if not self._operation_graph_built:
            self.build_operation_graph()
        self._native.ensure_engine_configs()
        for index in range(len(self._native.engine_configs)):
            if self._plan_is_selected(index):
                return self._engine_behavior_notes(index)
        return []

    def get_behavior_notes_for_plan_at_index(self, index: int) -> list[int]:
        self._ensure_plan_index_available(int(index))
        return self._engine_behavior_notes(int(index))

    def select_numeric_notes(self, notes):
        self._selected_numeric_notes = tuple(int(note) for note in notes)
        return self

    def select_behavior_notes(self, notes):
        self._selected_behavior_notes = tuple(int(note) for note in notes)
        return self

    def deselect_engines(self, engine_names):
        self._deselected_engine_names = tuple(str(name) for name in engine_names)
        return self

    def deselect_numeric_notes(self, notes):
        self._deselected_numeric_notes = tuple(int(note) for note in notes)
        return self

    def deselect_behavior_notes(self, notes):
        self._deselected_behavior_notes = tuple(int(note) for note in notes)
        return self

    def deselect_workspace_greater_than(self, workspace: int):
        self._max_workspace = int(workspace)
        return self

    def deselect_shared_mem_greater_than(self, shared_memory: int):
        self._max_shared_memory = int(shared_memory)
        return self

    def autotune(self, *args, **kwargs):
        raise NotImplementedError(
            "muDNN backend does not expose execution-plan timing for autotune"
        )

    def _engine_descriptor(self, index: int):
        from . import _backend as backend

        self._native.validate_engine_index(index)
        descriptors = self._native.engine_configs[index].get_descriptors(
            backend.MUDNN_ATTR_ENGINECFG_ENGINE
        )
        if not descriptors:
            raise RuntimeError("engine config does not expose an engine descriptor")
        return descriptors[0]

    def _engine_behavior_notes(self, index: int) -> list[int]:
        from . import _backend as backend

        if not hasattr(self._native.engine_configs[index], "get_int64"):
            return self._behavior_notes()
        try:
            notes = list(
                self._engine_descriptor(index).get_enums(
                    backend.MUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                    backend.MUDNN_TYPE_BEHAVIOR_NOTE,
                )
            )
        except backend.BackendError:
            notes = []
        # The current muDNN backend does not populate engine behavior notes;
        # keep the historical graph-level answer in that case.
        return notes if notes else self._behavior_notes()

    def _engine_knobs(self, index: int):
        from . import _backend as backend
        from . import knob, KnobType

        engine = self._engine_descriptor(index)
        count = engine.get_descriptor_count(backend.MUDNN_ATTR_ENGINE_KNOB_INFO)
        if count == 0:
            return []
        descriptors = engine.get_descriptors(
            backend.MUDNN_ATTR_ENGINE_KNOB_INFO, requested=count
        )
        result = []
        for descriptor in descriptors:
            knob_type = descriptor.get_enums(
                backend.MUDNN_ATTR_KNOB_INFO_TYPE, backend.MUDNN_TYPE_KNOB_TYPE
            )[0]
            result.append(
                knob(
                    type=KnobType(knob_type),
                    min_value=descriptor.get_int64(
                        backend.MUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE
                    )[0],
                    max_value=descriptor.get_int64(
                        backend.MUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE
                    )[0],
                    stride=descriptor.get_int64(
                        backend.MUDNN_ATTR_KNOB_INFO_STRIDE
                    )[0],
                )
            )
        return result

    def _plan_is_selected(self, index: int) -> bool:
        from . import _backend as backend

        config = self._native.engine_configs[index]
        if not hasattr(config, "get_int64"):
            generated_name = f"mudnn_engine_config_{index}"
            return not any(
                blocked in generated_name or blocked == str(index)
                for blocked in self._deselected_engine_names
            )
        if config.get_int64(backend.MUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE)[0] > self._max_workspace:
            return False
        if config.get_int64(backend.MUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED)[0] > self._max_shared_memory:
            return False
        engine = self._engine_descriptor(index)
        engine_id = engine.get_int64(backend.MUDNN_ATTR_ENGINE_GLOBAL_INDEX)[0]
        generated_name = f"mudnn_engine_config_{index}"
        if any(
            blocked in generated_name or blocked == str(engine_id)
            for blocked in self._deselected_engine_names
        ):
            return False
        numeric = set(
            engine.get_enums(
                backend.MUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                backend.MUDNN_TYPE_NUMERICAL_NOTE,
            )
        )
        behavior = set(
            engine.get_enums(
                backend.MUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
                backend.MUDNN_TYPE_BEHAVIOR_NOTE,
            )
        )
        return (
            all(note in numeric for note in self._selected_numeric_notes)
            and not any(note in numeric for note in self._deselected_numeric_notes)
            and all(note in behavior for note in self._selected_behavior_notes)
            and not any(note in behavior for note in self._deselected_behavior_notes)
        )

    def _ensure_plan_index_available(self, index: int):
        if not self._operation_graph_built:
            self.build_operation_graph()
        self._native.validate_engine_index(index)
        if not self._plan_is_selected(index):
            from . import cudnnGraphNotSupportedError

            raise cudnnGraphNotSupportedError(
                f"execution plan at index {index} was deselected by graph filters"
            )

    def query_tensor_attributes_of_uid(self, uid: int) -> GraphTensor:
        target = int(uid)
        for tensor in self.tensors:
            if tensor.uid == target:
                return tensor
        raise KeyError(f"graph does not contain tensor uid {target}")

    def _validate_dynamic_shape_overrides(
        self, override_uids=None, override_shapes=None, override_strides=None
    ):
        provided = (
            override_uids is not None,
            override_shapes is not None,
            override_strides is not None,
        )
        if not any(provided):
            return
        if not all(provided):
            raise ValueError(
                "dynamic-shape overrides require override_uids, "
                "override_shapes, and override_strides"
            )
        uids = [int(uid) for uid in override_uids]
        shapes = [tuple(int(dim) for dim in shape) for shape in override_shapes]
        strides = [
            tuple(int(stride) for stride in stride) for stride in override_strides
        ]
        if len(uids) != len(shapes) or len(uids) != len(strides):
            raise ValueError("dynamic-shape override lengths must match")
        if len(set(uids)) != len(uids):
            raise ValueError("dynamic-shape override_uids must be unique")
        for uid, shape, stride in zip(uids, shapes, strides):
            tensor = self.query_tensor_attributes_of_uid(uid)
            shape_rank = len(shape)
            descriptor_rank = len(tensor.dims)
            stride_rank = len(stride)
            ragged_physical_stride = (
                tensor.ragged_offset is not None
                and shape_rank == descriptor_rank
                and stride_rank + 1 == descriptor_rank
            )
            if shape_rank != descriptor_rank or (
                stride_rank != descriptor_rank and not ragged_physical_stride
            ):
                raise ValueError(
                    "dynamic-shape override rank must match tensor descriptor"
                )
            if any(dim <= 0 for dim in shape):
                raise ValueError("dynamic-shape override dimensions must be positive")
            if any(value < 0 for value in stride):
                raise ValueError("dynamic-shape override strides must be nonnegative")

    def _prepare_variant_pack_template(self):
        if not self._operation_graph_built:
            self.build_operation_graph()
        return self

    def _get_variant_pack_uids_sorted(self) -> list[int]:
        return sorted(self._external_tensor_uids())

    def execute(
        self,
        data_ptrs: Mapping[object, int] | None = None,
        workspace=None,
        handle=None,
        *,
        variant_pack: Mapping[object, int] | None = None,
        workspace_ptr=None,
        override_uids=None,
        override_shapes=None,
        override_strides=None,
    ):
        self._validate_dynamic_shape_overrides(
            override_uids, override_shapes, override_strides
        )
        if data_ptrs is None:
            data_ptrs = variant_pack
        elif variant_pack is not None:
            raise ValueError("pass either data_ptrs or variant_pack, not both")
        if data_ptrs is None:
            raise ValueError("execute requires data pointers")
        if workspace_ptr is not None and workspace is not None:
            raise ValueError("pass either workspace or workspace_ptr, not both")
        self._set_runtime_external_uids(self._data_ptr_uids(data_ptrs))
        if self._maybe_execute_torch_cpu_fallback(data_ptrs, workspace):
            return self
        if workspace_ptr is None:
            workspace_ptr = workspace
        if workspace_ptr is not None:
            workspace_ptr = _pointer_value(workspace_ptr)
        if not self._plans_built:
            self.build_plans(handle)
        uid_to_pointer = self._normalize_data_ptrs(data_ptrs)
        self._native.execute(uid_to_pointer, workspace_ptr=workspace_ptr)
        return self

    def _maybe_execute_torch_cpu_fallback(
        self, data_ptrs: Mapping[object, int], workspace=None
    ) -> bool:
        try:
            import torch
        except ImportError:
            return False

        def fake_cuda_cpu_remap_active() -> bool:
            try:
                return "Fake CPU CUDA" in str(torch.cuda.get_device_name())
            except Exception:
                return False

        host_pointer_buffers = []
        workspace_tensor = workspace if isinstance(workspace, torch.Tensor) else None

        def tensor_from_host_pointer(pointer: int, tensor: GraphTensor):
            if not fake_cuda_cpu_remap_active():
                return None
            if int(pointer) == 0:
                return None
            storage_elements = _strided_storage_elements(tensor.dims, tensor.strides)
            if storage_elements <= 0:
                return None
            byte_count = storage_elements * _data_type_size(tensor.data_type)
            raw_type = ctypes.c_ubyte * byte_count
            raw = raw_type.from_address(int(pointer))
            host_pointer_buffers.append(raw)
            base = torch.frombuffer(
                raw,
                dtype=_torch_dtype_for_data_type(tensor.data_type),
                count=storage_elements,
            )
            return torch.as_strided(base, tuple(tensor.dims), tuple(tensor.strides))

        def logical_view(value, tensor: GraphTensor):
            if value is None:
                return None
            if tuple(value.shape) == tuple(tensor.dims) and tuple(
                value.stride()
            ) == tuple(tensor.strides):
                return value
            storage_elements = _strided_storage_elements(tensor.dims, tensor.strides)
            if storage_elements <= 0 or value.numel() < storage_elements:
                return None
            if value.is_contiguous():
                base = value.reshape(-1)
            else:
                try:
                    base = value.as_strided(
                        (value.untyped_storage().nbytes() // value.element_size(),),
                        (1,),
                    )
                except RuntimeError:
                    return None
            try:
                return torch.as_strided(base, tuple(tensor.dims), tuple(tensor.strides))
            except RuntimeError:
                return None

        def storage_flat_view(value):
            if value.is_contiguous():
                return value.reshape(-1)
            try:
                return value.as_strided(
                    (value.untyped_storage().nbytes() // value.element_size(),),
                    (1,),
                )
            except RuntimeError:
                return None

        def conv_pad_tuple(pre_padding, post_padding):
            padding = []
            for pre, post in zip(reversed(pre_padding), reversed(post_padding)):
                padding.extend((int(pre), int(post)))
            return tuple(padding)

        def broadcast_attention_tensor(value, score_shape):
            if value.dim() == 2:
                return value.reshape(1, 1, value.shape[0], value.shape[1])
            if value.dim() == 3:
                if value.shape[0] == score_shape[0] * score_shape[1]:
                    return value.reshape(
                        score_shape[0], score_shape[1], value.shape[1], value.shape[2]
                    )
                return value.unsqueeze(1)
            return value

        def block_scale_values(scale, x_shape, block_size: int, axis: int):
            expected_shape = list(x_shape)
            expected_shape[axis] = (
                int(expected_shape[axis]) + int(block_size) - 1
            ) // int(block_size)
            slices = tuple(slice(0, dim) for dim in expected_shape)
            scale = scale[slices]
            repeated = torch.repeat_interleave(
                scale.to(dtype=torch.float32), int(block_size), dim=axis
            )
            x_slices = tuple(slice(0, dim) for dim in x_shape)
            return repeated[x_slices]

        def scalar_node_value(node: GraphNode, name: str, default: float) -> float:
            if name in node.inputs:
                value = values.get(node.inputs[name].uid)
                if value is None:
                    return default
                return float(value.reshape(-1)[0].item())
            return float(node.attributes.get(name, default))

        def sdpa_scale_value(node: GraphNode, default: float) -> float:
            attr_value = node.attributes.get("attn_scale")
            scale_tensor = node.inputs.get("scale")
            if scale_tensor is not None and not scale_tensor.is_by_value:
                value = values.get(scale_tensor.uid)
                if value is not None:
                    return float(value.reshape(-1)[0].item())
            if attr_value is not None:
                return float(attr_value)
            if scale_tensor is not None:
                value = values.get(scale_tensor.uid)
                if value is not None:
                    return float(value.reshape(-1)[0].item())
            return float(default)

        def alibi_bias(scores):
            import math

            h_q = int(scores.shape[1])
            seq_q = int(scores.shape[-2])
            seq_k = int(scores.shape[-1])
            index_row = torch.arange(
                seq_q, dtype=torch.float32, device=scores.device
            ).reshape(seq_q, 1)
            index_col = torch.arange(seq_k, dtype=torch.float32, device=scores.device)
            distance = index_col - index_row
            n = 2 ** math.floor(math.log2(h_q))
            m_0 = 2.0 ** (-8.0 / n)
            slopes = torch.pow(
                torch.tensor(m_0, dtype=torch.float32, device=scores.device),
                torch.arange(1, 1 + n, dtype=torch.float32, device=scores.device),
            )
            if n < h_q:
                m_hat_0 = 2.0 ** (-4.0 / n)
                m_hat = torch.pow(
                    torch.tensor(m_hat_0, dtype=torch.float32, device=scores.device),
                    torch.arange(
                        1,
                        1 + 2 * (h_q - n),
                        2,
                        dtype=torch.float32,
                        device=scores.device,
                    ),
                )
                slopes = torch.cat([slopes, m_hat])
            return distance.reshape(1, 1, seq_q, seq_k) * slopes.reshape(1, h_q, 1, 1)

        def unpack_sdpa_block_mask(block_mask, score_shape):
            tile_m = 128
            tile_n = 128
            batch, heads, seq_q, seq_kv = (int(dim) for dim in score_shape)
            bits = (
                block_mask.to(dtype=torch.uint8)[..., None]
                & (1 << torch.arange(8, dtype=torch.uint8, device=block_mask.device))
            ) != 0
            bits = bits.reshape(
                int(block_mask.shape[0]),
                int(block_mask.shape[1]),
                int(block_mask.shape[2]),
                int(block_mask.shape[3]) * 8,
            )
            expanded = bits.unsqueeze(3).unsqueeze(5).repeat(1, 1, 1, tile_m, 1, tile_n)
            expanded = expanded.reshape(
                expanded.shape[0],
                expanded.shape[1],
                expanded.shape[2] * tile_m,
                expanded.shape[4] * tile_n,
            )
            return expanded[:batch, :heads, :seq_q, :seq_kv]

        def norm_reduce_axes(mode: NormMode, x) -> tuple[int, ...]:
            if x.dim() < 1:
                return ()
            if mode == NormMode.INSTANCE_NORM:
                if x.dim() != 4:
                    return ()
                return (2, 3)
            return tuple(range(1, x.dim()))

        by_name = {tensor.name: tensor for tensor in self.tensors}
        by_uid = {tensor.uid: tensor for tensor in self.tensors}
        provided: dict[int, object] = {}
        for uid, storage in self._by_value_data.items():
            tensor = by_uid[uid]
            provided[uid] = torch.tensor(
                [storage.value],
                dtype=_torch_dtype_for_data_type(tensor.data_type),
            )
        for key, value in data_ptrs.items():
            if key is None or value is None:
                continue
            if isinstance(key, GraphTensor):
                tensor = key
            elif isinstance(key, str):
                tensor = by_name[key]
            else:
                tensor = by_uid.get(int(key))
                if tensor is None:
                    continue
            if isinstance(value, torch.Tensor):
                provided[tensor.uid] = value
            elif isinstance(value, (int, ctypes.c_void_p)):
                pointer = _pointer_value(value)
                host_tensor = tensor_from_host_pointer(pointer, tensor)
                if host_tensor is None:
                    return False
                provided[tensor.uid] = host_tensor
            else:
                provided[tensor.uid] = value

        external_uids = self._external_tensor_uids()
        if not external_uids.issubset(provided):
            return False
        if not all(isinstance(provided[uid], torch.Tensor) for uid in external_uids):
            return False
        if any(provided[uid].device.type != "cpu" for uid in external_uids):
            return False

        values = dict(provided)
        consumed_uids = {
            tensor.uid for node in self.nodes for tensor in node.inputs.values()
        }
        consumers_by_input_uid: dict[int, list[GraphNode]] = {}
        for node in self.nodes:
            for tensor in node.inputs.values():
                consumers_by_input_uid.setdefault(tensor.uid, []).append(node)

        def is_rmsnorm_low_precision_swish_intermediate(tensor: GraphTensor) -> bool:
            for consumer in consumers_by_input_uid.get(tensor.uid, []):
                if consumer.op_type != "pointwise":
                    continue
                if PointwiseMode(consumer.attributes["mode"]) != PointwiseMode.SWISH_FWD:
                    continue
                output = consumer.outputs.get("y")
                if output is not None and output.data_type in (
                    DataType.FP8_E4M3,
                    DataType.FP4_E2M1,
                ):
                    return True
            return False

        def sm100_rmsnorm_silu_fp4_workspace_size(tokens: int, channels: int) -> int:
            rs_size = int(tokens) * 4
            rs_size_aligned = ((rs_size + 127) // 128) * 128
            fp8_scale_aligned = ((rs_size_aligned + 4 + 127) // 128) * 128
            return fp8_scale_aligned + int(tokens) * (int(channels) // 16)

        def pack_sm100_rmsnorm_silu_fp4(result, output: GraphTensor):
            if len(output.dims) < 2 or int(output.dims[1]) % 16 != 0:
                return None
            tokens = int(output.dims[0])
            channels = int(output.dims[1])
            if result.numel() != tokens * channels:
                return None
            values_f32 = result.reshape(tokens, channels).to(dtype=torch.float32)
            blocks = channels // 16
            scale = values_f32.reshape(tokens, blocks, 16).abs().amax(dim=2)
            scale = torch.clamp(scale, min=torch.finfo(torch.float32).tiny)
            scale = scale / 6.0
            scaled = values_f32.reshape(tokens, blocks, 16) / scale.unsqueeze(2)
            fp4_positive = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                dtype=torch.float32,
                device=values_f32.device,
            )
            magnitudes = scaled.abs()
            signs = (scaled < 0).to(dtype=torch.uint8)
            diffs = (magnitudes.unsqueeze(-1) - fp4_positive).abs()
            nibbles = diffs.argmin(dim=-1).to(dtype=torch.uint8) + signs * 8
            nibbles = nibbles.reshape(tokens, channels)
            packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
            if workspace_tensor is not None:
                required = sm100_rmsnorm_silu_fp4_workspace_size(tokens, channels)
                flat_workspace = workspace_tensor.reshape(-1)
                if flat_workspace.numel() >= required:
                    rs_size = tokens * 4
                    rs_size_aligned = ((rs_size + 127) // 128) * 128
                    scale_offset = ((rs_size_aligned + 4 + 127) // 128) * 128
                    scale_dtype = (
                        torch.float8_e4m3fn
                        if hasattr(torch, "float8_e4m3fn")
                        else torch.uint8
                    )
                    scale_fp8 = scale.to(dtype=scale_dtype).view(torch.uint8)
                    flat_workspace[
                        scale_offset : scale_offset + scale_fp8.numel()
                    ].copy_(scale_fp8.reshape(-1))
            return packed.reshape(-1)

        def execute_pointwise_node(node: GraphNode) -> bool:
            output = node.outputs.get("y") or node.outputs.get("dx")
            if output is None:
                return False
            mode = PointwiseMode(node.attributes["mode"])
            x = values.get(node.inputs["x"].uid)
            if x is None:
                return False

            def binary_input():
                value = values.get(node.inputs["b"].uid)
                return value

            if mode == PointwiseMode.ADD:
                b = binary_input()
                if b is None:
                    return False
                result = node.attributes["alpha1"] * x + node.attributes["alpha2"] * b
            elif mode == PointwiseMode.SUB:
                b = binary_input()
                if b is None:
                    return False
                result = node.attributes["alpha1"] * x - node.attributes["alpha2"] * b
            elif mode == PointwiseMode.MUL:
                b = binary_input()
                if b is None:
                    return False
                result = node.attributes["alpha1"] * x * node.attributes["alpha2"] * b
            elif mode == PointwiseMode.DIV:
                b = binary_input()
                if b is None:
                    return False
                result = (node.attributes["alpha1"] * x) / (
                    node.attributes["alpha2"] * b
                )
            elif mode == PointwiseMode.IDENTITY:
                result = node.attributes["alpha1"] * x
            elif mode == PointwiseMode.GEN_INDEX:
                axis = int(node.attributes["axis"])
                index = torch.arange(
                    int(x.shape[axis]), dtype=torch.int32, device=x.device
                )
                shape = [1] * x.dim()
                shape[axis] = int(x.shape[axis])
                result = index.reshape(shape).expand(tuple(x.shape))
            elif mode in (
                PointwiseMode.CMP_EQ,
                PointwiseMode.CMP_NEQ,
                PointwiseMode.CMP_GT,
                PointwiseMode.CMP_GE,
                PointwiseMode.CMP_LT,
                PointwiseMode.CMP_LE,
            ):
                b = binary_input()
                if b is None:
                    return False
                if mode == PointwiseMode.CMP_EQ:
                    result = x == b
                elif mode == PointwiseMode.CMP_NEQ:
                    result = x != b
                elif mode == PointwiseMode.CMP_GT:
                    result = x > b
                elif mode == PointwiseMode.CMP_GE:
                    result = x >= b
                elif mode == PointwiseMode.CMP_LT:
                    result = x < b
                else:
                    result = x <= b
            elif mode in (PointwiseMode.LOGICAL_AND, PointwiseMode.LOGICAL_OR):
                b = binary_input()
                if b is None:
                    return False
                lhs = x.to(dtype=torch.bool)
                rhs = b.to(dtype=torch.bool)
                result = lhs & rhs if mode == PointwiseMode.LOGICAL_AND else lhs | rhs
            elif mode == PointwiseMode.LOGICAL_NOT:
                result = ~x.to(dtype=torch.bool)
            elif mode == PointwiseMode.BINARY_SELECT:
                b = binary_input()
                t = values.get(node.inputs["t"].uid)
                if b is None or t is None:
                    return False
                result = torch.where(t.to(dtype=torch.bool), x, b)
            elif mode == PointwiseMode.SIGMOID_FWD:
                result = torch.sigmoid(x)
            elif mode == PointwiseMode.SWISH_FWD:
                beta = float(node.attributes.get("swish_beta", 1.0))
                result = x * torch.sigmoid(beta * x)
            elif mode == PointwiseMode.RELU_FWD:
                lower_clip = node.attributes.get("relu_lower_clip")
                upper_clip = node.attributes.get("relu_upper_clip")
                lower_slope = node.attributes.get("relu_lower_clip_slope")
                if lower_slope is not None and lower_slope != 0:
                    lower = 0.0 if lower_clip is None else float(lower_clip)
                    result = torch.where(
                        x > lower, x, lower + float(lower_slope) * (x - lower)
                    )
                elif lower_clip is not None:
                    result = torch.clamp(x, min=float(lower_clip))
                else:
                    result = torch.relu(x)
                if upper_clip is not None:
                    result = torch.clamp(result, max=float(upper_clip))
            elif mode == PointwiseMode.RELU_BWD:
                dy = values.get(node.inputs["dy"].uid)
                if dy is None:
                    return False
                lower_clip = node.attributes.get("relu_lower_clip")
                upper_clip = node.attributes.get("relu_upper_clip")
                lower_slope = float(node.attributes.get("relu_lower_clip_slope", 0.0))
                lower = 0.0 if lower_clip is None else float(lower_clip)
                derivative = torch.where(
                    x > lower,
                    torch.ones_like(x, dtype=dy.dtype),
                    torch.full_like(x, lower_slope, dtype=dy.dtype),
                )
                if upper_clip is not None:
                    derivative = torch.where(
                        x <= float(upper_clip), derivative, torch.zeros_like(derivative)
                    )
                result = dy * derivative
            elif mode == PointwiseMode.GELU_FWD:
                result = torch.nn.functional.gelu(x, approximate="none")
            elif mode == PointwiseMode.GELU_APPROX_TANH_FWD:
                result = torch.nn.functional.gelu(x, approximate="tanh")
            elif mode == PointwiseMode.FLOOR:
                result = torch.floor(x)
            elif mode == PointwiseMode.CEIL:
                result = torch.ceil(x)
            elif mode == PointwiseMode.EXP:
                result = torch.exp(x)
            elif mode == PointwiseMode.LOG:
                result = torch.log(x)
            else:
                return False
            if output.data_type == DataType.FP4_E2M1 and mode == PointwiseMode.SWISH_FWD:
                packed = pack_sm100_rmsnorm_silu_fp4(result, output)
                if packed is None:
                    return False
                values[output.uid] = packed
            else:
                values[output.uid] = result.to(
                    dtype=_torch_dtype_for_data_type(output.data_type)
                )
            return True

        for node in self.nodes:
            if node.attributes.get("_score_mod_node", False) or node.attributes.get(
                "_score_mod_bprop_node", False
            ):
                continue
            if node.op_type == "slice":
                x = values.get(node.inputs["x"].uid)
                if x is None:
                    return False
                slices = tuple(
                    slice(start, limit, stride)
                    for start, limit, stride in zip(
                        node.attributes["start_indices"],
                        node.attributes["limit_indices"],
                        node.attributes["strides"],
                    )
                )
                values[node.outputs["y"].uid] = x[slices]
            elif node.op_type == "matmul":
                a = values.get(node.inputs["a"].uid)
                b = values.get(node.inputs["b"].uid)
                if a is None or b is None:
                    return False
                output = node.outputs["c"]
                compute_type = DataType(node.attributes["compute_data_type"])
                if compute_type == DataType.FLOAT and output.uid in consumed_uids:
                    a = a.to(dtype=torch.float32)
                    b = b.to(dtype=torch.float32)
                values[output.uid] = torch.matmul(a, b).to(
                    dtype=_torch_dtype_for_data_type(output.data_type)
                )
            elif node.op_type == "pointwise":
                if not execute_pointwise_node(node):
                    return False
            elif node.op_type == "conv_fprop":
                x = values.get(node.inputs["x"].uid)
                w = values.get(node.inputs["w"].uid)
                if x is None or w is None or len(x.shape) < 3 or len(x.shape) > 5:
                    return False
                if (
                    ConvolutionMode(node.attributes["mode"])
                    != ConvolutionMode.CROSS_CORRELATION
                ):
                    return False
                output = node.outputs["y"]
                pre_padding = tuple(
                    node.attributes.get("pre_padding", node.attributes["padding"])
                )
                post_padding = tuple(
                    node.attributes.get("post_padding", node.attributes["padding"])
                )
                if pre_padding != post_padding:
                    x = torch.nn.functional.pad(
                        x, conv_pad_tuple(pre_padding, post_padding)
                    )
                    padding = tuple(0 for _ in pre_padding)
                else:
                    padding = pre_padding
                conv2d_fn = getattr(
                    torch.nn.functional,
                    "_mudnn_cpu_remap_orig_conv2d",
                    torch.nn.functional.conv2d,
                )
                conv_fn = {
                    1: torch.nn.functional.conv1d,
                    2: conv2d_fn,
                    3: torch.nn.functional.conv3d,
                }[len(x.shape) - 2]
                values[output.uid] = conv_fn(
                    x,
                    w,
                    padding=padding,
                    stride=tuple(node.attributes["stride"]),
                    dilation=tuple(node.attributes["dilation"]),
                    groups=int(node.attributes["group_count"]),
                ).to(dtype=_torch_dtype_for_data_type(output.data_type))
            elif node.op_type == "conv_bwd_filter":
                x = values.get(node.inputs["x"].uid)
                dy = values.get(node.inputs["dy"].uid)
                if x is None or dy is None or len(x.shape) != 4 or len(dy.shape) != 4:
                    return False
                if (
                    ConvolutionMode(node.attributes["mode"])
                    != ConvolutionMode.CROSS_CORRELATION
                ):
                    return False
                output = node.outputs["dw"]
                pre_padding = tuple(
                    node.attributes.get("pre_padding", node.attributes["padding"])
                )
                post_padding = tuple(
                    node.attributes.get("post_padding", node.attributes["padding"])
                )
                if pre_padding != post_padding:
                    x = torch.nn.functional.pad(
                        x,
                        (
                            pre_padding[1],
                            post_padding[1],
                            pre_padding[0],
                            post_padding[0],
                        ),
                    )
                    padding = (0, 0)
                else:
                    padding = pre_padding
                compute_type = DataType(node.attributes["compute_data_type"])
                if compute_type == DataType.FLOAT:
                    x = x.to(dtype=torch.float32)
                    dy = dy.to(dtype=torch.float32)
                result = torch.nn.grad.conv2d_weight(
                    x,
                    tuple(output.dims),
                    dy,
                    padding=padding,
                    stride=tuple(node.attributes["stride"]),
                    dilation=tuple(node.attributes["dilation"]),
                    groups=int(node.attributes["group_count"]),
                )
                alpha = float(node.attributes.get("alpha", 1.0))
                beta = float(node.attributes.get("beta", 0.0))
                result = alpha * result
                if beta != 0.0 and output.uid in values:
                    prior = values[output.uid]
                    if tuple(prior.shape) == tuple(output.dims):
                        result = result + beta * prior.to(dtype=result.dtype)
                    else:
                        return False
                values[output.uid] = result.to(
                    dtype=_torch_dtype_for_data_type(output.data_type)
                )
            elif node.op_type == "reduction":
                if "group_offset" in node.inputs:
                    return False
                x = values.get(node.inputs["x"].uid)
                if x is None:
                    return False
                output = node.outputs["y"]
                if len(output.dims) != len(x.shape):
                    return False
                axes = tuple(
                    index
                    for index, (x_dim, y_dim) in enumerate(
                        zip(tuple(x.shape), output.dims)
                    )
                    if int(x_dim) != int(y_dim) and int(y_dim) == 1
                )
                if node.attributes["mode"] != ReductionMode.ADD:
                    return False
                values[output.uid] = x.sum(dim=axes, keepdim=True).to(
                    dtype=_torch_dtype_for_data_type(output.data_type)
                )
            elif node.op_type == "genstats":
                x = values.get(node.inputs["x"].uid)
                if x is None:
                    return False
                sum_output = node.outputs["sum"]
                sqsum_output = node.outputs["sqsum"]
                axes = tuple(node.attributes["axis"])
                keepdim = len(sum_output.dims) == len(x.shape)
                sum_value = x.to(dtype=torch.float32).sum(dim=axes, keepdim=keepdim)
                sqsum_value = torch.square(x.to(dtype=torch.float32)).sum(
                    dim=axes, keepdim=keepdim
                )
                values[sum_output.uid] = sum_value.to(
                    dtype=_torch_dtype_for_data_type(sum_output.data_type)
                )
                values[sqsum_output.uid] = sqsum_value.to(
                    dtype=_torch_dtype_for_data_type(sqsum_output.data_type)
                )
            elif node.op_type == "block_scale_dequantize":
                x_tensor = node.inputs["x"]
                scale_tensor = node.inputs["scale"]
                x = logical_view(values.get(x_tensor.uid), x_tensor)
                scale = values.get(scale_tensor.uid)
                if x is None or scale is None:
                    return False
                output = node.outputs["y"]
                axis = int(node.attributes.get("block_axis", len(x_tensor.dims) - 1))
                block_size = int(node.attributes["block_size"])
                scale_view = logical_view(scale, scale_tensor)
                if scale_view is None:
                    scale_view = scale
                try:
                    factors = block_scale_values(
                        scale_view, tuple(x_tensor.dims), block_size, axis
                    )
                except (IndexError, RuntimeError):
                    return False
                sign = -1.0 if node.attributes.get("is_negative_scale", False) else 1.0
                result = x.to(dtype=torch.float32) * factors * sign
                values[output.uid] = result.to(
                    dtype=_torch_dtype_for_data_type(output.data_type)
                )
            elif node.op_type == "block_scale_quantize":
                x_tensor = node.inputs["x"]
                x = logical_view(values.get(x_tensor.uid), x_tensor)
                if x is None:
                    return False
                y_output = node.outputs["y"]
                scale_output = node.outputs["scale"]
                axis = int(node.attributes.get("block_axis", len(x_tensor.dims) - 1))
                block_size = int(node.attributes["block_size"])
                x_float = x.to(dtype=torch.float32)
                scale_shape = list(x_tensor.dims)
                scale_shape[axis] = (
                    int(scale_shape[axis]) + int(block_size) - 1
                ) // int(block_size)
                quantized = torch.empty_like(x_float)
                scale_result = torch.zeros(
                    tuple(scale_shape), dtype=torch.float32, device=x_float.device
                )
                for block in range(scale_shape[axis]):
                    start = block * block_size
                    end = min(start + block_size, int(x_tensor.dims[axis]))
                    x_slices = [slice(None)] * x_float.dim()
                    x_slices[axis] = slice(start, end)
                    scale_slices = [slice(None)] * x_float.dim()
                    scale_slices[axis] = block
                    block_values = x_float[tuple(x_slices)]
                    block_scale = block_values.abs().amax(dim=axis, keepdim=False)
                    block_scale = torch.where(
                        block_scale == 0.0,
                        torch.ones_like(block_scale),
                        block_scale,
                    )
                    scale_result[tuple(scale_slices)] = block_scale
                    quantized[tuple(x_slices)] = block_values / block_scale.unsqueeze(
                        axis
                    )
                values[y_output.uid] = quantized.to(
                    dtype=_torch_dtype_for_data_type(y_output.data_type)
                )
                values[scale_output.uid] = scale_result.to(
                    dtype=_torch_dtype_for_data_type(scale_output.data_type)
                )
            elif node.op_type == "norm_fwd":
                norm_mode = NormMode(node.attributes["mode"])
                if norm_mode not in (
                    NormMode.LAYER_NORM,
                    NormMode.RMS_NORM,
                    NormMode.INSTANCE_NORM,
                    NormMode.BATCH_NORM,
                ):
                    return False
                x_tensor = node.inputs["x"]
                scale_tensor = node.inputs["scale"]
                bias_tensor = node.inputs.get("bias")
                x = logical_view(values.get(x_tensor.uid), x_tensor)
                scale = logical_view(values.get(scale_tensor.uid), scale_tensor)
                bias = (
                    None
                    if bias_tensor is None
                    else logical_view(values.get(bias_tensor.uid), bias_tensor)
                )
                if (
                    x is None
                    or scale is None
                    or (bias_tensor is not None and bias is None)
                ):
                    return False
                axes = (
                    (0, 2, 3)
                    if norm_mode == NormMode.BATCH_NORM
                    else norm_reduce_axes(norm_mode, x)
                )
                if not axes:
                    return False
                epsilon = scalar_node_value(node, "epsilon", 1.0e-5)
                if norm_mode == NormMode.LAYER_NORM:
                    x_float = x.to(dtype=torch.float32)
                    mean = x_float.mean(dim=axes, keepdim=True)
                    variance = torch.square(x_float - mean).mean(dim=axes, keepdim=True)
                    inv_variance = torch.rsqrt(variance + epsilon)
                    result = (
                        (x_float - mean) * inv_variance * scale.to(dtype=torch.float32)
                    )
                    if bias is not None:
                        result = result + bias.to(dtype=torch.float32)
                elif norm_mode == NormMode.RMS_NORM:
                    mean = None
                    if is_rmsnorm_low_precision_swish_intermediate(node.outputs["y"]):
                        x_float = x.to(dtype=torch.float32)
                        variance = torch.mean(
                            x_float * x_float, dim=axes, keepdim=True
                        )
                        inv_variance = torch.rsqrt(variance + epsilon)
                        result = x_float * inv_variance * scale.to(dtype=torch.float32)
                        if bias is not None:
                            result = result + bias.to(dtype=torch.float32)
                    else:
                        variance = torch.mean(x * x, dim=axes, keepdim=True)
                        inv_variance = torch.rsqrt(variance.float() + epsilon)
                        result = scale * (x * inv_variance.to(dtype=x.dtype))
                        if bias is not None:
                            result = result + bias
                elif norm_mode == NormMode.INSTANCE_NORM:
                    if x.dim() != 4:
                        return False
                    weight = scale.reshape(int(x.shape[1]))
                    bias_1d = None if bias is None else bias.reshape(int(x.shape[1]))
                    result = torch.nn.functional.instance_norm(
                        x,
                        weight=weight,
                        bias=bias_1d,
                        eps=epsilon,
                    )
                    x_float = x.to(dtype=torch.float32)
                    mean = x_float.mean(dim=axes, keepdim=True)
                    inv_variance = torch.rsqrt(
                        torch.var(x_float, dim=axes, keepdim=True) + epsilon
                    )
                elif norm_mode == NormMode.BATCH_NORM:
                    if x.dim() != 4:
                        return False
                    x_float = x.to(dtype=torch.float32)
                    scale_float = scale.to(dtype=torch.float32)
                    if bias is None:
                        return False
                    bias_float = bias.to(dtype=torch.float32)
                    if NormFwdPhase(node.attributes["phase"]) == NormFwdPhase.INFERENCE:
                        mean = logical_view(
                            values.get(node.inputs["mean"].uid),
                            node.inputs["mean"],
                        )
                        inv_variance = logical_view(
                            values.get(node.inputs["variance"].uid),
                            node.inputs["variance"],
                        )
                        if mean is None or inv_variance is None:
                            return False
                        result = (
                            x_float - mean.to(dtype=torch.float32)
                        ) * inv_variance.to(
                            dtype=torch.float32
                        ) * scale_float + bias_float
                    else:
                        momentum = scalar_node_value(node, "momentum", 0.1)
                        running_mean = logical_view(
                            values.get(node.inputs["input_running_mean"].uid),
                            node.inputs["input_running_mean"],
                        )
                        running_var = logical_view(
                            values.get(node.inputs["input_running_variance"].uid),
                            node.inputs["input_running_variance"],
                        )
                        if running_mean is None or running_var is None:
                            return False
                        mean = x_float.mean(dim=axes, keepdim=True)
                        variance = torch.square(x_float - mean).mean(
                            dim=axes, keepdim=True
                        )
                        inv_for_y = torch.rsqrt(variance + epsilon)
                        inv_variance = torch.rsqrt(
                            torch.var(x_float, dim=axes, keepdim=True) + epsilon
                        )
                        result = (x_float - mean) * inv_for_y * scale_float + bias_float
                        if "output_running_mean" in node.outputs:
                            out_mean = node.outputs["output_running_mean"]
                            values[out_mean.uid] = (
                                (1.0 - momentum) * running_mean.to(dtype=torch.float32)
                                + momentum * mean
                            ).to(dtype=_torch_dtype_for_data_type(out_mean.data_type))
                        if "output_running_variance" in node.outputs:
                            out_var = node.outputs["output_running_variance"]
                            unbiased_var = torch.var(x_float, dim=axes, keepdim=True)
                            values[out_var.uid] = (
                                (1.0 - momentum) * running_var.to(dtype=torch.float32)
                                + momentum * unbiased_var
                            ).to(dtype=_torch_dtype_for_data_type(out_var.data_type))
                y_output = node.outputs["y"]
                if (
                    norm_mode == NormMode.RMS_NORM
                    and is_rmsnorm_low_precision_swish_intermediate(y_output)
                ):
                    values[y_output.uid] = result.to(dtype=torch.float32)
                else:
                    values[y_output.uid] = result.to(
                        dtype=_torch_dtype_for_data_type(y_output.data_type)
                    )
                if "saved_mean" in node.outputs:
                    mean_output = node.outputs["saved_mean"]
                    values[mean_output.uid] = mean.to(
                        dtype=_torch_dtype_for_data_type(mean_output.data_type)
                    )
                if "saved_inv_variance" in node.outputs:
                    inv_output = node.outputs["saved_inv_variance"]
                    values[inv_output.uid] = inv_variance.to(
                        dtype=_torch_dtype_for_data_type(inv_output.data_type)
                    )
            elif node.op_type == "norm_bwd":
                norm_mode = NormMode(node.attributes["mode"])
                if norm_mode not in (
                    NormMode.LAYER_NORM,
                    NormMode.RMS_NORM,
                    NormMode.INSTANCE_NORM,
                    NormMode.BATCH_NORM,
                ):
                    return False
                x_tensor = node.inputs["x"]
                dy_tensor = node.inputs["dy"]
                scale_tensor = node.inputs["scale"]
                mean_tensor = node.inputs.get("mean")
                inv_tensor = node.inputs["variance"]
                x = logical_view(values.get(x_tensor.uid), x_tensor)
                dy = logical_view(values.get(dy_tensor.uid), dy_tensor)
                scale = logical_view(values.get(scale_tensor.uid), scale_tensor)
                mean = (
                    None
                    if mean_tensor is None
                    else logical_view(values.get(mean_tensor.uid), mean_tensor)
                )
                inv_variance = logical_view(values.get(inv_tensor.uid), inv_tensor)
                if (
                    x is None
                    or dy is None
                    or scale is None
                    or inv_variance is None
                    or (
                        norm_mode
                        in (
                            NormMode.LAYER_NORM,
                            NormMode.INSTANCE_NORM,
                            NormMode.BATCH_NORM,
                        )
                        and mean is None
                    )
                ):
                    return False
                axes = (
                    (0, 2, 3)
                    if norm_mode == NormMode.BATCH_NORM
                    else norm_reduce_axes(norm_mode, x)
                )
                if not axes:
                    return False
                if norm_mode == NormMode.LAYER_NORM:
                    epsilon = scalar_node_value(node, "epsilon", 1.0e-5)
                    try:
                        x_req = x.detach().clone().requires_grad_(True)
                        scale_req = scale.detach().clone().requires_grad_(True)
                        bias_req = torch.zeros_like(scale_req, requires_grad=True)
                        normalized_shape = list(x.shape[1:])
                        result = torch.nn.functional.layer_norm(
                            x_req,
                            normalized_shape,
                            weight=scale_req.reshape(normalized_shape),
                            bias=bias_req.reshape(normalized_shape),
                            eps=epsilon,
                        )
                        dx, dscale, dbias = torch.autograd.grad(
                            result,
                            (x_req, scale_req, bias_req),
                            dy.detach().to(dtype=result.dtype),
                        )
                    except RuntimeError:
                        return False
                elif norm_mode == NormMode.RMS_NORM:
                    try:
                        x_req = x.detach().clone().requires_grad_(True)
                        scale_req = scale.detach().clone().requires_grad_(True)
                        bias_req = torch.zeros_like(scale_req, requires_grad=True)
                        norm = torch.mean(x_req * x_req, dim=axes, keepdim=True)
                        saved_inv = inv_variance.detach().to(dtype=torch.float32)
                        epsilon = torch.reciprocal(
                            saved_inv * saved_inv
                        ) - norm.detach().to(dtype=torch.float32)
                        inv = torch.rsqrt(norm.to(dtype=torch.float32) + epsilon)
                        result = scale_req * (x_req * inv.to(dtype=x_req.dtype))
                        result = result + bias_req
                        dx, dscale, dbias = torch.autograd.grad(
                            result,
                            (x_req, scale_req, bias_req),
                            dy.detach().to(dtype=result.dtype),
                        )
                    except RuntimeError:
                        return False
                elif norm_mode == NormMode.INSTANCE_NORM:
                    epsilon = scalar_node_value(node, "epsilon", 1.0e-5)
                    try:
                        x_req = x.detach().clone().requires_grad_(True)
                        scale_req = scale.detach().clone().requires_grad_(True)
                        bias_req = torch.zeros_like(scale_req, requires_grad=True)
                        result = torch.nn.functional.instance_norm(
                            x_req,
                            weight=scale_req.reshape(int(x_req.shape[1])),
                            bias=bias_req.reshape(int(x_req.shape[1])),
                            eps=epsilon,
                        )
                        dx, dscale_1d, dbias_1d = torch.autograd.grad(
                            result,
                            (x_req, scale_req, bias_req),
                            dy.detach().to(dtype=result.dtype),
                        )
                        dscale = dscale_1d.reshape(tuple(scale.shape))
                        dbias = dbias_1d.reshape(tuple(scale.shape))
                    except RuntimeError:
                        return False
                elif norm_mode == NormMode.BATCH_NORM:
                    reduced_count = 1
                    for axis in axes:
                        reduced_count *= int(x.shape[axis])
                    x_float = x.to(dtype=torch.float32)
                    dy_float = dy.to(dtype=torch.float32)
                    scale_float = scale.to(dtype=torch.float32)
                    mean_float = mean.to(dtype=torch.float32)
                    inv_float = inv_variance.to(dtype=torch.float32)
                    xhat = (x_float - mean_float) * inv_float
                    dy_scale = dy_float * scale_float
                    dx = inv_float * (
                        dy_scale
                        - dy_scale.sum(dim=axes, keepdim=True) / float(reduced_count)
                        - xhat
                        * (dy_scale * xhat).sum(dim=axes, keepdim=True)
                        / float(reduced_count)
                    )
                    dscale = (dy_float * xhat).sum(dim=axes, keepdim=True)
                    dbias = dy_float.sum(dim=axes, keepdim=True)
                else:
                    return False
                dx_output = node.outputs["dx"]
                dscale_output = node.outputs["dscale"]
                values[dx_output.uid] = dx.to(
                    dtype=_torch_dtype_for_data_type(dx_output.data_type)
                )
                values[dscale_output.uid] = dscale.to(
                    dtype=_torch_dtype_for_data_type(dscale_output.data_type)
                )
                if "dbias" in node.outputs:
                    dbias_output = node.outputs["dbias"]
                    values[dbias_output.uid] = dbias.to(
                        dtype=_torch_dtype_for_data_type(dbias_output.data_type)
                    )
            elif node.op_type == "moe_grouped_matmul":
                token_tensor = node.inputs["token"]
                weight_tensor = node.inputs["weight"]
                offset_tensor = node.inputs["first_token_offset"]
                token = logical_view(values.get(token_tensor.uid), token_tensor)
                weight = logical_view(values.get(weight_tensor.uid), weight_tensor)
                first_token_offset = logical_view(
                    values.get(offset_tensor.uid), offset_tensor
                )
                if token is None or weight is None or first_token_offset is None:
                    return False
                mode = MoeGroupedMatmulMode(node.attributes["mode"])
                if mode != MoeGroupedMatmulMode.NONE:
                    return False
                output = node.outputs["output"]
                result = torch.zeros(
                    tuple(output.dims),
                    dtype=_torch_dtype_for_data_type(output.data_type),
                    device=token.device,
                )
                offsets = first_token_offset.reshape(-1).to(dtype=torch.int64)
                token_count = int(token.shape[1])
                for expert in range(int(weight.shape[0])):
                    start = int(offsets[expert].item())
                    end = (
                        int(offsets[expert + 1].item())
                        if expert + 1 < int(offsets.numel())
                        else token_count
                    )
                    start = max(0, min(start, token_count))
                    end = max(start, min(end, token_count))
                    if start < end:
                        result[0, start:end] = (
                            token[0, start:end].to(dtype=torch.float32)
                            @ weight[expert].to(dtype=torch.float32)
                        ).to(dtype=result.dtype)
                values[output.uid] = result
            elif node.op_type == "moe_grouped_matmul_bwd":
                doutput_tensor = node.inputs["doutput"]
                token_tensor = node.inputs["token"]
                offset_tensor = node.inputs["first_token_offset"]
                doutput = logical_view(values.get(doutput_tensor.uid), doutput_tensor)
                token = logical_view(values.get(token_tensor.uid), token_tensor)
                first_token_offset = logical_view(
                    values.get(offset_tensor.uid), offset_tensor
                )
                if doutput is None or token is None or first_token_offset is None:
                    return False
                dweight = node.outputs["dweight"]
                result = torch.zeros(
                    tuple(dweight.dims),
                    dtype=_torch_dtype_for_data_type(dweight.data_type),
                    device=token.device,
                )
                offsets = first_token_offset.reshape(-1).to(dtype=torch.int64)
                token_count = int(token.shape[1])
                for expert in range(int(result.shape[0])):
                    start = int(offsets[expert].item())
                    end = (
                        int(offsets[expert + 1].item())
                        if expert + 1 < int(offsets.numel())
                        else token_count
                    )
                    start = max(0, min(start, token_count))
                    end = max(start, min(end, token_count))
                    if start < end:
                        result[expert] = (
                            token[0, start:end].to(dtype=torch.float32).transpose(0, 1)
                            @ doutput[0, start:end].to(dtype=torch.float32)
                        ).to(dtype=result.dtype)
                values[dweight.uid] = result
            elif node.op_type == "sdpa":
                q_tensor = node.inputs["q"]
                k_tensor = node.inputs["k"]
                v_tensor = node.inputs["v"]
                output = node.outputs["o"]
                if (
                    q_tensor.ragged_offset is not None
                    and output.ragged_offset is not None
                    and "seq_len_q" in node.inputs
                    and "seq_len_kv" in node.inputs
                    and "bias" not in node.inputs
                    and "mask" not in node.inputs
                ):
                    q_physical = values.get(q_tensor.uid)
                    k_physical = values.get(k_tensor.uid)
                    v_physical = values.get(v_tensor.uid)
                    sink_token = None
                    if "sink_token" in node.inputs:
                        sink_token = logical_view(
                            values.get(node.inputs["sink_token"].uid),
                            node.inputs["sink_token"],
                        )
                    seq_len_q = values.get(node.inputs["seq_len_q"].uid)
                    seq_len_kv = values.get(node.inputs["seq_len_kv"].uid)
                    q_offsets = values.get(q_tensor.ragged_offset.uid)
                    k_offsets = (
                        values.get(k_tensor.ragged_offset.uid)
                        if k_tensor.ragged_offset is not None
                        else None
                    )
                    v_offsets = (
                        values.get(v_tensor.ragged_offset.uid)
                        if v_tensor.ragged_offset is not None
                        else None
                    )
                    o_offsets = values.get(output.ragged_offset.uid)
                    if (
                        q_physical is None
                        or k_physical is None
                        or v_physical is None
                        or seq_len_q is None
                        or seq_len_kv is None
                        or q_offsets is None
                        or o_offsets is None
                        or q_physical.dim() not in (3, 4)
                        or k_physical.dim() not in (3, 4)
                        or v_physical.dim() not in (3, 4)
                        or ("sink_token" in node.inputs and sink_token is None)
                        or (k_physical.dim() == 3 and k_offsets is None)
                        or (v_physical.dim() == 3 and v_offsets is None)
                    ):
                        return False
                    scale = sdpa_scale_value(node, q_tensor.dims[-1] ** -0.5)
                    dropout_probability = float(
                        node.attributes.get("dropout_probability", 0.0)
                    )
                    if dropout_probability != 0.0:
                        if "rng_dump" not in node.outputs:
                            return False
                        rng_dump = node.outputs["rng_dump"]
                        rng_value = values.get(rng_dump.uid)
                        rng_shape = (
                            tuple(rng_value.shape)
                            if hasattr(rng_value, "shape")
                            else tuple(rng_dump.dims)
                        )
                        values[rng_dump.uid] = torch.ones(
                            rng_shape,
                            dtype=_torch_dtype_for_data_type(rng_dump.data_type),
                            device=q_physical.device,
                        )
                    result = torch.full(
                        tuple(q_physical.shape[:-1]) + (int(output.dims[-1]),),
                        float("nan"),
                        dtype=_torch_dtype_for_data_type(output.data_type),
                        device=q_physical.device,
                    )
                    aux_results = {}
                    aux_offset_values = {}
                    aux_token_scales = {}
                    for aux_name in ("stats", "score_max", "score_sum_exp"):
                        if aux_name not in node.outputs:
                            continue
                        aux_output = node.outputs[aux_name]
                        aux_value = values.get(aux_output.uid)
                        aux_shape = (
                            tuple(aux_value.shape)
                            if hasattr(aux_value, "shape")
                            else tuple(q_physical.shape[:-1]) + (1,)
                        )
                        aux_results[aux_name] = torch.full(
                            aux_shape,
                            float("nan"),
                            dtype=_torch_dtype_for_data_type(aux_output.data_type),
                            device=q_physical.device,
                        )
                        aux_offsets = None
                        if aux_output.ragged_offset is not None:
                            aux_offsets = values.get(aux_output.ragged_offset.uid)
                        if aux_offsets is None:
                            aux_offsets = o_offsets
                        aux_offset_values[aux_name] = aux_offsets.reshape(-1).to(
                            dtype=torch.int64
                        )
                        aux_token_scales[aux_name] = int(aux_output.dims[1]) * int(
                            aux_output.dims[3]
                        )
                    seq_q_values = seq_len_q.reshape(-1).to(dtype=torch.int64)
                    seq_kv_values = seq_len_kv.reshape(-1).to(dtype=torch.int64)
                    q_offset_values = q_offsets.reshape(-1).to(dtype=torch.int64)
                    k_offset_values = (
                        k_offsets.reshape(-1).to(dtype=torch.int64)
                        if k_offsets is not None
                        else None
                    )
                    v_offset_values = (
                        v_offsets.reshape(-1).to(dtype=torch.int64)
                        if v_offsets is not None
                        else None
                    )
                    o_offset_values = o_offsets.reshape(-1).to(dtype=torch.int64)
                    q_token_scale = int(q_tensor.dims[1]) * int(q_tensor.dims[3])
                    k_token_scale = int(k_tensor.dims[1]) * int(k_tensor.dims[3])
                    v_token_scale = int(v_tensor.dims[1]) * int(v_tensor.dims[3])
                    o_token_scale = int(output.dims[1]) * int(output.dims[3])
                    for batch in range(int(q_tensor.dims[0])):
                        q_len = int(seq_q_values[batch].item())
                        kv_len = int(seq_kv_values[batch].item())
                        if q_len <= 0 or kv_len <= 0:
                            continue
                        if q_physical.dim() == 3:
                            q_start = (
                                int(q_offset_values[batch].item()) // q_token_scale
                            )
                            o_start = (
                                int(o_offset_values[batch].item()) // o_token_scale
                            )
                            q_batch = q_physical[q_start : q_start + q_len].permute(
                                1, 0, 2
                            )
                        else:
                            o_start = None
                            q_batch = q_physical[batch, :, :q_len, :]
                        if k_physical.dim() == 3:
                            k_start = (
                                int(k_offset_values[batch].item()) // k_token_scale
                            )
                            k_batch = k_physical[k_start : k_start + kv_len].permute(
                                1, 0, 2
                            )
                        else:
                            k_batch = k_physical[batch, :, :kv_len, :]
                        if v_physical.dim() == 3:
                            v_start = (
                                int(v_offset_values[batch].item()) // v_token_scale
                            )
                            v_batch = v_physical[v_start : v_start + kv_len].permute(
                                1, 0, 2
                            )
                        else:
                            v_batch = v_physical[batch, :, :kv_len, :]
                        if k_batch.shape[0] != q_batch.shape[0]:
                            if (
                                q_batch.shape[0] % k_batch.shape[0] != 0
                                or v_batch.shape[0] != k_batch.shape[0]
                            ):
                                return False
                            repeat = q_batch.shape[0] // k_batch.shape[0]
                            k_batch = k_batch.repeat_interleave(repeat, dim=0)
                            v_batch = v_batch.repeat_interleave(repeat, dim=0)
                        scores = (
                            torch.matmul(
                                q_batch.to(dtype=torch.float32),
                                k_batch.to(dtype=torch.float32).transpose(-2, -1),
                            )
                            * scale
                        )
                        diagonal_alignment = node.attributes.get("diagonal_alignment")
                        diagonal_left_bound = node.attributes.get(
                            "diagonal_band_left_bound"
                        )
                        diagonal_right_bound = node.attributes.get(
                            "diagonal_band_right_bound"
                        )
                        left_query_keep = None
                        if diagonal_left_bound is not None:
                            left_query_keep = (
                                torch.arange(
                                    q_len,
                                    dtype=torch.int64,
                                    device=scores.device,
                                ).reshape(1, q_len, 1)
                                < scores.shape[-1] + int(diagonal_left_bound) - 1
                            )
                            scores = scores * left_query_keep.to(dtype=scores.dtype)
                        if node.attributes.get("causal_mask", False) or (
                            diagonal_alignment is not None
                            and (
                                diagonal_left_bound is not None
                                or diagonal_right_bound is not None
                            )
                        ):
                            alignment = (
                                DiagonalAlignment.TOP_LEFT
                                if diagonal_alignment is None
                                else DiagonalAlignment(diagonal_alignment)
                            )
                            masked = _sdpa_diagonal_band_mask(
                                scores.shape,
                                scores.device,
                                alignment,
                                diagonal_left_bound,
                                0
                                if diagonal_right_bound is None
                                else diagonal_right_bound,
                            )
                            scores = torch.where(
                                masked.unsqueeze(0),
                                torch.full_like(scores, -torch.inf),
                                scores,
                            )
                        softmax_scores = scores
                        if sink_token is not None:
                            sink_scores = torch.broadcast_to(
                                sink_token.to(dtype=torch.float32),
                                (1, q_batch.shape[0], q_len, 1),
                            )[0]
                            softmax_scores = torch.cat([sink_scores, scores], dim=-1)
                        score_max = softmax_scores.max(dim=-1, keepdim=True).values
                        score_sum_exp = (
                            torch.exp(softmax_scores - score_max)
                            .nan_to_num()
                            .sum(dim=-1, keepdim=True)
                        )
                        probs = torch.softmax(softmax_scores, dim=-1).nan_to_num()
                        if sink_token is not None:
                            probs = probs[..., 1:]
                        if dropout_probability != 0.0:
                            probs = probs / (1.0 - dropout_probability)
                        if left_query_keep is not None:
                            probs = probs * left_query_keep.to(dtype=probs.dtype)
                        probs = probs.to(
                            dtype=_torch_dtype_for_data_type(output.data_type)
                        ).float()
                        result_batch = torch.matmul(
                            probs, v_batch.to(dtype=torch.float32)
                        ).permute(1, 0, 2)
                        if result.dim() == 3:
                            result[o_start : o_start + q_len] = result_batch.to(
                                dtype=result.dtype
                            )
                        else:
                            result[batch, :, :q_len, :] = result_batch.permute(
                                1, 0, 2
                            ).to(dtype=result.dtype)
                        aux_batches = {
                            "stats": torch.log(score_sum_exp) + score_max,
                            "score_max": score_max,
                            "score_sum_exp": score_sum_exp,
                        }
                        for aux_name, aux_result in aux_results.items():
                            aux_batch = aux_batches[aux_name].permute(1, 0, 2)
                            if aux_result.dim() == 3:
                                aux_start = (
                                    int(aux_offset_values[aux_name][batch].item())
                                    // aux_token_scales[aux_name]
                                )
                                aux_result[aux_start : aux_start + q_len] = (
                                    aux_batch.to(dtype=aux_result.dtype)
                                )
                            else:
                                aux_result[batch, :, :q_len, :] = aux_batch.permute(
                                    1, 0, 2
                                ).to(dtype=aux_result.dtype)
                    values[output.uid] = result
                    for aux_name, aux_result in aux_results.items():
                        values[node.outputs[aux_name].uid] = aux_result
                    continue
                q = logical_view(values.get(q_tensor.uid), q_tensor)
                k = logical_view(values.get(k_tensor.uid), k_tensor)
                v = logical_view(values.get(v_tensor.uid), v_tensor)
                if q is None or k is None or v is None:
                    return False
                if k.shape[1] != q.shape[1]:
                    if q.shape[1] % k.shape[1] != 0 or v.shape[1] != k.shape[1]:
                        return False
                    repeat = q.shape[1] // k.shape[1]
                    k = k.repeat_interleave(repeat, dim=1)
                    v = v.repeat_interleave(repeat, dim=1)
                diagonal_alignment = node.attributes.get("diagonal_alignment")
                diagonal_left_bound = node.attributes.get("diagonal_band_left_bound")
                diagonal_right_bound = node.attributes.get("diagonal_band_right_bound")
                scale = sdpa_scale_value(node, q.shape[-1] ** -0.5)
                scores = (
                    torch.matmul(
                        q.to(dtype=torch.float32),
                        k.to(dtype=torch.float32).transpose(-2, -1),
                    )
                    * scale
                )
                left_query_keep = None
                if diagonal_left_bound is not None:
                    left_query_keep = (
                        torch.arange(
                            scores.shape[-2],
                            dtype=torch.int64,
                            device=scores.device,
                        ).reshape(1, 1, scores.shape[-2], 1)
                        < scores.shape[-1] + int(diagonal_left_bound) - 1
                    )
                    scores = scores * left_query_keep.to(dtype=scores.dtype)
                if "bias" in node.inputs:
                    bias = logical_view(
                        values.get(node.inputs["bias"].uid), node.inputs["bias"]
                    )
                    if bias is None:
                        return False
                    scores = scores + broadcast_attention_tensor(bias, scores.shape)
                if "mask" in node.inputs:
                    mask = logical_view(
                        values.get(node.inputs["mask"].uid), node.inputs["mask"]
                    )
                    if mask is None:
                        return False
                    mask = broadcast_attention_tensor(mask, scores.shape)
                    if mask.dtype == torch.bool:
                        scores = torch.where(
                            mask, scores, torch.full_like(scores, -torch.inf)
                        )
                    else:
                        scores = scores + mask.to(dtype=torch.float32)
                if node.attributes.get("use_alibi_mask", False):
                    scores = scores + alibi_bias(scores)
                valid_query_rows = None
                effective_diagonal_offsets = None
                effective_k_lengths = None
                if "seq_len_q" in node.inputs or "seq_len_kv" in node.inputs:
                    seq_q = scores.shape[-2]
                    seq_k = scores.shape[-1]
                    if "seq_len_q" in node.inputs:
                        seq_len_q = values.get(node.inputs["seq_len_q"].uid)
                        if seq_len_q is None:
                            return False
                        q_lengths = seq_len_q.reshape(-1).to(
                            device=scores.device, dtype=torch.int64
                        )
                    else:
                        q_lengths = torch.full(
                            (scores.shape[0],),
                            seq_q,
                            device=scores.device,
                            dtype=torch.int64,
                        )
                    if "seq_len_kv" in node.inputs:
                        seq_len_kv = values.get(node.inputs["seq_len_kv"].uid)
                        if seq_len_kv is None:
                            return False
                        kv_lengths = seq_len_kv.reshape(-1).to(
                            device=scores.device, dtype=torch.int64
                        )
                    else:
                        kv_lengths = torch.full(
                            (scores.shape[0],),
                            seq_k,
                            device=scores.device,
                            dtype=torch.int64,
                        )
                    if (
                        q_lengths.numel() < scores.shape[0]
                        or kv_lengths.numel() < scores.shape[0]
                    ):
                        return False
                    q_index = torch.arange(
                        seq_q, dtype=torch.int64, device=scores.device
                    ).reshape(1, seq_q)
                    k_index = torch.arange(
                        seq_k, dtype=torch.int64, device=scores.device
                    ).reshape(1, seq_k)
                    valid_q = q_index < q_lengths[: scores.shape[0]].reshape(-1, 1)
                    valid_k = k_index < kv_lengths[: scores.shape[0]].reshape(-1, 1)
                    valid_query_rows = valid_q[:, None, :, None]
                    effective_k_lengths = kv_lengths[: scores.shape[0]].reshape(
                        -1, 1, 1, 1
                    )
                    effective_diagonal_offsets = (
                        kv_lengths[: scores.shape[0]] - q_lengths[: scores.shape[0]]
                    ).reshape(-1, 1, 1, 1)
                    valid_scores = valid_q[:, None, :, None] & valid_k[:, None, None, :]
                    scores = torch.where(
                        valid_scores, scores, torch.full_like(scores, -torch.inf)
                    )
                if node.attributes.get("causal_mask", False):
                    seq_q = scores.shape[-2]
                    seq_k = scores.shape[-1]
                    causal = torch.ones(
                        (seq_q, seq_k), dtype=torch.bool, device=scores.device
                    ).triu(1)
                    scores = torch.where(
                        causal, torch.full_like(scores, -torch.inf), scores
                    )
                if diagonal_alignment is not None and (
                    diagonal_left_bound is not None or diagonal_right_bound is not None
                ):
                    alignment = DiagonalAlignment(diagonal_alignment)
                    diagonal_mask = _sdpa_diagonal_band_mask(
                        scores.shape,
                        scores.device,
                        alignment,
                        diagonal_left_bound,
                        diagonal_right_bound,
                        effective_diagonal_offsets,
                    )
                    scores = torch.where(
                        diagonal_mask, torch.full_like(scores, -torch.inf), scores
                    )
                if "block_mask" in node.inputs:
                    block_mask = logical_view(
                        values.get(node.inputs["block_mask"].uid),
                        node.inputs["block_mask"],
                    )
                    if block_mask is None:
                        return False
                    keep_blocks = unpack_sdpa_block_mask(block_mask, scores.shape)
                    scores = torch.where(
                        keep_blocks, scores, torch.full_like(scores, -torch.inf)
                    )
                if node.attributes.get("score_mod", False):
                    score_input_uid = node.attributes.get("score_mod_input_uid")
                    score_output_uid = node.attributes.get("score_mod_output_uid")
                    if score_input_uid is None or score_output_uid is None:
                        return False
                    values[int(score_input_uid)] = scores
                    for score_node in self.nodes:
                        if score_node.attributes.get("_score_mod_node", False):
                            if not execute_pointwise_node(score_node):
                                return False
                    modified_scores = values.get(int(score_output_uid))
                    if modified_scores is None:
                        return False
                    scores = modified_scores.to(dtype=torch.float32)
                softmax_scores = scores
                if "sink_token" in node.inputs:
                    sink_token = logical_view(
                        values.get(node.inputs["sink_token"].uid),
                        node.inputs["sink_token"],
                    )
                    if sink_token is None:
                        return False
                    sink_scores = broadcast_attention_tensor(
                        sink_token.to(dtype=torch.float32),
                        tuple(scores.shape[:-1]) + (1,),
                    ).expand(tuple(scores.shape[:-1]) + (1,))
                    if valid_query_rows is not None:
                        sink_scores = torch.where(
                            valid_query_rows,
                            sink_scores,
                            torch.full_like(sink_scores, -torch.inf),
                        )
                    softmax_scores = torch.cat([sink_scores, scores], dim=-1)
                row_has_scores = torch.isfinite(softmax_scores).any(
                    dim=-1, keepdim=True
                )
                scores_for_softmax = torch.where(
                    row_has_scores, softmax_scores, torch.zeros_like(softmax_scores)
                )
                score_max = scores_for_softmax.max(dim=-1, keepdim=True).values
                score_sum_exp = torch.exp(scores_for_softmax - score_max).sum(
                    dim=-1, keepdim=True
                )
                score_sum_exp = torch.where(
                    row_has_scores, score_sum_exp, torch.zeros_like(score_sum_exp)
                )
                probs = torch.softmax(scores_for_softmax, dim=-1)
                if "sink_token" in node.inputs:
                    probs = probs[..., 1:]
                dropout_probability = float(
                    node.attributes.get("dropout_probability", 0.0)
                )
                if dropout_probability != 0.0:
                    if "rng_dump" not in node.outputs:
                        return False
                    rng_dump = node.outputs["rng_dump"]
                    dropout_mask = torch.ones_like(probs, dtype=torch.float32)
                    values[rng_dump.uid] = dropout_mask.to(
                        dtype=_torch_dtype_for_data_type(rng_dump.data_type)
                    )
                    probs = probs * dropout_mask / (1.0 - dropout_probability)
                if left_query_keep is not None:
                    probs = probs * left_query_keep.to(dtype=probs.dtype)
                probs = probs * row_has_scores.to(dtype=probs.dtype)
                values[output.uid] = torch.matmul(probs, v.to(dtype=torch.float32)).to(
                    dtype=_torch_dtype_for_data_type(output.data_type)
                )
                if "stats" in node.outputs:
                    stats = node.outputs["stats"]
                    values[stats.uid] = (score_max + torch.log(score_sum_exp)).to(
                        dtype=_torch_dtype_for_data_type(stats.data_type)
                    )
                if "score_max" in node.outputs:
                    out_score_max = node.outputs["score_max"]
                    values[out_score_max.uid] = score_max.to(
                        dtype=_torch_dtype_for_data_type(out_score_max.data_type)
                    )
                if "score_sum_exp" in node.outputs:
                    out_score_sum_exp = node.outputs["score_sum_exp"]
                    values[out_score_sum_exp.uid] = score_sum_exp.to(
                        dtype=_torch_dtype_for_data_type(out_score_sum_exp.data_type)
                    )
            elif node.op_type == "sdpa_bwd":
                q_tensor = node.inputs["q"]
                k_tensor = node.inputs["k"]
                v_tensor = node.inputs["v"]
                d_o_tensor = node.inputs["d_o"]
                if (
                    q_tensor.ragged_offset is not None
                    and k_tensor.ragged_offset is not None
                    and v_tensor.ragged_offset is not None
                    and d_o_tensor.ragged_offset is not None
                    and "seq_len_q" in node.inputs
                    and "seq_len_kv" in node.inputs
                    and "bias" not in node.inputs
                    and "mask" not in node.inputs
                    and "d_bias" not in node.outputs
                    and "sink_token" not in node.inputs
                    and "dSink_token" not in node.outputs
                ):
                    q_physical = values.get(q_tensor.uid)
                    k_physical = values.get(k_tensor.uid)
                    v_physical = values.get(v_tensor.uid)
                    o_physical = values.get(node.inputs["o"].uid)
                    d_o_physical = values.get(d_o_tensor.uid)
                    seq_len_q = values.get(node.inputs["seq_len_q"].uid)
                    seq_len_kv = values.get(node.inputs["seq_len_kv"].uid)
                    q_offsets = values.get(q_tensor.ragged_offset.uid)
                    k_offsets = values.get(k_tensor.ragged_offset.uid)
                    v_offsets = values.get(v_tensor.ragged_offset.uid)
                    o_offsets = values.get(d_o_tensor.ragged_offset.uid)
                    if (
                        q_physical is None
                        or k_physical is None
                        or v_physical is None
                        or o_physical is None
                        or d_o_physical is None
                        or seq_len_q is None
                        or seq_len_kv is None
                        or q_offsets is None
                        or k_offsets is None
                        or v_offsets is None
                        or o_offsets is None
                        or q_physical.dim() not in (3, 4)
                        or k_physical.dim() not in (3, 4)
                        or v_physical.dim() not in (3, 4)
                        or o_physical.dim() not in (3, 4)
                        or d_o_physical.dim() not in (3, 4)
                    ):
                        return False
                    scale = sdpa_scale_value(node, q_tensor.dims[-1] ** -0.5)
                    d_q_result = torch.full_like(q_physical, float("nan"))
                    d_k_result = torch.full_like(k_physical, float("nan"))
                    d_v_result = torch.full_like(v_physical, float("nan"))
                    seq_q_values = seq_len_q.reshape(-1).to(dtype=torch.int64)
                    seq_kv_values = seq_len_kv.reshape(-1).to(dtype=torch.int64)
                    q_offset_values = q_offsets.reshape(-1).to(dtype=torch.int64)
                    k_offset_values = k_offsets.reshape(-1).to(dtype=torch.int64)
                    v_offset_values = v_offsets.reshape(-1).to(dtype=torch.int64)
                    o_offset_values = o_offsets.reshape(-1).to(dtype=torch.int64)
                    q_token_scale = int(q_tensor.dims[1]) * int(q_tensor.dims[3])
                    k_token_scale = int(k_tensor.dims[1]) * int(k_tensor.dims[3])
                    v_token_scale = int(v_tensor.dims[1]) * int(v_tensor.dims[3])
                    o_token_scale = int(d_o_tensor.dims[1]) * int(d_o_tensor.dims[3])
                    diagonal_alignment = node.attributes.get("diagonal_alignment")
                    diagonal_left_bound = node.attributes.get(
                        "diagonal_band_left_bound"
                    )
                    diagonal_right_bound = node.attributes.get(
                        "diagonal_band_right_bound"
                    )
                    try:
                        for batch in range(int(q_tensor.dims[0])):
                            q_len = int(seq_q_values[batch].item())
                            kv_len = int(seq_kv_values[batch].item())
                            q_start = None
                            k_start = None
                            v_start = None
                            if q_physical.dim() == 3:
                                q_start = (
                                    int(q_offset_values[batch].item()) // q_token_scale
                                )
                            if k_physical.dim() == 3:
                                k_start = (
                                    int(k_offset_values[batch].item()) // k_token_scale
                                )
                            if v_physical.dim() == 3:
                                v_start = (
                                    int(v_offset_values[batch].item()) // v_token_scale
                                )
                            if q_len <= 0 or kv_len <= 0:
                                if q_len > 0:
                                    if d_q_result.dim() == 3:
                                        d_q_result[q_start : q_start + q_len] = 0
                                    else:
                                        d_q_result[batch, :, :q_len, :] = 0
                                if kv_len > 0:
                                    if d_k_result.dim() == 3:
                                        d_k_result[k_start : k_start + kv_len] = 0
                                    else:
                                        d_k_result[batch, :, :kv_len, :] = 0
                                    if d_v_result.dim() == 3:
                                        d_v_result[v_start : v_start + kv_len] = 0
                                    else:
                                        d_v_result[batch, :, :kv_len, :] = 0
                                continue
                            if q_physical.dim() == 3:
                                q_batch = q_physical[q_start : q_start + q_len].permute(
                                    1, 0, 2
                                )
                            else:
                                q_start = None
                                q_batch = q_physical[batch, :, :q_len, :]
                            if k_physical.dim() == 3:
                                k_batch = k_physical[
                                    k_start : k_start + kv_len
                                ].permute(1, 0, 2)
                            else:
                                k_start = None
                                k_batch = k_physical[batch, :, :kv_len, :]
                            if v_physical.dim() == 3:
                                v_batch = v_physical[
                                    v_start : v_start + kv_len
                                ].permute(1, 0, 2)
                            else:
                                v_start = None
                                v_batch = v_physical[batch, :, :kv_len, :]
                            if d_o_physical.dim() == 3:
                                o_start = (
                                    int(o_offset_values[batch].item()) // o_token_scale
                                )
                                o_batch = o_physical[o_start : o_start + q_len].permute(
                                    1, 0, 2
                                )
                                d_o_batch = d_o_physical[
                                    o_start : o_start + q_len
                                ].permute(1, 0, 2)
                            else:
                                o_batch = o_physical[batch, :, :q_len, :]
                                d_o_batch = d_o_physical[batch, :, :q_len, :]
                            q_req = q_batch.detach().to(dtype=torch.float32)
                            k_req = k_batch.detach().to(dtype=torch.float32)
                            v_req = v_batch.detach().to(dtype=torch.float32)
                            o_batch = o_batch.detach().to(dtype=torch.float32)
                            d_o_batch = d_o_batch.detach().to(dtype=torch.float32)
                            k_for_scores = k_req
                            v_for_scores = v_req
                            repeat = 1
                            if k_for_scores.shape[0] != q_req.shape[0]:
                                if (
                                    q_req.shape[0] % k_for_scores.shape[0] != 0
                                    or v_for_scores.shape[0] != k_for_scores.shape[0]
                                ):
                                    return False
                                repeat = q_req.shape[0] // k_for_scores.shape[0]
                                k_for_scores = k_for_scores.repeat_interleave(
                                    repeat, dim=0
                                )
                                v_for_scores = v_for_scores.repeat_interleave(
                                    repeat, dim=0
                                )
                            scores = (
                                torch.matmul(
                                    q_req,
                                    k_for_scores.transpose(-2, -1),
                                )
                                * scale
                            )
                            if node.attributes.get("causal_mask", False) or (
                                diagonal_alignment is not None
                                and (
                                    diagonal_left_bound is not None
                                    or diagonal_right_bound is not None
                                )
                            ):
                                alignment = (
                                    DiagonalAlignment.TOP_LEFT
                                    if diagonal_alignment is None
                                    else DiagonalAlignment(diagonal_alignment)
                                )
                                masked = _sdpa_diagonal_band_mask(
                                    scores.shape,
                                    scores.device,
                                    alignment,
                                    diagonal_left_bound,
                                    0
                                    if diagonal_right_bound is None
                                    else diagonal_right_bound,
                                )
                                scores = torch.where(
                                    masked.unsqueeze(0),
                                    torch.full_like(scores, -torch.inf),
                                    scores,
                                )
                            row_has_scores = torch.isfinite(scores).any(
                                dim=-1, keepdim=True
                            )
                            scores_for_softmax = torch.where(
                                row_has_scores, scores, torch.zeros_like(scores)
                            )
                            probs = torch.softmax(
                                scores_for_softmax, dim=-1
                            ).nan_to_num()
                            probs = probs * row_has_scores.to(dtype=probs.dtype)
                            probs_for_dv = probs.to(
                                dtype=_torch_dtype_for_data_type(q_tensor.data_type)
                            ).float()
                            o_ref = torch.zeros(
                                (
                                    q_req.shape[0],
                                    q_len,
                                    v_for_scores.shape[-1],
                                ),
                                dtype=torch.float32,
                                device=q_req.device,
                            )
                            m_old = torch.full(
                                (q_req.shape[0], q_len, 1),
                                -torch.inf,
                                dtype=torch.float32,
                                device=q_req.device,
                            )
                            l_old = torch.zeros_like(m_old)
                            block_size = 128
                            for start in range(0, kv_len, block_size):
                                end = min(start + block_size, kv_len)
                                score_block = scores[..., start:end]
                                m_block = score_block.max(dim=-1, keepdim=True).values
                                m_new = torch.maximum(m_old, m_block)
                                correction = torch.exp(m_old - m_new).nan_to_num()
                                o_ref = o_ref * correction
                                l_old = l_old * correction
                                p_block = (
                                    torch.exp(score_block - m_new)
                                    .nan_to_num()
                                    .to(
                                        dtype=_torch_dtype_for_data_type(
                                            q_tensor.data_type
                                        )
                                    )
                                    .float()
                                )
                                o_ref = o_ref + torch.matmul(
                                    p_block, v_for_scores[:, start:end, :]
                                )
                                l_old = l_old + p_block.sum(dim=-1, keepdim=True)
                                m_old = m_new
                            o_for_d = (
                                (o_ref / l_old.clamp(min=1.0))
                                .to(
                                    dtype=_torch_dtype_for_data_type(q_tensor.data_type)
                                )
                                .float()
                            )
                            d_value = (o_for_d * d_o_batch).sum(dim=-1, keepdim=True)
                            d_p = torch.matmul(
                                d_o_batch, v_for_scores.transpose(-2, -1)
                            )
                            d_s = probs * (d_p - d_value) * scale
                            d_s = d_s.to(
                                dtype=_torch_dtype_for_data_type(q_tensor.data_type)
                            ).float()
                            d_q = torch.matmul(d_s, k_for_scores)
                            d_k_expanded = torch.matmul(d_s.transpose(-2, -1), q_req)
                            d_v_expanded = torch.matmul(
                                probs_for_dv.transpose(-2, -1), d_o_batch
                            )
                            if repeat != 1:
                                d_k = d_k_expanded.reshape(
                                    k_req.shape[0], repeat, kv_len, q_req.shape[-1]
                                ).sum(dim=1)
                                d_v = d_v_expanded.reshape(
                                    v_req.shape[0], repeat, kv_len, v_req.shape[-1]
                                ).sum(dim=1)
                            else:
                                d_k = d_k_expanded
                                d_v = d_v_expanded
                            if d_q_result.dim() == 3:
                                d_q_result[q_start : q_start + q_len] = d_q.permute(
                                    1, 0, 2
                                ).to(dtype=d_q_result.dtype)
                            else:
                                d_q_result[batch, :, :q_len, :] = d_q.to(
                                    dtype=d_q_result.dtype
                                )
                            if d_k_result.dim() == 3:
                                d_k_result[k_start : k_start + kv_len] = d_k.permute(
                                    1, 0, 2
                                ).to(dtype=d_k_result.dtype)
                            else:
                                d_k_result[batch, :, :kv_len, :] = d_k.to(
                                    dtype=d_k_result.dtype
                                )
                            if d_v_result.dim() == 3:
                                d_v_result[v_start : v_start + kv_len] = d_v.permute(
                                    1, 0, 2
                                ).to(dtype=d_v_result.dtype)
                            else:
                                d_v_result[batch, :, :kv_len, :] = d_v.to(
                                    dtype=d_v_result.dtype
                                )
                    except RuntimeError:
                        return False
                    values[node.outputs["d_q"].uid] = d_q_result
                    values[node.outputs["d_k"].uid] = d_k_result
                    values[node.outputs["d_v"].uid] = d_v_result
                    continue
                q = logical_view(values.get(q_tensor.uid), q_tensor)
                k = logical_view(values.get(k_tensor.uid), k_tensor)
                v = logical_view(values.get(v_tensor.uid), v_tensor)
                d_o = logical_view(values.get(d_o_tensor.uid), d_o_tensor)
                if q is None or k is None or v is None or d_o is None:
                    return False
                scale = sdpa_scale_value(node, q.shape[-1] ** -0.5)
                try:
                    q_req = q.detach().to(dtype=torch.float32).requires_grad_(True)
                    k_req = k.detach().to(dtype=torch.float32).requires_grad_(True)
                    v_req = v.detach().to(dtype=torch.float32).requires_grad_(True)
                    bias_req = None
                    k_for_scores = k_req
                    v_for_scores = v_req
                    if k_for_scores.shape[1] != q_req.shape[1]:
                        repeat = q_req.shape[1] // k_for_scores.shape[1]
                        k_for_scores = k_for_scores.repeat_interleave(repeat, dim=1)
                        v_for_scores = v_for_scores.repeat_interleave(repeat, dim=1)
                    scores = torch.matmul(q_req, k_for_scores.transpose(-2, -1)) * scale
                    if "bias" in node.inputs:
                        bias = logical_view(
                            values.get(node.inputs["bias"].uid), node.inputs["bias"]
                        )
                        if bias is None:
                            return False
                        bias_value = bias.detach().to(dtype=torch.float32)
                        if "d_bias" in node.outputs:
                            bias_req = bias_value.requires_grad_(True)
                            bias_value = bias_req
                        scores = scores + broadcast_attention_tensor(
                            bias_value, scores.shape
                        ).to(dtype=torch.float32)
                    if "mask" in node.inputs:
                        mask = logical_view(
                            values.get(node.inputs["mask"].uid), node.inputs["mask"]
                        )
                        if mask is None:
                            return False
                        mask = broadcast_attention_tensor(mask, scores.shape)
                        if mask.dtype == torch.bool:
                            scores = torch.where(
                                mask, scores, torch.full_like(scores, -torch.inf)
                            )
                        else:
                            scores = scores + mask.to(dtype=torch.float32)
                    valid_query_rows = None
                    effective_diagonal_offsets = None
                    if "seq_len_q" in node.inputs or "seq_len_kv" in node.inputs:
                        seq_q = scores.shape[-2]
                        seq_k = scores.shape[-1]
                        if "seq_len_q" in node.inputs:
                            seq_len_q = values.get(node.inputs["seq_len_q"].uid)
                            if seq_len_q is None:
                                return False
                            q_lengths = seq_len_q.reshape(-1).to(
                                device=scores.device, dtype=torch.int64
                            )
                        else:
                            q_lengths = torch.full(
                                (scores.shape[0],),
                                seq_q,
                                device=scores.device,
                                dtype=torch.int64,
                            )
                        if "seq_len_kv" in node.inputs:
                            seq_len_kv = values.get(node.inputs["seq_len_kv"].uid)
                            if seq_len_kv is None:
                                return False
                            kv_lengths = seq_len_kv.reshape(-1).to(
                                device=scores.device, dtype=torch.int64
                            )
                        else:
                            kv_lengths = torch.full(
                                (scores.shape[0],),
                                seq_k,
                                device=scores.device,
                                dtype=torch.int64,
                            )
                        if (
                            q_lengths.numel() < scores.shape[0]
                            or kv_lengths.numel() < scores.shape[0]
                        ):
                            return False
                        q_index = torch.arange(
                            seq_q, dtype=torch.int64, device=scores.device
                        ).reshape(1, seq_q)
                        k_index = torch.arange(
                            seq_k, dtype=torch.int64, device=scores.device
                        ).reshape(1, seq_k)
                        valid_q = q_index < q_lengths[: scores.shape[0]].reshape(-1, 1)
                        valid_k = k_index < kv_lengths[: scores.shape[0]].reshape(-1, 1)
                        valid_query_rows = valid_q[:, None, :, None]
                        effective_diagonal_offsets = (
                            kv_lengths[: scores.shape[0]] - q_lengths[: scores.shape[0]]
                        ).reshape(-1, 1, 1, 1)
                        valid_scores = (
                            valid_q[:, None, :, None] & valid_k[:, None, None, :]
                        )
                        scores = torch.where(
                            valid_scores, scores, torch.full_like(scores, -torch.inf)
                        )
                    if node.attributes.get("causal_mask", False):
                        seq_q = scores.shape[-2]
                        seq_k = scores.shape[-1]
                        causal = torch.ones(
                            (seq_q, seq_k), dtype=torch.bool, device=scores.device
                        ).triu(1)
                        scores = torch.where(
                            causal, torch.full_like(scores, -torch.inf), scores
                        )
                    diagonal_alignment = node.attributes.get("diagonal_alignment")
                    diagonal_left_bound = node.attributes.get(
                        "diagonal_band_left_bound"
                    )
                    diagonal_right_bound = node.attributes.get(
                        "diagonal_band_right_bound"
                    )
                    if diagonal_alignment is not None and (
                        diagonal_left_bound is not None
                        or diagonal_right_bound is not None
                    ):
                        alignment = DiagonalAlignment(diagonal_alignment)
                        diagonal_mask = _sdpa_diagonal_band_mask(
                            scores.shape,
                            scores.device,
                            alignment,
                            diagonal_left_bound,
                            diagonal_right_bound,
                            effective_diagonal_offsets,
                        )
                        scores = torch.where(
                            diagonal_mask, torch.full_like(scores, -torch.inf), scores
                        )
                    if node.attributes.get("score_mod", False):
                        score_input_uid = node.attributes.get("score_mod_input_uid")
                        score_output_uid = node.attributes.get("score_mod_output_uid")
                        if score_input_uid is None or score_output_uid is None:
                            return False
                        values[int(score_input_uid)] = scores
                        for score_node in self.nodes:
                            if score_node.attributes.get("_score_mod_node", False):
                                if not execute_pointwise_node(score_node):
                                    return False
                        modified_scores = values.get(int(score_output_uid))
                        if modified_scores is None:
                            return False
                        scores = modified_scores.to(dtype=torch.float32)
                    softmax_scores = scores
                    sink_req = None
                    if "sink_token" in node.inputs:
                        sink_token = logical_view(
                            values.get(node.inputs["sink_token"].uid),
                            node.inputs["sink_token"],
                        )
                        if sink_token is None:
                            return False
                        sink_req = (
                            sink_token.detach()
                            .to(dtype=torch.float32)
                            .requires_grad_(True)
                        )
                        sink_scores = broadcast_attention_tensor(
                            sink_req, tuple(scores.shape[:-1]) + (1,)
                        ).expand(tuple(scores.shape[:-1]) + (1,))
                        if valid_query_rows is not None:
                            sink_scores = torch.where(
                                valid_query_rows,
                                sink_scores,
                                torch.full_like(sink_scores, -torch.inf),
                            )
                        softmax_scores = torch.cat([sink_scores, scores], dim=-1)
                    row_has_scores = torch.isfinite(softmax_scores).any(
                        dim=-1, keepdim=True
                    )
                    scores_for_softmax = torch.where(
                        row_has_scores,
                        softmax_scores,
                        torch.zeros_like(softmax_scores),
                    )
                    probs = torch.softmax(scores_for_softmax, dim=-1)
                    if sink_req is not None:
                        probs = probs[..., 1:]
                    probs = probs * row_has_scores.to(dtype=probs.dtype)
                    result = torch.matmul(probs, v_for_scores)
                    grad_inputs = [q_req, k_req, v_req]
                    if bias_req is not None:
                        grad_inputs.append(bias_req)
                    if sink_req is not None:
                        grad_inputs.append(sink_req)
                    gradients = torch.autograd.grad(
                        result,
                        tuple(grad_inputs),
                        d_o.detach().to(dtype=result.dtype),
                        allow_unused=True,
                    )
                except RuntimeError:
                    return False
                d_q, d_k, d_v = gradients[:3]
                grad_index = 3
                d_bias = None
                if bias_req is not None:
                    d_bias = gradients[grad_index]
                    grad_index += 1
                d_sink = gradients[grad_index] if len(gradients) > grad_index else None
                if d_q is None or d_k is None or d_v is None:
                    return False
                d_q_output = node.outputs["d_q"]
                d_k_output = node.outputs["d_k"]
                d_v_output = node.outputs["d_v"]
                values[d_q_output.uid] = d_q.to(
                    dtype=_torch_dtype_for_data_type(d_q_output.data_type)
                )
                values[d_k_output.uid] = d_k.to(
                    dtype=_torch_dtype_for_data_type(d_k_output.data_type)
                )
                values[d_v_output.uid] = d_v.to(
                    dtype=_torch_dtype_for_data_type(d_v_output.data_type)
                )
                if "d_bias" in node.outputs:
                    if d_bias is None:
                        return False
                    d_bias_output = node.outputs["d_bias"]
                    values[d_bias_output.uid] = d_bias.to(
                        dtype=_torch_dtype_for_data_type(d_bias_output.data_type)
                    )
                if "dSink_token" in node.outputs:
                    if d_sink is None:
                        return False
                    d_sink_output = node.outputs["dSink_token"]
                    values[d_sink_output.uid] = d_sink.to(
                        dtype=_torch_dtype_for_data_type(d_sink_output.data_type)
                    )
            else:
                return False

        for tensor in self.tensors:
            if (
                tensor.is_virtual
                or tensor.uid not in values
                or tensor.uid not in provided
            ):
                continue
            source = values[tensor.uid]
            target = provided[tensor.uid]
            if source is not target:
                if tuple(source.shape) != tuple(target.shape):
                    logical_target = logical_view(target, tensor)
                    if (
                        logical_target is not None
                        and _strided_storage_elements(tensor.dims, tensor.strides)
                        >= source.numel()
                        and tuple(source.shape) == tuple(logical_target.shape)
                    ):
                        target = logical_target
                    elif source.numel() == target.numel():
                        source = source.reshape(tuple(target.shape))
                    elif source.numel() <= target.numel():
                        flat_target = storage_flat_view(target)
                        if flat_target is None or flat_target.numel() < source.numel():
                            return False
                        flat_target[: source.numel()].copy_(
                            source.reshape(-1).to(dtype=target.dtype)
                        )
                        continue
                    else:
                        return False
                target.copy_(source.to(dtype=target.dtype))
        return True

    def _execute(
        self,
        var_pack: Mapping[object, int],
        workspace=None,
        handle=None,
        override_uids=None,
        override_shapes=None,
        override_strides=None,
    ):
        return self.execute(
            var_pack,
            workspace=workspace,
            handle=handle,
            override_uids=override_uids,
            override_shapes=override_shapes,
            override_strides=override_strides,
        )

    def execute_plan_at_index(
        self,
        data_ptrs: Mapping[object, int] | None = None,
        workspace=None,
        index: int = 0,
        *,
        variant_pack: Mapping[object, int] | None = None,
        workspace_ptr=None,
        handle=None,
        override_uids=None,
        override_shapes=None,
        override_strides=None,
    ):
        self._validate_dynamic_shape_overrides(
            override_uids, override_shapes, override_strides
        )
        if data_ptrs is None:
            data_ptrs = variant_pack
        elif variant_pack is not None:
            raise ValueError("pass either data_ptrs or variant_pack, not both")
        if data_ptrs is None:
            raise ValueError("execute_plan_at_index requires data pointers")
        if workspace_ptr is not None and workspace is not None:
            raise ValueError("pass either workspace or workspace_ptr, not both")
        if workspace_ptr is None:
            workspace_ptr = workspace
        if workspace_ptr is not None:
            workspace_ptr = _pointer_value(workspace_ptr)
        if any(node.attributes.get("score_mod", False) for node in self.nodes):
            raise NotImplementedError("SDPA score_mod execution is not supported")
        self._set_runtime_external_uids(self._data_ptr_uids(data_ptrs))
        if not self._plans_built:
            self.build_plan_at_index(index, handle)
        uid_to_pointer = self._normalize_data_ptrs(data_ptrs)
        self._native.execute(uid_to_pointer, workspace_ptr=workspace_ptr, index=index)
        return self

    def _execute_plan_at_index(
        self,
        var_pack: Mapping[object, int],
        workspace=None,
        index: int = 0,
        handle=None,
        override_uids=None,
        override_shapes=None,
        override_strides=None,
    ):
        return self.execute_plan_at_index(
            var_pack,
            workspace=workspace,
            index=index,
            handle=handle,
            override_uids=override_uids,
            override_shapes=override_shapes,
            override_strides=override_strides,
        )

    def _execute_with_ptrs(self, user_ptrs, workspace=None, handle=None):
        uids = self._get_variant_pack_uids_sorted()
        ptrs = [_pointer_value(pointer) for pointer in user_ptrs]
        if len(ptrs) != len(uids):
            raise ValueError("user_ptrs length must match variant-pack UID count")
        return self.execute(
            dict(zip(uids, ptrs)),
            workspace=workspace,
            handle=handle,
        )

    def _execute_with_raw_ptrs(
        self, user_ptrs_array, n_user: int, workspace=None, handle=None
    ):
        count = int(n_user)
        if count < 0:
            raise ValueError("n_user must be non-negative")
        address = _pointer_value(user_ptrs_array)
        if count > 0 and address == 0:
            raise ValueError("user_ptrs_array is null")
        array_type = ctypes.c_void_p * count
        raw = array_type.from_address(address)
        return self._execute_with_ptrs(
            [raw[index] for index in range(count)],
            workspace=workspace,
            handle=handle,
        )

    def populate_cuda_graph(self, handle, var_pack, workspace, cuda_graph):
        raise NotImplementedError("CUDA graph population is not supported")

    def update_cuda_graph(self, handle, var_pack, workspace, cuda_graph):
        raise NotImplementedError("CUDA graph update is not supported")

    def __repr__(self):
        try:
            payload = json.loads(self.serialize().decode("utf-8"))
        except Exception:
            return (
                f"pygraph(num_tensors={len(self.tensors)}, num_nodes={len(self.nodes)})"
            )
        return json.dumps(payload, indent=4, sort_keys=True)

    def _normalize_data_ptrs(self, data_ptrs: Mapping[object, int]) -> dict[int, int]:
        by_name = {tensor.name: tensor for tensor in self.tensors}
        by_uid = {tensor.uid: tensor for tensor in self.tensors}
        normalized = {}
        for uid, value in self._by_value_data.items():
            normalized[uid] = ctypes.addressof(value)
        for key, pointer in data_ptrs.items():
            if key is None or pointer is None:
                continue
            if isinstance(key, GraphTensor):
                uid = key.uid
            elif isinstance(key, str):
                uid = by_name[key].uid
            else:
                tensor = by_uid.get(int(key))
                if tensor is None:
                    continue
                uid = tensor.uid
            normalized[uid] = _pointer_value(pointer)
        external_uids = self._external_tensor_uids()
        missing = sorted(external_uids.difference(normalized))
        if missing:
            raise ValueError(f"missing data pointers for tensor uids: {missing}")
        return {uid: normalized[uid] for uid in sorted(external_uids)}

    def _data_ptr_uids(self, data_ptrs: Mapping[object, int]) -> set[int]:
        by_name = {tensor.name: tensor for tensor in self.tensors}
        by_uid = {tensor.uid: tensor for tensor in self.tensors}
        uids = set()
        for key, value in data_ptrs.items():
            if key is None or value is None:
                continue
            if isinstance(key, GraphTensor):
                uids.add(key.uid)
            elif isinstance(key, str):
                uids.add(by_name[key].uid)
            else:
                tensor = by_uid.get(int(key))
                if tensor is not None:
                    uids.add(tensor.uid)
        return uids

    def _set_runtime_external_uids(self, uids: set[int] | None):
        runtime_uids = None if uids is None else set(uids)
        if runtime_uids == self._runtime_external_uids:
            return
        self._runtime_external_uids = runtime_uids
        self._invalidate_lowering()

    def _produced_tensor_info(self) -> dict[int, dict[str, object]]:
        produced = {}
        for node in self.nodes:
            for index, (name, tensor) in enumerate(node.outputs.items()):
                produced[tensor.uid] = {
                    "tensor": tensor,
                    "name": name,
                    "is_primary": index == 0,
                }
        return produced

    def _consumed_tensor_uids(self) -> set[int]:
        return {tensor.uid for node in self.nodes for tensor in node.inputs.values()}

    def _graph_output_uids(self) -> set[int]:
        produced_info = self._produced_tensor_info()
        consumed_uids = self._consumed_tensor_uids()
        output_uids = {
            tensor.uid
            for tensor in self.tensors
            if tensor.is_output and tensor.uid in produced_info
        }
        if self._output_tuples is not None:
            output_uids.update(tensor.uid for tensor in self._output_tuples)
        else:
            # Workspace size is queried before execute receives pointers.
            # Only a node's primary unconsumed output is implicit; auxiliary
            # outputs stay virtual unless the caller names them. Using a
            # broader set here omits those tensors from the workspace, and
            # the execute rebuild then writes past the allocation.
            output_uids.update(
                uid
                for uid, info in produced_info.items()
                if info["is_primary"] and uid not in consumed_uids
            )
        if self._runtime_external_uids is not None:
            output_uids.update(
                uid for uid in self._runtime_external_uids if uid in produced_info
            )
        return output_uids

    def _internal_tensor_uids(self) -> set[int]:
        produced_info = self._produced_tensor_info()
        output_uids = self._graph_output_uids()
        internal_uids = {
            uid
            for uid, info in produced_info.items()
            if uid not in output_uids and not info["tensor"].is_output
        }
        for node in self.nodes:
            if node.attributes.get("_score_mod_node", False) or node.attributes.get(
                "_score_mod_bprop_node", False
            ):
                internal_uids.update(tensor.uid for tensor in node.outputs.values())
            if node.op_type in ("sdpa", "sdpa_bwd"):
                for attr_name in (
                    "score_mod_input_uid",
                    "score_mod_output_uid",
                    "score_mod_bprop_input_uid",
                    "score_mod_bprop_output_uid",
                ):
                    uid = node.attributes.get(attr_name)
                    if uid is not None:
                        internal_uids.add(int(uid))
        return internal_uids

    def _external_tensor_uids(self) -> set[int]:
        internal_uids = self._internal_tensor_uids()
        return {
            tensor.uid for tensor in self.tensors if tensor.uid not in internal_uids
        }


def pygraph(*args, **kwargs) -> Graph:
    return Graph(*args, **kwargs)


class _NativeGraph:
    def __init__(self, handle, operation_graph, descriptors):
        self.handle = handle
        self.operation_graph = operation_graph
        self.descriptors = descriptors
        self.engine_configs = []
        self.plans = []

    def ensure_engine_configs(self):
        if self.engine_configs:
            return
        from . import _backend as backend

        heur = backend.Descriptor(backend.MUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR)
        heur.set_descriptor(
            backend.MUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, self.operation_graph
        )
        heur.set_enum(
            backend.MUDNN_ATTR_ENGINEHEUR_MODE,
            backend.MUDNN_TYPE_HEUR_MODE,
            HeurMode.INSTANT,
        )
        heur.finalize()
        self.descriptors.append(heur)
        count = heur.get_descriptor_count(backend.MUDNN_ATTR_ENGINEHEUR_RESULTS)
        self.engine_configs = (
            heur.get_descriptors(
                backend.MUDNN_ATTR_ENGINEHEUR_RESULTS, requested=count
            )
            if count
            else []
        )

    def validate_engine_index(self, index: int):
        self.ensure_engine_configs()
        if index < 0 or index >= len(self.engine_configs):
            raise IndexError(f"execution plan index {index} out of range")

    def build_plans(self, modes: Iterable[HeurMode]):
        from . import _backend as backend

        self.ensure_engine_configs()
        for _mode in modes:
            self.build_plan_at_index(0)

    def build_plan_at_index(self, index: int):
        from . import _backend as backend

        self.ensure_engine_configs()
        self.validate_engine_index(index)
        while len(self.plans) <= index:
            self.plans.append(None)
        if self.plans[index] is not None:
            return
        plan = backend.Descriptor(backend.MUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)
        plan.set_handle(backend.MUDNN_ATTR_EXECUTION_PLAN_HANDLE, self.handle)
        plan.set_descriptor(
            backend.MUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, self.engine_configs[index]
        )
        plan.finalize()
        self.plans[index] = plan
        self.descriptors.append(plan)

    def execute(
        self, uid_to_pointer: dict[int, int], *, workspace_ptr=None, index: int = 0
    ):
        from . import _backend as backend

        if index < 0:
            raise IndexError(f"execution plan index {index} out of range")
        if len(self.plans) <= index or self.plans[index] is None:
            self.build_plan_at_index(index)
        variant_pack = backend.create_variant_pack(
            uid_to_pointer, workspace_ptr=workspace_ptr
        )
        self.descriptors.append(variant_pack)
        backend.execute(self.handle, self.plans[index], variant_pack)


class _CpuFallbackNativeGraph:
    class _Plan:
        def __init__(self, workspace_size: int = 0):
            self.workspace_size = int(workspace_size)

        def get_int64(self, attr):
            return [self.workspace_size]

    def __init__(self, handle, graph: Graph | None = None):
        self.handle = handle
        self.graph = graph
        self.operation_graph = None
        self.descriptors = []
        self.engine_configs = [object()]
        self.plans = []

    def ensure_engine_configs(self):
        return

    def validate_engine_index(self, index: int):
        if index != 0:
            raise IndexError(f"execution plan index {index} out of range")

    def build_plans(self, modes: Iterable[HeurMode]):
        self.build_plan_at_index(0)

    def build_plan_at_index(self, index: int):
        self.validate_engine_index(index)
        while len(self.plans) <= index:
            self.plans.append(None)
        if self.plans[index] is None:
            self.plans[index] = self._Plan(self._workspace_size())

    def _workspace_size(self) -> int:
        if self.graph is None:
            return 0
        max_size = 0
        for node in self.graph.nodes:
            if node.op_type != "pointwise":
                continue
            if PointwiseMode(node.attributes["mode"]) != PointwiseMode.SWISH_FWD:
                continue
            output = node.outputs.get("y")
            if output is None or output.data_type != DataType.FP4_E2M1:
                continue
            if len(output.dims) < 2 or int(output.dims[1]) <= 0:
                continue
            tokens = int(output.dims[0])
            channels = int(output.dims[1])
            if channels % 16 != 0:
                continue
            rs_size = tokens * 4
            rs_size_aligned = ((rs_size + 127) // 128) * 128
            fp8_scale_aligned = ((rs_size_aligned + 4 + 127) // 128) * 128
            max_size = max(max_size, fp8_scale_aligned + tokens * (channels // 16))
        return max_size

    def execute(
        self, uid_to_pointer: dict[int, int], *, workspace_ptr=None, index: int = 0
    ):
        raise NotImplementedError(
            "this graph is executable only through the torch CPU fallback"
        )


class _NativeGraphLowerer:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.descriptors = []
        self.tensor_descs = {}

    @staticmethod
    def backend_attr_workspace_size() -> int:
        from . import _backend as backend

        return backend.MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE

    def lower(self, handle):
        from . import _backend as backend

        native_handle = (
            handle if isinstance(handle, backend.Handle) else backend.Handle()
        )

        def sdpa_requires_torch_cpu_fallback(node: GraphNode) -> bool:
            if node.op_type == "sdpa_bwd":
                try:
                    import torch

                    if "Fake CPU CUDA" in str(torch.cuda.get_device_name()):
                        return True
                except Exception:
                    pass
                return bool(
                    node.attributes.get("score_mod", False)
                    or node.attributes.get("score_mod_bprop", False)
                    or "sink_token" in node.inputs
                    or "dSink_token" in node.outputs
                )
            if node.op_type != "sdpa":
                return False
            try:
                import torch

                if "Fake CPU CUDA" in str(torch.cuda.get_device_name()):
                    return True
            except Exception:
                pass
            q = node.inputs.get("q")
            o = node.outputs.get("o")
            if (
                q is not None
                and o is not None
                and (q.ragged_offset is not None or o.ragged_offset is not None)
            ):
                return True
            if node.attributes.get("score_mod", False):
                return True
            if "block_mask" in node.inputs:
                return True
            if "sink_token" in node.inputs:
                return True
            if (
                "paged_attention_k_table" in node.inputs
                or "paged_attention_v_table" in node.inputs
                or node.attributes.get("paged_attention_max_seq_len_kv") is not None
            ):
                return True
            if "score_max" in node.outputs or "score_sum_exp" in node.outputs:
                return True
            if float(node.attributes.get("dropout_probability", 0.0)) != 0.0:
                return True
            alignment = node.attributes.get("diagonal_alignment")
            left_bound = node.attributes.get("diagonal_band_left_bound")
            right_bound = node.attributes.get("diagonal_band_right_bound")
            if alignment is None and left_bound is None and right_bound is None:
                return False
            if alignment is None:
                alignment = DiagonalAlignment.TOP_LEFT
            alignment = DiagonalAlignment(alignment)
            return not (
                alignment == DiagonalAlignment.TOP_LEFT
                and left_bound is None
                and (right_bound is None or int(right_bound) == 0)
            )

        def conv_requires_torch_cpu_fallback(node: GraphNode) -> bool:
            if node.op_type != "conv_fprop":
                return False
            x = node.inputs.get("x")
            w = node.inputs.get("w")
            output = node.outputs.get("y")
            if x is None or w is None or output is None:
                return False
            return any(tensor.data_type == DataType.INT8 for tensor in (x, w, output))

        def matmul_graph_requires_torch_cpu_fallback() -> bool:
            if not any(node.op_type == "matmul" for node in self.graph.nodes):
                return False
            if any(tensor.data_type == DataType.INT8 for tensor in self.graph.tensors):
                return True
            for node in self.graph.nodes:
                if node.op_type != "pointwise":
                    continue
                if PointwiseMode(node.attributes["mode"]) != PointwiseMode.IDENTITY:
                    continue
                x = node.inputs.get("x")
                y = node.outputs.get("y")
                if x is not None and y is not None and x.data_type != y.data_type:
                    return True
            return False

        def norm_graph_requires_torch_cpu_fallback() -> bool:
            fake_cuda_cpu_remap = False
            try:
                import torch

                fake_cuda_cpu_remap = "Fake CPU CUDA" in str(
                    torch.cuda.get_device_name()
                )
            except Exception:
                pass
            for node in self.graph.nodes:
                if node.op_type not in ("norm_fwd", "norm_bwd"):
                    continue
                if fake_cuda_cpu_remap:
                    return True
                if NormMode(node.attributes["mode"]) != NormMode.BATCH_NORM:
                    continue
                tensors = tuple(node.inputs.values()) + tuple(node.outputs.values())
                if any(
                    getattr(tensor.device, "type", None) == "cpu" for tensor in tensors
                ):
                    return True
            return False

        def block_scale_graph_requires_torch_cpu_fallback() -> bool:
            for node in self.graph.nodes:
                if node.op_type == "block_scale_quantize":
                    x = node.inputs["x"]
                    y = node.outputs["y"]
                    scale = node.outputs["scale"]
                    low_precision = any(
                        tensor.data_type
                        in _UNSUPPORTED_LOW_PRECISION_EXECUTION_DATA_TYPES
                        for tensor in (x, y, scale)
                    )
                    if scale.data_type != DataType.FLOAT:
                        return True
                    if (
                        node.attributes.get("block_axis", len(x.dims) - 1)
                        != len(x.dims) - 1
                        and low_precision
                    ):
                        return True
                    if low_precision:
                        return True
                elif node.op_type == "block_scale_dequantize":
                    x = node.inputs["x"]
                    scale = node.inputs["scale"]
                    y = node.outputs["y"]
                    low_precision = any(
                        tensor.data_type
                        in _UNSUPPORTED_LOW_PRECISION_EXECUTION_DATA_TYPES
                        for tensor in (x, scale, y)
                    )
                    if scale.data_type != DataType.FLOAT:
                        return True
                    if (
                        node.attributes.get("block_axis", len(x.dims) - 1)
                        != len(x.dims) - 1
                        and low_precision
                    ):
                        return True
                    if low_precision:
                        return True
            return False

        if (
            any(
                sdpa_requires_torch_cpu_fallback(node)
                or conv_requires_torch_cpu_fallback(node)
                or node.op_type in ("moe_grouped_matmul", "moe_grouped_matmul_bwd")
                for node in self.graph.nodes
            )
            or matmul_graph_requires_torch_cpu_fallback()
            or norm_graph_requires_torch_cpu_fallback()
            or block_scale_graph_requires_torch_cpu_fallback()
        ):
            return _CpuFallbackNativeGraph(native_handle, self.graph)
        self._preflight_native_block_scale_support()
        internal_uids = self.graph._internal_tensor_uids()
        uid_to_tensor = {int(tensor.uid): tensor for tensor in self.graph.tensors}
        resolving_tensor_uids = set()

        def ensure_tensor_desc(tensor: GraphTensor):
            uid = int(tensor.uid)
            if uid in self.tensor_descs:
                return self.tensor_descs[uid]
            if uid in resolving_tensor_uids:
                raise ValueError("ragged offset tensor dependency cycle")
            resolving_tensor_uids.add(uid)
            ragged_offset_desc = None
            if tensor.ragged_offset is not None:
                ragged_uid = int(tensor.ragged_offset.uid)
                ragged_tensor = uid_to_tensor.get(ragged_uid)
                if ragged_tensor is None:
                    raise ValueError(
                        f"ragged offset tensor uid {ragged_uid} is not part of the graph"
                    )
                ragged_offset_desc = ensure_tensor_desc(ragged_tensor)
            desc = backend.create_tensor_desc(
                tensor,
                is_virtual=uid in internal_uids,
                ragged_offset_desc=ragged_offset_desc,
            )
            self.tensor_descs[uid] = desc
            self.descriptors.append(desc)
            resolving_tensor_uids.remove(uid)
            return desc

        for tensor in self.graph.tensors:
            ensure_tensor_desc(tensor)

        op_descs = []
        for node in self.graph.nodes:
            if node.op_type == "pointwise":
                op_descs.append(self._lower_pointwise(node, backend))
            elif node.op_type == "conv_fprop":
                op_descs.append(self._lower_conv_fprop(node, backend))
            elif node.op_type == "conv_bwd_data":
                op_descs.append(self._lower_conv_bwd_data(node, backend))
            elif node.op_type == "conv_bwd_filter":
                op_descs.append(self._lower_conv_bwd_filter(node, backend))
            elif node.op_type == "matmul":
                op_descs.append(self._lower_matmul(node, backend))
            elif node.op_type == "reduction":
                op_descs.append(self._lower_reduction(node, backend))
            elif node.op_type == "genstats":
                op_descs.append(self._lower_genstats(node, backend))
            elif node.op_type == "bn_finalize":
                op_descs.append(self._lower_bn_finalize(node, backend))
            elif node.op_type == "bn_bwd_weights":
                op_descs.append(self._lower_bn_bwd_weights(node, backend))
            elif node.op_type == "resample_fwd":
                op_descs.append(self._lower_resample_fwd(node, backend))
            elif node.op_type == "resample_bwd":
                op_descs.append(self._lower_resample_bwd(node, backend))
            elif node.op_type == "concat":
                op_descs.append(self._lower_concat(node, backend))
            elif node.op_type == "reshape":
                op_descs.append(self._lower_reshape(node, backend))
            elif node.op_type == "transpose":
                op_descs.append(self._lower_transpose(node, backend))
            elif node.op_type == "slice":
                op_descs.append(self._lower_slice(node, backend))
            elif node.op_type == "rng":
                op_descs.append(self._lower_rng(node, backend))
            elif node.op_type == "norm_fwd":
                op_descs.append(self._lower_norm_fwd(node, backend))
            elif node.op_type == "norm_bwd":
                op_descs.append(self._lower_norm_bwd(node, backend))
            elif node.op_type == "block_scale_quantize":
                op_descs.append(self._lower_block_scale_quantize(node, backend))
            elif node.op_type == "block_scale_dequantize":
                op_descs.append(self._lower_block_scale_dequantize(node, backend))
            elif node.op_type == "paged_cache_load":
                op_descs.append(self._lower_paged_cache_load(node, backend))
            elif node.op_type == "signal":
                op_descs.append(self._lower_signal(node, backend))
            elif node.op_type == "moe_grouped_matmul":
                op_descs.append(self._lower_moe_grouped_matmul(node, backend))
            elif node.op_type == "moe_grouped_matmul_bwd":
                op_descs.append(self._lower_moe_grouped_matmul_bwd(node, backend))
            elif node.op_type == "sdpa":
                op_descs.append(self._lower_sdpa(node, backend))
            elif node.op_type == "sdpa_bwd":
                op_descs.append(self._lower_sdpa_bwd(node, backend))
            else:
                raise NotImplementedError(f"unsupported native op: {node.op_type}")

        operation_graph = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
        )
        operation_graph.set_handle(
            backend.MUDNN_ATTR_OPERATIONGRAPH_HANDLE, native_handle
        )
        operation_graph.set_descriptors(backend.MUDNN_ATTR_OPERATIONGRAPH_OPS, op_descs)
        operation_graph.finalize()
        self.descriptors.append(operation_graph)
        return _NativeGraph(native_handle, operation_graph, self.descriptors)

    def _preflight_native_block_scale_support(self):
        for node in self.graph.nodes:
            if node.op_type == "block_scale_quantize":
                x = node.inputs["x"]
                scale = node.outputs["scale"]
                if (
                    node.attributes.get("block_axis", len(x.dims) - 1)
                    != len(x.dims) - 1
                ):
                    raise NotImplementedError(
                        "native block_scale_quantize supports only last-dimension block scales"
                    )
                if scale.data_type != DataType.FLOAT:
                    raise NotImplementedError(
                        "native block_scale_quantize supports only FLOAT scale tensors"
                    )
            elif node.op_type == "block_scale_dequantize":
                x = node.inputs["x"]
                scale = node.inputs["scale"]
                if (
                    node.attributes.get("block_axis", len(x.dims) - 1)
                    != len(x.dims) - 1
                ):
                    raise NotImplementedError(
                        "native block_scale_dequantize supports only last-dimension block scales"
                    )
                if scale.data_type != DataType.FLOAT:
                    raise NotImplementedError(
                        "native block_scale_dequantize supports only FLOAT scale tensors"
                    )

    def _tensor_desc(self, tensor: GraphTensor):
        return self.tensor_descs[tensor.uid]

    def _compute_type(self, node: GraphNode):
        # Resolution mirrors the historical behavior when the graph compute
        # data type defaulted to FLOAT: a node-level attribute wins, then the
        # graph context, then FLOAT. NOT_SET means "not specified" at each
        # step, matching the cuDNN 0.0.1 defaults.
        value = node.attributes.get("compute_data_type")
        if value is None or DataType(value) == DataType.NOT_SET:
            value = self.graph.compute_data_type
        if value is None or DataType(value) == DataType.NOT_SET:
            return DataType.FLOAT
        return DataType(value)

    def _math_precision(self, node: GraphNode):
        value = node.attributes.get("math_precision")
        if value is None or DataType(value) == DataType.NOT_SET:
            return self._compute_type(node)
        return DataType(value)

    def _lower_pointwise(self, node: GraphNode, backend):
        pw = backend.Descriptor(backend.MUDNN_BACKEND_POINTWISE_DESCRIPTOR)
        pw.set_enum(
            backend.MUDNN_ATTR_POINTWISE_MODE,
            backend.MUDNN_TYPE_POINTWISE_MODE,
            node.attributes["mode"],
        )
        pw.set_enum(
            backend.MUDNN_ATTR_POINTWISE_MATH_PREC,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._compute_type(node),
        )
        if "swish_beta" in node.attributes:
            pw.set_double(
                backend.MUDNN_ATTR_POINTWISE_SWISH_BETA,
                node.attributes["swish_beta"],
            )
        if "relu_lower_clip" in node.attributes:
            pw.set_double(
                backend.MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                node.attributes["relu_lower_clip"],
            )
        if "relu_upper_clip" in node.attributes:
            pw.set_double(
                backend.MUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                node.attributes["relu_upper_clip"],
            )
        if "relu_lower_clip_slope" in node.attributes:
            pw.set_double(
                backend.MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                node.attributes["relu_lower_clip_slope"],
            )
        if "elu_alpha" in node.attributes:
            pw.set_double(
                backend.MUDNN_ATTR_POINTWISE_ELU_ALPHA,
                node.attributes["elu_alpha"],
            )
        if "softplus_beta" in node.attributes:
            pw.set_double(
                backend.MUDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                node.attributes["softplus_beta"],
            )
        if "axis" in node.attributes:
            pw.set_int64(backend.MUDNN_ATTR_POINTWISE_AXIS, [node.attributes["axis"]])
        pw.finalize()
        self.descriptors.append(pw)

        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
        op.set_descriptor(backend.MUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pw)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_POINTWISE_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        if "b" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_POINTWISE_BDESC,
                self._tensor_desc(node.inputs["b"]),
            )
        if "t" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_POINTWISE_TDESC,
                self._tensor_desc(node.inputs["t"]),
            )
        if "dy" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_POINTWISE_DYDESC,
                self._tensor_desc(node.inputs["dy"]),
            )
        if "y" in node.outputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_POINTWISE_YDESC,
                self._tensor_desc(node.outputs["y"]),
            )
        if "dx" in node.outputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_POINTWISE_DXDESC,
                self._tensor_desc(node.outputs["dx"]),
            )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_POINTWISE_ALPHA1,
            node.attributes.get("alpha1", 1.0),
        )
        if "b" in node.inputs:
            op.set_double(
                backend.MUDNN_ATTR_OPERATION_POINTWISE_ALPHA2,
                node.attributes.get("alpha2", 1.0),
            )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_matmul(self, node: GraphNode, backend):
        matmul = backend.Descriptor(backend.MUDNN_BACKEND_MATMUL_DESCRIPTOR)
        matmul.set_enum(
            backend.MUDNN_ATTR_MATMUL_COMP_TYPE,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._compute_type(node),
        )
        matmul.finalize()
        self.descriptors.append(matmul)

        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
        op.set_descriptor(backend.MUDNN_ATTR_OPERATION_MATMUL_DESC, matmul)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_MATMUL_ADESC,
            self._tensor_desc(node.inputs["a"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_MATMUL_BDESC,
            self._tensor_desc(node.inputs["b"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_MATMUL_CDESC,
            self._tensor_desc(node.outputs["c"]),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    @staticmethod
    def _current_conv_group_count(node: GraphNode) -> int:
        if node.op_type == "conv_fprop":
            x = node.inputs["x"]
            w = node.inputs["w"]
            if w.dims[1] > 0 and x.dims[1] % w.dims[1] == 0:
                return x.dims[1] // w.dims[1]
        elif node.op_type == "conv_bwd_data":
            dx = node.outputs["dx"]
            w = node.inputs["w"]
            if w.dims[1] > 0 and dx.dims[1] % w.dims[1] == 0:
                return dx.dims[1] // w.dims[1]
        elif node.op_type == "conv_bwd_filter":
            x = node.inputs["x"]
            dw = node.outputs["dw"]
            if dw.dims[1] > 0 and x.dims[1] % dw.dims[1] == 0:
                return x.dims[1] // dw.dims[1]
        return int(node.attributes.get("group_count", 1))

    def _lower_conv_desc(self, node: GraphNode, backend):
        conv = backend.Descriptor(backend.MUDNN_BACKEND_CONVOLUTION_DESCRIPTOR)
        conv.set_enum(
            backend.MUDNN_ATTR_CONVOLUTION_COMP_TYPE,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._compute_type(node),
        )
        conv.set_enum(
            backend.MUDNN_ATTR_CONVOLUTION_CONV_MODE,
            backend.MUDNN_TYPE_CONVOLUTION_MODE,
            node.attributes["mode"],
        )
        conv.set_int64(
            backend.MUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
            [len(node.attributes["padding"])],
        )
        conv.set_int64(
            backend.MUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
            node.attributes.get("pre_padding", node.attributes["padding"]),
        )
        conv.set_int64(
            backend.MUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
            node.attributes.get("post_padding", node.attributes["padding"]),
        )
        conv.set_int64(
            backend.MUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
            node.attributes["stride"],
        )
        conv.set_int64(
            backend.MUDNN_ATTR_CONVOLUTION_DILATIONS,
            node.attributes["dilation"],
        )
        conv.set_int64(
            backend.MUDNN_ATTR_CONVOLUTION_GROUP_COUNT,
            [self._current_conv_group_count(node)],
        )
        conv.finalize()
        self.descriptors.append(conv)
        return conv

    def _lower_conv_fprop(self, node: GraphNode, backend):
        conv = self._lower_conv_desc(node, backend)
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, conv
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
            self._tensor_desc(node.inputs["w"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
            self._tensor_desc(node.outputs["y"]),
        )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
            node.attributes.get("alpha", 1.0),
        )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
            node.attributes.get("beta", 0.0),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_conv_bwd_data(self, node: GraphNode, backend):
        conv = self._lower_conv_desc(node, backend)
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, conv
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
            self._tensor_desc(node.inputs["w"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
            self._tensor_desc(node.outputs["dx"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
            self._tensor_desc(node.inputs["dy"]),
        )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
            node.attributes.get("alpha", 1.0),
        )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
            node.attributes.get("beta", 0.0),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_conv_bwd_filter(self, node: GraphNode, backend):
        conv = self._lower_conv_desc(node, backend)
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, conv
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
            self._tensor_desc(node.outputs["dw"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
            self._tensor_desc(node.inputs["dy"]),
        )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
            node.attributes.get("alpha", 1.0),
        )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
            node.attributes.get("beta", 0.0),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_reduction(self, node: GraphNode, backend):
        if "group_offset" in node.inputs:
            raise NotImplementedError(
                "grouped reduction native lowering is not supported"
            )
        reduction = backend.Descriptor(backend.MUDNN_BACKEND_REDUCTION_DESCRIPTOR)
        reduction.set_enum(
            backend.MUDNN_ATTR_REDUCTION_OPERATOR,
            backend.MUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
            node.attributes["mode"],
        )
        reduction.set_enum(
            backend.MUDNN_ATTR_REDUCTION_COMP_TYPE,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._compute_type(node),
        )
        reduction.finalize()
        self.descriptors.append(reduction)

        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
        op.set_descriptor(backend.MUDNN_ATTR_OPERATION_REDUCTION_DESC, reduction)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_REDUCTION_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_REDUCTION_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_genstats(self, node: GraphNode, backend):
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR)
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_GENSTATS_MODE,
            backend.MUDNN_TYPE_GENSTATS_MODE,
            node.attributes["mode"],
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._math_precision(node),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_GENSTATS_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_GENSTATS_SUMDESC,
            self._tensor_desc(node.outputs["sum"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC,
            self._tensor_desc(node.outputs["sqsum"]),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_bn_finalize(self, node: GraphNode, backend):
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE,
            backend.MUDNN_TYPE_BN_FINALIZE_STATS_MODE,
            node.attributes["mode"],
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._math_precision(node),
        )
        for key, attr in (
            ("scale", backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC),
            ("bias", backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC),
            (
                "prev_running_mean",
                backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC,
            ),
            (
                "prev_running_var",
                backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC,
            ),
        ):
            op.set_descriptor(attr, self._tensor_desc(node.inputs[key]))
        if node.attributes["mode"] == BnFinalizeStatsMode.TRAINING:
            for key, attr in (
                ("y_sum", backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC),
                (
                    "y_sq_sum",
                    backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC,
                ),
                (
                    "accum_count",
                    backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC,
                ),
            ):
                op.set_descriptor(attr, self._tensor_desc(node.inputs[key]))
            for key, attr in (
                (
                    "updated_running_mean",
                    backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC,
                ),
                (
                    "updated_running_var",
                    backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC,
                ),
                (
                    "saved_mean",
                    backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC,
                ),
                (
                    "saved_inv_std",
                    backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC,
                ),
            ):
                op.set_descriptor(attr, self._tensor_desc(node.outputs[key]))
            op.set_double(
                backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC,
                node.attributes["momentum"],
            )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC,
            self._tensor_desc(node.outputs["eq_scale"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC,
            self._tensor_desc(node.outputs["eq_bias"]),
        )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC,
            node.attributes["epsilon"],
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_bn_bwd_weights(self, node: GraphNode, backend):
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._math_precision(node),
        )
        for key, attr in (
            ("mean", backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC),
            ("invstd", backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC),
            ("scale", backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC),
            ("x", backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC),
            ("dy", backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC),
        ):
            op.set_descriptor(attr, self._tensor_desc(node.inputs[key]))
        for key, attr in (
            (
                "dbn_scale",
                backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC,
            ),
            ("dbn_bias", backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC),
            (
                "eq_dy_scale",
                backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC,
            ),
            (
                "eq_x_scale",
                backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC,
            ),
            ("eq_bias", backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS),
        ):
            op.set_descriptor(attr, self._tensor_desc(node.outputs[key]))
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_resample_desc(self, node: GraphNode, backend):
        resample = backend.Descriptor(backend.MUDNN_BACKEND_RESAMPLE_DESCRIPTOR)
        resample.set_enum(
            backend.MUDNN_ATTR_RESAMPLE_MODE,
            backend.MUDNN_TYPE_RESAMPLE_MODE,
            node.attributes["mode"],
        )
        resample.set_enum(
            backend.MUDNN_ATTR_RESAMPLE_COMP_TYPE,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._compute_type(node),
        )
        resample.set_int64(backend.MUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, [2])
        resample.set_int64(
            backend.MUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
            node.attributes["window"],
        )
        resample.set_int64(
            backend.MUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
            node.attributes["padding"],
        )
        resample.set_int64(
            backend.MUDNN_ATTR_RESAMPLE_POST_PADDINGS,
            node.attributes["padding"],
        )
        resample.set_int64(
            backend.MUDNN_ATTR_RESAMPLE_STRIDES,
            node.attributes["stride"],
        )
        resample.finalize()
        self.descriptors.append(resample)
        return resample

    def _lower_resample_fwd(self, node: GraphNode, backend):
        resample = self._lower_resample_desc(node, backend)
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
        op.set_descriptor(backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC, resample)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        if "idx" in node.outputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                self._tensor_desc(node.outputs["idx"]),
            )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA,
            node.attributes.get("alpha", 1.0),
        )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA,
            node.attributes.get("beta", 0.0),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_resample_bwd(self, node: GraphNode, backend):
        resample = self._lower_resample_desc(node, backend)
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
        op.set_descriptor(backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC, resample)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC,
            self._tensor_desc(node.outputs["dx"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC,
            self._tensor_desc(node.inputs["dy"]),
        )
        if "idx" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC,
                self._tensor_desc(node.inputs["idx"]),
            )
        if "x" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC,
                self._tensor_desc(node.inputs["x"]),
            )
        if "y" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC,
                self._tensor_desc(node.inputs["y"]),
            )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA,
            node.attributes.get("alpha", 1.0),
        )
        op.set_double(
            backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA,
            node.attributes.get("beta", 0.0),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_concat(self, node: GraphNode, backend):
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR)
        op.set_int64(
            backend.MUDNN_ATTR_OPERATION_CONCAT_AXIS, [node.attributes["axis"]]
        )
        op.set_descriptors(
            backend.MUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS,
            [self._tensor_desc(tensor) for tensor in node.inputs.values()],
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC,
            self._tensor_desc(node.outputs["y"]),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_reshape(self, node: GraphNode, backend):
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_RESHAPE_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_RESHAPE_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        reshape_mode = node.attributes.get("reshape_mode", ReshapeMode.VIEW_ONLY)
        if reshape_mode != ReshapeMode.VIEW_ONLY:
            op.set_enum(
                backend.MUDNN_ATTR_OPERATION_RESHAPE_MODE,
                backend.MUDNN_TYPE_RESHAPE_MODE,
                reshape_mode,
            )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_transpose(self, node: GraphNode, backend):
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_TRANSPOSE_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_TRANSPOSE_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        op.set_int64(
            backend.MUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION,
            node.attributes["permutation"],
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_slice(self, node: GraphNode, backend):
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SLICE_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SLICE_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        op.set_int64(
            backend.MUDNN_ATTR_OPERATION_SLICE_START_INDICES,
            node.attributes["start_indices"],
        )
        op.set_int64(
            backend.MUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES,
            node.attributes["limit_indices"],
        )
        op.set_int64(
            backend.MUDNN_ATTR_OPERATION_SLICE_STRIDES,
            node.attributes["strides"],
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_rng(self, node: GraphNode, backend):
        rng = backend.Descriptor(backend.MUDNN_BACKEND_RNG_DESCRIPTOR)
        rng.set_enum(
            backend.MUDNN_ATTR_RNG_DISTRIBUTION,
            backend.MUDNN_TYPE_RNG_DISTRIBUTION,
            node.attributes["distribution"],
        )
        rng.set_double(
            backend.MUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM,
            node.attributes["minimum"],
        )
        rng.set_double(
            backend.MUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM,
            node.attributes["maximum"],
        )
        rng.set_double(backend.MUDNN_ATTR_RNG_NORMAL_DIST_MEAN, node.attributes["mean"])
        rng.set_double(
            backend.MUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION,
            node.attributes["stddev"],
        )
        rng.set_double(
            backend.MUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY,
            node.attributes["probability"],
        )
        rng.finalize()
        self.descriptors.append(rng)

        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_RNG_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        op.set_int64(backend.MUDNN_ATTR_OPERATION_RNG_SEED, [node.attributes["seed"]])
        op.set_descriptor(backend.MUDNN_ATTR_OPERATION_RNG_DESC, rng)
        if "offset" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_RNG_OFFSET_DESC,
                self._tensor_desc(node.inputs["offset"]),
            )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_norm_fwd(self, node: GraphNode, backend):
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
        phase = NormFwdPhase(node.attributes["phase"])
        if (
            node.attributes["mode"] == NormMode.RMS_NORM
            and "saved_inv_variance" in node.outputs
        ):
            phase = NormFwdPhase.TRAINING
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_NORM_FWD_MODE,
            backend.MUDNN_TYPE_NORM_MODE,
            node.attributes["mode"],
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
            backend.MUDNN_TYPE_NORM_FWD_PHASE,
            phase,
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        mean_tensor = node.outputs.get("saved_mean")
        if mean_tensor is None and "mean" in node.inputs:
            mean_tensor = node.inputs["mean"]
        if mean_tensor is not None:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                self._tensor_desc(mean_tensor),
            )
        inv_variance_tensor = node.outputs.get("saved_inv_variance")
        if inv_variance_tensor is None and "variance" in node.inputs:
            inv_variance_tensor = node.inputs["variance"]
        if inv_variance_tensor is not None:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
                self._tensor_desc(inv_variance_tensor),
            )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
            self._tensor_desc(node.inputs["scale"]),
        )
        if "bias" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                self._tensor_desc(node.inputs["bias"]),
            )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        if (
            phase == NormFwdPhase.TRAINING
            and node.attributes["mode"] == NormMode.BATCH_NORM
        ):
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
                self._tensor_desc(node.inputs["input_running_mean"]),
            )
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
                self._tensor_desc(node.inputs["input_running_variance"]),
            )
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
                self._tensor_desc(node.outputs["output_running_mean"]),
            )
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
                self._tensor_desc(node.outputs["output_running_variance"]),
            )
            if "momentum" in node.inputs:
                op.set_descriptor(
                    backend.MUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
                    self._tensor_desc(node.inputs["momentum"]),
                )
            else:
                op.set_double(
                    backend.MUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
                    node.attributes.get("momentum", 0.1),
                )
        if "epsilon" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                self._tensor_desc(node.inputs["epsilon"]),
            )
        else:
            op.set_double(
                backend.MUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                node.attributes.get("epsilon", 1.0e-5),
            )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_norm_bwd(self, node: GraphNode, backend):
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_NORM_BWD_MODE,
            backend.MUDNN_TYPE_NORM_MODE,
            node.attributes["mode"],
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_NORM_BWD_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_NORM_BWD_DYDESC,
            self._tensor_desc(node.inputs["dy"]),
        )
        if "mean" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC,
                self._tensor_desc(node.inputs["mean"]),
            )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC,
            self._tensor_desc(node.inputs["variance"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC,
            self._tensor_desc(node.inputs["scale"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_NORM_BWD_DXDESC,
            self._tensor_desc(node.outputs["dx"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC,
            self._tensor_desc(node.outputs["dscale"]),
        )
        if "dbias" in node.outputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC,
                self._tensor_desc(node.outputs["dbias"]),
            )
        if "epsilon" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC,
                self._tensor_desc(node.inputs["epsilon"]),
            )
        else:
            op.set_double(
                backend.MUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC,
                node.attributes.get("epsilon", 1.0e-5),
            )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_block_scale_quantize(self, node: GraphNode, backend):
        x = node.inputs["x"]
        scale = node.outputs["scale"]
        if node.attributes.get("block_axis", len(x.dims) - 1) != len(x.dims) - 1:
            raise NotImplementedError(
                "native block_scale_quantize supports only last-dimension block scales"
            )
        if scale.data_type != DataType.FLOAT:
            raise NotImplementedError(
                "native block_scale_quantize supports only FLOAT scale tensors"
            )
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC,
            self._tensor_desc(node.outputs["scale"]),
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._math_precision(node),
        )
        op.set_int64(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
            [node.attributes["block_size"]],
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_block_scale_dequantize(self, node: GraphNode, backend):
        if node.attributes.get("is_negative_scale", False):
            raise NotImplementedError("negative block scales are not supported")
        x = node.inputs["x"]
        scale = node.inputs["scale"]
        if node.attributes.get("block_axis", len(x.dims) - 1) != len(x.dims) - 1:
            raise NotImplementedError(
                "native block_scale_dequantize supports only last-dimension block scales"
            )
        if scale.data_type != DataType.FLOAT:
            raise NotImplementedError(
                "native block_scale_dequantize supports only FLOAT scale tensors"
            )
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
            self._tensor_desc(node.inputs["scale"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
            backend.MUDNN_TYPE_DATA_TYPE,
            self._math_precision(node),
        )
        op.set_int64(
            backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
            [node.attributes["block_size"]],
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_paged_cache_load(self, node: GraphNode, backend):
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC,
            self._tensor_desc(node.inputs["container"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC,
            self._tensor_desc(node.inputs["sequence"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC,
            self._tensor_desc(node.inputs["page_table"]),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_signal(self, node: GraphNode, backend):
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR)
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_SIGNAL_MODE,
            backend.MUDNN_TYPE_SIGNAL_MODE,
            node.attributes["mode"],
        )
        op.set_int64(
            backend.MUDNN_ATTR_OPERATION_SIGNAL_VALUE,
            [node.attributes["value"]],
        )
        flag = node.outputs.get("flag", node.inputs.get("flag"))
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC,
            self._tensor_desc(flag),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SIGNAL_XDESC,
            self._tensor_desc(node.inputs["x"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SIGNAL_YDESC,
            self._tensor_desc(node.outputs["y"]),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_moe_grouped_matmul(self, node: GraphNode, backend):
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
            backend.MUDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
            node.attributes["mode"],
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
            backend.MUDNN_TYPE_DATA_TYPE,
            node.attributes["compute_data_type"],
        )
        for key, attr in (
            ("token", backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC),
            ("weight", backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC),
            (
                "first_token_offset",
                backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
            ),
        ):
            op.set_descriptor(attr, self._tensor_desc(node.inputs[key]))
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
            self._tensor_desc(node.outputs["output"]),
        )
        if "token_index" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                self._tensor_desc(node.inputs["token_index"]),
            )
        if "token_ks" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                self._tensor_desc(node.inputs["token_ks"]),
            )
        if int(node.attributes.get("top_k", 0)) > 0:
            op.set_int64(
                backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K,
                [node.attributes["top_k"]],
            )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_moe_grouped_matmul_bwd(self, node: GraphNode, backend):
        op = backend.Descriptor(
            backend.MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR
        )
        op.set_enum(
            backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
            backend.MUDNN_TYPE_DATA_TYPE,
            node.attributes["compute_data_type"],
        )
        for key, attr in (
            ("token", backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC),
            (
                "first_token_offset",
                backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
            ),
            (
                "doutput",
                backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
            ),
        ):
            op.set_descriptor(attr, self._tensor_desc(node.inputs[key]))
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
            self._tensor_desc(node.outputs["dweight"]),
        )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_sdpa(self, node: GraphNode, backend):
        fp8_inputs = [
            name
            for name in (
                "descale_q",
                "descale_k",
                "descale_v",
                "descale_s",
                "scale_s",
                "scale_o",
            )
            if name in node.inputs
        ]
        if fp8_inputs:
            api_name = "sdpa_fp8" if node.attributes.get("fp8") else "sdpa_mxfp8"
            raise NotImplementedError(
                f"{api_name} native lowering is not supported; unsupported "
                f"scale/descale tensors: {', '.join(fp8_inputs)}"
            )
        q = node.inputs["q"]
        k = node.inputs["k"]
        o = node.outputs["o"]
        hidden_uid = 1_000_000 + len(self.descriptors) * 2
        attn_probs = GraphTensor(
            uid=hidden_uid,
            name=f"{o.name}_attn_probs",
            dims=(q.dims[0], q.dims[1], q.dims[2], k.dims[2]),
            strides=(
                q.dims[1] * q.dims[2] * k.dims[2],
                q.dims[2] * k.dims[2],
                k.dims[2],
                1,
            ),
            data_type=q.data_type,
            is_virtual=True,
        )
        dropout_mask = GraphTensor(
            uid=hidden_uid + 1,
            name=f"{o.name}_dropout_mask",
            dims=attn_probs.dims,
            strides=attn_probs.strides,
            data_type=DataType.BOOLEAN,
            is_virtual=True,
        )
        attn_probs_desc = backend.create_tensor_desc(attn_probs, is_virtual=True)
        dropout_mask_desc = backend.create_tensor_desc(dropout_mask, is_virtual=True)
        self.descriptors.extend([attn_probs_desc, dropout_mask_desc])

        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
            self._tensor_desc(node.inputs["q"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
            self._tensor_desc(node.inputs["k"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
            self._tensor_desc(node.inputs["v"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
            self._tensor_desc(node.outputs["o"]),
        )
        if "stats" in node.outputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC,
                self._tensor_desc(node.outputs["stats"]),
            )
        if "scale" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
                self._tensor_desc(node.inputs["scale"]),
            )
        elif "sdpa_scale" in node.attributes:
            op.set_double(
                backend.MUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
                node.attributes["sdpa_scale"],
            )
        if "mask" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC,
                self._tensor_desc(node.inputs["mask"]),
            )
        if "bias" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_SDPA_FWD_BIAS_DESC,
                self._tensor_desc(node.inputs["bias"]),
            )
        if "seq_len_q" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
                self._tensor_desc(node.inputs["seq_len_q"]),
            )
        if "seq_len_kv" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC,
                self._tensor_desc(node.inputs["seq_len_kv"]),
            )
        if node.attributes.get("use_padding_mask", False):
            op.set_bool(
                backend.MUDNN_ATTR_OPERATION_SDPA_FWD_USE_PADDING_MASK,
                True,
            )
        if node.attributes.get("causal_mask", False):
            op.set_bool(
                backend.MUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK,
                True,
            )
        op.finalize()
        self.descriptors.append(op)
        return op

    def _lower_sdpa_bwd(self, node: GraphNode, backend):
        if node.attributes.get("fp8_bwd") or node.attributes.get("mxfp8_bwd"):
            fp8_inputs = [
                name
                for name in (
                    "descale_q",
                    "descale_q_T",
                    "descale_k",
                    "descale_k_T",
                    "descale_v",
                    "descale_o",
                    "descale_dO",
                    "descale_dO_T",
                    "descale_s",
                    "descale_dP",
                    "scale_s",
                    "scale_dQ",
                    "scale_dK",
                    "scale_dV",
                    "scale_dP",
                )
                if name in node.inputs
            ]
            api_name = (
                "sdpa_fp8_backward"
                if node.attributes.get("fp8_bwd")
                else "sdpa_mxfp8_backward"
            )
            detail = (
                f"; unsupported scale/descale tensors: {', '.join(fp8_inputs)}"
                if fp8_inputs
                else ""
            )
            raise NotImplementedError(
                f"{api_name} native lowering is not supported{detail}"
            )
        if "bias" in node.inputs or "mask" in node.inputs:
            raise NotImplementedError(
                "sdpa_backward bias/mask lowering is not supported by the "
                "cuDNN 9.23 graph SDPA descriptor"
            )
        if "seq_len_q" in node.inputs or "seq_len_kv" in node.inputs:
            raise NotImplementedError(
                "sdpa_backward sequence-length native lowering is not supported"
            )
        op = backend.Descriptor(backend.MUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_BWD_QDESC,
            self._tensor_desc(node.inputs["q"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_BWD_KDESC,
            self._tensor_desc(node.inputs["k"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_BWD_VDESC,
            self._tensor_desc(node.inputs["v"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_BWD_ODESC,
            self._tensor_desc(node.inputs["o"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC,
            self._tensor_desc(node.inputs["d_o"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC,
            self._tensor_desc(node.inputs["stats"]),
        )
        if "scale" in node.inputs:
            op.set_descriptor(
                backend.MUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC,
                self._tensor_desc(node.inputs["scale"]),
            )
        elif "sdpa_scale" in node.attributes:
            op.set_double(
                backend.MUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC,
                node.attributes["sdpa_scale"],
            )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC,
            self._tensor_desc(node.outputs["d_q"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC,
            self._tensor_desc(node.outputs["d_k"]),
        )
        op.set_descriptor(
            backend.MUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC,
            self._tensor_desc(node.outputs["d_v"]),
        )
        if node.attributes.get("causal_mask", False):
            op.set_bool(
                backend.MUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK,
                True,
            )
        op.finalize()
        self.descriptors.append(op)
        return op


@dataclass(frozen=True)
class Operation:
    op_type: str
    inputs: Mapping[str, Tensor]
    outputs: Mapping[str, Tensor]
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PointWiseDesc:
    mode: PointwiseMode = PointwiseMode.ADD
    math_precision: DataType = DataType.FLOAT
    relu_lower_clip: float | None = None
    relu_upper_clip: float | None = None
    relu_lower_clip_slope: float | None = None
    elu_alpha: float | None = None
    softplus_beta: float | None = None
    swish_beta: float | None = None
    axis: int | None = None


class PointWiseDescBuilder:
    def __init__(self):
        self._mode = PointwiseMode.ADD
        self._math_precision = DataType.FLOAT
        self._relu_lower_clip: float | None = None
        self._relu_upper_clip: float | None = None
        self._relu_lower_clip_slope: float | None = None
        self._elu_alpha: float | None = None
        self._softplus_beta: float | None = None
        self._swish_beta: float | None = None
        self._axis: int | None = None

    def setMode(self, mode: PointwiseMode):
        self._mode = PointwiseMode(mode)
        return self

    def setMathPrecision(self, precision: DataType):
        self._math_precision = DataType(precision)
        return self

    def setReluLowerClip(self, value: float):
        self._relu_lower_clip = float(value)
        return self

    def setReluUpperClip(self, value: float):
        self._relu_upper_clip = float(value)
        return self

    def setReluLowerClipSlope(self, value: float):
        self._relu_lower_clip_slope = float(value)
        return self

    def setEluAlpha(self, value: float):
        self._elu_alpha = float(value)
        return self

    def setSoftplusBeta(self, value: float):
        self._softplus_beta = float(value)
        return self

    def setSwishBeta(self, value: float):
        self._swish_beta = float(value)
        return self

    def setAxis(self, axis: int):
        self._axis = int(axis)
        return self

    def build(self) -> PointWiseDesc:
        return PointWiseDesc(
            mode=self._mode,
            math_precision=self._math_precision,
            relu_lower_clip=self._relu_lower_clip,
            relu_upper_clip=self._relu_upper_clip,
            relu_lower_clip_slope=self._relu_lower_clip_slope,
            elu_alpha=self._elu_alpha,
            softplus_beta=self._softplus_beta,
            swish_beta=self._swish_beta,
            axis=self._axis,
        )


PointwiseDesc = PointWiseDesc
PointwiseDescBuilder = PointWiseDescBuilder


@dataclass(frozen=True)
class MatMulDesc:
    math_precision: DataType = DataType.FLOAT
    padding_value: float | None = None


class MatMulDescBuilder:
    def __init__(self):
        self._math_precision = DataType.FLOAT
        self._padding_value: float | None = None

    def setMathPrecision(self, precision: DataType):
        self._math_precision = DataType(precision)
        return self

    def setPaddingValue(self, padding_value: float):
        self._padding_value = float(padding_value)
        return self

    def build(self) -> MatMulDesc:
        return MatMulDesc(
            math_precision=self._math_precision,
            padding_value=self._padding_value,
        )


@dataclass(frozen=True)
class ConvDesc:
    math_precision: DataType = DataType.FLOAT
    mode: ConvolutionMode = ConvolutionMode.CROSS_CORRELATION
    spatial_dims: int = 2
    pre_paddings: tuple[int, ...] = (0, 0)
    post_paddings: tuple[int, ...] = (0, 0)
    filter_strides: tuple[int, ...] = (1, 1)
    dilations: tuple[int, ...] = (1, 1)
    group_count: int = 1


class ConvDescBuilder:
    def __init__(self):
        self._math_precision = DataType.FLOAT
        self._mode = ConvolutionMode.CROSS_CORRELATION
        self._spatial_dims = 2
        self._pre_paddings = (0, 0)
        self._post_paddings = (0, 0)
        self._filter_strides = (1, 1)
        self._dilations = (1, 1)
        self._group_count = 1

    def setMathPrecision(self, precision: DataType):
        self._math_precision = DataType(precision)
        return self

    def setComputeType(self, precision: DataType):
        return self.setMathPrecision(precision)

    def setConvolutionMode(self, mode: ConvolutionMode):
        self._mode = ConvolutionMode(mode)
        return self

    def setMathMode(self, mode: ConvolutionMode):
        return self.setConvolutionMode(mode)

    def setSpatialDimCount(self, count: int):
        self._spatial_dims = int(count)
        return self

    def setPrePadding(self, *args):
        self._pre_paddings = tuple(
            int(v) for v in _sequence_from_counted_args(args, "pre padding")
        )
        return self

    def setPostPadding(self, *args):
        self._post_paddings = tuple(
            int(v) for v in _sequence_from_counted_args(args, "post padding")
        )
        return self

    def setStride(self, *args):
        self._filter_strides = tuple(
            int(v) for v in _sequence_from_counted_args(args, "stride")
        )
        return self

    def setStrides(self, *args):
        return self.setStride(*args)

    def setSpatialStride(self, *args):
        return self.setStride(*args)

    def setDilation(self, *args):
        self._dilations = tuple(
            int(v) for v in _sequence_from_counted_args(args, "dilation")
        )
        return self

    def setGroupCount(self, group_count: int):
        self._group_count = int(group_count)
        return self

    def build(self) -> ConvDesc:
        return ConvDesc(
            math_precision=self._math_precision,
            mode=self._mode,
            spatial_dims=self._spatial_dims,
            pre_paddings=self._pre_paddings,
            post_paddings=self._post_paddings,
            filter_strides=self._filter_strides,
            dilations=self._dilations,
            group_count=self._group_count,
        )


ConvolutionDesc = ConvDesc
ConvolutionDescBuilder = ConvDescBuilder


@dataclass(frozen=True)
class ResampleDesc:
    mode: ResampleMode = ResampleMode.AVGPOOL_INCLUDE_PADDING
    math_precision: DataType = DataType.FLOAT
    spatial_dims: int = 2
    window_dims: tuple[int, ...] = (2, 2)
    pre_paddings: tuple[int, ...] = (0, 0)
    post_paddings: tuple[int, ...] = (0, 0)
    strides: tuple[int, ...] = (2, 2)


class ResampleDescBuilder:
    def __init__(self):
        self._mode = ResampleMode.AVGPOOL_INCLUDE_PADDING
        self._math_precision = DataType.FLOAT
        self._spatial_dims = 2
        self._window_dims = (2, 2)
        self._pre_paddings = (0, 0)
        self._post_paddings = (0, 0)
        self._strides = (2, 2)

    def setMode(self, mode: ResampleMode):
        self._mode = ResampleMode(mode)
        return self

    def setMathPrecision(self, precision: DataType):
        self._math_precision = DataType(precision)
        return self

    def setSpatialDimCount(self, count: int):
        self._spatial_dims = int(count)
        return self

    def setWindowDim(self, values: Sequence[int]):
        self._window_dims = tuple(int(v) for v in values)
        return self

    def setPrePadding(self, values: Sequence[int]):
        self._pre_paddings = tuple(int(v) for v in values)
        return self

    def setPostPadding(self, values: Sequence[int]):
        self._post_paddings = tuple(int(v) for v in values)
        return self

    def setStride(self, values: Sequence[int]):
        self._strides = tuple(int(v) for v in values)
        return self

    def build(self) -> ResampleDesc:
        return ResampleDesc(
            mode=self._mode,
            math_precision=self._math_precision,
            spatial_dims=self._spatial_dims,
            window_dims=self._window_dims,
            pre_paddings=self._pre_paddings,
            post_paddings=self._post_paddings,
            strides=self._strides,
        )


PoolingDesc = ResampleDesc
PoolingDescBuilder = ResampleDescBuilder


@dataclass(frozen=True)
class RngDesc:
    distribution: RngDistribution = RngDistribution.UNIFORM
    normal_mean: float = 0.0
    normal_stddev: float = 1.0
    uniform_minimum: float = 0.0
    uniform_maximum: float = 1.0
    bernoulli_probability: float = 0.5


class RngDescBuilder:
    def __init__(self):
        self._distribution = RngDistribution.UNIFORM
        self._normal_mean = 0.0
        self._normal_stddev = 1.0
        self._uniform_minimum = 0.0
        self._uniform_maximum = 1.0
        self._bernoulli_probability = 0.5

    def setDistribution(self, distribution: RngDistribution):
        self._distribution = RngDistribution(distribution)
        return self

    def setNormalDistMean(self, value: float):
        self._normal_mean = float(value)
        return self

    def setNormalDistStandardDeviation(self, value: float):
        self._normal_stddev = float(value)
        return self

    def setUniformDistMinimum(self, value: float):
        self._uniform_minimum = float(value)
        return self

    def setUniformDistMaximum(self, value: float):
        self._uniform_maximum = float(value)
        return self

    def setBernoulliDistProbability(self, value: float):
        self._bernoulli_probability = float(value)
        return self

    def build(self) -> RngDesc:
        return RngDesc(
            distribution=self._distribution,
            normal_mean=self._normal_mean,
            normal_stddev=self._normal_stddev,
            uniform_minimum=self._uniform_minimum,
            uniform_maximum=self._uniform_maximum,
            bernoulli_probability=self._bernoulli_probability,
        )


@dataclass(frozen=True)
class ReductionDesc:
    mode: ReductionMode = ReductionMode.ADD
    math_precision: DataType = DataType.FLOAT


class ReductionDescBuilder:
    def __init__(self):
        self._mode = ReductionMode.ADD
        self._math_precision = DataType.FLOAT

    def setReductionOp(self, mode: ReductionMode):
        self._mode = ReductionMode(mode)
        return self

    def setMathPrecision(self, precision: DataType):
        self._math_precision = DataType(precision)
        return self

    def build(self) -> ReductionDesc:
        return ReductionDesc(
            mode=self._mode,
            math_precision=self._math_precision,
        )


class OperationBuilder:
    _DESCRIPTOR_TYPES = {
        _backend_constants.MUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR: "pointwise",
        _backend_constants.MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR: "conv_fprop",
        _backend_constants.MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR: "conv_bwd_data",
        _backend_constants.MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR: "conv_bwd_filter",
        _backend_constants.MUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR: "matmul",
        _backend_constants.MUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR: "genstats",
        _backend_constants.MUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR: "bn_finalize",
        _backend_constants.MUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR: "bn_bwd_weights",
        _backend_constants.MUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR: "reduction",
        _backend_constants.MUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR: "resample_fwd",
        _backend_constants.MUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR: "resample_bwd",
        _backend_constants.MUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR: "concat",
        _backend_constants.MUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR: "norm_fwd",
        _backend_constants.MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR: "norm_bwd",
        _backend_constants.MUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR: "reshape",
        _backend_constants.MUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR: "transpose",
        _backend_constants.MUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR: "slice",
        _backend_constants.MUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR: "rng",
        _backend_constants.MUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR: "signal",
        _backend_constants.MUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR: "paged_cache_load",
        _backend_constants.MUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR: "block_scale_quantize",
        _backend_constants.MUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR: "block_scale_dequantize",
        _backend_constants.MUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR: "sdpa",
        _backend_constants.MUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR: "sdpa_bwd",
        _backend_constants.MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR: "moe_grouped_matmul",
        _backend_constants.MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR: "moe_grouped_matmul_bwd",
        _backend_constants.MUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR: "rope",
        _backend_constants.MUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR: "rope_bwd",
    }

    def __init__(self, op_type: str | int):
        self._op_type = self._DESCRIPTOR_TYPES.get(op_type, op_type)
        self._inputs: dict[str, Tensor] = {}
        self._outputs: dict[str, Tensor] = {}
        self._attributes: dict[str, object] = {}

    def setxDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setXDesc(self, tensor: Tensor):
        return self.setxDesc(tensor)

    def setbDesc(self, tensor: Tensor):
        self._inputs["b"] = tensor
        return self

    def setBDesc(self, tensor: Tensor):
        return self.setbDesc(tensor)

    def settDesc(self, tensor: Tensor):
        self._inputs["t"] = tensor
        return self

    def setTDesc(self, tensor: Tensor):
        return self.settDesc(tensor)

    def setyDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setYDesc(self, tensor: Tensor):
        return self.setyDesc(tensor)

    def setdxDesc(self, tensor: Tensor):
        self._outputs["dx"] = tensor
        return self

    def setDxDesc(self, tensor: Tensor):
        return self.setdxDesc(tensor)

    def setDXDesc(self, tensor: Tensor):
        return self.setdxDesc(tensor)

    def setdyDesc(self, tensor: Tensor):
        self._inputs["dy"] = tensor
        return self

    def setDyDesc(self, tensor: Tensor):
        return self.setdyDesc(tensor)

    def setDYDesc(self, tensor: Tensor):
        return self.setdyDesc(tensor)

    def setpwDesc(self, desc: PointWiseDesc | PointwiseMode):
        if isinstance(desc, PointWiseDesc):
            self._attributes["pointwise_desc"] = desc
            self._attributes["pointwise_mode"] = desc.mode
        else:
            self._attributes["pointwise_mode"] = PointwiseMode(desc)
        return self

    def setPwDesc(self, desc: PointWiseDesc | PointwiseMode):
        return self.setpwDesc(desc)

    def setmatmulDesc(self, desc: MatMulDesc):
        self._attributes["matmul_desc"] = desc
        return self

    def setMatmulDesc(self, desc: MatMulDesc):
        return self.setmatmulDesc(desc)

    def setconvDesc(self, desc: ConvDesc):
        self._attributes["conv_desc"] = desc
        return self

    def setConvDesc(self, desc: ConvDesc):
        return self.setconvDesc(desc)

    def setreductionDesc(self, desc: ReductionDesc | ReductionMode):
        if isinstance(desc, ReductionDesc):
            self._attributes["reduction_desc"] = desc
            self._attributes["reduction_mode"] = desc.mode
        else:
            self._attributes["reduction_mode"] = ReductionMode(desc)
        return self

    def setReductionDesc(self, desc: ReductionDesc | ReductionMode):
        return self.setreductionDesc(desc)

    def setresampleDesc(self, desc: ResampleDesc | ResampleMode):
        if isinstance(desc, ResampleDesc):
            self._attributes["resample_desc"] = desc
            self._attributes["resample_mode"] = desc.mode
        else:
            self._attributes["resample_mode"] = ResampleMode(desc)
        return self

    def setResampleDesc(self, desc: ResampleDesc | ResampleMode):
        return self.setresampleDesc(desc)

    def setPoolingDesc(self, desc: ResampleDesc | ResampleMode):
        return self.setresampleDesc(desc)

    def setaMatDesc(self, tensor: Tensor):
        self._inputs["a"] = tensor
        return self

    def setAMatDesc(self, tensor: Tensor):
        return self.setaMatDesc(tensor)

    def setbMatDesc(self, tensor: Tensor):
        self._inputs["b"] = tensor
        return self

    def setBMatDesc(self, tensor: Tensor):
        return self.setbMatDesc(tensor)

    def setcMatDesc(self, tensor: Tensor):
        self._outputs["c"] = tensor
        return self

    def setCMatDesc(self, tensor: Tensor):
        return self.setcMatDesc(tensor)

    def setReductionXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setReductionYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setGroupOffsetDesc(self, tensor: Tensor):
        self._inputs["group_offset"] = tensor
        return self

    def setGenStatsMode(self, mode: GenStatsMode):
        self._attributes["mode"] = GenStatsMode(mode)
        return self

    def setGenStatsMathPrecision(self, precision: DataType):
        self._attributes["math_precision"] = DataType(precision)
        return self

    def setGenStatsXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setGenStatsSumDesc(self, tensor: Tensor):
        self._outputs["sum"] = tensor
        return self

    def setGenStatsSqSumDesc(self, tensor: Tensor):
        self._outputs["sqsum"] = tensor
        return self

    def setBnFinalizeStatsMode(self, mode: BnFinalizeStatsMode):
        self._attributes["mode"] = BnFinalizeStatsMode(mode)
        return self

    def setBnFinalizeMathPrecision(self, precision: DataType):
        self._attributes["math_precision"] = DataType(precision)
        return self

    def setBnFinalizeYSumDesc(self, tensor: Tensor):
        self._inputs["y_sum"] = tensor
        return self

    def setBnFinalizeYSqSumDesc(self, tensor: Tensor):
        self._inputs["y_sq_sum"] = tensor
        return self

    def setBnFinalizeScaleDesc(self, tensor: Tensor):
        self._inputs["scale"] = tensor
        return self

    def setBnFinalizeBiasDesc(self, tensor: Tensor):
        self._inputs["bias"] = tensor
        return self

    def setBnFinalizePrevRunningMeanDesc(self, tensor: Tensor):
        self._inputs["prev_running_mean"] = tensor
        return self

    def setBnFinalizePrevRunningVarDesc(self, tensor: Tensor):
        self._inputs["prev_running_var"] = tensor
        return self

    def setBnFinalizeUpdatedRunningMeanDesc(self, tensor: Tensor):
        self._outputs["updated_running_mean"] = tensor
        return self

    def setBnFinalizeUpdatedRunningVarDesc(self, tensor: Tensor):
        self._outputs["updated_running_var"] = tensor
        return self

    def setBnFinalizeSavedMeanDesc(self, tensor: Tensor):
        self._outputs["saved_mean"] = tensor
        return self

    def setBnFinalizeSavedInvStdDesc(self, tensor: Tensor):
        self._outputs["saved_inv_std"] = tensor
        return self

    def setBnFinalizeEqScaleDesc(self, tensor: Tensor):
        self._outputs["eq_scale"] = tensor
        return self

    def setBnFinalizeEqBiasDesc(self, tensor: Tensor):
        self._outputs["eq_bias"] = tensor
        return self

    def setBnFinalizeAccumCountDesc(self, tensor: Tensor):
        self._inputs["accum_count"] = tensor
        return self

    def setBnFinalizeEpsilon(self, epsilon: float):
        self._attributes["epsilon"] = float(epsilon)
        return self

    def setBnFinalizeExpAvgFactor(self, factor: float):
        self._attributes["momentum"] = float(factor)
        return self

    def setBnBwdWeightsMathPrecision(self, precision: DataType):
        self._attributes["math_precision"] = DataType(precision)
        return self

    def setBnBwdWeightsMeanDesc(self, tensor: Tensor):
        self._inputs["mean"] = tensor
        return self

    def setBnBwdWeightsInvstdDesc(self, tensor: Tensor):
        self._inputs["invstd"] = tensor
        return self

    def setBnBwdWeightsInvStdDesc(self, tensor: Tensor):
        return self.setBnBwdWeightsInvstdDesc(tensor)

    def setBnBwdWeightsScaleDesc(self, tensor: Tensor):
        self._inputs["scale"] = tensor
        return self

    def setBnBwdWeightsBnScaleDesc(self, tensor: Tensor):
        return self.setBnBwdWeightsScaleDesc(tensor)

    def setBnBwdWeightsXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setBnBwdWeightsDyDesc(self, tensor: Tensor):
        self._inputs["dy"] = tensor
        return self

    def setBnBwdWeightsDYDesc(self, tensor: Tensor):
        return self.setBnBwdWeightsDyDesc(tensor)

    def setBnBwdWeightsDBnScaleDesc(self, tensor: Tensor):
        self._outputs["dbn_scale"] = tensor
        return self

    def setBnBwdWeightsDBNScaleDesc(self, tensor: Tensor):
        return self.setBnBwdWeightsDBnScaleDesc(tensor)

    def setBnBwdWeightsDBnBiasDesc(self, tensor: Tensor):
        self._outputs["dbn_bias"] = tensor
        return self

    def setBnBwdWeightsDBNBiasDesc(self, tensor: Tensor):
        return self.setBnBwdWeightsDBnBiasDesc(tensor)

    def setBnBwdWeightsEqDyScaleDesc(self, tensor: Tensor):
        self._outputs["eq_dy_scale"] = tensor
        return self

    def setBnBwdWeightsEqDYScaleDesc(self, tensor: Tensor):
        return self.setBnBwdWeightsEqDyScaleDesc(tensor)

    def setBnBwdWeightsEqXScaleDesc(self, tensor: Tensor):
        self._outputs["eq_x_scale"] = tensor
        return self

    def setBnBwdWeightsEqBiasDesc(self, tensor: Tensor):
        self._outputs["eq_bias"] = tensor
        return self

    def setResampleXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setResampleYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setResampleDxDesc(self, tensor: Tensor):
        self._outputs["dx"] = tensor
        return self

    def setResampleDXDesc(self, tensor: Tensor):
        return self.setResampleDxDesc(tensor)

    def setResampleDyDesc(self, tensor: Tensor):
        self._inputs["dy"] = tensor
        return self

    def setResampleDYDesc(self, tensor: Tensor):
        return self.setResampleDyDesc(tensor)

    def setResampleIndexDesc(self, tensor: Tensor):
        self._outputs["idx"] = tensor
        return self

    def setPoolingXDesc(self, tensor: Tensor):
        return self.setResampleXDesc(tensor)

    def setPoolingYDesc(self, tensor: Tensor):
        return self.setResampleYDesc(tensor)

    def setPoolingDxDesc(self, tensor: Tensor):
        return self.setResampleDxDesc(tensor)

    def setPoolingDyDesc(self, tensor: Tensor):
        return self.setResampleDyDesc(tensor)

    def setConcatInputDescs(self, tensors: Sequence[Tensor]):
        self._inputs = {f"x{index}": tensor for index, tensor in enumerate(tensors)}
        return self

    def setConcatOutputDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setConcatAxis(self, axis: int):
        self._attributes["axis"] = int(axis)
        return self

    def setConcatInplaceIndex(self, index: int):
        self._attributes["inplace_index"] = int(index)
        return self

    def setReshapeXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setReshapeYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setReshapeMode(self, mode: ReshapeMode):
        self._attributes["reshape_mode"] = ReshapeMode(mode)
        return self

    def setTransposeXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setTransposeYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setTransposePermutation(self, values: Sequence[int]):
        self._attributes["permutation"] = tuple(int(value) for value in values)
        return self

    def setSliceXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setSliceYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setSliceStartIndices(self, values: Sequence[int]):
        self._attributes["start_indices"] = tuple(int(value) for value in values)
        return self

    def setSliceLimitIndices(self, values: Sequence[int]):
        self._attributes["limit_indices"] = tuple(int(value) for value in values)
        return self

    def setSliceStrides(self, values: Sequence[int]):
        self._attributes["strides"] = tuple(int(value) for value in values)
        return self

    def setRngDesc(self, desc: "RngDesc"):
        self._attributes["rng_desc"] = desc
        self._attributes["rng_distribution"] = desc.distribution
        return self

    def setRngYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setRngSeed(self, seed: int):
        self._attributes["seed"] = int(seed)
        return self

    def setRngOffsetDesc(self, tensor: Tensor):
        self._inputs["offset"] = tensor
        return self

    def setNormMode(self, mode: NormMode):
        self._attributes["mode"] = NormMode(mode)
        return self

    def setNormFwdPhase(self, phase: NormFwdPhase):
        self._attributes["phase"] = NormFwdPhase(phase)
        return self

    def setNormFwdXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setNormFwdMeanDesc(self, tensor: Tensor):
        self._inputs["mean"] = tensor
        return self

    def setNormFwdInvVarianceDesc(self, tensor: Tensor):
        self._inputs["variance"] = tensor
        return self

    def setNormFwdScaleDesc(self, tensor: Tensor):
        self._inputs["scale"] = tensor
        return self

    def setNormFwdBiasDesc(self, tensor: Tensor):
        self._inputs["bias"] = tensor
        return self

    def setNormFwdInputRunningMeanDesc(self, tensor: Tensor):
        self._inputs["input_running_mean"] = tensor
        return self

    def setNormFwdInputRunningVarDesc(self, tensor: Tensor):
        self._inputs["input_running_variance"] = tensor
        return self

    def setNormFwdOutputRunningMeanDesc(self, tensor: Tensor):
        self._outputs["output_running_mean"] = tensor
        return self

    def setNormFwdOutputRunningVarDesc(self, tensor: Tensor):
        self._outputs["output_running_variance"] = tensor
        return self

    def setNormFwdYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setNormFwdEpsilon(self, epsilon: float | Tensor):
        if isinstance(epsilon, Tensor):
            self._inputs["epsilon"] = epsilon
        else:
            self._attributes["epsilon"] = float(epsilon)
        return self

    def setNormFwdExpAvgFactor(self, factor: float | Tensor):
        if isinstance(factor, Tensor):
            self._inputs["momentum"] = factor
        else:
            self._attributes["momentum"] = float(factor)
        return self

    def setNormBwdMode(self, mode: NormMode):
        self._attributes["mode"] = NormMode(mode)
        return self

    def setNormBwdXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setNormBwdMeanDesc(self, tensor: Tensor):
        self._inputs["mean"] = tensor
        return self

    def setNormBwdInvVarianceDesc(self, tensor: Tensor):
        self._inputs["variance"] = tensor
        return self

    def setNormBwdDyDesc(self, tensor: Tensor):
        self._inputs["dy"] = tensor
        return self

    def setNormBwdScaleDesc(self, tensor: Tensor):
        self._inputs["scale"] = tensor
        return self

    def setNormBwdDxDesc(self, tensor: Tensor):
        self._outputs["dx"] = tensor
        return self

    def setNormBwdDScaleDesc(self, tensor: Tensor):
        self._outputs["dscale"] = tensor
        return self

    def setNormBwdDBiasDesc(self, tensor: Tensor):
        self._outputs["dbias"] = tensor
        return self

    def setNormBwdEpsilon(self, epsilon: float | Tensor):
        if isinstance(epsilon, Tensor):
            self._inputs["epsilon"] = epsilon
        else:
            self._attributes["epsilon"] = float(epsilon)
        return self

    def setBlockScaleQuantizeXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setBlockScaleQuantizeYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setBlockScaleQuantizeScaleDesc(self, tensor: Tensor):
        self._outputs["scale"] = tensor
        return self

    def setBlockScaleQuantizeMathPrecision(self, precision: DataType):
        self._attributes["math_precision"] = DataType(precision)
        return self

    def setBlockScaleQuantizeBlockSize(self, block_size: int):
        self._attributes["block_size"] = int(block_size)
        return self

    def setBlockScaleDequantizeXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setBlockScaleDequantizeScaleDesc(self, tensor: Tensor):
        self._inputs["scale"] = tensor
        return self

    def setBlockScaleDequantizeYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setBlockScaleDequantizeMathPrecision(self, precision: DataType):
        self._attributes["math_precision"] = DataType(precision)
        return self

    def setBlockScaleDequantizeBlockSize(self, block_size: int):
        self._attributes["block_size"] = int(block_size)
        return self

    def setBlockScaleDequantizeNegativeScale(self, value: bool):
        self._attributes["is_negative_scale"] = bool(value)
        return self

    def setBlockScaleDequantizeIsNegativeScale(self, value: bool):
        return self.setBlockScaleDequantizeNegativeScale(value)

    def setMoeGroupedMatmulMode(self, mode: MoeGroupedMatmulMode):
        self._attributes["mode"] = MoeGroupedMatmulMode(mode)
        return self

    def setMoeGroupedMatmulMathPrecision(self, precision: DataType):
        self._attributes["compute_data_type"] = DataType(precision)
        return self

    def setMoeGroupedMatmulTokenDesc(self, tensor: Tensor):
        self._inputs["token"] = tensor
        return self

    def setMoeGroupedMatmulWeightDesc(self, tensor: Tensor):
        self._inputs["weight"] = tensor
        return self

    def setMoeGroupedMatmulFirstTokenOffsetDesc(self, tensor: Tensor):
        self._inputs["first_token_offset"] = tensor
        return self

    def setMoeGroupedMatmulOutputDesc(self, tensor: Tensor):
        self._outputs["output"] = tensor
        return self

    def setMoeGroupedMatmulTokenIndexDesc(self, tensor: Tensor):
        self._inputs["token_index"] = tensor
        return self

    def setMoeGroupedMatmulTokenKsDesc(self, tensor: Tensor):
        self._inputs["token_ks"] = tensor
        return self

    def setMoeGroupedMatmulTopK(self, top_k: int):
        self._attributes["top_k"] = int(top_k)
        return self

    def setMoeGroupedMatmulBwdMathPrecision(self, precision: DataType):
        self._attributes["compute_data_type"] = DataType(precision)
        return self

    def setMoeGroupedMatmulBwdTokenDesc(self, tensor: Tensor):
        self._inputs["token"] = tensor
        return self

    def setMoeGroupedMatmulBwdDWeightDesc(self, tensor: Tensor):
        self._outputs["dweight"] = tensor
        return self

    def setMoeGroupedMatmulBwdFirstTokenOffsetDesc(self, tensor: Tensor):
        self._inputs["first_token_offset"] = tensor
        return self

    def setMoeGroupedMatmulBwdDOutputDesc(self, tensor: Tensor):
        self._inputs["doutput"] = tensor
        return self

    def setPagedCacheLoadContainerDesc(self, tensor: Tensor):
        self._inputs["container"] = tensor
        return self

    def setPagedCacheLoadYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setPagedCacheLoadSequenceDesc(self, tensor: Tensor):
        self._inputs["sequence"] = tensor
        return self

    def setPagedCacheLoadPageTableDesc(self, tensor: Tensor):
        self._inputs["page_table"] = tensor
        return self

    def setSignalMode(self, mode: SignalMode):
        self._attributes["mode"] = SignalMode(mode)
        return self

    def setSignalValue(self, value: int):
        self._attributes["value"] = int(value)
        return self

    def setSignalFlagDesc(self, tensor: Tensor):
        self._inputs["flag"] = tensor
        return self

    def setSignalXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setSignalYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setConvXDesc(self, tensor: Tensor):
        self._inputs["x"] = tensor
        return self

    def setConvWDesc(self, tensor: Tensor):
        self._inputs["w"] = tensor
        return self

    def setConvYDesc(self, tensor: Tensor):
        self._outputs["y"] = tensor
        return self

    def setConvDxDesc(self, tensor: Tensor):
        self._outputs["dx"] = tensor
        return self

    def setConvDXDesc(self, tensor: Tensor):
        return self.setConvDxDesc(tensor)

    def setConvDyDesc(self, tensor: Tensor):
        self._inputs["dy"] = tensor
        return self

    def setConvDYDesc(self, tensor: Tensor):
        return self.setConvDyDesc(tensor)

    def setConvDwDesc(self, tensor: Tensor):
        self._outputs["dw"] = tensor
        return self

    def setConvDWDesc(self, tensor: Tensor):
        return self.setConvDwDesc(tensor)

    def setqDesc(self, tensor: Tensor):
        self._inputs["q"] = tensor
        return self

    def setQDesc(self, tensor: Tensor):
        return self.setqDesc(tensor)

    def setkDesc(self, tensor: Tensor):
        self._inputs["k"] = tensor
        return self

    def setKDesc(self, tensor: Tensor):
        return self.setkDesc(tensor)

    def setvDesc(self, tensor: Tensor):
        self._inputs["v"] = tensor
        return self

    def setVDesc(self, tensor: Tensor):
        return self.setvDesc(tensor)

    def setoDesc(self, tensor: Tensor):
        if self._op_type == "sdpa_bwd":
            self._inputs["o"] = tensor
        else:
            self._outputs["o"] = tensor
        return self

    def setODesc(self, tensor: Tensor):
        return self.setoDesc(tensor)

    def setattnProbsDesc(self, tensor: Tensor):
        self._outputs["attn_probs"] = tensor
        return self

    def setAttnProbsDesc(self, tensor: Tensor):
        return self.setattnProbsDesc(tensor)

    def setdropoutMaskDesc(self, tensor: Tensor):
        self._outputs["dropout_mask"] = tensor
        return self

    def setDropoutMaskDesc(self, tensor: Tensor):
        return self.setdropoutMaskDesc(tensor)

    def setSdpaStatsDesc(self, tensor: Tensor):
        self._outputs["stats"] = tensor
        return self

    def setSdpaFwdStatsDesc(self, tensor: Tensor):
        return self.setSdpaStatsDesc(tensor)

    def setmaskDesc(self, tensor: Tensor):
        self._inputs["mask"] = tensor
        return self

    def setMaskDesc(self, tensor: Tensor):
        return self.setmaskDesc(tensor)

    def setbiasDesc(self, tensor: Tensor):
        self._inputs["bias"] = tensor
        return self

    def setBiasDesc(self, tensor: Tensor):
        return self.setbiasDesc(tensor)

    def setAlpha(self, alpha: float):
        self._attributes["alpha1"] = float(alpha)
        return self

    def setAlpha2(self, alpha: float):
        self._attributes["alpha2"] = float(alpha)
        return self

    def setBeta(self, beta: float):
        self._attributes["beta"] = float(beta)
        return self

    def setSdpaScale(self, scale: float):
        self._attributes["sdpa_scale"] = float(scale)
        return self

    def setSdpaScaleDesc(self, tensor: Tensor):
        self._inputs["scale"] = tensor
        return self

    def setSdpaInference(self, is_inference: bool):
        self._attributes["sdpa_is_inference"] = bool(is_inference)
        return self

    def setSdpaGenerateStats(self, generate_stats: bool):
        self._attributes["sdpa_is_inference"] = not bool(generate_stats)
        self._attributes["sdpa_generate_stats"] = bool(generate_stats)
        return self

    def setSdpaCausalMask(self, causal_mask: bool):
        self._attributes["sdpa_causal_mask"] = bool(causal_mask)
        return self

    def setSdpaUsePaddingMask(self, use_padding_mask: bool):
        self._attributes["sdpa_use_padding_mask"] = bool(use_padding_mask)
        return self

    def setSdpaUseCausalMaskBottomRight(self, use_bottom_right: bool):
        self._attributes["sdpa_use_causal_mask_bottom_right"] = bool(use_bottom_right)
        return self

    def setSdpaDiagonalAlignment(self, alignment: DiagonalAlignment | int | str):
        self._attributes["sdpa_diagonal_alignment"] = _normalize_diagonal_alignment(
            alignment
        )
        return self

    def setSdpaDiagonalBandRightBound(self, bound: int):
        self._attributes["sdpa_diagonal_band_right_bound"] = int(bound)
        if int(bound) == 0:
            self._attributes["sdpa_causal_mask"] = True
        return self

    def setSdpaDiagonalBandLeftBound(self, bound: int):
        self._attributes["sdpa_diagonal_band_left_bound"] = int(bound)
        return self

    def setSdpaUseAlibiMask(self, use_alibi_mask: bool):
        if bool(use_alibi_mask):
            raise ValueError("mudnn_frontend does not support ALiBi SDPA")
        return self

    def setSdpaDropout(self, dropout: float):
        if float(dropout) != 0.0:
            raise ValueError("mudnn_frontend does not support SDPA dropout")
        return self

    def setSdpaSlidingWindowLength(self, length: int):
        raise ValueError("mudnn_frontend does not support SDPA sliding_window_length")

    def setSdpaSeqLenQDesc(self, tensor: Tensor):
        self._inputs["seq_len_q"] = tensor
        return self

    def setSdpaSeqLenKvDesc(self, tensor: Tensor):
        self._inputs["seq_len_kv"] = tensor
        return self

    def setSdpaSeqLenKVDesc(self, tensor: Tensor):
        return self.setSdpaSeqLenKvDesc(tensor)

    def setSdpaMaxTotalSeqLenQ(self, max_total_seq_len: int):
        raise ValueError(
            "mudnn_frontend does not support ragged SDPA max sequence lengths"
        )

    def setSdpaMaxTotalSeqLenKv(self, max_total_seq_len: int):
        raise ValueError(
            "mudnn_frontend does not support ragged SDPA max sequence lengths"
        )

    def setSdpaMaxTotalSeqLenKV(self, max_total_seq_len: int):
        return self.setSdpaMaxTotalSeqLenKv(max_total_seq_len)

    def setSdpaScoreMod(self, score_mod):
        raise ValueError("mudnn_frontend does not support SDPA score modifiers")

    def setSdpaScoreSumExpDesc(self, tensor: Tensor):
        raise ValueError(
            "mudnn_frontend does not support SDPA score_sum_exp descriptors"
        )

    def setSdpaScoreMaxDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA score_max descriptors")

    def setSdpaSinkTokenDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA sink-token descriptors")

    def setSdpaDescaleQDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_q")

    def setSdpaDescaleKDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_k")

    def setSdpaDescaleVDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_v")

    def setSdpaDescaleSDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_s")

    def setSdpaScaleSDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 scale_s")

    def setSdpaScaleODesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 scale_o")

    def setSdpaBwdDoDesc(self, tensor: Tensor):
        self._inputs["d_o"] = tensor
        return self

    def setSdpaBwdDODesc(self, tensor: Tensor):
        return self.setSdpaBwdDoDesc(tensor)

    def setSdpaBwdStatsDesc(self, tensor: Tensor):
        self._inputs["stats"] = tensor
        return self

    def setSdpaBwdDqDesc(self, tensor: Tensor):
        self._outputs["d_q"] = tensor
        return self

    def setSdpaBwdDQDesc(self, tensor: Tensor):
        return self.setSdpaBwdDqDesc(tensor)

    def setSdpaBwdDkDesc(self, tensor: Tensor):
        self._outputs["d_k"] = tensor
        return self

    def setSdpaBwdDKDesc(self, tensor: Tensor):
        return self.setSdpaBwdDkDesc(tensor)

    def setSdpaBwdDvDesc(self, tensor: Tensor):
        self._outputs["d_v"] = tensor
        return self

    def setSdpaBwdDVDesc(self, tensor: Tensor):
        return self.setSdpaBwdDvDesc(tensor)

    def setSdpaBwdDBiasDesc(self, tensor: Tensor):
        self._outputs["d_bias"] = tensor
        return self

    def setSdpaBwdScale(self, scale: float):
        self._attributes["sdpa_scale"] = float(scale)
        return self

    def setSdpaBwdScaleDesc(self, tensor: Tensor):
        self._inputs["scale"] = tensor
        return self

    def setSdpaBwdCausalMask(self, causal_mask: bool):
        self._attributes["sdpa_causal_mask"] = bool(causal_mask)
        return self

    def setSdpaBwdUsePaddingMask(self, use_padding_mask: bool):
        self._attributes["sdpa_use_padding_mask"] = bool(use_padding_mask)
        return self

    def setSdpaBwdUseCausalMaskBottomRight(self, use_bottom_right: bool):
        self._attributes["sdpa_use_causal_mask_bottom_right"] = bool(use_bottom_right)
        return self

    def setSdpaBwdDiagonalAlignment(self, alignment: DiagonalAlignment | int | str):
        self._attributes["sdpa_diagonal_alignment"] = _normalize_diagonal_alignment(
            alignment
        )
        return self

    def setSdpaBwdDiagonalBandRightBound(self, bound: int):
        self._attributes["sdpa_diagonal_band_right_bound"] = int(bound)
        if int(bound) == 0:
            self._attributes["sdpa_causal_mask"] = True
        return self

    def setSdpaBwdDiagonalBandLeftBound(self, bound: int):
        self._attributes["sdpa_diagonal_band_left_bound"] = int(bound)
        return self

    def setSdpaBwdDeterministic(self, deterministic: bool):
        self._attributes["sdpa_bwd_deterministic"] = bool(deterministic)
        return self

    def setSdpaBwdUseDeterministicAlgorithm(self, deterministic: bool):
        return self.setSdpaBwdDeterministic(deterministic)

    def setSdpaBwdUseAlibiMask(self, use_alibi_mask: bool):
        return self.setSdpaUseAlibiMask(use_alibi_mask)

    def setSdpaBwdDropout(self, dropout: float):
        return self.setSdpaDropout(dropout)

    def setSdpaBwdSlidingWindowLength(self, length: int):
        return self.setSdpaSlidingWindowLength(length)

    def setSdpaBwdSeqLenQDesc(self, tensor: Tensor):
        self._inputs["seq_len_q"] = tensor
        return self

    def setSdpaBwdSeqLenKvDesc(self, tensor: Tensor):
        self._inputs["seq_len_kv"] = tensor
        return self

    def setSdpaBwdSeqLenKVDesc(self, tensor: Tensor):
        return self.setSdpaBwdSeqLenKvDesc(tensor)

    def setSdpaBwdMaxTotalSeqLenQ(self, max_total_seq_len: int):
        return self.setSdpaMaxTotalSeqLenQ(max_total_seq_len)

    def setSdpaBwdMaxTotalSeqLenKv(self, max_total_seq_len: int):
        return self.setSdpaMaxTotalSeqLenKv(max_total_seq_len)

    def setSdpaBwdMaxTotalSeqLenKV(self, max_total_seq_len: int):
        return self.setSdpaBwdMaxTotalSeqLenKv(max_total_seq_len)

    def setSdpaBwdScoreMod(self, score_mod):
        raise ValueError(
            "mudnn_frontend does not support SDPA backward score modifiers"
        )

    def setSdpaBwdScoreModBprop(self, score_mod):
        raise ValueError(
            "mudnn_frontend does not support SDPA backward score modifiers"
        )

    def setSdpaBwdSinkTokenDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA sink-token descriptors")

    def setSdpaBwdDSinkTokenDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA dSink-token descriptors")

    def setSdpaBwdDescaleQDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_q")

    def setSdpaBwdDescaleKDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_k")

    def setSdpaBwdDescaleVDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_v")

    def setSdpaBwdDescaleODesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_o")

    def setSdpaBwdDescaleDoDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_dO")

    def setSdpaBwdDescaleDODesc(self, tensor: Tensor):
        return self.setSdpaBwdDescaleDoDesc(tensor)

    def setSdpaBwdDescaleSDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_s")

    def setSdpaBwdDescaleDpDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 descale_dP")

    def setSdpaBwdDescaleDPDesc(self, tensor: Tensor):
        return self.setSdpaBwdDescaleDpDesc(tensor)

    def setSdpaBwdScaleSDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 scale_s")

    def setSdpaBwdScaleDqDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 scale_dQ")

    def setSdpaBwdScaleDQDesc(self, tensor: Tensor):
        return self.setSdpaBwdScaleDqDesc(tensor)

    def setSdpaBwdScaleDkDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 scale_dK")

    def setSdpaBwdScaleDKDesc(self, tensor: Tensor):
        return self.setSdpaBwdScaleDkDesc(tensor)

    def setSdpaBwdScaleDvDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 scale_dV")

    def setSdpaBwdScaleDVDesc(self, tensor: Tensor):
        return self.setSdpaBwdScaleDvDesc(tensor)

    def setSdpaBwdScaleDpDesc(self, tensor: Tensor):
        raise ValueError("mudnn_frontend does not support SDPA FP8 scale_dP")

    def setSdpaBwdScaleDPDesc(self, tensor: Tensor):
        return self.setSdpaBwdScaleDpDesc(tensor)

    def build(self) -> Operation:
        return Operation(
            op_type=self._op_type,
            inputs=dict(self._inputs),
            outputs=dict(self._outputs),
            attributes=dict(self._attributes),
        )


@dataclass(frozen=True)
class OperationGraph:
    handle: object
    operations: tuple[Operation, ...]


class OperationGraphBuilder:
    def __init__(self):
        self._handle = None
        self._operations: tuple[Operation, ...] = ()

    def setHandle(self, handle):
        self._handle = handle
        return self

    def setOperationGraph(self, *args):
        self._operations = tuple(_sequence_from_counted_args(args, "operation graph"))
        return self

    def build(self) -> OperationGraph:
        if not self._operations:
            raise ValueError("operation graph requires at least one operation")
        return OperationGraph(handle=self._handle, operations=self._operations)


@dataclass(frozen=True)
class Engine:
    operation_graph: OperationGraph
    global_index: int = 0


class EngineBuilder:
    def __init__(self):
        self._operation_graph: OperationGraph | None = None
        self._global_index = 0

    def setOperationGraph(self, operation_graph: OperationGraph):
        self._operation_graph = operation_graph
        return self

    def setGlobalEngineIdx(self, global_index: int):
        self._global_index = int(global_index)
        return self

    def build(self) -> Engine:
        if self._operation_graph is None:
            raise ValueError("engine requires an operation graph")
        return Engine(
            operation_graph=self._operation_graph,
            global_index=self._global_index,
        )


@dataclass(frozen=True)
class EngineConfig:
    engine: Engine
    workspace_size: int = 0


class EngineConfigBuilder:
    def __init__(self):
        self._engine: Engine | None = None

    def setEngine(self, engine: Engine):
        self._engine = engine
        return self

    def build(self) -> EngineConfig:
        if self._engine is None:
            raise ValueError("engine config requires an engine")
        return EngineConfig(engine=self._engine)


@dataclass(frozen=True)
class EngineHeuristics:
    operation_graph: OperationGraph
    mode: HeurMode = HeurMode.INSTANT

    def getEngineConfig(self, max_count: int = 1) -> tuple[EngineConfig, ...]:
        if max_count <= 0:
            return ()
        engine = EngineBuilder().setOperationGraph(self.operation_graph).build()
        return (EngineConfigBuilder().setEngine(engine).build(),)


class EngineHeuristicsBuilder:
    def __init__(self):
        self._operation_graph: OperationGraph | None = None
        self._mode = HeurMode.INSTANT

    def setOperationGraph(self, operation_graph: OperationGraph):
        self._operation_graph = operation_graph
        return self

    def setHeurMode(self, mode: HeurMode):
        self._mode = HeurMode(mode)
        return self

    def build(self) -> EngineHeuristics:
        if self._operation_graph is None:
            raise ValueError("engine heuristics requires an operation graph")
        return EngineHeuristics(
            operation_graph=self._operation_graph,
            mode=self._mode,
        )


@dataclass(frozen=True)
class ExecutionPlan:
    handle: object
    engine_config: EngineConfig

    def getWorkspaceSize(self) -> int:
        return self.engine_config.workspace_size


class ExecutionPlanBuilder:
    def __init__(self):
        self._handle = None
        self._engine_config: EngineConfig | None = None

    def setHandle(self, handle):
        self._handle = handle
        return self

    def setEngineConfig(self, engine_config: EngineConfig):
        self._engine_config = engine_config
        return self

    def build(self) -> ExecutionPlan:
        if self._engine_config is None:
            raise ValueError("execution plan requires an engine config")
        return ExecutionPlan(
            handle=self._handle,
            engine_config=self._engine_config,
        )


@dataclass(frozen=True)
class VariantPack:
    uids: tuple[int, ...]
    data_pointers: tuple[int, ...]
    workspace_pointer: int | None = None


class VariantPackBuilder:
    def __init__(self):
        self._uids: tuple[int, ...] = ()
        self._data_pointers: tuple[int, ...] = ()
        self._workspace_pointer: int | None = None

    def setUids(self, *args):
        self._uids = tuple(
            int(uid) for uid in _sequence_from_counted_args(args, "uids")
        )
        return self

    def setDataPointers(self, *args):
        self._data_pointers = tuple(
            int(pointer)
            for pointer in _sequence_from_counted_args(args, "data pointers")
        )
        return self

    def setWorkspacePointer(self, pointer: int):
        self._workspace_pointer = int(pointer)
        return self

    def build(self) -> VariantPack:
        if len(self._uids) != len(self._data_pointers):
            raise ValueError("uids and data_pointers must have the same length")
        return VariantPack(self._uids, self._data_pointers, self._workspace_pointer)
