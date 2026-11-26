"""Python helpers for building muDNN backend graph descriptors.

This module mirrors a small, intentionally compatible subset of NVIDIA's
``cudnn_frontend`` Python surface while targeting muDNN names and enums.
"""

import ctypes
import json
import re
from contextlib import contextmanager
from enum import IntEnum
from functools import wraps
from pathlib import Path

from . import _backend
from .graph import (
    DataType,
    HeurMode,
    PointwiseMode,
    AttentionImplementation,
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
    DiagonalAlignment,
    Graph,
    GraphNode,
    GraphTensor,
    Tensor,
    TensorBuilder,
    PointWiseDesc,
    PointWiseDescBuilder,
    PointwiseDesc,
    PointwiseDescBuilder,
    MatMulDesc,
    MatMulDescBuilder,
    ConvDesc,
    ConvDescBuilder,
    ConvolutionDesc,
    ConvolutionDescBuilder,
    ResampleDesc,
    ResampleDescBuilder,
    PoolingDesc,
    PoolingDescBuilder,
    RngDesc,
    RngDescBuilder,
    ReductionDesc,
    ReductionDescBuilder,
    Operation,
    OperationBuilder,
    OperationGraph,
    OperationGraphBuilder,
    Engine,
    EngineBuilder,
    EngineConfig,
    EngineConfigBuilder,
    EngineHeuristics,
    EngineHeuristicsBuilder,
    ExecutionPlan,
    ExecutionPlanBuilder,
    VariantPack,
    VariantPackBuilder,
    pygraph,
    get_data_ptr as _get_data_ptr,
)

# Upstream exposes pygraph as the compiled class, not a separate factory
# function.  Graph accepts the same positional constructor arguments.
pygraph = Graph


def _install_builder_alias(cls, name, target):
    if hasattr(cls, name):
        return

    def method(self, *args, **kwargs):
        return getattr(self, target)(*args, **kwargs)

    method.__name__ = name
    setattr(cls, name, method)


_install_builder_alias(PointWiseDescBuilder, "setComputeType", "setMathPrecision")
_install_builder_alias(MatMulDescBuilder, "setComputeType", "setMathPrecision")
_install_builder_alias(ConvDescBuilder, "setComputePrecision", "setComputeType")
_install_builder_alias(ConvDescBuilder, "setNDims", "setSpatialDimCount")
_install_builder_alias(ResampleDescBuilder, "setComputeType", "setMathPrecision")
_install_builder_alias(ResampleDescBuilder, "setResampleMode", "setMode")
_install_builder_alias(ResampleDescBuilder, "setSpatialDim", "setWindowDim")
_install_builder_alias(ResampleDescBuilder, "setSpatialStride", "setStride")
_install_builder_alias(RngDescBuilder, "setRngDistribution", "setDistribution")
_install_builder_alias(
    RngDescBuilder, "setNormalDistStdDev", "setNormalDistStandardDeviation"
)
_install_builder_alias(RngDescBuilder, "setUniformDistMax", "setUniformDistMaximum")
_install_builder_alias(RngDescBuilder, "setUniformDistMin", "setUniformDistMinimum")
_install_builder_alias(ReductionDescBuilder, "setComputeType", "setMathPrecision")


def _set_ragged_offset(self, tensor):
    self._ragged_offset = tensor
    return self


def _set_clipping(self, lower, upper):
    return self.setReluLowerClip(lower).setReluUpperClip(upper)


def _set_normal_dist_params(self, mean, stddev):
    return self.setNormalDistMean(mean).setNormalDistStandardDeviation(stddev)


def _set_uniform_dist_params(self, maximum, minimum):
    return self.setUniformDistMaximum(maximum).setUniformDistMinimum(minimum)


def _store_builder_value(attribute):
    def setter(self, value):
        setattr(self, attribute, value)
        return self

    return setter


TensorBuilder.setRaggedOffset = _set_ragged_offset
PointWiseDescBuilder.setClipping = _set_clipping
RngDescBuilder.setNormalDistParams = _set_normal_dist_params
RngDescBuilder.setUniformDistParams = _set_uniform_dist_params
ResampleDescBuilder.setNanPropagation = _store_builder_value("_nan_propagation")
ResampleDescBuilder.setPaddingMode = _store_builder_value("_padding_mode")
ReductionDescBuilder.setIsDeterministic = _store_builder_value("_is_deterministic")
OperationGraphBuilder.setIsDynamicShapeEnabled = _store_builder_value(
    "_is_dynamic_shape_enabled"
)
OperationGraphBuilder.setIsOverrideShapeEnabled = _store_builder_value(
    "_is_override_shape_enabled"
)
ExecutionPlanBuilder.setKernelCache = _store_builder_value("_kernel_cache")


_OPERATION_BUILDER_ALIASES = {
    "setAccumCountTensor": "setBnFinalizeAccumCountDesc",
    "setBNFinalizeMode": "setBnFinalizeStatsMode",
    "setBias": "setNormFwdBiasDesc",
    "setDBias": "setNormBwdDBiasDesc",
    "setDScale": "setNormBwdDScaleDesc",
    "setEpsilonTensor": "setNormFwdEpsilon",
    "setExpDecayFactorTensor": "setNormFwdExpAvgFactor",
    "setNormalizationMode": "setNormMode",
    "setOffsetDesc": "setRngOffsetDesc",
    "setSavedInvVar": "setNormFwdInvVarianceDesc",
    "setScale": "setNormFwdScaleDesc",
    "setSqSumDesc": "setGenStatsSqSumDesc",
    "setSumDesc": "setGenStatsSumDesc",
    "setcDesc": "setConvDesc",
    "setcontainerDesc": "setPagedCacheLoadContainerDesc",
    "setdwDesc": "setConvDwDesc",
    "setidxDesc": "setResampleIndexDesc",
    "setpageTableDesc": "setPagedCacheLoadPageTableDesc",
    "setsequenceDesc": "setPagedCacheLoadSequenceDesc",
    "setwDesc": "setConvWDesc",
}
for _alias_name, _target_name in _OPERATION_BUILDER_ALIASES.items():
    _install_builder_alias(OperationBuilder, _alias_name, _target_name)


def _set_matmul_override(name):
    def setter(self, tensor):
        self._inputs[name] = tensor
        return self

    return setter


OperationBuilder.setmOverrideDesc = _set_matmul_override("m_override")
OperationBuilder.setnOverrideDesc = _set_matmul_override("n_override")
OperationBuilder.setkOverrideDesc = _set_matmul_override("k_override")


def _chain_operation_setters(*targets):
    def setter(self, *values):
        if len(targets) != len(values):
            raise TypeError(f"expected {len(targets)} tensor arguments")
        for target, value in zip(targets, values):
            getattr(self, target)(value)
        return self

    return setter


OperationBuilder.setDScaleAndDBias = _chain_operation_setters(
    "setNormBwdDScaleDesc", "setNormBwdDBiasDesc"
)
OperationBuilder.setScaleAndBias = _chain_operation_setters(
    "setNormFwdScaleDesc", "setNormFwdBiasDesc"
)
OperationBuilder.setSavedMeanAndInvVar = _chain_operation_setters(
    "setNormFwdMeanDesc", "setNormFwdInvVarianceDesc"
)
OperationBuilder.setPrevRunningMeanAndVar = _chain_operation_setters(
    "setNormFwdInputRunningMeanDesc", "setNormFwdInputRunningVarDesc"
)
OperationBuilder.setNextRunningMeanAndVar = _chain_operation_setters(
    "setNormFwdOutputRunningMeanDesc", "setNormFwdOutputRunningVarDesc"
)
OperationBuilder.setEqScaleAndBias = _chain_operation_setters(
    "setBnFinalizeEqScaleDesc", "setBnFinalizeEqBiasDesc"
)
OperationBuilder.setEqScalesAndBias = _chain_operation_setters(
    "setBnBwdWeightsEqDyScaleDesc",
    "setBnBwdWeightsEqXScaleDesc",
    "setBnBwdWeightsEqBiasDesc",
)


def _unsupported_seed_descriptor(self, tensor):
    raise NotImplementedError(
        "muDNN RNG accepts a scalar seed but not a seed tensor descriptor"
    )


OperationBuilder.setSeedDesc = _unsupported_seed_descriptor

from ._version import __version__  # noqa: E402

BackendError = _backend.BackendError


class cudnnGraphNotSupportedError(BackendError):
    pass


class BuildPlanPolicy(IntEnum):
    HEURISTICS_CHOICE = 0
    ALL = 1


class TensorReordering(IntEnum):
    NONE = 0
    INT8x32 = 1
    F16x16 = 2
    F8_128x4 = 3


class ScalarType(IntEnum):
    RUNTIME_PARAM = 0
    COMPILE_TIME_CONST = 1
    INT32 = 100
    INT64 = 101
    FLOAT = 102
    DOUBLE = 103


class BehaviorNote(IntEnum):
    RUNTIME_COMPILATION = 0
    REQUIRES_FILTER_INT8x32_REORDER = 1
    REQUIRES_BIAS_INT8x32_REORDER = 2
    SUPPORTS_MUSA_GRAPH_NATIVE_API = 3
    SUPPORTS_CUDA_GRAPH_NATIVE_API = 3
    MUBLASLT_DEPENDENCY = 4
    CUBLASLT_DEPENDENCY = 4


class NumericalNote(IntEnum):
    TENSOR_CORE = 0
    DOWN_CONVERT_INPUTS = 1
    REDUCED_PRECISION_REDUCTION = 2
    FFT = 3
    NONDETERMINISTIC = 4
    WINOGRAD = 5
    WINOGRAD_TILE_4x4 = 6
    WINOGRAD_TILE_6x6 = 7
    WINOGRAD_TILE_13x13 = 8
    STRICT_NAN_PROP = 9


class KnobType(IntEnum):
    NOT_SET = 0
    SPLIT_K = 0
    SWIZZLE = 1
    TILE_SIZE = 2
    USE_TEX = 3
    EDGE = 3
    MULTIPLY = 4
    SPLIT_K_BUF = 5
    TILEK = 6
    STAGES = 7
    REDUCTION_MODE = 8
    SPLIT_K_SLC = 9
    IDX_MODE = 10
    SPECFILT = 11
    KERNEL_CFG = 12
    WORKSPACE = 13
    TILE_CGA_M = 14
    TILE_CGA_N = 15
    BLOCK_SIZE = 16
    OCCUPANCY = 17
    ARRAY_SIZE_PER_THREAD = 18
    SPLIT_COLS = 19
    TILE_ROWS = 20
    TILE_COLS = 21
    LOAD_SIZE = 22
    CTA_COUNT = 23
    STREAM_K = 24
    SPLIT_P_SLC = 25
    TILE_M = 26
    TILE_N = 27
    WARP_SPEC_CFG = 28
    SWAP_AB = 29
    INPUT_TMA_ENABLE = 30
    OUTPUT_TMA_ENABLE = 31


class knob:
    def __init__(self, type=None, max_value=0, min_value=0, stride=1):
        self.type = type
        self.max_value = int(max_value)
        self.min_value = int(min_value)
        self.stride = int(stride)


class _UnsupportedOpaqueHandle:
    def __init__(self, name: str, payload=None):
        self.name = name
        self._payload = dict(payload or {})
        self._deserialized = False

    def __repr__(self):
        return f"<{self.name} unsupported by muDNN frontend>"

    def serialize(self) -> str:
        payload = {
            "format": f"mudnn_frontend.{self.name}",
            "version": 1,
            "payload": self._payload,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def deserialize(self, data):
        if self._deserialized or self._payload:
            raise RuntimeError(f"{self.name} has already been initialized")
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        payload = json.loads(str(data))
        if payload.get("format") != f"mudnn_frontend.{self.name}":
            raise ValueError(f"serialized data is not a {self.name}")
        self._payload = dict(payload.get("payload", {}))
        self._deserialized = True
        return self


class kernel_cache(_UnsupportedOpaqueHandle):
    def __init__(self, payload=None):
        super().__init__("kernel_cache", payload)


class device_properties(_UnsupportedOpaqueHandle):
    def __init__(self, payload=None):
        super().__init__("device_properties", payload)


_CUDNN_COMPAT_BACKEND_VERSION = 92300


def _fallback_backend_version() -> int:
    for include_name in ("mudnn_frontend.h",):
        version_header = Path(__file__).resolve().parents[2] / "include" / include_name
        try:
            text = version_header.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"#define\s+CUDNN_VERSION\s+(\d+)", text)
        if match:
            return int(match.group(1))

    version_file = (
        Path(__file__).resolve().parents[3]
        / "include"
        / "mudnn"
        / "api"
        / "config"
        / "version.md"
    )
    try:
        text = version_file.read_text(encoding="utf-8")
    except OSError:
        return 0
    values = {}
    for name in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"#define\s+MUDNN_VERSION_{name}\s+(\d+)", text)
        if match:
            values[name] = int(match.group(1))
    if len(values) != 3:
        return 0
    return values["MAJOR"] * 1000 + values["MINOR"] * 100 + values["PATCH"]


def backend_version() -> int:
    return _fallback_backend_version() or _CUDNN_COMPAT_BACKEND_VERSION


def backend_version_string() -> str:
    version = backend_version()
    if version <= 0:
        return "0.0.0"
    if version >= 90000:
        major = version // 10000
        minor = (version % 10000) // 100
        patch = version % 100
    else:
        major = version // 1000
        minor = (version % 1000) // 100
        patch = version % 100
    return f"{major}.{minor}.{patch}"


def get_last_error_string() -> str:
    return ""


def create_handle():
    return _backend.Handle()


def destroy_handle(handle):
    if hasattr(handle, "close"):
        handle.close()
        return None
    lib = _backend.get_library()
    lib.check(lib.lib.mudnnDestroy(ctypes.c_void_p(int(handle))), "mudnnDestroy")
    return None


def _handle_pointer(handle):
    if hasattr(handle, "ptr"):
        return handle.ptr
    return ctypes.c_void_p(int(handle))


def set_stream(handle, stream):
    lib = _backend.get_library()
    lib.lib.mudnnSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.lib.mudnnSetStream.restype = ctypes.c_int
    lib.check(
        lib.lib.mudnnSetStream(_handle_pointer(handle), ctypes.c_void_p(int(stream))),
        "mudnnSetStream",
    )
    return None


def get_stream(handle):
    lib = _backend.get_library()
    lib.lib.mudnnGetStream.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.lib.mudnnGetStream.restype = ctypes.c_int
    stream = ctypes.c_void_p()
    lib.check(
        lib.lib.mudnnGetStream(_handle_pointer(handle), ctypes.byref(stream)),
        "mudnnGetStream",
    )
    return stream.value


def create_kernel_cache(data=None):
    cache = kernel_cache()
    if data is not None:
        cache.deserialize(data)
    return cache


def create_device_properties(device_or_data=None):
    if isinstance(device_or_data, (str, bytes, bytearray)):
        props = device_properties()
        props.deserialize(device_or_data)
        return props
    payload = {}
    if device_or_data is not None:
        payload["device"] = int(device_or_data)
    return device_properties(payload)


def graph_cache(key_fn, maxsize=256):
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            if key not in cache:
                if len(cache) >= maxsize:
                    cache.pop(next(iter(cache)))
                cache[key] = func(*args, **kwargs)
            return cache[key]

        return wrapper

    return decorator


def jit(heur_modes=HeurMode.A, **kwargs):
    modes = heur_modes if isinstance(heur_modes, list) else [heur_modes]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **inner_kwargs):
            graph_obj, tensors = func(*args, **inner_kwargs)
            if not getattr(graph_obj, "_plans_built", False):
                graph_obj.build(modes)
            return graph_obj, [tensor.get_uid() for tensor in tensors]

        return wrapper

    return decorator


@contextmanager
def graph(
    handle,
    name: str = "cudnn_graph",
    io_data_type: DataType = DataType.HALF,
    intermediate_data_type: DataType = DataType.FLOAT,
    compute_data_type: DataType = DataType.FLOAT,
):
    graph_obj = pygraph(
        handle=handle,
        name=name,
        io_data_type=io_data_type,
        intermediate_data_type=intermediate_data_type,
        compute_data_type=compute_data_type,
    )
    yield graph_obj, []


def causal_conv1d_forward(*args, **kwargs):
    from mudnn.causal_conv1d import causal_conv1d_forward as _forward

    return _forward(*args, **kwargs)


def causal_conv1d_backward(*args, **kwargs):
    from mudnn.causal_conv1d import causal_conv1d_backward as _backward

    return _backward(*args, **kwargs)


def causal_conv1d_nwh_forward(*args, **kwargs):
    from mudnn.causal_conv1d import causal_conv1d_nwh_forward as _forward

    return _forward(*args, **kwargs)


def causal_conv1d_nwh_backward(*args, **kwargs):
    from mudnn.causal_conv1d import causal_conv1d_nwh_backward as _backward

    return _backward(*args, **kwargs)


def b2b_causal_conv1d_forward(*args, **kwargs):
    from mudnn.b2b_causal_conv1d import b2b_causal_conv1d_forward as _forward

    return _forward(*args, **kwargs)


def b2b_causal_conv1d_backward(*args, **kwargs):
    from mudnn.b2b_causal_conv1d import b2b_causal_conv1d_backward as _backward

    return _backward(*args, **kwargs)


data_type = DataType
heur_mode = HeurMode
norm_forward_phase = NormFwdPhase
build_plan_policy = BuildPlanPolicy
tensor_reordering = TensorReordering
reshape_mode = ReshapeMode
scalar_type = ScalarType
behavior_note = BehaviorNote
numerical_note = NumericalNote
knob_type = KnobType
tensor = GraphTensor
pointwise_mode = PointwiseMode
attention_implementation = AttentionImplementation
reduction_mode = ReductionMode
convolution_mode = ConvolutionMode
resample_mode = ResampleMode
pooling_mode = ResampleMode
rng_distribution = RngDistribution
genstats_mode = GenStatsMode
bn_finalize_stats_mode = BnFinalizeStatsMode
signal_mode = SignalMode
norm_mode = NormMode
norm_fwd_phase = NormFwdPhase
moe_grouped_matmul_mode = MoeGroupedMatmulMode
diagonal_alignment = DiagonalAlignment

CUDNN_STATUS_SUCCESS = _backend.MUDNN_STATUS_SUCCESS
CUDNN_STATUS_NOT_INITIALIZED = _backend.MUDNN_STATUS_NOT_INITIALIZED
CUDNN_STATUS_NOT_SUPPORTED = _backend.MUDNN_STATUS_NOT_SUPPORTED
CUDNN_STATUS_BAD_PARAM = _backend.MUDNN_STATUS_BAD_PARAM
CUDNN_STATUS_INTERNAL_ERROR = _backend.MUDNN_STATUS_INTERNAL_ERROR
CUDNN_STATUS_ALLOC_FAILED = _backend.MUDNN_STATUS_ALLOC_FAILED

CUDNN_CONVOLUTION = ConvolutionMode.CONVOLUTION
CUDNN_CROSS_CORRELATION = ConvolutionMode.CROSS_CORRELATION
CUDNN_RESAMPLE_NEAREST = ResampleMode.NEAREST
CUDNN_RESAMPLE_BILINEAR = ResampleMode.BILINEAR
CUDNN_RESAMPLE_AVGPOOL = ResampleMode.AVGPOOL
CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING = ResampleMode.AVGPOOL_INCLUDE_PADDING
CUDNN_RESAMPLE_MAXPOOL = ResampleMode.MAXPOOL
CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING = ResampleMode.AVGPOOL_EXCLUDE_PADDING
CUDNN_RNG_DISTRIBUTION_BERNOULLI = RngDistribution.BERNOULLI
CUDNN_RNG_DISTRIBUTION_UNIFORM = RngDistribution.UNIFORM
CUDNN_RNG_DISTRIBUTION_NORMAL = RngDistribution.NORMAL
CUDNN_GENSTATS_SUM_SQSUM = GenStatsMode.SUM_SQSUM
CUDNN_BN_FINALIZE_STATISTICS_TRAINING = BnFinalizeStatsMode.TRAINING
CUDNN_BN_FINALIZE_STATISTICS_INFERENCE = BnFinalizeStatsMode.INFERENCE
CUDNN_SIGNAL_SET = SignalMode.SET
CUDNN_SIGNAL_WAIT = SignalMode.WAIT
CUDNN_LAYER_NORM = NormMode.LAYER_NORM
CUDNN_INSTANCE_NORM = NormMode.INSTANCE_NORM
CUDNN_BATCH_NORM = NormMode.BATCH_NORM
CUDNN_GROUP_NORM = NormMode.GROUP_NORM
CUDNN_RMS_NORM = NormMode.RMS_NORM
CUDNN_ADA_LAYER_NORM = NormMode.ADA_LAYER_NORM
CUDNN_NORM_FWD_NOT_SET = NormFwdPhase.NOT_SET
CUDNN_NORM_FWD_INFERENCE = NormFwdPhase.INFERENCE
CUDNN_NORM_FWD_TRAINING = NormFwdPhase.TRAINING

CUDNN_TYPE_HANDLE = _backend.MUDNN_TYPE_HANDLE
CUDNN_TYPE_DATA_TYPE = _backend.MUDNN_TYPE_DATA_TYPE
CUDNN_TYPE_BOOLEAN = _backend.MUDNN_TYPE_BOOLEAN
CUDNN_TYPE_INT64 = _backend.MUDNN_TYPE_INT64
CUDNN_TYPE_FLOAT = _backend.MUDNN_TYPE_FLOAT
CUDNN_TYPE_DOUBLE = _backend.MUDNN_TYPE_DOUBLE
CUDNN_TYPE_VOID_PTR = _backend.MUDNN_TYPE_VOID_PTR
CUDNN_TYPE_CONVOLUTION_MODE = _backend.MUDNN_TYPE_CONVOLUTION_MODE
CUDNN_TYPE_HEUR_MODE = _backend.MUDNN_TYPE_HEUR_MODE
CUDNN_TYPE_POINTWISE_MODE = _backend.MUDNN_TYPE_POINTWISE_MODE
CUDNN_TYPE_BACKEND_DESCRIPTOR = _backend.MUDNN_TYPE_BACKEND_DESCRIPTOR
CUDNN_TYPE_REDUCTION_OPERATOR_TYPE = _backend.MUDNN_TYPE_REDUCTION_OPERATOR_TYPE
CUDNN_TYPE_RESAMPLE_MODE = _backend.MUDNN_TYPE_RESAMPLE_MODE
CUDNN_TYPE_NORM_MODE = _backend.MUDNN_TYPE_NORM_MODE
CUDNN_TYPE_NORM_FWD_PHASE = _backend.MUDNN_TYPE_NORM_FWD_PHASE
CUDNN_TYPE_RNG_DISTRIBUTION = _backend.MUDNN_TYPE_RNG_DISTRIBUTION
CUDNN_TYPE_GENSTATS_MODE = _backend.MUDNN_TYPE_GENSTATS_MODE
CUDNN_TYPE_BN_FINALIZE_STATS_MODE = _backend.MUDNN_TYPE_BN_FINALIZE_STATS_MODE
CUDNN_TYPE_SIGNAL_MODE = _backend.MUDNN_TYPE_SIGNAL_MODE
CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE = _backend.MUDNN_TYPE_MOE_GROUPED_MATMUL_MODE
CUDNN_TYPE_RESHAPE_MODE = _backend.MUDNN_TYPE_RESHAPE_MODE

CUDNN_DATA_FLOAT = DataType.FLOAT
CUDNN_DATA_NOT_SET = DataType.NOT_SET
CUDNN_DATA_DOUBLE = DataType.DOUBLE
CUDNN_DATA_HALF = DataType.HALF
CUDNN_DATA_INT8 = DataType.INT8
CUDNN_DATA_INT8x4 = DataType.INT8x4
CUDNN_DATA_UINT8 = DataType.UINT8
CUDNN_DATA_UINT8x4 = DataType.UINT8x4
CUDNN_DATA_INT32 = DataType.INT32
CUDNN_DATA_INT8x32 = DataType.INT8x32
CUDNN_DATA_BFLOAT16 = DataType.BFLOAT16
CUDNN_DATA_INT64 = DataType.INT64
CUDNN_DATA_BOOLEAN = DataType.BOOLEAN
CUDNN_DATA_FP8_E4M3 = DataType.FP8_E4M3
CUDNN_DATA_FP8_E5M2 = DataType.FP8_E5M2
CUDNN_DATA_FAST_FLOAT_FOR_FP8 = DataType.FAST_FLOAT_FOR_FP8
CUDNN_DATA_FP8_E8M0 = DataType.FP8_E8M0
CUDNN_DATA_FP4_E2M1 = DataType.FP4_E2M1
CUDNN_DATA_INT4 = DataType.INT4
CUDNN_DATA_UINT4 = DataType.UINT4
CUDNN_DATA_UINT32 = DataType.UINT32
CUDNN_DATA_COMPLEX_FP32 = DataType.COMPLEX_FP32
CUDNN_DATA_COMPLEX_FP64 = DataType.COMPLEX_FP64

CUDNN_TENSOR_REORDERING_NONE = TensorReordering.NONE
CUDNN_TENSOR_REORDERING_INT8x32 = TensorReordering.INT8x32
CUDNN_TENSOR_REORDERING_F16x16 = TensorReordering.F16x16
CUDNN_TENSOR_REORDERING_F8_128x4 = TensorReordering.F8_128x4

CUDNN_NUMERICAL_NOTE_TENSOR_CORE = NumericalNote.TENSOR_CORE
CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS = NumericalNote.DOWN_CONVERT_INPUTS
CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION = (
    NumericalNote.REDUCED_PRECISION_REDUCTION
)
CUDNN_NUMERICAL_NOTE_FFT = NumericalNote.FFT
CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC = NumericalNote.NONDETERMINISTIC
CUDNN_NUMERICAL_NOTE_TYPE_COUNT = len(NumericalNote)
CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION = BehaviorNote.RUNTIME_COMPILATION
CUDNN_BEHAVIOR_NOTE_SUPPORTS_MUSA_GRAPH_NATIVE_API = (
    BehaviorNote.SUPPORTS_MUSA_GRAPH_NATIVE_API
)
CUDNN_BEHAVIOR_NOTE_MUBLASLT_DEPENDENCY = BehaviorNote.MUBLASLT_DEPENDENCY
CUDNN_BEHAVIOR_NOTE_SUPPORTS_CUDA_GRAPH_NATIVE_API = (
    BehaviorNote.SUPPORTS_CUDA_GRAPH_NATIVE_API
)
CUDNN_BEHAVIOR_NOTE_CUBLASLT_DEPENDENCY = BehaviorNote.CUBLASLT_DEPENDENCY
CUDNN_BEHAVIOR_NOTE_TYPE_COUNT = len(BehaviorNote)

CUDNN_POINTWISE_ADD = PointwiseMode.ADD
CUDNN_POINTWISE_MUL = PointwiseMode.MUL
CUDNN_POINTWISE_MIN = PointwiseMode.MIN
CUDNN_POINTWISE_MAX = PointwiseMode.MAX
CUDNN_POINTWISE_SQRT = PointwiseMode.SQRT
CUDNN_POINTWISE_ADD_SQUARE = PointwiseMode.ADD_SQUARE
CUDNN_POINTWISE_DIV = PointwiseMode.DIV
CUDNN_POINTWISE_MOD = PointwiseMode.MOD
CUDNN_POINTWISE_POW = PointwiseMode.POW
CUDNN_POINTWISE_SUB = PointwiseMode.SUB
CUDNN_POINTWISE_ABS = PointwiseMode.ABS
CUDNN_POINTWISE_CEIL = PointwiseMode.CEIL
CUDNN_POINTWISE_COS = PointwiseMode.COS
CUDNN_POINTWISE_EXP = PointwiseMode.EXP
CUDNN_POINTWISE_FLOOR = PointwiseMode.FLOOR
CUDNN_POINTWISE_LOG = PointwiseMode.LOG
CUDNN_POINTWISE_NEG = PointwiseMode.NEG
CUDNN_POINTWISE_SIN = PointwiseMode.SIN
CUDNN_POINTWISE_RSQRT = PointwiseMode.RSQRT
CUDNN_POINTWISE_TAN = PointwiseMode.TAN
CUDNN_POINTWISE_ERF = PointwiseMode.ERF
CUDNN_POINTWISE_IDENTITY = PointwiseMode.IDENTITY
CUDNN_POINTWISE_RECIPROCAL = PointwiseMode.RECIPROCAL
CUDNN_POINTWISE_RELU_FWD = PointwiseMode.RELU_FWD
CUDNN_POINTWISE_TANH_FWD = PointwiseMode.TANH_FWD
CUDNN_POINTWISE_SIGMOID_FWD = PointwiseMode.SIGMOID_FWD
CUDNN_POINTWISE_ELU_FWD = PointwiseMode.ELU_FWD
CUDNN_POINTWISE_GELU_FWD = PointwiseMode.GELU_FWD
CUDNN_POINTWISE_SOFTPLUS_FWD = PointwiseMode.SOFTPLUS_FWD
CUDNN_POINTWISE_SWISH_FWD = PointwiseMode.SWISH_FWD
CUDNN_POINTWISE_GELU_APPROX_TANH_FWD = PointwiseMode.GELU_APPROX_TANH_FWD
CUDNN_POINTWISE_RELU_BWD = PointwiseMode.RELU_BWD
CUDNN_POINTWISE_TANH_BWD = PointwiseMode.TANH_BWD
CUDNN_POINTWISE_SIGMOID_BWD = PointwiseMode.SIGMOID_BWD
CUDNN_POINTWISE_ELU_BWD = PointwiseMode.ELU_BWD
CUDNN_POINTWISE_GELU_BWD = PointwiseMode.GELU_BWD
CUDNN_POINTWISE_SOFTPLUS_BWD = PointwiseMode.SOFTPLUS_BWD
CUDNN_POINTWISE_SWISH_BWD = PointwiseMode.SWISH_BWD
CUDNN_POINTWISE_GELU_APPROX_TANH_BWD = PointwiseMode.GELU_APPROX_TANH_BWD
CUDNN_POINTWISE_CMP_EQ = PointwiseMode.CMP_EQ
CUDNN_POINTWISE_CMP_NEQ = PointwiseMode.CMP_NEQ
CUDNN_POINTWISE_CMP_GT = PointwiseMode.CMP_GT
CUDNN_POINTWISE_CMP_GE = PointwiseMode.CMP_GE
CUDNN_POINTWISE_CMP_LT = PointwiseMode.CMP_LT
CUDNN_POINTWISE_CMP_LE = PointwiseMode.CMP_LE
CUDNN_POINTWISE_LOGICAL_AND = PointwiseMode.LOGICAL_AND
CUDNN_POINTWISE_LOGICAL_OR = PointwiseMode.LOGICAL_OR
CUDNN_POINTWISE_LOGICAL_NOT = PointwiseMode.LOGICAL_NOT
CUDNN_POINTWISE_GEN_INDEX = PointwiseMode.GEN_INDEX
CUDNN_POINTWISE_BINARY_SELECT = PointwiseMode.BINARY_SELECT

CUDNN_REDUCE_TENSOR_NOT_SET = ReductionMode.NOT_SET
CUDNN_REDUCE_TENSOR_ADD = ReductionMode.ADD
CUDNN_REDUCE_TENSOR_MUL = ReductionMode.MUL
CUDNN_REDUCE_TENSOR_MIN = ReductionMode.MIN
CUDNN_REDUCE_TENSOR_MAX = ReductionMode.MAX
CUDNN_REDUCE_TENSOR_AMAX = ReductionMode.AMAX
CUDNN_REDUCE_TENSOR_AVG = ReductionMode.AVG
CUDNN_REDUCE_TENSOR_NORM1 = ReductionMode.NORM1
CUDNN_REDUCE_TENSOR_NORM2 = ReductionMode.NORM2
CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = ReductionMode.MUL_NO_ZEROS

CUDNN_RESHAPE_VIEW_ONLY = ReshapeMode.VIEW_ONLY
CUDNN_RESHAPE_LOGICAL = ReshapeMode.LOGICAL

CUDNN_HEUR_MODE_INSTANT = HeurMode.INSTANT
CUDNN_HEUR_MODE_B = HeurMode.B
CUDNN_HEUR_MODE_FALLBACK = HeurMode.FALLBACK
CUDNN_HEUR_MODE_A = HeurMode.A
CUDNN_HEUR_MODE_OPENSOURCE = HeurMode.OPENSOURCE

CUDNN_BACKEND_POINTWISE_DESCRIPTOR = _backend.MUDNN_BACKEND_POINTWISE_DESCRIPTOR
CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR = _backend.MUDNN_BACKEND_CONVOLUTION_DESCRIPTOR
CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR
)
CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR
)
CUDNN_BACKEND_TENSOR_DESCRIPTOR = _backend.MUDNN_BACKEND_TENSOR_DESCRIPTOR
CUDNN_BACKEND_ENGINE_DESCRIPTOR = _backend.MUDNN_BACKEND_ENGINE_DESCRIPTOR
CUDNN_BACKEND_ENGINECFG_DESCRIPTOR = _backend.MUDNN_BACKEND_ENGINECFG_DESCRIPTOR
CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR = _backend.MUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR
CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR
)
CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR = _backend.MUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR
CUDNN_BACKEND_MATMUL_DESCRIPTOR = _backend.MUDNN_BACKEND_MATMUL_DESCRIPTOR
CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR
)
CUDNN_BACKEND_REDUCTION_DESCRIPTOR = _backend.MUDNN_BACKEND_REDUCTION_DESCRIPTOR
CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR
)
CUDNN_BACKEND_RESAMPLE_DESCRIPTOR = _backend.MUDNN_BACKEND_RESAMPLE_DESCRIPTOR
CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR
)
CUDNN_BACKEND_RNG_DESCRIPTOR = _backend.MUDNN_BACKEND_RNG_DESCRIPTOR
CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR = _backend.MUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR
CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR
)
CUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR = (
    _backend.MUDNN_BACKEND_OPERATION_ROPE_BWD_DESCRIPTOR
)

CUDNN_MOE_GROUPED_MATMUL_MODE_NONE = MoeGroupedMatmulMode.NONE
CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER = MoeGroupedMatmulMode.GATHER
CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER = MoeGroupedMatmulMode.SCATTER

CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT = _backend.MUDNN_ATTR_TENSOR_BYTE_ALIGNMENT
CUDNN_ATTR_TENSOR_DATA_TYPE = _backend.MUDNN_ATTR_TENSOR_DATA_TYPE
CUDNN_ATTR_TENSOR_DIMENSIONS = _backend.MUDNN_ATTR_TENSOR_DIMENSIONS
CUDNN_ATTR_TENSOR_STRIDES = _backend.MUDNN_ATTR_TENSOR_STRIDES
CUDNN_ATTR_TENSOR_UNIQUE_ID = _backend.MUDNN_ATTR_TENSOR_UNIQUE_ID
CUDNN_ATTR_TENSOR_IS_VIRTUAL = _backend.MUDNN_ATTR_TENSOR_IS_VIRTUAL
CUDNN_ATTR_TENSOR_IS_BY_VALUE = _backend.MUDNN_ATTR_TENSOR_IS_BY_VALUE
CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC = _backend.MUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC
CUDNN_ATTR_TENSOR_CONSTANT_VALUE = _backend.MUDNN_ATTR_TENSOR_CONSTANT_VALUE
CUDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER = (
    _backend.MUDNN_ATTR_TENSOR_RAGGED_OFFSET_MULTIPLIER
)
CUDNN_ATTR_POINTWISE_MODE = _backend.MUDNN_ATTR_POINTWISE_MODE
CUDNN_ATTR_POINTWISE_MATH_PREC = _backend.MUDNN_ATTR_POINTWISE_MATH_PREC
CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP = _backend.MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP
CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP = _backend.MUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP
CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = (
    _backend.MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE
)
CUDNN_ATTR_POINTWISE_ELU_ALPHA = _backend.MUDNN_ATTR_POINTWISE_ELU_ALPHA
CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA = _backend.MUDNN_ATTR_POINTWISE_SOFTPLUS_BETA
CUDNN_ATTR_POINTWISE_SWISH_BETA = _backend.MUDNN_ATTR_POINTWISE_SWISH_BETA
CUDNN_ATTR_POINTWISE_AXIS = _backend.MUDNN_ATTR_POINTWISE_AXIS
CUDNN_ATTR_CONVOLUTION_COMP_TYPE = _backend.MUDNN_ATTR_CONVOLUTION_COMP_TYPE
CUDNN_ATTR_CONVOLUTION_CONV_MODE = _backend.MUDNN_ATTR_CONVOLUTION_CONV_MODE
CUDNN_ATTR_CONVOLUTION_DILATIONS = _backend.MUDNN_ATTR_CONVOLUTION_DILATIONS
CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = _backend.MUDNN_ATTR_CONVOLUTION_FILTER_STRIDES
CUDNN_ATTR_CONVOLUTION_POST_PADDINGS = _backend.MUDNN_ATTR_CONVOLUTION_POST_PADDINGS
CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS = _backend.MUDNN_ATTR_CONVOLUTION_PRE_PADDINGS
CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS = _backend.MUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS
CUDNN_ATTR_CONVOLUTION_GROUP_COUNT = _backend.MUDNN_ATTR_CONVOLUTION_GROUP_COUNT
CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA
)
CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA
)
CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC
)
CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W
)
CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X
)
CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X
)
CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = (
    _backend.MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY
)
CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = (
    _backend.MUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR
)
CUDNN_ATTR_OPERATION_POINTWISE_XDESC = _backend.MUDNN_ATTR_OPERATION_POINTWISE_XDESC
CUDNN_ATTR_OPERATION_POINTWISE_BDESC = _backend.MUDNN_ATTR_OPERATION_POINTWISE_BDESC
CUDNN_ATTR_OPERATION_POINTWISE_YDESC = _backend.MUDNN_ATTR_OPERATION_POINTWISE_YDESC
CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1 = _backend.MUDNN_ATTR_OPERATION_POINTWISE_ALPHA1
CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2 = _backend.MUDNN_ATTR_OPERATION_POINTWISE_ALPHA2
CUDNN_ATTR_OPERATION_POINTWISE_DXDESC = _backend.MUDNN_ATTR_OPERATION_POINTWISE_DXDESC
CUDNN_ATTR_OPERATION_POINTWISE_DYDESC = _backend.MUDNN_ATTR_OPERATION_POINTWISE_DYDESC
CUDNN_ATTR_OPERATION_POINTWISE_TDESC = _backend.MUDNN_ATTR_OPERATION_POINTWISE_TDESC
CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_QDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_QDESC
)
CUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_KVDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_CU_SEQ_LEN_KVDESC
)
CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_KVDESC = (
    _backend.MUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_KVDESC
)
CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_QDESC = (
    _backend.MUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_CU_SEQ_LEN_QDESC
)
CUDNN_ATTR_OPERATION_ROPE_FWD_XDESC = _backend.MUDNN_ATTR_OPERATION_ROPE_FWD_XDESC
CUDNN_ATTR_OPERATION_ROPE_FWD_YDESC = _backend.MUDNN_ATTR_OPERATION_ROPE_FWD_YDESC
CUDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC = (
    _backend.MUDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC
)
CUDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE = (
    _backend.MUDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE
)
CUDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM = _backend.MUDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM
CUDNN_ATTR_OPERATION_ROPE_BWD_DYDESC = _backend.MUDNN_ATTR_OPERATION_ROPE_BWD_DYDESC
CUDNN_ATTR_OPERATION_ROPE_BWD_DXDESC = _backend.MUDNN_ATTR_OPERATION_ROPE_BWD_DXDESC
CUDNN_ATTR_OPERATION_ROPE_BWD_FREQSDESC = (
    _backend.MUDNN_ATTR_OPERATION_ROPE_BWD_FREQSDESC
)
CUDNN_ATTR_OPERATION_ROPE_BWD_OUTPUT_SCALE = (
    _backend.MUDNN_ATTR_OPERATION_ROPE_BWD_OUTPUT_SCALE
)
CUDNN_ATTR_OPERATION_ROPE_BWD_ROPE_DIM = _backend.MUDNN_ATTR_OPERATION_ROPE_BWD_ROPE_DIM
CUDNN_ATTR_OPERATIONGRAPH_HANDLE = _backend.MUDNN_ATTR_OPERATIONGRAPH_HANDLE
CUDNN_ATTR_OPERATIONGRAPH_OPS = _backend.MUDNN_ATTR_OPERATIONGRAPH_OPS
CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = (
    _backend.MUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT
)
CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED = (
    _backend.MUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED
)
CUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY = (
    _backend.MUDNN_ATTR_OPERATIONGRAPH_IS_SAME_TOPOLOGY
)
CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED = (
    _backend.MUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED
)
CUDNN_ATTR_OPERATIONGRAPH_MODE = _backend.MUDNN_ATTR_OPERATIONGRAPH_MODE
CUDNN_ATTR_ENGINEHEUR_MODE = _backend.MUDNN_ATTR_ENGINEHEUR_MODE
CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = _backend.MUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH
CUDNN_ATTR_ENGINEHEUR_RESULTS = _backend.MUDNN_ATTR_ENGINEHEUR_RESULTS
CUDNN_ATTR_ENGINECFG_ENGINE = _backend.MUDNN_ATTR_ENGINECFG_ENGINE
CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE = _backend.MUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE
CUDNN_ATTR_EXECUTION_PLAN_HANDLE = _backend.MUDNN_ATTR_EXECUTION_PLAN_HANDLE
CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG = (
    _backend.MUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG
)
CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = (
    _backend.MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE
)
CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS = _backend.MUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS
CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = _backend.MUDNN_ATTR_VARIANT_PACK_DATA_POINTERS
CUDNN_ATTR_VARIANT_PACK_WORKSPACE = _backend.MUDNN_ATTR_VARIANT_PACK_WORKSPACE
CUDNN_ATTR_ENGINE_OPERATION_GRAPH = _backend.MUDNN_ATTR_ENGINE_OPERATION_GRAPH
CUDNN_ATTR_ENGINE_GLOBAL_INDEX = _backend.MUDNN_ATTR_ENGINE_GLOBAL_INDEX
CUDNN_ATTR_MATMUL_COMP_TYPE = _backend.MUDNN_ATTR_MATMUL_COMP_TYPE
CUDNN_ATTR_MATMUL_PADDING_VALUE = _backend.MUDNN_ATTR_MATMUL_PADDING_VALUE
CUDNN_ATTR_OPERATION_MATMUL_ADESC = _backend.MUDNN_ATTR_OPERATION_MATMUL_ADESC
CUDNN_ATTR_OPERATION_MATMUL_BDESC = _backend.MUDNN_ATTR_OPERATION_MATMUL_BDESC
CUDNN_ATTR_OPERATION_MATMUL_CDESC = _backend.MUDNN_ATTR_OPERATION_MATMUL_CDESC
CUDNN_ATTR_OPERATION_MATMUL_DESC = _backend.MUDNN_ATTR_OPERATION_MATMUL_DESC
CUDNN_ATTR_REDUCTION_OPERATOR = _backend.MUDNN_ATTR_REDUCTION_OPERATOR
CUDNN_ATTR_REDUCTION_COMP_TYPE = _backend.MUDNN_ATTR_REDUCTION_COMP_TYPE
CUDNN_ATTR_REDUCTION_IS_DETERMINISTIC = (
    _backend.MUDNN_ATTR_REDUCTION_IS_DETERMINISTIC
)
CUDNN_ATTR_OPERATION_REDUCTION_XDESC = _backend.MUDNN_ATTR_OPERATION_REDUCTION_XDESC
CUDNN_ATTR_OPERATION_REDUCTION_YDESC = _backend.MUDNN_ATTR_OPERATION_REDUCTION_YDESC
CUDNN_ATTR_OPERATION_REDUCTION_DESC = _backend.MUDNN_ATTR_OPERATION_REDUCTION_DESC
CUDNN_ATTR_OPERATION_REDUCTION_GROUP_OFFSET_DESC = (
    _backend.MUDNN_ATTR_OPERATION_REDUCTION_GROUP_OFFSET_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC
)
CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS = (
    _backend.MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS
)
CUDNN_ATTR_OPERATION_GENSTATS_MODE = _backend.MUDNN_ATTR_OPERATION_GENSTATS_MODE
CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC = (
    _backend.MUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC
)
CUDNN_ATTR_OPERATION_GENSTATS_XDESC = _backend.MUDNN_ATTR_OPERATION_GENSTATS_XDESC
CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC = _backend.MUDNN_ATTR_OPERATION_GENSTATS_SUMDESC
CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC = (
    _backend.MUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC
)
CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC
)
CUDNN_ATTR_RESAMPLE_MODE = _backend.MUDNN_ATTR_RESAMPLE_MODE
CUDNN_ATTR_RESAMPLE_COMP_TYPE = _backend.MUDNN_ATTR_RESAMPLE_COMP_TYPE
CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS = _backend.MUDNN_ATTR_RESAMPLE_SPATIAL_DIMS
CUDNN_ATTR_RESAMPLE_POST_PADDINGS = _backend.MUDNN_ATTR_RESAMPLE_POST_PADDINGS
CUDNN_ATTR_RESAMPLE_PRE_PADDINGS = _backend.MUDNN_ATTR_RESAMPLE_PRE_PADDINGS
CUDNN_ATTR_RESAMPLE_STRIDES = _backend.MUDNN_ATTR_RESAMPLE_STRIDES
CUDNN_ATTR_RESAMPLE_WINDOW_DIMS = _backend.MUDNN_ATTR_RESAMPLE_WINDOW_DIMS
CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC
)
CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC
)
CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC
)
CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA
)
CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA = _backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA
CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC = _backend.MUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC
CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC
)
CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC
)
CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC
)
CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA
)
CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA = _backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA
CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC = _backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC
CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC
)
CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC = (
    _backend.MUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC
)
CUDNN_ATTR_OPERATION_CONCAT_AXIS = _backend.MUDNN_ATTR_OPERATION_CONCAT_AXIS
CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS = (
    _backend.MUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS
)
CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX = (
    _backend.MUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX
)
CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC = (
    _backend.MUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_MODE = _backend.MUDNN_ATTR_OPERATION_NORM_FWD_MODE
CUDNN_ATTR_OPERATION_NORM_FWD_PHASE = _backend.MUDNN_ATTR_OPERATION_NORM_FWD_PHASE
CUDNN_ATTR_OPERATION_NORM_FWD_XDESC = _backend.MUDNN_ATTR_OPERATION_NORM_FWD_XDESC
CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC
)
CUDNN_ATTR_OPERATION_NORM_FWD_YDESC = _backend.MUDNN_ATTR_OPERATION_NORM_FWD_YDESC
CUDNN_ATTR_OPERATION_NORM_BWD_MODE = _backend.MUDNN_ATTR_OPERATION_NORM_BWD_MODE
CUDNN_ATTR_OPERATION_NORM_BWD_XDESC = _backend.MUDNN_ATTR_OPERATION_NORM_BWD_XDESC
CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC
)
CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC
)
CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC = _backend.MUDNN_ATTR_OPERATION_NORM_BWD_DYDESC
CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC
)
CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC
)
CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC
)
CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC = (
    _backend.MUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC
)
CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC = _backend.MUDNN_ATTR_OPERATION_NORM_BWD_DXDESC
CUDNN_ATTR_OPERATION_RESHAPE_XDESC = _backend.MUDNN_ATTR_OPERATION_RESHAPE_XDESC
CUDNN_ATTR_OPERATION_RESHAPE_YDESC = _backend.MUDNN_ATTR_OPERATION_RESHAPE_YDESC
CUDNN_ATTR_OPERATION_RESHAPE_MODE = _backend.MUDNN_ATTR_OPERATION_RESHAPE_MODE
CUDNN_ATTR_OPERATION_SLICE_XDESC = _backend.MUDNN_ATTR_OPERATION_SLICE_XDESC
CUDNN_ATTR_OPERATION_SLICE_YDESC = _backend.MUDNN_ATTR_OPERATION_SLICE_YDESC
CUDNN_ATTR_OPERATION_SLICE_START_INDICES = (
    _backend.MUDNN_ATTR_OPERATION_SLICE_START_INDICES
)
CUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES = (
    _backend.MUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES
)
CUDNN_ATTR_OPERATION_SLICE_STRIDES = _backend.MUDNN_ATTR_OPERATION_SLICE_STRIDES
CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC = _backend.MUDNN_ATTR_OPERATION_TRANSPOSE_XDESC
CUDNN_ATTR_OPERATION_TRANSPOSE_YDESC = _backend.MUDNN_ATTR_OPERATION_TRANSPOSE_YDESC
CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION = (
    _backend.MUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION
)
CUDNN_ATTR_RNG_DISTRIBUTION = _backend.MUDNN_ATTR_RNG_DISTRIBUTION
CUDNN_ATTR_RNG_NORMAL_DIST_MEAN = _backend.MUDNN_ATTR_RNG_NORMAL_DIST_MEAN
CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = (
    _backend.MUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION
)
CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM = _backend.MUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM
CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM = _backend.MUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM
CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY = (
    _backend.MUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY
)
CUDNN_ATTR_OPERATION_RNG_YDESC = _backend.MUDNN_ATTR_OPERATION_RNG_YDESC
CUDNN_ATTR_OPERATION_RNG_SEED = _backend.MUDNN_ATTR_OPERATION_RNG_SEED
CUDNN_ATTR_OPERATION_RNG_DESC = _backend.MUDNN_ATTR_OPERATION_RNG_DESC
CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC = _backend.MUDNN_ATTR_OPERATION_RNG_OFFSET_DESC
CUDNN_ATTR_OPERATION_SIGNAL_MODE = _backend.MUDNN_ATTR_OPERATION_SIGNAL_MODE
CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC = _backend.MUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC
CUDNN_ATTR_OPERATION_SIGNAL_VALUE = _backend.MUDNN_ATTR_OPERATION_SIGNAL_VALUE
CUDNN_ATTR_OPERATION_SIGNAL_XDESC = _backend.MUDNN_ATTR_OPERATION_SIGNAL_XDESC
CUDNN_ATTR_OPERATION_SIGNAL_YDESC = _backend.MUDNN_ATTR_OPERATION_SIGNAL_YDESC
CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC = (
    _backend.MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC
)
CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC = (
    _backend.MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC
)
CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC
)
CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE
)
CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE = (
    _backend.MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE
)
CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC = _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_QDESC
CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC = _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_KDESC
CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC = _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_VDESC
CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC = _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_ODESC
CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC
)
CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC
)
CUDNN_ATTR_OPERATION_SDPA_FWD_BIAS_DESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_BIAS_DESC
)
CUDNN_ATTR_OPERATION_SDPA_FWD_USE_PADDING_MASK = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_USE_PADDING_MASK
)
CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC
)
CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC
)
CUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK
)
CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC = _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_QDESC
CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC = _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_KDESC
CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC = _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_VDESC
CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC = _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_ODESC
CUDNN_ATTR_OPERATION_SDPA_BWD_DODESC = _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC
CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC
)
CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC
)
CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC
)
CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC
)
CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC = _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC
CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC = _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC
CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC = _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC
CUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK = (
    _backend.MUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC
)
CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC = (
    _backend.MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC
)

__all__ = [
    "__version__",
    "DataType",
    "HeurMode",
    "PointwiseMode",
    "AttentionImplementation",
    "ReductionMode",
    "ConvolutionMode",
    "ResampleMode",
    "RngDistribution",
    "GenStatsMode",
    "BnFinalizeStatsMode",
    "SignalMode",
    "NormMode",
    "NormFwdPhase",
    "MoeGroupedMatmulMode",
    "DiagonalAlignment",
    "Graph",
    "GraphNode",
    "GraphTensor",
    "cudnnGraphNotSupportedError",
    "BuildPlanPolicy",
    "TensorReordering",
    "ReshapeMode",
    "ScalarType",
    "BehaviorNote",
    "NumericalNote",
    "KnobType",
    "knob",
    "Tensor",
    "TensorBuilder",
    "PointWiseDesc",
    "PointWiseDescBuilder",
    "PointwiseDesc",
    "PointwiseDescBuilder",
    "MatMulDesc",
    "MatMulDescBuilder",
    "ConvDesc",
    "ConvDescBuilder",
    "ConvolutionDesc",
    "ConvolutionDescBuilder",
    "ResampleDesc",
    "ResampleDescBuilder",
    "PoolingDesc",
    "PoolingDescBuilder",
    "RngDesc",
    "RngDescBuilder",
    "ReductionDesc",
    "ReductionDescBuilder",
    "Operation",
    "OperationBuilder",
    "OperationGraph",
    "OperationGraphBuilder",
    "Engine",
    "EngineBuilder",
    "EngineConfig",
    "EngineConfigBuilder",
    "EngineHeuristics",
    "EngineHeuristicsBuilder",
    "ExecutionPlan",
    "ExecutionPlanBuilder",
    "VariantPack",
    "VariantPackBuilder",
    "pygraph",
    "_get_data_ptr",
    "BackendError",
    "backend_version",
    "backend_version_string",
    "get_last_error_string",
    "create_handle",
    "kernel_cache",
    "create_kernel_cache",
    "device_properties",
    "create_device_properties",
    "destroy_handle",
    "set_stream",
    "get_stream",
    "graph_cache",
    "jit",
    "graph",
    "causal_conv1d_forward",
    "causal_conv1d_backward",
    "causal_conv1d_nwh_forward",
    "causal_conv1d_nwh_backward",
    "b2b_causal_conv1d_forward",
    "b2b_causal_conv1d_backward",
    "data_type",
    "heur_mode",
    "norm_forward_phase",
    "build_plan_policy",
    "tensor_reordering",
    "reshape_mode",
    "scalar_type",
    "behavior_note",
    "numerical_note",
    "knob_type",
    "tensor",
    "pointwise_mode",
    "attention_implementation",
    "reduction_mode",
    "convolution_mode",
    "resample_mode",
    "pooling_mode",
    "rng_distribution",
    "genstats_mode",
    "bn_finalize_stats_mode",
    "signal_mode",
    "norm_mode",
    "norm_fwd_phase",
    "moe_grouped_matmul_mode",
    "diagonal_alignment",
]

# Keep the Python backend-C surface synchronized with include/cudnn_frontend.h.
# Constants already modeled by _backend are exported mechanically; the
# remaining values are ABI constants from muDNN 3.3 / cuDNN 9.23.
for _mudnn_name, _mudnn_value in vars(_backend).items():
    if _mudnn_name.startswith("MUDNN_"):
        globals().setdefault(
            "CUDNN_" + _mudnn_name[len("MUDNN_") :], _mudnn_value
        )

_CUDNN_923_CONSTANTS = {
    "CUDNN_ATTR_DEVICEPROP_DEVICE_ID": 2700,
    "CUDNN_ATTR_DEVICEPROP_HANDLE": 2701,
    "CUDNN_ATTR_DEVICEPROP_JSON_REPRESENTATION": 2702,
    "CUDNN_ATTR_ENGINEHEUR_DEVICEPROP": 204,
    "CUDNN_ATTR_ENGINEHEUR_SM_COUNT_TARGET": 203,
    "CUDNN_ATTR_ENGINE_DEVICEPROP": 1307,
    "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION": 405,
    "CUDNN_ATTR_EXECUTION_PLAN_KERNEL_CACHE": 406,
    "CUDNN_ATTR_KERNEL_CACHE_JSON_REPRESENTATION": 2402,
    "CUDNN_ATTR_KERNEL_CACHE_OPERATION_GRAPH": 2400,
    "CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE": 600,
    "CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE": 601,
    "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC": 3005,
    "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE": 3007,
    "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC": 3003,
    "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC": 3001,
    "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC": 3002,
    "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SHIFT_RIGHT_BOUND_DESC": 3004,
    "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC": 3000,
    "CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC": 3006,
    "CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC": 1527,
    "CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC": 1525,
    "CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC": 1526,
    "CUDNN_ATTR_OPERATION_SOFTMAX_MAX_DESC": 3103,
    "CUDNN_ATTR_OPERATION_SOFTMAX_SINK_DESC": 3105,
    "CUDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC": 3102,
    "CUDNN_ATTR_OPERATION_SOFTMAX_SUM_EXP_DESC": 3104,
    "CUDNN_ATTR_OPERATION_SOFTMAX_XDESC": 3100,
    "CUDNN_ATTR_OPERATION_SOFTMAX_YDESC": 3101,
    "CUDNN_ATTR_POINTWISE_NAN_PROPAGATION": 2,
    "CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION": 1707,
    "CUDNN_ATTR_RESAMPLE_PADDING_MODE": 1708,
    "CUDNN_ATTR_VARIANT_PACK_OVERRIDE_SHAPES": 1011,
    "CUDNN_ATTR_VARIANT_PACK_OVERRIDE_STRIDES": 1012,
    "CUDNN_ATTR_VARIANT_PACK_OVERRIDE_UNIQUE_IDS": 1010,
    "CUDNN_BACKEND_DEVICEPROP_DESCRIPTOR": 38,
    "CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR": 6,
    "CUDNN_BACKEND_KERNEL_CACHE_DESCRIPTOR": 34,
    "CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR": 7,
    "CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR": 8,
    "CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR": 9,
    "CUDNN_DEFAULT_REORDER": 0,
    "CUDNN_DIM_MAX": 10,
    "CUDNN_EDGE_VAL_PAD": 2,
    "CUDNN_HEUR_MODES_COUNT": 4,
    "CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD": 31,
    "CUDNN_KNOB_TYPE_BLOCK_SIZE": 29,
    "CUDNN_KNOB_TYPE_COUNTS": 46,
    "CUDNN_KNOB_TYPE_CTA_COUNT": 37,
    "CUDNN_KNOB_TYPE_EDGE": 4,
    "CUDNN_KNOB_TYPE_IDX_MODE": 18,
    "CUDNN_KNOB_TYPE_KERNEL_CFG": 24,
    "CUDNN_KNOB_TYPE_LOAD_SIZE": 36,
    "CUDNN_KNOB_TYPE_MULTIPLY": 11,
    "CUDNN_KNOB_TYPE_OCCUPANCY": 30,
    "CUDNN_KNOB_TYPE_REDUCTION_MODE": 15,
    "CUDNN_KNOB_TYPE_SPECFILT": 23,
    "CUDNN_KNOB_TYPE_SPLIT_COLS": 33,
    "CUDNN_KNOB_TYPE_SPLIT_K_BUF": 12,
    "CUDNN_KNOB_TYPE_SPLIT_K_SLC": 17,
    "CUDNN_KNOB_TYPE_SPLIT_P_SLC": 39,
    "CUDNN_KNOB_TYPE_STAGES": 14,
    "CUDNN_KNOB_TYPE_STREAM_K": 38,
    "CUDNN_KNOB_TYPE_SWIZZLE": 1,
    "CUDNN_KNOB_TYPE_TILEK": 13,
    "CUDNN_KNOB_TYPE_TILE_CGA_M": 27,
    "CUDNN_KNOB_TYPE_TILE_CGA_N": 28,
    "CUDNN_KNOB_TYPE_TILE_COLS": 35,
    "CUDNN_KNOB_TYPE_TILE_M": 40,
    "CUDNN_KNOB_TYPE_TILE_N": 41,
    "CUDNN_KNOB_TYPE_TILE_ROWS": 34,
    "CUDNN_KNOB_TYPE_TILE_SIZE": 2,
    "CUDNN_KNOB_TYPE_WARP_SPEC_CFG": 42,
    "CUDNN_KNOB_TYPE_WORKSPACE": 25,
    "CUDNN_NEG_INF_PAD": 1,
    "CUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP": 9,
    "CUDNN_NUMERICAL_NOTE_WINOGRAD": 5,
    "CUDNN_PROPAGATE_NAN": 1,
    "CUDNN_STATUS_INVALID_VALUE": 2001,
    "CUDNN_STATUS_VERSION_MISMATCH": 1002,
    "CUDNN_TENSOR_NCHW_VECT_C": 2,
    "CUDNN_TYPE_CHAR": 24,
    "CUDNN_TYPE_FRACTION": 26,
    "CUDNN_TYPE_INT32": 23,
    "CUDNN_TYPE_NAN_PROPOGATION": 10,
    "CUDNN_TYPE_PADDING_MODE": 22,
    "CUDNN_ZERO_PAD": 0,
}
globals().update(_CUDNN_923_CONSTANTS)
__all__.extend(
    sorted(name for name in globals() if name.startswith("CUDNN_") and name not in __all__)
)

__all__ += [name for name in globals() if name.startswith("CUDNN_")]
