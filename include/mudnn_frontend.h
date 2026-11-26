#pragma once

#include <mudnn.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef CUDNN_VERSION
#define CUDNN_VERSION 92300
#endif

#if defined(__has_include) && !defined(MUDNN_FRONTEND_FORCE_FALLBACK_JSON)
#if __has_include(<mudnn_frontend/thirdparty/nlohmann/json.hpp>)
#include <mudnn_frontend/thirdparty/nlohmann/json.hpp>
#define MUDNN_FRONTEND_HAS_NLOHMANN_JSON 1
using json = nlohmann::json;
#elif __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#define MUDNN_FRONTEND_HAS_NLOHMANN_JSON 1
using json = nlohmann::json;
#endif
#endif

#ifndef MUDNN_FRONTEND_HAS_NLOHMANN_JSON
namespace mudnn_frontend_detail {
class fallback_json {
 public:
  fallback_json() = default;
  fallback_json(const char* value) : payload_(value ? value : "") {}
  fallback_json(const std::string& value) : payload_(value) {}

  template <typename T>
  fallback_json(const T&) {}

  template <typename T>
  T get() const {
    return T{};
  }

  template <typename T>
  T value(const std::string&, const T& default_value) const {
    return default_value;
  }

  const fallback_json& at(const std::string&) const { return *this; }
  operator std::string() const { return payload_; }

  friend bool operator==(const fallback_json& lhs, const fallback_json& rhs) {
    return lhs.payload_ == rhs.payload_;
  }

  friend bool operator!=(const fallback_json& lhs, const fallback_json& rhs) {
    return !(lhs == rhs);
  }

  template <typename T>
  friend bool operator==(const fallback_json&, const T&) {
    return true;
  }

  template <typename T>
  friend bool operator==(const T&, const fallback_json&) {
    return true;
  }

  template <typename T>
  friend bool operator!=(const fallback_json& lhs, const T& rhs) {
    return !(lhs == rhs);
  }

  template <typename T>
  friend bool operator!=(const T& lhs, const fallback_json& rhs) {
    return !(lhs == rhs);
  }

  friend std::ostream& operator<<(std::ostream& os, const fallback_json& j) {
    os << j.payload_;
    return os;
  }

 private:
  std::string payload_;
};
}  // namespace mudnn_frontend_detail
using json = mudnn_frontend_detail::fallback_json;
#endif

namespace mudnn_frontend {

namespace detail {
inline size_t get_backend_version() { return CUDNN_VERSION; }

inline std::string convert_version_to_str(size_t version) {
  size_t major = 0;
  size_t minor = 0;
  size_t patch = 0;
  if (version >= 90000) {
    major = version / 10000;
    minor = (version % 10000) / 100;
    patch = version % 100;
  } else {
    major = version / 1000;
    minor = (version % 1000) / 100;
    patch = version % 100;
  }
  return std::to_string(major) + "." + std::to_string(minor) + "." +
      std::to_string(patch);
}
}  // namespace detail

enum class DiagonalAlignment { TOP_LEFT = 0, BOTTOM_RIGHT = 1 };
using DiagonalAlignment_t = DiagonalAlignment;
enum class PaddingMode_t {
  NOT_SET = -1,
  ZERO_PAD = MUDNN_ZERO_PAD,
  NEG_INF_PAD = MUDNN_NEG_INF_PAD,
  EDGE_VAL_PAD = MUDNN_EDGE_VAL_PAD,
};
enum class AttentionImplementation_t {
  AUTO = 0,
  COMPOSITE = 1,
  UNIFIED = 2,
};
enum class ReshapeMode_t {
  VIEW_ONLY = MUDNN_RESHAPE_VIEW_ONLY,
  LOGICAL = MUDNN_RESHAPE_LOGICAL,
};
enum class BehaviorNote_t {
  RUNTIME_COMPILATION = MUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION,
  REQUIRES_FILTER_INT8x32_REORDER =
      MUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER,
  REQUIRES_BIAS_INT8x32_REORDER =
      MUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER,
  SUPPORTS_CUDA_GRAPH_NATIVE_API =
      MUDNN_BEHAVIOR_NOTE_SUPPORTS_MUSA_GRAPH_NATIVE_API,
  CUBLASLT_DEPENDENCY = MUDNN_BEHAVIOR_NOTE_MUBLASLT_DEPENDENCY,
};
enum class NumericalNote_t {
  TENSOR_CORE = MUDNN_NUMERICAL_NOTE_TENSOR_CORE,
  DOWN_CONVERT_INPUTS = MUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS,
  REDUCED_PRECISION_REDUCTION =
      MUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION,
  FFT = MUDNN_NUMERICAL_NOTE_FFT,
  NONDETERMINISTIC = MUDNN_NUMERICAL_NOTE_NONDETERMINISTIC,
  WINOGRAD = MUDNN_NUMERICAL_NOTE_WINOGRAD,
  WINOGRAD_TILE_4x4 = MUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_4x4,
  WINOGRAD_TILE_6x6 = MUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_6x6,
  WINOGRAD_TILE_13x13 = MUDNN_NUMERICAL_NOTE_WINOGRAD_TILE_13x13,
  STRICT_NAN_PROP = MUDNN_NUMERICAL_NOTE_STRICT_NAN_PROP,
};
enum class DataType_t {
  NOT_SET = -1,
  FLOAT = MUDNN_DATA_FLOAT,
  DOUBLE = MUDNN_DATA_DOUBLE,
  HALF = MUDNN_DATA_HALF,
  INT8 = MUDNN_DATA_INT8,
  INT8x4 = MUDNN_DATA_INT8x4,
  UINT8 = MUDNN_DATA_UINT8,
  UINT8x4 = MUDNN_DATA_UINT8x4,
  INT32 = MUDNN_DATA_INT32,
  INT8x32 = MUDNN_DATA_INT8x32,
  BFLOAT16 = MUDNN_DATA_BFLOAT16,
  INT64 = MUDNN_DATA_INT64,
  BOOLEAN = MUDNN_DATA_BOOLEAN,
  FP8_E4M3 = MUDNN_DATA_FP8_E4M3,
  FP8_E5M2 = MUDNN_DATA_FP8_E5M2,
  FAST_FLOAT_FOR_FP8 = MUDNN_DATA_FAST_FLOAT_FOR_FP8,
  FP8_E8M0 = MUDNN_DATA_FP8_E8M0,
  FP4_E2M1 = MUDNN_DATA_FP4_E2M1,
  INT4 = MUDNN_DATA_INT4,
  UINT4 = MUDNN_DATA_UINT4,
  UINT32 = MUDNN_DATA_UINT32,
  COMPLEX_FP32 = MUDNN_DATA_COMPLEX_FP32,
  COMPLEX_FP64 = MUDNN_DATA_COMPLEX_FP64,
};
inline int64_t get_element_size_in_bits(DataType_t data_type) {
  switch (data_type) {
    case DataType_t::DOUBLE:
    case DataType_t::COMPLEX_FP64:
      return 64;
    case DataType_t::FLOAT:
    case DataType_t::INT32:
    case DataType_t::UINT32:
    case DataType_t::COMPLEX_FP32:
      return 32;
    case DataType_t::HALF:
    case DataType_t::BFLOAT16:
      return 16;
    case DataType_t::INT4:
    case DataType_t::UINT4:
    case DataType_t::FP4_E2M1:
      return 4;
    case DataType_t::BOOLEAN:
      return 1;
    case DataType_t::INT8:
    case DataType_t::UINT8:
    case DataType_t::FP8_E4M3:
    case DataType_t::FP8_E5M2:
    case DataType_t::FP8_E8M0:
    case DataType_t::FAST_FLOAT_FOR_FP8:
      return 8;
    case DataType_t::INT8x4:
    case DataType_t::UINT8x4:
      return 32;
    case DataType_t::INT8x32:
      return 256;
    case DataType_t::INT64:
      return 64;
    case DataType_t::NOT_SET:
    default:
      return 8;
  }
}
namespace detail {
inline int64_t get_element_size_in_bits(DataType_t data_type) {
  return mudnn_frontend::get_element_size_in_bits(data_type);
}
}  // namespace detail
enum class TensorReordering_t {
  NONE = 0,
  INT8x32 = MUDNN_TENSOR_REORDERING_INT8x32,
  F16x16 = MUDNN_TENSOR_REORDERING_F16x16,
  F8_128x4 = MUDNN_TENSOR_REORDERING_F8_128x4,
};
enum class KnobType_t {
  NOT_SET = 0,
  SWIZZLE = 1,
  TILE_SIZE = 2,
  EDGE = 3,
  MULTIPLY = 4,
  SPLIT_K_BUF = 5,
  TILEK = 6,
  STAGES = 7,
  REDUCTION_MODE = 8,
  SPLIT_K_SLC = 9,
  IDX_MODE = 10,
  SPECFILT = 11,
  KERNEL_CFG = 12,
  WORKSPACE = 13,
  TILE_CGA_M = 14,
  TILE_CGA_N = 15,
  BLOCK_SIZE = 16,
  OCCUPANCY = 17,
  ARRAY_SIZE_PER_THREAD = 18,
  SPLIT_COLS = 19,
  TILE_ROWS = 20,
  TILE_COLS = 21,
  LOAD_SIZE = 22,
  CTA_COUNT = 23,
  STREAM_K = 24,
  SPLIT_P_SLC = 25,
  TILE_M = 26,
  TILE_N = 27,
  WARP_SPEC_CFG = 28,
  SWAP_AB = 29,
  INPUT_TMA_ENABLE = 30,
  OUTPUT_TMA_ENABLE = 31,
};
struct Knob {
  KnobType_t type = KnobType_t::NOT_SET;
  int64_t minValue = 0;
  int64_t maxValue = 0;
  int64_t stride = 1;
  int64_t choice = -1;

  Knob() = default;
  Knob(KnobType_t type_, int64_t max_value, int64_t min_value,
       int64_t stride_)
      : type(type_),
        minValue(min_value),
        maxValue(max_value),
        stride(stride_) {}
  void setChoice(uint64_t value) { choice = static_cast<int64_t>(value); }
  int64_t getChoice() const { return choice; }
  mudnnBackendKnobType_t getKnobType() const {
    return static_cast<mudnnBackendKnobType_t>(type);
  }
  int64_t getMinValue() const { return minValue; }
  int64_t getMaxValue() const { return maxValue; }
  int64_t getStride() const { return stride; }
};
enum class PointwiseMode_t {
  NOT_SET = -1,
  ADD = MUDNN_POINTWISE_ADD,
  MUL = MUDNN_POINTWISE_MUL,
  MIN = MUDNN_POINTWISE_MIN,
  MAX = MUDNN_POINTWISE_MAX,
  SQRT = MUDNN_POINTWISE_SQRT,
  ADD_SQUARE = MUDNN_POINTWISE_ADD_SQUARE,
  DIV = MUDNN_POINTWISE_DIV,
  MOD = MUDNN_POINTWISE_MOD,
  POW = MUDNN_POINTWISE_POW,
  SUB = MUDNN_POINTWISE_SUB,
  ABS = MUDNN_POINTWISE_ABS,
  CEIL = MUDNN_POINTWISE_CEIL,
  COS = MUDNN_POINTWISE_COS,
  EXP = MUDNN_POINTWISE_EXP,
  FLOOR = MUDNN_POINTWISE_FLOOR,
  LOG = MUDNN_POINTWISE_LOG,
  NEG = MUDNN_POINTWISE_NEG,
  SIN = MUDNN_POINTWISE_SIN,
  RSQRT = MUDNN_POINTWISE_RSQRT,
  TAN = MUDNN_POINTWISE_TAN,
  ERF = MUDNN_POINTWISE_ERF,
  IDENTITY = MUDNN_POINTWISE_IDENTITY,
  RECIPROCAL = MUDNN_POINTWISE_RECIPROCAL,
  RELU_FWD = MUDNN_POINTWISE_RELU_FWD,
  TANH_FWD = MUDNN_POINTWISE_TANH_FWD,
  SIGMOID_FWD = MUDNN_POINTWISE_SIGMOID_FWD,
  ELU_FWD = MUDNN_POINTWISE_ELU_FWD,
  GELU_FWD = MUDNN_POINTWISE_GELU_FWD,
  SOFTPLUS_FWD = MUDNN_POINTWISE_SOFTPLUS_FWD,
  SWISH_FWD = MUDNN_POINTWISE_SWISH_FWD,
  GELU_APPROX_TANH_FWD = MUDNN_POINTWISE_GELU_APPROX_TANH_FWD,
  RELU_BWD = MUDNN_POINTWISE_RELU_BWD,
  TANH_BWD = MUDNN_POINTWISE_TANH_BWD,
  SIGMOID_BWD = MUDNN_POINTWISE_SIGMOID_BWD,
  ELU_BWD = MUDNN_POINTWISE_ELU_BWD,
  GELU_BWD = MUDNN_POINTWISE_GELU_BWD,
  SOFTPLUS_BWD = MUDNN_POINTWISE_SOFTPLUS_BWD,
  SWISH_BWD = MUDNN_POINTWISE_SWISH_BWD,
  GELU_APPROX_TANH_BWD = MUDNN_POINTWISE_GELU_APPROX_TANH_BWD,
  CMP_EQ = MUDNN_POINTWISE_CMP_EQ,
  CMP_NEQ = MUDNN_POINTWISE_CMP_NEQ,
  CMP_GT = MUDNN_POINTWISE_CMP_GT,
  CMP_GE = MUDNN_POINTWISE_CMP_GE,
  CMP_LT = MUDNN_POINTWISE_CMP_LT,
  CMP_LE = MUDNN_POINTWISE_CMP_LE,
  LOGICAL_AND = MUDNN_POINTWISE_LOGICAL_AND,
  LOGICAL_OR = MUDNN_POINTWISE_LOGICAL_OR,
  LOGICAL_NOT = MUDNN_POINTWISE_LOGICAL_NOT,
  BINARY_SELECT = MUDNN_POINTWISE_BINARY_SELECT,
  GEN_INDEX = MUDNN_POINTWISE_GEN_INDEX,
};
enum class ReductionMode_t {
  NOT_SET = -1,
  ADD = MUDNN_REDUCE_TENSOR_ADD,
  MUL = MUDNN_REDUCE_TENSOR_MUL,
  MIN = MUDNN_REDUCE_TENSOR_MIN,
  MAX = MUDNN_REDUCE_TENSOR_MAX,
  AMAX = MUDNN_REDUCE_TENSOR_AMAX,
  AVG = MUDNN_REDUCE_TENSOR_AVG,
  NORM1 = MUDNN_REDUCE_TENSOR_NORM1,
  NORM2 = MUDNN_REDUCE_TENSOR_NORM2,
  MUL_NO_ZEROS = MUDNN_REDUCE_TENSOR_MUL_NO_ZEROS,
};
enum class GenStatsMode_t {
  SUM_SQSUM = MUDNN_GENSTATS_SUM_SQSUM,
};
enum class RngDistribution_t {
  NOT_SET = -1,
  BERNOULLI = MUDNN_RNG_DISTRIBUTION_BERNOULLI,
  UNIFORM = MUDNN_RNG_DISTRIBUTION_UNIFORM,
  NORMAL = MUDNN_RNG_DISTRIBUTION_NORMAL,
};
enum class ConvolutionMode_t {
  NOT_SET = -1,
  CONVOLUTION = MUDNN_CONVOLUTION,
  CROSS_CORRELATION = MUDNN_CROSS_CORRELATION,
};
enum class ResampleMode_t {
  NOT_SET = -1,
  NEAREST = MUDNN_RESAMPLE_NEAREST,
  BILINEAR = MUDNN_RESAMPLE_BILINEAR,
  AVGPOOL_INCLUDE_PADDING = MUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING,
  AVGPOOL_EXCLUDE_PADDING = MUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING,
  MAXPOOL = MUDNN_RESAMPLE_MAXPOOL,
};
enum class MoeGroupedMatmulMode_t {
  NONE = MUDNN_MOE_GROUPED_MATMUL_MODE_NONE,
  GATHER = MUDNN_MOE_GROUPED_MATMUL_MODE_GATHER,
  SCATTER = MUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER,
};
enum class DescriptorType_t {
  NOT_SET = -1,
  POINTWISE_DESCRIPTOR = MUDNN_BACKEND_POINTWISE_DESCRIPTOR,
  CONVOLUTION_DESCRIPTOR = MUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
  ENGINE_DESCRIPTOR = MUDNN_BACKEND_ENGINE_DESCRIPTOR,
  ENGINECFG_DESCRIPTOR = MUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
  ENGINEHEUR_DESCRIPTOR = MUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
  EXECUTION_PLAN_DESCRIPTOR = MUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
  INTERMEDIATE_INFO_DESCRIPTOR = MUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR,
  KNOB_CHOICE_DESCRIPTOR = MUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR,
  KNOB_INFO_DESCRIPTOR = MUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
  LAYOUT_INFO_DESCRIPTOR = MUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR,
  OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
  OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
  OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
  OPERATION_POINTWISE_DESCRIPTOR = MUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
  OPERATION_GEN_STATS_DESCRIPTOR = MUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
  OPERATIONGRAPH_DESCRIPTOR = MUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
  VARIANT_PACK_DESCRIPTOR = MUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
  TENSOR_DESCRIPTOR = MUDNN_BACKEND_TENSOR_DESCRIPTOR,
  MATMUL_DESCRIPTOR = MUDNN_BACKEND_MATMUL_DESCRIPTOR,
  OPERATION_MATMUL_DESCRIPTOR = MUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
  OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
  REDUCTION_DESCRIPTOR = MUDNN_BACKEND_REDUCTION_DESCRIPTOR,
  OPERATION_REDUCTION_DESCRIPTOR = MUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
  OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
  RESAMPLE_DESCRIPTOR = MUDNN_BACKEND_RESAMPLE_DESCRIPTOR,
  OPERATION_RESAMPLE_FWD_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR,
  OPERATION_RESAMPLE_BWD_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR,
  OPERATION_CONCAT_DESCRIPTOR = MUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR,
  OPERATION_SIGNAL_DESCRIPTOR = MUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR,
  OPERATION_NORM_FORWARD_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR,
  OPERATION_NORM_BACKWARD_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR,
  OPERATION_RESHAPE_DESCRIPTOR = MUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR,
  RNG_DESCRIPTOR = MUDNN_BACKEND_RNG_DESCRIPTOR,
  OPERATION_RNG_DESCRIPTOR = MUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR,
  OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
  OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR,
  OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR,
  OPERATION_CONCATENATE_DESCRIPTOR = MUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR,
  OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR,
  OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR =
      MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR,
};

enum class NormMode_t {
  LAYER_NORM = MUDNN_LAYER_NORM,
  INSTANCE_NORM = MUDNN_INSTANCE_NORM,
  BATCH_NORM = MUDNN_BATCH_NORM,
  GROUP_NORM = MUDNN_GROUP_NORM,
  RMS_NORM = MUDNN_RMS_NORM,
  ADA_LAYER_NORM = MUDNN_ADA_LAYER_NORM,
};
enum class NormFwdPhase_t {
  NOT_SET = -1,
  INFERENCE = MUDNN_NORM_FWD_INFERENCE,
  TRAINING = MUDNN_NORM_FWD_TRAINING,
};
enum class BnFinalizeStatsMode_t {
  TRAINING = MUDNN_BN_FINALIZE_STATISTICS_TRAINING,
  INFERENCE = MUDNN_BN_FINALIZE_STATISTICS_INFERENCE,
};
enum class HeurMode_t {
  INSTANT = MUDNN_HEUR_MODE_INSTANT,
  B = MUDNN_HEUR_MODE_B,
  FALLBACK = MUDNN_HEUR_MODE_FALLBACK,
  A = MUDNN_HEUR_MODE_A,
  OPENSOURCE = MUDNN_HEUR_MODE_FALLBACK,
};
enum class BuildPlanPolicy_t { HEURISTICS_CHOICE = 0, ALL = 1 };

inline mudnnDataType_t to_mudnn(DataType_t type) {
  return static_cast<mudnnDataType_t>(type);
}
inline mudnnPointwiseMode_t to_mudnn(PointwiseMode_t mode) {
  return static_cast<mudnnPointwiseMode_t>(mode);
}
inline mudnnReduceTensorOp_t to_mudnn(ReductionMode_t mode) {
  return static_cast<mudnnReduceTensorOp_t>(mode);
}
inline mudnnGenStatsMode_t to_mudnn(GenStatsMode_t mode) {
  return static_cast<mudnnGenStatsMode_t>(mode);
}
inline mudnnRngDistribution_t to_mudnn(RngDistribution_t distribution) {
  return static_cast<mudnnRngDistribution_t>(distribution);
}
inline mudnnConvolutionMode_t to_mudnn(ConvolutionMode_t mode) {
  return static_cast<mudnnConvolutionMode_t>(mode);
}
inline mudnnResampleMode_t to_mudnn(ResampleMode_t mode) {
  return static_cast<mudnnResampleMode_t>(mode);
}
inline mudnnMoeGroupedMatmulMode_t to_mudnn(MoeGroupedMatmulMode_t mode) {
  return static_cast<mudnnMoeGroupedMatmulMode_t>(mode);
}
inline mudnnBackendNormMode_t to_mudnn(NormMode_t mode) {
  return static_cast<mudnnBackendNormMode_t>(mode);
}
inline mudnnBackendNormFwdPhase_t to_mudnn(NormFwdPhase_t phase) {
  return static_cast<mudnnBackendNormFwdPhase_t>(phase);
}
inline mudnnBnFinalizeStatsMode_t to_mudnn(BnFinalizeStatsMode_t mode) {
  return static_cast<mudnnBnFinalizeStatsMode_t>(mode);
}
inline mudnnBackendHeurMode_t to_mudnn(HeurMode_t mode) {
  return static_cast<mudnnBackendHeurMode_t>(mode);
}

class mudnnException : public std::runtime_error {
 public:
  mudnnException(mudnnStatus_t status, const std::string& message)
      : std::runtime_error(message + ": " + mudnnGetErrorString(status)),
        status_(status) {}

  mudnnStatus_t getCudnnStatus() const { return status_; }
  mudnnStatus_t getMudnnStatus() const { return status_; }

 private:
  mudnnStatus_t status_;
};

inline void throw_if_error(mudnnStatus_t status, const char* message) {
  if (status != MUDNN_STATUS_SUCCESS) {
    throw mudnnException(status, message);
  }
}

class BackendDescriptor {
 public:
  BackendDescriptor() = default;

  explicit BackendDescriptor(mudnnBackendDescriptorType_t type) {
    mudnnBackendDescriptor_t desc = nullptr;
    throw_if_error(mudnnBackendCreateDescriptor(type, &desc),
                   "mudnnBackendCreateDescriptor failed");
    desc_.reset(desc, [](mudnnBackendDescriptor_t ptr) {
      if (ptr != nullptr) {
        mudnnBackendDestroyDescriptor(ptr);
      }
    });
  }

  mudnnBackendDescriptor_t get_backend_descriptor() const {
    return desc_.get();
  }
  mudnnBackendDescriptor_t get_raw_desc() const { return desc_.get(); }
  explicit operator bool() const { return desc_ != nullptr; }

  void finalize() const {
    throw_if_error(mudnnBackendFinalize(desc_.get()),
                   "mudnnBackendFinalize failed");
  }

 protected:
  enum class Ownership { kOwning, kBorrowed };

  BackendDescriptor(mudnnBackendDescriptor_t desc, Ownership ownership) {
    if (ownership == Ownership::kOwning) {
      desc_.reset(desc, [](mudnnBackendDescriptor_t ptr) {
        if (ptr != nullptr) {
          mudnnBackendDestroyDescriptor(ptr);
        }
      });
    } else {
      desc_.reset(desc, [](mudnnBackendDescriptor_t) {});
    }
  }

  template <typename T>
  void set_attribute(mudnnBackendAttributeName_t name,
                     mudnnBackendAttributeType_t type, int64_t count,
                     const T* data) {
    throw_if_error(
        mudnnBackendSetAttribute(desc_.get(), name, type, count, data),
        "mudnnBackendSetAttribute failed");
  }

  void set_descriptor_attribute(mudnnBackendAttributeName_t name,
                                const BackendDescriptor& value) {
    auto raw = value.get_backend_descriptor();
    throw_if_error(
        mudnnBackendSetAttribute(desc_.get(), name,
                                 MUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &raw),
        "mudnnBackendSetAttribute descriptor failed");
  }

  template <typename T>
  int64_t get_attribute(mudnnBackendAttributeName_t name,
                        mudnnBackendAttributeType_t type,
                        int64_t requested_count, T* data) const {
    int64_t returned = 0;
    throw_if_error(mudnnBackendGetAttribute(desc_.get(), name, type,
                                            requested_count, &returned, data),
                   "mudnnBackendGetAttribute failed");
    return returned;
  }

 private:
  std::shared_ptr<void> desc_;
};

class Tensor : public BackendDescriptor {
 public:
  Tensor() = default;
  int64_t getId() const {
    int64_t value = 0;
    get_attribute(MUDNN_ATTR_TENSOR_UNIQUE_ID, MUDNN_TYPE_INT64, 1, &value);
    return value;
  }
  int64_t getAlignment() const {
    int64_t value = 0;
    get_attribute(MUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, MUDNN_TYPE_INT64, 1,
                  &value);
    return value;
  }
  mudnnDataType_t getDataType() const {
    mudnnDataType_t value = MUDNN_DATA_FLOAT;
    get_attribute(MUDNN_ATTR_TENSOR_DATA_TYPE, MUDNN_TYPE_DATA_TYPE, 1,
                  &value);
    return value;
  }
  int64_t getDimensionCount() const {
    return get_attribute<int64_t>(MUDNN_ATTR_TENSOR_DIMENSIONS,
                                  MUDNN_TYPE_INT64, 0, nullptr);
  }
  int64_t getDimCount() const { return getDimensionCount(); }
  std::vector<int64_t> getDimArray() const {
    std::vector<int64_t> values(static_cast<size_t>(getDimensionCount()));
    if (!values.empty()) {
      get_attribute(MUDNN_ATTR_TENSOR_DIMENSIONS, MUDNN_TYPE_INT64,
                    static_cast<int64_t>(values.size()), values.data());
    }
    return values;
  }
  std::vector<int64_t> getStrideArray() const {
    const auto count = get_attribute<int64_t>(MUDNN_ATTR_TENSOR_STRIDES,
                                               MUDNN_TYPE_INT64, 0, nullptr);
    std::vector<int64_t> values(static_cast<size_t>(count));
    if (!values.empty()) {
      get_attribute(MUDNN_ATTR_TENSOR_STRIDES, MUDNN_TYPE_INT64, count,
                    values.data());
    }
    return values;
  }
  const int64_t* getDim() const {
    cached_dims_ = getDimArray();
    return cached_dims_.data();
  }
  const int64_t* getStride() const {
    cached_strides_ = getStrideArray();
    return cached_strides_.data();
  }
  int64_t getPackedElementCount() const {
    int64_t result = 1;
    for (auto dim : getDimArray()) result *= dim;
    return result;
  }

 private:
  friend class TensorBuilder;
  mutable std::vector<int64_t> cached_dims_;
  mutable std::vector<int64_t> cached_strides_;
  using BackendDescriptor::BackendDescriptor;
};

class TensorBuilder {
 public:
  TensorBuilder& cloneFrom(const Tensor& tensor, int64_t id) {
    return setDim(tensor.getDimArray())
        .setStride(tensor.getStrideArray())
        .setId(id)
        .setDataType(tensor.getDataType())
        .setAlignment(tensor.getAlignment());
  }
  TensorBuilder& cloneFrom(const Tensor& tensor) {
    return cloneFrom(tensor, tensor.getId());
  }
  TensorBuilder& setDim(int64_t nb_dims, const int64_t* dims) {
    dims_.assign(dims, dims + nb_dims);
    return *this;
  }
  TensorBuilder& setDim(const std::vector<int64_t>& dims) {
    dims_ = dims;
    return *this;
  }
  TensorBuilder& setDim(std::initializer_list<int64_t> dims) {
    dims_.assign(dims.begin(), dims.end());
    return *this;
  }
  TensorBuilder& setStrides(int64_t nb_dims, const int64_t* strides) {
    strides_.assign(strides, strides + nb_dims);
    return *this;
  }
  TensorBuilder& setStride(int64_t nb_dims, const int64_t* strides) {
    return setStrides(nb_dims, strides);
  }
  TensorBuilder& setStrides(const std::vector<int64_t>& strides) {
    strides_ = strides;
    return *this;
  }
  TensorBuilder& setStride(const std::vector<int64_t>& strides) {
    return setStrides(strides);
  }
  TensorBuilder& setStrides(std::initializer_list<int64_t> strides) {
    strides_.assign(strides.begin(), strides.end());
    return *this;
  }
  TensorBuilder& setStride(std::initializer_list<int64_t> strides) {
    return setStrides(strides);
  }
  TensorBuilder& setId(int64_t id) {
    uid_ = id;
    return *this;
  }
  TensorBuilder& setAlignment(int64_t alignment) {
    alignment_ = alignment;
    return *this;
  }
  TensorBuilder& setDataType(mudnnDataType_t data_type) {
    data_type_ = data_type;
    return *this;
  }
  TensorBuilder& setDataType(DataType_t data_type) {
    return setDataType(static_cast<mudnnDataType_t>(data_type));
  }
  TensorBuilder& setRaggedOffset(const Tensor& tensor) {
    ragged_offset_ = &tensor;
    return *this;
  }
  TensorBuilder& setRaggedOffset(std::shared_ptr<Tensor>& tensor) {
    return setRaggedOffset(*tensor);
  }
  TensorBuilder& setVirtual(bool is_virtual) {
    is_virtual_ = is_virtual;
    return *this;
  }
  TensorBuilder& setByValue(bool is_by_value) {
    is_by_value_ = is_by_value;
    return *this;
  }
  TensorBuilder& setVectorCountAndDimension(int64_t vector_count,
                                            int64_t vector_dimension) {
    vector_count_ = vector_count;
    vector_dimension_ = vector_dimension;
    return *this;
  }
  TensorBuilder& setReorderType(TensorReordering_t reordering_type) {
    reordering_type_ = reordering_type;
    return *this;
  }

  Tensor build() const {
    Tensor tensor(MUDNN_BACKEND_TENSOR_DESCRIPTOR);
    tensor.set_attribute(MUDNN_ATTR_TENSOR_DATA_TYPE, MUDNN_TYPE_DATA_TYPE, 1,
                         &data_type_);
    tensor.set_attribute(MUDNN_ATTR_TENSOR_DIMENSIONS, MUDNN_TYPE_INT64,
                         static_cast<int64_t>(dims_.size()), dims_.data());
    tensor.set_attribute(MUDNN_ATTR_TENSOR_STRIDES, MUDNN_TYPE_INT64,
                         static_cast<int64_t>(strides_.size()),
                         strides_.data());
    tensor.set_attribute(MUDNN_ATTR_TENSOR_UNIQUE_ID, MUDNN_TYPE_INT64, 1,
                         &uid_);
    tensor.set_attribute(MUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, MUDNN_TYPE_INT64, 1,
                         &alignment_);
    tensor.set_attribute(MUDNN_ATTR_TENSOR_VECTOR_COUNT, MUDNN_TYPE_INT64, 1,
                         &vector_count_);
    tensor.set_attribute(MUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION,
                         MUDNN_TYPE_INT64, 1, &vector_dimension_);
    tensor.set_attribute(MUDNN_ATTR_TENSOR_IS_VIRTUAL, MUDNN_TYPE_BOOLEAN, 1,
                         &is_virtual_);
    tensor.set_attribute(MUDNN_ATTR_TENSOR_IS_BY_VALUE, MUDNN_TYPE_BOOLEAN, 1,
                         &is_by_value_);
    tensor.set_attribute(MUDNN_ATTR_TENSOR_REORDERING_MODE,
                         MUDNN_TYPE_TENSOR_REORDERING_MODE, 1,
                         &reordering_type_);
    if (ragged_offset_ != nullptr) {
      tensor.set_descriptor_attribute(MUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC,
                                      *ragged_offset_);
    }
    tensor.finalize();
    return tensor;
  }

 private:
  std::vector<int64_t> dims_;
  std::vector<int64_t> strides_;
  int64_t uid_ = 0;
  int64_t alignment_ = 16;
  int64_t vector_count_ = 1;
  int64_t vector_dimension_ = -1;
  mudnnDataType_t data_type_ = MUDNN_DATA_FLOAT;
  bool is_virtual_ = false;
  bool is_by_value_ = false;
  TensorReordering_t reordering_type_ = TensorReordering_t::NONE;
  const Tensor* ragged_offset_ = nullptr;
};

class PointWiseDesc : public BackendDescriptor {
 public:
  PointWiseDesc() = default;
  PointwiseMode_t getPointWiseMode() const { return mode_; }
  int64_t getPortCount() const {
    switch (mode_) {
      case PointwiseMode_t::ADD:
      case PointwiseMode_t::MUL:
      case PointwiseMode_t::MIN:
      case PointwiseMode_t::MAX:
      case PointwiseMode_t::SUB:
      case PointwiseMode_t::DIV:
      case PointwiseMode_t::MOD:
      case PointwiseMode_t::POW:
      case PointwiseMode_t::CMP_EQ:
      case PointwiseMode_t::CMP_NEQ:
      case PointwiseMode_t::CMP_GT:
      case PointwiseMode_t::CMP_GE:
      case PointwiseMode_t::CMP_LT:
      case PointwiseMode_t::CMP_LE:
      case PointwiseMode_t::LOGICAL_AND:
      case PointwiseMode_t::LOGICAL_OR:
        return 3;
      case PointwiseMode_t::BINARY_SELECT:
        return 4;
      default:
        return 2;
    }
  }

 private:
  friend class PointWiseDescBuilder;
  PointwiseMode_t mode_ = PointwiseMode_t::NOT_SET;
  using BackendDescriptor::BackendDescriptor;
};

class PointWiseDescBuilder {
 public:
  PointWiseDescBuilder& setMode(mudnnPointwiseMode_t mode) {
    mode_ = mode;
    return *this;
  }
  PointWiseDescBuilder& setMode(PointwiseMode_t mode) {
    return setMode(static_cast<mudnnPointwiseMode_t>(mode));
  }
  PointWiseDescBuilder& setComputeType(DataType_t type) {
    return setMathPrecision(static_cast<mudnnDataType_t>(type));
  }
  PointWiseDescBuilder& setComputeType(mudnnDataType_t type) {
    return setMathPrecision(type);
  }
  PointWiseDescBuilder& setClipping(double lower, double upper) {
    return setReluLowerClip(lower).setReluUpperClip(upper);
  }
  PointWiseDescBuilder& setMathPrecision(mudnnDataType_t precision) {
    precision_ = precision;
    return *this;
  }
  PointWiseDescBuilder& setReluLowerClip(double value) {
    relu_lower_clip_ = value;
    has_relu_lower_clip_ = true;
    return *this;
  }
  PointWiseDescBuilder& setReluUpperClip(double value) {
    relu_upper_clip_ = value;
    has_relu_upper_clip_ = true;
    return *this;
  }
  PointWiseDescBuilder& setReluLowerClipSlope(double value) {
    relu_lower_clip_slope_ = value;
    has_relu_lower_clip_slope_ = true;
    return *this;
  }
  PointWiseDescBuilder& setEluAlpha(double value) {
    elu_alpha_ = value;
    has_elu_alpha_ = true;
    return *this;
  }
  PointWiseDescBuilder& setSoftplusBeta(double value) {
    softplus_beta_ = value;
    has_softplus_beta_ = true;
    return *this;
  }
  PointWiseDescBuilder& setSwishBeta(double value) {
    swish_beta_ = value;
    has_swish_beta_ = true;
    return *this;
  }
  PointWiseDescBuilder& setAxis(int64_t axis) {
    axis_ = axis;
    has_axis_ = true;
    return *this;
  }

  PointWiseDesc build() const {
    PointWiseDesc desc(MUDNN_BACKEND_POINTWISE_DESCRIPTOR);
    desc.mode_ = static_cast<PointwiseMode_t>(mode_);
    desc.set_attribute(MUDNN_ATTR_POINTWISE_MODE, MUDNN_TYPE_POINTWISE_MODE, 1,
                       &mode_);
    desc.set_attribute(MUDNN_ATTR_POINTWISE_MATH_PREC, MUDNN_TYPE_DATA_TYPE, 1,
                       &precision_);
    if (has_relu_lower_clip_) {
      desc.set_attribute(MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                         MUDNN_TYPE_DOUBLE, 1, &relu_lower_clip_);
    }
    if (has_relu_upper_clip_) {
      desc.set_attribute(MUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                         MUDNN_TYPE_DOUBLE, 1, &relu_upper_clip_);
    }
    if (has_relu_lower_clip_slope_) {
      desc.set_attribute(MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                         MUDNN_TYPE_DOUBLE, 1, &relu_lower_clip_slope_);
    }
    if (has_elu_alpha_) {
      desc.set_attribute(MUDNN_ATTR_POINTWISE_ELU_ALPHA, MUDNN_TYPE_DOUBLE, 1,
                         &elu_alpha_);
    }
    if (has_softplus_beta_) {
      desc.set_attribute(MUDNN_ATTR_POINTWISE_SOFTPLUS_BETA, MUDNN_TYPE_DOUBLE,
                         1, &softplus_beta_);
    }
    if (has_swish_beta_) {
      desc.set_attribute(MUDNN_ATTR_POINTWISE_SWISH_BETA, MUDNN_TYPE_DOUBLE, 1,
                         &swish_beta_);
    }
    if (has_axis_) {
      desc.set_attribute(MUDNN_ATTR_POINTWISE_AXIS, MUDNN_TYPE_INT64, 1,
                         &axis_);
    }
    desc.finalize();
    return desc;
  }

 private:
  mudnnPointwiseMode_t mode_ = MUDNN_POINTWISE_ADD;
  mudnnDataType_t precision_ = MUDNN_DATA_FLOAT;
  double relu_lower_clip_ = 0.0;
  double relu_upper_clip_ = 0.0;
  double relu_lower_clip_slope_ = 0.0;
  double elu_alpha_ = 1.0;
  double softplus_beta_ = 1.0;
  double swish_beta_ = 1.0;
  int64_t axis_ = 0;
  bool has_relu_lower_clip_ = false;
  bool has_relu_upper_clip_ = false;
  bool has_relu_lower_clip_slope_ = false;
  bool has_elu_alpha_ = false;
  bool has_softplus_beta_ = false;
  bool has_swish_beta_ = false;
  bool has_axis_ = false;
};

using PointwiseDesc = PointWiseDesc;
using PointwiseDescBuilder = PointWiseDescBuilder;

class Operation : public BackendDescriptor {
 public:
  Operation() = default;

 private:
  friend class OperationBuilder;
  using BackendDescriptor::BackendDescriptor;
};

class MatMulDesc : public BackendDescriptor {
 public:
  MatMulDesc() = default;

 private:
  friend class MatMulDescBuilder;
  using BackendDescriptor::BackendDescriptor;
};

class MatMulDescBuilder {
 public:
  MatMulDescBuilder& setComputeType(DataType_t type) {
    return setMathPrecision(static_cast<mudnnDataType_t>(type));
  }
  MatMulDescBuilder& setComputeType(mudnnDataType_t type) {
    return setMathPrecision(type);
  }
  MatMulDescBuilder& setMathPrecision(mudnnDataType_t precision) {
    precision_ = precision;
    return *this;
  }
  MatMulDescBuilder& setPaddingValue(double padding_value) {
    padding_value_ = padding_value;
    has_padding_value_ = true;
    return *this;
  }

  MatMulDesc build() const {
    MatMulDesc desc(MUDNN_BACKEND_MATMUL_DESCRIPTOR);
    desc.set_attribute(MUDNN_ATTR_MATMUL_COMP_TYPE, MUDNN_TYPE_DATA_TYPE, 1,
                       &precision_);
    if (has_padding_value_) {
      desc.set_attribute(MUDNN_ATTR_MATMUL_PADDING_VALUE, MUDNN_TYPE_DOUBLE, 1,
                         &padding_value_);
    }
    desc.finalize();
    return desc;
  }

 private:
  mudnnDataType_t precision_ = MUDNN_DATA_FLOAT;
  double padding_value_ = 0.0;
  bool has_padding_value_ = false;
};

class ConvDesc : public BackendDescriptor {
 public:
  ConvDesc() = default;
  DataType_t getComputePrecision() const { return compute_type_; }
  DataType_t getComputeType() const { return compute_type_; }
  int64_t getSpatialDimCount() const { return spatial_dims_; }
  int64_t getDimensionCount() const { return getSpatialDimCount(); }
  const int64_t* getPadding() const { return pre_paddings_.data(); }
  const int64_t* getSpatialStride() const { return strides_.data(); }
  const int64_t* getStride() const { return getSpatialStride(); }
  const int64_t* getDilation() const { return dilations_.data(); }
  mudnnConvolutionMode_t getMathMode() const { return mode_; }

 private:
  friend class ConvDescBuilder;
  DataType_t compute_type_ = DataType_t::FLOAT;
  mudnnConvolutionMode_t mode_ = MUDNN_CROSS_CORRELATION;
  int64_t spatial_dims_ = 2;
  std::vector<int64_t> pre_paddings_ = {0, 0};
  std::vector<int64_t> strides_ = {1, 1};
  std::vector<int64_t> dilations_ = {1, 1};
  using BackendDescriptor::BackendDescriptor;
};

class ConvDescBuilder {
 public:
  ConvDescBuilder& setMathPrecision(mudnnDataType_t precision) {
    precision_ = precision;
    return *this;
  }
  ConvDescBuilder& setComputeType(mudnnDataType_t precision) {
    return setMathPrecision(precision);
  }
  ConvDescBuilder& setComputeType(DataType_t precision) {
    return setComputeType(static_cast<mudnnDataType_t>(precision));
  }
  ConvDescBuilder& setComputePrecision(mudnnDataType_t precision) {
    return setComputeType(precision);
  }
  ConvDescBuilder& setComputePrecision(DataType_t precision) {
    return setComputeType(precision);
  }
  ConvDescBuilder& setNDims(int64_t count) {
    return setSpatialDimCount(count);
  }
  ConvDescBuilder& setConvolutionMode(mudnnConvolutionMode_t mode) {
    mode_ = mode;
    return *this;
  }
  ConvDescBuilder& setMathMode(mudnnConvolutionMode_t mode) {
    return setConvolutionMode(mode);
  }
  ConvDescBuilder& setSpatialDimCount(int64_t count) {
    spatial_dims_ = count;
    return *this;
  }
  ConvDescBuilder& setPrePadding(int64_t count, const int64_t* values) {
    pre_paddings_.assign(values, values + count);
    return *this;
  }
  ConvDescBuilder& setPrePadding(const std::vector<int64_t>& values) {
    pre_paddings_ = values;
    return *this;
  }
  ConvDescBuilder& setPrePadding(std::initializer_list<int64_t> values) {
    pre_paddings_.assign(values.begin(), values.end());
    return *this;
  }
  ConvDescBuilder& setPostPadding(int64_t count, const int64_t* values) {
    post_paddings_.assign(values, values + count);
    return *this;
  }
  ConvDescBuilder& setPostPadding(const std::vector<int64_t>& values) {
    post_paddings_ = values;
    return *this;
  }
  ConvDescBuilder& setPostPadding(std::initializer_list<int64_t> values) {
    post_paddings_.assign(values.begin(), values.end());
    return *this;
  }
  ConvDescBuilder& setStrides(int64_t count, const int64_t* values) {
    strides_.assign(values, values + count);
    return *this;
  }
  ConvDescBuilder& setStride(int64_t count, const int64_t* values) {
    return setStrides(count, values);
  }
  ConvDescBuilder& setStride(const std::vector<int64_t>& values) {
    strides_ = values;
    return *this;
  }
  ConvDescBuilder& setStrides(const std::vector<int64_t>& values) {
    return setStride(values);
  }
  ConvDescBuilder& setStride(std::initializer_list<int64_t> values) {
    strides_.assign(values.begin(), values.end());
    return *this;
  }
  ConvDescBuilder& setStrides(std::initializer_list<int64_t> values) {
    return setStride(values);
  }
  ConvDescBuilder& setSpatialStride(int64_t count, const int64_t* values) {
    return setStrides(count, values);
  }
  ConvDescBuilder& setSpatialStride(const std::vector<int64_t>& values) {
    return setStrides(values);
  }
  ConvDescBuilder& setDilation(int64_t count, const int64_t* values) {
    dilations_.assign(values, values + count);
    return *this;
  }
  ConvDescBuilder& setDilation(const std::vector<int64_t>& values) {
    dilations_ = values;
    return *this;
  }
  ConvDescBuilder& setDilation(std::initializer_list<int64_t> values) {
    dilations_.assign(values.begin(), values.end());
    return *this;
  }
  ConvDescBuilder& setGroupCount(int64_t group_count) {
    group_count_ = group_count;
    return *this;
  }

  ConvDesc build() const {
    ConvDesc desc(MUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    desc.compute_type_ = static_cast<DataType_t>(precision_);
    desc.mode_ = mode_;
    desc.spatial_dims_ = spatial_dims_;
    desc.pre_paddings_ = pre_paddings_;
    desc.strides_ = strides_;
    desc.dilations_ = dilations_;
    desc.set_attribute(MUDNN_ATTR_CONVOLUTION_COMP_TYPE, MUDNN_TYPE_DATA_TYPE,
                       1, &precision_);
    desc.set_attribute(MUDNN_ATTR_CONVOLUTION_CONV_MODE,
                       MUDNN_TYPE_CONVOLUTION_MODE, 1, &mode_);
    desc.set_attribute(MUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, MUDNN_TYPE_INT64, 1,
                       &spatial_dims_);
    desc.set_attribute(MUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, MUDNN_TYPE_INT64,
                       static_cast<int64_t>(pre_paddings_.size()),
                       pre_paddings_.data());
    desc.set_attribute(MUDNN_ATTR_CONVOLUTION_POST_PADDINGS, MUDNN_TYPE_INT64,
                       static_cast<int64_t>(post_paddings_.size()),
                       post_paddings_.data());
    desc.set_attribute(MUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, MUDNN_TYPE_INT64,
                       static_cast<int64_t>(strides_.size()), strides_.data());
    desc.set_attribute(MUDNN_ATTR_CONVOLUTION_DILATIONS, MUDNN_TYPE_INT64,
                       static_cast<int64_t>(dilations_.size()),
                       dilations_.data());
    desc.set_attribute(MUDNN_ATTR_CONVOLUTION_GROUP_COUNT, MUDNN_TYPE_INT64, 1,
                       &group_count_);
    desc.finalize();
    return desc;
  }

 private:
  mudnnDataType_t precision_ = MUDNN_DATA_FLOAT;
  mudnnConvolutionMode_t mode_ = MUDNN_CROSS_CORRELATION;
  int64_t spatial_dims_ = 2;
  std::vector<int64_t> pre_paddings_ = {0, 0};
  std::vector<int64_t> post_paddings_ = {0, 0};
  std::vector<int64_t> strides_ = {1, 1};
  std::vector<int64_t> dilations_ = {1, 1};
  int64_t group_count_ = 1;
};

using ConvolutionDesc = ConvDesc;
using ConvolutionDescBuilder = ConvDescBuilder;

class ResampleDesc : public BackendDescriptor {
 public:
  ResampleDesc() = default;
  DataType_t getComputeType() const { return compute_type_; }
  int64_t getSpatialDimCount() const { return spatial_dims_; }
  mudnnNanPropagation_t getNanOpt() const { return nan_propagation_; }
  ResampleMode_t getMode() const { return mode_; }
  PaddingMode_t getPaddingMode() const { return padding_mode_; }
  const mudnnFraction_t* getSpatialStride() const { return strides_.data(); }
  const mudnnFraction_t* getPrePadding() const { return pre_paddings_.data(); }
  const mudnnFraction_t* getPostPadding() const { return post_paddings_.data(); }
  const mudnnFraction_t* getWindowDim() const { return window_dims_.data(); }

 private:
  friend class ResampleDescBuilder;
  DataType_t compute_type_ = DataType_t::FLOAT;
  int64_t spatial_dims_ = 2;
  mudnnNanPropagation_t nan_propagation_ = MUDNN_NOT_PROPAGATE_NAN;
  ResampleMode_t mode_ = ResampleMode_t::AVGPOOL_INCLUDE_PADDING;
  PaddingMode_t padding_mode_ = PaddingMode_t::ZERO_PAD;
  std::vector<mudnnFraction_t> strides_;
  std::vector<mudnnFraction_t> pre_paddings_;
  std::vector<mudnnFraction_t> post_paddings_;
  std::vector<mudnnFraction_t> window_dims_;
  using BackendDescriptor::BackendDescriptor;
};

class ResampleDescBuilder {
 public:
  ResampleDescBuilder& setMode(mudnnResampleMode_t mode) {
    mode_ = mode;
    return *this;
  }
  ResampleDescBuilder& setMathPrecision(mudnnDataType_t precision) {
    precision_ = precision;
    return *this;
  }
  ResampleDescBuilder& setComputeType(DataType_t precision) {
    return setMathPrecision(static_cast<mudnnDataType_t>(precision));
  }
  ResampleDescBuilder& setComputeType(mudnnDataType_t precision) {
    return setMathPrecision(precision);
  }
  ResampleDescBuilder& setResampleMode(ResampleMode_t mode) {
    return setMode(static_cast<mudnnResampleMode_t>(mode));
  }
  ResampleDescBuilder& setResampleMode(mudnnResampleMode_t mode) {
    return setMode(mode);
  }
  ResampleDescBuilder& setNanPropagation(mudnnNanPropagation_t value) {
    nan_propagation_ = value;
    has_nan_propagation_ = true;
    return *this;
  }
  ResampleDescBuilder& setPaddingMode(PaddingMode_t mode) {
    padding_mode_ = static_cast<mudnnPaddingMode_t>(mode);
    has_padding_mode_ = true;
    return *this;
  }
  ResampleDescBuilder& setPaddingMode(mudnnPaddingMode_t mode) {
    padding_mode_ = mode;
    has_padding_mode_ = true;
    return *this;
  }
  ResampleDescBuilder& setSpatialDimCount(int64_t count) {
    spatial_dims_ = count;
    return *this;
  }
  ResampleDescBuilder& setWindowDim(const std::vector<int64_t>& values) {
    window_dims_ = values;
    return *this;
  }
  ResampleDescBuilder& setWindowDim(std::initializer_list<int64_t> values) {
    window_dims_.assign(values.begin(), values.end());
    return *this;
  }
  ResampleDescBuilder& setSpatialDim(int64_t count,
                                     const mudnnFraction_t* values) {
    window_fractions_.assign(values, values + count);
    return *this;
  }
  ResampleDescBuilder& setPrePadding(int64_t count,
                                     const mudnnFraction_t* values) {
    pre_padding_fractions_.assign(values, values + count);
    return *this;
  }
  ResampleDescBuilder& setPostPadding(int64_t count,
                                      const mudnnFraction_t* values) {
    post_padding_fractions_.assign(values, values + count);
    return *this;
  }
  ResampleDescBuilder& setSpatialStride(int64_t count,
                                        const mudnnFraction_t* values) {
    stride_fractions_.assign(values, values + count);
    return *this;
  }
  ResampleDescBuilder& setPrePadding(const std::vector<int64_t>& values) {
    pre_paddings_ = values;
    return *this;
  }
  ResampleDescBuilder& setPrePadding(std::initializer_list<int64_t> values) {
    pre_paddings_.assign(values.begin(), values.end());
    return *this;
  }
  ResampleDescBuilder& setPostPadding(const std::vector<int64_t>& values) {
    post_paddings_ = values;
    return *this;
  }
  ResampleDescBuilder& setPostPadding(std::initializer_list<int64_t> values) {
    post_paddings_.assign(values.begin(), values.end());
    return *this;
  }
  ResampleDescBuilder& setStride(const std::vector<int64_t>& values) {
    strides_ = values;
    return *this;
  }
  ResampleDescBuilder& setStride(std::initializer_list<int64_t> values) {
    strides_.assign(values.begin(), values.end());
    return *this;
  }

  ResampleDesc build() const {
    ResampleDesc desc(MUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
    const auto as_fractions = [](const std::vector<int64_t>& values) {
      std::vector<mudnnFraction_t> result;
      result.reserve(values.size());
      for (auto value : values) result.push_back({value, 1});
      return result;
    };
    desc.compute_type_ = static_cast<DataType_t>(precision_);
    desc.spatial_dims_ = spatial_dims_;
    desc.nan_propagation_ = nan_propagation_;
    desc.mode_ = static_cast<ResampleMode_t>(mode_);
    desc.padding_mode_ = static_cast<PaddingMode_t>(padding_mode_);
    desc.window_dims_ = window_fractions_.empty()
                            ? as_fractions(window_dims_)
                            : window_fractions_;
    desc.pre_paddings_ = pre_padding_fractions_.empty()
                             ? as_fractions(pre_paddings_)
                             : pre_padding_fractions_;
    desc.post_paddings_ = post_padding_fractions_.empty()
                              ? as_fractions(post_paddings_)
                              : post_padding_fractions_;
    desc.strides_ = stride_fractions_.empty() ? as_fractions(strides_)
                                               : stride_fractions_;
    desc.set_attribute(MUDNN_ATTR_RESAMPLE_MODE, MUDNN_TYPE_RESAMPLE_MODE, 1,
                       &mode_);
    desc.set_attribute(MUDNN_ATTR_RESAMPLE_COMP_TYPE, MUDNN_TYPE_DATA_TYPE, 1,
                       &precision_);
    if (has_nan_propagation_) {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_NAN_PROPAGATION,
                         MUDNN_TYPE_NAN_PROPOGATION, 1, &nan_propagation_);
    }
    if (has_padding_mode_) {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_PADDING_MODE,
                         MUDNN_TYPE_PADDING_MODE, 1, &padding_mode_);
    }
    desc.set_attribute(MUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, MUDNN_TYPE_INT64, 1,
                       &spatial_dims_);
    if (window_fractions_.empty()) {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_WINDOW_DIMS, MUDNN_TYPE_INT64,
                         static_cast<int64_t>(window_dims_.size()),
                         window_dims_.data());
    } else {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_WINDOW_DIMS, MUDNN_TYPE_FRACTION,
                         static_cast<int64_t>(window_fractions_.size()),
                         window_fractions_.data());
    }
    if (pre_padding_fractions_.empty()) {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_PRE_PADDINGS, MUDNN_TYPE_INT64,
                         static_cast<int64_t>(pre_paddings_.size()),
                         pre_paddings_.data());
    } else {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                         MUDNN_TYPE_FRACTION,
                         static_cast<int64_t>(pre_padding_fractions_.size()),
                         pre_padding_fractions_.data());
    }
    if (post_padding_fractions_.empty()) {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_POST_PADDINGS, MUDNN_TYPE_INT64,
                         static_cast<int64_t>(post_paddings_.size()),
                         post_paddings_.data());
    } else {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_POST_PADDINGS,
                         MUDNN_TYPE_FRACTION,
                         static_cast<int64_t>(post_padding_fractions_.size()),
                         post_padding_fractions_.data());
    }
    if (stride_fractions_.empty()) {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_STRIDES, MUDNN_TYPE_INT64,
                         static_cast<int64_t>(strides_.size()),
                         strides_.data());
    } else {
      desc.set_attribute(MUDNN_ATTR_RESAMPLE_STRIDES, MUDNN_TYPE_FRACTION,
                         static_cast<int64_t>(stride_fractions_.size()),
                         stride_fractions_.data());
    }
    desc.finalize();
    return desc;
  }

 private:
  mudnnResampleMode_t mode_ = MUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
  mudnnDataType_t precision_ = MUDNN_DATA_FLOAT;
  int64_t spatial_dims_ = 2;
  std::vector<int64_t> window_dims_ = {2, 2};
  std::vector<int64_t> pre_paddings_ = {0, 0};
  std::vector<int64_t> post_paddings_ = {0, 0};
  std::vector<int64_t> strides_ = {2, 2};
  std::vector<mudnnFraction_t> window_fractions_;
  std::vector<mudnnFraction_t> pre_padding_fractions_;
  std::vector<mudnnFraction_t> post_padding_fractions_;
  std::vector<mudnnFraction_t> stride_fractions_;
  mudnnNanPropagation_t nan_propagation_ = MUDNN_NOT_PROPAGATE_NAN;
  mudnnPaddingMode_t padding_mode_ = MUDNN_ZERO_PAD;
  bool has_nan_propagation_ = false;
  bool has_padding_mode_ = false;
};

using PoolingDesc = ResampleDesc;
using PoolingDescBuilder = ResampleDescBuilder;

class RngDesc : public BackendDescriptor {
 public:
  RngDesc() = default;
  double getNormalDistMean() const { return normal_mean_; }
  double getNormalDistStdDev() const { return normal_stddev_; }
  double getUniformDistMax() const { return uniform_maximum_; }
  double getUniformDistMin() const { return uniform_minimum_; }
  double getBernoulliDistProbability() const { return bernoulli_probability_; }
  RngDistribution_t getDistribution() const { return distribution_; }

 private:
  friend class RngDescBuilder;
  RngDistribution_t distribution_ = RngDistribution_t::UNIFORM;
  double normal_mean_ = 0.0;
  double normal_stddev_ = 1.0;
  double uniform_minimum_ = 0.0;
  double uniform_maximum_ = 1.0;
  double bernoulli_probability_ = 0.5;
  using BackendDescriptor::BackendDescriptor;
};

class RngDescBuilder {
 public:
  RngDescBuilder& setDistribution(mudnnRngDistribution_t distribution) {
    distribution_ = distribution;
    return *this;
  }
  RngDescBuilder& setRngDistribution(RngDistribution_t distribution) {
    return setDistribution(static_cast<mudnnRngDistribution_t>(distribution));
  }
  RngDescBuilder& setRngDistribution(mudnnRngDistribution_t distribution) {
    return setDistribution(distribution);
  }
  RngDescBuilder& setNormalDistMean(double value) {
    normal_mean_ = value;
    return *this;
  }
  RngDescBuilder& setNormalDistStandardDeviation(double value) {
    normal_stddev_ = value;
    return *this;
  }
  RngDescBuilder& setNormalDistStdDev(double value) {
    return setNormalDistStandardDeviation(value);
  }
  RngDescBuilder& setNormalDistParams(double mean, double stddev) {
    return setNormalDistMean(mean).setNormalDistStandardDeviation(stddev);
  }
  RngDescBuilder& setUniformDistMinimum(double value) {
    uniform_minimum_ = value;
    return *this;
  }
  RngDescBuilder& setUniformDistMaximum(double value) {
    uniform_maximum_ = value;
    return *this;
  }
  RngDescBuilder& setUniformDistMin(double value) {
    return setUniformDistMinimum(value);
  }
  RngDescBuilder& setUniformDistMax(double value) {
    return setUniformDistMaximum(value);
  }
  RngDescBuilder& setUniformDistParams(double maximum, double minimum) {
    return setUniformDistMaximum(maximum).setUniformDistMinimum(minimum);
  }
  RngDescBuilder& setBernoulliDistProbability(double value) {
    bernoulli_probability_ = value;
    return *this;
  }

  RngDesc build() const {
    RngDesc rng(MUDNN_BACKEND_RNG_DESCRIPTOR);
    rng.distribution_ = static_cast<RngDistribution_t>(distribution_);
    rng.normal_mean_ = normal_mean_;
    rng.normal_stddev_ = normal_stddev_;
    rng.uniform_minimum_ = uniform_minimum_;
    rng.uniform_maximum_ = uniform_maximum_;
    rng.bernoulli_probability_ = bernoulli_probability_;
    rng.set_attribute(MUDNN_ATTR_RNG_DISTRIBUTION, MUDNN_TYPE_RNG_DISTRIBUTION,
                      1, &distribution_);
    rng.set_attribute(MUDNN_ATTR_RNG_NORMAL_DIST_MEAN, MUDNN_TYPE_DOUBLE, 1,
                      &normal_mean_);
    rng.set_attribute(MUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION,
                      MUDNN_TYPE_DOUBLE, 1, &normal_stddev_);
    rng.set_attribute(MUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM, MUDNN_TYPE_DOUBLE, 1,
                      &uniform_minimum_);
    rng.set_attribute(MUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM, MUDNN_TYPE_DOUBLE, 1,
                      &uniform_maximum_);
    rng.set_attribute(MUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY,
                      MUDNN_TYPE_DOUBLE, 1, &bernoulli_probability_);
    rng.finalize();
    return rng;
  }

 private:
  mudnnRngDistribution_t distribution_ = MUDNN_RNG_DISTRIBUTION_UNIFORM;
  double normal_mean_ = 0.0;
  double normal_stddev_ = 1.0;
  double uniform_minimum_ = 0.0;
  double uniform_maximum_ = 1.0;
  double bernoulli_probability_ = 0.5;
};

class ReductionDesc : public BackendDescriptor {
 public:
  ReductionDesc() = default;

 private:
  friend class ReductionDescBuilder;
  using BackendDescriptor::BackendDescriptor;
};

class ReductionDescBuilder {
 public:
  ReductionDescBuilder& setComputeType(DataType_t type) {
    return setMathPrecision(static_cast<mudnnDataType_t>(type));
  }
  ReductionDescBuilder& setComputeType(mudnnDataType_t type) {
    return setMathPrecision(type);
  }
  ReductionDescBuilder& setReductionOp(ReductionMode_t op) {
    return setReductionOp(static_cast<mudnnReduceTensorOp_t>(op));
  }
  ReductionDescBuilder& setIsDeterministic(bool deterministic) {
    deterministic_ = deterministic;
    has_deterministic_ = true;
    return *this;
  }
  ReductionDescBuilder& setReductionOp(mudnnReduceTensorOp_t op) {
    op_ = op;
    return *this;
  }
  ReductionDescBuilder& setMathPrecision(mudnnDataType_t precision) {
    precision_ = precision;
    return *this;
  }

  ReductionDesc build() const {
    ReductionDesc desc(MUDNN_BACKEND_REDUCTION_DESCRIPTOR);
    desc.set_attribute(MUDNN_ATTR_REDUCTION_OPERATOR,
                       MUDNN_TYPE_REDUCTION_OPERATOR_TYPE, 1, &op_);
    desc.set_attribute(MUDNN_ATTR_REDUCTION_COMP_TYPE, MUDNN_TYPE_DATA_TYPE, 1,
                       &precision_);
    if (has_deterministic_) {
      desc.set_attribute(MUDNN_ATTR_REDUCTION_IS_DETERMINISTIC,
                         MUDNN_TYPE_BOOLEAN, 1, &deterministic_);
    }
    desc.finalize();
    return desc;
  }

 private:
  mudnnReduceTensorOp_t op_ = MUDNN_REDUCE_TENSOR_ADD;
  mudnnDataType_t precision_ = MUDNN_DATA_FLOAT;
  bool deterministic_ = false;
  bool has_deterministic_ = false;
};


#include "mudnn_frontend/backend_runtime.h"
#include "mudnn_frontend/graph.h"

using error_code_t = graph::error_code_t;
using error_t = graph::error_t;

// cudnn-frontend 0.0.1 keeps the _v8 spellings as public source-compatibility
// names and aliases the short names to them.  muDNN's implementations use the
// short names internally, so expose the inverse aliases here.
using Tensor_v8 = Tensor;
using TensorBuilder_v8 = TensorBuilder;
using PointWiseDesc_v8 = PointWiseDesc;
using PointWiseDescBuilder_v8 = PointWiseDescBuilder;
using MatMulDesc_v8 = MatMulDesc;
using MatMulDescBuilder_v8 = MatMulDescBuilder;
using ConvDesc_v8 = ConvDesc;
using ConvDescBuilder_v8 = ConvDescBuilder;
using ResampleDesc_v8 = ResampleDesc;
using ResampleDescBuilder_v8 = ResampleDescBuilder;
using RngDesc_v8 = RngDesc;
using RngDescBuilder_v8 = RngDescBuilder;
using ReductionDesc_v8 = ReductionDesc;
using ReductionDescBuilder_v8 = ReductionDescBuilder;
using Operation_v8 = Operation;
using OperationBuilder_v8 = OperationBuilder;
using OperationGraph_v8 = OperationGraph;
using OperationGraphBuilder_v8 = OperationGraphBuilder;
using Engine_v8 = Engine;
using EngineBuilder_v8 = EngineBuilder;
using EngineConfig_v8 = EngineConfig;
using EngineConfigBuilder_v8 = EngineConfigBuilder;
using EngineHeuristics_v8 = EngineHeuristics;
using EngineHeuristicsBuilder_v8 = EngineHeuristicsBuilder;
using EngineFallbackList_v8 = EngineFallbackList;
using EngineFallbackListBuilder_v8 = EngineFallbackListBuilder;
using ExecutionPlan_v8 = ExecutionPlan;
using ExecutionPlanBuilder_v8 = ExecutionPlanBuilder;
using VariantPack_v8 = VariantPack;
using VariantPackBuilder_v8 = VariantPackBuilder;

}  // namespace mudnn_frontend



using cudnnHandle_t                  = mudnnHandle_t;
using cudnnStatus_t                  = mudnnStatus_t;
using cudnnDataType_t                = mudnnDataType_t;
using cudnnBackendDescriptor_t       = mudnnBackendDescriptor_t;
using cudnnBackendDescriptorType_t   = mudnnBackendDescriptorType_t;
using cudnnBackendAttributeName_t    = mudnnBackendAttributeName_t;
using cudnnBackendAttributeType_t    = mudnnBackendAttributeType_t;
using cudnnPointwiseMode_t           = mudnnPointwiseMode_t;
using cudnnBackendHeurMode_t         = mudnnBackendHeurMode_t;
using cudnnReduceTensorOp_t          = mudnnReduceTensorOp_t;
using cudnnConvolutionMode_t         = mudnnConvolutionMode_t;
using cudnnResampleMode_t            = mudnnResampleMode_t;
using cudnnRngDistribution_t         = mudnnRngDistribution_t;
using cudnnGenStatsMode_t            = mudnnGenStatsMode_t;
using cudnnBnFinalizeStatsMode_t     = mudnnBnFinalizeStatsMode_t;
using cudnnSignalMode_t              = mudnnSignalMode_t;
using cudnnMoeGroupedMatmulMode_t    = mudnnMoeGroupedMatmulMode_t;
using cudnnBackendNormMode_t         = mudnnBackendNormMode_t;
using cudnnBackendNormFwdPhase_t     = mudnnBackendNormFwdPhase_t;
using cudnnBackendNumericalNote_t    = mudnnBackendNumericalNote_t;
using cudnnBackendBehaviorNote_t     = mudnnBackendBehaviorNote_t;
using cudnnBackendTensorReordering_t = mudnnBackendTensorReordering_t;

static constexpr auto CUDNN_STATUS_SUCCESS         = MUDNN_STATUS_SUCCESS;
static constexpr auto CUDNN_STATUS_NOT_INITIALIZED = MUDNN_STATUS_NOT_INITIALIZED;
static constexpr auto CUDNN_STATUS_NOT_SUPPORTED   = MUDNN_STATUS_NOT_SUPPORTED;
static constexpr auto CUDNN_STATUS_BAD_PARAM       = MUDNN_STATUS_BAD_PARAM;
static constexpr auto CUDNN_STATUS_INTERNAL_ERROR  = MUDNN_STATUS_INTERNAL_ERROR;
static constexpr auto CUDNN_STATUS_ALLOC_FAILED    = MUDNN_STATUS_ALLOC_FAILED;

static constexpr auto CUDNN_DATA_FLOAT              = MUDNN_DATA_FLOAT;
static constexpr auto CUDNN_DATA_DOUBLE             = MUDNN_DATA_DOUBLE;
static constexpr auto CUDNN_DATA_HALF               = MUDNN_DATA_HALF;
static constexpr auto CUDNN_DATA_INT8               = MUDNN_DATA_INT8;
static constexpr auto CUDNN_DATA_INT8x4             = MUDNN_DATA_INT8x4;
static constexpr auto CUDNN_DATA_UINT8              = MUDNN_DATA_UINT8;
static constexpr auto CUDNN_DATA_UINT8x4            = MUDNN_DATA_UINT8x4;
static constexpr auto CUDNN_DATA_INT32              = MUDNN_DATA_INT32;
static constexpr auto CUDNN_DATA_INT8x32            = MUDNN_DATA_INT8x32;
static constexpr auto CUDNN_DATA_BFLOAT16           = MUDNN_DATA_BFLOAT16;
static constexpr auto CUDNN_DATA_INT64              = MUDNN_DATA_INT64;
static constexpr auto CUDNN_DATA_BOOLEAN            = MUDNN_DATA_BOOLEAN;
static constexpr auto CUDNN_DATA_FP8_E4M3           = MUDNN_DATA_FP8_E4M3;
static constexpr auto CUDNN_DATA_FP8_E5M2           = MUDNN_DATA_FP8_E5M2;
static constexpr auto CUDNN_DATA_FAST_FLOAT_FOR_FP8 = MUDNN_DATA_FAST_FLOAT_FOR_FP8;
static constexpr auto CUDNN_DATA_FP8_E8M0           = MUDNN_DATA_FP8_E8M0;
static constexpr auto CUDNN_DATA_FP4_E2M1           = MUDNN_DATA_FP4_E2M1;
static constexpr auto CUDNN_DATA_INT4               = MUDNN_DATA_INT4;
static constexpr auto CUDNN_DATA_UINT4              = MUDNN_DATA_UINT4;
static constexpr auto CUDNN_DATA_UINT32             = MUDNN_DATA_UINT32;
static constexpr auto CUDNN_DATA_COMPLEX_FP32       = MUDNN_DATA_COMPLEX_FP32;
static constexpr auto CUDNN_DATA_COMPLEX_FP64       = MUDNN_DATA_COMPLEX_FP64;

static constexpr auto CUDNN_TENSOR_REORDERING_NONE     = MUDNN_TENSOR_REORDERING_NONE;
static constexpr auto CUDNN_TENSOR_REORDERING_INT8x32  = MUDNN_TENSOR_REORDERING_INT8x32;
static constexpr auto CUDNN_TENSOR_REORDERING_F16x16   = MUDNN_TENSOR_REORDERING_F16x16;
static constexpr auto CUDNN_TENSOR_REORDERING_F8_128x4 = MUDNN_TENSOR_REORDERING_F8_128x4;
static constexpr auto CUDNN_RESHAPE_VIEW_ONLY          = MUDNN_RESHAPE_VIEW_ONLY;
static constexpr auto CUDNN_RESHAPE_LOGICAL            = MUDNN_RESHAPE_LOGICAL;

static constexpr auto CUDNN_TYPE_HANDLE                  = MUDNN_TYPE_HANDLE;
static constexpr auto CUDNN_TYPE_DATA_TYPE               = MUDNN_TYPE_DATA_TYPE;
static constexpr auto CUDNN_TYPE_BOOLEAN                 = MUDNN_TYPE_BOOLEAN;
static constexpr auto CUDNN_TYPE_INT64                   = MUDNN_TYPE_INT64;
static constexpr auto CUDNN_TYPE_FLOAT                   = MUDNN_TYPE_FLOAT;
static constexpr auto CUDNN_TYPE_DOUBLE                  = MUDNN_TYPE_DOUBLE;
static constexpr auto CUDNN_TYPE_VOID_PTR                = MUDNN_TYPE_VOID_PTR;
static constexpr auto CUDNN_TYPE_CONVOLUTION_MODE        = MUDNN_TYPE_CONVOLUTION_MODE;
static constexpr auto CUDNN_TYPE_HEUR_MODE               = MUDNN_TYPE_HEUR_MODE;
static constexpr auto CUDNN_TYPE_POINTWISE_MODE          = MUDNN_TYPE_POINTWISE_MODE;
static constexpr auto CUDNN_TYPE_BACKEND_DESCRIPTOR      = MUDNN_TYPE_BACKEND_DESCRIPTOR;
static constexpr auto CUDNN_TYPE_REDUCTION_OPERATOR_TYPE = MUDNN_TYPE_REDUCTION_OPERATOR_TYPE;
static constexpr auto CUDNN_TYPE_RESAMPLE_MODE           = MUDNN_TYPE_RESAMPLE_MODE;
static constexpr auto CUDNN_TYPE_NORM_MODE               = MUDNN_TYPE_NORM_MODE;
static constexpr auto CUDNN_TYPE_NORM_FWD_PHASE          = MUDNN_TYPE_NORM_FWD_PHASE;
static constexpr auto CUDNN_TYPE_RNG_DISTRIBUTION        = MUDNN_TYPE_RNG_DISTRIBUTION;
static constexpr auto CUDNN_TYPE_GENSTATS_MODE           = MUDNN_TYPE_GENSTATS_MODE;
static constexpr auto CUDNN_TYPE_BN_FINALIZE_STATS_MODE  = MUDNN_TYPE_BN_FINALIZE_STATS_MODE;
static constexpr auto CUDNN_TYPE_SIGNAL_MODE             = MUDNN_TYPE_SIGNAL_MODE;
static constexpr auto CUDNN_TYPE_MOE_GROUPED_MATMUL_MODE = MUDNN_TYPE_MOE_GROUPED_MATMUL_MODE;
static constexpr auto CUDNN_TYPE_RESHAPE_MODE            = MUDNN_TYPE_RESHAPE_MODE;

static constexpr auto CUDNN_NUMERICAL_NOTE_TENSOR_CORE         = MUDNN_NUMERICAL_NOTE_TENSOR_CORE;
static constexpr auto CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS = MUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS;
static constexpr auto CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION =
    MUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION;
static constexpr auto CUDNN_NUMERICAL_NOTE_FFT                = MUDNN_NUMERICAL_NOTE_FFT;
static constexpr auto CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC   = MUDNN_NUMERICAL_NOTE_NONDETERMINISTIC;
static constexpr auto CUDNN_NUMERICAL_NOTE_TYPE_COUNT         = MUDNN_NUMERICAL_NOTE_TYPE_COUNT;
static constexpr auto CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION = MUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
static constexpr auto CUDNN_BEHAVIOR_NOTE_TYPE_COUNT          = MUDNN_BEHAVIOR_NOTE_TYPE_COUNT;

static constexpr auto CUDNN_BACKEND_POINTWISE_DESCRIPTOR   = MUDNN_BACKEND_POINTWISE_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR = MUDNN_BACKEND_CONVOLUTION_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR = MUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR      = MUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_TENSOR_DESCRIPTOR              = MUDNN_BACKEND_TENSOR_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_ENGINE_DESCRIPTOR              = MUDNN_BACKEND_ENGINE_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_ENGINECFG_DESCRIPTOR           = MUDNN_BACKEND_ENGINECFG_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR          = MUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR      = MUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR        = MUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_MATMUL_DESCRIPTOR              = MUDNN_BACKEND_MATMUL_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR    = MUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR = MUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_REDUCTION_DESCRIPTOR              = MUDNN_BACKEND_REDUCTION_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR    = MUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_RESAMPLE_DESCRIPTOR               = MUDNN_BACKEND_RESAMPLE_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR = MUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR = MUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR       = MUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR = MUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR   = MUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR = MUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR     = MUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_RNG_DESCRIPTOR                 = MUDNN_BACKEND_RNG_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR       = MUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR    = MUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR = MUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR = MUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR;
static constexpr auto CUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR =
    MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR;

static constexpr auto CUDNN_POINTWISE_ADD                  = MUDNN_POINTWISE_ADD;
static constexpr auto CUDNN_POINTWISE_MUL                  = MUDNN_POINTWISE_MUL;
static constexpr auto CUDNN_POINTWISE_MIN                  = MUDNN_POINTWISE_MIN;
static constexpr auto CUDNN_POINTWISE_MAX                  = MUDNN_POINTWISE_MAX;
static constexpr auto CUDNN_POINTWISE_SQRT                 = MUDNN_POINTWISE_SQRT;
static constexpr auto CUDNN_POINTWISE_ADD_SQUARE           = MUDNN_POINTWISE_ADD_SQUARE;
static constexpr auto CUDNN_POINTWISE_DIV                  = MUDNN_POINTWISE_DIV;
static constexpr auto CUDNN_POINTWISE_MOD                  = MUDNN_POINTWISE_MOD;
static constexpr auto CUDNN_POINTWISE_POW                  = MUDNN_POINTWISE_POW;
static constexpr auto CUDNN_POINTWISE_SUB                  = MUDNN_POINTWISE_SUB;
static constexpr auto CUDNN_POINTWISE_ABS                  = MUDNN_POINTWISE_ABS;
static constexpr auto CUDNN_POINTWISE_CEIL                 = MUDNN_POINTWISE_CEIL;
static constexpr auto CUDNN_POINTWISE_COS                  = MUDNN_POINTWISE_COS;
static constexpr auto CUDNN_POINTWISE_EXP                  = MUDNN_POINTWISE_EXP;
static constexpr auto CUDNN_POINTWISE_FLOOR                = MUDNN_POINTWISE_FLOOR;
static constexpr auto CUDNN_POINTWISE_LOG                  = MUDNN_POINTWISE_LOG;
static constexpr auto CUDNN_POINTWISE_NEG                  = MUDNN_POINTWISE_NEG;
static constexpr auto CUDNN_POINTWISE_SIN                  = MUDNN_POINTWISE_SIN;
static constexpr auto CUDNN_POINTWISE_RSQRT                = MUDNN_POINTWISE_RSQRT;
static constexpr auto CUDNN_POINTWISE_TAN                  = MUDNN_POINTWISE_TAN;
static constexpr auto CUDNN_POINTWISE_ERF                  = MUDNN_POINTWISE_ERF;
static constexpr auto CUDNN_POINTWISE_IDENTITY             = MUDNN_POINTWISE_IDENTITY;
static constexpr auto CUDNN_POINTWISE_RECIPROCAL           = MUDNN_POINTWISE_RECIPROCAL;
static constexpr auto CUDNN_POINTWISE_RELU_FWD             = MUDNN_POINTWISE_RELU_FWD;
static constexpr auto CUDNN_POINTWISE_TANH_FWD             = MUDNN_POINTWISE_TANH_FWD;
static constexpr auto CUDNN_POINTWISE_SIGMOID_FWD          = MUDNN_POINTWISE_SIGMOID_FWD;
static constexpr auto CUDNN_POINTWISE_ELU_FWD              = MUDNN_POINTWISE_ELU_FWD;
static constexpr auto CUDNN_POINTWISE_GELU_FWD             = MUDNN_POINTWISE_GELU_FWD;
static constexpr auto CUDNN_POINTWISE_SOFTPLUS_FWD         = MUDNN_POINTWISE_SOFTPLUS_FWD;
static constexpr auto CUDNN_POINTWISE_SWISH_FWD            = MUDNN_POINTWISE_SWISH_FWD;
static constexpr auto CUDNN_POINTWISE_GELU_APPROX_TANH_FWD = MUDNN_POINTWISE_GELU_APPROX_TANH_FWD;
static constexpr auto CUDNN_POINTWISE_RELU_BWD             = MUDNN_POINTWISE_RELU_BWD;
static constexpr auto CUDNN_POINTWISE_TANH_BWD             = MUDNN_POINTWISE_TANH_BWD;
static constexpr auto CUDNN_POINTWISE_SIGMOID_BWD          = MUDNN_POINTWISE_SIGMOID_BWD;
static constexpr auto CUDNN_POINTWISE_ELU_BWD              = MUDNN_POINTWISE_ELU_BWD;
static constexpr auto CUDNN_POINTWISE_GELU_BWD             = MUDNN_POINTWISE_GELU_BWD;
static constexpr auto CUDNN_POINTWISE_SOFTPLUS_BWD         = MUDNN_POINTWISE_SOFTPLUS_BWD;
static constexpr auto CUDNN_POINTWISE_SWISH_BWD            = MUDNN_POINTWISE_SWISH_BWD;
static constexpr auto CUDNN_POINTWISE_GELU_APPROX_TANH_BWD = MUDNN_POINTWISE_GELU_APPROX_TANH_BWD;
static constexpr auto CUDNN_POINTWISE_CMP_EQ               = MUDNN_POINTWISE_CMP_EQ;
static constexpr auto CUDNN_POINTWISE_CMP_NEQ              = MUDNN_POINTWISE_CMP_NEQ;
static constexpr auto CUDNN_POINTWISE_CMP_GT               = MUDNN_POINTWISE_CMP_GT;
static constexpr auto CUDNN_POINTWISE_CMP_GE               = MUDNN_POINTWISE_CMP_GE;
static constexpr auto CUDNN_POINTWISE_CMP_LT               = MUDNN_POINTWISE_CMP_LT;
static constexpr auto CUDNN_POINTWISE_CMP_LE               = MUDNN_POINTWISE_CMP_LE;
static constexpr auto CUDNN_POINTWISE_LOGICAL_AND          = MUDNN_POINTWISE_LOGICAL_AND;
static constexpr auto CUDNN_POINTWISE_LOGICAL_OR           = MUDNN_POINTWISE_LOGICAL_OR;
static constexpr auto CUDNN_POINTWISE_LOGICAL_NOT          = MUDNN_POINTWISE_LOGICAL_NOT;
static constexpr auto CUDNN_POINTWISE_GEN_INDEX            = MUDNN_POINTWISE_GEN_INDEX;
static constexpr auto CUDNN_POINTWISE_BINARY_SELECT        = MUDNN_POINTWISE_BINARY_SELECT;

static constexpr auto CUDNN_CONVOLUTION       = MUDNN_CONVOLUTION;
static constexpr auto CUDNN_CROSS_CORRELATION = MUDNN_CROSS_CORRELATION;

static constexpr auto CUDNN_REDUCE_TENSOR_NOT_SET      = static_cast<mudnnReduceTensorOp_t>(-1);
static constexpr auto CUDNN_REDUCE_TENSOR_ADD          = MUDNN_REDUCE_TENSOR_ADD;
static constexpr auto CUDNN_REDUCE_TENSOR_MUL          = MUDNN_REDUCE_TENSOR_MUL;
static constexpr auto CUDNN_REDUCE_TENSOR_MIN          = MUDNN_REDUCE_TENSOR_MIN;
static constexpr auto CUDNN_REDUCE_TENSOR_MAX          = MUDNN_REDUCE_TENSOR_MAX;
static constexpr auto CUDNN_REDUCE_TENSOR_AMAX         = MUDNN_REDUCE_TENSOR_AMAX;
static constexpr auto CUDNN_REDUCE_TENSOR_AVG          = MUDNN_REDUCE_TENSOR_AVG;
static constexpr auto CUDNN_REDUCE_TENSOR_NORM1        = MUDNN_REDUCE_TENSOR_NORM1;
static constexpr auto CUDNN_REDUCE_TENSOR_NORM2        = MUDNN_REDUCE_TENSOR_NORM2;
static constexpr auto CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = MUDNN_REDUCE_TENSOR_MUL_NO_ZEROS;

static constexpr auto CUDNN_RESAMPLE_NEAREST                 = MUDNN_RESAMPLE_NEAREST;
static constexpr auto CUDNN_RESAMPLE_BILINEAR                = MUDNN_RESAMPLE_BILINEAR;
static constexpr auto CUDNN_RESAMPLE_AVGPOOL                 = MUDNN_RESAMPLE_AVGPOOL;
static constexpr auto CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING = MUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
static constexpr auto CUDNN_RESAMPLE_MAXPOOL                 = MUDNN_RESAMPLE_MAXPOOL;
static constexpr auto CUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING = MUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING;

static constexpr auto CUDNN_RNG_DISTRIBUTION_BERNOULLI       = MUDNN_RNG_DISTRIBUTION_BERNOULLI;
static constexpr auto CUDNN_RNG_DISTRIBUTION_UNIFORM         = MUDNN_RNG_DISTRIBUTION_UNIFORM;
static constexpr auto CUDNN_RNG_DISTRIBUTION_NORMAL          = MUDNN_RNG_DISTRIBUTION_NORMAL;
static constexpr auto CUDNN_GENSTATS_SUM_SQSUM               = MUDNN_GENSTATS_SUM_SQSUM;
static constexpr auto CUDNN_BN_FINALIZE_STATISTICS_TRAINING  = MUDNN_BN_FINALIZE_STATISTICS_TRAINING;
static constexpr auto CUDNN_BN_FINALIZE_STATISTICS_INFERENCE = MUDNN_BN_FINALIZE_STATISTICS_INFERENCE;
static constexpr auto CUDNN_SIGNAL_SET                       = MUDNN_SIGNAL_SET;
static constexpr auto CUDNN_SIGNAL_WAIT                      = MUDNN_SIGNAL_WAIT;
static constexpr auto CUDNN_MOE_GROUPED_MATMUL_MODE_NONE     = MUDNN_MOE_GROUPED_MATMUL_MODE_NONE;
static constexpr auto CUDNN_MOE_GROUPED_MATMUL_MODE_GATHER   = MUDNN_MOE_GROUPED_MATMUL_MODE_GATHER;
static constexpr auto CUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER  = MUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;

static constexpr auto CUDNN_LAYER_NORM         = MUDNN_LAYER_NORM;
static constexpr auto CUDNN_INSTANCE_NORM      = MUDNN_INSTANCE_NORM;
static constexpr auto CUDNN_BATCH_NORM         = MUDNN_BATCH_NORM;
static constexpr auto CUDNN_GROUP_NORM         = MUDNN_GROUP_NORM;
static constexpr auto CUDNN_RMS_NORM           = MUDNN_RMS_NORM;
static constexpr auto CUDNN_ADA_LAYER_NORM     = MUDNN_ADA_LAYER_NORM;
static constexpr auto CUDNN_NORM_FWD_NOT_SET   = static_cast<mudnnBackendNormFwdPhase_t>(-1);
static constexpr auto CUDNN_NORM_FWD_INFERENCE = MUDNN_NORM_FWD_INFERENCE;
static constexpr auto CUDNN_NORM_FWD_TRAINING  = MUDNN_NORM_FWD_TRAINING;

static constexpr auto CUDNN_HEUR_MODE_INSTANT    = MUDNN_HEUR_MODE_INSTANT;
static constexpr auto CUDNN_HEUR_MODE_B          = MUDNN_HEUR_MODE_B;
static constexpr auto CUDNN_HEUR_MODE_FALLBACK   = MUDNN_HEUR_MODE_FALLBACK;
static constexpr auto CUDNN_HEUR_MODE_A          = MUDNN_HEUR_MODE_A;
static constexpr auto CUDNN_HEUR_MODE_OPENSOURCE = MUDNN_HEUR_MODE_FALLBACK;

static constexpr auto CUDNN_ATTR_POINTWISE_MODE                  = MUDNN_ATTR_POINTWISE_MODE;
static constexpr auto CUDNN_ATTR_POINTWISE_MATH_PREC             = MUDNN_ATTR_POINTWISE_MATH_PREC;
static constexpr auto CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP       = MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP;
static constexpr auto CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP       = MUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP;
static constexpr auto CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = MUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE;
static constexpr auto CUDNN_ATTR_POINTWISE_SWISH_BETA            = MUDNN_ATTR_POINTWISE_SWISH_BETA;
static constexpr auto CUDNN_ATTR_POINTWISE_AXIS                  = MUDNN_ATTR_POINTWISE_AXIS;

static constexpr auto CUDNN_ATTR_CONVOLUTION_COMP_TYPE      = MUDNN_ATTR_CONVOLUTION_COMP_TYPE;
static constexpr auto CUDNN_ATTR_CONVOLUTION_CONV_MODE      = MUDNN_ATTR_CONVOLUTION_CONV_MODE;
static constexpr auto CUDNN_ATTR_CONVOLUTION_DILATIONS      = MUDNN_ATTR_CONVOLUTION_DILATIONS;
static constexpr auto CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = MUDNN_ATTR_CONVOLUTION_FILTER_STRIDES;
static constexpr auto CUDNN_ATTR_CONVOLUTION_POST_PADDINGS  = MUDNN_ATTR_CONVOLUTION_POST_PADDINGS;
static constexpr auto CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS   = MUDNN_ATTR_CONVOLUTION_PRE_PADDINGS;
static constexpr auto CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS   = MUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS;
static constexpr auto CUDNN_ATTR_CONVOLUTION_GROUP_COUNT    = MUDNN_ATTR_CONVOLUTION_GROUP_COUNT;

static constexpr auto CUDNN_ATTR_ENGINEHEUR_MODE            = MUDNN_ATTR_ENGINEHEUR_MODE;
static constexpr auto CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = MUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH;
static constexpr auto CUDNN_ATTR_ENGINEHEUR_RESULTS         = MUDNN_ATTR_ENGINEHEUR_RESULTS;

static constexpr auto CUDNN_ATTR_ENGINECFG_ENGINE         = MUDNN_ATTR_ENGINECFG_ENGINE;
static constexpr auto CUDNN_ATTR_ENGINECFG_KNOB_CHOICES   = MUDNN_ATTR_ENGINECFG_KNOB_CHOICES;
static constexpr auto CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE = MUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE;

static constexpr auto CUDNN_ATTR_EXECUTION_PLAN_HANDLE         = MUDNN_ATTR_EXECUTION_PLAN_HANDLE;
static constexpr auto CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG  = MUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG;
static constexpr auto CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE = MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE;

static constexpr auto CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR   = MUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA = MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA  = MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC =
    MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W      = MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X      = MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y      = MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA = MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA  = MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC =
    MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W  = MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX = MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY = MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA =
    MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA =
    MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC =
    MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW = MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X  = MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X;
static constexpr auto CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY = MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY;
static constexpr auto CUDNN_ATTR_OPERATION_POINTWISE_XDESC           = MUDNN_ATTR_OPERATION_POINTWISE_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_POINTWISE_BDESC           = MUDNN_ATTR_OPERATION_POINTWISE_BDESC;
static constexpr auto CUDNN_ATTR_OPERATION_POINTWISE_YDESC           = MUDNN_ATTR_OPERATION_POINTWISE_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1          = MUDNN_ATTR_OPERATION_POINTWISE_ALPHA1;
static constexpr auto CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2          = MUDNN_ATTR_OPERATION_POINTWISE_ALPHA2;
static constexpr auto CUDNN_ATTR_OPERATION_POINTWISE_DXDESC          = MUDNN_ATTR_OPERATION_POINTWISE_DXDESC;
static constexpr auto CUDNN_ATTR_OPERATION_POINTWISE_DYDESC          = MUDNN_ATTR_OPERATION_POINTWISE_DYDESC;
static constexpr auto CUDNN_ATTR_OPERATION_POINTWISE_TDESC           = MUDNN_ATTR_OPERATION_POINTWISE_TDESC;

static constexpr auto CUDNN_ATTR_OPERATIONGRAPH_HANDLE              = MUDNN_ATTR_OPERATIONGRAPH_HANDLE;
static constexpr auto CUDNN_ATTR_OPERATIONGRAPH_OPS                 = MUDNN_ATTR_OPERATIONGRAPH_OPS;
static constexpr auto CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = MUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT;
static constexpr auto CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED =
    MUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED;
static constexpr auto CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED =
    MUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED;

static constexpr auto CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT = MUDNN_ATTR_TENSOR_BYTE_ALIGNMENT;
static constexpr auto CUDNN_ATTR_TENSOR_DATA_TYPE      = MUDNN_ATTR_TENSOR_DATA_TYPE;
static constexpr auto CUDNN_ATTR_TENSOR_DIMENSIONS     = MUDNN_ATTR_TENSOR_DIMENSIONS;
static constexpr auto CUDNN_ATTR_TENSOR_STRIDES        = MUDNN_ATTR_TENSOR_STRIDES;
static constexpr auto CUDNN_ATTR_TENSOR_VECTOR_COUNT   = MUDNN_ATTR_TENSOR_VECTOR_COUNT;
static constexpr auto CUDNN_ATTR_TENSOR_UNIQUE_ID      = MUDNN_ATTR_TENSOR_UNIQUE_ID;
static constexpr auto CUDNN_ATTR_TENSOR_IS_VIRTUAL     = MUDNN_ATTR_TENSOR_IS_VIRTUAL;
static constexpr auto CUDNN_ATTR_TENSOR_IS_BY_VALUE    = MUDNN_ATTR_TENSOR_IS_BY_VALUE;
static constexpr auto CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC = MUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC;

static constexpr auto CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS    = MUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS;
static constexpr auto CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = MUDNN_ATTR_VARIANT_PACK_DATA_POINTERS;
static constexpr auto CUDNN_ATTR_VARIANT_PACK_WORKSPACE     = MUDNN_ATTR_VARIANT_PACK_WORKSPACE;

static constexpr auto CUDNN_ATTR_ENGINE_OPERATION_GRAPH = MUDNN_ATTR_ENGINE_OPERATION_GRAPH;
static constexpr auto CUDNN_ATTR_ENGINE_GLOBAL_INDEX    = MUDNN_ATTR_ENGINE_GLOBAL_INDEX;
static constexpr auto CUDNN_ATTR_ENGINE_KNOB_INFO       = MUDNN_ATTR_ENGINE_KNOB_INFO;

static constexpr auto CUDNN_ATTR_MATMUL_COMP_TYPE       = MUDNN_ATTR_MATMUL_COMP_TYPE;
static constexpr auto CUDNN_ATTR_MATMUL_PADDING_VALUE   = MUDNN_ATTR_MATMUL_PADDING_VALUE;
static constexpr auto CUDNN_ATTR_OPERATION_MATMUL_ADESC = MUDNN_ATTR_OPERATION_MATMUL_ADESC;
static constexpr auto CUDNN_ATTR_OPERATION_MATMUL_BDESC = MUDNN_ATTR_OPERATION_MATMUL_BDESC;
static constexpr auto CUDNN_ATTR_OPERATION_MATMUL_CDESC = MUDNN_ATTR_OPERATION_MATMUL_CDESC;
static constexpr auto CUDNN_ATTR_OPERATION_MATMUL_DESC  = MUDNN_ATTR_OPERATION_MATMUL_DESC;

static constexpr auto CUDNN_ATTR_REDUCTION_OPERATOR        = MUDNN_ATTR_REDUCTION_OPERATOR;
static constexpr auto CUDNN_ATTR_REDUCTION_COMP_TYPE       = MUDNN_ATTR_REDUCTION_COMP_TYPE;
static constexpr auto CUDNN_ATTR_OPERATION_REDUCTION_XDESC = MUDNN_ATTR_OPERATION_REDUCTION_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_REDUCTION_YDESC = MUDNN_ATTR_OPERATION_REDUCTION_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_REDUCTION_DESC  = MUDNN_ATTR_OPERATION_REDUCTION_DESC;

static constexpr auto CUDNN_ATTR_OPERATION_GENSTATS_MODE      = MUDNN_ATTR_OPERATION_GENSTATS_MODE;
static constexpr auto CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC = MUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC;
static constexpr auto CUDNN_ATTR_OPERATION_GENSTATS_XDESC     = MUDNN_ATTR_OPERATION_GENSTATS_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC   = MUDNN_ATTR_OPERATION_GENSTATS_SUMDESC;
static constexpr auto CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC = MUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC;

static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE    = MUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC     = MUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC    = MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC = MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC    = MUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC     = MUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC =
    MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC =
    MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC =
    MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC =
    MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC =
    MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC =
    MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC = MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC  = MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC =
    MUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC = MUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC =
    MUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC   = MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC   = MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC = MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC =
    MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC  = MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC = MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC =
    MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC =
    MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC =
    MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC =
    MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS = MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS;

static constexpr auto CUDNN_ATTR_RESAMPLE_MODE                  = MUDNN_ATTR_RESAMPLE_MODE;
static constexpr auto CUDNN_ATTR_RESAMPLE_COMP_TYPE             = MUDNN_ATTR_RESAMPLE_COMP_TYPE;
static constexpr auto CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS          = MUDNN_ATTR_RESAMPLE_SPATIAL_DIMS;
static constexpr auto CUDNN_ATTR_RESAMPLE_POST_PADDINGS         = MUDNN_ATTR_RESAMPLE_POST_PADDINGS;
static constexpr auto CUDNN_ATTR_RESAMPLE_PRE_PADDINGS          = MUDNN_ATTR_RESAMPLE_PRE_PADDINGS;
static constexpr auto CUDNN_ATTR_RESAMPLE_STRIDES               = MUDNN_ATTR_RESAMPLE_STRIDES;
static constexpr auto CUDNN_ATTR_RESAMPLE_WINDOW_DIMS           = MUDNN_ATTR_RESAMPLE_WINDOW_DIMS;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC   = MUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC   = MUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC = MUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA   = MUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA    = MUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC    = MUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC  = MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC  = MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC = MUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA   = MUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA    = MUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC    = MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC   = MUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC   = MUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC;

static constexpr auto CUDNN_ATTR_OPERATION_CONCAT_AXIS                = MUDNN_ATTR_OPERATION_CONCAT_AXIS;
static constexpr auto CUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS         = MUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS;
static constexpr auto CUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX       = MUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX;
static constexpr auto CUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC         = MUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_MODE              = MUDNN_ATTR_OPERATION_NORM_FWD_MODE;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_PHASE             = MUDNN_ATTR_OPERATION_NORM_FWD_PHASE;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_XDESC             = MUDNN_ATTR_OPERATION_NORM_FWD_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC         = MUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC = MUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC        = MUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC         = MUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC      = MUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC =
    MUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC =
    MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC =
    MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC =
    MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC =
    MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_FWD_YDESC             = MUDNN_ATTR_OPERATION_NORM_FWD_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_MODE              = MUDNN_ATTR_OPERATION_NORM_BWD_MODE;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_XDESC             = MUDNN_ATTR_OPERATION_NORM_BWD_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC         = MUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC = MUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_DYDESC            = MUDNN_ATTR_OPERATION_NORM_BWD_DYDESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC        = MUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC      = MUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC       = MUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC        = MUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_NORM_BWD_DXDESC            = MUDNN_ATTR_OPERATION_NORM_BWD_DXDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESHAPE_XDESC              = MUDNN_ATTR_OPERATION_RESHAPE_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESHAPE_YDESC              = MUDNN_ATTR_OPERATION_RESHAPE_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RESHAPE_MODE               = MUDNN_ATTR_OPERATION_RESHAPE_MODE;
static constexpr auto CUDNN_ATTR_OPERATION_SLICE_XDESC                = MUDNN_ATTR_OPERATION_SLICE_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SLICE_YDESC                = MUDNN_ATTR_OPERATION_SLICE_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SLICE_START_INDICES        = MUDNN_ATTR_OPERATION_SLICE_START_INDICES;
static constexpr auto CUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES        = MUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES;
static constexpr auto CUDNN_ATTR_OPERATION_SLICE_STRIDES              = MUDNN_ATTR_OPERATION_SLICE_STRIDES;
static constexpr auto CUDNN_ATTR_OPERATION_TRANSPOSE_XDESC            = MUDNN_ATTR_OPERATION_TRANSPOSE_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_TRANSPOSE_YDESC            = MUDNN_ATTR_OPERATION_TRANSPOSE_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION      = MUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION;

static constexpr auto CUDNN_ATTR_RNG_DISTRIBUTION                   = MUDNN_ATTR_RNG_DISTRIBUTION;
static constexpr auto CUDNN_ATTR_RNG_NORMAL_DIST_MEAN               = MUDNN_ATTR_RNG_NORMAL_DIST_MEAN;
static constexpr auto CUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION = MUDNN_ATTR_RNG_NORMAL_DIST_STANDARD_DEVIATION;
static constexpr auto CUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM           = MUDNN_ATTR_RNG_UNIFORM_DIST_MAXIMUM;
static constexpr auto CUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM           = MUDNN_ATTR_RNG_UNIFORM_DIST_MINIMUM;
static constexpr auto CUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY     = MUDNN_ATTR_RNG_BERNOULLI_DIST_PROBABILITY;
static constexpr auto CUDNN_ATTR_OPERATION_RNG_YDESC                = MUDNN_ATTR_OPERATION_RNG_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_RNG_SEED                 = MUDNN_ATTR_OPERATION_RNG_SEED;
static constexpr auto CUDNN_ATTR_OPERATION_RNG_DESC                 = MUDNN_ATTR_OPERATION_RNG_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_RNG_OFFSET_DESC          = MUDNN_ATTR_OPERATION_RNG_OFFSET_DESC;

static constexpr auto CUDNN_ATTR_OPERATION_SIGNAL_MODE     = MUDNN_ATTR_OPERATION_SIGNAL_MODE;
static constexpr auto CUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC = MUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SIGNAL_VALUE    = MUDNN_ATTR_OPERATION_SIGNAL_VALUE;
static constexpr auto CUDNN_ATTR_OPERATION_SIGNAL_XDESC    = MUDNN_ATTR_OPERATION_SIGNAL_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SIGNAL_YDESC    = MUDNN_ATTR_OPERATION_SIGNAL_YDESC;

static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE = MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K = MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC =
    MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC;

static constexpr auto CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC =
    MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC = MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC =
    MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC =
    MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC;

static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC = MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC = MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC =
    MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC =
    MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE =
    MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC =
    MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC =
    MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC =
    MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC =
    MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE =
    MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE;
static constexpr auto CUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE =
    MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE;

static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC          = MUDNN_ATTR_OPERATION_SDPA_FWD_QDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC          = MUDNN_ATTR_OPERATION_SDPA_FWD_KDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC          = MUDNN_ATTR_OPERATION_SDPA_FWD_VDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC          = MUDNN_ATTR_OPERATION_SDPA_FWD_ODESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC      = MUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC      = MUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_BIAS_DESC      = MUDNN_ATTR_OPERATION_SDPA_FWD_BIAS_DESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC  = MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC = MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK    = MUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_QDESC          = MUDNN_ATTR_OPERATION_SDPA_BWD_QDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_KDESC          = MUDNN_ATTR_OPERATION_SDPA_BWD_KDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_VDESC          = MUDNN_ATTR_OPERATION_SDPA_BWD_VDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_ODESC          = MUDNN_ATTR_OPERATION_SDPA_BWD_ODESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_DODESC         = MUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC      = MUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC      = MUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC  = MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC = MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC         = MUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC         = MUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC         = MUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC;
static constexpr auto CUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK    = MUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK;

inline cudnnStatus_t
cudnnCreate(cudnnHandle_t* handle) {
    return mudnnCreate(handle);
}

inline cudnnStatus_t
cudnnDestroy(cudnnHandle_t handle) {
    return mudnnDestroy(handle);
}

inline cudnnStatus_t
cudnnSetStream(cudnnHandle_t handle, musaStream_t stream) {
    return mudnnSetStream(handle, stream);
}

inline cudnnStatus_t
cudnnGetStream(cudnnHandle_t handle, musaStream_t* stream) {
    return mudnnGetStream(handle, stream);
}

template <typename StreamT>
inline cudnnStatus_t
cudnnSetStream(cudnnHandle_t handle, StreamT stream) {
    return mudnnSetStream(handle, reinterpret_cast<musaStream_t>(stream));
}

template <typename StreamT>
inline cudnnStatus_t
cudnnGetStream(cudnnHandle_t handle, StreamT* stream) {
    musaStream_t musa_stream = nullptr;
    auto status              = mudnnGetStream(handle, &musa_stream);
    if (stream != nullptr) {
        *stream = reinterpret_cast<StreamT>(musa_stream);
    }
    return status;
}

inline const char*
cudnnGetErrorString(cudnnStatus_t status) {
    return mudnnGetErrorString(status);
}

inline size_t
cudnnGetVersion() {
    return mudnnGetVersion();
}

inline size_t
cudnnGetCudartVersion() {
    return mudnnGetMusartVersion();
}

inline cudnnStatus_t
cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptor_type, cudnnBackendDescriptor_t* descriptor) {
    return mudnnBackendCreateDescriptor(descriptor_type, descriptor);
}

inline cudnnStatus_t
cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor) {
    return mudnnBackendDestroyDescriptor(descriptor);
}

inline cudnnStatus_t
cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor) {
    return mudnnBackendFinalize(descriptor);
}

inline cudnnStatus_t
cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attribute_name,
                         cudnnBackendAttributeType_t attribute_type,
                         int64_t element_count,
                         const void* array_of_elements) {
    return mudnnBackendSetAttribute(descriptor, attribute_name, attribute_type, element_count, array_of_elements);
}

inline cudnnStatus_t
cudnnBackendGetAttribute(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attribute_name,
                         cudnnBackendAttributeType_t attribute_type,
                         int64_t requested_element_count,
                         int64_t* element_count,
                         void* array_of_elements) {
    return mudnnBackendGetAttribute(
        descriptor, attribute_name, attribute_type, requested_element_count, element_count, array_of_elements);
}

inline cudnnStatus_t
cudnnBackendExecute(cudnnHandle_t handle, cudnnBackendDescriptor_t plan, cudnnBackendDescriptor_t variant_pack) {
    return mudnnBackendExecute(handle, plan, variant_pack);
}
