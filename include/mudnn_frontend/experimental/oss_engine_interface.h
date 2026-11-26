#pragma once

#include "attention_utils.h"
#include "../../mudnn_frontend.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mudnn_frontend {
namespace experimental {

using CompatCudaStream_t = void*;

class IOssSdpaEngine {
 public:
  virtual ~IOssSdpaEngine() = default;

  virtual error_t check_support(AttentionShape_t shape, int sm_version) = 0;
  virtual error_t build() = 0;

  virtual error_t execute(
      int batch, int heads_q, int heads_kv, int seq_q, int seq_kv, int d,
      void* d_Q, const std::vector<int64_t>& q_strides, void* d_K,
      const std::vector<int64_t>& k_strides, void* d_V,
      const std::vector<int64_t>& v_strides, void* d_O,
      const std::vector<int64_t>& o_strides, void* d_max,
      const std::vector<int64_t>& max_strides, void* d_sum_exp,
      const std::vector<int64_t>& se_strides, void* workspace, int device,
      CompatCudaStream_t stream,
      std::optional<float> user_attn_scale = std::nullopt) = 0;

  static int64_t get_workspace_size() { return 16; }
};

enum class RmsNormSiluDtype : uint8_t {
  BF16 = 0,
  FP8 = 1,
  NVFP4 = 2,
};

struct NormSiluShape_t {
  int C = 0;
  int num_tokens = 0;
  RmsNormSiluDtype output_dtype = RmsNormSiluDtype::BF16;
};

struct RmsNormSiluExtraParams {
  void* fp8_scale = nullptr;
  void* fp8_scale_inv = nullptr;
  void* fp8_amax = nullptr;
  void* nvfp4_scale_row = nullptr;
};

class IOssNormEngine {
 public:
  virtual ~IOssNormEngine() = default;

  virtual error_t check_support(NormSiluShape_t shape, int sm_version) = 0;
  virtual error_t build() = 0;
  virtual error_t execute(void* input, void* output, void* weight, void* bias,
                          int rows, int cols, float epsilon, void* workspace,
                          int device, CompatCudaStream_t stream,
                          const RmsNormSiluExtraParams& extra = {}) = 0;
  virtual int64_t get_workspace_size() const = 0;
};

template <typename... Args>
inline error_t compile_and_load_kernel(Args&&...) {
  return {false,
          "CUDA/NVRTC OSS engine compilation is not supported by muDNN "
          "frontend",
          error_code_t::NOT_SUPPORTED};
}

}  // namespace experimental
}  // namespace mudnn_frontend
