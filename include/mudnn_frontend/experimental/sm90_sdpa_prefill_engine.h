#pragma once

#include "oss_engine_interface.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mudnn_frontend {
namespace experimental {

inline const KernelSpec* lookup_kernel_spec(int, int) { return nullptr; }

class Sm90SdpaPrefillEngine : public IOssSdpaEngine {
 public:
  Sm90SdpaPrefillEngine() = default;
  ~Sm90SdpaPrefillEngine() override = default;

  Sm90SdpaPrefillEngine(const Sm90SdpaPrefillEngine&) = delete;
  Sm90SdpaPrefillEngine& operator=(const Sm90SdpaPrefillEngine&) = delete;
  Sm90SdpaPrefillEngine(Sm90SdpaPrefillEngine&&) noexcept = default;
  Sm90SdpaPrefillEngine& operator=(Sm90SdpaPrefillEngine&&) noexcept = default;

  error_t check_support(AttentionShape_t shape, int sm_version) override {
    return unsupported("SM90 OSS SDPA prefill engine", shape.d_qk, sm_version);
  }

  static int64_t get_workspace_size() {
    return IOssSdpaEngine::get_workspace_size();
  }

  error_t build() override {
    return {false,
            "SM90 OSS SDPA prefill engine is not supported by muDNN frontend",
            error_code_t::NOT_SUPPORTED};
  }

  error_t execute(int, int, int, int, int, int, void*,
                  const std::vector<int64_t>&, void*,
                  const std::vector<int64_t>&, void*,
                  const std::vector<int64_t>&, void*,
                  const std::vector<int64_t>&, void*,
                  const std::vector<int64_t>&, void*,
                  const std::vector<int64_t>&, void*, int, CompatCudaStream_t,
                  std::optional<float> = std::nullopt) override {
    return {false,
            "SM90 OSS SDPA prefill engine execution is not supported by "
            "muDNN frontend",
            error_code_t::NOT_SUPPORTED};
  }

 private:
  static error_t unsupported(const char* engine, uint32_t d, int sm_version) {
    return {false,
            std::string(engine) +
                " is not available in muDNN frontend (d=" + std::to_string(d) +
                ", sm_version=" + std::to_string(sm_version) + ")",
            error_code_t::GRAPH_NOT_SUPPORTED};
  }
};

}  // namespace experimental
}  // namespace mudnn_frontend
