#pragma once

#include "../../mudnn_frontend.h"

#include <string>

namespace mudnn_frontend {
namespace experimental {
namespace detail {

inline bool nvrtc_is_loaded() { return false; }

inline std::string cuda_error_to_string(int err) {
  return "cuda error " + std::to_string(err);
}

}  // namespace detail
}  // namespace experimental
}  // namespace mudnn_frontend
