#pragma once

#include "mudnn_frontend.h"

#include <cstdlib>

namespace mudnn_frontend {

inline const char* get_environment(const char* name) { return std::getenv(name); }

}  // namespace mudnn_frontend
