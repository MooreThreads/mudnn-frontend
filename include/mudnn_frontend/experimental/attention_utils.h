#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace mudnn_frontend {
namespace experimental {

struct AttentionShape_t {
  uint32_t b = 0;
  uint32_t q_h = 0;
  uint32_t k_h = 0;
  uint32_t v_h = 0;
  uint32_t s_q = 0;
  uint32_t s_kv = 0;
  uint32_t d_qk = 0;
  uint32_t d_v = 0;
};

struct AttentionDescriptor_t {
  uint32_t b = 0;
  uint32_t q_h = 0;
  uint32_t k_h = 0;
  uint32_t v_h = 0;
  uint32_t s_q = 0;
  uint32_t s_kv = 0;
  uint32_t d_qk = 0;
  uint32_t d_v = 0;
  uint16_t q_heads_per_k = 0;
  uint16_t q_heads_per_v = 0;
  uint16_t min_q_heads_per_kv = 0;
};

struct FastDivisor_t {
  uint32_t val = 0;
  uint32_t shr = 0;
  uint32_t mul = 0;
};

struct tensor_descriptor {
  static constexpr int MAX_DIMS = 12;
  int64_t num_dims = 0;
  int64_t dims[MAX_DIMS] = {};
  int64_t strides[MAX_DIMS] = {};
};

struct KernelSpec {
  const char* source = nullptr;
  size_t source_len = 0;
  const char* flags_raw = nullptr;
  size_t flags_len = 0;
  const char* kernel_name = nullptr;
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;
  int smem_bytes = 0;
};

inline int div_up(int a, int b) { return (a + b - 1) / b; }

inline int find_log_2_floor(uint32_t x) {
  if (x <= 1) {
    return 0;
  }
  int result = 0;
  while ((1u << (result + 1)) <= x) {
    ++result;
  }
  return result;
}

inline FastDivisor_t make_fast_divisor(uint32_t divisor) {
  FastDivisor_t d;
  d.val = divisor;
  if (divisor <= 1) {
    d.mul = 0x80000000u;
    return d;
  }
  uint32_t x2 = 2u * divisor;
  int a = 0;
  for (uint32_t tmp = x2; tmp > 1; tmp >>= 1) {
    ++a;
  }
  if (x2 & (x2 - 1)) {
    ++a;
  }
  const uint32_t p = 31 + static_cast<uint32_t>(a);
  d.mul = static_cast<uint32_t>(((1ULL << p) + static_cast<uint64_t>(x2) - 1) /
                                static_cast<uint64_t>(x2));
  d.shr = p - 32;
  return d;
}

inline std::vector<std::string> parse_flags_string(const char* data,
                                                   size_t len) {
  std::vector<std::string> flags;
  if (data == nullptr || len == 0) {
    return flags;
  }
  std::istringstream stream(std::string(data, len));
  std::string line;
  while (std::getline(stream, line)) {
    const auto start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
      continue;
    }
    const auto end = line.find_last_not_of(" \t\r\n");
    flags.push_back(line.substr(start, end - start + 1));
  }
  return flags;
}

}  // namespace experimental
}  // namespace mudnn_frontend
