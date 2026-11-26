#pragma once

#include <musa_runtime.h>
#include <mute/config.hpp>
#include <mutlass/array.h>

namespace mate::flat::kda {

// The chunk scan rounds its intermediates; seed gradients need a FP32 path
// when beta approaches two and the state transition has negative eigenvalues.
template <class Collective>
__global__ void
chunk_kda_initial_state_grad_kernel(typename Collective::Params p,
                                    float *output, float const *d_final_state,
                                    typename Collective::Element *d_states) {
  using Element = typename Collective::Element;
  using ElementGate = typename Collective::ElementGate;
  constexpr int D = 128;
  constexpr float kLog2E = 1.4426950408889634074f;
  int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32;
  int group = int(blockIdx.x) % (D / 4);
  int head = (int(blockIdx.x) / (D / 4)) % p.H;
  int seq = int(blockIdx.x) / (D / 4 * p.H);
  int value = group * 4 + warp, d0 = lane * 4;
  int begin = p.cu_seqlens ? int(p.cu_seqlens[seq]) : seq * p.T;
  int end = p.cu_seqlens ? int(p.cu_seqlens[seq + 1]) : begin + p.T;
  int64_t state_offset = ((int64_t(seq) * p.H + head) * D + value) * D + d0;
  float4 acc =
      d_final_state
          ? *reinterpret_cast<float4 const *>(d_final_state + state_offset)
          : make_float4(0.f, 0.f, 0.f, 0.f);
  float a = 1.f;
  mutlass::Array<float, 4> dt{};
  if constexpr (Collective::SafeGate) {
    a = __musa_exp2_f(p.ptr_A_log[head] * kLog2E);
    dt = *reinterpret_cast<mutlass::Array<float, 4> const *>(p.ptr_dt_bias +
                                                             head * D + d0);
  }
  for (int token = end - 1; token >= begin; --token) {
    if (d_states && (token == end - 1 || (token - begin + 1) % 16 == 0)) {
      int base = p.cu_seqlens ? (begin + 15) / 16 + seq : seq * ((p.T + 15) / 16);
      int row = base + (token - begin) / 16;
      int64_t offset = ((int64_t(row) * p.H + head) * D + value) * D + d0;
      mutlass::Array<Element, 4> packed;
      packed[0] = Element(acc.x);
      packed[1] = Element(acc.y);
      packed[2] = Element(acc.z);
      packed[3] = Element(acc.w);
      *reinterpret_cast<mutlass::Array<Element, 4> *>(d_states + offset) =
          packed;
    }
    int64_t off = (int64_t(token) * p.H + head) * D + d0;
    auto q =
        *reinterpret_cast<mutlass::Array<Element, 4> const *>(p.ptr_q + off);
    auto k =
        *reinterpret_cast<mutlass::Array<Element, 4> const *>(p.ptr_k + off);
    auto g = *reinterpret_cast<mutlass::Array<ElementGate, 4> const *>(p.ptr_g +
                                                                       off);
    float qf[4], kf[4], af[4];
    float qq = 0.f, kk = 0.f;
    MUTE_UNROLL
    for (int j = 0; j < 4; ++j) {
      qf[j] = float(q[j]);
      kf[j] = float(k[j]);
      qq = fmaf(qf[j], qf[j], qq);
      kk = fmaf(kf[j], kf[j], kk);
      float raw = float(g[j]);
      if constexpr (Collective::SafeGate)
        raw =
            p.lower_bound / (1.f + __musa_exp2_f(-a * (raw + dt[j]) * kLog2E));
      if constexpr (!Collective::SafeGate && Collective::GateIsLinear)
        af[j] = fmaxf(raw,1e-10f);
      else
        af[j] = __musa_exp2_f(
            raw *
            (Collective::SafeGate || Collective::GateIsLog ? kLog2E : 1.f));
    }
    if constexpr (Collective::NormalizeQK) {
      MUTE_UNROLL
      for (int mask = 16; mask > 0; mask /= 2) {
        qq += __shfl_xor_sync(0xffffffffu, qq, mask, 32);
        kk += __shfl_xor_sync(0xffffffffu, kk, mask, 32);
      }
      float rq = rsqrtf(qq + 1e-6f), rk = rsqrtf(kk + 1e-6f);
      MUTE_UNROLL
      for (int j = 0; j < 4; ++j) {
        qf[j] *= rq;
        kf[j] *= rk;
      }
    }
    float beta = float(p.ptr_beta[int64_t(token) * p.H + head]);
    if constexpr (Collective::UseBetaSigmoid)
      beta = (Collective::AllowNegEigval ? 2.f : 1.f) /
             (1.f + __musa_exp2_f(-beta * kLog2E));
    float dy =
        float(p.ptr_dO[(int64_t(token) * p.H + head) * D + value]) * p.scale;
    float s[4] = {acc.x, acc.y, acc.z, acc.w};
    float dot = 0.f;
    MUTE_UNROLL
    for (int j = 0; j < 4; ++j) {
      s[j] = fmaf(dy, qf[j], s[j]);
      dot = fmaf(s[j], kf[j], dot);
    }
    MUTE_UNROLL
    for (int mask = 16; mask > 0; mask /= 2)
      dot += __shfl_xor_sync(0xffffffffu, dot, mask, 32);
    MUTE_UNROLL
    for (int j = 0; j < 4; ++j)
      s[j] = fmaf(-beta * dot, kf[j], s[j]) * af[j];
    acc = make_float4(s[0], s[1], s[2], s[3]);
  }
  *reinterpret_cast<float4 *>(output + state_offset) = acc;
}

} // namespace mate::flat::kda
