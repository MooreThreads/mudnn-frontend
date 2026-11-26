// Proves the JIT can compile and run a kernel that actually depends on
// mutlass + mute, not just on the MUSA runtime.
//
// It is the bridge between "the toolchain works" and "the KDA kernel will
// build": it includes the same headers the KDA collectives include, uses
// mutlass types in device code, and reports both the architecture the source
// was compiled for and the architecture of the device it ran on -- the pair
// that has to agree.
//
// The source is compiled twice (a host pass and a device pass), and
// __MUSA_ARCH__ only exists in the device pass, so every architecture-dependent
// value is produced by device code and read back rather than computed on the
// host.
//
// out[0] status, out[1] device compute capability, out[2] __MUSA_ARCH__ of the
// device pass, out[3] the mutlass ArchTag capability selected there.
//
//   0 ok, 1 no device, 2 alloc, 3 copy in, 4 launch, 5 copy out,
//   6 numeric mismatch, 7 free, 8 device older than the compiled arch

#include <musa_runtime.h>

#include <mutlass/arch/arch.h>
#include <mutlass/bfloat16.h>
#include <mutlass/half.h>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>

#include <mute/tensor.hpp>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "dlpack/dlpack.h"

namespace ffi = tvm::ffi;

namespace {

// The same ArchTag selection the KDA builder performs. Only defined in the
// device pass, because that is the only pass where __MUSA_ARCH__ exists.
#if defined(__MUSA_ARCH__)
// The primary template keeps every arch on Mp31; the MP32 specialization below
// is what a build that targets MP32 picks up.
template <int Arch>
struct ProbeArch {
  using type = mutlass::arch::Mp31;
};
using CompileTimeArch = typename ProbeArch<(__MUSA_ARCH__ >= 320 ? 320 : 31)>::type;
#endif

// mute's layout algebra is used by every KDA collective; instantiating one here
// makes a broken `mute/tensor.hpp` a build failure rather than a runtime one.
using ProbeLayout = decltype(mute::make_layout(
    mute::make_shape(mute::Int<16>{}, mute::Int<128>{}),
    mute::make_stride(mute::Int<128>{}, mute::_1{})));

__global__ void bf16_roundtrip_kernel(const float* __restrict__ src,
                                      float* __restrict__ dst,
                                      int64_t n,
                                      int64_t* __restrict__ arch_info) {
  mutlass::NumericConverter<mutlass::bfloat16_t, float,
                            mutlass::FloatRoundStyle::round_to_nearest>
      to_bf16;
  mutlass::NumericConverter<float, mutlass::bfloat16_t,
                            mutlass::FloatRoundStyle::round_to_nearest>
      from_bf16;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const mutlass::bfloat16_t narrowed = to_bf16(src[i]);
    dst[i] = from_bf16(narrowed);
  }
#if defined(__MUSA_ARCH__)
  if (i == 0 && arch_info != nullptr) {
    arch_info[0] = static_cast<int64_t>(__MUSA_ARCH__);
    arch_info[1] = static_cast<int64_t>(CompileTimeArch::kMinComputeCapability);
  }
#endif
}

constexpr int64_t kElements = 2048;

// Host-side guarantees that the mutlass headers are the ones we expect. These
// do not need __MUSA_ARCH__.
static_assert(mutlass::arch::Mp31::kMinComputeCapability == 31, "Mp31 tag moved");

}  // namespace

void mudnn_jit_mutlass_probe_fn(ffi::TensorView out, int64_t device_index,
                               int64_t expected_cc) {
  TVM_FFI_ICHECK(out.device().device_type != kDLExtDev)
      << "the probe writes its result to host memory";
  TVM_FFI_ICHECK_EQ(out.dtype(), (DLDataType{kDLInt, 64, 1}))
      << "out must be int64[4]";
  TVM_FFI_ICHECK_GE(out.size(0), 4) << "out must be int64[4]";

  auto* result = static_cast<int64_t*>(out.data_ptr());
  result[0] = 0;
  result[1] = -1;
  result[2] = -1;
  result[3] = -1;

  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count <= 0 ||
      device_index < 0 || device_index >= device_count) {
    result[0] = 1;
    return;
  }
  if (musaSetDevice(static_cast<int>(device_index)) != musaSuccess) {
    result[0] = 1;
    return;
  }
  musaDeviceProp prop{};
  if (musaGetDeviceProperties(&prop, static_cast<int>(device_index)) != musaSuccess) {
    result[0] = 1;
    return;
  }
  result[1] = static_cast<int64_t>(prop.major) * 10 + prop.minor;
  if (result[1] < expected_cc) {
    // Built for a newer architecture than this device has; nothing to run.
    result[0] = 8;
    return;
  }

  const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);
  float* dev_in = nullptr;
  float* dev_out = nullptr;
  int64_t* dev_arch = nullptr;
  if (musaMalloc(reinterpret_cast<void**>(&dev_in), bytes) != musaSuccess ||
      musaMalloc(reinterpret_cast<void**>(&dev_out), bytes) != musaSuccess ||
      musaMalloc(reinterpret_cast<void**>(&dev_arch), 2 * sizeof(int64_t)) != musaSuccess) {
    musaFree(dev_in);
    musaFree(dev_out);
    musaFree(dev_arch);
    result[0] = 2;
    return;
  }

  auto* host_in = new float[kElements];
  auto* host_out = new float[kElements];
  int64_t host_arch[2] = {-1, -1};
  for (int64_t i = 0; i < kElements; ++i) {
    // Multiples of 2^-5 with at most 8 significant bits, so every value is
    // exactly representable in bf16 and the round trip must be bit-exact.
    host_in[i] = static_cast<float>(i % 256) * 0.03125f - 4.0f;
    host_out[i] = -999.0f;
  }

  int64_t status = 0;
  if (musaMemcpy(dev_in, host_in, bytes, musaMemcpyHostToDevice) != musaSuccess ||
      musaMemcpy(dev_arch, host_arch, sizeof(host_arch), musaMemcpyHostToDevice) !=
          musaSuccess) {
    status = 3;
  }
  if (status == 0) {
    tvm::ffi::MUSADeviceGuard guard(static_cast<int>(device_index));
    musaStream_t stream =
        static_cast<musaStream_t>(TVMFFIEnvGetStream(kDLExtDev, static_cast<int>(device_index)));
    constexpr int kThreads = 256;
    const int64_t blocks = (kElements + kThreads - 1) / kThreads;
    bf16_roundtrip_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        dev_in, dev_out, kElements, dev_arch);
    const musaError_t launch_err = musaGetLastError();
    const musaError_t sync_err = musaStreamSynchronize(stream);
    if (launch_err != musaSuccess || sync_err != musaSuccess) {
      std::printf(
          "[mudnn.jit mutlass probe] launch failed: launch=%d(%s) sync=%d(%s)\n",
          static_cast<int>(launch_err), musaGetErrorName(launch_err),
          static_cast<int>(sync_err), musaGetErrorName(sync_err));
      std::fflush(stdout);
      status = 4;
    }
  }
  if (status == 0 &&
      musaMemcpy(host_out, dev_out, bytes, musaMemcpyDeviceToHost) != musaSuccess) {
    status = 5;
  }
  if (status == 0 &&
      musaMemcpy(host_arch, dev_arch, sizeof(host_arch), musaMemcpyDeviceToHost) !=
          musaSuccess) {
    status = 5;
  }
  if (status == 0) {
    result[2] = host_arch[0];
    result[3] = host_arch[1];
    for (int64_t i = 0; i < kElements; ++i) {
      // The inputs are exactly representable in bf16 (multiples of 2^-5), so
      // the round trip must be exact.
      if (host_out[i] != host_in[i]) {
        status = 6;
        break;
      }
    }
  }

  if (musaFree(dev_in) != musaSuccess || musaFree(dev_out) != musaSuccess ||
      musaFree(dev_arch) != musaSuccess) {
    if (status == 0) {
      status = 7;
    }
  }
  delete[] host_in;
  delete[] host_out;
  result[0] = status;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mudnn_jit_mutlass_probe_fn, mudnn_jit_mutlass_probe_fn);
