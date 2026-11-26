// Torch-free device self-check for the mudnn.jit toolchain.
//
// The question this answers is "does a kernel compiled by mudnn.jit actually
// execute on a MUSA device?", which is a different question from "does it
// link and load". It needs no framework: the kernel allocates, launches,
// copies back and verifies entirely through the MUSA runtime, so the only
// argument across the FFI boundary is a plain integer.
//
// Returns 0 on success. A non-zero value identifies the failing stage, so a
// failure is diagnosable from the return value alone:
//
//   0   ok
//   1   no MUSA device
//   2   musaMalloc failed
//   3   host->device copy failed
//   4   kernel launch failed
//   5   device->host copy failed
//   6   at least one element is wrong; |max_error| is in *max_error
//   7   musaFree failed
//   8   the device is older than the compiled architecture; not runnable here
//
// ``out_device_cc`` receives major * 10 + minor of the device the work ran on,
// so the caller can confirm it matches the architecture it compiled for.

#include <musa_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "dlpack/dlpack.h"

namespace ffi = tvm::ffi;

namespace {

__global__ void saxpy_selfcheck_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ c,
                                       float alpha,
                                       int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] * alpha + b[i];
  }
}

constexpr int64_t kElements = 4096;
constexpr float kAlpha = 1.75f;

}  // namespace

int64_t mudnn_jit_selfcheck(int64_t device_index, int64_t expected_cc,
                            int64_t* out_device_cc, double* out_max_error) {
  int device_count = 0;
  if (musaGetDeviceCount(&device_count) != musaSuccess || device_count <= 0) {
    return 1;
  }

  int device = static_cast<int>(device_index);
  if (device < 0 || device >= device_count) {
    return 1;
  }
  if (musaSetDevice(device) != musaSuccess) {
    return 1;
  }
  musaDeviceProp prop{};
  if (musaGetDeviceProperties(&prop, device) != musaSuccess) {
    return 1;
  }
  const int64_t device_cc =
      static_cast<int64_t>(prop.major) * 10 + prop.minor;
  if (out_device_cc != nullptr) {
    *out_device_cc = device_cc;
  }

  // A cubin only runs on the architecture it was built for (or a newer one in
  // compatibility mode), so report "not runnable here" instead of failing a
  // launch we already know cannot work. This is the same device-vs-build
  // agreement the real engines check in check_support.
  if (device_cc < expected_cc) {
    return 8;
  }

  const size_t bytes = static_cast<size_t>(kElements) * sizeof(float);
  float* dev_a = nullptr;
  float* dev_b = nullptr;
  float* dev_c = nullptr;
  if (musaMalloc(reinterpret_cast<void**>(&dev_a), bytes) != musaSuccess ||
      musaMalloc(reinterpret_cast<void**>(&dev_b), bytes) != musaSuccess ||
      musaMalloc(reinterpret_cast<void**>(&dev_c), bytes) != musaSuccess) {
    musaFree(dev_a);
    musaFree(dev_b);
    musaFree(dev_c);
    return 2;
  }

  float* host_a = new float[kElements];
  float* host_b = new float[kElements];
  float* host_c = new float[kElements];
  for (int64_t i = 0; i < kElements; ++i) {
    // Deterministic, wide enough to expose a wrong scale or a stale read.
    host_a[i] = static_cast<float>(i % 251) * 0.5f - 60.0f;
    host_b[i] = static_cast<float>(i % 97) * -0.25f + 12.0f;
    host_c[i] = -12345.0f;
  }

  int64_t status = 0;
  if (musaMemcpy(dev_a, host_a, bytes, musaMemcpyHostToDevice) != musaSuccess ||
      musaMemcpy(dev_b, host_b, bytes, musaMemcpyHostToDevice) != musaSuccess) {
    status = 3;
    goto cleanup;
  }

  {
    tvm::ffi::MUSADeviceGuard guard(device);
    musaStream_t stream = static_cast<musaStream_t>(TVMFFIEnvGetStream(kDLExtDev, device));
    constexpr int kThreads = 256;
    const int64_t blocks = (kElements + kThreads - 1) / kThreads;
    saxpy_selfcheck_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
        dev_a, dev_b, dev_c, kAlpha, kElements);
    if (musaGetLastError() != musaSuccess) {
      status = 4;
      goto cleanup;
    }
    if (musaStreamSynchronize(stream) != musaSuccess) {
      status = 4;
      goto cleanup;
    }
  }

  if (musaMemcpy(host_c, dev_c, bytes, musaMemcpyDeviceToHost) != musaSuccess) {
    status = 5;
    goto cleanup;
  }

  {
    double max_error = 0.0;
    for (int64_t i = 0; i < kElements; ++i) {
      const double expected = static_cast<double>(host_a[i]) * kAlpha + host_b[i];
      const double error = std::fabs(static_cast<double>(host_c[i]) - expected);
      if (error > max_error) {
        max_error = error;
      }
    }
    if (out_max_error != nullptr) {
      *out_max_error = max_error;
    }
    // Exact: the operands are small integers scaled by powers of two, so the
    // fused multiply-add is bit-exact and no tolerance is warranted.
    if (max_error != 0.0) {
      status = 6;
    }
  }

cleanup:
  if (musaFree(dev_a) != musaSuccess || musaFree(dev_b) != musaSuccess ||
      musaFree(dev_c) != musaSuccess) {
    if (status == 0) {
      status = 7;
    }
  }
  delete[] host_a;
  delete[] host_b;
  delete[] host_c;
  return status;
}

// Two plain outputs instead of a tensor, so the call needs no framework at all.
void mudnn_jit_selfcheck_fn(ffi::TensorView out, int64_t device_index,
                            int64_t expected_cc) {
  TVM_FFI_ICHECK(out.device().device_type != kDLExtDev)
      << "the self-check writes its result to host memory";
  TVM_FFI_ICHECK_EQ(out.dtype(), (DLDataType{kDLInt, 64, 1}))
      << "out must be int64[2]: [status, device_cc]";

  int64_t device_cc = -1;
  double max_error = -1.0;
  const int64_t status =
      mudnn_jit_selfcheck(device_index, expected_cc, &device_cc, &max_error);

  auto* data = static_cast<int64_t*>(out.data_ptr());
  data[0] = status;
  data[1] = device_cc;

  std::printf(
      "[mudnn.jit selfcheck] device=%lld status=%lld device_cc=%lld max_error=%.17g\n",
      static_cast<long long>(device_index), static_cast<long long>(status),
      static_cast<long long>(device_cc), max_error);
  std::fflush(stdout);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mudnn_jit_selfcheck_fn, mudnn_jit_selfcheck_fn);
