// Minimal MUSA kernel used to prove the mudnn.jit toolchain end to end:
// templated-free, single translation unit, no dependency on any project
// header. It only needs what the JIT always provides -- the MUSA runtime, the
// TVM-FFI C env API (which gives the current stream) and DLPack.
//
//   c[i] = a[i] * alpha + b[i]
//
// The point is not the arithmetic; it is that a source compiled at runtime by
// mcc through ninja, loaded by tvm_ffi, and called with torch tensors produces
// the right answer on both MUSA architectures.

#include <musa_runtime.h>

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/musa/device_guard.h>
#include <tvm/ffi/function.h>

#include <cstdint>

#include "dlpack/dlpack.h"

namespace ffi = tvm::ffi;

namespace {

// Local so this file depends on nothing but DLPack; the project's own
// dtype_utils.hpp defines the same constants for the real kernels.
constexpr DLDataType kFloat32 = DLDataType{kDLFloat, 32, 1};

__global__ void vec_add_scale_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c,
                                     float alpha,
                                     int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] * alpha + b[i];
  }
}

}  // namespace

void mudnn_jit_smoke_vec_add(ffi::TensorView a,
                             ffi::TensorView b,
                             ffi::TensorView c,
                             double alpha) {
  TVM_FFI_ICHECK(a.ndim() == 1 && b.ndim() == 1 && c.ndim() == 1)
      << "vec_add expects 1-D tensors";
  TVM_FFI_ICHECK_EQ(a.size(0), b.size(0));
  TVM_FFI_ICHECK_EQ(a.size(0), c.size(0));
  TVM_FFI_ICHECK_EQ(a.dtype(), kFloat32) << "a must be float32";
  TVM_FFI_ICHECK_EQ(b.dtype(), kFloat32) << "b must be float32";
  TVM_FFI_ICHECK_EQ(c.dtype(), kFloat32) << "c must be float32";
  TVM_FFI_ICHECK_EQ(a.device().device_type, kDLExtDev) << "a must be a MUSA tensor";

  const int64_t n = a.size(0);
  if (n == 0) {
    return;
  }

  constexpr int kThreads = 256;
  const int64_t blocks = (n + kThreads - 1) / kThreads;

  tvm::ffi::MUSADeviceGuard guard(a.device().device_id);
  musaStream_t stream =
      static_cast<musaStream_t>(TVMFFIEnvGetStream(kDLExtDev, a.device().device_id));

  vec_add_scale_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
      static_cast<const float*>(a.data_ptr()),
      static_cast<const float*>(b.data_ptr()),
      static_cast<float*>(c.data_ptr()),
      static_cast<float>(alpha),
      n);

  TVM_FFI_ICHECK_EQ(musaGetLastError(), musaSuccess) << "vec_add kernel launch failed";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mudnn_jit_smoke_vec_add, mudnn_jit_smoke_vec_add);
