# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.
#
# Device-memory helper for frontend execution tests.
#
# The allocation, copy, stride, and free behavior follows muDNN's test utilities
# (CreateTensor, CopyToNdarray, SetTensorStride, and ReleaseDeviceData) on the
# direct runtime path used when DISABLE_MUDNN_TEST_MEMPOOL is set. It does not
# use that test memory pool and does not import mudnn_tests_lib.

import ctypes
import ctypes.util
import os
from pathlib import Path

import numpy as np

_MUSA_SUCCESS = 0
_HOST_TO_DEVICE = 1
_DEVICE_TO_HOST = 2

# Names match muDNN GetmuDNNType. Storage dtypes match GetNumpyType.
_NUMPY_DTYPES = {
    "int8": "int8",
    "qint8": "int8",
    "qint4": "int8",
    "int16": "int16",
    "int32": "int32",
    "int": "int64",
    "int64": "int64",
    "uint8": "uint8",
    "float4_e2m1fn_x2": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint": "uint64",
    "uint64": "uint64",
    "half": "float16",
    "float16": "float16",
    "float32": "float32",
    "float": "float64",
    "float64": "float64",
    "double": "float64",
    "bool": "bool",
}

_ML_DTYPES = {
    "bfloat16": "bfloat16",
    "float8_e4m3fn": "float8_e4m3fn",
    "float8_e5m2": "float8_e5m2",
}

_lib = None


class _DeviceTensor:
    def __init__(self, ptr, shape, dtype, nbytes, element_strides):
        self.ptr = ptr
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype
        self.nbytes = int(nbytes)
        self.element_strides = element_strides


def _library_candidates():
    seen = set()
    candidates = []

    def add(path):
        if path and path not in seen:
            seen.add(path)
            candidates.append(path)

    for env_name in ("MUSA_LIB", "MUDNN_BACKEND_LIB"):
        lib_dir = os.environ.get(env_name)
        if lib_dir:
            add(str(Path(lib_dir) / "libmusart.so"))
    add("/usr/local/musa/lib/libmusart.so")
    add(ctypes.util.find_library("musart"))
    add("libmusart.so")
    return candidates


def _runtime():
    global _lib
    if _lib is not None:
        return _lib

    errors = []
    for path in _library_candidates():
        try:
            lib = ctypes.CDLL(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        lib.musaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.musaMalloc.restype = ctypes.c_int
        lib.musaFree.argtypes = [ctypes.c_void_p]
        lib.musaFree.restype = ctypes.c_int
        lib.musaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        lib.musaMemcpy.restype = ctypes.c_int
        lib.musaDeviceSynchronize.argtypes = []
        lib.musaDeviceSynchronize.restype = ctypes.c_int
        lib.musaGetErrorString.argtypes = [ctypes.c_int]
        lib.musaGetErrorString.restype = ctypes.c_char_p
        _lib = lib
        return lib
    raise RuntimeError("failed to load libmusart.so: " + "; ".join(errors))


def _check(status, action):
    status = int(status)
    if status == _MUSA_SUCCESS:
        return
    message = _runtime().musaGetErrorString(status)
    if message is not None:
        message = message.decode("utf-8", errors="replace")
    else:
        message = f"musa error {status}"
    raise RuntimeError(f"{action} failed: {message}")


def _as_dtype(dtype):
    if isinstance(dtype, np.dtype):
        return dtype
    name = str(dtype)
    numpy_name = _NUMPY_DTYPES.get(name)
    if numpy_name is not None:
        return np.dtype(numpy_name)
    ml_name = _ML_DTYPES.get(name)
    if ml_name is None:
        return np.dtype(name)
    import ml_dtypes

    return np.dtype(getattr(ml_dtypes, ml_name))


def _dense_strides(shape):
    strides = [0] * len(shape)
    acc = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = acc
        acc *= int(shape[index])
    return tuple(strides)


def create_tensor(host_ptr, nbytes, shape, dtype):
    nbytes = int(nbytes)
    shape = tuple(int(dim) for dim in shape)
    dtype = _as_dtype(dtype)
    ptr = None
    if nbytes:
        allocated = ctypes.c_void_p()
        _check(
            _runtime().musaMalloc(ctypes.byref(allocated), ctypes.c_size_t(nbytes)),
            f"musaMalloc({nbytes})",
        )
        ptr = allocated.value
        host_ptr = int(host_ptr or 0)
        if host_ptr:
            _check(
                _runtime().musaMemcpy(
                    ctypes.c_void_p(ptr),
                    ctypes.c_void_p(host_ptr),
                    ctypes.c_size_t(nbytes),
                    _HOST_TO_DEVICE,
                ),
                f"musaMemcpy host-to-device ({nbytes})",
            )
            _check(_runtime().musaDeviceSynchronize(), "musaDeviceSynchronize")
    return _DeviceTensor(ptr, shape, dtype, nbytes, _dense_strides(shape))


def get_tensor_addr(tensor):
    return (int(tensor.ptr or 0),)


def set_tensor_stride(tensor, stride):
    stride = tuple(int(item) for item in stride)
    if len(stride) != len(tensor.shape):
        raise ValueError(
            f"stride rank {len(stride)} does not match shape rank {len(tensor.shape)}"
        )
    tensor.element_strides = stride


def copy_to_ndarray(tensor, nbytes):
    nbytes = int(nbytes)
    if nbytes == 0 or not tensor.ptr:
        return np.zeros(tensor.shape, dtype=tensor.dtype)
    host = np.empty(nbytes, dtype=np.uint8)
    _check(
        _runtime().musaMemcpy(
            ctypes.c_void_p(host.ctypes.data),
            ctypes.c_void_p(tensor.ptr),
            ctypes.c_size_t(nbytes),
            _DEVICE_TO_HOST,
        ),
        f"musaMemcpy device-to-host ({nbytes})",
    )
    element_strides = tensor.element_strides
    if element_strides is not None and all(item == 0 for item in element_strides):
        element_strides = _dense_strides(tensor.shape)
    byte_strides = None
    if element_strides is not None:
        byte_strides = tuple(
            int(item) * tensor.dtype.itemsize for item in element_strides
        )
    return np.ndarray(
        tensor.shape, dtype=tensor.dtype, buffer=host, strides=byte_strides
    )


def release_device_data(tensor):
    if tensor.ptr:
        _check(_runtime().musaFree(ctypes.c_void_p(tensor.ptr)), "musaFree")
        tensor.ptr = None
