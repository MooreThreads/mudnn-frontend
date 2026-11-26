"""Small shared helpers for JIT sources: the TVM-FFI export preamble and the
torch-dtype to mutlass-element mapping.

Ported from the upstream JIT utilities.  The only change is that torch is
imported lazily: the JIT facility itself must be reusable by any MUSA kernel,
and a caller that only compiles a torch-free kernel should not need torch to
import this module.
"""

from __future__ import annotations

from functools import lru_cache

from jinja2 import Template


TVM_HEADER = """
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

"""

EXPORT_FUNC_SOURCE = """
\n
TVM_FFI_DLL_EXPORT_TYPED_FUNC({{func_name}}, {{func_name}});
"""

EXPORT_FUNC = Template(EXPORT_FUNC_SOURCE)


@lru_cache(maxsize=1)
def dtype_torch2mutlass_map() -> dict:
    """``{torch dtype: mutlass element type}``.

    Built on first use so importing this module does not require torch.
    """
    import torch

    return {
        torch.float16: "mutlass::half_t",
        torch.bfloat16: "mutlass::bfloat16_t",
        torch.float32: "float",
        torch.float8_e4m3fn: "mutlass::float_e4m3_t",
        torch.float8_e5m2: "mutlass::float_e5m2_t",
    }


def mutlass_element(dtype) -> str:
    """The mutlass element type name for a torch dtype."""
    return dtype_torch2mutlass_map()[dtype]


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


__all__ = [
    "EXPORT_FUNC",
    "EXPORT_FUNC_SOURCE",
    "TVM_HEADER",
    "dtype_torch2mutlass_map",
    "maybe_contiguous",
    "mutlass_element",
]
