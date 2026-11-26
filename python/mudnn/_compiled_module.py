"""Compatibility module matching cudnn-frontend's pybind export surface.

The implementation is Python/ctypes because the native library is muDNN, but
the objects and entry points intentionally match the 0.0.1 compiled module.
"""

from mudnn_frontend import (  # noqa: F401
    attention_implementation,
    backend_version,
    backend_version_string,
    behavior_note,
    build_plan_policy,
    create_device_properties,
    create_handle,
    create_kernel_cache,
    cudnnGraphNotSupportedError,
    data_type,
    destroy_handle,
    get_last_error_string,
    get_stream,
    heur_mode,
    knob,
    knob_type,
    numerical_note,
    norm_forward_phase,
    reduction_mode,
    set_stream,
    tensor,
    tensor_reordering,
)
from mudnn_frontend import DiagonalAlignment as diagonal_alignment
from mudnn_frontend import Graph as pygraph
from mudnn_frontend import MoeGroupedMatmulMode as moe_grouped_matmul_mode
from mudnn_frontend import ReshapeMode as reshape_mode
from mudnn_frontend import ScalarType as scalar_type
from mudnn_frontend.graph import get_data_ptr as _get_data_ptr


def _set_dlhandle_cudnn(_handle):
    """cuDNN's module installs a dlopen handle; ctypes loads muDNN itself."""


def _set_dlhandle_mudnn(_handle):
    """Accept a preloaded muDNN handle for API symmetry."""

