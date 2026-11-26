"""cuDNN Python package compatibility wrapper for muDNN frontend."""

import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

from mudnn_frontend import *  # noqa: F401,F403
from mudnn_frontend import __all__ as _mudnn_frontend_all
from mudnn_frontend import __version__ as __version__

__all__ = list(_mudnn_frontend_all)

from .datatypes import _is_torch_tensor, _library_type  # noqa: E402

try:
    from .deepseek_sparse_attention import DSA as DSA

    __all__.append("DSA")
except Exception:
    pass

try:
    from .native_sparse_attention import NSA as NSA

    __all__.append("NSA")
except Exception:
    pass

try:
    from .causal_conv1d import (
        causal_conv1d_backward,
        causal_conv1d_forward,
        causal_conv1d_nwh_backward,
        causal_conv1d_nwh_forward,
    )

    __all__ += [
        "causal_conv1d_backward",
        "causal_conv1d_forward",
        "causal_conv1d_nwh_backward",
        "causal_conv1d_nwh_forward",
    ]
except Exception:
    pass

try:
    from .b2b_causal_conv1d import (
        b2b_causal_conv1d_backward,
        b2b_causal_conv1d_forward,
    )

    __all__ += ["b2b_causal_conv1d_backward", "b2b_causal_conv1d_forward"]
except Exception:
    pass

try:
    from . import ops as ops

    __all__.append("ops")
except Exception:
    pass

try:
    from . import linear_attention as linear_attention

    __all__.append("linear_attention")
except Exception:
    pass

try:
    # ``mudnn.csa`` is a lazy package; importing it here does not import
    # PyTorch/TileLang and therefore keeps the base compatibility import light.
    from . import csa as csa
    from .csa import CSA as CSA

    __all__ += ["csa", "CSA"]
except Exception:
    pass


def __getattr__(name):
    # `import mudnn; mudnn.fla.accelerate_fla()` works like `import mudnn.fla`,
    # deferred so `import mudnn` never pays the FLA import.
    if name == "fla":
        import importlib

        module = importlib.import_module(".fla", __name__)
        globals()["fla"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

try:
    from . import grouped_gemm as grouped_gemm

    __all__.append("grouped_gemm")
except Exception:
    pass

try:
    from . import discrete_grouped_gemm as discrete_grouped_gemm

    __all__.append("discrete_grouped_gemm")
except Exception:
    pass


def is_windows():
    return sys.platform.startswith("win")


module_name = ".Release._compiled_module" if is_windows() else "._compiled_module"

symbols_to_import = [
    "backend_version",
    "backend_version_string",
    "get_last_error_string",
    "destroy_handle",
    "norm_forward_phase",
    "reduction_mode",
    "behavior_note",
    "knob_type",
    "create_handle",
    "create_kernel_cache",
    "create_device_properties",
    "get_stream",
    "numerical_note",
    "set_stream",
    "build_plan_policy",
    "data_type",
    "tensor_reordering",
    "heur_mode",
    "pygraph",
    "tensor",
    "knob",
    "cudnnGraphNotSupportedError",
    "diagonal_alignment",
    "attention_implementation",
    "moe_grouped_matmul_mode",
    "scalar_type",
    "reshape_mode",
]


_pybind_module = importlib.import_module(module_name, package=__name__)
for _symbol_name in symbols_to_import:
    globals()[_symbol_name] = getattr(_pybind_module, _symbol_name)


def load_cudnn():
    """Load the muDNN backend used by this compatibility package.

    Like upstream ``load_cudnn()``, this function exists for explicit loader
    initialization and returns ``None`` on success.
    """
    from mudnn_frontend import _backend

    _backend.get_library()
    return None


__all__ += ["is_windows", "load_cudnn", "module_name", "symbols_to_import"]


_OPTIONAL_PREFIXES = (
    "mudnn.deepseek_sparse_attention",
    "mudnn.discrete_grouped_gemm",
    "mudnn.gemm_amax",
    "mudnn.gemm_dsrelu",
    "mudnn.gemm_srelu",
    "mudnn.gemm_swiglu",
    "mudnn.grouped_gemm",
    "mudnn.native_sparse_attention",
    "mudnn.rmsnorm_rht_amax",
    "mudnn.sdpa.bwd",
    "mudnn.sdpa.fmha_utils",
    "mudnn.sdpa.fwd",
    "mudnn.sdpa.utils",
)


def _is_optional_module(fullname: str) -> bool:
    return any(
        fullname == prefix or fullname.startswith(prefix + ".")
        for prefix in _OPTIONAL_PREFIXES
    )


def _unsupported_optional_api(module_name: str, attr_name: str):
    message = f"{module_name}.{attr_name} is not supported by muDNN frontend yet"
    if attr_name and attr_name[0].isupper():
        try:
            from mudnn.api_base import APIBase
        except Exception:
            APIBase = object

        class UnsupportedOptionalApi(APIBase):
            def __init__(self, *args, **kwargs):
                if hasattr(super(), "__init__"):
                    super().__init__()
                self.args = args
                self.kwargs = kwargs

            def check_support(self):
                raise NotImplementedError(message)

            def compile(self):
                raise NotImplementedError(message)

            def execute(self, *args, **kwargs):
                raise NotImplementedError(message)

        UnsupportedOptionalApi.__name__ = attr_name
        UnsupportedOptionalApi.__qualname__ = attr_name
        UnsupportedOptionalApi.__module__ = module_name
        return UnsupportedOptionalApi

    def unsupported(*args, **kwargs):
        raise NotImplementedError(message)

    unsupported.__name__ = attr_name
    unsupported.__qualname__ = attr_name
    unsupported.__module__ = module_name
    return unsupported


_TOP_LEVEL_OPTIONAL_APIS = (
    "DiscreteGroupedGemmDswigluSm100",
    "DiscreteGroupedGemmSwigluSm100",
    "GemmAmaxSm100",
    "GemmDsreluSm100",
    "GemmSreluSm100",
    "GemmSwigluSm100",
    "GroupedGemmDsreluSm100",
    "GroupedGemmDgluSm100",
    "GroupedGemmDswigluSm100",
    "GroupedGemmGluSm100",
    "GroupedGemmGluHadamardSm100",
    "GroupedGemmQuantSm100",
    "GroupedGemmSreluSm100",
    "GroupedGemmSwigluSm100",
    "GroupedGemmWgradSm100",
    "RmsNormRhtAmaxSm100",
    "SdpabwdSm100D256",
    "SdpafwdSm100D256",
    "discrete_grouped_gemm_dswiglu_wrapper_sm100",
    "discrete_grouped_gemm_swiglu_wrapper_sm100",
    "gemm_amax_wrapper_sm100",
    "gemm_dsrelu_wrapper_sm100",
    "gemm_srelu_wrapper_sm100",
    "gemm_swiglu_wrapper_sm100",
    "grouped_gemm_dsrelu_wrapper_sm100",
    "grouped_gemm_dglu_wrapper_sm100",
    "grouped_gemm_dswiglu_wrapper_sm100",
    "grouped_gemm_glu_wrapper_sm100",
    "grouped_gemm_glu_hadamard_wrapper_sm100",
    "grouped_gemm_quant_wrapper_sm100",
    "grouped_gemm_srelu_wrapper_sm100",
    "grouped_gemm_swiglu_wrapper_sm100",
    "grouped_gemm_wgrad_wrapper_sm100",
    "rmsnorm_rht_amax_wrapper_sm100",
    "sdpa_bwd_wrapper_sm100_d256",
    "sdpa_fwd_wrapper_sm100_d256",
)


for _api_name in _TOP_LEVEL_OPTIONAL_APIS:
    globals()[_api_name] = _unsupported_optional_api(__name__, _api_name)

try:
    from mudnn.discrete_grouped_gemm.discrete_grouped_gemm_dswiglu.api import (
        DiscreteGroupedGemmDswigluSm100,
        discrete_grouped_gemm_dswiglu_wrapper_sm100,
    )
    from mudnn.discrete_grouped_gemm.discrete_grouped_gemm_swiglu.api import (
        DiscreteGroupedGemmSwigluSm100,
        discrete_grouped_gemm_swiglu_wrapper_sm100,
    )
    from mudnn.gemm_amax.api import GemmAmaxSm100, gemm_amax_wrapper_sm100
    from mudnn.gemm_dsrelu.api import GemmDsreluSm100, gemm_dsrelu_wrapper_sm100
    from mudnn.gemm_srelu.api import GemmSreluSm100, gemm_srelu_wrapper_sm100
    from mudnn.gemm_swiglu.api import GemmSwigluSm100, gemm_swiglu_wrapper_sm100
    from mudnn.grouped_gemm.grouped_gemm_dsrelu.api import (
        GroupedGemmDsreluSm100,
        grouped_gemm_dsrelu_wrapper_sm100,
    )
    from mudnn.grouped_gemm.grouped_gemm_dglu.api import (
        GroupedGemmDgluSm100,
        grouped_gemm_dglu_wrapper_sm100,
    )
    from mudnn.grouped_gemm.grouped_gemm_dswiglu.api import (
        GroupedGemmDswigluSm100,
        grouped_gemm_dswiglu_wrapper_sm100,
    )
    from mudnn.grouped_gemm.grouped_gemm_glu.api import (
        GroupedGemmGluSm100,
        grouped_gemm_glu_wrapper_sm100,
    )
    from mudnn.grouped_gemm.grouped_gemm_glu_hadamard.api import (
        GroupedGemmGluHadamardSm100,
        grouped_gemm_glu_hadamard_wrapper_sm100,
    )
    from mudnn.grouped_gemm.grouped_gemm_quant.api import (
        GroupedGemmQuantSm100,
        grouped_gemm_quant_wrapper_sm100,
    )
    from mudnn.grouped_gemm.grouped_gemm_srelu.api import (
        GroupedGemmSreluSm100,
        grouped_gemm_srelu_wrapper_sm100,
    )
    from mudnn.grouped_gemm.grouped_gemm_swiglu.api import (
        GroupedGemmSwigluSm100,
        grouped_gemm_swiglu_wrapper_sm100,
    )
    from mudnn.grouped_gemm.grouped_gemm_wgrad.api import (
        GroupedGemmWgradSm100,
        grouped_gemm_wgrad_wrapper_sm100,
    )
    from mudnn.rmsnorm_rht_amax.api import (
        RmsNormRhtAmaxSm100,
        rmsnorm_rht_amax_wrapper_sm100,
    )
    from mudnn.sdpa import (
        SdpabwdSm100D256,
        SdpafwdSm100D256,
        sdpa_bwd_wrapper_sm100_d256,
        sdpa_fwd_wrapper_sm100_d256,
    )
except Exception:
    pass

__all__ += list(_TOP_LEVEL_OPTIONAL_APIS)


class _OptionalModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _unsupported_optional_api(self.__name__, name)
        setattr(self, name, value)
        return value


class _OptionalModuleLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = _OptionalModule(spec.name)
        module.__file__ = "<mudnn_frontend_optional_compat>"
        module.__loader__ = self
        module.__package__ = spec.name
        module.__path__ = []
        module.__all__ = []
        return module

    def exec_module(self, module):
        return None


class _OptionalModuleFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not _is_optional_module(fullname):
            return None
        if importlib.machinery.PathFinder.find_spec(fullname, path) is not None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _OptionalModuleLoader(),
            is_package=True,
        )


if not any(isinstance(finder, _OptionalModuleFinder) for finder in sys.meta_path):
    sys.meta_path.append(_OptionalModuleFinder())
