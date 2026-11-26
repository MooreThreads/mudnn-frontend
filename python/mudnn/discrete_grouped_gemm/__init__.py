"""Discrete grouped GEMM optional compatibility package."""

from .discrete_grouped_gemm_dswiglu import (
    DiscreteGroupedGemmDswigluSm100,
    discrete_grouped_gemm_dswiglu_wrapper_sm100,
)
from .discrete_grouped_gemm_swiglu import (
    DiscreteGroupedGemmSwigluSm100,
    discrete_grouped_gemm_swiglu_wrapper_sm100,
)

__all__ = [
    "DiscreteGroupedGemmDswigluSm100",
    "DiscreteGroupedGemmSwigluSm100",
    "discrete_grouped_gemm_dswiglu_wrapper_sm100",
    "discrete_grouped_gemm_swiglu_wrapper_sm100",
]
