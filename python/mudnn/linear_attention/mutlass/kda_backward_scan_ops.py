"""JIT entry points for K3, the MP31 chunk-KDA backward scan.

K3 is *not* part of ``kda_ops.py`` on purpose: it is the one kernel of
the backward split that a second agent does not touch, so it keeps its own
template, its own rendered source and its own module cache.  The specialization
axes are the ones the rendered template actually reads -- the activation dtype,
the cu_seqlens dtype, varlen -- plus ``arch`` in the cache key, exactly like the
prepare/recurrence/backward entries of ``kda_ops``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Mapping

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..._jit import env as jit_env
from ..._jit.core import JitSpec, gen_jit_spec
from .kda_ops import (
    CXX_FLAGS,
    KDA_ARCH_TAGS,
    KDA_CU_SEQLENS_DTYPE_FFI,
    KDA_CU_SEQLENS_DTYPE_NAMES,
    KDA_CU_SEQLENS_ELEMENT_NAMES,
    KDA_DTYPE_FFI,
    KDA_DTYPE_NAMES,
    KDA_PREFILL_CHUNK_SIZE,
    KDA_PREFILL_HEAD_DIM,
    _get_kda_extra_cuda_cflags,
    resolve_kda_arch,
)
from ..._jit.utils import dtype_torch2mutlass_map

INCLUDE_PATHS = [
    jit_env.MUDNN_JIT_INCLUDE_DIR,
    jit_env.MUDNN_JIT_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
]

# The template static_asserts its own SharedStorageSize against this cap, so a
# layout change fails the build instead of the launch.
KDA_BACKWARD_SCAN_MAX_DYNAMIC_SMEM_BYTES = 192 * 1024

KDA_BACKWARD_SCAN_BOOL_KEYS = ("is_varlen",)
KDA_BACKWARD_SCAN_CONFIG_KEYS = (
    "element",
    "element_dtype",
    "dtype_name",
    "cu_seqlens_element",
    "cu_seqlens_dtype",
    "cu_seqlens_dtype_name",
    *KDA_BACKWARD_SCAN_BOOL_KEYS,
)
KDA_BACKWARD_SCAN_DEFAULT_CONFIG = {
    "element": dtype_torch2mutlass_map()[torch.bfloat16],
    "element_dtype": KDA_DTYPE_FFI[torch.bfloat16],
    "dtype_name": KDA_DTYPE_NAMES[torch.bfloat16],
    "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[torch.int64],
    "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[torch.int64],
    "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[torch.int64],
    "is_varlen": False,
}


@functools.lru_cache
def _get_backward_scan_template_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


KDA_BACKWARD_SCAN_TEMPLATE = _get_backward_scan_template_env(
    (jit_env.MUDNN_JIT_TEMPLATE_DIR / "flat" / "kda").as_posix()
).get_template("chunk_kda_backward_scan_kernel.j2")


def make_kda_backward_scan_ops_config(
    dtype: torch.dtype,
    *,
    cu_seqlens_dtype: torch.dtype | None = None,
    is_varlen: bool,
) -> dict[str, object]:
    if dtype not in KDA_DTYPE_FFI:
        raise TypeError(
            "chunk_kda_backward_scan supports torch.float16 and torch.bfloat16 inputs"
        )
    cu_seqlens_dtype = cu_seqlens_dtype or torch.int64
    if cu_seqlens_dtype not in KDA_CU_SEQLENS_DTYPE_FFI:
        raise TypeError("chunk_kda_backward_scan cu_seqlens must be int32 or int64")
    return {
        "element": dtype_torch2mutlass_map()[dtype],
        "element_dtype": KDA_DTYPE_FFI[dtype],
        "dtype_name": KDA_DTYPE_NAMES[dtype],
        "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[cu_seqlens_dtype],
        "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[cu_seqlens_dtype],
        "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[cu_seqlens_dtype],
        "is_varlen": bool(is_varlen),
    }


def _normalize_kda_backward_scan_ops_config(
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized = dict(KDA_BACKWARD_SCAN_DEFAULT_CONFIG)
    if config is not None:
        normalized.update(config)
    for key in KDA_BACKWARD_SCAN_CONFIG_KEYS:
        if key not in normalized:
            raise ValueError(f"missing KDA backward-scan JIT config key: {key}")
    for key in KDA_BACKWARD_SCAN_BOOL_KEYS:
        normalized[key] = bool(normalized[key])
    if not normalized["is_varlen"]:
        for key in ("cu_seqlens_element", "cu_seqlens_dtype", "cu_seqlens_dtype_name"):
            normalized[key] = KDA_BACKWARD_SCAN_DEFAULT_CONFIG[key]
    return {key: normalized[key] for key in KDA_BACKWARD_SCAN_CONFIG_KEYS}


# The arch leads the name for the same reason it does in kda_ops: one logical
# config renders a different kernel per arch, and the name is what the generated
# source path, the spec registry key and the module cache all derive from.
def get_kda_backward_scan_ops_function_name(
    config: Mapping[str, object] | None = None,
) -> str:
    config = _normalize_kda_backward_scan_ops_config(config)
    seqlens = (
        f"_seqlens_{config['cu_seqlens_dtype_name']}"
        if bool(config["is_varlen"])
        else ""
    )
    return (
        f"kda_backward_scan_dtype_{config['dtype_name']}"
        f"_varlen_{int(bool(config['is_varlen']))}{seqlens}"
        f"_arch_{resolve_kda_arch()}"
    )


def gen_kda_backward_scan_ops_spec(
    config: Mapping[str, object] | None = None,
) -> JitSpec:
    config = _normalize_kda_backward_scan_ops_config(config)
    arch = resolve_kda_arch()
    if arch not in ("mp31", "mp32"):
        raise NotImplementedError(
            f"chunk_kda_backward_scan supports MP31 and MP32; resolved arch is {arch}"
        )
    render_config = dict(config)
    render_config["arch_tag"] = KDA_ARCH_TAGS[arch]
    render_config["chunk_size"] = KDA_PREFILL_CHUNK_SIZE
    render_config["head_dim"] = KDA_PREFILL_HEAD_DIM
    render_config["max_dynamic_smem_bytes"] = KDA_BACKWARD_SCAN_MAX_DYNAMIC_SMEM_BYTES
    for key in KDA_BACKWARD_SCAN_BOOL_KEYS:
        render_config[key] = "true" if bool(config[key]) else "false"
    func_name = get_kda_backward_scan_ops_function_name(config)
    render_config["func_name"] = func_name
    source_file = Path(jit_env.MUDNN_JIT_GEN_SRC_DIR / "kda" / f"{func_name}.mu")
    return gen_jit_spec(
        func_name,
        [source_file],
        generated_sources={
            source_file: (
                KDA_BACKWARD_SCAN_TEMPLATE if arch == "mp31" else
                _get_backward_scan_template_env(
                    (jit_env.MUDNN_JIT_TEMPLATE_DIR / "flat" / "kda").as_posix()
                ).get_template("chunk_kda_backward_scan_mp32_kernel.j2")
            ).render(render_config)
        },
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=_get_kda_extra_cuda_cflags(),
        extra_include_paths=list(INCLUDE_PATHS),
    )


@functools.cache
def _load_backward_scan(frozen_config: tuple[tuple[str, object], ...]):
    return gen_kda_backward_scan_ops_spec(dict(frozen_config)).build_and_load()


def get_kda_backward_scan_ops_module(config: Mapping[str, object] | None = None):
    normalized = _normalize_kda_backward_scan_ops_config(config)
    return _load_backward_scan(
        (("arch", resolve_kda_arch()), *sorted(normalized.items()))
    )
