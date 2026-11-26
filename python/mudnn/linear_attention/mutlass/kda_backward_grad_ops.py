"""JIT entry points for K4, `chunk_kda_backward_grad`.

The chunk-parallel half of the KDA backward (one CTA per ``(chunk, head)``).
This module is deliberately self-contained: it shares only the infrastructure
in :mod:`mudnn._jit` with the K1/K2/K3 modules, so a change to ``kda_ops.py``
cannot change what K4 renders or how it caches its module.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Mapping

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..._jit import env as jit_env
from ..._jit.core import JitSpec, gen_jit_spec
from ..._jit.utils import dtype_torch2mutlass_map


# The same tuning flags K1/K2/K3 are built with: denormal flushing and the
# backend's fop-combining passes are part of the numerical contract.
CXX_FLAGS = ["-O3", "-Wno-switch-bool"]
CUDA_FLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

INCLUDE_PATHS = [
    jit_env.MUDNN_JIT_INCLUDE_DIR,
    jit_env.MUDNN_JIT_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
]

KDA_GRAD_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}
KDA_GRAD_DTYPE_FFI = {
    torch.float16: "dl_float16",
    torch.bfloat16: "dl_bfloat16",
}
KDA_GRAD_CU_SEQLENS_ELEMENT_NAMES = {torch.int32: "int32_t", torch.int64: "int64_t"}
KDA_GRAD_CU_SEQLENS_DTYPE_NAMES = {torch.int32: "i32", torch.int64: "i64"}
KDA_GRAD_CU_SEQLENS_DTYPE_FFI = {torch.int32: "dl_int32", torch.int64: "dl_int64"}

KDA_GRAD_CHUNK_SIZE = 16
KDA_GRAD_HEAD_DIM = 128

# K4 declares its own shared-memory budget and static_asserts it against this
# cap, so a staging-layout change fails the build instead of the launch.
KDA_GRAD_MAX_DYNAMIC_SMEM_BYTES = 192 * 1024

KDA_GRAD_ARCH_TAGS = {
    "mp31": "mutlass::arch::Mp31",
    "mp32": "mutlass::arch::Mp32",
}

KDA_GRAD_BOOL_KEYS = (
    "is_varlen",
    "normalize_qk",
    "reuse_prepare",
    "safe_gate",
    "use_beta_sigmoid",
    "allow_neg_eigval",
    "gate_is_log",
    "gate_is_linear",
)
KDA_GRAD_CONFIG_KEYS = (
    "element",
    "element_dtype",
    "dtype_name",
    "cu_seqlens_element",
    "cu_seqlens_dtype",
    "cu_seqlens_dtype_name",
    "gate_element",
    "gate_element_dtype",
    "beta_element",
    "beta_element_dtype",
    *KDA_GRAD_BOOL_KEYS,
)
KDA_GRAD_DEFAULT_CONFIG = {
    "element": dtype_torch2mutlass_map()[torch.bfloat16],
    "element_dtype": KDA_GRAD_DTYPE_FFI[torch.bfloat16],
    "dtype_name": KDA_GRAD_DTYPE_NAMES[torch.bfloat16],
    "cu_seqlens_element": KDA_GRAD_CU_SEQLENS_ELEMENT_NAMES[torch.int64],
    "cu_seqlens_dtype": KDA_GRAD_CU_SEQLENS_DTYPE_FFI[torch.int64],
    "cu_seqlens_dtype_name": KDA_GRAD_CU_SEQLENS_DTYPE_NAMES[torch.int64],
    "is_varlen": False,
    "normalize_qk": True,
    "reuse_prepare": False,
    "safe_gate": True,
    "use_beta_sigmoid": True,
    "allow_neg_eigval": False,
    "gate_is_log": False,
    "gate_is_linear": False,
    "gate_element": dtype_torch2mutlass_map()[torch.bfloat16],
    "gate_element_dtype": KDA_GRAD_DTYPE_FFI[torch.bfloat16],
    "beta_element": dtype_torch2mutlass_map()[torch.bfloat16],
    "beta_element_dtype": KDA_GRAD_DTYPE_FFI[torch.bfloat16],
}


@functools.lru_cache
def _get_template_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


@functools.lru_cache
def _get_template(template_name: str):
    template_dir = (jit_env.MUDNN_JIT_TEMPLATE_DIR / "flat" / "kda").as_posix()
    return _get_template_env(template_dir).get_template(template_name)


KDA_BACKWARD_GRAD_TEMPLATE = _get_template("chunk_kda_backward_grad_kernel.j2")


def _visible_device_arch() -> str | None:
    try:
        if torch.musa.is_available() and torch.musa.device_count() > 0:
            props = torch.musa.get_device_properties(torch.musa.current_device())
            return f"mp{int(props.major)}{int(props.minor)}"
    except Exception:
        return None
    return None


def resolve_kda_grad_arch() -> str:
    """Select the MP31 or MP32 K4 implementation."""
    if os.environ.get("MUDNN_JIT_MUSA_ARCH_LIST", "").strip():
        arch_list = jit_env.resolve_musa_arch_list()
    else:
        device_arch = _visible_device_arch()
        if device_arch in KDA_GRAD_ARCH_TAGS:
            return device_arch
        arch_list = jit_env.resolve_musa_arch_list()
    for arch in reversed(sorted(KDA_GRAD_ARCH_TAGS)):
        if arch in arch_list:
            return arch
    raise ValueError(
        f"chunk_kda_backward_grad supports {sorted(KDA_GRAD_ARCH_TAGS)}, "
        f"got resolved arch list {arch_list}"
    )


def make_kda_backward_grad_ops_config(
    dtype: torch.dtype,
    *,
    cu_seqlens_dtype: torch.dtype | None = None,
    is_varlen: bool,
    normalize_qk: bool,
    reuse_prepare: bool = False,
    gate_dtype: torch.dtype | None = None,
    beta_dtype: torch.dtype | None = None,
    safe_gate: bool = True,
    use_beta_sigmoid: bool = True,
    allow_neg_eigval: bool = False,
    gate_domain: str = "log2",
) -> dict[str, object]:
    if dtype not in KDA_GRAD_DTYPE_FFI:
        raise TypeError(
            "chunk_kda_backward_grad supports torch.float16 and torch.bfloat16"
        )
    cu_seqlens_dtype = cu_seqlens_dtype or torch.int64
    if cu_seqlens_dtype not in KDA_GRAD_CU_SEQLENS_DTYPE_FFI:
        raise TypeError("chunk_kda_backward_grad cu_seqlens must be int32 or int64")
    from .kda_ops import KDA_STATE_DTYPE_FFI

    gate_dtype, beta_dtype = gate_dtype or dtype, beta_dtype or dtype
    if gate_dtype not in KDA_STATE_DTYPE_FFI or beta_dtype not in KDA_STATE_DTYPE_FFI:
        raise TypeError("KDA gates and beta must be fp16, bf16 or fp32")
    return {
        "gate_element": dtype_torch2mutlass_map()[gate_dtype],
        "gate_element_dtype": KDA_STATE_DTYPE_FFI[gate_dtype],
        "beta_element": dtype_torch2mutlass_map()[beta_dtype],
        "beta_element_dtype": KDA_STATE_DTYPE_FFI[beta_dtype],
        "safe_gate": bool(safe_gate),
        "use_beta_sigmoid": bool(use_beta_sigmoid),
        "allow_neg_eigval": bool(allow_neg_eigval),
        "gate_is_log": gate_domain == "log" and not safe_gate,
        "gate_is_linear": gate_domain == "linear" and not safe_gate,
        "element": dtype_torch2mutlass_map()[dtype],
        "element_dtype": KDA_GRAD_DTYPE_FFI[dtype],
        "dtype_name": KDA_GRAD_DTYPE_NAMES[dtype],
        "cu_seqlens_element": KDA_GRAD_CU_SEQLENS_ELEMENT_NAMES[cu_seqlens_dtype],
        "cu_seqlens_dtype": KDA_GRAD_CU_SEQLENS_DTYPE_FFI[cu_seqlens_dtype],
        "cu_seqlens_dtype_name": KDA_GRAD_CU_SEQLENS_DTYPE_NAMES[cu_seqlens_dtype],
        "is_varlen": bool(is_varlen),
        "normalize_qk": bool(normalize_qk),
        "reuse_prepare": bool(reuse_prepare),
    }


def _normalize_kda_backward_grad_ops_config(
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized = dict(KDA_GRAD_DEFAULT_CONFIG)
    if config is not None:
        normalized.update(config)
    for key in KDA_GRAD_CONFIG_KEYS:
        if key not in normalized:
            raise ValueError(f"missing KDA backward-grad JIT config key: {key}")
    for axis in ("gate", "beta"):
        if config is None or f"{axis}_element" not in config:
            normalized[f"{axis}_element"] = normalized["element"]
            normalized[f"{axis}_element_dtype"] = normalized["element_dtype"]
    for key in KDA_GRAD_BOOL_KEYS:
        normalized[key] = bool(normalized[key])
    if not normalized["is_varlen"]:
        for key in ("cu_seqlens_element", "cu_seqlens_dtype", "cu_seqlens_dtype_name"):
            normalized[key] = KDA_GRAD_DEFAULT_CONFIG[key]
    return {key: normalized[key] for key in KDA_GRAD_CONFIG_KEYS}


def get_kda_backward_grad_ops_function_name(
    config: Mapping[str, object] | None = None,
) -> str:
    config = _normalize_kda_backward_grad_ops_config(config)
    seqlens = (
        f"_seqlens_{config['cu_seqlens_dtype_name']}"
        if bool(config["is_varlen"])
        else ""
    )
    # The arch is part of the name, not just of the rendered source: the name
    # is what the generated source path, the spec registry key and the module
    # cache all derive from.
    name = (
        f"kda_backward_grad_dtype_{config['dtype_name']}"
        f"_varlen_{int(bool(config['is_varlen']))}{seqlens}"
        f"_normqk_{int(bool(config['normalize_qk']))}"
        + ("_pc1" if config["reuse_prepare"] else "")
        + f"_arch_{resolve_kda_grad_arch()}"
    )
    from .kda_ops import _gate_specialization_suffix

    return name + _gate_specialization_suffix(config)


def gen_kda_backward_grad_ops_spec(
    config: Mapping[str, object] | None = None,
) -> JitSpec:
    config = _normalize_kda_backward_grad_ops_config(config)
    arch = resolve_kda_grad_arch()
    if config["reuse_prepare"] and arch != "mp32":
        raise ValueError("prepare cache requires MP32")
    render_config = dict(config)
    render_config["arch"] = arch
    render_config["arch_tag"] = KDA_GRAD_ARCH_TAGS[arch]
    render_config["chunk_size"] = KDA_GRAD_CHUNK_SIZE
    render_config["head_dim"] = KDA_GRAD_HEAD_DIM
    render_config["max_dynamic_smem_bytes"] = KDA_GRAD_MAX_DYNAMIC_SMEM_BYTES
    for key in KDA_GRAD_BOOL_KEYS:
        render_config[key] = "true" if bool(config[key]) else "false"
    func_name = get_kda_backward_grad_ops_function_name(config)
    render_config["func_name"] = func_name
    source_file = Path(jit_env.MUDNN_JIT_GEN_SRC_DIR / "kda" / f"{func_name}.mu")
    return gen_jit_spec(
        func_name,
        [source_file],
        generated_sources={
            source_file: KDA_BACKWARD_GRAD_TEMPLATE.render(render_config)
        },
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


@functools.cache
def _load_backward_grad(frozen_config: tuple[tuple[str, object], ...]):
    return gen_kda_backward_grad_ops_spec(dict(frozen_config)).build_and_load()


def get_kda_backward_grad_ops_module(config: Mapping[str, object] | None = None):
    normalized = _normalize_kda_backward_grad_ops_config(config)
    return _load_backward_grad(
        (("arch", resolve_kda_grad_arch()), *sorted(normalized.items()))
    )
