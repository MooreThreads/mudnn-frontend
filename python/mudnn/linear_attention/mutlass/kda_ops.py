from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Mapping

import torch
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from packaging.version import Version

from ..._jit import env as jit_env
from ..._jit.core import JitSpec, gen_jit_spec
from ..._jit.cpp_ext import get_mcc_version
from ..._jit.utils import dtype_torch2mutlass_map


CXX_FLAGS = ["-O3", "-Wno-switch-bool"]
CUDA_FLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

# The MP31 recurrence keeps five 2048-bit SQMMA accumulators alive -- its state
# fragment alone is a 64x128 fp32 fragment, i.e. 64 dword per thread in the one
# state squad -- which needs more than the 113 dword/thread that recurrence
# CTAs/MP = 3 allows.  Backends of the 5.1.0 line cannot allocate them at all
# and fail with "TEMP_2048"; halving the CTAs/MP halves the demand.  Keyed on
# the compiler version exactly like the FMHA BST overlap workaround
# (the upstream JIT fmha_fwd module), so a 5.3.0-line compiler keeps the fast
# configuration.
_MCC_MP31_KDA_REGPRESSURE_CUTOFF = Version("5.3.0")


def _get_kda_extra_cuda_cflags() -> list[str]:
    flags = list(CUDA_FLAGS)
    mcc_version = get_mcc_version()
    if mcc_version is not None and mcc_version < _MCC_MP31_KDA_REGPRESSURE_CUTOFF:
        flags.append("-DMATE_KDA_MP31_REC_CTAS_PER_MP=2")
    return flags


INCLUDE_PATHS = [
    jit_env.MUDNN_JIT_INCLUDE_DIR,
    jit_env.MUDNN_JIT_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
]

KDA_DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}
KDA_DTYPE_FFI = {
    torch.float16: "dl_float16",
    torch.bfloat16: "dl_bfloat16",
}
# The recurrent state pool may also be fp32, so it needs its own FFI map: the
# activation whitelist above must stay fp16/bf16-only.
KDA_STATE_DTYPE_FFI = {
    torch.float16: "dl_float16",
    torch.bfloat16: "dl_bfloat16",
    torch.float32: "dl_float32",
}
KDA_CU_SEQLENS_ELEMENT_NAMES = {torch.int32: "int32_t", torch.int64: "int64_t"}
KDA_CU_SEQLENS_DTYPE_NAMES = {torch.int32: "i32", torch.int64: "i64"}
KDA_CU_SEQLENS_DTYPE_FFI = {torch.int32: "dl_int32", torch.int64: "dl_int64"}

# Logical KDA prefill ABI shared by architecture backends.  SQMMA layouts and
# launch occupancy remain backend-specific.
KDA_PREFILL_CHUNK_SIZE = 16

KDA_PREFILL_HEAD_DIM = 128
KDA_PREPARE_CACHE_FLOATS = (
    KDA_PREFILL_CHUNK_SIZE * KDA_PREFILL_HEAD_DIM // 2 + 2 * KDA_PREFILL_CHUNK_SIZE
)
KDA_MP31_PREPARE_CTAS_PER_MP = 5
KDA_MP32_PREPARE_CTAS_PER_MP = 6

# The prepare collective is templated on ArchTag; resolve the tag from the
# same architecture list that drives the mcc --offload-arch flags, so MP31
# and MP32 builds select their own SQMMA/collective paths automatically.
KDA_ARCH_TAGS = {
    "mp31": "mutlass::arch::Mp31",
    "mp32": "mutlass::arch::Mp32",
}

# The prepare launch policy belongs to the arch the kernel was *built* for
# (ChunkKdaMp31SchedulePolicy / ChunkKdaMp32SchedulePolicy in
# chunk_kda_problem.hpp), not to the device it happens to run on.  The
# metadata region is sized from it, so it has to come from the same resolver
# that renders KDA_ARCH_TAGS: otherwise a cross build sizes the region for
# one arch and schedules for the other.
KDA_PREPARE_CTAS_PER_MP = {
    "mp31": KDA_MP31_PREPARE_CTAS_PER_MP,
    "mp32": KDA_MP32_PREPARE_CTAS_PER_MP,
}

KDA_BOOL_KEYS = (
    "has_state_in",
    "has_state_out",
    "has_chunk_states_out",
    "has_checkpoint_cadence",
    "state_fp32",
    "has_gate_params",
    "is_varlen",
    "normalize_qk",
    "save_prepare_cache",
    "use_beta_sigmoid",
    "allow_neg_eigval",
    "gate_is_log",
    "gate_is_linear",
)
#: The recurrence template renders the cadence into the kernel as a
#: ``std::integral_constant``, so it is a specialization axis and not just a
#: runtime value; ``has_checkpoint_cadence`` is derived from it (see
#: ``_normalize_kda_ops_config``) so the mode and the divisor cannot disagree.
KDA_INT_KEYS = ("checkpoint_every_n_tokens",)
KDA_CONFIG_KEYS = (
    "element",
    "element_dtype",
    "dtype_name",
    "state_element",
    "state_dtype_name",
    "gate_element",
    "gate_element_dtype",
    "beta_element",
    "beta_element_dtype",
    "cu_seqlens_element",
    "cu_seqlens_dtype",
    "cu_seqlens_dtype_name",
    *KDA_INT_KEYS,
    *KDA_BOOL_KEYS,
)
KDA_DEFAULT_CONFIG = {
    "element": dtype_torch2mutlass_map()[torch.bfloat16],
    "element_dtype": KDA_DTYPE_FFI[torch.bfloat16],
    "dtype_name": KDA_DTYPE_NAMES[torch.bfloat16],
    "state_element": dtype_torch2mutlass_map()[torch.bfloat16],
    "state_dtype_name": KDA_DTYPE_NAMES[torch.bfloat16],
    "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[torch.int64],
    "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[torch.int64],
    "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[torch.int64],
    "checkpoint_every_n_tokens": 0,
    "has_state_in": True,
    "has_state_out": True,
    "has_chunk_states_out": False,
    "has_checkpoint_cadence": False,
    "state_fp32": False,
    "has_gate_params": True,
    "is_varlen": False,
    "normalize_qk": True,
    "save_prepare_cache": False,
    "use_beta_sigmoid": True,
    "allow_neg_eigval": False,
    "gate_is_log": False,
    "gate_is_linear": False,
    "gate_element": dtype_torch2mutlass_map()[torch.bfloat16],
    "gate_element_dtype": KDA_STATE_DTYPE_FFI[torch.bfloat16],
    "beta_element": dtype_torch2mutlass_map()[torch.bfloat16],
    "beta_element_dtype": KDA_STATE_DTYPE_FFI[torch.bfloat16],
}


@functools.lru_cache
def _get_kda_template_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _get_kda_template(template_name: str):
    template_dir = (jit_env.MUDNN_JIT_TEMPLATE_DIR / "flat" / "kda").as_posix()
    return _get_kda_template_env(template_dir).get_template(template_name)


KDA_PREPARE_TEMPLATE = _get_kda_template("chunk_kda_prepare_kernel.j2")
KDA_RECURRENCE_TEMPLATE = _get_kda_template("chunk_kda_recurrence_kernel.j2")


def make_kda_ops_config(
    dtype: torch.dtype,
    *,
    state_dtype: torch.dtype,
    cu_seqlens_dtype: torch.dtype | None = None,
    has_state_in: bool,
    has_state_out: bool,
    has_chunk_states_out: bool = False,
    checkpoint_every_n_tokens: int = 0,
    state_fp32: bool,
    has_gate_params: bool,
    is_varlen: bool,
    normalize_qk: bool,
    save_prepare_cache: bool = False,
    gate_dtype: torch.dtype | None = None,
    beta_dtype: torch.dtype | None = None,
    use_beta_sigmoid: bool = True,
    allow_neg_eigval: bool = False,
    gate_domain: str = "log2",
) -> dict[str, object]:
    if dtype not in KDA_DTYPE_FFI:
        raise TypeError("chunk_kda supports torch.float16 and torch.bfloat16 inputs")
    if state_dtype not in KDA_DTYPE_NAMES:
        raise TypeError("chunk_kda state must be fp16, bf16, or fp32")
    cu_seqlens_dtype = cu_seqlens_dtype or torch.int64
    if cu_seqlens_dtype not in KDA_CU_SEQLENS_DTYPE_FFI:
        raise TypeError("chunk_kda cu_seqlens must be int32 or int64")
    gate_dtype, beta_dtype = gate_dtype or dtype, beta_dtype or dtype
    if gate_dtype not in KDA_STATE_DTYPE_FFI or beta_dtype not in KDA_STATE_DTYPE_FFI:
        raise TypeError("KDA gates and beta must be fp16, bf16 or fp32")
    if gate_domain not in ("log", "log2", "linear"):
        raise ValueError("unsupported KDA gate domain")
    return {
        "gate_element": dtype_torch2mutlass_map()[gate_dtype],
        "gate_element_dtype": KDA_STATE_DTYPE_FFI[gate_dtype],
        "beta_element": dtype_torch2mutlass_map()[beta_dtype],
        "beta_element_dtype": KDA_STATE_DTYPE_FFI[beta_dtype],
        "use_beta_sigmoid": bool(use_beta_sigmoid),
        "allow_neg_eigval": bool(allow_neg_eigval),
        "gate_is_log": gate_domain == "log" and not has_gate_params,
        "gate_is_linear": gate_domain == "linear" and not has_gate_params,
        "element": dtype_torch2mutlass_map()[dtype],
        "element_dtype": KDA_DTYPE_FFI[dtype],
        "dtype_name": KDA_DTYPE_NAMES[dtype],
        "state_element": dtype_torch2mutlass_map()[state_dtype],
        "state_dtype_name": KDA_DTYPE_NAMES[state_dtype],
        "cu_seqlens_element": KDA_CU_SEQLENS_ELEMENT_NAMES[cu_seqlens_dtype],
        "cu_seqlens_dtype": KDA_CU_SEQLENS_DTYPE_FFI[cu_seqlens_dtype],
        "cu_seqlens_dtype_name": KDA_CU_SEQLENS_DTYPE_NAMES[cu_seqlens_dtype],
        "checkpoint_every_n_tokens": int(checkpoint_every_n_tokens),
        "has_state_in": bool(has_state_in),
        "has_state_out": bool(has_state_out),
        "has_chunk_states_out": bool(has_chunk_states_out),
        "state_fp32": bool(state_fp32),
        "has_gate_params": bool(has_gate_params),
        "is_varlen": bool(is_varlen),
        "normalize_qk": bool(normalize_qk),
        "save_prepare_cache": bool(save_prepare_cache),
    }


def _normalize_kda_ops_config(
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    normalized = dict(KDA_DEFAULT_CONFIG)
    if config is not None:
        normalized.update(config)
    for key in KDA_CONFIG_KEYS:
        if key not in normalized:
            raise ValueError(f"missing KDA JIT config key: {key}")
    for key in KDA_INT_KEYS:
        normalized[key] = int(normalized[key])
    for axis in ("gate", "beta"):
        if config is None or f"{axis}_element" not in config:
            normalized[f"{axis}_element"] = normalized["element"]
            normalized[f"{axis}_element_dtype"] = normalized["element_dtype"]
    for key in KDA_BOOL_KEYS:
        normalized[key] = bool(normalized[key])
    # The cadence is a compile-time specialization axis, so the rendered source
    # has to agree with it: the mode flag is derived here rather than taken from
    # the caller, and a cadence the kernel cannot row-map is rejected before it
    # can be rendered.  the host contract leans on this ("the host has
    # already rejected anything that is not a positive multiple of the chunk");
    # its components static_assert the same contract.
    cadence = normalized["checkpoint_every_n_tokens"]
    if cadence < 0:
        raise ValueError(f"checkpoint_every_n_tokens must be >= 0, got {cadence}")
    if cadence % KDA_PREFILL_CHUNK_SIZE:
        raise ValueError(
            f"checkpoint_every_n_tokens must be a multiple of the kernel chunk "
            f"({KDA_PREFILL_CHUNK_SIZE}), got {cadence}"
        )
    normalized["has_checkpoint_cadence"] = cadence > 0
    if cadence > 0 and not normalized["has_chunk_states_out"]:
        raise ValueError(
            "a checkpoint cadence needs the pool it writes into: "
            "has_chunk_states_out must be true when checkpoint_every_n_tokens > 0"
        )
    # The recurrence template only needs this value for its chunk-state ABI
    # guard.  It is not an independent specialization axis: all state tensors
    # are required to have one dtype, and ``state_dtype_name`` is the source of
    # truth for that dtype.  Derive it here so callers that construct a config
    # directly cannot make the rendered source disagree with the recurrence
    # name/cache key (which intentionally omit ``state_fp32``).
    normalized["state_fp32"] = normalized["state_dtype_name"] == "fp32"
    if not normalized["is_varlen"]:
        for key in (
            "cu_seqlens_element",
            "cu_seqlens_dtype",
            "cu_seqlens_dtype_name",
        ):
            normalized[key] = KDA_DEFAULT_CONFIG[key]
    return {key: normalized[key] for key in KDA_CONFIG_KEYS}


def _visible_device_arch() -> str | None:
    try:
        if torch.musa.is_available() and torch.musa.device_count() > 0:
            props = torch.musa.get_device_properties(torch.musa.current_device())
            return f"mp{int(props.major)}{int(props.minor)}"
    except Exception:
        return None
    return None


def resolve_kda_arch() -> str:
    """The one KDA architecture decision: ``"mp31"`` or ``"mp32"``.

    Everything that depends on the architecture -- the ArchTag rendered into
    the source, the JIT cache key, and the launch policy the host sizes
    buffers from -- must come from here, so that a cross build cannot make
    two of them disagree.
    """
    # ArchTag describes the GPU the kernel will run on, and it must agree with
    # the mcc --offload-arch flags: an Mp31-tagged source compiled for an mp_32
    # target selects the MP31 SQMMA, whose instruction bodies are guarded by
    # MUTE_ARCH_SQMMA_MP31_ENABLED and therefore compiled out under
    # __MUSA_ARCH__ == 320. The gemm then emits nothing and MTGPU-DEPENDENCY-GRAPH
    # rejects the resulting empty TCE commit group.
    #
    # An explicit MUDNN_JIT_MUSA_ARCH_LIST wins, because that is the only way to
    # build for a target other than the locally visible device - cross
    # building, or validating an MP32 build from an MP31 host. It is also the
    # same variable that drives the --offload-arch flags, so honouring it keeps
    # the two in step by construction.
    if os.environ.get("MUDNN_JIT_MUSA_ARCH_LIST", "").strip():
        arch_list = jit_env.resolve_musa_arch_list()
    else:
        # Otherwise the visible device decides. Note that an MP32 device also
        # executes mp31-coded binaries in compatibility mode, so there is no
        # case for preferring a lower arch tag than the device reports.
        device_arch = _visible_device_arch()
        if device_arch in KDA_ARCH_TAGS:
            return device_arch
        arch_list = jit_env.resolve_musa_arch_list()
    for arch in reversed(sorted(KDA_ARCH_TAGS)):
        if arch in arch_list:
            return arch
    raise ValueError(
        f"chunk_kda supports {sorted(KDA_ARCH_TAGS)}, got resolved arch list {arch_list}"
    )


def resolve_kda_prepare_ctas_per_mp() -> int:
    """Prepare CTAs per MP for the resolved arch's launch policy."""
    return KDA_PREPARE_CTAS_PER_MP[resolve_kda_arch()]


def _resolve_kda_arch_tag() -> str:
    return KDA_ARCH_TAGS[resolve_kda_arch()]


def _render_config(config: Mapping[str, object], func_name: str) -> dict[str, object]:
    render_config = dict(config)
    render_config["func_name"] = func_name
    render_config["arch_tag"] = _resolve_kda_arch_tag()
    render_config["chunk_size"] = KDA_PREFILL_CHUNK_SIZE
    render_config["head_dim"] = KDA_PREFILL_HEAD_DIM
    for key in KDA_BOOL_KEYS:
        render_config[key] = "true" if bool(config[key]) else "false"
    return render_config


def _gate_specialization_suffix(config):
    suffix = ""
    for axis in ("gate", "beta"):
        if config[f"{axis}_element_dtype"] != config["element_dtype"]:
            suffix += f"_{axis}_{config[f'{axis}_element_dtype']}"
    if not config["use_beta_sigmoid"]:
        suffix += "_beta_plain"
    if config["allow_neg_eigval"]:
        suffix += "_negative"
    if config["gate_is_log"]:
        suffix += "_gate_ln"
    if config["gate_is_linear"]:
        suffix += "_gate_linear"
    if config.get("safe_gate") is False and not (
        config["gate_is_log"] or config["gate_is_linear"]
    ):
        suffix += "_gate_log2"
    return suffix


def _common_suffix(config: Mapping[str, object]) -> str:
    seqlens = (
        f"_seqlens_{config['cu_seqlens_dtype_name']}"
        if bool(config["is_varlen"])
        else ""
    )
    name = (
        f"dtype_{config['dtype_name']}"
        f"_gate_{int(bool(config['has_gate_params']))}"
        f"_varlen_{int(bool(config['is_varlen']))}"
        f"_normqk_{int(bool(config['normalize_qk']))}{seqlens}"
    )
    return name + _gate_specialization_suffix(config)


def _recurrence_suffix(config: Mapping[str, object]) -> str:
    """Name only the axes rendered by the recurrence template.

    Gate parameters and Q/K normalization belong to prepare.  They are not
    referenced by ``chunk_kda_recurrence_kernel.j2`` and must not create four
    duplicate recurrence modules for the same state/workspace specialization.
    ``state_fp32`` is likewise implied by ``state_dtype_name``.

    The checkpoint cadence *is* named here: the template bakes it into the
    source, and the name is what the generated ``.mu`` path and the spec
    registry key derive from, so two cadences sharing a name would hand back one
    rendered kernel for the other.
    """
    seqlens = (
        f"_seqlens_{config['cu_seqlens_dtype_name']}"
        if bool(config["is_varlen"])
        else ""
    )
    return (
        f"dtype_{config['dtype_name']}"
        f"_varlen_{int(bool(config['is_varlen']))}{seqlens}"
        f"_state_{config['state_dtype_name']}"
        f"_in_{int(bool(config['has_state_in']))}"
        f"_out_{int(bool(config['has_state_out']))}"
        f"_cs_{int(bool(config['has_chunk_states_out']))}"
        f"_ckpt_{int(config['checkpoint_every_n_tokens'])}"
    )


# The arch is part of the name, not just of the rendered source: the name is
# what the generated source path, the spec registry key and the module cache all
# derive from.  Without it a process that resolves two arches
# shares one generated .mu and hands back one arch's module for the other.
def get_kda_prepare_ops_function_name(
    config: Mapping[str, object] | None = None,
) -> str:
    config = _normalize_kda_ops_config(config)
    arch = resolve_kda_arch()
    frame = "_cs_1" if arch == "mp32" and config["has_chunk_states_out"] else ""
    cache = "_pc1" if config["save_prepare_cache"] else ""
    return f"prepare_kda_{_common_suffix(config)}{frame}{cache}_arch_{arch}"


def get_kda_recurrence_ops_function_name(
    config: Mapping[str, object] | None = None,
) -> str:
    config = _normalize_kda_ops_config(config)
    return f"kda_recurrence_{_recurrence_suffix(config)}_arch_{resolve_kda_arch()}"


def _gen_spec(
    config: Mapping[str, object],
    *,
    func_name: str,
    template,
) -> JitSpec:
    source_file = Path(jit_env.MUDNN_JIT_GEN_SRC_DIR / "kda" / f"{func_name}.mu")
    return gen_jit_spec(
        func_name,
        [source_file],
        generated_sources={
            source_file: template.render(_render_config(config, func_name))
        },
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=_get_kda_extra_cuda_cflags(),
        extra_include_paths=list(INCLUDE_PATHS),
    )


def gen_kda_prepare_ops_spec(config: Mapping[str, object] | None = None) -> JitSpec:
    config = _normalize_kda_ops_config(config)
    config.update(
        state_element=config["element"],
        state_dtype_name=config["dtype_name"],
        has_state_in=False,
        has_state_out=False,
        has_chunk_states_out=(
            resolve_kda_arch() == "mp32" and config["has_chunk_states_out"]
        ),
        state_fp32=False,
    )
    if config["save_prepare_cache"] and (
        resolve_kda_arch() != "mp32" or config["dtype_name"] != "bf16"
    ):
        raise ValueError("prepare cache requires MP32 and gate parameters")
    return _gen_spec(
        config,
        func_name=get_kda_prepare_ops_function_name(config),
        template=KDA_PREPARE_TEMPLATE,
    )


def gen_kda_recurrence_ops_spec(config: Mapping[str, object] | None = None) -> JitSpec:
    config = _normalize_kda_ops_config(config)
    return _gen_spec(
        config,
        func_name=get_kda_recurrence_ops_function_name(config),
        template=KDA_RECURRENCE_TEMPLATE,
    )


def _module_cache_key(
    config: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """The per-process JIT module cache key.

    The arch leads it because one logical config builds a different kernel
    per arch and this cache lives for the whole process.  The loaders drop the
    extra entry again in ``_normalize_kda_ops_config``.
    """
    return (("arch", resolve_kda_arch()), *sorted(config.items()))


def _recurrence_module_cache_key(
    config: Mapping[str, object],
) -> tuple[tuple[str, object], ...]:
    """Cache key for the recurrence's rendered inputs only.

    Keep the full normalized config for the prepare key, but drop controls that
    do not occur in the recurrence template.  This mirrors the function name
    and prevents a process-local cache from retaining four equivalent K2
    modules.
    """
    recurrence_only = {
        key: value
        for key, value in config.items()
        if key
        not in {"has_gate_params", "normalize_qk", "state_fp32", "save_prepare_cache"}
    }
    return _module_cache_key(recurrence_only)


@functools.cache
def _load_prepare(frozen_config: tuple[tuple[str, object], ...]):
    return gen_kda_prepare_ops_spec(dict(frozen_config)).build_and_load()


@functools.cache
def _load_recurrence(frozen_config: tuple[tuple[str, object], ...]):
    return gen_kda_recurrence_ops_spec(dict(frozen_config)).build_and_load()


def get_kda_prepare_ops_module(config: Mapping[str, object] | None = None):
    normalized = _normalize_kda_ops_config(config)
    # MP32 prepare needs the checkpoint frame; the other state options are unused.
    # The checkpoint cadence belongs here too: it is a recurrence axis, and
    # leaving it in the prepare key would build one identical prepare module per
    # cadence.
    normalized.update(
        state_element=normalized["element"],
        state_dtype_name=normalized["dtype_name"],
        has_state_in=False,
        has_state_out=False,
        has_chunk_states_out=(
            resolve_kda_arch() == "mp32" and normalized["has_chunk_states_out"]
        ),
        has_checkpoint_cadence=False,
        checkpoint_every_n_tokens=0,
        state_fp32=False,
    )
    return _load_prepare(_module_cache_key(normalized))


def get_kda_recurrence_ops_module(config: Mapping[str, object] | None = None):
    normalized = _normalize_kda_ops_config(config)
    return _load_recurrence(_recurrence_module_cache_key(normalized))
