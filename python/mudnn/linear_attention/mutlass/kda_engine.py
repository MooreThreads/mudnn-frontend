"""TileLang-free MUSA engines for KDA forward, on the mutlass kernels.

The kernels are compiled at first use by :mod:`mudnn._jit` and cached per
architecture, so nothing has to be pre-built.  ``check_support`` states exactly
what these engines can execute; everything else is declined with a reason rather
than approximated, because several cuDNN-frontend KDA options have no
counterpart in these kernels.

There is one engine per architecture, which is how cuDNN frontend expresses the
same split.  The ArchTag is rendered into the kernel source and has to agree
with the ``--offload-arch`` the module is compiled for, so the architecture is
part of the kernel ABI rather than a preference: :func:`kda_ops.resolve_kda_arch`
owns that single decision and each engine only reports whether it is the one it
names.
"""

from __future__ import annotations

import mudnn
import torch
from mudnn.engines.base import BaseEngine, CompiledPlan, ExecutionContext

from ..graph_analyzer import KdaFacts
from . import kda_ops, kda_run

#: The architectures these kernels exist for, in the order the family offers
#: them.  ``manifest`` owns the engine ids and this owns the split;
#: ``test_every_architecture_has_exactly_one_engine_and_the_block_is_open``
#: holds the two together.
ARCHITECTURES = ("mp31", "mp32")

_IO_DTYPES = (mudnn.data_type.HALF, mudnn.data_type.BFLOAT16)
_STATE_DTYPES = (mudnn.data_type.FLOAT, mudnn.data_type.BFLOAT16, None)


class KdaMutlassPlan(CompiledPlan):
    """Resolved once per configuration; ``execute`` runs the two kernels."""

    def __init__(self, facts: KdaFacts, arch: str):
        self.facts = facts
        self.arch = arch
        self.plan_name = f"kda_mutlass_{arch}"

    def execute(self, facts: KdaFacts, ctx: ExecutionContext, **buffers):
        # Always three elements: the contract's op has fixed arity and fills in
        # zero-size tensors for whatever the caller did not ask for.
        state = buffers["initial_state"]
        indices = buffers.get("state_indices")
        if indices is not None:
            indices = indices.contiguous()
        pool = buffers.get("final_state_out")
        copy_back = pool is not None and not pool.is_contiguous()
        if copy_back:
            state = (
                state if indices is None else state.index_select(0, indices)
            ).contiguous()
        elif state is not None:
            state = state.contiguous()
        final_state = torch.empty_like(state) if copy_back else pool
        if facts.wants_final_state and final_state is None:
            final_state = torch.empty(
                (len(facts.seqs), facts.h_o, facts.d_v, facts.d_qk),
                dtype=torch.float32 if state is None else state.dtype,
                device=buffers["q"].device,
            )
        q, k, v = (buffers[n] for n in ("q", "k", "v"))
        if facts.h_q != facts.h_k or facts.h_o != facts.h_v:
            q, k, v = (
                t.repeat_interleave(facts.h_o // t.shape[1], dim=1) for t in (q, k, v)
            )
        # Each sequence uses the same per-chunk arithmetic regardless of its batch.
        result = kda_run.run_chunk_kda(
            q,
            k,
            v,
            buffers["g"],
            buffers["beta"],
            scale=buffers["scale"],
            initial_state=state,
            output_final_state=buffers["output_final_state"],
            cu_seqlens=buffers["cu_seqlens"],
            A_log=buffers["a_log"],
            dt_bias=buffers["dt_bias"],
            lower_bound=buffers["lower_bound"],
            use_qk_l2norm_in_kernel=facts.use_qk_l2norm_in_kernel,
            checkpoint_every_n_tokens=facts.checkpoint_every_n_tokens,
            state_checkpoints=buffers["state_checkpoints"],
            final_state=final_state,
            initial_state_indices=None if copy_back else indices,
            use_beta_sigmoid=facts.use_beta_sigmoid_in_kernel,
            allow_neg_eigval=facts.allow_neg_eigval,
            gate_domain=facts.gate_domain,
        )
        if copy_back:
            if indices is None:
                pool.copy_(result[1])
            else:
                pool.index_copy_(0, indices.long(), result[1])
        if facts.wants_checkpoints:
            o, final_state, state_checkpoints = result
            return o, final_state, state_checkpoints
        o, final_state = result if isinstance(result, tuple) else (result, None)
        return o, final_state, None


class KdaMutlassEngine(BaseEngine):
    """KDA forward on the mutlass kernels, for one architecture.

    Subclasses set ``name`` and ``arch``, and nothing else differs between them:
    the architecture is resolved inside :mod:`kda_ops`, which is also the only
    place that decides it.  This base class is not registered on its own.
    """

    #: The architecture this engine's kernels are built for.
    arch: str = ""

    def check_support(self, facts: KdaFacts) -> None:
        """Decline anything these kernels cannot execute, naming the reason."""
        name = self.name
        if not self.arch:
            raise NotImplementedError(
                f"{name}: no architecture declared; register a subclass"
            )
        if facts is None or facts.op != "KDA":
            raise NotImplementedError(f"{name} supports exactly one KDA call")
        if facts.invalid:
            raise NotImplementedError(f"{name}: {facts.invalid}")
        if facts.device_type != "musa":
            raise NotImplementedError(
                f"{name}: the MUSA kernels are musa-only, got {facts.device_type!r}"
            )

        # The build target is part of the kernel ABI, so ask the driver what it
        # will compile for rather than keeping a second copy of the decision.
        try:
            built = kda_ops.resolve_kda_arch()
        except ValueError as exc:
            raise NotImplementedError(f"{name}: {exc}") from exc
        if built != self.arch:
            raise NotImplementedError(
                f"{name}: built for {self.arch}, this process builds for {built!r}"
            )

        if not facts.uniform_io:
            raise NotImplementedError(f"{name}: q/k/v dtypes must match")
        if facts.io_dtype not in _IO_DTYPES:
            raise NotImplementedError(
                f"{name}: q/k/v must be fp16 or bf16, got {facts.io_dtype}"
            )
        if facts.g_dtype not in (*_IO_DTYPES, mudnn.data_type.FLOAT):
            raise NotImplementedError(
                f"{name}: g must have the activation dtype {facts.io_dtype}, "
                f"got {facts.g_dtype}"
            )
        if facts.beta_dtype not in (facts.io_dtype, mudnn.data_type.FLOAT):
            raise NotImplementedError(
                f"{name}: beta must have the activation dtype {facts.io_dtype}, "
                f"got {facts.beta_dtype}"
            )
        if facts.state_dtype not in _STATE_DTYPES:
            raise NotImplementedError(
                f"{name}: initial_state must be fp32 or bf16, got {facts.state_dtype}"
            )
        if facts.cu_dtype not in (mudnn.data_type.INT32, mudnn.data_type.INT64):
            raise NotImplementedError(
                f"{name}: cu_seqlens must be int32 or int64, got {facts.cu_dtype}"
            )
        if (facts.d_qk, facts.d_v) != (128, 128):
            raise NotImplementedError(
                f"{name}: the kernels are compiled for K == V == 128, got "
                f"K={facts.d_qk}, V={facts.d_v}"
            )

        # The cadence is rendered into the kernel as a specialization axis, so
        # the kernel can only row-map a positive multiple of its own chunk.
        # Refusing it here names the option instead of failing inside mcc; the
        # pool itself is the driver's to size and allocate.
        if facts.wants_checkpoints and facts.checkpoint_every_n_tokens % kda_run.CHUNK:
            raise NotImplementedError(
                f"{name}: checkpoint_every_n_tokens must be a multiple of the "
                f"kernel chunk ({kda_run.CHUNK}), got "
                f"{facts.checkpoint_every_n_tokens}"
            )

    def build_plan(
        self, facts: KdaFacts, ctx: ExecutionContext = None
    ) -> KdaMutlassPlan:
        return KdaMutlassPlan(facts, self.arch)


class KdaMutlassMp31Engine(KdaMutlassEngine):
    """KDA forward on MP31: bf16/fp16 io, K == V == 128, chunk 16."""

    name = "kda_mutlass_mp31"
    arch = "mp31"


class KdaMutlassMp32Engine(KdaMutlassEngine):
    """KDA forward on MP32: bf16/fp16 io, K == V == 128, chunk 16."""

    name = "kda_mutlass_mp32"
    arch = "mp32"


__all__ = [
    "ARCHITECTURES",
    "KdaMutlassEngine",
    "KdaMutlassMp31Engine",
    "KdaMutlassMp32Engine",
    "KdaMutlassPlan",
]
