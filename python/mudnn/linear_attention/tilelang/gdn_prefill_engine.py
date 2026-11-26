"""Chunked inference kernels behind the frontend GDN engine contract."""

import torch

from mudnn.engines.base import BaseEngine, CompiledPlan

from .gdn_inference_common import check_common, prepare_gates
from .kernels.inference.helpers import MP31_ARCH, MP32_ARCH


class GdnPrefillTileLangPlan(CompiledPlan):
    plan_name = "gdn_prefill_tilelang"

    def __init__(self, facts):
        self.arch = facts.arch

    def execute(self, facts, ctx, **buffers):
        flags = facts.flags
        q, k, v = (buffers[name].unsqueeze(0) for name in ("q", "k", "v"))
        g, beta = prepare_gates(buffers["g"], buffers["beta"], flags, buffers.get("a_log"), buffers.get("dt_bias"))
        h0 = buffers.get("initial_state")
        final = buffers.get("final_state")
        common = dict(
            q=q,
            k=k,
            v=v,
            g=g.unsqueeze(0),
            b=beta.unsqueeze(0),
            cu_seqlens=buffers["cu_seqlens"],
            scale=flags["scale"],
            initial_state=h0,
            output_state=final,
            output_final_state=final is not None,
            is_log_space=True,
        )
        if self.arch == MP32_ARCH:
            from .kernels.inference.mp32.gdn_prefill import fused_chunk_gdn_prefill

            o, checkpoints, _ = fused_chunk_gdn_prefill(
                a=None, l2norm_eps=1e-6 if flags["use_qk_l2norm_in_kernel"] else None, checkpoint_every_n_tokens=flags["checkpoint_every_n_tokens"], **common
            )
        else:
            from .kernels.inference.gdn_kkt_solve import kkt_solve
            from .kernels.inference.gdn_l2norm import gdn_l2norm_
            from .kernels.inference.mp31.gdn_prefill import fused_chunk_gdn_prefill

            if flags["use_qk_l2norm_in_kernel"] and q.numel():
                q, k = q.clone(), k.clone()
                gdn_l2norm_(q)
                gdn_l2norm_(k)
                common.update(q=q, k=k)
            a = kkt_solve(k, beta.unsqueeze(0), cu_seqlens=buffers["cu_seqlens"]) if k.numel() else k.new_empty(1, 0, v.shape[2], 64)
            o, _, _ = fused_chunk_gdn_prefill(a=a, **common)
            checkpoints = None
        return o.squeeze(0), checkpoints


class GdnPrefillTileLangEngine(BaseEngine):
    name = "gdn_prefill_tilelang"

    def check_support(self, facts):
        check_common(self.name, facts)
        flags = facts.flags
        if facts.shapes[7] is not None:
            raise NotImplementedError(f"{self.name}: state_indices is served by decode")
        if flags["batch_invariant"]:
            raise NotImplementedError(f"{self.name}: batch_invariant is not yet qualified")
        if flags["checkpoint_every_n_tokens"] and facts.arch == MP31_ARCH:
            raise NotImplementedError(f"{self.name}: checkpoints require MP32")
        if any(stride[-1] != 1 for stride in facts.strides[:3]):
            raise NotImplementedError(f"{self.name}: Q/K/V require last-dimension stride 1")
        ho = facts.shapes[2][1]
        if not flags["safe_gate"] and flags["gate_domain"] == "log":
            if facts.strides[3] != (ho, 1):
                raise NotImplementedError(f"{self.name}: activated g must be contiguous")
        if not flags["use_beta_sigmoid_in_kernel"] and facts.strides[4] != (ho, 1):
            raise NotImplementedError(f"{self.name}: activated beta must be contiguous")
        state_stride = facts.strides[6]
        if state_stride is not None:
            shape = facts.shapes[6]
            if state_stride != (shape[1] * 128 * 128, 128 * 128, 128, 1):
                raise NotImplementedError(f"{self.name}: state must be contiguous VK")
        if facts.arch == MP31_ARCH and flags["use_qk_l2norm_in_kernel"]:
            for shape, strides, dtype in zip(facts.shapes[:2], facts.strides[:2], facts.dtypes[:2]):
                span = 1 + sum((n - 1) * s for n, s in zip(shape, strides))
                if span * torch.empty((), dtype=dtype).element_size() >= 2**31:
                    raise NotImplementedError(f"{self.name}: MP31 normalization span must be below 2 GiB")
        if facts.shapes[0][0] * facts.shapes[2][1] * 128 >= 2**31:
            raise NotImplementedError(f"{self.name}: output indexing exceeds int32 capacity")

    def build_plan(self, facts, ctx=None):
        return GdnPrefillTileLangPlan(facts)
