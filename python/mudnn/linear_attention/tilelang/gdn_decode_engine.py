"""Single-token inference engine with device-side packed-length handling."""

import torch

from mudnn.engines.base import BaseEngine, CompiledPlan

from .gdn_inference_common import check_common


class GdnDecodeTileLangPlan(CompiledPlan):
    plan_name = "gdn_decode_tilelang"

    def execute(self, facts, ctx, **buffers):
        from .kernels.inference.gdn_decode import run_gated_delta_rule_decode_vk_fp32
        from .kernels.inference.gdn_recurrent import run_non_singleton

        q, k, v = (buffers[name] for name in ("q", "k", "v"))
        n = buffers["cu_seqlens"].numel() - 1
        h0 = buffers.get("initial_state")
        if h0 is None:
            h0 = torch.zeros(n, v.shape[1], 128, 128, dtype=torch.float32, device=q.device)
        final = buffers.get("final_state")
        o = torch.empty(q.shape[0], v.shape[1], 128, dtype=q.dtype, device=q.device)
        a_log = buffers.get("a_log")
        dt_bias = buffers.get("dt_bias")
        flags = facts.flags
        if q.shape[0]:
            run_gated_delta_rule_decode_vk_fp32(
                q=q,
                k=k,
                v=v,
                state=h0,
                state_indices=buffers.get("state_indices"),
                A_log=a_log,
                a=buffers["g"],
                dt_bias=dt_bias,
                b=buffers["beta"],
                output=o,
                scale=flags["scale"],
                use_qk_l2norm=flags["use_qk_l2norm_in_kernel"],
                cu_seqlens=buffers["cu_seqlens"],
                final_state=final,
                safe_gate=flags["safe_gate"],
                gate_domain=flags["gate_domain"],
                use_beta_sigmoid=flags["use_beta_sigmoid_in_kernel"],
                allow_neg_eigval=flags["allow_neg_eigval"],
            )
        recurrent_flags = dict(flags, has_indices=buffers.get("state_indices") is not None)
        run_non_singleton(
            q, k, v, buffers["g"], buffers["beta"], buffers["cu_seqlens"], h0, final, buffers.get("state_indices"), a_log, dt_bias, o, recurrent_flags
        )
        return o, None


class GdnDecodeTileLangEngine(BaseEngine):
    name = "gdn_decode_tilelang"

    def check_support(self, facts):
        check_common(self.name, facts)
        if facts.flags["checkpoint_every_n_tokens"]:
            raise NotImplementedError(f"{self.name}: checkpoints are served by prefill")
        if facts.flags["batch_invariant"]:
            raise NotImplementedError(f"{self.name}: batch_invariant is not yet qualified")
        if facts.dtypes[8] not in (None, torch.float32):
            raise NotImplementedError(f"{self.name}: a_log currently requires float32")
        state_stride = facts.strides[6]
        if state_stride is not None:
            shape = facts.shapes[6]
            if state_stride != (shape[1] * 128 * 128, 128 * 128, 128, 1):
                raise NotImplementedError(f"{self.name}: state must be contiguous VK")

    def build_plan(self, facts, ctx=None):
        return GdnDecodeTileLangPlan()
