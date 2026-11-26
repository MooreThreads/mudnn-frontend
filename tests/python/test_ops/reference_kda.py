"""CPU/GPU reference implementation for the chunked KDA forward.

Verbatim from the original ``test_kda.py`` so the oracle this port is checked
against is the one the source kernels were validated with, not a retyped
approximation.  Runs in fp32 with the same per-step rounding to the activation
dtype that the kernels perform.
"""

from __future__ import annotations

from functools import wraps

import torch
import torch.nn.functional as F

HEAD_SIZE = 128
#: The reference chunks at 32; the kernels chunk at 16.  The result is
#: chunk-size independent, which is why the two can differ.
CHUNK_SIZE = 32


def _exp2_ftz_like(x: torch.Tensor) -> torch.Tensor:
    y = torch.exp2(torch.clamp(x, max=127.0))
    return torch.where(x < -126.0, torch.zeros_like(y), y)



def _full_precision_musa_matmul(fn):
    @wraps(fn)
    def wrapped(q: torch.Tensor, *args, **kwargs):
        if q.device.type != "musa":
            return fn(q, *args, **kwargs)
        allow_tf32 = torch.backends.mudnn.allow_tf32
        torch.backends.mudnn.allow_tf32 = False
        try:
            return fn(q, *args, **kwargs)
        finally:
            torch.backends.mudnn.allow_tf32 = allow_tf32

    return wrapped



def _normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    # Match K1's normalization exactly. The epsilon is observable for the
    # all-zero Q/K vectors used by the single-token residual-layout probe:
    # without it the reference computes 0 * inf, which is NaN on CI devices.
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1.0e-6)



def blockwise_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = q.shape[-1]
    Hqk = q.shape[-2]
    H = v.shape[-2]
    assert D == HEAD_SIZE
    assert k.shape[-2] == Hqk
    assert g.shape[-2] == H
    assert beta.shape[-1] == H
    assert H % Hqk == 0
    qk_group = H // Hqk

    if cu_seqlens is None and q.ndim == 4:
        batch_size, num_tokens = q.shape[:2]
        flat_q = q.reshape(-1, Hqk, D)
        flat_k = k.reshape(-1, Hqk, D)
        flat_v = v.reshape(-1, H, D)
        flat_g = g.reshape(-1, H, D)
        flat_beta = beta.reshape(-1, H)
        seq_offsets = [i * num_tokens for i in range(batch_size + 1)]
        out_shape = v.shape
    elif cu_seqlens is None:
        if q.ndim != 3:
            raise AssertionError(
                "Dense reference inputs must be [B, T, H, D] or flattened to [B*T, H, D]."
            )
        flat_q = q
        flat_k = k
        flat_v = v
        flat_g = g
        flat_beta = beta
        seq_len = flat_q.shape[0] // initial_state.shape[0]
        seq_offsets = [i * seq_len for i in range(initial_state.shape[0] + 1)]
        out_shape = v.shape
    else:
        flat_q = q
        flat_k = k
        flat_v = v
        flat_g = g
        flat_beta = beta
        seq_offsets = [int(x) for x in cu_seqlens.tolist()]
        out_shape = v.shape

    if use_qk_l2norm_in_kernel:
        qn = _normalize_last_dim(flat_q.float()).to(q.dtype).float()
        kn = _normalize_last_dim(flat_k.float()).to(k.dtype).float()
    else:
        qn = flat_q.float()
        kn = flat_k.float()
    out = torch.empty_like(flat_v)
    state = initial_state.clone()
    log2e = 1.4426950408889634
    qk_head_indices = torch.arange(H, device=q.device, dtype=torch.long) // qk_group

    def round_workspace(x: torch.Tensor) -> torch.Tensor:
        return x.to(q.dtype).float()

    for seq_idx, bos in enumerate(seq_offsets[:-1]):
        eos = seq_offsets[seq_idx + 1]
        s = state[seq_idx].float().transpose(-1, -2)
        for start in range(bos, eos, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, eos)
            actual = end - start
            q_blk = torch.zeros(
                (H, CHUNK_SIZE, D), device=q.device, dtype=torch.float32
            )
            k_blk = torch.zeros_like(q_blk)
            v_blk = torch.zeros_like(q_blk)
            g_blk = torch.zeros_like(q_blk)
            beta_blk = torch.full(
                (H, CHUNK_SIZE), -80.0, device=q.device, dtype=torch.float32
            )

            q_blk[:, :actual] = (
                qn[start:end].index_select(1, qk_head_indices).permute(1, 0, 2)
            )
            k_blk[:, :actual] = (
                kn[start:end].index_select(1, qk_head_indices).permute(1, 0, 2)
            )
            v_blk[:, :actual] = flat_v[start:end].permute(1, 0, 2).float()
            g_blk[:, :actual] = flat_g[start:end].permute(1, 0, 2).float()
            beta_blk[:, :actual] = flat_beta[start:end].transpose(0, 1).float()

            gate = (
                lower_bound
                * log2e
                * torch.sigmoid(
                    torch.exp(A_log)[:, None, None] * (g_blk + dt_bias[:, None, :])
                )
            )
            if actual < CHUNK_SIZE:
                gate[:, actual:] = 0.0
            g_cumsum = gate.cumsum(dim=1)
            g_total = g_cumsum[:, -1]
            shift = 0.5 * (g_cumsum[:, 0] + g_total)
            k_decayed = round_workspace(k_blk * _exp2_ftz_like(g_cumsum))
            q_decayed = round_workspace(q_blk * _exp2_ftz_like(g_cumsum) * scale)
            k_decayed_for_tri = round_workspace(
                k_blk * _exp2_ftz_like(g_cumsum - shift[:, None])
            )
            q_decayed_for_p = round_workspace(
                q_blk * _exp2_ftz_like(g_cumsum - shift[:, None]) * scale
            )
            k_inv = round_workspace(k_blk * _exp2_ftz_like(shift[:, None] - g_cumsum))
            beta_sigmoid = torch.sigmoid(beta_blk)
            l_mat = round_workspace(
                torch.bmm(k_decayed_for_tri, k_inv.transpose(1, 2)).tril(-1)
                * beta_sigmoid[:, :, None]
            )

            a = torch.zeros(
                (H, CHUNK_SIZE, CHUNK_SIZE), device=q.device, dtype=torch.float32
            )
            a[:, 0, 0] = beta_sigmoid[:, 0]
            for i in range(1, CHUNK_SIZE):
                a[:, i, i] = beta_sigmoid[:, i]
                a[:, i, :i] = -torch.bmm(l_mat[:, i : i + 1, :i], a[:, :i, :i]).squeeze(
                    1
                )
            a = round_workspace(a)
            p = round_workspace(
                torch.bmm(q_decayed_for_p, k_inv.transpose(1, 2)).tril()
            )
            s_committed = round_workspace(s)
            tmp = round_workspace(v_blk - torch.bmm(k_decayed, s_committed))
            u = round_workspace(torch.bmm(a, tmp))
            out_chunk = round_workspace(
                torch.bmm(q_decayed, s_committed) + torch.bmm(p, u)
            )
            out[start:end] = out_chunk[:, :actual].permute(1, 0, 2)

            k_restored = round_workspace(
                k_blk * _exp2_ftz_like(g_total[:, None] - g_cumsum)
            )
            s = s * _exp2_ftz_like(g_total)[:, None, :] + torch.bmm(
                k_restored.transpose(1, 2), u
            )
        state[seq_idx].copy_(s.transpose(-1, -2).to(initial_state.dtype))

    return out.reshape(out_shape), state


