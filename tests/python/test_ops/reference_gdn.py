# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Pure-PyTorch gated-delta-net reference for the THD operation contract."""

import torch


def relative_l2(reference: torch.Tensor, actual: torch.Tensor) -> float:
    """Global relative L2 of ``actual`` against ``reference``, ``||got - ref|| / ||ref||``."""
    ref = reference.float()
    return ((ref - actual.float()).norm() / ref.norm().clamp(min=1e-12)).item()


@torch.no_grad()
def naive_recurrent_gdn(q, k, v, g, beta, cu_seqlens, initial_state=None, output_final_state=False, scale=None):
    """Sequential fp32 ground truth over a packed batch, q/k L2-normalized.

    THD in and out: ``q``/``k``/``v`` ``[total, H, D]``, ``g``/``beta``
    ``[total, H]``, ``cu_seqlens`` ``[N + 1]``. Each output token reads the state
    after that token's rank-1 write. Returns ``(o [total, H, V], K-major
    final_state [N, H, K, V] or None)``.
    """
    total, heads, dim_k = q.shape
    dim_v = v.shape[-1]
    if scale is None:
        scale = dim_k**-0.5
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    bounds = cu_seqlens.tolist()
    o = torch.empty(total, heads, dim_v, dtype=torch.float32, device=q.device)
    states = []
    for seq, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        h = (
            torch.zeros(heads, dim_k, dim_v, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state[seq].float().clone()
        )
        for t in range(start, stop):
            h = h * g[t].exp()[..., None, None]
            pred = (h * k[t][..., None]).sum(dim=-2)
            delta = beta[t][..., None] * (v[t] - pred)
            h = h + k[t][..., None] * delta[..., None, :]
            o[t] = (h * q[t][..., None]).sum(dim=-2) * scale
        states.append(h)
    return o, (torch.stack(states) if output_final_state else None)


def _chunk_gdn_dense(q, k, v, g, beta, h, scale, chunk):
    """One sequence of the chunk-parallel (WY) recurrence; ``[1, T, H, D]`` in."""
    _b, seq, heads, dim_k = q.shape
    device = q.device
    tri_excl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=device), -1)
    tri_incl = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=device), 0)
    eye = torch.eye(chunk, dtype=torch.float32, device=device)
    outs = []
    for c in range(seq // chunk):
        rows = slice(c * chunk, (c + 1) * chunk)
        qc, kc, vc = (x[:, rows].transpose(1, 2) for x in (q, k, v))
        gc, bc = (x[:, rows].transpose(1, 2) for x in (g, beta))
        gcum = gc.cumsum(-1)
        decay = gcum.exp()
        pair = torch.where(tri_incl, gcum[..., :, None] - gcum[..., None, :], torch.zeros(1, device=device)).exp()
        kkt = kc @ kc.transpose(-1, -2)
        a = torch.where(tri_excl, bc[..., :, None] * kkt * pair, torch.zeros_like(kkt))
        inv = torch.linalg.solve_triangular(
            eye + a, eye.expand(1, heads, chunk, chunk), upper=False, unitriangular=True
        )
        delta = inv @ (bc[..., :, None] * (vc - decay[..., :, None] * (kc @ h)))
        qk = torch.where(tri_incl, (qc @ kc.transpose(-1, -2)) * pair, torch.zeros(1, device=device))
        outs.append((scale * (decay[..., :, None] * (qc @ h) + qk @ delta)).transpose(1, 2))
        kdec = kc * (gcum[..., -1:] - gcum).exp()[..., :, None]
        h = decay[..., -1:, None] * h + kdec.transpose(-1, -2) @ delta
    return torch.cat(outs, dim=1), h


def chunk_gdn_diff(q, k, v, g, beta, cu_seqlens, initial_state=None, output_final_state=False, scale=None):
    """Autograd-able fp32 oracle for the gradients. Sequences must be chunk-aligned."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    heads, dim_k, dim_v = q.shape[1], q.shape[-1], v.shape[-1]
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    bounds = cu_seqlens.tolist()
    outs, states = [], []
    for seq, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        rows = slice(start, stop)
        h = (
            torch.zeros(1, heads, dim_k, dim_v, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state[seq].float()[None]
        )
        out, h = _chunk_gdn_dense(*(x[rows][None] for x in (q, k, v, g, beta)), h, scale, 64)
        outs.append(out[0])
        states.append(h[0])
    return (torch.cat(outs), torch.stack(states) if output_final_state else None)


# Shared FP64 reference and deterministic training inputs


def recurrent_gdn_fp64(q, k, v, g, beta, cu, scale, h0=None):
    """Per-sequence fp64 recurrence; ``h0`` is ``[N, H, K, V]`` in fp64."""
    total, heads, dim_k = q.shape
    dim_v = v.shape[-1]
    o = torch.empty(total, heads, dim_v, dtype=torch.float64, device=q.device)
    bounds = [int(x) for x in cu.tolist()]
    finals = []
    for i, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        h = torch.zeros(heads, dim_k, dim_v, dtype=torch.float64, device=q.device) if h0 is None else h0[i].clone()
        for t in range(start, stop):
            h = h * g[t].exp()[..., None, None]
            pred = (h * k[t][..., None]).sum(dim=-2)
            delta = beta[t][..., None] * (v[t] - pred)
            h = h + k[t][..., None] * delta[..., None, :]
            o[t] = (h * q[t][..., None]).sum(dim=-2) * scale
        finals.append(h)
    return o, torch.stack(finals)


def relative_l2_fp64(ref, got):
    ref = ref.detach().cpu().double()
    got = got.detach().cpu().double()
    return ((ref - got).norm() / ref.norm().clamp_min(1e-30)).item()


def cu_seqlens(seqs, device="musa"):
    bounds = [0]
    for length in seqs:
        bounds.append(bounds[-1] + length)
    return torch.tensor(bounds, dtype=torch.int32, device=device)


def _normalize(x):
    xf = x.float()
    return (xf / (xf.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(x.dtype)


def make_gdn_inputs(dtype, seqs, hq, hv, with_state, seed=1, device="musa"):
    dim = 128
    torch.manual_seed(seed)
    total = sum(seqs)
    q = _normalize(torch.randn(total, hq, dim, dtype=dtype, device=device))
    k = _normalize(torch.randn(total, hq, dim, dtype=dtype, device=device))
    v = torch.randn(total, hv, dim, dtype=dtype, device=device) * 0.3
    g = (-torch.rand(total, hv, dtype=torch.float32, device=device)) * 0.03
    beta = torch.rand(total, hv, dtype=dtype, device=device)
    state = torch.randn(len(seqs), hv, dim, dim, dtype=torch.float32, device=device) * 0.05 if with_state else None
    return q, k, v, g, beta, state


def mp32_skip_reason(device: str = "musa:0") -> str | None:
    """Return a skip reason when *device* is not mp_32, else ``None``."""
    if torch is None:  # pragma: no cover - guarded by importorskip upstream
        return "torch is not available"
    try:
        from mudnn.linear_attention.tilelang.kernels.inference.helpers import (
            MP32_ARCH,
            get_arch,
        )

        index = torch.device(device).index
        arch = get_arch(torch.musa.current_device() if index is None else index)
    except Exception:  # noqa: BLE001 - collection must stay safe without a driver
        return "the GDN MP32 features need a queryable MUSA device, and this host " "has none"
    if arch == MP32_ARCH:
        return None
    return (
        f"the fused GDN training pairing and the prefill checkpoint series are "
        f"built for MP32; this device reports {arch}"
    )
