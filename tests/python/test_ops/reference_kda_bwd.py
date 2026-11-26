"""Chunked KDA forward + hand-derived chunked backward, in pure torch.

This file is the specification for the MP31 mutlass backward kernel.

  * ``chunked_forward`` mirrors ``test_kda.py::blockwise_kda_reference`` (the
    chunk formulation the K1 prepare + K2 recurrence implement) while
    recording every per-chunk intermediate, with an optional switch that
    reproduces the bf16/fp16 workspace rounding.
  * ``chunked_backward`` is the hand-derived reverse-mode chain rule for that
    exact chunk formulation: a reverse scan over chunks carrying the state
    gradient ``dS`` of shape ``[H, K, V]`` (K rows, V columns).
  * ``oracle_backward`` gets the same gradients from autograd on the
    differentiable (unquantized) forward -- the ground truth.

Tiles are ``[H, C, D]``; the state is ``[H, K, V]`` internally and ``[H, V, K]``
at the public boundary (V-major, matching the state pool layout).
"""

from __future__ import annotations

import torch

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


def _exp2_ftz(x: torch.Tensor) -> torch.Tensor:
    y = torch.exp2(torch.clamp(x, max=127.0))
    return torch.where(x < -126.0, torch.zeros_like(y), y)


def _ste_round(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Round to ``dtype`` but pass the gradient through unchanged.

    The kernels treat a rounded operand as the forward's true input, so the
    derivative of the quantisation itself is taken to be the identity.
    """
    xr = x.to(dtype).float()
    return x + (xr - x).detach()


def l2norm_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _unit_lower_inverse(L: torch.Tensor) -> torch.Tensor:
    """``(I + tril(L, -1))^-1`` for a strictly lower ``L`` (``[..., C, C]``).

    Autograd-safe: built row by row with no in-place writes.
    """
    C = L.shape[-1]
    eye = torch.eye(C, dtype=L.dtype, device=L.device)
    batch = L.shape[:-2]
    rows = []
    for i in range(C):
        r = eye[i].expand(*batch, C)
        if i > 0:
            prev = torch.stack(rows, dim=-2)  # [..., i, C]
            r = r - (L[..., i : i + 1, :i] @ prev)[..., 0, :]
        rows.append(r)
    return torch.stack(rows, dim=-2)


def _a_matrix(L: torch.Tensor, bs: torch.Tensor) -> torch.Tensor:
    """``a = (I + tril(L,-1))^-1 diag(bs)`` by forward substitution."""
    inv = _unit_lower_inverse(L)
    return inv * bs[..., None, :]


def chunked_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    scale: float,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    use_qk_l2norm: bool,
    chunk: int = 16,
    quantize: bool = False,
    use_shift: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Chunked KDA forward for one sequence.

    ``q, k, v, g``: ``[H, T, D]``; ``beta``: ``[H, T]``;
    ``initial_state``: ``[H, V, K]`` (public V-major layout).
    Returns ``(o, final_state, cache)``, ``final_state`` in ``[H, V, K]``.
    """
    H, T, D = q.shape
    dev = q.device
    rnd = (lambda x: _ste_round(x, out_dtype)) if quantize else (lambda x: x)

    if use_qk_l2norm:
        qn, kn = l2norm_last(q.float()), l2norm_last(k.float())
        if quantize:
            qn = _ste_round(qn, out_dtype)
            kn = _ste_round(kn, out_dtype)
    else:
        qn, kn = q.float(), k.float()

    o = torch.empty(H, T, D, device=dev, dtype=torch.float32)
    s = initial_state.float().transpose(-1, -2).contiguous()  # [H, K, V]
    cache: list[dict] = []

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        actual = end - start
        qb = torch.zeros(H, chunk, D, device=dev, dtype=torch.float32)
        kb = torch.zeros_like(qb)
        vb = torch.zeros_like(qb)
        gb = torch.zeros_like(qb)
        bb = torch.full((H, chunk), -80.0, device=dev, dtype=torch.float32)
        qb[:, :actual] = qn[:, start:end]
        kb[:, :actual] = kn[:, start:end]
        vb[:, :actual] = v.float()[:, start:end]
        gb[:, :actual] = g.float()[:, start:end]
        bb[:, :actual] = beta.float()[:, start:end]

        gate = (
            lower_bound
            * LOG2E
            * torch.sigmoid(
                torch.exp(A_log.float())[:, None, None]
                * (gb + dt_bias.float()[:, None, :])
            )
        )
        if actual < chunk:
            gate[:, actual:] = 0.0
        G = gate.cumsum(dim=1)
        Gtot = G[:, actual - 1]
        # K1 re-centres nothing; `use_shift=False` reproduces that exactly.
        shift = (0.5 * (G[:, 0] + Gtot)) if use_shift else torch.zeros_like(Gtot)

        e_G = _exp2_ftz(G)
        e_Gs = _exp2_ftz(G - shift[:, None, :])
        e_sG = _exp2_ftz(shift[:, None, :] - G)
        e_Gt = _exp2_ftz(Gtot[:, None, :] - G)
        e_Gtot = _exp2_ftz(Gtot)

        kd = rnd(kb * e_G)
        qd = rnd(qb * e_G * scale)
        kt_ = rnd(kb * e_Gs)
        qp = rnd(qb * e_Gs * scale)
        ki = rnd(kb * e_sG)
        kr = rnd(kb * e_Gt)

        bs = torch.sigmoid(bb)
        L = rnd(torch.tril(kt_ @ ki.transpose(-1, -2), -1) * bs[:, :, None])
        a = rnd(_a_matrix(L, bs))
        p = rnd(torch.tril(qp @ ki.transpose(-1, -2)))

        s_in = rnd(s)
        tmp = rnd(vb - kd @ s_in)
        u = rnd(a @ tmp)
        o[:, start:end] = rnd(qd @ s_in + p @ u)[:, :actual]
        s = s * e_Gtot[:, :, None] + kr.transpose(-1, -2) @ u

        cache.append(
            dict(
                start=start,
                actual=actual,
                qb=qb,
                kb=kb,
                vb=vb,
                gb=gb,
                bb=bb,
                gate=gate,
                G=G,
                Gtot=Gtot,
                shift=shift,
                e_G=e_G,
                e_Gs=e_Gs,
                e_sG=e_sG,
                e_Gt=e_Gt,
                e_Gtot=e_Gtot,
                kd=kd,
                qd=qd,
                kt_=kt_,
                qp=qp,
                ki=ki,
                kr=kr,
                bs=bs,
                L=L,
                a=a,
                p=p,
                s_in=s_in,
                tmp=tmp,
                u=u,
            )
        )

    return o, s.transpose(-1, -2).contiguous(), cache


def chunked_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dO: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    scale: float,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    use_qk_l2norm: bool,
    d_final_state: torch.Tensor | None = None,
    chunk: int = 16,
    quantize: bool = False,
    use_shift: bool = True,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Hand-derived gradients of :func:`chunked_forward`.

    ``dO``: ``[H, T, D]``; ``d_final_state``: ``[H, V, K]``.  All returned
    state gradients use the public ``[H, V, K]`` layout.
    """
    H, T, D = q.shape
    dev = q.device
    _, _, cache = chunked_forward(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        scale=scale,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm=use_qk_l2norm,
        chunk=chunk,
        quantize=quantize,
        use_shift=use_shift,
        out_dtype=out_dtype,
    )

    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    dg = torch.zeros_like(g, dtype=torch.float32)
    dbeta = torch.zeros_like(beta, dtype=torch.float32)
    dA_log = torch.zeros_like(A_log, dtype=torch.float32)
    ddt_bias = torch.zeros_like(dt_bias, dtype=torch.float32)

    collect = []
    dS = torch.zeros(H, D, D, device=dev, dtype=torch.float32)
    if d_final_state is not None:
        dS = dS + d_final_state.float().transpose(-1, -2).contiguous()

    for entry in reversed(cache):
        start, actual = entry["start"], entry["actual"]
        s_in, u, tmp, a, p, bs, L = (
            entry["s_in"],
            entry["u"],
            entry["tmp"],
            entry["a"],
            entry["p"],
            entry["bs"],
            entry["L"],
        )
        kd, qd, kt_, qp, ki, kr = (
            entry["kd"],
            entry["qd"],
            entry["kt_"],
            entry["qp"],
            entry["ki"],
            entry["kr"],
        )
        e_G, e_Gs, e_sG, e_Gt, e_Gtot = (
            entry["e_G"],
            entry["e_Gs"],
            entry["e_sG"],
            entry["e_Gt"],
            entry["e_Gtot"],
        )
        gb, bb = entry["gb"], entry["bb"]

        dOc = torch.zeros(H, chunk, D, device=dev, dtype=torch.float32)
        dOc[:, :actual] = dO.float()[:, start : start + actual]

        # o = qd @ s_in + p @ u
        dqd = dOc @ s_in.transpose(-1, -2)
        dS_in = qd.transpose(-1, -2) @ dOc
        dp = dOc @ u.transpose(-1, -2)
        dU = p.transpose(-1, -2) @ dOc

        # s_out = e_Gtot * s_in + kr^T @ u
        dkr = u @ dS.transpose(-1, -2)
        dU = dU + kr @ dS
        dS_prev = e_Gtot[:, :, None] * dS
        dGtot = (dS * s_in).sum(dim=-1) * e_Gtot * LN2

        # u = a @ tmp
        da = dU @ tmp.transpose(-1, -2)
        dtmp = a.transpose(-1, -2) @ dU

        # tmp = v - kd @ s_in
        dv[:, start : start + actual] += dtmp[:, :actual]
        dkd = -(dtmp @ s_in.transpose(-1, -2))
        dS_in = dS_in - kd.transpose(-1, -2) @ dtmp

        dS_next_iter = dS_prev + dS_in

        # a = (I + tril(L,-1))^-1 diag(bs)
        # a = inv @ diag(bs), inv = (I+tril(L,-1))^-1.  The adjoint of
        # d_a = -inv d_M a + inv diag(d_bs) is d_M = -inv^T d_a a^T.
        inv = _unit_lower_inverse(L)
        a_t = a.transpose(-1, -2)
        dM = -(inv.transpose(-1, -2) @ da @ a_t)
        dL = torch.tril(dM, -1)
        dbs = (inv.transpose(-1, -2) @ da).diagonal(dim1=-2, dim2=-1).contiguous()

        # L = tril(kt_ @ ki^T, -1) * bs[:, None]
        dLtmp = torch.tril(dL * bs[:, :, None], -1)
        dkt_ = dLtmp @ ki
        dki = dLtmp.transpose(-1, -2) @ kt_
        dbs = dbs + (dL * (kt_ @ ki.transpose(-1, -2))).sum(dim=-1)

        # p = tril(qp @ ki^T)
        dp_t = torch.tril(dp)
        dqp = dp_t @ ki
        dki = dki + dp_t.transpose(-1, -2) @ qp

        # decay vectors -> dq, dk, dG, dshift, dGtot
        dk[:, start : start + actual] += (
            dkd * e_G + dkt_ * e_Gs + dki * e_sG + dkr * e_Gt
        )[:, :actual]
        dq[:, start : start + actual] += ((dqd * e_G + dqp * e_Gs) * scale)[:, :actual]

        dG = (dkd * kd + dqd * qd + dkt_ * kt_ + dqp * qp - dki * ki - dkr * kr) * LN2
        dGtot = dGtot + (dkr * kr).sum(dim=1) * LN2

        # G = cumsum(gate); Gtot = G[actual-1]; shift = 0.5*(G[0] + Gtot).
        # With `use_shift=False` the shift is a constant, so it contributes no
        # derivative at all -- exactly the K1 formulation the kernel uses.
        if use_shift:
            dshift = (
                -(dkt_ * kt_).sum(dim=1) - (dqp * qp).sum(dim=1) + (dki * ki).sum(dim=1)
            ) * LN2
            dG[:, 0] = dG[:, 0] + 0.5 * dshift
            dG[:, actual - 1] = dG[:, actual - 1] + dGtot + 0.5 * dshift
        else:
            dG[:, actual - 1] = dG[:, actual - 1] + dGtot
        collect.append((dG.clone(), dGtot.clone()))
        dgate = torch.flip(torch.cumsum(torch.flip(dG, [1]), dim=1), [1])

        # gate = lb * log2e * sigmoid(exp(A_log) * (g + dt_bias))
        eA = torch.exp(A_log.float())[:, None, None]
        x = eA * (gb + dt_bias.float()[:, None, :])
        sig = torch.sigmoid(x)
        dx = dgate * (lower_bound * LOG2E * sig * (1.0 - sig))
        if actual < chunk:
            keep = torch.zeros(chunk, device=dev, dtype=torch.float32)
            keep[:actual] = 1.0
            dx = dx * keep[None, :, None]
        dg[:, start : start + actual] += (dx * eA)[:, :actual]
        dA_log += (dx * eA * (gb + dt_bias.float()[:, None, :])).sum(dim=(1, 2))
        ddt_bias += (dx * eA).sum(dim=1)

        dbeta[:, start : start + actual] += (dbs * bs * (1.0 - bs))[:, :actual]

        dS = dS_next_iter

    if use_qk_l2norm:
        qn, kn = l2norm_last(q.float()), l2norm_last(k.float())
        rq = torch.rsqrt((q.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        rk = torch.rsqrt((k.float() ** 2).sum(-1, keepdim=True) + 1e-6)
        dq = rq * (dq - (dq * qn).sum(-1, keepdim=True) * qn)
        dk = rk * (dk - (dk * kn).sum(-1, keepdim=True) * kn)

    return dict(
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        dbeta=dbeta,
        dstate0=dS.transpose(-1, -2).contiguous(),
        dA_log=dA_log,
        ddt_bias=ddt_bias,
        _dG=collect,
    )


def oracle_backward(
    q,
    k,
    v,
    g,
    beta,
    dO,
    initial_state,
    *,
    scale,
    A_log,
    dt_bias,
    lower_bound,
    use_qk_l2norm,
    d_final_state=None,
    chunk=16,
    quantize=False,
    use_shift=True,
):
    """Autograd reference: differentiate the unquantized chunked forward."""
    leaves = [
        t.clone().detach().requires_grad_(True)
        for t in (q, k, v, g, beta, initial_state, A_log, dt_bias)
    ]
    qq, kk, vv, gg, bb, s0, al, dtb = leaves
    o, fs, _ = chunked_forward(
        qq,
        kk,
        vv,
        gg,
        bb,
        s0,
        scale=scale,
        A_log=al,
        dt_bias=dtb,
        lower_bound=lower_bound,
        use_qk_l2norm=use_qk_l2norm,
        chunk=chunk,
        quantize=quantize,
        use_shift=use_shift,
    )
    loss = (o * dO).sum()
    if d_final_state is not None:
        loss = loss + (fs * d_final_state).sum()
    grads = torch.autograd.grad(loss, leaves)
    dq, dk, dv, dg, dbeta, ds0, dal, ddtb = grads
    return dict(
        dq=dq,
        dk=dk,
        dv=dv,
        dg=dg,
        dbeta=dbeta,
        dstate0=ds0,  # s0 is already [H, V, K]
        dA_log=dal,
        ddt_bias=ddtb,
    )
