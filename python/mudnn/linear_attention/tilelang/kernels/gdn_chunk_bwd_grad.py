"""GDN backward K3: dq/dk/dg/dbeta, grid (NT, B*H).

Consumes A (=T0 ungated inverse), the forward Hs/Delta handoffs and K2's
dv/DKst/Ds/DdLast.  q/k/do carry the TRUE length while every handoff, every
output and all FP32 scalars ([H, B, S] head-outermost) are chunk-aligned.
"""
import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T

from ._jit import GRAD_COMPILE_FLAGS, GRAD_PASS_CONFIGS
from .gdn_chunk_bwd_scan import (LOG2E, _DKST_PERM_TILE, _check_hbs_layout,
                                 dkst_perm_supported)
from .gdn_varlen import _rdp, map_dummy
from .gdn_shapes import gdn_shape_error, gdn_shape_supported


_CFG, _FLAGS = GRAD_PASS_CONFIGS, GRAD_COMPILE_FLAGS


@T.macro
def _stage_value_panels(DO, Delta, DVin, do_km, dl_km, de_km,
                        i_b, i_h, base, off, C, BV, SR,
                        qshift=None, send=None):
    """Issue one value-block's three panels as cp.async WITHOUT a drain.

    ``force_async_copy`` plus ``TL_DISABLE_THREAD_STORAGE_SYNC`` suppresses the
    commit+wait normally glued to the issue site, so the caller decides where the
    group is drained.

    ``DO`` is the caller's gradient and on the ragged path is only ``SR`` rows long,
    while ``Delta``/``DVin`` are ours and chunk-aligned. Pass ``SR=None`` on the
    padded path to keep the unpredicated read.
    """
    if True:
        for r, c in T.Parallel(C, BV):
            do_km[r, c] = (
                T.if_then_else(
                    base + r + qshift < send,
                    DO[i_b, T.max(T.min(base + r + qshift, send - 1), 0),
                       i_h, off + c], 0)
                if qshift is not None else
                (DO[i_b, base + r, i_h, off + c] if SR is None
                 else T.if_then_else(
                     base + r < SR,
                     DO[i_b, T.min(base + r, SR - 1), i_h, off + c], 0))
            )
            dl_km[r, c] = Delta[i_b, base + r, i_h, off + c]
            de_km[r, c] = DVin[i_b, base + r, i_h, off + c]


@tilelang.jit(target="musa", out_idx=[], pass_configs=_CFG, compile_flags=_FLAGS)
def gdn_chunk_bwd_grad_kernel(
    B, H, DK, DV, chunk_size=64, scale=None,
    input_dtype=T.bfloat16, threads=512, BV=64,
    use_direct_ddo_reduce=False, ragged=False,
    chunk_padded=False,
):
    assert threads == 512, "K3 shuffle reductions require exactly 512 threads"
    assert chunk_size == 64, "K3 shuffle reductions require chunk_size=64"
    use_perm_dkst = use_direct_ddo_reduce
    if scale is None:
        scale = DK ** -0.5
    C = chunk_size
    S = T.dynamic("S")
    NT = T.dynamic("NT")
    predicated = ragged or chunk_padded
    SR = T.dynamic("SR") if predicated else S
    NTV = T.dynamic("NTV")

    def rd(fn, row):
        """Read a row of a caller tensor; zero past the true length.

        The INDEX is clamped as well as the value guarded, so the address stays
        in range even if the load is speculated.
        """
        if not ragged:
            return fn(row)
        return T.if_then_else(row < SR, fn(T.min(row, SR - 1)), 0)
    if DV % BV or DV < BV:
        raise ValueError(f"K3 needs DV a multiple of BV; got DV={DV}, BV={BV}")
    if DK * BV < 8 * threads:
        raise ValueError(
            f"K3 needs DK*BV >= 8*threads; got DK={DK}, BV={BV}, threads={threads}")
    NV = DV // BV
    accum = T.float32

    @T.prim_func
    def kernel(
        Q: T.Tensor([B, SR, H, DK], input_dtype),
        K: T.Tensor([B, SR, H, DK], input_dtype),
        Beta: T.Tensor([H, B, S], accum),
        DO: T.Tensor([B, SR, H, DV], input_dtype),
        A0: T.Tensor([B, S, H, C], input_dtype),
        GPrefix: T.Tensor([H, B, S], accum),
        Delta: T.Tensor([B, S, H, DV], input_dtype),
        Hs: T.Tensor([B, H, NT, DK, DV], input_dtype),
        DVin: T.Tensor([B, S, H, DV], input_dtype),
        DKst: T.Tensor([B, S, H, DK], input_dtype),
        Ds: T.Tensor([H, B, S], accum),
        DdLast: T.Tensor([B, H, NT, DK], accum),
        DQ: T.Tensor([B, SR, H, DK] if chunk_padded else [B, S, H, DK],
                     input_dtype),
        DKo: T.Tensor([B, SR, H, DK] if chunk_padded else [B, S, H, DK],
                      input_dtype),
        DG: T.Tensor([H, B, S], accum),
        DBeta: T.Tensor([H, B, S], accum),
        CMap: T.Tensor([NTV, 4] if chunk_padded else [1, 4], T.int32),
    ):
        with T.Kernel(NT, B * H, threads=threads) as (ic, bh):
            T.annotate_min_blocks_per_sm(2)
            tx = T.get_thread_binding()
            i_b = bh // H
            i_h = bh % H
            base = ic * C
            if chunk_padded:
                qshift = T.alloc_var("int32")
                send = T.alloc_var("int32")
                qshift = CMap[ic, 1]
                send = CMap[ic, 2]

            q_km = T.alloc_shared([C, DK], input_dtype)
            q_mn = T.alloc_shared([C, DK], input_dtype)
            k_km = T.alloc_shared([C, DK], input_dtype)
            k_mn = T.alloc_shared([C, DK], input_dtype)
            W_km = T.alloc_shared([C, C], input_dtype)
            W_mn = T.alloc_shared([C, C], input_dtype)
            Hin = T.alloc_shared([DK, BV], input_dtype)
            do_km = T.alloc_shared([C, BV], input_dtype)
            dl_km = T.alloc_shared([C, BV], input_dtype)
            de_km = T.alloc_shared([C, BV], input_dtype)

            lsh = T.alloc_shared([1, C], accum, scope="shared")
            bsh = T.alloc_shared([1, C], accum, scope="shared")
            dsh = T.alloc_shared([1, C], accum, scope="shared")
            dgc = T.alloc_shared([1, C], accum, scope="shared")
            dbcS = T.alloc_shared([1, C], accum, scope="shared")
            ddoS = T.alloc_shared([1, C], accum, scope="shared")
            rsS = T.alloc_shared([1, C], accum, scope="shared")
            csS = T.alloc_shared([1, C], accum, scope="shared")
            tailS = T.alloc_shared([1], accum, scope="shared")

            out_s = T.alloc_shared([C, DK + 8], input_dtype)
            A0s = T.alloc_shared([C, C], input_dtype)
            qk = T.alloc_fragment([C, C], accum)
            dBm = T.alloc_fragment([C, C], accum)
            dTf = T.alloc_fragment([C, C], accum)
            dcy = T.alloc_shared([C, C], accum, scope="shared")
            dqc = T.alloc_fragment([C, DK], accum)
            dkc = T.alloc_fragment([C, DK], accum)
            if not use_direct_ddo_reduce:
                t3 = T.alloc_fragment([C, DK], accum)
                ddo = T.alloc_fragment([C], accum)
            hs_vec = T.alloc_local([8], input_dtype)
            tail_local = T.alloc_local([1], accum)
            tail_reduce = T.alloc_local([1], accum)
            scan_carry = T.alloc_local([1], accum)

            for c in T.Parallel(C):
                bsh[0, c] = Beta[i_h, i_b, base + c]
                lsh[0, c] = GPrefix[i_h, i_b, base + c]
                dsh[0, c] = T.exp2(lsh[0, c] * LOG2E)

            if True:
                for r, c in T.Parallel(C, DK):
                    q_km[r, c] = (
                        _rdp(lambda rw: Q[i_b, rw, i_h, c], base + r, qshift, send)
                        if chunk_padded else
                        rd(lambda rw: Q[i_b, rw, i_h, c], base + r))
                for r, c in T.Parallel(C, DK):
                    k_km[r, c] = (
                        _rdp(lambda rw: K[i_b, rw, i_h, c], base + r, qshift, send)
                        if chunk_padded else
                        rd(lambda rw: K[i_b, rw, i_h, c], base + r))

            for r, c in T.Parallel(C, DK):
                dqc[r, c] = 0.0
                dkc[r, c] = 0.0
            for r, c in T.Parallel(C, C):
                dBm[r, c] = 0.0
                dTf[r, c] = 0.0
            for vb in range(NV):
                off = vb * BV
                _stage_value_panels(DO, Delta, DVin, do_km, dl_km, de_km,
                                    i_b, i_h, base, off, C, BV,
                                    SR if ragged else None,
                                    *((qshift, send) if chunk_padded else ()))
                for u in T.serial(DK * BV // (8 * threads)):
                    for t in T.vectorized(8):
                        hs_vec[t] = Hs[
                            i_b, i_h, ic,
                            (tx + u * threads) // 8,
                            ((tx + u * threads) % 8) * (DV // 8) + vb * 8 + t,
                        ]
                    for t in T.unroll(8):
                        Hin[
                            (tx + u * threads) // 8,
                            t * 8 + ((tx + u * threads) % 8),
                        ] = hs_vec[t]
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                T.sync_threads()
                T.gemm(do_km, Hin, dqc, transpose_B=True)
                T.gemm(de_km, Hin, dkc, transpose_B=True)
                T.gemm(do_km, dl_km, dBm, transpose_B=True)
                T.gemm(de_km, dl_km, dTf, transpose_B=True)
                if vb == 0:
                    for r, c in T.Parallel(C, DK):
                        q_km[r, c] = (
                            q_km[r, c].astype(accum) * scale
                        ).astype(input_dtype)
                T.sync_threads()

            if use_direct_ddo_reduce:
                ddo_local = T.alloc_local([2], accum)
                ddo_local[0] = 0
                ddo_local[1] = 0
                row0 = (tx % 256) // 128 * 32 + (tx % 128) // 8
                col0 = tx // 256 * (DK // 2) + tx % 8
                for c in T.unroll(DK // 16):
                    col = col0 + c * 8
                    ddo_local[0] += (
                        q_km[row0, col].astype(accum) * dqc[row0, col]
                        - k_km[row0, col].astype(accum) * dkc[row0, col]
                    )
                    ddo_local[1] += (
                        q_km[row0 + 16, col].astype(accum)
                        * dqc[row0 + 16, col]
                        - k_km[row0 + 16, col].astype(accum)
                        * dkc[row0 + 16, col]
                    )
                ddo_local[0] += T.shfl_xor(ddo_local[0], 1)
                ddo_local[0] += T.shfl_xor(ddo_local[0], 2)
                ddo_local[0] += T.shfl_xor(ddo_local[0], 4)
                ddo_local[1] += T.shfl_xor(ddo_local[1], 1)
                ddo_local[1] += T.shfl_xor(ddo_local[1], 2)
                ddo_local[1] += T.shfl_xor(ddo_local[1], 4)
                if tx % 8 == 0:
                    if tx < 256:
                        rsS[0, row0] = ddo_local[0]
                        rsS[0, row0 + 16] = ddo_local[1]
                    else:
                        csS[0, row0] = ddo_local[0]
                        csS[0, row0 + 16] = ddo_local[1]
                T.sync_threads()
                for r in T.Parallel(C):
                    ddoS[0, r] = rsS[0, r] + csS[0, r]
            else:
                for r, c in T.Parallel(C, DK):
                    t3[r, c] = (
                        q_km[r, c].astype(accum) * dqc[r, c]
                        - k_km[r, c].astype(accum) * dkc[r, c]
                    )
            if not use_direct_ddo_reduce:
                T.reduce_sum(t3, ddo, dim=1, clear=True)
                for r in T.Parallel(C):
                    ddoS[0, r] = ddo[r]
            T.gemm(q_km, k_km, qk, clear_accum=True, transpose_B=True)

            if use_perm_dkst:
                dk_row0 = (tx % 256) // 128 * 32 + (tx % 128) // 8
                dk_col0 = tx // 256 * (DK // 2) + tx % 8
                for rr in T.unroll(2):
                    for t in T.vectorized(8):
                        hs_vec[t] = DKst[
                            i_b, base + dk_row0 + rr * 16, i_h,
                            tx // 256 * (DK // 2) + (tx % 8) * 8 + t,
                        ]
                    for t in T.unroll(8):
                        dkc[dk_row0 + rr * 16, dk_col0 + t * 8] = (
                            -dkc[dk_row0 + rr * 16, dk_col0 + t * 8]
                            * dsh[0, dk_row0 + rr * 16]
                            + hs_vec[t].astype(accum)
                        )
                for r, c in T.Parallel(C, DK):
                    dqc[r, c] = dqc[r, c] * dsh[0, r]
            else:
                for r, c in T.Parallel(C, DK):
                    dqc[r, c] = dqc[r, c] * dsh[0, r]
                    dkc[r, c] = (
                        -dkc[r, c] * dsh[0, r]
                        + DKst[i_b, base + r, i_h, c].astype(accum)
                    )
            T.sync_threads()

            for r, c in T.Parallel(C, DK):
                q_mn[r, c] = q_km[r, c]
                k_mn[r, c] = k_km[r, c]
            T.sync_threads()
            with T.attr(T.int32(0), "tl.force_async_copy", 1):
                for r, c in T.Parallel(C, C):
                    A0s[r, c] = A0[i_b, base + r, i_h, c]
            T.ptx_commit_group()

            for r, c in T.Parallel(C, C):
                dcy[r, c] = T.exp2(
                    T.min(lsh[0, r] - lsh[0, c], 0.0) * LOG2E
                )
            T.sync_threads()
            for r, c in T.Parallel(C, C):
                dBm[r, c] = T.if_then_else(
                    r >= c,
                    dBm[r, c] * dcy[r, c],
                    0.0,
                )
            for r, c in T.Parallel(C, C):
                qk[r, c] = dBm[r, c] * qk[r, c]
            for r, c in T.Parallel(C, C):
                W_km[r, c] = dBm[r, c].astype(input_dtype)
            for r, c in T.Parallel(C, C):
                W_mn[r, c] = dBm[c, r].astype(input_dtype)
            T.sync_threads()
            T.gemm(W_km, k_mn, dqc)
            T.gemm(W_mn, q_mn, dkc)
            T.ptx_wait_group(0)
            T.sync_threads()

            for r, c in T.Parallel(C, C):
                if r == c:
                    dbcS[0, r] = T.if_then_else(
                        bsh[0, r] != 0.0,
                        (dTf[r, c] / bsh[0, r]) / bsh[0, r],
                        0.0,
                    )
                dBm[r, c] = T.if_then_else(r > c, -dTf[r, c], 0.0)
            for r, c in T.Parallel(C, C):
                dTf[r, c] = T.if_then_else(
                    r > c, dBm[r, c] * A0s[c, r].astype(accum)
                    * dcy[r, c], 0.0)
                dBm[r, c] = T.if_then_else(
                    r > c, dBm[r, c] * dcy[r, c], 0.0)
            for r, c in T.Parallel(C, C):
                qk[r, c] += dTf[r, c]
            for r, c in T.Parallel(C, C):
                W_km[r, c] = dBm[r, c].astype(input_dtype)
            for r, c in T.Parallel(C, C):
                W_mn[r, c] = dBm[c, r].astype(input_dtype)
            T.sync_threads()
            T.gemm(W_km, k_mn, dkc)
            T.gemm(W_mn, k_mn, dkc)
            T.sync_threads()

            for r, c in T.Parallel(C, DK):
                out_s[r, c] = dkc[r, c].astype(input_dtype)
            T.sync_threads()
            for r, c in T.Parallel(C, DK):
                if chunk_padded:
                    prow = base + r + qshift
                    if prow < send:
                        DKo[i_b, prow, i_h, c] = out_s[r, c]
                else:
                    DKo[i_b, base + r, i_h, c] = out_s[r, c]

            for r, c in T.Parallel(C, C):
                q_mn[r, c] = qk[r, c].astype(input_dtype)
            T.sync_threads()
            for c in T.vectorized(8):
                hs_vec[c] = q_mn[tx // 8, (tx % 8) * 8 + c]
            tail_local[0] = 0.0
            for c in T.unroll(8):
                tail_local[0] += hs_vec[c].astype(accum)
            tail_local[0] += T.shfl_xor(tail_local[0], 1)
            tail_local[0] += T.shfl_xor(tail_local[0], 2)
            tail_local[0] += T.shfl_xor(tail_local[0], 4)
            if tx % 8 == 0:
                rsS[0, tx // 8] = tail_local[0]
            tail_local[0] = 0.0
            for r in T.serial(8):
                tail_local[0] += q_mn[(tx % 8) * 8 + r, tx // 8].astype(accum)
            tail_local[0] += T.shfl_xor(tail_local[0], 1)
            tail_local[0] += T.shfl_xor(tail_local[0], 2)
            tail_local[0] += T.shfl_xor(tail_local[0], 4)
            if tx % 8 == 0:
                csS[0, tx // 8] = tail_local[0]
            if tx < 32:
                if DK == 128:
                    tail_local[0] = (
                        DdLast[i_b, i_h, ic, tx] * dsh[0, C - 1]
                        + Ds[i_h, i_b, base + tx]
                        * T.exp2((lsh[0, C - 1] - lsh[0, tx]) * LOG2E)
                        + DdLast[i_b, i_h, ic, tx + 64] * dsh[0, C - 1]
                    ) + (
                        DdLast[i_b, i_h, ic, tx + 32] * dsh[0, C - 1]
                        + Ds[i_h, i_b, base + tx + 32]
                        * T.exp2(
                            (lsh[0, C - 1] - lsh[0, tx + 32]) * LOG2E
                        )
                        + DdLast[i_b, i_h, ic, tx + 96] * dsh[0, C - 1]
                    )
                else:
                    tail_local[0] = (
                        DdLast[i_b, i_h, ic, tx] * dsh[0, C - 1]
                        + Ds[i_h, i_b, base + tx]
                        * T.exp2((lsh[0, C - 1] - lsh[0, tx]) * LOG2E)
                    ) + (
                        DdLast[i_b, i_h, ic, tx + 32] * dsh[0, C - 1]
                        + Ds[i_h, i_b, base + tx + 32]
                        * T.exp2(
                            (lsh[0, C - 1] - lsh[0, tx + 32]) * LOG2E
                        )
                    )
                tail_reduce[0] = T.warp_reduce_sum(tail_local[0])
                if tx == 0:
                    tailS[0] = tail_reduce[0]
            T.sync_threads()
            for r in T.Parallel(C):
                scaled_dd = (
                    ddoS[0, r] * dsh[0, r]
                    - Ds[i_h, i_b, base + r]
                    * T.exp2((lsh[0, C - 1] - lsh[0, r]) * LOG2E)
                    + T.if_then_else(r == C - 1, tailS[0], 0.0)
                )
                dgc[0, r] = rsS[0, r] - csS[0, r] + scaled_dd
            for r, c in T.Parallel(C, DK):
                out_s[r, c] = (dqc[r, c] * scale).astype(input_dtype)
            T.sync_threads()
            for r, c in T.Parallel(C, DK):
                if chunk_padded:
                    prow = base + r + qshift
                    if prow < send:
                        DQ[i_b, prow, i_h, c] = out_s[r, c]
                else:
                    DQ[i_b, base + r, i_h, c] = out_s[r, c]
            if tx < 32:
                tail_local[0] = dgc[0, tx + 32]
                tail_reduce[0] = T.shfl_down(tail_local[0], 1)
                if tx < 31:
                    tail_local[0] += tail_reduce[0]
                tail_reduce[0] = T.shfl_down(tail_local[0], 2)
                if tx < 30:
                    tail_local[0] += tail_reduce[0]
                tail_reduce[0] = T.shfl_down(tail_local[0], 4)
                if tx < 28:
                    tail_local[0] += tail_reduce[0]
                tail_reduce[0] = T.shfl_down(tail_local[0], 8)
                if tx < 24:
                    tail_local[0] += tail_reduce[0]
                tail_reduce[0] = T.shfl_down(tail_local[0], 16)
                if tx < 16:
                    tail_local[0] += tail_reduce[0]
                dgc[0, tx + 32] = tail_local[0]

                scan_carry[0] = T.shfl_sync(
                    0xFFFFFFFF, tail_local[0], 0
                )
                tail_local[0] = dgc[0, tx]
                tail_reduce[0] = T.shfl_down(tail_local[0], 1)
                if tx < 31:
                    tail_local[0] += tail_reduce[0]
                tail_reduce[0] = T.shfl_down(tail_local[0], 2)
                if tx < 30:
                    tail_local[0] += tail_reduce[0]
                tail_reduce[0] = T.shfl_down(tail_local[0], 4)
                if tx < 28:
                    tail_local[0] += tail_reduce[0]
                tail_reduce[0] = T.shfl_down(tail_local[0], 8)
                if tx < 24:
                    tail_local[0] += tail_reduce[0]
                tail_reduce[0] = T.shfl_down(tail_local[0], 16)
                if tx < 16:
                    tail_local[0] += tail_reduce[0]
                tail_local[0] += scan_carry[0]
                dgc[0, tx] = tail_local[0]
            T.sync_threads()

            for r in T.Parallel(C):
                DG[i_h, i_b, base + r] = dgc[0, r]
                DBeta[i_h, i_b, base + r] = dbcS[0, r]

    return kernel


def gdn_chunk_bwd_grad(
    q, k, beta, do, A, GPrefix, Delta, Hs,
    DVin, DKst, Ds, DdLast,
    scale=None, chunk_size=64, BV=64, threads=512, vmaps=None,
):
    """Emit dq/dk/dg/dbeta from the forward and K2 backward handoffs.

    ``q``/``k``/``do`` may carry the TRUE length; ``Delta`` and every other
    handoff, plus all four outputs, are chunk-aligned.  ``beta``/``Ds`` arrive
    and ``dg``/``dbeta`` leave head-outermost as [H, B, S] (a chunk is then one
    contiguous 256 B line); ``gdn_chunk_bwd`` un-transposes before returning
    them to autograd.  The decay everywhere comes from ``GPrefix``; there is no
    separate ``g`` input.  ``DVin`` is K2's ``dv = beta*DRhs``, and ``DKst`` must be BF16,
    DK-permuted by K2 wherever ``dkst_perm_supported`` holds.
    """
    B, S_real, H, DK = q.shape
    S = Delta.shape[1]
    ragged = S_real != S
    DV = do.shape[-1]
    if scale is None:
        scale = DK ** -0.5
    if not gdn_shape_supported(B, S, H, DK, DV, chunk_size):
        raise ValueError(
            gdn_shape_error(
                "direct-Delta schedule", B, S, H, DK, DV, chunk_size
            )
        )
    if DKst.dtype != torch.bfloat16:
        raise ValueError("the direct-Delta schedule requires a BF16 DKst tensor")
    for name, t in (("beta", beta), ("GPrefix", GPrefix), ("Ds", Ds)):
        _check_hbs_layout(t, B, S, H, name=name)
    shape = (B, S, H, DK, DV, chunk_size)
    decode_perm = dkst_perm_supported(*shape)
    if decode_perm and (DK, BV, threads) != _DKST_PERM_TILE:
        raise ValueError(
            f"permuted DKst decode assumes (DK, BV, threads)={_DKST_PERM_TILE},"
            f" got {(DK, BV, threads)}; K2 has already published permuted "
            f"columns "
            "for this shape, so decoding here would silently corrupt dk"
        )
    kern = gdn_chunk_bwd_grad_kernel(
        B, H, DK, DV, chunk_size=chunk_size, scale=scale,
        BV=BV, threads=threads,
        use_direct_ddo_reduce=decode_perm, ragged=ragged,
        chunk_padded=vmaps is not None)
    n_row = q.shape[1] if vmaps is not None else S
    dq = torch.empty(B, n_row, H, DK, device=q.device, dtype=torch.bfloat16)
    dk = torch.empty(B, n_row, H, DK, device=q.device, dtype=torch.bfloat16)
    dg = torch.empty(H, B, S, device=q.device, dtype=torch.float32)
    dbeta = torch.empty(H, B, S, device=q.device, dtype=torch.float32)
    kern(
        q.bfloat16(), k.bfloat16(), beta.float(), do.bfloat16(),
        A.bfloat16(), GPrefix.float(), Delta.bfloat16(), Hs.bfloat16(),
        DVin.bfloat16(), DKst.bfloat16(),
        Ds.float(), DdLast.float(),
        dq, dk, dg, dbeta,
        vmaps.cmap if vmaps is not None else map_dummy(q.device, 4))
    return dq, dk, dg, dbeta
