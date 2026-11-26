"""Reverse chunk scan for the B1 GDN backward: classify, guard, replay, head.

Both per-chunk boundary handoffs are rebuilt from ``q``/``do`` instead of being
stored, reproducing their BF16 roundings so the replay stays bit-identical.
``q``/``k``/``do`` may carry the TRUE length; every handoff is chunk-aligned.
"""

import math

import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T

from ._jit import SCAN_COMPILE_FLAGS, SCAN_PASS_CONFIGS
from .gdn_shapes import (_CHUNK, _MAX_CLASSIFY_FLAGS, gdn_head_dim_error,
                         gdn_shape_error, gdn_shape_supported,
                         head_dim_supported)
from .gdn_varlen import _rdp, map_dummy


def _pin_sqmma_transpose_a_layout(*buffers):
    """Pin transpose_A A-operands to the instruction-M chunked SQMMA swizzle."""
    T.annotate_layout(
        {
            buf: tilelang.layout.make_sqmma_swizzled_layout(
                buf, continuity=int(buf.shape[-1]) // 2, k_major=False
            )
            for buf in buffers
        },
        allow_reannotation=True,
        allow_buffer_region=True,
    )


LOG2E = 1.4426950408889634
_FAST_LOG_LIMIT = math.log(1e-4)

_CFG, _FLAGS = SCAN_PASS_CONFIGS, SCAN_COMPILE_FLAGS


def _check_hbs_layout(tensor, B, S, H, name="Beta"):
    """Refuse a handoff that is not head-outermost [H, B, S].

    [B, S, H] has the same element count, so a swap would be reinterpreted
    silently and surface only as a wrong gradient.
    """
    want = (H, B, S)
    if tuple(tensor.shape) != want:
        raise ValueError(f"{name} must be {want}, got {tuple(tensor.shape)}")


@tilelang.jit(
    target="musa", out_idx=[], pass_configs=_CFG, compile_flags=_FLAGS,
)
def k2_guard_classify_kernel(
    B, H, chunk_size=64, fast_log_limit=_FAST_LOG_LIMIT, has_dht=False,
    threads=512, chunk_padded=False,
):
    """Classify heads FAST/SLOW and compact the slow ones to low block indices.

    The replay that follows is gridded ``B*H`` and its shared-memory footprint
    leaves one block resident per SM, so its wall time is the max over SMs of the
    sum of that SM's serial replays. Publishing the slow heads contiguously in
    ``Order[0 .. count-1]`` keeps every launch one replay deep while ``count`` does
    not exceed the SM count.

    ``Order`` is filled with -1 past ``count``; those blocks retire immediately.
    """
    C = chunk_size
    S = T.dynamic("S")
    accum = T.float32
    BH = B * H
    assert BH <= _MAX_CLASSIFY_FLAGS, "one shared flag per (batch, head)"
    NSH_CAP = _MAX_CLASSIFY_FLAGS
    NSEQ = T.dynamic("NSEQ")
    ORD = T.dynamic("ORD")
    NTV = T.dynamic("NTV")

    @T.prim_func
    def kernel(
        GPrefix: T.Tensor([H, B, S], accum),
        Order: T.Tensor([ORD] if chunk_padded else [BH], T.int32),
        Fast: T.Tensor([NSEQ, H] if chunk_padded else [B, H], T.int32),
        CMap: T.Tensor([NTV, 4] if chunk_padded else [1, 4], T.int32),
        nseq: T.int32,
    ):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            flag = T.alloc_shared([NSH_CAP if chunk_padded else BH],
                                  T.int32, scope="shared")
            nt = S // C
            last = nt if has_dht else nt - 1
            ntest = last - 1
            if chunk_padded:
                nsh = T.alloc_var("int32")
                nsh = nseq * H
                for i in T.Parallel(NSH_CAP):
                    if i < nsh:
                        flag[i] = 1
                T.sync_threads()
                total = (S // C) * H
                for j in T.serial(T.ceildiv(total, threads)):
                    idx = j * threads + tx
                    if idx < total:
                        gc = idx // H
                        i_h = idx % H
                        prev_last = T.alloc_var("int32")
                        prev_last = CMap[T.max(gc - 1, 0), 3] % 2
                        first = T.alloc_var("int32")
                        first = T.if_then_else(gc == 0, 1, prev_last)
                        drop_last = 0 if has_dht else 1
                        if CMap[gc, 3] // 2 == 1:
                            flag[CMap[gc, 0] * H + i_h] = 0
                        if first == 0 and (CMap[gc, 3] % 2 == 0 or drop_last == 0):
                            if GPrefix[i_h, 0, gc * C + C - 1] > fast_log_limit:
                                flag[CMap[gc, 0] * H + i_h] = 0
                T.sync_threads()
                for i in T.Parallel(NSH_CAP):
                    if i < nsh:
                        Fast[i // H, i % H] = flag[i]
                T.sync_threads()
                if tx == 0:
                    n = T.alloc_local([1], T.int32)
                    n[0] = 0
                    for bh in T.serial(nsh):
                        if flag[bh] == 0:
                            Order[n[0]] = bh
                            n[0] = n[0] + 1
                    for bh in T.serial(nsh):
                        if bh >= n[0]:
                            Order[bh] = -1
            else:
                for i in T.Parallel(BH):
                    flag[i] = T.if_then_else(ntest > 0, 1, 0)
                T.sync_threads()
                total = BH * ntest
                for j in T.serial(T.ceildiv(total, threads)):
                    idx = j * threads + tx
                    if idx < total:
                        bh = idx // ntest
                        ic = 1 + idx % ntest
                        if GPrefix[
                            bh % H, bh // H, ic * C + C - 1
                        ] > fast_log_limit:
                            flag[bh] = 0
                T.sync_threads()
                for i in T.Parallel(BH):
                    Fast[i // H, i % H] = flag[i]
                T.sync_threads()
                if tx == 0:
                    n = T.alloc_local([1], T.int32)
                    n[0] = 0
                    for bh in T.serial(BH):
                        if flag[bh] == 0:
                            Order[n[0]] = bh
                            n[0] = n[0] + 1
                    for bh in T.serial(BH):
                        if bh >= n[0]:
                            Order[bh] = -1

    return kernel


@tilelang.jit(
    target="musa", out_idx=[], pass_configs=_CFG, compile_flags=_FLAGS,
)
def k2_fused_guard_kernel(
    B, H, DK, DV, chunk_size=64,
    input_dtype=T.bfloat16, threads=512,
    has_dht=False, scale=None, ragged=False, chunk_padded=False,
):
    """For slow heads, publish every exact chunk boundary.

    No handoff tensor exists any more, so this kernel rebuilds ``Bmat^T@do``
    and ``(scale*d*q)^T@do`` on its own serial path.  Fast heads run zero
    iterations.

    S and NT are dynamic; B and H stay compile-time.  Every use of them here is
    a shape, a grid or a loop bound -- there is no tiling decision that needs S
    at build time.
    """
    C = chunk_size
    S = T.dynamic("S")
    NT = T.dynamic("NT")
    predicated = ragged or chunk_padded
    SR = T.dynamic("SR") if predicated else S
    NSEQ1 = T.dynamic("NSEQ1")

    def rd(fn, row):
        """Read a row of a caller tensor; zero past the true length.

        The index is clamped as well as the value guarded, so the address is in
        range even if the load gets speculated.
        """
        if not ragged:
            return fn(row)
        return T.if_then_else(row < SR, fn(T.min(row, SR - 1)), 0)

    accum = T.float32
    if scale is None:
        scale = DK ** -0.5
    if not head_dim_supported(DK, DV, C):
        raise ValueError(gdn_head_dim_error("k2_fused_guard_kernel", DK, DV, C))
    assert threads == 512

    @T.prim_func
    def kernel(
        K: T.Tensor([B, SR, H, DK], input_dtype),
        Qin: T.Tensor([B, SR, H, DK], input_dtype),
        DOin: T.Tensor([B, SR, H, DV], input_dtype),
        Beta: T.Tensor([H, B, S], accum),
        A0: T.Tensor([B, S, H, C], input_dtype),
        GPrefix: T.Tensor([H, B, S], accum),
        dht: T.Tensor([B, H, DK, DV], accum),
        SMap: T.Tensor([NSEQ1, 2] if chunk_padded else [1, 2], T.int32),
        Order: T.Tensor([(NSEQ1 - 1) * H] if chunk_padded else [B * H], T.int32),
        Bnd: T.Tensor([B, H, NT, DK, DV], input_dtype),
    ):
        with T.Kernel((NSEQ1 - 1) * H if chunk_padded else B * H,
                      threads=threads) as blk:
            slot = T.alloc_local([1], T.int32)
            slot[0] = Order[blk]
            bh = T.max(slot[0], 0)
            i_s = bh // H if chunk_padded else 0
            i_b = 0 if chunk_padded else bh // H
            i_h = bh % H
            if chunk_padded:
                c0 = T.alloc_var("int32")
                c_end = T.alloc_var("int32")
                qshift = T.alloc_var("int32")
                send = T.alloc_var("int32")
                c0 = SMap[i_s, 1]
                c_end = SMap[i_s + 1, 1]
                qshift = SMap[i_s, 0] - c0 * C
                send = SMap[i_s + 1, 0]

            k_km = T.alloc_shared([C, DK], input_dtype)
            dh_shared = T.alloc_shared([DK, DV], input_dtype)
            tri_km = T.alloc_shared([C, C], input_dtype)
            dd_s = T.alloc_shared([C, DV], input_dtype)
            mnT = T.alloc_shared([DK, C], input_dtype)
            q_bt = T.alloc_shared([C, DK], input_dtype)
            q_at = T.alloc_shared([C, DK], input_dtype)
            _pin_sqmma_transpose_a_layout(q_at)
            do_stage = T.alloc_shared([C, DV], input_dtype)

            lsh = T.alloc_shared([1, C], accum, scope="shared")
            bsh = T.alloc_shared([1, C], accum, scope="shared")
            dsh = T.alloc_shared([1, C], accum, scope="shared")
            row_scale = T.alloc_shared([1, C], accum, scope="shared")
            neg_db = T.alloc_shared([1, C], accum, scope="shared")

            dh_frag = T.alloc_fragment([DK, DV], accum)
            tmp_cv = T.alloc_fragment([C, DV], accum)
            qk_frag = T.alloc_fragment([C, C], accum)
            qtdo = T.alloc_fragment([DK, DV], accum)

            for r, c in T.Parallel(DK, DV):
                if has_dht:
                    dh_frag[r, c] = dht[i_b, i_h, r, c]
                else:
                    dh_frag[r, c] = 0.0

            steps = T.if_then_else(slot[0] >= 0,
                                   c_end - c0 if chunk_padded else NT, 0)
            for it in T.serial(steps):
                ic = (c_end - 1 - it) if chunk_padded else (NT - 1 - it)
                base = ic * C

                for c in T.Parallel(C):
                    bsh[0, c] = Beta[i_h, i_b, base + c]
                for c in T.Parallel(C):
                    lsh[0, c] = GPrefix[i_h, i_b, base + c]
                for r, c in T.Parallel(C, DK):
                    k_km[r, c] = (
                        _rdp(lambda rw: K[i_b, rw, i_h, c], base + r, qshift, send)
                        if chunk_padded else
                        rd(lambda rw: K[i_b, rw, i_h, c], base + r))
                for r, c in T.Parallel(C, DK):
                    q_at[r, c] = (
                        _rdp(lambda rw: Qin[i_b, rw, i_h, c], base + r, qshift, send)
                        if chunk_padded else
                        rd(lambda rw: Qin[i_b, rw, i_h, c], base + r))
                for r, c in T.Parallel(C, DV):
                    do_stage[r, c] = (
                        _rdp(lambda rw: DOin[i_b, rw, i_h, c], base + r, qshift, send)
                        if chunk_padded else
                        rd(lambda rw: DOin[i_b, rw, i_h, c], base + r))
                for r, c in T.Parallel(DK, DV):
                    dh_shared[r, c] = dh_frag[r, c].astype(input_dtype)
                T.sync_threads()
                for c in T.Parallel(C):
                    dsh[0, c] = T.exp2(lsh[0, c] * LOG2E)
                for r, c in T.Parallel(C, DK):
                    q_bt[r, c] = (
                        q_at[r, c].astype(accum) * scale
                    ).astype(input_dtype)
                T.sync_threads()

                for r, c in T.Parallel(DK, DV):
                    Bnd[i_b, i_h, ic, r, c] = dh_shared[r, c]

                for c in T.Parallel(C):
                    row_scale[0, c] = T.exp2(
                        (lsh[0, C - 1] - lsh[0, c]) * LOG2E
                    )
                    neg_db[0, c] = -(dsh[0, c] * bsh[0, c])
                T.gemm(
                    k_km, q_bt, qk_frag,
                    clear_accum=True, transpose_B=True,
                )
                T.gemm(k_km, dh_shared, tmp_cv, clear_accum=True)
                for r, c in T.Parallel(C, DV):
                    tmp_cv[r, c] = tmp_cv[r, c] * row_scale[0, r]
                for r, c in T.Parallel(C, C):
                    tri_km[r, c] = T.if_then_else(
                        c >= r,
                        qk_frag[r, c]
                        * T.exp2((lsh[0, c] - lsh[0, r]) * LOG2E),
                        0.0,
                    ).astype(input_dtype)
                T.sync_threads()
                T.gemm(tri_km, do_stage, tmp_cv, clear_accum=False)
                for r, c in T.Parallel(C, DV):
                    dd_s[r, c] = tmp_cv[r, c].astype(input_dtype)
                T.sync_threads()

                for r, c in T.Parallel(C, C):
                    tri_km[r, c] = T.if_then_else(
                        c >= r,
                        T.exp2((lsh[0, c] - lsh[0, r]) * LOG2E)
                        * T.if_then_else(
                            c == r,
                            1.0,
                            A0[i_b, base + c, i_h, r].astype(accum),
                        ),
                        0.0,
                    ).astype(input_dtype)
                T.sync_threads()

                T.gemm(tri_km, dd_s, tmp_cv, clear_accum=True)
                for r, c in T.Parallel(C, DV):
                    dd_s[r, c] = tmp_cv[r, c].astype(input_dtype)

                for r, c in T.Parallel(C, DK):
                    q_at[r, c] = (
                        q_bt[r, c].astype(accum) * dsh[0, r]
                    ).astype(input_dtype)
                T.sync_threads()
                T.gemm(
                    q_at, do_stage, qtdo,
                    clear_accum=True, transpose_A=True,
                )

                for r, c in T.Parallel(DK, DV):
                    dh_frag[r, c] = (
                        dh_frag[r, c] * dsh[0, C - 1]
                        + qtdo[r, c].astype(input_dtype).astype(accum)
                    )
                for kk, r in T.Parallel(DK, C):
                    mnT[kk, r] = (
                        neg_db[0, r] * k_km[r, kk].astype(accum)
                    ).astype(input_dtype)
                T.sync_threads()
                T.gemm(mnT, dd_s, dh_frag)
                T.sync_threads()

    return kernel


def k2_fused_guard(
    k, q, do, beta, A, GPrefix, dht=None, scale=None,
    vmaps=None,
):
    """Classify heads into ``Fast`` and publish ``Bnd`` for the slow ones."""
    B, S_real, H, DK = k.shape
    S = A.shape[1]
    ragged = S_real != S
    DV = do.shape[-1]
    has_dht = dht is not None
    if not has_dht:
        dht = torch.empty(B, H, DK, DV, device=k.device, dtype=torch.float32)
    _check_hbs_layout(beta, B, S, H)
    packed = vmaps is not None
    nseq = (vmaps.smap.shape[0] - 1) if packed else 1
    Order = torch.empty(nseq * H if packed else B * H,
                        device=k.device, dtype=torch.int32)
    Fast = torch.empty(nseq if packed else B, H,
                       device=k.device, dtype=torch.int32)
    classify = k2_guard_classify_kernel(
        B, H, fast_log_limit=_FAST_LOG_LIMIT, has_dht=has_dht,
        chunk_padded=packed,
    )
    classify(GPrefix.float(), Order, Fast,
             vmaps.cmap if packed else map_dummy(k.device, 4), nseq)
    kernel = k2_fused_guard_kernel(
        B, H, DK, DV,
        has_dht=has_dht, scale=scale, ragged=ragged, chunk_padded=packed,
    )
    Bnd = torch.empty(
        B, H, S // _CHUNK, DK, DV, device=k.device, dtype=torch.bfloat16,
    )
    kernel(
        k.bfloat16(), q.bfloat16(), do.bfloat16(), beta.float(),
        A.bfloat16(), GPrefix.float(), dht.float(),
        vmaps.smap if packed else map_dummy(k.device, 2), Order, Bnd,
    )
    return Fast, Bnd


def _k2_fused_common_prim(
    B, H, DK, DV, chunk_size=64,
    input_dtype=T.bfloat16, has_dht=False, dkst_dtype=T.bfloat16,
    scale=None, dkst_perm=False, ragged=False, chunk_padded=False,
):
    C = chunk_size
    S = T.dynamic("S")
    NT = T.dynamic("NT")
    predicated = ragged or chunk_padded
    SR = T.dynamic("SR") if predicated else S
    NTV = T.dynamic("NTV")
    NSQ = T.dynamic("NSQ")

    def rd(fn, row):
        """Read a row of a caller tensor; zero past the true length."""
        if not ragged:
            return fn(row)
        return T.if_then_else(row < SR, fn(T.min(row, SR - 1)), 0)

    accum = T.float32
    if scale is None:
        scale = DK ** -0.5
    if not head_dim_supported(DK, DV, C):
        raise ValueError(gdn_head_dim_error("fused K2 replay", DK, DV, C))
    assert DK in (C, 2 * C), "K2 implements one- and two-split forms only"
    HK = DK // 2

    @T.prim_func
    def kernel(
        K: T.Tensor([B, SR, H, DK], input_dtype),
        Qin: T.Tensor([B, SR, H, DK], input_dtype),
        DOin: T.Tensor([B, SR, H, DV], input_dtype),
        Beta: T.Tensor([H, B, S], accum),
        A0: T.Tensor([B, S, H, C], input_dtype),
        GPrefix: T.Tensor([H, B, S], accum),
        Delta: T.Tensor([B, S, H, DV], input_dtype),
        Hs: T.Tensor([B, H, NT, DK, DV], input_dtype),
        Bnd: T.Tensor([B, H, NT, DK, DV], input_dtype),
        Fast: T.Tensor([NSQ, H] if chunk_padded else [B, H], T.int32),
        dht: T.Tensor([B, H, DK, DV], accum),
        DRhs: T.Tensor([B, S, H, DV], input_dtype),
        DKst: T.Tensor([B, S, H, DK], dkst_dtype),
        Ds: T.Tensor([H, B, S], accum),
        DdLast: T.Tensor([B, H, NT, DK], accum),
        CMap: T.Tensor([NTV, 4] if chunk_padded else [1, 4], T.int32),
    ):
        with T.Kernel(NT, B * H, threads=512) as (ic, bh):
            i_b = bh // H
            i_h = bh % H
            base = ic * C
            tx = T.get_thread_binding()
            if chunk_padded:
                i_bs = T.alloc_var("int32")
                qshift = T.alloc_var("int32")
                send = T.alloc_var("int32")
                is_last = T.alloc_var("int32")
                i_bs = CMap[ic, 0]
                qshift = CMap[ic, 1]
                send = CMap[ic, 2]
                is_last = CMap[ic, 3] % 2

            do_stage = T.alloc_shared([C, DV], input_dtype)
            if DK == 2 * C:
                q_lo = T.alloc_shared([C, HK], input_dtype)
                q_hi = T.alloc_shared([C, HK], input_dtype)
                _pin_sqmma_transpose_a_layout(q_lo, q_hi)
            else:
                q_sp = T.alloc_shared([C, C], input_dtype)
                _pin_sqmma_transpose_a_layout(q_sp)

            k_km = T.alloc_shared([C, DK], input_dtype)
            q_own = T.alloc_shared([C, DK], input_dtype)
            dl_s = T.alloc_shared([C, DV], input_dtype)
            dd_s = T.alloc_shared([C, DV], input_dtype)
            dh_shared = T.alloc_shared([DK, DV], input_dtype)
            dh_shared_t = T.alloc_shared([DK, DV], input_dtype)
            tri_km = T.alloc_shared([C, C], input_dtype)
            a0_s = T.alloc_shared([C, C + 8], input_dtype, scope="shared")
            dkst_s = T.alloc_shared([C, DK + 8], dkst_dtype)

            lsh = T.alloc_shared([1, C], accum, scope="shared")
            bsh = T.alloc_shared([1, C], accum, scope="shared")
            row_scale = T.alloc_shared([1, C], accum, scope="shared")
            ds_halves = T.alloc_shared([2, C], accum, scope="shared")
            ddl_halves = T.alloc_shared([2, DK], accum, scope="shared")

            tmp_cv = T.alloc_fragment([C, DV], accum)
            dks_frag = T.alloc_fragment([DK, C], accum)
            dh_frag = T.alloc_fragment([DK, DV], input_dtype)
            Tt_frag = T.alloc_fragment([C, C], input_dtype)
            qk_frag = T.alloc_fragment([C, C], accum)
            ds_local = T.alloc_local([2], accum)
            ddl_local = T.alloc_local([DK // 32], accum)

            for c in T.Parallel(C):
                lsh[0, c] = GPrefix[i_h, i_b, base + c]
                bsh[0, c] = Beta[i_h, i_b, base + c]
            for r, c in T.Parallel(C, C):
                a0_s[r, c] = A0[i_b, base + r, i_h, c]

            if Fast[i_bs if chunk_padded else i_b, i_h] != 0:
                if (is_last != 0) if chunk_padded else (ic == NT - 1):
                    for r, c in T.Parallel(DK, DV):
                        if has_dht:
                            dh_frag[r, c] = dht[
                                i_b, i_h, r, c
                            ].astype(input_dtype)
                        else:
                            dh_frag[r, c] = T.cast(0.0, input_dtype)
                    for r, c in T.Parallel(DK, DV):
                        dh_shared[r, c] = dh_frag[r, c]
                else:
                    if DK == 2 * C:
                        for r, c in T.Parallel(C, HK):
                            q_lo[r, c] = (
                                (
                                    (_rdp(lambda rw: Qin[i_b, rw, i_h, c], base + C + r,
                                          qshift, send) if chunk_padded else
                                     rd(lambda rw: Qin[i_b, rw, i_h, c], base + C + r)).astype(accum)
                                    * scale
                                ).astype(input_dtype).astype(accum)
                                * T.exp2(
                                    GPrefix[i_h, i_b, base + C + r] * LOG2E
                                )
                            ).astype(input_dtype)
                            q_hi[r, c] = (
                                (
                                    (_rdp(lambda rw: Qin[i_b, rw, i_h, c + HK],
                                          base + C + r, qshift, send) if chunk_padded else
                                     rd(lambda rw: Qin[i_b, rw, i_h, c + HK],
                                        base + C + r)).astype(accum) * scale
                                ).astype(input_dtype).astype(accum)
                                * T.exp2(
                                    GPrefix[i_h, i_b, base + C + r] * LOG2E
                                )
                            ).astype(input_dtype)
                    else:
                        for r, c in T.Parallel(C, C):
                            q_sp[r, c] = (
                                (
                                    (_rdp(lambda rw: Qin[i_b, rw, i_h, c], base + C + r,
                                          qshift, send) if chunk_padded else
                                     rd(lambda rw: Qin[i_b, rw, i_h, c], base + C + r)).astype(accum)
                                    * scale
                                ).astype(input_dtype).astype(accum)
                                * T.exp2(
                                    GPrefix[i_h, i_b, base + C + r] * LOG2E
                                )
                            ).astype(input_dtype)
                    for r, c in T.Parallel(C, DV):
                        do_stage[r, c] = (
                            _rdp(lambda rw: DOin[i_b, rw, i_h, c], base + C + r,
                                 qshift, send) if chunk_padded else
                            rd(lambda rw: DOin[i_b, rw, i_h, c], base + C + r))
                    T.sync_threads()
                    if DK == 2 * C:
                        T.gemm(
                            q_lo, do_stage, tmp_cv,
                            clear_accum=True, transpose_A=True,
                        )
                        for r, c in T.Parallel(HK, DV):
                            dh_shared[r, c] = tmp_cv[r, c].astype(input_dtype)
                        T.gemm(
                            q_hi, do_stage, tmp_cv,
                            clear_accum=True, transpose_A=True,
                        )
                        for r, c in T.Parallel(HK, DV):
                            dh_shared[HK + r, c] = tmp_cv[r, c].astype(input_dtype)
                    else:
                        T.gemm(
                            q_sp, do_stage, tmp_cv,
                            clear_accum=True, transpose_A=True,
                        )
                        for r, c in T.Parallel(C, DV):
                            dh_shared[r, c] = tmp_cv[r, c].astype(input_dtype)
                    T.sync_threads()
                    for r, c in T.Parallel(DK, DV):
                        dh_frag[r, c] = dh_shared[r, c]
            else:
                for r, c in T.Parallel(DK, DV):
                    dh_frag[r, c] = Bnd[i_b, i_h, ic, r, c]
                for r, c in T.Parallel(DK, DV):
                    dh_shared[r, c] = dh_frag[r, c]

            for r, c in T.Parallel(C, DK):
                k_km[r, c] = (
                    _rdp(lambda rw: K[i_b, rw, i_h, c], base + r, qshift, send)
                    if chunk_padded else
                    rd(lambda rw: K[i_b, rw, i_h, c], base + r))
            for r, c in T.Parallel(C, DV):
                dl_s[r, c] = Delta[i_b, base + r, i_h, c]
            T.sync_threads()

            for c in T.Parallel(C):
                row_scale[0, c] = T.exp2(
                    (lsh[0, C - 1] - lsh[0, c]) * LOG2E
                )

            T.gemm(
                k_km, dh_shared, tmp_cv,
                clear_accum=True, wg_wait=-1,
            )
            ddl_row0 = (
                (tx % 256) // 128 * (DK // 2) + (tx % 128) // 8
            )
            ddl_col0 = tx // 256 * (DV // 2) + tx % 8
            for rr in T.unroll(DK // 32):
                ddl_local[rr] = 0.0
            for rr in T.unroll(DK // 32):
                for vv in T.unroll(DV // 16):
                    ddl_local[rr] += (
                        Hs[
                            i_b,
                            i_h,
                            ic,
                            ddl_row0 + rr * 16,
                            ((ddl_col0 + vv * 8) % 8) * (DV // 8)
                            + (ddl_col0 + vv * 8) // 8,
                        ].astype(accum)
                        * dh_shared[
                            ddl_row0 + rr * 16,
                            ddl_col0 + vv * 8,
                        ].astype(accum)
                    )
            for rr in T.unroll(DK // 32):
                ddl_local[rr] += T.shfl_xor(ddl_local[rr], 1)
                ddl_local[rr] += T.shfl_xor(ddl_local[rr], 2)
                ddl_local[rr] += T.shfl_xor(ddl_local[rr], 4)
            if tx % 8 == 0:
                half = tx // 256
                for rr in T.unroll(DK // 32):
                    ddl_halves[half, ddl_row0 + rr * 16] = ddl_local[rr]
            for r, c in T.Parallel(C, C):
                Tt_frag[r, c] = T.if_then_else(
                    c >= r,
                    T.exp2((lsh[0, c] - lsh[0, r]) * LOG2E)
                    * T.if_then_else(
                        c == r,
                        1.0,
                        a0_s[c, r].astype(accum),
                    ),
                    0.0,
                ).astype(input_dtype)
            T.wait_wgmma(0)
            T.sync_threads()
            for r, c in T.Parallel(C, DV):
                do_stage[r, c] = (
                    _rdp(lambda rw: DOin[i_b, rw, i_h, c], base + r, qshift, send)
                    if chunk_padded else
                    rd(lambda rw: DOin[i_b, rw, i_h, c], base + r))
            for r in T.Parallel(DK):
                DdLast[i_b, i_h, ic, r] = (
                    ddl_halves[0, r] + ddl_halves[1, r]
                )

            ds_local[0] = 0.0
            ds_local[1] = 0.0
            ds_row0 = (tx % 256) // 128 * 32 + (tx % 128) // 8
            ds_col0 = tx // 256 * (DV // 2) + tx % 8
            for c in T.unroll(DV // 16):
                ds_local[0] += (
                    dl_s[ds_row0, ds_col0 + c * 8].astype(accum)
                    * tmp_cv[ds_row0, ds_col0 + c * 8]
                )
                ds_local[1] += (
                    dl_s[ds_row0 + 16, ds_col0 + c * 8].astype(accum)
                    * tmp_cv[ds_row0 + 16, ds_col0 + c * 8]
                )
            ds_local[0] += T.shfl_xor(ds_local[0], 1)
            ds_local[0] += T.shfl_xor(ds_local[0], 2)
            ds_local[0] += T.shfl_xor(ds_local[0], 4)
            ds_local[1] += T.shfl_xor(ds_local[1], 1)
            ds_local[1] += T.shfl_xor(ds_local[1], 2)
            ds_local[1] += T.shfl_xor(ds_local[1], 4)
            if tx % 8 == 0:
                half = tx // 256
                ds_halves[half, ds_row0] = ds_local[0]
                ds_halves[half, ds_row0 + 16] = ds_local[1]

            for r, c in T.Parallel(C, DK):
                q_own[r, c] = (
                    (_rdp(lambda rw: Qin[i_b, rw, i_h, c], base + r, qshift, send)
                     if chunk_padded else
                     rd(lambda rw: Qin[i_b, rw, i_h, c], base + r)).astype(accum) * scale
                ).astype(input_dtype)

            T.sync_threads()
            T.gemm(
                q_own, k_km, qk_frag,
                clear_accum=True, transpose_B=True,
            )

            for r, c in T.Parallel(C, C):
                tri_km[c, r] = T.if_then_else(
                    r >= c,
                    qk_frag[r, c]
                    * T.exp2((lsh[0, r] - lsh[0, c]) * LOG2E),
                    0.0,
                ).astype(input_dtype)
            for r, c in T.Parallel(C, DV):
                tmp_cv[r, c] = tmp_cv[r, c] * row_scale[0, r]
            T.sync_threads()
            T.gemm(tri_km, do_stage, tmp_cv, clear_accum=False)
            for r, c in T.Parallel(C, DV):
                dd_s[r, c] = tmp_cv[r, c].astype(input_dtype)
            for r in T.Parallel(C):
                Ds[i_h, i_b, base + r] = (
                    ds_halves[0, r] + ds_halves[1, r]
                )
            T.sync_threads()
            for r, c in T.Parallel(C, C):
                tri_km[r, c] = Tt_frag[r, c]
            T.sync_threads()

            T.gemm(
                tri_km, dd_s, tmp_cv,
                clear_accum=True, wg_wait=-1,
            )
            T.wait_wgmma(0)
            for r, c in T.Parallel(DK, DV):
                dh_shared_t[r, c] = dh_frag[r, c]
            T.sync_threads()

            for r, c in T.Parallel(C, DV):
                dd_s[r, c] = (tmp_cv[r, c] * bsh[0, r]).astype(input_dtype)
            T.sync_threads()

            T.gemm(
                dh_shared_t, dl_s, dks_frag,
                clear_accum=True, transpose_B=True, wg_wait=-1,
            )
            for r, c in T.Parallel(C, DV):
                DRhs[i_b, base + r, i_h, c] = dd_s[r, c]
            T.wait_wgmma(0)
            T.sync_threads()
            for c, r in T.Parallel(DK, C):
                if dkst_perm:
                    dkst_s[
                        r, 64 * (c // 64) + (c % 8) * 8 + (c % 64) // 8
                    ] = (dks_frag[c, r] * row_scale[0, r]).astype(dkst_dtype)
                else:
                    dkst_s[r, c] = (
                        dks_frag[c, r] * row_scale[0, r]
                    ).astype(dkst_dtype)
            T.sync_threads()
            for r, c in T.Parallel(C, DK):
                DKst[i_b, base + r, i_h, c] = dkst_s[r, c]

    return kernel


k2_fused_common_kernel = tilelang.jit(
    target="musa", out_idx=[],
    pass_configs=_CFG, compile_flags=_FLAGS,
)(_k2_fused_common_prim)


@tilelang.jit(
    target="musa",
    out_idx=[-1],
    pass_configs=_CFG,
    compile_flags=_FLAGS,
)
def k2_fused_head_kernel(
    B, H, DK, DV, chunk_size=64,
    input_dtype=T.bfloat16, has_dht=False, scale=None,
):
    C = chunk_size
    S = T.dynamic("S")
    NT = T.dynamic("NT")
    accum = T.float32
    if scale is None:
        scale = DK ** -0.5
    if not head_dim_supported(DK, DV, C):
        raise ValueError(gdn_head_dim_error("k2_fused_head_kernel", DK, DV, C))

    @T.prim_func
    def kernel(
        K: T.Tensor([B, S, H, DK], input_dtype),
        Qin: T.Tensor([B, S, H, DK], input_dtype),
        DOin: T.Tensor([B, S, H, DV], input_dtype),
        GPrefix: T.Tensor([H, B, S], accum),
        Bnd: T.Tensor([B, H, NT, DK, DV], input_dtype),
        Fast: T.Tensor([B, H], T.int32),
        dht: T.Tensor([B, H, DK, DV], accum),
        DRhs: T.Tensor([B, S, H, DV], input_dtype),
        dh0: T.Tensor([B, H, DK, DV], accum),
    ):
        with T.Kernel(B * H, threads=512) as bh:
            i_b = bh // H
            i_h = bh % H

            dsh = T.alloc_shared([1, C], accum, scope="shared")
            gsh = T.alloc_shared([1, C], accum, scope="shared")
            dd_s = T.alloc_shared([C, DV], input_dtype)
            mnT = T.alloc_shared([DK, C], input_dtype)
            q_stage = T.alloc_shared([C, DK], input_dtype)
            _pin_sqmma_transpose_a_layout(q_stage)
            do_stage = T.alloc_shared([C, DV], input_dtype)
            dh_frag = T.alloc_fragment([DK, DV], accum)
            qtdo = T.alloc_fragment([DK, DV], accum)

            if Fast[i_b, i_h] != 0:
                if NT == 1:
                    for r, c in T.Parallel(DK, DV):
                        if has_dht:
                            dh_frag[r, c] = dht[
                                i_b, i_h, r, c
                            ].astype(input_dtype).astype(accum)
                        else:
                            dh_frag[r, c] = 0.0
                else:
                    for c in T.Parallel(C):
                        gsh[0, c] = T.exp2(
                            GPrefix[i_h, i_b, C + c] * LOG2E
                        )
                    for r, c in T.Parallel(C, DK):
                        q_stage[r, c] = (
                            (
                                Qin[i_b, C + r, i_h, c].astype(accum) * scale
                            ).astype(input_dtype).astype(accum)
                            * gsh[0, r]
                        ).astype(input_dtype)
                    for r, c in T.Parallel(C, DV):
                        do_stage[r, c] = DOin[i_b, C + r, i_h, c]
                    T.sync_threads()
                    T.gemm(
                        q_stage, do_stage, qtdo,
                        clear_accum=True, transpose_A=True,
                    )
                    for r, c in T.Parallel(DK, DV):
                        dh_frag[r, c] = qtdo[
                            r, c
                        ].astype(input_dtype).astype(accum)
            else:
                for r, c in T.Parallel(DK, DV):
                    dh_frag[r, c] = Bnd[
                        i_b, i_h, 0, r, c
                    ].astype(accum)

            for r in T.Parallel(C):
                dsh[0, r] = T.exp2(
                    GPrefix[i_h, i_b, r] * LOG2E
                )
            for r, c in T.Parallel(C, DV):
                dd_s[r, c] = DRhs[i_b, r, i_h, c]
            for r, c in T.Parallel(C, DK):
                q_stage[r, c] = (
                    (
                        Qin[i_b, r, i_h, c].astype(accum) * scale
                    ).astype(input_dtype).astype(accum)
                    * dsh[0, r]
                ).astype(input_dtype)
            for r, c in T.Parallel(C, DV):
                do_stage[r, c] = DOin[i_b, r, i_h, c]
            T.sync_threads()
            T.gemm(
                q_stage, do_stage, qtdo,
                clear_accum=True, transpose_A=True,
            )

            for r, c in T.Parallel(DK, DV):
                dh_frag[r, c] = (
                    dh_frag[r, c] * dsh[0, C - 1]
                    + qtdo[r, c].astype(input_dtype).astype(accum)
                )
            for kk, r in T.Parallel(DK, C):
                mnT[kk, r] = (
                    -dsh[0, r] * K[i_b, r, i_h, kk].astype(accum)
                ).astype(input_dtype)
            T.sync_threads()
            T.gemm(mnT, dd_s, dh_frag)
            T.sync_threads()
            for r, c in T.Parallel(DK, DV):
                dh0[i_b, i_h, r, c] = dh_frag[r, c]

    return kernel


def dkst_perm_supported(B, S, H, DK, DV, chunk_size=64):
    """Whether the permuted DKst column order is decodable.

    The permutation ``p(c) = 64*(c//64) + (c%8)*8 + (c%64)//8`` inverts the
    [C, DK] accumulator's lane mapping, which depends on the TILE SHAPE and THREAD
    COUNT only -- not on B, S or H.

    Producer and consumer must agree: a mismatch is SILENT and leaves dk wrong while
    dv/dg/dbeta stay clean. Both sides call this function so they cannot drift.
    """
    return DK == 128 and DV == 128 and chunk_size == 64


# The (DK, BV, threads) the permuted-DKst lane map is derived for; asserted at
# the K3 decode site because a mismatch corrupts dk silently.
_DKST_PERM_TILE = (128, 64, 512)


def k2_fused_scan(
    k, q, do, beta, A, GPrefix, Delta, Hs, Bnd, Fast,
    dht=None, chunk_size=64, need_dh0=True, scale=None, vmaps=None,
):
    """Reverse replay for the B1 no-DDelta contract, without stored handoffs.

    The per-chunk boundary is rebuilt from ``q``/``do`` of chunk ``ic+1`` and
    ``Bmat^T@do`` from ``q``/``do`` of the own chunk, so neither handoff tensor
    exists and the pre kernel is not launched at all.  ``Bnd``/``Fast`` are the
    guard's outputs, ``beta`` must be [H, B, S], and the returned ``Ds`` is
    published in the same layout; the scan publishes dv = beta*DRhs.
    """
    B, S_real, H, DK = k.shape
    S = Delta.shape[1]
    ragged = S_real != S
    DV = Delta.shape[-1]
    shape = (B, S, H, DK, DV, chunk_size)
    if not gdn_shape_supported(B, S, H, DK, DV, chunk_size):
        raise ValueError(
            gdn_shape_error("fused replay", B, S, H, DK, DV, chunk_size)
        )
    has_dht = dht is not None
    if not has_dht:
        dht = torch.empty(
            B, H, DK, DV, device=k.device, dtype=torch.float32
        )
    _check_hbs_layout(beta, B, S, H)
    dkst_perm = dkst_perm_supported(*shape)
    common_kernel = k2_fused_common_kernel(
        B, H, DK, DV,
        chunk_size=chunk_size,
        has_dht=has_dht,
        dkst_dtype=T.bfloat16,
        scale=scale,
        dkst_perm=dkst_perm,
        ragged=ragged,
        chunk_padded=vmaps is not None,
    )
    dev, NTc = k.device, S // chunk_size
    DRhs = torch.empty(B, S, H, DV, device=dev, dtype=torch.bfloat16)
    DKst = torch.empty(B, S, H, DK, device=dev, dtype=torch.bfloat16)
    Ds = torch.empty(H, B, S, device=dev, dtype=torch.float32)
    DdLast = torch.empty(B, H, NTc, DK, device=dev, dtype=torch.float32)
    common_kernel(
        k.bfloat16(), q.bfloat16(), do.bfloat16(), beta.float(), A.bfloat16(),
        GPrefix.float(), Delta.bfloat16(), Hs.bfloat16(), Bnd.bfloat16(),
        Fast, dht.float(), DRhs, DKst, Ds, DdLast,
        vmaps.cmap if vmaps is not None else map_dummy(k.device, 4),
    )
    common = (DRhs, DKst, Ds, DdLast)
    if not need_dh0:
        return (*common, None)
    if vmaps is not None:
        raise NotImplementedError("packed varlen does not carry dh0")
    head_kernel = k2_fused_head_kernel(
        B, H, DK, DV,
        chunk_size=chunk_size,
        has_dht=has_dht,
        scale=scale,
    )
    dh0 = head_kernel(
        k.bfloat16(), q.bfloat16(), do.bfloat16(),
        GPrefix.float(), Bnd.bfloat16(), Fast, dht.float(), common[0],
    )
    return (*common, dh0)


def k2_fused(
    q, k, beta, do, A, GPrefix, Delta, Hs, dht=None, scale=None,
    chunk_size=64, need_dh0=True, vmaps=None,
):
    """The whole K2 handoff: guard + fused replay, with no pre kernel.

    ``beta`` arrives head-outermost [H, B, S] and ``Ds`` is published in the
    same layout; ``_check_hbs_layout`` refuses a mismatched ``Beta`` rather
    than reinterpreting it, because [B, S, H] has the same element count and a
    silent swap would only show up as a wrong gradient.
    """
    Fast, Bnd = k2_fused_guard(
        k, q, do, beta, A, GPrefix, dht=dht, scale=scale, vmaps=vmaps,
    )
    return k2_fused_scan(
        k, q, do, beta, A, GPrefix, Delta, Hs, Bnd, Fast,
        dht=dht, chunk_size=chunk_size, need_dh0=need_dh0, scale=scale,
        vmaps=vmaps,
    )
