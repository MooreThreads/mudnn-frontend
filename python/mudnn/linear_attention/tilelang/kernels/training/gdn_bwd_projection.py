# SPDX-License-Identifier: Apache-2.0
"""Value-panel state projection and chunk-level gradient accumulation for chunk-64 GDN."""


import tilelang
import tilelang.language as T

from .._jit import GRAD_COMPILE_FLAGS
from .gdn_bwd_chunked import (
    BV,
    C,
    CONFIG,
    PAIR_ARRAYS,
    PAIR_P,
    PAIR_THREADS,
    _exp,
    _pair_pack,
    _pair_sums,
    _pairq_shape,
    _sum_eight,
    _is_symmetric_cut,
)

PROJECTION_THREADS = 512
STATE_FLAGS = [flag.replace("-mtgpu-maxregcnt=121", "-mtgpu-maxregcnt=80") for flag in GRAD_COMPILE_FLAGS]
FUSED_CONFIG = {**CONFIG, tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: False}
FUSED_FLAGS = [flag.replace("-mtgpu-maxregcnt=121", "-mtgpu-maxregcnt=200") for flag in GRAD_COMPILE_FLAGS]

# Warp partition of T.gemm for the fused kernels' accumulator ownership. MP32 SQMMA places warp groups of four
# warps along M while M / (4 m_warp) >= 32 and 32 n_warp <= N, and the Square policy (T.gemm's default) picks the
# most balanced such (m_warp, n_warp) (src/backend/musa/op/gemm.cc ComputeWarpGroupPartition, k_m_per_warp = 4,
# k_n_per_warp = 8). For the shapes here (M = C = 64, N = DV / DK or C) that is (8, 2) at 16 warps and (8, 4) at
# 32 warps: eight warps along M, i.e. 8 rows per warp, and `threads // 256` column parts - the ownership the
# `owner` / `pairl` layouts and the pair loops below assume. Passed explicitly so that requirement stays visible.
GEMM_POLICY = T.GemmWarpPolicy.Square


def _sqmma_warp_partition(M, N, threads):
    """Python mirror of the Square-policy MP32 SQMMA warp partition (used only for build-time asserts)."""
    warps = threads // 32
    assert warps % 4 == 0, "SQMMA warp groups need a multiple of 4 warps"
    best, best_score = (4, warps // 4), None
    for m_warp in range(4, min(warps, M // 4) + 1, 4):
        if warps % m_warp:
            continue
        n_warp = warps // m_warp
        if n_warp > N // 8:
            continue
        score = abs((M / (m_warp * 4)) / (N / (n_warp * 8)) - M / N)
        if best_score is None or score < best_score:
            best, best_score = (m_warp, n_warp), score
    assert best[0] * best[1] == warps, f"no Square partition for {M}x{N} on {threads} threads"
    return best


def _gemm_policy_for(threads, shapes):
    """T.gemm policy for a fused kernel of `threads` threads; asserts the partition its layouts depend on."""
    P = threads // 256
    for M, N in shapes:
        got = _sqmma_warp_partition(M, N, threads)
        assert got == (8, P), (
            f"T.gemm would partition {M}x{N} as {got} on {threads} threads, but the accumulator ownership needs "
            f"(8, {P})")
    return GEMM_POLICY


@T.macro
def _fragment_dot_rows(matrix, sample, parts, output, width):
    # Preserve MMA row ownership instead of redistributing accumulators.
    values = T.alloc_fragment((2, C, 4), "float32")
    T.annotate_layout(
        {
            values: T.Fragment(
                (2, C, 4),
                forward_thread_fn=lambda part, i, lane: part * (PROJECTION_THREADS // 2) + i % (PROJECTION_THREADS // 8) * 4 + lane,
                forward_index_fn=lambda part, i, lane: i // (PROJECTION_THREADS // 8),
            )
        }
    )
    for part, i, lane in T.Parallel(2, C, 4):
        value = T.alloc_local((1,), "float32")
        other = T.alloc_local((1,), "float32")
        value[0] = 0
        for j in T.serial(width // 8):
            column = part * (width // 2) + j * 4 + lane
            value[0] += matrix[i, column] * T.cast(sample[i, column], "float32")
        other[0] = T.shfl_down(value[0], 2)
        if lane < 2:
            value[0] += other[0]
        other[0] = T.shfl_down(value[0], 1)
        if lane == 0:
            value[0] += other[0]
        values[part, i, lane] = value[0]
    T.sync_threads()
    for part, i, lane in T.Parallel(2, C, 4):
        if lane == 0:
            parts[part, i] = values[part, i, lane]
    T.sync_threads()
    for i in T.Parallel(C):
        output[i] = parts[0, i] + parts[1, i]
    T.sync_threads()


@T.macro
def _sum_warp(value, lane):
    other = T.alloc_local((1,), "float32")
    other[0] = T.shfl_down(value[0], 16)
    if lane < 16:
        value[0] += other[0]
    other[0] = T.shfl_down(value[0], 8)
    if lane < 8:
        value[0] += other[0]
    _sum_eight(value, lane)


@T.macro
def _fragment_dot_rows3(m0, s0, m1, s1, m2, s2, parts, o0, o1, o2, width):
    # Three row dots sharing one set of barriers; same MMA-row ownership as _fragment_dot_rows.
    layout = T.Fragment(
        (2, C, 4),
        forward_thread_fn=lambda part, i, lane: part * (PROJECTION_THREADS // 2) + i % (PROJECTION_THREADS // 8) * 4 + lane,
        forward_index_fn=lambda part, i, lane: i // (PROJECTION_THREADS // 8),
    )
    v0 = T.alloc_fragment((2, C, 4), "float32")
    v1 = T.alloc_fragment((2, C, 4), "float32")
    v2 = T.alloc_fragment((2, C, 4), "float32")
    T.annotate_layout({v0: layout, v1: layout, v2: layout})
    for part, i, lane in T.Parallel(2, C, 4):
        a0 = T.alloc_local((1,), "float32")
        a1 = T.alloc_local((1,), "float32")
        a2 = T.alloc_local((1,), "float32")
        other = T.alloc_local((1,), "float32")
        a0[0], a1[0], a2[0] = 0, 0, 0
        for j in T.serial(width // 8):
            column = part * (width // 2) + j * 4 + lane
            a0[0] += m0[i, column] * T.cast(s0[i, column], "float32")
            a1[0] += m1[i, column] * T.cast(s1[i, column], "float32")
            a2[0] += m2[i, column] * T.cast(s2[i, column], "float32")
        other[0] = T.shfl_down(a0[0], 2)
        if lane < 2:
            a0[0] += other[0]
        other[0] = T.shfl_down(a1[0], 2)
        if lane < 2:
            a1[0] += other[0]
        other[0] = T.shfl_down(a2[0], 2)
        if lane < 2:
            a2[0] += other[0]
        other[0] = T.shfl_down(a0[0], 1)
        if lane == 0:
            a0[0] += other[0]
        other[0] = T.shfl_down(a1[0], 1)
        if lane == 0:
            a1[0] += other[0]
        other[0] = T.shfl_down(a2[0], 1)
        if lane == 0:
            a2[0] += other[0]
        v0[part, i, lane] = a0[0]
        v1[part, i, lane] = a1[0]
        v2[part, i, lane] = a2[0]
    T.sync_threads()
    for part, i, lane in T.Parallel(2, C, 4):
        if lane == 0:
            parts[part, i] = v0[part, i, lane]
            parts[2 + part, i] = v1[part, i, lane]
            parts[4 + part, i] = v2[part, i, lane]
    T.sync_threads()
    for i in T.Parallel(C):
        o0[i] = parts[0, i] + parts[1, i]
        o1[i] = parts[2, i] + parts[3, i]
        o2[i] = parts[4, i] + parts[5, i]
    T.sync_threads()


@T.macro
def _load_tokens(src, dst, start, end, head, col, width):
    if start + C <= end:
        T.copy(src[start : start + C, head, col : col + width], dst)
    else:
        for i, d in T.Parallel(C, width):
            dst[i, d] = T.if_then_else(start + i < end, src[T.min(start + i, end - 1), head, col + d], 0)


@T.macro
def _load_tokens_at(src, dst, row, start, end, head, col, width):
    if start + C <= end:
        T.copy(src[start : start + C, head, col : col + width], dst[row : row + C, :])
    else:
        for i, d in T.Parallel(C, width):
            dst[row + i, d] = T.if_then_else(start + i < end, src[T.min(start + i, end - 1), head, col + d], 0)


@T.macro
def _store_rows(src, dst, start, end, head, col, width):
    for i, d in T.Parallel(C, width):
        if start + i < end:
            dst[start + i, head, col + d] = src[i, d]


@T.macro
def _store_tokens(src, dst, start, end, head, col, width):
    if start + C <= end:
        T.copy(src[:, :width], dst[start : start + C, head, col : col + width])
    else:
        for i, d in T.Parallel(C, width):
            if start + i < end:
                dst[start + i, head, col + d] = src[i, d]


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=STATE_FLAGS)
def state_projection(H, DK, DV, dtype, cu_dtype, has_final, fused):
    """Per (chunk, value panel): end-of-chunk state gradient D, z = W^T dU and dV.

    ``fused`` derives D_c = SourceMap_{c+1} dO_{c+1} (DSFinal for the last chunk), valid when
    every transfer is cut; otherwise D is read from the serial scan output. z is written in
    zero-padded chunk layout so the gradient kernel can load it without guards.
    """
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")
    NV = DV // BV

    @T.prim_func
    def kernel(
        DO: T.Tensor((tokens, H, DV), dtype),
        B: T.Tensor((tokens, H), "float32"),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        CP: T.Tensor((capacity, H, DV, DK), dtype),
        Y: T.Tensor((capacity, H, C, DV), dtype),
        Exps: T.Tensor((capacity, H, 2, C), "float32"),
        StateDZ: T.Tensor((capacity, H, C, DK), dtype),
        SourceMap: T.Tensor((capacity, H, DK, C), dtype),
        SourceDZ: T.Tensor((capacity, H, C, C), dtype),
        DSFinal: T.Tensor((seqs, H, DV, DK), "float32"),
        DS: T.Tensor((capacity, H, DV, DK), dtype),
        ZC: T.Tensor((capacity, H, C, DV), dtype),
        DY: T.Tensor((tokens, H, DV), dtype),
        SR: T.Tensor((NV, capacity, H), "float32"),
        BP: T.Tensor((NV, tokens, H), "float32"),
    ):
        with T.Kernel(capacity, H, NV, threads=PROJECTION_THREADS) as (slot, head, panel):
            seq = T.alloc_var("int32")
            chunk = T.alloc_var("int32")
            offset = T.alloc_var("int32")
            seq, chunk, offset = -1, 0, 0
            for n in T.serial(seqs):
                count = T.ceildiv(CU[n + 1] - CU[n], C)
                if slot >= offset and slot < offset + count:
                    seq, chunk = n, slot - offset
                offset += count
            if seq >= 0:
                start, end = CU[seq] + chunk * C, CU[seq + 1]
                col = panel * BV
                state = T.alloc_shared((BV, DK), dtype)
                dzm = T.alloc_shared((C, DK), dtype)
                dst = T.alloc_shared((DK, BV), dtype)
                dsh = T.alloc_shared((BV, DK + 8), dtype)
                zm = T.alloc_shared((C, C), dtype)
                dos = T.alloc_shared((C, BV), dtype)
                y = T.alloc_shared((C, BV), dtype)
                out = T.alloc_shared((C, BV + 8), dtype)
                dot_parts = T.alloc_shared((2, C), "float32")
                srow = T.alloc_shared((BV,), "float32")
                rows = T.alloc_shared((C,), "float32")
                beta = T.alloc_shared((C,), "float32")
                z = T.alloc_fragment((C, BV), "float32")
                _load_tokens(DO, dos, start, end, head, col, BV)
                T.copy(SourceDZ[slot, head, :, :], zm)
                T.copy(CP[slot, head, col : col + BV, :], state)
                T.copy(StateDZ[slot, head, :, :], dzm)
                T.copy(Y[slot, head, :, col : col + BV], y)
                for i in T.Parallel(C):
                    beta[i] = T.if_then_else(start + i < end, B[T.min(start + i, end - 1), head], 0)
                if fused:
                    smn = T.alloc_shared((DK, C), dtype)
                    nd = T.alloc_shared((C, BV), dtype)
                    dt = T.alloc_fragment((DK, BV), "float32")
                    # slot + 1 may be an unwritten capacity slot (or another sequence); never multiply it by zero.
                    if start + C < end:
                        T.copy(SourceMap[slot + 1, head, :, :], smn)
                    else:
                        for d, i in T.Parallel(DK, C):
                            smn[d, i] = 0
                    _load_tokens(DO, nd, start + C, end, head, col, BV)
                    T.sync_threads()
                    T.gemm(smn, nd, dt, clear_accum=True)
                    if has_final:
                        for d, v in T.Parallel(DK, BV):
                            dt[d, v] += T.if_then_else(start + C < end, 0, DSFinal[seq, head, col + v, d])
                    for d, v in T.Parallel(DK, BV):
                        dst[d, v] = dt[d, v]
                        dsh[v, d] = dt[d, v]
                    T.sync_threads()
                    T.copy(dsh[:, :DK], DS[slot, head, col : col + BV, :])
                else:
                    T.copy(DS[slot, head, col : col + BV, :], dsh[:, :DK])
                    T.sync_threads()
                    for d, v in T.Parallel(DK, BV):
                        dst[d, v] = dsh[v, d]
                    T.sync_threads()
                for v, lane in T.Parallel(BV, 8):
                    value = T.alloc_local((1,), "float32")
                    value[0] = 0
                    for j in T.serial(DK // 8):
                        value[0] += T.cast(dsh[v, j * 8 + lane], "float32") * T.cast(state[v, j * 8 + lane], "float32")
                    _sum_eight(value, lane)
                    if lane == 0:
                        srow[v] = value[0]
                T.gemm(zm, dos, z, clear_accum=True)
                T.gemm(dzm, dst, z)
                _fragment_dot_rows(z, y, dot_parts, rows, BV)
                for i in T.Parallel(C):
                    if start + i < end:
                        BP[panel, start + i, head] = rows[i]
                if T.get_thread_binding() == 0:
                    total = T.alloc_local((1,), "float32")
                    total[0] = 0
                    for v in T.serial(BV):
                        total[0] += srow[v]
                    SR[panel, slot, head] = Exps[slot, head, 0, C - 1] * total[0]
                for i, v in T.Parallel(C, BV):
                    out[i, v] = T.if_then_else(start + i < end, z[i, v], 0)
                T.sync_threads()
                T.copy(out[:, :BV], ZC[slot, head, :, col : col + BV])
                T.sync_threads()
                for i, v in T.Parallel(C, BV):
                    out[i, v] = beta[i] * z[i, v]
                T.sync_threads()
                _store_tokens(out, DY, start, end, head, col, BV)

    return kernel


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=GRAD_COMPILE_FLAGS)
def chunk_grad(H, DK, DV, dtype, cu_dtype, scale):
    """Per chunk: accumulate every value-contracted product over panels, then do pair work once.

    Row scalings (alpha, beta, tail) commute with the value contraction and are applied after the
    panel loop, so every panel operand is a plain tile copy.
    """
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")
    NV = DV // BV

    @T.prim_func
    def kernel(
        Q: T.Tensor((tokens, H, DK), dtype),
        K: T.Tensor((tokens, H, DK), dtype),
        DO: T.Tensor((tokens, H, DV), dtype),
        B: T.Tensor((tokens, H), "float32"),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        CP: T.Tensor((capacity, H, DV, DK), dtype),
        U: T.Tensor((capacity, H, C, DV), dtype),
        Exps: T.Tensor((capacity, H, 2, C), "float32"),
        L: T.Tensor((capacity, H, C), "float32"),
        ZC: T.Tensor((capacity, H, C, DV), dtype),
        DS: T.Tensor((capacity, H, DV, DK), dtype),
        SR: T.Tensor((NV, capacity, H), "float32"),
        BP: T.Tensor((NV, tokens, H), "float32"),
        DQ: T.Tensor((tokens, H, DK), dtype),
        DKOut: T.Tensor((tokens, H, DK), dtype),
        DG: T.Tensor((tokens, H), "float32"),
        DB: T.Tensor((tokens, H), "float32"),
    ):
        with T.Kernel(capacity, H, threads=PROJECTION_THREADS) as (slot, head):
            seq = T.alloc_var("int32")
            chunk = T.alloc_var("int32")
            offset = T.alloc_var("int32")
            seq, chunk, offset = -1, 0, 0
            for n in T.serial(seqs):
                count = T.ceildiv(CU[n + 1] - CU[n], C)
                if slot >= offset and slot < offset + count:
                    seq, chunk = n, slot - offset
                offset += count
            if seq >= 0:
                start, end = CU[seq] + chunk * C, CU[seq + 1]
                dos, zz, uu = T.alloc_shared((C, BV), dtype), T.alloc_shared((C, BV), dtype), T.alloc_shared((C, BV), dtype)
                sp, dp = T.alloc_shared((BV, DK), dtype), T.alloc_shared((BV, DK), dtype)
                qs, ks = T.alloc_shared((C, DK), dtype), T.alloc_shared((C, DK), dtype)
                qm, km = T.alloc_shared((C, DK), dtype), T.alloc_shared((C, DK), dtype)
                wm, pm, mt = T.alloc_shared((C, C), dtype), T.alloc_shared((C, C), dtype), T.alloc_shared((C, C), dtype)
                reduction = T.alloc_shared((C, C + 8), dtype)
                out = T.alloc_shared((C, DK + 8), dtype)
                ls, alpha, tail = T.alloc_shared((C,), "float32"), T.alloc_shared((C,), "float32"), T.alloc_shared((C,), "float32")
                beta, dl, db = T.alloc_shared((C,), "float32"), T.alloc_shared((C,), "float32"), T.alloc_shared((C,), "float32")
                rows, columns, rows_b = T.alloc_shared((C,), "float32"), T.alloc_shared((C,), "float32"), T.alloc_shared((C,), "float32")
                dot_parts = T.alloc_shared((2, C), "float32")
                dq, dka, dkb = T.alloc_fragment((C, DK), "float32"), T.alloc_fragment((C, DK), "float32"), T.alloc_fragment((C, DK), "float32")
                c1, c2, kk = T.alloc_fragment((C, C), "float32"), T.alloc_fragment((C, C), "float32"), T.alloc_fragment((C, C), "float32")
                for i in T.Parallel(C):
                    ls[i] = L[slot, head, i]
                    alpha[i] = Exps[slot, head, 0, i]
                    tail[i] = Exps[slot, head, 1, i]
                    beta[i] = T.if_then_else(start + i < end, B[T.min(start + i, end - 1), head], 0)
                    total = T.alloc_local((1,), "float32")
                    total[0] = 0
                    for p in T.serial(NV):
                        total[0] += T.if_then_else(start + i < end, BP[p, T.min(start + i, end - 1), head], 0)
                    db[i] = total[0]
                T.clear(dq)
                T.clear(dka)
                T.clear(dkb)
                T.clear(c1)
                T.clear(c2)
                for panel in T.serial(NV):
                    col = panel * BV
                    _load_tokens(DO, dos, start, end, head, col, BV)
                    T.copy(ZC[slot, head, :, col : col + BV], zz)
                    T.copy(U[slot, head, :, col : col + BV], uu)
                    T.copy(CP[slot, head, col : col + BV, :], sp)
                    T.copy(DS[slot, head, col : col + BV, :], dp)
                    T.sync_threads()
                    T.gemm(dos, sp, dq)
                    T.gemm(zz, sp, dka)
                    T.gemm(uu, dp, dkb)
                    T.gemm(dos, uu, c1, transpose_B=True)
                    T.gemm(zz, uu, c2, transpose_B=True)
                    T.sync_threads()
                _load_tokens(Q, qs, start, end, head, 0, DK)
                _load_tokens(K, ks, start, end, head, 0, DK)
                _load_tokens(Q, qm, start, end, head, 0, DK)
                _load_tokens(K, km, start, end, head, 0, DK)
                for i, d in T.Parallel(C, DK):
                    dq[i, d] *= scale * alpha[i]
                    dka[i, d] *= -alpha[i] * beta[i]
                    dkb[i, d] *= tail[i]
                T.sync_threads()
                _fragment_dot_rows(dq, qs, dot_parts, rows, DK)
                _fragment_dot_rows(dka, ks, dot_parts, columns, DK)
                _fragment_dot_rows(dkb, ks, dot_parts, rows_b, DK)
                for i in T.Parallel(C):
                    dl[i] = rows[i] + columns[i] - rows_b[i]
                for i, d in T.Parallel(C, DK):
                    dka[i, d] += dkb[i, d]
                T.sync_threads()
                if T.get_thread_binding() == 0:
                    total = T.alloc_local((1,), "float32")
                    total[0] = 0
                    for i in T.serial(C):
                        total[0] += rows_b[i]
                    for p in T.serial(NV):
                        total[0] += SR[p, slot, head]
                    dl[C - 1] += total[0]
                T.gemm(ks, ks, kk, transpose_B=True, clear_accum=True)
                T.sync_threads()
                for i, j in T.Parallel(C, C):
                    pair = T.if_then_else(i > j, _exp(ls[i] - ls[j]), 0)
                    reduction[i, j] = -c2[i, j] * pair * kk[i, j]
                    wm[i, j] = -beta[i] * c2[i, j] * pair
                T.sync_threads()
                for i, lane in T.Parallel(C, 8):
                    row, column = T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32")
                    row[0], column[0] = 0, 0
                    for j in T.serial(C // 8):
                        index = j * 8 + lane
                        row[0] += T.cast(reduction[i, index], "float32")
                        column[0] += beta[index] * T.cast(reduction[index, i], "float32")
                    _sum_eight(row, lane)
                    _sum_eight(column, lane)
                    if lane == 0:
                        db[i] += row[0]
                        dl[i] += beta[i] * row[0] - column[0]
                T.sync_threads()
                T.gemm(qs, ks, kk, transpose_B=True, clear_accum=True)
                T.sync_threads()
                for i, j in T.Parallel(C, C):
                    rounded = T.alloc_local((1,), dtype)
                    rounded[0] = T.if_then_else(i >= j, scale * c1[i, j] * _exp(ls[i] - ls[j]), 0)
                    pm[i, j] = rounded[0]
                    reduction[i, j] = rounded[0] * kk[i, j]
                _pair_sums(reduction, rows, columns)
                for i in T.Parallel(C):
                    dl[i] += rows[i] - columns[i]
                T.gemm(pm, km, dq)
                for i, d in T.Parallel(C, DK):
                    out[i, d] = dq[i, d]
                for i, j in T.Parallel(C, C):
                    mt[i, j] = pm[j, i]
                T.sync_threads()
                _store_tokens(out, DQ, start, end, head, 0, DK)
                T.gemm(mt, qm, dka)
                T.gemm(wm, km, dka)
                T.sync_threads()
                for i, j in T.Parallel(C, C):
                    mt[i, j] = wm[j, i]
                T.sync_threads()
                T.gemm(mt, km, dka)
                for i, d in T.Parallel(C, DK):
                    out[i, d] = dka[i, d]
                T.sync_threads()
                _store_tokens(out, DKOut, start, end, head, 0, DK)
                for i in T.Parallel(C):
                    suffix = T.alloc_local((1,), "float32")
                    suffix[0] = 0
                    for j in T.serial(C):
                        if j >= i:
                            suffix[0] += dl[j]
                    if start + i < end:
                        DG[start + i, head] = suffix[0]
                        DB[start + i, head] = db[i]

    return kernel


@T.macro
def _suffix_step(lo, hi, total, lane, offset):
    a = T.alloc_local((1,), "float32")
    b = T.alloc_local((1,), "float32")
    a[0] = T.shfl_down(lo[0], offset)
    b[0] = T.shfl_down(hi[0], offset)
    if lane + offset < 32:
        lo[0] += a[0]
        hi[0] += b[0]
    total[0] += T.shfl_xor(total[0], offset)


@T.macro
def _lane_sum4(value, lane):
    other = T.alloc_local((1,), "float32")
    other[0] = T.shfl_down(value[0], 2)
    if lane < 2:
        value[0] += other[0]
    other[0] = T.shfl_down(value[0], 1)
    if lane == 0:
        value[0] += other[0]


def _fused_builder(H, DK, DV, dtype, cu_dtype, scale, from_ds, threads=PROJECTION_THREADS):
    """One CTA per (chunk, head) computes the whole chunk gradient with one global load phase.

    Cut path (from_ds=False), D_c = SourceMap_{c+1} dO_{c+1} with dO-independent maps from prepare
    (M_c = StateDZ_c SourceMap_{c+1}, RTt_c = (CP_c SourceMap_{c+1})^T):
        z = [SourceDZ | M] [dO_c; dO_{c+1}],   dK_B = (U dO_{c+1}^T) SourceMap_{c+1}^T,   sum(D o CP) = sum(dO_{c+1} o RTt)
    Scan path (from_ds=True), D_c read from the state scan (staged K-major and MN-major):
        z = SourceDZ dO_c + StateDZ D_c^T,      dK_B = U D_c,                              sum(D o CP) row-reduced in shared
    Both: masked pair coefficients (decay, scale, K K^T, Q K^T) come packed from prepare_next; c2^T = U ys^T carries
    beta_col; dQ += [P | 0] [K; Q], dK += [W + W^T | P^T] [K; Q]; the dQ row dot is taken after that GEMM (it then
    includes sum(P o QK)); row dots are accumulated in MMA row ownership and reduced once.
    """
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")
    # Scan path reads D_c K-major (DK, DV) as written by the k-major state scan; the cut path gets a placeholder.
    DS_SHAPE = (capacity, H, DK, DV) if from_ds else (capacity, H, DV, DK)
    assert threads == PAIR_THREADS, (
        f"the non-persistent fused kernel runs at {threads} threads but the PAIR / PAIRQ operands are packed for "
        f"{PAIR_THREADS}: use threads=PAIR_THREADS")
    # Accumulator ownership is part * 256 + 4 i + lane, i.e. 8 warps along M and threads // 256 along N.
    _gemm_policy_for(threads, ((C, DV), (C, DK), (C, C)))
    # MMA accumulators of a 64-row tile: row = (tid & 255) >> 2, column = (tid >> 8) * (N / P) + 4 j + (tid & 3).
    P = threads // 256
    owner = T.Fragment(
        (P, C, 4),
        forward_thread_fn=lambda part, i, lane: part * 256 + i % C * 4 + lane,
        forward_index_fn=lambda part, i, lane: i // C,
    )
    # Same ownership as a C x C GEMM accumulator, for dO-independent pair operands kept in registers.
    pairl = T.Fragment(
        (C, C),
        forward_thread_fn=lambda i, j: j // (C // P) * 256 + i * 4 + j % 4,
        forward_index_fn=lambda i, j: j % (C // P) // 4,
    )


    def psum(parts, index, i):
        total = parts[index * P, i]
        for p in range(1, P):
            total = total + parts[index * P + p, i]
        return total

    @T.prim_func
    def fused_chunk(
        Q: T.Tensor((tokens, H, DK), dtype),
        K: T.Tensor((tokens, H, DK), dtype),
        DO: T.Tensor((tokens, H, DV), dtype),
        B: T.Tensor((tokens, H), "float32"),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        CP: T.Tensor((capacity, H, DV, DK), dtype),
        U: T.Tensor((capacity, H, C, DV), dtype),
        Y: T.Tensor((capacity, H, C, DV), dtype),
        Exps: T.Tensor((capacity, H, 2, C), "float32"),
        L: T.Tensor((capacity, H, C), "float32"),
        SourceDZ: T.Tensor((capacity, H, C, C), dtype),
        StateDZ: T.Tensor((capacity, H, C, DK), dtype),
        M: T.Tensor((capacity, H, C, C), dtype),
        RTt: T.Tensor((capacity, H, C, DV), dtype),
        SourceMap: T.Tensor((capacity, H, DK, C), dtype),
        DS: T.Tensor(DS_SHAPE, dtype),
        PAIR: T.Tensor((capacity, H, PAIR_THREADS, PAIR_ARRAYS, C * C // PAIR_THREADS), dtype),
        PAIRQ: T.Tensor((capacity, H, PAIR_THREADS, C * C // PAIR_THREADS), "float32"),
        DQ: T.Tensor((tokens, H, DK), dtype),
        DKOut: T.Tensor((tokens, H, DK), dtype),
        DY: T.Tensor((tokens, H, DV), dtype),
        DG: T.Tensor((tokens, H), "float32"),
        DB: T.Tensor((tokens, H), "float32"),
    ):
        with T.Kernel(capacity, H, threads=threads) as (slot, head):
            seq = T.alloc_var("int32")
            chunk = T.alloc_var("int32")
            offset = T.alloc_var("int32")
            seq, chunk, offset = -1, 0, 0
            for n in T.serial(seqs):
                count = T.ceildiv(CU[n + 1] - CU[n], C)
                if slot >= offset and slot < offset + count:
                    seq, chunk = n, slot - offset
                offset += count
            if seq >= 0:
                start, end = CU[seq] + chunk * C, CU[seq + 1]
                if from_ds:
                    sdz, stdz = T.alloc_shared((C, C), dtype), T.alloc_shared((C, DK), dtype)
                    dmn = T.alloc_shared((C, DV), dtype)
                    dtk, dmd = T.alloc_shared((DK, DV), dtype), T.alloc_shared((DK, DV), dtype)
                    srow = T.alloc_shared((C,), "float32")
                else:
                    mz = T.alloc_shared((C, 2 * C), dtype)
                    dd = T.alloc_shared((2 * C, DV), dtype)
                    dnx, rtt = T.alloc_shared((C, DV), dtype), T.alloc_shared((C, DV), dtype)
                    sm, c3s = T.alloc_shared((DK, C), dtype), T.alloc_shared((C, C), dtype)
                    c3 = T.alloc_fragment((C, C), "float32")
                dos, uu, yy = T.alloc_shared((C, DV), dtype), T.alloc_shared((C, DV), dtype), T.alloc_shared((C, DV), dtype)
                sp, qk2 = T.alloc_shared((DV, DK), dtype), T.alloc_shared((2 * C, DK), dtype)
                zs, ys = T.alloc_shared((C, DV), dtype), T.alloc_shared((C, DV), dtype)
                ap, aw = T.alloc_shared((C, 2 * C), dtype), T.alloc_shared((C, 2 * C), dtype)
                alpha = T.alloc_shared((C,), "float32")
                pem, pet = T.alloc_fragment((C, C), dtype), T.alloc_fragment((C, C), dtype)
                pal, pau = T.alloc_fragment((C, C), dtype), T.alloc_fragment((C, C), dtype)
                palk, pauk = T.alloc_fragment((C, C), dtype), T.alloc_fragment((C, C), dtype)
                pqt, pdm = T.alloc_fragment((C, C), "float32"), T.alloc_fragment((C, C), dtype)
                T.annotate_layout({pem: pairl, pet: pairl, pal: pairl, pau: pairl, palk: pairl, pauk: pairl, pqt: pairl, pdm: pairl})
                # Issued first; consumed only in the pair phase, so the loads stay off the critical path. prepare_next
                # stores them packed in this ownership (thread, element), so each thread issues contiguous vector loads.
                for i, j in T.Parallel(C, C):
                    owner_thread, element = _pair_pack(i, j)
                    ev = T.cast(PAIR[slot, head, owner_thread, 0, element], "float32")
                    kv = T.cast(PAIR[slot, head, owner_thread, 1, element], "float32")
                    pem[i, j] = T.cast(T.if_then_else(i >= j, scale * ev, 0), dtype)
                    pet[i, j] = T.cast(T.if_then_else(j >= i, scale * ev, 0), dtype)
                    pal[i, j] = T.cast(T.if_then_else(i > j, -ev, 0), dtype)
                    pau[i, j] = T.cast(T.if_then_else(j > i, -ev, 0), dtype)
                    palk[i, j] = T.cast(T.if_then_else(i > j, -ev * kv, 0), dtype)
                    pauk[i, j] = T.cast(T.if_then_else(j > i, -ev * kv, 0), dtype)
                    pdm[i, j] = T.cast(T.if_then_else(i == j, 1, 0), dtype)
                    pqt[i, j] = PAIRQ[slot, head, owner_thread, element]
                tail, beta = T.alloc_shared((C,), "float32"), T.alloc_shared((C,), "float32")
                parts = T.alloc_shared((3 * P, C), "float32")
                z = T.alloc_fragment((C, DV), "float32")
                dq, dka, dkb = T.alloc_fragment((C, DK), "float32"), T.alloc_fragment((C, DK), "float32"), T.alloc_fragment((C, DK), "float32")
                c1, c2 = T.alloc_fragment((C, C), "float32"), T.alloc_fragment((C, C), "float32")
                c1t, c2t, diag = T.alloc_fragment((C, C), "float32"), T.alloc_fragment((C, C), "float32"), T.alloc_fragment((C, C), "float32")
                # Row sums are linear, so per-thread partials are combined into the three final row quantities
                # (dl, db, extra) before the lane reduction.
                a_dl, a_db, a_ex = T.alloc_fragment((P, C, 4), "float32"), T.alloc_fragment((P, C, 4), "float32"), T.alloc_fragment((P, C, 4), "float32")
                T.annotate_layout({a_dl: owner, a_db: owner, a_ex: owner})

                for i in T.Parallel(C):
                    alpha[i] = Exps[slot, head, 0, i]
                    tail[i] = Exps[slot, head, 1, i]
                    beta[i] = T.if_then_else(start + i < end, B[T.min(start + i, end - 1), head], 0)
                T.copy(U[slot, head, :, :], uu)
                T.copy(Y[slot, head, :, :], yy)
                T.copy(CP[slot, head, :, :], sp)
                if from_ds:
                    T.copy(SourceDZ[slot, head, :, :], sdz)
                    T.copy(StateDZ[slot, head, :, :], stdz)
                    T.copy(DS[slot, head, :, :], dtk)
                    T.copy(DS[slot, head, :, :], dmd)
                    if start + C <= end:
                        for i, v in T.Parallel(C, DV):
                            dos[i, v] = DO[start + i, head, v]
                            dmn[i, v] = DO[start + i, head, v]
                    else:
                        for i, v in T.Parallel(C, DV):
                            now = T.if_then_else(start + i < end, DO[T.min(start + i, end - 1), head, v], 0)
                            dos[i, v] = now
                            dmn[i, v] = now
                else:
                    T.copy(SourceDZ[slot, head, :, :], mz[:, :C])
                    T.copy(M[slot, head, :, :], mz[:, C:])
                    T.copy(RTt[slot, head, :, :], rtt)
                    if start + C <= end:
                        for i, v in T.Parallel(C, DV):
                            dos[i, v] = DO[start + i, head, v]
                            dd[i, v] = DO[start + i, head, v]
                    else:
                        for i, v in T.Parallel(C, DV):
                            now = T.if_then_else(start + i < end, DO[T.min(start + i, end - 1), head, v], 0)
                            dos[i, v] = now
                            dd[i, v] = now
                    if start + 2 * C <= end:
                        for i, v in T.Parallel(C, DV):
                            dnx[i, v] = DO[start + C + i, head, v]
                            dd[C + i, v] = DO[start + C + i, head, v]
                        for d, i in T.Parallel(DK, C):
                            sm[d, i] = SourceMap[slot + 1, head, d, i]
                    elif start + C < end:
                        for i, v in T.Parallel(C, DV):
                            nxt = T.if_then_else(start + C + i < end, DO[T.min(start + C + i, end - 1), head, v], 0)
                            dnx[i, v] = nxt
                            dd[C + i, v] = nxt
                        for d, i in T.Parallel(DK, C):
                            sm[d, i] = SourceMap[slot + 1, head, d, i]
                    else:
                        for i, v in T.Parallel(C, DV):
                            dnx[i, v] = 0
                            dd[C + i, v] = 0
                        for d, i in T.Parallel(DK, C):
                            sm[d, i] = 0
                # Plain parallel loads where the range is known valid: T.copy would add a per-vector
                # `< tokens` predicate for the dynamic token offset (one basic block per load).
                if start + C <= end:
                    for i, d in T.Parallel(C, DK):
                        qk2[i, d] = K[start + i, head, d]
                        qk2[C + i, d] = Q[start + i, head, d]
                else:
                    for i, d in T.Parallel(C, DK):
                        qk2[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                        qk2[C + i, d] = T.if_then_else(start + i < end, Q[T.min(start + i, end - 1), head, d], 0)
                T.sync_threads()
                if from_ds:
                    T.gemm(sdz, dmn, z, clear_accum=True)
                    T.gemm(stdz, dtk, z)
                else:
                    T.gemm(mz, dd, z, clear_accum=True)
                T.gemm(dos, sp, dq, clear_accum=True)
                T.gemm(dos, uu, c1, transpose_B=True, clear_accum=True)
                if from_ds:
                    T.gemm(uu, dmd, dkb, transpose_B=True, clear_accum=True)
                else:
                    T.gemm(uu, dnx, c3, transpose_B=True, clear_accum=True)
                T.gemm(uu, dos, c1t, transpose_B=True, clear_accum=True)
                if from_ds:
                    # sum(D o CP) with D stored K-major (DK, DV), folded onto C rows (DV is a multiple of C).
                    for i, lane in T.Parallel(C, 8):
                        acc = T.alloc_local((1,), "float32")
                        acc[0] = 0
                        for rr in T.serial(DV // C):
                            for j in T.serial(DK // 8):
                                column = lane * (DK // 8) + j
                                acc[0] += T.cast(dtk[column, rr * C + i], "float32") * T.cast(sp[rr * C + i, column], "float32")
                        _sum_eight(acc, lane)
                        if lane == 0:
                            srow[i] = acc[0]
                else:
                    T.gemm(dnx, rtt, diag, transpose_B=True, clear_accum=True)
                T.sync_threads()
                for part, i, lane in T.Parallel(P, C, 4):
                    sr = T.alloc_local((1,), "float32")
                    sr[0] = 0
                    if not from_ds:
                        # sum(dO_{c+1} o RTt) per row = diag(dO_{c+1} RTt^T); each diagonal entry has one owner.
                        for j in T.serial(C // (4 * P)):
                            sr[0] += diag[i, part * (C // P) + j * 4 + lane] * T.cast(pdm[i, part * (C // P) + j * 4 + lane], "float32")
                    a_ex[part, i, lane] = alpha[C - 1] * sr[0]
                for i, v in T.Parallel(C, DV):
                    zs[i, v] = T.if_then_else(start + i < end, z[i, v], 0)
                    ys[i, v] = beta[i] * z[i, v]
                if not from_ds:
                    for i, j in T.Parallel(C, C):
                        c3s[i, j] = c3[i, j]
                T.sync_threads()
                T.gemm(zs, sp, dka, clear_accum=True)
                T.gemm(zs, uu, c2, transpose_B=True, clear_accum=True)
                T.gemm(uu, ys, c2t, transpose_B=True, clear_accum=True)
                T.gemm(zs, yy, diag, transpose_B=True, clear_accum=True)
                if not from_ds:
                    T.gemm(c3s, sm, dkb, transpose_B=True, clear_accum=True)
                # Fragment rows hold every 4th column, so direct global stores are 2-byte scatters; stage via shared.
                if start + C <= end:
                    for i, v in T.Parallel(C, DV):
                        DY[start + i, head, v] = ys[i, v]
                else:
                    for i, v in T.Parallel(C, DV):
                        if start + i < end:
                            DY[start + i, head, v] = ys[i, v]
                T.sync_threads()
                for i, d in T.Parallel(C, DK):
                    dq[i, d] *= scale * alpha[i]
                    dka[i, d] *= -alpha[i] * beta[i]
                    dkb[i, d] *= tail[i]
                for part, i, lane in T.Parallel(P, C, 4):
                    pa, pb = T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32")
                    pa[0], pb[0] = 0, 0
                    for j in T.serial(DK // (4 * P)):
                        column = part * (DK // P) + j * 4 + lane
                        kv = T.cast(qk2[i, column], "float32")
                        pa[0] += dka[i, column] * kv
                        pb[0] += dkb[i, column] * kv
                    a_dl[part, i, lane] = pa[0] - pb[0]
                    a_ex[part, i, lane] += pb[0]
                for i, d in T.Parallel(C, DK):
                    dka[i, d] += dkb[i, d]
                # Pair terms in MMA row ownership. c1t = c1^T and c2t = c2^T come from extra GEMMs, so W + W^T and
                # P^T are written without transposed stores and every row/column sum is a register row reduction.
                for part, i, lane in T.Parallel(P, C, 4):
                    zy, s1, t1, t2 = T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32")
                    zy[0], s1[0], t1[0], t2[0] = 0, 0, 0, 0
                    # Masks, scale and decay are folded into the prepared coefficients; c2t already carries beta_col (U ys^T) and
                    # sum(Pm o QK) is taken from the final dQ row dot after GEMM5.
                    for j in T.serial(C // (4 * P)):
                        col = part * (C // P) + j * 4 + lane
                        pm, pt = T.alloc_local((1,), dtype), T.alloc_local((1,), dtype)
                        pm[0] = T.cast(c1[i, col] * T.cast(pem[i, col], "float32"), dtype)
                        pt[0] = T.cast(c1t[i, col] * T.cast(pet[i, col], "float32"), dtype)
                        zy[0] += diag[i, col] * T.cast(pdm[i, col], "float32")
                        s1[0] += c2[i, col] * T.cast(palk[i, col], "float32")
                        t1[0] += c2t[i, col] * T.cast(pauk[i, col], "float32")
                        t2[0] += T.cast(pt[0], "float32") * pqt[i, col]
                        ap[i, col] = pm[0]
                        ap[i, C + col] = 0
                        aw[i, col] = beta[i] * c2[i, col] * T.cast(pal[i, col], "float32") + c2t[i, col] * T.cast(pau[i, col], "float32")
                        aw[i, C + col] = pt[0]
                    a_dl[part, i, lane] += beta[i] * s1[0] - t1[0] - t2[0]
                    a_db[part, i, lane] = zy[0] + s1[0]
                T.sync_threads()
                T.gemm(ap, qk2, dq)
                T.gemm(aw, qk2, dka)
                # Row dot of the final dQ with Q: the dO_c CP term plus the pair term sum(Pm o QK).
                for part, i, lane in T.Parallel(P, C, 4):
                    pq = T.alloc_local((1,), "float32")
                    pq[0] = 0
                    for j in T.serial(DK // (4 * P)):
                        column = part * (DK // P) + j * 4 + lane
                        pq[0] += dq[i, column] * T.cast(qk2[C + i, column], "float32")
                    a_dl[part, i, lane] += pq[0]
                for part, i, lane in T.Parallel(P, C, 4):
                    v0, v1, v2 = T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32")
                    v0[0], v1[0], v2[0] = a_dl[part, i, lane], a_db[part, i, lane], a_ex[part, i, lane]
                    _lane_sum4(v0, lane)
                    _lane_sum4(v1, lane)
                    _lane_sum4(v2, lane)
                    if lane == 0:
                        parts[0 * P + part, i] = v0[0]
                        parts[1 * P + part, i] = v1[0]
                        parts[2 * P + part, i] = v2[0]
                for i, d in T.Parallel(C, DK):
                    if start + i < end:
                        DQ[start + i, head, d] = dq[i, d]
                        DKOut[start + i, head, d] = dka[i, d]
                T.sync_threads()
                # Suffix sums by one warp (lanes own rows lane and lane + 32): shuffle scans instead of a 64-step loop.
                tx = T.get_thread_binding()
                if tx < 32:
                    lo, hi, ex = T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32")
                    lo[0], hi[0] = psum(parts, 0, tx), psum(parts, 0, tx + 32)
                    ex[0] = psum(parts, 2, tx) + psum(parts, 2, tx + 32)
                    if from_ds:
                        ex[0] += alpha[C - 1] * (srow[tx] + srow[tx + 32])
                    _suffix_step(lo, hi, ex, tx, 1)
                    _suffix_step(lo, hi, ex, tx, 2)
                    _suffix_step(lo, hi, ex, tx, 4)
                    _suffix_step(lo, hi, ex, tx, 8)
                    _suffix_step(lo, hi, ex, tx, 16)
                    lo[0] += T.shfl_sync(hi[0], 0)
                    if start + tx < end:
                        DG[start + tx, head] = lo[0] + ex[0]
                        DB[start + tx, head] = psum(parts, 1, tx)
                    if start + 32 + tx < end:
                        DG[start + 32 + tx, head] = hi[0] + ex[0]
                        DB[start + 32 + tx, head] = psum(parts, 1, tx + 32)

    return fused_chunk


@tilelang.jit(target="musa", pass_configs=FUSED_CONFIG, compile_flags=FUSED_FLAGS)
def chunk_fused(H, DK, DV, dtype, cu_dtype, scale, threads=PROJECTION_THREADS):
    return _fused_builder(H, DK, DV, dtype, cu_dtype, scale, False, threads)


@tilelang.jit(target="musa", pass_configs=FUSED_CONFIG, compile_flags=FUSED_FLAGS)
def chunk_fused_ds(H, DK, DV, dtype, cu_dtype, scale, threads=PROJECTION_THREADS):
    return _fused_builder(H, DK, DV, dtype, cu_dtype, scale, True, threads)

@T.macro
def _issue_g1(MZ, DO, RTt, mz, dd, dos, dnx, rtt, slot, head, start, bar, GATES, gat, buf):
    T.tma_copy(GATES[slot, head, :, :], gat[buf, :, :], barrier=bar)
    T.tma_copy(MZ[slot, head, :, :], mz, barrier=bar)
    T.tma_copy(DO[start : start + 2 * C, head, :], dd, barrier=bar)
    T.tma_copy(DO[start : start + C, head, :], dos, barrier=bar)
    T.tma_copy(DO[start + C : start + 2 * C, head, :], dnx, barrier=bar)
    T.tma_copy(RTt[slot, head, :, :], rtt, barrier=bar)
    T.barrier_arrive(bar)


@T.macro
def _issue_g2(U, Y, CP, SourceMap, uu, yy, sp, sm, slot, head, bar):
    T.tma_copy(U[slot, head, :, :], uu, barrier=bar)
    T.tma_copy(Y[slot, head, :, :], yy, barrier=bar)
    T.tma_copy(CP[slot, head, :, :], sp, barrier=bar)
    T.tma_copy(SourceMap[slot + 1, head, :, :], sm, barrier=bar)
    T.barrier_arrive(bar)


@T.macro
def _issue_g1_ds(SourceDZ, StateDZ, DS, DO, sdz, stdz, dtk, dmd, dmn, dos, slot, head, start, bar, GATES, gat, buf):
    T.tma_copy(GATES[slot, head, :, :], gat[buf, :, :], barrier=bar)
    T.tma_copy(SourceDZ[slot, head, :, :], sdz, barrier=bar)
    T.tma_copy(StateDZ[slot, head, :, :], stdz, barrier=bar)
    T.tma_copy(DS[slot, head, :, :], dtk, barrier=bar)
    T.tma_copy(DS[slot, head, :, :], dmd, barrier=bar)
    T.tma_copy(DO[start : start + C, head, :], dmn, barrier=bar)
    T.tma_copy(DO[start : start + C, head, :], dos, barrier=bar)
    T.barrier_arrive(bar)


@T.macro
def _issue_g2_ds(U, Y, CP, uu, yy, sp, slot, head, bar):
    T.tma_copy(U[slot, head, :, :], uu, barrier=bar)
    T.tma_copy(Y[slot, head, :, :], yy, barrier=bar)
    T.tma_copy(CP[slot, head, :, :], sp, barrier=bar)
    T.barrier_arrive(bar)


@T.macro
def _issue_g3(KQ, qk2, slot, head, bar):
    T.tma_copy(KQ[slot, head, :, :], qk2, barrier=bar)
    T.barrier_arrive(bar)


@T.macro
def _trace_rows(dmd, sp, srow, DK, DV, dtype):
    # srow[row] = sum_v D[row, v] CP^T[row, v]; D and CP^T share one layout: each thread dots 32 contiguous elements.
    for row, q in T.Parallel(DK, 4):
        acc = T.alloc_local((1,), "float32")
        acc[0] = 0
        for g in T.serial(DV // 32):
            da, sa = T.alloc_local((8,), dtype), T.alloc_local((8,), dtype)
            for e in T.vectorized(8):
                da[e] = dmd[row, q * (DV // 4) + g * 8 + e]
                sa[e] = sp[row, q * (DV // 4) + g * 8 + e]
            for e in T.serial(8):
                acc[0] += T.cast(da[e], "float32") * T.cast(sa[e], "float32")
        _lane_sum4(acc, q)
        if q == 0:
            srow[row] = acc[0]


# Independent partial sums per row dot / pair accumulation in the persistent kernels (1 = single dependent chain).
ILP = 4


def _psum_local(acc):
    total = acc[0]
    for k in range(1, ILP):
        total = total + acc[k]
    return total


AW = -1
# Scan-path trace sum(D o CP^T) is produced by the separate state_trace kernel.


@T.macro
def _gemm_diagonal(left, right, diag, tx, policy, diagonal_squads_only):
    if diagonal_squads_only:
        # The 512-thread layout assigns the diagonal tiles to squads 0 and 3.
        T.clear(diag)
        if tx // 128 == 0 or tx // 128 == 3:
            T.gemm(left, right, diag, transpose_B=True, policy=policy, clear_accum=True, wg_wait=AW)
    else:
        T.gemm(left, right, diag, transpose_B=True, policy=policy, clear_accum=True, wg_wait=AW)


# Preserve the 128-wide shared strides and the original SQMMA operand layouts.
# The second half of zs is zero and does not contribute to dQ.
_HALF_K_DQ = r'''
#include <tl_templates/musa/gemm.h>
#if !defined(__MUSA_ARCH_LIST__) || (__MUSA_ARCH_LIST__ != 320)
#error GDN half-K projection requires MP32
#endif
namespace tl {
template <typename A, typename B, typename C>
TL_DEVICE void gdn_half_dq(A *pa, B *pb, C *accum) {
    using MMA = mp32_sqmma::GemmSS<64, 128, 128, 8, 2, false, false, false,
        128, 128, 0, 0, 32, 64, 64, 16, 256, 256, 32, 256, 256, 0, A, B, C>;
    const int squad = threadIdx.x / 128;
    const int m = (squad % 2) * 32;
    const int n = (squad / 2) * 64;
    auto params = mp32_sqmma::makeSqmmaParams<16, 256, 256, MMA::kAChunkCols,
        32, 256, 256, MMA::kBChunkCols, 0, A, B, C>(pa, pb,
        MMA::aByteOffset(m, 0), MMA::bByteOffset(0, n),
        MMA::kALayout, MMA::kBLayout, false);
    auto *pc = reinterpret_cast<int32_t *>(accum);
    mp32_sqmma::sqmmaMma<32, 64, 64, MMA::kSqmmaType>(
        pc, &params.a_desc, &params.b_desc, pc, params);
}
} // namespace tl
'''


def _fused_persistent_builder(H, DK, DV, dtype, cu_dtype, scale, ctas, from_ds=False, single_sequence=False):
    """Persistent fused kernel: `ctas` CTAs (one per MP) loop over (chunk, head) items.

    Same math as `_fused_builder`. With one CTA per MP the load phase was fully exposed, so each tile group of the next
    item is prefetched with TME into its buffers as soon as their last use in the current item has passed: G1 after GEMM1
    (cut: [SourceDZ|M], [dO_c; dO_{c+1}], dO_c, dO_{c+1}, RTt; scan: SourceDZ, StateDZ, D_c twice, dO_c twice),
    G2 = {U, Y, CP (, SourceMap_{c+1})} after GEMM3, G3 = {[K; Q]} after GEMM5. Cut items whose two chunks are not both
    full use per-thread loads; scan items always use TME and clear rows past the sequence end. Shared buffers are all
    live across the item loop (no merging); the pair operands reuse zs / ys, and dQ / dK are staged through them.
    """
    assert DV == 2 * C, "pair operands reuse the (C, DV) zs / ys buffers"
    tokens, seqs, capacity = T.dynamic("tokens"), 1 if single_sequence else T.dynamic("seqs"), T.dynamic("capacity")
    DS_SHAPE = (capacity, H, DK, DV) if from_ds else (capacity, H, DV, DK)
    threads = PAIR_THREADS
    # This specialization uses the MP32 512-thread ownership and 128-wide operands.
    symmetric_cut = _is_symmetric_cut(not from_ds, DK, DV)
    pairq_shape = _pairq_shape(capacity, H, symmetric_cut)
    pdt = dtype
    G2_ISSUER, G3_ISSUER = (32, 64)
    SUF_BASE = threads - 32
    P = threads // 256
    nvec = DV // (8 * 4 * P)
    span = 8 * nvec
    assert nvec * 8 * 4 * P == DV, f"the row-dot column cover must tile [0, {DV}) (got {nvec * 8 * 4 * P})"
    # The owner / pairl layouts and the pair loops need T.gemm's Square partition (8, threads // 256); the assert
    # verifies it for every GEMM shape used below and the policy is passed explicitly to pin it.  (At 1024 threads
    # the C x C GEMMs, N = C = 64, would get 16 columns per warp, below MP32 SQMMA's 32-column minimum - see the
    # notes on the C x C GEMMs in the body and the report.)
    GEMM_P = _gemm_policy_for(threads, ((C, DV), (C, DK), (C, C)))
    owner = T.Fragment(
        (P, C, 4),
        forward_thread_fn=lambda part, i, lane: part * 256 + i % C * 4 + lane,
        forward_index_fn=lambda part, i, lane: i // C,
    )
    pairl = T.Fragment(
        (C, C),
        forward_thread_fn=lambda i, j: j // (C // P) * 256 + i * 4 + j % 4,
        forward_index_fn=lambda i, j: j % (C // P) // 4,
    )
    G1_ISSUER = 0


    def psum(parts, index, i):
        total = parts[index * P, i]
        for p in range(1, P):
            total = total + parts[index * P + p, i]
        return total

    @T.prim_func
    def fused_persist(
        Q: T.Tensor((tokens, H, DK), dtype),
        K: T.Tensor((tokens, H, DK), dtype),
        DO: T.Tensor((tokens, H, DV), dtype),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        CP: T.Tensor((capacity, H, DV, DK), dtype),
        U: T.Tensor((capacity, H, C, DV), dtype),
        Y: T.Tensor((capacity, H, C, DV), dtype),
        GATES: T.Tensor((capacity, H, 3, C), "float32"),
        MZ: T.Tensor((capacity, H, C, 2 * C), dtype),
        RTt: T.Tensor((capacity, H, C, DV), dtype),
        SourceMap: T.Tensor((capacity, H, DK, C), dtype),
        SourceDZ: T.Tensor((capacity, H, C, C), dtype),
        StateDZ: T.Tensor((capacity, H, C, DK), dtype),
        DS: T.Tensor(DS_SHAPE, dtype),
        KQ: T.Tensor((capacity, H, 2 * C, DK), dtype),
        PAIR: T.Tensor((capacity, H, PAIR_THREADS, PAIR_ARRAYS, C * C // PAIR_THREADS), pdt),
        PAIRQ: T.Tensor(pairq_shape, "float32"),
        TRP: T.Tensor((capacity, H, 32), "float32"),
        DQ: T.Tensor((tokens, H, DK), dtype),
        DKOut: T.Tensor((tokens, H, DK), dtype),
        DY: T.Tensor((tokens, H, DV), dtype),
        DG: T.Tensor((tokens, H), "float32"),
        DB: T.Tensor((tokens, H), "float32"),
    ):
        with T.Kernel(ctas, threads=threads) as pid:
            if symmetric_cut:
                T.import_source(_HALF_K_DQ)
            if from_ds:
                sdz, stdz = T.alloc_shared((C, C), dtype), T.alloc_shared((C, DK), dtype)
                dmn = T.alloc_shared((C, DV), dtype)
                dtk, dmd = T.alloc_shared((DK, DV), dtype), T.alloc_shared((DK, DV), dtype)
                srow = T.alloc_shared((DK,), "float32")
            else:
                mz, dd = T.alloc_shared((C, 2 * C), dtype), T.alloc_shared((2 * C, DV), dtype)
                dnx, rtt = T.alloc_shared((C, DV), dtype), T.alloc_shared((C, DV), dtype)
                sm, c3s = T.alloc_shared((DK, C), dtype), T.alloc_shared((C, C), dtype)
            dos, uu, yy = T.alloc_shared((C, DV), dtype), T.alloc_shared((C, DV), dtype), T.alloc_shared((C, DV), dtype)
            sp, qk2 = T.alloc_shared((DV, DK), dtype), T.alloc_shared((2 * C, DK), dtype)
            # zs / ys are the pair operands [P | 0] / [W + W^T | P^T] once GEMM3 and the dY store are done.
            zs, ys = T.alloc_shared((C, DV), dtype), T.alloc_shared((C, DV), dtype)
            # Double-buffered per-item gates [alpha; tail; beta]: the next item's copy arrives with G1.
            gat = T.alloc_shared((2, 3, C), "float32")
            # Double-buffered by item parity: the suffix warp of item k reads its half while item k + 1 starts.
            parts = T.alloc_shared((2 * 3 * P, C), "float32")
            g1, g2, g3 = T.alloc_barrier(arrive_count=1), T.alloc_barrier(arrive_count=1), T.alloc_barrier(arrive_count=1)
            NE = C * C // PAIR_THREADS
            pco = T.alloc_local((PAIR_ARRAYS, NE), pdt)
            if not symmetric_cut:
                pqt = T.alloc_local((NE,), "float32")
            # Per-thread row gates [alpha_i, tail_i, beta_i, alpha_{C-1}] and the K / Q values of the row dots.
            grow = T.alloc_local((4,), "float32")
            kqv = T.alloc_local((DK // (4 * P),), dtype)
            if symmetric_cut:
                qv = T.alloc_local((DK // (4 * P),), dtype)
            trv = T.alloc_local((1,), "float32")
            # P / P^T of this thread's pair elements and sum(P^T o QK^T), formed while GEMM3 runs.
            pmr, ptr = T.alloc_local((C * C // PAIR_THREADS,), dtype), T.alloc_local((C * C // PAIR_THREADS,), dtype)
            if not symmetric_cut:
                t2r = T.alloc_local((ILP,), "float32")
            z = T.alloc_fragment((C, DV), "float32")
            dq, dka, dkb = T.alloc_fragment((C, DK), "float32"), T.alloc_fragment((C, DK), "float32"), T.alloc_fragment((C, DK), "float32")
            c1, c2, c3 = T.alloc_fragment((C, C), "float32"), T.alloc_fragment((C, C), "float32"), T.alloc_fragment((C, C), "float32")
            c1t, c2t, diag = (
                T.alloc_fragment((C, C), "float32"), T.alloc_fragment((C, C), "float32"),
                T.alloc_fragment((C, C), "float32"))
            a_dl, a_db, a_ex = T.alloc_fragment((P, C, 4), "float32"), T.alloc_fragment((P, C, 4), "float32"), T.alloc_fragment((P, C, 4), "float32")
            T.annotate_layout({a_dl: owner, a_db: owner, a_ex: owner})
            seq = T.alloc_var("int32")
            chunk = T.alloc_var("int32")
            nseq = T.alloc_var("int32")
            nchunk = T.alloc_var("int32")
            offset = T.alloc_var("int32")
            nfull = T.alloc_var("int32")
            ready = T.alloc_var("int32")
            phase = T.alloc_var("int32")
            ready, phase = 0, 0
            tx = T.get_thread_binding()
            if single_sequence:
                nstart = T.alloc_var(CU.dtype)
            items = capacity * H
            for it in T.serial(T.ceildiv(items - pid, ctas)):
                item = pid + it * ctas
                cb = it % 2
                slot, head = item // H, item % H
                nitem = item + ctas
                nslot, nhead = nitem // H, nitem % H
                if single_sequence:
                    seq, chunk = -1, 0
                    if slot < T.ceildiv(CU[1] - CU[0], C):
                        seq, chunk = 0, slot
                    nfull = 0
                    nstart = CU[0] + nslot * C
                    if nitem < items and nslot < T.ceildiv(CU[1] - CU[0], C):
                        if from_ds or nstart + 2 * C <= CU[1]:
                            nfull = 1
                else:
                    seq, chunk, offset = -1, 0, 0
                    for n in T.serial(seqs):
                        count = T.ceildiv(CU[n + 1] - CU[n], C)
                        if slot >= offset and slot < offset + count:
                            seq, chunk = n, slot - offset
                        offset += count
                    nseq, nchunk, offset, nfull = -1, 0, 0, 0
                    if nitem < items:
                        for n in T.serial(seqs):
                            count = T.ceildiv(CU[n + 1] - CU[n], C)
                            if nslot >= offset and nslot < offset + count:
                                nseq, nchunk = n, nslot - offset
                            offset += count
                        if nseq >= 0:
                            if from_ds or CU[nseq] + nchunk * C + 2 * C <= CU[nseq + 1]:
                                nfull = 1
                    nstart = CU[T.max(nseq, 0)] + nchunk * C
                if seq >= 0:
                    start, end = CU[seq] + chunk * C, CU[seq + 1]

                    if from_ds:
                        if ready == 0:
                            if tx == 0:
                                _issue_g1_ds(SourceDZ, StateDZ, DS, DO, sdz, stdz, dtk, dmd, dmn, dos, slot, head, start, g1, GATES, gat, cb)
                                _issue_g2_ds(U, Y, CP, uu, yy, sp, slot, head, g2)
                                _issue_g3(KQ, qk2, slot, head, g3)
                            ready = 1
                    else:
                        if ready == 0:
                            if start + 2 * C <= end:
                                if tx == 0:
                                    _issue_g1(MZ, DO, RTt, mz, dd, dos, dnx, rtt, slot, head, start, g1, GATES, gat, cb)
                                    _issue_g2(U, Y, CP, SourceMap, uu, yy, sp, sm, slot, head, g2)
                                    _issue_g3(KQ, qk2, slot, head, g3)
                                ready = 1
                            else:
                                for k, i in T.Parallel(3, C):
                                    gat[cb, k, i] = GATES[slot, head, k, i]
                                T.copy(MZ[slot, head, :, :], mz)
                                T.copy(KQ[slot, head, :, :], qk2)
                                T.copy(U[slot, head, :, :], uu)
                                T.copy(Y[slot, head, :, :], yy)
                                T.copy(CP[slot, head, :, :], sp)
                                T.copy(RTt[slot, head, :, :], rtt)
                                if start + C <= end:
                                    for i, v in T.Parallel(C, DV):
                                        dos[i, v] = DO[start + i, head, v]
                                        dd[i, v] = DO[start + i, head, v]
                                else:
                                    for i, v in T.Parallel(C, DV):
                                        now = T.if_then_else(start + i < end, DO[T.min(start + i, end - 1), head, v], 0)
                                        dos[i, v] = now
                                        dd[i, v] = now
                                if start + C < end:
                                    for i, v in T.Parallel(C, DV):
                                        nxt = T.if_then_else(start + C + i < end, DO[T.min(start + C + i, end - 1), head, v], 0)
                                        dnx[i, v] = nxt
                                        dd[C + i, v] = nxt
                                    for d, i in T.Parallel(DK, C):
                                        sm[d, i] = SourceMap[slot + 1, head, d, i]
                                else:
                                    for i, v in T.Parallel(C, DV):
                                        dnx[i, v] = 0
                                        dd[C + i, v] = 0
                                    for d, i in T.Parallel(DK, C):
                                        sm[d, i] = 0
                    # Prefetched items use the transaction barriers. Scalar
                    # staging still needs the CTA barrier before its readers.
                    if not symmetric_cut or it == 0 or ready == 0:
                        T.sync_threads()
                    if ready == 1:
                        T.barrier_wait(g1, phase % 2)
                        T.barrier_wait(g2, phase % 2)
                    if from_ds:
                        # TME boxes are whole chunks: clear dO rows past the sequence end (another sequence, or zero
                        # fill past `tokens`); the per-slot tiles are already zero-padded.
                        if start + C > end:
                            for i, v in T.Parallel(C, DV):
                                if start + i >= end:
                                    dos[i, v] = 0
                                    dmn[i, v] = 0
                            T.sync_threads()
                    if from_ds:
                        T.gemm(sdz, dmn, z, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                        T.gemm(stdz, dtk, z, policy=GEMM_P, wg_wait=AW)
                    else:
                        if symmetric_cut:
                            T.clear(z)
                        T.gemm(mz, dd, z, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    # Scan path: sp holds CP^T (DK, DV), the same operand layout as dmd (see the trace below).
                    if symmetric_cut:
                        T.clear(dq)
                    T.gemm(dos, sp, dq, transpose_B=from_ds, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    # The C x C GEMMs below (N = C = 64) are the ones that need 32 columns per warp: at 1024 threads
                    # they would get 16 and fall outside MP32 SQMMA's instruction shapes (inst_n in {128, 64, 32}).
                    if symmetric_cut:
                        T.clear(c1)
                    T.gemm(dos, uu, c1, transpose_B=True, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    if from_ds:
                        T.gemm(uu, dmd, dkb, transpose_B=True, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    else:
                        if symmetric_cut:
                            T.clear(c3)
                        T.gemm(uu, dnx, c3, transpose_B=True, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    if symmetric_cut:
                        T.clear(c1t)
                    T.gemm(uu, dos, c1t, transpose_B=True, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    if not from_ds:
                        _gemm_diagonal(dnx, rtt, diag, tx, GEMM_P, symmetric_cut)
                    T.wait_sqmma(0)
                    T.sync_threads()
                    if nfull == 1 and tx == G1_ISSUER:
                        if from_ds:
                            _issue_g1_ds(SourceDZ, StateDZ, DS, DO, sdz, stdz, dtk, dmd, dmn, dos,
                                         nslot,
                                         nhead,
                                         nstart,
                                         g1, GATES, gat, 1 - cb)
                        else:
                            _issue_g1(MZ, DO, RTt, mz, dd, dos, dnx, rtt,
                                      nslot,
                                      nhead,
                                      nstart,
                                      g1, GATES, gat, 1 - cb)
                    # Pair coefficients are loaded after GEMM1 (not at item start) so they do not compete with its accumulators;
                    # they are consumed in the pair phase, after elem2 and GEMM3.
                    for arr in T.unroll(PAIR_ARRAYS):
                        for e in T.vectorized(NE):
                            pco[arr, e] = PAIR[slot, head, tx, arr, e]
                    if not symmetric_cut:
                        for e in T.vectorized(NE):
                            pqt[e] = PAIRQ[slot, head, tx, e]
                    # Every accumulator / owner layout below gives thread tx the rows i == (tx % 256) // 4, so the row's gates
                    # are read once into registers (shared reloads after the zs / ys stores were serialised by aliasing).
                    for k in T.unroll(3):
                        grow[k] = gat[cb, k, tx % 256 // 4]
                    grow[3] = gat[cb, 0, C - 1]
                    if from_ds:
                        trv[0] = TRP[slot, head, tx % 32]
                    for part, i, lane in T.Parallel(P, C, 4):
                        sr = T.alloc_local((1,), "float32")
                        sr[0] = 0
                        if not from_ds:
                            for j in T.serial(C // (4 * P)):
                                col = part * (C // P) + j * 4 + lane
                                sr[0] += T.if_then_else(col == i, diag[i, col], 0)
                        a_ex[part, i, lane] = grow[3] * sr[0]
                    for i, v in T.Parallel(C, DV):
                        zs[i, v] = T.if_then_else(start + i < end, z[i, v], 0)
                        ys[i, v] = grow[2] * z[i, v]
                    if not from_ds:
                        for i, j in T.Parallel(C, C):
                            c3s[i, j] = c3[i, j]
                    T.sync_threads()

                    if symmetric_cut:
                        T.clear(dka)
                    T.gemm(zs, sp, dka, transpose_B=from_ds, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    if symmetric_cut:
                        T.clear(c2)
                    T.gemm(zs, uu, c2, transpose_B=True, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    if symmetric_cut:
                        T.clear(c2t)
                    T.gemm(uu, ys, c2t, transpose_B=True, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    _gemm_diagonal(zs, yy, diag, tx, GEMM_P, symmetric_cut)
                    if not from_ds:
                        if symmetric_cut:
                            T.clear(dkb)
                        T.gemm(c3s, sm, dkb, transpose_B=True, policy=GEMM_P, clear_accum=True, wg_wait=AW)
                    if start + C <= end:
                        for i, v in T.Parallel(C, DV):
                            DY[start + i, head, v] = ys[i, v]
                    else:
                        for i, v in T.Parallel(C, DV):
                            if start + i < end:
                                DY[start + i, head, v] = ys[i, v]
                    # While GEMM3 runs: the GEMM1-only half of the pair phase (P, P^T and sum(P^T o QK^T)) into registers, and the
                    # dQ scaling.
                    for i, d in T.Parallel(C, DK):
                        dq[i, d] *= scale * grow[0]
                    for part, i, lane in T.Parallel(P, C, 4):
                        if not symmetric_cut:
                            for k in T.unroll(ILP):
                                t2r[k] = 0
                        for j in T.unroll(C // (4 * P)):
                            col = part * (C // P) + j * 4 + lane
                            ev = T.cast(pco[0, j], "float32")
                            pmr[j] = T.cast(c1[i, col] * T.if_then_else(col <= i, scale * ev, 0), dtype)
                            ptr[j] = T.cast(c1t[i, col] * T.if_then_else(col >= i, scale * ev, 0), dtype)
                            if not symmetric_cut:
                                t2r[j % ILP] += T.cast(ptr[j], "float32") * pqt[j]
                    T.wait_sqmma(0)
                    T.sync_threads()
                    if nfull == 1 and tx == G2_ISSUER:
                        if from_ds:
                            _issue_g2_ds(U, Y, CP, uu, yy, sp,
                                         nslot,
                                         nhead, g2)
                        else:
                            _issue_g2(U, Y, CP, SourceMap, uu, yy, sp, sm,
                                      nslot,
                                      nhead, g2)
                    if ready == 1:
                        T.barrier_wait(g3, phase % 2)
                    for i, d in T.Parallel(C, DK):
                        dka[i, d] *= -grow[0] * grow[2]
                        dkb[i, d] *= grow[1]
                    for part, i, lane in T.Parallel(P, C, 4):
                        pa, pb = T.alloc_local((ILP,), "float32"), T.alloc_local((ILP,), "float32")
                        for k in T.unroll(ILP):
                            pa[k], pb[k] = 0, 0
                        # All K values first: interleaved load / FMA pairs were issued as dependent chains.
                        for j in T.unroll(DK // (4 * P)):
                            kqv[j] = qk2[i, part * (DK // P) + j * 4 + lane]
                        # ILP independent partial sums per dot: the dependent FMA chains were latency bound (4 warps / scheduler).
                        for j in T.unroll(DK // (4 * P)):
                            column = part * (DK // P) + j * 4 + lane
                            kv = T.cast(kqv[j], "float32")
                            pa[j % ILP] += dka[i, column] * kv
                            pb[j % ILP] += dkb[i, column] * kv
                        if symmetric_cut:
                            a_dl[part, i, lane] = 2 * _psum_local(pa)
                        else:
                            a_dl[part, i, lane] = _psum_local(pa) - _psum_local(pb)
                        a_ex[part, i, lane] += _psum_local(pb)
                    for i, d in T.Parallel(C, DK):
                        dka[i, d] += dkb[i, d]
                    for part, i, lane in T.Parallel(P, C, 4):
                        zy, s1 = T.alloc_local((1,), "float32"), T.alloc_local((ILP,), "float32")
                        if not symmetric_cut:
                            t1 = T.alloc_local((ILP,), "float32")
                        zy[0] = 0
                        for k in T.unroll(ILP):
                            s1[k] = 0
                            if not symmetric_cut:
                                t1[k] = 0
                        # Masks, scale and decay come from E / KK; c2t already carries
                        # beta_col (U ys^T) and sum(Pm o QK) is taken from the final dQ row dot after GEMM5. P / P^T were formed
                        # during GEMM3 (pmr / ptr).
                        for j in T.unroll(C // (4 * P)):
                            col = part * (C // P) + j * 4 + lane
                            ev = T.cast(pco[0, j], "float32")
                            kv = T.cast(pco[1, j], "float32")
                            al = T.if_then_else(col < i, -ev, 0)
                            au = T.if_then_else(col > i, -ev, 0)
                            zy[0] += T.if_then_else(col == i, diag[i, col], 0)
                            s1[j % ILP] += c2[i, col] * (al * kv)
                            if not symmetric_cut:
                                t1[j % ILP] += c2t[i, col] * (au * kv)
                            zs[i, col] = pmr[j]
                            if not symmetric_cut:
                                zs[i, C + col] = 0
                            ys[i, col] = grow[2] * c2[i, col] * al + c2t[i, col] * au
                            ys[i, C + col] = ptr[j]
                        if symmetric_cut:
                            a_dl[part, i, lane] += 2 * grow[2] * _psum_local(s1)
                        else:
                            a_dl[part, i, lane] += grow[2] * _psum_local(s1) - _psum_local(t1) - _psum_local(t2r)
                        a_db[part, i, lane] = zy[0] + _psum_local(s1)
                    if symmetric_cut:
                        T.fence_proxy_async()
                    T.sync_threads()
                    if symmetric_cut:
                        T.call_extern("handle", "tl::gdn_half_dq", T.access_ptr(zs, "r"), T.access_ptr(qk2, "r"), dq.data)
                    else:
                        T.gemm(zs, qk2, dq, policy=GEMM_P, wg_wait=AW)
                    T.gemm(ys, qk2, dka, policy=GEMM_P, wg_wait=AW)
                    # The Q values of the row dot below are loaded while GEMM5 runs (owner layout: tx = part * 256 + i * 4 + lane).
                    for j in T.unroll(DK // (4 * P)):
                        if symmetric_cut:
                            qv[j] = qk2[C + tx % 256 // 4, tx // 256 * (DK // P) + j * 4 + tx % 4]
                        else:
                            kqv[j] = qk2[C + tx % 256 // 4, tx // 256 * (DK // P) + j * 4 + tx % 4]
                    T.wait_sqmma(0)
                    # Row dot of the final dQ with Q: the dO_c CP term plus the pair term sum(Pm o QK).
                    for part, i, lane in T.Parallel(P, C, 4):
                        pq = T.alloc_local((ILP,), "float32")
                        for k in T.unroll(ILP):
                            pq[k] = 0
                        for j in T.unroll(DK // (4 * P)):
                            column = part * (DK // P) + j * 4 + lane
                            pq[j % ILP] += dq[i, column] * T.cast(qv[j] if symmetric_cut else kqv[j], "float32")
                        a_dl[part, i, lane] += _psum_local(pq)
                        if symmetric_cut:
                            # K.dK_final = pa + pb + beta*s1 + t1 + t2.
                            # Contract the final dK with the cached K row to
                            # recover the PAIRQ and upper-triangle contributions.
                            for k in T.unroll(ILP):
                                pq[k] = 0
                            for j in T.unroll(DK // (4 * P)):
                                column = part * (DK // P) + j * 4 + lane
                                pq[j % ILP] += dka[i, column] * T.cast(kqv[j], "float32")
                            a_dl[part, i, lane] -= _psum_local(pq)
                    for part, i, lane in T.Parallel(P, C, 4):
                        v0, v1, v2 = T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32")
                        v0[0], v1[0], v2[0] = a_dl[part, i, lane], a_db[part, i, lane], a_ex[part, i, lane]
                        _lane_sum4(v0, lane)
                        _lane_sum4(v1, lane)
                        _lane_sum4(v2, lane)
                        if lane == 0:
                            parts[(cb * 3 + 0) * P + part, i] = v0[0]
                            parts[(cb * 3 + 1) * P + part, i] = v1[0]
                            parts[(cb * 3 + 2) * P + part, i] = v2[0]
                    # dQ / dK leave through shared memory: accumulator rows hold every 4th column (2-byte global scatters), the
                    # staged tiles go out as 16-byte row vectors. The sync first lets every warp's GEMM5 finish reading zs / ys.
                    T.sync_threads()
                    for i, d in T.Parallel(C, DK):
                        zs[i, d] = dq[i, d]
                        ys[i, d] = dka[i, d]
                    if symmetric_cut:
                        T.fence_proxy_async()
                    T.sync_threads()
                    if symmetric_cut and start + C <= end:
                        # Finish the asynchronous shared read before zs/ys
                        # can be overwritten in the next iteration.
                        if tx == 128:
                            T.fence_proxy_async()
                            T.tma_copy(zs, DQ[start:start + C, head, :])
                            T.tma_copy(ys, DKOut[start:start + C, head, :])
                            T.tma_store_wait(0)
                    elif start + C <= end:
                        for i, d in T.Parallel(C, DK):
                            DQ[start + i, head, d] = zs[i, d]
                            DKOut[start + i, head, d] = ys[i, d]
                    else:
                        for i, d in T.Parallel(C, DK):
                            if start + i < end:
                                DQ[start + i, head, d] = zs[i, d]
                                DKOut[start + i, head, d] = ys[i, d]
                    if nfull == 1 and tx == G3_ISSUER:
                        _issue_g3(KQ, qk2, nslot, nhead, g3)
                    if ready == 1:
                        phase += 1
                    ready = nfull
                    if tx >= SUF_BASE and tx < SUF_BASE + 32:
                        sl = tx - SUF_BASE
                        lo, hi, ex = T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32")
                        lo[0], hi[0] = psum(parts, cb * 3 + 0, sl), psum(parts, cb * 3 + 0, sl + 32)
                        ex[0] = psum(parts, cb * 3 + 2, sl) + psum(parts, cb * 3 + 2, sl + 32)
                        if from_ds:
                            ex[0] += grow[3] * trv[0]
                        _suffix_step(lo, hi, ex, sl, 1)
                        _suffix_step(lo, hi, ex, sl, 2)
                        _suffix_step(lo, hi, ex, sl, 4)
                        _suffix_step(lo, hi, ex, sl, 8)
                        _suffix_step(lo, hi, ex, sl, 16)
                        lo[0] += T.shfl_sync(hi[0], 0)
                        if start + sl < end:
                            DG[start + sl, head] = lo[0] + ex[0]
                            DB[start + sl, head] = psum(parts, cb * 3 + 1, sl)
                        if start + 32 + sl < end:
                            DG[start + 32 + sl, head] = hi[0] + ex[0]
                            DB[start + 32 + sl, head] = psum(parts, cb * 3 + 1, sl + 32)
                else:
                    ready = 0

    return fused_persist


# Explicit T.tma_copy needs TMA lowering; prefetched buffers stay live across the item loop, so no shared merging.
PERSIST_CONFIG = {
    **FUSED_CONFIG,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
}


def _persist_flags(postra, reg_limit=244):
    """Fixed register cap; optionally enable post-RA scheduling."""
    flags = [flag.replace("-mtgpu-maxregcnt=121", f"-mtgpu-maxregcnt={reg_limit}") for flag in GRAD_COMPILE_FLAGS]
    if postra:
        for index, flag in enumerate(flags):
            if flag == "-mtgpu-enable-postra-sched=0":
                assert index > 0 and flags[index - 1] == "-mllvm"
                return flags[:index - 1] + flags[index + 1:]
    return flags


PERSIST_FLAGS = _persist_flags(False)
PERSIST_FLAGS_POSTRA = _persist_flags(True)
PERSIST_FLAGS_CUT_POSTRA = _persist_flags(True, reg_limit=256)


# Each compiler configuration has its own JIT cache entries.
@tilelang.jit(target="musa", pass_configs=PERSIST_CONFIG, compile_flags=PERSIST_FLAGS)
def _chunk_fused_persistent(H, DK, DV, dtype, cu_dtype, scale, ctas, from_ds=False, single_sequence=False):
    return _fused_persistent_builder(H, DK, DV, dtype, cu_dtype, scale, ctas, from_ds, single_sequence)


@tilelang.jit(target="musa", pass_configs=PERSIST_CONFIG, compile_flags=PERSIST_FLAGS_POSTRA)
def _chunk_fused_persistent_postra(H, DK, DV, dtype, cu_dtype, scale, ctas, from_ds=False, single_sequence=False):
    return _fused_persistent_builder(H, DK, DV, dtype, cu_dtype, scale, ctas, from_ds, single_sequence)


@tilelang.jit(target="musa", pass_configs=PERSIST_CONFIG, compile_flags=PERSIST_FLAGS_CUT_POSTRA)
def _chunk_fused_cut_postra(H, DK, DV, dtype, cu_dtype, scale, ctas, single_sequence):
    return _fused_persistent_builder(H, DK, DV, dtype, cu_dtype, scale, ctas, single_sequence=single_sequence)


def chunk_fused_persistent(H, DK, DV, dtype, cu_dtype, scale, ctas, from_ds=False, single_sequence=False):
    """Persistent body with post-RA scheduling enabled above 32 heads."""
    if H > 32:
        if _is_symmetric_cut(not from_ds, DK, DV):
            return _chunk_fused_cut_postra(H, DK, DV, dtype, cu_dtype, scale, ctas, single_sequence)
        return _chunk_fused_persistent_postra(H, DK, DV, dtype, cu_dtype, scale, ctas, from_ds, single_sequence)
    return _chunk_fused_persistent(H, DK, DV, dtype, cu_dtype, scale, ctas, from_ds, single_sequence)
