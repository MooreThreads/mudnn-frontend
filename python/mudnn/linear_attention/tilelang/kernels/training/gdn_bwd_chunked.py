# SPDX-License-Identifier: Apache-2.0
# Chunk-64 WY inverse-adjoint algorithm adapted from NVIDIA/cudnn-frontend
# 33c6fcb54d110f2c4fbeaf7c0330067492142189; MUSA scheduling is independent.

"""Separate state-gradient scan from chunk/value-parallel gradient projection."""

from dataclasses import dataclass

import torch
import tilelang
import tilelang.language as T

from .._jit import BASE_COMPILE_FLAGS, BASE_PASS_CONFIGS

CONFIG = {
    **BASE_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: False,
    # TMA stores cannot represent the padded shared-memory row pitch.
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
}
C, BV = 64, 64
SCAN_BV = 64
LOG2E = 1.4426950408889634


def _exp(value):
    return T.exp2(value * LOG2E)


@T.macro
def _sum_eight(value, lane):
    other = T.alloc_local((1,), "float32")
    other[0] = T.shfl_down(value[0], 4)
    if lane < 4:
        value[0] += other[0]
    other[0] = T.shfl_down(value[0], 2)
    if lane < 2:
        value[0] += other[0]
    other[0] = T.shfl_down(value[0], 1)
    if lane == 0:
        value[0] += other[0]


@T.macro
def _pair_sums(matrix, rows, columns):
    row_values = T.alloc_fragment((C, 8), "float32")
    column_values = T.alloc_fragment((C, 8), "float32")
    T.sync_threads()
    for i, lane in T.Parallel(C, 8):
        row = T.alloc_local((1,), "float32")
        column = T.alloc_local((1,), "float32")
        row[0], column[0] = 0, 0
        for j in T.serial(C // 8):
            row[0] += T.cast(matrix[i, j * 8 + lane], "float32")
            column[0] += T.cast(matrix[j * 8 + lane, i], "float32")
        _sum_eight(row, lane)
        _sum_eight(column, lane)
        row_values[i, lane], column_values[i, lane] = row[0], column[0]
    T.sync_threads()
    for i, lane in T.Parallel(C, 8):
        if lane == 0:
            rows[i], columns[i] = row_values[i, lane], column_values[i, lane]
    T.sync_threads()


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def prepare_tiles(H, DK, DV, dtype, cu_dtype, scale, with_yu=True):
    # with_yu=False: Y and U were already written by recompute_tme (same math), so the checkpoint read and the
    # two state GEMMs are skipped.
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")

    @T.prim_func
    def kernel(
        Q: T.Tensor((tokens, H, DK), dtype),
        K: T.Tensor((tokens, H, DK), dtype),
        V: T.Tensor((tokens, H, DV), dtype),
        B: T.Tensor((tokens, H), "float32"),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        W: T.Tensor((capacity, H, C, C), dtype),
        L: T.Tensor((capacity, H, C), "float32"),
        CP: T.Tensor((capacity, H, DV, DK), dtype),
        Y: T.Tensor((capacity, H, C, DV), dtype),
        U: T.Tensor((capacity, H, C, DV), dtype),
        WT: T.Tensor((capacity, H, C, C), dtype),
        PT: T.Tensor((capacity, H, C, C), dtype),
        QT: T.Tensor((capacity, H, DK, C), dtype),
        KT: T.Tensor((capacity, H, DK, C), dtype),
        Exps: T.Tensor((capacity, H, 2, C), "float32"),
        Decay: T.Tensor((capacity, H, C, C), "float32"),
    ):
        with T.Kernel(capacity, H, threads=128) as (slot, head):
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
                qs = T.alloc_shared((C, DK), dtype)
                ks = T.alloc_shared((C, DK), dtype)
                state = T.alloc_shared((DK, DV), dtype)
                wm = T.alloc_shared((C, C), dtype)
                rhs = T.alloc_shared((DV, C), dtype)
                ls = T.alloc_shared((C,), "float32")
                alpha = T.alloc_shared((C,), "float32")
                beta = T.alloc_shared((C,), "float32")
                cv = T.alloc_fragment((C, DV), "float32")
                cc = T.alloc_fragment((C, C), "float32")
                for i, d in T.Parallel(C, DK):
                    qs[i, d] = T.if_then_else(start + i < end, Q[T.min(start + i, end - 1), head, d], 0)
                    ks[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                if with_yu:
                    for d, v in T.Parallel(DK, DV):
                        state[d, v] = CP[slot, head, v, d]
                for i in T.Parallel(C):
                    ls[i] = L[slot, head, i]
                    beta[i] = T.if_then_else(start + i < end, B[T.min(start + i, end - 1), head], 0)
                for i, j in T.Parallel(C, C):
                    wm[i, j] = W[slot, head, i, j]
                    WT[slot, head, i, j] = W[slot, head, j, i]
                T.sync_threads()
                for i in T.Parallel(C):
                    alpha[i] = _exp(ls[i])
                    Exps[slot, head, 0, i] = _exp(ls[i])
                    Exps[slot, head, 1, i] = _exp(ls[C - 1] - ls[i])
                T.sync_threads()
                if with_yu:
                    T.gemm(ks, state, cv, clear_accum=True)
                    for i, v in T.Parallel(C, DV):
                        residual = T.alloc_local((1,), dtype)
                        value = T.if_then_else(start + i < end, V[T.min(start + i, end - 1), head, v], 0)
                        residual[0] = value - alpha[i] * cv[i, v]
                        Y[slot, head, i, v] = residual[0]
                        rhs[v, i] = beta[i] * residual[0]
                    T.sync_threads()
                    T.gemm(wm, rhs, cv, transpose_B=True, clear_accum=True)
                    for i, v in T.Parallel(C, DV):
                        U[slot, head, i, v] = cv[i, v]
                T.gemm(qs, ks, cc, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(C, C):
                    pair = T.if_then_else(i >= j, _exp(ls[i] - ls[j]), 0)
                    PT[slot, head, j, i] = T.if_then_else(i >= j, scale * pair * cc[i, j], 0)
                    Decay[slot, head, i, j] = pair
                for d, i in T.Parallel(DK, C):
                    QT[slot, head, d, i] = qs[i, d] * scale * alpha[i]
                    KT[slot, head, d, i] = -ks[i, d] * alpha[i]

    return kernel


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def prepare_maps(H, DK, dtype, cu_dtype):
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")

    @T.prim_func
    def kernel(
        K: T.Tensor((tokens, H, DK), dtype),
        B: T.Tensor((tokens, H), "float32"),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        WT: T.Tensor((capacity, H, C, C), dtype),
        PT: T.Tensor((capacity, H, C, C), dtype),
        QT: T.Tensor((capacity, H, DK, C), dtype),
        KT: T.Tensor((capacity, H, DK, C), dtype),
        Exps: T.Tensor((capacity, H, 2, C), "float32"),
        StateMap: T.Tensor((capacity, H, DK, DK), dtype),
        SourceMap: T.Tensor((capacity, H, DK, C), dtype),
        StateDZ: T.Tensor((capacity, H, C, DK), dtype),
        SourceDZ: T.Tensor((capacity, H, C, C), dtype),
    ):
        with T.Kernel(capacity, H, threads=512) as (slot, head):
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
                wt = T.alloc_shared((C, C), dtype)
                pt = T.alloc_shared((C, C), dtype)
                kt = T.alloc_shared((DK, C), dtype)
                tail_k = T.alloc_shared((DK, C), dtype)
                weighted_z = T.alloc_shared((C, DK), dtype)
                weighted_o = T.alloc_shared((C, C), dtype)
                beta = T.alloc_shared((C,), "float32")
                cz = T.alloc_fragment((C, DK), "float32")
                co = T.alloc_fragment((C, C), "float32")
                ss = T.alloc_fragment((DK, DK), "float32")
                so = T.alloc_fragment((DK, C), "float32")
                T.copy(WT[slot, head, :, :], wt)
                T.copy(PT[slot, head, :, :], pt)
                T.copy(KT[slot, head, :, :], kt)
                for i in T.Parallel(C):
                    beta[i] = T.if_then_else(start + i < end, B[T.min(start + i, end - 1), head], 0)
                for d, i in T.Parallel(DK, C):
                    tail_k[d, i] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0) * Exps[slot, head, 1, i]
                T.sync_threads()
                T.gemm(wt, tail_k, cz, transpose_B=True, clear_accum=True)
                T.gemm(wt, pt, co, clear_accum=True)
                for i, d in T.Parallel(C, DK):
                    rounded = T.alloc_local((1,), dtype)
                    rounded[0] = cz[i, d]
                    StateDZ[slot, head, i, d] = rounded[0]
                    weighted_z[i, d] = beta[i] * rounded[0]
                for i, j in T.Parallel(C, C):
                    rounded = T.alloc_local((1,), dtype)
                    rounded[0] = co[i, j]
                    SourceDZ[slot, head, i, j] = rounded[0]
                    weighted_o[i, j] = beta[i] * rounded[0]
                T.sync_threads()
                T.gemm(kt, weighted_z, ss, clear_accum=True)
                T.gemm(kt, weighted_o, so, clear_accum=True)
                for d, j in T.Parallel(DK, DK):
                    StateMap[slot, head, d, j] = ss[d, j]
                for d, i in T.Parallel(DK, C):
                    SourceMap[slot, head, d, i] = so[d, i] + QT[slot, head, d, i]

    return kernel


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def source_grad(H, DK, DV, dtype, cu_dtype):
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")
    SV, NV = SCAN_BV, DV // SCAN_BV

    @T.prim_func
    def kernel(
        DO: T.Tensor((tokens, H, DV), dtype),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        SourceMap: T.Tensor((capacity, H, DK, C), dtype),
        Source: T.Tensor((capacity, H, DK, DV), dtype),
    ):
        with T.Kernel(capacity, H, NV, threads=256) as (slot, head, panel):
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
                dos = T.alloc_shared((C, SV), dtype)
                sm = T.alloc_shared((DK, C), dtype)
                output = T.alloc_shared((DK, SV + 8), dtype)
                state = T.alloc_fragment((DK, SV), "float32")
                T.copy(SourceMap[slot, head, :, :], sm)
                for i, v in T.Parallel(C, SV):
                    dos[i, v] = T.if_then_else(start + i < end, DO[T.min(start + i, end - 1), head, panel * SV + v], 0)
                T.sync_threads()
                T.gemm(sm, dos, state, clear_accum=True)
                for d, v in T.Parallel(DK, SV):
                    output[d, v] = state[d, v]
                T.sync_threads()
                for d, v in T.Parallel(DK, SV):
                    Source[slot, head, d, panel * SV + v] = output[d, v]

    return kernel


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def state_scan(H, DK, DV, dtype, cu_dtype, has_final, kmajor=False, fuse_drive=False):
    seqs, capacity, tokens = T.dynamic("seqs"), T.dynamic("capacity"), T.dynamic("tokens")
    # StateMap is shared by all value panels of a head; wider panels cut its repeated loads.
    SV, NV = SCAN_BV, DV // SCAN_BV
    # kmajor: DS is written as (DK, DV) straight from the GEMM operand tile (no transposed record staging).
    DS_SHAPE = (capacity, H, DK, DV) if kmajor else (capacity, H, DV, DK)

    @T.prim_func
    def kernel(
        CU: T.Tensor((seqs + 1,), cu_dtype),
        StateMap: T.Tensor((capacity, H, DK, DK), dtype),
        Exps: T.Tensor((capacity, H, 2, C), "float32"),
        Source: T.Tensor((capacity, H, DK, DV), dtype),
        DO: T.Tensor((tokens, H, DV), dtype),
        SourceMap: T.Tensor((capacity, H, DK, C), dtype),
        DSFinal: T.Tensor((seqs, H, DV, DK), "float32"),
        DS: T.Tensor(DS_SHAPE, dtype),
        DS0: T.Tensor((seqs, H, DV, DK), "float32"),
    ):
        with T.Kernel(seqs, H, NV, threads=512) as (seq, head, panel):
            offset = T.alloc_var("int32")
            offset = 0
            for n in T.serial(seq):
                offset += T.ceildiv(CU[n + 1] - CU[n], C)
            count = T.ceildiv(CU[seq + 1] - CU[seq], C)
            matrix = T.alloc_shared((DK, DK), dtype)
            ds = T.alloc_shared((DK, SV), dtype)
            current_state = T.alloc_fragment((DK, SV), "float32")
            record = T.alloc_shared((SV, DK + 8), dtype)
            final_record = T.alloc_shared((SV, DK + 4), "float32")
            source = T.alloc_shared((DK, SV + 8), dtype)
            alpha_last = T.alloc_shared((1,), "float32")
            update = T.alloc_fragment((DK, SV), "float32")
            if fuse_drive:
                # Source_c = SourceMap_c dO_c accumulated into the same GEMM output (no drive kernel, fp32 source).
                smap = T.alloc_shared((DK, C), dtype)
                dob = T.alloc_shared((C, SV), dtype)
            for d, v in T.Parallel(DK, SV):
                current_state[d, v] = DSFinal[seq, head, panel * SV + v, d] if has_final else 0
            T.sync_threads()
            for reverse in T.serial(count):
                chunk = count - 1 - reverse
                slot = offset + chunk
                if not kmajor:
                    for d, v in T.Parallel(DK, SV):
                        record[v, d] = current_state[d, v]
                for d, v in T.Parallel(DK, SV):
                    current = current_state[d, v]
                    ds[d, v] = current
                T.sync_threads()
                if kmajor:
                    T.copy(ds, DS[slot, head, :, panel * SV : (panel + 1) * SV])
                else:
                    T.copy(record[:, :DK], DS[slot, head, panel * SV : (panel + 1) * SV, :])
                T.copy(StateMap[slot, head, :, :], matrix)
                if fuse_drive:
                    T.copy(SourceMap[slot, head, :, :], smap)
                    start, end = CU[seq] + chunk * C, CU[seq + 1]
                    if start + C <= end:
                        for i, v in T.Parallel(C, SV):
                            dob[i, v] = DO[start + i, head, panel * SV + v]
                    else:
                        for i, v in T.Parallel(C, SV):
                            dob[i, v] = T.if_then_else(start + i < end, DO[T.min(start + i, end - 1), head, panel * SV + v], 0)
                else:
                    T.copy(Source[slot, head, :, panel * SV : (panel + 1) * SV], source[:, :SV])
                if T.get_thread_binding() == 0:
                    alpha_last[0] = Exps[slot, head, 0, C - 1]
                T.sync_threads()
                T.gemm(matrix, ds, update, clear_accum=True)
                if fuse_drive:
                    T.gemm(smap, dob, update)
                for d, v in T.Parallel(DK, SV):
                    if fuse_drive:
                        current_state[d, v] = alpha_last[0] * current_state[d, v] + update[d, v]
                    else:
                        current_state[d, v] = alpha_last[0] * current_state[d, v] + update[d, v] + T.cast(source[d, v], "float32")
                T.sync_threads()
            for d, v in T.Parallel(DK, SV):
                final_record[v, d] = current_state[d, v]
            T.sync_threads()
            T.copy(final_record[:, :DK], DS0[seq, head, panel * SV : (panel + 1) * SV, :])

    return kernel


# Explicit T.tma_copy needs TMA lowering; prefetch buffers stay live across the chunk loop, so no shared merging.
SCAN_TME_CONFIG = {
    **CONFIG,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
}


@tilelang.jit(target="musa", pass_configs=SCAN_TME_CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def state_scan_tme(H, DK, DV, dtype, cu_dtype, has_final, stages=2):
    """K-major state scan with the drive GEMM fused (same contract as state_scan(kmajor=True, fuse_drive=True)).

    The step latency was dominated by loads that do not depend on the carried state, so they are pipelined with TME:
    StateMap_c is double-buffered two steps ahead, and Source_{c-1} = SourceMap_{c-1} dO_{c-1} is computed one step
    ahead into registers. Only the first processed chunk (the last, possibly partial chunk of a sequence) uses
    per-thread loads with zero fill. stages=1 keeps a single StateMap buffer (refilled right after its GEMM; ~72 KB of
    shared, three CTAs per SM) for grids with more than two scan CTAs per SM.
    """
    seqs, capacity, tokens = T.dynamic("seqs"), T.dynamic("capacity"), T.dynamic("tokens")
    SV, NV = SCAN_BV, DV // SCAN_BV

    @T.prim_func
    def scan_tme(
        CU: T.Tensor((seqs + 1,), cu_dtype),
        StateMap: T.Tensor((capacity, H, DK, DK), dtype),
        Exps: T.Tensor((capacity, H, 2, C), "float32"),
        Source: T.Tensor((capacity, H, DK, DV), dtype),
        DO: T.Tensor((tokens, H, DV), dtype),
        SourceMap: T.Tensor((capacity, H, DK, C), dtype),
        DSFinal: T.Tensor((seqs, H, DV, DK), "float32"),
        DS: T.Tensor((capacity, H, DK, DV), dtype),
        DS0: T.Tensor((seqs, H, DV, DK), "float32"),
    ):
        with T.Kernel(seqs, H, NV, threads=512) as (seq, head, panel):
            offset = T.alloc_var("int32")
            offset = 0
            for n in T.serial(seq):
                offset += T.ceildiv(CU[n + 1] - CU[n], C)
            count = T.ceildiv(CU[seq + 1] - CU[seq], C)
            mat0 = T.alloc_shared((DK, DK), dtype)
            # The single-buffer variant never touches mat1 (constant branches are still traced).
            mat1 = T.alloc_shared((DK, DK), dtype) if stages == 2 else mat0
            smap, dob = T.alloc_shared((DK, C), dtype), T.alloc_shared((C, SV), dtype)
            ds = T.alloc_shared((DK, SV), dtype)
            bm0, bm1, bs = T.alloc_barrier(arrive_count=1), T.alloc_barrier(arrive_count=1), T.alloc_barrier(arrive_count=1)
            current_state = T.alloc_fragment((DK, SV), "float32")
            update = T.alloc_fragment((DK, SV), "float32")
            src_cur = T.alloc_fragment((DK, SV), "float32")
            src_next = T.alloc_fragment((DK, SV), "float32")
            alpha_cur = T.alloc_var("float32")
            alpha_next = T.alloc_var("float32")
            tx = T.get_thread_binding()
            for d, v in T.Parallel(DK, SV):
                current_state[d, v] = DSFinal[seq, head, panel * SV + v, d] if has_final else 0
            if count > 0:
                last = offset + count - 1
                if tx == 0:
                    T.tma_copy(StateMap[last, head, :, :], mat0, barrier=bm0)
                    T.barrier_arrive(bm0)
                    if count > 1 and stages == 2:
                        T.tma_copy(StateMap[last - 1, head, :, :], mat1, barrier=bm1)
                        T.barrier_arrive(bm1)
                # Source of the last (possibly partial) chunk with per-thread loads and zero fill.
                start, end = CU[seq] + (count - 1) * C, CU[seq + 1]
                T.copy(SourceMap[last, head, :, :], smap)
                for i, v in T.Parallel(C, SV):
                    dob[i, v] = T.if_then_else(start + i < end, DO[T.min(start + i, end - 1), head, panel * SV + v], 0)
                T.sync_threads()
                T.gemm(smap, dob, src_cur, clear_accum=True)
                T.sync_threads()
                if count > 1 and tx == 0:
                    T.tma_copy(SourceMap[last - 1, head, :, :], smap, barrier=bs)
                    T.tma_copy(DO[CU[seq] + (count - 2) * C : CU[seq] + (count - 1) * C, head, panel * SV : (panel + 1) * SV], dob, barrier=bs)
                    T.barrier_arrive(bs)
                alpha_cur = Exps[last, head, 0, C - 1]
            for reverse in T.serial(count):
                chunk = count - 1 - reverse
                slot = offset + chunk
                if reverse + 1 < count:
                    alpha_next = Exps[slot - 1, head, 0, C - 1]
                for d, v in T.Parallel(DK, SV):
                    ds[d, v] = current_state[d, v]
                T.sync_threads()
                # Per-thread stores: with TMA lowering on, T.copy would become a TME store without a proxy fence.
                for d, v in T.Parallel(DK, SV):
                    DS[slot, head, d, panel * SV + v] = ds[d, v]
                if stages == 1:
                    T.barrier_wait(bm0, reverse % 2)
                    T.gemm(mat0, ds, update, clear_accum=True)
                elif reverse % 2 == 0:
                    T.barrier_wait(bm0, (reverse // 2) % 2)
                    T.gemm(mat0, ds, update, clear_accum=True)
                else:
                    T.barrier_wait(bm1, (reverse // 2) % 2)
                    T.gemm(mat1, ds, update, clear_accum=True)
                if reverse + 1 < count:
                    T.barrier_wait(bs, reverse % 2)
                    T.gemm(smap, dob, src_next, clear_accum=True)
                # Every warp is past both GEMMs: refill the consumed StateMap buffer two steps ahead and the
                # SourceMap / dO tiles one step ahead (chunks before the last one are always full).
                T.sync_threads()
                if tx == 0:
                    if stages == 1 and reverse + 1 < count:
                        T.tma_copy(StateMap[slot - 1, head, :, :], mat0, barrier=bm0)
                        T.barrier_arrive(bm0)
                    if reverse + 2 < count:
                        if stages == 1:
                            pass
                        elif reverse % 2 == 0:
                            T.tma_copy(StateMap[slot - 2, head, :, :], mat0, barrier=bm0)
                            T.barrier_arrive(bm0)
                        else:
                            T.tma_copy(StateMap[slot - 2, head, :, :], mat1, barrier=bm1)
                            T.barrier_arrive(bm1)
                        T.tma_copy(SourceMap[slot - 2, head, :, :], smap, barrier=bs)
                        T.tma_copy(DO[CU[seq] + (chunk - 2) * C : CU[seq] + (chunk - 1) * C, head, panel * SV : (panel + 1) * SV], dob, barrier=bs)
                        T.barrier_arrive(bs)
                for d, v in T.Parallel(DK, SV):
                    current_state[d, v] = alpha_cur * current_state[d, v] + update[d, v] + src_cur[d, v]
                    src_cur[d, v] = src_next[d, v]
                alpha_cur = alpha_next
            for d, v in T.Parallel(DK, SV):
                DS0[seq, head, panel * SV + v, d] = current_state[d, v]

    return scan_tme


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def state_trace(H, DK, DV, dtype):
    """TRP[slot, head, w] = warp w's part of sum(DS_c o CPT_c) (scan path; consumed by the persistent fused kernel).

    One 1024-thread CTA per (chunk slot, head); each thread dots DK * DV / 1024 contiguous elements with 16-byte loads.
    """
    capacity = T.dynamic("capacity")
    per = DK * DV // (1024 * 8)
    assert per * 1024 * 8 == DK * DV

    @T.prim_func
    def trace(
        DS: T.Tensor((capacity, H, DK, DV), dtype),
        CPT: T.Tensor((capacity, H, DK, DV), dtype),
        TRP: T.Tensor((capacity, H, 32), "float32"),
    ):
        with T.Kernel(capacity, H, threads=1024) as (slot, head):
            tx = T.get_thread_binding()
            acc = T.alloc_local((1,), "float32")
            da, ca = T.alloc_local((8,), dtype), T.alloc_local((8,), dtype)
            acc[0] = 0
            for g in T.serial(per):
                flat = (g * 1024 + tx) * 8
                for e in T.vectorized(8):
                    da[e] = DS[slot, head, flat // DV, flat % DV + e]
                    ca[e] = CPT[slot, head, flat // DV, flat % DV + e]
                for e in T.serial(8):
                    acc[0] += T.cast(da[e], "float32") * T.cast(ca[e], "float32")
            acc[0] += T.shfl_xor(acc[0], 16)
            acc[0] += T.shfl_xor(acc[0], 8)
            acc[0] += T.shfl_xor(acc[0], 4)
            acc[0] += T.shfl_xor(acc[0], 2)
            acc[0] += T.shfl_xor(acc[0], 1)
            if tx % 32 == 0:
                TRP[slot, head, tx // 32] = acc[0]

    return trace


@tilelang.jit(target="musa", pass_configs=SCAN_TME_CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def recompute_tme(H, DK, DV, dtype, cu_dtype, has_initial):
    """Forward state recompute (checkpoint per chunk) that also emits the chunk tiles Y and U.

    Same math and roundings as gdn_bwd_kernels.recompute + prepare_tiles' Y / U: with S_c the bf16 checkpoint,
    Y = V - alpha K S_c, U = W (beta Y), S_{c+1} = e^{L_last} S_c + K^T (U o e^{L_last - L}).
    One CTA per (sequence, head, value panel); K, V, W of the next chunk arrive by TME into the other buffer while
    the current chunk computes. The last (possibly partial) chunk of a sequence uses per-thread loads with zero fill.
    CP^T is prepared separately for the scan path; the CPT argument is an unused
    shape-compatible placeholder in this recomputation kernel.
    """
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")
    SV, NV = SCAN_BV, DV // SCAN_BV

    @T.macro
    def issue(K, V, W, ks, kts, vs, ws, bar, slot, head, start, panel):
        # K twice: K-major A operand (K S) and MN-major A operand (K^T U) need different shared layouts.
        T.tma_copy(K[start : start + C, head, :], ks, barrier=bar)
        T.tma_copy(K[start : start + C, head, :], kts, barrier=bar)
        T.tma_copy(V[start : start + C, head, panel * SV : (panel + 1) * SV], vs, barrier=bar)
        T.tma_copy(W[slot, head, :, :], ws, barrier=bar)
        T.barrier_arrive(bar)

    @T.prim_func
    def recompute_kernel(
        K: T.Tensor((tokens, H, DK), dtype),
        V: T.Tensor((tokens, H, DV), dtype),
        B: T.Tensor((tokens, H), "float32"),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        W: T.Tensor((capacity, H, C, C), dtype),
        L: T.Tensor((capacity, H, C), "float32"),
        S0: T.Tensor((seqs, H, DV, DK), "float32"),
        Checkpoints: T.Tensor((capacity, H, DV, DK), dtype),
        Y: T.Tensor((capacity, H, C, DV), dtype),
        U: T.Tensor((capacity, H, C, DV), dtype),
        CPT: T.Tensor((capacity, H, DK, DV), dtype),
    ):
        with T.Kernel(NV, seqs, H, threads=256) as (bx, by, bz):
            seq, head, panel = (by, bz, bx)
            offset = T.alloc_var("int32")
            offset = 0
            for n in T.serial(seq):
                offset += T.ceildiv(CU[n + 1] - CU[n], C)
            count = T.ceildiv(CU[seq + 1] - CU[seq], C)
            end = CU[seq + 1]
            k0, k1 = T.alloc_shared((C, DK), dtype), T.alloc_shared((C, DK), dtype)
            kt0, kt1 = T.alloc_shared((C, DK), dtype), T.alloc_shared((C, DK), dtype)
            # The state is staged once, transposed, as both
            # the K S operand and the checkpoint record, so the CTA fits in < 128 KB of shared memory (two CTAs per MP).
            v0, v1 = (T.alloc_shared((C, SV), dtype), T.alloc_shared((C, SV), dtype))
            w0, w1 = T.alloc_shared((C, C), dtype), T.alloc_shared((C, C), dtype)
            b0, b1 = T.alloc_barrier(arrive_count=1), T.alloc_barrier(arrive_count=1)
            k1, kt1, v1, w1, b1 = k0, kt0, v0, w0, b0
            sst = T.alloc_shared((SV, DK), dtype)
            rhs = T.alloc_shared((C, SV), dtype)
            us = T.alloc_shared((C, SV), dtype)
            gl = T.alloc_shared((2, C), "float32")
            gb = T.alloc_shared((2, C), "float32")
            state = T.alloc_fragment((DK, SV), "float32")
            cv = T.alloc_fragment((C, SV), "float32")
            update = T.alloc_fragment((DK, SV), "float32")
            nl, nb = T.alloc_local((1,), "float32"), T.alloc_local((1,), "float32")
            tx = T.get_thread_binding()
            for d, v in T.Parallel(DK, SV):
                state[d, v] = S0[seq, head, panel * SV + v, d] if has_initial else 0
            if count > 0:
                if tx < C:
                    gl[0, tx] = L[offset, head, tx]
                    gb[0, tx] = T.if_then_else(CU[seq] + tx < end, B[T.min(CU[seq] + tx, end - 1), head], 0)
                if tx == 0:
                    if count > 1:
                        issue(K, V, W, k0, kt0, v0, w0, b0, offset, head, CU[seq], panel)
            for chunk in T.serial(count):
                slot = offset + chunk
                start = CU[seq] + chunk * C
                cb = chunk % 2
                full = chunk + 1 < count
                # Next chunk's gates (in flight during this chunk's compute, stored at the end).
                if tx < C and chunk + 1 < count:
                    nl[0] = L[slot + 1, head, tx]
                    nb[0] = T.if_then_else(start + C + tx < end, B[T.min(start + C + tx, end - 1), head], 0)
                for d, v in T.Parallel(DK, SV):
                    sst[v, d] = state[d, v]
                if not full:
                    # Last chunk: per-thread loads with zero fill past the sequence end (buffer 0 / 1 by parity).
                    if cb == 0:
                        for i, d in T.Parallel(C, DK):
                            k0[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                            kt0[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                        for i, v in T.Parallel(C, SV):
                            v0[i, v] = T.if_then_else(start + i < end, V[T.min(start + i, end - 1), head, panel * SV + v], 0)
                        T.copy(W[slot, head, :, :], w0)
                    else:
                        for i, d in T.Parallel(C, DK):
                            k1[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                            kt1[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                        for i, v in T.Parallel(C, SV):
                            v1[i, v] = T.if_then_else(start + i < end, V[T.min(start + i, end - 1), head, panel * SV + v], 0)
                        T.copy(W[slot, head, :, :], w1)
                T.sync_threads()
                for v, d in T.Parallel(SV, DK):
                    Checkpoints[slot, head, panel * SV + v, d] = sst[v, d]
                if full:
                    T.barrier_wait(b0, chunk % 2)
                if cb == 0:
                    T.gemm(k0, sst, cv, transpose_B=True, clear_accum=True)
                else:
                    T.gemm(k1, sst, cv, transpose_B=True, clear_accum=True)
                for i, v in T.Parallel(C, SV):
                    residual = T.alloc_local((1,), dtype)
                    value = T.if_then_else(cb == 0, v0[i, v], v1[i, v])
                    residual[0] = value - _exp(gl[cb, i]) * cv[i, v]
                    Y[slot, head, i, panel * SV + v] = residual[0]
                    rhs[i, v] = gb[cb, i] * residual[0]
                T.sync_threads()
                if cb == 0:
                    T.gemm(w0, rhs, cv, clear_accum=True)
                else:
                    T.gemm(w1, rhs, cv, clear_accum=True)
                for i, v in T.Parallel(C, SV):
                    U[slot, head, i, panel * SV + v] = cv[i, v]
                    us[i, v] = cv[i, v] * _exp(gl[cb, C - 1] - gl[cb, i])
                T.sync_threads()
                if cb == 0:
                    T.gemm(kt0, us, update, transpose_A=True, clear_accum=True)
                else:
                    T.gemm(kt1, us, update, transpose_A=True, clear_accum=True)
                for d, v in T.Parallel(DK, SV):
                    state[d, v] = _exp(gl[cb, C - 1]) * state[d, v] + update[d, v]
                if tx < C and chunk + 1 < count:
                    gl[1 - cb, tx] = nl[0]
                    gb[1 - cb, tx] = nb[0]
                # All warps are past this chunk's GEMMs: refill its buffers with chunk + 2 (full chunks only).
                T.sync_threads()
                if tx == 0 and chunk + 1 < count - 1:
                    issue(K, V, W, k0, kt0, v0, w0, b0, slot + 1, head, start + C, panel)

    return recompute_kernel


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def initial_source(H, DK, DV, dtype, cu_dtype, has_final):
    """DS0 = SourceMap_0 dO_0 for sequences whose chunk-0 transfer is cut."""
    seqs, tokens, capacity = T.dynamic("seqs"), T.dynamic("tokens"), T.dynamic("capacity")
    SV, NV = SCAN_BV, DV // SCAN_BV

    @T.prim_func
    def kernel(
        DO: T.Tensor((tokens, H, DV), dtype),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        SourceMap: T.Tensor((capacity, H, DK, C), dtype),
        DSFinal: T.Tensor((seqs, H, DV, DK), "float32"),
        DS0: T.Tensor((seqs, H, DV, DK), "float32"),
    ):
        with T.Kernel(seqs, H, NV, threads=256) as (seq, head, panel):
            offset = T.alloc_var("int32")
            offset = 0
            for n in T.serial(seq):
                offset += T.ceildiv(CU[n + 1] - CU[n], C)
            start, end = CU[seq], CU[seq + 1]
            smap = T.alloc_shared((DK, C), dtype)
            dot = T.alloc_shared((SV, C), dtype)
            out = T.alloc_fragment((SV, DK), "float32")
            for d, i in T.Parallel(DK, C):
                smap[d, i] = T.if_then_else(start < end, SourceMap[T.min(offset, capacity - 1), head, d, i], 0)
            for v, i in T.Parallel(SV, C):
                dot[v, i] = T.if_then_else(start + i < end, DO[T.min(start + i, end - 1), head, panel * SV + v], 0)
            T.sync_threads()
            T.gemm(dot, smap, out, transpose_B=True, clear_accum=True)
            for v, d in T.Parallel(SV, DK):
                if has_final:
                    DS0[seq, head, panel * SV + v, d] = out[v, d] + T.if_then_else(start < end, 0, DSFinal[seq, head, panel * SV + v, d])
                else:
                    DS0[seq, head, panel * SV + v, d] = out[v, d]

    return kernel


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def reduce_grad(H, DK, dtype, panels=2):
    tokens = T.dynamic("tokens")
    BLOCK = 256

    @T.prim_func
    def kernel(
        QP: T.Tensor((panels, tokens, H, DK), dtype),
        KP: T.Tensor((panels, tokens, H, DK), dtype),
        GP: T.Tensor((panels, tokens, H), "float32"),
        BP: T.Tensor((panels, tokens, H), "float32"),
        DQ: T.Tensor((tokens, H, DK), dtype),
        DKOut: T.Tensor((tokens, H, DK), dtype),
        DG: T.Tensor((tokens, H), "float32"),
        DB: T.Tensor((tokens, H), "float32"),
    ):
        with T.Kernel(T.ceildiv(tokens * H * DK, BLOCK), threads=128) as block:
            for j in T.Parallel(BLOCK):
                index = block * BLOCK + j
                if index < tokens * H * DK:
                    token, head, dim = index // (H * DK), index // DK % H, index % DK
                    if panels == 2:
                        DQ[token, head, dim] = T.cast(QP[0, token, head, dim], "float32") + T.cast(QP[1, token, head, dim], "float32")
                        DKOut[token, head, dim] = T.cast(KP[0, token, head, dim], "float32") + T.cast(KP[1, token, head, dim], "float32")
                    else:
                        DQ[token, head, dim] = QP[0, token, head, dim]
                        DKOut[token, head, dim] = KP[0, token, head, dim]
                if index < tokens * H:
                    token_s, head_s = index // H, index % H
                    if panels == 2:
                        DG[token_s, head_s] = GP[0, token_s, head_s] + GP[1, token_s, head_s]
                        DB[token_s, head_s] = BP[0, token_s, head_s] + BP[1, token_s, head_s]
                    else:
                        DG[token_s, head_s] = GP[0, token_s, head_s]
                        DB[token_s, head_s] = BP[0, token_s, head_s]

    return kernel


CUT_BOUND = 2.0**-24
# Threads per CTA of the fused kernels, and the packing width of their pair operands. 512 -> P = 2 column parts of
# the C x C / C x D accumulators; 1024 -> P = 4 (32 warps, <= 128 registers per thread).
PAIR_THREADS = 512
# The non-persistent fused kernel loads PAIR / PAIRQ, so its own CTA follows the packing thread count.
FUSED_THREADS = PAIR_THREADS
# Pair operands (E, KK, QK, QK^T) are stored packed in the fused kernels' C x C accumulator ownership:
# element (i, j) lives at [thread, k] with thread = j // (C // P) * 256 + 4 i + j % 4 and
# k = j % (C // P) // 4, i.e. P column parts of C // P columns each (see _pair_pack).
PAIR_P = PAIR_THREADS // 256
PAIR_PART_N = C // PAIR_P
# Pair operand arrays per element, masks / scale / decay folded in by prepare_next. PAIR (io dtype): 0 Em = [i>=j] s e,
# 1 Et = [j>=i] s e, 2 Al = -[i>j] e, 3 Au = -[j>i] e, 4 Al KK, 5 Au KK, 6 [i==j]; mirrored entries share one rounding,
# so row and column sums of dg still cancel exactly. QK^T is fp32 (PAIRQ): the row sum sum(P o QK) now comes from the
# final dQ row dot (exact Q K^T products), and its column counterpart sum(P^T o QK^T) must use the same products.
# Compact PAIR (default): only 0 E = decay exp(-|L_i - L_j|) and 1 KK (io dtype, i.e. bf16-rounded); the fused kernels
# rebuild masks and scale from (i, j), mirrored entries still share one value. 7 -> 2 arrays cuts PAIR traffic by 71 %.
PAIR_ARRAYS = 2
# Persistent cut-path fused kernel (one CTA per MP, next-item tiles prefetched with TME during compute).


def _mp_count():
    return _multiprocessors()


def _pair_pack(i, j):
    # PAIR_PART_N == C // P == C * 256 // PAIR_THREADS (32 at 512 threads, 16 at 1024).
    return j // PAIR_PART_N * 256 + i * 4 + j % 4, j % PAIR_PART_N // 4


def _is_symmetric_cut(cut, dk, dv):
    return cut and dk == dv == 128


def _pairq_shape(capacity, heads, symmetric_cut):
    # The half-K cut kernel reconstructs the pair contribution from dK.
    # Keep a placeholder to share the call signature with the other kernels.
    return (1,) if symmetric_cut else (capacity, heads, PAIR_THREADS, C * C // PAIR_THREADS)


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def fold_heads(HQ, rep, D, dtype):
    """Sum grouped value-head gradients back onto q/k heads: fp32 accumulation, 16 B vector loads/stores."""
    tokens = T.dynamic("tokens")
    VEC = 8
    LANES = D // VEC
    THREADS = 256
    ROWS = THREADS // LANES

    @T.prim_func
    def kernel(
        XQ: T.Tensor((tokens, HQ * rep, D), dtype),
        XK: T.Tensor((tokens, HQ * rep, D), dtype),
        YQ: T.Tensor((tokens, HQ, D), dtype),
        YK: T.Tensor((tokens, HQ, D), dtype),
    ):
        with T.Kernel(T.ceildiv(tokens, ROWS), HQ, threads=THREADS) as (block, head):
            tid = T.get_thread_binding()
            row = block * ROWS + tid // LANES
            col = tid % LANES * VEC
            sq = T.alloc_local((VEC,), "float32")
            sk = T.alloc_local((VEC,), "float32")
            vq = T.alloc_local((VEC,), dtype)
            vk = T.alloc_local((VEC,), dtype)
            if row < tokens:
                for e in T.vectorized(VEC):
                    sq[e] = 0
                    sk[e] = 0
                for r in T.unroll(rep):
                    for e in T.vectorized(VEC):
                        vq[e] = XQ[row, head * rep + r, col + e]
                        vk[e] = XK[row, head * rep + r, col + e]
                    for e in T.vectorized(VEC):
                        sq[e] += T.cast(vq[e], "float32")
                        sk[e] += T.cast(vk[e], "float32")
                for e in T.vectorized(VEC):
                    vq[e] = sq[e]
                    vk[e] = sk[e]
                for e in T.vectorized(VEC):
                    YQ[row, head, col + e] = vq[e]
                    YK[row, head, col + e] = vk[e]

    return kernel


def _multiprocessors():
    return torch.musa.get_device_properties(torch.musa.current_device()).multi_processor_count


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def prepare_next(H, DK, DV, dtype, cu_dtype, scale, mode="both"):
    """Forward-only (dO-independent) maps for the fused cut kernel.

    M_c = StateDZ_c SourceMap_{c+1}, RTt_c = (CP_c SourceMap_{c+1})^T (zero for last chunks), MZ_c = [SourceDZ_c | M_c],
    KQ_c = [K_c; Q_c] (zero-padded rows), and the packed pair coefficients PAIR (see PAIR_ARRAYS) built from
    KK_c = K_c K_c^T, QK_c = Q_c K_c^T and the decay e[i, j] = exp(L_i - L_j) for i >= j, exp(L_j - L_i) otherwise.

    mode="cut" / "scan" emit only what the persistent cut / scan-path fused kernel reads (cut: no CPT or M; scan: no
    SourceMap read, no M / MZ / RTt and their GEMMs); "both" keeps every output.
    """
    assert mode in ("both", "cut", "scan"), "unsupported backward preparation mode"
    maps = mode in ("both", "cut")
    do_cpt = mode in ("both", "scan")
    symmetric_cut = _is_symmetric_cut(mode == "cut", DK, DV)
    cp_rows = 64 if symmetric_cut else DV
    THREADS = PAIR_THREADS if symmetric_cut else 256
    pdt = dtype
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")
    pairq_shape = _pairq_shape(capacity, H, symmetric_cut)

    @T.prim_func
    def kernel(
        Q: T.Tensor((tokens, H, DK), dtype),
        K: T.Tensor((tokens, H, DK), dtype),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        CP: T.Tensor((capacity, H, DV, DK), dtype),
        StateDZ: T.Tensor((capacity, H, C, DK), dtype),
        SourceMap: T.Tensor((capacity, H, DK, C), dtype),
        M: T.Tensor((capacity, H, C, C), dtype),
        RTt: T.Tensor((capacity, H, C, DV), dtype),
        L: T.Tensor((capacity, H, C), "float32"),
        PAIR: T.Tensor((capacity, H, PAIR_THREADS, PAIR_ARRAYS, C * C // PAIR_THREADS), pdt),
        PAIRQ: T.Tensor(pairq_shape, "float32"),
        SourceDZ: T.Tensor((capacity, H, C, C), dtype),
        MZ: T.Tensor((capacity, H, C, 2 * C), dtype),
        KQ: T.Tensor((capacity, H, 2 * C, DK), dtype),
        B: T.Tensor((tokens, H), "float32"),
        Exps: T.Tensor((capacity, H, 2, C), "float32"),
        GATES: T.Tensor((capacity, H, 3, C), "float32"),
        CPT: T.Tensor((capacity, H, DK, DV), dtype),
    ):
        with T.Kernel(capacity, H, threads=THREADS) as (slot, head):
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
                sm = T.alloc_shared((DK, C), dtype) if maps else None
                dz = T.alloc_shared((C, DK), dtype) if maps else None
                cp = T.alloc_shared((cp_rows, DK), dtype)
                if not symmetric_cut:
                    qs = T.alloc_shared((C, DK), dtype)
                ks = T.alloc_shared((C, DK), dtype)
                mf = T.alloc_fragment((C, C), "float32")
                rf = T.alloc_fragment((cp_rows, C), "float32")
                kf = T.alloc_fragment((C, C), "float32")
                tx = T.get_thread_binding()
                NE = C * C // PAIR_THREADS
                dks, kks = T.alloc_shared((C, C), pdt), T.alloc_shared((C, C), pdt)
                lsh = T.alloc_shared((C,), "float32")
                if not symmetric_cut:
                    qf = T.alloc_fragment((C, C), "float32")
                    qfs = T.alloc_shared((C, C), "float32")
                for i in T.Parallel(C):
                    lsh[i] = L[slot, head, i]
                if maps:
                    if start + C < end:
                        T.copy(SourceMap[slot + 1, head, :, :], sm)
                    else:
                        for d, i in T.Parallel(DK, C):
                            sm[d, i] = 0
                    T.copy(StateDZ[slot, head, :, :], dz)
                if not symmetric_cut:
                    T.copy(CP[slot, head, :, :], cp)
                for i, d in T.Parallel(C, DK):
                    if not symmetric_cut:
                        qs[i, d] = T.if_then_else(start + i < end, Q[T.min(start + i, end - 1), head, d], 0)
                    ks[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                T.sync_threads()
                if maps:
                    T.gemm(dz, sm, mf, clear_accum=True)
                    if symmetric_cut:
                        # Stage one CP panel at a time to reduce shared memory per CTA.
                        for panel in T.serial(DV // cp_rows):
                            T.copy(CP[slot, head, panel * cp_rows:(panel + 1) * cp_rows, :], cp)
                            T.sync_threads()
                            T.gemm(cp, sm, rf, clear_accum=True)
                            for v, i in T.Parallel(cp_rows, C):
                                RTt[slot, head, i, panel * cp_rows + v] = rf[v, i]
                            T.sync_threads()
                    else:
                        T.gemm(cp, sm, rf, clear_accum=True)
                T.gemm(ks, ks, kf, transpose_B=True, clear_accum=True)
                if not symmetric_cut:
                    T.gemm(qs, ks, qf, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(C, C):
                    if mode == "both":
                        M[slot, head, i, j] = mf[i, j]
                    if maps:
                        MZ[slot, head, i, C + j] = mf[i, j]
                        MZ[slot, head, i, j] = SourceDZ[slot, head, i, j]
                    dks[i, j] = _exp(T.if_then_else(i >= j, lsh[i] - lsh[j], lsh[j] - lsh[i]))
                    kks[i, j] = kf[i, j]
                    if not symmetric_cut:
                        qfs[i, j] = qf[i, j]
                T.sync_threads()
                # Each owner writes its PAIR row as a vector. PAIRQ stores
                # the transposed coefficient for the same owner.
                for r in T.serial(PAIR_THREADS // THREADS):
                    o = r * THREADS + tx
                    pi = o % 256 // 4
                    pd, pk = T.alloc_local((NE,), pdt), T.alloc_local((NE,), pdt)
                    if not symmetric_cut:
                        pq = T.alloc_local((NE,), "float32")
                    for e in T.unroll(NE):
                        pj = o // 256 * PAIR_PART_N + e * 4 + o % 4
                        pd[e] = dks[pi, pj]
                        pk[e] = kks[pi, pj]
                        if not symmetric_cut:
                            pq[e] = qfs[pj, pi]
                    for e in T.vectorized(NE):
                        PAIR[slot, head, o, 0, e] = pd[e]
                    for e in T.vectorized(NE):
                        PAIR[slot, head, o, 1, e] = pk[e]
                    if not symmetric_cut:
                        for e in T.vectorized(NE):
                            PAIRQ[slot, head, o, e] = pq[e]
                if maps and not symmetric_cut:
                    for v, i in T.Parallel(DV, C):
                        RTt[slot, head, i, v] = rf[v, i]
                # TME-friendly whole tiles for the persistent cut kernel: [SourceDZ | M] and zero-padded [K; Q].
                for i, d in T.Parallel(C, DK):
                    KQ[slot, head, i, d] = ks[i, d]
                    if symmetric_cut:
                        KQ[slot, head, C + i, d] = T.if_then_else(
                            start + i < end, Q[T.min(start + i, end - 1), head, d], 0
                        )
                    else:
                        KQ[slot, head, C + i, d] = qs[i, d]
                # CP^T for the persistent scan kernel (it shares the K-major D layout there, see trace(CP D)).
                if do_cpt:
                    for r in T.serial(DK * DV // 8 // THREADS):
                        idx = r * THREADS + tx
                        cd, cvb = idx // (DV // 8), idx % (DV // 8)
                        ct = T.alloc_local((8,), dtype)
                        for e in T.unroll(8):
                            ct[e] = cp[cvb * 8 + e, cd]
                        for e in T.vectorized(8):
                            CPT[slot, head, cd, cvb * 8 + e] = ct[e]
                # Per-chunk gates [alpha; tail; beta (zero past the sequence end)] as one TME tile.
                for i in T.Parallel(C):
                    GATES[slot, head, 0, i] = Exps[slot, head, 0, i]
                    GATES[slot, head, 1, i] = Exps[slot, head, 1, i]
                    GATES[slot, head, 2, i] = T.if_then_else(start + i < end, B[T.min(start + i, end - 1), head], 0)

    return kernel


@dataclass
class ChunkedBackward:
    heads: int
    key_dim: int
    value_dim: int
    tiles_kernel: object
    maps_kernel: object
    drive_kernel: object
    scan_kernel: object
    state_kernel: object
    grad_kernel: object
    reduce_kernel: object
    cut_kernel: object = None
    initial_kernel: object = None
    next_kernel: object = None
    fused_kernel: object = None
    fused_ds_kernel: object = None
    scan_kmajor_kernel: object = None
    scan_fused_drive_kernel: object = None
    fused_persistent: bool = False
    fused_ds_persistent: bool = False
    scan_tme: bool = False
    scan_tme_single_kernel: object = None
    trace_kernel: object = None
    # Forward recompute that also emits Y / U (then the tiles kernel skips them).
    recompute_kernel: object = None
    tiles_lite_kernel: object = None
    forward_yu: tuple | None = None
    tiles: tuple | None = None
    maps: tuple | None = None
    next_maps: tuple | None = None
    cut: bool = False
    need_ds0: bool = True
    # prepare_next specialised to what the persistent cut / scan fused kernels read.
    next_cut_kernel: object = None
    next_scan_kernel: object = None

    @torch.no_grad()
    def recompute_forward(self, k, v, beta, cu, w, prefix, state, checkpoints):
        """Checkpoints S_c (and the Y / U tiles) from the factors; returns False when unavailable."""
        if self.recompute_kernel is None:
            return False
        capacity, h, dv = w.shape[0], self.heads, self.value_dim
        y, u = [torch.empty(capacity, h, C, dv, device=k.device, dtype=k.dtype) for _ in range(2)]
        cpt = torch.empty(capacity, h, self.key_dim, dv, device=k.device, dtype=k.dtype)
        self.recompute_kernel(k, v, beta, cu, w, prefix, state, checkpoints, y, u, cpt)
        self.forward_yu = y, u
        return True

    @torch.no_grad()
    def bind_forward(self, q, k, v, beta, cu, w, prefix, checkpoints, boundaries=None, need_ds0=True):
        capacity, device, dtype = w.shape[0], q.device, q.dtype
        h, dk, dv = self.heads, self.key_dim, self.value_dim
        lite = self.forward_yu is not None and self.tiles_lite_kernel is not None
        if lite:
            y, u = self.forward_yu
            self.forward_yu = None
        else:
            y, u = [torch.empty(capacity, h, C, dv, device=device, dtype=dtype) for _ in range(2)]
        wt, pt = [torch.empty_like(w) for _ in range(2)]
        qt, kt = [torch.empty(capacity, h, dk, C, device=device, dtype=dtype) for _ in range(2)]
        decay = torch.empty(capacity, h, C, C, device=device, dtype=torch.float32)
        exps = torch.empty(capacity, h, 2, C, device=device, dtype=torch.float32)
        self.tiles = y, u, wt, pt, qt, kt, exps, decay
        (self.tiles_lite_kernel if lite else self.tiles_kernel)(q, k, v, beta, cu, w, prefix, checkpoints, *self.tiles)
        state_map = torch.empty(capacity, h, dk, dk, device=device, dtype=dtype)
        source_map = torch.empty_like(qt)
        state_dz = torch.empty(capacity, h, C, dk, device=device, dtype=dtype)
        source_dz = torch.empty_like(w)
        self.maps = state_map, source_map, state_dz, source_dz
        self.maps_kernel(k, beta, cu, wt, pt, qt, kt, exps, *self.maps)
        self.need_ds0 = need_ds0
        self.cut = False
        if self.cut_kernel is not None and boundaries is not None:
            # D_{c-1} = (alpha_last I + StateMap_c) D_c + Source_c; drop the transfer only
            # when its Frobenius bound is below FP32 resolution for every chunk that feeds it.
            slots, offset = [], 0
            for begin, finish in zip(boundaries, boundaries[1:]):
                count = -(-(finish - begin) // C)
                slots += range(offset + (0 if need_ds0 else 1), offset + count)
                offset += count
            if slots:
                index = torch.tensor(slots, device=device)
                # Per-chunk StateMap Frobenius norms in one read of the bf16 maps (no fp32 copy of StateMap).
                norms = torch.empty(capacity, h, device=device, dtype=torch.float32)
                state_norms(h, dk, dtype)(state_map, norms)
                bound = norms[index] + exps[index, :, 0, C - 1].abs() * dk**0.5
                self.cut = bool(bound.max() <= CUT_BOUND)
            else:
                self.cut = True
        self.next_maps = None
        if (self.cut and self.fused_kernel is not None) or (not self.cut and self.fused_ds_kernel is not None):
            recomputed = lite
            mmap = torch.empty_like(w)
            rtt = torch.empty(capacity, h, C, dv, device=device, dtype=dtype)
            mzc = torch.empty(capacity, h, C, 2 * C, device=device, dtype=dtype)
            cpt = torch.empty(capacity, h, dk, dv, device=device, dtype=dtype)
            pair = torch.empty(capacity, h, PAIR_THREADS, PAIR_ARRAYS, C * C // PAIR_THREADS, device=device,
                               dtype=dtype)
            pairq = torch.empty(
                _pairq_shape(capacity, h, _is_symmetric_cut(self.cut, dk, dv)), device=device, dtype=torch.float32
            )
            kq = torch.empty(capacity, h, 2 * C, dk, device=device, dtype=dtype)
            gates = torch.empty(capacity, h, 3, C, device=device, dtype=torch.float32)
            nxt = self.next_kernel
            if self.cut and self.fused_persistent and self.next_cut_kernel is not None:
                nxt = self.next_cut_kernel
            elif not self.cut and self.fused_ds_persistent:
                if self.next_scan_kernel is not None:
                    nxt = self.next_scan_kernel
            nxt(q, k, cu, checkpoints, state_dz, source_map, mmap, rtt, prefix, pair, pairq, source_dz, mzc, kq,
                beta, exps, gates, cpt)
            self.next_maps = mmap, rtt, pair, pairq, mzc, kq, gates, cpt
            # Per-warp partials of sum(DS o CPT) from state_trace (persistent scan path; a placeholder otherwise).
            self.trp = torch.empty(capacity, h, 32, device=device, dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, q, k, v, beta, do, cu, w, prefix, checkpoints, dfs, dq, dk, dv, dg, db, ds0):
        if self.tiles is None or self.maps is None:
            raise RuntimeError("forward-dependent tiles must be prepared before chunked backward")
        y, u, wt, pt, qt, kt, exps, decay = self.tiles
        state_map, source_map, state_dz, source_dz = self.maps
        capacity, total, h, key_dim, value_dim = w.shape[0], q.shape[0], self.heads, self.key_dim, self.value_dim
        panels = value_dim // BV
        device = q.device
        if self.cut and self.next_maps is not None:
            mmap, rtt, pair, pairq, mzc, kq, gates, _ = self.next_maps
            if self.fused_persistent:
                # DS <- checkpoints is a shape-compatible placeholder on the cut path.
                self.fused_kernel(q, k, do, cu, checkpoints, u, y, gates, mzc, rtt, source_map, source_dz, state_dz, checkpoints,
                                  kq, pair, pairq, self.trp, dq, dk, dv, dg, db)
                if self.need_ds0:
                    self.initial_kernel(do, cu, source_map, dfs, ds0)
                return
            # Unused operands of the shared signature get shape-compatible placeholders (DS <- checkpoints).
            self.fused_kernel(q, k, do, beta, cu, checkpoints, u, y, exps, prefix, source_dz, state_dz, mmap, rtt, source_map,
                              checkpoints, pair, pairq, dq, dk, dv, dg, db)
            if self.need_ds0:
                self.initial_kernel(do, cu, source_map, dfs, ds0)
            return
        sr = torch.empty(panels, capacity, h, device=device, dtype=torch.float32)
        bp = torch.empty(panels, total, h, device=device, dtype=torch.float32)
        ds = torch.empty(capacity, h, value_dim, key_dim, device=device, dtype=q.dtype)
        z_chunks = torch.empty_like(u)
        if self.cut:
            self.cut_kernel(do, beta, cu, checkpoints, y, exps, state_dz, source_map, source_dz, dfs, ds, z_chunks, dv, sr, bp)
            if self.need_ds0:
                self.initial_kernel(do, cu, source_map, dfs, ds0)
        else:
            kmajor = self.next_maps is not None and self.scan_kmajor_kernel is not None
            # Folding the Source GEMM into the serial scan pays off while scan CTAs (sequences x heads x panels) sit
            # at most two per SM; with more co-resident scan CTAs the separate, fully parallel drive kernel wins.
            scan_ctas = (cu.shape[0] - 1) * h * (value_dim // SCAN_BV)
            # The TME-pipelined scan (state_scan_tme) wins even with more than two scan CTAs per SM.
            fused_drive = kmajor and self.scan_fused_drive_kernel is not None and (
                self.scan_tme or scan_ctas <= 2 * _multiprocessors())
            source = None if fused_drive else torch.empty(capacity, h, key_dim, value_dim, device=device, dtype=q.dtype)
            if kmajor:
                ds = torch.empty(capacity, h, key_dim, value_dim, device=device, dtype=q.dtype)
                if fused_drive:
                    # Source is formed inside the scan; `ds` is only a shape-compatible placeholder (never read).
                    scan = self.scan_fused_drive_kernel
                    if self.scan_tme_single_kernel is not None and scan_ctas > 2 * _multiprocessors():
                        scan = self.scan_tme_single_kernel  # single StateMap buffer: three CTAs per SM
                    scan(cu, state_map, exps, ds, do, source_map, dfs, ds, ds0)
                else:
                    self.drive_kernel(do, cu, source_map, source)
                    self.scan_kmajor_kernel(cu, state_map, exps, source, do, source_map, dfs, ds, ds0)
            else:
                self.drive_kernel(do, cu, source_map, source)
                self.scan_kernel(cu, state_map, exps, source, do, source_map, dfs, ds, ds0)
            if self.next_maps is not None:
                mmap, rtt, pair, pairq, mzc, kq, gates, cpt = self.next_maps
                if self.fused_ds_persistent:
                    if self.trace_kernel is not None:
                        self.trace_kernel(ds, cpt, self.trp)
                    self.fused_ds_kernel(q, k, do, cu, cpt, u, y, gates, mzc, rtt, source_map, source_dz, state_dz, ds,
                                         kq, pair, pairq, self.trp, dq, dk, dv, dg, db)
                    return
                # Placeholders: M, RTt, SourceMap are unused on the scan path.
                self.fused_ds_kernel(q, k, do, beta, cu, checkpoints, u, y, exps, prefix, source_dz, state_dz, mmap, rtt, source_map,
                                     ds, pair, pairq, dq, dk, dv, dg, db)
                return
            self.state_kernel(do, beta, cu, checkpoints, y, exps, state_dz, source_map, source_dz, dfs, ds, z_chunks, dv, sr, bp)
        self.grad_kernel(q, k, do, beta, cu, checkpoints, u, exps, prefix, z_chunks, ds, sr, bp, dq, dk, dg, db)

    def get_kernel_source(self):
        kernels = (
            self.drive_kernel,
            self.scan_kernel,
            self.state_kernel,
            self.cut_kernel,
            self.initial_kernel,
            self.grad_kernel,
            self.reduce_kernel,
            self.next_kernel,
            self.fused_kernel,
            self.fused_ds_kernel,
            self.scan_kmajor_kernel,
            self.scan_fused_drive_kernel,
        )
        return "\n".join(kernel.get_kernel_source() for kernel in kernels if kernel is not None)


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=BASE_COMPILE_FLAGS)
def state_norms(H, DK, dtype):
    """Frobenius norm of each (DK, DK) StateMap: each thread sums the squares of one row in registers."""
    capacity = T.dynamic("capacity")
    rowl = T.Fragment((DK, DK), forward_thread_fn=lambda d, j: d, forward_index_fn=lambda d, j: j)

    @T.prim_func
    def norms_kernel(StateMap: T.Tensor((capacity, H, DK, DK), dtype), Norm: T.Tensor((capacity, H), "float32")):
        with T.Kernel(capacity, H, threads=DK) as (slot, head):
            sq = T.alloc_fragment((DK, DK), "float32")
            rows = T.alloc_fragment((DK,), "float32")
            red = T.alloc_shared((DK,), "float32")
            T.annotate_layout({sq: rowl})
            for d, j in T.Parallel(DK, DK):
                value = T.cast(StateMap[slot, head, d, j], "float32")
                sq[d, j] = value * value
            T.reduce_sum(sq, rows, dim=1)
            for d in T.Parallel(DK):
                red[d] = rows[d]
            T.sync_threads()
            if T.get_thread_binding() == 0:
                total = T.alloc_local((1,), "float32")
                total[0] = 0
                for d in T.serial(DK):
                    total[0] += red[d]
                Norm[slot, head] = T.sqrt(total[0])

    return norms_kernel


def backward(H, DK, DV, dtype, cu_dtype, has_final, scale, has_initial=False, single_sequence=False):
    from .gdn_bwd_projection import state_projection, chunk_grad, chunk_fused, chunk_fused_ds, chunk_fused_persistent

    # The fused cut path folds D_c = SourceMap_{c+1} dO_{c+1} into prepared maps; a final-state
    # gradient adds a non-factorable D for last chunks, so that case keeps the two-kernel cut path.
    fused = fused_ds = None
    persistent = DV == 2 * C
    if not has_final:
        if persistent:
            fused = chunk_fused_persistent(
                H, DK, DV, dtype, cu_dtype, scale, _mp_count(), single_sequence=single_sequence and DK == DV
            )
        else:
            fused = chunk_fused(H, DK, DV, dtype, cu_dtype, scale, threads=FUSED_THREADS)
    # The scan path (no cut, or a final-state gradient without cut) reads D_c from the scan.
    # The persistent scan kernel stages CP^T in the (DV, DK) CP buffer, so it needs DK == DV.
    if persistent and DK == DV:
        fused_ds = chunk_fused_persistent(H, DK, DV, dtype, cu_dtype, scale, _mp_count(), from_ds=True)
    else:
        fused_ds = chunk_fused_ds(H, DK, DV, dtype, cu_dtype, scale, threads=FUSED_THREADS)
    kmajor = fused_ds is not None
    fuse_drive = kmajor
    plan = ChunkedBackward(
        H,
        DK,
        DV,
        prepare_tiles(H, DK, DV, dtype, cu_dtype, scale),
        prepare_maps(H, DK, dtype, cu_dtype),
        source_grad(H, DK, DV, dtype, cu_dtype),
        state_scan(H, DK, DV, dtype, cu_dtype, has_final),
        state_projection(H, DK, DV, dtype, cu_dtype, has_final, False),
        chunk_grad(H, DK, DV, dtype, cu_dtype, scale),
        None,
        state_projection(H, DK, DV, dtype, cu_dtype, has_final, True),
        initial_source(H, DK, DV, dtype, cu_dtype, has_final),
        None if fused is None and fused_ds is None else prepare_next(H, DK, DV, dtype, cu_dtype, scale),
        fused,
        fused_ds,
        state_scan(H, DK, DV, dtype, cu_dtype, has_final, kmajor=True) if kmajor else None,
        state_scan_tme(H, DK, DV, dtype, cu_dtype, has_final) if fuse_drive else None,
        persistent,
        persistent and DK == DV,
        fuse_drive,
        state_scan_tme(H, DK, DV, dtype, cu_dtype, has_final, stages=1) if fuse_drive else None,
        state_trace(H, DK, DV, dtype) if fuse_drive and persistent and DK == DV else None,
        recompute_tme(H, DK, DV, dtype, cu_dtype, has_initial),
        prepare_tiles(H, DK, DV, dtype, cu_dtype, scale, with_yu=False),
    )
    if persistent:
        plan.next_cut_kernel = prepare_next(H, DK, DV, dtype, cu_dtype, scale, mode="cut")
        if DK == DV:
            plan.next_scan_kernel = prepare_next(H, DK, DV, dtype, cu_dtype, scale, mode="scan")
    return plan
