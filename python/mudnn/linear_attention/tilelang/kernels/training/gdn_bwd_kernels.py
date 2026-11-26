# SPDX-License-Identifier: Apache-2.0
# Algorithm adapted from NVIDIA/cudnn-frontend 33c6fcb54d110f2c4fbeaf7c0330067492142189.
# Upstream: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# This translation replaces CUDA/CuTe/TMEM scheduling with MUSA TileLang kernels.

"""Chunk-64 WY inverse-adjoint factors and checkpoint recompute."""

import tilelang
import tilelang.language as T

CONFIG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
}
BWD_CONFIG = {
    **CONFIG,
    # Shared phases are explicitly fenced, including reduction-result staging.
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}
FLAGS = ["-Od3", "-fno-signed-zeros"]
C = 64
LOG2E = 1.4426950408889634


def _exp(value):
    return T.exp2(value * LOG2E)


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=FLAGS)
def factors(H, DK, dtype, cu_dtype):
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")

    @T.prim_func
    def kernel(
        K: T.Tensor((tokens, H, DK), dtype),
        G: T.Tensor((tokens, H), "float32"),
        B: T.Tensor((tokens, H), "float32"),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        W: T.Tensor((capacity, H, C, C), dtype),
        L: T.Tensor((capacity, H, C), "float32"),
    ):
        with T.Kernel(capacity, H, threads=128) as (slot, head):
            seq = T.alloc_var("int32")
            ordinal = T.alloc_var("int32")
            offset = T.alloc_var("int32")
            seq = -1
            ordinal = 0
            offset = 0
            for n in T.serial(seqs):
                count = T.ceildiv(CU[n + 1] - CU[n], C)
                if slot >= offset and slot < offset + count:
                    seq = n
                    ordinal = slot - offset
                offset += count
            if seq >= 0:
                start = CU[seq] + ordinal * C
                end = CU[seq + 1]
                ks = T.alloc_shared((C, DK), dtype)
                gs = T.alloc_shared((C,), "float32")
                bs = T.alloc_shared((C,), "float32")
                ls = T.alloc_shared((C,), "float32")
                kk = T.alloc_fragment((C, C), "float32")
                diag = T.alloc_shared((2, 32, 32), "float32")
                inv0 = T.alloc_shared((32, 32), dtype)
                inv1 = T.alloc_shared((32, 32), dtype)
                cross = T.alloc_shared((32, 32), dtype)
                temporary = T.alloc_shared((32, 32), dtype)
                acc = T.alloc_fragment((32, 32), "float32")
                for i, d in T.Parallel(C, DK):
                    ks[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                for i in T.Parallel(C):
                    gs[i] = T.if_then_else(start + i < end, G[T.min(start + i, end - 1), head], 0)
                    bs[i] = T.if_then_else(start + i < end, B[T.min(start + i, end - 1), head], 0)
                T.sync_threads()
                for i in T.Parallel(C):
                    prefix = T.alloc_local((1,), "float32")
                    prefix[0] = 0
                    for j in T.serial(C):
                        if j <= i:
                            prefix[0] += gs[j]
                    ls[i] = prefix[0]
                    L[slot, head, i] = prefix[0]
                T.sync_threads()
                T.gemm(ks, ks, kk, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(C, C):
                    kk[i, j] = T.if_then_else(i > j, kk[i, j] * T.exp(ls[i] - ls[j]) * bs[i], 0)
                    if i // 32 == j // 32:
                        diag[i // 32, i % 32, j % 32] = T.if_then_else(i == j, 1, -kk[i, j])
                T.sync_threads()
                tx = T.get_thread_binding()
                if tx < 64:
                    block, row = tx // 32, tx % 32
                    values = T.alloc_local((32,), "float32")
                    for j in T.unroll(32):
                        values[j] = diag[block, row, j]
                    for pivot in T.unroll(31):
                        multiplier = T.if_then_else(row > pivot, values[pivot], 0)
                        for j in T.unroll(pivot):
                            source = T.shfl_sync(0xFFFFFFFF, values[j], pivot, 32)
                            values[j] += multiplier * source
                    for j in T.unroll(32):
                        if block == 0:
                            inv0[row, j] = values[j]
                        else:
                            inv1[row, j] = values[j]
                for i, j in T.Parallel(32, 32):
                    cross[i, j] = -kk[i + 32, j]
                T.sync_threads()
                T.gemm(inv1, cross, acc, clear_accum=True)
                T.copy(acc, temporary)
                T.sync_threads()
                T.gemm(temporary, inv0, acc, clear_accum=True)
                for i, j in T.Parallel(32, 32):
                    W[slot, head, i, j] = inv0[i, j]
                    W[slot, head, i, j + 32] = 0
                    W[slot, head, i + 32, j] = acc[i, j]
                    W[slot, head, i + 32, j + 32] = inv1[i, j]

    return kernel


@tilelang.jit(target="musa", pass_configs=CONFIG, compile_flags=FLAGS)
def recompute(H, DK, DV, dtype, cu_dtype, has_initial):
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")

    @T.prim_func
    def kernel(
        K: T.Tensor((tokens, H, DK), dtype),
        V: T.Tensor((tokens, H, DV), dtype),
        B: T.Tensor((tokens, H), "float32"),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        W: T.Tensor((capacity, H, C, C), dtype),
        L: T.Tensor((capacity, H, C), "float32"),
        S0: T.Tensor((seqs, H, DV, DK), "float32"),
        Checkpoints: T.Tensor((capacity, H, DV, DK), dtype),
    ):
        with T.Kernel(seqs, H, threads=256) as (seq, head):
            offset = T.alloc_var("int32")
            offset = 0
            for n in T.serial(seq):
                offset += T.ceildiv(CU[n + 1] - CU[n], C)
            count = T.ceildiv(CU[seq + 1] - CU[seq], C)
            state = T.alloc_fragment((DK, DV), "float32")
            state_s = T.alloc_shared((DK, DV), dtype)
            ks = T.alloc_shared((C, DK), dtype)
            kt = T.alloc_shared((DK, C), dtype)
            rhs = T.alloc_shared((C, DV), dtype)
            ws = T.alloc_shared((C, C), dtype)
            u = T.alloc_shared((C, DV), dtype)
            cv = T.alloc_fragment((C, DV), "float32")
            update = T.alloc_fragment((DK, DV), "float32")
            ls = T.alloc_shared((C,), "float32")
            for d, v in T.Parallel(DK, DV):
                state[d, v] = S0[seq, head, v, d] if has_initial else 0
            for chunk in T.serial(count):
                start, end = CU[seq] + chunk * C, CU[seq + 1]
                slot = offset + chunk
                for d, v in T.Parallel(DK, DV):
                    Checkpoints[slot, head, v, d] = state[d, v]
                    state_s[d, v] = state[d, v]
                for i, d in T.Parallel(C, DK):
                    ks[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                for i, j in T.Parallel(C, C):
                    ws[i, j] = W[slot, head, i, j]
                for i in T.Parallel(C):
                    ls[i] = L[slot, head, i]
                T.sync_threads()
                T.gemm(ks, state_s, cv, clear_accum=True)
                for i, v in T.Parallel(C, DV):
                    value = T.if_then_else(start + i < end, V[T.min(start + i, end - 1), head, v], 0)
                    beta = T.if_then_else(start + i < end, B[T.min(start + i, end - 1), head], 0)
                    rhs[i, v] = beta * (value - T.exp(ls[i]) * cv[i, v])
                T.sync_threads()
                T.gemm(ws, rhs, cv, clear_accum=True)
                for i, v in T.Parallel(C, DV):
                    u[i, v] = cv[i, v] * T.exp(ls[C - 1] - ls[i])
                for d, i in T.Parallel(DK, C):
                    kt[d, i] = ks[i, d]
                T.sync_threads()
                T.gemm(kt, u, update, clear_accum=True)
                for d, v in T.Parallel(DK, DV):
                    state[d, v] = T.exp(ls[C - 1]) * state[d, v] + update[d, v]
                T.sync_threads()

    return kernel


@tilelang.jit(target="musa", pass_configs=BWD_CONFIG, compile_flags=FLAGS)
def backward(H, DK, DV, dtype, cu_dtype, has_final, scale):
    tokens, seqs, capacity = T.dynamic("tokens"), T.dynamic("seqs"), T.dynamic("capacity")

    @T.prim_func
    def kernel(
        Q: T.Tensor((tokens, H, DK), dtype),
        K: T.Tensor((tokens, H, DK), dtype),
        V: T.Tensor((tokens, H, DV), dtype),
        B: T.Tensor((tokens, H), "float32"),
        DO: T.Tensor((tokens, H, DV), dtype),
        CU: T.Tensor((seqs + 1,), cu_dtype),
        W: T.Tensor((capacity, H, C, C), dtype),
        L: T.Tensor((capacity, H, C), "float32"),
        Checkpoints: T.Tensor((capacity, H, DV, DK), dtype),
        DSFinal: T.Tensor((seqs, H, DV, DK), "float32"),
        DQ: T.Tensor((tokens, H, DK), dtype),
        DKOut: T.Tensor((tokens, H, DK), dtype),
        DVOut: T.Tensor((tokens, H, DV), dtype),
        DG: T.Tensor((tokens, H), "float32"),
        DB: T.Tensor((tokens, H), "float32"),
        DS0: T.Tensor((seqs, H, DV, DK), "float32"),
    ):
        with T.Kernel(seqs, H, threads=128) as (seq, head):
            offset = T.alloc_var("int32")
            offset = 0
            for n in T.serial(seq):
                offset += T.ceildiv(CU[n + 1] - CU[n], C)
            count = T.ceildiv(CU[seq + 1] - CU[seq], C)
            qs = T.alloc_shared((C, DK), dtype)
            ks = T.alloc_shared((C, DK), dtype)
            dos = T.alloc_shared((C, DV), dtype)
            dot = T.alloc_shared((DV, C), dtype)
            state_s = T.alloc_shared((DK, DV), dtype)
            state_t = state_s if DK == DV else T.alloc_shared((DV, DK), dtype)
            ds_s = T.alloc_shared((DK, DV), dtype)
            ds_t = ds_s if DK == DV else T.alloc_shared((DV, DK), dtype)
            y = T.alloc_shared((C, DV), dtype)
            u = T.alloc_shared((C, DV), dtype)
            z = T.alloc_shared((C, DV), dtype)
            zt = T.alloc_shared((DV, C), dtype)
            wm = T.alloc_shared((C, C), dtype)
            pm = T.alloc_shared((C, C), dtype)
            mt = T.alloc_shared((C, C), dtype)
            qt = T.alloc_shared((DK, C), dtype)
            kt = T.alloc_shared((DK, C), dtype)
            ls = T.alloc_shared((C,), "float32")
            bs = T.alloc_shared((C,), "float32")
            dl = T.alloc_shared((C,), "float32")
            db = T.alloc_shared((C,), "float32")
            state_grad = T.alloc_fragment((DK, DV), "float32")
            state_update = T.alloc_fragment((DK, DV), "float32")
            cv = T.alloc_fragment((C, DV), "float32")
            ck = T.alloc_fragment((C, DK), "float32")
            dk_acc = T.alloc_fragment((C, DK), "float32")
            kk = T.alloc_fragment((C, C), "float32")
            cc = T.alloc_fragment((C, C), "float32")
            product = T.alloc_fragment((C, C), "float32")
            product_k = T.alloc_fragment((C, DK), "float32")
            product_v = T.alloc_fragment((C, DV), "float32")
            rows = T.alloc_fragment((C,), "float32")
            cols = T.alloc_fragment((C,), "float32")
            rows_k = T.alloc_fragment((C,), "float32")
            rows_v = T.alloc_fragment((C,), "float32")
            row_s = T.alloc_shared((C,), "float32")
            col_s = T.alloc_shared((C,), "float32")
            state_rows = T.alloc_fragment((DK,), "float32")
            state_row_s = T.alloc_shared((DK,), "float32")
            for d, v in T.Parallel(DK, DV):
                state_grad[d, v] = DSFinal[seq, head, v, d] if has_final else 0
            for reverse in T.serial(count):
                chunk = count - 1 - reverse
                slot = offset + chunk
                start, end = CU[seq] + chunk * C, CU[seq + 1]
                for i, d in T.Parallel(C, DK):
                    qs[i, d] = T.if_then_else(start + i < end, Q[T.min(start + i, end - 1), head, d], 0)
                    ks[i, d] = T.if_then_else(start + i < end, K[T.min(start + i, end - 1), head, d], 0)
                for i, v in T.Parallel(C, DV):
                    dos[i, v] = T.if_then_else(start + i < end, DO[T.min(start + i, end - 1), head, v], 0)
                for d, v in T.Parallel(DK, DV):
                    checkpoint = Checkpoints[slot, head, v, d]
                    state_s[d, v] = checkpoint
                    state_update[d, v] = state_grad[d, v] * checkpoint
                T.reduce_sum(state_update, state_rows, dim=1)
                T.copy(state_rows, state_row_s)
                for i in T.Parallel(C):
                    ls[i] = L[slot, head, i]
                    bs[i] = T.if_then_else(start + i < end, B[T.min(start + i, end - 1), head], 0)
                    dl[i] = 0
                    db[i] = 0
                for i, j in T.Parallel(C, C):
                    wm[i, j] = W[slot, head, i, j]
                T.sync_threads()
                if T.get_thread_binding() == 0:
                    state_sum = T.alloc_local((1,), "float32")
                    state_sum[0] = 0
                    for d in T.serial(DK):
                        state_sum[0] += state_row_s[d]
                    dl[C - 1] = state_sum[0] * _exp(ls[C - 1])
                T.sync_threads()

                T.gemm(ks, state_s, cv, clear_accum=True)
                for i, v in T.Parallel(C, DV):
                    value = T.if_then_else(start + i < end, V[T.min(start + i, end - 1), head, v], 0)
                    residual = T.alloc_local((1,), dtype)
                    residual[0] = value - _exp(ls[i]) * cv[i, v]
                    y[i, v] = residual[0]
                    zt[v, i] = bs[i] * residual[0]
                T.sync_threads()
                T.gemm(wm, zt, cv, transpose_B=True, clear_accum=True)
                T.copy(cv, u)
                T.gemm(qs, ks, cc, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(C, C):
                    pm[i, j] = T.if_then_else(i >= j, scale * _exp(ls[i] - ls[j]) * cc[i, j], 0)
                for d, v in T.Parallel(DK, DV):
                    ds_s[d, v] = state_grad[d, v]
                T.sync_threads()
                T.gemm(ks, ds_s, cv, clear_accum=True)
                for i, v in T.Parallel(C, DV):
                    cv[i, v] *= _exp(ls[C - 1] - ls[i])
                for i, j in T.Parallel(C, C):
                    mt[i, j] = pm[j, i]
                for v, i in T.Parallel(DV, C):
                    dot[v, i] = dos[i, v]
                T.sync_threads()
                T.gemm(mt, dot, cv, transpose_B=True)
                for v, i in T.Parallel(DV, C):
                    zt[v, i] = cv[i, v]
                for i, j in T.Parallel(C, C):
                    mt[i, j] = wm[j, i]
                T.sync_threads()
                T.gemm(mt, zt, cv, transpose_B=True, clear_accum=True)
                for i, v in T.Parallel(C, DV):
                    z[i, v] = cv[i, v]
                    product_v[i, v] = cv[i, v] * y[i, v]
                T.reduce_sum(product_v, rows_v, dim=1)
                for i in T.Parallel(C):
                    db[i] = rows_v[i]
                T.sync_threads()
                T.gemm(z, u, cc, transpose_B=True, clear_accum=True)
                T.gemm(ks, ks, kk, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(C, C):
                    product[i, j] = T.if_then_else(i > j, -cc[i, j] * _exp(ls[i] - ls[j]) * kk[i, j], 0)
                    wm[i, j] = T.if_then_else(i > j, -bs[i] * cc[i, j] * _exp(ls[i] - ls[j]), 0)
                T.reduce_sum(product, rows, dim=1)
                for i in T.Parallel(C):
                    db[i] += rows[i]
                for i, j in T.Parallel(C, C):
                    product[i, j] *= bs[i]
                T.reduce_sum(product, rows, dim=1)
                T.reduce_sum(product, cols, dim=0)
                T.copy(rows, row_s)
                T.copy(cols, col_s)
                T.sync_threads()
                for i in T.Parallel(C):
                    dl[i] += row_s[i] - col_s[i]
                for i, v in T.Parallel(C, DV):
                    gradient = bs[i] * z[i, v]
                    z[i, v] = gradient
                    zt[v, i] = gradient
                    if start + i < end:
                        DVOut[start + i, head, v] = gradient
                T.sync_threads()
                for v, d in T.Parallel(DV, DK):
                    state_t[v, d] = Checkpoints[slot, head, v, d]

                T.gemm(dos, u, cc, transpose_B=True, clear_accum=True)
                T.gemm(qs, ks, kk, transpose_B=True, clear_accum=True)
                for i, j in T.Parallel(C, C):
                    pair_gradient = T.alloc_local((1,), dtype)
                    pair_gradient[0] = T.if_then_else(i >= j, scale * cc[i, j] * _exp(ls[i] - ls[j]), 0)
                    pm[i, j] = pair_gradient[0]
                    product[i, j] = pair_gradient[0] * kk[i, j]
                T.reduce_sum(product, rows, dim=1)
                T.reduce_sum(product, cols, dim=0)
                T.copy(rows, row_s)
                T.copy(cols, col_s)
                T.sync_threads()
                for i in T.Parallel(C):
                    dl[i] += row_s[i] - col_s[i]
                T.sync_threads()
                T.gemm(dos, state_t, ck, clear_accum=True)
                for i, d in T.Parallel(C, DK):
                    ck[i, d] *= scale * _exp(ls[i])
                    product_k[i, d] = ck[i, d] * qs[i, d]
                T.reduce_sum(product_k, rows_k, dim=1)
                for i in T.Parallel(C):
                    dl[i] += rows_k[i]
                for d, i in T.Parallel(DK, C):
                    kt[d, i] = ks[i, d]
                T.sync_threads()
                T.gemm(pm, kt, ck, transpose_B=True)
                for i, d in T.Parallel(C, DK):
                    if start + i < end:
                        DQ[start + i, head, d] = ck[i, d]

                T.gemm(z, state_t, dk_acc, clear_accum=True)
                for i, d in T.Parallel(C, DK):
                    dk_acc[i, d] *= -_exp(ls[i])
                    product_k[i, d] = dk_acc[i, d] * ks[i, d]
                T.reduce_sum(product_k, rows_k, dim=1)
                for i in T.Parallel(C):
                    dl[i] += rows_k[i]
                for v, d in T.Parallel(DV, DK):
                    ds_t[v, d] = state_grad[d, v]
                T.sync_threads()
                T.gemm(u, ds_t, ck, clear_accum=True)
                for i, d in T.Parallel(C, DK):
                    ck[i, d] *= _exp(ls[C - 1] - ls[i])
                    product_k[i, d] = ck[i, d] * ks[i, d]
                    dk_acc[i, d] += ck[i, d]
                T.reduce_sum(product_k, rows_k, dim=1)
                T.copy(rows_k, row_s)
                for i in T.Parallel(C):
                    dl[i] -= rows_k[i]
                T.sync_threads()
                if T.get_thread_binding() == 0:
                    gate_sum = T.alloc_local((1,), "float32")
                    gate_sum[0] = 0
                    for i in T.serial(C):
                        gate_sum[0] += row_s[i]
                    dl[C - 1] += gate_sum[0]
                for i, j in T.Parallel(C, C):
                    mt[i, j] = pm[j, i]
                for d, i in T.Parallel(DK, C):
                    qt[d, i] = qs[i, d]
                T.sync_threads()
                T.gemm(mt, qt, dk_acc, transpose_B=True)
                T.gemm(wm, kt, dk_acc, transpose_B=True)
                for i, j in T.Parallel(C, C):
                    mt[i, j] = wm[j, i]
                T.sync_threads()
                T.gemm(mt, kt, dk_acc, transpose_B=True)
                for i, d in T.Parallel(C, DK):
                    if start + i < end:
                        DKOut[start + i, head, d] = dk_acc[i, d]

                T.sync_threads()
                for i in T.Parallel(C):
                    suffix = T.alloc_local((1,), "float32")
                    suffix[0] = 0
                    for j in T.serial(C):
                        if j >= i:
                            suffix[0] += dl[j]
                    if start + i < end:
                        DG[start + i, head] = suffix[0]
                        DB[start + i, head] = db[i]
                for i, d in T.Parallel(C, DK):
                    qt[d, i] = qs[i, d] * scale * _exp(ls[i])
                    kt[d, i] = ks[i, d] * -_exp(ls[i])
                for v, i in T.Parallel(DV, C):
                    dot[v, i] = dos[i, v]
                T.sync_threads()
                T.gemm(qt, dot, state_update, transpose_B=True, clear_accum=True)
                T.gemm(kt, zt, state_update, transpose_B=True)
                for d, v in T.Parallel(DK, DV):
                    state_grad[d, v] = _exp(ls[C - 1]) * state_grad[d, v] + state_update[d, v]
                T.sync_threads()
            for d, v in T.Parallel(DK, DV):
                DS0[seq, head, v, d] = state_grad[d, v]

    return kernel
