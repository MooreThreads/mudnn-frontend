# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import torch
import tilelang
from tilelang import language as T

from ..utils.runtime import MP31_ARCH, jit_for_arch
from .dsa_bwd_common import preprocess as _preprocess

# Share the verified row reduction while compiling it for this architecture.
preprocess = jit_for_arch(_preprocess, MP31_ARCH)

_CONFIG = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: False,
}
_TARGET = {"kind": "musa", "arch": "mp_31"}


# Gradient compile options follow the original FA BWD (_flash_attention_bwd_common.JIT_COMPILE_FLAGS).
_GRADIENT_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm", "-misched=mtgpu-max-ilp",
    "-mllvm", "-mtgpu-tiny-offset-hint=1",
    "-mllvm", "-misched-recompute-slotindex=1",
    "-mllvm", "-mtgpu-if-convert=1",
    "-mllvm", "-mtgpu-combine-instr-with-burst=1",
    "-mllvm", "-mtgpu-combine-fop-instr=1",
]
# Safe memory access is off: INDEX columns are padded to 64 and tiles stop at ceildiv(length, block),
# KV rows are clamped, dKV is written only for valid slots, and Q/dO/LSE/Delta/dQ stay in shape.
_GRADIENT_CONFIG = {
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}
_LOG2E = 1.4426950408889634


@tilelang.jit(target=_TARGET, out_idx=[], pass_configs={**_CONFIG, **_GRADIENT_CONFIG}, compile_flags=_GRADIENT_FLAGS)
def gradient(heads, topk, block=32, reuse_kv=True):
    """64 heads per CTA, two 256-thread roles split dQ/dKV columns; KV tiles are double-buffered and prefetched."""

    sq = T.dynamic("sq")
    skv = T.dynamic("skv")
    hg = 64
    columns = 32

    @T.macro
    def annotate(buf, major):
        T.annotate_layout({buf: tilelang.layout.make_sqmma_swizzled_layout(buf, k_major=major)},
                          allow_reannotation=True)

    @T.macro
    def dq_panel(KV, kvk, ids, valid, kvm, ds_a, dq, c, skv, slot):
        annotate(ds_a, True)
        annotate(kvm, False)
        for i, d in T.Parallel(block, 64):
            if reuse_kv:
                kvm[0, i, d] = kvk[slot, c // 2, i, (c % 2) * 64 + d]
            else:
                kvm[0, i, d] = T.if_then_else(valid[slot, i] != 0, KV[T.max(0, T.min(ids[slot, i], skv - 1)), c * 64 + d], 0)
        T.gemm(ds_a[0, :, :], kvm[0, :, :], dq, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def dkv_consume(DKV, ids, valid, qm, dom, ds_t, p_t, acc, row_cache, valid_cache, c, slot, ready, free, role):
        annotate(qm, False)
        annotate(dom, False)
        T.barrier_wait(ready[role * 2 + c % 2], (c // 2) & 1)
        T.gemm(ds_t, qm[c % 2, :, :], acc, transpose_A=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
        T.gemm(p_t, dom[c % 2, :, :], acc, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
        T.lma_wait()
        T.barrier_arrive(free[role * 2 + c % 2])
        for i, d in T.Parallel(block, columns):
            if valid_cache[i, d] != 0:
                T.atomic_add(DKV[row_cache[i, d] + (role * 8 + c) * columns + d], acc[i, d])

    @T.macro
    def dkv_load(Q, DO, qm, dom, s, h0, ready, free, role):
        annotate(qm, False)
        annotate(dom, False)
        for c in T.serial(8):
            if c >= 2:
                T.barrier_wait(free[role * 2 + c % 2], ((c // 2) - 1) & 1)
            T.copy(Q[s, h0:h0 + hg, (role * 8 + c) * columns:(role * 8 + c + 1) * columns], qm[c % 2, :, :], force_async_copy=True)
            T.copy(DO[s, h0:h0 + hg, (role * 8 + c) * columns:(role * 8 + c + 1) * columns], dom[c % 2, :, :], force_async_copy=True)
            T.lma_wait()
            T.barrier_arrive(ready[role * 2 + c % 2])

    @T.macro
    def prefetch(KV, Indices, kvk, ids, valid, s, tile, slot, length, skv):
        for i in T.Parallel(block):
            ids[slot, i] = Indices[s, tile * block + i]
            valid[slot, i] = T.cast(ids[slot, i] >= 0 and ids[slot, i] < skv and tile * block + i < length, "int32")
        for c in T.serial(4):
            for i, d in T.Parallel(block, 128):
                if valid[slot, i] != 0:
                    T.copy(KV[T.max(0, T.min(ids[slot, i], skv - 1)), c * 128 + d],
                           kvk[slot, c, i, d], force_async_copy=True)
                else:
                    kvk[slot, c, i, d] = 0
        # The role barrier below the call publishes the completed other slot.
        # Scalar gather already lowers to LDLMS plus a local completion wait.
        T.lma_wait()

    @T.prim_func
    def dsa_bwd_gradient(Q: T.Tensor((sq, heads, 512), "bfloat16"),
               KV: T.Tensor((skv, 512), "bfloat16"),
               DO: T.Tensor((sq, heads, 512), "bfloat16"),
               Indices: T.Tensor((sq, topk), "int32"),
               Lengths: T.Tensor((sq,), "int32"),
               LSE: T.Tensor((sq, heads), "float32"),
               Delta: T.Tensor((sq, heads), "float32"),
               Sink: T.Tensor((heads,), "float32"),
               DQ: T.Tensor((sq, heads, 512), "bfloat16"),
               DKV: T.Tensor((skv, 512), "float32"),
               Scale: T.float32):
        with T.Kernel(sq, heads // hg, threads=512) as (s, hb):
            # Early readers retire before the same storage receives dKV operands.
            # Four aligned segments form the 32 KiB arena; layouts change only after release.
            dp_s = T.alloc_shared((2, hg, block), "bfloat16")
            ds_a = T.alloc_shared((2, hg, block), "bfloat16")
            kvm0 = T.alloc_shared((2, block, 64), "bfloat16")
            kvm1 = T.alloc_shared((2, block, 64), "bfloat16")
            qm0 = dp_s
            dom0 = ds_a
            qm1 = T.view(kvm0, (64 // columns, hg, columns), "bfloat16")
            dom1 = T.view(kvm1, (64 // columns, hg, columns), "bfloat16")
            qk = T.alloc_shared((4, hg, 128), "bfloat16")
            dok = T.alloc_shared((1 if block == 32 else 4, hg, 128), "bfloat16")
            kvk = T.alloc_shared((2, 4, block, 128), "bfloat16")
            ds_t = T.alloc_shared((hg, block), "bfloat16")
            p_t = T.alloc_shared((hg, block), "bfloat16")
            ids = T.alloc_shared((2, block), "int32")
            valid = T.alloc_shared((2, block), "int32")
            # One barrier per phase boundary, each completed once per tile (phase = t & 1).
            bar = T.alloc_barrier([512] * 4)
            ready = T.alloc_barrier([128] * 4)
            free = T.alloc_barrier([128] * 4)
            dkv_flat = T.view(DKV, (skv * 512,), "float32")
            h0 = hb * hg
            tid = T.get_thread_binding()
            for c in T.serial(4):
                for h, d in T.Parallel(hg, 128):
                    qk[c, h, d] = Q[s, h0 + h, c * 128 + d]
                    if block != 32:
                        dok[c, h, d] = DO[s, h0 + h, c * 128 + d]
            length = T.min(T.max(Lengths[s], 0), topk)
            if tid < 256:
                s_f = T.alloc_fragment((hg, block), "float32")
                acc = T.alloc_fragment((block, columns), "float32")
                row_cache = T.alloc_fragment((block, columns), "int64")
                valid_cache = T.alloc_fragment((block, columns), "int32")
                dq0 = T.alloc_fragment((hg, 64), "float32")
                dq1 = T.alloc_fragment((hg, 64), "float32")
                dq2 = T.alloc_fragment((hg, 64), "float32")
                dq3 = T.alloc_fragment((hg, 64), "float32")
                T.clear(dq0)
                T.clear(dq1)
                T.clear(dq2)
                T.clear(dq3)
                for t in T.serial(T.ceildiv(length, block)):
                    T.lma_wait()
                    T.barrier_arrive(bar[0])
                    T.barrier_wait(bar[0], t & 1)
                    if t == 0:
                        for i in T.Parallel(block):
                            ids[t & 1, i] = Indices[s, t * block + i]
                            valid[t & 1, i] = T.cast(ids[t & 1, i] >= 0 and ids[t & 1, i] < skv and t * block + i < length, "int32")
                        for c in T.serial(4):
                            for i, d in T.Parallel(block, 128):
                                kvk[t & 1, c, i, d] = T.if_then_else(valid[t & 1, i] != 0,
                                                              KV[T.max(0, T.min(ids[t & 1, i], skv - 1)), c * 128 + d], 0)
                    T.lma_wait()
                    T.barrier_arrive(bar[1])
                    T.barrier_wait(bar[1], t & 1)
                    T.clear(s_f)
                    for c in T.unroll(4):
                        T.gemm(qk[c, :, :], kvk[t & 1, c, :, :], s_f, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for h, i in T.Parallel(hg, block):
                        s_f[h, i] = T.if_then_else(valid[t & 1, i] != 0,
                                                   T.exp2(s_f[h, i] * (Scale * _LOG2E) - LSE[s, h0 + h]), 0)
                        if Lengths[s] == 1 and Sink[h0 + h] == -T.infinity("float32"):
                            s_f[h, i] = T.cast(valid[t & 1, i], "float32")
                    T.lma_wait()
                    T.barrier_arrive(bar[2])
                    T.barrier_wait(bar[2], t & 1)
                    annotate(ds_a, True)
                    for h, i in T.Parallel(hg, block):
                        # Keep the condition in the expression; rebinding a Python scalar loses it.
                        ds = T.if_then_else(Lengths[s] == 1 and Sink[h0 + h] == -T.infinity("float32"),
                                            0, s_f[h, i] * (T.reinterpret("float32", T.cast(T.reinterpret("uint16", dp_s[0, h, i]), "uint32") | (T.cast(T.reinterpret("uint16", dp_s[1, h, i]), "uint32") << 16)) - Delta[s, h0 + h]) * Scale)
                        ds_a[0, h, i] = ds
                        ds_t[h, i] = ds
                        p_t[h, i] = s_f[h, i]
                    T.lma_wait()
                    T.barrier_arrive(bar[3])
                    T.barrier_wait(bar[3], 0)
                    dq_panel(KV, kvk, ids, valid, kvm0, ds_a, dq0, 0, skv, t & 1)
                    dq_panel(KV, kvk, ids, valid, kvm0, ds_a, dq1, 1, skv, t & 1)
                    dq_panel(KV, kvk, ids, valid, kvm0, ds_a, dq2, 2, skv, t & 1)
                    dq_panel(KV, kvk, ids, valid, kvm0, ds_a, dq3, 3, skv, t & 1)
                    T.lma_wait()
                    T.barrier_arrive(bar[3])
                    T.barrier_wait(bar[3], 1)
                    if tid < 128:
                        for i, d in T.Parallel(block, columns):
                            row_cache[i, d] = T.cast(ids[t & 1, i], "int64") * 512
                            valid_cache[i, d] = valid[t & 1, i]
                        for c in T.serial(8):
                            dkv_consume(dkv_flat, ids, valid, qm0, dom0, ds_t, p_t, acc, row_cache, valid_cache, c, t & 1, ready, free, 0)
                    else:
                        dkv_load(Q, DO, qm0, dom0, s, h0, ready, free, 0)
                        if t + 1 < T.ceildiv(length, block):
                            prefetch(KV, Indices, kvk, ids, valid, s, t + 1, (t + 1) & 1, length, skv)
                    T.sync_threads()
                for h, d in T.Parallel(hg, 64):
                    DQ[s, h0 + h, 0 + d] = dq0[h, d]
                    DQ[s, h0 + h, 64 + d] = dq1[h, d]
                    DQ[s, h0 + h, 128 + d] = dq2[h, d]
                    DQ[s, h0 + h, 192 + d] = dq3[h, d]
            else:
                dp_f = T.alloc_fragment((hg, block), "float32")
                acc = T.alloc_fragment((block, columns), "float32")
                row_cache = T.alloc_fragment((block, columns), "int64")
                valid_cache = T.alloc_fragment((block, columns), "int32")
                dq2 = T.alloc_fragment((hg, 64), "float32")
                dq3 = T.alloc_fragment((hg, 64), "float32")
                dq0 = T.alloc_fragment((hg, 64), "float32")
                dq1 = T.alloc_fragment((hg, 64), "float32")
                T.clear(dq0)
                T.clear(dq1)
                T.clear(dq2)
                T.clear(dq3)
                for t in T.serial(T.ceildiv(length, block)):
                    T.lma_wait()
                    T.barrier_arrive(bar[0])
                    T.barrier_wait(bar[0], t & 1)
                    T.lma_wait()
                    T.barrier_arrive(bar[1])
                    T.barrier_wait(bar[1], t & 1)
                    T.clear(dp_f)
                    for c in T.unroll(4):
                        if block == 32:
                            for h, d in T.Parallel(hg, 128):
                                dok[0, h, d] = DO[s, h0 + h, c * 128 + d]
                        T.gemm(dok[0 if block == 32 else c, :, :], kvk[t & 1, c, :, :], dp_f, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                    # Preserve all FP32 bits without a mixed-dtype alias in the arena.
                    for h, i in T.Parallel(hg, block):
                        bits = T.reinterpret("uint32", dp_f[h, i])
                        dp_s[0, h, i] = T.reinterpret("bfloat16", T.cast(bits & 65535, "uint16"))
                        dp_s[1, h, i] = T.reinterpret("bfloat16", T.cast(bits >> 16, "uint16"))
                    T.lma_wait()
                    T.barrier_arrive(bar[2])
                    T.barrier_wait(bar[2], t & 1)
                    T.lma_wait()
                    T.barrier_arrive(bar[3])
                    T.barrier_wait(bar[3], 0)
                    dq_panel(KV, kvk, ids, valid, kvm1, ds_a, dq0, 4, skv, t & 1)
                    dq_panel(KV, kvk, ids, valid, kvm1, ds_a, dq1, 5, skv, t & 1)
                    dq_panel(KV, kvk, ids, valid, kvm1, ds_a, dq2, 6, skv, t & 1)
                    dq_panel(KV, kvk, ids, valid, kvm1, ds_a, dq3, 7, skv, t & 1)
                    T.lma_wait()
                    T.barrier_arrive(bar[3])
                    T.barrier_wait(bar[3], 1)
                    if tid < 384:
                        for i, d in T.Parallel(block, columns):
                            row_cache[i, d] = T.cast(ids[t & 1, i], "int64") * 512
                            valid_cache[i, d] = valid[t & 1, i]
                        for c in T.serial(8):
                            dkv_consume(dkv_flat, ids, valid, qm1, dom1, ds_t, p_t, acc, row_cache, valid_cache, c, t & 1, ready, free, 1)
                    else:
                        dkv_load(Q, DO, qm1, dom1, s, h0, ready, free, 1)
                    T.sync_threads()
                for h, d in T.Parallel(hg, 64):
                    DQ[s, h0 + h, 256 + d] = dq0[h, d]
                    DQ[s, h0 + h, 320 + d] = dq1[h, d]
                    DQ[s, h0 + h, 384 + d] = dq2[h, d]
                    DQ[s, h0 + h, 448 + d] = dq3[h, d]
    return dsa_bwd_gradient
