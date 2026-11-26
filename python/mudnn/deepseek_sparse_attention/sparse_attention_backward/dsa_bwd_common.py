# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Shared DSA sparse-attention backward kernels.

The row preprocess runs unchanged on both architectures: it reduces
``O * dO`` to ``Delta``, folds the sink into an effective LSE and accumulates the
sink gradient. It lives here so the per-architecture modules depend on one
shared kernel instead of on each other.
"""

import tilelang
from tilelang import language as T

_CONFIG = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: False,
}
# The base target; callers retarget it per device with ``jit_for_arch``.
_TARGET = {"kind": "musa", "arch": "mp_32"}

_KERNEL_CONFIG = {
    **_CONFIG,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
}
# Scheduling flags shared with the sparse-MLA forward kernel; fast math stays off.
_FLAGS = [
    "-fmusa-flush-denormals-to-zero", "-fno-signed-zeros", "-fno-strict-aliasing",
    "-mllvm", "-misched=mtgpu-max-ilp", "-mllvm", "-mtgpu-if-convert=1",
    "-mllvm", "-mtgpu-tiny-offset-hint=1", "-mllvm", "-mtgpu-combine-fop-instr=1",
    "-mllvm", "-mtgpu-load-store-2d=1", "-mllvm", "-mtgpu-enable-postra-sched=1",
]
_LOG2E = 1.4426950408889634


# Delta and the sink-adjusted LSE per (query, 64 heads); the sink gradient is accumulated
# directly into a zeroed SinkGrad, replacing the serial sink reduction.
@tilelang.jit(target=_TARGET, out_idx=[], pass_configs=_KERNEL_CONFIG, compile_flags=_FLAGS)
def preprocess(heads):
    sq = T.dynamic("sq")

    @T.prim_func
    def dsa_bwd_preprocess(O: T.Tensor((sq, heads, 512), "bfloat16"),
               DO: T.Tensor((sq, heads, 512), "bfloat16"),
               LSE: T.Tensor((sq, heads), "float32"),
               Sink: T.Tensor((heads,), "float32"),
               Delta: T.Tensor((sq, heads), "float32"),
               EffectiveLSE: T.Tensor((sq, heads), "float32"),
               SinkGrad: T.Tensor((heads,), "float32")):
        with T.Kernel(sq, heads // 64, threads=256) as (s, hb):
            part = T.alloc_fragment((64, 128), "float32")
            chunk = T.alloc_fragment((64,), "float32")
            total = T.alloc_fragment((64,), "float32")
            # Row sums go through shared so each head is handled by exactly one thread:
            # the reduced fragment may be replicated across threads, which would repeat atomics.
            total_s = T.alloc_shared((64,), "float32")
            h0 = hb * 64
            T.clear(total)
            for c in T.serial(4):
                for h, d in T.Parallel(64, 128):
                    part[h, d] = T.cast(O[s, h0 + h, c * 128 + d], "float32") * T.cast(DO[s, h0 + h, c * 128 + d], "float32")
                T.reduce_sum(part, chunk, dim=1)
                for h in T.Parallel(64):
                    total[h] += chunk[h]
            T.copy(total, total_s)
            for h in T.Parallel(64):
                lse = LSE[s, h0 + h]
                sink = Sink[h0 + h]
                Delta[s, h0 + h] = total_s[h]
                EffectiveLSE[s, h0 + h] = lse * _LOG2E
                if lse == T.infinity("float32") or sink == T.infinity("float32"):
                    EffectiveLSE[s, h0 + h] = T.infinity("float32")
                elif sink != -T.infinity("float32"):
                    maximum = T.max(lse, sink)
                    effective = maximum + T.log(T.exp(lse - maximum) + T.exp(sink - maximum))
                    EffectiveLSE[s, h0 + h] = effective * _LOG2E
                    T.atomic_add(SinkGrad[h0 + h], -T.exp(sink - effective) * total_s[h])
    return dsa_bwd_preprocess
