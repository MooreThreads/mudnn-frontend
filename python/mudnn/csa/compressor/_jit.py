"""Shared MUSA TileLang compile settings for the CSA compressor kernels.

The compressor keeps fast math disabled for the online-softmax state. FMA
contraction is enabled separately for supported packed multiply-add expressions.
Kernels use explicit ``T.sync_threads()`` around shared-memory handoffs, so
the automatic storage-sync pass can be disabled. The remaining flags follow the MUSA
attention/linear-attention kernels: burst vectorization plus the common LLVM
scheduling and instruction-combining options.
"""

from __future__ import annotations

import tilelang


COMPILER_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    # tilelang.PassConfigKey.TL_ENABLE_FMA_CONTRACTION: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}


COMPILER_COMPILE_FLAGS = [
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-mtgpu-if-convert=1",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]
