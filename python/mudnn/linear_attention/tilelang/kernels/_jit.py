"""Shared TileLang JIT configuration for the GDN kernels.

Every GDN kernel starts from the same MUSA pass config and compile flags and
overrides a couple of entries. Mirrors ``flash_attention/_jit.py``.

Fast math is OFF for every GDN kernel and is not an override any of them may
take: under ``-ffast-math`` mcc optimizes ``-inf`` masking away, and the
chunk-boundary masks depend on it.

Each kernel's ``target=`` stays on its own decorator: some let TileLang detect
the target, and pinning it here would change their compile key.
"""

from __future__ import annotations

import tilelang

BASE_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
}

BASE_COMPILE_FLAGS = [
    "-Od3", "-fno-signed-zeros",
    "-mllvm", "-mtgpu-if-convert=1",
    "-mllvm", "-misched=mtgpu-max-ilp",
    "-mllvm", "-mtgpu-tiny-offset-hint=1",
    "-mllvm", "-misched-recompute-slotindex=1",
    "-mllvm", "-mtgpu-combine-fop-instr=1",
]

_NO_POSTRA = ["-mllvm", "-mtgpu-enable-postra-sched=0"]

KKT_PASS_CONFIGS = {
    **BASE_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
}
KKT_COMPILE_FLAGS = BASE_COMPILE_FLAGS

FWD_PASS_CONFIGS = {
    **BASE_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: False,
}
FWD_COMPILE_FLAGS = (BASE_COMPILE_FLAGS
                     + ["-fmusa-flush-denormals-to-zero"])

SCAN_PASS_CONFIGS = {
    **BASE_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: False,
}
SCAN_COMPILE_FLAGS = BASE_COMPILE_FLAGS

GRAD_PASS_CONFIGS = {
    **BASE_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_ENABLE_ASYNC_COPY: False,
    tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}
GRAD_COMPILE_FLAGS = (BASE_COMPILE_FLAGS
                      + ["-mllvm", "-mtgpu-maxregcnt=121"] + _NO_POSTRA)
