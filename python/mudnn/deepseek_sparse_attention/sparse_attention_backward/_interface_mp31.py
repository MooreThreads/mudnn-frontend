# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MP31 sparse attention backward: compile and launch over a staged workspace."""


def compile_mp31(heads: int, topk: int):
    from .dsa_bwd_mp31 import gradient, preprocess

    return preprocess(heads), gradient(heads, topk)


def launch_mp31(kernels, w, kv, scale: float) -> None:
    """Device kernels only; accumulates into the zeroed FP32 dKV workspace."""
    # The sink gradient is accumulated atomically by the row preprocess.
    w["d_sink"].zero_()
    pre, grad = kernels
    pre(w["out"], w["dout"], w["lse"], w["sink"], w["delta"], w["effective_lse"], w["d_sink"])
    grad(w["q"], kv, w["dout"], w["indices"], w["lengths"], w["effective_lse"], w["delta"], w["sink"], w["dq"], w["dkv"], scale)
