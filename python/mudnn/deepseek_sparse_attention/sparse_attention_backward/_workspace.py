# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from math import prod


def layout(sq, skv, heads, topk):
    import torch

    h, k = max(heads, 64), (topk + 63) // 64 * 64
    # Full head groups use caller buffers directly; only small heads need staging.
    padded_sq = sq if heads < h else 0
    specs = (("q", (padded_sq, h, 512), torch.bfloat16, 2),
             ("out", (padded_sq, h, 512), torch.bfloat16, 2),
             ("dout", (padded_sq, h, 512), torch.bfloat16, 2),
             ("lse", (sq, h), torch.float32, 4),
             ("sink", (h,), torch.float32, 4),
             ("indices", (sq, k), torch.int32, 4),
             ("lengths", (sq,), torch.int32, 4),
             ("delta", (sq, h), torch.float32, 4),
             ("effective_lse", (sq, h), torch.float32, 4),
             ("dq", (padded_sq, h, 512), torch.bfloat16, 2),
             ("dkv", (skv, 512), torch.float32, 4),
             ("d_sink", (h,), torch.float32, 4))
    offset, result = 0, {}
    for name, shape, dtype, itemsize in specs:
        size = prod(shape) * itemsize
        result[name] = (offset, size, shape, dtype)
        offset += (size + 255) // 256 * 256
    return result, offset


def views(workspace, specs):
    return {name: workspace.narrow(0, offset, size).view(dtype).view(shape)
            for name, (offset, size, shape, dtype) in specs.items()}
