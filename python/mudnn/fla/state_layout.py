# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Recurrent-state layout swap between ``[N, H, V, K]`` and ``[N, H, K, V]``."""

from __future__ import annotations

import torch


class TransposeState(torch.autograd.Function):
    """Swap the last two axes of a recurrent state, compact in both directions."""

    @staticmethod
    def forward(ctx, state):
        return state.transpose(-1, -2).contiguous()

    @staticmethod
    def backward(ctx, grad):
        return grad.transpose(-1, -2).contiguous()


__all__ = ["TransposeState"]
