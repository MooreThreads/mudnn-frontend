# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Input factories copied from upstream linear_attention/conftest.py."""

import torch
import torch.nn.functional as F


def multidist_randu(num_dists, dim, *, device, mean_std=0.05, lower=-0.25, upper=0.25):
    """Rows drawn from per-row uniform distributions with normally-distributed
    means: mildly heterogeneous data that still keeps the recurrence stable."""
    means = torch.randn(num_dists, 1, device=device, dtype=torch.float32) * mean_std
    return means + torch.rand(num_dists, dim, device=device, dtype=torch.float32) * (upper - lower) + lower


def gen_qkv(B, T, H, HV, K, V, dtype, device="musa"):
    """Dense [B, T, heads, dim] Q/K/V. K is l2-normalized along the feature
    dim so the delta-rule update (I - Beta K K^T) stays contractive for
    Beta in (0, 1]."""
    q = multidist_randu(B * T * H, K, device=device).reshape(B, T, H, K)
    k = multidist_randu(B * T * H, K, device=device).reshape(B, T, H, K)
    k = F.normalize(k, p=2.0, dim=-1)
    v = multidist_randu(B * T * HV, V, device=device).reshape(B, T, HV, V)
    return (
        q.to(dtype).contiguous(),
        k.to(dtype).contiguous(),
        v.to(dtype).contiguous(),
    )


from .reference_kda_recurrent import rms_ratio as _rms_ratio


def rms_ratio(out, ref):
    """Run the upstream FP64 comparison on CPU, where FP64 is supported."""
    return _rms_ratio(out.cpu(), ref.cpu())
