# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""MP31 sparse prefill forward with 64-head CTAs."""

import math

from ..utils.index_type import INT32_ADDRESS_SPACE_BYTES, _index_promotion_variant
from ..utils.prefill import prepare_sparse_mla_strided_tensor
from ..utils.runtime import compile_only, cosize, resolve_num_mps


def compile_mp31(q, kv, indices, attn_sink, topk_length, indexer_topk):
    from .dsa_fwd_mp31 import forward

    factory = forward
    if any(math.prod(t.shape) * size > INT32_ADDRESS_SPACE_BYTES for t, size in ((q, 2), (kv, 2), (indices, 4))):
        factory = _index_promotion_variant(factory)
    kernel = factory(
        *q.shape[1:],
        has_attn_sink=attn_sink is not None,
        has_topk_length=topk_length is not None,
        indexer_blocks=indexer_topk // 64,
    )
    return kernel, resolve_num_mps(q.device)


def launch_mp31(compiled, q, kv, indices, scale, attn_sink, topk_length, out, max_logits, lse, lse_indexer):
    kernel, persistent_blocks = compiled
    q, _ = prepare_sparse_mla_strided_tensor("q", q, multiple=8)
    kv, _ = prepare_sparse_mla_strided_tensor("kv", kv.unsqueeze(1), multiple=8)
    indices, _ = prepare_sparse_mla_strided_tensor("indices", indices.unsqueeze(1), multiple=8)
    args = [q, kv, indices, out, max_logits, lse]
    if lse_indexer is not None:
        args.append(lse_indexer)
    if topk_length is not None:
        args.append(topk_length)
    if attn_sink is not None:
        args.append(attn_sink)
    args.extend((scale * 1.44269504, persistent_blocks, cosize(kv.shape, kv.stride()) * kv.element_size()))
    if not compile_only():
        kernel(*args)
