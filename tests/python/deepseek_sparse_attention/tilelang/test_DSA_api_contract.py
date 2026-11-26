# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Sparse attention FWD/BWD surface parity with cuDNN frontend develop 584a4c6: parameter names, kinds and defaults."""

import importlib.util
import inspect

import pytest

import mudnn
from mudnn.api_base import APIBase
from mudnn import deepseek_sparse_attention as dsa

# Generated from cudnn-frontend 584a4c6 python/cudnn/deepseek_sparse_attention: (name, kind, repr(default)),
# kind P = positional-or-keyword, K = keyword-only; default None means required.
UPSTREAM_SIGNATURES = {
    'SparseAttentionForward.__init__': [('sample_q', 'P', None), ('sample_kv', 'P', None), ('sample_topk_idxs', 'P', None), ('sample_attn_sink', 'P', 'None'), ('sample_topk_length', 'P', 'None'), ('softmax_scale', 'P', 'None'), ('indexer_topk', 'P', '0')],
    'SparseAttentionForward.check_support': [],
    'SparseAttentionForward.compile': [],
    'SparseAttentionForward.execute': [('q', 'P', None), ('kv', 'P', None), ('topk_idxs', 'P', None), ('attn_sink', 'K', 'None'), ('topk_length', 'K', 'None'), ('softmax_scale', 'K', 'None'), ('out', 'K', 'None'), ('max_logits', 'K', 'None'), ('lse', 'K', 'None'), ('lse_indexer', 'K', 'None'), ('current_stream', 'K', 'None')],
    'sparse_attention_forward_wrapper': [('q', 'P', None), ('kv', 'P', None), ('topk_idxs', 'P', None), ('attn_sink', 'K', 'None'), ('topk_length', 'K', 'None'), ('softmax_scale', 'K', 'None'), ('indexer_topk', 'K', '0'), ('stream', 'K', 'None')],
    'SparseAttentionBackward.__init__': [('sample_q', 'P', None), ('sample_kv', 'P', None), ('sample_out', 'P', None), ('sample_dout', 'P', None), ('sample_lse', 'P', None), ('sample_attn_sink', 'P', None), ('sample_topk_idxs', 'P', None), ('sample_dq', 'P', 'None'), ('sample_dkv', 'P', 'None'), ('sample_topk_length', 'P', 'None'), ('softmax_scale', 'P', 'None'), ('block_tile', 'P', '64'), ('deterministic', 'P', 'False'), ('q_cluster_mode', 'P', "'off'")],
    'SparseAttentionBackward.check_support': [],
    'SparseAttentionBackward.compile': [],
    'SparseAttentionBackward.scratch_workspace_bytes': [],
    'SparseAttentionBackward.execute': [('q', 'P', None), ('kv', 'P', None), ('out', 'P', None), ('dout', 'P', None), ('lse', 'P', None), ('attn_sink', 'P', None), ('topk_idxs', 'P', None), ('dq', 'P', None), ('dkv', 'P', None), ('topk_length', 'P', 'None'), ('softmax_scale', 'P', 'None'), ('current_stream', 'P', 'None'), ('workspace', 'P', 'None'), ('d_sink', 'K', None)],
    'sparse_attention_backward_wrapper': [('q', 'P', None), ('kv', 'P', None), ('out', 'P', None), ('dout', 'P', None), ('lse', 'P', None), ('attn_sink', 'P', None), ('topk_idxs', 'P', None), ('softmax_scale', 'P', 'None'), ('topk_length', 'P', 'None'), ('dq', 'P', 'None'), ('dkv', 'P', 'None'), ('block_tile', 'P', '64'), ('deterministic', 'P', 'False'), ('stream', 'P', 'None'), ('workspace', 'P', 'None'), ('q_cluster_mode', 'P', "'off'")],
}

_KINDS = {inspect.Parameter.POSITIONAL_OR_KEYWORD: "P", inspect.Parameter.KEYWORD_ONLY: "K"}
_PUBLIC = sorted({name.split(".")[0] for name in UPSTREAM_SIGNATURES})


def _parameters(obj):
    params = list(inspect.signature(obj).parameters.values())
    return [
        (p.name, _KINDS.get(p.kind, str(p.kind)), None if p.default is inspect.Parameter.empty else repr(p.default))
        for p in params
        if p.name != "self"
    ]


@pytest.mark.parametrize("name", sorted(UPSTREAM_SIGNATURES))
def test_DSA_signature_matches_upstream(name):
    owner, _, method = name.partition(".")
    obj = getattr(dsa, owner)
    if method:
        obj = getattr(obj, method)
    assert _parameters(obj) == UPSTREAM_SIGNATURES[name]


@pytest.mark.parametrize("name", _PUBLIC)
def test_DSA_sparse_attention_exports_are_native(name):
    native = getattr(dsa, name)
    assert native.__module__.startswith("mudnn.deepseek_sparse_attention.sparse_attention_")
    assert getattr(dsa.DSA, name) is native
    assert name in dsa.__all__


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_DSA_sparse_attention_subpackage_exports_match_namespace(direction):
    package = importlib.import_module(f"mudnn.deepseek_sparse_attention.sparse_attention_{direction}")
    names = {f"SparseAttention{direction.title()}", f"sparse_attention_{direction}_wrapper"}
    assert set(package.__all__) == names
    for name in names:
        assert getattr(package, name) is getattr(dsa.DSA, name)
    assert issubclass(getattr(package, f"SparseAttention{direction.title()}"), APIBase)


def test_DSA_namespace_matches_upstream_layout():
    # cuDNN frontend has no top-level DSA package; cudnn.DSA is the namespace object.
    assert mudnn.DSA is dsa.DSA
    assert importlib.util.find_spec("mudnn.DSA") is None
