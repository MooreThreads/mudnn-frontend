# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from mudnn.deepseek_sparse_attention.utils.runtime import resolve_stream, torch_stream_context
from deepseek_sparse_attention.tilelang.dsa_utils import native_device


def test_torch_stream_context_preserves_legacy_default_stream():
    native_device()
    outer_stream = torch.musa.Stream()
    with torch.musa.stream(outer_stream):
        with torch_stream_context(0):
            assert torch.musa.current_stream() == torch.musa.default_stream()
        assert torch.musa.current_stream() == outer_stream
    device = torch.device("musa", torch.musa.current_device())
    assert resolve_stream(device, outer_stream.musa_stream).musa_stream == outer_stream.musa_stream


def test_torch_stream_context_runs_cpu_work_eagerly():
    with torch_stream_context(None, torch.device("cpu")):
        assert torch.ones(2).sum().item() == 2
    assert resolve_stream(torch.device("cpu"), None) is None


@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
def test_backward_workspace_reserves_head_staging_only_when_needed(heads):
    from mudnn.deepseek_sparse_attention.sparse_attention_backward._workspace import layout, views

    specs, size = layout(3, 9, heads, 65)
    buffers = views(torch.empty(size, dtype=torch.uint8), specs)
    for name in ("q", "out", "dout", "dq"):
        assert buffers[name].numel() == (3 * 64 * 512 if heads < 64 else 0)
    assert buffers["indices"].shape == (3, 128)
    assert buffers["dkv"].shape == (9, 512)
    assert all(offset % 256 == 0 for offset, *_ in specs.values())
    nonempty = [(offset, offset + count) for offset, count, *_ in specs.values() if count]
    assert all(left[1] <= right[0] for left, right in zip(nonempty, nonempty[1:]))
