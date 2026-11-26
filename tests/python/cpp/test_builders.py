# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

from .common import cpp_tests


def test_builds_pointwise_add_graph():
    # From cudnn_frontend C++ builder usage: tensor/pointwise/operation graph
    # construction should match the cuDNN frontend-compatible API surface.
    assert cpp_tests.mudnn_frontend_builds_pointwise_add_graph()
