# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

from .common import cpp_tests


def test_graph_api_executes_sdpa_graph():
    # From PyTorch/TransformerEngine/XLA/TensorFlow SDPA graph usage.
    assert cpp_tests.mudnn_frontend_graph_api_executes_sdpa_graph()


def test_graph_api_executes_sdpa_bias_bool_mask_graph():
    # From framework SDPA graphs with bias and boolean masks.
    assert cpp_tests.mudnn_frontend_graph_api_executes_sdpa_bias_bool_mask_graph()


def test_graph_api_executes_sdpa_generate_stats_graph():
    # From training-oriented SDPA graph usage that requests generated stats.
    assert cpp_tests.mudnn_frontend_graph_api_executes_sdpa_generate_stats_graph()


def test_builds_sdpa_graph():
    # From cudnn_frontend lower-level SDPA operation graph construction.
    assert cpp_tests.mudnn_frontend_builds_sdpa_graph()


def test_executes_sdpa_bias_bool_mask_graph():
    # From PyTorch/TransformerEngine SDPA execution with bias and bool mask.
    assert cpp_tests.mudnn_frontend_executes_sdpa_bias_bool_mask_graph()


def test_executes_sdpa_backward_graph():
    # From framework SDPA backward graph construction/execution coverage.
    assert cpp_tests.mudnn_frontend_executes_sdpa_backward_graph()


def test_graph_api_builds_sdpa_backward_graph():
    # From cudnn_frontend graph API SDPA backward builder coverage.
    assert cpp_tests.mudnn_frontend_graph_api_builds_sdpa_backward_graph()
