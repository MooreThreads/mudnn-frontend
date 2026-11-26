# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

from .common import cpp_tests


def test_graph_api_executes_concat_reshape_graph():
    # From cudnn_frontend graph API concat/reshape coverage used by framework
    # graph lowering and tensor view handling.
    assert cpp_tests.mudnn_frontend_graph_api_executes_concat_reshape_graph()


def test_graph_api_executes_reduction_graph():
    # From PyTorch/ONNX Runtime reduction graph patterns.
    assert cpp_tests.mudnn_frontend_graph_api_executes_reduction_graph()


def test_graph_api_executes_genstats_graph():
    # From framework normalization and convolution genstats graph usage.
    assert cpp_tests.mudnn_frontend_graph_api_executes_genstats_graph()


def test_graph_api_executes_rng_graph():
    # From cudnn_frontend RNG graph API construction.
    assert cpp_tests.mudnn_frontend_graph_api_executes_rng_graph()


def test_graph_api_builds_batchnorm_graphs():
    # From PyTorch/TensorFlow batchnorm graph construction.
    assert cpp_tests.mudnn_frontend_graph_api_builds_batchnorm_graphs()


def test_graph_api_builds_conv_resample_graphs():
    # From PyTorch/ONNX Runtime convolution and resample graph construction.
    assert cpp_tests.mudnn_frontend_graph_api_builds_conv_resample_graphs()


def test_graph_api_builds_batchnorm_forward_backward_graphs():
    # From framework training batchnorm forward/backward graph construction.
    assert cpp_tests.mudnn_frontend_graph_api_builds_batchnorm_forward_backward_graphs()


def test_graph_api_builds_norm_alias_graphs():
    # From cudnn_frontend normalization alias API coverage.
    assert cpp_tests.mudnn_frontend_graph_api_builds_norm_alias_graphs()


def test_graph_api_builds_block_scale_graphs():
    # From TransformerEngine/cudnn_frontend block-scale graph construction.
    assert cpp_tests.mudnn_frontend_graph_api_builds_block_scale_graphs()


def test_graph_api_builds_paged_cache_signal_graphs():
    # From XLA/TransformerEngine-style paged cache and signal graph nodes.
    assert cpp_tests.mudnn_frontend_graph_api_builds_paged_cache_signal_graphs()
