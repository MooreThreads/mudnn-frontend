# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

from .common import cpp_tests


def test_graph_api_executes_pointwise_aliases_graph():
    # From cudnn_frontend pointwise alias coverage used by framework graph
    # lowering paths.
    assert cpp_tests.mudnn_frontend_graph_api_executes_pointwise_aliases_graph()


def test_graph_api_executes_gen_index_graph():
    # From cudnn_frontend generated-index pointwise graph construction.
    assert cpp_tests.mudnn_frontend_graph_api_executes_gen_index_graph()


def test_graph_api_executes_serialized_scale_graph():
    # From cudnn_frontend graph serialization plus scale pointwise execution.
    assert cpp_tests.mudnn_frontend_graph_api_executes_serialized_scale_graph()


def test_executes_compare_select_graph():
    # From ONNX Runtime/PyTorch compare-and-select graph lowering patterns.
    assert cpp_tests.mudnn_frontend_executes_compare_select_graph()


def test_executes_scale_graph():
    # From framework scalar scale graph nodes represented as pointwise ops.
    assert cpp_tests.mudnn_frontend_executes_scale_graph()
