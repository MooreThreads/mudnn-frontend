# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

from .common import cpp_tests


def test_executes_matmul_bias_relu_graph():
    # From PyTorch/ONNX Runtime/XLA/TensorFlow matmul epilogue graph patterns.
    assert cpp_tests.mudnn_frontend_executes_matmul_bias_relu_graph()


def test_graph_api_executes_matmul_bias_graph():
    # From cudnn_frontend graph API matmul plus bias construction.
    assert cpp_tests.mudnn_frontend_graph_api_executes_matmul_bias_graph()


def test_executes_matmul_broadcast_bias_graph():
    # From TransformerEngine/PyTorch-style broadcast-bias matmul epilogues.
    assert cpp_tests.mudnn_frontend_executes_matmul_broadcast_bias_graph()
