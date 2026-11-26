# Framework Graph Tests

This directory contains standalone pybind-backed `mudnn_frontend` graph cases
derived from real framework usage patterns. The C++ side builds and executes
the frontend graphs through muDNN; pytest creates inputs, computes torch golden
outputs, and checks correctness.

- PyTorch and ONNX Runtime build cuDNN frontend operation graphs around
  convolution forward and fused activation paths.
- PyTorch, ONNX Runtime, XLA, TensorFlow, and cudnn-frontend FE API tests use
  graph matmul epilogues. The executable covers the currently supported
  matmul-plus-bias path with an expanded bias tensor.
- Transformer and compiler stacks use graph SDPA forward inference with causal
  masking.

The C++ pybind functions measure full graph execution with
`musaEventElapsedTime`. They intentionally do not use muDNN profiling because
frontend graphs can dispatch multiple underlying muDNN ops.

Build the frontend pybind test modules from the `mudnn-frontend` repository
root with:

```bash
source env.sh.example
./build_frontend_tests.sh
```

Then run pytest explicitly:

```bash
MUDNN_FRONTEND_FRAMEWORK_TEST_BUILD_DIR=$PWD/build/tests/cpp \
  pytest -q tests/python/framework/test_framework_graph.py
```

Kernel-path tests use `MUSA_LOG=0xffff` in a subprocess and parse
`muModuleGetFunction(...)` entries. This path does not depend on `torch_musa`
or Kineto, and validates the expected kernel count for each graph case:

```bash
python3 tests/python/framework/kernel_log_runner.py --case matmul_bias
python3 tests/python/framework/kernel_log_runner.py
pytest -q tests/python/framework/test_kernel_log_runner.py
```

For a manual CMake-only build:

```bash
cmake -S . -B build -DMUDNN_FRONTEND_BUILD_FRAMEWORK_TESTS=ON
cmake --build build --target mudnn_frontend_framework_test
```
