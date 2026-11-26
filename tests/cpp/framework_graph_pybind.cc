/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#include <mudnn_frontend.h>
#include <musa_runtime_api.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
namespace fe = mudnn_frontend::graph;

namespace {

constexpr int kWarmupIters = 3;
constexpr int kPerfIters = 20;

void CheckMudnn(mudnnStatus_t status, const char* expr) {
  if (status != MUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(expr) +
                             " failed: " + mudnnGetErrorString(status));
  }
}

void CheckMusa(musaError_t status, const char* expr) {
  if (status != musaSuccess) {
    throw std::runtime_error(std::string(expr) +
                             " failed: " + musaGetErrorString(status));
  }
}

void CheckGraph(const fe::error_t& status, const char* expr) {
  if (!status.is_good()) {
    throw std::runtime_error(std::string(expr) +
                             " failed: " + status.get_message());
  }
}

#define CHECK_MUDNN_EXPR(expr) CheckMudnn((expr), #expr)
#define CHECK_MUSA_EXPR(expr) CheckMusa((expr), #expr)
#define CHECK_GRAPH_EXPR(expr) CheckGraph((expr), #expr)

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes_ > 0) {
      CHECK_MUSA_EXPR(musaMalloc(&ptr_, bytes_));
    }
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)musaFree(ptr_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        (void)musaFree(ptr_);
      }
      ptr_ = other.ptr_;
      bytes_ = other.bytes_;
      other.ptr_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }

  void* get() const { return ptr_; }
  size_t bytes() const { return bytes_; }

  void copy_from_host(const void* host) const {
    if (bytes_ > 0) {
      CHECK_MUSA_EXPR(musaMemcpy(ptr_, host, bytes_, musaMemcpyHostToDevice));
    }
  }

  void copy_to_host(void* host) const {
    if (bytes_ > 0) {
      CHECK_MUSA_EXPR(musaMemcpy(host, ptr_, bytes_, musaMemcpyDeviceToHost));
    }
  }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

class MudnnHandle {
 public:
  MudnnHandle() { CHECK_MUDNN_EXPR(mudnnCreate(&handle_)); }
  ~MudnnHandle() {
    if (handle_ != nullptr) {
      (void)mudnnDestroy(handle_);
    }
  }
  mudnnHandle_t get() const { return handle_; }

 private:
  mudnnHandle_t handle_ = nullptr;
};

struct HostTensor {
  py::array_t<float, py::array::c_style | py::array::forcecast> array;
  std::vector<int64_t> dims;
  int64_t elements = 0;
};

struct HostBFloat16Tensor {
  py::array_t<uint16_t, py::array::c_style | py::array::forcecast> array;
  std::vector<int64_t> dims;
  int64_t elements = 0;
};

struct HostFloat16BitsTensor {
  py::array_t<uint16_t, py::array::c_style | py::array::forcecast> array;
  std::vector<int64_t> dims;
  int64_t elements = 0;
};

HostTensor RequireFloatTensor(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& array,
    const char* name, int64_t rank) {
  auto info = array.request();
  if (info.ndim != rank) {
    throw std::invalid_argument(std::string(name) + " rank mismatch");
  }
  HostTensor tensor;
  tensor.array = array;
  tensor.elements = 1;
  tensor.dims.reserve(static_cast<size_t>(info.ndim));
  for (py::ssize_t i = 0; i < info.ndim; ++i) {
    tensor.dims.push_back(static_cast<int64_t>(info.shape[i]));
    tensor.elements *= static_cast<int64_t>(info.shape[i]);
  }
  return tensor;
}

HostBFloat16Tensor RequireBFloat16Tensor(
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        array,
    const char* name, int64_t rank) {
  auto info = array.request();
  if (info.ndim != rank) {
    throw std::invalid_argument(std::string(name) + " rank mismatch");
  }
  HostBFloat16Tensor tensor;
  tensor.array = array;
  tensor.elements = 1;
  tensor.dims.reserve(static_cast<size_t>(info.ndim));
  for (py::ssize_t i = 0; i < info.ndim; ++i) {
    tensor.dims.push_back(static_cast<int64_t>(info.shape[i]));
    tensor.elements *= static_cast<int64_t>(info.shape[i]);
  }
  return tensor;
}

HostFloat16BitsTensor RequireFloat16BitsTensor(
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        array,
    const char* name, int64_t rank) {
  auto info = array.request();
  if (info.ndim != rank) {
    throw std::invalid_argument(std::string(name) + " rank mismatch");
  }
  HostFloat16BitsTensor tensor;
  tensor.array = array;
  tensor.elements = 1;
  tensor.dims.reserve(static_cast<size_t>(info.ndim));
  for (py::ssize_t i = 0; i < info.ndim; ++i) {
    tensor.dims.push_back(static_cast<int64_t>(info.shape[i]));
    tensor.elements *= static_cast<int64_t>(info.shape[i]);
  }
  return tensor;
}

std::vector<int64_t> RowMajorStrides(const std::vector<int64_t>& dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i) {
    strides[static_cast<size_t>(i)] =
        strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
  }
  return strides;
}

int64_t NumElements(const std::vector<int64_t>& dims) {
  return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

std::vector<py::ssize_t> PyShape(const std::vector<int64_t>& dims) {
  std::vector<py::ssize_t> shape;
  shape.reserve(dims.size());
  for (auto dim : dims) {
    shape.push_back(static_cast<py::ssize_t>(dim));
  }
  return shape;
}

std::shared_ptr<fe::Tensor_attributes> Tensor(
    fe::Graph& graph, const std::string& name, int64_t uid,
    const std::vector<int64_t>& dims,
    mudnn_frontend::DataType_t data_type = mudnn_frontend::DataType_t::FLOAT) {
  fe::Tensor_attributes attrs;
  attrs.set_name(name)
      .set_uid(uid)
      .set_dim(dims)
      .set_stride(RowMajorStrides(dims))
      .set_data_type(data_type);
  return graph.tensor(attrs);
}

void BuildGraph(fe::Graph& graph, mudnnHandle_t handle) {
  CHECK_GRAPH_EXPR(graph.build_operation_graph(handle));
  CHECK_GRAPH_EXPR(
      graph.create_execution_plans({mudnn_frontend::HeurMode_t::INSTANT}));
  CHECK_GRAPH_EXPR(graph.check_support(handle));
  CHECK_GRAPH_EXPR(graph.build_plans());
}

float TimeGraph(fe::Graph& graph, mudnnHandle_t handle,
                const std::unordered_map<std::shared_ptr<fe::Tensor_attributes>,
                                         void*>& variant_pack,
                void* workspace) {
  for (int i = 0; i < kWarmupIters; ++i) {
    CHECK_GRAPH_EXPR(graph.execute(handle, variant_pack, workspace));
  }
  CHECK_MUSA_EXPR(musaDeviceSynchronize());

  musaEvent_t start = nullptr;
  musaEvent_t stop = nullptr;
  CHECK_MUSA_EXPR(musaEventCreate(&start));
  CHECK_MUSA_EXPR(musaEventCreate(&stop));
  CHECK_MUSA_EXPR(musaEventRecord(start));
  for (int i = 0; i < kPerfIters; ++i) {
    CHECK_GRAPH_EXPR(graph.execute(handle, variant_pack, workspace));
  }
  CHECK_MUSA_EXPR(musaEventRecord(stop));
  CHECK_MUSA_EXPR(musaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_MUSA_EXPR(musaEventElapsedTime(&elapsed_ms, start, stop));
  CHECK_MUSA_EXPR(musaEventDestroy(start));
  CHECK_MUSA_EXPR(musaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(kPerfIters);
}

py::dict Result(py::array_t<float> output, float average_ms) {
  py::dict result;
  result["output"] = std::move(output);
  result["average_ms"] = average_ms;
  result["warmup_iters"] = kWarmupIters;
  result["perf_iters"] = kPerfIters;
  return result;
}

py::dict BFloat16Result(py::array_t<uint16_t> output, float average_ms) {
  py::dict result;
  result["output"] = std::move(output);
  result["average_ms"] = average_ms;
  result["warmup_iters"] = kWarmupIters;
  result["perf_iters"] = kPerfIters;
  result["output_dtype"] = "bfloat16_bits";
  return result;
}

py::dict Float16BitsResult(py::array_t<uint16_t> output, float average_ms) {
  py::dict result;
  result["output"] = std::move(output);
  result["average_ms"] = average_ms;
  result["warmup_iters"] = kWarmupIters;
  result["perf_iters"] = kPerfIters;
  result["output_dtype"] = "float16_bits";
  return result;
}

DeviceBuffer DeviceInput(const HostTensor& tensor) {
  DeviceBuffer buffer(static_cast<size_t>(tensor.elements) * sizeof(float));
  buffer.copy_from_host(tensor.array.request().ptr);
  return buffer;
}

DeviceBuffer DeviceInput(const HostBFloat16Tensor& tensor) {
  DeviceBuffer buffer(static_cast<size_t>(tensor.elements) * sizeof(uint16_t));
  buffer.copy_from_host(tensor.array.request().ptr);
  return buffer;
}

DeviceBuffer DeviceInput(const HostFloat16BitsTensor& tensor) {
  DeviceBuffer buffer(static_cast<size_t>(tensor.elements) * sizeof(uint16_t));
  buffer.copy_from_host(tensor.array.request().ptr);
  return buffer;
}

DeviceBuffer DeviceOutput(const std::vector<int64_t>& dims) {
  return DeviceBuffer(static_cast<size_t>(NumElements(dims)) * sizeof(float));
}

DeviceBuffer DeviceBFloat16Output(const std::vector<int64_t>& dims) {
  return DeviceBuffer(static_cast<size_t>(NumElements(dims)) *
                      sizeof(uint16_t));
}

DeviceBuffer Workspace(fe::Graph& graph) {
  const int64_t workspace_size = graph.get_workspace_size();
  return DeviceBuffer(
      static_cast<size_t>(std::max<int64_t>(workspace_size, 0)));
}

}  // namespace

py::dict matmul_bias(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        a_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        b_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        bias_array) {
  auto a_host = RequireFloatTensor(a_array, "A", 2);
  auto b_host = RequireFloatTensor(b_array, "B", 2);
  auto bias_host = RequireFloatTensor(bias_array, "bias", 2);
  if (a_host.dims[1] != b_host.dims[0]) {
    throw std::invalid_argument("matmul K dimensions mismatch");
  }
  const std::vector<int64_t> y_dims = {a_host.dims[0], b_host.dims[1]};
  if (bias_host.dims != y_dims) {
    throw std::invalid_argument(
        "bias must be expanded to the matmul output shape");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto a = Tensor(graph, "A", 1, a_host.dims);
  auto b = Tensor(graph, "B", 2, b_host.dims);
  auto bias = Tensor(graph, "bias", 3, bias_host.dims);
  auto mm = graph.matmul(a, b, fe::Matmul_attributes().set_name("matmul"));
  auto y = graph.add(mm, bias, fe::Pointwise_attributes().set_name("bias_add"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_a = DeviceInput(a_host);
  auto dev_b = DeviceInput(b_host);
  auto dev_bias = DeviceInput(bias_host);
  auto dev_y = DeviceOutput(y_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {a, dev_a.get()},
      {b, dev_b.get()},
      {bias, dev_bias.get()},
      {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(y_dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict matmul_bias_vector(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        a_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        b_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        bias_array) {
  auto a_host = RequireFloatTensor(a_array, "A", 2);
  auto b_host = RequireFloatTensor(b_array, "B", 2);
  auto bias_host = RequireFloatTensor(bias_array, "bias", 1);
  if (a_host.dims[1] != b_host.dims[0]) {
    throw std::invalid_argument("matmul K dimensions mismatch");
  }
  const std::vector<int64_t> y_dims = {a_host.dims[0], b_host.dims[1]};
  if (bias_host.dims[0] != y_dims[1]) {
    throw std::invalid_argument("bias vector must match matmul N dimension");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto a = Tensor(graph, "A", 4, a_host.dims);
  auto b = Tensor(graph, "B", 5, b_host.dims);
  auto bias = Tensor(graph, "bias", 6, bias_host.dims);
  auto mm = graph.matmul(a, b, fe::Matmul_attributes().set_name("matmul"));
  auto y = graph.add(mm, bias, fe::Pointwise_attributes().set_name("bias_add"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_a = DeviceInput(a_host);
  auto dev_b = DeviceInput(b_host);
  auto dev_bias = DeviceInput(bias_host);
  auto dev_y = DeviceOutput(y_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {a, dev_a.get()},
      {b, dev_b.get()},
      {bias, dev_bias.get()},
      {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(y_dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict matmul_bias_relu(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        a_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        b_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        bias_array) {
  auto a_host = RequireFloatTensor(a_array, "A", 2);
  auto b_host = RequireFloatTensor(b_array, "B", 2);
  auto bias_host = RequireFloatTensor(bias_array, "bias", 2);
  if (a_host.dims[1] != b_host.dims[0]) {
    throw std::invalid_argument("matmul K dimensions mismatch");
  }
  const std::vector<int64_t> y_dims = {a_host.dims[0], b_host.dims[1]};
  if (bias_host.dims != y_dims) {
    throw std::invalid_argument(
        "bias must be expanded to the matmul output shape");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto a = Tensor(graph, "A", 31, a_host.dims);
  auto b = Tensor(graph, "B", 32, b_host.dims);
  auto bias = Tensor(graph, "bias", 33, bias_host.dims);
  auto mm = graph.matmul(a, b, fe::Matmul_attributes().set_name("matmul"));
  auto biased =
      graph.add(mm, bias, fe::Pointwise_attributes().set_name("bias_add"));
  auto y = graph.relu(biased, fe::Pointwise_attributes().set_name("relu"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_a = DeviceInput(a_host);
  auto dev_b = DeviceInput(b_host);
  auto dev_bias = DeviceInput(bias_host);
  auto dev_y = DeviceOutput(y_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {a, dev_a.get()},
      {b, dev_b.get()},
      {bias, dev_bias.get()},
      {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(y_dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict pointwise_residual_gelu(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        x_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        residual_array) {
  auto x_host = RequireFloatTensor(x_array, "X", 2);
  auto residual_host = RequireFloatTensor(residual_array, "residual", 2);
  if (x_host.dims != residual_host.dims) {
    throw std::invalid_argument("X and residual dimensions must match");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto x = Tensor(graph, "X", 41, x_host.dims);
  auto residual = Tensor(graph, "residual", 42, residual_host.dims);
  auto summed = graph.add(x, residual,
                          fe::Pointwise_attributes().set_name("residual_add"));
  auto y = graph.gelu_approx_tanh(summed,
                                  fe::Pointwise_attributes().set_name("gelu"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_x = DeviceInput(x_host);
  auto dev_residual = DeviceInput(residual_host);
  auto dev_y = DeviceOutput(x_host.dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {x, dev_x.get()}, {residual, dev_residual.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(x_host.dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict reduction_sum(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        x_array,
    int64_t axis) {
  auto x_host = RequireFloatTensor(x_array, "X", 3);
  if (axis < 0) {
    axis += static_cast<int64_t>(x_host.dims.size());
  }
  if (axis < 0 || axis >= static_cast<int64_t>(x_host.dims.size())) {
    throw std::invalid_argument("reduction axis out of range");
  }
  std::vector<int64_t> y_dims = x_host.dims;
  y_dims[static_cast<size_t>(axis)] = 1;

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto x = Tensor(graph, "X", 51, x_host.dims);
  auto y = graph.sum(x, axis, fe::Reduction_attributes().set_name("sum"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_x = DeviceInput(x_host);
  auto dev_y = DeviceOutput(y_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {x, dev_x.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(y_dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict matmul_reduce(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        a_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        b_array,
    int64_t axis) {
  auto a_host = RequireFloatTensor(a_array, "A", 2);
  auto b_host = RequireFloatTensor(b_array, "B", 2);
  if (a_host.dims[1] != b_host.dims[0]) {
    throw std::invalid_argument("matmul K dimensions mismatch");
  }
  const std::vector<int64_t> mm_dims = {a_host.dims[0], b_host.dims[1]};
  if (axis < 0) {
    axis += static_cast<int64_t>(mm_dims.size());
  }
  if (axis < 0 || axis >= static_cast<int64_t>(mm_dims.size())) {
    throw std::invalid_argument("reduction axis out of range");
  }
  std::vector<int64_t> y_dims = mm_dims;
  y_dims[static_cast<size_t>(axis)] = 1;

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto a = Tensor(graph, "A", 56, a_host.dims);
  auto b = Tensor(graph, "B", 57, b_host.dims);
  auto mm = graph.matmul(a, b, fe::Matmul_attributes().set_name("matmul"));
  auto y = graph.sum(mm, axis, fe::Reduction_attributes().set_name("sum"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_a = DeviceInput(a_host);
  auto dev_b = DeviceInput(b_host);
  auto dev_y = DeviceOutput(y_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {a, dev_a.get()}, {b, dev_b.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(y_dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict matmul_bias_relu_vector(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        a_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        b_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        bias_array) {
  auto a_host = RequireFloatTensor(a_array, "A", 2);
  auto b_host = RequireFloatTensor(b_array, "B", 2);
  auto bias_host = RequireFloatTensor(bias_array, "bias", 1);
  if (a_host.dims[1] != b_host.dims[0]) {
    throw std::invalid_argument("matmul K dimensions mismatch");
  }
  const std::vector<int64_t> y_dims = {a_host.dims[0], b_host.dims[1]};
  if (bias_host.dims[0] != y_dims[1]) {
    throw std::invalid_argument("bias vector must match matmul N dimension");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto a = Tensor(graph, "A", 36, a_host.dims);
  auto b = Tensor(graph, "B", 37, b_host.dims);
  auto bias = Tensor(graph, "bias", 38, bias_host.dims);
  auto mm = graph.matmul(a, b, fe::Matmul_attributes().set_name("matmul"));
  auto biased =
      graph.add(mm, bias, fe::Pointwise_attributes().set_name("bias_add"));
  auto y = graph.relu(biased, fe::Pointwise_attributes().set_name("relu"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_a = DeviceInput(a_host);
  auto dev_b = DeviceInput(b_host);
  auto dev_bias = DeviceInput(bias_host);
  auto dev_y = DeviceOutput(y_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {a, dev_a.get()},
      {b, dev_b.get()},
      {bias, dev_bias.get()},
      {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(y_dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict matmul_bias_relu_bf16(
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        a_array,
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        b_array,
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        bias_array) {
  auto a_host = RequireBFloat16Tensor(a_array, "A", 2);
  auto b_host = RequireBFloat16Tensor(b_array, "B", 2);
  auto bias_host = RequireBFloat16Tensor(bias_array, "bias", 2);
  if (a_host.dims[1] != b_host.dims[0]) {
    throw std::invalid_argument("matmul K dimensions mismatch");
  }
  const std::vector<int64_t> y_dims = {a_host.dims[0], b_host.dims[1]};
  if (bias_host.dims != y_dims) {
    throw std::invalid_argument(
        "bias must be expanded to the matmul output shape");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::BFLOAT16)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::BFLOAT16)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto a = Tensor(graph, "A", 131, a_host.dims,
                  mudnn_frontend::DataType_t::BFLOAT16);
  auto b = Tensor(graph, "B", 132, b_host.dims,
                  mudnn_frontend::DataType_t::BFLOAT16);
  auto bias = Tensor(graph, "bias", 133, bias_host.dims,
                     mudnn_frontend::DataType_t::BFLOAT16);
  auto mm = graph.matmul(a, b, fe::Matmul_attributes().set_name("matmul"));
  auto biased =
      graph.add(mm, bias, fe::Pointwise_attributes().set_name("bias_add"));
  auto y = graph.relu(biased, fe::Pointwise_attributes().set_name("relu"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::BFLOAT16);
  BuildGraph(graph, handle.get());

  auto dev_a = DeviceInput(a_host);
  auto dev_b = DeviceInput(b_host);
  auto dev_bias = DeviceInput(bias_host);
  auto dev_y = DeviceBFloat16Output(y_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {a, dev_a.get()},
      {b, dev_b.get()},
      {bias, dev_bias.get()},
      {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<uint16_t> output(PyShape(y_dims));
  dev_y.copy_to_host(output.mutable_data());
  return BFloat16Result(std::move(output), ms);
}

py::dict pointwise_residual_gelu_bf16(
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        x_array,
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        residual_array) {
  auto x_host = RequireBFloat16Tensor(x_array, "X", 2);
  auto residual_host = RequireBFloat16Tensor(residual_array, "residual", 2);
  if (x_host.dims != residual_host.dims) {
    throw std::invalid_argument("X and residual dimensions must match");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::BFLOAT16)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::BFLOAT16)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto x = Tensor(graph, "X", 141, x_host.dims,
                  mudnn_frontend::DataType_t::BFLOAT16);
  auto residual = Tensor(graph, "residual", 142, residual_host.dims,
                         mudnn_frontend::DataType_t::BFLOAT16);
  auto summed = graph.add(x, residual,
                          fe::Pointwise_attributes().set_name("residual_add"));
  auto y = graph.gelu_approx_tanh(summed,
                                  fe::Pointwise_attributes().set_name("gelu"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::BFLOAT16);
  BuildGraph(graph, handle.get());

  auto dev_x = DeviceInput(x_host);
  auto dev_residual = DeviceInput(residual_host);
  auto dev_y = DeviceBFloat16Output(x_host.dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {x, dev_x.get()}, {residual, dev_residual.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<uint16_t> output(PyShape(x_host.dims));
  dev_y.copy_to_host(output.mutable_data());
  return BFloat16Result(std::move(output), ms);
}

py::dict reduction_sum_bf16(
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        x_array,
    int64_t axis) {
  auto x_host = RequireBFloat16Tensor(x_array, "X", 3);
  if (axis < 0) {
    axis += static_cast<int64_t>(x_host.dims.size());
  }
  if (axis < 0 || axis >= static_cast<int64_t>(x_host.dims.size())) {
    throw std::invalid_argument("reduction axis out of range");
  }
  std::vector<int64_t> y_dims = x_host.dims;
  y_dims[static_cast<size_t>(axis)] = 1;

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::BFLOAT16)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::BFLOAT16)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto x = Tensor(graph, "X", 151, x_host.dims,
                  mudnn_frontend::DataType_t::BFLOAT16);
  auto y = graph.sum(x, axis, fe::Reduction_attributes().set_name("sum"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::BFLOAT16);
  BuildGraph(graph, handle.get());

  auto dev_x = DeviceInput(x_host);
  auto dev_y = DeviceBFloat16Output(y_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {x, dev_x.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<uint16_t> output(PyShape(y_dims));
  dev_y.copy_to_host(output.mutable_data());
  return BFloat16Result(std::move(output), ms);
}

py::dict concat_reshape(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        x_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        y_array) {
  auto x_host = RequireFloatTensor(x_array, "X", 2);
  auto y_host = RequireFloatTensor(y_array, "Y", 2);
  if (x_host.dims[1] != y_host.dims[1]) {
    throw std::invalid_argument("concat non-axis dimensions must match");
  }
  const std::vector<int64_t> concat_dims = {x_host.dims[0] + y_host.dims[0],
                                            x_host.dims[1]};
  const std::vector<int64_t> out_dims = {2, NumElements(concat_dims) / 2};

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto x = Tensor(graph, "X", 61, x_host.dims);
  auto y = Tensor(graph, "Y", 62, y_host.dims);
  auto cat = graph.concat({x, y}, 0, "concat");
  auto out = graph.reshape(cat, out_dims, "reshape");
  out->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_x = DeviceInput(x_host);
  auto dev_y = DeviceInput(y_host);
  auto dev_out = DeviceOutput(out_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {x, dev_x.get()}, {y, dev_y.get()}, {out, dev_out.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(out_dims));
  dev_out.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict softmax_last_dim(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        x_array) {
  auto x_host = RequireFloatTensor(x_array, "X", 2);

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto x = Tensor(graph, "X", 71, x_host.dims);
  auto exp_x = graph.exp(x, fe::Pointwise_attributes().set_name("exp"));
  auto sum = graph.sum(exp_x, 1, fe::Reduction_attributes().set_name("sum"));
  auto y = graph.div(exp_x, sum, fe::Pointwise_attributes().set_name("div"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_x = DeviceInput(x_host);
  auto dev_y = DeviceOutput(x_host.dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {x, dev_x.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(x_host.dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict conv_relu(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        x_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        w_array) {
  auto x_host = RequireFloatTensor(x_array, "X", 4);
  auto w_host = RequireFloatTensor(w_array, "W", 4);
  if (x_host.dims[1] != w_host.dims[1]) {
    throw std::invalid_argument(
        "conv input/filter channel dimensions mismatch");
  }
  const std::vector<int64_t> y_dims = {x_host.dims[0], w_host.dims[0],
                                       x_host.dims[2], x_host.dims[3]};

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto x = Tensor(graph, "X", 11, x_host.dims);
  auto w = Tensor(graph, "W", 12, w_host.dims);
  auto conv = graph.conv_fprop(
      x, w,
      fe::Conv_attributes()
          .set_name("conv_fprop")
          .set_padding({1, 1})
          .set_stride({1, 1})
          .set_dilation({1, 1})
          .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT));
  auto y = graph.relu(conv, fe::Pointwise_attributes().set_name("relu"));
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_x = DeviceInput(x_host);
  auto dev_w = DeviceInput(w_host);
  auto dev_y = DeviceOutput(y_dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {x, dev_x.get()}, {w, dev_w.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(y_dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict sdpa_causal(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        q_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        k_array,
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        v_array,
    double scale) {
  auto q_host = RequireFloatTensor(q_array, "Q", 4);
  auto k_host = RequireFloatTensor(k_array, "K", 4);
  auto v_host = RequireFloatTensor(v_array, "V", 4);
  if (q_host.dims != k_host.dims || q_host.dims != v_host.dims) {
    throw std::invalid_argument("Q, K, and V must have the same shape");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto q = Tensor(graph, "Q", 21, q_host.dims);
  auto k = Tensor(graph, "K", 22, k_host.dims);
  auto v = Tensor(graph, "V", 23, v_host.dims);
  auto outputs =
      graph.sdpa(q, k, v,
                 fe::Sdpa_attributes()
                     .set_name("sdpa_causal")
                     .set_attn_scale(scale)
                     .set_causal_mask(true)
                     .set_is_inference(true)
                     .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT));
  auto y = outputs[0];
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::FLOAT);
  BuildGraph(graph, handle.get());

  auto dev_q = DeviceInput(q_host);
  auto dev_k = DeviceInput(k_host);
  auto dev_v = DeviceInput(v_host);
  auto dev_y = DeviceOutput(q_host.dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {q, dev_q.get()}, {k, dev_k.get()}, {v, dev_v.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<float> output(PyShape(q_host.dims));
  dev_y.copy_to_host(output.mutable_data());
  return Result(std::move(output), ms);
}

py::dict sdpa_causal_flash_f16(
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        q_array,
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        k_array,
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        v_array,
    double scale) {
  auto q_host = RequireFloat16BitsTensor(q_array, "Q", 4);
  auto k_host = RequireFloat16BitsTensor(k_array, "K", 4);
  auto v_host = RequireFloat16BitsTensor(v_array, "V", 4);
  if (q_host.dims != k_host.dims || q_host.dims != v_host.dims) {
    throw std::invalid_argument("Q, K, and V must have the same shape");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::HALF)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::FLOAT)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto q =
      Tensor(graph, "Q", 31, q_host.dims, mudnn_frontend::DataType_t::HALF);
  auto k =
      Tensor(graph, "K", 32, k_host.dims, mudnn_frontend::DataType_t::HALF);
  auto v =
      Tensor(graph, "V", 33, v_host.dims, mudnn_frontend::DataType_t::HALF);
  auto outputs =
      graph.sdpa(q, k, v,
                 fe::Sdpa_attributes()
                     .set_name("sdpa_causal_flash_f16")
                     .set_attn_scale(scale)
                     .set_causal_mask(true)
                     .set_generate_stats(true)
                     .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT));
  auto y = outputs[0];
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::HALF);
  BuildGraph(graph, handle.get());

  auto dev_q = DeviceInput(q_host);
  auto dev_k = DeviceInput(k_host);
  auto dev_v = DeviceInput(v_host);
  auto dev_y = DeviceBFloat16Output(q_host.dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {q, dev_q.get()}, {k, dev_k.get()}, {v, dev_v.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<uint16_t> output(PyShape(q_host.dims));
  dev_y.copy_to_host(output.mutable_data());
  return Float16BitsResult(std::move(output), ms);
}

py::dict sdpa_causal_flash_bf16(
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        q_array,
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        k_array,
    const py::array_t<uint16_t, py::array::c_style | py::array::forcecast>&
        v_array,
    double scale) {
  auto q_host = RequireBFloat16Tensor(q_array, "Q", 4);
  auto k_host = RequireBFloat16Tensor(k_array, "K", 4);
  auto v_host = RequireBFloat16Tensor(v_array, "V", 4);
  if (q_host.dims != k_host.dims || q_host.dims != v_host.dims) {
    throw std::invalid_argument("Q, K, and V must have the same shape");
  }

  MudnnHandle handle;
  fe::Graph graph;
  graph.set_io_data_type(mudnn_frontend::DataType_t::BFLOAT16)
      .set_intermediate_data_type(mudnn_frontend::DataType_t::BFLOAT16)
      .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT);
  auto q =
      Tensor(graph, "Q", 41, q_host.dims, mudnn_frontend::DataType_t::BFLOAT16);
  auto k =
      Tensor(graph, "K", 42, k_host.dims, mudnn_frontend::DataType_t::BFLOAT16);
  auto v =
      Tensor(graph, "V", 43, v_host.dims, mudnn_frontend::DataType_t::BFLOAT16);
  auto outputs =
      graph.sdpa(q, k, v,
                 fe::Sdpa_attributes()
                     .set_name("sdpa_causal_flash_bf16")
                     .set_attn_scale(scale)
                     .set_causal_mask(true)
                     .set_generate_stats(true)
                     .set_compute_data_type(mudnn_frontend::DataType_t::FLOAT));
  auto y = outputs[0];
  y->set_output(true).set_data_type(mudnn_frontend::DataType_t::BFLOAT16);
  BuildGraph(graph, handle.get());

  auto dev_q = DeviceInput(q_host);
  auto dev_k = DeviceInput(k_host);
  auto dev_v = DeviceInput(v_host);
  auto dev_y = DeviceBFloat16Output(q_host.dims);
  auto workspace = Workspace(graph);
  std::unordered_map<std::shared_ptr<fe::Tensor_attributes>, void*> pack = {
      {q, dev_q.get()}, {k, dev_k.get()}, {v, dev_v.get()}, {y, dev_y.get()}};
  const float ms = TimeGraph(graph, handle.get(), pack, workspace.get());

  py::array_t<uint16_t> output(PyShape(q_host.dims));
  dev_y.copy_to_host(output.mutable_data());
  return BFloat16Result(std::move(output), ms);
}

PYBIND11_MODULE(mudnn_frontend_framework_test, m) {
  m.doc() = "Standalone pybind tests for mudnn_frontend graph usage patterns";
  m.def("matmul_bias", &matmul_bias, py::arg("a"), py::arg("b"),
        py::arg("bias"));
  m.def("matmul_bias_vector", &matmul_bias_vector, py::arg("a"), py::arg("b"),
        py::arg("bias"));
  m.def("matmul_bias_relu", &matmul_bias_relu, py::arg("a"), py::arg("b"),
        py::arg("bias"));
  m.def("matmul_bias_relu_vector", &matmul_bias_relu_vector, py::arg("a"),
        py::arg("b"), py::arg("bias"));
  m.def("pointwise_residual_gelu", &pointwise_residual_gelu, py::arg("x"),
        py::arg("residual"));
  m.def("reduction_sum", &reduction_sum, py::arg("x"), py::arg("axis"));
  m.def("matmul_reduce", &matmul_reduce, py::arg("a"), py::arg("b"),
        py::arg("axis"));
  m.def("matmul_bias_relu_bf16", &matmul_bias_relu_bf16, py::arg("a"),
        py::arg("b"), py::arg("bias"));
  m.def("pointwise_residual_gelu_bf16", &pointwise_residual_gelu_bf16,
        py::arg("x"), py::arg("residual"));
  m.def("reduction_sum_bf16", &reduction_sum_bf16, py::arg("x"),
        py::arg("axis"));
  m.def("concat_reshape", &concat_reshape, py::arg("x"), py::arg("y"));
  m.def("softmax_last_dim", &softmax_last_dim, py::arg("x"));
  m.def("conv_relu", &conv_relu, py::arg("x"), py::arg("w"));
  m.def("sdpa_causal", &sdpa_causal, py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("scale"));
  m.def("sdpa_causal_flash_f16", &sdpa_causal_flash_f16, py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("scale"));
  m.def("sdpa_causal_flash_bf16", &sdpa_causal_flash_bf16, py::arg("q"),
        py::arg("k"), py::arg("v"), py::arg("scale"));
}
