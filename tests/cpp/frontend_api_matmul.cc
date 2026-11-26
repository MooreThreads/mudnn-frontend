/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#include "frontend_api_common.h"

namespace mudnn_frontend_tests {

bool MudnnFrontendExecutesMatmulBiasReluGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  void* bias_ptr = nullptr;
  void* y_ptr = nullptr;
  void* workspace = nullptr;

  try {
    constexpr int64_t a_dims[] = {2, 3};
    constexpr int64_t a_strides[] = {3, 1};
    constexpr int64_t b_dims[] = {3, 4};
    constexpr int64_t b_strides[] = {4, 1};
    constexpr int64_t y_dims[] = {2, 4};
    constexpr int64_t y_strides[] = {4, 1};

    auto a = mudnn_frontend::TensorBuilder()
                 .setDim(2, a_dims)
                 .setStrides(2, a_strides)
                 .setId(1)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto b = mudnn_frontend::TensorBuilder()
                 .setDim(2, b_dims)
                 .setStrides(2, b_strides)
                 .setId(2)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto matmul_y = mudnn_frontend::TensorBuilder()
                        .setDim(2, y_dims)
                        .setStrides(2, y_strides)
                        .setId(3)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .setVirtual(true)
                        .build();
    auto bias = mudnn_frontend::TensorBuilder()
                    .setDim(2, y_dims)
                    .setStrides(2, y_strides)
                    .setId(4)
                    .setDataType(CUDNN_DATA_FLOAT)
                    .build();
    auto biased = mudnn_frontend::TensorBuilder()
                      .setDim(2, y_dims)
                      .setStrides(2, y_strides)
                      .setId(5)
                      .setDataType(CUDNN_DATA_FLOAT)
                      .setVirtual(true)
                      .build();
    auto y = mudnn_frontend::TensorBuilder()
                 .setDim(2, y_dims)
                 .setStrides(2, y_strides)
                 .setId(6)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();

    auto matmul_desc = mudnn_frontend::MatMulDescBuilder()
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
    auto matmul_op = mudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                         .setMatmulDesc(matmul_desc)
                         .setAMatDesc(a)
                         .setBMatDesc(b)
                         .setCMatDesc(matmul_y)
                         .build();
    auto add_desc = mudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_ADD)
                        .setMathPrecision(CUDNN_DATA_FLOAT)
                        .build();
    auto add_op = mudnn_frontend::OperationBuilder(
                      CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setXDesc(matmul_y)
                      .setBDesc(bias)
                      .setYDesc(biased)
                      .setPwDesc(add_desc)
                      .setAlpha(1.0)
                      .setAlpha2(1.0)
                      .build();
    auto relu_desc =
        mudnn_frontend::PointWiseDescBuilder()
            .setMode(CUDNN_POINTWISE_RELU_FWD)
            .setMathPrecision(CUDNN_DATA_FLOAT)
            .setReluLowerClip(0.0)
            .setReluUpperClip(std::numeric_limits<double>::infinity())
            .setReluLowerClipSlope(0.0)
            .build();
    auto relu_op = mudnn_frontend::OperationBuilder(
                       CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                       .setXDesc(biased)
                       .setYDesc(y)
                       .setPwDesc(relu_desc)
                       .build();
    const mudnn_frontend::Operation* ops[] = {&matmul_op, &add_op, &relu_op};
    auto graph = mudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(3, ops)
                     .build();
    auto heur = mudnn_frontend::EngineHeuristicsBuilder()
                    .setOperationGraph(graph)
                    .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                    .build();
    auto configs = heur.getEngineConfig();
    if (configs.empty()) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    auto plan = mudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(configs[0])
                    .build();

    const std::array<float, 6> a_host = {1.0f, -2.0f, 3.0f, 0.5f, 4.0f, -1.5f};
    const std::array<float, 12> b_host = {2.0f,  -1.0f, 0.0f,  3.0f,
                                          1.5f,  2.0f,  -2.0f, 0.5f,
                                          -4.0f, 1.0f,  3.0f,  -1.0f};
    const std::array<float, 8> bias_host = {0.5f,   -3.0f, 1.0f, -2.0f,
                                            -10.0f, 0.25f, 4.0f, 1.5f};
    const std::array<float, 8> expected = {0.0f, 0.0f,  14.0f, 0.0f,
                                           3.0f, 6.25f, 0.0f,  6.5f};

    CHECK_MUSA(musaMalloc(&a_ptr, sizeof(float) * a_host.size()));
    CHECK_MUSA(musaMalloc(&b_ptr, sizeof(float) * b_host.size()));
    CHECK_MUSA(musaMalloc(&bias_ptr, sizeof(float) * bias_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(a_ptr, a_host.data(), sizeof(float) * a_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(b_ptr, b_host.data(), sizeof(float) * b_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(bias_ptr, bias_host.data(),
                          sizeof(float) * bias_host.size(),
                          musaMemcpyHostToDevice));

    const int64_t workspace_size = plan.getWorkspaceSize();
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }
    void* ptrs[] = {a_ptr, b_ptr, bias_ptr, y_ptr};
    int64_t uids[] = {1, 2, 4, 6};
    auto variant_builder =
        mudnn_frontend::VariantPackBuilder().setUids(4, uids).setDataPointers(
            4, ptrs);
    if (workspace != nullptr) {
      variant_builder.setWorkspacePointer(workspace);
    }
    auto variant_pack = variant_builder.build();
    CHECK_MUDNN(mudnn_frontend::execute(handle, plan, variant_pack));
    SyncMudnnHandleStream(handle);

    std::array<float, 8> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool element_ok = std::fabs(got[i] - expected[i]) < 1.0e-5f;
      if (!element_ok) {
        std::cerr << "mudnn_frontend matmul+bias+relu mismatch at " << i
                  << ": got " << got[i] << ", expected " << expected[i]
                  << std::endl;
      }
      ok = ok && element_ok;
    }

    if (workspace != nullptr) {
      CHECK_MUSA(musaFree(workspace));
    }
    CHECK_MUSA(musaFree(y_ptr));
    CHECK_MUSA(musaFree(bias_ptr));
    CHECK_MUSA(musaFree(b_ptr));
    CHECK_MUSA(musaFree(a_ptr));
    CHECK_MUDNN(mudnnDestroy(handle));
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend matmul+bias+relu threw: " << e.what()
              << std::endl;
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (bias_ptr != nullptr) {
      musaFree(bias_ptr);
    }
    if (b_ptr != nullptr) {
      musaFree(b_ptr);
    }
    if (a_ptr != nullptr) {
      musaFree(a_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  } catch (...) {
    std::cerr << "mudnn_frontend matmul+bias+relu threw unknown exception"
              << std::endl;
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (bias_ptr != nullptr) {
      musaFree(bias_ptr);
    }
    if (b_ptr != nullptr) {
      musaFree(b_ptr);
    }
    if (a_ptr != nullptr) {
      musaFree(a_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  }
}

bool MudnnFrontendGraphApiExecutesMatmulBiasGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  void* bias_ptr = nullptr;
  void* y_ptr = nullptr;
  void* workspace = nullptr;

  try {
    namespace fe = mudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto a = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("a")
                              .set_dim({2, 3})
                              .set_stride({3, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto b = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("b")
                              .set_dim({3, 2})
                              .set_stride({2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto bias = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({2, 2})
                                 .set_stride({2, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto scalar = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("scalar")
                                   .set_dim({1})
                                   .set_stride({1})
                                   .set_data_type(fe::DataType_t::INT64)
                                   .set_uid(99)
                                   .set_is_pass_by_value(true));
    auto matmul =
        graph.matmul(a, b, fe::graph::Matmul_attributes().set_name("matmul"));
    auto y = graph.pointwise(matmul, bias,
                             fe::graph::Pointwise_attributes()
                                 .set_name("bias_add")
                                 .set_mode(fe::PointwiseMode_t::ADD)
                                 .set_compute_data_type(fe::DataType_t::FLOAT));
    y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto serialized_graph = graph.serialize();
    fe::graph::Graph executable_graph;
    auto status = executable_graph.deserialize(serialized_graph);
    if (!status.is_good()) {
      std::cerr << "graph API deserialize failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    auto find_tensor = [&executable_graph](const std::string& name)
        -> std::shared_ptr<fe::graph::Tensor_attributes> {
      for (const auto& tensor : executable_graph.get_tensors()) {
        if (tensor && tensor->get_name() == name) {
          return tensor;
        }
      }
      return nullptr;
    };
    auto a_exec = find_tensor("a");
    auto b_exec = find_tensor("b");
    auto bias_exec = find_tensor("bias");
    auto scalar_exec = find_tensor("scalar");
    auto y_exec = find_tensor("bias_add");
    if (!a_exec || !b_exec || !bias_exec || !scalar_exec || !y_exec) {
      std::cerr << "graph API deserialized tensor lookup failed" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    if (!scalar_exec->is_pass_by_value() || scalar_exec->get_uid() != 99) {
      std::cerr << "graph API pass-by-value tensor metadata was not preserved"
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    status = executable_graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API validate failed: " << status.message << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = executable_graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API build_operation_graph failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = executable_graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API create_execution_plans failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = executable_graph.check_support();
    if (!status.is_good()) {
      std::cerr << "graph API check_support failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    const auto plan_count = executable_graph.get_execution_plan_count();
    if (plan_count < 1 || executable_graph.get_engine_count() != plan_count ||
        !executable_graph.get_knobs_for_engine(0).empty() ||
        executable_graph.get_plan_name_at_index(0) != "mudnn_engine_config_0" ||
        !executable_graph.get_behavior_notes().empty() ||
        !executable_graph.get_behavior_notes_for_plan_at_index(0).empty()) {
      std::cerr << "graph API plan query mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = executable_graph.create_execution_plan(0);
    if (!status.is_good()) {
      std::cerr << "graph API create_execution_plan failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = executable_graph.create_execution_plan(0, {{"KERNEL_CFG", 0}});
    if (status.is_good() ||
        status.message.find("custom execution-plan knobs") ==
            std::string::npos) {
      std::cerr << "graph API create_execution_plan custom knob check failed"
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = executable_graph.build_plan_at_index(plan_count);
    if (status.is_good() ||
        status.message.find("out of range") == std::string::npos) {
      std::cerr << "graph API build_plan_at_index bounds check failed"
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status =
        executable_graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API build_plans failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    const std::array<float, 6> a_host = {1.0f, -2.0f, 3.0f, 4.0f, 5.0f, -6.0f};
    const std::array<float, 6> b_host = {1.0f, 2.0f, -1.0f, 0.5f, 3.0f, -2.0f};
    const std::array<float, 4> bias_host = {0.5f, -1.0f, 2.0f, 3.0f};
    std::array<float, 4> y_host = {};

    CHECK_MUSA(musaMalloc(&a_ptr, sizeof(float) * a_host.size()));
    CHECK_MUSA(musaMalloc(&b_ptr, sizeof(float) * b_host.size()));
    CHECK_MUSA(musaMalloc(&bias_ptr, sizeof(float) * bias_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * y_host.size()));
    CHECK_MUSA(musaMemcpy(a_ptr, a_host.data(), sizeof(float) * a_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(b_ptr, b_host.data(), sizeof(float) * b_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(bias_ptr, bias_host.data(),
                          sizeof(float) * bias_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = executable_graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API get_workspace_size failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    int64_t plan_workspace_size = 0;
    status = executable_graph.get_workspace_size_plan_at_index(
        0, plan_workspace_size);
    if (!status.is_good() || plan_workspace_size != workspace_size) {
      std::cerr << "graph API get_workspace_size_plan_at_index failed: "
                << status.message << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{a_exec, a_ptr},
                        {b_exec, b_ptr},
                        {bias_exec, bias_ptr},
                        {y_exec, y_ptr}};
    status = executable_graph.execute_plan_at_index(handle, variant_pack,
                                                    workspace, 0);
    if (!status.is_good()) {
      std::cerr << "graph API execute_plan_at_index failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    SyncMudnnHandleStream(handle);
    CHECK_MUSA(musaMemcpy(y_host.data(), y_ptr, sizeof(float) * y_host.size(),
                          musaMemcpyDeviceToHost));

    const std::array<float, 4> expected = {
        a_host[0] * b_host[0] + a_host[1] * b_host[2] + a_host[2] * b_host[4] +
            bias_host[0],
        a_host[0] * b_host[1] + a_host[1] * b_host[3] + a_host[2] * b_host[5] +
            bias_host[1],
        a_host[3] * b_host[0] + a_host[4] * b_host[2] + a_host[5] * b_host[4] +
            bias_host[2],
        a_host[3] * b_host[1] + a_host[4] * b_host[3] + a_host[5] * b_host[5] +
            bias_host[3]};
    for (size_t i = 0; i < y_host.size(); ++i) {
      if (std::fabs(y_host[i] - expected[i]) > 1.0e-4f) {
        std::cerr << "graph API numeric mismatch at " << i << ": got "
                  << y_host[i] << ", expected " << expected[i] << std::endl;
        if (workspace != nullptr) {
          musaFree(workspace);
        }
        musaFree(a_ptr);
        musaFree(b_ptr);
        musaFree(bias_ptr);
        musaFree(y_ptr);
        CHECK_MUDNN(mudnnDestroy(handle));
        return false;
      }
    }

    if (workspace != nullptr) {
      CHECK_MUSA(musaFree(workspace));
    }
    CHECK_MUSA(musaFree(a_ptr));
    CHECK_MUSA(musaFree(b_ptr));
    CHECK_MUSA(musaFree(bias_ptr));
    CHECK_MUSA(musaFree(y_ptr));
    CHECK_MUDNN(mudnnDestroy(handle));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API smoke threw: " << e.what()
              << std::endl;
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (a_ptr != nullptr) {
      musaFree(a_ptr);
    }
    if (b_ptr != nullptr) {
      musaFree(b_ptr);
    }
    if (bias_ptr != nullptr) {
      musaFree(bias_ptr);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  }
}

bool MudnnFrontendExecutesMatmulBroadcastBiasGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  void* bias_ptr = nullptr;
  void* y_ptr = nullptr;
  void* workspace = nullptr;

  try {
    auto a = mudnn_frontend::TensorBuilder()
                 .setDim({2, 3})
                 .setStrides({3, 1})
                 .setId(1)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto b = mudnn_frontend::TensorBuilder()
                 .setDim({3, 4})
                 .setStrides({4, 1})
                 .setId(2)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto matmul_y = mudnn_frontend::TensorBuilder()
                        .setDim({2, 4})
                        .setStrides({4, 1})
                        .setId(3)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .setVirtual(true)
                        .build();
    auto bias = mudnn_frontend::TensorBuilder()
                    .setDim({1, 4})
                    .setStrides({4, 1})
                    .setId(4)
                    .setDataType(CUDNN_DATA_FLOAT)
                    .build();
    auto y = mudnn_frontend::TensorBuilder()
                 .setDim({2, 4})
                 .setStrides({4, 1})
                 .setId(5)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();

    auto matmul_desc = mudnn_frontend::MatMulDescBuilder()
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
    auto matmul_op = mudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                         .setMatmulDesc(matmul_desc)
                         .setAMatDesc(a)
                         .setBMatDesc(b)
                         .setCMatDesc(matmul_y)
                         .build();
    auto add_desc = mudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_ADD)
                        .setMathPrecision(CUDNN_DATA_FLOAT)
                        .build();
    auto add_op = mudnn_frontend::OperationBuilder(
                      CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setXDesc(matmul_y)
                      .setBDesc(bias)
                      .setYDesc(y)
                      .setPwDesc(add_desc)
                      .setAlpha(1.0)
                      .setAlpha2(1.0)
                      .build();
    const mudnn_frontend::Operation* ops[] = {&matmul_op, &add_op};
    auto graph = mudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(2, ops)
                     .build();
    auto heur = mudnn_frontend::EngineHeuristicsBuilder()
                    .setOperationGraph(graph)
                    .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                    .build();
    auto configs = heur.getEngineConfig();
    if (configs.empty()) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    auto plan = mudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(configs[0])
                    .build();

    const std::array<float, 6> a_host = {1.0f, -2.0f, 3.0f, 0.5f, 4.0f, -1.5f};
    const std::array<float, 12> b_host = {2.0f,  -1.0f, 0.0f,  3.0f,
                                          1.5f,  2.0f,  -2.0f, 0.5f,
                                          -4.0f, 1.0f,  3.0f,  -1.0f};
    const std::array<float, 4> bias_host = {0.5f, -3.0f, 1.0f, -2.0f};
    const std::array<float, 8> expected = {-12.5f, -5.0f, 14.0f,  -3.0f,
                                           13.5f,  3.0f,  -11.5f, 3.0f};

    CHECK_MUSA(musaMalloc(&a_ptr, sizeof(float) * a_host.size()));
    CHECK_MUSA(musaMalloc(&b_ptr, sizeof(float) * b_host.size()));
    CHECK_MUSA(musaMalloc(&bias_ptr, sizeof(float) * bias_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(a_ptr, a_host.data(), sizeof(float) * a_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(b_ptr, b_host.data(), sizeof(float) * b_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(bias_ptr, bias_host.data(),
                          sizeof(float) * bias_host.size(),
                          musaMemcpyHostToDevice));

    const int64_t workspace_size = plan.getWorkspaceSize();
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }
    auto variant_builder =
        mudnn_frontend::VariantPackBuilder()
            .setUids({1, 2, 4, 5})
            .setDataPointers({a_ptr, b_ptr, bias_ptr, y_ptr});
    if (workspace != nullptr) {
      variant_builder.setWorkspacePointer(workspace);
    }
    auto variant_pack = variant_builder.build();
    CHECK_MUDNN(mudnn_frontend::execute(handle, plan, variant_pack));
    SyncMudnnHandleStream(handle);

    std::array<float, 8> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool element_ok = std::fabs(got[i] - expected[i]) < 1.0e-5f;
      if (!element_ok) {
        std::cerr << "mudnn_frontend matmul+broadcast-bias mismatch at " << i
                  << ": got " << got[i] << ", expected " << expected[i]
                  << std::endl;
      }
      ok = ok && element_ok;
    }

    if (workspace != nullptr) {
      CHECK_MUSA(musaFree(workspace));
    }
    CHECK_MUSA(musaFree(y_ptr));
    CHECK_MUSA(musaFree(bias_ptr));
    CHECK_MUSA(musaFree(b_ptr));
    CHECK_MUSA(musaFree(a_ptr));
    CHECK_MUDNN(mudnnDestroy(handle));
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend matmul+broadcast-bias threw: " << e.what()
              << std::endl;
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (bias_ptr != nullptr) {
      musaFree(bias_ptr);
    }
    if (b_ptr != nullptr) {
      musaFree(b_ptr);
    }
    if (a_ptr != nullptr) {
      musaFree(a_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  } catch (...) {
    std::cerr << "mudnn_frontend matmul+broadcast-bias threw unknown exception"
              << std::endl;
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (bias_ptr != nullptr) {
      musaFree(bias_ptr);
    }
    if (b_ptr != nullptr) {
      musaFree(b_ptr);
    }
    if (a_ptr != nullptr) {
      musaFree(a_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  }
}

}  // namespace mudnn_frontend_tests
