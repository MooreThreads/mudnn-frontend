/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#include "frontend_api_common.h"

namespace mudnn_frontend_tests {

bool MudnnFrontendGraphApiExecutesPointwiseAliasesGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  void* true_ptr = nullptr;
  void* false_ptr = nullptr;
  void* y_ptr = nullptr;
  void* workspace = nullptr;

  auto cleanup = [&]() {
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (false_ptr != nullptr) {
      musaFree(false_ptr);
    }
    if (true_ptr != nullptr) {
      musaFree(true_ptr);
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
  };

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
                              .set_dim({2, 3})
                              .set_stride({3, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto true_value = graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("true_value")
                                       .set_dim({2, 3})
                                       .set_stride({3, 1})
                                       .set_data_type(fe::DataType_t::FLOAT));
    auto false_value = graph.tensor(fe::graph::Tensor_attributes()
                                        .set_name("false_value")
                                        .set_dim({2, 3})
                                        .set_stride({3, 1})
                                        .set_data_type(fe::DataType_t::FLOAT));
    auto pred_gt =
        graph.gt(a, b, fe::graph::Pointwise_attributes().set_name("pred_gt"));
    auto pred_eq =
        graph.eq(a, b, fe::graph::Pointwise_attributes().set_name("pred_eq"));
    auto pred_lt =
        graph.lt(a, b, fe::graph::Pointwise_attributes().set_name("pred_lt"));
    auto pred_le =
        graph.logical_or(pred_eq, pred_lt,
                         fe::graph::Pointwise_attributes().set_name("pred_le"));
    auto pred_not_le = graph.logical_not(
        pred_le, fe::graph::Pointwise_attributes().set_name("pred_not_le"));
    auto pred =
        graph.logical_and(pred_gt, pred_not_le,
                          fe::graph::Pointwise_attributes().set_name("pred"));
    if (pred_gt->get_data_type() != MUDNN_DATA_BOOLEAN ||
        pred_eq->get_data_type() != MUDNN_DATA_BOOLEAN ||
        pred_lt->get_data_type() != MUDNN_DATA_BOOLEAN ||
        pred_le->get_data_type() != MUDNN_DATA_BOOLEAN ||
        pred_not_le->get_data_type() != MUDNN_DATA_BOOLEAN ||
        pred->get_data_type() != MUDNN_DATA_BOOLEAN) {
      std::cerr << "graph API comparison/logical output type mismatch"
                << std::endl;
      cleanup();
      return false;
    }
    auto y =
        graph.select(pred, true_value, false_value,
                     fe::graph::Pointwise_attributes().set_name("selected"));
    y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API pointwise aliases validate failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API pointwise aliases build_operation_graph failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API pointwise aliases create_execution_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API pointwise aliases build_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }

    const std::array<float, 6> a_host = {1.0f, -2.0f, 3.0f, 0.25f, 4.0f, -5.0f};
    const std::array<float, 6> b_host = {0.5f, -3.0f, 4.0f, 0.5f, 2.0f, -6.0f};
    const std::array<float, 6> true_host = {10.0f, 20.0f, 30.0f,
                                            40.0f, 50.0f, 60.0f};
    const std::array<float, 6> false_host = {-10.0f, -20.0f, -30.0f,
                                             -40.0f, -50.0f, -60.0f};
    const std::array<float, 6> expected = {10.0f,  20.0f, -30.0f,
                                           -40.0f, 50.0f, 60.0f};

    CHECK_MUSA(musaMalloc(&a_ptr, sizeof(float) * a_host.size()));
    CHECK_MUSA(musaMalloc(&b_ptr, sizeof(float) * b_host.size()));
    CHECK_MUSA(musaMalloc(&true_ptr, sizeof(float) * true_host.size()));
    CHECK_MUSA(musaMalloc(&false_ptr, sizeof(float) * false_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(a_ptr, a_host.data(), sizeof(float) * a_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(b_ptr, b_host.data(), sizeof(float) * b_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(true_ptr, true_host.data(),
                          sizeof(float) * true_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(false_ptr, false_host.data(),
                          sizeof(float) * false_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API pointwise aliases get_workspace_size failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{a, a_ptr},
                        {b, b_ptr},
                        {true_value, true_ptr},
                        {false_value, false_ptr},
                        {y, y_ptr}};
    status = graph.execute_plan_at_index(handle, variant_pack, workspace, 0);
    if (!status.is_good()) {
      std::cerr << "graph API pointwise aliases execute failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<float, 6> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool element_ok = std::fabs(got[i] - expected[i]) < 1.0e-5f;
      if (!element_ok) {
        std::cerr << "graph API pointwise alias mismatch at " << i << ": got "
                  << got[i] << ", expected " << expected[i] << std::endl;
      }
      ok = ok && element_ok;
    }
    cleanup();
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API pointwise aliases threw: "
              << e.what() << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend graph API pointwise aliases threw unknown "
                 "exception"
              << std::endl;
  }
  cleanup();
  return false;
}

bool MudnnFrontendGraphApiExecutesGenIndexGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* x_ptr = nullptr;
  void* y_ptr = nullptr;
  void* workspace = nullptr;

  auto cleanup = [&]() {
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (x_ptr != nullptr) {
      musaFree(x_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
  };

  try {
    namespace fe = mudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto x = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("x")
                              .set_dim({2, 3, 4})
                              .set_stride({12, 4, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto y = graph.gen_index(
        x, 1, fe::graph::Pointwise_attributes().set_name("row_index"));
    if (y->get_data_type() != MUDNN_DATA_INT32) {
      std::cerr << "graph API gen_index output type mismatch" << std::endl;
      cleanup();
      return false;
    }
    y->set_output(true).set_data_type(fe::DataType_t::INT32);

    auto serialized_graph = graph.serialize();
    fe::graph::Graph executable_graph;
    auto status = executable_graph.deserialize(serialized_graph);
    if (!status.is_good()) {
      std::cerr << "graph API gen_index deserialize failed: " << status.message
                << std::endl;
      cleanup();
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
    auto x_exec = find_tensor("x");
    auto y_exec = find_tensor("row_index");
    if (!x_exec || !y_exec || y_exec->get_data_type() != MUDNN_DATA_INT32) {
      std::cerr << "graph API gen_index deserialized tensor mismatch"
                << std::endl;
      cleanup();
      return false;
    }

    status = executable_graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API gen_index validate failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    status = executable_graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API gen_index build_operation_graph failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = executable_graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API gen_index create_execution_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status =
        executable_graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API gen_index build_plans failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }

    std::array<float, 24> x_host{};
    std::array<int32_t, 24> expected{};
    for (size_t i = 0; i < x_host.size(); ++i) {
      x_host[i] = static_cast<float>(i);
      expected[i] = static_cast<int32_t>((i / 4) % 3);
    }
    CHECK_MUSA(musaMalloc(&x_ptr, sizeof(float) * x_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(int32_t) * expected.size()));
    CHECK_MUSA(musaMemcpy(x_ptr, x_host.data(), sizeof(float) * x_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = executable_graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API gen_index get_workspace_size failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{x_exec, x_ptr}, {y_exec, y_ptr}};
    status = executable_graph.execute_plan_at_index(handle, variant_pack,
                                                    workspace, 0);
    if (!status.is_good()) {
      std::cerr << "graph API gen_index execute failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<int32_t, 24> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(int32_t) * got.size(),
                          musaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool element_ok = got[i] == expected[i];
      if (!element_ok) {
        std::cerr << "graph API gen_index mismatch at " << i << ": got "
                  << got[i] << ", expected " << expected[i] << std::endl;
      }
      ok = ok && element_ok;
    }
    cleanup();
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API gen_index threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend graph API gen_index threw unknown exception"
              << std::endl;
  }
  cleanup();
  return false;
}

bool MudnnFrontendGraphApiExecutesSerializedScaleGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* x_ptr = nullptr;
  void* y_ptr = nullptr;
  void* workspace = nullptr;

  auto cleanup = [&]() {
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (x_ptr != nullptr) {
      musaFree(x_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
  };

  try {
    namespace fe = mudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto x = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("x")
                              .set_dim({2, 3})
                              .set_stride({3, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto y = graph.scale(x, -1.5,
                         fe::graph::Pointwise_attributes().set_name("scaled"));
    y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto serialized_graph = graph.serialize();
    fe::graph::Graph executable_graph;
    auto status = executable_graph.deserialize(serialized_graph);
    if (!status.is_good()) {
      std::cerr << "graph API scale deserialize failed: " << status.message
                << std::endl;
      cleanup();
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
    auto x_exec = find_tensor("x");
    auto y_exec = find_tensor("scaled");
    if (!x_exec || !y_exec) {
      std::cerr << "graph API scale deserialized tensor mismatch" << std::endl;
      cleanup();
      return false;
    }

    status = executable_graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API scale validate failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    status = executable_graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API scale build_operation_graph failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = executable_graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API scale create_execution_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status =
        executable_graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API scale build_plans failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }

    const std::array<float, 6> x_host = {1.0f, -2.0f, 3.0f, 0.25f, 4.0f, -5.0f};
    const std::array<float, 6> expected = {-1.5f,   3.0f,  -4.5f,
                                           -0.375f, -6.0f, 7.5f};
    CHECK_MUSA(musaMalloc(&x_ptr, sizeof(float) * x_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(x_ptr, x_host.data(), sizeof(float) * x_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = executable_graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API scale get_workspace_size failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{x_exec, x_ptr}, {y_exec, y_ptr}};
    status = executable_graph.execute_plan_at_index(handle, variant_pack,
                                                    workspace, 0);
    if (!status.is_good()) {
      std::cerr << "graph API scale execute failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<float, 6> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool element_ok = std::fabs(got[i] - expected[i]) < 1.0e-6f;
      if (!element_ok) {
        std::cerr << "graph API scale mismatch at " << i << ": got " << got[i]
                  << ", expected " << expected[i] << std::endl;
      }
      ok = ok && element_ok;
    }
    cleanup();
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API scale threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend graph API scale threw unknown exception"
              << std::endl;
  }
  cleanup();
  return false;
}

bool MudnnFrontendExecutesCompareSelectGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  void* true_ptr = nullptr;
  void* false_ptr = nullptr;
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
                 .setDim({2, 3})
                 .setStrides({3, 1})
                 .setId(2)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto pred = mudnn_frontend::TensorBuilder()
                    .setDim({2, 3})
                    .setStrides({3, 1})
                    .setId(3)
                    .setDataType(CUDNN_DATA_BOOLEAN)
                    .setVirtual(true)
                    .build();
    auto true_value = mudnn_frontend::TensorBuilder()
                          .setDim({2, 3})
                          .setStrides({3, 1})
                          .setId(4)
                          .setDataType(CUDNN_DATA_FLOAT)
                          .build();
    auto false_value = mudnn_frontend::TensorBuilder()
                           .setDim({2, 3})
                           .setStrides({3, 1})
                           .setId(5)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();
    auto y = mudnn_frontend::TensorBuilder()
                 .setDim({2, 3})
                 .setStrides({3, 1})
                 .setId(6)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();

    auto gt_desc = mudnn_frontend::PointWiseDescBuilder()
                       .setMode(CUDNN_POINTWISE_CMP_GT)
                       .setMathPrecision(CUDNN_DATA_FLOAT)
                       .build();
    auto gt_op = mudnn_frontend::OperationBuilder(
                     CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                     .setXDesc(a)
                     .setBDesc(b)
                     .setYDesc(pred)
                     .setPwDesc(gt_desc)
                     .setAlpha(1.0)
                     .setAlpha2(1.0)
                     .build();
    auto select_desc = mudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_BINARY_SELECT)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
    auto select_op = mudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setXDesc(true_value)
                         .setBDesc(false_value)
                         .setTDesc(pred)
                         .setYDesc(y)
                         .setPwDesc(select_desc)
                         .setAlpha(1.0)
                         .setAlpha2(1.0)
                         .build();
    const mudnn_frontend::Operation* ops[] = {&gt_op, &select_op};
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

    const std::array<float, 6> a_host = {1.0f, -2.0f, 3.0f, 0.25f, 4.0f, -5.0f};
    const std::array<float, 6> b_host = {0.5f, -3.0f, 4.0f, 0.5f, 2.0f, -6.0f};
    const std::array<float, 6> true_host = {10.0f, 20.0f, 30.0f,
                                            40.0f, 50.0f, 60.0f};
    const std::array<float, 6> false_host = {-10.0f, -20.0f, -30.0f,
                                             -40.0f, -50.0f, -60.0f};
    const std::array<float, 6> expected = {10.0f,  20.0f, -30.0f,
                                           -40.0f, 50.0f, 60.0f};

    CHECK_MUSA(musaMalloc(&a_ptr, sizeof(float) * a_host.size()));
    CHECK_MUSA(musaMalloc(&b_ptr, sizeof(float) * b_host.size()));
    CHECK_MUSA(musaMalloc(&true_ptr, sizeof(float) * true_host.size()));
    CHECK_MUSA(musaMalloc(&false_ptr, sizeof(float) * false_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(a_ptr, a_host.data(), sizeof(float) * a_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(b_ptr, b_host.data(), sizeof(float) * b_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(true_ptr, true_host.data(),
                          sizeof(float) * true_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(false_ptr, false_host.data(),
                          sizeof(float) * false_host.size(),
                          musaMemcpyHostToDevice));

    const int64_t workspace_size = plan.getWorkspaceSize();
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }
    auto variant_builder =
        mudnn_frontend::VariantPackBuilder()
            .setUids({1, 2, 4, 5, 6})
            .setDataPointers({a_ptr, b_ptr, true_ptr, false_ptr, y_ptr});
    if (workspace != nullptr) {
      variant_builder.setWorkspacePointer(workspace);
    }
    auto variant_pack = variant_builder.build();
    CHECK_MUDNN(mudnn_frontend::execute(handle, plan, variant_pack));
    SyncMudnnHandleStream(handle);

    std::array<float, 6> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool element_ok = std::fabs(got[i] - expected[i]) < 1.0e-5f;
      if (!element_ok) {
        std::cerr << "mudnn_frontend compare+select mismatch at " << i
                  << ": got " << got[i] << ", expected " << expected[i]
                  << std::endl;
      }
      ok = ok && element_ok;
    }

    if (workspace != nullptr) {
      CHECK_MUSA(musaFree(workspace));
    }
    CHECK_MUSA(musaFree(y_ptr));
    CHECK_MUSA(musaFree(false_ptr));
    CHECK_MUSA(musaFree(true_ptr));
    CHECK_MUSA(musaFree(b_ptr));
    CHECK_MUSA(musaFree(a_ptr));
    CHECK_MUDNN(mudnnDestroy(handle));
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend compare+select threw: " << e.what()
              << std::endl;
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (false_ptr != nullptr) {
      musaFree(false_ptr);
    }
    if (true_ptr != nullptr) {
      musaFree(true_ptr);
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
    std::cerr << "mudnn_frontend compare+select threw unknown exception"
              << std::endl;
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (false_ptr != nullptr) {
      musaFree(false_ptr);
    }
    if (true_ptr != nullptr) {
      musaFree(true_ptr);
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

bool MudnnFrontendExecutesScaleGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* x_ptr = nullptr;
  void* y_ptr = nullptr;

  try {
    auto x = mudnn_frontend::TensorBuilder()
                 .setDim({2, 3})
                 .setStrides({3, 1})
                 .setId(1)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto y = mudnn_frontend::TensorBuilder()
                 .setDim({2, 3})
                 .setStrides({3, 1})
                 .setId(2)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto scale_desc = mudnn_frontend::PointWiseDescBuilder()
                          .setMode(CUDNN_POINTWISE_IDENTITY)
                          .setMathPrecision(CUDNN_DATA_FLOAT)
                          .build();
    auto scale_op = mudnn_frontend::OperationBuilder(
                        CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setXDesc(x)
                        .setYDesc(y)
                        .setPwDesc(scale_desc)
                        .setAlpha(-1.5)
                        .build();
    const mudnn_frontend::Operation* ops[] = {&scale_op};
    auto graph = mudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(1, ops)
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

    const std::array<float, 6> x_host = {1.0f, -2.0f, 3.0f, 0.25f, 4.0f, -5.0f};
    const std::array<float, 6> expected = {-1.5f,   3.0f,  -4.5f,
                                           -0.375f, -6.0f, 7.5f};
    CHECK_MUSA(musaMalloc(&x_ptr, sizeof(float) * x_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(x_ptr, x_host.data(), sizeof(float) * x_host.size(),
                          musaMemcpyHostToDevice));

    auto variant_pack = mudnn_frontend::VariantPackBuilder()
                            .setUids({1, 2})
                            .setDataPointers({x_ptr, y_ptr})
                            .build();
    CHECK_MUDNN(mudnn_frontend::execute(handle, plan, variant_pack));
    SyncMudnnHandleStream(handle);

    std::array<float, 6> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool element_ok = std::fabs(got[i] - expected[i]) < 1.0e-6f;
      if (!element_ok) {
        std::cerr << "mudnn_frontend scale mismatch at " << i << ": got "
                  << got[i] << ", expected " << expected[i] << std::endl;
      }
      ok = ok && element_ok;
    }

    CHECK_MUSA(musaFree(y_ptr));
    CHECK_MUSA(musaFree(x_ptr));
    CHECK_MUDNN(mudnnDestroy(handle));
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend scale threw: " << e.what() << std::endl;
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (x_ptr != nullptr) {
      musaFree(x_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  } catch (...) {
    std::cerr << "mudnn_frontend scale threw unknown exception" << std::endl;
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (x_ptr != nullptr) {
      musaFree(x_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  }
}

}  // namespace mudnn_frontend_tests
