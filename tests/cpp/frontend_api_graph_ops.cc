/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#include "frontend_api_common.h"

namespace mudnn_frontend_tests {

bool MudnnFrontendGraphApiExecutesConcatReshapeGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* a_ptr = nullptr;
  void* b_ptr = nullptr;
  void* y_ptr = nullptr;
  void* workspace = nullptr;

  auto cleanup = [&]() {
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
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
                              .set_dim({2, 2})
                              .set_stride({2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto b = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("b")
                              .set_dim({2, 3})
                              .set_stride({3, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto concat = graph.concat({a, b}, -1, "concat");
    auto y = graph.reshape(concat, {5, 2}, "reshaped");
    y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API concat/reshape validate failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API concat/reshape build_operation_graph failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API concat/reshape create_execution_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API concat/reshape build_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }

    const std::array<float, 4> a_host = {1.0f, 2.0f, 6.0f, 7.0f};
    const std::array<float, 6> b_host = {3.0f, 4.0f, 5.0f, 8.0f, 9.0f, 10.0f};
    const std::array<float, 10> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                            6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    CHECK_MUSA(musaMalloc(&a_ptr, sizeof(float) * a_host.size()));
    CHECK_MUSA(musaMalloc(&b_ptr, sizeof(float) * b_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(a_ptr, a_host.data(), sizeof(float) * a_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(b_ptr, b_host.data(), sizeof(float) * b_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API concat/reshape get_workspace_size failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{a, a_ptr}, {b, b_ptr}, {y, y_ptr}};
    status = graph.execute_plan_at_index(handle, variant_pack, workspace, 0);
    if (!status.is_good()) {
      std::cerr << "graph API concat/reshape execute failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<float, 10> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool element_ok = std::fabs(got[i] - expected[i]) < 1.0e-6f;
      if (!element_ok) {
        std::cerr << "graph API concat/reshape mismatch at " << i << ": got "
                  << got[i] << ", expected " << expected[i] << std::endl;
      }
      ok = ok && element_ok;
    }
    cleanup();
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API concat/reshape threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend graph API concat/reshape threw unknown "
                 "exception"
              << std::endl;
  }
  cleanup();
  return false;
}

bool MudnnFrontendGraphApiExecutesReductionGraph() {
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
    auto y =
        graph.sum(x, -1, fe::graph::Reduction_attributes().set_name("sum"));
    y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API reduction validate failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API reduction build_operation_graph failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API reduction create_execution_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API reduction build_plans failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }

    const std::array<float, 6> x_host = {1.0f, -2.0f, 3.0f, 4.5f, 0.5f, -1.0f};
    const std::array<float, 2> expected = {2.0f, 4.0f};
    CHECK_MUSA(musaMalloc(&x_ptr, sizeof(float) * x_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(x_ptr, x_host.data(), sizeof(float) * x_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API reduction get_workspace_size failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{x, x_ptr}, {y, y_ptr}};
    status = graph.execute_plan_at_index(handle, variant_pack, workspace, 0);
    if (!status.is_good()) {
      std::cerr << "graph API reduction execute failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<float, 2> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      const bool element_ok = std::fabs(got[i] - expected[i]) < 1.0e-6f;
      if (!element_ok) {
        std::cerr << "graph API reduction mismatch at " << i << ": got "
                  << got[i] << ", expected " << expected[i] << std::endl;
      }
      ok = ok && element_ok;
    }
    cleanup();
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API reduction threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend graph API reduction threw unknown exception"
              << std::endl;
  }
  cleanup();
  return false;
}

bool MudnnFrontendGraphApiExecutesGenstatsGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* x_ptr = nullptr;
  void* sum_ptr = nullptr;
  void* sqsum_ptr = nullptr;
  void* workspace = nullptr;

  auto cleanup = [&]() {
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (sqsum_ptr != nullptr) {
      musaFree(sqsum_ptr);
    }
    if (sum_ptr != nullptr) {
      musaFree(sum_ptr);
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
                              .set_dim({2, 2, 2})
                              .set_stride({4, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto outputs =
        graph.genstats(x, std::vector<int64_t>{0, -1},
                       fe::graph::Genstats_attributes().set_name("stats"));
    auto sum = outputs[0];
    auto sqsum = outputs[1];
    sum->set_output(true).set_data_type(fe::DataType_t::FLOAT);
    sqsum->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API genstats validate failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API genstats build_operation_graph failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API genstats create_execution_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API genstats build_plans failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }

    const std::array<float, 8> x_host = {1.0f,  -2.0f, 3.0f, 0.5f,
                                         -1.5f, 2.0f,  4.0f, -0.25f};
    const std::array<float, 2> expected_sum = {-0.5f, 7.25f};
    const std::array<float, 2> expected_sqsum = {11.25f, 25.3125f};
    CHECK_MUSA(musaMalloc(&x_ptr, sizeof(float) * x_host.size()));
    CHECK_MUSA(musaMalloc(&sum_ptr, sizeof(float) * expected_sum.size()));
    CHECK_MUSA(musaMalloc(&sqsum_ptr, sizeof(float) * expected_sqsum.size()));
    CHECK_MUSA(musaMemcpy(x_ptr, x_host.data(), sizeof(float) * x_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API genstats get_workspace_size failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{x, x_ptr}, {sum, sum_ptr}, {sqsum, sqsum_ptr}};
    status = graph.execute_plan_at_index(handle, variant_pack, workspace, 0);
    if (!status.is_good()) {
      std::cerr << "graph API genstats execute failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<float, 2> got_sum{};
    std::array<float, 2> got_sqsum{};
    CHECK_MUSA(musaMemcpy(got_sum.data(), sum_ptr,
                          sizeof(float) * got_sum.size(),
                          musaMemcpyDeviceToHost));
    CHECK_MUSA(musaMemcpy(got_sqsum.data(), sqsum_ptr,
                          sizeof(float) * got_sqsum.size(),
                          musaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < got_sum.size(); ++i) {
      const bool sum_ok = std::fabs(got_sum[i] - expected_sum[i]) < 1.0e-5f;
      const bool sqsum_ok =
          std::fabs(got_sqsum[i] - expected_sqsum[i]) < 1.0e-5f;
      if (!sum_ok || !sqsum_ok) {
        std::cerr << "graph API genstats mismatch at " << i << ": got sum "
                  << got_sum[i] << ", sqsum " << got_sqsum[i]
                  << "; expected sum " << expected_sum[i] << ", sqsum "
                  << expected_sqsum[i] << std::endl;
      }
      ok = ok && sum_ok && sqsum_ok;
    }
    cleanup();
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API genstats threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend graph API genstats threw unknown exception"
              << std::endl;
  }
  cleanup();
  return false;
}

bool MudnnFrontendGraphApiExecutesRngGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* offset_ptr = nullptr;
  void* y_ptr = nullptr;
  void* workspace = nullptr;

  auto cleanup = [&]() {
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (y_ptr != nullptr) {
      musaFree(y_ptr);
    }
    if (offset_ptr != nullptr) {
      musaFree(offset_ptr);
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

    auto offset = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("offset")
                                   .set_dim({1})
                                   .set_stride({1})
                                   .set_data_type(fe::DataType_t::INT64));
    auto y = graph.rng({2, 4},
                       fe::graph::Rng_attributes()
                           .set_name("rng")
                           .set_seed(1234)
                           .set_distribution(fe::RngDistribution_t::UNIFORM)
                           .set_uniform_range(-0.25, 0.75),
                       offset);
    y->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API rng validate failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API rng build_operation_graph failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API rng create_execution_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API rng build_plans failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }

    const std::array<int64_t, 1> offset_host = {3};
    CHECK_MUSA(musaMalloc(&offset_ptr, sizeof(int64_t) * offset_host.size()));
    CHECK_MUSA(musaMalloc(&y_ptr, sizeof(float) * 8));
    CHECK_MUSA(musaMemcpy(offset_ptr, offset_host.data(),
                          sizeof(int64_t) * offset_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API rng get_workspace_size failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{offset, offset_ptr}, {y, y_ptr}};
    status = graph.execute_plan_at_index(handle, variant_pack, workspace, 0);
    if (!status.is_good()) {
      std::cerr << "graph API rng execute failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<float, 8> got{};
    CHECK_MUSA(musaMemcpy(got.data(), y_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));
    bool in_range = true;
    bool has_variation = false;
    for (size_t i = 0; i < got.size(); ++i) {
      in_range = in_range && got[i] >= -0.25f && got[i] < 0.75f;
      has_variation = has_variation || std::fabs(got[i] - got[0]) > 1.0e-7f;
    }
    if (!in_range || !has_variation) {
      std::cerr << "graph API rng output failed range/variation check"
                << std::endl;
    }
    cleanup();
    return in_range && has_variation;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API rng threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend graph API rng threw unknown exception"
              << std::endl;
  }
  cleanup();
  return false;
}

bool MudnnFrontendGraphApiBuildsBatchnormGraphs() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));
  try {
    namespace fe = mudnn_frontend;

    auto finalize_graph = [&](fe::graph::Graph& graph) {
      auto serialized = graph.serialize();
      fe::graph::Graph restored;
      auto status = restored.deserialize(serialized);
      if (!status.is_good()) {
        std::cerr << "batchnorm graph deserialize failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.validate();
      if (!status.is_good()) {
        std::cerr << "batchnorm graph validate failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.build_operation_graph(handle);
      if (!status.is_good()) {
        std::cerr << "batchnorm graph build_operation_graph failed: "
                  << status.message << std::endl;
        return false;
      }
      return true;
    };

    fe::graph::Graph inference_graph;
    inference_graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto inf_param =
        inference_graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("param")
                                   .set_dim({1, 2, 1, 1})
                                   .set_stride({2, 1, 1, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));
    auto inf_outputs = inference_graph.bn_finalize_inference(
        inf_param, inf_param, inf_param, inf_param,
        fe::graph::Bn_finalize_attributes()
            .set_name("bn_finalize_inference")
            .set_epsilon(1.0e-5)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    inf_outputs[0]->set_output(true);
    inf_outputs[1]->set_output(true);
    if (!finalize_graph(inference_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph training_graph;
    training_graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto train_param =
        training_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("param")
                                  .set_dim({1, 2, 1, 1})
                                  .set_stride({2, 1, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto accum_count =
        training_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("accum_count")
                                  .set_dim({1})
                                  .set_stride({1})
                                  .set_data_type(fe::DataType_t::INT64));
    auto train_outputs = training_graph.bn_finalize_training(
        train_param, train_param, train_param, train_param, train_param,
        train_param, accum_count,
        fe::graph::Bn_finalize_attributes()
            .set_name("bn_finalize_training")
            .set_momentum(0.25)
            .set_epsilon(1.0e-5)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    std::get<0>(train_outputs)->set_output(true);
    std::get<1>(train_outputs)->set_output(true);
    std::get<2>(train_outputs)->set_output(true);
    std::get<3>(train_outputs)->set_output(true);
    std::get<4>(train_outputs)->set_output(true);
    std::get<5>(train_outputs)->set_output(true);
    if (!finalize_graph(training_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph bwd_graph;
    bwd_graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto x = bwd_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("x")
                                  .set_dim({2, 2, 2, 2})
                                  .set_stride({8, 4, 2, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto dy = bwd_graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("dy")
                                   .set_dim({2, 2, 2, 2})
                                   .set_stride({8, 4, 2, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));
    auto bwd_param =
        bwd_graph.tensor(fe::graph::Tensor_attributes()
                             .set_name("param")
                             .set_dim({1, 2, 1, 1})
                             .set_stride({2, 1, 1, 1})
                             .set_data_type(fe::DataType_t::FLOAT));
    auto bwd_outputs = bwd_graph.bn_backward_weights(
        x, dy, bwd_param, bwd_param, bwd_param,
        fe::graph::Bn_backward_weights_attributes()
            .set_name("bn_backward_weights")
            .set_compute_data_type(fe::DataType_t::FLOAT));
    std::get<0>(bwd_outputs)->set_output(true);
    std::get<1>(bwd_outputs)->set_output(true);
    std::get<2>(bwd_outputs)->set_output(true);
    std::get<3>(bwd_outputs)->set_output(true);
    std::get<4>(bwd_outputs)->set_output(true);
    if (!finalize_graph(bwd_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    CHECK_MUDNN(mudnnDestroy(handle));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend batchnorm graph API threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend batchnorm graph API threw unknown exception"
              << std::endl;
  }
  if (handle != nullptr) {
    mudnnDestroy(handle);
  }
  return false;
}

bool MudnnFrontendGraphApiBuildsConvResampleGraphs() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));
  try {
    namespace fe = mudnn_frontend;

    auto finalize_graph = [&](const char* label, fe::graph::Graph& graph) {
      auto serialized = graph.serialize();
      fe::graph::Graph restored;
      auto status = restored.deserialize(serialized);
      if (!status.is_good()) {
        std::cerr << label << " graph deserialize failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.validate();
      if (!status.is_good()) {
        std::cerr << label << " graph validate failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.build_operation_graph(handle);
      if (!status.is_good()) {
        std::cerr << label
                  << " graph build_operation_graph failed: " << status.message
                  << std::endl;
        return false;
      }
      return true;
    };

    auto make_graph = []() {
      fe::graph::Graph graph;
      graph.set_io_data_type(fe::DataType_t::FLOAT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
      return graph;
    };

    fe::graph::Graph conv_graph = make_graph();
    auto x = conv_graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("x")
                                   .set_dim({1, 2, 4, 4})
                                   .set_stride({32, 16, 4, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));
    auto w = conv_graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("w")
                                   .set_dim({3, 2, 3, 3})
                                   .set_stride({18, 9, 3, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));
    auto y = conv_graph.conv_fprop(
        x, w,
        fe::graph::Conv_attributes()
            .set_name("conv")
            .set_padding({1, 1})
            .set_stride({1, 1})
            .set_dilation({1, 1})
            .set_compute_data_type(fe::DataType_t::FLOAT));
    if (y->get_dim() != std::vector<int64_t>({1, 3, 4, 4})) {
      std::cerr << "conv graph output dims mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    y->set_output(true);
    if (!finalize_graph("conv", conv_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph dgrad_graph = make_graph();
    auto dy = dgrad_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("dy")
                                     .set_dim({1, 3, 4, 4})
                                     .set_stride({48, 16, 4, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));
    auto dgrad_w =
        dgrad_graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("w")
                               .set_dim({3, 2, 3, 3})
                               .set_stride({18, 9, 3, 1})
                               .set_data_type(fe::DataType_t::FLOAT));
    auto dx = dgrad_graph.conv_dgrad(
        dy, dgrad_w, {1, 2, 4, 4},
        fe::graph::Conv_attributes()
            .set_name("dgrad")
            .set_padding({1, 1})
            .set_compute_data_type(fe::DataType_t::FLOAT));
    dx->set_output(true);
    if (!finalize_graph("conv_dgrad", dgrad_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph wgrad_graph = make_graph();
    auto wgrad_x =
        wgrad_graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("x")
                               .set_dim({1, 2, 4, 4})
                               .set_stride({32, 16, 4, 1})
                               .set_data_type(fe::DataType_t::FLOAT));
    auto wgrad_dy =
        wgrad_graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("dy")
                               .set_dim({1, 3, 4, 4})
                               .set_stride({48, 16, 4, 1})
                               .set_data_type(fe::DataType_t::FLOAT));
    auto dw = wgrad_graph.conv_wgrad(
        wgrad_x, wgrad_dy, {3, 2, 3, 3},
        fe::graph::Conv_attributes()
            .set_name("wgrad")
            .set_padding({1, 1})
            .set_compute_data_type(fe::DataType_t::FLOAT));
    dw->set_output(true);
    if (!finalize_graph("conv_wgrad", wgrad_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph pool_graph = make_graph();
    auto pool_x = pool_graph.tensor(fe::graph::Tensor_attributes()
                                        .set_name("pool_x")
                                        .set_dim({1, 1, 4, 4})
                                        .set_stride({16, 16, 4, 1})
                                        .set_data_type(fe::DataType_t::FLOAT));
    auto avg =
        pool_graph.avg_pool(pool_x,
                            fe::graph::Resample_attributes()
                                .set_name("avg")
                                .set_window({2, 2})
                                .set_stride({2, 2})
                                .set_compute_data_type(fe::DataType_t::FLOAT));
    avg->set_output(true);
    if (!finalize_graph("pool_avg", pool_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph max_pool_graph = make_graph();
    auto max_pool_x =
        max_pool_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("pool_x")
                                  .set_dim({1, 1, 4, 4})
                                  .set_stride({16, 16, 4, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto max_and_idx = max_pool_graph.max_pool_with_indices(
        max_pool_x,
        fe::graph::Resample_attributes()
            .set_name("max")
            .set_window({2, 2})
            .set_stride({1, 2})
            .set_compute_data_type(fe::DataType_t::FLOAT));
    max_and_idx[0]->set_output(true);
    max_and_idx[1]->set_output(true);
    if (!finalize_graph("pool_max", max_pool_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph pool_bwd_graph = make_graph();
    auto pool_bwd_x =
        pool_bwd_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("x")
                                  .set_dim({1, 1, 4, 4})
                                  .set_stride({16, 16, 4, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto pool_bwd_dy =
        pool_bwd_graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("dy")
                                  .set_dim({1, 1, 3, 2})
                                  .set_stride({6, 6, 2, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto pool_dx = pool_bwd_graph.max_pool_bwd(
        pool_bwd_x, pool_bwd_dy,
        fe::graph::Resample_attributes()
            .set_name("max_bwd")
            .set_window({2, 2})
            .set_stride({1, 2})
            .set_compute_data_type(fe::DataType_t::FLOAT));
    pool_dx->set_output(true);
    if (!finalize_graph("pool_bwd", pool_bwd_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    CHECK_MUDNN(mudnnDestroy(handle));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend conv/resample graph API threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr
        << "mudnn_frontend conv/resample graph API threw unknown exception"
        << std::endl;
  }
  if (handle != nullptr) {
    mudnnDestroy(handle);
  }
  return false;
}

bool MudnnFrontendGraphApiBuildsBatchnormForwardBackwardGraphs() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));
  try {
    namespace fe = mudnn_frontend;

    auto finalize_graph = [&](const char* label, fe::graph::Graph& graph) {
      auto serialized = graph.serialize();
      fe::graph::Graph restored;
      auto status = restored.deserialize(serialized);
      if (!status.is_good()) {
        std::cerr << label << " graph deserialize failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.validate();
      if (!status.is_good()) {
        std::cerr << label << " graph validate failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.build_operation_graph(handle);
      if (!status.is_good()) {
        std::cerr << label
                  << " graph build_operation_graph failed: " << status.message
                  << std::endl;
        return false;
      }
      return true;
    };

    auto make_graph = []() {
      fe::graph::Graph graph;
      graph.set_io_data_type(fe::DataType_t::FLOAT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
      return graph;
    };

    auto make_x = [](fe::graph::Graph& graph, const std::string& name) {
      return graph.tensor(fe::graph::Tensor_attributes()
                              .set_name(name)
                              .set_dim({2, 2, 2, 2})
                              .set_stride({8, 4, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    };
    auto make_param = [](fe::graph::Graph& graph, const std::string& name) {
      return graph.tensor(fe::graph::Tensor_attributes()
                              .set_name(name)
                              .set_dim({1, 2, 1, 1})
                              .set_stride({2, 1, 1, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    };
    auto make_scalar = [](fe::graph::Graph& graph, const std::string& name) {
      return graph.tensor(fe::graph::Tensor_attributes()
                              .set_name(name)
                              .set_dim({1})
                              .set_stride({1})
                              .set_data_type(fe::DataType_t::FLOAT)
                              .set_is_pass_by_value(true));
    };

    fe::graph::Graph inference_graph = make_graph();
    auto inf_x = make_x(inference_graph, "x");
    auto inf_param = make_param(inference_graph, "param");
    auto inf_epsilon = make_scalar(inference_graph, "epsilon");
    auto inf_y = inference_graph.batchnorm_inference(
        inf_x, inf_param, inf_param, inf_param, inf_param,
        fe::graph::Batchnorm_attributes()
            .set_name("bn_inference")
            .set_epsilon(inf_epsilon)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    inf_y->set_output(true);
    if (!finalize_graph("batchnorm_inference", inference_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph training_graph = make_graph();
    auto train_x = make_x(training_graph, "x");
    auto train_param = make_param(training_graph, "param");
    auto train_epsilon = make_scalar(training_graph, "epsilon");
    auto train_momentum = make_scalar(training_graph, "momentum");
    auto train_outputs = training_graph.batchnorm(
        train_x, train_param, train_param, train_param, train_param,
        fe::graph::Batchnorm_attributes()
            .set_name("bn_training")
            .set_momentum(train_momentum)
            .set_epsilon(train_epsilon)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    std::get<0>(train_outputs)->set_output(true);
    std::get<1>(train_outputs)->set_output(true);
    std::get<2>(train_outputs)->set_output(true);
    std::get<3>(train_outputs)->set_output(true);
    std::get<4>(train_outputs)->set_output(true);
    if (!finalize_graph("batchnorm_training", training_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph backward_graph = make_graph();
    auto bwd_x = make_x(backward_graph, "x");
    auto bwd_dy = make_x(backward_graph, "dy");
    auto bwd_param = make_param(backward_graph, "param");
    auto bwd_epsilon = make_scalar(backward_graph, "epsilon");
    auto bwd_outputs = backward_graph.batchnorm_backward(
        bwd_dy, bwd_x, bwd_param, bwd_param, bwd_param,
        fe::graph::Batchnorm_attributes()
            .set_name("bn_backward")
            .set_epsilon(bwd_epsilon)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    std::get<0>(bwd_outputs)->set_output(true);
    std::get<1>(bwd_outputs)->set_output(true);
    std::get<2>(bwd_outputs)->set_output(true);
    if (!finalize_graph("batchnorm_backward", backward_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    CHECK_MUDNN(mudnnDestroy(handle));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend batchnorm graph API threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend batchnorm graph API threw unknown exception"
              << std::endl;
  }
  if (handle != nullptr) {
    mudnnDestroy(handle);
  }
  return false;
}

bool MudnnFrontendGraphApiBuildsNormAliasGraphs() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));
  try {
    namespace fe = mudnn_frontend;

    auto make_graph = []() {
      fe::graph::Graph graph;
      graph.set_io_data_type(fe::DataType_t::FLOAT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
      return graph;
    };
    auto make_x = [](fe::graph::Graph& graph, const std::string& name) {
      return graph.tensor(fe::graph::Tensor_attributes()
                              .set_name(name)
                              .set_dim({4, 8, 1, 1})
                              .set_stride({8, 1, 1, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    };
    auto make_param = [](fe::graph::Graph& graph, const std::string& name) {
      return graph.tensor(fe::graph::Tensor_attributes()
                              .set_name(name)
                              .set_dim({1, 8, 1, 1})
                              .set_stride({8, 1, 1, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    };
    auto make_epsilon = [](fe::graph::Graph& graph) {
      return graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("epsilon")
                              .set_dim({1, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(fe::DataType_t::FLOAT)
                              .set_is_pass_by_value(true));
    };
    auto finalize_unsupported = [&](const char* label,
                                    fe::graph::Graph& graph) {
      auto serialized = graph.serialize();
      fe::graph::Graph restored;
      auto status = restored.deserialize(serialized);
      if (!status.is_good()) {
        std::cerr << label << " graph deserialize failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.validate();
      if (!status.is_good()) {
        std::cerr << label << " graph validate failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.build_operation_graph(handle);
      if (status.is_good()) {
        std::cerr << label << " graph unexpectedly finalized" << std::endl;
        return false;
      }
      return true;
    };
    auto finalize_supported = [&](const char* label, fe::graph::Graph& graph) {
      auto serialized = graph.serialize();
      fe::graph::Graph restored;
      auto status = restored.deserialize(serialized);
      if (!status.is_good()) {
        std::cerr << label << " graph deserialize failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.validate();
      if (!status.is_good()) {
        std::cerr << label << " graph validate failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.build_operation_graph(handle);
      if (!status.is_good()) {
        std::cerr << label << " graph finalize failed: " << status.message
                  << std::endl;
        return false;
      }
      return true;
    };

    fe::graph::Graph layer_graph = make_graph();
    auto ln_x = make_x(layer_graph, "x");
    auto ln_scale = make_param(layer_graph, "scale");
    auto ln_bias = make_param(layer_graph, "bias");
    auto ln_epsilon = make_epsilon(layer_graph);
    auto ln_outputs = layer_graph.layernorm(
        ln_x, ln_scale, ln_bias,
        fe::graph::Norm_attributes()
            .set_name("layernorm")
            .set_epsilon(ln_epsilon)
            .set_norm_forward_phase(fe::NormFwdPhase_t::TRAINING));
    if (std::get<0>(ln_outputs)->get_dim() != ln_x->get_dim() ||
        std::get<1>(ln_outputs)->get_dim() !=
            std::vector<int64_t>({4, 1, 1, 1}) ||
        std::get<2>(ln_outputs)->get_dim() !=
            std::vector<int64_t>({4, 1, 1, 1})) {
      std::cerr << "layernorm graph output dims mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    std::get<0>(ln_outputs)->set_output(true);
    std::get<1>(ln_outputs)->set_output(true);
    std::get<2>(ln_outputs)->set_output(true);
    if (!finalize_supported("layernorm", layer_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph ada_graph = make_graph();
    auto ada_x = make_x(ada_graph, "x");
    auto ada_scale = make_param(ada_graph, "scale");
    auto ada_bias = make_param(ada_graph, "bias");
    auto ada_epsilon = make_epsilon(ada_graph);
    auto ada_outputs = ada_graph.adalayernorm(
        ada_x, ada_scale, ada_bias,
        fe::graph::AdaLayernorm_attributes()
            .set_name("adalayernorm")
            .set_epsilon(ada_epsilon)
            .set_norm_forward_phase(fe::NormFwdPhase_t::TRAINING));
    if (std::get<0>(ada_outputs)->get_dim() != ada_x->get_dim() ||
        std::get<1>(ada_outputs)->get_dim() !=
            std::vector<int64_t>({4, 1, 1, 1}) ||
        std::get<2>(ada_outputs)->get_dim() !=
            std::vector<int64_t>({4, 1, 1, 1})) {
      std::cerr << "adalayernorm graph output dims mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    std::get<0>(ada_outputs)->set_output(true);
    std::get<1>(ada_outputs)->set_output(true);
    std::get<2>(ada_outputs)->set_output(true);
    if (!finalize_unsupported("adalayernorm", ada_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph rms_graph = make_graph();
    auto rms_x = make_x(rms_graph, "x");
    auto rms_scale = make_param(rms_graph, "scale");
    auto rms_outputs = rms_graph.rmsnorm(
        rms_x, rms_scale,
        fe::graph::Norm_attributes().set_name("rmsnorm").set_norm_forward_phase(
            fe::NormFwdPhase_t::TRAINING));
    if (rms_outputs[0]->get_dim() != rms_x->get_dim() ||
        rms_outputs[1]->get_dim() != std::vector<int64_t>({4, 1, 1, 1})) {
      std::cerr << "rmsnorm graph output dims mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    rms_outputs[0]->set_output(true);
    rms_outputs[1]->set_output(true);
    if (!finalize_supported("rmsnorm", rms_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph inst_graph = make_graph();
    auto inst_x = make_x(inst_graph, "x");
    auto inst_scale = make_param(inst_graph, "scale");
    auto inst_bias = make_param(inst_graph, "bias");
    auto inst_outputs =
        inst_graph.instancenorm(inst_x, inst_scale, inst_bias,
                                fe::graph::Norm_attributes().set_name("inst"));
    if (std::get<1>(inst_outputs)->get_dim() !=
            std::vector<int64_t>({4, 8, 1, 1}) ||
        std::get<2>(inst_outputs)->get_dim() !=
            std::vector<int64_t>({4, 8, 1, 1})) {
      std::cerr << "instancenorm graph output dims mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    std::get<0>(inst_outputs)->set_output(true);
    std::get<1>(inst_outputs)->set_output(true);
    std::get<2>(inst_outputs)->set_output(true);
    if (!finalize_supported("instancenorm", inst_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph rms_bwd_graph = make_graph();
    auto bwd_x = make_x(rms_bwd_graph, "x");
    auto bwd_dy = make_x(rms_bwd_graph, "dy");
    auto bwd_scale = make_param(rms_bwd_graph, "scale");
    auto bwd_inv =
        rms_bwd_graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("inv")
                                 .set_dim({4, 1, 1, 1})
                                 .set_stride({1, 1, 1, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto bwd_outputs = rms_bwd_graph.rmsnorm_backward(
        bwd_dy, bwd_x, bwd_scale, bwd_inv,
        fe::graph::Norm_attributes().set_name("rms_bwd").set_has_dbias(false));
    if (std::get<0>(bwd_outputs)->get_dim() != bwd_x->get_dim() ||
        std::get<1>(bwd_outputs)->get_dim() != bwd_scale->get_dim() ||
        std::get<2>(bwd_outputs)) {
      std::cerr << "rmsnorm backward graph output mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    std::get<0>(bwd_outputs)->set_output(true);
    std::get<1>(bwd_outputs)->set_output(true);
    if (!finalize_supported("rmsnorm_backward", rms_bwd_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph ada_bwd_graph = make_graph();
    auto ada_bwd_x = make_x(ada_bwd_graph, "x");
    auto ada_bwd_dy = make_x(ada_bwd_graph, "dy");
    auto ada_bwd_scale = make_param(ada_bwd_graph, "scale");
    auto ada_bwd_mean =
        ada_bwd_graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mean")
                                 .set_dim({4, 1, 1, 1})
                                 .set_stride({1, 1, 1, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto ada_bwd_inv =
        ada_bwd_graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("inv")
                                 .set_dim({4, 1, 1, 1})
                                 .set_stride({1, 1, 1, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto ada_bwd_outputs = ada_bwd_graph.adalayernorm_backward(
        ada_bwd_dy, ada_bwd_x, ada_bwd_scale, ada_bwd_mean, ada_bwd_inv,
        fe::graph::AdaLayernorm_backward_attributes().set_name("ada_bwd"));
    if (std::get<0>(ada_bwd_outputs)->get_dim() != ada_bwd_x->get_dim() ||
        std::get<1>(ada_bwd_outputs)->get_dim() != ada_bwd_scale->get_dim() ||
        std::get<2>(ada_bwd_outputs)->get_dim() != ada_bwd_scale->get_dim()) {
      std::cerr << "adalayernorm backward graph output mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    std::get<0>(ada_bwd_outputs)->set_output(true);
    std::get<1>(ada_bwd_outputs)->set_output(true);
    std::get<2>(ada_bwd_outputs)->set_output(true);
    if (!finalize_unsupported("adalayernorm_backward", ada_bwd_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    CHECK_MUDNN(mudnnDestroy(handle));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend norm alias graph API threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend norm alias graph API threw unknown exception"
              << std::endl;
  }
  if (handle != nullptr) {
    mudnnDestroy(handle);
  }
  return false;
}

bool MudnnFrontendGraphApiBuildsBlockScaleGraphs() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));
  try {
    namespace fe = mudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto x = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("x")
                              .set_dim({2, 5})
                              .set_stride({5, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto quantized = graph.block_scale_quantize(
        x,
        fe::graph::Block_scale_attributes()
            .set_name("quant")
            .set_block_size(2)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    if (quantized[0]->get_dim() != std::vector<int64_t>({2, 5}) ||
        quantized[1]->get_dim() != std::vector<int64_t>({2, 3})) {
      std::cerr << "block_scale graph output dims mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    auto y = graph.block_scale_dequantize(
        quantized[0], quantized[1],
        fe::graph::Block_scale_attributes()
            .set_name("dequant")
            .set_block_size(2)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    y->set_output(true);

    auto serialized = graph.serialize();
    fe::graph::Graph restored;
    auto status = restored.deserialize(serialized);
    if (!status.is_good()) {
      std::cerr << "block_scale graph deserialize failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = restored.validate();
    if (!status.is_good()) {
      std::cerr << "block_scale graph validate failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = restored.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "block_scale graph build_operation_graph failed: "
                << status.message << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    CHECK_MUDNN(mudnnDestroy(handle));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend block_scale graph API threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend block_scale graph API threw unknown exception"
              << std::endl;
  }
  if (handle != nullptr) {
    mudnnDestroy(handle);
  }
  return false;
}

bool MudnnFrontendGraphApiBuildsPagedCacheSignalGraphs() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));
  try {
    namespace fe = mudnn_frontend;

    auto finalize_graph = [&](const char* label, fe::graph::Graph& graph) {
      auto serialized = graph.serialize();
      fe::graph::Graph restored;
      auto status = restored.deserialize(serialized);
      if (!status.is_good()) {
        std::cerr << label << " graph deserialize failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.validate();
      if (!status.is_good()) {
        std::cerr << label << " graph validate failed: " << status.message
                  << std::endl;
        return false;
      }
      status = restored.build_operation_graph(handle);
      if (!status.is_good()) {
        std::cerr << label
                  << " graph build_operation_graph failed: " << status.message
                  << std::endl;
        return false;
      }
      return true;
    };

    auto make_graph = []() {
      fe::graph::Graph graph;
      graph.set_io_data_type(fe::DataType_t::FLOAT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);
      return graph;
    };

    fe::graph::Graph paged_graph = make_graph();
    auto container =
        paged_graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("container")
                               .set_dim({3, 2, 2})
                               .set_stride({4, 2, 1})
                               .set_data_type(fe::DataType_t::FLOAT));
    auto sequence =
        paged_graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("sequence")
                               .set_dim({2})
                               .set_stride({1})
                               .set_data_type(fe::DataType_t::INT64));
    auto page_table =
        paged_graph.tensor(fe::graph::Tensor_attributes()
                               .set_name("page_table")
                               .set_dim({3})
                               .set_stride({1})
                               .set_data_type(fe::DataType_t::INT64));
    auto paged_y = paged_graph.paged_cache_load(
        container, sequence, page_table,
        fe::graph::Paged_cache_load_attributes().set_name("paged"));
    if (paged_y->get_dim() != std::vector<int64_t>({2, 2, 2})) {
      std::cerr << "paged_cache_load graph output dims mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    paged_y->set_output(true);
    if (!finalize_graph("paged_cache_load", paged_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph signal_set_graph = make_graph();
    auto signal_x =
        signal_set_graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("x")
                                    .set_dim({2, 3})
                                    .set_stride({3, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));
    auto signal_flag =
        signal_set_graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("flag")
                                    .set_dim({1})
                                    .set_stride({1})
                                    .set_data_type(fe::DataType_t::INT64));
    auto signal_y = signal_set_graph.signal(
        MUDNN_SIGNAL_SET, signal_flag, signal_x,
        fe::graph::Signal_attributes().set_name("set").set_value(17));
    signal_y->set_output(true);
    signal_flag->set_output(true);
    if (!finalize_graph("signal_set", signal_set_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    fe::graph::Graph signal_wait_graph = make_graph();
    auto wait_x =
        signal_wait_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("x")
                                     .set_dim({2, 3})
                                     .set_stride({3, 1})
                                     .set_data_type(fe::DataType_t::FLOAT));
    auto wait_flag =
        signal_wait_graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("flag")
                                     .set_dim({1})
                                     .set_stride({1})
                                     .set_data_type(fe::DataType_t::INT64));
    auto wait_y = signal_wait_graph.signal(
        MUDNN_SIGNAL_WAIT, wait_flag, wait_x,
        fe::graph::Signal_attributes().set_name("wait").set_value(17));
    wait_y->set_output(true);
    if (!finalize_graph("signal_wait", signal_wait_graph)) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    CHECK_MUDNN(mudnnDestroy(handle));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend paged_cache/signal graph API threw: "
              << e.what() << std::endl;
  } catch (...) {
    std::cerr
        << "mudnn_frontend paged_cache/signal graph API threw unknown exception"
        << std::endl;
  }
  if (handle != nullptr) {
    mudnnDestroy(handle);
  }
  return false;
}

}  // namespace mudnn_frontend_tests
