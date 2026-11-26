/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#include "frontend_api_common.h"

namespace mudnn_frontend_tests {

bool MudnnFrontendGraphApiExecutesSdpaGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* q_ptr = nullptr;
  void* k_ptr = nullptr;
  void* v_ptr = nullptr;
  void* o_ptr = nullptr;
  void* workspace = nullptr;

  try {
    namespace fe = mudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto q = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("q")
                              .set_dim({1, 1, 2, 2})
                              .set_stride({4, 4, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto k = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("k")
                              .set_dim({1, 1, 3, 2})
                              .set_stride({6, 6, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto v = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("v")
                              .set_dim({1, 1, 3, 2})
                              .set_stride({6, 6, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto sdpa_outputs =
        graph.sdpa(q, k, v,
                   fe::graph::Sdpa_attributes()
                       .set_name("sdpa")
                       .set_attn_scale(0.5)
                       .set_is_inference(true)
                       .set_causal_mask(false)
                       .set_use_padding_mask(false)
                       .set_compute_data_type(fe::DataType_t::FLOAT));
    auto o = sdpa_outputs[0];
    if (!o || sdpa_outputs[1]) {
      std::cerr << "graph API sdpa output shape mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    o->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API sdpa validate failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa build_operation_graph failed: "
                << status.message << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API sdpa create_execution_plans failed: "
                << status.message << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa build_plans failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    const std::array<float, 4> q_host = {0.2f, -0.5f, 1.0f, 0.25f};
    const std::array<float, 6> k_host = {0.0f, 0.75f, -0.25f,
                                         0.5f, 1.0f,  -1.0f};
    const std::array<float, 6> v_host = {1.0f, -1.0f,  0.5f,
                                         2.0f, -0.25f, 0.75f};
    std::array<float, 4> expected{};
    for (int q_idx = 0; q_idx < 2; ++q_idx) {
      std::array<float, 3> logits{};
      float max_logit = -std::numeric_limits<float>::infinity();
      for (int k_idx = 0; k_idx < 3; ++k_idx) {
        const float dot = q_host[q_idx * 2] * k_host[k_idx * 2] +
            q_host[q_idx * 2 + 1] * k_host[k_idx * 2 + 1];
        logits[k_idx] = dot * 0.5f;
        max_logit = std::max(max_logit, logits[k_idx]);
      }
      float denom = 0.0f;
      for (float& logit : logits) {
        logit = std::exp(logit - max_logit);
        denom += logit;
      }
      for (int value_idx = 0; value_idx < 2; ++value_idx) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < 3; ++k_idx) {
          sum += (logits[k_idx] / denom) * v_host[k_idx * 2 + value_idx];
        }
        expected[q_idx * 2 + value_idx] = sum;
      }
    }

    CHECK_MUSA(musaMalloc(&q_ptr, sizeof(float) * q_host.size()));
    CHECK_MUSA(musaMalloc(&k_ptr, sizeof(float) * k_host.size()));
    CHECK_MUSA(musaMalloc(&v_ptr, sizeof(float) * v_host.size()));
    CHECK_MUSA(musaMalloc(&o_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(q_ptr, q_host.data(), sizeof(float) * q_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(k_ptr, k_host.data(), sizeof(float) * k_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(v_ptr, v_host.data(), sizeof(float) * v_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa get_workspace_size failed: "
                << status.message << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{q, q_ptr}, {k, k_ptr}, {v, v_ptr}, {o, o_ptr}};
    status = graph.execute(handle, variant_pack, workspace);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa execute failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<float, 4> got{};
    CHECK_MUSA(musaMemcpy(got.data(), o_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) {
      // Backend SDPA may use a different softmax accumulation order than this
      // scalar golden; keep the tolerance narrow but above observed fp32 drift.
      if (std::fabs(got[i] - expected[i]) > 5.0e-4f) {
        std::cerr << "graph API sdpa mismatch at " << i << ": got " << got[i]
                  << ", expected " << expected[i] << std::endl;
        if (workspace != nullptr) {
          musaFree(workspace);
        }
        musaFree(o_ptr);
        musaFree(v_ptr);
        musaFree(k_ptr);
        musaFree(q_ptr);
        CHECK_MUDNN(mudnnDestroy(handle));
        return false;
      }
    }

    if (workspace != nullptr) {
      CHECK_MUSA(musaFree(workspace));
    }
    CHECK_MUSA(musaFree(o_ptr));
    CHECK_MUSA(musaFree(v_ptr));
    CHECK_MUSA(musaFree(k_ptr));
    CHECK_MUSA(musaFree(q_ptr));
    CHECK_MUDNN(mudnnDestroy(handle));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API sdpa smoke threw: " << e.what()
              << std::endl;
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (o_ptr != nullptr) {
      musaFree(o_ptr);
    }
    if (v_ptr != nullptr) {
      musaFree(v_ptr);
    }
    if (k_ptr != nullptr) {
      musaFree(k_ptr);
    }
    if (q_ptr != nullptr) {
      musaFree(q_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  }
}

bool MudnnFrontendGraphApiExecutesSdpaBiasBoolMaskGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* q_ptr = nullptr;
  void* k_ptr = nullptr;
  void* v_ptr = nullptr;
  void* bias_ptr = nullptr;
  void* mask_ptr = nullptr;
  void* o_ptr = nullptr;
  void* workspace = nullptr;

  auto cleanup = [&]() {
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (o_ptr != nullptr) {
      musaFree(o_ptr);
    }
    if (mask_ptr != nullptr) {
      musaFree(mask_ptr);
    }
    if (bias_ptr != nullptr) {
      musaFree(bias_ptr);
    }
    if (v_ptr != nullptr) {
      musaFree(v_ptr);
    }
    if (k_ptr != nullptr) {
      musaFree(k_ptr);
    }
    if (q_ptr != nullptr) {
      musaFree(q_ptr);
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

    auto q = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("q")
                              .set_dim({1, 1, 2, 2})
                              .set_stride({4, 4, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto k = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("k")
                              .set_dim({1, 1, 3, 2})
                              .set_stride({6, 6, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto v = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("v")
                              .set_dim({1, 1, 3, 2})
                              .set_stride({6, 6, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto bias = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 1, 2, 3})
                                 .set_stride({6, 6, 3, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto mask = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mask")
                                 .set_dim({1, 1, 2, 3})
                                 .set_stride({6, 6, 3, 1})
                                 .set_data_type(fe::DataType_t::BOOLEAN));
    auto sdpa_outputs =
        graph.sdpa(q, k, v,
                   fe::graph::Sdpa_attributes()
                       .set_name("sdpa_bias_mask")
                       .set_attn_scale(0.5)
                       .set_is_inference(true)
                       .set_causal_mask(false)
                       .set_use_padding_mask(false)
                       .set_compute_data_type(fe::DataType_t::FLOAT),
                   bias, mask);
    auto o = sdpa_outputs[0];
    if (!o || sdpa_outputs[1]) {
      std::cerr << "graph API sdpa bias+mask output mismatch" << std::endl;
      cleanup();
      return false;
    }
    o->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API sdpa bias+mask validate failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa bias+mask build_operation_graph failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API sdpa bias+mask create_execution_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa bias+mask build_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }

    const std::array<float, 4> q_host = {0.2f, -0.5f, 1.0f, 0.25f};
    const std::array<float, 6> k_host = {0.0f, 0.75f, -0.25f,
                                         0.5f, 1.0f,  -1.0f};
    const std::array<float, 6> v_host = {1.0f, -1.0f,  0.5f,
                                         2.0f, -0.25f, 0.75f};
    const std::array<float, 6> bias_host = {0.0f, -1.0f, 0.25f,
                                            0.5f, 0.0f,  -0.75f};
    const std::array<bool, 6> mask_host = {true, false, true,
                                           true, true,  false};
    std::array<float, 4> expected{};
    for (int q_idx = 0; q_idx < 2; ++q_idx) {
      std::array<float, 3> logits{};
      float max_logit = -std::numeric_limits<float>::infinity();
      for (int k_idx = 0; k_idx < 3; ++k_idx) {
        const float dot = q_host[q_idx * 2] * k_host[k_idx * 2] +
            q_host[q_idx * 2 + 1] * k_host[k_idx * 2 + 1];
        const int mask_index = q_idx * 3 + k_idx;
        logits[k_idx] = mask_host[mask_index]
            ? dot * 0.5f + bias_host[mask_index]
            : -1.0e9f;
        max_logit = std::max(max_logit, logits[k_idx]);
      }
      float denom = 0.0f;
      for (float& logit : logits) {
        logit = std::exp(logit - max_logit);
        denom += logit;
      }
      for (int value_idx = 0; value_idx < 2; ++value_idx) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < 3; ++k_idx) {
          sum += (logits[k_idx] / denom) * v_host[k_idx * 2 + value_idx];
        }
        expected[q_idx * 2 + value_idx] = sum;
      }
    }

    CHECK_MUSA(musaMalloc(&q_ptr, sizeof(float) * q_host.size()));
    CHECK_MUSA(musaMalloc(&k_ptr, sizeof(float) * k_host.size()));
    CHECK_MUSA(musaMalloc(&v_ptr, sizeof(float) * v_host.size()));
    CHECK_MUSA(musaMalloc(&bias_ptr, sizeof(float) * bias_host.size()));
    CHECK_MUSA(musaMalloc(&mask_ptr, sizeof(bool) * mask_host.size()));
    CHECK_MUSA(musaMalloc(&o_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(q_ptr, q_host.data(), sizeof(float) * q_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(k_ptr, k_host.data(), sizeof(float) * k_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(v_ptr, v_host.data(), sizeof(float) * v_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(bias_ptr, bias_host.data(),
                          sizeof(float) * bias_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(mask_ptr, mask_host.data(),
                          sizeof(bool) * mask_host.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa bias+mask get_workspace_size failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {{q, q_ptr},       {k, k_ptr},       {v, v_ptr},
                        {bias, bias_ptr}, {mask, mask_ptr}, {o, o_ptr}};
    status = graph.execute(handle, variant_pack, workspace);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa bias+mask execute failed: " << status.message
                << std::endl;
      cleanup();
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<float, 4> got{};
    CHECK_MUSA(musaMemcpy(got.data(), o_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) {
      // Backend SDPA may use a different softmax accumulation order than this
      // scalar golden; keep the tolerance narrow but above observed fp32 drift.
      if (std::fabs(got[i] - expected[i]) > 5.0e-4f) {
        std::cerr << "graph API sdpa bias+mask mismatch at " << i << ": got "
                  << got[i] << ", expected " << expected[i] << std::endl;
        cleanup();
        return false;
      }
    }

    cleanup();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API sdpa bias+mask smoke threw: "
              << e.what() << std::endl;
    cleanup();
    return false;
  }
}

bool MudnnFrontendGraphApiExecutesSdpaGenerateStatsGraph() {
  if (!HasMusaDevice()) {
    return true;
  }

  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* q_ptr = nullptr;
  void* k_ptr = nullptr;
  void* v_ptr = nullptr;
  void* o_ptr = nullptr;
  void* stats_ptr = nullptr;
  void* workspace = nullptr;

  auto cleanup = [&]() {
    if (workspace != nullptr) {
      musaFree(workspace);
    }
    if (stats_ptr != nullptr) {
      musaFree(stats_ptr);
    }
    if (o_ptr != nullptr) {
      musaFree(o_ptr);
    }
    if (v_ptr != nullptr) {
      musaFree(v_ptr);
    }
    if (k_ptr != nullptr) {
      musaFree(k_ptr);
    }
    if (q_ptr != nullptr) {
      musaFree(q_ptr);
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
  };

  try {
    namespace fe = mudnn_frontend;
    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // MUSA Flash Attention requires head dimensions divisible by eight when
    // the fused forward materializes the Stats/LSE output.
    auto q = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("q")
                              .set_dim({1, 1, 2, 8})
                              .set_stride({16, 16, 8, 1})
                              .set_data_type(fe::DataType_t::HALF));
    auto k = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("k")
                              .set_dim({1, 1, 2, 8})
                              .set_stride({16, 16, 8, 1})
                              .set_data_type(fe::DataType_t::HALF));
    auto v = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("v")
                              .set_dim({1, 1, 2, 8})
                              .set_stride({16, 16, 8, 1})
                              .set_data_type(fe::DataType_t::HALF));
    auto sdpa_outputs =
        graph.sdpa(q, k, v,
                   fe::graph::Sdpa_attributes()
                       .set_name("sdpa_train")
                       .set_attn_scale(0.5)
                       .set_generate_stats(true)
                       .set_causal_mask(false)
                       .set_use_padding_mask(false)
                       .set_compute_data_type(fe::DataType_t::FLOAT));
    auto o = sdpa_outputs[0];
    auto stats = sdpa_outputs[1];
    if (!o || !stats) {
      std::cerr << "graph API sdpa generate-stats outputs missing" << std::endl;
      cleanup();
      return false;
    }
    o->set_output(true).set_data_type(fe::DataType_t::HALF);
    stats->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    auto status = graph.validate();
    if (!status.is_good()) {
      std::cerr << "graph API sdpa generate-stats validate failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_operation_graph(handle);
    if (!status.is_good()) {
      std::cerr
          << "graph API sdpa generate-stats build_operation_graph failed: "
          << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.create_execution_plans({fe::HeurMode_t::INSTANT});
    if (!status.is_good()) {
      std::cerr << "graph API sdpa generate-stats create_execution_plans "
                   "failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    status = graph.build_plans(fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa generate-stats build_plans failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }

    constexpr uint16_t kHalfZero = 0x0000;
    constexpr uint16_t kHalfOne = 0x3c00;
    std::array<uint16_t, 16> zeros{};
    zeros.fill(kHalfZero);
    std::array<uint16_t, 16> ones{};
    ones.fill(kHalfOne);

    CHECK_MUSA(musaMalloc(&q_ptr, sizeof(uint16_t) * zeros.size()));
    CHECK_MUSA(musaMalloc(&k_ptr, sizeof(uint16_t) * zeros.size()));
    CHECK_MUSA(musaMalloc(&v_ptr, sizeof(uint16_t) * ones.size()));
    CHECK_MUSA(musaMalloc(&o_ptr, sizeof(uint16_t) * ones.size()));
    CHECK_MUSA(musaMalloc(&stats_ptr, sizeof(float) * 2));
    CHECK_MUSA(musaMemset(o_ptr, 0, sizeof(uint16_t) * ones.size()));
    CHECK_MUSA(musaMemset(stats_ptr, 0, sizeof(float) * 2));
    CHECK_MUSA(musaMemcpy(q_ptr, zeros.data(), sizeof(uint16_t) * zeros.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(k_ptr, zeros.data(), sizeof(uint16_t) * zeros.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(v_ptr, ones.data(), sizeof(uint16_t) * ones.size(),
                          musaMemcpyHostToDevice));

    int64_t workspace_size = 0;
    status = graph.get_workspace_size(workspace_size);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa generate-stats get_workspace_size failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>
        variant_pack = {
            {q, q_ptr}, {k, k_ptr}, {v, v_ptr}, {o, o_ptr}, {stats, stats_ptr}};
    status = graph.execute(handle, variant_pack, workspace);
    if (!status.is_good()) {
      std::cerr << "graph API sdpa generate-stats execute failed: "
                << status.message << std::endl;
      cleanup();
      return false;
    }
    SyncMudnnHandleStream(handle);

    std::array<uint16_t, 16> got{};
    std::array<float, 2> stats_host{};
    CHECK_MUSA(musaMemcpy(got.data(), o_ptr, sizeof(uint16_t) * got.size(),
                          musaMemcpyDeviceToHost));
    CHECK_MUSA(musaMemcpy(stats_host.data(), stats_ptr,
                          sizeof(float) * stats_host.size(),
                          musaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) {
      if (got[i] != kHalfOne) {
        std::cerr << "graph API sdpa generate-stats mismatch at " << i
                  << ": got 0x" << std::hex << got[i] << ", expected 0x"
                  << kHalfOne << std::dec << std::endl;
        cleanup();
        return false;
      }
    }
    bool stats_written = false;
    for (float value : stats_host) {
      stats_written = stats_written || value != 0.0f;
      if (!std::isfinite(value)) {
        std::cerr << "graph API sdpa generate-stats wrote non-finite stats"
                  << std::endl;
        cleanup();
        return false;
      }
    }
    if (!stats_written) {
      std::cerr << "graph API sdpa generate-stats did not write stats"
                << std::endl;
      cleanup();
      return false;
    }

    cleanup();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend graph API sdpa generate-stats smoke threw: "
              << e.what() << std::endl;
    cleanup();
    return false;
  }
}

bool MudnnFrontendBuildsSdpaGraph() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  try {
    constexpr int64_t q_dims[] = {1, 1, 2, 2};
    constexpr int64_t q_strides[] = {4, 4, 2, 1};
    constexpr int64_t k_dims[] = {1, 1, 3, 2};
    constexpr int64_t k_strides[] = {6, 6, 2, 1};
    constexpr int64_t v_dims[] = {1, 1, 3, 2};
    constexpr int64_t v_strides[] = {6, 6, 2, 1};
    constexpr int64_t o_dims[] = {1, 1, 2, 2};
    constexpr int64_t o_strides[] = {4, 4, 2, 1};
    constexpr int64_t attn_dims[] = {1, 1, 2, 3};
    constexpr int64_t attn_strides[] = {6, 6, 3, 1};
    constexpr int64_t mask_dims[] = {1, 1, 2, 3};
    constexpr int64_t mask_strides[] = {6, 6, 3, 1};

    auto q = mudnn_frontend::TensorBuilder()
                 .setDim(4, q_dims)
                 .setStrides(4, q_strides)
                 .setId(1)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto k = mudnn_frontend::TensorBuilder()
                 .setDim(4, k_dims)
                 .setStrides(4, k_strides)
                 .setId(2)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto v = mudnn_frontend::TensorBuilder()
                 .setDim(4, v_dims)
                 .setStrides(4, v_strides)
                 .setId(3)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto o = mudnn_frontend::TensorBuilder()
                 .setDim(4, o_dims)
                 .setStrides(4, o_strides)
                 .setId(4)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto attn_probs = mudnn_frontend::TensorBuilder()
                          .setDim(4, attn_dims)
                          .setStrides(4, attn_strides)
                          .setId(100)
                          .setDataType(CUDNN_DATA_FLOAT)
                          .setVirtual(true)
                          .build();
    auto dropout_mask = mudnn_frontend::TensorBuilder()
                            .setDim(4, attn_dims)
                            .setStrides(4, attn_strides)
                            .setId(101)
                            .setDataType(CUDNN_DATA_BOOLEAN)
                            .setVirtual(true)
                            .build();
    auto mask = mudnn_frontend::TensorBuilder()
                    .setDim(4, mask_dims)
                    .setStrides(4, mask_strides)
                    .setId(5)
                    .setDataType(CUDNN_DATA_FLOAT)
                    .build();

    bool sliding_window_rejected = false;
    try {
      (void)mudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)
          .setSdpaSlidingWindowLength(2);
    } catch (const std::invalid_argument&) {
      sliding_window_rejected = true;
    }
    if (!sliding_window_rejected) {
      std::cerr << "sdpa sliding_window_length was not rejected" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    auto op = mudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)
                  .setQDesc(q)
                  .setKDesc(k)
                  .setVDesc(v)
                  .setMaskDesc(mask)
                  .setODesc(o)
                  .setAttnProbsDesc(attn_probs)
                  .setDropoutMaskDesc(dropout_mask)
                  .setSdpaScale(0.5)
                  .setSdpaInference(true)
                  .setSdpaCausalMask(true)
                  .setSdpaUsePaddingMask(false)
                  .build();
    int64_t bool_count = 0;
    bool got_causal_mask = false;
    CHECK_MUDNN(cudnnBackendGetAttribute(
        op.get_backend_descriptor(), CUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK,
        CUDNN_TYPE_BOOLEAN, 1, &bool_count, &got_causal_mask));
    if (bool_count != 1 || !got_causal_mask) {
      std::cerr << "sdpa causal mask attribute was not set" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    const mudnn_frontend::Operation* ops[] = {&op};
    auto graph = mudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(1, ops)
                     .build();
    auto heur = mudnn_frontend::EngineHeuristicsBuilder()
                    .setOperationGraph(graph)
                    .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                    .build();
    const auto configs = heur.getEngineConfig();
    CHECK_MUDNN(mudnnDestroy(handle));
    return !configs.empty();
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend sdpa build threw: " << e.what() << std::endl;
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  } catch (...) {
    std::cerr << "mudnn_frontend sdpa build threw unknown exception"
              << std::endl;
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  }
}

bool MudnnFrontendExecutesSdpaBiasBoolMaskGraph() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* q_ptr = nullptr;
  void* k_ptr = nullptr;
  void* v_ptr = nullptr;
  void* bias_ptr = nullptr;
  void* mask_ptr = nullptr;
  void* o_ptr = nullptr;

  try {
    constexpr int64_t q_dims[] = {1, 1, 2, 2};
    constexpr int64_t q_strides[] = {4, 4, 2, 1};
    constexpr int64_t k_dims[] = {1, 1, 3, 2};
    constexpr int64_t k_strides[] = {6, 6, 2, 1};
    constexpr int64_t v_dims[] = {1, 1, 3, 2};
    constexpr int64_t v_strides[] = {6, 6, 2, 1};
    constexpr int64_t o_dims[] = {1, 1, 2, 2};
    constexpr int64_t o_strides[] = {4, 4, 2, 1};
    constexpr int64_t attn_dims[] = {1, 1, 2, 3};
    constexpr int64_t attn_strides[] = {6, 6, 3, 1};

    auto q = mudnn_frontend::TensorBuilder()
                 .setDim(4, q_dims)
                 .setStrides(4, q_strides)
                 .setId(1)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto k = mudnn_frontend::TensorBuilder()
                 .setDim(4, k_dims)
                 .setStrides(4, k_strides)
                 .setId(2)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto v = mudnn_frontend::TensorBuilder()
                 .setDim(4, v_dims)
                 .setStrides(4, v_strides)
                 .setId(3)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto o = mudnn_frontend::TensorBuilder()
                 .setDim(4, o_dims)
                 .setStrides(4, o_strides)
                 .setId(4)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto bias = mudnn_frontend::TensorBuilder()
                    .setDim(4, attn_dims)
                    .setStrides(4, attn_strides)
                    .setId(5)
                    .setDataType(CUDNN_DATA_FLOAT)
                    .build();
    auto mask = mudnn_frontend::TensorBuilder()
                    .setDim(4, attn_dims)
                    .setStrides(4, attn_strides)
                    .setId(6)
                    .setDataType(CUDNN_DATA_BOOLEAN)
                    .build();
    auto attn_probs = mudnn_frontend::TensorBuilder()
                          .setDim(4, attn_dims)
                          .setStrides(4, attn_strides)
                          .setId(100)
                          .setDataType(CUDNN_DATA_FLOAT)
                          .setVirtual(true)
                          .build();
    auto dropout_mask = mudnn_frontend::TensorBuilder()
                            .setDim(4, attn_dims)
                            .setStrides(4, attn_strides)
                            .setId(101)
                            .setDataType(CUDNN_DATA_BOOLEAN)
                            .setVirtual(true)
                            .build();

    auto op = mudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)
                  .setQDesc(q)
                  .setKDesc(k)
                  .setVDesc(v)
                  .setBiasDesc(bias)
                  .setMaskDesc(mask)
                  .setODesc(o)
                  .setAttnProbsDesc(attn_probs)
                  .setDropoutMaskDesc(dropout_mask)
                  .setSdpaScale(0.5)
                  .setSdpaInference(true)
                  .setSdpaCausalMask(false)
                  .setSdpaUsePaddingMask(false)
                  .build();
    const mudnn_frontend::Operation* ops[] = {&op};
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

    const std::array<float, 4> q_host = {0.2f, -0.5f, 1.0f, 0.25f};
    const std::array<float, 6> k_host = {0.0f, 0.75f, -0.25f,
                                         0.5f, 1.0f,  -1.0f};
    const std::array<float, 6> v_host = {1.0f, -1.0f,  0.5f,
                                         2.0f, -0.25f, 0.75f};
    const std::array<float, 6> bias_host = {0.0f, -1.0f, 0.25f,
                                            0.5f, 0.0f,  -0.75f};
    const std::array<bool, 6> mask_host = {true, false, true,
                                           true, true,  false};
    std::array<float, 4> expected{};
    for (int q_idx = 0; q_idx < 2; ++q_idx) {
      std::array<float, 3> logits{};
      float max_logit = -std::numeric_limits<float>::infinity();
      for (int k_idx = 0; k_idx < 3; ++k_idx) {
        const float dot = q_host[q_idx * 2] * k_host[k_idx * 2] +
            q_host[q_idx * 2 + 1] * k_host[k_idx * 2 + 1];
        const int mask_index = q_idx * 3 + k_idx;
        logits[k_idx] = mask_host[mask_index]
            ? dot * 0.5f + bias_host[mask_index]
            : -1.0e9f;
        max_logit = std::max(max_logit, logits[k_idx]);
      }
      float denom = 0.0f;
      for (float& logit : logits) {
        logit = std::exp(logit - max_logit);
        denom += logit;
      }
      for (int value_idx = 0; value_idx < 2; ++value_idx) {
        float sum = 0.0f;
        for (int k_idx = 0; k_idx < 3; ++k_idx) {
          sum += (logits[k_idx] / denom) * v_host[k_idx * 2 + value_idx];
        }
        expected[q_idx * 2 + value_idx] = sum;
      }
    }

    CHECK_MUSA(musaMalloc(&q_ptr, sizeof(float) * q_host.size()));
    CHECK_MUSA(musaMalloc(&k_ptr, sizeof(float) * k_host.size()));
    CHECK_MUSA(musaMalloc(&v_ptr, sizeof(float) * v_host.size()));
    CHECK_MUSA(musaMalloc(&bias_ptr, sizeof(float) * bias_host.size()));
    CHECK_MUSA(musaMalloc(&mask_ptr, sizeof(bool) * mask_host.size()));
    CHECK_MUSA(musaMalloc(&o_ptr, sizeof(float) * expected.size()));
    CHECK_MUSA(musaMemcpy(q_ptr, q_host.data(), sizeof(float) * q_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(k_ptr, k_host.data(), sizeof(float) * k_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(v_ptr, v_host.data(), sizeof(float) * v_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(bias_ptr, bias_host.data(),
                          sizeof(float) * bias_host.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(mask_ptr, mask_host.data(),
                          sizeof(bool) * mask_host.size(),
                          musaMemcpyHostToDevice));

    void* ptrs[] = {q_ptr, k_ptr, v_ptr, o_ptr, bias_ptr, mask_ptr};
    int64_t uids[] = {1, 2, 3, 4, 5, 6};
    auto variant_pack = mudnn_frontend::VariantPackBuilder()
                            .setUids(6, uids)
                            .setDataPointers(6, ptrs)
                            .build();
    CHECK_MUDNN(mudnn_frontend::execute(handle, plan, variant_pack));
    SyncMudnnHandleStream(handle);

    std::array<float, 4> got{};
    CHECK_MUSA(musaMemcpy(got.data(), o_ptr, sizeof(float) * got.size(),
                          musaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < got.size(); ++i) {
      // Backend SDPA may use a different softmax accumulation order than this
      // scalar golden; keep the tolerance narrow but above observed fp32 drift.
      const bool element_ok = std::fabs(got[i] - expected[i]) < 5.0e-4f;
      if (!element_ok) {
        std::cerr << "mudnn_frontend sdpa bias+bool-mask mismatch at " << i
                  << ": got " << got[i] << ", expected " << expected[i]
                  << std::endl;
      }
      ok = ok && element_ok;
    }

    CHECK_MUSA(musaFree(o_ptr));
    CHECK_MUSA(musaFree(mask_ptr));
    CHECK_MUSA(musaFree(bias_ptr));
    CHECK_MUSA(musaFree(v_ptr));
    CHECK_MUSA(musaFree(k_ptr));
    CHECK_MUSA(musaFree(q_ptr));
    CHECK_MUDNN(mudnnDestroy(handle));
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend sdpa bias+bool-mask threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend sdpa bias+bool-mask threw unknown exception"
              << std::endl;
  }
  if (o_ptr != nullptr) {
    musaFree(o_ptr);
  }
  if (mask_ptr != nullptr) {
    musaFree(mask_ptr);
  }
  if (bias_ptr != nullptr) {
    musaFree(bias_ptr);
  }
  if (v_ptr != nullptr) {
    musaFree(v_ptr);
  }
  if (k_ptr != nullptr) {
    musaFree(k_ptr);
  }
  if (q_ptr != nullptr) {
    musaFree(q_ptr);
  }
  if (handle != nullptr) {
    mudnnDestroy(handle);
  }
  return false;
}

bool MudnnFrontendExecutesSdpaBackwardGraph() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  void* q_ptr = nullptr;
  void* k_ptr = nullptr;
  void* v_ptr = nullptr;
  void* d_o_ptr = nullptr;
  void* d_q_ptr = nullptr;
  void* d_k_ptr = nullptr;
  void* d_v_ptr = nullptr;
  void* workspace = nullptr;

  auto cleanup = [&]() {
    if (workspace != nullptr) {
      musaFree(workspace);
      workspace = nullptr;
    }
    if (d_v_ptr != nullptr) {
      musaFree(d_v_ptr);
      d_v_ptr = nullptr;
    }
    if (d_k_ptr != nullptr) {
      musaFree(d_k_ptr);
      d_k_ptr = nullptr;
    }
    if (d_q_ptr != nullptr) {
      musaFree(d_q_ptr);
      d_q_ptr = nullptr;
    }
    if (d_o_ptr != nullptr) {
      musaFree(d_o_ptr);
      d_o_ptr = nullptr;
    }
    if (v_ptr != nullptr) {
      musaFree(v_ptr);
      v_ptr = nullptr;
    }
    if (k_ptr != nullptr) {
      musaFree(k_ptr);
      k_ptr = nullptr;
    }
    if (q_ptr != nullptr) {
      musaFree(q_ptr);
      q_ptr = nullptr;
    }
    if (handle != nullptr) {
      mudnnDestroy(handle);
      handle = nullptr;
    }
  };

  try {
    constexpr int64_t qkv_dims[] = {1, 1, 2, 64};
    constexpr int64_t qkv_strides[] = {128, 128, 64, 1};
    constexpr int64_t attn_dims[] = {1, 1, 2, 2};
    constexpr int64_t attn_strides[] = {4, 4, 2, 1};
    constexpr int64_t stats_dims[] = {1, 1, 2, 1};
    constexpr int64_t stats_strides[] = {2, 2, 1, 1};

    auto q = mudnn_frontend::TensorBuilder()
                 .setDim(4, qkv_dims)
                 .setStrides(4, qkv_strides)
                 .setId(1)
                 .setDataType(CUDNN_DATA_HALF)
                 .build();
    auto k = mudnn_frontend::TensorBuilder()
                 .setDim(4, qkv_dims)
                 .setStrides(4, qkv_strides)
                 .setId(2)
                 .setDataType(CUDNN_DATA_HALF)
                 .build();
    auto v = mudnn_frontend::TensorBuilder()
                 .setDim(4, qkv_dims)
                 .setStrides(4, qkv_strides)
                 .setId(3)
                 .setDataType(CUDNN_DATA_HALF)
                 .build();
    auto o = mudnn_frontend::TensorBuilder()
                 .setDim(4, qkv_dims)
                 .setStrides(4, qkv_strides)
                 .setId(4)
                 .setDataType(CUDNN_DATA_HALF)
                 .setVirtual(true)
                 .build();
    auto d_o = mudnn_frontend::TensorBuilder()
                   .setDim(4, qkv_dims)
                   .setStrides(4, qkv_strides)
                   .setId(5)
                   .setDataType(CUDNN_DATA_HALF)
                   .build();
    auto stats = mudnn_frontend::TensorBuilder()
                     .setDim(4, stats_dims)
                     .setStrides(4, stats_strides)
                     .setId(6)
                     .setDataType(CUDNN_DATA_FLOAT)
                     .setVirtual(true)
                     .build();
    auto d_q = mudnn_frontend::TensorBuilder()
                   .setDim(4, qkv_dims)
                   .setStrides(4, qkv_strides)
                   .setId(7)
                   .setDataType(CUDNN_DATA_HALF)
                   .build();
    auto d_k = mudnn_frontend::TensorBuilder()
                   .setDim(4, qkv_dims)
                   .setStrides(4, qkv_strides)
                   .setId(8)
                   .setDataType(CUDNN_DATA_HALF)
                   .build();
    auto d_v = mudnn_frontend::TensorBuilder()
                   .setDim(4, qkv_dims)
                   .setStrides(4, qkv_strides)
                   .setId(9)
                   .setDataType(CUDNN_DATA_HALF)
                   .build();
    auto attn_probs = mudnn_frontend::TensorBuilder()
                          .setDim(4, attn_dims)
                          .setStrides(4, attn_strides)
                          .setId(100)
                          .setDataType(CUDNN_DATA_HALF)
                          .setVirtual(true)
                          .build();
    auto dropout_mask = mudnn_frontend::TensorBuilder()
                            .setDim(4, attn_dims)
                            .setStrides(4, attn_strides)
                            .setId(101)
                            .setDataType(CUDNN_DATA_BOOLEAN)
                            .setVirtual(true)
                            .build();

    auto fwd = mudnn_frontend::OperationBuilder(
                   CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)
                   .setQDesc(q)
                   .setKDesc(k)
                   .setVDesc(v)
                   .setODesc(o)
                   .setAttnProbsDesc(attn_probs)
                   .setDropoutMaskDesc(dropout_mask)
                   .setSdpaFwdStatsDesc(stats)
                   .setSdpaScale(0.125)
                   .setSdpaGenerateStats(true)
                   .build();
    auto bwd = mudnn_frontend::OperationBuilder(
                   CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)
                   .setQDesc(q)
                   .setKDesc(k)
                   .setVDesc(v)
                   .setODesc(o)
                   .setSdpaBwdDODesc(d_o)
                   .setSdpaBwdStatsDesc(stats)
                   .setSdpaBwdDQDesc(d_q)
                   .setSdpaBwdDKDesc(d_k)
                   .setSdpaBwdDVDesc(d_v)
                   .setSdpaBwdScale(0.125)
                   .build();
    auto bwd_causal = mudnn_frontend::OperationBuilder(
                          CUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)
                          .setQDesc(q)
                          .setKDesc(k)
                          .setVDesc(v)
                          .setODesc(o)
                          .setSdpaBwdDODesc(d_o)
                          .setSdpaBwdStatsDesc(stats)
                          .setSdpaBwdDQDesc(d_q)
                          .setSdpaBwdDKDesc(d_k)
                          .setSdpaBwdDVDesc(d_v)
                          .setSdpaBwdScale(0.125)
                          .setSdpaBwdCausalMask(true)
                          .build();
    int64_t bool_count = 0;
    bool got_causal_mask = false;
    CHECK_MUDNN(cudnnBackendGetAttribute(
        bwd_causal.get_backend_descriptor(),
        CUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK, CUDNN_TYPE_BOOLEAN, 1,
        &bool_count, &got_causal_mask));
    if (bool_count != 1 || !got_causal_mask) {
      std::cerr << "sdpa backward causal mask attribute was not set"
                << std::endl;
      cleanup();
      return false;
    }
    const mudnn_frontend::Operation* ops[] = {&fwd, &bwd};
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
      cleanup();
      return false;
    }
    auto plan = mudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(configs[0])
                    .build();

    constexpr uint16_t kHalfZero = 0x0000;
    constexpr uint16_t kHalfOne = 0x3c00;
    std::array<uint16_t, 128> zeros{};
    zeros.fill(kHalfZero);
    std::array<uint16_t, 128> ones{};
    ones.fill(kHalfOne);

    CHECK_MUSA(musaMalloc(&q_ptr, sizeof(uint16_t) * zeros.size()));
    CHECK_MUSA(musaMalloc(&k_ptr, sizeof(uint16_t) * zeros.size()));
    CHECK_MUSA(musaMalloc(&v_ptr, sizeof(uint16_t) * zeros.size()));
    CHECK_MUSA(musaMalloc(&d_o_ptr, sizeof(uint16_t) * ones.size()));
    CHECK_MUSA(musaMalloc(&d_q_ptr, sizeof(uint16_t) * zeros.size()));
    CHECK_MUSA(musaMalloc(&d_k_ptr, sizeof(uint16_t) * zeros.size()));
    CHECK_MUSA(musaMalloc(&d_v_ptr, sizeof(uint16_t) * zeros.size()));
    CHECK_MUSA(musaMemcpy(q_ptr, zeros.data(), sizeof(uint16_t) * zeros.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(k_ptr, zeros.data(), sizeof(uint16_t) * zeros.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(v_ptr, zeros.data(), sizeof(uint16_t) * zeros.size(),
                          musaMemcpyHostToDevice));
    CHECK_MUSA(musaMemcpy(d_o_ptr, ones.data(), sizeof(uint16_t) * ones.size(),
                          musaMemcpyHostToDevice));

    const int64_t workspace_size = plan.getWorkspaceSize();
    if (workspace_size > 0) {
      CHECK_MUSA(musaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }
    auto variant_builder = mudnn_frontend::VariantPackBuilder()
                               .setUids({1, 2, 3, 5, 7, 8, 9})
                               .setDataPointers({q_ptr, k_ptr, v_ptr, d_o_ptr,
                                                 d_q_ptr, d_k_ptr, d_v_ptr});
    if (workspace != nullptr) {
      variant_builder.setWorkspacePointer(workspace);
    }
    auto variant_pack = variant_builder.build();
    CHECK_MUDNN(mudnn_frontend::execute(handle, plan, variant_pack));
    SyncMudnnHandleStream(handle);

    std::array<uint16_t, 128> got_dq{};
    std::array<uint16_t, 128> got_dk{};
    std::array<uint16_t, 128> got_dv{};
    CHECK_MUSA(musaMemcpy(got_dq.data(), d_q_ptr,
                          sizeof(uint16_t) * got_dq.size(),
                          musaMemcpyDeviceToHost));
    CHECK_MUSA(musaMemcpy(got_dk.data(), d_k_ptr,
                          sizeof(uint16_t) * got_dk.size(),
                          musaMemcpyDeviceToHost));
    CHECK_MUSA(musaMemcpy(got_dv.data(), d_v_ptr,
                          sizeof(uint16_t) * got_dv.size(),
                          musaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < got_dq.size(); ++i) {
      const bool dq_ok = (got_dq[i] & 0x7fffu) == 0;
      const bool dk_ok = (got_dk[i] & 0x7fffu) == 0;
      const bool dv_ok = got_dv[i] == kHalfOne;
      if (!dq_ok || !dk_ok || !dv_ok) {
        std::cerr << "mudnn_frontend sdpa backward mismatch at " << i
                  << ": dQ=0x" << std::hex << got_dq[i] << ", dK=0x"
                  << got_dk[i] << ", dV=0x" << got_dv[i] << std::dec
                  << std::endl;
      }
      ok = ok && dq_ok && dk_ok && dv_ok;
    }

    cleanup();
    return ok;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend sdpa backward threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "mudnn_frontend sdpa backward threw unknown exception"
              << std::endl;
  }
  cleanup();
  return false;
}

bool MudnnFrontendGraphApiBuildsSdpaBackwardGraph() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));
  try {
    namespace fe = mudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    auto q = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("q")
                              .set_dim({1, 1, 2, 2})
                              .set_stride({4, 4, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto k = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("k")
                              .set_dim({1, 1, 3, 2})
                              .set_stride({6, 6, 2, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto v = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("v")
                              .set_dim({1, 1, 3, 4})
                              .set_stride({12, 12, 4, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto o = graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("o")
                              .set_dim({1, 1, 2, 4})
                              .set_stride({8, 8, 4, 1})
                              .set_data_type(fe::DataType_t::FLOAT));
    auto d_o = graph.tensor(fe::graph::Tensor_attributes()
                                .set_name("do")
                                .set_dim({1, 1, 2, 4})
                                .set_stride({8, 8, 4, 1})
                                .set_data_type(fe::DataType_t::FLOAT));
    auto stats = graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("stats")
                                  .set_dim({1, 1, 2, 1})
                                  .set_stride({2, 2, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
    auto bias = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("bias")
                                 .set_dim({1, 1, 2, 3})
                                 .set_stride({6, 6, 3, 1})
                                 .set_data_type(fe::DataType_t::FLOAT));
    auto d_bias = graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("dbias")
                                   .set_dim({1, 1, 2, 3})
                                   .set_stride({6, 6, 3, 1})
                                   .set_data_type(fe::DataType_t::FLOAT));
    auto mask = graph.tensor(fe::graph::Tensor_attributes()
                                 .set_name("mask")
                                 .set_dim({1, 1, 2, 3})
                                 .set_stride({6, 6, 3, 1})
                                 .set_data_type(fe::DataType_t::BOOLEAN));
    auto outputs =
        graph.sdpa_backward(q, k, v, o, d_o, stats,
                            fe::graph::Sdpa_attributes()
                                .set_name("attention_bwd")
                                .set_attn_scale(0.5)
                                .set_causal_mask(true)
                                .set_deterministic(true)
                                .set_compute_data_type(fe::DataType_t::FLOAT),
                            bias, d_bias, mask);
    if (std::get<0>(outputs)->get_dim() != q->get_dim() ||
        std::get<1>(outputs)->get_dim() != k->get_dim() ||
        std::get<2>(outputs)->get_dim() != v->get_dim()) {
      std::cerr << "sdpa_backward graph output dims mismatch" << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    std::get<0>(outputs)->set_output(true);
    std::get<1>(outputs)->set_output(true);
    std::get<2>(outputs)->set_output(true);
    d_bias->set_output(true);

    auto serialized = graph.serialize();
    fe::graph::Graph restored;
    auto status = restored.deserialize(serialized);
    if (!status.is_good()) {
      std::cerr << "sdpa_backward graph deserialize failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = restored.validate();
    if (!status.is_good()) {
      std::cerr << "sdpa_backward graph validate failed: " << status.message
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    status = restored.build_operation_graph(handle);
    if (status.is_good()) {
      std::cerr << "sdpa_backward FLOAT graph unexpectedly finalized"
                << std::endl;
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }

    CHECK_MUDNN(mudnnDestroy(handle));
    return true;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend sdpa_backward graph API threw: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr
        << "mudnn_frontend sdpa_backward graph API threw unknown exception"
        << std::endl;
  }
  if (handle != nullptr) {
    mudnnDestroy(handle);
  }
  return false;
}

}  // namespace mudnn_frontend_tests
