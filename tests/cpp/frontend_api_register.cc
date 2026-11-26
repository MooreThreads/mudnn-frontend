/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#include "frontend_api_common.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(mudnn_frontend_cpp_tests, m) {
  using namespace mudnn_frontend_tests;

  m.def("mudnn_frontend_builds_pointwise_add_graph",
        &MudnnFrontendBuildsPointwiseAddGraph);
  m.def("mudnn_frontend_executes_matmul_bias_relu_graph",
        &MudnnFrontendExecutesMatmulBiasReluGraph);
  m.def("mudnn_frontend_graph_api_executes_matmul_bias_graph",
        &MudnnFrontendGraphApiExecutesMatmulBiasGraph);
  m.def("mudnn_frontend_graph_api_executes_pointwise_aliases_graph",
        &MudnnFrontendGraphApiExecutesPointwiseAliasesGraph);
  m.def("mudnn_frontend_graph_api_executes_gen_index_graph",
        &MudnnFrontendGraphApiExecutesGenIndexGraph);
  m.def("mudnn_frontend_graph_api_executes_serialized_scale_graph",
        &MudnnFrontendGraphApiExecutesSerializedScaleGraph);
  m.def("mudnn_frontend_graph_api_executes_concat_reshape_graph",
        &MudnnFrontendGraphApiExecutesConcatReshapeGraph);
  m.def("mudnn_frontend_graph_api_executes_reduction_graph",
        &MudnnFrontendGraphApiExecutesReductionGraph);
  m.def("mudnn_frontend_graph_api_executes_genstats_graph",
        &MudnnFrontendGraphApiExecutesGenstatsGraph);
  m.def("mudnn_frontend_graph_api_executes_rng_graph",
        &MudnnFrontendGraphApiExecutesRngGraph);
  m.def("mudnn_frontend_graph_api_builds_batchnorm_graphs",
        &MudnnFrontendGraphApiBuildsBatchnormGraphs);
  m.def("mudnn_frontend_graph_api_builds_conv_resample_graphs",
        &MudnnFrontendGraphApiBuildsConvResampleGraphs);
  m.def("mudnn_frontend_graph_api_builds_batchnorm_forward_backward_graphs",
        &MudnnFrontendGraphApiBuildsBatchnormForwardBackwardGraphs);
  m.def("mudnn_frontend_graph_api_builds_norm_alias_graphs",
        &MudnnFrontendGraphApiBuildsNormAliasGraphs);
  m.def("mudnn_frontend_graph_api_builds_block_scale_graphs",
        &MudnnFrontendGraphApiBuildsBlockScaleGraphs);
  m.def("mudnn_frontend_graph_api_builds_paged_cache_signal_graphs",
        &MudnnFrontendGraphApiBuildsPagedCacheSignalGraphs);
  m.def("mudnn_frontend_graph_api_builds_sdpa_backward_graph",
        &MudnnFrontendGraphApiBuildsSdpaBackwardGraph);
  m.def("mudnn_frontend_graph_api_executes_sdpa_graph",
        &MudnnFrontendGraphApiExecutesSdpaGraph);
  m.def("mudnn_frontend_graph_api_executes_sdpa_bias_bool_mask_graph",
        &MudnnFrontendGraphApiExecutesSdpaBiasBoolMaskGraph);
  m.def("mudnn_frontend_graph_api_executes_sdpa_generate_stats_graph",
        &MudnnFrontendGraphApiExecutesSdpaGenerateStatsGraph);
  m.def("mudnn_frontend_executes_matmul_broadcast_bias_graph",
        &MudnnFrontendExecutesMatmulBroadcastBiasGraph);
  m.def("mudnn_frontend_executes_compare_select_graph",
        &MudnnFrontendExecutesCompareSelectGraph);
  m.def("mudnn_frontend_executes_scale_graph",
        &MudnnFrontendExecutesScaleGraph);
  m.def("mudnn_frontend_builds_sdpa_graph", &MudnnFrontendBuildsSdpaGraph);
  m.def("mudnn_frontend_executes_sdpa_bias_bool_mask_graph",
        &MudnnFrontendExecutesSdpaBiasBoolMaskGraph);
  m.def("mudnn_frontend_executes_sdpa_backward_graph",
        &MudnnFrontendExecutesSdpaBackwardGraph);
}
