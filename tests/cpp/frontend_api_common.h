/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#pragma once

#include "mudnn_frontend.h"

#include <musa_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mudnn_frontend_tests {

inline void CheckMudnn(mudnnStatus_t status, const char* expr) {
  if (status == MUDNN_STATUS_SUCCESS) {
    return;
  }
  throw std::runtime_error(std::string(expr) +
                           " failed: " + mudnnGetErrorString(status));
}

inline void CheckMusa(musaError_t status, const char* expr) {
  if (status == musaSuccess) {
    return;
  }
  throw std::runtime_error(std::string(expr) +
                           " failed: " + musaGetErrorString(status));
}

inline bool HasMusaDevice() {
  int device_count = 0;
  return musaGetDeviceCount(&device_count) == musaSuccess && device_count > 0;
}

inline void SyncMudnnHandleStream(mudnnHandle_t) {
  CheckMusa(musaDeviceSynchronize(), "musaDeviceSynchronize()");
}

bool MudnnFrontendBuildsPointwiseAddGraph();
bool MudnnFrontendExecutesMatmulBiasReluGraph();
bool MudnnFrontendGraphApiExecutesMatmulBiasGraph();
bool MudnnFrontendGraphApiExecutesPointwiseAliasesGraph();
bool MudnnFrontendGraphApiExecutesGenIndexGraph();
bool MudnnFrontendGraphApiExecutesSerializedScaleGraph();
bool MudnnFrontendGraphApiExecutesConcatReshapeGraph();
bool MudnnFrontendGraphApiExecutesReductionGraph();
bool MudnnFrontendGraphApiExecutesGenstatsGraph();
bool MudnnFrontendGraphApiExecutesRngGraph();
bool MudnnFrontendGraphApiExecutesSdpaGraph();
bool MudnnFrontendGraphApiExecutesSdpaBiasBoolMaskGraph();
bool MudnnFrontendGraphApiExecutesSdpaGenerateStatsGraph();
bool MudnnFrontendExecutesMatmulBroadcastBiasGraph();
bool MudnnFrontendExecutesCompareSelectGraph();
bool MudnnFrontendExecutesScaleGraph();
bool MudnnFrontendBuildsSdpaGraph();
bool MudnnFrontendExecutesSdpaBiasBoolMaskGraph();
bool MudnnFrontendExecutesSdpaBackwardGraph();
bool MudnnFrontendGraphApiBuildsBatchnormGraphs();
bool MudnnFrontendGraphApiBuildsConvResampleGraphs();
bool MudnnFrontendGraphApiBuildsBatchnormForwardBackwardGraphs();
bool MudnnFrontendGraphApiBuildsNormAliasGraphs();
bool MudnnFrontendGraphApiBuildsBlockScaleGraphs();
bool MudnnFrontendGraphApiBuildsPagedCacheSignalGraphs();
bool MudnnFrontendGraphApiBuildsSdpaBackwardGraph();

}  // namespace mudnn_frontend_tests

#define CHECK_MUDNN(expr) ::mudnn_frontend_tests::CheckMudnn((expr), #expr)
#define CHECK_MUSA(expr) ::mudnn_frontend_tests::CheckMusa((expr), #expr)
