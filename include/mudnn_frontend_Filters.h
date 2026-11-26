#pragma once

#include "mudnn_frontend.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>

namespace mudnn_frontend {

using EngineConfigList = std::vector<EngineConfig>;
using executionPlans_t = std::vector<ExecutionPlan>;
using Predicate = std::function<bool(cudnnBackendDescriptor_t)>;

inline void filter(EngineConfigList& from, EngineConfigList& to,
                   std::function<bool(cudnnBackendDescriptor_t)> filter_fn) {
  auto partition = std::stable_partition(
      from.begin(), from.end(), [&](EngineConfig& config) {
        return filter_fn(config.get_backend_descriptor());
      });
  to.insert(to.end(), std::make_move_iterator(partition),
            std::make_move_iterator(from.end()));
  from.erase(partition, from.end());
}

inline executionPlans_t filter(Predicate pred, executionPlans_t plans) {
  executionPlans_t kept;
  for (auto& plan : plans) {
    if (!pred || !pred(plan.get_backend_descriptor())) {
      kept.push_back(std::move(plan));
    }
  }
  return kept;
}

template <mudnnBackendNumericalNote_t>
inline bool hasNumericalNote(cudnnBackendDescriptor_t) {
  return false;
}

template <mudnnBackendBehaviorNote_t>
inline bool hasBehaviorNote(cudnnBackendDescriptor_t) {
  return false;
}

}  // namespace mudnn_frontend
