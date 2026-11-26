#pragma once

#include "mudnn_frontend_Filters.h"

#include <functional>
#include <vector>

namespace mudnn_frontend {

enum class CudnnFindSamplingTechnique {
  CUDNN_FIND_SAMPLE_ONCE,
  CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE,
  CUDNN_FIND_SAMPLE_TILL_STABLE,
};

class ExecutionPlanCache {
 public:
  bool is_fastest_plan_stable(const OperationGraph&, const std::string&) const {
    return false;
  }
  void add_plan_to_cache(const OperationGraph&, const ExecutionPlan&) {}
};

class EngineConfigGenerator {
 public:
  using Source = std::function<EngineConfigList(OperationGraph&)>;

  EngineConfigGenerator(int source_count, Source* sources)
      : sources_(sources, sources + source_count) {}

  EngineConfigGenerator(int source_count, Source const* sources)
      : sources_(sources, sources + source_count) {}

  EngineConfigList generate_engine_config(OperationGraph& graph) {
    EngineConfigList configs;
    for (auto& source : sources_) {
      auto next = source(graph);
      configs.insert(configs.end(), std::make_move_iterator(next.begin()),
                     std::make_move_iterator(next.end()));
    }
    return configs;
  }

  executionPlans_t cudnnGetPlan(cudnnHandle_t handle, OperationGraph& graph,
                                size_t max_plans = 1000) {
    executionPlans_t plans;
    for (auto& config : generate_engine_config(graph)) {
      if (plans.size() >= max_plans) {
        break;
      }
      try {
        plans.push_back(ExecutionPlanBuilder()
                            .setHandle(handle)
                            .setEngineConfig(config)
                            .build());
      } catch (...) {
      }
    }
    return plans;
  }

  executionPlans_t cudnnGetPlan(cudnnHandle_t handle, OperationGraph& graph,
                                Predicate pred, size_t max_plans = 1000) {
    return filter(pred, cudnnGetPlan(handle, graph, max_plans));
  }

  template <CudnnFindSamplingTechnique>
  executionPlans_t cudnnFindPlan(cudnnHandle_t handle, OperationGraph& graph,
                                 const VariantPack&) {
    return cudnnGetPlan(handle, graph);
  }

  template <CudnnFindSamplingTechnique>
  executionPlans_t cudnnFindPlan(cudnnHandle_t handle, OperationGraph& graph,
                                 const VariantPack&, Predicate pred) {
    return cudnnGetPlan(handle, graph, pred);
  }

  template <CudnnFindSamplingTechnique samplingTechnique>
  ExecutionPlan cudnnFindPlanAndCache(cudnnHandle_t handle,
                                      OperationGraph& graph,
                                      const VariantPack& pack,
                                      ExecutionPlanCache& cache,
                                      Predicate pred) {
    auto plans = cudnnFindPlan<samplingTechnique>(handle, graph, pack, pred);
    if (plans.empty()) {
      return ExecutionPlan();
    }
    cache.add_plan_to_cache(graph, plans.front());
    return plans.front();
  }

 private:
  std::vector<Source> sources_;
};

using GeneratorSource = EngineConfigGenerator::Source;

}  // namespace mudnn_frontend
