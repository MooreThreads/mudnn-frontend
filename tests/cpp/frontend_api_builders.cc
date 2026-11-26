/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#include "frontend_api_common.h"

namespace mudnn_frontend_tests {

bool MudnnFrontendBuildsPointwiseAddGraph() {
  mudnnHandle_t handle = nullptr;
  CHECK_MUDNN(mudnnCreate(&handle));

  try {
    const int64_t dims[] = {1, 2, 3};
    const int64_t strides[] = {6, 3, 1};
    auto x = mudnn_frontend::TensorBuilder()
                 .setDim(3, dims)
                 .setStrides(3, strides)
                 .setId(1)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto b = mudnn_frontend::TensorBuilder()
                 .setDim(3, dims)
                 .setStrides(3, strides)
                 .setId(2)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto y = mudnn_frontend::TensorBuilder()
                 .setDim(3, dims)
                 .setStrides(3, strides)
                 .setId(3)
                 .setDataType(CUDNN_DATA_FLOAT)
                 .build();
    auto pw = mudnn_frontend::PointWiseDescBuilder()
                  .setMode(CUDNN_POINTWISE_ADD)
                  .setMathPrecision(CUDNN_DATA_FLOAT)
                  .build();
    auto swish_pw = mudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_SWISH_FWD)
                        .setMathPrecision(CUDNN_DATA_FLOAT)
                        .setSwishBeta(1.25)
                        .build();
    auto op = mudnn_frontend::OperationBuilder(
                  CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                  .setXDesc(x)
                  .setBDesc(b)
                  .setYDesc(y)
                  .setPwDesc(pw)
                  .setAlpha(1.0)
                  .setAlpha2(1.0)
                  .build();
    const mudnn_frontend::Operation* ops[] = {&op};
    auto graph = mudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(1, ops)
                     .build();
    int64_t count = 0;
    int64_t engine_count = 0;
    CHECK_MUDNN(
        cudnnBackendGetAttribute(graph.get_backend_descriptor(),
                                 CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                                 CUDNN_TYPE_INT64, 1, &count, &engine_count));
    auto heur = mudnn_frontend::EngineHeuristicsBuilder()
                    .setOperationGraph(graph)
                    .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                    .build();
    auto configs = heur.getEngineConfig();
    if (configs.empty()) {
      CHECK_MUDNN(mudnnDestroy(handle));
      return false;
    }
    auto plan_from_heur = mudnn_frontend::ExecutionPlanBuilder()
                              .setHandle(handle)
                              .setEngineConfig(configs[0])
                              .build();
    auto engine = mudnn_frontend::EngineBuilder()
                      .setOperationGraph(graph)
                      .setGlobalEngineIdx(0)
                      .build();
    auto config =
        mudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
    auto plan = mudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(config)
                    .build();
    int data[3] = {};
    void* ptrs[] = {&data[0], &data[1], &data[2]};
    int64_t uids[] = {1, 2, 3};
    auto variant_pack = mudnn_frontend::VariantPackBuilder()
                            .setUids(3, uids)
                            .setDataPointers(3, ptrs)
                            .build();
    auto reduction_x = mudnn_frontend::TensorBuilder()
                           .setDim({2, 3})
                           .setStrides({3, 1})
                           .setId(4)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();
    auto reduction_y = mudnn_frontend::TensorBuilder()
                           .setDim({2, 1})
                           .setStrides({1, 1})
                           .setId(5)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();
    auto reduction = mudnn_frontend::ReductionDescBuilder()
                         .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                         .setMathPrecision(CUDNN_DATA_FLOAT)
                         .build();
    auto reduction_op = mudnn_frontend::OperationBuilder(
                            CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                            .setReductionDesc(reduction)
                            .setReductionXDesc(reduction_x)
                            .setReductionYDesc(reduction_y)
                            .build();
    auto conv_x = mudnn_frontend::TensorBuilder()
                      .setDim({1, 1, 3, 3})
                      .setStrides({9, 9, 3, 1})
                      .setId(6)
                      .setDataType(CUDNN_DATA_FLOAT)
                      .build();
    auto conv_w = mudnn_frontend::TensorBuilder()
                      .setDim({1, 1, 2, 2})
                      .setStrides({4, 4, 2, 1})
                      .setId(7)
                      .setDataType(CUDNN_DATA_FLOAT)
                      .build();
    auto conv_y = mudnn_frontend::TensorBuilder()
                      .setDim({1, 1, 2, 2})
                      .setStrides({4, 4, 2, 1})
                      .setId(8)
                      .setDataType(CUDNN_DATA_FLOAT)
                      .build();
    auto conv = mudnn_frontend::ConvolutionDescBuilder()
                    .setMathPrecision(CUDNN_DATA_FLOAT)
                    .setConvolutionMode(CUDNN_CROSS_CORRELATION)
                    .setPrePadding({0, 0})
                    .setPostPadding({0, 0})
                    .setStride({1, 1})
                    .setDilation({1, 1})
                    .build();
    auto conv_op = mudnn_frontend::OperationBuilder(
                       CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                       .setConvDesc(conv)
                       .setConvXDesc(conv_x)
                       .setConvWDesc(conv_w)
                       .setConvYDesc(conv_y)
                       .setAlpha(1.0)
                       .setBeta(0.0)
                       .build();
    auto conv_dy = mudnn_frontend::TensorBuilder()
                       .setDim({1, 1, 2, 2})
                       .setStrides({4, 4, 2, 1})
                       .setId(9)
                       .setDataType(CUDNN_DATA_FLOAT)
                       .build();
    auto conv_dx = mudnn_frontend::TensorBuilder()
                       .setDim({1, 1, 3, 3})
                       .setStrides({9, 9, 3, 1})
                       .setId(10)
                       .setDataType(CUDNN_DATA_FLOAT)
                       .build();
    auto conv_dw = mudnn_frontend::TensorBuilder()
                       .setDim({1, 1, 2, 2})
                       .setStrides({4, 4, 2, 1})
                       .setId(11)
                       .setDataType(CUDNN_DATA_FLOAT)
                       .build();
    auto conv_dgrad_op =
        mudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
            .setConvDesc(conv)
            .setConvWDesc(conv_w)
            .setConvDXDesc(conv_dx)
            .setConvDYDesc(conv_dy)
            .setAlpha(1.0)
            .setBeta(0.0)
            .build();
    auto conv_wgrad_op =
        mudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
            .setConvDesc(conv)
            .setConvDWDesc(conv_dw)
            .setConvXDesc(conv_x)
            .setConvDYDesc(conv_dy)
            .setAlpha(1.0)
            .setBeta(0.0)
            .build();
    auto pool_x = mudnn_frontend::TensorBuilder()
                      .setDim({1, 1, 4, 4})
                      .setStrides({16, 16, 4, 1})
                      .setId(12)
                      .setDataType(CUDNN_DATA_FLOAT)
                      .build();
    auto pool_y = mudnn_frontend::TensorBuilder()
                      .setDim({1, 1, 2, 2})
                      .setStrides({4, 4, 2, 1})
                      .setId(13)
                      .setDataType(CUDNN_DATA_FLOAT)
                      .build();
    auto pool_dx = mudnn_frontend::TensorBuilder()
                       .setDim({1, 1, 4, 4})
                       .setStrides({16, 16, 4, 1})
                       .setId(14)
                       .setDataType(CUDNN_DATA_FLOAT)
                       .build();
    auto pool = mudnn_frontend::PoolingDescBuilder()
                    .setMode(CUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING)
                    .setMathPrecision(CUDNN_DATA_FLOAT)
                    .setWindowDim({2, 2})
                    .setPrePadding({0, 0})
                    .setPostPadding({0, 0})
                    .setStride({2, 2})
                    .build();
    auto pool_fwd_op = mudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                           .setPoolingDesc(pool)
                           .setPoolingXDesc(pool_x)
                           .setPoolingYDesc(pool_y)
                           .setAlpha(1.0)
                           .setBeta(0.0)
                           .build();
    auto pool_bwd_op = mudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                           .setPoolingDesc(pool)
                           .setPoolingDxDesc(pool_dx)
                           .setPoolingDyDesc(pool_y)
                           .setAlpha(1.0)
                           .setBeta(0.0)
                           .build();
    auto concat_a = mudnn_frontend::TensorBuilder()
                        .setDim({2, 2})
                        .setStrides({2, 1})
                        .setId(15)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();
    auto concat_b = mudnn_frontend::TensorBuilder()
                        .setDim({2, 3})
                        .setStrides({3, 1})
                        .setId(16)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();
    auto concat_y = mudnn_frontend::TensorBuilder()
                        .setDim({2, 5})
                        .setStrides({5, 1})
                        .setId(17)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();
    auto reshape_y = mudnn_frontend::TensorBuilder()
                         .setDim({10})
                         .setStrides({1})
                         .setId(18)
                         .setDataType(CUDNN_DATA_FLOAT)
                         .build();
    std::vector<mudnn_frontend::Tensor> concat_inputs = {concat_a, concat_b};
    auto concat_op = mudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR)
                         .setConcatAxis(1)
                         .setConcatInputDescs(concat_inputs)
                         .setConcatOutputDesc(concat_y)
                         .build();
    auto reshape_op = mudnn_frontend::OperationBuilder(
                          CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                          .setReshapeXDesc(concat_y)
                          .setReshapeYDesc(reshape_y)
                          .build();
    auto rng_y = mudnn_frontend::TensorBuilder()
                     .setDim({2, 4})
                     .setStrides({4, 1})
                     .setId(19)
                     .setDataType(CUDNN_DATA_FLOAT)
                     .build();
    auto rng_offset = mudnn_frontend::TensorBuilder()
                          .setDim({1})
                          .setStrides({1})
                          .setId(20)
                          .setDataType(CUDNN_DATA_INT64)
                          .build();
    auto rng = mudnn_frontend::RngDescBuilder()
                   .setDistribution(CUDNN_RNG_DISTRIBUTION_UNIFORM)
                   .setUniformDistMinimum(-0.5)
                   .setUniformDistMaximum(0.5)
                   .build();
    auto rng_op =
        mudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
            .setRngDesc(rng)
            .setRngYDesc(rng_y)
            .setRngSeed(123)
            .setRngOffsetDesc(rng_offset)
            .build();
    auto norm_x = mudnn_frontend::TensorBuilder()
                      .setDim({2, 2, 2, 2})
                      .setStrides({8, 4, 2, 1})
                      .setId(21)
                      .setDataType(CUDNN_DATA_FLOAT)
                      .build();
    auto norm_param = mudnn_frontend::TensorBuilder()
                          .setDim({1, 2, 1, 1})
                          .setStrides({2, 1, 1, 1})
                          .setId(22)
                          .setDataType(CUDNN_DATA_FLOAT)
                          .build();
    auto genstats_sqsum = mudnn_frontend::TensorBuilder()
                              .setDim({1, 2, 1, 1})
                              .setStrides({2, 1, 1, 1})
                              .setId(34)
                              .setDataType(CUDNN_DATA_FLOAT)
                              .build();
    auto bn_finalize_accum = mudnn_frontend::TensorBuilder()
                                 .setDim({1})
                                 .setStrides({1})
                                 .setId(35)
                                 .setDataType(CUDNN_DATA_INT64)
                                 .build();
    auto norm_y = mudnn_frontend::TensorBuilder()
                      .setDim({2, 2, 2, 2})
                      .setStrides({8, 4, 2, 1})
                      .setId(23)
                      .setDataType(CUDNN_DATA_FLOAT)
                      .build();
    auto genstats_op = mudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR)
                           .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
                           .setGenStatsMathPrecision(CUDNN_DATA_FLOAT)
                           .setGenStatsXDesc(norm_x)
                           .setGenStatsSumDesc(norm_param)
                           .setGenStatsSqSumDesc(genstats_sqsum)
                           .build();
    auto bn_finalize_inf_op =
        mudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
            .setBnFinalizeStatsMode(CUDNN_BN_FINALIZE_STATISTICS_INFERENCE)
            .setBnFinalizeMathPrecision(CUDNN_DATA_FLOAT)
            .setBnFinalizeScaleDesc(norm_param)
            .setBnFinalizeBiasDesc(norm_param)
            .setBnFinalizePrevRunningMeanDesc(norm_param)
            .setBnFinalizePrevRunningVarDesc(norm_param)
            .setBnFinalizeEqScaleDesc(norm_param)
            .setBnFinalizeEqBiasDesc(genstats_sqsum)
            .setBnFinalizeEpsilon(1.0e-5)
            .build();
    auto bn_finalize_train_op =
        mudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
            .setBnFinalizeStatsMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
            .setBnFinalizeMathPrecision(CUDNN_DATA_FLOAT)
            .setBnFinalizeYSumDesc(norm_param)
            .setBnFinalizeYSqSumDesc(genstats_sqsum)
            .setBnFinalizeScaleDesc(norm_param)
            .setBnFinalizeBiasDesc(norm_param)
            .setBnFinalizePrevRunningMeanDesc(norm_param)
            .setBnFinalizePrevRunningVarDesc(norm_param)
            .setBnFinalizeUpdatedRunningMeanDesc(norm_param)
            .setBnFinalizeUpdatedRunningVarDesc(genstats_sqsum)
            .setBnFinalizeSavedMeanDesc(norm_param)
            .setBnFinalizeSavedInvStdDesc(genstats_sqsum)
            .setBnFinalizeEqScaleDesc(norm_param)
            .setBnFinalizeEqBiasDesc(genstats_sqsum)
            .setBnFinalizeAccumCountDesc(bn_finalize_accum)
            .setBnFinalizeExpAvgFactor(0.25)
            .setBnFinalizeEpsilon(1.0e-5)
            .build();
    auto bn_bwd_weights_op =
        mudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)
            .setBnBwdWeightsMathPrecision(CUDNN_DATA_FLOAT)
            .setBnBwdWeightsMeanDesc(norm_param)
            .setBnBwdWeightsInvstdDesc(norm_param)
            .setBnBwdWeightsBnScaleDesc(norm_param)
            .setBnBwdWeightsXDesc(norm_x)
            .setBnBwdWeightsDyDesc(norm_y)
            .setBnBwdWeightsDBnScaleDesc(norm_param)
            .setBnBwdWeightsDBnBiasDesc(genstats_sqsum)
            .setBnBwdWeightsEqDyScaleDesc(norm_param)
            .setBnBwdWeightsEqXScaleDesc(genstats_sqsum)
            .setBnBwdWeightsEqBiasDesc(norm_param)
            .build();
    auto norm_op = mudnn_frontend::OperationBuilder(
                       CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
                       .setNormMode(CUDNN_BATCH_NORM)
                       .setNormFwdPhase(CUDNN_NORM_FWD_INFERENCE)
                       .setNormFwdXDesc(norm_x)
                       .setNormFwdMeanDesc(norm_param)
                       .setNormFwdInvVarianceDesc(norm_param)
                       .setNormFwdScaleDesc(norm_param)
                       .setNormFwdBiasDesc(norm_param)
                       .setNormFwdYDesc(norm_y)
                       .setNormFwdEpsilon(1.0e-5)
                       .build();
    auto norm_train_op = mudnn_frontend::OperationBuilder(
                             CUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
                             .setNormMode(CUDNN_BATCH_NORM)
                             .setNormFwdPhase(CUDNN_NORM_FWD_TRAINING)
                             .setNormFwdXDesc(norm_x)
                             .setNormFwdMeanDesc(norm_param)
                             .setNormFwdInvVarianceDesc(norm_param)
                             .setNormFwdScaleDesc(norm_param)
                             .setNormFwdBiasDesc(norm_param)
                             .setNormFwdInputRunningMeanDesc(norm_param)
                             .setNormFwdInputRunningVarDesc(norm_param)
                             .setNormFwdOutputRunningMeanDesc(norm_param)
                             .setNormFwdOutputRunningVarDesc(norm_param)
                             .setNormFwdYDesc(norm_y)
                             .setNormFwdExpAvgFactor(0.25)
                             .setNormFwdEpsilon(1.0e-5)
                             .build();
    auto norm_bwd_op = mudnn_frontend::OperationBuilder(
                           CUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
                           .setNormBwdMode(CUDNN_BATCH_NORM)
                           .setNormBwdXDesc(norm_x)
                           .setNormBwdMeanDesc(norm_param)
                           .setNormBwdInvVarianceDesc(norm_param)
                           .setNormBwdDyDesc(norm_x)
                           .setNormBwdScaleDesc(norm_param)
                           .setNormBwdDxDesc(norm_y)
                           .setNormBwdDScaleDesc(norm_param)
                           .setNormBwdDBiasDesc(norm_param)
                           .setNormBwdEpsilon(1.0e-5)
                           .build();
    auto block_x = mudnn_frontend::TensorBuilder()
                       .setDim({2, 5})
                       .setStrides({5, 1})
                       .setId(24)
                       .setDataType(CUDNN_DATA_FLOAT)
                       .build();
    auto block_scale = mudnn_frontend::TensorBuilder()
                           .setDim({2, 3})
                           .setStrides({3, 1})
                           .setId(25)
                           .setDataType(CUDNN_DATA_FLOAT)
                           .build();
    auto block_y = mudnn_frontend::TensorBuilder()
                       .setDim({2, 5})
                       .setStrides({5, 1})
                       .setId(26)
                       .setDataType(CUDNN_DATA_FLOAT)
                       .build();
    auto block_quant_op =
        mudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR)
            .setBlockScaleQuantizeXDesc(block_x)
            .setBlockScaleQuantizeYDesc(block_y)
            .setBlockScaleQuantizeScaleDesc(block_scale)
            .setBlockScaleQuantizeMathPrecision(CUDNN_DATA_FLOAT)
            .setBlockScaleQuantizeBlockSize(2)
            .build();
    auto block_dequant_op =
        mudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR)
            .setBlockScaleDequantizeXDesc(block_y)
            .setBlockScaleDequantizeScaleDesc(block_scale)
            .setBlockScaleDequantizeYDesc(block_x)
            .setBlockScaleDequantizeMathPrecision(CUDNN_DATA_FLOAT)
            .setBlockScaleDequantizeBlockSize(2)
            .build();
    auto paged_container = mudnn_frontend::TensorBuilder()
                               .setDim({3, 2, 2})
                               .setStrides({4, 2, 1})
                               .setId(27)
                               .setDataType(CUDNN_DATA_FLOAT)
                               .build();
    auto paged_y = mudnn_frontend::TensorBuilder()
                       .setDim({2, 2, 2})
                       .setStrides({4, 2, 1})
                       .setId(28)
                       .setDataType(CUDNN_DATA_FLOAT)
                       .build();
    auto paged_sequence = mudnn_frontend::TensorBuilder()
                              .setDim({2})
                              .setStrides({1})
                              .setId(29)
                              .setDataType(CUDNN_DATA_INT64)
                              .build();
    auto paged_page_table = mudnn_frontend::TensorBuilder()
                                .setDim({3})
                                .setStrides({1})
                                .setId(30)
                                .setDataType(CUDNN_DATA_INT64)
                                .build();
    auto paged_cache_op =
        mudnn_frontend::OperationBuilder(
            CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR)
            .setPagedCacheLoadContainerDesc(paged_container)
            .setPagedCacheLoadYDesc(paged_y)
            .setPagedCacheLoadSequenceDesc(paged_sequence)
            .setPagedCacheLoadPageTableDesc(paged_page_table)
            .build();
    auto signal_x = mudnn_frontend::TensorBuilder()
                        .setDim({2, 3})
                        .setStrides({3, 1})
                        .setId(31)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();
    auto signal_y = mudnn_frontend::TensorBuilder()
                        .setDim({2, 3})
                        .setStrides({3, 1})
                        .setId(32)
                        .setDataType(CUDNN_DATA_FLOAT)
                        .build();
    auto signal_flag = mudnn_frontend::TensorBuilder()
                           .setDim({1})
                           .setStrides({1})
                           .setId(33)
                           .setDataType(CUDNN_DATA_INT64)
                           .build();
    auto signal_op = mudnn_frontend::OperationBuilder(
                         CUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR)
                         .setSignalMode(CUDNN_SIGNAL_SET)
                         .setSignalValue(17)
                         .setSignalFlagDesc(signal_flag)
                         .setSignalXDesc(signal_x)
                         .setSignalYDesc(signal_y)
                         .build();
    CHECK_MUDNN(mudnnDestroy(handle));
    return count == 1 && engine_count >= 1 &&
        plan_from_heur.getWorkspaceSize() == plan.getWorkspaceSize() &&
        variant_pack.get_backend_descriptor() != nullptr &&
        swish_pw.get_backend_descriptor() != nullptr &&
        reduction_op.get_backend_descriptor() != nullptr &&
        conv_op.get_backend_descriptor() != nullptr &&
        conv_dgrad_op.get_backend_descriptor() != nullptr &&
        conv_wgrad_op.get_backend_descriptor() != nullptr &&
        pool_fwd_op.get_backend_descriptor() != nullptr &&
        pool_bwd_op.get_backend_descriptor() != nullptr &&
        concat_op.get_backend_descriptor() != nullptr &&
        reshape_op.get_backend_descriptor() != nullptr &&
        rng_op.get_backend_descriptor() != nullptr &&
        genstats_op.get_backend_descriptor() != nullptr &&
        bn_finalize_inf_op.get_backend_descriptor() != nullptr &&
        bn_finalize_train_op.get_backend_descriptor() != nullptr &&
        bn_bwd_weights_op.get_backend_descriptor() != nullptr &&
        norm_op.get_backend_descriptor() != nullptr &&
        norm_train_op.get_backend_descriptor() != nullptr &&
        norm_bwd_op.get_backend_descriptor() != nullptr &&
        block_quant_op.get_backend_descriptor() != nullptr &&
        block_dequant_op.get_backend_descriptor() != nullptr &&
        paged_cache_op.get_backend_descriptor() != nullptr &&
        signal_op.get_backend_descriptor() != nullptr;
  } catch (const std::exception& e) {
    std::cerr << "mudnn_frontend pointwise builder smoke threw: " << e.what()
              << std::endl;
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  } catch (...) {
    std::cerr
        << "mudnn_frontend pointwise builder smoke threw unknown exception"
        << std::endl;
    if (handle != nullptr) {
      mudnnDestroy(handle);
    }
    return false;
  }
}

}  // namespace mudnn_frontend_tests
