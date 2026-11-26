/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 */

#include "frontend_api_common.h"

// Keep the public node-header layout source-compatible with cudnn-frontend.
// These are intentionally included one by one so a missing/renamed wrapper is
// caught by the normal -Werror frontend build rather than by downstream users.
#include <mudnn_frontend/node/adaptive_layernorm.h>
#include <mudnn_frontend/node/batchnorm.h>
#include <mudnn_frontend/node/batchnorm_inference.h>
#include <mudnn_frontend/node/block_scale_dequantize.h>
#include <mudnn_frontend/node/block_scale_quantize.h>
#include <mudnn_frontend/node/bn_finalize.h>
#include <mudnn_frontend/node/concatenate.h>
#include <mudnn_frontend/node/conv_dgrad.h>
#include <mudnn_frontend/node/conv_fprop.h>
#include <mudnn_frontend/node/conv_wgrad.h>
#include <mudnn_frontend/node/dbn.h>
#include <mudnn_frontend/node/dbn_weight.h>
#include <mudnn_frontend/node/diagonal_band_mask.h>
#include <mudnn_frontend/node/dln.h>
#include <mudnn_frontend/node/genstats.h>
#include <mudnn_frontend/node/instancenorm.h>
#include <mudnn_frontend/node/layernorm.h>
#include <mudnn_frontend/node/matmul.h>
#include <mudnn_frontend/node/matmul_fp8.h>
#include <mudnn_frontend/node/moe_grouped_matmul.h>
#include <mudnn_frontend/node/moe_grouped_matmul_bwd.h>
#include <mudnn_frontend/node/paged_cache_load.h>
#include <mudnn_frontend/node/pointwise.h>
#include <mudnn_frontend/node/reduction.h>
#include <mudnn_frontend/node/resample.h>
#include <mudnn_frontend/node/reshape.h>
#include <mudnn_frontend/node/rmsnorm.h>
#include <mudnn_frontend/node/rng.h>
#include <mudnn_frontend/node/rope.h>
#include <mudnn_frontend/node/rope_backward.h>
#include <mudnn_frontend/node/scaled_dot_product_flash_attention.h>
#include <mudnn_frontend/node/sdpa_fp8_bwd.h>
#include <mudnn_frontend/node/sdpa_support_surface.h>
#include <mudnn_frontend/node/slice.h>
#include <mudnn_frontend/node/softmax.h>
#include <mudnn_frontend/node/transpose.h>
#include <mudnn_frontend_Engine.h>
#include <mudnn_frontend_ExecutionPlan.h>
#include <mudnn_frontend_Operation.h>
#include <mudnn_frontend_OperationGraph.h>
#include <mudnn_frontend_Tensor.h>
#include <mudnn_frontend_VariantPack.h>

#include <type_traits>

namespace mudnn_frontend_tests {

static_assert(CUDNN_TYPE_BACKEND_DESCRIPTOR == MUDNN_TYPE_BACKEND_DESCRIPTOR);
static_assert(CUDNN_ATTR_TENSOR_DIMENSIONS == MUDNN_ATTR_TENSOR_DIMENSIONS);
static_assert(CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC ==
              MUDNN_ATTR_OPERATION_SDPA_FWD_QDESC);
static_assert(CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC ==
              MUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC);
static_assert(CUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK ==
              MUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK);
static_assert(CUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK ==
              MUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK);
static_assert(CUDNN_POINTWISE_SQRT == MUDNN_POINTWISE_SQRT);
static_assert(CUDNN_POINTWISE_GELU_APPROX_TANH_FWD ==
              MUDNN_POINTWISE_GELU_APPROX_TANH_FWD);
static_assert(CUDNN_POINTWISE_SWISH_FWD == MUDNN_POINTWISE_SWISH_FWD);
static_assert(CUDNN_POINTWISE_RELU_BWD == MUDNN_POINTWISE_RELU_BWD);
static_assert(CUDNN_POINTWISE_SWISH_BWD == MUDNN_POINTWISE_SWISH_BWD);
static_assert(CUDNN_ATTR_POINTWISE_SWISH_BETA ==
              MUDNN_ATTR_POINTWISE_SWISH_BETA);
static_assert(CUDNN_POINTWISE_CMP_EQ == MUDNN_POINTWISE_CMP_EQ);
static_assert(CUDNN_POINTWISE_CMP_GT == MUDNN_POINTWISE_CMP_GT);
static_assert(CUDNN_POINTWISE_CMP_GE == MUDNN_POINTWISE_CMP_GE);
static_assert(CUDNN_POINTWISE_CMP_LT == MUDNN_POINTWISE_CMP_LT);
static_assert(CUDNN_POINTWISE_CMP_LE == MUDNN_POINTWISE_CMP_LE);
static_assert(CUDNN_POINTWISE_LOGICAL_OR == MUDNN_POINTWISE_LOGICAL_OR);
static_assert(CUDNN_POINTWISE_LOGICAL_NOT == MUDNN_POINTWISE_LOGICAL_NOT);
static_assert(CUDNN_POINTWISE_GEN_INDEX == MUDNN_POINTWISE_GEN_INDEX);
static_assert(CUDNN_POINTWISE_BINARY_SELECT == MUDNN_POINTWISE_BINARY_SELECT);
static_assert(CUDNN_ATTR_OPERATION_POINTWISE_DXDESC ==
              MUDNN_ATTR_OPERATION_POINTWISE_DXDESC);
static_assert(CUDNN_ATTR_OPERATION_POINTWISE_DYDESC ==
              MUDNN_ATTR_OPERATION_POINTWISE_DYDESC);
static_assert(CUDNN_ATTR_OPERATION_POINTWISE_TDESC ==
              MUDNN_ATTR_OPERATION_POINTWISE_TDESC);
static_assert(CUDNN_REDUCE_TENSOR_ADD == MUDNN_REDUCE_TENSOR_ADD);
static_assert(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR ==
              MUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
static_assert(CUDNN_ATTR_OPERATION_REDUCTION_DESC ==
              MUDNN_ATTR_OPERATION_REDUCTION_DESC);
static_assert(CUDNN_RESAMPLE_MAXPOOL == MUDNN_RESAMPLE_MAXPOOL);
static_assert(CUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR ==
              MUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
static_assert(CUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC ==
              MUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC);
static_assert(CUDNN_ATTR_ENGINECFG_KNOB_CHOICES ==
              MUDNN_ATTR_ENGINECFG_KNOB_CHOICES);
static_assert(CUDNN_ATTR_ENGINE_KNOB_INFO == MUDNN_ATTR_ENGINE_KNOB_INFO);
static_assert(CUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED ==
              MUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED);
static_assert(CUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED ==
              MUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED);
static_assert(CUDNN_ATTR_TENSOR_VECTOR_COUNT == MUDNN_ATTR_TENSOR_VECTOR_COUNT);
static_assert(CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC ==
              MUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC);
static_assert(std::is_same_v<mudnn_frontend::Tensor_v8,
                             mudnn_frontend::Tensor>);
static_assert(std::is_same_v<mudnn_frontend::TensorBuilder_v8,
                             mudnn_frontend::TensorBuilder>);
static_assert(std::is_same_v<mudnn_frontend::Operation_v8,
                             mudnn_frontend::Operation>);
static_assert(std::is_same_v<mudnn_frontend::OperationGraph_v8,
                             mudnn_frontend::OperationGraph>);
static_assert(std::is_same_v<mudnn_frontend::Engine_v8,
                             mudnn_frontend::Engine>);
static_assert(std::is_same_v<mudnn_frontend::ExecutionPlan_v8,
                             mudnn_frontend::ExecutionPlan>);
static_assert(std::is_same_v<mudnn_frontend::VariantPack_v8,
                             mudnn_frontend::VariantPack>);
using CompatGraph = mudnn_frontend::graph::Graph;
using CompatTensor =
    std::shared_ptr<mudnn_frontend::graph::Tensor_attributes>;
static_assert(std::is_same_v<
              decltype(std::declval<CompatGraph&>().genstats(
                  std::declval<CompatTensor>(),
                  mudnn_frontend::graph::Genstats_attributes{})),
              std::array<CompatTensor, 2>>);
static_assert(std::is_same_v<
              decltype(std::declval<CompatGraph&>().sdpa(
                  std::declval<CompatTensor>(), std::declval<CompatTensor>(),
                  std::declval<CompatTensor>(),
                  mudnn_frontend::graph::SDPA_attributes{})),
              std::array<CompatTensor, 2>>);
static_assert(std::is_same_v<mudnn_frontend::graph::SDPA_attributes,
                             mudnn_frontend::graph::Sdpa_attributes>);
static_assert(std::is_same_v<mudnn_frontend::graph::Conv_fprop_attributes,
                             mudnn_frontend::graph::Conv_attributes>);
static_assert(std::is_same_v<mudnn_frontend::graph::Rmsnorm_attributes,
                             mudnn_frontend::graph::Norm_attributes>);
static_assert(std::is_same_v<mudnn_frontend::graph::BN_finalize_attributes,
                             mudnn_frontend::graph::Bn_finalize_attributes>);
static_assert(std::is_same_v<mudnn_frontend::graph::PagedCacheLoad_attributes,
                             mudnn_frontend::graph::Paged_cache_load_attributes>);
static_assert(CUDNN_CONVOLUTION == MUDNN_CONVOLUTION);
static_assert(CUDNN_CROSS_CORRELATION == MUDNN_CROSS_CORRELATION);
static_assert(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR ==
              MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
static_assert(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR ==
              MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR);
static_assert(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR ==
              MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
static_assert(CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC ==
              MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC);
static_assert(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC ==
              MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC);
static_assert(CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC ==
              MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC);
}  // namespace mudnn_frontend_tests
