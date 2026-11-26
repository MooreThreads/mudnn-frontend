#pragma once

class KernelCache;

class OperationBuilder {
 public:
  explicit OperationBuilder(mudnnBackendDescriptorType_t type) : type_(type) {}
  explicit OperationBuilder(DescriptorType_t type)
      : type_(static_cast<mudnnBackendDescriptorType_t>(type)) {}

  OperationBuilder& setxDesc(const Tensor& tensor) {
    x_ = &tensor;
    return *this;
  }
  OperationBuilder& setXDesc(const Tensor& tensor) { return setxDesc(tensor); }
  OperationBuilder& setbDesc(const Tensor& tensor) {
    b_ = &tensor;
    return *this;
  }
  OperationBuilder& setBDesc(const Tensor& tensor) { return setbDesc(tensor); }
  OperationBuilder& settDesc(const Tensor& tensor) {
    t_ = &tensor;
    return *this;
  }
  OperationBuilder& setTDesc(const Tensor& tensor) { return settDesc(tensor); }
  OperationBuilder& setyDesc(const Tensor& tensor) {
    y_ = &tensor;
    return *this;
  }
  OperationBuilder& setYDesc(const Tensor& tensor) { return setyDesc(tensor); }
  OperationBuilder& setdxDesc(const Tensor& tensor) {
    dx_ = &tensor;
    return *this;
  }
  OperationBuilder& setDxDesc(const Tensor& tensor) {
    return setdxDesc(tensor);
  }
  OperationBuilder& setDXDesc(const Tensor& tensor) {
    return setdxDesc(tensor);
  }
  OperationBuilder& setdyDesc(const Tensor& tensor) {
    dy_ = &tensor;
    return *this;
  }
  OperationBuilder& setDyDesc(const Tensor& tensor) {
    return setdyDesc(tensor);
  }
  OperationBuilder& setDYDesc(const Tensor& tensor) {
    return setdyDesc(tensor);
  }
  OperationBuilder& setpwDesc(const PointWiseDesc& desc) {
    pw_ = &desc;
    return *this;
  }
  OperationBuilder& setPwDesc(const PointWiseDesc& desc) {
    return setpwDesc(desc);
  }
  OperationBuilder& setmatmulDesc(const MatMulDesc& desc) {
    matmul_ = &desc;
    return *this;
  }
  OperationBuilder& setMatmulDesc(const MatMulDesc& desc) {
    return setmatmulDesc(desc);
  }
  OperationBuilder& setconvDesc(const ConvDesc& desc) {
    conv_ = &desc;
    return *this;
  }
  OperationBuilder& setConvDesc(const ConvDesc& desc) {
    return setconvDesc(desc);
  }
  OperationBuilder& setreductionDesc(const ReductionDesc& desc) {
    reduction_ = &desc;
    return *this;
  }
  OperationBuilder& setReductionDesc(const ReductionDesc& desc) {
    return setreductionDesc(desc);
  }
  OperationBuilder& setresampleDesc(const ResampleDesc& desc) {
    resample_ = &desc;
    return *this;
  }
  OperationBuilder& setResampleDesc(const ResampleDesc& desc) {
    return setresampleDesc(desc);
  }
  OperationBuilder& setPoolingDesc(const ResampleDesc& desc) {
    return setresampleDesc(desc);
  }
  OperationBuilder& setaMatDesc(const Tensor& tensor) {
    a_mat_ = &tensor;
    return *this;
  }
  OperationBuilder& setAMatDesc(const Tensor& tensor) {
    return setaMatDesc(tensor);
  }
  OperationBuilder& setbMatDesc(const Tensor& tensor) {
    b_mat_ = &tensor;
    return *this;
  }
  OperationBuilder& setBMatDesc(const Tensor& tensor) {
    return setbMatDesc(tensor);
  }
  OperationBuilder& setcMatDesc(const Tensor& tensor) {
    c_mat_ = &tensor;
    return *this;
  }
  OperationBuilder& setmOverrideDesc(const Tensor& tensor) {
    m_override_ = &tensor;
    return *this;
  }
  OperationBuilder& setnOverrideDesc(const Tensor& tensor) {
    n_override_ = &tensor;
    return *this;
  }
  OperationBuilder& setkOverrideDesc(const Tensor& tensor) {
    k_override_ = &tensor;
    return *this;
  }
  OperationBuilder& setcDesc(const ConvDesc& desc) { return setconvDesc(desc); }
  OperationBuilder& setwDesc(const Tensor& tensor) {
    return setConvWDesc(tensor);
  }
  OperationBuilder& setdwDesc(const Tensor& tensor) {
    return setConvDwDesc(tensor);
  }
  OperationBuilder& setidxDesc(const Tensor& tensor) {
    return setResampleIndexDesc(tensor);
  }
  OperationBuilder& setcontainerDesc(const Tensor& tensor) {
    return setPagedCacheLoadContainerDesc(tensor);
  }
  OperationBuilder& setpageTableDesc(const Tensor& tensor) {
    return setPagedCacheLoadPageTableDesc(tensor);
  }
  OperationBuilder& setsequenceDesc(const Tensor& tensor) {
    return setPagedCacheLoadSequenceDesc(tensor);
  }
  OperationBuilder& setSeedDesc(const Tensor&) {
    throw mudnnException(
        MUDNN_STATUS_NOT_SUPPORTED,
        "muDNN RNG accepts a scalar seed but not a seed tensor descriptor");
  }
  OperationBuilder& setOffsetDesc(const Tensor& tensor) {
    return setRngOffsetDesc(tensor);
  }
  OperationBuilder& setCMatDesc(const Tensor& tensor) {
    return setcMatDesc(tensor);
  }
  OperationBuilder& setConvXDesc(const Tensor& tensor) {
    conv_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setConvWDesc(const Tensor& tensor) {
    conv_w_ = &tensor;
    return *this;
  }
  OperationBuilder& setConvYDesc(const Tensor& tensor) {
    conv_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setConvDxDesc(const Tensor& tensor) {
    conv_dx_ = &tensor;
    return *this;
  }
  OperationBuilder& setConvDXDesc(const Tensor& tensor) {
    return setConvDxDesc(tensor);
  }
  OperationBuilder& setConvDyDesc(const Tensor& tensor) {
    conv_dy_ = &tensor;
    return *this;
  }
  OperationBuilder& setConvDYDesc(const Tensor& tensor) {
    return setConvDyDesc(tensor);
  }
  OperationBuilder& setConvDwDesc(const Tensor& tensor) {
    conv_dw_ = &tensor;
    return *this;
  }
  OperationBuilder& setConvDWDesc(const Tensor& tensor) {
    return setConvDwDesc(tensor);
  }
  OperationBuilder& setReductionXDesc(const Tensor& tensor) {
    reduction_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setReductionYDesc(const Tensor& tensor) {
    reduction_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setGroupOffsetDesc(const Tensor& tensor) {
    group_offset_ = &tensor;
    return *this;
  }
  OperationBuilder& setGenStatsMode(mudnnGenStatsMode_t mode) {
    genstats_mode_ = mode;
    has_genstats_mode_ = true;
    return *this;
  }
  OperationBuilder& setGenStatsMathPrecision(mudnnDataType_t dtype) {
    genstats_math_prec_ = dtype;
    has_genstats_math_prec_ = true;
    return *this;
  }
  OperationBuilder& setGenStatsXDesc(const Tensor& tensor) {
    genstats_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setGenStatsSumDesc(const Tensor& tensor) {
    genstats_sum_ = &tensor;
    return *this;
  }
  OperationBuilder& setGenStatsSqSumDesc(const Tensor& tensor) {
    genstats_sqsum_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeStatsMode(mudnnBnFinalizeStatsMode_t mode) {
    bn_finalize_mode_ = mode;
    has_bn_finalize_mode_ = true;
    return *this;
  }
  OperationBuilder& setBnFinalizeMathPrecision(mudnnDataType_t dtype) {
    bn_finalize_math_prec_ = dtype;
    has_bn_finalize_math_prec_ = true;
    return *this;
  }
  OperationBuilder& setBnFinalizeYSumDesc(const Tensor& tensor) {
    bn_finalize_y_sum_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeYSqSumDesc(const Tensor& tensor) {
    bn_finalize_y_sq_sum_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeScaleDesc(const Tensor& tensor) {
    bn_finalize_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeBiasDesc(const Tensor& tensor) {
    bn_finalize_bias_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizePrevRunningMeanDesc(const Tensor& tensor) {
    bn_finalize_prev_running_mean_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizePrevRunningVarDesc(const Tensor& tensor) {
    bn_finalize_prev_running_var_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeUpdatedRunningMeanDesc(const Tensor& tensor) {
    bn_finalize_updated_running_mean_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeUpdatedRunningVarDesc(const Tensor& tensor) {
    bn_finalize_updated_running_var_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeSavedMeanDesc(const Tensor& tensor) {
    bn_finalize_saved_mean_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeSavedInvStdDesc(const Tensor& tensor) {
    bn_finalize_saved_inv_std_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeEqScaleDesc(const Tensor& tensor) {
    bn_finalize_eq_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeEqBiasDesc(const Tensor& tensor) {
    bn_finalize_eq_bias_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeAccumCountDesc(const Tensor& tensor) {
    bn_finalize_accum_count_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnFinalizeEpsilon(double epsilon) {
    bn_finalize_epsilon_ = epsilon;
    has_bn_finalize_epsilon_ = true;
    return *this;
  }
  OperationBuilder& setBnFinalizeExpAvgFactor(double factor) {
    bn_finalize_exp_avg_factor_ = factor;
    has_bn_finalize_exp_avg_factor_ = true;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsMathPrecision(mudnnDataType_t dtype) {
    bn_bwd_weights_math_prec_ = dtype;
    has_bn_bwd_weights_math_prec_ = true;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsMeanDesc(const Tensor& tensor) {
    bn_bwd_weights_mean_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsInvstdDesc(const Tensor& tensor) {
    bn_bwd_weights_invstd_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsInvStdDesc(const Tensor& tensor) {
    return setBnBwdWeightsInvstdDesc(tensor);
  }
  OperationBuilder& setBnBwdWeightsScaleDesc(const Tensor& tensor) {
    bn_bwd_weights_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsBnScaleDesc(const Tensor& tensor) {
    return setBnBwdWeightsScaleDesc(tensor);
  }
  OperationBuilder& setBnBwdWeightsXDesc(const Tensor& tensor) {
    bn_bwd_weights_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsDyDesc(const Tensor& tensor) {
    bn_bwd_weights_dy_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsDYDesc(const Tensor& tensor) {
    return setBnBwdWeightsDyDesc(tensor);
  }
  OperationBuilder& setBnBwdWeightsDBnScaleDesc(const Tensor& tensor) {
    bn_bwd_weights_dbn_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsDBNScaleDesc(const Tensor& tensor) {
    return setBnBwdWeightsDBnScaleDesc(tensor);
  }
  OperationBuilder& setBnBwdWeightsDBnBiasDesc(const Tensor& tensor) {
    bn_bwd_weights_dbn_bias_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsDBNBiasDesc(const Tensor& tensor) {
    return setBnBwdWeightsDBnBiasDesc(tensor);
  }
  OperationBuilder& setBnBwdWeightsEqDyScaleDesc(const Tensor& tensor) {
    bn_bwd_weights_eq_dy_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsEqDYScaleDesc(const Tensor& tensor) {
    return setBnBwdWeightsEqDyScaleDesc(tensor);
  }
  OperationBuilder& setBnBwdWeightsEqXScaleDesc(const Tensor& tensor) {
    bn_bwd_weights_eq_x_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setBnBwdWeightsEqBiasDesc(const Tensor& tensor) {
    bn_bwd_weights_eq_bias_ = &tensor;
    return *this;
  }
  OperationBuilder& setResampleXDesc(const Tensor& tensor) {
    resample_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setResampleYDesc(const Tensor& tensor) {
    resample_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setResampleDxDesc(const Tensor& tensor) {
    resample_dx_ = &tensor;
    return *this;
  }
  OperationBuilder& setResampleDXDesc(const Tensor& tensor) {
    return setResampleDxDesc(tensor);
  }
  OperationBuilder& setResampleDyDesc(const Tensor& tensor) {
    resample_dy_ = &tensor;
    return *this;
  }
  OperationBuilder& setResampleDYDesc(const Tensor& tensor) {
    return setResampleDyDesc(tensor);
  }
  OperationBuilder& setResampleIndexDesc(const Tensor& tensor) {
    resample_idx_ = &tensor;
    return *this;
  }
  OperationBuilder& setPoolingXDesc(const Tensor& tensor) {
    return setResampleXDesc(tensor);
  }
  OperationBuilder& setPoolingYDesc(const Tensor& tensor) {
    return setResampleYDesc(tensor);
  }
  OperationBuilder& setPoolingDxDesc(const Tensor& tensor) {
    return setResampleDxDesc(tensor);
  }
  OperationBuilder& setPoolingDyDesc(const Tensor& tensor) {
    return setResampleDyDesc(tensor);
  }
  OperationBuilder& setConcatInputDescs(const std::vector<Tensor>& tensors) {
    concat_inputs_.clear();
    for (const auto& tensor : tensors) {
      concat_inputs_.push_back(&tensor);
    }
    return *this;
  }
  OperationBuilder& setConcatOutputDesc(const Tensor& tensor) {
    concat_output_ = &tensor;
    return *this;
  }
  OperationBuilder& setConcatAxis(int64_t axis) {
    concat_axis_ = axis;
    has_concat_axis_ = true;
    return *this;
  }
  OperationBuilder& setConcatInplaceIndex(int64_t index) {
    concat_inplace_index_ = index;
    has_concat_inplace_index_ = true;
    return *this;
  }
  OperationBuilder& setReshapeXDesc(const Tensor& tensor) {
    reshape_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setReshapeYDesc(const Tensor& tensor) {
    reshape_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setReshapeMode(mudnnBackendReshapeMode_t mode) {
    reshape_mode_ = mode;
    has_reshape_mode_ = true;
    return *this;
  }
  OperationBuilder& setTransposeXDesc(const Tensor& tensor) {
    transpose_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setTransposeYDesc(const Tensor& tensor) {
    transpose_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setTransposePermutation(
      const std::vector<int64_t>& values) {
    transpose_permutation_ = values;
    return *this;
  }
  OperationBuilder& setSliceXDesc(const Tensor& tensor) {
    slice_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setSliceYDesc(const Tensor& tensor) {
    slice_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setSliceStartIndices(const std::vector<int64_t>& values) {
    slice_start_indices_ = values;
    return *this;
  }
  OperationBuilder& setSliceLimitIndices(const std::vector<int64_t>& values) {
    slice_limit_indices_ = values;
    return *this;
  }
  OperationBuilder& setSliceStrides(const std::vector<int64_t>& values) {
    slice_strides_ = values;
    return *this;
  }
  OperationBuilder& setRngDesc(const RngDesc& desc) {
    rng_ = &desc;
    return *this;
  }
  OperationBuilder& setRngYDesc(const Tensor& tensor) {
    rng_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setRngSeed(int64_t seed) {
    rng_seed_ = seed;
    has_rng_seed_ = true;
    return *this;
  }
  OperationBuilder& setRngOffsetDesc(const Tensor& tensor) {
    rng_offset_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormMode(mudnnBackendNormMode_t mode) {
    norm_mode_ = mode;
    has_norm_mode_ = true;
    return *this;
  }
  OperationBuilder& setNormalizationMode(NormMode_t mode) {
    return setNormMode(static_cast<mudnnBackendNormMode_t>(mode));
  }
  OperationBuilder& setNormalizationMode(mudnnBackendNormMode_t mode) {
    return setNormMode(mode);
  }
  OperationBuilder& setNormFwdPhase(mudnnBackendNormFwdPhase_t phase) {
    norm_fwd_phase_ = phase;
    has_norm_fwd_phase_ = true;
    return *this;
  }
  OperationBuilder& setBNFinalizeMode(mudnnBnFinalizeStatsMode_t mode) {
    return setBnFinalizeStatsMode(mode);
  }
  OperationBuilder& setAccumCountTensor(const Tensor& tensor) {
    return setBnFinalizeAccumCountDesc(tensor);
  }
  OperationBuilder& setEpsilonTensor(const Tensor& tensor) {
    if (type_ == MUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR) {
      return setNormFwdEpsilon(tensor);
    }
    if (type_ == MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR) {
      return setNormBwdEpsilon(tensor);
    }
    throw mudnnException(MUDNN_STATUS_NOT_SUPPORTED,
                         "epsilon tensor is unsupported for this operation");
  }
  OperationBuilder& setExpDecayFactorTensor(const Tensor& tensor) {
    return setNormFwdExpAvgFactor(tensor);
  }
  OperationBuilder& setPrevRunningMeanAndVar(const Tensor& mean,
                                              const Tensor& variance) {
    setNormFwdInputRunningMeanDesc(mean);
    return setNormFwdInputRunningVarDesc(variance);
  }
  OperationBuilder& setNextRunningMeanAndVar(const Tensor& mean,
                                              const Tensor& variance) {
    setNormFwdOutputRunningMeanDesc(mean);
    return setNormFwdOutputRunningVarDesc(variance);
  }
  OperationBuilder& setSavedMeanAndInvVar(const Tensor& mean,
                                           const Tensor& inv_variance) {
    if (type_ == MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR) {
      setNormBwdMeanDesc(mean);
      return setNormBwdInvVarianceDesc(inv_variance);
    }
    setNormFwdMeanDesc(mean);
    return setNormFwdInvVarianceDesc(inv_variance);
  }
  OperationBuilder& setSavedInvVar(const Tensor& inv_variance) {
    return type_ == MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR
               ? setNormBwdInvVarianceDesc(inv_variance)
               : setNormFwdInvVarianceDesc(inv_variance);
  }
  OperationBuilder& setScale(const Tensor& tensor) {
    return type_ == MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR
               ? setNormBwdScaleDesc(tensor)
               : setNormFwdScaleDesc(tensor);
  }
  OperationBuilder& setBias(const Tensor& tensor) {
    return setNormFwdBiasDesc(tensor);
  }
  OperationBuilder& setScaleAndBias(const Tensor& scale,
                                     const Tensor& bias) {
    setScale(scale);
    return setBias(bias);
  }
  OperationBuilder& setDScale(const Tensor& tensor) {
    return setNormBwdDScaleDesc(tensor);
  }
  OperationBuilder& setDBias(const Tensor& tensor) {
    return setNormBwdDBiasDesc(tensor);
  }
  OperationBuilder& setDScaleAndDBias(const Tensor& scale,
                                      const Tensor& bias) {
    setDScale(scale);
    return setDBias(bias);
  }
  OperationBuilder& setEqScaleAndBias(const Tensor& scale,
                                       const Tensor& bias) {
    setBnFinalizeEqScaleDesc(scale);
    return setBnFinalizeEqBiasDesc(bias);
  }
  OperationBuilder& setEqScalesAndBias(const Tensor& scale0,
                                        const Tensor& scale1,
                                        const Tensor& bias) {
    setBnBwdWeightsEqDyScaleDesc(scale0);
    setBnBwdWeightsEqXScaleDesc(scale1);
    return setBnBwdWeightsEqBiasDesc(bias);
  }
  OperationBuilder& setSumDesc(const Tensor& tensor) {
    return setGenStatsSumDesc(tensor);
  }
  OperationBuilder& setSqSumDesc(const Tensor& tensor) {
    return setGenStatsSqSumDesc(tensor);
  }
  OperationBuilder& setNormFwdXDesc(const Tensor& tensor) {
    norm_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdMeanDesc(const Tensor& tensor) {
    norm_mean_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdInvVarianceDesc(const Tensor& tensor) {
    norm_inv_variance_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdScaleDesc(const Tensor& tensor) {
    norm_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdBiasDesc(const Tensor& tensor) {
    norm_bias_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdInputRunningMeanDesc(const Tensor& tensor) {
    norm_input_running_mean_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdInputRunningVarDesc(const Tensor& tensor) {
    norm_input_running_var_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdOutputRunningMeanDesc(const Tensor& tensor) {
    norm_output_running_mean_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdOutputRunningVarDesc(const Tensor& tensor) {
    norm_output_running_var_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdYDesc(const Tensor& tensor) {
    norm_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormFwdEpsilon(double epsilon) {
    norm_epsilon_ = epsilon;
    has_norm_epsilon_ = true;
    norm_epsilon_desc_ = nullptr;
    return *this;
  }
  OperationBuilder& setNormFwdEpsilon(const Tensor& tensor) {
    norm_epsilon_desc_ = &tensor;
    has_norm_epsilon_ = false;
    return *this;
  }
  OperationBuilder& setNormFwdExpAvgFactor(double factor) {
    norm_exp_avg_factor_ = factor;
    has_norm_exp_avg_factor_ = true;
    norm_exp_avg_factor_desc_ = nullptr;
    return *this;
  }
  OperationBuilder& setNormFwdExpAvgFactor(const Tensor& tensor) {
    norm_exp_avg_factor_desc_ = &tensor;
    has_norm_exp_avg_factor_ = false;
    return *this;
  }
  OperationBuilder& setNormBwdMode(mudnnBackendNormMode_t mode) {
    norm_bwd_mode_ = mode;
    has_norm_bwd_mode_ = true;
    return *this;
  }
  OperationBuilder& setNormBwdXDesc(const Tensor& tensor) {
    norm_bwd_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormBwdMeanDesc(const Tensor& tensor) {
    norm_bwd_mean_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormBwdInvVarianceDesc(const Tensor& tensor) {
    norm_bwd_inv_variance_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormBwdDyDesc(const Tensor& tensor) {
    norm_bwd_dy_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormBwdScaleDesc(const Tensor& tensor) {
    norm_bwd_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormBwdDxDesc(const Tensor& tensor) {
    norm_bwd_dx_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormBwdDScaleDesc(const Tensor& tensor) {
    norm_bwd_dscale_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormBwdDBiasDesc(const Tensor& tensor) {
    norm_bwd_dbias_ = &tensor;
    return *this;
  }
  OperationBuilder& setNormBwdEpsilon(double epsilon) {
    norm_bwd_epsilon_ = epsilon;
    has_norm_bwd_epsilon_ = true;
    norm_bwd_epsilon_desc_ = nullptr;
    return *this;
  }
  OperationBuilder& setNormBwdEpsilon(const Tensor& tensor) {
    norm_bwd_epsilon_desc_ = &tensor;
    has_norm_bwd_epsilon_ = false;
    return *this;
  }
  OperationBuilder& setBlockScaleQuantizeXDesc(const Tensor& tensor) {
    block_scale_quantize_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setBlockScaleQuantizeYDesc(const Tensor& tensor) {
    block_scale_quantize_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setBlockScaleQuantizeScaleDesc(const Tensor& tensor) {
    block_scale_quantize_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setBlockScaleQuantizeMathPrecision(mudnnDataType_t dtype) {
    block_scale_quantize_math_prec_ = dtype;
    has_block_scale_quantize_math_prec_ = true;
    return *this;
  }
  OperationBuilder& setBlockScaleQuantizeBlockSize(int64_t block_size) {
    block_scale_quantize_block_size_ = block_size;
    has_block_scale_quantize_block_size_ = true;
    return *this;
  }
  OperationBuilder& setBlockScaleDequantizeXDesc(const Tensor& tensor) {
    block_scale_dequantize_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setBlockScaleDequantizeScaleDesc(const Tensor& tensor) {
    block_scale_dequantize_scale_ = &tensor;
    return *this;
  }
  OperationBuilder& setBlockScaleDequantizeYDesc(const Tensor& tensor) {
    block_scale_dequantize_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setBlockScaleDequantizeMathPrecision(
      mudnnDataType_t dtype) {
    block_scale_dequantize_math_prec_ = dtype;
    has_block_scale_dequantize_math_prec_ = true;
    return *this;
  }
  OperationBuilder& setBlockScaleDequantizeBlockSize(int64_t block_size) {
    block_scale_dequantize_block_size_ = block_size;
    has_block_scale_dequantize_block_size_ = true;
    return *this;
  }
  OperationBuilder& setBlockScaleDequantizeNegativeScale(bool value) {
    block_scale_dequantize_negative_scale_ = value;
    has_block_scale_dequantize_negative_scale_ = true;
    return *this;
  }
  OperationBuilder& setBlockScaleDequantizeIsNegativeScale(bool value) {
    return setBlockScaleDequantizeNegativeScale(value);
  }
  OperationBuilder& setMoeGroupedMatmulMode(mudnnMoeGroupedMatmulMode_t mode) {
    moe_grouped_matmul_mode_ = mode;
    has_moe_grouped_matmul_mode_ = true;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulMathPrecision(mudnnDataType_t dtype) {
    moe_grouped_matmul_math_prec_ = dtype;
    has_moe_grouped_matmul_math_prec_ = true;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulTokenDesc(const Tensor& tensor) {
    moe_grouped_matmul_token_ = &tensor;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulWeightDesc(const Tensor& tensor) {
    moe_grouped_matmul_weight_ = &tensor;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulFirstTokenOffsetDesc(
      const Tensor& tensor) {
    moe_grouped_matmul_first_token_offset_ = &tensor;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulOutputDesc(const Tensor& tensor) {
    moe_grouped_matmul_output_ = &tensor;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulTokenIndexDesc(const Tensor& tensor) {
    moe_grouped_matmul_token_index_ = &tensor;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulTokenKsDesc(const Tensor& tensor) {
    moe_grouped_matmul_token_ks_ = &tensor;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulTopK(int32_t top_k) {
    moe_grouped_matmul_top_k_ = top_k;
    has_moe_grouped_matmul_top_k_ = true;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulBwdMathPrecision(mudnnDataType_t dtype) {
    moe_grouped_matmul_bwd_math_prec_ = dtype;
    has_moe_grouped_matmul_bwd_math_prec_ = true;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulBwdTokenDesc(const Tensor& tensor) {
    moe_grouped_matmul_bwd_token_ = &tensor;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulBwdDWeightDesc(const Tensor& tensor) {
    moe_grouped_matmul_bwd_dweight_ = &tensor;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulBwdFirstTokenOffsetDesc(
      const Tensor& tensor) {
    moe_grouped_matmul_bwd_first_token_offset_ = &tensor;
    return *this;
  }
  OperationBuilder& setMoeGroupedMatmulBwdDOutputDesc(const Tensor& tensor) {
    moe_grouped_matmul_bwd_doutput_ = &tensor;
    return *this;
  }
  OperationBuilder& setPagedCacheLoadContainerDesc(const Tensor& tensor) {
    paged_cache_container_ = &tensor;
    return *this;
  }
  OperationBuilder& setPagedCacheLoadYDesc(const Tensor& tensor) {
    paged_cache_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setPagedCacheLoadSequenceDesc(const Tensor& tensor) {
    paged_cache_sequence_ = &tensor;
    return *this;
  }
  OperationBuilder& setPagedCacheLoadPageTableDesc(const Tensor& tensor) {
    paged_cache_page_table_ = &tensor;
    return *this;
  }
  OperationBuilder& setSignalMode(mudnnSignalMode_t mode) {
    signal_mode_ = mode;
    has_signal_mode_ = true;
    return *this;
  }
  OperationBuilder& setSignalValue(int64_t value) {
    signal_value_ = value;
    has_signal_value_ = true;
    return *this;
  }
  OperationBuilder& setSignalFlagDesc(const Tensor& tensor) {
    signal_flag_ = &tensor;
    return *this;
  }
  OperationBuilder& setSignalXDesc(const Tensor& tensor) {
    signal_x_ = &tensor;
    return *this;
  }
  OperationBuilder& setSignalYDesc(const Tensor& tensor) {
    signal_y_ = &tensor;
    return *this;
  }
  OperationBuilder& setqDesc(const Tensor& tensor) {
    q_ = &tensor;
    return *this;
  }
  OperationBuilder& setQDesc(const Tensor& tensor) { return setqDesc(tensor); }
  OperationBuilder& setkDesc(const Tensor& tensor) {
    k_ = &tensor;
    return *this;
  }
  OperationBuilder& setKDesc(const Tensor& tensor) { return setkDesc(tensor); }
  OperationBuilder& setvDesc(const Tensor& tensor) {
    v_ = &tensor;
    return *this;
  }
  OperationBuilder& setVDesc(const Tensor& tensor) { return setvDesc(tensor); }
  OperationBuilder& setoDesc(const Tensor& tensor) {
    o_ = &tensor;
    return *this;
  }
  OperationBuilder& setODesc(const Tensor& tensor) { return setoDesc(tensor); }
  OperationBuilder& setattnProbsDesc(const Tensor& tensor) {
    attn_probs_ = &tensor;
    return *this;
  }
  OperationBuilder& setAttnProbsDesc(const Tensor& tensor) {
    return setattnProbsDesc(tensor);
  }
  OperationBuilder& setdropoutMaskDesc(const Tensor& tensor) {
    dropout_mask_ = &tensor;
    return *this;
  }
  OperationBuilder& setDropoutMaskDesc(const Tensor& tensor) {
    return setdropoutMaskDesc(tensor);
  }
  OperationBuilder& setSdpaStatsDesc(const Tensor& tensor) {
    sdpa_stats_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaFwdStatsDesc(const Tensor& tensor) {
    return setSdpaStatsDesc(tensor);
  }
  OperationBuilder& setmaskDesc(const Tensor& tensor) {
    sdpa_mask_ = &tensor;
    return *this;
  }
  OperationBuilder& setMaskDesc(const Tensor& tensor) {
    return setmaskDesc(tensor);
  }
  OperationBuilder& setbiasDesc(const Tensor& tensor) {
    sdpa_bias_ = &tensor;
    return *this;
  }
  OperationBuilder& setBiasDesc(const Tensor& tensor) {
    return setbiasDesc(tensor);
  }
  OperationBuilder& setAlpha(double alpha) {
    alpha1_ = alpha;
    has_alpha1_ = true;
    return *this;
  }
  OperationBuilder& setAlpha2(double alpha) {
    alpha2_ = alpha;
    has_alpha2_ = true;
    return *this;
  }
  OperationBuilder& setBeta(double beta) {
    alpha2_ = beta;
    has_alpha2_ = true;
    return *this;
  }
  OperationBuilder& setSdpaScale(double scale) {
    sdpa_scale_ = scale;
    has_sdpa_scale_ = true;
    return *this;
  }
  OperationBuilder& setSdpaScaleDesc(const Tensor& tensor) {
    sdpa_scale_desc_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaInference(bool is_inference) {
    sdpa_inference_ = is_inference;
    has_sdpa_inference_ = true;
    return *this;
  }
  OperationBuilder& setSdpaGenerateStats(bool generate_stats) {
    sdpa_inference_ = !generate_stats;
    has_sdpa_inference_ = true;
    return *this;
  }
  OperationBuilder& setSdpaCausalMask(bool causal_mask) {
    sdpa_causal_mask_ = causal_mask;
    has_sdpa_causal_mask_ = true;
    return *this;
  }
  OperationBuilder& setSdpaUsePaddingMask(bool use_padding_mask) {
    sdpa_use_padding_mask_ = use_padding_mask;
    has_sdpa_use_padding_mask_ = true;
    return *this;
  }
  OperationBuilder& setSdpaUseCausalMaskBottomRight(bool use_bottom_right) {
    if (use_bottom_right) {
      throw std::invalid_argument(
          "mudnn_frontend does not support bottom-right SDPA causal masking");
    }
    return *this;
  }
  OperationBuilder& setSdpaDiagonalAlignment(DiagonalAlignment alignment) {
    if (alignment != DiagonalAlignment::TOP_LEFT) {
      throw std::invalid_argument(
          "mudnn_frontend only supports top-left causal diagonal masking");
    }
    return *this;
  }
  OperationBuilder& setSdpaDiagonalBandRightBound(int64_t bound) {
    if (bound != 0) {
      throw std::invalid_argument(
          "mudnn_frontend only supports diagonal_band_right_bound=0");
    }
    return setSdpaCausalMask(true);
  }
  OperationBuilder& setSdpaDiagonalBandLeftBound(int64_t) {
    throw std::invalid_argument(
        "mudnn_frontend does not support general diagonal bands");
  }
  OperationBuilder& setSdpaUseAlibiMask(bool use_alibi_mask) {
    if (use_alibi_mask) {
      throw std::invalid_argument("mudnn_frontend does not support ALiBi SDPA");
    }
    return *this;
  }
  OperationBuilder& setSdpaDropout(double dropout) {
    if (dropout != 0.0) {
      throw std::invalid_argument(
          "mudnn_frontend does not support SDPA dropout");
    }
    return *this;
  }
  OperationBuilder& setSdpaSlidingWindowLength(int64_t) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA sliding_window_length");
  }
  OperationBuilder& setSdpaSeqLenQDesc(const Tensor& tensor) {
    sdpa_seq_len_q_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaSeqLenKvDesc(const Tensor& tensor) {
    sdpa_seq_len_kv_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaSeqLenKVDesc(const Tensor& tensor) {
    return setSdpaSeqLenKvDesc(tensor);
  }
  OperationBuilder& setSdpaMaxTotalSeqLenQ(int64_t) {
    throw std::invalid_argument(
        "mudnn_frontend does not support ragged SDPA max sequence lengths");
  }
  OperationBuilder& setSdpaMaxTotalSeqLenKv(int64_t) {
    throw std::invalid_argument(
        "mudnn_frontend does not support ragged SDPA max sequence lengths");
  }
  OperationBuilder& setSdpaMaxTotalSeqLenKV(int64_t max_total_seq_len) {
    return setSdpaMaxTotalSeqLenKv(max_total_seq_len);
  }
  template <typename ScoreMod>
  OperationBuilder& setSdpaScoreMod(ScoreMod&&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA score modifiers");
  }
  OperationBuilder& setSdpaScoreSumExpDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA score_sum_exp descriptors");
  }
  OperationBuilder& setSdpaScoreMaxDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA score_max descriptors");
  }
  OperationBuilder& setSdpaSinkTokenDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA sink-token descriptors");
  }
  OperationBuilder& setSdpaDescaleQDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_q");
  }
  OperationBuilder& setSdpaDescaleKDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_k");
  }
  OperationBuilder& setSdpaDescaleVDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_v");
  }
  OperationBuilder& setSdpaDescaleSDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_s");
  }
  OperationBuilder& setSdpaScaleSDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 scale_s");
  }
  OperationBuilder& setSdpaScaleODesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 scale_o");
  }
  OperationBuilder& setSdpaBwdDoDesc(const Tensor& tensor) {
    sdpa_bwd_d_o_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaBwdDODesc(const Tensor& tensor) {
    return setSdpaBwdDoDesc(tensor);
  }
  OperationBuilder& setSdpaBwdStatsDesc(const Tensor& tensor) {
    sdpa_stats_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaBwdDqDesc(const Tensor& tensor) {
    sdpa_bwd_d_q_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaBwdDQDesc(const Tensor& tensor) {
    return setSdpaBwdDqDesc(tensor);
  }
  OperationBuilder& setSdpaBwdDkDesc(const Tensor& tensor) {
    sdpa_bwd_d_k_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaBwdDKDesc(const Tensor& tensor) {
    return setSdpaBwdDkDesc(tensor);
  }
  OperationBuilder& setSdpaBwdDvDesc(const Tensor& tensor) {
    sdpa_bwd_d_v_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaBwdDVDesc(const Tensor& tensor) {
    return setSdpaBwdDvDesc(tensor);
  }
  OperationBuilder& setSdpaBwdDBiasDesc(const Tensor& tensor) {
    sdpa_bwd_d_bias_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaBwdScale(double scale) {
    sdpa_scale_ = scale;
    has_sdpa_scale_ = true;
    return *this;
  }
  OperationBuilder& setSdpaBwdScaleDesc(const Tensor& tensor) {
    sdpa_scale_desc_ = &tensor;
    return *this;
  }
  OperationBuilder& setSdpaBwdCausalMask(bool causal_mask) {
    sdpa_causal_mask_ = causal_mask;
    has_sdpa_causal_mask_ = true;
    return *this;
  }
  OperationBuilder& setSdpaBwdUsePaddingMask(bool use_padding_mask) {
    sdpa_use_padding_mask_ = use_padding_mask;
    has_sdpa_use_padding_mask_ = true;
    return *this;
  }
  OperationBuilder& setSdpaBwdUseCausalMaskBottomRight(bool use_bottom_right) {
    if (use_bottom_right) {
      throw std::invalid_argument(
          "mudnn_frontend does not support bottom-right SDPA causal masking");
    }
    return *this;
  }
  OperationBuilder& setSdpaBwdDiagonalAlignment(DiagonalAlignment alignment) {
    return setSdpaDiagonalAlignment(alignment);
  }
  OperationBuilder& setSdpaBwdDiagonalBandRightBound(int64_t bound) {
    if (bound != 0) {
      throw std::invalid_argument(
          "mudnn_frontend only supports diagonal_band_right_bound=0");
    }
    return setSdpaBwdCausalMask(true);
  }
  OperationBuilder& setSdpaBwdDiagonalBandLeftBound(int64_t) {
    throw std::invalid_argument(
        "mudnn_frontend does not support general diagonal bands");
  }
  OperationBuilder& setSdpaBwdDeterministic(bool deterministic) {
    sdpa_bwd_deterministic_ = deterministic;
    has_sdpa_bwd_deterministic_ = true;
    return *this;
  }
  OperationBuilder& setSdpaBwdUseDeterministicAlgorithm(bool deterministic) {
    return setSdpaBwdDeterministic(deterministic);
  }
  OperationBuilder& setSdpaBwdUseAlibiMask(bool use_alibi_mask) {
    return setSdpaUseAlibiMask(use_alibi_mask);
  }
  OperationBuilder& setSdpaBwdDropout(double dropout) {
    return setSdpaDropout(dropout);
  }
  OperationBuilder& setSdpaBwdSlidingWindowLength(int64_t length) {
    return setSdpaSlidingWindowLength(length);
  }
  OperationBuilder& setSdpaBwdSeqLenQDesc(const Tensor& tensor) {
    return setSdpaSeqLenQDesc(tensor);
  }
  OperationBuilder& setSdpaBwdSeqLenKvDesc(const Tensor& tensor) {
    return setSdpaSeqLenKvDesc(tensor);
  }
  OperationBuilder& setSdpaBwdSeqLenKVDesc(const Tensor& tensor) {
    return setSdpaBwdSeqLenKvDesc(tensor);
  }
  OperationBuilder& setSdpaBwdMaxTotalSeqLenQ(int64_t max_total_seq_len) {
    return setSdpaMaxTotalSeqLenQ(max_total_seq_len);
  }
  OperationBuilder& setSdpaBwdMaxTotalSeqLenKv(int64_t max_total_seq_len) {
    return setSdpaMaxTotalSeqLenKv(max_total_seq_len);
  }
  OperationBuilder& setSdpaBwdMaxTotalSeqLenKV(int64_t max_total_seq_len) {
    return setSdpaBwdMaxTotalSeqLenKv(max_total_seq_len);
  }
  template <typename ScoreMod>
  OperationBuilder& setSdpaBwdScoreMod(ScoreMod&&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA backward score modifiers");
  }
  template <typename ScoreMod>
  OperationBuilder& setSdpaBwdScoreModBprop(ScoreMod&&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA backward score modifiers");
  }
  OperationBuilder& setSdpaBwdSinkTokenDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA sink-token descriptors");
  }
  OperationBuilder& setSdpaBwdDSinkTokenDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA dSink-token descriptors");
  }
  OperationBuilder& setSdpaBwdDescaleQDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_q");
  }
  OperationBuilder& setSdpaBwdDescaleKDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_k");
  }
  OperationBuilder& setSdpaBwdDescaleVDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_v");
  }
  OperationBuilder& setSdpaBwdDescaleODesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_o");
  }
  OperationBuilder& setSdpaBwdDescaleDoDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_dO");
  }
  OperationBuilder& setSdpaBwdDescaleDODesc(const Tensor& tensor) {
    return setSdpaBwdDescaleDoDesc(tensor);
  }
  OperationBuilder& setSdpaBwdDescaleSDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_s");
  }
  OperationBuilder& setSdpaBwdDescaleDpDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 descale_dP");
  }
  OperationBuilder& setSdpaBwdDescaleDPDesc(const Tensor& tensor) {
    return setSdpaBwdDescaleDpDesc(tensor);
  }
  OperationBuilder& setSdpaBwdScaleSDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 scale_s");
  }
  OperationBuilder& setSdpaBwdScaleDqDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 scale_dQ");
  }
  OperationBuilder& setSdpaBwdScaleDQDesc(const Tensor& tensor) {
    return setSdpaBwdScaleDqDesc(tensor);
  }
  OperationBuilder& setSdpaBwdScaleDkDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 scale_dK");
  }
  OperationBuilder& setSdpaBwdScaleDKDesc(const Tensor& tensor) {
    return setSdpaBwdScaleDkDesc(tensor);
  }
  OperationBuilder& setSdpaBwdScaleDvDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 scale_dV");
  }
  OperationBuilder& setSdpaBwdScaleDVDesc(const Tensor& tensor) {
    return setSdpaBwdScaleDvDesc(tensor);
  }
  OperationBuilder& setSdpaBwdScaleDpDesc(const Tensor&) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA FP8 scale_dP");
  }
  OperationBuilder& setSdpaBwdScaleDPDesc(const Tensor& tensor) {
    return setSdpaBwdScaleDpDesc(tensor);
  }

  Operation build() const {
    Operation op(type_);
    if (type_ == MUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR) {
      if (pw_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, *pw_);
      }
      if (x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_POINTWISE_XDESC, *x_);
      }
      if (b_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_POINTWISE_BDESC, *b_);
      }
      if (t_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_POINTWISE_TDESC, *t_);
      }
      if (y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_POINTWISE_YDESC, *y_);
      }
      if (dx_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_POINTWISE_DXDESC,
                                    *dx_);
      }
      if (dy_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_POINTWISE_DYDESC,
                                    *dy_);
      }
      if (has_alpha1_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_POINTWISE_ALPHA1,
                         MUDNN_TYPE_DOUBLE, 1, &alpha1_);
      }
      if (has_alpha2_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_POINTWISE_ALPHA2,
                         MUDNN_TYPE_DOUBLE, 1, &alpha2_);
      }
    } else if (type_ ==
               MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
      if (conv_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, *conv_);
      }
      if (conv_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                    *conv_x_);
      }
      if (conv_w_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                    *conv_w_);
      }
      if (conv_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                    *conv_y_);
      }
      if (has_alpha1_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha1_);
      }
      if (has_alpha2_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha2_);
      }
    } else if (type_ ==
               MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
      if (conv_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC, *conv_);
      }
      if (conv_w_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                    *conv_w_);
      }
      if (conv_dx_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, *conv_dx_);
      }
      if (conv_dy_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, *conv_dy_);
      }
      if (has_alpha1_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha1_);
      }
      if (has_alpha2_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha2_);
      }
    } else if (type_ ==
               MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
      if (conv_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, *conv_);
      }
      if (conv_dw_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, *conv_dw_);
      }
      if (conv_x_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, *conv_x_);
      }
      if (conv_dy_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, *conv_dy_);
      }
      if (has_alpha1_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha1_);
      }
      if (has_alpha2_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha2_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR) {
      if (matmul_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_MATMUL_DESC, *matmul_);
      }
      if (a_mat_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_MATMUL_ADESC, *a_mat_);
      }
      if (b_mat_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_MATMUL_BDESC, *b_mat_);
      }
      if (c_mat_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_MATMUL_CDESC, *c_mat_);
      }
      if (m_override_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC, *m_override_);
      }
      if (n_override_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC, *n_override_);
      }
      if (k_override_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC, *k_override_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR) {
      if (reduction_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_REDUCTION_DESC,
                                    *reduction_);
      }
      if (reduction_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                    *reduction_x_);
      }
      if (reduction_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_REDUCTION_YDESC,
                                    *reduction_y_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR) {
      if (has_genstats_mode_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_GENSTATS_MODE,
                         MUDNN_TYPE_GENSTATS_MODE, 1, &genstats_mode_);
      }
      if (has_genstats_math_prec_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC,
                         MUDNN_TYPE_DATA_TYPE, 1, &genstats_math_prec_);
      }
      if (genstats_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_GENSTATS_XDESC,
                                    *genstats_x_);
      }
      if (genstats_sum_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_GENSTATS_SUMDESC,
                                    *genstats_sum_);
      }
      if (genstats_sqsum_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC,
                                    *genstats_sqsum_);
      }
    } else if (type_ ==
               MUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR) {
      if (has_bn_finalize_mode_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE,
                         MUDNN_TYPE_BN_FINALIZE_STATS_MODE, 1,
                         &bn_finalize_mode_);
      }
      if (has_bn_finalize_math_prec_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC,
                         MUDNN_TYPE_DATA_TYPE, 1, &bn_finalize_math_prec_);
      }
      if (bn_finalize_y_sum_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC,
                                    *bn_finalize_y_sum_);
      }
      if (bn_finalize_y_sq_sum_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC,
            *bn_finalize_y_sq_sum_);
      }
      if (bn_finalize_scale_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC,
                                    *bn_finalize_scale_);
      }
      if (bn_finalize_bias_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC,
                                    *bn_finalize_bias_);
      }
      if (bn_finalize_prev_running_mean_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC,
            *bn_finalize_prev_running_mean_);
      }
      if (bn_finalize_prev_running_var_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC,
            *bn_finalize_prev_running_var_);
      }
      if (bn_finalize_updated_running_mean_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC,
            *bn_finalize_updated_running_mean_);
      }
      if (bn_finalize_updated_running_var_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC,
            *bn_finalize_updated_running_var_);
      }
      if (bn_finalize_saved_mean_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC,
            *bn_finalize_saved_mean_);
      }
      if (bn_finalize_saved_inv_std_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC,
            *bn_finalize_saved_inv_std_);
      }
      if (bn_finalize_eq_scale_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC,
            *bn_finalize_eq_scale_);
      }
      if (bn_finalize_eq_bias_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC,
            *bn_finalize_eq_bias_);
      }
      if (bn_finalize_accum_count_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC,
            *bn_finalize_accum_count_);
      }
      if (has_bn_finalize_epsilon_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC,
                         MUDNN_TYPE_DOUBLE, 1, &bn_finalize_epsilon_);
      }
      if (has_bn_finalize_exp_avg_factor_) {
        op.set_attribute(
            MUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC,
            MUDNN_TYPE_DOUBLE, 1, &bn_finalize_exp_avg_factor_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR) {
      if (has_bn_bwd_weights_math_prec_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC,
                         MUDNN_TYPE_DATA_TYPE, 1, &bn_bwd_weights_math_prec_);
      }
      if (bn_bwd_weights_mean_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC,
            *bn_bwd_weights_mean_);
      }
      if (bn_bwd_weights_invstd_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC,
            *bn_bwd_weights_invstd_);
      }
      if (bn_bwd_weights_scale_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC,
            *bn_bwd_weights_scale_);
      }
      if (bn_bwd_weights_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC,
                                    *bn_bwd_weights_x_);
      }
      if (bn_bwd_weights_dy_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC,
                                    *bn_bwd_weights_dy_);
      }
      if (bn_bwd_weights_dbn_scale_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC,
            *bn_bwd_weights_dbn_scale_);
      }
      if (bn_bwd_weights_dbn_bias_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC,
            *bn_bwd_weights_dbn_bias_);
      }
      if (bn_bwd_weights_eq_dy_scale_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC,
            *bn_bwd_weights_eq_dy_scale_);
      }
      if (bn_bwd_weights_eq_x_scale_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC,
            *bn_bwd_weights_eq_x_scale_);
      }
      if (bn_bwd_weights_eq_bias_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS,
                                    *bn_bwd_weights_eq_bias_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR) {
      if (resample_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_FWD_DESC,
                                    *resample_);
      }
      if (resample_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                    *resample_x_);
      }
      if (resample_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
                                    *resample_y_);
      }
      if (resample_idx_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                                    *resample_idx_);
      }
      if (has_alpha1_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_FWD_ALPHA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha1_);
      }
      if (has_alpha2_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_FWD_BETA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha2_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR) {
      if (resample_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DESC,
                                    *resample_);
      }
      if (resample_dx_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DXDESC,
                                    *resample_dx_);
      }
      if (resample_dy_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_BWD_DYDESC,
                                    *resample_dy_);
      }
      if (resample_idx_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_BWD_IDXDESC,
                                    *resample_idx_);
      }
      if (resample_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC,
                                    *resample_x_);
      }
      if (resample_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC,
                                    *resample_y_);
      }
      if (has_alpha1_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha1_);
      }
      if (has_alpha2_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_RESAMPLE_BWD_BETA,
                         MUDNN_TYPE_DOUBLE, 1, &alpha2_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR) {
      if (has_concat_axis_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_CONCAT_AXIS, MUDNN_TYPE_INT64, 1,
                         &concat_axis_);
      }
      if (!concat_inputs_.empty()) {
        std::vector<mudnnBackendDescriptor_t> inputs;
        inputs.reserve(concat_inputs_.size());
        for (const auto* input : concat_inputs_) {
          inputs.push_back(input->get_backend_descriptor());
        }
        op.set_attribute(MUDNN_ATTR_OPERATION_CONCAT_INPUT_DESCS,
                         MUDNN_TYPE_BACKEND_DESCRIPTOR,
                         static_cast<int64_t>(inputs.size()), inputs.data());
      }
      if (concat_output_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_CONCAT_OUTPUT_DESC,
                                    *concat_output_);
      }
      if (has_concat_inplace_index_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_CONCAT_INPLACE_INDEX,
                         MUDNN_TYPE_INT64, 1, &concat_inplace_index_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR) {
      if (reshape_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESHAPE_XDESC,
                                    *reshape_x_);
      }
      if (reshape_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RESHAPE_YDESC,
                                    *reshape_y_);
      }
      if (has_reshape_mode_ && reshape_mode_ != MUDNN_RESHAPE_VIEW_ONLY) {
        op.set_attribute(MUDNN_ATTR_OPERATION_RESHAPE_MODE,
                         MUDNN_TYPE_RESHAPE_MODE, 1, &reshape_mode_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR) {
      if (transpose_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_TRANSPOSE_XDESC,
                                    *transpose_x_);
      }
      if (transpose_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_TRANSPOSE_YDESC,
                                    *transpose_y_);
      }
      if (!transpose_permutation_.empty()) {
        op.set_attribute(MUDNN_ATTR_OPERATION_TRANSPOSE_PERMUTATION,
                         MUDNN_TYPE_INT64,
                         static_cast<int64_t>(transpose_permutation_.size()),
                         transpose_permutation_.data());
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR) {
      if (slice_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SLICE_XDESC,
                                    *slice_x_);
      }
      if (slice_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SLICE_YDESC,
                                    *slice_y_);
      }
      if (!slice_start_indices_.empty()) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SLICE_START_INDICES,
                         MUDNN_TYPE_INT64,
                         static_cast<int64_t>(slice_start_indices_.size()),
                         slice_start_indices_.data());
      }
      if (!slice_limit_indices_.empty()) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SLICE_LIMIT_INDICES,
                         MUDNN_TYPE_INT64,
                         static_cast<int64_t>(slice_limit_indices_.size()),
                         slice_limit_indices_.data());
      }
      if (!slice_strides_.empty()) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SLICE_STRIDES, MUDNN_TYPE_INT64,
                         static_cast<int64_t>(slice_strides_.size()),
                         slice_strides_.data());
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR) {
      if (rng_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RNG_YDESC, *rng_y_);
      }
      if (has_rng_seed_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_RNG_SEED, MUDNN_TYPE_INT64, 1,
                         &rng_seed_);
      }
      if (rng_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RNG_DESC, *rng_);
      }
      if (rng_offset_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_RNG_OFFSET_DESC,
                                    *rng_offset_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR) {
      if (has_norm_mode_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_MODE,
                         MUDNN_TYPE_NORM_MODE, 1, &norm_mode_);
      }
      if (has_norm_fwd_phase_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_PHASE,
                         MUDNN_TYPE_NORM_FWD_PHASE, 1, &norm_fwd_phase_);
      }
      if (norm_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_XDESC,
                                    *norm_x_);
      }
      if (norm_mean_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_MEAN_DESC,
                                    *norm_mean_);
      }
      if (norm_inv_variance_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_NORM_FWD_INV_VARIANCE_DESC,
            *norm_inv_variance_);
      }
      if (norm_scale_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_SCALE_DESC,
                                    *norm_scale_);
      }
      if (norm_bias_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_BIAS_DESC,
                                    *norm_bias_);
      }
      if (norm_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_YDESC,
                                    *norm_y_);
      }
      if (norm_input_running_mean_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_MEAN_DESC,
            *norm_input_running_mean_);
      }
      if (norm_input_running_var_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_NORM_FWD_INPUT_RUNNING_VAR_DESC,
            *norm_input_running_var_);
      }
      if (norm_output_running_mean_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_MEAN_DESC,
            *norm_output_running_mean_);
      }
      if (norm_output_running_var_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_NORM_FWD_OUTPUT_RUNNING_VAR_DESC,
            *norm_output_running_var_);
      }
      if (norm_epsilon_desc_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                                    *norm_epsilon_desc_);
      } else if (has_norm_epsilon_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_EPSILON_DESC,
                         MUDNN_TYPE_DOUBLE, 1, &norm_epsilon_);
      }
      if (has_norm_exp_avg_factor_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
                         MUDNN_TYPE_DOUBLE, 1, &norm_exp_avg_factor_);
      } else if (norm_exp_avg_factor_desc_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_NORM_FWD_EXP_AVG_FACTOR_DESC,
            *norm_exp_avg_factor_desc_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR) {
      if (has_norm_bwd_mode_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_MODE,
                         MUDNN_TYPE_NORM_MODE, 1, &norm_bwd_mode_);
      }
      if (norm_bwd_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_XDESC,
                                    *norm_bwd_x_);
      }
      if (norm_bwd_mean_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_MEAN_DESC,
                                    *norm_bwd_mean_);
      }
      if (norm_bwd_inv_variance_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_NORM_BWD_INV_VARIANCE_DESC,
            *norm_bwd_inv_variance_);
      }
      if (norm_bwd_dy_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_DYDESC,
                                    *norm_bwd_dy_);
      }
      if (norm_bwd_scale_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_SCALE_DESC,
                                    *norm_bwd_scale_);
      }
      if (norm_bwd_dx_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_DXDESC,
                                    *norm_bwd_dx_);
      }
      if (norm_bwd_dscale_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_DSCALE_DESC,
                                    *norm_bwd_dscale_);
      }
      if (norm_bwd_dbias_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_DBIAS_DESC,
                                    *norm_bwd_dbias_);
      }
      if (norm_bwd_epsilon_desc_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC,
                                    *norm_bwd_epsilon_desc_);
      } else if (has_norm_bwd_epsilon_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_NORM_BWD_EPSILON_DESC,
                         MUDNN_TYPE_DOUBLE, 1, &norm_bwd_epsilon_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR) {
      if (paged_cache_container_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_CONTAINER_DESC,
            *paged_cache_container_);
      }
      if (paged_cache_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_YDESC,
                                    *paged_cache_y_);
      }
      if (paged_cache_sequence_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_SEQUENCE_DESC,
            *paged_cache_sequence_);
      }
      if (paged_cache_page_table_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_PAGED_CACHE_LOAD_PAGE_TABLE_DESC,
            *paged_cache_page_table_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR) {
      if (has_signal_mode_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SIGNAL_MODE,
                         MUDNN_TYPE_SIGNAL_MODE, 1, &signal_mode_);
      }
      if (has_signal_value_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SIGNAL_VALUE, MUDNN_TYPE_INT64, 1,
                         &signal_value_);
      }
      if (signal_flag_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SIGNAL_FLAGDESC,
                                    *signal_flag_);
      }
      if (signal_x_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SIGNAL_XDESC,
                                    *signal_x_);
      }
      if (signal_y_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SIGNAL_YDESC,
                                    *signal_y_);
      }
    } else if (type_ ==
               MUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR) {
      if (block_scale_quantize_x_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
            *block_scale_quantize_x_);
      }
      if (block_scale_quantize_y_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC,
            *block_scale_quantize_y_);
      }
      if (block_scale_quantize_scale_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC,
            *block_scale_quantize_scale_);
      }
      if (has_block_scale_quantize_math_prec_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                         MUDNN_TYPE_DATA_TYPE, 1,
                         &block_scale_quantize_math_prec_);
      }
      if (has_block_scale_quantize_block_size_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
                         MUDNN_TYPE_INT64, 1,
                         &block_scale_quantize_block_size_);
      }
    } else if (type_ ==
               MUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR) {
      if (block_scale_dequantize_x_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
            *block_scale_dequantize_x_);
      }
      if (block_scale_dequantize_scale_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
            *block_scale_dequantize_scale_);
      }
      if (block_scale_dequantize_y_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
            *block_scale_dequantize_y_);
      }
      if (has_block_scale_dequantize_math_prec_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                         MUDNN_TYPE_DATA_TYPE, 1,
                         &block_scale_dequantize_math_prec_);
      }
      if (has_block_scale_dequantize_block_size_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                         MUDNN_TYPE_INT64, 1,
                         &block_scale_dequantize_block_size_);
      }
      if (has_block_scale_dequantize_negative_scale_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                         MUDNN_TYPE_BOOLEAN, 1,
                         &block_scale_dequantize_negative_scale_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR) {
      if (has_moe_grouped_matmul_mode_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                         MUDNN_TYPE_MOE_GROUPED_MATMUL_MODE, 1,
                         &moe_grouped_matmul_mode_);
      }
      if (has_moe_grouped_matmul_math_prec_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                         MUDNN_TYPE_DATA_TYPE, 1,
                         &moe_grouped_matmul_math_prec_);
      }
      if (moe_grouped_matmul_token_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
            *moe_grouped_matmul_token_);
      }
      if (moe_grouped_matmul_weight_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
            *moe_grouped_matmul_weight_);
      }
      if (moe_grouped_matmul_first_token_offset_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
            *moe_grouped_matmul_first_token_offset_);
      }
      if (moe_grouped_matmul_output_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
            *moe_grouped_matmul_output_);
      }
      if (moe_grouped_matmul_token_index_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
            *moe_grouped_matmul_token_index_);
      }
      if (moe_grouped_matmul_token_ks_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
            *moe_grouped_matmul_token_ks_);
      }
      if (has_moe_grouped_matmul_top_k_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K,
                         MUDNN_TYPE_INT32, 1, &moe_grouped_matmul_top_k_);
      }
    } else if (type_ ==
               MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR) {
      if (has_moe_grouped_matmul_bwd_math_prec_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                         MUDNN_TYPE_DATA_TYPE, 1,
                         &moe_grouped_matmul_bwd_math_prec_);
      }
      if (moe_grouped_matmul_bwd_token_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
            *moe_grouped_matmul_bwd_token_);
      }
      if (moe_grouped_matmul_bwd_dweight_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
            *moe_grouped_matmul_bwd_dweight_);
      }
      if (moe_grouped_matmul_bwd_first_token_offset_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
            *moe_grouped_matmul_bwd_first_token_offset_);
      }
      if (moe_grouped_matmul_bwd_doutput_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
            *moe_grouped_matmul_bwd_doutput_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR) {
      if (q_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, *q_);
      }
      if (k_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, *k_);
      }
      if (v_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, *v_);
      }
      if (o_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, *o_);
      }
      if (sdpa_stats_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC,
                                    *sdpa_stats_);
      }
      if (sdpa_scale_desc_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
                                    *sdpa_scale_desc_);
      } else if (has_sdpa_scale_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
                         MUDNN_TYPE_DOUBLE, 1, &sdpa_scale_);
      }
      if (sdpa_mask_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC, *sdpa_mask_);
      }
      if (sdpa_bias_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_BIAS_DESC,
                                    *sdpa_bias_);
      }
      if (sdpa_seq_len_q_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
                                    *sdpa_seq_len_q_);
      }
      if (sdpa_seq_len_kv_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC, *sdpa_seq_len_kv_);
      }
      if (has_sdpa_use_padding_mask_ && sdpa_use_padding_mask_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_USE_PADDING_MASK,
                         MUDNN_TYPE_BOOLEAN, 1, &sdpa_use_padding_mask_);
      }
      if (has_sdpa_causal_mask_ && sdpa_causal_mask_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SDPA_FWD_CAUSAL_MASK,
                         MUDNN_TYPE_BOOLEAN, 1, &sdpa_causal_mask_);
      }
    } else if (type_ == MUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR) {
      if (q_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_QDESC, *q_);
      }
      if (k_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_KDESC, *k_);
      }
      if (v_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_VDESC, *v_);
      }
      if (o_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_ODESC, *o_);
      }
      if (sdpa_bwd_d_o_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_DODDESC,
                                    *sdpa_bwd_d_o_);
      }
      if (sdpa_stats_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_STATSDESC,
                                    *sdpa_stats_);
      }
      if (sdpa_scale_desc_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC,
                                    *sdpa_scale_desc_);
      } else if (has_sdpa_scale_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_SCALEDESC,
                         MUDNN_TYPE_DOUBLE, 1, &sdpa_scale_);
      }
      if (sdpa_bwd_d_q_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_DQDESC,
                                    *sdpa_bwd_d_q_);
      }
      if (sdpa_bwd_d_k_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_DKDESC,
                                    *sdpa_bwd_d_k_);
      }
      if (sdpa_bwd_d_v_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_DVDESC,
                                    *sdpa_bwd_d_v_);
      }
      if (sdpa_seq_len_q_ != nullptr) {
        op.set_descriptor_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_QDESC,
                                    *sdpa_seq_len_q_);
      }
      if (sdpa_seq_len_kv_ != nullptr) {
        op.set_descriptor_attribute(
            MUDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KVDESC, *sdpa_seq_len_kv_);
      }
      if (has_sdpa_causal_mask_ && sdpa_causal_mask_) {
        op.set_attribute(MUDNN_ATTR_OPERATION_SDPA_BWD_CAUSAL_MASK,
                         MUDNN_TYPE_BOOLEAN, 1, &sdpa_causal_mask_);
      }
    }
    op.finalize();
    return op;
  }

 private:
  mudnnBackendDescriptorType_t type_;
  const Tensor* x_ = nullptr;
  const Tensor* b_ = nullptr;
  const Tensor* t_ = nullptr;
  const Tensor* y_ = nullptr;
  const Tensor* dx_ = nullptr;
  const Tensor* dy_ = nullptr;
  const Tensor* a_mat_ = nullptr;
  const Tensor* b_mat_ = nullptr;
  const Tensor* c_mat_ = nullptr;
  const Tensor* m_override_ = nullptr;
  const Tensor* n_override_ = nullptr;
  const Tensor* k_override_ = nullptr;
  const Tensor* conv_x_ = nullptr;
  const Tensor* conv_w_ = nullptr;
  const Tensor* conv_y_ = nullptr;
  const Tensor* conv_dx_ = nullptr;
  const Tensor* conv_dy_ = nullptr;
  const Tensor* conv_dw_ = nullptr;
  const Tensor* reduction_x_ = nullptr;
  const Tensor* reduction_y_ = nullptr;
  const Tensor* group_offset_ = nullptr;
  const Tensor* resample_x_ = nullptr;
  const Tensor* resample_y_ = nullptr;
  const Tensor* resample_dx_ = nullptr;
  const Tensor* resample_dy_ = nullptr;
  const Tensor* resample_idx_ = nullptr;
  std::vector<const Tensor*> concat_inputs_;
  const Tensor* concat_output_ = nullptr;
  const Tensor* reshape_x_ = nullptr;
  const Tensor* reshape_y_ = nullptr;
  mudnnBackendReshapeMode_t reshape_mode_ = MUDNN_RESHAPE_VIEW_ONLY;
  bool has_reshape_mode_ = false;
  const Tensor* transpose_x_ = nullptr;
  const Tensor* transpose_y_ = nullptr;
  const Tensor* rng_y_ = nullptr;
  const Tensor* rng_offset_ = nullptr;
  const Tensor* genstats_x_ = nullptr;
  const Tensor* genstats_sum_ = nullptr;
  const Tensor* genstats_sqsum_ = nullptr;
  const Tensor* bn_finalize_y_sum_ = nullptr;
  const Tensor* bn_finalize_y_sq_sum_ = nullptr;
  const Tensor* bn_finalize_scale_ = nullptr;
  const Tensor* bn_finalize_bias_ = nullptr;
  const Tensor* bn_finalize_prev_running_mean_ = nullptr;
  const Tensor* bn_finalize_prev_running_var_ = nullptr;
  const Tensor* bn_finalize_updated_running_mean_ = nullptr;
  const Tensor* bn_finalize_updated_running_var_ = nullptr;
  const Tensor* bn_finalize_saved_mean_ = nullptr;
  const Tensor* bn_finalize_saved_inv_std_ = nullptr;
  const Tensor* bn_finalize_eq_scale_ = nullptr;
  const Tensor* bn_finalize_eq_bias_ = nullptr;
  const Tensor* bn_finalize_accum_count_ = nullptr;
  const Tensor* bn_bwd_weights_mean_ = nullptr;
  const Tensor* bn_bwd_weights_invstd_ = nullptr;
  const Tensor* bn_bwd_weights_scale_ = nullptr;
  const Tensor* bn_bwd_weights_x_ = nullptr;
  const Tensor* bn_bwd_weights_dy_ = nullptr;
  const Tensor* bn_bwd_weights_dbn_scale_ = nullptr;
  const Tensor* bn_bwd_weights_dbn_bias_ = nullptr;
  const Tensor* bn_bwd_weights_eq_dy_scale_ = nullptr;
  const Tensor* bn_bwd_weights_eq_x_scale_ = nullptr;
  const Tensor* bn_bwd_weights_eq_bias_ = nullptr;
  const Tensor* norm_x_ = nullptr;
  const Tensor* norm_mean_ = nullptr;
  const Tensor* norm_inv_variance_ = nullptr;
  const Tensor* norm_scale_ = nullptr;
  const Tensor* norm_bias_ = nullptr;
  const Tensor* norm_y_ = nullptr;
  const Tensor* norm_input_running_mean_ = nullptr;
  const Tensor* norm_input_running_var_ = nullptr;
  const Tensor* norm_output_running_mean_ = nullptr;
  const Tensor* norm_output_running_var_ = nullptr;
  const Tensor* norm_bwd_x_ = nullptr;
  const Tensor* norm_bwd_mean_ = nullptr;
  const Tensor* norm_bwd_inv_variance_ = nullptr;
  const Tensor* norm_bwd_dy_ = nullptr;
  const Tensor* norm_bwd_scale_ = nullptr;
  const Tensor* norm_bwd_dx_ = nullptr;
  const Tensor* norm_bwd_dscale_ = nullptr;
  const Tensor* norm_bwd_dbias_ = nullptr;
  const Tensor* block_scale_quantize_x_ = nullptr;
  const Tensor* block_scale_quantize_y_ = nullptr;
  const Tensor* block_scale_quantize_scale_ = nullptr;
  const Tensor* block_scale_dequantize_x_ = nullptr;
  const Tensor* block_scale_dequantize_scale_ = nullptr;
  const Tensor* block_scale_dequantize_y_ = nullptr;
  const Tensor* moe_grouped_matmul_token_ = nullptr;
  const Tensor* moe_grouped_matmul_weight_ = nullptr;
  const Tensor* moe_grouped_matmul_first_token_offset_ = nullptr;
  const Tensor* moe_grouped_matmul_output_ = nullptr;
  const Tensor* moe_grouped_matmul_token_index_ = nullptr;
  const Tensor* moe_grouped_matmul_token_ks_ = nullptr;
  const Tensor* moe_grouped_matmul_bwd_token_ = nullptr;
  const Tensor* moe_grouped_matmul_bwd_dweight_ = nullptr;
  const Tensor* moe_grouped_matmul_bwd_first_token_offset_ = nullptr;
  const Tensor* moe_grouped_matmul_bwd_doutput_ = nullptr;
  const Tensor* paged_cache_container_ = nullptr;
  const Tensor* paged_cache_y_ = nullptr;
  const Tensor* paged_cache_sequence_ = nullptr;
  const Tensor* paged_cache_page_table_ = nullptr;
  const Tensor* signal_flag_ = nullptr;
  const Tensor* signal_x_ = nullptr;
  const Tensor* signal_y_ = nullptr;
  const Tensor* slice_x_ = nullptr;
  const Tensor* slice_y_ = nullptr;
  const Tensor* q_ = nullptr;
  const Tensor* k_ = nullptr;
  const Tensor* v_ = nullptr;
  const Tensor* o_ = nullptr;
  const Tensor* attn_probs_ = nullptr;
  const Tensor* dropout_mask_ = nullptr;
  const Tensor* sdpa_stats_ = nullptr;
  const Tensor* sdpa_scale_desc_ = nullptr;
  const Tensor* sdpa_mask_ = nullptr;
  const Tensor* sdpa_bias_ = nullptr;
  const Tensor* sdpa_seq_len_q_ = nullptr;
  const Tensor* sdpa_seq_len_kv_ = nullptr;
  const Tensor* sdpa_bwd_d_o_ = nullptr;
  const Tensor* sdpa_bwd_d_q_ = nullptr;
  const Tensor* sdpa_bwd_d_k_ = nullptr;
  const Tensor* sdpa_bwd_d_v_ = nullptr;
  const Tensor* sdpa_bwd_d_bias_ = nullptr;
  const PointWiseDesc* pw_ = nullptr;
  const MatMulDesc* matmul_ = nullptr;
  const ConvDesc* conv_ = nullptr;
  const ReductionDesc* reduction_ = nullptr;
  const ResampleDesc* resample_ = nullptr;
  const RngDesc* rng_ = nullptr;
  const Tensor* norm_epsilon_desc_ = nullptr;
  const Tensor* norm_exp_avg_factor_desc_ = nullptr;
  const Tensor* norm_bwd_epsilon_desc_ = nullptr;
  double alpha1_ = 1.0;
  double alpha2_ = 1.0;
  int64_t concat_axis_ = 0;
  int64_t concat_inplace_index_ = -1;
  std::vector<int64_t> transpose_permutation_;
  std::vector<int64_t> slice_start_indices_;
  std::vector<int64_t> slice_limit_indices_;
  std::vector<int64_t> slice_strides_;
  int64_t rng_seed_ = 0;
  mudnnBackendNormMode_t norm_mode_ = MUDNN_BATCH_NORM;
  mudnnBackendNormFwdPhase_t norm_fwd_phase_ = MUDNN_NORM_FWD_INFERENCE;
  mudnnBackendNormMode_t norm_bwd_mode_ = MUDNN_BATCH_NORM;
  double norm_epsilon_ = 1.0e-5;
  double norm_exp_avg_factor_ = 0.1;
  double norm_bwd_epsilon_ = 1.0e-5;
  mudnnDataType_t block_scale_quantize_math_prec_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t block_scale_dequantize_math_prec_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t moe_grouped_matmul_math_prec_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t moe_grouped_matmul_bwd_math_prec_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t genstats_math_prec_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t bn_finalize_math_prec_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t bn_bwd_weights_math_prec_ = MUDNN_DATA_FLOAT;
  int64_t block_scale_quantize_block_size_ = 0;
  int64_t block_scale_dequantize_block_size_ = 0;
  bool block_scale_dequantize_negative_scale_ = false;
  mudnnMoeGroupedMatmulMode_t moe_grouped_matmul_mode_ =
      MUDNN_MOE_GROUPED_MATMUL_MODE_NONE;
  int32_t moe_grouped_matmul_top_k_ = 0;
  mudnnGenStatsMode_t genstats_mode_ = MUDNN_GENSTATS_SUM_SQSUM;
  mudnnBnFinalizeStatsMode_t bn_finalize_mode_ =
      MUDNN_BN_FINALIZE_STATISTICS_INFERENCE;
  mudnnSignalMode_t signal_mode_ = MUDNN_SIGNAL_SET;
  int64_t signal_value_ = 0;
  double bn_finalize_epsilon_ = 1.0e-5;
  double bn_finalize_exp_avg_factor_ = 0.1;
  double sdpa_scale_ = 0.0;
  bool sdpa_inference_ = true;
  bool sdpa_causal_mask_ = false;
  bool sdpa_use_padding_mask_ = false;
  bool sdpa_bwd_deterministic_ = false;
  bool has_alpha1_ = false;
  bool has_alpha2_ = false;
  bool has_concat_axis_ = false;
  bool has_concat_inplace_index_ = false;
  bool has_rng_seed_ = false;
  bool has_norm_mode_ = false;
  bool has_norm_fwd_phase_ = false;
  bool has_norm_epsilon_ = false;
  bool has_norm_exp_avg_factor_ = false;
  bool has_norm_bwd_mode_ = false;
  bool has_norm_bwd_epsilon_ = false;
  bool has_block_scale_quantize_math_prec_ = false;
  bool has_block_scale_quantize_block_size_ = false;
  bool has_block_scale_dequantize_math_prec_ = false;
  bool has_block_scale_dequantize_block_size_ = false;
  bool has_block_scale_dequantize_negative_scale_ = false;
  bool has_moe_grouped_matmul_mode_ = false;
  bool has_moe_grouped_matmul_math_prec_ = false;
  bool has_moe_grouped_matmul_top_k_ = false;
  bool has_moe_grouped_matmul_bwd_math_prec_ = false;
  bool has_genstats_mode_ = false;
  bool has_genstats_math_prec_ = false;
  bool has_bn_finalize_mode_ = false;
  bool has_bn_finalize_math_prec_ = false;
  bool has_bn_finalize_epsilon_ = false;
  bool has_bn_finalize_exp_avg_factor_ = false;
  bool has_bn_bwd_weights_math_prec_ = false;
  bool has_signal_mode_ = false;
  bool has_signal_value_ = false;
  bool has_sdpa_scale_ = false;
  bool has_sdpa_inference_ = false;
  bool has_sdpa_causal_mask_ = false;
  bool has_sdpa_use_padding_mask_ = false;
  bool has_sdpa_bwd_deterministic_ = false;
};

class OperationGraph : public BackendDescriptor {
 public:
  OperationGraph() = default;
  int64_t getEngineCount() const {
    int64_t value = 0;
    get_attribute(MUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT,
                  MUDNN_TYPE_INT64, 1, &value);
    return value;
  }
  uint64_t getOpCount() const { return operations_.size(); }
  const std::vector<Operation>& getOps() const { return operations_; }
  bool setFeatureVector(const std::vector<int64_t>& feature_vector) {
    feature_vectors_.push_back(feature_vector);
    return true;
  }

 private:
  friend class OperationGraphBuilder;
  std::vector<Operation> operations_;
  std::vector<std::vector<int64_t>> feature_vectors_;
  using BackendDescriptor::BackendDescriptor;
};

class OperationGraphBuilder {
 public:
  OperationGraphBuilder& setHandle(mudnnHandle_t handle) {
    handle_ = handle;
    return *this;
  }
  OperationGraphBuilder& setOperationGraph(int64_t count,
                                           const Operation* const* ops) {
    ops_.clear();
    operations_.clear();
    for (int64_t i = 0; i < count; ++i) {
      ops_.push_back(ops[i]->get_backend_descriptor());
      operations_.push_back(*ops[i]);
    }
    return *this;
  }
  OperationGraphBuilder& setOperationGraph(const std::vector<Operation>& ops) {
    operations_ = ops;
    ops_.clear();
    for (const auto& op : ops) {
      ops_.push_back(op.get_backend_descriptor());
    }
    return *this;
  }
  OperationGraphBuilder& addOperation(const Operation& operation) {
    operations_.push_back(operation);
    ops_.push_back(operation.get_backend_descriptor());
    return *this;
  }
  OperationGraphBuilder& setIsDynamicShapeEnabled(bool enabled) {
    dynamic_shape_enabled_ = enabled;
    return *this;
  }
  OperationGraphBuilder& setIsOverrideShapeEnabled(bool enabled) {
    override_shape_enabled_ = enabled;
    return *this;
  }

  OperationGraph build() const {
    OperationGraph graph(MUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    graph.operations_ = operations_;
    graph.set_attribute(MUDNN_ATTR_OPERATIONGRAPH_HANDLE, MUDNN_TYPE_HANDLE, 1,
                        &handle_);
    graph.set_attribute(MUDNN_ATTR_OPERATIONGRAPH_OPS,
                        MUDNN_TYPE_BACKEND_DESCRIPTOR,
                        static_cast<int64_t>(ops_.size()), ops_.data());
    if (dynamic_shape_enabled_) {
      graph.set_attribute(MUDNN_ATTR_OPERATIONGRAPH_IS_DYNAMIC_SHAPE_ENABLED,
                          MUDNN_TYPE_BOOLEAN, 1, &dynamic_shape_enabled_);
    }
    if (override_shape_enabled_) {
      graph.set_attribute(MUDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED,
                          MUDNN_TYPE_BOOLEAN, 1, &override_shape_enabled_);
    }
    graph.finalize();
    return graph;
  }

 private:
  mudnnHandle_t handle_ = nullptr;
  std::vector<mudnnBackendDescriptor_t> ops_;
  std::vector<Operation> operations_;
  bool dynamic_shape_enabled_ = false;
  bool override_shape_enabled_ = false;
};

class EngineConfig : public BackendDescriptor {
 public:
  EngineConfig() = default;
  static EngineConfig Borrowed(mudnnBackendDescriptor_t desc) {
    return EngineConfig(desc, Ownership::kBorrowed);
  }

  int64_t getWorkspaceSize() const {
    int64_t value = 0;
    get_attribute(MUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, MUDNN_TYPE_INT64, 1,
                  &value);
    return value;
  }

  int64_t getSharedMemoryUsed() const {
    int64_t value = 0;
    get_attribute(MUDNN_ATTR_ENGINECFG_SHARED_MEMORY_USED, MUDNN_TYPE_INT64, 1,
                  &value);
    return value;
  }

  mudnnBackendDescriptor_t getEngineDescriptor() const {
    mudnnBackendDescriptor_t value = nullptr;
    get_attribute(MUDNN_ATTR_ENGINECFG_ENGINE, MUDNN_TYPE_BACKEND_DESCRIPTOR,
                  1, &value);
    return value;
  }
  std::vector<Knob>& getFinalizedKnobs();

 private:
  friend class EngineConfigBuilder;
  friend class EngineHeuristics;
  mutable std::vector<Knob> knobs_;
  EngineConfig(mudnnBackendDescriptor_t desc, Ownership ownership)
      : BackendDescriptor(desc, ownership) {}
  using BackendDescriptor::BackendDescriptor;
};

class Engine : public BackendDescriptor {
 public:
  Engine() = default;
  static Engine Borrowed(mudnnBackendDescriptor_t desc) {
    return Engine(desc, Ownership::kBorrowed);
  }

  int64_t getGlobalIndex() const {
    int64_t value = 0;
    get_attribute(MUDNN_ATTR_ENGINE_GLOBAL_INDEX, MUDNN_TYPE_INT64, 1, &value);
    return value;
  }

  std::vector<Knob> getKnobs() const {
    const auto count = get_attribute<mudnnBackendDescriptor_t>(
        MUDNN_ATTR_ENGINE_KNOB_INFO, MUDNN_TYPE_BACKEND_DESCRIPTOR, 0,
        nullptr);
    std::vector<mudnnBackendDescriptor_t> descriptors(
        static_cast<size_t>(count), nullptr);
    if (count > 0) {
      get_attribute(MUDNN_ATTR_ENGINE_KNOB_INFO, MUDNN_TYPE_BACKEND_DESCRIPTOR,
                    count, descriptors.data());
    }
    std::vector<Knob> result;
    result.reserve(static_cast<size_t>(count));
    for (auto descriptor : descriptors) {
      Knob knob;
      int64_t returned = 0;
      mudnnBackendKnobType_t type = MUDNN_KNOB_TYPE_COUNTS;
      throw_if_error(mudnnBackendGetAttribute(
                         descriptor, MUDNN_ATTR_KNOB_INFO_TYPE,
                         MUDNN_TYPE_KNOB_TYPE, 1, &returned, &type),
                     "query engine knob type failed");
      knob.type = static_cast<KnobType_t>(type);
      throw_if_error(mudnnBackendGetAttribute(
                         descriptor, MUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE,
                         MUDNN_TYPE_INT64, 1, &returned, &knob.minValue),
                     "query engine knob minimum failed");
      throw_if_error(mudnnBackendGetAttribute(
                         descriptor, MUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE,
                         MUDNN_TYPE_INT64, 1, &returned, &knob.maxValue),
                     "query engine knob maximum failed");
      throw_if_error(mudnnBackendGetAttribute(
                         descriptor, MUDNN_ATTR_KNOB_INFO_STRIDE,
                         MUDNN_TYPE_INT64, 1, &returned, &knob.stride),
                     "query engine knob stride failed");
      result.push_back(knob);
    }
    return result;
  }
  std::vector<Knob>& getSupportedKnobs() {
    knobs_ = getKnobs();
    return knobs_;
  }
  const std::vector<Knob>& getFinalizedKnobs() const {
    if (knobs_.empty()) knobs_ = getKnobs();
    return knobs_;
  }

  std::vector<NumericalNote_t> getNumericalNotes() const {
    return get_notes<NumericalNote_t, mudnnBackendNumericalNote_t>(
        MUDNN_ATTR_ENGINE_NUMERICAL_NOTE, MUDNN_TYPE_NUMERICAL_NOTE);
  }

  std::vector<BehaviorNote_t> getBehaviorNotes() const {
    return get_notes<BehaviorNote_t, mudnnBackendBehaviorNote_t>(
        MUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, MUDNN_TYPE_BEHAVIOR_NOTE);
  }

 private:
  friend class EngineBuilder;
  mutable std::vector<Knob> knobs_;
  Engine(mudnnBackendDescriptor_t desc, Ownership ownership)
      : BackendDescriptor(desc, ownership) {}

  template <typename FrontendNote, typename BackendNote>
  std::vector<FrontendNote> get_notes(mudnnBackendAttributeName_t name,
                                      mudnnBackendAttributeType_t type) const {
    const auto count = get_attribute<BackendNote>(name, type, 0, nullptr);
    std::vector<BackendNote> notes(static_cast<size_t>(count));
    if (count > 0) get_attribute(name, type, count, notes.data());
    std::vector<FrontendNote> result;
    result.reserve(notes.size());
    for (auto note : notes) result.push_back(static_cast<FrontendNote>(note));
    return result;
  }
  using BackendDescriptor::BackendDescriptor;
};

inline std::vector<Knob>& EngineConfig::getFinalizedKnobs() {
  if (knobs_.empty() && getEngineDescriptor() != nullptr) {
    knobs_ = Engine::Borrowed(getEngineDescriptor()).getKnobs();
  }
  return knobs_;
}

class EngineBuilder {
 public:
  EngineBuilder& setOperationGraph(const OperationGraph& graph) {
    graph_ = &graph;
    return *this;
  }
  EngineBuilder& setGlobalEngineIdx(int64_t index) {
    index_ = index;
    return *this;
  }

  Engine build() const {
    Engine engine(MUDNN_BACKEND_ENGINE_DESCRIPTOR);
    if (graph_ != nullptr) {
      engine.set_descriptor_attribute(MUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                                      *graph_);
    }
    engine.set_attribute(MUDNN_ATTR_ENGINE_GLOBAL_INDEX, MUDNN_TYPE_INT64, 1,
                         &index_);
    engine.finalize();
    return engine;
  }

 private:
  const OperationGraph* graph_ = nullptr;
  int64_t index_ = 0;
};

class EngineConfigBuilder {
 public:
  EngineConfigBuilder& setEngine(const Engine& engine) {
    engine_ = &engine;
    return *this;
  }

  EngineConfig build() const {
    EngineConfig config(MUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    if (engine_ != nullptr) {
      config.set_descriptor_attribute(MUDNN_ATTR_ENGINECFG_ENGINE, *engine_);
    }
    config.finalize();
    return config;
  }

 private:
  const Engine* engine_ = nullptr;
};

class EngineHeuristics : public BackendDescriptor {
 public:
  EngineHeuristics() = default;

  int64_t getEngineConfigCount() const {
    int64_t returned = 0;
    throw_if_error(mudnnBackendGetAttribute(
                       get_backend_descriptor(), MUDNN_ATTR_ENGINEHEUR_RESULTS,
                       MUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &returned, nullptr),
                   "mudnnBackendGetAttribute heuristics count failed");
    return returned;
  }

  std::vector<EngineConfig> getEngineConfig(int64_t max_count = 1) const {
    std::vector<mudnnBackendDescriptor_t> raw(max_count, nullptr);
    int64_t returned = 0;
    throw_if_error(mudnnBackendGetAttribute(get_backend_descriptor(),
                                            MUDNN_ATTR_ENGINEHEUR_RESULTS,
                                            MUDNN_TYPE_BACKEND_DESCRIPTOR,
                                            max_count, &returned, raw.data()),
                   "mudnnBackendGetAttribute heuristics results failed");
    std::vector<EngineConfig> configs;
    configs.reserve(static_cast<size_t>(returned));
    for (int64_t i = 0; i < returned; ++i) {
      configs.push_back(EngineConfig::Borrowed(raw[i]));
    }
    return configs;
  }

 private:
  friend class EngineHeuristicsBuilder;
  using BackendDescriptor::BackendDescriptor;
};

class EngineHeuristicsBuilder {
 public:
  EngineHeuristicsBuilder& setOperationGraph(const OperationGraph& graph) {
    graph_ = &graph;
    return *this;
  }
  EngineHeuristicsBuilder& setHeurMode(mudnnBackendHeurMode_t mode) {
    mode_ = mode;
    return *this;
  }

  EngineHeuristics build() const {
    EngineHeuristics heur(MUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    if (graph_ != nullptr) {
      heur.set_descriptor_attribute(MUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                    *graph_);
    }
    heur.set_attribute(MUDNN_ATTR_ENGINEHEUR_MODE, MUDNN_TYPE_HEUR_MODE, 1,
                       &mode_);
    heur.finalize();
    return heur;
  }

 private:
  const OperationGraph* graph_ = nullptr;
  mudnnBackendHeurMode_t mode_ = MUDNN_HEUR_MODE_INSTANT;
};

using ManagedOpaqueDescriptor = EngineConfig;
using EngineConfigList = std::vector<ManagedOpaqueDescriptor>;

class EngineFallbackList {
 public:
  std::vector<ManagedOpaqueDescriptor>& getFallbackList() {
    return engine_configs_;
  }

 private:
  friend class EngineFallbackListBuilder;
  std::vector<ManagedOpaqueDescriptor> engine_configs_;
};

class EngineFallbackListBuilder {
 public:
  EngineFallbackListBuilder& setOperationGraph(OperationGraph& graph) {
    graph_ = &graph;
    return *this;
  }

  EngineFallbackListBuilder& setOperation(mudnnBackendDescriptorType_t mode) {
    operation_ = mode;
    return *this;
  }

  EngineFallbackList build() const {
    if (graph_ == nullptr) {
      throw std::invalid_argument(
          "EngineFallbackListBuilder requires an operation graph");
    }
    (void)operation_;
    auto heuristics = EngineHeuristicsBuilder()
                          .setOperationGraph(*graph_)
                          .setHeurMode(MUDNN_HEUR_MODE_FALLBACK)
                          .build();
    EngineFallbackList result;
    result.engine_configs_ =
        heuristics.getEngineConfig(heuristics.getEngineConfigCount());
    return result;
  }

 private:
  OperationGraph* graph_ = nullptr;
  mudnnBackendDescriptorType_t operation_ =
      MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
};

class ExecutionPlan : public BackendDescriptor {
 public:
  ExecutionPlan() = default;

  void setExecutionTime(float time) { execution_time_ms_ = time; }
  float getExecutionTime() const { return execution_time_ms_; }
  const std::vector<mudnnBackendNumericalNote_t>& getAllNumericNotes() const {
    return numeric_notes_;
  }
  const std::vector<mudnnBackendBehaviorNote_t>& getAllBehaviorNotes() const {
    return behavior_notes_;
  }
  const std::array<mudnnBackendNumericalNote_t,
                   MUDNN_NUMERICAL_NOTE_TYPE_COUNT>&
  getNumericNotes() const {
    return numeric_note_array_;
  }
  const std::array<mudnnBackendBehaviorNote_t, MUDNN_BEHAVIOR_NOTE_TYPE_COUNT>&
  getBehaviorNotes() const {
    return behavior_note_array_;
  }
  std::string getJsonRepresentation() const {
    const auto count = get_attribute<char>(
        MUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION, MUDNN_TYPE_CHAR, 0,
        nullptr);
    std::string result(static_cast<size_t>(count), '\0');
    if (count > 0) {
      get_attribute(MUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                    MUDNN_TYPE_CHAR, count, result.data());
    }
    return result;
  }

  int64_t getWorkspaceSize() const {
    int64_t count = 0;
    int64_t size = 0;
    throw_if_error(
        mudnnBackendGetAttribute(get_backend_descriptor(),
                                 MUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                 MUDNN_TYPE_INT64, 1, &count, &size),
        "mudnnBackendGetAttribute workspace failed");
    return size;
  }

 private:
  friend class ExecutionPlanBuilder;
  float execution_time_ms_ = 0.0f;
  std::vector<mudnnBackendNumericalNote_t> numeric_notes_;
  std::vector<mudnnBackendBehaviorNote_t> behavior_notes_;
  std::array<mudnnBackendNumericalNote_t, MUDNN_NUMERICAL_NOTE_TYPE_COUNT>
      numeric_note_array_{};
  std::array<mudnnBackendBehaviorNote_t, MUDNN_BEHAVIOR_NOTE_TYPE_COUNT>
      behavior_note_array_{};
  using BackendDescriptor::BackendDescriptor;
};

class ExecutionPlanBuilder {
 public:
  ExecutionPlanBuilder& setHandle(mudnnHandle_t handle) {
    handle_ = handle;
    return *this;
  }
  ExecutionPlanBuilder& setEngineConfig(const EngineConfig& config) {
    config_ = &config;
    return *this;
  }
  ExecutionPlanBuilder& setEngineConfig(const EngineConfig& config,
                                        const std::string&) {
    return setEngineConfig(config);
  }
  ExecutionPlanBuilder& setKernelCache(std::shared_ptr<KernelCache> cache) {
    kernel_cache_ = std::move(cache);
    return *this;
  }

  ExecutionPlan build() const {
    if (kernel_cache_) {
      throw mudnnException(
          MUDNN_STATUS_NOT_SUPPORTED,
          "muDNN execution plans do not accept a cudnn-frontend kernel cache");
    }
    ExecutionPlan plan(MUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    plan.set_attribute(MUDNN_ATTR_EXECUTION_PLAN_HANDLE, MUDNN_TYPE_HANDLE, 1,
                       &handle_);
    if (config_ != nullptr) {
      plan.set_descriptor_attribute(MUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                    *config_);
      auto engine = Engine::Borrowed(config_->getEngineDescriptor());
      for (auto note : engine.getNumericalNotes()) {
        plan.numeric_notes_.push_back(
            static_cast<mudnnBackendNumericalNote_t>(note));
      }
      for (auto note : engine.getBehaviorNotes()) {
        plan.behavior_notes_.push_back(
            static_cast<mudnnBackendBehaviorNote_t>(note));
      }
      for (size_t i = 0;
           i < plan.numeric_notes_.size() && i < plan.numeric_note_array_.size();
           ++i) {
        plan.numeric_note_array_[i] = plan.numeric_notes_[i];
      }
      for (size_t i = 0;
           i < plan.behavior_notes_.size() && i < plan.behavior_note_array_.size();
           ++i) {
        plan.behavior_note_array_[i] = plan.behavior_notes_[i];
      }
    }
    plan.finalize();
    return plan;
  }

 private:
  mudnnHandle_t handle_ = nullptr;
  const EngineConfig* config_ = nullptr;
  std::shared_ptr<KernelCache> kernel_cache_;
};

class VariantPack : public BackendDescriptor {
 public:
  VariantPack() = default;

 private:
  friend class VariantPackBuilder;
  using BackendDescriptor::BackendDescriptor;
};

class VariantPackBuilder {
 public:
  VariantPackBuilder& setDataPointers(int64_t count, void* const* pointers) {
    pointers_.assign(pointers, pointers + count);
    return *this;
  }
  VariantPackBuilder& setDataPointers(const std::vector<void*>& pointers) {
    pointers_ = pointers;
    return *this;
  }
  VariantPackBuilder& setDataPointers(std::initializer_list<void*> pointers) {
    pointers_.assign(pointers.begin(), pointers.end());
    return *this;
  }
  VariantPackBuilder& setUids(int64_t count, const int64_t* uids) {
    uids_.assign(uids, uids + count);
    return *this;
  }
  VariantPackBuilder& setUids(const std::vector<int64_t>& uids) {
    uids_ = uids;
    return *this;
  }
  VariantPackBuilder& setUids(std::initializer_list<int64_t> uids) {
    uids_.assign(uids.begin(), uids.end());
    return *this;
  }
  VariantPackBuilder& setWorkspacePointer(void* workspace) {
    workspace_ = workspace;
    has_workspace_ = true;
    return *this;
  }

  VariantPack build() const {
    VariantPack pack(MUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    pack.set_attribute(MUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, MUDNN_TYPE_INT64,
                       static_cast<int64_t>(uids_.size()), uids_.data());
    pack.set_attribute(
        MUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, MUDNN_TYPE_VOID_PTR,
        static_cast<int64_t>(pointers_.size()), pointers_.data());
    if (has_workspace_) {
      pack.set_attribute(MUDNN_ATTR_VARIANT_PACK_WORKSPACE, MUDNN_TYPE_VOID_PTR,
                         1, &workspace_);
    }
    pack.finalize();
    return pack;
  }

 private:
  std::vector<void*> pointers_;
  std::vector<int64_t> uids_;
  void* workspace_ = nullptr;
  bool has_workspace_ = false;
};

inline mudnnStatus_t execute(mudnnHandle_t handle, const ExecutionPlan& plan,
                             const VariantPack& variant_pack) {
  return mudnnBackendExecute(handle, plan.get_backend_descriptor(),
                             variant_pack.get_backend_descriptor());
}

class DeviceProperties {
 public:
  struct status_t {
    bool good = true;
    std::string message;

    bool is_good() const { return good; }
    bool is_bad() const { return !good; }
    const std::string& get_message() const { return message; }
    explicit operator bool() const { return good; }
  };

  DeviceProperties& set_device_id(int64_t device_id) {
    device_id_ = device_id;
    has_device_id_ = true;
    return *this;
  }

  status_t build() {
    built_ = true;
    return {true, ""};
  }

  status_t serialize(std::vector<uint8_t>& data) const {
    std::ostringstream out;
    out << "mudnn_frontend.device_properties 1 " << has_device_id_ << ' '
        << device_id_ << ' ' << built_;
    const auto str = out.str();
    data.assign(str.begin(), str.end());
    return {true, ""};
  }

  status_t deserialize(const std::vector<uint8_t>& data) {
    return deserialize(std::string(data.begin(), data.end()));
  }

  status_t deserialize(const std::string& data) {
    std::istringstream in(data);
    std::string format;
    int version = 0;
    in >> format >> version >> has_device_id_ >> device_id_ >> built_;
    if (!in || format != "mudnn_frontend.device_properties" || version != 1) {
      return {false, "unsupported serialized device properties"};
    }
    return {true, ""};
  }

  bool has_device_id() const { return has_device_id_; }
  int64_t get_device_id() const { return device_id_; }

 private:
  bool has_device_id_ = false;
  int64_t device_id_ = 0;
  bool built_ = false;
};

class KernelCache {
 public:
  using status_t = DeviceProperties::status_t;

  status_t build(mudnnBackendDescriptor_t = nullptr) {
    finalized_ = true;
    return {true, ""};
  }

  bool is_finalized() const { return finalized_; }

  status_t serialize(std::vector<uint8_t>& data) const {
    std::ostringstream out;
    out << "mudnn_frontend.kernel_cache 1 " << finalized_;
    const auto str = out.str();
    data.assign(str.begin(), str.end());
    return {true, ""};
  }

  status_t deserialize(const std::vector<uint8_t>& data) {
    return deserialize(std::string(data.begin(), data.end()));
  }

  status_t deserialize(const std::string& data) {
    if (deserialized_ || finalized_) {
      return {false, "kernel cache has already been initialized"};
    }
    std::istringstream in(data);
    std::string format;
    int version = 0;
    in >> format >> version >> finalized_;
    if (!in || format != "mudnn_frontend.kernel_cache" || version != 1) {
      return {false, "unsupported serialized kernel cache"};
    }
    deserialized_ = true;
    return {true, ""};
  }

 private:
  bool finalized_ = false;
  bool deserialized_ = false;
};
