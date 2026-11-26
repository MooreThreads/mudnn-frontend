#pragma once

namespace graph {

enum class ScalarType { RUNTIME_PARAM = 0, COMPILE_TIME_CONST = 1 };

enum class error_code_t {
  OK = 0,
  ATTRIBUTE_NOT_SET = 1,
  INVALID_VALUE = 2,
  NOT_SUPPORTED = 3,
  GRAPH_NOT_SUPPORTED = 4,
};

struct error_t {
  bool good = true;
  std::string message;
  error_code_t code = error_code_t::OK;

  bool is_good() const { return good; }
  bool is_bad() const { return !good; }
  error_code_t get_code() const { return good ? error_code_t::OK : code; }
  const std::string& get_message() const { return message; }
  explicit operator bool() const { return good; }
};

class Tensor_attributes {
 public:
  using uid_t = int64_t;

  Tensor_attributes() = default;
  Tensor_attributes(float value) {
    scalar_float_value_ = value;
    has_scalar_float_value_ = true;
    dims_ = {1};
    strides_ = {1};
    data_type_ = MUDNN_DATA_FLOAT;
    is_by_value_ = true;
  }
  Tensor_attributes(double value) {
    scalar_double_value_ = value;
    has_scalar_double_value_ = true;
    dims_ = {1};
    strides_ = {1};
    data_type_ = MUDNN_DATA_DOUBLE;
    is_by_value_ = true;
  }

  Tensor_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Tensor_attributes& set_dim(const std::vector<int64_t>& dims) {
    dims_ = dims;
    return *this;
  }
  Tensor_attributes& set_dim(std::initializer_list<int64_t> dims) {
    dims_.assign(dims.begin(), dims.end());
    return *this;
  }
  Tensor_attributes& set_stride(const std::vector<int64_t>& strides) {
    strides_ = strides;
    return *this;
  }
  Tensor_attributes& set_stride(std::initializer_list<int64_t> strides) {
    strides_.assign(strides.begin(), strides.end());
    return *this;
  }
  Tensor_attributes& set_data_type(DataType_t data_type) {
    data_type_ = to_mudnn(data_type);
    return *this;
  }
  Tensor_attributes& set_data_type(mudnnDataType_t data_type) {
    data_type_ = data_type;
    return *this;
  }
  Tensor_attributes& set_output(bool is_output) {
    is_output_ = is_output;
    if (is_output_) {
      is_virtual_ = false;
    }
    return *this;
  }
  Tensor_attributes& set_uid(uid_t uid) {
    uid_ = uid;
    has_uid_ = true;
    return *this;
  }
  Tensor_attributes& set_is_virtual(bool is_virtual) {
    is_virtual_ = is_virtual;
    return *this;
  }
  Tensor_attributes& set_is_pass_by_value(bool is_by_value = true) {
    is_by_value_ = is_by_value;
    return *this;
  }
  Tensor_attributes& set_reordering_type(TensorReordering_t reordering_type) {
    reordering_type_ = reordering_type;
    return *this;
  }
  Tensor_attributes& set_alignment(int64_t alignment) {
    alignment_ = alignment;
    return *this;
  }
  Tensor_attributes& set_vector_count_and_dimension(int64_t vector_count,
                                                    int64_t vector_dimension) {
    vector_count_ = vector_count;
    vector_dimension_ = vector_dimension;
    return *this;
  }
  Tensor_attributes& set_ragged_offset(
      const std::shared_ptr<Tensor_attributes>& ragged_offset) {
    ragged_offset_ = ragged_offset;
    return *this;
  }
  Tensor_attributes& set_ragged_offset_multiplier(int64_t multiplier) {
    ragged_offset_multiplier_ = multiplier;
    return *this;
  }

  const std::string& get_name() const { return name_; }
  const std::vector<int64_t>& get_dim() const { return dims_; }
  const std::vector<int64_t>& get_stride() const { return strides_; }
  mudnnDataType_t get_data_type() const { return data_type_; }
  bool is_output() const { return is_output_; }
  bool is_virtual() const { return is_virtual_; }
  bool get_is_virtual() const { return is_virtual_; }
  bool is_pass_by_value() const { return is_by_value_; }
  bool get_is_pass_by_value() const { return is_by_value_; }
  bool get_has_compile_time_constant() const {
    return has_scalar_float_value_ || has_scalar_double_value_;
  }
  bool has_uid() const { return has_uid_; }
  uid_t get_uid() const { return uid_; }
  TensorReordering_t get_reordering_type() const { return reordering_type_; }
  int64_t get_alignment() const { return alignment_; }
  int64_t get_vector_count() const { return vector_count_; }
  int64_t get_vector_dimension() const { return vector_dimension_; }
  std::shared_ptr<Tensor_attributes> get_ragged_offset() const {
    return ragged_offset_;
  }
  int64_t get_ragged_offset_multiplier() const {
    return ragged_offset_multiplier_;
  }

  bool operator==(const Tensor_attributes& other) const {
    return name_ == other.name_ && dims_ == other.dims_ &&
        strides_ == other.strides_ && data_type_ == other.data_type_ &&
        uid_ == other.uid_ && has_uid_ == other.has_uid_ &&
        is_output_ == other.is_output_ && is_virtual_ == other.is_virtual_ &&
        is_by_value_ == other.is_by_value_ &&
        reordering_type_ == other.reordering_type_ &&
        alignment_ == other.alignment_ &&
        vector_count_ == other.vector_count_ &&
        vector_dimension_ == other.vector_dimension_ &&
        ragged_offset_multiplier_ == other.ragged_offset_multiplier_ &&
        static_cast<bool>(ragged_offset_) ==
        static_cast<bool>(other.ragged_offset_);
  }

  bool operator!=(const Tensor_attributes& other) const {
    return !(*this == other);
  }

  error_t validate() const {
    if (dims_.empty()) {
      return {false, "Tensor '" + name_ + "' dims not set",
              error_code_t::ATTRIBUTE_NOT_SET};
    }
    if (strides_.empty()) {
      return {false, "Tensor '" + name_ + "' strides not set",
              error_code_t::ATTRIBUTE_NOT_SET};
    }
    if (dims_.size() != strides_.size()) {
      return {false,
              "Tensor '" + name_ +
                  "' does not equal dimensionality in dim and stride",
              error_code_t::ATTRIBUTE_NOT_SET};
    }
    if (alignment_ <= 0 || vector_count_ <= 0) {
      return {false, "Tensor '" + name_ + "' vector metadata invalid",
              error_code_t::INVALID_VALUE};
    }
    if (ragged_offset_multiplier_ <= 0) {
      return {false, "Tensor '" + name_ + "' ragged offset multiplier invalid",
              error_code_t::INVALID_VALUE};
    }
    return {true, ""};
  }

 private:
  friend class Graph;

  std::string name_;
  std::vector<int64_t> dims_;
  std::vector<int64_t> strides_;
  mudnnDataType_t data_type_ =
      static_cast<mudnnDataType_t>(DataType_t::NOT_SET);
  uid_t uid_ = 0;
  bool has_uid_ = false;
  bool is_output_ = false;
  bool is_virtual_ = false;
  bool is_by_value_ = false;
  TensorReordering_t reordering_type_ = TensorReordering_t::NONE;
  int64_t alignment_ = 16;
  int64_t vector_count_ = 1;
  int64_t vector_dimension_ = -1;
  std::shared_ptr<Tensor_attributes> ragged_offset_;
  int64_t ragged_offset_multiplier_ = 1;
  float scalar_float_value_ = 0.0f;
  double scalar_double_value_ = 0.0;
  bool has_scalar_float_value_ = false;
  bool has_scalar_double_value_ = false;
};

class Graph;

namespace attn {
namespace score_modifiers {
class Softcap {
 public:
  std::shared_ptr<Tensor_attributes> forward(
      const std::shared_ptr<Graph>&,
      const std::shared_ptr<Tensor_attributes>& score,
      const std::shared_ptr<Tensor_attributes>&) {
    return score;
  }

  std::shared_ptr<Tensor_attributes> backward(
      const std::shared_ptr<Graph>&,
      const std::shared_ptr<Tensor_attributes>& grad,
      const std::shared_ptr<Tensor_attributes>&) {
    return grad;
  }
};

std::shared_ptr<Tensor_attributes> bias(
    const std::shared_ptr<Graph>& graph,
    const std::shared_ptr<Tensor_attributes>& score,
    const std::shared_ptr<Tensor_attributes>& bias);
}  // namespace score_modifiers
}  // namespace attn

class Pointwise_attributes {
 public:
  Pointwise_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Pointwise_attributes& set_mode(PointwiseMode_t mode) {
    mode_ = to_mudnn(mode);
    return *this;
  }
  Pointwise_attributes& set_mode(mudnnPointwiseMode_t mode) {
    mode_ = mode;
    return *this;
  }
  Pointwise_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Pointwise_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Pointwise_attributes& set_alpha(double alpha) {
    alpha1_ = alpha;
    return *this;
  }
  Pointwise_attributes& set_alpha2(double alpha) {
    alpha2_ = alpha;
    return *this;
  }
  Pointwise_attributes& set_relu_lower_clip(double value) {
    relu_lower_clip_ = value;
    has_relu_lower_clip_ = true;
    return *this;
  }
  Pointwise_attributes& set_relu_upper_clip(double value) {
    relu_upper_clip_ = value;
    has_relu_upper_clip_ = true;
    return *this;
  }
  Pointwise_attributes& set_relu_lower_clip_slope(double value) {
    relu_lower_clip_slope_ = value;
    has_relu_lower_clip_slope_ = true;
    return *this;
  }
  Pointwise_attributes& set_axis(int64_t axis) {
    axis_ = axis;
    has_axis_ = true;
    return *this;
  }
  Pointwise_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Pointwise_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  mudnnPointwiseMode_t mode_ = MUDNN_POINTWISE_IDENTITY;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  double alpha1_ = 1.0;
  double alpha2_ = 1.0;
  double relu_lower_clip_ = 0.0;
  double relu_upper_clip_ = 0.0;
  double relu_lower_clip_slope_ = 0.0;
  int64_t axis_ = 0;
  bool has_axis_ = false;
  bool has_relu_lower_clip_ = false;
  bool has_relu_upper_clip_ = false;
  bool has_relu_lower_clip_slope_ = false;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
};

class Matmul_attributes {
 public:
  Matmul_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Matmul_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Matmul_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
};

class Reduction_attributes {
 public:
  Reduction_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Reduction_attributes& set_mode(ReductionMode_t mode) {
    mode_ = to_mudnn(mode);
    return *this;
  }
  Reduction_attributes& set_mode(mudnnReduceTensorOp_t mode) {
    mode_ = mode;
    return *this;
  }
  Reduction_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Reduction_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Reduction_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Reduction_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  mudnnReduceTensorOp_t mode_ = MUDNN_REDUCE_TENSOR_ADD;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
};

class Genstats_attributes {
 public:
  Genstats_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Genstats_attributes& set_mode(GenStatsMode_t mode) {
    mode_ = to_mudnn(mode);
    return *this;
  }
  Genstats_attributes& set_mode(mudnnGenStatsMode_t mode) {
    mode_ = mode;
    return *this;
  }
  Genstats_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Genstats_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Genstats_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Genstats_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  mudnnGenStatsMode_t mode_ = MUDNN_GENSTATS_SUM_SQSUM;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
};

class Rng_attributes {
 public:
  Rng_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Rng_attributes& set_seed(int64_t seed) {
    seed_ = seed;
    return *this;
  }
  Rng_attributes& set_distribution(RngDistribution_t distribution) {
    distribution_ = to_mudnn(distribution);
    return *this;
  }
  Rng_attributes& set_distribution(mudnnRngDistribution_t distribution) {
    distribution_ = distribution;
    return *this;
  }
  Rng_attributes& set_uniform_range(double minimum, double maximum) {
    uniform_minimum_ = minimum;
    uniform_maximum_ = maximum;
    return *this;
  }
  Rng_attributes& set_normal_parameters(double mean, double stddev) {
    normal_mean_ = mean;
    normal_stddev_ = stddev;
    return *this;
  }
  Rng_attributes& set_bernoulli_probability(double probability) {
    bernoulli_probability_ = probability;
    return *this;
  }
  Rng_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Rng_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  int64_t seed_ = 0;
  mudnnRngDistribution_t distribution_ = MUDNN_RNG_DISTRIBUTION_UNIFORM;
  double normal_mean_ = 0.0;
  double normal_stddev_ = 1.0;
  double uniform_minimum_ = 0.0;
  double uniform_maximum_ = 1.0;
  double bernoulli_probability_ = 0.5;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
};

class Sdpa_attributes {
 public:
  Sdpa_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Sdpa_attributes& set_attn_scale(double scale) {
    attn_scale_ = scale;
    has_attn_scale_ = true;
    attn_scale_tensor_.reset();
    return *this;
  }
  Sdpa_attributes& set_attn_scale(
      const std::shared_ptr<Tensor_attributes>& scale) {
    attn_scale_tensor_ = scale;
    has_attn_scale_ = static_cast<bool>(scale);
    return *this;
  }
  Sdpa_attributes& set_causal_mask(bool causal_mask) {
    causal_mask_ = causal_mask;
    return *this;
  }
  Sdpa_attributes& set_is_inference(bool is_inference) {
    is_inference_ = is_inference;
    return *this;
  }
  Sdpa_attributes& set_generate_stats(bool generate_stats) {
    is_inference_ = !generate_stats;
    return *this;
  }
  Sdpa_attributes& set_alibi_mask(bool alibi_mask) {
    alibi_mask_ = alibi_mask;
    return *this;
  }
  Sdpa_attributes& set_diagonal_alignment(DiagonalAlignment_t alignment) {
    diagonal_alignment_ = alignment;
    return *this;
  }
  Sdpa_attributes& set_diagonal_band_right_bound(int64_t bound) {
    diagonal_band_right_bound_ = bound;
    has_diagonal_band_right_bound_ = true;
    if (bound == 0 && diagonal_alignment_ == DiagonalAlignment_t::TOP_LEFT) {
      causal_mask_ = true;
    }
    return *this;
  }
  Sdpa_attributes& set_diagonal_band_left_bound(int64_t bound) {
    diagonal_band_left_bound_ = bound;
    has_diagonal_band_left_bound_ = true;
    return *this;
  }
  Sdpa_attributes& set_dropout(
      double dropout, const std::shared_ptr<Tensor_attributes>& seed = nullptr,
      const std::shared_ptr<Tensor_attributes>& offset = nullptr) {
    if (dropout != 0.0) {
      dropout_probability_ = dropout;
      dropout_seed_ = seed;
      dropout_offset_ = offset;
    }
    return *this;
  }
  Sdpa_attributes& set_dropout(
      const std::shared_ptr<Tensor_attributes>& dropout_scale,
      const std::shared_ptr<Tensor_attributes>& dropout_mask) {
    dropout_scale_ = dropout_scale;
    dropout_mask_ = dropout_mask;
    return *this;
  }
  Sdpa_attributes& set_bias(const std::shared_ptr<Tensor_attributes>& bias) {
    bias_ = bias;
    return *this;
  }
  Sdpa_attributes& set_block_mask(
      const std::shared_ptr<Tensor_attributes>& mask) {
    block_mask_ = mask;
    return *this;
  }
  Sdpa_attributes& set_logit_max(
      const std::shared_ptr<Tensor_attributes>& logit_max) {
    logit_max_ = logit_max;
    return *this;
  }
  Sdpa_attributes& set_score_sum_exp(
      const std::shared_ptr<Tensor_attributes>& score_sum_exp) {
    score_sum_exp_ = score_sum_exp;
    return *this;
  }
  Sdpa_attributes& set_dbias(const std::shared_ptr<Tensor_attributes>& d_bias) {
    d_bias_ = d_bias;
    return *this;
  }
  Sdpa_attributes& set_sink_token(
      const std::shared_ptr<Tensor_attributes>& sink_token) {
    sink_token_ = sink_token;
    return *this;
  }
  Sdpa_attributes& set_dsink_token(
      const std::shared_ptr<Tensor_attributes>& d_sink_token) {
    d_sink_token_ = d_sink_token;
    return *this;
  }
  Sdpa_attributes& set_paged_attention_k_table(
      const std::shared_ptr<Tensor_attributes>& page_table) {
    paged_attention_k_table_ = page_table;
    return *this;
  }
  Sdpa_attributes& set_paged_attention_v_table(
      const std::shared_ptr<Tensor_attributes>& page_table) {
    paged_attention_v_table_ = page_table;
    return *this;
  }
  Sdpa_attributes& set_paged_attention_max_seq_len_kv(int64_t max_seq_len_kv) {
    paged_attention_max_seq_len_kv_ = max_seq_len_kv;
    has_paged_attention_max_seq_len_kv_ = true;
    return *this;
  }
  Sdpa_attributes& set_padding_mask(bool use_padding_mask) {
    return set_use_padding_mask(use_padding_mask);
  }
  Sdpa_attributes& set_use_padding_mask(bool use_padding_mask) {
    use_padding_mask_ = use_padding_mask;
    return *this;
  }
  Sdpa_attributes& set_seq_len_q(
      const std::shared_ptr<Tensor_attributes>& seq_len_q) {
    seq_len_q_ = seq_len_q;
    return *this;
  }
  Sdpa_attributes& set_cu_seq_len_q(
      const std::shared_ptr<Tensor_attributes>& cu_seq_len_q) {
    cu_seq_len_q_ = cu_seq_len_q;
    return *this;
  }
  Sdpa_attributes& set_seq_len_kv(
      const std::shared_ptr<Tensor_attributes>& seq_len_kv) {
    seq_len_kv_ = seq_len_kv;
    return *this;
  }
  Sdpa_attributes& set_cu_seq_len_kv(
      const std::shared_ptr<Tensor_attributes>& cu_seq_len_kv) {
    cu_seq_len_kv_ = cu_seq_len_kv;
    return *this;
  }
  Sdpa_attributes& set_seq_len_kv(
      const std::shared_ptr<Tensor_attributes>& seq_len_kv, bool /*unused*/) {
    return set_seq_len_kv(seq_len_kv);
  }
  Sdpa_attributes& set_seq_len_kv_desc(
      const std::shared_ptr<Tensor_attributes>& seq_len_kv) {
    return set_seq_len_kv(seq_len_kv);
  }
  Sdpa_attributes& set_seq_len_kv_tensor(
      const std::shared_ptr<Tensor_attributes>& seq_len_kv) {
    return set_seq_len_kv(seq_len_kv);
  }
  Sdpa_attributes& set_sliding_window_length(int64_t) {
    throw std::invalid_argument(
        "mudnn_frontend does not support SDPA sliding_window_length");
  }
  Sdpa_attributes& set_deterministic(bool deterministic) {
    deterministic_ = deterministic;
    return *this;
  }
  Sdpa_attributes& set_use_deterministic_algorithm(bool deterministic) {
    return set_deterministic(deterministic);
  }
  Sdpa_attributes& set_deterministic_algorithm(bool deterministic) {
    return set_deterministic(deterministic);
  }
  template <typename ScoreModifier>
  Sdpa_attributes& set_score_mod(ScoreModifier&&) {
    has_score_mod_ = true;
    return *this;
  }
  template <typename ScoreModifier>
  Sdpa_attributes& set_score_mod_bprop(ScoreModifier&&) {
    has_score_mod_bprop_ = true;
    return *this;
  }
  Sdpa_attributes& set_implementation(
      AttentionImplementation_t implementation) {
    implementation_ = implementation;
    has_implementation_ = true;
    return *this;
  }
  Sdpa_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Sdpa_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  double attn_scale_ = 0.0;
  bool has_attn_scale_ = false;
  std::shared_ptr<Tensor_attributes> attn_scale_tensor_;
  bool causal_mask_ = false;
  bool is_inference_ = true;
  bool alibi_mask_ = false;
  DiagonalAlignment_t diagonal_alignment_ = DiagonalAlignment_t::TOP_LEFT;
  int64_t diagonal_band_left_bound_ = 0;
  int64_t diagonal_band_right_bound_ = 0;
  bool has_diagonal_band_left_bound_ = false;
  bool has_diagonal_band_right_bound_ = false;
  double dropout_probability_ = 0.0;
  std::shared_ptr<Tensor_attributes> dropout_seed_;
  std::shared_ptr<Tensor_attributes> dropout_offset_;
  std::shared_ptr<Tensor_attributes> dropout_scale_;
  std::shared_ptr<Tensor_attributes> dropout_mask_;
  std::shared_ptr<Tensor_attributes> bias_;
  std::shared_ptr<Tensor_attributes> block_mask_;
  std::shared_ptr<Tensor_attributes> logit_max_;
  std::shared_ptr<Tensor_attributes> score_sum_exp_;
  std::shared_ptr<Tensor_attributes> d_bias_;
  std::shared_ptr<Tensor_attributes> sink_token_;
  std::shared_ptr<Tensor_attributes> d_sink_token_;
  std::shared_ptr<Tensor_attributes> paged_attention_k_table_;
  std::shared_ptr<Tensor_attributes> paged_attention_v_table_;
  int64_t paged_attention_max_seq_len_kv_ = 0;
  bool has_paged_attention_max_seq_len_kv_ = false;
  std::shared_ptr<Tensor_attributes> seq_len_q_;
  std::shared_ptr<Tensor_attributes> seq_len_kv_;
  std::shared_ptr<Tensor_attributes> cu_seq_len_q_;
  std::shared_ptr<Tensor_attributes> cu_seq_len_kv_;
  bool use_padding_mask_ = false;
  bool deterministic_ = false;
  bool has_score_mod_ = false;
  bool has_score_mod_bprop_ = false;
  AttentionImplementation_t implementation_ = AttentionImplementation_t::AUTO;
  bool has_implementation_ = false;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
};

using SDPA_attributes = Sdpa_attributes;
using SDPA_backward_attributes = Sdpa_attributes;

class Sdpa_fp8_attributes {
 public:
  Sdpa_fp8_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Sdpa_fp8_attributes& set_generate_stats(bool generate_stats) {
    generate_stats_ = generate_stats;
    return *this;
  }
  Sdpa_fp8_attributes& set_causal_mask(bool causal_mask) {
    causal_mask_ = causal_mask;
    return *this;
  }
  Sdpa_fp8_attributes& set_causal_mask_bottom_right(bool causal_mask) {
    causal_mask_bottom_right_ = causal_mask;
    return *this;
  }
  Sdpa_fp8_attributes& set_attn_scale(double scale) {
    attn_scale_ = scale;
    has_attn_scale_ = true;
    return *this;
  }
  Sdpa_fp8_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Sdpa_fp8_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  bool generate_stats_ = true;
  bool causal_mask_ = false;
  bool causal_mask_bottom_right_ = false;
  double attn_scale_ = 0.0;
  bool has_attn_scale_ = false;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
};

using SDPA_fp8_attributes = Sdpa_fp8_attributes;

class Sdpa_fp8_backward_attributes {
 public:
  Sdpa_fp8_backward_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Sdpa_fp8_backward_attributes& set_attn_scale(double scale) {
    attn_scale_ = scale;
    has_attn_scale_ = true;
    return *this;
  }
  Sdpa_fp8_backward_attributes& set_causal_mask(bool causal_mask) {
    causal_mask_ = causal_mask;
    return *this;
  }
  Sdpa_fp8_backward_attributes& set_causal_mask_bottom_right(bool causal_mask) {
    causal_mask_bottom_right_ = causal_mask;
    return *this;
  }
  Sdpa_fp8_backward_attributes& set_deterministic_algorithm(
      bool deterministic) {
    deterministic_ = deterministic;
    return *this;
  }
  Sdpa_fp8_backward_attributes& set_use_deterministic_algorithm(
      bool deterministic) {
    return set_deterministic_algorithm(deterministic);
  }
  Sdpa_fp8_backward_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Sdpa_fp8_backward_attributes& set_compute_data_type(
      mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  double attn_scale_ = 0.0;
  bool has_attn_scale_ = false;
  bool causal_mask_ = false;
  bool causal_mask_bottom_right_ = false;
  bool deterministic_ = false;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
};

using SDPA_fp8_backward_attributes = Sdpa_fp8_backward_attributes;

class Conv_attributes {
 public:
  Conv_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Conv_attributes& set_padding(const std::vector<int64_t>& padding) {
    padding_ = padding;
    pre_padding_ = padding;
    post_padding_ = padding;
    return *this;
  }
  Conv_attributes& set_padding(std::initializer_list<int64_t> padding) {
    padding_.assign(padding.begin(), padding.end());
    pre_padding_ = padding_;
    post_padding_ = padding_;
    return *this;
  }
  Conv_attributes& set_pre_padding(const std::vector<int64_t>& padding) {
    pre_padding_ = padding;
    padding_ = pre_padding_;
    return *this;
  }
  Conv_attributes& set_pre_padding(std::initializer_list<int64_t> padding) {
    pre_padding_.assign(padding.begin(), padding.end());
    padding_ = pre_padding_;
    return *this;
  }
  Conv_attributes& set_post_padding(const std::vector<int64_t>& padding) {
    post_padding_ = padding;
    return *this;
  }
  Conv_attributes& set_post_padding(std::initializer_list<int64_t> padding) {
    post_padding_.assign(padding.begin(), padding.end());
    return *this;
  }
  Conv_attributes& set_stride(const std::vector<int64_t>& stride) {
    stride_ = stride;
    return *this;
  }
  Conv_attributes& set_stride(std::initializer_list<int64_t> stride) {
    stride_.assign(stride.begin(), stride.end());
    return *this;
  }
  Conv_attributes& set_dilation(const std::vector<int64_t>& dilation) {
    dilation_ = dilation;
    return *this;
  }
  Conv_attributes& set_dilation(std::initializer_list<int64_t> dilation) {
    dilation_.assign(dilation.begin(), dilation.end());
    return *this;
  }
  Conv_attributes& set_mode(ConvolutionMode_t mode) {
    mode_ = to_mudnn(mode);
    return *this;
  }
  Conv_attributes& set_mode(mudnnConvolutionMode_t mode) {
    mode_ = mode;
    return *this;
  }
  Conv_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Conv_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Conv_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Conv_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }
  Conv_attributes& set_alpha(double alpha) {
    alpha_ = alpha;
    return *this;
  }
  Conv_attributes& set_beta(double beta) {
    beta_ = beta;
    return *this;
  }
  Conv_attributes& set_group_count(int64_t group_count) {
    group_count_ = group_count;
    has_group_count_ = true;
    return *this;
  }

  const std::string& get_name() const { return name_; }
  const std::vector<int64_t>& get_padding() const { return padding_; }
  const std::vector<int64_t>& get_pre_padding() const { return pre_padding_; }
  const std::vector<int64_t>& get_post_padding() const { return post_padding_; }
  const std::vector<int64_t>& get_stride() const { return stride_; }
  const std::vector<int64_t>& get_dilation() const { return dilation_; }
  mudnnConvolutionMode_t get_mode() const { return mode_; }
  mudnnDataType_t get_compute_data_type() const { return compute_data_type_; }
  mudnnDataType_t get_output_data_type() const { return output_data_type_; }
  bool has_output_data_type() const { return has_output_data_type_; }
  double get_alpha() const { return alpha_; }
  double get_beta() const { return beta_; }
  int64_t get_group_count() const { return group_count_; }
  bool has_group_count() const { return has_group_count_; }

  bool operator==(const Conv_attributes& other) const {
    return name_ == other.name_ && padding_ == other.padding_ &&
        pre_padding_ == other.pre_padding_ &&
        post_padding_ == other.post_padding_ && stride_ == other.stride_ &&
        dilation_ == other.dilation_ && mode_ == other.mode_ &&
        compute_data_type_ == other.compute_data_type_ &&
        output_data_type_ == other.output_data_type_ &&
        has_output_data_type_ == other.has_output_data_type_ &&
        group_count_ == other.group_count_ &&
        has_group_count_ == other.has_group_count_ && alpha_ == other.alpha_ &&
        beta_ == other.beta_;
  }

  bool operator!=(const Conv_attributes& other) const {
    return !(*this == other);
  }

 private:
  friend class Graph;

  std::string name_;
  std::vector<int64_t> padding_ = {0, 0};
  std::vector<int64_t> pre_padding_ = {0, 0};
  std::vector<int64_t> post_padding_ = {0, 0};
  std::vector<int64_t> stride_ = {1, 1};
  std::vector<int64_t> dilation_ = {1, 1};
  mudnnConvolutionMode_t mode_ = MUDNN_CROSS_CORRELATION;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
  int64_t group_count_ = 1;
  bool has_group_count_ = false;
  double alpha_ = 1.0;
  double beta_ = 0.0;
};

using Conv_fprop_attributes = Conv_attributes;
using Conv_dgrad_attributes = Conv_attributes;
using Conv_wgrad_attributes = Conv_attributes;

class Resample_attributes {
 public:
  Resample_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Resample_attributes& set_mode(ResampleMode_t mode) {
    mode_ = to_mudnn(mode);
    return *this;
  }
  Resample_attributes& set_mode(mudnnResampleMode_t mode) {
    mode_ = mode;
    return *this;
  }
  Resample_attributes& set_resampling_mode(ResampleMode_t mode) {
    return set_mode(mode);
  }
  Resample_attributes& set_resampling_mode(mudnnResampleMode_t mode) {
    return set_mode(mode);
  }
  Resample_attributes& set_window(const std::vector<int64_t>& window) {
    window_ = window;
    return *this;
  }
  Resample_attributes& set_window(std::initializer_list<int64_t> window) {
    window_.assign(window.begin(), window.end());
    return *this;
  }
  Resample_attributes& set_padding(const std::vector<int64_t>& padding) {
    padding_ = padding;
    return *this;
  }
  Resample_attributes& set_padding(std::initializer_list<int64_t> padding) {
    padding_.assign(padding.begin(), padding.end());
    return *this;
  }
  Resample_attributes& set_pre_padding(const std::vector<int64_t>& padding) {
    padding_ = padding;
    pre_padding_ = padding;
    has_pre_padding_ = true;
    return *this;
  }
  Resample_attributes& set_pre_padding(std::initializer_list<int64_t> padding) {
    padding_.assign(padding.begin(), padding.end());
    pre_padding_ = padding_;
    has_pre_padding_ = true;
    return *this;
  }
  Resample_attributes& set_post_padding(const std::vector<int64_t>& padding) {
    post_padding_ = padding;
    has_post_padding_ = true;
    return *this;
  }
  Resample_attributes& set_post_padding(
      std::initializer_list<int64_t> padding) {
    post_padding_.assign(padding.begin(), padding.end());
    has_post_padding_ = true;
    return *this;
  }
  Resample_attributes& set_padding_mode(PaddingMode_t padding_mode) {
    padding_mode_ = padding_mode;
    return *this;
  }
  Resample_attributes& set_stride(const std::vector<int64_t>& stride) {
    stride_ = stride;
    has_stride_ = true;
    return *this;
  }
  Resample_attributes& set_stride(std::initializer_list<int64_t> stride) {
    stride_.assign(stride.begin(), stride.end());
    has_stride_ = true;
    return *this;
  }
  Resample_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Resample_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Resample_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Resample_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }
  Resample_attributes& set_alpha(double alpha) {
    alpha_ = alpha;
    return *this;
  }
  Resample_attributes& set_beta(double beta) {
    beta_ = beta;
    return *this;
  }
  Resample_attributes& set_return_indices(bool return_indices = true) {
    return_indices_ = return_indices;
    return *this;
  }
  Resample_attributes& set_generate_index(bool generate_index = true) {
    return set_return_indices(generate_index);
  }

 private:
  friend class Graph;

  std::string name_;
  mudnnResampleMode_t mode_ = MUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
  PaddingMode_t padding_mode_ = PaddingMode_t::ZERO_PAD;
  std::vector<int64_t> window_ = {2, 2};
  std::vector<int64_t> padding_ = {0, 0};
  std::vector<int64_t> pre_padding_;
  std::vector<int64_t> post_padding_;
  std::vector<int64_t> stride_ = {2, 2};
  bool has_pre_padding_ = false;
  bool has_post_padding_ = false;
  bool has_stride_ = false;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
  double alpha_ = 1.0;
  double beta_ = 0.0;
  bool return_indices_ = false;
};

class Batchnorm_attributes {
 public:
  Batchnorm_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Batchnorm_attributes& set_epsilon(double epsilon) {
    epsilon_ = epsilon;
    epsilon_tensor_.reset();
    return *this;
  }
  Batchnorm_attributes& set_epsilon(
      const std::shared_ptr<Tensor_attributes>& epsilon) {
    epsilon_tensor_ = epsilon;
    return *this;
  }
  Batchnorm_attributes& set_momentum(double momentum) {
    momentum_ = momentum;
    momentum_tensor_.reset();
    return *this;
  }
  Batchnorm_attributes& set_momentum(
      const std::shared_ptr<Tensor_attributes>& momentum) {
    momentum_tensor_ = momentum;
    return *this;
  }
  Batchnorm_attributes& set_previous_running_stats(
      const std::shared_ptr<Tensor_attributes>& running_mean,
      const std::shared_ptr<Tensor_attributes>& running_var, double momentum) {
    previous_running_mean_ = running_mean;
    previous_running_var_ = running_var;
    set_momentum(momentum);
    return *this;
  }
  Batchnorm_attributes& set_previous_running_stats(
      const std::shared_ptr<Tensor_attributes>& running_mean,
      const std::shared_ptr<Tensor_attributes>& running_var,
      const std::shared_ptr<Tensor_attributes>& momentum) {
    previous_running_mean_ = running_mean;
    previous_running_var_ = running_var;
    set_momentum(momentum);
    return *this;
  }
  Batchnorm_attributes& set_saved_mean_and_inv_variance(
      const std::shared_ptr<Tensor_attributes>& mean,
      const std::shared_ptr<Tensor_attributes>& inv_variance) {
    saved_mean_ = mean;
    saved_inv_variance_ = inv_variance;
    return *this;
  }
  Batchnorm_attributes& set_peer_stats(
      const std::vector<std::shared_ptr<Tensor_attributes>>& peer_stats) {
    peer_stats_ = peer_stats;
    return *this;
  }
  Batchnorm_attributes& set_peer_stats(
      std::initializer_list<std::shared_ptr<Tensor_attributes>> peer_stats) {
    peer_stats_.assign(peer_stats.begin(), peer_stats.end());
    return *this;
  }
  Batchnorm_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Batchnorm_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Batchnorm_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Batchnorm_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  double epsilon_ = 1.0e-5;
  double momentum_ = 0.1;
  std::shared_ptr<Tensor_attributes> epsilon_tensor_;
  std::shared_ptr<Tensor_attributes> momentum_tensor_;
  std::shared_ptr<Tensor_attributes> previous_running_mean_;
  std::shared_ptr<Tensor_attributes> previous_running_var_;
  std::shared_ptr<Tensor_attributes> saved_mean_;
  std::shared_ptr<Tensor_attributes> saved_inv_variance_;
  std::vector<std::shared_ptr<Tensor_attributes>> peer_stats_;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
};

using Batchnorm_backward_attributes = Batchnorm_attributes;
using Batchnorm_inference_attributes = Batchnorm_attributes;

class Norm_attributes {
 public:
  Norm_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Norm_attributes& set_epsilon(double epsilon) {
    epsilon_ = epsilon;
    epsilon_tensor_.reset();
    return *this;
  }
  Norm_attributes& set_epsilon(
      const std::shared_ptr<Tensor_attributes>& epsilon) {
    epsilon_tensor_ = epsilon;
    return *this;
  }
  Norm_attributes& set_norm_forward_phase(NormFwdPhase_t phase) {
    phase_ = to_mudnn(phase);
    return *this;
  }
  Norm_attributes& set_norm_forward_phase(mudnnBackendNormFwdPhase_t phase) {
    phase_ = phase;
    return *this;
  }
  Norm_attributes& set_forward_phase(NormFwdPhase_t phase) {
    return set_norm_forward_phase(phase);
  }
  Norm_attributes& set_forward_phase(mudnnBackendNormFwdPhase_t phase) {
    return set_norm_forward_phase(phase);
  }
  Norm_attributes& set_saved_mean_and_inv_variance(
      const std::shared_ptr<Tensor_attributes>& mean,
      const std::shared_ptr<Tensor_attributes>& inv_variance) {
    saved_mean_ = mean;
    saved_inv_variance_ = inv_variance;
    return *this;
  }
  Norm_attributes& set_bias(const std::shared_ptr<Tensor_attributes>& bias) {
    bias_ = bias;
    return *this;
  }
  Norm_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Norm_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Norm_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Norm_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }
  Norm_attributes& set_has_dbias(bool has_dbias) {
    has_dbias_ = has_dbias;
    return *this;
  }
  Norm_attributes& has_dbias(bool has_dbias) {
    return set_has_dbias(has_dbias);
  }

 private:
  friend class Graph;

  std::string name_;
  double epsilon_ = 1.0e-5;
  std::shared_ptr<Tensor_attributes> epsilon_tensor_;
  std::shared_ptr<Tensor_attributes> saved_mean_;
  std::shared_ptr<Tensor_attributes> saved_inv_variance_;
  std::shared_ptr<Tensor_attributes> bias_;
  mudnnBackendNormFwdPhase_t phase_ = MUDNN_NORM_FWD_TRAINING;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
  bool has_dbias_ = true;
};

using Layernorm_attributes = Norm_attributes;
using Layernorm_backward_attributes = Norm_attributes;
using AdaLayernorm_attributes = Norm_attributes;
using AdaLayernorm_backward_attributes = Norm_attributes;
using Rmsnorm_attributes = Norm_attributes;
using Rmsnorm_backward_attributes = Norm_attributes;
using Instancenorm_attributes = Norm_attributes;
using Instancenorm_backward_attributes = Norm_attributes;

class Concatenate_attributes {
 public:
  Concatenate_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Concatenate_attributes& set_axis(int64_t axis) {
    axis_ = axis;
    return *this;
  }
  Concatenate_attributes& set_in_place_index(int64_t index) {
    in_place_index_ = index;
    return *this;
  }
  const std::string& get_name() const { return name_; }
  int64_t get_axis() const { return axis_; }
  int64_t get_in_place_index() const { return in_place_index_; }

 private:
  friend class Graph;

  std::string name_;
  int64_t axis_ = 0;
  int64_t in_place_index_ = -1;
};

class Reshape_attributes {
 public:
  Reshape_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Reshape_attributes& set_reshape_mode(ReshapeMode_t mode) {
    reshape_mode_ = static_cast<mudnnBackendReshapeMode_t>(mode);
    return *this;
  }
  Reshape_attributes& set_reshape_mode(mudnnBackendReshapeMode_t mode) {
    reshape_mode_ = mode;
    return *this;
  }
  const std::string& get_name() const { return name_; }
  mudnnBackendReshapeMode_t get_reshape_mode() const { return reshape_mode_; }

 private:
  friend class Graph;

  std::string name_;
  mudnnBackendReshapeMode_t reshape_mode_ = MUDNN_RESHAPE_VIEW_ONLY;
};

class Transpose_attributes {
 public:
  Transpose_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Transpose_attributes& set_permutation(
      const std::vector<int64_t>& permutation) {
    permutation_ = permutation;
    return *this;
  }
  Transpose_attributes& set_permutation(
      std::initializer_list<int64_t> permutation) {
    permutation_.assign(permutation.begin(), permutation.end());
    return *this;
  }
  Transpose_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Transpose_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  const std::string& get_name() const { return name_; }
  const std::vector<int64_t>& get_permutation() const { return permutation_; }

 private:
  friend class Graph;

  std::string name_;
  std::vector<int64_t> permutation_;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
};

class Slice_attributes {
 public:
  Slice_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Slice_attributes& set_slices(
      const std::vector<std::pair<int64_t, int64_t>>& slices) {
    slices_ = slices;
    return *this;
  }
  Slice_attributes& set_slices(
      std::initializer_list<std::pair<int64_t, int64_t>> slices) {
    slices_.assign(slices.begin(), slices.end());
    return *this;
  }
  Slice_attributes& set_strides(const std::vector<int64_t>& strides) {
    strides_ = strides;
    return *this;
  }
  Slice_attributes& set_strides(std::initializer_list<int64_t> strides) {
    strides_.assign(strides.begin(), strides.end());
    return *this;
  }
  const std::string& get_name() const { return name_; }
  const std::vector<std::pair<int64_t, int64_t>>& get_slices() const {
    return slices_;
  }
  const std::vector<int64_t>& get_strides() const { return strides_; }

 private:
  friend class Graph;

  std::string name_;
  std::vector<std::pair<int64_t, int64_t>> slices_;
  std::vector<int64_t> strides_;
};

class Bn_finalize_attributes {
 public:
  Bn_finalize_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Bn_finalize_attributes& set_epsilon(double epsilon) {
    epsilon_ = epsilon;
    return *this;
  }
  Bn_finalize_attributes& set_momentum(double momentum) {
    momentum_ = momentum;
    momentum_tensor_.reset();
    return *this;
  }
  Bn_finalize_attributes& set_momentum(
      const std::shared_ptr<Tensor_attributes>& momentum) {
    momentum_tensor_ = momentum;
    return *this;
  }
  Bn_finalize_attributes& set_previous_running_stats(
      const std::shared_ptr<Tensor_attributes>& running_mean,
      const std::shared_ptr<Tensor_attributes>& running_var, double momentum) {
    previous_running_mean_ = running_mean;
    previous_running_var_ = running_var;
    set_momentum(momentum);
    return *this;
  }
  Bn_finalize_attributes& set_previous_running_stats(
      const std::shared_ptr<Tensor_attributes>& running_mean,
      const std::shared_ptr<Tensor_attributes>& running_var,
      const std::shared_ptr<Tensor_attributes>& momentum) {
    previous_running_mean_ = running_mean;
    previous_running_var_ = running_var;
    set_momentum(momentum);
    return *this;
  }
  Bn_finalize_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Bn_finalize_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Bn_finalize_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Bn_finalize_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  double epsilon_ = 1.0e-5;
  double momentum_ = 0.1;
  std::shared_ptr<Tensor_attributes> momentum_tensor_;
  std::shared_ptr<Tensor_attributes> previous_running_mean_;
  std::shared_ptr<Tensor_attributes> previous_running_var_;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
};

using BN_finalize_attributes = Bn_finalize_attributes;

class Bn_backward_weights_attributes {
 public:
  Bn_backward_weights_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Bn_backward_weights_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Bn_backward_weights_attributes& set_compute_data_type(
      mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Bn_backward_weights_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Bn_backward_weights_attributes& set_output_data_type(
      mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
};

using DBN_weight_attributes = Bn_backward_weights_attributes;

class Block_scale_attributes {
 public:
  Block_scale_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Block_scale_attributes& set_block_size(int64_t block_size) {
    block_size_ = block_size;
    block_axis_tail_offset_ = 0;
    return *this;
  }
  Block_scale_attributes& set_block_size(
      const std::vector<int64_t>& block_size) {
    const auto normalized = normalize_block_size(block_size);
    block_size_ = normalized.first;
    block_axis_tail_offset_ = normalized.second;
    return *this;
  }
  Block_scale_attributes& set_block_size(
      std::initializer_list<int64_t> block_size) {
    return set_block_size(std::vector<int64_t>(block_size));
  }
  Block_scale_attributes& set_block_size(
      const std::vector<int32_t>& block_size) {
    std::vector<int64_t> values(block_size.begin(), block_size.end());
    return set_block_size(values);
  }
  Block_scale_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Block_scale_attributes& set_compute_data_type(mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Block_scale_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Block_scale_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }
  Block_scale_attributes& set_scale_data_type(DataType_t data_type) {
    scale_data_type_ = to_mudnn(data_type);
    has_scale_data_type_ = true;
    return *this;
  }
  Block_scale_attributes& set_scale_data_type(mudnnDataType_t data_type) {
    scale_data_type_ = data_type;
    has_scale_data_type_ = true;
    return *this;
  }

 private:
  static std::pair<int64_t, int64_t> normalize_block_size(
      const std::vector<int64_t>& block_size) {
    if (block_size.empty()) {
      throw std::invalid_argument("block_scale block_size must be non-empty");
    }
    if (block_size.size() == 1) {
      return {block_size[0], 0};
    }
    int64_t blocked_index = -1;
    for (size_t i = 0; i < block_size.size(); ++i) {
      if (block_size[i] != 1) {
        if (blocked_index >= 0) {
          throw std::invalid_argument(
              "block_scale multi-axis block_size is not supported");
        }
        blocked_index = static_cast<int64_t>(i);
      }
    }
    if (blocked_index < 0) {
      return {1, 0};
    }
    const auto tail_offset =
        blocked_index - static_cast<int64_t>(block_size.size() - 1);
    return {block_size[blocked_index], tail_offset};
  }

  friend class Graph;

  std::string name_;
  int64_t block_size_ = 1;
  int64_t block_axis_tail_offset_ = 0;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
  mudnnDataType_t scale_data_type_ = MUDNN_DATA_FLOAT;
  bool has_scale_data_type_ = false;
};

class Block_scale_quantize_attributes : public Block_scale_attributes {
 public:
  Block_scale_quantize_attributes& set_name(const std::string& name) {
    Block_scale_attributes::set_name(name);
    return *this;
  }
  Block_scale_quantize_attributes& set_block_size(int64_t block_size) {
    Block_scale_attributes::set_block_size(block_size);
    return *this;
  }
  Block_scale_quantize_attributes& set_block_size(
      const std::vector<int64_t>& block_size) {
    Block_scale_attributes::set_block_size(block_size);
    return *this;
  }
  Block_scale_quantize_attributes& set_block_size(
      std::initializer_list<int64_t> block_size) {
    Block_scale_attributes::set_block_size(block_size);
    return *this;
  }
  Block_scale_quantize_attributes& set_block_size(
      const std::vector<int32_t>& block_size) {
    Block_scale_attributes::set_block_size(block_size);
    return *this;
  }
  Block_scale_quantize_attributes& set_axis(int64_t axis) {
    axis_ = axis;
    has_axis_ = true;
    return *this;
  }
  Block_scale_quantize_attributes& set_transpose(bool transpose) {
    transpose_ = transpose;
    return *this;
  }
  Block_scale_quantize_attributes& set_compute_data_type(DataType_t data_type) {
    Block_scale_attributes::set_compute_data_type(data_type);
    return *this;
  }
  Block_scale_quantize_attributes& set_compute_data_type(
      mudnnDataType_t data_type) {
    Block_scale_attributes::set_compute_data_type(data_type);
    return *this;
  }
  Block_scale_quantize_attributes& set_output_data_type(DataType_t data_type) {
    Block_scale_attributes::set_output_data_type(data_type);
    return *this;
  }
  Block_scale_quantize_attributes& set_output_data_type(
      mudnnDataType_t data_type) {
    Block_scale_attributes::set_output_data_type(data_type);
    return *this;
  }
  Block_scale_quantize_attributes& set_scale_data_type(DataType_t data_type) {
    Block_scale_attributes::set_scale_data_type(data_type);
    return *this;
  }
  Block_scale_quantize_attributes& set_scale_data_type(
      mudnnDataType_t data_type) {
    Block_scale_attributes::set_scale_data_type(data_type);
    return *this;
  }

 private:
  friend class Graph;

  int64_t axis_ = -1;
  bool has_axis_ = false;
  bool transpose_ = false;
};

class Block_scale_dequantize_attributes : public Block_scale_attributes {
 public:
  Block_scale_dequantize_attributes& set_name(const std::string& name) {
    Block_scale_attributes::set_name(name);
    return *this;
  }
  Block_scale_dequantize_attributes& set_block_size(int64_t block_size) {
    Block_scale_attributes::set_block_size(block_size);
    return *this;
  }
  Block_scale_dequantize_attributes& set_block_size(
      const std::vector<int64_t>& block_size) {
    Block_scale_attributes::set_block_size(block_size);
    return *this;
  }
  Block_scale_dequantize_attributes& set_block_size(
      std::initializer_list<int64_t> block_size) {
    Block_scale_attributes::set_block_size(block_size);
    return *this;
  }
  Block_scale_dequantize_attributes& set_block_size(
      const std::vector<int32_t>& block_size) {
    Block_scale_attributes::set_block_size(block_size);
    return *this;
  }
  Block_scale_dequantize_attributes& set_compute_data_type(
      DataType_t data_type) {
    Block_scale_attributes::set_compute_data_type(data_type);
    return *this;
  }
  Block_scale_dequantize_attributes& set_compute_data_type(
      mudnnDataType_t data_type) {
    Block_scale_attributes::set_compute_data_type(data_type);
    return *this;
  }
  Block_scale_dequantize_attributes& set_output_data_type(
      DataType_t data_type) {
    Block_scale_attributes::set_output_data_type(data_type);
    return *this;
  }
  Block_scale_dequantize_attributes& set_output_data_type(
      mudnnDataType_t data_type) {
    Block_scale_attributes::set_output_data_type(data_type);
    return *this;
  }
  Block_scale_dequantize_attributes& set_is_negative_scale(bool value) {
    is_negative_scale_ = value;
    return *this;
  }
  bool get_is_negative_scale() const { return is_negative_scale_; }

 private:
  friend class Graph;

  bool is_negative_scale_ = false;
};

class Moe_grouped_matmul_attributes {
 public:
  Moe_grouped_matmul_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Moe_grouped_matmul_attributes& set_mode(MoeGroupedMatmulMode_t mode) {
    mode_ = to_mudnn(mode);
    return *this;
  }
  Moe_grouped_matmul_attributes& set_mode(mudnnMoeGroupedMatmulMode_t mode) {
    mode_ = mode;
    return *this;
  }
  Moe_grouped_matmul_attributes& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Moe_grouped_matmul_attributes& set_compute_data_type(
      mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }
  Moe_grouped_matmul_attributes& set_top_k(int32_t top_k) {
    top_k_ = top_k;
    return *this;
  }
  Moe_grouped_matmul_attributes& set_top_k(int64_t top_k) {
    top_k_ = static_cast<int32_t>(top_k);
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  mudnnMoeGroupedMatmulMode_t mode_ = MUDNN_MOE_GROUPED_MATMUL_MODE_NONE;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
  int32_t top_k_ = 0;
};

class Moe_grouped_matmul_bwd_attributes {
 public:
  Moe_grouped_matmul_bwd_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Moe_grouped_matmul_bwd_attributes& set_compute_data_type(
      DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Moe_grouped_matmul_bwd_attributes& set_compute_data_type(
      mudnnDataType_t data_type) {
    compute_data_type_ = data_type;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  mudnnDataType_t compute_data_type_ = MUDNN_DATA_FLOAT;
};

class Paged_cache_load_attributes {
 public:
  Paged_cache_load_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Paged_cache_load_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Paged_cache_load_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
};

// Match cudnn-frontend's public spelling while retaining the original muDNN
// spelling for existing users.
using PagedCacheLoad_attributes = Paged_cache_load_attributes;

class Signal_attributes {
 public:
  Signal_attributes& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Signal_attributes& set_value(int64_t value) {
    value_ = value;
    return *this;
  }
  Signal_attributes& set_output_data_type(DataType_t data_type) {
    output_data_type_ = to_mudnn(data_type);
    has_output_data_type_ = true;
    return *this;
  }
  Signal_attributes& set_output_data_type(mudnnDataType_t data_type) {
    output_data_type_ = data_type;
    has_output_data_type_ = true;
    return *this;
  }

 private:
  friend class Graph;

  std::string name_;
  int64_t value_ = 0;
  mudnnDataType_t output_data_type_ = MUDNN_DATA_FLOAT;
  bool has_output_data_type_ = false;
};

#ifdef MUDNN_FRONTEND_HAS_NLOHMANN_JSON
mudnnDataType_t upstream_data_type_from_json(const nlohmann::json& value);
void from_json(const nlohmann::json& j, Tensor_attributes& tensor);
#endif

class Graph {
 public:
#ifdef MUDNN_FRONTEND_HAS_NLOHMANN_JSON
  friend void to_json(nlohmann::json& j, const Graph& graph);
#endif

  Graph() = default;

  Graph& set_io_data_type(DataType_t data_type) {
    io_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Graph& set_intermediate_data_type(DataType_t data_type) {
    intermediate_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Graph& set_compute_data_type(DataType_t data_type) {
    compute_data_type_ = to_mudnn(data_type);
    return *this;
  }
  Graph& set_sm_count(int64_t sm_count) {
    sm_count_ = sm_count;
    return *this;
  }
  Graph& set_sm_version(int64_t sm_version) {
    sm_version_ = sm_version;
    return *this;
  }
  Graph& set_name(const std::string& name) {
    name_ = name;
    return *this;
  }
  Graph& set_dynamic_shape_enabled(bool enabled) {
    dynamic_shape_enabled_ = enabled;
    return *this;
  }
  Graph& set_override_shape_enabled(bool enabled) {
    override_shape_enabled_ = enabled;
    return *this;
  }
  Graph& set_device_properties(
      const std::shared_ptr<const DeviceProperties>& device_properties) {
    device_properties_ = device_properties;
    return *this;
  }
  Graph& set_kernel_cache(const std::shared_ptr<KernelCache>& kernel_cache) {
    kernel_cache_ = kernel_cache;
    return *this;
  }
  Graph& deselect_engines(const std::vector<std::string>& names) {
    deselected_engine_names_ = names;
    refresh_engine_selection();
    return *this;
  }
  Graph& deselect_shared_mem_greater_than(int64_t limit) {
    max_shared_memory_ = limit;
    refresh_engine_selection();
    return *this;
  }
  Graph& deselect_workspace_greater_than(int64_t limit) {
    max_workspace_ = limit;
    refresh_engine_selection();
    return *this;
  }

  std::shared_ptr<Tensor_attributes> tensor(
      const Tensor_attributes& attributes) {
    auto tensor = std::make_shared<Tensor_attributes>(attributes);
    // Upstream cudnn-frontend semantics: a tensor created through the graph
    // without an explicit data type inherits the graph io data type.
    if (tensor->data_type_ ==
        static_cast<mudnnDataType_t>(DataType_t::NOT_SET)) {
      tensor->set_data_type(io_data_type_);
    }
    register_tensor(tensor);
    return tensor;
  }

  std::shared_ptr<Tensor_attributes> tensor_like(
      const std::shared_ptr<Tensor_attributes>& other) {
    if (!other) {
      throw std::invalid_argument("tensor_like requires a tensor");
    }
    Tensor_attributes attributes;
    attributes.dims_ = other->dims_;
    attributes.strides_ = other->strides_;
    attributes.data_type_ = other->data_type_;
    attributes.is_virtual_ = other->is_virtual_;
    attributes.is_by_value_ = other->is_by_value_;
    attributes.alignment_ = other->alignment_;
    attributes.vector_count_ = other->vector_count_;
    attributes.vector_dimension_ = other->vector_dimension_;
    attributes.reordering_type_ = other->reordering_type_;
    attributes.ragged_offset_ = other->ragged_offset_;
    attributes.ragged_offset_multiplier_ = other->ragged_offset_multiplier_;
    return tensor(attributes);
  }

  std::shared_ptr<Tensor_attributes> tensor_like(
      const std::shared_ptr<Tensor_attributes>& other,
      const std::string& name) {
    auto result = tensor_like(other);
    result->set_name(name);
    return result;
  }

  std::shared_ptr<Tensor_attributes> tensor(
      float scalar, ScalarType scalar_type = ScalarType::COMPILE_TIME_CONST) {
    return make_scalar_by_value("scalar_" + std::to_string(next_uid_), scalar,
                                scalar_type, MUDNN_DATA_FLOAT);
  }

  std::shared_ptr<Tensor_attributes> tensor(
      double scalar, ScalarType scalar_type = ScalarType::COMPILE_TIME_CONST) {
    return make_scalar_by_value("scalar_" + std::to_string(next_uid_), scalar,
                                scalar_type, MUDNN_DATA_DOUBLE);
  }

  std::shared_ptr<Tensor_attributes> tensor(
      int32_t scalar, ScalarType scalar_type = ScalarType::COMPILE_TIME_CONST) {
    return make_scalar_by_value("scalar_" + std::to_string(next_uid_), scalar,
                                scalar_type, MUDNN_DATA_INT32);
  }

  std::shared_ptr<Tensor_attributes> tensor(
      int64_t scalar, ScalarType scalar_type = ScalarType::COMPILE_TIME_CONST) {
    return make_scalar_by_value("scalar_" + std::to_string(next_uid_), scalar,
                                scalar_type, MUDNN_DATA_INT64);
  }

  const std::vector<std::shared_ptr<Tensor_attributes>>& get_tensors() const {
    return tensors_;
  }

  error_t query_tensor_attributes_of_uid(int64_t uid,
                                         Tensor_attributes& output) const {
    for (const auto& tensor : tensors_) {
      if (tensor && tensor->has_uid_ && tensor->uid_ == uid) {
        output = *tensor;
        return {true, ""};
      }
    }
    return {false, "tensor uid " + std::to_string(uid) + " not found",
            error_code_t::INVALID_VALUE};
  }

  std::shared_ptr<Tensor_attributes> pointwise(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes) {
    return pointwise_impl(x, nullptr, nullptr, attributes);
  }

  std::shared_ptr<Tensor_attributes> pointwise(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes) {
    return pointwise_impl(x, b, nullptr, attributes);
  }

  std::shared_ptr<Tensor_attributes> add(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_ADD);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> sub(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_SUB);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> mul(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_MUL);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> div(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_DIV);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> mod(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_MOD);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> pow(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_POW);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> add_square(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_ADD_SQUARE);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> max(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_MAX);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> min(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_MIN);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> sqrt(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_SQRT);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> rsqrt(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_RSQRT);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> reciprocal(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_RECIPROCAL);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> abs(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_ABS);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> ceil(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_CEIL);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> floor(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_FLOOR);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> exp(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_EXP);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> log(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_LOG);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> sin(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_SIN);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> cos(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_COS);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> tan(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_TAN);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> erf(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_ERF);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> tanh(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_TANH_FWD);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> sigmoid(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_SIGMOID_FWD);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> gelu(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_GELU_FWD);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> gelu_approx_tanh(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_GELU_APPROX_TANH_FWD);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> scale(
      const std::shared_ptr<Tensor_attributes>& x, double scalar,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_IDENTITY).set_alpha(scalar);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> identity(
      const std::shared_ptr<Tensor_attributes>& input,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_IDENTITY);
    return pointwise_impl(input, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> relu(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_RELU_FWD);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> cmp_eq(
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& comparison,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return compare_impl(MUDNN_POINTWISE_CMP_EQ, input, comparison, attributes);
  }

  std::shared_ptr<Tensor_attributes> cmp_ne(
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& comparison,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return compare_impl(MUDNN_POINTWISE_CMP_NEQ, input, comparison, attributes);
  }

  std::shared_ptr<Tensor_attributes> cmp_neq(
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& comparison,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return cmp_ne(input, comparison, attributes);
  }

  std::shared_ptr<Tensor_attributes> cmp_gt(
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& comparison,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return compare_impl(MUDNN_POINTWISE_CMP_GT, input, comparison, attributes);
  }

  std::shared_ptr<Tensor_attributes> cmp_ge(
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& comparison,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return compare_impl(MUDNN_POINTWISE_CMP_GE, input, comparison, attributes);
  }

  std::shared_ptr<Tensor_attributes> cmp_lt(
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& comparison,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return compare_impl(MUDNN_POINTWISE_CMP_LT, input, comparison, attributes);
  }

  std::shared_ptr<Tensor_attributes> cmp_le(
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& comparison,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return compare_impl(MUDNN_POINTWISE_CMP_LE, input, comparison, attributes);
  }

  std::shared_ptr<Tensor_attributes> eq(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return cmp_eq(a, b, attributes);
  }

  std::shared_ptr<Tensor_attributes> ne(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return cmp_ne(a, b, attributes);
  }

  std::shared_ptr<Tensor_attributes> gt(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return cmp_gt(a, b, attributes);
  }

  std::shared_ptr<Tensor_attributes> ge(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return cmp_ge(a, b, attributes);
  }

  std::shared_ptr<Tensor_attributes> lt(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return cmp_lt(a, b, attributes);
  }

  std::shared_ptr<Tensor_attributes> le(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return cmp_le(a, b, attributes);
  }

  std::shared_ptr<Tensor_attributes> logical_and(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return logical_impl(MUDNN_POINTWISE_LOGICAL_AND, a, b, attributes);
  }

  std::shared_ptr<Tensor_attributes> logical_or(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return logical_impl(MUDNN_POINTWISE_LOGICAL_OR, a, b, attributes);
  }

  std::shared_ptr<Tensor_attributes> logical_not(
      const std::shared_ptr<Tensor_attributes>& x,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    if (!x) {
      throw std::invalid_argument("logical_not requires input");
    }
    if (x->data_type_ != MUDNN_DATA_BOOLEAN) {
      throw std::invalid_argument("logical_not input must be boolean");
    }
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_LOGICAL_NOT)
        .set_output_data_type(MUDNN_DATA_BOOLEAN);
    return pointwise_impl(x, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> select(
      const std::shared_ptr<Tensor_attributes>& pred,
      const std::shared_ptr<Tensor_attributes>& true_value,
      const std::shared_ptr<Tensor_attributes>& false_value,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    return binary_select(true_value, false_value, pred, attributes);
  }

  std::shared_ptr<Tensor_attributes> binary_select(
      const std::shared_ptr<Tensor_attributes>& input0,
      const std::shared_ptr<Tensor_attributes>& input1,
      const std::shared_ptr<Tensor_attributes>& mask,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    if (!input0 || !input1 || !mask) {
      throw std::invalid_argument(
          "binary_select requires input0, input1, and mask");
    }
    if (mask->data_type_ != MUDNN_DATA_BOOLEAN &&
        mask->data_type_ != MUDNN_DATA_INT32) {
      throw std::invalid_argument(
          "binary_select mask must be boolean or int32");
    }
    (void)broadcast_dims(broadcast_dims(input0->dims_, input1->dims_),
                         mask->dims_);
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_BINARY_SELECT);
    attrs.set_output_data_type(input0->data_type_);
    return pointwise_impl(input0, input1, mask, attrs);
  }

  std::shared_ptr<Tensor_attributes> gen_index(
      const std::shared_ptr<Tensor_attributes>& input, int64_t axis,
      const Pointwise_attributes& attributes = Pointwise_attributes()) {
    if (!input) {
      throw std::invalid_argument("gen_index requires input");
    }
    if (axis < 0 || axis >= static_cast<int64_t>(input->dims_.size())) {
      throw std::invalid_argument("gen_index axis is out of range");
    }
    auto attrs = attributes;
    attrs.set_mode(MUDNN_POINTWISE_GEN_INDEX)
        .set_axis(axis)
        .set_output_data_type(MUDNN_DATA_INT32);
    return pointwise_impl(input, nullptr, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> matmul(
      const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Matmul_attributes& attributes = Matmul_attributes()) {
    if (!a || !b) {
      throw std::invalid_argument("matmul requires non-null inputs");
    }
    if (a->dims_.size() < 2 || b->dims_.size() < 2) {
      throw std::invalid_argument("matmul expects rank >= 2 tensors");
    }
    if (a->dims_.back() != b->dims_[b->dims_.size() - 2]) {
      throw std::invalid_argument("matmul K dimensions do not match");
    }
    auto y = std::make_shared<Tensor_attributes>();
    y->set_name(attributes.name_.empty()
                    ? ("matmul_" + std::to_string(nodes_.size()))
                    : attributes.name_);
    auto batch = broadcast_batch_dims(a->dims_, b->dims_, 2, "matmul");
    std::vector<int64_t> dims = batch;
    dims.push_back(a->dims_[a->dims_.size() - 2]);
    dims.push_back(b->dims_.back());
    y->set_dim(dims);
    y->set_stride(row_major_strides(dims));
    y->set_data_type(io_data_type_);
    y->set_is_virtual(true);
    register_tensor(y);
    Node node;
    node.kind = NodeKind::kMatmul;
    node.name = attributes.name_;
    node.inputs = {a, b};
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> reduction(
      mudnnReduceTensorOp_t mode, const std::shared_ptr<Tensor_attributes>& x,
      const std::vector<int64_t>& axes,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    if (!x) {
      throw std::invalid_argument("reduction requires a non-null input");
    }
    const auto normalized_axes =
        normalize_axes(axes, static_cast<int64_t>(x->dims_.size()));
    std::vector<int64_t> dims = x->dims_;
    for (auto axis : normalized_axes) {
      dims[static_cast<size_t>(axis)] = 1;
    }

    auto y = std::make_shared<Tensor_attributes>();
    y->set_name(attributes.name_.empty()
                    ? ("reduction_" + std::to_string(nodes_.size()))
                    : attributes.name_);
    y->set_dim(dims);
    y->set_stride(row_major_strides(dims));
    y->set_data_type(attributes.has_output_data_type_
                         ? attributes.output_data_type_
                         : x->data_type_);
    y->set_is_virtual(true);
    register_tensor(y);

    Node node;
    node.kind = NodeKind::kReduction;
    node.name = attributes.name_;
    node.inputs = {x};
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.reduction_mode = mode;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> reduction(
      ReductionMode_t mode, const std::shared_ptr<Tensor_attributes>& x,
      const std::vector<int64_t>& axes,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    return reduction(to_mudnn(mode), x, axes, attributes);
  }

  std::shared_ptr<Tensor_attributes> reduction(
      mudnnReduceTensorOp_t mode, const std::shared_ptr<Tensor_attributes>& x,
      int64_t axis,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    return reduction(mode, x, std::vector<int64_t>{axis}, attributes);
  }

  std::shared_ptr<Tensor_attributes> reduction(
      ReductionMode_t mode, const std::shared_ptr<Tensor_attributes>& x,
      int64_t axis,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    return reduction(to_mudnn(mode), x, axis, attributes);
  }

  std::shared_ptr<Tensor_attributes> reduction(
      const std::shared_ptr<Tensor_attributes>& x,
      const Reduction_attributes& attributes) {
    if (!x) {
      throw std::invalid_argument("reduction requires a non-null input");
    }
    std::vector<int64_t> axes;
    axes.reserve(x->dims_.size());
    for (int64_t i = 0; i < static_cast<int64_t>(x->dims_.size()); ++i) {
      axes.push_back(i);
    }
    return reduction(attributes.mode_, x, axes, attributes);
  }

  std::shared_ptr<Tensor_attributes> sum(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::vector<int64_t>& axes,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    return reduction(MUDNN_REDUCE_TENSOR_ADD, x, axes, attributes);
  }

  std::shared_ptr<Tensor_attributes> sum(
      const std::shared_ptr<Tensor_attributes>& x, int64_t axis,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    return reduction(MUDNN_REDUCE_TENSOR_ADD, x, axis, attributes);
  }

  std::shared_ptr<Tensor_attributes> mean(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::vector<int64_t>& axes,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    return reduction(MUDNN_REDUCE_TENSOR_AVG, x, axes, attributes);
  }

  std::shared_ptr<Tensor_attributes> mean(
      const std::shared_ptr<Tensor_attributes>& x, int64_t axis,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    return reduction(MUDNN_REDUCE_TENSOR_AVG, x, axis, attributes);
  }

  std::shared_ptr<Tensor_attributes> amax(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::vector<int64_t>& axes,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    return reduction(MUDNN_REDUCE_TENSOR_AMAX, x, axes, attributes);
  }

  std::shared_ptr<Tensor_attributes> amax(
      const std::shared_ptr<Tensor_attributes>& x, int64_t axis,
      const Reduction_attributes& attributes = Reduction_attributes()) {
    return reduction(MUDNN_REDUCE_TENSOR_AMAX, x, axis, attributes);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 2>
  genstats(const std::shared_ptr<Tensor_attributes>& x,
           const std::vector<int64_t>& axes,
           const Genstats_attributes& attributes = Genstats_attributes()) {
    if (!x) {
      throw std::invalid_argument("genstats requires a non-null input");
    }
    const auto normalized_axes =
        normalize_axes(axes, static_cast<int64_t>(x->dims_.size()));
    std::vector<int64_t> dims = x->dims_;
    for (auto axis : normalized_axes) {
      dims[static_cast<size_t>(axis)] = 1;
    }
    const auto out_type = attributes.has_output_data_type_
        ? attributes.output_data_type_
        : x->data_type_;
    if (out_type != x->data_type_) {
      throw std::invalid_argument("genstats output data type must match input");
    }
    const auto base_name = attributes.name_.empty()
        ? ("genstats_" + std::to_string(nodes_.size()))
        : attributes.name_;

    auto sum = std::make_shared<Tensor_attributes>();
    sum->set_name(base_name + "_sum");
    sum->set_dim(dims);
    sum->set_stride(row_major_strides(dims));
    sum->set_data_type(out_type);
    sum->set_is_virtual(true);
    register_tensor(sum);

    auto sqsum = std::make_shared<Tensor_attributes>();
    sqsum->set_name(base_name + "_sqsum");
    sqsum->set_dim(dims);
    sqsum->set_stride(row_major_strides(dims));
    sqsum->set_data_type(out_type);
    sqsum->set_is_virtual(true);
    register_tensor(sqsum);

    Node node;
    node.kind = NodeKind::kGenstats;
    node.name = base_name;
    node.inputs = {x};
    node.output = sum;
    node.stats = sqsum;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.genstats_mode = attributes.mode_;
    nodes_.push_back(std::move(node));
    return {sum, sqsum};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 2>
  genstats(const std::shared_ptr<Tensor_attributes>& x, int64_t axis,
           const Genstats_attributes& attributes = Genstats_attributes()) {
    return genstats(x, std::vector<int64_t>{axis}, attributes);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 2>
  genstats(const std::shared_ptr<Tensor_attributes>& x,
           const Genstats_attributes& attributes) {
    if (!x) {
      throw std::invalid_argument("genstats requires a non-null input");
    }
    if (x->dims_.size() == 4) {
      return genstats(x, std::vector<int64_t>{0, 2, 3}, attributes);
    }
    std::vector<int64_t> axes;
    axes.reserve(x->dims_.size());
    for (int64_t axis = 0; axis < static_cast<int64_t>(x->dims_.size());
         ++axis) {
      axes.push_back(axis);
    }
    return genstats(x, axes, attributes);
  }

  std::shared_ptr<Tensor_attributes> rng(
      const std::vector<int64_t>& shape,
      const Rng_attributes& attributes = Rng_attributes(),
      const std::shared_ptr<Tensor_attributes>& offset = nullptr) {
    num_elements(shape);
    if (attributes.distribution_ == MUDNN_RNG_DISTRIBUTION_UNIFORM &&
        !(attributes.uniform_minimum_ < attributes.uniform_maximum_)) {
      throw std::invalid_argument(
          "uniform RNG minimum must be less than maximum");
    }
    if (attributes.distribution_ == MUDNN_RNG_DISTRIBUTION_NORMAL &&
        attributes.normal_stddev_ <= 0.0) {
      throw std::invalid_argument(
          "normal RNG standard deviation must be positive");
    }
    if (attributes.distribution_ == MUDNN_RNG_DISTRIBUTION_BERNOULLI &&
        (attributes.bernoulli_probability_ < 0.0 ||
         attributes.bernoulli_probability_ > 1.0)) {
      throw std::invalid_argument(
          "bernoulli RNG probability must be in [0, 1]");
    }
    if (offset &&
        (offset->data_type_ != MUDNN_DATA_INT64 ||
         offset->dims_ != std::vector<int64_t>{1})) {
      throw std::invalid_argument("rng offset must be an int64 scalar tensor");
    }

    auto y = std::make_shared<Tensor_attributes>();
    y->set_name(attributes.name_.empty()
                    ? ("rng_" + std::to_string(nodes_.size()))
                    : attributes.name_);
    y->set_dim(shape);
    y->set_stride(row_major_strides(shape));
    y->set_data_type(attributes.has_output_data_type_
                         ? attributes.output_data_type_
                         : io_data_type_);
    y->set_is_virtual(true);
    register_tensor(y);

    Node node;
    node.kind = NodeKind::kRng;
    node.name = attributes.name_;
    if (offset) {
      node.inputs = {offset};
      node.rng_has_offset = true;
    }
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = compute_data_type_;
    node.rng_distribution = attributes.distribution_;
    node.rng_seed = attributes.seed_;
    node.rng_normal_mean = attributes.normal_mean_;
    node.rng_normal_stddev = attributes.normal_stddev_;
    node.rng_uniform_minimum = attributes.uniform_minimum_;
    node.rng_uniform_maximum = attributes.uniform_maximum_;
    node.rng_bernoulli_probability = attributes.bernoulli_probability_;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::array<std::shared_ptr<Tensor_attributes>, 2>
  sdpa(const std::shared_ptr<Tensor_attributes>& q,
       const std::shared_ptr<Tensor_attributes>& k,
       const std::shared_ptr<Tensor_attributes>& v,
       const Sdpa_attributes& attributes = Sdpa_attributes(),
       const std::shared_ptr<Tensor_attributes>& bias = nullptr,
       const std::shared_ptr<Tensor_attributes>& mask = nullptr) {
    if (!q || !k || !v) {
      throw std::invalid_argument("sdpa requires non-null q, k, and v inputs");
    }
    if (q->dims_.size() != 4 || k->dims_.size() != 4 || v->dims_.size() != 4) {
      throw std::invalid_argument("sdpa expects rank-4 q, k, and v tensors");
    }
    if (q->dims_[0] != k->dims_[0] || q->dims_[0] != v->dims_[0] ||
        k->dims_[2] != v->dims_[2] || q->dims_[3] != k->dims_[3]) {
      throw std::invalid_argument("sdpa q/k/v dimensions are incompatible");
    }
    auto o = std::make_shared<Tensor_attributes>();
    o->set_name(attributes.name_.empty()
                    ? ("sdpa_" + std::to_string(nodes_.size()))
                    : attributes.name_);
    std::vector<int64_t> dims = {q->dims_[0], q->dims_[1], q->dims_[2],
                                 v->dims_[3]};
    o->set_dim(dims);
    o->set_stride(row_major_strides(dims));
    o->set_data_type(io_data_type_);
    o->set_is_virtual(true);
    register_tensor(o);

    std::shared_ptr<Tensor_attributes> stats;
    if (!attributes.is_inference_) {
      stats = std::make_shared<Tensor_attributes>();
      stats->set_name(o->name_ + "_stats");
      stats->set_dim({q->dims_[0], q->dims_[1], q->dims_[2], 1});
      stats->set_stride(row_major_strides(stats->dims_));
      stats->set_data_type(MUDNN_DATA_FLOAT);
      stats->set_is_virtual(true);
      register_tensor(stats);
    }

    const auto effective_bias = bias ? bias : attributes.bias_;
    const auto effective_mask = mask ? mask : attributes.block_mask_;
    std::vector<std::shared_ptr<Tensor_attributes>> inputs = {q, k, v};
    if (attributes.attn_scale_tensor_) {
      inputs.push_back(attributes.attn_scale_tensor_);
    } else if (attributes.has_attn_scale_) {
      inputs.push_back(
          make_scalar_by_value(o->name_ + "_scale", attributes.attn_scale_));
    }
    if (effective_bias) {
      inputs.push_back(effective_bias);
    }
    if (effective_mask) {
      inputs.push_back(effective_mask);
    }
    if (attributes.seq_len_q_) {
      check_sdpa_seq_len("sdpa seq_len_q", attributes.seq_len_q_, q->dims_[0]);
      inputs.push_back(attributes.seq_len_q_);
    }
    if (attributes.seq_len_kv_) {
      check_sdpa_seq_len("sdpa seq_len_kv", attributes.seq_len_kv_,
                         q->dims_[0]);
      inputs.push_back(attributes.seq_len_kv_);
    }
    Node node;
    node.kind = NodeKind::kSdpa;
    node.name = attributes.name_;
    node.inputs = std::move(inputs);
    node.output = o;
    node.stats = stats;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.sdpa_scale = attributes.attn_scale_;
    node.sdpa_has_scale = attributes.has_attn_scale_;
    node.sdpa_causal_mask = attributes.causal_mask_;
    node.sdpa_is_inference = attributes.is_inference_;
    node.sdpa_use_padding_mask = attributes.use_padding_mask_;
    node.sdpa_diagonal_alignment = attributes.diagonal_alignment_;
    node.sdpa_diagonal_band_left_bound = attributes.diagonal_band_left_bound_;
    node.sdpa_diagonal_band_right_bound = attributes.diagonal_band_right_bound_;
    node.sdpa_has_diagonal_band_left_bound =
        attributes.has_diagonal_band_left_bound_;
    node.sdpa_has_diagonal_band_right_bound =
        attributes.has_diagonal_band_right_bound_;
    node.sdpa_has_bias = static_cast<bool>(effective_bias);
    node.sdpa_has_mask = static_cast<bool>(effective_mask);
    node.sdpa_has_seq_len_q = static_cast<bool>(attributes.seq_len_q_);
    node.sdpa_has_seq_len_kv = static_cast<bool>(attributes.seq_len_kv_);
    if (attributes.has_diagonal_band_left_bound_ ||
        (attributes.has_diagonal_band_right_bound_ &&
         !(attributes.diagonal_alignment_ == DiagonalAlignment_t::TOP_LEFT &&
           attributes.diagonal_band_right_bound_ == 0))) {
      unsupported_reason_ =
          "SDPA diagonal-band native lowering is not supported";
    }
    nodes_.push_back(std::move(node));
    return {o, stats};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  sdpa_backward(const std::shared_ptr<Tensor_attributes>& q,
                const std::shared_ptr<Tensor_attributes>& k,
                const std::shared_ptr<Tensor_attributes>& v,
                const std::shared_ptr<Tensor_attributes>& o,
                const std::shared_ptr<Tensor_attributes>& d_o,
                const std::shared_ptr<Tensor_attributes>& stats,
                const Sdpa_attributes& attributes = Sdpa_attributes(),
                const std::shared_ptr<Tensor_attributes>& bias = nullptr,
                const std::shared_ptr<Tensor_attributes>& d_bias = nullptr,
                const std::shared_ptr<Tensor_attributes>& mask = nullptr) {
    if (!q || !k || !v || !o || !d_o || !stats) {
      throw std::invalid_argument(
          "sdpa_backward requires q, k, v, o, d_o, and stats");
    }
    check_sdpa_backward_tensors(q, k, v, o, d_o, stats);
    const auto effective_bias = bias ? bias : attributes.bias_;
    const auto effective_d_bias = d_bias ? d_bias : attributes.d_bias_;
    if (effective_bias) {
      check_sdpa_attention_tensor("sdpa_backward bias", effective_bias, q, k,
                                  q->data_type_, true);
    }
    if (effective_d_bias) {
      check_sdpa_attention_tensor("sdpa_backward d_bias", effective_d_bias, q,
                                  k, q->data_type_, true);
    }
    if (mask) {
      check_sdpa_attention_tensor("sdpa_backward mask", mask, q, k,
                                  q->data_type_, false);
    }

    const auto base = attributes.name_.empty()
        ? ("sdpa_bwd_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto d_q = make_virtual_tensor(base + "_dq", q->dims_, q->data_type_);
    auto d_k = make_virtual_tensor(base + "_dk", k->dims_, k->data_type_);
    auto d_v = make_virtual_tensor(base + "_dv", v->dims_, v->data_type_);

    std::vector<std::shared_ptr<Tensor_attributes>> inputs = {q, k,   v,
                                                              o, d_o, stats};
    if (attributes.attn_scale_tensor_) {
      inputs.push_back(attributes.attn_scale_tensor_);
    } else if (attributes.has_attn_scale_) {
      inputs.push_back(
          make_scalar_by_value(base + "_scale", attributes.attn_scale_));
    }
    if (effective_bias) {
      inputs.push_back(effective_bias);
    }
    if (mask) {
      inputs.push_back(mask);
    }
    if (attributes.seq_len_q_) {
      check_sdpa_seq_len("sdpa_backward seq_len_q", attributes.seq_len_q_,
                         q->dims_[0]);
      inputs.push_back(attributes.seq_len_q_);
    }
    if (attributes.seq_len_kv_) {
      check_sdpa_seq_len("sdpa_backward seq_len_kv", attributes.seq_len_kv_,
                         q->dims_[0]);
      inputs.push_back(attributes.seq_len_kv_);
    }

    Node node;
    node.kind = NodeKind::kSdpaBwd;
    node.name = base;
    node.inputs = std::move(inputs);
    node.output = d_q;
    node.stats = d_k;
    node.aux_outputs = {d_v};
    if (effective_d_bias) {
      node.aux_outputs.push_back(effective_d_bias);
    }
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.sdpa_scale = attributes.attn_scale_;
    node.sdpa_has_scale = attributes.has_attn_scale_;
    node.sdpa_causal_mask = attributes.causal_mask_;
    node.sdpa_use_padding_mask = attributes.use_padding_mask_ ||
        (mask && mask->dims_ == std::vector<int64_t>{q->dims_[0], k->dims_[2]});
    node.sdpa_diagonal_alignment = attributes.diagonal_alignment_;
    node.sdpa_diagonal_band_left_bound = attributes.diagonal_band_left_bound_;
    node.sdpa_diagonal_band_right_bound = attributes.diagonal_band_right_bound_;
    node.sdpa_has_diagonal_band_left_bound =
        attributes.has_diagonal_band_left_bound_;
    node.sdpa_has_diagonal_band_right_bound =
        attributes.has_diagonal_band_right_bound_;
    node.sdpa_has_bias = static_cast<bool>(effective_bias);
    node.sdpa_has_mask = static_cast<bool>(mask);
    node.sdpa_has_seq_len_q = static_cast<bool>(attributes.seq_len_q_);
    node.sdpa_has_seq_len_kv = static_cast<bool>(attributes.seq_len_kv_);
    node.sdpa_bwd_has_d_bias = static_cast<bool>(effective_d_bias);
    node.sdpa_bwd_deterministic = attributes.deterministic_;
    if (attributes.has_diagonal_band_left_bound_ ||
        (attributes.has_diagonal_band_right_bound_ &&
         !(attributes.diagonal_alignment_ == DiagonalAlignment_t::TOP_LEFT &&
           attributes.diagonal_band_right_bound_ == 0))) {
      unsupported_reason_ =
          "SDPA backward diagonal-band native lowering is not supported";
    }
    nodes_.push_back(std::move(node));
    return {d_q, d_k, d_v};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 4>
  sdpa_fp8(const std::shared_ptr<Tensor_attributes>& q,
           const std::shared_ptr<Tensor_attributes>& k,
           const std::shared_ptr<Tensor_attributes>& v,
           const std::shared_ptr<Tensor_attributes>& descale_q,
           const std::shared_ptr<Tensor_attributes>& descale_k,
           const std::shared_ptr<Tensor_attributes>& descale_v,
           const std::shared_ptr<Tensor_attributes>& descale_s,
           const std::shared_ptr<Tensor_attributes>& scale_s,
           const std::shared_ptr<Tensor_attributes>& scale_o,
           const Sdpa_fp8_attributes& attributes = Sdpa_fp8_attributes()) {
    throw mudnnException(
        MUDNN_STATUS_NOT_SUPPORTED,
        "sdpa_fp8 is not executable because muDNN has no FP8 SDPA backend "
        "descriptor");
    if (!q || !k || !v || !descale_q || !descale_k || !descale_v ||
        !descale_s || !scale_s || !scale_o) {
      throw std::invalid_argument("sdpa_fp8 requires non-null tensor inputs");
    }
    const auto base = attributes.name_.empty()
        ? ("sdpa_fp8_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto o = make_virtual_tensor(base + "_o", q->dims_, q->data_type_);
    std::shared_ptr<Tensor_attributes> stats;
    if (attributes.generate_stats_) {
      stats = make_virtual_tensor(base + "_stats",
                                  {q->dims_[0], q->dims_[1], q->dims_[2], 1},
                                  MUDNN_DATA_FLOAT);
    }
    auto amax_s =
        make_virtual_tensor(base + "_amax_s", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);
    auto amax_o =
        make_virtual_tensor(base + "_amax_o", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);
    unsupported_reason_ =
        "mudnn_frontend does not support executable SDPA FP8 forward graphs";
    return {o, stats, amax_s, amax_o};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  sdpa_fp8(const std::shared_ptr<Tensor_attributes>& q,
           const std::shared_ptr<Tensor_attributes>& k,
           const std::shared_ptr<Tensor_attributes>& v,
           const std::shared_ptr<Tensor_attributes>& sf_q,
           const std::shared_ptr<Tensor_attributes>& sf_k,
           const std::shared_ptr<Tensor_attributes>& sf_v,
           const Sdpa_fp8_attributes& attributes = Sdpa_fp8_attributes()) {
    throw mudnnException(
        MUDNN_STATUS_NOT_SUPPORTED,
        "sdpa_mxfp8 is not executable because muDNN has no MXFP8 SDPA "
        "backend descriptor");
    if (!q || !k || !v || !sf_q || !sf_k || !sf_v) {
      throw std::invalid_argument("sdpa_fp8 requires non-null tensor inputs");
    }
    const auto base = attributes.name_.empty()
        ? ("sdpa_mxfp8_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto o = make_virtual_tensor(base + "_o", q->dims_, q->data_type_);
    std::shared_ptr<Tensor_attributes> stats;
    if (attributes.generate_stats_) {
      stats = make_virtual_tensor(base + "_stats",
                                  {q->dims_[0], q->dims_[1], q->dims_[2], 1},
                                  MUDNN_DATA_FLOAT);
    }
    auto amax_o =
        make_virtual_tensor(base + "_amax_o", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);
    unsupported_reason_ =
        "mudnn_frontend does not support executable SDPA MXFP8 forward graphs";
    return {o, stats, amax_o};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 7>
  sdpa_fp8_backward(const std::shared_ptr<Tensor_attributes>& q,
                    const std::shared_ptr<Tensor_attributes>& k,
                    const std::shared_ptr<Tensor_attributes>& v,
                    const std::shared_ptr<Tensor_attributes>& o,
                    const std::shared_ptr<Tensor_attributes>& d_o,
                    const std::shared_ptr<Tensor_attributes>& stats,
                    const std::shared_ptr<Tensor_attributes>& descale_q,
                    const std::shared_ptr<Tensor_attributes>& descale_k,
                    const std::shared_ptr<Tensor_attributes>& descale_v,
                    const std::shared_ptr<Tensor_attributes>& descale_o,
                    const std::shared_ptr<Tensor_attributes>& descale_d_o,
                    const std::shared_ptr<Tensor_attributes>& descale_s,
                    const std::shared_ptr<Tensor_attributes>& descale_d_p,
                    const std::shared_ptr<Tensor_attributes>& scale_s,
                    const std::shared_ptr<Tensor_attributes>& scale_d_q,
                    const std::shared_ptr<Tensor_attributes>& scale_d_k,
                    const std::shared_ptr<Tensor_attributes>& scale_d_v,
                    const std::shared_ptr<Tensor_attributes>& scale_d_p,
                    const Sdpa_fp8_backward_attributes& attributes =
                        Sdpa_fp8_backward_attributes()) {
    throw mudnnException(
        MUDNN_STATUS_NOT_SUPPORTED,
        "sdpa_fp8_backward is not executable because muDNN has no FP8 SDPA "
        "backward backend descriptor");
    if (!q || !k || !v || !o || !d_o || !stats || !descale_q || !descale_k ||
        !descale_v || !descale_o || !descale_d_o || !descale_s ||
        !descale_d_p || !scale_s || !scale_d_q || !scale_d_k || !scale_d_v ||
        !scale_d_p) {
      throw std::invalid_argument(
          "sdpa_fp8_backward requires non-null tensor inputs");
    }

    const auto base = attributes.name_.empty()
        ? ("sdpa_fp8_bwd_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto d_q = make_virtual_tensor(base + "_dq", q->dims_, q->data_type_);
    auto d_k = make_virtual_tensor(base + "_dk", k->dims_, k->data_type_);
    auto d_v = make_virtual_tensor(base + "_dv", v->dims_, v->data_type_);
    auto amax_d_q =
        make_virtual_tensor(base + "_amax_dq", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);
    auto amax_d_k =
        make_virtual_tensor(base + "_amax_dk", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);
    auto amax_d_v =
        make_virtual_tensor(base + "_amax_dv", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);
    auto amax_d_p =
        make_virtual_tensor(base + "_amax_dp", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);

    unsupported_reason_ =
        "mudnn_frontend does not support executable SDPA FP8 backward graphs";
    return {d_q, d_k, d_v, amax_d_q, amax_d_k, amax_d_v, amax_d_p};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 6>
  sdpa_fp8_backward(const std::shared_ptr<Tensor_attributes>& q,
                    const std::shared_ptr<Tensor_attributes>& q_t,
                    const std::shared_ptr<Tensor_attributes>& k,
                    const std::shared_ptr<Tensor_attributes>& k_t,
                    const std::shared_ptr<Tensor_attributes>& v,
                    const std::shared_ptr<Tensor_attributes>& o,
                    const std::shared_ptr<Tensor_attributes>& d_o_f16,
                    const std::shared_ptr<Tensor_attributes>& d_o,
                    const std::shared_ptr<Tensor_attributes>& d_o_t,
                    const std::shared_ptr<Tensor_attributes>& stats,
                    const std::shared_ptr<Tensor_attributes>& sf_q,
                    const std::shared_ptr<Tensor_attributes>& sf_q_t,
                    const std::shared_ptr<Tensor_attributes>& sf_k,
                    const std::shared_ptr<Tensor_attributes>& sf_k_t,
                    const std::shared_ptr<Tensor_attributes>& sf_v,
                    const std::shared_ptr<Tensor_attributes>& sf_d_o,
                    const std::shared_ptr<Tensor_attributes>& sf_d_o_t,
                    const Sdpa_fp8_backward_attributes& attributes =
                        Sdpa_fp8_backward_attributes()) {
    throw mudnnException(
        MUDNN_STATUS_NOT_SUPPORTED,
        "sdpa_mxfp8_backward is not executable because muDNN has no MXFP8 "
        "SDPA backward backend descriptor");
    if (!q || !q_t || !k || !k_t || !v || !o || !d_o_f16 || !d_o || !d_o_t ||
        !stats || !sf_q || !sf_q_t || !sf_k || !sf_k_t || !sf_v || !sf_d_o ||
        !sf_d_o_t) {
      throw std::invalid_argument(
          "sdpa_fp8_backward requires non-null MXFP8 tensor inputs");
    }

    const auto base = attributes.name_.empty()
        ? ("sdpa_mxfp8_bwd_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto d_q = make_virtual_tensor(base + "_dq", q->dims_, MUDNN_DATA_FLOAT);
    auto d_k = make_virtual_tensor(base + "_dk", k->dims_, MUDNN_DATA_FLOAT);
    auto d_v = make_virtual_tensor(base + "_dv", v->dims_, MUDNN_DATA_FLOAT);
    auto amax_d_q =
        make_virtual_tensor(base + "_amax_dq", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);
    auto amax_d_k =
        make_virtual_tensor(base + "_amax_dk", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);
    auto amax_d_v =
        make_virtual_tensor(base + "_amax_dv", {1, 1, 1, 1}, MUDNN_DATA_FLOAT);

    unsupported_reason_ =
        "mudnn_frontend does not support executable SDPA MXFP8 backward graphs";
    return {d_q, d_k, d_v, amax_d_q, amax_d_k, amax_d_v};
  }

  std::shared_ptr<Tensor_attributes> concat(
      const std::vector<std::shared_ptr<Tensor_attributes>>& inputs,
      int64_t axis, const std::string& name = "") {
    if (inputs.empty()) {
      throw std::invalid_argument("concat requires at least one input tensor");
    }
    if (!inputs[0]) {
      throw std::invalid_argument("concat inputs must be non-null");
    }
    const auto rank = static_cast<int64_t>(inputs[0]->dims_.size());
    if (rank <= 0) {
      throw std::invalid_argument(
          "concat inputs must have rank greater than 0");
    }
    const int64_t normalized_axis = normalize_axis(axis, rank);
    std::vector<int64_t> dims = inputs[0]->dims_;
    dims[static_cast<size_t>(normalized_axis)] = 0;
    for (const auto& input : inputs) {
      if (!input) {
        throw std::invalid_argument("concat inputs must be non-null");
      }
      if (static_cast<int64_t>(input->dims_.size()) != rank) {
        throw std::invalid_argument("concat inputs must have the same rank");
      }
      if (input->data_type_ != inputs[0]->data_type_) {
        throw std::invalid_argument(
            "concat inputs must have the same data type");
      }
      for (int64_t i = 0; i < rank; ++i) {
        const auto index = static_cast<size_t>(i);
        if (i == normalized_axis) {
          dims[index] += input->dims_[index];
        } else if (input->dims_[index] != inputs[0]->dims_[index]) {
          throw std::invalid_argument("concat non-axis dimensions must match");
        }
      }
    }

    auto y = std::make_shared<Tensor_attributes>();
    y->set_name(name.empty() ? ("concat_" + std::to_string(nodes_.size()))
                             : name);
    y->set_dim(dims);
    y->set_stride(row_major_strides(dims));
    y->set_data_type(inputs[0]->data_type_);
    y->set_is_virtual(true);
    register_tensor(y);

    Node node;
    node.kind = NodeKind::kConcat;
    node.name = name;
    node.inputs = inputs;
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = compute_data_type_;
    node.concat_axis = normalized_axis;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> concatenate(
      const std::vector<std::shared_ptr<Tensor_attributes>>& inputs,
      const Concatenate_attributes& attributes = Concatenate_attributes()) {
    return concat(inputs, attributes.axis_, attributes.name_);
  }

  std::shared_ptr<Tensor_attributes> slice(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::vector<std::pair<int64_t, int64_t>>& slices,
      const std::vector<int64_t>& slice_strides = {},
      const std::string& name = "") {
    if (!x) {
      throw std::invalid_argument("slice requires a non-null input");
    }
    const auto rank = x->dims_.size();
    if (rank == 0 || slices.size() != rank) {
      throw std::invalid_argument("slice rank must match input rank");
    }
    std::vector<int64_t> starts;
    std::vector<int64_t> limits;
    std::vector<int64_t> strides(rank, 1);
    std::vector<int64_t> dims;
    std::vector<int64_t> output_strides;
    starts.reserve(rank);
    limits.reserve(rank);
    dims.reserve(rank);
    output_strides.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
      if (i < slice_strides.size()) {
        strides[i] = slice_strides[i];
      }
      const int64_t start = slices[i].first;
      const int64_t limit = slices[i].second;
      const int64_t stride = strides[i];
      if (start < 0 || limit < start || limit > x->dims_[i] || stride <= 0) {
        throw std::invalid_argument("invalid slice bounds");
      }
      const int64_t dim = (limit - start + stride - 1) / stride;
      if (dim <= 0) {
        throw std::invalid_argument("slice output dimensions must be positive");
      }
      starts.push_back(start);
      limits.push_back(limit);
      dims.push_back(dim);
      output_strides.push_back(x->strides_[i] * stride);
    }

    auto y = std::make_shared<Tensor_attributes>();
    y->set_name(name.empty() ? ("slice_" + std::to_string(nodes_.size()))
                             : name);
    y->set_dim(dims);
    y->set_stride(output_strides);
    y->set_data_type(x->data_type_);
    y->set_is_virtual(true);
    register_tensor(y);

    Node node;
    node.kind = NodeKind::kSlice;
    node.name = name;
    node.inputs = {x};
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = compute_data_type_;
    node.param0 = std::move(starts);
    node.param1 = std::move(limits);
    node.param2 = std::move(strides);
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> slice(
      const std::shared_ptr<Tensor_attributes>& x,
      const Slice_attributes& attributes) {
    return slice(x, attributes.slices_, attributes.strides_, attributes.name_);
  }

  std::shared_ptr<Tensor_attributes> reshape(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::vector<int64_t>& shape, const std::string& name = "") {
    if (!x) {
      throw std::invalid_argument("reshape requires a non-null input");
    }
    if (num_elements(x->dims_) != num_elements(shape)) {
      throw std::invalid_argument(
          "reshape output shape must preserve element count");
    }

    auto y = std::make_shared<Tensor_attributes>();
    y->set_name(name.empty() ? ("reshape_" + std::to_string(nodes_.size()))
                             : name);
    y->set_dim(shape);
    y->set_stride(row_major_strides(shape));
    y->set_data_type(x->data_type_);
    y->set_is_virtual(true);
    register_tensor(y);

    Node node;
    node.kind = NodeKind::kReshape;
    node.name = name;
    node.inputs = {x};
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = compute_data_type_;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> reshape(
      const std::shared_ptr<Tensor_attributes>& x,
      const Reshape_attributes& attributes) {
    if (!x) {
      throw std::invalid_argument("reshape requires a non-null input");
    }
    auto y = std::make_shared<Tensor_attributes>();
    y->set_name(attributes.name_.empty()
                    ? ("reshape_" + std::to_string(nodes_.size()))
                    : attributes.name_);
    y->set_dim(x->dims_);
    y->set_stride(x->strides_.empty() ? row_major_strides(x->dims_)
                                      : x->strides_);
    y->set_data_type(x->data_type_);
    y->set_is_virtual(true);
    register_tensor(y);

    Node node;
    node.kind = NodeKind::kReshape;
    node.name = attributes.name_;
    node.inputs = {x};
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = compute_data_type_;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> transpose(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::vector<int64_t>& permutation, const std::string& name = "") {
    if (!x) {
      throw std::invalid_argument("transpose requires a non-null input");
    }
    const auto rank = x->dims_.size();
    if (rank == 0 || permutation.size() != rank) {
      throw std::invalid_argument(
          "transpose permutation rank must match input rank");
    }
    std::vector<bool> seen(rank, false);
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    dims.reserve(rank);
    strides.reserve(rank);
    for (const auto dim : permutation) {
      if (dim < 0 || static_cast<size_t>(dim) >= rank) {
        throw std::invalid_argument("transpose permutation is out of range");
      }
      const auto index = static_cast<size_t>(dim);
      if (seen[index]) {
        throw std::invalid_argument(
            "transpose permutation contains duplicates");
      }
      seen[index] = true;
      dims.push_back(x->dims_[index]);
      strides.push_back(x->strides_[index]);
    }

    auto y = std::make_shared<Tensor_attributes>();
    y->set_name(name.empty() ? ("transpose_" + std::to_string(nodes_.size()))
                             : name);
    y->set_dim(dims);
    y->set_stride(strides);
    y->set_data_type(x->data_type_);
    y->set_is_virtual(true);
    register_tensor(y);

    Node node;
    node.kind = NodeKind::kTranspose;
    node.name = name;
    node.inputs = {x};
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = compute_data_type_;
    node.param0 = permutation;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> transpose(
      const std::shared_ptr<Tensor_attributes>& x,
      const Transpose_attributes& attributes) {
    return transpose(x, attributes.permutation_, attributes.name_);
  }

  std::shared_ptr<Tensor_attributes> conv_fprop(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& w,
      const Conv_attributes& attributes = Conv_attributes()) {
    if (!x || !w) {
      throw std::invalid_argument("conv_fprop requires x and w");
    }
    if (x->dims_.empty() || w->dims_.empty()) {
      // Upstream cudnn-frontend semantics: missing tensor attributes are
      // reported by graph.validate() (ATTRIBUTE_NOT_SET), not at node
      // creation time. Defer all shape checks by recording the node with
      // unresolved shapes.
      auto y = std::make_shared<Tensor_attributes>();
      y->set_name(attributes.name_.empty()
                      ? ("conv_fprop_" + std::to_string(nodes_.size()))
                      : attributes.name_);
      y->set_data_type(attributes.has_output_data_type_
                           ? attributes.output_data_type_
                           : io_data_type_);
      y->set_is_virtual(true);
      register_tensor(y);

      Node node;
      node.kind = NodeKind::kConvFprop;
      node.name = y->name_;
      node.inputs = {x, w};
      node.output = y;
      node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
      node.compute_data_type = attributes.compute_data_type_;
      node.conv_mode = attributes.mode_;
      node.conv_group_count = attributes.has_group_count_
          ? attributes.group_count_
          : 1;
      node.param0 = attributes.pre_padding_.empty() ? attributes.padding_
                                                    : attributes.pre_padding_;
      node.param1 = attributes.stride_;
      node.param2 = attributes.dilation_;
      node.param3 = attributes.post_padding_;
      node.alpha = attributes.alpha_;
      node.beta = attributes.beta_;
      nodes_.push_back(std::move(node));
      return y;
    }
    if (x->dims_.size() != w->dims_.size() || x->dims_.size() < 3 ||
        x->dims_.size() > 5) {
      throw std::invalid_argument(
          "conv_fprop expects rank-3 to rank-5 NC... tensors");
    }
    if (w->dims_[1] <= 0 || x->dims_[1] % w->dims_[1] != 0) {
      throw std::invalid_argument(
          "conv_fprop input and weight channel dimensions differ");
    }
    const int64_t inferred_group_count = x->dims_[1] / w->dims_[1];
    const int64_t group_count = attributes.has_group_count_
        ? attributes.group_count_
        : inferred_group_count;
    if (group_count <= 0 || group_count != inferred_group_count ||
        w->dims_[0] % group_count != 0) {
      throw std::invalid_argument(
          "conv_fprop invalid grouped convolution channel dimensions");
    }
    const int64_t spatial_rank = static_cast<int64_t>(x->dims_.size()) - 2;
    const auto pre_padding = normalize_conv_param(
        attributes.pre_padding_, "pre_padding", false, spatial_rank);
    const auto post_padding = normalize_conv_param(
        attributes.post_padding_, "post_padding", false, spatial_rank);
    const auto stride =
        normalize_conv_param(attributes.stride_, "stride", true, spatial_rank);
    const auto dilation = normalize_conv_param(attributes.dilation_, "dilation",
                                               true, spatial_rank);
    const auto spatial = conv_output_spatial(
        std::vector<int64_t>(x->dims_.begin() + 2, x->dims_.end()),
        std::vector<int64_t>(w->dims_.begin() + 2, w->dims_.end()), pre_padding,
        post_padding, stride, dilation);
    if (std::any_of(spatial.begin(), spatial.end(),
                    [](int64_t value) { return value <= 0; })) {
      throw std::invalid_argument(
          "conv_fprop output dimensions must be positive");
    }
    std::vector<int64_t> dims = {x->dims_[0], w->dims_[0]};
    dims.insert(dims.end(), spatial.begin(), spatial.end());
    auto y = make_virtual_tensor(
        attributes.name_.empty()
            ? ("conv_fprop_" + std::to_string(nodes_.size()))
            : attributes.name_,
        dims,
        attributes.has_output_data_type_ ? attributes.output_data_type_
                                         : io_data_type_);

    Node node;
    node.kind = NodeKind::kConvFprop;
    node.name = y->name_;
    node.inputs = {x, w};
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.conv_mode = attributes.mode_;
    node.conv_group_count = group_count;
    node.param0 = pre_padding;
    node.param1 = stride;
    node.param2 = dilation;
    node.param3 = post_padding;
    node.alpha = attributes.alpha_;
    node.beta = attributes.beta_;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> conv_bwd_data(
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::shared_ptr<Tensor_attributes>& w,
      const std::vector<int64_t>& x_shape,
      const Conv_attributes& attributes = Conv_attributes()) {
    if (!dy || !w) {
      throw std::invalid_argument("conv_bwd_data requires dy and w");
    }
    if (dy->dims_.size() != 4 || w->dims_.size() != 4) {
      throw std::invalid_argument("conv_bwd_data expects rank-4 NCHW tensors");
    }
    if (x_shape.size() != 4) {
      throw std::invalid_argument("conv_bwd_data x_shape must be rank-4");
    }
    if (dy->dims_[0] != x_shape[0]) {
      throw std::invalid_argument("conv_bwd_data batch dimensions differ");
    }
    if (w->dims_[1] <= 0 || x_shape[1] % w->dims_[1] != 0) {
      throw std::invalid_argument("conv_bwd_data channel dimensions differ");
    }
    const int64_t inferred_group_count = x_shape[1] / w->dims_[1];
    const int64_t group_count = attributes.has_group_count_
        ? attributes.group_count_
        : inferred_group_count;
    if (group_count <= 0 || group_count != inferred_group_count ||
        dy->dims_[1] != w->dims_[0] || w->dims_[0] % group_count != 0) {
      throw std::invalid_argument("conv_bwd_data channel dimensions differ");
    }
    const auto pre_padding =
        normalize_conv_param(attributes.pre_padding_, "pre_padding", false);
    const auto post_padding =
        normalize_conv_param(attributes.post_padding_, "post_padding", false);
    const auto stride =
        normalize_conv_param(attributes.stride_, "stride", true);
    const auto dilation =
        normalize_conv_param(attributes.dilation_, "dilation", true);
    const auto spatial = conv_output_spatial(
        {x_shape[2], x_shape[3]}, {w->dims_[2], w->dims_[3]}, pre_padding,
        post_padding, stride, dilation);
    if (dy->dims_[2] != spatial[0] || dy->dims_[3] != spatial[1]) {
      throw std::invalid_argument(
          "conv_bwd_data dy dimensions do not match convolution");
    }
    auto dx = make_virtual_tensor(
        attributes.name_.empty()
            ? ("conv_bwd_data_" + std::to_string(nodes_.size()))
            : attributes.name_,
        x_shape,
        attributes.has_output_data_type_ ? attributes.output_data_type_
                                         : io_data_type_);

    Node node;
    node.kind = NodeKind::kConvBwdData;
    node.name = dx->name_;
    node.inputs = {dy, w};
    node.output = dx;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.conv_mode = attributes.mode_;
    node.conv_group_count = group_count;
    node.param0 = pre_padding;
    node.param1 = stride;
    node.param2 = dilation;
    node.param3 = post_padding;
    node.alpha = attributes.alpha_;
    node.beta = attributes.beta_;
    nodes_.push_back(std::move(node));
    return dx;
  }

  std::shared_ptr<Tensor_attributes> conv_dgrad(
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::shared_ptr<Tensor_attributes>& w,
      const std::vector<int64_t>& x_shape,
      const Conv_attributes& attributes = Conv_attributes()) {
    return conv_bwd_data(dy, w, x_shape, attributes);
  }

  std::shared_ptr<Tensor_attributes> conv_dgrad(
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::shared_ptr<Tensor_attributes>& w,
      const Conv_attributes& attributes) {
    if (!dy || !w) {
      throw std::invalid_argument("conv_dgrad requires dy and w");
    }
    if (dy->dims_.size() != 4 || w->dims_.size() != 4) {
      throw std::invalid_argument("conv_dgrad expects rank-4 NCHW tensors");
    }
    const auto pre_padding =
        normalize_conv_param(attributes.pre_padding_, "pre_padding", false);
    const auto post_padding =
        normalize_conv_param(attributes.post_padding_, "post_padding", false);
    const auto stride =
        normalize_conv_param(attributes.stride_, "stride", true);
    const auto dilation =
        normalize_conv_param(attributes.dilation_, "dilation", true);
    const int64_t group_count =
        attributes.has_group_count_ ? attributes.group_count_ : 1;
    std::vector<int64_t> x_shape = {
        dy->dims_[0],
        w->dims_[1] * group_count,
        (dy->dims_[2] - 1) * stride[0] - pre_padding[0] - post_padding[0] +
            dilation[0] * (w->dims_[2] - 1) + 1,
        (dy->dims_[3] - 1) * stride[1] - pre_padding[1] - post_padding[1] +
            dilation[1] * (w->dims_[3] - 1) + 1,
    };
    return conv_bwd_data(dy, w, x_shape, attributes);
  }

  std::shared_ptr<Tensor_attributes> conv_bwd_filter(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::vector<int64_t>& w_shape,
      const Conv_attributes& attributes = Conv_attributes()) {
    if (!x || !dy) {
      throw std::invalid_argument("conv_bwd_filter requires x and dy");
    }
    if (x->dims_.size() != 4 || dy->dims_.size() != 4) {
      throw std::invalid_argument(
          "conv_bwd_filter expects rank-4 NCHW tensors");
    }
    if (w_shape.size() != 4) {
      throw std::invalid_argument("conv_bwd_filter w_shape must be rank-4");
    }
    if (x->dims_[0] != dy->dims_[0]) {
      throw std::invalid_argument("conv_bwd_filter batch dimensions differ");
    }
    if (w_shape[1] <= 0 || x->dims_[1] % w_shape[1] != 0) {
      throw std::invalid_argument("conv_bwd_filter channel dimensions differ");
    }
    const int64_t inferred_group_count = x->dims_[1] / w_shape[1];
    const int64_t group_count = attributes.has_group_count_
        ? attributes.group_count_
        : inferred_group_count;
    if (group_count <= 0 || group_count != inferred_group_count ||
        dy->dims_[1] != w_shape[0] || w_shape[0] % group_count != 0) {
      throw std::invalid_argument("conv_bwd_filter channel dimensions differ");
    }
    const auto pre_padding =
        normalize_conv_param(attributes.pre_padding_, "pre_padding", false);
    const auto post_padding =
        normalize_conv_param(attributes.post_padding_, "post_padding", false);
    const auto stride =
        normalize_conv_param(attributes.stride_, "stride", true);
    const auto dilation =
        normalize_conv_param(attributes.dilation_, "dilation", true);
    const auto spatial = conv_output_spatial(
        {x->dims_[2], x->dims_[3]}, {w_shape[2], w_shape[3]}, pre_padding,
        post_padding, stride, dilation);
    if (dy->dims_[2] != spatial[0] || dy->dims_[3] != spatial[1]) {
      throw std::invalid_argument(
          "conv_bwd_filter dy dimensions do not match convolution");
    }
    auto dw = make_virtual_tensor(
        attributes.name_.empty()
            ? ("conv_bwd_filter_" + std::to_string(nodes_.size()))
            : attributes.name_,
        w_shape,
        attributes.has_output_data_type_ ? attributes.output_data_type_
                                         : io_data_type_);

    Node node;
    node.kind = NodeKind::kConvBwdFilter;
    node.name = dw->name_;
    node.inputs = {x, dy};
    node.output = dw;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.conv_mode = attributes.mode_;
    node.conv_group_count = group_count;
    node.param0 = pre_padding;
    node.param1 = stride;
    node.param2 = dilation;
    node.param3 = post_padding;
    node.alpha = attributes.alpha_;
    node.beta = attributes.beta_;
    nodes_.push_back(std::move(node));
    return dw;
  }

  std::shared_ptr<Tensor_attributes> conv_wgrad(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::vector<int64_t>& w_shape,
      const Conv_attributes& attributes = Conv_attributes()) {
    return conv_bwd_filter(x, dy, w_shape, attributes);
  }

  std::shared_ptr<Tensor_attributes> conv_wgrad(
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::shared_ptr<Tensor_attributes>& x,
      const Conv_attributes& attributes = Conv_attributes()) {
    if (!x || !dy) {
      throw std::invalid_argument("conv_wgrad requires dy and x");
    }
    if (x->dims_.size() != 4 || dy->dims_.size() != 4) {
      throw std::invalid_argument("conv_wgrad expects rank-4 NCHW tensors");
    }
    const auto pre_padding =
        normalize_conv_param(attributes.pre_padding_, "pre_padding", false);
    const auto post_padding =
        normalize_conv_param(attributes.post_padding_, "post_padding", false);
    const auto stride =
        normalize_conv_param(attributes.stride_, "stride", true);
    const auto dilation =
        normalize_conv_param(attributes.dilation_, "dilation", true);
    const int64_t group_count =
        attributes.has_group_count_ ? attributes.group_count_ : 1;
    if (group_count <= 0 || x->dims_[1] % group_count != 0 ||
        dy->dims_[1] % group_count != 0) {
      throw std::invalid_argument("conv_wgrad channel dimensions differ");
    }
    auto infer_kernel = [](int64_t x_dim, int64_t dy_dim, int64_t pre_pad,
                           int64_t post_pad, int64_t stride_dim,
                           int64_t dilation_dim) {
      const int64_t numerator =
          x_dim + pre_pad + post_pad - 1 - (dy_dim - 1) * stride_dim;
      if (numerator < 0 || numerator % dilation_dim != 0) {
        throw std::invalid_argument(
            "conv_wgrad could not infer filter spatial dimensions");
      }
      const int64_t kernel = numerator / dilation_dim + 1;
      if (kernel <= 0) {
        throw std::invalid_argument(
            "conv_wgrad inferred non-positive filter spatial dimension");
      }
      return kernel;
    };
    std::vector<int64_t> w_shape = {
        dy->dims_[1], x->dims_[1] / group_count,
        infer_kernel(x->dims_[2], dy->dims_[2], pre_padding[0], post_padding[0],
                     stride[0], dilation[0]),
        infer_kernel(x->dims_[3], dy->dims_[3], pre_padding[1], post_padding[1],
                     stride[1], dilation[1])};
    return conv_bwd_filter(x, dy, w_shape, attributes);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 2>
  resample(const std::shared_ptr<Tensor_attributes>& x,
           const Resample_attributes& attributes = Resample_attributes()) {
    if (!x) {
      throw std::invalid_argument("resample requires x");
    }
    if (x->dims_.size() != 4) {
      throw std::invalid_argument("resample expects rank-4 NCHW tensors");
    }
    const auto window =
        normalize_conv_param(attributes.window_, "window", true);
    const auto padding =
        normalize_conv_param(attributes.padding_, "padding", false);
    const auto stride = normalize_conv_param(
        attributes.has_stride_ ? attributes.stride_ : attributes.window_,
        "stride", true);
    const auto spatial = resample2d_output_spatial({x->dims_[2], x->dims_[3]},
                                                   window, padding, stride);
    if (spatial[0] <= 0 || spatial[1] <= 0) {
      throw std::invalid_argument(
          "resample output dimensions must be positive");
    }
    std::vector<int64_t> dims = {x->dims_[0], x->dims_[1], spatial[0],
                                 spatial[1]};
    const auto name = attributes.name_.empty()
        ? ("resample_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto y = make_virtual_tensor(name, dims,
                                 attributes.has_output_data_type_
                                     ? attributes.output_data_type_
                                     : io_data_type_);
    std::shared_ptr<Tensor_attributes> indices;
    if (attributes.return_indices_) {
      if (attributes.mode_ != MUDNN_RESAMPLE_MAXPOOL) {
        throw std::invalid_argument(
            "resample indices are only supported for max-pool");
      }
      indices = make_virtual_tensor(name + "_idx", dims, MUDNN_DATA_INT64);
    }

    Node node;
    node.kind = NodeKind::kResampleFwd;
    node.name = name;
    node.inputs = {x};
    node.output = y;
    node.stats = indices;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.resample_mode = attributes.mode_;
    node.param0 = window;
    node.param1 = padding;
    node.param2 = stride;
    node.alpha = attributes.alpha_;
    node.beta = attributes.beta_;
    nodes_.push_back(std::move(node));
    return {y, indices};
  }

  std::shared_ptr<Tensor_attributes> avg_pool(
      const std::shared_ptr<Tensor_attributes>& x,
      const Resample_attributes& attributes = Resample_attributes()) {
    auto attrs = attributes;
    if (attrs.mode_ != MUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING) {
      attrs.mode_ = MUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
    }
    return std::get<0>(resample(x, attrs));
  }

  std::shared_ptr<Tensor_attributes> max_pool(
      const std::shared_ptr<Tensor_attributes>& x,
      const Resample_attributes& attributes = Resample_attributes()) {
    auto attrs = attributes;
    attrs.mode_ = MUDNN_RESAMPLE_MAXPOOL;
    return std::get<0>(resample(x, attrs));
  }

  std::array<std::shared_ptr<Tensor_attributes>, 2>
  max_pool_with_indices(
      const std::shared_ptr<Tensor_attributes>& x,
      const Resample_attributes& attributes = Resample_attributes()) {
    auto attrs = attributes;
    attrs.mode_ = MUDNN_RESAMPLE_MAXPOOL;
    attrs.return_indices_ = true;
    return resample(x, attrs);
  }

  std::shared_ptr<Tensor_attributes> resample_bwd(
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::vector<int64_t>& x_shape,
      const Resample_attributes& attributes = Resample_attributes(),
      const std::shared_ptr<Tensor_attributes>& x = nullptr,
      const std::shared_ptr<Tensor_attributes>& y = nullptr,
      const std::shared_ptr<Tensor_attributes>& indices = nullptr) {
    if (!dy) {
      throw std::invalid_argument("resample_bwd requires dy");
    }
    if (dy->dims_.size() != 4) {
      throw std::invalid_argument("resample_bwd expects rank-4 NCHW tensors");
    }
    if (x_shape.size() != 4) {
      throw std::invalid_argument("resample_bwd x_shape must be rank-4");
    }
    const auto window =
        normalize_conv_param(attributes.window_, "window", true);
    const auto padding =
        normalize_conv_param(attributes.padding_, "padding", false);
    const auto stride = normalize_conv_param(
        attributes.has_stride_ ? attributes.stride_ : attributes.window_,
        "stride", true);
    const auto spatial = resample2d_output_spatial({x_shape[2], x_shape[3]},
                                                   window, padding, stride);
    if (dy->dims_[0] != x_shape[0] || dy->dims_[1] != x_shape[1] ||
        dy->dims_[2] != spatial[0] || dy->dims_[3] != spatial[1]) {
      throw std::invalid_argument(
          "resample_bwd dy dimensions do not match resample");
    }
    if (indices &&
        (indices->dims_ != dy->dims_ ||
         indices->data_type_ != MUDNN_DATA_INT64)) {
      throw std::invalid_argument(
          "resample_bwd indices must be int64 with dy dimensions");
    }
    if (x && (x->dims_ != x_shape || x->data_type_ != dy->data_type_)) {
      throw std::invalid_argument(
          "resample_bwd x dimensions and data type must match");
    }
    if (y && (y->dims_ != dy->dims_ || y->data_type_ != dy->data_type_)) {
      throw std::invalid_argument(
          "resample_bwd y dimensions and data type must match dy");
    }
    if (attributes.mode_ == MUDNN_RESAMPLE_MAXPOOL && !indices) {
      throw std::invalid_argument("max-pool backward requires index tensor");
    }
    auto dx = make_virtual_tensor(
        attributes.name_.empty()
            ? ("resample_bwd_" + std::to_string(nodes_.size()))
            : attributes.name_,
        x_shape,
        attributes.has_output_data_type_ ? attributes.output_data_type_
                                         : io_data_type_);

    Node node;
    node.kind = NodeKind::kResampleBwd;
    node.name = dx->name_;
    node.inputs = {dy};
    if (indices) {
      node.inputs.push_back(indices);
      node.resample_has_indices = true;
    }
    if (x) {
      node.inputs.push_back(x);
      node.resample_has_x = true;
    }
    if (y) {
      node.inputs.push_back(y);
      node.resample_has_y = true;
    }
    node.output = dx;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.resample_mode = attributes.mode_;
    node.param0 = window;
    node.param1 = padding;
    node.param2 = stride;
    node.alpha = attributes.alpha_;
    node.beta = attributes.beta_;
    nodes_.push_back(std::move(node));
    return dx;
  }

  std::shared_ptr<Tensor_attributes> avg_pool_bwd(
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::vector<int64_t>& x_shape,
      const Resample_attributes& attributes = Resample_attributes()) {
    auto attrs = attributes;
    if (attrs.mode_ != MUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING) {
      attrs.mode_ = MUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
    }
    return resample_bwd(dy, x_shape, attrs);
  }

  std::shared_ptr<Tensor_attributes> max_pool_bwd(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& dy,
      const Resample_attributes& attributes = Resample_attributes()) {
    if (!x || !dy) {
      throw std::invalid_argument("max_pool_bwd requires x and dy");
    }
    auto attrs = attributes;
    attrs.mode_ = MUDNN_RESAMPLE_MAXPOOL;
    auto forward_attrs = attrs;
    forward_attrs.name_ = attrs.name_.empty() ? "" : attrs.name_ + "_forward";
    forward_attrs.return_indices_ = true;
    forward_attrs.output_data_type_ = dy->data_type_;
    forward_attrs.has_output_data_type_ = true;
    auto fwd = resample(x, forward_attrs);
    attrs.name_ = attrs.name_.empty() ? "" : attrs.name_;
    attrs.output_data_type_ = attributes.has_output_data_type_
        ? attributes.output_data_type_
        : x->data_type_;
    attrs.has_output_data_type_ = true;
    return resample_bwd(dy, x->dims_, attrs, x, std::get<0>(fwd),
                        std::get<1>(fwd));
  }

  std::shared_ptr<Tensor_attributes> batch_norm_inference(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& mean,
      const std::shared_ptr<Tensor_attributes>& inv_variance,
      const std::shared_ptr<Tensor_attributes>& scale,
      const std::shared_ptr<Tensor_attributes>& bias,
      const Batchnorm_attributes& attributes = Batchnorm_attributes()) {
    if (!x || !mean || !inv_variance || !scale || !bias) {
      throw std::invalid_argument(
          "batch_norm_inference requires x, mean, inv_variance, scale, and "
          "bias");
    }
    if (attributes.epsilon_tensor_) {
      check_scalar_float_tensor("batch_norm_inference", "epsilon",
                                attributes.epsilon_tensor_);
    } else if (attributes.epsilon_ < 0.0) {
      throw std::invalid_argument(
          "batch_norm_inference epsilon must be nonnegative");
    }
    check_batch_norm_param("mean", mean, x);
    check_batch_norm_param("inv_variance", inv_variance, x);
    check_batch_norm_param("scale", scale, x);
    check_batch_norm_param("bias", bias, x);
    auto y = make_virtual_tensor(
        attributes.name_.empty()
            ? ("batch_norm_inference_" + std::to_string(nodes_.size()))
            : attributes.name_,
        x->dims_,
        attributes.has_output_data_type_ ? attributes.output_data_type_
                                         : x->data_type_);

    Node node;
    node.kind = NodeKind::kNormFwd;
    node.name = y->name_;
    node.inputs = {x, mean, inv_variance, scale, bias};
    if (attributes.epsilon_tensor_) {
      node.inputs.push_back(attributes.epsilon_tensor_);
    }
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.norm_mode = MUDNN_BATCH_NORM;
    node.norm_phase = MUDNN_NORM_FWD_INFERENCE;
    node.norm_epsilon = attributes.epsilon_;
    node.norm_momentum = attributes.momentum_;
    node.norm_has_epsilon_tensor =
        static_cast<bool>(attributes.epsilon_tensor_);
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> batchnorm_inference(
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& mean,
      const std::shared_ptr<Tensor_attributes>& inv_variance,
      const std::shared_ptr<Tensor_attributes>& scale,
      const std::shared_ptr<Tensor_attributes>& bias,
      const Batchnorm_attributes& attributes = Batchnorm_attributes()) {
    return batch_norm_inference(input, mean, inv_variance, scale, bias,
                                attributes);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 5>
  batch_norm_training(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& input_running_mean,
      const std::shared_ptr<Tensor_attributes>& input_running_variance,
      const std::shared_ptr<Tensor_attributes>& scale,
      const std::shared_ptr<Tensor_attributes>& bias,
      const Batchnorm_attributes& attributes = Batchnorm_attributes()) {
    if (!x || !input_running_mean || !input_running_variance || !scale ||
        !bias) {
      throw std::invalid_argument(
          "batch_norm_training requires x, running mean/variance, scale, and "
          "bias");
    }
    if (attributes.epsilon_tensor_) {
      check_scalar_float_tensor("batch_norm_training", "epsilon",
                                attributes.epsilon_tensor_);
    } else if (attributes.epsilon_ < 0.0) {
      throw std::invalid_argument(
          "batch_norm_training epsilon must be nonnegative");
    }
    if (attributes.momentum_tensor_) {
      check_scalar_float_tensor("batch_norm_training", "momentum",
                                attributes.momentum_tensor_);
    }
    check_batch_norm_param("input_running_mean", input_running_mean, x);
    check_batch_norm_param("input_running_variance", input_running_variance, x);
    check_batch_norm_param("scale", scale, x);
    check_batch_norm_param("bias", bias, x);
    const auto base_name = attributes.name_.empty()
        ? ("batch_norm_training_" + std::to_string(nodes_.size()))
        : attributes.name_;
    const auto param_dims = batch_norm_param_dims(x);
    auto y = make_virtual_tensor(base_name, x->dims_,
                                 attributes.has_output_data_type_
                                     ? attributes.output_data_type_
                                     : x->data_type_);
    auto output_running_mean = make_virtual_tensor(base_name + "_running_mean",
                                                   param_dims, x->data_type_);
    auto output_running_variance = make_virtual_tensor(
        base_name + "_running_variance", param_dims, x->data_type_);
    auto saved_mean = make_virtual_tensor(base_name + "_saved_mean", param_dims,
                                          x->data_type_);
    auto saved_inv_variance = make_virtual_tensor(
        base_name + "_saved_inv_variance", param_dims, x->data_type_);

    Node node;
    node.kind = NodeKind::kNormFwd;
    node.name = base_name;
    node.inputs = {x, input_running_mean, input_running_variance, scale, bias};
    if (attributes.epsilon_tensor_) {
      node.inputs.push_back(attributes.epsilon_tensor_);
    }
    if (attributes.momentum_tensor_) {
      node.inputs.push_back(attributes.momentum_tensor_);
    }
    for (const auto& peer_stat : attributes.peer_stats_) {
      if (!peer_stat) {
        throw std::invalid_argument(
            "batch_norm_training peer stats must be non-null");
      }
      node.inputs.push_back(peer_stat);
    }
    node.output = y;
    node.stats = output_running_mean;
    node.aux_outputs = {output_running_variance, saved_mean,
                        saved_inv_variance};
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.norm_mode = MUDNN_BATCH_NORM;
    node.norm_phase = MUDNN_NORM_FWD_TRAINING;
    node.norm_epsilon = attributes.epsilon_;
    node.norm_momentum = attributes.momentum_;
    node.norm_has_epsilon_tensor =
        static_cast<bool>(attributes.epsilon_tensor_);
    node.norm_has_momentum_tensor =
        static_cast<bool>(attributes.momentum_tensor_);
    nodes_.push_back(std::move(node));
    return {y, output_running_mean, output_running_variance, saved_mean,
            saved_inv_variance};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 5>
  batchnorm(const std::shared_ptr<Tensor_attributes>& input,
            const std::shared_ptr<Tensor_attributes>& scale,
            const std::shared_ptr<Tensor_attributes>& bias,
            const Batchnorm_attributes& attributes = Batchnorm_attributes()) {
    if (!attributes.previous_running_mean_ ||
        !attributes.previous_running_var_) {
      throw std::invalid_argument(
          "batchnorm attributes require previous running stats");
    }
    auto outputs = batch_norm_training(input, attributes.previous_running_mean_,
                                       attributes.previous_running_var_, scale,
                                       bias, attributes);
    return {std::get<0>(outputs), std::get<3>(outputs), std::get<4>(outputs),
            std::get<1>(outputs), std::get<2>(outputs)};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 5>
  batchnorm(const std::shared_ptr<Tensor_attributes>& input,
            const std::shared_ptr<Tensor_attributes>& scale,
            const std::shared_ptr<Tensor_attributes>& bias,
            const std::shared_ptr<Tensor_attributes>& in_running_mean,
            const std::shared_ptr<Tensor_attributes>& in_running_var,
            const Batchnorm_attributes& attributes = Batchnorm_attributes()) {
    auto outputs = batch_norm_training(input, in_running_mean, in_running_var,
                                       scale, bias, attributes);
    return {std::get<0>(outputs), std::get<3>(outputs), std::get<4>(outputs),
            std::get<1>(outputs), std::get<2>(outputs)};
  }

  // Batch-norm backward over the statistics saved by the training forward
  // pass. ``inv_variance`` is the saved inverse standard deviation, i.e. it
  // already folds in the forward epsilon; ``attributes.epsilon_`` is accepted
  // for symmetry with the forward builders but is not applied a second time.
  std::array<std::shared_ptr<Tensor_attributes>, 3>
  batch_norm_backward(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::shared_ptr<Tensor_attributes>& mean,
      const std::shared_ptr<Tensor_attributes>& inv_variance,
      const std::shared_ptr<Tensor_attributes>& scale,
      const Batchnorm_attributes& attributes = Batchnorm_attributes()) {
    if (!x || !dy || !mean || !inv_variance || !scale) {
      throw std::invalid_argument(
          "batch_norm_backward requires x, dy, mean, inv_variance, and scale");
    }
    if (attributes.epsilon_tensor_) {
      check_scalar_float_tensor("batch_norm_backward", "epsilon",
                                attributes.epsilon_tensor_);
    } else if (attributes.epsilon_ < 0.0) {
      throw std::invalid_argument(
          "batch_norm_backward epsilon must be nonnegative");
    }
    if (dy->dims_ != x->dims_ || dy->data_type_ != x->data_type_) {
      throw std::invalid_argument(
          "batch_norm_backward dy shape and data type must match x");
    }
    check_batch_norm_param("mean", mean, x);
    check_batch_norm_param("inv_variance", inv_variance, x);
    check_batch_norm_param("scale", scale, x);
    const auto base_name = attributes.name_.empty()
        ? ("batch_norm_backward_" + std::to_string(nodes_.size()))
        : attributes.name_;
    const auto param_dims = batch_norm_param_dims(x);
    auto dx = make_virtual_tensor(base_name, x->dims_,
                                  attributes.has_output_data_type_
                                      ? attributes.output_data_type_
                                      : x->data_type_);
    auto dscale =
        make_virtual_tensor(base_name + "_dscale", param_dims, x->data_type_);
    auto dbias =
        make_virtual_tensor(base_name + "_dbias", param_dims, x->data_type_);

    Node node;
    node.kind = NodeKind::kNormBwd;
    node.name = base_name;
    node.inputs = {x, dy, mean, inv_variance, scale};
    if (attributes.epsilon_tensor_) {
      node.inputs.push_back(attributes.epsilon_tensor_);
    }
    node.output = dx;
    node.stats = dscale;
    node.aux_outputs = {dbias};
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.norm_mode = MUDNN_BATCH_NORM;
    node.norm_epsilon = attributes.epsilon_;
    node.norm_has_epsilon_tensor =
        static_cast<bool>(attributes.epsilon_tensor_);
    nodes_.push_back(std::move(node));
    return {dx, dscale, dbias};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  batchnorm_backward(
      const std::shared_ptr<Tensor_attributes>& grad,
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& scale,
      const std::shared_ptr<Tensor_attributes>& mean,
      const std::shared_ptr<Tensor_attributes>& inv_variance,
      const Batchnorm_attributes& attributes = Batchnorm_attributes()) {
    return batch_norm_backward(input, grad, mean, inv_variance, scale,
                               attributes);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  batchnorm_backward(const std::shared_ptr<Tensor_attributes>& grad,
                     const std::shared_ptr<Tensor_attributes>& input,
                     const std::shared_ptr<Tensor_attributes>& scale,
                     const Batchnorm_attributes& attributes) {
    if (!attributes.saved_mean_ || !attributes.saved_inv_variance_) {
      throw std::invalid_argument(
          "batchnorm_backward attributes require saved mean and inv_variance");
    }
    return batchnorm_backward(grad, input, scale, attributes.saved_mean_,
                              attributes.saved_inv_variance_, attributes);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  layernorm(const std::shared_ptr<Tensor_attributes>& input,
            const std::shared_ptr<Tensor_attributes>& scale,
            const std::shared_ptr<Tensor_attributes>& bias,
            const Norm_attributes& attributes = Norm_attributes()) {
    return norm_forward_impl("layernorm", MUDNN_LAYER_NORM, input, scale, bias,
                             attributes, true);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  layernorm_backward(const std::shared_ptr<Tensor_attributes>& grad,
                     const std::shared_ptr<Tensor_attributes>& input,
                     const std::shared_ptr<Tensor_attributes>& scale,
                     const std::shared_ptr<Tensor_attributes>& mean,
                     const std::shared_ptr<Tensor_attributes>& inv_variance,
                     const Norm_attributes& attributes = Norm_attributes()) {
    return norm_backward_impl("layernorm_backward", MUDNN_LAYER_NORM, grad,
                              input, scale, mean, inv_variance, attributes,
                              true);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  layernorm_backward(const std::shared_ptr<Tensor_attributes>& grad,
                     const std::shared_ptr<Tensor_attributes>& input,
                     const std::shared_ptr<Tensor_attributes>& scale,
                     const Norm_attributes& attributes) {
    if (!attributes.saved_mean_ || !attributes.saved_inv_variance_) {
      throw std::invalid_argument(
          "layernorm_backward attributes require saved mean and inv_variance");
    }
    return layernorm_backward(grad, input, scale, attributes.saved_mean_,
                              attributes.saved_inv_variance_, attributes);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  adalayernorm(const std::shared_ptr<Tensor_attributes>& input,
               const std::shared_ptr<Tensor_attributes>& scale,
               const std::shared_ptr<Tensor_attributes>& bias,
               const Norm_attributes& attributes = Norm_attributes()) {
    return norm_forward_impl("adalayernorm", MUDNN_ADA_LAYER_NORM, input, scale,
                             bias, attributes, true);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  adalayernorm_backward(const std::shared_ptr<Tensor_attributes>& grad,
                        const std::shared_ptr<Tensor_attributes>& input,
                        const std::shared_ptr<Tensor_attributes>& scale,
                        const std::shared_ptr<Tensor_attributes>& mean,
                        const std::shared_ptr<Tensor_attributes>& inv_variance,
                        const Norm_attributes& attributes = Norm_attributes()) {
    return norm_backward_impl("adalayernorm_backward", MUDNN_ADA_LAYER_NORM,
                              grad, input, scale, mean, inv_variance,
                              attributes, true);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  adalayernorm_backward(const std::shared_ptr<Tensor_attributes>& grad,
                        const std::shared_ptr<Tensor_attributes>& input,
                        const std::shared_ptr<Tensor_attributes>& scale,
                        const Norm_attributes& attributes) {
    if (!attributes.saved_mean_ || !attributes.saved_inv_variance_) {
      throw std::invalid_argument(
          "adalayernorm_backward attributes require saved mean and "
          "inv_variance");
    }
    return adalayernorm_backward(grad, input, scale, attributes.saved_mean_,
                                 attributes.saved_inv_variance_, attributes);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 2>
  rmsnorm(const std::shared_ptr<Tensor_attributes>& input,
          const std::shared_ptr<Tensor_attributes>& scale,
          const Norm_attributes& attributes = Norm_attributes(),
          const std::shared_ptr<Tensor_attributes>& bias = nullptr) {
    const auto effective_bias = bias ? bias : attributes.bias_;
    auto outputs = norm_forward_impl("rmsnorm", MUDNN_RMS_NORM, input, scale,
                                     effective_bias, attributes, false);
    return {std::get<0>(outputs), std::get<2>(outputs)};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  rmsnorm_backward(const std::shared_ptr<Tensor_attributes>& grad,
                   const std::shared_ptr<Tensor_attributes>& input,
                   const std::shared_ptr<Tensor_attributes>& scale,
                   const std::shared_ptr<Tensor_attributes>& inv_variance,
                   const Norm_attributes& attributes = Norm_attributes()) {
    return norm_backward_impl("rmsnorm_backward", MUDNN_RMS_NORM, grad, input,
                              scale, nullptr, inv_variance, attributes,
                              attributes.has_dbias_);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  instancenorm(const std::shared_ptr<Tensor_attributes>& input,
               const std::shared_ptr<Tensor_attributes>& scale,
               const std::shared_ptr<Tensor_attributes>& bias,
               const Norm_attributes& attributes = Norm_attributes()) {
    return norm_forward_impl("instancenorm", MUDNN_INSTANCE_NORM, input, scale,
                             bias, attributes, true);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  instancenorm_backward(const std::shared_ptr<Tensor_attributes>& grad,
                        const std::shared_ptr<Tensor_attributes>& input,
                        const std::shared_ptr<Tensor_attributes>& scale,
                        const std::shared_ptr<Tensor_attributes>& mean,
                        const std::shared_ptr<Tensor_attributes>& inv_variance,
                        const Norm_attributes& attributes = Norm_attributes()) {
    return norm_backward_impl("instancenorm_backward", MUDNN_INSTANCE_NORM,
                              grad, input, scale, mean, inv_variance,
                              attributes, true);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 2>
  block_scale_quantize(
      const std::shared_ptr<Tensor_attributes>& x,
      const Block_scale_attributes& attributes = Block_scale_attributes()) {
    if (!x) {
      throw std::invalid_argument("block_scale_quantize requires x");
    }
    const auto scale_dims = block_scale_scale_dims(x, attributes);
    const auto base_name = attributes.name_.empty()
        ? ("block_scale_quantize_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto y = make_virtual_tensor(base_name, x->dims_,
                                 attributes.has_output_data_type_
                                     ? attributes.output_data_type_
                                     : x->data_type_);
    auto scale = make_virtual_tensor(base_name + "_scale", scale_dims,
                                     attributes.has_scale_data_type_
                                         ? attributes.scale_data_type_
                                         : x->data_type_);

    Node node;
    node.kind = NodeKind::kBlockScaleQuantize;
    node.name = base_name;
    node.inputs = {x};
    node.output = y;
    node.stats = scale;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.block_scale_block_size = attributes.block_size_;
    node.block_scale_block_axis = block_scale_axis(x, attributes);
    nodes_.push_back(std::move(node));
    return {y, scale};
  }

  std::shared_ptr<Tensor_attributes> block_scale_dequantize(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& scale,
      const Block_scale_attributes& attributes = Block_scale_attributes()) {
    return block_scale_dequantize_impl(x, scale, attributes, false, false);
  }

  std::shared_ptr<Tensor_attributes> block_scale_dequantize(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& scale,
      const Block_scale_dequantize_attributes& attributes) {
    return block_scale_dequantize_impl(x, scale, attributes,
                                       attributes.is_negative_scale_, true);
  }

  std::shared_ptr<Tensor_attributes> moe_grouped_matmul(
      const std::shared_ptr<Tensor_attributes>& token,
      const std::shared_ptr<Tensor_attributes>& weight,
      const std::shared_ptr<Tensor_attributes>& first_token_offset,
      const std::shared_ptr<Tensor_attributes>& token_index,
      const std::shared_ptr<Tensor_attributes>& token_ks,
      const Moe_grouped_matmul_attributes& attributes =
          Moe_grouped_matmul_attributes()) {
    if (!token || !weight || !first_token_offset) {
      throw std::invalid_argument(
          "moe_grouped_matmul requires token, weight, and first_token_offset");
    }
    if (token->dims_.size() != 3 || weight->dims_.size() != 3) {
      throw std::invalid_argument(
          "moe_grouped_matmul token and weight must be rank-3");
    }
    const auto expert_count = moe_first_token_offset_experts(
        first_token_offset, "moe_grouped_matmul");
    if (token->dims_[2] != weight->dims_[1]) {
      throw std::invalid_argument("moe_grouped_matmul K dimensions must match");
    }
    if (expert_count != weight->dims_[0]) {
      throw std::invalid_argument(
          "moe_grouped_matmul first_token_offset length must match experts");
    }
    if (attributes.mode_ == MUDNN_MOE_GROUPED_MATMUL_MODE_GATHER ||
        attributes.mode_ == MUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER) {
      if (!token_index) {
        throw std::invalid_argument(
            "moe_grouped_matmul token_index is required");
      }
      if (token_index->dims_.size() != 2) {
        throw std::invalid_argument(
            "moe_grouped_matmul token_index must be rank-2");
      }
    }
    if (attributes.mode_ == MUDNN_MOE_GROUPED_MATMUL_MODE_SCATTER) {
      if (!token_ks) {
        throw std::invalid_argument("moe_grouped_matmul token_ks is required");
      }
      if (token_ks->dims_.size() != 1) {
        throw std::invalid_argument(
            "moe_grouped_matmul token_ks must be rank-1");
      }
      if (attributes.top_k_ <= 0) {
        throw std::invalid_argument(
            "moe_grouped_matmul top_k must be positive");
      }
    }

    const auto token_count =
        attributes.mode_ == MUDNN_MOE_GROUPED_MATMUL_MODE_GATHER
        ? token_index->dims_[1]
        : token->dims_[1];
    const auto base_name = attributes.name_.empty()
        ? ("moe_grouped_matmul_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto y = make_virtual_tensor(base_name, {1, token_count, weight->dims_[2]},
                                 token->data_type_);

    Node node;
    node.kind = NodeKind::kMoeGroupedMatmul;
    node.name = base_name;
    node.inputs = {token, weight, first_token_offset};
    if (token_index) {
      node.inputs.push_back(token_index);
    }
    if (token_ks) {
      node.inputs.push_back(token_ks);
    }
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.moe_grouped_matmul_mode = attributes.mode_;
    node.moe_grouped_matmul_top_k = attributes.top_k_;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> moe_grouped_matmul_bwd(
      const std::shared_ptr<Tensor_attributes>& doutput,
      const std::shared_ptr<Tensor_attributes>& token,
      const std::shared_ptr<Tensor_attributes>& first_token_offset,
      const Moe_grouped_matmul_bwd_attributes& attributes =
          Moe_grouped_matmul_bwd_attributes()) {
    if (!doutput || !token || !first_token_offset) {
      throw std::invalid_argument(
          "moe_grouped_matmul_bwd requires doutput, token, and "
          "first_token_offset");
    }
    if (doutput->dims_.size() != 3 || token->dims_.size() != 3) {
      throw std::invalid_argument(
          "moe_grouped_matmul_bwd doutput and token must be rank-3");
    }
    const auto expert_count = moe_first_token_offset_experts(
        first_token_offset, "moe_grouped_matmul_bwd");
    if (doutput->dims_[1] != token->dims_[1]) {
      throw std::invalid_argument(
          "moe_grouped_matmul_bwd token counts must match");
    }
    const auto base_name = attributes.name_.empty()
        ? ("moe_grouped_matmul_bwd_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto dweight = make_virtual_tensor(
        base_name, {expert_count, token->dims_[2], doutput->dims_[2]},
        token->data_type_);

    Node node;
    node.kind = NodeKind::kMoeGroupedMatmulBwd;
    node.name = base_name;
    node.inputs = {doutput, token, first_token_offset};
    node.output = dweight;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    nodes_.push_back(std::move(node));
    return dweight;
  }

 private:
  std::shared_ptr<Tensor_attributes> block_scale_dequantize_impl(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& scale,
      const Block_scale_attributes& attributes, bool is_negative_scale,
      bool use_cudnn_output_name) {
    if (!x || !scale) {
      throw std::invalid_argument(
          "block_scale_dequantize requires x and scale");
    }
    check_block_scale_scale("block_scale_dequantize scale", scale, x,
                            attributes);
    const auto base_name = attributes.name_.empty()
        ? ("block_scale_dequantize_" + std::to_string(nodes_.size()))
        : attributes.name_;
    const auto output_name =
        use_cudnn_output_name ? (base_name + "::Y") : base_name;
    auto y = make_virtual_tensor(output_name, x->dims_,
                                 attributes.has_output_data_type_
                                     ? attributes.output_data_type_
                                     : x->data_type_);

    Node node;
    node.kind = NodeKind::kBlockScaleDequantize;
    node.name = base_name;
    node.inputs = {x, scale};
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.block_scale_block_size = attributes.block_size_;
    node.block_scale_block_axis = block_scale_axis(x, attributes);
    node.block_scale_is_negative = is_negative_scale;
    nodes_.push_back(std::move(node));
    return y;
  }

 public:
  std::shared_ptr<Tensor_attributes> paged_cache_load(
      const std::shared_ptr<Tensor_attributes>& container,
      const std::shared_ptr<Tensor_attributes>& sequence,
      const std::shared_ptr<Tensor_attributes>& page_table,
      const Paged_cache_load_attributes& attributes =
          Paged_cache_load_attributes()) {
    if (!container || !sequence || !page_table) {
      throw std::invalid_argument(
          "paged_cache_load requires container, sequence, and page_table");
    }
    const auto dims =
        paged_cache_load_output_dims(container, sequence, page_table);
    const auto out_type = attributes.has_output_data_type_
        ? attributes.output_data_type_
        : container->data_type_;
    if (out_type != container->data_type_) {
      throw std::invalid_argument(
          "paged_cache_load output data type must match container");
    }
    auto y = make_virtual_tensor(
        attributes.name_.empty()
            ? ("paged_cache_load_" + std::to_string(nodes_.size()))
            : attributes.name_,
        dims, out_type);

    Node node;
    node.kind = NodeKind::kPagedCacheLoad;
    node.name = y->name_;
    node.inputs = {container, sequence, page_table};
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = compute_data_type_;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> signal(
      mudnnSignalMode_t mode, const std::shared_ptr<Tensor_attributes>& flag,
      const std::shared_ptr<Tensor_attributes>& x,
      const Signal_attributes& attributes = Signal_attributes()) {
    if (!flag || !x) {
      throw std::invalid_argument("signal requires flag and x");
    }
    check_signal_flag(flag);
    const auto out_type = attributes.has_output_data_type_
        ? attributes.output_data_type_
        : x->data_type_;
    if (out_type != x->data_type_) {
      throw std::invalid_argument("signal output data type must match x");
    }
    auto y = make_virtual_tensor(
        attributes.name_.empty() ? ("signal_" + std::to_string(nodes_.size()))
                                 : attributes.name_,
        x->dims_, out_type);

    Node node;
    node.kind = NodeKind::kSignal;
    node.name = y->name_;
    node.inputs = {x};
    if (mode == MUDNN_SIGNAL_WAIT) {
      node.inputs.push_back(flag);
    } else if (mode == MUDNN_SIGNAL_SET) {
      node.stats = flag;
    } else {
      throw std::invalid_argument("unsupported signal mode");
    }
    node.output = y;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = compute_data_type_;
    node.signal_mode = mode;
    node.signal_value = attributes.value_;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::array<std::shared_ptr<Tensor_attributes>, 2>
  bn_finalize_inference(
      const std::shared_ptr<Tensor_attributes>& scale,
      const std::shared_ptr<Tensor_attributes>& bias,
      const std::shared_ptr<Tensor_attributes>& running_mean,
      const std::shared_ptr<Tensor_attributes>& running_var,
      const Bn_finalize_attributes& attributes = Bn_finalize_attributes()) {
    if (!scale || !bias || !running_mean || !running_var) {
      throw std::invalid_argument(
          "bn_finalize_inference requires scale, bias, running_mean, and "
          "running_var");
    }
    if (attributes.epsilon_ < 0.0) {
      throw std::invalid_argument("bn_finalize epsilon must be nonnegative");
    }
    check_same_shape_type("bn_finalize bias", bias, scale);
    check_same_shape_type("bn_finalize running_mean", running_mean, scale);
    check_same_shape_type("bn_finalize running_var", running_var, scale);
    const auto out_type = attributes.has_output_data_type_
        ? attributes.output_data_type_
        : scale->data_type_;
    if (out_type != scale->data_type_) {
      throw std::invalid_argument(
          "bn_finalize output data type must match scale");
    }
    const auto base_name = attributes.name_.empty()
        ? ("bn_finalize_inference_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto eq_scale = make_virtual_like(scale, base_name + "_eq_scale", out_type);
    auto eq_bias = make_virtual_like(scale, base_name + "_eq_bias", out_type);

    Node node;
    node.kind = NodeKind::kBnFinalize;
    node.name = base_name;
    node.inputs = {scale, bias, running_mean, running_var};
    node.output = eq_scale;
    node.stats = eq_bias;
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.bn_finalize_mode = MUDNN_BN_FINALIZE_STATISTICS_INFERENCE;
    node.bn_finalize_epsilon = attributes.epsilon_;
    node.bn_finalize_momentum = attributes.momentum_;
    nodes_.push_back(std::move(node));
    return {eq_scale, eq_bias};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 6>
  bn_finalize_training(
      const std::shared_ptr<Tensor_attributes>& y_sum,
      const std::shared_ptr<Tensor_attributes>& y_sq_sum,
      const std::shared_ptr<Tensor_attributes>& scale,
      const std::shared_ptr<Tensor_attributes>& bias,
      const std::shared_ptr<Tensor_attributes>& prev_running_mean,
      const std::shared_ptr<Tensor_attributes>& prev_running_var,
      const std::shared_ptr<Tensor_attributes>& accum_count,
      const Bn_finalize_attributes& attributes = Bn_finalize_attributes()) {
    if (!y_sum || !y_sq_sum || !scale || !bias || !prev_running_mean ||
        !prev_running_var || !accum_count) {
      throw std::invalid_argument(
          "bn_finalize_training requires statistics, parameters, running "
          "state, and accum_count");
    }
    if (attributes.epsilon_ < 0.0) {
      throw std::invalid_argument("bn_finalize epsilon must be nonnegative");
    }
    if (accum_count->data_type_ != MUDNN_DATA_INT64 ||
        accum_count->dims_ != std::vector<int64_t>{1}) {
      throw std::invalid_argument(
          "bn_finalize accum_count must be an int64 scalar tensor");
    }
    check_same_shape_type("bn_finalize y_sum", y_sum, scale);
    check_same_shape_type("bn_finalize y_sq_sum", y_sq_sum, scale);
    check_same_shape_type("bn_finalize bias", bias, scale);
    check_same_shape_type("bn_finalize prev_running_mean", prev_running_mean,
                          scale);
    check_same_shape_type("bn_finalize prev_running_var", prev_running_var,
                          scale);
    const auto out_type = attributes.has_output_data_type_
        ? attributes.output_data_type_
        : scale->data_type_;
    if (out_type != scale->data_type_) {
      throw std::invalid_argument(
          "bn_finalize output data type must match scale");
    }
    const auto base_name = attributes.name_.empty()
        ? ("bn_finalize_training_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto saved_mean =
        make_virtual_like(scale, base_name + "_saved_mean", out_type);
    auto saved_inv_std =
        make_virtual_like(scale, base_name + "_saved_inv_std", out_type);
    auto updated_running_mean =
        make_virtual_like(scale, base_name + "_updated_running_mean", out_type);
    auto updated_running_var =
        make_virtual_like(scale, base_name + "_updated_running_var", out_type);
    auto eq_scale = make_virtual_like(scale, base_name + "_eq_scale", out_type);
    auto eq_bias = make_virtual_like(scale, base_name + "_eq_bias", out_type);

    Node node;
    node.kind = NodeKind::kBnFinalize;
    node.name = base_name;
    node.inputs = {y_sum,      y_sq_sum,          scale,
                   bias,       prev_running_mean, prev_running_var,
                   accum_count};
    node.output = saved_mean;
    node.stats = saved_inv_std;
    node.aux_outputs = {updated_running_mean, updated_running_var, eq_scale,
                        eq_bias};
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.bn_finalize_mode = MUDNN_BN_FINALIZE_STATISTICS_TRAINING;
    node.bn_finalize_epsilon = attributes.epsilon_;
    node.bn_finalize_momentum = attributes.momentum_;
    nodes_.push_back(std::move(node));
    return {saved_mean,          saved_inv_std, updated_running_mean,
            updated_running_var, eq_scale,      eq_bias};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 6>
  bn_finalize(
      const std::shared_ptr<Tensor_attributes>& y_sum,
      const std::shared_ptr<Tensor_attributes>& y_sq_sum,
      const std::shared_ptr<Tensor_attributes>& scale,
      const std::shared_ptr<Tensor_attributes>& bias,
      const std::shared_ptr<Tensor_attributes>& epsilon,
      const std::shared_ptr<Tensor_attributes>& accum_count,
      const Bn_finalize_attributes& attributes = Bn_finalize_attributes()) {
    if (!attributes.previous_running_mean_ ||
        !attributes.previous_running_var_) {
      throw std::invalid_argument(
          "bn_finalize attributes require previous running stats");
    }
    auto attrs = attributes;
    if (epsilon) {
      check_scalar_float_tensor("bn_finalize", "epsilon", epsilon);
    }
    auto outputs = bn_finalize_training(
        y_sum, y_sq_sum, scale, bias, attrs.previous_running_mean_,
        attrs.previous_running_var_, accum_count, attrs);
    return {std::get<4>(outputs), std::get<5>(outputs), std::get<0>(outputs),
            std::get<1>(outputs), std::get<2>(outputs), std::get<3>(outputs)};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 5>
  bn_backward_weights(const std::shared_ptr<Tensor_attributes>& x,
                      const std::shared_ptr<Tensor_attributes>& dy,
                      const std::shared_ptr<Tensor_attributes>& mean,
                      const std::shared_ptr<Tensor_attributes>& invstd,
                      const std::shared_ptr<Tensor_attributes>& scale,
                      const Bn_backward_weights_attributes& attributes =
                          Bn_backward_weights_attributes()) {
    if (!x || !dy || !mean || !invstd || !scale) {
      throw std::invalid_argument(
          "bn_backward_weights requires x, dy, mean, invstd, and scale");
    }
    if (dy->dims_ != x->dims_ || dy->data_type_ != x->data_type_) {
      throw std::invalid_argument(
          "bn_backward_weights dy shape and data type must match x");
    }
    check_batch_norm_param("mean", mean, x);
    check_batch_norm_param("invstd", invstd, x);
    check_batch_norm_param("scale", scale, x);
    const auto out_type = attributes.has_output_data_type_
        ? attributes.output_data_type_
        : scale->data_type_;
    if (out_type != scale->data_type_) {
      throw std::invalid_argument(
          "bn_backward_weights output data type must match scale");
    }
    const auto base_name = attributes.name_.empty()
        ? ("bn_backward_weights_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto dbn_scale =
        make_virtual_like(scale, base_name + "_dbn_scale", out_type);
    auto dbn_bias = make_virtual_like(scale, base_name + "_dbn_bias", out_type);
    auto eq_dy_scale =
        make_virtual_like(scale, base_name + "_eq_dy_scale", out_type);
    auto eq_x_scale =
        make_virtual_like(scale, base_name + "_eq_x_scale", out_type);
    auto eq_bias = make_virtual_like(scale, base_name + "_eq_bias", out_type);

    Node node;
    node.kind = NodeKind::kBnBwdWeights;
    node.name = base_name;
    node.inputs = {x, dy, mean, invstd, scale};
    node.output = dbn_scale;
    node.stats = dbn_bias;
    node.aux_outputs = {eq_dy_scale, eq_x_scale, eq_bias};
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    nodes_.push_back(std::move(node));
    return {dbn_scale, dbn_bias, eq_dy_scale, eq_x_scale, eq_bias};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 5>
  dbn_weight(
      const std::shared_ptr<Tensor_attributes>& dy,
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& mean,
      const std::shared_ptr<Tensor_attributes>& invstd,
      const std::shared_ptr<Tensor_attributes>& scale,
      const DBN_weight_attributes& attributes = DBN_weight_attributes()) {
    return bn_backward_weights(x, dy, mean, invstd, scale, attributes);
  }

  error_t validate() const {
    std::map<int64_t, std::string> uid_names;
    for (const auto& tensor : tensors_) {
      if (!tensor) {
        return {false, "graph contains a null tensor",
                error_code_t::INVALID_VALUE};
      }
      auto tensor_status = tensor->validate();
      if (!tensor_status.is_good()) {
        return tensor_status;
      }
      if (tensor->has_uid_) {
        auto [it, inserted] = uid_names.emplace(tensor->uid_, tensor->name_);
        if (!inserted) {
          return {false,
                  "Tensor '" + tensor->name_ + "' has duplicate uid " +
                      std::to_string(tensor->uid_),
                  error_code_t::INVALID_VALUE};
        }
      }
    }
    if (!unsupported_reason_.empty()) {
      return {false, unsupported_reason_, error_code_t::GRAPH_NOT_SUPPORTED};
    }
    return {true, ""};
  }

  error_t build_operation_graph(mudnnHandle_t handle) {
    try {
      handle_ = handle;
      build_backend_tensors();
      pointwise_descs_.clear();
      matmul_descs_.clear();
      conv_descs_.clear();
      reduction_descs_.clear();
      resample_descs_.clear();
      rng_descs_.clear();
      internal_tensor_descs_.clear();
      std::vector<Operation> ops;
      ops.reserve(nodes_.size());
      for (const auto& node : nodes_) {
        if (node.kind == NodeKind::kPointwise) {
          auto pw_builder = PointWiseDescBuilder()
                                .setMode(node.pointwise_mode)
                                .setMathPrecision(node.compute_data_type);
          if (node.pointwise_has_axis) {
            pw_builder.setAxis(node.pointwise_axis);
          }
          if (node.pointwise_has_relu_lower_clip) {
            pw_builder.setReluLowerClip(node.pointwise_relu_lower_clip);
          }
          if (node.pointwise_has_relu_upper_clip) {
            pw_builder.setReluUpperClip(node.pointwise_relu_upper_clip);
          }
          if (node.pointwise_has_relu_lower_clip_slope) {
            pw_builder.setReluLowerClipSlope(
                node.pointwise_relu_lower_clip_slope);
          }
          pointwise_descs_.push_back(pw_builder.build());
          auto builder =
              OperationBuilder(MUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                  .setXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setYDesc(tensor_descs_.at(node.output.get()))
                  .setPwDesc(pointwise_descs_.back())
                  .setAlpha(node.pointwise_alpha1);
          if (node.inputs.size() > 1 && node.inputs[1]) {
            builder.setBDesc(tensor_descs_.at(node.inputs[1].get()))
                .setAlpha2(node.pointwise_alpha2);
          }
          if (node.inputs.size() > 2 && node.inputs[2]) {
            builder.setTDesc(tensor_descs_.at(node.inputs[2].get()));
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kMatmul) {
          matmul_descs_.push_back(MatMulDescBuilder()
                                      .setMathPrecision(node.compute_data_type)
                                      .build());
          ops.push_back(
              OperationBuilder(MUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                  .setMatmulDesc(matmul_descs_.back())
                  .setAMatDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setBMatDesc(tensor_descs_.at(node.inputs[1].get()))
                  .setCMatDesc(tensor_descs_.at(node.output.get()))
                  .build());
        } else if (node.kind == NodeKind::kConvFprop) {
          conv_descs_.push_back(
              ConvDescBuilder()
                  .setMathPrecision(node.compute_data_type)
                  .setConvolutionMode(node.conv_mode)
                  .setSpatialDimCount(static_cast<int64_t>(node.param0.size()))
                  .setPrePadding(node.param0)
                  .setPostPadding(node.param3.empty() ? node.param0
                                                      : node.param3)
                  .setStride(node.param1)
                  .setDilation(node.param2)
                  .setGroupCount(node.conv_group_count)
                  .build());
          ops.push_back(
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                  .setConvDesc(conv_descs_.back())
                  .setConvXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setConvWDesc(tensor_descs_.at(node.inputs[1].get()))
                  .setConvYDesc(tensor_descs_.at(node.output.get()))
                  .setAlpha(node.alpha)
                  .setAlpha2(node.beta)
                  .build());
        } else if (node.kind == NodeKind::kConvBwdData) {
          conv_descs_.push_back(
              ConvDescBuilder()
                  .setMathPrecision(node.compute_data_type)
                  .setConvolutionMode(node.conv_mode)
                  .setSpatialDimCount(static_cast<int64_t>(node.param0.size()))
                  .setPrePadding(node.param0)
                  .setPostPadding(node.param3.empty() ? node.param0
                                                      : node.param3)
                  .setStride(node.param1)
                  .setDilation(node.param2)
                  .setGroupCount(node.conv_group_count)
                  .build());
          ops.push_back(
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
                  .setConvDesc(conv_descs_.back())
                  .setConvDYDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setConvWDesc(tensor_descs_.at(node.inputs[1].get()))
                  .setConvDXDesc(tensor_descs_.at(node.output.get()))
                  .setAlpha(node.alpha)
                  .setAlpha2(node.beta)
                  .build());
        } else if (node.kind == NodeKind::kConvBwdFilter) {
          conv_descs_.push_back(
              ConvDescBuilder()
                  .setMathPrecision(node.compute_data_type)
                  .setConvolutionMode(node.conv_mode)
                  .setSpatialDimCount(static_cast<int64_t>(node.param0.size()))
                  .setPrePadding(node.param0)
                  .setPostPadding(node.param3.empty() ? node.param0
                                                      : node.param3)
                  .setStride(node.param1)
                  .setDilation(node.param2)
                  .setGroupCount(node.conv_group_count)
                  .build());
          ops.push_back(
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
                  .setConvDesc(conv_descs_.back())
                  .setConvXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setConvDYDesc(tensor_descs_.at(node.inputs[1].get()))
                  .setConvDWDesc(tensor_descs_.at(node.output.get()))
                  .setAlpha(node.alpha)
                  .setAlpha2(node.beta)
                  .build());
        } else if (node.kind == NodeKind::kReduction) {
          reduction_descs_.push_back(
              ReductionDescBuilder()
                  .setReductionOp(node.reduction_mode)
                  .setMathPrecision(node.compute_data_type)
                  .build());
          ops.push_back(
              OperationBuilder(MUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                  .setReductionDesc(reduction_descs_.back())
                  .setReductionXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setReductionYDesc(tensor_descs_.at(node.output.get()))
                  .build());
        } else if (node.kind == NodeKind::kGenstats) {
          ops.push_back(
              OperationBuilder(MUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR)
                  .setGenStatsMode(node.genstats_mode)
                  .setGenStatsMathPrecision(node.compute_data_type)
                  .setGenStatsXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setGenStatsSumDesc(tensor_descs_.at(node.output.get()))
                  .setGenStatsSqSumDesc(tensor_descs_.at(node.stats.get()))
                  .build());
        } else if (node.kind == NodeKind::kRng) {
          rng_descs_.push_back(
              RngDescBuilder()
                  .setDistribution(node.rng_distribution)
                  .setNormalDistMean(node.rng_normal_mean)
                  .setNormalDistStandardDeviation(node.rng_normal_stddev)
                  .setUniformDistMinimum(node.rng_uniform_minimum)
                  .setUniformDistMaximum(node.rng_uniform_maximum)
                  .setBernoulliDistProbability(node.rng_bernoulli_probability)
                  .build());
          auto builder =
              OperationBuilder(MUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                  .setRngDesc(rng_descs_.back())
                  .setRngYDesc(tensor_descs_.at(node.output.get()))
                  .setRngSeed(node.rng_seed);
          if (node.rng_has_offset) {
            builder.setRngOffsetDesc(tensor_descs_.at(node.inputs[0].get()));
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kBlockScaleQuantize) {
          if (!block_scale_is_last_axis(node)) {
            throw std::invalid_argument(
                "native block_scale_quantize supports only last-dimension "
                "block scales");
          }
          if (node.stats->data_type_ != MUDNN_DATA_FLOAT) {
            throw std::invalid_argument(
                "native block_scale_quantize supports "
                "only FLOAT scale tensors");
          }
          ops.push_back(
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR)
                  .setBlockScaleQuantizeXDesc(
                      tensor_descs_.at(node.inputs[0].get()))
                  .setBlockScaleQuantizeYDesc(
                      tensor_descs_.at(node.output.get()))
                  .setBlockScaleQuantizeScaleDesc(
                      tensor_descs_.at(node.stats.get()))
                  .setBlockScaleQuantizeMathPrecision(node.compute_data_type)
                  .setBlockScaleQuantizeBlockSize(node.block_scale_block_size)
                  .build());
        } else if (node.kind == NodeKind::kBlockScaleDequantize) {
          if (node.block_scale_is_negative) {
            throw std::invalid_argument(
                "negative block scales are not supported");
          }
          if (!block_scale_is_last_axis(node)) {
            throw std::invalid_argument(
                "native block_scale_dequantize supports only last-dimension "
                "block scales");
          }
          if (node.inputs[1]->data_type_ != MUDNN_DATA_FLOAT) {
            throw std::invalid_argument(
                "native block_scale_dequantize supports only FLOAT scale "
                "tensors");
          }
          ops.push_back(
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR)
                  .setBlockScaleDequantizeXDesc(
                      tensor_descs_.at(node.inputs[0].get()))
                  .setBlockScaleDequantizeScaleDesc(
                      tensor_descs_.at(node.inputs[1].get()))
                  .setBlockScaleDequantizeYDesc(
                      tensor_descs_.at(node.output.get()))
                  .setBlockScaleDequantizeMathPrecision(node.compute_data_type)
                  .setBlockScaleDequantizeBlockSize(node.block_scale_block_size)
                  .build());
        } else if (node.kind == NodeKind::kMoeGroupedMatmul) {
          auto builder =
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR)
                  .setMoeGroupedMatmulMode(node.moe_grouped_matmul_mode)
                  .setMoeGroupedMatmulMathPrecision(node.compute_data_type)
                  .setMoeGroupedMatmulTokenDesc(
                      tensor_descs_.at(node.inputs[0].get()))
                  .setMoeGroupedMatmulWeightDesc(
                      tensor_descs_.at(node.inputs[1].get()))
                  .setMoeGroupedMatmulFirstTokenOffsetDesc(
                      tensor_descs_.at(node.inputs[2].get()))
                  .setMoeGroupedMatmulOutputDesc(
                      tensor_descs_.at(node.output.get()));
          if (node.inputs.size() > 3 && node.inputs[3]) {
            builder.setMoeGroupedMatmulTokenIndexDesc(
                tensor_descs_.at(node.inputs[3].get()));
          }
          if (node.inputs.size() > 4 && node.inputs[4]) {
            builder.setMoeGroupedMatmulTokenKsDesc(
                tensor_descs_.at(node.inputs[4].get()));
          }
          if (node.moe_grouped_matmul_top_k > 0) {
            builder.setMoeGroupedMatmulTopK(node.moe_grouped_matmul_top_k);
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kMoeGroupedMatmulBwd) {
          ops.push_back(
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR)
                  .setMoeGroupedMatmulBwdMathPrecision(node.compute_data_type)
                  .setMoeGroupedMatmulBwdDOutputDesc(
                      tensor_descs_.at(node.inputs[0].get()))
                  .setMoeGroupedMatmulBwdTokenDesc(
                      tensor_descs_.at(node.inputs[1].get()))
                  .setMoeGroupedMatmulBwdFirstTokenOffsetDesc(
                      tensor_descs_.at(node.inputs[2].get()))
                  .setMoeGroupedMatmulBwdDWeightDesc(
                      tensor_descs_.at(node.output.get()))
                  .build());
        } else if (node.kind == NodeKind::kPagedCacheLoad) {
          ops.push_back(
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR)
                  .setPagedCacheLoadContainerDesc(
                      tensor_descs_.at(node.inputs[0].get()))
                  .setPagedCacheLoadYDesc(tensor_descs_.at(node.output.get()))
                  .setPagedCacheLoadSequenceDesc(
                      tensor_descs_.at(node.inputs[1].get()))
                  .setPagedCacheLoadPageTableDesc(
                      tensor_descs_.at(node.inputs[2].get()))
                  .build());
        } else if (node.kind == NodeKind::kSignal) {
          auto builder =
              OperationBuilder(MUDNN_BACKEND_OPERATION_SIGNAL_DESCRIPTOR)
                  .setSignalMode(node.signal_mode)
                  .setSignalValue(node.signal_value)
                  .setSignalXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setSignalYDesc(tensor_descs_.at(node.output.get()));
          if (node.signal_mode == MUDNN_SIGNAL_SET) {
            builder.setSignalFlagDesc(tensor_descs_.at(node.stats.get()));
          } else {
            builder.setSignalFlagDesc(tensor_descs_.at(node.inputs[1].get()));
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kConcat) {
          std::vector<Tensor> concat_inputs;
          concat_inputs.reserve(node.inputs.size());
          for (const auto& input : node.inputs) {
            concat_inputs.push_back(tensor_descs_.at(input.get()));
          }
          ops.push_back(
              OperationBuilder(MUDNN_BACKEND_OPERATION_CONCAT_DESCRIPTOR)
                  .setConcatAxis(node.concat_axis)
                  .setConcatInputDescs(concat_inputs)
                  .setConcatOutputDesc(tensor_descs_.at(node.output.get()))
                  .build());
        } else if (node.kind == NodeKind::kReshape) {
          ops.push_back(
              OperationBuilder(MUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR)
                  .setReshapeXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setReshapeYDesc(tensor_descs_.at(node.output.get()))
                  .build());
        } else if (node.kind == NodeKind::kTranspose) {
          ops.push_back(
              OperationBuilder(MUDNN_BACKEND_OPERATION_TRANSPOSE_DESCRIPTOR)
                  .setTransposeXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setTransposeYDesc(tensor_descs_.at(node.output.get()))
                  .setTransposePermutation(node.param0)
                  .build());
        } else if (node.kind == NodeKind::kSlice) {
          ops.push_back(
              OperationBuilder(MUDNN_BACKEND_OPERATION_SLICE_DESCRIPTOR)
                  .setSliceXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setSliceYDesc(tensor_descs_.at(node.output.get()))
                  .setSliceStartIndices(node.param0)
                  .setSliceLimitIndices(node.param1)
                  .setSliceStrides(node.param2)
                  .build());
        } else if (node.kind == NodeKind::kResampleFwd) {
          resample_descs_.push_back(
              ResampleDescBuilder()
                  .setMode(node.resample_mode)
                  .setMathPrecision(node.compute_data_type)
                  .setSpatialDimCount(2)
                  .setWindowDim(node.param0)
                  .setPrePadding(node.param1)
                  .setPostPadding(node.param1)
                  .setStride(node.param2)
                  .build());
          auto builder =
              OperationBuilder(MUDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR)
                  .setResampleDesc(resample_descs_.back())
                  .setPoolingXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setPoolingYDesc(tensor_descs_.at(node.output.get()))
                  .setAlpha(node.alpha)
                  .setAlpha2(node.beta);
          if (node.stats) {
            builder.setResampleIndexDesc(tensor_descs_.at(node.stats.get()));
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kResampleBwd) {
          resample_descs_.push_back(
              ResampleDescBuilder()
                  .setMode(node.resample_mode)
                  .setMathPrecision(node.compute_data_type)
                  .setSpatialDimCount(2)
                  .setWindowDim(node.param0)
                  .setPrePadding(node.param1)
                  .setPostPadding(node.param1)
                  .setStride(node.param2)
                  .build());
          auto builder =
              OperationBuilder(MUDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR)
                  .setResampleDesc(resample_descs_.back())
                  .setPoolingDyDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setPoolingDxDesc(tensor_descs_.at(node.output.get()))
                  .setAlpha(node.alpha)
                  .setAlpha2(node.beta);
          size_t input_index = 1;
          if (node.resample_has_indices) {
            builder.setResampleIndexDesc(
                tensor_descs_.at(node.inputs[input_index++].get()));
          }
          if (node.resample_has_x) {
            builder.setPoolingXDesc(
                tensor_descs_.at(node.inputs[input_index++].get()));
          }
          if (node.resample_has_y) {
            builder.setPoolingYDesc(
                tensor_descs_.at(node.inputs[input_index].get()));
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kNormFwd) {
          auto builder =
              OperationBuilder(MUDNN_BACKEND_OPERATION_NORM_FORWARD_DESCRIPTOR)
                  .setNormMode(node.norm_mode)
                  .setNormFwdPhase(node.norm_phase)
                  .setNormFwdXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setNormFwdYDesc(tensor_descs_.at(node.output.get()));
          if (node.norm_has_epsilon_tensor) {
            const size_t epsilon_input_index =
                node.norm_mode == MUDNN_BATCH_NORM ? 5 : node.inputs.size() - 1;
            builder.setNormFwdEpsilon(
                tensor_descs_.at(node.inputs[epsilon_input_index].get()));
          } else {
            builder.setNormFwdEpsilon(node.norm_epsilon);
          }
          if (node.norm_mode == MUDNN_BATCH_NORM) {
            builder.setNormFwdScaleDesc(tensor_descs_.at(node.inputs[3].get()))
                .setNormFwdBiasDesc(tensor_descs_.at(node.inputs[4].get()));
          } else {
            builder.setNormFwdScaleDesc(tensor_descs_.at(node.inputs[1].get()));
            if (node.norm_has_bias) {
              builder.setNormFwdBiasDesc(
                  tensor_descs_.at(node.inputs[2].get()));
            }
          }
          if (node.norm_mode == MUDNN_BATCH_NORM &&
              node.norm_phase == MUDNN_NORM_FWD_TRAINING) {
            builder
                .setNormFwdInputRunningMeanDesc(
                    tensor_descs_.at(node.inputs[1].get()))
                .setNormFwdInputRunningVarDesc(
                    tensor_descs_.at(node.inputs[2].get()))
                .setNormFwdOutputRunningMeanDesc(
                    tensor_descs_.at(node.stats.get()))
                .setNormFwdOutputRunningVarDesc(
                    tensor_descs_.at(node.aux_outputs[0].get()))
                .setNormFwdMeanDesc(tensor_descs_.at(node.aux_outputs[1].get()))
                .setNormFwdInvVarianceDesc(
                    tensor_descs_.at(node.aux_outputs[2].get()));
            if (node.norm_has_momentum_tensor) {
              const size_t momentum_input_index =
                  node.norm_has_epsilon_tensor ? 6 : 5;
              builder.setNormFwdExpAvgFactor(
                  tensor_descs_.at(node.inputs[momentum_input_index].get()));
            } else {
              builder.setNormFwdExpAvgFactor(node.norm_momentum);
            }
          } else if (node.norm_mode == MUDNN_BATCH_NORM) {
            builder.setNormFwdMeanDesc(tensor_descs_.at(node.inputs[1].get()))
                .setNormFwdInvVarianceDesc(
                    tensor_descs_.at(node.inputs[2].get()));
          } else {
            if (node.norm_has_mean) {
              builder.setNormFwdMeanDesc(tensor_descs_.at(node.stats.get()));
            }
            builder.setNormFwdInvVarianceDesc(
                tensor_descs_.at(node.aux_outputs[0].get()));
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kNormBwd) {
          auto builder =
              OperationBuilder(MUDNN_BACKEND_OPERATION_NORM_BACKWARD_DESCRIPTOR)
                  .setNormBwdMode(node.norm_mode)
                  .setNormBwdXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setNormBwdDyDesc(tensor_descs_.at(node.inputs[1].get()))
                  .setNormBwdDxDesc(tensor_descs_.at(node.output.get()))
                  .setNormBwdDScaleDesc(tensor_descs_.at(node.stats.get()));
          if (node.norm_has_epsilon_tensor) {
            builder.setNormBwdEpsilon(
                tensor_descs_.at(node.inputs.back().get()));
          } else {
            builder.setNormBwdEpsilon(node.norm_epsilon);
          }
          if (node.norm_mode == MUDNN_BATCH_NORM) {
            builder.setNormBwdMeanDesc(tensor_descs_.at(node.inputs[2].get()))
                .setNormBwdInvVarianceDesc(
                    tensor_descs_.at(node.inputs[3].get()))
                .setNormBwdScaleDesc(tensor_descs_.at(node.inputs[4].get()))
                .setNormBwdDBiasDesc(
                    tensor_descs_.at(node.aux_outputs[0].get()));
          } else {
            size_t input_index = 2;
            if (node.norm_has_mean) {
              builder.setNormBwdMeanDesc(
                  tensor_descs_.at(node.inputs[input_index++].get()));
            }
            builder
                .setNormBwdInvVarianceDesc(
                    tensor_descs_.at(node.inputs[input_index++].get()))
                .setNormBwdScaleDesc(
                    tensor_descs_.at(node.inputs[input_index].get()));
            if (node.norm_has_dbias) {
              builder.setNormBwdDBiasDesc(
                  tensor_descs_.at(node.aux_outputs[0].get()));
            }
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kSdpa) {
          const auto& q = node.inputs[0];
          const auto& k = node.inputs[1];
          auto builder =
              OperationBuilder(MUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR)
                  .setQDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setKDesc(tensor_descs_.at(node.inputs[1].get()))
                  .setVDesc(tensor_descs_.at(node.inputs[2].get()))
                  .setODesc(tensor_descs_.at(node.output.get()))
                  .setSdpaInference(node.sdpa_is_inference)
                  .setSdpaCausalMask(node.sdpa_causal_mask)
                  .setSdpaUsePaddingMask(node.sdpa_use_padding_mask);
          if (node.sdpa_has_scale) {
            builder.setSdpaScaleDesc(tensor_descs_.at(node.inputs[3].get()));
          }
          if (node.stats) {
            builder.setSdpaFwdStatsDesc(tensor_descs_.at(node.stats.get()));
          }
          size_t input_index = node.sdpa_has_scale ? 4 : 3;
          if (node.sdpa_has_bias) {
            builder.setBiasDesc(
                tensor_descs_.at(node.inputs[input_index++].get()));
          }
          if (node.sdpa_has_mask) {
            builder.setMaskDesc(
                tensor_descs_.at(node.inputs[input_index++].get()));
          }
          if (node.sdpa_has_seq_len_q) {
            builder.setSdpaSeqLenQDesc(
                tensor_descs_.at(node.inputs[input_index++].get()));
          }
          if (node.sdpa_has_seq_len_kv) {
            builder.setSdpaSeqLenKvDesc(
                tensor_descs_.at(node.inputs[input_index].get()));
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kSdpaBwd) {
          if (node.sdpa_has_bias || node.sdpa_has_mask) {
            throw std::runtime_error(
                "sdpa_backward bias/mask lowering is not supported by the "
                "cuDNN 9.23 graph SDPA descriptor");
          }
          auto builder =
              OperationBuilder(MUDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR)
                  .setQDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setKDesc(tensor_descs_.at(node.inputs[1].get()))
                  .setVDesc(tensor_descs_.at(node.inputs[2].get()))
                  .setODesc(tensor_descs_.at(node.inputs[3].get()))
                  .setSdpaBwdDODesc(tensor_descs_.at(node.inputs[4].get()))
                  .setSdpaBwdStatsDesc(tensor_descs_.at(node.inputs[5].get()))
                  .setSdpaBwdDQDesc(tensor_descs_.at(node.output.get()))
                  .setSdpaBwdDKDesc(tensor_descs_.at(node.stats.get()))
                  .setSdpaBwdDVDesc(tensor_descs_.at(node.aux_outputs[0].get()))
                  .setSdpaBwdCausalMask(node.sdpa_causal_mask)
                  .setSdpaBwdUsePaddingMask(node.sdpa_use_padding_mask)
                  .setSdpaBwdDeterministic(node.sdpa_bwd_deterministic);
          if (node.sdpa_has_scale) {
            builder.setSdpaBwdScaleDesc(tensor_descs_.at(node.inputs[6].get()));
          }
          size_t input_index = node.sdpa_has_scale ? 7 : 6;
          if (node.sdpa_has_bias) {
            builder.setBiasDesc(
                tensor_descs_.at(node.inputs[input_index++].get()));
          }
          if (node.sdpa_has_mask) {
            builder.setMaskDesc(
                tensor_descs_.at(node.inputs[input_index].get()));
          }
          if (node.sdpa_has_seq_len_q || node.sdpa_has_seq_len_kv) {
            throw std::runtime_error(
                "sdpa_backward sequence-length native lowering is not "
                "supported");
          }
          if (node.sdpa_bwd_has_d_bias) {
            builder.setSdpaBwdDBiasDesc(
                tensor_descs_.at(node.aux_outputs[1].get()));
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kBnFinalize) {
          auto builder =
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
                  .setBnFinalizeStatsMode(node.bn_finalize_mode)
                  .setBnFinalizeMathPrecision(node.compute_data_type)
                  .setBnFinalizeEpsilon(node.bn_finalize_epsilon);
          if (node.bn_finalize_mode == MUDNN_BN_FINALIZE_STATISTICS_TRAINING) {
            builder.setBnFinalizeYSumDesc(
                tensor_descs_.at(node.inputs[0].get()));
            builder.setBnFinalizeYSqSumDesc(
                tensor_descs_.at(node.inputs[1].get()));
            builder.setBnFinalizeScaleDesc(
                tensor_descs_.at(node.inputs[2].get()));
            builder.setBnFinalizeBiasDesc(
                tensor_descs_.at(node.inputs[3].get()));
            builder.setBnFinalizePrevRunningMeanDesc(
                tensor_descs_.at(node.inputs[4].get()));
            builder.setBnFinalizePrevRunningVarDesc(
                tensor_descs_.at(node.inputs[5].get()));
            builder.setBnFinalizeAccumCountDesc(
                tensor_descs_.at(node.inputs[6].get()));
            builder.setBnFinalizeSavedMeanDesc(
                tensor_descs_.at(node.output.get()));
            builder.setBnFinalizeSavedInvStdDesc(
                tensor_descs_.at(node.stats.get()));
            builder.setBnFinalizeUpdatedRunningMeanDesc(
                tensor_descs_.at(node.aux_outputs[0].get()));
            builder.setBnFinalizeUpdatedRunningVarDesc(
                tensor_descs_.at(node.aux_outputs[1].get()));
            builder.setBnFinalizeEqScaleDesc(
                tensor_descs_.at(node.aux_outputs[2].get()));
            builder.setBnFinalizeEqBiasDesc(
                tensor_descs_.at(node.aux_outputs[3].get()));
            builder.setBnFinalizeExpAvgFactor(node.bn_finalize_momentum);
          } else {
            builder.setBnFinalizeScaleDesc(
                tensor_descs_.at(node.inputs[0].get()));
            builder.setBnFinalizeBiasDesc(
                tensor_descs_.at(node.inputs[1].get()));
            builder.setBnFinalizePrevRunningMeanDesc(
                tensor_descs_.at(node.inputs[2].get()));
            builder.setBnFinalizePrevRunningVarDesc(
                tensor_descs_.at(node.inputs[3].get()));
            builder.setBnFinalizeEqScaleDesc(
                tensor_descs_.at(node.output.get()));
            builder.setBnFinalizeEqBiasDesc(tensor_descs_.at(node.stats.get()));
          }
          ops.push_back(builder.build());
        } else if (node.kind == NodeKind::kBnBwdWeights) {
          ops.push_back(
              OperationBuilder(
                  MUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)
                  .setBnBwdWeightsMathPrecision(node.compute_data_type)
                  .setBnBwdWeightsXDesc(tensor_descs_.at(node.inputs[0].get()))
                  .setBnBwdWeightsDyDesc(tensor_descs_.at(node.inputs[1].get()))
                  .setBnBwdWeightsMeanDesc(
                      tensor_descs_.at(node.inputs[2].get()))
                  .setBnBwdWeightsInvstdDesc(
                      tensor_descs_.at(node.inputs[3].get()))
                  .setBnBwdWeightsBnScaleDesc(
                      tensor_descs_.at(node.inputs[4].get()))
                  .setBnBwdWeightsDBNScaleDesc(
                      tensor_descs_.at(node.output.get()))
                  .setBnBwdWeightsDBNBiasDesc(
                      tensor_descs_.at(node.stats.get()))
                  .setBnBwdWeightsEqDyScaleDesc(
                      tensor_descs_.at(node.aux_outputs[0].get()))
                  .setBnBwdWeightsEqXScaleDesc(
                      tensor_descs_.at(node.aux_outputs[1].get()))
                  .setBnBwdWeightsEqBiasDesc(
                      tensor_descs_.at(node.aux_outputs[2].get()))
                  .build());
        }
      }
      operations_ = std::move(ops);
      operation_graph_ = OperationGraphBuilder()
                             .setHandle(handle_)
                             .setOperationGraph(operations_)
                             .setIsDynamicShapeEnabled(dynamic_shape_enabled_)
                             .setIsOverrideShapeEnabled(override_shape_enabled_)
                             .build();
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t build_operation_graph() {
    if (!device_properties_) {
      return {false, "Device properties are not set.",
              error_code_t::ATTRIBUTE_NOT_SET};
    }
    return build_operation_graph(nullptr);
  }

  error_t create_execution_plans(const std::vector<HeurMode_t>& modes) {
    plan_modes_.clear();
    for (auto mode : modes) {
      plan_modes_.push_back(to_mudnn(mode));
    }
    return {true, ""};
  }

  error_t create_execution_plans(std::initializer_list<HeurMode_t> modes) {
    return create_execution_plans(std::vector<HeurMode_t>(modes));
  }

  error_t build(mudnnHandle_t handle, const std::vector<HeurMode_t>& modes) {
    auto status = validate();
    if (!status.is_good()) {
      return status;
    }
    status = build_operation_graph(handle);
    if (!status.is_good()) {
      return status;
    }
    status = create_execution_plans(modes);
    if (!status.is_good()) {
      return status;
    }
    status = check_support(handle);
    if (!status.is_good()) {
      return status;
    }
    return build_plans();
  }

  error_t build(
      mudnnHandle_t handle, const std::vector<HeurMode_t>& modes,
      BuildPlanPolicy_t policy, bool do_multithreaded_builds = false) {
    auto status = validate();
    if (!status.is_good()) return status;
    status = build_operation_graph(handle);
    if (!status.is_good()) return status;
    status = create_execution_plans(modes);
    if (!status.is_good()) return status;
    status = check_support();
    if (!status.is_good()) return status;
    return build_plans(policy, do_multithreaded_builds);
  }

  error_t build(mudnnHandle_t handle, std::initializer_list<HeurMode_t> modes) {
    return build(handle, std::vector<HeurMode_t>(modes));
  }

  error_t build(const std::vector<HeurMode_t>& modes) {
    if (!device_properties_) {
      return {false, "Device properties are not set.",
              error_code_t::ATTRIBUTE_NOT_SET};
    }
    return build(nullptr, modes, BuildPlanPolicy_t::HEURISTICS_CHOICE, false);
  }

  error_t build(
      const std::vector<HeurMode_t>& modes, BuildPlanPolicy_t policy,
      bool do_multithreaded_builds = false) {
    if (!device_properties_) {
      return {false, "Device properties are not set.",
              error_code_t::ATTRIBUTE_NOT_SET};
    }
    return build(nullptr, modes, policy, do_multithreaded_builds);
  }

  error_t build(std::initializer_list<HeurMode_t> modes) {
    return build(nullptr, std::vector<HeurMode_t>(modes));
  }

  error_t build_plans(
      BuildPlanPolicy_t = BuildPlanPolicy_t::HEURISTICS_CHOICE) {
    try {
      plans_.clear();
      candidate_plan_ = -1;
      ensure_engine_configs();
      plans_.resize(engine_configs_.size());
      for (int64_t i = 0; i < static_cast<int64_t>(engine_configs_.size());
           ++i) {
        if (engine_config_selected_.at(static_cast<size_t>(i))) {
          build_plan_at_index_impl(i);
          if (candidate_plan_ < 0) candidate_plan_ = i;
        }
      }
      if (std::none_of(plans_.begin(), plans_.end(),
                       [](const ExecutionPlan& plan) {
                         return static_cast<bool>(plan);
                       })) {
        return {false, "no execution plans were built"};
      }
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t build_plans(BuildPlanPolicy_t policy, bool /*do_benchmark*/) {
    return build_plans(policy);
  }

  error_t build_plans(
      mudnnHandle_t handle,
      BuildPlanPolicy_t policy = BuildPlanPolicy_t::HEURISTICS_CHOICE) {
    handle_ = handle;
    return build_plans(policy);
  }

  error_t check_support(mudnnHandle_t handle = nullptr) {
    try {
      if (!operation_graph_) {
        auto status =
            build_operation_graph(handle != nullptr ? handle : handle_);
        if (!status.is_good()) {
          return status;
        }
      }
      ensure_engine_configs();
      if (std::none_of(engine_config_selected_.begin(),
                       engine_config_selected_.end(),
                       [](bool selected) { return selected; })) {
        return {false, "all execution plans were deselected",
                error_code_t::GRAPH_NOT_SUPPORTED};
      }
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t warmup(mudnnHandle_t handle = nullptr) {
    try {
      if (handle != nullptr) {
        handle_ = handle;
      }
      if (plans_.empty()) {
        auto status = build_plans();
        if (!status.is_good()) {
          return status;
        }
      }
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  int64_t get_execution_plan_count() {
    ensure_engine_configs();
    return static_cast<int64_t>(engine_configs_.size());
  }

  int64_t get_engine_count() { return get_execution_plan_count(); }

  error_t get_engine_count(int64_t& count) {
    try {
      count = get_engine_count();
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  std::map<std::string, int64_t> get_knobs_for_engine(int64_t index) {
    validate_engine_index(index);
    const auto engine = Engine::Borrowed(
        engine_configs_[static_cast<size_t>(index)].getEngineDescriptor());
    std::map<std::string, int64_t> result;
    for (const auto& knob : engine.getKnobs()) {
      result.emplace(std::to_string(static_cast<int64_t>(knob.type)),
                     knob.minValue);
    }
    return result;
  }

  error_t get_knobs_for_engine(int64_t index, std::vector<Knob>& knobs) {
    try {
      validate_engine_index(index);
      const auto engine = Engine::Borrowed(
          engine_configs_[static_cast<size_t>(index)].getEngineDescriptor());
      knobs = engine.getKnobs();
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  std::pair<int64_t, std::map<std::string, int64_t>>
  get_engine_and_knobs_at_index(int64_t index) {
    validate_engine_index(index);
    const auto engine = Engine::Borrowed(
        engine_configs_[static_cast<size_t>(index)].getEngineDescriptor());
    return {engine.getGlobalIndex(), get_knobs_for_engine(index)};
  }

  error_t get_engine_and_knobs_at_index(
      int64_t index, int64_t& engine_id,
      std::unordered_map<KnobType_t, int64_t>& knobs) {
    try {
      validate_engine_index(index);
      const auto engine = Engine::Borrowed(
          engine_configs_[static_cast<size_t>(index)].getEngineDescriptor());
      engine_id = engine.getGlobalIndex();
      knobs.clear();
      for (const auto& knob : engine.getKnobs()) {
        knobs.emplace(knob.type, knob.minValue);
      }
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t create_execution_plan(
      int64_t index, const std::map<std::string, int64_t>& knobs = {}) {
    try {
      if (!knobs.empty()) {
        return {false,
                "mudnn_frontend does not support custom execution-plan knobs"};
      }
      return build_plan_at_index(index);
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t create_execution_plan(
      int64_t index, const std::unordered_map<KnobType_t, int64_t>& knobs) {
    try {
      if (!knobs.empty()) {
        return {false,
                "mudnn_frontend does not support custom execution-plan knobs"};
      }
      return build_plan_at_index(index);
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t build_plan_at_index(int64_t index, mudnnHandle_t handle = nullptr) {
    try {
      if (handle != nullptr) {
        handle_ = handle;
      }
      ensure_engine_configs();
      validate_engine_index(index);
      while (plans_.size() <= static_cast<size_t>(index)) {
        plans_.push_back(ExecutionPlan());
      }
      build_plan_at_index_impl(index);
      if (candidate_plan_ < 0) candidate_plan_ = index;
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t get_workspace_size_plan_at_index(int64_t index, int64_t& size,
                                           mudnnHandle_t handle = nullptr) {
    try {
      auto status = build_plan_at_index(index, handle);
      if (!status.is_good()) {
        return status;
      }
      size = plans_[static_cast<size_t>(index)].getWorkspaceSize();
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t get_workspace_size_plan_at_index(
      mudnnHandle_t handle, int64_t index, int64_t& size,
      const std::vector<int64_t>&, const std::vector<std::vector<int64_t>>&,
      const std::vector<std::vector<int64_t>>&) {
    return get_workspace_size_plan_at_index(index, size, handle);
  }

  int64_t get_workspace_size_plan_at_index(int64_t index) {
    int64_t size = 0;
    auto result = get_workspace_size_plan_at_index(index, size);
    if (!result.is_good()) {
      throw std::runtime_error(result.get_message());
    }
    return size;
  }

  int64_t get_autotune_workspace_size() const { return get_workspace_size(); }

  error_t autotune(
      mudnnHandle_t,
      const std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>&,
      void* = nullptr) {
    return {false,
            "muDNN backend does not expose execution-plan timing for autotune",
            error_code_t::GRAPH_NOT_SUPPORTED};
  }

  error_t autotune(mudnnHandle_t, const std::unordered_map<int64_t, void*>&,
                   void* = nullptr) {
    return {false,
            "muDNN backend does not expose execution-plan timing for autotune",
            error_code_t::GRAPH_NOT_SUPPORTED};
  }

  std::string get_plan_name_at_index(int64_t index) {
    validate_engine_index(index);
    return "mudnn_engine_config_" + std::to_string(index);
  }

  error_t get_plan_name_at_index(int64_t index, std::string& name) {
    try {
      name = get_plan_name_at_index(index);
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t get_plan_name(std::string& name) const {
    if (candidate_plan_ < 0) {
      return {false, "no candidate execution plan has been selected",
              error_code_t::ATTRIBUTE_NOT_SET};
    }
    name = "mudnn_engine_config_" + std::to_string(candidate_plan_);
    return {true, ""};
  }

  Graph& select_behavior_notes(const std::vector<BehaviorNote_t>& notes) {
    selected_behavior_notes_ = notes;
    refresh_engine_selection();
    return *this;
  }

  Graph& deselect_behavior_notes(const std::vector<BehaviorNote_t>& notes) {
    deselected_behavior_notes_ = notes;
    refresh_engine_selection();
    return *this;
  }

  Graph& select_numeric_notes(const std::vector<NumericalNote_t>& notes) {
    selected_numeric_notes_ = notes;
    refresh_engine_selection();
    return *this;
  }

  Graph& deselect_numeric_notes(const std::vector<NumericalNote_t>& notes) {
    deselected_numeric_notes_ = notes;
    refresh_engine_selection();
    return *this;
  }

  std::vector<BehaviorNote_t> get_behavior_notes() const {
    if (candidate_plan_ < 0 ||
        candidate_plan_ >= static_cast<int64_t>(engine_configs_.size())) {
      return {};
    }
    auto engine = Engine::Borrowed(
        engine_configs_[static_cast<size_t>(candidate_plan_)]
            .getEngineDescriptor());
    return engine.getBehaviorNotes();
  }

  std::vector<BehaviorNote_t> get_behavior_notes_for_plan_at_index(
      int64_t index) {
    validate_engine_index(index);
    auto engine = Engine::Borrowed(
        engine_configs_[static_cast<size_t>(index)].getEngineDescriptor());
    return engine.getBehaviorNotes();
  }

  std::vector<uint8_t> serialize() const {
    return serialize_impl(/*include_plan_modes=*/true,
                          /*include_shapes=*/true);
  }

  // Upstream cudnn-frontend semantics: the graph key is stable across
  // create_execution_plans/check_support/build_plans, so plan modes are
  // excluded; with dynamic shapes enabled, tensor dims/strides are excluded
  // as well so graphs differing only in shape share the same key.
  size_t key() const {
    const auto data =
        serialize_impl(/*include_plan_modes=*/false,
                       /*include_shapes=*/!dynamic_shape_enabled_);
    return std::hash<std::string>{}(std::string(data.begin(), data.end()));
  }

 private:
  std::vector<uint8_t> serialize_impl(bool include_plan_modes,
                                      bool include_shapes) const {
    std::ostringstream out;
    out << "mudnn_frontend.graph 22\n";
    out << static_cast<int64_t>(io_data_type_) << ' '
        << static_cast<int64_t>(intermediate_data_type_) << ' '
        << static_cast<int64_t>(compute_data_type_) << ' ' << next_uid_ << '\n';
    out << std::quoted(name_) << ' ' << sm_count_ << ' ' << sm_version_ << ' '
        << dynamic_shape_enabled_ << ' ' << override_shape_enabled_ << '\n';
    if (include_plan_modes) {
      out << plan_modes_.size();
      for (auto mode : plan_modes_) {
        out << ' ' << static_cast<int64_t>(mode);
      }
    } else {
      out << 0;
    }
    out << '\n';

    std::map<const Tensor_attributes*, size_t> tensor_indices;
    out << tensors_.size() << '\n';
    for (size_t i = 0; i < tensors_.size(); ++i) {
      const auto& tensor = tensors_[i];
      tensor_indices.emplace(tensor.get(), i);
      out << std::quoted(tensor->name_) << ' ' << tensor->uid_ << ' '
          << tensor->has_uid_ << ' ' << tensor->is_output_ << ' '
          << tensor->is_virtual_ << ' ' << tensor->is_by_value_ << ' '
          << static_cast<int64_t>(tensor->data_type_) << ' ';
      if (include_shapes) {
        write_int_vector(out, tensor->dims_);
        out << ' ';
        write_int_vector(out, tensor->strides_);
      } else {
        out << "0 0";
      }
      out << '\n';
    }

    out << nodes_.size() << '\n';
    for (const auto& node : nodes_) {
      out << static_cast<int>(node.kind) << ' ' << std::quoted(node.name) << ' '
          << node.inputs.size();
      for (const auto& input : node.inputs) {
        out << ' ' << tensor_indices.at(input.get());
      }
      out << ' ' << tensor_indices.at(node.output.get()) << ' '
          << (node.stats
                  ? static_cast<int64_t>(tensor_indices.at(node.stats.get()))
                  : int64_t{-1})
          << ' ' << static_cast<int64_t>(node.pointwise_mode) << ' '
          << static_cast<int64_t>(node.compute_data_type) << ' '
          << node.pointwise_alpha1 << ' ' << node.pointwise_alpha2 << ' '
          << node.pointwise_axis << ' ' << node.pointwise_has_axis << ' '
          << node.sdpa_scale << ' ' << node.sdpa_has_scale << ' '
          << node.sdpa_causal_mask << ' ' << node.sdpa_is_inference << ' '
          << node.sdpa_use_padding_mask << ' ' << node.sdpa_has_bias << ' '
          << node.sdpa_has_mask << ' ' << node.concat_axis << ' '
          << static_cast<int64_t>(node.reduction_mode) << ' '
          << static_cast<int64_t>(node.genstats_mode) << ' '
          << static_cast<int64_t>(node.rng_distribution) << ' ' << node.rng_seed
          << ' ' << node.rng_normal_mean << ' ' << node.rng_normal_stddev << ' '
          << node.rng_uniform_minimum << ' ' << node.rng_uniform_maximum << ' '
          << node.rng_bernoulli_probability << ' ' << node.rng_has_offset << ' '
          << static_cast<int64_t>(node.bn_finalize_mode) << ' '
          << node.bn_finalize_epsilon << ' ' << node.bn_finalize_momentum << ' '
          << node.aux_outputs.size();
      for (const auto& output : node.aux_outputs) {
        out << ' ' << tensor_indices.at(output.get());
      }
      out << ' ' << static_cast<int64_t>(node.conv_mode) << ' '
          << static_cast<int64_t>(node.resample_mode) << ' ' << node.alpha
          << ' ' << node.beta << ' ' << node.resample_has_indices << ' '
          << node.resample_has_x << ' ' << node.resample_has_y << ' ';
      write_int_vector(out, node.param0);
      out << ' ';
      write_int_vector(out, node.param1);
      out << ' ';
      write_int_vector(out, node.param2);
      out << ' ';
      write_int_vector(out, node.param3);
      out << ' ' << static_cast<int64_t>(node.norm_mode) << ' '
          << static_cast<int64_t>(node.norm_phase) << ' ' << node.norm_epsilon
          << ' ' << node.norm_momentum << ' ' << node.block_scale_block_size
          << ' ' << node.block_scale_block_axis << ' '
          << static_cast<int64_t>(node.signal_mode) << ' ' << node.signal_value
          << ' ' << node.sdpa_bwd_deterministic << ' '
          << node.sdpa_bwd_has_d_bias << ' ' << node.norm_has_mean << ' '
          << node.norm_has_bias << ' ' << node.norm_has_dbias << ' '
          << node.norm_has_epsilon_tensor << ' '
          << node.norm_has_momentum_tensor << ' '
          << node.block_scale_is_negative << ' ' << node.conv_group_count << ' '
          << node.sdpa_has_seq_len_q << ' ' << node.sdpa_has_seq_len_kv << ' '
          << static_cast<int64_t>(node.moe_grouped_matmul_mode) << ' '
          << node.moe_grouped_matmul_top_k << ' '
          << node.pointwise_relu_lower_clip << ' '
          << node.pointwise_relu_upper_clip << ' '
          << node.pointwise_relu_lower_clip_slope << ' '
          << node.pointwise_has_relu_lower_clip << ' '
          << node.pointwise_has_relu_upper_clip << ' '
          << node.pointwise_has_relu_lower_clip_slope;
      out << '\n';
    }

    const auto str = out.str();
    return std::vector<uint8_t>(str.begin(), str.end());
  }

 public:
  std::string print() const {
    const auto data = serialize();
    return std::string(data.begin(), data.end());
  }

  error_t serialize(std::vector<uint8_t>& data) const {
    try {
      data = serialize();
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t deserialize(mudnnHandle_t handle, const std::vector<uint8_t>& data) {
    auto status = deserialize(data);
    if (!status.is_good()) {
      return status;
    }
    if (handle != nullptr && !nodes_.empty()) {
      status = build_operation_graph(handle);
      if (!status.is_good()) {
        return status;
      }
    }
    return {true, ""};
  }

  error_t deserialize(const std::vector<uint8_t>& data) {
    return deserialize(std::string(data.begin(), data.end()));
  }

  error_t deserialize(const std::string& data) {
    try {
      std::istringstream in(data);
      std::string format;
      int version = 0;
      in >> format >> version;
      if (format != "mudnn_frontend.graph" ||
          (version != 1 && version != 2 && version != 3 && version != 4 &&
           version != 5 && version != 6 && version != 7 && version != 8 &&
           version != 9 && version != 10 && version != 11 && version != 12 &&
           version != 13 && version != 14 && version != 15 && version != 16 &&
           version != 17 && version != 18 && version != 19 && version != 20 &&
           version != 21 && version != 22)) {
        return {false, "unsupported serialized graph format"};
      }

      int64_t io_type = 0;
      int64_t intermediate_type = 0;
      int64_t compute_type = 0;
      in >> io_type >> intermediate_type >> compute_type >> next_uid_;
      io_data_type_ = static_cast<mudnnDataType_t>(io_type);
      intermediate_data_type_ = static_cast<mudnnDataType_t>(intermediate_type);
      compute_data_type_ = static_cast<mudnnDataType_t>(compute_type);
      if (version >= 22) {
        in >> std::quoted(name_) >> sm_count_ >> sm_version_ >>
            dynamic_shape_enabled_ >> override_shape_enabled_;
      }

      size_t plan_mode_count = 0;
      in >> plan_mode_count;
      plan_modes_.clear();
      for (size_t i = 0; i < plan_mode_count; ++i) {
        int64_t mode = 0;
        in >> mode;
        plan_modes_.push_back(static_cast<mudnnBackendHeurMode_t>(mode));
      }

      size_t tensor_count = 0;
      in >> tensor_count;
      tensors_.clear();
      tensors_.reserve(tensor_count);
      for (size_t i = 0; i < tensor_count; ++i) {
        auto tensor = std::make_shared<Tensor_attributes>();
        int64_t dtype = 0;
        in >> std::quoted(tensor->name_) >> tensor->uid_ >> tensor->has_uid_ >>
            tensor->is_output_ >> tensor->is_virtual_ >> tensor->is_by_value_ >>
            dtype;
        tensor->data_type_ = static_cast<mudnnDataType_t>(dtype);
        tensor->dims_ = read_int_vector(in);
        tensor->strides_ = read_int_vector(in);
        tensors_.push_back(std::move(tensor));
      }

      size_t node_count = 0;
      in >> node_count;
      nodes_.clear();
      nodes_.reserve(node_count);
      for (size_t i = 0; i < node_count; ++i) {
        int kind = 0;
        size_t input_count = 0;
        int64_t output_index = -1;
        int64_t stats_index = -1;
        int64_t pointwise_mode = 0;
        int64_t compute_type_node = 0;
        Node node{};
        in >> kind >> std::quoted(node.name) >> input_count;
        node.kind = static_cast<NodeKind>(kind);
        node.inputs.reserve(input_count);
        for (size_t j = 0; j < input_count; ++j) {
          size_t input_index = 0;
          in >> input_index;
          node.inputs.push_back(tensors_.at(input_index));
        }
        in >> output_index >> stats_index >> pointwise_mode >>
            compute_type_node >> node.pointwise_alpha1 >>
            node.pointwise_alpha2 >> node.pointwise_axis >>
            node.pointwise_has_axis >> node.sdpa_scale >> node.sdpa_has_scale >>
            node.sdpa_causal_mask >> node.sdpa_is_inference >>
            node.sdpa_use_padding_mask >> node.sdpa_has_bias >>
            node.sdpa_has_mask;
        if (version >= 2) {
          in >> node.concat_axis;
        }
        if (version >= 3) {
          int64_t reduction_mode = MUDNN_REDUCE_TENSOR_ADD;
          in >> reduction_mode;
          node.reduction_mode =
              static_cast<mudnnReduceTensorOp_t>(reduction_mode);
        }
        if (version >= 4) {
          int64_t genstats_mode = MUDNN_GENSTATS_SUM_SQSUM;
          in >> genstats_mode;
          node.genstats_mode = static_cast<mudnnGenStatsMode_t>(genstats_mode);
        }
        if (version >= 5) {
          int64_t rng_distribution = MUDNN_RNG_DISTRIBUTION_UNIFORM;
          in >> rng_distribution >> node.rng_seed >> node.rng_normal_mean >>
              node.rng_normal_stddev >> node.rng_uniform_minimum >>
              node.rng_uniform_maximum >> node.rng_bernoulli_probability >>
              node.rng_has_offset;
          node.rng_distribution =
              static_cast<mudnnRngDistribution_t>(rng_distribution);
        }
        if (version >= 6) {
          int64_t bn_finalize_mode = MUDNN_BN_FINALIZE_STATISTICS_INFERENCE;
          size_t aux_output_count = 0;
          in >> bn_finalize_mode >> node.bn_finalize_epsilon >>
              node.bn_finalize_momentum >> aux_output_count;
          node.bn_finalize_mode =
              static_cast<mudnnBnFinalizeStatsMode_t>(bn_finalize_mode);
          node.aux_outputs.reserve(aux_output_count);
          for (size_t j = 0; j < aux_output_count; ++j) {
            size_t aux_output_index = 0;
            in >> aux_output_index;
            node.aux_outputs.push_back(tensors_.at(aux_output_index));
          }
        }
        if (version >= 7) {
          int64_t conv_mode = MUDNN_CROSS_CORRELATION;
          int64_t resample_mode = MUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
          in >> conv_mode >> resample_mode >> node.alpha >> node.beta >>
              node.resample_has_indices >> node.resample_has_x >>
              node.resample_has_y;
          node.conv_mode = static_cast<mudnnConvolutionMode_t>(conv_mode);
          node.resample_mode = static_cast<mudnnResampleMode_t>(resample_mode);
          node.param0 = read_int_vector(in);
          node.param1 = read_int_vector(in);
          node.param2 = read_int_vector(in);
          if (version >= 21) {
            node.param3 = read_int_vector(in);
          } else {
            node.param3 = node.param0;
          }
        }
        if (version >= 8) {
          int64_t norm_mode = MUDNN_BATCH_NORM;
          int64_t norm_phase = MUDNN_NORM_FWD_INFERENCE;
          in >> norm_mode >> norm_phase >> node.norm_epsilon >>
              node.norm_momentum;
          node.norm_mode = static_cast<mudnnBackendNormMode_t>(norm_mode);
          node.norm_phase = static_cast<mudnnBackendNormFwdPhase_t>(norm_phase);
        }
        if (version >= 9) {
          in >> node.block_scale_block_size;
        }
        if (version >= 18) {
          in >> node.block_scale_block_axis;
        } else if (!node.inputs.empty()) {
          node.block_scale_block_axis =
              static_cast<int64_t>(node.inputs[0]->dims_.size()) - 1;
        }
        if (version >= 10) {
          int64_t signal_mode = MUDNN_SIGNAL_SET;
          in >> signal_mode >> node.signal_value;
          node.signal_mode = static_cast<mudnnSignalMode_t>(signal_mode);
        }
        if (version >= 11) {
          in >> node.sdpa_bwd_deterministic >> node.sdpa_bwd_has_d_bias;
        } else if (node.kind == NodeKind::kSdpaBwd) {
          node.sdpa_bwd_has_d_bias = node.aux_outputs.size() > 1;
        }
        if (version >= 12) {
          in >> node.norm_has_mean >> node.norm_has_bias >> node.norm_has_dbias;
        }
        if (version >= 13) {
          in >> node.norm_has_epsilon_tensor;
        }
        if (version >= 14) {
          in >> node.norm_has_momentum_tensor;
        }
        if (version >= 15) {
          in >> node.block_scale_is_negative;
        }
        if (version >= 16) {
          in >> node.conv_group_count;
        }
        if (version >= 17) {
          in >> node.sdpa_has_seq_len_q >> node.sdpa_has_seq_len_kv;
        }
        if (version >= 19) {
          int64_t moe_mode = MUDNN_MOE_GROUPED_MATMUL_MODE_NONE;
          in >> moe_mode >> node.moe_grouped_matmul_top_k;
          node.moe_grouped_matmul_mode =
              static_cast<mudnnMoeGroupedMatmulMode_t>(moe_mode);
        }
        if (version >= 20) {
          in >> node.pointwise_relu_lower_clip >>
              node.pointwise_relu_upper_clip >>
              node.pointwise_relu_lower_clip_slope >>
              node.pointwise_has_relu_lower_clip >>
              node.pointwise_has_relu_upper_clip >>
              node.pointwise_has_relu_lower_clip_slope;
        }
        node.output = tensors_.at(static_cast<size_t>(output_index));
        if (stats_index >= 0) {
          node.stats = tensors_.at(static_cast<size_t>(stats_index));
        }
        node.pointwise_mode = static_cast<mudnnPointwiseMode_t>(pointwise_mode);
        node.compute_data_type =
            static_cast<mudnnDataType_t>(compute_type_node);
        nodes_.push_back(std::move(node));
      }

      if (!in) {
        return {false, "invalid serialized graph payload"};
      }
      clear_compiled_state();
      return {true, ""};
    } catch (const std::exception& e) {
      clear_compiled_state();
      return {false, e.what()};
    }
  }

#ifdef MUDNN_FRONTEND_HAS_NLOHMANN_JSON
  error_t deserialize(const nlohmann::json& data) {
    try {
      if (data.is_string()) {
        return deserialize(data.get<std::string>());
      }
      if (data.is_object() && data.contains("payload")) {
        return deserialize(data.at("payload").get<std::string>());
      }
      if (data.is_object() && data.contains("mudnn_payload")) {
        return deserialize(data.at("mudnn_payload").get<std::string>());
      }
      if (!data.is_object() || !data.contains("json_version") ||
          !data.contains("tensors") || !data.contains("nodes")) {
        return {false, "unsupported serialized graph json payload"};
      }

      tensors_.clear();
      nodes_.clear();
      next_uid_ = 1;
      const auto& context = data.value("context", nlohmann::json::object());
      name_ = context.value("name", std::string{});
      if (context.contains("io_data_type")) {
        io_data_type_ = upstream_data_type_from_json(context["io_data_type"]);
      }
      if (context.contains("intermediate_data_type")) {
        intermediate_data_type_ =
            upstream_data_type_from_json(context["intermediate_data_type"]);
      }
      if (context.contains("compute_data_type")) {
        compute_data_type_ =
            upstream_data_type_from_json(context["compute_data_type"]);
      }
      sm_count_ = context.value("sm_count", int64_t{-1});
      dynamic_shape_enabled_ =
          context.value("is_dynamic_shape_enabled", false);
      override_shape_enabled_ =
          context.value("is_override_shape_enabled", false);

      std::unordered_map<std::string, std::shared_ptr<Tensor_attributes>>
          tensors_by_key;
      for (const auto& entry : data.at("tensors").items()) {
        auto tensor = std::make_shared<Tensor_attributes>();
        from_json(entry.value(), *tensor);
        if (!tensor->has_uid()) tensor->set_uid(next_uid_);
        next_uid_ = std::max(next_uid_, tensor->get_uid() + 1);
        tensors_by_key.emplace(entry.key(), tensor);
        tensors_by_key.emplace(tensor->get_name(), tensor);
        tensors_by_key.emplace(std::to_string(tensor->get_uid()), tensor);
        tensors_.push_back(tensor);
      }

      const auto resolve = [&tensors_by_key](const nlohmann::json& value) {
        const auto key = value.is_number_integer()
                             ? std::to_string(value.get<int64_t>())
                             : value.get<std::string>();
        auto found = tensors_by_key.find(key);
        if (found == tensors_by_key.end()) {
          throw std::invalid_argument("serialized graph references tensor " + key);
        }
        return found->second;
      };
      const auto pointwise_mode = [](const nlohmann::json& value) {
        if (value.is_number_integer()) {
          return static_cast<mudnnPointwiseMode_t>(value.get<int64_t>());
        }
        const auto name = value.get<std::string>();
#define MUDNN_FRONTEND_PARSE_POINTWISE(enum_name) \
  if (name == #enum_name) return MUDNN_POINTWISE_##enum_name
        MUDNN_FRONTEND_PARSE_POINTWISE(ADD);
        MUDNN_FRONTEND_PARSE_POINTWISE(MUL);
        MUDNN_FRONTEND_PARSE_POINTWISE(MIN);
        MUDNN_FRONTEND_PARSE_POINTWISE(MAX);
        MUDNN_FRONTEND_PARSE_POINTWISE(SQRT);
        MUDNN_FRONTEND_PARSE_POINTWISE(ADD_SQUARE);
        MUDNN_FRONTEND_PARSE_POINTWISE(DIV);
        MUDNN_FRONTEND_PARSE_POINTWISE(MOD);
        MUDNN_FRONTEND_PARSE_POINTWISE(POW);
        MUDNN_FRONTEND_PARSE_POINTWISE(SUB);
        MUDNN_FRONTEND_PARSE_POINTWISE(ABS);
        MUDNN_FRONTEND_PARSE_POINTWISE(CEIL);
        MUDNN_FRONTEND_PARSE_POINTWISE(COS);
        MUDNN_FRONTEND_PARSE_POINTWISE(EXP);
        MUDNN_FRONTEND_PARSE_POINTWISE(FLOOR);
        MUDNN_FRONTEND_PARSE_POINTWISE(LOG);
        MUDNN_FRONTEND_PARSE_POINTWISE(NEG);
        MUDNN_FRONTEND_PARSE_POINTWISE(RSQRT);
        MUDNN_FRONTEND_PARSE_POINTWISE(SIN);
        MUDNN_FRONTEND_PARSE_POINTWISE(TAN);
        MUDNN_FRONTEND_PARSE_POINTWISE(ERF);
        MUDNN_FRONTEND_PARSE_POINTWISE(IDENTITY);
        MUDNN_FRONTEND_PARSE_POINTWISE(RECIPROCAL);
        MUDNN_FRONTEND_PARSE_POINTWISE(RELU_FWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(TANH_FWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(SIGMOID_FWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(ELU_FWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(GELU_FWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(SOFTPLUS_FWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(SWISH_FWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(GELU_APPROX_TANH_FWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(RELU_BWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(TANH_BWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(SIGMOID_BWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(ELU_BWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(GELU_BWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(SOFTPLUS_BWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(SWISH_BWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(GELU_APPROX_TANH_BWD);
        MUDNN_FRONTEND_PARSE_POINTWISE(GEN_INDEX);
        MUDNN_FRONTEND_PARSE_POINTWISE(BINARY_SELECT);
        MUDNN_FRONTEND_PARSE_POINTWISE(CMP_EQ);
        MUDNN_FRONTEND_PARSE_POINTWISE(CMP_NEQ);
        MUDNN_FRONTEND_PARSE_POINTWISE(CMP_GT);
        MUDNN_FRONTEND_PARSE_POINTWISE(CMP_GE);
        MUDNN_FRONTEND_PARSE_POINTWISE(CMP_LT);
        MUDNN_FRONTEND_PARSE_POINTWISE(CMP_LE);
        MUDNN_FRONTEND_PARSE_POINTWISE(LOGICAL_AND);
        MUDNN_FRONTEND_PARSE_POINTWISE(LOGICAL_OR);
        MUDNN_FRONTEND_PARSE_POINTWISE(LOGICAL_NOT);
#undef MUDNN_FRONTEND_PARSE_POINTWISE
        throw std::invalid_argument("unknown pointwise mode " + name);
      };

      for (const auto& item : data.at("nodes")) {
        Node node{};
        node.name = item.value("name", std::string{});
        node.compute_data_type = item.contains("compute_data_type")
                                     ? upstream_data_type_from_json(
                                           item["compute_data_type"])
                                     : compute_data_type_;
        const auto& inputs = item.at("inputs");
        const auto& outputs = item.at("outputs");
        const auto tag = item.at("tag").get<std::string>();
        if (tag == "POINTWISE") {
          node.kind = NodeKind::kPointwise;
          node.inputs.push_back(resolve(inputs.at("IN_0")));
          if (inputs.contains("IN_1")) node.inputs.push_back(resolve(inputs.at("IN_1")));
          if (inputs.contains("IN_2")) node.inputs.push_back(resolve(inputs.at("IN_2")));
          node.output = resolve(outputs.at("OUT_0"));
          node.pointwise_mode = pointwise_mode(item.at("mode"));
          if (item.contains("axis") && !item["axis"].is_null()) {
            node.pointwise_axis = item["axis"].get<int64_t>();
            node.pointwise_has_axis = true;
          }
        } else if (tag == "MATMUL") {
          node.kind = NodeKind::kMatmul;
          node.inputs = {resolve(inputs.at("A")), resolve(inputs.at("B"))};
          node.output = resolve(outputs.at("C"));
          node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
        } else if (tag == "CONV_FPROP" || tag == "CONV_DGRAD" ||
                   tag == "CONV_WGRAD") {
          node.kind = tag == "CONV_FPROP"
                          ? NodeKind::kConvFprop
                          : (tag == "CONV_DGRAD" ? NodeKind::kConvBwdData
                                                  : NodeKind::kConvBwdFilter);
          node.inputs = tag == "CONV_FPROP"
                            ? std::vector<std::shared_ptr<Tensor_attributes>>{
                                  resolve(inputs.at("X")), resolve(inputs.at("W"))}
                            : (tag == "CONV_DGRAD"
                                   ? std::vector<std::shared_ptr<Tensor_attributes>>{
                                         resolve(inputs.at("DY")), resolve(inputs.at("W"))}
                                   : std::vector<std::shared_ptr<Tensor_attributes>>{
                                         resolve(inputs.at("DY")), resolve(inputs.at("X"))});
          node.output = resolve(outputs.at(tag == "CONV_FPROP"
                                               ? "Y"
                                               : (tag == "CONV_DGRAD" ? "DX" : "DW")));
          node.param0 = item.value("pre_padding", std::vector<int64_t>{});
          node.param1 = item.value("post_padding", std::vector<int64_t>{});
          node.param2 = item.value("stride", std::vector<int64_t>{});
          node.param3 = item.value("dilation", std::vector<int64_t>{});
          node.conv_mode = item.value("math_mode", std::string("CROSS_CORRELATION")) == "CONVOLUTION"
                               ? MUDNN_CONVOLUTION
                               : MUDNN_CROSS_CORRELATION;
          node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
        } else if (tag == "REDUCTION") {
          node.kind = NodeKind::kReduction;
          node.inputs = {resolve(inputs.at("X"))};
          node.output = resolve(outputs.at("Y"));
          const auto mode = item.at("mode").get<std::string>();
          if (mode == "ADD") node.reduction_mode = MUDNN_REDUCE_TENSOR_ADD;
          else if (mode == "MUL") node.reduction_mode = MUDNN_REDUCE_TENSOR_MUL;
          else if (mode == "MIN") node.reduction_mode = MUDNN_REDUCE_TENSOR_MIN;
          else if (mode == "MAX") node.reduction_mode = MUDNN_REDUCE_TENSOR_MAX;
          else if (mode == "AMAX") node.reduction_mode = MUDNN_REDUCE_TENSOR_AMAX;
          else if (mode == "AVG") node.reduction_mode = MUDNN_REDUCE_TENSOR_AVG;
          else throw std::invalid_argument("unknown reduction mode " + mode);
          node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
        } else if (tag == "GENSTATS") {
          node.kind = NodeKind::kGenstats;
          node.inputs = {resolve(inputs.at("X"))};
          node.output = resolve(outputs.at("SUM"));
          node.stats = resolve(outputs.at("SQ_SUM"));
          node.genstats_mode = MUDNN_GENSTATS_SUM_SQSUM;
          node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
        } else {
          return {false, "unsupported upstream JSON node tag " + tag,
                  error_code_t::GRAPH_NOT_SUPPORTED};
        }
        nodes_.push_back(std::move(node));
      }
      clear_compiled_state();
      return {true, ""};
    } catch (const std::exception& e) {
      clear_compiled_state();
      return {false, e.what()};
    }
  }

  // Mirrors the deserialize(handle, data) blob overload: rebuild the
  // operation graph on the given handle after a successful deserialize.
  error_t deserialize(mudnnHandle_t handle, const nlohmann::json& data) {
    auto status = deserialize(data);
    if (!status.is_good()) {
      return status;
    }
    if (handle != nullptr && !nodes_.empty()) {
      status = build_operation_graph(handle);
      if (!status.is_good()) {
        return status;
      }
    }
    return {true, ""};
  }
#endif

  std::vector<int64_t> get_variant_pack_uids_sorted() const {
    std::vector<int64_t> uids;
    for (const auto& tensor : tensors_) {
      if (tensor && tensor->has_uid_ &&
          (!tensor->is_virtual_ || tensor->is_output_)) {
        uids.push_back(tensor->uid_);
      }
    }
    std::sort(uids.begin(), uids.end());
    uids.erase(std::unique(uids.begin(), uids.end()), uids.end());
    return uids;
  }

  error_t get_workspace_size(int64_t& size) const {
    size = 0;
    for (const auto& plan : plans_) {
      if (plan) {
        size = std::max(size, plan.getWorkspaceSize());
      }
    }
    return {true, ""};
  }

  int64_t get_workspace_size() const {
    int64_t size = 0;
    auto result = get_workspace_size(size);
    if (!result.is_good()) {
      throw std::runtime_error(result.get_message());
    }
    return size;
  }

  error_t get_workspace_size(mudnnHandle_t, int64_t& size,
                             const std::vector<int64_t>&,
                             const std::vector<std::vector<int64_t>>&,
                             const std::vector<std::vector<int64_t>>&) const {
    return get_workspace_size(size);
  }

  error_t execute(mudnnHandle_t handle,
                  const std::unordered_map<std::shared_ptr<Tensor_attributes>,
                                           void*>& variant_pack,
                  void* workspace = nullptr) const {
    return execute_plan_at_index(handle, variant_pack, workspace,
                                 candidate_plan_ >= 0 ? candidate_plan_ : 0);
  }

  error_t execute(mudnnHandle_t handle,
                  const std::unordered_map<int64_t, void*>& variant_pack,
                  void* workspace = nullptr) const {
    return execute_plan_at_index(handle, variant_pack, workspace,
                                 candidate_plan_ >= 0 ? candidate_plan_ : 0);
  }

  error_t execute(mudnnHandle_t handle,
                  const std::unordered_map<std::shared_ptr<Tensor_attributes>,
                                           void*>& variant_pack,
                  void* workspace, const std::vector<int64_t>&,
                  const std::vector<std::vector<int64_t>>&,
                  const std::vector<std::vector<int64_t>>&) const {
    return execute(handle, variant_pack, workspace);
  }

  error_t execute(mudnnHandle_t handle,
                  const std::unordered_map<int64_t, void*>& variant_pack,
                  void* workspace, const std::vector<int64_t>&,
                  const std::vector<std::vector<int64_t>>&,
                  const std::vector<std::vector<int64_t>>&) const {
    return execute(handle, variant_pack, workspace);
  }

  error_t execute_plan_at_index(
      mudnnHandle_t handle,
      const std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>&
          variant_pack,
      void* workspace = nullptr, int64_t index = 0) const {
    try {
      execute_plan_at_index_impl(handle, variant_pack, workspace, index);
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t execute_plan_at_index(
      mudnnHandle_t handle,
      const std::unordered_map<int64_t, void*>& variant_pack,
      void* workspace = nullptr, int64_t index = 0) const {
    try {
      execute_plan_at_index_impl(handle, variant_pack, workspace, index);
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  error_t populate_cuda_graph(
      mudnnHandle_t handle,
      const std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>&
          variant_pack,
      void* workspace, musaGraph_t graph, int64_t index = 0) const {
    return populate_musa_graph_impl(handle, variant_pack, workspace, graph,
                                    index);
  }

  error_t populate_cuda_graph(
      mudnnHandle_t handle,
      const std::unordered_map<int64_t, void*>& variant_pack, void* workspace,
      musaGraph_t graph, int64_t index = 0) const {
    return populate_musa_graph_impl(handle, variant_pack, workspace, graph,
                                    index);
  }

  template <typename GraphT,
            typename = std::enable_if_t<
                std::is_pointer_v<std::decay_t<GraphT>> &&
                !std::is_same_v<std::decay_t<GraphT>, musaGraph_t>>>
  error_t populate_cuda_graph(
      mudnnHandle_t handle,
      const std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>&
          variant_pack,
      void* workspace, GraphT graph, int64_t index = 0) const {
    return populate_musa_graph_impl(handle, variant_pack, workspace,
                                    reinterpret_cast<musaGraph_t>(graph),
                                    index);
  }

  template <typename GraphT,
            typename = std::enable_if_t<
                std::is_pointer_v<std::decay_t<GraphT>> &&
                !std::is_same_v<std::decay_t<GraphT>, musaGraph_t>>>
  error_t populate_cuda_graph(
      mudnnHandle_t handle,
      const std::unordered_map<int64_t, void*>& variant_pack, void* workspace,
      GraphT graph, int64_t index = 0) const {
    return populate_musa_graph_impl(handle, variant_pack, workspace,
                                    reinterpret_cast<musaGraph_t>(graph),
                                    index);
  }

  error_t update_cuda_graph(
      mudnnHandle_t handle,
      const std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>&
          variant_pack,
      void* workspace, musaGraph_t graph, int64_t index = 0) const {
    return update_musa_graph_impl(handle, variant_pack, workspace, graph,
                                  index);
  }

  error_t update_cuda_graph(
      mudnnHandle_t handle,
      const std::unordered_map<int64_t, void*>& variant_pack, void* workspace,
      musaGraph_t graph, int64_t index = 0) const {
    return update_musa_graph_impl(handle, variant_pack, workspace, graph,
                                  index);
  }

  template <typename GraphT,
            typename = std::enable_if_t<
                std::is_pointer_v<std::decay_t<GraphT>> &&
                !std::is_same_v<std::decay_t<GraphT>, musaGraph_t>>>
  error_t update_cuda_graph(
      mudnnHandle_t handle,
      const std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>&
          variant_pack,
      void* workspace, GraphT graph, int64_t index = 0) const {
    return update_musa_graph_impl(handle, variant_pack, workspace,
                                  reinterpret_cast<musaGraph_t>(graph), index);
  }

  template <typename GraphT,
            typename = std::enable_if_t<
                std::is_pointer_v<std::decay_t<GraphT>> &&
                !std::is_same_v<std::decay_t<GraphT>, musaGraph_t>>>
  error_t update_cuda_graph(
      mudnnHandle_t handle,
      const std::unordered_map<int64_t, void*>& variant_pack, void* workspace,
      GraphT graph, int64_t index = 0) const {
    return update_musa_graph_impl(handle, variant_pack, workspace,
                                  reinterpret_cast<musaGraph_t>(graph), index);
  }

 private:
  enum class NodeKind {
    kPointwise,
    kMatmul,
    kSdpa,
    kConcat,
    kReshape,
    kReduction,
    kGenstats,
    kRng,
    kBnFinalize,
    kBnBwdWeights,
    kConvFprop,
    kConvBwdData,
    kConvBwdFilter,
    kResampleFwd,
    kResampleBwd,
    kNormFwd,
    kNormBwd,
    kBlockScaleQuantize,
    kBlockScaleDequantize,
    kPagedCacheLoad,
    kSignal,
    kSdpaBwd,
    kSlice,
    kTranspose,
    kMoeGroupedMatmul,
    kMoeGroupedMatmulBwd
  };
  struct Node {
    NodeKind kind;
    std::string name;
    std::vector<std::shared_ptr<Tensor_attributes>> inputs;
    std::shared_ptr<Tensor_attributes> output;
    std::shared_ptr<Tensor_attributes> stats;
    std::vector<std::shared_ptr<Tensor_attributes>> aux_outputs;
    mudnnPointwiseMode_t pointwise_mode;
    mudnnDataType_t compute_data_type;
    double pointwise_alpha1 = 1.0;
    double pointwise_alpha2 = 1.0;
    double alpha = 1.0;
    double beta = 0.0;
    int64_t pointwise_axis = 0;
    bool pointwise_has_axis = false;
    double pointwise_relu_lower_clip = 0.0;
    double pointwise_relu_upper_clip = 0.0;
    double pointwise_relu_lower_clip_slope = 0.0;
    bool pointwise_has_relu_lower_clip = false;
    bool pointwise_has_relu_upper_clip = false;
    bool pointwise_has_relu_lower_clip_slope = false;
    double sdpa_scale = 0.0;
    bool sdpa_has_scale = false;
    bool sdpa_causal_mask = false;
    bool sdpa_is_inference = true;
    bool sdpa_use_padding_mask = false;
    DiagonalAlignment_t sdpa_diagonal_alignment = DiagonalAlignment_t::TOP_LEFT;
    int64_t sdpa_diagonal_band_left_bound = 0;
    int64_t sdpa_diagonal_band_right_bound = 0;
    bool sdpa_has_diagonal_band_left_bound = false;
    bool sdpa_has_diagonal_band_right_bound = false;
    bool sdpa_has_bias = false;
    bool sdpa_has_mask = false;
    bool sdpa_has_seq_len_q = false;
    bool sdpa_has_seq_len_kv = false;
    bool sdpa_bwd_deterministic = false;
    bool sdpa_bwd_has_d_bias = false;
    int64_t concat_axis = 0;
    mudnnReduceTensorOp_t reduction_mode = MUDNN_REDUCE_TENSOR_ADD;
    mudnnGenStatsMode_t genstats_mode = MUDNN_GENSTATS_SUM_SQSUM;
    mudnnRngDistribution_t rng_distribution = MUDNN_RNG_DISTRIBUTION_UNIFORM;
    int64_t rng_seed = 0;
    double rng_normal_mean = 0.0;
    double rng_normal_stddev = 1.0;
    double rng_uniform_minimum = 0.0;
    double rng_uniform_maximum = 1.0;
    double rng_bernoulli_probability = 0.5;
    bool rng_has_offset = false;
    mudnnBnFinalizeStatsMode_t bn_finalize_mode =
        MUDNN_BN_FINALIZE_STATISTICS_INFERENCE;
    double bn_finalize_epsilon = 1.0e-5;
    double bn_finalize_momentum = 0.1;
    mudnnConvolutionMode_t conv_mode = MUDNN_CROSS_CORRELATION;
    int64_t conv_group_count = 1;
    mudnnResampleMode_t resample_mode = MUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING;
    std::vector<int64_t> param0;
    std::vector<int64_t> param1;
    std::vector<int64_t> param2;
    std::vector<int64_t> param3;
    bool resample_has_indices = false;
    bool resample_has_x = false;
    bool resample_has_y = false;
    mudnnBackendNormMode_t norm_mode = MUDNN_BATCH_NORM;
    mudnnBackendNormFwdPhase_t norm_phase = MUDNN_NORM_FWD_INFERENCE;
    double norm_epsilon = 1.0e-5;
    double norm_momentum = 0.1;
    bool norm_has_epsilon_tensor = false;
    bool norm_has_momentum_tensor = false;
    bool norm_has_mean = true;
    bool norm_has_bias = true;
    bool norm_has_dbias = true;
    int64_t block_scale_block_size = 1;
    int64_t block_scale_block_axis = -1;
    bool block_scale_is_negative = false;
    mudnnMoeGroupedMatmulMode_t moe_grouped_matmul_mode =
        MUDNN_MOE_GROUPED_MATMUL_MODE_NONE;
    int32_t moe_grouped_matmul_top_k = 0;
    mudnnSignalMode_t signal_mode = MUDNN_SIGNAL_SET;
    int64_t signal_value = 0;
  };

  static std::vector<int64_t> row_major_strides(
      const std::vector<int64_t>& dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(dims.size()) - 1; i >= 0; --i) {
      strides[static_cast<size_t>(i)] = stride;
      stride *= dims[static_cast<size_t>(i)];
    }
    return strides;
  }

  static std::vector<int64_t> broadcast_dims(const std::vector<int64_t>& lhs,
                                             const std::vector<int64_t>& rhs) {
    const size_t rank = std::max(lhs.size(), rhs.size());
    std::vector<int64_t> dims(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
      const int64_t lhs_dim = i < lhs.size() ? lhs[lhs.size() - 1 - i] : 1;
      const int64_t rhs_dim = i < rhs.size() ? rhs[rhs.size() - 1 - i] : 1;
      if (lhs_dim <= 0 || rhs_dim <= 0) {
        throw std::invalid_argument("broadcast dimensions must be positive");
      }
      if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
        throw std::invalid_argument("pointwise inputs are not broadcastable");
      }
      dims[rank - 1 - i] = std::max(lhs_dim, rhs_dim);
    }
    return dims;
  }

  static std::vector<int64_t> broadcast_batch_dims(
      const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs,
      size_t matrix_rank, const std::string& context) {
    const size_t lhs_batch = lhs.size() - matrix_rank;
    const size_t rhs_batch = rhs.size() - matrix_rank;
    const size_t rank = std::max(lhs_batch, rhs_batch);
    std::vector<int64_t> dims(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
      const size_t lhs_from_back = i + matrix_rank;
      const size_t rhs_from_back = i + matrix_rank;
      const int64_t lhs_dim =
          lhs_from_back < lhs.size() ? lhs[lhs.size() - 1 - lhs_from_back] : 1;
      const int64_t rhs_dim =
          rhs_from_back < rhs.size() ? rhs[rhs.size() - 1 - rhs_from_back] : 1;
      if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
        throw std::invalid_argument(context +
                                    " batch dimensions are not broadcastable");
      }
      dims[rank - 1 - i] = std::max(lhs_dim, rhs_dim);
    }
    return dims;
  }

  static int64_t normalize_axis(int64_t axis, int64_t rank) {
    if (rank <= 0) {
      throw std::invalid_argument("rank must be positive");
    }
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      throw std::invalid_argument("axis is out of range");
    }
    return axis;
  }

  static std::vector<int64_t> normalize_axes(std::vector<int64_t> axes,
                                             int64_t rank) {
    if (axes.empty()) {
      throw std::invalid_argument("axes must not be empty");
    }
    for (auto& axis : axes) {
      axis = normalize_axis(axis, rank);
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    return axes;
  }

  static std::vector<int64_t> normalize_conv_param(std::vector<int64_t> values,
                                                   const char* name,
                                                   bool must_be_positive,
                                                   int64_t rank = 2) {
    if (values.size() != static_cast<size_t>(rank)) {
      throw std::invalid_argument(std::string("convolution/resample ") + name +
                                  " must be rank-" + std::to_string(rank));
    }
    for (auto value : values) {
      if (must_be_positive ? value <= 0 : value < 0) {
        throw std::invalid_argument(
            "convolution/resample padding must be nonnegative; "
            "window/stride/dilation positive");
      }
    }
    return values;
  }

  static std::vector<int64_t> conv_output_spatial(
      const std::vector<int64_t>& input_spatial,
      const std::vector<int64_t>& filter_spatial,
      const std::vector<int64_t>& pre_padding,
      const std::vector<int64_t>& post_padding,
      const std::vector<int64_t>& stride,
      const std::vector<int64_t>& dilation) {
    std::vector<int64_t> output(input_spatial.size());
    for (size_t index = 0; index < input_spatial.size(); ++index) {
      output[index] =
          (input_spatial[index] + pre_padding[index] + post_padding[index] -
           dilation[index] * (filter_spatial[index] - 1) - 1) /
              stride[index] +
          1;
    }
    return output;
  }

  static std::vector<int64_t> resample2d_output_spatial(
      const std::vector<int64_t>& input_spatial,
      const std::vector<int64_t>& window, const std::vector<int64_t>& padding,
      const std::vector<int64_t>& stride) {
    return {(input_spatial[0] + 2 * padding[0] - window[0]) / stride[0] + 1,
            (input_spatial[1] + 2 * padding[1] - window[1]) / stride[1] + 1};
  }

  static int64_t num_elements(const std::vector<int64_t>& dims) {
    int64_t count = 1;
    for (auto dim : dims) {
      if (dim < 0) {
        throw std::invalid_argument("tensor dimensions must be non-negative");
      }
      count *= dim;
    }
    return count;
  }

  static std::vector<int64_t> block_scale_scale_dims(
      const std::shared_ptr<Tensor_attributes>& x,
      const Block_scale_attributes& attributes) {
    if (attributes.block_size_ <= 0) {
      throw std::invalid_argument("block_scale block_size must be positive");
    }
    if (x->dims_.empty()) {
      throw std::invalid_argument("block_scale expects a non-scalar tensor");
    }
    const auto axis = block_scale_axis(x, attributes);
    auto dims = x->dims_;
    dims[axis] =
        (dims[axis] + attributes.block_size_ - 1) / attributes.block_size_;
    return dims;
  }

  static void check_block_scale_scale(
      const char* name, const std::shared_ptr<Tensor_attributes>& scale,
      const std::shared_ptr<Tensor_attributes>& x,
      const Block_scale_attributes& attributes) {
    if (scale->dims_ != block_scale_scale_dims(x, attributes)) {
      throw std::invalid_argument(std::string(name) +
                                  " dimensions do not match block_size");
    }
    if (!is_supported_block_scale_scale_type(scale->data_type_)) {
      throw std::invalid_argument(std::string(name) +
                                  " data type is not supported");
    }
  }

  static int64_t block_scale_axis(const std::shared_ptr<Tensor_attributes>& x,
                                  const Block_scale_attributes& attributes) {
    const auto rank = static_cast<int64_t>(x->dims_.size());
    const auto axis = rank - 1 + attributes.block_axis_tail_offset_;
    if (axis < 0 || axis >= rank) {
      throw std::invalid_argument(
          "block_scale block_size rank exceeds tensor rank");
    }
    return axis;
  }

  static bool block_scale_is_last_axis(const Node& node) {
    const auto& dims = node.inputs[0]->dims_;
    return node.block_scale_block_axis == static_cast<int64_t>(dims.size()) - 1;
  }

  static bool is_supported_block_scale_scale_type(mudnnDataType_t data_type) {
    return data_type == MUDNN_DATA_FLOAT || data_type == MUDNN_DATA_FP8_E4M3 ||
        data_type == MUDNN_DATA_FP8_E5M2 || data_type == MUDNN_DATA_FP8_E8M0;
  }

  static int64_t moe_first_token_offset_experts(
      const std::shared_ptr<Tensor_attributes>& first_token_offset,
      const std::string& op_name) {
    if (!first_token_offset) {
      throw std::invalid_argument(op_name + " first_token_offset is required");
    }
    if (first_token_offset->dims_.size() == 1) {
      return first_token_offset->dims_[0];
    }
    if (first_token_offset->dims_.size() == 3 &&
        first_token_offset->dims_[1] == 1 &&
        first_token_offset->dims_[2] == 1) {
      return first_token_offset->dims_[0];
    }
    throw std::invalid_argument(op_name +
                                " first_token_offset must be rank-1 or "
                                "[experts, 1, 1]");
  }

  static std::vector<int64_t> paged_cache_load_output_dims(
      const std::shared_ptr<Tensor_attributes>& container,
      const std::shared_ptr<Tensor_attributes>& sequence,
      const std::shared_ptr<Tensor_attributes>& page_table) {
    if (container->dims_.empty()) {
      throw std::invalid_argument(
          "paged_cache_load container must be non-scalar");
    }
    if (sequence->dims_.size() != 1) {
      throw std::invalid_argument("paged_cache_load sequence must be rank-1");
    }
    if (page_table->dims_.size() != 1) {
      throw std::invalid_argument("paged_cache_load page_table must be rank-1");
    }
    if (sequence->data_type_ != MUDNN_DATA_INT64 ||
        page_table->data_type_ != MUDNN_DATA_INT64) {
      throw std::invalid_argument(
          "paged_cache_load sequence and page_table must be int64");
    }
    std::vector<int64_t> dims;
    dims.reserve(container->dims_.size());
    dims.push_back(sequence->dims_[0]);
    dims.insert(dims.end(), container->dims_.begin() + 1,
                container->dims_.end());
    return dims;
  }

  static void check_signal_flag(
      const std::shared_ptr<Tensor_attributes>& flag) {
    if (flag->data_type_ != MUDNN_DATA_INT64) {
      throw std::invalid_argument("signal flag data type must be int64");
    }
    if (num_elements(flag->dims_) != 1) {
      throw std::invalid_argument(
          "signal flag must contain exactly one element");
    }
  }

  static void check_sdpa_backward_tensors(
      const std::shared_ptr<Tensor_attributes>& q,
      const std::shared_ptr<Tensor_attributes>& k,
      const std::shared_ptr<Tensor_attributes>& v,
      const std::shared_ptr<Tensor_attributes>& o,
      const std::shared_ptr<Tensor_attributes>& d_o,
      const std::shared_ptr<Tensor_attributes>& stats) {
    for (const auto& tensor : {q, k, v, o, d_o}) {
      if (tensor->dims_.size() != 4) {
        throw std::invalid_argument("sdpa_backward expects rank-4 tensors");
      }
    }
    if (stats->dims_.size() != 4) {
      throw std::invalid_argument("sdpa_backward expects stats to be rank-4");
    }
    if (q->dims_[0] != k->dims_[0] || q->dims_[0] != v->dims_[0]) {
      throw std::invalid_argument("sdpa_backward batch dimensions must match");
    }
    if (q->dims_[1] != k->dims_[1] || q->dims_[1] != v->dims_[1]) {
      throw std::invalid_argument("sdpa_backward head dimensions must match");
    }
    if (k->dims_[2] != v->dims_[2]) {
      throw std::invalid_argument(
          "sdpa_backward key/value sequence dimensions must match");
    }
    if (q->dims_[3] != k->dims_[3]) {
      throw std::invalid_argument(
          "sdpa_backward q/k head dimensions must match");
    }
    const std::vector<int64_t> o_dims = {q->dims_[0], q->dims_[1], q->dims_[2],
                                         v->dims_[3]};
    if (o->dims_ != o_dims) {
      throw std::invalid_argument(
          "sdpa_backward output tensor dimensions are invalid");
    }
    if (d_o->dims_ != o->dims_) {
      throw std::invalid_argument("sdpa_backward d_o dimensions must match o");
    }
    if (stats->dims_ !=
        std::vector<int64_t>({q->dims_[0], q->dims_[1], q->dims_[2], 1})) {
      throw std::invalid_argument("sdpa_backward stats dimensions are invalid");
    }
    if (stats->data_type_ != MUDNN_DATA_FLOAT) {
      throw std::invalid_argument(
          "sdpa_backward stats data type must be FLOAT");
    }
    for (const auto& tensor : {k, v, o, d_o}) {
      if (tensor->data_type_ != q->data_type_) {
        throw std::invalid_argument(
            "sdpa_backward tensor data type must match q");
      }
    }
  }

  static void check_sdpa_attention_tensor(
      const char* name, const std::shared_ptr<Tensor_attributes>& tensor,
      const std::shared_ptr<Tensor_attributes>& q,
      const std::shared_ptr<Tensor_attributes>& k, mudnnDataType_t q_type,
      bool require_q_type) {
    if (require_q_type) {
      if (tensor->data_type_ != q_type) {
        throw std::invalid_argument(std::string(name) +
                                    " data type must match q");
      }
    } else if (tensor->data_type_ != q_type &&
               tensor->data_type_ != MUDNN_DATA_BOOLEAN) {
      throw std::invalid_argument(std::string(name) +
                                  " data type must be boolean or match q");
    }
    const std::vector<std::vector<int64_t>> valid_shapes = {
        {q->dims_[2], k->dims_[2]},
        {q->dims_[0] * q->dims_[1], q->dims_[2], k->dims_[2]},
        {q->dims_[0], q->dims_[2], k->dims_[2]},
        {1, q->dims_[2], k->dims_[2]},
        {q->dims_[0], q->dims_[1], q->dims_[2], k->dims_[2]},
        {q->dims_[0], 1, q->dims_[2], k->dims_[2]},
        {1, q->dims_[1], q->dims_[2], k->dims_[2]},
        {1, 1, q->dims_[2], k->dims_[2]},
        {q->dims_[0], k->dims_[2]}};
    if (std::find(valid_shapes.begin(), valid_shapes.end(), tensor->dims_) ==
        valid_shapes.end()) {
      throw std::invalid_argument(std::string(name) +
                                  " dimensions are not supported");
    }
  }

  static void check_sdpa_seq_len(
      const char* name, const std::shared_ptr<Tensor_attributes>& tensor,
      int64_t batch) {
    if (tensor->data_type_ != MUDNN_DATA_INT32 &&
        tensor->data_type_ != MUDNN_DATA_INT64) {
      throw std::invalid_argument(std::string(name) +
                                  " data type must be INT32 or INT64");
    }
    const std::vector<std::vector<int64_t>> valid_shapes = {
        {batch}, {batch, 1}, {batch, 1, 1, 1}};
    if (std::find(valid_shapes.begin(), valid_shapes.end(), tensor->dims_) ==
        valid_shapes.end()) {
      throw std::invalid_argument(std::string(name) +
                                  " dimensions must be batch-shaped");
    }
  }

  static void check_same_shape_type(
      const char* name, const std::shared_ptr<Tensor_attributes>& tensor,
      const std::shared_ptr<Tensor_attributes>& reference) {
    if (tensor->dims_ != reference->dims_) {
      throw std::invalid_argument(std::string(name) +
                                  " dimensions must match reference");
    }
    if (tensor->data_type_ != reference->data_type_) {
      throw std::invalid_argument(std::string(name) +
                                  " data type must match reference");
    }
  }

  static std::vector<int64_t> batch_norm_param_dims(
      const std::shared_ptr<Tensor_attributes>& x) {
    if (x->dims_.size() != 4) {
      throw std::invalid_argument("batch_norm expects rank-4 NCHW tensors");
    }
    return {1, x->dims_[1], 1, 1};
  }

  static void check_batch_norm_param(
      const char* name, const std::shared_ptr<Tensor_attributes>& tensor,
      const std::shared_ptr<Tensor_attributes>& x) {
    if (tensor->dims_ != batch_norm_param_dims(x)) {
      throw std::invalid_argument(std::string("batch_norm ") + name +
                                  " dimensions must be (1, C, 1, 1)");
    }
    if (tensor->data_type_ != x->data_type_) {
      throw std::invalid_argument(std::string("batch_norm ") + name +
                                  " data type must match x");
    }
  }

  static std::vector<int64_t> norm_stat_dims(
      mudnnBackendNormMode_t mode,
      const std::shared_ptr<Tensor_attributes>& x) {
    if (x->dims_.empty()) {
      throw std::invalid_argument("norm expects a non-scalar input tensor");
    }
    if (mode == MUDNN_INSTANCE_NORM) {
      if (x->dims_.size() != 4) {
        throw std::invalid_argument("instancenorm expects rank-4 NCHW tensors");
      }
      return {x->dims_[0], x->dims_[1], 1, 1};
    }
    std::vector<int64_t> dims(x->dims_.size(), 1);
    dims[0] = x->dims_[0];
    return dims;
  }

  static std::vector<int64_t> norm_param_dims(
      mudnnBackendNormMode_t mode,
      const std::shared_ptr<Tensor_attributes>& x) {
    if (x->dims_.empty()) {
      throw std::invalid_argument("norm expects a non-scalar input tensor");
    }
    if (mode == MUDNN_INSTANCE_NORM) {
      if (x->dims_.size() != 4) {
        throw std::invalid_argument("instancenorm expects rank-4 NCHW tensors");
      }
      return {1, x->dims_[1], 1, 1};
    }
    std::vector<int64_t> dims = x->dims_;
    dims[0] = 1;
    return dims;
  }

  static void check_norm_param(const char* op_name, const char* name,
                               const std::shared_ptr<Tensor_attributes>& tensor,
                               const std::shared_ptr<Tensor_attributes>& x,
                               mudnnBackendNormMode_t mode) {
    if (tensor->dims_ != norm_param_dims(mode, x)) {
      throw std::invalid_argument(std::string(op_name) + " " + name +
                                  " dimensions are invalid");
    }
    if (tensor->data_type_ != x->data_type_) {
      throw std::invalid_argument(std::string(op_name) + " " + name +
                                  " data type must match input");
    }
  }

  static void check_scalar_float_tensor(
      const char* op_name, const char* name,
      const std::shared_ptr<Tensor_attributes>& tensor) {
    int64_t elements = 1;
    if (tensor) {
      for (auto dim : tensor->dims_) {
        elements *= dim;
      }
    }
    if (!tensor || elements != 1) {
      throw std::invalid_argument(std::string(op_name) + " " + name +
                                  " tensor must contain one element");
    }
    if (tensor->data_type_ != MUDNN_DATA_FLOAT &&
        tensor->data_type_ != MUDNN_DATA_DOUBLE) {
      throw std::invalid_argument(std::string(op_name) + " " + name +
                                  " tensor data type must be FLOAT");
    }
  }

  static void write_int_vector(std::ostream& out,
                               const std::vector<int64_t>& values) {
    out << values.size();
    for (auto value : values) {
      out << ' ' << value;
    }
  }

  static std::vector<int64_t> read_int_vector(std::istream& in) {
    size_t size = 0;
    in >> size;
    std::vector<int64_t> values(size);
    for (auto& value : values) {
      in >> value;
    }
    return values;
  }

  void clear_compiled_state() {
    tensor_descs_.clear();
    internal_tensor_descs_.clear();
    pointwise_descs_.clear();
    matmul_descs_.clear();
    conv_descs_.clear();
    reduction_descs_.clear();
    resample_descs_.clear();
    rng_descs_.clear();
    operations_.clear();
    operation_graph_ = OperationGraph();
    heuristics_.clear();
    engine_configs_.clear();
    plans_.clear();
    variant_pack_ = VariantPack();
  }

  void register_tensor(const std::shared_ptr<Tensor_attributes>& tensor) {
    if (!tensor->has_uid_) {
      tensor->set_uid(next_uid_++);
    } else {
      next_uid_ = std::max(next_uid_, tensor->uid_ + 1);
    }
    tensors_.push_back(tensor);
  }

  std::shared_ptr<Tensor_attributes> make_virtual_like(
      const std::shared_ptr<Tensor_attributes>& reference,
      const std::string& name, mudnnDataType_t data_type) {
    auto tensor = std::make_shared<Tensor_attributes>();
    tensor->set_name(name);
    tensor->set_dim(reference->dims_);
    tensor->set_stride(row_major_strides(reference->dims_));
    tensor->set_data_type(data_type);
    tensor->set_is_virtual(true);
    register_tensor(tensor);
    return tensor;
  }

  std::shared_ptr<Tensor_attributes> make_virtual_tensor(
      const std::string& name, const std::vector<int64_t>& dims,
      mudnnDataType_t data_type) {
    auto tensor = std::make_shared<Tensor_attributes>();
    tensor->set_name(name);
    tensor->set_dim(dims);
    tensor->set_stride(row_major_strides(dims));
    tensor->set_data_type(data_type);
    tensor->set_is_virtual(true);
    register_tensor(tensor);
    return tensor;
  }

  std::shared_ptr<Tensor_attributes> make_scalar_by_value(
      const std::string& name, double value) {
    auto storage = std::make_shared<float>(static_cast<float>(value));
    auto tensor = std::make_shared<Tensor_attributes>();
    tensor->set_name(name);
    tensor->set_dim({1});
    tensor->set_stride({1});
    tensor->set_data_type(MUDNN_DATA_FLOAT);
    tensor->set_is_pass_by_value(true);
    register_tensor(tensor);
    by_value_scalar_ptrs_[tensor.get()] = storage.get();
    by_value_scalars_.emplace_back(std::move(storage));
    return tensor;
  }

  template <typename T>
  std::shared_ptr<Tensor_attributes> make_scalar_by_value(
      const std::string& name, T value, ScalarType /*scalar_type*/,
      mudnnDataType_t data_type) {
    auto storage = std::make_shared<T>(value);
    auto tensor = std::make_shared<Tensor_attributes>();
    tensor->set_name(name);
    tensor->set_dim({1});
    tensor->set_stride({1});
    tensor->set_data_type(data_type);
    tensor->set_is_pass_by_value(true);
    register_tensor(tensor);
    by_value_scalar_ptrs_[tensor.get()] = storage.get();
    by_value_scalars_.emplace_back(std::move(storage));
    return tensor;
  }

  std::shared_ptr<Tensor_attributes> pointwise_impl(
      const std::shared_ptr<Tensor_attributes>& x,
      const std::shared_ptr<Tensor_attributes>& b,
      const std::shared_ptr<Tensor_attributes>& t,
      const Pointwise_attributes& attributes) {
    if (!x) {
      throw std::invalid_argument("pointwise requires a non-null input");
    }
    auto y = std::make_shared<Tensor_attributes>();
    y->set_name(attributes.name_.empty()
                    ? ("pointwise_" + std::to_string(nodes_.size()))
                    : attributes.name_);
    auto dims = x->dims_;
    if (b) {
      dims = broadcast_dims(dims, b->dims_);
    }
    if (t) {
      dims = broadcast_dims(dims, t->dims_);
    }
    y->set_dim(dims);
    // Upstream cudnn-frontend shape deduction: when an input's dims match the
    // broadcast output dims, the output inherits that input's strides.
    if (dims == x->dims_ && !x->strides_.empty()) {
      y->set_stride(x->strides_);
    } else if (b && dims == b->dims_ && !b->strides_.empty()) {
      y->set_stride(b->strides_);
    } else if (t && dims == t->dims_ && !t->strides_.empty()) {
      y->set_stride(t->strides_);
    } else {
      y->set_stride(row_major_strides(dims));
    }
    y->set_data_type(attributes.has_output_data_type_
                         ? attributes.output_data_type_
                         : io_data_type_);
    y->set_is_virtual(true);
    register_tensor(y);
    std::vector<std::shared_ptr<Tensor_attributes>> inputs = {x};
    if (b) {
      inputs.push_back(b);
    }
    if (t) {
      inputs.push_back(t);
    }
    Node node;
    node.kind = NodeKind::kPointwise;
    node.name = attributes.name_;
    node.inputs = std::move(inputs);
    node.output = y;
    node.pointwise_mode = attributes.mode_;
    node.compute_data_type = attributes.compute_data_type_;
    node.pointwise_alpha1 = attributes.alpha1_;
    node.pointwise_alpha2 = attributes.alpha2_;
    node.pointwise_axis = attributes.axis_;
    node.pointwise_has_axis = attributes.has_axis_;
    node.pointwise_relu_lower_clip = attributes.relu_lower_clip_;
    node.pointwise_relu_upper_clip = attributes.relu_upper_clip_;
    node.pointwise_relu_lower_clip_slope = attributes.relu_lower_clip_slope_;
    node.pointwise_has_relu_lower_clip = attributes.has_relu_lower_clip_;
    node.pointwise_has_relu_upper_clip = attributes.has_relu_upper_clip_;
    node.pointwise_has_relu_lower_clip_slope =
        attributes.has_relu_lower_clip_slope_;
    nodes_.push_back(std::move(node));
    return y;
  }

  std::shared_ptr<Tensor_attributes> compare_impl(
      mudnnPointwiseMode_t mode,
      const std::shared_ptr<Tensor_attributes>& input,
      const std::shared_ptr<Tensor_attributes>& comparison,
      const Pointwise_attributes& attributes) {
    if (!input || !comparison) {
      throw std::invalid_argument("comparison requires input and comparison");
    }
    auto attrs = attributes;
    attrs.set_mode(mode).set_output_data_type(MUDNN_DATA_BOOLEAN);
    return pointwise_impl(input, comparison, nullptr, attrs);
  }

  std::shared_ptr<Tensor_attributes> logical_impl(
      mudnnPointwiseMode_t mode, const std::shared_ptr<Tensor_attributes>& a,
      const std::shared_ptr<Tensor_attributes>& b,
      const Pointwise_attributes& attributes) {
    if (!a || !b) {
      throw std::invalid_argument("logical operation requires two inputs");
    }
    if (a->data_type_ != MUDNN_DATA_BOOLEAN ||
        b->data_type_ != MUDNN_DATA_BOOLEAN) {
      throw std::invalid_argument("logical operation inputs must be boolean");
    }
    auto attrs = attributes;
    attrs.set_mode(mode).set_output_data_type(MUDNN_DATA_BOOLEAN);
    return pointwise_impl(a, b, nullptr, attrs);
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  norm_forward_impl(const char* op_name, mudnnBackendNormMode_t mode,
                    const std::shared_ptr<Tensor_attributes>& input,
                    const std::shared_ptr<Tensor_attributes>& scale,
                    const std::shared_ptr<Tensor_attributes>& bias,
                    const Norm_attributes& attributes, bool has_mean) {
    if (!input || !scale || (bias == nullptr && mode != MUDNN_RMS_NORM)) {
      throw std::invalid_argument(std::string(op_name) +
                                  " requires input, scale, and bias");
    }
    if (attributes.epsilon_tensor_) {
      check_scalar_float_tensor(op_name, "epsilon", attributes.epsilon_tensor_);
    } else if (attributes.epsilon_ < 0.0) {
      throw std::invalid_argument(std::string(op_name) +
                                  " epsilon must be nonnegative");
    }
    check_norm_param(op_name, "scale", scale, input, mode);
    if (bias) {
      check_norm_param(op_name, "bias", bias, input, mode);
    }
    const auto base_name = attributes.name_.empty()
        ? (std::string(op_name) + "_" + std::to_string(nodes_.size()))
        : attributes.name_;
    const auto stat_dims = norm_stat_dims(mode, input);
    auto y = make_virtual_tensor(base_name, input->dims_,
                                 attributes.has_output_data_type_
                                     ? attributes.output_data_type_
                                     : input->data_type_);
    std::shared_ptr<Tensor_attributes> mean;
    if (has_mean) {
      mean = make_virtual_tensor(base_name + "_mean", stat_dims,
                                 attributes.compute_data_type_);
    }
    auto inv_variance = make_virtual_tensor(
        base_name + "_inv_variance", stat_dims, attributes.compute_data_type_);

    Node node;
    node.kind = NodeKind::kNormFwd;
    node.name = base_name;
    node.inputs = {input, scale};
    if (bias) {
      node.inputs.push_back(bias);
    }
    if (attributes.epsilon_tensor_) {
      node.inputs.push_back(attributes.epsilon_tensor_);
    }
    node.output = y;
    node.stats = mean;
    node.aux_outputs = {inv_variance};
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.norm_mode = mode;
    node.norm_phase = attributes.phase_;
    node.norm_epsilon = attributes.epsilon_;
    node.norm_has_epsilon_tensor =
        static_cast<bool>(attributes.epsilon_tensor_);
    node.norm_has_mean = has_mean;
    node.norm_has_bias = static_cast<bool>(bias);
    node.norm_has_dbias = true;
    nodes_.push_back(std::move(node));
    return {y, mean, inv_variance};
  }

  std::array<std::shared_ptr<Tensor_attributes>, 3>
  norm_backward_impl(const char* op_name, mudnnBackendNormMode_t mode,
                     const std::shared_ptr<Tensor_attributes>& grad,
                     const std::shared_ptr<Tensor_attributes>& input,
                     const std::shared_ptr<Tensor_attributes>& scale,
                     const std::shared_ptr<Tensor_attributes>& mean,
                     const std::shared_ptr<Tensor_attributes>& inv_variance,
                     const Norm_attributes& attributes, bool has_dbias) {
    if (!grad || !input || !scale || !inv_variance) {
      throw std::invalid_argument(std::string(op_name) +
                                  " requires grad, input, scale, and stats");
    }
    if (mean == nullptr && mode != MUDNN_RMS_NORM) {
      throw std::invalid_argument(std::string(op_name) + " requires mean");
    }
    if (attributes.epsilon_tensor_) {
      check_scalar_float_tensor(op_name, "epsilon", attributes.epsilon_tensor_);
    } else if (attributes.epsilon_ < 0.0) {
      throw std::invalid_argument(std::string(op_name) +
                                  " epsilon must be nonnegative");
    }
    if (grad->dims_ != input->dims_ || grad->data_type_ != input->data_type_) {
      throw std::invalid_argument(std::string(op_name) +
                                  " grad shape and data type must match input");
    }
    check_norm_param(op_name, "scale", scale, input, mode);
    const auto stat_dims = norm_stat_dims(mode, input);
    if (mean && mean->dims_ != stat_dims) {
      throw std::invalid_argument(std::string(op_name) +
                                  " mean dimensions are invalid");
    }
    if (inv_variance->dims_ != stat_dims) {
      throw std::invalid_argument(std::string(op_name) +
                                  " inv_variance dimensions are invalid");
    }
    const auto base_name = attributes.name_.empty()
        ? (std::string(op_name) + "_" + std::to_string(nodes_.size()))
        : attributes.name_;
    auto dx = make_virtual_tensor(base_name, input->dims_,
                                  attributes.has_output_data_type_
                                      ? attributes.output_data_type_
                                      : input->data_type_);
    auto dscale = make_virtual_tensor(base_name + "_dscale", scale->dims_,
                                      scale->data_type_);
    std::shared_ptr<Tensor_attributes> dbias;
    if (has_dbias) {
      dbias = make_virtual_tensor(base_name + "_dbias", scale->dims_,
                                  scale->data_type_);
    }

    Node node;
    node.kind = NodeKind::kNormBwd;
    node.name = base_name;
    node.inputs = {input, grad};
    if (mean) {
      node.inputs.push_back(mean);
    }
    node.inputs.push_back(inv_variance);
    node.inputs.push_back(scale);
    if (attributes.epsilon_tensor_) {
      node.inputs.push_back(attributes.epsilon_tensor_);
    }
    node.output = dx;
    node.stats = dscale;
    if (dbias) {
      node.aux_outputs = {dbias};
    }
    node.pointwise_mode = MUDNN_POINTWISE_IDENTITY;
    node.compute_data_type = attributes.compute_data_type_;
    node.norm_mode = mode;
    node.norm_epsilon = attributes.epsilon_;
    node.norm_has_epsilon_tensor =
        static_cast<bool>(attributes.epsilon_tensor_);
    node.norm_has_mean = static_cast<bool>(mean);
    node.norm_has_bias = true;
    node.norm_has_dbias = has_dbias;
    nodes_.push_back(std::move(node));
    return {dx, dscale, dbias};
  }

  void build_backend_tensors() {
    tensor_descs_.clear();
    for (const auto& tensor : tensors_) {
      const bool is_virtual = tensor->is_virtual_ && !tensor->is_output_;
      auto desc = TensorBuilder()
                      .setDim(tensor->dims_)
                      .setStrides(tensor->strides_.empty()
                                      ? row_major_strides(tensor->dims_)
                                      : tensor->strides_)
                      .setId(tensor->uid_)
                      .setDataType(tensor->data_type_)
                      .setAlignment(tensor->alignment_)
                      .setVectorCountAndDimension(tensor->vector_count_,
                                                  tensor->vector_dimension_)
                      .setReorderType(tensor->reordering_type_)
                      .setVirtual(is_virtual)
                      .setByValue(tensor->is_by_value_)
                      .build();
      tensor_descs_.emplace(tensor.get(), desc);
    }
  }

  void ensure_engine_configs() {
    if (!engine_configs_.empty()) {
      return;
    }
    if (!operation_graph_) {
      throw std::runtime_error("operation graph has not been built");
    }
    if (plan_modes_.empty()) {
      plan_modes_.push_back(MUDNN_HEUR_MODE_INSTANT);
    }
    heuristics_.clear();
    engine_configs_.clear();
    heuristics_.reserve(plan_modes_.size());
    for (auto mode : plan_modes_) {
      heuristics_.push_back(EngineHeuristicsBuilder()
                                .setOperationGraph(operation_graph_)
                                .setHeurMode(mode)
                                .build());
      const auto count = heuristics_.back().getEngineConfigCount();
      auto configs = heuristics_.back().getEngineConfig(count);
      for (auto& config : configs) {
        engine_configs_.push_back(std::move(config));
      }
    }
    refresh_engine_selection();
  }

  void validate_engine_index(int64_t index) {
    ensure_engine_configs();
    if (index < 0 || index >= static_cast<int64_t>(engine_configs_.size())) {
      throw std::out_of_range("execution plan index " + std::to_string(index) +
                              " out of range");
    }
  }

  void build_plan_at_index_impl(int64_t index) {
    validate_engine_index(index);
    const auto slot = static_cast<size_t>(index);
    if (!engine_config_selected_.at(slot)) {
      throw std::runtime_error("execution plan at index " +
                               std::to_string(index) +
                               " was deselected by graph filters");
    }
    while (plans_.size() <= slot) {
      plans_.push_back(ExecutionPlan());
    }
    if (plans_[slot]) {
      return;
    }
    plans_[slot] = ExecutionPlanBuilder()
                       .setHandle(handle_)
                       .setEngineConfig(engine_configs_[slot])
                       .build();
  }

  template <typename Note>
  static bool contains_note(const std::vector<Note>& notes, Note expected) {
    return std::find(notes.begin(), notes.end(), expected) != notes.end();
  }

  bool config_is_selected(size_t index) const {
    const auto& config = engine_configs_.at(index);
    auto engine = Engine::Borrowed(config.getEngineDescriptor());
    const auto engine_id = engine.getGlobalIndex();
    const auto generated_name =
        "mudnn_engine_config_" + std::to_string(index);
    for (const auto& blocked : deselected_engine_names_) {
      if (generated_name.find(blocked) != std::string::npos ||
          std::to_string(engine_id) == blocked) {
        return false;
      }
    }
    if (config.getWorkspaceSize() > max_workspace_ ||
        config.getSharedMemoryUsed() > max_shared_memory_) {
      return false;
    }
    const auto numeric = engine.getNumericalNotes();
    const auto behavior = engine.getBehaviorNotes();
    for (auto note : selected_numeric_notes_) {
      if (!contains_note(numeric, note)) return false;
    }
    for (auto note : deselected_numeric_notes_) {
      if (contains_note(numeric, note)) return false;
    }
    for (auto note : selected_behavior_notes_) {
      if (!contains_note(behavior, note)) return false;
    }
    for (auto note : deselected_behavior_notes_) {
      if (contains_note(behavior, note)) return false;
    }
    return true;
  }

  void refresh_engine_selection() {
    if (engine_configs_.empty()) return;
    engine_config_selected_.assign(engine_configs_.size(), false);
    for (size_t i = 0; i < engine_configs_.size(); ++i) {
      engine_config_selected_[i] = config_is_selected(i);
    }
  }

  VariantPack build_variant_pack(
      const std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>&
          variant_pack,
      void* workspace) const {
    std::vector<int64_t> uids;
    std::vector<void*> pointers;
    uids.reserve(variant_pack.size() + by_value_scalar_ptrs_.size());
    pointers.reserve(variant_pack.size() + by_value_scalar_ptrs_.size());
    for (const auto& entry : by_value_scalar_ptrs_) {
      if (!entry.first || !entry.first->has_uid()) {
        throw std::invalid_argument("by-value tensor has no uid");
      }
      uids.push_back(entry.first->get_uid());
      pointers.push_back(entry.second);
    }
    for (const auto& entry : variant_pack) {
      if (!entry.first || !entry.first->has_uid()) {
        throw std::invalid_argument("variant pack tensor has no uid");
      }
      uids.push_back(entry.first->get_uid());
      pointers.push_back(entry.second);
    }
    auto builder = VariantPackBuilder().setUids(uids).setDataPointers(pointers);
    if (workspace != nullptr) {
      builder.setWorkspacePointer(workspace);
    }
    return builder.build();
  }

  VariantPack build_variant_pack(
      const std::unordered_map<int64_t, void*>& variant_pack,
      void* workspace) const {
    std::vector<int64_t> uids;
    std::vector<void*> pointers;
    uids.reserve(variant_pack.size() + by_value_scalar_ptrs_.size());
    pointers.reserve(variant_pack.size() + by_value_scalar_ptrs_.size());
    for (const auto& entry : by_value_scalar_ptrs_) {
      if (!entry.first || !entry.first->has_uid()) {
        throw std::invalid_argument("by-value tensor has no uid");
      }
      uids.push_back(entry.first->get_uid());
      pointers.push_back(entry.second);
    }
    for (const auto& entry : variant_pack) {
      uids.push_back(entry.first);
      pointers.push_back(entry.second);
    }
    auto builder = VariantPackBuilder().setUids(uids).setDataPointers(pointers);
    if (workspace != nullptr) {
      builder.setWorkspacePointer(workspace);
    }
    return builder.build();
  }

  void execute_plan_at_index_impl(
      mudnnHandle_t handle,
      const std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>&
          variant_pack,
      void* workspace, int64_t index) const {
    if (index < 0 || index >= static_cast<int64_t>(plans_.size()) ||
        !plans_[static_cast<size_t>(index)]) {
      throw std::out_of_range("execution plan index " + std::to_string(index) +
                              " out of range");
    }
    variant_pack_ = build_variant_pack(variant_pack, workspace);
    throw_if_error(
        mudnn_frontend::execute(handle, plans_[static_cast<size_t>(index)],
                                variant_pack_),
        "mudnn_frontend::graph::Graph execute failed");
  }

  void execute_plan_at_index_impl(
      mudnnHandle_t handle,
      const std::unordered_map<int64_t, void*>& variant_pack, void* workspace,
      int64_t index) const {
    if (index < 0 || index >= static_cast<int64_t>(plans_.size()) ||
        !plans_[static_cast<size_t>(index)]) {
      throw std::out_of_range("execution plan index " + std::to_string(index) +
                              " out of range");
    }
    variant_pack_ = build_variant_pack(variant_pack, workspace);
    throw_if_error(
        mudnn_frontend::execute(handle, plans_[static_cast<size_t>(index)],
                                variant_pack_),
        "mudnn_frontend::graph::Graph execute failed");
  }

  template <typename VariantPackMap>
  error_t populate_musa_graph_impl(mudnnHandle_t handle,
                                   const VariantPackMap& variant_pack,
                                   void* workspace, musaGraph_t graph,
                                   int64_t index) const {
    try {
      if (index < 0 || index >= static_cast<int64_t>(plans_.size()) ||
          !plans_[static_cast<size_t>(index)]) {
        throw std::out_of_range("execution plan index " +
                                std::to_string(index) + " out of range");
      }
      variant_pack_ = build_variant_pack(variant_pack, workspace);
      throw_if_error(
          mudnnBackendPopulateMusaGraph(
              handle,
              plans_[static_cast<size_t>(index)].get_backend_descriptor(),
              variant_pack_.get_backend_descriptor(), graph),
          "mudnn_frontend::graph::Graph populate_cuda_graph failed");
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  template <typename VariantPackMap>
  error_t update_musa_graph_impl(mudnnHandle_t handle,
                                 const VariantPackMap& variant_pack,
                                 void* workspace, musaGraph_t graph,
                                 int64_t index) const {
    try {
      if (index < 0 || index >= static_cast<int64_t>(plans_.size()) ||
          !plans_[static_cast<size_t>(index)]) {
        throw std::out_of_range("execution plan index " +
                                std::to_string(index) + " out of range");
      }
      variant_pack_ = build_variant_pack(variant_pack, workspace);
      throw_if_error(
          mudnnBackendUpdateMusaGraph(
              handle,
              plans_[static_cast<size_t>(index)].get_backend_descriptor(),
              variant_pack_.get_backend_descriptor(), graph),
          "mudnn_frontend::graph::Graph update_cuda_graph failed");
      return {true, ""};
    } catch (const std::exception& e) {
      return {false, e.what()};
    }
  }

  mudnnDataType_t io_data_type_ =
      static_cast<mudnnDataType_t>(DataType_t::NOT_SET);
  mudnnDataType_t intermediate_data_type_ =
      static_cast<mudnnDataType_t>(DataType_t::NOT_SET);
  mudnnDataType_t compute_data_type_ =
      static_cast<mudnnDataType_t>(DataType_t::NOT_SET);
  int64_t sm_count_ = -1;
  int64_t sm_version_ = -1;
  bool dynamic_shape_enabled_ = false;
  bool override_shape_enabled_ = false;
  std::string name_;
  std::shared_ptr<const DeviceProperties> device_properties_;
  std::shared_ptr<KernelCache> kernel_cache_;
  std::vector<BehaviorNote_t> selected_behavior_notes_;
  std::vector<BehaviorNote_t> deselected_behavior_notes_;
  std::vector<NumericalNote_t> selected_numeric_notes_;
  std::vector<NumericalNote_t> deselected_numeric_notes_;
  std::vector<std::string> deselected_engine_names_;
  int64_t max_workspace_ = std::numeric_limits<int64_t>::max();
  int64_t max_shared_memory_ = std::numeric_limits<int64_t>::max();
  int64_t next_uid_ = 1;
  mudnnHandle_t handle_ = nullptr;
  std::vector<std::shared_ptr<Tensor_attributes>> tensors_;
  std::vector<Node> nodes_;
  std::map<const Tensor_attributes*, Tensor> tensor_descs_;
  std::vector<Tensor> internal_tensor_descs_;
  std::vector<PointWiseDesc> pointwise_descs_;
  std::vector<MatMulDesc> matmul_descs_;
  std::vector<ConvDesc> conv_descs_;
  std::vector<ReductionDesc> reduction_descs_;
  std::vector<ResampleDesc> resample_descs_;
  std::vector<RngDesc> rng_descs_;
  std::vector<Operation> operations_;
  OperationGraph operation_graph_;
  std::vector<mudnnBackendHeurMode_t> plan_modes_;
  std::vector<EngineHeuristics> heuristics_;
  std::vector<EngineConfig> engine_configs_;
  std::vector<bool> engine_config_selected_;
  std::vector<ExecutionPlan> plans_;
  int64_t candidate_plan_ = -1;
  std::vector<std::shared_ptr<void>> by_value_scalars_;
  std::unordered_map<const Tensor_attributes*, void*> by_value_scalar_ptrs_;
  mutable VariantPack variant_pack_;
  std::string unsupported_reason_;
};

namespace attn {
namespace score_modifiers {
inline std::shared_ptr<Tensor_attributes> bias(
    const std::shared_ptr<Graph>& graph,
    const std::shared_ptr<Tensor_attributes>& score,
    const std::shared_ptr<Tensor_attributes>& bias) {
  if (!graph) {
    throw std::invalid_argument("score_modifiers::bias requires a graph");
  }
  return graph->add(score, bias,
                    Pointwise_attributes().set_name("score_modifier_bias"));
}
}  // namespace score_modifiers
}  // namespace attn

#ifdef MUDNN_FRONTEND_HAS_NLOHMANN_JSON
inline const char* upstream_data_type_name(mudnnDataType_t type) {
  switch (type) {
    case MUDNN_DATA_FLOAT: return "FLOAT";
    case MUDNN_DATA_DOUBLE: return "DOUBLE";
    case MUDNN_DATA_HALF: return "HALF";
    case MUDNN_DATA_INT8: return "INT8";
    case MUDNN_DATA_INT32: return "INT32";
    case MUDNN_DATA_INT8x4: return "INT8x4";
    case MUDNN_DATA_UINT8: return "UINT8";
    case MUDNN_DATA_UINT8x4: return "UINT8x4";
    case MUDNN_DATA_INT8x32: return "INT8x32";
    case MUDNN_DATA_BFLOAT16: return "BFLOAT16";
    case MUDNN_DATA_INT64: return "INT64";
    case MUDNN_DATA_BOOLEAN: return "BOOLEAN";
    case MUDNN_DATA_FP8_E4M3: return "FP8_E4M3";
    case MUDNN_DATA_FP8_E5M2: return "FP8_E5M2";
    case MUDNN_DATA_FAST_FLOAT_FOR_FP8: return "FAST_FLOAT_FOR_FP8";
    case MUDNN_DATA_FP8_E8M0: return "FP8_E8M0";
    case MUDNN_DATA_FP4_E2M1: return "FP4_E2M1";
    case MUDNN_DATA_INT4: return "INT4";
    case MUDNN_DATA_COMPLEX_FP32: return "COMPLEX_FP32";
    case MUDNN_DATA_COMPLEX_FP64: return "COMPLEX_FP64";
    default: return "NOT_SET";
  }
}

inline mudnnDataType_t upstream_data_type_from_json(
    const nlohmann::json& value) {
  if (value.is_null()) return static_cast<mudnnDataType_t>(DataType_t::NOT_SET);
  if (value.is_number_integer()) {
    return static_cast<mudnnDataType_t>(value.get<int64_t>());
  }
  const auto name = value.get<std::string>();
#define MUDNN_FRONTEND_PARSE_DATA_TYPE(enum_name) \
  if (name == #enum_name) return MUDNN_DATA_##enum_name
  MUDNN_FRONTEND_PARSE_DATA_TYPE(FLOAT);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(DOUBLE);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(HALF);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(INT8);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(INT32);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(INT8x4);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(UINT8);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(UINT8x4);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(INT8x32);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(BFLOAT16);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(INT64);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(BOOLEAN);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(FP8_E4M3);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(FP8_E5M2);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(FAST_FLOAT_FOR_FP8);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(FP8_E8M0);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(FP4_E2M1);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(INT4);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(COMPLEX_FP32);
  MUDNN_FRONTEND_PARSE_DATA_TYPE(COMPLEX_FP64);
#undef MUDNN_FRONTEND_PARSE_DATA_TYPE
  throw std::invalid_argument("unknown cudnn-frontend data type " + name);
}

inline const char* upstream_pointwise_mode_name(mudnnPointwiseMode_t mode) {
  switch (mode) {
    case MUDNN_POINTWISE_ADD: return "ADD";
    case MUDNN_POINTWISE_MUL: return "MUL";
    case MUDNN_POINTWISE_MIN: return "MIN";
    case MUDNN_POINTWISE_MAX: return "MAX";
    case MUDNN_POINTWISE_SQRT: return "SQRT";
    case MUDNN_POINTWISE_ADD_SQUARE: return "ADD_SQUARE";
    case MUDNN_POINTWISE_DIV: return "DIV";
    case MUDNN_POINTWISE_MOD: return "MOD";
    case MUDNN_POINTWISE_POW: return "POW";
    case MUDNN_POINTWISE_SUB: return "SUB";
    case MUDNN_POINTWISE_ABS: return "ABS";
    case MUDNN_POINTWISE_CEIL: return "CEIL";
    case MUDNN_POINTWISE_COS: return "COS";
    case MUDNN_POINTWISE_EXP: return "EXP";
    case MUDNN_POINTWISE_FLOOR: return "FLOOR";
    case MUDNN_POINTWISE_LOG: return "LOG";
    case MUDNN_POINTWISE_NEG: return "NEG";
    case MUDNN_POINTWISE_RSQRT: return "RSQRT";
    case MUDNN_POINTWISE_SIN: return "SIN";
    case MUDNN_POINTWISE_TAN: return "TAN";
    case MUDNN_POINTWISE_ERF: return "ERF";
    case MUDNN_POINTWISE_IDENTITY: return "IDENTITY";
    case MUDNN_POINTWISE_RECIPROCAL: return "RECIPROCAL";
    case MUDNN_POINTWISE_RELU_FWD: return "RELU_FWD";
    case MUDNN_POINTWISE_TANH_FWD: return "TANH_FWD";
    case MUDNN_POINTWISE_SIGMOID_FWD: return "SIGMOID_FWD";
    case MUDNN_POINTWISE_ELU_FWD: return "ELU_FWD";
    case MUDNN_POINTWISE_GELU_FWD: return "GELU_FWD";
    case MUDNN_POINTWISE_SOFTPLUS_FWD: return "SOFTPLUS_FWD";
    case MUDNN_POINTWISE_SWISH_FWD: return "SWISH_FWD";
    case MUDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return "GELU_APPROX_TANH_FWD";
    case MUDNN_POINTWISE_RELU_BWD: return "RELU_BWD";
    case MUDNN_POINTWISE_TANH_BWD: return "TANH_BWD";
    case MUDNN_POINTWISE_SIGMOID_BWD: return "SIGMOID_BWD";
    case MUDNN_POINTWISE_ELU_BWD: return "ELU_BWD";
    case MUDNN_POINTWISE_GELU_BWD: return "GELU_BWD";
    case MUDNN_POINTWISE_SOFTPLUS_BWD: return "SOFTPLUS_BWD";
    case MUDNN_POINTWISE_SWISH_BWD: return "SWISH_BWD";
    case MUDNN_POINTWISE_GELU_APPROX_TANH_BWD:
      return "GELU_APPROX_TANH_BWD";
    case MUDNN_POINTWISE_GEN_INDEX: return "GEN_INDEX";
    case MUDNN_POINTWISE_BINARY_SELECT: return "BINARY_SELECT";
    case MUDNN_POINTWISE_CMP_EQ: return "CMP_EQ";
    case MUDNN_POINTWISE_CMP_NEQ: return "CMP_NEQ";
    case MUDNN_POINTWISE_CMP_GT: return "CMP_GT";
    case MUDNN_POINTWISE_CMP_GE: return "CMP_GE";
    case MUDNN_POINTWISE_CMP_LT: return "CMP_LT";
    case MUDNN_POINTWISE_CMP_LE: return "CMP_LE";
    case MUDNN_POINTWISE_LOGICAL_AND: return "LOGICAL_AND";
    case MUDNN_POINTWISE_LOGICAL_OR: return "LOGICAL_OR";
    case MUDNN_POINTWISE_LOGICAL_NOT: return "LOGICAL_NOT";
    default: return "NOT_SET";
  }
}

inline void to_json(nlohmann::json& j, const Tensor_attributes& tensor) {
  j = nlohmann::json{
      {"name", tensor.get_name()},
      {"data_type", upstream_data_type_name(tensor.get_data_type())},
      {"dim", tensor.get_dim()},
      {"stride", tensor.get_stride()},
      {"is_virtual", tensor.is_virtual()},
      {"pass_by_value", nullptr},
      {"is_pass_by_value", tensor.is_pass_by_value()},
      {"reordering_type", "NONE"},
      {"uid", tensor.get_uid()},
      {"uid_assigned", tensor.has_uid()}};
  if (tensor.get_ragged_offset()) {
    j["ragged_offset_uid"] = tensor.get_ragged_offset()->get_uid();
    j["ragged_offset_name"] = tensor.get_ragged_offset()->get_name();
  }
}

inline void from_json(const nlohmann::json& j, Tensor_attributes& tensor) {
  tensor = Tensor_attributes();
  tensor.set_name(j.value("name", std::string{}))
      .set_dim(j.value("dim", j.value("dims", std::vector<int64_t>{})))
      .set_stride(
          j.value("stride", j.value("strides", std::vector<int64_t>{})))
      .set_data_type(upstream_data_type_from_json(
          j.contains("data_type") ? j.at("data_type") : nlohmann::json()))
      .set_is_virtual(j.value("is_virtual", false))
      .set_is_pass_by_value(j.value("is_pass_by_value", false))
      .set_alignment(j.value("alignment", int64_t{16}))
      .set_vector_count_and_dimension(j.value("vector_count", int64_t{1}),
                                      j.value("vector_dimension", int64_t{-1}))
      .set_ragged_offset_multiplier(
          j.value("ragged_offset_multiplier", int64_t{1}));
  if (j.value("uid_assigned", j.value("has_uid", false))) {
    tensor.set_uid(j.value("uid", int64_t{0}));
  }
  if (j.value("is_output", false)) {
    tensor.set_output(true);
  }
}

inline void to_json(nlohmann::json& j, const Conv_attributes& attributes) {
  j = nlohmann::json{
      {"name", attributes.get_name()},
      {"padding", attributes.get_padding()},
      {"pre_padding", attributes.get_pre_padding()},
      {"post_padding", attributes.get_post_padding()},
      {"stride", attributes.get_stride()},
      {"dilation", attributes.get_dilation()},
      {"mode", static_cast<int64_t>(attributes.get_mode())},
      {"compute_data_type",
       static_cast<int64_t>(attributes.get_compute_data_type())},
      {"output_data_type",
       static_cast<int64_t>(attributes.get_output_data_type())},
      {"has_output_data_type", attributes.has_output_data_type()},
      {"group_count", attributes.get_group_count()},
      {"has_group_count", attributes.has_group_count()},
      {"alpha", attributes.get_alpha()},
      {"beta", attributes.get_beta()}};
}

inline void from_json(const nlohmann::json& j, Conv_attributes& attributes) {
  attributes = Conv_attributes();
  const auto padding = j.value("padding", std::vector<int64_t>{0, 0});
  attributes.set_name(j.value("name", std::string{}))
      .set_padding(padding)
      .set_pre_padding(j.value("pre_padding", padding))
      .set_post_padding(j.value("post_padding", padding))
      .set_stride(j.value("stride", std::vector<int64_t>{1, 1}))
      .set_dilation(j.value("dilation", std::vector<int64_t>{1, 1}))
      .set_mode(static_cast<mudnnConvolutionMode_t>(
          j.value("mode", static_cast<int64_t>(MUDNN_CROSS_CORRELATION))))
      .set_compute_data_type(static_cast<mudnnDataType_t>(
          j.value("compute_data_type", static_cast<int64_t>(MUDNN_DATA_FLOAT))))
      .set_alpha(j.value("alpha", 1.0))
      .set_beta(j.value("beta", 0.0));
  if (j.value("has_group_count", false)) {
    attributes.set_group_count(j.value("group_count", int64_t{1}));
  }
  if (j.value("has_output_data_type", false)) {
    attributes.set_output_data_type(static_cast<mudnnDataType_t>(
        j.value("output_data_type", static_cast<int64_t>(MUDNN_DATA_FLOAT))));
  }
}

inline void to_json(nlohmann::json& j, const Graph& graph) {
  const auto data = graph.serialize();
  j = nlohmann::json{
      {"context",
       {{"name", graph.name_},
        {"compute_data_type",
         upstream_data_type_name(graph.compute_data_type_)},
        {"intermediate_data_type",
         upstream_data_type_name(graph.intermediate_data_type_)},
        {"io_data_type", upstream_data_type_name(graph.io_data_type_)},
        {"sm_count", graph.sm_count_},
        {"is_dynamic_shape_enabled", graph.dynamic_shape_enabled_},
        {"is_override_shape_enabled", graph.override_shape_enabled_}}},
      {"json_version", "1.0"},
      {"cudnn_backend_version",
       detail::convert_version_to_str(detail::get_backend_version())},
      {"cudnn_frontend_version", "0.0.1"},
      {"nodes", nlohmann::json::array()},
      {"tensors", nlohmann::json::object()},
      {"mudnn_payload", std::string(data.begin(), data.end())}};

  for (const auto& tensor : graph.tensors_) {
    j["tensors"][std::to_string(tensor->get_uid())] = *tensor;
  }

  const auto ref = [](const std::shared_ptr<Tensor_attributes>& tensor) {
    return tensor ? nlohmann::json(tensor->get_uid()) : nlohmann::json();
  };
  const auto reduction_mode = [](mudnnReduceTensorOp_t mode) {
    switch (mode) {
      case MUDNN_REDUCE_TENSOR_ADD: return "ADD";
      case MUDNN_REDUCE_TENSOR_MUL: return "MUL";
      case MUDNN_REDUCE_TENSOR_MIN: return "MIN";
      case MUDNN_REDUCE_TENSOR_MAX: return "MAX";
      case MUDNN_REDUCE_TENSOR_AMAX: return "AMAX";
      case MUDNN_REDUCE_TENSOR_AVG: return "AVG";
      case MUDNN_REDUCE_TENSOR_NORM1: return "NORM1";
      case MUDNN_REDUCE_TENSOR_NORM2: return "NORM2";
      case MUDNN_REDUCE_TENSOR_MUL_NO_ZEROS: return "MUL_NO_ZEROS";
      default: return "NOT_SET";
    }
  };
  const auto resample_mode = [](mudnnResampleMode_t mode) {
    switch (mode) {
      case MUDNN_RESAMPLE_AVGPOOL_EXCLUDE_PADDING:
        return "AVGPOOL_EXCLUDE_PADDING";
      case MUDNN_RESAMPLE_AVGPOOL_INCLUDE_PADDING:
        return "AVGPOOL_INCLUDE_PADDING";
      case MUDNN_RESAMPLE_BILINEAR: return "BILINEAR";
      case MUDNN_RESAMPLE_NEAREST: return "NEAREST";
      case MUDNN_RESAMPLE_MAXPOOL: return "MAXPOOL";
      default: return "NOT_SET";
    }
  };

  for (const auto& node : graph.nodes_) {
    nlohmann::json item{{"name", node.name},
                        {"compute_data_type",
                         upstream_data_type_name(node.compute_data_type)},
                        {"inputs", nlohmann::json::object()},
                        {"outputs", nlohmann::json::object()}};
    switch (node.kind) {
      case Graph::NodeKind::kPointwise:
        item["tag"] = "POINTWISE";
        if (!node.inputs.empty()) item["inputs"]["IN_0"] = ref(node.inputs[0]);
        if (node.inputs.size() > 1) item["inputs"]["IN_1"] = ref(node.inputs[1]);
        if (node.inputs.size() > 2) item["inputs"]["IN_2"] = ref(node.inputs[2]);
        item["outputs"]["OUT_0"] = ref(node.output);
        item["mode"] = upstream_pointwise_mode_name(node.pointwise_mode);
        item["axis"] = node.pointwise_has_axis
                            ? nlohmann::json(node.pointwise_axis)
                            : nlohmann::json();
        break;
      case Graph::NodeKind::kMatmul:
        item["tag"] = "MATMUL";
        item["inputs"]["A"] = ref(node.inputs.at(0));
        item["inputs"]["B"] = ref(node.inputs.at(1));
        item["outputs"]["C"] = ref(node.output);
        item["padding_value"] = 0.0;
        break;
      case Graph::NodeKind::kConvFprop:
      case Graph::NodeKind::kConvBwdData:
      case Graph::NodeKind::kConvBwdFilter: {
        const bool fprop = node.kind == Graph::NodeKind::kConvFprop;
        const bool dgrad = node.kind == Graph::NodeKind::kConvBwdData;
        item["tag"] = fprop ? "CONV_FPROP" : (dgrad ? "CONV_DGRAD" : "CONV_WGRAD");
        item["inputs"][fprop ? "X" : "DY"] = ref(node.inputs.at(0));
        item["inputs"][fprop || dgrad ? "W" : "X"] = ref(node.inputs.at(1));
        item["outputs"][fprop ? "Y" : (dgrad ? "DX" : "DW")] = ref(node.output);
        item["pre_padding"] = node.param0;
        item["post_padding"] = node.param1;
        item["stride"] = node.param2;
        item["dilation"] = node.param3;
        item["math_mode"] = node.conv_mode == MUDNN_CONVOLUTION
                                  ? "CONVOLUTION"
                                  : "CROSS_CORRELATION";
        break;
      }
      case Graph::NodeKind::kReduction:
        item["tag"] = "REDUCTION";
        item["inputs"]["X"] = ref(node.inputs.at(0));
        item["outputs"]["Y"] = ref(node.output);
        item["mode"] = reduction_mode(node.reduction_mode);
        item["is_deterministic"] = false;
        break;
      case Graph::NodeKind::kGenstats:
        item["tag"] = "GENSTATS";
        item["inputs"]["X"] = ref(node.inputs.at(0));
        item["outputs"]["SUM"] = ref(node.output);
        item["outputs"]["SQ_SUM"] = ref(node.stats);
        break;
      case Graph::NodeKind::kResampleFwd:
        item["tag"] = "RESAMPLE";
        item["inputs"]["X"] = ref(node.inputs.at(0));
        item["outputs"]["Y"] = ref(node.output);
        if (node.stats) item["outputs"]["Index"] = ref(node.stats);
        item["generate_index"] = node.resample_has_indices;
        item["resample_mode"] = resample_mode(node.resample_mode);
        item["padding_mode"] = "ZERO_PAD";
        item["pre_padding"] = node.param0;
        item["post_padding"] = node.param1;
        item["stride"] = node.param2;
        item["window"] = node.param3;
        break;
      case Graph::NodeKind::kSdpa:
        item["tag"] = "SDPA";
        item["inputs"]["Q"] = ref(node.inputs.at(0));
        item["inputs"]["K"] = ref(node.inputs.at(1));
        item["inputs"]["V"] = ref(node.inputs.at(2));
        item["outputs"]["O"] = ref(node.output);
        if (node.stats) item["outputs"]["Stats"] = ref(node.stats);
        item["generate_stats"] = static_cast<bool>(node.stats);
        item["padding_mask"] = node.sdpa_use_padding_mask;
        item["attn_scale_value"] = node.sdpa_has_scale
                                         ? nlohmann::json(node.sdpa_scale)
                                         : nlohmann::json();
        item["implementation"] = "UNIFIED";
        break;
      case Graph::NodeKind::kSlice:
        item["tag"] = "SLICE";
        item["inputs"]["X"] = ref(node.inputs.at(0));
        item["outputs"]["Y"] = ref(node.output);
        break;
      default:
        item["tag"] = "MUDNN_" + std::to_string(static_cast<int>(node.kind));
        for (size_t i = 0; i < node.inputs.size(); ++i) {
          item["inputs"][std::to_string(i)] = ref(node.inputs[i]);
        }
        item["outputs"]["0"] = ref(node.output);
        break;
    }
    j["nodes"].push_back(std::move(item));
  }
}
#endif

inline std::ostream& operator<<(std::ostream& os, const Graph& graph) {
  os << graph.print();
  return os;
}

}  // namespace graph
