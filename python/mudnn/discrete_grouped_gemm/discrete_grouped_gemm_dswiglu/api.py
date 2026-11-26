import os

try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_DiscreteGroupedGemmDswigluSm100Objects = {}


class DiscreteGroupedGemmDswigluSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def check_support(self):
        raise NotImplementedError(
            "mudnn.DiscreteGroupedGemmDswigluSm100 is not supported by muDNN frontend yet"
        )

    def compile(self):
        raise NotImplementedError(
            "mudnn.DiscreteGroupedGemmDswigluSm100 is not supported by muDNN frontend yet"
        )

    def execute(self, **kwargs):
        raise NotImplementedError(
            "mudnn.DiscreteGroupedGemmDswigluSm100 is not supported by muDNN frontend yet"
        )


def _shape_of(tensor):
    return tuple(getattr(tensor, "shape", ()))


def _dtype_of(tensor):
    return getattr(tensor, "dtype", None)


def _stride_of(tensor):
    stride = getattr(tensor, "stride", None)
    if callable(stride):
        try:
            return tuple(stride())
        except TypeError:
            return None
    if stride is not None:
        return tuple(stride)
    shape = _shape_of(tensor)
    if not shape:
        return ()
    running = 1
    strides = []
    for dim in reversed(shape):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


def _stride_order_of(tensor):
    strides = _stride_of(tensor)
    return tuple(
        index for index, _ in sorted(enumerate(strides), key=lambda item: item[1])
    )


def _valid_m(a_tensor):
    shape = _shape_of(a_tensor)
    return shape[0] if shape else None


def _cache_key(**kwargs):
    a_tensor = kwargs.get("a_tensor")
    c_tensor = kwargs.get("c_tensor")
    full_dynamic = os.getenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"
    return (
        full_dynamic,
        _shape_of(a_tensor)[1:] if not full_dynamic else None,
        _shape_of(c_tensor)[1:] if not full_dynamic else None,
        _shape_of(kwargs.get("b_ptrs")) if not full_dynamic else None,
        _shape_of(kwargs.get("sfb_ptrs")) if not full_dynamic else None,
        kwargs.get("n") if not full_dynamic else None,
        _dtype_of(a_tensor),
        _dtype_of(c_tensor),
        _dtype_of(kwargs.get("b_ptrs")),
        kwargs.get("b_dtype"),
        _stride_order_of(a_tensor),
        _stride_order_of(c_tensor),
        _dtype_of(kwargs.get("sfa_tensor")),
        _dtype_of(kwargs.get("sfb_ptrs")),
        _shape_of(kwargs.get("padded_offsets")) if not full_dynamic else None,
        _stride_of(kwargs.get("padded_offsets")) if not full_dynamic else None,
        _dtype_of(kwargs.get("padded_offsets")),
        _dtype_of(kwargs.get("norm_const_tensor")),
        kwargs.get("acc_dtype"),
        kwargs.get("d_dtype"),
        kwargs.get("cd_major", "n"),
        tuple(kwargs.get("mma_tiler_mn") or ()),
        tuple(kwargs.get("cluster_shape_mn") or ()),
        kwargs.get("sf_vec_size"),
        kwargs.get("vector_f32"),
        kwargs.get("m_aligned"),
        kwargs.get("discrete_col_sfd"),
        kwargs.get("act_func"),
        kwargs.get("b_major"),
    )


def _tuple_dict(**kwargs):
    return TupleDict(**kwargs)


def discrete_grouped_gemm_dswiglu_wrapper_sm100(**kwargs):
    outputs = _tuple_dict(
        d_row_tensor=kwargs.get("d_row_tensor"),
        d_col_tensor=kwargs.get("d_col_tensor"),
        dprob_tensor=kwargs.get("dprob_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
        sfd_row_tensor=kwargs.get("sfd_row_tensor"),
        sfd_col_tensor=kwargs.get("sfd_col_tensor"),
    )
    if _valid_m(kwargs.get("a_tensor")) == 0:
        return outputs

    key = _cache_key(**kwargs)
    cached = _cache_of_DiscreteGroupedGemmDswigluSm100Objects.get(key)
    if cached is None:
        op = DiscreteGroupedGemmDswigluSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b_ptrs=kwargs.get("b_ptrs"),
            sample_c=kwargs.get("c_tensor"),
            sample_sfa=kwargs.get("sfa_tensor"),
            sample_sfb_ptrs=kwargs.get("sfb_ptrs"),
            sample_padded_offsets=kwargs.get("padded_offsets"),
            sample_alpha=kwargs.get("alpha_tensor"),
            sample_beta=kwargs.get("beta_tensor"),
            sample_prob=kwargs.get("prob_tensor"),
            sample_dprob=kwargs.get("dprob_tensor"),
            sample_norm_const=kwargs.get("norm_const_tensor"),
            n=kwargs.get("n"),
            b_dtype=kwargs.get("b_dtype"),
            acc_dtype=kwargs.get("acc_dtype"),
            mma_tiler_mn=kwargs.get("mma_tiler_mn"),
            cluster_shape_mn=kwargs.get("cluster_shape_mn"),
            sf_vec_size=kwargs.get("sf_vec_size"),
            vector_f32=kwargs.get("vector_f32"),
            m_aligned=kwargs.get("m_aligned"),
            discrete_col_sfd=kwargs.get("discrete_col_sfd"),
            act_func=kwargs.get("act_func"),
            b_major=kwargs.get("b_major"),
        )
        try:
            is_supported = op.check_support()
            if not is_supported:
                raise NotImplementedError
            op.compile()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "mudnn.discrete_grouped_gemm_dswiglu_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_DiscreteGroupedGemmDswigluSm100Objects[key] = (
            op,
            kwargs.get("amax_tensor"),
            kwargs.get("beta_tensor"),
        )
    else:
        op = cached[0]
    op.execute(**kwargs)
    return outputs
