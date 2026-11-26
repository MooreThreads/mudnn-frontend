import os

try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict

from mudnn.grouped_gemm.grouped_gemm_dswiglu.api import (
    _compute_grouped_dswiglu,
    _empty_output,
    _grouped_shapes,
    _is_supported_tensor,
    _torch,
)


_cache_of_GroupedGemmDgluSm100Objects = {}


class GroupedGemmDgluSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if (self.kwargs.get("act_func") or "dswiglu").lower() != "dswiglu":
            raise NotImplementedError(
                "mudnn.GroupedGemmDgluSm100 torch fallback supports only act_func='dswiglu'"
            )
        if (
            self.kwargs.get("sample_a") is None
            or self.kwargs.get("sample_b") is None
            or self.kwargs.get("sample_c") is None
        ):
            raise NotImplementedError(
                "mudnn.GroupedGemmDgluSm100 requires sample_a, sample_b, and "
                "sample_c tensors"
            )
        return (
            _is_supported_tensor(self.kwargs.get("sample_a"))
            and _is_supported_tensor(self.kwargs.get("sample_b"))
            and _is_supported_tensor(self.kwargs.get("sample_c"))
        )

    def compile(self):
        if not self.check_support():
            raise NotImplementedError(
                "mudnn.GroupedGemmDgluSm100 torch fallback supports only FLOAT, "
                "HALF, and BFLOAT16 tensors"
            )
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if (
            kwargs.get("act_func") or self.kwargs.get("act_func") or "dswiglu"
        ).lower() != "dswiglu":
            raise NotImplementedError(
                "mudnn.GroupedGemmDgluSm100 torch fallback supports only act_func='dswiglu'"
            )
        if not self._compiled:
            self.compile()
        d_f32, dprob_f32, dbias_f32, amax_f32 = _compute_grouped_dswiglu(**kwargs)
        d_row_tensor = kwargs.get("d_row_tensor")
        if d_row_tensor is None:
            d_row_tensor = kwargs.get("d_tensor")
        if d_row_tensor is None:
            raise ValueError("GroupedGemmDglu execute requires d_row_tensor")
        d_row_tensor.copy_(d_f32.to(dtype=d_row_tensor.dtype))
        d_col_tensor = kwargs.get("d_col_tensor")
        if d_col_tensor is not None and tuple(d_col_tensor.shape) == tuple(d_f32.shape):
            d_col_tensor.copy_(d_f32.to(dtype=d_col_tensor.dtype))
        dprob_tensor = kwargs.get("dprob_tensor")
        if dprob_tensor is not None:
            dprob_tensor.copy_(dprob_f32.to(dtype=dprob_tensor.dtype))
        dbias_tensor = kwargs.get("dbias_tensor")
        if dbias_tensor is not None:
            dbias_tensor.copy_(dbias_f32.to(dtype=dbias_tensor.dtype))
        amax_tensor = kwargs.get("amax_tensor")
        if amax_tensor is not None:
            amax_tensor.copy_(amax_f32.reshape_as(amax_tensor))
        return self


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


def _expert_count(b_tensor, b_ptrs):
    if b_tensor is not None and len(_shape_of(b_tensor)) >= 3:
        return _shape_of(b_tensor)[2]
    shape = _shape_of(b_ptrs)
    return shape[0] if shape else None


def _cache_key(**kwargs):
    a_tensor = kwargs.get("a_tensor")
    b_tensor = kwargs.get("b_tensor")
    b_ptrs = kwargs.get("b_ptrs")
    c_tensor = kwargs.get("c_tensor")
    is_dense = b_tensor is not None
    full_dynamic = (
        is_dense and os.getenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"
    )
    if full_dynamic:
        shape_key = ("full_dynamic", _expert_count(b_tensor, b_ptrs))
    elif is_dense:
        shape_key = (
            "partial_dynamic",
            _shape_of(a_tensor)[1:],
            _shape_of(b_tensor)[1:],
            _shape_of(c_tensor)[1:],
            _expert_count(b_tensor, b_ptrs),
        )
    else:
        shape_key = (
            "discrete_dynamic_m",
            _shape_of(a_tensor)[1:],
            _shape_of(c_tensor)[1:],
            kwargs.get("n"),
            _expert_count(b_tensor, b_ptrs),
        )
    return (
        "dense" if is_dense else "discrete",
        shape_key,
        _dtype_of(a_tensor),
        _dtype_of(b_tensor) if is_dense else kwargs.get("b_dtype"),
        _dtype_of(c_tensor),
        _stride_order_of(a_tensor),
        _stride_order_of(b_tensor) if is_dense else _stride_order_of(b_ptrs),
        _stride_order_of(c_tensor),
        _dtype_of(kwargs.get("sfa_tensor")),
        (
            _dtype_of(kwargs.get("sfb_tensor"))
            if is_dense
            else _dtype_of(kwargs.get("sfb_ptrs"))
        ),
        _dtype_of(kwargs.get("alpha_tensor")),
        _dtype_of(kwargs.get("beta_tensor")),
        _dtype_of(kwargs.get("prob_tensor")),
        _dtype_of(kwargs.get("dprob_tensor")),
        _dtype_of(kwargs.get("norm_const_tensor")),
        _shape_of(kwargs.get("padded_offsets")) if not full_dynamic else None,
        _stride_of(kwargs.get("padded_offsets")) if not full_dynamic else None,
        _dtype_of(kwargs.get("padded_offsets")),
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
        kwargs.get("linear_offset"),
    )


def _tuple_dict(**kwargs):
    return TupleDict(**kwargs)


def grouped_gemm_dglu_wrapper_sm100(**kwargs):
    if _valid_m(kwargs.get("a_tensor")) == 0:
        return _tuple_dict(
            d_row_tensor=kwargs.get("d_row_tensor"),
            d_col_tensor=kwargs.get("d_col_tensor"),
            dprob_tensor=kwargs.get("dprob_tensor"),
            dbias_tensor=kwargs.get("dbias_tensor"),
            amax_tensor=kwargs.get("amax_tensor"),
            sfd_row_tensor=kwargs.get("sfd_row_tensor"),
            sfd_col_tensor=kwargs.get("sfd_col_tensor"),
        )
    a_tensor = kwargs.get("a_tensor")
    b_tensor = kwargs.get("b_tensor")
    c_tensor = kwargs.get("c_tensor")
    if (
        _is_supported_tensor(a_tensor)
        and _is_supported_tensor(b_tensor)
        and _is_supported_tensor(c_tensor)
    ):
        if (kwargs.get("act_func") or "dswiglu").lower() != "dswiglu":
            raise NotImplementedError(
                "mudnn.grouped_gemm_dglu_wrapper_sm100 torch fallback supports only "
                "act_func='dswiglu'"
            )
        m, n, _k, l, _offsets = _grouped_shapes(
            a_tensor, b_tensor, c_tensor, kwargs.get("padded_offsets")
        )
        if kwargs.get("d_row_tensor") is None:
            kwargs["d_row_tensor"] = _empty_output(
                (m, n * 2, 1), kwargs.get("d_dtype"), c_tensor
            )
        if kwargs.get("dprob_tensor") is None:
            torch = _torch()
            kwargs["dprob_tensor"] = torch.empty(
                (m, 1, 1),
                dtype=torch.float32,
                device=getattr(c_tensor, "device", None),
            )
        if kwargs.get("amax_tensor") is None and kwargs.get("d_dtype") is not None:
            torch = _torch()
            kwargs["amax_tensor"] = torch.empty(
                (l, 2, 1), dtype=torch.float32, device=getattr(c_tensor, "device", None)
            )

    key = _cache_key(**kwargs)
    op = _cache_of_GroupedGemmDgluSm100Objects.get(key)
    if op is None:
        op = GroupedGemmDgluSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b=kwargs.get("b_tensor"),
            sample_c=kwargs.get("c_tensor"),
            sample_sfa=kwargs.get("sfa_tensor"),
            sample_sfb=kwargs.get("sfb_tensor"),
            sample_padded_offsets=kwargs.get("padded_offsets"),
            sample_alpha=kwargs.get("alpha_tensor"),
            sample_beta=kwargs.get("beta_tensor"),
            sample_prob=kwargs.get("prob_tensor"),
            sample_dprob=kwargs.get("dprob_tensor"),
            sample_norm_const=kwargs.get("norm_const_tensor"),
            acc_dtype=kwargs.get("acc_dtype"),
            mma_tiler_mn=kwargs.get("mma_tiler_mn"),
            cluster_shape_mn=kwargs.get("cluster_shape_mn"),
            sf_vec_size=kwargs.get("sf_vec_size"),
            vector_f32=kwargs.get("vector_f32"),
            m_aligned=kwargs.get("m_aligned"),
            discrete_col_sfd=kwargs.get("discrete_col_sfd"),
            act_func=kwargs.get("act_func"),
            linear_offset=kwargs.get("linear_offset"),
        )
        try:
            is_supported = op.check_support()
            if not is_supported:
                raise NotImplementedError
            op.compile()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "mudnn.grouped_gemm_dglu_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GroupedGemmDgluSm100Objects[key] = op
    op.execute(**kwargs)
    return _tuple_dict(
        d_row_tensor=kwargs.get("d_row_tensor"),
        d_col_tensor=kwargs.get("d_col_tensor"),
        dprob_tensor=kwargs.get("dprob_tensor"),
        dbias_tensor=kwargs.get("dbias_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
        sfd_row_tensor=kwargs.get("sfd_row_tensor"),
        sfd_col_tensor=kwargs.get("sfd_col_tensor"),
    )
