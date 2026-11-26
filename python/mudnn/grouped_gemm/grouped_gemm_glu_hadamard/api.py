try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_GroupedGemmGluHadamardSm100Objects = {}


from mudnn.grouped_gemm.grouped_gemm_glu.api import (
    _compute_grouped_glu,
    _empty_output,
    _grouped_shapes,
    _is_supported_tensor,
    _torch,
)

from .hadamard_utils import HADAMARD_SIZE, hadamard_matrix


def _compute_post_rht_amax(d_f32, offsets):
    torch = _torch()
    hadamard = hadamard_matrix(HADAMARD_SIZE, dtype=torch.float32, device=d_f32.device)
    amax = torch.empty((len(offsets), 1), dtype=torch.float32, device=d_f32.device)
    start = 0
    for expert, end in enumerate(offsets):
        group_m = end - start
        if group_m == 0:
            amax[expert, 0] = 0.0
        elif group_m % HADAMARD_SIZE != 0:
            raise NotImplementedError(
                "GroupedGemmGluHadamard post-RHT fallback requires expert row "
                "counts divisible by 16"
            )
        else:
            n_out = int(d_f32.shape[1])
            group_t = d_f32[start:end, :, 0].to(torch.bfloat16).to(torch.float32)
            group_t = group_t.T.contiguous().view(
                n_out, group_m // HADAMARD_SIZE, HADAMARD_SIZE
            )
            transformed = (group_t @ hadamard) * (HADAMARD_SIZE**-0.5)
            amax[expert, 0] = torch.amax(torch.abs(transformed))
        start = end
    return amax


class GroupedGemmGluHadamardSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if self.kwargs.get("sample_a") is None or self.kwargs.get("sample_b") is None:
            raise NotImplementedError(
                "mudnn.GroupedGemmGluHadamardSm100 requires sample_a and sample_b tensors"
            )
        if not _is_supported_tensor(
            self.kwargs.get("sample_a")
        ) or not _is_supported_tensor(self.kwargs.get("sample_b")):
            raise NotImplementedError(
                "mudnn.GroupedGemmGluHadamardSm100 torch fallback supports only "
                "FLOAT, HALF, and BFLOAT16 tensors"
            )
        return True

    def compile(self):
        self.check_support()
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        runtime_kwargs = {**self.kwargs, **kwargs}
        c_f32, d_f32, amax_f32 = _compute_grouped_glu(**runtime_kwargs)
        c_tensor = kwargs.get("c_tensor")
        d_tensor = kwargs.get("d_tensor")
        if c_tensor is None or d_tensor is None:
            raise ValueError(
                "GroupedGemmGluHadamard execute requires c_tensor and d_tensor"
            )
        c_tensor.copy_(c_f32.to(dtype=c_tensor.dtype))
        d_tensor.copy_(d_f32.to(dtype=d_tensor.dtype))
        amax_tensor = kwargs.get("amax_tensor")
        if amax_tensor is not None:
            amax_tensor.copy_(amax_f32.reshape_as(amax_tensor))
        post_rht_amax_tensor = kwargs.get("post_rht_amax_tensor")
        if post_rht_amax_tensor is not None:
            _m, _n, _k, _l, offsets = _grouped_shapes(
                runtime_kwargs.get("a_tensor"),
                runtime_kwargs.get("b_tensor"),
                runtime_kwargs.get("padded_offsets"),
            )
            post_amax = _compute_post_rht_amax(d_f32, offsets)
            post_rht_amax_tensor.copy_(post_amax.reshape_as(post_rht_amax_tensor))
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
    d_tensor = kwargs.get("d_tensor")
    is_dense = b_tensor is not None
    return (
        "dense" if is_dense else "discrete",
        kwargs.get("act_func"),
        _shape_of(a_tensor)[1:],
        _shape_of(b_tensor)[1:] if is_dense else kwargs.get("n"),
        _expert_count(b_tensor, b_ptrs),
        _shape_of(c_tensor)[1:],
        _shape_of(d_tensor)[1:],
        _dtype_of(a_tensor),
        _dtype_of(b_tensor) if is_dense else kwargs.get("b_dtype"),
        _dtype_of(c_tensor),
        _dtype_of(d_tensor),
        _stride_order_of(a_tensor),
        _stride_order_of(b_tensor) if is_dense else kwargs.get("b_major", "k"),
        _stride_order_of(c_tensor),
        _dtype_of(kwargs.get("sfa_tensor")),
        (
            _dtype_of(kwargs.get("sfb_tensor"))
            if is_dense
            else _dtype_of(kwargs.get("sfb_ptrs"))
        ),
        _dtype_of(kwargs.get("alpha_tensor")),
        _dtype_of(kwargs.get("prob_tensor")),
        _dtype_of(kwargs.get("bias_tensor")),
        _shape_of(kwargs.get("padded_offsets")),
        _stride_of(kwargs.get("padded_offsets")),
        _dtype_of(kwargs.get("padded_offsets")),
        kwargs.get("acc_dtype"),
        tuple(kwargs.get("mma_tiler_mn") or ()),
        tuple(kwargs.get("cluster_shape_mn") or ()),
        kwargs.get("sf_vec_size"),
        kwargs.get("vector_f32"),
        kwargs.get("m_aligned"),
        kwargs.get("use_dynamic_sched"),
        kwargs.get("use_tmem_post_rht_amax"),
    )


def _tuple_dict(**kwargs):
    return TupleDict(**kwargs)


def grouped_gemm_glu_hadamard_wrapper_sm100(**kwargs):
    if _valid_m(kwargs.get("a_tensor")) == 0:
        return _tuple_dict(
            c_tensor=kwargs.get("c_tensor"),
            d_tensor=kwargs.get("d_tensor"),
            amax_tensor=kwargs.get("amax_tensor"),
            post_rht_amax_tensor=kwargs.get("post_rht_amax_tensor"),
        )

    a_tensor = kwargs.get("a_tensor")
    b_tensor = kwargs.get("b_tensor")
    if _is_supported_tensor(a_tensor) and _is_supported_tensor(b_tensor):
        m, n, _k, l, _offsets = _grouped_shapes(
            a_tensor, b_tensor, kwargs.get("padded_offsets")
        )
        if kwargs.get("c_tensor") is None:
            kwargs["c_tensor"] = _empty_output(
                (m, n, 1), kwargs.get("c_dtype"), a_tensor
            )
        if kwargs.get("d_tensor") is None:
            kwargs["d_tensor"] = _empty_output(
                (m, n // 2, 1), kwargs.get("d_dtype"), a_tensor
            )
        if kwargs.get("amax_tensor") is None and kwargs.get("d_dtype") is not None:
            torch = _torch()
            kwargs["amax_tensor"] = torch.empty(
                (l, 1), dtype=torch.float32, device=getattr(a_tensor, "device", None)
            )
        if kwargs.get("post_rht_amax_tensor") is None and kwargs.get(
            "use_tmem_post_rht_amax"
        ):
            torch = _torch()
            kwargs["post_rht_amax_tensor"] = torch.empty(
                (l, 1), dtype=torch.float32, device=getattr(a_tensor, "device", None)
            )

    key = _cache_key(**kwargs)
    op = _cache_of_GroupedGemmGluHadamardSm100Objects.get(key)
    if op is None:
        op = GroupedGemmGluHadamardSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b=kwargs.get("b_tensor"),
            sample_c=kwargs.get("c_tensor"),
            sample_d=kwargs.get("d_tensor"),
            sample_sfa=kwargs.get("sfa_tensor"),
            sample_sfb=kwargs.get("sfb_tensor"),
            sample_padded_offsets=kwargs.get("padded_offsets"),
            sample_alpha=kwargs.get("alpha_tensor"),
            sample_prob=kwargs.get("prob_tensor"),
            sample_bias=kwargs.get("bias_tensor"),
            acc_dtype=kwargs.get("acc_dtype"),
            mma_tiler_mn=kwargs.get("mma_tiler_mn"),
            cluster_shape_mn=kwargs.get("cluster_shape_mn"),
            sf_vec_size=kwargs.get("sf_vec_size"),
            vector_f32=kwargs.get("vector_f32"),
            m_aligned=kwargs.get("m_aligned"),
            act_func=kwargs.get("act_func"),
            use_dynamic_sched=kwargs.get("use_dynamic_sched"),
            use_tmem_post_rht_amax=kwargs.get("use_tmem_post_rht_amax"),
        )
        try:
            is_supported = op.check_support()
            if not is_supported:
                raise NotImplementedError
            op.compile()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "mudnn.grouped_gemm_glu_hadamard_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GroupedGemmGluHadamardSm100Objects[key] = op
    op.execute(**kwargs)
    return _tuple_dict(
        c_tensor=kwargs.get("c_tensor"),
        d_tensor=kwargs.get("d_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
        post_rht_amax_tensor=kwargs.get("post_rht_amax_tensor"),
    )
