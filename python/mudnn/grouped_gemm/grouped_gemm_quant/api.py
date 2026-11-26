import os

try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_GroupedGemmQuantSm100Objects = {}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("mudnn.GroupedGemmQuantSm100 requires torch") from exc
    return torch


def _is_supported_tensor(tensor) -> bool:
    if tensor is None or not hasattr(tensor, "dtype"):
        return False
    torch = _torch()
    return tensor.dtype in {torch.float32, torch.float16, torch.bfloat16}


def _as_float(tensor):
    return tensor.to(dtype=_torch().float32)


def _normalize_dtype(dtype, sample):
    if dtype is None:
        return sample.dtype
    if not isinstance(dtype, str):
        return dtype
    torch = _torch()
    dtype_map = {
        "float": torch.float32,
        "float32": torch.float32,
        "half": torch.float16,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtype_map[dtype.lower()]
    except KeyError as exc:
        raise NotImplementedError(
            f"GroupedGemmQuant dtype {dtype!r} is not supported"
        ) from exc


def _scale_tensor(value, reference, name):
    if value is None:
        return 1.0
    if not _is_supported_tensor(value):
        raise NotImplementedError(
            "GroupedGemmQuant torch fallback supports only ordinary FLOAT, HALF, "
            "and BFLOAT16 scale tensors"
        )
    scaled = _as_float(value)
    if scaled.shape == reference.shape:
        return scaled
    try:
        return scaled.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError(
            f"GroupedGemmQuant {name} tensor is not broadcastable"
        ) from exc


def _grouped_shapes(a_tensor, b_tensor, padded_offsets):
    if b_tensor is None:
        raise NotImplementedError(
            "GroupedGemmQuant torch fallback requires dense B tensor"
        )
    if len(a_tensor.shape) != 3 or len(b_tensor.shape) != 3:
        raise ValueError("GroupedGemmQuant expects rank-3 A and B tensors")
    m, k, l_a = tuple(int(dim) for dim in a_tensor.shape)
    n, bk, l = tuple(int(dim) for dim in b_tensor.shape)
    if l_a != 1:
        raise ValueError("GroupedGemmQuant expects A shape (M, K, 1)")
    if k != bk:
        raise ValueError("GroupedGemmQuant A/B K dimensions are incompatible")
    if padded_offsets is None or not hasattr(padded_offsets, "shape"):
        raise ValueError("GroupedGemmQuant requires padded_offsets")
    if int(padded_offsets.numel()) != l:
        raise ValueError(
            "GroupedGemmQuant padded_offsets length must match expert count"
        )
    offsets = [
        int(value) for value in padded_offsets.detach().cpu().reshape(-1).tolist()
    ]
    if offsets != sorted(offsets):
        raise ValueError("GroupedGemmQuant padded_offsets must be sorted")
    if offsets and offsets[-1] > m:
        raise ValueError("GroupedGemmQuant padded_offsets exceed A dimension")
    return m, n, k, l, offsets


def _empty_output(shape, dtype, sample):
    torch = _torch()
    return torch.empty(
        shape,
        dtype=_normalize_dtype(dtype, sample),
        device=getattr(sample, "device", None),
    )


def _is_fp8_dtype(dtype):
    return "float8" in str(dtype).lower()


def _compute_grouped_quant(**kwargs):
    torch = _torch()
    a = kwargs.get("a_tensor")
    b = kwargs.get("b_tensor")
    if not _is_supported_tensor(a) or not _is_supported_tensor(b):
        raise NotImplementedError(
            "GroupedGemmQuant torch fallback supports FLOAT, HALF, and BFLOAT16 tensors"
        )
    if kwargs.get("prob_tensor") is None:
        raise ValueError("GroupedGemmQuant prob_tensor is required")
    if (
        kwargs.get("sfd_row_tensor") is not None
        or kwargs.get("sfd_col_tensor") is not None
    ):
        raise NotImplementedError(
            "GroupedGemmQuant torch fallback does not support SFD quantized outputs"
        )

    m, n, _k, l, offsets = _grouped_shapes(a, b, kwargs.get("padded_offsets"))
    a_f32 = _as_float(a)
    b_f32 = _as_float(b)
    a_scaled = a_f32 * _scale_tensor(kwargs.get("sfa_tensor"), a_f32, "SFA")
    b_scaled = b_f32 * _scale_tensor(kwargs.get("sfb_tensor"), b_f32, "SFB")
    d_f32 = torch.empty((m, n, 1), dtype=torch.float32, device=a.device)
    alpha = kwargs.get("alpha_tensor")
    prob = _as_float(kwargs.get("prob_tensor"))
    row_scale = kwargs.get("row_scale_tensor")
    if row_scale is not None and not _is_supported_tensor(row_scale):
        raise NotImplementedError(
            "GroupedGemmQuant torch fallback supports only ordinary row_scale tensors"
        )
    bias = kwargs.get("bias_tensor")
    if bias is not None and not _is_supported_tensor(bias):
        raise NotImplementedError(
            "GroupedGemmQuant torch fallback supports only ordinary bias tensors"
        )
    bias_f32 = _as_float(bias) if bias is not None else None
    start = 0
    for expert, end in enumerate(offsets):
        if end < start:
            raise ValueError("GroupedGemmQuant padded_offsets must be monotonic")
        if end > start:
            expert_d = torch.einsum(
                "mk,nk->mn", a_scaled[start:end, :, 0], b_scaled[:, :, expert]
            )
            if alpha is not None:
                expert_d *= float(alpha.reshape(-1)[expert])
            if row_scale is not None:
                expert_d *= _as_float(row_scale)[start:end].reshape(-1, 1)
            if bias_f32 is None:
                expert_d *= prob[start:end].expand(end - start, n, 1)[:, :, 0]
            else:
                expert_d += prob[start:end, 0, 0].reshape(-1, 1) * bias_f32[
                    :, expert
                ].reshape(1, -1)
            d_f32[start:end, :, 0] = expert_d
        start = end
    if start < m:
        d_f32[start:, :, 0] = 0.0

    amax = torch.empty((l, 1), dtype=torch.float32, device=a.device)
    start = 0
    for expert, end in enumerate(offsets):
        if end > start:
            amax[expert, 0] = torch.amax(torch.abs(d_f32[start:end, :, 0]))
        else:
            amax[expert, 0] = 0.0
        start = end
    return d_f32, amax


class GroupedGemmQuantSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if self.kwargs.get("sample_a") is None or self.kwargs.get("sample_b") is None:
            raise NotImplementedError(
                "mudnn.GroupedGemmQuantSm100 requires sample_a and sample_b tensors"
            )
        if not _is_supported_tensor(
            self.kwargs.get("sample_a")
        ) or not _is_supported_tensor(self.kwargs.get("sample_b")):
            raise NotImplementedError(
                "mudnn.GroupedGemmQuantSm100 torch fallback supports only FLOAT, "
                "HALF, and BFLOAT16 tensors"
            )
        return True

    def compile(self):
        self.check_support()
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        d_f32, amax_f32 = _compute_grouped_quant(**{**self.kwargs, **kwargs})
        d_tensor = kwargs.get("d_tensor")
        if d_tensor is None:
            raise ValueError("GroupedGemmQuant execute requires d_tensor")
        d_tensor.copy_(d_f32.to(dtype=d_tensor.dtype))
        d_col_tensor = kwargs.get("d_col_tensor")
        if d_col_tensor is not None and tuple(d_col_tensor.shape) == tuple(d_f32.shape):
            d_col_tensor.copy_(d_f32.to(dtype=d_col_tensor.dtype))
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
            _expert_count(b_tensor, b_ptrs),
        )
    else:
        shape_key = (
            "discrete_dynamic_m",
            _shape_of(a_tensor)[1:],
            kwargs.get("n"),
            _expert_count(b_tensor, b_ptrs),
        )
    return (
        "dense" if is_dense else "discrete",
        shape_key,
        _dtype_of(a_tensor),
        _dtype_of(b_tensor) if is_dense else kwargs.get("b_dtype"),
        _stride_order_of(a_tensor),
        _stride_order_of(b_tensor) if is_dense else _stride_order_of(b_ptrs),
        _dtype_of(kwargs.get("sfa_tensor")),
        (
            _dtype_of(kwargs.get("sfb_tensor"))
            if is_dense
            else _dtype_of(kwargs.get("sfb_ptrs"))
        ),
        _dtype_of(kwargs.get("bias_tensor")),
        _dtype_of(kwargs.get("alpha_tensor")),
        _dtype_of(kwargs.get("norm_const_tensor")),
        _dtype_of(kwargs.get("prob_tensor")),
        _dtype_of(kwargs.get("row_scale_tensor")),
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
        kwargs.get("b_major", "k"),
    )


def _tuple_dict(**kwargs):
    return TupleDict(**kwargs)


def grouped_gemm_quant_wrapper_sm100(**kwargs):
    if not kwargs:
        raise NotImplementedError(
            "mudnn.grouped_gemm_quant_wrapper_sm100 is not supported by muDNN frontend yet"
        )
    if kwargs.get("prob_tensor") is None:
        raise ValueError("GroupedGemmQuant prob_tensor is required")
    if (
        _is_fp8_dtype(_dtype_of(kwargs.get("a_tensor")))
        and kwargs.get("norm_const_tensor") is None
    ):
        raise ValueError(
            "GroupedGemmQuant norm_const_tensor is required when FP8 inputs are used"
        )
    if _valid_m(kwargs.get("a_tensor")) == 0:
        return _tuple_dict(
            d_tensor=kwargs.get("d_tensor"),
            d_col_tensor=kwargs.get("d_col_tensor"),
            amax_tensor=kwargs.get("amax_tensor"),
            sfd_row_tensor=kwargs.get("sfd_row_tensor"),
            sfd_col_tensor=kwargs.get("sfd_col_tensor"),
        )

    a_tensor = kwargs.get("a_tensor")
    b_tensor = kwargs.get("b_tensor")
    if _is_supported_tensor(a_tensor) and _is_supported_tensor(b_tensor):
        m, n, _k, l, _offsets = _grouped_shapes(
            a_tensor, b_tensor, kwargs.get("padded_offsets")
        )
        if kwargs.get("d_tensor") is None:
            kwargs["d_tensor"] = _empty_output(
                (m, n, 1), kwargs.get("d_dtype"), a_tensor
            )
        if kwargs.get("amax_tensor") is None and kwargs.get("d_dtype") is not None:
            torch = _torch()
            kwargs["amax_tensor"] = torch.empty(
                (l, 1), dtype=torch.float32, device=getattr(a_tensor, "device", None)
            )

    key = _cache_key(**kwargs)
    op = _cache_of_GroupedGemmQuantSm100Objects.get(key)
    if op is None:
        op = GroupedGemmQuantSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b=kwargs.get("b_tensor"),
            sample_sfa=kwargs.get("sfa_tensor"),
            sample_sfb=kwargs.get("sfb_tensor"),
            sample_padded_offsets=kwargs.get("padded_offsets"),
            sample_alpha=kwargs.get("alpha_tensor"),
            sample_bias=kwargs.get("bias_tensor"),
            sample_norm_const=kwargs.get("norm_const_tensor"),
            sample_prob=kwargs.get("prob_tensor"),
            sample_row_scale=kwargs.get("row_scale_tensor"),
            acc_dtype=kwargs.get("acc_dtype"),
            mma_tiler_mn=kwargs.get("mma_tiler_mn"),
            cluster_shape_mn=kwargs.get("cluster_shape_mn"),
            sf_vec_size=kwargs.get("sf_vec_size"),
            vector_f32=kwargs.get("vector_f32"),
            m_aligned=kwargs.get("m_aligned"),
            discrete_col_sfd=kwargs.get("discrete_col_sfd"),
        )
        try:
            is_supported = op.check_support()
            if not is_supported:
                raise NotImplementedError
            op.compile()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "mudnn.grouped_gemm_quant_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GroupedGemmQuantSm100Objects[key] = op
    op.execute(**kwargs)
    return _tuple_dict(
        d_tensor=kwargs.get("d_tensor"),
        d_col_tensor=kwargs.get("d_col_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
        sfd_row_tensor=kwargs.get("sfd_row_tensor"),
        sfd_col_tensor=kwargs.get("sfd_col_tensor"),
    )
