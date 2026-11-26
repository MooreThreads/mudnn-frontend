import os

try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_GroupedGemmSwigluSm100Objects = {}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError(
            "mudnn.GroupedGemmSwigluSm100 requires torch"
        ) from exc
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
            f"GroupedGemmSwiglu dtype {dtype!r} is not supported"
        ) from exc


def _scale_tensor(value, reference, name):
    if value is None:
        return 1.0
    if not _is_supported_tensor(value):
        raise NotImplementedError(
            "GroupedGemmSwiglu torch fallback supports only ordinary FLOAT, HALF, "
            "and BFLOAT16 scale tensors"
        )
    scaled = _as_float(value)
    if scaled.shape == reference.shape:
        return scaled
    try:
        return scaled.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError(
            f"GroupedGemmSwiglu {name} tensor is not broadcastable"
        ) from exc


def _grouped_shapes(a_tensor, b_tensor, padded_offsets):
    if b_tensor is None:
        raise NotImplementedError(
            "GroupedGemmSwiglu torch fallback requires dense B tensor"
        )
    if len(a_tensor.shape) != 3 or len(b_tensor.shape) != 3:
        raise ValueError("GroupedGemmSwiglu expects rank-3 A and B tensors")
    m, k, l_a = tuple(int(dim) for dim in a_tensor.shape)
    n, bk, l = tuple(int(dim) for dim in b_tensor.shape)
    if l_a != 1:
        raise ValueError("GroupedGemmSwiglu expects A shape (M, K, 1)")
    if k != bk:
        raise ValueError("GroupedGemmSwiglu A/B K dimensions are incompatible")
    if n % 64 != 0:
        raise ValueError(
            "GroupedGemmSwiglu B dimension N must contain an even number of "
            "32-column blocks"
        )
    if padded_offsets is None or not hasattr(padded_offsets, "shape"):
        raise ValueError("GroupedGemmSwiglu requires padded_offsets")
    if int(padded_offsets.numel()) != l:
        raise ValueError(
            "GroupedGemmSwiglu padded_offsets length must match expert count"
        )
    offsets = [
        int(value) for value in padded_offsets.detach().cpu().reshape(-1).tolist()
    ]
    if offsets != sorted(offsets):
        raise ValueError("GroupedGemmSwiglu padded_offsets must be sorted")
    if offsets and offsets[-1] > m:
        raise ValueError("GroupedGemmSwiglu padded_offsets exceed A dimension")
    return m, n, k, l, offsets


def _empty_output(shape, dtype, sample):
    torch = _torch()
    return torch.empty(
        shape,
        dtype=_normalize_dtype(dtype, sample),
        device=getattr(sample, "device", None),
    )


def _compute_grouped_swiglu(**kwargs):
    torch = _torch()
    a = kwargs.get("a_tensor")
    b = kwargs.get("b_tensor")
    if not _is_supported_tensor(a) or not _is_supported_tensor(b):
        raise NotImplementedError(
            "GroupedGemmSwiglu torch fallback supports FLOAT, HALF, and BFLOAT16 tensors"
        )
    m, n, _k, l, offsets = _grouped_shapes(a, b, kwargs.get("padded_offsets"))
    a_f32 = _as_float(a)
    b_f32 = _as_float(b)
    a_scaled = a_f32 * _scale_tensor(kwargs.get("sfa_tensor"), a_f32, "SFA")
    b_scaled = b_f32 * _scale_tensor(kwargs.get("sfb_tensor"), b_f32, "SFB")
    c_f32 = torch.empty((m, n, 1), dtype=torch.float32, device=a.device)
    start = 0
    alpha = kwargs.get("alpha_tensor")
    for expert, end in enumerate(offsets):
        if end < start:
            raise ValueError("GroupedGemmSwiglu padded_offsets must be monotonic")
        if end > start:
            c_f32[start:end, :, 0] = torch.einsum(
                "mk,nk->mn", a_scaled[start:end, :, 0], b_scaled[:, :, expert]
            )
            if alpha is not None:
                c_f32[start:end, :, 0] *= float(alpha.reshape(-1)[expert])
        start = end
    if start < m:
        c_f32[start:, :, 0] = 0.0

    group = 32
    num_blocks = n // group
    cols = torch.arange(n, device=c_f32.device, dtype=torch.long)
    block_cols = cols.view(num_blocks, group)
    gate_idx = block_cols[0::2].reshape(-1)
    up_idx = block_cols[1::2].reshape(-1)
    gate_f32 = c_f32.index_select(1, gate_idx)
    up_f32 = c_f32.index_select(1, up_idx)
    d_f32 = up_f32 * (gate_f32 * torch.sigmoid(gate_f32))

    prob = kwargs.get("prob_tensor")
    if prob is not None:
        prob_f32 = _as_float(prob)
        try:
            d_f32 = d_f32 * prob_f32.expand(m, n // 2, 1)
        except RuntimeError as exc:
            raise ValueError(
                "GroupedGemmSwiglu probability tensor is not broadcastable"
            ) from exc

    amax = torch.empty((l, 1), dtype=torch.float32, device=a.device)
    start = 0
    for expert, end in enumerate(offsets):
        if end > start:
            amax[expert, 0] = torch.amax(torch.abs(d_f32[start:end, :, 0]))
        else:
            amax[expert, 0] = 0.0
        start = end
    return c_f32, d_f32, amax


class GroupedGemmSwigluSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if self.kwargs.get("sample_a") is None or self.kwargs.get("sample_b") is None:
            raise NotImplementedError(
                "mudnn.GroupedGemmSwigluSm100 requires sample_a and sample_b tensors"
            )
        return _is_supported_tensor(
            self.kwargs.get("sample_a")
        ) and _is_supported_tensor(self.kwargs.get("sample_b"))

    def compile(self):
        if not self.check_support():
            raise NotImplementedError(
                "mudnn.GroupedGemmSwigluSm100 torch fallback supports only FLOAT, "
                "HALF, and BFLOAT16 tensors"
            )
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        c_f32, d_f32, amax_f32 = _compute_grouped_swiglu(**kwargs)
        c_tensor = kwargs.get("c_tensor")
        d_tensor = kwargs.get("d_tensor")
        if c_tensor is None or d_tensor is None:
            raise ValueError("GroupedGemmSwiglu execute requires c_tensor and d_tensor")
        c_tensor.copy_(c_f32.to(dtype=c_tensor.dtype))
        d_tensor.copy_(d_f32.to(dtype=d_tensor.dtype))
        d_col_tensor = kwargs.get("d_col_tensor")
        if d_col_tensor is not None and tuple(d_col_tensor.shape) == tuple(d_f32.shape):
            d_col_tensor.copy_(d_f32.to(dtype=d_col_tensor.dtype))
        amax_tensor = kwargs.get("amax_tensor")
        if amax_tensor is not None:
            amax_tensor.copy_(amax_f32.reshape_as(amax_tensor))
        return self

    def _unsupported(self):
        raise NotImplementedError(
            "mudnn.GroupedGemmSwigluSm100 is not supported by muDNN frontend yet"
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
    b_tensor = kwargs.get("b_tensor")
    c_tensor = kwargs.get("c_tensor")
    full_dynamic = os.getenv("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1") != "0"
    return (
        full_dynamic,
        _shape_of(a_tensor)[1:] if not full_dynamic else None,
        _shape_of(b_tensor) if not full_dynamic else None,
        _shape_of(c_tensor)[1:] if not full_dynamic else None,
        _dtype_of(a_tensor),
        _dtype_of(b_tensor),
        _dtype_of(c_tensor),
        _stride_order_of(a_tensor),
        _stride_order_of(b_tensor),
        _stride_order_of(c_tensor),
        _dtype_of(kwargs.get("norm_const_tensor")),
        _shape_of(kwargs.get("padded_offsets")) if not full_dynamic else None,
        _stride_of(kwargs.get("padded_offsets")) if not full_dynamic else None,
        _dtype_of(kwargs.get("padded_offsets")),
        kwargs.get("acc_dtype"),
        kwargs.get("c_dtype"),
        kwargs.get("d_dtype"),
        kwargs.get("cd_major", "n"),
        tuple(kwargs.get("mma_tiler_mn") or ()),
        tuple(kwargs.get("cluster_shape_mn") or ()),
        kwargs.get("sf_vec_size"),
        kwargs.get("vector_f32"),
        kwargs.get("m_aligned"),
        kwargs.get("discrete_col_sfd"),
        kwargs.get("prob_tensor") is not None,
    )


def _tuple_dict(**kwargs):
    return TupleDict(**kwargs)


def grouped_gemm_swiglu_wrapper_sm100(**kwargs):
    if _valid_m(kwargs.get("a_tensor")) == 0:
        return _tuple_dict(
            c_tensor=kwargs.get("c_tensor"),
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

    key = _cache_key(**kwargs)
    cached = _cache_of_GroupedGemmSwigluSm100Objects.get(key)
    if cached is None:
        op = GroupedGemmSwigluSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b=kwargs.get("b_tensor"),
            sample_c=kwargs.get("c_tensor"),
            sample_d=kwargs.get("d_tensor"),
            sample_sfa=kwargs.get("sfa_tensor"),
            sample_sfb=kwargs.get("sfb_tensor"),
            sample_padded_offsets=kwargs.get("padded_offsets"),
            sample_alpha=kwargs.get("alpha_tensor"),
            sample_norm_const=kwargs.get("norm_const_tensor"),
            sample_prob=kwargs.get("prob_tensor"),
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
                "mudnn.grouped_gemm_swiglu_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GroupedGemmSwigluSm100Objects[key] = (op, kwargs.get("amax_tensor"))
    else:
        op = cached[0]
    op.execute(**kwargs)
    return _tuple_dict(
        c_tensor=kwargs.get("c_tensor"),
        d_tensor=kwargs.get("d_tensor"),
        d_col_tensor=kwargs.get("d_col_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
        sfd_row_tensor=kwargs.get("sfd_row_tensor"),
        sfd_col_tensor=kwargs.get("sfd_col_tensor"),
    )
