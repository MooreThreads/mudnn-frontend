import os

try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_GroupedGemmDswigluSm100Objects = {}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError(
            "mudnn.GroupedGemmDswigluSm100 requires torch"
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
            f"GroupedGemmDswiglu dtype {dtype!r} is not supported"
        ) from exc


def _scale_tensor(value, reference, name):
    if value is None:
        return 1.0
    if not _is_supported_tensor(value):
        raise NotImplementedError(
            "GroupedGemmDswiglu torch fallback supports only ordinary FLOAT, HALF, "
            "and BFLOAT16 scale tensors"
        )
    scaled = _as_float(value)
    if scaled.shape == reference.shape:
        return scaled
    try:
        return scaled.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError(
            f"GroupedGemmDswiglu {name} tensor is not broadcastable"
        ) from exc


def _grouped_shapes(a_tensor, b_tensor, c_tensor, padded_offsets):
    if b_tensor is None:
        raise NotImplementedError(
            "GroupedGemmDswiglu torch fallback requires dense B tensor"
        )
    if len(a_tensor.shape) != 3 or len(b_tensor.shape) != 3:
        raise ValueError("GroupedGemmDswiglu expects rank-3 A and B tensors")
    m, k, l_a = tuple(int(dim) for dim in a_tensor.shape)
    n, bk, l = tuple(int(dim) for dim in b_tensor.shape)
    if l_a != 1:
        raise ValueError("GroupedGemmDswiglu expects A shape (M, K, 1)")
    if k != bk:
        raise ValueError("GroupedGemmDswiglu A/B K dimensions are incompatible")
    if n % 32 != 0:
        raise ValueError("GroupedGemmDswiglu B dimension N must be divisible by 32")
    if c_tensor is None or not hasattr(c_tensor, "shape"):
        raise ValueError("GroupedGemmDswiglu requires C tensor")
    if tuple(int(dim) for dim in c_tensor.shape) != (m, n * 2, 1):
        raise ValueError("GroupedGemmDswiglu C dimensions are incompatible")
    if padded_offsets is None or not hasattr(padded_offsets, "shape"):
        raise ValueError("GroupedGemmDswiglu requires padded_offsets")
    if int(padded_offsets.numel()) != l:
        raise ValueError(
            "GroupedGemmDswiglu padded_offsets length must match expert count"
        )
    offsets = [
        int(value) for value in padded_offsets.detach().cpu().reshape(-1).tolist()
    ]
    if offsets != sorted(offsets):
        raise ValueError("GroupedGemmDswiglu padded_offsets must be sorted")
    if offsets and offsets[-1] > m:
        raise ValueError("GroupedGemmDswiglu padded_offsets exceed A dimension")
    return m, n, k, l, offsets


def _empty_output(shape, dtype, sample):
    torch = _torch()
    return torch.empty(
        shape,
        dtype=_normalize_dtype(dtype, sample),
        device=getattr(sample, "device", None),
    )


def _compute_grouped_dswiglu(**kwargs):
    torch = _torch()
    a = kwargs.get("a_tensor")
    b = kwargs.get("b_tensor")
    c = kwargs.get("c_tensor")
    if (
        not _is_supported_tensor(a)
        or not _is_supported_tensor(b)
        or not _is_supported_tensor(c)
    ):
        raise NotImplementedError(
            "GroupedGemmDswiglu torch fallback supports FLOAT, HALF, and BFLOAT16 tensors"
        )
    m, n, _k, l, offsets = _grouped_shapes(a, b, c, kwargs.get("padded_offsets"))
    a_f32 = _as_float(a)
    b_f32 = _as_float(b)
    c_f32 = _as_float(c)
    a_scaled = a_f32 * _scale_tensor(kwargs.get("sfa_tensor"), a_f32, "SFA")
    b_scaled = b_f32 * _scale_tensor(kwargs.get("sfb_tensor"), b_f32, "SFB")
    ref = torch.empty((m, n, 1), dtype=torch.float32, device=a.device)
    start = 0
    alpha = kwargs.get("alpha_tensor")
    for expert, end in enumerate(offsets):
        if end < start:
            raise ValueError("GroupedGemmDswiglu padded_offsets must be monotonic")
        if end > start:
            ref[start:end, :, 0] = torch.einsum(
                "mk,nk->mn", a_scaled[start:end, :, 0], b_scaled[:, :, expert]
            )
            if alpha is not None:
                alpha_value = float(alpha.reshape(-1)[expert])
                ref[start:end, :, 0] *= alpha_value * alpha_value
        start = end
    if start < m:
        ref[start:, :, 0] = 0.0

    c_full = c_f32.clone()
    beta = kwargs.get("beta_tensor")
    if beta is not None:
        start = 0
        for expert, end in enumerate(offsets):
            if end > start:
                c_full[start:end, :, 0] *= float(beta.reshape(-1)[expert])
            start = end

    group = 32
    cols_2n = torch.arange(n * 2, dtype=torch.long, device=a.device)
    block_cols_2n = cols_2n.view((n * 2) // group, group)
    dest_idx_glu = block_cols_2n[0::2].reshape(-1)
    dest_idx_ab = block_cols_2n[1::2].reshape(-1)
    cols_n = torch.arange(n, dtype=torch.long, device=a.device)
    src_idx_n = cols_n.view(n // group, group).reshape(-1)

    c_input = c_full.index_select(dim=1, index=dest_idx_ab)
    c_gate = c_full.index_select(dim=1, index=dest_idx_glu)
    sig = torch.sigmoid(c_gate)
    swish = c_gate * sig
    prob = kwargs.get("prob_tensor")
    if prob is None:
        prob_f32 = 1.0
    else:
        prob_f32 = _as_float(prob)
        try:
            prob_f32 = prob_f32.expand(m, n, 1)
        except RuntimeError as exc:
            raise ValueError(
                "GroupedGemmDswiglu probability tensor is not broadcastable"
            ) from exc

    ab = ref * prob_f32 * swish
    dswiglu = ref * prob_f32 * c_input * sig * (1 + c_gate * (1 - sig))
    d_f32 = torch.empty_like(c_full)
    d_f32.index_copy_(dim=1, index=dest_idx_ab, source=ab.index_select(1, src_idx_n))
    d_f32.index_copy_(
        dim=1, index=dest_idx_glu, source=dswiglu.index_select(1, src_idx_n)
    )
    dprob_f32 = torch.sum(swish * c_input * ref, dim=1, keepdim=True)

    amax = torch.empty((l, 2, 1), dtype=torch.float32, device=a.device)
    dbias = torch.empty((l, n * 2, 1), dtype=torch.float32, device=a.device)
    start = 0
    for expert, end in enumerate(offsets):
        if end > start:
            amax[expert, 0, 0] = torch.amax(torch.abs(dswiglu[start:end, :, 0]))
            amax[expert, 1, 0] = torch.amax(torch.abs(ab[start:end, :, 0]))
            dbias[expert, :, 0] = d_f32[start:end, :, 0].sum(dim=0)
        else:
            amax[expert, :, 0] = 0.0
            dbias[expert, :, 0] = 0.0
        start = end
    return d_f32, dprob_f32, dbias, amax


class GroupedGemmDswigluSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if (
            self.kwargs.get("sample_a") is None
            or self.kwargs.get("sample_b") is None
            or self.kwargs.get("sample_c") is None
        ):
            raise NotImplementedError(
                "mudnn.GroupedGemmDswigluSm100 requires sample_a, sample_b, and "
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
                "mudnn.GroupedGemmDswigluSm100 torch fallback supports only FLOAT, "
                "HALF, and BFLOAT16 tensors"
            )
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        d_f32, dprob_f32, dbias_f32, amax_f32 = _compute_grouped_dswiglu(**kwargs)
        d_row_tensor = kwargs.get("d_row_tensor")
        if d_row_tensor is None:
            d_row_tensor = kwargs.get("d_tensor")
        if d_row_tensor is None:
            raise ValueError("GroupedGemmDswiglu execute requires d_row_tensor")
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
        _dtype_of(kwargs.get("sfa_tensor")),
        _dtype_of(kwargs.get("sfb_tensor")),
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
        kwargs.get("epilogue_op"),
    )


def _tuple_dict(**kwargs):
    return TupleDict(**kwargs)


def grouped_gemm_dswiglu_wrapper_sm100(**kwargs):
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
    cached = _cache_of_GroupedGemmDswigluSm100Objects.get(key)
    if cached is None:
        op = GroupedGemmDswigluSm100(
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
            epilogue_op=kwargs.get("epilogue_op"),
        )
        try:
            is_supported = op.check_support()
            if not is_supported:
                raise NotImplementedError
            op.compile()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "mudnn.grouped_gemm_dswiglu_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GroupedGemmDswigluSm100Objects[key] = (
            op,
            kwargs.get("amax_tensor"),
            kwargs.get("beta_tensor"),
        )
    else:
        op = cached[0]
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
