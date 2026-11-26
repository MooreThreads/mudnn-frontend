try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_GemmSwigluSm100Objects = {}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("mudnn.GemmSwigluSm100 requires torch") from exc
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
            f"GemmSwiglu dtype {dtype!r} is not supported"
        ) from exc


def _scale_tensor(value, reference):
    if value is None:
        return 1.0
    if not _is_supported_tensor(value):
        raise NotImplementedError(
            "GemmSwiglu torch fallback supports only ordinary FLOAT, HALF, "
            "and BFLOAT16 scale tensors"
        )
    scaled = _as_float(value)
    if scaled.shape == reference.shape:
        return scaled
    try:
        return scaled.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError("GemmSwiglu scale tensor is not broadcastable") from exc


def _output_shapes(a_tensor, b_tensor):
    if len(a_tensor.shape) != 3 or len(b_tensor.shape) != 3:
        raise ValueError("GemmSwiglu expects rank-3 A and B tensors")
    m, k, l = tuple(int(dim) for dim in a_tensor.shape)
    n, bk, bl = tuple(int(dim) for dim in b_tensor.shape)
    if k != bk or l != bl:
        raise ValueError("GemmSwiglu A/B dimensions are incompatible")
    if n % 64 != 0:
        raise ValueError(
            "GemmSwiglu B dimension N must contain an even number of 32-column blocks"
        )
    return (m, n, l), (m, n // 2, l)


def _empty_output(shape, dtype, sample, c_major="n"):
    torch = _torch()
    out_dtype = _normalize_dtype(dtype, sample)
    device = getattr(sample, "device", None)
    m, n, _l = tuple(int(dim) for dim in shape)
    if c_major == "m":
        return torch.empty_strided(shape, (1, m, m * n), dtype=out_dtype, device=device)
    if c_major == "n":
        return torch.empty_strided(shape, (n, 1, m * n), dtype=out_dtype, device=device)
    raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")


def _compute_swiglu(**kwargs):
    torch = _torch()
    a = kwargs.get("a_tensor")
    b = kwargs.get("b_tensor")
    if not _is_supported_tensor(a) or not _is_supported_tensor(b):
        raise NotImplementedError(
            "GemmSwiglu torch fallback supports FLOAT, HALF, and BFLOAT16 tensors"
        )
    ab12_shape, _c_shape = _output_shapes(a, b)
    a_f32 = _as_float(a)
    b_f32 = _as_float(b)
    a_scaled = a_f32 * _scale_tensor(kwargs.get("sfa_tensor"), a_f32)
    b_scaled = b_f32 * _scale_tensor(kwargs.get("sfb_tensor"), b_f32)
    ab12_f32 = float(kwargs.get("alpha", 1.0)) * torch.einsum(
        "mkl,nkl->mnl", a_scaled, b_scaled
    )
    if tuple(int(dim) for dim in ab12_f32.shape) != ab12_shape:
        raise ValueError("GemmSwiglu computed AB12 dimensions are incompatible")

    group = 32
    n = ab12_shape[1]
    num_blocks = n // group
    cols = torch.arange(n, device=ab12_f32.device, dtype=torch.long)
    block_cols = cols.view(num_blocks, group)
    input_idx = block_cols[0::2].reshape(-1)
    gate_idx = block_cols[1::2].reshape(-1)
    input_f32 = ab12_f32.index_select(1, input_idx)
    gate_f32 = ab12_f32.index_select(1, gate_idx)
    c_f32 = input_f32 * (gate_f32 * torch.sigmoid(gate_f32))
    return ab12_f32, c_f32


class GemmSwigluSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if self.kwargs.get("sample_a") is None or self.kwargs.get("sample_b") is None:
            raise NotImplementedError(
                "mudnn.GemmSwigluSm100 requires sample_a and sample_b tensors"
            )
        return _is_supported_tensor(
            self.kwargs.get("sample_a")
        ) and _is_supported_tensor(self.kwargs.get("sample_b"))

    def compile(self):
        if not self.check_support():
            raise NotImplementedError(
                "mudnn.GemmSwigluSm100 torch fallback supports only FLOAT, HALF, "
                "and BFLOAT16 tensors"
            )
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        torch = _torch()
        ab12_f32, c_f32 = _compute_swiglu(**kwargs)
        ab12_tensor = kwargs.get("ab12_tensor")
        c_tensor = kwargs.get("c_tensor")
        if ab12_tensor is None or c_tensor is None:
            raise ValueError("GemmSwiglu execute requires ab12_tensor and c_tensor")
        if tuple(int(dim) for dim in ab12_tensor.shape) != tuple(ab12_f32.shape):
            raise ValueError("GemmSwiglu AB12 output dimensions are incompatible")
        if tuple(int(dim) for dim in c_tensor.shape) != tuple(c_f32.shape):
            raise ValueError("GemmSwiglu C output dimensions are incompatible")
        ab12_tensor.copy_(ab12_f32.to(dtype=ab12_tensor.dtype))
        c_tensor.copy_(c_f32.to(dtype=c_tensor.dtype))
        amax_tensor = kwargs.get("amax_tensor")
        if amax_tensor is not None:
            amax_tensor.copy_(torch.amax(torch.abs(c_f32)).reshape_as(amax_tensor))
        norm_const_tensor = kwargs.get("norm_const_tensor")
        if norm_const_tensor is not None:
            norm_const_tensor.fill_(1.0)
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


def _cache_key(**kwargs):
    return (
        _shape_of(kwargs.get("a_tensor")),
        _shape_of(kwargs.get("b_tensor")),
        _dtype_of(kwargs.get("a_tensor")),
        _dtype_of(kwargs.get("b_tensor")),
        _stride_of(kwargs.get("a_tensor")),
        _stride_of(kwargs.get("b_tensor")),
        kwargs.get("alpha", 1.0),
        kwargs.get("c_major", "n"),
        kwargs.get("ab12_dtype"),
        kwargs.get("c_dtype"),
        kwargs.get("acc_dtype"),
        tuple(kwargs.get("mma_tiler_mn") or ()),
        tuple(kwargs.get("cluster_shape_mn") or ()),
        _shape_of(kwargs.get("sfa_tensor")),
        _shape_of(kwargs.get("sfb_tensor")),
        _stride_of(kwargs.get("sfa_tensor")),
        _stride_of(kwargs.get("sfb_tensor")),
        _dtype_of(kwargs.get("sfa_tensor")),
        _dtype_of(kwargs.get("sfb_tensor")),
        _shape_of(kwargs.get("norm_const_tensor")),
        _stride_of(kwargs.get("norm_const_tensor")),
        _dtype_of(kwargs.get("norm_const_tensor")),
        kwargs.get("sf_vec_size"),
        kwargs.get("vector_f32"),
        kwargs.get("ab12_stages"),
    )


def _tuple_dict(**kwargs):
    return TupleDict(**kwargs)


def gemm_swiglu_wrapper_sm100(
    a_tensor=None,
    b_tensor=None,
    **kwargs,
):
    if a_tensor is not None:
        kwargs["a_tensor"] = a_tensor
    if b_tensor is not None:
        kwargs["b_tensor"] = b_tensor
    a_tensor = kwargs.get("a_tensor")
    b_tensor = kwargs.get("b_tensor")
    if _is_supported_tensor(a_tensor) and _is_supported_tensor(b_tensor):
        ab12_shape, c_shape = _output_shapes(a_tensor, b_tensor)
        if kwargs.get("ab12_tensor") is None:
            kwargs["ab12_tensor"] = _empty_output(
                ab12_shape,
                kwargs.get("ab12_dtype"),
                a_tensor,
                kwargs.get("c_major", "n"),
            )
        if kwargs.get("c_tensor") is None:
            kwargs["c_tensor"] = _empty_output(
                c_shape,
                kwargs.get("c_dtype"),
                a_tensor,
                kwargs.get("c_major", "n"),
            )
    key = _cache_key(**kwargs)
    op = _cache_of_GemmSwigluSm100Objects.get(key)
    if op is None:
        op = GemmSwigluSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b=kwargs.get("b_tensor"),
            sample_ab12=kwargs.get("ab12_tensor"),
            sample_c=kwargs.get("c_tensor"),
            alpha=kwargs.get("alpha", 1.0),
            acc_dtype=kwargs.get("acc_dtype"),
            mma_tiler_mn=kwargs.get("mma_tiler_mn"),
            cluster_shape_mn=kwargs.get("cluster_shape_mn"),
            sample_sfa=kwargs.get("sfa_tensor"),
            sample_sfb=kwargs.get("sfb_tensor"),
            sample_amax=kwargs.get("amax_tensor"),
            sample_sfc=kwargs.get("sfc_tensor"),
            sample_norm_const=kwargs.get("norm_const_tensor"),
            sf_vec_size=kwargs.get("sf_vec_size"),
            vector_f32=kwargs.get("vector_f32"),
            ab12_stages=kwargs.get("ab12_stages"),
        )
        try:
            is_supported = op.check_support()
            if not is_supported:
                raise NotImplementedError
            op.compile()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "mudnn.gemm_swiglu_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GemmSwigluSm100Objects[key] = op
    op.execute(**kwargs)
    outputs = _tuple_dict(
        ab12_tensor=kwargs.get("ab12_tensor"),
        c_tensor=kwargs.get("c_tensor"),
        sfc_tensor=kwargs.get("sfc_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
    )
    return outputs
