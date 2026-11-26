try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_GemmAmaxSm100Objects = {}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("mudnn.GemmAmaxSm100 requires torch") from exc
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
        raise NotImplementedError(f"GemmAmax dtype {dtype!r} is not supported") from exc


def _scale_tensor(value, reference):
    if value is None:
        return 1.0
    if not _is_supported_tensor(value):
        raise NotImplementedError(
            "GemmAmax torch fallback supports only ordinary FLOAT, HALF, "
            "and BFLOAT16 scale tensors"
        )
    scaled = _as_float(value)
    if scaled.shape == reference.shape:
        return scaled
    try:
        return scaled.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError("GemmAmax scale tensor is not broadcastable") from exc


def _output_shape(a_tensor, b_tensor):
    if len(a_tensor.shape) != 3 or len(b_tensor.shape) != 3:
        raise ValueError("GemmAmax expects rank-3 A and B tensors")
    m, k, l = tuple(int(dim) for dim in a_tensor.shape)
    n, bk, bl = tuple(int(dim) for dim in b_tensor.shape)
    if k != bk or l != bl:
        raise ValueError("GemmAmax A/B dimensions are incompatible")
    return m, n, l


def _empty_output(shape, dtype, sample, c_major="n"):
    torch = _torch()
    out_dtype = _normalize_dtype(dtype, sample)
    device = getattr(sample, "device", None)
    m, n, l = tuple(int(dim) for dim in shape)
    if c_major == "m":
        return torch.empty_strided(shape, (1, m, m * n), dtype=out_dtype, device=device)
    if c_major == "n":
        return torch.empty_strided(shape, (n, 1, m * n), dtype=out_dtype, device=device)
    raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")


def _compute_gemm_amax(**kwargs):
    a = kwargs.get("a_tensor")
    b = kwargs.get("b_tensor")
    if not _is_supported_tensor(a) or not _is_supported_tensor(b):
        raise NotImplementedError(
            "GemmAmax torch fallback supports FLOAT, HALF, and BFLOAT16 tensors"
        )
    shape = _output_shape(a, b)
    a_f32 = _as_float(a)
    b_f32 = _as_float(b)
    a_scaled = a_f32 * _scale_tensor(kwargs.get("sfa_tensor"), a_f32)
    b_scaled = b_f32 * _scale_tensor(kwargs.get("sfb_tensor"), b_f32)
    c_f32 = _torch().einsum("mkl,nkl->mnl", a_scaled, b_scaled)
    if tuple(int(dim) for dim in c_f32.shape) != shape:
        raise ValueError("GemmAmax computed C dimensions are incompatible")
    return c_f32


class GemmAmaxSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if self.kwargs.get("sample_a") is None or self.kwargs.get("sample_b") is None:
            raise NotImplementedError(
                "mudnn.GemmAmaxSm100 requires sample_a and sample_b tensors"
            )
        return _is_supported_tensor(
            self.kwargs.get("sample_a")
        ) and _is_supported_tensor(self.kwargs.get("sample_b"))

    def compile(self):
        if not self.check_support():
            raise NotImplementedError(
                "mudnn.GemmAmaxSm100 torch fallback supports only FLOAT, HALF, "
                "and BFLOAT16 tensors"
            )
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        torch = _torch()
        c_f32 = _compute_gemm_amax(**kwargs)
        c_tensor = kwargs.get("c_tensor")
        amax_tensor = kwargs.get("amax_tensor")
        if c_tensor is None or amax_tensor is None:
            raise ValueError("GemmAmax execute requires c_tensor and amax_tensor")
        if tuple(int(dim) for dim in c_tensor.shape) != tuple(c_f32.shape):
            raise ValueError("GemmAmax C output dimensions are incompatible")
        c_tensor.copy_(c_f32.to(dtype=c_tensor.dtype))
        amax_tensor.copy_(torch.amax(torch.abs(c_f32)).reshape_as(amax_tensor))
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


def _cache_key(**kwargs):
    return (
        _shape_of(kwargs.get("a_tensor")),
        _shape_of(kwargs.get("b_tensor")),
        _dtype_of(kwargs.get("a_tensor")),
        _dtype_of(kwargs.get("b_tensor")),
        _stride_order_of(kwargs.get("a_tensor")),
        _stride_order_of(kwargs.get("b_tensor")),
        _dtype_of(kwargs.get("sfa_tensor")),
        _dtype_of(kwargs.get("sfb_tensor")),
        kwargs.get("c_major", "n"),
        kwargs.get("c_dtype"),
        kwargs.get("acc_dtype"),
        tuple(kwargs.get("mma_tiler_mn") or ()),
        tuple(kwargs.get("cluster_shape_mn") or ()),
        kwargs.get("sf_vec_size"),
    )


def gemm_amax_wrapper_sm100(
    a_tensor=None,
    b_tensor=None,
    sfa_tensor=None,
    sfb_tensor=None,
    **kwargs,
):
    if a_tensor is not None:
        kwargs["a_tensor"] = a_tensor
    if b_tensor is not None:
        kwargs["b_tensor"] = b_tensor
    if sfa_tensor is not None:
        kwargs["sfa_tensor"] = sfa_tensor
    if sfb_tensor is not None:
        kwargs["sfb_tensor"] = sfb_tensor
    a_tensor = kwargs.get("a_tensor")
    b_tensor = kwargs.get("b_tensor")
    if _is_supported_tensor(a_tensor) and _is_supported_tensor(b_tensor):
        shape = _output_shape(a_tensor, b_tensor)
        if kwargs.get("c_tensor") is None:
            kwargs["c_tensor"] = _empty_output(
                shape,
                kwargs.get("c_dtype"),
                a_tensor,
                kwargs.get("c_major", "n"),
            )
        if kwargs.get("amax_tensor") is None:
            torch = _torch()
            kwargs["amax_tensor"] = torch.empty(
                (1, 1, 1), dtype=torch.float32, device=getattr(a_tensor, "device", None)
            )
    key = _cache_key(**kwargs)
    cached = _cache_of_GemmAmaxSm100Objects.get(key)
    if cached is None:
        op = GemmAmaxSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b=kwargs.get("b_tensor"),
            sample_sfa=kwargs.get("sfa_tensor"),
            sample_sfb=kwargs.get("sfb_tensor"),
            sample_c=kwargs.get("c_tensor"),
            sample_amax=kwargs.get("amax_tensor"),
            acc_dtype=kwargs.get("acc_dtype"),
            mma_tiler_mn=kwargs.get("mma_tiler_mn"),
            cluster_shape_mn=kwargs.get("cluster_shape_mn"),
            sf_vec_size=kwargs.get("sf_vec_size"),
        )
        try:
            is_supported = op.check_support()
            if not is_supported:
                raise NotImplementedError
            op.compile()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "mudnn.gemm_amax_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GemmAmaxSm100Objects[key] = op
    else:
        op = cached
    op.execute(**kwargs)
    return TupleDict(
        c_tensor=kwargs.get("c_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
    )


def gemm_amax(*args, **kwargs):
    return gemm_amax_wrapper_sm100(*args, **kwargs)
