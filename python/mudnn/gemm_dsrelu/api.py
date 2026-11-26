import os

try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_GemmDsreluSm100Objects = {}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("mudnn.GemmDsreluSm100 requires torch") from exc
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
            f"GemmDsrelu dtype {dtype!r} is not supported"
        ) from exc


def _scale_tensor(value, reference):
    if value is None:
        return 1.0
    scaled = _as_float(value)
    if scaled.shape == reference.shape:
        return scaled
    try:
        return scaled.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError("GemmDsrelu scale tensor is not broadcastable") from exc


def _output_shape(a_tensor, b_tensor):
    if len(a_tensor.shape) != 3 or len(b_tensor.shape) != 3:
        raise ValueError("GemmDsrelu expects rank-3 A and B tensors")
    m, k, l = tuple(int(dim) for dim in a_tensor.shape)
    n, bk, bl = tuple(int(dim) for dim in b_tensor.shape)
    if k != bk or l != bl:
        raise ValueError("GemmDsrelu A/B dimensions are incompatible")
    return m, n, l


def _empty_output(shape, dtype, sample, d_major="n"):
    torch = _torch()
    out_dtype = _normalize_dtype(dtype, sample)
    device = getattr(sample, "device", None)
    m, n, _l = tuple(int(dim) for dim in shape)
    if d_major == "m":
        return torch.empty_strided(shape, (1, m, m * n), dtype=out_dtype, device=device)
    if d_major == "n":
        return torch.empty_strided(shape, (n, 1, m * n), dtype=out_dtype, device=device)
    raise ValueError(f"d_major must be either 'm' or 'n', got {d_major}")


def _tuple_dict(**kwargs):
    return TupleDict(**kwargs)


def _compute_dsrelu(**kwargs):
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
            "GemmDsrelu torch fallback supports FLOAT, HALF, and BFLOAT16 tensors"
        )
    m, n, l = _output_shape(a, b)
    if tuple(int(dim) for dim in c.shape) != (m, n, l):
        raise ValueError("GemmDsrelu C dimensions are incompatible")
    a_f32 = _as_float(a)
    b_f32 = _as_float(b)
    c_f32 = _as_float(c)
    a_scaled = a_f32 * _scale_tensor(kwargs.get("sfa_tensor"), a_f32)
    b_scaled = b_f32 * _scale_tensor(kwargs.get("sfb_tensor"), b_f32)
    x_f32 = float(kwargs.get("alpha", 1.0)) * torch.einsum(
        "mkl,nkl->mnl", a_scaled, b_scaled
    )
    prob = kwargs.get("prob_tensor")
    if prob is None:
        prob_f32 = 1.0
    else:
        prob_f32 = _as_float(prob)
        try:
            prob_f32 = prob_f32.expand(m, n, l)
        except RuntimeError as exc:
            raise ValueError(
                "GemmDsrelu probability tensor is not broadcastable"
            ) from exc
    relu_x = torch.relu(x_f32)
    d_f32 = c_f32 * prob_f32 * 2 * relu_x
    dprob_f32 = torch.sum(relu_x**2 * c_f32, dim=1, keepdim=True)
    return d_f32, dprob_f32


class GemmDsreluSm100:
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
                "mudnn.GemmDsreluSm100 requires sample_a, sample_b, and sample_c tensors"
            )
        return (
            _is_supported_tensor(self.kwargs.get("sample_a"))
            and _is_supported_tensor(self.kwargs.get("sample_b"))
            and _is_supported_tensor(self.kwargs.get("sample_c"))
        )

    def compile(self):
        if not self.check_support():
            raise NotImplementedError(
                "mudnn.GemmDsreluSm100 torch fallback supports only FLOAT, HALF, "
                "and BFLOAT16 tensors"
            )
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        torch = _torch()
        d_f32, dprob_f32 = _compute_dsrelu(**kwargs)
        d_tensor = kwargs.get("d_tensor")
        dprob_tensor = kwargs.get("dprob_tensor")
        if d_tensor is None or dprob_tensor is None:
            raise ValueError("GemmDsrelu execute requires d_tensor and dprob_tensor")
        d_tensor.copy_(d_f32.to(dtype=d_tensor.dtype))
        dprob_tensor.copy_(dprob_f32.to(dtype=dprob_tensor.dtype))
        amax_tensor = kwargs.get("amax_tensor")
        if amax_tensor is not None:
            amax_tensor.copy_(torch.amax(torch.abs(d_f32)).reshape_as(amax_tensor))
        norm_const_tensor = kwargs.get("norm_const_tensor")
        if norm_const_tensor is not None:
            norm_const_tensor.fill_(1.0)
        return self


def _shape_of(tensor):
    return tuple(getattr(tensor, "shape", ()))


def _cache_key(**kwargs):
    full_dynamic = os.getenv("CUDNN_FE_GEMM_DYNAMIC_MNKL") == "1"
    dynamic_m = os.getenv("CUDNN_FE_GEMM_DYNAMIC_M") == "1"
    config = (
        kwargs.get("d_major"),
        kwargs.get("d_dtype"),
        kwargs.get("acc_dtype"),
        tuple(kwargs.get("mma_tiler_mn") or ()),
        tuple(kwargs.get("cluster_shape_mn") or ()),
        kwargs.get("sf_vec_size"),
        kwargs.get("vector_f32"),
    )
    if full_dynamic:
        return ("full_dynamic", config)
    if dynamic_m:
        return ("dynamic_m", _shape_of(kwargs.get("b_tensor")), config)
    return (
        "static",
        _shape_of(kwargs.get("a_tensor")),
        _shape_of(kwargs.get("b_tensor")),
        _shape_of(kwargs.get("c_tensor")),
        config,
    )


def gemm_dsrelu_wrapper_sm100(
    a_tensor=None,
    b_tensor=None,
    c_tensor=None,
    sfa_tensor=None,
    sfb_tensor=None,
    prob_tensor=None,
    **kwargs,
):
    if a_tensor is not None:
        kwargs["a_tensor"] = a_tensor
    if b_tensor is not None:
        kwargs["b_tensor"] = b_tensor
    if c_tensor is not None:
        kwargs["c_tensor"] = c_tensor
    if sfa_tensor is not None:
        kwargs["sfa_tensor"] = sfa_tensor
    if sfb_tensor is not None:
        kwargs["sfb_tensor"] = sfb_tensor
    if prob_tensor is not None:
        kwargs["prob_tensor"] = prob_tensor
    a_tensor = kwargs.get("a_tensor")
    b_tensor = kwargs.get("b_tensor")
    c_tensor = kwargs.get("c_tensor")
    if (
        _is_supported_tensor(a_tensor)
        and _is_supported_tensor(b_tensor)
        and _is_supported_tensor(c_tensor)
    ):
        m, _n, l = _output_shape(a_tensor, b_tensor)
        if kwargs.get("d_tensor") is None:
            kwargs["d_tensor"] = _empty_output(
                c_tensor.shape,
                kwargs.get("d_dtype"),
                c_tensor,
                kwargs.get("d_major", "n"),
            )
        if kwargs.get("dprob_tensor") is None:
            torch = _torch()
            kwargs["dprob_tensor"] = torch.empty(
                (m, 1, l), dtype=torch.float32, device=getattr(c_tensor, "device", None)
            )
        if kwargs.get("amax_tensor") is None:
            torch = _torch()
            kwargs["amax_tensor"] = torch.empty(
                (1,), dtype=torch.float32, device=getattr(c_tensor, "device", None)
            )
    key = _cache_key(**kwargs)
    op = _cache_of_GemmDsreluSm100Objects.get(key)
    if op is None:
        op = GemmDsreluSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b=kwargs.get("b_tensor"),
            sample_c=kwargs.get("c_tensor"),
            sample_prob=kwargs.get("prob_tensor"),
            sample_norm_const=kwargs.get("norm_const_tensor"),
            acc_dtype=kwargs.get("acc_dtype"),
            mma_tiler_mn=kwargs.get("mma_tiler_mn"),
            cluster_shape_mn=kwargs.get("cluster_shape_mn"),
            sf_vec_size=kwargs.get("sf_vec_size"),
            vector_f32=kwargs.get("vector_f32"),
        )
        try:
            is_supported = op.check_support()
            if not is_supported:
                raise NotImplementedError
            op.compile()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "mudnn.gemm_dsrelu_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GemmDsreluSm100Objects[key] = op
    op.execute(**kwargs)
    return _tuple_dict(
        d_tensor=kwargs.get("d_tensor"),
        dprob_tensor=kwargs.get("dprob_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
        sfd_tensor=kwargs.get("sfd_tensor"),
    )
