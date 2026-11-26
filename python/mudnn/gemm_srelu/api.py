import os

try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_GemmSreluSm100Objects = {}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("mudnn.GemmSreluSm100 requires torch") from exc
    return torch


def _is_supported_tensor(tensor) -> bool:
    if tensor is None or not hasattr(tensor, "dtype"):
        return False
    torch = _torch()
    supported = {torch.float32, torch.float16, torch.bfloat16}
    return tensor.dtype in supported


def _as_float(tensor):
    return tensor.to(dtype=_torch().float32)


def _scale_tensor(value, reference):
    if value is None:
        return 1.0
    scaled = _as_float(value)
    if scaled.shape == reference.shape:
        return scaled
    try:
        return scaled.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError("GemmSrelu scale tensor is not broadcastable") from exc


def _output_shape(a_tensor, b_tensor):
    if len(a_tensor.shape) != 3 or len(b_tensor.shape) != 3:
        raise ValueError("GemmSrelu expects rank-3 A and B tensors")
    m, k, l = tuple(int(dim) for dim in a_tensor.shape)
    n, bk, bl = tuple(int(dim) for dim in b_tensor.shape)
    if k != bk or l != bl:
        raise ValueError("GemmSrelu A/B dimensions are incompatible")
    return m, n, l


def _empty_output(shape, dtype, sample, c_major="n"):
    torch = _torch()
    out_dtype = dtype or sample.dtype
    device = getattr(sample, "device", None)
    m, n, l = tuple(int(dim) for dim in shape)
    if c_major == "m":
        return torch.empty_strided(shape, (1, m, m * n), dtype=out_dtype, device=device)
    if c_major == "n":
        return torch.empty_strided(shape, (n, 1, m * n), dtype=out_dtype, device=device)
    raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")


def _compute_srelu(**kwargs):
    torch = _torch()
    a = kwargs.get("a_tensor")
    b = kwargs.get("b_tensor")
    if not _is_supported_tensor(a) or not _is_supported_tensor(b):
        raise NotImplementedError(
            "GemmSrelu torch fallback supports FLOAT, HALF, and BFLOAT16 tensors"
        )
    m, n, l = _output_shape(a, b)
    a_f32 = _as_float(a)
    b_f32 = _as_float(b)
    a_scaled = a_f32 * _scale_tensor(kwargs.get("sfa_tensor"), a_f32)
    b_scaled = b_f32 * _scale_tensor(kwargs.get("sfb_tensor"), b_f32)
    c_f32 = float(kwargs.get("alpha", 1.0)) * torch.einsum(
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
                "GemmSrelu probability tensor is not broadcastable"
            ) from exc
    d_f32 = torch.relu(c_f32) ** 2 * prob_f32
    return c_f32, d_f32


class GemmSreluSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if self.kwargs.get("sample_a") is None or self.kwargs.get("sample_b") is None:
            raise NotImplementedError(
                "mudnn.GemmSreluSm100 requires sample_a and sample_b tensors"
            )
        return _is_supported_tensor(
            self.kwargs.get("sample_a")
        ) and _is_supported_tensor(self.kwargs.get("sample_b"))

    def compile(self):
        if not self.check_support():
            raise NotImplementedError(
                "mudnn.GemmSreluSm100 torch fallback supports only FLOAT, HALF, "
                "and BFLOAT16 tensors"
            )
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        torch = _torch()
        c_f32, d_f32 = _compute_srelu(**kwargs)
        c_tensor = kwargs.get("c_tensor")
        d_tensor = kwargs.get("d_tensor")
        if c_tensor is None or d_tensor is None:
            raise ValueError("GemmSrelu execute requires c_tensor and d_tensor outputs")
        c_tensor.copy_(c_f32.to(dtype=c_tensor.dtype))
        d_tensor.copy_(d_f32.to(dtype=d_tensor.dtype))
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
        kwargs.get("c_major"),
        kwargs.get("c_dtype"),
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
        config,
    )


def gemm_srelu_wrapper_sm100(
    a_tensor=None,
    b_tensor=None,
    sfa_tensor=None,
    sfb_tensor=None,
    prob_tensor=None,
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
    if prob_tensor is not None:
        kwargs["prob_tensor"] = prob_tensor
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
        if kwargs.get("d_tensor") is None:
            kwargs["d_tensor"] = _empty_output(
                shape,
                kwargs.get("d_dtype"),
                a_tensor,
                kwargs.get("c_major", "n"),
            )
        if kwargs.get("amax_tensor") is None and _is_supported_tensor(a_tensor):
            torch = _torch()
            kwargs["amax_tensor"] = torch.empty(
                (1,), dtype=torch.float32, device=getattr(a_tensor, "device", None)
            )
    key = _cache_key(**kwargs)
    op = _cache_of_GemmSreluSm100Objects.get(key)
    if op is None:
        op = GemmSreluSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b=kwargs.get("b_tensor"),
            sample_c=kwargs.get("c_tensor"),
            sample_d=kwargs.get("d_tensor"),
            sample_sfa=kwargs.get("sfa_tensor"),
            sample_sfb=kwargs.get("sfb_tensor"),
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
                "mudnn.gemm_srelu_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GemmSreluSm100Objects[key] = op
    op.execute(**kwargs)
    return TupleDict(
        c_tensor=kwargs.get("c_tensor"),
        d_tensor=kwargs.get("d_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
        sfd_tensor=kwargs.get("sfd_tensor"),
        norm_const_tensor=kwargs.get("norm_const_tensor"),
    )
