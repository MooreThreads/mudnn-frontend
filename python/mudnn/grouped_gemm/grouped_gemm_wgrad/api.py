_cache_of_GroupedGemmWgradSm100Objects = {}


try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("mudnn.GroupedGemmWgradSm100 requires torch") from exc
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
            f"GroupedGemmWgrad dtype {dtype!r} is not supported"
        ) from exc


def _scale_tensor(value, reference, name):
    if value is None:
        return 1.0
    if not _is_supported_tensor(value):
        raise NotImplementedError(
            "GroupedGemmWgrad torch fallback supports only ordinary FLOAT, HALF, "
            "and BFLOAT16 scale tensors"
        )
    scaled = _as_float(value)
    if scaled.shape == reference.shape:
        return scaled
    try:
        return scaled.expand_as(reference)
    except RuntimeError as exc:
        raise ValueError(
            f"GroupedGemmWgrad {name} tensor is not broadcastable"
        ) from exc


def _wgrad_shapes(a_tensor, b_tensor, offsets_tensor):
    if len(a_tensor.shape) != 2 or len(b_tensor.shape) != 2:
        raise ValueError("GroupedGemmWgrad expects rank-2 A and B tensors")
    m, tokens = tuple(int(dim) for dim in a_tensor.shape)
    b_tokens, n = tuple(int(dim) for dim in b_tensor.shape)
    if tokens != b_tokens:
        raise ValueError("GroupedGemmWgrad A/B token dimensions are incompatible")
    if offsets_tensor is None or not hasattr(offsets_tensor, "shape"):
        raise ValueError("GroupedGemmWgrad requires offsets_tensor")
    offsets = [
        int(value) for value in offsets_tensor.detach().cpu().reshape(-1).tolist()
    ]
    if offsets != sorted(offsets):
        raise ValueError("GroupedGemmWgrad offsets_tensor must be sorted")
    if offsets and offsets[-1] > tokens:
        raise ValueError("GroupedGemmWgrad offsets_tensor exceeds token dimension")
    return m, n, len(offsets), offsets


def _empty_output(shape, dtype, sample):
    torch = _torch()
    return torch.empty(
        shape,
        dtype=_normalize_dtype(dtype, sample),
        device=getattr(sample, "device", None),
    )


def _compute_grouped_wgrad(**kwargs):
    torch = _torch()
    a = kwargs.get("a_tensor")
    b = kwargs.get("b_tensor")
    if not _is_supported_tensor(a) or not _is_supported_tensor(b):
        raise NotImplementedError(
            "GroupedGemmWgrad torch fallback supports FLOAT, HALF, and BFLOAT16 tensors"
        )
    output_mode = kwargs.get("output_mode", "dense")
    input_order = kwargs.get("input_order", "tensor2d")
    if output_mode != "dense" or input_order != "tensor2d":
        raise NotImplementedError(
            "GroupedGemmWgrad torch fallback supports only dense tensor2d output"
        )
    m, n, experts, offsets = _wgrad_shapes(a, b, kwargs.get("offsets_tensor"))
    a_f32 = _as_float(a)
    b_f32 = _as_float(b)
    a_scaled = a_f32 * _scale_tensor(kwargs.get("sfa_tensor"), a_f32, "SFA")
    b_scaled = b_f32 * _scale_tensor(kwargs.get("sfb_tensor"), b_f32, "SFB")
    global_scale_a = kwargs.get("global_scale_a")
    global_scale_b = kwargs.get("global_scale_b")
    wgrad_f32 = torch.empty((experts, m, n), dtype=torch.float32, device=a.device)
    start = 0
    for expert, end in enumerate(offsets):
        if end > start:
            expert_wgrad = torch.matmul(a_scaled[:, start:end], b_scaled[start:end, :])
            if global_scale_a is not None:
                expert_wgrad *= float(global_scale_a.reshape(-1)[expert])
            if global_scale_b is not None:
                expert_wgrad *= float(global_scale_b.reshape(-1)[expert])
            wgrad_f32[expert, :, :] = expert_wgrad
        else:
            wgrad_f32[expert, :, :] = 0.0
        start = end
    return wgrad_f32


class GroupedGemmWgradSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if self.kwargs.get("output_mode", "dense") != "dense":
            raise NotImplementedError(
                "mudnn.GroupedGemmWgradSm100 torch fallback supports only dense output"
            )
        if self.kwargs.get("input_order", "tensor2d") != "tensor2d":
            raise NotImplementedError(
                "mudnn.GroupedGemmWgradSm100 torch fallback supports only tensor2d input"
            )
        if not _is_supported_tensor(
            self.kwargs.get("sample_a")
        ) or not _is_supported_tensor(self.kwargs.get("sample_b")):
            raise NotImplementedError(
                "mudnn.GroupedGemmWgradSm100 torch fallback supports only FLOAT, "
                "HALF, and BFLOAT16 tensors"
            )
        return True

    def compile(self):
        if not self.check_support():
            raise NotImplementedError(
                "mudnn.GroupedGemmWgradSm100 torch fallback supports only FLOAT, "
                "HALF, and BFLOAT16 tensors"
            )
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        wgrad_f32 = _compute_grouped_wgrad(
            **{**self.kwargs, **kwargs},
        )
        wgrad_tensor = kwargs.get("wgrad_tensor")
        if wgrad_tensor is None:
            raise ValueError("GroupedGemmWgrad execute requires wgrad_tensor")
        wgrad_tensor.copy_(wgrad_f32.to(dtype=wgrad_tensor.dtype))
        return self


def _cache_key(**kwargs):
    return (
        kwargs.get("output_mode", "dense"),
        kwargs.get("input_order", "tensor2d"),
        kwargs.get("acc_dtype"),
        kwargs.get("wgrad_dtype"),
        tuple(kwargs.get("mma_tiler_mn") or ()),
        tuple(kwargs.get("cluster_shape_mn") or ()),
        kwargs.get("sf_vec_size"),
    )


def grouped_gemm_wgrad_wrapper_sm100(**kwargs):
    a_tensor = kwargs.get("a_tensor")
    b_tensor = kwargs.get("b_tensor")
    if (
        kwargs.get("wgrad_tensor") is None
        and kwargs.get("output_mode", "dense") == "dense"
        and kwargs.get("input_order", "tensor2d") == "tensor2d"
        and _is_supported_tensor(a_tensor)
        and _is_supported_tensor(b_tensor)
    ):
        m, n, experts, _offsets = _wgrad_shapes(
            a_tensor, b_tensor, kwargs.get("offsets_tensor")
        )
        kwargs["wgrad_tensor"] = _empty_output(
            (experts, m, n), kwargs.get("wgrad_dtype"), a_tensor
        )

    key = _cache_key(**kwargs)
    op = _cache_of_GroupedGemmWgradSm100Objects.get(key)
    if op is None:
        op = GroupedGemmWgradSm100(
            sample_a=kwargs.get("a_tensor"),
            sample_b=kwargs.get("b_tensor"),
            sample_sfa=kwargs.get("sfa_tensor"),
            sample_sfb=kwargs.get("sfb_tensor"),
            sample_offsets=kwargs.get("offsets_tensor"),
            sample_global_scale_a=kwargs.get("global_scale_a"),
            sample_global_scale_b=kwargs.get("global_scale_b"),
            output_mode=kwargs.get("output_mode", "dense"),
            input_order=kwargs.get("input_order", "tensor2d"),
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
                "mudnn.grouped_gemm_wgrad_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_GroupedGemmWgradSm100Objects[key] = op
    op.execute(**kwargs)
    return TupleDict(wgrad_tensor=kwargs.get("wgrad_tensor"))
