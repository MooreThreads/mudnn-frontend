try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict


_cache_of_RmsNormRhtAmaxSm100Objects = {}

DEFAULT_NUM_THREADS_BY_N = {
    2048: 128,
    4096: 256,
    7168: 128,
    8192: 512,
    16384: 1024,
    32768: 512,
}
RPC_CANDIDATES = (2, 4, 8)
TARGET_MIN_CTAS = 148


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError("mudnn.RmsNormRhtAmaxSm100 requires torch") from exc
    return torch


def _is_supported_tensor(tensor) -> bool:
    if tensor is None or not hasattr(tensor, "dtype"):
        return False
    torch = _torch()
    return tensor.dtype in {torch.float32, torch.float16, torch.bfloat16}


def best_num_threads(n: int):
    n = int(n)
    for num_threads in (1024, 512, 256, 128, 64):
        if n % num_threads != 0:
            continue
        elements_per_thread = n // num_threads
        if elements_per_thread >= 8 and elements_per_thread % 8 == 0:
            return num_threads
    return None


def pick_rows_per_cta(m: int) -> int:
    m = int(m)
    for rows_per_cta in reversed(RPC_CANDIDATES):
        if m % rows_per_cta != 0:
            continue
        if m // rows_per_cta >= TARGET_MIN_CTAS:
            return rows_per_cta
    return RPC_CANDIDATES[0]


def _validate_shapes(x_tensor, w_tensor, rows_per_cta):
    if len(x_tensor.shape) != 2 or len(w_tensor.shape) != 1:
        raise ValueError("RmsNormRhtAmax expects rank-2 X and rank-1 W tensors")
    m, n = tuple(int(dim) for dim in x_tensor.shape)
    (wn,) = tuple(int(dim) for dim in w_tensor.shape)
    if n != wn:
        raise ValueError("RmsNormRhtAmax X/W dimensions are incompatible")
    if n % 16 != 0:
        raise ValueError("RmsNormRhtAmax requires N to be divisible by 16")
    rows = int(rows_per_cta or 1)
    if rows <= 0:
        raise ValueError("RmsNormRhtAmax rows_per_cta must be positive")
    if m % rows != 0:
        raise ValueError("RmsNormRhtAmax M must be divisible by rows_per_cta")
    return m, n, rows


def _hadamard16(device):
    torch = _torch()
    matrix = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while matrix.shape[0] < 16:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix / 4.0


def _empty_like_x(x_tensor):
    torch = _torch()
    return torch.empty_like(x_tensor)


def _empty_amax(x_tensor, rows_per_cta):
    torch = _torch()
    m = int(x_tensor.shape[0])
    rows = int(rows_per_cta or 1)
    return torch.empty(
        (m // rows,), dtype=torch.float32, device=getattr(x_tensor, "device", None)
    )


def _compute_rmsnorm_rht_amax(**kwargs):
    torch = _torch()
    x = kwargs.get("x_tensor")
    w = kwargs.get("w_tensor")
    if not _is_supported_tensor(x) or not _is_supported_tensor(w):
        raise NotImplementedError(
            "RmsNormRhtAmax torch fallback supports FLOAT, HALF, and BFLOAT16 tensors"
        )
    eps = float(kwargs.get("eps", 1e-5))
    m, n, rows = _validate_shapes(x, w, kwargs.get("rows_per_cta"))
    x_f32 = x.to(dtype=torch.float32)
    w_f32 = w.to(dtype=torch.float32)
    rms = torch.sqrt((x_f32 * x_f32).mean(dim=-1, keepdim=True) + eps)
    y = x_f32 / rms * w_f32.unsqueeze(0)
    hadamard = _hadamard16(getattr(x, "device", None))
    y = torch.matmul(y.view(m, n // 16, 16), hadamard).view(m, n)
    amax = y.abs().view(m // rows, rows, n).amax(dim=(1, 2)).to(torch.float32)
    return y, amax


class RmsNormRhtAmaxSm100:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self._compiled = False

    def check_support(self):
        if self.kwargs.get("sample_x") is None or self.kwargs.get("sample_w") is None:
            raise NotImplementedError(
                "mudnn.RmsNormRhtAmaxSm100 requires sample_x and sample_w tensors"
            )
        return _is_supported_tensor(
            self.kwargs.get("sample_x")
        ) and _is_supported_tensor(self.kwargs.get("sample_w"))

    def compile(self):
        if not self.check_support():
            raise NotImplementedError(
                "mudnn.RmsNormRhtAmaxSm100 torch fallback supports only FLOAT, HALF, "
                "and BFLOAT16 tensors"
            )
        _validate_shapes(
            self.kwargs.get("sample_x"),
            self.kwargs.get("sample_w"),
            self.kwargs.get("rows_per_cta"),
        )
        self._compiled = True
        return self

    def execute(self, **kwargs):
        if not self._compiled:
            self.compile()
        run_kwargs = dict(self.kwargs)
        run_kwargs.update(kwargs)
        if run_kwargs.get("x_tensor") is None:
            run_kwargs["x_tensor"] = run_kwargs.get("sample_x")
        if run_kwargs.get("w_tensor") is None:
            run_kwargs["w_tensor"] = run_kwargs.get("sample_w")
        y_f32, amax_f32 = _compute_rmsnorm_rht_amax(**run_kwargs)
        o_tensor = run_kwargs.get("o_tensor")
        amax_tensor = run_kwargs.get("amax_tensor")
        if o_tensor is None or amax_tensor is None:
            raise ValueError(
                "RmsNormRhtAmax execute requires o_tensor and amax_tensor outputs"
            )
        if tuple(int(dim) for dim in o_tensor.shape) != tuple(y_f32.shape):
            raise ValueError("RmsNormRhtAmax output dimensions are incompatible")
        if tuple(int(dim) for dim in amax_tensor.shape) != tuple(amax_f32.shape):
            raise ValueError("RmsNormRhtAmax amax dimensions are incompatible")
        o_tensor.copy_(y_f32.to(dtype=o_tensor.dtype))
        amax_tensor.copy_(amax_f32)
        return self


def _shape_of(tensor):
    return tuple(getattr(tensor, "shape", ()))


def _dtype_of(tensor):
    return getattr(tensor, "dtype", None)


def _cache_key(**kwargs):
    x_tensor = kwargs.get("x_tensor")
    w_tensor = kwargs.get("w_tensor")
    return (
        _shape_of(x_tensor),
        _shape_of(w_tensor),
        _dtype_of(x_tensor),
        _dtype_of(w_tensor),
        kwargs.get("eps"),
        kwargs.get("num_threads"),
        kwargs.get("rows_per_cta"),
    )


def _tuple_dict(**kwargs):
    return TupleDict(**kwargs)


def rmsnorm_rht_amax_wrapper_sm100(**kwargs):
    x_tensor = kwargs.get("x_tensor")
    w_tensor = kwargs.get("w_tensor")
    if _is_supported_tensor(x_tensor) and _is_supported_tensor(w_tensor):
        _validate_shapes(x_tensor, w_tensor, kwargs.get("rows_per_cta"))
        if kwargs.get("o_tensor") is None:
            kwargs["o_tensor"] = _empty_like_x(x_tensor)
        if kwargs.get("amax_tensor") is None:
            kwargs["amax_tensor"] = _empty_amax(x_tensor, kwargs.get("rows_per_cta"))
    key = _cache_key(**kwargs)
    cached = _cache_of_RmsNormRhtAmaxSm100Objects.get(key)
    if cached is None:
        op = RmsNormRhtAmaxSm100(
            sample_x=kwargs.get("x_tensor"),
            sample_w=kwargs.get("w_tensor"),
            sample_o=kwargs.get("o_tensor"),
            sample_amax=kwargs.get("amax_tensor"),
            eps=kwargs.get("eps"),
            num_threads=kwargs.get("num_threads"),
            rows_per_cta=kwargs.get("rows_per_cta"),
        )
        try:
            is_supported = op.check_support()
            if not is_supported:
                raise NotImplementedError
            op.compile()
        except NotImplementedError as exc:
            raise NotImplementedError(
                "mudnn.rmsnorm_rht_amax_wrapper_sm100 is not supported by muDNN frontend yet"
            ) from exc
        _cache_of_RmsNormRhtAmaxSm100Objects[key] = op
    else:
        op = cached
    op.execute(**kwargs)
    return _tuple_dict(
        o_tensor=kwargs.get("o_tensor"),
        amax_tensor=kwargs.get("amax_tensor"),
    )
