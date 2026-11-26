"""Pure Python Hadamard helpers for grouped GEMM GLU-Hadamard compatibility."""

HADAMARD_SIZE = 16


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - upstream tests require torch
        raise NotImplementedError("hadamard_matrix requires torch") from exc
    return torch


def hadamard_matrix(n, dtype=None, device=None):
    torch = _torch()
    if dtype is None:
        dtype = torch.float32
    n = int(n)
    if n < 1:
        raise ValueError("n must be a positive integer")
    if n & (n - 1):
        raise ValueError("n must be a power of 2")

    kwargs = {"dtype": dtype}
    if device is not None:
        kwargs["device"] = device
    matrix = torch.tensor([[1]], **kwargs)
    base = torch.tensor([[1, 1], [1, -1]], **kwargs)
    while matrix.shape[0] < n:
        matrix = torch.kron(matrix, base)
    return matrix


__all__ = ["HADAMARD_SIZE", "hadamard_matrix"]
