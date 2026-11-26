from .._torch_fallback import IndexerTopK, indexer_top_k_wrapper

try:
    from mudnn.api_base import TupleDict
except Exception:
    TupleDict = dict

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in frontend tests.
    torch = None


def _require_torch():
    if torch is None:
        raise NotImplementedError("cudnn DSA index utilities require torch")
    return torch


def _validate_index_tensor(name, tensor, ranks, dtypes):
    torch_mod = _require_torch()
    if tensor is None or not hasattr(tensor, "shape"):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dim() not in ranks:
        raise ValueError(f"{name} must be rank-{sorted(ranks)}")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} must be int32 or int64")
    return torch_mod


def _batch_for_thd_rows(num_rows: int, cu_seqlens_q):
    torch_mod = _require_torch()
    row_ids = torch_mod.arange(num_rows, device=cu_seqlens_q.device)
    return torch_mod.searchsorted(cu_seqlens_q[1:], row_ids, right=True)


def local_to_global_wrapper(
    local_indices,
    seqlen_k: int,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    stream=None,
):
    """Convert local top-k indices to global BSHD/THD index space."""
    del stream
    torch_mod = _validate_index_tensor(
        "local_indices", local_indices, {2, 3}, {_require_torch().int32, torch.int64}
    )
    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None
    if is_varlen:
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError(
                "THD local_to_global requires both cu_seqlens_q and cu_seqlens_k"
            )
        if local_indices.dim() != 2:
            raise ValueError("THD local_indices must be 2D")
        for tensor, name in (
            (cu_seqlens_q, "cu_seqlens_q"),
            (cu_seqlens_k, "cu_seqlens_k"),
        ):
            if tensor is None or not hasattr(tensor, "shape"):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.dim() != 1 or tensor.dtype != torch_mod.int32:
                raise ValueError(f"{name} must be a 1D int32 tensor")
            if tensor.device != local_indices.device:
                raise ValueError(f"{name} must be on the same device as local_indices")
        if cu_seqlens_q.shape != cu_seqlens_k.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_k must have the same shape")
        batches = _batch_for_thd_rows(int(local_indices.shape[0]), cu_seqlens_q)
        offsets = cu_seqlens_k.to(torch_mod.int64).index_select(0, batches)
        global_indices = local_indices.to(torch_mod.int64) + offsets.unsqueeze(-1)
    else:
        if local_indices.dim() != 3:
            raise ValueError("BSHD local_indices must be 3D")
        batch = torch_mod.arange(
            int(local_indices.shape[0]),
            dtype=torch_mod.int64,
            device=local_indices.device,
        )
        offsets = batch.view(-1, 1, 1) * int(seqlen_k)
        global_indices = local_indices.to(torch_mod.int64) + offsets
    global_indices = torch_mod.where(
        local_indices >= 0,
        global_indices,
        torch_mod.full_like(global_indices, -1),
    )
    return TupleDict(indices=global_indices.to(torch_mod.int32))


def compactify_wrapper(indices, stream=None):
    """Pack non-negative indices to the front of each row.

    Row ``r`` of the result holds the row's valid (``>= 0``) indices in their
    original relative order, followed by ``-1`` padding, and ``topk_length[r]``
    is the number of valid slots.  This is a stable partition, which is what the
    upstream kernel's ordered compaction emits.

    Implemented as one stable sort on a per-row validity key instead of a
    per-row Python loop: at the DSV4 training shape the loop ran 4096 rows of
    several small torch calls each, which cost ~800 ms per call against ~1 ms
    for the single vectorized pass.
    """
    del stream
    torch_mod = _validate_index_tensor(
        "indices", indices, {2, 3}, {_require_torch().int32}
    )
    original_ndim = indices.dim()
    flat = indices.contiguous().reshape(-1, int(indices.shape[-1]))
    width = int(flat.shape[-1])
    valid = flat >= 0
    topk_length = valid.sum(dim=-1).to(torch_mod.int32)
    # Stable sort with ``invalid`` as the key keeps every valid index in its
    # original relative order and moves the ``-1`` slots to the tail.
    order = torch_mod.argsort(
        (~valid).to(torch_mod.int8), dim=-1, stable=True
    )
    out = torch_mod.gather(flat, 1, order)
    positions = torch_mod.arange(width, device=flat.device)
    out = torch_mod.where(
        positions < topk_length.unsqueeze(-1), out, torch_mod.full_like(out, -1)
    )
    if original_ndim == 3:
        out = out.reshape(indices.shape)
    return TupleDict(indices=out, topk_length=topk_length)


__all__ = [
    "IndexerTopK",
    "indexer_top_k_wrapper",
    "local_to_global_wrapper",
    "compactify_wrapper",
]
