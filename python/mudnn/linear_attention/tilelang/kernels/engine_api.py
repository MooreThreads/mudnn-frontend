"""Public names for the kernel entry points the GDN engine calls."""

from __future__ import annotations

from .gdn_chunk_bwd import _UNPADDED_ENABLED
from .gdn_chunk_bwd import _prepare_dense as prepare_dense
from .gdn_chunk_bwd import _run_gdn_chunk_bwd as run_backward
from .gdn_chunk_fwd import _tl_contiguous as tl_contiguous
from .gdn_chunk_fwd import gdn_chunk_fwd as chunk_forward
from .gdn_shapes import (CHUNK, HEAD_DIMS, gdn_dense_error, gdn_dense_supported,
                         gdn_max_heads, gdn_unpadded_supported, gdn_varlen_error,
                         gdn_varlen_supported)
from .gdn_varlen import varlen_maps


def dense_pad_rows(B, S, H, DK, DV, initial_state, output_final_state) -> int:
    """Rows appended to a dense call of B x S x H with the given head dims;
    0 when no padding is needed.
    """
    pad = (-S) % CHUNK
    unpadded = (
        _UNPADDED_ENABLED
        and bool(pad)
        and not output_final_state
        and initial_state is None
        and gdn_unpadded_supported(B, S, H, DK, DV, CHUNK)
    )
    return 0 if unpadded else pad


__all__ = [
    "CHUNK",
    "HEAD_DIMS",
    "chunk_forward",
    "dense_pad_rows",
    "gdn_dense_error",
    "gdn_dense_supported",
    "gdn_max_heads",
    "gdn_varlen_error",
    "gdn_varlen_supported",
    "prepare_dense",
    "run_backward",
    "tl_contiguous",
    "varlen_maps",
]
