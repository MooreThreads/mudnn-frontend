"""Structural precondition of the TileLang GDN kernels; a shape it rejects raises."""

from __future__ import annotations

#: An explicit SET, never ``DK % 64 == 0``: DK=96 compiles and silently returns
#: garbage, DK=192 needs a K2 split form nothing here implements.
_HEAD_DIMS = (64, 128)
_CHUNK = 64
HEAD_DIMS = _HEAD_DIMS
CHUNK = _CHUNK
_MAX_CLASSIFY_FLAGS = 4096

_SHARED_LIMIT = 196608
_RELAYOUT_TILES = 2
_RELAYOUT_TS = 64


def gdn_max_heads() -> int:
    """Largest H whose backward scalar relayout still fits in shared memory."""
    return _SHARED_LIMIT // (_RELAYOUT_TILES * (_RELAYOUT_TS + 1) * 4)


def head_dim_supported(DK, DV, chunk_size: int = _CHUNK) -> bool:
    """True for a head-dim tier the kernels are derived for.

    ``DK == DV`` is the invariant K2's single ``tmp_cv`` accumulator rests on;
    an asymmetric tier builds clean and returns four wrong gradients.
    """
    return DK == DV and DK in _HEAD_DIMS and chunk_size == _CHUNK


def gdn_head_dim_error(kind, DK, DV, chunk_size=_CHUNK) -> str:
    """The message a kernel raises for a head dim it is not derived for."""
    return (f"{kind} is derived for DK == DV in {_HEAD_DIMS} at chunk "
            f"{_CHUNK}; got DK={DK}, DV={DV}, chunk={chunk_size}")


def _gdn_shape_reasons(B, S, H, DK, DV, chunk_size=64) -> list[str]:
    """Every clause the shape fails, phrased for a caller. Empty means supported."""
    reasons = []
    if DK != DV or DK not in _HEAD_DIMS:
        reasons.append(
            f"DK and DV must be equal and in {_HEAD_DIMS}, got DK={DK} DV={DV}")
    if chunk_size != _CHUNK:
        reasons.append(f"chunk_size must be {_CHUNK}, got {chunk_size}")
    elif S % chunk_size:
        reasons.append(f"S must be a multiple of {chunk_size}, got S={S}")
    elif S // chunk_size < 2:
        reasons.append(
            f"S must span at least two chunks (S >= {2 * chunk_size}), got S={S}"
        )
    if B < 1:
        reasons.append(f"B must be at least 1, got B={B}")
    limit = gdn_max_heads()
    if not 1 <= H <= limit:
        reasons.append(f"H must be in 1..{limit}, got H={H}")
    if B >= 1 and H >= 1 and B * H > _MAX_CLASSIFY_FLAGS:
        reasons.append(
            f"B*H must be at most {_MAX_CLASSIFY_FLAGS}, got B*H={B * H}")
    return reasons


def gdn_shape_supported(B, S, H, DK, DV, chunk_size=64) -> bool:
    """True when the TileLang GDN kernels are structurally valid for a shape."""
    return not _gdn_shape_reasons(B, S, H, DK, DV, chunk_size)


def gdn_unpadded_supported(B, S, H, DK, DV, chunk_size=64) -> bool:
    """True when a non-aligned ``S`` can run without padding q/k/v/g/beta.

    Two conditions beyond the usual precondition:

    * ``B == 1`` -- for B > 1 the flattened layout and the backward's chunk-aligned
      addressing disagree, and cu_seqlens cannot express the gap.
    * the caller must not want the final state; that clause is the caller's to
      apply, not a property of the shape.
    """
    if S % chunk_size == 0 or B != 1:
        return False
    if S < 2 * chunk_size:
        return False
    aligned = -(-S // chunk_size) * chunk_size
    return gdn_shape_supported(B, aligned, H, DK, DV, chunk_size)


def gdn_dense_supported(B, S, H, DK, DV, chunk_size=_CHUNK) -> bool:
    """True when a dense call runs, padding ``S`` to the chunk if it must.

    ``gdn_shape_supported`` is the predicate for an already chunk-aligned ``S``;
    this is the question a caller with a ragged ``S`` actually has.
    ``gdn_unpadded_supported`` then says whether that padding costs a copy.
    """
    aligned = -(-S // chunk_size) * chunk_size
    return gdn_shape_supported(B, aligned, H, DK, DV, chunk_size)


def gdn_dense_error(kind, B, S, H, DK, DV, chunk_size=_CHUNK) -> str:
    """The message a caller gets for a dense shape the operator cannot serve."""
    aligned = -(-S // chunk_size) * chunk_size
    note = f" (S={aligned} after padding from {S})" if aligned != S else ""
    return gdn_shape_error(kind, B, aligned, H, DK, DV, chunk_size) + note


def _gdn_varlen_reasons(seqs, H, DK, DV, chunk_size=_CHUNK) -> list[str]:
    """Every clause a packed batch fails. ``seqs`` is the list of token counts."""
    seqs = list(seqs)
    if not seqs:
        return ["cu_seqlens must describe at least one sequence"]
    if any(L <= 0 for L in seqs):
        return [f"every sequence must be non-empty, got {seqs}"]
    aligned = sum(-(-L // chunk_size) * chunk_size for L in seqs)
    reasons = _gdn_shape_reasons(1, aligned, H, DK, DV, chunk_size)
    short = [L for L in seqs if -(-L // chunk_size) < 2]
    if short:
        reasons.append(
            f"every sequence must span at least two chunks, got {short} "
            f"({[-(-L // chunk_size) for L in short]} chunks)")
    if len(seqs) * H > _MAX_CLASSIFY_FLAGS:
        reasons.append(
            f"NS*H must be at most {_MAX_CLASSIFY_FLAGS}, "
            f"got NS*H={len(seqs) * H}")
    return reasons


def gdn_varlen_supported(seqs, H, DK, DV, chunk_size=_CHUNK) -> bool:
    """True when the kernels can serve a packed batch of these lengths."""
    return not _gdn_varlen_reasons(seqs, H, DK, DV, chunk_size)


def gdn_varlen_error(kind, seqs, H, DK, DV, chunk_size=_CHUNK) -> str:
    """The message a caller gets for a packed batch the kernels cannot serve."""
    reasons = _gdn_varlen_reasons(seqs, H, DK, DV, chunk_size) or ["unsupported"]
    return (f"{kind} does not support the packed batch "
            f"(NS={len(list(seqs))}, H={H}, DK={DK}, DV={DV}, C={chunk_size}): "
            + "; ".join(reasons))


def gdn_shape_error(kind, B, S, H, DK, DV, chunk_size=64) -> str:
    """The message a kernel raises when handed an unsupported shape."""
    reasons = _gdn_shape_reasons(B, S, H, DK, DV, chunk_size) or ["shape unsupported"]
    return (
        f"{kind} does not support (B,S,H,DK,DV,C)="
        f"({B}, {S}, {H}, {DK}, {DV}, {chunk_size}): " + "; ".join(reasons)
    )
