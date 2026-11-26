"""MP31  indexer forward API.

Runs the indexer forward pass with the MP31 TileLang kernel. The kernel is
imported lazily so this module can be imported without the TileLang runtime.
BF16 Q/K only; the mxfp8 path stays MP32-only.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch

from mudnn.api_base import APIBase, TupleDict

_TMA_ALIGN_ELEMS = 4
# MP31 (mp_31) measured limit: the driver rejects an opt-in dynamic-smem
# request of 201728 B ("musa_module.cc:240: Fatal") while ~184 KB requests
# launch fine, so budget against 192 KB.
_SMEM_CAPACITY_BYTES = 192 * 1024


_DTYPE_TO_TILELANG = {
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _resolve_stream(current_stream, device):
    if current_stream is None:
        return torch.musa.current_stream(device)
    if isinstance(current_stream, torch.musa.Stream):
        return current_stream
    return torch.musa.ExternalStream(int(current_stream), device=device)


def _validate_q_causal_offsets(q_causal_offsets, batch: int, device):
    if q_causal_offsets is None:
        return None
    if q_causal_offsets.dtype != torch.int32:
        raise ValueError("q_causal_offsets must be int32")
    if q_causal_offsets.ndim != 1 or q_causal_offsets.shape[0] != batch:
        raise ValueError(
            f"q_causal_offsets must have shape ({batch},), got {tuple(q_causal_offsets.shape)}"
        )
    if q_causal_offsets.device.type != "musa":
        raise ValueError("q_causal_offsets must be a MUSA tensor")
    if q_causal_offsets.device != device:
        raise ValueError("q_causal_offsets must be on the same device as q")
    if q_causal_offsets.stride(0) != 1:
        q_causal_offsets = q_causal_offsets.contiguous()
    return q_causal_offsets


def _smem_bytes(
    stage_kv: int,
    tile_kv: int,
    tile_qh: int,
    headdim: int,
    dtype_bytes: int,
    weight_bytes: int,
    scale_words: int = 0,
) -> int:
    q_bytes = tile_qh * headdim * dtype_bytes
    kv_bytes = stage_kv * tile_kv * headdim * dtype_bytes
    w_bytes = tile_qh * weight_bytes
    scale_bytes = scale_words * 4
    return kv_bytes + q_bytes + w_bytes + scale_bytes


class IndexerForward(APIBase):
    """MP31 BF16 Q/K APIBase shell with BF16 or FP32 per-head weights.

    The interface layer owns lazy compilation and its kernel cache. The
    optional stream is a raw MUSA handle (``musaStream_t``); ``None`` uses
    the current MUSA stream.
    """

    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_w: torch.Tensor,
        sample_out: torch.Tensor,
        ratio: int = 4,
        qhead_per_kv_head: Optional[int] = None,
        m_block_size: int = 128,
        n_block_size: int = 256,
        q_stage: int = 2,
        kv_stage: int = 4,
        sm_scale: float = 1.0,
    ):
        super().__init__()

        self.q_desc = self._make_tensor_desc(sample_q, name="sample_q")
        self.k_desc = self._make_tensor_desc(sample_k, name="sample_k")
        self.w_desc = self._make_tensor_desc(sample_w, name="sample_w")
        self.o_desc = self._make_tensor_desc(sample_out, name="sample_out")

        self.ratio = int(ratio)
        self.qhead_per_kv_head = qhead_per_kv_head
        self.m_block_size = int(m_block_size)
        self.n_block_size = int(n_block_size)
        self.q_stage = int(q_stage)
        self.kv_stage = int(kv_stage)
        self.sm_scale = float(sm_scale)

        self.batch_size = None
        self.s_q = None
        self.s_k = None
        self.s_k_padded = None
        self.h_q = None
        self.h_kv = None
        self.head_dim = None

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self._value_error_if(
            self.q_desc.ndim != 4,
            f"Q must be 4-D (B, S_q, H_q, D), got {self.q_desc.shape}",
        )
        self._value_error_if(
            self.k_desc.ndim != 4,
            f"K must be 4-D (B, S_k, H_kv, D), got {self.k_desc.shape}",
        )
        self._value_error_if(
            self.w_desc.ndim != 3,
            f"W must be 3-D (B, S_q, H_q), got {self.w_desc.shape}",
        )
        self._value_error_if(
            self.o_desc.ndim != 3,
            f"Out must be 3-D (B, S_q, S_k_padded), got {self.o_desc.shape}",
        )

        b, s_q, h_q, d = self.q_desc.shape
        b_k, s_k, h_kv, d_k = self.k_desc.shape
        b_o, s_q_out, s_k_padded = self.o_desc.shape

        self._value_error_if(b != b_k, f"Batch size mismatch Q={b} vs K={b_k}")
        self._value_error_if(b != b_o, f"Batch size mismatch Q={b} vs Out={b_o}")
        self._value_error_if(s_q != s_q_out, f"S_q mismatch Q={s_q} vs Out={s_q_out}")
        self._value_error_if(d != d_k, f"Head dim mismatch Q={d} vs K={d_k}")
        self._value_error_if(
            d != 128,
            f"IndexerForward is tuned for head_dim=128 only, got {d}",
        )

        qhpkv = (
            self.qhead_per_kv_head
            if self.qhead_per_kv_head is not None
            else (h_q // h_kv)
        )
        self._value_error_if(
            qhpkv * h_kv != h_q,
            f"qhead_per_kv_head * h_kv != h_q ({qhpkv} * {h_kv} != {h_q})",
        )
        self._value_error_if(
            qhpkv not in (32, 64),
            f"qhead_per_kv_head must be 32 or 64, got {qhpkv}",
        )
        self.qhead_per_kv_head = qhpkv

        self._check_dtype(self.q_desc, torch.bfloat16, name="Q")
        self._check_dtype(self.k_desc, torch.bfloat16, name="K")
        self._check_dtype(
            self.w_desc, [torch.bfloat16, torch.float32], name="W"
        )
        if (
            self.w_desc.dtype == torch.float32
            and self.w_desc.stride[-1] != 1
        ):
            raise NotImplementedError(
                f"FP32 W requires unit last stride, got strides {self.w_desc.stride}"
            )
        self._check_dtype(self.o_desc, torch.float32, name="Out")

        self._value_error_if(
            s_k_padded % _TMA_ALIGN_ELEMS != 0,
            f"Out seqlen_k dim must be a multiple of {_TMA_ALIGN_ELEMS}, got {s_k_padded}",
        )
        self._value_error_if(
            s_k_padded < s_k,
            f"Out seqlen_k dim ({s_k_padded}) must cover logical K length ({s_k})",
        )
        self._value_error_if(
            self.m_block_size not in (64, 128),
            f"MP31 indexer forward supports m_block_size in (64, 128), got {self.m_block_size}",
        )
        self._value_error_if(
            self.m_block_size % qhpkv != 0,
            f"m_block_size ({self.m_block_size}) must be divisible by "
            f"qhead_per_kv_head ({qhpkv})",
        )
        max_q_tokens = 2 if qhpkv == 64 else 4
        self._value_error_if(
            self.m_block_size // qhpkv > max_q_tokens,
            f"MP31 indexer forward supports at most {max_q_tokens} q tokens per tile, "
            f"got m_block_size={self.m_block_size}",
        )
        self._value_error_if(
            self.n_block_size not in (128, 256, 512),
            f"MP31 indexer forward supports n_block_size in (128, 256, 512), got {self.n_block_size}",
        )
        self._value_error_if(
            self.q_stage != 2,
            f"MP31 indexer forward supports q_stage=2 only, got {self.q_stage}",
        )
        self._value_error_if(
            self.kv_stage % 2 != 0 or not (2 <= self.kv_stage <= 12),
            f"MP31 indexer forward requires an even kv_stage in [2, 12], got {self.kv_stage}",
        )

        stage_kv = self.kv_stage // 2
        weight_bytes = 4 if self.w_desc.dtype == torch.float32 else 2
        self._value_error_if(
            _smem_bytes(stage_kv, self.n_block_size, self.m_block_size, d, 2, weight_bytes)
            > _SMEM_CAPACITY_BYTES,
            f"MP31 indexer forward smem budget exceeded for "
            f"n_block_size={self.n_block_size}, kv_stage={self.kv_stage}",
        )

        if torch.musa.is_available():
            capability = torch.musa.get_device_capability(self.q_desc.device)
            self._runtime_error_if(
                tuple(capability) != (3, 1),
                f"IndexerForward requires MP31 (3.1), found {capability}",
            )
        else:
            self._runtime_error_if(
                True, "IndexerForward requires a MUSA device"
            )

        self.batch_size = b
        self.s_q = s_q
        self.s_k = s_k
        self.s_k_padded = s_k_padded
        self.h_q = h_q
        self.h_kv = h_kv
        self.head_dim = d
        self._is_supported = True
        return True

    def compile(self) -> None:
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        self._compiled_kernel = True
        return None

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        w: torch.Tensor,
        out: torch.Tensor,
        current_stream: Optional[int] = None,
    ) -> None:
        self._logger.debug("Entering execute")
        if self._compiled_kernel is None:
            raise ValueError("IndexerForward kernel not compiled")
        self._check_dtype(w, self.w_desc.dtype, name="W")

        logical_out = out[..., : self.s_k]
        _indexer_fwd_mp31(
            q,
            k,
            w,
            ratio=self.ratio,
            qhead_per_kv_head=self.qhead_per_kv_head,
            out=logical_out,
            m_block_size=self.m_block_size,
            n_block_size=self.n_block_size,
            q_stage=self.q_stage,
            kv_stage=self.kv_stage,
            sm_scale=self.sm_scale,
            current_stream=current_stream,
        )
        return None


def _indexer_fwd_mp31(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    m_block_size: int = 128,
    n_block_size: int = 256,
    num_threads: Optional[int] = None,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    current_stream=None,
):
    """Run the BF16 MP31 indexer forward.

    ``num_threads`` is derived from ``n_block_size`` (128 + n_block_size);
    pass ``None`` or the matching value. ``out`` is written directly by the
    kernel, so no S_k alignment padding is required. BF16 weights are converted
    to FP32 before the FP32 multiply until MTCC exposes the mixed-precision FMA.

    THD inputs take q/k/w as ``(total_q, H, D)``/``(total_k, 1, D)``/
    ``(total_q, H)``, require ``cu_seqlens_q/k`` and ``max_seqlen_q/k``, and
    write ``out`` as ``(total_q, max_seqlen_k)`` with segment-local KV columns.
    Both BSHD and THD compile with or without a ``q_causal_offsets`` argument.
    """

    is_varlen = cu_seqlens_q is not None
    if is_varlen != (cu_seqlens_k is not None):
        raise ValueError("THD input requires both cu_seqlens_q and cu_seqlens_k")
    if is_varlen and (max_seqlen_q is None or max_seqlen_k is None):
        raise ValueError("THD input requires max_seqlen_q and max_seqlen_k")

    if ratio <= 0:
        raise ValueError(f"ratio must be > 0, got {ratio}")
    if q_stage != 2:
        raise ValueError(
            f"MP31 indexer forward only supports q_stage=2, got {q_stage}"
        )
    if m_block_size not in (64, 128):
        raise ValueError(
            f"MP31 indexer forward supports m_block_size in (64, 128), got {m_block_size}"
        )
    if n_block_size not in (128, 256, 512):
        raise ValueError(
            f"MP31 indexer forward supports n_block_size in (128, 256, 512), got {n_block_size}"
        )
    derived_threads = 128 + int(n_block_size)
    if num_threads is not None and int(num_threads) != derived_threads:
        raise ValueError(
            f"MP31 indexer forward derives num_threads={derived_threads} for "
            f"n_block_size={n_block_size}, got {num_threads}"
        )
    if kv_stage % 2 != 0 or not (2 <= kv_stage <= 12):
        raise ValueError(
            f"MP31 indexer forward requires an even kv_stage in [2, 12], got {kv_stage}"
        )

    if is_varlen:
        if q.ndim != 3:
            raise ValueError(f"THD q must be 3-D (total_q, H_q, D), got {q.ndim}-D")
        if k.ndim != 3:
            raise ValueError(f"THD k must be 3-D (total_k, H_kv, D), got {k.ndim}-D")
        if w.ndim != 2:
            raise ValueError(f"THD w must be 2-D (total_q, H_q), got {w.ndim}-D")
    else:
        if q.ndim != 4:
            raise ValueError(f"q must be 4-D (B, S_q, H_q, D), got {q.ndim}-D")
        if k.ndim != 4:
            raise ValueError(f"k must be 4-D (B, S_k, H_kv, D), got {k.ndim}-D")
        if w.ndim != 3:
            raise ValueError(f"w must be 3-D (B, S_q, H_q), got {w.ndim}-D")

    if q.dtype != torch.bfloat16:
        raise ValueError(f"q must be bfloat16, got {q.dtype}")
    if k.dtype != torch.bfloat16:
        raise ValueError(f"k must be bfloat16, got {k.dtype}")
    if w.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"w must be bfloat16 or float32, got {w.dtype}")
    if w.dtype == torch.float32 and w.stride(-1) != 1:
        raise NotImplementedError(
            f"FP32 w requires unit last stride, got strides {w.stride()}"
        )

    if q.device != k.device or q.device != w.device:
        raise ValueError("q, k, w must be on the same device")

    if is_varlen:
        total_q, n_heads_q, head_dim = q.shape
        total_k, n_heads_kv, head_dim_k = k.shape
        batch = int(cu_seqlens_q.shape[0]) - 1
        seqlen_q = int(max_seqlen_q)
        seqlen_k = int(max_seqlen_k)
        for cu_name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
            if cu.dtype != torch.int32 or cu.ndim != 1 or cu.shape[0] != batch + 1:
                raise ValueError(
                    f"{cu_name} must be int32 with shape ({batch + 1},)"
                )
            if cu.device != q.device:
                raise ValueError(f"{cu_name} must be on {q.device}")
            if cu.stride(0) != 1:
                raise ValueError(f"{cu_name} must be contiguous")
        if w.shape != (total_q, n_heads_q):
            raise ValueError(
                f"THD w must have shape {(total_q, n_heads_q)}, got {tuple(w.shape)}"
            )
    else:
        batch, seqlen_q, n_heads_q, head_dim = q.shape
        batch_k, seqlen_k, n_heads_kv, head_dim_k = k.shape
        if batch != batch_k:
            raise ValueError(f"batch mismatch q={batch} vs k={batch_k}")
        if w.shape != (batch, seqlen_q, n_heads_q):
            raise ValueError(
                f"w must have shape {(batch, seqlen_q, n_heads_q)}, got {tuple(w.shape)}"
            )

    if head_dim != head_dim_k:
        raise ValueError(f"head_dim mismatch q={head_dim} vs k={head_dim_k}")
    if head_dim != 128:
        raise ValueError(f"MP31 indexer forward requires head_dim=128, got {head_dim}")
    if n_heads_kv != 1:
        raise ValueError(
            f"MP31 indexer forward requires n_heads_kv=1 (MQA), got {n_heads_kv}"
        )

    if qhead_per_kv_head is None:
        qhead_per_kv_head = n_heads_q // n_heads_kv
    if qhead_per_kv_head * n_heads_kv != n_heads_q:
        raise ValueError(
            f"qhead_per_kv_head ({qhead_per_kv_head}) must equal n_heads_q // n_heads_kv ({n_heads_q // n_heads_kv})"
        )
    if qhead_per_kv_head not in (32, 64):
        raise ValueError(
            f"qhead_per_kv_head must be 32 or 64, got {qhead_per_kv_head}"
        )
    if m_block_size % qhead_per_kv_head != 0:
        raise ValueError(
            f"m_block_size ({m_block_size}) must be divisible by "
            f"qhead_per_kv_head ({qhead_per_kv_head})"
        )
    max_q_tokens = 2 if qhead_per_kv_head == 64 else 4
    if m_block_size // qhead_per_kv_head > max_q_tokens:
        raise ValueError(
            f"MP31 indexer forward supports at most {max_q_tokens} q tokens per tile, "
            f"got m_block_size={m_block_size}"
        )

    tile_qh = int(m_block_size)
    tile_kv = int(n_block_size)
    stage_q = q_stage // 2
    stage_kv = kv_stage // 2
    dtype_weight = _DTYPE_TO_TILELANG[w.dtype]
    weight_bytes = 4 if w.dtype == torch.float32 else 2
    dtype_bytes = 2
    scale_words = 0
    if (
        _smem_bytes(
            stage_kv,
            tile_kv,
            tile_qh,
            head_dim,
            dtype_bytes,
            weight_bytes,
            scale_words,
        )
        > _SMEM_CAPACITY_BYTES
    ):
        raise ValueError(
            f"MP31 indexer forward smem budget exceeded for "
            f"n_block_size={n_block_size}, kv_stage={kv_stage}"
        )

    stream = _resolve_stream(current_stream, q.device)
    active = torch.musa.current_stream(q.device)
    stream_ctx = torch.musa.stream(stream) if stream != active else nullcontext()
    with stream_ctx:
        if q.stride(-1) != 1:
            q = q.contiguous()
        if k.stride(-1) != 1:
            k = k.contiguous()
        if w.stride(-1) != 1:
            w = w.contiguous()

        k3 = k.squeeze(1) if is_varlen else k.squeeze(2)

        q_causal_offsets = _validate_q_causal_offsets(
            q_causal_offsets, batch, q.device
        )
        has_q_causal_offsets = q_causal_offsets is not None

        out_shape = (total_q, seqlen_k) if is_varlen else (batch, seqlen_q, seqlen_k)
        if out is None:
            out = torch.empty(
                out_shape,
                dtype=torch.float32,
                device=q.device,
            )
        else:
            if tuple(out.shape) != tuple(out_shape):
                raise ValueError(
                    f"out must have shape {out_shape}, got {tuple(out.shape)}"
                )
            if out.dtype != torch.float32:
                raise ValueError(f"out must be float32, got {out.dtype}")
            if out.device != q.device:
                raise ValueError(f"out must be on {q.device}, got {out.device}")
            if out.stride(-1) != 1:
                raise ValueError(f"out requires unit last stride, got {out.stride()}")

        from .indexer_forward_mp31 import indexer_forward_mp31

        out.fill_(float("-inf"))
        kernel = indexer_forward_mp31(
            headdim=128,
            head_q=qhead_per_kv_head,
            tile_qh=tile_qh,
            tile_kv=tile_kv,
            stage_q=stage_q,
            stage_kv=stage_kv,
            dtype="bfloat16",
            dtype_weight=dtype_weight,
            is_varlen=is_varlen,
            ratio=ratio,
            sm_scale=sm_scale,
            has_q_causal_offsets=has_q_causal_offsets,
        )

        if is_varlen:
            if has_q_causal_offsets:
                kernel(q, k3, w, q_causal_offsets, out, cu_seqlens_q, cu_seqlens_k, seqlen_q)
            else:
                kernel(q, k3, w, out, cu_seqlens_q, cu_seqlens_k, seqlen_q)
        elif has_q_causal_offsets:
            kernel(q, k3, w, q_causal_offsets, out)
        else:
            kernel(q, k3, w, out)
        return out








def indexer_forward_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ratio: int = 4,
    qhead_per_kv_head: Optional[int] = None,
    m_block_size: int = 128,
    n_block_size: int = 256,
    q_stage: int = 2,
    kv_stage: int = 4,
    sm_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    stream: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    precision: str = "bf16",
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_scale_padded: Optional[torch.Tensor] = None,
    sf_vec_size: int = 32,
    return_lse: bool = False,
    lse_out: Optional[torch.Tensor] = None,
) -> TupleDict:
    """High-level wrapper. Allocates an exact-shape output buffer and writes it directly.

    Q/K are BF16 and per-head W may be BF16 or FP32 (``precision='bf16'``,
    the only value MP31 supports). Returns ``{'scores': (B, S_q, S_k) FP32}``.
    The ratio causal mask marks positions outside the valid KV range with
    -inf. ``q_causal_offsets`` may specify the global uncompressed token
    index for each batch/THD segment's local q[0].

    ``stream`` is a raw MUSA stream handle (``musaStream_t``) or a
    ``torch.musa.Stream``; ``None`` uses the current MUSA stream.
    """
    if return_lse or lse_out is not None:
        raise NotImplementedError("MP31 dense indexer forward does not expose LSE")

    precision = precision.lower()
    if precision == "bf16":
        if (
            q_scale is not None
            or k_scale is not None
            or cu_seqlens_q_scale_padded is not None
            or cu_seqlens_k_scale_padded is not None
        ):
            raise ValueError(
                "q_scale, k_scale, and scale padded cu_seqlens are not "
                "supported on MP31 (bf16 only)"
            )
        scores = _indexer_fwd_mp31(
            q,
            k,
            w,
            ratio=ratio,
            qhead_per_kv_head=qhead_per_kv_head,
            out=out,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            kv_stage=kv_stage,
            sm_scale=sm_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_causal_offsets=q_causal_offsets,
            current_stream=stream,
        )
    elif precision == "mxfp8":
        raise NotImplementedError(
            "MP31  indexer forward supports precision='bf16' only; "
            "mxfp8 stays MP32-only"
        )
    else:
        raise ValueError(
            f"MP31 indexer forward supports precision='bf16' only, "
            f"got {precision!r}"
        )
    return TupleDict(scores=scores)
