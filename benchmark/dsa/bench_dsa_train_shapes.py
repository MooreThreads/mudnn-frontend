"""DSV4 training-shape DSA/CSA operator benchmark.

Measures the public ``mudnn.DSA`` / ``mudnn.CSA`` entry points at the shapes
``deepseek-v4-lite/run_pretrain_deepseekv4_musa.sh`` configures, so the same
case table can be run on an mp_31 and an mp_32 machine and differenced.

The case table is deliberately split by *why* a case is listed:

``--group added``   the three entry points this work had to add
                    (``indexer_forward_top_k_wrapper`` and the two candidate
                    buffer sizing helpers)
``--group dsa``     the rest of the DSA family the training graph calls
``--group attn``    the two sparse-attention entry points that dominate the
                    DSV4 step, at the exact shapes the reference spreadsheet
                    lists for ``CSASparseAttnFunc`` / ``CSASparseAttnFuncBackward``
``--group csa``     the fused CSA/HCA compressor pair

Timing is wrapper-level: one ``mean_us`` per call includes launch overhead,
because that is what the training step pays.  It is *not* the trace kernel
time the reference spreadsheet reports for the traced ops, so compare these
numbers across machines, not against the spreadsheet's ``mean_us_mp31``
column.

Every case is wrapped in try/except: an unsupported arch, a missing optional
dependency or an OOM records ``status`` and the reason instead of aborting the
run, so one bad case cannot hide the rest.  A case that raises is *not*
reported as a timing.

Usage::

    python benchmark/dsa/bench_dsa_train_shapes.py --group all \
        --out /tmp/dsa_mp31.json

    python benchmark/dsa/bench_dsa_train_shapes.py --group added --device 3
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import sys
import time
import traceback

import torch

DEVICE = "musa"

#: Which Python package tree provides the DSA/CSA surface.
#:
#: ``mudnn``  -- this checkout (``python/mudnn``), the frontend under test.
#: ``cudnn``  -- an upstream-style ``cudnn-frontend`` checkout, selected with
#:               ``--cudnn-frontend-path``.  Used to reproduce the reference
#:               numbers against a separately checked-out cuDNN Frontend whose
#:               ``cudnn.DSA`` the ``--cudnn-frontend-path`` points at.
_FRONTEND = {"flavor": "mudnn", "ns": None}


def select_frontend(flavor: str, cudnn_path: str = ""):
    """Bind ``_FRONTEND`` to the ``DSA``/``CSA`` namespace of ``flavor``.

    Both flavors expose the same wrapper names, so every builder below reads the
    namespace from here instead of importing ``mudnn`` directly.  The CSA side
    differs in spelling (``cudnn.csa.compressor`` vs ``mudnn.CSA``), which is why
    the compressor is resolved through ``_csa_compressors()``.
    """
    global _FRONTEND
    if flavor == "mudnn":
        import mudnn

        from mudnn.csa.compressor import (
            csa_compressor_backward_wrapper,
            csa_compressor_forward_wrapper,
        )

        _FRONTEND = {
            "flavor": "mudnn",
            "ns": mudnn.DSA,
            "csa_fwd": csa_compressor_forward_wrapper,
            "csa_bwd": csa_compressor_backward_wrapper,
            "module": getattr(mudnn, "__file__", "?"),
        }
        return
    if flavor != "cudnn":
        raise ValueError(f"unknown frontend flavor {flavor!r}")
    if not cudnn_path:
        raise ValueError("--frontend cudnn requires --cudnn-frontend-path")
    import os
    import sys

    sys.path.insert(0, cudnn_path)
    import importlib

    module = importlib.import_module("cudnn")
    from importlib import import_module as _import

    compressor = _import("cudnn.csa.compressor")
    _FRONTEND = {
        "flavor": "cudnn",
        "ns": module.DSA,
        "csa_fwd": compressor.csa_compressor_forward_wrapper,
        "csa_bwd": compressor.csa_compressor_backward_wrapper,
        "module": os.path.join(cudnn_path, "cudnn", "__init__.py"),
    }


def _dsa():
    """The DSA namespace of the selected frontend."""
    if _FRONTEND["ns"] is None:
        select_frontend("mudnn")
    return _FRONTEND["ns"]

#: DSV4 training config (``run_pretrain_deepseekv4_musa.sh``).
SEQ = 4096
HEADS = 64
D_IDX = 128
D_ATTN = 576
D_V = 512
TOPK = 512
RATIO_EVEN = 4
RATIO_ODD = 128


def _capability():
    try:
        return tuple(torch.musa.get_device_capability())
    except Exception:  # noqa: BLE001 - metadata only
        return None


def _device_name():
    try:
        return torch.musa.get_device_name()
    except Exception:  # noqa: BLE001
        try:
            return torch.musa.get_device_properties(0).name
        except Exception:  # noqa: BLE001
            return "unknown"


def _git_value(path, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _env_metadata(frontend_root):
    meta = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device_name": _device_name(),
        "capability": _capability(),
        "device_count": int(torch.musa.device_count()),
        "frontend_flavor": _FRONTEND["flavor"],
        "frontend_module_file": _FRONTEND.get("module", "?"),
        "mudnn_frontend_commit": _git_value(frontend_root, "rev-parse", "HEAD"),
        "mudnn_frontend_branch": _git_value(
            frontend_root, "rev-parse", "--abbrev-ref", "HEAD"
        ),
    }
    for name in ("torch_musa", "tilelang_musa", "tilelang"):
        try:
            module = __import__(name)
            meta[name] = getattr(module, "__version__", "imported")
        except Exception as exc:  # noqa: BLE001
            meta[name] = f"unavailable: {type(exc).__name__}"
    try:
        import mudnn_frontend

        meta["mudnn_frontend"] = getattr(mudnn_frontend, "__version__", "?")
    except Exception as exc:  # noqa: BLE001
        meta["mudnn_frontend"] = f"unavailable: {type(exc).__name__}"
    import os

    meta["musa_home"] = os.environ.get("MUSA_HOME", "")
    try:
        if _FRONTEND["flavor"] == "mudnn":
            import mudnn

            meta["backend_version"] = str(
                mudnn.backend_version_string() if callable(mudnn.backend_version_string)
                else mudnn.backend_version_string
            )
    except Exception:  # noqa: BLE001
        pass
    return meta


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def _sync():
    torch.musa.synchronize()


def measure(fn, warmup: int, iters: int):
    """Return ``(mean_us, min_us, max_us, stdev_us)`` for ``iters`` calls of ``fn``.

    A warm-up pass runs first so JIT compilation, plan caching and allocator
    growth stay out of the measurement; the first timed call is then discarded
    as well, which matters for the wrappers that build a descriptor cache keyed
    by tensor metadata.
    """
    for _ in range(warmup):
        fn()
    _sync()
    fn()  # discard: first call may still populate a plan cache
    _sync()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    mean = sum(samples) / len(samples)
    mean_sq = sum(value * value for value in samples) / len(samples)
    variance = max(mean_sq - mean * mean, 0.0)
    return mean, samples[0], samples[-1], variance**0.5


# ---------------------------------------------------------------------------
# input builders (one per case, all on-device bf16/fp32 as the API wants)
# ---------------------------------------------------------------------------


def _randn(shape, scale=0.1, dtype=torch.bfloat16, device=DEVICE, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, device=device).mul(scale).to(dtype)


def _topk_indices(s_q, s_k, topk, device=DEVICE, seed=0):
    """Descending-by-value top-k ids, packed to the front and padded with -1."""
    torch.manual_seed(seed)
    width = min(topk, s_k)
    idx = torch.argsort(
        torch.rand((s_q, s_k), device=device), dim=-1, descending=True
    )[:, :width].to(torch.int32)
    if width < topk:
        idx = torch.nn.functional.pad(idx, (0, topk - width), value=-1)
    return idx


class Case:
    """One benchmark case: a builder returning the callable plus its metadata.

    ``traffic`` is a zero-argument callable returning ``(bytes, flops)`` for one
    call -- the *logical* DRAM traffic and useful arithmetic of the public
    operation, counted from its declared inputs and outputs (see the
    ``_traffic_*`` helpers below).  ``None`` means the case has no device work
    worth a roofline number (a pure host computation).
    """

    def __init__(self, group, name, api, dims, build, traffic=None):
        self.group = group
        self.name = name
        self.api = api
        self.dims = dims
        self.build = build
        self.traffic = traffic


# ---------------------------------------------------------------------------
# roofline model: logical bytes and useful FLOPs per call
# ---------------------------------------------------------------------------
#
# These are *logical* counts: every input is read once and every output written
# once, exactly the traffic an ideal fused kernel needs.  A reference
# implementation that materialises an intermediate (the dense score tile, the
# gathered K) moves strictly more, which is one of the reasons it is slower;
# the achieved-vs-peak percentage is therefore an upper bound on the
# implementation's efficiency, not a measurement of its real traffic.  Where an
# operator really does re-read a tensor per row (the Top-K gathers) that traffic
# is counted, because no fusion can remove it -- those are the bandwidth-bound
# cases and the percentage is meaningful for them.

_BF16 = 2
_FP32 = 4
_I32 = 4


def _t_q_idx():
    """Indexer query ``(1, SEQ, HEADS, D_IDX)`` BF16."""
    return SEQ * HEADS * D_IDX * _BF16


def _t_k_idx(s_k, heads=1):
    """Indexer key ``(1, s_k, heads, D_IDX)`` BF16."""
    return s_k * heads * D_IDX * _BF16


def _t_w():
    """Indexer weights ``(1, SEQ, HEADS)`` BF16."""
    return SEQ * HEADS * _BF16


def _t_topk_out(rows, dtype=_I32):
    """One ``(rows, TOPK)`` output; FP32 logits/softmax and INT32 ids."""
    return rows * TOPK * dtype


def _traffic_indexer_forward(s_k):
    """Dense indexer scores: one QK GEMM, relu, head-weighted reduce, mask."""

    def spec():
        reads = _t_q_idx() + _t_k_idx(s_k) + _t_w()
        writes = SEQ * s_k * _FP32  # scores
        # QK^T: 2 flops/MAC; then relu + weighted head sum over HEADS.
        flops = 2 * SEQ * s_k * HEADS * D_IDX + 3 * SEQ * s_k * HEADS
        return reads + writes, flops

    return spec


def _traffic_indexer_topk(s_k):
    """Fused score + Top-K: the GEMM plus a selection that writes ids/logits."""

    def spec():
        reads = _t_q_idx() + _t_k_idx(s_k) + _t_w()
        writes = _t_topk_out(SEQ) * 3  # indices + logits + fused softmax
        flops = 2 * SEQ * s_k * HEADS * D_IDX + 3 * SEQ * s_k * HEADS
        return reads + writes, flops

    return spec


def _traffic_indexer_top_k(s_k):
    """Standalone Top-K over an existing ``(SEQ, s_k)`` FP32 score matrix."""

    def spec():
        reads = SEQ * s_k * _FP32
        writes = _t_topk_out(SEQ) * 2  # indices + values
        return reads + writes, 0

    return spec


def _traffic_compactify():
    """Stable partition of ``(1, SEQ, TOPK)`` ids plus the per-row lengths."""

    def spec():
        return 2 * _t_topk_out(SEQ) + SEQ * _I32, 0

    return spec


def _traffic_local_to_global():
    def spec():
        return 2 * _t_topk_out(SEQ), SEQ * TOPK

    return spec


def _traffic_sparse_indexer_score(s_k=SEQ // RATIO_EVEN):
    """Predict distribution over the selected Top-K slots.

    K is only ``s_k`` rows, so like the attention fold its DRAM traffic is one
    pass over that tile rather than one per gather: the gathered rows are re-read
    across queries and served from cache, and counting every gather would push
    the roofline percentage past the measured peak.
    """

    def spec():
        reads = _t_q_idx() + min(SEQ * TOPK, s_k) * D_IDX * _BF16 + _t_w()
        writes = _t_topk_out(SEQ, _FP32)
        flops = 2 * SEQ * TOPK * HEADS * D_IDX + 3 * SEQ * TOPK * HEADS + 5 * SEQ * TOPK
        return reads + writes, flops

    return spec


def _traffic_sparse_attn_score(s_k=SEQ // RATIO_EVEN):
    """``exp(qk*scale - lse)`` accumulated over heads for the Top-K slots."""

    def spec():
        reads = (
            _t_q_idx()
            + min(SEQ * TOPK, s_k) * D_IDX * _BF16  # see the forward note
            + SEQ * HEADS * _FP32  # lse
        )
        writes = _t_topk_out(SEQ, _FP32)
        flops = 2 * SEQ * TOPK * HEADS * D_IDX + 5 * SEQ * TOPK * HEADS
        return reads + writes, flops

    return spec


def _traffic_dense_indexer_score(s_k):
    """Dense indexer scores plus their per-row log-sum-exp denom."""

    def spec():
        reads = _t_q_idx() + _t_k_idx(s_k) + _t_w()
        writes = SEQ * s_k * _FP32 + SEQ * _FP32
        flops = 2 * SEQ * s_k * HEADS * D_IDX + 3 * SEQ * s_k * HEADS
        return reads + writes, flops

    return spec


def _traffic_dense_attn_score(s_k, d_attn):
    def spec():
        reads = (
            SEQ * HEADS * d_attn * _BF16
            + s_k * d_attn * _BF16
            + SEQ * HEADS * _FP32
        )
        writes = SEQ * s_k * _FP32 + SEQ * _FP32
        flops = 2 * SEQ * s_k * HEADS * d_attn + 5 * SEQ * s_k * HEADS
        return reads + writes, flops

    return spec


def _traffic_sparse_indexer_backward(s_k):
    """Sparse indexer backward: score recompute + dQ, dW and dK passes.

    Three GEMM-equivalent passes over the gathered Top-K slots: the forward
    recompute of the scores, then one reduction each for ``dQ`` and ``dK``.
    """

    def spec():
        reads = (
            _t_q_idx()
            + min(SEQ * TOPK, s_k) * D_IDX * _BF16  # one pass over the K tile
            + _t_w()
            + _t_topk_out(SEQ, _FP32) * 2  # attn_score + index_score
        )
        writes = _t_q_idx() + _t_w() + _t_k_idx(s_k, 1)
        flops = 3 * 2 * SEQ * TOPK * HEADS * D_IDX + 3 * SEQ * TOPK * HEADS
        return reads + writes, flops

    return spec


def _traffic_dense_indexer_backward(s_k):
    def spec():
        reads = (
            _t_q_idx()
            + _t_k_idx(s_k)
            + _t_w()
            + SEQ * s_k * _FP32 * 2  # attn_score + index_score
        )
        writes = _t_q_idx() + _t_w() + _t_k_idx(s_k, 1)
        flops = 3 * 2 * SEQ * s_k * HEADS * D_IDX + 3 * SEQ * s_k * HEADS
        return reads + writes, flops

    return spec


def _traffic_sparse_attention_forward(kv_rows, topk):
    """Sparse attention forward: QK^T over the selected slots, softmax, PV.

    The KV gather is `SEQ * topk` rows of `D_V`; it is counted as real traffic
    (a gathered row is re-read per query row even when the cache absorbs some
    of it, so this is the upper bound an infinitely-caching kernel would hit).
    """

    def spec():
        # K/V is only ``kv_rows`` rows, and every selected row is re-read many
        # times across queries, so its DRAM traffic is bounded by one pass over
        # the cache -- not by ``SEQ * topk`` gathers.  Counting the gathers makes
        # the model exceed the measured peak (104% at kv5120/topk640), which is
        # the tell that it is an L2-bound figure rather than DRAM traffic.  The
        # unavoidable reading is therefore ``min(gathers, distinct rows)``.
        kv_dram_rows = min(SEQ * topk, kv_rows)
        reads = (
            SEQ * HEADS * D_V * _BF16
            + kv_dram_rows * D_V * _BF16
            + SEQ * topk * _I32
            + HEADS * _FP32  # attn_sink
        )
        # out (BF16), plus max_logits and lse (FP32 per head)
        writes = SEQ * HEADS * D_V * _BF16 + SEQ * HEADS * _FP32 * 2
        flops = 4 * SEQ * topk * HEADS * D_V + 5 * SEQ * topk * HEADS
        return reads + writes, flops

    return spec


def _traffic_sparse_attention_backward(kv_rows, topk):
    """Sparse attention backward: recompute + dP/dQ and dKV, ~2.5x the forward."""

    def spec():
        reads = (
            SEQ * HEADS * D_V * _BF16 * 2  # q + dout
            + min(SEQ * topk, kv_rows) * D_V * _BF16  # see the forward note
            + SEQ * HEADS * _FP32 * 2  # lse + out
            + SEQ * topk * _I32
        )
        writes = SEQ * HEADS * D_V * _BF16 + kv_rows * D_V * _FP32
        flops = int(2.5 * (4 * SEQ * topk * HEADS * D_V + 5 * SEQ * topk * HEADS))
        return reads + writes, flops

    return spec


def _traffic_csa(ratio, coff, head_dim, backward):
    """CSA compressor.

    Follows ``doc/csa_compressor_performance_comparison.md``: the forward moves
    ``4TW + 2ND`` bytes (BF16 kv/score in, BF16 out) and the backward ``8TW``
    (BF16 kv/score in, BF16 dKV/dScore out), with ``T`` tokens, ``W = coff*D``
    and ``N`` output rows.  Arithmetic is the gather add, the ratio-wide softmax
    and the gated sum: ~6 flops per (token, channel).
    """

    def spec():
        w = coff * head_dim
        rows = SEQ // ratio
        if backward:
            return 8 * SEQ * w, 6 * SEQ * w
        return 4 * SEQ * w + 2 * rows * head_dim, 6 * SEQ * w

    return spec


def _cases_added():
    DSA = _dsa()

    def build_compact(s_k, ratio, topk):
        q = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=11)
        k = _randn((1, s_k, 1, D_IDX), scale=0.05, seed=12)
        w = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.bfloat16) * 0.1
        kw = dict(top_k=topk, ratio=ratio, topk_indices_global=False)

        def run():
            return DSA.indexer_forward_top_k_wrapper(q, k, w, **kw)

        return run

    def build_size_bshd():
        return lambda: DSA.compress_topk_cand_buffer_size(
            1, SEQ, SEQ // RATIO_EVEN, RATIO_EVEN, microbatch_rows=0
        )

    def build_size_thd():
        cu_q = torch.tensor([0, SEQ], dtype=torch.int32, device=DEVICE)
        cu_k = torch.tensor([0, SEQ // RATIO_EVEN], dtype=torch.int32, device=DEVICE)
        return lambda: DSA.compress_topk_cand_buffer_size_thd(cu_q, cu_k, RATIO_EVEN)

    return [
        Case(
            "added",
            f"indexer_forward_top_k r{RATIO_EVEN} S_k={SEQ // RATIO_EVEN}",
            "DSA.indexer_forward_top_k_wrapper",
            f"q[1,{SEQ},{HEADS},{D_IDX}] k[1,{SEQ // RATIO_EVEN},1,{D_IDX}] "
            f"w[1,{SEQ},{HEADS}] top_k={TOPK} ratio={RATIO_EVEN}",
            lambda: build_compact(SEQ // RATIO_EVEN, RATIO_EVEN, TOPK),
            traffic=_traffic_indexer_topk(SEQ // RATIO_EVEN),
        ),
        Case(
            "added",
            f"indexer_forward_top_k r{RATIO_ODD} S_k={SEQ // RATIO_ODD}",
            "DSA.indexer_forward_top_k_wrapper",
            f"q[1,{SEQ},{HEADS},{D_IDX}] k[1,{SEQ // RATIO_ODD},1,{D_IDX}] "
            f"w[1,{SEQ},{HEADS}] top_k={TOPK} ratio={RATIO_ODD}",
            lambda: build_compact(SEQ // RATIO_ODD, RATIO_ODD, TOPK),
            traffic=_traffic_indexer_topk(SEQ // RATIO_ODD),
        ),
        Case(
            "added",
            "compress_topk_cand_buffer_size",
            "DSA.compress_topk_cand_buffer_size",
            f"bs=1 seqlen_q={SEQ} seqlen_k={SEQ // RATIO_EVEN} ratio={RATIO_EVEN} "
            "microbatch_rows=0",
            lambda: build_size_bshd(),
            traffic=None,  # pure host arithmetic, no device work
        ),
        Case(
            "added",
            "compress_topk_cand_buffer_size_thd",
            "DSA.compress_topk_cand_buffer_size_thd",
            f"cu_seqlens_q=[0,{SEQ}] cu_seqlens_k=[0,{SEQ // RATIO_EVEN}] "
            f"ratio={RATIO_EVEN}",
            lambda: build_size_thd(),
            traffic=_traffic_thd_size(),
        ),
    ]


def _cases_dsa():
    DSA = _dsa()

    def build_indexer_forward(s_k, ratio):
        q = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=21)
        k = _randn((1, s_k, 1, D_IDX), scale=0.05, seed=22)
        w = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.bfloat16) * 0.1
        return lambda: DSA.indexer_forward_wrapper(q, k, w, ratio=ratio)

    def build_indexer_top_k(s_k):
        torch.manual_seed(31)
        scores = (
            torch.rand((1, SEQ, s_k), device=DEVICE, dtype=torch.float32) * 20.0
        ).reshape(SEQ, s_k)
        seq_lens = torch.full((1,), s_k, dtype=torch.int32, device=DEVICE)
        return lambda: DSA.indexer_top_k_wrapper(
            scores, seq_lens, top_k=min(TOPK, s_k), next_n=SEQ, return_val=True
        )

    def build_compactify():
        global_ids = _topk_indices(SEQ, SEQ // RATIO_EVEN, TOPK, seed=32)
        return lambda: DSA.compactify_wrapper(global_ids)

    def build_local_to_global():
        local = _topk_indices(SEQ, SEQ // RATIO_EVEN, TOPK, seed=33)
        return lambda: DSA.local_to_global_wrapper(
            local.unsqueeze(0), seqlen_k=SEQ // RATIO_EVEN
        )

    def build_sparse_indexer_score(s_k, ratio):
        q = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=34)
        k = _randn((1, s_k, D_IDX), scale=0.05, seed=35)
        w = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.bfloat16) * 0.1
        idx = _topk_indices(SEQ, s_k, TOPK, seed=36).unsqueeze(0)
        lengths = torch.full((1, SEQ), min(TOPK, s_k), dtype=torch.int32, device=DEVICE)
        return lambda: DSA.sparse_indexer_score_recompute_wrapper(
            q, k, w, idx, topk_length=lengths
        )

    def build_sparse_attn_score(s_k):
        q = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=37)
        k = _randn((1, s_k, D_IDX), scale=0.05, seed=38)
        lse = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.float32) + 1.0
        idx = _topk_indices(SEQ, s_k, TOPK, seed=39).unsqueeze(0)
        lengths = torch.full((1, SEQ), min(TOPK, s_k), dtype=torch.int32, device=DEVICE)
        return lambda: DSA.sparse_attn_score_recompute_wrapper(
            q, k, lse, idx, D_IDX**-0.5, topk_length=lengths
        )

    def build_dense_indexer_score(s_k, ratio):
        q = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=41)
        k = _randn((1, s_k, 1, D_IDX), scale=0.05, seed=42)
        w = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.bfloat16) * 0.1
        return lambda: DSA.dense_indexer_score_recompute_wrapper(
            q, k, w, ratio=ratio
        )

    def build_dense_attn_score(s_k, ratio):
        q = _randn((1, SEQ, HEADS, D_ATTN), scale=0.05, seed=43)
        k = _randn((1, s_k, 1, D_ATTN), scale=0.05, seed=44)
        lse = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.float32) + 1.0
        return lambda: DSA.dense_attn_score_recompute_wrapper(
            q, k, lse, D_ATTN**-0.5, ratio=ratio
        )

    def build_indexer_backward():
        q = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=51)
        k = _randn((1, SEQ // RATIO_EVEN, D_IDX), scale=0.05, seed=52)
        w = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.bfloat16) * 0.1
        idx = _topk_indices(SEQ, SEQ // RATIO_EVEN, TOPK, seed=53).unsqueeze(0)
        attn = torch.rand((1, SEQ, TOPK), device=DEVICE, dtype=torch.float32)
        index_score = torch.rand((1, SEQ, TOPK), device=DEVICE, dtype=torch.float32)
        scale = D_IDX**-0.5
        return lambda: DSA.indexer_backward_wrapper(
            q, w, k, attn, index_score, idx, sm_scale=scale, loss_coeff=float(SEQ)
        )

    def build_dense_indexer_backward():
        q = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=61)
        k = _randn((1, SEQ // RATIO_EVEN, D_IDX), scale=0.05, seed=62)
        w = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.bfloat16) * 0.1
        attn_score = torch.rand(
            (1, SEQ, SEQ // RATIO_EVEN), device=DEVICE, dtype=torch.float32
        )
        attn_l1norm = attn_score.sum(-1)
        index_score = torch.rand(
            (1, SEQ, SEQ // RATIO_EVEN), device=DEVICE, dtype=torch.float32
        )
        index_lse = torch.zeros((1, SEQ), device=DEVICE, dtype=torch.float32)
        scale = D_IDX**-0.5
        return lambda: DSA.dense_indexer_backward_wrapper(
            q,
            w,
            k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            sm_scale=scale,
            loss_coeff=float(SEQ),
            ratio=RATIO_EVEN,
            block_I=64,
        )

    return [
        Case(
            "dsa",
            f"indexer_forward r{RATIO_EVEN} S_k={SEQ // RATIO_EVEN}",
            "DSA.indexer_forward_wrapper",
            f"q[1,{SEQ},{HEADS},{D_IDX}] k[1,{SEQ // RATIO_EVEN},1,{D_IDX}] ratio={RATIO_EVEN}",
            lambda: build_indexer_forward(SEQ // RATIO_EVEN, RATIO_EVEN),
            traffic=_traffic_indexer_forward(SEQ // RATIO_EVEN),
        ),
        Case(
            "dsa",
            f"indexer_forward r{RATIO_ODD} S_k={SEQ // RATIO_ODD}",
            "DSA.indexer_forward_wrapper",
            f"q[1,{SEQ},{HEADS},{D_IDX}] k[1,{SEQ // RATIO_ODD},1,{D_IDX}] ratio={RATIO_ODD}",
            lambda: build_indexer_forward(SEQ // RATIO_ODD, RATIO_ODD),
            traffic=_traffic_indexer_forward(SEQ // RATIO_ODD),
        ),
        Case(
            "dsa",
            f"indexer_top_k S_k={SEQ // RATIO_EVEN} topk={TOPK}",
            "DSA.indexer_top_k_wrapper",
            f"scores[{SEQ},{SEQ // RATIO_EVEN}] top_k={TOPK} next_n={SEQ}",
            lambda: build_indexer_top_k(SEQ // RATIO_EVEN),
            traffic=_traffic_indexer_top_k(SEQ // RATIO_EVEN),
        ),
        Case(
            "dsa",
            f"compactify topk={TOPK}",
            "DSA.compactify_wrapper",
            f"indices[1,{SEQ},{TOPK}]",
            lambda: build_compactify(),
            traffic=_traffic_compactify(),
        ),
        Case(
            "dsa",
            f"local_to_global topk={TOPK}",
            "DSA.local_to_global_wrapper",
            f"local_indices[1,{SEQ},{TOPK}] seqlen_k={SEQ // RATIO_EVEN}",
            lambda: build_local_to_global(),
            traffic=_traffic_local_to_global(),
        ),
        Case(
            "dsa",
            f"sparse_indexer_score r{RATIO_EVEN}",
            "DSA.sparse_indexer_score_recompute_wrapper",
            f"q[1,{SEQ},{HEADS},{D_IDX}] k[1,{SEQ // RATIO_EVEN},{D_IDX}] topk={TOPK}",
            lambda: build_sparse_indexer_score(SEQ // RATIO_EVEN, RATIO_EVEN),
            traffic=_traffic_sparse_indexer_score(),
        ),
        Case(
            "dsa",
            f"sparse_attn_score S_k={SEQ // RATIO_EVEN}",
            "DSA.sparse_attn_score_recompute_wrapper",
            f"q[1,{SEQ},{HEADS},{D_IDX}] k[1,{SEQ // RATIO_EVEN},{D_IDX}] topk={TOPK}",
            lambda: build_sparse_attn_score(SEQ // RATIO_EVEN),
            traffic=_traffic_sparse_attn_score(),
        ),
        Case(
            "dsa",
            f"dense_indexer_score r{RATIO_EVEN}",
            "DSA.dense_indexer_score_recompute_wrapper",
            f"q[1,{SEQ},{HEADS},{D_IDX}] k[1,{SEQ // RATIO_EVEN},1,{D_IDX}] ratio={RATIO_EVEN}",
            lambda: build_dense_indexer_score(SEQ // RATIO_EVEN, RATIO_EVEN),
            traffic=_traffic_dense_indexer_score(SEQ // RATIO_EVEN),
        ),
        Case(
            "dsa",
            f"dense_attn_score r{RATIO_EVEN} D={D_ATTN}",
            "DSA.dense_attn_score_recompute_wrapper",
            f"q[1,{SEQ},{HEADS},{D_ATTN}] k[1,{SEQ // RATIO_EVEN},1,{D_ATTN}] ratio={RATIO_EVEN}",
            lambda: build_dense_attn_score(SEQ // RATIO_EVEN, RATIO_EVEN),
            traffic=_traffic_dense_attn_score(SEQ // RATIO_EVEN, D_ATTN),
        ),
        Case(
            "dsa",
            f"indexer_backward topk={TOPK}",
            "DSA.indexer_backward_wrapper",
            f"q[1,{SEQ},{HEADS},{D_IDX}] k[1,{SEQ // RATIO_EVEN},{D_IDX}] topk={TOPK}",
            lambda: build_indexer_backward(),
            traffic=_traffic_sparse_indexer_backward(SEQ // RATIO_EVEN),
        ),
        Case(
            "dsa",
            f"dense_indexer_backward r{RATIO_EVEN}",
            "DSA.dense_indexer_backward_wrapper",
            f"q[1,{SEQ},{HEADS},{D_IDX}] k[1,{SEQ // RATIO_EVEN},{D_IDX}] ratio={RATIO_EVEN}",
            lambda: build_dense_indexer_backward(),
            traffic=_traffic_dense_indexer_backward(SEQ // RATIO_EVEN),
        ),
    ]


def _cases_attn():
    """The two entry points behind the spreadsheet's CSASparseAttnFunc rows."""
    DSA = _dsa()

    def build_forward(kv_rows, topk):
        q = _randn((SEQ, HEADS, D_V), scale=0.05, seed=71)
        kv = _randn((kv_rows, D_V), scale=0.05, seed=72)
        idx = torch.randint(
            0, kv_rows, (SEQ, topk), dtype=torch.int32, device=DEVICE, generator=None
        )
        sink = torch.linspace(-1, 1, HEADS, dtype=torch.float32, device=DEVICE)
        lengths = torch.full((SEQ,), topk, dtype=torch.int32, device=DEVICE)
        return lambda: DSA.sparse_attention_forward_wrapper(
            q, kv, idx, attn_sink=sink, topk_length=lengths
        )

    def build_backward(kv_rows, topk):
        q = _randn((SEQ, HEADS, D_V), scale=0.05, seed=73)
        kv = _randn((kv_rows, D_V), scale=0.05, seed=74)
        out = _randn((SEQ, HEADS, D_V), scale=0.05, seed=75)
        dout = _randn((SEQ, HEADS, D_V), scale=0.05, seed=76)
        lse = torch.rand((SEQ, HEADS), device=DEVICE, dtype=torch.float32) + 1.0
        sink = torch.linspace(-1, 1, HEADS, dtype=torch.float32, device=DEVICE)
        idx = torch.randint(0, kv_rows, (SEQ, topk), dtype=torch.int32, device=DEVICE)
        lengths = torch.full((SEQ,), topk, dtype=torch.int32, device=DEVICE)
        scale = D_V**-0.5
        return lambda: DSA.sparse_attention_backward_wrapper(
            q,
            kv,
            out,
            dout,
            lse,
            sink,
            idx,
            softmax_scale=scale,
            topk_length=lengths,
        )

    return [
        Case(
            "attn",
            "sparse_attention_forward kv5120 topk640",
            "DSA.sparse_attention_forward_wrapper",
            f"q[{SEQ},{HEADS},{D_V}] kv[5120,{D_V}] topk=640",
            lambda: build_forward(5120, 640),
            traffic=_traffic_sparse_attention_forward(5120, 640),
        ),
        Case(
            "attn",
            "sparse_attention_forward kv4128 topk160",
            "DSA.sparse_attention_forward_wrapper",
            f"q[{SEQ},{HEADS},{D_V}] kv[4128,{D_V}] topk=160",
            lambda: build_forward(4128, 160),
            traffic=_traffic_sparse_attention_forward(4128, 160),
        ),
        Case(
            "attn",
            "sparse_attention_forward kv4096 topk128",
            "DSA.sparse_attention_forward_wrapper",
            f"q[{SEQ},{HEADS},{D_V}] kv[4096,{D_V}] topk=128",
            lambda: build_forward(4096, 128),
            traffic=_traffic_sparse_attention_forward(4096, 128),
        ),
        Case(
            "attn",
            "sparse_attention_backward kv4096 topk128",
            "DSA.sparse_attention_backward_wrapper",
            f"q[{SEQ},{HEADS},{D_V}] kv[4096,{D_V}] dout[{SEQ},{HEADS},{D_V}] topk=128",
            lambda: build_backward(4096, 128),
            traffic=_traffic_sparse_attention_backward(4096, 128),
        ),
    ]


def _cases_csa():
    if _FRONTEND["ns"] is None:
        select_frontend("mudnn")
    csa_compressor_forward_wrapper = _FRONTEND["csa_fwd"]
    csa_compressor_backward_wrapper = _FRONTEND["csa_bwd"]

    def build(ratio, coff, head_dim):
        section = max(SEQ // ratio, 1)
        cu = torch.tensor([0, SEQ], dtype=torch.int32, device=DEVICE)
        comp = torch.tensor(
            [0, section], dtype=torch.int32, device=DEVICE
        )
        kv = _randn((SEQ, coff * head_dim), scale=0.05, seed=81)
        score = _randn((SEQ, coff * head_dim), scale=0.05, seed=82)
        ape = (
            torch.randn((ratio, coff * head_dim), dtype=torch.float32, device=DEVICE)
            * 0.01
        )
        out = _randn((section, head_dim), scale=0.05, seed=83)
        return ratio, coff, head_dim, cu, comp, kv, score, ape, out

    def fwd(ratio, coff, head_dim):
        _, _, _, cu, comp, kv, score, ape, _ = build(ratio, coff, head_dim)
        return lambda: csa_compressor_forward_wrapper(
            kv, score, ape, cu, comp, ratio=ratio, head_dim=head_dim, coff=coff
        )

    def bwd(ratio, coff, head_dim):
        _, _, _, cu, comp, kv, score, ape, out = build(ratio, coff, head_dim)
        return lambda: csa_compressor_backward_wrapper(
            kv,
            score,
            ape,
            cu,
            comp,
            out,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
        )

    return [
        Case(
            "csa",
            f"csa_compressor_forward r{RATIO_EVEN} coff2 d512",
            "CSA.csa_compressor_forward_wrapper",
            f"kv[{SEQ},{2 * 512}] score[{SEQ},{2 * 512}] ape[{RATIO_EVEN},{2 * 512}]",
            lambda: fwd(RATIO_EVEN, 2, 512),
            traffic=_traffic_csa(RATIO_EVEN, 2, 512, False),
        ),
        Case(
            "csa",
            f"csa_compressor_backward r{RATIO_EVEN} coff2 d512",
            "CSA.csa_compressor_backward_wrapper",
            f"kv[{SEQ},{2 * 512}] score[{SEQ},{2 * 512}] ape[{RATIO_EVEN},{2 * 512}]",
            lambda: bwd(RATIO_EVEN, 2, 512),
            traffic=_traffic_csa(RATIO_EVEN, 2, 512, True),
        ),
        Case(
            "csa",
            f"csa_compressor_forward r{RATIO_ODD} coff1 d512",
            "CSA.csa_compressor_forward_wrapper",
            f"kv[{SEQ},512] score[{SEQ},512] ape[{RATIO_ODD},512]",
            lambda: fwd(RATIO_ODD, 1, 512),
            traffic=_traffic_csa(RATIO_ODD, 1, 512, False),
        ),
        Case(
            "csa",
            f"csa_compressor_backward r{RATIO_ODD} coff1 d512",
            "CSA.csa_compressor_backward_wrapper",
            f"kv[{SEQ},512] score[{SEQ},512] ape[{RATIO_ODD},512]",
            lambda: bwd(RATIO_ODD, 1, 512),
            traffic=_traffic_csa(RATIO_ODD, 1, 512, True),
        ),
    ]


def _cases_fused():
    """Compose the three excel rows that have no MP31 column.

    The reference spreadsheet's ``CheckpointFunction`` / ``FusedCSAIndexerSparseAttnFunc``
    / ``FusedCSAIndexerSparseAttnFuncBackward`` rows are Megatron's fused
    autograd functions; those live in ``megatron-lm-musa-patch`` and are not
    importable outside the training script.  What *is* reachable is the same
    work expressed through the public frontend calls the fusion makes, so each
    case below chains them at the training shape (``S_q=4096``, ``H=64``,
    ``D_idx=128``, ``D_v=512``, ``indexer_topk=512``, ``ratio`` 4 / 128) and
    times the whole chain.  That is the honest upper bound: a real fusion
    removes intermediate tensors, so a fused kernel can only be faster.

    The names carry the excel row so the numbers can be dropped into it.
    """
    DSA = _dsa()

    def build_fwd_indexer_topk(ratio, s_k):
        q_idx = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=91)
        k_idx = _randn((1, s_k, 1, D_IDX), scale=0.05, seed=92)
        w = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.bfloat16) * 0.1
        q_a = _randn((SEQ, HEADS, D_V), scale=0.05, seed=93)
        kv = _randn((s_k * 4, D_V), scale=0.05, seed=94) if s_k * 4 > 512 else _randn(
            (SEQ, D_V), scale=0.05, seed=94
        )
        sink = torch.linspace(-1, 1, HEADS, dtype=torch.float32, device=DEVICE)
        kv_rows = int(kv.shape[0])

        def run():
            topk = DSA.indexer_forward_top_k_wrapper(
                q_idx, k_idx, w, top_k=TOPK, ratio=ratio, topk_indices_global=False
            )
            packed = DSA.compactify_wrapper(topk["indices"])
            idx = packed["indices"][0].to(torch.int32)
            lengths = packed["topk_length"]
            return DSA.sparse_attention_forward_wrapper(
                q_a, kv, idx, attn_sink=sink, topk_length=lengths
            )

        return run

    def build_bwd_indexer_topk(ratio, s_k):
        q_idx = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=95)
        k_idx = _randn((1, s_k, 1, D_IDX), scale=0.05, seed=96)
        w = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.bfloat16) * 0.1
        q_a = _randn((SEQ, HEADS, D_V), scale=0.05, seed=97)
        kv = _randn((max(s_k * 4, 512), D_V), scale=0.05, seed=98)
        out = _randn((SEQ, HEADS, D_V), scale=0.05, seed=99)
        dout = _randn((SEQ, HEADS, D_V), scale=0.05, seed=100)
        lse = torch.rand((SEQ, HEADS), device=DEVICE, dtype=torch.float32) + 1.0
        sink = torch.linspace(-1, 1, HEADS, dtype=torch.float32, device=DEVICE)
        attn = torch.rand((1, SEQ, TOPK), device=DEVICE, dtype=torch.float32)
        index_score = torch.rand((1, SEQ, TOPK), device=DEVICE, dtype=torch.float32)

        # Resolve the non-differentiable front-end once: the TopK selection is
        # not part of the backward graph, so timing it again would be wrong.
        with torch.no_grad():
            topk = DSA.indexer_forward_top_k_wrapper(
                q_idx, k_idx, w, top_k=TOPK, ratio=ratio, topk_indices_global=False
            )
            packed = DSA.compactify_wrapper(topk["indices"])
            idx = packed["indices"][0].to(torch.int32)
            lengths = packed["topk_length"]
        scale = D_IDX**-0.5

        def run():
            DSA.sparse_attention_backward_wrapper(
                q_a,
                kv,
                out,
                dout,
                lse,
                sink,
                idx,
                softmax_scale=D_V**-0.5,
                topk_length=lengths,
            )
            return DSA.indexer_backward_wrapper(
                q_idx,
                w,
                k_idx.squeeze(2),
                attn,
                index_score,
                idx.unsqueeze(0),
                sm_scale=scale,
                loss_coeff=float(SEQ),
            )

        return run

    def build_checkpoint_compact(ratio, s_k):
        """Excel row 8: ``CheckpointFunction`` packing the TopK ids.

        Its kernels are ``dsa_compactify_kernel`` + ``dsa_indexer_topk_normalize_kernel``
        + the indexer ``main_kernel``, i.e. compact the selected ids and normalise
        the logits -- exactly ``indexer_forward_top_k_wrapper`` followed by
        ``compactify_wrapper``.
        """
        q_idx = _randn((1, SEQ, HEADS, D_IDX), scale=0.05, seed=101)
        k_idx = _randn((1, s_k, 1, D_IDX), scale=0.05, seed=102)
        w = torch.rand((1, SEQ, HEADS), device=DEVICE, dtype=torch.bfloat16) * 0.1

        def run():
            topk = DSA.indexer_forward_top_k_wrapper(
                q_idx, k_idx, w, top_k=TOPK, ratio=ratio, topk_indices_global=False
            )
            return DSA.compactify_wrapper(topk["indices"])

        return run

    cases = [
        Case(
            "fused",
            f"[row17] fused indexer+sparseattn FWD r{RATIO_EVEN}",
            "indexer_forward_top_k + compactify + sparse_attention_forward",
            f"S_q={SEQ} H={HEADS} D_idx={D_IDX} D_v={D_V} topk={TOPK} ratio={RATIO_EVEN}",
            lambda: build_fwd_indexer_topk(RATIO_EVEN, SEQ // RATIO_EVEN),
            traffic=_traffic_fused_fwd(RATIO_EVEN),
        ),
        Case(
            "fused",
            f"[row17] fused indexer+sparseattn FWD r{RATIO_ODD}",
            "indexer_forward_top_k + compactify + sparse_attention_forward",
            f"S_q={SEQ} H={HEADS} D_idx={D_IDX} D_v={D_V} topk={TOPK} ratio={RATIO_ODD}",
            lambda: build_fwd_indexer_topk(RATIO_ODD, SEQ // RATIO_ODD),
            traffic=_traffic_fused_fwd(RATIO_ODD),
        ),
        Case(
            "fused",
            f"[row9] fused indexer+sparseattn BWD r{RATIO_EVEN}",
            "sparse_attention_backward + indexer_backward",
            f"S_q={SEQ} H={HEADS} D_idx={D_IDX} D_v={D_V} topk={TOPK} ratio={RATIO_EVEN}",
            lambda: build_bwd_indexer_topk(RATIO_EVEN, SEQ // RATIO_EVEN),
            traffic=_traffic_fused_bwd(RATIO_EVEN),
        ),
        Case(
            "fused",
            f"[row8] Checkpoint compact TopK r{RATIO_EVEN}",
            "indexer_forward_top_k + compactify",
            f"S_q={SEQ} H={HEADS} D_idx={D_IDX} topk={TOPK} ratio={RATIO_EVEN}",
            lambda: build_checkpoint_compact(RATIO_EVEN, SEQ // RATIO_EVEN),
            traffic=_traffic_checkpoint_compact(RATIO_EVEN),
        ),
    ]
    return cases


def _traffic_thd_size():
    """THD candidate sizing: read two ``(bs+1,)`` INT32 vectors, write INT64 offsets.

    The call also performs one scalar device-to-host read of the total; that is a
    synchronization, not traffic, so it is not counted.
    """

    def spec():
        return (3 * _I32 + 2 * 8) + SEQ * _I32, SEQ

    return spec


def _traffic_fused_fwd(ratio):
    """Excel row 17: fused indexer+TopK then sparse attention forward.

    Counted as the sum of the public calls the chain makes, including the
    compacted id/length tensors passed between them.  A real fusion eliminates
    those intermediates, so this over-counts -- which is why the fused row's
    percentage is a floor, not an efficiency claim.
    """
    s_k = SEQ // ratio

    def spec():
        a_b, a_f = _traffic_indexer_topk(s_k)()
        c_b, c_f = _traffic_compactify()()
        kv_rows = max(s_k * 4, 512)
        d_b, d_f = _traffic_sparse_attention_forward(kv_rows, TOPK)()
        return a_b + c_b + d_b, a_f + c_f + d_f

    return spec


def _traffic_fused_bwd(ratio):
    """Excel row 9: sparse attention backward plus indexer backward."""
    s_k = SEQ // ratio

    def spec():
        a_b, a_f = _traffic_sparse_attention_backward(max(s_k * 4, 512), TOPK)()
        b_b, b_f = _traffic_sparse_indexer_backward(s_k)()
        return a_b + b_b, a_f + b_f

    return spec


def _traffic_checkpoint_compact(ratio):
    """Excel row 8: the fused Top-K then the id compaction."""
    s_k = SEQ // ratio

    def spec():
        a_b, a_f = _traffic_indexer_topk(s_k)()
        c_b, c_f = _traffic_compactify()()
        return a_b + c_b, a_f + c_f

    return spec


# ---------------------------------------------------------------------------
# device peak (so achieved numbers can be expressed as a fraction of peak)
# ---------------------------------------------------------------------------


def measure_peak(args):
    """Return the device's sustained peaks, measured rather than quoted.

    Bandwidth is probed three ways because a plain device-to-device ``copy_``
    under-reports badly on this stack: on an MP32 device it sustains ~470 GB/s while a
    pure read reaches ~1320 GB/s and a pure write ~1920 GB/s, so using the copy
    number would make every operator look efficient.  ``peak_bw_gbs`` is the best
    of the three, which is the closest this box gets to its streaming limit, and
    the individual probes are kept in the report so the choice is auditable.

    Compute is a square BF16 matmul, counted as ``2 * n^3``.  It is the
    dense-GEMM ceiling; an operator that is not a pure GEMM cannot reach it.
    """
    peak = {}

    def _probe(name, fn, nbytes, iters):
        try:
            mean_us, _, _, _ = measure(fn, max(args.warmup, 2), max(iters, 5))
            peak[f"{name}_us"] = round(mean_us, 1)
            peak[f"{name}_gbs"] = round(nbytes / (mean_us * 1e3), 1)
        except Exception as exc:  # noqa: BLE001
            peak[f"{name}_gbs"] = None
            peak[f"{name}_error"] = f"{type(exc).__name__}: {exc}"

    try:
        n = 512 * 1024 * 1024 // 4  # 512 MiB of fp32
        buf = torch.empty(n, dtype=torch.float32, device=DEVICE)
        buf.fill_(1.0)
        dst = torch.empty_like(buf)
        torch.musa.synchronize()
        _probe("peak_bw_read", lambda: buf.sum(), n * 4, args.iters)
        _probe("peak_bw_write", lambda: buf.fill_(2.0), n * 4, args.iters)
        _probe("peak_bw_copy", lambda: dst.copy_(buf), 2 * n * 4, args.iters)
        del buf, dst
        torch.musa.empty_cache()
    except Exception as exc:  # noqa: BLE001
        peak["peak_bw_error"] = f"{type(exc).__name__}: {exc}"

    candidates = [
        peak.get("peak_bw_read_gbs"),
        peak.get("peak_bw_write_gbs"),
        peak.get("peak_bw_copy_gbs"),
    ]
    candidates = [value for value in candidates if isinstance(value, (int, float))]
    peak["peak_bw_gbs"] = max(candidates) if candidates else None
    peak["peak_bw_basis"] = (
        "max of read/write/copy probes; copy alone under-reports on this stack"
    )

    try:
        m = 8192
        a = torch.randn((m, m), dtype=torch.bfloat16, device=DEVICE)
        b = torch.randn((m, m), dtype=torch.bfloat16, device=DEVICE)

        def matmul():
            torch.mm(a, b)

        mean_us, _, _, _ = measure(matmul, max(args.warmup, 2), max(args.iters, 5))
        peak["peak_tflops"] = round((2 * m**3) / (mean_us * 1e6), 1)
        peak["peak_matmul_us"] = round(mean_us, 1)
        del a, b
        torch.musa.empty_cache()
    except Exception as exc:  # noqa: BLE001
        peak["peak_tflops"] = None
        peak["peak_tflops_error"] = f"{type(exc).__name__}: {exc}"
    return peak


GROUPS = {
    "added": _cases_added,
    "dsa": _cases_dsa,
    "attn": _cases_attn,
    "csa": _cases_csa,
    "fused": _cases_fused,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        default="all",
        choices=[*GROUPS, "all"],
        help="case group(s) to run; 'all' runs every group",
    )
    parser.add_argument("--device", type=int, default=0, help="device index")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--only", default="", help="substring filter on the case name")
    parser.add_argument("--out", default="", help="write the JSON report here")
    parser.add_argument(
        "--frontend-root",
        default="",
        help="checkout root recorded in the report metadata",
    )
    parser.add_argument(
        "--no-peak",
        action="store_true",
        help="skip the device peak measurement (reuse the stored peaks instead)",
    )
    parser.add_argument(
        "--frontend",
        default="mudnn",
        choices=["mudnn", "cudnn"],
        help="'mudnn' = this checkout's python/mudnn; 'cudnn' = an upstream "
        "cudnn-frontend checkout (needs --cudnn-frontend-path)",
    )
    parser.add_argument(
        "--cudnn-frontend-path",
        default="",
        help="PYTHONPATH root of an upstream cudnn-frontend checkout "
        "(the directory containing 'cudnn/')",
    )
    args = parser.parse_args(argv)

    torch.musa.set_device(args.device)
    device = torch.device(DEVICE, args.device)
    torch.musa.set_device(device)

    import os

    select_frontend(args.frontend, args.cudnn_frontend_path)
    frontend_root = args.frontend_root or os.environ.get("MUDNN_FRONTEND_ROOT", "")
    meta = _env_metadata(frontend_root)
    meta.update(
        {
            "device_index": args.device,
            "warmup": args.warmup,
            "iters": args.iters,
            "group": args.group,
            "timing": "wrapper-level mean over iters, launch overhead included",
        }
    )

    if not args.no_peak:
        print("measuring device peak ...", flush=True)
        peak = measure_peak(args)
        meta.update(peak)
        print(
            f"  peak: {peak.get('peak_bw_gbs')} GB/s (best of read/write/copy), "
            f"{peak.get('peak_tflops')} TFLOP/s bf16 8192 matmul"
        )

    groups = list(GROUPS) if args.group == "all" else [args.group]
    results = []
    for group in groups:
        try:
            cases = GROUPS[group]()
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "group": group,
                    "case": "<group import>",
                    "api": "-",
                    "status": f"error: {type(exc).__name__}: {exc}",
                }
            )
            continue
        for case in cases:
            if args.only and args.only not in case.name:
                continue
            record = {
                "group": case.group,
                "case": case.name,
                "api": case.api,
                "inputs": case.dims,
            }
            print(f"[{case.group}] {case.name} ...", end="", flush=True)
            try:
                run = case.build()
                run()
                torch.musa.synchronize()
                mean, low, high, stdev = measure(run, args.warmup, args.iters)
                record.update(
                    {
                        "status": "ok",
                        "mean_us": round(mean, 3),
                        "min_us": round(low, 3),
                        "max_us": round(high, 3),
                        "stdev_us": round(stdev, 3),
                    }
                )
                detail = ""
                if case.traffic is not None:
                    nbytes, flops = case.traffic()
                    gbs = nbytes / (mean * 1e3)  # bytes / (us * 1e3) = GB/s
                    tflops = flops / (mean * 1e6)  # flops / (us * 1e6) = TFLOP/s
                    record.update(
                        {
                            "bytes": int(nbytes),
                            "flops": int(flops),
                            "bw_gbs": round(gbs, 1),
                            "tflops": round(tflops, 2),
                        }
                    )
                    if meta.get("peak_bw_gbs"):
                        record["bw_pct_of_peak"] = round(
                            100.0 * gbs / meta["peak_bw_gbs"], 1
                        )
                    if meta.get("peak_tflops"):
                        record["tflops_pct_of_peak"] = round(
                            100.0 * tflops / meta["peak_tflops"], 1
                        )
                    detail = f"  {gbs:7.1f} GB/s {tflops:6.2f} TF/s"
                print(f" {mean:10.1f} us{detail}")
            except Exception as exc:  # noqa: BLE001
                record.update(
                    {
                        "status": f"error: {type(exc).__name__}: {exc}".strip(),
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
                print(f" SKIP ({type(exc).__name__}: {exc})")
            results.append(record)

    report = {"meta": meta, "results": results}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.out}")

    print("\n=== summary ===")
    print(
        f"{'group':6s} {'mean_us':>10s} {'GB/s':>9s} {'TF/s':>7s}  case"
    )
    for record in results:
        value = record.get("mean_us")
        shown = f"{value:10.1f}" if isinstance(value, (int, float)) else f"{'--':>10s}"
        bw = record.get("bw_gbs")
        tf = record.get("tflops")
        bw_s = f"{bw:9.1f}" if isinstance(bw, (int, float)) else f"{'--':>9s}"
        tf_s = f"{tf:7.2f}" if isinstance(tf, (int, float)) else f"{'--':>7s}"
        print(f"{record['group']:6s} {shown} {bw_s} {tf_s}  {record['case']}")
    bad = [r for r in results if r["status"] != "ok"]
    if bad:
        print(f"\n{len(bad)} case(s) did not run:")
        for record in bad:
            print(f"  - {record['case']}: {record['status']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
