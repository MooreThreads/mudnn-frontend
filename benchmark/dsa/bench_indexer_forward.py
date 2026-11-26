"""BSHD MP32 indexer-forward kernel benchmark.

The default case follows the reference non-paged MQA benchmark shape where
the indexer supports it: one sequence, large prefill-style Q/K, BF16 precision.
"""

from __future__ import annotations

import argparse

import torch
import torch_musa  # noqa: F401

from mudnn.deepseek_sparse_attention.indexer_forward.api_mp32 import (
    indexer_forward_wrapper,
)


DEVICE = "musa"
HEADDIM = 128


def _bench_kineto(fn, kernel_name, num_tests=30, flush_l2=True):
    fn()
    flush_size = int(8e9 // 4)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.MUSA],
        acc_events=True,
    ) as profiler:
        for _ in range(num_tests):
            if flush_l2:
                torch.empty(flush_size, dtype=torch.int, device=DEVICE).zero_()
            fn()
    events = [event for event in profiler.key_averages() if kernel_name in event.key]
    if not events:
        raise RuntimeError(f"Profiler did not find kernel {kernel_name!r}")
    total_time_us = sum(event.device_time_total for event in events)
    total_calls = sum(event.count for event in events)
    if total_calls <= 0:
        raise RuntimeError(f"Profiler reported no calls for {kernel_name!r}")
    return total_time_us / total_calls / 1e6


def _count_bytes(*tensors):
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in tensors
        if tensor is not None
    )


def _make_inputs(args, precision):
    batch = args.batch_size
    seqlen_q = args.seqlen_q
    seqlen_kv = args.seqlen_kv
    heads = args.heads

    torch.manual_seed(666)
    if precision == "mxfp8":
        q = (
            torch.randn(batch, seqlen_q, heads, HEADDIM, dtype=torch.float32)
            .to(torch.float8_e4m3fn)
            .to(DEVICE)
        )
        k = (
            torch.randn(batch, seqlen_kv, 1, HEADDIM, dtype=torch.float32)
            .to(torch.float8_e4m3fn)
            .to(DEVICE)
        )
        w = torch.randn(batch, seqlen_q, heads, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
        q_scale = torch.full(
            (batch, seqlen_q * heads, 4), 127, dtype=torch.uint8, device=DEVICE
        )
        k_scale = torch.full(
            (batch, seqlen_kv, 4), 127, dtype=torch.uint8, device=DEVICE
        )
    else:
        q = torch.randn(batch, seqlen_q, heads, HEADDIM, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
        k = torch.randn(batch, seqlen_kv, 1, HEADDIM, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
        w = torch.randn(batch, seqlen_q, heads, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
        q_scale = None
        k_scale = None

    if args.q_causal_offset == "auto":
        offset = seqlen_kv * args.ratio - seqlen_q if args.causal else seqlen_kv * args.ratio
    else:
        offset = args.q_causal_offset
    q_causal_offsets = torch.full(
        (batch,), offset, dtype=torch.int32, device=DEVICE
    )
    out = torch.empty(batch, seqlen_q, seqlen_kv, dtype=torch.float32, device=DEVICE)
    return q, k, w, q_scale, k_scale, q_causal_offsets, out, offset


def _num_logits(args, offset):
    if not args.causal:
        return args.batch_size * args.seqlen_q * args.seqlen_kv
    total = 0
    for q_local in range(args.seqlen_q):
        total += min(args.seqlen_kv, (offset + q_local + 1) // args.ratio)
    return args.batch_size * total


def benchmark(args, precision):
    q, k, w, q_scale, k_scale, q_causal_offsets, out, offset = _make_inputs(
        args, precision
    )

    def run():
        indexer_forward_wrapper(
            q,
            k,
            w,
            ratio=args.ratio,
            qhead_per_kv_head=args.heads,
            m_block_size=args.m_block,
            n_block_size=args.n_block,
            kv_stage=args.kv_stage,
            sm_scale=args.sm_scale,
            out=out,
            q_causal_offsets=q_causal_offsets,
            precision=precision,
            q_scale=q_scale,
            k_scale=k_scale,
        )

    seconds = _bench_kineto(
        run,
        "run_kernel",
        num_tests=args.num_tests,
        flush_l2=not args.no_flush_l2,
    )

    num_logits = _num_logits(args, offset)
    flops = 2 * num_logits * args.heads * HEADDIM
    tensors = [q, k, w, out]
    if q_scale is not None:
        tensors.extend([q_scale, k_scale])
    bytes_ = _count_bytes(*tensors) + num_logits * torch.float32.itemsize

    print(
        f"{precision}: B={args.batch_size}, Sq={args.seqlen_q}, "
        f"Sk={args.seqlen_kv}, H={args.heads}, D={HEADDIM}, "
        f"ratio={args.ratio}, causal={int(args.causal)}, "
        f"m={args.m_block}, n={args.n_block}, kv={args.kv_stage}: "
        f"{seconds * 1e6:.2f} us, {flops / seconds / 1e12:.2f} TFLOP/s, "
        f"{bytes_ / seconds / 1e9:.2f} GB/s"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark BSHD MP32 indexer forward."
    )
    parser.add_argument(
        "--precision", choices=("bf16", "mxfp8", "both"), default="bf16"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seqlen-q", type=int, default=6144)
    parser.add_argument(
        "--seqlen-kv", "--seqlen-k", dest="seqlen_kv", type=int, default=12288
    )
    parser.add_argument(
        "--heads", "--num-heads", dest="heads", type=int, default=64
    )
    parser.add_argument("--head-dim", type=int, default=HEADDIM)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--m-block", type=int, default=128)
    parser.add_argument("--n-block", type=int, default=512)
    parser.add_argument("--kv-stage", type=int, default=2)
    parser.add_argument("--sm-scale", type=float, default=1.0)
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use bottom-right causal offsets; default is full KV visibility.",
    )
    parser.add_argument(
        "--q-causal-offset",
        default="auto",
        help="Override q_causal_offset; auto uses full KV or bottom-right causal.",
    )
    parser.add_argument("--num-tests", type=int, default=30)
    parser.add_argument("--no-flush-l2", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.seqlen_q <= 0 or args.seqlen_kv <= 0:
        parser.error("batch-size, seqlen-q, and seqlen-kv must be positive")
    if args.head_dim != HEADDIM:
        parser.error(f"--head-dim must be {HEADDIM}")
    if args.heads not in (32, 64):
        parser.error("--heads must be 32 or 64 for the MP32 indexer")
    if args.ratio <= 0:
        parser.error("--ratio must be positive")
    if args.m_block not in (64, 128):
        parser.error("--m-block must be 64 or 128")
    if args.n_block not in (128, 256, 512):
        parser.error("--n-block must be 128, 256, or 512")
    if args.kv_stage % 2 != 0 or not (2 <= args.kv_stage <= 12):
        parser.error("--kv-stage must be an even value in [2, 12]")
    if args.causal and args.seqlen_q > args.seqlen_kv * args.ratio:
        parser.error("--causal requires seqlen-q <= seqlen-kv * ratio")
    if args.q_causal_offset != "auto":
        try:
            args.q_causal_offset = int(args.q_causal_offset)
        except ValueError:
            parser.error("--q-causal-offset must be an integer or 'auto'")

    precisions = ("bf16", "mxfp8") if args.precision == "both" else (args.precision,)
    for precision in precisions:
        benchmark(args, precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
