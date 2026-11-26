# SPDX-FileCopyrightText: Copyright (c) 2026 Moore Threads Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TileLang CSA/HCA Compressor forward region.

The layout and reference follow the competing CuTe-DSL test: packed THD
inputs are gathered into ratio-sized blocks, ``coff == 2`` applies the
overlap transform, and the gated softmax is evaluated in an eager reference.
The suite covers forward, recompute backward, static-capacity padding and MUSA
graph capture with the same behavior classes as the competing CuTe-DSL suite.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mudnn.csa.compressor import (  # noqa: E402
    CSACompressorBackward,
    CSACompressorForward,
    csa_compressor_backward_wrapper,
    csa_compressor_forward_wrapper,
)
import mudnn.csa.compressor.compressor_sm100 as compressor_sm100  # noqa: E402
import mudnn.csa.compressor.compressor_sm100_r128 as compressor_sm100_r128  # noqa: E402
from mudnn.csa.compressor.compressor_sm100_r128 import (  # noqa: E402
    fused_backward_schedule_r128,
)


# ---------------------------------------------------------------------------
# Eager reference (self-contained mirror of the competitor test)
# ---------------------------------------------------------------------------


def _batch_of_row(cu_seqlens, total):
    """Return the segment owning every packed compressed row.

    This is the competitor's ``bucketize(..., right=True)`` expression.  The
    MUSA PyTorch build does not implement device ``bucketize``, so the test
    oracle evaluates this tiny boundary-vector operation on the host and moves
    only the resulting row-to-segment map back to the input device.
    """

    n_seg = int(cu_seqlens.shape[0] - 1)
    row_idx = torch.arange(total, device="cpu", dtype=torch.int64)
    if total == 0:
        return row_idx.to(cu_seqlens.device)
    batch_ids = torch.bucketize(row_idx, cu_seqlens.detach().cpu()[1:], right=True).clamp(max=max(n_seg - 1, 0))
    return batch_ids.to(cu_seqlens.device)


def _overlap_transform_thd(tensor, is_first_in_seg, head_dim, fill_value=0):
    """Mirror the competitor's ``Compressor._overlap_transform_thd``.

    ``tensor`` is ``[total_comp, ratio, 1, coff * head_dim]`` and the result is
    ``[total_comp, 2 * ratio, 1, head_dim]``.
    """

    n, ratio, b_dim, _ = tensor.size()
    d = int(head_dim)
    new_tensor = tensor.new_full((n, 2 * ratio, b_dim, d), fill_value)
    new_tensor[:, ratio:] = tensor[:, :, :, d:]
    prev_data = torch.roll(tensor[:, :, :, :d], shifts=1, dims=0)
    prev_data[is_first_in_seg] = fill_value
    new_tensor[:, :ratio] = prev_data
    return new_tensor


def _eager_pool(
    kv,
    score,
    ape,
    cu_seqlens,
    cu_seqlens_comp,
    total_comp,
    ratio,
    head_dim,
    coff,
    mode="fp32",
):
    """Reference gated pooling with the competitor's ``upstream/fp32/fp64`` modes."""

    device = kv.device
    total_comp = int(total_comp)
    ratio = int(ratio)
    head_dim = int(head_dim)
    if kv.ndim != 2 or score.shape != kv.shape or ape.ndim != 2:
        raise ValueError("kv/score must be rank-2 with equal shape and ape must be rank-2")
    if cu_seqlens.ndim != 1 or cu_seqlens_comp.ndim != 1:
        raise ValueError("cumulative offsets must be rank-1")
    if cu_seqlens.shape != cu_seqlens_comp.shape:
        raise ValueError("cumulative offsets must have the same shape")
    cu_values = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    cuc_values = [int(x) for x in cu_seqlens_comp.detach().cpu().tolist()]
    if len(cu_values) < 2 or cu_values[0] != 0 or cuc_values[0] != 0:
        raise ValueError("cumulative offsets must start at zero and contain B + 1 values")
    if any(a > b for a, b in zip(cu_values, cu_values[1:])) or any(a > b for a, b in zip(cuc_values, cuc_values[1:])):
        raise ValueError("cumulative offsets must be nondecreasing")
    if cu_values[-1] != int(kv.shape[0]):
        raise ValueError("cu_seqlens[-1] must equal kv.shape[0]")
    expected = sum((b - a) // ratio for a, b in zip(cu_values, cu_values[1:]))
    if cuc_values[-1] != expected:
        raise ValueError(f"expected {expected} compressed rows, got {cuc_values[-1]}")
    if total_comp < cuc_values[-1]:
        raise ValueError("output capacity must cover all valid compressed rows")
    row_idx = torch.arange(total_comp, device=device, dtype=cu_seqlens_comp.dtype)
    batch_ids = _batch_of_row(cu_seqlens_comp, total_comp)
    valid_comp = row_idx < cu_seqlens_comp[-1]
    local_pos = row_idx - cu_seqlens_comp[batch_ids]
    local_pos = torch.where(valid_comp, local_pos, torch.zeros_like(local_pos))
    base = cu_seqlens[batch_ids].unsqueeze(1) + local_pos.unsqueeze(1) * ratio
    base = torch.where(valid_comp.unsqueeze(1), base, torch.zeros_like(base))
    offsets = torch.arange(ratio, device=device, dtype=base.dtype).unsqueeze(0)
    gather_idx = base + offsets

    if mode == "fp32":
        kv_work, score_work, ape_work = kv.float(), score.float(), ape.float()
    elif mode == "fp64":
        kv_work, score_work, ape_work = kv.double(), score.double(), ape.double()
    elif mode == "upstream":
        kv_work, score_work, ape_work = kv, score, ape
    else:
        raise ValueError(f"unknown reference mode {mode!r}")

    # Keep the singleton batch dimension from the competitor's THD reference.
    kv_grouped = kv_work[gather_idx].unsqueeze(2)
    score_grouped = score_work[gather_idx].unsqueeze(2)
    score_grouped = score_grouped + ape_work.view(1, ratio, 1, -1)

    if coff == 2:
        is_first = local_pos == 0
        kv_grouped = _overlap_transform_thd(kv_grouped, is_first, head_dim, fill_value=0)
        score_grouped = _overlap_transform_thd(score_grouped, is_first, head_dim, fill_value=float("-inf"))

    if mode == "upstream":
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32).to(kv_grouped.dtype)
        out = (kv_grouped * weights).sum(dim=1)
    elif mode == "fp32":
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float32)
        out = (kv_grouped * weights).sum(dim=1).to(torch.bfloat16)
    else:
        weights = torch.softmax(score_grouped, dim=1, dtype=torch.float64)
        out = (kv_grouped * weights).sum(dim=1)
    return out.squeeze(1)


# ---------------------------------------------------------------------------
# Input construction and small runners
# ---------------------------------------------------------------------------


def _make_inputs(lens, head_dim, ratio, coff, seed=1234, device="cpu"):
    """Build deterministic packed ``(kv, score, ape, cu, cuc, total_comp)`` inputs."""

    lens = tuple(int(x) for x in lens)
    total = sum(lens)
    width = int(coff) * int(head_dim)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.randn(total, width, generator=generator, dtype=torch.float32).bfloat16()
    score = (torch.randn(total, width, generator=generator, dtype=torch.float32).mul_(1.5)).bfloat16()
    ape = torch.randn(ratio, width, generator=generator, dtype=torch.float32).mul_(0.25)
    cu_values = [0]
    compressed_values = [0]
    for length in lens:
        cu_values.append(cu_values[-1] + length)
        compressed_values.append(compressed_values[-1] + length // ratio)
    cu = torch.tensor(cu_values, dtype=torch.int32, device=device)
    cuc = torch.tensor(compressed_values, dtype=torch.int32, device=device)
    total_comp = int(compressed_values[-1])
    return (
        kv.to(device),
        score.to(device),
        ape.to(device),
        cu,
        cuc,
        total_comp,
    )


def _reference_forward(kv, score, ape, cu, cuc, total_comp, ratio, head_dim, coff, mode="fp32"):
    return _eager_pool(
        kv,
        score,
        ape,
        cu,
        cuc,
        total_comp,
        ratio,
        head_dim,
        coff,
        mode=mode,
    )


def _run_fused(kv, score, ape, cu, cuc, ratio, head_dim, coff):
    return csa_compressor_forward_wrapper(
        kv,
        score,
        ape,
        cu,
        cuc,
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
    )["out"]


def _make_grad_out(total_comp, head_dim, seed=7, device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(total_comp, head_dim, generator=generator, dtype=torch.float32).bfloat16().to(device)


def _run_eager_fwd_bwd(
    kv,
    score,
    ape,
    cu,
    cuc,
    total_comp,
    ratio,
    head_dim,
    coff,
    grad_out,
    mode,
):
    """Forward and autograd backward through the self-contained eager oracle."""

    dtype = torch.float64 if mode == "fp64" else None
    kv_l = (kv.to(dtype) if dtype else kv.clone()).requires_grad_(True)
    score_l = (score.to(dtype) if dtype else score.clone()).requires_grad_(True)
    ape_l = (ape.to(dtype) if dtype else ape.clone()).requires_grad_(True)
    out = _reference_forward(
        kv_l,
        score_l,
        ape_l,
        cu,
        cuc,
        total_comp,
        ratio,
        head_dim,
        coff,
        mode=mode,
    )
    out.backward(grad_out.to(out.dtype))
    return (
        out.detach(),
        kv_l.grad.detach(),
        score_l.grad.detach(),
        ape_l.grad.detach(),
    )


def _run_fused_fwd_bwd(kv, score, ape, cu, cuc, total_comp, ratio, head_dim, coff, grad_out):
    out = csa_compressor_forward_wrapper(
        kv,
        score,
        ape,
        cu,
        cuc,
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
        total_comp=total_comp,
    )["out"]
    grads = csa_compressor_backward_wrapper(
        kv,
        score,
        ape,
        cu,
        cuc,
        grad_out,
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
    )
    return out, grads["grad_kv"], grads["grad_score"], grads["grad_ape"]


def _assert_grads_vs_fp32(gkv, gs, ref_kv, ref_s, ratio):
    # TileLang now pins mul.rn/fma.rn to the same expression tree as CuTe.  The
    # torch_musa eager softmax still differs from TileLang's device exp on a tiny
    # number of one-BF16-ULP short-ratio results, so those use a 0.2% mismatch cap;
    # ratio 128 keeps the competitor's 0.1% cap.  The max-abs and fp64-parity gates
    # remain unchanged.
    mismatch_fraction = 0.002 if ratio in (2, 4) else 0.001
    for name, fused_t, ref_t in (("dKV", gkv, ref_kv), ("dScore", gs, ref_s)):
        diff = (fused_t.float() - ref_t.float()).abs()
        n_diff = int((fused_t != ref_t).sum().item())
        assert n_diff <= max(1, int(mismatch_fraction * fused_t.numel())), (
            name,
            n_diff,
            diff.max().item(),
        )
        assert diff.max().item() <= 1.6e-2, (name, diff.max().item())


def _require_musa_tilelang():
    pytest.importorskip("torch_musa")
    pytest.importorskip("tilelang")
    if not getattr(torch, "musa", None) or not torch.musa.is_available():
        pytest.skip("MUSA is unavailable")


def _meta(shape, dtype, stride=None):
    """Metadata-only tensor, matching the competitor's support-test helper."""

    if stride is None:
        return torch.empty(shape, dtype=dtype, device="meta")
    return torch.empty_strided(shape, stride, dtype=dtype, device="meta")


def _meta_samples(
    total=512,
    head_dim=128,
    ratio=4,
    coff=2,
    n_seg=2,
    total_comp=None,
    kv_dtype=None,
    ape_dtype=None,
    cu_dtype=None,
    out_dtype=None,
    score_shape=None,
    cuc_len=None,
    kv_stride=None,
):
    """Build consistent meta descriptors with field overrides for negative tests."""

    kv_dtype = torch.bfloat16 if kv_dtype is None else kv_dtype
    ape_dtype = torch.float32 if ape_dtype is None else ape_dtype
    cu_dtype = torch.int32 if cu_dtype is None else cu_dtype
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    width = coff * head_dim
    if total_comp is None:
        total_comp = total // ratio
    return dict(
        sample_kv=_meta((total, width), kv_dtype, stride=kv_stride),
        sample_score=_meta(score_shape or (total, width), kv_dtype),
        sample_ape=_meta((ratio, width), ape_dtype),
        sample_cu_seqlens=_meta((n_seg + 1,), cu_dtype),
        sample_cu_seqlens_comp=_meta((cuc_len or (n_seg + 1),), cu_dtype),
        sample_out=_meta((total_comp, head_dim), out_dtype),
    )


# Exact forward/backward numerical matrix from the competing suite.
_SHAPES = [
    pytest.param((2048,), 128, 4, 2, id="b1-d128-r4"),
    pytest.param((1023, 2048, 509), 128, 4, 2, id="ragged3-d128-r4"),
    pytest.param((2048,), 512, 4, 2, id="b1-d512-r4"),
    pytest.param((6144,), 512, 4, 2, id="b1x6144-d512-r4"),
    pytest.param((3, 515, 1024, 129), 128, 4, 2, id="short-seg-d128-r4"),
    pytest.param((64, 0, 253, 3), 128, 4, 2, id="empty-seg-d128-r4"),
    pytest.param((2048,), 128, 4, 1, id="b1-d128-r4-coff1"),
    pytest.param((1023, 2048, 509), 128, 4, 1, id="ragged3-d128-r4-coff1"),
    pytest.param((2048,), 512, 4, 1, id="b1-d512-r4-coff1"),
    pytest.param((3, 515, 1024, 129), 128, 4, 1, id="short-seg-d128-r4-coff1"),
    pytest.param((64, 0, 253, 3), 128, 4, 1, id="empty-seg-d128-r4-coff1"),
    pytest.param((2048,), 128, 2, 2, id="b1-d128-r2"),
    pytest.param((1023, 2048, 509), 128, 2, 2, id="ragged3-d128-r2"),
    pytest.param((2048,), 512, 2, 2, id="b1-d512-r2"),
    pytest.param((1, 515, 1024, 129), 128, 2, 2, id="short-seg-d128-r2"),
    pytest.param((64, 0, 253, 3), 128, 2, 2, id="empty-seg-d128-r2"),
    pytest.param((2048,), 128, 2, 1, id="b1-d128-r2-coff1"),
    pytest.param((1023, 2048, 509), 128, 2, 1, id="ragged3-d128-r2-coff1"),
    pytest.param((2048,), 512, 2, 1, id="b1-d512-r2-coff1"),
    pytest.param((1, 515, 1024, 129), 128, 2, 1, id="short-seg-d128-r2-coff1"),
    pytest.param((64, 0, 253, 3), 128, 2, 1, id="empty-seg-d128-r2-coff1"),
    pytest.param((8192,), 128, 128, 1, id="b1x8192-d128-r128c1"),
    pytest.param((1023, 2048, 509), 128, 128, 1, id="ragged3-d128-r128c1"),
    pytest.param(
        (127, 8192, 0, 129, 128, 3, 515, 1024),
        128,
        128,
        1,
        id="edgepack-d128-r128c1",
    ),
    pytest.param((8192,), 128, 128, 2, id="b1x8192-d128-r128c2"),
    pytest.param(
        (127, 8192, 0, 129, 128, 3, 515, 1024),
        128,
        128,
        2,
        id="edgepack-d128-r128c2",
    ),
    pytest.param((2048, 509), 512, 128, 1, id="ragged2-d512-r128c1"),
    pytest.param((8192,), 512, 128, 2, id="b1x8192-d512-r128c2"),
]


# ---------------------------------------------------------------------------
# Reference and support contracts
# ---------------------------------------------------------------------------


def test_backward_schedules_split_head_dim_and_window_chunks():
    assert compressor_sm100._backward_r4_vec4_schedule(2, 128, 1, 4096, 2) == (8, 1, False)
    assert compressor_sm100._backward_r4_vec4_schedule(4, 128, 2, 2048, 2) == (4, 1, False)
    assert compressor_sm100._backward_r4_vec4_schedule(4, 128, 2, 6144, 4) == (2, 1, False)
    assert compressor_sm100._backward_r4_vec4_schedule(4, 512, 2, 2048, 2) == (8, 2, False)
    assert compressor_sm100._backward_r4_vec4_schedule(4, 512, 2, 6144, 4) == (8, 4, False)
    assert fused_backward_schedule_r128(128, 128, 1, 1024) == (
        "rowgroup", 4, 32, 4, 4, 16
    )
    assert fused_backward_schedule_r128(128, 512, 2, 1024) == (
        "rowgroup", 4, 32, 4, 4, 16
    )


def test_r128_backward_small_row_dispatch_boundary():
    assert fused_backward_schedule_r128(128, 128, 1, 128) == ("tilewarp", 1, 32, 4, 4, 16)
    assert fused_backward_schedule_r128(128, 128, 1, 129) == ("tilewarp", 1, 32, 4, 4, 16)
    assert fused_backward_schedule_r128(128, 128, 1, 512) == ("tilewarp", 1, 32, 4, 4, 16)
    assert fused_backward_schedule_r128(128, 128, 1, 513) == ("tilewarp", 1, 16, 4, 4, 16)


def test_backward_r4_vec4_schedule_uses_capacity_buckets():
    schedule = compressor_sm100._backward_r4_vec4_schedule
    for ratio, dim, coff, rows, num_offsets, expected in (
        (4, 128, 1, 1, 2, (4, 1, True)),
        (4, 128, 1, 256, 2, (4, 4, True)),
        (4, 512, 1, 512, 2, (8, 4, True)),
        (4, 128, 1, 1024, 2, (8, 1, False)),
        (4, 128, 1, 16384, 2, (8, 2, False)),
        (4, 128, 2, 128, 2, (2, 2, True)),
        (4, 512, 2, 1024, 2, (8, 4, True)),
        (4, 128, 2, 4096, 2, (16, 4, False)),
        (4, 512, 2, 43008, 2, (64, 16, False)),
        (4, 512, 2, 43008, 4, (48, 12, False)),
        (2, 128, 1, 4096, 2, (8, 1, False)),
        (4, 96, 1, 4096, 2, None),
    ):
        assert schedule(ratio, dim, coff, rows, num_offsets) == expected


@pytest.mark.parametrize(
    "rows,num_offsets,head_dim,expected",
    [
        (2048, 2, 128, (4, 1, False)),
        (2048, 2, 512, (8, 2, False)),
        (6144, 4, 128, (2, 1, False)),
        (6144, 4, 512, (8, 4, False)),
        (2048, 4, 128, (16, 4, False)),
        (6144, 2, 512, (8, 2, False)),
        (1024, 2, 128, (4, 4, True)),
    ],
)
def test_r4_c2_backward_vec4_small_schedule(rows, num_offsets, head_dim, expected):
    assert compressor_sm100._backward_r4_vec4_schedule(
        4, head_dim, 2, rows, num_offsets
    ) == expected


@pytest.mark.parametrize("num_offsets", [2, 4])
def test_r4_c2_backward_precompile_only_active_large_path(monkeypatch, num_offsets):
    _require_musa_tilelang()
    builds = []
    monkeypatch.setattr(compressor_sm100, "_COMPILED", {})
    monkeypatch.setattr(
        compressor_sm100,
        "_compile_bwd_r4_vec4",
        lambda *args, **kwargs: builds.append((args, kwargs)),
    )
    device = torch.device("musa", torch.musa.current_device())
    compressor_sm100.precompile_bwd(
        4, 512, 2, device, nb_total=43008, num_offsets=num_offsets
    )
    assert len(builds) == 1
    assert builds[0][0][2] == 2
    assert builds[0][0][5] is (num_offsets == 2)
    assert builds[0][0][6] is False
    assert builds[0][1] == {"ratio": 4}


@pytest.mark.parametrize("ratio", [2, 4])
def test_short_backward_precompile_requires_shape_for_vec4_dispatch(ratio):
    with pytest.raises(ValueError, match="requires nb_total and num_offsets"):
        compressor_sm100.precompile_bwd(ratio, 128, 1, object())
    with pytest.raises(ValueError, match="ratio=2 or ratio=4"):
        compressor_sm100._build_backward_program_vec4_r4(128, 1, ratio=3)


@pytest.mark.parametrize("ratio", [2, 4])
def test_short_backward_precompile_reuses_warm_kernel_during_capture(monkeypatch, ratio):
    _require_musa_tilelang()
    compiled = {}
    compile_calls = []
    monkeypatch.setattr(compressor_sm100, "_COMPILED", compiled)
    monkeypatch.setattr(
        compressor_sm100, "_build_backward_program_vec4_r4",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        compressor_sm100.tilelang, "compile",
        lambda *args, **kwargs: compile_calls.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(torch.musa, "is_current_stream_capturing", lambda: False)
    device = torch.device("musa", torch.musa.current_device())
    compressor_sm100.precompile_bwd(
        ratio, 128, 1, device, nb_total=64, num_offsets=2
    )
    assert len(compile_calls) == 1

    monkeypatch.setattr(torch.musa, "is_current_stream_capturing", lambda: True)
    compressor_sm100.precompile_bwd(
        ratio, 128, 1, device, nb_total=64, num_offsets=2
    )
    assert len(compile_calls) == 1

    with pytest.raises(RuntimeError, match="MUSA graph capture"):
        compressor_sm100.precompile_bwd(
            ratio, 128, 2, device, nb_total=64, num_offsets=2
        )
    assert len(compile_calls) == 1


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("coff", [1, 2])
def test_check_support_accepts_short_ratio_envelope(ratio, coff):
    for api_cls in (CSACompressorForward, CSACompressorBackward):
        api = api_cls(**_meta_samples(ratio=ratio, coff=coff), ratio=ratio, coff=coff)
        assert api.check_support() is True
        assert api.head_dim == 128
        assert api.total_tokens == 512
        assert api.total_comp == 512 // ratio


@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("coff", [1, 2])
def test_check_support_accepts_ratio128_envelope(head_dim, coff):
    for api_cls in (CSACompressorForward, CSACompressorBackward):
        api = api_cls(
            **_meta_samples(total=1024, head_dim=head_dim, ratio=128, coff=coff),
            ratio=128,
            coff=coff,
        )
        assert api.check_support() is True
        assert api.head_dim == head_dim
        assert api.total_comp == 8


@pytest.mark.parametrize("capability", [(3, 0), (3, 1), (4, 0)])
def test_check_support_accepts_musa_arch_major_3_or_newer(monkeypatch, capability):
    _require_musa_tilelang()
    monkeypatch.setattr(torch.musa, "get_device_capability", lambda device=None: capability)
    api = CSACompressorForward(**_meta_samples(), ratio=4, coff=2)
    assert api.check_support() is True


def test_check_support_rejects_musa_arch_major_below_3(monkeypatch):
    _require_musa_tilelang()
    monkeypatch.setattr(torch.musa, "get_device_capability", lambda device=None: (2, 2))
    api = CSACompressorForward(**_meta_samples(), ratio=4, coff=2)
    with pytest.raises(RuntimeError, match="MUSA architecture major >= 3"):
        api.check_support()


@pytest.mark.parametrize(
    "sample_kwargs,ctor_kwargs,match",
    [
        ({}, {"ratio": 128, "coff": 3}, "coff in"),
        ({}, {"ratio": 8, "coff": 2}, "ratio in"),
        ({"coff": 3}, {"ratio": 4, "coff": 3}, "coff in"),
        ({}, {"ratio": 4, "coff": 0}, "coff in"),
        ({"ratio": 2, "coff": 3}, {"ratio": 2, "coff": 3}, "coff in"),
        ({"ratio": 2}, {"ratio": 2, "coff": 0}, "coff in"),
        ({"kv_dtype": torch.float16}, {}, "kv"),
        ({"ape_dtype": torch.bfloat16}, {}, "ape"),
        ({"cu_dtype": torch.int64}, {}, "cu_seqlens"),
        ({"out_dtype": torch.float32}, {}, "out"),
        ({"score_shape": (512, 128)}, {}, "score shape"),
        ({"cuc_len": 4}, {}, r"B \+ 1"),
        ({"total": 2**25, "head_dim": 128}, {}, "less than 2\\*\\*31"),
        (
            {"total": 4, "head_dim": 128 * 65535 + 1, "total_comp": 1},
            {},
            "head_dim",
        ),
        ({"total": 2, "total_comp": 1}, {}, "requires at least ratio"),
        (
            {"total": 1, "total_comp": 1, "ratio": 2},
            {"ratio": 2},
            "requires at least ratio",
        ),
        ({"kv_stride": (512, 2)}, {}, "contiguous"),
        (
            {"total": 1024, "head_dim": 96, "ratio": 128, "coff": 1},
            {"ratio": 128, "coff": 1},
            "ratio=128 is validated for head_dim",
        ),
    ],
)
def test_check_support_rejects_outside_envelope(sample_kwargs, ctor_kwargs, match):
    api = CSACompressorForward(
        **_meta_samples(**sample_kwargs),
        ratio=ctor_kwargs.get("ratio", 4),
        coff=ctor_kwargs.get("coff", 2),
    )
    with pytest.raises(ValueError, match=match):
        api.check_support()


def test_class_check_support_rejects_cpu_tensors():
    samples = _meta_samples()
    samples["sample_kv"] = torch.empty((512, 256), dtype=torch.bfloat16, device="cpu")
    api = CSACompressorForward(**samples, ratio=4, coff=2)
    with pytest.raises(ValueError, match="MUSA"):
        api.check_support()


def test_forward_requires_optimized_head_dim():
    api = CSACompressorForward(**_meta_samples(head_dim=65), ratio=4, coff=2)
    with pytest.raises(ValueError, match="head_dim in \\(128, 512\\)"):
        api.check_support()


# ---------------------------------------------------------------------------
# MUSA forward numerics and API execution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lens,head_dim,ratio,coff", _SHAPES)
def test_forward_matches_competitor_style_reference(lens, head_dim, ratio, coff):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs(lens, head_dim, ratio, coff, device="musa")
    grad_out = _make_grad_out(total_comp, head_dim, device="musa")
    fused = _run_fused_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        total_comp,
        ratio,
        head_dim,
        coff,
        grad_out,
    )
    fp32_ref = _run_eager_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        total_comp,
        ratio,
        head_dim,
        coff,
        grad_out,
        "fp32",
    )
    upstream_ref = _run_eager_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        total_comp,
        ratio,
        head_dim,
        coff,
        grad_out,
        "upstream",
    )
    # MUSA eager does not expose all float64 indexing/reduction primitives.
    fp64_ref = _run_eager_fwd_bwd(
        kv.cpu(),
        score.cpu(),
        ape.cpu(),
        cu.cpu(),
        cuc.cpu(),
        total_comp,
        ratio,
        head_dim,
        coff,
        grad_out.cpu(),
        "fp64",
    )

    _assert_grads_vs_fp32(fused[1], fused[2], fp32_ref[1], fp32_ref[2], ratio)
    if ratio == 128:
        _assert_r128_forward_vs_fp32(fused[0], fp32_ref[0])
    else:
        fwd_diff = (fused[0].float() - fp32_ref[0].float()).abs()
        n_diff = int((fused[0] != fp32_ref[0]).sum().item())
        assert n_diff <= max(1, int(0.001 * fused[0].numel())), n_diff
        assert fwd_diff.max().item() <= 1.6e-2
    assert (fused[3] - fp32_ref[3]).abs().max().item() <= 1e-3

    for fused_t, upstream_t in zip(fused, upstream_ref):
        torch.testing.assert_close(fused_t.float(), upstream_t.float(), rtol=0, atol=0.1)

    eager_baseline = fp32_ref if ratio == 128 else upstream_ref
    for fused_t, eager_t, oracle_t in zip(fused, eager_baseline, fp64_ref):
        fused_error = (fused_t.cpu().double() - oracle_t.double()).abs().max().item()
        eager_error = (eager_t.cpu().double() - oracle_t.double()).abs().max().item()
        assert fused_error <= eager_error * (1 + 1e-6) + 1e-4


def test_ratio128_first_block_ignores_empty_partial_with_very_negative_logits():
    """Inactive previous-half chunks must not change the online-softmax max."""

    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs((128,), 128, 128, 2, device="musa")
    kv.fill_(1)
    score.fill_(-1000)
    ape.zero_()
    got = _run_fused(kv, score, ape, cu, cuc, 128, 128, 2)
    want = _reference_forward(kv, score, ape, cu, cuc, total_comp, 128, 128, 2, mode="fp32")
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert torch.equal(got, torch.ones_like(got))


@pytest.mark.parametrize("ratio,lens", [(4, (2,)), (128, (127, 0, 3))])
@pytest.mark.parametrize("coff", [1, 2])
def test_empty_output_on_musa(ratio, lens, coff):
    """A pack with no complete ratio-sized block launches no TileLang kernel."""

    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs(lens, 128, ratio, coff, device="musa")
    assert total_comp == 0
    got = _run_fused(kv, score, ape, cu, cuc, ratio, 128, coff)
    want = _reference_forward(kv, score, ape, cu, cuc, 0, ratio, 128, coff)
    assert got.shape == (0, 128)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    grad_out = torch.empty((0, 128), dtype=torch.bfloat16, device="musa")
    grads = csa_compressor_backward_wrapper(
        kv,
        score,
        ape,
        cu,
        cuc,
        grad_out,
        ratio=ratio,
        head_dim=128,
        coff=coff,
    )
    assert grads["grad_kv"].abs().sum().item() == 0
    assert grads["grad_score"].abs().sum().item() == 0
    assert grads["grad_ape"].abs().sum().item() == 0


def test_class_api_rejects_misaligned_vec2_storage_on_musa():
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs((128,), 128, 2, 1, device="musa")
    storage = torch.empty(kv.numel() + 1, dtype=torch.bfloat16, device="musa")
    misaligned_kv = storage[1:].view_as(kv)
    misaligned_kv.copy_(kv)
    assert misaligned_kv.is_contiguous()
    assert misaligned_kv.data_ptr() % 16 == 2

    sample_out = torch.empty((total_comp, 128), dtype=torch.bfloat16, device="meta")
    api = CSACompressorForward(misaligned_kv, score, ape, cu, cuc, sample_out, ratio=2, coff=1)
    assert api.check_support() is True
    api.compile()
    out = torch.empty((total_comp, 128), dtype=torch.bfloat16, device="musa")
    with pytest.raises(ValueError, match="kv base pointer must be 16-byte aligned"):
        api.execute(misaligned_kv, score, ape, cu, cuc, out)


# ---------------------------------------------------------------------------
# Competitor parity: determinism and complete ratio-128 dispatch envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("coff", [1, 2])
def test_replay_determinism(ratio, coff):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs((1023, 2048, 509), 128, ratio, coff, device="musa")
    grad_out = _make_grad_out(total_comp, 128, device="musa")
    runs = [_run_fused_fwd_bwd(kv, score, ape, cu, cuc, total_comp, ratio, 128, coff, grad_out) for _ in range(3)]
    for other in runs[1:]:
        for index in range(3):
            assert torch.equal(runs[0][index], other[index])
        torch.testing.assert_close(runs[0][3], other[3], rtol=0, atol=1e-3)


def test_replay_determinism_r128():
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs((1023, 2048, 509), 128, 128, 1, device="musa")
    grad_out = _make_grad_out(total_comp, 128, device="musa")
    runs = [_run_fused_fwd_bwd(kv, score, ape, cu, cuc, total_comp, 128, 128, 1, grad_out) for _ in range(3)]
    for other in runs[1:]:
        for index in range(3):
            assert torch.equal(runs[0][index], other[index])
        torch.testing.assert_close(runs[0][3], other[3], rtol=0, atol=1e-3)


def test_r128_backward_dispatch_boundaries():
    for coff in (1, 2):
        for head_dim in (128, 512):
            for rows in (1, 64, 128, 512, 1024, 4096, 8192, 16384):
                kind, row_cta, tile, warps, chunk, shards = (
                    fused_backward_schedule_r128(128, head_dim, coff, rows)
                )
                assert kind in ("rowgroup", "tilewarp")
                assert row_cta in (1, 2, 4, 8)
                assert tile in (8, 16, 32, 64, 128)
                assert warps in (4, 8)
                assert chunk == 4
                assert shards in (8, 16)
                assert tile % warps == 0
    with pytest.raises(ValueError):
        fused_backward_schedule_r128(64, 128, 1, 128)
    with pytest.raises(ValueError):
        fused_backward_schedule_r128(128, 256, 1, 128)

    # The transition rows are production schedule boundaries.
    for coff, head_dim, rows, expected in (
        (1, 128, 512, ("tilewarp", 1, 32, 4, 4, 16)),
        (1, 128, 513, ("tilewarp", 1, 16, 4, 4, 16)),
        (1, 128, 768, ("tilewarp", 1, 16, 4, 4, 16)),
        (1, 128, 769, ("rowgroup", 4, 32, 4, 4, 16)),
        (1, 512, 256, ("tilewarp", 1, 16, 4, 4, 16)),
        (1, 512, 257, ("rowgroup", 8, 16, 8, 4, 16)),
        (2, 128, 128, ("tilewarp", 1, 32, 4, 4, 16)),
        (2, 128, 129, ("tilewarp", 1, 16, 4, 4, 16)),
        (2, 128, 512, ("tilewarp", 1, 16, 4, 4, 16)),
        (2, 128, 513, ("rowgroup", 4, 64, 4, 4, 16)),
        (2, 128, 2048, ("rowgroup", 8, 32, 8, 4, 16)),
        (2, 128, 2049, ("rowgroup", 4, 32, 4, 4, 16)),
        (2, 512, 256, ("tilewarp", 1, 16, 4, 4, 16)),
        (2, 512, 257, ("rowgroup", 8, 32, 8, 4, 16)),
        (2, 512, 2048, ("rowgroup", 4, 32, 4, 4, 16)),
        (2, 512, 2049, ("rowgroup", 4, 16, 4, 4, 16)),
        (2, 512, 4097, ("rowgroup", 4, 16, 4, 4, 8)),
    ):
        assert fused_backward_schedule_r128(128, head_dim, coff, rows) == expected


_R128_EXPECTED_SCHEDULES = [
    # Small measured rows use tilewarp; larger rows use rowgroup APE-TMA.
    (1, 128, 64, ("tilewarp", 1, 16, 4, 4, True)),
    (1, 128, 256, ("tilewarp", 1, 16, 4, 4, True)),
    (1, 128, 1024, ("tma", 16, 64, True)),
    (2, 128, 64, ("tilewarp", 1, 16, 4, 4, True)),
    (2, 128, 256, ("tilewarp", 1, 16, 4, 4, True)),
    (1, 512, 256, ("tma", 8, 32, True)),
    (1, 512, 1024, ("tma", 8, 32, True)),
    (2, 512, 64, ("tilewarp", 1, 32, 4, 8, True)),
    (2, 512, 256, ("tma", 16, 64, True)),
    (2, 512, 1024, ("tma", 16, 64, True)),
    (2, 128, 128, ("tilewarp", 1, 16, 4, 4, True)),
    (1, 512, 64, ("tilewarp", 1, 32, 4, 8, True)),
]

_R128_EXPECTED_BACKWARD_SCHEDULES = {
    (1, 128, 64): ("tilewarp", 1, 32, 4, 4, 16),
    (1, 128, 256): ("tilewarp", 1, 32, 4, 4, 16),
    (1, 128, 1024): ("rowgroup", 4, 32, 4, 4, 16),
    (2, 128, 64): ("tilewarp", 1, 32, 4, 4, 16),
    (2, 128, 256): ("tilewarp", 1, 16, 4, 4, 16),
    (1, 512, 256): ("tilewarp", 1, 16, 4, 4, 16),
    (1, 512, 1024): ("rowgroup", 8, 32, 8, 4, 16),
    (2, 512, 64): ("tilewarp", 1, 16, 4, 4, 16),
    (2, 512, 256): ("tilewarp", 1, 16, 4, 4, 16),
    (2, 512, 1024): ("rowgroup", 4, 32, 4, 4, 16),
    (2, 128, 128): ("tilewarp", 1, 32, 4, 4, 16),
    (1, 512, 64): ("tilewarp", 1, 32, 4, 4, 16),
}


def test_r128_dispatch_expected_schedules():
    for coff, head_dim, rows, expected in _R128_EXPECTED_SCHEDULES:
        assert compressor_sm100_r128._forward_kernel_schedule_r128(128, head_dim, coff, rows, 2) == expected
        assert fused_backward_schedule_r128(128, head_dim, coff, rows) == (_R128_EXPECTED_BACKWARD_SCHEDULES[(coff, head_dim, rows)])



@pytest.mark.parametrize("coff,head_dim", [(1, 128), (1, 512), (2, 128), (2, 512)])
def test_r128_default_precompile_covers_every_backward_bucket(monkeypatch, coff, head_dim):
    _require_musa_tilelang()
    compiled = set()
    monkeypatch.setattr(compressor_sm100_r128, "_COMPILED", {})
    monkeypatch.setattr(
        compressor_sm100_r128,
        "_compile_bwd_r128_fused",
        lambda key, ratio, d, c, schedule: compiled.add(schedule),
    )
    device = torch.device("musa", torch.musa.current_device())
    compressor_sm100_r128.precompile_bwd_r128(128, head_dim, coff, device)
    expected = {
        ("rowgroup", warps, tile, warps, 4, shards)
        for _, tile, warps, shards
        in compressor_sm100_r128._ROWGROUP_BWD_ROW_BUCKETS[(coff, head_dim)]
    } | {
        ("tilewarp", *bucket[1:])
        for bucket
        in compressor_sm100_r128._TILEWARP_BWD_ROW_BUCKETS[(coff, head_dim)]
    }
    assert compiled == expected


def test_r128_backward_precompile_reuses_warm_schedule_during_capture(monkeypatch):
    _require_musa_tilelang()
    compiled = {}
    builds = []
    compile_calls = []
    main_program = object()
    reduce_program = object()
    main_kernel = object()
    reduce_kernel = object()

    def fake_rowgroup(*args, **kwargs):
        builds.append(("rowgroup", args, kwargs))
        return main_program

    def fake_tilewarp(*args, **kwargs):
        builds.append(("tilewarp", args, kwargs))
        return main_program

    def fake_reduce(*args):
        builds.append(("reduce", args))
        return reduce_program

    def fake_compile(program, *, target, pass_configs, compile_flags):
        compile_calls.append((program, target, pass_configs, compile_flags))
        return main_kernel if program is main_program else reduce_kernel

    monkeypatch.setattr(compressor_sm100_r128, "_COMPILED", compiled)
    monkeypatch.setattr(compressor_sm100_r128, "_build_backward_program_r128_fused_ablation", fake_rowgroup)
    monkeypatch.setattr(compressor_sm100_r128, "_build_backward_program_r128_fused_tilewarp", fake_tilewarp)
    monkeypatch.setattr(compressor_sm100_r128, "_build_backward_ape_reduce_shards_r128", fake_reduce)
    monkeypatch.setattr(compressor_sm100_r128.tilelang, "compile", fake_compile)
    monkeypatch.setattr(torch.musa, "is_current_stream_capturing", lambda: False)
    device = torch.device("musa", torch.musa.current_device())

    for coff, head_dim, rows in ((1, 128, 128), (2, 512, 8192)):
        compressor_sm100_r128.precompile_bwd_r128(128, head_dim, coff, device, nb_total=rows)
    assert [entry[0] for entry in builds if entry[0] != "reduce"] == [
        fused_backward_schedule_r128(128, d, coff, rows)[0]
        for coff, d, rows in ((1, 128, 128), (2, 512, 8192))
    ]
    assert len(compile_calls) == 4
    assert all(call[1:] == ("musa", compressor_sm100_r128._PASS_CONFIGS,
                             compressor_sm100_r128._COMPILE_FLAGS)
               for call in compile_calls)
    assert all(value == (main_kernel, reduce_kernel) for value in compiled.values())

    monkeypatch.setattr(torch.musa, "is_current_stream_capturing", lambda: True)
    compressor_sm100_r128.precompile_bwd_r128(128, 128, 1, device, nb_total=128)
    assert len(compile_calls) == 4
    with pytest.raises(RuntimeError, match="MUSA graph capture"):
        compressor_sm100_r128.precompile_bwd_r128(128, 512, 1, device, nb_total=1024)
    assert len(compile_calls) == 4


@pytest.mark.parametrize(
    "coff,head_dim,rows,expected",
    [pytest.param(*item, id=f"c{item[0]}d{item[1]}-n{item[2]}") for item in _R128_EXPECTED_SCHEDULES],
)
def test_r128_envelope_execution(coff, head_dim, rows, expected):
    _require_musa_tilelang()
    assert compressor_sm100_r128._forward_kernel_schedule_r128(128, head_dim, coff, rows, 2) == expected
    kv, score, ape, cu, cuc, total_comp = _make_inputs((rows * 128,), head_dim, 128, coff, device="musa")
    assert total_comp == rows
    grad_out = _make_grad_out(total_comp, head_dim, device="musa")
    first = _run_fused_fwd_bwd(kv, score, ape, cu, cuc, total_comp, 128, head_dim, coff, grad_out)
    second = _run_fused_fwd_bwd(kv, score, ape, cu, cuc, total_comp, 128, head_dim, coff, grad_out)
    for index in range(3):
        assert torch.equal(first[index], second[index])
    torch.testing.assert_close(first[3], second[3], rtol=0, atol=1e-3)

    fp32_ref = _run_eager_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        total_comp,
        128,
        head_dim,
        coff,
        grad_out,
        "fp32",
    )
    _assert_grads_vs_fp32(first[1], first[2], fp32_ref[1], fp32_ref[2], 128)
    _assert_r128_forward_vs_fp32(first[0], fp32_ref[0])
    assert (first[3] - fp32_ref[3]).abs().max().item() <= 1e-3


# ---------------------------------------------------------------------------
# Static capacity, empty packs and complete gradient-buffer writes
# ---------------------------------------------------------------------------


_PADDING_SHAPES = [
    ((3, 515, 1024, 129), 128, 4, 2, 8),
    ((1023, 2048, 509), 128, 4, 2, 8),
    ((3, 515, 1024, 129), 128, 4, 1, 8),
    ((1023, 2048, 509), 128, 4, 1, 8),
    ((1, 515, 1024, 129), 128, 2, 2, 8),
    ((1023, 2048, 509), 128, 2, 2, 8),
    ((1, 1), 128, 2, 2, 8),
    ((1, 515, 1024, 129), 128, 2, 1, 8),
    ((1023, 2048, 509), 128, 2, 1, 8),
    ((1, 1), 128, 2, 1, 8),
]


@pytest.mark.parametrize("lens,head_dim,ratio,coff,pad", _PADDING_SHAPES)
def test_static_capacity_padding(lens, head_dim, ratio, coff, pad):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_true = _make_inputs(lens, head_dim, ratio, coff, device="musa")
    capacity = total_true + pad
    grad_out = _make_grad_out(capacity, head_dim, device="musa")
    grad_out_zero_pad = grad_out.clone()
    grad_out_zero_pad[total_true:] = 0

    fused = _run_fused_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        capacity,
        ratio,
        head_dim,
        coff,
        grad_out,
    )
    ref = _run_eager_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        capacity,
        ratio,
        head_dim,
        coff,
        grad_out_zero_pad,
        "fp32",
    )
    assert int((fused[0] != ref[0]).sum().item()) <= max(1, int(0.001 * fused[0].numel()))
    assert (fused[0].float() - ref[0].float()).abs().max().item() <= 1.6e-2
    _assert_grads_vs_fp32(fused[1], fused[2], ref[1], ref[2], ratio)
    assert (fused[3] - ref[3]).abs().max().item() <= 1e-3

    fused_zero = _run_fused_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        capacity,
        ratio,
        head_dim,
        coff,
        grad_out_zero_pad,
    )
    assert torch.equal(fused[1], fused_zero[1])
    assert torch.equal(fused[2], fused_zero[2])


@pytest.mark.parametrize("coff", [1, 2])
def test_backward_wrapper_zeros_when_no_blocks(coff):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs((3, 2, 1), 128, 4, coff, device="musa")
    assert total_comp == 0
    grad_out = torch.empty((0, 128), dtype=torch.bfloat16, device="musa")
    grads = csa_compressor_backward_wrapper(kv, score, ape, cu, cuc, grad_out, ratio=4, head_dim=128, coff=coff)
    assert grads["grad_kv"].abs().sum().item() == 0
    assert grads["grad_score"].abs().sum().item() == 0
    assert grads["grad_ape"].abs().sum().item() == 0


def _never_consumed_mask(lens, total_tokens, head_dim, ratio, coff):
    width = coff * head_dim
    mask = torch.zeros(total_tokens, width, dtype=torch.bool)
    position = 0
    for segment_length in lens:
        blocks = segment_length // ratio
        if blocks == 0:
            mask[position : position + segment_length] = True
        else:
            mask[position + blocks * ratio : position + segment_length] = True
            if coff == 2:
                mask[
                    position + (blocks - 1) * ratio : position + blocks * ratio,
                    :head_dim,
                ] = True
        position += segment_length
    mask[position:] = True
    return mask


_CANARY_SHAPES = [
    ((2048,), 128, 4, 2, 0, 0),
    ((1023, 2048, 509), 128, 4, 2, 0, 0),
    ((3, 515, 1024, 129), 128, 4, 2, 0, 0),
    ((5, 6, 7), 128, 4, 2, 0, 0),
    ((1023, 2048, 509), 512, 4, 2, 0, 0),
    ((3, 515, 1024, 129), 128, 4, 2, 8, 0),
    ((1023, 2048, 509), 128, 4, 2, 8, 0),
    ((1023, 2048, 509), 128, 4, 2, 0, 37),
    ((3, 515, 1024, 129), 128, 4, 2, 8, 21),
    ((2048,), 128, 4, 1, 0, 0),
    ((1023, 2048, 509), 128, 4, 1, 0, 0),
    ((3, 515, 1024, 129), 128, 4, 1, 0, 0),
    ((5, 6, 7), 128, 4, 1, 0, 0),
    ((1023, 2048, 509), 512, 4, 1, 0, 0),
    ((3, 515, 1024, 129), 128, 4, 1, 8, 0),
    ((1023, 2048, 509), 128, 4, 1, 8, 0),
    ((1023, 2048, 509), 128, 4, 1, 0, 37),
    ((3, 515, 1024, 129), 128, 4, 1, 8, 21),
    ((2048,), 128, 2, 2, 0, 0),
    ((1023, 2048, 509), 128, 2, 2, 0, 0),
    ((1, 515, 1024, 129), 128, 2, 2, 0, 0),
    ((1, 2, 3), 128, 2, 2, 0, 0),
    ((1023, 2048, 509), 512, 2, 2, 0, 0),
    ((1, 515, 1024, 129), 128, 2, 2, 8, 0),
    ((1023, 2048, 509), 128, 2, 2, 8, 0),
    ((1023, 2048, 509), 128, 2, 2, 0, 37),
    ((1, 515, 1024, 129), 128, 2, 2, 8, 21),
    ((2048,), 128, 2, 1, 0, 0),
    ((1023, 2048, 509), 128, 2, 1, 0, 0),
    ((1, 515, 1024, 129), 128, 2, 1, 0, 0),
    ((1, 2, 3), 128, 2, 1, 0, 0),
    ((1023, 2048, 509), 512, 2, 1, 0, 0),
    ((1, 515, 1024, 129), 128, 2, 1, 8, 0),
    ((1023, 2048, 509), 128, 2, 1, 8, 0),
    ((1023, 2048, 509), 128, 2, 1, 0, 37),
    ((1, 515, 1024, 129), 128, 2, 1, 8, 21),
    ((1023, 2048, 509), 128, 128, 1, 0, 0),
    ((127, 8192, 0, 129, 128, 3, 515, 1024), 128, 128, 1, 8, 21),
    ((127, 8192, 0, 129, 128, 3, 515, 1024), 128, 128, 2, 8, 21),
    ((2048, 509), 512, 128, 2, 8, 21),
]


@pytest.mark.parametrize("lens,head_dim,ratio,coff,pad,token_pad", _CANARY_SHAPES)
def test_backward_fills_uninitialized_buffers(lens, head_dim, ratio, coff, pad, token_pad):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_true = _make_inputs(lens, head_dim, ratio, coff, device="musa")
    total_comp = total_true + pad
    width = coff * head_dim
    generator = torch.Generator(device="cpu").manual_seed(11)
    if token_pad:
        extra_kv = torch.randn(token_pad, width, generator=generator, dtype=torch.float32).bfloat16().to("musa")
        extra_score = torch.randn(token_pad, width, generator=generator, dtype=torch.float32).bfloat16().to("musa")
        kv = torch.cat((kv, extra_kv))
        score = torch.cat((score, extra_score))
    grad_out = _make_grad_out(total_comp, head_dim, seed=11, device="musa")
    sample_out = torch.empty((total_comp, head_dim), dtype=torch.bfloat16, device="meta")
    api = CSACompressorBackward(kv, score, ape, cu, cuc, sample_out, ratio=ratio, coff=coff)
    assert api.check_support()
    api.compile()

    def run(poison):
        grad_kv = torch.empty_like(kv)
        grad_score = torch.empty_like(score)
        if poison:
            grad_kv.fill_(float("nan"))
            grad_score.fill_(float("nan"))
        else:
            grad_kv.zero_()
            grad_score.zero_()
        grad_ape = torch.zeros_like(ape)
        api.execute(
            kv,
            score,
            ape,
            cu,
            cuc,
            grad_out,
            grad_kv,
            grad_score,
            grad_ape,
        )
        return grad_kv, grad_score, grad_ape

    zero_run = run(False)
    poison_run = run(True)
    assert not torch.isnan(poison_run[0]).any()
    assert not torch.isnan(poison_run[1]).any()
    assert torch.equal(poison_run[0], zero_run[0])
    assert torch.equal(poison_run[1], zero_run[1])
    torch.testing.assert_close(poison_run[2], zero_run[2], rtol=0, atol=1e-3)

    mask = _never_consumed_mask(lens, kv.shape[0], head_dim, ratio, coff).to("musa")
    assert torch.equal(poison_run[0][mask], torch.zeros_like(poison_run[0][mask]))
    assert torch.equal(poison_run[1][mask], torch.zeros_like(poison_run[1][mask]))


# ---------------------------------------------------------------------------
# Backward API contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coff", [1, 2])
def test_backward_rejects_deterministic_mode(coff):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs((16,), 128, 4, coff, device="musa")
    grad_out = _make_grad_out(total_comp, 128, device="musa")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        with pytest.raises(RuntimeError, match="not deterministic"):
            csa_compressor_backward_wrapper(kv, score, ape, cu, cuc, grad_out, ratio=4, coff=coff)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.parametrize("coff", [1, 2])
def test_backward_warns_in_warn_only_deterministic_mode(coff):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs((16,), 128, 4, coff, device="musa")
    grad_out = _make_grad_out(total_comp, 128, device="musa")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with pytest.warns(RuntimeWarning, match="not deterministic") as records:
            csa_compressor_backward_wrapper(kv, score, ape, cu, cuc, grad_out, ratio=4, coff=coff)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
    assert len(records) == 1


@pytest.mark.parametrize("coff", [1, 2])
def test_class_api_matches_wrapper(coff):
    _require_musa_tilelang()
    ratio, head_dim = 4, 128
    kv, score, ape, cu, cuc, total_comp = _make_inputs((1023, 509), head_dim, ratio, coff, device="musa")
    grad_out = _make_grad_out(total_comp, head_dim, device="musa")
    wrapped = _run_fused_fwd_bwd(
        kv,
        score,
        ape,
        cu,
        cuc,
        total_comp,
        ratio,
        head_dim,
        coff,
        grad_out,
    )

    sample_out = torch.empty((total_comp, head_dim), dtype=torch.bfloat16, device="meta")
    fwd = CSACompressorForward(kv, score, ape, cu, cuc, sample_out, ratio=ratio, coff=coff)
    bwd = CSACompressorBackward(kv, score, ape, cu, cuc, sample_out, ratio=ratio, coff=coff)
    assert fwd.check_support() and bwd.check_support()
    fwd.compile()
    bwd.compile()
    out = torch.empty_like(wrapped[0])
    grad_kv = torch.empty_like(kv)
    grad_score = torch.empty_like(score)
    grad_ape = torch.zeros_like(ape)
    fwd.execute(kv, score, ape, cu, cuc, out)
    bwd.execute(
        kv,
        score,
        ape,
        cu,
        cuc,
        grad_out,
        grad_kv,
        grad_score,
        grad_ape,
    )
    assert torch.equal(out, wrapped[0])
    assert torch.equal(grad_kv, wrapped[1])
    assert torch.equal(grad_score, wrapped[2])
    torch.testing.assert_close(grad_ape, wrapped[3], rtol=0, atol=1e-3)


def test_backward_grad_ape_zeroing_contract():
    _require_musa_tilelang()
    ratio, head_dim, coff = 128, 128, 1
    kv, score, ape, cu, cuc, total_comp = _make_inputs((640, 259), head_dim, ratio, coff, device="musa")
    grad_out = _make_grad_out(total_comp, head_dim, device="musa")
    sample_out = torch.empty((total_comp, head_dim), dtype=torch.bfloat16, device="meta")
    api = CSACompressorBackward(kv, score, ape, cu, cuc, sample_out, ratio=ratio, coff=coff)
    assert api.check_support()
    api.compile()
    grad_kv = torch.empty_like(kv)
    grad_score = torch.empty_like(score)
    reference = torch.zeros_like(ape)
    api.execute(
        kv,
        score,
        ape,
        cu,
        cuc,
        grad_out,
        grad_kv,
        grad_score,
        reference,
    )
    accumulated = torch.zeros_like(ape)
    for _ in range(2):
        api.execute(
            kv,
            score,
            ape,
            cu,
            cuc,
            grad_out,
            grad_kv,
            grad_score,
            accumulated,
        )
    torch.testing.assert_close(accumulated, 2 * reference, rtol=0, atol=2e-3)

    graph_accumulated = torch.zeros_like(ape)
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        api.execute(
            kv,
            score,
            ape,
            cu,
            cuc,
            grad_out,
            grad_kv,
            grad_score,
            graph_accumulated,
        )
    graph.replay()
    graph.replay()
    torch.musa.synchronize()
    torch.testing.assert_close(graph_accumulated, 2 * reference, rtol=0, atol=2e-3)

    fresh = csa_compressor_backward_wrapper(
        kv,
        score,
        ape,
        cu,
        cuc,
        grad_out,
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
    )["grad_ape"]
    torch.testing.assert_close(fresh, reference, rtol=0, atol=1e-3)


# ---------------------------------------------------------------------------
# MUSA graph capture
# ---------------------------------------------------------------------------


def _run_musa_graph_capture_case(ratio, head_dim, coff, lens, replay_lens):
    if not hasattr(torch.musa, "MUSAGraph") or not hasattr(torch.musa, "graph"):
        pytest.skip("MUSA graph capture is unavailable")

    kv, score, ape, cu, cuc, total_true = _make_inputs(lens, head_dim, ratio, coff, device="musa")
    capacity = total_true + 8
    total_tokens = kv.shape[0]
    kv_static = kv.clone()
    score_static = score.clone()
    ape_static = ape.clone()
    grad_out_static = torch.zeros((capacity, head_dim), dtype=torch.bfloat16, device="musa")
    grad_out_static[:total_true] = _make_grad_out(total_true, head_dim, device="musa")

    def fused_fwd_bwd():
        out = csa_compressor_forward_wrapper(
            kv_static,
            score_static,
            ape_static,
            cu,
            cuc,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
            total_comp=capacity,
        )["out"]
        grads = csa_compressor_backward_wrapper(
            kv_static,
            score_static,
            ape_static,
            cu,
            cuc,
            grad_out_static,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
        )
        return out, grads

    # Compile both configurations before capture, matching the public contract.
    side = torch.musa.Stream()
    side.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(side):
        fused_fwd_bwd()
    torch.musa.current_stream().wait_stream(side)
    torch.musa.synchronize()

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        out_captured, grads_captured = fused_fwd_bwd()
    grad_kv_captured = grads_captured["grad_kv"]
    grad_score_captured = grads_captured["grad_score"]
    grad_ape_captured = grads_captured["grad_ape"]

    graph.replay()
    torch.musa.synchronize()
    out_direct, grads_direct = fused_fwd_bwd()
    torch.musa.synchronize()
    assert torch.equal(out_captured, out_direct)
    assert torch.equal(grad_kv_captured, grads_direct["grad_kv"])
    assert torch.equal(grad_score_captured, grads_direct["grad_score"])
    torch.testing.assert_close(grad_ape_captured, grads_direct["grad_ape"], rtol=0, atol=1e-3)

    # Keep allocations static while changing device-side metadata to a smaller
    # true token/row count, then validate every output after replay.
    kv_new, score_new, ape_new, cu_new, cuc_new, total_new = _make_inputs(replay_lens, head_dim, ratio, coff, seed=99, device="musa")
    new_tokens = sum(replay_lens)
    assert new_tokens <= total_tokens and total_new <= capacity
    kv_static.zero_()
    score_static.zero_()
    kv_static[:new_tokens].copy_(kv_new)
    score_static[:new_tokens].copy_(score_new)
    ape_static.copy_(ape_new)
    cu.copy_(cu_new)
    cuc.copy_(cuc_new)
    grad_out_static.zero_()
    grad_out_static[:total_new] = _make_grad_out(total_new, head_dim, seed=99, device="musa")

    graph.replay()
    torch.musa.synchronize()
    ref = _run_eager_fwd_bwd(
        kv_static[:new_tokens],
        score_static[:new_tokens],
        ape_static,
        cu,
        cuc,
        capacity,
        ratio,
        head_dim,
        coff,
        grad_out_static,
        "fp32",
    )
    _assert_grads_vs_fp32(
        grad_kv_captured[:new_tokens],
        grad_score_captured[:new_tokens],
        ref[1],
        ref[2],
        ratio,
    )
    assert torch.count_nonzero(grad_kv_captured[new_tokens:]).item() == 0
    assert torch.count_nonzero(grad_score_captured[new_tokens:]).item() == 0
    if ratio == 128:
        _assert_r128_forward_vs_fp32(out_captured, ref[0])
    else:
        assert int((out_captured != ref[0]).sum().item()) <= max(1, int(0.001 * out_captured.numel()))
        assert (out_captured.float() - ref[0].float()).abs().max().item() <= 1.6e-2
    assert (grad_ape_captured - ref[3]).abs().max().item() <= 1e-3

    replay_direct, replay_grads_direct = fused_fwd_bwd()
    torch.musa.synchronize()
    assert torch.equal(out_captured, replay_direct)
    assert torch.equal(grad_kv_captured, replay_grads_direct["grad_kv"])
    assert torch.equal(grad_score_captured, replay_grads_direct["grad_score"])


@pytest.mark.parametrize("coff", [1, 2])
def test_musa_graph_capture(coff):
    _require_musa_tilelang()
    _run_musa_graph_capture_case(
        ratio=4,
        head_dim=128,
        coff=coff,
        lens=(512, 256),
        replay_lens=(384, 128),
    )


@pytest.mark.parametrize("coff", [1, 2])
@pytest.mark.filterwarnings("ignore:The MUSA Graph is empty.*:UserWarning")
def test_musa_graph_capture_rejects_cold_jit(coff, monkeypatch):
    _require_musa_tilelang()
    if not hasattr(torch.musa, "MUSAGraph") or not hasattr(torch.musa, "graph"):
        pytest.skip("MUSA graph capture is unavailable")

    # Isolate this cold-cache check from other tests.
    monkeypatch.setattr(compressor_sm100, "_COMPILED", {})
    cold_head_dim = 512
    kv, score, ape, cu, cuc, total_comp = _make_inputs((256,), cold_head_dim, 4, coff, device="musa")
    cold_graph = torch.musa.MUSAGraph()
    with pytest.raises(RuntimeError, match="MUSA graph capture"):
        with torch.musa.graph(cold_graph):
            csa_compressor_forward_wrapper(
                kv,
                score,
                ape,
                cu,
                cuc,
                ratio=4,
                head_dim=cold_head_dim,
                coff=coff,
                total_comp=total_comp,
            )
    torch.musa.synchronize()
    assert torch.ones(8, device="musa").sum().item() == 8


@pytest.mark.parametrize(
    "head_dim,coff,lens,replay_lens",
    [
        (128, 1, (640, 259), (384, 128)),
        (512, 2, (128 * 300,), (128 * 290,)),
    ],
)
def test_musa_graph_capture_r128(head_dim, coff, lens, replay_lens):
    _require_musa_tilelang()
    _run_musa_graph_capture_case(
        ratio=128,
        head_dim=head_dim,
        coff=coff,
        lens=lens,
        replay_lens=replay_lens,
    )


# ---------------------------------------------------------------------------
# Runtime hazards: alignment, explicit streams and multiple devices
# ---------------------------------------------------------------------------


def test_explicit_stream_input_lifetime():
    _require_musa_tilelang()
    if not hasattr(torch.musa, "_sleep"):
        pytest.skip("torch.musa._sleep is unavailable")

    head_dim, ratio, coff = 128, 4, 2
    lens = (4096,)
    kv, score, ape, cu, cuc, total_comp = _make_inputs(lens, head_dim, ratio, coff, device="musa")
    expected = csa_compressor_forward_wrapper(
        kv.clone(),
        score.clone(),
        ape.clone(),
        cu.clone(),
        cuc.clone(),
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
        total_comp=total_comp,
    )["out"]
    torch.musa.synchronize()

    side = torch.musa.Stream()
    with torch.musa.stream(side):
        torch.musa._sleep(int(5e8))
    out = csa_compressor_forward_wrapper(
        kv,
        score,
        ape,
        cu,
        cuc,
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
        total_comp=total_comp,
        stream=side,
    )["out"]
    del kv, score, ape, cu, cuc
    width = coff * head_dim
    junk = [torch.full((sum(lens), width), 7.0, device="musa", dtype=torch.bfloat16) for _ in range(4)]
    junk.append(torch.full((ratio, width), 7.0, device="musa", dtype=torch.float32))
    junk.append(torch.full((64,), 7, device="musa", dtype=torch.int32))
    torch.musa.synchronize()
    assert torch.equal(out, expected)


def test_multi_device_launch():
    _require_musa_tilelang()
    if torch.musa.device_count() < 2:
        pytest.skip("needs at least two visible MUSA devices")
    if torch.musa.get_device_capability(1)[0] < 3:
        pytest.skip("second MUSA device is below mp_30")

    original_device = torch.musa.current_device()
    try:
        torch.musa.set_device(0)
        head_dim, ratio, coff = 128, 4, 2
        kv, score, ape, cu, cuc, total_comp = _make_inputs((1024,), head_dim, ratio, coff, device="musa:1")
        grad_out = _make_grad_out(total_comp, head_dim, device="musa:1")
        assert torch.musa.current_device() == 0
        foreign = _run_fused_fwd_bwd(
            kv,
            score,
            ape,
            cu,
            cuc,
            total_comp,
            ratio,
            head_dim,
            coff,
            grad_out,
        )
        with torch.musa.device(1):
            torch.musa.synchronize()
            native = _run_fused_fwd_bwd(
                kv,
                score,
                ape,
                cu,
                cuc,
                total_comp,
                ratio,
                head_dim,
                coff,
                grad_out,
            )
            torch.musa.synchronize()
        assert foreign[0].device == torch.device("musa", 1)
        assert foreign[0].abs().sum().item() > 0
        for foreign_tensor, native_tensor in zip(foreign[:3], native[:3]):
            assert torch.equal(foreign_tensor, native_tensor)
        torch.testing.assert_close(foreign[3], native[3], rtol=0, atol=1e-3)
    finally:
        torch.musa.set_device(original_device)


def _assert_r128_forward_vs_fp32(actual, expected):
    """Bound BF16 output error from online exp2 versus eager FP32 softmax."""
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.01, atol=0.001)


@pytest.mark.parametrize("ratio", [2, 4])
@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("coff", [1, 2])
@pytest.mark.parametrize("num_offsets", [None, 2, 4])
def test_short_forward_dispatch_uses_rowgroup(ratio, head_dim, coff, num_offsets):
    for rows in (None, 4, 2048, 8192, 32768):
        strategy = compressor_sm100._forward_strategy(
            ratio, head_dim, coff, rows, num_offsets
        )
        assert strategy.startswith("rowgroup_")
        assert (strategy[9] in ("f", "x")) == (num_offsets == 2)


def test_r4_c2_large_forward_uses_retained_rowgroup_tile():
    assert compressor_sm100._forward_strategy(4, 128, 2, 344064, 2) == "rowgroup_x8"
    assert compressor_sm100._forward_strategy(4, 512, 2, 86016, 2) == "rowgroup_x2"
    for dim, rows, strategy in (
        (128, 3072, "rowgroup_f1"),
        (128, 4096, "rowgroup_x2"),
        (128, 262143, "rowgroup_x2"),
        (128, 262144, "rowgroup_x8"),
        (512, 512, "rowgroup_f1"),
        (512, 767, "rowgroup_f1"),
        (512, 768, "rowgroup_x2"),
    ):
        assert compressor_sm100._forward_strategy(4, dim, 2, rows, 2) == strategy
    for ratio, head_dim, coff in ((3, 128, 2), (4, 256, 2), (4, 128, 3)):
        with pytest.raises(ValueError, match="unsupported forward configuration"):
            compressor_sm100._forward_strategy(ratio, head_dim, coff, 8192, 2)


@pytest.mark.parametrize(
    "head_dim,lens,padding",
    [
        (128, (16384,), 0),
        (128, (32768,), 0),
        (512, (16384,), 0),
        (512, (32768,), 0),
        (128, (8193, 8192, 1), 2),
        (512, (8193, 8192, 1), 2),
    ],
)
def test_r4_c2_forward_large_row_dispatch_matches_reference(
    head_dim, lens, padding
):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, total_comp = _make_inputs(
        lens, head_dim, 4, 2, device="musa"
    )
    total_comp += padding
    got = csa_compressor_forward_wrapper(
        kv,
        score,
        ape,
        cu,
        cuc,
        ratio=4,
        head_dim=head_dim,
        coff=2,
        total_comp=total_comp,
    )["out"]
    want = _reference_forward(
        kv, score, ape, cu, cuc, total_comp, 4, head_dim, 2, mode="fp32"
    )
    delta = (got.float() - want.float()).abs()
    assert delta.max().item() <= 1.6e-2
    assert int((got != want).sum().item()) <= max(1, int(0.001 * got.numel()))


def test_r128_precompile_reuses_warm_schedule_during_capture(monkeypatch):
    _require_musa_tilelang()
    compiled = {}
    builds = []
    compile_calls = []
    program = object()
    kernel = object()

    def fake_tma(head_dim, ape_tile_tokens, *, row_tile, coff, single_segment):
        builds.append(("tma", head_dim, ape_tile_tokens, row_tile, coff, single_segment))
        return program

    def fake_tilewarp(head_dim, *, rows_per_cta, ape_tile_tokens,
                      warps_per_cta, chunk_tokens, coff, single_segment):
        builds.append(("tilewarp", head_dim, rows_per_cta, ape_tile_tokens,
                       warps_per_cta, chunk_tokens, coff, single_segment))
        return program

    def fake_compile(actual_program, *, target, pass_configs, compile_flags):
        compile_calls.append((actual_program, target, pass_configs, compile_flags))
        return kernel

    monkeypatch.setattr(compressor_sm100_r128, "_COMPILED", compiled)
    monkeypatch.setattr(compressor_sm100_r128, "_build_forward_program_r128_rowgroup_ape_tma", fake_tma)
    monkeypatch.setattr(compressor_sm100_r128, "_build_forward_program_r128_tilewarp", fake_tilewarp)
    monkeypatch.setattr(compressor_sm100_r128.tilelang, "compile", fake_compile)
    monkeypatch.setattr(torch.musa, "is_current_stream_capturing", lambda: False)
    device = torch.device("musa", torch.musa.current_device())

    compressor_sm100_r128.precompile_fwd_r128(128, 128, 1, device, nb_total=1)
    assert set(builds) == {
        ("tilewarp", 128, 1, 16, 4, 4, 1, True),
        ("tilewarp", 128, 1, 16, 4, 4, 1, False),
    }
    assert compile_calls == [
        (program, "musa", compressor_sm100_r128._PASS_CONFIGS,
         compressor_sm100_r128._COMPILE_FLAGS)
    ] * 2
    assert next(iter(compiled.values())) is kernel

    monkeypatch.setattr(torch.musa, "is_current_stream_capturing", lambda: True)
    compressor_sm100_r128.precompile_fwd_r128(128, 128, 1, device, nb_total=1)
    assert len(builds) == 2
    assert len(compile_calls) == 2

    with pytest.raises(RuntimeError, match="MUSA graph capture"):
        compressor_sm100_r128.precompile_fwd_r128(128, 128, 1, device, nb_total=513)
    assert len(builds) == 2
    assert len(compile_calls) == 2


@pytest.mark.parametrize("coff,head_dim", [(1, 128), (1, 512), (2, 128), (2, 512)])
def test_r128_default_precompile_covers_every_forward_bucket(monkeypatch, coff, head_dim):
    _require_musa_tilelang()
    compiled = set()
    monkeypatch.setattr(compressor_sm100_r128, "_COMPILED", {})
    monkeypatch.setattr(
        compressor_sm100_r128,
        "_compile_fwd_r128",
        lambda key, ratio, d, c, schedule: compiled.add(schedule),
    )
    device = torch.device("musa", torch.musa.current_device())
    compressor_sm100_r128.precompile_fwd_r128(128, head_dim, coff, device)
    expected = {
        compressor_sm100_r128._forward_kernel_schedule_r128(
            128, head_dim, coff, rows, num_offsets
        )
        for rows in (*range(1, 4098), 8191, 8192)
        for num_offsets in (2, 3)
    }
    assert compiled == expected


def test_r128_forward_dispatch_for_small_and_segmented_packs():
    select = compressor_sm100_r128._forward_kernel_schedule_r128
    assert select(128, 128, 1, 4095, 2) == ("tma", 16, 64, True)
    assert select(128, 128, 1, 4096, 2) == ("tma", 8, 32, True)
    assert select(128, 128, 1, 8192, 2) == ("tma", 8, 32, True)
    assert select(128, 512, 1, 8192, 2) == ("tma", 16, 64, True)
    assert select(128, 512, 1, 4095, 2) == ("tma", 8, 32, True)
    assert select(128, 512, 1, 4096, 2) == ("tma", 4, 16, True)
    assert select(128, 512, 1, 8191, 2) == ("tma", 4, 16, True)
    assert select(128, 128, 2, 1024, 2) == ("tilewarp", 1, 16, 4, 4, True)
    assert select(128, 512, 2, 160, 2) == ("tma", 4, 16, True)
    assert select(128, 512, 2, 161, 2) == ("tma", 16, 64, True)
    assert select(128, 512, 2, 512, 2) == ("tma", 16, 64, True)
    assert select(128, 128, 1, 32, 2) == ("tilewarp", 1, 16, 4, 4, True)
    assert select(128, 128, 1, 256, 3) == ("tilewarp", 1, 16, 4, 4, False)
    assert select(128, 128, 1, 514, 3) == ("tma", 4, 16, False)
    assert select(128, 128, 1, 768, 3) == ("tma", 16, 64, False)
    assert select(128, 512, 2, 32, 4) == ("tilewarp", 1, 32, 4, 8, False)
    assert select(128, 512, 2, 130, 4) == ("tma", 4, 16, False)
    assert select(128, 512, 2, 194, 4) == ("tma", 16, 64, False)
    for coff in (1, 2):
        for dim in (128, 512):
            assert select(128, dim, coff, 8192, 3)[0] == "tma"
            assert select(128, dim, coff, 8192, 3)[-1] is False


@pytest.mark.parametrize(
    "coff,head_dim,rows",
    [(1, 128, 1024), (1, 512, 512), (2, 128, 2048), (2, 512, 512)],
)
def test_r128_forward_rowgroup_production_matches_reference(coff, head_dim, rows):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, valid_rows = _make_inputs(
        (rows * 128,), head_dim, 128, coff, device="musa"
    )
    out = torch.empty((valid_rows, head_dim), dtype=torch.bfloat16, device="musa")
    api = CSACompressorForward(kv, score, ape, cu, cuc, out, ratio=128, coff=coff)
    assert api.check_support()
    api.compile()
    api.execute(kv, score, ape, cu, cuc, out)
    expected = _reference_forward(
        kv, score, ape, cu, cuc, valid_rows, 128, head_dim, coff, mode="fp32"
    )
    _assert_r128_forward_vs_fp32(out, expected)


@pytest.mark.parametrize("coff,head_dim", [(1, 128), (1, 512), (2, 128), (2, 512)])
def test_r128_forward_small_multi_segment_matches_reference(coff, head_dim):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, valid_rows = _make_inputs(
        (8 * 128, 0, 8 * 128, 16 * 128), head_dim, 128, coff, device="musa"
    )
    capacity = valid_rows + 2
    out = torch.empty((capacity, head_dim), dtype=torch.bfloat16, device="musa")
    api = CSACompressorForward(kv, score, ape, cu, cuc, out, ratio=128, coff=coff)
    assert api.check_support()
    api.compile()
    api.execute(kv, score, ape, cu, cuc, out)
    expected = _reference_forward(
        kv, score, ape, cu, cuc, capacity, 128, head_dim, coff, mode="fp32"
    )
    _assert_r128_forward_vs_fp32(out, expected)


@pytest.mark.parametrize("coff", [1, 2])
def test_r128_forward_tma_single_segment_capacity_padding(coff):
    _require_musa_tilelang()
    head_dim, rows = 512, 512
    kv, score, ape, cu, cuc, valid_rows = _make_inputs(
        (rows * 128,), head_dim, 128, coff, device="musa"
    )
    capacity = valid_rows + 2
    out = torch.empty((capacity, head_dim), dtype=torch.bfloat16, device="musa")
    api = CSACompressorForward(kv, score, ape, cu, cuc, out, ratio=128, coff=coff)
    assert api.check_support()
    api.compile()
    api.execute(kv, score, ape, cu, cuc, out)
    expected = _reference_forward(
        kv, score, ape, cu, cuc, capacity, 128, head_dim, coff, mode="fp32"
    )
    _assert_r128_forward_vs_fp32(out, expected)


@pytest.mark.parametrize(
    "coff,head_dim,rows",
    [(1, 128, 32), (1, 128, 256), (1, 512, 16), (1, 512, 64),
     (2, 128, 32), (2, 128, 256), (2, 512, 16), (2, 512, 64),
     (2, 512, 56)],
)
def test_r128_small_row_column_tiles_match_reference(coff, head_dim, rows):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, valid_rows = _make_inputs((rows * 128,), head_dim, 128, coff, device="musa")
    out = torch.empty((valid_rows, head_dim), dtype=torch.bfloat16, device="musa")
    api = CSACompressorForward(kv, score, ape, cu, cuc, out, ratio=128, coff=coff)
    assert api.check_support()
    api.compile()
    api.execute(kv, score, ape, cu, cuc, out)
    expected = _reference_forward(kv, score, ape, cu, cuc, valid_rows, 128, head_dim, coff, mode="fp32")
    _assert_r128_forward_vs_fp32(out, expected)


@pytest.mark.parametrize(
    "head_dim,lens",
    [
        (128, (512 * 128 + 17, 9 * 128 + 3, 0)),
        (512, (160 * 128 + 5, 3 * 128 + 7, 0)),
    ],
)
@pytest.mark.parametrize("coff", [1, 2])
def test_r128_vec4_wide_load_ragged_with_capacity(head_dim, lens, coff):
    _require_musa_tilelang()
    kv, score, ape, cu, cuc, valid_rows = _make_inputs(
        lens, head_dim, 128, coff, device="musa"
    )
    capacity = valid_rows + 2
    out = torch.empty((capacity, head_dim), dtype=torch.bfloat16, device="musa")
    api = CSACompressorForward(kv, score, ape, cu, cuc, out, ratio=128, coff=coff)
    assert api.check_support()
    api.compile()
    api.execute(kv, score, ape, cu, cuc, out)
    expected = _reference_forward(
        kv, score, ape, cu, cuc, capacity, 128, head_dim, coff, mode="fp32"
    )
    _assert_r128_forward_vs_fp32(out, expected)


def test_r128_backward_normal_load_across_segments(monkeypatch):
    """The first row of each segment skips the invalid previous half."""
    _require_musa_tilelang()
    monkeypatch.setattr(compressor_sm100_r128, "_COMPILED", {})
    ratio, head_dim, coff = 128, 512, 2
    kv, score, ape, cu, cuc, total_comp = _make_inputs(
        (3 * ratio, 5 * ratio), head_dim, ratio, coff, device="musa"
    )
    grad_out = _make_grad_out(total_comp, head_dim, device="musa")
    fused = _run_fused_fwd_bwd(
        kv, score, ape, cu, cuc, total_comp, ratio, head_dim, coff, grad_out
    )
    ref = _run_eager_fwd_bwd(
        kv, score, ape, cu, cuc, total_comp, ratio, head_dim, coff, grad_out,
        "fp32",
    )
    _assert_grads_vs_fp32(fused[1], fused[2], ref[1], ref[2], ratio)
    assert (fused[3] - ref[3]).abs().max().item() <= 1e-3
