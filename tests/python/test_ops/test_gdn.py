# Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.

"""Public GDN API: inference, training pairs, routing and old-engine coverage."""

import importlib
import inspect

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F

import mudnn
from mudnn.engines import engine_for_id, engines_for
from mudnn.linear_attention import gated_delta_net
from test_ops.reference_gdn import (
    chunk_gdn_diff,
    naive_recurrent_gdn,
    relative_l2,
    recurrent_gdn_fp64,
    relative_l2_fp64,
    make_gdn_inputs,
    cu_seqlens,
    mp32_skip_reason,
)

DEVICE = "musa"
DIM = 128
REL_TOL = 2e-2
GPU = pytest.mark.skipif(not getattr(torch, "musa", None) or not torch.musa.is_available(), reason="requires MUSA")
_mp32_reason = mp32_skip_reason()
MP32_ONLY = pytest.mark.skipif(_mp32_reason is not None, reason=_mp32_reason or "")
ops = importlib.import_module("mudnn.linear_attention.ops.gdn")
fla = importlib.import_module("mudnn.fla.gated_delta_rule")


# Original training engine through the unified API


def _l2(x):
    xf = x.float()
    return (xf / (xf.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(x.dtype)


def _old_engine_inputs(seqs, heads, dim=DIM, value_heads=None, requires_grad=False, seed=2026):
    """THD inputs for a gated-delta-net call.

    q and k are [T, H, D] and v is [T, HV, D] in bfloat16, g is [T, HV] in
    float32, beta is [T, HV] in bfloat16, and cu_seqlens is int32 [N + 1].
    q and k are l2-normalized, g is a log-space decay, beta lies in (0, 1).
    """
    torch.manual_seed(seed)
    (total, hv) = (sum(seqs), heads if value_heads is None else value_heads)
    make = lambda h, d: torch.randn(total, h, d, device=DEVICE, dtype=torch.bfloat16)
    (q, k) = (_l2(make(heads, dim)) for _ in range(2))
    v = make(hv, dim)
    g = F.logsigmoid(torch.rand(total, hv, device=DEVICE, dtype=torch.float32))
    beta = torch.rand(total, hv, device=DEVICE, dtype=torch.bfloat16)
    tensors = [q, k, v, g, beta]
    if requires_grad:
        tensors = [t.detach().requires_grad_(True) for t in tensors]
    return (*tensors, cu_seqlens(seqs))


def _meta(tensor):
    return None if tensor is None else torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")


@GPU
def test_manifest_offers_the_tilelang_engine():
    engines = engines_for(["GDN"])
    assert {"gdn_tilelang", "gdn_prefill_tilelang", "gdn_decode_tilelang", "gdn_tilelang_v2"} <= {
        e.name for e in engines
    }
    for engine in engines:
        assert engine_for_id(engine.engine_id) is engine


@GPU
def test_unknown_plan_name_is_refused():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128,), 2)
    with pytest.raises(mudnn.cudnnGraphNotSupportedError, match="no 'nope'"):
        gated_delta_net(q, k, v, g, beta, cu, plan_name="nope")


@GPU
def test_old_engine_unsupported_head_dim_reports_every_engine_reason():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128,), 2, dim=96)
    with pytest.raises(mudnn.cudnnGraphNotSupportedError, match="gdn_tilelang:"):
        gated_delta_net(q, k, v, g, beta, cu, plan_name="gdn_tilelang")


@GPU
def test_ragged_batch_reports_a_per_sequence_state():
    q, k, v, g, beta, cu = _old_engine_inputs((128, 192), 2)
    with torch.no_grad():
        out, final = gated_delta_net(q, k, v, g, beta, cu, output_final_state=True)
    assert out.shape == (320, 2, DIM)
    assert final.shape == (2, 2, DIM, DIM)
    assert torch.isfinite(final).all()


@GPU
def test_old_engine_strict_dtypes():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128,), 2)
    with pytest.raises(TypeError, match="bfloat16"):
        gated_delta_net(q.half(), k.half(), v.half(), g, beta.half(), cu, plan_name="gdn_tilelang")
    with pytest.raises(NotImplementedError, match="fp32 g"):
        gated_delta_net(q, k, v, g.half(), beta, cu, plan_name="gdn_tilelang")
    with pytest.raises(NotImplementedError, match="activated beta in the io dtype"):
        gated_delta_net(q, k, v, g, beta.float(), cu, plan_name="gdn_tilelang")


@GPU
def test_old_engine_a_log_without_safe_gate_is_an_error():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128,), 2)
    a_log = torch.zeros(2, device=DEVICE)
    with pytest.raises(ValueError, match="require safe_gate"):
        gated_delta_net(q, k, v, g, beta, cu, a_log=a_log, plan_name="gdn_tilelang")


@GPU
@pytest.mark.parametrize("seqs", [(128, 128), (192,)])
def test_old_engine_forward_matches_the_fp32_recurrence(seqs):
    (q, k, v, g, beta, cu) = _old_engine_inputs(seqs, 4)
    with torch.no_grad():
        (out, state) = gated_delta_net(q, k, v, g, beta, cu, output_final_state=True, plan_name="gdn_tilelang")
    (ref_out, ref_state) = naive_recurrent_gdn(q, k, v, g, beta, cu, output_final_state=True)
    assert relative_l2(ref_out, out) < REL_TOL
    assert relative_l2(ref_state, state.transpose(-1, -2)) < REL_TOL


@GPU
def test_old_engine_forward_with_initial_state():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128, 128), 4)
    h0 = torch.randn(2, 4, DIM, DIM, device=DEVICE, dtype=torch.float32) * 0.1
    with torch.no_grad():
        (out, state) = gated_delta_net(
            q,
            k,
            v,
            g,
            beta,
            cu,
            initial_state=h0.transpose(-1, -2).contiguous(),
            output_final_state=True,
            plan_name="gdn_tilelang",
        )
    (ref_out, ref_state) = naive_recurrent_gdn(q, k, v, g, beta, cu, initial_state=h0, output_final_state=True)
    assert relative_l2(ref_out, out) < REL_TOL
    assert relative_l2(ref_state, state.transpose(-1, -2)) < REL_TOL


@GPU
def test_old_engine_forward_unaligned_length():
    """A sequence length that is not a multiple of the 64-token chunk."""
    (q, k, v, g, beta, cu) = _old_engine_inputs((200,), 4)
    with torch.no_grad():
        (out, _) = gated_delta_net(q, k, v, g, beta, cu, plan_name="gdn_tilelang")
    (ref_out, _) = naive_recurrent_gdn(q, k, v, g, beta, cu)
    assert relative_l2(ref_out, out) < REL_TOL


@GPU
def test_old_engine_forward_ragged_batch():
    seqs = (128, 192)
    (q, k, v, g, beta, cu) = _old_engine_inputs(seqs, 4)
    with torch.no_grad():
        (out, state) = gated_delta_net(q, k, v, g, beta, cu, plan_name="gdn_tilelang")
    (ref_out, _) = naive_recurrent_gdn(q, k, v, g, beta, cu)
    assert state.numel() == 0
    assert relative_l2(ref_out, out) < REL_TOL


@GPU
def test_old_engine_grouped_query_heads():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128,), 2, value_heads=4)
    with torch.no_grad():
        (out, _) = gated_delta_net(q, k, v, g, beta, cu, plan_name="gdn_tilelang")
    repeats = v.shape[1] // q.shape[1]
    grow = lambda t: t.repeat_interleave(repeats, dim=1)
    (ref_out, _) = naive_recurrent_gdn(grow(q), grow(k), v, g, beta, cu)
    assert out.shape == v.shape
    assert relative_l2(ref_out, out) < REL_TOL


@GPU
def test_old_engine_fusion_knobs_match_their_torch_definition():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128,), 4)
    (a_log, dt_bias) = (torch.randn(4, device=DEVICE) * 0.1 for _ in range(2))
    with torch.no_grad():
        (fused, _) = gated_delta_net(
            q,
            k,
            v,
            g,
            beta,
            cu,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=True,
            safe_gate=True,
            a_log=a_log,
            dt_bias=dt_bias,
            plan_name="gdn_tilelang",
        )
        (plain, _) = gated_delta_net(
            _l2(q),
            _l2(k),
            v,
            -a_log.float().exp() * F.softplus(g.float() + dt_bias.float()),
            (2.0 * torch.sigmoid(beta.float())).bfloat16(),
            cu,
            plan_name="gdn_tilelang",
        )
    assert torch.equal(fused, plain)


@GPU
@pytest.mark.parametrize("seqs", [(128, 128), (128, 192)])
def test_old_engine_backward_matches_the_fp32_oracle(seqs):
    (q, k, v, g, beta, cu) = _old_engine_inputs(seqs, 4, requires_grad=True)
    (out, _) = gated_delta_net(q, k, v, g, beta, cu, plan_name="gdn_tilelang")
    out.float().square().sum().backward()
    refs = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    (ref_out, _) = chunk_gdn_diff(*refs, cu)
    ref_out.square().sum().backward()
    for name, got, want in zip("qkvgb", (q, k, v, g, beta), refs):
        assert relative_l2(want.grad, got.grad) < REL_TOL, f"d{name}"


@GPU
def test_old_engine_backward_through_the_state():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128, 128), 4, requires_grad=True)
    h0 = (torch.randn(2, 4, DIM, DIM, device=DEVICE, dtype=torch.float32) * 0.1).requires_grad_(True)
    (out, state) = gated_delta_net(
        q,
        k,
        v,
        g,
        beta,
        cu,
        initial_state=h0.transpose(-1, -2).contiguous(),
        output_final_state=True,
        plan_name="gdn_tilelang",
    )
    (out.float().square().sum() + state.square().sum()).backward()
    refs = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta, h0)]
    (ref_out, ref_state) = chunk_gdn_diff(*refs[:5], cu, initial_state=refs[5], output_final_state=True)
    (ref_out.square().sum() + ref_state.square().sum()).backward()
    for name, got, want in zip(("q", "k", "v", "g", "beta", "h0"), (q, k, v, g, beta, h0), refs):
        assert relative_l2(want.grad, got.grad) < REL_TOL, f"d{name}"


@GPU
def test_old_engine_head_dim_64():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128,), 2, dim=64)
    with torch.no_grad():
        (out, _) = gated_delta_net(q, k, v, g, beta, cu, plan_name="gdn_tilelang")
    (ref_out, _) = naive_recurrent_gdn(q, k, v, g, beta, cu)
    assert relative_l2(ref_out, out) < REL_TOL


@GPU
def test_old_engine_fake_shapes_match_the_real_op():
    """The registered fake matches the real op's output shapes and dtypes."""
    (q, k, v, g, beta, cu) = _old_engine_inputs((128, 128), 4)
    h0 = torch.zeros(2, 4, DIM, DIM, device=DEVICE, dtype=torch.float32)
    args = (q, k, v, g, beta, cu, DIM ** (-0.5))
    with torch.no_grad():
        real = torch.ops.mudnn.gated_delta_net_fwd(*args, initial_state=h0, output_final_state=True)
    fake = torch.ops.mudnn.gated_delta_net_fwd(
        *(_meta(t) for t in args[:6]), args[6], initial_state=_meta(h0), output_final_state=True
    )
    assert [(t.shape, t.dtype) for t in real] == [(t.shape, t.dtype) for t in fake]


@GPU
def test_old_engine_inference_path_matches_the_training_path():
    (q, k, v, g, beta, cu) = _old_engine_inputs((128, 128), 4)
    grads = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    with torch.no_grad():
        (inference, _) = gated_delta_net(q, k, v, g, beta, cu, plan_name="gdn_tilelang")
    (training, _) = gated_delta_net(*grads, cu, plan_name="gdn_tilelang")
    assert torch.equal(inference, training)


@GPU
def test_old_engine_backward_op_matches_the_autograd_path():
    """The registered backward op returns the gradients autograd produces."""
    from mudnn.linear_attention.ops import gdn as gdn_ops

    (seqs, scale) = ((128, 128), DIM ** (-0.5))
    (q, k, v, g, beta, cu) = _old_engine_inputs(seqs, 4, requires_grad=True)
    (out, _) = gated_delta_net(q, k, v, g, beta, cu, plan_name="gdn_tilelang")
    dO = torch.randn_like(out)
    out.backward(dO)
    plain = [t.detach() for t in (q, k, v, g, beta)]
    (_o, _state, _checkpoints, *handoff) = gdn_ops._run_forward(*plain, cu, scale, None, False, "gdn_tilelang")
    grads = torch.ops.mudnn.gated_delta_net_bwd(dO, *plain, cu, handoff, scale)
    for name, got, want in zip("qkvgb", grads, (q, k, v, g, beta)):
        assert torch.equal(got, want.grad), f"d{name}"
    assert grads[5].numel() == 0
    fake = torch.ops.mudnn.gated_delta_net_bwd(
        *(_meta(t) for t in (dO, *plain)), _meta(cu), [_meta(t) for t in handoff], scale
    )
    assert [(t.shape, t.dtype) for t in grads] == [(t.shape, t.dtype) for t in fake]


# Canonical API and migrated prefill/decode kernels

PARAMETERS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "cu_seqlens",
    "scale",
    "initial_state",
    "output_final_state",
    "use_qk_l2norm_in_kernel",
    "batch_invariant",
    "use_beta_sigmoid_in_kernel",
    "allow_neg_eigval",
    "safe_gate",
    "gate_domain",
    "a_log",
    "dt_bias",
    "checkpoint_every_n_tokens",
    "plan_name",
    "overwrite_initial_state",
    "state_indices",
)


def test_signature_matches_pinned_cudnn_contract():
    for module in ("mudnn.linear_attention", "mudnn.linear_attention.ops", "mudnn.linear_attention.ops.gdn"):
        assert not hasattr(importlib.import_module(module), "gated_delta_net_legacy")
    signature = inspect.signature(gated_delta_net)
    assert tuple(signature.parameters) == PARAMETERS
    assert signature.parameters["gate_domain"].default == "log"
    assert signature.parameters["overwrite_initial_state"].default is False


def test_fake_outputs_and_mutation_contract():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(129, 2, 128, dtype=torch.bfloat16)
        g = torch.empty(129, 2, dtype=torch.float32)
        cu = torch.empty(3, dtype=torch.int32)
        state = torch.empty(2, 2, 128, 128)
        o, final, checkpoints = gated_delta_net(
            q, q, q, g, g, cu, initial_state=state, output_final_state=True, checkpoint_every_n_tokens=64
        )
        assert o.shape == q.shape
        assert final.shape == state.shape and final.dtype == torch.float32
        assert checkpoints.shape == (4, 2, 128, 128)
        _, final = gated_delta_net(q, q, q, g, g, cu, initial_state=state, overwrite_initial_state=True)
        assert final is state


def _inference_inputs(lengths, dtype=torch.bfloat16, heads=2, value_heads=None):
    torch.manual_seed(2026)
    hv = heads if value_heads is None else value_heads
    t = sum(lengths)
    make = lambda h: torch.randn(t, h, 128, device="musa", dtype=dtype)
    q, k, v = make(heads), make(heads), make(hv)
    # Normalized inputs keep the recurrence stable without hiding layout errors.
    for x in (q, k):
        x.copy_((x.float() / (x.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(dtype))
    g = -torch.rand(t, hv, device="musa", dtype=torch.float32) * 0.1
    beta = torch.rand(t, hv, device="musa", dtype=torch.float32)
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    cu = torch.tensor(bounds, dtype=torch.int32, device="musa")
    state = torch.randn(len(lengths), hv, 128, 128, device="musa") * 0.03
    return q, k, v, g, beta, cu, state


def _inference_reference(q, k, v, g, beta, lengths, initial, scale=None):
    """Use the same independent FP64 recurrence for inference and training."""
    group = v.shape[1] // q.shape[1]
    q, k = (t.cpu().double().repeat_interleave(group, dim=1) for t in (q, k))
    v, g, beta = (t.cpu().double() for t in (v, g, beta))
    out, state = recurrent_gdn_fp64(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens(lengths, device="cpu"),
        128**-0.5 if scale is None else scale,
        initial.cpu().double().transpose(-1, -2),
    )
    return out.float(), state.transpose(-1, -2).float()


def _assert_inference_numerics(actual, expected, dtype):
    actual = actual.float().cpu()
    tolerance = 0.025 if dtype == torch.bfloat16 else 0.006
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-6)
    assert error < tolerance, float(error)


@GPU
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "plan,lengths",
    [
        ("gdn_prefill_tilelang", [65, 0, 129]),
        ("gdn_decode_tilelang", [1, 1, 1]),
        ("gdn_decode_tilelang", [0, 2, 1]),
    ],
)
def test_migrated_forward_and_readonly_state(plan, lengths, dtype):
    q, k, v, g, beta, cu, state = _inference_inputs(lengths, dtype)
    before = state.clone()
    expected_o, expected_state = _inference_reference(q, k, v, g, beta, lengths, state)
    o, final = gated_delta_net(q, k, v, g, beta, cu, initial_state=state, output_final_state=True, plan_name=plan)
    assert torch.equal(state, before)
    _assert_inference_numerics(o, expected_o, dtype)
    _assert_inference_numerics(final, expected_state, dtype)


@GPU
@pytest.mark.parametrize("plan", ["gdn_prefill_tilelang", "gdn_decode_tilelang"])
def test_raw_gates_and_norm_do_not_mutate_qk(plan):
    q, k, v, _, _, cu, state = _inference_inputs([1, 1] if "decode" in plan else [65])
    q.mul_(3)
    k.mul_(4)
    before_q, before_k = q.clone(), k.clone()
    raw_g = torch.randn(q.shape[0], v.shape[1], device="musa", dtype=torch.bfloat16)
    raw_b = torch.randn_like(raw_g)
    a_log = torch.randn(v.shape[1], device="musa") * 0.1
    bias = torch.randn(v.shape[1], device="musa")
    g = -a_log.exp() * torch.nn.functional.softplus(raw_g.float() + bias)
    beta = raw_b.float().sigmoid()
    qn = q.float() / (q.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    kn = k.float() / (k.float().square().sum(-1, keepdim=True) + 1e-6).sqrt()
    lengths = [1, 1] if "decode" in plan else [65]
    expected_o, expected_state = _inference_reference(qn, kn, v, g, beta, lengths, state)
    o, final = gated_delta_net(
        q,
        k,
        v,
        raw_g,
        raw_b,
        cu,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        safe_gate=True,
        a_log=a_log,
        dt_bias=bias,
        use_beta_sigmoid_in_kernel=True,
        plan_name=plan,
    )
    assert torch.equal(q, before_q) and torch.equal(k, before_k)
    _assert_inference_numerics(o, expected_o, q.dtype)
    _assert_inference_numerics(final, expected_state, q.dtype)


@GPU
@MP32_ONLY
def test_zero_tokens_checkpoint_and_scale_zero():
    q, k, v, g, beta, cu, state = _inference_inputs([0, 0])
    o, final, checkpoints = gated_delta_net(
        q, k, v, g, beta, cu, initial_state=state, output_final_state=True, checkpoint_every_n_tokens=64
    )
    assert o.numel() == 0 and torch.equal(final, state)
    assert checkpoints.shape == (2, 2, 128, 128)
    q, k, v, g, beta, cu, state = _inference_inputs([65])
    o, _ = gated_delta_net(q, k, v, g, beta, cu, scale=0.0)
    assert not o.any()


@GPU
@MP32_ONLY
def test_checkpoint_matches_prefix_state_and_changed_boundaries():
    q, k, v, g, beta, cu, state = _inference_inputs([65, 64])
    _, _, checkpoints = gated_delta_net(
        q, k, v, g, beta, cu, initial_state=state, checkpoint_every_n_tokens=64, plan_name="gdn_prefill_tilelang"
    )
    _, prefix = _inference_reference(q[:64], k[:64], v[:64], g[:64], beta[:64], [64], state[:1])
    _assert_inference_numerics(checkpoints[1], prefix[0], q.dtype)
    cu.copy_(torch.tensor([0, 1, 129], device="musa", dtype=torch.int32))
    _, _, changed = gated_delta_net(
        q, k, v, g, beta, cu, initial_state=state, checkpoint_every_n_tokens=64, plan_name="gdn_prefill_tilelang"
    )
    _assert_inference_numerics(changed[1], state[1].cpu(), q.dtype)


@GPU
@MP32_ONLY
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_inference_keeps_checkpoint_cadences_above_64(dtype):
    q, k, v, g, beta, cu, state = _inference_inputs([192], dtype)
    _, _, checkpoints = gated_delta_net(
        q, k, v, g, beta, cu, initial_state=state, checkpoint_every_n_tokens=128, plan_name="gdn_prefill_tilelang"
    )
    _, prefix_state = _inference_reference(q[:128], k[:128], v[:128], g[:128], beta[:128], [128], state)
    assert checkpoints.shape == (2, 2, 128, 128)
    _assert_inference_numerics(checkpoints[0], state[0].cpu(), dtype)
    _assert_inference_numerics(checkpoints[1], prefix_state[0], dtype)


@GPU
@pytest.mark.parametrize("plan", ["gdn_prefill_tilelang", "gdn_decode_tilelang"])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_overwrite_state_including_empty_sequences(plan, output_final_state):
    lengths = [0, 65] if "prefill" in plan else [0, 1]
    q, k, v, g, beta, cu, state = _inference_inputs(lengths)
    expected_o, expected_state = _inference_reference(q, k, v, g, beta, lengths, state)
    o, final = gated_delta_net(
        q,
        k,
        v,
        g,
        beta,
        cu,
        initial_state=state,
        overwrite_initial_state=True,
        output_final_state=output_final_state,
        plan_name=plan,
    )
    assert final is state
    _assert_inference_numerics(o, expected_o, q.dtype)
    _assert_inference_numerics(state, expected_state, q.dtype)


@GPU
@pytest.mark.parametrize("plan", ["gdn_prefill_tilelang", "gdn_decode_tilelang"])
def test_split_qkv_grouped_heads_and_int64_prefixes(plan):
    lengths = [65, 1] if "prefill" in plan else [0, 2, 1]
    q, k, v, g, beta, cu, state = _inference_inputs(lengths, value_heads=4)
    packed = torch.empty(q.shape[0], 2, 3, 128, dtype=q.dtype, device=q.device)
    packed[:, :, 0].copy_(q)
    packed[:, :, 1].copy_(k)
    q, k = packed[:, :, 0], packed[:, :, 1]
    strided_v = torch.empty(*v.shape[:-1], 256, dtype=v.dtype, device=v.device)
    strided_v[..., ::2].copy_(v)
    v = strided_v[..., ::2] if "decode" in plan else v
    expected_o, expected_state = _inference_reference(q, k, v, g, beta, lengths, state)
    o, final = gated_delta_net(
        q, k, v, g, beta, cu.to(torch.int64), initial_state=state, output_final_state=True, plan_name=plan
    )
    _assert_inference_numerics(o, expected_o, q.dtype)
    _assert_inference_numerics(final, expected_state, q.dtype)


@GPU
@pytest.mark.parametrize("plan", ["gdn_prefill_tilelang", "gdn_decode_tilelang"])
def test_linear_decay_without_initial_state(plan):
    lengths = [65] if "prefill" in plan else [0, 2, 1]
    q, k, v, g, beta, cu, state = _inference_inputs(lengths)
    state.zero_()
    expected_o, expected_state = _inference_reference(q, k, v, g, beta, lengths, state)
    o, final = gated_delta_net(
        q, k, v, g.exp(), beta, cu, gate_domain="linear", output_final_state=True, plan_name=plan
    )
    _assert_inference_numerics(o, expected_o, q.dtype)
    _assert_inference_numerics(final, expected_state, q.dtype)


@GPU
@pytest.mark.parametrize("plan", ["gdn_prefill_tilelang", "gdn_decode_tilelang"])
def test_strided_raw_gates_with_optional_parameters_missing(plan):
    lengths = [65] if "prefill" in plan else [0, 2, 1]
    q, k, v, _, _, cu, state = _inference_inputs(lengths)
    g = torch.randn(2, q.shape[0], device=q.device).transpose(0, 1)
    beta = torch.randn_like(g)
    expected_o, expected_state = _inference_reference(
        q, k, v, -torch.nn.functional.softplus(g), beta.sigmoid() * 2, lengths, state
    )
    o, final = gated_delta_net(
        q,
        k,
        v,
        g,
        beta,
        cu,
        initial_state=state,
        output_final_state=True,
        safe_gate=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        plan_name=plan,
    )
    _assert_inference_numerics(o, expected_o, q.dtype)
    _assert_inference_numerics(final, expected_state, q.dtype)


@GPU
@pytest.mark.parametrize("write", [False, True])
def test_mixed_length_pool_with_negative_and_strided_slots(write):
    lengths = [1, 2, 0, 1]
    q, k, v, g, beta, cu, _ = _inference_inputs(lengths)
    slots = torch.tensor([-1, 9, 3, 9, 1, 9, 2, 9], device=q.device, dtype=torch.int32)[::2]
    pool = torch.randn(5, 2, 128, 128, device=q.device) * 0.03
    before = pool.clone()
    initial = before[slots.clamp_min(0).long()]
    expected_o, expected_state = _inference_reference(q, k, v, g, beta, lengths, initial)
    expected_o[0].zero_()
    o, final = gated_delta_net(
        q,
        k,
        v,
        g,
        beta,
        cu,
        initial_state=pool,
        state_indices=slots,
        overwrite_initial_state=write,
        plan_name="gdn_decode_tilelang",
    )
    _assert_inference_numerics(o, expected_o, q.dtype)
    if write:
        assert final is pool
        _assert_inference_numerics(pool[slots[1:].long()], expected_state[1:], q.dtype)
        assert torch.equal(pool[[0, 4]], before[[0, 4]])
    else:
        assert final.numel() == 0 and torch.equal(pool, before)


@GPU
def test_prefill_then_decode_uses_the_same_state_layout():
    q, k, v, g, beta, cu, state = _inference_inputs([66])
    expected_o, expected_state = _inference_reference(q, k, v, g, beta, [66], state)
    _, prefill_state = gated_delta_net(
        q[:65],
        k[:65],
        v[:65],
        g[:65],
        beta[:65],
        torch.tensor([0, 65], device=q.device, dtype=cu.dtype),
        initial_state=state,
        output_final_state=True,
        plan_name="gdn_prefill_tilelang",
    )
    o, final = gated_delta_net(
        q[65:],
        k[65:],
        v[65:],
        g[65:],
        beta[65:],
        torch.tensor([0, 1], device=q.device, dtype=cu.dtype),
        initial_state=prefill_state,
        output_final_state=True,
        plan_name="gdn_decode_tilelang",
    )
    _assert_inference_numerics(o, expected_o[65:], q.dtype)
    _assert_inference_numerics(final, expected_state, q.dtype)


@GPU
@pytest.mark.parametrize("plan", ["gdn_prefill_tilelang", "gdn_decode_tilelang"])
def test_capture_replay_reads_changed_device_prefixes(plan):
    if "prefill" in plan and mp32_skip_reason() is not None:
        # The prefill checkpoint series is MP32-only.
        pytest.skip(mp32_skip_reason())
    q, k, v, g, beta, cu, state = _inference_inputs([1, 1])
    kwargs = dict(initial_state=state, output_final_state=True, plan_name=plan)
    if "prefill" in plan:
        kwargs["checkpoint_every_n_tokens"] = 64
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(2):
            gated_delta_net(q, k, v, g, beta, cu, **kwargs)
    torch.musa.current_stream().wait_stream(stream)
    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph, stream=stream):
        result = gated_delta_net(q, k, v, g, beta, cu, **kwargs)
    cu.copy_(torch.tensor([0, 0, 2], device=q.device, dtype=cu.dtype))
    expected_o, expected_state = _inference_reference(q, k, v, g, beta, [0, 2], state)
    graph.replay()
    o, final = result[:2]
    _assert_inference_numerics(o, expected_o, q.dtype)
    _assert_inference_numerics(final, expected_state, q.dtype)
    if "prefill" in plan:
        # Empty sequences contribute no pre-token checkpoints.
        _assert_inference_numerics(result[2][0], state[1].cpu(), q.dtype)


@GPU
@pytest.mark.parametrize("plan", ["gdn_prefill_tilelang", "gdn_decode_tilelang"])
def test_concurrent_streams_do_not_share_scratch(plan):
    lengths = [65, 1] if "prefill" in plan else [0, 2, 1]
    q, k, v, g, beta, cu, state = _inference_inputs(lengths)
    other_state = state.clone().mul_(2)
    gated_delta_net(q, k, v, g, beta, cu, initial_state=state, output_final_state=True, plan_name=plan)
    streams = [torch.musa.Stream(), torch.musa.Stream()]
    results = []
    for stream, initial in zip(streams, (state, other_state)):
        stream.wait_stream(torch.musa.current_stream())
        with torch.musa.stream(stream):
            results.append(
                gated_delta_net(q, k, v, g, beta, cu, initial_state=initial, output_final_state=True, plan_name=plan)
            )
    for stream in streams:
        torch.musa.current_stream().wait_stream(stream)
    for (o, final), initial in zip(results, (state, other_state)):
        expected_o, expected_state = _inference_reference(q, k, v, g, beta, lengths, initial)
        _assert_inference_numerics(o, expected_o, q.dtype)
        _assert_inference_numerics(final, expected_state, q.dtype)


@GPU
def test_empty_decode_and_zero_scale_preserve_state_semantics():
    q, k, v, g, beta, cu, state = _inference_inputs([0, 0])
    o, final = gated_delta_net(
        q, k, v, g, beta, cu, initial_state=state, output_final_state=True, plan_name="gdn_decode_tilelang"
    )
    assert o.numel() == 0 and torch.equal(final, state)
    q, k, v, g, beta, cu, state = _inference_inputs([0, 2, 1])
    _, expected_state = _inference_reference(q, k, v, g, beta, [0, 2, 1], state)
    o, final = gated_delta_net(
        q, k, v, g, beta, cu, initial_state=state, scale=0.0, output_final_state=True, plan_name="gdn_decode_tilelang"
    )
    assert not o.any()
    _assert_inference_numerics(final, expected_state, q.dtype)


@GPU
def test_unsupported_inference_capabilities_raise_before_mutation():
    q, k, v, g, beta, cu, state = _inference_inputs([1, 1])
    before = state.clone()
    for kwargs, message in (
        ({"batch_invariant": True}, "batch_invariant"),
        ({"plan_name": "not_registered"}, "no"),
        ({"initial_state": state.to(torch.bfloat16)}, "initial_state must be float32"),
        ({"g": g.transpose(0, 1)}, "activated g must be contiguous"),
    ):
        arguments = dict(
            q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu, initial_state=state, overwrite_initial_state=True
        )
        arguments.update(kwargs)
        with pytest.raises(mudnn.cudnnGraphNotSupportedError, match=message):
            gated_delta_net(**arguments)
        assert torch.equal(state, before)


# Training pairing against the independent FP64 recurrence


@GPU
@MP32_ONLY
@pytest.mark.parametrize(
    "dtype,seqs,hq,hv",
    [
        (torch.bfloat16, (128, 128), 4, 4),
        (torch.bfloat16, (192,), 4, 4),
        (torch.bfloat16, (128, 128), 2, 4),
        (torch.bfloat16, (128, 128), 1, 4),
        (torch.bfloat16, (0, 128), 4, 4),
        (torch.bfloat16, (64, 64, 64, 64), 4, 4),
        (torch.float16, (128, 128), 4, 4),
        (torch.float16, (128, 128), 2, 4),
    ],
)
def test_training_pair_matches_the_fp64_recurrence(dtype, seqs, hq, hv):
    q, k, v, g, beta, _ = make_gdn_inputs(dtype, seqs, hq, hv, False)
    cu = cu_seqlens(seqs)
    scale = DIM**-0.5

    group = hv // hq
    qd = q.cpu().double()
    kd = k.cpu().double()
    qx = (qd.repeat_interleave(group, 1) if group > 1 else qd).requires_grad_(True)
    kx = (kd.repeat_interleave(group, 1) if group > 1 else kd).requires_grad_(True)
    vx = v.cpu().double().requires_grad_(True)
    gx = g.cpu().double().requires_grad_(True)
    bx = beta.cpu().double().requires_grad_(True)

    ref_o, _ = recurrent_gdn_fp64(qx, kx, vx, gx, bx, cu.cpu(), scale)
    dO = 2.0 * ref_o.detach()
    dqx, dkx, dvx, dgx, dbx = torch.autograd.grad((ref_o * dO).sum(), (qx, kx, vx, gx, bx))

    def fold(t):
        return t.view(sum(seqs), hq, group, DIM).sum(2) if group > 1 else t

    tq, tk, tv, tg, tb = (t.clone().requires_grad_(True) for t in (q, k, v, g, beta))
    out, _ = gated_delta_net(tq, tk, tv, tg, tb, cu, plan_name="gdn_tilelang_v2")
    out.backward(dO.to(DEVICE).to(dtype))
    torch.musa.synchronize()

    assert relative_l2_fp64(ref_o, out.detach()) < REL_TOL, "forward"
    for name, ref, got in (("q", dqx, tq), ("k", dkx, tk), ("v", dvx, tv), ("g", dgx, tg), ("beta", dbx, tb)):
        want = fold(ref) if name in ("q", "k") else ref
        assert relative_l2_fp64(want, got.grad) < REL_TOL, f"d{name}"


@GPU
@MP32_ONLY
@pytest.mark.parametrize(
    "dtype,final_only,cadence",
    [
        (torch.bfloat16, False, 0),
        (torch.bfloat16, True, 64),
        (torch.float16, True, 0),
        (torch.float16, False, 64),
    ],
)
def test_training_pair_carries_state(dtype, final_only, cadence):
    """Pairwise coverage of IO dtype, checkpoint reuse and final-state-only loss."""
    seqs, hq, hv = (128, 128), 4, 4
    q, k, v, g, beta, state = make_gdn_inputs(dtype, seqs, hq, hv, True, seed=7)
    cu = cu_seqlens(seqs)
    scale = DIM**-0.5

    h0x = state.cpu().double().transpose(-1, -2).contiguous().requires_grad_(True)
    qx = q.cpu().double().requires_grad_(True)
    kx = k.cpu().double().requires_grad_(True)
    vx = v.cpu().double().requires_grad_(True)
    gx = g.cpu().double().requires_grad_(True)
    bx = beta.cpu().double().requires_grad_(True)
    ref_o, ref_h = recurrent_gdn_fp64(qx, kx, vx, gx, bx, cu.cpu(), scale, h0x)
    dO = torch.zeros_like(ref_o) if final_only else 2.0 * ref_o.detach()
    dH = 2.0 * ref_h.detach()
    grads = torch.autograd.grad((ref_o * dO).sum() + (ref_h * dH).sum(), (qx, kx, vx, gx, bx, h0x))
    dh0x = grads[5]

    tq, tk, tv, tg, tb = (t.clone().requires_grad_(True) for t in (q, k, v, g, beta))
    tstate = state.clone().requires_grad_(True)
    if (dtype == torch.bfloat16 and cadence == 0) or (dtype == torch.float16 and cadence == 64):
        # Exercise strided VK with both recomputation and checkpoint reuse.
        tstate = state.transpose(-1, -2).contiguous().transpose(-1, -2).requires_grad_(True)
    result = gated_delta_net(
        tq, tk, tv, tg, tb, cu, initial_state=tstate, output_final_state=True, checkpoint_every_n_tokens=cadence
    )
    out, final = result[:2]
    if cadence:
        assert len(result) == 3 and not result[2].requires_grad
    d_final = dH.transpose(-1, -2).contiguous().to(DEVICE).float()
    if final_only:
        final.backward(d_final)
    else:
        torch.autograd.backward((out, final), (dO.to(DEVICE).to(dtype), d_final))
    torch.musa.synchronize()

    assert relative_l2_fp64(ref_o, out.detach()) < REL_TOL, "forward"
    # the canonical layout is K-last, so the K-major reference transposes
    assert relative_l2_fp64(ref_h, final.detach().transpose(-1, -2)) < REL_TOL, "final state"
    assert relative_l2_fp64(dh0x, tstate.grad.transpose(-1, -2)) < REL_TOL, "dh0"
    for name, ref, leaf in zip(("q", "k", "v", "g", "beta"), grads[:5], (tq, tk, tv, tg, tb)):
        assert relative_l2_fp64(ref, leaf.grad) < REL_TOL, f"d{name}"


@GPU
@MP32_ONLY
def test_training_pair_needs_a_differentiable_call():
    """Pinning the training pairing without gradients is a contract error."""
    seqs = (128,)
    q, k, v, g, beta, _ = make_gdn_inputs(torch.bfloat16, seqs, 4, 4, False)
    with torch.no_grad():
        with pytest.raises(Exception, match="training plan"):
            gated_delta_net(q, k, v, g, beta, cu_seqlens(seqs), plan_name="gdn_tilelang_v2")
    q.requires_grad_()
    with pytest.raises(mudnn.cudnnGraphNotSupportedError, match="inference only"):
        gated_delta_net(q, k, v, g, beta, cu_seqlens(seqs), plan_name="gdn_decode_tilelang")


# Default architecture routing and state continuation


@GPU
def test_default_training_selects_the_architecture_pair():
    q, k, v, g, beta, state = make_gdn_inputs(torch.bfloat16, (128,), 2, 2, True)
    cu = cu_seqlens((128,))
    leaves = [t.requires_grad_() for t in (q, k, v, g, beta, state)]
    ops.fprop_cache.clear()
    ops.bprop_cache.clear()
    out, final = gated_delta_net(*leaves[:5], cu, initial_state=leaves[5], output_final_state=True)
    out.float().square().sum().backward()
    expected = "gdn_tilelang_v2" if torch.musa.get_device_capability(q.device) == (3, 2) else "gdn_tilelang"
    assert {p.plan_name for p in ops.fprop_cache.values()} == {expected}
    assert {p.plan_name for p in ops.bprop_cache.values()} == {expected}
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in leaves)
    qx, kx, vx, gx, bx, hx = [t.detach().cpu().double().requires_grad_() for t in leaves]
    ref, ref_final = recurrent_gdn_fp64(qx, kx, vx, gx, bx, cu.cpu(), 128**-0.5, hx.transpose(-1, -2))
    grads = torch.autograd.grad(ref.square().sum(), (qx, kx, vx, gx, bx, hx))
    assert relative_l2_fp64(ref, out) < 0.02
    assert relative_l2_fp64(ref_final, final.transpose(-1, -2)) < 0.02
    for ref_grad, leaf in zip(grads, leaves):
        assert relative_l2_fp64(ref_grad, leaf.grad) < 0.02


@GPU
def test_old_engine_zero_state_then_continuation():
    if torch.musa.get_device_capability() != (3, 1):
        pytest.skip("old-engine numerical qualification uses MP31; MP32 baseline is known unsound")
    (q, k, v, g, beta, _) = make_gdn_inputs(torch.bfloat16, (256,), 2, 2, False)
    with torch.no_grad():
        (out, state) = gated_delta_net(
            q[:128],
            k[:128],
            v[:128],
            g[:128],
            beta[:128],
            cu_seqlens((128,)),
            output_final_state=True,
            plan_name="gdn_tilelang",
        )
        (tail, final) = gated_delta_net(
            q[128:],
            k[128:],
            v[128:],
            g[128:],
            beta[128:],
            cu_seqlens((128,)),
            initial_state=state,
            output_final_state=True,
            plan_name="gdn_tilelang",
        )
    (ref, ref_final) = recurrent_gdn_fp64(
        *(t.cpu().double() for t in (q, k, v, g, beta)), cu_seqlens((256,)).cpu(), 128 ** (-0.5)
    )
    assert relative_l2_fp64(ref, torch.cat((out, tail))) < 0.02
    assert relative_l2_fp64(ref_final, final.transpose(-1, -2)) < 0.02


# API rejection and FLA layout behavior without kernel compilation


def _cpu_inputs():
    q = torch.zeros(4, 1, 128, dtype=torch.bfloat16)
    g = torch.zeros(4, 1)
    beta = torch.ones(4, 1, dtype=q.dtype)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    return q, q.clone(), q.clone(), g, beta, cu


@pytest.mark.parametrize("v_first", [False, True])
@pytest.mark.parametrize("with_state", [False, True])
def test_fla_preserves_its_state_contract(monkeypatch, v_first, with_state):
    canonical = torch.arange(128 * 128, dtype=torch.float32).reshape(1, 1, 128, 128)
    weight = canonical + 1
    incoming = (canonical if v_first else canonical.transpose(-1, -2)).detach().requires_grad_()
    seen = []

    def forward(q, k, v, g, beta, cu, *, initial_state, plan_name, **kwargs):
        assert plan_name == "gdn_tilelang"
        seen.append(initial_state)
        state = canonical if initial_state is None else initial_state
        return q.clone(), state * weight + 7

    monkeypatch.setattr(fla, "gated_delta_net", forward)
    q, k, v, g, beta, _ = _cpu_inputs()
    _, final = fla._to_native(
        *(t.unsqueeze(0) for t in (q, k, v, g, beta)),
        scale=None,
        initial_state=incoming if with_state else None,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        state_v_first=v_first,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
    )
    assert seen == [None] if not with_state else torch.equal(seen[0], canonical)
    expected = canonical * weight + 7
    assert torch.equal(final, expected if v_first else expected.transpose(-1, -2))
    if with_state:
        final.sum().backward()
        assert torch.equal(incoming.grad, weight if v_first else weight.transpose(-1, -2))


@GPU
@pytest.mark.parametrize("v_first", [False, True])
def test_old_engine_fla_state_gradients_match_fp64(v_first):
    if torch.musa.get_device_capability() != (3, 1):
        pytest.skip("old-engine numerical qualification uses MP31")
    q, k, v, g, beta, canonical = make_gdn_inputs(torch.bfloat16, (128,), 2, 2, True, seed=17)
    incoming = canonical if v_first else canonical.transpose(-1, -2).contiguous()
    leaves = [t.detach().requires_grad_() for t in (q, k, v, g, beta, incoming)]
    state_before = leaves[5].detach().clone()
    out, final = fla._to_native(
        *(t.unsqueeze(0) for t in leaves[:5]),
        scale=None,
        initial_state=leaves[5],
        output_final_state=True,
        cu_seqlens=cu_seqlens((128,)),
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        state_v_first=v_first,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
    )
    (out.float().square().sum() + final.square().sum()).backward()
    assert torch.equal(leaves[5], state_before)
    refs = [t.detach().cpu().double().requires_grad_() for t in leaves]
    ref_initial = refs[5].transpose(-1, -2) if v_first else refs[5]
    ref_o, ref_h = recurrent_gdn_fp64(*refs[:5], cu_seqlens((128,), device="cpu"), DIM**-0.5, ref_initial)
    grads = torch.autograd.grad(ref_o.square().sum() + ref_h.square().sum(), refs)
    assert relative_l2_fp64(ref_o, out.squeeze(0)) < REL_TOL
    assert relative_l2_fp64(ref_h, final.transpose(-1, -2) if v_first else final) < REL_TOL
    for ref, leaf in zip(grads, leaves):
        assert relative_l2_fp64(ref, leaf.grad) < REL_TOL


def test_unsupported_training_cadence_is_rejected_before_forward(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("unsupported cadence launched a forward")

    monkeypatch.setattr(ops, "_run_forward", unexpected)
    args = list(_cpu_inputs())
    args[0].requires_grad_()
    with pytest.raises(NotImplementedError, match="cadence 0 or 64"):
        gated_delta_net(*args, checkpoint_every_n_tokens=128)
