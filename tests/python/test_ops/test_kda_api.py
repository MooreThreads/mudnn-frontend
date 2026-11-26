"""CPU API/FakeTensor contracts from cuDNN frontend 33c6fcb54d11."""

import inspect
from pathlib import Path
import re

import pytest

torch = pytest.importorskip("torch")
from mudnn import linear_attention
from mudnn.linear_attention.ops import kda
from mudnn.linear_attention.mutlass import kda_ops


REQUIRED = inspect.Parameter.empty
PREFIX = [(n, REQUIRED) for n in ("q", "k", "v", "g", "beta", "cu_seqlens")]
OPTIONS = [
    ("use_qk_l2norm_in_kernel", False),
    ("batch_invariant", False),
    ("use_beta_sigmoid_in_kernel", False),
    ("allow_neg_eigval", False),
    ("safe_gate", False),
    ("gate_lower_bound", None),
    ("gate_domain", "log"),
    ("a_log", None),
    ("dt_bias", None),
    ("checkpoint_every_n_tokens", 0),
    ("plan_name", None),
]
SUMMARY_OPTIONS = [
    p
    for p in OPTIONS
    if p[0] not in ("batch_invariant", "checkpoint_every_n_tokens", "plan_name")
]
SUMMARY_OPTIONS += [("batch_invariant", False), ("plan_name", None)]
SUMMARY_PREFIX = [(n, REQUIRED) for n in ("k", "v", "g", "beta", "cu_seqlens")]
BWD_SUMMARY_PREFIX = [
    (n, REQUIRED) for n in ("dO", "q", "k", "g", "beta", "cu_seqlens")
]


@pytest.mark.parametrize(
    "name,expected,keyword_start",
    [
        (
            "kimi_delta_attention",
            PREFIX
            + [("scale", None), ("initial_state", None), ("output_final_state", False)]
            + OPTIONS
            + [("overwrite_initial_state", False), ("state_indices", None)],
            None,
        ),
        (
            "kimi_delta_attention_summary",
            SUMMARY_PREFIX + [("initial_state", None)] + SUMMARY_OPTIONS,
            5,
        ),
        (
            "kimi_delta_attention_summary_bwd",
            [(n, REQUIRED) for n in ("q", "k", "g", "beta", "dO", "cu_seqlens")]
            + [("scale", None), ("d_final_state", None), ("output_transition", False)]
            + SUMMARY_OPTIONS,
            6,
        ),
    ],
)
def test_public_signatures_match_upstream(name, expected, keyword_start):
    function = getattr(linear_attention, name)
    assert function is getattr(kda, name)
    parameters = list(inspect.signature(function).parameters.values())
    assert [(p.name, p.default) for p in parameters] == expected
    for i, p in enumerate(parameters):
        kind = (
            inspect.Parameter.KEYWORD_ONLY
            if keyword_start is not None and i >= keyword_start
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        assert p.kind == kind


@pytest.mark.parametrize(
    "name,expected,arity",
    [
        (
            "kimi_delta_attention_fwd",
            PREFIX
            + [
                ("scale", REQUIRED),
                ("initial_state", None),
                ("output_final_state", False),
            ]
            + OPTIONS,
            3,
        ),
        (
            "kimi_delta_attention_fwd_overwrite_state",
            PREFIX
            + [("scale", REQUIRED), ("initial_state", REQUIRED)]
            + OPTIONS
            + [("state_indices", None), ("output_final_state", True)],
            2,
        ),
        (
            "kimi_delta_attention_bwd",
            [("dO", REQUIRED)]
            + PREFIX
            + [
                ("scale", REQUIRED),
                ("initial_state", None),
                ("d_final_state", None),
                ("state_checkpoints", None),
                ("checkpoint_every_n_tokens", 0),
            ]
            + OPTIONS[:-2]
            + [("plan_name", None)],
            8,
        ),
        (
            "kimi_delta_attention_summary",
            SUMMARY_PREFIX
            + [("initial_state", None), ("output_transition", False)]
            + SUMMARY_OPTIONS,
            2,
        ),
        (
            "kimi_delta_attention_summary_bwd",
            BWD_SUMMARY_PREFIX
            + [
                ("scale", REQUIRED),
                ("d_final_state", None),
                ("output_transition", False),
            ]
            + SUMMARY_OPTIONS,
            2,
        ),
    ],
)
def test_custom_op_schemas_match_upstream(name, expected, arity):
    schema = getattr(torch.ops.mudnn, name).default._schema
    got = [
        (a.name, a.default_value if a.has_default_value() else REQUIRED)
        for a in schema.arguments
    ]
    assert got == expected
    assert len(schema.returns) == arity
    if name.endswith("overwrite_state"):
        assert next(
            a for a in schema.arguments if a.name == "initial_state"
        ).alias_info.is_write


def meta_inputs(requires_grad=False):
    def make(shape, dtype):
        return torch.empty(
            shape, device="meta", dtype=dtype, requires_grad=requires_grad
        )

    q, k = [make((17, 2, 64), torch.bfloat16) for _ in range(2)]
    v = make((17, 4, 96), torch.bfloat16)
    g, beta = make((17, 4, 64), torch.float32), make((17, 4), torch.float32)
    cu = torch.empty(3, device="meta", dtype=torch.int64)
    return q, k, v, g, beta, cu


def test_fake_forward_backward_shapes_dtypes_and_checkpoint_autograd():
    args = meta_inputs(requires_grad=True)
    initial = torch.empty((2, 4, 96, 64), device="meta", requires_grad=True)
    out, final, checkpoints = kda.kimi_delta_attention(
        *args,
        initial_state=initial,
        output_final_state=True,
        checkpoint_every_n_tokens=16,
    )
    assert out.shape == (17, 4, 96) and out.dtype == torch.bfloat16
    assert final.shape == initial.shape and final.dtype == torch.float32
    assert checkpoints.shape == (3, 4, 96, 64) and checkpoints.dtype == torch.bfloat16
    assert not checkpoints.requires_grad
    gradients = torch.autograd.grad(out.sum() + final.sum(), (*args[:5], initial))
    for got, want in zip(gradients, (*args[:5], initial)):
        assert got.shape == want.shape and got.dtype == want.dtype


@pytest.mark.parametrize(
    "name", ["kimi_delta_attention_summary", "kimi_delta_attention_summary_bwd"]
)
def test_fake_summary_outputs_are_fp32_and_nondifferentiable(name):
    q, k, v, g, beta, cu = meta_inputs(requires_grad=True)
    if name.endswith("_bwd"):
        state, transition = kda.kimi_delta_attention_summary_bwd(
            q, k, g, beta, v, cu, output_transition=True
        )
    else:
        state, transition = kda.kimi_delta_attention_summary(k, v, g, beta, cu)
    assert state.shape == (2, 4, 96, 64)
    assert transition.shape == (2, 4, 64, 64)
    for tensor in (state, transition):
        assert tensor.dtype == torch.float32 and not tensor.requires_grad


@pytest.mark.parametrize("arch", ["mp31", "mp32"])
def test_metadata_capacity_agrees_with_native_prepare_policy(monkeypatch, arch):
    root = Path(__file__).resolve().parents[3]
    policy = root / f"kernels/include/mate/flat/kda/chunk_kda_{arch}_policy.hpp"
    if not policy.is_file():
        pytest.skip(f"{arch} policy is not published in this snapshot")
    text = policy.read_text()
    native = int(
        re.search(r"SchedulePolicy\s*:\s*ChunkKdaPrefillSchedulePolicy<(\d+)", text)[1]
    )
    monkeypatch.setattr(kda_ops, "resolve_kda_arch", lambda: arch)
    assert kda_ops.resolve_kda_prepare_ctas_per_mp() == native
