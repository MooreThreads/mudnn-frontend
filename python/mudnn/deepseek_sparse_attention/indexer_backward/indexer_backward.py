"""Sparse indexer backward kernel (empty placeholder).

Counterpart of the architecture-specific factory modules in ``cudnn-frontend``'s
``python/cudnn/deepseek_sparse_attention/indexer_backward/`` package. Upstream
splits them one module per architecture; this checkout keeps a single,
architecture-neutral module because nothing here is architecture-specific yet.

``cudnn-frontend``'s default backend runs a three-stage pipeline: an in-place
score-grad precompute that overwrites ``attn_score`` with ``grad_signal``, a
warp-specialized backward GEMM producing ``d_index_q``, ``d_weights`` and an
fp32 ``d_index_k`` accumulator, and a trailing fp32 -> output-dtype cast. Its
opt-in ``backend="v2"`` variant shares that first stage and replaces the
GEMM stage with a persistent, dynamic-ticket kernel that splits the fp32
per-slot gradient matrix into a two-term bf16 expansion (``hi + lo``) before the
MMAs and reduces ``d_weights`` deterministically.

No MUSA kernel is implemented in this checkout yet, so the factories below only
declare the callable contract ``api.py`` will dispatch to and decline
deterministically. Until they are implemented,
``mudnn.deepseek_sparse_attention.indexer_backward`` serves the same public
surface with the pure-torch fallback in ``mudnn.DSA``. See ``api.py`` in this
package and ``doc/cudnn_frontend_op.md``.
"""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - the stub must import without torch
    torch = None

__all__ = ["indexer_backward", "indexer_backward_v2"]


def indexer_backward(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    topk,
    sm_scale=1.0,
    block_I=128,
    topk_indices_global: bool = True,
):
    """Return the indexer-backward runner (not implemented).

    ``batch``/``seqlen``/``seqlen_k``/``heads``/``dim``/``topk``/``sm_scale``/
    ``block_I``/``topk_indices_global`` are the plan-time specialization keys,
    matching ``cudnn-frontend``'s per-architecture factories.

    The returned callable takes
    ``(IndexQ, Weights, IndexK, dIndexQ, dWeights, dIndexK, AttnScore,
    IndexScore, TopkIndices, GradLoss, grad_scale, current_stream=None)``.

    ``grad_scale`` is deliberately not a factory argument: it is a runtime host
    scalar, so changing the loss scaling does not recompile.
    """
    raise NotImplementedError(
        "indexer_backward is an empty placeholder in muDNN frontend: no "
        "indexer-backward kernel is implemented yet. The public surface in "
        "mudnn.deepseek_sparse_attention.indexer_backward is served by the "
        "mudnn.DSA torch fallback."
    )


def indexer_backward_v2(
    batch,
    seqlen,
    seqlen_k,
    heads,
    dim,
    topk,
    sm_scale=1.0,
    block_I=128,
    topk_indices_global: bool = True,
    dw_out_dtype=None,
):
    """Return the opt-in v2 indexer-backward runner (not implemented).

    Selects ``cudnn-frontend``'s opt-in v2 GEMM stage (its selector is named
    after the target architecture; ``backend="v2"`` here is the
    device-neutral spelling used by this frontend).
    The plan-time specialization keys are the same as :func:`indexer_backward`,
    plus ``dw_out_dtype``, which selects the compiled ``d_weights`` store dtype
    (``torch.float32`` -- the value ``None`` resolves to, matching upstream's
    default -- or ``torch.bfloat16``). The returned callable has the same
    positional contract as the default backend's.

    The upstream envelope is ``H == 64``, ``D == 128``, ``block_I == 128``,
    ``topk % 128 == 0`` with ``128 <= topk <= 2048``, ``sm_scale > 0`` and
    contiguous same-device tensors, and the backend is request-or-fail outside
    it.
    """
    if dw_out_dtype is None and torch is not None:
        dw_out_dtype = torch.float32
    raise NotImplementedError(
        "indexer_backward_v2 is an empty placeholder in muDNN frontend: no v2 "
        "indexer-backward GEMM stage is implemented yet. Use backend='default' "
        "(served by the mudnn.DSA torch fallback)."
    )
