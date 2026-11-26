"""MP31/MP32 chunk-KDA backward driver and public compatible entry point.

The four-stage pipeline rebuilds K1/K2, runs the reverse K3 state scan, and
launches the chunk-parallel K4 gradient kernel.  The kernels are kept in the
vendored ``kernels/`` tree; this module only adapts their driver to the
frontend JIT and packed THD API.
"""

from __future__ import annotations

import torch

from . import kda_run

_SUPPORTED_QKVA_DTYPES = (torch.float16, torch.bfloat16)


def allocate_kda_workspace(tile_shape, device, dtype):
    records, heads = tile_shape
    return kda_run.allocate_kda_workspace(records, heads, device, dtype)


kda_chunk_state_rows = kda_run.kda_chunk_state_rows
get_physical_num_mps = kda_run._physical_num_mps


def _as_4d_varlen_input(
    x: torch.Tensor,
    *,
    name: str,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    if cu_seqlens is None:
        if x.ndim != 4:
            raise ValueError(f"{name} must be a 4D tensor [B, T, H, 128].")
        return x, False
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        if x.shape[0] != 1:
            raise ValueError(
                f"{name}.shape[0] must be 1 when cu_seqlens is provided; "
                "flatten variable-length input as [S, H, 128]."
            )
        return x, False
    raise ValueError(f"{name} must be [S, H, 128] or [1, S, H, 128] for varlen.")


def chunk_kda_bwd(
    dO: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    use_qk_l2norm_in_kernel: bool = True,
    plan_name: str | None = None,
    safe_gate: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    gate_domain: str = "log2",
    precise_state_scan: bool = False,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Reverse-mode gradients of :func:`chunk_kda`.

    Mirrors cuDNN-frontend's ``kimi_delta_attention_bwd`` argument order and
    returns ``(dq, dk, dv, dg, dbeta, d_initial_state)``.

    Args:
        dO: gradient w.r.t. the forward output ``o``; same shape as ``v``.
        q, k, v, g, beta: the exact forward inputs (see :func:`chunk_kda`).
        scale: QK scale used by the forward. Defaults to ``128 ** -0.5``.
        initial_state: the forward's ``initial_state``; its gradient is
            returned as ``d_initial_state``.
        d_final_state: optional gradient w.r.t. the forward's ``final_state``
            (``[N, Hv, Dv, D]``, fp32).  It seeds the reverse state scan.
        cu_seqlens: varlen boundaries, matching the forward.
        A_log, dt_bias, lower_bound: the forward's safe-gate parameters.
        use_qk_l2norm_in_kernel: must match the forward.
        plan_name: accepted for interface parity; unused.

    Returns:
        ``(dq, dk, dv, dg, dbeta, d_initial_state)``.  ``dq``/``dk``/``dv``
        carry ``q``'s dtype, ``dg``/``dbeta`` are float32 (they are chunk
        reductions), and ``d_initial_state`` is float32 with the forward's
        state shape.  It is an empty tensor when ``initial_state`` was not
        given.

    Notes:
        The MP31/MP32 backends and the option subset listed in
        ``docs/kda_backward.md`` are implemented; everything else is rejected
        here rather than silently approximated.
    """
    del plan_name  # interface parity only
    from . import kda_ops as _kda_ops
    from . import kda_backward_grad_ops as _kda_grad_ops
    from . import kda_backward_scan_ops as _kda_scan_ops

    if q.dim() != 4 and q.dim() != 3:
        raise ValueError("expected THD [B, T, H, D] or [S, H, D] tensors")
    q, squeeze_varlen = _as_4d_varlen_input(q, name="q", cu_seqlens=cu_seqlens)
    k, _ = _as_4d_varlen_input(k, name="k", cu_seqlens=cu_seqlens)
    v, _ = _as_4d_varlen_input(v, name="v", cu_seqlens=cu_seqlens)
    g, _ = _as_4d_varlen_input(g, name="g", cu_seqlens=cu_seqlens)
    dO, _ = _as_4d_varlen_input(dO, name="dO", cu_seqlens=cu_seqlens)
    if beta.ndim == 2 and cu_seqlens is not None:
        beta = beta.unsqueeze(0)

    if q.shape != k.shape:
        raise ValueError("k must have the same shape as q.")
    if q.shape[:2] != v.shape[:2] or v.shape != g.shape:
        raise ValueError("q, v and g must have matching [B, T] dimensions.")
    if v.shape != dO.shape:
        raise ValueError("dO must have the same shape as v (the forward output).")
    if q.shape[2] != v.shape[2]:
        raise ValueError("chunk_kda_bwd currently requires Hqk == Hv (no GQA).")
    if q.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("chunk_kda_bwd currently requires D=128.")
    if beta.shape != v.shape[:3]:
        raise ValueError("beta must have shape [B, T, Hv].")
    if q.dtype not in _SUPPORTED_QKVA_DTYPES:
        raise TypeError("chunk_kda_bwd supports torch.float16 and torch.bfloat16.")
    if any(t.dtype != q.dtype for t in (k, v, dO)):
        raise TypeError("k, v and dO must have the same dtype as q.")
    for name, tensor in (
        ("k", k),
        ("v", v),
        ("g", g),
        ("beta", beta),
        ("dO", dO),
        ("cu_seqlens", cu_seqlens),
    ):
        if tensor is not None and tensor.device != q.device:
            raise ValueError(f"chunk_kda_bwd: {name} must be on q's device")
    if g.dtype not in (*_SUPPORTED_QKVA_DTYPES, torch.float32) or beta.dtype not in (
        q.dtype,
        torch.float32,
    ):
        raise TypeError("unsupported g/beta storage dtype")
    if (A_log is None) != (dt_bias is None):
        raise ValueError("A_log and dt_bias must be provided together.")
    if safe_gate and A_log is None:
        raise NotImplementedError(
            "chunk_kda_bwd currently supports the safe-gate path only; "
            "provide A_log and dt_bias"
        )
    if A_log is not None:
        if A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
            raise TypeError("A_log and dt_bias must be float32.")
        if A_log.shape != (v.shape[2],) or dt_bias.shape != (v.shape[2], 128):
            raise ValueError("A_log must be [Hv] and dt_bias [Hv, 128].")
        if A_log.device != q.device or dt_bias.device != q.device:
            raise ValueError("A_log and dt_bias must be on q's device.")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if cu_seqlens is not None:
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must be int32 or int64.")
        cu_seqlens = cu_seqlens.contiguous()

    nseq = int(cu_seqlens.numel() - 1) if cu_seqlens is not None else q.shape[0]
    state_tail = (v.shape[2], 128, 128)
    if initial_state is not None:
        if initial_state.shape != (nseq, *state_tail):
            raise ValueError(
                f"initial_state must be [N, Hv, Dv, D] = {[nseq, *state_tail]}, "
                f"got {tuple(initial_state.shape)}."
            )
        if initial_state.dtype not in (q.dtype, torch.bfloat16, torch.float32):
            raise TypeError(
                "initial_state dtype must be the activation dtype, bf16 or fp32; "
                f"got {initial_state.dtype}."
            )
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on q's device.")
    if d_final_state is not None:
        if d_final_state.shape != (nseq, *state_tail):
            raise ValueError(
                f"d_final_state must be [N, Hv, Dv, D] = {[nseq, *state_tail]}, "
                f"got {tuple(d_final_state.shape)}."
            )
        if d_final_state.dtype != torch.float32:
            raise TypeError("d_final_state must be float32.")
        if d_final_state.device != q.device:
            raise ValueError("d_final_state must be on q's device.")
        d_final_state = d_final_state.contiguous()

    device = q.device
    # K3 skips the padded tail of a partial chunk, so the outputs have to be
    # zero-initialised rather than left as empty.
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dg = torch.zeros(g.shape, device=device, dtype=torch.float32)
    dbeta = torch.zeros(beta.shape, device=device, dtype=torch.float32)
    dstate0: torch.Tensor
    if initial_state is not None:
        dstate0 = torch.zeros((nseq, *state_tail), device=device, dtype=torch.float32)
    else:
        dstate0 = torch.empty((nseq, *state_tail), device=device, dtype=torch.float32)

    # ---- K1: rebuild the forward's five workspace fields -------------------
    reuse_prepare = (
        _kda_ops.resolve_kda_arch() == "mp32"
        and q.dtype == torch.bfloat16
        and (not safe_gate or -5 <= lower_bound <= 0)
    )
    prepare_config = _kda_ops.make_kda_ops_config(
        q.dtype,
        gate_dtype=g.dtype,
        beta_dtype=beta.dtype,
        use_beta_sigmoid=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        gate_domain=gate_domain,
        state_dtype=q.dtype,
        cu_seqlens_dtype=cu_seqlens.dtype if cu_seqlens is not None else None,
        has_state_in=False,
        has_state_out=False,
        # MP32's checkpoint recurrence consumes the N frame emitted by this K1 variant.
        has_chunk_states_out=_kda_ops.resolve_kda_arch() == "mp32",
        state_fp32=False,
        has_gate_params=safe_gate,
        is_varlen=cu_seqlens is not None,
        normalize_qk=bool(use_qk_l2norm_in_kernel),
        save_prepare_cache=reuse_prepare,
    )
    chunks = (
        q.shape[1] + _kda_ops.KDA_PREFILL_CHUNK_SIZE - 1
    ) // _kda_ops.KDA_PREFILL_CHUNK_SIZE
    workspace_records = chunks + nseq if cu_seqlens is not None else nseq * chunks
    (
        workspace_decayed,
        workspace_k_restored,
        workspace_inverse,
        workspace_p,
        workspace_total,
    ) = allocate_kda_workspace((workspace_records, v.shape[2]), device, q.dtype)
    metadata_capacity = (
        _kda_ops.resolve_kda_prepare_ctas_per_mp() * get_physical_num_mps(device)
        if cu_seqlens is not None and nseq > 1
        else 1
    )
    workspace_metadata = torch.empty(
        (metadata_capacity, 4), device=device, dtype=torch.int32
    )
    prepare_cache = (
        torch.empty(
            (workspace_records, q.shape[2], _kda_ops.KDA_PREPARE_CACHE_FLOATS),
            device=device,
            dtype=torch.float32,
        )
        if reuse_prepare
        else None
    )
    _kda_ops.get_kda_prepare_ops_module(prepare_config).get_function(
        _kda_ops.get_kda_prepare_ops_function_name(prepare_config)
    )(
        q,
        k,
        g,
        beta,
        workspace_decayed,
        workspace_k_restored,
        workspace_inverse,
        workspace_p,
        workspace_total,
        workspace_metadata,
        cu_seqlens,
        A_log,
        dt_bias,
        float(scale),
        float(lower_bound),
        bool(use_qk_l2norm_in_kernel),
        *((prepare_cache,) if reuse_prepare else ()),
    )

    # ---- K2: replay the forward recurrence to get the S_in checkpoints -----
    chunk_rows = kda_chunk_state_rows(
        batch=q.shape[0], num_tokens=q.shape[1], cu_seqlens=cu_seqlens
    )
    chunk_states = torch.empty((chunk_rows, *state_tail), device=device, dtype=q.dtype)
    state_indices = torch.arange(nseq, device=device, dtype=torch.int32)
    recurrence_config = _kda_ops.make_kda_ops_config(
        q.dtype,
        state_dtype=q.dtype if initial_state is None else initial_state.dtype,
        cu_seqlens_dtype=cu_seqlens.dtype if cu_seqlens is not None else None,
        has_state_in=initial_state is not None,
        has_state_out=False,
        has_chunk_states_out=True,
        state_fp32=initial_state is not None and initial_state.dtype == torch.float32,
        has_gate_params=safe_gate,
        is_varlen=cu_seqlens is not None,
        normalize_qk=bool(use_qk_l2norm_in_kernel),
    )
    # `output` is only used to satisfy the FFI contract here.
    scratch_out = torch.empty_like(v)
    _kda_ops.get_kda_recurrence_ops_module(recurrence_config).get_function(
        _kda_ops.get_kda_recurrence_ops_function_name(recurrence_config)
    )(
        v,
        scratch_out,
        workspace_decayed,
        workspace_k_restored,
        workspace_inverse,
        workspace_p,
        workspace_total,
        state_indices,
        initial_state,
        None,
        cu_seqlens,
        chunk_states,
    )

    # ---- K3: the reverse chunk scan, publishing the dstate pool ------------
    # One CTA per (sequence, head, 64-wide value block); it carries dS and
    # publishes the *incoming* dS of every chunk so K4 can be chunk-parallel.
    dstate = torch.empty((chunk_rows, *state_tail), device=device, dtype=q.dtype)
    grad_config = _kda_grad_ops.make_kda_backward_grad_ops_config(
        q.dtype,
        gate_dtype=g.dtype,
        beta_dtype=beta.dtype,
        safe_gate=safe_gate,
        use_beta_sigmoid=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        gate_domain=gate_domain,
        cu_seqlens_dtype=cu_seqlens.dtype if cu_seqlens is not None else None,
        is_varlen=cu_seqlens is not None,
        normalize_qk=bool(use_qk_l2norm_in_kernel),
        reuse_prepare=reuse_prepare,
    )
    precise_scan = (
        precise_state_scan or allow_neg_eigval or not use_beta_sigmoid_in_kernel
    )
    if not precise_scan:
        scan_config = _kda_scan_ops.make_kda_backward_scan_ops_config(
            q.dtype,
            cu_seqlens_dtype=cu_seqlens.dtype if cu_seqlens is not None else None,
            is_varlen=cu_seqlens is not None,
        )
        _kda_scan_ops.get_kda_backward_scan_ops_module(scan_config).get_function(
            _kda_scan_ops.get_kda_backward_scan_ops_function_name(scan_config)
        )(
            dO,
            workspace_decayed,
            workspace_k_restored,
            workspace_inverse,
            workspace_p,
            workspace_total,
            dstate,
            dstate0,
            d_final_state,
            cu_seqlens,
        )

    if precise_scan or initial_state is not None:
        _kda_grad_ops.get_kda_backward_grad_ops_module(grad_config).get_function(
            _kda_grad_ops.get_kda_backward_grad_ops_function_name(grad_config)
            + "_initial_state"
        )(
            q,
            k,
            g,
            beta,
            dO,
            dstate0,
            d_final_state,
            A_log,
            dt_bias,
            float(scale),
            float(lower_bound),
            cu_seqlens,
            dstate if precise_scan else None,
        )

    # ---- K4: the chunk-parallel gradients ---------------------------------
    _kda_grad_ops.get_kda_backward_grad_ops_module(grad_config).get_function(
        _kda_grad_ops.get_kda_backward_grad_ops_function_name(grad_config)
    )(
        q,
        k,
        v,
        g,
        beta,
        dO,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        workspace_decayed,
        workspace_k_restored,
        workspace_inverse,
        workspace_p,
        workspace_total,
        chunk_states,
        dstate,
        A_log,
        dt_bias,
        float(scale),
        float(lower_bound),
        cu_seqlens,
        *((prepare_cache,) if reuse_prepare else ()),
    )

    if squeeze_varlen:
        dq = dq.squeeze(0)
        dk = dk.squeeze(0)
        dv = dv.squeeze(0)
        dg = dg.squeeze(0)
        dbeta = dbeta.squeeze(0)
    if initial_state is None:
        dstate0 = dstate0.new_empty(0)
    return dq, dk, dv, dg, dbeta, dstate0


__all__ = ["chunk_kda_bwd"]
