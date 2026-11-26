"""Bring the FP32 per-token scalars (dg, dbeta) back from [H, B, S] to [B, S, H].

They are staged head-outermost because in ``[B, S, H]`` the 64 rows of a chunk
are 128 B apart, so every 4 B store burns its own DDR sector; ``[H, B, S]``
makes a chunk one contiguous line.
"""
import torch
import torch_musa  # noqa: F401
import tilelang
import tilelang.language as T

# Single definition: gdn_shapes.gdn_max_heads() derives its H bound from this
# tile height, and the two must not drift.
from .gdn_shapes import _RELAYOUT_TS as _TS

_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
}


_THREADS = 512

_DTYPES = {
    torch.float32: T.float32,
    torch.bfloat16: T.bfloat16,
    torch.float16: T.float16,
}


@tilelang.jit(target="musa", out_idx=[], pass_configs=_CFG)
def scalar_untranspose_kernel(B, H, beta_dtype=T.float32, TS=_TS,
                              threads=_THREADS):
    accum = T.float32
    S = T.dynamic("S")
    NS = T.ceildiv(S, TS)

    @T.prim_func
    def kernel(
        DGin: T.Tensor([H, B, S], accum),
        DBin: T.Tensor([H, B, S], accum),
        DG: T.Tensor([B, S, H], accum),
        DBeta: T.Tensor([B, S, H], beta_dtype),
    ):
        with T.Kernel(NS, B, threads=threads) as (i_s, i_b):
            s0 = i_s * TS
            gsh = T.alloc_shared([H, TS + 1], accum, scope="shared")
            bsh = T.alloc_shared([H, TS + 1], accum, scope="shared")
            for h, r in T.Parallel(H, TS):
                gsh[h, r] = DGin[h, i_b, s0 + r]
            for h, r in T.Parallel(H, TS):
                bsh[h, r] = DBin[h, i_b, s0 + r]
            T.sync_threads()
            for r, h in T.Parallel(TS, H):
                DG[i_b, s0 + r, h] = gsh[h, r]
            for r, h in T.Parallel(TS, H):
                DBeta[i_b, s0 + r, h] = bsh[h, r].astype(beta_dtype)

    return kernel


def scalar_untranspose(dg_hbs, dbeta_hbs, dbeta_dtype=torch.float32):
    """([H,B,S], [H,B,S]) fp32 -> ([B,S,H] fp32, [B,S,H] dbeta_dtype)."""
    H, B, S = dg_hbs.shape
    if tuple(dbeta_hbs.shape) != (H, B, S):
        raise ValueError("dg/dbeta must share the [H, B, S] handoff layout")
    if S % _TS:
        raise ValueError(f"S must be a multiple of {_TS}")
    kern = scalar_untranspose_kernel(
        B, H, beta_dtype=_DTYPES[dbeta_dtype],
    )
    DG = torch.empty(B, S, H, device=dg_hbs.device, dtype=torch.float32)
    DBeta = torch.empty(B, S, H, device=dg_hbs.device, dtype=dbeta_dtype)
    kern(dg_hbs.contiguous(), dbeta_hbs.contiguous(), DG, DBeta)
    return DG, DBeta
