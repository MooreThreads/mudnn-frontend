"""Device-side safety path for empty/non-singleton packed decode sequences.

No MTP cache or rollback is exposed. A second launch keeps sequence lengths on
the device while the original tuned pipeline handles all singleton sequences.
"""

import tilelang
import tilelang.language as T
import torch


@tilelang.jit
def recurrent_kernel(
    hq,
    hv,
    io_dtype,
    g_dtype,
    beta_dtype,
    cu_dtype,
    bias_dtype,
    use_indices,
    store_final,
    normalize,
    safe_gate,
    linear_gate,
    sigmoid_beta,
    negative_eigenvalues,
    has_a_log,
    has_bias,
):
    total, n, pool = T.dynamic("total"), T.dynamic("n"), T.dynamic("pool")
    qs, qh, qd = T.dynamic("qs"), T.dynamic("qh"), T.dynamic("qd")
    ks, kh, kd = T.dynamic("ks"), T.dynamic("kh"), T.dynamic("kd")
    vs, vh, vd = T.dynamic("vs"), T.dynamic("vh"), T.dynamic("vd")
    gs, gh = T.dynamic("gs"), T.dynamic("gh")
    bs, bh = T.dynamic("bs"), T.dynamic("bh")
    index_stride = T.dynamic("index_stride")
    tile_v = 4

    @T.prim_func
    def recurrent(
        q: T.StridedTensor((total, hq, 128), (qs, qh, qd), io_dtype),
        k: T.StridedTensor((total, hq, 128), (ks, kh, kd), io_dtype),
        v: T.StridedTensor((total, hv, 128), (vs, vh, vd), io_dtype),
        g: T.StridedTensor((total, hv), (gs, gh), g_dtype),
        beta: T.StridedTensor((total, hv), (bs, bh), beta_dtype),
        cu: T.Tensor((n + 1,), cu_dtype),
        h0: T.Tensor((pool, hv, 128, 128), "float32"),
        ht: T.Tensor((pool, hv, 128, 128), "float32"),
        indices: T.StridedTensor((n,), (index_stride,), "int32"),
        a_log: T.Tensor((hv,), "float32"),
        bias: T.Tensor((hv,), bias_dtype),
        o: T.Tensor((total, hv, 128), io_dtype),
        scale: T.float32,
    ):
        with T.Kernel(n, hv, 128 // tile_v, threads=128) as (seq, head, slab):
            start, end = cu[seq], cu[seq + 1]
            slot = seq
            if use_indices:
                slot = indices[seq]
            if end - start != 1:
                state = T.alloc_fragment((tile_v, 128), "float32")
                products = T.alloc_fragment((tile_v, 128), "float32")
                qr = T.alloc_fragment((128,), "float32")
                kr = T.alloc_fragment((128,), "float32")
                sq = T.alloc_fragment((128,), "float32")
                sk = T.alloc_fragment((128,), "float32")
                qsum = T.alloc_fragment((1,), "float32")
                ksum = T.alloc_fragment((1,), "float32")
                pred = T.alloc_fragment((tile_v,), "float32")
                out = T.alloc_fragment((tile_v,), "float32")
                alpha = T.alloc_var("float32")
                strength = T.alloc_var("float32")
                if slot >= 0:
                    for row, dim in T.Parallel(tile_v, 128):
                        state[row, dim] = h0[slot, head, slab * tile_v + row, dim]
                    for token in T.serial(start, end):
                        for dim in T.Parallel(128):
                            qr[dim] = T.cast(q[token, head // (hv // hq), dim], "float32")
                            kr[dim] = T.cast(k[token, head // (hv // hq), dim], "float32")
                        if normalize:
                            for dim in T.Parallel(128):
                                sq[dim] = qr[dim] * qr[dim]
                                sk[dim] = kr[dim] * kr[dim]
                            T.reduce_sum(sq, qsum, dim=0)
                            T.reduce_sum(sk, ksum, dim=0)
                            for dim in T.Parallel(128):
                                qr[dim] *= T.rsqrt(qsum[0] + 1e-6)
                                kr[dim] *= T.rsqrt(ksum[0] + 1e-6)
                        alpha = T.cast(g[token, head], "float32")
                        if safe_gate:
                            if has_bias:
                                alpha += T.cast(bias[head], "float32")
                            alpha = T.if_then_else(alpha > 20.0, alpha, T.log(1.0 + T.exp(alpha)))
                            if has_a_log:
                                alpha *= T.exp(a_log[head])
                            alpha = T.exp(-alpha)
                        elif linear_gate:
                            alpha = T.max(alpha, 1e-10)
                        else:
                            alpha = T.exp(alpha)
                        strength = T.cast(beta[token, head], "float32")
                        if sigmoid_beta:
                            strength = 1.0 / (1.0 + T.exp(-strength))
                            if negative_eigenvalues:
                                strength *= 2.0
                        for row, dim in T.Parallel(tile_v, 128):
                            state[row, dim] *= alpha
                            products[row, dim] = state[row, dim] * kr[dim]
                        T.reduce_sum(products, pred, dim=1)
                        for row, dim in T.Parallel(tile_v, 128):
                            state[row, dim] += strength * (T.cast(v[token, head, slab * tile_v + row], "float32") - pred[row]) * kr[dim]
                            products[row, dim] = state[row, dim] * qr[dim] * scale
                        T.reduce_sum(products, out, dim=1)
                        for row in T.Parallel(tile_v):
                            o[token, head, slab * tile_v + row] = out[row]
                    if store_final:
                        for row, dim in T.Parallel(tile_v, 128):
                            ht[slot, head, slab * tile_v + row, dim] = state[row, dim]
                else:
                    for token in T.serial(start, end):
                        for row in T.Parallel(tile_v):
                            o[token, head, slab * tile_v + row] = T.cast(0, io_dtype)

    return recurrent


def run_non_singleton(q, k, v, g, beta, cu, h0, ht, indices, a_log, bias, o, flags):
    n, hv = cu.numel() - 1, v.shape[1]
    has_a_log, has_bias = a_log is not None, bias is not None
    if indices is None:
        indices = torch.empty(n, dtype=torch.int32, device=q.device)
    if a_log is None:
        a_log = torch.empty(hv, dtype=torch.float32, device=q.device)
    if bias is None:
        bias = torch.empty(hv, dtype=torch.float32, device=q.device)
    kernel = recurrent_kernel(
        q.shape[1],
        hv,
        q.dtype,
        g.dtype,
        beta.dtype,
        cu.dtype,
        bias.dtype,
        flags.get("has_indices", False),
        ht is not None,
        flags["use_qk_l2norm_in_kernel"],
        flags["safe_gate"],
        flags["gate_domain"] == "linear",
        flags["use_beta_sigmoid_in_kernel"],
        flags["allow_neg_eigval"],
        has_a_log,
        has_bias,
    )
    kernel(q, k, v, g, beta, cu, h0, h0 if ht is None else ht, indices, a_log, bias, o, float(flags["scale"]))
