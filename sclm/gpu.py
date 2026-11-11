# ctypes binding of the engine's GPU kernels (engine/gpu.cu), called on PyTorch device memory: the kernels run in
# PyTorch's CUDA context on its default stream, in order with its own operations.
import ctypes as C
import os

import torch

DLL = os.path.join(os.path.dirname(__file__), "..", "engine", "build", "sclm_gpu.dll")
active = False  # the host turns this on around a forward whose single-token steps should use the kernels
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = C.CDLL(os.path.abspath(DLL))  # finds the CUDA runtime torch has already loaded
        _lib.sclm_gpu_act_quant.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
        _lib.sclm_gpu_fp8_linear.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        _lib.sclm_gpu_fp4_expert.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_float, C.c_void_p, C.c_float, C.c_void_p]
        _lib.sclm_gpu_hc_mixes.argtypes = [C.c_void_p] * 5 + [C.c_int, C.c_int, C.c_float, C.c_float, C.c_int]
        _lib.sclm_gpu_hc_pre_norm.argtypes = [C.c_void_p] * 4 + [C.c_int, C.c_int, C.c_float]
        _lib.sclm_gpu_hc_post.argtypes = [C.c_void_p] * 5 + [C.c_int, C.c_int]
        _lib.sclm_gpu_fp8_expert.argtypes = [C.c_void_p] * 6 + [C.c_int, C.c_int, C.c_float, C.c_void_p, C.c_void_p, C.c_int]
        _lib.sclm_gpu_gate.argtypes = [C.c_void_p] * 4 + [C.c_int] * 3 + [C.c_float, C.c_float, C.c_int]
        _lib.sclm_gpu_moe_finish.argtypes = [C.c_void_p] * 3 + [C.c_int]
        _lib.sclm_gpu_attn_q.argtypes = [C.c_void_p] * 9 + [C.c_int] * 5 + [C.c_float]
        _lib.sclm_gpu_attn_kv.argtypes = [C.c_void_p] * 6 + [C.c_int] * 3 + [C.c_float]
        _lib.sclm_gpu_sparse_attn.argtypes = [C.c_void_p] * 6 + [C.c_int] * 4 + [C.c_float]
        _lib.sclm_gpu_attn_out.argtypes = [C.c_void_p] * 7 + [C.c_int] * 6
        _lib.sclm_gpu_compress.argtypes = [C.c_void_p] * 6 + [C.c_int] * 2 + [C.c_void_p] * 5 + [C.c_int] * 4 + [C.c_float]
    return _lib


def _check(status: int) -> None:
    assert status == 0, f"GPU kernel launch failed ({status})"


def act_quant(x: torch.Tensor) -> torch.Tensor:
    """The reference's fp8 fake-quant (blocks of 32, power-of-two scales) of a bf16 vector, as fp32."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.numel() % 32 == 0
    xq = torch.empty(x.numel(), dtype=torch.float32, device=x.device)
    _check(lib().sclm_gpu_act_quant(x.data_ptr(), xq.data_ptr(), x.numel()))
    return xq


def fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, groups: int = 1) -> torch.Tensor:
    """upstream linear() for one token and an fp8 weight [n, k] with block scales: x (bf16, groups * k values) ->
    bf16 [n]. With groups > 1 the weight is block-diagonal: rows i * n / groups .. multiply x's i-th k values."""
    n, k = weight.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.numel() == groups * k and weight.is_contiguous()
    y = torch.empty(n, dtype=torch.bfloat16, device=x.device)
    _check(lib().sclm_gpu_fp8_linear(weight.data_ptr(), scale.data_ptr(), n, k, groups, x.data_ptr(), y.data_ptr()))
    return y


def fp4_expert(slot: torch.Tensor, dim: int, inter: int, limit: float, xq: torch.Tensor, weight: float, y: torch.Tensor) -> None:
    """y (fp32 [dim]) += weight * the routed expert held in `slot` (its checkpoint bytes: three scales, then three
    packed-FP4 weights) applied to the layer input quantized as xq (act_quant)."""
    assert slot.is_contiguous() and xq.dtype == y.dtype == torch.float32 and xq.numel() == y.numel() == dim
    _check(lib().sclm_gpu_fp4_expert(slot.data_ptr(), dim, inter, limit, xq.data_ptr(), weight, y.data_ptr()))


def hc_mixes(h: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, norm_eps: float,
             eps: float, iters: int) -> torch.Tensor:
    """upstream Block.hc_mixes for one token: the stream h (bf16, hc * dim values) -> fp32 [(2 + hc) * hc], which is
    pre[hc], post[hc] and comb[hc][hc] back to back."""
    assert h.dtype == torch.bfloat16 and h.is_contiguous() and fn.dtype == scale.dtype == base.dtype == torch.float32
    out = torch.empty((2 + hc) * hc, dtype=torch.float32, device=h.device)
    _check(lib().sclm_gpu_hc_mixes(h.data_ptr(), fn.data_ptr(), scale.data_ptr(), base.data_ptr(), out.data_ptr(),
                                   h.numel() // hc, hc, norm_eps, eps, iters))
    return out


def hc_pre_norm(h: torch.Tensor, pre: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm (bf16 weight [dim]) of upstream Block.hc_pre(h, pre) for one token -> bf16 [dim]."""
    hc, dim = pre.numel(), weight.numel()
    assert h.dtype == weight.dtype == torch.bfloat16 and h.is_contiguous() and pre.dtype == torch.float32 and pre.is_contiguous()
    out = torch.empty(dim, dtype=torch.bfloat16, device=h.device)
    _check(lib().sclm_gpu_hc_pre_norm(h.data_ptr(), pre.data_ptr(), weight.data_ptr(), out.data_ptr(), dim, hc, eps))
    return out


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """upstream Block.hc_post for one token: sublayer output x (bf16 [dim]) and the stream before it -> bf16 [hc, dim]."""
    hc, dim = post.numel(), x.numel()
    assert x.dtype == residual.dtype == torch.bfloat16 and x.is_contiguous() and residual.is_contiguous()
    assert post.is_contiguous() and comb.is_contiguous() and comb.numel() == hc * hc
    out = torch.empty(hc, dim, dtype=torch.bfloat16, device=x.device)
    _check(lib().sclm_gpu_hc_post(x.data_ptr(), residual.data_ptr(), post.data_ptr(), comb.data_ptr(), out.data_ptr(), dim, hc))
    return out


def fp8_expert(expert, limit: float, xq: torch.Tensor, y: torch.Tensor, store: bool = False) -> None:
    """y (fp32 [dim]) += an upstream Expert with fp8 weights (the shared expert) on the layer input quantized as xq;
    store: y is overwritten instead."""
    w1, w2, w3 = expert.w1.weight, expert.w2.weight, expert.w3.weight
    assert xq.dtype == y.dtype == torch.float32 and xq.numel() == y.numel() == w1.size(1)
    _check(lib().sclm_gpu_fp8_expert(w1.data_ptr(), w1.scale.data_ptr(), w2.data_ptr(), w2.scale.data_ptr(), w3.data_ptr(),
                                     w3.scale.data_ptr(), w1.size(1), w1.size(0), limit, xq.data_ptr(), y.data_ptr(), store))


def attn_q(wq_a, norm_weight: torch.Tensor, wq_b, x: torch.Tensor, freqs: int, heads: int, head_dim: int, rd: int,
           eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The query side of upstream Attention.forward for one token x (bf16 [dim]): (qr [rank], q [heads * head_dim]),
    q carrying the rotary embedding. wq_a and wq_b are upstream fp8 Linears; freqs is the address of the position's
    row of freqs_cis."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    rank = wq_a.weight.size(0)
    qr = torch.empty(rank, dtype=torch.bfloat16, device=x.device)
    q = torch.empty(heads * head_dim, dtype=torch.bfloat16, device=x.device)
    _check(lib().sclm_gpu_attn_q(wq_a.weight.data_ptr(), wq_a.scale.data_ptr(), norm_weight.data_ptr(), wq_b.weight.data_ptr(),
                                 wq_b.scale.data_ptr(), x.data_ptr(), freqs, qr.data_ptr(), q.data_ptr(), x.numel(), rank,
                                 heads, head_dim, rd, eps))
    return qr, q


def attn_kv(wkv, norm_weight: torch.Tensor, x: torch.Tensor, freqs: int, kv: torch.Tensor, rd: int, eps: float) -> None:
    """Writes one token's key/value (upstream Attention._window_kv: projected, normed, rotated, fp8-quantized) into
    kv, its slot of the window ring (bf16 [head_dim])."""
    assert x.dtype == kv.dtype == torch.bfloat16 and x.is_contiguous() and kv.is_contiguous()
    _check(lib().sclm_gpu_attn_kv(wkv.weight.data_ptr(), wkv.scale.data_ptr(), norm_weight.data_ptr(), x.data_ptr(), freqs,
                                  kv.data_ptr(), x.numel(), kv.numel(), rd, eps))


def compress(compressor, indexer, x: torch.Tensor, pos: int, freqs_cis: torch.Tensor, kv_cache: torch.Tensor, rd: int) -> None:
    """One decode token x (bf16 [dim]) at position pos through an upstream Compressor: its state takes the token,
    and when the token completes a group of compress_ratio, the group's rotated FP4 latent goes into kv_cache
    ([groups, d]) and its index key into the k_cache of `indexer` (an upstream Indexer that owns the keys, or None).
    What upstream's Attention._compress_kv and Indexer.forward do for it; freqs_cis is the attention's."""
    ratio, d = compressor.compress_ratio, compressor.head_dim
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and kv_cache.is_contiguous()
    pooled = ratio > 1
    freqs = freqs_cis.data_ptr() + (pos + 1 - ratio) * rd * 4  # of the group's first position
    key = indexer is not None and indexer.owns_k
    _check(lib().sclm_gpu_compress(
        compressor.wkv.weight.data_ptr(), compressor.wgate.weight.data_ptr() if pooled else None,
        compressor.norm.weight.data_ptr(), x.data_ptr(), compressor.kv_state.data_ptr() if pooled else None,
        compressor.score_state.data_ptr() if pooled else None, ratio, pos % ratio,
        indexer.wk.weight.data_ptr() if key else None, indexer.k_norm.weight.data_ptr() if key else None, freqs,
        indexer.k_cache[0, pos // ratio].data_ptr() if key else None, kv_cache[pos // ratio].data_ptr(), x.numel(), d,
        indexer.index_head_dim if key else 32, rd, compressor.norm.eps))


def sparse_attn(q: torch.Tensor, window: torch.Tensor, compressed: torch.Tensor, idx: torch.Tensor, sink: torch.Tensor,
                heads: int, win: int, scale: float) -> torch.Tensor:
    """upstream sparse_attn for one token: q (bf16 [heads * d]) over the positions idx (int32) names, a position below
    `win` in `window` ([win, d]) and the rest in `compressed` ([.., d]), -1 for none -> bf16 [heads * d]."""
    assert q.dtype == window.dtype == compressed.dtype == torch.bfloat16 and idx.dtype == torch.int32 and sink.dtype == torch.float32
    assert q.is_contiguous() and window.is_contiguous() and compressed.is_contiguous() and idx.is_contiguous()
    out = torch.empty_like(q)
    _check(lib().sclm_gpu_sparse_attn(q.data_ptr(), window.data_ptr(), compressed.data_ptr(), idx.data_ptr(), sink.data_ptr(),
                                      out.data_ptr(), heads, q.numel() // heads, idx.numel(), win, scale))
    return out


def attn_out(o: torch.Tensor, freqs: int, wo_a_weight: torch.Tensor, wo_a_scale: torch.Tensor, wo_b, heads: int, rd: int,
             groups: int) -> torch.Tensor:
    """The output side of upstream Attention.forward for one token: o (bf16 [heads * head_dim], changed in place)
    -> bf16 [dim]. wo_a is fp8 with block scales, block-diagonal over `groups`; wo_b an upstream fp8 Linear."""
    assert o.dtype == torch.bfloat16 and o.is_contiguous()
    dim = wo_b.weight.size(0)
    out = torch.empty(dim, dtype=torch.bfloat16, device=o.device)
    _check(lib().sclm_gpu_attn_out(o.data_ptr(), freqs, wo_a_weight.data_ptr(), wo_a_scale.data_ptr(), wo_b.weight.data_ptr(),
                                   wo_b.scale.data_ptr(), out.data_ptr(), heads, o.numel() // heads, rd, groups,
                                   wo_a_weight.size(0), dim))
    return out


def gate(module, x: torch.Tensor) -> torch.Tensor:
    """upstream Gate.forward ("sqrtsoftplus" scores) for one token x (bf16 [dim]) -> fp32 [2 * topk + dim]: the picked
    experts' indices, their weights, then x as fp32, so that one transfer gives the host all it needs."""
    assert module.score_func == "sqrtsoftplus" and x.dtype == module.weight.dtype == torch.bfloat16 and x.is_contiguous()
    n, dim = module.weight.shape
    pack = torch.empty(2 * module.topk + dim, dtype=torch.float32, device=x.device)
    _check(lib().sclm_gpu_gate(module.weight.data_ptr(), module.bias.data_ptr(), x.data_ptr(), pack.data_ptr(), n, module.topk,
                               dim, module.gate_temp, module.route_scale, module.norm_topk_prob))
    return pack


def moe_finish(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    """bf16 of routed + shared (fp32 [dim] each)."""
    assert routed.dtype == shared.dtype == torch.float32 and routed.numel() == shared.numel()
    out = torch.empty(routed.numel(), dtype=torch.bfloat16, device=routed.device)
    _check(lib().sclm_gpu_moe_finish(routed.data_ptr(), shared.data_ptr(), out.data_ptr(), routed.numel()))
    return out
