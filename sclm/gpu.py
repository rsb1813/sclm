# ctypes binding of the engine's GPU kernels (engine/gpu.cu), called on PyTorch device memory: the kernels run in
# PyTorch's CUDA context on its default stream, in order with its own operations.
import ctypes as C
import os
from functools import lru_cache
from types import SimpleNamespace

import torch

from . import ACT_BLOCK, MODEL

DLL = os.path.join(os.path.dirname(__file__), "..", "engine", "build", "sclm_gpu.dll")
ATTENDED = 2304  # the most positions sparse_attn takes (engine/gpu.cu)
active = False  # the host turns this on around a forward whose single-token steps should use the kernels
# the host turns this on around a forward of many tokens that continue a sequence (Host.prefill): attention computes
# them as tensors (host_v4.V4Attention.together), and the MoE with the original experts (TieredMoE.together)
together = False
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = C.CDLL(os.path.abspath(DLL))  # finds the CUDA runtime torch has already loaded
        # the model's activation block, the block of an FP8 weight's scales, the window KV's block and whether its
        # rotated values are rounded, the query's head norm, the rounding in front of wo_a (engine/gpu.cu, Model)
        _lib.sclm_gpu_set_model.argtypes = [C.c_int] * 6
        _lib.sclm_gpu_set_model(*((ACT_BLOCK, 128, 64, 0, 1, 0) if MODEL == "v4" else (ACT_BLOCK, 32, 32, 1, 0, 1)))
        _lib.sclm_gpu_act_quant.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
        _lib.sclm_gpu_fp8_linear.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        _lib.sclm_gpu_fp4_experts.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int,
                                              C.c_float, C.c_void_p, C.c_void_p]
        _lib.sclm_gpu_hc_mixes.argtypes = [C.c_void_p] * 5 + [C.c_int, C.c_int, C.c_float, C.c_float, C.c_int]
        _lib.sclm_gpu_hc_pre_norm.argtypes = [C.c_void_p] * 4 + [C.c_int, C.c_int, C.c_float]
        _lib.sclm_gpu_hc_post.argtypes = [C.c_void_p] * 5 + [C.c_int, C.c_int]
        _lib.sclm_gpu_fp8_expert.argtypes = [C.c_void_p] * 6 + [C.c_int, C.c_int, C.c_float, C.c_void_p, C.c_void_p, C.c_int]
        _lib.sclm_gpu_gate.argtypes = [C.c_void_p] * 4 + [C.c_int] * 3 + [C.c_float, C.c_float, C.c_int, C.c_void_p]
        _lib.sclm_gpu_gate_guess.argtypes = [C.c_void_p] * 6 + [C.c_int] * 4 + [C.c_float, C.c_float]
        _lib.sclm_gpu_moe_begin.argtypes = ([C.c_void_p] * 5 + [C.c_int] * 3 + [C.c_float] * 2 + [C.c_int] + [C.c_void_p] * 5 +
                                            [C.c_int] * 2 + [C.c_float] * 2 + [C.c_void_p] * 6 + [C.c_int, C.c_float] +
                                            [C.c_void_p] * 4)
        _lib.sclm_gpu_pool.argtypes = [C.c_void_p] * 7 + [C.c_int] * 3 + [C.c_void_p] * 2 + [C.c_int] * 3 + [C.c_float, C.c_int, C.c_int]
        _lib.sclm_gpu_moe_guess.argtypes = []
        _lib.sclm_gpu_moe_lane.argtypes = [C.c_int]
        _lib.sclm_gpu_moe_wait.argtypes = [C.c_int, C.c_int]
        _lib.sclm_gpu_moe_end.argtypes = [C.c_void_p] * 5 + [C.c_int]
        _lib.sclm_gpu_copy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        _lib.sclm_gpu_copies_wait.argtypes = []
        _lib.sclm_gpu_attn_q.argtypes = [C.c_void_p] * 9 + [C.c_int] * 5 + [C.c_float]
        _lib.sclm_gpu_attn_kv.argtypes = [C.c_void_p] * 6 + [C.c_int] * 3 + [C.c_float]
        _lib.sclm_gpu_sparse_attn.argtypes = [C.c_void_p] * 6 + [C.c_int] * 4 + [C.c_float]
        _lib.sclm_gpu_attn_out.argtypes = [C.c_void_p] * 7 + [C.c_int] * 6
        _lib.sclm_gpu_compress.argtypes = [C.c_void_p] * 6 + [C.c_int] * 2 + [C.c_void_p] * 5 + [C.c_int] * 4 + [C.c_float]
        _lib.sclm_gpu_index_pick.argtypes = [C.c_void_p] * 8 + [C.c_int] * 8 + [C.c_float, C.c_int]
    return _lib


def _check(status: int) -> None:
    assert status == 0, f"GPU kernel launch failed ({status})"


LATER = object()  # in bind(): an argument given at each launch


def bind(name: str, *args):
    """The kernel sclm_gpu_`name` with its arguments (those of engine/gpu.cu) bound once, a tensor as its address
    (the function keeps the tensor). Returns the function that launches it, taking the LATER arguments in order.
    For the kernels of a decode token, where a wrapper's checks, address lookups and fresh output tensor cost as
    much as the launch."""
    fn = getattr(lib(), "sclm_gpu_" + name)
    raw = [a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args]
    later = [i for i, a in enumerate(raw) if a is LATER]

    def launch(*given):
        for i, value in zip(later, given):
            raw[i] = value
        _check(fn(*raw))

    launch.bound = args
    return launch


def act_quant(x: torch.Tensor) -> torch.Tensor:
    """The reference's fp8 fake-quant (the model's blocks, power-of-two scales) of a bf16 vector, as fp32."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.numel() % ACT_BLOCK == 0
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


_SLOTS, _WEIGHTS = (C.c_int * 8)(), (C.c_float * 8)()  # fp4_experts' arguments: the engine reads them during the call


def fp4_experts(store: torch.Tensor, slots, weights, dim: int, inter: int, limit: float, xq: torch.Tensor, y: torch.Tensor) -> None:
    """y (fp32 [dim]) += the sum, in the order given, of weights[i] * the routed expert in row slots[i] of `store`
    (uint8 [rows, bytes], a row being an expert's checkpoint bytes: three scales, then three packed-FP4 weights)
    applied to the layer input quantized as xq (act_quant). One to eight experts."""
    n = len(slots)
    assert store.dim() == 2 and store.is_contiguous() and xq.dtype == y.dtype == torch.float32 and xq.numel() == y.numel() == dim
    _SLOTS[:n], _WEIGHTS[:n] = slots, weights
    _check(lib().sclm_gpu_fp4_experts(store.data_ptr(), store.size(1), _SLOTS, _WEIGHTS, n, dim, inter, limit, xq.data_ptr(),
                                      y.data_ptr()))


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


def pool(compressor, x: torch.Tensor, pos: int, freqs_cis: torch.Tensor, norm_weight: torch.Tensor, out: torch.Tensor) -> None:
    """One decode token x (bf16 [dim]) at position pos > 0 through a Compressor of V4-Flash-0731: its state takes the
    token, and when the token completes a group of compress_ratio, out (bf16 [head_dim], the group's row of the cache
    the compressor fills) gets the group's latent: normed (norm_weight: the norm's weight as bf16), rotated, and
    rounded as upstream rounds it (FP8 on the unrotated values, or the index's Hadamard-rotated FP4 keys for a
    compressor made with rotate). freqs_cis: the attention's rotations. The projections are upstream's fp32 ones or
    the host's bf16 ones (host.Bf16Proj)."""
    ratio, d, rd = compressor.compress_ratio, compressor.head_dim, compressor.rope_head_dim
    assert x.dtype == out.dtype == norm_weight.dtype == torch.bfloat16 and x.is_contiguous() and out.numel() == d
    _check(lib().sclm_gpu_pool(
        compressor.wkv.weight.data_ptr(), compressor.wgate.weight.data_ptr(), compressor.ape.data_ptr(), norm_weight.data_ptr(),
        x.data_ptr(), compressor.kv_state.data_ptr(), compressor.score_state.data_ptr(), ratio, compressor.overlap, pos % ratio,
        freqs_cis.data_ptr() + (pos + 1 - ratio) * rd * 4, out.data_ptr(), x.numel(), d, rd, compressor.norm.eps,
        compressor.rotate, compressor.wkv.weight.dtype == torch.bfloat16))


def index_pick(indexer, qr: torch.Tensor, x: torch.Tensor, freqs: int, keys: torch.Tensor, count: int, out: torch.Tensor,
               offset: int) -> None:
    """One decode token through an upstream Indexer once it has to choose: out (int32 [index_topk]) = the positions
    of the index_topk highest scores among the first `count` of keys (bf16 [.., index_head_dim]), in order, each plus
    offset. qr is the attention's normed low-rank query and x the token (bf16), freqs the address of the position's
    rotation. Scores are upstream's up to their last bit; of equal ones the earlier position is kept.
    V4-Flash-0731's Indexer rotates its query (rotate_activation) before rounding it."""
    v4 = MODEL == "v4"
    heads, d = indexer.n_local_heads, indexer.head_dim if v4 else indexer.index_head_dim
    assert qr.dtype == x.dtype == keys.dtype == torch.bfloat16 and out.dtype == torch.int32 and keys.is_contiguous()
    assert indexer.wq_b.weight.dtype == torch.float8_e4m3fn and out.numel() == indexer.index_topk < count <= keys.size(0)
    _check(lib().sclm_gpu_index_pick(
        indexer.wq_b.weight.data_ptr(), indexer.wq_b.scale.data_ptr(), qr.data_ptr(), indexer.weights_proj.weight.data_ptr(),
        x.data_ptr(), freqs, keys.data_ptr(), out.data_ptr(), qr.numel(), heads, d, indexer.rope_head_dim, x.numel(), count,
        out.numel(), offset, indexer.softmax_scale * indexer.n_heads ** -0.5, v4))


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
    """The output side of upstream Attention.forward for one token: o (bf16 [heads * head_dim]) -> bf16 [dim]. wo_a is fp8 with block scales, block-diagonal over `groups`; wo_b an upstream fp8 Linear."""
    assert o.dtype == torch.bfloat16 and o.is_contiguous()
    dim = wo_b.weight.size(0)
    out = torch.empty(dim, dtype=torch.bfloat16, device=o.device)
    _check(lib().sclm_gpu_attn_out(o.data_ptr(), freqs, wo_a_weight.data_ptr(), wo_a_scale.data_ptr(), wo_b.weight.data_ptr(),
                                   wo_b.scale.data_ptr(), out.data_ptr(), heads, o.numel() // heads, rd, groups,
                                   wo_a_weight.size(0), dim))
    return out


def gate_args(module) -> tuple:
    """(the bias's address or None, the temperature, whether the picks' weights are normalized) of an upstream Gate:
    V4.1's has all three as attributes; V4-Flash-0731's has no temperature, always normalizes, and has no bias in
    the layers that route by the token id."""
    bias = module.bias.data_ptr() if module.bias is not None else None
    return bias, getattr(module, "gate_temp", 1.0), getattr(module, "norm_topk_prob", True)


def gate(module, x: torch.Tensor, room: int = 0, fixed: torch.Tensor | None = None) -> torch.Tensor:
    """upstream Gate.forward ("sqrtsoftplus" scores) for one token x (bf16 [dim]) -> fp32 [2 * topk + dim]: the picked
    experts' indices, their weights, then x as fp32, so that one transfer gives the host all it needs. room: floats
    left free after them (for gate_guess). fixed (int32 [topk]): the experts to weight instead of the gate's choice
    (V4-Flash-0731's layers that route by the token id)."""
    assert module.score_func == "sqrtsoftplus" and x.dtype == module.weight.dtype == torch.bfloat16 and x.is_contiguous()
    assert fixed is None or (fixed.dtype == torch.int32 and fixed.numel() == module.topk and fixed.is_contiguous())
    n, dim = module.weight.shape
    bias, temp, norm = gate_args(module)
    pack = torch.empty(2 * module.topk + dim + room, dtype=torch.float32, device=x.device)
    _check(lib().sclm_gpu_gate(module.weight.data_ptr(), bias, x.data_ptr(), pack.data_ptr(), n, module.topk, dim, temp,
                               module.route_scale, norm, None if fixed is None else fixed.data_ptr()))
    return pack


def gate_guess(module, norm_weight: torch.Tensor, h: torch.Tensor, pre: torch.Tensor, eps: float, out: torch.Tensor) -> None:
    """out (fp32 [2 * k]) = the k experts the gate `module` would pick, then their scores, if its layer's input were
    the RMSNorm (bf16 norm_weight) of the stream h mixed by pre (hc_pre_norm): the next layer's gate on this layer's
    input. Called after gate() for the same token."""
    n, dim = module.weight.shape
    assert module.score_func == "sqrtsoftplus" and h.dtype == torch.bfloat16 and h.is_contiguous() and pre.is_contiguous()
    assert out.dtype == pre.dtype == torch.float32 and out.is_contiguous() and norm_weight.numel() == dim
    bias, temp, _ = gate_args(module)
    _check(lib().sclm_gpu_gate_guess(module.weight.data_ptr(), bias, h.data_ptr(), pre.data_ptr(), norm_weight.data_ptr(),
                                     out.data_ptr(), n, out.numel() // 2, dim, pre.numel(), temp, eps))


@lru_cache(2)
def moe_buffers(topk: int, dim: int, guess: int, device: torch.device) -> SimpleNamespace:
    """What moe_begin and moe_end work in, one set for every layer (their kernels run in order on one stream):
    pack (the gate's output and 2 * guess floats for the guess), host (its page-locked copy on the host), and
    the fp32 [dim] vectors xq, shared, routed and ram, and out (bf16 [dim]), the MoE's output until the next
    moe_end."""
    n = 2 * topk + dim + 2 * guess
    vec = lambda k: torch.empty(k, dtype=torch.float32, device=device)  # noqa: E731
    return SimpleNamespace(pack=vec(n), host=torch.empty(n, dtype=torch.float32, device="cpu", pin_memory=True), xq=vec(dim),
                           shared=vec(dim), routed=vec(dim), ram=vec(dim),
                           out=torch.empty(dim, dtype=torch.bfloat16, device=device))


def moe_begin(module, x: torch.Tensor, expert, limit: float, buf: SimpleNamespace, guess: int = 0, later=None,
              norm_weight: torch.Tensor | None = None, h: torch.Tensor | None = None, pre: torch.Tensor | None = None,
              eps: float = 0.0, fixed: torch.Tensor | None = None) -> None:
    """The start of the two-tier MoE for one token x (bf16 [dim]) in one call, into buf (moe_buffers): gate(module, x)
    into buf.pack; with guess, gate_guess(later, norm_weight, h, pre, eps) for `guess` picks after it; buf.routed = 0,
    buf.xq = act_quant(x) and buf.shared = fp8_expert(expert, limit, xq, store=True). Returns when the gate's part
    of the pack is in buf.host, without waiting for the rest; the guess's part is there after moe_guess()."""
    n, dim = module.weight.shape
    w1, w2, w3 = expert.w1.weight, expert.w2.weight, expert.w3.weight
    assert module.score_func == "sqrtsoftplus" and x.dtype == module.weight.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.numel() == dim == w1.size(1) and dim % 32 == 0 and buf.pack.numel() >= 2 * (module.topk + guess) + dim
    if guess:
        assert later.score_func == "sqrtsoftplus" and later.weight.shape == module.weight.shape and norm_weight.numel() == dim
        assert h.dtype == torch.bfloat16 and h.is_contiguous() and pre.dtype == torch.float32 and pre.is_contiguous()
        later_bias, later_temp, _ = gate_args(later)
        ahead = (later.weight.data_ptr(), later_bias, h.data_ptr(), pre.data_ptr(), norm_weight.data_ptr(), guess,
                 pre.numel(), later_temp, eps)
    else:
        ahead = (None, None, None, None, None, 0, 0, 0.0, 0.0)
    bias, temp, norm = gate_args(module)
    _check(lib().sclm_gpu_moe_begin(
        module.weight.data_ptr(), bias, x.data_ptr(), buf.pack.data_ptr(), buf.host.data_ptr(), n, module.topk,
        dim, temp, module.route_scale, norm, *ahead, w1.data_ptr(), w1.scale.data_ptr(),
        w2.data_ptr(), w2.scale.data_ptr(), w3.data_ptr(), w3.scale.data_ptr(), w1.size(0), limit, buf.xq.data_ptr(),
        buf.shared.data_ptr(), buf.routed.data_ptr(), None if fixed is None else fixed.data_ptr()))


def moe_guess() -> None:
    """Waits until the guess of the last moe_begin is on the host."""
    _check(lib().sclm_gpu_moe_guess())


def moe_lane(lane: int) -> None:
    """The lane of the moe_begin calls from now on. 0: a decode token, whose call returns when its routing is on the
    host. Another lane (up to 64): a token computed together with others; its call returns at once, and moe_wait
    waits for what it queued."""
    _check(lib().sclm_gpu_moe_lane(lane))


def moe_wait(lane: int, guess: bool = False) -> None:
    """Waits until the routing of the lane's last moe_begin is on the host, or (guess) its guess."""
    _check(lib().sclm_gpu_moe_wait(lane, guess))


def copy(to: int, source: int, nbytes: int) -> None:
    """Queues a copy of nbytes between page-locked host memory and device memory (addresses, either direction) on a
    stream of its own: it runs beside the kernels, after the copies asked for before it. The caller keeps kernels and
    copies that touch the same memory in order (copies_wait)."""
    _check(lib().sclm_gpu_copy(to, source, nbytes))


def copies_wait() -> None:
    """The kernels launched from now on start when every copy() so far is done."""
    _check(lib().sclm_gpu_copies_wait())


def moe_end(buf: SimpleNamespace, ram: torch.Tensor) -> torch.Tensor:
    """buf.out = bf16 of (buf.routed + ram) + buf.shared: the MoE's output. ram (fp32 [dim], page-locked) is the CPU
    tier's sum on the host; it is uploaded without waiting, so it has to stay as it is until the device was waited
    for."""
    assert ram.dtype == torch.float32 and ram.numel() == buf.ram.numel() and not ram.is_cuda
    _check(lib().sclm_gpu_moe_end(buf.routed.data_ptr(), ram.data_ptr(), buf.ram.data_ptr(), buf.shared.data_ptr(),
                                  buf.out.data_ptr(), ram.numel()))
    return buf.out
