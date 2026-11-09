# Pure-PyTorch drop-in for the upstream tilelang kernel.py, mirroring its quantization, GEMM and attention numerics.
from typing import Optional

import torch

# The GEMMs below feed bf16 operands to cuBLAS; keep its reductions in fp32 like the upstream kernels.
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

FP8_MAX = 448.0
FP4_MAX = 6.0
# e2m1 magnitudes indexed by the low 3 bits of a code; bit 3 is the sign
FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) for positive normal fp32, read off the exponent/mantissa bits (upstream fast_log2_ceil)."""
    bits = x.float().contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    return exp - 127 + (man != 0).to(torch.int32)


def pow2(e: torch.Tensor) -> torch.Tensor:
    """2**e for integer e, built from the exponent bits (upstream fast_pow2)."""
    return ((e.to(torch.int32) + 127) << 23).view(torch.float32)


def round_fp4(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Round values already clamped to [-6, 6] onto the e2m1 grid, ties to even code.
    Returns the rounded fp32 values and their 4-bit codes."""
    grid = torch.tensor(FP4_GRID, device=v.device)
    a = v.abs()
    hi_idx = torch.bucketize(a, grid).clamp(1, len(FP4_GRID) - 1)
    lo, hi = grid[hi_idx - 1], grid[hi_idx]
    d_lo, d_hi = a - lo, hi - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi_idx % 2 == 0))
    mag_code = torch.where(pick_hi, hi_idx, hi_idx - 1)
    code = mag_code | ((v < 0) & (mag_code > 0)).to(mag_code.dtype) << 3
    return torch.copysign(grid[mag_code], v), code


def pack_fp4(code: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes pairwise along the last dim (even index in the low nibble) into float4_e2m1fn_x2."""
    c = code.to(torch.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).view(torch.float4_e2m1fn_x2)


def unpack_fp4(packed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """float4_e2m1fn_x2 [..., K//2] -> `dtype` [..., K] (every e2m1 value is exact in bf16 and fp32)."""
    table = torch.tensor(FP4_GRID + tuple(-g for g in FP4_GRID), device=packed.device, dtype=dtype)
    b = packed.view(torch.uint8)
    codes = torch.stack([b & 0x0F, b >> 4], dim=-1).flatten(-2)
    return table[codes.long()]


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise FP8 quantization. inplace=True does fused quant+dequant back to x's dtype.
    When scale_fmt is set, scales are rounded up to a power of 2 (MXFP)."""
    N = x.size(-1)
    assert N % block_size == 0
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(-1).clamp_min(1e-4)
    s = pow2(ceil_log2(amax * (1 / FP8_MAX))) if scale_fmt is not None else amax * (1 / FP8_MAX)
    q = (xb / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), s.to(scale_dtype)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
    """FP4 with E8M0 scales for the indexer or E4M3 scales for compressed KV.
    inplace=True writes the dequantized values back to x."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        # training's compressed KV keeps even an all-zero group's scale nonzero
        s = (amax.clamp_min(FP4_MAX * 2**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = pow2(ceil_log2(amax.clamp_min(FP4_MAX * 2**-126) * (1 / FP4_MAX)))
    v, code = round_fp4((xb / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((v * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return pack_fp4(code.flatten(-2)), s.to(scale_dtype)


def _operand_dtype(a_s: torch.Tensor, b_s: torch.Tensor) -> torch.dtype:
    """With power-of-2 (E8M0) scales on both sides, every dequantized fp8/fp4 value is exactly representable in
    bf16, so bf16 operands with fp32 accumulation give the fp32 result on tensor cores. Otherwise stay in fp32."""
    return torch.bfloat16 if a_s.dtype == b_s.dtype == torch.float8_e8m0fnu else torch.float32


def _dequant_act(a: torch.Tensor, a_s: torch.Tensor, block_size: int, dtype: torch.dtype) -> torch.Tensor:
    return (a.to(dtype).unflatten(-1, (-1, block_size)) * a_s.float().to(dtype).unsqueeze(-1)).flatten(-2)


def _mm_fp32_out(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """a @ w.T with an fp32 result, so bf16 operands are not rounded back to bf16 before the caller's cast."""
    a2 = a.reshape(-1, a.size(-1))
    out = torch.mm(a2, w.t(), out_dtype=torch.float32) if a.dtype == torch.bfloat16 else a2 @ w.t()
    return out.view(*a.shape[:-1], w.size(0))


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T with per-block FP8 scaling, accumulated in fp32."""
    assert block_size in (32, 128)
    K, N = a.size(-1), b.size(0)
    assert K % block_size == 0
    assert b_s.shape == ((N + block_size - 1) // block_size, K // block_size)
    dt = _operand_dtype(a_s, b_s)
    sb = b_s.float().to(dt).repeat_interleave(block_size, 0)[:N].repeat_interleave(block_size, 1)
    w = b.to(dt) * sb
    return _mm_fp32_out(_dequant_act(a, a_s, block_size, dt), w).to(torch.get_default_dtype())


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T. A has per-32 or per-128 scales; B has per-32 E8M0 scales on K.
    B is stored as [N, K//2] float4_e2m1fn_x2, two values per byte packed along K."""
    assert act_block_size in (32, 128)
    K, N = a.size(-1), b.size(0)
    assert K % act_block_size == 0
    assert b_s.shape == (N, K // 32)
    dt = _operand_dtype(a_s, b_s)
    w = (unpack_fp4(b, dt).unflatten(-1, (-1, 32)) * b_s.float().to(dt).unsqueeze(-1)).flatten(-2)
    return _mm_fp32_out(_dequant_act(a, a_s, act_block_size, dt), w).to(torch.get_default_dtype())


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    chunk: int = 256,
) -> torch.Tensor:
    """Attention over gathered KV positions (kv is both key and value), with a learned per-head sink logit.
    Index -1 marks an empty slot; a query with no valid slot gets a zero output, as upstream."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    batch = torch.arange(b, device=q.device).view(b, 1, 1)
    sink = attn_sink.float().view(1, 1, h)
    for s0 in range(0, m, chunk):
        idx = topk_idxs[:, s0 : s0 + chunk].long()
        valid = idx >= 0
        kv_g = kv[batch, idx.clamp_min(0)].float() * valid.unsqueeze(-1)  # [b, c, t, d]
        scores = torch.einsum("bchd,bctd->bcht", q[:, s0 : s0 + chunk].float(), kv_g) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
        # upstream starts its running max at -1e30 rather than -inf, which is what zeroes empty rows
        mx = scores.amax(-1).clamp_min(-1e30)
        p = torch.exp(scores - mx.unsqueeze(-1))
        denom = p.sum(-1) + torch.exp(sink - mx)
        # upstream feeds P to the PV GEMM in bf16
        o = torch.einsum("bcht,bctd->bchd", p.to(torch.bfloat16).float(), kv_g) / denom.unsqueeze(-1)
        out[:, s0 : s0 + chunk] = o.to(q.dtype)
    return out


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split hyper-connection mixes into pre [.., hc], post [.., hc] and a Sinkhorn-normalized comb [.., hc, hc]."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
