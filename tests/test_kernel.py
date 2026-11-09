# Checks the pure-PyTorch kernels against independent formulations (brute force, dense attention, convert.py's FP4 table).
import math

import pytest
import torch

import kernel

DEV = "cuda"
# convert.py's decode table: code -> value, sign in bit 3
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def test_ceil_log2_and_pow2():
    vals = torch.tensor([1e-30, 2**-126, 0.3, 0.5, 1.0, 1.0000001, 3.0, 4.0, 448.0, 1e30], device=DEV)
    got = kernel.ceil_log2(vals).tolist()
    want = [math.ceil(math.log2(v)) for v in vals.tolist()]
    assert got == want
    e = torch.arange(-126, 128, device=DEV)
    assert torch.equal(kernel.pow2(e), torch.pow(2.0, e.double()).float())


def _brute_fp4(v: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 value by exhaustive search, ties to the even code."""
    grid = torch.tensor(kernel.FP4_GRID, device=v.device)
    d = (v.abs().unsqueeze(-1) - grid).abs()
    best = d.min(-1, keepdim=True).values
    tied = d == best
    # prefer an even code among tied candidates, else the only candidate
    even = tied & (torch.arange(8, device=v.device) % 2 == 0)
    pick = torch.where(even.any(-1, keepdim=True), even, tied).float().argmax(-1)
    return torch.copysign(grid[pick], v)


def test_round_fp4_grid_ties_and_random():
    grid = torch.tensor(kernel.FP4_GRID, device=DEV)
    v, _ = kernel.round_fp4(torch.cat([grid, -grid]))
    assert torch.equal(v, torch.cat([grid, -grid]))
    mids = (grid[1:] + grid[:-1]) / 2
    v, _ = kernel.round_fp4(mids)
    assert torch.equal(v, _brute_fp4(mids))
    assert v.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]
    r = (torch.rand(100_000, device=DEV) * 12 - 6)
    v, code = kernel.round_fp4(r)
    assert torch.equal(v, _brute_fp4(r))
    assert torch.equal(FP4_TABLE.to(DEV)[code.long()], torch.where(v == 0, 0.0, v))


@pytest.mark.parametrize("scale_fmt", [None, "ue8m0"])
def test_act_quant_inplace_matches_split(scale_fmt):
    x = torch.randn(64, 512, device=DEV, dtype=torch.bfloat16) * 3
    sd = torch.float8_e8m0fnu if scale_fmt else torch.float32
    q, s = kernel.act_quant(x.clone(), 32, scale_fmt, sd)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (64, 16)
    if scale_fmt:
        sf = s.float()
        assert torch.equal(sf, torch.pow(2.0, torch.log2(sf).round()))
    deq = (q.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
    y = kernel.act_quant(x.clone(), 32, scale_fmt, sd, inplace=True)
    assert torch.equal(y, deq)
    # e4m3 has 3 mantissa bits: relative error <= 2^-4 for values away from the subnormal range
    big = x.float().abs() > x.float().abs().amax() / 16
    assert ((y.float() - x.float()).abs()[big] <= x.float().abs()[big] * 2**-4 + 1e-6).all()


@pytest.mark.parametrize("scale_dtype", [torch.float8_e8m0fnu, torch.float8_e4m3fn])
def test_fp4_act_quant_pack_roundtrip(scale_dtype):
    x = torch.randn(16, 512, device=DEV, dtype=torch.bfloat16)
    bs = 32 if scale_dtype == torch.float8_e8m0fnu else 16
    packed, s = kernel.fp4_act_quant(x.clone(), bs, False, scale_dtype)
    assert packed.dtype == torch.float4_e2m1fn_x2 and packed.shape == (16, 256)
    b = packed.view(torch.uint8)
    codes = torch.stack([b & 0x0F, b >> 4], dim=-1).flatten(-2)
    deq = (FP4_TABLE.to(DEV)[codes.long()].unflatten(-1, (-1, bs)) * s.float().unsqueeze(-1)).flatten(-2)
    y = kernel.fp4_act_quant(x.clone(), bs, True, scale_dtype)
    assert torch.equal(y, deq.to(torch.bfloat16))


def test_fp4_gemm_matches_dense():
    torch.manual_seed(0)
    N, K, M = 96, 256, 8
    w_codes = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=DEV)
    w_s = torch.pow(2.0, torch.randint(-9, -5, (N, K // 32), device=DEV).float()).to(torch.float8_e8m0fnu)
    a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    a_q, a_s = kernel.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
    got = kernel.fp4_gemm(a_q, a_s, w_codes.view(torch.float4_e2m1fn_x2), w_s, torch.float8_e8m0fnu, 32)
    codes = torch.stack([w_codes & 0x0F, w_codes >> 4], dim=-1).flatten(-2)
    w = FP4_TABLE.to(DEV)[codes.long()] * w_s.float().repeat_interleave(32, 1)
    a_deq = kernel.act_quant(a.clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True).double()
    want = a_deq @ w.double().t()
    assert torch.allclose(got.double(), want, rtol=2e-2, atol=1e-3)


def test_bf16_operand_path_matches_fp32_path():
    """Power-of-2 scales take the bf16 tensor-core path; the same scales as fp32 take the fp32 path. They must agree
    to fp32 accumulation error, since every dequantized operand is exact in bf16."""
    torch.manual_seed(4)
    N, K, M = 256, 512, 16
    w_codes = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=DEV).view(torch.float4_e2m1fn_x2)
    w_s = torch.pow(2.0, torch.randint(-9, -5, (N, K // 32), device=DEV).float()).to(torch.float8_e8m0fnu)
    a_q, a_s = kernel.act_quant(torch.randn(M, K, device=DEV, dtype=torch.bfloat16), 32, "ue8m0", torch.float8_e8m0fnu)
    fast = kernel.fp4_gemm(a_q, a_s, w_codes, w_s, torch.float8_e8m0fnu, 32)
    slow = kernel.fp4_gemm(a_q, a_s.float(), w_codes, w_s.float(), torch.float32, 32)
    assert torch.allclose(fast, slow, rtol=1e-5, atol=1e-6)
    w8 = torch.randn(N, K, device=DEV).to(torch.float8_e4m3fn)
    w8_s = torch.pow(2.0, torch.randint(-7, -3, (N // 32, K // 32), device=DEV).float()).to(torch.float8_e8m0fnu)
    fast = kernel.fp8_gemm(a_q, a_s, w8, w8_s, torch.float8_e8m0fnu, 32)
    slow = kernel.fp8_gemm(a_q, a_s.float(), w8, w8_s.float(), torch.float32, 32)
    assert torch.allclose(fast, slow, rtol=1e-5, atol=1e-6)


def test_fp8_gemm_matches_dense():
    torch.manual_seed(1)
    N, K, M = 96, 256, 8
    w = torch.randn(N, K, device=DEV).clamp(-448, 448).to(torch.float8_e4m3fn)
    w_s = torch.pow(2.0, torch.randint(-7, -3, (N // 32, K // 32), device=DEV).float()).to(torch.float8_e8m0fnu)
    a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    a_q, a_s = kernel.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
    got = kernel.fp8_gemm(a_q, a_s, w, w_s, torch.float8_e8m0fnu, 32)
    w_deq = w.double() * w_s.double().repeat_interleave(32, 0).repeat_interleave(32, 1)
    a_deq = kernel.act_quant(a.clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True).double()
    assert torch.allclose(got.double(), a_deq @ w_deq.t(), rtol=2e-2, atol=1e-3)


def test_sparse_attn_matches_dense_with_sink():
    torch.manual_seed(2)
    b, m, h, d, n, t = 2, 5, 4, 64, 20, 7
    q = torch.randn(b, m, h, d, device=DEV, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, device=DEV, dtype=torch.bfloat16)
    sink = torch.randn(h, device=DEV)
    idx = torch.stack([torch.stack([torch.randperm(n, device=DEV)[:t] for _ in range(m)]) for _ in range(b)]).int()
    idx[:, :, -2:] = -1  # some empty slots
    idx[1, 3] = -1  # a query with nothing to attend to
    scale = d**-0.5
    got = kernel.sparse_attn(q, kv, sink, idx, scale).float()

    scores = torch.einsum("bmhd,bnd->bmhn", q.float(), kv.float()) * scale
    mask = torch.zeros(b, m, n, dtype=torch.bool, device=DEV)
    for bi in range(b):
        for mi in range(m):
            sel = idx[bi, mi][idx[bi, mi] >= 0].long()
            mask[bi, mi, sel] = True
    scores = scores.masked_fill(~mask.unsqueeze(2), -torch.inf)
    logits = torch.cat([scores, sink.view(1, 1, h, 1).expand(b, m, h, 1)], dim=-1)
    p = logits.softmax(-1)[..., :n].nan_to_num()
    want = torch.einsum("bmhn,bnd->bmhd", p, kv.float())
    want[1, 3] = 0
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2)
    assert torch.equal(got[1, 3], torch.zeros_like(got[1, 3]))


def test_hc_split_sinkhorn_properties():
    torch.manual_seed(3)
    hc = 4
    mixes = torch.randn(2, 6, (2 + hc) * hc, device=DEV)
    pre, post, comb = kernel.hc_split_sinkhorn(mixes, torch.ones(3, device=DEV), torch.zeros(24, device=DEV), hc, 20, 1e-6)
    assert pre.shape == (2, 6, hc) and post.shape == (2, 6, hc) and comb.shape == (2, 6, hc, hc)
    assert ((pre > 0) & (pre < 1 + 1e-5)).all() and ((post > 0) & (post < 2)).all()
    assert torch.allclose(comb.sum(-2), torch.ones(2, 6, hc, device=DEV), atol=1e-4)
    assert torch.allclose(comb.sum(-1), torch.ones(2, 6, hc, device=DEV), atol=1e-3)
