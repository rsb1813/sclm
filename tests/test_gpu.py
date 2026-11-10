# Checks the engine's GPU kernels against the reference's kernels they replace (which widen each weight to bf16).
import os
from types import SimpleNamespace

import pytest
import torch

from sclm import gpu
from sclm.checkpoint import widen_fp8
from sclm.upstream import model as m

import kernel

pytestmark = pytest.mark.skipif(not os.path.exists(gpu.DLL), reason="run engine\\build_gpu.bat first")
DEV = "cuda"


@pytest.fixture(autouse=True)
def bf16_default():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(prev)


def close(a: torch.Tensor, b: torch.Tensor) -> None:
    """Equal up to the order of fp32 additions: almost every bf16 value identical, the rest one step apart."""
    a, b = a.float().flatten(), b.float().flatten()
    assert (a == b).float().mean() > 0.98, (a == b).float().mean()
    assert ((a - b).abs() <= b.abs() * 2**-7 + 1e-30).all()


def rand_fp8(g, out, k):
    weight = torch.randn(out, k, generator=g, device=DEV, dtype=torch.float32).to(torch.float8_e4m3fn)
    exp = torch.randint(-8, -3, ((out + 31) // 32, k // 32), generator=g, device=DEV)
    return weight, torch.pow(2.0, exp.float()).to(torch.float8_e8m0fnu)


def test_act_quant_equals_the_reference():
    g = torch.Generator(device=DEV).manual_seed(0)
    x = torch.randn(4096, generator=g, device=DEV, dtype=torch.float32)
    x *= torch.pow(10.0, torch.randint(-6, 4, (128, 1), generator=g, device=DEV).float()).expand(128, 32).reshape(-1)
    x[:32] = 0  # an all-zero block
    x[32:64] *= 1e-7  # below the scale floor
    x = x.to(torch.bfloat16)
    want = kernel.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True).float()
    assert torch.equal(gpu.act_quant(x), want)


@pytest.mark.parametrize("out,k", [(256, 1280), (1280, 5120), (5120, 8192), (96, 64)])
def test_fp8_linear_matches_upstream_linear(out, k):
    g = torch.Generator(device=DEV).manual_seed(out)
    weight, scale = rand_fp8(g, out, k)
    weight.scale = scale
    x = (3 * torch.randn(1, 1, k, generator=g, device=DEV, dtype=torch.float32)).to(torch.bfloat16)
    close(gpu.fp8_linear(x, weight, scale), m.linear(x, weight))


def test_grouped_fp8_linear_matches_the_block_diagonal_product():
    g = torch.Generator(device=DEV).manual_seed(1)
    groups, rank, k = 8, 128, 512
    weight, scale = rand_fp8(g, groups * rank, k)
    o = (3 * torch.randn(1, 1, groups, k, generator=g, device=DEV, dtype=torch.float32)).to(torch.bfloat16)
    got = gpu.fp8_linear(o, weight, scale, groups)
    # upstream Attention.forward: quantize o, then einsum with the widened weight
    oq = kernel.act_quant(o.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
    want = torch.einsum("bsgd,grd->bsgr", oq, widen_fp8(weight, scale).view(groups, rank, -1))
    close(got, want)


def test_fp4_expert_matches_upstream_expert():
    g = torch.Generator(device=DEV).manual_seed(2)
    dim, inter, limit = 512, 256, 10.0
    shapes = {"w1": (inter, dim), "w2": (dim, inter), "w3": (inter, dim)}
    scales = [torch.randint(119, 123, (out * k // 32,), generator=g, device=DEV, dtype=torch.uint8) for out, k in shapes.values()]
    weights = [torch.randint(0, 256, (out * k // 2,), generator=g, device=DEV, dtype=torch.uint8) for out, k in shapes.values()]
    slot = torch.cat(scales + weights)
    mats = {}
    for (name, (out, k)), s, w in zip(shapes.items(), scales, weights):
        weight = w.view(torch.float4_e2m1fn_x2).view(out, k // 2)
        weight.scale = s.view(torch.float8_e8m0fnu).view(out, k // 32)
        mats[name] = lambda v, weight=weight: m.linear(v, weight)
    expert = SimpleNamespace(**mats, swiglu_limit=limit)
    x = (3 * torch.randn(1, dim, generator=g, device=DEV, dtype=torch.float32)).to(torch.bfloat16)
    y = torch.zeros(dim, dtype=torch.float32, device=DEV)
    xq = gpu.act_quant(x)
    for gate in (0.3, 0.11):  # two calls add up
        gpu.fp4_expert(slot, dim, inter, limit, xq, gate, y)
    want = sum(m.Expert.forward(expert, x, torch.tensor([[gate]], dtype=torch.float32, device=DEV)).float() for gate in (0.3, 0.11))
    a, b = y, want.flatten()
    assert ((a - b).abs() <= b.abs() * 2**-6 + 1e-30).all() and (a == b).float().mean() > 0.9


def test_hyper_connection_steps_match_upstream_block():
    g = torch.Generator(device=DEV).manual_seed(3)
    hc, dim, eps = 4, 1024, 1e-6
    rows = (2 + hc) * hc
    block = SimpleNamespace(norm_eps=1e-20, hc_mult=hc, hc_sinkhorn_iters=20, hc_eps=eps)
    rand = lambda *shape: torch.randn(*shape, generator=g, device=DEV, dtype=torch.float32)  # noqa: E731
    h = (2 * rand(1, 1, hc, dim)).to(torch.bfloat16)
    fn, scale, base = 0.05 * rand(rows, hc * dim), rand(3), 0.5 * rand(rows)

    pre, post, comb = m.Block.hc_mixes(block, h, fn, scale, base)
    mixes = gpu.hc_mixes(h, fn, scale, base, hc, block.norm_eps, eps, 20)
    want = torch.cat([pre.flatten(), post.flatten(), comb.flatten()])
    assert torch.allclose(mixes, want, rtol=1e-4, atol=1e-7)
    assert torch.allclose(comb.sum(-1), torch.ones_like(comb.sum(-1)), atol=1e-3)  # the reference itself is what we think

    norm = m.RMSNorm(dim, 1e-20).to(DEV)
    with torch.no_grad():
        norm.weight.copy_((1 + 0.1 * rand(dim)).to(torch.bfloat16))
    close(gpu.hc_pre_norm(h, pre.flatten().contiguous(), norm.weight, norm.eps), norm(m.Block.hc_pre(None, h, pre)))

    x = rand(1, 1, dim).to(torch.bfloat16)
    close(gpu.hc_post(x, h, post.flatten().contiguous(), comb.flatten().contiguous()), m.Block.hc_post(None, x, h, post, comb))
