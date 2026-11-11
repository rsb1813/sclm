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
        gpu.fp4_experts(slot.view(1, -1), [0], [gate], dim, inter, limit, xq, y)
    batch = torch.zeros(dim, dtype=torch.float32, device=DEV)
    gpu.fp4_experts(slot.view(1, -1), [0, 0], [0.3, 0.11], dim, inter, limit, xq, batch)
    assert torch.equal(batch, y)  # a batch adds its experts in order, as separate calls do
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


def test_sparse_attn_matches_upstream():
    g = torch.Generator(device=DEV).manual_seed(4)
    heads, d, win, more = 8, 64, 16, 40
    rand = lambda *shape: torch.randn(*shape, generator=g, device=DEV, dtype=torch.float32)  # noqa: E731
    q, window, compressed = (rand(*shape).to(torch.bfloat16) for shape in ((heads, d), (win, d), (more, d)))
    sink = rand(heads)
    idx = torch.cat([torch.arange(win, device=DEV), win + torch.randperm(more, generator=g, device=DEV)[:24]]).int()
    idx[3], idx[20] = -1, -1  # empty slots
    for use in (idx, torch.full_like(idx, -1)):  # the second: nothing to attend to, the output is zero
        want = kernel.sparse_attn(q.view(1, 1, heads, d), torch.cat([window, compressed])[None], sink, use.view(1, 1, -1), d**-0.5)
        close(gpu.sparse_attn(q.flatten(), window, compressed, use, sink, heads, win, d**-0.5), want)


@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_compress_matches_upstream_compressor_and_index_key(ratio):
    g = torch.Generator(device=DEV).manual_seed(7 + ratio)
    dim, d, kd, rd, groups = 256, 64, 32, 16, 6
    rand = lambda *shape: torch.randn(*shape, generator=g, device=DEV, dtype=torch.float32)  # noqa: E731

    def linear(out, k, dtype):
        lin = torch.nn.Linear(k, out, bias=False, device=DEV, dtype=dtype)
        with torch.no_grad():
            lin.weight.copy_((rand(out, k) / k**0.5).to(dtype))
        return lin

    def norm(n):
        mod = m.RMSNorm(n, 1e-6).to(DEV)
        with torch.no_grad():
            mod.weight.copy_((1 + 0.1 * rand(n)).to(torch.bfloat16))
        return mod

    def state():
        return torch.zeros(1, ratio, d, dtype=torch.float32, device=DEV), torch.full((1, ratio, d), -torch.inf, dtype=torch.float32, device=DEV)

    wdt = torch.float32 if ratio > 1 else torch.bfloat16  # as upstream Compressor builds them
    parts = dict(compress_ratio=ratio, head_dim=d, norm=norm(d), wkv=linear(d, dim, wdt), wgate=linear(d, dim, wdt))
    ours, theirs = (SimpleNamespace(**parts, kv_state=kv, score_state=score) for kv, score in (state(), state()))
    wk, k_norm = linear(kd, d, torch.bfloat16), norm(kd)
    indexer = SimpleNamespace(owns_k=True, wk=wk, k_norm=k_norm, index_head_dim=kd,
                              k_cache=torch.zeros(1, groups, kd, dtype=torch.bfloat16, device=DEV))
    freqs_cis = torch.polar(torch.ones(groups * ratio + 1, rd // 2, device=DEV, dtype=torch.float32),
                            6.28 * torch.rand(groups * ratio + 1, rd // 2, generator=g, device=DEV, dtype=torch.float32))
    kv_cache = torch.zeros(groups, d, dtype=torch.bfloat16, device=DEV)
    same = total = 0
    for pos in range(1, groups * ratio):  # position 0 is the prefill's
        x = (2 * rand(dim)).to(torch.bfloat16)
        gpu.compress(ours, indexer, x, pos, freqs_cis, kv_cache, rd)
        latent = m.Compressor.forward(theirs, x.view(1, 1, -1), pos)
        if ratio > 1:
            assert torch.allclose(ours.kv_state, theirs.kv_state, rtol=1e-4, atol=1e-5)
            assert torch.allclose(ours.score_state, theirs.score_state, rtol=1e-4, atol=1e-5)
        if latent is None:
            assert (pos + 1) % ratio
            continue
        freqs = freqs_cis[pos + 1 - ratio].unsqueeze(0)
        key = k_norm(wk(latent))  # Indexer.forward
        m.apply_rotary_emb(key[..., -rd:], freqs)
        kernel.fp4_act_quant(key, 32, True)
        m.apply_rotary_emb(latent[..., -rd:], freqs)  # Attention._compress_kv
        kernel.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
        for got, want in ((kv_cache[pos // ratio], latent), (indexer.k_cache[0, pos // ratio], key)):
            same, total = same + int((got == want.flatten()).sum()), total + got.numel()
            # a last bit that differs before the rounding can move a value to the next FP4 step, not further
            assert ((got.float() - want.flatten().float()).abs() <= want.flatten().float().abs() * 0.5 + 1e-3).all()
    assert total >= (groups - 1) * (d + kd) and same / total > 0.97, same / total


def test_fp8_expert_matches_upstream_expert():
    g = torch.Generator(device=DEV).manual_seed(5)
    dim, inter, limit = 512, 256, 10.0
    m.default_dtype = torch.float8_e4m3fn
    expert = m.Expert(dim, inter, swiglu_limit=limit).to(DEV)
    with torch.no_grad():
        for lin in (expert.w1, expert.w2, expert.w3):
            weight, scale = rand_fp8(g, *lin.weight.shape)
            lin.weight.copy_(weight)
            lin.scale.copy_(scale)
    x = (3 * torch.randn(1, dim, generator=g, device=DEV, dtype=torch.float32)).to(torch.bfloat16)
    y = torch.zeros(dim, dtype=torch.float32, device=DEV)
    gpu.fp8_expert(expert, limit, gpu.act_quant(x), y)
    close(y, expert(x))


def test_moe_begin_and_end_equal_the_separate_kernels():
    g = torch.Generator(device=DEV).manual_seed(8)
    dim, inter, limit, hc, topk = 512, 256, 10.0, 4, 6
    rand = lambda *shape: torch.randn(*shape, generator=g, device=DEV, dtype=torch.float32)  # noqa: E731
    args = m.ModelArgs(dim=dim, n_routed_experts=48, n_activated_experts=topk, route_scale=2.5)
    gates = [m.Gate(i, args).to(DEV) for i in range(2)]  # the layer's and the next layer's
    m.default_dtype = torch.float8_e4m3fn
    expert = m.Expert(dim, inter, swiglu_limit=limit).to(DEV)
    with torch.no_grad():
        for gate in gates:
            gate.weight.copy_((0.1 * rand(*gate.weight.shape)).to(torch.bfloat16))
            gate.bias.copy_(0.05 * rand(*gate.bias.shape))
        for lin in (expert.w1, expert.w2, expert.w3):
            weight, scale = rand_fp8(g, *lin.weight.shape)
            lin.weight.copy_(weight)
            lin.scale.copy_(scale)
    norm_weight = (1 + 0.1 * rand(dim)).to(torch.bfloat16)
    ram = torch.empty(dim, dtype=torch.float32, device="cpu", pin_memory=True)  # the CPU tier's sum
    for step in range(6):
        guess = 4 * (step % 2)
        x, h, pre = (3 * rand(dim)).to(torch.bfloat16), (2 * rand(hc * dim)).to(torch.bfloat16), rand(hc)
        routed = rand(dim)  # stands for the sum of the routed experts on the device
        ram.copy_(rand(dim))
        # the calls a decode token made one by one
        pack = gpu.gate(gates[0], x, 2 * guess)
        if guess:
            gpu.gate_guess(gates[1], norm_weight, h, pre, 1e-6, pack[2 * topk + dim:])
        xq = gpu.act_quant(x)
        shared = torch.empty(dim, dtype=torch.float32, device=DEV)
        gpu.fp8_expert(expert, limit, xq, shared, store=True)
        want = ((routed + ram.to(DEV)) + shared).to(torch.bfloat16)

        buf = gpu.moe_buffers(topk, dim, guess, torch.device(DEV))
        buf.host.fill_(torch.nan)
        if guess:
            gpu.moe_begin(gates[0], x, expert, limit, buf, guess, gates[1], norm_weight, h, pre, 1e-6)
        else:
            gpu.moe_begin(gates[0], x, expert, limit, buf)
        head = buf.host[:2 * topk + dim].clone()  # on the host when moe_begin returns
        if guess:
            gpu.moe_guess()
        assert torch.equal(head, pack[:2 * topk + dim].cpu()) and torch.equal(buf.host, pack.cpu())
        assert torch.equal(buf.xq, xq) and torch.equal(buf.shared, shared) and not buf.routed.any()
        buf.routed.copy_(routed)
        assert torch.equal(gpu.moe_end(buf, ram), want)


def test_gate_matches_upstream_gate():
    g = torch.Generator(device=DEV).manual_seed(6)
    args = m.ModelArgs(dim=512, n_routed_experts=48, n_activated_experts=6, route_scale=2.5)
    gate = m.Gate(0, args).to(DEV)
    with torch.no_grad():
        gate.weight.copy_((0.1 * torch.randn(gate.weight.shape, generator=g, device=DEV, dtype=torch.float32)).to(torch.bfloat16))
        gate.bias.copy_(0.05 * torch.randn(gate.bias.shape, generator=g, device=DEV, dtype=torch.float32))
    for _ in range(4):
        x = (3 * torch.randn(1, 512, generator=g, device=DEV, dtype=torch.float32)).to(torch.bfloat16)
        weights, indices = gate(x)
        pack = gpu.gate(gate, x.flatten())
        assert pack[:6].int().tolist() == indices.flatten().tolist()
        assert torch.allclose(pack[6:12], weights.flatten(), rtol=1e-5)
        assert torch.equal(pack[12:], x.flatten().float())
