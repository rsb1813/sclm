# The engine's GPU kernels with DeepSeek-V4-Flash-0731's settings (activation blocks of 128, an FP8 weight scale per
# 128 x 128, its attention's own steps) against upstream's code. Run with SCLM_MODEL=v4.
import os
from types import SimpleNamespace

import pytest
import torch

from sclm import ACT_BLOCK, gpu, upstream
from sclm.checkpoint import widen_fp8
from test_gpu import close
from test_v4_model_small import SMALL, build

import kernel

pytestmark = pytest.mark.skipif(upstream.MODEL != "v4" or not os.path.exists(gpu.DLL), reason="set SCLM_MODEL=v4 (and build the GPU engine)")
m = upstream.model
DEV = "cuda"
# sizes the kernels' quantization blocks divide: the window KV rounds 64 values at a time, the others 128
ATTN = {**SMALL, "head_dim": 256, "rope_head_dim": 64, "q_lora_rank": 128, "o_lora_rank": 32, "o_groups": 4, "n_heads": 4}


@pytest.fixture(autouse=True)
def bf16_default():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    # what building upstream's Transformer for the released config sets
    m.default_dtype, m.scale_fmt, m.scale_dtype = torch.float8_e4m3fn, "ue8m0", torch.float8_e8m0fnu
    yield
    torch.set_default_dtype(prev)


def rand_fp8(g, out, k):
    weight = torch.randn(out, k, generator=g, device=DEV, dtype=torch.float32).to(torch.float8_e4m3fn)
    exp = torch.randint(-8, -3, ((out + 127) // 128, k // 128), generator=g, device=DEV)
    return weight, torch.pow(2.0, exp.float()).to(torch.float8_e8m0fnu)


def test_act_quant_equals_the_reference():
    assert ACT_BLOCK == 128
    g = torch.Generator(device=DEV).manual_seed(0)
    x = torch.randn(4096, generator=g, device=DEV, dtype=torch.float32)
    x *= torch.pow(10.0, torch.randint(-6, 4, (32, 1), generator=g, device=DEV).float()).expand(32, 128).reshape(-1)
    x[:128] = 0  # an all-zero block
    x[128:256] *= 1e-7  # below the scale floor
    x = x.to(torch.bfloat16)
    want = kernel.act_quant(x.clone(), 128, "ue8m0", torch.float8_e8m0fnu, inplace=True).float()
    assert torch.equal(gpu.act_quant(x), want)


@pytest.mark.parametrize("out,k", [(256, 1024), (1024, 4096), (4096, 8192), (32768, 1024), (96, 128)])
def test_fp8_linear_matches_upstream_linear(out, k):
    g = torch.Generator(device=DEV).manual_seed(out)
    weight, scale = rand_fp8(g, out, k)
    weight.scale = scale
    x = (3 * torch.randn(1, 1, k, generator=g, device=DEV, dtype=torch.float32)).to(torch.bfloat16)
    close(gpu.fp8_linear(x, weight, scale), m.linear(x, weight))


@pytest.mark.parametrize("dim,inter", [(512, 256), (4096, 2048)])
def test_fp4_expert_matches_upstream_expert(dim, inter):
    g = torch.Generator(device=DEV).manual_seed(2)
    limit = 10.0
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


def test_fp8_expert_matches_upstream_expert():
    g = torch.Generator(device=DEV).manual_seed(5)
    dim, inter, limit = 512, 256, 10.0
    expert = SimpleNamespace(swiglu_limit=limit)
    for name, (out, k) in {"w1": (inter, dim), "w2": (dim, inter), "w3": (inter, dim)}.items():
        weight, scale = rand_fp8(g, out, k)
        weight.scale = scale
        lin = lambda v, weight=weight: m.linear(v, weight)  # noqa: E731
        lin.weight = weight
        setattr(expert, name, lin)
    x = (3 * torch.randn(1, dim, generator=g, device=DEV, dtype=torch.float32)).to(torch.bfloat16)
    y = torch.zeros(dim, dtype=torch.float32, device=DEV)
    gpu.fp8_expert(expert, limit, gpu.act_quant(x), y)
    want = m.Expert.forward(expert, x).float().flatten()
    assert ((y - want).abs() <= want.abs() * 2**-6 + 1e-30).all() and (y == want).float().mean() > 0.9


@pytest.fixture
def attention():
    args = m.ModelArgs(**ATTN)
    for cached in (m.get_window_topk_idxs, m.get_compress_topk_idxs, m.precompute_freqs_cis):
        cached.cache_clear()
    with torch.device(DEV):
        net = build(args)
    return args, net.layers[0].attn


def test_attention_query_matches_upstream(attention):
    args, attn = attention
    g = torch.Generator(device=DEV).manual_seed(6)
    rd, heads, d, pos = attn.rope_head_dim, attn.n_local_heads, attn.head_dim, 37
    x = torch.randn(1, 1, args.dim, generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    freqs = attn.freqs_cis[pos:pos + 1]
    with torch.no_grad():
        qr = attn.q_norm(attn.wq_a(x))
        q = attn.wq_b(qr).unflatten(-1, (heads, d))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + attn.eps)
        m.apply_rotary_emb(q[..., -rd:], freqs)
    got_qr, got_q = gpu.attn_q(attn.wq_a, attn.q_norm.weight.to(torch.bfloat16), attn.wq_b, x.view(-1),
                               attn.freqs_cis.data_ptr() + pos * rd * 4, heads, d, rd, attn.eps)
    close(got_qr, qr)
    # the head norm is a chain of bf16 operations whose sum upstream adds in another order: a head's factor may land
    # one bf16 step away, and then so do its values
    a, b = got_q.float(), q.float().flatten()
    assert ((a - b).abs() <= b.abs() * 2**-6 + 1e-30).all() and (a == b).float().mean() > 0.5


def test_window_key_value_matches_upstream(attention):
    args, attn = attention
    g = torch.Generator(device=DEV).manual_seed(7)
    rd, pos = attn.rope_head_dim, 21
    x = torch.randn(1, 1, args.dim, generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    with torch.no_grad():
        kv = attn.kv_norm(attn.wkv(x))
        m.apply_rotary_emb(kv[..., -rd:], attn.freqs_cis[pos:pos + 1])
        rotated = kv[..., -rd:].clone()
        m.act_quant(kv[..., :-rd], 64, m.scale_fmt, m.scale_dtype, True)
    got = torch.empty(attn.head_dim, dtype=torch.bfloat16, device=DEV)
    gpu.attn_kv(attn.wkv, attn.kv_norm.weight.to(torch.bfloat16), x.view(-1), attn.freqs_cis.data_ptr() + pos * rd * 4, got, rd, attn.eps)
    close(got, kv)
    assert torch.equal(kv[..., -rd:], rotated)  # upstream leaves the rotated values unrounded, and so does the kernel
    assert not torch.equal(kernel.act_quant(got[-rd:].clone(), 64, "ue8m0", torch.float8_e8m0fnu, True), got[-rd:])


def test_attention_output_matches_upstream(attention):
    args, attn = attention
    g = torch.Generator(device=DEV).manual_seed(8)
    rd, heads, d, groups, pos = attn.rope_head_dim, attn.n_local_heads, attn.head_dim, attn.n_local_groups, 5
    wo_a, wo_a_scale = rand_fp8(g, groups * attn.o_lora_rank, heads * d // groups)
    o = torch.randn(1, 1, heads, d, generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)
    got = gpu.attn_out(o.flatten().clone(), attn.freqs_cis.data_ptr() + pos * rd * 4, wo_a, wo_a_scale, attn.wo_b, heads, rd, groups)
    with torch.no_grad():
        m.apply_rotary_emb(o[..., -rd:], attn.freqs_cis[pos:pos + 1], True)
        low = torch.einsum("bsgd,grd->bsgr", o.view(1, 1, groups, -1), widen_fp8(wo_a, wo_a_scale).view(groups, attn.o_lora_rank, -1))
        want = attn.wo_b(low.flatten(2))
    a, b = got.float(), want.float().flatten()
    assert ((a - b).abs() <= b.abs() * 2**-5 + 1e-30).all() and (a == b).float().mean() > 0.9


@pytest.fixture
def small_net():
    args = m.ModelArgs(**ATTN)
    for cached in (m.get_window_topk_idxs, m.get_compress_topk_idxs, m.precompute_freqs_cis):
        cached.cache_clear()
    with torch.device(DEV):
        return args, build(args), build(args)  # two models with the same weights


@pytest.mark.parametrize("layer", [0, 3])  # a layer that routes by the token id, and one that chooses
def test_gate_matches_upstream_gate(small_net, layer):
    args, net, _ = small_net
    gate = net.layers[layer].ffn.gate
    g = torch.Generator(device=DEV).manual_seed(9 + layer)
    if gate.bias is not None:
        with torch.no_grad():
            gate.bias.copy_(0.05 * torch.randn(gate.bias.shape, generator=g, device=DEV, dtype=torch.float32))
    topk = gate.topk
    for token in (3, 500, 999):
        x = torch.randn(1, args.dim, generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)
        weights, indices = gate(x, torch.tensor([token], device=DEV))
        pack = gpu.gate(gate, x.view(-1), fixed=gate.tid2eid[token] if gate.hash else None)
        assert pack[:topk].long().tolist() == indices.flatten().tolist()
        assert torch.allclose(pack[topk:2 * topk], weights.flatten().float(), rtol=1e-5)
        assert torch.equal(pack[2 * topk:], x.flatten().float())


def test_hyper_connection_steps_match_upstream_block():
    g = torch.Generator(device=DEV).manual_seed(3)
    hc, dim, eps = 4, 1024, 1e-6
    rows = (2 + hc) * hc
    block = SimpleNamespace(norm_eps=eps, hc_mult=hc, hc_sinkhorn_iters=20, hc_eps=eps)
    rand = lambda *shape: torch.randn(*shape, generator=g, device=DEV, dtype=torch.float32)  # noqa: E731
    h = (2 * rand(1, 1, hc, dim)).to(torch.bfloat16)
    fn, scale, base = 0.05 * rand(rows, hc * dim), rand(3), 0.5 * rand(rows)
    y, post, comb = m.Block.hc_pre(block, h, fn, scale, base)
    mixes = gpu.hc_mixes(h, fn, scale, base, hc, block.norm_eps, eps, 20)
    assert torch.allclose(mixes[hc:], torch.cat([post.flatten(), comb.flatten()]), rtol=1e-4, atol=1e-7)
    norm = m.RMSNorm(dim, eps).to(DEV)
    with torch.no_grad():
        norm.weight.copy_((1 + 0.1 * rand(dim)).to(torch.bfloat16))
    close(gpu.hc_pre_norm(h, mixes[:hc].contiguous(), norm.weight.to(torch.bfloat16), norm.eps), norm(y))
    x = rand(1, 1, dim).to(torch.bfloat16)
    close(gpu.hc_post(x, h, post.flatten().contiguous(), comb.flatten().contiguous()), m.Block.hc_post(None, x, h, post, comb))


@pytest.mark.parametrize("which", ["ratio 4", "ratio 16", "index"])
def test_pool_matches_upstream_compressor(small_net, which):
    args, ours_net, theirs_net = small_net
    layer = 3 if which == "ratio 16" else 2
    pick = (lambda net: net.layers[layer].attn.indexer.compressor) if which == "index" else (lambda net: net.layers[layer].attn.compressor)
    ours, theirs = pick(ours_net), pick(theirs_net)
    assert ours.overlap == (which != "ratio 16") and ours.rotate == (which == "index")
    g = torch.Generator(device=DEV).manual_seed(11)
    ratio, d = ours.compress_ratio, ours.head_dim
    freqs = ours_net.layers[layer].attn.freqs_cis
    with torch.no_grad():
        for comp in (ours, theirs):
            comp.ape.copy_(0.5 * torch.randn(comp.ape.shape, generator=torch.Generator(device=DEV).manual_seed(12), device=DEV, dtype=torch.float32))
            comp.kv_cache, comp.freqs_cis = torch.zeros(1, 16, d, dtype=torch.bfloat16, device=DEV), freqs
    norm_w = ours.norm.weight.to(torch.bfloat16)
    groups = 0
    for pos in range(1, 3 * ratio + 2):
        x = torch.randn(1, 1, args.dim, generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)
        with torch.no_grad():
            theirs(x, pos)
        gpu.pool(ours, x.view(-1), pos, freqs, norm_w, ours.kv_cache[0, pos // ratio])
        assert torch.allclose(ours.kv_state, theirs.kv_state, rtol=1e-4, atol=1e-5)
        assert torch.allclose(ours.score_state, theirs.score_state, rtol=1e-4, atol=1e-5)
        if (pos + 1) % ratio == 0:
            groups += 1
            a, b = ours.kv_cache[0, pos // ratio].float(), theirs.kv_cache[0, pos // ratio].float()
            assert b.abs().max() > 0
            # FP4 keys have few levels: a value at a rounding boundary lands on the neighbouring level
            assert (a == b).float().mean() > (0.85 if which == "index" else 0.97), (pos, (a == b).float().mean())
            assert ((a - b).abs() <= (0.5 if which == "index" else 2**-6) * b.abs().max()).all()
    assert groups == 3
