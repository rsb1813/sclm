# Runs the tiered host on a small random model on the CPU, prefill then decode, against upstream's Transformer with
# the same experts (coded ones computed with the engine's int8 tables, one token at a time).
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sclm import core, gpu, host
from sclm.checkpoint import Checkpoint, widen_fp8
from sclm.layerwise import load_params_
from sclm.upstream import model as m

import kernel
import test_core as tc
from test_model_small import init_random_
from test_tiered import SHAPES, write_checkpoint

pytestmark = pytest.mark.skipif(not os.path.exists(core.DLL), reason="run engine\\build.bat first")

PLANS = [["v8c8", "orig", "v2c6", "orig", "v8c8", "orig", "orig", "v2c6", "orig", "orig"],
         ["orig", "orig", "v2c6", "v2c6", "orig", "v8c8", "orig", "orig", "v8c8", "orig"]]


class Tabled(nn.Module):
    """A coded expert as the engine computes it, in upstream MoE's place for an expert."""

    def __init__(self, coded):
        super().__init__()
        self.ns = SimpleNamespace(**{w: (lambda v, mod=getattr(coded, w): tc.table_linear(mod, v)) for w in SHAPES},
                                  swiglu_limit=tc.LIMIT)

    def forward(self, x, weights):  # the emulation lives on the CPU
        out = [m.Expert.forward(self.ns, x[j:j + 1].cpu(), weights[j:j + 1].cpu()) for j in range(x.size(0))]
        return torch.cat(out).to(x.device)


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_host_decodes_as_upstream_with_the_same_experts(tmp_path, monkeypatch, dev):
    if dev == "cpu":
        # the kernel's GEMM (bf16 operands, fp32 result) exists on CUDA only; on the CPU both models take the same
        # products in fp32
        monkeypatch.setattr(kernel, "_mm_fp32_out", lambda a, w: a.float() @ w.float().t())
    for cached in (m.precompute_freqs_cis, m.get_window_topk_idxs):  # they hand back tensors of the device used first
        cached.cache_clear()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        args = m.ModelArgs(dim=tc.DIM, moe_inter_dim=tc.INTER, n_routed_experts=len(PLANS[0]), n_activated_experts=4,
                           swiglu_limit=tc.LIMIT, vocab_size=1000, max_batch_size=1, max_seq_len=64, temperature=0)
        with torch.device(dev):
            ref = m.Transformer(args)
        init_random_(ref, dev=dev)
        g = torch.Generator(device=dev).manual_seed(4)
        wo_a = []  # per layer, fp8 with a scale per 32 x 32 block, as the released checkpoint stores it
        with torch.no_grad():
            ref.head.weight.copy_(ref.head.weight.to(torch.bfloat16))  # the released head is bf16
            for layer in ref.layers:
                out, k = layer.attn.wo_a.weight.shape
                wo_a.append((torch.randn(out, k, generator=g, dtype=torch.float32, device=dev).to(torch.float8_e4m3fn),
                             torch.full((out // 32, k // 32), 2.0**-5, dtype=torch.float32, device=dev).to(torch.float8_e8m0fnu)))
                layer.attn.wo_a.weight.copy_(widen_fp8(*wo_a[-1]))  # upstream holds it widened
        placed = (".ffn.experts.", ".attn.wo_a.", "embed.", "head.")
        state = {name: p for name, p in ref.named_parameters() if not any(s in name for s in placed)}
        write_checkpoint(tmp_path, {name: p.detach().view(torch.uint8).cpu().numpy() for name, p in ref.named_parameters()
                                    if ".ffn.experts." in name})
        for i, layer in enumerate(ref.layers):
            coded = tc.make_layer(str(tmp_path / f"layer_{i:02d}.safetensors"), PLANS[i % 2])
            for e, c in enumerate(coded):
                if c is not None:
                    layer.ffn.experts[e] = Tabled(c)

        net = host.build_net(args, device=dev)
        for layer, (weight, scale) in zip(net.layers, wo_a):
            layer.attn.wo_a = host.Fp8Weight(weight, scale)
        load_params_(net, state, "")
        net.embed, net.head = host.HostEmbedding(ref.embed.weight.detach()), host.Bf16Head(ref.head.weight.detach().to(torch.bfloat16))
        cache = host.attach_tiers_(net, args, Checkpoint(str(tmp_path)), str(tmp_path), slots=5, lam=0.1, device=dev)
        reads, submit = [], cache.reader.submit
        cache.reader.submit = lambda layer, e, buf: reads.append((layer, e)) or submit(layer, e, buf)
        decoder = host.Host(net, cache, device=dev)
        assert decoder.kernels == (dev == "cuda" and os.path.exists(gpu.DLL))

        x = torch.randint(0, args.vocab_size, (1, 28), generator=torch.Generator().manual_seed(3))
        # on the CPU the two are equal; the GPU kernels add in another order than the reference GEMMs, so now and
        # then a bf16 value lands one step away (measured: 1e-14, one step 1e-7). Anything larger is a real fault:
        # slot uploads racing the next read gave 4e-3
        limit = 1e-6 if dev == "cpu" else 1e-5
        with torch.device(dev):
            for start, end in [(0, 20)] + [(i, i + 1) for i in range(20, 28)]:  # prefill, then a token at a time
                want = ref(x[:, start:end].to(dev), start)[1]
                got = decoder.logits(x[:, start:end], start)
                assert got.dtype == torch.float32 and tc.rel(got, want) < limit, (start, tc.rel(got, want))
        assert len(set(reads)) < len(reads)  # experts came back after being evicted
        assert len({layer for layer, _ in reads}) == args.n_layers
    finally:
        torch.set_default_dtype(prev)
