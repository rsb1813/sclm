# The tiered host for DeepSeek-V4-Flash-0731 with every routed expert original (no codes): upstream's code with the
# experts taken from the SSD cache has to give upstream's logits. Run with SCLM_MODEL=v4.
import pytest
import torch

from sclm import host, upstream
from sclm.checkpoint import Checkpoint
from sclm.layerwise import load_params_
from test_tiered import write_checkpoint
from test_v4_model_small import DEV, SMALL, build

pytestmark = pytest.mark.skipif(upstream.MODEL != "v4", reason="set SCLM_MODEL=v4")
m = upstream.model


def test_lossless_host_gives_upstream_logits(tmp_path):
    args = m.ModelArgs(**{**SMALL, "dim": 512})  # an expert's scales and weights then fill whole pages, as the reader expects
    for cached in (m.get_window_topk_idxs, m.get_compress_topk_idxs, m.precompute_freqs_cis):
        cached.cache_clear()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(DEV):
            ref = build(args)
        with torch.no_grad():
            ref.head.weight.copy_(ref.head.weight.to(torch.bfloat16))  # the released head is bf16
        placed = (".ffn.experts.", "embed.", "head.")
        state = {name: p for name, p in ref.named_parameters() if not any(s in name for s in placed)}
        write_checkpoint(tmp_path, {name: p.detach().view(torch.uint8).cpu().numpy() for name, p in ref.named_parameters()
                                    if ".ffn.experts." in name})
        net = host.build_net(args, device=DEV)
        load_params_(net, state, "")
        net.embed, net.head = host.HostEmbedding(ref.embed.weight.detach()), host.Bf16Head(ref.head.weight.detach().to(torch.bfloat16))
        cache = host.attach_tiers_(net, args, Checkpoint(str(tmp_path)), None, slots=5, lam=0.1, device=DEV)
        reads, submit = [], cache.reader.submit
        cache.reader.submit = lambda layer, e, buf: reads.append((layer, e)) or submit(layer, e, buf)
        decoder = host.Host(net, cache, device=DEV)
        assert not decoder.kernels  # the engine's kernels are not ported to this model yet
        x = torch.randint(0, args.vocab_size, (1, 60), generator=torch.Generator().manual_seed(3))
        steps = [(0, 30)] + [(i, i + 1) for i in range(30, 60)]  # a prefill, then a token at a time
        with torch.device(DEV):
            for start, end in steps:
                want = ref(x[:, start:end].to(DEV), start)[1].float()
                got = decoder.logits(x[:, start:end], start).float()
                # the blocks' outputs are equal to the bit; the head sums bf16 products in another order than upstream's
                assert float((got - want).abs().max()) < 1e-5, (start, float((got - want).abs().max()))
        assert len(set(reads)) < len(reads)  # 5 slots for 6 layers of 8 experts: experts came back after leaving
        assert len({layer for layer, _ in reads}) == args.n_layers
    finally:
        torch.set_default_dtype(prev)
