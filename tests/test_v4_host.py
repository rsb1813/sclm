# The tiered host for DeepSeek-V4-Flash-0731 with every routed expert original (no codes) against upstream's code with
# the same weights: upstream's path with the experts taken from the SSD cache, and a decode token on the engine's GPU
# kernels. Run with SCLM_MODEL=v4.
import pytest
import torch
import torch.nn.functional as F

from sclm import host, upstream
from sclm.checkpoint import Checkpoint, widen_fp8
from sclm.layerwise import load_params_
from test_tiered import write_checkpoint
from test_v4_gpu import rand_fp8
from test_v4_model_small import DEV, SMALL, build

pytestmark = pytest.mark.skipif(upstream.MODEL != "v4", reason="set SCLM_MODEL=v4")
m = upstream.model
# an expert's scales and weights fill whole pages (the reader's unit), and the attention's widths are multiples of the
# kernels' quantization blocks
CONFIG = {**SMALL, "dim": 512, "head_dim": 256, "n_heads": 4, "q_lora_rank": 128, "o_lora_rank": 32, "o_groups": 4}


@pytest.fixture
def models(tmp_path):
    """upstream's model, and the host's with the same weights: (args, upstream's, the host's, its expert cache)."""
    args = m.ModelArgs(**CONFIG)
    for cached in (m.get_window_topk_idxs, m.get_compress_topk_idxs, m.precompute_freqs_cis):
        cached.cache_clear()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(DEV):
            ref = build(args)
        net = host.build_net(args, device=DEV)
        g = torch.Generator(device=DEV).manual_seed(4)
        with torch.no_grad():
            ref.head.weight.copy_(ref.head.weight.to(torch.bfloat16))  # the released head is bf16
            placed = (".ffn.experts.", "embed.", "head.")
            load_params_(net, {name: p for name, p in ref.named_parameters() if not any(s in name for s in placed)}, "")
            for theirs, ours in zip(ref.layers, net.layers):  # wo_a: fp8 with block scales, as the checkpoint stores it
                weight, scale = rand_fp8(g, *theirs.attn.wo_a.weight.shape)
                theirs.attn.wo_a.weight.copy_(widen_fp8(weight, scale))
                ours.attn.wo_a = host.Fp8Weight(weight, scale)
        write_checkpoint(tmp_path, {name: p.detach().view(torch.uint8).cpu().numpy() for name, p in ref.named_parameters()
                                    if ".ffn.experts." in name})
        net.embed, net.head = host.HostEmbedding(ref.embed.weight.detach()), host.Bf16Head(ref.head.weight.detach().to(torch.bfloat16))
        cache = host.attach_tiers_(net, args, Checkpoint(str(tmp_path)), None, slots=5, lam=0.1, device=DEV)
        yield args, ref, net, cache
    finally:
        torch.set_default_dtype(prev)


def test_lossless_host_gives_upstream_logits(models):
    args, ref, net, cache = models
    reads, submit = [], cache.reader.submit
    cache.reader.submit = lambda layer, e, buf: reads.append((layer, e)) or submit(layer, e, buf)
    decoder = host.Host(net, cache, device=DEV, kernels=False)
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


@pytest.mark.parametrize("synced", [True, False])
def test_kernel_decode_follows_upstream(models, synced):
    """synced: before every decode token the host's attention state (window and compressed key/values, the
    compressors' groups, the index's keys) is set to upstream's, so each token is compared from the same state: the
    window ring wrapping, groups of 4 and 16 completing, the index choosing. Not synced: the two run on their own,
    and what differs is rounding that the random model magnifies over the tokens."""
    args, ref, net, cache = models
    decoder = host.Host(net, cache, device=DEV)
    assert decoder.kernels
    n, split = 90, 20  # past the window (16) and past what the index keeps (8 groups of 4)
    x = torch.randint(0, args.vocab_size, (1, n), generator=torch.Generator().manual_seed(3))
    outs = {"ref": [], "host": []}
    for name, model in (("ref", ref), ("host", net)):
        for layer in model.layers:
            layer.register_forward_hook(lambda _m, _i, out, name=name: outs[name].append(out.float().flatten().clone()))
    state = dict(ref.named_buffers())
    cos, blocks = [], []
    with torch.device(DEV):
        want = ref(x[:, :split].to(DEV), 0)[1].float()
        assert float((decoder.logits(x[:, :split], 0).float() - want).abs().max()) < 1e-5  # the prefill is upstream's
        for i in range(split, n):
            if synced:
                with torch.no_grad():
                    for name, buf in net.named_buffers():
                        buf.copy_(state[name])
            outs["ref"].clear(), outs["host"].clear()
            want = ref(x[:, i:i + 1].to(DEV), i)[1].float()
            got = decoder.logits(x[:, i:i + 1], i).float()
            cos.append(float(F.cosine_similarity(got.flatten(), want.flatten(), dim=0)))
            assert len(outs["host"]) == args.n_layers
            blocks.append(min(float(F.cosine_similarity(a, b, dim=0)) for a, b in zip(outs["ref"], outs["host"])))
    print("synced" if synced else "free", "worst block:", min(blocks), "at token", split + blocks.index(min(blocks)),
          "| logits: min", min(cos), "at token", split + cos.index(min(cos)), "median", sorted(cos)[len(cos) // 2])
    if synced:
        # Most tokens agree to cosine 1.0000 in every block. In about one of six a value lands on the other side of a
        # rounding step (the kernels add in another order than upstream) or the gate picks another of its 8 experts,
        # and the layers after it magnify that; these tokens come at no particular event and start at any layer.
        assert sum(b > 0.9999 for b in blocks) > 0.75 * len(blocks) and min(blocks) > 0.97
        assert sorted(cos)[len(cos) // 2] > 0.9999 and min(cos) > 0.97
    else:
        assert blocks[0] > 0.9999 and min(cos) > 0.9 and sorted(cos)[len(cos) // 2] > 0.99


@pytest.mark.parametrize("kernels", [False, True])
@pytest.mark.parametrize("prefix", [10, 38])
def test_restore_takes_the_sequence_back(models, kernels, prefix):
    """Tokens fed after a snapshot() leave no trace once restore() has taken the sequence back: what follows gives the
    logits of a sequence that never held them. The snapshot is made inside a group of 4 and of 16, before (10) and
    after (38) the index starts choosing; the dropped tokens wrap the window ring (16), complete groups of both
    ratios and leave compressed positions and index keys behind, past the snapshot's."""
    args, _, net, cache = models
    decoder = host.Host(net, cache, device=DEV, kernels=kernels)
    dropped, more = 45, 40
    x = torch.randint(0, args.vocab_size, (1, prefix + dropped + more), generator=torch.Generator().manual_seed(5))

    def feed(tokens, pos):  # a token at a time, as the chat feeds
        return [decoder.logits(tokens[:, i:i + 1], pos + i).float().clone() for i in range(tokens.size(1))]

    feed(x[:, :prefix], 0)
    kept = decoder.snapshot()
    feed(x[:, prefix:prefix + dropped], prefix)
    decoder.restore(kept, dropped)
    got = feed(x[:, prefix + dropped:], prefix)
    want = feed(torch.cat([x[:, :prefix], x[:, prefix + dropped:]], dim=1), 0)[prefix:]
    assert len(got) == len(want) == more and all(torch.equal(a, b) for a, b in zip(got, want))
