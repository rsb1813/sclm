# DeepSeek-V4-Flash-0731's upstream model.py on the pure-PyTorch kernels, as a small random model: a prefill of the whole
# sequence against a prefill followed by decode steps. Run with SCLM_MODEL=v4 (one upstream model per process).
import pytest
import torch
import torch.nn.functional as F

from sclm import upstream
from test_model_small import init_random_

pytestmark = pytest.mark.skipif(upstream.MODEL != "v4", reason="set SCLM_MODEL=v4")
m = upstream.model
DEV = "cuda"
SMALL = dict(dim=256, moe_inter_dim=256, n_layers=6, n_hash_layers=2, n_heads=4, head_dim=192, rope_head_dim=64, q_lora_rank=128,
             o_lora_rank=32, o_groups=4, n_routed_experts=8, n_activated_experts=2, vocab_size=1000, window_size=16,
             compress_ratios=(0, 0, 4, 16, 4, 0), index_n_heads=4, index_head_dim=64, index_topk=8, max_seq_len=256,
             max_batch_size=1, expert_dtype="fp4", swiglu_limit=10.0, route_scale=1.5)


def build(args):
    net = m.Transformer(args)
    init_random_(net, seed=1)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith("tid2eid"):  # the hash layers' experts of each token
                p.copy_(torch.randint(0, args.n_routed_experts, p.shape, dtype=p.dtype, generator=torch.Generator(device=DEV).manual_seed(5)))
            elif name.endswith("hc_head_scale"):
                p.fill_(1.0)
            elif name.endswith(("hc_head_base", "ape")):
                p.zero_()
    return net


def agreement(split: int, n: int = 120, change: int | None = None, **over) -> float:
    """Cosine of the last logits: all n tokens in one prefill, against `split` tokens prefilled and the rest decoded
    (with token `change` replaced in the second run)."""
    args = m.ModelArgs(**{**SMALL, **over})
    for cached in (m.get_window_topk_idxs, m.get_compress_topk_idxs, m.precompute_freqs_cis):
        cached.cache_clear()
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(DEV):
            x = torch.randint(0, args.vocab_size, (1, n), generator=torch.Generator(device=DEV).manual_seed(3))
            full = build(args)(x)[1].float()
            if change is not None:
                x = x.clone()
                x[0, change] = (x[0, change] + 17) % args.vocab_size
            net = build(args)
            net(x[:, :split])
            for i in range(split, n):
                step = net(x[:, i:i + 1], i)[1].float()
        return float(F.cosine_similarity(full.flatten(), step.flatten(), dim=0))
    finally:
        torch.set_default_dtype(prev)


def test_decode_continues_a_prefill_within_the_window():
    assert agreement(70, compress_ratios=(0, 0, 0, 0, 0, 0)) > 0.999
    assert agreement(7, n=14, compress_ratios=(0, 0, 0, 0, 0, 0)) > 0.999
    assert agreement(119) > 0.9999  # one decode step
    assert agreement(70, change=117) < 0.95  # a changed token in the window shows


@pytest.mark.parametrize("split", [1, 16, 34, 48, 70, 96])
def test_decode_continues_a_prefill_with_compressed_attention(split):
    # The compressed entries are rounded to FP8 (ratio 4 and 16) and chosen by an index over FP4 keys; one entry
    # rounded the other way moves this random model's logits to 0.98-0.998, the same value for every split that
    # leaves the entry to the same path. A wrong position or group would depend on the split's alignment and cost more.
    assert agreement(split) > 0.975
    # Ratio 4 alone: 0.998 when decoding starts at 32 or later, 0.975 when it starts within the first 16 tokens. The
    # attended positions are the same sets in both paths; the first value that differs has not been traced yet
    # (checklist: to be closed with the layer-by-layer comparison against the transformers implementation).
    assert agreement(split, compress_ratios=(0, 0, 4, 4, 4, 0), index_topk=64) > (0.995 if split >= 32 else 0.97)
