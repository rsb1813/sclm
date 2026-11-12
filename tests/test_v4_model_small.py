# DeepSeek-V4-Flash-0731's upstream model.py on the pure-PyTorch kernels, as a small random model: a prefill of the whole
# sequence against a prefill followed by decode steps. Run with SCLM_MODEL=v4 (one upstream model per process).
import pytest
import torch
import torch.nn.functional as F

import kernel
from sclm import upstream
from sclm.checkpoint import widen_fp8
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
            if name.endswith("tid2eid"):  # the hash layers' experts of each token, distinct as in the released tables
                order = torch.rand(p.size(0), args.n_routed_experts, generator=torch.Generator(device=DEV).manual_seed(5), dtype=torch.float32)
                p.copy_(order.argsort(-1)[:, :p.size(1)].to(p.dtype))
            elif name.endswith("hc_head_scale"):
                p.fill_(1.0)
            elif name.endswith(("hc_head_base", "ape")):
                p.zero_()
    return net


def exact_linear(x, weight, bias=None):
    """upstream's linear without the FP8 rounding of its input: the dequantized weight, in fp32."""
    if weight.dtype == torch.float4_e2m1fn_x2:
        w = (kernel.unpack_fp4(weight).unflatten(-1, (-1, 32)) * weight.scale.float().unsqueeze(-1)).flatten(-2)
    elif weight.dtype == torch.float8_e4m3fn:
        w = widen_fp8(weight, weight.scale).float()
    else:
        w = weight.float()
    return F.linear(x.float(), w).to(torch.get_default_dtype())


def rotate_any(x):
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)


def agreement(split: int, n: int = 120, change: int | None = None, exact: bool = False, **over) -> tuple[float, float]:
    """The last logits of all n tokens in one prefill against `split` tokens prefilled and the rest decoded (with
    token `change` replaced in the second run): their cosine, and the largest difference over the largest logit.
    exact: in fp32 with no rounding of activations, where the two paths differ only if their logic does."""
    args = m.ModelArgs(**{**SMALL, **over})
    for cached in (m.get_window_topk_idxs, m.get_compress_topk_idxs, m.precompute_freqs_cis):
        cached.cache_clear()
    prev, patched = torch.get_default_dtype(), (m.linear, m.act_quant, m.fp4_act_quant, m.rotate_activation)
    torch.set_default_dtype(torch.float32 if exact else torch.bfloat16)
    if exact:
        m.linear, m.rotate_activation = exact_linear, rotate_any
        m.act_quant = m.fp4_act_quant = lambda x, *a, **k: x  # upstream's own calls are the in-place ones

    def make():
        net = build(args)
        if exact:  # the parameters upstream fixes to bf16 (wo_a, the index's weights_proj)
            for p in net.parameters():
                if p.dtype == torch.bfloat16:
                    p.data = p.data.float()
        return net

    try:
        with torch.device(DEV), torch.no_grad():
            x = torch.randint(0, args.vocab_size, (1, n), generator=torch.Generator(device=DEV).manual_seed(3))
            full = make()(x)[1].float()
            if change is not None:
                x = x.clone()
                x[0, change] = (x[0, change] + 17) % args.vocab_size
            net = make()
            net(x[:, :split])
            for i in range(split, n):
                step = net(x[:, i:i + 1], i)[1].float()
        return float(F.cosine_similarity(full.flatten(), step.flatten(), dim=0)), float((full - step).abs().max() / full.abs().max())
    finally:
        torch.set_default_dtype(prev)
        m.linear, m.act_quant, m.fp4_act_quant, m.rotate_activation = patched


CASES = [dict(compress_ratios=(0,) * 6), dict(compress_ratios=(0, 0, 4, 4, 4, 0), index_topk=64), dict(compress_ratios=(0, 0, 4, 4, 4, 0)),
         dict(compress_ratios=(0, 0, 16, 16, 16, 0)), {}]


@pytest.mark.parametrize("split", [1, 16, 34, 70])
def test_decode_continues_a_prefill_exactly_without_rounding(split):
    # window alone; ratio 4 with the index keeping every entry and keeping 8; ratio 16; all of them (SMALL)
    for over in CASES:
        cosine, diff = agreement(split, exact=True, **over)
        assert cosine > 0.999999 and diff < 1e-3, (over, cosine, diff)


def test_a_changed_token_shows_in_the_exact_comparison():
    for change in (5, 60, 117):  # beyond the window (compressed), and inside it
        assert agreement(34, exact=True, change=change)[0] < 0.999


@pytest.mark.parametrize("split", [1, 34, 70])
def test_decode_continues_a_prefill_as_upstream_computes(split):
    # In bf16 with activations rounded to FP8/FP4 the two paths sum in different orders, and a value rounded the other
    # way is magnified by this random model: 0.99-0.9999 here, against exact agreement without the rounding (above).
    for over in CASES:
        assert agreement(split, **over)[0] > 0.97, over


@pytest.mark.parametrize("parts", [(0, 120), (0, 1, 38, 39, 97, 120), (0, 33, 34, 98)])
def test_tokens_computed_together_continue_a_sequence_exactly_without_rounding(parts):
    """host_v4.V4Attention.together computes a run of tokens that continues a sequence as tensors. In fp32 with no
    rounding, the logits after each run, the logits of the tokens decoded after the runs and the state they leave
    (window ring, compressed key/values, index keys, the compressors' groups) are those of the tokens one at a
    time. The runs begin and end inside groups of 4 and of 16, cross the window ring (16) and what the index keeps
    (8), start the sequence, follow each other and are followed by single tokens."""
    from sclm import gpu
    from sclm.host_v4 import V4Attention

    args = m.ModelArgs(**SMALL)
    for cached in (m.get_window_topk_idxs, m.get_compress_topk_idxs, m.precompute_freqs_cis):
        cached.cache_clear()
    prev, patched = torch.get_default_dtype(), (m.linear, m.act_quant, m.fp4_act_quant, m.rotate_activation)
    torch.set_default_dtype(torch.float32)
    m.linear, m.rotate_activation = exact_linear, rotate_any
    m.act_quant = m.fp4_act_quant = lambda x, *a, **k: x
    upstream_attention, m.Block.attention_cls = m.Block.attention_cls, V4Attention
    n = 120

    def make():
        net = build(args)
        for p in net.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.float()
        return net

    try:
        with torch.device(DEV), torch.no_grad():
            x = torch.randint(0, args.vocab_size, (1, n), generator=torch.Generator(device=DEV).manual_seed(3))
            one, two = make(), make()
            want = [one(x[:, i:i + 1], i)[1].float() for i in range(n)]
            for start, end in zip(parts, parts[1:]):
                gpu.together = True
                try:
                    got = two(x[:, start:end], start)[1].float()
                finally:
                    gpu.together = False
                assert float(F.cosine_similarity(got.flatten(), want[end - 1].flatten(), dim=0)) > 0.999999, (start, end)
            for i in range(parts[-1], n):  # upstream's one-token form goes on from the state the runs left
                got = two(x[:, i:i + 1], i)[1].float()
                assert float(F.cosine_similarity(got.flatten(), want[i].flatten(), dim=0)) > 0.999999, i
            theirs, held = dict(one.named_modules()), 0
            for name, mod in two.named_modules():
                rows = {}  # the rows of each buffer that hold something: a compressor's are the groups not yet pooled
                if isinstance(mod, m.Compressor):
                    live = n % mod.compress_ratio + (mod.compress_ratio if mod.overlap else 0)
                    rows = {"kv_state": live, "score_state": live}
                elif isinstance(mod, (m.Attention, m.Indexer)):
                    rows = {"kv_cache": mod.kv_cache.size(1)}
                for buf, live in rows.items():
                    a, b = (getattr(net, buf)[:, :live].nan_to_num(neginf=-1e9) for net in (theirs[name], mod))
                    assert torch.allclose(a, b, rtol=1e-3, atol=1e-3 * float(a.abs().max())), (name, buf)
                    held += 1
            assert held > 12
    finally:
        torch.set_default_dtype(prev)
        m.linear, m.act_quant, m.fp4_act_quant, m.rotate_activation = patched
        m.Block.attention_cls = upstream_attention
