# Runs the vendored upstream model end to end on a small random config and checks prefill/decode consistency.
import pytest
import torch

DEV = "cuda"


@pytest.fixture
def bf16_cuda_defaults():
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(DEV)
    yield
    torch.set_default_device("cpu")
    torch.set_default_dtype(prev_dtype)


def init_random_(model: torch.nn.Module, seed: int = 0) -> None:
    """Fill uninitialized parameters with values of a plausible scale for each storage dtype."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    for name, p in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        with torch.no_grad():
            if p.dtype == torch.float4_e2m1fn_x2:
                p.view(torch.uint8).copy_(torch.randint(0, 256, p.shape, generator=g, device=DEV, dtype=torch.uint8))
            elif p.dtype == torch.float8_e4m3fn:
                p.copy_(torch.randn(p.shape, generator=g, device=DEV).to(torch.float8_e4m3fn))
            elif p.dtype == torch.float8_e8m0fnu:
                # fp4 codes have std ~3 and fp8 weights std ~1; aim for weight std ~0.02
                is_fp4 = "experts" in name and "shared" not in name
                p.copy_(torch.full(p.shape, 2.0**-7 if is_fp4 else 2.0**-5, device=DEV).to(torch.float8_e8m0fnu))
            elif "norm" in name or leaf in ("hc_attn_scale", "hc_ffn_scale", "q_weight", "k_weight"):
                p.fill_(1.0)
            elif leaf in ("attn_sink", "bias", "bias_vl", "hc_attn_base", "hc_ffn_base"):
                p.zero_()
            else:
                p.copy_(torch.randn(p.shape, generator=g, device=DEV) * 0.02)


@pytest.fixture
def small_model(bf16_cuda_defaults):
    import model as m

    torch.manual_seed(0)
    args = m.ModelArgs(dspark_block_size=6, dspark_target_layer_ids=(3, 4), temperature=0)
    net = m.Transformer(args)
    init_random_(net)
    return m, args, net


def test_prefill_decode_and_spec_run(small_model):
    m, args, net = small_model
    x = torch.randint(0, args.vocab_size, (2, 150))
    ids, logits, main_hidden = net(x[:, :128])
    assert torch.isfinite(logits).all()
    assert net.forward_spec(ids, main_hidden) is None  # prefill only seeds the draft KV
    for i in range(128, 150):
        ids, logits, main_hidden = net(x[:, i : i + 1], i)
        assert torch.isfinite(logits).all()
        draft_ids, draft_logits, confidence = net.forward_spec(ids, main_hidden, i)
        assert draft_ids.shape == (2, args.dspark_block_size + 1)
        assert torch.isfinite(draft_logits).all() and torch.isfinite(confidence).all()


def test_decode_matches_prefill(small_model):
    m, args, net = small_model
    x = torch.randint(0, args.vocab_size, (2, 150))
    _, ref_logits, _ = net(x)
    net(x[:, :128])
    for i in range(128, 150):
        _, logits, _ = net(x[:, i : i + 1], i)
    cos = torch.nn.functional.cosine_similarity(logits.float(), ref_logits.float(), dim=-1)
    assert (cos > 0.995).all(), cos
    assert torch.equal(logits.argmax(-1), ref_logits.argmax(-1))

    # control: changing only distant tokens (reached through compressed KV) must be visible at this threshold
    y = x.clone()
    y[:, :20] = torch.randint(0, args.vocab_size, (2, 20))
    _, far_logits, _ = net(y)
    far_cos = torch.nn.functional.cosine_similarity(far_logits.float(), ref_logits.float(), dim=-1)
    assert (far_cos < 0.995).all(), far_cos
