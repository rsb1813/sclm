# Checks this project's reference for DeepSeek-V4-Flash-0731 (upstream's inference code on the pure-PyTorch kernels)
# against the transformers implementation: the first N layers of the real checkpoint, the same text, the next-token
# distributions of the two. transformers computes with dequantized weights and no FP8 rounding of the activations,
# so the two agree to a tolerance, not to the bit; a wrong position, norm or routing shows as a large difference.
# Run with SCLM_MODEL=v4. The transformers side runs on the CPU (an expert layer is 13 GB as bf16).
import argparse
import os
import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, DeepseekV4Config, DeepseekV4ForCausalLM
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.initialization import no_init_weights
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf

from sclm.checkpoint import Checkpoint, widen_fp8
from sclm.layerwise import LayerwiseRunner, load_args, load_params_
from sclm.upstream import MODEL, model as m

import kernel  # noqa: E402  (sclm.upstream puts it on the path)


class Experts(nn.Module):
    """Stands in for transformers' DeepseekV4Experts with the same arithmetic, the weights held as bf16 (exact for an
    fp4 value times a power-of-two scale) and widened one expert at a time: 256 experts in fp32 would be 26 GB a layer."""

    def __init__(self, config):
        super().__init__()
        self.num_experts, self.limit = config.num_local_experts, config.swiglu_limit
        self.gate_up = self.down = None  # bf16 [E, 2 * inter, dim] and [E, dim, inter], set after the model is built

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        for e in top_k_index.unique().tolist():
            token_idx, top_k_pos = torch.where(top_k_index == e)
            gate, up = F.linear(hidden_states[token_idx], self.gate_up[e].float()).chunk(2, dim=-1)
            current = F.silu(gate.clamp(max=self.limit)) * up.clamp(min=-self.limit, max=self.limit)
            current = F.linear(current, self.down[e].float()) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final


def renamed(key: str, renames: list) -> str:
    for src, dst in renames:
        key = re.sub(src, dst, key)
    return key


def hf_model(ck: Checkpoint, root: str, n_layers: int):
    """transformers' model cut to the first n_layers, with the checkpoint's weights dequantized into it."""
    cfg = DeepseekV4Config.from_pretrained(root)
    cfg.num_hidden_layers, cfg.layer_types, cfg.mlp_layer_types = n_layers, cfg.layer_types[:n_layers], cfg.mlp_layer_types[:n_layers]
    hf.DeepseekV4Experts = Experts
    with no_init_weights():
        model = DeepseekV4ForCausalLM(cfg).eval()
    renames = []
    for rule in get_checkpoint_conversion_mapping("deepseek_v4"):
        if type(rule).__name__ == "WeightRenaming":
            src, dst = rule.source_patterns, rule.target_patterns
            renames.append((src[0] if isinstance(src, list) else src, (dst[0] if isinstance(dst, list) else dst).replace("\\", "")))
    state, loaded = model.state_dict(), set()
    wanted = [k for k in ck.weight_map if not k.startswith("mtp.") and (not k.startswith("layers.") or int(k.split(".")[1]) < n_layers)]
    for key in wanted:
        if key.endswith(".scale") or ".ffn.experts." in key:
            continue
        value = ck.get(key)
        if value.dtype == torch.float8_e4m3fn:
            value = widen_fp8(value, ck.get(key[:-len("weight")] + "scale"))
        name = renamed(key, renames)
        name = name if name in state else "model." + name
        assert name in state and state[name].shape == value.shape, (key, name, tuple(value.shape))
        state[name].copy_(value.to(state[name].dtype))
        loaded.add(name)
    missing = sorted(set(state) - loaded)
    assert not missing, missing[:8]
    for i, layer in enumerate(model.model.layers):
        mats = {}
        for w in ("w1", "w2", "w3"):
            rows = []
            for e in range(cfg.n_routed_experts):
                packed = ck.get(f"layers.{i}.ffn.experts.{e}.{w}.weight").view(torch.float4_e2m1fn_x2)
                scale = ck.get(f"layers.{i}.ffn.experts.{e}.{w}.scale").float()
                rows.append((kernel.unpack_fp4(packed).unflatten(-1, (-1, 32)) * scale.unsqueeze(-1)).flatten(-2).bfloat16())
            mats[w] = torch.stack(rows)
        layer.mlp.experts.gate_up, layer.mlp.experts.down = torch.cat([mats["w1"], mats["w3"]], dim=1), mats["w2"]
        print(f"transformers layer {i}: experts in place", flush=True)
    return model


def main() -> None:
    assert MODEL == "v4", "run with SCLM_MODEL=v4"
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4-Flash-0731")
    ap.add_argument("--layers", type=int, default=4, help="the first N layers of both (0-3: three hash layers, ratios 0, 0, 4, 128)")
    ap.add_argument("--file", default=os.path.join(os.path.dirname(__file__), "..", "design.md"))
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--no-act-quant", action="store_true", help="run the reference without its FP8/FP4 rounding of "
                    "activations, as transformers computes: what is left of the difference is not that rounding")
    cli = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(cli.ckpt)
    with open(cli.file, encoding="utf-8") as f:
        ids = torch.tensor([[tok.bos_token_id] + tok.encode(f.read(), add_special_tokens=False)[:cli.tokens - 1]])
    ck = Checkpoint(cli.ckpt)

    if cli.no_act_quant:
        def exact_linear(x, weight, bias=None):
            if weight.dtype == torch.float4_e2m1fn_x2:
                w = (kernel.unpack_fp4(weight).unflatten(-1, (-1, 32)) * weight.scale.float().unsqueeze(-1)).flatten(-2)
            elif weight.dtype == torch.float8_e4m3fn:
                w = widen_fp8(weight, weight.scale).float()
            else:
                w = weight.float()
            return F.linear(x.float(), w).to(torch.get_default_dtype())

        m.linear = exact_linear
        m.act_quant = m.fp4_act_quant = lambda x, *a, **k: x  # the in-place uses (window and compressed KV, index keys)
    torch.set_default_dtype(torch.bfloat16)  # as upstream's code runs
    runner = LayerwiseRunner(ck, load_args(max_batch_size=1, max_seq_len=max(1024, cli.tokens)), tok)
    with torch.inference_mode():
        x = runner.hidden(ids, n_layers=cli.layers)
        norm = m.RMSNorm(runner.args.dim, runner.args.norm_eps).to(runner.device)
        load_params_(norm, {"weight": ck.get("norm.weight")}, "norm.")
        ours = F.log_softmax(F.linear(norm(x)[0].float(), ck.get("head.weight").to(runner.device).float()), dim=-1).cpu()
    torch.cuda.empty_cache()

    torch.set_default_dtype(torch.float32)
    model = hf_model(ck, cli.ckpt, cli.layers)
    with torch.inference_mode():
        theirs = F.log_softmax(model(input_ids=ids).logits[0].float(), dim=-1)

    kl = (ours.exp() * (ours - theirs)).sum(-1)
    same = (ours.argmax(-1) == theirs.argmax(-1)).float()
    win = runner.args.window_size
    for label, part in (("all positions", slice(None)), (f"positions below {win} (the window alone)", slice(0, win)),
                        (f"positions from {win} (compressed attention too)", slice(win, None))):
        if kl[part].numel():
            print(f"{label}: KL {kl[part].mean():.5f} (max {kl[part].max():.4f}), same first choice {100 * same[part].mean():.1f}%, "
                  f"largest difference of a log-probability among the reference's top 5: "
                  f"{(ours - theirs).gather(-1, ours.topk(5, -1).indices)[part].abs().max():.3f}")
    # a control: how far apart two different positions' distributions are, to read the numbers above against
    print(f"control, each position against the next one: KL {(ours[:-1].exp() * (ours[:-1] - theirs[1:])).sum(-1).mean():.3f}")


if __name__ == "__main__":
    main()
