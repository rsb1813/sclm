# Phase 1 probe: how much one layer's routed experts can be compressed, measured as MoE-output error on real activations.
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from sclm import quant
from sclm.checkpoint import Checkpoint
from sclm.layerwise import LayerwiseRunner, load_args
from sclm.upstream import model as m

import kernel  # noqa: E402  (on sys.path via sclm.upstream)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DEV = "cuda"
SWIGLU_LIMIT = 10.0

# name -> (kind, params); bpw counts weight bits only (group scales included for RTN)
CONFIGS = {
    "rtn3_g64": ("rtn", {"bits": 3, "group": 64}),
    "rtn2_g64": ("rtn", {"bits": 2, "group": 64}),
    "vq_d8_c12": ("vq", {"dim": 8, "code_bits": 12}),
    "vq_d8_c8": ("vq", {"dim": 8, "code_bits": 8}),
    "vq_d16_c12": ("vq", {"dim": 16, "code_bits": 12}),
    "vq_d16_c8": ("vq", {"dim": 16, "code_bits": 8}),
}


def bpw(kind: str, p: dict) -> float:
    return p["bits"] + 16 / p["group"] if kind == "rtn" else p["code_bits"] / p["dim"]


def collect(runner: LayerwiseRunner, layer: int, seqs: torch.Tensor, path: str) -> dict:
    """FFN input (post ffn_norm) and routing of `layer` for every token of `seqs`, run through layers 0..layer."""
    store = {"x": [], "w": [], "idx": []}

    def before_block(i, block):
        if i == layer:
            def on_gate(_mod, inp, out):
                store["x"].append(inp[0].cpu())
                store["w"].append(out[0].cpu())
                store["idx"].append(out[1].cpu())
            block.ffn.gate.register_forward_hook(on_gate)

    runner.hidden_batches(list(seqs.split(2)), n_layers=layer + 1, before_block=before_block)
    data = {k: torch.cat(v) for k, v in store.items()}
    torch.save(data, path)
    return data


def act_q(x: torch.Tensor) -> torch.Tensor:
    """The fp8 activation fake-quant upstream applies before every quantized GEMM."""
    return kernel.act_quant(x.to(torch.bfloat16).contiguous(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True).float()


def expert_out(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor, gate_w: torch.Tensor) -> torch.Tensor:
    """Upstream Expert.forward with dense fp32 weights (routing weight applied before w2, as upstream)."""
    xq = act_q(x)
    g, u = xq @ w1.t(), xq @ w3.t()
    u = u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    g = g.clamp(max=SWIGLU_LIMIT)
    h = F.silu(g) * u * gate_w
    return act_q(h) @ w2.t()


def fp4_weight(state: dict, name: str) -> torch.Tensor:
    w, s = state[name + ".weight"].to(DEV), state[name + ".scale"].to(DEV)
    return (kernel.unpack_fp4(w).unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)).flatten(-2)


def fp8_weight(state: dict, name: str) -> torch.Tensor:
    w, s = state[name + ".weight"].to(DEV).float(), state[name + ".scale"].to(DEV).float()
    return w * s.repeat_interleave(32, 0)[: w.size(0)].repeat_interleave(32, 1)[:, : w.size(1)]


def build_quantizer(kind: str, p: dict, mats, rot: quant.BlockHadamard | None):
    """Returns f(w) -> dequantized w for one matrix type; VQ trains one codebook shared by all experts.
    `mats` is an iterable of sample weights, consumed lazily (all 384 experts in fp32 would be 54GB)."""
    pre = (lambda w: rot.rotate(w)) if rot else (lambda w: w.float())
    post = (lambda w: rot.unrotate(w)) if rot else (lambda w: w)
    if kind == "rtn":
        return lambda w: post(quant.rtn(pre(w), p["bits"], p["group"]))
    g = torch.Generator().manual_seed(0)
    sample = []
    for w in mats:  # ~4K vectors per expert -> ~1.5M training vectors
        v = quant.to_vectors(quant.row_normalize(pre(w))[0], p["dim"])
        sample.append(v[torch.randint(0, v.size(0), (4096,), generator=g).to(DEV)])
    codebook = quant.kmeans(torch.cat(sample), 2 ** p["code_bits"], iters=12)

    def f(w):
        wn, s = quant.row_normalize(pre(w))
        return post(quant.vq_dequant(wn, codebook) * s)

    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    cli = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)

    cache = os.path.join(DATA, f"probe_L{cli.layer}.pt")
    if os.path.exists(cache):
        data = torch.load(cache, weights_only=True)
    else:
        tokens = torch.from_numpy(np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["tokens"]).long()
        seqs = tokens[::6][:8]  # spread over the categories
        runner = LayerwiseRunner(Checkpoint(cli.ckpt), load_args(max_batch_size=2, max_seq_len=tokens.size(1)),
                                 AutoTokenizer.from_pretrained(cli.ckpt))
        t0 = time.time()
        data = collect(runner, cli.layer, seqs, cache)
        print(f"collected {data['x'].shape[0]} tokens in {time.time() - t0:.0f}s", flush=True)

    ckpt = Checkpoint(cli.ckpt)
    state = ckpt.module_state(f"layers.{cli.layer}.ffn.")
    x = data["x"].to(DEV).float()
    idx, gw = data["idx"].to(DEV).long(), data["w"].to(DEV).float()
    n_exp = 384
    tokens_of = [torch.where(idx == e) for e in range(n_exp)]  # (token rows, top-k slot)

    def ref_weights(e: int) -> list[torch.Tensor]:
        return [fp4_weight(state, f"experts.{e}.w{j}") for j in (1, 2, 3)]

    def routed(weights_for) -> torch.Tensor:
        """Routed MoE output; weights are produced and dropped one expert at a time."""
        y = torch.zeros_like(x)
        for e in range(n_exp):
            rows, slot = tokens_of[e]
            if rows.numel():
                y.index_add_(0, rows, expert_out(x[rows], *weights_for(e), gw[rows, slot, None]))
        return y

    y_ref = routed(ref_weights)
    shared = [fp8_weight(state, f"shared_experts.w{j}") for j in (1, 2, 3)]
    y_full = y_ref + expert_out(x, *shared, torch.ones(1, device=DEV))
    in_dims = [w.size(1) for w in ref_weights(0)]
    results = {}
    for name in cli.configs.split(","):
        kind, p = CONFIGS[name]
        for use_rot in (False, True):
            t0 = time.time()
            qs = []
            for j in (0, 1, 2):  # w1, w2, w3 each get their own quantizer/codebook
                rot = quant.BlockHadamard(in_dims[j], seed=17 + j) if use_rot else None
                qs.append(build_quantizer(kind, p, (ref_weights(e)[j] for e in range(0, n_exp, 4)), rot))
            wnmse = []
            for e in range(0, n_exp, 16):
                for j, w in enumerate(ref_weights(e)):
                    wnmse.append(((qs[j](w) - w).pow(2).sum() / w.pow(2).sum()).item())
            wnmse = float(np.mean(wnmse))
            y_q = routed(lambda e: [qs[j](w) for j, w in enumerate(ref_weights(e))])
            d = (y_q - y_ref).pow(2).sum()
            key = name + ("+had" if use_rot else "")
            results[key] = {
                "bpw": bpw(kind, p),
                "weight_nmse": wnmse,
                "routed_nmse": (d / y_ref.pow(2).sum()).item(),
                "moe_nmse": (d / y_full.pow(2).sum()).item(),
                "cos_moe": F.cosine_similarity(y_full, y_full - y_ref + y_q, dim=-1).mean().item(),
            }
            print(f"{key:16s} bpw={results[key]['bpw']:.2f} weight_nmse={wnmse:.4f} "
                  f"routed_nmse={results[key]['routed_nmse']:.4f} moe_nmse={results[key]['moe_nmse']:.4f} "
                  f"cos={results[key]['cos_moe']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(DATA, f"probe_L{cli.layer}_results.json"), "w") as f:
        json.dump({"layer": cli.layer, "tokens": int(x.size(0)), "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
