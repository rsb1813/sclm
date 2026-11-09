# Phase 1 probe: does the RMSNorm-aware routed distortion (NRD, design.md 4.2 T2) at layer 10 rank compressors like whole-model KL?
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from sclm import quant
from sclm.checkpoint import Checkpoint
from sclm.layerwise import LayerwiseRunner, load_args
from scripts.quant_probe import DATA, DEV, expert_out, fp4_weight, fp8_weight

N_EXP = 384
# method -> eval_*.json holding its whole-model KL (data/eval_<name>.json)
METHODS = ["rtn3_had", "rtn2_had", "rtn2_had_ub", "proj256", "proj512", "proj512_in", "proj512_out"]


def collect_norms(runner: LayerwiseRunner, layer: int, seqs: torch.Tensor) -> dict:
    """Per token at `layer`: c = sum_s pre_next[s] * ffn_post[s] (how the FFN output enters the next sublayer's
    input) and r, that input before its RMSNorm. Same token order as quant_probe.collect."""
    store = {"c": [], "r": []}

    def before_block(i, block):
        if i != layer:
            return
        calls = []
        orig = block.hc_mixes

        def hc_mixes(*a, **k):
            out = orig(*a, **k)
            calls.append(out)
            return out

        block.hc_mixes = hc_mixes

        def on_block(_mod, _inp, out):
            x_out, pre_next = out
            post = calls[-1][1]  # the second hc_mixes call of each forward is the FFN's
            calls.clear()
            r = (pre_next.unsqueeze(-1).float() * x_out.float()).sum(2)
            store["r"].append(r.flatten(0, 1).to(torch.bfloat16).cpu())
            store["c"].append((pre_next.float() * post.float()).sum(-1).flatten().cpu())

        block.register_forward_hook(on_block)

    runner.hidden_batches(list(seqs.split(2)), n_layers=layer + 1, before_block=before_block)
    return {k: torch.cat(v) for k, v in store.items()}


def make_method(name: str, layer: int):
    """Returns f(w1, w2, w3) -> compressed (w1, w2, w3) in the original basis."""
    if name.startswith("rtn"):
        bits, ub = int(name[3]), name.endswith("_ub")
        rots = {k: quant.BlockHadamard(k, seed=17 + k) for k in (5120, 2304)}  # same seeds as quant_eval

        def q(w):
            rot = rots[w.size(1)]
            wr = rot.rotate(w)
            wq = quant.rtn(wr, bits, 64)
            if ub:
                wq = wq / ((wq * wr).sum(1) / (wr * wr).sum(1)).clamp_min(0.05)[:, None]
            return rot.unrotate(wq)

        return lambda w1, w2, w3: (q(w1), q(w2), q(w3))
    r = int(name[4:7])
    bases = torch.load(os.path.join(DATA, "bases_v1", f"layer{layer}.pt"), weights_only=True)
    u, v = bases["in_basis"][:, :r].to(DEV), bases["out_basis"][:, :r].to(DEV)
    p_in, p_out = u @ u.t(), v @ v.t()
    side_in, side_out = not name.endswith("_out"), not name.endswith("_in")
    return lambda w1, w2, w3: (w1 @ p_in if side_in else w1, p_out @ w2 if side_out else w2,
                               w3 @ p_in if side_in else w3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    ap.add_argument("--layer", type=int, default=10)
    cli = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)

    norms_path = os.path.join(DATA, f"probe_L{cli.layer}_nrd.pt")
    if not os.path.exists(norms_path):
        tokens = torch.from_numpy(np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["tokens"]).long()
        runner = LayerwiseRunner(Checkpoint(cli.ckpt), load_args(max_batch_size=2, max_seq_len=tokens.size(1)),
                                 AutoTokenizer.from_pretrained(cli.ckpt))
        torch.save(collect_norms(runner, cli.layer, tokens[::6][:8]), norms_path)  # quant_probe's sequences
    nrm = torch.load(norms_path, weights_only=True)
    data = torch.load(os.path.join(DATA, f"probe_L{cli.layer}.pt"), weights_only=True)
    state = Checkpoint(cli.ckpt).module_state(f"layers.{cli.layer}.ffn.")
    x, idx, gw = data["x"].to(DEV).float(), data["idx"].to(DEV).long(), data["w"].to(DEV).float()
    c, r = nrm["c"].to(DEV).float(), nrm["r"].to(DEV).float()
    assert r.size(0) == x.size(0), (r.shape, x.shape)
    r_hat = r / r.norm(dim=-1, keepdim=True)
    tokens_of = [torch.where(idx == e) for e in range(N_EXP)]
    ref = lambda e: [fp4_weight(state, f"experts.{e}.w{j}") for j in (1, 2, 3)]  # noqa: E731

    def routed(f) -> torch.Tensor:
        y = torch.zeros_like(x)
        for e in range(N_EXP):
            rows, slot = tokens_of[e]
            if rows.numel():
                y.index_add_(0, rows, expert_out(x[rows], *f(*ref(e)), gw[rows, slot, None]))
        return y

    def seen(d: torch.Tensor) -> torch.Tensor:
        """Per-token change of the next sublayer's normalized input caused by an FFN-output change d (first order:
        RMSNorm drops the component along r and divides by |r|)."""
        d_perp = d - (d * r_hat).sum(-1, keepdim=True) * r_hat
        return (c.abs() * d_perp.norm(dim=-1) / r.norm(dim=-1)).pow(2)

    y_ref = routed(lambda *w: w)
    shared = [fp8_weight(state, f"shared_experts.w{j}") for j in (1, 2, 3)]
    y_full = y_ref + expert_out(x, *shared, torch.ones(1, device=DEV))
    sig = seen(y_full)
    report = {"layer": cli.layer, "tokens": int(x.size(0)),
              "routed_signal_rms": float(seen(y_ref).mean().sqrt()), "ffn_signal_rms": float(sig.mean().sqrt())}
    for name in METHODS:
        y_q = routed(make_method(name, cli.layer))
        d = y_q - y_ref
        nrd = seen(d)
        kl_path = os.path.join(DATA, f"eval_{name}.json")
        kl = json.load(open(kl_path))["kl_top64"] if os.path.exists(kl_path) else None
        res = {"kl": kl, "energy_nmse": float(d.pow(2).sum() / y_full.pow(2).sum()),
               "nrd_rms": float(nrd.mean().sqrt()), "nrd_rel": float((nrd.mean() / sig.mean()).sqrt()),
               "out_gain": float((y_q * y_ref).sum() / y_ref.pow(2).sum())}
        report[name] = res
        print(name, json.dumps(res), flush=True)
    with open(os.path.join(DATA, f"nrd_L{cli.layer}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("routed_signal_rms", "ffn_signal_rms")}))


if __name__ == "__main__":
    main()
