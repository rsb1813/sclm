# Phase 1: whole-model quality with every layer's routed experts compressed: KL, top-1 agreement and ppl vs the original.
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
from sclm.layerwise import LayerwiseRunner, load_args, load_params_
from sclm.upstream import model as m

import kernel  # noqa: E402  (on sys.path via sclm.upstream)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DEV = "cuda"
TOPK = 64  # reference distribution kept as its top-64 tokens per position
CHUNK = 256
# name -> (bits, group, hadamard) for routed experts; "none" is the uncompressed reference
CONFIGS = {"none": None, "rtn3_had": (3, 64, True), "rtn2_had": (2, 64, True), "rtn3": (3, 64, False), "rtn2": (2, 64, False)}


def quantize_experts_(block, bits: int, group: int, use_rot: bool) -> None:
    """Replace every routed expert Linear of `block` by an RTNLinear, one at a time so peak VRAM stays near one layer."""
    rots = {}
    for expert in block.ffn.experts:
        for name in ("w1", "w2", "w3"):
            lin = getattr(expert, name)
            w = (kernel.unpack_fp4(lin.weight).unflatten(-1, (-1, 32)) * lin.scale.float().unsqueeze(-1)).flatten(-2)
            k = w.size(1)
            rot = rots.setdefault(k, quant.BlockHadamard(k, seed=17 + k)) if use_rot else None
            setattr(expert, name, quant.RTNLinear(w, bits, group, rot))
            del lin, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    cli = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)

    tokens = torch.from_numpy(np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["tokens"]).long()
    seqs = tokens[3::6][:8]  # disjoint from the layer probe's tokens[::6], spread over categories
    runner = LayerwiseRunner(Checkpoint(cli.ckpt), load_args(max_batch_size=2, max_seq_len=tokens.size(1)),
                             AutoTokenizer.from_pretrained(cli.ckpt))
    cfg = CONFIGS[cli.config]
    before_block = (lambda i, block: quantize_experts_(block, *cfg)) if cfg else None
    t0 = time.time()
    hidden = runner.hidden_batches(list(seqs.split(2)), before_block=before_block)
    t_fwd = time.time() - t0

    norm = m.RMSNorm(runner.args.dim, runner.args.norm_eps).to(DEV)
    load_params_(norm, {"weight": runner.ckpt.get("norm.weight")}, "norm.")
    head = runner.ckpt.get("head.weight").to(DEV).float()
    ref_path = os.path.join(DATA, "eval_ref.pt")
    ref = None if cfg is None else torch.load(ref_path, weights_only=True)
    out_ids, out_lp = [], []
    nll_sum = kl_sum = agree = n_pos = n_tgt = 0.0
    for b, h in enumerate(hidden):
        x = norm(h.to(DEV))
        ids_b = seqs[2 * b : 2 * b + 2].to(DEV)
        for s0 in range(0, x.size(1), CHUNK):
            lp = F.log_softmax(F.linear(x[:, s0 : s0 + CHUNK].float(), head), dim=-1)
            tgt = ids_b[:, s0 + 1 : s0 + 1 + CHUNK]
            nll_sum += -lp[:, : tgt.size(1)].gather(-1, tgt.unsqueeze(-1)).sum().item()
            n_tgt += tgt.numel()
            if ref is None:
                top = lp.topk(TOPK, dim=-1)
                out_ids.append(top.indices.to(torch.int32).cpu())
                out_lp.append(top.values.to(torch.float16).cpu())
            else:
                r_ids = ref["ids"][len(out_ids)].to(DEV).long()
                r_lp = ref["lp"][len(out_ids)].to(DEV).float()
                out_ids.append(None)
                kl_sum += (r_lp.exp() * (r_lp - lp.gather(-1, r_ids))).sum().item()
                agree += (lp.argmax(-1) == r_ids[..., 0]).sum().item()
                n_pos += r_ids[..., 0].numel()
    result = {"config": cli.config, "tokens": int(seqs.numel()), "ppl": float(np.exp(nll_sum / n_tgt)),
              "forward_min": t_fwd / 60}
    if ref is None:
        torch.save({"ids": out_ids, "lp": out_lp}, ref_path)
    else:
        result.update({"kl_top64": kl_sum / n_pos, "top1_agree": agree / n_pos})
    with open(os.path.join(DATA, f"eval_{cli.config}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
