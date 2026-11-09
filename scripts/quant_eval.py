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
# name -> (bits, group, hadamard) for routed experts; "none" is the uncompressed reference, "dense" runs the
# uncompressed FP4 weights through the same fp32 path RTNLinear uses (isolates the GEMM path from compression)
CONFIGS = {"none": None, "dense": "dense", "rtn3_had": (3, 64, True), "rtn2_had": (2, 64, True),
           "rtn3": (3, 64, False), "rtn2": (2, 64, False),
           # routed experts restricted to the per-layer top-r activation subspaces (scripts/collect_bases.py)
           "proj256": ("proj", 256), "proj512": ("proj", 512),
           "proj512_in": ("proj", 512, "in"), "proj512_out": ("proj", 512, "out")}


class DenseFP4Linear(torch.nn.Module):
    """Uncompressed expert Linear computed like RTNLinear: fp8 activation fake-quant, fp32 dequantized weight, fp32 GEMM."""

    def __init__(self, lin):
        super().__init__()
        self.weight, self.scale = lin.weight, lin.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = kernel.act_quant(x.to(torch.bfloat16).contiguous().clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
        w = (kernel.unpack_fp4(self.weight).unflatten(-1, (-1, 32)) * self.scale.float().unsqueeze(-1)).flatten(-2)
        return F.linear(xq.float(), w).to(torch.get_default_dtype())


def dense_experts_(block) -> None:
    for expert in block.ffn.experts:
        for name in ("w1", "w2", "w3"):
            setattr(expert, name, DenseFP4Linear(getattr(expert, name)))


def act_q(x: torch.Tensor) -> torch.Tensor:
    return kernel.act_quant(x.to(torch.bfloat16).contiguous().clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True).float()


def fp4_dense(lin) -> torch.Tensor:
    return (kernel.unpack_fp4(lin.weight).unflatten(-1, (-1, 32)) * lin.scale.float().unsqueeze(-1)).flatten(-2)


class ProjIn(torch.nn.Module):
    """w1/w3 restricted to the layer's top-r FFN-input directions: W x ~= (W U)(U^T x), U shared by all experts."""

    def __init__(self, w: torch.Tensor, u: torch.Tensor):
        super().__init__()
        self.u, self.wu = u, w @ u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((act_q(x) @ self.u) @ self.wu.t()).to(torch.get_default_dtype())


class ProjOut(torch.nn.Module):
    """w2 restricted to the layer's top-r routed-output directions: W h ~= V (V^T W h)."""

    def __init__(self, w: torch.Tensor, v: torch.Tensor):
        super().__init__()
        self.v, self.vw = v, v.t() @ w

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return ((act_q(h) @ self.vw.t()) @ self.v.t()).to(torch.get_default_dtype())


def project_experts_(layer: int, block, r: int, sides: str = "inout") -> None:
    bases = torch.load(os.path.join(DATA, "bases_v1", f"layer{layer}.pt"), weights_only=True)
    u = bases["in_basis"][:, :r].to(DEV)
    v = bases["out_basis"][:, :r].to(DEV)
    for expert in block.ffn.experts:
        if "in" in sides:
            expert.w1 = ProjIn(fp4_dense(expert.w1), u)
            expert.w3 = ProjIn(fp4_dense(expert.w3), u)
        if "out" in sides:
            expert.w2 = ProjOut(fp4_dense(expert.w2), v)


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


@torch.inference_mode()
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
    if cfg == "dense":
        before_block = lambda i, block: dense_experts_(block)  # noqa: E731
    elif isinstance(cfg, tuple) and cfg[0] == "proj":
        before_block = lambda i, block: project_experts_(i, block, *cfg[1:])  # noqa: E731
    else:
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
    acc = {k: np.zeros(seqs.size(0)) for k in ("nll", "n_tgt", "kl", "agree", "n_pos")}  # per sequence
    for b, h in enumerate(hidden):
        x = norm(h.to(DEV))
        rows = slice(2 * b, 2 * b + x.size(0))
        ids_b = seqs[rows].to(DEV)
        for s0 in range(0, x.size(1), CHUNK):
            lp = F.log_softmax(F.linear(x[:, s0 : s0 + CHUNK].float(), head), dim=-1)
            tgt = ids_b[:, s0 + 1 : s0 + 1 + CHUNK]
            acc["nll"][rows] += (-lp[:, : tgt.size(1)].gather(-1, tgt.unsqueeze(-1))).sum((1, 2)).cpu().numpy()
            acc["n_tgt"][rows] += tgt.size(1)
            if ref is None:
                top = lp.topk(TOPK, dim=-1)
                out_ids.append(top.indices.to(torch.int32).cpu())
                out_lp.append(top.values.to(torch.float16).cpu())
            else:
                r_ids = ref["ids"][len(out_ids)].to(DEV).long()
                r_lp = ref["lp"][len(out_ids)].to(DEV).float()
                out_ids.append(None)
                acc["kl"][rows] += (r_lp.exp() * (r_lp - lp.gather(-1, r_ids))).sum((1, 2)).cpu().numpy()
                acc["agree"][rows] += (lp.argmax(-1) == r_ids[..., 0]).sum(1).cpu().numpy()
                acc["n_pos"][rows] += lp.size(1)
    names = np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["names"]
    cats = np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["category"][3::6][:8]
    per_seq = [{"category": str(names[c]), "ppl": float(np.exp(acc["nll"][i] / acc["n_tgt"][i]))} for i, c in enumerate(cats)]
    result = {"config": cli.config, "tokens": int(seqs.numel()),
              "ppl": float(np.exp(acc["nll"].sum() / acc["n_tgt"].sum())), "forward_min": t_fwd / 60}
    if ref is None:
        torch.save({"ids": out_ids, "lp": out_lp}, ref_path)
    else:
        result.update({"kl_top64": acc["kl"].sum() / acc["n_pos"].sum(), "top1_agree": acc["agree"].sum() / acc["n_pos"].sum()})
        for i, s in enumerate(per_seq):
            s.update({"kl": float(acc["kl"][i] / acc["n_pos"][i]), "top1": float(acc["agree"][i] / acc["n_pos"][i])})
    result["per_seq"] = per_seq
    with open(os.path.join(DATA, f"eval_{cli.config}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
