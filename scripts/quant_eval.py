# Phase 1: whole-model quality with every layer's routed experts compressed: KL, top-1 agreement and ppl vs the original.
import argparse
import json
import os
import time
import weakref

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
SCALES = (0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3)  # logit multipliers for the oracle-temperature KL diagnostic
# name -> (bits, group, hadamard) for routed experts; "none" is the uncompressed reference, "dense" runs the
# uncompressed FP4 weights through the same fp32 path RTNLinear uses (isolates the GEMM path from compression)
CONFIGS = {"none": None, "dense": "dense", "rtn3_had": (3, 64, True), "rtn2_had": (2, 64, True),
           "rtn3": (3, 64, False), "rtn2": (2, 64, False),
           # rows rescaled by 1/beta so rounding does not shrink them (RTNLinear unbiased)
           "rtn3_had_ub": (3, 64, True, True), "rtn2_had_ub": (2, 64, True, True),
           # routed experts restricted to the per-layer top-r activation subspaces (scripts/collect_bases.py)
           "proj256": ("proj", 256), "proj512": ("proj", 512),
           "proj512_in": ("proj", 512, "in"), "proj512_out": ("proj", 512, "out"),
           # per-expert bits from scripts/allocate_bits.py (data/alloc_<name>.json); 4 keeps the original FP4 expert
           "alloc_2p75": ("alloc", "2p75"),
           # GPTQ per expert from calibration sequences (GPTQExperts)
           "gptq2": ("gptq", 2), "gptq3": ("gptq", 3)}


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


def quantize_experts_(block, bits, group: int, use_rot: bool, unbiased: bool = False) -> None:
    """Replace every routed expert Linear of `block` by an RTNLinear, one at a time so peak VRAM stays near one layer.
    `bits` is one width for all experts or a per-expert list, where 4 leaves the original FP4 expert in place."""
    rots = {}
    for e, expert in enumerate(block.ffn.experts):
        b = bits if isinstance(bits, int) else bits[e]
        if not isinstance(bits, int) and b == 4:
            continue
        for name in ("w1", "w2", "w3"):
            lin = getattr(expert, name)
            w = (kernel.unpack_fp4(lin.weight).unflatten(-1, (-1, 32)) * lin.scale.float().unsqueeze(-1)).flatten(-2)
            k = w.size(1)
            rot = rots.setdefault(k, quant.BlockHadamard(k, seed=17 + k)) if use_rot else None
            setattr(expert, name, quant.RTNLinear(w, b, group, rot, unbiased))
            del lin, w


def rotate_h(h: torch.Tensor, rot: quant.BlockHadamard) -> torch.Tensor:
    """R^T H R for Hessians [E, K, K] (the Hadamard basis the weights are quantized in)."""
    from scripts.gptq_probe import rot_cols

    return rot_cols(rot, rot_cols(rot, h).transpose(-1, -2))


def gptq_unbiased(w: torch.Tensor, hr: torch.Tensor, bits: int, rot: quant.BlockHadamard):
    """GPTQ of w [E, N, K] in the Hadamard basis under Hessians hr [E, K, K] already in that basis (rotate_h), rows
    rescaled by the output shrinkage <w_hat, w>_H / <w, w>_H. Returns codes and steps (rotated basis) and the
    dequantized weight (original basis)."""
    from scripts.gptq_probe import rot_cols

    wr = rot_cols(rot, w)
    codes, steps = quant.gptq_codes(wr, hr, bits, 64)
    wq = quant.rtn_dequant(codes, steps, bits, 64)
    beta = (((wq @ hr) * wr).sum(-1) / ((wr @ hr) * wr).sum(-1)).clamp_min(0.05)
    return codes, steps / beta[..., None], rot_cols(rot, wq / beta[..., None], inverse=True)


class GPTQExperts:
    """Collects a layer's FFN inputs and routing while the calibration batches pass, then, right before the first eval
    batch, replaces its routed experts by GPTQ codes: Hadamard basis, per-expert gate^2-weighted X^T X mixed with the
    layer covariance (beta), H-metric unbiasing. The calibration stream stays the uncompressed model's."""

    def __init__(self, bits: int, n_cal: int, beta: float = 0.5, batch: int = 8):
        self.bits, self.n_cal, self.beta, self.batch = bits, n_cal, beta, batch
        self.block, self.store, self.cur, self.t0 = None, [], 0, time.time()
        self.rots = {k: quant.BlockHadamard(k, seed=17 + k) for k in (5120, 2304)}  # quant_eval's usual seeds

    def before_block(self, i: int, block) -> None:
        self.layer, self.block, self.store = i, weakref.ref(block), []  # weak: the runner frees each block itself
        block.ffn.gate.register_forward_hook(self.on_gate)

    def on_gate(self, _mod, inp, out) -> None:
        if self.cur < self.n_cal:
            self.store.append((inp[0].detach().to(torch.bfloat16), out[0].float(), out[1]))

    def before_batch(self, i: int, j: int) -> None:
        self.cur = j
        if j == self.n_cal:
            self.quantize_(self.block())
            self.store = []

    def quantize_(self, block) -> None:
        from scripts.gptq_probe import tr_norm
        from scripts.quant_probe import SWIGLU_LIMIT

        t0 = time.time()
        xq = act_q(torch.cat([s[0] for s in self.store]))
        gw = torch.cat([s[1] for s in self.store])
        idx = torch.cat([s[2] for s in self.store]).long()
        self.store = []
        h_layer = tr_norm(xq.t() @ xq)
        eye = torch.eye(2304, device=DEV, dtype=torch.float32)
        experts = block.ffn.experts
        experts.to("cpu")  # frees ~7GB of FP4 weights for the Hessians; each batch is copied back for its GPTQ
        torch.cuda.empty_cache()
        for e0 in range(0, len(experts), self.batch):
            es = list(range(e0, min(e0 + self.batch, len(experts))))
            toks = [torch.where(idx == e) for e in es]
            w1, w2, w3 = (torch.stack([fp4_dense(getattr(experts[e], n).to(DEV)) for e in es])
                          for n in ("w1", "w2", "w3"))
            h1 = []
            for rows, slot in toks:
                xr = xq[rows]
                h_e = tr_norm(xr.t() @ (gw[rows, slot].pow(2)[:, None] * xr)) if rows.numel() else h_layer
                h1.append((1 - self.beta) * h_e + self.beta * h_layer)
            h1 = rotate_h(torch.stack(h1), self.rots[5120])
            c1, s1, q1 = gptq_unbiased(w1, h1, self.bits, self.rots[5120])
            c3, s3, q3 = gptq_unbiased(w3, h1, self.bits, self.rots[5120])
            del h1, w1, w3
            h2 = []
            for k, (rows, slot) in enumerate(toks):
                if not rows.numel():
                    h2.append(eye)
                    continue
                xr = xq[rows]
                g, u = xr @ q1[k].t(), (xr @ q3[k].t()).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
                hid = act_q(F.silu(g.clamp(max=SWIGLU_LIMIT)) * u * gw[rows, slot, None])
                h2.append((1 - self.beta) * tr_norm(hid.t() @ hid) + self.beta * eye)
            c2, s2, _ = gptq_unbiased(w2, rotate_h(torch.stack(h2), self.rots[2304]), self.bits, self.rots[2304])
            del h2, w2, q1, q3
            for k, e in enumerate(es):
                for name, c, st, rot in (("w1", c1, s1, 5120), ("w2", c2, s2, 2304), ("w3", c3, s3, 5120)):
                    setattr(experts[e], name, quant.RTNLinear.from_codes(c[k], st[k], self.bits, 64, self.rots[rot]))
        print(f"gptq layer {self.layer:2d}: {xq.size(0)} calib tokens, {time.time() - t0:.0f}s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB, total {(time.time() - self.t0) / 60:.0f} min",
              flush=True)


def freeze_routing_(block, layer: int, ref_idx: list, cur: dict) -> None:
    """Make the gate pick the reference model's experts (from data/routing_v1.npz) while computing their weights from
    its own scores, so routing flips caused by upstream error are removed and only expert-output error remains."""
    gate = block.ffn.gate

    def forward(x, image_mask=None):
        scores = m.linear(x.float(), gate.weight.float()) / gate.gate_temp
        if gate.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif gate.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        indices = ref_idx[cur["batch"]][:, layer].to(x.device)
        weights = scores.gather(1, indices)
        if gate.norm_topk_prob and gate.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
        return weights * gate.route_scale, indices

    gate.forward = forward


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--freeze-routing", action="store_true", help="force the reference model's expert choices")
    ap.add_argument("--record-routing", action="store_true", help="with --config none: save the expert choices")
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
    elif isinstance(cfg, tuple) and cfg[0] == "alloc":
        with open(os.path.join(DATA, f"alloc_{cfg[1]}.json")) as f:
            alloc = json.load(f)["bits"]
        before_block = lambda i, block: quantize_experts_(block, alloc[i], 64, True, True)  # noqa: E731
    elif isinstance(cfg, tuple) and cfg[0] == "gptq":
        before_block = None  # set below with the calibration batches
    else:
        before_block = (lambda i, block: quantize_experts_(block, *cfg)) if cfg else None
    t0 = time.time()
    # expert choices of the uncompressed model on the eval batches: [batch][layer] -> [B*S, topk]. routing_v1.npz
    # cannot serve: it was traced before the bf16-operand GEMM change and differs from today's reference path
    routing_path = os.path.join(DATA, "eval_routing_ref.pt")
    before_batch, tag, cur = None, cli.config, {"batch": 0}
    if cli.freeze_routing or cli.record_routing:
        assert not (cli.record_routing and (cfg is not None or cli.freeze_routing))
        inner = before_block
        if cli.freeze_routing:
            tag += "_frz"
            saved = torch.load(routing_path, weights_only=True)
            ref_idx = [torch.stack(layers, 1).long() for layers in saved]  # [B*S, n_layers, topk] per batch
        else:
            recorded = [[] for _ in range(len(seqs) // 2)]

        def before_block(i, block):
            if inner is not None:
                inner(i, block)
            if cli.freeze_routing:
                freeze_routing_(block, i, ref_idx, cur)
            else:
                block.ffn.gate.register_forward_hook(
                    lambda _m, _inp, out: recorded[cur["batch"]].append(out[1].to(torch.int16).cpu()))

        def before_batch(i, j):
            cur["batch"] = j

    batches = list(seqs.split(2))
    n_cal = 0
    if isinstance(cfg, tuple) and cfg[0] == "gptq":
        assert not (cli.freeze_routing or cli.record_routing)
        from scripts.collect_bases import calib_indices

        cal = list(tokens[calib_indices(tokens.size(0), 16)].split(2))  # 15 sequences, none of them eval
        n_cal, batches = len(cal), cal + batches
        gq = GPTQExperts(cfg[1], n_cal)
        before_block, before_batch = gq.before_block, gq.before_batch
    hidden = runner.hidden_batches(batches, before_block=before_block, before_batch=before_batch)[n_cal:]
    t_fwd = time.time() - t0

    norm = m.RMSNorm(runner.args.dim, runner.args.norm_eps).to(DEV)
    load_params_(norm, {"weight": runner.ckpt.get("norm.weight")}, "norm.")
    head = runner.ckpt.get("head.weight").to(DEV).float()
    ref_path = os.path.join(DATA, "eval_ref.pt")
    # the uncompressed run writes the reference; every other run (incl. "none" with frozen routing) compares to it
    ref = None if cfg is None and not cli.freeze_routing else torch.load(ref_path, weights_only=True)
    out_ids, out_lp = [], []
    acc = {k: np.zeros(seqs.size(0)) for k in ("nll", "n_tgt", "kl", "agree", "n_pos", "conf", "conf_ref")}  # per seq
    acc["kl_s"] = np.zeros((seqs.size(0), len(SCALES)))
    for b, h in enumerate(hidden):
        x = norm(h.to(DEV))
        rows = slice(2 * b, 2 * b + x.size(0))
        ids_b = seqs[rows].to(DEV)
        for s0 in range(0, x.size(1), CHUNK):
            z = F.linear(x[:, s0 : s0 + CHUNK].float(), head)
            lp = F.log_softmax(z, dim=-1)
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
                # does a global temperature explain part of the KL? (top-1 log-prob = how sharp each model is)
                acc["conf"][rows] += lp.amax(-1).sum(1).cpu().numpy()
                acc["conf_ref"][rows] += r_lp[..., 0].sum(1).cpu().numpy()
                for a_i, a in enumerate(SCALES):
                    lp_a = F.log_softmax(z * a, dim=-1).gather(-1, r_ids)
                    acc["kl_s"][rows, a_i] += (r_lp.exp() * (r_lp - lp_a)).sum((1, 2)).cpu().numpy()
    names = np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["names"]
    cats = np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["category"][3::6][:8]
    per_seq = [{"category": str(names[c]), "ppl": float(np.exp(acc["nll"][i] / acc["n_tgt"][i]))} for i, c in enumerate(cats)]
    result = {"config": tag, "tokens": int(seqs.numel()),
              "ppl": float(np.exp(acc["nll"].sum() / acc["n_tgt"].sum())), "forward_min": t_fwd / 60}
    if ref is None:
        torch.save({"ids": out_ids, "lp": out_lp}, ref_path)
    if cli.record_routing:
        torch.save(recorded, routing_path)
    else:
        n_pos = acc["n_pos"].sum()
        result.update({"kl_top64": acc["kl"].sum() / n_pos, "top1_agree": acc["agree"].sum() / n_pos,
                       "top1_logprob": acc["conf"].sum() / n_pos, "top1_logprob_ref": acc["conf_ref"].sum() / n_pos,
                       "kl_by_logit_scale": {str(a): acc["kl_s"][:, a_i].sum() / n_pos for a_i, a in enumerate(SCALES)}})
        for i, s in enumerate(per_seq):
            s.update({"kl": float(acc["kl"][i] / acc["n_pos"][i]), "top1": float(acc["agree"][i] / acc["n_pos"][i])})
    result["per_seq"] = per_seq
    with open(os.path.join(DATA, f"eval_{tag}.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
