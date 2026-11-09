# Phase 1 probe: GPTQ variants vs RTN on one layer's experts; Hessians from held-in tokens, NRD on held-out tokens.
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from sclm import quant
from sclm.checkpoint import Checkpoint
from scripts.quant_probe import DATA, DEV, SWIGLU_LIMIT, act_q, expert_out, fp4_weight, fp8_weight

N_EXP, GROUP, BATCH = 384, 64, 8  # 8 experts: ~6GB of 5120x5120 fp32 Hessians and their factors
# name -> (method, bits, beta, weighting). beta mixes the trace-normalized per-expert Hessian with a prior (the layer's
# input covariance for w1/w3, identity for w2): H = (1 - beta) H_e + beta H_prior. weighting "plain" is X^T X,
# "nrd" weights each token by g^2 c^2 / |r|^2, its share of the next sublayer's normalized-input change (design.md T2)
CONFIGS = {
    "rtn2": ("rtn", 2, None, None),
    "gptq2_plain_b05": ("gptq", 2, 0.5, "plain"),
    "gptq2_nrd_b02": ("gptq", 2, 0.2, "nrd"),
    "gptq2_nrd_b05": ("gptq", 2, 0.5, "nrd"),
    "gptq2_nrd_b1": ("gptq", 2, 1.0, "nrd"),
    "rtn3": ("rtn", 3, None, None),
    "gptq3_nrd_b05": ("gptq", 3, 0.5, "nrd"),
}


def tr_norm(h: torch.Tensor) -> torch.Tensor:
    """Scale [..., K, K] to unit mean diagonal (all-zero matrices stay zero)."""
    return h / torch.diagonal(h, dim1=-2, dim2=-1).mean(-1)[..., None, None].clamp_min(1e-30)


def rot_cols(rot: quant.BlockHadamard, w: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    f = rot.unrotate if inverse else rot.rotate
    return f(w.reshape(-1, w.size(-1))).reshape(w.shape)


def unbias(wq: torch.Tensor, w: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
    """Per output row, divide out the output shrinkage beta = <wq, w>_H / <w, w>_H. Without H this is the weight-space
    ratio RTNLinear uses, fine for RTN whose error is isotropic; GPTQ puts its error where inputs rarely point, so its
    weight-space ratio says nothing about the output and only the input-covariance metric is right."""
    hw = w if h is None else w @ h
    return wq / ((wq * hw).sum(-1, keepdim=True) / (w * hw).sum(-1, keepdim=True)).clamp_min(0.05)


def quantize(w: torch.Tensor, h: torch.Tensor | None, bits: int, rot: quant.BlockHadamard) -> torch.Tensor:
    """w [E, N, K] -> unbiased dequantized weight in the original basis; GPTQ when h [E, K, K] is given."""
    wr, hr = rot_cols(rot, w), None
    if h is None:
        wq = quant.rtn(wr.reshape(-1, wr.size(-1)), bits, GROUP).reshape(wr.shape)
    else:
        hr = rot_cols(rot, rot_cols(rot, h).transpose(-1, -2))  # R^T H R
        wq = quant.rtn_dequant(*quant.gptq_codes(wr, hr, bits, GROUP), bits, GROUP)
    return rot_cols(rot, unbias(wq, wr, hr), inverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    cli = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)

    data = torch.load(os.path.join(DATA, f"probe_L{cli.layer}.pt"), weights_only=True)
    nrm = torch.load(os.path.join(DATA, f"probe_L{cli.layer}_nrd.pt"), weights_only=True)  # scripts/nrd_probe.py
    state = Checkpoint(cli.ckpt).module_state(f"layers.{cli.layer}.ffn.")
    x, idx, gw = data["x"].to(DEV).float(), data["idx"].to(DEV).long(), data["w"].to(DEV).float()
    c, r = nrm["c"].to(DEV).float(), nrm["r"].to(DEV).float()
    n = x.size(0)
    is_fit = (torch.arange(n, device=DEV) // 256) % 2 == 0  # interleaved chunks: both halves span every category
    fit, ev = is_fit.nonzero().flatten(), (~is_fit).nonzero().flatten()
    xq = act_q(x)
    omega = c.pow(2) / r.pow(2).sum(-1)  # per-token NRD weight without the gate (added per expert)
    h_layer = {"plain": tr_norm(xq[fit].t() @ xq[fit]),
               "nrd": tr_norm(xq[fit].t() @ (omega[fit, None] * xq[fit]))}
    tok_fit = [torch.where(idx[fit] == e) for e in range(N_EXP)]
    tok_ev = [torch.where(idx[ev] == e) for e in range(N_EXP)]
    xe, gwe = x[ev], gw[ev]
    r_hat = r[ev] / r[ev].norm(dim=-1, keepdim=True)

    def seen(d: torch.Tensor) -> torch.Tensor:  # as in nrd_probe: change of the next normalized input
        d_perp = d - (d * r_hat).sum(-1, keepdim=True) * r_hat
        return (c[ev].abs() * d_perp.norm(dim=-1) / r[ev].norm(dim=-1)).pow(2)

    ref = lambda e: [fp4_weight(state, f"experts.{e}.w{j}") for j in (1, 2, 3)]  # noqa: E731

    def routed(weights_iter) -> torch.Tensor:
        y = torch.zeros_like(xe)
        for e, w1, w2, w3 in weights_iter:
            rows, slot = tok_ev[e]
            if rows.numel():
                y.index_add_(0, rows, expert_out(xe[rows], w1, w2, w3, gwe[rows, slot, None]))
        return y

    def compressed(method: str, bits: int, beta, weighting):
        rots = {k: quant.BlockHadamard(k, seed=17 + k) for k in (5120, 2304)}
        eye = torch.eye(2304, device=DEV, dtype=torch.float32)
        for e0 in range(0, N_EXP, BATCH):
            es = list(range(e0, min(e0 + BATCH, N_EXP)))
            ws = [ref(e) for e in es]
            w1, w2, w3 = (torch.stack([w[j] for w in ws]) for j in range(3))
            h1 = None
            if method == "gptq":
                h1 = []
                for e in es:
                    rows, slot = tok_fit[e]
                    xr = xq[fit[rows]]
                    wt = gw[fit[rows], slot].pow(2) * (omega[fit[rows]] if weighting == "nrd" else 1.0)
                    h_e = tr_norm(xr.t() @ (wt[:, None] * xr)) if rows.numel() else h_layer[weighting]
                    h1.append((1 - beta) * h_e + beta * h_layer[weighting])
                h1 = torch.stack(h1)
            w1q, w3q = quantize(w1, h1, bits, rots[5120]), quantize(w3, h1, bits, rots[5120])
            del h1
            h2 = None
            if method == "gptq":
                h2 = []
                for i, e in enumerate(es):
                    rows, slot = tok_fit[e]
                    if not rows.numel():
                        h2.append(eye)
                        continue
                    xr = xq[fit[rows]]
                    g, u = xr @ w1q[i].t(), (xr @ w3q[i].t()).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
                    hid = act_q(F.silu(g.clamp(max=SWIGLU_LIMIT)) * u * gw[fit[rows], slot, None])
                    wt = omega[fit[rows]] if weighting == "nrd" else torch.ones(rows.numel(), device=DEV, dtype=torch.float32)
                    h2.append((1 - beta) * tr_norm(hid.t() @ (wt[:, None] * hid)) + beta * eye)
                h2 = torch.stack(h2)
            w2q = quantize(w2, h2, bits, rots[2304])
            for i, e in enumerate(es):
                yield e, w1q[i], w2q[i], w3q[i]

    y_ref = routed((e, *ref(e)) for e in range(N_EXP))
    shared = [fp8_weight(state, f"shared_experts.w{j}") for j in (1, 2, 3)]
    sig = seen(y_ref + expert_out(xe, *shared, torch.ones(1, device=DEV))).mean()
    out_path = os.path.join(DATA, f"gptq_L{cli.layer}.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name in cli.configs.split(","):
        t0 = time.time()
        y_q = routed(compressed(*CONFIGS[name]))
        d = y_q - y_ref
        results[name] = {"nrd_rel": float((seen(d).mean() / sig).sqrt()),
                         "routed_nmse": float(d.pow(2).sum() / y_ref.pow(2).sum()),
                         "out_gain": float((y_q * y_ref).sum() / y_ref.pow(2).sum()), "sec": time.time() - t0}
        print(name, json.dumps(results[name]), flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
