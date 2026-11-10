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
    # optional (levels, group): 1 bit, ternary and coarser groups for the tiered allocation (design.md T3)
    "rtn1_g128": ("rtn", 1, None, None, 2, 128),
    "rtnt_g128": ("rtn", 2, None, None, 3, 128),
    "gptq1_g128": ("gptq", 1, 0.5, "plain", 2, 128),
    "gptqt_g128": ("gptq", 2, 0.5, "plain", 3, 128),
    "gptq2_g128": ("gptq", 2, 0.5, "plain", 4, 128),
    "gptq3_g128": ("gptq", 3, 0.5, "plain", 8, 128),
    # optional unbiasing metric: "fit" (the GPTQ Hessian), "prior" (layer covariance / identity), "none"
    "gptq1_g128_ubprior": ("gptq", 1, 0.5, "plain", 2, 128, "prior"),
    "gptq1_g128_ubnone": ("gptq", 1, 0.5, "plain", 2, 128, "none"),
    "gptqt_g128_ubprior": ("gptq", 2, 0.5, "plain", 3, 128, "prior"),
    "gptq2_g128_ubprior": ("gptq", 2, 0.5, "plain", 4, 128, "prior"),
    "gptqt_g128_ubnone": ("gptq", 2, 0.5, "plain", 3, 128, "none"),
    "gptq2_g128_ubnone": ("gptq", 2, 0.5, "plain", 4, 128, "none"),
    "gptq3_g128_ubnone": ("gptq", 3, 0.5, "plain", 8, 128, "none"),
    # optional refit: least-squares group steps after GPTQ under the same Hessian mix (quant.refit_steps, design.md T8)
    "gptq1_g128_refit": ("gptq", 1, 0.5, "plain", 2, 128, "none", True),
    "gptqt_g128_refit": ("gptq", 2, 0.5, "plain", 3, 128, "none", True),
    "gptq2_g128_refit": ("gptq", 2, 0.5, "plain", 4, 128, "none", True),
    "gptq3_g128_refit": ("gptq", 3, 0.5, "plain", 8, 128, "none", True),
    # optional Hessian estimate for data-starved experts: "n0=N" sets beta per expert to N / (n_eff + N) (Kish
    # effective count of its weighted tokens), "nb=M,lam=L" adds the tokens that rank the expert 7..6+M, weighted by
    # L times the gate weight they would have had squared
    "gptq1_g128_n025": ("gptq", 1, 0.5, "plain", 2, 128, "none", False, "n0=25"),
    "gptq1_g128_n050": ("gptq", 1, 0.5, "plain", 2, 128, "none", False, "n0=50"),
    "gptq1_g128_n0100": ("gptq", 1, 0.5, "plain", 2, 128, "none", False, "n0=100"),
    "gptq1_g128_nb6": ("gptq", 1, 0.5, "plain", 2, 128, "none", False, "nb=6,lam=1"),
    "gptq1_g128_nb6h": ("gptq", 1, 0.5, "plain", 2, 128, "none", False, "nb=6,lam=0.5"),
    "gptq2_g128_n050": ("gptq", 2, 0.5, "plain", 4, 128, "none", False, "n0=50"),
    "gptq2_g128_nb6": ("gptq", 2, 0.5, "plain", 4, 128, "none", False, "nb=6,lam=1"),
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


def quantize(w: torch.Tensor, h: torch.Tensor | None, bits: int, rot: quant.BlockHadamard, levels: int | None = None,
             group: int = GROUP, h_ub: "torch.Tensor | str" = "fit", refit: list | None = None) -> torch.Tensor:
    """w [E, N, K] -> unbiased dequantized weight in the original basis; GPTQ when h [E, K, K] is given. h_ub is the
    unbiasing metric: "fit" (h itself), "none", or a [K, K] matrix. refit gives each expert's token sets
    [(x [T, K], weight [T], mix)] (original basis) to re-solve the GPTQ steps by least squares."""
    wr, hr = rot_cols(rot, w), None
    if h is None:
        wq = quant.rtn(wr.reshape(-1, wr.size(-1)), bits, group, levels=levels).reshape(wr.shape)
    else:
        hr = rot_cols(rot, rot_cols(rot, h).transpose(-1, -2))  # R^T H R
        codes, steps = quant.gptq_codes(wr, hr, bits, group, levels=levels)
        if refit is not None:
            q = quant.rtn_dequant(codes, torch.ones_like(steps), bits, group, levels)  # levels in units of the step
            steps = torch.stack([quant.refit_steps(q[e], wr[e], [(rot.rotate(x), wt, m) for x, wt, m in sets], group)
                                 for e, sets in enumerate(refit)])
        wq = quant.rtn_dequant(codes, steps, bits, group, levels)
    if isinstance(h_ub, str) and h_ub == "none":
        return rot_cols(rot, wq, inverse=True)
    hu = hr if isinstance(h_ub, str) else rot_cols(rot, rot_cols(rot, h_ub).transpose(-1, -2))
    return rot_cols(rot, unbias(wq, wr, hu), inverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--fit-every", type=int, default=1, help="keep every m-th Hessian chunk (calibration size study)")
    cli = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)

    data = torch.load(os.path.join(DATA, f"probe_L{cli.layer}.pt"), weights_only=True)
    nrm = torch.load(os.path.join(DATA, f"probe_L{cli.layer}_nrd.pt"), weights_only=True)  # scripts/nrd_probe.py
    state = Checkpoint(cli.ckpt).module_state(f"layers.{cli.layer}.ffn.")
    x, idx, gw = data["x"].to(DEV).float(), data["idx"].to(DEV).long(), data["w"].to(DEV).float()
    c, r = nrm["c"].to(DEV).float(), nrm["r"].to(DEV).float()
    n = x.size(0)
    t = torch.arange(n, device=DEV)
    is_ev = (t // 256) % 2 == 1  # interleaved chunks: both halves span every category
    is_fit = ~is_ev & ((t // 512) % cli.fit_every == 0)  # thinning the fit half leaves the held-out half as it is
    fit, ev = is_fit.nonzero().flatten(), is_ev.nonzero().flatten()
    xq = act_q(x)
    omega = c.pow(2) / r.pow(2).sum(-1)  # per-token NRD weight without the gate (added per expert)
    h_layer = {"plain": tr_norm(xq[fit].t() @ xq[fit]),
               "nrd": tr_norm(xq[fit].t() @ (omega[fit, None] * xq[fit]))}
    tok_fit = [torch.where(idx[fit] == e) for e in range(N_EXP)]
    tok_ev = [torch.where(idx[ev] == e) for e in range(N_EXP)]
    # router ranks beyond the top 6 on the fit tokens (routing_trace's selection score), with the gate weight a
    # neighbor would get if selected: its score over the top-6 score sum, times the route scale
    scores = F.softplus(x[fit] @ state["gate.weight"].to(DEV).float().t()).sqrt()
    order = (scores + state["gate.bias"].to(DEV).float()).argsort(-1, descending=True)
    print(f"router top-6 matches recorded routing: {(order[:, :6].sort(-1)[0] == idx[fit].sort(-1)[0]).all(-1).float().mean():.4f}")
    gw_nb = scores.gather(1, order[:, 6:18]) / scores.gather(1, order[:, :6]).sum(-1, keepdim=True) * gw[fit].sum(-1).mean()
    nb_idx = order[:, 6:18]
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

    def compressed(method: str, bits: int, beta, weighting, levels: int | None = None, group: int = GROUP,
                   ub: str = "fit", refit: bool = False, hess: str = ""):
        opt = dict(kv.split("=") for kv in hess.split(",")) if hess else {}
        n0, m, lam = float(opt.get("n0", 0)), int(opt.get("nb", 0)), float(opt.get("lam", 1))
        assert not (refit and opt), "the refit builds its token sets from the routed tokens only"

        def tokens(e: int):  # absolute positions, gate weights, Hessian weight multipliers (lam for neighbors)
            rows, slot = tok_fit[e]
            pos, gt = fit[rows], gw[fit[rows], slot]
            lw = torch.ones(rows.numel(), device=DEV, dtype=torch.float32)
            if m:
                r2, s2 = torch.where(nb_idx[:, :m] == e)
                pos, gt = torch.cat([pos, fit[r2]]), torch.cat([gt, gw_nb[r2, s2]])
                lw = torch.cat([lw, torch.full((r2.numel(),), lam, device=DEV, dtype=torch.float32)])
            return pos, gt, lw

        def mix(wt: torch.Tensor) -> float:  # shrinkage toward the prior falls as the effective token count grows
            return float(n0 / (wt.sum() ** 2 / wt.pow(2).sum().clamp_min(1e-30) + n0)) if n0 else beta

        rots = {k: quant.BlockHadamard(k, seed=17 + k) for k in (5120, 2304)}
        eye = torch.eye(2304, device=DEV, dtype=torch.float32)
        ub1, ub2 = (h_layer[weighting], eye) if ub == "prior" else (ub, ub)
        # the refit's stand-ins for the Hessian priors: 512 layer tokens for w1/w3, unit vectors (identity) for w2
        x_prior = xq[fit[:: max(1, fit.numel() // 512)][:512]]
        ones = lambda t: torch.ones(t.size(0), device=DEV, dtype=torch.float32)  # noqa: E731
        for e0 in range(0, N_EXP, BATCH):
            es = list(range(e0, min(e0 + BATCH, N_EXP)))
            ws = [ref(e) for e in es]
            w1, w2, w3 = (torch.stack([w[j] for w in ws]) for j in range(3))
            h1 = None
            if method == "gptq":
                h1, betas = [], []
                for e in es:
                    pos, gt, lw = tokens(e)
                    xr = xq[pos]
                    wt = gt.pow(2) * lw * (omega[pos] if weighting == "nrd" else 1.0)
                    betas.append(mix(wt) if pos.numel() else beta)
                    h_e = tr_norm(xr.t() @ (wt[:, None] * xr)) if pos.numel() else h_layer[weighting]
                    h1.append((1 - betas[-1]) * h_e + betas[-1] * h_layer[weighting])
                h1 = torch.stack(h1)
            sets1 = None
            if refit:
                sets1 = []
                for e in es:
                    rows, slot = tok_fit[e]
                    xr = xq[fit[rows]]
                    own = [(xr, gw[fit[rows], slot].pow(2), 1 - beta)] if rows.numel() else []
                    sets1.append(own + [(x_prior, ones(x_prior), beta if own else 1.0)])
            w1q, w3q = (quantize(w, h1, bits, rots[5120], levels, group, ub1, sets1) for w in (w1, w3))
            del h1
            h2, sets2 = None, [] if refit else None
            if method == "gptq":
                h2 = []
                for i, e in enumerate(es):
                    pos, gt, lw = tokens(e)
                    if not pos.numel():
                        h2.append(eye)
                        if refit:
                            sets2.append([(eye, ones(eye), 1.0)])
                        continue
                    xr = xq[pos]
                    g, u = xr @ w1q[i].t(), (xr @ w3q[i].t()).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
                    hid = act_q(F.silu(g.clamp(max=SWIGLU_LIMIT)) * u * gt[:, None])
                    wt = (omega[pos] if weighting == "nrd" else 1.0) * lw
                    h2.append((1 - betas[i]) * tr_norm(hid.t() @ (wt[:, None] * hid)) + betas[i] * eye)
                    if refit:
                        sets2.append([(hid, wt, 1 - beta), (eye, ones(eye), beta)])
                h2 = torch.stack(h2)
            w2q = quantize(w2, h2, bits, rots[2304], levels, group, ub2, sets2)
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
        key = name if cli.fit_every == 1 else f"{name}_fit1of{cli.fit_every}"
        results[key] = {"nrd_rel": float((seen(d).mean() / sig).sqrt()),
                        "routed_nmse": float(d.pow(2).sum() / y_ref.pow(2).sum()),
                        "out_gain": float((y_q * y_ref).sum() / y_ref.pow(2).sum()), "sec": time.time() - t0}
        print(key, json.dumps(results[key]), flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
