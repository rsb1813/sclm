# Phase 1 probe: low-dimensional structure in one layer's experts (activation subspace, shared weight basis) and its MoE error.
import argparse
import json
import os

import numpy as np
import torch

from sclm import CKPT
from sclm.checkpoint import Checkpoint
from scripts.quant_probe import DATA, DEV, expert_out, fp4_weight, fp8_weight, outlier_free_nmse

RANKS = (256, 512, 1024, 1536, 2048, 3072)


def energy_at(evals: torch.Tensor, ranks) -> dict:
    """Share of total eigen-energy in the top-r eigenvalues (evals ascending, as eigh returns them)."""
    e = evals.flip(0).clamp_min(0)
    c = e.cumsum(0) / e.sum()
    return {r: float(c[r - 1]) for r in ranks if r <= e.numel()}


def top_basis(sym: torch.Tensor, r: int) -> torch.Tensor:
    """Top-r eigenvectors [d, r] of a symmetric PSD matrix."""
    _, vecs = torch.linalg.eigh(sym)
    return vecs[:, -r:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--layer", type=int, default=10)
    cli = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)

    data = torch.load(os.path.join(DATA, f"probe_L{cli.layer}.pt"), weights_only=True)
    state = Checkpoint(cli.ckpt).module_state(f"layers.{cli.layer}.ffn.")
    x = data["x"].to(DEV).float()
    idx, gw = data["idx"].to(DEV).long(), data["w"].to(DEV).float()
    n, n_exp = x.size(0), 384
    fit, ev = slice(0, n // 2), slice(n // 2, n)  # bases from the first half, errors on the second
    ref = lambda e: [fp4_weight(state, f"experts.{e}.w{j}") for j in (1, 2, 3)]  # noqa: E731
    report = {"layer": cli.layer, "tokens": n}

    # 1. how many directions the FFN input occupies
    h_x = x[fit].t() @ x[fit] / (n // 2)
    report["input_energy"] = energy_at(torch.linalg.eigvalsh(h_x), RANKS)

    # 2. shared right basis of w1 rows over all experts vs each expert's own SVD
    g_w = torch.zeros(5120, 5120, device=DEV, dtype=torch.float32)  # default dtype is bf16 here
    own = {r: [] for r in RANKS}
    for e in range(n_exp):
        w1 = ref(e)[0]
        g_w += w1.t() @ w1
        if e % 16 == 0:
            s2 = torch.linalg.svdvals(w1).pow(2)
            c = s2.cumsum(0) / s2.sum()
            for r in RANKS:
                if r <= c.numel():
                    own[r].append(float(c[r - 1]))
    report["w1_shared_basis_energy"] = energy_at(torch.linalg.eigvalsh(g_w), RANKS)
    report["w1_own_svd_energy"] = {r: float(np.mean(v)) for r, v in own.items() if v}

    # 3. functional: project w1/w3 inputs onto the top-r activation directions (and w2 outputs onto the top-r
    #    directions of the routed output), measure routed-MoE NMSE on held-out tokens
    tokens_of = [torch.where(idx[ev] == e) for e in range(n_exp)]
    xe = x[ev]

    def routed(weights_for) -> torch.Tensor:
        y = torch.zeros_like(xe)
        for e in range(n_exp):
            rows, slot = tokens_of[e]
            if rows.numel():
                y.index_add_(0, rows, expert_out(xe[rows], *weights_for(e), gw[ev][rows, slot, None]))
        return y

    y_ref = routed(ref)
    nmse_ex1 = outlier_free_nmse(y_ref)
    shared = [fp8_weight(state, f"shared_experts.w{j}") for j in (1, 2, 3)]
    y_full_norm = (y_ref + expert_out(xe, *shared, torch.ones(1, device=DEV))).pow(2).sum()
    # output covariance of the routed part, from the fit half
    tok_fit = [torch.where(idx[fit] == e) for e in range(n_exp)]
    y_fit = torch.zeros_like(x[fit])
    for e in range(n_exp):
        rows, slot = tok_fit[e]
        if rows.numel():
            y_fit.index_add_(0, rows, expert_out(x[fit][rows], *ref(e), gw[fit][rows, slot, None]))
    c_y = y_fit.t() @ y_fit
    report["routed_output_energy"] = energy_at(torch.linalg.eigvalsh(c_y), RANKS)

    report["functional"] = {}
    for r in RANKS:
        u_in = top_basis(h_x, r)
        p_in = u_in @ u_in.t()
        u_out = top_basis(c_y, r)
        p_out = u_out @ u_out.t()
        res = {}
        for name, fn in (
            ("w1w3_input_proj", lambda e: (lambda w: [w[0] @ p_in, w[1], w[2] @ p_in])(ref(e))),
            ("w2_output_proj", lambda e: (lambda w: [w[0], p_out @ w[1], w[2]])(ref(e))),
        ):
            y_q = routed(fn)
            d = (y_q - y_ref).pow(2).sum()
            res[name] = {"routed_nmse": float(d / y_ref.pow(2).sum()), "routed_nmse_ex1": nmse_ex1(y_q),
                         "moe_nmse": float(d / y_full_norm)}
        report["functional"][r] = res
        print(r, json.dumps(res), flush=True)

    with open(os.path.join(DATA, f"structure_L{cli.layer}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "functional"}, indent=1))


if __name__ == "__main__":
    main()
