# Phase 1: per-expert bit allocation by a routing-weighted Lagrangian (design.md 4.2 T3), from calibration routing only.
import argparse
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
# expert-output NMSE per option, layer 10 RTN+Had (scripts/gain_probe.py); 4 = keep the original FP4 weights (exact)
D_OUT = {0: 1.0, 2: 0.213, 3: 0.061, 4: 0.0}
COST = {0: 0.0, 2: 2.25, 3: 3.25, 4: 4.25}  # bpw incl. the fp16 step per 64 weights (FP4: e8m0 scale per 32)


def allocate(w: np.ndarray, options: list[int], bpw: float) -> np.ndarray:
    """argmin sum_e w_e D(b_e) s.t. mean cost <= bpw, by bisection on the Lagrange multiplier."""
    d = np.array([D_OUT[b] for b in options])
    c = np.array([COST[b] for b in options])
    lo, hi = 0.0, float(w.max()) * 10 + 1.0
    for _ in range(100):
        lam = (lo + hi) / 2
        pick = np.argmin(w[:, None] * d[None] + lam * c[None], axis=1)
        lo, hi = (lam, hi) if c[pick].mean() > bpw else (lo, lam)
    pick = np.argmin(w[:, None] * d[None] + hi * c[None], axis=1)
    return np.array(options)[pick]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpw", type=float, required=True)
    ap.add_argument("--options", default="2,3,4")
    ap.add_argument("--name", required=True)
    cli = ap.parse_args()
    options = [int(b) for b in cli.options.split(",")]

    d = np.load(os.path.join(DATA, "routing_v1.npz"))
    seq_len = int(d["seq_len"])
    seq = np.arange(d["indices"].shape[0]) // seq_len
    calib = seq % 6 != 3  # quant_eval's eval sequences are tokens[3::6][:8]
    idx, g = d["indices"][calib].astype(np.int64), d["weights"][calib].astype(np.float64)
    bits, report = [], []
    for layer in range(idx.shape[1]):
        w = np.zeros(384)
        np.add.at(w, idx[:, layer].ravel(), (g[:, layer] ** 2).ravel())
        b = allocate(w, options, cli.bpw)
        uniform = min((o for o in options if COST[o] >= cli.bpw), key=lambda o: COST[o])
        report.append({"bpw": float(np.mean([COST[x] for x in b])),
                       "counts": {int(o): int((b == o).sum()) for o in options},
                       "distortion_vs_uniform": float((w * np.array([D_OUT[x] for x in b])).sum()
                                                      / (w * D_OUT[uniform]).sum())})
        bits.append(b.tolist())
    out = {"bpw": cli.bpw, "options": options, "calib_tokens": int(calib.sum()), "bits": bits, "layers": report}
    with open(os.path.join(DATA, f"alloc_{cli.name}.json"), "w") as f:
        json.dump(out, f)
    mean_bpw = np.mean([r["bpw"] for r in report])
    print(f"{cli.name}: mean {mean_bpw:.3f} bpw, layer 0 {report[0]}, layer 20 {report[20]}")


if __name__ == "__main__":
    main()
