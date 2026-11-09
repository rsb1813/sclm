# Phase 1: per-expert storage tier and GPTQ option under a RAM capacity and an SSD bandwidth budget (design.md T3).
import argparse
import json
import os
from collections import OrderedDict

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
N_LAYERS, EXPERT_PARAMS = 40, 3 * 2304 * 5120
SSD_BYTES_S = 7.4e9  # sequential read, design.md 1
ORIG_BYTES = EXPERT_PARAMS * 4.25 / 8  # FP4 + e8m0 scale per 32, read straight from the original checkpoint
# RAM option -> (bpw incl. the fp16 step per group, its layer-10 probe in data/gptq_L10.json: GPTQ without unbiasing,
# which beats H-metric unbiasing at every width); ternary packs 5 per byte
RAM_OPTIONS = {"1b128": (1.125, "gptq1_g128_ubnone"), "t128": (1.725, "gptqt_g128_ubnone"),
               "2b128": (2.125, "gptq2_g128_ubnone"), "3b128": (3.125, "gptq3_g128_ubnone")}


def allocate(w: np.ndarray, f: np.ndarray, d: np.ndarray, c: np.ndarray, ram: float, ssd: float) -> np.ndarray:
    """argmin sum_e w_e d_o(e) s.t. sum_e c_o(e) <= ram and sum_{e on SSD} f_e <= ssd. Options are the RAM ones plus a
    last SSD one (d 0, RAM cost 0); both constraints are priced by Lagrange multipliers found by nested bisection."""
    d_x, c_x, s_x = np.append(d, 0.0), np.append(c, 0.0), np.append(np.zeros_like(c), 1.0)

    def pick(lam: float, mu: float) -> np.ndarray:
        return (w[:, None] * d_x + lam * c_x + mu * f[:, None] * s_x).argmin(1)

    def fit_ram(mu: float) -> np.ndarray:
        lo, hi = 0.0, (float(w.max() * d.max()) + mu * float(f.max())) * 10 / float(c.min()) + 1.0  # hi: all on SSD
        for _ in range(60):
            lam = (lo + hi) / 2
            lo, hi = (lam, hi) if c_x[pick(lam, mu)].sum() > ram else (lo, lam)
        return pick(hi, mu)

    lo, hi = 0.0, float((w / np.maximum(f, 1e-12)).max() * d.max()) * 10 + 1.0
    for _ in range(60):
        mu = (lo + hi) / 2
        lo, hi = (mu, hi) if f[fit_ram(mu) == len(c)].sum() > ssd else (lo, mu)
    return fit_ram(hi)


def miss_share(idx_l: np.ndarray, n_exp: int, horizon: float) -> np.ndarray:
    """Per expert, the share of its calls whose previous call lies more than `horizon` tokens back: what an LRU cache
    whose items live `horizon` tokens misses (Che's approximation). idx_l [tokens, topk] in decode order."""
    pos = np.repeat(np.arange(idx_l.shape[0]), idx_l.shape[1])
    exp = idx_l.ravel()
    order = np.lexsort((pos, exp))
    pos, exp = pos[order], exp[order]
    miss = np.ones(pos.size, dtype=bool)
    miss[1:] = (exp[1:] != exp[:-1]) | (np.diff(pos) > horizon)
    calls, misses = np.bincount(exp, minlength=n_exp), np.bincount(exp, weights=miss, minlength=n_exp)
    return np.where(calls > 0, misses / np.maximum(calls, 1), 1.0)


def replay(idx: np.ndarray, on_ssd: np.ndarray, capacity: int) -> tuple[int, int]:
    """Exact global LRU over (layer, expert) on SSD. idx [tokens, layers, topk] in decode order, on_ssd [layers,
    experts]. Returns (SSD calls, cache misses)."""
    layer = np.arange(idx.shape[1])[None, :, None]
    keys = (layer * on_ssd.shape[1] + idx)[on_ssd[layer, idx]].tolist()  # boolean indexing keeps token-major order
    cache, misses = OrderedDict(), 0
    for k in keys:
        if k in cache:
            cache.move_to_end(k)
            continue
        misses += 1
        cache[k] = None
        if len(cache) > capacity:
            cache.popitem(last=False)
    return len(keys), misses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-s", type=float, required=True, help="decode speed whose SSD budget the tier may use")
    ap.add_argument("--cache-experts", type=int, default=0, help="VRAM LRU cache for SSD-tier experts (FP4)")
    ap.add_argument("--ram-bpw", type=float, default=1.10, help="RAM for routed experts, as bpw over all of them")
    ap.add_argument("--options", default=",".join(RAM_OPTIONS))
    ap.add_argument("--name", required=True)
    cli = ap.parse_args()
    names = cli.options.split(",")
    probe = json.load(open(os.path.join(DATA, "gptq_L10.json")))
    d = np.array([probe[RAM_OPTIONS[o][1]]["nrd_rel"] ** 2 for o in names])  # nrd_rel is an RMS ratio
    c = np.array([RAM_OPTIONS[o][0] for o in names])
    ssd_reads = SSD_BYTES_S / (cli.tok_s * N_LAYERS * ORIG_BYTES)  # SSD expert reads per layer per token

    r = np.load(os.path.join(DATA, "routing_v1.npz"))
    seq = np.arange(r["indices"].shape[0]) // int(r["seq_len"])
    calib = seq % 6 != 3  # quant_eval's eval sequences are tokens[3::6][:8]
    idx_all = r["indices"].astype(np.int64)
    idx, g = idx_all[calib], r["weights"][calib].astype(np.float64)
    n_tok, n_exp = idx.shape[0], 384
    w, f = np.zeros((idx.shape[1], n_exp)), np.zeros((idx.shape[1], n_exp))
    for layer in range(idx.shape[1]):
        np.add.at(w[layer], idx[:, layer].ravel(), (g[:, layer] ** 2).ravel() / n_tok)
        np.add.at(f[layer], idx[:, layer].ravel(), 1.0 / n_tok)

    def tiers(horizon: float) -> tuple[list, list, np.ndarray]:
        plan, report = [], []
        for layer in range(idx.shape[1]):
            reads = f[layer] * miss_share(idx[:, layer], n_exp, horizon)
            p = allocate(w[layer], reads, d, c, cli.ram_bpw * n_exp, ssd_reads)
            on_ram = p < len(c)
            opts = [names[i] if i < len(c) else "orig" for i in p]
            report.append({"counts": {o: opts.count(o) for o in names + ["orig"]},
                           "ram_bpw_all": float(c[p[on_ram]].sum() / n_exp),
                           "ram_bpw_resident": float(c[p[on_ram]].mean()),
                           "ssd_call_share": float(f[layer][~on_ram].sum() / f[layer].sum()),
                           "distortion": float((w[layer][on_ram] * d[p[on_ram]]).sum() / w[layer].sum())})
            plan.append(opts)
        return plan, report, np.array([[o == "orig" for o in opts] for opts in plan])

    # an SSD expert costs its cache misses, modeled as calls more than `horizon` tokens after the previous one. The
    # model is optimistic for a shared LRU (misses come in bursts), so the horizon is the largest one whose plan still
    # fits the SSD budget when the calibration trace is replayed through the exact LRU
    plan, report, on_ssd = tiers(0.0)
    lo = 0.0
    if cli.cache_experts:
        hi = 64.0
        for _ in range(8):
            h = (lo + hi) / 2
            cand = tiers(h)
            calls, misses = replay(idx, cand[2], cli.cache_experts)
            print(f"  horizon {h:5.1f}: {calls / n_tok:6.2f} SSD calls/token, {misses / n_tok:5.2f} reads/token",
                  flush=True)
            if misses / n_tok <= ssd_reads * idx.shape[1]:
                lo, (plan, report, on_ssd) = h, cand
            else:
                hi = h
    sim = {}
    for split, mask in (("calib", calib), ("eval", ~calib)):
        calls, misses = replay(idx_all[mask], on_ssd, cli.cache_experts)
        sim[split] = {"calls_per_token": calls / mask.sum(), "reads_per_token": misses / mask.sum(),
                      "ssd_bound_tok_s": SSD_BYTES_S / (misses / mask.sum() * ORIG_BYTES)}
    out = {"tok_s": cli.tok_s, "ram_bpw": cli.ram_bpw, "cache_experts": cli.cache_experts, "horizon": lo,
           "options": plan, "d": dict(zip(names, d.tolist())), "ssd_reads_per_layer": ssd_reads,
           "calib_tokens": int(n_tok), "replay": sim, "layers": report}
    with open(os.path.join(DATA, f"alloc_{cli.name}.json"), "w") as fh:
        json.dump(out, fh)
    mean = lambda k: np.mean([x[k] for x in report])  # noqa: E731
    print(f"{cli.name}: SSD budget {ssd_reads:.3f} reads/layer/token, RAM {mean('ram_bpw_all'):.3f} bpw "
          f"(resident {mean('ram_bpw_resident'):.3f}), SSD call share {mean('ssd_call_share'):.3f}, "
          f"distortion {mean('distortion'):.4f}")
    for split, v in sim.items():
        print(f"  {split} replay: {v['calls_per_token']:.2f} SSD calls/token, {v['reads_per_token']:.2f} reads/token, "
              f"SSD-bound {v['ssd_bound_tok_s']:.1f} tok/s")
    for layer in (0, 20, 39):
        print(f"  layer {layer}: {report[layer]}")


if __name__ == "__main__":
    main()
