# Turns the routing trace into the numbers the runtime design depends on: cache hit rates, reuse, adaptive top-k, prefetch recall.
import json
import os
from collections import OrderedDict

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
N_EXPERTS = 384
# GPU expert-cache capacities to evaluate, in experts out of 40 * 384 = 15360
CAPACITIES = (500, 930, 1600, 2500, 4000)


def split(idx: np.ndarray, seq_len: int):
    """Even sequences pick the cache contents, odd sequences measure it."""
    seqs = idx.reshape(-1, seq_len, *idx.shape[1:])
    return seqs[0::2].reshape(-1, *idx.shape[1:]), seqs[1::2].reshape(-1, *idx.shape[1:])


def counts(idx: np.ndarray) -> np.ndarray:
    """[n_layers, N_EXPERTS] activation counts."""
    n_layers = idx.shape[1]
    out = np.zeros((n_layers, N_EXPERTS), dtype=np.int64)
    for layer in range(n_layers):
        out[layer] = np.bincount(idx[:, layer].ravel(), minlength=N_EXPERTS)
    return out


def static_coverage(train: np.ndarray, test: np.ndarray, capacity: int) -> float:
    """Share of test activations served by the `capacity` (layer, expert) pairs most used in train."""
    c_train, c_test = counts(train).ravel(), counts(test).ravel()
    top = np.argsort(-c_train)[:capacity]
    return c_test[top].sum() / c_test.sum()


def lru_hit_rate(test: np.ndarray, capacity: int, pinned: set[int] = frozenset()) -> float:
    """Decode-order simulation (token by token, layer 0..L-1): pinned pairs always hit, the rest share an LRU."""
    n_layers = test.shape[1]
    keys = test.astype(np.int64) + (np.arange(n_layers)[None, :, None] * N_EXPERTS)
    lru, room = OrderedDict(), capacity - len(pinned)
    hits = total = 0
    for k in keys.reshape(-1).tolist():
        total += 1
        if k in pinned:
            hits += 1
        elif k in lru:
            hits += 1
            lru.move_to_end(k)
        elif room > 0:
            lru[k] = None
            if len(lru) > room:
                lru.popitem(last=False)
    return hits / total


def main() -> None:
    d = np.load(os.path.join(DATA, "routing_v1.npz"))
    idx, w, seq_len = d["indices"], d["weights"].astype(np.float32), int(d["seq_len"])
    names, category = list(d["names"]), d["category"]
    n_tok, n_layers, k = idx.shape
    train, test = split(idx, seq_len)
    report = {"tokens": int(n_tok), "layers": int(n_layers), "top_k": int(k)}

    # 1. skew: per-layer coverage of the hottest p% of experts, ranked on train, measured on test
    c_train, c_test = counts(train), counts(test)
    per_layer = {}
    for p in (5, 10, 20, 30, 50):
        n = int(N_EXPERTS * p / 100)
        cov = [c_test[l][np.argsort(-c_train[l])[:n]].sum() / c_test[l].sum() for l in range(n_layers)]
        per_layer[f"{p}%"] = {"mean": float(np.mean(cov)), "min": float(np.min(cov)), "max": float(np.max(cov))}
    report["per_layer_hot_coverage"] = per_layer
    report["experts_never_used_in_train"] = int((c_train == 0).sum())

    # 2. static global cache and LRU / pinned+LRU, decode order on the test half
    report["cache"] = {}
    for cap in CAPACITIES:
        top = np.argsort(-c_train.ravel())
        pinned = set(int(x) for x in top[: cap // 2])
        report["cache"][cap] = {
            "static": float(static_coverage(train, test, cap)),
            "lru": float(lru_hit_rate(test, cap)),
            "pinned_half_plus_lru": float(lru_hit_rate(test, cap, pinned)),
        }
        print("cache", cap, report["cache"][cap], flush=True)

    # 3. temporal reuse between consecutive tokens of the same sequence
    seqs = test.reshape(-1, seq_len, n_layers, k)
    prev, cur = seqs[:, :-1], seqs[:, 1:]
    overlap = (cur[..., :, None] == prev[..., None, :]).any(-1).sum(-1) / k  # [seq, t, layer]
    report["consecutive_overlap"] = {"mean": float(overlap.mean()), "random_baseline": k / N_EXPERTS}

    # 4. adaptive top-k: experts needed to reach a share of the routing weight
    ws = np.sort(w / w.sum(-1, keepdims=True), axis=-1)[..., ::-1]
    cum = np.cumsum(ws, axis=-1)
    report["adaptive_topk"] = {
        str(t): {"mean_k": float(((cum < t).sum(-1) + 1).clip(max=k).mean())} for t in (0.8, 0.9, 0.95)
    }
    report["weight_by_rank"] = [float(x) for x in ws.reshape(-1, k).mean(0)]

    # 5. per-category hit rate of a general-purpose static cache (capacity 1600)
    top = np.argsort(-c_train.ravel())[:1600]
    test_cat = category[1::2]
    by_cat = {}
    for ci, name in enumerate(names):
        mask = np.repeat(test_cat == ci, seq_len)
        if mask.any():
            c = counts(test[mask]).ravel()
            by_cat[str(name)] = float(c[top].sum() / c.sum())
    report["static1600_by_category"] = by_cat

    # 6. prefetch predictors (recorded during the trace)
    with open(os.path.join(DATA, "routing_v1_predictors.json")) as f:
        pred = json.load(f)
    report["prefetch_recall"] = {
        p: {f"top{kk}": float(np.mean([r[j] for r in pred["recall"][p][1 if p == "A" else 0 :]])) for j, kk in enumerate(pred["recall_k"])}
        for p in ("A", "B")
    }

    with open(os.path.join(DATA, "routing_v1_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k_: v for k_, v in report.items() if k_ != "cache"}, indent=1))


if __name__ == "__main__":
    main()
