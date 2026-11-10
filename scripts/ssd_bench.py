# Measures the SSD tier on the real checkpoint: expert read speed by batch and chunk size, then the eval routing trace
# replayed through the engine's cache with every miss actually read (on demand per layer, and a token's misses at once).
import argparse
import json
import os
import time

import numpy as np
import torch

from scripts import cache_policy
from sclm import core
from sclm.checkpoint import Checkpoint
from sclm.ssd import ExpertReader

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def verify(ck: Checkpoint, rd: ExpertReader, experts: list) -> None:
    """The reader's bytes equal the tensors read the plain way."""
    for layer, e in experts:
        buf = rd.staging()
        starts = rd.submit(layer, e, buf)
        rd.wait()
        for start, part in zip(starts, ("scale", "weight")):
            want = np.concatenate([ck.get(f"layers.{layer}.ffn.experts.{e}.{w}.{part}").view(torch.uint8).numpy().reshape(-1)
                                   for w in ("w1", "w2", "w3")])
            assert np.array_equal(buf[start:start + len(want)], want), (layer, e, part)


def raw(ck: Checkpoint, experts: list, pool: list, chunks: list, batches: list) -> list:
    out = []
    for chunk in chunks:
        rd = ExpertReader(ck, chunk)
        for batch in batches:
            t0 = time.perf_counter()
            for i in range(0, len(experts), batch):
                for (layer, e), buf in zip(experts[i:i + batch], pool):
                    rd.submit(layer, e, buf)
                rd.wait()
            dt = time.perf_counter() - t0
            out.append({"chunk": chunk, "batch": batch, "ms_per_expert": 1e3 * dt / len(experts),
                        "gb_per_s": len(experts) * sum(rd.sizes) / dt / 1e9})
            print("raw", json.dumps({k: round(v, 2) for k, v in out[-1].items()}), flush=True)
    return out


def calls_of(idx: np.ndarray, on_ssd: np.ndarray):
    """Per token, per layer: the SSD-tier experts among the routed ones, as cache keys."""
    n_exp = on_ssd.shape[1]
    for t in range(idx.shape[0]):
        yield t, [[layer * n_exp + int(e) for e in idx[t, layer] if on_ssd[layer, e]] for layer in range(idx.shape[1])]


def replay(rd, idx, on_ssd, slots, lam, windows, pool, mode):
    """Runs the cache over the whole trace; inside the windows the misses are read. Returns (misses per token over
    the trace, [(ms, misses)] per window token)."""
    cache, n_exp, misses, timed = core.Cache(on_ssd.size, slots, lam), on_ssd.shape[1], 0, []
    for t, layers in calls_of(idx, on_ssd):
        live = any(a <= t < b for a, b in windows)
        t0, n, used = time.perf_counter(), 0, 0
        for keys in layers:
            if not keys:
                continue
            miss = cache.use(keys, t)[1]
            n += int(miss.sum())
            if not live:
                continue
            for k in np.asarray(keys)[miss].tolist():
                rd.submit(k // n_exp, k % n_exp, pool[used])
                used += 1
            if mode == "layer":
                rd.wait()
                used = 0
        if live:
            rd.wait()
            timed.append((1e3 * (time.perf_counter() - t0), n))
        misses += n
    return misses / idx.shape[0], timed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    ap.add_argument("--alloc", default="tier20v100")
    ap.add_argument("--slots", type=int, default=370)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--experts", type=int, default=48, help="experts read per raw-speed setting")
    ap.add_argument("--tokens", type=int, default=50, help="tokens read per window")
    ap.add_argument("--windows", type=int, default=3, help="windows, each in a different eval sequence")
    ap.add_argument("--chunk", type=int, default=4 << 20, help="read size of the replay")
    cli = ap.parse_args()

    with open(os.path.join(DATA, f"alloc_{cli.alloc}.json")) as f:
        on_ssd = np.array([[o == "orig" for o in layer] for layer in json.load(f)["options"]])
    r = np.load(os.path.join(DATA, "routing_v1.npz"))
    seq_len = int(r["seq_len"])
    idx = r["indices"].astype(np.int64)
    idx = idx[(np.arange(idx.shape[0]) // seq_len) % 6 == 3]  # the eval split, as cache_policy.py
    ck = Checkpoint(cli.ckpt)
    rd = ExpertReader(ck, cli.chunk)
    out = {"alloc": cli.alloc, "slots": cli.slots, "lam": cli.lam, "expert_bytes": sum(rd.sizes), "eval_tokens": int(idx.shape[0])}

    rng = np.random.default_rng(0)
    ssd = np.argwhere(on_ssd)
    experts = [tuple(map(int, x)) for x in ssd[rng.choice(len(ssd), cli.experts, replace=False)]]
    verify(ck, rd, experts[:2])
    print("verified: reader bytes equal Checkpoint.get", flush=True)

    # the staging pool: as many buffers as the busiest window token needs, touched once so no page is faulted in while timing
    windows = [(s * 2 * seq_len + seq_len // 3, s * 2 * seq_len + seq_len // 3 + cli.tokens) for s in range(cli.windows)]
    dry = core.Cache(on_ssd.size, cli.slots, cli.lam)
    per_token = [sum(int(dry.use(k, t)[1].sum()) for k in layers if k) for t, layers in calls_of(idx, on_ssd)]
    need = max(16, max(max(per_token[a:b]) for a, b in windows))
    pool = [rd.staging() for _ in range(need)]
    for buf in pool:
        buf.fill(0)
    keys, toks = cache_policy.ssd_keys(idx, on_ssd)
    out["misses_per_token"] = {"engine": float(np.mean(per_token)), "lrfu_replay": cache_policy.lrfu(keys, toks, cli.slots, cli.lam) / len(idx),
                               "lru_replay": cache_policy.lru(keys, cli.slots) / len(idx), "calls": len(keys) / len(idx)}
    print("misses/token", json.dumps({k: round(v, 2) for k, v in out["misses_per_token"].items()}), "staging buffers", need, flush=True)

    out["raw"] = raw(ck, experts, pool, [256 << 10, 1 << 20, 4 << 20, 32 << 20], [1, 4, 16])

    out["replay"] = {}
    for mode in ("layer", "token"):
        _, timed = replay(rd, idx, on_ssd, cli.slots, cli.lam, windows, pool, mode)
        ms, n = np.array([m for m, _ in timed]), np.array([n for _, n in timed])
        res = {"tokens": len(ms), "misses_per_token": float(n.mean()), "ms_per_token": float(ms.mean()),
               "ms_p50": float(np.median(ms)), "ms_p90": float(np.percentile(ms, 90)), "ms_max": float(ms.max()),
               "ms_per_miss": float(ms.sum() / n.sum()), "gb_per_s": float(n.sum() * sum(rd.sizes) / ms.sum() / 1e6)}
        res["ms_per_token_at_trace_rate"] = res["ms_per_miss"] * out["misses_per_token"]["engine"]
        out["replay"][mode] = res
        print("replay", mode, json.dumps({k: round(v, 2) for k, v in res.items()}), flush=True)
    with open(os.path.join(DATA, "ssd_reader.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
