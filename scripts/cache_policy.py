# Replays the routing trace through VRAM cache policies for the SSD tier: LRU, S3-FIFO and the offline optimum (MIN).
import argparse
import heapq
import json
import math
import os
from collections import OrderedDict, deque

import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def ssd_keys(idx: np.ndarray, on_ssd: np.ndarray) -> tuple[list, list]:
    """(layer, expert) keys of the SSD-tier calls in decode order (token-major, then layer, then slot), and the
    token index of each."""
    layer = np.arange(idx.shape[1])[None, :, None]
    sel = on_ssd[layer, idx]
    tok = np.broadcast_to(np.arange(idx.shape[0])[:, None, None], idx.shape)
    return (layer * on_ssd.shape[1] + idx)[sel].tolist(), tok[sel].tolist()


def lru(keys: list, cap: int) -> int:
    cache, misses = OrderedDict(), 0
    for k in keys:
        if k in cache:
            cache.move_to_end(k)
            continue
        misses += 1
        cache[k] = None
        if len(cache) > cap:
            cache.popitem(last=False)
    return misses


def s3fifo(keys: list, cap: int, small: float = 0.1, promote: int = 1) -> int:
    """S3-FIFO (Yang et al. 2023): a small probationary FIFO, a main FIFO with lazy promotion, and a ghost FIFO of
    keys evicted from the small one. `promote`: hits in the small FIFO needed to move to the main one."""
    n_s = max(1, int(cap * small))
    n_m = cap - n_s
    s, m, g, freq, where, misses = deque(), deque(), OrderedDict(), {}, {}, 0

    def evict_m():
        while True:
            t = m.popleft()
            if freq[t] > 0:
                freq[t] -= 1
                m.append(t)
            else:
                del freq[t], where[t]
                return

    def evict_s():
        t = s.popleft()
        if freq[t] >= promote:
            m.append(t)
            where[t] = "m"
            if len(m) > n_m:
                evict_m()
        else:
            del freq[t], where[t]
            g[t] = None
            if len(g) > n_m:
                g.popitem(last=False)

    for k in keys:
        if k in where:
            freq[k] = min(freq[k] + 1, 3)
            continue
        misses += 1
        freq[k] = 0
        if k in g:
            del g[k]
            m.append(k)
            where[k] = "m"
            if len(m) > n_m:
                evict_m()
        else:
            s.append(k)
            where[k] = "s"
            if len(s) > n_s:
                evict_s()
    return misses


def belady(keys: list, cap: int, bypass: bool) -> int:
    """Offline optimum: evict the cached key used furthest in the future. With bypass, a missed key whose next use
    is later than every cached key's is computed from the staging buffer and not cached."""
    n = len(keys)
    nxt, last = [n] * n, {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(keys[i], n)
        last[keys[i]] = i
    cached, heap, misses = {}, [], 0  # key -> its next use; max-heap of (-next use, key) with lazy deletion
    for i, k in enumerate(keys):
        if k in cached:
            cached[k] = nxt[i]
            heapq.heappush(heap, (-nxt[i], k))
            continue
        misses += 1
        if len(cached) >= cap:
            while -heap[0][0] != cached.get(heap[0][1], -1):
                heapq.heappop(heap)
            if bypass and nxt[i] >= -heap[0][0]:
                continue
            del cached[heapq.heappop(heap)[1]]
        cached[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return misses


def lrfu(keys: list, toks: list, cap: int, lam: float) -> int:
    """LRFU over every expert's full history (cached or not; there are only 15,360): score = sum over past uses of
    exp(-lam * tokens since that use). Evicts the lowest score; a missed key scoring below every cached key is
    computed from the staging buffer and not cached (bypass). lam -> inf is LRU, lam -> 0 is LFU."""
    loga, cached, heap, misses = {}, {}, [], 0  # log(sum exp(lam * t_use)): the common exp(-lam * t) cancels
    for k, t in zip(keys, toks):
        a = lam * t
        if k in loga:
            b = loga[k]
            a = max(a, b) + math.log1p(math.exp(-abs(a - b)))
        loga[k] = a
        if k not in cached:
            misses += 1
            if len(cached) >= cap:
                while heap[0][0] != cached.get(heap[0][1]):
                    heapq.heappop(heap)
                if a <= heap[0][0]:
                    continue
                del cached[heapq.heappop(heap)[1]]
        cached[k] = a
        heapq.heappush(heap, (a, k))
    return misses


AGE_EDGES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024])


def reuse_table(keys: list, toks: list, n_keys: int, horizon: int, prior: float = 8.0) -> np.ndarray:
    """[key, age bin] -> P(next use within `horizon` tokens | not used for the bin's age), from one trace's gaps,
    shrunk toward the all-key table with `prior` pseudo-gaps."""
    k, t = np.array(keys), np.array(toks)
    o = np.lexsort((t, k))
    k, t = k[o], t[o]
    same = k[1:] == k[:-1]
    gap, kg = (t[1:] - t[:-1])[same], k[1:][same]
    hit, tot = np.zeros((n_keys, len(AGE_EDGES))), np.zeros((n_keys, len(AGE_EDGES)))
    for b, a in enumerate(AGE_EDGES):
        alive = gap >= max(a, 1)
        hit[:, b] = np.bincount(kg[alive & (gap <= a + horizon)], minlength=n_keys)
        tot[:, b] = np.bincount(kg[alive], minlength=n_keys)
    glob = hit.sum(0) / np.maximum(tot.sum(0), 1)
    return (hit + prior * glob) / (tot + prior)


def hazard(keys: list, toks: list, cap: int, table: np.ndarray) -> int:
    """Evict the cached key least likely to be reused soon given its age (reuse_table); bypass a missed key that is
    less likely than all of them."""
    slot_key, slot_last = np.full(cap, -1), np.zeros(cap, dtype=np.int64)
    where, misses, n = {}, 0, 0
    for k, t in zip(keys, toks):
        if k in where:
            slot_last[where[k]] = t
            continue
        misses += 1
        if n < cap:
            i, n = n, n + 1
        else:
            score = table[slot_key, np.searchsorted(AGE_EDGES, t - slot_last, side="right") - 1]
            i = int(score.argmin())
            if table[k, 0] <= score[i]:
                continue
            del where[int(slot_key[i])]
        slot_key[i], slot_last[i], where[k] = k, t, i
    return misses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", default="tier20g")
    ap.add_argument("--capacities", default="200,370,600")
    ap.add_argument("--lams", default="0.01,0.03,0.1,0.3,1", help="LRFU decay per token")
    ap.add_argument("--horizons", default="4,10,25", help="reuse horizons (tokens) of the hazard policy")
    cli = ap.parse_args()
    with open(os.path.join(DATA, f"alloc_{cli.alloc}.json")) as f:
        on_ssd = np.array([[o == "orig" for o in layer] for layer in json.load(f)["options"]])
    r = np.load(os.path.join(DATA, "routing_v1.npz"))
    idx = r["indices"].astype(np.int64)
    seq = np.arange(idx.shape[0]) // int(r["seq_len"])
    out, n_keys = {}, on_ssd.size
    fit_keys, fit_toks = ssd_keys(idx[seq % 6 != 3], on_ssd)  # the hazard tables learn from the calib split only
    tables = {h: reuse_table(fit_keys, fit_toks, n_keys, h) for h in map(int, cli.horizons.split(","))}
    for split, mask in (("calib", seq % 6 != 3), ("eval", seq % 6 == 3)):
        keys, toks = ssd_keys(idx[mask], on_ssd)
        n_tok = int(mask.sum())
        for cap in map(int, cli.capacities.split(",")):
            res = {"calls_per_token": len(keys) / n_tok}
            for name, fn in (("lru", lambda: lru(keys, cap)), ("s3fifo", lambda: s3fifo(keys, cap)),
                             ("s3fifo_p2", lambda: s3fifo(keys, cap, promote=2)),
                             ("min", lambda: belady(keys, cap, False)), ("min_bypass", lambda: belady(keys, cap, True)),
                             *((f"lrfu_{lam:g}", lambda lam=lam: lrfu(keys, toks, cap, lam))
                               for lam in map(float, cli.lams.split(","))),
                             *((f"hazard_{h}", lambda h=h: hazard(keys, toks, cap, tables[h])) for h in tables)):
                res[name] = fn() / n_tok  # misses (SSD reads) per token
            out[f"{split}_{cap}"] = res
            print(split, cap, json.dumps({k: round(v, 2) for k, v in res.items()}), flush=True)
    with open(os.path.join(DATA, f"cache_policy_{cli.alloc}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
