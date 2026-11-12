# Measures the server's engine (scripts/serve.py) with several requests open together. Sequences of the calibration
# set are fed as prompts: a replay, the same tokens whatever the schedule. A token's cost is the time since the token
# before it (one is computed at a time), averaged over each stream's tokens after its first --skip, which also fill
# the cache. Every setting feeds the same texts (n open: the first n) and starts with the expert cache empty.
#   python -m scripts.serve_bench --streams 1,2,4,8
#   python -m scripts.serve_bench --streams 4 --slices 8,64,256,0
import argparse
import json
import os
import time

import numpy as np
import torch

from sclm import CKPT, CODES, DATA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="1,2,4,8", help="numbers of requests open together")
    ap.add_argument("--slices", default="64", help="tokens a request is given before the next one's turn (0: one after the other)")
    ap.add_argument("--tokens", type=int, default=400, help="tokens of each stream")
    ap.add_argument("--skip", type=int, default=100, help="tokens at the start of each stream that are not counted")
    ap.add_argument("--set", default="calib_v2", help="the calibration set the texts come from")
    ap.add_argument("--name", default="serve_bench")
    ap.add_argument("--context", type=int, default=262144)
    ap.add_argument("--reserve", type=int, default=32768)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--codes", default=CODES, type=lambda v: None if v == "none" else v)
    ap.add_argument("--slots", type=int, default=-1)
    ap.add_argument("--ram-cache", type=int, default=-1)
    ap.add_argument("--wait-share", type=float, default=0.0)
    ap.add_argument("--guess", type=int, default=3)
    cli = ap.parse_args()
    cli.lam, cli.threads, cli.min_ram_margin, cli.ram_share, cli.topk, cli.touch = 0.1, 8, 0.5, 0.0, 0, 8
    from scripts.generate import load_runtime
    from scripts.serve import CHUNK, Engine, Job

    cli.max_seq_len, cli.positions = CHUNK, cli.context
    tok, cache, decoder, _ = load_runtime(cli, background=28)
    engine = Engine(tok, decoder, cli.context, keep=0, reserve=min(cli.reserve, cli.context))  # nothing is kept
    engine.long = 0  # the texts go in a few tokens at a time: what is measured is the tiered model's tokens
    held = np.load(os.path.join(DATA, cli.set, "tokens.npz"))
    # a text of each category in turn, so that every setting gets a like mix
    rank = np.zeros(len(held["category"]), int)
    for c in np.unique(held["category"]):
        rank[held["category"] == c] = np.arange(int((held["category"] == c).sum()))
    texts = held["tokens"][np.lexsort((held["category"], rank))].tolist()
    log, inner = [], decoder.logits

    def logits(ids, pos):  # when each token was done, in which sequence, and the cache's counters so far
        out = inner(ids, pos)
        torch.cuda.synchronize()
        log.append((time.perf_counter(), decoder.seq, dict(cache.stats)))
        return out

    decoder.logits = logits

    def run(n: int) -> list:
        cache.resize(cache.n_slots)  # empty, and the level in RAM with it
        cache.ram_free += cache.ram_cache.values()
        cache.ram_cache.clear()
        jobs = [Job(text[:cli.tokens], 0) for text in texts[:n]]
        for job in jobs:
            engine.submit(job)
        while engine.active or engine.waiting or not engine.inbox.empty():
            engine.turn()
        return jobs

    rows = []
    print(f"{cache.n_slots} slots, RAM cache {cli.ram_cache}; {cli.tokens} tokens a stream, the first {cli.skip} not counted", flush=True)
    for slice in (int(v) for v in cli.slices.split(",")):
        for n in (int(v) for v in cli.streams.split(",")):
            engine.slice = slice
            del log[:]
            t0 = time.perf_counter()
            jobs = run(n)
            seen, costs = {}, []
            for (t, seq, stats), (before, _, earlier) in zip(log[1:], log):
                seen[seq] = seen.get(seq, 0) + 1
                if seen[seq] > cli.skip:
                    costs.append({"total_s": t - before, **{k: v - earlier.get(k, 0.0) for k, v in stats.items()}})
            mean = {k: sum(c.get(k, 0.0) for c in costs) / len(costs) for k in sorted({k for c in costs for k in c})}
            first = [job.times["start"] - job.times["submit"] for job in jobs]  # until the request's first token
            done = [job.times["end"] - job.times["submit"] for job in jobs]
            row = {"streams": n, "slice": slice, "tok_s": 1 / mean["total_s"], "stream_tok_s": cli.tokens * n / (time.perf_counter() - t0) / n,
                   "first_s": sum(first) / n, "first_max_s": max(first), "done_s": sum(done) / n, "per_token": mean}
            rows.append(row)
            print(f"{n} open, slice {slice}: {row['tok_s']:.2f} tok/s in all ({1e3 * mean['total_s']:.1f} ms a token: device "
                  f"{1e3 * mean.get('device_s', 0):.1f}, CPU tier {1e3 * mean.get('ram_s', 0):.1f}, reads {1e3 * mean.get('read_s', 0):.1f} ms, "
                  f"{mean.get('reads', 0):.1f} reads), {row['stream_tok_s']:.2f} tok/s a stream; first token after "
                  f"{row['first_s']:.1f}s (latest {row['first_max_s']:.1f}s), done after {row['done_s']:.0f}s", flush=True)
    with open(os.path.join(DATA, f"{cli.name}.json"), "w", encoding="utf-8") as f:
        json.dump({"tokens": cli.tokens, "skip": cli.skip, "slots": cache.n_slots, "ram_cache": cli.ram_cache, "wait_share": cli.wait_share,
                   "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
