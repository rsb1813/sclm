# Replays the routing trace through the runtime's VRAM cache policy with a second level of expert bytes in host RAM
# behind it: SSD reads per token for each size and policy of the RAM level.
import argparse
import json
import os
from collections import OrderedDict

import numpy as np

from sclm import core

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
EXPERT_GB = 18.8e6 / 1e9  # an expert's bytes in a pinned buffer


def vram_misses(idx: np.ndarray, on_ssd: np.ndarray, slots: int, lam: float) -> tuple[list, int]:
    """The engine's policy (core.Cache, as ssd.ExpertCache drives it) over the trace, a token at a time and a layer
    at a time. Returns the SSD-tier calls that do not find their expert in VRAM, in decode order, as
    (key, whether the policy keeps it, the key it evicts or -1), and the number of SSD-tier calls."""
    policy, loaded, out, calls = core.Cache(on_ssd.size, slots, lam), [-1] * slots, [], 0
    n_experts = on_ssd.shape[1]
    for t, token in enumerate(idx.tolist()):
        for layer, ids in enumerate(token):
            keys = [layer * n_experts + e for e in ids if on_ssd[layer, e]]
            if not keys:
                continue
            calls += len(keys)
            slot, miss = policy.use(keys, t)
            for k, s, missed in zip(keys, slot.tolist(), miss.tolist()):
                if missed and s < slots:
                    out.append((k, True, loaded[s]))
                    loaded[s] = k
                elif missed:
                    out.append((k, False, -1))
    return out, calls


def replay(misses: list, n: int, policy: str) -> tuple[int, int]:
    """SSD reads left with a RAM level of n experts, and how many of the reads it saved were of bypassed experts.
    fifo: every expert read from the SSD enters, the oldest entry leaves. lru: the same, and a hit makes the entry
    the newest. evicted (what ssd.ExpertCache.keep_in_ram does): an expert enters when it leaves VRAM and leaves
    when it goes back, so its bytes are in one place only; the oldest leaves; a bypassed expert stays, as the
    newest."""
    ram, reads, bypass_hits = OrderedDict(), 0, 0
    for key, kept, evicted in misses:
        hit = key in ram
        reads += not hit
        bypass_hits += hit and not kept
        if policy == "evicted":
            if hit and kept:
                del ram[key]  # its buffer is free once the bytes are back in VRAM, after the evicted expert went out
            elif hit:
                ram.move_to_end(key)
            if evicted >= 0:
                if ram and len(ram) + (hit and kept) >= n:
                    ram.popitem(last=False)
                if len(ram) + (hit and kept) < n:
                    ram[evicted] = None
        elif hit and policy == "lru":
            ram.move_to_end(key)
        elif not hit:
            ram[key] = None
            if len(ram) > n:
                ram.popitem(last=False)
    return reads, bypass_hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", default="tier20v100")
    ap.add_argument("--slots", type=int, default=320)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--sizes", default="0,64,128,256,400,800", help="experts the RAM level holds")
    ap.add_argument("--read-ms", type=float, default=2.1, help="what the token waits for one read")
    ap.add_argument("--waited", type=float, default=0.56, help="share of the reads the token waits for: the others were "
                    "started a layer early on the gate's guess (measured: 10.1 of 17.9 a token)")
    ap.add_argument("--warm", type=int, default=256, help="tokens at the start that are not counted (the caches fill)")
    cli = ap.parse_args()
    with open(os.path.join(DATA, f"alloc_{cli.alloc}.json")) as f:
        on_ssd = np.array([[o == "orig" for o in layer] for layer in json.load(f)["options"]])
    r = np.load(os.path.join(DATA, "routing_v1.npz"))
    idx = r["indices"].astype(np.int64)
    idx = idx[(np.arange(idx.shape[0]) // int(r["seq_len"])) % 6 == 3]  # the evaluation split
    n_tok = idx.shape[0] - cli.warm

    def counted(slots: int) -> tuple[list, list, int]:
        """The misses of the whole trace, those of its warm-up tokens (a prefix of them), and the SSD-tier calls."""
        misses, calls = vram_misses(idx, on_ssd, slots, cli.lam)
        return misses, misses[:len(vram_misses(idx[:cli.warm], on_ssd, slots, cli.lam)[0])], calls

    misses, warm, calls = counted(cli.slots)
    base = (len(misses) - len(warm)) / n_tok
    out = {"tokens": n_tok, "ssd_calls_per_token": calls / idx.shape[0], "reads_per_token": base,
           "bypassed_per_token": sum(not kept for _, kept, _ in misses[len(warm):]) / n_tok}
    print(json.dumps({k: round(v, 2) for k, v in out.items()}), flush=True)
    for n in map(int, cli.sizes.split(",")):
        res = {"ram_gb": n * EXPERT_GB}
        for policy in ("fifo", "lru", "evicted"):
            reads, bypass_hits = replay(misses, n, policy)
            res[policy] = (reads - replay(warm, n, policy)[0]) / n_tok
            res[policy + "_saved_ms"] = (base - res[policy]) * cli.waited * cli.read_ms
            res[policy + "_bypass_hits"] = bypass_hits / idx.shape[0]
        # the same bytes as VRAM slots instead (there is no room for them: what one policy over both levels reaches)
        more, more_warm, _ = counted(cli.slots + n)
        res["one_cache"] = (len(more) - len(more_warm)) / n_tok
        out[f"ram_{n}"] = res
        print(n, json.dumps({k: round(v, 2) for k, v in res.items()}), flush=True)
    with open(os.path.join(DATA, f"ram_cache_{cli.alloc}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
