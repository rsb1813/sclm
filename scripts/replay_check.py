# Feeds the tokens of a thinking_check run to the tiered runtime again under other options and scores the result: what
# a setting costs per token and what it does to the distributions, on the same text.
#   python -m scripts.replay_check --name q175_long ref                               (the original's distributions, once)
#   python -m scripts.replay_check --name q175_long run --tag ws015 --wait-share 0.15 --tokens 1600
#   python -m scripts.replay_check --name q175_long score base ws015                  (replays against the original)
# A replay cannot show a reply drifting or looping: for that the runtime has to write the text (thinking_check write).
import argparse
import os
import time

import torch
import torch.nn.functional as F

from scripts.thinking_check import TOPK, original, report_costs
from sclm import CKPT, CODES, DATA, MODEL


def saved(name: str, tag: str) -> str:
    return os.path.join(DATA, f"replay_{name}_{tag}.pt")


def compare(ref_ids: torch.Tensor, ref_lp: torch.Tensor, ids: torch.Tensor, lp: torch.Tensor) -> tuple:
    """Per position, from two models' top entries (ids and log-probabilities [tokens, TOPK]): KL(reference || other)
    over the reference's entries, one the other's lack getting their floor; whether the first choices agree; and
    whether the other gives its first choice more than 30% while the reference gives that token less than 2%."""
    ref_ids, ids, ref_lp, lp = ref_ids.long(), ids.long(), ref_lp.float(), lp.float()
    at = ids[:, None, :] == ref_ids[:, :, None]
    other = torch.where(at.any(-1), (lp[:, None, :] * at).sum(-1), lp[:, -1:].expand(-1, TOPK))
    hit = ref_ids == ids[:, :1]
    p_ref = torch.where(hit.any(-1), (ref_lp.exp() * hit).sum(-1), torch.zeros(len(ids)))
    return (ref_lp.exp() * (ref_lp - other)).sum(-1), ids[:, 0] == ref_ids[:, 0], (lp[:, 0].exp() > 0.3) & (p_ref < 0.02)


def ref(cli) -> None:
    run = torch.load(os.path.join(DATA, f"thinking_{cli.name}.pt"), weights_only=True)
    top = original(cli.ckpt, torch.tensor([run["ids"]]))[0].topk(TOPK)
    torch.save({"top_ids": top.indices.to(torch.int32), "top_lp": top.values}, saved(cli.name, "ref"))


def run(cli) -> None:
    from scripts.generate import load_runtime

    written = torch.load(os.path.join(DATA, f"thinking_{cli.name}.pt"), weights_only=True)
    ids = written["ids"][:cli.tokens] if cli.tokens else written["ids"]
    _, cache, decoder, _ = load_runtime(cli)
    costs, top_ids, top_lp = [], [], []
    for i, token in enumerate(ids):
        cache.stats.clear()
        t0 = time.perf_counter()
        out = decoder.logits(torch.tensor([[token]]), i)
        top = F.log_softmax(out[0].float(), dim=-1).topk(TOPK)
        costs.append({"total_s": time.perf_counter() - t0, **cache.stats})
        top_ids.append(top.indices.to(torch.int32).cpu())
        top_lp.append(top.values.cpu())
    top_ids, top_lp = torch.stack(top_ids), torch.stack(top_lp)
    torch.save({"top_ids": top_ids, "top_lp": top_lp}, saved(cli.name, cli.tag))
    first = written["prompt_tokens"] + 64  # the first tokens also fill the cache
    print(f"{cli.tag}: {len(ids)} tokens, {cli.slots} slots, RAM cache {cli.ram_cache}")
    report_costs(costs, first)
    # the exact runtime gives the written run's distributions again, to the bit
    kl, agree, _ = compare(top_ids, top_lp, written["top_ids"][:len(ids)], written["top_lp"][:len(ids)])
    print(f"against the run as it was written: KL {float(kl[first:].mean()):.5f} (max {float(kl[first:].max()):.4f}), "
          f"same first choice {float(agree[first:].float().mean()):.4f}")
    print(f"VRAM: tensors {torch.cuda.memory_allocated() / 2**30:.2f} GiB (peak {torch.cuda.max_memory_allocated() / 2**30:.2f}), "
          f"held {torch.cuda.memory_reserved() / 2**30:.2f} GiB, free on the device {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB")


def score(cli) -> None:
    first = torch.load(os.path.join(DATA, f"thinking_{cli.name}.pt"), weights_only=True)["prompt_tokens"] - 1
    theirs = torch.load(saved(cli.name, "ref"), weights_only=True)
    for tag in cli.tags:
        ours = torch.load(saved(cli.name, tag), weights_only=True)
        rows = slice(first, ours["top_ids"].size(0) - 1)  # the positions that predicted a token of the reply
        kl, agree, sure = compare(theirs["top_ids"][rows], theirs["top_lp"][rows], ours["top_ids"][rows], ours["top_lp"][rows])
        print(f"{tag}: {len(kl)} tokens, KL {float(kl.mean()):.4f} (max {float(kl.max()):.2f}), first choice "
              f"{float(agree.float().mean()):.4f}, confident where the original is below 2%: {int(sure.sum())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ref", "run", "score"])
    ap.add_argument("tags", nargs="*", help="score: the replays to score")
    ap.add_argument("--name", required=True, help="the thinking_check run whose tokens are fed")
    ap.add_argument("--tag", default="base", help="run: the replay's name")
    ap.add_argument("--tokens", type=int, default=0, help="run: only the first so many tokens (0: all)")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--codes", default=CODES, type=lambda v: None if v == "none" else v)
    ap.add_argument("--slots", type=int, default=-1)
    ap.add_argument("--ram-cache", type=int, default=-1)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--guess", type=int, default=3 if MODEL == "v4" else 4)
    ap.add_argument("--touch", type=int, default=8)
    ap.add_argument("--wait-share", type=float, default=0.0)
    ap.add_argument("--ram-share", type=float, default=0.0)
    ap.add_argument("--topk", type=int, default=0)
    cli = ap.parse_args()
    cli.lam, cli.threads, cli.min_ram_margin = 0.1, 8, 0.5
    {"ref": ref, "run": run, "score": score}[cli.phase](cli)


if __name__ == "__main__":
    main()
