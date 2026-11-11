# Generates text with the tiered host on the GPU and reports where each token's time goes, with VRAM and RAM use.
import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from scripts import status
from sclm import CKPT, DATA
from sclm import core, host
from sclm.checkpoint import Checkpoint
from sclm.layerwise import VRAM_MARGIN, load_args

GIB = 2**30


def load_runtime(cli, background: int | None = None) -> tuple:
    """(tokenizer, expert cache, decoder, facts about the load) of the tiered model for the options in cli (ckpt,
    codes, slots, lam, threads, max_seq_len, min_ram_margin, wait_share, ram_share, topk, guess, touch, ram_cache). background:
    staging buffers for unwaited reads (default: what the options given need)."""
    torch.set_default_dtype(torch.bfloat16)
    # Windows pages over-subscribed VRAM out to system RAM instead of failing: cap the process so it fails instead
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction((free - VRAM_MARGIN) / total)
    out = {"slots": cli.slots, "lam": cli.lam, "threads": cli.threads, "codes": cli.codes, "ram_cache": cli.ram_cache,
           "vram_free_at_start_gib": free / GIB, "ram_used_at_start_gib": status.ram()[0]}

    tok = AutoTokenizer.from_pretrained(cli.ckpt)
    ck = Checkpoint(cli.ckpt)
    args = load_args(max_batch_size=1, max_seq_len=cli.max_seq_len, **host.TEXT_ONLY)
    t0 = time.perf_counter()
    net = host.build_net(args) if host.V4 else host.build_net(args, tok)
    host.load_checkpoint_(net, ck)
    out["resident_gib"] = torch.cuda.memory_allocated() / GIB
    # the code files go into RAM whole: stop here rather than push the machine into paging. No codes (--codes none):
    # every routed expert is an original one, the lossless model
    need = sum(os.path.getsize(os.path.join(cli.codes, f)) for f in os.listdir(cli.codes) if f.startswith("layer_")) / GIB if cli.codes else 0.0
    used, total_ram = status.ram()
    out.update(codes_gib=need, ram_margin_gib=total_ram - used - need)
    print(f"RAM: {total_ram - used:.1f} GiB available, the codes take {need:.1f} GiB", flush=True)
    if out["ram_margin_gib"] < cli.min_ram_margin:
        raise SystemExit(f"only {out['ram_margin_gib']:.1f} GiB of RAM would be left (--min-ram-margin {cli.min_ram_margin})")
    if background is None:
        background = (12 if cli.wait_share else 0) + (16 if cli.guess else 0)
    cache = host.attach_tiers_(net, args, ck, cli.codes, cli.slots, cli.lam, background=background)
    cache.topk, cache.wait_share, cache.ram_share, cache.guess = cli.topk, cli.wait_share, cli.ram_share, cli.guess
    cache.touch = cli.touch
    if cli.ram_cache < 0:  # what fits, leaving 3 GiB free: measured now that the codes and the model are in RAM
        # (an estimate made before loading was 2 GiB and more too high, and tier20v100's 63 GiB of codes left no room)
        cli.ram_cache = out["ram_cache"] = max(0, min(1200, int((total_ram - status.ram()[0] - 3.0) * GIB / cache.store.size(1))))
    pinned = cli.ram_cache * cache.store.size(1) / GIB  # it cannot be paged out
    if pinned and total_ram - status.ram()[0] - pinned < cli.min_ram_margin:
        raise SystemExit(f"--ram-cache {cli.ram_cache} takes {pinned:.1f} GiB and would leave less than "
                         f"--min-ram-margin {cli.min_ram_margin} GiB of RAM")
    cache.keep_in_ram(cli.ram_cache)
    out.update(load_s=time.perf_counter() - t0, allocated_gib=torch.cuda.memory_allocated() / GIB,
               ram_used_loaded_gib=status.ram()[0])
    print(f"loaded in {out['load_s']:.0f}s: resident {out['resident_gib']:.2f} GiB, with the cache "
          f"{out['allocated_gib']:.2f} GiB of VRAM; system RAM in use {out['ram_used_loaded_gib']:.1f} GiB", flush=True)
    core.set_threads(cli.threads)
    return tok, cache, host.Host(net, cache), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--codes", default=r"C:\models\sclm-codes\tier20v100")
    ap.add_argument("--slots", type=int, default=370, help="original experts the VRAM cache keeps")
    ap.add_argument("--lam", type=float, default=0.1, help="cache score decay per token")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads of the RAM tier")
    ap.add_argument("--prompt", default="대한민국의 수도와 가장 큰 항구 도시를 알려 주고, 두 도시의 차이를 세 문장으로 설명해 줘.")
    ap.add_argument("--raw", action="store_true", help="continue the prompt as plain text, without the chat template")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--min-ram-margin", type=float, default=1.0, help="GiB of RAM that must stay free after loading")
    ap.add_argument("--wait-share", type=float, default=0.0, help="lossy: an expert that has to be read from the SSD is "
                    "waited for only if it carries this share of the token's routing weight (above 1: never); else the "
                    "token goes on without it and it is read in the background")
    ap.add_argument("--ram-share", type=float, default=0.0, help="lossy: leave out CPU-tier experts with less than this "
                    "share of the token's routing weight")
    ap.add_argument("--topk", type=int, default=0, help="lossy: routed experts computed per token (0: all six)")
    ap.add_argument("--guess", type=int, default=4, help="experts of the next layer each layer guesses, to read those on "
                    "the SSD a layer early (0: none)")
    ap.add_argument("--touch", type=int, default=8, help="guessed experts of the next layer whose CPU-tier codes are read "
                    "into the CPU's cache while the device computes the attention before them (0: none)")
    ap.add_argument("--ram-cache", type=int, default=0, help="experts kept in RAM after they leave the VRAM cache, "
                    "18.8 MB each of pinned memory (scripts/ram_cache.py); -1: what fits, up to 1200, leaving 3 GiB free")
    ap.add_argument("--name", default="stage3")
    cli = ap.parse_args()

    tok, cache, decoder, out = load_runtime(cli)

    text = cli.prompt if cli.raw else tok.apply_chat_template([{"role": "user", "content": cli.prompt}],
                                                              add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text)
    t0 = time.perf_counter()
    for pos, token in enumerate(ids):  # a token at a time: a batched prefill runs upstream's kernels, which widen
        logits = decoder.logits(torch.tensor([[token]]), pos)  # every weight and need about 1 GiB more VRAM
    torch.cuda.synchronize()
    out.update(prompt_tokens=len(ids), prefill_s=time.perf_counter() - t0, prefill_stats=dict(cache.stats))
    print(f"prefill: {len(ids)} tokens in {out['prefill_s']:.1f}s", flush=True)

    new, per_token = [], []
    for pos in range(len(ids), len(ids) + cli.max_new):
        nxt = int(logits.argmax(-1))  # greedy, so runs can be compared
        if nxt == tok.eos_token_id:
            break
        new.append(nxt)
        cache.stats.clear()
        t0 = time.perf_counter()
        logits = decoder.logits(torch.tensor([[nxt]]), pos)
        torch.cuda.synchronize()
        per_token.append({"total_s": time.perf_counter() - t0, **cache.stats})
    out["text"] = tok.decode(new)
    print(out["text"], flush=True)

    steady = per_token[len(per_token) // 4:]  # the first tokens also fill the cache
    mean = {k: sum(t.get(k, 0.0) for t in steady) / len(steady) for k in sorted({k for t in steady for k in t})}
    mean["rest_s"] = mean["total_s"] - sum(mean.get(k, 0.0) for k in ("device_s", "ram_s", "read_s", "launch_s"))
    out.update(new_tokens=len(new), per_token=per_token, steady_mean=mean, tok_s=1 / mean["total_s"],
               vram_peak_gib=torch.cuda.max_memory_allocated() / GIB, vram_reserved_gib=torch.cuda.memory_reserved() / GIB,
               gpu_used_gib=(status.gpu() or [None])[0], ram_used_end_gib=status.ram()[0])
    print(f"{out['tok_s']:.2f} tok/s over the last {len(steady)} tokens; per token: " +
          ", ".join(f"{k[:-2]} {1e3 * v:.1f} ms" if k.endswith("_s") else f"{k} {v:.1f}" for k, v in mean.items()))
    print(f"VRAM peak {out['vram_peak_gib']:.2f} GiB (reserved {out['vram_reserved_gib']:.2f}), "
          f"system RAM in use {out['ram_used_end_gib']:.1f} GiB")
    with open(os.path.join(DATA, f"gen_{cli.name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
