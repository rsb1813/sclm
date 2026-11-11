# Quality and speed of the runtime itself: the tiered host decodes eval text a token at a time (teacher forcing) and its
# next-token distributions are compared with the original model's (data/eval_ref.pt, as scripts/quant_eval.py does).
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from sclm import CKPT, DATA
from sclm import core, host
from sclm.checkpoint import Checkpoint
from sclm.layerwise import VRAM_MARGIN, load_args

CHUNK = 256  # positions per entry of eval_ref.pt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--codes", default=r"C:\models\sclm-codes\tier20v100")
    ap.add_argument("--slots", type=int, default=320)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seqs", default="0,2,4,7", help="eval sequences (quant_eval's order: two en_web, ko_wiki, chat_en, "
                    "chat_ko, code_inst, code_raw, math)")
    ap.add_argument("--tokens", type=int, default=256, help="tokens decoded per sequence")
    ap.add_argument("--ram-share", type=float, default=0.0, help="lossy: leave out CPU-tier experts with less than this "
                    "share of the token's routing weight")
    ap.add_argument("--topk", type=int, default=0, help="lossy: routed experts computed per token (0: all)")
    ap.add_argument("--wait-share", type=float, default=0.0, help="lossy: an expert that has to be read is waited for "
                    "only if it carries this share of the token's routing weight (above 1: never); else it is left out "
                    "of the token and read in the background")
    ap.add_argument("--guess", type=int, default=4, help="experts of the next layer each layer guesses, to read those on "
                    "the SSD a layer early (0: none)")
    ap.add_argument("--touch", type=int, default=8, help="guessed experts of the next layer whose CPU-tier codes are read "
                    "into the CPU's cache while the device computes the attention before them (0: none)")
    ap.add_argument("--ram-cache", type=int, default=0, help="experts kept in RAM after they leave the VRAM cache, "
                    "18.8 MB each of pinned memory (scripts/ram_cache.py)")
    ap.add_argument("--name", required=True)
    cli = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction((free - VRAM_MARGIN) / total)
    tok = AutoTokenizer.from_pretrained(cli.ckpt)
    ck = Checkpoint(cli.ckpt)
    args = load_args(max_batch_size=1, max_seq_len=max(1024, cli.tokens), **host.TEXT_ONLY)
    net = host.build_net(args, tok)
    host.load_checkpoint_(net, ck)
    cache = host.attach_tiers_(net, args, ck, cli.codes, cli.slots, cli.lam,
                               background=(12 if cli.wait_share else 0) + (16 if cli.guess else 0))
    cache.topk, cache.wait_share, cache.ram_share, cache.guess = cli.topk, cli.wait_share, cli.ram_share, cli.guess
    cache.touch = cli.touch
    cache.keep_in_ram(cli.ram_cache)
    core.set_threads(cli.threads)
    decoder = host.Host(net, cache)

    tokens = torch.from_numpy(np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["tokens"]).long()[3::6][:8]
    ref = torch.load(os.path.join(DATA, "eval_ref.pt"), weights_only=True)
    names = np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))
    cats = [str(names["names"][c]) for c in names["category"][3::6][:8]]
    per_seq, seconds, stats = [], 0.0, {}
    for s in map(int, cli.seqs.split(",")):
        kl, agree = [], 0.0
        cache.stats.clear()
        for p in range(cli.tokens):
            t0 = time.perf_counter()
            logits = decoder.logits(tokens[s, p].view(1, 1), p)
            lp = F.log_softmax(logits[0].float(), dim=-1)
            entry = (s // 2) * (tokens.size(1) // CHUNK) + p // CHUNK
            r_ids = ref["ids"][entry][s % 2, p % CHUNK].to(lp.device).long()
            r_lp = ref["lp"][entry][s % 2, p % CHUNK].to(lp.device).float()
            kl.append(float((r_lp.exp() * (r_lp - lp[r_ids])).sum()))
            agree += float(lp.argmax() == r_ids[0])
            if p >= cli.tokens // 4:  # speed over the part after the cache has filled
                seconds += time.perf_counter() - t0
        for k, v in cache.stats.items():
            stats[k] = stats.get(k, 0.0) + v
        per_seq.append({"category": cats[s], "kl": sum(kl) / cli.tokens, "top1": agree / cli.tokens, "kl_by_position": kl})
        print(json.dumps({k: v for k, v in per_seq[-1].items() if k != "kl_by_position"}), flush=True)
    n = len(per_seq) * cli.tokens
    timed = len(per_seq) * (cli.tokens - cli.tokens // 4)
    out = {"name": cli.name, "slots": cli.slots, "topk": cli.topk, "wait_share": cli.wait_share, "ram_share": cli.ram_share, "guess": cli.guess, "touch": cli.touch, "ram_cache": cli.ram_cache, "tokens": n,
           "kl_top64": sum(x["kl"] for x in per_seq) / len(per_seq), "top1_agree": sum(x["top1"] for x in per_seq) / len(per_seq),
           "tok_s": timed / seconds, "ms_per_token": 1e3 * seconds / timed,
           "vram_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
           "per_token": {k: (1e3 * v / n if k.endswith("_s") else v / n) for k, v in stats.items()}, "per_seq": per_seq}
    out["touch_asked_mb"], out["touch_read_mb"] = (b / 1e6 / (len(per_seq) * cli.tokens) for b in core.touch_bytes())
    with open(os.path.join(DATA, f"hosteval_{cli.name}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_seq"}), flush=True)


if __name__ == "__main__":
    main()
