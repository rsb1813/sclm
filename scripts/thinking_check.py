# Scores a long thinking-mode reply written by the tiered runtime against the original model, token by token: the
# runtime answers a question (sampling as the chat does), then the original reads the same tokens and both models'
# next-token distributions are compared at every generated position. Two runs, since the two do not fit in VRAM together:
#   python -m scripts.thinking_check --name q175 write    (the runtime writes; its distributions are saved)
#   python -m scripts.thinking_check --name q175 score    (the original reads; the comparison is printed and saved)
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from sclm import CKPT, CODES, DATA, MODEL

TOPK = 64
FAR = 2048  # from this position on a ratio-4 layer holds more compressed positions than its index keeps
QUESTION = "Rust로 간단하게 숫자 값을 양자화하는 코드 만들어줄래?"


def write(cli) -> None:
    from scripts.chat import Session
    from scripts.generate import load_runtime

    tok, cache, decoder, _ = load_runtime(cli)
    kept, logits_of = [], decoder.logits

    def recording(ids, pos):  # the distribution after every token fed, as its top entries, and what the token cost
        cache.stats.clear()
        t0 = time.perf_counter()
        out = logits_of(ids, pos)
        top = F.log_softmax(out[0].float(), dim=-1).topk(TOPK)
        kept.append((int(ids), top.indices.to(torch.int32).cpu(), top.values.cpu()))
        costs.append({"total_s": time.perf_counter() - t0, **cache.stats})
        return out

    costs = []

    decoder.logits = recording
    session = Session(tok, decoder, cli.max_seq_len)
    session.effort = cli.effort
    torch.manual_seed(cli.seed)
    text = []
    stats = session.ask(cli.question, cli.max_new, cli.temperature, cli.top_p, lambda kind, piece: text.append(piece))
    path = os.path.join(DATA, f"thinking_{cli.name}.pt")
    torch.save({"ids": [k[0] for k in kept], "top_ids": torch.stack([k[1] for k in kept]), "top_lp": torch.stack([k[2] for k in kept]),
                "prompt_tokens": stats["prompt_tokens"], "question": cli.question, "codes": cli.codes, "effort": cli.effort,
                "temperature": cli.temperature, "top_p": cli.top_p, "seed": cli.seed, "looped": stats["looped"]}, path)
    print("".join(text)[-600:])
    print(f"{stats['tokens']} tokens written ({stats['tokens'] / stats['seconds']:.1f} tok/s), looped: {stats['looped']} -> {path}")
    first = stats["prompt_tokens"] + 64  # the first tokens also fill the cache
    for label, part in ((f"positions below {FAR}", costs[first:FAR]), (f"positions from {FAR}", costs[FAR:])):
        if part:
            mean = {k: sum(c.get(k, 0.0) for c in part) / len(part) for k in sorted({k for c in part for k in c})}
            rest = mean["total_s"] - sum(mean.get(k, 0.0) for k in ("device_s", "ram_s", "read_s", "launch_s"))
            print(f"{label}: {len(part)} tokens, {1 / mean['total_s']:.1f} tok/s; per token: " + ", ".join(
                f"{k[:-2]} {1e3 * v:.1f} ms" if k.endswith("_s") else f"{k} {v:.1f}" for k, v in mean.items()) + f", rest {1e3 * rest:.1f} ms")


def score(cli) -> None:
    from transformers import AutoTokenizer

    from sclm.checkpoint import Checkpoint
    from sclm.layerwise import LayerwiseRunner, load_args, load_params_
    from sclm.upstream import model as m

    run = torch.load(os.path.join(DATA, f"thinking_{cli.name}.pt"), weights_only=True)
    ids = torch.tensor([run["ids"]])
    torch.set_default_dtype(torch.bfloat16)
    ck = Checkpoint(cli.ckpt)
    tok = AutoTokenizer.from_pretrained(cli.ckpt)
    runner = LayerwiseRunner(ck, load_args(max_batch_size=1, max_seq_len=max(1024, ids.size(1))), tok)
    with torch.inference_mode():
        x = runner.hidden(ids)
        norm = m.RMSNorm(runner.args.dim, runner.args.norm_eps).to(runner.device)
        load_params_(norm, {"weight": ck.get("norm.weight")}, "norm.")
        ref = F.log_softmax(F.linear(norm(x)[0].float(), ck.get("head.weight").to(runner.device).float()), dim=-1).cpu()
    # position i holds the distribution after token i; the positions from the end of the prompt on predicted a token
    # the runtime then wrote (the last one predicted the token after the reply, which was not fed)
    first = run["prompt_tokens"] - 1
    rows = slice(first, ids.size(1) - 1)
    ref, top_ids, top_lp, chosen = ref[rows], run["top_ids"][rows].long(), run["top_lp"][rows].float(), ids[0, first + 1:]
    ref_top = ref.topk(TOPK)
    # KL(original || runtime) over the original's top entries; a token the runtime's saved entries lack gets its floor
    floor = top_lp[:, -1:]
    at = (top_ids[:, None, :] == ref_top.indices[:, :, None])
    run_lp = torch.where(at.any(-1), (top_lp[:, None, :] * at).sum(-1), floor.expand(-1, TOPK))
    kl = (ref_top.values.exp() * (ref_top.values - run_lp)).sum(-1)
    agree = (top_ids[:, 0] == ref_top.indices[:, 0]).float()
    p_ref = ref.gather(1, chosen[:, None])[:, 0].exp()
    hit = top_ids == chosen[:, None]
    p_run = torch.where(hit.any(-1), (top_lp.exp() * hit).sum(-1), torch.zeros(len(chosen)))
    sure = (p_run > 0.3) & (p_ref < 0.02)  # the runtime was confident of a token the original finds very unlikely
    think_end = tok.convert_tokens_to_ids("</think>")
    cut = (chosen == think_end).nonzero()
    n_think = int(cut[0]) + 1 if len(cut) else len(chosen)
    out = {"name": cli.name, "codes": run["codes"], "tokens": len(chosen), "thinking_tokens": n_think, "looped": run["looped"]}
    far = max(0, FAR - (first + 1))  # the written token whose position is FAR
    for label, part in (("all", slice(None)), ("thinking", slice(0, n_think)), ("answer", slice(n_think, None)),
                        (f"positions_below_{FAR}", slice(0, far)), (f"positions_from_{FAR}", slice(far, None))):
        if kl[part].numel():
            out[label] = {"kl": float(kl[part].mean()), "top1_agree": float(agree[part].mean()),
                          "confident_where_original_below_2pct": int(sure[part].sum()),
                          "mean_logprob_of_written_token_under_original": float(p_ref[part].clamp_min(1e-12).log().mean())}
    worst = sorted(range(len(chosen)), key=lambda i: -float(p_run[i] - p_ref[i]))[:8]
    out["largest_gaps"] = [{"at": i, "token": tok.decode([int(chosen[i])]), "runtime_p": round(float(p_run[i]), 3),
                            "original_p": round(float(p_ref[i]), 4)} for i in worst]
    with open(os.path.join(DATA, f"thinking_{cli.name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["write", "score"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--codes", default=CODES, type=lambda v: None if v == "none" else v)
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--effort", default="high", help="the reply's reasoning effort (high: a long thinking pass)")
    ap.add_argument("--max-new", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=400)
    ap.add_argument("--ram-cache", type=int, default=-1)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    cli = ap.parse_args()
    # the runtime's other options, as the chat starts with them
    cli.lam, cli.threads, cli.min_ram_margin, cli.wait_share, cli.ram_share, cli.topk, cli.touch = 0.1, 8, 0.5, 0.0, 0.0, 0, 8
    cli.guess = 3 if MODEL == "v4" else 4
    (write if cli.phase == "write" else score)(cli)


if __name__ == "__main__":
    main()
