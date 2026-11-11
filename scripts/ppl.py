# Perplexity of the reference model on a few fixed texts: the end-to-end correctness check for the PyTorch port.
import argparse
import math
import time

import torch
from transformers import AutoTokenizer

from sclm import CKPT
from sclm.checkpoint import Checkpoint
from sclm.layerwise import LayerwiseRunner, load_args

TEXTS = {
    "en": (
        "The Industrial Revolution began in Great Britain in the late eighteenth century and spread to Europe and "
        "North America over the following decades. It marked a shift from hand production to machines, new chemical "
        "manufacturing and iron production processes, the increasing use of steam power and water power, and the rise "
        "of the mechanized factory system. Output greatly increased, and the result was an unprecedented rise in "
        "population and in the rate of population growth."
    ),
    "ko": (
        "조선은 1392년 태조 이성계가 건국하여 1897년까지 한반도를 통치한 왕조이다. 수도는 한양이었으며, 유교를 통치 "
        "이념으로 삼았다. 세종 대에는 훈민정음이 창제되어 백성들이 쉽게 글을 읽고 쓸 수 있게 되었고, 과학 기술과 "
        "농업, 음악 등 여러 분야에서 큰 발전이 이루어졌다."
    ),
    "code": (
        "def binary_search(items, target):\n"
        "    lo, hi = 0, len(items) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if items[mid] == target:\n"
        "            return mid\n"
        "        if items[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--file", help="score this text file instead of the built-in samples")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--no-compressed",
        action="store_true",
        help="ablation: drop every compressed-KV position so attention sees only the sliding window",
    )
    args_cli = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args_cli.ckpt)
    texts = TEXTS
    if args_cli.file:
        with open(args_cli.file, encoding="utf-8") as f:
            texts = {"file": f.read()}
    encoded = {k: [tok.bos_token_id] + tok.encode(v, add_special_tokens=False)[: args_cli.max_tokens - 1] for k, v in texts.items()}
    max_len = max(1024, max(len(v) for v in encoded.values()))
    runner = LayerwiseRunner(Checkpoint(args_cli.ckpt), load_args(max_batch_size=1, max_seq_len=max_len), tok)
    if args_cli.no_compressed:
        from sclm.upstream import model as m

        compress_kv = m.Attention._compress_kv

        def window_only(self, *a):
            kv, idxs = compress_kv(self, *a)
            return kv, torch.full_like(idxs, -1)

        m.Attention._compress_kv = window_only
    for name, ids in encoded.items():
        ids = torch.tensor([ids])
        t0 = time.time()
        nll = runner.token_nll(ids)[0].float()
        ppl = lambda x: math.exp(x.mean().item()) if len(x) else float("nan")
        print(f"{name:5s} tokens={ids.size(1):4d} ppl={ppl(nll):8.3f} "
              f"first-half={ppl(nll[: len(nll) // 2]):8.3f} second-half={ppl(nll[len(nll) // 2 :]):8.3f} "
              f"pos<128={ppl(nll[:127]):8.3f} pos>=256={ppl(nll[255:]):8.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
