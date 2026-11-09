# Perplexity of the reference model on a few fixed texts: the end-to-end correctness check for the PyTorch port.
import argparse
import math
import time

import torch
from transformers import AutoTokenizer

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
    parser.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    args_cli = parser.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args_cli.ckpt)
    runner = LayerwiseRunner(Checkpoint(args_cli.ckpt), load_args(max_batch_size=1, max_seq_len=1024), tok)
    for name, text in TEXTS.items():
        ids = torch.tensor([[tok.bos_token_id] + tok.encode(text, add_special_tokens=False)])
        t0 = time.time()
        nll = runner.token_nll(ids)[0].float()
        print(f"{name:5s} tokens={ids.size(1):4d} ppl={math.exp(nll.mean().item()):8.3f} "
              f"first-half={math.exp(nll[: len(nll) // 2].mean().item()):8.3f} "
              f"second-half={math.exp(nll[len(nll) // 2 :].mean().item()):8.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
