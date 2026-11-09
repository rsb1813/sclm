# Records MoE routing over the calibration set and scores two expert predictors usable for prefetching.
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from sclm.checkpoint import Checkpoint
from sclm.layerwise import LayerwiseRunner, load_args

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RECALL_K = (6, 12, 24)


def predict(block, raw: torch.Tensor) -> torch.Tensor:
    """Top-24 experts layer `block` would pick if its FFN input were ffn_norm(raw); [n, 24]."""
    gate = block.ffn.gate
    assert gate.score_func == "sqrtsoftplus"
    x = block.ffn_norm(raw).flatten(0, -2)
    scores = F.softplus(F.linear(x.float(), gate.weight.float())).sqrt() + gate.bias
    return scores.topk(max(RECALL_K), dim=-1).indices


class Collector:
    """Hooks one block at a time. Predictor A: the previous layer's pre-norm FFN input, available a whole layer
    early. Predictor B: this layer's pre-norm attention input, available while attention runs."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.hits = {p: np.zeros((n_layers, len(RECALL_K))) for p in "AB"}
        self.tokens = np.zeros(n_layers)
        self.prev_raw = None
        self.indices, self.weights = [], []

    def start_batch(self):
        self.prev_raw = None
        self.indices.append([None] * self.n_layers)
        self.weights.append([None] * self.n_layers)

    def before_block(self, i: int, block) -> None:
        pred = {"A": predict(block, self.prev_raw) if self.prev_raw is not None else None}

        def on_attn_norm(_mod, inp, _out):
            pred["B"] = predict(block, inp[0])

        def on_ffn_norm(_mod, inp, _out):
            self.prev_raw = inp[0]

        def on_gate(_mod, _inp, out):
            weights, idx = out
            self.tokens[i] += idx.size(0)
            for p, pk in pred.items():
                if pk is None:
                    continue
                for j, k in enumerate(RECALL_K):
                    self.hits[p][i, j] += (pk[:, :k, None] == idx[:, None, :]).any(1).sum().item()
            self.indices[-1][i] = idx.to(torch.int16).cpu()
            self.weights[-1][i] = weights.to(torch.float16).cpu()

        block.attn_norm.register_forward_hook(on_attn_norm)
        block.ffn_norm.register_forward_hook(on_ffn_norm)
        block.ffn.gate.register_forward_hook(on_gate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    parser.add_argument("--batch", type=int, default=4)
    cli = parser.parse_args()

    calib = np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))
    tokens = torch.from_numpy(calib["tokens"]).long()
    torch.set_default_dtype(torch.bfloat16)
    args = load_args(max_batch_size=cli.batch, max_seq_len=tokens.size(1))
    runner = LayerwiseRunner(Checkpoint(cli.ckpt), args, AutoTokenizer.from_pretrained(cli.ckpt))
    col = Collector(args.n_layers)
    t_start = time.time()
    for b0 in range(0, tokens.size(0), cli.batch):
        t0 = time.time()
        col.start_batch()
        runner.hidden(tokens[b0 : b0 + cli.batch], before_block=col.before_block)
        print(f"batch {b0 // cli.batch + 1}/{-(-tokens.size(0) // cli.batch)} {time.time() - t0:.0f}s "
              f"vram_peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)

    # [n_tokens, n_layers, k], tokens in sequence-major order
    idx = np.concatenate([np.stack([l.numpy() for l in batch], axis=1) for batch in col.indices])
    w = np.concatenate([np.stack([l.numpy() for l in batch], axis=1) for batch in col.weights])
    np.savez(os.path.join(DATA, "routing_v1.npz"), indices=idx, weights=w, category=calib["category"],
             names=calib["names"], seq_len=tokens.size(1))
    recall = {p: (col.hits[p] / (col.tokens[:, None] * args.n_activated_experts)).tolist() for p in "AB"}
    with open(os.path.join(DATA, "routing_v1_predictors.json"), "w") as f:
        json.dump({"recall_k": RECALL_K, "recall": recall, "note": "A has no prediction for layer 0"}, f, indent=1)
    print(f"done in {(time.time() - t_start) / 60:.1f} min, trace {idx.shape}")


if __name__ == "__main__":
    main()
