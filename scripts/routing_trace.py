# Records MoE routing over the calibration set and scores two expert predictors usable for prefetching.
import argparse
import json
import os
import time
import weakref

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from sclm import CKPT, DATA
from sclm.checkpoint import Checkpoint
from sclm.layerwise import LayerwiseRunner, load_args

DEVICE = "cuda"
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

    def __init__(self, n_layers: int, n_batches: int):
        self.hits = {p: np.zeros((n_layers, len(RECALL_K))) for p in "AB"}
        self.tokens = np.zeros(n_layers)
        self.prev_raw = [None] * n_batches  # per batch, previous layer's pre-norm FFN input (CPU)
        self.indices = [[None] * n_layers for _ in range(n_batches)]
        self.weights = [[None] * n_layers for _ in range(n_batches)]
        self.layer = self.block = self.batch = None
        self.pred = {}

    def before_block(self, i: int, block) -> None:
        # weak: a strong reference would keep this ~7.4GB block alive while the runner builds the next one
        self.layer, self.block = i, weakref.ref(block)

        def on_attn_norm(_mod, inp, _out):
            self.pred["B"] = predict(block, inp[0])

        def on_ffn_norm(_mod, inp, _out):
            self.prev_raw[self.batch] = inp[0].cpu()

        def on_gate(_mod, _inp, out):
            weights, idx = out
            self.tokens[i] += idx.size(0)
            for p, pk in self.pred.items():
                if pk is None:
                    continue
                for j, k in enumerate(RECALL_K):
                    self.hits[p][i, j] += (pk[:, :k, None] == idx[:, None, :]).any(1).sum().item()
            self.indices[self.batch][i] = idx.to(torch.int16).cpu()
            self.weights[self.batch][i] = weights.to(torch.float16).cpu()

        block.attn_norm.register_forward_hook(on_attn_norm)
        block.ffn_norm.register_forward_hook(on_ffn_norm)
        block.ffn.gate.register_forward_hook(on_gate)

    def before_batch(self, i: int, j: int) -> None:
        self.batch = j
        raw = self.prev_raw[j]
        self.pred = {"A": predict(self.block(), raw.to(DEVICE)) if raw is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--batch", type=int, default=2, help="sequences per forward call (VRAM bound)")
    cli = parser.parse_args()

    calib = np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))
    tokens = torch.from_numpy(calib["tokens"]).long()
    torch.set_default_dtype(torch.bfloat16)
    args = load_args(max_batch_size=cli.batch, max_seq_len=tokens.size(1))
    runner = LayerwiseRunner(Checkpoint(cli.ckpt), args, AutoTokenizer.from_pretrained(cli.ckpt))
    batches = list(tokens.split(cli.batch))
    col = Collector(args.n_layers, len(batches))
    t_start, t_layer = time.time(), [time.time()]

    def before_block(i, block):
        if i:
            print(f"layer {i - 1:2d} done {time.time() - t_layer[0]:5.0f}s "
                  f"vram_peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB "
                  f"reserved={torch.cuda.memory_reserved() / 2**30:.1f}GB", flush=True)
        t_layer[0] = time.time()
        torch.cuda.reset_peak_memory_stats()
        col.before_block(i, block)

    runner.hidden_batches(batches, before_block=before_block, before_batch=col.before_batch)

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
