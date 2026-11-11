# Collects per-layer PCA bases of the FFN input and of the routed-expert output over calibration tokens (for subspace projection).
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from sclm import CKPT, DATA
from sclm.checkpoint import Checkpoint
from sclm.layerwise import LayerwiseRunner, load_args

KEEP = 1024  # eigenvectors kept per basis


def calib_indices(n_seq: int, count: int) -> list[int]:
    """Spread over categories and disjoint from the eval set tokens[3::6]."""
    return [i for i in range(n_seq) if i % 6 != 3][::3][:count]


class CovCollector:
    def __init__(self, dim: int):
        self.dim = dim
        self.h_in = self.c_out = self.shared = None

    def before_block(self, i: int, block) -> None:
        self.h_in = torch.zeros(self.dim, self.dim, device="cuda", dtype=torch.float32)
        self.c_out = torch.zeros(self.dim, self.dim, device="cuda", dtype=torch.float32)

        def on_gate(_mod, inp, _out):
            x = inp[0].float()
            self.h_in += x.t() @ x

        def on_shared(_mod, _inp, out):
            self.shared = out.float().reshape(-1, self.dim)

        def on_moe(_mod, _inp, out):
            # MoE returns (routed + shared) in bf16; the routed part is what w2 projection must preserve
            y = out.float().reshape(-1, self.dim) - self.shared
            self.c_out += y.t() @ y

        block.ffn.gate.register_forward_hook(on_gate)
        block.ffn.shared_experts.register_forward_hook(on_shared)
        block.ffn.register_forward_hook(on_moe)

    def bases(self) -> dict:
        out = {}
        for name, mat in (("in", self.h_in), ("out", self.c_out)):
            evals, evecs = torch.linalg.eigh(mat)
            out[name + "_evals"] = evals.flip(0).cpu()
            out[name + "_basis"] = evecs[:, -KEEP:].flip(1).contiguous().cpu()  # columns by descending eigenvalue
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--seqs", type=int, default=16)
    cli = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)
    out_dir = os.path.join(DATA, "bases_v1")
    os.makedirs(out_dir, exist_ok=True)

    tokens = torch.from_numpy(np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["tokens"]).long()
    idx = calib_indices(tokens.size(0), cli.seqs)
    runner = LayerwiseRunner(Checkpoint(cli.ckpt), load_args(max_batch_size=2, max_seq_len=tokens.size(1)),
                             AutoTokenizer.from_pretrained(cli.ckpt))
    col = CovCollector(runner.args.dim)
    t0, prev = time.time(), [None]

    def before_block(i, block):
        if prev[0] is not None:  # the previous layer has seen every batch: finalize its bases
            torch.save(col.bases(), os.path.join(out_dir, f"layer{prev[0]}.pt"))
            print(f"layer {prev[0]:2d} saved {time.time() - t0:6.0f}s", flush=True)
        prev[0] = i
        col.before_block(i, block)

    runner.hidden_batches(list(tokens[idx].split(2)), before_block=before_block)
    torch.save(col.bases(), os.path.join(out_dir, f"layer{prev[0]}.pt"))
    torch.save({"calib_indices": idx}, os.path.join(out_dir, "meta.pt"))
    print(f"done in {(time.time() - t0) / 60:.1f} min, calib seqs {idx}", flush=True)


if __name__ == "__main__":
    main()
