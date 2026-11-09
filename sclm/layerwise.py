# Runs the upstream model one block at a time, so a checkpoint far larger than VRAM can do full-sequence forwards.
import gc
import json

import torch
import torch.nn.functional as F

from .checkpoint import Checkpoint, MappedEngramEmbedding
from .upstream import CONFIG_PATH, engram, model as m

QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2)


class _NoTable(torch.nn.Module):
    """Stands in for ParallelEngramEmbedding while a block is built, so the ~100GB table is never allocated."""

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()


def load_args(**overrides) -> "m.ModelArgs":
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    cfg.update(overrides)
    return m.ModelArgs(**cfg)


def load_params_(module: torch.nn.Module, state: dict[str, torch.Tensor], where: str) -> None:
    """Copy `state` into `module`'s parameters, requiring exactly matching names, shapes and quantized dtypes."""
    params = dict(module.named_parameters())
    missing, unexpected = sorted(set(params) - set(state)), sorted(set(state) - set(params))
    assert not missing and not unexpected, (where, missing[:5], unexpected[:5])
    with torch.no_grad():
        for name, p in params.items():
            t = state[name]
            assert t.shape == p.shape, (where + name, t.shape, p.shape)
            if p.dtype in QUANT_DTYPES:
                assert t.dtype == p.dtype, (where + name, t.dtype, p.dtype)
                p.view(torch.uint8).copy_(t.view(torch.uint8))
            else:
                p.copy_(t)


class LayerwiseRunner:
    """start_pos=0, text-only forwards of the backbone, building and freeing one Block at a time.
    Mirrors upstream Transformer.forward; cross-layer attention state flows through upstream's shared_attn."""

    def __init__(self, ckpt: Checkpoint, args: "m.ModelArgs", tokenizer, device: str = "cuda"):
        # upstream sets these globals in Transformer.__init__, which is never built here
        m.world_size, m.rank = 1, 0
        m.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        self.ckpt, self.args, self.device = ckpt, args, device
        self.layout = engram.EngramLayout.from_args(args)
        with torch.device(device):
            self.hash = engram.NgramHashState(args, self.layout, tokenizer)

    def build_block(self, layer_id: int) -> "m.Block":
        prefix = f"layers.{layer_id}."
        upstream_table = m.ParallelEngramEmbedding
        m.ParallelEngramEmbedding = _NoTable
        try:
            with torch.device(self.device):
                block = m.Block(layer_id, self.args, self.layout)
        finally:
            m.ParallelEngramEmbedding = upstream_table
        load_params_(block, self.ckpt.module_state(prefix, skip=(".engram.embed.",)), prefix)
        if block.engram is not None:
            block.engram.embed = MappedEngramEmbedding(
                self.ckpt.memmap(prefix + "engram.embed.weight"), self.ckpt.memmap(prefix + "engram.embed.scale")
            )
        return block

    @torch.inference_mode()
    def hidden(self, input_ids: torch.Tensor, n_layers: int | None = None) -> torch.Tensor:
        """Hidden state [B, S, dim] after `n_layers` blocks (all by default), hc copies collapsed as upstream does."""
        # upstream creates index tensors without a device and relies on the default device (generate.py sets cuda)
        with torch.device(self.device):
            return self._hidden(input_ids, n_layers)

    def _hidden(self, input_ids: torch.Tensor, n_layers: int | None) -> torch.Tensor:
        args = self.args
        ids = input_ids.to(self.device)
        hashes = self.hash(ids, 0)
        h = F.embedding(ids, self.ckpt.get("embed.weight").to(self.device))
        h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        pre_mix = m.make_identity_pre_mix(h, args.hc_mult)
        for i in range(args.n_layers if n_layers is None else n_layers):
            block = self.build_block(i)
            if block.engram is not None:
                h = block.engram(h, hashes[:, :, block.engram.layer_hash_index, :], None)
            h, pre_mix = block(h, 0, pre_mix, None)
            del block
            gc.collect()
            torch.cuda.empty_cache()
        return m.Block.hc_pre(None, h, pre_mix)

    @torch.inference_mode()
    def token_nll(self, input_ids: torch.Tensor, chunk: int = 512) -> torch.Tensor:
        """Per-position negative log-likelihood [B, S-1] of input_ids[:, 1:] under the full model."""
        x = self.hidden(input_ids)
        norm = m.RMSNorm(self.args.dim, self.args.norm_eps).to(self.device)
        load_params_(norm, {"weight": self.ckpt.get("norm.weight")}, "norm.")
        x = norm(x)
        head = self.ckpt.get("head.weight").to(self.device).float()  # upstream keeps the head in fp32
        targets = input_ids.to(self.device)[:, 1:]
        out = []
        for s0 in range(0, targets.size(1), chunk):
            logits = F.linear(x[:, s0 : s0 + chunk].float(), head)[:, : targets.size(1) - s0]
            out.append(F.cross_entropy(logits.transpose(1, 2), targets[:, s0 : s0 + chunk], reduction="none"))
        return torch.cat(out, dim=1)
