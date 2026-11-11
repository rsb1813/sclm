# Runs the upstream model one block at a time, so a checkpoint far larger than VRAM can do full-sequence forwards.
import gc
import json
import os
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn.functional as F

from .checkpoint import Checkpoint, MappedEngramEmbedding
from .upstream import CONFIG_PATH, MODEL, engram, model as m

V4 = MODEL == "v4"  # DeepSeek-V4-Flash-0731: no engram, no state handed from layer to layer, a hyper-connection head
QUANT_DTYPES = (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float4_e2m1fn_x2)
# cross-layer attention state upstream hands down the stack through its global `shared_attn`
SHARED_ATTN_FIELDS = ("compress_kv", "index_k", "topk_idxs", "candidates")
VRAM_MARGIN = 512 << 20  # left free for the driver and other processes


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
        if V4:
            m.scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
            m.scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        self.ckpt, self.args, self.device = ckpt, args, device
        if device.startswith("cuda"):
            # Windows (WDDM) pages over-subscribed VRAM out to system RAM instead of failing, so PyTorch's caching
            # allocator never hits an OOM, never trims its cache, and slows to a crawl. Cap it below free VRAM.
            index = torch.device(device).index
            index = torch.cuda.current_device() if index is None else index
            free, total = torch.cuda.mem_get_info(index)
            torch.cuda.set_per_process_memory_fraction((free - VRAM_MARGIN) / total, index)
        self.layout = self.hash = None
        if not V4:
            self.layout = engram.EngramLayout.from_args(args)
            with torch.device(device):
                self.hash = engram.NgramHashState(args, self.layout, tokenizer)

    def build_block(self, layer_id: int) -> "m.Block":
        prefix = f"layers.{layer_id}."
        if V4:
            with torch.device(self.device):
                block = m.Block(layer_id, self.args)
            load_params_(block, self.ckpt.module_state(prefix), prefix)
            return block
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
    def hidden(
        self,
        input_ids: torch.Tensor,
        n_layers: int | None = None,
        before_block: Callable[[int, "m.Block"], None] | None = None,
    ) -> torch.Tensor:
        """Hidden state [B, S, dim] after `n_layers` blocks (all by default), hc copies collapsed as upstream does.
        `before_block(layer_id, block)` runs after each block is loaded and before it executes (e.g. to add hooks)."""
        return self.hidden_batches([input_ids], n_layers, before_block)[0].to(self.device)

    @torch.inference_mode()
    def hidden_batches(
        self,
        batches: list[torch.Tensor],
        n_layers: int | None = None,
        before_block: Callable[[int, "m.Block"], None] | None = None,
        before_batch: Callable[[int, int], None] | None = None,
        checkpoint: str | None = None,
        every: int = 4,
    ) -> list[torch.Tensor]:
        """Like hidden() for each batch of input ids, returned on CPU, but every block is loaded once and run over
        all batches. `before_batch(layer_id, batch_index)` runs before each batch passes through a block.
        `checkpoint`: a file that receives the state between layers after every `every` of them; when it exists the
        run resumes from it, so the caller names it after everything the state depends on and removes it when done
        (hooks see only the remaining layers)."""
        # upstream creates index tensors without a device and relies on the default device (generate.py sets cuda)
        with torch.device(self.device):
            return self._hidden_batches(batches, n_layers, before_block, before_batch, checkpoint, every)

    def _hidden_batches(self, batches, n_layers, before_block, before_batch, checkpoint, every) -> list[torch.Tensor]:
        args = self.args
        n_layers = args.n_layers if n_layers is None else n_layers
        hashes = [None if V4 else self.hash(ids.to(self.device), 0) for ids in batches]
        if checkpoint and os.path.exists(checkpoint):
            start, hs, mixes, shared = torch.load(checkpoint, weights_only=True)
        else:
            embed = self.ckpt.get("embed.weight").to(self.device)
            start, hs, mixes, shared = 0, [], [], []
            for ids in batches:
                h = F.embedding(ids.to(self.device), embed).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
                mixes.append(None if V4 else m.make_identity_pre_mix(h, args.hc_mult))
                hs.append(h.cpu())
                shared.append({})
            del embed
        for i in range(start, n_layers):
            block = self.build_block(i)
            if before_block is not None:
                before_block(i, block)
            for j in range(len(batches)):
                for name, value in shared[j].items():
                    setattr(m.shared_attn, name, value.to(self.device))
                if before_batch is not None:
                    before_batch(i, j)
                h = hs[j].to(self.device)
                if V4:  # every block keeps its own attention state, and the token ids pick the hash layers' experts
                    hs[j] = block(h, 0, batches[j].to(self.device)).cpu()
                    continue
                if block.engram is not None:
                    h = block.engram(h, hashes[j][:, :, block.engram.layer_hash_index, :], None)
                h, mixes[j] = block(h, 0, mixes[j], None)
                hs[j] = h.cpu()
                # source layers publish their own persistent caches here, which the next batch overwrites: copy (to
                # the CPU, so hundreds of calibration batches do not pile up in VRAM)
                shared[j] = {n: v.to("cpu", copy=True) for n in SHARED_ATTN_FIELDS if (v := getattr(m.shared_attn, n)) is not None}
            del block
            gc.collect()
            torch.cuda.empty_cache()
            if checkpoint and (i + 1) % every == 0 and i + 1 < n_layers:
                torch.save((i + 1, hs, mixes, shared), checkpoint + ".tmp")
                os.replace(checkpoint + ".tmp", checkpoint)
        if V4:
            head = SimpleNamespace(norm_eps=args.norm_eps, hc_eps=args.hc_eps)  # what upstream's hc_head reads of its block
            fn, scale, base = (self.ckpt.get(f"hc_head_{k}").to(self.device) for k in ("fn", "scale", "base"))
            return [m.Block.hc_head(head, h.to(self.device), fn, scale, base).cpu() for h in hs]
        return [m.Block.hc_pre(None, h.to(self.device), mix).cpu() for h, mix in zip(hs, mixes)]

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
