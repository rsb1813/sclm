# Token-by-token decode on one device: upstream's Transformer with every block resident and its routed experts on two
# tiers (sclm.tiered). PyTorch hosts the always-on path for now; the engine core computes the RAM tier and reads the SSD tier.
import os

import torch
import torch.nn.functional as F
from torch import nn

from sclm import core, ssd
from sclm.checkpoint import Checkpoint, MappedEngramEmbedding
from sclm.layerwise import _NoTable, load_params_
from sclm.tiered import TieredMoE
from sclm.upstream import model as m

import kernel  # noqa: E402  (sclm.upstream puts it on the path)

# config overrides of the text-only decode model: no draft (MTP) blocks, no vision tower
TEXT_ONLY = {"dspark_block_size": 0, "dspark_target_layer_ids": (), "vision_n_layers": 0}


class HostEmbedding(nn.Module):
    """The token embedding table left in RAM: a lookup of a few rows needs no device memory."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight.cpu()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x.cpu(), self.weight).to(x.device)


class Bf16Head(nn.Module):
    """upstream's head with the weight kept in bf16 as the checkpoint stores it (upstream widens it to fp32, twice
    the memory). The logits still accumulate in fp32."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor, full_logits: bool = False) -> torch.Tensor:
        if not full_logits:
            x = x[:, -1]
        return kernel._mm_fp32_out(x.to(torch.bfloat16), self.weight)


def build_net(args: "m.ModelArgs", tokenizer=None, device: str = "cuda") -> "m.Transformer":
    """upstream's Transformer for `args` (with TEXT_ONLY applied) without what stays off the device: the routed
    experts (each block's ffn is a TieredMoE) and the engram tables. Parameters are uninitialized."""
    assert not args.dspark_block_size and not args.vision_enabled
    upstream = m.MoE, m.ParallelEngramEmbedding
    m.MoE, m.ParallelEngramEmbedding = TieredMoE, _NoTable
    try:
        with torch.device(device):
            return m.Transformer(args, tokenizer)
    finally:
        m.MoE, m.ParallelEngramEmbedding = upstream


def load_checkpoint_(net: "m.Transformer", ckpt: Checkpoint, device: str = "cuda") -> None:
    """Fills build_net's model from the released checkpoint. The engram tables stay in the checkpoint files, the
    embedding table in RAM, the head in bf16."""
    net.embed = HostEmbedding(ckpt.get("embed.weight"))
    for i, block in enumerate(net.layers):
        prefix = f"layers.{i}."
        # bias_vl routes image tokens only
        load_params_(block, ckpt.module_state(prefix, skip=(".engram.embed.", ".ffn.experts.", ".ffn.gate.bias_vl")), prefix)
        if block.engram is not None:
            block.engram.embed = MappedEngramEmbedding(
                ckpt.memmap(prefix + "engram.embed.weight"), ckpt.memmap(prefix + "engram.embed.scale")
            )
    load_params_(net.norm, {"weight": ckpt.get("norm.weight")}, "norm.")
    net.head = Bf16Head(ckpt.get("head.weight").to(device))


def attach_tiers_(net: "m.Transformer", args: "m.ModelArgs", ckpt: Checkpoint, codes_dir: str, slots: int, lam: float,
                  device: str = "cuda") -> ssd.ExpertCache:
    """Gives every block its coded experts (codes_dir/layer_NN.safetensors, held in RAM) and the shared cache of
    original experts, read from `ckpt` on demand."""
    cache = ssd.ExpertCache(ssd.ExpertReader(ckpt), args.n_layers, args.n_routed_experts, args.dim, args.moe_inter_dim,
                            args.swiglu_limit, slots, lam, device, scratch=args.n_activated_experts)
    for i, block in enumerate(net.layers):
        path = os.path.join(codes_dir, f"layer_{i:02d}.safetensors")
        block.ffn.ram = core.Layer(path, args.dim, args.moe_inter_dim, args.swiglu_limit)
        block.ffn.cache = cache
    return cache


class Host:
    """The tiered model as a decoder: prefill with start_pos 0, then one token per call."""

    def __init__(self, net: "m.Transformer", cache: ssd.ExpertCache, device: str = "cuda"):
        self.net, self.cache, self.device = net, cache, device

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Next-token logits [batch, vocab] (fp32) after input_ids, which continue the sequence at start_pos."""
        self.cache.token = start_pos
        with torch.device(self.device):  # upstream creates index tensors on the default device
            return self.net(input_ids.to(self.device), start_pos)[1]
