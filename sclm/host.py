# Token-by-token decode on one device: upstream's Transformer with every block resident and its routed experts on two
# tiers (sclm.tiered). PyTorch hosts the always-on path for now; the engine core computes the RAM tier and reads the SSD tier.
import os
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from sclm import core, gpu, ssd
from sclm.checkpoint import Checkpoint, widen_fp8
from sclm.layerwise import load_params_
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


class Fp8Weight(nn.Module):
    """attn.wo_a as the checkpoint stores it, fp8 with block scales. upstream widens it to bf16 once at load (twice
    the memory); here `weight` widens it at each use."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.fp8, self.scale = weight, scale

    @property
    def weight(self) -> torch.Tensor:
        return widen_fp8(self.fp8, self.scale)


_upstream_linear = m.linear


def kernel_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """upstream linear(), with an fp8 weight applied to a single token by the engine's kernel."""
    if weight.dtype == torch.float8_e4m3fn and x.numel() == weight.size(1) and x.dtype == torch.bfloat16:
        return gpu.fp8_linear(x.contiguous(), weight, weight.scale).view(*x.shape[:-1], -1)
    return _upstream_linear(x, weight, bias)


@lru_cache(4)
def attended(win: int, topk: int, device: torch.device) -> torch.Tensor:
    """[win, win + topk] int32. The start of row p lists what a decode token at position p attends to while nothing
    has to be chosen (the last row: any later position): the window slots filled so far, -1 for the others, then the
    compressed positions in order."""
    slots = torch.arange(win, device=device, dtype=torch.int32)
    ring = torch.where(slots[None] > slots[:, None], -1, slots[None])
    more = (win + torch.arange(topk, device=device, dtype=torch.int32)).expand(win, -1)
    return torch.cat([ring, more], dim=1).contiguous()


class HostAttention(m.Attention):
    """upstream Attention; decoding one token with the kernels on, the engine computes the query, the token's
    key/value, the attention itself and the output projection (wo_a read as stored, see Fp8Weight). While every
    compressed position still fits in index_topk it also makes the compressed KV and its index keys, and nothing is
    scored: the top-k of fewer than k positions is all of them. Past that, the compressed KV and its index are
    upstream's code."""

    def __init__(self, layer_id: int, args: "m.ModelArgs"):
        super().__init__(layer_id, args)
        self.index_topk = args.index_topk

    def forward(self, x: torch.Tensor, start_pos: int):
        if not (gpu.active and x.size(0) * x.size(1) == 1 and start_pos):  # position 0 is upstream's prefill case
            return super().forward(x, start_pos)
        rd, win, heads, ratio = self.rope_head_dim, self.window_size, self.n_local_heads, self.compress_ratio
        freqs = self.freqs_cis.data_ptr() + start_pos * rd * 4  # the position's row: rd / 2 complex64
        token = x.contiguous()
        qr, q = gpu.attn_q(self.wq_a, self.q_norm.weight, self.wq_b, token, freqs, heads, self.head_dim, rd, self.eps)
        gpu.attn_kv(self.wkv, self.kv_norm.weight, token, freqs, self.window_kv_cache[0, start_pos % win], rd, self.eps)
        count = (start_pos + 1) // ratio if ratio else 0
        if count <= self.index_topk:
            if self.is_kv_source:
                gpu.compress(self.compressor, self.indexer, token.view(-1), start_pos, self.freqs_cis,
                             self.compress_kv_cache[0], rd)
                m.shared_attn.compress_kv = self.compress_kv_cache
                if self.indexer is not None and self.indexer.owns_k:
                    m.shared_attn.index_k = self.indexer.k_cache
            compressed = m.shared_attn.compress_kv if ratio else self.window_kv_cache
            idxs = attended(win, self.index_topk, x.device)[min(start_pos, win - 1), :win + count]
        else:
            with torch.device(x.device):  # upstream creates index tensors on the default device
                compressed, compress_idxs = self._compress_kv(x, qr.view(1, 1, -1), start_pos, win)
                idxs = torch.cat([m.get_window_topk_idxs(win, 1, 1, start_pos).view(-1), compress_idxs.view(-1)])
        o = gpu.sparse_attn(q, self.window_kv_cache, compressed, idxs, self.attn_sink, heads, win, self.softmax_scale)
        out = gpu.attn_out(o, freqs, self.wo_a.fp8, self.wo_a.scale, self.wo_b, heads, rd, self.n_local_groups)
        return out.view(1, 1, -1)


class HostBlock(m.Block):
    """upstream Block; decoding one token with the kernels on, the hyper-connection steps and the norm in front of
    each sublayer run as fused kernels (upstream spends about a hundred small operations on them per sublayer)."""

    attention_cls = HostAttention

    def forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
        if not (gpu.active and x.size(0) * x.size(1) == 1):
            return super().forward(x, start_pos, pre_mix, image_mask, *attn_args)
        hc, shape = self.hc_mult, x.shape
        mixes = lambda h, part: gpu.hc_mixes(  # noqa: E731  -> pre[hc], post[hc], comb[hc][hc]
            h, getattr(self, f"hc_{part}_fn"), getattr(self, f"hc_{part}_scale"), getattr(self, f"hc_{part}_base"), hc,
            self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters)
        h = x.contiguous()
        attn_mix = mixes(h, "attn")
        a = gpu.hc_pre_norm(h, pre_mix.contiguous(), self.attn_norm.weight, self.attn_norm.eps)
        a = self.attn(a.view(1, 1, -1), start_pos, *attn_args)
        h = gpu.hc_post(a.contiguous(), h, attn_mix[hc:2 * hc], attn_mix[2 * hc:])
        ffn_mix = mixes(h, "ffn")
        f = gpu.hc_pre_norm(h, attn_mix[:hc], self.ffn_norm.weight, self.ffn_norm.eps)  # the mix attention computed
        f = self.ffn(f.view(1, 1, -1), image_mask)
        h = gpu.hc_post(f.contiguous(), h, ffn_mix[hc:2 * hc], ffn_mix[2 * hc:])
        return h.view(shape), ffn_mix[:hc].view(1, 1, hc)


class _Absent(nn.Module):
    """Stands in, while the model is built, for a module whose weights are placed afterwards."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def build_net(args: "m.ModelArgs", tokenizer=None, device: str = "cuda") -> "m.Transformer":
    """upstream's Transformer for `args` (with TEXT_ONLY applied) without what is placed afterwards or stays off the
    device: the routed experts (each block's ffn is a TieredMoE), the engram tables, the embedding and the head.
    Parameters are uninitialized."""
    assert not args.dspark_block_size and not args.vision_enabled
    names = ("Block", "MoE", "ParallelEngramEmbedding", "ParallelEmbedding", "ParallelHead")
    upstream = [getattr(m, n) for n in names]
    m.Block, m.MoE = HostBlock, TieredMoE
    m.ParallelEngramEmbedding = m.ParallelEmbedding = m.ParallelHead = _Absent
    try:
        with torch.device(device):
            return m.Transformer(args, tokenizer)
    finally:
        for n, cls in zip(names, upstream):
            setattr(m, n, cls)


def load_checkpoint_(net: "m.Transformer", ckpt: Checkpoint, device: str = "cuda") -> None:
    """Fills build_net's model from the released checkpoint. The engram tables stay in the checkpoint files, the
    embedding table in RAM, the head in bf16 and attn.wo_a in fp8."""
    net.embed = HostEmbedding(ckpt.get("embed.weight"))
    for i, block in enumerate(net.layers):
        prefix = f"layers.{i}."
        block.attn.wo_a = Fp8Weight(*(ckpt.get(f"{prefix}attn.wo_a.{part}").to(device) for part in ("weight", "scale")))
        # bias_vl routes image tokens only
        skip = (".engram.embed.", ".ffn.experts.", ".ffn.gate.bias_vl", ".attn.wo_a.")
        load_params_(block, ckpt.module_state(prefix, skip=skip), prefix)
        if block.engram is not None:
            block.engram.embed = ssd.EngramRows(ckpt, prefix + "engram.embed.")
    load_params_(net.norm, {"weight": ckpt.get("norm.weight")}, "norm.")
    net.head = Bf16Head(ckpt.get("head.weight").to(device))


def attach_tiers_(net: "m.Transformer", args: "m.ModelArgs", ckpt: Checkpoint, codes_dir: str, slots: int, lam: float,
                  device: str = "cuda", background: int = 0) -> ssd.ExpertCache:
    """Gives every block its coded experts (codes_dir/layer_NN.safetensors, held in RAM) and the shared cache of
    original experts, read from `ckpt` on demand."""
    cache = ssd.ExpertCache(ssd.ExpertReader(ckpt), args.n_layers, args.n_routed_experts, args.dim, args.moe_inter_dim,
                            args.swiglu_limit, slots, lam, device, scratch=args.n_activated_experts,
                            background=background)
    for i, block in enumerate(net.layers):
        path = os.path.join(codes_dir, f"layer_{i:02d}.safetensors")
        block.ffn.ram = core.Layer(path, args.dim, args.moe_inter_dim, args.swiglu_limit)
        block.ffn.cache = cache
    return cache


class Host:
    """The tiered model as a decoder: prefill with start_pos 0, then one token per call. kernels: single-token
    steps use the engine's GPU kernels (by default when the model is on a GPU and they are built)."""

    def __init__(self, net: "m.Transformer", cache: ssd.ExpertCache, device: str = "cuda", kernels: bool | None = None):
        self.net, self.cache, self.device = net, cache, device
        self.origin = self.end = 0  # the cache's clock at the start of this sequence and after its last token
        on_gpu = torch.device(device).type == "cuda"
        self.kernels = on_gpu and os.path.exists(gpu.DLL) if kernels is None else kernels

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Next-token logits [batch, vocab] (fp32) after input_ids, which continue the sequence at start_pos."""
        if start_pos == 0:  # a new sequence: the cache's clock goes on, its experts are still there
            self.origin = self.end
        self.cache.token = self.origin + start_pos
        self.end = self.cache.token + input_ids.size(-1)
        gpu.active, m.linear = self.kernels, kernel_linear if self.kernels else _upstream_linear
        try:
            if self.kernels and start_pos and input_ids.numel() == 1:
                # the default-device context costs every small tensor call 0.6 us, ten thousand times a token;
                # the kernel path enters it only around upstream's index code (HostAttention)
                return self.net(input_ids.to(self.device), start_pos)[1]
            with torch.device(self.device):  # upstream creates index tensors on the default device
                return self.net(input_ids.to(self.device), start_pos)[1]
        finally:
            gpu.active, m.linear = False, _upstream_linear
