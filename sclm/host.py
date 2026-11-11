# Token-by-token decode on one device: upstream's Transformer with every block resident and its routed experts on two
# tiers (sclm.tiered). PyTorch hosts the always-on path for now; the engine core computes the RAM tier and reads the SSD tier.
import os
from functools import lru_cache

import numpy as np
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


class HostHash:
    """upstream NgramHashState (the engram rows of each position) for text tokens on the host, a token at a time, on
    numpy integers: the same row numbers, without the thirty small tensor operations upstream spends on one token.
    It has to be given every token of the sequence (put, or rows)."""

    def __init__(self, state):
        self.token_map, self.pad = state.token_map.cpu().numpy(), int(state.pad_id)
        self.multipliers = state.multipliers.cpu().numpy()  # [engram layers, n-gram sizes]
        self.primes, self.offsets = state.primes.cpu().numpy(), state.offsets.cpu().numpy()
        self.seen = np.empty(state.cache.size(1), np.int64)  # the sequence's tokens in the compressed vocabulary
        assert self.multipliers.dtype == self.primes.dtype == self.offsets.dtype == np.int64

    def put(self, input_ids, start_pos: int) -> None:
        """The tokens of positions start_pos .. (any tensor or list of ids)."""
        ids = np.asarray(input_ids, np.int64).reshape(-1)
        self.seen[start_pos:start_pos + len(ids)] = self.token_map[ids]

    def rows(self, token: int, pos: int) -> list:
        """[engram layer][column] row numbers of the token at position pos."""
        self.seen[pos] = self.token_map[token]
        n = self.multipliers.shape[1]
        tokens = np.full(n, self.pad, np.int64)  # the token and the ones before it; padding before the sequence
        back = min(n, pos + 1)
        tokens[:back] = self.seen[pos - back + 1:pos + 1][::-1]
        # the running XOR of the multiplied ids after i look-backs is the hash of the (i + 1)-gram, and each lands in
        # the bucket range of its (n-gram size, head)
        rolling = np.bitwise_xor.accumulate(tokens * self.multipliers, axis=1)
        return ((rolling[:, 1:, None] % self.primes).reshape(len(rolling), -1) + self.offsets).tolist()


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


@lru_cache(4)
def picked(win: int, topk: int, device: torch.device) -> torch.Tensor:
    """int32 [win + topk]: what a decode token attends to once the index chooses: the whole window ring, then the
    compressed positions the last index source chose (gpu.index_pick), which the layers after it share as upstream's
    do."""
    return torch.arange(win + topk, device=device, dtype=torch.int32)


class HostAttention(m.Attention):
    """upstream Attention; decoding one token with the kernels on, the engine computes the query, the token's
    key/value, the attention itself and the output projection (wo_a read as stored, see Fp8Weight), the compressed
    KV with its index keys, and the index's choice. While every compressed position still fits in index_topk nothing
    is scored: the top-k of fewer than k positions is all of them. Upstream's code is left with the positions past
    pick_limit, where its candidate blocks start to exclude some."""

    def __init__(self, layer_id: int, args: "m.ModelArgs"):
        super().__init__(layer_id, args)
        self.index_topk = args.index_topk
        # the most compressed positions the kernel chooses among: every candidate block is still kept
        self.pick_limit = args.candidate_topk_blocks * args.candidate_block_size if args.candidate_source_layer >= 0 else 1 << 16
        if args.index_n_heads > 32 or args.index_head_dim % 32 or args.dtype != "fp8" or args.index_topk < args.window_size:
            self.pick_limit = 0
        self.bound = None

    def bind(self, x: torch.Tensor) -> tuple:
        """The kernels of a decode token with their weights and output buffers bound (gpu.bind), at the first such
        token: the weights have to be in place by then."""
        assert x.dtype == torch.bfloat16
        rd, win, heads, d, dim, later = self.rope_head_dim, self.window_size, self.n_local_heads, self.head_dim, self.dim, gpu.LATER
        rank = self.wq_a.weight.size(0)
        qr, q, o, out = (torch.empty(n, dtype=torch.bfloat16, device=x.device) for n in (rank, heads * d, heads * d, dim))
        rows = attended(win, self.index_topk, x.device)
        self.bound = (
            gpu.bind("attn_q", self.wq_a.weight, self.wq_a.scale, self.q_norm.weight, self.wq_b.weight, self.wq_b.scale,
                     later, later, qr, q, dim, rank, heads, d, rd, self.eps),  # the token, freqs
            gpu.bind("attn_kv", self.wkv.weight, self.wkv.scale, self.kv_norm.weight, later, later, later, dim, d, rd,
                     self.eps),  # the token, freqs, its slot of the window ring
            gpu.bind("sparse_attn", q, self.window_kv_cache, later, later, self.attn_sink, o, heads, d, later, win,
                     self.softmax_scale),  # the compressed KV, the attended positions, their count
            gpu.bind("attn_out", o, later, self.wo_a.fp8, self.wo_a.scale, self.wo_b.weight, self.wo_b.scale, out, heads, d,
                     rd, self.n_local_groups, self.wo_a.fp8.size(0), dim),  # freqs
            self.freqs_cis.data_ptr(), self.window_kv_cache.data_ptr(), rows, qr.view(1, 1, -1), out.view(1, 1, -1))
        return self.bound

    def forward(self, x: torch.Tensor, start_pos: int):
        if not (gpu.active and x.size(0) * x.size(1) == 1 and start_pos):  # position 0 is upstream's prefill case
            return super().forward(x, start_pos)
        attn_q, attn_kv, sparse_attn, attn_out, freqs, window, rows, qr, out = self.bound or self.bind(x)
        rd, win, ratio = self.rope_head_dim, self.window_size, self.compress_ratio
        freqs += start_pos * rd * 4  # the position's row: rd / 2 complex64
        token = x.contiguous()
        at = token.data_ptr()
        attn_q(at, freqs)
        attn_kv(at, freqs, window + start_pos % win * self.head_dim * 2)  # bf16 [win, head_dim]
        count = (start_pos + 1) // ratio if ratio else 0
        if count <= max(self.index_topk, self.pick_limit):
            if self.is_kv_source:
                gpu.compress(self.compressor, self.indexer, token.view(-1), start_pos, self.freqs_cis,
                             self.compress_kv_cache[0], rd)
                m.shared_attn.compress_kv = self.compress_kv_cache
                if self.indexer is not None and self.indexer.owns_k:
                    m.shared_attn.index_k = self.indexer.k_cache
            compressed = m.shared_attn.compress_kv if ratio else self.window_kv_cache
            if count <= self.index_topk:  # the start of the position's row of `attended`
                sparse_attn(compressed.data_ptr(), rows.data_ptr() + min(start_pos, win - 1) * rows.size(1) * 4, win + count)
            else:
                idxs = picked(win, self.index_topk, x.device)
                if self.is_index_source:
                    gpu.index_pick(self.indexer, qr.view(-1), token.view(-1), freqs, m.shared_attn.index_k[0], count,
                                   idxs[win:], win)
                sparse_attn(compressed.data_ptr(), idxs.data_ptr(), idxs.numel())
        else:
            with torch.device(x.device):  # upstream creates index tensors on the default device
                compressed, compress_idxs = self._compress_kv(x, qr, start_pos, win)
                idxs = torch.cat([m.get_window_topk_idxs(win, 1, 1, start_pos).view(-1), compress_idxs.view(-1)])
            sparse_attn(compressed.data_ptr(), idxs.data_ptr(), idxs.numel())
        attn_out(freqs)
        return out


class HostBlock(m.Block):
    """upstream Block; decoding one token with the kernels on, the hyper-connection steps and the norm in front of
    each sublayer run as fused kernels (upstream spends about a hundred small operations on them per sublayer)."""

    attention_cls = HostAttention
    bound = None

    def bind(self, x: torch.Tensor) -> tuple:
        """As HostAttention.bind. The block keeps its own buffers, so what it hands on (the stream, the mix for the
        next sublayer, the FFN's `stream`) stays as it is until the block's next token."""
        assert x.dtype == torch.bfloat16
        hc, dim, later = self.hc_mult, x.size(-1), gpu.LATER
        mid, out = (torch.empty(hc, dim, dtype=torch.bfloat16, device=x.device) for _ in range(2))  # the stream after each sublayer
        a, f = (torch.empty(dim, dtype=torch.bfloat16, device=x.device) for _ in range(2))  # the sublayers' inputs
        # pre[hc], post[hc], comb[hc][hc]
        attn_mix, ffn_mix = (torch.empty((2 + hc) * hc, dtype=torch.float32, device=x.device) for _ in range(2))

        def mixes(part, h, mix):
            return gpu.bind("hc_mixes", h, getattr(self, f"hc_{part}_fn"), getattr(self, f"hc_{part}_scale"),
                            getattr(self, f"hc_{part}_base"), mix, dim, hc, self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters)

        def post(mix, residual, out):
            return gpu.bind("hc_post", later, residual, mix[hc:2 * hc], mix[2 * hc:], out, dim, hc)

        self.ffn.stream = (mid, attn_mix[:hc])
        self.bound = (
            mixes("attn", later, attn_mix),  # the stream
            gpu.bind("hc_pre_norm", later, later, self.attn_norm.weight, a, dim, hc, self.attn_norm.eps),  # the stream, its mix
            post(attn_mix, later, mid),  # the sublayer's output, the stream
            mixes("ffn", mid, ffn_mix),
            gpu.bind("hc_pre_norm", mid, attn_mix, self.ffn_norm.weight, f, dim, hc, self.ffn_norm.eps),  # the mix attention computed
            post(ffn_mix, mid, out),  # the sublayer's output
            a.view(1, 1, -1), f.view(1, 1, -1), out.view(1, 1, hc, dim), ffn_mix[:hc].view(1, 1, hc))
        return self.bound

    def forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
        if not (gpu.active and x.size(0) * x.size(1) == 1):
            return super().forward(x, start_pos, pre_mix, image_mask, *attn_args)
        attn_mix, attn_in, attn_post, ffn_mix, ffn_in, ffn_post, a, f, out, pre = self.bound or self.bind(x)
        h, pre_mix = x.contiguous(), pre_mix.contiguous()
        at = h.data_ptr()
        attn_mix(at)
        attn_in(at, pre_mix.data_ptr())
        y = self.attn(a, start_pos, *attn_args).contiguous()
        attn_post(y.data_ptr(), at)
        ffn_mix()
        ffn_in()
        y = self.ffn(f, image_mask).contiguous()
        ffn_post(y.data_ptr())
        return out, pre


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
        if i:  # in a tuple, so that the block before does not register this one's modules as its own
            net.layers[i - 1].ffn.after = (block.ffn, block.ffn_norm.weight, block.ffn_norm.eps)
    return cache


class Host:
    """The tiered model as a decoder: prefill with start_pos 0, then one token per call. kernels: single-token
    steps use the engine's GPU kernels (by default when the model is on a GPU and they are built)."""

    def __init__(self, net: "m.Transformer", cache: ssd.ExpertCache, device: str = "cuda", kernels: bool | None = None):
        self.net, self.cache, self.device = net, cache, device
        self.origin = self.end = 0  # the cache's clock at the start of this sequence and after its last token
        on_gpu = torch.device(device).type == "cuda"
        self.kernels = on_gpu and os.path.exists(gpu.DLL) if kernels is None else kernels
        self.first_mix = None  # upstream's identity pre-mix of the first block, made once
        # the n-gram hash on the host for decode(). It is given every token of the sequence; net.engram_hash sees
        # the ones upstream's forward runs.
        self.hash = HostHash(net.engram_hash) if self.kernels and net.engram_hash is not None else None

    def decode(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """upstream Transformer.forward for one text token (on the CPU) down to its logits: nothing is sampled
        from them."""
        net = self.net
        rows = self.hash.rows(int(input_ids), start_pos) if self.hash is not None else None
        if rows is not None:
            for layer in net.layers:  # the rows are read while the layers before run
                if layer.engram is not None:
                    layer.engram.embed.start(rows[layer.engram.layer_hash_index], self.device)
        h = net.embed(input_ids).to(self.device).unsqueeze(2).repeat(1, 1, net.hc_mult, 1)
        if self.first_mix is None:
            self.first_mix = m.make_identity_pre_mix(h, net.hc_mult)
        pre_mix = self.first_mix
        for layer in net.layers:
            if layer.engram is not None:
                h = layer.engram(h, rows[layer.engram.layer_hash_index])
            h, pre_mix = layer(h, start_pos, pre_mix, None)
        return net.head(net.norm(layer.hc_pre(h, pre_mix)))

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
                return self.decode(input_ids.cpu(), start_pos)
            if self.hash is not None:
                self.hash.put(input_ids.cpu(), start_pos)
            with torch.device(self.device):  # upstream creates index tensors on the default device
                return self.net(input_ids.to(self.device), start_pos)[1]
        finally:
            gpu.active, m.linear = False, _upstream_linear
