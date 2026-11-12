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
from sclm.layerwise import V4, load_params_
from sclm.tiered import TieredMoE
from sclm.upstream import model as m

import kernel  # noqa: E402  (sclm.upstream puts it on the path)

# config overrides of the text-only decode model: no draft (MTP) blocks, no vision tower (V4.1 has one)
TEXT_ONLY = {"dspark_block_size": 0, "dspark_target_layer_ids": ()} | ({} if V4 else {"vision_n_layers": 0})


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


class Bf16Proj(nn.Module):
    """A projection of V4-Flash-0731's Compressor (wkv, wgate) as the checkpoint stores it, bf16. upstream keeps an
    fp32 copy of the same values (twice the memory, 0.6 GiB over the model); here upstream's fp32 path widens it at
    each use, and the engine's pool kernel reads it as it is."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.bfloat16), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


def keep_bf16_(net: nn.Module) -> None:
    """Every Compressor's projections go back to bf16 (Bf16Proj)."""
    for comp in net.modules():
        if isinstance(comp, m.Compressor):
            comp.wkv, comp.wgate = Bf16Proj(comp.wkv.weight), Bf16Proj(comp.wgate.weight)


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


class _NoCodes:
    """Stands in for a layer's coded experts (core.Layer) when it has none: every routed expert is an original one."""

    experts = {}

    def start(self, experts: list, gates: list, x) -> None:
        pass

    def wait(self, out) -> None:
        out[:] = 0  # the CPU tier's sum

    def touch(self, experts: list) -> None:
        pass


class _Absent(nn.Module):
    """Stands in, while the model is built, for a module whose weights are placed afterwards."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def build_net(args: "m.ModelArgs", tokenizer=None, device: str = "cuda") -> "m.Transformer":
    """upstream's Transformer for `args` (with TEXT_ONLY applied) without what is placed afterwards or stays off the
    device: the routed experts (each block's ffn is a TieredMoE), the engram tables, the embedding and the head.
    Parameters are uninitialized. V4-Flash-0731 has its own block (sclm/host_v4.py)."""
    if V4:
        from sclm.host_v4 import V4Block

        assert not args.dspark_block_size
        names = ("Block", "MoE", "ParallelEmbedding", "ParallelHead")
        upstream = [getattr(m, n) for n in names]
        m.Block, m.MoE, m.ParallelEmbedding, m.ParallelHead = V4Block, TieredMoE, _Absent, _Absent
        try:
            with torch.device(device):
                return m.Transformer(args)
        finally:
            for n, cls in zip(names, upstream):
                setattr(m, n, cls)
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
    embedding table in RAM, the head in bf16, attn.wo_a in fp8 and V4-Flash-0731's compressor projections in bf16."""
    net.embed = HostEmbedding(ckpt.get("embed.weight"))
    if V4:
        for i, block in enumerate(net.layers):
            prefix = f"layers.{i}."
            block.attn.wo_a = Fp8Weight(*(ckpt.get(f"{prefix}attn.wo_a.{part}").to(device) for part in ("weight", "scale")))
            load_params_(block, ckpt.module_state(prefix, skip=(".ffn.experts.", ".attn.wo_a.")), prefix)
            keep_bf16_(block)
        torch.cuda.empty_cache()  # the fp32 copies' memory goes back to the device: the expert cache is sized by what is free
        with torch.no_grad():
            for part in ("fn", "base", "scale"):  # the head's hyper-connection
                getattr(net, "hc_head_" + part).copy_(ckpt.get("hc_head_" + part))
        load_params_(net.norm, {"weight": ckpt.get("norm.weight")}, "norm.")
        net.head = Bf16Head(ckpt.get("head.weight").to(device))
        return
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


def extend_positions_(net: "m.Transformer", args: "m.ModelArgs", positions: int) -> None:
    """Gives every attention of V4-Flash-0731 rotations for `positions` positions: upstream's table, which the model's
    max_seq_len only cuts short. A sequence can then be longer than the state the model was built with (Sequence),
    which is better built small: its memory is not given back whole when it is no longer in use."""
    tables = {}
    for layer in net.layers:
        attn = layer.attn  # as upstream's Attention chooses its table
        key = (args.original_seq_len, args.compress_rope_theta) if attn.compress_ratio else (0, args.rope_theta)
        if key not in tables:
            with torch.device(attn.freqs_cis.device):  # where upstream computes it: the values are the device's
                tables[key] = m.precompute_freqs_cis.__wrapped__(attn.rope_head_dim, positions, *key, args.rope_factor,
                                                                 args.beta_fast, args.beta_slow)
        attn.freqs_cis = tables[key]


def attach_tiers_(net: "m.Transformer", args: "m.ModelArgs", ckpt: Checkpoint, codes_dir: str | None, slots: int, lam: float,
                  device: str = "cuda", background: int = 0) -> ssd.ExpertCache:
    """Gives every block its coded experts (codes_dir/layer_NN.safetensors, held in RAM) and the shared cache of
    original experts, read from `ckpt` on demand. Without codes_dir every routed expert is an original one: the
    lossless model."""
    cache = ssd.ExpertCache(ssd.ExpertReader(ckpt), args.n_layers, args.n_routed_experts, args.dim, args.moe_inter_dim,
                            args.swiglu_limit, slots, lam, device, scratch=args.n_activated_experts,
                            background=background)
    for i, block in enumerate(net.layers):
        if codes_dir is None:
            block.ffn.ram = _NoCodes()
        else:
            path = os.path.join(codes_dir, f"layer_{i:02d}.safetensors")
            block.ffn.ram = core.Layer(path, args.dim, args.moe_inter_dim, args.swiglu_limit)
        block.ffn.cache = cache
        if i:  # in a tuple, so that the block before does not register this one's modules as its own
            net.layers[i - 1].ffn.after = (block.ffn, block.ffn_norm.weight, block.ffn_norm.eps)
    return cache


class Sequence:
    """The attention state of one token sequence (V4-Flash-0731), with room for `capacity` tokens: every layer's
    window ring and compressed key/values, its compressors' groups and the index's keys. Host.switch makes it the
    sequence the model computes on. Without a capacity: the state the model was built with."""

    def __init__(self, net: "m.Transformer", capacity: int | None = None):
        self.capacity = capacity or net.max_seq_len
        self.bound = [None] * len(net.layers)  # each layer's V4Attention.bound: its kernels hold this state's addresses
        self.state, self.compressors = [], []  # (module, buffer name, tensor); the compressors, which keep a view
        for layer in net.layers:
            attn = layer.attn
            held = [(attn, "kv_cache")]
            if attn.compress_ratio:
                held += [(attn.compressor, "kv_state"), (attn.compressor, "score_state")]
                self.compressors.append(attn.compressor)
                if attn.indexer is not None:
                    comp = attn.indexer.compressor
                    held += [(attn.indexer, "kv_cache"), (comp, "kv_state"), (comp, "score_state")]
                    self.compressors.append(comp)
            for module, name in held:
                t = getattr(module, name)
                self.state.append((module, name, t if capacity is None else self.room(module, name, t, capacity)))

    @staticmethod
    def rows(module, capacity: int) -> int:
        """Rows of a module's key/values for `capacity` tokens: a row per group of tokens, after the rows that do not
        depend on the length (an attention's window ring)."""
        fixed = module.window_size if isinstance(module, m.Attention) else 0
        return fixed + (capacity // module.compress_ratio if module.compress_ratio else 0)

    @staticmethod
    def room(module, name: str, like: torch.Tensor, capacity: int) -> torch.Tensor:
        """An empty tensor of the state like `like`, for `capacity` tokens."""
        if name != "kv_cache":
            return torch.full_like(like, float("-inf") if name == "score_state" else 0.0)
        return torch.zeros(1, Sequence.rows(module, capacity), like.size(2), dtype=like.dtype, device=like.device)

    def kept(self) -> list:
        """What a snapshot keeps: every layer's window ring and its compressors' groups."""
        return [t[:, :module.window_size] if name == "kv_cache" else t for module, name, t in self.state
                if name != "kv_cache" or isinstance(module, m.Attention)]


class Host:
    """The tiered model as a decoder: prefill with start_pos 0, then one token per call. kernels: single-token
    steps use the engine's GPU kernels (by default when the model is on a GPU and they are built).
    V4-Flash-0731: the model computes on one token sequence at a time (`seq`), and switch() changes which."""

    def __init__(self, net: "m.Transformer", cache: ssd.ExpertCache, device: str = "cuda", kernels: bool | None = None):
        self.net, self.cache, self.device = net, cache, device
        self.clock = 0  # the cache's clock: tokens computed so far, over every sequence
        self.seq = Sequence(net) if V4 else None
        self.slots = None  # the expert cache's slots if no sequence had any room (reserve)
        self.lanes = []  # host_v4.Lane: the buffers of tokens computed together (feed)
        on_gpu = torch.device(device).type == "cuda"
        self.kernels = on_gpu and os.path.exists(gpu.DLL) if kernels is None else kernels
        self.first_mix = None  # upstream's identity pre-mix of the first block, made once
        # the n-gram hash on the host for decode(). It is given every token of the sequence; net.engram_hash sees
        # the ones upstream's forward runs.
        self.hash = HostHash(net.engram_hash) if self.kernels and getattr(net, "engram_hash", None) is not None else None

    def decode(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """upstream Transformer.forward for one text token (on the CPU) down to its logits: nothing is sampled
        from them."""
        net = self.net
        if V4:  # no engram; every block mixes its own stream, and the token id picks the first layers' experts
            token = int(input_ids)
            h = net.embed(input_ids).to(self.device).unsqueeze(2).repeat(1, 1, net.hc_mult, 1)
            for layer in net.layers:
                h = layer(h, start_pos, token)
            return net.head(net.norm(layer.hc_head(h, net.hc_head_fn, net.hc_head_scale, net.hc_head_base)))
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

    def sequence(self, capacity: int) -> Sequence:
        """A new, empty sequence with room for `capacity` tokens; switch() to it before its first token."""
        return Sequence(self.net, capacity)

    def resize(self, seq: Sequence, capacity: int) -> None:
        """Gives `seq` room for `capacity` tokens; what it holds stays. A tensor at a time: the device never holds two
        whole states."""
        active = seq is self.seq
        seq.bound = [None] * len(seq.bound)  # the kernels are bound again, to the new addresses
        if active:
            for layer in self.net.layers:
                layer.attn.bound = None
            for comp in seq.compressors:
                comp.kv_cache = None
        for i, (module, name, old) in enumerate(seq.state):
            if name == "kv_cache":
                t = Sequence.room(module, name, old, capacity)
                rows = min(old.size(1), t.size(1))
                t[:, :rows].copy_(old[:, :rows])
                seq.state[i] = (module, name, t)
                if active:
                    setattr(module, name, t)
        seq.capacity = capacity

    def reserve(self, tokens: int) -> None:
        """Sets aside device memory for sequences that hold `tokens` tokens in all; the expert cache keeps the slots
        that leaves (and starts empty). The first call counts from the cache's slots as they are and the state the
        model was built with, room for net.max_seq_len tokens: it comes once that state is no longer in use."""
        per = self.cache.store.size(1)

        def nbytes(capacity: int) -> int:  # of the key/values, which are all that grows with the length
            return sum(Sequence.rows(mod, capacity) * t.size(2) * t.element_size() for mod, name, t in self.seq.state if name == "kv_cache")

        if self.slots is None:
            self.slots = self.cache.n_slots + nbytes(self.net.max_seq_len) // per
        slots = self.slots - -(-nbytes(tokens) // per)
        assert slots > 0, f"no device memory is left for the expert cache with sequences of {tokens} tokens"
        self._resize(slots)

    def _resize(self, slots: int) -> None:
        """cache.resize(slots), or the most slots below that the device has one piece of memory for: what was given
        back to the allocator in the meantime may not come back whole."""
        while True:
            try:
                return self.cache.resize(slots)
            except torch.OutOfMemoryError:
                if slots <= 16:
                    raise
                slots -= 16

    def switch(self, seq: Sequence) -> None:
        """The model computes on `seq` from now on: its state takes the place of the last sequence's in the modules."""
        if seq is self.seq:
            return
        for i, layer in enumerate(self.net.layers):
            self.seq.bound[i], layer.attn.bound = layer.attn.bound, seq.bound[i]
        for module, name, t in seq.state:
            setattr(module, name, t)
        for comp in seq.compressors:
            comp.kv_cache = None  # upstream takes the view again from the state now in place
        self.seq = seq

    LANES = 16  # tokens feed() computes together

    @torch.inference_mode()
    def feed(self, ids: list, start_pos: int) -> torch.Tensor:
        """logits() for each of the tokens `ids` in turn, which continue the sequence at start_pos; returns the logits
        after the last. With the kernels (V4-Flash-0731) up to LANES tokens go through the layers together, to the
        same result: token k leaves a layer before token k + 1 enters its MoE, and enters the next layer's attention
        after token k - 1 has. While the CPU computes one token's tier, the device computes the next tokens'
        attention, no token waits for its routing, and a layer's experts are read for all the tokens when the first
        one enters it."""
        if not (V4 and self.kernels) or start_pos == 0:
            out = self.logits(torch.tensor([ids[:1]]), start_pos)
            ids, start_pos = ids[1:], start_pos + 1
            if not (V4 and self.kernels):
                for i, token in enumerate(ids):
                    out = self.logits(torch.tensor([[token]]), start_pos + i)
                return out
        net, layers = self.net, self.net.layers
        if not self.lanes:
            from sclm.host_v4 import Lane

            self.lanes = [Lane(k + 1, net.hc_mult, net.embed.weight.size(1), layers[0].ffn.gate.topk, self.device)
                          for k in range(self.LANES)]
        gpu.active, m.linear = True, kernel_linear
        try:
            for at in range(0, len(ids), self.LANES):
                part = ids[at:at + self.LANES]
                lanes, pos = self.lanes[:len(part)], start_pos + at
                for k, (lane, token) in enumerate(zip(lanes, part)):
                    lane.token = self.clock + k
                    lane.h = net.embed(torch.tensor([[token]])).to(self.device).unsqueeze(2).repeat(1, 1, net.hc_mult, 1).contiguous()
                self.clock += len(part)
                begun = [layers[0].start(lane, lane.h.data_ptr(), pos + k, part[k]) for k, lane in enumerate(lanes)]
                for i, layer in enumerate(layers):
                    layer.ffn.read_for(lanes)
                    for k, lane in enumerate(lanes):
                        layer.end(lane, begun[k])
                        if i + 1 < len(layers):
                            begun[k] = layers[i + 1].start(lane, lane.out.data_ptr(), pos + k, part[k])
                out = lanes[-1].out.view(1, 1, net.hc_mult, -1)
            if not ids:
                return out
            return net.head(net.norm(layers[-1].hc_head(out, net.hc_head_fn, net.hc_head_scale, net.hc_head_base)))
        finally:
            gpu.active, m.linear = False, _upstream_linear

    PART = 4096  # tokens prefill() takes through the layers at a time
    WORK = 5 << 29  # device memory a part of that length works in (2.5 GiB), lent by the expert cache meanwhile

    @torch.inference_mode()
    def prefill(self, ids: list, start_pos: int) -> torch.Tensor:
        """The logits after the tokens `ids`, which continue the sequence at start_pos (V4-Flash-0731), for a long run
        of them: PART tokens at a time go through each layer as tensors, as upstream computes a prompt, and the MoE
        with the original experts, each read from the checkpoint once for all the tokens routed to it
        (TieredMoE.together). A part takes the time of reading the model's experts, about 22 s, whatever its length;
        below a few hundred tokens feed() is faster. The numbers are upstream's for these tokens, not the tiered
        model's: the prompt is computed with the original weights. The state is left for any form to go on from."""
        assert V4 and ids
        net, out = self.net, None
        if start_pos == 0:  # as logits(): the last sequence's groups are not this one's
            for module in net.modules():
                if isinstance(module, m.Compressor):
                    module.kv_state.zero_()
                    module.score_state.fill_(float("-inf"))
        slots = self.cache.n_slots
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize()  # what the tokens before queued is done: the slots and the state are read here
            short = self.WORK - torch.cuda.mem_get_info()[0]
            if short > 0:  # the expert cache gives up slots for the working memory, and starts empty afterwards
                self.cache.resize(max(1, slots - -(-short // self.cache.store.size(1))))
        gpu.together = True
        try:
            with torch.device(self.device):  # upstream creates index tensors on the default device
                for at in range(0, len(ids), self.PART):
                    part = torch.tensor([ids[at:at + self.PART]])
                    h = net.embed(part).to(self.device).unsqueeze(2).repeat(1, 1, net.hc_mult, 1)
                    part = part.to(self.device)
                    for layer in net.layers:
                        h = layer.together(h, start_pos + at, part)
                    self.clock += part.size(1)
                h = layer.hc_head(h[:, -1:], net.hc_head_fn, net.hc_head_scale, net.hc_head_base)
                return net.head(net.norm(h))
        finally:
            gpu.together = False
            if self.cache.n_slots != slots:
                del h
                self._resize(slots)

    def snapshot(self) -> list:
        """The attention state of the sequence at this point (V4-Flash-0731). restore() takes the sequence back to it."""
        return [t.clone() for t in self.seq.kept()]

    def restore(self, kept: list) -> None:
        """Takes the sequence back to a snapshot(): the next token continues from that position. The compressed
        positions written since are overwritten as the sequence passes them again."""
        with torch.no_grad():
            for t, saved in zip(self.seq.kept(), kept):
                t.copy_(saved)

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Next-token logits [batch, vocab] (fp32) after input_ids, which continue the sequence at start_pos."""
        if start_pos == 0 and V4:
            # A compressor with overlapping windows keeps the group before the current one, and a prefill shorter
            # than a group leaves that part as it is: the last sequence's would join the first group's pooling.
            for module in self.net.modules():
                if isinstance(module, m.Compressor):
                    module.kv_state.zero_()
                    module.score_state.fill_(float("-inf"))
        self.cache.token = self.clock  # a new sequence finds the cache's experts still there
        self.clock += input_ids.size(-1)
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
