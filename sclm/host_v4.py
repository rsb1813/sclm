# DeepSeek-V4-Flash-0731's attention and block for the tiered host: a decode token runs on the engine's GPU kernels.
import torch

from sclm import gpu
from sclm.upstream import model as m


def attended(win: int, more: int, device: torch.device) -> torch.Tensor:
    """[win, win + more] int32. The start of row p lists what a decode token at position p attends to while every
    compressed position is kept (the last row: any later position): the window slots filled so far, -1 for the
    others, then the compressed positions in order."""
    slots = torch.arange(win, device=device, dtype=torch.int32)
    ring = torch.where(slots[None] > slots[:, None], -1, slots[None])
    return torch.cat([ring, (win + torch.arange(more, device=device, dtype=torch.int32)).expand(win, -1)], dim=1).contiguous()


class V4Attention(m.Attention):
    """upstream Attention; decoding one token with the kernels on, the engine computes the query, the token's
    key/value, the compressed key/value and the index's keys when the token completes a group, the attention itself
    and the output projection (wo_a read as the checkpoint stores it, host.Fp8Weight). While every compressed
    position fits in index_topk the index scores nothing: the top-k of fewer than k positions is all of them. Past
    that, the engine scores them and picks (sclm_gpu_index_pick)."""

    def __init__(self, layer_id: int, args: "m.ModelArgs"):
        super().__init__(layer_id, args)
        self.index_topk = args.index_topk
        self.bound = None

    def bind(self, x: torch.Tensor) -> tuple:
        """The kernels of a decode token with their weights and output buffers bound (gpu.bind), at the first such
        token: the weights have to be in place by then."""
        assert x.dtype == self.kv_cache.dtype == torch.bfloat16
        rd, win, heads, d, dim, later = self.rope_head_dim, self.window_size, self.n_local_heads, self.head_dim, self.dim, gpu.LATER
        rank, cache = self.wq_a.weight.size(0), self.kv_cache[0]
        qr, q, o, out = (torch.empty(n, dtype=torch.bfloat16, device=x.device) for n in (rank, heads * d, heads * d, dim))
        # upstream keeps the norms' weights in fp32 (the checkpoint's bf16 values); the kernels read bf16
        q_norm, kv_norm = self.q_norm.weight.to(torch.bfloat16), self.kv_norm.weight.to(torch.bfloat16)

        def pool(comp, rows):  # the token, its number in the group, the group's rotation, the group's row of `rows`
            return gpu.bind("pool", comp.wkv.weight, comp.wgate.weight, comp.ape, comp.norm.weight.to(torch.bfloat16), later,
                            comp.kv_state, comp.score_state, comp.compress_ratio, comp.overlap, later, later, later, dim,
                            comp.head_dim, rd, comp.norm.eps, comp.rotate), rows.data_ptr(), comp.head_dim * 2

        ratio = self.compress_ratio
        index = self.indexer if ratio else None
        if index is not None:  # the window ring, then the compressed positions the index picks
            picked = torch.arange(win + self.index_topk, device=x.device, dtype=torch.int32)
            pick = gpu.bind("index_pick", index.wq_b.weight, index.wq_b.scale, qr, index.weights_proj.weight, later, later,
                            index.kv_cache[0], picked[win:], rank, index.n_local_heads, index.head_dim, rd, dim, later,
                            self.index_topk, win, index.softmax_scale * index.n_heads ** -0.5, True), picked  # the token, freqs, the keys' count
        self.bound = (
            gpu.bind("attn_q", self.wq_a.weight, self.wq_a.scale, q_norm, self.wq_b.weight, self.wq_b.scale, later, later, qr, q,
                     dim, rank, heads, d, rd, self.eps),  # the token, freqs
            gpu.bind("attn_kv", self.wkv.weight, self.wkv.scale, kv_norm, later, later, later, dim, d, rd, self.eps),  # the token, freqs, its slot of the window ring
            gpu.bind("sparse_attn", q, cache, cache[win:], later, self.attn_sink, o, heads, d, later, win,
                     self.softmax_scale),  # the attended positions, their count
            gpu.bind("attn_out", o, later, self.wo_a.fp8, self.wo_a.scale, self.wo_b.weight, self.wo_b.scale, out, heads, d, rd,
                     self.n_local_groups, self.wo_a.fp8.size(0), dim),  # freqs
            pool(self.compressor, cache[win:]) if ratio else None,
            pool(index.compressor, index.kv_cache[0]) if index is not None else None,
            self.freqs_cis.data_ptr(), cache.data_ptr(), attended(win, self.index_topk, x.device),
            pick if index is not None else None, out.view(1, 1, -1))
        return self.bound

    def forward(self, x: torch.Tensor, start_pos: int):
        if not (gpu.active and x.size(0) * x.size(1) == 1 and start_pos):  # position 0 is upstream's prefill case
            return super().forward(x, start_pos)
        attn_q, attn_kv, sparse_attn, attn_out, pool, index_pool, freqs, cache, rows, pick, out = self.bound or self.bind(x)
        rd, win, ratio = self.rope_head_dim, self.window_size, self.compress_ratio
        at_pos = freqs + start_pos * rd * 4  # the position's row: rd / 2 complex64
        token = x.contiguous()
        at = token.data_ptr()
        attn_q(at, at_pos)
        attn_kv(at, at_pos, cache + start_pos % win * self.head_dim * 2)  # bf16 [win, head_dim]
        count = (start_pos + 1) // ratio if ratio else 0
        idxs, n = rows.data_ptr() + min(start_pos, win - 1) * rows.size(1) * 4, win + count
        if ratio:
            group = freqs + (start_pos + 1 - ratio) * rd * 4  # the rotation of the group's first position
            launch, base, row = pool
            launch(at, start_pos % ratio, group, base + start_pos // ratio * row)
            if index_pool is not None:
                launch, base, row = index_pool
                launch(at, start_pos % ratio, group, base + start_pos // ratio * row)
                if count > self.index_topk:
                    launch, picked = pick
                    launch(at, at_pos, count)
                    idxs, n = picked.data_ptr(), picked.numel()
        sparse_attn(idxs, n)
        attn_out(at_pos)
        return out


class V4Block(m.Block):
    """upstream Block; decoding one token with the kernels on, the hyper-connection steps and the norm in front of
    each sublayer run as fused kernels. The token id reaches the FFN as a number (the host knows it), not as a
    tensor."""

    attention_cls = V4Attention
    bound = None

    def bind(self, x: torch.Tensor) -> tuple:
        """As V4Attention.bind. The block keeps its own buffers, so what it hands on (the stream, the FFN's
        `stream`) stays as it is until the block's next token."""
        assert x.dtype == torch.bfloat16
        hc, dim, later = self.hc_mult, x.size(-1), gpu.LATER
        mid, out = (torch.empty(hc, dim, dtype=torch.bfloat16, device=x.device) for _ in range(2))  # the stream after each sublayer
        a, f = (torch.empty(dim, dtype=torch.bfloat16, device=x.device) for _ in range(2))  # the sublayers' inputs
        # pre[hc], post[hc], comb[hc][hc]
        attn_mix, ffn_mix = (torch.empty((2 + hc) * hc, dtype=torch.float32, device=x.device) for _ in range(2))

        def mixes(part, h, mix):
            return gpu.bind("hc_mixes", h, getattr(self, f"hc_{part}_fn"), getattr(self, f"hc_{part}_scale"),
                            getattr(self, f"hc_{part}_base"), mix, dim, hc, self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters)

        def sublayer_in(h, mix, norm):
            return gpu.bind("hc_pre_norm", h, mix[:hc], norm.weight.to(torch.bfloat16), a if norm is self.attn_norm else f, dim, hc, norm.eps)

        def post(mix, residual, out):
            return gpu.bind("hc_post", later, residual, mix[hc:2 * hc], mix[2 * hc:], out, dim, hc)

        self.ffn.stream = (mid, ffn_mix[:hc])  # the FFN's input before its norm: what the next layer's gate is guessed from
        self.bound = (
            mixes("attn", later, attn_mix), sublayer_in(later, attn_mix, self.attn_norm),  # the stream
            post(attn_mix, later, mid),  # the sublayer's output, the stream
            mixes("ffn", mid, ffn_mix), sublayer_in(mid, ffn_mix, self.ffn_norm),
            post(ffn_mix, mid, out),  # the sublayer's output
            a.view(1, 1, -1), f.view(1, 1, -1), out.view(1, 1, hc, dim))
        return self.bound

    def forward(self, x, start_pos, input_ids, *attn_args):
        if not (gpu.active and x.size(0) * x.size(1) == 1):
            return super().forward(x, start_pos, input_ids, *attn_args)
        attn_mix, attn_in, attn_post, ffn_mix, ffn_in, ffn_post, a, f, out = self.bound or self.bind(x)
        h = x.contiguous()
        at = h.data_ptr()
        attn_mix(at)
        attn_in(at)
        y = self.attn(a, start_pos, *attn_args).contiguous()
        attn_post(y.data_ptr(), at)
        ffn_mix()
        ffn_in()
        y = self.ffn(f, input_ids).contiguous()
        ffn_post(y.data_ptr())
        return out
