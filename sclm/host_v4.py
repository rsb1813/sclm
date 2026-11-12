# DeepSeek-V4-Flash-0731's attention and block for the tiered host: a decode token runs on the engine's GPU kernels.
from functools import lru_cache

import torch

from sclm import gpu
from sclm.upstream import model as m

import kernel  # noqa: E402  (sclm.upstream puts it on the path)

ROWS = 256  # tokens of a long run whose own steps (query, hyper-connection) are computed at a time


@lru_cache(4)
def attended(win: int, more: int, device: torch.device) -> torch.Tensor:
    """[win, win + more] int32, one table for every layer and sequence. The start of row p lists what a decode token
    at position p attends to while every compressed position is kept (the last row: any later position): the window
    slots filled so far, -1 for the others, then the compressed positions in order."""
    slots = torch.arange(win, device=device, dtype=torch.int32)
    ring = torch.where(slots[None] > slots[:, None], -1, slots[None])
    return torch.cat([ring, (win + torch.arange(more, device=device, dtype=torch.int32)).expand(win, -1)], dim=1).contiguous()


class Lane:
    """What one of several tokens computed together (Host.feed) keeps to itself from one half of a block to the next:
    the hyper-connection stream after attention (`mid`) and after the block (`out`), the FFN's mix, the MoE's buffers
    and the CPU tier's sum. The buffers a block's kernels only hand on to each other are shared by all the tokens:
    the device runs the kernels in the order they were queued. index: the lane's number in the engine (1 and up)."""

    def __init__(self, index: int, hc: int, dim: int, topk: int, device):
        self.index, self.token = index, 0  # token: the expert cache's clock at the lane's token
        self.h = None  # the token's embedding as the first block's stream
        self.mid, self.out = (torch.empty(hc, dim, dtype=torch.bfloat16, device=device) for _ in range(2))
        self.ffn_mix = torch.empty((2 + hc) * hc, dtype=torch.float32, device=device)  # pre[hc], post[hc], comb[hc][hc]
        self.buf = gpu.moe_buffers.__wrapped__(topk, dim, 32, device)  # room for a guess of up to 32 experts
        self.ram_out = torch.empty(dim, dtype=torch.float32, pin_memory=True)
        self.bound = {}  # a block or (its MoE, the guess) -> their kernels on this lane's buffers

    def moe(self, ffn, guess: int) -> tuple:
        if (ffn, guess) not in self.bound:
            self.bound[ffn, guess] = ffn.kernels(self.buf, guess, (self.mid, self.ffn_mix[:self.mid.size(0)]), self.ram_out)
        return self.bound[ffn, guess]


def marked(start_pos: int, n: int) -> list:
    """The positions among the n tokens from start_pos after which the state is to be kept (gpu.marks)."""
    return [] if gpu.marks is None else [at for at in gpu.marks[0] if start_pos < at <= start_pos + n]


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
            launch = gpu.bind("pool", comp.wkv.weight, comp.wgate.weight, comp.ape, comp.norm.weight.to(torch.bfloat16), later,
                              comp.kv_state, comp.score_state, comp.compress_ratio, comp.overlap, later, later, later, dim,
                              comp.head_dim, rd, comp.norm.eps, comp.rotate, comp.wkv.weight.dtype == torch.bfloat16)
            return launch, rows.data_ptr(), comp.head_dim * 2

        ratio = self.compress_ratio
        index = self.indexer if ratio else None
        # the compressed positions a token attends to at most: what the index keeps, or without an index all of them,
        # up to the last position the rotations reach
        kept = self.index_topk if index is not None else self.freqs_cis.size(0) // ratio if ratio else 0
        assert win + kept <= gpu.ATTENDED
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
            self.freqs_cis.data_ptr(), cache.data_ptr(), attended(win, kept, x.device),
            pick if index is not None else None, out.view(1, 1, -1))
        return self.bound

    def squeeze(self, comp, x: torch.Tensor, start_pos: int, cache: torch.Tensor) -> None:
        """upstream Compressor.forward for the tokens x [1, n, dim] that continue the sequence at start_pos, as its
        one-token form computes them: the groups these tokens complete go into their rows of `cache`, and the state
        (the group under way, and with overlapping groups the one before it) is left as they leave it."""
        n, ratio, d, rd = x.size(1), comp.compress_ratio, comp.head_dim, comp.rope_head_dim
        at = start_pos + torch.arange(n, device=x.device)
        kv, score = comp.wkv(x.float())[0], comp.wgate(x.float())[0] + comp.ape[at % ratio]
        held = start_pos % ratio + (ratio if comp.overlap else 0)  # rows of the state these tokens' groups pool
        kv, score = torch.cat([comp.kv_state[0, :held], kv]), torch.cat([comp.score_state[0, :held], score])
        for at in marked(start_pos, n):  # the state after `at` tokens: the group under way then, and the one before
            base, done = held - start_pos % ratio, at // ratio * ratio  # `done`: where that group begins
            row = held - start_pos  # a position's row of kv and score
            k, s = comp.kv_state.clone(), comp.score_state.clone()
            if comp.overlap:
                k[0, :ratio], s[0, :ratio] = kv[row + done - ratio:row + done], score[row + done - ratio:row + done]
            k[0, base:base + at - done], s[0, base:base + at - done] = kv[row + done:row + at], score[row + done:row + at]
            gpu.marks[1][at][comp, "kv_state"], gpu.marks[1][at][comp, "score_state"] = k.cpu(), s.cpu()
        full = kv.size(0) // ratio * ratio  # rows in whole groups
        groups = full // ratio - comp.overlap  # groups completed (with overlap, the first whole one is the one before)
        if groups > 0:
            k, s = kv[:full].unflatten(0, (-1, ratio)), score[:full].unflatten(0, (-1, ratio))
            if comp.overlap:  # a group pools the first half of the group before and its own second half
                comp.kv_state[0, :ratio], comp.score_state[0, :ratio] = k[-1], s[-1]
                k, s = torch.cat([k[:-1, :, :d], k[1:, :, d:]], dim=1), torch.cat([s[:-1, :, :d], s[1:, :, d:]], dim=1)
            out = comp.norm((k * s.softmax(dim=1)).sum(dim=1).to(x.dtype)).unsqueeze(0)  # [1, groups, d]
            first = start_pos // ratio
            m.apply_rotary_emb(out[..., -rd:], self.freqs_cis[first * ratio:(first + groups) * ratio:ratio])
            if comp.rotate:
                out = m.rotate_activation(out)
                m.fp4_act_quant(out, m.fp4_block_size, True)
            else:
                m.act_quant(out[..., :-rd], 64, m.scale_fmt, m.scale_dtype, True)
            cache[first:first + groups] = out[0]
        under = slice(ratio if comp.overlap else 0, (ratio if comp.overlap else 0) + kv.size(0) - full)  # the group under way
        comp.kv_state[0, under], comp.score_state[0, under] = kv[full:], score[full:]

    def choose(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, count: torch.Tensor) -> torch.Tensor:
        """upstream Indexer.forward's choice for each of the tokens x [1, n, dim]: the index_topk compressed positions
        that score highest among the count[i] the token sees, -1 where it sees fewer ([n, at most index_topk])."""
        index, n, rd, most = self.indexer, x.size(1), self.rope_head_dim, int(count[-1])
        self.squeeze(index.compressor, x, start_pos, index.kv_cache[0])  # the keys of the groups these tokens complete
        weights = index.weights_proj(x) * (index.softmax_scale * index.n_heads ** -0.5)
        keys, k, picked = index.kv_cache[:, :most], min(index.index_topk, most), []
        if not k:
            return torch.empty(n, 0, dtype=torch.long, device=x.device)
        hidden = torch.arange(most, device=x.device) >= count[:, None]  # the positions a token does not see yet
        step = max(1, (1 << 27) // (index.n_local_heads * most))  # the scores of so many tokens at a time
        for s0 in range(0, n, step):
            rows = slice(s0, s0 + step)
            q = index.wq_b(qr[:, rows]).unflatten(-1, (index.n_local_heads, index.head_dim))
            m.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos + s0:start_pos + s0 + q.size(1)])
            q = m.rotate_activation(q)
            m.fp4_act_quant(q, m.fp4_block_size, True)
            score = (torch.einsum("bshd,btd->bsht", q, keys).relu_() * weights[:, rows].unsqueeze(-1)).sum(dim=2)[0]
            top = score.masked_fill_(hidden[rows], float("-inf")).topk(k, dim=-1)[1]
            picked.append(torch.where(top >= count[rows, None], -1, top))
        return torch.cat(picked)

    def together(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """upstream Attention.forward for the tokens x [1, n, dim] that continue the sequence at start_pos, each
        attending to what upstream's one-token form lets it: its window, and the compressed positions completed up to
        it, those these tokens complete among them (all of them, or the index's choice). The state is left as the
        tokens one by one leave it, for whichever form comes next. (upstream's own form for several tokens only starts
        a sequence, and holds every token's scores of every position at once.)"""
        n, win, ratio, rd, dev = x.size(1), self.window_size, self.compress_ratio, self.rope_head_dim, x.device
        pos, cache = start_pos + torch.arange(n, device=dev), self.kv_cache[0]
        freqs = self.freqs_cis[start_pos:start_pos + n]
        qr = self.q_norm(self.wq_a(x))
        kv = self.kv_norm(self.wkv(x))
        m.apply_rotary_emb(kv[..., -rd:], freqs)
        m.act_quant(kv[..., :-rd], 64, m.scale_fmt, m.scale_dtype, True)
        # the window: the positions before these tokens that one of them still sees, oldest first, then the tokens
        seen = min(start_pos, win - 1)
        rows = torch.cat([cache[(start_pos - seen + torch.arange(seen, device=dev)) % win], kv[0]])
        idxs = seen + torch.arange(n, device=dev)[:, None] - torch.arange(win - 1, -1, -1, device=dev)
        idxs = torch.where(idxs < 0, -1, idxs)
        for at in marked(start_pos, n):  # the window ring after `at` tokens
            ring, last = self.kv_cache[:, :win].clone(), torch.arange(max(start_pos, at - win), at, device=dev)
            ring[0, last % win] = kv[0, last - start_pos]
            gpu.marks[1][at][self, "kv_cache"] = ring.cpu()
        if ratio:
            self.squeeze(self.compressor, x, start_pos, cache[win:])
            count = (pos + 1) // ratio  # the compressed positions each token sees
            if self.indexer is None:
                picked = torch.arange(int(count[-1]), device=dev).expand(n, -1)
                picked = torch.where(picked < count[:, None], picked, -1)
            else:
                picked = self.choose(x, qr, start_pos, count)
            idxs = torch.cat([idxs, torch.where(picked < 0, -1, picked + rows.size(0))], dim=1)
            rows = torch.cat([rows, cache[win:win + int(count[-1])]])
        cache[pos[-win:] % win] = kv[0, -win:]  # the window ring keeps the last `win` positions
        wo_a, out = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1), []
        for s0 in range(0, n, ROWS):  # a token's query is sixty-four times its key/value
            part = slice(s0, s0 + ROWS)
            q = self.wq_b(qr[:, part]).unflatten(-1, (self.n_local_heads, self.head_dim))
            q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
            m.apply_rotary_emb(q[..., -rd:], freqs[part])
            o = kernel.sparse_attn(q, rows[None], self.attn_sink, idxs[None, part], self.softmax_scale,
                                   chunk=max(8, (1 << 16) // idxs.size(1)))
            m.apply_rotary_emb(o[..., -rd:], freqs[part], True)
            o = torch.einsum("bsgd,grd->bsgr", o.view(1, q.size(1), self.n_local_groups, -1), wo_a)
            out.append(self.wo_b(o.flatten(2)))
        return torch.cat(out, dim=1)

    def forward(self, x: torch.Tensor, start_pos: int):
        if gpu.together:
            return self.together(x, start_pos)
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
        self.shared = (a, f, attn_mix)  # what a lane's kernels read and write of this block's own buffers
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

    def start(self, lane: Lane, h: int, start_pos: int, token: int) -> tuple:
        """The first half of forward for a lane's token whose stream is at address h: attention, and the MoE as far
        as it goes without the routing (TieredMoE.begin). Nothing is waited for."""
        attn_mix, attn_in, _, _, _, _, a, f, _ = self.bound or self.bind(lane.out)
        if self not in lane.bound:
            hc, dim, mix = self.hc_mult, lane.out.size(-1), self.shared[2]
            norm = self.ffn_norm.weight.to(torch.bfloat16)
            lane.bound[self] = (
                gpu.bind("hc_post", gpu.LATER, gpu.LATER, mix[hc:2 * hc], mix[2 * hc:], lane.mid, dim, hc),  # attention's output, the stream
                gpu.bind("hc_mixes", lane.mid, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, lane.ffn_mix, dim, hc,
                         self.norm_eps, self.hc_eps, self.hc_sinkhorn_iters),
                gpu.bind("hc_pre_norm", lane.mid, lane.ffn_mix[:hc], norm, self.shared[1], dim, hc, self.ffn_norm.eps),
                gpu.bind("hc_post", gpu.LATER, lane.mid, lane.ffn_mix[hc:2 * hc], lane.ffn_mix[2 * hc:], lane.out, dim, hc))  # the MoE's output
        attn_post, ffn_mix, ffn_in, _ = lane.bound[self]
        attn_mix(h)
        attn_in(h)
        y = self.attn(a, start_pos).contiguous()
        attn_post(y.data_ptr(), h)
        ffn_mix()
        ffn_in()
        return self.ffn.begin(f.view(-1), token, lane)

    def end(self, lane: Lane, begun: tuple) -> None:
        """The second half: the rest of the MoE (TieredMoE.finish) and the stream after the block, into lane.out."""
        lane.bound[self][3](self.ffn.finish(begun, lane).data_ptr())

    def together(self, h: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        """upstream Block.forward for a long run of tokens (the stream h [1, n, hc, dim]). The hyper-connection steps
        are each token's own, and upstream's take sixteen times the stream's memory: ROWS tokens at a time."""
        parts = [slice(s0, s0 + ROWS) for s0 in range(0, h.size(1), ROWS)]

        def sublayer(name: str, norm, run) -> torch.Tensor:
            fn, scale, base = (getattr(self, f"hc_{name}_{part}") for part in ("fn", "scale", "base"))
            mixed = [self.hc_pre(h[:, part], fn, scale, base) for part in parts]
            y = run(norm(torch.cat([x for x, _, _ in mixed], dim=1)))
            return torch.cat([self.hc_post(y[:, part], h[:, part], post, comb) for part, (_, post, comb) in zip(parts, mixed)], dim=1)

        h = sublayer("attn", self.attn_norm, lambda x: self.attn(x, start_pos))
        return sublayer("ffn", self.ffn_norm, lambda x: self.ffn(x, input_ids))
