# The upstream MoE with its routed experts on two tiers: vector-coded ones computed on the CPU by the engine core, and
# original FP4 ones computed by the upstream kernels from the device-memory cache.
from time import perf_counter

import numpy as np
import torch
from torch import nn

from sclm import gpu
from sclm.upstream import model as m


class TieredMoE(nn.Module):
    """Stands in for upstream MoE: the gate and the shared expert keep their parameter names, the routed experts have
    no parameters. `ram` (core.Layer) holds the layer's coded experts; every other expert comes from `cache`
    (ssd.ExpertCache), whose `token` the host sets to the position of the forward's first token.
    `after`: (the next block's TieredMoE, its ffn_norm weight and eps), for guessing its routing; the block sets
    `stream` (its hyper-connection stream and the mix of the FFN input) before each decode."""

    def __init__(self, layer_id: int, args: "m.ModelArgs"):
        super().__init__()
        self.layer_id, self.dim, self.inter, self.limit = layer_id, args.dim, args.moe_inter_dim, args.swiglu_limit
        self.gate = m.Gate(layer_id, args)
        self.shared_experts = m.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
        self.ram = self.cache = self.after = self.stream = self.bound = None

    def bind(self, buf, guess: int) -> tuple:
        """gpu.moe_begin and gpu.moe_end with this layer's arguments bound (gpu.bind). Their wrappers took 15 to 20 us
        each, and what the host does between the CPU tier's result and the next layer's first kernels is time the
        device stands idle."""
        gate, ex, ahead = self.gate, self.shared_experts, (None, None, None, None, None, 0, 0, 0.0, 0.0)
        w1, w2, w3 = ex.w1.weight, ex.w2.weight, ex.w3.weight
        n, dim = gate.weight.shape
        assert gate.score_func == "sqrtsoftplus" and dim == self.dim == w1.size(1) and buf.pack.numel() >= 2 * (gate.topk + guess) + dim
        if guess:
            later, norm_weight, eps = self.after
            h, pre = self.stream
            assert later.gate.score_func == "sqrtsoftplus" and later.gate.weight.shape == gate.weight.shape
            ahead = (later.gate.weight, later.gate.bias, h, pre, norm_weight, guess, pre.numel(), later.gate.gate_temp, eps)
        begin = gpu.bind("moe_begin", gate.weight, gate.bias, gpu.LATER, buf.pack, buf.host, n, gate.topk, dim, gate.gate_temp,
                         gate.route_scale, gate.norm_topk_prob, *ahead, w1, w1.scale, w2, w2.scale, w3, w3.scale, w1.size(0),
                         self.limit, buf.xq, buf.shared, buf.routed)  # the token
        end = gpu.bind("moe_end", buf.routed, self.cache.ram_out, buf.ram, buf.shared, buf.out, dim)
        self.bound = (begin, end, buf, guess)
        return self.bound

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """forward for one token (bf16 [dim]) on the engine's GPU kernels. The host waits for the device once, to
        learn the routing and take x for the CPU tier; while the CPU computes its tier, the device computes the shared
        expert (queued before that wait, but not waited for) and the routed experts it already holds."""
        cache, topk, stats, t0 = self.cache, self.gate.topk, self.cache.stats, perf_counter()
        ahead = max(cache.guess, cache.touch)
        guess = ahead if self.after else 0
        buf = gpu.moe_buffers(topk, self.dim, ahead, x.device)
        # with guess: what the next layer's gate makes of this layer's input, best first: of the first cache.guess, the
        # SSD experts are read a layer early; of the first cache.touch, the CPU-tier experts' codes are read into the
        # CPU's cache
        bound = self.bound
        begin, end, _, _ = bound if bound is not None and bound[2] is buf and bound[3] == guess else self.bind(buf, guess)
        later = self.after[0] if guess else None
        begin(x.data_ptr())
        cache.uploading = False  # the routing was copied after every slot upload queued so far
        host, xq, y = buf.host.numpy(), buf.xq, buf.routed
        t1 = perf_counter()
        ids, gates = host[:topk].astype(np.int64).tolist(), host[topk:2 * topk].tolist()
        total = sum(gates)
        if cache.topk:  # lossy: the weakest picks are dropped (the gate lists them best first)
            ids, gates = ids[:cache.topk], gates[:cache.topk]
        ram = [k for k, e in enumerate(ids) if e in self.ram.experts and gates[k] >= cache.ram_share * total]
        ssd = [k for k, e in enumerate(ids) if e not in self.ram.experts]
        wait = [gates[k] >= cache.wait_share * total for k in ssd] if cache.wait_share else None
        slots, pending = cache.request(self.layer_id, [ids[k] for k in ssd], cache.token, wait)
        if guess:
            gpu.moe_guess()  # as a rule it has arrived while the reads above were started
            picks = host[2 * topk + self.dim:2 * topk + self.dim + guess].astype(np.int64).tolist()
            cache.read_ahead(later.layer_id, [e for e in picks[:cache.guess] if e not in later.ram.experts])
        if wait:  # lossy: the token goes on without the light experts that are not in the cache yet
            ssd, slots = [k for k, s in zip(ssd, slots) if s is not None], [s for s in slots if s is not None]
        kept = sum(gates[k] for k in ram + ssd)
        if kept < total and kept > 0:  # what is computed keeps the routing weight of all six
            gates = [g * (total / kept) for g in gates]
        reading = {slot for slot, *_ in pending}
        held = [(slot, gates[k]) for k, slot in zip(ssd, slots) if slot not in reading]
        read = [(slot, gates[k]) for k, slot in zip(ssd, slots) if slot in reading]
        if held:
            gpu.fp4_experts(cache.store, *zip(*held), self.dim, self.inter, self.limit, xq, y)
        self.ram.forward([ids[k] for k in ram], [gates[k] for k in ram], host[2 * topk:2 * topk + self.dim],
                         cache.ram_out.numpy())
        if guess and cache.touch:  # the CPU's threads and the memory idle until the next layer's routing is known
            named = [e for e in picks[:cache.touch] if e in later.ram.experts]
            if named:
                later.ram.touch(named)
        t2 = perf_counter()
        cache.finish(pending)
        t3 = perf_counter()
        if read:
            gpu.fp4_experts(cache.store, *zip(*read), self.dim, self.inter, self.limit, xq, y)
        # the CPU tier's sum is not waited for: the device takes the buffer before the next layer's routing comes
        # back, and the CPU tier writes it again only after that
        end()
        out = buf.out
        t4 = perf_counter()
        for key, value in (("device_s", t1 - t0), ("ram_s", t2 - t1), ("read_s", t3 - t2), ("launch_s", t4 - t3),
                           ("ram_calls", len(ram)), ("ssd_calls", len(ssd)),
                           ("reads", sum(ticket == 0 for *_, ticket in pending))):
            stats[key] += value
        return out

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        if gpu.active and x.numel() == self.dim and self.gate.score_func == "sqrtsoftplus" and image_mask is None:
            return self.decode(x.reshape(-1)).view(shape)
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        stats, t0 = self.cache.stats, perf_counter()
        # moving x to the host waits for the device to finish everything queued before this layer's experts
        x_host, rows, gates = x.float().cpu().numpy(), indices.tolist(), weights.tolist()
        t1 = perf_counter()
        stats["device_s"] += t1 - t0
        for r, ids in enumerate(rows):  # one token at a time, in decode order
            ram = [k for k, e in enumerate(ids) if e in self.ram.experts]
            ssd = [k for k, e in enumerate(ids) if e not in self.ram.experts]
            slots, pending = self.cache.request(self.layer_id, [ids[k] for k in ssd], self.cache.token + r)
            if ram:  # the CPU computes while the SSD reads
                out = self.ram.forward([ids[k] for k in ram], [gates[r][k] for k in ram], x_host[r])
                y[r] += torch.from_numpy(out).to(y.device)
            t2 = perf_counter()
            self.cache.finish(pending)
            t3 = perf_counter()
            if gpu.active:  # straight from the slots' FP4 bytes
                xq = gpu.act_quant(x[r])
                if ssd:
                    gpu.fp4_experts(self.cache.store, slots, [gates[r][k] for k in ssd], self.dim, self.inter, self.limit, xq, y[r])
            else:
                for k, slot in zip(ssd, slots):
                    y[r] += m.Expert.forward(self.cache.experts[slot], x[r:r + 1], weights[r, k].view(1, 1))[0]
            t4 = perf_counter()
            stats["ram_s"] += t2 - t1  # includes starting the reads
            stats["read_s"] += t3 - t2  # what is left of the reads after the CPU tier, and the copies into the slots
            stats["launch_s"] += t4 - t3  # queueing the device's expert kernels (they run asynchronously)
            stats["ram_calls"] += len(ram)
            stats["ssd_calls"] += len(ssd)
            stats["reads"] += sum(ticket == 0 for *_, ticket in pending)
            t1 = t4
        if gpu.active and len(rows) == 1:
            gpu.fp8_expert(self.shared_experts, self.limit, xq, y[0])
        else:
            y += self.shared_experts(x)
        return y.type_as(x).view(shape)
