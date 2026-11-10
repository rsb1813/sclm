# The upstream MoE with its routed experts on two tiers: vector-coded ones computed on the CPU by the engine core, and
# original FP4 ones computed by the upstream kernels from the device-memory cache.
import torch
from torch import nn

from sclm.upstream import model as m


class TieredMoE(nn.Module):
    """Stands in for upstream MoE: the gate and the shared expert keep their parameter names, the routed experts have
    no parameters. `ram` (core.Layer) holds the layer's coded experts; every other expert comes from `cache`
    (ssd.ExpertCache), whose `token` the host sets to the position of the forward's first token."""

    def __init__(self, layer_id: int, args: "m.ModelArgs"):
        super().__init__()
        self.layer_id, self.dim = layer_id, args.dim
        self.gate = m.Gate(layer_id, args)
        self.shared_experts = m.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
        self.ram = self.cache = None

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        x_host, rows, gates = x.float().cpu().numpy(), indices.tolist(), weights.tolist()
        for r, ids in enumerate(rows):  # one token at a time, in decode order
            ram = [k for k, e in enumerate(ids) if e in self.ram.experts]
            ssd = [k for k, e in enumerate(ids) if e not in self.ram.experts]
            slots, pending = self.cache.request(self.layer_id, [ids[k] for k in ssd], self.cache.token + r)
            if ram:  # the CPU computes while the SSD reads
                out = self.ram.forward([ids[k] for k in ram], [gates[r][k] for k in ram], x_host[r])
                y[r] += torch.from_numpy(out).to(y.device)
            self.cache.finish(pending)
            for k, slot in zip(ssd, slots):
                y[r] += m.Expert.forward(self.cache.experts[slot], x[r:r + 1], weights[r, k].view(1, 1))[0]
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)
