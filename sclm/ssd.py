# SSD tier: a routed expert's original bytes (FP4 weights and their scales) read from the checkpoint into a staging
# buffer, and the device-memory cache of such experts that the upstream kernels compute from.
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

from sclm import core
from sclm.checkpoint import Checkpoint
from sclm.upstream import model as m


class ExpertReader:
    """An expert lies in its layer's shard as two runs: w1, w2, w3 scales back to back, and far from them the three
    packed-FP4 weights back to back. Both are read as they are, with the engine's unbuffered reader."""

    def __init__(self, ck: Checkpoint, chunk: int = 4 << 20):
        self.ck = ck
        paths = sorted({ck.path(k) for k in ck.weight_map if ".ffn.experts." in k})
        self.file = {p: i for i, p in enumerate(paths)}
        self.reader = core.Reader(paths, chunk)
        self.sizes = [n for _, _, n in self.extents(0, 0)]  # the same for every expert
        self.nbytes = sum(self.sizes) + 4 * core.PAGE
        self.where = {}  # (layer, expert) -> [(file index, offset)] of its two runs, looked up once

    def extents(self, layer: int, expert: int) -> list:
        """[(shard path, file offset, byte count)] of the expert's scales and of its weights."""
        prefix = f"layers.{layer}.ffn.experts.{expert}."
        return [self.ck.extent([f"{prefix}{w}.{part}" for w in ("w1", "w2", "w3")]) for part in ("scale", "weight")]

    def staging(self) -> np.ndarray:
        return core.pages(self.nbytes)

    def submit(self, layer: int, expert: int, buf: np.ndarray) -> tuple[int, int]:
        """Starts reading the expert into buf (from staging()). Returns where its scales and its weights begin in buf,
        valid once wait() returns."""
        runs = self.where.get((layer, expert))
        if runs is None:
            found = self.extents(layer, expert)
            assert [n for _, _, n in found] == self.sizes
            runs = self.where[layer, expert] = [(self.file[path], offset) for path, offset, _ in found]
        at, starts = 0, []
        for (file, offset), size in zip(runs, self.sizes):
            room = size + 2 * core.PAGE
            starts.append(at + self.reader.submit(file, offset, size, buf[at:at + room]))
            at += room
        return starts[0], starts[1]

    def wait(self) -> None:
        self.reader.wait()


class ExpertCache:
    """Original FP4 experts held in device memory: `slots` of them chosen by the engine's policy (core.Cache), plus
    `scratch` slots for experts the policy does not keep. A slot is the expert's bytes as they lie in the checkpoint
    (scales, then weights), and the upstream kernels compute from views of them."""

    def __init__(self, reader: ExpertReader, n_layers: int, n_experts: int, dim: int, inter: int, limit: float,
                 slots: int, lam: float, device, scratch: int):
        self.reader, self.n_experts = reader, n_experts
        self.token = 0  # position of the first token of the forward in progress, set by the host
        self.stats = defaultdict(float)  # seconds and counts the tiered MoE adds up, for whoever measures
        self.uploading = False  # slot uploads may still be reading the staging buffers
        self.policy = core.Cache(n_layers * n_experts, slots, lam)
        n_scale, n_weight = reader.sizes
        self.store = torch.empty(slots + scratch, n_scale + n_weight, dtype=torch.uint8, device=device)
        pin = torch.device(device).type == "cuda"
        self.ram_out = torch.empty(dim, dtype=torch.float32, pin_memory=pin)  # where the CPU tier writes a token's sum
        self.staging = []  # page-aligned host buffers the reader fills, one per expert a call can miss
        for _ in range(scratch):
            buf = torch.empty(reader.nbytes + core.PAGE, dtype=torch.uint8, pin_memory=pin)
            start = -buf.data_ptr() % core.PAGE
            self.staging.append(buf[start:start + reader.nbytes])
        self.experts = []  # slot -> what upstream Expert.forward needs, as views of the slot
        for row in self.store:
            at_s, at_w, mats = 0, n_scale, {}
            for name, (out, k) in (("w1", (inter, dim)), ("w2", (dim, inter)), ("w3", (inter, dim))):
                weight = row[at_w:at_w + out * k // 2].view(torch.float4_e2m1fn_x2).view(out, k // 2)
                weight.scale = row[at_s:at_s + out * k // m.fp4_block_size].view(m.scale_dtype).view(out, k // m.fp4_block_size)
                mats[name] = lambda v, w=weight: m.linear(v, w)
                at_s, at_w = at_s + out * k // m.fp4_block_size, at_w + out * k // 2
            assert (at_s, at_w) == (n_scale, n_scale + n_weight)
            self.experts.append(SimpleNamespace(**mats, swiglu_limit=limit))

    def request(self, layer: int, experts: list, token: int) -> tuple[list, list]:
        """Slots of a layer's SSD-tier experts at a token, with the reads of the missing ones started.
        Returns (slots, pending); finish(pending) has to run before the slots are used."""
        if self.uploading:  # the reads below go into the buffers those uploads copy from
            torch.cuda.synchronize()
            self.uploading = False
        slots, miss = self.policy.use([layer * self.n_experts + e for e in experts], token)
        pending = []
        for e, slot, buf in zip(np.asarray(experts)[miss].tolist(), slots[miss].tolist(), self.staging):
            pending.append((slot, buf, self.reader.submit(layer, e, buf.numpy())))
        return slots.tolist(), pending

    def finish(self, pending: list) -> None:
        """Waits for the reads and copies each expert into its slot."""
        if not pending:
            return
        self.reader.wait()
        n_scale, n_weight = self.reader.sizes
        for slot, buf, (at_s, at_w) in pending:
            self.store[slot, :n_scale].copy_(buf[at_s:at_s + n_scale], non_blocking=True)
            self.store[slot, n_scale:].copy_(buf[at_w:at_w + n_weight], non_blocking=True)
        self.uploading = self.store.is_cuda
