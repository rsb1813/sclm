# SSD tier: a routed expert's original bytes (FP4 weights and their scales) read from the checkpoint into a staging
# buffer, and the device-memory cache of such experts that the upstream kernels compute from.
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

from sclm import core, gpu
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

    def submit(self, layer: int, expert: int, buf: np.ndarray, ticket: int = 0) -> tuple[int, int]:
        """Starts reading the expert into buf (from staging()). Returns where its scales and its weights begin in buf,
        valid once wait() returns (ticket 0) or the reader says the ticket is done."""
        runs = self.where.get((layer, expert))
        if runs is None:
            found = self.extents(layer, expert)
            assert [n for _, _, n in found] == self.sizes
            runs = self.where[layer, expert] = [(self.file[path], offset) for path, offset, _ in found]
        at, starts = 0, []
        for (file, offset), size in zip(runs, self.sizes):
            room = size + 2 * core.PAGE
            starts.append(at + self.reader.submit(file, offset, size, buf[at:at + room], ticket))
            at += room
        return starts[0], starts[1]

    def wait(self) -> None:
        """Waits for the reads submitted under ticket 0."""
        self.reader.wait(0)


class ExpertCache:
    """Original FP4 experts held in device memory: `slots` of them chosen by the engine's policy (core.Cache), plus
    `scratch` slots for experts the policy does not keep. A slot is the expert's bytes as they lie in the checkpoint
    (scales, then weights), and the upstream kernels compute from views of them. `background`: staging buffers for
    experts read without anyone waiting (request's wait argument, read_ahead)."""

    def __init__(self, reader: ExpertReader, n_layers: int, n_experts: int, dim: int, inter: int, limit: float,
                 slots: int, lam: float, device, scratch: int, background: int = 0):
        self.reader, self.n_experts, self.n_slots = reader, n_experts, slots
        self.loaded = [None] * slots  # slot -> the key whose bytes it holds (or is being given by a waited read)
        self.held = {}  # the same, key -> slot
        self.loads = {}  # staging buffer -> (slot, key, where the bytes land) of the background read filling it
        # key -> (staging buffer, where the bytes land) of a read started on a guess (read_ahead), under way or done
        self.ahead = {}
        self.idle = list(range(scratch, scratch + background))  # staging buffers free for a background read
        self.cooling = []  # buffers whose upload was queued by the last request: idle again at the next one
        self.token = 0  # position of the first token of the forward in progress, set by the host
        self.stats = defaultdict(float)  # seconds and counts the tiered MoE adds up, for whoever measures
        self.uploading = False  # slot uploads may still be reading the staging buffers
        # the decode path (sclm.tiered) turns this on: copies into and out of the slots run on the engine's copy
        # stream (gpu.copy), beside the kernels instead of in line with them. `fresh`: the slots such copies write,
        # until barrier()
        self.beside, self.fresh = False, set()
        # lossy speed options of the decode path (sclm.tiered), off by default
        self.topk = 0  # > 0: only the best `topk` routed experts of a token are computed
        # > 0: an expert that has to be read is waited for only if it carries at least this share of the token's
        # routing weight; otherwise the token goes on without it and it is read in the background
        self.wait_share = 0.0
        self.ram_share = 0.0  # > 0: a CPU-tier expert with less than this share of the routing weight is left out
        self.guess = 0  # > 0: a layer's gate guesses this many experts of the next layer, which are read ahead
        # > 0: the CPU-tier experts among this many guessed ones have their codes read into the CPU's cache while the
        # device computes the attention before them (core.Layer.touch)
        self.touch = 0
        self.ram_cache = {}  # key -> host buffer holding the expert's bytes (keep_in_ram), the oldest first
        self.ram_free = []  # the host buffers of keep_in_ram that hold nothing
        self.policy = core.Cache(n_layers * n_experts, slots, lam)
        n_scale, n_weight = reader.sizes
        self.store = torch.empty(slots + scratch, n_scale + n_weight, dtype=torch.uint8, device=device)
        pin = torch.device(device).type == "cuda"
        self.ram_out = torch.empty(dim, dtype=torch.float32, pin_memory=pin)  # where the CPU tier writes a token's sum
        self.staging = []  # page-aligned host buffers the reader fills, one per expert a call can miss
        for _ in range(scratch + background):
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

    def _upload(self, slot: int, buf: torch.Tensor, at_s: int, at_w: int) -> None:
        n_scale, n_weight = self.reader.sizes
        if self.beside:
            to, source = self.store.data_ptr() + slot * self.store.size(1), buf.data_ptr()
            gpu.copy(to, source + at_s, n_scale)
            gpu.copy(to + n_scale, source + at_w, n_weight)
            self.fresh.add(slot)
        else:
            self.store[slot, :n_scale].copy_(buf[at_s:at_s + n_scale], non_blocking=True)
            self.store[slot, n_scale:].copy_(buf[at_w:at_w + n_weight], non_blocking=True)
        self.uploading = self.store.is_cuda

    def _move(self, slot: int, buf: torch.Tensor, out: bool) -> None:
        """Queues the copy of a whole expert between its slot and a host buffer of the RAM level (out: to the host)."""
        if self.beside:
            at = self.store.data_ptr() + slot * self.store.size(1)
            gpu.copy(*((buf.data_ptr(), at) if out else (at, buf.data_ptr())), self.store.size(1))
            if not out:
                self.fresh.add(slot)
        elif out:
            buf.copy_(self.store[slot], non_blocking=True)
        else:
            self.store[slot].copy_(buf, non_blocking=True)

    def barrier(self) -> None:
        """The kernels launched from now on find the bytes of every slot copy queued so far in place."""
        if self.fresh:
            gpu.copies_wait()
            self.fresh.clear()

    def keep_in_ram(self, n: int) -> None:
        """Adds a second level of n experts in host memory (pinned, an expert's bytes each): an expert that leaves
        its slot is copied there by the device, and while it is among the n the level holds, request() takes it
        from there instead of reading it from the SSD. It stays in the level when it is copied back to a slot (as
        the newest), so only an expert that came from the SSD is copied when it leaves. The host never touches these
        buffers: the device fills and reads them in the order the copies were queued, so nothing has to be waited
        for."""
        row = self.store.size(1)
        while n > 0:  # torch rounds a pinned allocation up to a power of two: blocks that fill one, of at most 1 GiB
            size = 1 << 30
            while size // 2 >= row and size // row > n:
                size //= 2
            block = torch.empty(min(n, size // row), row, dtype=torch.uint8, pin_memory=self.store.is_cuda)
            self.ram_free += block.unbind(0)
            n -= block.size(0)

    def _give(self, slot: int, key: int | None) -> None:
        """The slot holds, or is being given, the bytes of `key` from now on (None: of nobody). The expert it held
        goes to the RAM level (keep_in_ram), into a free buffer or that of the expert that left longest ago: this
        has to be queued before the copy of the new bytes into the slot."""
        old = self.loaded[slot]
        self.held.pop(old, None)
        if old is not None and old not in self.ram_cache and (self.ram_free or self.ram_cache):
            buf = self.ram_free.pop() if self.ram_free else self.ram_cache.pop(next(iter(self.ram_cache)))
            self._move(slot, buf, out=True)
            self.ram_cache[old] = buf
        self.loaded[slot] = key
        if key is not None:
            self.held[key] = slot

    def read_ahead(self, layer: int, experts: list) -> None:
        """Starts reading experts the layer is expected to ask for (a guess made a layer early), unless they are in
        the cache (either level) or being read already. They wait in staging buffers, where request() finds them; a
        guess that does not come true costs its read and nothing else. Called after the token's request() for the
        layer before."""
        loading = {key for _, key, _ in self.loads.values()}
        for e in experts:
            key = layer * self.n_experts + e
            if key in self.held or key in self.ahead or key in loading or key in self.ram_cache:
                continue
            if not self.idle:  # the oldest guess nobody asked for gives up its buffer, if its read is over
                self.reader.reader.poll()
                old = next(iter(self.ahead), None)
                if old is None or not self.reader.reader.done(self.ahead[old][0]):
                    return
                self.idle.append(self.ahead.pop(old)[0])
            i = self.idle.pop()
            self.ahead[key] = (i, self.reader.submit(layer, e, self.staging[i].numpy(), i))
            self.stats["ahead"] += 1

    def request(self, layer: int, experts: list, token: int, wait: list | None = None) -> tuple[list, list]:
        """Slots of a layer's SSD-tier experts at a token, with the reads of the missing ones started.
        Returns (slots, pending); finish(pending) has to run before the slots of `pending` (its first field) are used.
        An expert read ahead is taken from its staging buffer, at once if its read is over; one that keep_in_ram
        holds is copied back from there, whether it is waited for or not.
        wait[i] False (lossy): if the expert's bytes are not in its slot, its slot comes back None; it is read in the
        background when a staging buffer is free, and is there for the tokens after."""
        if self.uploading:  # the reads below go into the buffers those uploads copy from
            torch.cuda.synchronize()
            self.uploading = False
        self.idle += self.cooling
        self.cooling = []
        if self.loads:
            self.reader.reader.poll()
            for i in [i for i in self.loads if self.reader.reader.done(i)]:
                slot, key, (at_s, at_w) = self.loads.pop(i)
                if self.loaded[slot] is None:  # else a waited read has given the slot to another expert since
                    self._upload(slot, self.staging[i], at_s, at_w)
                    self._give(slot, key)
                self.cooling.append(i)
        keys = [layer * self.n_experts + e for e in experts]
        slots = self.policy.use(keys, token)[0].tolist()
        out, pending, loading, waited = [], [], {slot for slot, _, _ in self.loads.values()}, 0
        if self.ahead:
            self.reader.reader.poll()
        for k, (e, key, slot) in enumerate(zip(experts, keys, slots)):
            kept = slot < self.n_slots  # not a scratch slot
            if kept and self.loaded[slot] == key:
                out.append(slot)
            elif key in self.ram_cache:  # it left its slot not long ago: the device copies it back, nobody waits
                buf = self.ram_cache.pop(key)
                if kept:
                    self._give(slot, key)
                # it stays, as the newest: when it leaves its slot again, nothing has to be copied (a copy back to
                # the host shares the memory's bandwidth with the CPU tier)
                self.ram_cache[key] = buf
                self._move(slot, buf, out=False)
                # listed with the experts being read (no ticket: finish() has nothing to do for it), so that the
                # token adds its experts in the order it would without the RAM level, and its result is the same
                pending.append((slot, buf, None, None))
                out.append(slot)
                self.stats["ram_cache_hits"] += 1
            elif key in self.ahead and (self.reader.reader.done(self.ahead[key][0]) or wait is None or wait[k]):
                i, (at_s, at_w) = self.ahead.pop(key)
                if kept:
                    self._give(slot, key)
                if self.reader.reader.done(i):
                    self._upload(slot, self.staging[i], at_s, at_w)
                else:
                    pending.append((slot, self.staging[i], (at_s, at_w), i))
                self.cooling.append(i)
                out.append(slot)
                self.stats["ahead_used"] += 1
            elif key in self.ahead:  # lossy, and still arriving: it goes on into its slot as a background read
                out.append(None)
                if kept and slot not in loading:
                    i, starts = self.ahead.pop(key)
                    self._give(slot, None)
                    self.loads[i] = (slot, key, starts)
                    loading.add(slot)
                    self.stats["loads"] += 1
            elif wait is None or wait[k]:
                buf, waited = self.staging[waited], waited + 1
                pending.append((slot, buf, self.reader.submit(layer, e, buf.numpy()), 0))
                if kept:
                    self._give(slot, key)
                out.append(slot)
            else:
                out.append(None)
                # a slot whose previous owner is still arriving waits its turn: the expert is asked for again
                if kept and slot not in loading and self.idle:
                    i = self.idle.pop()
                    self._give(slot, None)
                    self.loads[i] = (slot, key, self.reader.submit(layer, e, self.staging[i].numpy(), i))
                    loading.add(slot)
                    self.stats["loads"] += 1
        return out, pending

    def finish(self, pending: list) -> None:
        """Waits for the reads (each named by its ticket, the last field) and copies each expert into its slot."""
        if not pending:
            return
        pending = [p for p in pending if p[-1] is not None]  # (the others came from RAM: their copies are queued)
        for ticket in {ticket for *_, ticket in pending}:
            self.reader.reader.wait(ticket)
        for slot, buf, (at_s, at_w), _ in pending:
            self._upload(slot, buf, at_s, at_w)


class EngramRows(torch.nn.Module):
    """Rows of an engram hash table read from the checkpoint with the engine's unbuffered reader. A decode token
    needs a few dozen rows scattered over a file of about 100 GB; mapping the file and faulting them in took 3 ms a
    layer. Same output as upstream ParallelEngramEmbedding for one rank."""

    tables = None  # the fp32 value of every E4M3 byte and of every E8M0 byte

    def __init__(self, ck: Checkpoint, prefix: str, block_size: int = 32):
        super().__init__()
        (weight, self.at_w, n_w), (scale, self.at_s, n_s) = ck.extent([prefix + "weight"]), ck.extent([prefix + "scale"])
        rows = ck._header(weight)[0][prefix + "weight"]["shape"][0]
        self.dim, self.cols, self.block_size = n_w // rows, n_s // rows, block_size
        self.reader = core.Reader([weight, scale])
        self.room = self.dim + 2 * core.PAGE + -self.dim % core.PAGE  # a row's read, start rounded down and end up
        self.buf = core.pages(0)
        self.started = None  # (the rows, where they land, the device for the result) of reads start() has under way
        self.plan = None  # start()'s constant arguments for the number of rows it was last given
        self.pin = self.bits = self.dev = None  # decoded()'s host buffer (page-locked), its bits, the device's copy

    def start(self, indices, device=None) -> None:
        """Starts reading the rows forward(indices) returns (a tensor, or a list of row numbers). A decode token
        knows them before its first layer, and the layers before the engram's hide the read (0.85 ms). device: where
        forward puts those rows (default: where its indices are), so that the indices can stay on the host."""
        rows, room = indices if isinstance(indices, list) else indices.reshape(-1).tolist(), self.room
        if self.started is not None:  # reads nobody took are still landing in the buffer
            self.reader.wait()
        n = len(rows)
        if self.buf.nbytes < 2 * n * room:
            self.buf = core.pages(2 * n * room)
        if self.plan is None or len(self.plan[0]) != 2 * n:  # of a row: its weights from file 0, its scales from file 1
            self.plan = (np.tile(np.array([0, 1], np.int32), n), np.tile(np.array([self.dim, self.cols], np.uint64), n),
                         np.arange(2 * n, dtype=np.uint64) * np.uint64(room))
        files, sizes, at = self.plan
        r, offsets = np.asarray(rows, np.uint64), np.empty(2 * n, np.uint64)
        offsets[0::2], offsets[1::2] = self.at_w + r * np.uint64(self.dim), self.at_s + r * np.uint64(self.cols)
        heads = self.reader.submit_many(files, offsets, sizes, at + np.uint64(self.buf.ctypes.data))
        land = (at.astype(np.int64) + heads).tolist()  # where each row's weights and scales land in the buffer
        self.started = (rows, list(zip(land[0::2], land[1::2])), device)

    def forward(self, indices) -> torch.Tensor:
        """The rows as upstream's embedding returns them (bf16 [*indices.shape, dim]). Given the list of row numbers
        start() was given (a decode token), they are decoded on the host: [1, 1, rows, dim]."""
        if isinstance(indices, list):
            assert self.started is not None and self.started[0] == indices
            return self.decoded().view(1, 1, len(indices), self.dim)
        if self.started is None or indices.is_cuda or self.started[0] != indices.reshape(-1).tolist():
            self.start(indices)  # other rows than the ones under way (which the reader still finishes)
        (_, at, device), self.started = self.started, None
        device = device or indices.device
        self.reader.wait()
        w = torch.from_numpy(np.stack([self.buf[a:a + self.dim] for a, _ in at])).to(device).view(torch.float8_e4m3fn)
        s = torch.from_numpy(np.stack([self.buf[b:b + self.cols] for _, b in at])).to(device).view(torch.float8_e8m0fnu)
        values = (w.float().unflatten(-1, (-1, self.block_size)) * s.float().unsqueeze(-1)).flatten(-2)
        return values.to(torch.bfloat16).view(*indices.shape, -1)

    def decoded(self) -> torch.Tensor:
        """The rows start() is reading, as forward computes them but on the host and flat (bf16 [rows * dim]), sent
        to the device without waiting. forward's two copies from pageable memory each wait for the device, and its
        conversions are half a dozen small device operations; here a byte is a table lookup. The same values: an fp8
        weight times a power-of-two scale is exact, and so is its bf16 unless it is subnormal (then it is rounded to
        nearest even, as torch rounds)."""
        (rows, at, device), self.started = self.started, None
        self.reader.wait()
        n = len(rows)
        if self.pin is None or self.pin.numel() != n * self.dim:
            if EngramRows.tables is None:
                codes = torch.arange(256, dtype=torch.uint8)
                EngramRows.tables = tuple(codes.view(t).float().numpy() for t in (torch.float8_e4m3fn, torch.float8_e8m0fnu))
            self.pin = torch.empty(n * self.dim, dtype=torch.bfloat16, pin_memory=torch.device(device).type == "cuda")
            self.bits = self.pin.view(torch.int16).numpy().view(np.uint16)
            self.dev = torch.empty(n * self.dim, dtype=torch.bfloat16, device=device)
        e4m3, e8m0 = EngramRows.tables
        w = np.stack([self.buf[a:a + self.dim] for a, _ in at])
        s = np.stack([self.buf[b:b + self.cols] for _, b in at])
        u = (e4m3[w].reshape(n, self.cols, self.block_size) * e8m0[s][:, :, None]).reshape(-1).view(np.uint32)
        self.bits[:] = (u + np.uint32(0x7fff) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
        self.dev.copy_(self.pin, non_blocking=True)
        return self.dev
