# ctypes binding of the engine core: a layer's RAM-tier experts computed from its sclm-vq-1 file (engine/core.cpp),
# and the SSD tier's file reader and cache policy (engine/ssd.cpp).
import ctypes as C
import json
import os
import struct

import numpy as np

DLL = os.path.join(os.path.dirname(__file__), "..", "engine", "build", "sclm_core.dll")


class Format(C.Structure):
    _fields_ = [("codebook", C.c_void_p), ("d", C.c_int), ("bits", C.c_int)]


class Expert(C.Structure):
    _fields_ = [("format", C.c_int), ("codes", C.c_void_p * 3), ("scales", C.c_void_p * 3)]


def _dll():
    lib = C.CDLL(os.path.abspath(DLL))
    lib.sclm_layer_new.restype = C.c_void_p
    lib.sclm_layer_new.argtypes = [C.c_int, C.c_int, C.c_float, C.POINTER(Format), C.c_int,
                                   C.c_void_p, C.c_int, C.c_void_p, C.c_int]
    lib.sclm_layer_free.argtypes = [C.c_void_p]
    lib.sclm_layer_forward.argtypes = [C.c_void_p, C.POINTER(Expert), C.c_void_p, C.c_int, C.c_void_p, C.c_void_p]
    lib.sclm_layer_start.argtypes = [C.c_void_p, C.POINTER(Expert), C.c_void_p, C.c_int, C.c_void_p]
    lib.sclm_layer_wait.argtypes = [C.c_void_p, C.c_void_p]
    lib.sclm_set_threads.argtypes = [C.c_int]
    lib.sclm_reader_new.restype = C.c_void_p
    lib.sclm_reader_new.argtypes = [C.POINTER(C.c_wchar_p), C.c_int, C.c_int]
    lib.sclm_reader_free.argtypes = [C.c_void_p]
    lib.sclm_reader_submit.argtypes = [C.c_void_p, C.c_int, C.c_uint64, C.c_uint64, C.c_void_p, C.c_int]
    lib.sclm_reader_poll.argtypes = [C.c_void_p]
    lib.sclm_reader_open.argtypes = [C.c_void_p, C.c_int]
    lib.sclm_reader_wait_ticket.argtypes = [C.c_void_p, C.c_int]
    lib.sclm_reader_wait.argtypes = [C.c_void_p]
    lib.sclm_cache_new.restype = C.c_void_p
    lib.sclm_cache_new.argtypes = [C.c_int, C.c_int, C.c_double]
    lib.sclm_cache_free.argtypes = [C.c_void_p]
    lib.sclm_cache_use.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_longlong, C.c_void_p, C.c_void_p]
    return lib


def set_threads(threads: int) -> None:
    """Threads that share each Layer.forward (1 at start). The output does not depend on it."""
    _dll().sclm_set_threads(threads)


def phase_seconds() -> list:
    """Seconds the engine has spent so far in each phase of the layer calls. Summed over the threads, which work on
    them at the same time: input quantization and rotation, input tables, w1 and w3, hidden vectors with their tables,
    w2. Then the sum, and by the clock the time from a call's start to its last expert."""
    out = (C.c_double * 7)()
    _dll().sclm_phase_seconds(out)
    return list(out)


class Layer:
    """The quantized routed experts of one layer file (codes.save_layer), held in RAM."""

    def __init__(self, path: str, dim: int, inter: int, swiglu_limit: float):
        self.lib, self.dim = _dll(), dim
        meta, self.arrays = read_file(path)  # the engine keeps pointers into these
        assert meta["format"] == "sclm-vq-1" and int(meta["tile"]) == 256
        options = json.loads(meta["options"])
        formats = (Format * len(options))()
        self.experts = {}  # expert index -> Expert
        for i, (opt, (d, bits)) in enumerate(options.items()):
            formats[i] = Format(self.arrays[f"{opt}.codebook"].ctypes.data, d, bits)
            for k, e in enumerate(self.arrays[f"{opt}.experts"].tolist()):
                rows = [self.arrays[f"{opt}.{w}.{part}"][k].ctypes.data for part in ("codes", "scales") for w in ("w1", "w2", "w3")]
                self.experts[e] = Expert(i, (C.c_void_p * 3)(*rows[:3]), (C.c_void_p * 3)(*rows[3:]))
        signs = [self.arrays[f"rot{k}.signs"] for k in (dim, inter)]
        self.handle = self.lib.sclm_layer_new(dim, inter, swiglu_limit, formats, len(options), signs[0].ctypes.data,
                                              int(meta[f"rot{dim}.block"]), signs[1].ctypes.data, int(meta[f"rot{inter}.block"]))

    def forward(self, experts: list, gates: list, x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """sum of gates[i] * expert experts[i] of x (fp32 [dim]), as MoE.forward sums one token's routed experts.
        Written into `out` (fp32 [dim]) when given."""
        ex = (Expert * len(experts))(*[self.experts[e] for e in experts])
        x, g = np.ascontiguousarray(x, np.float32), np.asarray(gates, np.float32)
        y = np.empty(self.dim, np.float32) if out is None else out
        self.lib.sclm_layer_forward(self.handle, ex, g.ctypes.data, len(experts), x.ctypes.data, y.ctypes.data)
        return y

    def start(self, experts: list, gates: list, x: np.ndarray) -> None:
        """Starts forward on the other threads and returns at once: the caller does other work, then takes the result
        with wait(). One call at a time over all layers (they share the threads)."""
        ex = (Expert * len(experts))(*[self.experts[e] for e in experts])
        x, g = np.ascontiguousarray(x, np.float32), np.asarray(gates, np.float32)
        self.started = (ex, x)  # the engine reads them until wait()
        self.lib.sclm_layer_start(self.handle, ex, g.ctypes.data, len(experts), x.ctypes.data)

    def wait(self, out: np.ndarray | None = None) -> np.ndarray:
        """Joins the call started and returns its result, written into `out` (fp32 [dim]) when given."""
        y = np.empty(self.dim, np.float32) if out is None else out
        self.lib.sclm_layer_wait(self.handle, y.ctypes.data)
        self.started = None
        return y

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.sclm_layer_free(self.handle)


PAGE = 4096


def pages(nbytes: int) -> np.ndarray:
    """A page-aligned uint8 buffer, as unbuffered reads need."""
    buf = np.empty(nbytes + PAGE, np.uint8)
    start = -buf.ctypes.data % PAGE
    return buf[start:start + nbytes]


class Reader:
    """Unbuffered overlapped reads from files (the checkpoint shards): nothing passes through the OS file cache."""

    def __init__(self, paths: list, chunk: int = 4 << 20):
        self.lib = _dll()
        self.handle = self.lib.sclm_reader_new((C.c_wchar_p * len(paths))(*paths), len(paths), chunk)
        assert self.handle, "a checkpoint file could not be opened"

    def submit(self, file: int, offset: int, nbytes: int, dst: np.ndarray, ticket: int = 0) -> int:
        """Starts reading bytes [offset, offset + nbytes) of paths[file] into dst (from pages(), nbytes + 2 pages or
        more, alive until the read is done). Returns where in dst the first byte lands. `ticket` (below 256) names the
        read for wait(ticket) and done(ticket); ticket 0 is for reads about to be waited for, which go ahead of the
        others on the SSD."""
        assert dst.ctypes.data % PAGE == 0 and dst.nbytes >= nbytes + 2 * PAGE
        return self.lib.sclm_reader_submit(self.handle, file, offset, nbytes, dst.ctypes.data, ticket)

    def wait(self, ticket: int | None = None) -> None:
        """Waits for every read submitted, or only for those submitted under the ticket."""
        if ticket is None:
            failed = self.lib.sclm_reader_wait(self.handle)
        else:
            failed = self.lib.sclm_reader_wait_ticket(self.handle, ticket)
        assert failed == 0, f"{failed} reads failed"

    def poll(self) -> None:
        """Checks, without waiting, that no read has failed."""
        failed = self.lib.sclm_reader_poll(self.handle)
        assert failed == 0, f"{failed} reads failed"

    def done(self, ticket: int) -> bool:
        return self.lib.sclm_reader_open(self.handle, ticket) == 0

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.sclm_reader_free(self.handle)


class Cache:
    """Which SSD-tier experts the VRAM cache holds: LRFU over every expert's history, with bypass."""

    def __init__(self, n_keys: int, slots: int, lam: float):
        self.lib = _dll()
        self.handle = self.lib.sclm_cache_new(n_keys, slots, lam)

    def use(self, keys: list, t: int) -> tuple[np.ndarray, np.ndarray]:
        """One layer's SSD-tier experts at token t -> (slot each is computed from, whether it has to be read first).
        Slots past the cache size are scratch (the expert is not kept)."""
        k, slot, miss = np.asarray(keys, np.int32), np.empty(len(keys), np.int32), np.empty(len(keys), np.int32)
        self.lib.sclm_cache_use(self.handle, k.ctypes.data, len(k), t, slot.ctypes.data, miss.ctypes.data)
        return slot, miss.astype(bool)

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.sclm_cache_free(self.handle)


ST_DTYPES = {"F32": np.float32, "F16": np.float16, "I32": np.int32, "I8": np.int8, "U8": np.uint8}


def read_file(path: str) -> tuple[dict, dict]:
    """(metadata, arrays by name) of a safetensors file, its data read in one piece into this process's memory.
    A buffered or memory-mapped read would leave a second copy in the OS file cache, and the RAM tier is most of
    the machine's memory."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta, size = header.pop("__metadata__"), os.path.getsize(path) - 8 - n
    buf = pages(size + 2 * PAGE)
    reader = Reader([path])
    head = reader.submit(0, 8 + n, size, buf)
    reader.wait()
    arrays = {}
    for name, info in header.items():
        start, end = info["data_offsets"]
        arrays[name] = buf[head + start:head + end].view(ST_DTYPES[info["dtype"]]).reshape(info["shape"])
    return meta, arrays
