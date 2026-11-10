# ctypes binding of the CPU engine (engine/core.cpp): a layer's RAM-tier experts computed from its sclm-vq-1 file.
import ctypes as C
import json
import os

import numpy as np
from safetensors import safe_open

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
    return lib


class Layer:
    """The quantized routed experts of one layer file (codes.save_layer), held in RAM."""

    def __init__(self, path: str, dim: int, inter: int, swiglu_limit: float):
        self.lib, self.dim = _dll(), dim
        with safe_open(path, "np") as f:
            meta = f.metadata()
            assert meta["format"] == "sclm-vq-1" and int(meta["tile"]) == 256
            self.arrays = {k: np.ascontiguousarray(f.get_tensor(k)) for k in f.keys()}  # the engine keeps pointers into these
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

    def forward(self, experts: list, gates: list, x: np.ndarray) -> np.ndarray:
        """sum of gates[i] * expert experts[i] of x (fp32 [dim]), as MoE.forward sums one token's routed experts."""
        ex = (Expert * len(experts))(*[self.experts[e] for e in experts])
        x, g, y = np.ascontiguousarray(x, np.float32), np.asarray(gates, np.float32), np.empty(self.dim, np.float32)
        self.lib.sclm_layer_forward(self.handle, ex, g.ctypes.data, len(experts), x.ctypes.data, y.ctypes.data)
        return y

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.sclm_layer_free(self.handle)
