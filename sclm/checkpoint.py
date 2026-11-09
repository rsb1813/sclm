# Reads the HF-format checkpoint: tensors by name, upstream (single-rank convert.py) conversions, and memory-mapped engram rows.
import json
import os
import struct

import numpy as np
import torch
from safetensors import safe_open


def read_header(path: str) -> tuple[dict, int]:
    """safetensors header of `path` and the byte offset where tensor data starts."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class Checkpoint:
    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles = {}

    def path(self, name: str) -> str:
        return os.path.join(self.root, self.weight_map[name])

    def get(self, name: str) -> torch.Tensor:
        path = self.path(name)
        handle = self._handles.get(path)
        if handle is None:
            handle = self._handles[path] = safe_open(path, framework="pt", device="cpu")
        return handle.get_tensor(name)

    def module_state(self, prefix: str, skip: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
        """Tensors under `prefix`, keyed relative to it, converted as upstream convert.py does for one rank:
        wo_a is dequantized from fp8 to bf16 and routed-expert int8 storage is reinterpreted as packed fp4."""
        names = [k for k in self.weight_map if k.startswith(prefix) and not any(s in k for s in skip)]
        state = {k[len(prefix) :]: self.get(k) for k in names}
        for key in [k for k in state if k.endswith("wo_a.weight")]:
            weight, scale = state[key], state.pop(key[: -len("weight")] + "scale")
            bo, bi = weight.size(0) // scale.size(0), weight.size(1) // scale.size(1)
            weight = weight.unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)).float() * scale[:, None, :, None].float()
            state[key] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
        for key, value in state.items():
            if ".experts." in key and "shared_experts" not in key and value.dtype == torch.int8:
                state[key] = value.view(torch.float4_e2m1fn_x2)
        return state

    def memmap(self, name: str) -> np.memmap:
        """Read-only uint8 view of a tensor's raw bytes, shaped like the tensor (1-byte dtypes only)."""
        path = self.path(name)
        header, data_start = read_header(path)
        info = header[name]
        assert info["dtype"] in ("F8_E4M3", "F8_E8M0", "U8", "I8"), info["dtype"]
        start, _ = info["data_offsets"]
        return np.memmap(path, dtype=np.uint8, mode="r", offset=data_start + start, shape=tuple(info["shape"]))


class MappedEngramEmbedding(torch.nn.Module):
    """Engram hash table left in the checkpoint file; only looked-up rows are read and moved to the GPU.
    Same output as upstream ParallelEngramEmbedding with world_size == 1."""

    def __init__(self, weight: np.memmap, scale: np.memmap, block_size: int = 32):
        super().__init__()
        self.weight, self.scale, self.block_size = weight, scale, block_size

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.reshape(-1).cpu().numpy()
        w = torch.from_numpy(self.weight[idx]).to(indices.device).view(torch.float8_e4m3fn)
        s = torch.from_numpy(self.scale[idx]).to(indices.device).view(torch.float8_e8m0fnu)
        values = (w.float().unflatten(-1, (-1, self.block_size)) * s.float().unsqueeze(-1)).flatten(-2)
        return values.to(torch.bfloat16).view(*indices.shape, -1)
