# Reads the HF-format checkpoint: tensors by name, upstream (single-rank convert.py) conversions, and memory-mapped engram rows.
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

ST_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
}


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
        self._headers: dict[str, tuple[dict, int]] = {}

    def path(self, name: str) -> str:
        return os.path.join(self.root, self.weight_map[name])

    def _header(self, path: str) -> tuple[dict, int]:
        if path not in self._headers:
            self._headers[path] = read_header(path)
        return self._headers[path]

    def get(self, name: str) -> torch.Tensor:
        """One tensor read into fresh CPU memory with plain file reads. Memory-mapping whole shards (safe_open)
        keeps every touched page resident; over 40 shards that exhausted RAM and crashed with an access violation."""
        path = self.path(name)
        header, data_start = self._header(path)
        info = header[name]
        start, end = info["data_offsets"]
        buf = torch.empty(end - start, dtype=torch.uint8, device="cpu")
        view = memoryview(buf.numpy())
        with open(path, "rb", buffering=0) as f:
            f.seek(data_start + start)
            done = 0
            while done < len(view):  # raw reads may return short on large requests
                n = f.readinto(view[done:])
                assert n, f"unexpected EOF reading {name}"
                done += n
        return buf.view(ST_DTYPES[info["dtype"]]).reshape(info["shape"])

    def extent(self, names: list[str]) -> tuple[str, int, int]:
        """(shard path, file offset, byte count) of tensors stored back to back in one shard, in this order."""
        path = self.path(names[0])
        header, data_start = self._header(path)
        spans = [header[n]["data_offsets"] for n in names]
        assert all(self.path(n) == path for n in names) and all(a[1] == b[0] for a, b in zip(spans, spans[1:])), names
        return path, data_start + spans[0][0], spans[-1][1] - spans[0][0]

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
        header, data_start = self._header(path)
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
        # Rows are scattered over a ~100GB file, so reading them one page fault at a time runs at SSD queue depth 1
        # (~5K rows/s here). Deduplicate, sort, and fault pages in from several threads (~65K rows/s).
        rows, inverse = np.unique(indices.reshape(-1).cpu().numpy(), return_inverse=True)
        # A view of its own for this lookup: the pages a view faults in stay in the working set until it is unmapped
        # (0.2GiB per 2048-token sequence, which ran 247 calibration sequences out of RAM)
        weight, scale = (np.memmap(a.filename, dtype=a.dtype, mode="r", offset=a.offset, shape=a.shape)
                         for a in (self.weight, self.scale))
        parts = list(_row_reader().map(lambda c: (weight[c], scale[c]), np.array_split(rows, ROW_CHUNKS)))
        del weight, scale
        w = torch.from_numpy(np.concatenate([p[0] for p in parts])).to(indices.device).view(torch.float8_e4m3fn)
        s = torch.from_numpy(np.concatenate([p[1] for p in parts])).to(indices.device).view(torch.float8_e8m0fnu)
        values = (w.float().unflatten(-1, (-1, self.block_size)) * s.float().unsqueeze(-1)).flatten(-2)
        values = values.to(torch.bfloat16)[torch.from_numpy(inverse).to(indices.device)]
        return values.view(*indices.shape, -1)


READ_THREADS = 16
ROW_CHUNKS = READ_THREADS * 4
_reader: ThreadPoolExecutor | None = None


def _row_reader() -> ThreadPoolExecutor:
    global _reader
    if _reader is None:
        _reader = ThreadPoolExecutor(READ_THREADS)
    return _reader
