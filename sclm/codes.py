# RAM-tier files: one safetensors file per layer with the routed experts' vector codes in bench/vq_gemv.cpp's layout.
import json
import os

import torch
from safetensors.torch import save_file

TILE = 256  # rows per tile (bench/vq_gemv.cpp)


def tile_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Codes [N, G] (uint8, one per row and code step) as the kernel reads them: N/256 row tiles, each the G steps of
    its 256 rows' codes. 8-bit codes take one byte per row; 6-bit codes are packed, code p of a 64-row chunk at bit 6p
    of its 48 bytes."""
    n, g = codes.shape
    t = codes.view(n // TILE, TILE, g).transpose(1, 2)
    if bits == 8:
        return t.reshape(-1)
    assert bits == 6
    c = t.reshape(-1, 4).int()
    v = c[:, 0] | c[:, 1] << 6 | c[:, 2] << 12 | c[:, 3] << 18
    return torch.stack([v & 255, v >> 8 & 255, v >> 16], dim=1).to(torch.uint8).reshape(-1)


def untile_codes(buf: torch.Tensor, n: int, g: int, bits: int) -> torch.Tensor:
    """Inverse of tile_codes."""
    if bits == 6:
        b = buf.view(-1, 3).int()
        v = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
        buf = torch.stack([v & 63, v >> 6 & 63, v >> 12 & 63, v >> 18], dim=1).to(torch.uint8)
    return buf.reshape(n // TILE, g, TILE).transpose(1, 2).reshape(n, g)


def save_layer(path: str, plan: list, experts, options: dict, rots: dict) -> None:
    """Writes a layer's quantized routed experts (VQLinear w1, w2, w3 on the CPU). `plan` names each expert's entry of
    `options` (name -> (columns per code, code bits, ...)); an "orig" expert stays in the original checkpoint and is
    not in the file. Per option o: o.experts (int32 [E]), o.codebook (fp32 [d, C], dimension-major as the kernel's
    table builder reads it) and for each matrix o.w1.codes (uint8 [E, bytes], tile_codes) and o.w1.scales (fp16
    [E, N]). A matrix is scales[:, None] * codebook[codes] in the rotated basis: its input is the Hadamard transform
    over blocks of rot<K>.block of rot<K>.signs * x, divided by sqrt(block)."""
    tensors = {f"rot{k}.signs": r.signs.to(torch.int8).cpu() for k, r in rots.items()}
    meta = {"format": "sclm-vq-1", "tile": str(TILE), "options": {}}
    meta.update({f"rot{k}.block": str(r.b) for k, r in rots.items()})
    for opt in dict.fromkeys(p for p in plan if p != "orig"):
        es = [e for e, p in enumerate(plan) if p == opt]
        d, bits = options[opt][:2]
        meta["options"][opt] = [d, bits]
        tensors[f"{opt}.experts"] = torch.tensor(es, dtype=torch.int32)
        tensors[f"{opt}.codebook"] = experts[es[0]].w1.codebook.float().t().contiguous().cpu()
        for name in ("w1", "w2", "w3"):
            mods = [getattr(experts[e], name) for e in es]
            assert all(mod.scales.size(-1) == 1 for mod in mods), "one scale per row"
            tensors[f"{opt}.{name}.codes"] = torch.stack([tile_codes(mod.codes, bits) for mod in mods])
            tensors[f"{opt}.{name}.scales"] = torch.stack([mod.scales.view(-1) for mod in mods])
    meta["options"] = json.dumps(meta["options"])
    save_file(tensors, path + ".tmp", metadata=meta)
    os.replace(path + ".tmp", path)  # a killed run leaves no half-written layer
