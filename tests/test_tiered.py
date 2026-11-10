# Checks the two-tier MoE on the CPU against upstream's sum over experts: coded experts with the engine's int8 tables,
# original FP4 experts read from a synthetic checkpoint through the cache.
import json
import os
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sclm import core, ssd, tiered
from sclm.checkpoint import Checkpoint
from sclm.upstream import model as m

import kernel
import test_core as tc

pytestmark = pytest.mark.skipif(not os.path.exists(core.DLL), reason="run engine\\build.bat first")

PLAN = ["v8c8", "orig", "v2c6", "orig", "v8c8", "orig", "orig", "v2c6", "orig", "orig"]
TOPK = 4
SHAPES = {"w1": (tc.INTER, tc.DIM), "w2": (tc.DIM, tc.INTER), "w3": (tc.INTER, tc.DIM)}  # (out, in)


def write_checkpoint(root, experts: dict) -> dict:
    """One shard laid out as the released ones: every expert's three scales back to back, then every expert's three
    packed-FP4 weights back to back. `experts` maps "layers.L.ffn.experts.E.w1.scale" and so on to uint8 arrays;
    returns them in file order."""
    data = {name: a for part in ("scale", "weight") for name, a in experts.items() if name.endswith(part)}
    header, at = {}, 0
    for name, a in data.items():
        header[name] = {"dtype": "F8_E8M0" if name.endswith("scale") else "I8", "shape": list(a.shape), "data_offsets": [at, at + a.size]}
        at += a.size
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(root / "model-00001-of-00001.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(blob)) + blob)
        for a in data.values():
            f.write(a.tobytes())
    with open(root / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {name: "model-00001-of-00001.safetensors" for name in data}}, f)
    return data


def original_expert(data: dict, e: int) -> "m.Expert":
    expert = m.Expert(tc.DIM, tc.INTER, dtype=torch.float4_e2m1fn_x2, swiglu_limit=tc.LIMIT)
    with torch.no_grad():
        for name, p in expert.named_parameters():
            p.view(torch.uint8).copy_(torch.from_numpy(data[f"layers.0.ffn.experts.{e}.{name}"]))
    return expert


def test_tiered_moe_matches_the_sum_over_reference_experts(tmp_path, monkeypatch):
    m.world_size, m.rank, m.default_dtype = 1, 0, torch.float8_e4m3fn  # as building upstream's Transformer sets them
    # the kernel's GEMM (bf16 operands, fp32 result) exists on CUDA only; on the CPU both sides of the comparison
    # take the same products in fp32
    monkeypatch.setattr(kernel, "_mm_fp32_out", lambda a, w: a.float() @ w.float().t())
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # as the model runs
    try:
        rng = np.random.default_rng(0)
        data = write_checkpoint(tmp_path, {  # scales of 2^-8 .. 2^-5, any FP4 code pair
            f"layers.0.ffn.experts.{e}.{w}.{part}": rng.integers(lo, hi, (out, k // per_byte), dtype=np.uint8)
            for e in range(len(PLAN)) for w, (out, k) in SHAPES.items()
            for part, per_byte, lo, hi in (("scale", m.fp4_block_size, 119, 123), ("weight", 2, 0, 256))})
        coded = tc.make_layer(str(tmp_path / "layer_00.safetensors"), PLAN)
        experts = [original_expert(data, e) if c is None else
                   SimpleNamespace(**{w: (lambda v, mod=getattr(c, w): tc.table_linear(mod, v)) for w in SHAPES}, swiglu_limit=tc.LIMIT)
                   for e, c in enumerate(coded)]

        args = m.ModelArgs(dim=tc.DIM, moe_inter_dim=tc.INTER, n_routed_experts=len(PLAN), n_activated_experts=TOPK, swiglu_limit=tc.LIMIT)
        moe = tiered.TieredMoE(0, args)
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():
            moe.gate.weight.copy_(torch.randn(moe.gate.weight.shape, generator=g, dtype=torch.float32) * 0.05)
            moe.gate.bias.zero_()
            for name, p in moe.shared_experts.named_parameters():
                if name.endswith("scale"):
                    p.copy_(torch.full(p.shape, 2.0**-5, dtype=torch.float32).to(p.dtype))
                else:
                    p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32).to(p.dtype))
        moe.ram = core.Layer(str(tmp_path / "layer_00.safetensors"), tc.DIM, tc.INTER, tc.LIMIT)
        reader = ssd.ExpertReader(Checkpoint(str(tmp_path)), chunk=64 * 1024)
        moe.cache = ssd.ExpertCache(reader, 1, len(PLAN), tc.DIM, tc.INTER, tc.LIMIT, slots=2, lam=0.1, device="cpu", scratch=TOPK)
        reads, submit = [], reader.submit
        reader.submit = lambda layer, e, buf: reads.append(e) or submit(layer, e, buf)

        calls = 0
        with torch.inference_mode():
            for step in range(4):  # a 6-token prefill, then single tokens: the cache fills, hits, evicts and bypasses
                x = (3 * torch.randn(1, 6 if step == 0 else 1, tc.DIM, generator=g, dtype=torch.float32)).to(torch.bfloat16)
                y = moe(x)
                moe.cache.token += x.size(1)
                rows = x.view(-1, tc.DIM)
                weights, indices = moe.gate(rows)
                want = torch.zeros_like(rows, dtype=torch.float32)
                for r, ids in enumerate(indices.tolist()):  # a token at a time: the table emulation takes one
                    for e in sorted(ids):  # upstream MoE.forward adds in expert order
                        want[r] += m.Expert.forward(experts[e], rows[r:r + 1], weights[r, ids.index(e)].view(1, 1))[0]
                        calls += PLAN[e] == "orig"
                want = (want + moe.shared_experts(rows)).type_as(x).view(x.shape)
                # the tiers add a token's experts in another order than upstream, so a bf16 rounding may differ
                assert tc.rel(y.float(), want.float()) < 1e-7, step
        assert 0 < len(reads) < calls  # some experts were read, some came from the cache
    finally:
        torch.set_default_dtype(prev)


def test_cache_reads_unwaited_experts_in_the_background(tmp_path):
    rng = np.random.default_rng(1)
    data = write_checkpoint(tmp_path, {
        f"layers.0.ffn.experts.{e}.{w}.{part}": rng.integers(0, 256, (out, k // per_byte), dtype=np.uint8)
        for e in range(3) for w, (out, k) in SHAPES.items() for part, per_byte in (("scale", m.fp4_block_size), ("weight", 2))})
    reader = ssd.ExpertReader(Checkpoint(str(tmp_path)), chunk=64 * 1024)
    cache = ssd.ExpertCache(reader, 1, 3, tc.DIM, tc.INTER, tc.LIMIT, slots=2, lam=0.1, device="cpu", scratch=TOPK, background=2)

    def holds(slot: int, e: int) -> bool:
        want = [data[f"layers.0.ffn.experts.{e}.{w}.{part}"].ravel() for part in ("scale", "weight") for w in SHAPES]
        return np.array_equal(cache.store[slot].numpy(), np.concatenate(want))

    slots, pending = cache.request(0, [0, 1], 0, wait=[False, False])
    assert slots == [None, None] and not pending  # nothing is waited for, the token goes on without them
    reader.reader.wait()  # (the background reads complete)
    slots, pending = cache.request(0, [0, 1], 1, wait=[False, False])
    assert not pending and all(holds(slot, e) for slot, e in zip(slots, (0, 1)))
    slots, pending = cache.request(0, [2, 0], 2, wait=[True, False])  # a waited expert is read as before
    cache.finish(pending)
    assert len(pending) == 1 and holds(slots[0], 2) and holds(slots[1], 0)
