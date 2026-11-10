# Checks the CPU engine against the upstream Expert with VQLinear inside: equal once the GEMVs use the engine's int8 tables.
import os
from types import SimpleNamespace

import pytest
import torch

from sclm import codes, core, quant, upstream

import kernel  # noqa: E402  (sclm.upstream puts it on the path)

pytestmark = pytest.mark.skipif(not os.path.exists(core.DLL), reason="run engine\\build.bat first")

DIM, INTER, LIMIT = 512, 256, 10.0
OPTIONS = {"v8c8": (8, 8, False), "v2c6": (2, 6, False)}
PLAN = ["v8c8", "orig", "v2c6", "v8c8", "v2c6"]


def make_layer(path):
    g = torch.Generator().manual_seed(0)
    rots = {k: quant.BlockHadamard(k, seed=17 + k, device="cpu") for k in (DIM, INTER)}
    cbs = {o: torch.randn(1 << bits, d, generator=g) for o, (d, bits, _) in OPTIONS.items()}

    def lin(opt, n, k):
        d, bits, _ = OPTIONS[opt]
        scales = (torch.rand(n, 1, generator=g) + 0.5) / k**0.5
        return quant.VQLinear(torch.randint(0, 1 << bits, (n, k // d), generator=g), scales, cbs[opt], rots[k])

    experts = [None if o == "orig" else SimpleNamespace(w1=lin(o, INTER, DIM), w2=lin(o, DIM, INTER), w3=lin(o, INTER, DIM),
                                                        swiglu_limit=LIMIT) for o in PLAN]
    codes.save_layer(path, PLAN, experts, OPTIONS, rots)
    return experts


def table_linear(mod, v):
    """VQLinear.forward with the engine's GEMV: per token, <codebook vector, input group> as int8 on one scale."""
    vq = kernel.act_quant(v.to(torch.bfloat16).contiguous().clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True).float()
    vr = mod.rot.rotate(vq.reshape(-1, vq.size(-1)))[0]
    t = vr.view(-1, mod.codebook.size(1)) @ mod.codebook.float().t()  # [groups, codes]
    scale = t.abs().max() / 127
    sums = torch.round(t * (1 / scale)).gather(1, mod.codes.long().t()).sum(0)
    return (sums * scale * mod.scales.float().view(-1))[None].to(torch.get_default_dtype())


def rel(a, b):
    return ((a - b).square().sum() / b.square().sum()).item()


def test_layer_forward_matches_the_reference_expert(tmp_path):
    path = str(tmp_path / "layer_00.safetensors")
    experts = make_layer(path)
    tabled = [e and SimpleNamespace(**{w: (lambda v, m=getattr(e, w): table_linear(m, v)) for w in ("w1", "w2", "w3")},
                                    swiglu_limit=LIMIT) for e in experts]
    layer = core.Layer(path, DIM, INTER, LIMIT)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)  # as the model runs
    try:
        g = torch.Generator().manual_seed(1)
        for ids in ([0], [2], [0, 2, 3, 4]):
            x = (3 * torch.randn(1, DIM, generator=g, dtype=torch.float32)).to(torch.bfloat16)
            gates = torch.rand(len(ids), generator=g, dtype=torch.float32)
            ref, emu = (sum(upstream.model.Expert.forward(es[e], x, gates[i].view(1, 1)).float() for i, e in enumerate(ids))[0]
                        for es in (experts, tabled))
            y = torch.from_numpy(layer.forward(ids, gates.tolist(), x.float().numpy()[0]))
            assert rel(y, emu) < 1e-9, ids  # code layout, rotation, FP8 and bf16 rounding all as the reference
            assert rel(y, ref) < 1e-2, ids  # what the int8 tables cost at this toy size
    finally:
        torch.set_default_dtype(prev)
