# Checks the RAM-tier files: the kernel's code layout and that a saved layer reproduces the modules' weights.
import json

import numpy as np
import torch
from safetensors import safe_open

from sclm import codes, quant

DEV = "cuda"


def test_tile_codes_round_trip():
    for bits in (8, 6):
        c = torch.randint(0, 1 << bits, (512, 40), dtype=torch.uint8)
        t = codes.tile_codes(c, bits)
        assert t.numel() == 512 * 40 * bits // 8
        assert torch.equal(codes.untile_codes(t, 512, 40, bits), c)


def test_six_bit_codes_sit_where_the_kernel_reads_them():
    # bench/vq_gemv.cpp: step j of a tile is 4 x 48 bytes, row p of a 64-row chunk at bit 6p
    c = torch.randint(0, 64, (256, 3), dtype=torch.uint8)
    step = codes.tile_codes(c, 6).numpy().reshape(3, 4, 48)[1]  # step 1 of the only tile
    for chunk in range(4):
        bit = np.unpackbits(step[chunk], bitorder="little").reshape(64, 6)
        assert np.array_equal(bit @ (1 << np.arange(6)), c[64 * chunk : 64 * chunk + 64, 1].numpy())


def test_eight_bit_step_is_the_tiles_rows():
    c = torch.randint(0, 256, (512, 3), dtype=torch.uint8)
    t = codes.tile_codes(c, 8).view(2, 3, 256)
    assert torch.equal(t[1, 2], c[256:, 2])


def test_saved_layer_reproduces_the_weights(tmp_path):
    options = {"v4c8": (4, 8, False), "v2c6": (2, 6, False)}
    rots = {k: quant.BlockHadamard(k, seed=17 + k) for k in (512, 256)}
    plan = ["v4c8", "orig", "v2c6", "v4c8"]
    cbs = {o: torch.randn(1 << bits, d, device=DEV) for o, (d, bits, _) in options.items()}

    def lin(opt, n, k):
        d, bits, _ = options[opt]
        return quant.VQLinear(torch.randint(0, 1 << bits, (n, k // d)), torch.rand(n, 1) + 0.5, cbs[opt], rots[k])

    experts = [None if o == "orig" else torch.nn.Module() for o in plan]
    for e, o in zip(experts, plan):
        if e is not None:
            e.w1, e.w2, e.w3 = lin(o, 256, 512), lin(o, 512, 256), lin(o, 256, 512)
    path = str(tmp_path / "layer_00.safetensors")
    codes.save_layer(path, plan, experts, options, rots)

    with safe_open(path, "pt") as f:
        meta = f.metadata()
        assert meta["format"] == "sclm-vq-1" and meta["rot512.block"] == "512"
        assert torch.equal(f.get_tensor("rot512.signs").float(), rots[512].signs.cpu())
        for opt, (d, bits) in json.loads(meta["options"]).items():
            es = f.get_tensor(f"{opt}.experts").tolist()
            assert es == [e for e, o in enumerate(plan) if o == opt]
            cb = f.get_tensor(f"{opt}.codebook").t()
            for name in ("w1", "w2", "w3"):
                buf, scales = f.get_tensor(f"{opt}.{name}.codes"), f.get_tensor(f"{opt}.{name}.scales")
                for k, e in enumerate(es):
                    mod = getattr(experts[e], name).to(DEV)  # the codebook is on the GPU
                    c = codes.untile_codes(buf[k], scales.size(1), mod.codes.size(1), bits)
                    w = quant.vq_codes_dequant(c.long(), scales[k].float().unsqueeze(-1), cb)
                    assert torch.equal(w, mod.weight_dequant().cpu())

    loaded = codes.load_layer(path, DEV)
    assert sorted(loaded) == [e for e, o in enumerate(plan) if o != "orig"]
    for e, mods in loaded.items():
        for name, mod in zip(("w1", "w2", "w3"), mods):
            ref = getattr(experts[e], name)
            x = torch.randn(3, mod.rot.signs.numel(), device=DEV)
            assert torch.equal(mod.weight_dequant(), ref.weight_dequant()) and torch.equal(mod(x), ref(x))
