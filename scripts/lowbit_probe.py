# Phase 1 probe (CPU): weight NMSE of 1-bit and ternary scalar codes on a few layer-10 experts (CPU, Hadamard basis).
import torch
from sclm import quant
from sclm.checkpoint import Checkpoint
from sclm.upstream import model as _m  # noqa: F401
import kernel

torch.set_num_threads(16)
state = Checkpoint(r"C:\models\DeepSeek-V4.1-Flash").module_state("layers.10.ffn.")
rot = quant.BlockHadamard(5120, seed=17 + 5120, device="cpu")

def dense(name):
    w, s = state[name + ".weight"], state[name + ".scale"]
    return (kernel.unpack_fp4(w).unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)).flatten(-2)

def grid_search(g, recon, ratios):
    """Per group pick the scale ratio (x group RMS) minimizing MSE; recon(g, a) reconstructs with scale a [.., 1]."""
    rms = g.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    best, err = None, None
    for r in ratios:
        q = recon(g, (rms * r).half().float())
        e = (q - g).pow(2).sum(-1, keepdim=True)
        best, err = (q, e) if best is None else (torch.where(e < err, q, best), torch.minimum(e, err))
    return best

sign1 = lambda g, a: torch.where(g >= 0, a, -a)                                   # 1 bit: +-a
tern = lambda g, a: torch.where(g.abs() > a / 2, torch.sign(g) * a, torch.zeros_like(g))  # {-a, 0, a}
def uni2(g, a):                                                                 # 2 bits: mid-rise, step a
    return (torch.floor(g / a).clamp(-2, 1) + 0.5) * a

ratios = torch.linspace(0.3, 2.0, 35).tolist()
rows = {}
for e in range(4):
    w = rot.rotate(dense(f"experts.{e}.w1"))
    for gsz in (64, 128):
        g = w.reshape(w.size(0), -1, gsz)
        for name, f in (("1bit", sign1), ("ternary", tern), ("2bit", uni2)):
            q = grid_search(g, f, ratios).reshape(w.shape)
            rows.setdefault((name, gsz), []).append(((q - w).pow(2).sum() / w.pow(2).sum()).item())
    rows.setdefault(("rtn2_current", 64), []).append(((quant.rtn(w, 2, 64) - w).pow(2).sum() / w.pow(2).sum()).item())
    rows.setdefault(("rtn1_current", 64), []).append(((quant.rtn(w, 1, 64) - w).pow(2).sum() / w.pow(2).sum()).item())
bits = {"1bit": 1, "ternary": 1.6, "2bit": 2, "rtn2_current": 2, "rtn1_current": 1}
for (name, gsz), v in rows.items():
    b = bits[name] + 16 / gsz
    print(f"{name:13s} g{gsz:<3d} {b:.3f} bpw  weight NMSE {sum(v)/len(v):.3f}   (Gaussian bound at {bits[name]} bits: {2 ** (-2 * bits[name]):.3f})")
