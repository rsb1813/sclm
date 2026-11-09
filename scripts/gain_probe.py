# Phase 1 probe: shrinkage factor beta = <w_hat, w>/<w, w> of low-bit quantizers on layer-10 experts, at weight and output level.
import torch
from sclm import quant
from sclm.checkpoint import Checkpoint
from scripts.quant_probe import DEV, expert_out, fp4_weight

torch.set_default_dtype(torch.bfloat16)
data = torch.load("data/probe_L10.pt", weights_only=True)
state = Checkpoint(r"C:\models\DeepSeek-V4.1-Flash").module_state("layers.10.ffn.")
x, idx, gw = data["x"].to(DEV).float(), data["idx"].to(DEV).long(), data["w"].to(DEV).float()
rots = {k: quant.BlockHadamard(k, seed=17 + k) for k in (5120, 2304)}
had = lambda f: (lambda w: rots[w.size(1)].unrotate(f(rots[w.size(1)].rotate(w))))  # noqa: E731
qs = {"rtn3_had": had(lambda w: quant.rtn(w, 3, 64)), "rtn2_had": had(lambda w: quant.rtn(w, 2, 64)),
      "rtn1_had": had(lambda w: quant.rtn(w, 1, 64))}
for name, q in qs.items():
    b_rows, nm, y_num, y_den, y_err, y_err_c = [], [], 0.0, 0.0, 0.0, 0.0
    for e in range(0, 384, 8):
        ws = [fp4_weight(state, f"experts.{e}.w{j}") for j in (1, 2, 3)]
        wq = [q(w) for w in ws]
        beta = [((a * b).sum(1) / (a * a).sum(1)) for a, b in zip(ws, wq)]  # per output row
        b_rows += [bb.mean().item() for bb in beta]
        nm += [((b - a).pow(2).sum() / a.pow(2).sum()).item() for a, b in zip(ws, wq)]
        rows, slot = torch.where(idx == e)
        if rows.numel():
            yr = expert_out(x[rows], *ws, gw[rows, slot, None])
            yq = expert_out(x[rows], *wq, gw[rows, slot, None])
            yc = expert_out(x[rows], *[w / b[:, None] for w, b in zip(wq, beta)], gw[rows, slot, None])
            y_num += (yq * yr).sum().item(); y_den += yr.pow(2).sum().item()
            y_err += (yq - yr).pow(2).sum().item(); y_err_c += (yc - yr).pow(2).sum().item()
    print(f"{name}: weight beta {sum(b_rows)/len(b_rows):.3f}, weight nmse {sum(nm)/len(nm):.3f} | "
          f"expert-output beta {y_num/y_den:.3f}, out nmse {y_err/y_den:.3f} -> with 1/beta rows {y_err_c/y_den:.3f}")
