# Phase 1: whole-model quality with every layer's routed experts compressed: KL, top-1 agreement and ppl vs the original.
import argparse
import json
import os
import shutil
import time
import weakref

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from sclm import quant
from sclm.checkpoint import Checkpoint
from sclm.codes import load_layer, save_layer
from sclm.layerwise import LayerwiseRunner, load_args, load_params_
from sclm.upstream import model as m

import kernel  # noqa: E402  (on sys.path via sclm.upstream)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
DEV = "cuda"
TOPK = 64  # reference distribution kept as its top-64 tokens per position
CHUNK = 256
SCALES = (0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3)  # logit multipliers for the oracle-temperature KL diagnostic
MARGIN_EDGES = torch.logspace(-5, 0, 26)  # bins of the 6th-7th selection-score margin (--flip-stats)
TOKEN_CHUNK = 16384  # calibration tokens per Hessian accumulation step of one expert
# name -> (bits, group, hadamard) for routed experts; "none" is the uncompressed reference, "dense" runs the
# uncompressed FP4 weights through the same fp32 path RTNLinear uses (isolates the GEMM path from compression)
CONFIGS = {"none": None, "dense": "dense", "rtn3_had": (3, 64, True), "rtn2_had": (2, 64, True),
           "rtn3": (3, 64, False), "rtn2": (2, 64, False),
           # rows rescaled by 1/beta so rounding does not shrink them (RTNLinear unbiased)
           "rtn3_had_ub": (3, 64, True, True), "rtn2_had_ub": (2, 64, True, True),
           # routed experts restricted to the per-layer top-r activation subspaces (scripts/collect_bases.py)
           "proj256": ("proj", 256), "proj512": ("proj", 512),
           "proj512_in": ("proj", 512, "in"), "proj512_out": ("proj", 512, "out"),
           # per-expert bits from scripts/allocate_bits.py (data/alloc_<name>.json); 4 keeps the original FP4 expert
           "alloc_2p75": ("alloc", "2p75"),
           # GPTQ per expert from calibration sequences (GPTQExperts); the last field is the H-metric unbiasing
           "gptq2": ("gptq", 2, True), "gptq3": ("gptq", 3, True),
           # per-expert GPTQ option or the original FP4 expert (the SSD tier) from scripts/allocate_tiers.py
           "tier20": ("gptq_alloc", "tier20", False), "tier20g": ("gptq_alloc", "tier20g", False),
           "tier20v": ("gptq_alloc", "tier20v", False), "tier20v110": ("gptq_alloc", "tier20v110", False),
           "tier20v100": ("gptq_alloc", "tier20v100", False), "tier20r100": ("gptq_alloc", "tier20r100", False),
           # the codes a gptq_alloc run saved (--codes DIR): no calibration, the experts not in a file stay original
           "codes": "codes"}
# GPTQ storage options: name -> (bits, levels, group); "orig" keeps the original FP4 expert
GPTQ_OPTIONS = {"1b128": (1, 2, 128), "t128": (2, 3, 128), "2b128": (2, 4, 128), "2b64": (2, 4, 64),
                "3b64": (3, 8, 64), "3b128": (3, 8, 128)}
# vector-code options (quant.gptq_vq, one fp16 scale per row): name -> (columns per code, code bits, least-squares scale)
VQ_OPTIONS = {"v16c8": (16, 8, True), "v8c8": (8, 8, True), "v4c8": (4, 8, False), "v2c6": (2, 6, False)}


class DenseFP4Linear(torch.nn.Module):
    """Uncompressed expert Linear computed like RTNLinear: fp8 activation fake-quant, fp32 dequantized weight, fp32 GEMM."""

    def __init__(self, lin):
        super().__init__()
        self.weight, self.scale = lin.weight, lin.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = kernel.act_quant(x.to(torch.bfloat16).contiguous().clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
        w = (kernel.unpack_fp4(self.weight).unflatten(-1, (-1, 32)) * self.scale.float().unsqueeze(-1)).flatten(-2)
        return F.linear(xq.float(), w).to(torch.get_default_dtype())


def dense_experts_(block) -> None:
    for expert in block.ffn.experts:
        for name in ("w1", "w2", "w3"):
            setattr(expert, name, DenseFP4Linear(getattr(expert, name)))


def coded_experts_(layer: int, block, codes_dir: str, tables: bool) -> None:
    """Replaces the routed experts found in the layer's saved code file (sclm/codes.py) by their VQLinear modules.
    tables: they sum int8 tables as the engine does instead of an fp32 GEMM."""
    for e, mods in load_layer(os.path.join(codes_dir, f"layer_{layer:02d}.safetensors"), DEV).items():
        for name, mod in zip(("w1", "w2", "w3"), mods):
            mod.tables = tables
            setattr(block.ffn.experts[e], name, mod)


def act_q(x: torch.Tensor) -> torch.Tensor:
    return kernel.act_quant(x.to(torch.bfloat16).contiguous().clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True).float()


def fp4_dense(lin) -> torch.Tensor:
    return (kernel.unpack_fp4(lin.weight).unflatten(-1, (-1, 32)) * lin.scale.float().unsqueeze(-1)).flatten(-2)


def ram_avail_gb() -> float:
    """Physical RAM Windows reports as available, in GiB."""
    import ctypes

    class Status(ctypes.Structure):  # MEMORYSTATUSEX
        _fields_ = [("length", ctypes.c_uint32), ("load", ctypes.c_uint32),
                    *((n, ctypes.c_uint64) for n in ("total", "avail", "page", "page_avail", "virt", "virt_avail", "ext"))]

    st = Status(length=ctypes.sizeof(Status))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.avail / 2**30


class ProjIn(torch.nn.Module):
    """w1/w3 restricted to the layer's top-r FFN-input directions: W x ~= (W U)(U^T x), U shared by all experts."""

    def __init__(self, w: torch.Tensor, u: torch.Tensor):
        super().__init__()
        self.u, self.wu = u, w @ u

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((act_q(x) @ self.u) @ self.wu.t()).to(torch.get_default_dtype())


class ProjOut(torch.nn.Module):
    """w2 restricted to the layer's top-r routed-output directions: W h ~= V (V^T W h)."""

    def __init__(self, w: torch.Tensor, v: torch.Tensor):
        super().__init__()
        self.v, self.vw = v, v.t() @ w

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return ((act_q(h) @ self.vw.t()) @ self.v.t()).to(torch.get_default_dtype())


def project_experts_(layer: int, block, r: int, sides: str = "inout") -> None:
    bases = torch.load(os.path.join(DATA, "bases_v1", f"layer{layer}.pt"), weights_only=True)
    u = bases["in_basis"][:, :r].to(DEV)
    v = bases["out_basis"][:, :r].to(DEV)
    for expert in block.ffn.experts:
        if "in" in sides:
            expert.w1 = ProjIn(fp4_dense(expert.w1), u)
            expert.w3 = ProjIn(fp4_dense(expert.w3), u)
        if "out" in sides:
            expert.w2 = ProjOut(fp4_dense(expert.w2), v)


def quantize_experts_(block, bits, group: int, use_rot: bool, unbiased: bool = False) -> None:
    """Replace every routed expert Linear of `block` by an RTNLinear, one at a time so peak VRAM stays near one layer.
    `bits` is one width for all experts or a per-expert list, where 4 leaves the original FP4 expert in place."""
    rots = {}
    for e, expert in enumerate(block.ffn.experts):
        b = bits if isinstance(bits, int) else bits[e]
        if not isinstance(bits, int) and b == 4:
            continue
        for name in ("w1", "w2", "w3"):
            lin = getattr(expert, name)
            w = (kernel.unpack_fp4(lin.weight).unflatten(-1, (-1, 32)) * lin.scale.float().unsqueeze(-1)).flatten(-2)
            k = w.size(1)
            rot = rots.setdefault(k, quant.BlockHadamard(k, seed=17 + k)) if use_rot else None
            setattr(expert, name, quant.RTNLinear(w, b, group, rot, unbiased))
            del lin, w


def rotate_h(h: torch.Tensor, rot: quant.BlockHadamard) -> torch.Tensor:
    """R^T H R for Hessians [E, K, K] (the Hadamard basis the weights are quantized in)."""
    from scripts.gptq_probe import rot_cols

    return rot_cols(rot, rot_cols(rot, h).transpose(-1, -2))


def gptq_in_basis(w: torch.Tensor, hr: torch.Tensor, bits: int, rot: quant.BlockHadamard, levels: int | None = None,
                  group: int = 64, unbias: bool = True):
    """GPTQ of w [E, N, K] in the Hadamard basis under Hessians hr [E, K, K] already in that basis (rotate_h), rows
    optionally rescaled by the output shrinkage <w_hat, w>_H / <w, w>_H. Returns codes and steps (rotated basis) and
    the dequantized weight (original basis)."""
    from scripts.gptq_probe import rot_cols

    wr = rot_cols(rot, w)
    codes, steps = quant.gptq_codes(wr, hr, bits, group, levels=levels)
    wq = quant.rtn_dequant(codes, steps, bits, group, levels)
    if not unbias:
        return codes, steps, rot_cols(rot, wq, inverse=True)
    beta = (((wq @ hr) * wr).sum(-1) / ((wr @ hr) * wr).sum(-1)).clamp_min(0.05)
    return codes, steps / beta[..., None], rot_cols(rot, wq / beta[..., None], inverse=True)


class GPTQExperts:
    """Collects a layer's FFN inputs and routing while the calibration batches pass, then, right before the first eval
    batch, replaces its routed experts by GPTQ codes: Hadamard basis, per-expert gate^2-weighted X^T X mixed with the
    layer covariance (beta), optional H-metric unbiasing. The calibration stream stays the uncompressed model's.
    `plan(layer)` names each expert's GPTQ_OPTIONS or VQ_OPTIONS entry, or "orig" to keep its FP4 weights.
    `n_tok`: calibration tokens in all (their FFN inputs of the current layer wait in one CPU buffer).
    The calibration batches skip the routed experts (unpacking all 384 for every batch was two thirds of a layer's
    forward time): quantize_ adds each expert's output over all of its tokens at once, and finish_ completes the
    batches' block outputs. `save_dir`: each layer's codes are written there (sclm/codes.py, vector options only)."""

    def __init__(self, plan, n_cal: int, n_tok: int, unbias: bool, beta: float = 0.5, batch: int = 8,
                 save_dir: str | None = None):
        self.plan, self.n_cal, self.unbias, self.beta, self.batch = plan, n_cal, unbias, beta, batch
        self.save_dir = save_dir
        self.block, self.cur, self.t0 = None, 0, time.time()
        self.xq = torch.empty(n_tok, 5120, dtype=torch.bfloat16, device="cpu")
        self.y = torch.empty(n_tok, 5120, dtype=torch.float32, device="cpu")  # the layer's FFN outputs
        self.rots = {k: quant.BlockHadamard(k, seed=17 + k) for k in (5120, 2304)}  # quant_eval's usual seeds

    def before_block(self, i: int, block) -> None:
        self.layer, self.block = i, weakref.ref(block)  # weak: the runner frees each block itself
        self.gates, self.n, self.h_layer, self.deferred = [], 0, 0, []
        block.ffn.gate.register_forward_hook(self.on_gate)
        ffn, ffn_forward, hc_post, pending = block.ffn, block.ffn.forward, block.hc_post, []

        def forward(x, image_mask=None):
            if self.cur >= self.n_cal:
                return ffn_forward(x, image_mask)
            n, xf = self.n, x.view(-1, ffn.dim)
            ffn.gate(xf, None)  # on_gate keeps the inputs and the routing
            self.y[n : self.n] = ffn.shared_experts(xf).float()
            pending.append(True)
            return x

        def post(x, residual, post_mix, comb):
            if not pending:
                return hc_post(x, residual, post_mix, comb)
            pending.clear()
            # the FFN output is not there yet: the runner gets the residual on the CPU, and keeps this very tensor
            # (.cpu() of a CPU tensor is the tensor itself), which finish_ overwrites with the block's output
            self.deferred.append((residual.cpu(), post_mix, comb))
            return self.deferred[-1][0]

        ffn.forward, block.hc_post = forward, post

    def on_gate(self, _mod, inp, out) -> None:
        if self.cur < self.n_cal:
            x = act_q(inp[0].detach().reshape(-1, 5120))
            self.h_layer = self.h_layer + x.t() @ x
            # fp8 values with power-of-two scales: exact in bf16. They wait on the CPU: 100K+ calibration tokens
            # would crowd the Hessians out of VRAM
            self.xq[self.n : self.n + x.size(0)] = x.bfloat16()
            self.n += x.size(0)
            self.gates.append((out[0].float(), out[1]))

    def before_batch(self, i: int, j: int) -> None:
        self.cur = j
        if j == self.n_cal:
            self.quantize_(self.block())
            self.finish_()

    def finish_(self) -> None:
        n = 0
        for out, post_mix, comb in self.deferred:
            t = out.size(0) * out.size(1)
            y = self.y[n : n + t].to(DEV).to(out.dtype).view(*out.shape[:2], -1)
            out.copy_(m.Block.hc_post(None, y, out.to(DEV), post_mix, comb))
            n += t
        self.deferred = []

    def add_outputs_(self, w1, w2, w3, xr, gate_w, rows_cpu) -> None:
        """Adds an expert's output on its tokens (fp8-quantized inputs xr, gate weights [T, 1]) to self.y, computed
        like upstream Expert.forward over the reference fp4_gemm (operands exact in bf16, fp32 accumulation)."""
        from scripts.quant_probe import SWIGLU_LIMIT

        mm = lambda a, w: kernel._mm_fp32_out(a.bfloat16(), w.bfloat16()).bfloat16().float()  # noqa: E731
        g, u = mm(xr, w1), mm(xr, w3)
        hid = F.silu(g.clamp(max=SWIGLU_LIMIT)) * u.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT) * gate_w
        self.y.index_add_(0, rows_cpu, mm(act_q(hid), w2).cpu())

    def quantize_(self, block) -> None:
        from scripts.gptq_probe import tr_norm

        t0 = time.time()
        gw = torch.cat([g for g, _ in self.gates])
        idx = torch.cat([i for _, i in self.gates]).long()
        xq, h_layer, self.gates, self.h_layer = self.xq[: self.n], tr_norm(self.h_layer), [], 0
        experts = block.ffn.experts
        experts.to("cpu")  # frees ~7GB of FP4 weights for the Hessians; each batch is copied back for its GPTQ
        torch.cuda.empty_cache()
        plan = self.plan(self.layer)
        # kept FP4 experts return to the GPU last, after the Hessians of the quantized ones are freed
        for opt in sorted(dict.fromkeys(plan), key=lambda o: o == "orig"):
            es = [e for e, p in enumerate(plan) if p == opt]
            if opt == "orig":
                for e in es:
                    experts[e].to(DEV)
                    rows, slot = torch.where(idx == e)
                    rc = rows.cpu()
                    w1, w2, w3 = (fp4_dense(getattr(experts[e], n)) for n in ("w1", "w2", "w3"))
                    for c in range(0, rows.numel(), TOKEN_CHUNK):
                        self.add_outputs_(w1, w2, w3, xq[rc[c : c + TOKEN_CHUNK]].to(DEV).float(),
                                          gw[rows[c : c + TOKEN_CHUNK], slot[c : c + TOKEN_CHUNK], None], rc[c : c + TOKEN_CHUNK])
            else:
                self.quantize_option_(experts, es, opt, xq, gw, idx, h_layer)
        if self.save_dir:
            save_layer(os.path.join(self.save_dir, f"layer_{self.layer:02d}.safetensors"), plan, experts, VQ_OPTIONS,
                       self.rots)
        experts.to(DEV)
        counts = {o: plan.count(o) for o in dict.fromkeys(plan)}
        print(f"gptq layer {self.layer:2d}: {xq.size(0)} calib tokens, {counts}, {time.time() - t0:.0f}s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB, RAM free {ram_avail_gb():.1f}GiB, "
              f"total {(time.time() - self.t0) / 60:.0f} min", flush=True)

    def codebook(self, experts, members: list, d: int, bits: int) -> torch.Tensor:
        """k-means codes for d consecutive weights (rotated basis, rows at unit rms) of 8 of the option's experts,
        trained as scripts/gptq_probe.py does."""
        torch.manual_seed(0)  # k-means draws from the global generators: a resumed run trains the same codebook
        sample = []
        for e in members[:: max(1, len(members) // 8)][:8]:
            for name in ("w1", "w2", "w3"):
                w = fp4_dense(getattr(experts[e], name).to(DEV)).float()
                wr = self.rots[w.size(-1)].rotate(w)
                v = (wr / wr.pow(2).mean(-1, keepdim=True).sqrt()).reshape(-1, d)
                sample.append(v[torch.randperm(v.size(0), device="cpu")[: 1 << 17].to(DEV)])
        return quant.kmeans(torch.cat(sample), 1 << bits)

    def quantize_option_(self, experts, members: list, opt: str, xq, gw, idx, h_layer) -> None:
        from scripts.gptq_probe import rot_cols, tr_norm
        from scripts.quant_probe import SWIGLU_LIMIT

        # gptq(w [E, N, K], Hessians in the rotated basis, rot) -> the experts' modules and dequantized weights;
        # parts > 1: the rows are that many matrices sharing the Hessians, returned as one module list each
        if opt in VQ_OPTIONS:
            d, bits, refit = VQ_OPTIONS[opt]
            cb = self.codebook(experts, members, d, bits)

            def gptq(w, hr, rot, parts=1):
                codes, scales = quant.gptq_vq(rot_cols(rot, w), hr, cb, w.size(-1), refit=refit)
                return ([[quant.VQLinear(c, s, cb, rot) for c, s in zip(cp, sp)]
                         for cp, sp in zip(codes.chunk(parts, 1), scales.chunk(parts, 1))],
                        rot_cols(rot, quant.vq_codes_dequant(codes, scales, cb), inverse=True))
        else:
            bits, levels, group = GPTQ_OPTIONS[opt]

            def gptq(w, hr, rot, parts=1):
                codes, steps, wq = gptq_in_basis(w, hr, bits, rot, levels, group, self.unbias)
                return ([[quant.RTNLinear.from_codes(c, s, bits, group, rot, levels) for c, s in zip(cp, sp)]
                         for cp, sp in zip(codes.chunk(parts, 1), steps.chunk(parts, 1))], wq)

        eye = torch.eye(2304, device=DEV, dtype=torch.float32)
        for e0 in range(0, len(members), self.batch):
            es = members[e0 : e0 + self.batch]
            toks = [torch.where(idx == e) for e in es]
            rows_cpu = [rows.cpu() for rows, _ in toks]
            w1, w2, w3 = (torch.stack([fp4_dense(getattr(experts[e], n).to(DEV)) for e in es])
                          for n in ("w1", "w2", "w3"))
            h1 = []
            for k, ((rows, slot), rc) in enumerate(zip(toks, rows_cpu)):
                h_e = 0
                for c in range(0, rows.numel(), TOKEN_CHUNK):  # a busy expert sees 100K+ calibration tokens
                    xr = xq[rc[c : c + TOKEN_CHUNK]].to(DEV).float()
                    gwc = gw[rows[c : c + TOKEN_CHUNK], slot[c : c + TOKEN_CHUNK], None]
                    h_e = h_e + xr.t() @ (gwc.pow(2) * xr)
                    self.add_outputs_(w1[k], w2[k], w3[k], xr, gwc, rc[c : c + TOKEN_CHUNK])
                h_e = tr_norm(h_e) if rows.numel() else h_layer
                h1.append((1 - self.beta) * h_e + self.beta * h_layer)
            h1 = rotate_h(torch.stack(h1), self.rots[5120])
            # w1 and w3 see the same inputs: one GPTQ pass over both (rows are independent given the Hessian)
            w13 = torch.cat([w1, w3], dim=1)
            del w1, w3
            (l1, l3), q13 = gptq(w13, h1, self.rots[5120], parts=2)
            q1, q3 = q13.chunk(2, 1)
            del h1, w13, q13
            h2 = []
            for k, (rows, slot) in enumerate(toks):
                if not rows.numel():
                    h2.append(eye)
                    continue
                h_e = 0
                for c in range(0, rows.numel(), TOKEN_CHUNK):
                    xr = xq[rows_cpu[k][c : c + TOKEN_CHUNK]].to(DEV).float()
                    g, u = xr @ q1[k].t(), (xr @ q3[k].t()).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
                    hid = act_q(F.silu(g.clamp(max=SWIGLU_LIMIT)) * u * gw[rows[c : c + TOKEN_CHUNK], slot[c : c + TOKEN_CHUNK], None])
                    h_e = h_e + hid.t() @ hid
                h2.append((1 - self.beta) * tr_norm(h_e) + self.beta * eye)
            (l2,), _ = gptq(w2, rotate_h(torch.stack(h2), self.rots[2304]), self.rots[2304])
            del h2, w2, q1, q3
            for k, e in enumerate(es):
                experts[e].w1, experts[e].w2, experts[e].w3 = l1[k], l2[k], l3[k]
                experts[e].to("cpu")  # up to 4GiB of codes in a layer: off the GPU while the others are quantized


def freeze_routing_(block, layer: int, ref_idx: list, cur: dict) -> None:
    """Make the gate pick the reference model's experts (from data/routing_v1.npz) while computing their weights from
    its own scores, so routing flips caused by upstream error are removed and only expert-output error remains."""
    gate = block.ffn.gate

    def forward(x, image_mask=None):
        scores = m.linear(x.float(), gate.weight.float()) / gate.gate_temp
        if gate.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif gate.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        indices = ref_idx[cur["batch"]][:, layer].to(x.device)
        weights = scores.gather(1, indices)
        if gate.norm_topk_prob and gate.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
        return weights * gate.route_scale, indices

    gate.forward = forward


def selection(gate, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate.forward's scores and the bias-shifted ones it picks experts by (the sqrtsoftplus gate of this model)."""
    assert gate.score_func == "sqrtsoftplus"
    s = F.softplus(m.linear(x.float(), gate.weight.float()) / gate.gate_temp).sqrt()
    return s, s + gate.bias


def flip_stats_(block, layer: int, ref_idx: list, cur: dict, stats: list) -> None:
    """Per bin of the model's own 6th-7th selection margin: tokens, and how often the reference picked the own 7th
    expert but not the own 6th (a boundary flip, design.md T7)."""
    gate = block.ffn.gate

    def hook(_mod, inp, _out):
        _, sel = selection(gate, inp[0])
        top = sel.topk(7, dim=-1)
        ref = ref_idx[cur["batch"]][:, layer].to(sel.device)
        in_ref = (ref[:, :, None] == top.indices[:, None, 5:7]).any(1)
        flip = (in_ref[:, 1] & ~in_ref[:, 0]).double()
        b = torch.bucketize(top.values[:, 5] - top.values[:, 6], MARGIN_EDGES.to(sel.device))
        stats[layer][0] += torch.bincount(b, minlength=len(MARGIN_EDGES) + 1).cpu()
        stats[layer][1] += torch.bincount(b, weights=flip, minlength=len(MARGIN_EDGES) + 1).cpu()

    gate.register_forward_hook(hook)


def fit_sigma(n: torch.Tensor, k: torch.Tensor) -> float:
    """Maximum-likelihood sigma of p(flip | margin) = Phi(-margin / sigma) over margin bins (counts n, flips k)."""
    e = MARGIN_EDGES.double()
    centers = torch.cat([e[:1] / 2, (e[1:] * e[:-1]).sqrt(), e[-1:] * 2])
    best = None
    for sig in torch.logspace(-6, 0, 241, dtype=torch.float64):
        p = (0.5 * torch.erfc(centers / (sig * 2**0.5))).clamp(1e-12, 1 - 1e-12)
        ll = float((k * p.log() + (n - k) * (1 - p).log()).sum())
        best = max(best or (ll, 0.0), (ll, float(sig)))
    return best[1]


def hedge_routing_(block, layer: int, sigma: float, hedged: torch.Tensor, p_min: float = 0.02) -> None:
    """T7: route the own 7th expert too and split the 6th slot's weight, (1 - p) to the 6th and p to the 7th, where
    p = Phi(-margin / sigma) is the chance the 6th-7th order is flipped. Tokens with p < p_min keep top-6."""
    gate = block.ffn.gate

    def forward(x, image_mask=None):
        s, sel = selection(gate, x)
        top = sel.topk(7, dim=-1)
        w = s.gather(1, top.indices)
        w = w / (w[:, :6].sum(-1, keepdim=True) + 1e-20) * gate.route_scale  # norm_topk_prob over the top 6
        p = 0.5 * torch.erfc((top.values[:, 5] - top.values[:, 6]) / (sigma * 2**0.5))
        p = torch.where(p < p_min, 0.0, p)
        w[:, 5] *= 1 - p
        w[:, 6] *= p
        hedged[layer] += torch.tensor([float((p > 0).sum()), p.numel()], dtype=torch.float64, device="cpu")
        return w, top.indices

    gate.forward = forward


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"C:\models\DeepSeek-V4.1-Flash")
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--freeze-routing", action="store_true", help="force the reference model's expert choices")
    ap.add_argument("--record-routing", action="store_true", help="with --config none: save the expert choices")
    ap.add_argument("--layers", help="lo:hi, compress only layers lo..hi-1 (per-block configs; layer sensitivity)")
    ap.add_argument("--flip-stats", action="store_true", help="T7: boundary-flip rate by margin, fit sigma per layer")
    ap.add_argument("--hedge", type=float, help="T7: hedge the 6th slot with sigma = this x the --flip-stats fit")
    ap.add_argument("--calib-seqs", type=int, help="GPTQ calibration sequences, spread over the non-eval ones "
                    "(default: the 15 of collect_bases.calib_indices)")
    ap.add_argument("--calib-extra", help="token set (e.g. calib_v2, disjoint from calib_v1) added to that pool")
    ap.add_argument("--checkpoint-every", type=int, help="GPTQ configs: save the state between layers to "
                    "data/ckpt_<tag>.pt every this many layers, and resume from that file when it exists")
    ap.add_argument("--codes", help="config codes: directory with each layer's saved codes")
    ap.add_argument("--tables", action="store_true", help="config codes: int8 table sums as the engine computes them")
    ap.add_argument("--save-codes", help="gptq_alloc configs with vector options: directory for each layer's codes "
                    "(sclm/codes.py), the allocation and the evaluation result")
    cli = ap.parse_args()
    torch.set_default_dtype(torch.bfloat16)

    tokens = torch.from_numpy(np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["tokens"]).long()
    seqs = tokens[3::6][:8]  # disjoint from the layer probe's tokens[::6], spread over categories
    runner = LayerwiseRunner(Checkpoint(cli.ckpt), load_args(max_batch_size=2, max_seq_len=tokens.size(1)),
                             AutoTokenizer.from_pretrained(cli.ckpt))
    cfg = CONFIGS[cli.config]
    if cfg == "dense":
        before_block = lambda i, block: dense_experts_(block)  # noqa: E731
    elif cfg == "codes":
        before_block = lambda i, block: coded_experts_(i, block, cli.codes, cli.tables)  # noqa: E731
    elif isinstance(cfg, tuple) and cfg[0] == "proj":
        before_block = lambda i, block: project_experts_(i, block, *cfg[1:])  # noqa: E731
    elif isinstance(cfg, tuple) and cfg[0] == "alloc":
        with open(os.path.join(DATA, f"alloc_{cfg[1]}.json")) as f:
            alloc = json.load(f)["bits"]
        before_block = lambda i, block: quantize_experts_(block, alloc[i], 64, True, True)  # noqa: E731
    elif isinstance(cfg, tuple) and cfg[0] in ("gptq", "gptq_alloc"):
        before_block = None  # set below with the calibration batches
    else:
        before_block = (lambda i, block: quantize_experts_(block, *cfg)) if cfg else None
    if cli.layers:
        assert before_block is not None, "GPTQ configs quantize in before_batch"
        lo, hi = map(int, cli.layers.split(":"))
        only = before_block
        before_block = lambda i, block: only(i, block) if lo <= i < hi else None  # noqa: E731
    t0 = time.time()
    # expert choices of the uncompressed model on the eval batches: [batch][layer] -> [B*S, topk]. routing_v1.npz
    # cannot serve: it was traced before the bf16-operand GEMM change and differs from today's reference path
    routing_path = os.path.join(DATA, "eval_routing_ref.pt")
    before_batch, tag, cur = None, cli.config, {"batch": 0}
    if cfg == "codes":
        tag += "_" + os.path.basename(os.path.normpath(cli.codes)) + ("_tables" if cli.tables else "")
    if cli.layers:
        tag += "_L" + cli.layers.replace(":", "-")
    if cli.freeze_routing or cli.record_routing:
        assert not (cli.record_routing and (cfg is not None or cli.freeze_routing))
        inner = before_block
        if cli.freeze_routing:
            tag += "_frz"
            saved = torch.load(routing_path, weights_only=True)
            ref_idx = [torch.stack(layers, 1).long() for layers in saved]  # [B*S, n_layers, topk] per batch
        else:
            recorded = [[] for _ in range(len(seqs) // 2)]

        def before_block(i, block):
            if inner is not None:
                inner(i, block)
            if cli.freeze_routing:
                freeze_routing_(block, i, ref_idx, cur)
            else:
                block.ffn.gate.register_forward_hook(
                    lambda _m, _inp, out: recorded[cur["batch"]].append(out[1].to(torch.int16).cpu()))

        def before_batch(i, j):
            cur["batch"] = j

    stats = hedged = None
    if cli.flip_stats or cli.hedge:
        assert not (cli.freeze_routing or cli.record_routing or (isinstance(cfg, tuple) and cfg[0] in ("gptq", "gptq_alloc")))
        inner_t7 = before_block
        n_layers = runner.args.n_layers
        if cli.flip_stats:
            tag += "_flips"
            ref_idx = [torch.stack(layers, 1).long() for layers in torch.load(routing_path, weights_only=True)]
            stats = [torch.zeros(2, len(MARGIN_EDGES) + 1, dtype=torch.float64) for _ in range(n_layers)]
        else:
            tag += f"_hedge{cli.hedge:g}"
            with open(os.path.join(DATA, f"flips_{cli.config}.json")) as f:
                sigma = [cli.hedge * sg for sg in json.load(f)["sigma"]]
            hedged = torch.zeros(n_layers, 2, dtype=torch.float64)

        def before_block(i, block):
            if inner_t7 is not None:
                inner_t7(i, block)
            if stats is not None:
                flip_stats_(block, i, ref_idx, cur, stats)
            else:
                hedge_routing_(block, i, sigma[i], hedged)

        def before_batch(i, j):
            cur["batch"] = j

    batches = list(seqs.split(2))
    n_cal = 0
    if isinstance(cfg, tuple) and cfg[0] in ("gptq", "gptq_alloc"):
        assert not (cli.freeze_routing or cli.record_routing)
        from scripts.collect_bases import calib_indices

        pool_tokens = tokens
        if cli.calib_extra:
            extra = np.load(os.path.join(DATA, cli.calib_extra, "tokens.npz"))["tokens"]
            pool_tokens = torch.cat([tokens, torch.from_numpy(extra).long()])
            tag += "_" + cli.calib_extra
        if cli.calib_seqs:
            # every sequence but the eval ones
            pool = [i for i in range(pool_tokens.size(0)) if i >= tokens.size(0) or i % 6 != 3]
            chosen = [pool[i] for i in np.linspace(0, len(pool) - 1, min(cli.calib_seqs, len(pool))).round().astype(int)]
            tag += f"_cal{len(chosen)}"
        else:
            assert not cli.calib_extra, "--calib-extra needs --calib-seqs"
            chosen = calib_indices(tokens.size(0), 16)  # 15 sequences, none of them eval
        cal = list(pool_tokens[chosen].split(2))
        n_cal, batches = len(cal), cal + batches
        if cfg[0] == "gptq":
            plan = lambda i: [f"{cfg[1]}b64"] * runner.args.n_routed_experts  # noqa: E731
        else:
            with open(os.path.join(DATA, f"alloc_{cfg[1]}.json")) as f:
                plan = json.load(f)["options"].__getitem__
        if cli.save_codes:
            os.makedirs(cli.save_codes, exist_ok=True)
            shutil.copy(os.path.join(DATA, f"alloc_{cfg[1]}.json"), os.path.join(cli.save_codes, "alloc.json"))
        gq = GPTQExperts(plan, n_cal, sum(b.numel() for b in cal), cfg[2], save_dir=cli.save_codes)
        before_block, before_batch = gq.before_block, gq.before_batch
    ckpt = None
    if cli.checkpoint_every:
        assert n_cal, "the hooks of the other configs keep state across layers"
        ckpt = os.path.join(DATA, f"ckpt_{tag}.pt")
    hidden = runner.hidden_batches(batches, before_block=before_block, before_batch=before_batch, checkpoint=ckpt,
                                   every=cli.checkpoint_every or 4)[n_cal:]
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
    t_fwd = time.time() - t0

    norm = m.RMSNorm(runner.args.dim, runner.args.norm_eps).to(DEV)
    load_params_(norm, {"weight": runner.ckpt.get("norm.weight")}, "norm.")
    head = runner.ckpt.get("head.weight").to(DEV).float()
    ref_path = os.path.join(DATA, "eval_ref.pt")
    # the uncompressed run writes the reference; every other run (incl. "none" with frozen routing) compares to it
    ref = None if cfg is None and not cli.freeze_routing else torch.load(ref_path, weights_only=True)
    out_ids, out_lp = [], []
    acc = {k: np.zeros(seqs.size(0)) for k in ("nll", "n_tgt", "kl", "agree", "n_pos", "conf", "conf_ref")}  # per seq
    acc["kl_s"] = np.zeros((seqs.size(0), len(SCALES)))
    for b, h in enumerate(hidden):
        x = norm(h.to(DEV))
        rows = slice(2 * b, 2 * b + x.size(0))
        ids_b = seqs[rows].to(DEV)
        for s0 in range(0, x.size(1), CHUNK):
            z = F.linear(x[:, s0 : s0 + CHUNK].float(), head)
            lp = F.log_softmax(z, dim=-1)
            tgt = ids_b[:, s0 + 1 : s0 + 1 + CHUNK]
            acc["nll"][rows] += (-lp[:, : tgt.size(1)].gather(-1, tgt.unsqueeze(-1))).sum((1, 2)).cpu().numpy()
            acc["n_tgt"][rows] += tgt.size(1)
            if ref is None:
                top = lp.topk(TOPK, dim=-1)
                out_ids.append(top.indices.to(torch.int32).cpu())
                out_lp.append(top.values.to(torch.float16).cpu())
            else:
                r_ids = ref["ids"][len(out_ids)].to(DEV).long()
                r_lp = ref["lp"][len(out_ids)].to(DEV).float()
                out_ids.append(None)
                acc["kl"][rows] += (r_lp.exp() * (r_lp - lp.gather(-1, r_ids))).sum((1, 2)).cpu().numpy()
                acc["agree"][rows] += (lp.argmax(-1) == r_ids[..., 0]).sum(1).cpu().numpy()
                acc["n_pos"][rows] += lp.size(1)
                # does a global temperature explain part of the KL? (top-1 log-prob = how sharp each model is)
                acc["conf"][rows] += lp.amax(-1).sum(1).cpu().numpy()
                acc["conf_ref"][rows] += r_lp[..., 0].sum(1).cpu().numpy()
                for a_i, a in enumerate(SCALES):
                    lp_a = F.log_softmax(z * a, dim=-1).gather(-1, r_ids)
                    acc["kl_s"][rows, a_i] += (r_lp.exp() * (r_lp - lp_a)).sum((1, 2)).cpu().numpy()
    names = np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["names"]
    cats = np.load(os.path.join(DATA, "calib_v1", "tokens.npz"))["category"][3::6][:8]
    per_seq = [{"category": str(names[c]), "ppl": float(np.exp(acc["nll"][i] / acc["n_tgt"][i]))} for i, c in enumerate(cats)]
    result = {"config": tag, "tokens": int(seqs.numel()),
              "ppl": float(np.exp(acc["nll"].sum() / acc["n_tgt"].sum())), "forward_min": t_fwd / 60}
    if ref is None:
        torch.save({"ids": out_ids, "lp": out_lp}, ref_path)
    if cli.record_routing:
        torch.save(recorded, routing_path)
    else:
        n_pos = acc["n_pos"].sum()
        result.update({"kl_top64": acc["kl"].sum() / n_pos, "top1_agree": acc["agree"].sum() / n_pos,
                       "top1_logprob": acc["conf"].sum() / n_pos, "top1_logprob_ref": acc["conf_ref"].sum() / n_pos,
                       "kl_by_logit_scale": {str(a): acc["kl_s"][:, a_i].sum() / n_pos for a_i, a in enumerate(SCALES)}})
        for i, s in enumerate(per_seq):
            s.update({"kl": float(acc["kl"][i] / acc["n_pos"][i]), "top1": float(acc["agree"][i] / acc["n_pos"][i])})
    result["per_seq"] = per_seq
    if stats is not None:
        n, k = torch.stack([st[0] for st in stats]), torch.stack([st[1] for st in stats])
        sig = [fit_sigma(n[i], k[i]) for i in range(len(stats))]
        result.update({"flip_rate": float(k.sum() / n.sum()), "sigma": sig})
        with open(os.path.join(DATA, f"flips_{cli.config}.json"), "w") as f:
            json.dump({"edges": MARGIN_EDGES.tolist(), "tokens": n.tolist(), "flips": k.tolist(), "sigma": sig}, f)
    if hedged is not None:
        result["hedged_share"] = float(hedged[:, 0].sum() / hedged[:, 1].sum())
    with open(os.path.join(DATA, f"eval_{tag}.json"), "w") as f:
        json.dump(result, f, indent=1)
    if cli.save_codes:
        shutil.copy(os.path.join(DATA, f"eval_{tag}.json"), os.path.join(cli.save_codes, "eval.json"))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
