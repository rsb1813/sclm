# Weight quantizers for compression experiments: grouped RTN, block Hadamard rotation, and shared-codebook vector quantization.
import torch
import torch.nn.functional as F


def largest_pow2_divisor(n: int, cap: int = 1024) -> int:
    b = 1
    while n % (b * 2) == 0 and b * 2 <= cap:
        b *= 2
    return b


def fwht_(x: torch.Tensor) -> torch.Tensor:
    """In-place unnormalized fast Walsh-Hadamard transform over the last dim (a power of 2)."""
    n = x.size(-1)
    h = 1
    while h < n:
        y = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :].clone(), y[..., 1, :]
        y[..., 0, :] += b
        y[..., 1, :] = a - b
        h *= 2
    return x


class BlockHadamard:
    """Orthogonal transform R = diag(signs) * blockdiag(H_b / sqrt(b)) along the input dim of a [N, K] weight.
    W @ x == (W R) @ (R^T x), so quantizing W R and rotating activations by R^T is lossless before quantization."""

    def __init__(self, k: int, seed: int, device="cuda"):
        self.b = largest_pow2_divisor(k)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.signs = (torch.randint(0, 2, (k,), generator=g, device="cpu") * 2 - 1).float().to(device)

    def _h(self, w: torch.Tensor) -> torch.Tensor:
        n, k = w.shape
        return fwht_(w.reshape(n, k // self.b, self.b).clone()).reshape(n, k) / self.b**0.5

    def rotate(self, w: torch.Tensor) -> torch.Tensor:
        return self._h(w.float() * self.signs)

    def unrotate(self, w: torch.Tensor) -> torch.Tensor:
        return self._h(w.float()) * self.signs


def _grid(levels: int) -> tuple[int, int, float]:
    """(lo, hi, offset): index i in [lo, hi] reconstructs to (i + offset) * step. An even level count is mid-rise
    (offset 0.5, no zero level, e.g. 2**bits); an odd one is mid-tread (zero included; 3 levels = ternary)."""
    if levels % 2 == 0:
        return -levels // 2, levels // 2 - 1, 0.5
    return -(levels // 2), levels // 2, 0.0


def _index(x: torch.Tensor, step: torch.Tensor, levels: int) -> torch.Tensor:
    lo, hi, off = _grid(levels)
    return (torch.floor(x / step) if off else torch.round(x / step)).clamp(lo, hi)


def rtn_codes(w: torch.Tensor, bits: int, group: int, clip_grid: int = 8,
              levels: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric uniform quantization to `levels` (default 2**bits) levels per `group` weights along K, with the clip
    ratio searched per group for minimum MSE. Returns codes (uint8 [N, K], in [0, levels)) and the fp16-representable
    step per group (fp32 [N, K/group]). Storage: log2(levels) + 16/group bpw."""
    levels = levels or 2**bits
    n, k = w.shape
    g = w.float().reshape(n, k // group, group)
    best_step = _best_step(g, bits, clip_grid, levels)
    codes = (_index(g, best_step, levels) - _grid(levels)[0]).to(torch.uint8)
    return codes.reshape(n, k), best_step.squeeze(-1)


def _best_step(g: torch.Tensor, bits: int, clip_grid: int = 8, levels: int | None = None) -> torch.Tensor:
    """Per-group step [..., 1] (fp16-representable) minimizing the group's RTN MSE over a grid of clip ratios. With
    2 or 3 levels the best outer level sits far inside the group max (~0.33x for 1 bit), so the grid starts lower."""
    levels = levels or 2**bits
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    _, hi, off = _grid(levels)
    lo_r, n_r = (0.5, clip_grid) if levels > 3 else (0.2, 2 * clip_grid)
    best_step, best_err = None, None
    for r in torch.linspace(lo_r, 1.0, n_r, device=g.device, dtype=torch.float32):
        step = (amax * r / (hi + off)).half().float()  # the outer level is (hi + off) * step; stored as fp16
        q = (_index(g, step, levels) + off) * step
        err = (q - g).pow(2).sum(-1, keepdim=True)
        if best_step is None:
            best_step, best_err = step, err
        else:
            take = err < best_err
            best_step, best_err = torch.where(take, step, best_step), torch.minimum(err, best_err)
    return best_step


def gptq_codes(
    w: torch.Tensor, h: torch.Tensor, bits: int, group: int, damp: float = 0.01, block: int = 128,
    levels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ with static groups: quantize columns in order, pushing each column's error onto the not-yet-quantized
    columns through the Cholesky factor of H^-1 (H = X^T X of the layer input), so the error lands where inputs
    rarely point. Same output format as rtn_codes. w [..., N, K] and h [..., K, K] may carry leading batch dims
    (e.g. a group of experts), processed together. `block` must be a multiple of `group`."""
    assert block % group == 0
    *batch, n, k = w.shape
    W, H = w.float().clone(), h.float().clone()
    diag = torch.diagonal(H, dim1=-2, dim2=-1)  # a view: edits below land in H
    dead = diag == 0
    diag.masked_fill_(dead, 1)
    W.masked_fill_(dead.unsqueeze(-2), 0)
    diag += damp * diag.mean(-1, keepdim=True)
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    levels = levels or 2**bits
    lo, _, off = _grid(levels)
    codes = torch.empty(*batch, n, k, dtype=torch.uint8, device=w.device)
    steps = torch.empty(*batch, n, k // group, device=w.device, dtype=torch.float32)  # callers may default to bf16
    for b0 in range(0, k, block):
        b1 = min(b0 + block, k)
        w1, err = W[..., b0:b1].clone(), torch.zeros(*batch, n, b1 - b0, device=w.device, dtype=torch.float32)
        ub = u[..., b0:b1, b0:b1]
        for i in range(b1 - b0):
            col = b0 + i
            if col % group == 0:  # the group's step comes from its columns after earlier error feedback
                step = _best_step(w1[..., i : i + group], bits, levels=levels)
                steps[..., col // group] = step.squeeze(-1)
            idx = _index(w1[..., i : i + 1], step, levels)
            codes[..., col] = (idx - lo).squeeze(-1).to(torch.uint8)
            e = (w1[..., i : i + 1] - (idx + off) * step) / ub[..., i : i + 1, i : i + 1]
            w1[..., i:] -= e * ub[..., i : i + 1, i:]
            err[..., i : i + 1] = e
        W[..., b1:] -= err @ u[..., b0:b1, b1:]
    return codes, steps


def gptq_vq(
    w: torch.Tensor, h: torch.Tensor, codebook: torch.Tensor, group: int, damp: float = 0.01, block: int = 128,
    refit: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ with vector rounding: d columns at a time go to the row of `codebook` [C, d] (times the group's scale)
    that is nearest in the metric the error feedback leaves on them, |delta U_BB^-1|^2 with U the Cholesky factor
    gptq_codes uses, i.e. the cost that remains after the later columns have compensated. A group's scale is the rms
    of its columns once earlier feedback has landed (fp16; group >= K: one per row, the rms of the row as given, or
    with `refit` the least-squares scale of the chosen codes under h). Batch dims as in gptq_codes.
    Returns codes (long [..., N, K/d]) and scales (fp32 [..., N, K/group or 1])."""
    d = codebook.size(-1)
    *batch, n, k = w.shape
    per_row = group >= k
    assert block % d == 0 and k % d == 0 and (per_row or (block % group == 0 and group % d == 0))
    assert per_row or not refit
    W, H = w.float().clone(), h.float().clone()
    diag = torch.diagonal(H, dim1=-2, dim2=-1)  # a view: edits below land in H
    dead = diag == 0
    diag.masked_fill_(dead, 1)
    W.masked_fill_(dead.unsqueeze(-2), 0)
    diag += damp * diag.mean(-1, keepdim=True)
    u = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    w0 = W.clone() if refit else None
    cb = codebook.float()
    # every code's d x d diagonal block of U inverted at once, and the codewords in each block's metric: one
    # linalg.inv per code was a fifth of the run time
    blocks = u.unflatten(-2, (k // d, d)).unflatten(-1, (k // d, d)).diagonal(dim1=-4, dim2=-2).movedim(-1, -3)
    inv_all = torch.linalg.inv(blocks)  # [..., K/d, d, d]
    ct_all = cb @ inv_all  # [..., K/d, C, d]
    ct2_all = ct_all.pow(2).sum(-1)
    del blocks
    rms = lambda g: g.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12).half().float()  # noqa: E731
    codes = torch.empty(*batch, n, k // d, dtype=torch.long, device=w.device)
    scales = torch.empty(*batch, n, 1 if per_row else k // group, device=w.device, dtype=torch.float32)
    if per_row:
        scale = rms(W)
        scales[..., 0] = scale.squeeze(-1)
    for b0 in range(0, k, block):
        b1 = min(b0 + block, k)
        w1, err = W[..., b0:b1].clone(), torch.zeros(*batch, n, b1 - b0, device=w.device, dtype=torch.float32)
        ub = u[..., b0:b1, b0:b1]
        for i in range(0, b1 - b0, d):
            col = b0 + i
            if not per_row and col % group == 0:
                scale = rms(w1[..., i : i + group])
                scales[..., col // group] = scale.squeeze(-1)
            # targets [..., N, d] and codewords [..., C, d] in the metric
            t, ct = w1[..., i : i + d] @ inv_all[..., col // d, :, :], ct_all[..., col // d, :, :]
            # |t - scale * c|^2 up to terms and a factor that do not depend on the codeword
            j = (scale / 2 * ct2_all[..., col // d, :].unsqueeze(-2) - t @ ct.transpose(-1, -2)).argmin(-1)
            codes[..., col // d] = j
            e = t - scale * ct.gather(-2, j.unsqueeze(-1).expand(*j.shape, d))
            w1[..., i + d :] -= e @ ub[..., i : i + d, i + d :]
            err[..., i : i + d] = e
        W[..., b1:] -= err @ u[..., b0:b1, b1:]
    if refit:
        q = cb[codes].flatten(-2)
        qh = q @ H
        scales = ((qh * w0).sum(-1, keepdim=True) / (qh * q).sum(-1, keepdim=True).clamp_min(1e-12)).half().float()
    return codes, scales


def vq_codes_dequant(codes: torch.Tensor, scales: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Weights [..., N, K] of gptq_vq's codes and scales."""
    q = codebook.float()[codes]  # [..., N, K/d, d]
    return (q.unflatten(-2, (scales.size(-1), -1)) * scales[..., None, None]).flatten(-3)


def rtn_dequant(codes: torch.Tensor, step: torch.Tensor, bits: int, group: int,
                levels: int | None = None) -> torch.Tensor:
    lo, _, off = _grid(levels or 2**bits)
    return (codes.float() + lo + off) * step.float().repeat_interleave(group, -1)


def rtn(w: torch.Tensor, bits: int, group: int, clip_grid: int = 8, levels: int | None = None) -> torch.Tensor:
    """Dequantized result of rtn_codes (fp32)."""
    return rtn_dequant(*rtn_codes(w, bits, group, clip_grid, levels), bits, group, levels)


def refit_steps(q: torch.Tensor, w: torch.Tensor, sets: list, group: int, ridge: float = 1e-4,
                chunk: int = 512) -> torch.Tensor:
    """Least-squares group steps for fixed levels q [N, K] (the weight in units of its step, e.g. rtn_dequant with unit
    steps). Per row, minimizes over token sets (x [T, K], wt [T], mix) the sum of
    mix / tr * sum_t wt_t (x_t.(q*s) - x_t.w)^2, with tr = sum_t wt_t |x_t|^2 / K normalizing each set the way tr_norm
    does a Hessian. The output is linear in the steps, so this is one G x G system per row (G = K / group). Returns
    s [N, G] (fp32)."""
    n, k = q.shape
    g = k // group
    qg = q.float().reshape(n, g, group)
    a = torch.zeros(n, g, g, device=q.device, dtype=torch.float32)
    b = torch.zeros(n, g, device=q.device, dtype=torch.float32)
    for x, wt, mix in sets:
        x, wt = x.float(), wt.float()
        tr = float((wt[:, None] * x.pow(2)).sum()) / k
        if tr <= 0:
            continue
        for t0 in range(0, x.size(0), chunk):
            xc, wc = x[t0 : t0 + chunk], wt[t0 : t0 + chunk] * (mix / tr)
            z = torch.einsum("tgk,ngk->tng", xc.reshape(-1, g, group), qg)  # each group's output per unit step
            y = xc @ w.float().t()
            a += torch.einsum("tng,tnh->ngh", z * wc[:, None, None], z)
            b += torch.einsum("tng,tn->ng", z * wc[:, None, None], y)
    a += ridge * torch.diagonal(a, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)[:, None, None] * torch.eye(
        g, device=a.device, dtype=torch.float32)
    return torch.linalg.solve(a, b)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes < 16, [..., K] -> [..., K/2] (even index in the low nibble)."""
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2)


def pack_crumbs(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes < 4, [..., K] -> [..., K/4] (index 4i in the lowest two bits)."""
    return codes[..., 0::4] | (codes[..., 1::4] << 2) | (codes[..., 2::4] << 4) | (codes[..., 3::4] << 6)


def unpack_crumbs(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack([packed & 3, (packed >> 2) & 3, (packed >> 4) & 3, packed >> 6], dim=-1).flatten(-2)


class RTNLinear(torch.nn.Module):
    """Drop-in for an upstream expert Linear: RTN codes (packed 4 per byte up to 2 bits, else 2 per byte) with
    per-group fp16 steps, optionally kept in a block-Hadamard-rotated basis (activations are rotated instead of the
    weight). Applies the same fp8 activation fake-quant upstream uses before its quantized GEMMs, then an fp32 GEMM
    (rotated activations are not exact in bf16, and a measurement tool should not add its own rounding).

    unbiased: write w_hat = beta * w + n per output row (n orthogonal to w) and store w_hat / beta, so the row keeps
    its full gain. Low-bit rounding shrinks rows (beta ~0.89 at 2 bits), and SwiGLU multiplies the shrinkage of
    w1, w3 and w2 into the expert output (~0.70 at 2 bits). Costs nothing: 1/beta is folded into the row's steps."""

    def __init__(self, w: torch.Tensor, bits: int, group: int, rot: "BlockHadamard | None" = None,
                 unbiased: bool = False, levels: int | None = None):
        super().__init__()
        self.bits, self.group, self.rot, self.levels = bits, group, rot, levels or 2**bits
        assert self.levels <= 16
        wr = rot.rotate(w) if rot else w.float()
        codes, step = rtn_codes(wr, bits, group, levels=self.levels)
        if unbiased:
            wq = rtn_dequant(codes, step, bits, group, self.levels)
            beta = (wq * wr).sum(1) / (wr * wr).sum(1).clamp_min(1e-30)
            step = step / beta.clamp_min(0.05)[:, None]
        self.register_buffer("packed", pack_crumbs(codes) if self.levels <= 4 else pack_nibbles(codes))
        self.register_buffer("step", step.half())

    @classmethod
    def from_codes(cls, codes: torch.Tensor, step: torch.Tensor, bits: int, group: int,
                   rot: "BlockHadamard | None" = None, levels: int | None = None) -> "RTNLinear":
        """Wrap codes and steps made elsewhere (e.g. gptq_codes; in the rotated basis when rot is given)."""
        self = cls.__new__(cls)
        torch.nn.Module.__init__(self)
        self.bits, self.group, self.rot, self.levels = bits, group, rot, levels or 2**bits
        self.register_buffer("packed", pack_crumbs(codes) if self.levels <= 4 else pack_nibbles(codes))
        self.register_buffer("step", step.half())
        return self

    def weight_dequant(self) -> torch.Tensor:
        """The effective weight in the rotated basis (fp32)."""
        codes = unpack_crumbs(self.packed) if self.levels <= 4 else unpack_nibbles(self.packed)
        return rtn_dequant(codes, self.step, self.bits, self.group, self.levels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from . import upstream  # noqa: F401  (puts kernel on sys.path)
        import kernel

        xq = kernel.act_quant(x.to(torch.bfloat16).contiguous().clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
        xq = xq.float()
        if self.rot is not None:
            xq = self.rot.rotate(xq.reshape(-1, xq.size(-1))).reshape(xq.shape)
        return F.linear(xq, self.weight_dequant()).to(torch.get_default_dtype())


class VQLinear(RTNLinear):
    """RTNLinear's drop-in role for gptq_vq's output: one code byte per codebook vector, one fp16 scale per row (or
    per group) and the codebook [C <= 256, d] the layer's experts share, in the block-Hadamard-rotated basis."""

    def __init__(self, codes: torch.Tensor, scales: torch.Tensor, codebook: torch.Tensor,
                 rot: "BlockHadamard | None" = None):
        torch.nn.Module.__init__(self)
        assert codebook.size(0) <= 256
        self.rot, self.codebook = rot, codebook
        self.register_buffer("codes", codes.to(torch.uint8))
        self.register_buffer("scales", scales.half())

    def weight_dequant(self) -> torch.Tensor:
        return vq_codes_dequant(self.codes.long(), self.scales.float(), self.codebook)


def kmeanspp_init(x: torch.Tensor, n_codes: int, seed: int = 0, max_points: int = 64) -> torch.Tensor:
    """k-means++ seeding on a random subsample of at most max_points * n_codes rows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sub = x[torch.randperm(x.size(0), generator=g, device="cpu")[: max_points * n_codes].to(x.device)]
    codes = [sub[torch.randint(0, sub.size(0), (1,), generator=g, device="cpu").item()]]
    d2 = (sub - codes[0]).pow(2).sum(-1)
    for _ in range(n_codes - 1):
        nxt = sub[torch.multinomial(d2 / d2.sum(), 1).item()]
        codes.append(nxt)
        d2 = torch.minimum(d2, (sub - nxt).pow(2).sum(-1))
    return torch.stack(codes)


def kmeans(x: torch.Tensor, n_codes: int, iters: int = 15, chunk: int = 1 << 16, seed: int = 0) -> torch.Tensor:
    """k-means (k-means++ seeded) on rows of x [m, d] (fp32, GPU). Returns the codebook [n_codes, d]."""
    codebook = kmeanspp_init(x, n_codes, seed)
    for _ in range(iters):
        assign = vq_assign(x, codebook, chunk)
        sums = torch.zeros_like(codebook).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=n_codes).float().unsqueeze(-1)
        empty = counts.squeeze(-1) == 0
        codebook = torch.where(empty.unsqueeze(-1), codebook, sums / counts.clamp_min(1))
        if empty.any():  # re-seed dead codes from random points
            codebook[empty] = x[torch.randint(0, x.size(0), (int(empty.sum()),), device=x.device)]
    return codebook


def vq_assign(x: torch.Tensor, codebook: torch.Tensor, chunk: int = 1 << 16) -> torch.Tensor:
    """Index of the nearest code for each row of x."""
    c2 = codebook.pow(2).sum(-1)
    out = torch.empty(x.size(0), dtype=torch.long, device=x.device)
    for s in range(0, x.size(0), chunk):
        xs = x[s : s + chunk]
        out[s : s + chunk] = (c2 - 2 * xs @ codebook.t()).argmin(-1)
    return out


def row_normalize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w / rms(row) and the per-row rms (stored as fp16: 16/K bpw, negligible)."""
    s = w.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return w.float() / s, s


def to_vectors(w: torch.Tensor, dim: int) -> torch.Tensor:
    """[N, K] -> [N * K / dim, dim], consecutive weights along K form a vector."""
    return w.reshape(-1, dim)


def vq_dequant(w_norm: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Quantize a row-normalized [N, K] weight with a codebook of dim d; returns the dequantized normalized weight."""
    d = codebook.size(1)
    return codebook[vq_assign(to_vectors(w_norm, d), codebook)].reshape(w_norm.shape)
