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
        self.signs = (torch.randint(0, 2, (k,), generator=g) * 2 - 1).float().to(device)

    def _h(self, w: torch.Tensor) -> torch.Tensor:
        n, k = w.shape
        return fwht_(w.reshape(n, k // self.b, self.b).clone()).reshape(n, k) / self.b**0.5

    def rotate(self, w: torch.Tensor) -> torch.Tensor:
        return self._h(w.float() * self.signs)

    def unrotate(self, w: torch.Tensor) -> torch.Tensor:
        return self._h(w.float()) * self.signs


def rtn_codes(w: torch.Tensor, bits: int, group: int, clip_grid: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric mid-rise uniform quantization to 2**bits levels per `group` weights along K, with the clip ratio
    searched per group for minimum MSE. Returns codes (uint8 [N, K], in [0, 2**bits)) and the fp16-representable
    step per group (fp32 [N, K/group]). Storage: bits + 16/group bpw."""
    n, k = w.shape
    g = w.float().reshape(n, k // group, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    half = 2 ** (bits - 1)  # levels at (i + 0.5) * step for i in [-half, half)
    best_step, best_err = None, None
    for r in torch.linspace(0.5, 1.0, clip_grid, device=w.device):
        step = (amax * r / (half - 0.5)).half().float()  # the step is stored as fp16
        q = (torch.floor(g / step).clamp(-half, half - 1) + 0.5) * step
        err = (q - g).pow(2).sum(-1, keepdim=True)
        if best_step is None:
            best_step, best_err = step, err
        else:
            take = err < best_err
            best_step, best_err = torch.where(take, step, best_step), torch.minimum(err, best_err)
    codes = (torch.floor(g / best_step).clamp(-half, half - 1) + half).to(torch.uint8)
    return codes.reshape(n, k), best_step.squeeze(-1)


def rtn_dequant(codes: torch.Tensor, step: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    half = 2 ** (bits - 1)
    return (codes.float() - half + 0.5) * step.float().repeat_interleave(group, -1)


def rtn(w: torch.Tensor, bits: int, group: int, clip_grid: int = 8) -> torch.Tensor:
    """Dequantized result of rtn_codes (fp32)."""
    return rtn_dequant(*rtn_codes(w, bits, group, clip_grid), bits, group)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes < 16, [..., K] -> [..., K/2] (even index in the low nibble)."""
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack([packed & 0x0F, packed >> 4], dim=-1).flatten(-2)


class RTNLinear(torch.nn.Module):
    """Drop-in for an upstream expert Linear: RTN codes packed 4-bit with per-group fp16 steps, optionally kept in a
    block-Hadamard-rotated basis (activations are rotated instead of the weight). Applies the same fp8 activation
    fake-quant upstream uses before its quantized GEMMs, then an fp32 GEMM (rotated activations are not exact in
    bf16, and a measurement tool should not add its own rounding)."""

    def __init__(self, w: torch.Tensor, bits: int, group: int, rot: "BlockHadamard | None" = None):
        super().__init__()
        assert bits <= 4
        self.bits, self.group, self.rot = bits, group, rot
        codes, step = rtn_codes(rot.rotate(w) if rot else w.float(), bits, group)
        self.register_buffer("packed", pack_nibbles(codes))
        self.register_buffer("step", step.half())

    def weight_dequant(self) -> torch.Tensor:
        """The effective weight in the rotated basis (fp32)."""
        return rtn_dequant(unpack_nibbles(self.packed), self.step, self.bits, self.group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from . import upstream  # noqa: F401  (puts kernel on sys.path)
        import kernel

        xq = kernel.act_quant(x.to(torch.bfloat16).contiguous().clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
        xq = xq.float()
        if self.rot is not None:
            xq = self.rot.rotate(xq.reshape(-1, xq.size(-1))).reshape(xq.shape)
        return F.linear(xq, self.weight_dequant()).to(torch.get_default_dtype())


def kmeanspp_init(x: torch.Tensor, n_codes: int, seed: int = 0, max_points: int = 64) -> torch.Tensor:
    """k-means++ seeding on a random subsample of at most max_points * n_codes rows."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sub = x[torch.randperm(x.size(0), generator=g)[: max_points * n_codes].to(x.device)]
    codes = [sub[torch.randint(0, sub.size(0), (1,), generator=g).item()]]
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
