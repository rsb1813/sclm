# Stands in for the fast_hadamard_transform package that upstream's rotate_activation imports (a CUDA extension that
# does not build here): the Walsh-Hadamard transform over the last dimension, in pure PyTorch.
import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """x @ H * scale over the last dimension (a power of two), H the Sylvester Hadamard matrix. Computed in fp32 and
    returned in x's dtype (upstream feeds bf16; the package's own rounding inside the transform is not known here)."""
    n = x.size(-1)
    assert n & (n - 1) == 0
    y, h = x.float(), 1
    while h < n:
        y = y.unflatten(-1, (-1, 2, h))
        y = torch.stack([y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]], dim=-2).flatten(-3)
        h *= 2
    return (y * scale).to(x.dtype)
