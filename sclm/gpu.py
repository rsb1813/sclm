# ctypes binding of the engine's GPU kernels (engine/gpu.cu), called on PyTorch device memory: the kernels run in
# PyTorch's CUDA context on its default stream, in order with its own operations.
import ctypes as C
import os

import torch

DLL = os.path.join(os.path.dirname(__file__), "..", "engine", "build", "sclm_gpu.dll")
active = False  # the host turns this on around a forward whose single-token steps should use the kernels
_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = C.CDLL(os.path.abspath(DLL))  # finds the CUDA runtime torch has already loaded
        _lib.sclm_gpu_act_quant.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
        _lib.sclm_gpu_fp8_linear.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        _lib.sclm_gpu_fp4_expert.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_float, C.c_void_p, C.c_float, C.c_void_p]
        _lib.sclm_gpu_hc_mixes.argtypes = [C.c_void_p] * 5 + [C.c_int, C.c_int, C.c_float, C.c_float, C.c_int]
        _lib.sclm_gpu_hc_pre_norm.argtypes = [C.c_void_p] * 4 + [C.c_int, C.c_int, C.c_float]
        _lib.sclm_gpu_hc_post.argtypes = [C.c_void_p] * 5 + [C.c_int, C.c_int]
    return _lib


def _check(status: int) -> None:
    assert status == 0, f"GPU kernel launch failed ({status})"


def act_quant(x: torch.Tensor) -> torch.Tensor:
    """The reference's fp8 fake-quant (blocks of 32, power-of-two scales) of a bf16 vector, as fp32."""
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.numel() % 32 == 0
    xq = torch.empty(x.numel(), dtype=torch.float32, device=x.device)
    _check(lib().sclm_gpu_act_quant(x.data_ptr(), xq.data_ptr(), x.numel()))
    return xq


def fp8_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, groups: int = 1) -> torch.Tensor:
    """upstream linear() for one token and an fp8 weight [n, k] with block scales: x (bf16, groups * k values) ->
    bf16 [n]. With groups > 1 the weight is block-diagonal: rows i * n / groups .. multiply x's i-th k values."""
    n, k = weight.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.numel() == groups * k and weight.is_contiguous()
    y = torch.empty(n, dtype=torch.bfloat16, device=x.device)
    _check(lib().sclm_gpu_fp8_linear(weight.data_ptr(), scale.data_ptr(), n, k, groups, x.data_ptr(), y.data_ptr()))
    return y


def fp4_expert(slot: torch.Tensor, dim: int, inter: int, limit: float, xq: torch.Tensor, weight: float, y: torch.Tensor) -> None:
    """y (fp32 [dim]) += weight * the routed expert held in `slot` (its checkpoint bytes: three scales, then three
    packed-FP4 weights) applied to the layer input quantized as xq (act_quant)."""
    assert slot.is_contiguous() and xq.dtype == y.dtype == torch.float32 and xq.numel() == y.numel() == dim
    _check(lib().sclm_gpu_fp4_expert(slot.data_ptr(), dim, inter, limit, xq.data_ptr(), weight, y.data_ptr()))


def hc_mixes(h: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, norm_eps: float,
             eps: float, iters: int) -> torch.Tensor:
    """upstream Block.hc_mixes for one token: the stream h (bf16, hc * dim values) -> fp32 [(2 + hc) * hc], which is
    pre[hc], post[hc] and comb[hc][hc] back to back."""
    assert h.dtype == torch.bfloat16 and h.is_contiguous() and fn.dtype == scale.dtype == base.dtype == torch.float32
    out = torch.empty((2 + hc) * hc, dtype=torch.float32, device=h.device)
    _check(lib().sclm_gpu_hc_mixes(h.data_ptr(), fn.data_ptr(), scale.data_ptr(), base.data_ptr(), out.data_ptr(),
                                   h.numel() // hc, hc, norm_eps, eps, iters))
    return out


def hc_pre_norm(h: torch.Tensor, pre: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm (bf16 weight [dim]) of upstream Block.hc_pre(h, pre) for one token -> bf16 [dim]."""
    hc, dim = pre.numel(), weight.numel()
    assert h.dtype == weight.dtype == torch.bfloat16 and h.is_contiguous() and pre.dtype == torch.float32 and pre.is_contiguous()
    out = torch.empty(dim, dtype=torch.bfloat16, device=h.device)
    _check(lib().sclm_gpu_hc_pre_norm(h.data_ptr(), pre.data_ptr(), weight.data_ptr(), out.data_ptr(), dim, hc, eps))
    return out


def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """upstream Block.hc_post for one token: sublayer output x (bf16 [dim]) and the stream before it -> bf16 [hc, dim]."""
    hc, dim = post.numel(), x.numel()
    assert x.dtype == residual.dtype == torch.bfloat16 and x.is_contiguous() and residual.is_contiguous()
    assert post.is_contiguous() and comb.is_contiguous() and comb.numel() == hc * hc
    out = torch.empty(hc, dim, dtype=torch.bfloat16, device=x.device)
    _check(lib().sclm_gpu_hc_post(x.data_ptr(), residual.data_ptr(), post.data_ptr(), comb.data_ptr(), out.data_ptr(), dim, hc))
    return out
