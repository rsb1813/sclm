# Checks the quantizers: Hadamard orthogonality, RTN error ordering, and k-means/VQ behavior on synthetic data.
import torch

from sclm import quant

DEV = "cuda"


def test_fwht_matches_dense_hadamard():
    n = 8
    h = torch.tensor([[1.0]])
    while h.size(0) < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    x = torch.randn(3, n, device=DEV)
    assert torch.allclose(quant.fwht_(x.clone()), x @ h.to(DEV).t(), atol=1e-5)


def test_block_hadamard_is_orthogonal_and_preserves_products():
    k = 5120  # 5 x 1024 blocks
    r = quant.BlockHadamard(k, seed=0)
    assert r.b == 1024
    w = torch.randn(16, k, device=DEV)
    x = torch.randn(k, 4, device=DEV)
    assert torch.allclose(r.unrotate(r.rotate(w)), w, atol=1e-4)
    # (W R)(R^T x) == W x, with R^T x obtained by rotating x^T as rows
    assert torch.allclose(r.rotate(w) @ r.rotate(x.t()).t(), w @ x, atol=1e-2, rtol=1e-4)


def test_rtn_error_drops_with_bits():
    w = torch.randn(64, 512, device=DEV)
    errs = [((quant.rtn(w, b, 64) - w).pow(2).mean() / w.pow(2).mean()).item() for b in (2, 3, 4)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.02  # 4-bit Gaussian RTN NMSE is ~0.01


def test_rtn_codes_roundtrip_and_packing():
    w = torch.randn(32, 256, device=DEV)
    codes, step = quant.rtn_codes(w, 3, 64)
    assert codes.dtype == torch.uint8 and int(codes.max()) < 8 and step.shape == (32, 4)
    assert torch.equal(quant.unpack_nibbles(quant.pack_nibbles(codes)), codes)
    assert torch.equal(quant.rtn(w, 3, 64), quant.rtn_dequant(codes, step, 3, 64))


def test_rtn_linear_matches_dense_rotated_weight():
    """y = Q(W R) (R^T act_q(x)) must equal the dense dequantized weight R-unrotated, applied to act_q(x)."""
    import sclm.upstream  # noqa: F401
    import kernel

    torch.manual_seed(5)
    w, x = torch.randn(64, 512, device=DEV) * 0.02, torch.randn(10, 512, device=DEV)
    for rot in (None, quant.BlockHadamard(512, seed=3)):
        lin = quant.RTNLinear(w, 3, 64, rot)
        w_eff = rot.unrotate(lin.weight_dequant()) if rot else lin.weight_dequant()
        xq = kernel.act_quant(x.to(torch.bfloat16).clone(), 32, "ue8m0", torch.float8_e8m0fnu, inplace=True).float()
        want = xq @ w_eff.t()
        got = lin(x).float()
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-5), (got - want).abs().max()
        assert ((w_eff - w).pow(2).mean() / w.pow(2).mean()) < 0.06  # 3-bit RTN error, rotated or not


def test_kmeans_recovers_clusters_and_vq_error():
    torch.manual_seed(0)
    centers = torch.randn(16, 4, device=DEV) * 5
    x = centers[torch.randint(0, 16, (20000,), device=DEV)] + 0.05 * torch.randn(20000, 4, device=DEV)
    cb = quant.kmeans(x, 16, iters=25)
    err = (cb[quant.vq_assign(x, cb)] - x).pow(2).mean() / x.pow(2).mean()
    assert err < 0.01
    w = torch.randn(32, 256, device=DEV)
    wn, s = quant.row_normalize(w)
    cb = quant.kmeans(quant.to_vectors(wn, 8), 256)
    nmse = ((quant.vq_dequant(wn, cb) * s - w).pow(2).mean() / w.pow(2).mean()).item()
    assert 0.1 < nmse < 0.6  # 1 bpw on Gaussian data: well above 4-bit RTN, well below no-information (1.0)
