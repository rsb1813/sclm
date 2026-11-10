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


def test_seeded_helpers_work_under_cuda_default_device():
    # the layerwise runner executes blocks inside torch.device("cuda"); CPU-generator draws must still work there
    with torch.device(DEV):
        r = quant.BlockHadamard(2304, seed=1)
        cb = quant.kmeans(torch.randn(4096, 8), 16, iters=2)
    assert r.signs.device.type == "cuda" and cb.shape == (16, 8)


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


def test_gptq_beats_rtn_on_correlated_inputs():
    torch.manual_seed(6)
    k, n, m = 256, 64, 4096
    x = torch.randn(m, 32, device=DEV) @ torch.randn(32, k, device=DEV) + 0.1 * torch.randn(m, k, device=DEV)
    w = torch.randn(n, k, device=DEV)
    h = x.t() @ x
    out_err = lambda wq: ((wq - w) @ x.t()).pow(2).sum() / (w @ x.t()).pow(2).sum()  # noqa: E731
    codes, step = quant.gptq_codes(w, h, 3, 64)
    assert codes.dtype == torch.uint8 and int(codes.max()) < 8 and step.shape == (n, 4)
    e_gptq, e_rtn = out_err(quant.rtn_dequant(codes, step, 3, 64)), out_err(quant.rtn(w, 3, 64))
    assert e_gptq < 0.5 * e_rtn, (e_gptq, e_rtn)


def test_gptq_batched_matches_one_at_a_time():
    torch.manual_seed(7)
    w = torch.randn(3, 32, 128, device=DEV)
    x = torch.randn(3, 512, 128, device=DEV)
    h = x.transpose(1, 2) @ x
    h[1, :, 5] = h[1, 5, :] = 0  # a dead input column in one item
    codes, step = quant.gptq_codes(w, h, 2, 64)
    for b in range(3):
        c1, s1 = quant.gptq_codes(w[b], h[b], 2, 64)
        assert torch.equal(codes[b], c1) and torch.allclose(step[b], s1)


def test_gptq_ignores_bf16_default_dtype():
    # probe and eval scripts set the default dtype to bf16; GPTQ's internal buffers must stay fp32
    torch.manual_seed(8)
    w = torch.randn(32, 128, device=DEV)
    x = torch.randn(512, 128, device=DEV)
    ref = quant.gptq_codes(w, x.t() @ x, 2, 64)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        got = quant.gptq_codes(w, x.t() @ x, 2, 64)
    finally:
        torch.set_default_dtype(prev)
    assert torch.equal(got[0], ref[0]) and got[1].dtype == torch.float32 and torch.equal(got[1], ref[1])


def test_rtn_linear_unbiased_restores_row_gain():
    torch.manual_seed(9)
    w = torch.randn(64, 256, device=DEV)
    rot = quant.BlockHadamard(256, seed=3)
    for unbiased in (False, True):
        lin = quant.RTNLinear(w, 2, 64, rot, unbiased=unbiased)
        wq = rot.unrotate(lin.weight_dequant())
        beta = (wq * w).sum(1) / (w * w).sum(1)
        if unbiased:
            assert torch.allclose(beta, torch.ones_like(beta), atol=2e-3), beta
        else:
            assert beta.mean() < 0.95  # 2-bit rounding shrinks rows


def test_level_counts_one_bit_and_ternary():
    torch.manual_seed(12)
    w = torch.randn(256, 1024, device=DEV)
    nmse = lambda q: ((q - w).pow(2).sum() / w.pow(2).sum()).item()  # noqa: E731
    assert nmse(quant.rtn(w, 1, 64)) < 0.40  # scalar 1-bit optimum on a Gaussian is 0.363
    codes, step = quant.rtn_codes(w, 2, 64, levels=3)
    assert set(codes.unique().tolist()) == {0, 1, 2}
    assert nmse(quant.rtn_dequant(codes, step, 2, 64, levels=3)) < 0.22
    lin = quant.RTNLinear(w, 2, 64, levels=3)
    assert lin.packed.shape == (256, 256) and torch.allclose(lin.weight_dequant(), quant.rtn(w, 2, 64, levels=3),
                                                             rtol=1e-3, atol=1e-4)


def test_rtn_unchanged_for_power_of_two_levels():
    # the level-count generalization must keep the 2**bits mid-rise path bit-identical to the original formula
    torch.manual_seed(13)
    w = torch.randn(64, 512, device=DEV)
    for bits in (2, 3, 4):
        g, half = w.reshape(64, -1, 64), 2 ** (bits - 1)
        amax, best, err = g.abs().amax(-1, keepdim=True), None, None
        for r in torch.linspace(0.5, 1.0, 8, device=DEV, dtype=torch.float32):
            step = (amax * r / (half - 0.5)).half().float()
            e = ((torch.floor(g / step).clamp(-half, half - 1) + 0.5) * step - g).pow(2).sum(-1, keepdim=True)
            best, err = (step, e) if best is None else (torch.where(e < err, step, best), torch.minimum(e, err))
        want = ((torch.floor(g / best).clamp(-half, half - 1) + 0.5) * best).reshape(w.shape)
        assert torch.equal(quant.rtn(w, bits, 64), want)


def test_gptq_ternary_beats_rtn_ternary():
    torch.manual_seed(14)
    x = torch.randn(4096, 32, device=DEV) @ torch.randn(32, 256, device=DEV) + 0.1 * torch.randn(4096, 256, device=DEV)
    w = torch.randn(64, 256, device=DEV)
    out_err = lambda wq: ((wq - w) @ x.t()).pow(2).sum() / (w @ x.t()).pow(2).sum()  # noqa: E731
    codes, step = quant.gptq_codes(w, x.t() @ x, 2, 64, levels=3)
    assert int(codes.max()) <= 2
    assert out_err(quant.rtn_dequant(codes, step, 2, 64, levels=3)) < 0.5 * out_err(quant.rtn(w, 2, 64, levels=3))


def test_refit_steps_recovers_steps_and_lowers_gptq_output_error():
    torch.manual_seed(15)
    q = torch.randint(-2, 2, (16, 256), device=DEV).float() + 0.5
    s_true = torch.rand(16, 2, device=DEV) + 0.5
    x = torch.randn(1000, 256, device=DEV)
    ones = torch.ones(1000, device=DEV)
    s = quant.refit_steps(q, q * s_true.repeat_interleave(128, -1), [(x, ones, 1.0)], 128)
    assert torch.allclose(s, s_true, atol=1e-3)
    # after GPTQ, refitting the steps can only lower the output error both of them minimize
    x = torch.randn(1000, 32, device=DEV) @ torch.randn(32, 256, device=DEV) + 0.1 * torch.randn(1000, 256, device=DEV)
    w = torch.randn(64, 256, device=DEV)
    codes, step = quant.gptq_codes(w, x.t() @ x, 1, 64)
    q = quant.rtn_dequant(codes, torch.ones_like(step), 1, 64)
    s = quant.refit_steps(q, w, [(x, ones, 1.0)], 64)
    err = lambda st: ((q * st.repeat_interleave(64, -1) - w) @ x.t()).pow(2).sum()  # noqa: E731
    assert err(s) < 0.95 * err(step), (err(s), err(step))


def test_two_bit_codes_pack_four_per_byte():
    torch.manual_seed(11)
    codes = torch.randint(0, 4, (6, 256), device=DEV, dtype=torch.uint8)
    assert torch.equal(quant.unpack_crumbs(quant.pack_crumbs(codes)), codes)
    w = torch.randn(32, 256, device=DEV)
    lin = quant.RTNLinear(w, 2, 64)
    assert lin.packed.shape == (32, 64)
    assert torch.allclose(lin.weight_dequant(), quant.rtn(w, 2, 64), rtol=1e-3, atol=1e-4)  # fp16 steps


def test_rtn_linear_from_codes_matches_constructor():
    torch.manual_seed(10)
    w = torch.randn(32, 256, device=DEV)
    rot = quant.BlockHadamard(256, seed=4)
    a = quant.RTNLinear(w, 3, 64, rot)
    b = quant.RTNLinear.from_codes(*quant.rtn_codes(rot.rotate(w), 3, 64), 3, 64, rot)
    x = torch.randn(5, 256, device=DEV)
    assert torch.equal(a.packed, b.packed) and torch.equal(a.step, b.step) and torch.equal(a(x), b(x))


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


def test_gptq_vq_identity_hessian_is_nearest_codeword():
    torch.manual_seed(9)
    n, k, d = 16, 256, 8
    w, cb = torch.randn(n, k, device=DEV), torch.randn(64, d, device=DEV)
    for group in (128, k):  # a scale per 128 columns, and one per row
        codes, scales = quant.gptq_vq(w, torch.eye(k, device=DEV), cb, group)
        s = w.reshape(n, -1, group).pow(2).mean(-1).sqrt().half().float()
        assert codes.shape == (n, k // d) and torch.equal(scales, s)
        v = w.reshape(n, k // group, group // d, d) / s[..., None, None]
        want = quant.vq_assign(v.reshape(-1, d), cb).reshape(n, k // d)
        assert torch.equal(codes, want)
        wq = quant.vq_codes_dequant(codes, scales, cb)
        assert torch.allclose(wq, (cb[want].reshape(v.shape) * s[..., None, None]).reshape(n, k))


def test_gptq_vq_beats_plain_vq_on_correlated_inputs():
    torch.manual_seed(10)
    k, n, m = 256, 64, 4096
    x = torch.randn(m, 32, device=DEV) @ torch.randn(32, k, device=DEV) + 0.1 * torch.randn(m, k, device=DEV)
    w = torch.randn(3, n, k, device=DEV)
    h = (x.t() @ x).expand(3, k, k)
    cb = quant.kmeans(torch.randn(1 << 16, 4, device=DEV), 256)  # 2 bpw
    out_err = lambda wq: ((wq - w) @ x.t()).pow(2).sum() / (w @ x.t()).pow(2).sum()  # noqa: E731
    codes, scales = quant.gptq_vq(w, h, cb, 128)
    plain = quant.vq_codes_dequant(*quant.gptq_vq(w, torch.eye(k, device=DEV).expand(3, k, k), cb, 128), cb)
    e_gptq, e_plain = out_err(quant.vq_codes_dequant(codes, scales, cb)), out_err(plain)
    assert e_gptq < 0.5 * e_plain, (e_gptq, e_plain)
    for b in range(3):  # batched ~ one at a time: a near-tie may flip a code and the feedback then carries it on
        c1, s1 = quant.gptq_vq(w[b], h[b], cb, 128)
        assert (codes[b] == c1).float().mean() > 0.9 and torch.allclose(scales[b], s1, rtol=2e-3)  # fp16 ulp
        e1 = ((quant.vq_codes_dequant(c1, s1, cb) - w[b]) @ x.t()).pow(2).sum() / (w[b] @ x.t()).pow(2).sum()
        assert abs(e1 / e_gptq - 1) < 0.1
    # the refit keeps the codes and can only lower the error under h
    c0, s0 = quant.gptq_vq(w, h, cb, k)
    c1, s1 = quant.gptq_vq(w, h, cb, k, refit=True)
    assert torch.equal(c0, c1) and s1.shape == (3, n, 1)
    assert out_err(quant.vq_codes_dequant(c1, s1, cb)) < out_err(quant.vq_codes_dequant(c0, s0, cb))
