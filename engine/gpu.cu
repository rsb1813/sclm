// GPU kernels of the engine core for one decode token, launched on the host's device pointers (default stream, the
// host's CUDA context): the reference's FP8 and FP4 GEMMs computed straight from the stored bytes, with the activation
// quantization and the SwiGLU between them. The reference widens every weight to bf16 on each call instead.
#include <cuda_runtime.h>
#include <cstdint>

#define API extern "C" __declspec(dllexport)

namespace {

const int SCRATCH = 1 << 16;  // widest vector a call handles

__device__ __forceinline__ float e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15, m = b & 7;
    const float v = e ? __uint_as_float((e + 120) << 23 | m << 20) : float(m) * 0.001953125f;  // below 2^-6: m * 2^-9
    return b & 128 ? -v : v;
}

__device__ __forceinline__ float e8m0(uint32_t b) { return __uint_as_float(b << 23); }  // 2^(b - 127)

// Twice the E2M1 value of a 4-bit code (magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign).
__device__ __forceinline__ float e2m1x2(uint32_t code) {
    const int m = (0xC8643210u >> ((code & 7) * 4)) & 15;
    return float(code & 8 ? -m : m);
}

__device__ __forceinline__ float from_bf16(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }

__device__ __forceinline__ uint16_t to_bf16(float f) {  // nearest, ties to even
    const uint32_t u = __float_as_uint(f);
    return uint16_t((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}

// The sum of v over the block's threads (a multiple of 32, at most 256 of them); thread 0 gets it.
__device__ float block_sum(float v) {
    __shared__ float part[8];
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        v = 0;
        for (unsigned i = 0; i < blockDim.x >> 5; ++i) v += part[i];
    }
    return v;
}

// kernel.act_quant(x, 32, "ue8m0", inplace): a block of 32 is scaled by a power of two so that its largest magnitude
// is at most 448, and rounded to E4M3. One CUDA block of 32 threads per quantization block.
template <class In>
__global__ void act_quant_k(const In* x, float* xq) {
    const int i = blockIdx.x * 32 + threadIdx.x;
    const float v = sizeof(In) == 2 ? from_bf16(uint16_t(x[i])) : float(x[i]);
    float mx = fabsf(v);
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    const uint32_t bits = __float_as_uint(fmaxf(mx, 1e-4f) * float(1.0 / 448.0));
    const int se = int(bits >> 23 & 255) - 127 + ((bits & 0x7fffff) != 0);  // the scale is 2^ceil(log2(amax / 448))
    const float t = scalbnf(v, -se);
    const int q = max(ilogbf(t), -6) - 3;  // E4M3 keeps 4 significant bits down to 2^-6 and steps of 2^-9 below
    xq[i] = scalbnf(rintf(scalbnf(t, -q)), q + se);
}

// y[row] = sum over k of w[row][k] * scale[row / 32][k / 32] * xq[k], w in E4M3 and scale in E8M0. One block per row.
// Rows come in groups of rows_per_group, group g reading xq[g * k ..] (attention's block-diagonal wo_a; one group
// otherwise).
__global__ void fp8_gemv_k(const uint8_t* w, const uint8_t* scale, const float* xq, uint16_t* y, int k, int rows_per_group) {
    const int row = blockIdx.x;
    const uint8_t* wr = w + size_t(row) * k;
    const uint8_t* sr = scale + size_t(row >> 5) * (k >> 5);
    const float* x = xq + size_t(row / rows_per_group) * k;
    float acc = 0;
    for (int c = threadIdx.x * 4; c < k; c += blockDim.x * 4) {
        const uint32_t b = *(const uint32_t*)(wr + c);
        acc += e8m0(sr[c >> 5]) * (e4m3(b & 255) * x[c] + e4m3(b >> 8 & 255) * x[c + 1] + e4m3(b >> 16 & 255) * x[c + 2] +
                                   e4m3(b >> 24) * x[c + 3]);
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) y[row] = to_bf16(acc);
}

// The same for packed FP4: w[row] is k / 2 bytes (the even column in the low nibble), scale[row][k / 32] in E8M0.
// The sum is rounded to bf16 as the reference's GEMM output is; add: it is added to the fp32 y, else stored as bf16.
__global__ void fp4_gemv_k(const uint8_t* w, const uint8_t* scale, const float* xq, void* y, int k, bool add) {
    const int row = blockIdx.x;
    const uint8_t* wr = w + size_t(row) * (k >> 1);
    const uint8_t* sr = scale + size_t(row) * (k >> 5);
    float acc = 0;
    for (int c = threadIdx.x * 8; c < k; c += blockDim.x * 8) {
        const uint32_t b = *(const uint32_t*)(wr + (c >> 1));
        float s = 0;
        for (int i = 0; i < 8; ++i) s += e2m1x2(b >> (4 * i)) * xq[c + i];
        acc += 0.5f * e8m0(sr[c >> 5]) * s;
    }
    acc = block_sum(acc);
    if (threadIdx.x) return;
    const uint16_t r = to_bf16(acc);
    if (add)
        ((float*)y)[row] += from_bf16(r);
    else
        ((uint16_t*)y)[row] = r;
}

// The reference Expert's middle: h = weight * silu(min(gate, limit)) * clamp(up, -limit, limit), rounded to bf16.
__global__ void swiglu_k(const uint16_t* gate, const uint16_t* up, uint16_t* h, int n, float limit, float weight) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = from_bf16(gate[i]), u = from_bf16(up[i]);
    if (limit > 0) g = fminf(g, limit), u = fminf(fmaxf(u, -limit), limit);
    h[i] = to_bf16(weight * (g / (1.0f + expf(-g)) * u));
}

// Hyper-connection mixes, first half (Block.hc_mixes): block i < rows computes <fn[i], h> over the n values of the
// flattened stream h (bf16), block `rows` its sum of squares.
__global__ void hc_dots_k(const uint16_t* h, const float* fn, float* dots, int n, int rows) {
    const int row = blockIdx.x;
    float acc = 0;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float v = from_bf16(h[c]);
        acc += (row < rows ? fn[size_t(row) * n + c] : v) * v;
    }
    acc = block_sum(acc);
    if (threadIdx.x == 0) dots[row] = acc;
}

// Second half (kernel.hc_split_sinkhorn on mixes = dots * rsqrt(mean square + norm_eps)), one thread:
// out = pre[hc], post[hc], comb[hc][hc] with comb made doubly stochastic by `iters` Sinkhorn rounds.
__global__ void hc_split_k(const float* dots, const float* scale, const float* base, float* out, int n, int hc,
                           float norm_eps, float eps, int iters) {
    const float r = rsqrtf(dots[(2 + hc) * hc] / float(n) + norm_eps);
    float comb[8][8];
    for (int i = 0; i < hc; ++i) {
        out[i] = 1.0f / (1.0f + expf(-(dots[i] * r * scale[0] + base[i]))) + eps;
        out[hc + i] = 2.0f / (1.0f + expf(-(dots[hc + i] * r * scale[1] + base[hc + i])));
    }
    for (int j = 0; j < hc; ++j) {  // softmax over each row, then columns, then alternate
        float mx = -INFINITY, sum = 0;
        for (int c = 0; c < hc; ++c) {
            comb[j][c] = dots[2 * hc + j * hc + c] * r * scale[2] + base[2 * hc + j * hc + c];
            mx = fmaxf(mx, comb[j][c]);
        }
        for (int c = 0; c < hc; ++c) sum += comb[j][c] = expf(comb[j][c] - mx);
        for (int c = 0; c < hc; ++c) comb[j][c] = comb[j][c] / sum + eps;
    }
    for (int it = 0; it < 2 * iters - 1; ++it)
        for (int a = 0; a < hc; ++a) {  // even rounds: columns (sum over rows); odd rounds: rows
            float sum = 0;
            for (int b = 0; b < hc; ++b) sum += it & 1 ? comb[a][b] : comb[b][a];
            for (int b = 0; b < hc; ++b) (it & 1 ? comb[a][b] : comb[b][a]) /= sum + eps;
        }
    for (int j = 0; j < hc; ++j)
        for (int c = 0; c < hc; ++c) out[2 * hc + j * hc + c] = comb[j][c];
}

// Block.hc_pre then RMSNorm: y = bf16(sum over the hc copies of pre[c] * h[c]), out = bf16(w * y * rsqrt(mean(y^2) + eps)).
// One block of 1024 threads over the dim values.
__global__ void hc_pre_norm_k(const uint16_t* h, const float* pre, const uint16_t* w, uint16_t* out, int dim, int hc, float eps) {
    __shared__ float part[32];
    float sq = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float y = 0;
        for (int c = 0; c < hc; ++c) y += pre[c] * from_bf16(h[c * dim + d]);
        y = from_bf16(to_bf16(y));
        sq += y * y;
    }
    for (int o = 16; o; o >>= 1) sq += __shfl_down_sync(0xffffffff, sq, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    float total = 0;
    for (unsigned i = 0; i < blockDim.x >> 5; ++i) total += part[i];
    const float r = rsqrtf(total / float(dim) + eps);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float y = 0;
        for (int c = 0; c < hc; ++c) y += pre[c] * from_bf16(h[c * dim + d]);
        out[d] = to_bf16(from_bf16(w[d]) * (from_bf16(to_bf16(y)) * r));
    }
}

// Block.hc_post: out[c] = bf16(post[c] * x + sum over j of comb[j][c] * residual[j]).
__global__ void hc_post_k(const uint16_t* x, const uint16_t* residual, const float* post, const float* comb, uint16_t* out, int dim, int hc) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const float v = from_bf16(x[d]);
    for (int c = 0; c < hc; ++c) {
        float mix = 0;
        for (int j = 0; j < hc; ++j) mix += comb[j * hc + c] * from_bf16(residual[j * dim + d]);
        out[c * dim + d] = to_bf16(post[c] * v + mix);
    }
}

struct Scratch {
    float* xq = nullptr;  // a quantized activation
    float* dots = nullptr;
    uint16_t *gate = nullptr, *up = nullptr, *h = nullptr;
} scratch;

void ensure_scratch() {
    if (scratch.xq) return;
    cudaMalloc(&scratch.xq, SCRATCH * sizeof(float));
    cudaMalloc(&scratch.dots, 256 * sizeof(float));
    cudaMalloc(&scratch.gate, SCRATCH * sizeof(uint16_t));
    cudaMalloc(&scratch.up, SCRATCH * sizeof(uint16_t));
    cudaMalloc(&scratch.h, SCRATCH * sizeof(uint16_t));
}

int threads_for(int k, int per_thread) { return k / per_thread >= 512 ? 64 : 32; }

}  // namespace

// xq[k] (fp32) = the reference's fp8 fake-quant of x[k] (bf16). k is a multiple of 32.
API int sclm_gpu_act_quant(const uint16_t* x, float* xq, int k) {
    act_quant_k<uint16_t><<<k / 32, 32>>>(x, xq);
    return cudaGetLastError();
}

// y[n] (bf16) = the reference linear() of an fp8 weight [n, k] with block scales on x (bf16): x holds `groups`
// vectors of k, and rows i * n / groups .. of the weight multiply vector i.
API int sclm_gpu_fp8_linear(const uint8_t* w, const uint8_t* scale, int n, int k, int groups, const uint16_t* x, uint16_t* y) {
    if (groups * k > SCRATCH) return -1;
    ensure_scratch();
    act_quant_k<uint16_t><<<groups * k / 32, 32>>>(x, scratch.xq);
    fp8_gemv_k<<<n, threads_for(k, 4)>>>(w, scale, scratch.xq, y, k, n / groups);
    return cudaGetLastError();
}

// y[dim] (fp32) += weight-scaled output of the routed expert whose checkpoint bytes are at `slot` (scales of w1, w2,
// w3, then their packed FP4 weights), on the layer input already quantized as xq[dim] (sclm_gpu_act_quant).
API int sclm_gpu_fp4_expert(const uint8_t* slot, int dim, int inter, float limit, const float* xq, float weight, float* y) {
    if (dim > SCRATCH || inter > SCRATCH) return -1;
    ensure_scratch();
    const uint8_t *s1 = slot, *s2 = s1 + size_t(inter) * (dim / 32), *s3 = s2 + size_t(dim) * (inter / 32);
    const uint8_t *w1 = s3 + size_t(inter) * (dim / 32), *w2 = w1 + size_t(inter) * (dim / 2), *w3 = w2 + size_t(dim) * (inter / 2);
    fp4_gemv_k<<<inter, threads_for(dim, 8)>>>(w1, s1, xq, scratch.gate, dim, false);
    fp4_gemv_k<<<inter, threads_for(dim, 8)>>>(w3, s3, xq, scratch.up, dim, false);
    swiglu_k<<<(inter + 255) / 256, 256>>>(scratch.gate, scratch.up, scratch.h, inter, limit, weight);
    act_quant_k<uint16_t><<<inter / 32, 32>>>(scratch.h, scratch.xq);
    fp4_gemv_k<<<dim, threads_for(inter, 8)>>>(w2, s2, scratch.xq, y, inter, true);
    return cudaGetLastError();
}

// mixes[(2 + hc) * hc] (fp32) = Block.hc_mixes of the stream h[hc * dim] (bf16): pre[hc], post[hc], comb[hc][hc].
// fn is [(2 + hc) * hc][hc * dim], scale [3], base [(2 + hc) * hc], all fp32.
API int sclm_gpu_hc_mixes(const uint16_t* h, const float* fn, const float* scale, const float* base, float* mixes, int dim,
                          int hc, float norm_eps, float eps, int iters) {
    if (hc > 8) return -1;
    ensure_scratch();
    const int rows = (2 + hc) * hc;
    hc_dots_k<<<rows + 1, 256>>>(h, fn, scratch.dots, hc * dim, rows);
    hc_split_k<<<1, 1>>>(scratch.dots, scale, base, mixes, hc * dim, hc, norm_eps, eps, iters);
    return cudaGetLastError();
}

// out[dim] (bf16) = RMSNorm with weight w (bf16) of Block.hc_pre(h, pre): the sublayer's input.
API int sclm_gpu_hc_pre_norm(const uint16_t* h, const float* pre, const uint16_t* w, uint16_t* out, int dim, int hc, float eps) {
    hc_pre_norm_k<<<1, 1024>>>(h, pre, w, out, dim, hc, eps);
    return cudaGetLastError();
}

// out[hc * dim] (bf16) = Block.hc_post of the sublayer output x[dim] and the stream it started from.
API int sclm_gpu_hc_post(const uint16_t* x, const uint16_t* residual, const float* post, const float* comb, uint16_t* out, int dim, int hc) {
    hc_post_k<<<(dim + 255) / 256, 256>>>(x, residual, post, comb, out, dim, hc);
    return cudaGetLastError();
}
