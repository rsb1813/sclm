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

// The sum (or the maximum) of v over the block's threads (a multiple of 32, at most 1024), returned to every thread.
__device__ float block_reduce(float v, bool take_max) {
    __shared__ float part[33];
    for (int o = 16; o; o >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, v, o);
        v = take_max ? fmaxf(v, other) : v + other;
    }
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        v = part[0];
        for (unsigned i = 1; i < blockDim.x >> 5; ++i) v = take_max ? fmaxf(v, part[i]) : v + part[i];
        part[32] = v;
    }
    __syncthreads();
    return part[32];
}

// kernel.act_quant(x, 32, "ue8m0", inplace): a block of 32 is scaled by a power of two so that its largest magnitude
// is at most 448, and rounded to E4M3. One CUDA block of 32 threads per quantization block. Out is fp32, or bf16
// (which holds every such value exactly); xq may be x.
template <class Out>
__global__ void act_quant_k(const uint16_t* x, Out* xq) {
    const int i = blockIdx.x * 32 + threadIdx.x;
    const float v = from_bf16(x[i]);
    float mx = fabsf(v);
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    const uint32_t bits = __float_as_uint(fmaxf(mx, 1e-4f) * float(1.0 / 448.0));
    const int se = int(bits >> 23 & 255) - 127 + ((bits & 0x7fffff) != 0);  // the scale is 2^ceil(log2(amax / 448))
    const float t = scalbnf(v, -se);
    const int q = max(ilogbf(t), -6) - 3;  // E4M3 keeps 4 significant bits down to 2^-6 and steps of 2^-9 below
    const float r = scalbnf(rintf(scalbnf(t, -q)), q + se);
    if (sizeof(Out) == 2)
        xq[i] = Out(to_bf16(r));
    else
        xq[i] = Out(r);
}

// upstream RMSNorm: out = bf16(w * x * rsqrt(mean(x^2) + eps)) over n values, in one block; out may be x.
__global__ void rmsnorm_k(const uint16_t* x, const uint16_t* w, uint16_t* out, int n, float eps) {
    float sq = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sq += from_bf16(x[i]) * from_bf16(x[i]);
    const float r = rsqrtf(block_reduce(sq, false) / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = to_bf16(from_bf16(w[i]) * (from_bf16(x[i]) * r));
}

// upstream apply_rotary_emb on the last rd values of each of `heads` vectors of head_dim (bf16, in place): adjacent
// pairs are complex numbers multiplied by freqs[pair] (re, im), conjugated when inverse. A thread per pair.
__global__ void rope_k(uint16_t* x, const float* freqs, int heads, int head_dim, int rd, bool inverse) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, pairs = rd / 2;
    if (i >= heads * pairs) return;
    uint16_t* v = x + (i / pairs) * head_dim + head_dim - rd + 2 * (i % pairs);
    const float a = from_bf16(v[0]), b = from_bf16(v[1]), c = freqs[2 * (i % pairs)];
    const float d = inverse ? -freqs[2 * (i % pairs) + 1] : freqs[2 * (i % pairs) + 1];
    v[0] = to_bf16(a * c - b * d), v[1] = to_bf16(a * d + b * c);
}

// kernel.sparse_attn for one query token, block h for head h: kv is both key and value, position idx[t] being
// window[idx] below `win` and compressed[idx - win] from there on, -1 an empty slot. A learned sink logit joins the
// softmax's denominator; the probabilities go through bf16 before weighting the values, as upstream.
__global__ void sparse_attn_k(const uint16_t* q, const uint16_t* window, const uint16_t* compressed, const int* idx,
                              const float* sink, uint16_t* out, int count, int d, int win, float scale) {
    __shared__ float p[1024];
    const uint16_t* qh = q + size_t(blockIdx.x) * d;
    float mx = -INFINITY;
    for (int t = threadIdx.x; t < count; t += blockDim.x) {
        const int at = idx[t];
        float sc = -INFINITY;
        if (at >= 0) {
            const uint16_t* kv = at < win ? window + size_t(at) * d : compressed + size_t(at - win) * d;
            sc = 0;
            for (int i = 0; i < d; ++i) sc += from_bf16(qh[i]) * from_bf16(kv[i]);
            sc *= scale;
        }
        p[t] = sc;
        mx = fmaxf(mx, sc);
    }
    mx = fmaxf(block_reduce(mx, true), -1e30f);  // upstream's running maximum starts at -1e30: an empty row gives zeros
    float sum = 0;
    for (int t = threadIdx.x; t < count; t += blockDim.x) {
        const float e = expf(p[t] - mx);
        sum += e;
        p[t] = from_bf16(to_bf16(e));
    }
    const float denom = block_reduce(sum, false) + expf(sink[blockIdx.x] - mx);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float o = 0;
        for (int t = 0; t < count; ++t) {
            const int at = idx[t];
            if (at < 0) continue;
            o += p[t] * from_bf16((at < win ? window + size_t(at) * d : compressed + size_t(at - win) * d)[i]);
        }
        out[size_t(blockIdx.x) * d + i] = to_bf16(o / denom);
    }
}

// y[row] = sum over k of w[row][k] * scale[row / 32][k / 32] * xq[k], w in E4M3 and scale in E8M0. One block per row.
// Rows come in groups of rows_per_group, group g reading xq[g * k ..] (attention's block-diagonal wo_a; one group
// otherwise).
// The sum is rounded to bf16 as the reference's GEMM output is; add: it is added to the fp32 y, else stored as bf16.
__global__ void fp8_gemv_k(const uint8_t* w, const uint8_t* scale, const float* xq, void* y, int k, int rows_per_group, bool add) {
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
    if (threadIdx.x) return;
    const uint16_t r = to_bf16(acc);
    if (add)
        ((float*)y)[row] += from_bf16(r);
    else
        ((uint16_t*)y)[row] = r;
}

// The same for packed FP4: w[row] is k / 2 bytes (the even column in the low nibble), scale[row][k / 32] in E8M0.
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
    float sq = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float y = 0;
        for (int c = 0; c < hc; ++c) y += pre[c] * from_bf16(h[c * dim + d]);
        y = from_bf16(to_bf16(y));
        sq += y * y;
    }
    const float r = rsqrtf(block_reduce(sq, false) / float(dim) + eps);
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
    uint16_t *gate = nullptr, *up = nullptr, *h = nullptr;  // an expert's branches and hidden vector; h also other intermediates
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
    act_quant_k<float><<<k / 32, 32>>>(x, xq);
    return cudaGetLastError();
}

// y[n] (bf16) = the reference linear() of an fp8 weight [n, k] with block scales on x (bf16): x holds `groups`
// vectors of k, and rows i * n / groups .. of the weight multiply vector i.
API int sclm_gpu_fp8_linear(const uint8_t* w, const uint8_t* scale, int n, int k, int groups, const uint16_t* x, uint16_t* y) {
    if (groups * k > SCRATCH) return -1;
    ensure_scratch();
    act_quant_k<float><<<groups * k / 32, 32>>>(x, scratch.xq);
    fp8_gemv_k<<<n, threads_for(k, 4)>>>(w, scale, scratch.xq, y, k, n / groups, false);
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
    act_quant_k<float><<<inter / 32, 32>>>(scratch.h, scratch.xq);
    fp4_gemv_k<<<dim, threads_for(inter, 8)>>>(w2, s2, scratch.xq, y, inter, true);
    return cudaGetLastError();
}

// y[dim] (fp32) += the upstream Expert with fp8 weights and block scales (the shared expert) on the layer input
// quantized as xq[dim]. w1 and w3 are [inter, dim], w2 is [dim, inter].
API int sclm_gpu_fp8_expert(const uint8_t* w1, const uint8_t* s1, const uint8_t* w2, const uint8_t* s2, const uint8_t* w3,
                            const uint8_t* s3, int dim, int inter, float limit, const float* xq, float* y) {
    if (dim > SCRATCH || inter > SCRATCH) return -1;
    ensure_scratch();
    fp8_gemv_k<<<inter, threads_for(dim, 4)>>>(w1, s1, xq, scratch.gate, dim, inter, false);
    fp8_gemv_k<<<inter, threads_for(dim, 4)>>>(w3, s3, xq, scratch.up, dim, inter, false);
    swiglu_k<<<(inter + 255) / 256, 256>>>(scratch.gate, scratch.up, scratch.h, inter, limit, 1.0f);
    act_quant_k<float><<<inter / 32, 32>>>(scratch.h, scratch.xq);
    fp8_gemv_k<<<dim, threads_for(inter, 4)>>>(w2, s2, scratch.xq, y, inter, dim, true);
    return cudaGetLastError();
}

// The query side of upstream Attention.forward for one token: qr[rank] = RMSNorm(wq_a x), then
// q[heads * head_dim] = wq_b qr with the rotary embedding on the last rd values of each head. All bf16; the weights
// are fp8 with block scales, freqs the position's rd / 2 complex rotations.
API int sclm_gpu_attn_q(const uint8_t* wa, const uint8_t* sa, const uint16_t* norm_w, const uint8_t* wb, const uint8_t* sb,
                        const uint16_t* x, const float* freqs, uint16_t* qr, uint16_t* q, int dim, int rank, int heads,
                        int head_dim, int rd, float eps) {
    ensure_scratch();
    act_quant_k<float><<<dim / 32, 32>>>(x, scratch.xq);
    fp8_gemv_k<<<rank, threads_for(dim, 4)>>>(wa, sa, scratch.xq, qr, dim, rank, false);
    rmsnorm_k<<<1, 1024>>>(qr, norm_w, qr, rank, eps);
    act_quant_k<float><<<rank / 32, 32>>>(qr, scratch.xq);
    fp8_gemv_k<<<heads * head_dim, threads_for(rank, 4)>>>(wb, sb, scratch.xq, q, rank, heads * head_dim, false);
    rope_k<<<(heads * rd / 2 + 255) / 256, 256>>>(q, freqs, heads, head_dim, rd, false);
    return cudaGetLastError();
}

// The key/value of one token as upstream Attention._window_kv makes it: kv[head_dim] = fp8-quantized
// rotary(RMSNorm(wkv x)), written where the caller points (its slot of the window ring).
API int sclm_gpu_attn_kv(const uint8_t* w, const uint8_t* s, const uint16_t* norm_w, const uint16_t* x, const float* freqs,
                         uint16_t* kv, int dim, int head_dim, int rd, float eps) {
    ensure_scratch();
    act_quant_k<float><<<dim / 32, 32>>>(x, scratch.xq);
    fp8_gemv_k<<<head_dim, threads_for(dim, 4)>>>(w, s, scratch.xq, kv, dim, head_dim, false);
    rmsnorm_k<<<1, 1024>>>(kv, norm_w, kv, head_dim, eps);
    rope_k<<<(rd / 2 + 255) / 256, 256>>>(kv, freqs, 1, head_dim, rd, false);
    act_quant_k<uint16_t><<<head_dim / 32, 32>>>(kv, kv);
    return cudaGetLastError();
}

// out[heads * d] (bf16) = kernel.sparse_attn of one token's q over the `count` positions idx names (see sparse_attn_k).
API int sclm_gpu_sparse_attn(const uint16_t* q, const uint16_t* window, const uint16_t* compressed, const int* idx,
                             const float* sink, uint16_t* out, int heads, int d, int count, int win, float scale) {
    if (count > 1024) return -1;
    sparse_attn_k<<<heads, 256>>>(q, window, compressed, idx, sink, out, count, d, win, scale);
    return cudaGetLastError();
}

// The output side of upstream Attention.forward for one token: the rotary embedding is taken back off o (in place),
// then out[dim] = wo_b (wo_a o) with wo_a block-diagonal over `groups` and both fp8 with block scales. n_a is wo_a's
// row count.
API int sclm_gpu_attn_out(uint16_t* o, const float* freqs, const uint8_t* wa, const uint8_t* sa, const uint8_t* wb,
                          const uint8_t* sb, uint16_t* out, int heads, int head_dim, int rd, int groups, int n_a, int dim) {
    const int width = heads * head_dim;
    if (width > SCRATCH || n_a > SCRATCH) return -1;
    ensure_scratch();
    rope_k<<<(heads * rd / 2 + 255) / 256, 256>>>(o, freqs, heads, head_dim, rd, true);
    act_quant_k<float><<<width / 32, 32>>>(o, scratch.xq);
    fp8_gemv_k<<<n_a, threads_for(width / groups, 4)>>>(wa, sa, scratch.xq, scratch.h, width / groups, n_a / groups, false);
    act_quant_k<float><<<n_a / 32, 32>>>(scratch.h, scratch.xq);
    fp8_gemv_k<<<dim, threads_for(n_a, 4)>>>(wb, sb, scratch.xq, out, n_a, dim, false);
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
