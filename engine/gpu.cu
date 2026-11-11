// GPU kernels of the engine core for one decode token, launched on the host's device pointers (default stream, the
// host's CUDA context): the reference's FP8 and FP4 GEMMs computed straight from the stored bytes, with the activation
// quantization and the SwiGLU between them. The reference widens every weight to bf16 on each call instead.
#include <cuda_runtime.h>
#include <cstdint>

#define API extern "C" __declspec(dllexport)

namespace {

const int SCRATCH = 1 << 16;  // widest vector a call handles
const int NORM_MAX = 8192;  // longest vector norm_quant handles (it keeps it in shared memory)

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

__device__ __forceinline__ float widen(uint16_t v) { return from_bf16(v); }  // a stored bf16 or fp32 value
__device__ __forceinline__ float widen(float v) { return v; }

// kernel.round_fp4: a value in [-6, 6] to the nearest E2M1 magnitude (0, .5, 1, 1.5, 2, 3, 4, 6), a tie to the
// even code.
__device__ __forceinline__ float round_e2m1(float v) {
    const float a = fabsf(v);
    int hi = 1;
    while (hi < 7 && a > 0.5f * float((0xC8643210u >> (hi * 4)) & 15)) ++hi;
    const float lo = 0.5f * float((0xC8643210u >> (hi * 4 - 4)) & 15), up = 0.5f * float((0xC8643210u >> (hi * 4)) & 15);
    const bool pick_up = up - a < a - lo || (up - a == a - lo && hi % 2 == 0);
    return copysignf(pick_up ? up : lo, v);
}

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

// Called by every thread of a block when the block's results are in global memory: true in the block that finishes
// last of the `blocks` blocks that share `count`, which may then read what all of them wrote (through a volatile
// pointer) and do a step that needs all their results, saving the launch of a kernel for it. `count` counts the
// blocks that are done and is left at zero.
__device__ bool last_block(int* count, int blocks) {
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(count, 1) == blocks - 1;
        if (last) *count = 0;
    }
    __syncthreads();
    return last;
}

int* done_counts() {  // last_block's counters: SCRATCH / 32 of them, zero between kernels
    static int* counts = nullptr;
    if (!counts) {
        cudaMalloc(&counts, SCRATCH / 32 * sizeof(int));
        cudaMemset(counts, 0, SCRATCH / 32 * sizeof(int));
    }
    return counts;
}

// One value of kernel.act_quant(x, 32, "ue8m0", inplace), mx being the largest magnitude of its block of 32.
__device__ __forceinline__ float fp8_round(float v, float mx) {
    const uint32_t bits = __float_as_uint(fmaxf(mx, 1e-4f) * float(1.0 / 448.0));
    const int se = int(bits >> 23 & 255) - 127 + ((bits & 0x7fffff) != 0);  // the scale is 2^ceil(log2(amax / 448))
    const float t = scalbnf(v, -se);
    const int q = max(ilogbf(t), -6) - 3;  // E4M3 keeps 4 significant bits down to 2^-6 and steps of 2^-9 below
    return scalbnf(rintf(scalbnf(t, -q)), q + se);
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
    const float r = fp8_round(v, mx);
    if (sizeof(Out) == 2)
        xq[i] = Out(to_bf16(r));
    else
        xq[i] = Out(r);
}

// What attention does with a projection's output y[n] (bf16), by one block, a warp taking 32 values at a time:
// upstream RMSNorm with weight w, normed = bf16(w * y * rsqrt(mean(y^2) + eps)) (stored if asked for); with freqs,
// upstream apply_rotary_emb on its last rd values (adjacent pairs are complex numbers multiplied by freqs[pair]
// (re, im), rounded to bf16); then kernel.act_quant -> xq (fp32) or q (bf16). n is a multiple of 32, n - rd even.
// The squares are added up as one block of 1024 threads would: a thread every 1024th value, its warps by shuffles,
// then the warps in order.
__device__ void norm_quant(const volatile uint16_t* y, const uint16_t* w, const float* freqs, uint16_t* normed, float* xq,
                           uint16_t* q, int n, int rd, float eps) {
    extern __shared__ float vals[];  // y
    __shared__ float part[33];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    for (int i = threadIdx.x; i < n; i += blockDim.x) vals[i] = from_bf16(y[i]);
    __syncthreads();
    for (int o = warp; o < 32; o += warps) {  // warp o of the 1024 threads
        float sq = 0;
        for (int i = o * 32 + lane; i < n; i += 1024) sq += vals[i] * vals[i];
        for (int s = 16; s; s >>= 1) sq += __shfl_down_sync(0xffffffff, sq, s);
        if (lane == 0) part[o] = sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = part[0];
        for (int i = 1; i < 32; ++i) v = v + part[i];
        part[32] = v;
    }
    __syncthreads();
    const float r = rsqrtf(part[32] / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = from_bf16(to_bf16(from_bf16(w[i]) * (vals[i] * r)));
        if (normed) normed[i] = to_bf16(v);
        const float other = __shfl_xor_sync(0xffffffff, v, 1);  // the pair's other value
        const int at = i - (n - rd);
        if (freqs && at >= 0) {
            const float c = freqs[at & ~1], d = freqs[(at & ~1) + 1];
            v = from_bf16(to_bf16(at & 1 ? other * d + v * c : v * c - other * d));
        }
        float mx = fabsf(v);
        for (int s = 16; s; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, s));
        const float rounded = fp8_round(v, mx);
        if (xq)
            xq[i] = rounded;
        else
            q[i] = to_bf16(rounded);
    }
}

// The start of the output side of upstream Attention.forward for one token: the query's rotation is taken back off
// o[heads][head_dim] (bf16; apply_rotary_emb's inverse on the last rd values of each head, rounded to bf16) and the
// result quantized (kernel.act_quant) as fp32. A block of 32 threads per 32 values; head_dim - rd is even.
__global__ void unrope_quant_k(const uint16_t* o, const float* freqs, float* xq, int head_dim, int rd) {
    const int i = blockIdx.x * 32 + threadIdx.x, at = i % head_dim - (head_dim - rd);
    float v = from_bf16(o[i]);
    const float other = __shfl_xor_sync(0xffffffff, v, 1);  // the pair's other value
    if (at >= 0) {
        const float c = freqs[at & ~1], d = -freqs[(at & ~1) + 1];
        v = from_bf16(to_bf16(at & 1 ? other * d + v * c : v * c - other * d));
    }
    float mx = fabsf(v);
    for (int s = 16; s; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, s));
    xq[i] = fp8_round(v, mx);
}

// kernel.sparse_attn for one query token, block h for head h: kv is both key and value, position idx[t] being
// window[idx] below `win` and compressed[idx - win] from there on, -1 an empty slot. A learned sink logit joins the
// softmax's denominator; the probabilities go through bf16 before weighting the values, as upstream.
// Every read takes four adjacent bf16 values (d is a multiple of 4), and neighbouring lanes read neighbouring
// words: a thread walking a vector of its own made each read of a warp 32 scattered ones (0.6 us a position).
__device__ __forceinline__ float bf16_at(uint64_t w, int i) { return from_bf16(uint16_t(w >> (16 * i))); }

__global__ void sparse_attn_k(const uint16_t* q, const uint16_t* window, const uint16_t* compressed, const int* idx,
                              const float* sink, uint16_t* out, int count, int d, int win, float scale) {
    __shared__ float p[1024];
    __shared__ float part[4096];  // the output's partial sums: [share of the positions][d]
    const uint16_t* qh = q + size_t(blockIdx.x) * d;
    const int lane = threadIdx.x & 31;
    float mx = -INFINITY;
    for (int t = threadIdx.x >> 5; t < count; t += blockDim.x >> 5) {  // a warp scores one position at a time
        const int at = idx[t];
        float sc = -INFINITY;
        if (at >= 0) {
            const uint16_t* kv = at < win ? window + size_t(at) * d : compressed + size_t(at - win) * d;
            sc = 0;
            for (int i = 4 * lane; i < d; i += 512) {  // four reads of each under way at a time
                uint64_t a[4], b[4];
                for (int j = 0; j < 4 && i + 128 * j < d; ++j)
                    a[j] = *(const uint64_t*)(qh + i + 128 * j), b[j] = *(const uint64_t*)(kv + i + 128 * j);
                for (int j = 0; j < 4 && i + 128 * j < d; ++j)
                    for (int k = 0; k < 4; ++k) sc += bf16_at(a[j], k) * bf16_at(b[j], k);
            }
            for (int o = 16; o; o >>= 1) sc += __shfl_xor_sync(0xffffffff, sc, o);
            sc *= scale;
        }
        if (lane == 0) p[t] = sc;
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
    // a thread weights four adjacent values of every parts-th position; the shares are added up afterwards
    const int groups = d / 4, parts = blockDim.x / groups, g = threadIdx.x % groups, share = threadIdx.x / groups;
    if (share < parts) {
        float o[4] = {0, 0, 0, 0};
        for (int t = share; t < count; t += 4 * parts) {  // four positions' reads under way at a time
            int at[4];
            uint64_t w[4];
            for (int j = 0; j < 4; ++j) at[j] = t + j * parts < count ? idx[t + j * parts] : -1;
            for (int j = 0; j < 4; ++j)
                if (at[j] >= 0)
                    w[j] = *(const uint64_t*)((at[j] < win ? window + size_t(at[j]) * d : compressed + size_t(at[j] - win) * d) + 4 * g);
            for (int j = 0; j < 4; ++j)
                if (at[j] >= 0)
                    for (int k = 0; k < 4; ++k) o[k] += p[t + j * parts] * bf16_at(w[j], k);
        }
        for (int k = 0; k < 4; ++k) part[share * d + 4 * g + k] = o[k];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float o = 0;
        for (int k = 0; k < parts; ++k) o += part[k * d + i];
        out[size_t(blockIdx.x) * d + i] = to_bf16(o / denom);
    }
}

// y[row] = sum over k of w[row][k] * scale[row / 32][k / 32] * xq[k], w in E4M3 and scale in E8M0.
// Rows come in groups of rows_per_group, group g reading xq[g * k ..] (attention's block-diagonal wo_a; one group
// otherwise).
// The sum is rounded to bf16 as the reference's GEMM output is, then stored as bf16 (STORE), added to the fp32 y
// (ADD) or stored there (STORE_F32).
enum Mode { STORE, ADD, STORE_F32 };

__device__ __forceinline__ void put(void* y, int row, float acc, Mode mode) {
    const uint16_t r = to_bf16(acc);
    if (mode == STORE)
        ((uint16_t*)y)[row] = r;
    else
        ((float*)y)[row] = (mode == ADD ? ((float*)y)[row] : 0.0f) + from_bf16(r);
}

// Four weights of an fp8 row times their x.
__device__ __forceinline__ float fp8_dot4(uint32_t b, const float* x) {
    return e4m3(b & 255) * x[0] + e4m3(b >> 8 & 255) * x[1] + e4m3(b >> 16 & 255) * x[2] + e4m3(b >> 24) * x[3];
}

// A row's sum of the product above, for the first of the row's threads; `row` is set for all of them. A block
// computes blockDim / per_row rows, per_row threads (32, or 64 for a long row) to a row: 32768 blocks of one short
// row each took a third longer. The order of the additions is fixed: a thread adds every per_row-th group of four
// columns in turn, a warp adds its threads up by shuffles, then the row's warps are added in order. Four reads are
// under way at a time.
__device__ __forceinline__ float fp8_row(const uint8_t* w, const uint8_t* scale, const float* xq, int k, int rows_per_group,
                                         int per_row, int& row) {
    __shared__ float part[32];
    row = blockIdx.x * (blockDim.x / per_row) + threadIdx.x / per_row;
    const int step = per_row * 4;
    const uint8_t* wr = w + size_t(row) * k;
    const uint8_t* sr = scale + size_t(row >> 5) * (k >> 5);
    const float* x = xq + size_t(row / rows_per_group) * k;
    float acc = 0;
    int c = threadIdx.x % per_row * 4;
    for (; c + 3 * step < k; c += 4 * step) {
        uint32_t b[4];
        for (int i = 0; i < 4; ++i) b[i] = *(const uint32_t*)(wr + c + i * step);
        for (int i = 0; i < 4; ++i) acc += e8m0(sr[(c + i * step) >> 5]) * fp8_dot4(b[i], x + c + i * step);
    }
    for (; c < k; c += step) acc += e8m0(sr[c >> 5]) * fp8_dot4(*(const uint32_t*)(wr + c), x + c);
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
    __syncthreads();
    acc = 0;
    if (threadIdx.x % per_row == 0)
        for (int i = 0; i < per_row >> 5; ++i) acc += part[(threadIdx.x >> 5) + i];
    return acc;
}

__global__ void fp8_gemv_k(const uint8_t* w, const uint8_t* scale, const float* xq, void* y, int k, int rows_per_group, Mode mode,
                           int per_row) {
    int row;
    const float acc = fp8_row(w, scale, xq, k, rows_per_group, per_row, row);
    if (threadIdx.x % per_row == 0) put(y, row, acc, mode);
}

// fp8_gemv_k for attention's wq_b, with the rotary embedding on its output q[heads][head_dim] (bf16): upstream
// apply_rotary_emb on the last rd values of each head, as in norm_quant. A block holds an even number of rows, so
// both values of a pair; head_dim and rd are even.
__global__ void fp8_gemv_rope_k(const uint8_t* w, const uint8_t* scale, const float* xq, uint16_t* q, int k, int per_row,
                                const float* freqs, int head_dim, int rd) {
    __shared__ float vals[32];  // the block's rows
    int row;
    float v = from_bf16(to_bf16(fp8_row(w, scale, xq, k, 1 << 30, per_row, row)));
    const bool first = threadIdx.x % per_row == 0;
    const int at = row % head_dim - (head_dim - rd), mine = threadIdx.x / per_row;
    if (first) vals[mine] = v;
    __syncthreads();
    if (!first) return;
    if (at >= 0) {
        const float other = vals[mine ^ 1], c = freqs[at & ~1], d = freqs[(at & ~1) + 1];
        v = at & 1 ? other * d + v * c : v * c - other * d;
    }
    q[row] = to_bf16(v);
}

// fp8_gemv_k for a projection that attention normalizes: the rows go to y[n] (bf16), and the block that finishes
// last does the rest (norm_quant), which saves the launches of three small kernels.
__global__ void fp8_gemv_norm_k(const uint8_t* w, const uint8_t* scale, const float* xq_in, uint16_t* y, int k, int per_row, int n,
                                int* count, const uint16_t* norm_w, const float* freqs, uint16_t* normed, float* xq,
                                uint16_t* q, int rd, float eps) {
    int row;
    const float acc = fp8_row(w, scale, xq_in, k, n, per_row, row);
    if (threadIdx.x % per_row == 0) y[row] = to_bf16(acc);
    if (last_block(count, gridDim.x)) norm_quant(y, norm_w, freqs, normed, xq, q, n, rd, eps);
}

// fp8_gemv_k for attention's wo_a: the rows go to y (bf16), and of the blocks that share 32 rows, the one that
// finishes last quantizes them for wo_b as act_quant_k does -> xq (fp32), which is not xq_in. A block's rows divide 32.
__global__ void fp8_gemv_quant_k(const uint8_t* w, const uint8_t* scale, const float* xq_in, uint16_t* y, int k, int rows_per_group,
                                 int per_row, int* counts, float* xq) {
    int row;
    const float acc = fp8_row(w, scale, xq_in, k, rows_per_group, per_row, row);
    if (threadIdx.x % per_row == 0) y[row] = to_bf16(acc);
    const int blocks = 32 * per_row / blockDim.x, group = blockIdx.x / blocks;
    if (!last_block(counts + group, blocks) || threadIdx.x >= 32) return;
    const int i = group * 32 + threadIdx.x;
    const float v = from_bf16(((const volatile uint16_t*)y)[i]);
    float mx = fabsf(v);
    for (int s = 16; s; s >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, s));
    xq[i] = fp8_round(v, mx);
}

// What the expert kernels need of one SwiGLU expert: its weights with their scales, and its routing weight.
struct Mats {
    const uint8_t *w1, *s1, *w2, *s2, *w3, *s3;
    float weight;
};
struct Batch {
    Mats m[8];
    int n;
};

// One warp's <w[row], x> for a stored row of k weights: packed FP4 (k / 2 bytes, the even column in the low nibble,
// an E8M0 scale per 32 columns) or FP8 (as fp8_gemv_k reads it); sr is the row's scales. Lane 0 gets the sum. The
// additions run in fp8_gemv_k's order, which gives a long row 64 threads: a lane then does the work of two in turn.
template <bool FP4>
__device__ __forceinline__ float row_dot(const uint8_t* wr, const uint8_t* sr, const float* x, int k, int lane) {
    const int per = FP4 ? 8 : 4, warps = k / per >= 512 ? 2 : 1;
    float total = 0;
    for (int w = 0; w < warps; ++w) {
        float acc = 0;
        for (int c = (w * 32 + lane) * per; c < k; c += warps * 32 * per) {
            if (FP4) {
                const uint32_t b = *(const uint32_t*)(wr + (c >> 1));
                float s = 0;
                for (int i = 0; i < 8; ++i) s += e2m1x2(b >> (4 * i)) * x[c + i];
                acc += 0.5f * e8m0(sr[c >> 5]) * s;
            } else {
                const uint32_t b = *(const uint32_t*)(wr + c);
                acc += e8m0(sr[c >> 5]) * (e4m3(b & 255) * x[c] + e4m3(b >> 8 & 255) * x[c + 1] + e4m3(b >> 16 & 255) * x[c + 2] +
                                           e4m3(b >> 24) * x[c + 3]);
            }
        }
        for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
        total += acc;
    }
    return total;
}

// The first half of the reference Expert for a batch of experts on the quantized layer input xq. A block computes
// 32 rows of the hidden vector of expert blockIdx / (inter / 32), a warp per row: the row of w1 and of w3 (each
// rounded to bf16 as the reference's GEMM output is), h = weight * silu(min(gate, limit)) * clamp(up, -limit, limit)
// rounded to bf16, and the block's 32 values quantized as kernel.act_quant does before w2 -> hq[expert][inter].
template <bool FP4>
__global__ void expert_hidden_k(Batch b, const float* xq, float* hq, int dim, int inter, float limit) {
    __shared__ float vals[32];
    const int blocks = inter / 32, e = blockIdx.x / blocks, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = (blockIdx.x % blocks) * 32 + warp;
    const Mats& m = b.m[e];
    const size_t at = size_t(row) * (FP4 ? dim / 2 : dim), scale_at = size_t(FP4 ? row : row >> 5) * (dim >> 5);
    const float gate = row_dot<FP4>(m.w1 + at, m.s1 + scale_at, xq, dim, lane);
    const float up = row_dot<FP4>(m.w3 + at, m.s3 + scale_at, xq, dim, lane);
    if (lane == 0) {
        float g = from_bf16(to_bf16(gate)), u = from_bf16(to_bf16(up));
        if (limit > 0) g = fminf(g, limit), u = fminf(fmaxf(u, -limit), limit);
        vals[warp] = from_bf16(to_bf16(m.weight * (g / (1.0f + expf(-g)) * u)));
    }
    __syncthreads();
    if (lane) return;
    float mx = 0;
    for (int i = 0; i < 32; ++i) mx = fmaxf(mx, fabsf(vals[i]));
    hq[size_t(e) * inter + row] = fp8_round(vals[warp], mx);
}

// The second half, one warp per output row: y[row] (fp32), or zero with STORE_F32, plus each expert's
// <w2[row], hq[expert]> rounded to bf16, added in the batch's order.
template <bool FP4>
__global__ void expert_out_k(Batch b, const float* hq, float* y, int dim, int inter, Mode mode) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const size_t at = size_t(row) * (FP4 ? inter / 2 : inter), scale_at = size_t(FP4 ? row : row >> 5) * (inter >> 5);
    float sum = lane == 0 && mode == ADD ? y[row] : 0.0f;
    for (int e = 0; e < b.n; ++e)
        sum += from_bf16(to_bf16(row_dot<FP4>(b.m[e].w2 + at, b.m[e].s2 + scale_at, hq + size_t(e) * inter, inter, lane)));
    if (lane == 0) y[row] = sum;
}

// <wr, x> over k values for a bf16 or fp32 wr and x in bf16, by one block; thread 0 gets it. A thread adds every
// blockDim-th product in turn, with four reads under way at a time.
template <class W>
__device__ __forceinline__ float dense_dot(const W* wr, const uint16_t* x, int k) {
    const int step = blockDim.x;
    float acc = 0;
    int c = threadIdx.x;
    for (; c + 3 * step < k; c += 4 * step) {
        float a[4], b[4];
        for (int i = 0; i < 4; ++i) a[i] = widen(wr[c + i * step]), b[i] = from_bf16(x[c + i * step]);
        for (int i = 0; i < 4; ++i) acc += a[i] * b[i];
    }
    for (; c < k; c += step) acc += widen(wr[c]) * from_bf16(x[c]);
    return block_sum(acc);
}

// out[row] = <w[row], x> for a bf16 or fp32 weight with rows of k values and x in bf16; block `row` computes one.
// F.linear on unquantized weights.
template <class W>
__global__ void dots_k(const W* w, const uint16_t* x, float* out, int k) {
    const W* wr = w + size_t(blockIdx.x) * k;
    const float acc = dense_dot(wr, x, k);
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// upstream Compressor.forward for one token, after its projections: v[i] = sum over the group's `ratio` tokens s of
// kv[s][i] * softmax over s of score[s][i] (ratio 1: kv[i] itself), rounded to bf16; out = RMSNorm of v with
// weight w. One block of n threads.
__global__ void latent_k(const float* kv, const float* score, int ratio, const uint16_t* w, uint16_t* out, int n, float eps) {
    const int i = threadIdx.x;
    float v = kv[i];
    if (ratio > 1) {
        float mx = -INFINITY, den = 0;
        for (int s = 0; s < ratio; ++s) mx = fmaxf(mx, score[s * n + i]);
        for (int s = 0; s < ratio; ++s) den += expf(score[s * n + i] - mx);
        v = 0;
        for (int s = 0; s < ratio; ++s) v += kv[s * n + i] * (expf(score[s * n + i] - mx) / den);
    }
    v = from_bf16(to_bf16(v));
    const float r = rsqrtf(block_reduce(v * v, false) / float(n) + eps);
    out[i] = to_bf16(from_bf16(w[i]) * (v * r));
}

// What upstream does to a compressed latent or its index key before caching it, one block of n threads:
// v = bf16(in); with norm_w, v = RMSNorm(v); the last rd values are rotated (apply_rotary_emb with freqs) and
// rounded to bf16; then kernel.fp4_act_quant in place over groups of `group` values, with power-of-two scales
// (the index keys) or E4M3 scales (the compressed KV).
template <class In>
__global__ void norm_rope_fp4_k(const In* in, const uint16_t* norm_w, const float* freqs, uint16_t* out, int n, int rd,
                                int group, bool e4m3_scale, float eps) {
    __shared__ float vals[1024];
    const int i = threadIdx.x;
    float v = from_bf16(to_bf16(widen(in[i])));
    if (norm_w) {
        const float r = rsqrtf(block_reduce(v * v, false) / float(n) + eps);
        v = from_bf16(to_bf16(from_bf16(norm_w[i]) * (v * r)));
    }
    vals[i] = v;
    __syncthreads();
    if (i >= n - rd) {
        const int pair = (i - (n - rd)) >> 1, at = n - rd + 2 * pair;
        const float a = vals[at], b = vals[at + 1], c = freqs[2 * pair], d = freqs[2 * pair + 1];
        v = from_bf16(to_bf16(i == at ? a * c - b * d : a * d + b * c));
    }
    __syncthreads();
    vals[i] = v;
    __syncthreads();
    float mx = 0, s;
    for (int k = 0; k < group; ++k) mx = fmaxf(mx, fabsf(vals[i / group * group + k]));
    if (e4m3_scale) {  // amax / 6 rounded to E4M3; even an all-zero group keeps a nonzero scale
        const float t = fmaxf(mx, scalbnf(6.0f, -9)) / 6.0f;
        const int q = max(ilogbf(t), -6) - 3;
        s = scalbnf(rintf(scalbnf(t, -q)), q);
    } else {  // 2^ceil(log2(amax / 6))
        const uint32_t bits = __float_as_uint(fmaxf(mx, scalbnf(6.0f, -126)) * float(1.0 / 6.0));
        s = scalbnf(1.0f, int(bits >> 23 & 255) - 127 + ((bits & 0x7fffff) != 0));
    }
    out[i] = to_bf16(round_e2m1(fminf(fmaxf(v / s, -6.0f), 6.0f)) * s);
}

// kernel.fp4_act_quant in place on bf16 values with power-of-two scales, a warp to each group of 32.
__global__ void fp4_quant_k(uint16_t* v) {
    const int i = blockIdx.x * 32 + threadIdx.x;
    const float x = from_bf16(v[i]);
    float mx = fabsf(x);
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
    const uint32_t bits = __float_as_uint(fmaxf(mx, scalbnf(6.0f, -126)) * float(1.0 / 6.0));  // 2^ceil(log2(amax / 6))
    const float s = scalbnf(1.0f, int(bits >> 23 & 255) - 127 + ((bits & 0x7fffff) != 0));
    v[i] = to_bf16(round_e2m1(fminf(fmaxf(x / s, -6.0f), 6.0f)) * s);
}

// upstream Indexer.forward's score of compressed position t (block t) for one token: the sum over the heads h of
// relu(<q[h], key[t]>) * weight[h], where weight[h] = dots[h] * scale and every product and sum is rounded to bf16,
// as upstream's bf16 tensors are. q[heads][d] and keys[..][d] are bf16; a warp takes a head.
__global__ void index_score_k(const uint16_t* q, const uint16_t* keys, const float* dots, float scale, float* scores, int d) {
    __shared__ float part[32];
    const int h = threadIdx.x >> 5, lane = threadIdx.x & 31, heads = blockDim.x >> 5;
    const uint16_t* qh = q + h * d;
    const uint16_t* key = keys + size_t(blockIdx.x) * d;
    float acc = 0;
    for (int c = lane; c < d; c += 32) acc += from_bf16(qh[c]) * from_bf16(key[c]);
    for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane == 0) {
        const float w = from_bf16(to_bf16(from_bf16(to_bf16(dots[h])) * from_bf16(to_bf16(scale))));
        part[h] = from_bf16(to_bf16(fmaxf(from_bf16(to_bf16(acc)), 0.0f) * w));
    }
    __syncthreads();
    if (threadIdx.x) return;
    float s = 0;
    for (int i = 0; i < heads; ++i) s += part[i];
    scores[blockIdx.x] = from_bf16(to_bf16(s));
}

// pick[i] = whether scores[i] is one of the k largest of the n scores (of equal ones, the earlier). A thread takes
// a score.
__global__ void topk_mark_k(const float* scores, int* pick, int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = scores[i];
    int above = 0;
    for (int j = 0; j < n; ++j) {
        const float o = scores[j];
        above += o > s || (o == s && j < i);
    }
    pick[i] = above < k;
}

// out = the positions that pick[n] marks, in order, each plus `offset`. One block; a thread takes a run of positions.
__global__ void topk_list_k(const int* pick, int* out, int n, int offset) {
    __shared__ int marked[1024];
    const int run = (n + blockDim.x - 1) / blockDim.x, lo = threadIdx.x * run, hi = min(n, lo + run);
    int mine = 0;
    for (int i = lo; i < hi; ++i) mine += pick[i];
    marked[threadIdx.x] = mine;
    __syncthreads();
    int at = 0;
    for (unsigned t = 0; t < threadIdx.x; ++t) at += marked[t];
    for (int i = lo; i < hi; ++i)
        if (pick[i]) out[at++] = i + offset;
}

// upstream Gate.forward for one token x[k] (bf16) with the "sqrtsoftplus" score. Block e computes expert e's score
// <w[e], x> and copies its share of the first `copy` values of x; the block that finishes last makes the choice:
// a score becomes sqrt(softplus(score / temp)), the topk largest of score + bias are picked (best first, the lower
// index of equal ones) and weighted by their unbiased scores, normalized over the picks when norm is set, times
// route_scale.
// pack = the topk indices (as fp32), their weights, then x as fp32: all the host needs of this layer in one transfer.
__global__ void gate_k(const uint16_t* w, const float* bias, const uint16_t* x, float* scores, int* count, float* pack, int n,
                       int topk, int k, int copy, float temp, float route_scale, bool norm) {
    extern __shared__ float mem[];  // key[n], score[n]
    __shared__ float best_key[32];
    __shared__ int best_at[32];
    const float dot = dense_dot(w + size_t(blockIdx.x) * k, x, k);
    if (threadIdx.x == 0) scores[blockIdx.x] = dot;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < copy; c += gridDim.x * blockDim.x) pack[2 * topk + c] = from_bf16(x[c]);
    if (!last_block(count, gridDim.x)) return;
    float *key = mem, *score = mem + n;
    for (int e = threadIdx.x; e < n; e += blockDim.x) {
        const float v = ((const volatile float*)scores)[e] / temp;
        score[e] = sqrtf(v > 20.0f ? v : log1pf(expf(v)));  // F.softplus and its threshold
        key[e] = score[e] + bias[e];
    }
    __syncthreads();
    float last_key = INFINITY;
    int last_at = -1;
    for (int pick = 0; pick < topk; ++pick) {  // the first of the experts that come after the last pick
        float top = -INFINITY;
        int at = n;
        for (int e = threadIdx.x; e < n; e += blockDim.x)
            if ((key[e] < last_key || (key[e] == last_key && e > last_at)) && (key[e] > top || (key[e] == top && e < at)))
                top = key[e], at = e;
        for (int o = 16; o; o >>= 1) {
            const float other = __shfl_xor_sync(0xffffffff, top, o);
            const int other_at = __shfl_xor_sync(0xffffffff, at, o);
            if (other > top || (other == top && other_at < at)) top = other, at = other_at;
        }
        if ((threadIdx.x & 31) == 0) best_key[threadIdx.x >> 5] = top, best_at[threadIdx.x >> 5] = at;
        __syncthreads();
        for (unsigned i = 0; i < blockDim.x >> 5; ++i)
            if (best_key[i] > top || (best_key[i] == top && best_at[i] < at)) top = best_key[i], at = best_at[i];
        __syncthreads();
        last_key = top, last_at = at;
        if (threadIdx.x == 0) pack[pick] = float(at), pack[topk + pick] = score[at];
    }
    if (threadIdx.x) return;
    float sum = 0;
    for (int i = 0; i < topk; ++i) sum += pack[topk + i];
    for (int i = 0; i < topk; ++i) {
        if (norm && topk > 1) pack[topk + i] /= sum + 1e-20f;
        pack[topk + i] *= route_scale;
    }
}

// The two-tier MoE's last step: out = bf16((routed + ram) + shared), three fp32 sums of bf16 expert outputs.
__global__ void moe_sum_k(const float* routed, const float* ram, const float* shared, uint16_t* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = to_bf16((routed[i] + ram[i]) + shared[i]);
}

// Second half of the hyper-connection mixes (kernel.hc_split_sinkhorn on mixes = dots * rsqrt(mean square +
// norm_eps)), one warp: out = pre[hc], post[hc], comb[hc][hc] with comb made doubly stochastic by `iters` Sinkhorn
// rounds. A lane holds one entry of comb (hc * hc <= 32); a single thread took 30 us over the divisions.
__device__ void hc_split(const volatile float* dots, const float* scale, const float* base, float* out, int n, int hc,
                         float norm_eps, float eps, int iters) {
    const unsigned all = 0xffffffff;
    const int lane = threadIdx.x, j = min(lane / hc, hc - 1), c = lane % hc;  // lanes past hc * hc repeat the last row
    const float r = rsqrtf(dots[(2 + hc) * hc] / float(n) + norm_eps);
    if (lane < hc) {
        out[lane] = 1.0f / (1.0f + expf(-(dots[lane] * r * scale[0] + base[lane]))) + eps;
        out[hc + lane] = 2.0f / (1.0f + expf(-(dots[hc + lane] * r * scale[1] + base[hc + lane])));
    }
    // softmax over each row, then columns, then alternate; every sum runs over its row or column in order
    float v = dots[2 * hc + j * hc + c] * r * scale[2] + base[2 * hc + j * hc + c], mx = -INFINITY, sum = 0;
    for (int k = 0; k < hc; ++k) mx = fmaxf(mx, __shfl_sync(all, v, j * hc + k));
    v = expf(v - mx);
    for (int k = 0; k < hc; ++k) sum += __shfl_sync(all, v, j * hc + k);
    v = v / sum + eps;
    for (int it = 0; it < 2 * iters - 1; ++it) {  // even rounds: columns (sum over rows); odd rounds: rows
        sum = 0;
        for (int k = 0; k < hc; ++k) sum += __shfl_sync(all, v, it & 1 ? j * hc + k : k * hc + c);
        v /= sum + eps;
    }
    if (lane < hc * hc) out[2 * hc + lane] = v;
}

// Hyper-connection mixes (Block.hc_mixes). Block i < rows computes <fn[i], h> over the n values of the flattened
// stream h (bf16), block `rows` its sum of squares. The additions keep one order (a thread adds every 256th value
// in turn; the mixes magnify a changed last bit), but four reads are under way at a time. The block that finishes
// last turns the dots into the mixes (hc_split).
__global__ void hc_mixes_k(const uint16_t* h, const float* fn, float* dots, int n, int rows, int* count, const float* scale,
                           const float* base, float* out, int hc, float norm_eps, float eps, int iters) {
    const int row = blockIdx.x, step = blockDim.x;
    const float* f = fn + size_t(row < rows ? row : 0) * n;
    float acc = 0;
    int c = threadIdx.x;
    for (; c + 3 * step < n; c += 4 * step) {
        float v[4], w[4];
        for (int k = 0; k < 4; ++k) v[k] = from_bf16(h[c + k * step]), w[k] = row < rows ? f[c + k * step] : v[k];
        for (int k = 0; k < 4; ++k) acc += w[k] * v[k];
    }
    for (; c < n; c += step) {
        const float v = from_bf16(h[c]);
        acc += (row < rows ? f[c] : v) * v;
    }
    acc = block_reduce(acc, false);
    if (threadIdx.x == 0) dots[row] = acc;
    if (last_block(count, gridDim.x) && threadIdx.x < 32) hc_split(dots, scale, base, out, n, hc, norm_eps, eps, iters);
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
    uint16_t* h = nullptr;  // an intermediate vector
    int* picks = nullptr;  // marks of topk_mark_k
} scratch;

void ensure_scratch() {
    if (scratch.xq) return;
    cudaMalloc(&scratch.xq, SCRATCH * sizeof(float));
    cudaMalloc(&scratch.dots, 4096 * sizeof(float));
    cudaMalloc(&scratch.h, SCRATCH * sizeof(uint16_t));
    cudaMalloc(&scratch.picks, SCRATCH * sizeof(int));
}

int threads_for(int k, int per_thread) { return k / per_thread >= 512 ? 64 : 32; }

// How the GEMV kernels are launched over n rows of k columns: the threads a row takes, and four rows to a block.
struct Rows {
    int per_row, blocks, threads;
    Rows(int n, int k) : per_row(threads_for(k, 4)), blocks(n % 4 ? n : n / 4), threads(n % 4 ? per_row : 4 * per_row) {}
};

template <bool FP4>
int launch_experts(const Batch& b, int dim, int inter, float limit, const float* xq, float* y, Mode mode) {
    if (dim > SCRATCH || b.n * inter > SCRATCH || inter % 32) return -1;
    ensure_scratch();
    expert_hidden_k<FP4><<<b.n * (inter / 32), 1024>>>(b, xq, scratch.xq, dim, inter, limit);
    expert_out_k<FP4><<<dim, 32>>>(b, scratch.xq, y, dim, inter, mode);
    return cudaGetLastError();
}

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
    const Rows rows(n, k);
    fp8_gemv_k<<<rows.blocks, rows.threads>>>(w, scale, scratch.xq, y, k, n / groups, STORE, rows.per_row);
    return cudaGetLastError();
}

// y[dim] (fp32) += the sum over n routed experts (1 to 8, added in order) of weights[i] * expert slots[i]. An
// expert's checkpoint bytes lie at store + slots[i] * stride: the scales of w1, w2, w3, then their packed FP4
// weights. xq[dim] is the layer input already quantized (sclm_gpu_act_quant).
API int sclm_gpu_fp4_experts(const uint8_t* store, uint64_t stride, const int* slots, const float* weights, int n, int dim,
                             int inter, float limit, const float* xq, float* y) {
    if (n < 1 || n > 8) return -1;
    Batch b;
    b.n = n;
    for (int i = 0; i < n; ++i) {
        const uint8_t *s1 = store + slots[i] * stride, *s2 = s1 + size_t(inter) * (dim / 32), *s3 = s2 + size_t(dim) * (inter / 32);
        const uint8_t *w1 = s3 + size_t(inter) * (dim / 32), *w2 = w1 + size_t(inter) * (dim / 2), *w3 = w2 + size_t(dim) * (inter / 2);
        b.m[i] = {w1, s1, w2, s2, w3, s3, weights[i]};
    }
    return launch_experts<true>(b, dim, inter, limit, xq, y, ADD);
}

// y[dim] (fp32) += the upstream Expert with fp8 weights and block scales (the shared expert) on the layer input
// quantized as xq[dim]; store: y is overwritten instead. w1 and w3 are [inter, dim], w2 is [dim, inter].
API int sclm_gpu_fp8_expert(const uint8_t* w1, const uint8_t* s1, const uint8_t* w2, const uint8_t* s2, const uint8_t* w3,
                            const uint8_t* s3, int dim, int inter, float limit, const float* xq, float* y, int store) {
    Batch b;
    b.n = 1, b.m[0] = {w1, s1, w2, s2, w3, s3, 1.0f};
    return launch_experts<false>(b, dim, inter, limit, xq, y, store ? STORE_F32 : ADD);
}

// The query side of upstream Attention.forward for one token: qr[rank] = RMSNorm(wq_a x), then
// q[heads * head_dim] = wq_b qr with the rotary embedding on the last rd values of each head. All bf16; the weights
// are fp8 with block scales, freqs the position's rd / 2 complex rotations.
API int sclm_gpu_attn_q(const uint8_t* wa, const uint8_t* sa, const uint16_t* norm_w, const uint8_t* wb, const uint8_t* sb,
                        const uint16_t* x, const float* freqs, uint16_t* qr, uint16_t* q, int dim, int rank, int heads,
                        int head_dim, int rd, float eps) {
    if (rank > NORM_MAX || rank % 32 || head_dim % 4 || rd % 2) return -1;
    ensure_scratch();
    const Rows a(rank, dim), b(heads * head_dim, rank);
    act_quant_k<float><<<dim / 32, 32>>>(x, scratch.xq);
    // the quantized qr takes the place of the quantized x, which the last block no longer needs
    fp8_gemv_norm_k<<<a.blocks, a.threads, rank * sizeof(float)>>>(wa, sa, scratch.xq, scratch.h, dim, a.per_row, rank, done_counts(),
                                                                   norm_w, nullptr, qr, scratch.xq, nullptr, 0, eps);
    fp8_gemv_rope_k<<<b.blocks, b.threads>>>(wb, sb, scratch.xq, q, rank, b.per_row, freqs, head_dim, rd);
    return cudaGetLastError();
}

// The key/value of one token as upstream Attention._window_kv makes it: kv[head_dim] = fp8-quantized
// rotary(RMSNorm(wkv x)), written where the caller points (its slot of the window ring).
API int sclm_gpu_attn_kv(const uint8_t* w, const uint8_t* s, const uint16_t* norm_w, const uint16_t* x, const float* freqs,
                         uint16_t* kv, int dim, int head_dim, int rd, float eps) {
    if (head_dim > NORM_MAX || head_dim % 32 || rd % 2) return -1;
    ensure_scratch();
    const Rows rows(head_dim, dim);
    act_quant_k<float><<<dim / 32, 32>>>(x, scratch.xq);
    fp8_gemv_norm_k<<<rows.blocks, rows.threads, head_dim * sizeof(float)>>>(w, s, scratch.xq, scratch.h, dim, rows.per_row, head_dim,
                                                                             done_counts(), norm_w, freqs, nullptr, nullptr, kv, rd, eps);
    return cudaGetLastError();
}

// One decode token of upstream's compressed KV at a layer that makes it (Compressor.forward, the index key of
// Indexer.forward, the rotation and rounding of Attention._compress_kv), `ratio` tokens to a group.
// ratio > 1: kv_state and score_state [ratio][d] (fp32) get the token's projections at `slot` (wkv and wgate are
// fp32 [d, dim]); ratio 1: wkv is bf16 and nothing is pooled. When the token completes its group
// (slot == ratio - 1), kv_out[d] gets the group's latent with the rotation `freqs` of its first position, as FP4
// values over groups of 16 with E4M3 scales, and k_out[kd] its index key (wk bf16 [kd, d], or null for none) as FP4
// values over groups of 32. x, the norm weights and the outputs are bf16.
API int sclm_gpu_compress(const void* wkv, const float* wgate, const uint16_t* norm_w, const uint16_t* x, float* kv_state,
                          float* score_state, int ratio, int slot, const uint16_t* wk, const uint16_t* k_norm_w,
                          const float* freqs, uint16_t* k_out, uint16_t* kv_out, int dim, int d, int kd, int rd, float eps) {
    if (d > 1024 || d % 32 || kd > 1024 || kd % 32) return -1;
    ensure_scratch();
    const float* kv = kv_state;
    if (ratio > 1) {
        dots_k<float><<<d, 128>>>((const float*)wkv, x, kv_state + slot * d, dim);
        dots_k<float><<<d, 128>>>(wgate, x, score_state + slot * d, dim);
        if (slot != ratio - 1) return cudaGetLastError();
    } else {
        dots_k<uint16_t><<<d, 128>>>((const uint16_t*)wkv, x, scratch.xq, dim);
        kv = scratch.xq;
    }
    latent_k<<<1, d>>>(kv, score_state, ratio, norm_w, scratch.h, d, eps);
    if (wk) {
        dots_k<uint16_t><<<kd, 128>>>(wk, scratch.h, scratch.dots, d);
        norm_rope_fp4_k<float><<<1, kd>>>(scratch.dots, k_norm_w, freqs, k_out, kd, rd, 32, false, eps);
    }
    norm_rope_fp4_k<uint16_t><<<1, d>>>(scratch.h, nullptr, freqs, kv_out, d, rd, 16, true, eps);
    return cudaGetLastError();
}

// One decode token of upstream Indexer.forward once there are more compressed positions than it keeps: out[topk]
// (int32) = the positions of the topk highest scores among the first `count` of keys[..][d] (bf16), in order, each
// plus `offset`. wb/sb: the indexer's fp8 wq_b [heads * d, rank]; qr[rank]: attention's normed low-rank query; wp:
// weights_proj [heads, dim]; x[dim]: the token (all bf16); freqs: the position's rotation; scale: of the weights.
// Equal scores (bf16 has many) go to the earlier position; upstream's choice among them is unspecified.
API int sclm_gpu_index_pick(const uint8_t* wb, const uint8_t* sb, const uint16_t* qr, const uint16_t* wp, const uint16_t* x,
                            const float* freqs, const uint16_t* keys, int* out, int rank, int heads, int d, int rd, int dim,
                            int count, int topk, int offset, float scale) {
    if (heads > 32 || d % 32 || heads * d > SCRATCH || rank % 32 || rank > SCRATCH || rd % 2 || count > SCRATCH) return -1;
    ensure_scratch();
    const Rows b(heads * d, rank);
    act_quant_k<float><<<rank / 32, 32>>>(qr, scratch.xq);
    fp8_gemv_rope_k<<<b.blocks, b.threads>>>(wb, sb, scratch.xq, scratch.h, rank, b.per_row, freqs, d, rd);
    fp4_quant_k<<<heads * d / 32, 32>>>(scratch.h);
    dots_k<uint16_t><<<heads, 128>>>(wp, x, scratch.dots, dim);
    index_score_k<<<count, heads * 32>>>(scratch.h, keys, scratch.dots, scale, scratch.xq, d);  // the quantized qr is used up
    topk_mark_k<<<(count + 255) / 256, 256>>>(scratch.xq, scratch.picks, count, topk);
    topk_list_k<<<1, 1024>>>(scratch.picks, out, count, offset);
    return cudaGetLastError();
}

// out[heads * d] (bf16) = kernel.sparse_attn of one token's q over the `count` positions idx names (see sparse_attn_k).
API int sclm_gpu_sparse_attn(const uint16_t* q, const uint16_t* window, const uint16_t* compressed, const int* idx,
                             const float* sink, uint16_t* out, int heads, int d, int count, int win, float scale) {
    if (count > 1024 || d % 4 || d > 4096) return -1;
    sparse_attn_k<<<heads, 1024>>>(q, window, compressed, idx, sink, out, count, d, win, scale);
    return cudaGetLastError();
}

// The output side of upstream Attention.forward for one token: the rotary embedding is taken back off o, then
// out[dim] = wo_b (wo_a o) with wo_a block-diagonal over `groups` and both fp8 with block scales. n_a is wo_a's
// row count.
API int sclm_gpu_attn_out(const uint16_t* o, const float* freqs, const uint8_t* wa, const uint8_t* sa, const uint8_t* wb,
                          const uint8_t* sb, uint16_t* out, int heads, int head_dim, int rd, int groups, int n_a, int dim) {
    const int width = heads * head_dim;
    if (width + n_a > SCRATCH || n_a % 32 || (head_dim - rd) % 2 || head_dim % 32) return -1;
    ensure_scratch();
    const Rows a(n_a, width / groups), b(dim, n_a);
    unrope_quant_k<<<width / 32, 32>>>(o, freqs, scratch.xq, head_dim, rd);
    float* xq = scratch.xq + width;  // wo_a's output, quantized
    fp8_gemv_quant_k<<<a.blocks, a.threads>>>(wa, sa, scratch.xq, scratch.h, width / groups, n_a / groups, a.per_row, done_counts(), xq);
    fp8_gemv_k<<<b.blocks, b.threads>>>(wb, sb, xq, out, n_a, dim, STORE, b.per_row);
    return cudaGetLastError();
}

// mixes[(2 + hc) * hc] (fp32) = Block.hc_mixes of the stream h[hc * dim] (bf16): pre[hc], post[hc], comb[hc][hc].
// fn is [(2 + hc) * hc][hc * dim], scale [3], base [(2 + hc) * hc], all fp32.
API int sclm_gpu_hc_mixes(const uint16_t* h, const float* fn, const float* scale, const float* base, float* mixes, int dim,
                          int hc, float norm_eps, float eps, int iters) {
    if (hc > 5) return -1;  // hc_split keeps comb in one warp
    ensure_scratch();
    const int rows = (2 + hc) * hc;
    hc_mixes_k<<<rows + 1, 256>>>(h, fn, scratch.dots, hc * dim, rows, done_counts(), scale, base, mixes, hc, norm_eps, eps, iters);
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

// pack[2 * topk + dim] (fp32) = upstream Gate.forward of one token x[dim] (bf16) with the "sqrtsoftplus" score: the
// topk expert indices, their routing weights, then x itself as fp32. weight is bf16 [n, dim], bias fp32 [n].
API int sclm_gpu_gate(const uint16_t* weight, const float* bias, const uint16_t* x, float* pack, int n, int topk, int dim,
                      float temp, float route_scale, int norm) {
    if (n > 1024) return -1;
    ensure_scratch();
    gate_k<<<n, 128, 2 * n * sizeof(float)>>>(weight, bias, x, scratch.dots, done_counts(), pack, n, topk, dim, dim, temp, route_scale,
                                              norm != 0);
    return cudaGetLastError();
}

// out[2 * k] (fp32) = the k experts a gate (weight, bias) would pick, and their scores, if its layer's input were
// RMSNorm (weight norm_w) of Block.hc_pre(h, pre): the gate of the next layer applied to this layer's input, a
// guess at what the next layer will route to. Shares scratch with sclm_gpu_gate, so it is launched after it.
API int sclm_gpu_gate_guess(const uint16_t* weight, const float* bias, const uint16_t* h, const float* pre,
                            const uint16_t* norm_w, float* out, int n, int k, int dim, int hc, float temp, float eps) {
    if (n > 1024 || k > n) return -1;
    ensure_scratch();
    hc_pre_norm_k<<<1, 1024>>>(h, pre, norm_w, scratch.h, dim, hc, eps);
    gate_k<<<n, 128, 2 * n * sizeof(float)>>>(weight, bias, scratch.h, scratch.dots, done_counts(), out, n, k, dim, 0, temp, 1.0f, false);
    return cudaGetLastError();
}

static cudaEvent_t moe_copied[2];  // reached when the last sclm_gpu_moe_begin's routing is on the host; when its guess is

// The start of the two-tier MoE for one token x[dim], in one call that returns as soon as the host has the routing.
// Queued in this order: sclm_gpu_gate into pack and the copy of those 2 * topk + dim floats to `host` (page-locked);
// with g_weight, sclm_gpu_gate_guess (k picks of the gate g_weight, g_bias on the stream h mixed by pre and normed
// by norm_w) into the 2 * k floats after them and their copy (sclm_gpu_moe_guess waits for it); then what does not
// depend on the routing: routed[dim] = 0, xq[dim] = sclm_gpu_act_quant(x) and shared[dim] = the shared expert
// (sclm_gpu_fp8_expert, stored). Only the first copy is waited for: the device goes on while the host works.
API int sclm_gpu_moe_begin(const uint16_t* weight, const float* bias, const uint16_t* x, float* pack, float* host, int n,
                           int topk, int dim, float temp, float route_scale, int norm, const uint16_t* g_weight,
                           const float* g_bias, const uint16_t* h, const float* pre, const uint16_t* norm_w, int k, int hc,
                           float g_temp, float eps, const uint8_t* w1, const uint8_t* s1, const uint8_t* w2,
                           const uint8_t* s2, const uint8_t* w3, const uint8_t* s3, int inter, float limit, float* xq,
                           float* shared, float* routed) {
    if (!moe_copied[0])
        for (cudaEvent_t& event : moe_copied) cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    const int head = 2 * topk + dim;
    int status = sclm_gpu_gate(weight, bias, x, pack, n, topk, dim, temp, route_scale, norm);
    if (status) return status;
    cudaMemcpyAsync(host, pack, head * sizeof(float), cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(moe_copied[0], 0);
    if (g_weight) {
        status = sclm_gpu_gate_guess(g_weight, g_bias, h, pre, norm_w, pack + head, n, k, dim, hc, g_temp, eps);
        if (status) return status;
        cudaMemcpyAsync(host + head, pack + head, 2 * k * sizeof(float), cudaMemcpyDeviceToHost, 0);
        cudaEventRecord(moe_copied[1], 0);
    }
    cudaMemsetAsync(routed, 0, dim * sizeof(float), 0);
    act_quant_k<float><<<dim / 32, 32>>>(x, xq);
    status = sclm_gpu_fp8_expert(w1, s1, w2, s2, w3, s3, dim, inter, limit, xq, shared, 1);
    if (status) return status;
    return cudaEventSynchronize(moe_copied[0]);
}

// Waits until the guess of the last sclm_gpu_moe_begin is on the host.
API int sclm_gpu_moe_guess() { return cudaEventSynchronize(moe_copied[1]); }

// The end of the two-tier MoE: out[n] (bf16) = (routed[n] + ram[n]) + shared[n], ram being the CPU tier's sum in
// (page-locked) host memory, uploaded into the device buffer `staged` without waiting.
API int sclm_gpu_moe_end(const float* routed, const float* ram, float* staged, const float* shared, uint16_t* out, int n) {
    cudaMemcpyAsync(staged, ram, n * sizeof(float), cudaMemcpyHostToDevice, 0);
    moe_sum_k<<<(n + 255) / 256, 256>>>(routed, staged, shared, out, n);
    return cudaGetLastError();
}
