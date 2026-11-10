// CPU engine for the RAM tier: a layer's vector-coded SwiGLU experts, computed step for step as the reference computes
// them (quant.VQLinear inside the upstream Expert), except that the GEMVs sum int8 tables instead of fp32 products.
#include <malloc.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "vq_kernel.h"

extern "C" {
struct SclmFormat {
    const float* codebook;  // [d][1 << bits]
    int d, bits;
};
struct SclmExpert {
    int format;
    const uint8_t* codes[3];    // w1, w2, w3 in vq_kernel.h's layout
    const uint16_t* scales[3];  // fp16, one per row
};
}

namespace {

template <class T>
T* aligned(size_t n) { return (T*)_aligned_malloc(n * sizeof(T), 64); }

struct Rotation {
    std::vector<float> signs;
    int block = 0;
};

struct Format {
    float* cb;
    int d;
    Tables in, mid;  // over the layer input (w1, w3) and over an expert's hidden vector (w2)
};

// The reference's activation fake-quant before a quantized GEMM (kernel.act_quant: blocks of 32, power-of-2 scale):
// a block is scaled so its largest magnitude is at most 448, then rounded to FP8 E4M3.
void fp8_blocks(float* x, int n) {
    const __m512 zero = _mm512_setzero_ps();
    for (int b = 0; b < n; b += 32) {
        const __m512 mx = _mm512_max_ps(_mm512_abs_ps(_mm512_loadu_ps(x + b)), _mm512_abs_ps(_mm512_loadu_ps(x + b + 16)));
        int e;
        const float m = std::frexp(std::max(_mm512_reduce_max_ps(mx), 1e-4f) * float(1.0 / 448.0), &e);
        const __m512 se = _mm512_set1_ps(float(m == 0.5f ? e - 1 : e));  // the scale is 2^ceil(log2(amax / 448))
        for (int i = b; i < b + 32; i += 16) {
            const __m512 v = _mm512_scalef_ps(_mm512_loadu_ps(x + i), _mm512_sub_ps(zero, se));
            // E4M3 keeps 4 significant bits down to 2^-6 and steps of 2^-9 below that
            const __m512 q = _mm512_sub_ps(_mm512_max_ps(_mm512_getexp_ps(v), _mm512_set1_ps(-6.0f)), _mm512_set1_ps(3.0f));
            const __m512 r = _mm512_roundscale_ps(_mm512_scalef_ps(v, _mm512_sub_ps(zero, q)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_ps(x + i, _mm512_scalef_ps(r, _mm512_add_ps(q, se)));
        }
    }
}

// Rounds to bfloat16 (nearest even) where the reference casts to it.
void bf16(float* x, int n) {
    for (int i = 0; i < n; i += 16) {
        const __m512i u = _mm512_castps_si512(_mm512_loadu_ps(x + i));
        const __m512i odd = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
        const __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), odd));
        _mm512_storeu_ps(x + i, _mm512_castsi512_ps(_mm512_and_si512(r, _mm512_set1_epi32(0xffff0000))));
    }
}

// x <- H (signs * x) / sqrt(block) over each block (quant.BlockHadamard.rotate).
void rotate(float* x, int n, const Rotation& rot) {
    for (int i = 0; i < n; ++i) x[i] *= rot.signs[i];
    const float norm = float(std::sqrt(double(rot.block)));
    for (float* p = x; p < x + n; p += rot.block) {
        for (int h = 1; h < rot.block; h *= 2)
            for (int i = 0; i < rot.block; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    const float a = p[j], b = p[j + h];
                    p[j] = a + b, p[j + h] = a - b;
                }
        for (int i = 0; i < rot.block; ++i) p[i] /= norm;
    }
}

// out[0..rows) = the matrix's rows times the input the tables were built from.
void matvec(const uint8_t* codes, const uint16_t* scales, const Tables& tb, int rows, float* out) {
    const size_t tile_bytes = size_t(tb.groups) * (tb.codes == 256 ? TILE : 192);
    for (int t = 0; t < rows / TILE; ++t) gemv_tile(codes + t * tile_bytes, tb, out + t * TILE);
    for (int i = 0; i < rows; i += 16) {
        const __m512 s = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(scales + i)));
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(out + i), s));
    }
}

Tables tables(int groups, int codes) {
    Tables tb;
    tb.codes = codes, tb.t = aligned<int8_t>(size_t(groups) * codes), tb.tmp = aligned<float>(size_t(groups) * codes);
    return tb;
}

}  // namespace

struct SclmLayer {
    int dim, inter;
    float limit;
    std::vector<Format> formats;
    Rotation rot_dim, rot_inter;
    std::vector<char> built;
    std::vector<float> x, gate, up, out;
};

extern "C" {

// dim and inter are multiples of 256. The codebooks and signs are copied; the experts' codes and scales are not.
__declspec(dllexport) SclmLayer* sclm_layer_new(int dim, int inter, float swiglu_limit, const SclmFormat* formats, int n_formats,
                                                const int8_t* signs_dim, int block_dim, const int8_t* signs_inter, int block_inter) {
    auto* L = new SclmLayer{dim, inter, swiglu_limit};
    for (int f = 0; f < n_formats; ++f) {
        const int d = formats[f].d, codes = 1 << formats[f].bits;
        float* cb = aligned<float>(size_t(d) * codes);
        std::memcpy(cb, formats[f].codebook, sizeof(float) * d * codes);
        L->formats.push_back({cb, d, tables(dim / d, codes), tables(inter / d, codes)});
    }
    L->rot_dim = {std::vector<float>(signs_dim, signs_dim + dim), block_dim};
    L->rot_inter = {std::vector<float>(signs_inter, signs_inter + inter), block_inter};
    L->built.resize(n_formats), L->x.resize(dim), L->gate.resize(inter), L->up.resize(inter), L->out.resize(dim);
    return L;
}

__declspec(dllexport) void sclm_layer_free(SclmLayer* L) {
    for (Format& f : L->formats)
        for (void* p : {(void*)f.cb, (void*)f.in.t, (void*)f.in.tmp, (void*)f.mid.t, (void*)f.mid.tmp}) _aligned_free(p);
    delete L;
}

// y[dim] = sum of gates[e] * expert e of x[dim] over the n experts (MoE.forward's routed sum for one token).
__declspec(dllexport) void sclm_layer_forward(SclmLayer* L, const SclmExpert* experts, const float* gates, int n, const float* x, float* y) {
    const int dim = L->dim, inter = L->inter;
    float *xr = L->x.data(), *gate = L->gate.data(), *up = L->up.data(), *out = L->out.data();
    std::memcpy(xr, x, sizeof(float) * dim);
    fp8_blocks(xr, dim);
    rotate(xr, dim, L->rot_dim);
    std::fill(L->built.begin(), L->built.end(), 0);
    std::memset(y, 0, sizeof(float) * dim);
    for (int e = 0; e < n; ++e) {
        const SclmExpert& ex = experts[e];
        Format& f = L->formats[ex.format];
        if (!L->built[ex.format])  // w1 and w3 of every expert of a format read the same tables
            build_tables(f.cb, f.d, xr, dim, f.in), L->built[ex.format] = 1;
        matvec(ex.codes[0], ex.scales[0], f.in, inter, gate);
        matvec(ex.codes[2], ex.scales[2], f.in, inter, up);
        bf16(gate, inter), bf16(up, inter);
        for (int i = 0; i < inter; ++i) {
            const float g = L->limit > 0 ? std::min(gate[i], L->limit) : gate[i];
            const float u = L->limit > 0 ? std::clamp(up[i], -L->limit, L->limit) : up[i];
            gate[i] = gates[e] * (g / (1.0f + std::exp(-g)) * u);
        }
        bf16(gate, inter);
        fp8_blocks(gate, inter);
        rotate(gate, inter, L->rot_inter);
        build_tables(f.cb, f.d, gate, inter, f.mid);
        matvec(ex.codes[1], ex.scales[1], f.mid, dim, out);
        bf16(out, dim);
        for (int i = 0; i < dim; ++i) y[i] += out[i];
    }
}

}  // extern "C"
