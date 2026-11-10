// AVX-512 VBMI table-lookup GEMV benchmark for vector-quantized RAM-tier experts (256 codes per 8 or 4 columns).
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#include <malloc.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

// A matrix [N, K] with one code byte per D columns is N/256 row tiles; a tile is K/D steps of 256 bytes, byte p of
// step j being row p's code for columns D*j .. D*j+D-1. Per token, tables[j][c] = <codebook[c], x[D*j ..]> as int8 on
// one common scale, so a row's product is the sum of one table entry per step.
constexpr int TILE = 256, CODES = 256;

struct Tables {
    int groups = 0;
    float scale = 0;
    int8_t* t = nullptr;  // [groups][256]
    float* tmp = nullptr;
};

// cb is [D][256]: dimension-major, so one input value scales a whole 256-entry column.
static void build_tables(const float* cb, int d, const float* x, int k, Tables& tb) {
    const int g = k / d;
    __m512 mx = _mm512_setzero_ps();
    for (int j = 0; j < g; ++j)
        for (int v = 0; v < CODES; v += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int i = 0; i < d; ++i)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(x[j * d + i]), _mm512_load_ps(cb + i * CODES + v), acc);
            _mm512_store_ps(tb.tmp + j * CODES + v, acc);
            mx = _mm512_max_ps(mx, _mm512_abs_ps(acc));
        }
    tb.groups = g, tb.scale = _mm512_reduce_max_ps(mx) / 127.0f;
    const __m512 inv = _mm512_set1_ps(1.0f / tb.scale);
    for (int n = 0; n < g * CODES; n += 16)
        _mm_store_si128((__m128i*)(tb.t + n), _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_load_ps(tb.tmp + n), inv))));
}

// y[0..255] = table sums of one tile, times the table scale (row scales are applied by the caller).
static void gemv_tile(const uint8_t* codes, const Tables& tb, float* y) {
    __m512i a[8], s[16];
    for (auto& v : a) v = _mm512_setzero_si512();
    for (auto& v : s) v = _mm512_setzero_si512();
    for (int j = 0; j < tb.groups; ++j) {
        const __m512i* t = (const __m512i*)(tb.t + j * CODES);
        const __m512i t0 = _mm512_load_si512(t), t1 = _mm512_load_si512(t + 1), t2 = _mm512_load_si512(t + 2), t3 = _mm512_load_si512(t + 3);
        for (int r = 0; r < 4; ++r) {
            const __m512i c = _mm512_load_si512(codes + j * TILE + 64 * r);
            const __m512i v = _mm512_mask_blend_epi8(_mm512_movepi8_mask(c), _mm512_permutex2var_epi8(t0, c, t1),
                                                     _mm512_permutex2var_epi8(t2, c, t3));
            a[2 * r] = _mm512_add_epi16(a[2 * r], _mm512_cvtepi8_epi16(_mm512_castsi512_si256(v)));
            a[2 * r + 1] = _mm512_add_epi16(a[2 * r + 1], _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(v, 1)));
        }
        if ((j & 255) == 255 || j == tb.groups - 1)  // 256 entries of at most 127 fit int16
            for (int r = 0; r < 8; ++r) {
                s[2 * r] = _mm512_add_epi32(s[2 * r], _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a[r])));
                s[2 * r + 1] = _mm512_add_epi32(s[2 * r + 1], _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a[r], 1)));
                a[r] = _mm512_setzero_si512();
            }
    }
    const __m512 sc = _mm512_set1_ps(tb.scale);
    for (int r = 0; r < 16; ++r) _mm512_storeu_ps(y + 16 * r, _mm512_mul_ps(_mm512_cvtepi32_ps(s[r]), sc));
}

static void pin(int t, int threads) { SetThreadAffinityMask(GetCurrentThread(), 1ull << (threads <= 8 ? 2 * t : t)); }

int main(int argc, char** argv) {
    const int d = argc > 1 ? atoi(argv[1]) : 8, threads = argc > 2 ? atoi(argv[2]) : 8;
    const double pool_gb = argc > 3 ? atof(argv[3]) : 4.0;
    const int N = 2304, K = 5120, groups = K / d;
    const size_t mat = size_t(N) * groups;  // code bytes of w1 or w3; w2 [5120, 2304] has as many
    std::mt19937_64 rng(1);
    std::normal_distribution<float> nd;
    auto* cb = (float*)_aligned_malloc(sizeof(float) * d * CODES, 64);
    auto* x = (float*)_aligned_malloc(sizeof(float) * K, 64);
    for (int i = 0; i < d * CODES; ++i) cb[i] = nd(rng);
    for (int i = 0; i < K; ++i) x[i] = nd(rng);
    Tables tb;
    tb.t = (int8_t*)_aligned_malloc(size_t(groups) * CODES, 64), tb.tmp = (float*)_aligned_malloc(sizeof(float) * groups * CODES, 64);

    const size_t n_mat = std::max<size_t>(threads, size_t(pool_gb * 1e9 / mat));
    auto* pool = (uint8_t*)VirtualAlloc(nullptr, n_mat * mat, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!pool) return printf("VirtualAlloc failed\n"), 1;
    {
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t] {
                std::mt19937_64 r(100 + t);
                for (size_t i = t * (n_mat * mat / 8 / threads); i < (t + 1) * (n_mat * mat / 8 / threads); ++i) ((uint64_t*)pool)[i] = r();
            });
        for (auto& t : ts) t.join();
    }

    pin(0, threads);
    auto t0 = clk::now();
    for (int i = 0; i < 200; ++i) build_tables(cb, d, x, K, tb);
    printf("d=%d: %d groups, tables %.0f KB, build %.1f us\n", d, groups, groups * CODES / 1e3, 1e6 * since(t0) / 200);

    // correctness: float reference of the first tile
    alignas(64) float y[TILE];
    gemv_tile(pool, tb, y);
    double err = 0, ref2 = 0;
    for (int r = 0; r < TILE; ++r) {
        double ref = 0;
        for (int j = 0; j < groups; ++j)
            for (int i = 0; i < d; ++i) ref += double(cb[i * CODES + pool[j * TILE + r]]) * x[j * d + i];
        err += (y[r] - ref) * (y[r] - ref), ref2 += ref * ref;
    }
    printf("  int8-table output error vs float: %.2e (relative energy)\n", err / ref2);

    // one thread, one matrix in L3: the compute ceiling
    double chk = 0;
    t0 = clk::now();
    const int reps = 300;
    for (int i = 0; i < reps; ++i)
        for (int tile = 0; tile < N / TILE; ++tile) gemv_tile(pool + size_t(tile) * groups * TILE, tb, y), chk += y[i & 255];
    double s = since(t0);
    printf("  1 thread, L3: %.1f GB/s of codes, %.0f Gweight/s, %.1f us per matrix\n", reps * mat / s / 1e9,
           reps * mat * double(d) / s / 1e9, 1e6 * s / reps);

    // all threads over the pool in random order: RAM-bound
    const size_t per_thread = n_mat / threads;
    std::vector<double> chks(threads);
    t0 = clk::now();
    {
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t] {
                pin(t, threads);
                std::vector<size_t> order(per_thread);
                for (size_t i = 0; i < per_thread; ++i) order[i] = t * per_thread + i;
                std::shuffle(order.begin(), order.end(), std::mt19937_64(7 + t));
                alignas(64) float yt[TILE];
                for (size_t m : order)
                    for (int tile = 0; tile < N / TILE; ++tile) gemv_tile(pool + m * mat + size_t(tile) * groups * TILE, tb, yt), chks[t] += yt[0];
            });
        for (auto& t : ts) t.join();
    }
    s = since(t0);
    const double bytes = double(per_thread) * threads * mat;
    for (double c : chks) chk += c;
    printf("  %d threads, %.1f GB pool: %.1f GB/s of codes, %.0f Gweight/s, %.1f us per expert (3 matrices) (checksum %.3g)\n",
           threads, n_mat * mat / 1e9, bytes / s / 1e9, bytes * d / s / 1e9, 1e6 * s * 3 * mat / bytes, chk);
    VirtualFree(pool, 0, MEM_RELEASE);
    return 0;
}
