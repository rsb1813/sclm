// AVX-512 VBMI table-lookup GEMV benchmark for vector-quantized RAM-tier experts (256 codes per 16, 8 or 4 columns, or 64 per 2).
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

#include "../engine/vq_kernel.h"

using clk = std::chrono::steady_clock;
static double since(clk::time_point t0) { return std::chrono::duration<double>(clk::now() - t0).count(); }

static void pin(int t, int threads) { SetThreadAffinityMask(GetCurrentThread(), 1ull << (threads <= 8 ? 2 * t : t)); }

int main(int argc, char** argv) {
    const int d = argc > 1 ? atoi(argv[1]) : 8, threads = argc > 2 ? atoi(argv[2]) : 8;
    const double pool_gb = argc > 3 ? atof(argv[3]) : 4.0;
    const int N = 2304, K = 5120, groups = K / d, codes = d == 2 ? 64 : 256, step = codes == 256 ? TILE : 192;
    const size_t mat = size_t(N / TILE) * groups * step;  // code bytes of w1 or w3; w2 [5120, 2304] has as many
    std::mt19937_64 rng(1);
    std::normal_distribution<float> nd;
    auto* cb = (float*)_aligned_malloc(sizeof(float) * d * codes, 64);
    auto* x = (float*)_aligned_malloc(sizeof(float) * K, 64);
    for (int i = 0; i < d * codes; ++i) cb[i] = nd(rng);
    for (int i = 0; i < K; ++i) x[i] = nd(rng);
    Tables tb;
    tb.codes = codes;
    tb.t = (int8_t*)_aligned_malloc(size_t(groups) * codes, 64);

    const size_t n_mat = std::max<size_t>(threads, size_t(pool_gb * 1e9 / mat));
    auto* pool = (uint8_t*)VirtualAlloc(nullptr, n_mat * mat + 64, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);  // 6-bit loads read 16 bytes on
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
    for (int i = 0; i < 200; ++i) build_tables(cb, d, x, 2304, tb);  // w2's input: per expert and token
    const double build_w2 = 1e6 * since(t0) / 200;
    t0 = clk::now();
    for (int i = 0; i < 200; ++i) build_tables(cb, d, x, K, tb);  // w1's and w3's input: per layer and token
    printf("d=%d, %d codes: %d groups, tables %.0f KB, build %.1f us (w2's 2304 inputs: %.1f us)\n", d, codes, groups,
           groups * codes / 1e3, 1e6 * since(t0) / 200, build_w2);

    // correctness: float reference of the first tile
    alignas(64) float y[TILE];
    gemv_tile(pool, tb, y);
    double err = 0, ref2 = 0;
    for (int r = 0; r < TILE; ++r) {
        double ref = 0;
        for (int j = 0; j < groups; ++j) {
            int c;
            if (codes == 64) {
                const uint8_t* p = pool + j * 192 + 48 * (r / 64);
                const int bit = 6 * (r % 64);
                c = ((p[bit / 8] | p[bit / 8 + 1] << 8) >> (bit % 8)) & 63;
            } else {
                c = pool[j * TILE + r];
            }
            for (int i = 0; i < d; ++i) ref += double(cb[i * codes + c]) * x[j * d + i];
        }
        err += (y[r] - ref) * (y[r] - ref), ref2 += ref * ref;
    }
    printf("  int8-table output error vs float: %.2e (relative energy)\n", err / ref2);

    // one thread, one matrix in L3: the compute ceiling
    const double weights = double(d) * TILE / step;  // per code byte
    double chk = 0;
    t0 = clk::now();
    const int reps = 300;
    for (int i = 0; i < reps; ++i)
        for (int tile = 0; tile < N / TILE; ++tile) gemv_tile(pool + size_t(tile) * groups * step, tb, y), chk += y[i & 255];
    double s = since(t0);
    printf("  1 thread, L3: %.1f GB/s of codes, %.0f Gweight/s, %.1f us per matrix\n", reps * mat / s / 1e9,
           reps * mat * weights / s / 1e9, 1e6 * s / reps);

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
                    for (int tile = 0; tile < N / TILE; ++tile) gemv_tile(pool + m * mat + size_t(tile) * groups * step, tb, yt), chks[t] += yt[0];
            });
        for (auto& t : ts) t.join();
    }
    s = since(t0);
    const double bytes = double(per_thread) * threads * mat;
    for (double c : chks) chk += c;
    printf("  %d threads, %.1f GB pool: %.1f GB/s of codes, %.0f Gweight/s, %.1f us per expert (3 matrices) (checksum %.3g)\n",
           threads, n_mat * mat / 1e9, bytes / s / 1e9, bytes * weights / s / 1e9, 1e6 * s * 3 * mat / bytes, chk);
    VirtualFree(pool, 0, MEM_RELEASE);
    return 0;
}
