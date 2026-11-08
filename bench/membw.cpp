// Multi-threaded DRAM and L3 bandwidth benchmark (read / non-temporal write) for Phase 0.
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

static __m512i read_chunk(const __m512i* p, size_t n, int reps) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    for (int r = 0; r < reps; ++r)
        for (size_t i = 0; i < n; i += 4) {
            a0 = _mm512_add_epi64(a0, _mm512_load_si512(p + i));
            a1 = _mm512_add_epi64(a1, _mm512_load_si512(p + i + 1));
            a2 = _mm512_add_epi64(a2, _mm512_load_si512(p + i + 2));
            a3 = _mm512_add_epi64(a3, _mm512_load_si512(p + i + 3));
        }
    return _mm512_add_epi64(_mm512_add_epi64(a0, a1), _mm512_add_epi64(a2, a3));
}

static void write_chunk(__m512i* p, size_t n, int reps) {
    __m512i v = _mm512_set1_epi64(1);
    for (int r = 0; r < reps; ++r)
        for (size_t i = 0; i < n; ++i) _mm512_stream_si512(p + i, v);
    _mm_sfence();
}

// Runs one pass with `threads` workers, each on its own slice; returns GB/s.
static double run(__m512i* buf, size_t bytes, int threads, int reps, bool write) {
    size_t per = bytes / sizeof(__m512i) / threads / 4 * 4;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> ts;
    std::vector<__m512i> sink(threads);
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([&, t] {
            // Spread across physical cores first (logical CPUs 0,2,4,... are distinct cores).
            int cpu = threads <= 8 ? t * 2 : t;
            SetThreadAffinityMask(GetCurrentThread(), 1ull << cpu);
            ready++;
            while (!go.load()) _mm_pause();
            __m512i* p = buf + per * t;
            if (write) write_chunk(p, per, reps);
            else sink[t] = read_chunk(p, per, reps);
        });
    while (ready.load() < threads) _mm_pause();
    auto t0 = clk::now();
    go = true;
    for (auto& th : ts) th.join();
    double s = std::chrono::duration<double>(clk::now() - t0).count();
    volatile long long keep = _mm512_reduce_add_epi64(sink[0]);
    (void)keep;
    return double(per) * sizeof(__m512i) * threads * reps / s / 1e9;
}

int main() {
    const size_t dram = 4ull << 30, l3 = 32ull << 20;
    auto* buf = (__m512i*)VirtualAlloc(nullptr, dram, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buf) { std::printf("alloc failed\n"); return 1; }
    memset(buf, 1, dram);  // fault in all pages before timing
    const int counts[] = {1, 2, 4, 8, 16};
    std::printf("%-6s %-8s %10s\n", "test", "threads", "GB/s(best of 5)");
    for (int write = 0; write < 2; ++write)
        for (int th : counts) {
            double best = 0;
            for (int i = 0; i < 5; ++i) best = std::max(best, run(buf, dram, th, 1, write));
            std::printf("%-6s %-8d %10.1f\n", write ? "dramW" : "dramR", th, best);
        }
    for (int th : counts) {
        double best = 0;
        for (int i = 0; i < 5; ++i) best = std::max(best, run(buf, l3, th, 200, false));
        std::printf("%-6s %-8d %10.1f\n", "l3R", th, best);
    }
    VirtualFree(buf, 0, MEM_RELEASE);
    return 0;
}
