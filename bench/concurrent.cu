// Measures combined DRAM bandwidth when CPU AVX-512 reads and pinned H2D DMA run at the same time.
#define NOMINMAX
#include <cuda_runtime.h>
#include <windows.h>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const int cpuThreads = argc > 1 ? atoi(argv[1]) : 8;
    const double seconds = 5.0;
    const size_t cpuBytes = 4ull << 30, dmaBytes = 512ull << 20;

    auto* cbuf = (__m512i*)VirtualAlloc(nullptr, cpuBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    memset(cbuf, 1, cpuBytes);
    void *h, *d;
    cudaHostAlloc(&h, dmaBytes, cudaHostAllocDefault);
    cudaMalloc(&d, dmaBytes);
    memset(h, 1, dmaBytes);
    cudaStream_t s;
    cudaStreamCreate(&s);

    for (int mode = 0; mode < 3; ++mode) {  // 0: CPU only, 1: DMA only, 2: both
        std::atomic<bool> stop{false};
        std::atomic<unsigned long long> cpuRead{0};
        std::vector<std::thread> ts;
        if (mode != 1)
            for (int t = 0; t < cpuThreads; ++t)
                ts.emplace_back([&, t] {
                    SetThreadAffinityMask(GetCurrentThread(), 1ull << (t * 2 % 16));
                    size_t per = cpuBytes / sizeof(__m512i) / cpuThreads;
                    const __m512i* p = cbuf + per * t;
                    __m512i acc = _mm512_setzero_si512();
                    while (!stop.load(std::memory_order_relaxed)) {
                        for (size_t i = 0; i < per; ++i) acc = _mm512_add_epi64(acc, _mm512_load_si512(p + i));
                        cpuRead += per * sizeof(__m512i);
                    }
                    volatile long long keep = _mm512_reduce_add_epi64(acc);
                    (void)keep;
                });
        unsigned long long dmaDone = 0;
        auto t0 = clk::now();
        if (mode != 0)
            while (std::chrono::duration<double>(clk::now() - t0).count() < seconds) {
                cudaMemcpyAsync(d, h, dmaBytes, cudaMemcpyHostToDevice, s);
                cudaStreamSynchronize(s);
                dmaDone += dmaBytes;
            }
        else
            std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        stop = true;
        for (auto& th : ts) th.join();
        double sec = std::chrono::duration<double>(clk::now() - t0).count();
        double c = cpuRead / sec / 1e9, g = dmaDone / sec / 1e9;
        const char* names[] = {"cpu only", "dma only", "cpu + dma"};
        std::printf("%-10s cpu %5.1f GB/s  dma %5.1f GB/s  total %5.1f GB/s\n", names[mode], c, g, c + g);
    }
    return 0;
}
