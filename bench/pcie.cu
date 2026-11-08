// PCIe bandwidth, CPU-GPU sync latency, mapped-memory ping-pong, VRAM bandwidth and pinned-memory limit for Phase 0.
#include <cuda_runtime.h>
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("%s failed: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }

__global__ void empty_kernel() {}

// GPU waits for the host to bump `req`, then echoes it to `ack`; runs `iters` round trips.
__global__ void pingpong(volatile int* req, volatile int* ack, int iters) {
    for (int i = 1; i <= iters; ++i) {
        while (*req != i) {}
        *ack = i;
        __threadfence_system();
    }
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s\n", prop.name);
    size_t freeB, totalB;
    CK(cudaMemGetInfo(&freeB, &totalB));
    std::printf("vram free %.2f / total %.2f GiB\n", freeB / 1073741824.0, totalB / 1073741824.0);

    const size_t big = 512ull << 20;
    void *h, *d, *d2;
    CK(cudaHostAlloc(&h, big, cudaHostAllocDefault));
    CK(cudaMalloc(&d, big));
    CK(cudaMalloc(&d2, big));
    memset(h, 1, big);
    cudaStream_t s;
    CK(cudaStreamCreate(&s));

    const size_t sizes[] = {64ull << 10, 1ull << 20, 8ull << 20, 64ull << 20, 512ull << 20};
    for (size_t sz : sizes) {
        int reps = int(std::max<size_t>(4, (2ull << 30) / sz));
        for (int dir = 0; dir < 2; ++dir) {
            CK(cudaStreamSynchronize(s));
            auto t0 = clk::now();
            for (int i = 0; i < reps; ++i)
                CK(dir ? cudaMemcpyAsync(h, d, sz, cudaMemcpyDeviceToHost, s) : cudaMemcpyAsync(d, h, sz, cudaMemcpyHostToDevice, s));
            CK(cudaStreamSynchronize(s));
            std::printf("pinned %s %8zu KiB : %6.1f GB/s\n", dir ? "D2H" : "H2D", sz >> 10, double(sz) * reps / since(t0) / 1e9);
        }
    }

    std::vector<char> pageable(big, 1);
    auto t0 = clk::now();
    for (int i = 0; i < 4; ++i) CK(cudaMemcpy(d, pageable.data(), big, cudaMemcpyHostToDevice));
    std::printf("pageable H2D 512 MiB : %6.1f GB/s\n", double(big) * 4 / since(t0) / 1e9);

    CK(cudaMemcpy(d2, d, big, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    t0 = clk::now();
    for (int i = 0; i < 10; ++i) CK(cudaMemcpyAsync(d2, d, big, cudaMemcpyDeviceToDevice, s));
    CK(cudaStreamSynchronize(s));
    std::printf("vram D2D copy        : %6.1f GB/s (read+write counted)\n", 2.0 * big * 10 / since(t0) / 1e9);

    const int n = 5000;
    t0 = clk::now();
    for (int i = 0; i < n; ++i) { empty_kernel<<<1, 1, 0, s>>>(); CK(cudaStreamSynchronize(s)); }
    std::printf("launch+sync latency  : %6.1f us\n", since(t0) / n * 1e6);

    t0 = clk::now();
    for (int i = 0; i < n; ++i) {
        CK(cudaMemcpyAsync(d, h, 40 << 10, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(h, d, 40 << 10, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
    }
    std::printf("40KiB H2D+D2H+sync   : %6.1f us\n", since(t0) / n * 1e6);

    int *hreq, *hack, *dreq, *dack;
    CK(cudaHostAlloc(&hreq, 64, cudaHostAllocMapped));
    CK(cudaHostAlloc(&hack, 64, cudaHostAllocMapped));
    *hreq = 0; *hack = 0;
    CK(cudaHostGetDevicePointer(&dreq, hreq, 0));
    CK(cudaHostGetDevicePointer(&dack, hack, 0));
    pingpong<<<1, 1, 0, s>>>(dreq, dack, n);
    t0 = clk::now();
    for (int i = 1; i <= n; ++i) {
        *(volatile int*)hreq = i;
        while (*(volatile int*)hack != i) {}
    }
    std::printf("mapped ping-pong RTT : %6.1f us\n", since(t0) / n * 1e6);
    CK(cudaStreamSynchronize(s));

    CK(cudaFreeHost(h));
    MEMORYSTATUSEX ms{sizeof(ms)};
    GlobalMemoryStatusEx(&ms);
    double availG = ms.ullAvailPhys / 1073741824.0;
    // Leave at least 16 GiB for the OS and other apps; never pin more than 64 GiB.
    int capChunks = int(std::min(64.0, availG - 16.0) / 4);
    std::printf("ram avail %.1f GiB, pinned-limit probe up to %d GiB\n", availG, capChunks * 4);
    std::vector<void*> chunks;
    for (int i = 0; i < capChunks; ++i) {
        void* p;
        if (cudaHostAlloc(&p, 4ull << 30, cudaHostAllocDefault) != cudaSuccess) { cudaGetLastError(); break; }
        chunks.push_back(p);
    }
    std::printf("pinned allocated     : %zu GiB\n", chunks.size() * 4);
    for (void* p : chunks) cudaFreeHost(p);
    return 0;
}
