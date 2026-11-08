// Unbuffered SSD read benchmark (sequential large blocks, random small blocks at several queue depths) for Phase 0.
#define NOMINMAX
#include <windows.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;

// Each of `qd` threads issues synchronous unbuffered reads of `block` bytes; prints MB/s and IOPS.
static void bench(const wchar_t* path, unsigned long long fileSize, DWORD block, int qd, bool random, double seconds, const char* label) {
    std::atomic<unsigned long long> ops{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> ts;
    for (int t = 0; t < qd; ++t)
        ts.emplace_back([&, t] {
            HANDLE f = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
            void* buf = VirtualAlloc(nullptr, block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            std::mt19937_64 rng(t * 7919 + 1);
            unsigned long long blocks = fileSize / block, idx = blocks / qd * t;
            while (!stop.load(std::memory_order_relaxed)) {
                unsigned long long b = random ? rng() % blocks : (idx++ % blocks);
                LARGE_INTEGER off; off.QuadPart = b * block;
                SetFilePointerEx(f, off, nullptr, FILE_BEGIN);
                DWORD got;
                ReadFile(f, buf, block, &got, nullptr);
                ops++;
            }
            VirtualFree(buf, 0, MEM_RELEASE);
            CloseHandle(f);
        });
    auto t0 = clk::now();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (auto& th : ts) th.join();
    double s = std::chrono::duration<double>(clk::now() - t0).count();
    std::printf("%-22s qd=%-3d : %8.0f MB/s %10.0f IOPS\n", label, qd, ops * double(block) / s / 1e6, ops / s);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: ssdbench <test-file-path>\n"); return 1; }
    wchar_t path[MAX_PATH];
    MultiByteToWideChar(CP_UTF8, 0, argv[1], -1, path, MAX_PATH);
    const unsigned long long size = 4ull << 30;
    const DWORD chunk = 64 << 20;

    HANDLE f = CreateFileW(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (f == INVALID_HANDLE_VALUE) { std::printf("create failed\n"); return 1; }
    auto* buf = (unsigned char*)VirtualAlloc(nullptr, chunk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    std::mt19937 rng(42);
    for (DWORD i = 0; i < chunk; ++i) buf[i] = (unsigned char)rng();  // incompressible data
    for (unsigned long long w = 0; w < size; w += chunk) { DWORD put; WriteFile(f, buf, chunk, &put, nullptr); }
    CloseHandle(f);
    VirtualFree(buf, 0, MEM_RELEASE);

    bench(path, size, 4 << 20, 4, false, 5, "seq 4MiB");
    bench(path, size, 4 << 20, 16, true, 5, "rand 4MiB");
    bench(path, size, 4096, 1, true, 5, "rand 4KiB");
    bench(path, size, 4096, 32, true, 5, "rand 4KiB");
    bench(path, size, 16384, 32, true, 5, "rand 16KiB");

    DeleteFileW(path);
    return 0;
}
