// SSD tier: reads original FP4 experts from the checkpoint shards with unbuffered overlapped I/O, and decides which
// of them the VRAM cache holds (LRFU over every expert's full history, with bypass; scripts/cache_policy.py).
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

namespace {

const uint64_t PAGE = 4096;  // a multiple of the sector size: unbuffered reads start and end on it

struct Request {
    OVERLAPPED ov;
    DWORD need;  // bytes that have to arrive (a read reaching past the end of the file comes back short)
};

}  // namespace

struct SclmReader {
    std::vector<HANDLE> files;
    HANDLE port = nullptr;
    uint64_t chunk = 0;
    std::deque<Request> requests;  // of the reads submitted since the last wait; a deque keeps their addresses
    int pending = 0, failed = 0;
};

struct SclmCache {
    int slots;
    double lam;
    std::vector<double> score;        // key -> log(sum over its uses of exp(lam * token)); exp(-lam * now) is common to all
    std::vector<int> slot_of, key_in; // key -> slot or -1, slot -> key
    std::vector<long long> call_of;   // slot -> the call that last used it
    long long call = 0;
    int filled = 0;
};

extern "C" {

// chunk (a multiple of 4096) is the largest single read. Null if a file cannot be opened.
__declspec(dllexport) SclmReader* sclm_reader_new(const wchar_t* const* paths, int n_files, int chunk) {
    auto* R = new SclmReader;
    R->chunk = chunk;
    R->port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 1);
    for (int f = 0; f < n_files; ++f) {
        const HANDLE h = CreateFileW(paths[f], GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                     FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h != INVALID_HANDLE_VALUE) R->files.push_back(h);
        if (h == INVALID_HANDLE_VALUE || !CreateIoCompletionPort(h, R->port, 0, 0)) {
            for (HANDLE o : R->files) CloseHandle(o);
            CloseHandle(R->port);
            delete R;
            return nullptr;
        }
    }
    return R;
}

__declspec(dllexport) void sclm_reader_free(SclmReader* R) {
    for (HANDLE h : R->files) CloseHandle(h);
    CloseHandle(R->port);
    delete R;
}

// Starts reading bytes [offset, offset + bytes) of file `file` into dst and returns where in dst the first of them
// lands (the read starts on the page boundary below offset). dst is page-aligned and holds bytes + 2 pages.
__declspec(dllexport) int sclm_reader_submit(SclmReader* R, int file, uint64_t offset, uint64_t bytes, uint8_t* dst) {
    const uint64_t head = offset % PAGE, begin = offset - head, end = (offset + bytes + PAGE - 1) / PAGE * PAGE;
    for (uint64_t at = begin; at < end; at += R->chunk) {
        const uint64_t len = std::min(R->chunk, end - at);
        Request& q = R->requests.emplace_back();
        q.ov = {};
        q.ov.Offset = DWORD(at), q.ov.OffsetHigh = DWORD(at >> 32);
        q.need = DWORD(std::min(len, offset + bytes - at));
        ++R->pending;
        if (!ReadFile(R->files[file], dst + (at - begin), DWORD(len), nullptr, &q.ov) && GetLastError() != ERROR_IO_PENDING)
            --R->pending, ++R->failed;  // no completion is queued for a read that failed to start
    }
    return int(head);
}

// Waits for every read submitted since the last wait. Returns how many of them failed.
__declspec(dllexport) int sclm_reader_wait(SclmReader* R) {
    while (R->pending) {
        DWORD got;
        ULONG_PTR key;
        OVERLAPPED* ov;
        const BOOL ok = GetQueuedCompletionStatus(R->port, &got, &key, &ov, INFINITE);
        if (!ov) return -1;  // the port itself failed
        --R->pending;
        if (!ok || got < CONTAINING_RECORD(ov, Request, ov)->need) ++R->failed;
    }
    const int failed = R->failed;
    R->requests.clear(), R->failed = 0;
    return failed;
}

// A cache of `slots` experts out of n_keys; lam is the score's decay per token.
__declspec(dllexport) SclmCache* sclm_cache_new(int n_keys, int slots, double lam) {
    auto* C = new SclmCache{slots, lam};
    C->score.assign(n_keys, -std::numeric_limits<double>::infinity());
    C->slot_of.assign(n_keys, -1), C->key_in.assign(slots, -1), C->call_of.assign(slots, 0);
    return C;
}

__declspec(dllexport) void sclm_cache_free(SclmCache* C) { delete C; }

// One layer's SSD-tier experts at token t: slot[i] is the VRAM slot keys[i] is computed from and miss[i] says
// whether the expert has to be read into it first. Slots >= `slots` are scratch: the expert scores below everything
// cached and is computed without being kept (bypass). A slot used by this call is not evicted by this call.
__declspec(dllexport) void sclm_cache_use(SclmCache* C, const int* keys, int n, long long t, int* slot, int* miss) {
    ++C->call;
    for (int i = 0; i < n; ++i) {
        const int k = keys[i];
        double a = C->lam * double(t);
        if (const double b = C->score[k]; std::isfinite(b)) a = std::max(a, b) + std::log1p(std::exp(-std::fabs(a - b)));
        C->score[k] = a;
        int s = C->slot_of[k];
        miss[i] = s < 0;
        if (s < 0) {
            if (C->filled < C->slots) {
                s = C->filled++;
            } else {
                int v = -1;  // the lowest (score, key) among the slots this call does not use
                for (int j = 0; j < C->slots; ++j) {
                    if (C->call_of[j] == C->call) continue;
                    const int kj = C->key_in[j], kv = v < 0 ? -1 : C->key_in[v];
                    if (v < 0 || C->score[kj] < C->score[kv] || (C->score[kj] == C->score[kv] && kj < kv)) v = j;
                }
                if (v < 0 || a <= C->score[C->key_in[v]]) {
                    slot[i] = C->slots + i;
                    continue;
                }
                C->slot_of[C->key_in[v]] = -1, s = v;
            }
            C->slot_of[k] = s, C->key_in[s] = k;
        }
        C->call_of[s] = C->call, slot[i] = s;
    }
}

}  // extern "C"
