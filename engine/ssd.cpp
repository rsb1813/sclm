// SSD tier: reads original FP4 experts from the checkpoint shards with unbuffered overlapped I/O, and decides which
// of them the VRAM cache holds (LRFU over every expert's full history, with bypass; scripts/cache_policy.py).
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace {

const uint64_t PAGE = 4096;  // a multiple of the sector size: unbuffered reads start and end on it

struct Request {
    OVERLAPPED ov;
    DWORD need;  // bytes that have to arrive (a read reaching past the end of the file comes back short)
    HANDLE file;
    void* dst;
    DWORD len;
    int ticket;  // the caller's name for the read this chunk belongs to
};

const int TICKETS = 256;

}  // namespace

// Starting a read costs about 0.3 ms per 18 MB (the buffer's pages are locked for the transfer), so a thread of its
// own issues them: the caller queues a request and goes on computing.
struct SclmReader {
    std::vector<HANDLE> files;
    HANDLE port = nullptr;
    uint64_t chunk = 0;
    std::deque<Request> requests;  // of the reads submitted since the last wait; a deque keeps their addresses
    int pending = 0, failed = 0;   // the caller's: reads not yet completed, reads that failed
    int open[TICKETS] = {};        // the caller's: chunks not yet completed, per ticket
    std::mutex mutex;
    std::condition_variable wake;
    std::vector<Request*> queue;  // submitted, not yet issued
    bool stop = false;
    std::thread issuer;

    void issue() {
        std::vector<Request*> batch;
        for (;;) {
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [this] { return stop || !queue.empty(); });
                if (stop) return;
                batch.swap(queue);
            }
            for (Request* q : batch)
                if (!ReadFile(q->file, q->dst, q->len, nullptr, &q->ov) && GetLastError() != ERROR_IO_PENDING)
                    PostQueuedCompletionStatus(port, 0, 1, &q->ov);  // key 1: the read did not start
            batch.clear();
        }
    }
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
    R->issuer = std::thread([R] { R->issue(); });
    return R;
}

__declspec(dllexport) void sclm_reader_free(SclmReader* R) {
    {
        std::lock_guard lock(R->mutex);
        R->stop = true;
    }
    R->wake.notify_one();
    R->issuer.join();
    for (HANDLE h : R->files) CloseHandle(h);
    CloseHandle(R->port);
    delete R;
}

// Starts reading bytes [offset, offset + bytes) of file `file` into dst and returns where in dst the first of them
// lands (the read starts on the page boundary below offset). dst is page-aligned and holds bytes + 2 pages.
// `ticket` (below 256) names the read for sclm_reader_open and sclm_reader_wait_ticket.
__declspec(dllexport) int sclm_reader_submit(SclmReader* R, int file, uint64_t offset, uint64_t bytes, uint8_t* dst, int ticket) {
    const uint64_t head = offset % PAGE, begin = offset - head, end = (offset + bytes + PAGE - 1) / PAGE * PAGE;
    const size_t first = R->requests.size();
    for (uint64_t at = begin; at < end; at += R->chunk) {
        const uint64_t len = std::min(R->chunk, end - at);
        Request& q = R->requests.emplace_back();
        q.ov = {};
        q.ov.Offset = DWORD(at), q.ov.OffsetHigh = DWORD(at >> 32);
        q.need = DWORD(std::min(len, offset + bytes - at));
        q.file = R->files[file], q.dst = dst + (at - begin), q.len = DWORD(len), q.ticket = ticket;
        ++R->pending, ++R->open[ticket];
    }
    {
        std::lock_guard lock(R->mutex);
        for (size_t i = first; i < R->requests.size(); ++i) R->queue.push_back(&R->requests[i]);
    }
    R->wake.notify_one();
    return int(head);
}

// Takes one completed chunk off the port, waiting up to `wait` ms. Returns 1, 0 if none completed, -1 if the port failed.
static int take(SclmReader* R, DWORD wait) {
    DWORD got;
    ULONG_PTR key;
    OVERLAPPED* ov;
    const BOOL ok = GetQueuedCompletionStatus(R->port, &got, &key, &ov, wait);
    if (!ov) return GetLastError() == WAIT_TIMEOUT ? 0 : -1;
    const Request* q = CONTAINING_RECORD(ov, Request, ov);
    --R->pending, --R->open[q->ticket];
    if (!ok || key == 1 || got < q->need) ++R->failed;
    return 1;
}

// How many reads failed since the last call that reported them.
static int settle(SclmReader* R) {
    if (!R->pending) R->requests.clear();
    const int failed = R->failed;
    R->failed = 0;
    return failed;
}

// Takes in the reads that have completed, without waiting. Returns how many failed.
__declspec(dllexport) int sclm_reader_poll(SclmReader* R) {
    while (R->pending) {
        const int took = take(R, 0);
        if (took < 0) return -1;
        if (!took) break;
    }
    return settle(R);
}

// Chunks of the ticket's reads that have not completed, as of the last poll or wait.
__declspec(dllexport) int sclm_reader_open(SclmReader* R, int ticket) { return R->open[ticket]; }

// Waits for the reads submitted under the ticket (others may still run). Returns how many reads failed.
__declspec(dllexport) int sclm_reader_wait_ticket(SclmReader* R, int ticket) {
    while (R->open[ticket])
        if (take(R, INFINITE) < 0) return -1;
    return settle(R);
}

// Waits for every read submitted. Returns how many of them failed.
__declspec(dllexport) int sclm_reader_wait(SclmReader* R) {
    while (R->pending)
        if (take(R, INFINITE) < 0) return -1;
    return settle(R);
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
