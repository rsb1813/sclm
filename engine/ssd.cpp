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
const int AHEAD = 4;  // chunks of unwaited reads (tickets other than 0) under way at a time

}  // namespace

// Starting a read costs about 0.3 ms per 18 MB (the buffer's pages are locked for the transfer), so a thread of its
// own issues them: the caller queues a request and goes on computing. The same thread takes in the completions, so
// it knows how much is under way: a read somebody waits for (ticket 0) is issued at once, the others (read-ahead
// guesses, background loads) only a few chunks at a time, so that they do not share the SSD's speed with it.
struct SclmReader {
    std::vector<HANDLE> files;
    HANDLE port = nullptr;
    uint64_t chunk = 0;
    std::mutex mutex;              // guards what follows
    std::condition_variable done;  // a chunk completed
    std::deque<Request> requests;  // of the reads not all completed yet; a deque keeps their addresses
    std::deque<Request*> queue;    // submitted, not yet issued
    int pending = 0, failed = 0;   // chunks not yet completed, chunks that failed
    int open[TICKETS] = {};        // chunks not yet completed, per ticket
    int flight = 0;                // chunks issued and not yet completed
    bool stop = false;
    std::thread server;

    void serve() {
        std::vector<Request*> batch;
        for (;;) {
            {
                std::lock_guard lock(mutex);
                if (stop) return;
                for (const bool waited : {true, false})
                    for (auto it = queue.begin(); it != queue.end();) {
                        if (((*it)->ticket == 0) == waited && (waited || flight < AHEAD)) {
                            batch.push_back(*it), ++flight;
                            it = queue.erase(it);
                        } else {
                            ++it;
                        }
                    }
            }
            for (Request* q : batch)
                if (!ReadFile(q->file, q->dst, q->len, nullptr, &q->ov) && GetLastError() != ERROR_IO_PENDING)
                    PostQueuedCompletionStatus(port, 0, 1, &q->ov);  // key 1: the read did not start
            batch.clear();
            DWORD got;
            ULONG_PTR key;
            OVERLAPPED* ov;
            const BOOL ok = GetQueuedCompletionStatus(port, &got, &key, &ov, INFINITE);
            if (!ov) continue;  // woken: something was submitted, or the reader stops
            const Request* q = CONTAINING_RECORD(ov, Request, ov);
            {
                std::lock_guard lock(mutex);
                --flight, --pending, --open[q->ticket];
                if (!ok || key == 1 || got < q->need) ++failed;
            }
            done.notify_all();
        }
    }

    // How many chunks failed since the last call that reported them. With the mutex held.
    int settle() {
        if (!pending) requests.clear();
        const int n = failed;
        failed = 0;
        return n;
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
    R->server = std::thread([R] { R->serve(); });
    return R;
}

__declspec(dllexport) void sclm_reader_free(SclmReader* R) {
    {
        std::lock_guard lock(R->mutex);
        R->stop = true;
    }
    PostQueuedCompletionStatus(R->port, 0, 0, nullptr);
    R->server.join();
    for (HANDLE h : R->files) CloseHandle(h);
    CloseHandle(R->port);
    delete R;
}

// Starts reading bytes [offset, offset + bytes) of file `file` into dst and returns where in dst the first of them
// lands (the read starts on the page boundary below offset). dst is page-aligned and holds bytes + 2 pages.
// `ticket` (below 256) names the read for sclm_reader_open and sclm_reader_wait_ticket; ticket 0 is for reads the
// caller is about to wait for, which go ahead of the others.
__declspec(dllexport) int sclm_reader_submit(SclmReader* R, int file, uint64_t offset, uint64_t bytes, uint8_t* dst, int ticket) {
    const uint64_t head = offset % PAGE, begin = offset - head, end = (offset + bytes + PAGE - 1) / PAGE * PAGE;
    {
        std::lock_guard lock(R->mutex);
        for (uint64_t at = begin; at < end; at += R->chunk) {
            const uint64_t len = std::min(R->chunk, end - at);
            Request& q = R->requests.emplace_back();
            q.ov = {};
            q.ov.Offset = DWORD(at), q.ov.OffsetHigh = DWORD(at >> 32);
            q.need = DWORD(std::min(len, offset + bytes - at));
            q.file = R->files[file], q.dst = dst + (at - begin), q.len = DWORD(len), q.ticket = ticket;
            R->queue.push_back(&q);
            ++R->pending, ++R->open[ticket];
        }
    }
    PostQueuedCompletionStatus(R->port, 0, 0, nullptr);
    return int(head);
}

// sclm_reader_submit for n reads in one call: read i is bytes[i] at offsets[i] of files[i] into dsts[i], and heads[i]
// is where its first byte lands. (The engram rows of a token are a hundred reads.)
__declspec(dllexport) void sclm_reader_submit_many(SclmReader* R, int n, const int* files, const uint64_t* offsets,
                                                   const uint64_t* bytes, uint8_t* const* dsts, int ticket, int* heads) {
    for (int i = 0; i < n; ++i) heads[i] = sclm_reader_submit(R, files[i], offsets[i], bytes[i], dsts[i], ticket);
}

// How many chunks failed since the last call that reported them (nothing is waited for).
__declspec(dllexport) int sclm_reader_poll(SclmReader* R) {
    std::lock_guard lock(R->mutex);
    return R->settle();
}

// Chunks of the ticket's reads that have not completed.
__declspec(dllexport) int sclm_reader_open(SclmReader* R, int ticket) {
    std::lock_guard lock(R->mutex);
    return R->open[ticket];
}

// Waits for the reads submitted under the ticket (others may still run). Returns how many chunks failed.
__declspec(dllexport) int sclm_reader_wait_ticket(SclmReader* R, int ticket) {
    std::unique_lock lock(R->mutex);
    R->done.wait(lock, [&] { return R->open[ticket] == 0; });
    return R->settle();
}

// Waits for every read submitted. Returns how many chunks failed.
__declspec(dllexport) int sclm_reader_wait(SclmReader* R) {
    std::unique_lock lock(R->mutex);
    R->done.wait(lock, [&] { return R->pending == 0; });
    return R->settle();
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
