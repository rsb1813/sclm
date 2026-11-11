// CPU engine for the RAM tier: a layer's vector-coded SwiGLU experts, computed step for step as the reference computes
// them (quant.VQLinear inside the upstream Expert), except that the GEMVs sum int8 tables instead of fp32 products.
// A call is split into tasks (row tiles, parts of tables) that a thread pool shares; the arithmetic does not depend on
// the thread count.
#include <malloc.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>
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

constexpr int PARTS = 16;  // the tables over the layer input are built in this many parts

struct Format {
    float* cb;
    int d;
    Tables in;          // over the layer input: w1 and w3 of every expert of the format read them
    float peak[PARTS];  // the largest entry of each part
};

// One expert of the call in progress.
struct Slot {
    const SclmExpert* ex = nullptr;
    float weight = 0;
    Tables mid;  // over the expert's hidden vector (w2)
    std::vector<float> gate, up, out;
};

// Tasks of one kind of a call. They can start when what they read is complete, and something waits for the last one.
struct Group {
    int count = 0;
    std::atomic<int> next{0}, left{0};  // the first unclaimed task, tasks not finished
    std::atomic<bool> ready{false};
};

// Runs fn(ctx, i) for i < count on the workers and, once it joins, the calling thread. Idle workers spin for about
// 0.1 ms, then sleep.
struct Pool {
    std::vector<std::thread> workers;
    std::atomic<uint64_t> next{0};   // count << 32 | first unclaimed index, so a claim reads and takes in one step
    std::atomic<uint32_t> epoch{0};  // bumped per job: what sleeping workers wait on
    std::atomic<int> done{0};
    std::atomic<bool> stop{false};
    void (*fn)(void*, int) = nullptr;
    void* ctx = nullptr;
    int count = 0;

    explicit Pool(int threads) {
        for (int t = 1; t < threads; ++t) workers.emplace_back([this] { work(); });
    }
    ~Pool() {
        stop = true, ++epoch, epoch.notify_all();
        for (auto& w : workers) w.join();
    }
    bool claim(int& i) {
        uint64_t v = next.load();
        while (uint32_t(v) < uint32_t(v >> 32))
            if (next.compare_exchange_weak(v, v + 1)) return i = int(uint32_t(v)), true;
        return false;
    }
    void work() {
        for (;;) {
            uint32_t seen = epoch.load();
            uint64_t t0 = __rdtsc();
            int i;
            while (!claim(i)) {
                if (stop) return;
                if (__rdtsc() - t0 < 400000) {
                    _mm_pause();
                } else {
                    epoch.wait(seen);
                    seen = epoch.load(), t0 = __rdtsc();
                }
            }
            fn(ctx, i);  // set before the job's `next`, and the job cannot end before this task does
            ++done;
        }
    }
    void start(int n, void (*f)(void*, int), void* c) {
        fn = f, ctx = c, done = 0, count = n;
        next = uint64_t(n) << 32;
        ++epoch, epoch.notify_all();
    }
    // The calling thread takes what is unclaimed and returns when the job is done.
    void join() {
        for (int i; claim(i); ++done) fn(ctx, i);
        while (done.load() < count) _mm_pause();
    }
};

Pool* pool = new Pool(1);

// Memory the workers read ahead of the call expected to need it (sclm_layer_touch), so that the call finds it in the
// cache: a call's time is mostly reading codes from memory, and between two calls the threads and the memory idle.
struct Touch {
    static constexpr int RANGES = 48;
    static constexpr size_t CHUNK = 64 << 10;
    const uint8_t* at[RANGES];
    size_t first[RANGES + 1] = {};  // the chunks before each range
    int n = 0;
    std::atomic<size_t> next{0};  // the first unclaimed chunk
    std::atomic<bool> stop{false};
    bool on = false;  // the pool's job is this
    const uint8_t* end[RANGES];
} touch;
std::atomic<uint64_t> touch_asked{0}, touch_read{0};  // bytes, since the process started
std::atomic<uint64_t> touch_sink{0};
// The most workers that read ahead. One reads nearly as fast as seven (a thread fills the memory's bandwidth), and
// more of them slow the calling thread's own work down: 8 guessed experts a layer, 55.8 ms a token without, 54.2
// with one worker, 55.0 with two, 55.9 with seven.
int touch_threads = 1;

// One worker's share: chunks in order (the caller lists the likeliest memory first) until none is left or the job is
// stopped. A load a cache line is enough; the result only keeps the loads from being dropped.
void touch_work(void*, int) {
    __m512i sum = _mm512_setzero_si512();
    uint64_t bytes = 0;
    for (int r = 0; !touch.stop.load(std::memory_order_relaxed);) {
        const size_t c = touch.next.fetch_add(1);
        if (c >= touch.first[touch.n]) break;
        while (c >= touch.first[r + 1]) ++r;
        const uint8_t* p = touch.at[r] + (c - touch.first[r]) * Touch::CHUNK;
        const uint8_t* e = std::min(p + Touch::CHUNK, touch.end[r]);
        bytes += e - p;
        for (; p + 64 <= e; p += 64) sum = _mm512_xor_si512(sum, _mm512_loadu_si512(p));
    }
    touch_sink += _mm512_reduce_add_epi64(sum);
    touch_read += bytes;
}

// Ends the read-ahead, if one runs: the pool is free for the next job when this returns.
void touch_end() {
    if (!touch.on) return;
    touch.stop = true;
    pool->join();  // takes the shares no worker has claimed (they return at once) and waits for the others
    touch.on = false;
}

// Seconds spent in each phase of the layer calls since the process started (a measuring aid).
std::atomic<double> phase_seconds[7];
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Activations per scale of the reference's fake-quant: 32 in DeepSeek-V4.1-Flash, 128 in DeepSeek-V4-Flash-0731.
int act_block = 32;

// The reference's activation fake-quant before a quantized GEMM (kernel.act_quant: blocks of act_block, power-of-2
// scale): a block is scaled so its largest magnitude is at most 448, then rounded to FP8 E4M3.
void fp8_blocks(float* x, int n) {
    const __m512 zero = _mm512_setzero_ps();
    for (int b = 0; b < n; b += act_block) {
        __m512 mx = _mm512_abs_ps(_mm512_loadu_ps(x + b));
        for (int i = b + 16; i < b + act_block; i += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(x + i)));
        int e;
        const float m = std::frexp(std::max(_mm512_reduce_max_ps(mx), 1e-4f) * float(1.0 / 448.0), &e);
        const __m512 se = _mm512_set1_ps(float(m == 0.5f ? e - 1 : e));  // the scale is 2^ceil(log2(amax / 448))
        for (int i = b; i < b + act_block; i += 16) {
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

// x <- H (signs * x) / sqrt(block) over each block (quant.BlockHadamard.rotate); a block is a power of two, 16 or
// more. Step h of the transform takes every pair (a, b) that is h apart to (a + b, a - b): 16 pairs at a time here.
void rotate(float* x, int n, const Rotation& rot) {
    const __m512 norm = _mm512_set1_ps(float(std::sqrt(double(rot.block))));
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    for (float* p = x; p < x + n; p += rot.block) {
        for (int i = 0; i < rot.block; i += 16) {  // h = 1, 2, 4, 8: both of a pair are in one vector
            __m512 v = _mm512_mul_ps(_mm512_loadu_ps(p + i), _mm512_loadu_ps(rot.signs.data() + (p - x) + i));
            for (int h = 1; h < 16; h *= 2) {
                const __m512i bit = _mm512_set1_epi32(h);
                const __m512 other = _mm512_permutexvar_ps(_mm512_xor_si512(lane, bit), v);
                // the upper of a pair holds b and gets a - b
                v = _mm512_mask_sub_ps(_mm512_add_ps(v, other), _mm512_test_epi32_mask(lane, bit), other, v);
            }
            _mm512_storeu_ps(p + i, v);
        }
        for (int h = 16; h < rot.block; h *= 2)
            for (int i = 0; i < rot.block; i += 2 * h)
                for (int j = i; j < i + h; j += 16) {
                    const __m512 a = _mm512_loadu_ps(p + j), b = _mm512_loadu_ps(p + j + h);
                    _mm512_storeu_ps(p + j, _mm512_add_ps(a, b)), _mm512_storeu_ps(p + j + h, _mm512_sub_ps(a, b));
                }
        for (int i = 0; i < rot.block; i += 16) _mm512_storeu_ps(p + i, _mm512_div_ps(_mm512_loadu_ps(p + i), norm));
    }
}

// out[rows of tile t] = those rows of the matrix times the input the tables were built from.
void matvec_tile(const uint8_t* codes, const uint16_t* scales, const Tables& tb, int t, float* out) {
    const size_t tile_bytes = size_t(tb.groups) * (tb.codes == 256 ? TILE : 192);
    gemv_tile(codes + t * tile_bytes, tb, out + t * TILE);
    for (int i = t * TILE; i < (t + 1) * TILE; i += 16) {
        const __m512 s = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(scales + i)));
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(out + i), s));
    }
}

Tables tables(int groups, int codes) {
    Tables tb;
    tb.groups = groups, tb.codes = codes, tb.t = aligned<int8_t>(size_t(groups) * codes);
    return tb;
}

}  // namespace

struct SclmLayer {
    int dim, inter;
    float limit;
    std::vector<Format> formats;
    Rotation rot_dim, rot_inter;
    std::vector<float> x;
    // the call in progress
    const float* in = nullptr;
    std::vector<int> used;    // its formats
    std::vector<Slot> slots;  // its experts
    int n = 0;
    // its tasks: the input (one task), then per format the largest table entry (PARTS tasks) and the tables (PARTS),
    // then per expert w1 with w3 (a task a tile) and w2
    std::unique_ptr<Group[]> groups;
    int n_groups = 0, room = 0;
    std::atomic<int> open{0};  // experts not finished
    double started = 0, ended = 0;
};

namespace {

// SwiGLU with the routing weight, then the input of w2 prepared as the reference prepares it.
void hidden(SclmLayer* L, int e) {
    Slot& s = L->slots[e];
    const Format& f = L->formats[s.ex->format];
    const int inter = L->inter;
    float *gate = s.gate.data(), *up = s.up.data();
    bf16(gate, inter), bf16(up, inter);
    for (int i = 0; i < inter; ++i) {
        const float g = L->limit > 0 ? std::min(gate[i], L->limit) : gate[i];
        const float u = L->limit > 0 ? std::clamp(up[i], -L->limit, L->limit) : up[i];
        gate[i] = s.weight * (g / (1.0f + std::exp(-g)) * u);
    }
    bf16(gate, inter);
    fp8_blocks(gate, inter);
    rotate(gate, inter, L->rot_inter);
    s.mid.codes = f.in.codes;
    build_tables(f.cb, f.d, gate, inter, s.mid);
}

// Task i of group g. Returns its phase (see sclm_phase_seconds).
int task(SclmLayer* L, int g, int i) {
    const int dim = L->dim, formats = int(L->used.size());
    float* x = L->x.data();
    if (g == 0) {
        std::memcpy(x, L->in, sizeof(float) * dim);
        fp8_blocks(x, dim);
        rotate(x, dim, L->rot_dim);
        return 0;
    }
    if (g <= 2 * formats) {  // part i of a format's input tables: first its largest entry, then the entries on the scale
        Format& f = L->formats[L->used[(g - 1) % formats]];
        const int j0 = i * f.in.groups / PARTS, j1 = (i + 1) * f.in.groups / PARTS;
        if (g <= formats) f.peak[i] = table_max(f.cb, f.d, f.in.codes, x, j0, j1);
        else table_fill(f.cb, f.d, f.in.codes, x, j0, j1, f.in.scale, f.in.t);
        return 1;
    }
    Slot& s = L->slots[(g - 1 - 2 * formats) % L->n];
    if (g <= 2 * formats + L->n) {  // a tile of w1, then of w3
        const int tiles = L->inter / TILE, w = i / tiles;
        matvec_tile(s.ex->codes[2 * w], s.ex->scales[2 * w], L->formats[s.ex->format].in, i % tiles, (w ? s.up : s.gate).data());
        return 2;
    }
    matvec_tile(s.ex->codes[1], s.ex->scales[1], s.mid, i, s.out.data());  // a tile of w2
    bf16(s.out.data() + i * TILE, TILE);
    return 4;
}

// Group g's last task is done: the calling thread does what follows it and lets the tasks that waited for it start.
void finish(SclmLayer* L, int g) {
    Group* groups = L->groups.get();
    const int formats = int(L->used.size()), n = L->n;
    if (g == 0) {
        for (int f = 1; f <= formats; ++f) groups[f].ready = true;
    } else if (g <= formats) {
        Format& f = L->formats[L->used[g - 1]];
        f.in.scale = *std::max_element(f.peak, f.peak + PARTS) / 127.0f;
        groups[g + formats].ready = true;
    } else if (g <= 2 * formats) {
        for (int e = 0; e < n; ++e)
            if (L->slots[e].ex->format == L->used[g - 1 - formats]) groups[1 + 2 * formats + e].ready = true;
    } else if (g <= 2 * formats + n) {
        hidden(L, g - 1 - 2 * formats);
        groups[g + n].ready = true;
    } else if (L->open.fetch_sub(1) == 1) {
        L->ended = now();
    }
}

// One thread's share of a call: tasks of the groups that are ready, the earlier group first, until every expert is
// done. While the tiles keep the memory busy, the tables and hidden vectors cost the call next to nothing.
void work(void* ctx, int) {
    auto* L = (SclmLayer*)ctx;
    double seconds[5] = {};
    while (L->open.load() > 0) {
        int g = 0, i = 0;
        for (; g < L->n_groups; ++g) {
            Group& group = L->groups[g];
            if (group.ready.load() && group.next.load() < group.count && (i = group.next.fetch_add(1)) < group.count) break;
        }
        if (g == L->n_groups) {  // what is left waits for a task another thread is at
            _mm_pause();
            continue;
        }
        const double t0 = now();
        const int phase = task(L, g, i);
        const double t1 = now();
        seconds[phase] += t1 - t0;
        if (L->groups[g].left.fetch_sub(1) == 1) {
            finish(L, g);
            if (phase == 2) seconds[3] += now() - t1;  // the expert's hidden vector
        }
    }
    for (int p = 0; p < 5; ++p) phase_seconds[p] += seconds[p];
}

}  // namespace

extern "C" {

// dim and inter are multiples of 256. The codebooks and signs are copied; the experts' codes and scales are not.
__declspec(dllexport) SclmLayer* sclm_layer_new(int dim, int inter, float swiglu_limit, const SclmFormat* formats, int n_formats,
                                                const int8_t* signs_dim, int block_dim, const int8_t* signs_inter, int block_inter) {
    auto* L = new SclmLayer{dim, inter, swiglu_limit};
    for (int f = 0; f < n_formats; ++f) {
        const int d = formats[f].d, codes = 1 << formats[f].bits;
        float* cb = aligned<float>(size_t(d) * codes);
        std::memcpy(cb, formats[f].codebook, sizeof(float) * d * codes);
        L->formats.push_back({cb, d, tables(dim / d, codes)});
    }
    L->rot_dim = {std::vector<float>(signs_dim, signs_dim + dim), block_dim};
    L->rot_inter = {std::vector<float>(signs_inter, signs_inter + inter), block_inter};
    L->x.resize(dim);
    return L;
}

__declspec(dllexport) void sclm_layer_free(SclmLayer* L) {
    for (Format& f : L->formats)
        for (void* p : {(void*)f.cb, (void*)f.in.t}) _aligned_free(p);
    for (Slot& s : L->slots) _aligned_free(s.mid.t);
    delete L;
}

// Threads that share a layer call (the caller's included). Not to be called while a call runs.
// The model's activation block (a multiple of 16 that divides the experts' widths), before any layer call.
__declspec(dllexport) void sclm_set_act_block(int block) { act_block = block; }

__declspec(dllexport) void sclm_set_threads(int threads) {
    touch_end();
    delete pool;
    pool = new Pool(threads);
}

// Starts y = sum of gates[e] * expert e of x[dim] over the n experts (MoE.forward's routed sum for one token) on the
// pool's workers and returns: the caller can do other work until it takes y with sclm_layer_wait. experts and x are
// read until then. One call at a time over all layers: the pool is shared.
__declspec(dllexport) void sclm_layer_start(SclmLayer* L, const SclmExpert* experts, const float* gates, int n, const float* x) {
    const int dim = L->dim, inter = L->inter;
    touch_end();
    L->started = L->ended = now();
    while (int(L->slots.size()) < n) {  // a slot's tables are sized for the largest format
        size_t entries = 0;
        for (const Format& f : L->formats) entries = std::max(entries, size_t(inter / f.d) * f.in.codes);
        Slot& s = L->slots.emplace_back();
        s.mid.t = aligned<int8_t>(entries);
        s.gate.resize(inter), s.up.resize(inter), s.out.resize(dim);
    }
    L->used.clear();
    for (int e = 0; e < n; ++e) {
        L->slots[e].ex = experts + e, L->slots[e].weight = gates[e];
        if (std::find(L->used.begin(), L->used.end(), experts[e].format) == L->used.end()) L->used.push_back(experts[e].format);
    }
    const int formats = int(L->used.size());
    L->in = x, L->n = n, L->n_groups = 1 + 2 * formats + 2 * n, L->open = n;
    if (L->room < L->n_groups) L->groups.reset(new Group[L->room = L->n_groups]);
    for (int g = 0; g < L->n_groups; ++g) {
        Group& group = L->groups[g];
        group.count = g == 0 ? 1 : g <= 2 * formats ? PARTS : g <= 2 * formats + n ? 2 * (inter / TILE) : dim / TILE;
        group.next = 0, group.left = group.count, group.ready = g == 0;
    }
    if (n) pool->start(int(pool->workers.size()) + 1, work, L);  // every thread works until the call is done
}

// Starts reading the codes of n experts of L (the likeliest first) into the cache on the pool's workers and returns:
// for experts a call on L is expected to use soon. The read-ahead ends by itself or when the next call starts; it
// changes no result. Nothing happens without workers.
__declspec(dllexport) void sclm_layer_touch(SclmLayer* L, const SclmExpert* experts, int n) {
    touch_end();
    if (pool->workers.empty()) return;
    int r = 0;
    for (int e = 0; e < n && r + 3 <= Touch::RANGES; ++e) {
        const Format& f = L->formats[experts[e].format];
        const size_t step = f.in.codes == 256 ? TILE : 192;  // bytes of a tile's codes for one group of columns
        const size_t wide = size_t(L->inter / TILE) * (L->dim / f.d) * step, tall = size_t(L->dim / TILE) * (L->inter / f.d) * step;
        for (int w : {0, 2, 1}) {  // w1, w3, w2: the order a call reads them in
            const size_t bytes = w == 1 ? tall : wide;
            touch.at[r] = experts[e].codes[w], touch.end[r] = touch.at[r] + bytes;
            touch.first[r + 1] = touch.first[r] + (bytes + Touch::CHUNK - 1) / Touch::CHUNK;
            touch_asked += bytes, ++r;
        }
    }
    if (!r) return;
    touch.n = r, touch.next = 0, touch.stop = false, touch.on = true;
    pool->start(std::min(int(pool->workers.size()), touch_threads), touch_work, nullptr);
}

__declspec(dllexport) void sclm_touch_threads(int threads) { touch_threads = threads; }

// out[2] = bytes sclm_layer_touch was asked to read ahead so far, and bytes it read before the next call started.
__declspec(dllexport) void sclm_touch_bytes(double* out) { out[0] = double(touch_asked.load()), out[1] = double(touch_read.load()); }

// The calling thread joins the call started on L, and y[dim] is its result.
__declspec(dllexport) void sclm_layer_wait(SclmLayer* L, float* y) {
    const int dim = L->dim, n = L->n;
    if (n) pool->join();
    const double t0 = now();
    std::memset(y, 0, sizeof(float) * dim);
    for (int e = 0; e < n; ++e)  // in call order, as the reference adds them
        for (int i = 0; i < dim; ++i) y[i] += L->slots[e].out[i];
    phase_seconds[5] += now() - t0, phase_seconds[6] += L->ended - L->started;
}

__declspec(dllexport) void sclm_layer_forward(SclmLayer* L, const SclmExpert* experts, const float* gates, int n, const float* x, float* y) {
    sclm_layer_start(L, experts, gates, n, x);
    sclm_layer_wait(L, y);
}

// out[7] = seconds so far in the phases of the layer calls. The first five are summed over the threads, which work
// on them at the same time: input quantization and rotation, the input tables, w1 and w3, the hidden vectors with
// their tables, w2. Then the sum of the experts, and by the clock the time from a call's start to its last expert.
__declspec(dllexport) void sclm_phase_seconds(double* out) {
    for (int p = 0; p < 7; ++p) out[p] = phase_seconds[p];
}

}  // extern "C"
