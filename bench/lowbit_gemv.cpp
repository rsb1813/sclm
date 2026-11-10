// AVX-512 VNNI GEMV benchmark for the RAM-tier expert formats (GPTQ 1-bit, ternary, 2-bit, 3-bit at group 128).
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#include <malloc.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

using clk = std::chrono::steady_clock;
#define FI __forceinline

// A matrix [N, K] is N/16 row tiles; a row tile is K/256 consecutive blocks. A block holds the codes of 16 rows x 256
// columns as 64 "steps" of 64 bytes (byte p of step j is row p/4, column 4j + p%4: the operand order of vpdpbusd),
// packed per format, then the fp16 steps of its two 128-column groups (16 rows each).
//   1b: 8 chunks, step j in bit j%8 of chunk j/8      2b: 16 chunks, bits 2(j%4) of chunk j/4
//   3b: 2b plane + 1b plane (bit 2 of the code)        t: base 3, 5 digits per byte (12 chunks + 1 of 4 digits)
enum Fmt { F1, FT, F2, F3, NFMT };
static const char* NAME[] = {"1b", "t", "2b", "3b"};
constexpr int LEVELS[] = {2, 3, 4, 8};
constexpr float CENTER[] = {-0.5f, -1.0f, -1.5f, -3.5f};  // weight = (code + CENTER) * step
constexpr int CODE_BYTES[] = {512, 832, 1024, 1536};
constexpr int block_bytes(int f) { return CODE_BYTES[f] + 64; }
constexpr int TILE = 16, BLOCK = 256, GROUP = 128, DIM = 5120, INTER = 2304;
constexpr int MAT_BYTES_PER_BB = INTER / TILE * (DIM / BLOCK);  // blocks per matrix; same for w1/w3 and w2 (2880)

alignas(64) static uint8_t TRIT[5][256];  // digit s of a base-3 byte

static uint16_t to_half(float v) { return uint16_t(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(v), 0), 0)); }
static float from_half(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

struct QVec {  // x ~ a_g * q over each 128-group; sum_g = a_g * sum(q in g)
    alignas(64) int8_t q[DIM];
    float a[DIM / GROUP], sum[DIM / GROUP];
};

template <int F, int J> FI __m512i codes(const uint8_t* b) {
    if constexpr (F == F1) {
        return _mm512_and_si512(_mm512_srli_epi16(_mm512_load_si512(b + J / 8 * 64), J % 8), _mm512_set1_epi8(1));
    } else if constexpr (F == F2) {
        return _mm512_and_si512(_mm512_srli_epi16(_mm512_load_si512(b + J / 4 * 64), 2 * (J % 4)), _mm512_set1_epi8(3));
    } else if constexpr (F == F3) {
        __m512i lo = _mm512_srli_epi16(_mm512_load_si512(b + J / 4 * 64), 2 * (J % 4));
        __m512i hi = _mm512_load_si512(b + 1024 + J / 8 * 64);
        if constexpr (J % 8 >= 2) hi = _mm512_srli_epi16(hi, J % 8 - 2);
        else hi = _mm512_slli_epi16(hi, 2 - J % 8);  // spill into the next byte stays below bit 2
        return _mm512_or_si512(_mm512_and_si512(lo, _mm512_set1_epi8(3)), _mm512_and_si512(hi, _mm512_set1_epi8(4)));
    } else {
        constexpr int c = J < 60 ? J / 5 : 12, s = J < 60 ? J % 5 : J - 60;
        const __m512i v = _mm512_load_si512(b + c * 64);
        const __m512i* t = (const __m512i*)TRIT[s];
        const __m512i lo = _mm512_permutex2var_epi8(t[0], v, t[1]);  // byte values 0..127
        const __m512i hi = _mm512_permutex2var_epi8(t[2], v, t[3]);  // 128..255
        return _mm512_mask_blend_epi8(_mm512_movepi8_mask(v), lo, hi);
    }
}

template <int F, int R, int J> FI void step(const uint8_t* const* b, const int32_t* x4, __m512i* d) {
    const __m512i xb = _mm512_set1_epi32(x4[J]);
    for (int r = 0; r < R; ++r) d[r] = _mm512_dpbusd_epi32(d[r], codes<F, J>(b[r]), xb);
}

template <int F, int R, int G>
FI void group(const uint8_t* const* b, const int32_t* x4, float a, float c_sum, __m512* acc) {
    __m512i d[R];
    for (int r = 0; r < R; ++r) d[r] = _mm512_setzero_si512();
    [&]<int... J>(std::integer_sequence<int, J...>) { (step<F, R, 32 * G + J>(b, x4, d), ...); }
    (std::make_integer_sequence<int, 32>{});
    for (int r = 0; r < R; ++r) {
        const __m512 s = _mm512_cvtph_ps(_mm256_load_si256((const __m256i*)(b[r] + CODE_BYTES[F] + 32 * G)));
        const __m512 t = _mm512_fmadd_ps(_mm512_cvtepi32_ps(d[r]), _mm512_set1_ps(a), _mm512_set1_ps(c_sum));
        acc[r] = _mm512_fmadd_ps(s, t, acc[r]);
    }
}

// out[r][0..16) = (row tile tiles[r]) x; R row tiles share each activation broadcast.
template <int F, int R> static void gemv(const uint8_t* const* tiles, int nblk, const QVec& x, float* const* out) {
    __m512 acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm512_setzero_ps();
    const uint8_t* b[R];
    for (int k = 0; k < nblk; ++k) {
        for (int r = 0; r < R; ++r) b[r] = tiles[r] + size_t(k) * block_bytes(F);
        const int32_t* x4 = (const int32_t*)(x.q + k * BLOCK);
        group<F, R, 0>(b, x4, x.a[2 * k], CENTER[F] * x.sum[2 * k], acc);
        group<F, R, 1>(b, x4, x.a[2 * k + 1], CENTER[F] * x.sum[2 * k + 1], acc);
    }
    for (int r = 0; r < R; ++r) _mm512_storeu_ps(out[r], acc[r]);
}

using GemvFn = void (*)(const uint8_t* const*, int, const QVec&, float* const*);
static const GemvFn GEMV4[] = {gemv<F1, 4>, gemv<FT, 4>, gemv<F2, 4>, gemv<F3, 4>};

static void fwht(float* x, int n) {
    for (int h = 1; h < n; h *= 2)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
}

// Block Hadamard of n values (block = largest power of 2 dividing n, at most 1024) after a sign flip.
static void rotate(float* x, int n, const float* signs) {
    const int b = std::min(n & -n, 1024);
    for (int i = 0; i < n; ++i) x[i] *= signs[i];
    for (int i = 0; i < n; i += b) fwht(x + i, b);
    const float s = 1.0f / std::sqrt(float(b));
    for (int i = 0; i < n; ++i) x[i] *= s;
}

// int8 with one absmax scale per 128 values; g0 is the first group's index in out.
static void quantize(const float* x, int n, QVec& out, int g0 = 0) {
    for (int g = 0; g < n / GROUP; ++g) {
        const float* v = x + g * GROUP;
        float m = 0;
        for (int i = 0; i < GROUP; ++i) m = std::max(m, std::fabs(v[i]));
        const float a = m > 0 ? m / 127 : 1, inv = 1 / a;
        int s = 0;
        for (int i = 0; i < GROUP; ++i) {
            const int q = int(std::lrintf(v[i] * inv));
            out.q[(g0 + g) * GROUP + i] = int8_t(q);
            s += q;
        }
        out.a[g0 + g] = a;
        out.sum[g0 + g] = a * s;
    }
}

// Packs codes u [n, k] (< LEVELS[f]) and steps st [n, k/128] into the layout above.
static void pack(int f, const uint8_t* u, const float* st, int n, int k, uint8_t* out) {
    static const int POW3[] = {1, 3, 9, 27, 81};
    const int nb = k / BLOCK;
    for (int rt = 0; rt < n / TILE; ++rt)
        for (int blk = 0; blk < nb; ++blk) {
            uint8_t* b = out + (size_t(rt) * nb + blk) * block_bytes(f);
            memset(b, 0, block_bytes(f));
            for (int j = 0; j < 64; ++j)
                for (int p = 0; p < 64; ++p) {
                    const int v = u[(rt * TILE + p / 4) * k + blk * BLOCK + 4 * j + p % 4];
                    if (f == F1) b[j / 8 * 64 + p] |= v << (j % 8);
                    else if (f == F2) b[j / 4 * 64 + p] |= v << 2 * (j % 4);
                    else if (f == F3) {
                        b[j / 4 * 64 + p] |= (v & 3) << 2 * (j % 4);
                        b[1024 + j / 8 * 64 + p] |= (v >> 2) << (j % 8);
                    } else b[(j < 60 ? j / 5 : 12) * 64 + p] += v * POW3[j < 60 ? j % 5 : j - 60];
                }
            for (int g = 0; g < 2; ++g)
                for (int r = 0; r < TILE; ++r)
                    ((uint16_t*)(b + CODE_BYTES[f] + 32 * g))[r] = to_half(st[(rt * TILE + r) * (k / GROUP) + blk * 2 + g]);
        }
}

// Kernel vs exact integer reference (same quantized activations) and vs the float reference (activation error).
static bool selftest() {
    const int n = 64, k = 512, ng = k / GROUP;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd;
    bool ok = true;
    for (int f = 0; f < NFMT; ++f) {
        std::vector<uint8_t> u(n * k);
        std::vector<float> st(n * ng), x(k), y(n);
        for (auto& v : u) v = uint8_t(rng() % LEVELS[f]);
        for (auto& v : st) v = from_half(to_half(0.01f + 0.02f * (rng() % 1000) / 1000.0f));
        for (auto& v : x) v = nd(rng);
        auto* buf = (uint8_t*)_aligned_malloc(size_t(n / TILE) * (k / BLOCK) * block_bytes(f), 64);
        pack(f, u.data(), st.data(), n, k, buf);
        auto q = std::make_unique<QVec>();
        quantize(x.data(), k, *q);
        const uint8_t* tiles[4];
        float* outs[4];
        for (int r = 0; r < 4; ++r) tiles[r] = buf + size_t(r) * (k / BLOCK) * block_bytes(f), outs[r] = y.data() + 16 * r;
        GEMV4[f](tiles, k / BLOCK, *q, outs);
        double e_int = 0, e_act = 0, p_int = 0, p_act = 0;
        for (int i = 0; i < n; ++i) {
            double ri = 0, rf = 0;
            for (int g = 0; g < ng; ++g) {
                long long d = 0;
                for (int c = g * GROUP; c < (g + 1) * GROUP; ++c) {
                    d += u[i * k + c] * q->q[c];
                    rf += (u[i * k + c] + CENTER[f]) * double(st[i * ng + g]) * x[c];
                }
                ri += st[i * ng + g] * (double(q->a[g]) * d + CENTER[f] * double(q->sum[g]));
            }
            e_int = std::max(e_int, std::fabs(y[i] - ri)), p_int += ri * ri;
            e_act += (y[i] - rf) * (y[i] - rf), p_act += rf * rf;
        }
        const double rel_int = e_int / std::sqrt(p_int / n), rel_act = std::sqrt(e_act / p_act);
        printf("selftest %-2s: max err vs int reference %.2e (rms-relative), int8 activation error %.3f%%\n", NAME[f],
               rel_int, 100 * rel_act);
        ok &= rel_int < 1e-4;
        _aligned_free(buf);
    }
    return ok;
}

static void pin(int t, int threads) { SetThreadAffinityMask(GetCurrentThread(), 1ull << (threads <= 8 ? 2 * t : t)); }

static void fill_matrix(uint8_t* m, int f, uint64_t seed) {
    uint64_t s = seed * 0x9E3779B97F4A7C15ull | 1;
    for (int blk = 0; blk < MAT_BYTES_PER_BB; ++blk, m += block_bytes(f)) {
        for (int i = 0; i < CODE_BYTES[f]; i += 8) {
            s ^= s << 13, s ^= s >> 7, s ^= s << 17;
            memcpy(m + i, &s, 8);
        }
        for (int i = 0; i < 32; ++i) ((uint16_t*)(m + CODE_BYTES[f]))[i] = 0x2E66;  // fp16 ~0.1
    }
}

// Single thread, one w1-sized matrix (L3 resident): the compute ceiling per core.
static void compute_ceiling() {
    auto x = std::make_unique<QVec>();
    std::mt19937 rng(2);
    std::normal_distribution<float> nd;
    std::vector<float> v(DIM), signs(DIM), y(INTER);
    for (int i = 0; i < DIM; ++i) v[i] = nd(rng), signs[i] = rng() % 2 ? 1.0f : -1.0f;
    auto t0 = clk::now();
    for (int r = 0; r < 1000; ++r) rotate(v.data(), DIM, signs.data()), quantize(v.data(), DIM, *x);
    printf("activation rotate + int8, %d values: %.2f us\n", DIM,
           std::chrono::duration<double, std::micro>(clk::now() - t0).count() / 1000);
    for (int f = 0; f < NFMT; ++f) {
        const size_t bytes = size_t(MAT_BYTES_PER_BB) * block_bytes(f);
        auto* m = (uint8_t*)_aligned_malloc(bytes, 64);
        fill_matrix(m, f, 7);
        int reps = 0;
        t0 = clk::now();
        double s = 0;
        while (s < 0.3) {
            for (int rt = 0; rt < INTER / TILE; rt += 4) {
                const uint8_t* tiles[4];
                float* outs[4];
                for (int r = 0; r < 4; ++r)
                    tiles[r] = m + size_t(rt + r) * (DIM / BLOCK) * block_bytes(f), outs[r] = y.data() + 16 * (rt + r);
                GEMV4[f](tiles, DIM / BLOCK, *x, outs);
            }
            ++reps;
            s = std::chrono::duration<double>(clk::now() - t0).count();
        }
        printf("1 thread, L3-resident %-2s: %6.1f GB/s packed, %6.1f Gweight/s (%zu KB matrix)\n", NAME[f],
               bytes * reps / s / 1e9, double(INTER) * DIM * reps / s / 1e9, bytes / 1024);
        _aligned_free(m);
    }
}

struct Expert {
    int f;
    const uint8_t* w[3];  // w1, w3 [INTER, DIM]; w2 [DIM, INTER]
};

struct Layer {
    int n;
    const Expert* e[6];
    float gate[6];
    const QVec* x;
    std::atomic<int> next[3];
};

struct Barrier {
    std::atomic<int> count{0}, gen{0};
    int n;
    void wait() {
        const int g = gen.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            gen.store(g + 1, std::memory_order_release);
        } else
            while (gen.load(std::memory_order_acquire) == g) _mm_pause();
    }
};

static float silu(float v) { return v / (1 + std::exp(-v)); }

int main(int argc, char** argv) {
    const int threads = argc > 1 ? atoi(argv[1]) : 8;
    const double cpu_share = argc > 2 ? atof(argv[2]) : 0.567;  // routed calls served by the RAM tier
    double mix[NFMT] = {0.22, 0.11, 0.28, 0.39};                 // call shares by format (tier20g)
    if (argc > 3) sscanf(argv[3], "%lf,%lf,%lf,%lf", &mix[0], &mix[1], &mix[2], &mix[3]);
    const double pool_gb = argc > 4 ? atof(argv[4]) : 8;
    const int n_layers = argc > 5 ? atoi(argv[5]) : 4000;

    for (int s = 0, p = 1; s < 5; ++s, p *= 3)
        for (int v = 0; v < 256; ++v) TRIT[s][v] = uint8_t(v < 243 ? v / p % 3 : 0);
    pin(0, 1);
    if (!selftest()) return printf("SELFTEST FAILED\n"), 1;
    compute_ceiling();

    // Expert pool: each format gets pool_gb * mix bytes (at least 4 experts), far beyond the 96MB L3.
    std::vector<std::vector<Expert>> pool(NFMT);
    std::vector<std::pair<uint8_t*, size_t>> regions;
    for (int f = 0; f < NFMT; ++f) {
        if (mix[f] <= 0) continue;
        const size_t mat = size_t(MAT_BYTES_PER_BB) * block_bytes(f);
        const int cnt = std::max(4, int(pool_gb * 1e9 * mix[f] / (3 * mat)));
        auto* base = (uint8_t*)VirtualAlloc(nullptr, cnt * 3 * mat, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!base) return printf("VirtualAlloc failed\n"), 1;
        regions.push_back({base, cnt * 3 * mat});
        for (int i = 0; i < cnt; ++i) pool[f].push_back({f, {base + (3 * i) * mat, base + (3 * i + 1) * mat, base + (3 * i + 2) * mat}});
    }
    {
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t] {
                int k = 0;
                for (int f = 0; f < NFMT; ++f)
                    for (auto& e : pool[f])
                        for (int m = 0; m < 3; ++m, ++k)
                            if (k % threads == t) fill_matrix((uint8_t*)e.w[m], f, k + 11);
            });
        for (auto& t : ts) t.join();
    }

    std::mt19937_64 rng(3);
    std::normal_distribution<float> nd;
    std::vector<QVec> xs(16);
    std::vector<float> sv(DIM), signs(DIM);
    for (int i = 0; i < DIM; ++i) signs[i] = rng() % 2 ? 1.0f : -1.0f;
    for (auto& x : xs) {
        for (auto& v : sv) v = nd(rng);
        quantize(sv.data(), DIM, x);
    }
    std::binomial_distribution<int> nexp(6, cpu_share);
    std::discrete_distribution<int> fmt(mix, mix + NFMT);
    std::unique_ptr<Layer[]> layers(new Layer[n_layers]);
    double bytes = 0;
    long long calls = 0;
    for (int l = 0; l < n_layers; ++l) {
        Layer& ly = layers[l];
        ly.n = nexp(rng), ly.x = &xs[l % 16];
        for (int i = 0; i < ly.n; ++i) {
            const int f = fmt(rng);
            ly.e[i] = &pool[f][rng() % pool[f].size()], ly.gate[i] = 0.05f + 0.25f * (rng() % 1000) / 1000.0f;
            if (l >= 40) bytes += 3.0 * MAT_BYTES_PER_BB * block_bytes(f), ++calls;
        }
        for (auto& c : ly.next) c = 0;
    }

    // One layer: A) w1, w3 and SwiGLU over (expert, 32-row) items; B) rotate + int8 the hidden of each (expert,
    // 256-block); C) w2 over 64-row items, summing the gated experts. Barriers between phases; layer 40 on is timed.
    static float H[6][INTER];
    static QVec HQ[6];
    alignas(64) static float Y[DIM];
    std::vector<float> hsigns(INTER);
    for (auto& v : hsigns) v = rng() % 2 ? 1.0f : -1.0f;
    std::vector<double> phase_s(3, 0);
    Barrier bar;
    bar.n = threads;
    auto work = [&](int t) {
        pin(t, threads);
        float o[4][16];
        float* outs[4] = {o[0], o[1], o[2], o[3]};
        auto mark = clk::now();
        for (int l = 0; l < n_layers; ++l) {
            Layer& ly = layers[l];
            for (int i; (i = ly.next[0].fetch_add(1, std::memory_order_relaxed)) < ly.n * (INTER / 32);) {
                const Expert& e = *ly.e[i / (INTER / 32)];
                const int p = i % (INTER / 32);
                const size_t tb = size_t(DIM / BLOCK) * block_bytes(e.f);
                const uint8_t* tiles[4] = {e.w[0] + 2 * p * tb, e.w[1] + 2 * p * tb, e.w[0] + (2 * p + 1) * tb, e.w[1] + (2 * p + 1) * tb};
                GEMV4[e.f](tiles, DIM / BLOCK, *ly.x, outs);
                float* h = H[i / (INTER / 32)] + 32 * p;
                for (int r = 0; r < 16; ++r) h[r] = silu(o[0][r]) * o[1][r], h[16 + r] = silu(o[2][r]) * o[3][r];
            }
            bar.wait();
            auto now = clk::now();
            if (t == 0 && l >= 40) phase_s[0] += std::chrono::duration<double>(now - mark).count();
            mark = now;
            for (int i; (i = ly.next[1].fetch_add(1, std::memory_order_relaxed)) < ly.n * (INTER / BLOCK);) {
                const int e = i / (INTER / BLOCK), blk = i % (INTER / BLOCK);
                float* h = H[e] + blk * BLOCK;
                rotate(h, BLOCK, hsigns.data() + blk * BLOCK);
                quantize(h, BLOCK, HQ[e], 2 * blk);
            }
            bar.wait();
            now = clk::now();
            if (t == 0 && l >= 40) phase_s[1] += std::chrono::duration<double>(now - mark).count();
            mark = now;
            for (int i; (i = ly.next[2].fetch_add(1, std::memory_order_relaxed)) < DIM / 64;) {
                float y[64] = {};
                for (int k = 0; k < ly.n; ++k) {
                    const Expert& e = *ly.e[k];
                    const size_t tb = size_t(INTER / BLOCK) * block_bytes(e.f);
                    const uint8_t* tiles[4];
                    for (int r = 0; r < 4; ++r) tiles[r] = e.w[2] + (4 * i + r) * tb;
                    GEMV4[e.f](tiles, INTER / BLOCK, HQ[k], outs);
                    for (int r = 0; r < 64; ++r) y[r] += ly.gate[k] * o[r / 16][r % 16];
                }
                memcpy(Y + 64 * i, y, sizeof y);
            }
            bar.wait();
            now = clk::now();
            if (t == 0 && l >= 40) phase_s[2] += std::chrono::duration<double>(now - mark).count();
            mark = now;
        }
    };
    std::vector<std::thread> ts;
    for (int t = 1; t < threads; ++t) ts.emplace_back(work, t);
    work(0);
    for (auto& t : ts) t.join();

    const double total = phase_s[0] + phase_s[1] + phase_s[2], timed = n_layers - 40;
    double check = 0;
    for (float v : Y) check += v;
    printf("layers: %d threads, cpu share %.3f, mix 1b/t/2b/3b %.2f/%.2f/%.2f/%.2f, pool %.1f GB\n", threads, cpu_share,
           mix[0], mix[1], mix[2], mix[3], pool_gb);
    printf("  %.1f experts/layer, %.2f MB/layer, %.1f us/layer (A %.1f, B %.1f, C %.1f), %.1f GB/s, %.1f us/expert\n",
           calls / timed, bytes / timed / 1e6, 1e6 * total / timed, 1e6 * phase_s[0] / timed,
           1e6 * phase_s[1] / timed, 1e6 * phase_s[2] / timed, bytes / total / 1e9, 1e6 * total / calls);
    printf("  40 layers: %.2f ms/token CPU tier (checksum %.3g)\n", 40e3 * total / timed, check);
    for (auto& r : regions) VirtualFree(r.first, 0, MEM_RELEASE);
    return 0;
}
