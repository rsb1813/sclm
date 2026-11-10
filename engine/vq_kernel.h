// AVX-512 VBMI table-lookup GEMV over vector-quantized weights (256 codes per 16, 8 or 4 columns, or 64 per 2).
#pragma once
#include <immintrin.h>
#include <cstdint>

// A matrix [N, K] with one code per D columns is N/256 row tiles; a tile is K/D steps, a step holding the codes of its
// 256 rows for columns D*j .. D*j+D-1: one byte per row with 256 codes, or 4 x 48 bytes with 64 codes (row p of a
// 64-row chunk at bit 6p). Per token, tables[j][c] = <codebook[c], x[D*j ..]> as int8 on one common scale, so a
// row's product is the sum of one table entry per step.
constexpr int TILE = 256;

struct Tables {
    int groups = 0, codes = 0;
    float scale = 0;
    int8_t* t = nullptr;  // [groups][codes]
    float* tmp = nullptr;
};

// cb is [D][codes]: dimension-major, so one input value scales a whole column of entries.
static void build_tables(const float* cb, int d, const float* x, int k, Tables& tb) {
    const int g = k / d, codes = tb.codes;
    __m512 mx = _mm512_setzero_ps();
    for (int j = 0; j < g; ++j)
        for (int v = 0; v < codes; v += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int i = 0; i < d; ++i)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(x[j * d + i]), _mm512_load_ps(cb + i * codes + v), acc);
            _mm512_store_ps(tb.tmp + j * codes + v, acc);
            mx = _mm512_max_ps(mx, _mm512_abs_ps(acc));
        }
    tb.groups = g, tb.scale = _mm512_reduce_max_ps(mx) / 127.0f;
    const __m512 inv = _mm512_set1_ps(1.0f / tb.scale);
    for (int n = 0; n < g * codes; n += 16)
        _mm_store_si128((__m128i*)(tb.t + n), _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_load_ps(tb.tmp + n), inv))));
}

// y[0..255] = sums over the steps of lookup(step, r), the int8 table entries of rows 64r .. 64r+63, times the scale.
template <class Lookup>
static void sum_tile(int groups, float scale, float* y, Lookup lookup) {
    __m512i a[8], s[16];
    for (auto& v : a) v = _mm512_setzero_si512();
    for (auto& v : s) v = _mm512_setzero_si512();
    for (int j = 0; j < groups; ++j) {
        for (int r = 0; r < 4; ++r) {
            const __m512i v = lookup(j, r);
            a[2 * r] = _mm512_add_epi16(a[2 * r], _mm512_cvtepi8_epi16(_mm512_castsi512_si256(v)));
            a[2 * r + 1] = _mm512_add_epi16(a[2 * r + 1], _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(v, 1)));
        }
        if ((j & 255) == 255 || j == groups - 1)  // 256 entries of at most 127 fit int16
            for (int r = 0; r < 8; ++r) {
                s[2 * r] = _mm512_add_epi32(s[2 * r], _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a[r])));
                s[2 * r + 1] = _mm512_add_epi32(s[2 * r + 1], _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a[r], 1)));
                a[r] = _mm512_setzero_si512();
            }
    }
    const __m512 sc = _mm512_set1_ps(scale);
    for (int r = 0; r < 16; ++r) _mm512_storeu_ps(y + 16 * r, _mm512_mul_ps(_mm512_cvtepi32_ps(s[r]), sc));
}

// One tile's table sums (row scales are applied by the caller). The codes need no alignment and no padding.
static void gemv_tile(const uint8_t* codes, const Tables& tb, float* y) {
    if (tb.codes == 256) {
        sum_tile(tb.groups, tb.scale, y, [&](int j, int r) {
            const __m512i* t = (const __m512i*)(tb.t + j * 256);
            const __m512i c = _mm512_loadu_si512(codes + j * TILE + 64 * r);
            return _mm512_mask_blend_epi8(_mm512_movepi8_mask(c), _mm512_permutex2var_epi8(_mm512_load_si512(t), c, _mm512_load_si512(t + 1)),
                                          _mm512_permutex2var_epi8(_mm512_load_si512(t + 2), c, _mm512_load_si512(t + 3)));
        });
        return;
    }
    // 64 codes: move the 6 bytes of every 8 codes into a qword, then cut a byte at each multiple of 6 bits; the table
    // lookup reads only the low 6 bits of an index, so the 2 bits of the next code above them need no masking
    alignas(64) uint8_t spread[64];
    for (int i = 0; i < 64; ++i) spread[i] = uint8_t(i % 8 < 6 ? 6 * (i / 8) + i % 8 : 0);
    const __m512i sp = _mm512_load_si512(spread), cut = _mm512_set1_epi64(0x2a241e18120c0600);
    sum_tile(tb.groups, tb.scale, y, [&](int j, int r) {
        const __m512i chunk = _mm512_maskz_loadu_epi8(0xffffffffffffull, codes + j * 192 + 48 * r);  // 48 bytes, no read past them
        return _mm512_permutexvar_epi8(_mm512_multishift_epi64_epi8(cut, _mm512_permutexvar_epi8(sp, chunk)), _mm512_load_si512(tb.t + j * 64));
    });
}
