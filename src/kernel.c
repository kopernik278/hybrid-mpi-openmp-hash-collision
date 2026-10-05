/*
 * kernel.c - the per-trial toy_hash search kernel (see kernel.h).
 *
 * CITS3402/CITS5507 Assignment 2 (2026) - distributed birthday attack.
 * Author: Shaoming Wu (24914408)
 */
#include "kernel.h"

#include <stdio.h>
#include <string.h>

#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

static const unsigned char HEX[] = "0123456789abcdef";

static inline uint64_t fmix48(uint64_t h)
{
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h & HASH_MASK;
}

/* Run LANES FNV-1a chains over the same byte string.  The inner loop over
 * lanes has no dependencies, so the compiler can interleave (or vectorise)
 * the chains; one chain alone is bound by the 3-4 cycle multiply latency. */
static void fnv_lanes(uint64_t h[LANES], const unsigned char *s, size_t n)
{
    uint64_t x[LANES];
    for (int j = 0; j < LANES; j++) x[j] = h[j];
    for (size_t i = 0; i < n; i++) {
        const uint64_t c = s[i];
        for (int j = 0; j < LANES; j++) x[j] = (x[j] ^ c) * FNV_PRIME;
    }
    for (int j = 0; j < LANES; j++) h[j] = x[j];
}

static uint64_t pow_mod64(uint64_t b, uint64_t e)
{
    uint64_t r = 1;
    while (e) {
        if (e & 1) r *= b;
        b *= b;
        e >>= 1;
    }
    return r;
}

static void kfile_init(kfile_t *kf, const pdf_file_t *f, int fast)
{
    uint64_t h = FNV_OFFSET;
    for (size_t i = 0; i < f->nonce_off; i++) h = (h ^ f->data[i]) * FNV_PRIME;
    kf->pre = h;
    kf->suf = f->data + f->nonce_off + NONCE_HEX_LEN;
    kf->suf_len = f->len - f->nonce_off - NONCE_HEX_LEN;
    kf->Pn = pow_mod64(FNV_PRIME, kf->suf_len);
    if (!fast) return;
    /* C[r] = S(r) - P^n r, evaluated LANES residues at a time. */
    for (int r0 = 0; r0 < 256; r0 += LANES) {
        uint64_t v[LANES];
        for (int j = 0; j < LANES; j++) v[j] = (uint64_t)(r0 + j);
        fnv_lanes(v, kf->suf, kf->suf_len);
        for (int j = 0; j < LANES; j++) kf->C[r0 + j] = v[j] - kf->Pn * (uint64_t)(r0 + j);
    }
}

void kernel_init(kernel_t *k, const pdf_file_t *a, const pdf_file_t *b, int fast)
{
    memset(k, 0, sizeof *k);
    k->fast = fast;
    kfile_init(&k->f[0], a, fast);
    kfile_init(&k->f[1], b, fast);
}

void kernel_hash_unit(const kernel_t *k, int file, uint64_t nonce0, uint64_t out[LANES])
{
    const kfile_t *kf = &k->f[file];
    uint64_t h[LANES];
    for (int j = 0; j < LANES; j++) h[j] = kf->pre;

    /* 16 hex digits of the nonce, most significant first. */
    for (int d = 0; d < NONCE_HEX_LEN; d++) {
        const int sh = 60 - 4 * d;
        for (int j = 0; j < LANES; j++) {
            const uint64_t c = HEX[((nonce0 + (uint64_t)j) >> sh) & 0xf];
            h[j] = (h[j] ^ c) * FNV_PRIME;
        }
    }

    if (k->fast) {
        for (int j = 0; j < LANES; j++) h[j] = kf->Pn * h[j] + kf->C[h[j] & 0xff];
    } else {
        fnv_lanes(h, kf->suf, kf->suf_len);
    }
    for (int j = 0; j < LANES; j++) out[j] = fmix48(h[j]);
}

int kernel_selftest(const kernel_t *k, const pdf_file_t *a, const pdf_file_t *b)
{
    static const uint64_t probes[] = {0, 0x0123456789abcdefULL, 0xfffffffffffffff0ULL,
                                      (7ULL << 40) | 12345ULL};
    int bad = 0;
    for (int file = 0; file < 2; file++) {
        const pdf_file_t *f = file ? b : a;
        for (size_t p = 0; p < sizeof probes / sizeof probes[0]; p++) {
            uint64_t out[LANES];
            kernel_hash_unit(k, file, probes[p], out);
            for (int j = 0; j < LANES; j += (LANES > 1 ? LANES - 1 : 1)) { /* first and last lane */
                uint64_t ref;
                pdf_verify_pair(f, f, probes[p] + (uint64_t)j, probes[p] + (uint64_t)j,
                                out[j], &ref, NULL);
                if (ref != out[j]) {
                    fprintf(stderr, "selftest: file %d nonce %016llx kernel %012llx ref %012llx\n",
                            file, (unsigned long long)(probes[p] + (uint64_t)j),
                            (unsigned long long)out[j], (unsigned long long)ref);
                    bad++;
                }
            }
        }
    }
    return bad;
}
