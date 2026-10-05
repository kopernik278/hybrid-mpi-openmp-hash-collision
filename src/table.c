/*
 * table.c - open-addressing collision table (see table.h).
 *
 * CITS3402/CITS5507 Assignment 2 (2026) - distributed birthday attack.
 * Author: Shaoming Wu (24914408)
 */
#include "table.h"

#include <stdio.h>
#include <stdlib.h>

#define OCC (1ULL << 63)
#define MATCHED (1ULL << 62)
#define HMASK 0x0000ffffffffffffULL

/* owner(h) = h mod p uses the low bits of h, so the slot index must not:
 * a multiplicative (Fibonacci) hash of h selects the top log2cap bits. */
static inline size_t slot_of(const table_t *t, uint64_t h)
{
    return (size_t)((h * 0x9e3779b97f4a7c15ULL) >> (64 - t->log2cap));
}

static void alloc_slots(table_t *t, int log2cap)
{
    t->log2cap = log2cap;
    t->cap = (size_t)1 << log2cap;
    t->count = 0;
    t->key = calloc(t->cap, sizeof *t->key);
    t->nonce = malloc(t->cap * sizeof *t->nonce);
    if (!t->key || !t->nonce) {
        fprintf(stderr, "table: cannot allocate %zu slots\n", t->cap);
        exit(1);
    }
}

void table_init(table_t *t, size_t expected)
{
    int lg = 10;
    while (((size_t)1 << lg) * 6 / 10 < expected) lg++;   /* load factor <= 0.6 */
    alloc_slots(t, lg);
}

void table_free(table_t *t)
{
    free(t->key);
    free(t->nonce);
    t->key = t->nonce = NULL;
}

/* Double the capacity (only if the a-priori size estimate was too small). */
static void grow(table_t *t)
{
    uint64_t *ok = t->key, *on = t->nonce;
    size_t oc = t->cap, cnt = t->count;
    alloc_slots(t, t->log2cap + 1);
    for (size_t i = 0; i < oc; i++) {
        if (!ok[i]) continue;
        size_t s = slot_of(t, ok[i] & HMASK);
        while (t->key[s]) s = (s + 1) & (t->cap - 1);
        t->key[s] = ok[i];
        t->nonce[s] = on[i];
    }
    t->count = cnt;
    free(ok);
    free(on);
}

ins_result_t table_insert(table_t *t, record_t r, uint64_t *na, uint64_t *nb)
{
    const uint64_t h = r.key & HMASK;
    const uint64_t file = r.key & (1ULL << 48);
    size_t s = slot_of(t, h);
    for (;;) {
        const uint64_t k = t->key[s];
        if (!k) break;
        if ((k & HMASK) == h) {
            if (k & MATCHED) return INS_ALREADY;
            if ((k & (1ULL << 48)) == file) return INS_SAME_FILE;
            /* Different files, same hash: a genuine A/B collision. */
            t->key[s] = k | MATCHED;
            if (file) { *na = t->nonce[s]; *nb = r.nonce; }
            else      { *na = r.nonce; *nb = t->nonce[s]; }
            return INS_COLLISION;
        }
        s = (s + 1) & (t->cap - 1);
    }
    t->key[s] = h | file | OCC;
    t->nonce[s] = r.nonce;
    if (++t->count * 10 > t->cap * 7) grow(t);
    return INS_NEW;
}
