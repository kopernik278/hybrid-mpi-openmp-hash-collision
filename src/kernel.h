/*
 * kernel.h - the per-trial toy_hash search kernel.
 *
 * CITS3402/CITS5507 Assignment 2 (2026) - distributed birthday attack.
 * Author: Shaoming Wu (24914408)
 *
 * toy_hash is FNV-1a over the whole file followed by a 64-bit finaliser.
 * The nonce sits at byte 16, so the FNV state of the 16-byte prefix is
 * computed once.  Each trial then hashes the 16 nonce digits and the
 * remaining file ("suffix", 64 KiB - 920 KiB).
 *
 *  - FULL kernel (default): the suffix is hashed byte by byte, LANES
 *    independent nonces at a time so that the multiply latency of one
 *    FNV chain is hidden behind the others (instruction-level parallelism).
 *
 *  - FAST kernel (--fast): exact algebraic shortcut.  Because
 *    h ^ c == h + c - 2(h & c) and the low byte of an FNV state only depends
 *    on the low byte of the previous state, the whole suffix map is
 *        S(h) = P^n * h + C[h mod 256]  (mod 2^64)
 *    with a 256-entry table C computed once.  A trial costs O(16) steps
 *    instead of O(file size).  Results are identical to the full kernel.
 */
#ifndef KERNEL_H
#define KERNEL_H

#include <stddef.h>
#include <stdint.h>

#include "pdf_io.h"

#ifndef LANES
#define LANES 8                       /* nonces hashed together per work unit */
#endif
#define HASH_MASK 0x0000ffffffffffffULL

typedef struct {
    uint64_t pre;                     /* FNV state after bytes [0, nonce_off) */
    const unsigned char *suf;         /* bytes after the nonce field */
    size_t suf_len;
    uint64_t Pn;                      /* FNV_PRIME^suf_len mod 2^64 */
    uint64_t C[256];                  /* affine offsets for the FAST kernel */
} kfile_t;

typedef struct {
    kfile_t f[2];                     /* f[0] = file A, f[1] = file B */
    int fast;
} kernel_t;

/* Prepare the kernel for files a and b (student id must already be set). */
void kernel_init(kernel_t *k, const pdf_file_t *a, const pdf_file_t *b, int fast);

/* Hash nonces nonce0 .. nonce0+LANES-1 of `file` (0 = A, 1 = B).
 * out[j] receives the 48-bit toy_hash for nonce0 + j. */
void kernel_hash_unit(const kernel_t *k, int file, uint64_t nonce0, uint64_t out[LANES]);

/* Compare the kernel against the reference toy_hash for a few nonces.
 * Returns the number of mismatches (0 = OK). */
int kernel_selftest(const kernel_t *k, const pdf_file_t *a, const pdf_file_t *b);

#endif
