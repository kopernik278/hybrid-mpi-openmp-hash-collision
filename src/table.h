/*
 * table.h - one process's slice of the distributed collision table.
 *
 * CITS3402/CITS5507 Assignment 2 (2026) - distributed birthday attack.
 * Author: Shaoming Wu (24914408)
 *
 * Open addressing with linear probing.  A slot stores the 48-bit hash, the
 * file (A/B) of the first record seen with that hash, a "matched" flag and
 * the nonce.  Only the owning process touches its table (in the hybrid build
 * only the master thread), so no locking is required.
 */
#ifndef TABLE_H
#define TABLE_H

#include <stddef.h>
#include <stdint.h>

/* A candidate record as it travels between processes (16 bytes). */
typedef struct {
    uint64_t key;    /* bits 0-47: toy_hash, bit 48: file (0 = A, 1 = B) */
    uint64_t nonce;
} record_t;

#define REC_FILE_BIT (1ULL << 48)

typedef struct {
    uint64_t *key;   /* 0 = empty, else hash | file<<48 | OCC | MATCHED */
    uint64_t *nonce;
    size_t cap;      /* power of two */
    size_t count;
    int log2cap;
} table_t;

typedef enum {
    INS_NEW = 0,       /* hash not seen before: stored */
    INS_SAME_FILE,     /* hash already seen from the same file: ignored */
    INS_COLLISION,     /* first A/B match for this hash: reported once */
    INS_ALREADY        /* hash already produced a collision: ignored */
} ins_result_t;

void table_init(table_t *t, size_t expected);
void table_free(table_t *t);

/* Insert a record.  On INS_COLLISION *na and *nb receive the A and B nonces. */
ins_result_t table_insert(table_t *t, record_t r, uint64_t *na, uint64_t *nb);

#endif
