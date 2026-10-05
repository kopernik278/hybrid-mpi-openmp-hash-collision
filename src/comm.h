/*
 * comm.h - candidate routing, batching, distributed collision detection and
 * global termination.
 *
 * CITS3402/CITS5507 Assignment 2 (2026) - distributed birthday attack.
 * Author: Shaoming Wu (24914408)
 *
 * Every 48-bit hash h has exactly one owner process, owner(h) = h mod p.
 * Records (h, file, nonce) are appended to a per-destination buffer and sent
 * with MPI_Isend once the buffer holds `batch` records (or, optionally, when
 * the flush interval expires, which bounds the detection latency of a
 * candidate); records owned by the local process are inserted directly.  The owner inserts received records
 * into its slice of the collision table, so collisions are detected where
 * the hash lives and each hash can be reported at most once globally.
 *
 * Without USE_MPI (serial / OpenMP builds) p = 1 and every record is local.
 *
 * All functions are called by one thread only (the OpenMP master thread in
 * the hybrid build), which is why MPI_THREAD_FUNNELED is sufficient.
 */
#ifndef COMM_H
#define COMM_H

#include <stddef.h>
#include <stdint.h>

#include "table.h"

typedef struct {
    double t_detect;       /* seconds since search start (owner's clock) */
    uint64_t hash, na, nb;
    int32_t owner, pad;
} collision_t;

typedef struct comm_mpi comm_mpi_t;   /* MPI-only state, opaque */

typedef struct {
    int rank, size;
    int batch;             /* records per message */
    long K;                /* collisions requested (search mode) */
    int bench;             /* benchmark mode: never stop early */
    double t0;             /* search start time */
    double flush_interval; /* send partial batches at least this often (s), 0 = never */
    double last_flush;

    table_t table;
    collision_t *coll;
    size_t ncoll, capcoll;

    int stop;              /* search mode: global target reached */
    long long global_found;
    double t_stop;         /* time the stop decision was made */

    /* statistics */
    uint64_t msgs_sent, bytes_sent, msgs_recv, recs_remote, recs_local;
    double t_comm;         /* time spent in comm_* calls (master thread) */
    double t_insert;       /* subset of t_comm spent inserting into the table */
    double t_drain;        /* comm_finish: final flush + drain (includes waiting for peers) */

    comm_mpi_t *mpi;
} comm_t;

double now_sec(void);

void comm_init(comm_t *c, int rank, int size, int batch, long K, int bench,
               double flush_interval, size_t expected_local_records);
/* Route n records: buffer per destination (send when full) or insert locally. */
void comm_route(comm_t *c, const record_t *r, size_t n);
/* Receive and insert pending batches, complete sends, update the global count. */
void comm_progress(comm_t *c);
/* Flush partially filled batches (if flush) and drain every in-flight message. */
void comm_finish(comm_t *c, int flush);
/* Gather all collisions found on every process to rank 0 (collective). */
void comm_gather(comm_t *c, collision_t **all, size_t *nall);
void comm_free(comm_t *c);

#endif
