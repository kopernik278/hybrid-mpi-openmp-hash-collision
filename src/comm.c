/*
 * comm.c - candidate routing, batching, distributed collision detection and
 * global termination (see comm.h).
 *
 * CITS3402/CITS5507 Assignment 2 (2026) - distributed birthday attack.
 * Author: Shaoming Wu (24914408)
 *
 * Termination protocol (MPI build)
 *  1. Search mode: each progress call keeps one MPI_Iallreduce(SUM) of the
 *     local collision count in flight.  Every process sees the same sequence
 *     of reduction results, so all of them observe "global >= K" at the same
 *     reduction epoch and stop together without any extra messages.
 *     Benchmark mode: each process stops after its fixed share of trials.
 *  2. comm_finish: optionally flush partial batches, then exchange per-peer
 *     message counts with a non-blocking MPI_Ialltoall while still receiving,
 *     and keep receiving until every expected message has arrived and every
 *     send has completed.  Nothing is left in flight at MPI_Finalize, and no
 *     process ever blocks while a peer may be waiting for it to receive.
 */
#include "comm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_MPI
#include <mpi.h>
#define TAG_DATA 1
/* Batches in flight to one destination.  Sends are synchronous-mode
 * (MPI_Issend): they complete only once the owner has started receiving, so
 * this cap is a credit-based flow control that throttles producers to the
 * rate at which owners can insert, and bounds MPI's unexpected-message queue. */
#define MAX_INFLIGHT 2
#endif

double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void add_collision(comm_t *c, uint64_t h, uint64_t na, uint64_t nb)
{
    if (c->ncoll == c->capcoll) {
        c->capcoll = c->capcoll ? 2 * c->capcoll : 16;
        c->coll = realloc(c->coll, c->capcoll * sizeof *c->coll);
    }
    collision_t *x = &c->coll[c->ncoll++];
    memset(x, 0, sizeof *x);
    x->t_detect = now_sec() - c->t0;
    x->hash = h;
    x->na = na;
    x->nb = nb;
    x->owner = c->rank;
}

static void insert_records(comm_t *c, const record_t *r, size_t n)
{
    double t = now_sec();
    for (size_t i = 0; i < n; i++) {
        uint64_t na, nb;
        if (table_insert(&c->table, r[i], &na, &nb) == INS_COLLISION)
            add_collision(c, r[i].key & 0x0000ffffffffffffULL, na, nb);
    }
    c->t_insert += now_sec() - t;
}

#ifdef USE_MPI
struct comm_mpi {
    record_t **cur;          /* current batch buffer per destination */
    int *fill;               /* records in cur[d] */
    long long *sent_to;      /* messages sent to each peer */
    long long *recv_from;    /* messages received from each peer */
    long long *expect_from;  /* filled by the final Ialltoall */
    MPI_Request *req;        /* in-flight sends */
    record_t **reqbuf;
    int *reqdest;
    int *idx;
    int ninf;
    int *inflight;           /* in-flight sends per destination (<= MAX_INFLIGHT) */
    record_t **freebuf;      /* recycled batch buffers */
    int nfree, capfree;
    record_t *rbuf;          /* receive buffer (one batch) */
    record_t *local;         /* records owned by this process, per route call */
    size_t caplocal;
    MPI_Request red_req;     /* outstanding Iallreduce of the collision count */
    int red_active;
    long long red_in, red_out;
};

static record_t *get_buffer(comm_t *c)
{
    comm_mpi_t *m = c->mpi;
    if (m->nfree) return m->freebuf[--m->nfree];
    record_t *b = malloc((size_t)c->batch * sizeof *b);
    if (!b) { fprintf(stderr, "rank %d: out of memory for batches\n", c->rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    return b;
}

static void put_buffer(comm_t *c, record_t *b)
{
    comm_mpi_t *m = c->mpi;
    if (m->nfree == m->capfree) {
        m->capfree = m->capfree ? 2 * m->capfree : 64;
        m->freebuf = realloc(m->freebuf, (size_t)m->capfree * sizeof *m->freebuf);
    }
    m->freebuf[m->nfree++] = b;
}

/* Retire completed sends and recycle their buffers. */
static void test_sends(comm_t *c)
{
    comm_mpi_t *m = c->mpi;
    if (!m->ninf) return;
    int outc;
    MPI_Testsome(m->ninf, m->req, &outc, m->idx, MPI_STATUSES_IGNORE);
    if (outc <= 0) return;
    for (int i = 0; i < outc; i++) {
        put_buffer(c, m->reqbuf[m->idx[i]]);
        m->inflight[m->reqdest[m->idx[i]]]--;
    }
    int k = 0;   /* compact: completed requests are now MPI_REQUEST_NULL */
    for (int i = 0; i < m->ninf; i++)
        if (m->req[i] != MPI_REQUEST_NULL) {
            m->req[k] = m->req[i];
            m->reqbuf[k] = m->reqbuf[i];
            m->reqdest[k] = m->reqdest[i];
            k++;
        }
    m->ninf = k;
}

/* Receive and insert batches that have already arrived.  At most one round
 * (p messages) per call, so a busy owner still returns to hashing and to the
 * termination test instead of being kept in this loop by fast producers. */
static void poll_recv(comm_t *c)
{
    comm_mpi_t *m = c->mpi;
    for (int n = 0; n < c->size; n++) {
        int flag, nbytes;
        MPI_Message msg;
        MPI_Status st;
        MPI_Improbe(MPI_ANY_SOURCE, TAG_DATA, MPI_COMM_WORLD, &flag, &msg, &st);
        if (!flag) break;
        MPI_Get_count(&st, MPI_BYTE, &nbytes);
        MPI_Mrecv(m->rbuf, nbytes, MPI_BYTE, &msg, MPI_STATUS_IGNORE);
        m->recv_from[st.MPI_SOURCE]++;
        c->msgs_recv++;
        c->recs_remote += (uint64_t)nbytes / sizeof(record_t);
        insert_records(c, m->rbuf, (size_t)nbytes / sizeof(record_t));
    }
}

static void send_batch(comm_t *c, int d)
{
    comm_mpi_t *m = c->mpi;
    /* Out of credit for d: wait, but keep receiving so that a peer which is
     * itself waiting on us can always make progress (no deadlock). */
    while (m->inflight[d] >= MAX_INFLIGHT) { test_sends(c); poll_recv(c); }
    const int nbytes = m->fill[d] * (int)sizeof(record_t);
    MPI_Issend(m->cur[d], nbytes, MPI_BYTE, d, TAG_DATA, MPI_COMM_WORLD, &m->req[m->ninf]);
    m->reqbuf[m->ninf] = m->cur[d];
    m->reqdest[m->ninf++] = d;
    m->inflight[d]++;
    m->sent_to[d]++;
    c->msgs_sent++;
    c->bytes_sent += (uint64_t)nbytes;
    m->cur[d] = get_buffer(c);
    m->fill[d] = 0;
}

/* Search mode: keep one Iallreduce of the local collision count in flight. */
static void update_global(comm_t *c)
{
    comm_mpi_t *m = c->mpi;
    if (c->bench || c->stop) return;
    if (m->red_active) {
        int flag;
        MPI_Test(&m->red_req, &flag, MPI_STATUS_IGNORE);
        if (!flag) return;
        m->red_active = 0;
        c->global_found = m->red_out;
        if (c->global_found >= c->K) {
            c->stop = 1;
            c->t_stop = now_sec() - c->t0;
            return;
        }
    }
    m->red_in = (long long)c->ncoll;
    MPI_Iallreduce(&m->red_in, &m->red_out, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD, &m->red_req);
    m->red_active = 1;
}
#endif /* USE_MPI */

void comm_init(comm_t *c, int rank, int size, int batch, long K, int bench,
               double flush_interval, size_t expected_local_records)
{
    memset(c, 0, sizeof *c);
    c->flush_interval = flush_interval;
    c->rank = rank;
    c->size = size;
    c->batch = batch < 1 ? 1 : batch;
    c->K = K;
    c->bench = bench;
    table_init(&c->table, expected_local_records);
#ifdef USE_MPI
    comm_mpi_t *m = calloc(1, sizeof *m);
    c->mpi = m;
    m->cur = calloc((size_t)size, sizeof *m->cur);
    m->fill = calloc((size_t)size, sizeof *m->fill);
    m->sent_to = calloc((size_t)size, sizeof *m->sent_to);
    m->recv_from = calloc((size_t)size, sizeof *m->recv_from);
    m->expect_from = calloc((size_t)size, sizeof *m->expect_from);
    const size_t maxinf = (size_t)MAX_INFLIGHT * (size_t)size;
    m->req = malloc(maxinf * sizeof *m->req);
    m->reqbuf = malloc(maxinf * sizeof *m->reqbuf);
    m->reqdest = malloc(maxinf * sizeof *m->reqdest);
    m->idx = malloc(maxinf * sizeof *m->idx);
    m->inflight = calloc((size_t)size, sizeof *m->inflight);
    for (int d = 0; d < size; d++)
        if (d != rank) m->cur[d] = get_buffer(c);
    m->rbuf = malloc((size_t)c->batch * sizeof *m->rbuf);
#endif
}

void comm_route(comm_t *c, const record_t *r, size_t n)
{
    double t = now_sec();
#ifdef USE_MPI
    comm_mpi_t *m = c->mpi;
    const uint64_t p = (uint64_t)c->size;
    if (n > m->caplocal) {
        m->caplocal = n;
        m->local = realloc(m->local, n * sizeof *m->local);
    }
    size_t nl = 0;
    for (size_t i = 0; i < n; i++) {
        const int d = (int)((r[i].key & 0x0000ffffffffffffULL) % p);   /* owner(h) = h mod p */
        if (d == c->rank) {          /* we own it: no message needed */
            m->local[nl++] = r[i];
            continue;
        }
        m->cur[d][m->fill[d]++] = r[i];
        if (m->fill[d] == c->batch) send_batch(c, d);
    }
    insert_records(c, m->local, nl);
    c->recs_local += nl;
#else
    insert_records(c, r, n);
    c->recs_local += n;
#endif
    c->t_comm += now_sec() - t;
}

void comm_progress(comm_t *c)
{
    double t = now_sec();
#ifdef USE_MPI
    comm_mpi_t *m = c->mpi;
    if (c->flush_interval > 0 && t - c->last_flush >= c->flush_interval) {
        for (int d = 0; d < c->size; d++)
            if (d != c->rank && m->fill[d] > 0) send_batch(c, d);
        c->last_flush = t;
    }
    test_sends(c);
    poll_recv(c);
    update_global(c);
#else
    if (!c->bench && !c->stop && (long)c->ncoll >= c->K) {
        c->stop = 1;
        c->global_found = (long long)c->ncoll;
        c->t_stop = now_sec() - c->t0;
    }
#endif
    c->t_comm += now_sec() - t;
}

void comm_finish(comm_t *c, int flush)
{
    double t = now_sec();
#ifdef USE_MPI
    comm_mpi_t *m = c->mpi;
    if (flush)
        for (int d = 0; d < c->size; d++)
            if (d != c->rank && m->fill[d] > 0) send_batch(c, d);
    /* A search-mode reduction may still be outstanding; finish it so the
     * collective sequence is identical on all processes. */
    if (m->red_active) {
        int done = 0;
        while (!done) { MPI_Test(&m->red_req, &done, MPI_STATUS_IGNORE); poll_recv(c); test_sends(c); }
        m->red_active = 0;
    }
    MPI_Request a2a;
    MPI_Ialltoall(m->sent_to, 1, MPI_LONG_LONG, m->expect_from, 1, MPI_LONG_LONG,
                  MPI_COMM_WORLD, &a2a);
    int a2a_done = 0;
    for (;;) {
        poll_recv(c);
        test_sends(c);
        if (!a2a_done) MPI_Test(&a2a, &a2a_done, MPI_STATUS_IGNORE);
        if (!a2a_done || m->ninf) continue;
        int all = 1;
        for (int s = 0; s < c->size; s++)
            if (m->recv_from[s] != m->expect_from[s]) { all = 0; break; }
        if (all) break;
    }
#else
    (void)flush;
#endif
    c->t_drain += now_sec() - t;
}

void comm_gather(comm_t *c, collision_t **all, size_t *nall)
{
#ifdef USE_MPI
    int nbytes = (int)(c->ncoll * sizeof(collision_t));
    int *counts = NULL, *displs = NULL;
    if (c->rank == 0) {
        counts = malloc((size_t)c->size * sizeof *counts);
        displs = malloc((size_t)c->size * sizeof *displs);
    }
    MPI_Gather(&nbytes, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    char *buf = NULL;
    if (c->rank == 0) {
        int tot = 0;
        for (int i = 0; i < c->size; i++) { displs[i] = tot; tot += counts[i]; }
        buf = malloc(tot ? (size_t)tot : 1);
        *nall = (size_t)tot / sizeof(collision_t);
    }
    MPI_Gatherv(c->coll, nbytes, MPI_BYTE, buf, counts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (c->rank == 0) {
        *all = (collision_t *)buf;
        free(counts);
        free(displs);
    } else {
        *all = NULL;
        *nall = 0;
    }
#else
    *nall = c->ncoll;
    *all = malloc((c->ncoll ? c->ncoll : 1) * sizeof(collision_t));
    memcpy(*all, c->coll, c->ncoll * sizeof(collision_t));
#endif
}

void comm_free(comm_t *c)
{
#ifdef USE_MPI
    comm_mpi_t *m = c->mpi;
    for (int d = 0; d < c->size; d++) free(m->cur[d]);
    for (int i = 0; i < m->nfree; i++) free(m->freebuf[i]);
    free(m->cur); free(m->fill); free(m->sent_to); free(m->recv_from); free(m->expect_from);
    free(m->req); free(m->reqbuf); free(m->reqdest); free(m->idx); free(m->inflight); free(m->freebuf); free(m->rbuf); free(m->local);
    free(m);
#endif
    table_free(&c->table);
    free(c->coll);
}
