/*
 * birthday.c - distributed birthday attack on the 48-bit toy_hash.
 *
 * CITS3402/CITS5507 Assignment 2 (2026).
 * Author: Shaoming Wu (24914408)
 *
 * One source file, four builds (see Makefile):
 *   birthday_serial  no MPI, no OpenMP
 *   birthday_omp     OpenMP only (one process, shared memory)
 *   birthday_mpi     pure MPI            (-DUSE_MPI)
 *   birthday_hybrid  MPI + OpenMP        (-DUSE_MPI -fopenmp, MPI_THREAD_FUNNELED)
 *
 * Search loop (per MPI process):
 *   Work is split into "units" of LANES consecutive nonces for one file;
 *   unit u hashes file (u & 1) with nonces  base | (u>>1)*LANES + j,  where
 *   base = seed<<56 | rank<<40, so no two processes or threads ever try the
 *   same (file, nonce) and A and B are sampled equally.
 *   Each iteration the OpenMP team hashes a chunk of units into one of two
 *   record buffers (schedule(dynamic)).  Meanwhile the master thread routes
 *   the previous chunk to the hash owners, receives batches from other
 *   processes, inserts them into its table slice and tests for termination.
 *   Only the master thread calls MPI, hence MPI_THREAD_FUNNELED.
 */
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "comm.h"
#include "kernel.h"
#include "pdf_io.h"
#include "table.h"

#ifndef DEFAULT_STUDENT_ID
#define DEFAULT_STUDENT_ID "00000000"
#endif

#if defined(USE_MPI) && defined(_OPENMP)
#define IMPL_NAME "hybrid"
#elif defined(USE_MPI)
#define IMPL_NAME "mpi"
#elif defined(_OPENMP)
#define IMPL_NAME "omp"
#else
#define IMPL_NAME "serial"
#endif

typedef struct {
    const char *file_a, *file_b, *sid, *outdir;
    long K;
    int batch;
    unsigned long long bench;   /* 0 = search mode, else total trials */
    int fast;
    int write;
    long chunk;                 /* units per thread per iteration, 0 = auto */
    double flush_ms;            /* max age of a partial batch, 0 = size-only batching */
    unsigned seed;
    int quiet;
} opts_t;

static int g_rank = 0, g_size = 1;

static void die(const char *msg)
{
    if (g_rank == 0) fprintf(stderr, "error: %s\n", msg);
#ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    exit(1);
}

static void usage(const char *prog)
{
    if (g_rank != 0) return;
    fprintf(stderr,
            "usage: %s -a FILE_A -b FILE_B [options]\n"
            "  -s, --student ID     8-digit student number inserted in both files (default %s)\n"
            "  -K, --collisions K   number of distinct collision hashes to find (default 1)\n"
            "  -B, --batch N        candidate records per MPI message (default 1024)\n"
            "      --flush-ms T     also send partial batches every T ms (default 100, 0 = off)\n"
            "      --benchmark N    fixed-work mode: perform N trials in total, never stop early\n"
            "      --fast           use the exact affine suffix kernel (O(1) per trial)\n"
            "  -o, --outdir DIR     directory for solved files (default solved)\n"
            "      --no-write       do not write solved files\n"
            "      --chunk U        hashing units (of %d nonces) per thread per iteration\n"
            "      --seed S         nonce-space seed 0..255 (default 0)\n"
            "  -q, --quiet          only print the JSON summary line\n"
            "OpenMP threads are taken from OMP_NUM_THREADS.\n",
            prog, DEFAULT_STUDENT_ID, LANES);
}

static void parse(int argc, char **argv, opts_t *o)
{
    memset(o, 0, sizeof *o);
    o->sid = DEFAULT_STUDENT_ID;
    o->outdir = "solved";
    o->K = 1;
    o->batch = 1024;
    o->write = 1;
    o->flush_ms = 100;
    static struct option lo[] = {
        {"file-a", required_argument, 0, 'a'}, {"file-b", required_argument, 0, 'b'},
        {"student", required_argument, 0, 's'}, {"collisions", required_argument, 0, 'K'},
        {"batch", required_argument, 0, 'B'}, {"benchmark", required_argument, 0, 1},
        {"fast", no_argument, 0, 2}, {"outdir", required_argument, 0, 'o'},
        {"no-write", no_argument, 0, 3}, {"chunk", required_argument, 0, 4},
        {"seed", required_argument, 0, 5}, {"quiet", no_argument, 0, 'q'},
        {"flush-ms", required_argument, 0, 6},
        {"help", no_argument, 0, 'h'}, {0, 0, 0, 0}};
    int ch;
    while ((ch = getopt_long(argc, argv, "a:b:s:K:B:o:qh", lo, NULL)) != -1) {
        switch (ch) {
        case 'a': o->file_a = optarg; break;
        case 'b': o->file_b = optarg; break;
        case 's': o->sid = optarg; break;
        case 'K': o->K = atol(optarg); break;
        case 'B': o->batch = atoi(optarg); break;
        case 1: o->bench = strtoull(optarg, NULL, 10); break;
        case 2: o->fast = 1; break;
        case 'o': o->outdir = optarg; break;
        case 3: o->write = 0; break;
        case 4: o->chunk = atol(optarg); break;
        case 5: o->seed = (unsigned)atoi(optarg) & 0xff; break;
        case 6: o->flush_ms = atof(optarg); break;
        case 'q': o->quiet = 1; break;
        default: usage(argv[0]); die("bad arguments");
        }
    }
    if (!o->file_a || !o->file_b) { usage(argv[0]); die("both -a and -b are required"); }
    if (o->K < 1) die("K must be >= 1");
    if (o->batch < 1) die("batch size must be >= 1");
}

/* Rank 0 reads the file and broadcasts it, so 100s of processes do not all
 * hit the parallel file system. */
static void load_bcast(pdf_file_t *f, const char *path)
{
    long long len = 0;
    unsigned char *data = NULL;
    if (g_rank == 0) {
        if (pdf_load(f, path) != 0) die("cannot load input file");
        len = (long long)f->len;
        data = f->data;
    }
#ifdef USE_MPI
    MPI_Bcast(&len, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    if (g_rank != 0) data = malloc((size_t)len);
    MPI_Bcast(data, (int)len, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (g_rank != 0 && pdf_from_buffer(f, path, data, (size_t)len) != 0) die("bad input file");
#else
    (void)len;
    (void)data;
#endif
}

#ifdef USE_MPI
static inline int thread_id(void)
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}
#endif

static void compute_unit(const kernel_t *k, uint64_t unit, uint64_t nonce_base, record_t *out)
{
    const int file = (int)(unit & 1);
    const uint64_t nonce0 = nonce_base | ((unit >> 1) * LANES);
    uint64_t h[LANES];
    kernel_hash_unit(k, file, nonce0, h);
    const uint64_t fb = file ? REC_FILE_BIT : 0;
    for (int j = 0; j < LANES; j++) {
        out[j].key = h[j] | fb;
        out[j].nonce = nonce0 + (uint64_t)j;
    }
}

/* Rank-local trial index at which (nonce, file) was generated. */
static uint64_t trial_index(uint64_t nonce, int file)
{
    const uint64_t c = nonce & ((1ULL << 40) - 1);
    return (2 * (c / LANES) + (uint64_t)file) * LANES + c % LANES;
}

static int cmp_detect(const void *x, const void *y)
{
    const collision_t *a = x, *b = y;
    return (a->t_detect > b->t_detect) - (a->t_detect < b->t_detect);
}

static int cmp_u64(const void *x, const void *y)
{
    const uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y;
    return (a > b) - (a < b);
}

/* "1_alpha_a.pdf" -> "1_alpha" */
static void pair_name(const char *path, char *out, size_t n)
{
    const char *b = strrchr(path, '/');
    b = b ? b + 1 : path;
    snprintf(out, n, "%s", b);
    char *dot = strrchr(out, '.');
    if (dot) *dot = 0;
    size_t l = strlen(out);
    if (l > 2 && out[l - 2] == '_') out[l - 2] = 0;
}

int main(int argc, char **argv)
{
#if defined(USE_MPI) && defined(_OPENMP)
    int provided = 0;
#endif
#ifdef USE_MPI
#ifdef _OPENMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Required MPI thread support (MPI_THREAD_FUNNELED) is unavailable.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#else
    MPI_Init(&argc, &argv);
#endif
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    if (g_size > 65536) die("at most 65536 processes are supported");
#endif
    opts_t o;
    parse(argc, argv, &o);

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    int nodes = 1;
#ifdef USE_MPI
    {
        MPI_Comm node;
        int lr, one;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &node);
        MPI_Comm_rank(node, &lr);
        one = (lr == 0);
        MPI_Allreduce(&one, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Comm_free(&node);
    }
#endif

    /* ---------------- setup (not timed) ---------------- */
    pdf_file_t fa, fb;
    load_bcast(&fa, o.file_a);
    load_bcast(&fb, o.file_b);
    if (pdf_set_student_id(&fa, o.sid) || pdf_set_student_id(&fb, o.sid))
        die("student id must be exactly 8 digits");

    kernel_t k;
    kernel_init(&k, &fa, &fb, o.fast);
    int bad = 0;
    if (g_rank == 0) bad = kernel_selftest(&k, &fa, &fb);
#ifdef USE_MPI
    MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
    if (bad) die("kernel self-test against reference toy_hash failed");

    /* Units per rank: benchmark = fixed share of N (even, so A and B are
     * balanced); search = unbounded. */
    uint64_t unit_limit = UINT64_MAX;
    if (o.bench) {
        uint64_t share = o.bench / (uint64_t)g_size + ((uint64_t)g_rank < o.bench % (uint64_t)g_size);
        uint64_t units = (share + LANES - 1) / LANES;
        unit_limit = (units + 1) & ~1ULL;
    }
    /* Iteration size: ~2^24 byte-steps per thread (roughly 5-50 ms of hashing)
     * and at least 8 units, so that the dynamic schedule can absorb the
     * master's communication work and uneven cores, while the master still
     * services MPI many times per second. */
    long per_thread = o.chunk;
    if (per_thread <= 0) {
        const double cost = o.fast ? 48.0 : (double)k.f[0].suf_len + 16.0;
        per_thread = (long)((double)(1 << 24) / (LANES * cost));
        if (per_thread < 8) per_thread = 8;
        if (per_thread > 16384) per_thread = 16384;   /* record buffers <= 2 MiB per thread */
    }
    const long U = per_thread * threads;
    const long poke_every = per_thread / 16 > 1 ? per_thread / 16 : 1;   /* ~16 MPI polls per iteration */
    (void)poke_every;
    record_t *rec[2];
    rec[0] = malloc((size_t)U * LANES * sizeof(record_t));
    rec[1] = malloc((size_t)U * LANES * sizeof(record_t));

    /* Expected records: N for a benchmark; for a search the birthday bound
     * 2^25 Gamma(K+1/2)/Gamma(K), plus a margin for the spread of the K-th
     * collision (relative spread ~ 1/sqrt(K)).  The table grows if needed. */
    double expect = (double)o.bench;
    if (!o.bench)
        expect = 33554432.0 * exp(lgamma((double)o.K + 0.5) - lgamma((double)o.K)) *
                 (1.0 + 0.6 / sqrt((double)o.K));
    comm_t comm;
    comm_init(&comm, g_rank, g_size, o.batch, o.K, o.bench != 0, o.flush_ms / 1000.0,
              (size_t)(expect / g_size) + 1024);
    const uint64_t nonce_base = ((uint64_t)o.seed << 56) | ((uint64_t)g_rank << 40);

    if (g_rank == 0 && !o.quiet) {
        printf("birthday_%s: %d process(es) on %d node(s) x %d thread(s), mode=%s, K=%ld, batch=%d, flush=%gms, kernel=%s\n",
               IMPL_NAME, g_size, nodes, threads, o.bench ? "benchmark" : "search", o.K, o.batch,
               o.flush_ms, o.fast ? "fast" : "full");
#if defined(USE_MPI) && defined(_OPENMP)
        printf("MPI thread support: requested MPI_THREAD_FUNNELED, provided level %d\n", provided);
#endif
        printf("files: %s, %s (%zu / %zu bytes), student id %s\n", o.file_a, o.file_b, fa.len,
               fb.len, o.sid);
        fflush(stdout);
    }

    /* ---------------- timed search ---------------- */
    uint64_t unit_next = 0, base_cur = 0, trials = 0;
    long n_cur, n_prev = 0;
    int cur = 0, stop = 0;
    n_cur = (long)(unit_limit < (uint64_t)U ? unit_limit : (uint64_t)U);
    unit_next = (uint64_t)n_cur;
    if (n_cur == 0) stop = 1;

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    const double t0 = now_sec();
    comm.t0 = comm.last_flush = t0;

#pragma omp parallel default(shared)
    {
        long pokes = 0;   /* private: used by the master only */
        (void)pokes;
        while (!stop) {
            /* master: route previous chunk + service MPI, overlapped with hashing */
#pragma omp master
            {
                if (n_prev) comm_route(&comm, rec[cur ^ 1], (size_t)n_prev * LANES);
                comm_progress(&comm);
            }
#pragma omp for schedule(dynamic, 1)
            for (long u = 0; u < n_cur; u++) {
                compute_unit(&k, base_cur + (uint64_t)u, nonce_base, rec[cur] + u * LANES);
#ifdef USE_MPI
                /* The master also services MPI between its own units: MPI
                 * libraries without asynchronous progress would otherwise
                 * stall a rendezvous transfer to/from this process for a
                 * whole iteration. */
                if (thread_id() == 0 && ++pokes % poke_every == 0) comm_progress(&comm);
#endif
            }
            /* implicit barrier: chunk complete */
#pragma omp master
            {
                trials += (uint64_t)n_cur * LANES;
                n_prev = n_cur;
                cur ^= 1;
                if (!o.bench && comm.stop) {
                    stop = 1;
                } else {
                    const uint64_t left = unit_limit - unit_next;
                    n_cur = (long)(left < (uint64_t)U ? left : (uint64_t)U);
                    base_cur = unit_next;
                    unit_next += (uint64_t)n_cur;
                    if (n_cur == 0) stop = 1;
                }
            }
#pragma omp barrier
        }
    }
    if (o.bench && n_prev) comm_route(&comm, rec[cur ^ 1], (size_t)n_prev * LANES);
    comm_finish(&comm, o.bench != 0);

    collision_t *all;
    size_t nall;
    comm_gather(&comm, &all, &nall);
    const double elapsed = now_sec() - t0;
    /* ---------------- end of timed region ---------------- */

    double el_max = elapsed, tc_max = comm.t_comm, tc_sum = comm.t_comm, ti_max = comm.t_insert;
    double td_max = comm.t_drain;
    unsigned long long st[5] = {trials, comm.msgs_sent, comm.bytes_sent, comm.recs_remote,
                                comm.recs_local};
#ifdef USE_MPI
    unsigned long long st_sum[5];
    MPI_Reduce(&elapsed, &el_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm.t_comm, &tc_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm.t_comm, &tc_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm.t_insert, &ti_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm.t_drain, &td_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(st, st_sum, 5, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    memcpy(st, st_sum, sizeof st);
#endif

    int rc = 0;
    if (g_rank == 0) {
        qsort(all, nall, sizeof *all, cmp_detect);
        const size_t want = o.bench ? nall : (size_t)(nall < (size_t)o.K ? nall : (size_t)o.K);
        size_t verified = 0, distinct = 1;
        for (size_t i = 0; i < want; i++)
            for (size_t j = 0; j < i; j++)
                if (all[i].hash == all[j].hash) distinct = 0;

        /* Verify each reported collision independently (full reference hash). */
        for (size_t i = 0; i < want; i++) {
            uint64_t ha, hb;
            int ok = pdf_verify_pair(&fa, &fb, all[i].na, all[i].nb, all[i].hash, &ha, &hb);
            verified += (size_t)ok;
            if (!o.quiet)
                printf("collision %zu: hash=%012llx nonceA=%016llx nonceB=%016llx owner=%d t=%.3fs %s\n",
                       i + 1, (unsigned long long)all[i].hash, (unsigned long long)all[i].na,
                       (unsigned long long)all[i].nb, all[i].owner, all[i].t_detect,
                       ok ? "VERIFIED" : "FAILED");
        }

        /* Earliest rank-local trial index at which the K-th found collision existed. */
        double avail = -1;
        if (nall) {
            uint64_t *av = malloc(nall * sizeof *av);
            for (size_t i = 0; i < nall; i++) {
                uint64_t ia = trial_index(all[i].na, 0), ib = trial_index(all[i].nb, 1);
                av[i] = ia > ib ? ia : ib;
            }
            qsort(av, nall, sizeof *av, cmp_u64);
            size_t kk = (size_t)o.K <= nall ? (size_t)o.K - 1 : nall - 1;
            avail = (double)av[kk];
            free(av);
        }

        char pair[256];
        pair_name(o.file_a, pair, sizeof pair);
        const double thr = (double)st[0] / el_max;
        if (!o.quiet) {
            printf("search time %.3f s, %llu trials, throughput %.4g trials/s\n", el_max, st[0], thr);
            printf("messages %llu (%.4g MB), remote records %llu, local records %llu, comm time max %.3f s, drain %.3f s\n",
                   st[1], (double)st[2] / 1e6, st[3], st[4], tc_max, td_max);
            if (!o.bench)
                printf("found %zu distinct collision(s) (requested %ld), %zu verified, stop decided at %.3f s\n",
                       nall, o.K, verified, comm.t_stop);
        }

        /* Solved files: collision 1, re-verified on the exact bytes written. */
        if (!o.bench && want > 0 && verified == want && distinct && o.write) {
            pdf_set_nonce(&fa, all[0].na);
            pdf_set_nonce(&fb, all[0].nb);
            uint64_t ha = pdf_hash(&fa), hb = pdf_hash(&fb);
            if (ha == hb && ha == all[0].hash) {
                pdf_write(&fa, o.outdir);
                pdf_write(&fb, o.outdir);
                char path[1024];
                snprintf(path, sizeof path, "%s/%s_collisions.txt", o.outdir, pair);
                FILE *fp = fopen(path, "w");
                if (fp) {
                    fprintf(fp, "# %s: %zu verified distinct collision(s), student id %s\n", pair, want, o.sid);
                    fprintf(fp, "# i hash nonceA nonceB\n");
                    for (size_t i = 0; i < want; i++)
                        fprintf(fp, "%zu %012llx %016llx %016llx\n", i + 1,
                                (unsigned long long)all[i].hash, (unsigned long long)all[i].na,
                                (unsigned long long)all[i].nb);
                    fclose(fp);
                    printf("wrote %s\n", path);
                }
            } else {
                fprintf(stderr, "final verification of solved files failed\n");
                rc = 1;
            }
        }
        if (!o.bench && (verified < (size_t)o.K || !distinct)) rc = 1;

        char host[256] = "unknown";
        gethostname(host, sizeof host);
        printf("JSON {\"impl\":\"%s\",\"ranks\":%d,\"threads\":%d,\"nodes\":%d,\"mode\":\"%s\","
               "\"K\":%ld,\"batch\":%d,\"flush_ms\":%g,\"kernel\":\"%s\",\"pair\":\"%s\",\"file_bytes\":%zu,"
               "\"trials\":%llu,\"elapsed\":%.6f,\"throughput\":%.6g,\"msgs\":%llu,\"bytes\":%llu,"
               "\"recs_remote\":%llu,\"recs_local\":%llu,\"t_comm_max\":%.6f,\"t_comm_mean\":%.6f,"
               "\"t_insert_max\":%.6f,\"t_drain_max\":%.6f,\"found\":%zu,\"verified\":%zu,\"t_first\":%.6f,\"t_stop\":%.6f,"
               "\"avail_idx\":%.0f,\"trials_per_rank\":%.1f,\"seed\":%u,\"units_iter\":%ld,"
               "\"host\":\"%s\"}\n",
               IMPL_NAME, g_size, threads, nodes, o.bench ? "bench" : "search", o.K, o.batch, o.flush_ms,
               o.fast ? "fast" : "full", pair, fa.len, st[0], el_max, thr, st[1], st[2], st[3], st[4],
               tc_max, tc_sum / g_size, ti_max, td_max, nall, verified, nall ? all[0].t_detect : -1.0,
               comm.t_stop, avail, (double)st[0] / g_size, o.seed, U, host);
        fflush(stdout);
    }

    free(all);
    free(rec[0]);
    free(rec[1]);
    comm_free(&comm);
    pdf_free(&fa);
    pdf_free(&fb);
#ifdef USE_MPI
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
#endif
    return rc;
}
