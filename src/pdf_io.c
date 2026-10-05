/*
 * pdf_io.c - loading, patching, verifying and writing the nonce PDFs.
 *
 * CITS3402/CITS5507 Assignment 2 (2026) - distributed birthday attack.
 * Author: Shaoming Wu (24914408)
 */
#include "pdf_io.h"
#include "toy_hash.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char HEX[] = "0123456789abcdef";

/* Find the first occurrence of tag in the first `limit` bytes of buf. */
static long find_tag(const unsigned char *buf, size_t len, const char *tag, size_t limit)
{
    size_t tl = strlen(tag);
    if (limit > len) limit = len;
    for (size_t i = 0; i + tl <= limit; i++)
        if (memcmp(buf + i, tag, tl) == 0) return (long)i;
    return -1;
}

int pdf_from_buffer(pdf_file_t *f, const char *path, unsigned char *data, size_t len)
{
    memset(f, 0, sizeof *f);
    f->data = data;
    f->len = len;
    f->path = strdup(path);
    /* The fields live in the header; only search the first 4 KiB. */
    long n = find_tag(f->data, f->len, "%NONCE=", 4096);
    long s = find_tag(f->data, f->len, "%STUDENT_ID=", 4096);
    if (n < 0 || s < 0) {
        fprintf(stderr, "%s: NONCE/STUDENT_ID field not found\n", path);
        return -1;
    }
    f->nonce_off = (size_t)n + 7;
    f->sid_off = (size_t)s + 12;
    if (f->nonce_off + NONCE_HEX_LEN > f->len || f->sid_off + STUDENT_ID_LEN > f->len) {
        fprintf(stderr, "%s: truncated header\n", path);
        return -1;
    }
    return 0;
}

int pdf_load(pdf_file_t *f, const char *path)
{
    memset(f, 0, sizeof *f);
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    unsigned char *data = sz > 0 ? malloc((size_t)sz) : NULL;
    if (!data || fread(data, 1, (size_t)sz, fp) != (size_t)sz) {
        fclose(fp);
        free(data);
        fprintf(stderr, "%s: read failed\n", path);
        return -1;
    }
    fclose(fp);
    return pdf_from_buffer(f, path, data, (size_t)sz);
}

void pdf_free(pdf_file_t *f)
{
    free(f->data);
    free(f->path);
    memset(f, 0, sizeof *f);
}

int pdf_set_student_id(pdf_file_t *f, const char *sid)
{
    if (strlen(sid) != STUDENT_ID_LEN) return -1;
    for (int i = 0; i < STUDENT_ID_LEN; i++)
        if (sid[i] < '0' || sid[i] > '9') return -1;
    memcpy(f->data + f->sid_off, sid, STUDENT_ID_LEN);
    return 0;
}

void pdf_set_nonce(pdf_file_t *f, uint64_t n)
{
    for (int k = 0; k < NONCE_HEX_LEN; k++)
        f->data[f->nonce_off + k] = (unsigned char)HEX[(n >> (60 - 4 * k)) & 0xf];
}

uint64_t pdf_hash(const pdf_file_t *f)
{
    return toy_hash(f->data, f->len);
}

/* Hash a copy of f with nonce n, leaving f untouched. */
static uint64_t hash_with_nonce(const pdf_file_t *f, uint64_t n)
{
    pdf_file_t tmp = *f;
    tmp.data = malloc(f->len);
    memcpy(tmp.data, f->data, f->len);
    pdf_set_nonce(&tmp, n);
    uint64_t h = toy_hash(tmp.data, tmp.len);
    free(tmp.data);
    return h;
}

int pdf_verify_pair(const pdf_file_t *a, const pdf_file_t *b, uint64_t na,
                    uint64_t nb, uint64_t expect, uint64_t *ha, uint64_t *hb)
{
    uint64_t x = hash_with_nonce(a, na);
    uint64_t y = hash_with_nonce(b, nb);
    if (ha) *ha = x;
    if (hb) *hb = y;
    return x == y && x == expect;
}

int pdf_write(const pdf_file_t *f, const char *dir)
{
    mkdir(dir, 0755); /* ignore EEXIST */
    const char *base = strrchr(f->path, '/');
    base = base ? base + 1 : f->path;
    size_t need = strlen(dir) + strlen(base) + 2;
    char *out = malloc(need);
    snprintf(out, need, "%s/%s", dir, base);
    FILE *fp = fopen(out, "wb");
    if (!fp) {
        fprintf(stderr, "cannot write %s: %s\n", out, strerror(errno));
        free(out);
        return -1;
    }
    size_t w = fwrite(f->data, 1, f->len, fp);
    fclose(fp);
    if (w != f->len) { fprintf(stderr, "short write on %s\n", out); free(out); return -1; }
    printf("wrote %s\n", out);
    free(out);
    return 0;
}
