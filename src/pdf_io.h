/*
 * pdf_io.h - loading, patching, verifying and writing the nonce PDFs.
 *
 * CITS3402/CITS5507 Assignment 2 (2026) - distributed birthday attack.
 * Author: Shaoming Wu (24914408)
 *
 * Each supplied file starts with
 *     %PDF-1.4\n%NONCE=<16 hex digits>\n%STUDENT_ID=<8 digits>\n...
 * The nonce field holds a 64-bit nonce written as 16 lower-case hex digits.
 */
#ifndef PDF_IO_H
#define PDF_IO_H

#include <stddef.h>
#include <stdint.h>

#define NONCE_HEX_LEN 16
#define STUDENT_ID_LEN 8

typedef struct {
    char *path;            /* file name as given on the command line */
    unsigned char *data;   /* full file contents (owned) */
    size_t len;
    size_t nonce_off;      /* offset of the first nonce hex digit */
    size_t sid_off;        /* offset of the first student-id digit */
} pdf_file_t;

/* Load a file and locate its NONCE / STUDENT_ID fields. Returns 0 on success. */
int pdf_load(pdf_file_t *f, const char *path);
/* Same as pdf_load for contents already in memory (takes ownership of data). */
int pdf_from_buffer(pdf_file_t *f, const char *path, unsigned char *data, size_t len);
void pdf_free(pdf_file_t *f);

/* Overwrite the student-id field (exactly 8 decimal digits). Returns 0 on success. */
int pdf_set_student_id(pdf_file_t *f, const char *sid);

/* Write nonce n into the nonce field as 16 lower-case hex digits. */
void pdf_set_nonce(pdf_file_t *f, uint64_t n);

/* toy_hash of the whole file as it currently is in memory. */
uint64_t pdf_hash(const pdf_file_t *f);

/* Independently verify a collision: patch copies of A and B with the given
 * nonces and recompute toy_hash over the complete files with the reference
 * implementation.  Returns 1 if H(A,na) == H(B,nb) == expect, else 0.
 * The computed hashes are returned through ha/hb (may be NULL). */
int pdf_verify_pair(const pdf_file_t *a, const pdf_file_t *b, uint64_t na,
                    uint64_t nb, uint64_t expect, uint64_t *ha, uint64_t *hb);

/* Write f (with its current nonce / student id) to dir/<basename(path)>. */
int pdf_write(const pdf_file_t *f, const char *dir);

#endif
