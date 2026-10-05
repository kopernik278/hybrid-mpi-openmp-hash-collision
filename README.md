# CITS3402/CITS5507 Assignment 2 (2026): distributed birthday attack

**Author:** Shaoming Wu (24914408), individual submission

This program finds nonce pairs `(nA, nB)` such that `toy_hash(A, nA) == toy_hash(B, nB)` for the supplied
PDF pairs. It can find **K distinct collisions** and runs serially, with OpenMP, with pure MPI, or
with hybrid MPI + OpenMP. The report is in `report/report.pdf`, and solved files are in `solved/`.

## Build

```bash
make                         # builds all four executables
make STUDENT_ID=24914408     # default student number compiled in (already the default; -s overrides it)
make test                    # quick end-to-end check (needs mpiexec and python3)
```

The Makefile detects the platform:

| Platform | Compiler | OpenMP flag |
|---|---|---|
| Setonix (Cray PE, `CRAYPE_VERSION` set) | `cc` wrapper with Cray MPICH | `-fopenmp` |
| Kaya and generic Linux | `gcc` for the serial/OpenMP builds, `mpicc` for the MPI builds | `-fopenmp` |
| macOS | `clang` with Homebrew `libomp` | Homebrew `libomp` flags |

Optional settings:
- `ARCH=-march=native` turns on CPU-specific tuning; build on a compute node if you use it.
- `LANES=8` sets how many nonces the kernel hashes together. Keep the default of 8 on x86-64.

| Executable | Parallelism | MPI initialisation |
|---|---|---|
| `birthday_serial` | none | – |
| `birthday_omp` | OpenMP threads, 1 process | – |
| `birthday_mpi` | pure MPI | `MPI_Init` |
| `birthday_hybrid` | MPI + OpenMP | `MPI_Init_thread(MPI_THREAD_FUNNELED)`, aborts if not provided |

## Running

```
<exe> -a FILE_A -b FILE_B [options]
  -s, --student ID     8-digit student number written into both files
  -K, --collisions K   number of distinct collision hashes to find (default 1)
  -B, --batch N        candidate records per MPI message (default 1024)
      --flush-ms T     also send partial batches every T ms (default 100, 0 = size-only batching)
      --benchmark N    fixed-work mode: exactly N trials in total (rounded up to 16 per process), no early stop
      --fast           exact affine suffix kernel (see below); default is the byte-by-byte kernel
  -o, --outdir DIR     where solved files go (default solved/)
      --no-write       verify but do not write files
      --seed S         nonce-space seed 0..255 (different runs → different collisions)
      --chunk U        hashing units per thread per iteration (default: automatic)
  -q, --quiet          print only the JSON summary line
```
OpenMP threads come from `OMP_NUM_THREADS`.

### Examples

```bash
# serial and OpenMP (shared memory)
./birthday_serial -a 1_alpha_a.pdf -b 1_alpha_b.pdf -s 24914408
OMP_NUM_THREADS=8 ./birthday_omp -a 1_alpha_a.pdf -b 1_alpha_b.pdf -s 24914408

# pure MPI, 8 processes, batches of 1024 records
mpiexec -n 8 ./birthday_mpi -a 1_alpha_a.pdf -b 1_alpha_b.pdf -B 1024 -s 24914408

# hybrid MPI + OpenMP: 4 processes x 8 threads
OMP_NUM_THREADS=8 mpiexec -n 4 ./birthday_hybrid -a 1_alpha_a.pdf -b 1_alpha_b.pdf -s 24914408
#   with Slurm:  srun -n 4 -c 8 --cpu-bind=cores ./birthday_hybrid ...

# different communication batch sizes
mpiexec -n 8 ./birthday_mpi -a 1_alpha_a.pdf -b 1_alpha_b.pdf -B 1     # one record per message
mpiexec -n 8 ./birthday_mpi -a 1_alpha_a.pdf -b 1_alpha_b.pdf -B 16384 --flush-ms 0

# K = 5 distinct collisions
mpiexec -n 8 ./birthday_mpi -a 2_beta_a.pdf -b 2_beta_b.pdf -K 5

# benchmark mode: exactly 10,000,000 trials in total, throughput reported
mpiexec -n 8 ./birthday_mpi -a 1_alpha_a.pdf -b 1_alpha_b.pdf --benchmark 10000000
```

### Output

Rank 0 prints:
- every collision with its hash, its nonces and a `VERIFIED` or `FAILED` flag;
- the search time, the number of trials and the throughput;
- message and byte counts;
- a final `JSON {...}` line that the analysis scripts parse.

In search mode, the program verifies each of the first K collisions on its own: it writes the nonce into a
copy of each original file and recomputes `toy_hash` over the whole file with the reference function. It
then writes the collision-1 pair to `solved/<name>.pdf`, checking the hashes once more on the exact bytes
it writes. All K collisions go to `solved/<pair>_collisions.txt`. Exit status 0 means K distinct
collisions were found and verified.

The timed region starts after file loading, broadcast, kernel set-up and table allocation, at a barrier. It
ends after termination, the message drain and the gathering of collisions to rank 0. Verification and file
writing are not timed.

You can check a solved file independently: `python3 check_toy_hash.py solved/1_alpha_a.pdf`.

## Design (summary; details in the report)

**Trials.** One trial is (file, nonce). The nonce is written as 16 lower-case hex digits at byte 16.
Process `r` uses nonces `seed<<56 | r<<40 | counter`, so processes never repeat a trial. Inside a process,
OpenMP threads take disjoint 8-nonce units (`schedule(dynamic)`), and units alternate between A and B.

**Kernel.**
- The FNV state of the 16-byte prefix is computed once.
- The default kernel hashes the rest of the file for 8 nonces at a time, so their multiplies overlap
  (instruction-level parallelism).
- `--fast` uses an exact identity. The suffix map is `S(h) = P^n·h + C[h mod 256] (mod 2^64)`, because
  `h ^ c = h + c − 2(h & c)` and the low byte of an FNV state depends only on the previous low byte. Each
  trial then costs O(16) steps.
- Both kernels are checked against the reference `toy_hash` at start-up.

**Distributed table.**
- `owner(h) = h mod p`. Each process keeps an open-addressing table for the hashes it owns.
- A collision is an A/B match on the same hash. It is marked so that it is counted once, which makes the K
  collisions distinct globally.

**Communication.**
- Each process has one buffer of `B` records per destination. A full buffer goes out with `MPI_Issend`.
- At most 2 batches per destination can be in flight. This credit limit is the flow control.
- `MPI_Improbe`/`MPI_Mrecv` receive batches, at most p per poll.
- `--flush-ms` limits how long a partial batch can wait before it is sent.
- The master also polls MPI about 16 times per iteration, between its own work units. Many MPI libraries
  only advance a large (rendezvous) transfer when the process calls into MPI.

**Termination.**
- One `MPI_Iallreduce` of the local collision count is always in flight. All processes see the same
  results, so they stop at the same reduction once the total reaches K.
- After that, a non-blocking `MPI_Ialltoall` of per-peer message counts lets every process drain all
  in-flight batches before `MPI_Finalize`.

**Hybrid.**
- In each iteration, the OpenMP team hashes chunk *i*. At the same time the master thread routes chunk
  *i−1*, receives batches, inserts records into the table and tests for termination.
- Only the master thread calls MPI, so `MPI_THREAD_FUNNELED` is enough.

## Running on Setonix and Kaya

Both clusters use the same driver (`scripts/suite.sh`). Each job appends JSON lines to
`results/<platform>.jsonl` and the full program output to `results/<platform>.log`.

**Setonix** (Pawsey): the scripts follow Pawsey's CPU job examples (`#!/bin/bash --login`,
`--exclusive`, `srun -c ... -m block:block:block`, `MPICH_OFI_STARTUP_CONNECT=1`). Work from
`$MYSCRATCH`: `/home` is limited to 1 GB, and `/scratch` is purged after 21 days and not backed up.
```bash
ssh <username>@setonix.pawsey.org.au
cd $MYSCRATCH && git clone https://github.com/kopernik278/hybrid-mpi-openmp-hash-collision.git
cd hybrid-mpi-openmp-hash-collision && make
# set --account=<project> in slurm/setonix.slurm (and slurm/solve_one.slurm); `groups` lists your projects
sbatch --nodes=1 --time=00:05:00 --export=ALL,EXPS=smoke slurm/setonix.slurm   # quick check first
sbatch --nodes=1 slurm/setonix.slurm
sbatch --nodes=2 slurm/setonix.slurm
sbatch --nodes=4 slurm/setonix.slurm
```

**Kaya** (UWA, on campus or VPN): set the partition, the physical cores per node and the module names
(lines marked `<-- CHANGE` in `slurm/kaya.slurm`). The job picks `srun` or OpenMPI's `mpirun`
automatically, whichever starts a 2-process MPI job.
```bash
ssh <username>@kaya.hpc.uwa.edu.au
git clone https://github.com/kopernik278/hybrid-mpi-openmp-hash-collision.git
cd hybrid-mpi-openmp-hash-collision
sinfo -s; sinfo -N -o "%N %P %c %m" | head; module avail openmpi   # values for the CHANGE lines
sbatch --nodes=1 --time=00:05:00 --export=ALL,EXPS=smoke slurm/kaya.slurm
sbatch --nodes=1 slurm/kaya.slurm    # likewise for --nodes=2 and --nodes=4
```

Copy `results/*.jsonl` back and run `python3 scripts/analyse.py`. The figures and tables in
`report/` gain the Kaya and Setonix panels automatically.

## Experiments, Slurm scripts and report

```bash
# locally (results/m4.jsonl)
PLATFORM=m4 CPN=8 EXPS="kernel scale batch search kfast kfull solve" scripts/suite.sh

# Setonix / Kaya: one job per node count (edit the account/partition/modules first)
sbatch --nodes=1 slurm/setonix.slurm
sbatch --nodes=2 slurm/setonix.slurm
sbatch --nodes=4 slurm/setonix.slurm
sbatch --nodes=1 slurm/kaya.slurm   # likewise for 2 and 4 nodes
sbatch --nodes=1 slurm/solve_one.slurm   # solve all pairs with the hybrid build

# figures + tables from every results/*.jsonl, then the PDF
python3 scripts/analyse.py
cd report && pdflatex report.tex && pdflatex report.tex
```

## Files

| Path | Contents |
|---|---|
| `src/birthday.c` | driver, search loop, OpenMP pipeline, verification, output |
| `src/comm.[ch]` | routing, batching, flow control, termination, gather |
| `src/table.[ch]` | the collision table owned by each process |
| `src/kernel.[ch]` | full and fast hashing kernels, self-test |
| `src/pdf_io.[ch]` | load, patch, verify and write PDFs |
| `src/toy_hash.[ch]` | the supplied reference hash (unchanged) |
| `scripts/suite.sh` | experiment driver (local or `srun`) |
| `scripts/analyse.py` | figures and tables for the report |
| `slurm/*.slurm` | Kaya and Setonix job scripts |
| `results/` | raw JSON-lines results and logs |
| `report/` | LaTeX source, figures, `report.pdf` |
| `solved/` | solved PDF pairs (student number 24914408) and `*_collisions.txt` |
| `solved/K5_alpha/` | a K = 5 run: the 5 verified collision hashes and nonce pairs, and the program output |
