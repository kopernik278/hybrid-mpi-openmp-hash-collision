# CITS3402/CITS5507 Assignment 2 (2026) - distributed birthday attack
# Author: Shaoming Wu (24914408)
#
#   make                 build all four executables
#   make STUDENT_ID=12345678   change the default student number compiled in
#   make ARCH=-march=native    optional CPU-specific tuning (build on a compute node)
#   make test            quick correctness test (needs mpiexec)
#   make dist            build the submission zip
#   make clean
#
# Executables
#   birthday_serial   serial            (no MPI, no OpenMP)
#   birthday_omp      OpenMP            (shared memory, one process)
#   birthday_mpi      pure MPI          (MPI_Init)
#   birthday_hybrid   MPI + OpenMP      (MPI_Init_thread, MPI_THREAD_FUNNELED)

STUDENT_ID ?= 24914408
ARCH ?=
LANES ?= 8

SRCS := src/birthday.c src/comm.c src/kernel.c src/pdf_io.c src/table.c src/toy_hash.c
HDRS := $(wildcard src/*.h)
BINS := birthday_serial birthday_omp birthday_mpi birthday_hybrid

UNAME := $(shell uname -s)

ifdef CRAYPE_VERSION
  # Setonix / Cray Programming Environment: the cc wrapper adds Cray MPICH.
  CC      := cc
  MPICC   := cc
  OMPFLAG := -fopenmp
else ifeq ($(UNAME),Darwin)
  # macOS development machine: Apple clang + Homebrew libomp.
  LIBOMP  ?= $(shell brew --prefix libomp 2>/dev/null)
  CC      := clang
  MPICC   ?= mpicc
  OMPFLAG := -Xpreprocessor -fopenmp -I$(LIBOMP)/include -L$(LIBOMP)/lib -lomp
  export MPICH_CC := clang
  export OMPI_CC := clang
else
  # Kaya / generic Linux: GCC + an MPI module (OpenMPI, MPICH, ...).
  CC      := gcc
  MPICC   ?= mpicc
  OMPFLAG := -fopenmp
endif

CFLAGS ?= -O3
CFLAGS += -std=gnu11 -Wall -Wextra -Wno-unknown-pragmas $(ARCH) -DDEFAULT_STUDENT_ID=\"$(STUDENT_ID)\" -DLANES=$(LANES)
LDLIBS := -lm

.PHONY: all clean test dist

all: $(BINS)

birthday_serial: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDLIBS)

birthday_omp: $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(OMPFLAG) -o $@ $(SRCS) $(LDLIBS)

birthday_mpi: $(SRCS) $(HDRS)
	$(MPICC) $(CFLAGS) -DUSE_MPI -o $@ $(SRCS) $(LDLIBS)

birthday_hybrid: $(SRCS) $(HDRS)
	$(MPICC) $(CFLAGS) -DUSE_MPI $(OMPFLAG) -o $@ $(SRCS) $(LDLIBS)

# Quick end-to-end check: 3 distinct collisions on the smallest pair using the
# fast kernel, then the reference Python checker on the written files.
test: all
	./birthday_serial -a 1_alpha_a.pdf -b 1_alpha_b.pdf -K 3 --fast -o test_out
	mpiexec -n 4 ./birthday_mpi -a 1_alpha_a.pdf -b 1_alpha_b.pdf -K 3 -B 256 --fast -o test_out
	OMP_NUM_THREADS=2 mpiexec -n 2 ./birthday_hybrid -a 1_alpha_a.pdf -b 1_alpha_b.pdf -K 3 --fast -o test_out
	python3 check_toy_hash.py test_out/1_alpha_a.pdf
	python3 check_toy_hash.py test_out/1_alpha_b.pdf

# Submission archive: sources, Makefile, Slurm scripts, README, report, solved pairs, results.
DIST := cits3402_a2_submission
dist:
	rm -rf $(DIST) $(DIST).zip
	mkdir -p $(DIST)/report
	cp -R src scripts slurm results solved Makefile README.md check_toy_hash.py toy_hash.c toy_hash.h $(DIST)/
	cp [1-6]_*_[ab].pdf $(DIST)/
	cp -R report/report.tex report/report.pdf report/figures report/tables $(DIST)/report/
	zip -qr $(DIST).zip $(DIST) -x '*.DS_Store'
	rm -rf $(DIST)
	@echo "created $(DIST).zip"

clean:
	rm -f $(BINS)
	rm -rf test_out *.dSYM
