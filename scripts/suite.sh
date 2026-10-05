#!/bin/bash
# suite.sh - run the experiment matrix and append one JSON line per run.
#
# CITS3402/CITS5507 Assignment 2 (2026), Shaoming Wu (24914408).  Used both locally (mpiexec) and
# inside the Slurm jobs in slurm/ (srun).  Configure through the environment:
#
#   PLATFORM  label stored with every result         (default: local)
#   NODES     nodes to use                           (default: 1)
#   CPN       cores per node to use                  (default: nproc)
#   LAUNCHER  srun | ompi | mpiexec                  (default: mpiexec)
#             srun:    Slurm launch (Setonix; Kaya if its MPI supports srun)
#             ompi:    OpenMPI mpirun with core binding (Kaya fallback)
#             mpiexec: plain mpiexec (laptop)
#   SRUN_EXTRA extra srun flags, e.g. "-m block:block:block" on Setonix
#   EXPS      experiments to run, any of:
#             smoke kernel scale batch search kfast kfull solve
#   OUT       results file                           (default: results/$PLATFORM.jsonl)
#   SID       student id inserted in the files       (default: compiled-in)
#   SOLVE_OUT output directory of the solve experiment (default: solved)
#   REPEAT    repetitions of each scale/batch configuration (default 1;
#             the analysis keeps the best run, removing interference)
#   SEEDS     seeds for the kfast experiment (default 0..9)
#
# Example (local):  PLATFORM=m4 CPN=8 EXPS="scale batch" scripts/suite.sh
set -u
cd "$(dirname "$0")/.."

PLATFORM=${PLATFORM:-local}
NODES=${NODES:-1}
CPN=${CPN:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
LAUNCHER=${LAUNCHER:-mpiexec}
EXPS=${EXPS:-"kernel scale batch search kfast kfull solve"}
OUT=${OUT:-results/$PLATFORM.jsonl}
LOG=${OUT%.jsonl}.log                 # full output of every run (errors included)
SRUN_EXTRA=${SRUN_EXTRA:-}
SID=${SID:-}
REPEAT=${REPEAT:-1}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
TOTAL=$((NODES * CPN))
mkdir -p "$(dirname "$OUT")"

SIDARG=""
[ -n "$SID" ] && SIDARG="-s $SID"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# run EXP RANKS THREADS BINARY ARGS...   (ranks/threads are totals per job)
run() {
    local exp=$1 n=$2 t=$3 bin=$4
    shift 4
    export OMP_NUM_THREADS=$t
    local cmd
    case $bin in
    birthday_serial|birthday_omp)
        if [ "$LAUNCHER" = srun ]; then
            cmd="srun --nodes=1 --ntasks=1 --cpus-per-task=$t --cpu-bind=cores $SRUN_EXTRA ./$bin"
        else
            cmd="./$bin"
        fi ;;
    *)
        local nn=$NODES
        [ "$n" -lt "$NODES" ] && nn=$n
        local ppn=$(( (n + nn - 1) / nn ))
        case $LAUNCHER in
        srun) cmd="srun --nodes=$nn --ntasks=$n --ntasks-per-node=$ppn --cpus-per-task=$t --cpu-bind=cores $SRUN_EXTRA ./$bin" ;;
        ompi) cmd="mpirun -np $n --map-by ppr:$ppn:node:PE=$t --bind-to core -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND ./$bin" ;;
        *)    cmd="mpiexec -n $n ./$bin" ;;
        esac ;;
    esac
    echo "[$(date +%H:%M:%S)] $exp: $cmd $* $SIDARG" >&2
    echo "### $exp: $cmd $* $SIDARG" >> "$LOG"
    $cmd "$@" $SIDARG -q 2>&1 | tee -a "$LOG" | grep '^JSON ' |
        sed "s/^JSON {/{\"platform\":\"$PLATFORM\",\"exp\":\"$exp\",/" | tee -a "$OUT"
}

# runr: run REPEAT times
runr() {
    local i=0
    while [ $i -lt "$REPEAT" ]; do run "$@"; i=$((i + 1)); done
}

A1="-a 1_alpha_a.pdf -b 1_alpha_b.pdf"
A3="-a 3_gamma_a.pdf -b 3_gamma_b.pdf"

pow2_upto() {  # 1 2 4 ... <= $1, plus $1 itself
    local v=1 out=""
    while [ $v -lt "$1" ]; do out="$out $v"; v=$((v * 2)); done
    echo "$out $1"
}

for exp in $EXPS; do
case $exp in

smoke)  # quick end-to-end check of every build (a few seconds; use it first)
    t2=2; [ "$CPN" -lt 2 ] && t2=1
    run smoke 1 1 birthday_serial $A1 --benchmark 4096
    run smoke 1 "$t2" birthday_omp $A1 --benchmark 4096
    run smoke 2 1 birthday_mpi $A1 -K 2 --fast --no-write
    run smoke 2 "$t2" birthday_hybrid $A1 -K 2 --fast --no-write
    run smoke "$TOTAL" 1 birthday_mpi $A1 -K 3 --fast --no-write
    ;;

kernel)  # single-core kernel throughput for every pair, both kernels
    for f in 1_alpha 2_beta 3_gamma 4_delta 5_epsilon 6_zeta; do
        run kernel 1 1 birthday_serial -a ${f}_a.pdf -b ${f}_b.pdf --benchmark 65536
        run kernel 1 1 birthday_serial -a ${f}_a.pdf -b ${f}_b.pdf --benchmark 8388608 --fast
    done ;;

scale)  # fixed-work throughput: serial, OpenMP, MPI and hybrid (gamma pair, full kernel)
    nbench() { local c=$1 n=$(($1 * 131072)); [ $n -lt 262144 ] && n=262144; echo $n; }
    if [ "$NODES" -eq 1 ]; then
        runr scale 1 1 birthday_serial $A3 --benchmark $(nbench 1)
        for t in $(pow2_upto "$CPN"); do
            runr scale 1 "$t" birthday_omp $A3 --benchmark $(nbench "$t")
            runr scale "$t" 1 birthday_mpi $A3 --benchmark $(nbench "$t")
        done
    else
        runr scale "$TOTAL" 1 birthday_mpi $A3 --benchmark $(nbench "$TOTAL")
    fi
    # hybrid: same total core count, different ranks x threads splits
    # (powers of two plus CPN/2 and CPN, e.g. 2 4 8 16 32 48 96 on 96-core nodes)
    for t in $(printf '%s\n' 2 4 8 16 32 64 128 $((CPN / 2)) "$CPN" | sort -n | uniq); do
        [ "$t" -lt 2 ] && continue
        [ "$t" -gt "$CPN" ] && break
        [ $((CPN % t)) -ne 0 ] && continue
        runr scale $((TOTAL / t)) "$t" birthday_hybrid $A3 --benchmark $(nbench "$TOTAL")
    done ;;

batch)  # batch size: messages, communication time, throughput (no timed flush)
    for b in 1 4 16 64 256 1024 4096 16384; do
        runr batch "$TOTAL" 1 birthday_mpi $A1 --benchmark $((TOTAL * 262144)) -B $b --flush-ms 0
        runr batch "$TOTAL" 1 birthday_mpi $A1 --benchmark $((TOTAL * 4194304)) -B $b --flush-ms 0 --fast
    done
    ht=4; [ "$CPN" -lt 4 ] && ht=1
    for b in 1 16 256 4096; do
        runr batch $((TOTAL / ht)) $ht birthday_hybrid $A1 --benchmark $((TOTAL * 262144)) -B $b --flush-ms 0
    done ;;

search)  # time to detect the first collision vs batch size (fixed seed)
    # per-destination buffers fill at (rate / p), so keep the largest batch
    # affordable at large process counts (detection delay ~ B * p / rate)
    blist="16 256 4096 65536"; [ "$TOTAL" -gt 64 ] && blist="16 256 1024 4096"
    bmax=${blist##* }
    for b in $blist; do
        run search "$TOTAL" 1 birthday_mpi $A1 -B $b --flush-ms 0 --no-write
    done
    run search "$TOTAL" 1 birthday_mpi $A1 -B $bmax --flush-ms 100 --no-write
    ;;

kfast)  # effect of K on trials/time: fast kernel, several seeds
    for K in 1 2 4 8 16; do
        for s in $SEEDS; do
            run kfast "$TOTAL" 1 birthday_mpi $A1 -K $K --seed $s --fast --no-write
        done
    done ;;

kfull)  # effect of K with the full kernel (one seed)
    ht=4; [ "$CPN" -lt 4 ] && ht=1
    for K in 1 4 16; do
        run kfull $((TOTAL / ht)) $ht birthday_hybrid $A1 -K $K --no-write
    done ;;

solve)  # actual time to a collision for every pair (full kernel), writes solved/
    ht=4; [ "$CPN" -lt 4 ] && ht=1
    for f in 1_alpha 2_beta 3_gamma 4_delta 5_epsilon 6_zeta; do
        run solve $((TOTAL / ht)) $ht birthday_hybrid -a ${f}_a.pdf -b ${f}_b.pdf -o "${SOLVE_OUT:-solved}"
    done ;;

*) echo "unknown experiment $exp" >&2 ;;
esac
done
