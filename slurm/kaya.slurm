#!/bin/bash --login
# Kaya (UWA) - same experiment matrix as Setonix.  Author: Shaoming Wu (24914408)
#
# Submit from the project root, once per node count:
#   sbatch --nodes=1 slurm/kaya.slurm
#   sbatch --nodes=2 slurm/kaya.slurm
#   sbatch --nodes=4 slurm/kaya.slurm
# Results are appended to results/kaya.jsonl (full output in results/kaya.log).
#
# Kaya course partition: cits3402 = 4 nodes (k007-k010), 96 CPUs and ~1.5 TB each,
# 2 h limit.  A 4-node --exclusive job needs the whole partition, so it may queue.
# If 96 "CPUs" turn out to be 48 cores x 2 hyper-threads, the job detects it,
# uses the physical cores and adds --hint=nomultithread.  Only the module line
# may need changing:  module avail 2>&1 | grep -iE "gcc|openmpi|mpich"
#
#SBATCH --job-name=birthday
#SBATCH --partition=cits3402         # Kaya course CPU partition (sinfo -s)
#SBATCH --ntasks-per-node=96         # CPUs per cits3402 node (sinfo -N -o "%N %P %c %m")
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=results/kaya-%j.out

set -u
cd "${SLURM_SUBMIT_DIR}"

module load gcc openmpi 2>/dev/null || true  # <-- CHANGE if Kaya's module names differ (module avail)
command -v mpicc >/dev/null || { echo "mpicc not found: load Kaya's compiler + MPI modules (module avail)" >&2; exit 1; }
make

export PLATFORM=kaya
export NODES=${SLURM_JOB_NUM_NODES}
tpn=${SLURM_NTASKS_PER_NODE:-96}
CPN=${CPN:-${tpn%%(*}}
# Use physical cores only: with hyper-threading, run one rank/thread per core.
phys=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
tpc=$(lscpu 2>/dev/null | awk -F: '/Thread\(s\) per core/ {gsub(/ /, "", $2); print $2}')
export SRUN_EXTRA=""
if [ "${tpc:-1}" -gt 1 ]; then
    export SRUN_EXTRA="--hint=nomultithread"
fi
if [ "$phys" -gt 0 ] && [ "$phys" -lt "$CPN" ]; then
    echo "note: $CPN CPUs but $phys physical cores per node -> using CPN=$phys" >&2
    CPN=$phys
fi
export CPN

# Use whichever launcher really starts a 2-process MPI job (srun needs an MPI
# built with Slurm/PMI support; otherwise fall back to OpenMPI's mpirun).
probe() { $1 ./birthday_mpi -a 1_alpha_a.pdf -b 1_alpha_b.pdf --benchmark 64 -q 2>/dev/null | grep -q '"ranks":2,'; }
if probe "srun --nodes=1 --ntasks=2 --cpus-per-task=1 $SRUN_EXTRA"; then
    export LAUNCHER=srun
elif probe "mpirun -np 2"; then
    export LAUNCHER=ompi
else
    echo "cannot start a 2-process MPI job with srun or mpirun - check the MPI module" >&2
    exit 1
fi
echo "launcher: $LAUNCHER, nodes: $NODES, cores per node: $CPN, threads/core: ${tpc:-?} $SRUN_EXTRA" >&2

if [ -z "${EXPS:-}" ]; then
    case $NODES in
    1) EXPS="kernel scale kfast" ;;
    2) EXPS="scale" ;;
    *) EXPS="scale batch search kfast kfull solve" ;;
    esac
fi
export EXPS
lscpu > "results/kaya-${SLURM_JOB_ID}.lscpu" 2>/dev/null
scripts/suite.sh
