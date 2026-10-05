#!/bin/bash --login
# Kaya (UWA) - same experiment matrix as Setonix.  Author: Shaoming Wu (24914408)
#
# Submit from the project root, once per node count:
#   sbatch --nodes=1 slurm/kaya.slurm
#   sbatch --nodes=2 slurm/kaya.slurm
#   sbatch --nodes=4 slurm/kaya.slurm
# Results are appended to results/kaya.jsonl (full output in results/kaya.log).
#
# Before the first run, check the three lines marked "<-- CHANGE":
#   partition:        sinfo -s
#   cores per node:   sinfo -N -o "%N %P %c %m" | head      (use PHYSICAL cores)
#   modules:          module avail gcc ; module avail openmpi
#
#SBATCH --job-name=birthday
#SBATCH --partition=work             # <-- CHANGE to a Kaya CPU partition (see `sinfo -s`)
#SBATCH --ntasks-per-node=32         # <-- CHANGE to the physical cores per Kaya node
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=results/kaya-%j.out

set -u
cd "${SLURM_SUBMIT_DIR}"

module purge 2>/dev/null
module load gcc openmpi 2>/dev/null || true  # <-- CHANGE to Kaya's module names if different
command -v mpicc >/dev/null || { echo "mpicc not found: load Kaya's compiler + MPI modules (module avail)" >&2; exit 1; }
make

export PLATFORM=kaya
export NODES=${SLURM_JOB_NUM_NODES}
tpn=${SLURM_NTASKS_PER_NODE:-32}
export CPN=${CPN:-${tpn%%(*}}
phys=$(lscpu -p=core,socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
if [ "$phys" -gt 0 ] && [ "$phys" != "$CPN" ]; then
    echo "warning: node has $phys physical cores but CPN=$CPN (fix --ntasks-per-node)" >&2
fi

# Use whichever launcher really starts a 2-process MPI job (srun needs an MPI
# built with Slurm/PMI support; otherwise fall back to OpenMPI's mpirun).
probe() { $1 ./birthday_mpi -a 1_alpha_a.pdf -b 1_alpha_b.pdf --benchmark 64 -q 2>/dev/null | grep -q '"ranks":2,'; }
if probe "srun --nodes=1 --ntasks=2 --cpus-per-task=1"; then
    export LAUNCHER=srun
elif probe "mpirun -np 2"; then
    export LAUNCHER=ompi
else
    echo "cannot start a 2-process MPI job with srun or mpirun - check the MPI module" >&2
    exit 1
fi
echo "launcher: $LAUNCHER, nodes: $NODES, cores per node: $CPN" >&2

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
