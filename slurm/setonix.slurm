#!/bin/bash --login
# Setonix (Pawsey) - CPU partition "work": 128 cores (2 x 64-core AMD EPYC) and
# 230 GiB per node.  Author: Shaoming Wu (24914408)
#
# Run from a copy of the project on /scratch (not /home, which is 1 GB and slow):
#   cd $MYSCRATCH/hybrid-mpi-openmp-hash-collision
#   sbatch --nodes=1 slurm/setonix.slurm      # kernel, scaling, K (single node)
#   sbatch --nodes=2 slurm/setonix.slurm      # multi-node scaling
#   sbatch --nodes=4 slurm/setonix.slurm      # scaling, batch, search, K, solve
# Results are appended to results/setonix.jsonl (full output in results/setonix.log).
#
# Estimated cost (exclusive nodes are charged 128 cores each):
#   ~15 min x 1 node + ~5 min x 2 nodes + ~20 min x 4 nodes ~= 230 core-hours.
#   Run fewer experiments with e.g.  sbatch --nodes=4 --export=ALL,EXPS="scale batch" ...
#
#SBATCH --job-name=birthday
#SBATCH --account=courses0100        # <-- CHANGE to your Pawsey project code (run `groups` on Setonix)
#SBATCH --partition=work
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=results/setonix-%j.out

set -u
cd "${SLURM_SUBMIT_DIR}"

module load PrgEnv-gnu 2>/dev/null || true   # default on Setonix: cc wrapper + Cray MPICH
make

export MPICH_OFI_STARTUP_CONNECT=1           # Pawsey recommendation for multi-node MPI
export PLATFORM=setonix
export NODES=${SLURM_JOB_NUM_NODES}
export CPN=128
export LAUNCHER=srun
export SRUN_EXTRA="-m block:block:block"     # pack ranks/threads on contiguous cores
if [ -z "${EXPS:-}" ]; then
    case $NODES in
    1) EXPS="kernel scale kfast" ;;
    2) EXPS="scale" ;;
    *) EXPS="scale batch search kfast kfull solve" ;;
    esac
fi
export EXPS
lscpu > "results/setonix-${SLURM_JOB_ID}.lscpu" 2>/dev/null
scripts/suite.sh
