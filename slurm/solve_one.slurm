#!/bin/bash --login
# Solve pairs with the hybrid build on one Setonix node (16 ranks x 8 threads).
# Author: Shaoming Wu (24914408)
#   sbatch slurm/solve_one.slurm                                   # all six pairs, K=1
#   sbatch --export=ALL,PAIRS="6_zeta",K=4 slurm/solve_one.slurm   # one pair, 4 collisions
# On Kaya: remove the --account line, set --partition, and use 'module load gcc openmpi'.
#
#SBATCH --job-name=birthday-solve
#SBATCH --account=courses0100        # <-- CHANGE (Setonix project code; delete on Kaya)
#SBATCH --partition=work             # <-- CHANGE on Kaya
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=results/solve-%j.out

set -u
cd "${SLURM_SUBMIT_DIR}"
make

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
PAIRS=${PAIRS:-"1_alpha 2_beta 3_gamma 4_delta 5_epsilon 6_zeta"}
K=${K:-1}
for f in $PAIRS; do
    srun -N "${SLURM_JOB_NUM_NODES}" -n "${SLURM_NTASKS}" -c "${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
        ./birthday_hybrid -a "${f}_a.pdf" -b "${f}_b.pdf" -K "$K" -B 1024 -o solved
done
