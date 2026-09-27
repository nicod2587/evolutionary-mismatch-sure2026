#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --job-name=landscape
#SBATCH --output=landscape_%j.log
#SBATCH --export=ALL

# Computes the R_bar x kappa fitness landscape at 11 mismatch levels (0 = ancestral,
# 1 = modern) and writes results/ladder/ladder_stack.npz. Takes about 35 hours on
# 32 cores. If the job is killed, submit it again and it carries on from the last
# finished mismatch level.
#
# This is the exact run behind the poster (Stanage HPC, University of Sheffield).

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a

cd ~/SURE_2026
source hpc_env/bin/activate
mkdir -p results/ladder

python -u -m scripts.compute_ladder \
       --res 60 --n_worlds 8 --t_eval 25000 --max_spikes 20000 --n_jobs 32 \
       --r_min 0 --r_max 200 --k_min 0.1 --k_max 50 \
       --obs_noise 0.1 --obs_noise_tau 50 \
       --rungs 0.0 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.0 \
       --out_dir results/ladder

echo "=== best fitness at each mismatch level ==="
python -c "import numpy as np; z=np.load('results/ladder/ladder_stack.npz'); [print(f'  mismatch={m:.2f}  best_fitness={g:+.3f}') for m,g in zip(z['rungs'], z['global_max_per_rung'])]"
