#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=landscape
#SBATCH --output=landscape_%j.log
#SBATCH --export=ALL

# Computes the R_bar x kappa fitness landscape at 8 mismatch levels (0 = novelty
# tracks value, 1 = novelty is free; nothing else about the world changes) and
# writes results/ladder_res<RES>/ladder_stack.npz. Stanage HPC, University of Sheffield.
#
#   sbatch run_landscape.sh                                        # 10 x 10 first look, ~1 h
#   sbatch --time=30:00:00 --export=ALL,RES=40 run_landscape.sh    # 40 x 40, ~16x the cells
#
# If the job is killed, submit it again with the same RES: it carries on from the
# last finished mismatch level. The two ends and mismatch 0.10 are done first.
#
# The printed poster used an older model (results/poster_v1).

RES=${RES:-10}

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a

cd "$SLURM_SUBMIT_DIR"          # submit from the repo root
source hpc_env/bin/activate     # a venv with requirements.txt installed
python tests/test_pure_mismatch.py || exit 1

OUT=results/ladder_res${RES}
mkdir -p $OUT
python -u -m scripts.compute_ladder \
       --res $RES --spacing log --n_worlds 8 --t_eval 25000 --max_spikes 20000 --n_jobs 32 \
       --r_min 1 --r_max 200 --k_min 0.1 --k_max 50 \
       --obs_noise 0.1 --obs_noise_tau 50 \
       --rungs 0.0 0.05 0.10 0.20 0.30 0.50 0.75 1.0 \
       --out_dir $OUT
