#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32       # 32 cores for the parallel res^3 genome grid
#SBATCH --mem=16G                # Request 16 Gigabytes of memory
#SBATCH --time=08:00:00          # res=12 -> 1728 genomes x 6 worlds. The regrounded model is
                                 # ~4x slower per eval (~3-4h expected), so 8h is a conservative
                                 # ceiling. genome_landscape checkpoints to fitness_cube_partial.npz,
                                 # so a kill is resumable: just resubmit and it continues.
#SBATCH --job-name=ev_cube       # Name of your job
#SBATCH --output=cube_run_%j.log # Where the live per-genome progress + figures log (%j = job id,
                                 # unique per submission so logs never overwrite/race)
#SBATCH --export=ALL             # propagate TRAIN_MISMATCH / OBS_NOISE from the submitting shell so
                                 # the cube is scored on the SAME landscape the GA trained on (some
                                 # site sbatch defaults are --export=NONE, which would silently drop
                                 # them and rebuild the ancestral-only cube). Do not remove.

# 1. Clean and load the system core Python modules (same as run_ga.sh / run_sweep.sh)
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a

# 2. Move into project directory and activate your virtual environment
cd ~/SURE_2026
source hpc_env/bin/activate

# 3. Sweep the genome cube and render the 3D fitness landscape + 3D evolution
#    trajectory. Run this AFTER run_ga.sh has produced gen_NNNN.pkl checkpoints
#    (and hpc_sweep.py has written champion.pkl) in ga_results/ — the champion
#    is then overlaid on the cube and the run is drawn as a 3D path.
#    -u = unbuffered so the per-genome progress streams into cube_output.log live.
#    -m scripts.genome_landscape runs it as a package module so `from scripts...`
#    imports resolve from the repo root.
#
#    --res 12   : 12^3 = 1728 grid genomes (raise for finer landscape, costs ~res^3).
#    Scored on the GA's fixed evaluation seeds so the cube IS the surface the GA climbed.
# Eval config MATCHED to the GA (hpc_run.py: T_EVAL_TRAIN=30000, N_EVAL_WORLDS=12,
# MAX_SPIKES=4000) so the cube is the EXACT surface the GA climbed -> the evolved
# champion (red diamond) lands on the grid max (gold diamond). n_worlds must match so
# ga_fixed_seeds() draws the identical world-seed SET the GA averaged over; a different
# t_eval/max_spikes shifts fitness values and knocks the champion off the optimum.
python -u -m scripts.genome_landscape --res 12 --n_jobs 32 --run_dir ga_results \
       --t_eval 30000 --n_worlds 12 --max_spikes 4000

# ── Bounds question (already handled) ────────────────────────────────────────
# GENOME_BOUNDS has been WIDENED (R_bar 0.25-25, kappa 0.1-6, tau_pfc 20-500), and
# this cube sweeps exactly those bounds — so the default cube above already covers
# the full evolvable box. Just read the cube: if the bright region sits INTERIOR
# (doesn't touch a face), the optimum is real; if it still hugs a face, that bound
# needs widening further. No separate --wide pass to remember.
#
# To go even wider than GENOME_BOUNDS (only if the bright region still touches a
# face), the --wide flag sweeps an extra-large box into a separate dir:
# python -u -m scripts.genome_landscape --res 12 --n_jobs 32 --wide \
#        --run_dir ga_results --out_dir ga_results/cube_wider
