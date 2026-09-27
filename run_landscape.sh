#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=36:00:00           # 11 rungs, res=60. max_spikes RAISED 4000->20000 (the 4000 cap
                                  # truncated the modern-world horizon and created a spurious high-kappa
                                  # shelf), so high-mismatch rungs now run the FULL horizon and cost more
                                  # (~16.5h at cap=4000 -> ~22-28h expected at 20000). 36h gives headroom.
                                  # This is the DEFINITIVE run. Per-rung checkpoint => a kill is resumable
                                  # (just resubmit); if your queue caps below 36h, submit at the max and
                                  # resubmit once.
#SBATCH --job-name=ladder_gif
#SBATCH --output=ladder_gif_%j.log
#SBATCH --export=ALL

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a

cd ~/SURE_2026
source hpc_env/bin/activate

# ── ONE bundled submission: compute all rungs -> render all frames -> assemble GIF.
#    Fully self-contained; no manual steps between stages. Resumable (per-rung
#    checkpoint in compute_ladder). Produces mismatch_ladder.gif at the end.
#
# Bounds match the Step-2 box (R_bar 0-200, kappa 0-50). NOTE (2026-07-17): the R_bar x
# kappa ridge is a DEGENERATE DIAGONAL that never turns over (fitness ~97% of peak out to
# R_bar=1000) and kappa PLATEAUS past ~35-45 -- the landscape is a broad plateau you climb
# ONTO, not a bounded peak. So this box is the biologically-relevant WINDOW, not a
# "contain the peak" box. The finding is HOW the plateau deforms across mismatch.
R_MAX=200
K_MAX=50
# Ancestral-adapted reference lines (evolved champion R_bar, kappa) -- the "where we sit"
# marker drawn on every frame. From ga_results/champion.pkl.
REF_R=22.6
REF_K=6.4

mkdir -p ga_results/ladder

# 1) compute the per-rung landscape stack (the expensive, resumable part)
#    DEFINITIVE 11-rung sweep. Starts at 0.0 = TRUE ancestral baseline (novelty perfectly
#    tracks value, h=1, undeformed landscape) so frame 0 is the honest "before". FINE 0.05
#    spacing through 0.0-0.30 -- the action zone where the scout found the peak kappa crashing
#    16.7->2.7 -- for a smooth relocation animation; coarsens (0.10 then 0.25) in the settled
#    tail. If any early frame looks identical/dull, drop it at render time (render is cheap,
#    re-runnable off the stack; the compute is the part we don't want to repeat).
python -u -m scripts.compute_ladder \
       --res 60 --n_worlds 8 --t_eval 25000 --max_spikes 20000 --n_jobs 32 \
       --tau 256 --r_min 0 --r_max ${R_MAX} --k_min 0.1 --k_max ${K_MAX} \
       --obs_noise 0.1 --obs_noise_tau 50 \
       --rungs 0.0 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.0 \
       --out_dir ga_results/ladder

# 2) render frames (shared scale, reference lines, fitness readout)
python -m scripts.render_ladder ga_results/ladder/ladder_stack.npz \
       --ref_r ${REF_R} --ref_k ${REF_K}

# 3) assemble the GIF (+ best-effort mp4)
python -m scripts.assemble_ladder ga_results/ladder/frames \
       --out ga_results/ladder/mismatch_ladder.gif --fps 1.5

# 4) the pivotal readout: peak MOVED or landscape DIMMED?
echo "=== global-max fitness per rung (moved vs dimmed) ==="
python -c "import numpy as np; z=np.load('ga_results/ladder/ladder_stack.npz'); [print(f'  mismatch={m:.2f}  best_fitness={g:+.3f}') for m,g in zip(z['rungs'], z['global_max_per_rung'])]"
