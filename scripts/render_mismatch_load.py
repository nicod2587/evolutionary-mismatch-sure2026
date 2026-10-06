"""
Figure: the mismatch load.

  red    one fixed genome tuned to the ancestral world, dropped into each world in
         turn. The genome is derived from the ladder itself (compute_ladder.
         ancestral_genome at params.ANCESTRAL_MISMATCH).
  green  the best genome in the biological box (R_bar <= 25, kappa <= 15) for that
         world, picked from the ladder.

  grey   a forager with kappa = 0, which ignores novelty. It cannot be affected by
         the mismatch, so it is the floor that the best genome sinks towards as
         novelty stops carrying information.

The gap between red and green is the mismatch load; green's fall towards grey is
the opportunity lost when novelty stops being informative.

All three lines are re-scored on N_HELDOUT held-out worlds (seeds the ladder never
saw), the same worlds for all three. Picking the best of many noisy cells and then reporting that
cell's own score would flatter green (the winner's curse); scoring it again on new
worlds does not. The levels are the ladder's rungs from ANCESTRAL_MISMATCH up.

The re-scoring is cached in results/mismatch_load_data.npz and redone automatically when
the ladder changes.

  python -m scripts.render_mismatch_load                 # plot (re-scores if needed)
  python -m scripts.render_mismatch_load --recompute     # force the re-scoring
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.fitness import BOX_R, BOX_K, evaluate_genome, eval_seeds
from scripts.params import ANCESTRAL_GENOME, ANCESTRAL_MISMATCH, BASE_PARAMS
from scripts.compute_ladder import ancestral_genome

DATA = 'results/mismatch_load_data.npz'
STACK = 'results/ladder_res40/ladder_stack.npz'
HELDOUT_SEED_BASE = 0     # the ladder uses 20260709, so these are fresh worlds
N_HELDOUT = 32            # more worlds than a ladder cell: only 3 genomes per level


def _box_best(F_rung, R, K):
    rm = R <= BOX_R; km = K <= BOX_K
    sub = F_rung[np.ix_(rm, km)]
    i, j = np.unravel_index(np.nanargmax(sub), sub.shape)
    return float(R[rm][i]), float(K[km][j])


def _job(genome, mm, z, seeds):
    world_kwargs = {'mismatch_index': float(mm), 'obs_noise': float(z['obs_noise']),
                    'obs_noise_tau': float(z['obs_noise_tau']),
                    'space_initial_patches': False}
    return (np.array(genome), BASE_PARAMS, seeds, float(z['t_eval']),
            int(z['max_spikes']), world_kwargs)


def recompute(out=DATA, stack=STACK, n_jobs=os.cpu_count()):
    z = np.load(stack)
    R, K, F, rungs = z['R_bar'], z['kappa'], z['F'], z['rungs']
    seeds = eval_seeds(N_HELDOUT, HELDOUT_SEED_BASE)
    i_anc = int(np.argmin(np.abs(rungs - ANCESTRAL_MISMATCH)))
    red_genome = ancestral_genome(F[i_anc], R, K)
    if not np.allclose(red_genome, ANCESTRAL_GENOME, rtol=0.02):
        print(f'NOTE: params.ANCESTRAL_GENOME={ANCESTRAL_GENOME} but this ladder gives '
              f'({red_genome[0]:.1f}, {red_genome[1]:.2f}); update params.py')
    levels = rungs[rungs >= rungs[i_anc] - 1e-9]
    # The value-only forager: kappa = 0, with the R_bar that does best on the
    # ladder's lowest-kappa row. With kappa = 0 it never sees novelty, so it
    # behaves the same at every mismatch level.
    floor_genome = (float(R[np.argmax(F[:, :, 0].mean(axis=0))]), 0.0)
    green_genomes = [_box_best(F[int(np.argmin(np.abs(rungs - mm)))], R, K) for mm in levels]
    jobs = []
    for mm, g in zip(levels, green_genomes):
        jobs += [_job(red_genome, mm, z, seeds), _job(g, mm, z, seeds),
                 _job(floor_genome, mm, z, seeds)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        scores = list(pool.map(evaluate_genome, jobs))
    red, green, floor = scores[0::3], scores[1::3], scores[2::3]
    for mm, g, rf, gf, ff in zip(levels, green_genomes, red, green, floor):
        print(f'mm={mm:.2f}  red={rf:+.3f}  green={gf:+.3f} at R={g[0]:.1f} k={g[1]:.2f}'
              f'  value-only={ff:+.3f}', flush=True)
    np.savez(out, mm=levels, red=red, green=green, green_genomes=np.array(green_genomes),
             red_genome=np.array(red_genome), floor=floor,
             floor_genome=np.array(floor_genome), n_heldout=N_HELDOUT,
             model=str(z['model']), stack_F=F)
    print('saved', out)


def _cache_is_current(data=DATA, stack=STACK):
    if not os.path.exists(data):
        return False
    with np.load(data) as d, np.load(stack) as z:
        return ('n_heldout' in d.files and int(d['n_heldout']) == N_HELDOUT and d['stack_F'].shape == z['F'].shape
                and np.array_equal(d['stack_F'], z['F'], equal_nan=True))


def render(out_path='results/mismatch_load.png', data=DATA):
    d = np.load(data)
    mm, red, green_raw, floor = d['mm'], d['red'], d['green'], d['floor']
    # The red and value-only genomes are themselves in the box, so the best genome
    # available is never worse than either. On fresh worlds they can beat the
    # box-best cell by noise.
    green = np.maximum.reduce([red, green_raw, floor])

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    lo = min(-0.05, red.min() - 0.05)
    hi = max(green.max() + 0.08, 0.1)
    if lo < 0:
        ax.axhspan(lo, 0.0, color='#bbbbbb', alpha=0.16, zorder=0)
    ax.fill_between(mm, red, green, color='#e0b0b0', alpha=0.5, zorder=1)
    ax.plot(mm, green, '-o', color='#2a7a2a', lw=2.6, ms=6, zorder=3,
            label='best available (any genome in the box)')
    ax.plot(mm, red, '-o', color='#b03030', lw=2.6, ms=6, zorder=3,
            label='ancestral-tuned genome (fixed)')
    ax.plot(mm, floor, '--', color='#777', lw=1.6, zorder=2,
            label='ignores novelty (κ = 0)')
    ax.axhline(0, color='#555', lw=0.9, ls=':', zorder=2)

    i_big = int(np.argmax(green - red))
    y_label = 0.5 * (red[i_big] + min(green[i_big], floor[i_big]))   # clear of the grey line
    ax.annotate('mismatch\nload', (mm[i_big], y_label),
                fontsize=11, color='#9a4040', ha='center', va='center',
                style='italic', weight='bold')

    ax.set_xlabel('mismatch   (novelty tracks value  →  novelty is free)', fontsize=10)
    ax.set_ylabel('foraging fitness (held-out worlds)', fontsize=10)
    ax.set_title('Mismatch load', fontsize=12.5, weight='bold', pad=10)
    ax.set_ylim(lo, hi)
    ax.set_xlim(mm.min() - 0.03, mm.max() + 0.03)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=1,
              fontsize=9, framealpha=0.0, borderaxespad=0.0)
    ax.grid(alpha=0.18)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print('red  :', [f'{v:+.3f}' for v in red])
    print('green:', [f'{v:+.3f}' for v in green])
    print('saved', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/mismatch_load.png')
    ap.add_argument('--stack', default=STACK)
    ap.add_argument('--recompute', action='store_true', help='force the re-scoring')
    args = ap.parse_args()
    if args.recompute or not _cache_is_current(stack=args.stack):
        recompute(stack=args.stack)
    render(args.out)


if __name__ == '__main__':
    main()
