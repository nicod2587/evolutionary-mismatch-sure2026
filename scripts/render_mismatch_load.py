"""
Figure: the mismatch load.

Red is one genome tuned to the ancestral world, dropped into each world in turn.
Green is the best genome in the biological box for that world. The gap is the load.

Both are re-scored on 32 held-out worlds the ladder never saw, so green isn't
flattered by being the luckiest of many noisy cells. The scores are cached in
results/mismatch_load_data.npz and redone when the ladder changes.

  python -m scripts.render_mismatch_load [--recompute]
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
N_HELDOUT = 32            # more worlds than a ladder cell: only 2 genomes per level


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
    green_genomes = [_box_best(F[int(np.argmin(np.abs(rungs - mm)))], R, K) for mm in levels]
    jobs = []
    for mm, g in zip(levels, green_genomes):
        jobs += [_job(red_genome, mm, z, seeds), _job(g, mm, z, seeds)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        scores = list(pool.map(evaluate_genome, jobs))
    red, green = scores[0::2], scores[1::2]
    for mm, g, rf, gf in zip(levels, green_genomes, red, green):
        print(f'mm={mm:.2f}  red={rf:+.3f}  green={gf:+.3f} at R={g[0]:.1f} k={g[1]:.2f}',
              flush=True)
    np.savez(out, mm=levels, red=red, green=green, green_genomes=np.array(green_genomes),
             red_genome=np.array(red_genome), n_heldout=N_HELDOUT,
             model=str(z['model']), stack_F=F)
    print('saved', out)


def _cache_is_current(data=DATA, stack=STACK):
    if not os.path.exists(data):
        return False
    with np.load(data) as d, np.load(stack) as z:
        return ('n_heldout' in d.files and int(d['n_heldout']) == N_HELDOUT
                and d['stack_F'].shape == z['F'].shape
                and np.array_equal(d['stack_F'], z['F'], equal_nan=True))


def render(out_path='results/mismatch_load.png', data=DATA):
    d = np.load(data)
    mm, red, green_raw = d['mm'], d['red'], d['green']
    # The red genome is itself in the box, so the best genome available is never
    # worse than it. On fresh worlds it can beat the box-best cell by noise.
    green = np.maximum(red, green_raw)

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
    ax.axhline(0, color='#555', lw=0.9, ls=':', zorder=2)

    i_big = int(np.argmax(green - red))
    ax.annotate('mismatch\nload', (mm[i_big], 0.5 * (red[i_big] + green[i_big])),
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
