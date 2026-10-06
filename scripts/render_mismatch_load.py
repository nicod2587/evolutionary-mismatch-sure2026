"""
Poster figure: the mismatch load (mismatch load.png on the poster).

  red    one fixed genome tuned to the ancestral world (params.ANCESTRAL_GENOME),
         dropped into each world in turn. Its fitness falls below zero as the
         world modernises.
  green  the best fitness any genome in the biological box (R_bar <= 25,
         kappa <= 15) reaches in that world, read off results/ladder/ladder_stack.npz.
         It stays high.

The gap between them is the mismatch load.

The red line needs 56 fresh simulations (7 worlds x 8 seeds, about 20 minutes), so
its numbers are cached in results/fig2_data.npz.

  python -m scripts.render_fig2                 # plot from the cache
  python -m scripts.render_fig2 --recompute     # re-run the simulations, then plot
"""
import argparse
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.fitness import BOX_R, BOX_K, FORAGING_REF_RATE, Genome, eval_seeds
from scripts.params import ANCESTRAL_GENOME, BASE_PARAMS

DATA = 'results/fig2_data.npz'
STACK = 'results/ladder/ladder_stack.npz'
LEVELS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00]


def _measure(genome_arr, mm, seeds, cap=20000, t_eval=25000.0):
    """Average (info_yield, engaged fraction, fitness) of one genome in one world.

    Same quantities as fitness.compute_fitness, kept separate because this was
    how the cached numbers were made (the engaged-fraction sum groups its terms
    slightly differently, which matters in the last decimal place)."""
    from scripts.world import World
    from scripts.agent import Agent
    p = Genome.from_array(genome_arr).to_params(BASE_PARAMS)
    iys, engs, fl = [], [], []
    for s in seeds:
        w = World(rng=np.random.default_rng(s), mismatch_index=mm,
                  space_initial_patches=False)   # as run for the poster
        a = Agent(p, w, max_spikes=cap, rng=np.random.default_rng(s))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            h = a.simulate(t_span=(0.0, t_eval))
        T = float(h.t[-1]); hu = np.asarray(h.harvested_utility)
        iy = 1.0 - np.exp(-((hu.sum()) / T) / FORAGING_REF_RATE) if T > 0 else 0.0
        dt = np.diff(h.t); wm = 0.5 * (h.eng_weight[:-1] + h.eng_weight[1:])
        eng = float(np.sum(dt * wm) / T) if T > 0 else 0.0
        iys.append(iy); engs.append(eng); fl.append(iy - 0.3 * eng)
    return float(np.mean(iys)), float(np.mean(engs)), float(np.mean(fl))


def _box_peak(stack, mm, box_r=BOX_R, box_k=BOX_K):
    """Best fitness inside the biological box at mismatch mm, and the kappa there."""
    z = np.load(stack)
    R, K, rungs, F = z['R_bar'], z['kappa'], z['rungs'], z['F']
    i = int(np.argmin(np.abs(rungs - mm)))
    rm = R <= box_r; km = K <= box_k
    sub = F[i][np.ix_(rm, km)]
    Kgrid = K[km]
    j = np.unravel_index(np.nanargmax(sub), sub.shape)
    return float(sub[j]), float(Kgrid[j[1]])


def recompute(out=DATA, stack=STACK):
    seeds = eval_seeds(8, 0)
    red, green, greenk, iy_red, eng_red = [], [], [], [], []
    for mm in LEVELS:
        gf, k_peak = _box_peak(stack, mm)
        iy, eng, rf = _measure(np.array(ANCESTRAL_GENOME), mm, seeds)
        red.append(rf); green.append(gf); greenk.append(k_peak)
        iy_red.append(iy); eng_red.append(eng)
        print(f'mm={mm:.2f}  fixed(red)={rf:+.3f}  landscape-best(green)={gf:+.3f}  '
              f'(kappa-at-peak={k_peak:.2f})  iy={iy:.3f}  eng={eng:.3f}', flush=True)
    np.savez(out, mm=LEVELS, red=red, green=green, greenk=greenk, iy_red=iy_red, eng_red=eng_red)
    print('saved', out)


def render(out_path='results/poster_fig2_draft.png', data=DATA):
    d = np.load(data)
    mm, red, green_raw = d['mm'], d['red'], d['green']
    # The fixed genome is itself one point in the box, and at the ancestral end it
    # can beat the grid's best cell by a hair of evaluation noise. Taking the max
    # keeps green >= red so the shaded gap is never negative.
    green = np.maximum(red, green_raw)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.axhspan(min(-0.30, red.min() - 0.03), 0.0, color='#bbbbbb', alpha=0.16, zorder=0)
    ax.fill_between(mm, red, green, color='#e0b0b0', alpha=0.5, zorder=1)
    ax.plot(mm, green, '-o', color='#2a7a2a', lw=2.6, ms=6, zorder=3,
            label='best available (any genome)')
    ax.plot(mm, red, '-o', color='#b03030', lw=2.6, ms=6, zorder=3,
            label='ancestral-tuned genome (fixed)')
    ax.axhline(0, color='#555', lw=0.9, ls=':', zorder=2)

    i_mid = int(np.argmin(np.abs(mm - 0.30)))
    y_mid = 0.5 * (red[i_mid] + green[i_mid])
    ax.annotate('mismatch\nload', (mm[i_mid], y_mid), fontsize=11, color='#9a4040',
                ha='center', va='center', style='italic', weight='bold')

    ax.set_xlabel('amount of mismatch   (ancestral → modern)', fontsize=10)
    ax.set_ylabel('foraging fitness', fontsize=10)
    ax.set_title('The fitness gap grows with the amount of mismatch',
                 fontsize=12.5, weight='bold', pad=10)
    ax.set_ylim(min(-0.30, red.min() - 0.05), 0.88); ax.set_xlim(0.07, 1.03)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2,
              fontsize=9, framealpha=0.0, borderaxespad=0.0)
    ax.grid(alpha=0.18)
    fig.tight_layout(); fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print('red  :', [f'{v:+.3f}' for v in red])
    print('green:', [f'{v:+.3f}' for v in green])
    print('saved', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/poster_fig2_draft.png')
    ap.add_argument('--recompute', action='store_true', help='re-run the simulations first')
    args = ap.parse_args()
    if args.recompute or not os.path.exists(DATA):
        recompute()
    render(args.out)


if __name__ == '__main__':
    main()
