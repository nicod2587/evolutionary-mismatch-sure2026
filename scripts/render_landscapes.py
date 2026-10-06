"""
Poster figure: the R_bar x kappa fitness landscape at several mismatch levels, side
by side on one shared colour scale (bright = fit). On the poster this is
colour_plots.png.

  python -m scripts.render_poster_hero --stack results/ladder/ladder_stack.npz \
         --out results/poster_hero.png --rungs 0.00 0.10 0.25 0.50 1.00
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render_hero(stack_path, out_path, rungs_wanted=None):
    z = np.load(stack_path)
    R = z['R_bar']; K = z['kappa']; F = z['F']; rungs = z['rungs']

    idx = list(range(len(rungs)))
    if rungs_wanted is not None:
        show = [i for i in idx if any(abs(rungs[i] - w) < 1e-6 for w in rungs_wanted)]
    elif len(idx) <= 5:
        show = idx
    else:   # about 5 evenly spaced rungs, always including the first and last
        pick = np.linspace(0, len(idx) - 1, 5).round().astype(int)
        show = [idx[p] for p in sorted(set(pick))]
    if not show:
        raise SystemExit('no matching rungs to render')

    vmin = float(z['global_vmin']); vmax = float(z['global_vmax'])
    extent = [R.min(), R.max(), K.min(), K.max()]

    n = len(show)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n + 0.9, 3.15),
                             gridspec_kw={'wspace': 0.08})
    if n == 1:
        axes = [axes]
    im = None
    for ax, i in zip(axes, show):
        im = ax.imshow(F[i].T, origin='lower', extent=extent, aspect='auto',
                       cmap='plasma', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(f'mismatch = {rungs[i]:.2f}', fontsize=10, weight='bold', pad=4)
        ax.set_xlabel('R̄  (richness prior)', fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('κ  (novelty bonus)', fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

    axes[0].annotate('ancestral: high κ still fit', (0.5, -0.34), xycoords='axes fraction',
                     ha='center', fontsize=8.5, color='#8a5616', weight='bold')
    axes[-1].annotate('modern: only low κ fit', (0.5, -0.34), xycoords='axes fraction',
                      ha='center', fontsize=8.5, color='#2b7a86', weight='bold')

    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label('foraging fitness', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle('The fitness landscape deforms as the world modernises',
                 fontsize=12, weight='bold', y=1.04)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'shown rungs: {[f"{rungs[i]:.2f}" for i in show]}  vmin={vmin:.3f} vmax={vmax:.3f}')
    print(f'saved {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Render the landscape filmstrip')
    p.add_argument('--stack', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rungs', type=float, nargs='+', default=None)
    a = p.parse_args()
    render_hero(a.stack, a.out, a.rungs)
