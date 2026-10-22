"""
Figure: the fitness landscape at several mismatch levels, on one colour scale.

  python -m scripts.render_landscapes --rungs 0 0.1 0.3 0.5 1
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter


def cell_edges(axis, log):
    """Edges of the cells centred on the grid points of `axis`."""
    a = np.log(axis) if log else np.asarray(axis, dtype=float)
    mid = 0.5 * (a[1:] + a[:-1])
    edges = np.concatenate([[a[0] - (mid[0] - a[0])], mid, [a[-1] + (a[-1] - mid[-1])]])
    return np.exp(edges) if log else edges


def render(stack_path, out_path, rungs_wanted=None, even=False):
    z = np.load(stack_path)
    R = z['R_bar']; K = z['kappa']; F = z['F']; rungs = z['rungs']
    log = 'spacing' in z.files and str(z['spacing']) == 'log'

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
    Re, Ke = cell_edges(R, log), cell_edges(K, log)
    plot_log = log and not even   # 'even' = plain linear axes (old poster style)

    n = len(show)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n + 0.9, 3.15),
                             gridspec_kw={'wspace': 0.08})
    if n == 1:
        axes = [axes]
    im = None
    for ax, i in zip(axes, show):
        im = ax.pcolormesh(Re, Ke, F[i].T, cmap='plasma', vmin=vmin, vmax=vmax,
                           shading='flat')
        if even:
            ax.set_xlim(0, 200); ax.set_ylim(0, 50)
        if plot_log:
            ax.set_xscale('log'); ax.set_yscale('log')
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
                axis.set_minor_formatter(NullFormatter())
            ax.tick_params(which='minor', length=0)
        ax.set_title(f'mismatch = {rungs[i]:.2f}', fontsize=10, weight='bold', pad=4)
        ax.set_xlabel('R̄  (richness prior)', fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('κ  (novelty bonus)', fontsize=8)
        else:
            ax.set_yticklabels([], minor=True); ax.set_yticklabels([])
        ax.tick_params(labelsize=7)

    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label('foraging fitness', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle('The fitness landscape as novelty comes loose from value',
                 fontsize=12, weight='bold', y=1.04)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'shown rungs: {[f"{rungs[i]:.2f}" for i in show]}  vmin={vmin:.3f} vmax={vmax:.3f}')
    print(f'saved {out_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Render the landscape filmstrip')
    p.add_argument('--stack', default='results/ladder_res40/ladder_stack.npz')
    p.add_argument('--out', default='results/landscapes.png')
    p.add_argument('--even', action='store_true', help='linear axes, R 0-200, kappa 0-50')
    p.add_argument('--rungs', type=float, nargs='+', default=None)
    a = p.parse_args()
    render(a.stack, a.out, a.rungs, a.even)
