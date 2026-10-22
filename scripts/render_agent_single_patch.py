"""
Figure: one forager, one patch. Left, its path through the scent. Right, V and beta
over the same run.
"""
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.world import World
from scripts.fitness import Genome
from scripts.params import ANCESTRAL_GENOME, BASE_PARAMS
from scripts.agent import Agent

START = (-620.0, -620.0)   # bottom-left corner, well outside the patch's scent


def render(out_path='results/agent_single_patch.png', seed=8, t_end=4000.0,
           mismatch_index=0.10):
    p = Genome.from_array(ANCESTRAL_GENOME).to_params(BASE_PARAMS)
    world = World(rng=np.random.default_rng(seed), mismatch_index=mismatch_index, n_patches=1,
                  space_initial_patches=False)   # as in the other scripts
    world.patches[0].x = 0.0
    world.patches[0].y = 0.0
    agent = Agent(p, world, max_spikes=20000, rng=np.random.default_rng(seed),
                  x0=START[0], y0=START[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = agent.simulate(t_span=(0.0, t_end))

    extent = [-world.world_half, world.world_half, -world.world_half, world.world_half]
    # Draw the patch as it was at the start. By the end of the run the forager has
    # eaten most of it and the centre would look washed out.
    grid = world.compute_grid(0.0, grid_size=160)

    fig, (ax_traj, ax_dyn) = plt.subplots(1, 2, figsize=(8.6, 3.8),
                                          gridspec_kw={'width_ratios': [1, 1.15], 'wspace': 0.28})

    ax_traj.imshow(grid, extent=extent, origin='lower', cmap='inferno',
                   vmin=0, vmax=grid.max())
    ax_traj.plot(h.x, h.y, color='#5ec8f5', lw=1.7, alpha=0.95)
    ax_traj.scatter([h.x[0]], [h.y[0]], s=42, facecolor='white', edgecolor='k', lw=1.0, zorder=5)
    ax_traj.annotate('start', (h.x[0], h.y[0]), textcoords='offset points', xytext=(7, 6),
                     fontsize=8, color='white')
    ax_traj.set_xlim(extent[0], extent[1]); ax_traj.set_ylim(extent[2], extent[3])
    ax_traj.set_aspect('equal')
    ax_traj.set_xticks([]); ax_traj.set_yticks([])
    for s in ax_traj.spines.values():
        s.set_visible(False)
    ax_traj.set_title('where it goes', fontsize=9.5, color='#444')

    # V and beta live on very different scales, so each gets its own y-axis.
    ax_beta = ax_dyn.twinx()
    ln_v, = ax_dyn.plot(h.t, h.V, color='#5ec8f5', lw=1.0, label='V  (membrane)')
    ln_b, = ax_beta.plot(h.t, h.beta_PFC, color='#e8532e', lw=1.3, label=r'$\beta_{PFC}$  (opp.-cost)')
    ax_dyn.set_xlabel('time', fontsize=8)
    ax_dyn.set_ylabel('V', fontsize=8, color='#3a9fcf')
    ax_beta.set_ylabel(r'$\beta_{PFC}$', fontsize=8, color='#c0431a')
    ax_dyn.tick_params(labelsize=7, axis='both')
    ax_dyn.tick_params(axis='y', labelcolor='#3a9fcf')
    ax_beta.tick_params(labelsize=7, axis='y', labelcolor='#c0431a')
    ax_dyn.legend(handles=[ln_v, ln_b], fontsize=7.5, frameon=False, loc='upper left')
    ax_dyn.spines['top'].set_visible(False)
    ax_beta.spines['top'].set_visible(False)
    ax_dyn.set_title('what it feels: fires when engaged, gives up as β catches up',
                     fontsize=9.5, color='#444')

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=170, bbox_inches='tight')
    print(f'saved {out_path}  (harvested={float(np.sum(h.harvested_utility)):.1f}, '
          f'n_steps={len(h.t)})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/agent_single_patch.png')
    ap.add_argument('--seed', type=int, default=8)
    ap.add_argument('--t_end', type=float, default=4000.0)
    ap.add_argument('--mismatch_index', type=float, default=0.10)
    a = ap.parse_args()
    render(a.out, a.seed, a.t_end, a.mismatch_index)


if __name__ == '__main__':
    main()
