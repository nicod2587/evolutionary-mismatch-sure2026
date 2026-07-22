"""
render_agent_single_patch.py — the poster's "model & agent" figure: one agent,
one stimulus, two views of the SAME run. Left = the resource-grid + trajectory
(visualize_model.ipynb style): the agent wanders (low scent, no gradient) while
far from the patch, then locks on and moves straight in as scent rises near it.
Right = the neuron's own internal state over that same run: membrane V and the
opportunity-cost signal beta_PFC — the two numbers the equations on this poster
panel are actually about.

Single patch is pinned to the world centre; the agent starts off-centre so the
"wander -> engage -> home in" shape reads as one continuous blue line.

Saves ga_results/poster_agent_single_patch.png.

Usage:
  python -m scripts.render_agent_single_patch --seed 8
"""
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.natural_world import AncestralWorld
from scripts.genetic_algorithm import Genome
from scripts.model_dynamics import params as base_params
from scripts.agent import Agent

CHAMPION = np.array([14.7, 4.33])   # ancestral champion (mean top-10% in box @ mm=0.10);
                                     # tau_pfc = genetic_algorithm.TAU_PFC_FIXED
START = (-620.0, -620.0)           # bottom-left corner: outside the scent gradient's reach
                                    # (sigma=110), so the early leg is genuine random walk


def render(out_path='ga_results/poster_agent_single_patch.png', seed=8, t_end=4000.0,
           mismatch_index=0.10):
    g = Genome.from_array(CHAMPION)
    p = g.to_params(base_params)
    world = AncestralWorld(rng=np.random.default_rng(seed), mismatch_index=mismatch_index,
                           n_patches=1, obs_noise=0.1, obs_noise_tau=50.0)
    world.patches[0].x = 0.0
    world.patches[0].y = 0.0
    agent = Agent(p, world=world, world_half=world.world_half, max_spikes=20000,
                  rng=np.random.default_rng(seed), wander_speed=4.0, step_size=5.0,
                  x0=START[0], y0=START[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = agent.simulate(t_span=(0.0, t_end))

    extent = [-world.world_half, world.world_half, -world.world_half, world.world_half]
    # render the stimulus FRESH (t=0), not post-drain: compute_grid(h.t[-1]) shows the
    # patch after the agent has already harvested it down, which washes the core out.
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

    # right panel: the neuron's own state over the identical run — V (membrane, left axis)
    # and beta_PFC (opportunity-cost / give-up signal, right axis) — different natural
    # scales, so twin axes rather than squeezing beta into a flat line near zero.
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
    harvested = float(np.sum(h.harvested_utility))
    print(f'saved {out_path}  (harvested={harvested:.1f}, n_steps={len(h.t)})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='ga_results/poster_agent_single_patch.png')
    ap.add_argument('--seed', type=int, default=8)
    ap.add_argument('--t_end', type=float, default=4000.0)
    ap.add_argument('--mismatch_index', type=float, default=0.10)
    a = ap.parse_args()
    render(a.out, a.seed, a.t_end, a.mismatch_index)


if __name__ == '__main__':
    main()
