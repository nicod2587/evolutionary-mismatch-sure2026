"""
Phone-sized (1080 x 1920) video of the ancestral-tuned forager, with V and beta
scrolling underneath. Needs imageio-ffmpeg. Saves results/forager_<label>.mp4.

  python -m scripts.render_forager_video                       # ancestral world
  python -m scripts.render_forager_video --mismatch_index 1.0  # modern world
"""
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg

from scripts.world import World
from scripts.fitness import Genome
from scripts.params import ANCESTRAL_GENOME, BASE_PARAMS
from scripts.agent import Agent

V_COLOR, B_COLOR = '#1f78b4', '#d9480f'      # traces on the light page
BG, FG, DIM = '#ffffff', '#1a1a1a', '#6b6b76'
# the world square stays dark (inferno on black, as on the poster) inside the light page
SCENT_CMAP = 'inferno'
MAP_TRAIL, MAP_FEED, MAP_LEAVE, MAP_TEXT = '#5ec8f5', '#5ec8f5', '#ff8a5c', '#f2f2f2'
VIDEO_MIN_SEP = 400.0   # patch spacing for the video only: enough that neighbouring glows
                        # (~3 sigma = 330) don't merge, loose enough that patches can land anywhere
TRAIL_SPAN = 6000.0     # sim time of path shown behind the forager, so the trail doesn't clutter
LEAVE_FLASH = 500.0     # how long (sim time) "gave up, leaving" stays on screen

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()


def simulate(seed, t_end, mismatch_index, n_frames):
    p = Genome.from_array(ANCESTRAL_GENOME).to_params(BASE_PARAMS)
    world = World(rng=np.random.default_rng(seed), mismatch_index=mismatch_index,
                  min_patch_separation=VIDEO_MIN_SEP)
    leave_times = []
    despawn = world._despawn

    def _logged_despawn(patch, t):   # note each give-up for the on-screen counter
        leave_times.append(float(t))
        despawn(patch, t)

    world._despawn = _logged_despawn
    agent = Agent(p, world, max_spikes=20000, rng=np.random.default_rng(seed))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = agent.simulate(t_span=(0.0, t_end), record_frames=n_frames)
    return world, h, np.array(leave_times)


def scent_grid(patches, prof, X, Y, t):
    """Scent field for one frame's patch snapshot (same maths as World.compute_grid)."""
    total = np.zeros_like(X)
    for (px, py, sal) in patches:
        if sal <= 1e-7:
            continue
        R = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
        env = prof['A'] * np.exp(-((R ** 2) / (2 * prof['sigma'] ** 2)) ** (prof['n'] / 2))
        total += sal * env * (1.0 + prof['alpha'] * np.sin(prof['omega'] * t - prof['k'] * R))
    return total


def render(out_path, seed=21, t_end=18000.0, mismatch_index=0.0, seconds=40, fps=60,
           window=3000.0, grid_size=140):
    n_frames = int(seconds * fps)
    world, h, leave_times = simulate(seed, t_end, mismatch_index, n_frames)
    frames = h.frames
    prof = world.profile
    half = world.world_half
    xs = np.linspace(-half, half, grid_size)
    X, Y = np.meshgrid(xs, xs)
    vmax = prof['A'] * (1.0 + prof['alpha'])
    fire_times = np.asarray(h.fire_times)
    harvest_cum = np.concatenate([[0.0], np.cumsum(h.harvested_utility)])
    dt_frame = t_end / max(n_frames - 1, 1)

    world_name = ('ancestral world' if mismatch_index <= 0.0 else
                  'modern world' if mismatch_index >= 1.0 else f'mismatch = {mismatch_index:.2f}')

    fig = plt.figure(figsize=(6, 10.667), dpi=180, facecolor=BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.42, 0.42], hspace=0.28,
                          left=0.13, right=0.95, top=0.9, bottom=0.05)
    ax_w = fig.add_subplot(gs[0]); ax_v = fig.add_subplot(gs[1]); ax_b = fig.add_subplot(gs[2])
    fig.text(0.5, 0.965, f'One forager, {world_name}', ha='center', color=FG,
             fontsize=15, weight='bold')
    fig.text(0.5, 0.935, 'fires while it feeds, leaves once β catches up', ha='center',
             color=DIM, fontsize=10)

    img = ax_w.imshow(np.zeros((grid_size, grid_size)), extent=[-half, half, -half, half],
                      origin='lower', cmap=SCENT_CMAP, vmin=0, vmax=vmax)
    trail, = ax_w.plot([], [], color=MAP_TRAIL, lw=1.4, alpha=0.9)
    dot = ax_w.scatter([0], [0], s=70, facecolor='white', edgecolor='k', lw=1.0, zorder=5)
    status = ax_w.text(0.03, 0.96, '', transform=ax_w.transAxes, color=MAP_TEXT, fontsize=11,
                       va='top', weight='bold')
    counter = ax_w.text(0.5, -0.03, '', transform=ax_w.transAxes, color=FG, fontsize=10,
                        ha='center', va='top')
    ax_w.set_xticks([]); ax_w.set_yticks([])
    for s in ax_w.spines.values():
        s.set_visible(False)

    for ax, lab, col in ((ax_v, 'V  membrane (mV)', V_COLOR), (ax_b, 'β  opportunity cost', B_COLOR)):
        ax.set_facecolor(BG)
        ax.set_title(lab, color=col, fontsize=11, loc='left')
        ax.tick_params(colors=DIM, labelsize=8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(DIM)
    ax_b.set_xlabel('time', color=DIM, fontsize=9)
    ln_v, = ax_v.plot([], [], color=V_COLOR, lw=0.8)
    ln_b, = ax_b.plot([], [], color=B_COLOR, lw=1.6)
    ax_v.set_ylim(min(h.V.min(), -66) - 0.5, -54.5)
    ax_b.set_ylim(-0.05 * max(h.beta_PFC.max(), 1e-3), 1.1 * max(h.beta_PFC.max(), 1e-3))

    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=6000,
                          extra_args=['-pix_fmt', 'yuv420p', '-movflags', '+faststart'])
    with writer.saving(fig, out_path, dpi=180):
        for i, fr in enumerate(frames):
            t = fr['t']
            img.set_data(scent_grid(fr['patches'], prof, X, Y, t))
            k = np.searchsorted(h.t, t, side='right')
            k0 = np.searchsorted(h.t, t - TRAIL_SPAN)
            trail.set_data(h.x[k0:k], h.y[k0:k])
            dot.set_offsets([[fr['x'], fr['y']]])

            feeding = np.any((fire_times > t - 3 * dt_frame) & (fire_times <= t))
            just_left = np.any((leave_times > t - LEAVE_FLASH) & (leave_times <= t))
            if just_left:
                status.set_text('gave up, leaving'); status.set_color(MAP_LEAVE)
            elif feeding:
                status.set_text('feeding'); status.set_color(MAP_FEED)
            else:
                status.set_text('searching'); status.set_color(MAP_TEXT)
            dot.set_facecolor(MAP_FEED if feeding else 'white')
            n_spk = int(np.searchsorted(fire_times, t, side='right'))
            n_left = int(np.sum(leave_times <= t))
            counter.set_text(f'value harvested {harvest_cum[n_spk]:.0f}   ·   patches finished {n_left}')

            lo = max(0.0, t - window)
            ks = np.searchsorted(h.t, lo)
            ln_v.set_data(h.t[ks:k], h.V[ks:k])
            ln_b.set_data(h.t[ks:k], h.beta_PFC[ks:k])
            for ax in (ax_v, ax_b):
                ax.set_xlim(lo, lo + window if t < window else t)
            writer.grab_frame(facecolor=BG)
            if i % 150 == 0:
                print(f'  frame {i}/{len(frames)}')

    print(f'saved {out_path}  (harvested={harvest_cum[-1]:.1f}, patches left={len(leave_times)})')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--t_end', type=float, default=18000.0)
    ap.add_argument('--mismatch_index', type=float, default=0.0)
    ap.add_argument('--seconds', type=float, default=40)
    ap.add_argument('--fps', type=int, default=60)
    a = ap.parse_args()
    label = 'ancestral' if a.mismatch_index <= 0.0 else f'mm{a.mismatch_index:.2f}'
    out = a.out or f'results/forager_{label}.mp4'
    render(out, a.seed, a.t_end, a.mismatch_index, a.seconds, a.fps)


if __name__ == '__main__':
    main()
