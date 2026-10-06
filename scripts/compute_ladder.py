"""
Compute the R_bar x kappa fitness landscape at a series of mismatch levels ("rungs").

Every cell is scored with fitness.evaluate_genome on the same fixed worlds; the only
thing that changes from rung to rung is mismatch_index, and that only changes whether
novelty tracks real value (see world.py). The result is one file, ladder_stack.npz,
holding F[rung, R_bar, kappa]. All the figures read it.

Both traits are scale parameters that matter over orders of magnitude (the best kappa
moves by about a decade along the ladder), so by default both axes are spaced evenly
on a log scale: a coarse grid then still has several cells inside the biological box.

Cost is res * res * len(rungs) cells, each one n_worlds simulations. The rungs are
scored in PRIORITY order (the two ends first, then the reference rung, then the
rest) and ladder_stack_partial.npz is saved after each one, so a job killed at its
time limit still leaves the most useful rungs. Run the same command again and it
picks up from the last finished rung.

  python -u -m scripts.compute_ladder --res 10 --n_jobs 32   # first look -> results/ladder_res10
  python -u -m scripts.compute_ladder --res 40 --n_jobs 32   # full ladder -> results/ladder_res40
"""
import argparse
import os
import time

import numpy as np

from scripts.params import BASE_PARAMS, TAU_PFC, ANCESTRAL_MISMATCH
from scripts.fitness import BOX_R, BOX_K, evaluate_genome_per_world, eval_seeds

# Bumped whenever the model changes in a way that changes fitness values, so a
# partial run from an older model is never resumed into a new one.
#   poster-v1  mismatch also moved patch count, value, scent shape and travel speed
#   pure-v2    mismatch moves only h (world.py)
MODEL_VERSION = 'pure-v2'

DEFAULT_RUNGS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0]


def make_axis(lo, hi, res, spacing):
    if spacing == 'log':
        return np.geomspace(lo, hi, res)
    return np.linspace(lo, hi, res)


def priority_order(rungs):
    """Indices of rungs in the order to score them: both ends, the reference rung
    (ANCESTRAL_MISMATCH), the middle, then the rest bisecting the biggest gaps."""
    rungs = list(rungs)
    first = [int(np.argmin(rungs)), int(np.argmax(rungs)),
             int(np.argmin(np.abs(np.array(rungs) - ANCESTRAL_MISMATCH))),
             int(np.argmin(np.abs(np.array(rungs) - 0.5)))]
    order = []
    for i in first:
        if i not in order:
            order.append(i)
    while len(order) < len(rungs):
        done = sorted(rungs[i] for i in order)
        gaps = [(b - a, (a + b) / 2) for a, b in zip(done[:-1], done[1:])] or [(0, 0)]
        mid = max(gaps)[1]
        left = [i for i in range(len(rungs)) if i not in order]
        order.append(min(left, key=lambda i: abs(rungs[i] - mid)))
    return order


def ancestral_genome(F_rung, r_axis, k_axis, top=0.10, box_r=BOX_R, box_k=BOX_K):
    """The 'ancestral-tuned' genome: mean (R_bar, kappa) of the top `top` share of
    cells inside the biological box. Used for params.ANCESTRAL_GENOME."""
    rm = r_axis <= box_r
    km = k_axis <= box_k
    sub = F_rung[np.ix_(rm, km)]
    RR, KK = np.meshgrid(r_axis[rm], k_axis[km], indexing='ij')
    n = max(1, int(round(top * sub.size)))
    best = np.argsort(sub, axis=None)[::-1][:n]
    return float(RR.flat[best].mean()), float(KK.flat[best].mean())


def score_landscape(r_axis, k_axis, seeds, t_eval, max_spikes, world_kwargs, n_jobs):
    """Score every (R_bar, kappa) pair on the grid for one world setting. Returns
    the fitness in each world, shape (len(r_axis), len(k_axis), len(seeds))."""
    jobs = [(np.array([r, k]), BASE_PARAMS, seeds, t_eval, max_spikes, world_kwargs)
            for r in r_axis for k in k_axis]
    flat = np.full((len(jobs), len(seeds)), np.nan)
    if n_jobs == 1:
        for i, job in enumerate(jobs):
            flat[i] = evaluate_genome_per_world(job)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = {pool.submit(evaluate_genome_per_world, job): i
                    for i, job in enumerate(jobs)}
            for fut in as_completed(futs):
                flat[futs[fut]] = fut.result()
    return flat.reshape(len(r_axis), len(k_axis), len(seeds))


def compute_ladder(r_min, r_max, k_min, k_max, res, spacing, rungs, n_worlds, t_eval,
                   max_spikes, n_jobs, obs_noise, obs_noise_tau, seed_base, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    r_axis = make_axis(r_min, r_max, res, spacing)
    k_axis = make_axis(k_min, k_max, res, spacing)
    seeds = eval_seeds(n_worlds, seed_base)
    rungs = sorted(float(m) for m in rungs)
    n = len(rungs)

    F_worlds = np.full((n, res, res, len(seeds)), np.nan)
    done = np.zeros(n, dtype=bool)
    partial = os.path.join(out_dir, 'ladder_stack_partial.npz')
    meta = dict(R_bar=r_axis, kappa=k_axis, rungs=np.array(rungs), tau=TAU_PFC,
                spacing=spacing, model=MODEL_VERSION, seeds=np.array(seeds),
                t_eval=float(t_eval), max_spikes=int(max_spikes),
                obs_noise=float(obs_noise), obs_noise_tau=float(obs_noise_tau))

    # Pick up a previous partial run, but only if it is the same model, grid and rungs.
    if os.path.exists(partial):
        try:
            with np.load(partial) as z:
                same = ('model' in z.files and str(z['model']) == MODEL_VERSION
                        and z['F_worlds'].shape == F_worlds.shape
                        and np.allclose(z['rungs'], rungs)
                        and np.allclose(z['R_bar'], r_axis)
                        and np.allclose(z['kappa'], k_axis)
                        and np.array_equal(z['seeds'], seeds)
                        and float(z['tau']) == TAU_PFC
                        and float(z['t_eval']) == float(t_eval))
                if same:
                    F_worlds = z['F_worlds'].copy()
                    done = z['done_rungs'].copy()
                    print(f"  resuming: {int(done.sum())}/{n} rungs already scored")
                else:
                    print("  partial run is from a different model/grid; starting fresh")
        except Exception as e:
            print(f"  ignoring unreadable partial ({e}); starting fresh")

    def save_partial():
        tmp = partial + '.tmp.npz'
        np.savez(tmp, F_worlds=F_worlds, done_rungs=done, **meta)
        os.replace(tmp, partial)

    t0 = time.perf_counter()
    for step, i in enumerate(priority_order(rungs)):
        if done[i]:
            continue
        m = rungs[i]
        print(f"[{step+1}/{n}] mismatch={m:.2f} ...", flush=True)
        world_kwargs = {'mismatch_index': m, 'obs_noise': obs_noise,
                        'obs_noise_tau': obs_noise_tau,
                        'space_initial_patches': False}
        F_worlds[i] = score_landscape(r_axis, k_axis, seeds, t_eval, max_spikes,
                                      world_kwargs, n_jobs)
        done[i] = True
        save_partial()
        print(f"  done, peak={np.nanmax(F_worlds[i].mean(axis=-1)):.3f} "
              f"({time.perf_counter()-t0:.0f}s total)", flush=True)

    # F is the mean over worlds (what every figure plots); F_se its standard error.
    F = F_worlds.mean(axis=-1)
    F_se = F_worlds.std(axis=-1, ddof=1) / np.sqrt(len(seeds))
    out = os.path.join(out_dir, 'ladder_stack.npz')
    np.savez(out, F=F, F_se=F_se, F_worlds=F_worlds,
             global_max_per_rung=np.nanmax(F, axis=(1, 2)),
             global_vmax=float(np.nanmax(F)), global_vmin=float(np.nanmin(F)), **meta)
    os.remove(partial)
    print(f"saved {out}  (fitness range {np.nanmin(F):.3f} to {np.nanmax(F):.3f})")
    return out


def summarise(path):
    """Print the best cell per rung (whole grid and inside the box) and the
    ancestral-tuned genome that params.ANCESTRAL_GENOME should hold."""
    z = np.load(path)
    R, K, F, rungs = z['R_bar'], z['kappa'], z['F'], z['rungs']
    rm, km = R <= BOX_R, K <= BOX_K
    model = str(z['model']) if 'model' in z.files else 'poster-v1'
    spacing = str(z['spacing']) if 'spacing' in z.files else 'linear'
    print(f"model={model}  grid {len(R)}x{len(K)} ({spacing})  "
          f"box has {rm.sum()}x{km.sum()} cells")
    if 'F_se' in z.files:
        print(f"standard error of a cell (between worlds): median {np.nanmedian(z['F_se']):.3f}, "
              f"max {np.nanmax(z['F_se']):.3f}")
    for i, m in enumerate(rungs):
        a = np.unravel_index(np.nanargmax(F[i]), F[i].shape)
        sub = F[i][np.ix_(rm, km)]
        b = np.unravel_index(np.nanargmax(sub), sub.shape)
        print(f"  mismatch={m:.2f}  best {F[i][a]:+.3f} at R={R[a[0]]:.1f} k={K[a[1]]:.2f}"
              f"  | box best {sub[b]:+.3f} at R={R[rm][b[0]]:.1f} k={K[km][b[1]]:.2f}")
    i = int(np.argmin(np.abs(rungs - ANCESTRAL_MISMATCH)))
    r, k = ancestral_genome(F[i], R, K)
    print(f"ancestral-tuned genome (top 10% of box at mismatch={rungs[i]:.2f}): "
          f"ANCESTRAL_GENOME = ({r:.1f}, {k:.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--res', type=int, default=10, help='grid points per axis')
    p.add_argument('--spacing', choices=['log', 'linear'], default='log')
    p.add_argument('--n_worlds', type=int, default=8, help='worlds averaged per cell')
    p.add_argument('--t_eval', type=float, default=25_000.0, help='simulated time per world')
    p.add_argument('--max_spikes', type=int, default=20_000)
    p.add_argument('--n_jobs', type=int, default=1, help='worker processes')
    p.add_argument('--r_min', type=float, default=1.0)
    p.add_argument('--r_max', type=float, default=200.0)
    p.add_argument('--k_min', type=float, default=0.1)
    p.add_argument('--k_max', type=float, default=50.0)
    p.add_argument('--obs_noise', type=float, default=0.1)
    p.add_argument('--obs_noise_tau', type=float, default=50.0)
    p.add_argument('--seed_base', type=int, default=20260709)
    p.add_argument('--rungs', type=float, nargs='+', default=DEFAULT_RUNGS)
    p.add_argument('--out_dir', type=str, default=None,
                   help='default: results/ladder_res<res>')
    p.add_argument('--summarise', type=str, default=None, metavar='STACK',
                   help='only print a summary of an existing ladder_stack.npz')
    a = p.parse_args()
    if a.summarise:
        summarise(a.summarise)
        return
    a.out_dir = a.out_dir or f'results/ladder_res{a.res}'
    print(f"ladder ({MODEL_VERSION}): rungs={a.rungs} res={a.res} ({a.spacing}) "
          f"R_bar[{a.r_min},{a.r_max}] kappa[{a.k_min},{a.k_max}]", flush=True)
    out = compute_ladder(a.r_min, a.r_max, a.k_min, a.k_max, a.res, a.spacing, a.rungs,
                         a.n_worlds, a.t_eval, a.max_spikes, a.n_jobs, a.obs_noise,
                         a.obs_noise_tau, a.seed_base, a.out_dir)
    summarise(out)


if __name__ == '__main__':
    main()
