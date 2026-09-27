"""
Compute the R_bar x kappa fitness landscape at a series of mismatch levels ("rungs").

Every cell is scored with fitness.evaluate_genome on the same fixed worlds; the only
thing that changes from rung to rung is mismatch_index. The result is one file,
ladder_stack.npz, holding F[rung, R_bar, kappa]. All the poster figures read it.

This is slow (about 35 hours on 32 cores for the full 60 x 60 x 11 run), so it saves
ladder_stack_partial.npz after each rung. Run the same command again and it picks
up from the last finished rung.

The defaults reproduce the poster run (see run_landscape.sh):
  python -u -m scripts.compute_ladder --n_jobs 32
"""
import argparse
import os
import time

import numpy as np

from scripts.params import BASE_PARAMS, TAU_PFC
from scripts.fitness import evaluate_genome, eval_seeds

DEFAULT_RUNGS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]


def score_landscape(r_axis, k_axis, seeds, t_eval, max_spikes, world_kwargs, n_jobs):
    """Score every (R_bar, kappa) pair on the grid for one world setting."""
    jobs = [(np.array([r, k]), BASE_PARAMS, seeds, t_eval, max_spikes, world_kwargs)
            for r in r_axis for k in k_axis]
    flat = np.full(len(jobs), np.nan)
    if n_jobs == 1:
        for i, job in enumerate(jobs):
            flat[i] = evaluate_genome(job)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = {pool.submit(evaluate_genome, job): i for i, job in enumerate(jobs)}
            for fut in as_completed(futs):
                flat[futs[fut]] = fut.result()
    return flat.reshape(len(r_axis), len(k_axis))


def compute_ladder(r_min, r_max, k_min, k_max, res, rungs, n_worlds, t_eval, max_spikes,
                   n_jobs, obs_noise, obs_noise_tau, seed_base, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    r_axis = np.linspace(r_min, r_max, res)
    k_axis = np.linspace(k_min, k_max, res)
    seeds = eval_seeds(n_worlds, seed_base)
    rungs = list(rungs)
    n = len(rungs)

    F = np.full((n, res, res), np.nan)
    done = np.zeros(n, dtype=bool)
    partial = os.path.join(out_dir, 'ladder_stack_partial.npz')

    # Pick up a previous partial run, but only if it used the same grid and rungs.
    if os.path.exists(partial):
        try:
            with np.load(partial) as z:
                if (z['F'].shape == F.shape
                        and np.allclose(z['rungs'], rungs)
                        and np.allclose(z['R_bar'], r_axis)
                        and np.allclose(z['kappa'], k_axis)
                        and float(z['tau']) == TAU_PFC):
                    F = z['F'].copy()
                    done = z['done_rungs'].copy()
                    print(f"  resuming: {int(done.sum())}/{n} rungs already scored")
        except Exception as e:
            print(f"  ignoring unreadable partial ({e}); starting fresh")

    def save_partial():
        tmp = partial + '.tmp.npz'
        np.savez(tmp, R_bar=r_axis, kappa=k_axis, rungs=np.array(rungs),
                 F=F, tau=TAU_PFC, done_rungs=done)
        os.replace(tmp, partial)

    t0 = time.perf_counter()
    for i, m in enumerate(rungs):
        if done[i]:
            continue
        print(f"[rung {i+1}/{n}] mismatch={m:.2f} ...", flush=True)
        world_kwargs = {'mismatch_index': m, 'obs_noise': obs_noise,
                        'obs_noise_tau': obs_noise_tau,
                        'space_initial_patches': False}   # as run for the poster
        F[i] = score_landscape(r_axis, k_axis, seeds, t_eval, max_spikes, world_kwargs, n_jobs)
        done[i] = True
        save_partial()
        print(f"  rung {i+1} done, peak={np.nanmax(F[i]):.3f} "
              f"({time.perf_counter()-t0:.0f}s total)", flush=True)

    global_max_per_rung = np.array([float(np.nanmax(F[i])) for i in range(n)])
    out = os.path.join(out_dir, 'ladder_stack.npz')
    np.savez(out, R_bar=r_axis, kappa=k_axis, rungs=np.array(rungs), F=F,
             tau=TAU_PFC, global_max_per_rung=global_max_per_rung,
             global_vmax=float(np.nanmax(F)), global_vmin=float(np.nanmin(F)))
    os.remove(partial)
    print(f"saved {out}  (fitness range {np.nanmin(F):.3f} to {np.nanmax(F):.3f})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--res', type=int, default=60, help='grid points per axis')
    p.add_argument('--n_worlds', type=int, default=8, help='worlds averaged per cell')
    p.add_argument('--t_eval', type=float, default=25_000.0, help='simulated time per world')
    p.add_argument('--max_spikes', type=int, default=20_000)
    p.add_argument('--n_jobs', type=int, default=1, help='worker processes')
    p.add_argument('--r_min', type=float, default=0.0)
    p.add_argument('--r_max', type=float, default=200.0)
    p.add_argument('--k_min', type=float, default=0.1)
    p.add_argument('--k_max', type=float, default=50.0)
    p.add_argument('--obs_noise', type=float, default=0.1)
    p.add_argument('--obs_noise_tau', type=float, default=50.0)
    p.add_argument('--seed_base', type=int, default=20260709)
    p.add_argument('--rungs', type=float, nargs='+', default=DEFAULT_RUNGS)
    p.add_argument('--out_dir', type=str, default='results/ladder')
    a = p.parse_args()
    print(f"ladder: rungs={a.rungs} res={a.res} "
          f"R_bar[{a.r_min},{a.r_max}] kappa[{a.k_min},{a.k_max}]", flush=True)
    compute_ladder(a.r_min, a.r_max, a.k_min, a.k_max, a.res, a.rungs, a.n_worlds,
                   a.t_eval, a.max_spikes, a.n_jobs, a.obs_noise, a.obs_noise_tau,
                   a.seed_base, a.out_dir)


if __name__ == '__main__':
    main()
