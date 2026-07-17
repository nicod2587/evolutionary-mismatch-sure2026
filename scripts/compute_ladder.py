"""
compute_ladder.py — recompute the Step-2 R_bar x kappa slice at each mismatch rung.

Produces one ladder_stack.npz holding a landscape F[rung] per mismatch level, all
scored with the IDENTICAL objective/seeds/tau as the Step-2 definitive slice
(scripts.plot_rk_slice) — only mismatch_index varies across rungs. This is the data
behind the mismatch-ladder GIF. It does NOT re-evolve anything.

Resumable: writes ladder_stack_partial.npz after each rung; re-running skips done rungs.

Run (HPC, multi-hour — one Step-2 slice per rung):
  python -u -m scripts.compute_ladder --res 60 --n_worlds 8 --t_eval 25000 \
         --n_jobs 32 --tau 256 --r_max 200 --k_max 50 --obs_noise 0.1 \
         --out_dir ga_results/ladder
"""
import argparse
import os
import time

import numpy as np

from scripts.model_dynamics import params as base_params
from scripts.genetic_algorithm import _evaluate_one
from scripts.genome_landscape import ga_fixed_seeds

DEFAULT_RUNGS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.0]


def _score_one_landscape(r_axis, k_axis, tau, seeds, t_eval, max_spikes,
                         mismatch_index, obs_noise, obs_noise_tau, n_jobs):
    """Score a single res x res R_bar x kappa grid at one mismatch level.
    Job tuple is identical to plot_rk_slice.main's."""
    world_kwargs = {'mismatch_index': mismatch_index,
                    'obs_noise': obs_noise, 'obs_noise_tau': obs_noise_tau}
    agent_kwargs = {'wander_speed': 4.0, 'step_size': 5.0}
    fitness_weights = (1.0, 0.3)
    jobs = []
    for ir in r_axis:
        for ik in k_axis:
            g = np.array([ir, ik, tau])
            jobs.append((g, base_params, seeds, t_eval, max_spikes,
                         fitness_weights, agent_kwargs, world_kwargs, 0))
    flat = np.full(len(jobs), np.nan)
    if n_jobs == 1:
        for i, job in enumerate(jobs):
            flat[i] = _evaluate_one(job)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futs = {pool.submit(_evaluate_one, job): i for i, job in enumerate(jobs)}
            for fut in as_completed(futs):
                flat[futs[fut]] = fut.result()
    return flat.reshape(len(r_axis), len(k_axis))


def compute_ladder(r_min, r_max, k_min, k_max, res, rungs, tau, n_worlds,
                   t_eval, max_spikes, n_jobs, obs_noise, obs_noise_tau,
                   seed_base, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    r_axis = np.linspace(r_min, r_max, res)
    k_axis = np.linspace(k_min, k_max, res)
    seeds = ga_fixed_seeds(n_worlds, seed_base)
    rungs = list(rungs)
    n = len(rungs)

    F = np.full((n, res, res), np.nan)
    done = np.zeros(n, dtype=bool)
    partial = os.path.join(out_dir, 'ladder_stack_partial.npz')

    # Resume: reuse a matching partial (same axes/rungs/tau).
    if os.path.exists(partial):
        try:
            with np.load(partial) as z:
                same = (z['F'].shape == F.shape
                        and np.allclose(z['rungs'], rungs)
                        and np.allclose(z['R_bar'], r_axis)
                        and np.allclose(z['kappa'], k_axis)
                        and float(z['tau']) == tau)
                if same:
                    F = z['F'].copy()
                    done = z['done_rungs'].copy()
                    print(f"  resuming: {int(done.sum())}/{n} rungs already scored")
        except Exception as e:
            print(f"  ignoring unreadable partial ({e}); starting fresh")

    def _save_partial():
        tmp = partial + '.tmp.npz'
        np.savez(tmp, R_bar=r_axis, kappa=k_axis, rungs=np.array(rungs),
                 F=F, tau=float(tau), done_rungs=done)
        os.replace(tmp, partial)

    t0 = time.perf_counter()
    for i, m in enumerate(rungs):
        if done[i]:
            continue
        print(f"[rung {i+1}/{n}] mismatch={m:.2f} ...", flush=True)
        F[i] = _score_one_landscape(r_axis, k_axis, tau, seeds, t_eval,
                                    max_spikes, m, obs_noise, obs_noise_tau, n_jobs)
        done[i] = True
        _save_partial()
        print(f"  rung {i+1} done, peak={np.nanmax(F[i]):.3f} "
              f"({time.perf_counter()-t0:.0f}s total)", flush=True)

    if not done.all():
        raise RuntimeError(f"ladder incomplete: {int((~done).sum())} rungs unscored")

    global_max_per_rung = np.array([float(np.nanmax(F[i])) for i in range(n)])
    out = os.path.join(out_dir, 'ladder_stack.npz')
    np.savez(out, R_bar=r_axis, kappa=k_axis, rungs=np.array(rungs), F=F,
             tau=float(tau), global_max_per_rung=global_max_per_rung,
             global_vmax=float(np.nanmax(F)), global_vmin=float(np.nanmin(F)))
    if os.path.exists(partial):
        os.remove(partial)
    print(f"ladder -> {out}  (global vmin/vmax {np.nanmin(F):.3f}/{np.nanmax(F):.3f})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--res', type=int, default=60)
    p.add_argument('--n_worlds', type=int, default=8)
    p.add_argument('--t_eval', type=float, default=25_000.0)
    p.add_argument('--max_spikes', type=int, default=4_000)
    p.add_argument('--n_jobs', type=int, default=1)
    p.add_argument('--tau', type=float, default=256.0)
    p.add_argument('--r_min', type=float, default=0.0)
    p.add_argument('--r_max', type=float, default=200.0)   # Step-2 default; override if WIDENed
    p.add_argument('--k_min', type=float, default=0.1)
    p.add_argument('--k_max', type=float, default=50.0)
    p.add_argument('--obs_noise', type=float, default=0.1)
    p.add_argument('--obs_noise_tau', type=float, default=50.0)
    p.add_argument('--seed_base', type=int, default=20260709)
    p.add_argument('--rungs', type=float, nargs='+', default=DEFAULT_RUNGS)
    p.add_argument('--out_dir', type=str, default='ga_results/ladder')
    args = p.parse_args()
    print(f"ladder: rungs={args.rungs} res={args.res} tau={args.tau} "
          f"R_bar[{args.r_min},{args.r_max}] kappa[{args.k_min},{args.k_max}]", flush=True)
    compute_ladder(args.r_min, args.r_max, args.k_min, args.k_max, args.res,
                   args.rungs, args.tau, args.n_worlds, args.t_eval, args.max_spikes,
                   args.n_jobs, args.obs_noise, args.obs_noise_tau, args.seed_base,
                   args.out_dir)


if __name__ == '__main__':
    main()
