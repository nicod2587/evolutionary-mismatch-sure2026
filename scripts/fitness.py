"""
fitness.py — the foraging-fitness objective scored at every cell of the R_bar x kappa landscape.

  genome  = (R_bar, kappa); tau_pfc is fixed at TAU_PFC_FIXED
  fitness = info_yield - 0.3 * opportunity_cost, averaged over fixed world seeds

Grounding: Charnov 1976 (marginal value theorem); McNamara & Dall 2010 (information as a
fitness resource); Hayden, Pearson & Platt 2011 and Kolling et al. 2012 (ACC/PFC
opportunity-cost signal); Monosov 2024 (curiosity gain kappa); De Block & Du Laing 2010
and Lim, Yong & Tan 2025 (supernormal stimuli / evolutionary mismatch).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

TAU_PFC_FIXED: float = 256.0  # ms-grounded PFC/ACC intrinsic timescale
FORAGING_REF_RATE: float = 1.0 / 200.0  # reference harvest rate for saturating info_yield
BOX_R, BOX_K = 25.0, 15.0  # biologically plausible R_bar / kappa range


@dataclass
class Genome:
    R_bar: float
    kappa: float

    @classmethod
    def from_array(cls, arr) -> 'Genome':
        return cls(R_bar=float(arr[0]), kappa=float(arr[1]))

    def to_params(self, base_params: dict) -> dict:
        p = dict(base_params)
        p['R_bar'] = self.R_bar
        p['kappa'] = self.kappa
        p['tau_pfc'] = TAU_PFC_FIXED
        return p


def eval_seeds(n_worlds: int = 6, base: int = 20260709) -> List[int]:
    """Fixed world seeds, so every landscape cell is scored on identical worlds."""
    return [int(np.random.default_rng([base, i]).integers(0, 2**31))
            for i in range(n_worlds)]


def compute_fitness(history, weights: Tuple[float, float] = (1.0, 0.3)) -> float:
    """F = w_yield * info_yield - w_opp * opportunity_cost.

    info_yield: REAL value harvested (not perceived scent), rate-normalised to [0, 1).
    opportunity_cost: fraction of time engaged on a patch (MVT cost of staying).
    Nothing here reads mismatch_index; the mismatch cost emerges from behaviour.
    """
    T = float(history.t[-1]) if len(history.t) > 0 else 0.0
    if T < 1.0 or len(history.t) < 2:
        return 0.0
    w_yield, w_opp = weights

    harvested = getattr(history, 'harvested_utility', None)
    total = float(np.sum(harvested)) if harvested is not None and len(harvested) else 0.0
    info_yield = float(1.0 - np.exp(-(total / T) / FORAGING_REF_RATE))

    eng = getattr(history, 'eng_weight', None)
    if eng is not None and len(eng) > 1:
        dt = np.diff(history.t)
        opportunity_cost = float(np.sum(dt * 0.5 * (eng[:-1] + eng[1:])) / T)
    else:
        opportunity_cost = 0.0

    return float(w_yield * info_yield - w_opp * opportunity_cost)


def evaluate_genome(args: tuple) -> float:  # module-level so picklable for ProcessPoolExecutor
    """Mean fitness of one genome across world seeds.

    args = (genome_arr, base_params, world_seeds, t_eval, max_spikes,
            fitness_weights, agent_kwargs, world_kwargs)
    """
    (genome_arr, base_params, world_seeds, t_eval,
     max_spikes, fitness_weights, agent_kwargs, world_kwargs) = args

    from scripts.natural_world import AncestralWorld
    from scripts.agent import Agent

    p_agent = Genome.from_array(genome_arr).to_params(base_params)
    scores = []
    for seed in world_seeds:
        world = AncestralWorld(rng=np.random.default_rng(seed), **world_kwargs)
        # agent RNG keyed to the world seed only: every genome faces the same wander noise
        agent = Agent(p_agent, world=world, world_half=world.world_half,
                      max_spikes=max_spikes, rng=np.random.default_rng(seed), **agent_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            history = agent.simulate(t_span=(0.0, float(t_eval)))
        scores.append(compute_fitness(history, weights=fitness_weights))
        del history
    return float(np.mean(scores))
