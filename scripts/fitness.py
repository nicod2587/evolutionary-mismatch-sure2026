"""
How well a forager does, and how one cell of the fitness landscape gets its score.

A genome is just the two traits (R_bar, kappa). Its fitness in one world is

    fitness = info_yield - 0.3 * opportunity_cost

info_yield        real value harvested per unit time, squashed into [0, 1) with
                  1 - exp(-rate / FORAGING_REF_RATE). Scent doesn't count, only value.
opportunity_cost  fraction of the run spent engaged on a patch. Time on one patch
                  is time not spent finding a better one (Charnov 1976).

Nothing in here looks at mismatch_index. Any difference between ancestral and
modern worlds comes from how the same forager behaves in them.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List

import numpy as np

FORAGING_REF_RATE = 1.0 / 200.0   # harvest rate at which info_yield reaches 1 - 1/e
OPPORTUNITY_COST_WEIGHT = 0.3

# The "biological box": the part of trait space we treat as plausible. The landscape
# itself is computed over a wider range (R_bar up to 200, kappa up to 50).
BOX_R, BOX_K = 25.0, 15.0


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
        return p


def eval_seeds(n_worlds: int = 8, base: int = 20260709) -> List[int]:
    """A fixed list of world seeds, so every cell of the landscape is scored on
    exactly the same worlds."""
    return [int(np.random.default_rng([base, i]).integers(0, 2**31))
            for i in range(n_worlds)]


def compute_fitness(history) -> float:
    T = float(history.t[-1]) if len(history.t) > 0 else 0.0
    if T < 1.0 or len(history.t) < 2:
        return 0.0

    total = float(np.sum(history.harvested_utility))
    info_yield = float(1.0 - np.exp(-(total / T) / FORAGING_REF_RATE))

    eng = history.eng_weight
    dt = np.diff(history.t)
    opportunity_cost = float(np.sum(dt * 0.5 * (eng[:-1] + eng[1:])) / T)   # trapezoid rule

    return float(info_yield - OPPORTUNITY_COST_WEIGHT * opportunity_cost)


def evaluate_genome(args: tuple) -> float:
    """Mean fitness of one genome over a list of world seeds.

    args = (genome_arr, base_params, world_seeds, t_eval, max_spikes, world_kwargs),
    packed in one tuple so this can be handed to a ProcessPoolExecutor.
    """
    return float(np.mean(evaluate_genome_per_world(args)))


def evaluate_genome_per_world(args: tuple) -> np.ndarray:
    """Fitness of one genome in each world of the list (same args as
    evaluate_genome), so the spread between worlds can be reported too."""
    genome_arr, base_params, world_seeds, t_eval, max_spikes, world_kwargs = args

    from scripts.world import World
    from scripts.agent import Agent

    params = Genome.from_array(genome_arr).to_params(base_params)
    scores = []
    for seed in world_seeds:
        world = World(rng=np.random.default_rng(seed), **world_kwargs)
        # The forager's own randomness is seeded from the world seed alone, so every
        # genome meets the same worlds *and* the same movement noise.
        agent = Agent(params, world, max_spikes=max_spikes, rng=np.random.default_rng(seed))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            history = agent.simulate(t_span=(0.0, float(t_eval)))
        scores.append(compute_fitness(history))
    return np.array(scores)
