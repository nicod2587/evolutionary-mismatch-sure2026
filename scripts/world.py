"""
The world the forager lives in: a square arena holding a few information patches.

Each patch has a real value that the forager drains by sampling it, and a scent
that the forager actually perceives and follows. The whole project turns on how
those two relate, which is set by `mismatch_index` (0 = ancestral, 1 = modern):

  scent   = value_fraction + NOVELTY_WEIGHT * kappa * novelty
  novelty = h * (value_fraction * freshness) + (1 - h) * PATCH_NOVELTY,   h = 1 - mismatch

In the ancestral world (h = 1) novelty only exists while a patch still has value,
so a drained patch smells drained. In the modern world (h = 0) every patch looks
equally novel whatever is left in it, so an empty patch keeps its pull. That is
the supernormal stimulus (De Block & Du Laing 2010; Lim, Yong & Tan 2025).

mismatch_index changes h and nothing else. The number of patches, their value, how
far their scent carries and how fast the forager travels are the same at every
level, and a given seed lays out the same map at every level. So any difference
between levels comes from novelty coming loose from value, not from a richer or
easier world.

Patches follow the marginal value theorem set-up (Charnov 1976): they deplete as
the forager works them, and a spent patch is removed once the forager leaves it,
then replaced after a random delay somewhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# The patches, the same at every mismatch level.
#   A, sigma, n         spatial shape of the scent: A * exp(-(r^2 / 2 sigma^2)^(n/2)).
#                       With n near 1 this falls off exponentially rather than like a
#                       Gaussian, so a patch can be picked up from a long way off.
#   alpha, omega, k     a small travelling ripple: * (1 + alpha sin(omega t - k r))
#   max_utility         average real value a fresh patch holds
PATCH_PROFILE = dict(
    A=30.0, sigma=110.0, n=1.1,        # wide, easy-to-find patches
    alpha=0.07, omega=0.04, k=0.0,     # steady scent
    max_utility=40.0,                  # few patches, each worth a lot
)
N_PATCHES = 3

PATCH_NOVELTY = 1.0      # how novel every patch looks in the fully modern world
NOVELTY_WEIGHT = 0.5     # weight of the kappa * novelty term against the value term
FRESHNESS_BUDGET = 150.0 # samples after which a patch is fully familiar (freshness 0)
VALUE_CV = 0.6           # spread of real value between patches (coefficient of variation)


def resource_profile(mismatch_index: float) -> Dict[str, float]:
    """Patch parameters at a point on the ancestral (0) to modern (1) axis. Only
    mismatch_index itself differs between points; Patch.scent reads it."""
    profile = dict(PATCH_PROFILE)
    profile['mismatch_index'] = float(np.clip(mismatch_index, 0.0, 1.0))
    return profile


@dataclass
class Patch:
    """One information patch.

    current_utility is the real value left in it. The forager harvests this on
    every spike and it is what fitness counts. sensory_salience is the scent the
    patch gives off (with kappa = 1); the forager reweights the novelty part by its
    own kappa when it perceives the patch (Agent._patch_intensity).
    """
    x: float
    y: float
    profile: Dict[str, float]
    max_utility: float
    current_utility: float
    sensory_salience: float = 1.0
    patch_id: int = 0
    spawn_time: float = 0.0
    fade_in_window: float = 0.0          # a respawned patch fades in over this long
    n_samples: int = 0                   # how many times the forager has sampled it
    freshness_budget: float = FRESHNESS_BUDGET

    def deplete(self, amount: float) -> float:
        """Take up to `amount` of real value. Returns what was actually taken, which
        is 0 on an empty patch even if it still smells good."""
        amount = max(0.0, float(amount))
        harvested = min(amount, self.current_utility)
        self.current_utility = max(0.0, self.current_utility - amount)
        self.n_samples += 1
        return harvested

    def freshness(self) -> float:
        """1 for an untouched patch, falling linearly to 0 as it gets sampled."""
        if self.freshness_budget <= 0:
            return 0.0
        f = 1.0 - self.n_samples / self.freshness_budget
        return 0.0 if f < 0.0 else f

    def value_fraction(self) -> float:
        """Share of the patch's real value still left, from 1 (full) to 0 (empty)."""
        if self.max_utility <= 0:
            return 0.0
        # A plain min/max clamp: np.clip on a Python float is slow, and this runs
        # hundreds of thousands of times per simulation.
        vf = self.current_utility / self.max_utility
        return 0.0 if vf < 0.0 else (1.0 if vf > 1.0 else vf)

    def scent(self, novelty_gain: float = 1.0, value_noise: float = 0.0) -> float:
        """Perceived scent of the patch (before the spatial envelope), for a forager
        with curiosity gain kappa = novelty_gain. See the module docstring for the
        formula. `value_noise` blurs the forager's reading of how much value is left
        (World.obs_noise_at)."""
        vf = self.value_fraction()
        if value_noise != 0.0:
            vf += value_noise
            vf = 0.0 if vf < 0.0 else (1.0 if vf > 1.0 else vf)
        h = 1.0 - float(self.profile['mismatch_index'])
        novelty = h * (vf * self.freshness()) + (1.0 - h) * PATCH_NOVELTY
        return vf + NOVELTY_WEIGHT * novelty_gain * novelty

    def fade_in_factor(self, t: float) -> float:
        """0 at the moment the patch spawns, rising to 1 over fade_in_window."""
        if self.fade_in_window <= 0:
            return 1.0
        f = (t - self.spawn_time) / self.fade_in_window
        return 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)

    def refresh_salience(self, t: float) -> float:
        self.sensory_salience = self.fade_in_factor(t) * self.scent(1.0)
        return self.sensory_salience


class World:
    """A square arena of side 2 * world_half holding up to n_patches patches.

    The patch count dips below n_patches while a removed patch waits to respawn.
    Call advance(t) each step to run the respawn timers.
    """

    def __init__(
        self,
        world_half: float = 700.0,
        rng: Optional[np.random.Generator] = None,
        mismatch_index: float = 0.0,
        n_patches: int = N_PATCHES,
        placement_margin: float = 100.0,       # keep patches this far from the walls
        min_patch_separation: float = 150.0,
        husk_threshold: float = 0.5,           # a patch left with less than this share
                                               # of its value is removed
        respawn_delay_mean: float = 500.0,
        fade_in_window: float = 200.0,
        value_cv: float = VALUE_CV,
        freshness_budget: float = FRESHNESS_BUDGET,
        obs_noise: float = 0.1,                # std of the noise on perceived value
        obs_noise_tau: float = 50.0,           # its correlation time
        obs_noise_dt: float = 10.0,
        noise_horizon: float = 80_000.0,       # length of the precomputed noise paths
        space_initial_patches: bool = True,    # False reproduces the poster (see below)
    ):
        self.world_half = float(world_half)
        self.mismatch_index = float(np.clip(mismatch_index, 0.0, 1.0))
        self.profile = resource_profile(self.mismatch_index)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_patches = int(n_patches)
        self.margin = float(placement_margin)
        self.min_sep = float(min_patch_separation)
        self.husk_threshold = float(husk_threshold)
        self.respawn_delay_mean = float(respawn_delay_mean)
        self.fade_in_window = float(fade_in_window)
        self.value_cv = float(value_cv)
        self.freshness_budget = float(freshness_budget)

        # Observation noise gets its own generator, seeded from the world's, so the
        # noise paths never shift the random numbers used to place patches.
        self.obs_noise = float(obs_noise)
        self.obs_noise_tau = float(obs_noise_tau)
        self.obs_noise_dt = float(obs_noise_dt)
        self.noise_horizon = float(noise_horizon)
        self._noise_paths = {}
        self._noise_rng = (np.random.default_rng(int(self.rng.integers(0, 2**31 - 1)))
                           if self.obs_noise > 0.0 else None)

        self._pending_respawns: List[float] = []
        self._next_patch_id = 0
        # The starting patches are there at full strength from t = 0.
        #
        # _sample_position keeps a new patch away from the ones already in
        # self.patches. The poster runs built all the starting patches in one go,
        # before any were in self.patches, so they were not spaced from each other
        # (respawns always were). space_initial_patches=False does the same; every
        # script here passes it, so the stored landscapes were computed that way.
        self.patches: List[Patch] = []
        if space_initial_patches:
            for _ in range(self.n_patches):
                self.patches.append(self._make_patch(t=0.0, fade_in=False))
        else:
            self.patches = [self._make_patch(t=0.0, fade_in=False)
                            for _ in range(self.n_patches)]

    # -- observation noise ---------------------------------------------------

    def obs_noise_at(self, patch: Patch, t: float) -> float:
        """Noise on the forager's reading of `patch`'s value at time t.

        Each patch gets one slowly varying (Ornstein-Uhlenbeck) path, so the
        forager has to average over time to judge a patch, which is the job of
        the slow beta signal."""
        if self.obs_noise <= 0.0:
            return 0.0
        path = self._noise_paths.get(patch.patch_id)
        if path is None:
            path = self._build_noise_path()
            self._noise_paths[patch.patch_id] = path
        idx = t / self.obs_noise_dt
        i0 = int(idx)
        if i0 < 0:
            return float(path[0])
        if i0 >= len(path) - 1:
            return float(path[-1])
        frac = idx - i0
        return float(path[i0] * (1.0 - frac) + path[i0 + 1] * frac)

    def _build_noise_path(self) -> np.ndarray:
        n = int(self.noise_horizon / self.obs_noise_dt) + 2
        z = self._noise_rng.standard_normal(n)
        if self.obs_noise_tau <= 0.0:
            return self.obs_noise * z
        a = math.exp(-self.obs_noise_dt / self.obs_noise_tau)
        s = self.obs_noise * math.sqrt(max(0.0, 1.0 - a * a))
        path = np.empty(n)
        x = 0.0
        for i in range(n):
            x = a * x + s * z[i]
            path[i] = x
        return path

    # -- spawning ------------------------------------------------------------

    def _sample_position(self) -> Tuple[float, float]:
        """Random spot away from the walls and from existing patches. If the arena
        is crowded the spacing rule is relaxed step by step."""
        lo = -self.world_half + self.margin
        hi = self.world_half - self.margin
        occupied = [(p.x, p.y) for p in getattr(self, 'patches', [])]

        for min_d in (self.min_sep, self.min_sep * 0.5, self.min_sep * 0.2, 0.0):
            min_d_sq = min_d * min_d
            for _ in range(250):
                x = float(self.rng.uniform(lo, hi))
                y = float(self.rng.uniform(lo, hi))
                if all((x - ox) ** 2 + (y - oy) ** 2 >= min_d_sq for ox, oy in occupied):
                    return x, y
        return float(self.rng.uniform(lo, hi)), float(self.rng.uniform(lo, hi))

    def _draw_value(self, mean_value: float) -> float:
        """Real value of a new patch, drawn from a lognormal around the profile mean.

        Patches differ in worth, but the scent only shows the *fraction* left
        (always 1 on a fresh patch), so the forager can't tell a rich patch from a
        poor one at a glance. It has to sample."""
        if self.value_cv <= 0.0 or mean_value <= 0.0:
            return float(mean_value)
        sigma_ln = math.sqrt(math.log(1.0 + self.value_cv ** 2))
        mu_ln = math.log(mean_value) - 0.5 * sigma_ln ** 2
        return float(self.rng.lognormal(mu_ln, sigma_ln))

    def _make_patch(self, t: float = 0.0, fade_in: bool = True) -> Patch:
        x, y = self._sample_position()
        value = self._draw_value(self.profile['max_utility'])
        patch = Patch(
            x=x, y=y, profile=self.profile,
            max_utility=value, current_utility=value,
            patch_id=self._next_patch_id, spawn_time=float(t),
            fade_in_window=self.fade_in_window if fade_in else 0.0,
            freshness_budget=self.freshness_budget,
        )
        patch.refresh_salience(float(t))
        self._next_patch_id += 1
        return patch

    def _despawn(self, patch: Patch, t: float) -> None:
        """Remove a patch and book its replacement after an exponential delay."""
        if patch in self.patches:
            self.patches.remove(patch)
        delay = float(self.rng.exponential(self.respawn_delay_mean))
        self._pending_respawns.append(float(t) + delay)

    def advance(self, t: float) -> None:
        """Move the world clock to t: spawn any patches that are due and update
        every patch's scent."""
        t = float(t)
        if self._pending_respawns:
            still_pending: List[float] = []
            for spawn_t in self._pending_respawns:
                if spawn_t <= t and len(self.patches) < self.n_patches:
                    self.patches.append(self._make_patch(t=t, fade_in=True))
                else:
                    still_pending.append(spawn_t)
            self._pending_respawns = still_pending
        for p in self.patches:
            p.refresh_salience(t)

    # -- called by the agent --------------------------------------------------

    def engage_patch(self, patch: Patch, amount: float, t: float) -> float:
        """The forager spiked on `patch`: harvest up to `amount` of real value.
        A patch is never removed here, so an empty modern patch can keep the
        forager hooked until its own beta signal pulls it away."""
        if patch not in self.patches:
            return 0.0
        harvested = patch.deplete(amount)
        patch.refresh_salience(t)
        return harvested

    def leave_patch(self, patch: Patch, t: float) -> bool:
        """The forager has moved on from `patch`. Remove the patch if it is mostly
        used up; leave it if it is still worth coming back to."""
        if patch not in self.patches:
            return False
        if patch.value_fraction() <= self.husk_threshold:
            self._despawn(patch, t)
            return True
        return False

    # -- plotting -------------------------------------------------------------

    def compute_grid(self, t: float, grid_size: int = 160) -> np.ndarray:
        """Total scent over the arena on a grid_size x grid_size grid."""
        xs = np.linspace(-self.world_half, self.world_half, grid_size)
        X, Y = np.meshgrid(xs, xs)
        total = np.zeros((grid_size, grid_size), dtype=float)
        for p in self.patches:
            if p.sensory_salience <= 1e-7:
                continue
            prof = p.profile
            R = np.sqrt((X - p.x) ** 2 + (Y - p.y) ** 2)
            base_env = prof['A'] * np.exp(-((R ** 2) / (2 * prof['sigma'] ** 2)) ** (prof['n'] / 2))
            ripple = 1.0 + prof['alpha'] * np.sin(prof['omega'] * t - prof['k'] * R)
            total += p.sensory_salience * base_env * ripple
        return total

    def __repr__(self) -> str:
        return (f"World(mismatch_index={self.mismatch_index:.2f}, "
                f"patches={len(self.patches)}/{self.n_patches})")
