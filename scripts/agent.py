"""
agent.py — Chemotaxis-based engagement model, wired to the Patch-depletion
world (see natural_world.py).

Replaces the binary engage/wander toggle with a biased random walk modulated
by the agent's internal reward expectation (E_R). This eliminates the border-freeze
artifact and keeps the agent trapped near a stimulus patch even after homeostatic
suppression stops active spiking — simulating the cognitive lag of compulsive
scrolling.

World-aware operation
---------------------
Pass a world (AncestralWorld instance) to the constructor and all world-specific
overrides are activated automatically — no subclass or factory needed.

  agent = Agent(params, world=world, rng=rng)

When world is provided:
  - _total_I_world reads each Patch's `sensory_salience` directly (the
    perceptual-pull number, NOT `current_utility`) — this is what feeds the
    neural input current I(r,t). In the ancestral regime a depleted patch's
    salience decays with it, so the neural drive fades and the agent leaves
    naturally. In the modern regime salience stays pinned near 1.0 even at
    zero utility, so the neural network keeps perceiving high reward
    availability from an empty patch — the supernormal-stimulus trap.
  - _dominant_source resolves the world's dominant Patch via
    `world.dominant_patch()` and wraps it as a perceptual source via
    `Patch.to_source()`, with a silent-source fallback when no patch is
    perceptible.
  - Every call to _dominant_source also tracks patch *occupancy*: the moment
    the agent's dominant patch changes, `world.notify_agent_exit()` fires on
    the patch just vacated, which is what allows a depleted patch to
    despawn and a replacement to spawn elsewhere.
  - On each spike (the discrete "engagement" event), the agent calls
    `world.engage_patch(patch, amount)` and accumulates the ACTUAL harvested
    utility this returns — a number that is completely decoupled from the
    perceived stimulus intensity that drove the spike. This is the real
    fitness yield; perceived salience and actual yield are tracked as two
    separate numbers throughout.
  - _biased_wander_step applies a global travel-friction penalty to
    wander_speed, scaled by `world.mismatch_index`: high friction (slow,
    costly travel between patches) in the ancestral regime, friction ->0
    (instantaneous, thumb-swipe patch switching) in the modern regime.

When world is None (legacy single-source mode):
  - stimuli must be supplied explicitly
  - all methods fall back to the original self.stimuli path
  - no patch tracking, harvesting, or friction scaling applies
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from .stimulus import StimulusSource
from .constants import (
    STEP_SIZE_DEFAULT, REFRACTORY_DROP_DEFAULT, MAX_SPIKES_DEFAULT,
    WANDER_DT_DEFAULT, WANDER_SPEED_DEFAULT, TURN_NOISE_DEFAULT,
    MIN_DT, STALL_WARN_TICKS,
)

if TYPE_CHECKING:
    from .natural_world import AncestralWorld, Patch


@dataclass
class AgentHistory:
    """Container for aligned simulation history arrays."""
    t:          np.ndarray
    S:          np.ndarray
    V:          np.ndarray
    E_R:        np.ndarray
    beta_PFC:   np.ndarray
    r:          np.ndarray      # distance to dominant source
    x:          np.ndarray
    y:          np.ndarray
    fire_times: np.ndarray
    I:          np.ndarray      # total summed stimulus intensity (perceived)
    eng_weight: np.ndarray      # engagement weight in [0,1] at each timestep
    harvested_utility: np.ndarray
    # ^ REAL fitness yield harvested per spike, aligned 1:1 with fire_times
    # (NOT with t). This is `world.engage_patch()`'s return value at each
    # spike — the actual utility drained from the dominant patch, clipped to
    # what remained. Deliberately decoupled from `I`/`eng_weight`, which
    # reflect perceived salience and can stay high in the modern regime even
    # while this collapses to zero. Legacy (non-world) agents get an
    # all-zero array of the same length as fire_times, since there is no
    # patch to harvest from.
    total_distance: float
    # ^ Total scalar path length travelled over the whole run — the sum of
    # per-iteration Euclidean displacements in (x, y), covering BOTH spike
    # steps (the discrete step_size hop toward the dominant source) and
    # wander steps (the biased-random-walk movement). A single number, not
    # a per-timestep array — useful as a cheap "how frantically did this
    # genome move" diagnostic alongside n_spikes / harvested_utility.


class Agent:
    def __init__(
        self,
        params: dict,
        stimuli: Optional[List[StimulusSource]] = None,
        *,
        world: Optional["AncestralWorld"] = None,
        genome: Optional[dict] = None,
        x0: float = -200.0,
        y0: float = -240.0,
        step_size: float = STEP_SIZE_DEFAULT,
        refractory_drop: float = REFRACTORY_DROP_DEFAULT,
        max_spikes: int = MAX_SPIKES_DEFAULT,
        wander_dt: float = WANDER_DT_DEFAULT,
        wander_speed: float = WANDER_SPEED_DEFAULT,
        turn_noise: float = TURN_NOISE_DEFAULT,
        E_R_engage_fraction: float = 0.5,
        E_R_scale: float = 3.0,
        gradient_bias_strength: float = np.pi / 3,
        min_speed_fraction: float = 0.25,
        world_half: Optional[float] = None,
        travel_friction_scale: float = 3.0,
        patch_harvest_amount: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        travel_friction_scale — maximum wander-speed penalty (subtracted
            from wander_speed while travelling) at mismatch_index=0.0
            (pure ancestral). Scales linearly to 0 penalty at
            mismatch_index=1.0 (pure modern). Only used in world mode.

        patch_harvest_amount — the amount of `current_utility` a single
            spike attempts to drain from the dominant patch via
            `world.engage_patch()`. The call is clipped internally to
            whatever utility actually remains, so this is a ceiling on
            per-spike yield, not a guaranteed amount. Only used in world
            mode.
        """
        # ── world vs stimuli ──────────────────────────────────────────────────
        # Exactly one of world or stimuli must be provided.
        if world is None and (stimuli is None or len(stimuli) == 0):
            raise ValueError(
                "Provide either world= (AncestralWorld) or a non-empty stimuli= list."
            )
        if world is not None and stimuli is not None:
            raise ValueError(
                "Provide world= or stimuli=, not both. "
                "In world mode, stimuli are managed by the world."
            )

        self.world = world
        self._current_patch: Optional["Patch"] = None  # world-mode occupancy tracker

        if world is not None:
            # World mode: seed self.stimuli with whatever is active at t=0 so
            # the non-None constraint elsewhere is satisfied. _total_I and
            # _dominant_source bypass self.stimuli entirely at runtime.
            from .natural_world import _silent_source
            initial = world.get_active_sources(0.0) or [_silent_source(world.world_half)]
            self.stimuli = initial
            # world_half comes from the world unless explicitly overridden
            if world_half is None:
                world_half = world.world_half
        else:
            self.stimuli = list(stimuli)

        # ── params + genome ───────────────────────────────────────────────────
        self.p = dict(params)
        if genome is not None:
            self.p.update(genome)

        for k in ['tau_pfc']:
            if k not in self.p:
                raise KeyError(f"Missing required param '{k}'.")

        if 'beta_pfc_max' not in self.p:
            if 'beta_pfc' in self.p:
                self.p['beta_pfc_max'] = self.p['beta_pfc']
            else:
                raise KeyError("Need 'beta_pfc_max' (or legacy 'beta_pfc') in params.")

        if refractory_drop <= 0:
            raise ValueError("refractory_drop must be > 0.")
        if self.p['theta'] - refractory_drop <= self.p['V_rest']:
            raise ValueError(
                f"theta - refractory_drop ({self.p['theta'] - refractory_drop:.2f}) "
                f"must stay above V_rest ({self.p['V_rest']}). "
                "Lower refractory_drop or raise theta."
            )

        # ── movement + integration params ─────────────────────────────────────
        self.x = float(x0)
        self.y = float(y0)
        self.step_size         = float(step_size)
        self.refractory_drop   = float(refractory_drop)
        self.max_spikes        = int(max_spikes)
        self.wander_dt         = float(wander_dt)
        self.wander_speed      = float(wander_speed)
        self.turn_noise        = float(turn_noise)
        self.E_R_engage_fraction = float(E_R_engage_fraction)
        self.E_R_scale         = float(E_R_scale)
        self.gradient_bias_strength = float(gradient_bias_strength)
        self.min_speed_fraction = float(min_speed_fraction)
        self.world_half        = float(world_half) if world_half is not None else None
        self.travel_friction_scale = float(travel_friction_scale)
        self.patch_harvest_amount  = float(patch_harvest_amount)
        self.rng               = rng if rng is not None else np.random.default_rng()
        self.heading           = float(self.rng.uniform(0, 2 * np.pi))

    # ── neural threshold ──────────────────────────────────────────────────────

    @property
    def I_fire(self) -> float:
        """Minimum stimulus intensity required to reach firing threshold from rest."""
        p = self.p
        return p['theta'] - p['V_rest'] + p['beta_pfc_max'] - p['beta_bg']

    # ── engagement ────────────────────────────────────────────────────────────

    @property
    def _E_R_engage_threshold(self) -> float:
        """E_R inflection point where engagement weight equals 0.5."""
        return self.E_R_engage_fraction * self.I_fire

    def _engagement_weight(self, E_R: float) -> float:
        """
        Maps E_R to a smooth [0,1] engagement weight via a sigmoid.

        Uses E_R (lagging reward expectation) rather than raw I so that
        engagement remains high — trapping the agent — even if homeostatic
        suppression has completely killed active spiking.
        """
        z = (E_R - self._E_R_engage_threshold) / self.E_R_scale
        return float(1.0 / (1.0 + np.exp(-z)))

    # ── stimulus ──────────────────────────────────────────────────────────────

    def _total_I(self, t: float) -> float:
        """
        Total stimulus intensity at current position.

        World mode: fast hot path — reads sensory_salience directly off each
        Patch, zero StimulusSource object construction.

        Legacy mode: sums eval_at() across self.stimuli.
        """
        if self.world is not None:
            return self._total_I_world(t)
        return float(sum(s.eval_at(self.x, self.y, t) for s in self.stimuli))

    def _total_I_world(self, t: float) -> float:
        """
        Hot path for world mode. Called only from _total_I.

        Sums perceptual intensity across all patches using each patch's
        `sensory_salience` — the perceptual-pull number — NOT
        `current_utility`. This is deliberate: in the modern regime a patch
        can sit at current_utility=0 while sensory_salience stays pinned
        near its salience_floor ceiling, keeping the neural input current
        I(r,t) — and therefore this function's output — high even though
        there is nothing left to actually gain. Mirrors the maths in
        `Patch.to_source().eval_at()` without constructing intermediate
        StimulusSource objects.
        """
        total = 0.0
        ax, ay = self.x, self.y
        for patch in self.world.patches:
            salience = patch.sensory_salience
            if salience <= 1e-7:
                continue
            prof = patch.profile
            r = np.sqrt((ax - patch.x) ** 2 + (ay - patch.y) ** 2)
            env    = prof['A'] * np.exp(-((r ** 2) / (2.0 * prof['sigma'] ** 2)) ** (prof['n'] / 2.0))
            ripple = 1.0 + prof['alpha'] * np.sin(prof['omega'] * t - prof['k'] * r)
            total += salience * env * ripple
        return float(total)

    def _dominant_source(self, t: float) -> StimulusSource:
        """
        Returns the stimulus source with the highest local intensity.

        World mode: resolves `world.dominant_patch()` and wraps it via
        `Patch.to_source()`; falls back to the silent dummy source if no
        patch is currently perceptible (all salience ~0). Every call also
        updates the agent's tracked patch occupancy (see
        `_update_current_patch`) — this is the single choke point through
        which `world.notify_agent_exit()` gets fired, so callers don't need
        to manage patch transitions themselves.

        Legacy mode: argmax over self.stimuli.
        """
        if self.world is not None:
            from .natural_world import _silent_source
            patch = self.world.dominant_patch(self.x, self.y, t)
            self._update_current_patch(patch)
            if patch is None:
                return _silent_source(self.world.world_half)
            src = patch.to_source(t)
            if src is None:
                # Salience collapsed to ~0 between selection and wrapping —
                # shouldn't normally happen since dominant_patch only
                # returns patches with positive salience, but guard anyway.
                return _silent_source(self.world.world_half)
            return src
        return max(self.stimuli, key=lambda s: s.eval_at(self.x, self.y, t))

    def _update_current_patch(self, patch: Optional["Patch"]) -> None:
        """
        Track which patch the agent is currently dominated by. The instant
        the dominant patch changes — including transitioning to/from
        None — notify the world that the agent has left the *previous*
        patch. This is the exit hook that lets a depleted patch actually
        despawn and a fresh one spawn elsewhere; until this fires, a spent
        patch just sits there (still glowing, in the modern regime).
        """
        if patch is self._current_patch:
            return
        if self._current_patch is not None:
            self.world.notify_agent_exit(self._current_patch)
        self._current_patch = patch

    def _harvest_patch(self, patch: Optional["Patch"]) -> float:
        """
        Drive one discrete engagement/harvest event against `patch` (called
        on spike). Returns the ACTUAL utility harvested — clipped to
        whatever remained — via `world.engage_patch()`. This is completely
        separate from perceived stimulus intensity: a modern-regime patch
        can keep firing spikes (high perceived I) while this returns 0.0
        because current_utility is already exhausted.
        """
        if patch is None or self.world is None:
            return 0.0
        return float(self.world.engage_patch(patch, self.patch_harvest_amount))

    # ── movement ──────────────────────────────────────────────────────────────

    def _gradient_direction_to(self, sx: float, sy: float) -> np.ndarray:
        """Unit vector from current position toward (sx, sy). Zero if already there."""
        dx, dy = sx - self.x, sy - self.y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist < 1e-9:
            return np.array([0.0, 0.0])
        return np.array([dx / dist, dy / dist])

    def _biased_wander_step(self, E_R: float, t: float) -> None:
        """
        Chemotactic step blending random walk and directional gradient pull.

        As engagement weight w increases:
          - Turning noise drops to zero.
          - Speed scales down to min_speed_fraction floor.
          - Heading is pulled toward the dominant source.

        World mode adds two behaviours:

          Silent suppression: when no patch is currently perceptible the
          gradient pull is suppressed (E_R passed as 0) to prevent spurious
          attraction toward the silent dummy source at the world origin.

          Travel friction: wander_speed is reduced by a penalty that scales
          with `world.mismatch_index` —
              penalty = (1 - mismatch_index) * travel_friction_scale
          modelling the real travel/exploration cost of the ancestral world.
          As mismatch_index -> 1 (modern) the penalty vanishes entirely,
          so the same genome can switch patches instantly and frictionlessly
          — a thumb swipe. Speed is clamped to 0 minimum; the original value
          is always restored in a finally block.
        """
        if self.world is not None:
            self._biased_wander_step_world(E_R, t)
        else:
            self._biased_wander_step_core(E_R, t)

    def _biased_wander_step_world(self, E_R: float, t: float) -> None:
        """World-mode wander step. Called only from _biased_wander_step."""
        dom = self._dominant_source(t)
        patch = getattr(dom, '_patch', None)

        if patch is None:
            # No perceptible patch: suppress dummy-source gradient pull
            self._biased_wander_step_core(0.0, t)
            return

        mismatch_index = self.world.mismatch_index
        friction = (1.0 - mismatch_index) * self.travel_friction_scale
        original_speed = self.wander_speed
        self.wander_speed = max(0.0, original_speed - friction)
        try:
            self._biased_wander_step_core(E_R, t)
        finally:
            self.wander_speed = original_speed

    def _biased_wander_step_core(self, E_R: float, t: float) -> None:
        """
        Pure movement logic — no world awareness. Shared by both modes.
        All world-specific logic lives in _biased_wander_step_world.
        """
        w = self._engagement_weight(E_R)

        # Random walk component
        eff_noise = self.turn_noise * (1.0 - w)
        self.heading += self.rng.normal(0.0, max(eff_noise, 1e-6))

        eff_speed = self.wander_speed * (
            self.min_speed_fraction + (1.0 - self.min_speed_fraction) * (1.0 - w)
        )
        rand_dx = eff_speed * np.cos(self.heading)
        rand_dy = eff_speed * np.sin(self.heading)

        # Gradient component
        grad_dx, grad_dy = 0.0, 0.0
        if w > 0.01:
            dom = self._dominant_source(t)
            gdir = self._gradient_direction_to(dom.x, dom.y)
            grad_scale = w * self.wander_speed * (1.0 - self.min_speed_fraction)
            grad_dx = grad_scale * gdir[0]
            grad_dy = grad_scale * gdir[1]

        new_x = self.x + rand_dx + grad_dx
        new_y = self.y + rand_dy + grad_dy

        # Boundary reflection
        if self.world_half is not None:
            wh = self.world_half
            if new_x > wh or new_x < -wh:
                self.heading = np.pi - self.heading
                new_x = np.clip(new_x, -wh, wh)
            if new_y > wh or new_y < -wh:
                self.heading = -self.heading
                new_y = np.clip(new_y, -wh, wh)

        self.x = new_x
        self.y = new_y

    # ── ODE system ────────────────────────────────────────────────────────────

    def _odes(self, t: float, y: np.ndarray) -> list:
        """4-state ODE: [S, V, E_R, beta_PFC]."""
        p = self.p
        S, V, E_R, beta_PFC = y
        I = self._total_I(t)

        relu = lambda v: np.maximum(0.0, v)

        dSdt = (1 / p['tau_s']) * (
            p['gamma_s'] * (p['S_rest'] - S) - p['alpha_s'] * relu(I - E_R)
        )
        if S <= 0.0:
            dSdt = max(0.0, dSdt)

        dVdt = (1 / p['tau_v']) * (
            -(V - p['V_rest']) + (I * S) - beta_PFC + p['beta_bg']
        )
        dE_Rdt  = (1 / p['tau_r']) * (I - E_R)
        dbetadt = (1 / p['tau_pfc']) * (p['gamma_pfc'] * (p['beta_pfc_max'] - beta_PFC))

        return [dSdt, dVdt, dE_Rdt, dbetadt]

    # ── main loop ─────────────────────────────────────────────────────────────

    def simulate(self, t_span: tuple, rtol: float = 1e-6, atol: float = 1e-8) -> AgentHistory:
        """
        Runs the core integration loop, halting at upward threshold crossings.

        Spikes trigger:
          - a discrete engagement/harvest event against the dominant patch
            (world mode): world.engage_patch() drains current_utility and
            returns the actual amount harvested, accumulated separately from
            perceived intensity.
          - a discrete step toward the dominant source
          - per-spike PFC fatigue decrement
          - refractory voltage reset

        Between spikes the agent calls _biased_wander_step for chemotactic
        movement, subject to world-mismatch-scaled travel friction.

        Patch occupancy (and therefore despawn/respawn via
        world.notify_agent_exit) is tracked transparently every time
        _dominant_source() is called — no separate bookkeeping needed here.
        """
        t_start, t_end = t_span
        p = self.p

        def spike_event(t, y):
            return y[1] - p['theta']

        spike_event.terminal  = True
        spike_event.direction = 1

        state = np.array([p['S_rest'], p['V_rest'], 0.0, p['beta_pfc_max']], dtype=float)
        t_current = t_start

        total_distance = 0.0

        times, S_h, V_h, E_h, B_h = [], [], [], [], []
        x_h, y_h, r_h, I_h, ew_h = [], [], [], [], []
        fire_times = []
        harvested_h = []

        n_spikes   = 0
        slow_ticks = 0

        while t_current < t_end:
            old_x, old_y = self.x, self.y  # position before this iteration's movement
            segment_end = min(t_current + self.wander_dt, t_end)

            sol = solve_ivp(
                fun=self._odes,
                t_span=(t_current, segment_end),
                y0=state,
                method='RK45',
                events=spike_event,
                rtol=rtol,
                atol=atol,
                max_step=np.inf,
            )

            if not sol.success:
                raise RuntimeError(f"Integration failed at t={t_current}: {sol.message}")

            start_idx = 1 if times else 0
            dom = self._dominant_source(sol.t[-1])
            for j in range(start_idx, len(sol.t)):
                E_R_j = sol.y[2, j]
                times.append(sol.t[j])
                S_h.append(sol.y[0, j]); V_h.append(sol.y[1, j])
                E_h.append(E_R_j);       B_h.append(sol.y[3, j])
                x_h.append(self.x);      y_h.append(self.y)
                r_h.append(dom.r_from(self.x, self.y))
                I_h.append(self._total_I(sol.t[j]))
                ew_h.append(self._engagement_weight(E_R_j))

            state     = sol.y[:, -1].copy()
            t_new     = sol.t[-1]
            spiked    = sol.status == 1
            E_R_now   = state[2]
            w_now     = self._engagement_weight(E_R_now)

            if spiked:
                n_spikes += 1
                fire_times.append(t_new)
                slow_ticks = 0

                dom = self._dominant_source(t_new)
                patch = getattr(dom, '_patch', None)
                harvested_h.append(self._harvest_patch(patch))

                direction = self._gradient_direction_to(dom.x, dom.y)
                self.x += self.step_size * direction[0]
                self.y += self.step_size * direction[1]

                state[3] = max(0.0, state[3] - p['alpha_pfc'])
                state[1] = p['theta'] - self.refractory_drop

                if n_spikes >= self.max_spikes:
                    break

            else:
                self._biased_wander_step(E_R_now, t_new)

                if w_now > 0.7:
                    slow_ticks += 1
                    if slow_ticks == STALL_WARN_TICKS:
                        import warnings
                        S_now, beta_PFC_now = state[0], state[3]
                        suppressed = (
                            S_now < 0.5 * p['S_rest']
                            or beta_PFC_now < 0.2 * p['beta_pfc_max']
                        )
                        if suppressed:
                            warnings.warn(
                                f"Agent gravitationally trapped (w={w_now:.2f}) but not "
                                f"spiking at t={t_new:.1f} — homeostatic suppression: "
                                f"S={S_now:.2f}, beta_PFC={beta_PFC_now:.3f}. "
                                f"E_R still elevated ({E_R_now:.2f}), escape is slow — "
                                "intended mismatch dynamic.",
                                RuntimeWarning,
                            )
                        else:
                            warnings.warn(
                                f"Agent gravitationally trapped (w={w_now:.2f}) but not "
                                f"spiking at t={t_new:.1f}, S/beta_PFC near resting. "
                                "Possible parameter mismatch — check I_fire vs source amplitudes.",
                                RuntimeWarning,
                            )
                else:
                    slow_ticks = 0

            # Physical distance covered this iteration — whichever branch ran
            # above (spike hop or wander step) already updated self.x/self.y,
            # so this captures both movement types uniformly. Accumulated
            # before the break check below so a run that halts mid-iteration
            # (MIN_DT stall) still counts the movement that already happened.
            total_distance += float(np.linalg.norm([self.x - old_x, self.y - old_y]))

            if t_new - t_current < MIN_DT and not spiked:
                break

            t_current = t_new

        return AgentHistory(
            t=np.array(times),       S=np.array(S_h),       V=np.array(V_h),
            E_R=np.array(E_h),       beta_PFC=np.array(B_h),
            r=np.array(r_h),         x=np.array(x_h),        y=np.array(y_h),
            fire_times=np.array(fire_times),
            I=np.array(I_h),         eng_weight=np.array(ew_h),
            harvested_utility=np.array(harvested_h),
            total_distance=total_distance
        )