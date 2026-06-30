"""
agent.py — Chemotaxis-based engagement model.

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
  - _total_I       uses the fast searchsorted hot path (zero object construction)
  - _dominant_source reads world.get_active_sources() with silent-source fallback
  - _biased_wander_step applies exploration cost friction (sampling_cost ×
    world.novel_cost_scale) when a novel-category patch is dominant, and
    suppresses gradient pull during silent periods.

When world is None (legacy single-source mode):
  - stimuli must be supplied explicitly
  - all three methods fall back to the original self.stimuli path
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
    from .natural_world import AncestralWorld


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
    I:          np.ndarray      # total summed stimulus intensity
    eng_weight: np.ndarray      # engagement weight in [0,1] at each timestep


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
        rng: Optional[np.random.Generator] = None,
    ):
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

        World mode: fast searchsorted path — zero Python object construction,
        reads event list and RESOURCE_TYPES directly.

        Legacy mode: sums eval_at() across self.stimuli.
        """
        if self.world is not None:
            return self._total_I_world(t)
        return float(sum(s.eval_at(self.x, self.y, t) for s in self.stimuli))

    def _total_I_world(self, t: float) -> float:
        """Hot path for world mode. Called only from _total_I."""
        from .natural_world import RESOURCE_TYPES, _MAX_EVENT_LIFESPAN
        t_starts = self.world._t_starts
        if len(t_starts) == 0:
            return 0.0
        idx_lo = int(np.searchsorted(t_starts, t - _MAX_EVENT_LIFESPAN, side='left'))
        idx_hi = int(np.searchsorted(t_starts, t, side='right'))
        total = 0.0
        ax, ay = self.x, self.y
        for i in range(idx_lo, idx_hi):
            e = self.world._events[i]
            if not e.is_visible(t):
                continue
            f = e.amplitude(t)
            if f <= 0.0:
                continue
            p = RESOURCE_TYPES[e.resource_type]
            r = np.sqrt((ax - e.x) ** 2 + (ay - e.y) ** 2)
            env    = f * p['A'] * np.exp(-((r ** 2) / (2.0 * p['sigma'] ** 2)) ** (p['n'] / 2.0))
            ripple = 1.0 + p['alpha'] * np.sin(p['omega'] * t - p['k'] * r)
            total += env * ripple
        return float(total)

    def _dominant_source(self, t: float) -> StimulusSource:
        """
        Returns the stimulus source with the highest local intensity.

        World mode: builds transient FadingStimulusSource objects from active
        events; falls back to the silent dummy if the world is empty.

        Legacy mode: argmax over self.stimuli.
        """
        if self.world is not None:
            from .natural_world import _silent_source
            sources = self.world.get_active_sources(t)
            if not sources:
                return _silent_source(self.world.world_half)
            return max(sources, key=lambda s: s.eval_at(self.x, self.y, t))
        return max(self.stimuli, key=lambda s: s.eval_at(self.x, self.y, t))

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

          Silent suppression: when no events are active the gradient pull is
          suppressed (E_R passed as 0) to prevent spurious attraction toward
          the silent dummy source at the world origin.

          Exploration cost friction: when the dominant source is a novel-category
          patch, wander_speed is reduced by
              penalty = sampling_cost × world.novel_cost_scale
          modelling the physical cost of novelty-seeking in the ancestral world.
          When novel_cost_scale → 0 (modern world) the penalty vanishes and the
          same genome explores compulsively. Speed is clamped to 0 minimum;
          the original value is always restored in a finally block.
        """
        if self.world is not None:
            self._biased_wander_step_world(E_R, t)
        else:
            self._biased_wander_step_core(E_R, t)

    def _biased_wander_step_world(self, E_R: float, t: float) -> None:
        """World-mode wander step. Called only from _biased_wander_step."""
        from .natural_world import RESOURCE_TYPES, is_novel

        has_source = any(self.world.active_events(t))

        if not has_source:
            # Silent world: suppress dummy-source gradient pull
            self._biased_wander_step_core(0.0, t)
            return

        dom = self._dominant_source(t)
        dom_type = getattr(dom, '_resource_type', None)

        if dom_type is not None and is_novel(dom_type):
            cost_scale = getattr(self.world, 'novel_cost_scale', 1.0)
            penalty = RESOURCE_TYPES[dom_type]['sampling_cost'] * cost_scale
            original_speed = self.wander_speed
            self.wander_speed = max(0.0, original_speed - penalty)
            try:
                self._biased_wander_step_core(E_R, t)
            finally:
                self.wander_speed = original_speed
        else:
            # Essential patch or untagged source — no cost penalty
            self._biased_wander_step_core(E_R, t)

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
          - a discrete step toward the dominant source
          - per-spike PFC fatigue decrement
          - refractory voltage reset

        Between spikes the agent calls _biased_wander_step for chemotactic movement.
        """
        t_start, t_end = t_span
        p = self.p

        def spike_event(t, y):
            return y[1] - p['theta']

        spike_event.terminal  = True
        spike_event.direction = 1

        state = np.array([p['S_rest'], p['V_rest'], 0.0, p['beta_pfc_max']], dtype=float)
        t_current = t_start

        times, S_h, V_h, E_h, B_h = [], [], [], [], []
        x_h, y_h, r_h, I_h, ew_h = [], [], [], [], []
        fire_times = []

        n_spikes   = 0
        slow_ticks = 0

        while t_current < t_end:
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

            if t_new - t_current < MIN_DT and not spiked:
                break

            t_current = t_new

        return AgentHistory(
            t=np.array(times),       S=np.array(S_h),       V=np.array(V_h),
            E_R=np.array(E_h),       beta_PFC=np.array(B_h),
            r=np.array(r_h),         x=np.array(x_h),        y=np.array(y_h),
            fire_times=np.array(fire_times),
            I=np.array(I_h),         eng_weight=np.array(ew_h),
        )