"""
The agent integrates a 4-state ODE driven by the *sum* of all active
stimulus sources.  Engagement is checked per-source (each source has its
own analytically-derived threshold).  When a spike fires, the agent steps
toward whichever source is currently dominant (highest I contribution).

Key design choices
------------------
- stimuli : List[StimulusSource]
    Any number of sources.  Single-source runs work exactly as before,
    just pass a one-element list.  `model_dynamics.default_source()` is
    the quick way to build the default source.

- engage_threshold is no longer a constructor argument.
    It is derived per-source from each source's own A/sigma/alpha and the
    agent's neural I_fire threshold.  See StimulusSource.engage_threshold().

- history.r
    Stores distance to the *dominant* source (highest I) at each timepoint.
    For single-source runs this is identical to the previous behaviour.
    For multi-source runs it tracks whichever source the agent is most
    attracted to at that moment.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import List, Optional
from functools import partial

from .stimulus import StimulusSource
from .constants import (
    STEP_SIZE_DEFAULT, REFRACTORY_DROP_DEFAULT, MAX_SPIKES_DEFAULT,
    WANDER_DT_DEFAULT, WANDER_SPEED_DEFAULT, TURN_NOISE_DEFAULT,
    MIN_DT, STALL_WARN_TICKS,
)


@dataclass
class AgentHistory:
    """Simulation output arrays, all aligned to the same time axis."""
    t:          np.ndarray
    S:          np.ndarray
    V:          np.ndarray
    E_R:        np.ndarray
    beta_PFC:   np.ndarray
    r:          np.ndarray   # distance to dominant source at each timestep
    x:          np.ndarray
    y:          np.ndarray
    fire_times: np.ndarray
    I:          np.ndarray   # total summed stimulus intensity


class Agent:
    """

    Parameters
    ----------
    params : dict
        Neural parameters (from model_dynamics.params).
    stimuli : List[StimulusSource]
        All stimulus sources in the world.  Must be non-empty.
    genome : dict, optional
        Overrides for evolvable params (S_rest, alpha_s, tau_r, alpha_pfc).
    x0, y0 : float
        Starting position.
    step_size : float
        Distance moved toward dominant source on each spike.
    refractory_drop : float
        Post-spike V reset = theta - refractory_drop.
        Must keep reset above V_rest (enforced in __init__).
    max_spikes : int
        Safety cap; simulation ends if hit.
    wander_dt : float
        Max ODE segment length.  Also the cadence at which engagement is
        checked and foraging steps are taken.  Should be well below a
        typical ISI so wandering doesn't blur out fast spiking.
    wander_speed : float
        Spatial units moved per foraging step.
    turn_noise : float
        Std-dev of heading change per step (radians).  Smaller = smoother.
    world_half : float, optional
        World boundary.  Agent reflects off edges if set.
    rng : np.random.Generator, optional
        Pass a seeded generator for reproducible runs.
    """

    def __init__(
        self,
        params: dict,
        stimuli: List[StimulusSource],
        *,
        genome: Optional[dict] = None,
        x0: float = -200.0,
        y0: float = -240.0,
        step_size: float = STEP_SIZE_DEFAULT,
        refractory_drop: float = REFRACTORY_DROP_DEFAULT,
        max_spikes: int = MAX_SPIKES_DEFAULT,
        wander_dt: float = WANDER_DT_DEFAULT,
        wander_speed: float = WANDER_SPEED_DEFAULT,
        turn_noise: float = TURN_NOISE_DEFAULT,
        world_half: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        if not stimuli:
            raise ValueError("stimuli must be a non-empty list of StimulusSource objects.")

        self.p = dict(params)
        if genome is not None:
            self.p.update(genome)

        required = ['tau_pfc']
        missing = [k for k in required if k not in self.p]
        if missing:
            raise KeyError(f"Missing required params {missing}.")

        if 'beta_pfc_max' not in self.p:
            if 'beta_pfc' in self.p:
                self.p['beta_pfc_max'] = self.p['beta_pfc']
            else:
                raise KeyError("Need 'beta_pfc_max' (or legacy 'beta_pfc') in params.")

        if refractory_drop <= 0:
            raise ValueError("refractory_drop must be > 0.")
        if self.p['theta'] - refractory_drop <= self.p['V_rest']:
            raise ValueError(
                f"theta - refractory_drop ({self.p['theta'] - refractory_drop:.2f}) must stay "
                f"above V_rest ({self.p['V_rest']}). Lower refractory_drop or raise theta."
            )

        self.stimuli = list(stimuli)
        self.x = float(x0)
        self.y = float(y0)
        self.step_size = float(step_size)
        self.refractory_drop = float(refractory_drop)
        self.max_spikes = int(max_spikes)
        self.wander_dt = float(wander_dt)
        self.wander_speed = float(wander_speed)
        self.turn_noise = float(turn_noise)
        self.world_half = float(world_half) if world_half is not None else None
        self.rng = rng if rng is not None else np.random.default_rng()
        self.heading = float(self.rng.uniform(0, 2 * np.pi))

    # ------------------------------------------------------------------
    # Neural firing threshold in stimulus units
    # ------------------------------------------------------------------

    @property
    def I_fire(self) -> float:
        """
        Stimulus intensity needed to drive V_eq to theta at resting state
        (S=S_rest=1, beta_PFC=beta_pfc_max).

            V_eq = V_rest + I·S - beta_PFC + beta_bg = theta
            => I_fire = theta - V_rest + beta_pfc_max - beta_bg

        At defaults: -55 - (-65) + 0.5 - 3.0 = 7.5.

        This is what StimulusSource.catchment_radius() and
        StimulusSource.engage_threshold() need — the link between
        neural params and stimulus geometry.
        """
        p = self.p
        return p['theta'] - p['V_rest'] + p['beta_pfc_max'] - p['beta_bg']

    # ------------------------------------------------------------------
    # Stimulus helpers
    # ------------------------------------------------------------------

    def _total_I(self, t: float) -> float:
        """Sum of I contributions from all sources at current position and time t."""
        return float(sum(s.eval_at(self.x, self.y, t) for s in self.stimuli))

    def _dominant_source(self, t: float) -> StimulusSource:
        """Source providing the highest I at current position and time t."""
        return max(self.stimuli, key=lambda s: s.eval_at(self.x, self.y, t))

    def _is_engaged(self, t: float) -> bool:
        """
        True if ANY source's I exceeds that source's own engage_threshold.
        Each threshold is derived from the source's shape params + self.I_fire,
        so it stays valid across any A/sigma combination.
        """
        I_fire = self.I_fire
        return any(
            s.eval_at(self.x, self.y, t) >= s.engage_threshold(I_fire)
            for s in self.stimuli
        )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _gradient_direction_to(self, sx: float, sy: float) -> np.ndarray:
        """Unit vector from current position toward (sx, sy). Zero if already there."""
        dx, dy = sx - self.x, sy - self.y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist < 1e-9:
            return np.array([0.0, 0.0])
        return np.array([dx / dist, dy / dist])

    def _wander_step(self) -> None:
        self.heading += self.rng.normal(0.0, self.turn_noise)
        new_x = self.x + self.wander_speed * np.cos(self.heading)
        new_y = self.y + self.wander_speed * np.sin(self.heading)

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

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------

    def _odes(self, t: float, y: np.ndarray) -> list:
        """
        4-state system: S, V, E_R, beta_PFC.
        Drive term uses total I (sum of all sources) at current agent position.
        Agent position is fixed for the duration of a segment, so this is
        correct — position only changes at spike events or wander steps,
        both of which trigger a new segment with a fresh solve_ivp call.
        """
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
        dE_Rdt = (1 / p['tau_r']) * (I - E_R)

        # Recovery only — fatigue is applied per-spike in simulate() (see note there).
        dbeta_PFCdt = (1 / p['tau_pfc']) * (
            p['gamma_pfc'] * (p['beta_pfc_max'] - beta_PFC)
        )

        return [dSdt, dVdt, dE_Rdt, dbeta_PFCdt]

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def simulate(self, t_span: tuple, rtol: float = 1e-6, atol: float = 1e-8) -> AgentHistory:
        """
        Integrate continuously; halt at each upward V-crossing of theta.

        Movement modes (mutually exclusive per tick):
          - SPIKE: step toward the dominant source (highest current I),
            apply PFC fatigue, reset V.
          - WANDER: correlated random walk when not engaged with any source.

        Spike event note: solve_ivp reads .terminal/.direction off the
        exact callable object passed to events=.  A new event function is
        built each segment (because _odes no longer takes r as a parameter,
        this is now simpler — one event function, no partial binding needed).
        Attributes are set directly on the function object per loop iteration
        to avoid the flat-trajectory bug documented in the original agent.py.
        """
        t_start, t_end = t_span
        p = self.p

        def spike_event(t, y):
            return y[1] - p['theta']

        spike_event.terminal = True
        spike_event.direction = 1  # rising crossings only

        state = np.array([p['S_rest'], p['V_rest'], 0.0, p['beta_pfc_max']], dtype=float)
        t_current = t_start

        times, S_h, V_h, E_h, B_h = [], [], [], [], []
        x_h, y_h, r_h, I_h = [], [], [], []
        fire_times = []

        n_spikes = 0
        engaged_no_spike_ticks = 0

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

            # Record solver-chosen points; skip first on later segments to
            # avoid duplicating the previous segment's final point.
            start_idx = 1 if times else 0
            dom = self._dominant_source(sol.t[-1])  # dominant source for r recording
            for j in range(start_idx, len(sol.t)):
                times.append(sol.t[j])
                S_h.append(sol.y[0, j]); V_h.append(sol.y[1, j])
                E_h.append(sol.y[2, j]); B_h.append(sol.y[3, j])
                x_h.append(self.x); y_h.append(self.y)
                r_h.append(dom.r_from(self.x, self.y))
                I_h.append(self._total_I(sol.t[j]))

            state = sol.y[:, -1].copy()
            t_new = sol.t[-1]
            spiked = sol.status == 1

            if spiked:
                n_spikes += 1
                fire_times.append(t_new)
                engaged_no_spike_ticks = 0

                # Move toward whichever source is currently dominant.
                dom = self._dominant_source(t_new)
                direction = self._gradient_direction_to(dom.x, dom.y)
                self.x += self.step_size * direction[0]
                self.y += self.step_size * direction[1]

                # PFC fatigue per spike (not via ODE — see model comment in
                # original agent.py, section 4 of CONTEXT_HANDOFF.txt).
                state[3] = max(0.0, state[3] - p['alpha_pfc'])

                # Refractory reset: land between V_rest and theta.
                state[1] = p['theta'] - self.refractory_drop

                if n_spikes >= self.max_spikes:
                    break

            else:
                # No spike this segment. Wander if not engaged with any source.
                if not self._is_engaged(t_new):
                    self._wander_step()
                    engaged_no_spike_ticks = 0
                else:
                    engaged_no_spike_ticks += 1
                    if engaged_no_spike_ticks == STALL_WARN_TICKS:
                        import warnings
                        S_now, beta_PFC_now = state[0], state[3]
                        suppressed = (
                            S_now < 0.5 * p['S_rest']
                            or beta_PFC_now < 0.2 * p['beta_pfc_max']
                        )
                        if suppressed:
                            warnings.warn(
                                f"Agent engaged but not spiking for {STALL_WARN_TICKS} ticks "
                                f"at t={t_new:.1f} — looks like homeostatic suppression: "
                                f"S={S_now:.2f} (rest {p['S_rest']}), "
                                f"beta_PFC={beta_PFC_now:.3f} (ceiling {p['beta_pfc_max']}). "
                                f"Sustained firing has driven sensitivity/PFC brake down far enough "
                                f"that V_eq dropped back below theta — mismatch dynamic as intended.",
                                RuntimeWarning,
                            )
                        else:
                            warnings.warn(
                                f"Agent engaged but not spiking for {STALL_WARN_TICKS} ticks "
                                f"at t={t_new:.1f}, with S/beta_PFC near resting values "
                                f"(S={S_now:.2f}, beta_PFC={beta_PFC_now:.3f}). "
                                f"Possible dead zone between engage_threshold and true catchment "
                                f"radius — check I_fire vs source amplitudes, or audit "
                                f"StimulusSource.engage_threshold() for these params.",
                                RuntimeWarning,
                            )

            if t_new - t_current < MIN_DT and not spiked:
                break

            t_current = t_new

        return AgentHistory(
            t=np.array(times), S=np.array(S_h), V=np.array(V_h),
            E_R=np.array(E_h), beta_PFC=np.array(B_h),
            r=np.array(r_h), x=np.array(x_h), y=np.array(y_h),
            fire_times=np.array(fire_times), I=np.array(I_h),
        )