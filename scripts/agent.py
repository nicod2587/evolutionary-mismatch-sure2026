"""
if not spiking, a random-walk happens drifting by a small Gaussian increment each tick (`turn_noise`)
if spiking I(r, t) > theta, (with scipi)
    - agent steps `step_size` toward the source.
    - V is reset to a value below theta (refractory reset)
    - Integration resumes from the spike time.
"""
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Optional
from functools import partial

from .constants import (
    STEP_SIZE_DEFAULT, REFRACTORY_DROP_DEFAULT, MAX_SPIKES_DEFAULT,
    WANDER_DT_DEFAULT, WANDER_SPEED_DEFAULT, TURN_NOISE_DEFAULT,
    ENGAGE_THRESHOLD_DEFAULT, MIN_DT, STALL_WARN_TICKS,
)


@dataclass
class AgentHistory:
#data from simulation of agent
    t:          np.ndarray
    S:          np.ndarray
    V:          np.ndarray
    E_R:        np.ndarray
    beta_PFC:   np.ndarray
    r:          np.ndarray
    x:          np.ndarray
    y:          np.ndarray
    fire_times: np.ndarray
    I:          np.ndarray


class Agent:
    """
    engage_threshold : float
        I(r, t) value above which the agent is considered "engaged"
        with a stimulus and wandering stops (spike-driven movement
        takes over). Two things have to both hold for this to work, and
        both were checked empirically rather than assumed:
          1. The threshold must correspond to an r value *inside* the
             firing catchment radius, not just outside it -- I(r) falls
             off with distance, so a threshold set right at the
             catchment boundary stops wandering exactly where V_eq is
             still below theta, and the agent can sit "engaged" but
             silent indefinitely (this happened during tuning).
          2. The stimulus ripple (1 + alpha*sin(...)) makes I(r, t)
             oscillate by roughly +-20% of its envelope value at any
             fixed r, not just decay smoothly with r -- so a threshold
             that looks fine against the time-averaged envelope can
             still flicker in and out of "engaged" every ripple cycle
             at some radii. Default 8.0 sits below the ripple's *trough*
             at r~55 (about 18 units inside the r~73 catchment radius at
             A=22/sigma=50), so the agent is unconditionally engaged
             once it reaches r~55, not just on favorable ripple phases.
             A genuine flicker band still exists further out (~r 60-90,
             where the ripple swings the threshold crossing back and
             forth) -- that's left as-is rather than chased to zero,
             since it reflects real fluctuating stimulus intensity
             rather than a bug; verified it doesn't cause pathological
             back-and-forth freezing, just normal foraging variability
             (the agent sometimes wanders away again before fully
             committing, which is realistic).
        If you change A, sigma, theta, beta_bg, V_rest, or the ripple
        parameters (alpha, omega, k), recompute both the catchment
        radius and the ripple's min/max envelope at your intended
        engagement radius before trusting this default.
    """

    def __init__(
        self,
        params: dict, #from model_dynamics.py
        genome: Optional[dict] = None, # overrides for evolvable traits
        x0: float = -200.0, #starting position, this is just for testing
        y0: float = -240.0,
        source_x: float = 0.0, #idk if this should be here, might change later when i do the poission random enviroement
        source_y: float = 0.0, #will need to figure out how the agent will know where the source is though
        step_size: float = STEP_SIZE_DEFAULT,
        refractory_drop: float = REFRACTORY_DROP_DEFAULT, #how far below theta to rest, biologically isnt this below resting?
        stimulus_func: Optional[Callable] = None, #dont really understand this yet  stimulus_func : callable, optional f(r, t, params) -> float. Falls back to the default Gaussian envelope x ripple if not provided.
        max_spikes: int = MAX_SPIKES_DEFAULT, #just saftey for the moment
        wander_dt: float = WANDER_DT_DEFAULT, #Tick length for random walk, should be below typical ISI (5-20 here) so wandering doesn't itself blur out fast spiking
        wander_speed: float = WANDER_SPEED_DEFAULT, #should be around world half/ sigma but ill put it 4 for the sim
        turn_noise: float = TURN_NOISE_DEFAULT, #smaller = smoother
        engage_threshold: float = ENGAGE_THRESHOLD_DEFAULT, #input value at which agent is engaged
        world_half: Optional[float] = None, #world size to know where edge is
        rng: Optional[np.random.Generator] = None, #if i want same run every time parse in input
    ):
        self.p = dict(params)
        if genome is not None:
            self.p.update(genome)

        #error handling, better to have than not
        required = ['tau_pfc']
        missing = [k for k in required if k not in self.p]
        if missing:
            raise KeyError(
                f"Missing required params {missing}. "
                f"Add them explicitly, e.g. params['tau_pfc'] = 50.0"
            )
        # beta_pfc_max: the ceiling the PFC brake recovers toward.
        # Accept either 'beta_pfc_max' or legacy 'pfc_base'/'beta_pfc'.
        if 'beta_pfc_max' not in self.p:
            if 'beta_pfc' in self.p:
                self.p['beta_pfc_max'] = self.p['beta_pfc']
            else:
                raise KeyError(
                    "Need one of 'beta_pfc_max', or 'beta_pfc' in params."
                )

        if refractory_drop <= 0:
            raise ValueError("refractory_drop must be > 0, or spikes cannot re-trigger.")
        if self.p['theta'] - refractory_drop <= self.p['V_rest']:
            raise ValueError(
                f"theta - refractory_drop ({self.p['theta'] - refractory_drop:.2f}) must stay "
                f"above V_rest ({self.p['V_rest']}), or the post-spike reset isn't a biophysically "
                f"valid hyperpolarization (reset should land between V_rest and theta, not below "
                f"V_rest). Lower refractory_drop or raise theta."
            )

        self.x = float(x0)
        self.y = float(y0)
        self.source_x = float(source_x)
        self.source_y = float(source_y)
        self.step_size = float(step_size)
        self.refractory_drop = float(refractory_drop)
        self.stimulus_func = stimulus_func
        self.max_spikes = int(max_spikes)

        # --- Wandering / foraging state ---
        self.wander_dt = float(wander_dt)
        self.wander_speed = float(wander_speed)
        self.turn_noise = float(turn_noise)
        self.engage_threshold = float(engage_threshold)
        self.world_half = float(world_half) if world_half is not None else None
        self.rng = rng if rng is not None else np.random.default_rng()
        # this keeps walk corrilated
        self.heading = float(self.rng.uniform(0, 2 * np.pi))

    def _r(self, x: float, y: float) -> float:
        return np.sqrt((x - self.source_x) ** 2 + (y - self.source_y) ** 2)

    def _gradient_direction(self, x: float, y: float) -> np.ndarray:
        """Unit vector from (x, y) toward the source. Zero if already there."""
        dx = self.source_x - x
        dy = self.source_y - y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist < 1e-9:
            return np.array([0.0, 0.0])
        return np.array([dx / dist, dy / dist])

    def _stimulus(self, r: float, t: float) -> float:
        if self.stimulus_func is not None:
            return float(self.stimulus_func(r, t, self.p))
        p = self.p
        envelope = p['A'] * np.exp(-((r ** 2) / (2 * p['sigma'] ** 2)) ** (p['n'] / 2))
        ripple = 1 + p['alpha'] * np.sin(p['omega'] * t - p['k'] * r)
        return float(envelope * ripple)

    def _wander_step(self) -> None:
        self.heading += self.rng.normal(0.0, self.turn_noise)
        new_x = self.x + self.wander_speed * np.cos(self.heading)
        new_y = self.y + self.wander_speed * np.sin(self.heading)

        if self.world_half is not None:
            wh = self.world_half
            if new_x > wh or new_x < -wh:
                self.heading = np.pi - self.heading  # mirror horizontal component
                new_x = np.clip(new_x, -wh, wh)
            if new_y > wh or new_y < -wh:
                self.heading = -self.heading  # mirror vertical component
                new_y = np.clip(new_y, -wh, wh)

        self.x = new_x
        self.y = new_y

    def _odes(self, t: float, y: np.ndarray, r: float) -> list:
        p = self.p
        S, V, E_R, beta_PFC = y
        I = self._stimulus(r, t)

        relu = lambda v: np.maximum(0.0, v)

        dSdt = (1 / p['tau_s']) * (
            p['gamma_s'] * (p['S_rest'] - S) - p['alpha_s'] * relu(I - E_R) #because relu(I - E_R) is always >= 0, this term can only downregulate S, never upregulate it. very interesting finding when s is at peak
        )
        if S <= 0.0:
            dSdt = max(0.0, dSdt)
        dVdt = (1 / p['tau_v']) * (
            -(V - p['V_rest']) + (I * S) - beta_PFC + p['beta_bg']
        )
        dE_Rdt = (1 / p['tau_r']) * (I - E_R)
        # NOTE on beta_PFC: the *recovery* half of this equation
        # (gamma_pfc * (beta_pfc_max - beta_PFC)) is integrated continuously
        # here, exactly as in your original model. The *fatigue* half is
        # NOT integrated here -- see simulate() for why and how it's applied.
        dbeta_PFCdt = (1 / p['tau_pfc']) * (
            p['gamma_pfc'] * (p['beta_pfc_max'] - beta_PFC)
        )

        return [dSdt, dVdt, dE_Rdt, dbeta_PFCdt]

    # ------------------------------------------------------------------
    def simulate(self, t_span: tuple, rtol: float = 1e-6, atol: float = 1e-8) -> AgentHistory:
        """
        Integrate continuously; halt exactly at each upward threshold
        crossing, apply the spike rule (move + refractory reset), resume.

        Foraging behavior (2026-06 addition): each integration segment is
        now capped at `wander_dt` (in addition to ending early on a spike
        event, as before). At the end of every segment -- spiked or not --
        if I(r, t) is below `engage_threshold`, the agent is considered
        "not engaged" and takes one correlated-random-walk step via
        `_wander_step()`. If I(r, t) is at or above `engage_threshold`,
        wandering is skipped for that tick: the agent is close enough to
        a stimulus that the existing event-driven spike-and-jump mechanic
        is left to do the work (V is already being pulled toward theta by
        real drive, so it will reach threshold on its own without needing
        a random nudge). This means movement always comes from exactly
        one source at a time -- the random walk OR the spike-driven jump,
        never both in the same tick -- so the two mechanisms can't fight
        each other or double-move the agent.
        """
        t_start, t_end = t_span
        p = self.p

        def spike_event(t, y, r):
            return y[1] - p['theta']
        # IMPORTANT: solve_ivp reads .terminal / .direction directly off the
        # *exact object* passed to `events=`. Neither a plain
        # `lambda t, y: spike_event(t, y, r_now)` nor a bare
        # `functools.partial(spike_event, r=r_now)` forwards attributes set
        # on the original function -- both create a new callable object
        # that solve_ivp sees as having no .terminal/.direction, silently
        # defaulting to non-terminal. That mismatch (events get detected
        # internally and appear in sol.t_events, but sol.status never
        # becomes 1) was the root cause of the flat trajectory bug: spikes
        # were being found but never acted on. The fix is to set the
        # attributes on the *bound* object actually passed in, each loop
        # iteration, since r_now changes every segment.

        state = np.array([p['S_rest'], p['V_rest'], 0.0, p['beta_pfc_max']], dtype=float)
        t_current = t_start

        times, S_h, V_h, E_h, B_h, x_h, y_h, r_h, I_h = [], [], [], [], [], [], [], [], []
        fire_times = []

        n_spikes = 0
        engaged_no_spike_ticks = 0  # diagnostic counter, see warning below
        # MIN_DT and STALL_WARN_TICKS now come from constants.py (imported above) --
        # same values (1e-9, 200) this method always used.

        while t_current < t_end:
            r_now = self._r(self.x, self.y)

            event_fn = partial(spike_event, r=r_now)
            event_fn.terminal = True
            event_fn.direction = 1  # only rising (V increasing through theta) crossings

            # Cap this segment at wander_dt (on top of the existing t_end
            # cap) so the loop gets a chance to check engagement and move
            # the agent even when no spike occurs in this window. Without
            # this cap, a non-engaged agent (I(r,t) always below theta,
            # never spiking) would integrate in one shot straight to
            # t_end and never wander at all.
            segment_end = min(t_current + self.wander_dt, t_end)

            sol = solve_ivp(
                fun=partial(self._odes, r=r_now),
                t_span=(t_current, segment_end),
                y0=state,
                method='RK45',
                events=event_fn,
                rtol=rtol,
                atol=atol,
                max_step=np.inf,
            )

            if not sol.success:
                raise RuntimeError(f"Integration failed at t={t_current}: {sol.message}")

            # Append every solver-chosen point in this segment (skip the
            # very first point on segments after the first, since it
            # duplicates the previous segment's final point).
            start_idx = 1 if times else 0
            for j in range(start_idx, len(sol.t)):
                times.append(sol.t[j])
                S_h.append(sol.y[0, j]); V_h.append(sol.y[1, j])
                E_h.append(sol.y[2, j]); B_h.append(sol.y[3, j])
                x_h.append(self.x); y_h.append(self.y); r_h.append(r_now)
                I_h.append(self._stimulus(r_now, sol.t[j]))

            state = sol.y[:, -1].copy()
            t_new = sol.t[-1]

            spiked = sol.status == 1  # terminated by event, not by reaching t_end

            if spiked:
                n_spikes += 1
                fire_times.append(t_new)
                engaged_no_spike_ticks = 0

                # Move one step toward the source
                dir_vec = self._gradient_direction(self.x, self.y)
                self.x += self.step_size * dir_vec[0]
                self.y += self.step_size * dir_vec[1]

                # PFC fatigue is applied here, per spike, instead of via
                # the relu(V-theta) term in the ODE.
                #
                # Why: under this event-driven model, integration always
                # halts the instant V == theta and is then reset below
                # threshold, so V *never* spends measurable time above
                # theta. relu(V-theta) is therefore numerically ~0 at
                # every point we ever evaluate it (verified: max value
                # ~1e-14 across a full run) and the original equation's
                # fatigue term cannot do anything. The recovery half of
                # the equation (gamma_pfc pulling beta_PFC back toward
                # beta_pfc_max) is still integrated continuously in
                # _odes -- only the fatigue half moves here, re-expressed
                # as a fixed decrement applied once per firing event
                # rather than a rate integrated over a near-zero window.
                # alpha_pfc is now interpreted as "PFC brake lost per
                # spike" instead of "fatigue rate while above threshold" --
                # still the same evolvable gene, same biological story
                # (repeated firing erodes the brake), just expressed in a
                # form that's actually nonzero under this spiking scheme.
                state[3] = max(0.0, state[3] - p['alpha_pfc'])

                # Refractory reset (post-spike hyperpolarization): force V
                # below theta so the next rising crossing is always
                # geometrically possible, regardless of where the drive
                # equilibrium sits (this is the fix for the flat-trajectory
                # bug). At current defaults this also lands biophysically
                # correctly -- between V_rest and theta, not below V_rest --
                # matching standard LIF reset conventions (see
                # refractory_drop docstring). The constructor guard above
                # keeps it from drifting below V_rest if theta/V_rest/
                # refractory_drop are changed later (e.g. by evolution).
                state[1] = p['theta'] - self.refractory_drop

                if n_spikes >= self.max_spikes:
                    # Safety valve: append final state and stop, rather
                    # than looping forever on pathological parameters.
                    break
            else:
                # No spike this segment. If the agent isn't close enough
                # to a stimulus to be meaningfully engaged, take one
                # foraging step.
                I_now = self._stimulus(r_now, t_new)
                if I_now < self.engage_threshold:
                    self._wander_step()
                    engaged_no_spike_ticks = 0
                else:
                    # Engaged but didn't spike this tick. Verified
                    # empirically (see engage_threshold docstring) that
                    # at default parameters this resolves itself within a
                    # few ticks -- the ripple term reliably pushes I over
                    # the firing equilibrium before any real stall
                    # develops, so no movement is taken here on purpose:
                    # adding a "creep toward source" fallback would mask
                    # a genuine stall instead of surfacing it. Instead we
                    # just count consecutive engaged-but-silent ticks and
                    # warn loudly if that streak gets unreasonably long,
                    # since that would mean some future parameter choice
                    # (e.g. a much slower ripple) has reopened a dead zone
                    # this default configuration doesn't have.
                    engaged_no_spike_ticks += 1
                    if engaged_no_spike_ticks == STALL_WARN_TICKS:
                        import warnings
                        # Two genuinely different things can produce this
                        # symptom, and they call for opposite responses:
                        #   1. An engagement-boundary dead zone (threshold
                        #      tuning problem) -- fix by adjusting
                        #      engage_threshold/wander_dt.
                        #   2. Legitimate homeostatic suppression: S and/or
                        #      beta_PFC have been driven down far enough
                        #      by sustained firing that V_eq has dropped
                        #      back below theta even at high I -- this is
                        #      the model doing exactly what it's designed
                        #      to do (the "runaway compulsion crashes
                        #      itself out" regime from the mismatch
                        #      tuning), not a bug. Distinguish using the
                        #      current S/beta_PFC against their resting
                        #      values, both readable off `state` here.
                        S_now, beta_PFC_now = state[0], state[3]
                        suppressed = (
                            S_now < 0.5 * p['S_rest']
                            or beta_PFC_now < 0.2 * p['beta_pfc_max']
                        )
                        if suppressed:
                            warnings.warn(
                                f"Agent engaged but not spiking for "
                                f"{STALL_WARN_TICKS} ticks at r={r_now:.1f}, "
                                f"t={t_new:.1f} -- looks like homeostatic "
                                f"suppression, not a tuning bug: "
                                f"S={S_now:.2f} (rest {p['S_rest']}), "
                                f"beta_PFC={beta_PFC_now:.3f} (ceiling "
                                f"{p['beta_pfc_max']}). Sustained firing has "
                                f"driven sensitivity/PFC brake down far "
                                f"enough that V_eq has dropped back below "
                                f"theta even at high I. This is the "
                                f"mismatch dynamic working as intended.",
                                RuntimeWarning,
                            )
                        else:
                            warnings.warn(
                                f"Agent has been 'engaged' (I >= "
                                f"engage_threshold) without spiking for "
                                f"{STALL_WARN_TICKS} consecutive wander_dt "
                                f"ticks at r={r_now:.1f}, t={t_new:.1f}, "
                                f"with S/beta_PFC near their resting values "
                                f"(S={S_now:.2f}, beta_PFC={beta_PFC_now:.3f}) "
                                f"-- this does NOT look like homeostatic "
                                f"suppression. This parameter set may have "
                                f"reopened a dead zone between "
                                f"engage_threshold and the true firing "
                                f"catchment radius -- check V_eq(r) against "
                                f"theta, or lower engage_threshold / shorten "
                                f"wander_dt.",
                                RuntimeWarning,
                            )

            # Guard against the solver returning a zero-length step
            # (would otherwise spin forever at fixed t_current).
            if t_new - t_current < MIN_DT and not spiked:
                break

            t_current = t_new

        return AgentHistory(
            t=np.array(times), S=np.array(S_h), V=np.array(V_h), E_R=np.array(E_h),
            beta_PFC=np.array(B_h), r=np.array(r_h), x=np.array(x_h), y=np.array(y_h),
            fire_times=np.array(fire_times), I=np.array(I_h),
        )
