"""
A neural agent for the 2D space. Integrates the (S, V, E_R, beta_PFC)
ODE system continuously, and on each threshold-crossing spike, takes one
step toward the stimulus source.

After each spike:
  1. The agent steps toward the source.
  2. V is reset to a value below theta so it can cross again
  3. Integration resumes from the spike time.

explicitly reset V below theta on every spike (neural refractory
reset) so an agent driven above threshold will correctly spike repeatedly
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Optional
from functools import partial


@dataclass
class AgentHistory:
    #data from Agent.simulate()
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
    def __init__(
        self,
        params: dict, #from model_dynamics.py
        genome: Optional[dict] = None, # overrides for evolvable traits
        x0: float = -200.0, #starting position, this is just for testing
        y0: float = -240.0,
        source_x: float = 0.0, #idk if this should be here, might change later when i do the poission random enviroement
        source_y: float = 0.0, #will need to figure out how the agent will know where the source is though
        step_size: float = 5.0,
        refractory_drop: float = 5.0, #how far below theta to rest, biologically isnt this below resting?
        stimulus_func: Optional[Callable] = None, #dont really understand this yet  stimulus_func : callable, optional f(r, t, params) -> float. Falls back to the default Gaussian envelope x ripple if not provided.
        max_spikes: int = 100_000, #just saftey for the moment
    ):
        self.p = dict(params)
        if genome is not None:
            self.p.update(genome)

        # !!!! this part was added by ai but i dont want to remove as i dont see the harm in keeping it
        
        # Fail loudly instead of
        # silently inventing a number for a parameter that changes the
        # dynamics (tau_pfc, beta_pfc_max are both load-bearing).
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
            if 'pfc_base' in self.p:
                self.p['beta_pfc_max'] = self.p['pfc_base']
            elif 'beta_pfc' in self.p:
                self.p['beta_pfc_max'] = self.p['beta_pfc']
            else:
                raise KeyError(
                    "Need one of 'beta_pfc_max', 'pfc_base', or 'beta_pfc' in params."
                )

        if refractory_drop <= 0:
            raise ValueError("refractory_drop must be > 0, or spikes cannot re-trigger.")

        self.x = float(x0)
        self.y = float(y0)
        self.source_x = float(source_x)
        self.source_y = float(source_y)
        self.step_size = float(step_size)
        self.refractory_drop = float(refractory_drop)
        self.stimulus_func = stimulus_func
        self.max_spikes = int(max_spikes)



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

    def _odes(self, t: float, y: np.ndarray, r: float) -> list:
        p = self.p
        S, V, E_R, beta_PFC = y
        I = self._stimulus(r, t)

        relu = lambda v: np.maximum(0.0, v)

        dSdt = (1 / p['tau_s']) * (
            p['gamma_s'] * (p['S_rest'] - S) - p['alpha_s'] * relu(I - E_R)
        )
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
        MIN_DT = 1e-9  # guards against zero-length integration windows

        while t_current < t_end:
            r_now = self._r(self.x, self.y)

            event_fn = partial(spike_event, r=r_now)
            event_fn.terminal = True
            event_fn.direction = 1  # only rising (V increasing through theta) crossings

            sol = solve_ivp(
                fun=partial(self._odes, r=r_now),
                t_span=(t_current, t_end),
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

                # True refractory reset: force V *below* theta so the
                # next rising crossing is always geometrically possible,
                # regardless of where the drive equilibrium sits. This
                # is the fix for the flat-trajectory bug.
                state[1] = p['theta'] - self.refractory_drop

                if n_spikes >= self.max_spikes:
                    # Safety valve: append final state and stop, rather
                    # than looping forever on pathological parameters.
                    break

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
