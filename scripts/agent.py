"""
The forager: one neuron with two variables, moving around a World.

  V     membrane potential, a leaky integrate-and-fire unit (Hayden et al. 2011).
        Scent pushes it up, and each spike takes a bite of value from a patch.
  beta  opportunity cost (Kolling et al. 2012). It builds up on a patch that smells
        worse than R_bar, and it is subtracted from V's input.

        dV/dt    = (-(V - V_rest) + I - beta) / tau_v
        dbeta/dt = (prox * (R_bar - I) - (1 - prox) * beta) / tau_pfc

V needs a drive of 10 to fire, so the forager leaves once the scent drops below
max(10, R_bar). It steers by scent, never by real value, so in the modern world an
empty patch that still smells novel can keep it hooked.
"""

import math
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from .world import World, Patch

MIN_DT = 1e-9   # stop if the solver stops making progress


@dataclass
class AgentHistory:
    """What one simulate() call records. fire_times and harvested_utility have one
    entry per spike; everything else shares the time axis t."""
    t: np.ndarray
    V: np.ndarray
    beta_PFC: np.ndarray
    r: np.ndarray                  # distance to the patch the forager is heading for
    x: np.ndarray
    y: np.ndarray
    I: np.ndarray                  # total perceived scent (the neuron's input)
    eng_weight: np.ndarray         # engagement, 0 (searching) to 1 (locked on a patch)
    fire_times: np.ndarray
    harvested_utility: np.ndarray  # real value gained by each spike (0 on an empty patch)
    frames: Optional[list] = None  # snapshots for animation, see simulate(record_frames=)


class Agent:
    """One forager in a World. params holds the neuron constants plus R_bar and kappa.

    Every wander_dt the forager takes a step that mixes a correlated random walk
    with a pull toward the strongest patch. Engagement sets the mix: a sigmoid of
    scent minus beta, centred near the firing drive (engage_fraction, engage_scale).
    With a patch in range it moves travel_friction slower. Each spike takes
    patch_harvest_amount of value.
    """

    def __init__(
        self,
        params: dict,
        world: "World",
        *,
        x0: float = -200.0,
        y0: float = -240.0,
        refractory_drop: float = 5.0,     # after a spike V resets to theta - this
        max_spikes: int = 100_000,        # stop the run early after this many spikes
        wander_dt: float = 2.0,
        wander_speed: float = 4.0,
        turn_noise: float = 0.15,
        engage_fraction: float = 1.0,
        engage_scale: float = 3.0,
        min_speed_fraction: float = 0.25,
        travel_friction: float = 3.0,
        patch_harvest_amount: float = 0.10,
        beta_cap: float = 45.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.p = dict(params)
        self.world = world
        self._current_patch: Optional["Patch"] = None

        if refractory_drop <= 0:
            raise ValueError("refractory_drop must be > 0.")
        if self.p['theta'] - refractory_drop <= self.p['V_rest']:
            raise ValueError("theta - refractory_drop must stay above V_rest.")

        self.x = float(x0)
        self.y = float(y0)
        self.world_half = world.world_half
        self.refractory_drop = float(refractory_drop)
        self.max_spikes = int(max_spikes)
        self.wander_dt = float(wander_dt)
        self.wander_speed = float(wander_speed)
        self.turn_noise = float(turn_noise)
        self.engage_fraction = float(engage_fraction)
        self.engage_scale = float(engage_scale)
        self.min_speed_fraction = float(min_speed_fraction)
        self.travel_friction = float(travel_friction)
        self.patch_harvest_amount = float(patch_harvest_amount)
        self.beta_cap = float(beta_cap)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.heading = float(self.rng.uniform(0, 2 * np.pi))

    # -- engagement ----------------------------------------------------------

    @property
    def I_fire(self) -> float:
        """Drive needed to reach threshold from rest when beta is 0. V settles at
        V_rest + I - beta, so firing needs I - beta >= theta - V_rest (= 10)."""
        return self.p['theta'] - self.p['V_rest']

    def _engagement_weight(self, net_drive: float) -> float:
        z = (net_drive - self.engage_fraction * self.I_fire) / self.engage_scale
        if z < -700.0:   # exp would overflow; the forager is fully disengaged
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    # -- perception ----------------------------------------------------------
    # These run inside the ODE right-hand side, hundreds of thousands of times per
    # simulation, so they use scalar math.* rather than numpy.

    def _patch_intensity(self, patch: "Patch", t: float) -> float:
        """Scent this forager gets from one patch, where it stands now."""
        eff = patch.fade_in_factor(t) * patch.scent(self.p['kappa'],
                                                    self.world.obs_noise_at(patch, t))
        if eff <= 1e-7:
            return 0.0
        prof = patch.profile
        dx = self.x - patch.x; dy = self.y - patch.y
        r = math.sqrt(dx * dx + dy * dy)
        env = prof['A'] * math.exp(-((r * r) / (2.0 * prof['sigma'] ** 2)) ** (prof['n'] / 2.0))
        ripple = 1.0 + prof['alpha'] * math.sin(prof['omega'] * t - prof['k'] * r)
        return eff * env * ripple

    def _total_I(self, t: float) -> float:
        return float(sum(self._patch_intensity(p, t) for p in self.world.patches))

    def _drive_and_proximity(self, t: float) -> tuple:
        """(I, prox) in one pass. prox runs from 1 on top of a patch to 0 in the
        open and ignores value, so it can tell an empty patch from open ground."""
        I = 0.0
        best_env = 0.0
        Aref = 1e-9
        kappa = self.p['kappa']
        for patch in self.world.patches:
            prof = patch.profile
            if prof['A'] > Aref:
                Aref = prof['A']
            dx = self.x - patch.x; dy = self.y - patch.y
            r = math.sqrt(dx * dx + dy * dy)
            env = prof['A'] * math.exp(-((r * r) / (2.0 * prof['sigma'] ** 2)) ** (prof['n'] / 2.0))
            if env > best_env:
                best_env = env
            # same arithmetic as _patch_intensity, reusing env
            eff = patch.fade_in_factor(t) * patch.scent(kappa, self.world.obs_noise_at(patch, t))
            if eff > 1e-7:
                ripple = 1.0 + prof['alpha'] * math.sin(prof['omega'] * t - prof['k'] * r)
                I += eff * env * ripple
        prox = best_env / Aref
        return I, (1.0 if prox > 1.0 else prox)

    def _dominant_patch(self, t: float) -> Optional["Patch"]:
        """The patch the forager smells most, or None. Switching away from a patch
        tells the world, which is how spent patches get cleared."""
        best, best_val = None, 0.0
        for p in self.world.patches:
            v = self._patch_intensity(p, t)
            if v > best_val:
                best_val, best = v, p
        if best is not self._current_patch:
            if self._current_patch is not None:
                self.world.leave_patch(self._current_patch, t)
            self._current_patch = best
        if best is None or best.sensory_salience <= 1e-7:
            return None
        return best

    def _distance_to(self, patch: Optional["Patch"]) -> float:
        # With no patch in range the forager is measured against the arena centre.
        sx, sy = (patch.x, patch.y) if patch is not None else (0.0, 0.0)
        return float(np.sqrt((self.x - sx) ** 2 + (self.y - sy) ** 2))

    # -- movement ------------------------------------------------------------

    def _wander_step(self, net_drive: float, t: float) -> None:
        """One movement step between spikes."""
        if self._dominant_patch(t) is None:
            self._move(-np.inf, t)   # nothing to smell: pure random search
            return
        base_speed = self.wander_speed
        self.wander_speed = max(0.0, base_speed - self.travel_friction)
        try:
            self._move(net_drive, t)
        finally:
            self.wander_speed = base_speed

    def _move(self, net_drive: float, t: float) -> None:
        """Correlated random walk plus a pull toward the strongest patch, mixed by
        the engagement weight. Bounces off the arena walls."""
        w = self._engagement_weight(net_drive)

        self.heading += self.rng.normal(0.0, max(self.turn_noise * (1.0 - w), 1e-6))
        speed = self.wander_speed * (
            self.min_speed_fraction + (1.0 - self.min_speed_fraction) * (1.0 - w))
        rand_dx = speed * np.cos(self.heading)
        rand_dy = speed * np.sin(self.heading)

        grad_dx, grad_dy = 0.0, 0.0
        if w > 0.01:
            patch = self._dominant_patch(t)
            tx, ty = (patch.x, patch.y) if patch is not None else (0.0, 0.0)
            dx, dy = tx - self.x, ty - self.y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist >= 1e-9:
                pull = w * self.wander_speed * (1.0 - self.min_speed_fraction)
                grad_dx = pull * (dx / dist)
                grad_dy = pull * (dy / dist)

        new_x = self.x + rand_dx + grad_dx
        new_y = self.y + rand_dy + grad_dy
        wh = self.world_half
        if new_x > wh or new_x < -wh:
            self.heading = np.pi - self.heading
            new_x = np.clip(new_x, -wh, wh)
        if new_y > wh or new_y < -wh:
            self.heading = -self.heading
            new_y = np.clip(new_y, -wh, wh)
        self.x = new_x
        self.y = new_y

    # -- neuron --------------------------------------------------------------

    def _odes(self, t: float, y: np.ndarray) -> list:
        p = self.p
        V, beta = y
        I, prox = self._drive_and_proximity(t)
        beta_eff = max(0.0, beta)

        dVdt = (1.0 / p['tau_v']) * (-(V - p['V_rest']) + I - beta_eff)

        # beta only builds up while the forager is on a patch (prox near 1) and
        # decays while it travels (prox near 0). Without this gate, beta would
        # climb to its cap during long ancestral journeys and the forager would
        # arrive at the next patch already wanting to leave.
        dbetadt = (1.0 / p['tau_pfc']) * (
            prox * (p['R_bar'] - I) - (1.0 - prox) * beta_eff
        )
        if beta <= 0.0 and dbetadt < 0.0:
            dbetadt = 0.0
        if beta >= self.beta_cap and dbetadt > 0.0:
            dbetadt = 0.0
        return [dVdt, dbetadt]

    # -- main loop -----------------------------------------------------------

    def simulate(self, t_span: tuple, rtol: float = 1e-6, atol: float = 1e-8,
                 record_frames: int = 0) -> AgentHistory:
        """Run from t_span[0] to t_span[1], in chunks of wander_dt. A chunk that
        hits threshold is a spike (take a bite, reset V); otherwise the forager
        takes a step. record_frames > 0 keeps that many snapshots for the video."""
        t_start, t_end = t_span
        p = self.p

        def spike_event(t, y):
            return y[0] - p['theta']
        spike_event.terminal = True
        spike_event.direction = 1

        state = np.array([p['V_rest'], 0.0], dtype=float)   # [V, beta]
        t_current = t_start

        times, V_h, B_h, x_h, y_h, r_h, I_h, ew_h = [], [], [], [], [], [], [], []
        fire_times, harvested_h = [], []

        frames = None
        if record_frames and record_frames > 0:
            frames = []
            frame_times = np.linspace(t_start, t_end, int(record_frames))
            next_frame = 0

        def _record_frame(t_now: float, cur_state: np.ndarray, spiked_now: bool) -> None:
            net_drive = self._total_I(t_now) - max(0.0, cur_state[1])
            frames.append({
                't': float(t_now), 'x': float(self.x), 'y': float(self.y),
                'V': float(cur_state[0]),
                'w': float(self._engagement_weight(net_drive)),
                'spiked': bool(spiked_now),
                'patches': [(pt.x, pt.y, pt.sensory_salience) for pt in self.world.patches],
            })

        n_spikes = 0
        while t_current < t_end:
            self.world.advance(t_current)
            segment_end = min(t_current + self.wander_dt, t_end)
            sol = solve_ivp(
                fun=self._odes, t_span=(t_current, segment_end), y0=state,
                method='RK45', events=spike_event, rtol=rtol, atol=atol, max_step=np.inf,
            )
            if not sol.success:
                raise RuntimeError(f"Integration failed at t={t_current}: {sol.message}")

            start_idx = 1 if times else 0   # skip the point shared with the last chunk
            r_now = self._distance_to(self._dominant_patch(sol.t[-1]))
            for j in range(start_idx, len(sol.t)):
                times.append(sol.t[j])
                V_h.append(sol.y[0, j]); B_h.append(max(0.0, sol.y[1, j]))
                x_h.append(self.x); y_h.append(self.y)
                r_h.append(r_now)
                I_j = self._total_I(sol.t[j])
                I_h.append(I_j)
                ew_h.append(self._engagement_weight(I_j - max(0.0, sol.y[1, j])))

            state = sol.y[:, -1].copy()
            t_new = sol.t[-1]
            spiked = sol.status == 1

            if spiked:
                n_spikes += 1
                fire_times.append(t_new)
                patch = self._dominant_patch(t_new)
                harvested = 0.0
                if patch is not None:
                    harvested = float(self.world.engage_patch(patch, self.patch_harvest_amount, t_new))
                harvested_h.append(harvested)
                state[0] = p['theta'] - self.refractory_drop
                if n_spikes >= self.max_spikes:
                    break
            else:
                net_now = self._total_I(t_new) - max(0.0, state[1])
                self._wander_step(net_now, t_new)

            if frames is not None:
                while next_frame < len(frame_times) and t_new >= frame_times[next_frame]:
                    _record_frame(frame_times[next_frame], state, spiked)
                    next_frame += 1

            if t_new - t_current < MIN_DT and not spiked:
                break
            t_current = t_new

        if frames is not None:
            while next_frame < len(frame_times):
                _record_frame(frame_times[next_frame], state, False)
                next_frame += 1

        return AgentHistory(
            t=np.array(times), V=np.array(V_h), beta_PFC=np.array(B_h),
            r=np.array(r_h), x=np.array(x_h), y=np.array(y_h),
            I=np.array(I_h), eng_weight=np.array(ew_h),
            fire_times=np.array(fire_times),
            harvested_utility=np.array(harvested_h),
            frames=frames,
        )
