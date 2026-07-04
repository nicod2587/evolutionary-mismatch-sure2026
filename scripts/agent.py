"""
agent.py — Chemotaxis engagement model, wired to the patch world (natural_world.py).

The agent moves by a biased random walk modulated by its reward expectation E_R
(rather than a binary engage/wander toggle), so it stays pulled toward a patch
even after homeostatic suppression stops its spiking — the lag of compulsive
scrolling. Integrate the ODEs and run the loop with simulate().

Pass world=AncestralWorld(...) to enable world mode (no subclass needed). Then:
  - I(r,t) is summed from each patch's sensory_salience (perceived pull, not
    utility): an ancestral patch's drive fades as it drains, a modern one stays
    high even when empty — the supernormal trap.
  - Each step: world.advance(t) runs respawn timers + fade-in; the dominant patch
    comes from world.dominant_patch().
  - On a spike: world.engage_patch() harvests real utility (tracked separately
    from perceived salience) and spike-fades the patch, despawning it if worked
    out. A spike also resets the giving-up clock.
  - Local (per-patch) habituation builds up while it lingers unrewarded and
    erodes that patch's pull, so it tires of a patch and wanders off toward
    novelty; leaving a spent husk lets the world despawn it (world.leave_patch).
    This is the trap-breaker — a stuck agent habituates and leaves.
  - Travel friction scales wander_speed by mismatch_index: costly ancestral
    travel, frictionless modern patch-switching.

With world=None (legacy single-source mode) stimuli are supplied explicitly and
none of the patch/harvest/friction machinery applies.
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
    # ^ Real utility harvested per spike (world.engage_patch's return), aligned
    # 1:1 with fire_times, NOT t. Decoupled from I/eng_weight, which track
    # perceived salience and stay high on an empty modern patch. All-zero in
    # legacy mode (no patch to harvest).
    total_distance: float
    # ^ Total path length over the run (spike hops + wander steps). Cheap "how
    # frantically did it move" diagnostic.
    wander_distance: float = 0.0
    # ^ Path length from wander steps only (excludes spike hops) — the real
    # locomotor travel metabolic_burn is charged against. A patch-trapped agent
    # jitters in place (low wander_distance) while a forager ranging between rare
    # patches accumulates a lot, so this avoids scoring the trap as high effort.
    frames: Optional[list] = None
    # ^ Opt-in animation frames, only when simulate(record_frames>0); else None
    # (GA hot path untouched). Each frame: {'t','x','y','V','w','spiked',
    # 'patches'} with 'patches' = list of (x, y, salience). Consumed by
    # animation.create_world_animation.


def net_travel_distance(history: "AgentHistory", window: float = 500.0) -> float:
    """
    Net travel distance: resample the trajectory onto a time grid of spacing
    `window` and sum the straight-line hops between grid points. Sub-window
    jitter cancels, so an agent fidgeting on a patch scores ~0 while one making
    long journeys between patches accumulates real distance. This is the range
    measure metabolic_burn is charged against (unlike the raw path lengths
    total_distance / wander_distance).
    """
    t = history.t
    if len(t) < 2:
        return 0.0
    tg = np.arange(float(t[0]), float(t[-1]), float(window))
    if len(tg) < 2:
        return float(np.hypot(history.x[-1] - history.x[0],
                              history.y[-1] - history.y[0]))
    xg = np.interp(tg, t, history.x)
    yg = np.interp(tg, t, history.y)
    return float(np.sum(np.hypot(np.diff(xg), np.diff(yg))))


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
        patch_harvest_amount: float = 0.10,
        habituation_rise: float = 200.0,
        habituation_recovery: float = 2000.0,
        habituation_reward_relief: float = 0.15,
        habituation_radius: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        World-mode knobs (ignored when world is None):

        travel_friction_scale — max wander-speed penalty at mismatch_index=0
            (ancestral), scaling to 0 at mismatch_index=1 (modern).

        patch_harvest_amount — utility a spike tries to drain (clipped to what's
            left). Also the per-spike salience fade, so with the profile's
            salience_scale it sets the depletion timescale: worked-out despawn
            takes ~0.75*salience_scale/patch_harvest_amount spikes. Smaller =
            more gradual (a fresh patch spikes fast, so too large a value drains
            it almost instantly); 0.10 keeps the despawn watchable.

        Local habituation — a per-patch "I've had enough of this one" signal the
        agent builds up while sitting in a patch, which suppresses that patch's
        perceived pull (in _eff_salience) so the agent tires of it and wanders
        off toward novelty. It is what lets the agent leave a patch without the
        world deleting it (see world.leave_patch), and it is the trap-breaker: a
        stuck, non-spiking agent habituates and leaves instead of freezing.
          habituation_rise — timescale (sim-time) to habituate to a patch while
            sitting in it with no reward; smaller = leaves sooner.
          habituation_recovery — timescale to un-habituate once away, so the
            agent can eventually be drawn back. Keep >> the time to walk out of a
            patch, or it gets re-captured before it escapes.
          habituation_reward_relief — how much a rewarded spike knocks habituation
            back down, so a patch the agent is actually harvesting does not get
            abandoned (only unproductive lingering builds habituation up).
          habituation_radius — the agent only habituates while within this many
            sigma of the patch centre (i.e. actually in it).
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
        self._habituation: dict = {}   # patch_id -> local habituation h in [0,1)

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
        self.habituation_rise      = float(habituation_rise)
        self.habituation_recovery  = float(habituation_recovery)
        self.habituation_reward_relief = float(habituation_reward_relief)
        self.habituation_radius    = float(habituation_radius)
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

    def _hab(self, patch: "Patch") -> float:
        """This agent's local habituation to `patch`, in [0,1)."""
        return self._habituation.get(patch.patch_id, 0.0)

    def _patch_intensity(self, patch: "Patch", t: float) -> float:
        """
        Perceived I(r,t) this agent gets from `patch` at its current position,
        using habituation-suppressed salience (eff = salience * (1 - h)). Both
        the total drive and the choice of dominant patch use this, so a patch the
        agent has tired of both drives the neuron less and stops being dominant.
        """
        eff = patch.sensory_salience * (1.0 - self._hab(patch))
        if eff <= 1e-7:
            return 0.0
        prof = patch.profile
        r = np.sqrt((self.x - patch.x) ** 2 + (self.y - patch.y) ** 2)
        env    = prof['A'] * np.exp(-((r ** 2) / (2.0 * prof['sigma'] ** 2)) ** (prof['n'] / 2.0))
        ripple = 1.0 + prof['alpha'] * np.sin(prof['omega'] * t - prof['k'] * r)
        return float(eff * env * ripple)

    def _total_I_world(self, t: float) -> float:
        """World-mode hot path: sum habituation-suppressed I(r,t) over patches, so
        an empty-but-glowing modern patch still drives the neuron (utility gone,
        salience high) but a habituated one drives it less."""
        return float(sum(self._patch_intensity(p, t) for p in self.world.patches))

    def _dominant_source(self, t: float) -> StimulusSource:
        """
        The strongest local stimulus source (by habituation-suppressed
        intensity). World mode: pick the max-intensity patch, wrap it via
        Patch.to_source() (silent-source fallback if none is perceptible), and
        update tracked occupancy. Legacy mode: argmax over self.stimuli.
        """
        if self.world is not None:
            from .natural_world import _silent_source
            best, best_val = None, 0.0
            for p in self.world.patches:
                v = self._patch_intensity(p, t)
                if v > best_val:
                    best_val, best = v, p
            self._update_current_patch(best, t)
            if best is None:
                return _silent_source(self.world.world_half)
            src = best.to_source(t)
            if src is None:
                return _silent_source(self.world.world_half)
            return src
        return max(self.stimuli, key=lambda s: s.eval_at(self.x, self.y, t))

    def _update_current_patch(self, patch: Optional["Patch"], t: float) -> None:
        """Track the dominant patch. On leaving one, tell the world (which
        despawns it only if it is a spent husk — see world.leave_patch)."""
        if patch is self._current_patch:
            return
        if self._current_patch is not None:
            self.world.leave_patch(self._current_patch, t)
        self._current_patch = patch

    def _update_habituation(self, dt: float) -> None:
        """
        Advance local habituation by dt. The current patch habituates (h -> 1)
        while the agent sits in its core with no reward; every other patch
        un-habituates (h -> 0). Rewarded spikes relieve the current patch's
        habituation at the spike site, so a patch being actively harvested does
        not get abandoned. When h gets high the patch's perceived pull collapses
        (see _patch_intensity) and the agent wanders off toward novelty.
        """
        if dt <= 0.0:
            return
        cur = self._current_patch
        cur_id = None
        if cur is not None:
            core = self.habituation_radius * cur.profile['sigma']
            if (self.x - cur.x) ** 2 + (self.y - cur.y) ** 2 < core * core:
                cur_id = cur.patch_id
                h = self._habituation.get(cur_id, 0.0)
                self._habituation[cur_id] = h + (1.0 - h) * (1.0 - np.exp(-dt / self.habituation_rise))
        decay = np.exp(-dt / self.habituation_recovery)
        for pid in list(self._habituation.keys()):
            if pid == cur_id:
                continue
            h = self._habituation[pid] * decay
            if h < 1e-3:
                del self._habituation[pid]
            else:
                self._habituation[pid] = h

    def _harvest_patch(self, patch: Optional["Patch"], t: float) -> float:
        """
        Harvest `patch` on a spike via world.engage_patch (drains utility and
        spike-fades salience); returns the real utility harvested — 0.0 on an
        exhausted modern patch even though it keeps firing spikes.
        """
        if patch is None or self.world is None:
            return 0.0
        return float(self.world.engage_patch(patch, self.patch_harvest_amount, t))

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
        Chemotactic step blending a random walk with gradient pull toward the
        dominant source. As engagement weight w rises, turn noise drops, speed
        falls to the min_speed_fraction floor, and heading locks onto the source.
        World mode adds silent-source suppression and travel friction (see the
        two helpers below).
        """
        if self.world is not None:
            self._biased_wander_step_world(E_R, t)
        else:
            self._biased_wander_step_core(E_R, t)

    def _biased_wander_step_world(self, E_R: float, t: float) -> None:
        """World-mode wander step: apply travel friction (full ancestral, none
        modern), and suppress gradient pull when no patch is perceptible."""
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

    def simulate(self, t_span: tuple, rtol: float = 1e-6, atol: float = 1e-8,
                 record_frames: int = 0) -> AgentHistory:
        """
        Integrate the ODEs segment by segment, halting each segment at a
        threshold crossing (a spike). A spike harvests the dominant patch
        (world.engage_patch), decrements the PFC brake, and resets V; between
        spikes the agent chemotaxis-wanders (approach is entirely chemotaxis — no
        spatial lunge). The world clock, respawns, and habituation are updated
        inline in the loop.

        record_frames > 0 captures that many evenly-spaced snapshots (pose +
        patch layout) into AgentHistory.frames for the animation; it is off by
        default so the GA hot path pays nothing.
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
        wander_distance = 0.0

        times, S_h, V_h, E_h, B_h = [], [], [], [], []
        x_h, y_h, r_h, I_h, ew_h = [], [], [], [], []
        fire_times = []
        harvested_h = []

        # ── opt-in animation frame recording ──────────────────────────────────
        frames = None
        frame_times = None
        next_frame = 0
        if record_frames and record_frames > 0:
            frames = []
            frame_times = np.linspace(t_start, t_end, int(record_frames))

        def _record_frame(t_now: float, cur_state: np.ndarray, spiked_now: bool) -> None:
            if self.world is not None:
                patch_snap = [(pt.x, pt.y, pt.sensory_salience) for pt in self.world.patches]
            else:
                patch_snap = [(s.x, s.y, 1.0) for s in self.stimuli]
            frames.append({
                't': float(t_now), 'x': float(self.x), 'y': float(self.y),
                'V': float(cur_state[1]),
                'w': float(self._engagement_weight(cur_state[2])),
                'spiked': bool(spiked_now), 'patches': patch_snap,
            })

        n_spikes   = 0
        slow_ticks = 0

        while t_current < t_end:
            old_x, old_y = self.x, self.y  # position before this iteration's movement
            # Step the world clock: process any Poisson-delayed patch respawns
            # and advance fade-in ramps so this segment sees current salience.
            if self.world is not None:
                self.world.advance(t_current)
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
                harvested = self._harvest_patch(patch, t_new)
                harvested_h.append(harvested)

                # A rewarded spike relieves habituation, so a patch the agent is
                # actually harvesting keeps its pull; only unproductive lingering
                # builds habituation up. (No spatial "lunge" — approach is left
                # entirely to chemotaxis, so movement stays coupled to the walk.)
                if harvested > 0.0 and self._current_patch is not None:
                    pid = self._current_patch.patch_id
                    self._habituation[pid] = max(
                        0.0, self._habituation.get(pid, 0.0) - self.habituation_reward_relief
                    )

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

            # Advance local habituation: the current patch tires the agent while
            # it lingers there unrewarded, which erodes that patch's pull (via
            # _patch_intensity) until the agent disengages and moves on — the
            # trap-breaker, and what makes a full modern patch a passing graze
            # rather than a permanent hook. Leaving is emergent: as the pull
            # falls, the dominant patch changes and _update_current_patch tells
            # the world (which despawns the patch only if it is a spent husk).
            if self.world is not None:
                self._update_habituation(t_new - t_current)

            # Distance this iteration. Charge only wander steps to wander_distance.
            disp = float(np.linalg.norm([self.x - old_x, self.y - old_y]))
            total_distance += disp
            if not spiked:
                wander_distance += disp

            # Record any animation frames whose scheduled time has now elapsed.
            if frames is not None:
                while next_frame < len(frame_times) and t_new >= frame_times[next_frame]:
                    _record_frame(frame_times[next_frame], state, spiked)
                    next_frame += 1

            if t_new - t_current < MIN_DT and not spiked:
                break

            t_current = t_new

        # Emit any still-pending frames (e.g. a run that halted early on a MIN_DT
        # stall) so the frame list always has the requested length.
        if frames is not None:
            while next_frame < len(frame_times):
                _record_frame(frame_times[next_frame], state, False)
                next_frame += 1

        return AgentHistory(
            t=np.array(times),       S=np.array(S_h),       V=np.array(V_h),
            E_R=np.array(E_h),       beta_PFC=np.array(B_h),
            r=np.array(r_h),         x=np.array(x_h),        y=np.array(y_h),
            fire_times=np.array(fire_times),
            I=np.array(I_h),         eng_weight=np.array(ew_h),
            harvested_utility=np.array(harvested_h),
            total_distance=total_distance,
            wander_distance=wander_distance,
            frames=frames,
        )