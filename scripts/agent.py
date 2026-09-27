"""
agent.py — An information-foraging agent with a minimal 2-state neuron.

The agent forages for information across patches (natural_world.py). Its brain is
two numbers:

  V     — a leaky integrate-and-fire membrane. Scent drives it up; when it crosses
          threshold it spikes, harvests real value from the patch, and resets.
          (The primate dACC foraging neuron; Hayden et al. 2011.)
  β_PFC — an opportunity-cost "time to leave" signal. It accumulates how far the
          patch falls short of the agent's heritable richness prior R̄:
          dβ/dt = (R̄ − I)/τ_pfc. When a patch underperforms, β rises, suppresses
          V, and the agent wanders off — an emergent give-up rule.
          (Kolling et al. 2012 https://www.science.org/doi/10.1126/science.1216930;
           Charnov's Marginal Value Theorem, 1976
           https://www.sciencedirect.com/science/article/abs/pii/004058097690040X)

Key twist: the agent moves on SCENT, not real value. Ancestrally, scent fades as a
patch drains, so a spent patch looks spent. In the modern regime scent stays high
on a worked-out husk (a supernormal cue; Lim & Tan 2024
https://link.springer.com/article/10.1007/s40806-024-00398-z), so nothing pulls
the agent off it — only its own β_PFC can. See natural_world.Patch.scent for the
formula.

That is the whole mismatch: executive control works fine, but it runs an ancestral
computation on modern, decoupled inputs. Not weak willpower — a mismatch.

Usage: pass world=AncestralWorld(...) to forage. world=None is a legacy
single-source mode that bypasses the patch/harvest/β_PFC machinery.
"""

import math
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
    V:          np.ndarray
    beta_PFC:   np.ndarray      # opportunity-cost / value-of-leaving signal
    r:          np.ndarray      # distance to dominant source
    x:          np.ndarray
    y:          np.ndarray
    fire_times: np.ndarray
    I:          np.ndarray      # total perceived information scent (drive)
    eng_weight: np.ndarray      # engagement weight in [0,1] at each timestep
    harvested_utility: np.ndarray
    # ^ Real information value harvested per spike (world.engage_patch's return),
    # aligned 1:1 with fire_times, NOT t. Decoupled from I (perceived scent), which
    # stays high on an empty modern patch. All-zero in legacy mode.
    acquisition_cost: np.ndarray
    # ^ Cost of acquiring information, charged per spike (the patch profile's
    # sampling_cost c), aligned 1:1 with fire_times. High ancestral, ~0 modern.
    total_distance: float
    # ^ Total path length over the run (spike hops + wander steps).
    wander_distance: float = 0.0
    # ^ Path length from wander steps only (excludes spike hops).
    frames: Optional[list] = None
    # ^ Opt-in animation frames, only when simulate(record_frames>0); else None.
    # Each frame: {'t','x','y','V','w','spiked','patches'} with 'patches' =
    # list of (x, y, salience). Consumed by animation.create_world_animation.


def net_travel_distance(history: "AgentHistory", window: float = 500.0) -> float:
    """
    Net travel distance: resample the trajectory onto a time grid of spacing
    `window` and sum the straight-line hops between grid points. Sub-window
    jitter cancels, so an agent fidgeting on a patch scores ~0 while one making
    long journeys between patches accumulates real distance. Kept as a movement
    diagnostic (no longer charged in fitness).
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
        engage_fraction: float = 1.0,
        engage_scale: float = 3.0,
        gradient_bias_strength: float = np.pi / 3,
        min_speed_fraction: float = 0.25,
        world_half: Optional[float] = None,
        travel_friction_scale: float = 3.0,
        patch_harvest_amount: float = 0.10,
        beta_cap: float = 45.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        World-mode knobs (ignored when world is None):

        engage_fraction / engage_scale — the engagement weight is a sigmoid of the
            NET drive (perceived scent minus β_PFC): w = σ((net_drive −
            engage_fraction·I_fire)/engage_scale). When β_PFC suppresses the net
            drive below threshold the agent disengages and wanders off — the
            emergent giving-up that replaces the old habituation system.

        travel_friction_scale — max wander-speed penalty at mismatch_index=0
            (ancestral), scaling to 0 at mismatch_index=1 (modern): costly ancestral
            travel, frictionless modern patch-switching.

        patch_harvest_amount — real information value a spike tries to drain (clipped
            to what's left). Smaller = more gradual depletion.

        beta_cap — ceiling on β_PFC so the opportunity-cost signal saturates instead
            of integrating without bound.

        The genome supplies three neural traits via `params`/`genome` (see
        fitness.Genome): R_bar (background-richness prior), kappa
        (curiosity gain on novelty), tau_pfc (opportunity-cost accumulation rate).
        """
        # ── world vs stimuli ──────────────────────────────────────────────────
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
            from .natural_world import _silent_source
            initial = world.get_active_sources(0.0) or [_silent_source(world.world_half)]
            self.stimuli = initial
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

        # Grounded genome traits with safe fallbacks (so a bare params dict runs).
        self.p.setdefault('R_bar', 6.0)     # background-richness prior (drive units)
        self.p.setdefault('kappa', 1.0)     # curiosity gain on the novelty term

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
        self.engage_fraction   = float(engage_fraction)
        self.engage_scale      = float(engage_scale)
        self.gradient_bias_strength = float(gradient_bias_strength)
        self.min_speed_fraction = float(min_speed_fraction)
        self.world_half        = float(world_half) if world_half is not None else None
        self.travel_friction_scale = float(travel_friction_scale)
        self.patch_harvest_amount  = float(patch_harvest_amount)
        self.beta_cap          = float(beta_cap)
        self.rng               = rng if rng is not None else np.random.default_rng()
        self.heading           = float(self.rng.uniform(0, 2 * np.pi))

    # ── neural threshold ──────────────────────────────────────────────────────

    @property
    def I_fire(self) -> float:
        """Minimum drive (perceived scent, with β_PFC=0) to reach firing threshold
        from rest. With dV/dt = (−(V−V_rest) + I − β_PFC)/τ_v, V_eq = V_rest + I −
        β_PFC, so firing (V_eq ≥ θ) needs I − β_PFC ≥ θ − V_rest."""
        return self.p['theta'] - self.p['V_rest']

    # ── engagement ────────────────────────────────────────────────────────────

    @property
    def _engage_threshold(self) -> float:
        """Net-drive inflection point where the engagement weight equals 0.5."""
        return self.engage_fraction * self.I_fire

    def _engagement_weight(self, net_drive: float) -> float:
        """
        Map NET drive (perceived scent − β_PFC) to a smooth [0,1] engagement weight
        via a sigmoid. High net drive → locked onto the patch (exploit); once β_PFC
        pushes the net drive below threshold the weight collapses and the agent
        wanders off — the emergent MVT giving-up.
        """
        z = (net_drive - self._engage_threshold) / self.engage_scale
        # Guard against overflow in exp for very negative net_drive (disengaged).
        if z < -700.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    # ── stimulus / scent ──────────────────────────────────────────────────────

    def _total_I(self, t: float) -> float:
        """Total perceived information scent (drive) at the current position."""
        if self.world is not None:
            return self._total_I_world(t)
        return float(sum(s.eval_at(self.x, self.y, t) for s in self.stimuli))

    def _patch_intensity(self, patch: "Patch", t: float) -> float:
        """
        Perceived scent I(r,t) this agent gets from `patch` at its current position.
        The perceptual value is patch.scent evaluated with the agent's curiosity gain κ
        on the novelty term (patch.scent(κ)), times the fade-in ramp, the spatial
        envelope, and the temporal ripple.
        """
        eff = patch.fade_in_factor(t) * patch.scent(self.p['kappa'],
                                                    self.world.obs_noise_at(patch, t))
        if eff <= 1e-7:
            return 0.0
        prof = patch.profile
        # Scalar math.* is ~5-10x faster than np.* here; this is a per-patch hot path
        # called 100k+ times per simulation (see profiling notes on _drive_and_proximity).
        dx = self.x - patch.x; dy = self.y - patch.y
        r = math.sqrt(dx * dx + dy * dy)
        env    = prof['A'] * math.exp(-((r * r) / (2.0 * prof['sigma'] ** 2)) ** (prof['n'] / 2.0))
        ripple = 1.0 + prof['alpha'] * math.sin(prof['omega'] * t - prof['k'] * r)
        return eff * env * ripple

    def _total_I_world(self, t: float) -> float:
        """World-mode hot path: sum perceived scent over patches, so an
        empty-but-glowing modern patch still drives the neuron (value gone, scent
        high)."""
        return float(sum(self._patch_intensity(p, t) for p in self.world.patches))

    def _drive_and_proximity(self, t: float) -> tuple:
        """
        Return (I, prox) in a single patch pass:

          I     — total perceived information scent driving the neuron (same value
                  as _total_I), value+novelty weighted.
          prox  — spatial-proximity scalar in [0, 1]: max patch spatial envelope
                  at the agent's position, normalised by the profile amplitude A.
                  It is VALUE-INDEPENDENT (pure geometry), so prox ≈ 1 means "sitting
                  on a patch" and prox ≈ 0 means "travelling between patches",
                  REGARDLESS of whether that patch is depleted. This is the one
                  cue that separates "on a worked-out patch" (I low because value
                  drained) from "in transit" (I low because no patch is near) —
                  the two states the old beta_PFC rule conflated.

        Used by _odes to gate the opportunity-cost signal: accumulate while on a
        patch, relax while travelling.
        """
        if self.world is None:
            I = 0.0
            best_env = 0.0
            Aref = 1e-9
            for s in self.stimuli:
                I += s.eval_at(self.x, self.y, t)
                env = s.envelope_at(s.r_from(self.x, self.y))
                if env > best_env:
                    best_env = env
                if s.A > Aref:
                    Aref = s.A
            return float(I), float(min(1.0, best_env / Aref))

        I = 0.0
        best_env = 0.0
        Aref = 1e-9
        kappa = self.p['kappa']
        # Hot path (top of the profile): scalar math.* instead of np.* — the ODE
        # solver calls this ~20x per integration segment, 100k+ times per sim.
        for patch in self.world.patches:
            prof = patch.profile
            if prof['A'] > Aref:
                Aref = prof['A']
            dx = self.x - patch.x; dy = self.y - patch.y
            r = math.sqrt(dx * dx + dy * dy)
            env = prof['A'] * math.exp(-((r * r) / (2.0 * prof['sigma'] ** 2)) ** (prof['n'] / 2.0))
            if env > best_env:
                best_env = env
            # perceived scent contribution (reuses the same envelope) — matches
            # _patch_intensity exactly so I stays consistent with _total_I.
            eff = patch.fade_in_factor(t) * patch.scent(kappa, self.world.obs_noise_at(patch, t))
            if eff > 1e-7:
                ripple = 1.0 + prof['alpha'] * math.sin(prof['omega'] * t - prof['k'] * r)
                I += eff * env * ripple
        prox = best_env / Aref
        return I, (1.0 if prox > 1.0 else prox)

    def _dominant_source(self, t: float) -> StimulusSource:
        """
        The strongest local source by perceived scent. World mode: pick the
        max-intensity patch, wrap it via Patch.to_source() (silent-source fallback
        if none is perceptible), and update tracked occupancy. Legacy mode: argmax
        over self.stimuli.
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
        """Track the dominant patch. On leaving one, tell the world (which despawns
        it only if it is a spent husk — see world.leave_patch)."""
        if patch is self._current_patch:
            return
        if self._current_patch is not None:
            self.world.leave_patch(self._current_patch, t)
        self._current_patch = patch

    def _harvest_patch(self, patch: Optional["Patch"], t: float) -> float:
        """
        Harvest `patch` on a spike via world.engage_patch (drains real information
        value); returns the value actually harvested — 0.0 on an exhausted modern
        patch even though it keeps firing spikes.
        """
        if patch is None or self.world is None:
            return 0.0
        return float(self.world.engage_patch(patch, self.patch_harvest_amount, t))

    def _patch_cost(self, patch: Optional["Patch"]) -> float:
        """Acquisition cost c charged for sampling `patch` (profile sampling_cost)."""
        if patch is None or self.world is None:
            return 0.0
        return float(patch.profile.get('sampling_cost', 0.0))

    # ── movement ──────────────────────────────────────────────────────────────

    def _gradient_direction_to(self, sx: float, sy: float) -> np.ndarray:
        """Unit vector from current position toward (sx, sy). Zero if already there."""
        dx, dy = sx - self.x, sy - self.y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist < 1e-9:
            return np.array([0.0, 0.0])
        return np.array([dx / dist, dy / dist])

    def _biased_wander_step(self, net_drive: float, t: float) -> None:
        """Chemotactic step blending a random walk with gradient pull toward the
        dominant source, modulated by the engagement weight. World mode adds
        silent-source suppression and travel friction."""
        if self.world is not None:
            self._biased_wander_step_world(net_drive, t)
        else:
            self._biased_wander_step_core(net_drive, t)

    def _biased_wander_step_world(self, net_drive: float, t: float) -> None:
        """World-mode wander step: travel friction (full ancestral, none modern),
        and suppressed gradient pull when no patch is perceptible."""
        dom = self._dominant_source(t)
        patch = getattr(dom, '_patch', None)

        if patch is None:
            self._biased_wander_step_core(-np.inf, t)  # no pull: pure search
            return

        mismatch_index = self.world.mismatch_index
        friction = (1.0 - mismatch_index) * self.travel_friction_scale
        original_speed = self.wander_speed
        self.wander_speed = max(0.0, original_speed - friction)
        try:
            self._biased_wander_step_core(net_drive, t)
        finally:
            self.wander_speed = original_speed

    def _biased_wander_step_core(self, net_drive: float, t: float) -> None:
        """Pure movement logic — no world awareness. Shared by both modes."""
        w = self._engagement_weight(net_drive)

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
        """2-state ODE: [V, β_PFC].

        V      — leaky integrate-and-fire membrane driven by perceived scent I,
                 gated by the opportunity-cost signal β_PFC.
        β_PFC  — accumulates (R̄ − I)/τ_pfc: rises when the current patch's
                 perceived intake falls below the background-richness prior R̄,
                 suppressing V and triggering the giving-up. Soft-clamped to
                 [0, beta_cap].
        """
        p = self.p
        V, beta_PFC = y
        I, prox = self._drive_and_proximity(t)
        beta_eff = max(0.0, beta_PFC)

        dVdt = (1.0 / p['tau_v']) * (-(V - p['V_rest']) + I - beta_eff)

        # beta_PFC = value-of-leaving / opportunity-cost signal. It accumulates the
        # intake shortfall (R_bar − I) ONLY while the agent is physically on a patch
        # (prox ≈ 1), and relaxes toward 0 while travelling between patches
        # (prox ≈ 0).
        #
        # The old rule integrated (R_bar − I) unconditionally. In sparse ancestral
        # worlds the agent spends most of its time in transit with I ≈ 0, so beta
        # was driven to its cap and PINNED there — which zeroed the engagement
        # weight, killed gradient-homing, and locked the agent out of ever reaching
        # a patch to harvest (0 spikes on unlucky layouts). Gating by spatial
        # proximity makes beta a true patch-level giving-up rule: it rises only when
        # the CURRENT patch's honest intake has dropped below R_bar (depleted → leave)
        # and forgets during travel so the agent arrives at the next patch fresh.
        #
        # The modern trap is unchanged: there the supernormal husk keeps I high even
        # when drained, so (R_bar − I) stays negative and beta stays ≈ 0 whether or
        # not prox is 1 — the agent still over-stays. The mismatch is untouched.
        dbetadt = (1.0 / p['tau_pfc']) * (
            prox * (p['R_bar'] - I) - (1.0 - prox) * beta_eff
        )
        if beta_PFC <= 0.0 and dbetadt < 0.0:
            dbetadt = 0.0
        if beta_PFC >= self.beta_cap and dbetadt > 0.0:
            dbetadt = 0.0

        return [dVdt, dbetadt]

    # ── main loop ─────────────────────────────────────────────────────────────

    def simulate(self, t_span: tuple, rtol: float = 1e-6, atol: float = 1e-8,
                 record_frames: int = 0) -> AgentHistory:
        """
        Integrate the ODEs segment by segment, halting each segment at a threshold
        crossing (a spike). A spike harvests the dominant patch (world.engage_patch),
        resets V, and records the acquisition cost; between spikes the agent
        chemotaxis-wanders. The world clock and respawns are updated inline.

        record_frames > 0 captures that many evenly-spaced snapshots into
        AgentHistory.frames for the animation (off by default so landscape scoring
        pays nothing).
        """
        t_start, t_end = t_span
        p = self.p

        def spike_event(t, y):
            return y[0] - p['theta']

        spike_event.terminal  = True
        spike_event.direction = 1

        state = np.array([p['V_rest'], 0.0], dtype=float)  # [V, β_PFC]
        t_current = t_start

        total_distance = 0.0
        wander_distance = 0.0

        times, V_h, B_h = [], [], []
        x_h, y_h, r_h, I_h, ew_h = [], [], [], [], []
        fire_times = []
        harvested_h = []
        cost_h = []

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
            net_drive = self._total_I(t_now) - max(0.0, cur_state[1])
            frames.append({
                't': float(t_now), 'x': float(self.x), 'y': float(self.y),
                'V': float(cur_state[0]),
                'w': float(self._engagement_weight(net_drive)),
                'spiked': bool(spiked_now), 'patches': patch_snap,
            })

        n_spikes   = 0
        slow_ticks = 0

        while t_current < t_end:
            old_x, old_y = self.x, self.y
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
                times.append(sol.t[j])
                V_h.append(sol.y[0, j]); B_h.append(max(0.0, sol.y[1, j]))
                x_h.append(self.x);      y_h.append(self.y)
                r_h.append(dom.r_from(self.x, self.y))
                I_j = self._total_I(sol.t[j])
                I_h.append(I_j)
                ew_h.append(self._engagement_weight(I_j - max(0.0, sol.y[1, j])))

            state     = sol.y[:, -1].copy()
            t_new     = sol.t[-1]
            spiked    = sol.status == 1
            I_now     = self._total_I(t_new)
            net_now   = I_now - max(0.0, state[1])
            w_now     = self._engagement_weight(net_now)

            if spiked:
                n_spikes += 1
                fire_times.append(t_new)
                slow_ticks = 0

                dom = self._dominant_source(t_new)
                patch = getattr(dom, '_patch', None)
                harvested = self._harvest_patch(patch, t_new)
                harvested_h.append(harvested)
                cost_h.append(self._patch_cost(patch))

                state[0] = p['theta'] - self.refractory_drop  # reset V

                if n_spikes >= self.max_spikes:
                    break

            else:
                self._biased_wander_step(net_now, t_new)

                if w_now > 0.7:
                    slow_ticks += 1
                    if slow_ticks == STALL_WARN_TICKS:
                        import warnings
                        beta_now = max(0.0, state[1])
                        warnings.warn(
                            f"Agent engaged (w={w_now:.2f}) but not spiking at "
                            f"t={t_new:.1f}: β_PFC={beta_now:.2f} suppressing drive "
                            f"I={I_now:.2f}. If persistent, a supernormal husk trap "
                            "(intended) or a parameter mismatch (check R_bar / I_fire).",
                            RuntimeWarning,
                        )
                else:
                    slow_ticks = 0

            # Distance this iteration. Charge only wander steps to wander_distance.
            disp = float(np.linalg.norm([self.x - old_x, self.y - old_y]))
            total_distance += disp
            if not spiked:
                wander_distance += disp

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
            t=np.array(times),       V=np.array(V_h),        beta_PFC=np.array(B_h),
            r=np.array(r_h),         x=np.array(x_h),         y=np.array(y_h),
            fire_times=np.array(fire_times),
            I=np.array(I_h),         eng_weight=np.array(ew_h),
            harvested_utility=np.array(harvested_h),
            acquisition_cost=np.array(cost_h),
            total_distance=total_distance,
            wander_distance=wander_distance,
            frames=frames,
        )
