import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class AgentHistory:
    """Time-series data produced by the simulation loop."""
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
        params: dict,
        genome: Optional[dict] = None,
        x0: float = -200.0,
        y0: float = 0.0,
        source_x: float = 0.0,
        source_y: float = 0.0,
        step_size: float = 5.0,
        stimulus_func: Optional[Callable] = None,
    ):
        self.p = dict(params)
        if genome is not None:
            self.p.update(genome)

        # Standardizing keys to prevent KeyErrors
        if 'tau_pfc' not in self.p:     self.p['tau_pfc'] = 50.0
        if 'beta_pfc_max' not in self.p: self.p['beta_pfc_max'] = self.p.get('beta_pfc', 0.5)

        self.x = float(x0)
        self.y = float(y0)
        self.source_x = float(source_x)
        self.source_y = float(source_y)
        self.step_size = float(step_size)
        self.stimulus_func = stimulus_func

    def _r(self, x: float, y: float) -> float:
        return np.sqrt((x - self.source_x)**2 + (y - self.source_y)**2)

    def _gradient_direction(self, x: float, y: float) -> np.ndarray:
        dx = self.source_x - x
        dy = self.source_y - y
        dist = np.sqrt(dx**2 + dy**2)
        return np.array([dx / dist, dy / dist]) if dist > 1e-9 else np.array([0.0, 0.0])

    def _stimulus(self, r: float, t: float) -> float:
        if self.stimulus_func is not None:
            return float(self.stimulus_func(r, t, self.p))
        envelope = self.p['A'] * np.exp(-((r**2) / (2 * self.p['sigma']**2))**(self.p['n'] / 2))
        ripple   = 1 + self.p['alpha'] * np.sin(self.p['omega'] * t - self.p['k'] * r)
        return float(envelope * ripple)

    def _odes(self, t: float, y: np.ndarray, r: float) -> list:
        S, V, E_R, beta_PFC = y
        I = self._stimulus(r, t)
        
        relu = lambda x: np.maximum(0.0, x)

        dSdt = (1 / self.p['tau_s']) * (self.p['gamma_s'] * (self.p['S_rest'] - S) - self.p['alpha_s'] * relu(I - E_R))
        dVdt = (1 / self.p['tau_v']) * (-(V - self.p['V_rest']) + (I * S) - beta_PFC + self.p['beta_bg'])
        dE_Rdt = (1 / self.p['tau_r']) * (I - E_R)
        
        # Fixed equation to match fatigue dynamics properly
        dbeta_PFCdt = (1 / self.p['tau_pfc']) * (
            self.p['gamma_pfc'] * (self.p['beta_pfc_max'] - beta_PFC) 
            - self.p['alpha_pfc'] * relu(V - self.p['theta'])
        )

        return [dSdt, dVdt, dE_Rdt, dbeta_PFCdt]

    def simulate(self, t_span: tuple, rtol: float = 1e-5, atol: float = 1e-7) -> AgentHistory:
        """Runs continuous simulation, interrupting instantly on exact spike events."""
        t_start, t_end = t_span
        
        # Setup event tracking for the spiking threshold
        def spike_event(t, y, r):
            return y[1] - self.p['theta']  # V - theta = 0
        spike_event.terminal = True        # Force SciPy to stop integrating when hit
        spike_event.direction = 1          # Only trigger on rising cross

        # Initial state setup
        state = np.array([self.p['S_rest'], self.p['V_rest'], 0.0, self.p['beta_pfc_max']], dtype=float)
        t_current = t_start
        
        # Recording bins
        times, S_h, V_h, E_h, B_h, x_h, y_h, r_h, I_h = [], [], [], [], [], [], [], [], []
        fire_times = []

        while t_current < t_end:
            r_now = self._r(self.x, self.y)
            
            sol = solve_ivp(
                fun=lambda t, y: self._odes(t, y, r_now),
                t_span=(t_current, t_end),
                y0=state,
                method='RK45',
                events=lambda t, y: spike_event(t, y, r_now),
                rtol=rtol, atol=atol
            )

            # Store continuous trajectory points found by solver
            for j in range(len(sol.t)):
                times.append(sol.t[j])
                S_h.append(sol.y[0, j]); V_h.append(sol.y[1, j])
                E_h.append(sol.y[2, j]); B_h.append(sol.y[3, j])
                x_h.append(self.x); y_h.append(self.y); r_h.append(r_now)
                I_h.append(self._stimulus(r_now, sol.t[j]))

            state = sol.y[:, -1]
            t_current = sol.t[-1]

            # If the solver halted because it hit the event threshold:
            if sol.status == 1: 
                fire_times.append(t_current)
                
                # Discrete step up the gradient peak
                dir_vec = self._gradient_direction(self.x, self.y)
                self.x += self.step_size * dir_vec[0]
                self.y += self.step_size * dir_vec[1]
                
                # Soft reset: reset membrane voltage directly down to the threshold
                state[1] = self.p['V_rest']  # or use p['theta'] for soft clamping

        return AgentHistory(
            t=np.array(times), S=np.array(S_h), V=np.array(V_h), E_R=np.array(E_h),
            beta_PFC=np.array(B_h), r=np.array(r_h), x=np.array(x_h), y=np.array(y_h),
            fire_times=np.array(fire_times), I=np.array(I_h)
        )