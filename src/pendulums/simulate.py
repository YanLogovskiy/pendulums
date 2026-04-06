from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from .physics import DoubleParams, ModelName, Params, SingleParams, derivatives, energy, validate_initial_state


@dataclass(frozen=True, slots=True)
class SimulationResult:
    t: np.ndarray  # (N,)
    y: np.ndarray  # (state_dim, N)
    energies: np.ndarray  # (N,)
    energy_drift_abs_max: float
    energy_drift_rel_max: float


def simulate(
    *,
    model: ModelName,
    params: Params,
    y0: np.ndarray,
    duration: float,
    fps: float = 60.0,
    method: str = "DOP853",
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: Optional[float] = None,
) -> SimulationResult:
    if duration <= 0:
        raise ValueError("duration must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    if model == "single":
        if not isinstance(params, SingleParams):
            raise TypeError("params must be SingleParams for model='single'")
        params.validate()
    elif model == "double":
        if not isinstance(params, DoubleParams):
            raise TypeError("params must be DoubleParams for model='double'")
        params.validate()
    else:
        raise ValueError(f"Unknown model: {model}")

    y0 = validate_initial_state(model, y0)

    n_frames = int(np.floor(duration * fps)) + 1
    t_eval = np.linspace(0.0, float(duration), n_frames, dtype=float)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return derivatives(model, t, y, params)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, float(duration)),
        y0=y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=np.inf if max_step is None else float(max_step),
        vectorized=False,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    y = np.asarray(sol.y, dtype=float)
    t = np.asarray(sol.t, dtype=float)

    energies = np.array([energy(model, y[:, i], params) for i in range(y.shape[1])], dtype=float)
    e0 = float(energies[0])
    drift_abs = np.max(np.abs(energies - e0))
    drift_rel = drift_abs / (abs(e0) + 1e-12)

    return SimulationResult(
        t=t,
        y=y,
        energies=energies,
        energy_drift_abs_max=float(drift_abs),
        energy_drift_rel_max=float(drift_rel),
    )

