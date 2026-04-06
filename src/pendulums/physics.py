from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np


ModelName = Literal["single", "double"]


@dataclass(frozen=True, slots=True)
class SingleParams:
    m: float = 1.0
    l: float = 1.0
    g: float = 9.81

    def validate(self) -> None:
        if not (self.m > 0 and self.l > 0 and self.g > 0):
            raise ValueError("SingleParams requires m>0, l>0, g>0")


@dataclass(frozen=True, slots=True)
class DoubleParams:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81

    def validate(self) -> None:
        if not (self.m1 > 0 and self.m2 > 0 and self.l1 > 0 and self.l2 > 0 and self.g > 0):
            raise ValueError("DoubleParams requires m1,m2,l1,l2,g > 0")


Params = Union[SingleParams, DoubleParams]


def derivatives(model: ModelName, t: float, y: np.ndarray, params: Params) -> np.ndarray:
    """
    Right-hand side for ODE y' = f(t, y).

    y for single: [theta, omega]
    y for double: [theta1, omega1, theta2, omega2]
    """
    if model == "single":
        assert isinstance(params, SingleParams)
        theta, omega = y
        dtheta = omega
        domega = -(params.g / params.l) * np.sin(theta)
        return np.array([dtheta, domega], dtype=float)

    if model == "double":
        assert isinstance(params, DoubleParams)
        theta1, omega1, theta2, omega2 = y
        m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g

        delta = theta1 - theta2
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)

        denom1 = l1 * (2 * m1 + m2 - m2 * np.cos(2 * delta))
        denom2 = l2 * (2 * m1 + m2 - m2 * np.cos(2 * delta))

        dtheta1 = omega1
        dtheta2 = omega2

        # Standard double pendulum equations (point masses, massless rods)
        domega1 = (
            -g * (2 * m1 + m2) * np.sin(theta1)
            - m2 * g * np.sin(theta1 - 2 * theta2)
            - 2 * sin_delta * m2 * (omega2**2 * l2 + omega1**2 * l1 * cos_delta)
        ) / denom1

        domega2 = (
            2
            * sin_delta
            * (
                omega1**2 * l1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(theta1)
                + omega2**2 * l2 * m2 * cos_delta
            )
        ) / denom2

        return np.array([dtheta1, domega1, dtheta2, domega2], dtype=float)

    raise ValueError(f"Unknown model: {model}")


def positions(model: ModelName, y: np.ndarray, params: Params) -> np.ndarray:
    """
    Return positions in world coordinates.

    single: [[x1, y1]]
    double: [[x1, y1], [x2, y2]]
    """
    if model == "single":
        assert isinstance(params, SingleParams)
        theta = float(y[0])
        x1 = params.l * np.sin(theta)
        y1 = -params.l * np.cos(theta)
        return np.array([[x1, y1]], dtype=float)

    if model == "double":
        assert isinstance(params, DoubleParams)
        theta1, _, theta2, _ = y
        x1 = params.l1 * np.sin(theta1)
        y1 = -params.l1 * np.cos(theta1)
        x2 = x1 + params.l2 * np.sin(theta2)
        y2 = y1 - params.l2 * np.cos(theta2)
        return np.array([[x1, y1], [x2, y2]], dtype=float)

    raise ValueError(f"Unknown model: {model}")


def energy(model: ModelName, y: np.ndarray, params: Params) -> float:
    """
    Mechanical energy T+V (up to an arbitrary constant).
    Used for drift monitoring; absolute zero level is irrelevant.
    """
    if model == "single":
        assert isinstance(params, SingleParams)
        theta, omega = y
        v2 = (params.l * omega) ** 2
        T = 0.5 * params.m * v2
        V = params.m * params.g * params.l * (1 - np.cos(theta))
        return float(T + V)

    if model == "double":
        assert isinstance(params, DoubleParams)
        theta1, omega1, theta2, omega2 = y
        m1, m2, l1, l2, g = params.m1, params.m2, params.l1, params.l2, params.g

        # Positions (y positive up): y1 = -l1 cos(theta1), y2 = y1 - l2 cos(theta2)
        y1 = -l1 * np.cos(theta1)
        y2 = y1 - l2 * np.cos(theta2)

        # Velocities squared for masses
        v1_sq = (l1 * omega1) ** 2
        v2_sq = v1_sq + (l2 * omega2) ** 2 + 2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)

        T = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq
        V = m1 * g * y1 + m2 * g * y2
        return float(T + V)

    raise ValueError(f"Unknown model: {model}")


def validate_initial_state(model: ModelName, y0: np.ndarray) -> np.ndarray:
    y0 = np.asarray(y0, dtype=float)
    expected = 2 if model == "single" else 4
    if y0.shape != (expected,):
        raise ValueError(f"y0 must have shape ({expected},) for model={model}")
    if not np.all(np.isfinite(y0)):
        raise ValueError("y0 must be finite")
    return y0

