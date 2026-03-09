"""
PID Controller  —  Stage 4 (inner loop)
=========================================
Discrete-time PID with:
  • Anti-windup     : integral clamped to [-max_integral, +max_integral]
  • Output clamp    : hard limits on control signal
  • Back-calculation: integrator held when output is saturated
  • Derivative      : computed from error signal (not measurement, to
                      avoid derivative kick on setpoint changes)

CartesianPIDController wraps three independent scalar PIDs for X, Y, Z
and produces a 3-D velocity (or position-delta) command vector.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import PIDGains

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scalar PID
# ---------------------------------------------------------------------------

class PIDController:
    """
    Single-axis discrete-time PID controller.

    Parameters
    ----------
    gains : PIDGains dataclass (kp, ki, kd, limits, …).
    dt    : Control period [seconds].
    """

    def __init__(self, gains: PIDGains, dt: float) -> None:
        self._kp   = gains.kp
        self._ki   = gains.ki
        self._kd   = gains.kd
        self._imax = gains.max_integral
        self._omin = gains.output_min
        self._omax = gains.output_max
        self._dt   = dt

        self._integral:   float = 0.0
        self._prev_error: float = 0.0
        self._ready:      bool  = False   # False until first step called

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0
        self._ready      = False

    def step(self, error: float) -> float:
        """
        Advance one control step.

        Parameters
        ----------
        error : Signed error = target − current.

        Returns
        -------
        Control output (clipped to [output_min, output_max]).
        """
        # Proportional
        p = self._kp * error

        # Derivative  (backward Euler; skip on first call to avoid kick)
        d = self._kd * (error - self._prev_error) / self._dt if self._ready else 0.0

        # Un-saturated output (before integral)
        raw = p + d

        # Integral with anti-windup: only accumulate if output is not saturated
        if self._omin < raw < self._omax:
            self._integral += error * self._dt
            self._integral  = float(np.clip(self._integral, -self._imax, self._imax))

        i   = self._ki * self._integral
        out = float(np.clip(raw + i, self._omin, self._omax))

        self._prev_error = error
        self._ready      = True
        return out

    # Convenience properties
    @property
    def integral(self) -> float:
        return self._integral

    @property
    def dt(self) -> float:
        return self._dt


# ---------------------------------------------------------------------------
# Cartesian (3-axis) PID
# ---------------------------------------------------------------------------

class CartesianPIDController:
    """
    Three independent PIDControllers for the X, Y, Z Cartesian axes.

    Produces a 3-D velocity command vector given a 3-D position error.
    """

    def __init__(self, gains: PIDGains, dt: float) -> None:
        self._pids = [PIDController(gains, dt) for _ in range(3)]
        self._dt   = dt

    def reset(self) -> None:
        for pid in self._pids:
            pid.reset()

    def step(self, position_error: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        position_error : shape (3,) array — (target - current) in metres.

        Returns
        -------
        Control signal  shape (3,) — Cartesian velocity [m/s].
        """
        assert position_error.shape == (3,), "Expected 3-D error vector"
        return np.array(
            [pid.step(float(e)) for pid, e in zip(self._pids, position_error)],
            dtype=np.float64,
        )

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def integrals(self) -> np.ndarray:
        return np.array([pid.integral for pid in self._pids])
