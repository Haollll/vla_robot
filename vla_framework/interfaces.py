"""
Hardware-Agnostic Interfaces
============================
Runtime-checkable Protocol definitions that decouple the pipeline from
concrete hardware implementations (LeRobot, MuJoCo, mock, …).

Protocols
---------
RobotStateProtocol   — snapshot of robot state returned by RobotInterface.get_state()
SnapshotResultProtocol — camera snapshot returned by CameraStreamerInterface.snapshot()
RobotInterface       — minimal robot API consumed by ExecuteStage
SimRobotInterface    — RobotInterface + step() for simulation backends
CameraStreamerInterface — live camera / stereo streamer consumed by VLAPipeline.run()

Type alias
----------
SimStepCallback = Callable[[], None]
    Called once per control tick in ExecuteStage._servo_to().
    Real hardware: time.sleep(dt)
    MuJoCo:        mj_step(model, data)  (injected by Phase-2 sim harness)
"""
from __future__ import annotations

from typing import Callable, Protocol, Tuple, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# SimStepCallback
# ---------------------------------------------------------------------------

SimStepCallback = Callable[[], None]
"""Called once per control tick inside ExecuteStage._servo_to().

Real robot:  ``lambda: time.sleep(dt)``
MuJoCo sim:  ``lambda: mujoco.mj_step(model, data)``
"""


# ---------------------------------------------------------------------------
# RobotStateProtocol
# ---------------------------------------------------------------------------

@runtime_checkable
class RobotStateProtocol(Protocol):
    """Minimal robot-state snapshot consumed by ExecuteStage."""

    @property
    def end_effector_pos(self) -> np.ndarray:
        """(3,) float64 EE position in robot base frame [m]."""
        ...

    @property
    def joint_angles(self) -> np.ndarray:
        """(N,) float64 joint angles [rad]."""
        ...


# ---------------------------------------------------------------------------
# SnapshotResultProtocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SnapshotResultProtocol(Protocol):
    """Camera snapshot returned by CameraStreamerInterface.snapshot()."""

    @property
    def rgb_snapshot(self) -> np.ndarray:
        """uint8 (H, W, 3) RGB image."""
        ...

    @property
    def stable_depth(self) -> np.ndarray:
        """float32 (H, W) depth map [m]."""
        ...


# ---------------------------------------------------------------------------
# RobotInterface
# ---------------------------------------------------------------------------

@runtime_checkable
class RobotInterface(Protocol):
    """Minimal robot API consumed by ExecuteStage.

    LeRobotInterface satisfies this structurally (no inheritance needed).
    """

    def connect(self) -> None:
        """Open connection to the robot."""
        ...

    def disconnect(self) -> None:
        """Close connection to the robot."""
        ...

    def get_state(self) -> RobotStateProtocol:
        """Return current robot state."""
        ...

    def send_cartesian_velocity(self, velocity: np.ndarray) -> None:
        """Send (3,) Cartesian velocity command [m/s]."""
        ...

    def set_gripper(self, value: float) -> None:
        """Set gripper openness: 0.0 = open, 1.0 = closed."""
        ...


# ---------------------------------------------------------------------------
# SimRobotInterface
# ---------------------------------------------------------------------------

@runtime_checkable
class SimRobotInterface(RobotInterface, Protocol):
    """RobotInterface extended with a physics step for sim backends."""

    def step(self) -> None:
        """Advance the simulation by one physics step."""
        ...


# ---------------------------------------------------------------------------
# CameraStreamerInterface
# ---------------------------------------------------------------------------

@runtime_checkable
class CameraStreamerInterface(Protocol):
    """Live camera / stereo streamer consumed by VLAPipeline.run()."""

    def snapshot(self) -> SnapshotResultProtocol:
        """Return a synchronised (RGB, depth) snapshot.

        Raises NotReadyError (or similar) if the buffer is not full yet.
        """
        ...


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "SimStepCallback",
    "RobotStateProtocol",
    "SnapshotResultProtocol",
    "RobotInterface",
    "SimRobotInterface",
    "CameraStreamerInterface",
]
