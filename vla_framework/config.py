"""
VLA Framework Configuration
All hardware parameters, gains, and geometric offsets live here.
Swap values per robot/camera setup; no code changes required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Action taxonomy
# ---------------------------------------------------------------------------

class ActionType(Enum):
    APPROACH  = "approach"   # Move above target at safety height
    PRE_GRASP = "pre_grasp"  # Descend to just above the object
    GRASP     = "grasp"      # Close gripper at object centroid
    LIFT      = "lift"       # Raise grasped object to lift height
    MOVE      = "move"       # Translate at safety height
    PLACE     = "place"      # Lower object to destination surface
    RETREAT   = "retreat"    # Rise clear of workspace after release
    HOME      = "home"       # Return to home pose


# ---------------------------------------------------------------------------
# Camera intrinsics  (pinhole model)
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    """Per-camera calibration — replace with values from camera_info topic."""
    fx: float = 615.3    # Focal length x  [px]
    fy: float = 615.3    # Focal length y  [px]
    cx: float = 320.0    # Principal point x [px]
    cy: float = 240.0    # Principal point y [px]
    width:  int = 640
    height: int = 480

    @property
    def K(self) -> np.ndarray:
        """3×3 camera matrix."""
        return np.array([
            [self.fx,       0, self.cx],
            [      0, self.fy, self.cy],
            [      0,       0,       1],
        ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Camera ↔ robot extrinsics
# ---------------------------------------------------------------------------

@dataclass
class CameraExtrinsics:
    """
    4×4 rigid-body transform T_cam→robot.
    Points expressed in the camera frame are transformed to the robot base
    frame by:  p_robot = T @ p_cam_homogeneous

    Default: camera ~55 cm above work surface, angled 30° downward,
    offset 25 cm in x and 30 cm in y from robot origin.
    Override with your actual hand-eye calibration result.
    """
    T: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [ 0.0, -1.0,  0.0,  0.25],
                [-1.0,  0.0,  0.0,  0.30],
                [ 0.0,  0.0, -1.0,  0.55],
                [ 0.0,  0.0,  0.0,  1.00],
            ],
            dtype=np.float64,
        )
    )


# ---------------------------------------------------------------------------
# PID gains  (identical structure reused for each Cartesian axis)
# ---------------------------------------------------------------------------

@dataclass
class PIDGains:
    kp: float = 2.0
    ki: float = 0.05
    kd: float = 0.20
    max_integral: float = 5.0
    output_min: float = -0.5   # [m/s] velocity clamp
    output_max: float =  0.5


# ---------------------------------------------------------------------------
# Geometric offsets appended to each action type
# ---------------------------------------------------------------------------

@dataclass
class ActionOffsets:
    """All values are metres, positive z = up in robot base frame."""
    safety_height:    float = 0.15   # z-lift during APPROACH / MOVE / RETREAT
    pre_grasp_height: float = 0.05   # z above object for PRE_GRASP
    grasp_descent:    float = 0.00   # extra z descent at GRASP (tune per gripper)
    lift_height:      float = 0.20   # z above grasp point after LIFT
    place_height:     float = 0.02   # z above surface at PLACE
    retreat_height:   float = 0.15   # z during RETREAT


# ---------------------------------------------------------------------------
# Top-level config bundle
# ---------------------------------------------------------------------------

@dataclass
class VLAConfig:
    camera_intrinsics:  CameraIntrinsics  = field(default_factory=CameraIntrinsics)
    camera_extrinsics:  CameraExtrinsics  = field(default_factory=CameraExtrinsics)
    pid_gains:          PIDGains          = field(default_factory=PIDGains)
    action_offsets:     ActionOffsets     = field(default_factory=ActionOffsets)

    # Gemini planner
    gemini_api_key:  str = ""
    # "gemini-robotics-er" when GA; use "gemini-2.5-flash" for dev/testing
    gemini_model:    str = "gemini-2.5-flash"

    # Trajectory
    interpolation_steps: int   = 50       # Points per waypoint segment
    waypoint_tolerance:  float = 0.005    # Convergence radius [m]

    # Robot
    robot_type:         str   = "so100"          # LeRobot robot identifier
    robot_port:         str   = "/dev/ttyUSB0"
    control_frequency:  float = 50.0             # Hz
    gripper_settle_s:   float = 0.40             # Wait after gripper command [s]
