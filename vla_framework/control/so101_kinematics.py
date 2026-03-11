"""
SO-101 Kinematics
=================
Planar 2-link IK/FK for the SO-101 arm (shoulder_lift + elbow_flex),
extended to 3-D via shoulder_pan rotation.

  forward_kinematics(pan, lift, elbow) → (x, y, z)  [m]
    x = x2d * cos(shoulder_pan)
    y = x2d * sin(shoulder_pan)
    z = y2d  (vertical)

  inverse_kinematics_3d(x, y, z) → (pan_deg, lift_deg, elbow_deg)

Also provides create_real_robot() to construct a live SO-101 instance
using the XLeRobot / LeRobot library pattern.

Link lengths (from SO-101 URDF):
  l1 = 0.1159 m  (upper arm: shoulder_lift → elbow_flex)
  l2 = 0.1350 m  (lower arm: elbow_flex → wrist)
"""
from __future__ import annotations

import math
from typing import List, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Robot factory
# ---------------------------------------------------------------------------

def create_real_robot(port: str, camera_index: int = 0, uid: str = "so101"):
    """
    Construct a real SO-101 robot from a serial port and camera index.
    Uses SO101FollowerConfig + make_robot_from_config from LeRobot.

    Raises ImportError if the lerobot package is not installed.
    Raises ValueError for unknown uid values.
    """
    if uid != "so101":
        raise ValueError(f"Unknown robot uid: {uid!r}")

    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig   # type: ignore[import-not-found]
    from lerobot.robots.utils import make_robot_from_config                         # type: ignore[import-not-found]
    from lerobot.cameras import ColorMode, Cv2Rotation                              # type: ignore[import-not-found]
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig      # type: ignore[import-not-found]

    robot_config = SO101FollowerConfig(
        port       = port,
        use_degrees = True,
        cameras    = {
            "base_camera": OpenCVCameraConfig(
                index_or_path = camera_index,
                fps           = 30,
                width         = 640,
                height        = 480,
                color_mode    = ColorMode.RGB,
                rotation      = Cv2Rotation.NO_ROTATION,
            )
        },
        id = "robot1",
    )
    return make_robot_from_config(robot_config)


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

class SO101Kinematics:
    """
    Planar 2-link IK/FK for the SO-101 shoulder_lift + elbow_flex joints,
    extended to 3-D by incorporating shoulder_pan rotation.

    All public methods use degrees for joint angles.
    """

    def __init__(self, l1: float = 0.1159, l2: float = 0.1350) -> None:
        self.l1 = l1  # upper-arm length [m]
        self.l2 = l2  # lower-arm length [m]

    # ------------------------------------------------------------------
    # 2-D planar FK / IK  (shoulder_lift + elbow_flex only)
    # ------------------------------------------------------------------

    def forward_kinematics_2d(
        self,
        joint2_deg: float,
        joint3_deg: float,
        l1: float | None = None,
        l2: float | None = None,
    ) -> Tuple[float, float]:
        """
        2-link planar FK.

        Parameters
        ----------
        joint2_deg : shoulder_lift angle [deg]
        joint3_deg : elbow_flex angle [deg]

        Returns
        -------
        (x_radial, y_vertical) [m] — x is forward reach, y is height.
        """
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        theta1_offset = math.atan2(0.028, 0.11257)
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

        theta1 = math.radians(90.0 - joint2_deg) - theta1_offset
        theta2 = math.radians(joint3_deg + 90.0)  - theta2_offset

        x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2 - math.pi)
        y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2 - math.pi)
        return x, y

    def inverse_kinematics(
        self,
        x: float,
        y: float,
        l1: float | None = None,
        l2: float | None = None,
    ) -> Tuple[float, float]:
        """
        2-link planar IK.

        Parameters
        ----------
        x : radial reach [m] (forward from shoulder)
        y : vertical height [m]

        Returns
        -------
        (shoulder_lift_deg, elbow_flex_deg)
        """
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        theta1_offset = math.atan2(0.028, 0.11257)
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

        r     = math.sqrt(x**2 + y**2)
        r_max = l1 + l2
        if r > r_max:
            scale = r_max / r
            x, y, r = x * scale, y * scale, r_max

        r_min = abs(l1 - l2)
        if 0 < r < r_min:
            scale = r_min / r
            x, y  = x * scale, y * scale

        cos_t2 = max(-1.0, min(1.0, -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)))
        theta2 = math.pi - math.acos(cos_t2)
        beta   = math.atan2(y, x)
        gamma  = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
        theta1 = beta + gamma

        joint2 = max(-0.1, min(3.45,      theta1 + theta1_offset))
        joint3 = max(-0.2, min(math.pi,   theta2 + theta2_offset))

        return 90.0 - math.degrees(joint2), math.degrees(joint3) - 90.0

    # ------------------------------------------------------------------
    # 3-D FK / IK  (shoulder_pan + shoulder_lift + elbow_flex)
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        shoulder_pan_deg: float,
        joint2_deg:       float,
        joint3_deg:       float,
    ) -> Tuple[float, float, float]:
        """
        3-D FK incorporating shoulder_pan rotation.

        Returns
        -------
        (x, y, z) in robot base frame [m]:
          x = x2d * cos(shoulder_pan)
          y = x2d * sin(shoulder_pan)
          z = y2d  (vertical)
        """
        x2d, y2d = self.forward_kinematics_2d(joint2_deg, joint3_deg)
        pan_rad  = math.radians(shoulder_pan_deg)
        return (
            x2d * math.cos(pan_rad),
            x2d * math.sin(pan_rad),
            y2d,
        )

    def inverse_kinematics_3d(
        self,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[float, float, float]:
        """
        3-D IK.

        Returns
        -------
        (shoulder_pan_deg, shoulder_lift_deg, elbow_flex_deg)
        """
        pan_deg  = math.degrees(math.atan2(y, x))
        r_radial = math.sqrt(x**2 + y**2)
        lift_deg, elbow_deg = self.inverse_kinematics(r_radial, z)
        return pan_deg, lift_deg, elbow_deg

    # ------------------------------------------------------------------
    # Sinusoidal velocity trajectory
    # ------------------------------------------------------------------

    def generate_sinusoidal_velocity_trajectory(
        self,
        start_point:        Union[List[float], np.ndarray],
        end_point:          Union[List[float], np.ndarray],
        control_freq:       float = 100.0,
        total_time:         float = 5.0,
        velocity_amplitude: float = 1.0,
        velocity_period:    float = 2.0,
        phase_offset:       float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a straight-line trajectory with sinusoidal velocity profile.

        The base speed is set to cover the total distance in total_time; a
        sinusoidal perturbation of amplitude velocity_amplitude is added on top.
        Negative velocities are clamped to 10 % of base speed.

        Parameters
        ----------
        start_point, end_point : 3-D coordinates [m]
        control_freq  : Hz
        total_time    : seconds
        velocity_amplitude : m/s — sinusoidal amplitude
        velocity_period    : seconds — sinusoid period
        phase_offset       : radians

        Returns
        -------
        trajectory  : (n, 3)  position array [m]
        velocities  : (n,)    velocity magnitude array [m/s]
        time_array  : (n,)    time stamps [s]
        """
        start = np.array(start_point, dtype=float)
        end   = np.array(end_point,   dtype=float)

        direction  = end - start
        total_dist = float(np.linalg.norm(direction))

        dt    = 1.0 / control_freq
        n_pts = int(total_time * control_freq) + 1
        t_arr = np.linspace(0.0, total_time, n_pts)

        omega    = 2.0 * math.pi / velocity_period
        base_vel = total_dist / total_time if total_time > 0 else 0.0

        velocities = base_vel + velocity_amplitude * np.sin(omega * t_arr + phase_offset)
        velocities = np.maximum(velocities, 0.1 * base_vel)

        # Integrate velocity → 1-D arc-length position
        pos_1d = np.zeros(n_pts)
        for i in range(1, n_pts):
            pos_1d[i] = pos_1d[i - 1] + velocities[i - 1] * dt

        # Rescale so the last point hits the target exactly
        if pos_1d[-1] > 0:
            pos_1d *= total_dist / pos_1d[-1]

        # Map 1-D arc-length to 3-D positions
        trajectory = np.zeros((n_pts, 3))
        for i in range(n_pts):
            progress      = pos_1d[i] / total_dist if total_dist > 0 else 0.0
            trajectory[i] = start + progress * direction

        return trajectory, velocities, t_arr
