"""
VLA Config Factory
==================
Assembles a VLAConfig from CLI parameters and hardware defaults.
Edit camera intrinsics, PID gains, and action offsets here to match your setup.

Camera: stereo UVC (rectified output 640×480)
Robot:  SO-101 6-DOF arm via LeRobot
"""
from __future__ import annotations

import numpy as np

from .config import (
    ActionOffsets,
    CameraExtrinsics,
    CameraIntrinsics,
    PIDGains,
    VLAConfig,
)


def build_config(
    api_key: str,
    model:   str,
    port:    str,
    no_mock: bool = False,
) -> VLAConfig:
    """
    Assemble a VLAConfig with hardware defaults.

    The extrinsic transform T_cam→robot is loaded automatically from
    calibration/camera_to_robot.npy if it exists (written by calibrate.py).
    Falls back to a hard-coded placeholder and emits a warning if absent.
    """
    intrinsics = CameraIntrinsics(
        fx=615.3, fy=615.3, cx=320.0, cy=240.0,
        width=640, height=480,
    )

    # CameraExtrinsics loads from calibration file automatically; see config.py.
    extrinsics = CameraExtrinsics()

    pid = PIDGains(
        kp=2.0, ki=0.05, kd=0.20,
        max_integral=5.0,
        output_min=-0.5, output_max=0.5,
    )

    offsets = ActionOffsets(
        safety_height    = 0.15,
        pre_grasp_height = 0.05,
        grasp_descent    = 0.00,
        lift_height      = 0.20,
        place_height     = 0.02,
        retreat_height   = 0.15,
    )

    return VLAConfig(
        camera_intrinsics   = intrinsics,
        camera_extrinsics   = extrinsics,
        pid_gains           = pid,
        action_offsets      = offsets,
        gemini_api_key      = api_key,
        gemini_model        = model,
        interpolation_steps = 50,
        waypoint_tolerance  = 0.005,
        robot_type          = "so101",
        robot_port          = port,
        robot_strict        = no_mock,
        control_frequency   = 50.0,
        gripper_settle_s    = 0.40,
    )
