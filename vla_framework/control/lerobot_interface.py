"""
LeRobot Interface  —  Stage 4 (hardware bridge)
================================================
Wraps the Hugging Face LeRobot library for the SO-101 follower arm.

Hardware
--------
  Robot   : SO-101 6-DOF manipulator (6 × STS3215 Feetech servos)
  Port    : /dev/ttyACM0  (Linux)
  Library : lerobot (XLeRobot / Hugging Face)

Joint order (SO-101 follower)
------------------------------
  [0] shoulder_pan   (motor 1)  — base rotation
  [1] shoulder_lift  (motor 2)  — shoulder pitch
  [2] elbow_flex     (motor 3)  — elbow pitch
  [3] wrist_flex     (motor 4)  — wrist pitch
  [4] wrist_roll     (motor 5)  — wrist rotation
  [5] gripper        (motor 6)  — 0–100 % normalised range

Forward / Inverse kinematics
------------------------------
  Provided by SO101Kinematics (vla_framework/control/so101_kinematics.py).
  FK covers shoulder_pan + shoulder_lift + elbow_flex → (x, y, z).
  Wrist joints are preserved but not solved by IK.

Cartesian velocity → joint position
-------------------------------------
  1. FK from current joint state → current EE position.
  2. Integrate velocity × dt → target EE position.
  3. IK on target → new (shoulder_pan, shoulder_lift, elbow_flex) in degrees.
  4. Send position command; wrist joints unchanged.

Mock mode
---------
  If lerobot is not installed or the port is unavailable the interface
  falls back to a stateful mock — unless strict=True, which raises
  RuntimeError immediately.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .so101_kinematics import SO101Kinematics, create_real_robot

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Joint names — SO-101 order matches motor IDs 1–6
# ---------------------------------------------------------------------------

_JOINT_NAMES = [
    "shoulder_pan",   # motor 1
    "shoulder_lift",  # motor 2
    "elbow_flex",     # motor 3
    "wrist_flex",     # motor 4
    "wrist_roll",     # motor 5
    "gripper",        # motor 6  (0–100 normalised, not radians)
]

_N_ARM  = 5   # revolute arm joints (all except gripper)
_STEPS  = 4096  # STS3215 encoder steps per revolution

# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

_DEFAULT_CALIB_PATH = (
    Path.home()
    / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json"
)


def load_calibration(path: Path = _DEFAULT_CALIB_PATH) -> Optional[Dict[str, Any]]:
    """
    Load a LeRobot calibration JSON file.
    Returns the parsed dict on success, None if the file does not exist.
    """
    if not path.exists():
        log.warning("Calibration file not found: %s", path)
        return None
    with path.open() as f:
        data = json.load(f)
    log.info("Loaded calibration from %s", path)
    return data


def encoder_to_rad(raw: int, calib: Dict[str, Any]) -> float:
    """
    Convert a raw STS3215 encoder value (0–4095) to joint angle in radians.

    1. Centre:  adjusted = raw − homing_offset
    2. Flip:    if drive_mode == 1  →  adjusted = −adjusted
    3. Scale:   angle_rad = adjusted × (2π / 4096)
    """
    adjusted = int(raw) - int(calib["homing_offset"])
    if int(calib.get("drive_mode", 0)) == 1:
        adjusted = -adjusted
    return adjusted * (2.0 * np.pi / _STEPS)


def encoder_to_gripper(raw: int, calib: Dict[str, Any]) -> float:
    """Map a raw gripper encoder to a 0.0 (open) – 1.0 (closed) fraction."""
    lo = int(calib["range_min"])
    hi = int(calib["range_max"])
    return float(np.clip((raw - lo) / max(1, hi - lo), 0.0, 1.0))


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

@dataclass
class RobotState:
    joint_positions_rad: np.ndarray   # (6,) — five arm joints [rad] + gripper [0–1]
    end_effector_pos:    np.ndarray   # (3,) — EE xyz in robot base frame [m]
    gripper:             float        # 0.0 = open, 1.0 = closed
    timestamp:           float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class LeRobotInterface:
    """
    Hardware bridge for the SO-101 follower arm via LeRobot.

    Automatically falls back to a stateful mock if the ``lerobot`` package
    is not installed or the port is unavailable — unless ``strict=True``.

    Parameters
    ----------
    robot_type   : LeRobot robot type string (currently only ``"so101"``).
    port         : Serial port (default ``"/dev/ttyACM0"``).
    camera_index : Camera device index for the onboard camera (default 0).
    calib_path   : Path to calibration JSON; uses LeRobot default if omitted.
    robot_id     : Robot ID string matched to the calibration file.
    strict       : If True, raise RuntimeError instead of falling back to mock.
    """

    def __init__(
        self,
        robot_type:   str           = "so101",
        port:         str           = "/dev/ttyACM0",
        camera_index: int           = 0,
        calib_path:   Optional[Path] = None,
        robot_id:     str           = "my_follower",
        strict:       bool          = False,
    ) -> None:
        self._robot_type   = robot_type
        self._port         = port
        self._camera_index = camera_index
        self._robot_id     = robot_id
        self._strict       = strict
        self._robot: Optional[Any] = None
        self._connected    = False

        # Kinematics helper (IK + FK + sinusoidal trajectory)
        self._kin = SO101Kinematics()

        # Tracked joint state: indices 0–4 are arm joints [rad]; 5 is gripper [0–1]
        self._q = np.zeros(6, dtype=np.float64)

        _calib_path    = calib_path or _DEFAULT_CALIB_PATH
        self._calib    = load_calibration(_calib_path)

        log.info(
            "LeRobotInterface created  type=%s  port=%s  cam=%d  strict=%s  calib=%s",
            robot_type, port, camera_index, strict,
            "loaded" if self._calib else "not found (using defaults)",
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        try:
            self._robot = create_real_robot(self._port, self._camera_index)
            self._robot.connect(calibrate=False)
            self._connected = True
            log.info("Connected to SO-101 on %s  cam=%d", self._port, self._camera_index)
        except ImportError as exc:
            msg = f"lerobot package not found — install with: pip install lerobot  ({exc})"
            if self._strict:
                raise RuntimeError(msg) from exc
            log.warning("%s — running in MOCK mode.", msg)
        except Exception as exc:
            msg = f"Robot connection failed: {exc}"
            if self._strict:
                raise RuntimeError(msg) from exc
            log.warning("%s — running in MOCK mode.", msg)

    def disconnect(self) -> None:
        if self._robot and self._connected:
            try:
                self._robot.disconnect()
            except Exception as exc:
                log.debug("Disconnect error (ignored): %s", exc)
        self._connected = False
        log.info("Robot disconnected")

    def __enter__(self) -> "LeRobotInterface":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # State reading
    # ------------------------------------------------------------------

    def get_state(self) -> RobotState:
        if self._connected and self._robot is not None:
            return self._read_hardware_state()
        return self._read_mock_state()

    def _read_hardware_state(self) -> RobotState:
        """
        Read hardware state via robot.get_observation().

        The LeRobot API returns:
          { "shoulder_pan.pos": float_degrees, …, "gripper.pos": float_0_100 }
        Arm joints are in degrees; gripper is in 0–100.
        """
        obs = self._robot.get_observation()

        q_rad = np.zeros(6, dtype=np.float64)
        for i, name in enumerate(_JOINT_NAMES[:_N_ARM]):
            q_rad[i] = math.radians(float(obs.get(f"{name}.pos", 0.0)))

        # Gripper: 0–100 → 0.0–1.0
        q_rad[5] = float(np.clip(obs.get("gripper.pos", 0.0) / 100.0, 0.0, 1.0))
        self._q  = q_rad.copy()

        ee = np.array(self._kin.forward_kinematics(
            math.degrees(q_rad[0]),   # shoulder_pan
            math.degrees(q_rad[1]),   # shoulder_lift
            math.degrees(q_rad[2]),   # elbow_flex
        ))
        return RobotState(
            joint_positions_rad = q_rad,
            end_effector_pos    = ee,
            gripper             = float(q_rad[5]),
            timestamp           = time.monotonic(),
        )

    def _read_mock_state(self) -> RobotState:
        q  = self._q.copy()
        ee = np.array(self._kin.forward_kinematics(
            math.degrees(q[0]),
            math.degrees(q[1]),
            math.degrees(q[2]),
        ))
        return RobotState(
            joint_positions_rad = q,
            end_effector_pos    = ee,
            gripper             = float(np.clip(q[5], 0.0, 1.0)),
            timestamp           = time.monotonic(),
        )

    # ------------------------------------------------------------------
    # Command sending
    # ------------------------------------------------------------------

    def send_cartesian_velocity(self, velocity_mps: np.ndarray) -> None:
        """
        Send a Cartesian-space velocity command [m/s].

        Converts velocity to a joint position command via IK:
          1. FK(current joints) → current EE position
          2. target = current + velocity × dt
          3. IK(target) → new (shoulder_pan, shoulder_lift, elbow_flex) [deg]
          4. Send position command; wrist joints unchanged.
        """
        dt = 1.0 / 50.0  # 50 Hz control loop

        # Current EE via kinematic FK
        x, y, z = self._kin.forward_kinematics(
            math.degrees(self._q[0]),
            math.degrees(self._q[1]),
            math.degrees(self._q[2]),
        )

        # Step EE by velocity
        tx = x + float(velocity_mps[0]) * dt
        ty = y + float(velocity_mps[1]) * dt
        tz = z + float(velocity_mps[2]) * dt

        # IK to get new joint angles for the three major joints
        pan_deg, lift_deg, elbow_deg = self._kin.inverse_kinematics_3d(tx, ty, tz)
        self._q[0] = math.radians(pan_deg)
        self._q[1] = math.radians(lift_deg)
        self._q[2] = math.radians(elbow_deg)
        # wrist_flex [3] and wrist_roll [4] are preserved unchanged

        if self._connected and self._robot is not None:
            action = {
                f"{name}.pos": float(math.degrees(self._q[i]))
                for i, name in enumerate(_JOINT_NAMES[:_N_ARM])
            }
            action["gripper.pos"] = float(np.clip(self._q[5], 0.0, 1.0) * 100.0)
            self._robot.send_action(action)
        else:
            ee_new = self._kin.forward_kinematics(
                math.degrees(self._q[0]),
                math.degrees(self._q[1]),
                math.degrees(self._q[2]),
            )
            log.debug("[MOCK] vel=%s → ee=%s", velocity_mps.round(4),
                      np.array(ee_new).round(4))

    def set_gripper(self, state: float) -> None:
        """
        Open or close the gripper.  state: 0.0 = fully open, 1.0 = fully closed.
        """
        state      = float(np.clip(state, 0.0, 1.0))
        self._q[5] = state

        if self._connected and self._robot is not None:
            self._robot.send_action({"gripper.pos": state * 100.0})
        else:
            log.debug("[MOCK] gripper → %.1f", state)

    # ------------------------------------------------------------------
    # Calibration utilities (public — useful for diagnostics)
    # ------------------------------------------------------------------

    def raw_to_rad(self, motor: str, raw: int) -> Optional[float]:
        """Convert raw encoder (0–4095) to radians using the loaded calibration."""
        if self._calib is None or motor not in self._calib:
            return None
        return encoder_to_rad(raw, self._calib[motor])

    def raw_to_gripper(self, raw: int) -> Optional[float]:
        """Convert raw gripper encoder to 0–1 fraction using the loaded calibration."""
        if self._calib is None or "gripper" not in self._calib:
            return None
        return encoder_to_gripper(raw, self._calib["gripper"])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def calibration(self) -> Optional[Dict[str, Any]]:
        """The loaded calibration dict, or None if not available."""
        return self._calib
