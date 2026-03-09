"""
LeRobot Interface  —  Stage 4 (hardware bridge)
================================================
Wraps the Hugging Face LeRobot library for the SO-101 follower arm.

Hardware
--------
  Robot   : SO-101 6-DOF manipulator (6 × STS3215 Feetech servos)
  Port    : /dev/ttyACM0  (Linux)
  Library : lerobot >= 0.1  (new unified API — lerobot.robots.so_follower)

Joint order (SO-101 follower)
------------------------------
  [0] shoulder_pan   (motor 1)  — base rotation
  [1] shoulder_lift  (motor 2)  — shoulder pitch
  [2] elbow_flex     (motor 3)  — elbow pitch
  [3] wrist_flex     (motor 4)  — wrist pitch
  [4] wrist_roll     (motor 5)  — wrist rotation
  [5] gripper        (motor 6)  — 0–100 % normalised range

Calibration
-----------
  Loaded from:
    ~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json
  JSON schema per motor:
    { "id": int, "drive_mode": int, "homing_offset": int,
      "range_min": int, "range_max": int }
  Raw encoder → radians (STS3215 = 4096 steps / revolution):
    adjusted = raw − homing_offset
    if drive_mode == 1:  adjusted = −adjusted
    angle_rad = adjusted × (2π / 4096)
  Gripper → 0–1:
    (raw − range_min) / (range_max − range_min)

Forward kinematics
------------------
  URDF-derived homogeneous transform chain from so101_new_calib.urdf
  (TheRobotStudio/SO-ARM100 repo).  Each joint:
    T_total = … × T_fixed(origin xyz, rpy) × Rz(θ_joint) × …
  The fixed gripper-frame offset is appended as the EE origin.

Cartesian velocity → joint position (hardware mode)
----------------------------------------------------
  A numerical Jacobian ∂EE/∂q (5×3) is computed from the FK.
  Joint angle deltas are obtained via the Moore–Penrose pseudo-inverse:
    Δq = J⁺ · v_cmd · dt
  Updated joint angles are sent as position commands to the robot.

Mock mode
---------
  If lerobot is not installed or the port is unavailable the interface
  falls back to a stateful mock, returning simulated state and discarding
  commands.  The full pipeline can run on any machine for testing.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

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

_N_ARM    = 5   # revolute arm joints
_STEPS    = 4096  # STS3215 encoder steps per revolution

# ---------------------------------------------------------------------------
# SO-101 forward kinematics — URDF-derived transform chain
# ---------------------------------------------------------------------------
# Joint origins extracted from so101_new_calib.urdf (TheRobotStudio/SO-ARM100).
# Each entry is (xyz [m], rpy [rad]) of the joint origin in its parent frame.
# All joint axes are z.  RPY follows URDF convention: R = Rz(yaw)·Ry(pitch)·Rx(roll).

_SO101_JOINT_ORIGINS = [
    # shoulder_pan  — base_link → shoulder_link
    ([0.0388353,   0.0,        0.0624   ], [np.pi,      0.0,          -np.pi    ]),
    # shoulder_lift — shoulder_link → upper_arm_link
    ([-0.0303992, -0.0182778, -0.0542   ], [-np.pi/2,  -np.pi/2,      0.0       ]),
    # elbow_flex    — upper_arm_link → lower_arm_link
    ([-0.11257,   -0.028,      0.0      ], [0.0,        0.0,           np.pi/2   ]),
    # wrist_flex    — lower_arm_link → wrist_link
    ([-0.1349,     0.0052,     0.0      ], [0.0,        0.0,          -np.pi/2   ]),
    # wrist_roll    — wrist_link → gripper_link
    ([0.0,        -0.0611,     0.0181   ], [np.pi/2,    0.0486795,     np.pi     ]),
]

# Fixed transform: gripper_link → gripper_frame_link (EE origin, fixed joint)
_SO101_EE_XYZ = [-0.0079, -0.000218121, -0.0981274]
_SO101_EE_RPY = [0.0, np.pi, 0.0]


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF RPY → 3×3 rotation matrix: R = Rz(yaw) · Ry(pitch) · Rx(roll)."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ], dtype=np.float64)


def _fixed_tf(xyz: list, rpy: list) -> np.ndarray:
    """4×4 homogeneous transform from a URDF joint origin (xyz, rpy)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rpy_matrix(*rpy)
    T[:3, 3]  = xyz
    return T


def _rotz(theta: float) -> np.ndarray:
    """4×4 rotation about the Z axis by theta radians."""
    T       = np.eye(4, dtype=np.float64)
    c, s    = np.cos(theta), np.sin(theta)
    T[0, 0] = c;  T[0, 1] = -s
    T[1, 0] = s;  T[1, 1] =  c
    return T


# Pre-compute the five fixed joint-origin transforms and the EE fixed transform
_TF_ORIGINS = [_fixed_tf(xyz, rpy) for xyz, rpy in _SO101_JOINT_ORIGINS]
_TF_EE      = _fixed_tf(_SO101_EE_XYZ, _SO101_EE_RPY)


def so101_fk(joint_angles_rad: np.ndarray) -> np.ndarray:
    """
    Forward kinematics for the SO-101 follower arm.

    Parameters
    ----------
    joint_angles_rad : shape (≥5,) — [shoulder_pan, shoulder_lift,
                       elbow_flex, wrist_flex, wrist_roll] in radians.
                       Index 5 (gripper) is ignored.

    Returns
    -------
    xyz : shape (3,) — end-effector position in robot base frame [m].
    """
    T = np.eye(4, dtype=np.float64)
    for i, T_origin in enumerate(_TF_ORIGINS):
        T = T @ T_origin @ _rotz(float(joint_angles_rad[i]))
    T = T @ _TF_EE
    return T[:3, 3]


def _numerical_jacobian(q_rad: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Numerical Jacobian ∂EE/∂q  shape (3, 5) using central differences.

    Parameters
    ----------
    q_rad : shape (≥5,) — arm joint angles in radians.
    eps   : Finite-difference step size [rad].
    """
    J = np.zeros((3, _N_ARM), dtype=np.float64)
    for i in range(_N_ARM):
        qp = q_rad.copy(); qp[i] += eps
        qm = q_rad.copy(); qm[i] -= eps
        J[:, i] = (so101_fk(qp) - so101_fk(qm)) / (2.0 * eps)
    return J

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
    Expected schema per motor key:
      { "id": int, "drive_mode": int, "homing_offset": int,
        "range_min": int, "range_max": int }
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

    Steps
    -----
    1. Centre:  adjusted = raw − homing_offset
    2. Flip:    if drive_mode == 1  →  adjusted = −adjusted
    3. Scale:   angle_rad = adjusted × (2π / 4096)
    """
    adjusted = int(raw) - int(calib["homing_offset"])
    if int(calib.get("drive_mode", 0)) == 1:
        adjusted = -adjusted
    return adjusted * (2.0 * np.pi / _STEPS)


def encoder_to_gripper(raw: int, calib: Dict[str, Any]) -> float:
    """
    Map a raw gripper encoder value to a 0.0 (open) – 1.0 (closed) fraction
    using the calibrated range_min / range_max.
    """
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
    is not installed or the port is unavailable.

    Parameters
    ----------
    robot_type     : LeRobot robot type string (default ``"so101"``).
    port           : Serial port (default ``"/dev/ttyACM0"``).
    calib_path     : Path to calibration JSON; uses the LeRobot default
                     location if not specified.
    robot_id       : Robot ID string used by LeRobot (matched to calib file).
    """

    def __init__(
        self,
        robot_type: str  = "so101",
        port:       str  = "/dev/ttyACM0",
        calib_path: Optional[Path] = None,
        robot_id:   str  = "my_follower",
    ) -> None:
        self._robot_type = robot_type
        self._port       = port
        self._robot_id   = robot_id
        self._robot: Optional[Any] = None
        self._connected  = False

        # Tracked joint state used for Jacobian IK (hardware) and mock sim
        # Indices 0–4: arm joint angles [rad]; index 5: gripper 0–1
        self._q = np.zeros(6, dtype=np.float64)

        # Load calibration
        _calib_path = calib_path or _DEFAULT_CALIB_PATH
        self._calib = load_calibration(_calib_path)

        log.info(
            "LeRobotInterface created  type=%s  port=%s  calib=%s",
            robot_type, port,
            "loaded" if self._calib else "not found (using defaults)",
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # type: ignore[import-not-found]

            cfg         = SO101FollowerConfig(port=self._port, id=self._robot_id)
            self._robot = SO101Follower(cfg)
            # calibrate=False: we rely on the calibration file being already
            # written to the motors (via lerobot-calibrate).
            self._robot.connect(calibrate=False)
            self._connected = True
            log.info("Connected to SO-101 on %s (id=%s)", self._port, self._robot_id)
        except ImportError:
            log.warning(
                "lerobot package not found — running in MOCK mode.  "
                "Install with:  pip install lerobot"
            )
        except Exception as exc:
            log.warning("Robot connection failed (%s) — running in MOCK mode.", exc)

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
        Read the hardware state via SO101Follower.get_observation().

        The new LeRobot API returns:
          { "shoulder_pan.pos": float_degrees, ..., "gripper.pos": float_0_100 }
        Arm joints are in degrees (DEGREES norm mode); gripper is in 0–100
        (RANGE_0_100 norm mode).
        """
        obs = self._robot.get_observation()

        q_rad = np.zeros(6, dtype=np.float64)
        for i, name in enumerate(_JOINT_NAMES[:_N_ARM]):
            deg = float(obs.get(f"{name}.pos", 0.0))
            q_rad[i] = np.deg2rad(deg)

        # Gripper: 0–100 → 0.0–1.0
        gripper_pct = float(obs.get("gripper.pos", 0.0))
        q_rad[5]    = np.clip(gripper_pct / 100.0, 0.0, 1.0)

        # Keep tracked state in sync for the Jacobian IK
        self._q = q_rad.copy()

        ee = so101_fk(q_rad)
        return RobotState(
            joint_positions_rad = q_rad,
            end_effector_pos    = ee,
            gripper             = float(q_rad[5]),
            timestamp           = time.monotonic(),
        )

    def _read_mock_state(self) -> RobotState:
        q  = self._q.copy()
        ee = so101_fk(q)
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

        Hardware mode
        ~~~~~~~~~~~~~
        A numerical Jacobian J = ∂EE/∂q (shape 3×5) is computed from the
        SO-101 FK.  Joint deltas are obtained via the pseudo-inverse and
        integrated into the tracked joint state, which is sent as a position
        command to the robot.

        Mock mode
        ~~~~~~~~~
        The joint state is Euler-integrated directly for sanity-checking.
        """
        dt = 1.0 / 50.0  # 50 Hz control loop

        if self._connected and self._robot is not None:
            # Jacobian pseudo-inverse velocity → joint-position integration
            J    = _numerical_jacobian(self._q)              # (3, 5)
            dq   = np.linalg.pinv(J) @ velocity_mps         # (5,)
            self._q[:_N_ARM] += dq * dt

            action = {
                f"{name}.pos": float(np.rad2deg(self._q[i]))
                for i, name in enumerate(_JOINT_NAMES[:_N_ARM])
            }
            # Preserve current gripper position (set_gripper handles changes)
            action["gripper.pos"] = float(np.clip(self._q[5], 0.0, 1.0) * 100.0)
            self._robot.send_action(action)
        else:
            # Mock: Euler-integrate using Jacobian (same path, no hardware call)
            J  = _numerical_jacobian(self._q)
            dq = np.linalg.pinv(J) @ velocity_mps
            self._q[:_N_ARM] += dq * dt
            ee_new = so101_fk(self._q)
            log.debug("[MOCK] vel=%s → ee_approx=%s", velocity_mps.round(4), ee_new.round(4))

    def set_gripper(self, state: float) -> None:
        """
        Open or close the gripper.

        Parameters
        ----------
        state : 0.0 = fully open, 1.0 = fully closed.
        """
        state = float(np.clip(state, 0.0, 1.0))
        self._q[5] = state   # update tracked state

        if self._connected and self._robot is not None:
            # SO-101 gripper uses RANGE_0_100 norm mode → send 0–100
            self._robot.send_action({"gripper.pos": state * 100.0})
        else:
            log.debug("[MOCK] gripper → %.1f", state)

    # ------------------------------------------------------------------
    # Calibration utilities (public — useful for diagnostics)
    # ------------------------------------------------------------------

    def raw_to_rad(self, motor: str, raw: int) -> Optional[float]:
        """
        Convert a raw encoder reading (0–4095) to radians using the loaded
        calibration.  Returns None if calibration was not loaded or the motor
        name is not in the calibration file.
        """
        if self._calib is None or motor not in self._calib:
            return None
        return encoder_to_rad(raw, self._calib[motor])

    def raw_to_gripper(self, raw: int) -> Optional[float]:
        """
        Convert a raw gripper encoder reading (0–4095) to a 0–1 fraction
        using the loaded calibration.
        """
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
