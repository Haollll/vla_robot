"""
OMY (Robotis OMY) Kinematics
=============================
Full URDF-based FK and numerical IK for the OMY 6-DOF arm.

Joint layout (from omy.xml):
  joint1 — base_link → link1   pos=(0, 0, 0.8715)   axis Z  (pan / yaw)
  joint2 — link1 → link2       pos=(0, -0.1215, 0)  axis Y  (shoulder lift)
  joint3 — link2 → link3       pos=(0, 0, 0.247)    axis Y  (elbow)
  joint4 — link3 → link4       pos=(0, 0.1215, 0.2195) axis Y (forearm)
  joint5 — link4 → link5       pos=(0, -0.113, 0)   axis Z  (wrist pan)
  joint6 — link5 → link6       pos=(0, 0, 0.1155)   axis Y  (wrist tilt)
  tcp    — link6 → tcp_link    pos=(0, -0.22, 0)    fixed

Public API (mirrors SO101Kinematics interface used by trajectory_builder):
  omy_fk(q_rad)           → (3,)   EE position [m] in robot base frame
  omy_fk_matrix(q_rad)    → (4,4)  T_EE→base
  OMyKinematics.inverse_kinematics_3d(x, y, z) → (j1..j6 deg)

Usage in trajectory_builder.py — replace:
  from ..control.so101_kinematics import SO101Kinematics
  self._kin = SO101Kinematics()
with:
  from ..control.omy_kinematics import OMyKinematics
  self._kin = OMyKinematics()
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Link transforms extracted from omy.xml
# Each entry: (xyz [m], axis, is_revolute)
# For revolute joints: rotation is about the given axis.
# ---------------------------------------------------------------------------

# Fixed offsets between consecutive joint frames (from XML pos= attributes)
_JOINT_OFFSETS = [
    np.array([0.0,     0.0,      0.8715], dtype=np.float64),  # base → link1 (joint1)
    np.array([0.0,    -0.1215,   0.0   ], dtype=np.float64),  # link1 → link2 (joint2)
    np.array([0.0,     0.0,      0.247 ], dtype=np.float64),  # link2 → link3 (joint3)
    np.array([0.0,     0.1215,   0.2195], dtype=np.float64),  # link3 → link4 (joint4)
    np.array([0.0,    -0.113,    0.0   ], dtype=np.float64),  # link4 → link5 (joint5)
    np.array([0.0,     0.0,      0.1155], dtype=np.float64),  # link5 → link6 (joint6)
]

# Joint rotation axes (unit vectors in parent frame, from omy.xml axis= attributes)
_JOINT_AXES = [
    np.array([0.0, 0.0, 1.0]),  # joint1: Z
    np.array([0.0, 1.0, 0.0]),  # joint2: Y
    np.array([0.0, 1.0, 0.0]),  # joint3: Y
    np.array([0.0, 1.0, 0.0]),  # joint4: Y
    np.array([0.0, 0.0, 1.0]),  # joint5: Z
    np.array([0.0, 1.0, 0.0]),  # joint6: Y
]

# Fixed TCP offset: link6 → tcp_link (pos="0 -0.22 0", no rotation)
_TCP_OFFSET = np.array([0.0, -0.22, 0.0], dtype=np.float64)


def _rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for rotation of `angle` rad about `axis`."""
    c, s = math.cos(angle), math.sin(angle)
    t    = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ], dtype=np.float64)


def omy_fk_matrix(joint_angles_rad: np.ndarray) -> np.ndarray:
    """
    Full forward kinematics for the OMY arm.

    Parameters
    ----------
    joint_angles_rad : array-like, shape (≥6,)
        [joint1, joint2, joint3, joint4, joint5, joint6] in radians.

    Returns
    -------
    T : (4, 4) homogeneous transform  T_EE→base_link.
    """
    T = np.eye(4, dtype=np.float64)

    for i in range(6):
        # Translate to joint origin
        T_offset = np.eye(4, dtype=np.float64)
        T_offset[:3, 3] = _JOINT_OFFSETS[i]

        # Rotate about joint axis
        T_rot = np.eye(4, dtype=np.float64)
        T_rot[:3, :3] = _rot_axis_angle(_JOINT_AXES[i], float(joint_angles_rad[i]))

        T = T @ T_offset @ T_rot

    # Apply fixed TCP offset
    T_tcp = np.eye(4, dtype=np.float64)
    T_tcp[:3, 3] = _TCP_OFFSET
    T = T @ T_tcp

    return T


def omy_fk(joint_angles_rad: np.ndarray) -> np.ndarray:
    """
    Returns EE position (x, y, z) [m] in robot base frame.
    """
    return omy_fk_matrix(joint_angles_rad)[:3, 3]


# ---------------------------------------------------------------------------
# Numerical IK via Jacobian pseudo-inverse (damped least-squares)
# ---------------------------------------------------------------------------

def _omy_jacobian(q: np.ndarray) -> np.ndarray:
    """
    Compute the 3×6 translational Jacobian for the OMY arm at configuration q.
    Uses finite differences (central) for robustness.
    """
    eps = 1e-5
    J   = np.zeros((3, 6), dtype=np.float64)
    p0  = omy_fk(q)
    for i in range(6):
        dq     = q.copy()
        dq[i] += eps
        J[:, i] = (omy_fk(dq) - p0) / eps
    return J


def omy_ik(
    target_xyz:  np.ndarray,
    q0:          np.ndarray | None = None,
    max_iters:   int   = 200,
    tol:         float = 1e-4,
    damping:     float = 1e-2,
    step_limit:  float = 0.2,
) -> np.ndarray:
    """
    Numerical IK using damped least-squares Jacobian.

    Parameters
    ----------
    target_xyz  : (3,) desired EE position [m] in base frame.
    q0          : (6,) initial joint angles [rad].  Defaults to zeros.
    max_iters   : maximum iterations.
    tol         : convergence tolerance [m].
    damping     : DLS damping factor λ.
    step_limit  : maximum joint change per step [rad] (prevents wild jumps).

    Returns
    -------
    q : (6,) joint angles [rad].
    """
    if q0 is None:
        # Reasonable home pose that puts the arm near table height
        q0 = np.array([0.0, -0.5, 1.0, 0.5, 0.0, 0.0], dtype=np.float64)

    q = q0.copy()

    # Joint limits from omy.xml (range="-6.28319 6.28319" for arm joints)
    q_lo = np.full(6, -6.28319)
    q_hi = np.full(6,  6.28319)

    for _ in range(max_iters):
        p   = omy_fk(q)
        err = target_xyz - p
        if np.linalg.norm(err) < tol:
            break

        J    = _omy_jacobian(q)
        # Damped least-squares: dq = J^T (J J^T + λ²I)^{-1} err
        JJT  = J @ J.T + (damping ** 2) * np.eye(3)
        dq   = J.T @ np.linalg.solve(JJT, err)

        # Clip step size
        scale = step_limit / (np.max(np.abs(dq)) + 1e-9)
        if scale < 1.0:
            dq *= scale

        q = np.clip(q + dq, q_lo, q_hi)

    return q


# ---------------------------------------------------------------------------
# OMyKinematics class — drop-in replacement for SO101Kinematics
# ---------------------------------------------------------------------------

class OMyKinematics:
    """
    Kinematics wrapper for the OMY arm.

    Provides the same interface as SO101Kinematics so that trajectory_builder
    and other consumers can swap robot models without code changes.

    All *_deg methods accept/return degrees; internal computation uses radians.
    """

    # Default home configuration: arm roughly above table at reachable height.
    # Tuned so that omy_fk(HOME_Q) ≈ (0.30, 0.00, 0.35) — similar to SO-101.
    HOME_Q_RAD = np.array([0.0, -0.4, 0.9, 0.6, 0.0, 0.0], dtype=np.float64)

    def __init__(self) -> None:
        # Verify FK is sane at home pose
        p = omy_fk(self.HOME_Q_RAD)
        # (informational — no assertion, just for callers who inspect)
        self._home_ee_pos = p

    # ------------------------------------------------------------------
    # FK
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        j1_deg: float,
        j2_deg: float,
        j3_deg: float,
        j4_deg: float = 0.0,
        j5_deg: float = 0.0,
        j6_deg: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        3-D FK.  Returns (x, y, z) [m] in base frame.
        """
        q = np.radians([j1_deg, j2_deg, j3_deg, j4_deg, j5_deg, j6_deg])
        p = omy_fk(q)
        return float(p[0]), float(p[1]), float(p[2])

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def inverse_kinematics_3d(
        self,
        x: float,
        y: float,
        z: float,
        q0: np.ndarray | None = None,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Numerical IK.  Returns (j1..j6) in degrees.

        Parameters
        ----------
        x, y, z : target EE position [m] in base frame.
        q0      : optional seed configuration [rad].
        """
        target = np.array([x, y, z], dtype=np.float64)
        if q0 is None:
            # Seed: point joint1 toward target in XY, rest at neutral
            pan = math.atan2(y, x)
            q0  = np.array([pan, -0.4, 0.9, 0.6, 0.0, 0.0], dtype=np.float64)

        q_rad = omy_ik(target, q0=q0)
        return tuple(float(math.degrees(qi)) for qi in q_rad)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Compatibility shim: SO101Kinematics used only 3 joints publicly.
    # These wrappers keep trajectory_builder working if it calls the
    # old 3-joint interface.
    # ------------------------------------------------------------------

    def forward_kinematics_2d(
        self,
        joint2_deg: float,
        joint3_deg: float,
    ) -> Tuple[float, float]:
        """
        Planar (radial, vertical) FK using joints 2 & 3 only (joints 1,4,5,6=0).
        Returns (x_radial, y_vertical) [m].
        """
        q = np.radians([0.0, joint2_deg, joint3_deg, 0.0, 0.0, 0.0])
        p = omy_fk(q)
        r = math.sqrt(p[0] ** 2 + p[1] ** 2)
        return r, float(p[2])

    def inverse_kinematics(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """
        Planar IK (radial reach x, height y) → (joint2_deg, joint3_deg).
        Uses numerical IK with joint1=0.
        """
        q_rad = omy_ik(np.array([x, 0.0, y]), q0=np.array([0.0, -0.4, 0.9, 0.6, 0.0, 0.0]))
        return math.degrees(q_rad[1]), math.degrees(q_rad[2])


    def generate_sinusoidal_velocity_trajectory(
        self, start_point, end_point,
        control_freq=100.0, total_time=5.0,
        velocity_amplitude=1.0, velocity_period=2.0, phase_offset=0.0,
    ):
        import math as _math
        start = np.array(start_point, dtype=float)
        end   = np.array(end_point,   dtype=float)
        direction  = end - start
        total_dist = float(np.linalg.norm(direction))
        dt    = 1.0 / control_freq
        n_pts = int(total_time * control_freq) + 1
        t_arr = np.linspace(0.0, total_time, n_pts)
        omega    = 2.0 * _math.pi / velocity_period
        base_vel = total_dist / total_time if total_time > 0 else 0.0
        velocities = base_vel + velocity_amplitude * np.sin(omega * t_arr + phase_offset)
        velocities = np.maximum(velocities, 0.1 * base_vel)
        pos_1d = np.zeros(n_pts)
        for i in range(1, n_pts):
            pos_1d[i] = pos_1d[i-1] + velocities[i-1] * dt
        if pos_1d[-1] > 0:
            pos_1d *= total_dist / pos_1d[-1]
        trajectory = np.zeros((n_pts, 3))
        for i in range(n_pts):
            progress = pos_1d[i] / total_dist if total_dist > 0 else 0.0
            trajectory[i] = start + progress * direction
        return trajectory, velocities, t_arr

# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("OMY Kinematics self-test")
    print("=" * 40)

    kin = OMyKinematics()
    print(f"Home EE position: {kin._home_ee_pos}")

    # Round-trip FK → IK → FK
    test_positions = [
        (0.30,  0.00, 0.35),
        (0.25,  0.15, 0.30),
        (0.20, -0.10, 0.40),
    ]

    for tx, ty, tz in test_positions:
        j_deg = kin.inverse_kinematics_3d(tx, ty, tz)
        q_rad = np.radians(j_deg)
        p_fk  = omy_fk(q_rad)
        err   = math.sqrt((p_fk[0]-tx)**2 + (p_fk[1]-ty)**2 + (p_fk[2]-tz)**2)
        print(
            f"  target=({tx:.3f},{ty:.3f},{tz:.3f})  "
            f"fk=({p_fk[0]:.3f},{p_fk[1]:.3f},{p_fk[2]:.3f})  "
            f"err={err*1000:.1f}mm  "
            f"joints={[f'{d:.1f}' for d in j_deg]}"
        )

    print("\nDone.")
