# vla_framework/sim/mujoco_interfaces.py
from __future__ import annotations
import logging, math, time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..control.lerobot_interface import RobotState
from ..control.so101_kinematics  import SO101Kinematics
from .mujoco_env import MuJoCoEnv

log = logging.getLogger(__name__)

JOINT_NAMES: list[str] = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
GRIPPER_OPEN_RAD:   float = 0.0
# SO-101 gripper joint ctrlrange = [-0.17453, 1.74533].
# 0.8 rad (≈46°) was not closing firmly enough to grip the cube.
# 1.4 rad (≈80°, 80% of max) applies substantially more closing force.
GRIPPER_CLOSED_RAD: float = 1.4

# Velocity / stability limits
_MAX_CART_VEL   = 0.80   # m/s — per-axis Cartesian velocity cap
_MAX_JOINT_DELTA = 0.15  # rad — max joint change per control call

# Home joint configuration: all joints zero.
# Full FK at zeros → EE ≈ [0.391, 0.000, 0.227] m (arm pointing forward,
# above the table — within 0.17 m of typical workspace waypoints).
_HOME_Q = np.zeros(6, dtype=np.float64)


@dataclass
class MuJoCoSnapshot:
    rgb_snapshot: np.ndarray
    stable_depth: np.ndarray
    frame_index:  int = 0


class MuJoCoRobotInterface:
    def __init__(self, env: MuJoCoEnv,
                 joint_names: list[str] = JOINT_NAMES,
                 ee_body_name: str = "end_effector",
                 control_hz: float = 50.0):
        self._env         = env
        self._joint_names = joint_names
        self._ee_body     = ee_body_name
        self._kin         = SO101Kinematics()   # kept for _fk_ee fallback only
        self._connected   = False
        self._q           = np.zeros(6, dtype=np.float64)
        self._ctrl_dt     = 1.0 / control_hz   # PID control period [s]
        self._vel_log_ctr = 0                   # throttle verbose velocity logs

        # Cache EE body id and arm DOF indices for Jacobian-based IK.
        # Also look up the "gripperframe" site which is the actual fingertip
        # contact point (~10 cm below moving_jaw body origin).  Using the site
        # as the EE reference avoids the arm stalling because the moving_jaw
        # body origin is several cm above the physical gripper contact surfaces.
        try:
            import mujoco as _mj
            self._ee_bid = _mj.mj_name2id(
                env.model, _mj.mjtObj.mjOBJ_BODY, ee_body_name)
            # gripperframe site: actual fingertip position in world frame.
            self._ee_site_id: int = _mj.mj_name2id(
                env.model, _mj.mjtObj.mjOBJ_SITE, "gripperframe")
            # DOF (velocity) indices for the 5 arm joints (not gripper).
            self._arm_dof_ids: list[int] = [
                int(env.model.jnt_dofadr[
                    _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_JOINT, n)])
                for n in joint_names[:5]
            ]
            log.info(
                "[SIM] ee_body_id=%d  gripperframe_site_id=%d  arm_dof_ids=%s",
                self._ee_bid, self._ee_site_id, self._arm_dof_ids,
            )
        except Exception as e:
            log.warning("[SIM] Jacobian setup failed: %s", e)
            self._ee_bid      = -1
            self._ee_site_id  = -1
            self._arm_dof_ids = list(range(5))

        # Cache actuator IDs: try "<joint_name>" first (so101_new_calib.xml
        # names actuators identically to joints), fall back to "<joint_name>_act".
        try:
            import mujoco as _mj
            ids = []
            for n in joint_names:
                aid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_ACTUATOR, n)
                if aid < 0:
                    aid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_ACTUATOR, f"{n}_act")
                ids.append(aid)
            self._act_ids: list[int] = ids
            log.info(
                "[SIM] actuator ids: %s",
                {n: i for n, i in zip(joint_names, ids)},
            )
        except Exception:
            self._act_ids = [-1] * len(joint_names)

    @property
    def is_connected(self): return self._connected

    def connect(self):
        self._connected = True
        # Ensure kinematic state (xpos, xmat, Jacobians) is up-to-date.
        try:
            import mujoco as _mj
            _mj.mj_forward(self._env.model, self._env.data)
        except Exception:
            pass
        try:
            q = self._env.get_joint_positions(self._joint_names)
            self._q[:] = q
            self._q[5] = float(np.clip(
                (self._q[5] - GRIPPER_OPEN_RAD) /
                max(1e-6, GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD), 0.0, 1.0))
        except Exception as e:
            log.warning("Init joint read failed: %s", e)
        state = self.get_state()
        ee = state.end_effector_pos
        log.info(
            "[SIM] connected — EE home position: x=%.4f  y=%.4f  z=%.4f",
            ee[0], ee[1], ee[2],
        )
        log.info(
            "[SIM] joint angles (rad): %s",
            "  ".join(f"{n}={v:.3f}" for n, v in zip(self._joint_names, self._q)),
        )

    def disconnect(self):
        self._connected = False
        log.info("[SIM] disconnected")

    def __enter__(self): self.connect(); return self
    def __exit__(self, *_): self.disconnect()

    def get_state(self) -> RobotState:
        try:
            # Prefer the gripperframe site (actual fingertip contact point) so
            # that servo targets match the physical geometry.  Fall back to the
            # body origin if the site was not found in the model.
            if self._ee_site_id >= 0:
                ee = self._env.data.site_xpos[self._ee_site_id].copy()
            else:
                import mujoco
                bid = mujoco.mj_name2id(self._env.model,
                                        mujoco.mjtObj.mjOBJ_BODY, self._ee_body)
                ee = self._env.data.xpos[bid].copy() if bid >= 0 else self._fk_ee()
        except Exception:
            ee = self._fk_ee()
        return RobotState(
            joint_positions_rad = self._q.copy(),
            end_effector_pos    = ee,
            gripper             = float(self._q[5]),
            timestamp           = time.monotonic(),
        )

    def _fk_ee(self):
        return np.array(self._kin.forward_kinematics(
            math.degrees(self._q[0]),
            math.degrees(self._q[1]),
            math.degrees(self._q[2]),
        ))

    def _sync_ctrl(self, indices=None):
        """
        Write self._q values into data.ctrl for the specified joint indices.
        Keeping ctrl in sync with qpos prevents position actuators from
        applying large corrective forces on the next mj_step (→ NaN in QACC).
        """
        if indices is None:
            indices = range(len(self._joint_names))
        for i in indices:
            aid = self._act_ids[i] if i < len(self._act_ids) else -1
            if aid < 0:
                continue
            if i < 5:
                self._env.data.ctrl[aid] = float(self._q[i])
            else:
                # Gripper: convert normalized [0,1] → raw joint position
                rad = GRIPPER_OPEN_RAD + self._q[5] * (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD)
                self._env.data.ctrl[aid] = rad

    def send_cartesian_velocity(self, velocity_mps: np.ndarray):
        """
        Convert a Cartesian velocity command (m/s, world frame) to joint
        position increments via the MuJoCo translational Jacobian.

        dq = J^T (J J^T + λI)^{-1}  v * dt   (damped least-squares)

        This operates entirely in MuJoCo world frame, avoiding the coordinate-
        frame mismatch of the analytical SO101Kinematics model.
        """
        import mujoco

        v = np.clip(np.asarray(velocity_mps, dtype=float)[:3],
                    -_MAX_CART_VEL, _MAX_CART_VEL)

        # Translational Jacobian (3, nv) in world frame.
        # Use mj_jacSite at the gripperframe (fingertip) when available so the
        # IK is consistent with the position reported by get_state().
        jacp = np.zeros((3, self._env.model.nv))
        if self._ee_site_id >= 0:
            mujoco.mj_jacSite(self._env.model, self._env.data,
                              jacp, None, self._ee_site_id)
        else:
            mujoco.mj_jacBody(self._env.model, self._env.data,
                              jacp, None, self._ee_bid)

        # Extract columns for the 5 arm DOFs only.
        J = jacp[:, self._arm_dof_ids]   # (3, 5)

        # Damped least-squares: dq = J^T (J J^T + λI)^{-1} v*dt
        # λ=0.05 provides strong regularisation near singularities and
        # joint limits, at the cost of some accuracy far from them.
        _DAMPING = 0.05
        A   = J @ J.T + _DAMPING * np.eye(3)
        dq  = J.T @ np.linalg.solve(A, v * self._ctrl_dt)  # (5,)

        # Clamp per-joint delta to prevent discontinuities.
        dq = np.clip(dq, -_MAX_JOINT_DELTA, _MAX_JOINT_DELTA)
        self._q[:5] += dq

        # Diagnostic log (every 50 calls) — includes Jacobian condition number.
        self._vel_log_ctr += 1
        if self._vel_log_ctr % 50 == 1:
            J_norm = float(np.linalg.norm(J))
            cond   = float(np.linalg.cond(J))
            log.debug(
                "[vel] v=(%.3f,%.3f,%.3f)  |J|=%.4f  cond(J)=%.1f  dq=%s  q[:5]=%s",
                v[0], v[1], v[2], J_norm, cond,
                [f"{x:.4f}" for x in dq],
                [f"{x:.4f}" for x in self._q[:5]],
            )

        # Drive actuators via ctrl — position servos apply smooth corrections.
        self._sync_ctrl(range(5))

    def set_gripper(self, state: float):
        state = float(np.clip(state, 0.0, 1.0))
        self._q[5] = state
        rad = GRIPPER_OPEN_RAD + state * (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD)
        aid = self._act_ids[5] if len(self._act_ids) > 5 else -1
        before = float(self._env.data.ctrl[aid]) if aid >= 0 else float("nan")
        self._env.set_joint_positions([self._joint_names[5]], np.array([rad]))
        self._sync_ctrl([5])
        after = float(self._env.data.ctrl[aid]) if aid >= 0 else float("nan")
        log.info(
            "[Gripper] set_gripper(%.2f) → rad=%.4f rad  ctrl[%d]: %.4f → %.4f",
            state, rad, aid, before, after,
        )

    def set_wrist_roll(self, angle_rad: float):
        """Set wrist_roll joint to a fixed angle and sync actuator ctrl.

        Call this after positioning the arm (IK will have set wrist_roll
        freely); this clamps it to the desired orientation for grasping.
        """
        self._q[4] = float(angle_rad)
        self._sync_ctrl([4])

    def step_simulation(self): self._env.step()
    def step(self): self._env.step()
    def reset(self):
        self._env.reset()
        self._q[:] = _HOME_Q.copy()
        # Initialise joints and actuator setpoints to home configuration.
        self._env.set_joint_positions(self._joint_names[:5], self._q[:5])
        self._sync_ctrl()

    def set_object_pose(self, name, position, quaternion=None):
        self._env.set_body_pose(name, position, quaternion)

    def get_camera_rgba(self, camera_name, height=480, width=640):
        rgb, _ = self._env.render_camera(camera_name)
        return rgb

    def get_camera_depth(self, camera_name, height=480, width=640):
        _, depth = self._env.render_camera(camera_name)
        return depth


class MuJoCoCameraStreamer:
    def __init__(self, env: MuJoCoEnv, camera_name: str = "overhead_cam",
                 debug_save_path: Optional[str] = None,
                 debug_bodies:    Optional[list] = None):
        """
        Parameters
        ----------
        debug_save_path : if set, every snapshot is saved as a PNG to this path
                          with a circle overlay marking each body in debug_bodies.
        debug_bodies    : list of MuJoCo body names to project and annotate.
        """
        self._env        = env
        self._cam        = camera_name
        self._idx        = 0
        self.is_ready    = True
        self._debug_path = debug_save_path
        self._debug_bods = debug_bodies or []

    def snapshot(self) -> MuJoCoSnapshot:
        rgb, depth = self._env.render_camera(self._cam)
        self._idx += 1
        if self._debug_path:
            self._save_debug(rgb)
        return MuJoCoSnapshot(rgb_snapshot=rgb, stable_depth=depth,
                              frame_index=self._idx)

    def _save_debug(self, rgb: np.ndarray) -> None:
        """Save rgb to disk with projected body positions annotated."""
        try:
            import mujoco
            from PIL import Image as _Image, ImageDraw as _Draw

            cam_id = mujoco.mj_name2id(
                self._env.model, mujoco.mjtObj.mjOBJ_CAMERA, self._cam)
            if cam_id < 0:
                return

            h, w = rgb.shape[:2]
            fovy_rad = math.radians(float(self._env.model.cam_fovy[cam_id]))
            fy = (h / 2.0) / math.tan(fovy_rad / 2.0)
            fx = fy
            cx, cy = w / 2.0, h / 2.0

            # Camera pose from MuJoCo: rows of cam_xmat = camera axes in world.
            R_mj  = self._env.data.cam_xmat[cam_id].reshape(3, 3).copy()
            t_cam = self._env.data.cam_xpos[cam_id].copy()
            log.info("[debug] cam_id=%d  name=%r", cam_id, self._cam)
            log.info("[debug] cam_xpos: [%.6f, %.6f, %.6f]", t_cam[0], t_cam[1], t_cam[2])
            log.info("[debug] cam_xmat row0: [%.6f, %.6f, %.6f]", R_mj[0,0], R_mj[0,1], R_mj[0,2])
            log.info("[debug] cam_xmat row1: [%.6f, %.6f, %.6f]", R_mj[1,0], R_mj[1,1], R_mj[1,2])
            log.info("[debug] cam_xmat row2: [%.6f, %.6f, %.6f]", R_mj[2,0], R_mj[2,1], R_mj[2,2])

            # Build the same T_cam→world that camera_utils / DepthProjector use.
            # R_c2w = R_mj.T @ diag([1, +1, -1])  (same as camera_utils._FLIP)
            _FLIP = np.diag([1.0, 1.0, -1.0])
            R_c2w = R_mj.T @ _FLIP
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_c2w
            T[:3,  3] = t_cam
            # Invert: T_world→cam_cv  (exact inverse of DepthProjector's pipeline)
            T_inv = np.linalg.inv(T)

            img  = _Image.fromarray(rgb)
            draw = _Draw.Draw(img)

            for body_name in self._debug_bods:
                bid = mujoco.mj_name2id(
                    self._env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid < 0:
                    log.warning("[debug] body %r not found", body_name)
                    continue

                p_world = self._env.data.xpos[bid].copy()     # (3,) world pos
                p_cam_h = T_inv @ np.array([*p_world, 1.0])   # OpenCV cam frame
                X_cam, Y_cam, Z_cam = p_cam_h[0], p_cam_h[1], p_cam_h[2]

                if Z_cam <= 0:
                    log.warning("[debug] %r is behind the camera (Z_cam=%.3f)",
                                body_name, Z_cam)
                    continue

                u = cx + fx * X_cam / Z_cam
                v = cy + fy * Y_cam / Z_cam
                log.info(
                    "[debug] %r world(%.3f,%.3f,%.3f)"
                    " cam_cv(%.3f,%.3f,%.3f) → pixel(%.0f, %.0f)",
                    body_name,
                    p_world[0], p_world[1], p_world[2],
                    X_cam, Y_cam, Z_cam, u, v,
                )

                r = 10
                draw.ellipse([u - r, v - r, u + r, v + r],
                             outline=(255, 0, 0), width=3)
                draw.text((u + r + 2, v - r), body_name, fill=(255, 0, 0))

            img.save(self._debug_path)
            log.info("[debug] snapshot saved → %s", self._debug_path)

        except Exception as exc:
            log.warning("[debug] save failed: %s", exc)

    def __enter__(self): return self
    def __exit__(self, *_): pass
