# vla_framework/sim/mujoco_interfaces.py
from __future__ import annotations
import logging, math, time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from ..control.lerobot_interface import RobotState
from .mujoco_env import MuJoCoEnv

log = logging.getLogger(__name__)

JOINT_NAMES: list[str] = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]

# Gripper actuator names and per-actuator max closed position (rad)
_GRIPPER_ACT_NAMES: list[str] = [
    "actuator_rh_r1", "actuator_rh_r2", "actuator_rh_l1", "actuator_rh_l2",
]
_GRIPPER_ACT_MAX: list[float] = [1.1, 1.0, 1.1, 1.0]

GRIPPER_OPEN_RAD:   float = 0.0
GRIPPER_CLOSED_RAD: float = 1.0   # normalized 0–1

# Velocity / stability limits
_MAX_CART_VEL    = 0.80   # m/s — per-axis Cartesian velocity cap
_MAX_JOINT_DELTA = 0.15   # rad — max joint change per control call

# Home joint configuration: all joints zero.
_HOME_Q = np.zeros(6, dtype=np.float64)


@dataclass
class MuJoCoSnapshot:
    rgb_snapshot: np.ndarray
    stable_depth: np.ndarray
    frame_index:  int = 0


class MuJoCoRobotInterface:
    def __init__(self, env: MuJoCoEnv,
                 joint_names: list[str] = JOINT_NAMES,
                 ee_body_name: str = "tcp_link",
                 control_hz: float = 50.0):
        self._env         = env
        self._joint_names = joint_names
        self._ee_body     = ee_body_name
        self._connected   = False
        self._q           = np.zeros(6, dtype=np.float64)   # 6 arm joints
        self._gripper_state: float = 0.0                     # normalized 0–1
        self._ctrl_dt     = 1.0 / control_hz
        self._vel_log_ctr = 0

        # Cache EE body id, gripperframe site id, and arm DOF indices.
        try:
            import mujoco as _mj
            self._ee_bid = _mj.mj_name2id(
                env.model, _mj.mjtObj.mjOBJ_BODY, ee_body_name)
            self._ee_site_id: int = _mj.mj_name2id(
                env.model, _mj.mjtObj.mjOBJ_SITE, "gripperframe")
            self._arm_dof_ids: list[int] = [
                int(env.model.jnt_dofadr[
                    _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_JOINT, n)])
                for n in joint_names
            ]
            log.info(
                "[SIM] ee_body_id=%d  gripperframe_site_id=%d  arm_dof_ids=%s",
                self._ee_bid, self._ee_site_id, self._arm_dof_ids,
            )
        except Exception as e:
            log.warning("[SIM] Jacobian setup failed: %s", e)
            self._ee_bid      = -1
            self._ee_site_id  = -1
            self._arm_dof_ids = list(range(6))

        # Arm actuator IDs: try exact name → {name}_act → actuator_{name}
        try:
            import mujoco as _mj
            ids = []
            for n in joint_names:
                aid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_ACTUATOR, n)
                if aid < 0:
                    aid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_ACTUATOR, f"{n}_act")
                if aid < 0:
                    aid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_ACTUATOR, f"actuator_{n}")
                ids.append(aid)
            self._act_ids: list[int] = ids
            log.info("[SIM] arm actuator ids: %s",
                     {n: i for n, i in zip(joint_names, ids)})
        except Exception:
            self._act_ids = [-1] * len(joint_names)

        # Gripper actuator IDs
        try:
            import mujoco as _mj
            self._gripper_act_ids: list[int] = [
                _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_ACTUATOR, n)
                for n in _GRIPPER_ACT_NAMES
            ]
            log.info("[SIM] gripper actuator ids: %s",
                     {n: i for n, i in zip(_GRIPPER_ACT_NAMES, self._gripper_act_ids)})
        except Exception:
            self._gripper_act_ids = [-1] * len(_GRIPPER_ACT_NAMES)

    @property
    def is_connected(self): return self._connected

    def _init_mocap(self):
        """Find mocap body ID for EE control."""
        try:
            import mujoco as _mj
            self._mocap_bid = _mj.mj_name2id(
                self._env.model, _mj.mjtObj.mjOBJ_BODY, "mocap_ee")
            if self._mocap_bid >= 0:
                self._mocap_id = int(
                    self._env.model.body_mocapid[self._mocap_bid])
                log.info("[SIM] mocap_ee found: body_id=%d mocap_id=%d",
                         self._mocap_bid, self._mocap_id)
            else:
                self._mocap_id = -1
                log.warning("[SIM] mocap_ee not found — falling back to Jacobian IK")
        except Exception as e:
            self._mocap_id = -1
            log.warning("[SIM] mocap init failed: %s", e)

    def connect(self):
        self._connected = True
        self._init_mocap()
        try:
            import mujoco as _mj
            _mj.mj_forward(self._env.model, self._env.data)
        except Exception:
            pass
        try:
            q = self._env.get_joint_positions(self._joint_names)
            self._q[:] = q
        except Exception as e:
            log.warning("Init joint read failed: %s", e)
        state = self.get_state()
        ee = state.end_effector_pos
        if hasattr(self, '_mocap_id') and self._mocap_id >= 0:
            self._env.data.mocap_pos[self._mocap_id] = ee.copy()
            log.info("[SIM] mocap_ee initialized to EE pos: (%.4f, %.4f, %.4f)",
                     ee[0], ee[1], ee[2])
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
            if self._ee_site_id >= 0:
                ee = self._env.data.site_xpos[self._ee_site_id].copy()
            else:
                import mujoco
                bid = mujoco.mj_name2id(self._env.model,
                                        mujoco.mjtObj.mjOBJ_BODY, self._ee_body)
                ee = self._env.data.xpos[bid].copy() if bid >= 0 else np.zeros(3)
        except Exception:
            ee = np.zeros(3)
        return RobotState(
            joint_positions_rad = self._q.copy(),
            end_effector_pos    = ee,
            gripper             = self._gripper_state,
            timestamp           = time.monotonic(),
        )

    def _sync_ctrl(self, indices=None):
        """Write self._q arm joint values into data.ctrl."""
        if indices is None:
            indices = range(len(self._joint_names))
        for i in indices:
            aid = self._act_ids[i] if i < len(self._act_ids) else -1
            if aid < 0:
                continue
            self._env.data.ctrl[aid] = float(self._q[i])

    def send_cartesian_velocity(self, velocity_mps: np.ndarray):
        """
        Move EE by updating mocap body position (if available),
        otherwise fall back to Jacobian IK.
        """
        v = np.clip(np.asarray(velocity_mps, dtype=float)[:3],
                    -_MAX_CART_VEL, _MAX_CART_VEL)

        if hasattr(self, '_mocap_id') and self._mocap_id >= 0:
            self._env.data.mocap_pos[self._mocap_id] += v * self._ctrl_dt
        else:
            import mujoco
            jacp = np.zeros((3, self._env.model.nv))
            if self._ee_site_id >= 0:
                mujoco.mj_jacSite(self._env.model, self._env.data,
                                  jacp, None, self._ee_site_id)
            else:
                mujoco.mj_jacBody(self._env.model, self._env.data,
                                  jacp, None, self._ee_bid)
            J = jacp[:, self._arm_dof_ids]
            _DAMPING = 0.05
            A  = J @ J.T + _DAMPING * np.eye(3)
            dq = J.T @ np.linalg.solve(A, v * self._ctrl_dt)
            dq = np.clip(dq, -_MAX_JOINT_DELTA, _MAX_JOINT_DELTA)
            self._q[:6] += dq
            self._sync_ctrl(range(6))

        self._vel_log_ctr += 1
        if self._vel_log_ctr % 50 == 1:
            ee = self.get_state().end_effector_pos
            log.debug("[vel] v=(%.3f,%.3f,%.3f)  ee=(%.3f,%.3f,%.3f)",
                      v[0], v[1], v[2], ee[0], ee[1], ee[2])

    def set_gripper(self, state: float):
        state = float(np.clip(state, 0.0, 1.0))
        self._gripper_state = state
        for aid, max_val in zip(self._gripper_act_ids, _GRIPPER_ACT_MAX):
            if aid >= 0:
                self._env.data.ctrl[aid] = state * max_val
        log.info("[Gripper] set_gripper(%.2f)", state)

    def set_wrist_roll(self, angle_rad: float):
        """Set joint5 (Z-rotation / wrist roll) to a fixed angle."""
        self._q[4] = float(angle_rad)
        self._sync_ctrl([4])

    def step_simulation(self): self._env.step()
    def step(self): self._env.step()
    def reset(self):
        self._env.reset()
        self._q[:] = _HOME_Q.copy()
        self._gripper_state = 0.0
        self._env.set_joint_positions(self._joint_names, self._q)
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

            _FLIP = np.diag([1.0, 1.0, -1.0])
            R_c2w = R_mj.T @ _FLIP
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_c2w
            T[:3,  3] = t_cam
            T_inv = np.linalg.inv(T)

            img  = _Image.fromarray(rgb)
            draw = _Draw.Draw(img)

            for body_name in self._debug_bods:
                bid = mujoco.mj_name2id(
                    self._env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid < 0:
                    log.warning("[debug] body %r not found", body_name)
                    continue

                p_world = self._env.data.xpos[bid].copy()
                p_cam_h = T_inv @ np.array([*p_world, 1.0])
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
