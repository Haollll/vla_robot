"""
SimVLAPipeline
==============
VLAPipeline variant with looser plausibility bounds for MuJoCo simulation,
plus sim-specific Gemini prompt enhancement and image annotation.

Why needed
----------
In a real-hardware pipeline, z < 0 reliably indicates a bad depth reading
(point below the robot's base plane).  In simulation the rendered depth of
the floor carries ≈ 0.1 mm floating-point noise, so a pixel pointing at the
floor at exactly camera-height produces z ≈ -1e-6, falsely failing the check.

This subclass relaxes _MIN_Z to -0.02 m (2 cm) while keeping the 1.5 m
distance guard.  The 2 cm margin covers rendering noise without masking
genuinely bad projections.

Usage (sim_main.py already uses this automatically):
    pipeline = SimVLAPipeline(config, robot=robot, streamer=streamer,
                              sim_step_fn=env.step)
"""
from __future__ import annotations

import logging
import math
import time
from typing import List, Optional

import numpy as np

from ..pipeline import VLAPipeline, ExecuteStage
from ..projection.depth_projection import Point3D

log = logging.getLogger(__name__)

# Looser z floor for simulation: 2 cm below robot base plane is still
# considered plausible (accounts for rendering noise at the floor level).
_SIM_MIN_Z    = -0.02   # m
_SIM_MAX_DIST =  1.5    # m  (same as production)

# Extra context appended to every command sent to Gemini in simulation.
_SIM_PROMPT_SUFFIX = (
    " IMPORTANT CONTEXT: This image is a top-down view of a robot workspace. "
    "A robot arm (metal links with a red-colored gripper tip) is visible in the image — "
    "DO NOT click on the robot arm or its gripper. "
    "The target object is a small red box (cube) sitting on the blue checkered "
    "floor/table surface, away from the robot arm. "
    "Click on the CENTER of the small red cube on the table surface, "
    "not on any part of the robot itself. "
    "The cube appears as a small distinct red square separate from the robot."
)


class _SimAnnotatingPlanStage:
    """
    Wraps PlanStage to ground Gemini's pixel output to the actual object position.

    Strategy
    --------
    1. Compute the ground-truth pixel of the primary debug body (e.g. red_cube)
       by projecting its MuJoCo world position through the calibrated camera.
    2. Let Gemini plan the waypoint sequence, action types, and gripper states.
    3. Override every waypoint's pixel_coords with the ground-truth pixel — so
       the planned trajectory goes to the correct location regardless of where
       Gemini clicked.

    This is robust: Gemini handles open-vocabulary understanding (what to do and
    in what order), while the pixel position is guaranteed accurate from physics.
    """

    def __init__(self, inner, env, config, debug_bodies):
        self._inner        = inner
        self._env          = env
        self._config       = config
        self._debug_bodies = debug_bodies or []

    def run(self, rgb: np.ndarray, command: str) -> list:
        # Log the full prompt before sending.
        log.info("[SimPlan] --- system_prompt (in gemini_planner._SYSTEM_PROMPT) ---")
        log.info("[SimPlan] --- user_prompt ---")
        log.info("[SimPlan] Image size: %dx%d", rgb.shape[1], rgb.shape[0])
        log.info("[SimPlan] Task command: %s", command)

        # Compute ground-truth pixel for the primary target body.
        gt_pixel = self._project_primary_body()

        waypoints = self._inner.run(rgb, command)

        # Override pixel coords with ground-truth if we have a valid projection.
        if gt_pixel is not None:
            u_gt, v_gt = gt_pixel
            log.info(
                "[SimPlan] Gemini returned %d waypoints — overriding all pixel "
                "coords with ground-truth (%d, %d) from %r",
                len(waypoints), u_gt, v_gt, self._debug_bodies[0],
            )
            for wp in waypoints:
                old = wp.pixel_coords
                wp.pixel_coords = (u_gt, v_gt)
                log.info(
                    "[SimPlan]   %-10s  Gemini=(%d,%d) → GT=(%d,%d)  %s",
                    wp.action_type.value, old[0], old[1], u_gt, v_gt,
                    wp.description,
                )
        else:
            log.warning(
                "[SimPlan] No ground-truth pixel available; keeping Gemini coords"
            )

        return waypoints

    def _project_primary_body(self):
        """Return (u, v) pixel for the first debug body, or None on failure."""
        if not self._debug_bodies:
            return None
        try:
            import mujoco
            body_name = self._debug_bodies[0]
            bid = mujoco.mj_name2id(
                self._env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                log.warning("[SimPlan] body %r not found", body_name)
                return None

            p_world = self._env.data.xpos[bid].copy()
            T_inv   = np.linalg.inv(self._config.camera_extrinsics.T)
            K       = self._config.camera_intrinsics

            p_cam_h = T_inv @ np.array([*p_world, 1.0])
            X_cam, Y_cam, Z_cam = p_cam_h[0], p_cam_h[1], p_cam_h[2]
            if Z_cam <= 0:
                log.warning("[SimPlan] %r is behind the camera", body_name)
                return None

            u = int(round(K.cx + K.fx * X_cam / Z_cam))
            v = int(round(K.cy + K.fy * Y_cam / Z_cam))
            log.info(
                "[SimPlan] ground-truth pixel for %r: (%d, %d)  "
                "world=(%.3f, %.3f, %.3f)",
                body_name, u, v, p_world[0], p_world[1], p_world[2],
            )
            return (u, v)
        except Exception as exc:
            log.warning("[SimPlan] projection failed: %s", exc)
            return None


# Servo iterations for simulation: 15 s × 200 Hz = 3000 steps per waypoint.
# (At 50 Hz this was 60 s; 4× freq → same iter count = 4× shorter wall-clock
# timeout, which is fine given the higher velocity cap of 0.8 m/s.)
_SIM_SERVO_MAX_ITERS = 3000
_SIM_SERVO_LOG_EVERY = 400   # log progress every N iterations (~2 s at 200 Hz)
# Accept waypoint as "reached" on timeout if remaining error < this threshold.
# Handles oscillation near the target that never quite hits waypoint_tolerance.
_SIM_CLOSE_ENOUGH    = 0.06  # m  (6 cm fallthrough — covers gripper-frame offset residuals)

# Contact-based grasp approach constants.
_GRASP_HOVER_Z               = 0.12   # m — hover height above table (high enough to avoid false contact)
_GRASP_CONTACT_VEL_Z         = -0.05  # m/s — slow descent velocity
_GRASP_CONTACT_MAX_STEPS     = 800    # max descent iterations before giving up
_GRASP_CONTACT_MIN_Z         = -0.01  # m — floor guard
_GRASP_CONTACT_ARM_Z         = 0.045  # m — ignore contacts above cube-top height (false-contact filter)
_GRASP_EXTRA_STEPS_AFTER_CONTACT = 200  # extra steps at vel_z after first contact
_GRASP_WRIST_ROLL_RAD        = 0.0    # rad — wrist neutral

# Grasp timing constants (all in physics steps at the sim timestep).
_GRASP_INIT_STEPS       = 50   # sim steps after initial gripper-open before any movement
_GRASP_POST_CLOSE_STEPS = 50   # sim steps after gripper close for contact forces to engage
_LIFT_SUCCESS_Z     = 0.05   # m — cube must reach this height after LIFT to confirm success
_MAX_GRASP_RETRIES  = 2      # maximum re-grasp attempts before proceeding anyway
# Z heights for the full re-grasp sequence executed on retry (fresh cube GT XY used).
_RETRY_APPROACH_Z   = 0.19   # m — safe approach height above table
_RETRY_PRE_GRASP_Z  = 0.09   # m — pre-grasp hover (one spline segment above cube)
_RETRY_LIFT_Z       = 0.24   # m — target lift height after re-close


class _SimExecuteStage(ExecuteStage):
    """ExecuteStage with longer timeout, progress logging, and soft close-enough fallback."""

    def _servo_to(self, target, dt, max_iters=_SIM_SERVO_MAX_ITERS):
        import numpy as _np
        log.info(
            "[SimServo] _servo_to  max_iters=%d (%.0f sim-s)  "
            "tol=%.3f m  target=(%.4f, %.4f, %.4f)",
            max_iters, max_iters * dt, self._tol,
            target[0], target[1], target[2],
        )
        for i in range(max_iters):
            state = self._robot.get_state()
            error = target - state.end_effector_pos
            err_norm = float(_np.linalg.norm(error))
            if err_norm < self._tol:
                log.info("[SimServo] converged  iters=%d  err=%.4f m", i, err_norm)
                return True
            if i % _SIM_SERVO_LOG_EVERY == 0:
                log.info(
                    "[SimServo] iter=%4d  err=%.4f m  ee=(%.4f,%.4f,%.4f)",
                    i, err_norm,
                    state.end_effector_pos[0],
                    state.end_effector_pos[1],
                    state.end_effector_pos[2],
                )
            self._robot.send_cartesian_velocity(self._pid.step(error))
            self._sim_step()

        remaining = float(_np.linalg.norm(
            target - self._robot.get_state().end_effector_pos))

        # --- Diagnostics at timeout: joint positions, limits, Jacobian cond ---
        try:
            import mujoco as _mj
            env   = self._robot._env
            model = env.model
            data  = env.data

            # Joint positions and limit check for the 5 arm joints.
            jnames = self._robot._joint_names[:5]
            qpos_vals = []
            at_limit  = []
            for jname in jnames:
                jid  = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, jname)
                qadr = int(model.jnt_qposadr[jid])
                q    = float(data.qpos[qadr])
                lo   = float(model.jnt_range[jid, 0])
                hi   = float(model.jnt_range[jid, 1])
                limited = bool(model.jnt_limited[jid])
                margin  = min(abs(q - lo), abs(q - hi)) if limited else float("inf")
                qpos_vals.append(q)
                if limited and margin < 0.05:   # within ~3° of limit
                    at_limit.append(f"{jname}(q={q:.3f} lo={lo:.3f} hi={hi:.3f})")

            log.warning(
                "[SimServo] timeout joints (rad): %s",
                "  ".join(f"{n}={v:.3f}" for n, v in zip(jnames, qpos_vals)),
            )
            if at_limit:
                log.warning("[SimServo] NEAR JOINT LIMITS: %s", "  ".join(at_limit))
            else:
                log.info("[SimServo] no joints near limits")

            # Jacobian condition number.
            site_id = getattr(self._robot, "_ee_site_id", -1)
            ee_bid  = self._robot._ee_bid
            dof_ids = self._robot._arm_dof_ids
            jacp = _np.zeros((3, model.nv))
            if site_id >= 0:
                _mj.mj_jacSite(model, data, jacp, None, site_id)
            else:
                _mj.mj_jacBody(model, data, jacp, None, ee_bid)
            J    = jacp[:, dof_ids]
            cond = float(_np.linalg.cond(J))
            log.warning(
                "[SimServo] Jacobian condition number at timeout: %.1f  %s",
                cond,
                "(NEAR SINGULARITY)" if cond > 100 else "(OK)",
            )
        except Exception as _diag_exc:
            log.warning("[SimServo] diagnostics failed: %s", _diag_exc)
        # --- end diagnostics ---

        if remaining < _SIM_CLOSE_ENOUGH:
            log.info(
                "[SimServo] close-enough after %d iters  remaining=%.4f m  "
                "(< _SIM_CLOSE_ENOUGH=%.3f m) — continuing",
                max_iters, remaining, _SIM_CLOSE_ENOUGH,
            )
            return True
        log.warning(
            "[SimServo] timeout after %d iters  remaining=%.4f m  "
            "(> _SIM_CLOSE_ENOUGH=%.3f m) — aborting waypoint",
            max_iters, remaining, _SIM_CLOSE_ENOUGH,
        )
        return False


    # ------------------------------------------------------------------
    # Contact-based descent
    # ------------------------------------------------------------------

    @staticmethod
    def _gripper_geom_ids(model) -> "set[int]":
        """Return geom IDs attached to the gripper body and moving_jaw child."""
        import mujoco as _mj
        ids: set = set()
        for bname in ("gripper", "moving_jaw_so101_v1"):
            bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                continue
            for gid in range(model.ngeom):
                if int(model.geom_bodyid[gid]) == bid:
                    ids.add(gid)
        return ids

    @staticmethod
    def _cube_geom_ids(model, cube_body_name: str = "red_cube") -> "set[int]":
        """Return geom IDs attached to the red_cube body."""
        import mujoco as _mj
        bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, cube_body_name)
        if bid < 0:
            return set()
        return {gid for gid in range(model.ngeom)
                if int(model.geom_bodyid[gid]) == bid}

    def _contact_descend(
        self,
        hover_xy: "np.ndarray",
        grip_aid: int = 5,
        cube_bid: int = -1,
    ) -> bool:
        """
        Descend at vel_z=-0.05 m/s with no XY steering (hover_xy is static).
        On first gripper↔cube contact below _GRASP_CONTACT_ARM_Z, continue
        descending for _GRASP_EXTRA_STEPS_AFTER_CONTACT more steps then return.
        """
        import numpy as _np
        env   = self._robot._env
        model = env.model
        data  = env.data

        grip_geoms = self._gripper_geom_ids(model)
        cube_geoms = self._cube_geom_ids(model)
        if not grip_geoms or not cube_geoms:
            log.warning("[ContactGrasp] could not resolve geom IDs — aborting")
            return False

        log.info(
            "[ContactGrasp] descend start  hover_xy=(%.4f,%.4f)  "
            "vel_z=%.2f m/s  max_steps=%d  grip_geoms=%s  cube_geoms=%s",
            hover_xy[0], hover_xy[1],
            _GRASP_CONTACT_VEL_Z, _GRASP_CONTACT_MAX_STEPS,
            sorted(grip_geoms), sorted(cube_geoms),
        )

        v = _np.array([0.0, 0.0, _GRASP_CONTACT_VEL_Z])

        for i in range(_GRASP_CONTACT_MAX_STEPS):
            self._robot.send_cartesian_velocity(v)
            self._robot.set_wrist_roll(_GRASP_WRIST_ROLL_RAD)
            self._sim_step()

            ee_z = float(self._robot.get_state().end_effector_pos[2])

            if i % 20 == 0:
                ee_pos = self._robot.get_state().end_effector_pos
                log.info(
                    "[ContactGrasp] iter=%d  ee_z=%.4f  ee_xy=(%.4f,%.4f)  ctrl[%d]=%.4f",
                    i, ee_z, ee_pos[0], ee_pos[1],
                    grip_aid, float(data.ctrl[grip_aid]),
                )

            if ee_z < _GRASP_CONTACT_MIN_Z:
                log.warning("[ContactGrasp] floor guard at iter=%d  ee_z=%.4f", i, ee_z)
                return False

            # Only scan for contact once below the false-contact threshold.
            if ee_z > _GRASP_CONTACT_ARM_Z:
                continue

            for c in range(int(data.ncon)):
                g1 = int(data.contact[c].geom1)
                g2 = int(data.contact[c].geom2)
                if (g1 in grip_geoms and g2 in cube_geoms) or \
                   (g2 in grip_geoms and g1 in cube_geoms):
                    log.info(
                        "[ContactGrasp] contact detected  iter=%d  ee_z=%.4f  "
                        "geoms=(%d,%d)  dist=%.5f — running %d extra steps",
                        i, ee_z, g1, g2, float(data.contact[c].dist),
                        _GRASP_EXTRA_STEPS_AFTER_CONTACT,
                    )
                    for _ in range(_GRASP_EXTRA_STEPS_AFTER_CONTACT):
                        self._robot.send_cartesian_velocity(v)
                        self._robot.set_wrist_roll(_GRASP_WRIST_ROLL_RAD)
                        self._sim_step()
                    log.info(
                        "[ContactGrasp] extra steps done  final ee_z=%.4f",
                        float(self._robot.get_state().end_effector_pos[2]),
                    )
                    return True

        log.warning(
            "[ContactGrasp] max steps (%d) reached without contact  final ee_z=%.4f",
            _GRASP_CONTACT_MAX_STEPS,
            float(self._robot.get_state().end_effector_pos[2]),
        )
        return False

    # ------------------------------------------------------------------
    # Grasp feedback loop
    # ------------------------------------------------------------------

    def run(self, trajectory: list) -> bool:
        """
        Extended run loop enforcing this exact gripper/motion sequence:

          0. set_gripper(0.0) + _GRASP_INIT_STEPS settle (before any movement)
          1–3. Servo through approach/pre_grasp/GRASP-descent (gripper stays open;
               gripper commands on non-keyframe interpolated points are ignored)
          4. At GRASP-close keyframe:
             a. Servo to (cube_XY, z=_GRASP_HOVER_Z) from directly above
             b. Align wrist_roll
             c. Contact descent: vel_z=-0.05 m/s until gripper ↔ cube contact
          5. set_gripper(1.0) immediately on contact + _GRASP_POST_CLOSE_STEPS settle
          6. Servo to LIFT (gripper stays closed)
          7. After last LIFT: cube-elevation check + re-grasp feedback loop
        """
        import mujoco as _mj
        from ..config import ActionType

        env      = self._robot._env
        cube_bid = _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_BODY, "red_cube")
        if cube_bid < 0:
            log.warning("[GraspCheck] body 'red_cube' not found — feedback disabled")

        # Ground-truth cube XY for GRASP overrides.
        # No X offset: arm naturally reaches cube_x; jaws (opening in X at 0.869 component)
        # span ±26mm around cube's ±20mm faces when centred on cube.
        _GRIPPERFRAME_X_OFFSET = 0.036  # m — X offset applied to cube GT XY for gripper targeting
        cube_gt_xy = env.data.xpos[cube_bid][:2].copy() if cube_bid >= 0 else None
        if cube_gt_xy is not None:
            cube_gt_xy[0] += _GRIPPERFRAME_X_OFFSET
            log.info(
                "[GraspCheck] cube GT XY = (%.4f, %.4f)  (after +%.3f m X offset)",
                cube_gt_xy[0], cube_gt_xy[1], _GRIPPERFRAME_X_OFFSET,
            )

        # ── Step 0: open gripper and let it settle before any movement ────
        _grip_aid = (self._robot._act_ids[5]
                     if hasattr(self._robot, "_act_ids") and len(self._robot._act_ids) > 5
                     else 5)
        log.info(
            "[GraspSeq] step 0 — set_gripper(0.0)  ctrl[%d] before=%.4f",
            _grip_aid, float(env.data.ctrl[_grip_aid]),
        )
        self._robot.set_gripper(0.0)
        for _ in range(_GRASP_INIT_STEPS):
            self._sim_step()
        log.info(
            "[GraspSeq] step 0 done — ctrl[%d] after settle=%.4f  "
            "(trajectory loop starting NOW)",
            _grip_aid, float(env.data.ctrl[_grip_aid]),
        )

        self._pid.reset()
        dt             = self._pid.dt
        prev_grip      = 0.0   # already open; don't retrigger on first point
        last_grasp_pos = None

        for idx, target in enumerate(trajectory):
            # ── Keyframe diagnostic log ───────────────────────────────────
            if target.is_keyframe:
                log.info(
                    "[KeyFrame] idx=%d  action=%-10s  gripper=%.1f  "
                    "pos=(%.4f, %.4f, %.4f)",
                    idx, target.action_type.value, target.gripper,
                    target.position[0], target.position[1], target.position[2],
                )

            # ── Build servo target — override XY for GRASP with GT cube pos ──
            servo_pos = target.position.copy()
            if target.action_type == ActionType.GRASP and cube_gt_xy is not None:
                servo_pos[:2] = cube_gt_xy

            # ── Enhanced GRASP close (keyframe only) ─────────────────────
            # Only fires once, at the keyframe where gripper transitions 0→1.
            # Interpolated GRASP points (is_keyframe=False) are skipped.
            _is_grasp_close = (
                target.is_keyframe
                and target.action_type == ActionType.GRASP
                and target.gripper > 0.5
                and abs(target.gripper - prev_grip) > 0.05
            )
            if target.is_keyframe:
                log.info(
                    "[KeyFrame] _is_grasp_close=%s  "
                    "(is_kf=%s  atype=%s  grip>0.5=%s  delta>0.05=%s  prev_grip=%.1f)",
                    _is_grasp_close,
                    target.is_keyframe, target.action_type.value,
                    target.gripper > 0.5,
                    abs(target.gripper - prev_grip) > 0.05,
                    prev_grip,
                )
            if _is_grasp_close:
                # Log wrist_roll joint angle to verify jaw opening direction.
                try:
                    import mujoco as _mj2
                    wr_jid  = _mj2.mj_name2id(env.model, _mj2.mjtObj.mjOBJ_JOINT, "wrist_roll")
                    wr_qadr = int(env.model.jnt_qposadr[wr_jid])
                    wr_rad  = float(env.data.qpos[wr_qadr])
                    # gripper body rotation matrix — column 0 is the jaw-opening axis in world
                    grip_bid  = _mj2.mj_name2id(env.model, _mj2.mjtObj.mjOBJ_BODY, "gripper")
                    R_grip    = env.data.xmat[grip_bid].reshape(3, 3)
                    jaw_open_world = R_grip[:, 0]   # local X → world (jaw opening direction)
                    log.info(
                        "[GraspSeq] wrist_roll=%.4f rad (%.1f°)  "
                        "jaw-opening axis in world: (%.3f, %.3f, %.3f)  "
                        "[should be mostly ±Y for side-grasp or ±X for front-grasp]",
                        wr_rad, wr_rad * 57.2958,
                        jaw_open_world[0], jaw_open_world[1], jaw_open_world[2],
                    )
                except Exception as _wr_exc:
                    log.warning("[GraspSeq] wrist_roll diagnostic failed: %s", _wr_exc)

                # 1. Servo to fixed hover height above the cube.
                hover_pos = servo_pos.copy()
                hover_pos[2] = _GRASP_HOVER_Z
                log.info(
                    "[GraspSeq] step 4a — hover → (%.4f, %.4f, %.4f)",
                    hover_pos[0], hover_pos[1], hover_pos[2],
                )
                self._servo_to(hover_pos, dt)

                # 2. Set wrist_roll to _GRASP_WRIST_ROLL_RAD (-π/2) so the jaw
                #    opens horizontally (Z≈0) during top-down descent.
                #    Predicted with FK scan: at -1.5708 rad the jaw-opening axis
                #    has |Z|≈0.034, well below the 0.400 at wrist_roll=0.
                import mujoco as _mj3
                import numpy as _np2

                _wr_jid   = _mj3.mj_name2id(env.model, _mj3.mjtObj.mjOBJ_JOINT, "wrist_roll")
                _wr_qadr  = int(env.model.jnt_qposadr[_wr_jid])
                _grip_bid = _mj3.mj_name2id(env.model, _mj3.mjtObj.mjOBJ_BODY, "gripper")
                _wr_before = float(env.data.qpos[_wr_qadr])

                self._robot.set_wrist_roll(_GRASP_WRIST_ROLL_RAD)

                # Use FK to predict the jaw direction at the new wrist angle
                # (no sim steps — physics runs during contact descent instead).
                _saved_wr = float(env.data.qpos[_wr_qadr])
                env.data.qpos[_wr_qadr] = _GRASP_WRIST_ROLL_RAD
                _mj3.mj_fwdPosition(env.model, env.data)
                _R_pred  = env.data.xmat[_grip_bid].reshape(3, 3)
                _jaw_pred = _R_pred[:, 0].copy()
                env.data.qpos[_wr_qadr] = _saved_wr           # restore actual qpos
                _mj3.mj_fwdPosition(env.model, env.data)       # restore FK

                log.info(
                    "[GraspSeq] step 4b — wrist_roll %.4f → %.4f rad  "
                    "predicted jaw-opening axis: (%.3f, %.3f, %.3f)  |Z|=%.4f",
                    _wr_before, _GRASP_WRIST_ROLL_RAD,
                    _jaw_pred[0], _jaw_pred[1], _jaw_pred[2],
                    abs(_jaw_pred[2]),
                )

                # 3. Contact descent — slow Z descent until gripper↔cube contact.
                #    Close immediately on first contact (GRASP_EXTRA_STEPS=0).
                log.info(
                    "[GraspSeq] step 4c — contact descent  ctrl[%d]=%.4f  "
                    "(must be OPEN ≈ 0.0 before descending)",
                    _grip_aid, float(env.data.ctrl[_grip_aid]),
                )
                self._contact_descend(servo_pos[:2], _grip_aid, cube_bid=cube_bid)

                # 4. Close gripper immediately on contact; settle for forces.
                log.info(
                    "[GraspSeq] step 5 — set_gripper(1.0)  ctrl[%d] before=%.4f",
                    _grip_aid, float(env.data.ctrl[_grip_aid]),
                )
                self._robot.set_gripper(1.0)
                log.info(
                    "[GraspSeq] step 5 — set_gripper done  ctrl[%d] after=%.4f  "
                    "settling %d steps",
                    _grip_aid, float(env.data.ctrl[_grip_aid]), _GRASP_POST_CLOSE_STEPS,
                )
                prev_grip = 1.0
                last_grasp_pos = self._robot.get_state().end_effector_pos.copy()
                cube_pos = env.data.xpos[cube_bid].copy() if cube_bid >= 0 else None
                log.info(
                    "[GraspCheck] gripperframe at close: (%.4f, %.4f, %.4f)  "
                    "cube_center: %s",
                    last_grasp_pos[0], last_grasp_pos[1], last_grasp_pos[2],
                    ("(%.4f, %.4f, %.4f)" % (cube_pos[0], cube_pos[1], cube_pos[2]))
                    if cube_pos is not None else "N/A",
                )
                if cube_pos is not None:
                    log.info(
                        "[GraspCheck] EE-cube offset: dx=%.4f  dy=%.4f  dz=%.4f",
                        last_grasp_pos[0] - cube_pos[0],
                        last_grasp_pos[1] - cube_pos[1],
                        last_grasp_pos[2] - cube_pos[2],
                    )
                for _ in range(_GRASP_POST_CLOSE_STEPS):
                    self._sim_step()

                # Position done; skip normal gripper+servo flow for this point.
                continue

            # ── Gripper — only at keyframe boundaries ─────────────────────
            if target.is_keyframe and abs(target.gripper - prev_grip) > 0.05:
                log.info(
                    "  [%4d] Gripper → %.0f  (%s)",
                    idx, target.gripper, target.description,
                )
                self._robot.set_gripper(target.gripper)
                prev_grip = target.gripper
                time.sleep(self._settle)

            # ── Servo ────────────────────────────────────────────────────
            if not self._servo_to(servo_pos, dt):
                log.error("Servo timeout at waypoint %d (%s)", idx, target.description)
                return False

            # ── After last LIFT point: verify cube elevation ──────────────
            is_last_lift = (
                target.action_type == ActionType.LIFT
                and (idx + 1 >= len(trajectory)
                     or trajectory[idx + 1].action_type != ActionType.LIFT)
            )
            if is_last_lift and cube_bid >= 0 and last_grasp_pos is not None:
                self._run_grasp_feedback(
                    cube_bid              = cube_bid,
                    env                   = env,
                    dt                    = dt,
                    grip_aid              = _grip_aid,
                    gripperframe_x_offset = _GRIPPERFRAME_X_OFFSET,
                )

        log.info("Execution complete ✓")
        return True

    def _run_grasp_feedback(
        self,
        cube_bid:             int,
        env,
        dt:                   float,
        grip_aid:             int,
        gripperframe_x_offset: float,
    ) -> None:
        """
        Check cube elevation after LIFT; if failed, retry the full grasp sequence
        using a fresh cube GT position at the start of each attempt.

        Retry sequence per attempt (no Gemini re-call):
          1. Open gripper
          2. Servo to approach  (fresh_cube_xy, _RETRY_APPROACH_Z)
          3. Servo to pre-grasp (fresh_cube_xy, _RETRY_PRE_GRASP_Z)
          4. Servo to hover     (grasp_xy,      _GRASP_HOVER_Z)
          5. Align wrist_roll
          6. Contact descent    (_contact_descend)
          7. Close gripper + settle
          8. Servo to lift      (fresh_cube_xy, _RETRY_LIFT_Z)
          9. Check cube_z
        """
        import numpy as _np

        def _cube_z() -> float:
            return float(env.data.xpos[cube_bid][2])

        cube_z = _cube_z()
        if cube_z > _LIFT_SUCCESS_Z:
            log.info("[GraspCheck] GRASP SUCCESS — cube at z=%.3f m", cube_z)
            return

        log.warning(
            "[GraspCheck] GRASP FAILED — cube at z=%.3f m (< %.2f m), "
            "initiating up to %d retries",
            cube_z, _LIFT_SUCCESS_Z, _MAX_GRASP_RETRIES,
        )

        for attempt in range(1, _MAX_GRASP_RETRIES + 1):
            # Read fresh cube GT position — cube may have been pushed during prior attempt.
            cube_pos = env.data.xpos[cube_bid].copy()
            cube_xy  = cube_pos[:2].copy()
            grasp_xy = cube_xy.copy()
            grasp_xy[0] += gripperframe_x_offset
            log.warning(
                "[GraspCheck] retry %d/%d — cube at (%.4f, %.4f) — re-targeting all waypoints",
                attempt, _MAX_GRASP_RETRIES, cube_pos[0], cube_pos[1],
            )

            # 1. Open gripper.
            log.info("[GraspCheck] retry %d — set_gripper(0.0)  ctrl[%d]=%.4f",
                     attempt, grip_aid, float(env.data.ctrl[grip_aid]))
            self._robot.set_gripper(0.0)
            time.sleep(self._settle)

            # 2. Approach.
            approach_pos = _np.array([cube_xy[0], cube_xy[1], _RETRY_APPROACH_Z])
            log.info("[GraspCheck] retry %d — approach (%.4f, %.4f, %.4f)",
                     attempt, *approach_pos)
            self._servo_to(approach_pos, dt)

            # 3. Pre-grasp hover.
            pre_grasp_pos = _np.array([cube_xy[0], cube_xy[1], _RETRY_PRE_GRASP_Z])
            log.info("[GraspCheck] retry %d — pre_grasp (%.4f, %.4f, %.4f)",
                     attempt, *pre_grasp_pos)
            self._servo_to(pre_grasp_pos, dt)

            # 4. Hover directly above grasp XY.
            hover_pos = _np.array([grasp_xy[0], grasp_xy[1], _GRASP_HOVER_Z])
            log.info("[GraspCheck] retry %d — hover (%.4f, %.4f, %.4f)",
                     attempt, *hover_pos)
            self._servo_to(hover_pos, dt)

            # 5. Align wrist_roll.
            self._robot.set_wrist_roll(_GRASP_WRIST_ROLL_RAD)

            # 6. Contact descent.
            log.info("[GraspCheck] retry %d — contact descent  ctrl[%d]=%.4f",
                     attempt, grip_aid, float(env.data.ctrl[grip_aid]))
            self._contact_descend(grasp_xy, grip_aid, cube_bid=cube_bid)

            # 7. Close gripper and settle.
            log.info("[GraspCheck] retry %d — set_gripper(1.0)  ctrl[%d] before=%.4f",
                     attempt, grip_aid, float(env.data.ctrl[grip_aid]))
            self._robot.set_gripper(1.0)
            log.info("[GraspCheck] retry %d — set_gripper done  ctrl[%d] after=%.4f  settling %d steps",
                     attempt, grip_aid, float(env.data.ctrl[grip_aid]), _GRASP_POST_CLOSE_STEPS)
            time.sleep(self._settle)
            for _ in range(_GRASP_POST_CLOSE_STEPS):
                self._sim_step()

            # 8. Lift.
            lift_pos = _np.array([cube_xy[0], cube_xy[1], _RETRY_LIFT_Z])
            log.info("[GraspCheck] retry %d — lift (%.4f, %.4f, %.4f)",
                     attempt, *lift_pos)
            self._servo_to(lift_pos, dt)

            # 9. Check result.
            cube_z = _cube_z()
            if cube_z > _LIFT_SUCCESS_Z:
                log.info(
                    "[GraspCheck] GRASP SUCCESS after retry %d — cube at z=%.3f m",
                    attempt, cube_z,
                )
                return

            log.warning(
                "[GraspCheck] retry %d/%d failed — cube still at z=%.3f m",
                attempt, _MAX_GRASP_RETRIES, cube_z,
            )

        log.warning(
            "[GraspCheck] all %d retries exhausted — cube at z=%.3f m, "
            "proceeding with pipeline",
            _MAX_GRASP_RETRIES, _cube_z(),
        )


class SimVLAPipeline(VLAPipeline):
    """
    Drop-in replacement for VLAPipeline that uses simulation-appropriate
    plausibility thresholds and Gemini prompt/image enhancements.
    """

    def __init__(self, config, streamer=None, robot=None, sim_step_fn=None):
        super().__init__(config, streamer=streamer, robot=robot,
                         sim_step_fn=sim_step_fn)
        # Replace execute_stage with sim-specific version (longer timeout).
        self.execute_stage = _SimExecuteStage(
            config, robot=robot, sim_step_fn=sim_step_fn,
        )
        log.info(
            "[Sim] SimExecuteStage active — class=%s  max_iters=%d  sim_timeout=%.0f s",
            type(self.execute_stage).__name__,
            _SIM_SERVO_MAX_ITERS,
            _SIM_SERVO_MAX_ITERS / config.control_frequency,
        )
        # Replace plan_stage with sim-aware wrapper if we have an env reference.
        env          = getattr(streamer, "_env",        None)
        debug_bodies = getattr(streamer, "_debug_bods", None)
        if env is not None:
            self.plan_stage = _SimAnnotatingPlanStage(
                self.plan_stage, env, config, debug_bodies,
            )
            log.info(
                "[SimPlan] plan_stage wrapped with annotation for bodies: %s",
                debug_bodies,
            )

    @staticmethod
    def _check_positions(positions: List[Optional[Point3D]]) -> bool:
        """
        Accept points with z >= -0.02 m (2 cm tolerance for renderer noise)
        and distance <= 1.5 m.  Logs every projected point for debugging.
        """
        valid = [p for p in positions if p is not None]
        log.info(
            "[Stage2] %d/%d waypoints projected (non-None)",
            len(valid), len(positions),
        )
        for i, p in enumerate(positions):
            if p is None:
                log.info("  wp[%d]  None (projection failed)", i)
            else:
                dist = (p.x ** 2 + p.y ** 2 + p.z ** 2) ** 0.5
                ok = p.z >= _SIM_MIN_Z and dist <= _SIM_MAX_DIST
                log.info(
                    "  wp[%d]  x=%.4f  y=%.4f  z=%.4f  dist=%.4f  %s",
                    i, p.x, p.y, p.z, dist, "OK" if ok else "FAIL",
                )
        if not valid:
            return False
        for p in valid:
            dist = (p.x ** 2 + p.y ** 2 + p.z ** 2) ** 0.5
            if p.z < _SIM_MIN_Z or dist > _SIM_MAX_DIST:
                return False
        return True
