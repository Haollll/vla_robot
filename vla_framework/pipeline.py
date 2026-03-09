"""
VLA Pipeline  —  Orchestrator
==============================
Wires the four stages together and exposes a single `.run()` entry point.

Stage flow
----------
  RGB + depth + command
        │
        ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 1. PLAN       GeminiPlanner                             │
  │               (RGB + text) → List[SemanticWaypoint]     │
  │               action_type + (u, v) pixel coords         │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. PROJECT    DepthProjector                            │
  │               (u,v) + depth → back-project →            │
  │               camera frame → robot base frame           │
  │               → List[Optional[Point3D]]                 │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 3. BUILD      TrajectoryBuilder                         │
  │               geometric offsets per action type +       │
  │               cubic spline interpolation                │
  │               → List[TrajectoryPoint]  (dense)          │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 4. EXECUTE    CartesianPIDController + LeRobotInterface  │
  │               closed-loop servo to each waypoint        │
  │               gripper state stepped at transitions      │
  └─────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np

from .config import VLAConfig
from .planner.gemini_planner      import GeminiPlanner, SemanticWaypoint
from .projection.depth_projection import DepthProjector, Point3D
from .path.trajectory_builder     import TrajectoryBuilder, TrajectoryPoint
from .control.pid_controller      import CartesianPIDController
from .control.lerobot_interface   import LeRobotInterface

log = logging.getLogger(__name__)


class VLAPipeline:
    """
    End-to-end hierarchical VLA system for a robotic arm.

    Usage
    -----
    >>> pipeline = VLAPipeline(config)
    >>> with pipeline:
    ...     success = pipeline.run(rgb, depth, "pick up the red cube")

    Or without context manager:
    >>> pipeline.connect()
    >>> success = pipeline.run(rgb, depth, command)
    >>> pipeline.disconnect()
    """

    def __init__(self, config: VLAConfig) -> None:
        self.cfg = config

        self.planner = GeminiPlanner(
            api_key    = config.gemini_api_key,
            model_name = config.gemini_model,
        )
        self.projector = DepthProjector(
            intrinsics = config.camera_intrinsics,
            extrinsics = config.camera_extrinsics,
        )
        self.traj_builder = TrajectoryBuilder(
            offsets             = config.action_offsets,
            interpolation_steps = config.interpolation_steps,
        )
        dt = 1.0 / config.control_frequency
        self.pid   = CartesianPIDController(config.pid_gains, dt)
        self.robot = LeRobotInterface(
            robot_type = config.robot_type,
            port       = config.robot_port,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self.robot.connect()

    def disconnect(self) -> None:
        self.robot.disconnect()

    def __enter__(self) -> "VLAPipeline":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Stage 1 — Plan
    # ------------------------------------------------------------------

    def plan(self, rgb_image: np.ndarray, command: str) -> List[SemanticWaypoint]:
        """Call Gemini to produce semantic 2-D waypoints."""
        log.info("── Stage 1: PLAN  command='%s'", command)
        waypoints = self.planner.plan(rgb_image, command)
        if not waypoints:
            log.error("Planner returned 0 waypoints")
        return waypoints

    # ------------------------------------------------------------------
    # Stage 2 — Project
    # ------------------------------------------------------------------

    def project_waypoints(
        self,
        waypoints:   List[SemanticWaypoint],
        depth_image: np.ndarray,
    ) -> List[Optional[Point3D]]:
        """Back-project each semantic waypoint to a 3-D robot-frame point."""
        log.info("── Stage 2: PROJECT  %d waypoints", len(waypoints))
        positions: List[Optional[Point3D]] = []
        for wp in waypoints:
            u, v = wp.pixel_coords
            p3d  = self.projector.project(u, v, depth_image)
            if p3d is not None:
                log.debug(
                    "  %-10s (%3d,%3d) → robot %s",
                    wp.action_type.value, u, v, p3d,
                )
            positions.append(p3d)

        valid = sum(1 for p in positions if p is not None)
        log.info("  %d/%d waypoints projected successfully", valid, len(waypoints))
        return positions

    # ------------------------------------------------------------------
    # Stage 3 — Build trajectory
    # ------------------------------------------------------------------

    def build_trajectory(
        self,
        waypoints: List[SemanticWaypoint],
        positions: List[Optional[Point3D]],
    ) -> List[TrajectoryPoint]:
        """Expand + smooth waypoints into a dense Cartesian trajectory."""
        log.info("── Stage 3: BUILD TRAJECTORY")
        keyframes  = self.traj_builder.build(waypoints, positions)
        trajectory = self.traj_builder.interpolate(keyframes)
        return trajectory

    # ------------------------------------------------------------------
    # Stage 4 — Execute
    # ------------------------------------------------------------------

    def execute(self, trajectory: List[TrajectoryPoint]) -> bool:
        """
        PID closed-loop servo along the trajectory.

        For each point:
          1. Step gripper if state changed.
          2. Servo EE to target position via CartesianPID until within
             `waypoint_tolerance` metres.

        Returns True on success, False if any servo step times out.
        """
        log.info("── Stage 4: EXECUTE  %d points", len(trajectory))
        self.pid.reset()
        dt        = self.pid.dt
        tol       = self.cfg.waypoint_tolerance
        settle    = self.cfg.gripper_settle_s
        prev_grip = -1.0          # force gripper update on first point

        for idx, target in enumerate(trajectory):
            # ── Gripper ──────────────────────────────────────────────
            if abs(target.gripper - prev_grip) > 0.05:
                log.info(
                    "  [%4d] Gripper → %.0f  (%s)",
                    idx, target.gripper, target.description,
                )
                self.robot.set_gripper(target.gripper)
                prev_grip = target.gripper
                time.sleep(settle)

            # ── Servo ────────────────────────────────────────────────
            if not self._servo_to(target.position, dt, tol):
                log.error(
                    "Servo timeout at waypoint %d (%s)", idx, target.description
                )
                return False

        log.info("Execution complete ✓")
        return True

    def _servo_to(
        self,
        target:    np.ndarray,
        dt:        float,
        tolerance: float,
        max_iters: int = 500,
    ) -> bool:
        """
        Inner PID loop: drive EE to `target` [m] within `tolerance` [m].

        Returns True when converged, False on timeout.
        """
        for _ in range(max_iters):
            state = self.robot.get_state()
            error = target - state.end_effector_pos
            dist  = float(np.linalg.norm(error))

            if dist < tolerance:
                return True

            velocity_cmd = self.pid.step(error)
            self.robot.send_cartesian_velocity(velocity_cmd)
            time.sleep(dt)

        state = self.robot.get_state()
        remaining = float(np.linalg.norm(target - state.end_effector_pos))
        log.warning("Servo timeout  remaining_error=%.4f m", remaining)
        return False

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        rgb_image:   np.ndarray,
        depth_image: np.ndarray,
        command:     str,
    ) -> bool:
        """
        Execute the complete VLA pipeline.

        Parameters
        ----------
        rgb_image   : uint8  (H, W, 3) — RGB frame from camera.
        depth_image : float32 (H, W)   — depth in metres
                      OR uint16 (H, W) — depth in millimetres (auto-detected).
        command     : Natural language task description.

        Returns
        -------
        True  — task completed successfully.
        False — one or more stages failed.
        """
        sep = "═" * 60
        log.info(sep)
        log.info("VLA PIPELINE  |  %s", command)
        log.info(sep)

        # 1. Plan
        waypoints = self.plan(rgb_image, command)
        if not waypoints:
            return False

        # 2. Project
        positions = self.project_waypoints(waypoints, depth_image)
        if all(p is None for p in positions):
            log.error("All waypoints failed depth projection — aborting")
            return False

        # 3. Build
        trajectory = self.build_trajectory(waypoints, positions)
        if not trajectory:
            log.error("Empty trajectory — aborting")
            return False

        # 4. Execute
        return self.execute(trajectory)
