"""
VLA Pipeline — Stage Classes + Orchestrator
============================================
Four single-responsibility stage classes wired by VLAPipeline.

  PlanStage     rgb + command   → List[SemanticWaypoint]
  ProjectStage  waypoints+depth → List[Optional[Point3D]]
  BuildStage    waypoints+pos   → List[TrajectoryPoint]
  ExecuteStage  trajectory      → bool  (PID servo + gripper)

VLAPipeline adds:
  - plausibility check after projection (z ≥ 0, distance ≤ 1.5 m)
  - one automatic retry of planning+projection if the check fails
  - lifecycle (connect / disconnect) delegated to ExecuteStage
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from stereo_depth import CameraStreamer

from .config import VLAConfig
from .planner.gemini_planner      import GeminiPlanner, SemanticWaypoint
from .projection.depth_projection import DepthProjector, Point3D
from .path.trajectory_builder     import TrajectoryBuilder, TrajectoryPoint
from .control.pid_controller      import CartesianPIDController
from .control.lerobot_interface   import LeRobotInterface

log = logging.getLogger(__name__)

# Plausibility bounds applied after depth projection
_MIN_Z    = 0.0   # metres — negative z is below the robot base plane
_MAX_DIST = 1.5   # metres — farther than this is almost certainly a bad reading


# ---------------------------------------------------------------------------
# Stage 1 — Plan
# ---------------------------------------------------------------------------

class PlanStage:
    """Calls GeminiPlanner to produce semantic 2-D waypoints."""

    def __init__(self, config: VLAConfig) -> None:
        self._planner = GeminiPlanner(
            api_key    = config.gemini_api_key,
            model_name = config.gemini_model,
        )

    def run(self, rgb: np.ndarray, command: str) -> List[SemanticWaypoint]:
        log.info("── Stage 1: PLAN  command='%s'", command)
        waypoints = self._planner.plan(rgb, command)
        if not waypoints:
            log.error("Planner returned 0 waypoints")
        return waypoints


# ---------------------------------------------------------------------------
# Stage 2 — Project
# ---------------------------------------------------------------------------

class ProjectStage:
    """Back-projects each semantic waypoint to a 3-D robot-frame point."""

    def __init__(self, config: VLAConfig) -> None:
        self._projector = DepthProjector(
            intrinsics = config.camera_intrinsics,
            extrinsics = config.camera_extrinsics,
        )

    def run(
        self,
        waypoints:   List[SemanticWaypoint],
        depth_image: np.ndarray,
    ) -> List[Optional[Point3D]]:
        log.info("── Stage 2: PROJECT  %d waypoints", len(waypoints))
        positions: List[Optional[Point3D]] = []
        for wp in waypoints:
            u, v = wp.pixel_coords
            p3d  = self._projector.project(u, v, depth_image)
            if p3d is not None:
                log.debug(
                    "  %-10s (%3d,%3d) → robot (%.3f, %.3f, %.3f)",
                    wp.action_type.value, u, v, p3d.x, p3d.y, p3d.z,
                )
            positions.append(p3d)
        valid = sum(1 for p in positions if p is not None)
        log.info("  %d/%d waypoints projected successfully", valid, len(waypoints))
        return positions


# ---------------------------------------------------------------------------
# Stage 3 — Build trajectory
# ---------------------------------------------------------------------------

class BuildStage:
    """Expands waypoints + positions into a dense Cartesian trajectory."""

    def __init__(self, config: VLAConfig) -> None:
        self._builder = TrajectoryBuilder(
            offsets             = config.action_offsets,
            interpolation_steps = config.interpolation_steps,
            control_freq        = config.control_frequency,
        )

    def run(
        self,
        waypoints: List[SemanticWaypoint],
        positions: List[Optional[Point3D]],
    ) -> List[TrajectoryPoint]:
        log.info("── Stage 3: BUILD TRAJECTORY")
        keyframes  = self._builder.build(waypoints, positions)
        trajectory = self._builder.interpolate(keyframes)
        return trajectory


# ---------------------------------------------------------------------------
# Stage 4 — Execute
# ---------------------------------------------------------------------------

class ExecuteStage:
    """
    PID closed-loop servo along the trajectory.

    connect() / disconnect() manage the robot lifecycle.
    run() returns True on success, False if any servo step times out.
    """

    def __init__(self, config: VLAConfig) -> None:
        dt = 1.0 / config.control_frequency
        self._pid    = CartesianPIDController(config.pid_gains, dt)
        self._robot  = LeRobotInterface(
            robot_type = config.robot_type,
            port       = config.robot_port,
            strict     = config.robot_strict,
        )
        self._tol    = config.waypoint_tolerance
        self._settle = config.gripper_settle_s

    def connect(self) -> None:
        self._robot.connect()

    def disconnect(self) -> None:
        self._robot.disconnect()

    def run(self, trajectory: List[TrajectoryPoint]) -> bool:
        log.info("── Stage 4: EXECUTE  %d points", len(trajectory))
        self._pid.reset()
        dt        = self._pid.dt
        prev_grip = -1.0   # force gripper update on first point

        for idx, target in enumerate(trajectory):
            # ── Gripper ──────────────────────────────────────────────────
            if abs(target.gripper - prev_grip) > 0.05:
                log.info(
                    "  [%4d] Gripper → %.0f  (%s)",
                    idx, target.gripper, target.description,
                )
                self._robot.set_gripper(target.gripper)
                prev_grip = target.gripper
                time.sleep(self._settle)

            # ── Servo ────────────────────────────────────────────────────
            if not self._servo_to(target.position, dt):
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
        max_iters: int = 500,
    ) -> bool:
        for _ in range(max_iters):
            state = self._robot.get_state()
            error = target - state.end_effector_pos
            if float(np.linalg.norm(error)) < self._tol:
                return True
            self._robot.send_cartesian_velocity(self._pid.step(error))
            time.sleep(dt)
        remaining = float(np.linalg.norm(target - self._robot.get_state().end_effector_pos))
        log.warning("Servo timeout  remaining_error=%.4f m", remaining)
        return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class VLAPipeline:
    """
    Wires PlanStage → ProjectStage → BuildStage → ExecuteStage.

    Adds a plausibility check after projection and retries planning once
    if the check fails (z < 0 or distance > 1.5 m).

    Usage
    -----
    >>> pipeline = VLAPipeline(config)
    >>> with pipeline:                           # connect() / disconnect()
    ...     success = pipeline.run(command)      # live camera
    ...     # or:
    ...     success = pipeline.run_from_images(rgb, depth, command)
    """

    def __init__(self, config: VLAConfig, streamer: "CameraStreamer | None" = None) -> None:
        self.cfg      = config
        self.streamer = streamer

        self.plan_stage    = PlanStage(config)
        self.project_stage = ProjectStage(config)
        self.build_stage   = BuildStage(config)
        self.execute_stage = ExecuteStage(config)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self.execute_stage.connect()

    def disconnect(self) -> None:
        self.execute_stage.disconnect()

    def __enter__(self) -> "VLAPipeline":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Plausibility check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_positions(positions: List[Optional[Point3D]]) -> bool:
        """
        Return True iff at least one position is valid and every non-None
        position satisfies:
          z ≥ 0        — not below the robot base plane
          distance ≤ 1.5 m — not unrealistically far
        """
        valid = [p for p in positions if p is not None]
        if not valid:
            return False
        for p in valid:
            dist = (p.x**2 + p.y**2 + p.z**2) ** 0.5
            if p.z < _MIN_Z or dist > _MAX_DIST:
                log.warning(
                    "Plausibility fail  point=(%.3f, %.3f, %.3f)  dist=%.3f",
                    p.x, p.y, p.z, dist,
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Plan + project with one retry on plausibility failure
    # ------------------------------------------------------------------

    def _plan_and_project(
        self,
        rgb:     np.ndarray,
        depth:   np.ndarray,
        command: str,
    ) -> tuple[List[SemanticWaypoint] | None, List[Optional[Point3D]] | None]:
        """
        Runs PlanStage then ProjectStage, checks plausibility, and retries
        planning once if the check fails.
        Returns (waypoints, positions) or (None, None) on failure.
        """
        for attempt in range(2):
            waypoints = self.plan_stage.run(rgb, command)
            if not waypoints:
                return None, None

            positions = self.project_stage.run(waypoints, depth)
            if self._check_positions(positions):
                return waypoints, positions

            if attempt == 0:
                log.warning("Plausibility check failed — retrying planning (attempt 2/2)")

        log.error("Plausibility check failed after retry — aborting")
        return None, None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, command: str) -> bool:
        """
        Execute the complete pipeline using live camera input.
        Requires a CameraStreamer (raises RuntimeError if none was provided).
        Raises NotReadyError if the streamer buffer is not full yet.
        """
        if self.streamer is None:
            raise RuntimeError("No CameraStreamer provided — cannot run live pipeline.")
        result      = self.streamer.snapshot()
        rgb_image   = result.rgb_snapshot
        depth_image = result.stable_depth
        return self.run_from_images(rgb_image, depth_image, command)

    def run_from_images(
        self,
        rgb_image:   np.ndarray,
        depth_image: np.ndarray,
        command:     str,
        dry_run:     bool = False,
    ) -> bool:
        """
        Execute pipeline from static images.  Does not require a CameraStreamer.
        When dry_run=True the trajectory is logged but ExecuteStage is skipped.
        """
        sep = "═" * 60
        log.info(sep)
        log.info("VLA PIPELINE  |  %s%s", command, "  [DRY RUN]" if dry_run else "")
        log.info(sep)

        waypoints, positions = self._plan_and_project(rgb_image, depth_image, command)
        if waypoints is None:
            return False

        trajectory = self.build_stage.run(waypoints, positions)
        if not trajectory:
            log.error("Empty trajectory — aborting")
            return False

        if dry_run:
            log.info("Trajectory has %d points — skipping execution", len(trajectory))
            stride = max(1, len(trajectory) // 10)
            for i, tp in enumerate(trajectory[::stride]):
                log.info(
                    "  [%4d] %-10s  pos=%s  grip=%.1f",
                    i * stride, tp.action_type.value,
                    tp.position.round(4), tp.gripper,
                )
            return True

        return self.execute_stage.run(trajectory)
