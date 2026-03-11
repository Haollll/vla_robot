"""
Trajectory Builder  —  Stage 3
================================
Converts semantic 3-D waypoints into a dense, smooth Cartesian trajectory
ready for PID tracking.

Two-pass algorithm
------------------
1. Expand  — each SemanticWaypoint becomes 1–2 TrajectoryPoints after
             applying per-action geometric offsets (safety heights, pre-
             grasp descent, place hover, etc.).
2. Smooth  — hybrid interpolation between consecutive keyframe pairs:

   Sinusoidal velocity profile (straight line, smooth speed modulation):
     Used for: APPROACH, MOVE, LIFT, RETREAT  (when dist ≥ 0.08 m)
     Provided by SO101Kinematics.generate_sinusoidal_velocity_trajectory()

   Cubic spline (or linear fallback if scipy absent):
     Used for: PRE_GRASP, GRASP, PLACE — and any segment shorter than 0.08 m

Gripper transitions
-------------------
Gripper commands are *not* interpolated; they are stepped at the first
TrajectoryPoint of each segment where the gripper state changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import ActionType, ActionOffsets
from ..planner.gemini_planner import SemanticWaypoint
from ..projection.depth_projection import Point3D
from ..control.so101_kinematics import SO101Kinematics

log = logging.getLogger(__name__)

# Action types that use sinusoidal velocity profile (when segment is long enough)
_SINUSOIDAL_ACTIONS  = frozenset({
    ActionType.APPROACH, ActionType.MOVE, ActionType.LIFT, ActionType.RETREAT
})
_SHORT_SEG_THRESHOLD = 0.08   # m — segments shorter than this always use spline
_AVG_CART_SPEED      = 0.10   # m/s — used to compute total_time from distance


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    position:    np.ndarray   # [x, y, z] in robot base frame  [m]
    gripper:     float        # 0.0 = open, 1.0 = closed
    action_type: ActionType
    description: str = ""

    def __repr__(self) -> str:
        p = self.position
        return (
            f"TrajectoryPoint({self.action_type.value}, "
            f"pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}], "
            f"grip={self.gripper:.1f})"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class TrajectoryBuilder:
    """
    Parameters
    ----------
    offsets             : ActionOffsets dataclass with per-action heights.
    interpolation_steps : Dense points per segment for spline segments.
    control_freq        : Control-loop frequency [Hz] for sinusoidal segments.
    """

    def __init__(
        self,
        offsets:             ActionOffsets,
        interpolation_steps: int   = 50,
        control_freq:        float = 50.0,
    ) -> None:
        self._off          = offsets
        self._steps        = max(2, interpolation_steps)
        self._control_freq = control_freq
        self._kin          = SO101Kinematics()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        waypoints:  List[SemanticWaypoint],
        positions:  List[Optional[Point3D]],
    ) -> List[TrajectoryPoint]:
        """
        Expand semantic waypoints → geometric keyframes.

        Parameters
        ----------
        waypoints : Planner output (action types + pixel coords).
        positions : Corresponding 3-D robot-frame positions; None entries
                    are skipped with a warning.

        Returns
        -------
        Ordered list of TrajectoryPoint keyframes (pre-interpolation).
        """
        if len(waypoints) != len(positions):
            raise ValueError(
                f"waypoints/positions length mismatch: "
                f"{len(waypoints)} vs {len(positions)}"
            )

        keyframes: List[TrajectoryPoint] = []
        for wp, p3d in zip(waypoints, positions):
            if p3d is None:
                log.warning(
                    "Skipping %s @ %s — no valid depth projection",
                    wp.action_type.value, wp.pixel_coords,
                )
                continue
            base     = p3d.as_array()
            expanded = self._expand(base, wp.action_type, wp.gripper_state, wp.description)
            keyframes.extend(expanded)

        log.info("Expanded to %d keyframe(s)", len(keyframes))
        return keyframes

    def interpolate(self, keyframes: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        Hybrid interpolation: sinusoidal velocity profile for long transit
        segments, cubic spline for precision segments and short moves.

        Each consecutive keyframe pair is processed independently:
          - APPROACH / MOVE / LIFT / RETREAT  AND  dist ≥ 0.08 m
              → sinusoidal velocity trajectory (straight line)
          - otherwise (PRE_GRASP / GRASP / PLACE, or short segment)
              → cubic spline  (or linear fallback without scipy)

        Returns the original keyframes unchanged if fewer than 2 exist.
        """
        if len(keyframes) < 2:
            log.warning("Too few keyframes (%d), returning as-is", len(keyframes))
            return keyframes

        result: List[TrajectoryPoint] = []

        for i in range(len(keyframes) - 1):
            start_kf = keyframes[i]
            end_kf   = keyframes[i + 1]
            dist     = float(np.linalg.norm(end_kf.position - start_kf.position))

            use_sin = (
                start_kf.action_type in _SINUSOIDAL_ACTIONS
                and dist >= _SHORT_SEG_THRESHOLD
            )

            if use_sin:
                pts = self._sinusoidal_segment(start_kf, end_kf, dist)
            else:
                pts = self._spline_segment(start_kf, end_kf)

            result.extend(pts)

        # Append the final keyframe
        result.append(keyframes[-1])

        log.info(
            "Interpolated %d keyframes → %d trajectory points",
            len(keyframes), len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Segment interpolators
    # ------------------------------------------------------------------

    def _sinusoidal_segment(
        self,
        start_kf: TrajectoryPoint,
        end_kf:   TrajectoryPoint,
        dist:     float,
    ) -> List[TrajectoryPoint]:
        """Straight-line segment with sinusoidal velocity profile."""
        if dist < 1e-9:
            return [start_kf]

        # Compute total time from distance and average speed; clamp to [1, 10] s
        total_time = float(np.clip(dist / _AVG_CART_SPEED, 1.0, 10.0))

        traj_pts, _, _ = self._kin.generate_sinusoidal_velocity_trajectory(
            start_point        = start_kf.position,
            end_point          = end_kf.position,
            control_freq       = self._control_freq,
            total_time         = total_time,
            velocity_amplitude = 0.3 * dist / total_time,
            velocity_period    = max(1.0, total_time / 2.0),
        )

        # Exclude last point — it becomes the start of the next segment
        pts: List[TrajectoryPoint] = []
        for pos in traj_pts[:-1]:
            pts.append(TrajectoryPoint(
                position    = pos,
                gripper     = start_kf.gripper,
                action_type = start_kf.action_type,
                description = start_kf.description,
            ))
        return pts

    def _spline_segment(
        self,
        start_kf: TrajectoryPoint,
        end_kf:   TrajectoryPoint,
    ) -> List[TrajectoryPoint]:
        """Cubic spline (or linear fallback) between two keyframes."""
        positions = np.array([start_kf.position, end_kf.position], dtype=np.float64)
        t_knots   = np.array([0.0, 1.0])

        # Exclude last point (t=1.0) — appended by the next iteration or final append
        t_fine = np.linspace(0.0, 1.0, self._steps + 1)[:-1]

        try:
            from scipy.interpolate import CubicSpline
            smooth = np.column_stack([
                CubicSpline(t_knots, positions[:, c], bc_type="natural")(t_fine)
                for c in range(3)
            ])
        except ImportError:
            log.warning("scipy not installed — using linear interpolation")
            smooth = np.column_stack([
                np.interp(t_fine, t_knots, positions[:, c]) for c in range(3)
            ])

        pts: List[TrajectoryPoint] = []
        for pos in smooth:
            pts.append(TrajectoryPoint(
                position    = pos,
                gripper     = start_kf.gripper,
                action_type = start_kf.action_type,
                description = start_kf.description,
            ))
        return pts

    # ------------------------------------------------------------------
    # Geometric offset expansion  (one waypoint → 1 or 2 keyframes)
    # ------------------------------------------------------------------

    def _expand(
        self,
        base:          np.ndarray,
        action_type:   ActionType,
        gripper_state: float,
        description:   str,
    ) -> List[TrajectoryPoint]:
        o  = self._off
        x, y, z = base

        def pt(pos, grip, desc_suffix="") -> TrajectoryPoint:
            return TrajectoryPoint(
                position    = np.array(pos, dtype=np.float64),
                gripper     = grip,
                action_type = action_type,
                description = f"{description} — {desc_suffix}" if desc_suffix else description,
            )

        if action_type == ActionType.APPROACH:
            return [pt([x, y, z + o.safety_height], 0.0, "approach")]

        if action_type == ActionType.PRE_GRASP:
            return [pt([x, y, z + o.pre_grasp_height], 0.0, "pre-grasp hover")]

        if action_type == ActionType.GRASP:
            return [
                pt([x, y, z + o.grasp_descent], 0.0, "grasp descent"),
                pt([x, y, z + o.grasp_descent], 1.0, "gripper close"),
            ]

        if action_type == ActionType.LIFT:
            return [pt([x, y, z + o.lift_height], 1.0, "lift")]

        if action_type == ActionType.MOVE:
            return [pt([x, y, z + o.safety_height], gripper_state, "move")]

        if action_type == ActionType.PLACE:
            return [
                pt([x, y, z + o.place_height], 1.0, "place descent"),
                pt([x, y, z + o.place_height], 0.0, "gripper release"),
            ]

        if action_type == ActionType.RETREAT:
            return [pt([x, y, z + o.retreat_height], 0.0, "retreat")]

        if action_type == ActionType.HOME:
            return [pt([x, y, z], 0.0, "home")]

        log.warning("Unknown ActionType '%s', passing through unchanged", action_type)
        return [pt([x, y, z], gripper_state)]
