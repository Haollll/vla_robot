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
2. Smooth  — cubic spline interpolation between the expanded keyframes
             produces `interpolation_steps` points per segment.

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

log = logging.getLogger(__name__)


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
    interpolation_steps : Dense points inserted between each pair of
                          keyframes by the spline smoother.
    """

    def __init__(
        self,
        offsets:             ActionOffsets,
        interpolation_steps: int = 50,
    ) -> None:
        self._off   = offsets
        self._steps = max(2, interpolation_steps)

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
            base = p3d.as_array()
            expanded = self._expand(base, wp.action_type, wp.gripper_state, wp.description)
            keyframes.extend(expanded)

        log.info("Expanded to %d keyframe(s)", len(keyframes))
        return keyframes

    def interpolate(self, keyframes: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        Cubic spline smooth between keyframes.

        Gripper state is assigned by nearest-keyframe (step, not lerp).
        Returns the original keyframes unchanged if fewer than 2 exist.
        """
        if len(keyframes) < 2:
            log.warning("Too few keyframes for interpolation (%d), returning as-is", len(keyframes))
            return keyframes

        positions = np.array([kf.position for kf in keyframes], dtype=np.float64)
        n = len(positions)

        # Parameterise by cumulative Euclidean arc length for uniform speed
        dists    = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        t_knots  = np.concatenate([[0.0], np.cumsum(dists)])
        # Guard against zero-length segments
        if t_knots[-1] < 1e-9:
            return keyframes

        t_knots /= t_knots[-1]   # normalise to [0, 1]

        # Deduplicate: CubicSpline requires strictly increasing knots.
        # Identical positions (e.g. grasp descent + gripper close at the same
        # spot) produce duplicate t values after normalisation; nudge each
        # duplicate forward by a tiny epsilon so the sequence is strict.
        eps = np.finfo(float).eps * 1024  # ~2.3e-13, negligible in [0,1]
        for i in range(1, len(t_knots)):
            if t_knots[i] <= t_knots[i - 1]:
                t_knots[i] = t_knots[i - 1] + eps

        try:
            from scipy.interpolate import CubicSpline
            cs_x = CubicSpline(t_knots, positions[:, 0], bc_type="natural")
            cs_y = CubicSpline(t_knots, positions[:, 1], bc_type="natural")
            cs_z = CubicSpline(t_knots, positions[:, 2], bc_type="natural")
        except ImportError:
            log.warning("scipy not installed — using linear interpolation (install scipy for cubic spline)")
            # Linear fallback: wrap numpy interp as callables
            def _lin(t_out, t_in, y_in):
                return np.interp(t_out, t_in, y_in)
            cs_x = lambda t, _t=t_knots, _y=positions[:, 0]: _lin(t, _t, _y)  # noqa: E731
            cs_y = lambda t, _t=t_knots, _y=positions[:, 1]: _lin(t, _t, _y)  # noqa: E731
            cs_z = lambda t, _t=t_knots, _y=positions[:, 2]: _lin(t, _t, _y)  # noqa: E731

        # Build dense parameter vector; ensure endpoints match exactly
        segments   = n - 1
        total_pts  = segments * self._steps + 1
        t_fine     = np.linspace(0.0, 1.0, total_pts)

        smooth_pos = np.column_stack([cs_x(t_fine), cs_y(t_fine), cs_z(t_fine)])

        # Assign metadata from nearest keyframe
        trajectory: List[TrajectoryPoint] = []
        for i, pos in enumerate(smooth_pos):
            frac = float(i) / max(total_pts - 1, 1)
            idx  = int(round(frac * (n - 1)))
            src  = keyframes[idx]
            trajectory.append(
                TrajectoryPoint(
                    position    = pos,
                    gripper     = src.gripper,
                    action_type = src.action_type,
                    description = src.description,
                )
            )

        log.info(
            "Interpolated %d keyframes → %d trajectory points",
            len(keyframes), len(trajectory),
        )
        return trajectory

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
            # Single point: above the target at safety height, gripper open
            return [pt([x, y, z + o.safety_height], 0.0, "approach")]

        if action_type == ActionType.PRE_GRASP:
            # Hover just above the object, gripper open
            return [pt([x, y, z + o.pre_grasp_height], 0.0, "pre-grasp hover")]

        if action_type == ActionType.GRASP:
            # 1) Descend to contact height
            # 2) Close gripper (same position)
            return [
                pt([x, y, z + o.grasp_descent], 0.0, "grasp descent"),
                pt([x, y, z + o.grasp_descent], 1.0, "gripper close"),
            ]

        if action_type == ActionType.LIFT:
            # Rise to carry height with gripper closed
            return [pt([x, y, z + o.lift_height], 1.0, "lift")]

        if action_type == ActionType.MOVE:
            # Translate at safety height, gripper stays as-is
            return [pt([x, y, z + o.safety_height], gripper_state, "move")]

        if action_type == ActionType.PLACE:
            # 1) Lower to hover above surface
            # 2) Release gripper (same position)
            return [
                pt([x, y, z + o.place_height], 1.0, "place descent"),
                pt([x, y, z + o.place_height], 0.0, "gripper release"),
            ]

        if action_type == ActionType.RETREAT:
            # Rise clear, gripper open
            return [pt([x, y, z + o.retreat_height], 0.0, "retreat")]

        if action_type == ActionType.HOME:
            return [pt([x, y, z], 0.0, "home")]

        log.warning("Unknown ActionType '%s', passing through unchanged", action_type)
        return [pt([x, y, z], gripper_state)]
