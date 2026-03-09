"""
3-D Projection  —  Stage 2
===========================
Back-projects 2-D pixel coordinates to metric 3-D points using the
depth map and camera intrinsics, then transforms to the robot base frame
via the camera extrinsic calibration.

Pinhole model
-------------
  X_cam = (u - cx) * depth / fx
  Y_cam = (v - cy) * depth / fy
  Z_cam = depth

  p_robot = T_cam→robot  @  [X_cam, Y_cam, Z_cam, 1]ᵀ

Depth conventions
-----------------
  float32  → assumed metres  (e.g. RealSense with depth scale = 0.001)
  uint16   → assumed millimetres; converted to metres automatically.
  Zero / NaN values are treated as missing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import CameraIntrinsics, CameraExtrinsics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def __repr__(self) -> str:
        return f"Point3D(x={self.x:.4f}, y={self.y:.4f}, z={self.z:.4f})"


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class DepthProjector:
    """
    Converts image-space (u, v) + depth map → 3-D robot-frame Point3D.

    Parameters
    ----------
    intrinsics  : Camera calibration (fx, fy, cx, cy).
    extrinsics  : 4×4 T_cam→robot rigid transform.
    sample_k    : Half-width of the median depth sampling kernel.
                  Depth is taken as median of a (2k+1)² patch to suppress
                  noise and small holes.
    depth_min_m : Points closer than this are rejected [metres].
    depth_max_m : Points farther than this are rejected [metres].
    """

    def __init__(
        self,
        intrinsics:   CameraIntrinsics,
        extrinsics:   CameraExtrinsics,
        sample_k:     int   = 3,
        depth_min_m:  float = 0.05,
        depth_max_m:  float = 2.00,
    ) -> None:
        self._K    = intrinsics
        self._T    = extrinsics.T           # (4,4) float64
        self._k    = sample_k
        self._dmin = depth_min_m
        self._dmax = depth_max_m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        u: int,
        v: int,
        depth_map: np.ndarray,
    ) -> Optional[Point3D]:
        """
        Full pipeline: (u, v) → camera frame → robot base frame.

        Returns None if depth is missing or out of valid range.
        """
        depth_m = self._sample_depth(u, v, depth_map)
        if depth_m is None:
            return None

        p_cam  = self._backproject(u, v, depth_m)
        p_robot = self._to_robot_frame(p_cam)
        return p_robot

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------

    def _sample_depth(self, u: int, v: int, depth_map: np.ndarray) -> Optional[float]:
        """Median-filtered depth sample at pixel (u, v)."""
        # Convert uint16 mm → float32 m
        if depth_map.dtype == np.uint16:
            depth_m = depth_map.astype(np.float32) / 1000.0
        else:
            depth_m = depth_map.astype(np.float32)

        H, W = depth_m.shape[:2]
        k = self._k
        u0, u1 = max(0, u - k), min(W, u + k + 1)
        v0, v1 = max(0, v - k), min(H, v + k + 1)

        patch  = depth_m[v0:v1, u0:u1].ravel()
        valid  = patch[(patch >= self._dmin) & (patch <= self._dmax)]

        if valid.size == 0:
            log.debug("No valid depth near pixel (%d, %d)", u, v)
            return None

        return float(np.median(valid))

    def _backproject(self, u: int, v: int, depth_m: float) -> np.ndarray:
        """Pixel + depth → (X, Y, Z, 1) in camera frame."""
        K  = self._K
        X  = (u - K.cx) * depth_m / K.fx
        Y  = (v - K.cy) * depth_m / K.fy
        Z  = depth_m
        return np.array([X, Y, Z, 1.0], dtype=np.float64)

    def _to_robot_frame(self, p_hom: np.ndarray) -> Point3D:
        """Apply T_cam→robot homogeneous transform."""
        p = self._T @ p_hom
        return Point3D(x=float(p[0]), y=float(p[1]), z=float(p[2]))

    # ------------------------------------------------------------------
    # Batch helper
    # ------------------------------------------------------------------

    def project_batch(
        self,
        pixels:    list[Tuple[int, int]],
        depth_map: np.ndarray,
    ) -> list[Optional[Point3D]]:
        """Project a list of (u, v) tuples in one call."""
        return [self.project(u, v, depth_map) for u, v in pixels]
