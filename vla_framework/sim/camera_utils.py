"""
MuJoCo Camera Utilities
========================
Derive camera intrinsics and extrinsics from a live MuJoCo model/data
in the pinhole / OpenCV convention expected by DepthProjector.

Convention mapping
------------------
MuJoCo camera frame (cam_xmat rows = camera axes in world):
    X_mj = right,  Y_mj = down-in-image,  Z_mj = out-of-screen (away from scene)

OpenCV / DepthProjector pinhole convention:
    X_cv = right,  Y_cv = down-in-image, Z_cv = depth (into scene)

For MuJoCo's overhead camera the Y image-axis points in the +Y_world direction
(i.e. Y_mj is already "down" in OpenCV terms), so only Z must be flipped:

    Flip matrix:  diag([1, +1, -1])
    R_c2w_cv = R_mj.T  @  diag([1, +1, -1])

The resulting T_cam→robot (= T_cam→world for sim):
    p_world = T @ [X_cv, Y_cv, Z_cv, 1]ᵀ

For an overhead camera at pos=[0.3, 0, 0.8] with no rotation (default
MuJoCo orientation = looking straight down):
    R_mj   = identity  →  R_c2w_cv = diag([1, +1, -1])
    T = [[1, 0,  0, 0.3],
         [0, +1, 0, 0.0],
         [0, 0, -1, 0.8],
         [0, 0,  0, 1.0]]
"""
from __future__ import annotations

import math
import numpy as np

from ..config import CameraExtrinsics, CameraIntrinsics


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def mujoco_camera_intrinsics(
    model,
    camera_name: str,
    width:  int = 640,
    height: int = 480,
) -> CameraIntrinsics:
    """
    Compute pinhole intrinsics from MuJoCo camera fovy.

    MuJoCo uses a vertical FOV for all perspective cameras:
        fy = (height / 2) / tan(fovy / 2)
        fx = fy   (square pixels)
        cx = width / 2,  cy = height / 2

    Parameters
    ----------
    model       : mujoco.MjModel
    camera_name : Named camera in the model.
    width, height : Render resolution (must match the Renderer dimensions).
    """
    import mujoco
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(
            f"Camera {camera_name!r} not found in model. "
            f"Available: {[model.camera(i).name for i in range(model.ncam)]}"
        )

    fovy_deg = float(model.cam_fovy[cam_id])
    fovy_rad = math.radians(fovy_deg)
    fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy   # MuJoCo uses square pixels

    return CameraIntrinsics(
        fx=fx, fy=fy,
        cx=width  / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# Extrinsics
# ---------------------------------------------------------------------------

def mujoco_camera_extrinsics(
    model,
    data,
    camera_name: str,
) -> CameraExtrinsics:
    """
    Compute T_camera→world (= T_camera→robot for sim) from the live MuJoCo
    camera pose, converting from MuJoCo convention to OpenCV/pinhole convention.

    Requires that mj_forward() has been called so that cam_xpos / cam_xmat
    reflect the current kinematic state.

    Parameters
    ----------
    model       : mujoco.MjModel
    data        : mujoco.MjData (must have had mj_forward called)
    camera_name : Named camera in the model.

    Returns
    -------
    CameraExtrinsics with T: (4, 4) float64 — T_cam_opencv → world/robot.
    """
    import mujoco
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(
            f"Camera {camera_name!r} not found. "
            f"Available: {[model.camera(i).name for i in range(model.ncam)]}"
        )

    # cam_xmat: rows = camera frame axes in world (MuJoCo convention)
    R_mj = data.cam_xmat[cam_id].reshape(3, 3).copy()   # (3,3)
    t    = data.cam_xpos[cam_id].copy()                  # (3,) world position

    # R_c2w_mj has camera axes as ROWS, so R_c2w (columns = axes) = R_mj.T
    # MuJoCo Y_cam is already "down-in-image" (same as OpenCV Y_cv), so only Z
    # must be negated (MuJoCo Z out-of-screen → OpenCV Z into-scene).
    _FLIP = np.diag([1.0, 1.0, -1.0])
    R_c2w = R_mj.T @ _FLIP   # (3,3): transforms OpenCV cam-frame vector → world

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_c2w
    T[:3,  3] = t
    return CameraExtrinsics(T=T)


# ---------------------------------------------------------------------------
# Convenience: compute both in one call
# ---------------------------------------------------------------------------

def mujoco_camera_config(
    model,
    data,
    camera_name: str,
    render_w: int = 640,
    render_h: int = 480,
) -> tuple[CameraIntrinsics, CameraExtrinsics]:
    """
    Return (intrinsics, extrinsics) for a named MuJoCo camera.

    Ensures mj_forward has been called so poses are current.

    Usage
    -----
    >>> intr, extr = mujoco_camera_config(env.model, env.data, "overhead_cam")
    >>> config.camera_intrinsics = intr
    >>> config.camera_extrinsics = extr
    """
    import mujoco
    mujoco.mj_forward(model, data)

    intr = mujoco_camera_intrinsics(model, camera_name, render_w, render_h)
    extr = mujoco_camera_extrinsics(model, data, camera_name)
    return intr, extr
