"""
Eye-to-Hand Calibrator
======================
Solves for the 4×4 rigid-body transform T_camera→robot_base using a single
ArUco marker attached to the robot's end-effector.

Setup
-----
  - Camera is *fixed* (mounted above the workspace).
  - Single ArUco marker is attached to the gripper / end-effector.
  - Robot FK gives T_EE→base at each captured pose.
  - Marker detection gives T_marker→camera at each pose.
  - cv2.calibrateHandEye (TSAI method) solves for T_camera→base.

Marker specification
--------------------
  Default: DICT_4X4_50, marker ID 0, physical size 0.04 m.
  Print at exactly the declared physical size and mount flat on a rigid
  surface attached to the end-effector.

Usage
-----
>>> cal = EyeToHandCalibrator(camera_matrix, dist_coeffs)
>>> for frame, q_rad in samples:
...     ok, overlay = cal.add_sample(frame, q_rad)
>>> T, err_mm = cal.calibrate()
>>> EyeToHandCalibrator.save_result(T, "calibration/camera_to_robot.npy")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ArUco detection
# ---------------------------------------------------------------------------

def _build_aruco_dict(dict_name: str = "DICT_4X4_50"):
    """Return an ArUco dictionary using whichever OpenCV API is available."""
    import cv2
    dict_id = getattr(cv2.aruco, dict_name)
    try:
        return cv2.aruco.getPredefinedDictionary(dict_id)   # OpenCV >= 4.7
    except AttributeError:
        return cv2.aruco.Dictionary_get(dict_id)            # OpenCV < 4.7


def _detect_aruco_marker(
    gray: np.ndarray,
    aruco_dict,
    marker_id: int,
    marker_size: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect a single ArUco marker and estimate its pose.

    Parameters
    ----------
    gray        : Grayscale image (H, W) uint8.
    aruco_dict  : ArUco dictionary object.
    marker_id   : The specific marker ID to look for.
    marker_size : Physical side length of the marker [m].
    camera_matrix, dist_coeffs : Camera intrinsics / distortion.

    Returns
    -------
    corners   : (1, 4, 2) float32 or None — detected marker corners.
    ids       : scalar int or None — the matched marker ID.
    rvec      : (3, 1) float64 or None — marker rotation in camera frame.
    tvec      : (3, 1) float64 or None — marker translation in camera frame [m].
    """
    import cv2

    all_corners, all_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)

    if all_ids is None or len(all_ids) == 0:
        return None, None, None, None

    # Find the requested marker ID
    for corners, mid in zip(all_corners, all_ids.flatten()):
        if mid == marker_id:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners], marker_size, camera_matrix, dist_coeffs
            )
            # rvecs/tvecs shape: (1, 1, 3) → squeeze to (3, 1)
            rvec = rvecs[0].reshape(3, 1).astype(np.float64)
            tvec = tvecs[0].reshape(3, 1).astype(np.float64)
            return corners, int(mid), rvec, tvec

    return None, None, None, None


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class EyeToHandCalibrator:
    """
    Interactive eye-to-hand calibrator using a single ArUco marker on the EE.

    Parameters
    ----------
    camera_matrix : (3, 3) float64 — intrinsic camera matrix K.
    dist_coeffs   : (4|5|8,) float64 — distortion coefficients.
    marker_id     : ArUco marker ID to track (must be printed on EE).
    marker_size   : Physical side length of the printed marker [m].
    dict_name     : ArUco dictionary name (attribute of cv2.aruco).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs:   np.ndarray,
        marker_id:    int   = 0,
        marker_size:  float = 0.04,
        dict_name:    str   = "DICT_4X4_50",
    ) -> None:
        self._K         = camera_matrix.astype(np.float64)
        self._dist      = dist_coeffs.astype(np.float64)
        self._marker_id = marker_id
        self._marker_sz = marker_size
        self._dict      = _build_aruco_dict(dict_name)
        self._dict_name = dict_name

        log.info(
            "ArUco marker  id=%d  size=%.3fm  dict=%s",
            marker_id, marker_size, dict_name,
        )

        # Accumulated samples
        self._rvecs:     List[np.ndarray] = []   # marker→cam  (3,1) each
        self._tvecs:     List[np.ndarray] = []   # marker→cam  (3,1) each
        self._T_ee2base: List[np.ndarray] = []   # EE→base     (4,4) each

    # ------------------------------------------------------------------
    # Sample capture
    # ------------------------------------------------------------------

    def add_sample(
        self,
        bgr_frame: np.ndarray,
        joint_angles_rad: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """
        Attempt to detect the ArUco marker and record one calibration sample.

        Parameters
        ----------
        bgr_frame        : uint8 (H, W, 3) — live BGR camera frame.
        joint_angles_rad : shape (≥5,) — current SO-101 joint angles [rad].

        Returns
        -------
        success : True if the marker was detected and the sample was saved.
        overlay : BGR frame annotated with detected marker / axes.
        """
        import cv2
        from ..control.so101_kinematics import so101_fk_matrix

        gray    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        overlay = bgr_frame.copy()

        corners, mid, rvec, tvec = _detect_aruco_marker(
            gray, self._dict, self._marker_id, self._marker_sz,
            self._K, self._dist,
        )

        if corners is not None:
            cv2.aruco.drawDetectedMarkers(
                overlay, [corners], np.array([[mid]])
            )

        if rvec is None or tvec is None:
            cv2.putText(overlay, "Marker NOT detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return False, overlay

        cv2.drawFrameAxes(overlay, self._K, self._dist, rvec, tvec,
                          self._marker_sz * 0.5)

        T_ee2base = so101_fk_matrix(joint_angles_rad)

        self._rvecs.append(rvec.copy())
        self._tvecs.append(tvec.copy())
        self._T_ee2base.append(T_ee2base.copy())

        n = len(self._rvecs)
        cv2.putText(
            overlay,
            f"Sample {n} captured  (need >= 15)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        log.info("Sample %d added  EE pos=%.3f,%.3f,%.3f",
                 n, T_ee2base[0, 3], T_ee2base[1, 3], T_ee2base[2, 3])
        return True, overlay

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self) -> Tuple[np.ndarray, float]:
        """
        Solve for T_camera→robot_base using cv2.calibrateHandEye (TSAI).

        Requires at least 10 samples (15–20 recommended).

        Returns
        -------
        T_cam2base : (4, 4) float64 — rigid transform camera→robot base.
        error_mm   : RMS position residual in millimetres.

        Raises
        ------
        RuntimeError if fewer than 10 samples have been collected.
        """
        import cv2

        n = len(self._rvecs)
        if n < 10:
            raise RuntimeError(
                f"Need at least 10 samples for calibration, have {n}."
            )

        # For eye-to-hand, pass the INVERSE (base→gripper) to calibrateHandEye.
        R_base2ee_list = []
        t_base2ee_list = []
        for T in self._T_ee2base:
            R_ee2base = T[:3, :3]
            t_ee2base = T[:3, 3]
            R_base2ee = R_ee2base.T
            t_base2ee = -R_base2ee @ t_ee2base
            R_base2ee_list.append(R_base2ee)
            t_base2ee_list.append(t_base2ee.reshape(3, 1))

        # Marker→camera rotations and translations
        R_marker2cam_list = [cv2.Rodrigues(rv)[0] for rv in self._rvecs]
        t_marker2cam_list = [tv.reshape(3, 1) for tv in self._tvecs]

        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_base2ee_list,
            t_base2ee_list,
            R_marker2cam_list,
            t_marker2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        T_cam2base = np.eye(4, dtype=np.float64)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3]  = t_cam2base.flatten()

        error_mm = self._reprojection_error(T_cam2base)
        log.info("Calibration complete  n=%d  RMS_error=%.2f mm", n, error_mm)
        return T_cam2base, error_mm

    def _reprojection_error(self, T_cam2base: np.ndarray) -> float:
        """
        Compute RMS position residual [mm].

        For each sample i, the marker origin is located in the base frame via
        two independent paths:
          (a) via camera: T_cam2base @ T_marker2cam_i
          (b) via FK:     T_ee2base_i @ T_marker2ee_mean
        The RMS of ||a_i[:3,3] - b_i[:3,3]|| is the residual.
        """
        import cv2

        pos_via_cam = []
        for rv, tv in zip(self._rvecs, self._tvecs):
            R_m2c = cv2.Rodrigues(rv)[0]
            T_m2c = np.eye(4)
            T_m2c[:3, :3] = R_m2c
            T_m2c[:3, 3]  = tv.flatten()
            pos_via_cam.append((T_cam2base @ T_m2c)[:3, 3])

        t_marker2ee_samples = []
        for i, T_ee2base in enumerate(self._T_ee2base):
            T_base2ee = np.linalg.inv(T_ee2base)
            T_m2base  = np.eye(4)
            T_m2base[:3, 3] = pos_via_cam[i]
            t_marker2ee_samples.append((T_base2ee @ T_m2base)[:3, 3])
        t_m2ee_mean = np.mean(t_marker2ee_samples, axis=0)

        errors = []
        for i, T_ee2base in enumerate(self._T_ee2base):
            pos_pred = T_ee2base[:3, :3] @ t_m2ee_mean + T_ee2base[:3, 3]
            errors.append(np.linalg.norm(pos_via_cam[i] - pos_pred))

        return float(np.sqrt(np.mean(np.array(errors) ** 2)) * 1000.0)  # → mm

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the calibration result (must call calibrate() first)."""
        raise RuntimeError(
            "Call calibrate() first to obtain T_cam2base, then pass it to save_result()."
        )

    @staticmethod
    def save_result(T_cam2base: np.ndarray, path: str | Path) -> None:
        """Save T_camera→robot_base to a .npy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, T_cam2base)
        log.info("Saved T_cam2base to %s", path)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of valid samples collected so far."""
        return len(self._rvecs)

    def draw_marker_preview(self, size_px: int = 300) -> np.ndarray:
        """Return a square image of the ArUco marker for visual reference."""
        import cv2
        try:
            img = cv2.aruco.generateImageMarker(
                self._dict, self._marker_id, size_px
            )
        except AttributeError:
            img = cv2.aruco.drawMarker(
                self._dict, self._marker_id, size_px
            )
        return img
