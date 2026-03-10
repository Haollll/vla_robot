"""
Eye-to-Hand Calibrator
======================
Solves for the 4×4 rigid-body transform T_camera→robot_base using a
ChArUco calibration board attached to the robot's end-effector.

Setup
-----
  - Camera is *fixed* (mounted above the workspace).
  - ChArUco board is attached to the gripper / end-effector.
  - Robot FK gives T_EE→base at each captured pose.
  - Board detection gives T_board→camera at each pose.
  - cv2.calibrateHandEye (TSAI method) solves for T_camera→base.

Board specification
-------------------
  Default: 7×5 ChArUco, square 0.03 m, marker 0.022 m, DICT_5X5_100.
  Print at exactly the declared physical size and mount flat on a rigid
  surface attached to the end-effector.

Usage
-----
>>> cal = EyeToHandCalibrator(camera_matrix, dist_coeffs)
>>> for frame, q_rad in samples:
...     ok, overlay = cal.add_sample(frame, q_rad)
>>> T, err_mm = cal.calibrate()
>>> cal.save("calibration/camera_to_robot.npy")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Minimum accepted samples before calibration is allowed
MIN_SAMPLES = 4


# ---------------------------------------------------------------------------
# OpenCV / ArUco compatibility shim  (handles API change in OpenCV 4.7)
# ---------------------------------------------------------------------------

def _build_charuco_board(squares_x: int, squares_y: int,
                          square_len: float, marker_len: float,
                          dict_name: str = "DICT_5X5_100"):
    """Return (board, aruco_dict) using whichever OpenCV ArUco API is available."""
    import cv2
    dict_id = getattr(cv2.aruco, dict_name)
    try:
        # OpenCV >= 4.7  (new API)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_len, marker_len, aruco_dict
        )
        return board, aruco_dict, "new"
    except AttributeError:
        # OpenCV < 4.7  (legacy API)
        aruco_dict = cv2.aruco.Dictionary_get(dict_id)
        board = cv2.aruco.CharucoBoard_create(
            squares_x, squares_y, square_len, marker_len, aruco_dict
        )
        return board, aruco_dict, "legacy"


def _detect_charuco(gray: np.ndarray, board, aruco_dict,
                    api: str, camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray
                    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                               Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect ChArUco corners and estimate board pose.

    Returns
    -------
    charuco_corners : (N, 1, 2) or None
    charuco_ids     : (N, 1)    or None
    rvec            : (3, 1)    or None  — board rotation in camera frame
    tvec            : (3, 1)    or None  — board translation in camera frame [m]
    """
    import cv2

    if api == "new":
        detector = cv2.aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if marker_ids is None or len(marker_ids) == 0:
            return None, None, None, None
        num, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
        if num < MIN_SAMPLES:
            return None, None, None, None

    if charuco_ids is None or len(charuco_ids) < MIN_SAMPLES:
        return None, None, None, None

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board,
        camera_matrix, dist_coeffs, None, None,
    )
    if not ok:
        return charuco_corners, charuco_ids, None, None

    return charuco_corners, charuco_ids, rvec, tvec


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class EyeToHandCalibrator:
    """
    Interactive eye-to-hand calibrator using a ChArUco board on the EE.

    Parameters
    ----------
    camera_matrix : (3, 3) float64 — intrinsic camera matrix K.
    dist_coeffs   : (4|5|8,) float64 — distortion coefficients.
    squares_x     : Number of chessboard squares along x (columns).
    squares_y     : Number of chessboard squares along y (rows).
    square_len    : Physical size of one chessboard square [m].
    marker_len    : Physical size of the ArUco marker [m].
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs:   np.ndarray,
        squares_x:   int   = 7,
        squares_y:   int   = 5,
        square_len:  float = 0.03,
        marker_len:  float = 0.022,
        dict_name:   str   = "DICT_5X5_100",
    ) -> None:
        self._K    = camera_matrix.astype(np.float64)
        self._dist = dist_coeffs.astype(np.float64)

        self._board, self._dict, self._api = _build_charuco_board(
            squares_x, squares_y, square_len, marker_len, dict_name
        )
        log.info(
            "ChArUco board %dx%d  sq=%.3fm  mk=%.3fm  dict=%s  OpenCV API=%s",
            squares_x, squares_y, square_len, marker_len, dict_name, self._api,
        )

        # Accumulated samples
        self._rvecs:         List[np.ndarray] = []   # board→cam  (3,1) each
        self._tvecs:         List[np.ndarray] = []   # board→cam  (3,1) each
        self._T_ee2base:     List[np.ndarray] = []   # EE→base    (4,4) each

    # ------------------------------------------------------------------
    # Sample capture
    # ------------------------------------------------------------------

    def add_sample(
        self,
        bgr_frame: np.ndarray,
        joint_angles_rad: np.ndarray,
    ) -> Tuple[bool, np.ndarray]:
        """
        Attempt to detect the ChArUco board and record one calibration sample.

        Parameters
        ----------
        bgr_frame        : uint8 (H, W, 3) — live BGR camera frame.
        joint_angles_rad : shape (≥5,) — current SO-101 joint angles [rad].

        Returns
        -------
        success : True if the board was detected and the sample was saved.
        overlay : BGR frame annotated with detected corners / axes.
        """
        import cv2
        from ..control.lerobot_interface import so101_fk_matrix

        gray    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        overlay = bgr_frame.copy()

        corners, ids, rvec, tvec = _detect_charuco(
            gray, self._board, self._dict, self._api, self._K, self._dist
        )

        if corners is not None and ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(overlay, corners, ids)

        if rvec is None or tvec is None:
            cv2.putText(overlay, "Board NOT detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return False, overlay

        # Draw board axes (0.03 m axis length)
        cv2.drawFrameAxes(overlay, self._K, self._dist, rvec, tvec, 0.03)

        # Record FK transform
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
        error_mm   : RMS position residual in millimetres (lower is better;
                     < 5 mm is generally acceptable).

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

        # Build R_gripper2base and t_gripper2base from FK transforms.
        # For eye-to-hand, pass the INVERSE (base→gripper) to calibrateHandEye.
        R_base2ee_list = []
        t_base2ee_list = []
        for T in self._T_ee2base:
            R_ee2base  = T[:3, :3]
            t_ee2base  = T[:3, 3]
            R_base2ee  = R_ee2base.T
            t_base2ee  = -R_base2ee @ t_ee2base
            R_base2ee_list.append(R_base2ee)
            t_base2ee_list.append(t_base2ee.reshape(3, 1))

        # Board→camera rotations and translations
        R_board2cam_list = [cv2.Rodrigues(rv)[0] for rv in self._rvecs]
        t_board2cam_list = [tv.reshape(3, 1) for tv in self._tvecs]

        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_base2ee_list,
            t_base2ee_list,
            R_board2cam_list,
            t_board2cam_list,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        T_cam2base = np.eye(4, dtype=np.float64)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3]  = t_cam2base.flatten()

        error_mm = self._reprojection_error(T_cam2base)

        log.info(
            "Calibration complete  n=%d  RMS_error=%.2f mm", n, error_mm
        )
        return T_cam2base, error_mm

    def _reprojection_error(self, T_cam2base: np.ndarray) -> float:
        """
        Compute RMS position residual [mm].

        For each sample i, the board origin is located in the base frame via
        two independent paths:
          (a) via camera: T_cam2base @ T_board2cam_i
          (b) via FK:     T_ee2base_i @ T_board2ee_mean

        where T_board2ee_mean is averaged over all samples.
        The RMS of ||a_i[:3,3] - b_i[:3,3]|| is the residual.
        """
        import cv2

        n = len(self._rvecs)

        # Path (a): board position in base via camera calibration
        pos_via_cam = []
        for rv, tv in zip(self._rvecs, self._tvecs):
            R_board2cam = cv2.Rodrigues(rv)[0]
            T_board2cam = np.eye(4)
            T_board2cam[:3, :3] = R_board2cam
            T_board2cam[:3, 3]  = tv.flatten()
            T_board2base = T_cam2base @ T_board2cam
            pos_via_cam.append(T_board2base[:3, 3])

        # Estimate T_board2ee for each sample, then average translation
        t_board2ee_samples = []
        for i, T_ee2base in enumerate(self._T_ee2base):
            T_base2ee = np.linalg.inv(T_ee2base)
            T_board2base = np.eye(4)
            T_board2base[:3, 3] = pos_via_cam[i]
            T_board2ee_i = T_base2ee @ T_board2base
            t_board2ee_samples.append(T_board2ee_i[:3, 3])
        t_board2ee_mean = np.mean(t_board2ee_samples, axis=0)

        # Path (b): predicted board position via FK + mean board offset
        errors = []
        for i, T_ee2base in enumerate(self._T_ee2base):
            pos_predicted = T_ee2base[:3, :3] @ t_board2ee_mean + T_ee2base[:3, 3]
            err = np.linalg.norm(pos_via_cam[i] - pos_predicted)
            errors.append(err)

        return float(np.sqrt(np.mean(np.array(errors) ** 2)) * 1000.0)  # → mm

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the calibration result (must call calibrate() first)."""
        raise RuntimeError(
            "Call calibrate() first to obtain T_cam2base, then pass it to save()."
        )

    @staticmethod
    def save_result(T_cam2base: np.ndarray, path: str | Path) -> None:
        """
        Save T_camera→robot_base to a .npy file.

        Parameters
        ----------
        T_cam2base : (4, 4) float64 — output of calibrate().
        path       : Destination file path (parent dirs created if needed).
        """
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

    def draw_board_preview(self, scale: float = 0.25) -> np.ndarray:
        """Return a scaled image of the ChArUco board for reference."""
        import cv2
        px_per_sq = 100
        w = int(7 * px_per_sq)
        h = int(5 * px_per_sq)
        img = self._board.generateImage((w, h))
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        return img
