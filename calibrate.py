#!/usr/bin/env python3
"""
Hand-Eye Calibration  —  eye-to-hand setup
===========================================
Collect 15-20 poses with the ChArUco board on the end-effector,
then solve for T_camera→robot_base and save it to
  calibration/camera_to_robot.npy

Usage
-----
  python calibrate.py --port /dev/ttyACM0 --camera 0

Controls during capture
-----------------------
  SPACE  — capture current frame as a calibration sample
  D      — discard the last sample
  P      — show board preview image
  Q      — stop collecting and run calibration (needs ≥ 10 samples)
  ESC    — abort without saving
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Eye-to-hand hand-eye calibration for SO-101 + RealSense",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--port", default="/dev/ttyACM0",
        help="Serial port of the SO-101 robot arm.",
    )
    p.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV VideoCapture device index for the calibration camera.",
    )
    p.add_argument(
        "--output", default="calibration/camera_to_robot.npy",
        help="Where to save the resulting 4×4 transform.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _print_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          Eye-to-Hand Calibration  —  SO-101 + RealSense          ║
╠══════════════════════════════════════════════════════════════════╣
║  Marker : ArUco  DICT_4X4_50  ID=0  physical size=0.04 m        ║
║  Mount the marker FLAT on the robot gripper / end-effector.     ║
╠══════════════════════════════════════════════════════════════════╣
║  Controls                                                        ║
║    SPACE  — capture sample at current pose                       ║
║    D      — discard last sample                                  ║
║    P      — show marker preview                                  ║
║    Q      — finish collecting and compute calibration            ║
║    ESC    — abort                                                ║
╠══════════════════════════════════════════════════════════════════╣
║  Tips                                                            ║
║    • Move the arm to 15-20 DIFFERENT poses before pressing Q.   ║
║    • Vary both position AND orientation each capture.            ║
║    • Keep the marker fully visible and well-lit.                 ║
║    • Avoid poses where the arm blocks the camera view.           ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# Pose hint table  (3 heights × 3 horizontal positions × 2 depths = 18)
# ---------------------------------------------------------------------------

_POSE_HINTS: list[str] = [
    # ── Near (close to camera) ─────────────────────────────────────────────
    "TOP-LEFT,     NEAR   — arm high, shifted left,   board flat",
    "TOP-CENTER,   NEAR   — arm high, centered,        board flat",
    "TOP-RIGHT,    NEAR   — arm high, shifted right,   board flat",
    "MID-LEFT,     NEAR   — arm mid,  shifted left,    tilt wrist 20° forward",
    "MID-CENTER,   NEAR   — arm mid,  centered,         tilt wrist 20° forward",
    "MID-RIGHT,    NEAR   — arm mid,  shifted right,   tilt wrist 20° forward",
    "LOW-LEFT,     NEAR   — arm low,  shifted left,    tilt wrist 40° forward",
    "LOW-CENTER,   NEAR   — arm low,  centered,         tilt wrist 40° forward",
    "LOW-RIGHT,    NEAR   — arm low,  shifted right,   tilt wrist 40° forward",
    # ── Far (away from camera) ─────────────────────────────────────────────
    "TOP-LEFT,     FAR    — arm high, shifted left,   rotate wrist 30° CW",
    "TOP-CENTER,   FAR    — arm high, centered,        rotate wrist 30° CW",
    "TOP-RIGHT,    FAR    — arm high, shifted right,   rotate wrist 30° CCW",
    "MID-LEFT,     FAR    — arm mid,  shifted left,    tilt wrist 30° left",
    "MID-CENTER,   FAR    — arm mid,  centered,         tilt wrist 30° right",
    "MID-RIGHT,    FAR    — arm mid,  shifted right,   tilt wrist 30° left",
    "LOW-LEFT,     FAR    — arm low,  shifted left,    tilt + rotate wrist",
    "LOW-CENTER,   FAR    — arm low,  centered,         tilt + rotate wrist",
    "LOW-RIGHT,    FAR    — arm low,  shifted right,   tilt + rotate wrist",
]

_TOTAL_POSES = len(_POSE_HINTS)   # 18


def _print_next_pose_hint(n_captured: int) -> None:
    """Print the suggested pose for the NEXT capture (0-indexed by n_captured)."""
    idx = n_captured  # next sample index
    if idx < _TOTAL_POSES:
        hint = _POSE_HINTS[idx]
        print(f"\n  ▶  Pose {idx + 1}/{_TOTAL_POSES}: {hint}")
    else:
        extra = idx - _TOTAL_POSES + 1
        print(f"\n  ▶  Bonus pose #{extra}: choose any new position / orientation")


def _print_diagnostics(cal, T_cam2base: np.ndarray) -> None:
    """Print per-sample and rotation-diversity diagnostics to help identify calibration error sources."""
    import cv2

    # ── Step 3: per-sample marker positions in robot base frame ──────────
    print("\n── Step 3: Per-sample marker position in robot base frame ──")
    positions = []
    for i, (rv, tv) in enumerate(zip(cal._rvecs, cal._tvecs)):
        R = cv2.Rodrigues(rv)[0]
        T_m2c = np.eye(4)
        T_m2c[:3, :3] = R
        T_m2c[:3, 3]  = tv.flatten()
        pos = (T_cam2base @ T_m2c)[:3, 3] * 100  # cm
        positions.append(pos)
        print(f"  [{i:02d}]  x={pos[0]:+6.1f}  y={pos[1]:+6.1f}  z={pos[2]:+6.1f}  cm")

    pos_arr = np.array(positions)
    spread  = pos_arr.max(axis=0) - pos_arr.min(axis=0)
    print(f"\n  Spread (cm):  x={spread[0]:.1f}  y={spread[1]:.1f}  z={spread[2]:.1f}")
    if spread.max() > 5.0:
        print("  ⚠ Large spread — likely FK model wrong (bad D_BASE/D_EE or stale my_follower.json)")
    else:
        print("  ✓ Positions consistent — FK model looks reasonable")

    # ── Step 4: FK vs physical sanity check ──────────────────────────────
    print("\n── Step 4: EE positions predicted by FK across all samples ──")
    for i, T_ee in enumerate(cal._T_ee2base):
        p = T_ee[:3, 3] * 100
        print(f"  [{i:02d}]  x={p[0]:+6.1f}  y={p[1]:+6.1f}  z={p[2]:+6.1f}  cm")
    print("  Compare z values to your physical arm height — should not be negative or > 50 cm")

    # ── Step 6: rotation diversity of captured poses ──────────────────────
    print("\n── Step 6: Rotation diversity of marker poses ──")
    angles = [float(np.linalg.norm(rv)) * 180.0 / np.pi for rv in cal._rvecs]
    print(f"  Rotation angles (deg):  min={min(angles):.1f}  max={max(angles):.1f}  "
          f"mean={np.mean(angles):.1f}  std={np.std(angles):.1f}")
    if np.std(angles) < 10.0:
        print("  ⚠ Low rotation diversity — redo capture and vary wrist_flex / wrist_roll each pose")
    else:
        print("  ✓ Sufficient rotation diversity")


def _print_matrix(T: np.ndarray) -> None:
    print("\nT_camera→robot_base  (4×4):")
    print("┌" + "─" * 54 + "┐")
    for row in T:
        vals = "  ".join(f"{v:+.6f}" for v in row)
        print(f"│  {vals}  │")
    print("└" + "─" * 54 + "┘")


# ---------------------------------------------------------------------------
# Main calibration loop
# ---------------------------------------------------------------------------

def run_calibration(args: argparse.Namespace) -> int:
    import cv2

    from vla_framework.calibration import EyeToHandCalibrator
    from vla_framework.control.lerobot_interface import LeRobotInterface
    from vla_framework.config import CameraIntrinsics

    log = logging.getLogger("calibrate")

    # ── Camera ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log.error("Cannot open camera %d", args.camera)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    log.info("Camera %d opened  %dx%d",
             args.camera,
             int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # ── Robot ────────────────────────────────────────────────────────────
    robot = LeRobotInterface(port=args.port, camera_index=None)
    robot.connect()
    if not robot.is_connected:
        log.error(
            "Robot not connected on %s — cannot calibrate without real joint angles.  "
            "Check the USB cable, power, and port, then retry.", args.port
        )
        cap.release()
        return 1

    # Disable torque so the arm can be moved freely to each calibration pose.
    try:
        robot._robot.bus.disable_torque()
        log.info("Motor torque disabled — arm can be moved freely.")
    except Exception as exc:
        log.warning("Could not disable torque: %s", exc)

    # ── Camera intrinsics ────────────────────────────────────────────────
    # Use the default D435 values; replace with your actual calibration if
    # you have run cv2.calibrateCamera on your specific unit.
    intr = CameraIntrinsics()
    K    = intr.K
    dist = np.zeros((4, 1), dtype=np.float64)   # assume no distortion

    # ── Calibrator ────────────────────────────────────────────────────────
    cal = EyeToHandCalibrator(K, dist,
                              marker_id=0, marker_size=0.04,
                              dict_name="DICT_4X4_50")

    _print_instructions()
    _print_next_pose_hint(0)
    print("\nMove the arm to the suggested pose, then press SPACE to capture.\n")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

    # Joint-change validation: reject samples where joints didn't move
    # (indicates a stale/mock read due to motor communication failure).
    _JOINT_CHANGE_THRESHOLD_RAD = math.radians(2.0)   # 2° minimum movement
    _last_accepted_q: np.ndarray | None = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.error("Failed to read frame from camera")
                break

            # Read current joint state from robot (or mock)
            state = robot.get_state()
            q_rad = state.joint_positions_rad

            # Detect stale reads: joints match last accepted sample exactly
            _joints_stale = (
                _last_accepted_q is not None
                and np.max(np.abs(q_rad[:5] - _last_accepted_q[:5])) < _JOINT_CHANGE_THRESHOLD_RAD
            )

            # Live preview: detect board without storing the sample
            _, overlay = _preview_detection(frame, cal, q_rad)

            # Joint-read health indicator
            joint_deg = np.degrees(q_rad[:5])
            joint_str = "  ".join(f"j{i+1}={v:+.0f}°" for i, v in enumerate(joint_deg))
            stale_tag = "  [STALE READ]" if _joints_stale else ""
            stale_color = (0, 0, 255) if _joints_stale else (200, 200, 200)
            cv2.putText(overlay, joint_str + stale_tag,
                        (10, overlay.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, stale_color, 1)

            # Status bar
            cv2.putText(
                overlay,
                f"Samples: {cal.n_samples}  |  SPACE=capture  D=discard  Q=calibrate  ESC=abort",
                (10, overlay.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            )

            cv2.imshow("Calibration", overlay)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:   # ESC
                print("\nAborted — no calibration saved.")
                return 0

            elif key == ord(' '):
                if _joints_stale:
                    print("  ✗ Joints unchanged from last sample — motor read may be stale.  "
                          "Move the arm further before capturing.")
                else:
                    ok, _ = cal.add_sample(frame, q_rad)
                    if ok:
                        _last_accepted_q = q_rad.copy()
                        print(f"  ✓ Sample {cal.n_samples} captured")
                        _print_next_pose_hint(cal.n_samples)
                    else:
                        print("  ✗ Marker not detected — move arm or improve lighting")

            elif key == ord('d') or key == ord('D'):
                if cal.n_samples > 0:
                    cal._rvecs.pop()
                    cal._tvecs.pop()
                    cal._T_ee2base.pop()
                    print(f"  Discarded last sample  ({cal.n_samples} remaining)")
                    _print_next_pose_hint(cal.n_samples)

            elif key == ord('p') or key == ord('P'):
                preview = cal.draw_marker_preview(size_px=300)
                cv2.imshow("Marker preview  (press any key to close)", preview)
                cv2.waitKey(0)
                cv2.destroyWindow("Marker preview  (press any key to close)")

            elif key == ord('q') or key == ord('Q'):
                if cal.n_samples < 10:
                    print(f"  Need at least 10 samples, have {cal.n_samples}.  Keep collecting.")
                else:
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()

    # ── Solve ────────────────────────────────────────────────────────────
    print(f"\nSolving with {cal.n_samples} samples …")
    try:
        T_cam2base, error_mm = cal.calibrate()
    except Exception as exc:
        log.error("Calibration failed: %s", exc)
        return 1

    _print_matrix(T_cam2base)
    print(f"\nRMS reprojection error : {error_mm:.2f} mm")
    if error_mm < 5.0:
        print("  ✓ Good calibration  (< 5 mm)")
    elif error_mm < 10.0:
        print("  ⚠ Acceptable, but consider recapturing with more varied poses")
    else:
        print("  ✗ High error — recapture with better board visibility and arm coverage")

    _print_diagnostics(cal, T_cam2base)

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    EyeToHandCalibrator.save_result(T_cam2base, out_path)
    print(f"\nSaved to:  {out_path.resolve()}")
    print("Restart your pipeline — CameraExtrinsics will auto-load this file.")
    return 0


def _preview_detection(
    frame: np.ndarray,
    cal: "EyeToHandCalibrator",
    q_rad: np.ndarray,
) -> tuple:
    """
    Run marker detection for live preview WITHOUT storing the sample.
    Returns (success, annotated_frame).
    """
    import cv2
    from vla_framework.calibration.eye_to_hand_calibrator import _detect_aruco_marker

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overlay = frame.copy()
    corners, mid, rvec, tvec = _detect_aruco_marker(
        gray, cal._detector, cal._marker_id, cal._marker_sz, cal._K, cal._dist
    )

    if corners is not None:
        cv2.aruco.drawDetectedMarkers(overlay, [corners], np.array([[mid]]))

    if rvec is not None:
        cv2.drawFrameAxes(overlay, cal._K, cal._dist, rvec, tvec,
                          cal._marker_sz * 0.5)
        cv2.putText(overlay, "Marker OK", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return True, overlay

    cv2.putText(overlay, "Marker NOT detected", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return False, overlay


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args_and_setup()
    return run_calibration(args)


def parse_args_and_setup() -> argparse.Namespace:
    args = _parse_args()
    logging.basicConfig(
        format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
        level   = getattr(logging, args.log_level.upper(), logging.INFO),
        stream  = sys.stdout,
    )
    return args


if __name__ == "__main__":
    sys.exit(main())
