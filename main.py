#!/usr/bin/env python3
"""
VLA Robot — CLI entry point
============================
Usage examples
--------------
# Dry-run (plan + project + build, no robot):
  python main.py --command "pick up the red cube" --api-key $GEMINI_KEY --dry-run

# Full run with live camera and robot:
  python main.py --command "place the mug on the coaster" \
                 --api-key $GEMINI_KEY \
                 --rgb   /path/to/frame.png \
                 --depth /path/to/depth.npy \
                 --port  /dev/ttyUSB0

# Specify a non-default Gemini model (e.g. gemini-robotics-er when available):
  python main.py ... --model gemini-robotics-er
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stereo_depth import CameraStreamer

import numpy as np

from vla_framework.config import (
    ActionOffsets,
    CameraExtrinsics,
    CameraIntrinsics,
    PIDGains,
    VLAConfig,
)
from vla_framework.pipeline import VLAPipeline


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
        level   = getattr(logging, level.upper(), logging.INFO),
        stream  = sys.stdout,
    )


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------

def make_demo_data(h: int = 480, w: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a plausible RGB + depth pair for smoke-testing without a camera.
    Depth: uniform 0.6 m background with a ~0.35 m "object" patch at centre.
    """
    rng = np.random.default_rng(42)

    # RGB: random noise scene
    rgb   = rng.integers(50, 200, size=(h, w, 3), dtype=np.uint8)
    # Fake "red cube" blob near centre
    rgb[200:260, 280:340, 0] = 200
    rgb[200:260, 280:340, 1] = 50
    rgb[200:260, 280:340, 2] = 50

    # Depth (float32 metres)
    depth = rng.uniform(0.55, 0.65, (h, w)).astype(np.float32)
    depth[200:260, 280:340] = 0.35   # object is closer

    return rgb, depth


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def build_config(
    api_key: str,
    model:   str,
    port:    str,
) -> VLAConfig:
    """
    Assemble VLAConfig.  Edit defaults here to match your hardware.

    Camera: Intel RealSense D435 @ 640×480
    Mount:  above the workspace, pointing ~30° downward
    Robot:  SO-100 6-DOF arm via LeRobot
    """
    intrinsics = CameraIntrinsics(
        fx=615.3, fy=615.3, cx=320.0, cy=240.0,
        width=640, height=480,
    )

    # T_cam→robot: camera is above and behind the workspace, pointing toward robot
    # Row order: [x_robot, y_robot, z_robot, translation]
    T = np.array(
        [
            [ 0.0, -1.0,  0.0,  0.25],
            [-1.0,  0.0,  0.0,  0.30],
            [ 0.0,  0.0, -1.0,  0.55],
            [ 0.0,  0.0,  0.0,  1.00],
        ],
        dtype=np.float64,
    )
    extrinsics = CameraExtrinsics(T=T)

    pid = PIDGains(
        kp=2.0, ki=0.05, kd=0.20,
        max_integral=5.0,
        output_min=-0.5, output_max=0.5,
    )

    offsets = ActionOffsets(
        safety_height    = 0.15,
        pre_grasp_height = 0.05,
        grasp_descent    = 0.00,
        lift_height      = 0.20,
        place_height     = 0.02,
        retreat_height   = 0.15,
    )

    return VLAConfig(
        camera_intrinsics  = intrinsics,
        camera_extrinsics  = extrinsics,
        pid_gains          = pid,
        action_offsets     = offsets,
        gemini_api_key     = api_key,
        gemini_model       = model,
        interpolation_steps= 50,
        waypoint_tolerance = 0.005,
        robot_type         = "so100",
        robot_port         = port,
        control_frequency  = 50.0,
        gripper_settle_s   = 0.40,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description = "Hierarchical VLA framework for robotic manipulation",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--command", "-c",
        default = "Pick up the red cube and place it in the blue bin",
        help    = "Natural language task command sent to the planner.",
    )
    p.add_argument(
        "--api-key", "-k",
        required = True,
        help     = "Google Gemini API key (or set GEMINI_API_KEY env var).",
    )
    p.add_argument(
        "--model",
        default = "gemini-2.5-flash",
        help    = 'Gemini model ID.  Use "gemini-robotics-er" when available.',
    )
    p.add_argument(
        "--rgb",
        default = None,
        help    = "Path to RGB image (PNG / JPEG).  Uses synthetic data if omitted.",
    )
    p.add_argument(
        "--depth",
        default = None,
        help    = (
            "Path to depth map.  Accepts .npy (float32, metres) or "
            ".png (uint16, millimetres).  Uses synthetic data if omitted."
        ),
    )
    p.add_argument(
        "--port",
        default = "/dev/ttyACM0",
        help    = "Serial port for the robot arm.",
    )
    p.add_argument(
        "--realsense",
        action  = "store_true",
        help    = (
            "Capture a live RGB+depth frame from the RealSense camera "
            "instead of reading from --rgb / --depth files.  "
            "Requires pyrealsense2 (pip install pyrealsense2)."
        ),
    )
    p.add_argument(
        "--rs-width",
        type    = int,
        default = 640,
        help    = "RealSense capture width (used with --realsense).",
    )
    p.add_argument(
        "--rs-height",
        type    = int,
        default = 480,
        help    = "RealSense capture height (used with --realsense).",
    )
    p.add_argument(
        "--rs-fps",
        type    = int,
        default = 30,
        help    = "RealSense capture frame rate (used with --realsense).",
    )
    p.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Plan + project + build trajectory, but skip robot execution.",
    )
    p.add_argument(
        "--calib",
        default = "/home/kevin/projects/stereo-depth-toolkit/outputs/calib/calib.yaml",
        help    = "Path to stereo calibration file (calib.yaml).",
    )
    p.add_argument(
        "--uvc-width",
        type    = int,
        default = 2560,
        help    = "UVC stereo camera capture width (side-by-side frame).",
    )
    p.add_argument(
        "--uvc-height",
        type    = int,
        default = 720,
        help    = "UVC stereo camera capture height.",
    )
    p.add_argument(
        "--uvc-fps",
        type    = int,
        default = 30,
        help    = "UVC stereo camera frame rate.",
    )
    p.add_argument(
        "--log-level",
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def load_images(rgb_path: str | None, depth_path: str | None):
    if rgb_path is None or depth_path is None:
        log = logging.getLogger("main")
        log.info("No image paths provided — using synthetic demo data")
        return make_demo_data()

    from PIL import Image as PILImage  # lazy import

    rgb   = np.array(PILImage.open(rgb_path).convert("RGB"))
    dp    = Path(depth_path)
    if dp.suffix == ".npy":
        depth = np.load(dp).astype(np.float32)
    else:
        depth = np.array(PILImage.open(dp)).astype(np.float32)
        if depth.max() > 10.0:   # assume mm
            depth /= 1000.0
    return rgb, depth


def build_streamer(
    calib_path:   str,
    device_index: int = 0,
    width:        int = 2560,
    height:       int = 720,
    fps:          int = 30,
) -> "CameraStreamer":
    """Construct a fully wired CameraStreamer from a calibration file."""
    from stereo_depth import CameraStreamer, RollingBuffer, StereoPipeline
    from stereo_depth.adapters.camera.uvc_source import UvcSource

    processor = StereoPipeline.from_yaml(calib_path)
    source    = UvcSource(device_index=device_index, width=width, height=height, fps=fps)
    buffer    = RollingBuffer(maxlen=15)
    return CameraStreamer(source, buffer, processor)


def capture_realsense(width: int, height: int, fps: int):
    """
    Capture one aligned RGB + depth frame from the RealSense camera.

    RGB and depth are acquired in two background threads (see
    RealSenseCamera internals) and joined before this function returns,
    guaranteeing a matched frame pair.
    """
    from vla_framework.camera import RealSenseCamera

    log = logging.getLogger("main")
    log.info("Opening RealSense camera  %dx%d @ %d fps", width, height, fps)
    with RealSenseCamera(width=width, height=height, fps=fps) as cam:
        if cam.is_mock:
            log.warning(
                "RealSense running in MOCK mode "
                "(no camera detected or pyrealsense2 not installed)"
            )
        rgb, depth = cam.capture()
    log.info(
        "RealSense capture done  rgb=%s  depth=%s  (mock=%s)",
        rgb.shape, depth.shape, cam.is_mock,
    )
    return rgb, depth


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("main")

    if args.realsense:
        rgb, depth = capture_realsense(args.rs_width, args.rs_height, args.rs_fps)
    else:
        rgb, depth = load_images(args.rgb, args.depth)
    log.info("Images loaded  rgb=%s  depth=%s", rgb.shape, depth.shape)

    config = build_config(args.api_key, args.model, args.port)

    if args.dry_run:
        pipeline = VLAPipeline(config)
        log.info("DRY RUN — planning only, robot will not move")
        waypoints  = pipeline.plan(rgb, args.command)
        positions  = pipeline.project_waypoints(waypoints, depth)
        trajectory = pipeline.build_trajectory(waypoints, positions)
        log.info("Trajectory has %d points", len(trajectory))
        stride = max(1, len(trajectory) // 10)
        for i, tp in enumerate(trajectory[::stride]):
            log.info("  [%4d] %s  pos=%s  grip=%.1f",
                     i * stride, tp.action_type.value,
                     tp.position.round(4), tp.gripper)
        return 0

    streamer = build_streamer(args.calib, width=args.uvc_width, height=args.uvc_height, fps=args.uvc_fps)
    pipeline = VLAPipeline(config, streamer=streamer)

    with pipeline.streamer:
        # wait for buffer to fill (~0.5s at 30fps)
        log.info("Waiting for camera buffer to fill...")
        while not pipeline.streamer.is_ready:
            time.sleep(0.05)
        log.info("Camera ready — starting pipeline")

        with pipeline:
            success = pipeline.run(args.command)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
