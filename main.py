#!/usr/bin/env python3
"""
VLA Robot — CLI entry point
============================
Dry run (plan + build, no robot):
  python main.py --command "pick up the red cube" --api-key $GEMINI_KEY --dry-run

Live run with stereo camera + robot:
  python main.py --command "place the mug on the coaster" \
                 --api-key $GEMINI_KEY \
                 --calib   /path/to/calib.yaml \
                 --port    /dev/ttyACM0

Live run from pre-captured images:
  python main.py --command "pick up the red cube" \
                 --api-key $GEMINI_KEY \
                 --rgb     /path/to/frame.png \
                 --depth   /path/to/depth.npy \
                 --port    /dev/ttyACM0
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

from vla_framework.config_factory import build_config
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
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description     = "Hierarchical VLA framework for robotic manipulation",
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
        help     = "Google Gemini API key.",
    )
    p.add_argument(
        "--model",
        default = "gemini-2.5-flash",
        help    = 'Gemini model ID.  Use "gemini-robotics-er" when available.',
    )
    p.add_argument(
        "--rgb",
        default = None,
        help    = "Path to RGB image (PNG/JPEG).  Uses synthetic data if omitted.",
    )
    p.add_argument(
        "--depth",
        default = None,
        help    = (
            "Path to depth map (.npy float32 metres or .png uint16 mm).  "
            "Uses synthetic data if omitted."
        ),
    )
    p.add_argument(
        "--port",
        default = "/dev/ttyACM0",
        help    = "Serial port for the robot arm.",
    )
    p.add_argument(
        "--calib",
        default = None,
        help    = "Path to stereo calibration YAML.  Required for live camera mode.",
    )
    p.add_argument("--uvc-width",  type=int, default=2560, help="UVC capture width.")
    p.add_argument("--uvc-height", type=int, default=720,  help="UVC capture height.")
    p.add_argument("--uvc-fps",    type=int, default=30,   help="UVC frame rate.")
    p.add_argument(
        "--dry-run",
        action = "store_true",
        help   = "Plan + project + build trajectory, but skip robot execution.",
    )
    p.add_argument(
        "--no-mock",
        action = "store_true",
        help   = (
            "Fail loudly if the robot arm or lerobot is unavailable "
            "instead of silently falling back to mock mode."
        ),
    )
    p.add_argument(
        "--log-level",
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_images(rgb_path: str | None, depth_path: str | None):
    """Load RGB + depth from disk, or return synthetic demo data."""
    log = logging.getLogger("main")
    if rgb_path is None or depth_path is None:
        log.info("No image paths provided — using synthetic demo data")
        return _make_demo_data()

    from PIL import Image as PILImage
    rgb = np.array(PILImage.open(rgb_path).convert("RGB"))
    dp  = Path(depth_path)
    if dp.suffix == ".npy":
        depth = np.load(dp).astype(np.float32)
    else:
        depth = np.array(PILImage.open(dp)).astype(np.float32)
        if depth.max() > 10.0:
            depth /= 1000.0   # mm → m
    return rgb, depth


def _make_demo_data(h: int = 480, w: int = 640):
    rng   = np.random.default_rng(42)
    rgb   = rng.integers(50, 200, size=(h, w, 3), dtype=np.uint8)
    rgb[200:260, 280:340, 0] = 200
    rgb[200:260, 280:340, 1] = 50
    rgb[200:260, 280:340, 2] = 50
    depth = rng.uniform(0.55, 0.65, (h, w)).astype(np.float32)
    depth[200:260, 280:340] = 0.35
    return rgb, depth


# ---------------------------------------------------------------------------
# Streamer helper
# ---------------------------------------------------------------------------

def build_streamer(calib: str, width: int, height: int, fps: int) -> "CameraStreamer":
    from vla_framework.camera.stereo_processor import build_streamer as _build
    return _build(calib, width=width, height=height, fps=fps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    log  = logging.getLogger("main")

    config = build_config(args.api_key, args.model, args.port, no_mock=args.no_mock)

    # ── Dry run: static images, no robot ────────────────────────────────────
    if args.dry_run:
        rgb, depth = load_images(args.rgb, args.depth)
        log.info("Images loaded  rgb=%s  depth=%s", rgb.shape, depth.shape)
        pipeline = VLAPipeline(config)
        success  = pipeline.run_from_images(rgb, depth, args.command, dry_run=True)
        return 0 if success else 1

    # ── Image-file run: static images, live robot ────────────────────────────
    if args.rgb is not None or args.depth is not None:
        rgb, depth = load_images(args.rgb, args.depth)
        log.info("Images loaded  rgb=%s  depth=%s", rgb.shape, depth.shape)
        pipeline = VLAPipeline(config)
        with pipeline:
            success = pipeline.run_from_images(rgb, depth, args.command)
        return 0 if success else 1

    # ── Live run: stereo camera + robot ─────────────────────────────────────
    if args.calib is None:
        log.error(
            "--calib is required for live camera mode.  "
            "Use --dry-run or --rgb/--depth for offline mode."
        )
        return 2

    streamer = build_streamer(args.calib, args.uvc_width, args.uvc_height, args.uvc_fps)
    pipeline = VLAPipeline(config, streamer=streamer)

    with streamer:
        log.info("Waiting for camera buffer to fill...")
        while not streamer.is_ready:
            time.sleep(0.05)
        log.info("Camera ready — starting pipeline")

        with pipeline:
            success = pipeline.run(args.command)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
