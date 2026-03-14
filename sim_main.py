#!/usr/bin/env python3
"""
VLA Robot — MuJoCo simulation entry point
==========================================
Dry-run:
  python sim_main.py --command "pick up the red cube" --api-key $GEMINI_KEY

With viewer:
  python sim_main.py --command "pick up the red cube" --api-key $GEMINI_KEY --viewer

With video recording:
  python sim_main.py --command "pick up the red cube" --api-key $GEMINI_KEY --record
  # Output: /tmp/sim_run.mp4  (30 fps, one frame captured every 5 physics steps)
"""
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import numpy as np

from vla_framework.config_factory import build_config
from vla_framework.sim.sim_pipeline import SimVLAPipeline
from vla_framework.sim.mujoco_env import MuJoCoEnv
from vla_framework.sim.mujoco_interfaces import MuJoCoRobotInterface, MuJoCoCameraStreamer


def setup_logging(level="INFO"):
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--command", "-c", default="Pick up the red cube")
    p.add_argument("--api-key", "-k", required=True)
    p.add_argument("--model",   default="gemini-2.5-flash")
    p.add_argument("--scene",   default="models/robotis_omy/scene.xml")
    p.add_argument("--camera",  default="overhead_cam")
    p.add_argument("--ee-body", default="tcp_link")
    p.add_argument("--rgb",     default=None)
    p.add_argument("--depth",   default=None)
    p.add_argument("--viewer",  action="store_true")
    p.add_argument("--record",  action="store_true",
                   help="Record execution video to /tmp/sim_run.mp4 (requires opencv-python)")
    p.add_argument("--record-camera", default="overhead_cam", metavar="CAM",
                   help="Camera name to use for video recording (default: overhead_cam). "
                        "Planning/projection always uses --camera; only the video capture "
                        "uses this camera, so you can record a front view while the pipeline "
                        "plans from the top-down camera.  Example: --record-camera front_cam")
    p.add_argument("--record-every", type=int, default=5, metavar="N",
                   help="Capture one video frame every N physics steps (default: 5)")
    p.add_argument("--record-out", default="/tmp/sim_run.mp4",
                   help="Output path for the recorded video")
    p.add_argument("--debug-snapshot", default="/tmp/gemini_input.png",
                   help="Save annotated Gemini input snapshot here ('' to disable)")
    p.add_argument("--debug-bodies", nargs="*", default=["red_cube"],
                   help="MuJoCo body names to mark on the debug snapshot")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def load_images(rgb_path, depth_path):
    from PIL import Image as PILImage
    rgb   = np.array(PILImage.open(rgb_path).convert("RGB"))
    dp    = Path(depth_path)
    depth = np.load(dp).astype(np.float32) if dp.suffix == ".npy" \
            else np.array(PILImage.open(dp)).astype(np.float32) / 1000.0
    return rgb, depth


def _save_video(frames_bgr: list, out_path: str, fps: float, log) -> None:
    """Encode a list of BGR uint8 arrays into an mp4 file.

    frames_bgr : pre-converted BGR frames (may be side-by-side or single-cam).
    """
    if not frames_bgr:
        log.warning("No frames recorded — skipping video save")
        return
    try:
        import cv2
    except ImportError:
        log.error(
            "opencv-python not installed — cannot save video.\n"
            "Install with:  pip install opencv-python"
        )
        return

    h, w = frames_bgr[0].shape[:2]
    # 'mp4v' codec works on macOS/Linux without extra libs; produces .mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        log.error("cv2.VideoWriter could not open %r", out_path)
        return

    for frame in frames_bgr:
        writer.write(frame)
    writer.release()
    log.info(
        "Video saved → %s  (%d frames @ %.0f fps = %.1f s  size=%dx%d)",
        out_path, len(frames_bgr), fps, len(frames_bgr) / fps, w, h,
    )


def main():
    args = parse_args()
    setup_logging(args.log_level)
    log  = logging.getLogger("sim_main")

    if not Path(args.scene).exists():
        log.error("Scene not found: %s", args.scene)
        return 1

    config = build_config(api_key=args.api_key, model=args.model,
                          port="", no_mock=False)
    # Simulation-specific overrides:
    #   - 3 cm convergence tolerance (arm can't always hit 5 mm in sim)
    #   - 10 interpolation steps per segment → ~50 trajectory points total
    #     (50 points keeps execution fast; dense 200+ point trajectories
    #     cause unnecessary per-waypoint timeouts)
    config.control_frequency   = 200.0   # 200 Hz PID; phys_steps≈2 per cycle
    config.waypoint_tolerance  = 0.05
    config.interpolation_steps = 10
    # The GT XY override uses the MuJoCo cube body centre directly (z=0.02 m),
    # so no additional z offset is needed.  grasp_descent left at default 0.0.

    with MuJoCoEnv(args.scene, viewer=args.viewer) as env:
        # Compute intrinsics + extrinsics from the actual MuJoCo camera pose.
        # This replaces the placeholder _DEFAULT_T and fixes Stage 2 projection.
        from vla_framework.sim.camera_utils import mujoco_camera_config
        intr, extr = mujoco_camera_config(
            env.model, env.data, args.camera,
            render_w=640, render_h=480,
        )
        config.camera_intrinsics = intr
        config.camera_extrinsics = extr
        log.info(
            "Camera %r  fx=%.2f  T_cam->robot:\n%s",
            args.camera, intr.fx, extr.T,
        )

        robot    = MuJoCoRobotInterface(env, ee_body_name=args.ee_body,
                                        control_hz=config.control_frequency)
        streamer = MuJoCoCameraStreamer(
            env, camera_name=args.camera,
            debug_save_path = args.debug_snapshot or None,
            debug_bodies    = args.debug_bodies or [],
        )

        # The PID runs at control_frequency (200 Hz → dt=0.005 s) but each
        # env.step() advances physics by only 0.001 s (MuJoCo timestep, halved
        # from 0.002 to prevent tunneling through thin gripper mesh geoms).
        # Step ~5× per PID cycle so the simulation advances ≈0.005 s.
        _phys_steps = max(1, round(1.0 / (config.control_frequency * env.dt)))

        # Build sim_step_fn — optionally wrapped to capture side-by-side video.
        # Every N physics steps both overhead_cam and --record-camera are
        # rendered, hstacked into a 1280×480 BGR frame, labelled, and appended.
        _frames: list = []   # list of BGR uint8 (480, 1280, 3) combined frames
        if args.record:
            _step_ctr = [0]
            import cv2 as _cv2   # fail early if opencv is missing

            # Label drawing constants (resolved once, not every frame).
            _FONT       = _cv2.FONT_HERSHEY_SIMPLEX
            _FONT_SCALE = 0.9
            _THICKNESS  = 2
            _COLOR      = (255, 255, 255)   # white text
            _SHADOW     = (0, 0, 0)         # black drop-shadow for readability
            _LABEL_POS  = (10, 30)          # top-left of each half

            def _label(bgr_half: "np.ndarray", text: str) -> "np.ndarray":
                """Draw drop-shadow label on a BGR frame half (in-place copy)."""
                out = bgr_half.copy()
                sx, sy = _LABEL_POS
                # shadow
                _cv2.putText(out, text, (sx + 1, sy + 1),
                             _FONT, _FONT_SCALE, _SHADOW, _THICKNESS, _cv2.LINE_AA)
                # foreground
                _cv2.putText(out, text, (sx, sy),
                             _FONT, _FONT_SCALE, _COLOR, _THICKNESS, _cv2.LINE_AA)
                return out

            def sim_step_fn():
                for _ in range(_phys_steps):
                    env.step()
                    _step_ctr[0] += 1
                    if _step_ctr[0] % args.record_every == 0:
                        # Render both cameras (each 640×480 RGB).
                        overhead_rgb, _ = env.render_camera(args.camera)
                        front_rgb,    _ = env.render_camera(args.record_camera)
                        # Convert to BGR for OpenCV.
                        overhead_bgr = _cv2.cvtColor(overhead_rgb, _cv2.COLOR_RGB2BGR)
                        front_bgr    = _cv2.cvtColor(front_rgb,    _cv2.COLOR_RGB2BGR)
                        # Label each half.
                        overhead_bgr = _label(overhead_bgr, "OVERHEAD")
                        front_bgr    = _label(front_bgr,    "FRONT")
                        # Side-by-side → 1280×480.
                        combined = np.hstack([overhead_bgr, front_bgr])
                        _frames.append(combined)

            log.info(
                "[record] side-by-side overhead_cam | %r  every %d physics steps → %s",
                args.record_camera, args.record_every, args.record_out,
            )
        else:
            sim_step_fn = lambda: [env.step() for _ in range(_phys_steps)]

        pipeline = SimVLAPipeline(config, robot=robot, streamer=streamer,
                                  sim_step_fn=sim_step_fn)

        with pipeline:
            if args.rgb or args.depth:
                rgb, depth = load_images(args.rgb, args.depth)
                success = pipeline.run_from_images(rgb, depth, args.command)
            else:
                success = pipeline.run(args.command)

            # Save a snapshot of the final sim state (shows cube being held
            # after a successful lift, or last position on failure).
            try:
                from PIL import Image as PILImage
                snap_rgb, _ = env.render_camera(args.camera)
                PILImage.fromarray(snap_rgb).save("/tmp/final_lifted.png")
                log.info("Final snapshot saved → /tmp/final_lifted.png  (success=%s)", success)
            except Exception as _snap_err:
                log.warning("Snapshot save failed: %s", _snap_err)

            # Encode collected frames into mp4.
            if args.record:
                _save_video(_frames, args.record_out, fps=30.0, log=log)

            if args.viewer:
                log.info("Pipeline done — Ctrl+C to close viewer")
                try:
                    while env.viewer_is_running():
                        sim_step_fn()
                        env.sync_viewer()
                except KeyboardInterrupt:
                    pass

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
