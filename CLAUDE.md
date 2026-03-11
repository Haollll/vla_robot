# CLAUDE.md — VLA Robot Project Guide

## Project Overview

A hierarchical Vision-Language-Action (VLA) framework for robotic manipulation.
Given a natural-language command and an RGB-D camera frame, the pipeline plans,
projects, builds, and executes a smooth Cartesian trajectory on a 6-DOF arm.

**Hardware targets:**
- Camera: UVC stereo camera (side-by-side 2560×720, stereo-depth-toolkit)
- Robot: SO-101 6-DOF arm via Hugging Face LeRobot
- Planner: Google Gemini (multimodal VLM, `google-genai` SDK)

---

## Architecture — Four Stages

Each stage is its own class in `vla_framework/pipeline.py`.

```
RGB + Depth + Command
        │
        ▼
Stage 1  PlanStage               vla_framework/pipeline.py
         wraps GeminiPlanner
         (RGB + text) → List[SemanticWaypoint]  (action_type, pixel u,v, confidence)
        │
        ▼
Stage 2  ProjectStage            vla_framework/pipeline.py
         wraps DepthProjector
         (u,v) + depth → camera frame → robot base frame
         → List[Optional[Point3D]]
         ↓ plausibility check (z≥0, dist≤1.5m); retry planning once on failure
        │
        ▼
Stage 3  BuildStage              vla_framework/pipeline.py
         wraps TrajectoryBuilder
         geometric offsets per action + cubic spline interpolation
         → List[TrajectoryPoint]  (dense, ~50 pts per segment)
        │
        ▼
Stage 4  ExecuteStage            vla_framework/pipeline.py
         wraps CartesianPIDController + LeRobotInterface
         closed-loop PID servo to each trajectory point;
         gripper state stepped (not interpolated) at transitions
```

**Orchestrator:** `VLAPipeline` in `vla_framework/pipeline.py` — wires all stages,
owns plausibility check + retry logic, manages robot lifecycle via `ExecuteStage`.

**Entry point:** `main.py` — thin CLI only (arg parsing, image loading, pipeline call).

**Config factory:** `vla_framework/config_factory.py` — `build_config()` assembles
`VLAConfig` from CLI args + hardware defaults.

---

## Running the Pipeline

### Dry run (no robot needed)
```bash
python main.py \
  --command "pick up the red cube" \
  --api-key $GEMINI_KEY \
  --dry-run
```

### Live run with stereo camera + robot
```bash
python main.py \
  --command "place the mug on the coaster" \
  --api-key $GEMINI_KEY \
  --calib  /path/to/calib.yaml \
  --port   /dev/ttyACM0
```

### Run from pre-captured images + live robot
```bash
python main.py \
  --command "pick up the red cube" \
  --api-key $GEMINI_KEY \
  --rgb   /path/to/frame.png \
  --depth /path/to/depth.npy \
  --port  /dev/ttyACM0
```

### Production mode (fail loudly on missing hardware)
```bash
python main.py ... --no-mock
```

### Programmatic use
```python
from vla_framework.config_factory import build_config
from vla_framework.pipeline import VLAPipeline

config = build_config(api_key="...", model="gemini-2.5-flash", port="/dev/ttyACM0")
pipeline = VLAPipeline(config)

# Live camera
with pipeline:
    success = pipeline.run("pick up the red cube")

# Static images
with pipeline:
    success = pipeline.run_from_images(rgb, depth, "pick up the red cube")

# Dry run (no robot connect needed)
success = pipeline.run_from_images(rgb, depth, "pick up the red cube", dry_run=True)
```

### CLI flags
| Flag | Default | Description |
|------|---------|-------------|
| `--command` / `-c` | `"Pick up the red cube..."` | Natural language task |
| `--api-key` / `-k` | *(required)* | Google Gemini API key |
| `--model` | `gemini-2.5-flash` | Gemini model ID |
| `--rgb` | None (synthetic) | Path to RGB image (PNG/JPEG) |
| `--depth` | None (synthetic) | Path to depth map (.npy metres or .png uint16 mm) |
| `--port` | `/dev/ttyACM0` | Robot serial port |
| `--calib` | None | Stereo calibration YAML (required for live camera) |
| `--uvc-width` | `2560` | UVC capture width |
| `--uvc-height` | `720` | UVC capture height |
| `--uvc-fps` | `30` | UVC frame rate |
| `--dry-run` | False | Plan + project + build only; skip execution |
| `--no-mock` | False | Fail loudly if robot/lerobot unavailable |
| `--log-level` | `INFO` | DEBUG / INFO / WARNING / ERROR |

---

## File Structure

```
vla_robot/
├── main.py                              Thin CLI: arg parsing, image loading, pipeline call
├── calibrate.py                         Interactive eye-to-hand calibration tool (18 poses)
├── requirements.txt
├── CLAUDE.md                            This file
├── calibration/
│   ├── README.md                        Full calibration workflow
│   └── eye_to_hand_calibrator.py        ArUco-based hand-eye calibration library
└── vla_framework/
    ├── config.py                        Hardware config dataclasses (VLAConfig, etc.)
    ├── config_factory.py                build_config() — assembles VLAConfig from CLI args
    ├── pipeline.py                      PlanStage, ProjectStage, BuildStage, ExecuteStage,
    │                                    VLAPipeline orchestrator
    ├── planner/
    │   └── gemini_planner.py            Stage 1: Gemini VLM → List[SemanticWaypoint]
    ├── projection/
    │   └── depth_projection.py          Stage 2: (u,v) + depth → Point3D robot frame
    ├── path/
    │   └── trajectory_builder.py        Stage 3: offsets + cubic spline interpolation
    ├── camera/
    │   └── stereo_processor.py          Stereo-depth toolkit wrapper + live CameraStreamer
    └── control/
        ├── pid_controller.py            Stage 4: scalar PID + CartesianPIDController
        └── lerobot_interface.py         Stage 4: SO-101 hardware bridge (+ mock mode)
```

---

## Key Configuration — `vla_framework/config.py`

All hardware parameters live here. No code changes needed when swapping hardware;
just edit the dataclasses or pass a custom `VLAConfig` to `VLAPipeline`.

### `CameraIntrinsics` — pinhole calibration
```python
CameraIntrinsics(fx=615.3, fy=615.3, cx=320.0, cy=240.0, width=640, height=480)
```
Replace with values from your stereo calibration (`camera_matrix` from calib.yaml).

### `CameraExtrinsics` — 4×4 T_cam→robot rigid transform
Auto-loads from `calibration/camera_to_robot.npy` (written by `calibrate.py`).
Falls back to a hard-coded placeholder and prints a warning if the file is missing.
Run `python calibrate.py` to generate the calibration file.

### `PIDGains` — Cartesian PID
```python
PIDGains(kp=2.0, ki=0.05, kd=0.20, max_integral=5.0, output_min=-0.5, output_max=0.5)
```
Output units are m/s (velocity command). Tune `kp` first; `ki`/`kd` can stay at defaults
for most table-top tasks.

### `ActionOffsets` — geometric offsets per action (metres)
```python
ActionOffsets(
    safety_height    = 0.15,   # z-lift for APPROACH / MOVE / RETREAT
    pre_grasp_height = 0.05,   # z above object for PRE_GRASP hover
    grasp_descent    = 0.00,   # extra z descent at GRASP contact (tune per gripper)
    lift_height      = 0.20,   # z above grasp point after LIFT
    place_height     = 0.02,   # z above surface at PLACE
    retreat_height   = 0.15,   # z during RETREAT
)
```

### `VLAConfig` — top-level bundle
```python
VLAConfig(
    gemini_model        = "gemini-2.5-flash",  # swap without code changes
    interpolation_steps = 50,                  # dense points per segment
    waypoint_tolerance  = 0.005,               # PID convergence radius [m]
    control_frequency   = 50.0,                # Hz
    gripper_settle_s    = 0.40,                # wait after gripper command [s]
    robot_type          = "so101",
    robot_port          = "/dev/ttyACM0",
    robot_strict        = False,               # True = --no-mock (raise on hw failure)
)
```

---

## Development Notes

### Stage class design
Each stage class (`PlanStage`, `ProjectStage`, `BuildStage`, `ExecuteStage`) takes
a `VLAConfig` in `__init__` and exposes a single `.run()` method. Internal components
are private (`self._planner`, `self._robot`, etc.). `VLAPipeline` exposes the stages
as public attributes (`pipeline.plan_stage`, `pipeline.project_stage`, …) for
inspection or testing without running the full pipeline.

### Plausibility check + retry
After `ProjectStage.run()`, `VLAPipeline._check_positions()` validates every non-None
`Point3D`:
- `z ≥ 0` — not below the robot base plane
- `distance ≤ 1.5 m` — not an unrealistically distant reading

If the check fails, planning is retried once. If it fails again, the pipeline aborts.

### Mock mode (no robot hardware needed)
`LeRobotInterface` silently falls back to mock mode if `lerobot` is not installed
or the serial port is unavailable. Mock mode integrates velocity commands forward
using simplified FK — enough to test convergence logic end-to-end.

```
WARNING  lerobot package not found — running in MOCK mode.
```

Use `--no-mock` (sets `robot_strict=True`) to disable this and raise loudly instead.

### LeRobot API compatibility
`LeRobotInterface.connect()` tries `So101RobotConfig` (new API) first, then falls
back to `SO101FollowerConfig` (old API) if the import fails. Both are handled
transparently.

### Lazy imports
- `scipy.interpolate.CubicSpline` is imported inside `interpolate()` at call time.
  If scipy is missing, a `numpy.interp` linear fallback is used automatically.
- `google.genai` is imported inside `GeminiPlanner.__init__()`.
- `lerobot` is imported inside `LeRobotInterface.connect()`.
- `PIL.Image` is imported inside `main.load_images()`.
- `stereo_depth` toolkit is imported inside `build_streamer()`.

This means the package imports cleanly even without all optional dependencies installed.

### Swapping Gemini model
Change `VLAConfig.gemini_model` or pass `--model` at CLI — no other code changes:
```bash
python main.py --model gemini-2.5-flash ...
python main.py --model gemini-robotics-er ...  # when GA
```

### CubicSpline strictly-increasing knot fix
`TrajectoryBuilder.interpolate()` deduplicates `t_knots` after arc-length normalisation.
`GRASP` and `PLACE` intentionally produce two keyframes at the same position (descent +
gripper toggle), which creates duplicate knot values. A small epsilon nudge
(`np.finfo(float).eps * 1024 ≈ 2.3e-13`) is applied to any non-increasing value before
passing to `CubicSpline`.

### Depth conventions
- `.npy` (float32) → assumed metres
- `.png` (uint16) → assumed millimetres; auto-divided by 1000
- `DepthProjector` uses a `(2k+1)²` median patch (default k=3) around each pixel
  to suppress noise and small holes before back-projection.

### Gemini rate-limit handling
`GeminiPlanner` retries on 429 / RESOURCE_EXHAUSTED / 5xx with exponential back-off
(default: 4 retries, starting at 5 s, doubling each time). The `retry-after` hint
in the error message is parsed and respected if present.

### Calibration (eye-to-hand)
Uses a single ArUco marker (DICT_4X4_50, ID 0, 40 mm) mounted on the end-effector.
18-pose grid: 3 heights × 3 horizontal positions × 2 depths.
Output: `calibration/camera_to_robot.npy` (4×4 float64), auto-loaded by `CameraExtrinsics`.
See `calibration/README.md` for the full step-by-step workflow.
