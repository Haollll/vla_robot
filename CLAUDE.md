# CLAUDE.md — VLA Robot Project Guide

## Project Overview

A hierarchical Vision-Language-Action (VLA) framework for robotic manipulation.
Given a natural-language command and an RGB-D camera frame, the pipeline plans,
projects, builds, and executes a smooth Cartesian trajectory on a 6-DOF arm.

**Hardware targets:**
- Camera: Intel RealSense D435 (640×480, pinhole model)
- Robot: SO-100 6-DOF arm via Hugging Face LeRobot
- Planner: Google Gemini (multimodal VLM)

---

## Architecture — Four Stages

```
RGB + Depth + Command
        │
        ▼
Stage 1  GeminiPlanner          vla_framework/planner/gemini_planner.py
         sends (image + text) to Gemini; parses JSON response into
         List[SemanticWaypoint]  — action_type, pixel (u,v), confidence
        │
        ▼
Stage 2  DepthProjector          vla_framework/projection/depth_projection.py
         back-projects each (u,v) + depth → camera frame → robot base frame
         using pinhole model + 4×4 T_cam→robot extrinsic transform
         → List[Optional[Point3D]]
        │
        ▼
Stage 3  TrajectoryBuilder       vla_framework/path/trajectory_builder.py
         applies per-action geometric offsets (safety heights, gripper offsets)
         to expand waypoints into keyframes, then cubic spline interpolation
         → List[TrajectoryPoint]  (dense, ~50 pts per segment)
        │
        ▼
Stage 4  CartesianPIDController  vla_framework/control/pid_controller.py
         + LeRobotInterface       vla_framework/control/lerobot_interface.py
         closed-loop PID servo to each trajectory point;
         gripper state stepped (not interpolated) at transitions
```

Orchestrator: `vla_framework/pipeline.py` — `VLAPipeline.run()` chains all stages.
Entry point: `main.py` — CLI arg parsing, config assembly, image loading.

---

## Running the Pipeline

### Dry run (no robot needed)
```bash
python main.py \
  --command "pick up the red cube" \
  --api-key $GEMINI_KEY \
  --dry-run
```

### Full run with real camera images
```bash
python main.py \
  --command "place the mug on the coaster" \
  --api-key $GEMINI_KEY \
  --rgb   /path/to/frame.png \
  --depth /path/to/depth.npy \
  --port  /dev/ttyUSB0
```

### Programmatic use
```python
from vla_framework.config import VLAConfig
from vla_framework.pipeline import VLAPipeline

config = VLAConfig(gemini_api_key="...", gemini_model="gemini-2.0-flash")
pipeline = VLAPipeline(config)
with pipeline:                          # calls robot.connect() / disconnect()
    success = pipeline.run(rgb, depth, "pick up the red cube")
```

### CLI flags
| Flag | Default | Description |
|------|---------|-------------|
| `--command` / `-c` | `"Pick up the red cube..."` | Natural language task |
| `--api-key` / `-k` | *(required)* | Google Gemini API key |
| `--model` | `gemini-2.0-flash` | Gemini model ID |
| `--rgb` | None (synthetic) | Path to RGB image (PNG/JPEG) |
| `--depth` | None (synthetic) | Path to depth map (.npy metres or .png uint16 mm) |
| `--port` | `/dev/ttyUSB0` | Robot serial port |
| `--dry-run` | False | Plan + project + build only; skip execution |
| `--log-level` | `INFO` | DEBUG / INFO / WARNING / ERROR |

---

## File Structure

```
vla_robot/
├── main.py                              CLI entry point; config factory; image loader
├── requirements.txt
├── CLAUDE.md                            This file
└── vla_framework/
    ├── config.py                        All hardware config dataclasses (edit here per setup)
    ├── pipeline.py                      VLAPipeline orchestrator — wires all 4 stages
    ├── planner/
    │   └── gemini_planner.py            Stage 1: Gemini VLM → List[SemanticWaypoint]
    ├── projection/
    │   └── depth_projection.py          Stage 2: (u,v) + depth → Point3D robot frame
    ├── path/
    │   └── trajectory_builder.py        Stage 3: offsets + cubic spline interpolation
    └── control/
        ├── pid_controller.py            Stage 4: scalar PID + CartesianPIDController
        └── lerobot_interface.py         Stage 4: LeRobot hardware bridge (+ mock mode)
```

---

## Key Configuration — `vla_framework/config.py`

All hardware parameters live here. No code changes needed when swapping hardware;
just edit the dataclasses or pass a custom `VLAConfig` to `VLAPipeline`.

### `CameraIntrinsics` — pinhole calibration
```python
CameraIntrinsics(fx=615.3, fy=615.3, cx=320.0, cy=240.0, width=640, height=480)
```
Replace with values from your camera's `camera_info` topic.

### `CameraExtrinsics` — 4×4 T_cam→robot rigid transform
Default assumes camera mounted ~55 cm above workspace, angled 30° downward.
Replace `T` with your hand-eye calibration result.

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
    gemini_model        = "gemini-2.0-flash",  # swap without code changes
    interpolation_steps = 50,                  # dense points per segment
    waypoint_tolerance  = 0.005,               # PID convergence radius [m]
    control_frequency   = 50.0,                # Hz
    gripper_settle_s    = 0.40,                # wait after gripper command [s]
    robot_type          = "so100",
    robot_port          = "/dev/ttyUSB0",
)
```

---

## Development Notes

### Mock mode (no robot hardware needed)
`LeRobotInterface` silently falls back to mock mode if `lerobot` is not installed
or the serial port is unavailable. Mock mode integrates velocity commands forward
using simplified FK — enough to test convergence logic end-to-end.

```
WARNING  lerobot package not found — running in MOCK mode.
```

No code change needed; just run normally without a robot connected.

### Lazy imports
- `scipy.interpolate.CubicSpline` is imported inside `interpolate()` at call time.
  If scipy is missing, a `numpy.interp` linear fallback is used automatically.
- `google.genai` is imported inside `GeminiPlanner.__init__()`.
- `lerobot` is imported inside `LeRobotInterface.connect()`.
- `PIL.Image` is imported inside `main.load_images()`.

This means the package imports cleanly even without all optional dependencies installed.

### Swapping Gemini model
Change `VLAConfig.gemini_model` or pass `--model` at CLI — no other code changes:
```bash
python main.py --model gemini-1.5-flash ...   # separate quota bucket
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
