# VLA Robot

A hierarchical Vision-Language-Action (VLA) framework for robotic manipulation.
It takes a natural-language command and an RGB-D camera frame, then plans, projects, builds, and executes a smooth Cartesian trajectory on a 6-DOF arm — all in four stages.

---

## Pipeline Overview

```
RGB + Depth + Command
        │
        ▼
 Stage 1 ─ PLAN        GeminiPlanner
                        (RGB + text) → semantic 2-D waypoints
        │
        ▼
 Stage 2 ─ PROJECT     DepthProjector
                        (u, v) + depth → 3-D robot-frame Point3D
        │
        ▼
 Stage 3 ─ BUILD       TrajectoryBuilder
                        geometric offsets + cubic spline → dense trajectory
        │
        ▼
 Stage 4 ─ EXECUTE     CartesianPIDController + LeRobotInterface
                        closed-loop servo, gripper stepped at transitions
```

---

## Requirements

- Python 3.10+
- Intel RealSense D435 (or any RGB-D camera)
- SO-100 6-DOF robot arm (via [LeRobot](https://github.com/huggingface/lerobot))
- Google Gemini API key

---

## Installation

```bash
git clone https://github.com/Haollll/vla_robot.git
cd vla_robot
pip install -r requirements.txt
```

For the latest LeRobot SO-100 support, install from source:

```bash
pip install git+https://github.com/huggingface/lerobot.git
```

---

## Usage

### Dry run (plan + project + build — no robot required)

```bash
python main.py \
  --command "pick up the red cube" \
  --api-key $GEMINI_KEY \
  --dry-run
```

### Full run with camera images

```bash
python main.py \
  --command "place the mug on the coaster" \
  --api-key $GEMINI_KEY \
  --rgb   /path/to/frame.png \
  --depth /path/to/depth.npy \
  --port  /dev/ttyUSB0
```

> `--depth` accepts `.npy` (float32, metres) or `.png` (uint16, millimetres).
> Omit `--rgb` / `--depth` to use built-in synthetic demo data.

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--command` / `-c` | `"Pick up the red cube..."` | Natural language task |
| `--api-key` / `-k` | *(required)* | Google Gemini API key |
| `--model` | `gemini-2.0-flash` | Gemini model ID |
| `--rgb` | None | Path to RGB image |
| `--depth` | None | Path to depth map |
| `--port` | `/dev/ttyUSB0` | Robot serial port |
| `--dry-run` | False | Skip robot execution |
| `--log-level` | `INFO` | DEBUG / INFO / WARNING / ERROR |

---

## Configuration

All hardware parameters are in [vla_framework/config.py](vla_framework/config.py) — camera intrinsics/extrinsics, PID gains, action offsets, and robot settings.
No code changes are needed when switching hardware; edit the dataclasses or pass a custom `VLAConfig` to `VLAPipeline`.

### Default camera (Intel RealSense D435 @ 640×480)

```python
CameraIntrinsics(fx=615.3, fy=615.3, cx=320.0, cy=240.0)
```

### Action offsets (metres)

| Action | Offset | Default |
|--------|--------|---------|
| APPROACH / MOVE / RETREAT | `safety_height` | 0.15 m |
| PRE_GRASP | `pre_grasp_height` | 0.05 m |
| GRASP | `grasp_descent` | 0.00 m |
| LIFT | `lift_height` | 0.20 m |
| PLACE | `place_height` | 0.02 m |

---

## Project Structure

```
vla_robot/
├── main.py                          # CLI entry point
├── requirements.txt
└── vla_framework/
    ├── config.py                    # All hardware config dataclasses
    ├── pipeline.py                  # Orchestrates the 4-stage pipeline
    ├── planner/
    │   └── gemini_planner.py        # Stage 1: Gemini VLM → semantic waypoints
    ├── projection/
    │   └── depth_projection.py      # Stage 2: pixel + depth → 3-D robot frame
    ├── path/
    │   └── trajectory_builder.py    # Stage 3: offsets + cubic spline
    └── control/
        ├── pid_controller.py        # Stage 4: Cartesian PID
        └── lerobot_interface.py     # Stage 4: LeRobot hardware interface
```

---

## Action Types

| Type | Description |
|------|-------------|
| `APPROACH` | Move above target at safety height, gripper open |
| `PRE_GRASP` | Descend to just above object |
| `GRASP` | Close gripper at object centroid |
| `LIFT` | Raise grasped object to lift height |
| `MOVE` | Translate at safety height |
| `PLACE` | Lower object to destination surface, release gripper |
| `RETREAT` | Rise clear of workspace |
| `HOME` | Return to home pose |

---

## License

MIT
