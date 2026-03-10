# Eye-to-Hand Calibration

Solves for **T_camera → robot_base** — the 4×4 rigid-body transform that maps
any 3-D point in the camera frame into the robot base frame.
The result is saved as `calibration/camera_to_robot.npy` and auto-loaded by
the VLA pipeline at every startup.

---

## Overview

**Setup type:** Eye-to-hand (camera is fixed; board moves with the robot arm)

| Captured data | Source |
|---|---|
| T_board → camera | OpenCV ArUco board detection |
| T_EE → base | SO-101 forward kinematics (`so101_fk_matrix`) |

After ≥ 10 pose pairs, `cv2.calibrateHandEye` (TSAI method) solves for
**T_camera → base** that is consistent across all collected pairs.

---

## Board Specification

| Parameter | Value |
|---|---|
| Type | ChArUco |
| Squares | 7 columns × 5 rows |
| Square side | **30 mm** |
| Marker side | **22 mm** |
| Dictionary | `DICT_5X5_100` |

### Step 1 — Generate and print the board

```python
python - <<'EOF'
import cv2
d   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
b   = cv2.aruco.CharucoBoard((7, 5), 0.030, 0.022, d)
img = b.generateImage((700, 500))
cv2.imwrite("charuco_7x5.png", img)
print("Saved  charuco_7x5.png  (print at exactly 210 × 150 mm)")
EOF
```

Print the saved image **at exactly 210 × 150 mm** — disable all printer
scaling / "fit-to-page" options.
Verify with a ruler: one square = **30 mm ± 0.5 mm**.

### Step 2 — Mount the board on the end-effector

- Glue or tape the printout to a **rigid, flat backing** (acrylic, cardboard).
- Attach the backing firmly to the **gripper face** of the SO-101 arm.
- The board must not flex, tilt, or shift between captures.

---

## Calibration Workflow

### Prerequisites

```bash
pip install opencv-contrib-python numpy
# SO-101 arm connected via USB
```

### Step 3 — Run the calibration script

```bash
python calibrate.py --port /dev/ttyACM0 --camera 0
```

| Flag | Default | Description |
|---|---|---|
| `--port` | `/dev/ttyACM0` | Serial port of the SO-101 arm |
| `--camera` | `0` | OpenCV `VideoCapture` device index |
| `--output` | `calibration/camera_to_robot.npy` | Where to save the result |
| `--log-level` | `INFO` | Verbosity: `DEBUG` / `INFO` / `WARNING` |

A live camera window opens. Green **"Board OK"** + coordinate axes appear
when the board is detected. The status bar shows how many samples you have.

### Step 4 — Collect 15–20 samples

For each sample:

1. **Move the arm** to a new pose (different position AND orientation).
2. Wait for **"Board OK"** to appear in the window.
3. Press **`SPACE`** to capture.

Repeat until you have 15–20 samples covering the full workspace.

**Controls:**

| Key | Action |
|---|---|
| `SPACE` | Capture sample at current pose |
| `D` | Discard the last sample |
| `P` | Show board preview image |
| `Q` | Stop collecting and compute calibration (needs ≥ 10) |
| `ESC` | Abort without saving |

**Tips for good calibration:**

- Vary **both position and orientation** — pure translations without rotation
  make the problem under-determined.
- Spread poses across the whole reachable workspace: left, right, near, far,
  tilted forward, tilted sideways.
- Keep the board **fully visible** — no partial crops at frame edges.
- Avoid arm cables **occluding** parts of the board.
- Use **diffuse, consistent lighting** — hard shadows degrade corner detection.

### Step 5 — Solve and save

Press **`Q`** once you have ≥ 10 samples. The script:

1. Runs `cv2.calibrateHandEye` (TSAI).
2. Prints the 4×4 T matrix and RMS reprojection error.
3. Saves `calibration/camera_to_robot.npy`.

Example output:

```
Solving with 17 samples …

T_camera→robot_base  (4×4):
┌──────────────────────────────────────────────────────┐
│  +0.012345  -0.999876  +0.009876  +0.253100  │
│  -0.999923  -0.012301  -0.001234  +0.301200  │
│  +0.001123  -0.009901  -0.999950  +0.548700  │
│  +0.000000  +0.000000  +0.000000  +1.000000  │
└──────────────────────────────────────────────────────┘

RMS reprojection error : 2.47 mm
  ✓ Good calibration  (< 5 mm)

Saved to:  /path/to/vla_robot/calibration/camera_to_robot.npy
Restart your pipeline — CameraExtrinsics will auto-load this file.
```

### Step 6 — Restart the pipeline

```bash
python main.py --command "pick up the red cube" --api-key $GEMINI_KEY
```

`CameraExtrinsics` reads `camera_to_robot.npy` automatically on startup —
no code changes needed.

---

## Reprojection Error Guide

| RMS error | Quality | Action |
|---|---|---|
| < 5 mm | ✓ Good | Use as-is |
| 5 – 10 mm | ⚠ Acceptable | Recapture with more varied poses |
| > 10 mm | ✗ Poor | Redo — check board mounting and lighting |

---

## Auto-Loading in the Pipeline

`vla_framework/config.py` loads the file at import time:

```python
_CALIB_FILE = Path(__file__).parent.parent / "calibration" / "camera_to_robot.npy"

def _load_extrinsics() -> np.ndarray:
    if _CALIB_FILE.exists():
        return np.load(_CALIB_FILE).astype(np.float64)   # your calibration
    warnings.warn("Camera extrinsics not calibrated …")
    return _DEFAULT_T.copy()                              # hard-coded fallback
```

The loaded matrix is injected into `DepthProjector` via `CameraExtrinsics.T`.
If the file is absent the pipeline runs with a placeholder transform and
emits a `WARNING`.

---

## When to Re-calibrate

- Camera is physically moved or re-mounted.
- Robot base is relocated.
- RMS error exceeds 5 mm after an initial run.
- A different robot arm is substituted.

Run `calibrate.py` again — it overwrites the existing `.npy` file.
