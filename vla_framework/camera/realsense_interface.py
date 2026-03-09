"""
RealSense Camera Interface
==========================
Captures spatially-aligned RGB + depth frames from an Intel RealSense
D-series camera (D415, D435, D455 …) using the pyrealsense2 SDK.

Capture model
-------------
RGB and depth are acquired in two background threads that post their
latest frames into a shared slot.  ``capture()`` joins both threads
(waits for at least one valid frame each) before returning, so the
caller always receives a matched pair with no partial-frame skew.

Return types
------------
  rgb   : np.ndarray  uint8   (H, W, 3)   — BGR→RGB converted
  depth : np.ndarray  float32 (H, W)  [m] — metric metres, NaN where invalid

Mock mode
---------
If ``pyrealsense2`` is not installed the camera silently runs in mock
mode, generating a synthetic scene (random background + foreground patch)
that is good enough for dry-run pipeline testing.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Type alias for the paired output
FramePair = Tuple[np.ndarray, np.ndarray]   # (rgb, depth_m)


# ---------------------------------------------------------------------------
# Mock frame generator
# ---------------------------------------------------------------------------

def _make_mock_frames(
    width: int = 640,
    height: int = 480,
    seed: Optional[int] = None,
) -> FramePair:
    """
    Generate a plausible synthetic RGB + depth pair for testing.

    Depth: uniform 0.60 m background with a ~0.35 m foreground patch
    centred at (width//2, height//2).
    """
    rng   = np.random.default_rng(seed)
    rgb   = rng.integers(50, 200, (height, width, 3), dtype=np.uint8)
    # Fake "red object" blob near centre
    cy, cx  = height // 2, width // 2
    h2, w2  = height // 8, width // 8
    rgb[cy - h2:cy + h2, cx - w2:cx + w2, 0] = 200
    rgb[cy - h2:cy + h2, cx - w2:cx + w2, 1] = 50
    rgb[cy - h2:cy + h2, cx - w2:cx + w2, 2] = 50

    depth = np.full((height, width), 0.60, dtype=np.float32)
    depth[cy - h2:cy + h2, cx - w2:cx + w2] = 0.35
    return rgb, depth


# ---------------------------------------------------------------------------
# Thread workers
# ---------------------------------------------------------------------------

class _CaptureThread(threading.Thread):
    """
    Background thread that captures frames until ``stop()`` is called.

    The most recent valid frame is stored in ``self.result``.
    ``self.event`` is set once the first frame has been received.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name, daemon=True)
        self.result: Optional[np.ndarray] = None
        self.event  = threading.Event()
        self._stop  = threading.Event()
        self._fn    = None          # set by subclass before start()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            frame = self._fn()
            if frame is not None:
                self.result = frame
                self.event.set()
            time.sleep(0.001)       # yield to other threads


# ---------------------------------------------------------------------------
# Main interface
# ---------------------------------------------------------------------------

class RealSenseCamera:
    """
    Intel RealSense aligned-stream camera interface.

    Parameters
    ----------
    width, height   : Desired stream resolution (must be supported by the
                      camera; D435 supports 640×480, 848×480, 1280×720).
    fps             : Stream frame rate (15, 30, or 60).
    warm_up_frames  : Number of frames to discard after pipeline start to
                      let auto-exposure and auto-white-balance settle.
    timeout_s       : Maximum seconds to wait for the first frame pair
                      before raising RuntimeError.

    Usage
    -----
    >>> cam = RealSenseCamera()
    >>> with cam:
    ...     rgb, depth = cam.capture()
    """

    def __init__(
        self,
        width:        int = 640,
        height:       int = 480,
        fps:          int = 30,
        warm_up_frames: int = 10,
        timeout_s:    float = 5.0,
    ) -> None:
        self._w          = width
        self._height     = height
        self._fps        = fps
        self._warm_up    = warm_up_frames
        self._timeout    = timeout_s
        self._pipeline   = None     # pyrealsense2 pipeline
        self._align      = None     # rs.align object
        self._mock       = False
        self._started    = False

        # Thread slots
        self._rgb_thread:   Optional[_CaptureThread] = None
        self._depth_thread: Optional[_CaptureThread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the RealSense pipeline and start capture threads."""
        if self._started:
            return

        try:
            import pyrealsense2 as rs  # type: ignore[import-not-found]
            self._rs = rs
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, self._w, self._height,
                              rs.format.bgr8, self._fps)
            cfg.enable_stream(rs.stream.depth, self._w, self._height,
                              rs.format.z16,  self._fps)
            self._pipeline.start(cfg)
            self._align = rs.align(rs.stream.color)

            # Warm-up: discard first N frames
            for _ in range(self._warm_up):
                self._pipeline.wait_for_frames()

            log.info(
                "RealSense pipeline started  %dx%d @ %d fps",
                self._w, self._height, self._fps,
            )
        except ImportError:
            log.warning(
                "pyrealsense2 not installed — running camera in MOCK mode.  "
                "Install with:  pip install pyrealsense2"
            )
            self._mock = True
        except RuntimeError as exc:
            log.warning(
                "RealSense device not found (%s) — running camera in MOCK mode.", exc
            )
            self._mock = True

        self._started = True
        self._start_threads()

    def stop(self) -> None:
        """Stop capture threads and close the pipeline."""
        if not self._started:
            return
        if self._rgb_thread:
            self._rgb_thread.stop()
        if self._depth_thread:
            self._depth_thread.stop()
        if self._rgb_thread:
            self._rgb_thread.join(timeout=2.0)
        if self._depth_thread:
            self._depth_thread.join(timeout=2.0)
        if self._pipeline and not self._mock:
            try:
                self._pipeline.stop()
            except Exception as exc:
                log.debug("Pipeline stop error (ignored): %s", exc)
        self._started = False
        log.info("RealSense camera stopped")

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def capture(self) -> FramePair:
        """
        Return the latest aligned (rgb, depth) pair.

        Blocks until both threads have posted at least one frame, or
        until ``timeout_s`` elapses (raises RuntimeError).

        Returns
        -------
        rgb   : np.ndarray  uint8  (H, W, 3)  — RGB channel order
        depth : np.ndarray  float32 (H, W)    — metres, NaN where invalid
        """
        if not self._started:
            raise RuntimeError("Camera not started — call start() or use as context manager")

        deadline = time.monotonic() + self._timeout
        for thr in (self._rgb_thread, self._depth_thread):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not thr.event.wait(timeout=remaining):
                raise RuntimeError(
                    f"Timed out waiting for first frame from {thr.name}"
                )

        rgb   = self._rgb_thread.result.copy()
        depth = self._depth_thread.result.copy()
        return rgb, depth

    # ------------------------------------------------------------------
    # Internal thread setup
    # ------------------------------------------------------------------

    def _start_threads(self) -> None:
        self._rgb_thread   = _CaptureThread("rs-rgb")
        self._depth_thread = _CaptureThread("rs-depth")

        if self._mock:
            # Both threads share a single mock generator call; seed varies
            # per call so the scene looks slightly different each capture.
            def _mock_rgb():
                rgb, _ = _make_mock_frames(self._w, self._height)
                return rgb

            def _mock_depth():
                _, depth = _make_mock_frames(self._w, self._height)
                return depth

            self._rgb_thread._fn   = _mock_rgb
            self._depth_thread._fn = _mock_depth
        else:
            self._rgb_thread._fn   = self._read_rgb
            self._depth_thread._fn = self._read_depth

        self._rgb_thread.start()
        self._depth_thread.start()

    def _read_rgb(self) -> Optional[np.ndarray]:
        """Grab the latest colour frame and return it as a uint8 RGB array."""
        try:
            frames        = self._pipeline.poll_for_frames()
            aligned       = self._align.process(frames)
            colour_frame  = aligned.get_color_frame()
            if not colour_frame:
                return None
            bgr = np.asanyarray(colour_frame.get_data(), dtype=np.uint8)
            return bgr[:, :, ::-1]   # BGR → RGB
        except Exception as exc:
            log.debug("RGB capture error: %s", exc)
            return None

    def _read_depth(self) -> Optional[np.ndarray]:
        """
        Grab the latest depth frame and return it as a float32 array [m].
        Zero-valued pixels (no return) are replaced with NaN.
        """
        try:
            frames       = self._pipeline.poll_for_frames()
            aligned      = self._align.process(frames)
            depth_frame  = aligned.get_depth_frame()
            if not depth_frame:
                return None
            raw   = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            depth = raw.astype(np.float32)
            # RealSense default depth scale: 1 unit = 0.001 m
            depth_scale = (
                self._pipeline.get_active_profile()
                .get_device()
                .first_depth_sensor()
                .get_depth_scale()
            )
            depth *= depth_scale
            depth[depth == 0] = np.nan   # mark invalid pixels
            return depth
        except Exception as exc:
            log.debug("Depth capture error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_mock(self) -> bool:
        """True when running without a physical RealSense device."""
        return self._mock

    @property
    def resolution(self) -> Tuple[int, int]:
        """(width, height) configured for this camera instance."""
        return self._w, self._height
