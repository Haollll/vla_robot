"""
Stereo Processor
================
Thin wrapper around the stereo-depth-toolkit's StereoPipeline.

Provides:
  - StereoProcessor   — offline per-frame rectification + disparity
  - build_streamer()  — construct a live CameraStreamer from a calib YAML

The toolkit is imported lazily so the rest of the framework still imports
cleanly when stereo-depth-toolkit is not installed (mock mode).

Usage
-----
>>> proc = StereoProcessor.from_yaml("calib.yaml")
>>> left_rect, right_rect = proc.rectify(left_bgr, right_bgr)
>>> depth_m = proc.depth(left_rect, right_rect)

Or with live streaming:
>>> streamer = build_streamer("calib.yaml", device_index=0)
>>> with streamer:
...     result = streamer.snapshot()
...     rgb, depth = result.rgb_snapshot, result.stable_depth
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from stereo_depth import CameraStreamer

log = logging.getLogger(__name__)


class StereoProcessor:
    """
    Wraps stereo-depth-toolkit's StereoPipeline for offline (single-frame)
    stereo rectification and depth computation.

    Parameters
    ----------
    pipeline : StereoPipeline instance loaded from a calibration YAML.
    """

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, calib_path: str) -> "StereoProcessor":
        """
        Build a StereoProcessor from a stereo calibration YAML file.

        Parameters
        ----------
        calib_path : Path to the calib.yaml produced by the toolkit.
        """
        from stereo_depth import StereoPipeline  # type: ignore[import-not-found]
        pipeline = StereoPipeline.from_yaml(calib_path)
        log.info("StereoProcessor loaded from %s", calib_path)
        return cls(pipeline)

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    def rectify(
        self,
        left_bgr:  np.ndarray,
        right_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stereo-rectify a raw left/right image pair.

        Parameters
        ----------
        left_bgr, right_bgr : uint8 (H, W, 3) BGR images.

        Returns
        -------
        left_rect, right_rect : rectified BGR images, same shape as input.
        """
        left_rect, right_rect = self._pipeline.rectify(left_bgr, right_bgr)
        return left_rect, right_rect

    def disparity(
        self,
        left_rect:  np.ndarray,
        right_rect: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a disparity map from a rectified stereo pair.

        Returns
        -------
        disp : float32 (H, W) disparity in pixels.  Invalid pixels are NaN.
        """
        return self._pipeline.disparity(left_rect, right_rect)

    def depth(
        self,
        left_rect:  np.ndarray,
        right_rect: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a metric depth map from a rectified stereo pair.

        Returns
        -------
        depth_m : float32 (H, W) depth in metres.  Invalid pixels are NaN.
        """
        return self._pipeline.depth(left_rect, right_rect)

    def process(
        self,
        left_bgr:  np.ndarray,
        right_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline: raw stereo pair → (rgb, depth_m).

        Equivalent to rectify() followed by depth(), returning the rectified
        left image as the RGB output alongside the metric depth map.

        Returns
        -------
        rgb     : uint8 (H, W, 3) — rectified left image.
        depth_m : float32 (H, W)  — metric depth in metres, NaN where invalid.
        """
        left_rect, right_rect = self.rectify(left_bgr, right_bgr)
        depth_m = self.depth(left_rect, right_rect)
        return left_rect, depth_m


# ---------------------------------------------------------------------------
# Live streaming helper
# ---------------------------------------------------------------------------

def build_streamer(
    calib_path:   str,
    device_index: int = 0,
    width:        int = 2560,
    height:       int = 720,
    fps:          int = 30,
    buffer_len:   int = 15,
) -> "CameraStreamer":
    """
    Construct a fully wired CameraStreamer from a calibration YAML.

    The returned streamer must be used as a context manager (or manually
    started/stopped).  Call ``streamer.snapshot()`` once ``streamer.is_ready``
    is True to obtain a ``(rgb_snapshot, stable_depth)`` result.

    Parameters
    ----------
    calib_path   : Path to the stereo calibration YAML.
    device_index : UVC camera device index (default 0).
    width, height, fps : Camera capture parameters for the side-by-side frame.
    buffer_len   : Number of frames in the rolling depth buffer.
    """
    from stereo_depth import CameraStreamer, RollingBuffer, StereoPipeline  # type: ignore[import-not-found]
    from stereo_depth.adapters.camera.uvc_source import UvcSource           # type: ignore[import-not-found]

    pipeline = StereoPipeline.from_yaml(calib_path)
    source   = UvcSource(device_index=device_index, width=width, height=height, fps=fps)
    buffer   = RollingBuffer(maxlen=buffer_len)
    log.info(
        "CameraStreamer created  calib=%s  dev=%d  %dx%d@%dfps  buf=%d",
        calib_path, device_index, width, height, fps, buffer_len,
    )
    return CameraStreamer(source, buffer, pipeline)
