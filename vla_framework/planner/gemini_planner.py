"""
Semantic Waypoint Planner  —  Stage 1
======================================
Sends (RGB image, natural language command) to Gemini and receives a
structured sequence of semantic 2D waypoints.

SDK
---
Uses the current `google-genai` package (google.genai).
Install:  pip install google-genai

Model note
----------
Targets "gemini-robotics-er" when GA.  Use "gemini-2.0-flash" or
"gemini-1.5-flash" for dev/testing.  Swap config.gemini_model; no other
code changes needed.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from ..config import ActionType

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SemanticWaypoint:
    action_type:   ActionType
    pixel_coords:  Tuple[int, int]  # (col u, row v) in image space
    description:   str
    confidence:    float            # 0–1 planner confidence
    gripper_state: float            # 0.0 = fully open, 1.0 = fully closed


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robotic manipulation planner. Given an RGB image and a natural
language command, generate the minimal ordered sequence of semantic
waypoints required to complete the task.

Each waypoint must include:
  action_type   – one of: approach | pre_grasp | grasp | lift | move | place | retreat
  pixel_u       – integer column (0 … image_width-1)
  pixel_v       – integer row    (0 … image_height-1)
  description   – one short phrase describing this step
  confidence    – float in [0, 1]
  gripper_state – 0.0 for open, 1.0 for closed

Action sequencing rules:
  • approach   : move end-effector over the target at a safe height, gripper open
  • pre_grasp  : descend to just above the object, gripper still open
  • grasp      : close the gripper at the object centroid
  • lift       : raise the object to carry height, gripper closed
  • move       : translate horizontally to destination, gripper closed
  • place      : lower object to the destination surface, then open gripper
  • retreat    : rise clear of the workspace, gripper open

Pixel coordinates should point to the centroid of the relevant object or
surface in the image.  If an action has no distinct pixel target (e.g. lift),
reuse the coordinates of the preceding waypoint.

Respond ONLY with a valid JSON array — no prose, no markdown fences.
Example (abbreviated):
[
  {"action_type":"approach",  "pixel_u":310, "pixel_v":220,
   "description":"approach red cube", "confidence":0.95, "gripper_state":0.0},
  {"action_type":"pre_grasp", "pixel_u":310, "pixel_v":222,
   "description":"descend above cube", "confidence":0.93, "gripper_state":0.0},
  {"action_type":"grasp",     "pixel_u":310, "pixel_v":222,
   "description":"close gripper on cube", "confidence":0.91, "gripper_state":1.0}
]
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class GeminiPlanner:
    """
    Wraps the google-genai SDK to produce SemanticWaypoint sequences.

    Parameters
    ----------
    api_key    : Google AI Studio API key.
    model_name : Gemini model ID.  Default = "gemini-2.0-flash".
    max_retries: How many times to retry on 429 / transient errors.
    retry_delay: Initial back-off seconds (doubles each retry).
    """

    def __init__(
        self,
        api_key:     str,
        model_name:  str = "gemini-2.0-flash",
        max_retries: int = 4,
        retry_delay: float = 5.0,
    ) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for GeminiPlanner.\n"
                "Install with:  pip install google-genai"
            ) from exc

        self._client      = genai.Client(api_key=api_key)
        self._model_name  = model_name
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._genai_types = genai_types
        log.info("GeminiPlanner ready  model=%s", model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, rgb_image: np.ndarray, command: str) -> List[SemanticWaypoint]:
        """
        Parameters
        ----------
        rgb_image : uint8 (H, W, 3) RGB array.
        command   : Natural language task description.

        Returns
        -------
        Ordered list of SemanticWaypoint objects.
        """
        h, w = rgb_image.shape[:2]
        pil_img = Image.fromarray(rgb_image.astype(np.uint8), mode="RGB")

        prompt = (
            f"Image size: {w}\u00d7{h} pixels.\n"
            f"Task command: \"{command}\"\n\n"
            f"{_SYSTEM_PROMPT}"
        )

        response_text = self._call_with_retry(pil_img, prompt)
        waypoints = self._parse(response_text, image_w=w, image_h=h)

        log.info(
            "Planned %d waypoints: %s",
            len(waypoints),
            [wp.action_type.value for wp in waypoints],
        )
        for i, wp in enumerate(waypoints):
            log.debug(
                "  [%d] %-10s pixel=(%d,%d)  conf=%.2f  gripper=%.1f  '%s'",
                i, wp.action_type.value, *wp.pixel_coords,
                wp.confidence, wp.gripper_state, wp.description,
            )
        return waypoints

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------

    def _call_with_retry(self, pil_img: Image.Image, prompt: str) -> str:
        types = self._genai_types
        delay = self._retry_delay

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model    = self._model_name,
                    contents = [pil_img, prompt],
                    config   = types.GenerateContentConfig(
                        temperature      = 0.2,
                        max_output_tokens= 2048,
                    ),
                )
                return response.text

            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "quota" in err_str.lower()
                )
                is_transient = (
                    "500" in err_str
                    or "503" in err_str
                    or "UNAVAILABLE" in err_str
                )

                if (is_rate_limit or is_transient) and attempt < self._max_retries:
                    # Parse suggested retry-after if present
                    suggested = self._parse_retry_after(err_str)
                    wait = max(delay, suggested)
                    log.warning(
                        "Gemini %s (attempt %d/%d) — retrying in %.0f s  [%s]",
                        "rate-limit" if is_rate_limit else "transient error",
                        attempt + 1, self._max_retries,
                        wait,
                        exc.__class__.__name__,
                    )
                    time.sleep(wait)
                    delay *= 2   # exponential back-off
                else:
                    # Non-retryable or exhausted retries
                    if is_rate_limit:
                        raise RuntimeError(
                            f"Gemini quota exhausted after {attempt + 1} attempt(s).\n"
                            "Options:\n"
                            "  1. Enable billing in Google AI Studio: https://aistudio.google.com\n"
                            "  2. Wait for quota reset (typically 1 min or 24 h)\n"
                            "  3. Use --model gemini-1.5-flash (separate quota bucket)\n"
                            f"Original error: {exc}"
                        ) from exc
                    raise

        raise RuntimeError("Unreachable")   # pragma: no cover

    @staticmethod
    def _parse_retry_after(err_str: str) -> float:
        """Extract 'retry in N seconds' hint from the error message."""
        m = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", err_str, re.IGNORECASE)
        return float(m.group(1)) if m else 0.0

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> str:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            return m.group(1).strip()
        start = text.find("[")
        end   = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text.strip()

    def _parse(
        self,
        response_text: str,
        image_w: int,
        image_h: int,
    ) -> List[SemanticWaypoint]:
        raw_json = self._extract_json(response_text)
        try:
            items = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            log.error("Gemini returned invalid JSON:\n%s\n---\n%s", exc, response_text)
            raise ValueError("Gemini planner: JSON parse failed") from exc

        waypoints: List[SemanticWaypoint] = []
        for idx, item in enumerate(items):
            try:
                action = ActionType(item["action_type"])
            except (KeyError, ValueError) as exc:
                log.warning("Skipping waypoint %d — unknown action: %s", idx, exc)
                continue

            u = int(np.clip(item.get("pixel_u", image_w // 2), 0, image_w - 1))
            v = int(np.clip(item.get("pixel_v", image_h // 2), 0, image_h - 1))

            waypoints.append(
                SemanticWaypoint(
                    action_type   = action,
                    pixel_coords  = (u, v),
                    description   = str(item.get("description", "")),
                    confidence    = float(np.clip(item.get("confidence", 1.0), 0.0, 1.0)),
                    gripper_state = float(np.clip(item.get("gripper_state", 0.0), 0.0, 1.0)),
                )
            )
        return waypoints
