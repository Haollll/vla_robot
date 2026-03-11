from collections import deque
import threading
import numpy as np
from typing import List, Tuple


class RollingBuffer:
    """Thread-safe rolling buffer for raw side-by-side stereo frames."""

    def __init__(self, maxlen: int = 15) -> None:
        self._deque: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._head_index: int = -1
        self._maxlen = maxlen

    def push(self, frame: np.ndarray) -> int:
        """
        Store one raw SBS frame. Returns the frame index assigned to it.
        Frame indices are monotonically increasing from 0.
        Thread-safe — safe to call from a background capture thread.
        """
        with self._lock:
            self._head_index += 1
            self._deque.append(frame)
            return self._head_index

    def snapshot(self) -> Tuple[int, List[np.ndarray]]:
        """
        Atomically read the current buffer state.
        Returns (head_index, list_of_frames) where:
          - head_index is the index of the most recently pushed frame
          - list_of_frames is a list copy of all frames currently stored
            (may be fewer than maxlen if buffer not yet full)
        Thread-safe — push() cannot interleave during this call.
        """
        with self._lock:
            return self._head_index, list(self._deque)

    @property
    def depth(self) -> int:
        """Number of frames currently stored (0 to maxlen)."""
        with self._lock:
            return len(self._deque)

    @property
    def head_index(self) -> int:
        """Index of the most recently pushed frame. -1 if empty."""
        with self._lock:
            return self._head_index

    @property
    def is_ready(self) -> bool:
        """True when buffer holds exactly maxlen frames (ready for process_stack)."""
        with self._lock:
            return len(self._deque) == self._maxlen
