import numpy as np
import pytest
from vla_framework.camera.rolling_buffer import RollingBuffer


DUMMY = np.zeros((2, 4, 3), dtype=np.uint8)


def test_empty_buffer():
    buf = RollingBuffer(maxlen=15)
    assert buf.depth == 0
    assert buf.head_index == -1
    assert buf.is_ready is False


def test_push_increments_index():
    buf = RollingBuffer(maxlen=15)
    assert buf.push(DUMMY) == 0
    assert buf.push(DUMMY) == 1
    assert buf.push(DUMMY) == 2


def test_maxlen_enforced():
    buf = RollingBuffer(maxlen=15)
    for _ in range(20):
        buf.push(DUMMY)
    assert buf.depth == 15


def test_is_ready():
    buf = RollingBuffer(maxlen=15)
    for _ in range(14):
        buf.push(DUMMY)
    assert buf.is_ready is False
    buf.push(DUMMY)
    assert buf.is_ready is True


def test_snapshot_atomic():
    buf = RollingBuffer(maxlen=15)
    for i in range(5):
        buf.push(np.full((2, 4, 3), i, dtype=np.uint8))
    head, frames = buf.snapshot()
    assert head == 4
    assert len(frames) == 5
    np.testing.assert_array_equal(frames[-1], np.full((2, 4, 3), 4, dtype=np.uint8))


def test_snapshot_is_copy():
    buf = RollingBuffer(maxlen=15)
    buf.push(DUMMY)
    _, frames = buf.snapshot()
    frames.clear()
    assert buf.depth == 1
