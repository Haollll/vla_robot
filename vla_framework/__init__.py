"""vla_framework — Hierarchical Vision-Language-Action robot control."""
from .pipeline import VLAPipeline
from .config   import VLAConfig, ActionType
from .interfaces import (
    SimStepCallback,
    RobotStateProtocol,
    SnapshotResultProtocol,
    RobotInterface,
    SimRobotInterface,
    CameraStreamerInterface,
)

__all__ = [
    "VLAPipeline",
    "VLAConfig",
    "ActionType",
    # Hardware-agnostic interfaces
    "SimStepCallback",
    "RobotStateProtocol",
    "SnapshotResultProtocol",
    "RobotInterface",
    "SimRobotInterface",
    "CameraStreamerInterface",
]
