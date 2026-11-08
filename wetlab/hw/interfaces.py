"""Hardware interfaces. The cell runtime only talks to these; sim and real both implement them.

Real implementations live next to this file (ur_rtde.py, door_relay.py, rtsp.py) and are
picked per cell in the service config. Safety note: these are the SOFTWARE mirror. The
authoritative interlock is hardware: the door's guard-locking switch is wired dual-channel
into the arm controller's safeguard-stop input (see docs/PLAN.md, "Safety").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass
class ArmState:
    q: list[float]                       # measured joint angles, rad
    gripper_mm: float
    safety: str = "normal"               # normal | protective_stop | safeguard_stop | estop | fault
    moving: bool = False
    t: float = 0.0                       # driver clock, s


@dataclass
class DoorState:
    closed: bool
    locked: bool
    angle_deg: float = 0.0               # sim only; real door reports 0/unknown
    source: str = "sim"


@dataclass
class CameraInfo:
    id: str
    width: int
    height: int
    fps: float
    extra: dict = field(default_factory=dict)


class ArmDriver(Protocol):
    def connect(self) -> None: ...
    def read_state(self) -> ArmState: ...
    def servo(self, q: Sequence[float], gripper_mm: float, dt: float) -> None:
        """Stream one joint setpoint (UR: servoJ at the plan rate). Must not block longer than dt."""
    def stop(self, reason: str) -> None:
        """Decelerate to a stop now (UR: stopJ / protective stop). Idempotent."""
    def reset(self) -> None:
        """Operator action: clear a protective stop so the arm can move again."""


class DoorController(Protocol):
    def state(self) -> DoorState: ...
    def lock(self) -> None: ...
    def unlock(self) -> None: ...


class CameraSource(Protocol):
    info: CameraInfo
    def frame(self) -> np.ndarray | None:
        """Latest RGB frame (H, W, 3) uint8, or None if the feed is down."""


class Instrument(Protocol):
    def run(self, action: str, params: dict) -> dict:
        """Run an instrument action to completion; return result data for the job's results."""
