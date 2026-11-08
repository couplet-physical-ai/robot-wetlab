"""The virtual cell: CPU-twin implementations of the real hardware interfaces (interfaces.py).

The service runs the same runtime against these as against ur_rtde / door relay / RTSP.
`clock` is injectable so tests can drive time.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import numpy as np

from ..plan import CellState
from ..spec import LabSpec
from .interfaces import ArmState, CameraInfo, DoorState


class VirtualArm:
    """servo() moves the measured joints toward the command with a tiny first-order lag (tau),
    so q_meas != q_cmd slightly, like a real servo loop. stop() latches a protective stop."""

    def __init__(self, q0: Sequence[float], gripper_mm: float = 110.0, tau_s: float = 0.004, clock: Callable[[], float] = time.time):
        self.q = list(q0)
        self.gripper_mm = gripper_mm
        self.tau_s = tau_s
        self.clock = clock
        self.safety = "normal"
        self.estop = False          # the physical E-stop button (tests press it)
        self.stop_reason: str | None = None
        self.n_servo = 0            # every servo() call, accepted or not
        self.moving = False

    def connect(self) -> None:
        pass

    def read_state(self) -> ArmState:
        return ArmState(list(self.q), self.gripper_mm, "estop" if self.estop else self.safety, self.moving, self.clock())

    def servo(self, q: Sequence[float], gripper_mm: float, dt: float) -> None:
        self.n_servo += 1
        if self.safety != "normal" or self.estop:
            self.moving = False
            return
        a = 1.0 - math.exp(-dt / self.tau_s)
        new = [m + a * (c - m) for m, c in zip(self.q, q)]
        self.moving = any(abs(n - m) > 1e-6 for n, m in zip(new, self.q))
        self.q, self.gripper_mm = new, float(gripper_mm)

    def stop(self, reason: str) -> None:
        if self.safety == "normal":
            self.safety, self.stop_reason = "protective_stop", reason
        self.moving = False

    def reset(self) -> None:
        if not self.estop:      # a pressed E-stop must be released physically first
            self.safety, self.stop_reason = "normal", None


class VirtualDoor:
    """open()/close() are a person's actions and animate over ~open_s. A locked door can still be
    forced open (the physical world wins). The lock engages only while the door is closed."""

    def __init__(self, lab: LabSpec, clock: Callable[[], float] = time.time, open_s: float = 1.5):
        self.max_deg = lab.doors[0].max_open_deg
        self.rate = self.max_deg / open_s
        self.clock = clock
        self._a0, self._t0, self._target = 0.0, clock(), 0.0
        self._lock_cmd = False

    def angle(self) -> float:
        step = self.rate * max(0.0, self.clock() - self._t0)
        return min(self._target, self._a0 + step) if self._target > self._a0 else max(self._target, self._a0 - step)

    def _go(self, target: float) -> None:
        self._a0, self._t0, self._target = self.angle(), self.clock(), target

    def open(self) -> None:
        self._go(self.max_deg)

    def close(self) -> None:
        self._go(0.0)

    def lock(self) -> None:
        self._lock_cmd = True

    def unlock(self) -> None:
        self._lock_cmd = False

    def state(self) -> DoorState:
        a = self.angle()
        closed = a < 1.0
        return DoorState(closed=closed, locked=self._lock_cmd and closed, angle_deg=a, source="sim")


class TwinCamera:
    """Renders the live virtual cell from `state_fn` (-> CellState) with the twin renderer.
    Door open + lab.safety.blank_stream_when_door_open -> a dark privacy frame instead."""

    def __init__(self, lab: LabSpec, state_fn: Callable[[], CellState], door: VirtualDoor | None = None, camera_id: str | None = None):
        cam = lab.camera(camera_id)
        self.lab, self.state_fn, self.door, self.camera_id = lab, state_fn, door, cam.id
        self.info = CameraInfo(cam.id, cam.resolution[0], cam.resolution[1], cam.fps, {"source": "twin"})
        self.alive = True           # tests flip this to simulate a lost feed
        self._renderer = None
        self._privacy = None

    def frame(self) -> np.ndarray | None:
        if not self.alive:
            return None
        door = self.door.state() if self.door else DoorState(True, True)
        if not door.closed and self.lab.safety.blank_stream_when_door_open:
            return self.privacy_frame()
        if self._renderer is None:
            from ..twin.render import TwinRenderer
            self._renderer = TwinRenderer(self.lab, camera_id=self.camera_id)
        return self._renderer.render(self.state_fn(), door_angle_deg=door.angle_deg)

    def privacy_frame(self) -> np.ndarray:
        if self._privacy is None:
            from PIL import Image, ImageDraw, ImageFont
            w, h = self.info.width, self.info.height
            im = Image.new("RGB", (w, h), (18, 18, 22))
            ImageDraw.Draw(im).text((w // 2, h // 2), "Door open: stream paused for privacy", fill=(170, 170, 180),
                                    font=ImageFont.load_default(size=max(16, h // 24)), anchor="mm")
            self._privacy = np.asarray(im)
        return self._privacy


class SimShaker:
    def __init__(self):
        self.runs: list[tuple[str, dict]] = []

    def run(self, action: str, params: dict) -> dict:
        self.runs.append((action, dict(params)))
        return {"rpm": float(params.get("rpm", 0)), "seconds": float(params.get("seconds", 0)), "ok": action == "shake"}
