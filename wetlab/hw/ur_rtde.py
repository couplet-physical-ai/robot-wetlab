"""UR5e ArmDriver over RTDE (pip install ur_rtde). UNTESTED ON HARDWARE: a stub to be validated
on the real cell (and against URSim first). The safety chain is hardwired; see interfaces.py.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from .interfaces import ArmState

# ur_rtde getSafetyMode() codes -> ArmState.safety
_SAFETY = {1: "normal", 2: "normal", 3: "protective_stop", 5: "safeguard_stop", 6: "estop", 7: "estop",
           12: "safeguard_stop", 13: "safeguard_stop"}


class Gripper(Protocol):
    def move(self, width_mm: float) -> None: ...
    def width_mm(self) -> float: ...


class NullGripper:
    """Placeholder gripper; remembers the last command."""

    def __init__(self, width_mm: float = 110.0):
        self._w = width_mm

    def move(self, width_mm: float) -> None:
        self._w = width_mm

    def width_mm(self) -> float:
        return self._w


class URRTDEArm:
    def __init__(self, host: str, gripper: Gripper | None = None, lookahead_s: float = 0.1, gain: int = 300):
        self.host, self.gripper = host, gripper or NullGripper()
        self.lookahead_s, self.gain = lookahead_s, gain
        self.c = self.r = None
        self._stopped = False

    def connect(self) -> None:
        import rtde_control
        import rtde_receive  # lazy: ur_rtde is an optional dependency
        self.c = rtde_control.RTDEControlInterface(self.host)
        self.r = rtde_receive.RTDEReceiveInterface(self.host)

    def read_state(self) -> ArmState:
        mode = _SAFETY.get(self.r.getSafetyMode(), "fault")
        if mode == "normal" and self.r.isProtectiveStopped():
            mode = "protective_stop"
        speed = max(abs(v) for v in self.r.getActualQd())
        return ArmState(list(self.r.getActualQ()), self.gripper.width_mm(), mode, speed > 1e-3, self.r.getTimestamp())

    def servo(self, q: Sequence[float], gripper_mm: float, dt: float) -> None:
        if self._stopped:
            return
        self.c.servoJ(list(q), 0.0, 0.0, dt, self.lookahead_s, self.gain)   # speed/accel unused by servoJ
        if abs(gripper_mm - self.gripper.width_mm()) > 0.5:
            self.gripper.move(gripper_mm)

    def stop(self, reason: str) -> None:
        self._stopped = True
        self.c.servoStop()
        self.c.stopJ(2.0)   # rad/s^2 controlled stop; the hardware safeguard stop already acted on door/E-stop

    def reset(self) -> None:
        # Clearing a protective stop goes through the dashboard server (port 29999).
        import dashboard_client
        d = dashboard_client.DashboardClient(self.host)
        d.connect()
        d.closeSafetyPopup()
        d.unlockProtectiveStop()
        d.disconnect()
        self._stopped = False
