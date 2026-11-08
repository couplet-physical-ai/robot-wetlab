"""Software interlock: the mirror of the hardwired safety chain (docs/PLAN.md, "Safety model").

The door guard-lock and E-stop stop the arm in hardware with no software in the loop. This
class watches the same signals every runtime tick so the service can stop streaming, abort
the job, latch, and record why. A latch is cleared only by an operator, and only once every
condition is clear again.
"""
from __future__ import annotations

from .hw.interfaces import ArmState, DoorState
from .spec import Safety

# Reasons: door_open, door_unlocked, estop, arm_safety:<mode>, camera_lost, slot_ended, entry_requested


class Interlock:
    def __init__(self, safety: Safety):
        self.safety = safety
        self.tripped = False
        self.reason: str | None = None
        self.current: list[str] = []      # faults seen at the last evaluate()

    def faults(self, door: DoorState, arm: ArmState | None, camera_age_s: float, estop: bool) -> list[str]:
        f = []
        if estop or (arm is not None and arm.safety == "estop"):
            f.append("estop")
        if self.safety.require_door_closed_and_locked:
            if not door.closed:
                f.append("door_open")
            elif not door.locked:
                f.append("door_unlocked")
        if arm is not None and arm.safety not in ("normal", "estop"):
            f.append(f"arm_safety:{arm.safety}")
        if camera_age_s > self.safety.stop_on_camera_loss_s:
            f.append("camera_lost")
        return f

    def evaluate(self, door: DoorState, arm: ArmState, camera_age_s: float, estop: bool, active: bool) -> str | None:
        """Called every tick. While a job runs (`active`) any fault trips; an E-stop always trips.
        Idle faults only block can_start(). Returns the reason if this call newly latched."""
        self.current = self.faults(door, arm, camera_age_s, estop)
        if self.current and (active or "estop" in self.current):
            return self.trip("estop" if "estop" in self.current else self.current[0])
        return None

    def trip(self, reason: str) -> str | None:
        if self.tripped:
            return None   # first reason wins
        self.tripped, self.reason = True, reason
        return reason

    def can_start(self) -> bool:
        return not self.tripped and not self.current

    def reset(self, door: DoorState, arm: ArmState, camera_age_s: float, estop: bool) -> list[str]:
        """Operator reset. Clears the latch only if no fault remains; returns the remaining faults."""
        self.current = self.faults(door, arm, camera_age_s, estop)
        if not self.current:
            self.tripped, self.reason = False, None
        return self.current
