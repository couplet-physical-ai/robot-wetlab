"""Plan + cell state: the contract between the planner (dry-run), the executor and the twins.

A Plan is a fixed-rate joint trajectory. The SAME waypoints are rendered in the CPU twin,
played in Isaac Sim, and streamed to the real arm (servoJ), so a dry-run is the run.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

DT = 0.02  # s, 50 Hz plan / servo rate


class CellState(BaseModel):
    q: list[float] = Field(min_length=6, max_length=6)      # joint angles, rad (UR order: base..wrist3)
    gripper_mm: float = 110.0
    held: str | None = None                                  # labware id in the gripper
    labware_at: dict[str, str] = {}                          # labware id -> location id (held ones absent)


class Waypoint(BaseModel):
    t: float
    q: list[float]
    gripper_mm: float


class PlanEvent(BaseModel):
    t: float
    step: int
    kind: str   # step_start | attach | detach | capture | instrument | wait | mark
    data: dict = {}


class Violation(BaseModel):
    step: int | None
    code: str   # schema | unknown_ref | unreachable | joint_limit | collision | keepout | speed | state | payload
    message: str
    t: float | None = None


class Plan(BaseModel):
    ok: bool
    program_name: str
    cell: str
    lab_sha: str
    ir_sha: str
    dt: float = DT
    duration_s: float = 0.0
    waypoints: list[Waypoint] = []
    events: list[PlanEvent] = []
    violations: list[Violation] = []
    start_state: CellState
    final_state: CellState
    stats: dict = {}   # max_tcp_speed_mps, min_clearance_m, path_length_m, ...
