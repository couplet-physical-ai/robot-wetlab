"""Lab spec: the one file (lab/<cell>.yaml) both the real cell and its twins are built from."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pose(_M):
    xyz: list[float] = Field(min_length=3, max_length=3)
    rpy_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0], min_length=3, max_length=3)

    def matrix(self) -> np.ndarray:
        return pose_matrix(self.xyz, self.rpy_deg)


class Survey(_M):
    status: Literal["nominal", "surveyed"] = "nominal"
    date: str = ""
    by: str = ""


class Room(_M):
    size: list[float] = Field(min_length=3, max_length=3)
    wall_thickness: float = 0.12


class DoorInterlock(_M):
    switch: str
    lock: str


class Door(_M):
    id: str
    wall: Literal["south", "north", "west", "east"]
    hinge_x: float
    width: float
    height: float
    thickness: float = 0.045
    swing: Literal["outward", "inward"] = "outward"
    hinge_side: Literal["left", "right"] = "left"
    max_open_deg: float = 95.0
    interlock: DoorInterlock | None = None


class Bench(_M):
    id: str
    pose: Pose
    size: list[float] = Field(min_length=3, max_length=3)


class Tool(_M):
    model: str
    tcp_offset_m: float
    open_mm: float = 50.0
    max_payload_kg: float = 1.0


class ArmLimits(_M):
    tcp_speed_mps: float = 0.25
    joint_speed_dps: float = 60.0
    joint_accel_dps2: float = 120.0


class Arm(_M):
    id: str
    model: Literal["ur5e"]
    mount: str
    base: Pose
    tool: Tool
    home_q_deg: list[float] = Field(min_length=6, max_length=6)
    limits: ArmLimits = ArmLimits()


class Body(_M):
    size: list[float] = Field(min_length=3, max_length=3)


class Location(_M):
    id: str
    kind: Literal["nest", "instrument"]
    xy: list[float] = Field(min_length=2, max_length=2)
    top: float
    yaw_deg: float = 0.0
    body: Body | None = None      # instrument housing below `top`
    driver: str | None = None


class Wells(_M):
    rows: int
    cols: int
    pitch: float
    a1_offset: list[float] = Field(min_length=2, max_length=2)  # from the plate's -x,+y corner (A1 is top-left)
    depth: float


class Grip(_M):
    width_mm: float
    z_from_top: float


class LabwareType(_M):
    size: list[float] = Field(min_length=3, max_length=3)
    wells: Wells | None = None
    grip: Grip


class Labware(_M):
    id: str
    type: str
    at: str


class Intrinsics(_M):
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = Field(default_factory=lambda: [0.0] * 5)  # OpenCV k1 k2 p1 p2 k3

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


class Extrinsics(_M):
    position: list[float] = Field(min_length=3, max_length=3)
    look_at: list[float] | None = None
    up: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    quat_wxyz: list[float] | None = None  # OpenCV optical frame orientation in room frame (from calibration)

    @model_validator(mode="after")
    def _one_orientation(self):
        if (self.look_at is None) == (self.quat_wxyz is None):
            raise ValueError("extrinsics needs exactly one of look_at or quat_wxyz")
        return self


class Calibration(_M):
    status: Literal["nominal", "calibrated"] = "nominal"
    date: str = ""
    rms_px: float | None = None


class Camera(_M):
    id: str
    model: str = ""
    resolution: list[int] = Field(min_length=2, max_length=2)
    fps: float = 15.0
    intrinsics: Intrinsics
    extrinsics: Extrinsics
    calibration: Calibration = Calibration()

    def T_room_cam(self) -> np.ndarray:
        """4x4 pose of the OpenCV optical frame (x right, y down, z forward) in the room frame."""
        e = self.extrinsics
        p = np.array(e.position, float)
        if e.quat_wxyz is not None:
            R = quat_to_matrix(e.quat_wxyz)
        else:
            z = np.array(e.look_at, float) - p
            z /= np.linalg.norm(z)
            x = np.cross(z, np.array(e.up, float))  # right = forward x up
            if np.linalg.norm(x) < 1e-9:            # looking along `up` (e.g. straight down): pick +Y as up
                x = np.cross(z, np.array([0.0, 1.0, 0.0]))
            x /= np.linalg.norm(x)
            y = np.cross(z, x)                      # down
            R = np.column_stack([x, y, z])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T


class Keepout(_M):
    id: str
    min: list[float] = Field(min_length=3, max_length=3)
    max: list[float] = Field(min_length=3, max_length=3)


class Safety(_M):
    require_door_closed_and_locked: bool = True
    stop_on_camera_loss_s: float = 2.0
    blank_stream_when_door_open: bool = True
    min_clearance_m: float = 0.01


class Billing(_M):
    slot_minute: float = 0.0
    arm_second: float = 0.0
    sim_dry_run: float = 0.0
    isaac_minute: float = 0.0
    cancel_refund_hours: float = 24.0


class LabSpec(_M):
    id: str
    name: str = ""
    survey: Survey = Survey()
    room: Room
    doors: list[Door]
    benches: list[Bench]
    arms: list[Arm] = Field(min_length=1)
    locations: list[Location]
    labware_types: dict[str, LabwareType]
    labware: list[Labware]
    cameras: list[Camera] = Field(min_length=1)
    keepouts: list[Keepout] = []
    safety: Safety = Safety()
    billing: Billing = Billing()

    @model_validator(mode="after")
    def _refs(self):
        loc_ids = {l.id for l in self.locations}
        ids = [x.id for x in [*self.doors, *self.benches, *self.arms, *self.locations, *self.labware, *self.cameras]]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"duplicate ids: {sorted(dup)}")
        bench_ids = {b.id for b in self.benches}
        for a in self.arms:
            if a.mount not in bench_ids:
                raise ValueError(f"arm {a.id} mounted on unknown bench {a.mount}")
        occupied: dict[str, str] = {}
        for lw in self.labware:
            if lw.type not in self.labware_types:
                raise ValueError(f"labware {lw.id}: unknown type {lw.type}")
            if lw.at not in loc_ids:
                raise ValueError(f"labware {lw.id}: unknown location {lw.at}")
            if lw.at in occupied:
                raise ValueError(f"location {lw.at} holds both {occupied[lw.at]} and {lw.id}")
            occupied[lw.at] = lw.id
        return self

    # -- lookups -------------------------------------------------------------
    @property
    def arm(self) -> Arm:
        return self.arms[0]  # shortcut: one arm per cell in the MVP; index by id when a cell gets two

    def location(self, loc_id: str) -> Location:
        return _by_id(self.locations, loc_id, "location")

    def camera(self, cam_id: str | None = None) -> Camera:
        return self.cameras[0] if cam_id is None else _by_id(self.cameras, cam_id, "camera")

    def labware_item(self, lw_id: str) -> Labware:
        return _by_id(self.labware, lw_id, "labware")

    def labware_type(self, lw_id: str) -> LabwareType:
        return self.labware_types[self.labware_item(lw_id).type]

    def sha256(self) -> str:
        """Stable hash of the spec; stamped into the USD stage and every plan so sim/real drift is detectable."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def _by_id(items, item_id, what):
    for it in items:
        if it.id == item_id:
            return it
    raise KeyError(f"unknown {what}: {item_id}")


def load_lab(path: str | Path) -> LabSpec:
    return LabSpec.model_validate(yaml.safe_load(Path(path).read_text()))


DEFAULT_LAB = Path(__file__).resolve().parent.parent / "lab" / "cell-a.yaml"


# -- math helpers shared by every twin backend ----------------------------------
def rpy_matrix(rpy_deg) -> np.ndarray:
    r, p, y = (math.radians(v) for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def pose_matrix(xyz, rpy_deg=(0, 0, 0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(rpy_deg)
    T[:3, 3] = xyz
    return T


def quat_to_matrix(q_wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, float) / np.linalg.norm(q_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> list[float]:
    """3x3 rotation -> [w, x, y, z] (Shepperd)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
    return q
