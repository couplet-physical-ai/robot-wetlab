"""Program IR: what a customer program compiles to. The cell only ever executes validated IR.

Customers either POST IR JSON directly or upload a Python script that uses `wetlab_sdk`
(run in a no-network sandbox) to emit it. Nothing a customer writes talks to hardware.
"""
from __future__ import annotations

import hashlib
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

IR_VERSION = "wetlab.ir/1"
MAX_STEPS = 2000


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Home(_S):
    op: Literal["home"]
    speed: float = Field(1.0, gt=0, le=1.0)   # fraction of the cell's speed limits


class MoveJoints(_S):
    op: Literal["move_joints"]
    q_deg: list[float] = Field(min_length=6, max_length=6)
    speed: float = Field(1.0, gt=0, le=1.0)


class Target(_S):
    """A TCP target, tool pointing down. Either an absolute room point or relative to labware."""
    xyz: list[float] | None = Field(None, min_length=3, max_length=3)   # room frame, metres
    labware: str | None = None
    well: str | None = None        # e.g. "A1"; requires labware
    location: str | None = None    # nest/instrument id: target its surface centre
    z_above: float = Field(0.05, ge=0.0, le=0.5)   # height of the TCP above the target point
    yaw_deg: float | None = None   # tool yaw in the room frame; default: the labware/location yaw

    @model_validator(mode="after")
    def _one_target(self):
        if sum(v is not None for v in (self.xyz, self.labware, self.location)) != 1:
            raise ValueError("target needs exactly one of xyz, labware, location")
        if self.well is not None and self.labware is None:
            raise ValueError("well needs labware")
        return self


class MoveTo(_S):
    op: Literal["move_to"]
    target: Target
    linear: bool = True            # straight-line TCP path (else joint-interpolated)
    speed: float = Field(1.0, gt=0, le=1.0)


class Gripper(_S):
    op: Literal["gripper"]
    width_mm: float = Field(ge=0.0, le=200.0)


class Pick(_S):
    op: Literal["pick"]
    labware: str
    approach_m: float = Field(0.08, ge=0.02, le=0.3)


class Place(_S):
    op: Literal["place"]
    labware: str
    at: str                         # destination location id
    approach_m: float = Field(0.08, ge=0.02, le=0.3)


class Wait(_S):
    op: Literal["wait"]
    seconds: float = Field(ge=0.0, le=3600.0)


class Capture(_S):
    op: Literal["capture"]
    camera: str | None = None
    tag: str = Field("frame", max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")


class Instrument(_S):
    op: Literal["instrument"]
    instrument: str                 # location id with a driver, e.g. shaker-1
    action: str = Field(max_length=32)
    params: dict[str, float | int | str | bool] = {}


class Mark(_S):
    op: Literal["mark"]
    text: str = Field(max_length=200)


Step = Annotated[Union[Home, MoveJoints, MoveTo, Gripper, Pick, Place, Wait, Capture, Instrument, Mark], Field(discriminator="op")]


class Program(_S):
    ir_version: Literal["wetlab.ir/1"] = IR_VERSION
    name: str = Field("untitled", max_length=120)
    cell: str | None = None
    steps: list[Step] = Field(min_length=1, max_length=MAX_STEPS)

    def sha256(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()
