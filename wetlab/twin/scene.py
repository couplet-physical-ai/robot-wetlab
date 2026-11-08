"""Where everything is for a given cell state: labware poses and the full box scene (from geometry.py)."""
from __future__ import annotations

import numpy as np

from .. import geometry as geo
from ..plan import CellState
from ..spec import LabSpec
from .kinematics import ArmModel


def held_pose(lab: LabSpec, lw_id: str, T_tcp: np.ndarray) -> np.ndarray:
    """Labware pose in the gripper: the grasp is always geometry.grasp_tcp, so labware = TCP * grasp^-1."""
    return T_tcp @ np.linalg.inv(geo.grasp_tcp(lab, lw_id, np.eye(4)))


def labware_poses(lab: LabSpec, state: CellState, q=None) -> dict[str, np.ndarray]:
    out = {lw: geo.location_frame(lab, loc) for lw, loc in state.labware_at.items()}
    if state.held:
        out[state.held] = held_pose(lab, state.held, ArmModel(lab).tcp(state.q if q is None else q))
    return out


def scene_boxes(lab: LabSpec, state: CellState, door_angle_deg: float = 0.0) -> list[geo.Box]:
    """Static room + every door at `door_angle_deg` + all labware (held labware rides on the TCP)."""
    boxes = geo.static_boxes(lab) + [geo.door_box(lab, d, door_angle_deg) for d in lab.doors]
    return boxes + [geo.labware_box(lab, lw, T) for lw, T in labware_poses(lab, state).items()]
