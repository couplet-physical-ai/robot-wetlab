"""Scene geometry derived from the lab spec.

This is the ONE place room/door/bench/instrument/labware geometry is computed. The CPU twin
(collision + camera render) and the Isaac Sim USD generator both consume these boxes, so the
two twins cannot drift apart from each other or from lab/<cell>.yaml.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .spec import Door, LabSpec, pose_matrix


@dataclass
class Box:
    id: str
    kind: str           # floor | wall | door | bench | instrument | nest | labware | ceiling
    T: np.ndarray       # 4x4 pose of the box centre in the room frame
    size: np.ndarray    # full extents along the box's local x, y, z
    collide: bool = True

    def corners(self) -> np.ndarray:
        """8x3 corners in the room frame."""
        h = self.size / 2
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
        return c @ self.T[:3, :3].T + self.T[:3, 3]


def _box(id_, kind, center, size, yaw_deg=0.0, collide=True) -> Box:
    return Box(id_, kind, pose_matrix(center, (0, 0, yaw_deg)), np.asarray(size, float), collide)


def static_boxes(lab: LabSpec) -> list[Box]:
    """Floor, walls (cut around doors), benches, instrument bodies and nest markers."""
    X, Y, Z = lab.room.size
    t = lab.room.wall_thickness
    out = [_box("floor", "floor", [X / 2, Y / 2, -0.025], [X + 2 * t, Y + 2 * t, 0.05])]

    # walls: (name, axis-along, fixed coord, length)
    walls = {
        "south": ("x", -t / 2, X), "north": ("x", Y + t / 2, X),
        "west": ("y", -t / 2, Y), "east": ("y", X + t / 2, Y),
    }
    for name, (along, fixed, length) in walls.items():
        doors = sorted((d for d in lab.doors if d.wall == name), key=lambda d: d.hinge_x)
        cuts, cursor = [], -t if along == "x" else 0.0
        end = length + (t if along == "x" else 0.0)  # south/north walls also cover the corners
        for d in doors:
            a, b = _door_span(d)
            cuts.append((cursor, a, 0.0, Z))   # solid wall up to the door
            cuts.append((a, b, d.height, Z))   # lintel above the door
            cursor = b
        cuts.append((cursor, end, 0.0, Z))
        for i, (a, b, z0, z1) in enumerate(cuts):
            if b - a < 1e-6 or z1 - z0 < 1e-6:
                continue
            mid, ln = (a + b) / 2, b - a
            if along == "x":
                c, s = [mid, fixed, (z0 + z1) / 2], [ln, t, z1 - z0]
            else:
                c, s = [fixed, mid, (z0 + z1) / 2], [t, ln, z1 - z0]
            out.append(_box(f"wall-{name}-{i}", "wall", c, s))
    out.append(_box("ceiling", "ceiling", [X / 2, Y / 2, Z + 0.025], [X, Y, 0.05], collide=True))

    for b in lab.benches:
        x, y, _ = b.pose.xyz
        sx, sy, h = b.size
        out.append(_box(b.id, "bench", [x, y, h / 2], [sx, sy, h], b.pose.rpy_deg[2]))

    for loc in lab.locations:
        x, y = loc.xy
        if loc.body is not None:
            sx, sy, sz = loc.body.size
            out.append(_box(loc.id, "instrument", [x, y, loc.top - sz / 2], [sx, sy, sz], loc.yaw_deg))
        else:
            out.append(_box(loc.id, "nest", [x, y, loc.top - 0.002], [0.135, 0.093, 0.004], loc.yaw_deg, collide=False))
    return out


def _door_span(d: Door) -> tuple[float, float]:
    a = d.hinge_x if d.hinge_side == "left" else d.hinge_x - d.width
    return a, a + d.width


def door_box(lab: LabSpec, door: Door, angle_deg: float) -> Box:
    """Door panel at `angle_deg` open (0 = closed). Hinge axis is vertical at hinge_x on its wall."""
    t = lab.room.wall_thickness
    X, Y, _ = lab.room.size
    # hinge point and the wall direction (closed panel runs from the hinge along +u)
    if door.wall == "south":
        hinge, u, out_n = np.array([door.hinge_x, -t / 2]), np.array([1.0, 0]), np.array([0, -1.0])
    elif door.wall == "north":
        hinge, u, out_n = np.array([door.hinge_x, Y + t / 2]), np.array([1.0, 0]), np.array([0, 1.0])
    elif door.wall == "west":
        hinge, u, out_n = np.array([-t / 2, door.hinge_x]), np.array([0, 1.0]), np.array([-1.0, 0])
    else:
        hinge, u, out_n = np.array([X + t / 2, door.hinge_x]), np.array([0, 1.0]), np.array([1.0, 0])
    if door.hinge_side == "right":
        u = -u
    n = out_n if door.swing == "outward" else -out_n
    a = math.radians(max(0.0, min(angle_deg, door.max_open_deg)))
    along = math.cos(a) * u + math.sin(a) * n
    centre = hinge + along * door.width / 2
    yaw = math.degrees(math.atan2(along[1], along[0]))
    return _box(door.id, "door", [centre[0], centre[1], door.height / 2], [door.width, door.thickness, door.height], yaw)


def door_hinge(lab: LabSpec, door: Door) -> tuple[np.ndarray, float]:
    """(hinge point on the floor, signed opening direction about +Z) for joint authoring."""
    closed, opened = door_box(lab, door, 0.0), door_box(lab, door, 10.0)
    yaw0 = math.atan2(closed.T[1, 0], closed.T[0, 0])
    yaw1 = math.atan2(opened.T[1, 0], opened.T[0, 0])
    c = closed.T[:3, 3] - closed.T[:3, 0] * door.width / 2
    return np.array([c[0], c[1], 0.0]), math.copysign(1.0, (yaw1 - yaw0 + math.pi) % (2 * math.pi) - math.pi)


# -- labware ---------------------------------------------------------------------
# Labware frame: origin at the centre of the bottom face, +x along the long (12-column) side,
# +y toward row A. A location frame: origin on the nest surface centre, yaw about +Z.

def location_frame(lab: LabSpec, loc_id: str) -> np.ndarray:
    loc = lab.location(loc_id)
    return pose_matrix([loc.xy[0], loc.xy[1], loc.top], (0, 0, loc.yaw_deg))


def labware_box(lab: LabSpec, lw_id: str, T_labware: np.ndarray) -> Box:
    size = np.array(lab.labware_type(lw_id).size, float)
    T = T_labware.copy()
    T[:3, 3] = T_labware[:3, :3] @ np.array([0, 0, size[2] / 2]) + T_labware[:3, 3]
    return Box(lw_id, "labware", T, size)


def tool_down(yaw_deg: float = 0.0) -> np.ndarray:
    """Rotation with tool z pointing straight down and tool x at `yaw_deg` in the room XY plane."""
    a = math.radians(yaw_deg)
    x = np.array([math.cos(a), math.sin(a), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def grasp_tcp(lab: LabSpec, lw_id: str, T_labware: np.ndarray) -> np.ndarray:
    """TCP pose that grasps the labware: fingers close across the short (y) side."""
    lt = lab.labware_type(lw_id)
    T = np.eye(4)
    yaw = math.degrees(math.atan2(T_labware[1, 0], T_labware[0, 0]))
    T[:3, :3] = tool_down(yaw)
    T[:3, 3] = T_labware[:3, 3] + T_labware[:3, :3] @ np.array([0, 0, lt.size[2] - lt.grip.z_from_top])
    return T


def well_point(lab: LabSpec, lw_id: str, well: str, T_labware: np.ndarray) -> np.ndarray:
    """Room-frame point at the top centre of a well, e.g. well='A1'."""
    lt = lab.labware_type(lw_id)
    if lt.wells is None:
        raise ValueError(f"{lw_id} has no wells")
    w = lt.wells
    row, col = ord(well[0].upper()) - ord("A"), int(well[1:]) - 1
    if not (0 <= row < w.rows and 0 <= col < w.cols):
        raise ValueError(f"{lw_id}: no well {well}")
    L, W, H = lt.size
    local = np.array([-L / 2 + w.a1_offset[0] + col * w.pitch, W / 2 - w.a1_offset[1] - row * w.pitch, H, 1.0])
    return (T_labware @ local)[:3]
