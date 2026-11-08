"""Arm-vs-scene clearance: link capsules (sampled points) and held labware against oriented boxes + keep-outs."""
from __future__ import annotations

import math

import numpy as np

from .. import geometry as geo
from ..plan import CellState
from ..spec import LabSpec
from .kinematics import ArmModel
from .scene import held_pose, scene_boxes

# UR5e physical link offsets along the parallel joint axes (UR ROS description, ur5e physical_parameters.yaml)
SHOULDER_OFFSET, ELBOW_OFFSET = 0.138, 0.131
LINK_RADII = {"base": 0.075, "shoulder": 0.075, "upper_arm": 0.06, "elbow": 0.06, "forearm": 0.05,
              "wrist1": 0.045, "wrist2": 0.045, "wrist3": 0.045, "tool": 0.045}
LINKS = list(LINK_RADII)
TOOL_END_ABOVE_TCP = 0.04   # tool body capsule stops this far above the TCP; fingers are not modelled
SPACING = 0.01              # m between sample points (capsules and held-labware surface)


def link_capsules(frames: list[np.ndarray]) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """(name, p0, p1, radius) per link from ArmModel.frames(q)."""
    o = [f[:3, 3] for f in frames]
    z = frames[1][:3, 2]  # joints 2, 3, 4 are parallel
    so, fo = SHOULDER_OFFSET * z, (SHOULDER_OFFSET - ELBOW_OFFSET) * z
    tool_end = o[7] - TOOL_END_ABOVE_TCP * frames[7][:3, 2]
    segs = [(o[0], o[1]), (o[1], o[1] + so), (o[1] + so, o[2] + so), (o[2] + so, o[2] + fo),
            (o[2] + fo, o[3] + fo), (o[3] + fo, o[4]), (o[4], o[5]), (o[5], o[6]), (o[6], tool_end)]
    return [(n, a, b, LINK_RADII[n]) for n, (a, b) in zip(LINKS, segs)]


def _box_surface_points(size: np.ndarray) -> np.ndarray:
    """Points on a box surface (local frame, centred) no more than SPACING apart."""
    axes = [np.linspace(-s / 2, s / 2, max(2, math.ceil(s / SPACING) + 1)) for s in size]
    pts = []
    for k in range(3):  # two faces normal to axis k
        i, j = [a for a in range(3) if a != k]
        u, v = np.meshgrid(axes[i], axes[j], indexing="ij")
        for side in (-1, 1):
            p = np.zeros((u.size, 3))
            p[:, i], p[:, j], p[:, k] = u.ravel(), v.ravel(), side * size[k] / 2
            pts.append(p)
    return np.unique(np.round(np.concatenate(pts), 9), axis=0)


def box_distance(points: np.ndarray, boxes: list[geo.Box]) -> np.ndarray:
    """Signed distance (M boxes, N points); negative inside."""
    C = np.array([b.T[:3, 3] for b in boxes])
    R = np.array([b.T[:3, :3] for b in boxes])
    H = np.array([b.size / 2 for b in boxes])
    local = np.einsum("mji,mnj->mni", R, points[None] - C[:, None])
    d = np.abs(local) - H[:, None]
    return np.linalg.norm(np.maximum(d, 0.0), axis=2) + np.minimum(d.max(axis=2), 0.0)


class CollisionScene:
    """Obstacles for one cell state (labware positions + held labware fixed), queried for many q."""

    def __init__(self, lab: LabSpec, state: CellState, door_angle_deg: float = 0.0, arm: ArmModel | None = None):
        self.lab, self.arm, self.held = lab, arm or ArmModel(lab), state.held
        self.boxes = [b for b in scene_boxes(lab, state, door_angle_deg) if b.collide and b.id != state.held]
        self.box_ids = [b.id for b in self.boxes]
        self.keepouts = [geo.Box(k.id, "keepout", geo.pose_matrix((np.add(k.min, k.max)) / 2),
                                 np.subtract(k.max, k.min)) for k in lab.keepouts]
        # fixed per-link sample parameters (all link lengths are constant)
        caps = link_capsules(self.arm.frames(self.arm.home_q))
        self._t = [np.linspace(0, 1, max(2, math.ceil(np.linalg.norm(b - a) / SPACING) + 1)) for _, a, b, _ in caps]
        self.names = LINKS + ([state.held] if state.held else [])
        self._link_of = np.concatenate([np.full(len(t), i) for i, t in enumerate(self._t)])
        self._radius = np.concatenate([np.full(len(t), r) for t, (_, _, _, r) in zip(self._t, caps)])
        if state.held:
            size = np.array(lab.labware_type(state.held).size)
            local = _box_surface_points(size) + [0.0, 0.0, size[2] / 2]  # labware origin is its bottom centre
            self._held_local = np.c_[local, np.ones(len(local))]
            self._link_of = np.r_[self._link_of, np.full(len(local), len(LINKS))]
            self._radius = np.r_[self._radius, np.zeros(len(local))]
        # base link vs its mount bench always touches
        self._always_skip = np.array([b == lab.arm.mount for b in self.box_ids])[:, None] & (self._link_of == 0)[None]

    def points(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Sample points (N,3) of the arm (+ held labware) and the TCP position."""
        frames = self.arm.frames(q)
        pts = [a + t[:, None] * (b - a) for t, (_, a, b, _) in zip(self._t, link_capsules(frames))]
        if self.held:
            T = held_pose(self.lab, self.held, frames[-1])
            pts.append((self._held_local @ T.T)[:, :3])
        return np.concatenate(pts), frames[-1][:3, 3]

    def check(self, q, ignore=frozenset()) -> tuple[float, tuple[str, str], tuple[str, str] | None]:
        """(min clearance, (link, box id), keep-out hit as (link|'tcp', keepout id) or None) at q.

        Clearance is arm/held-labware surface to obstacle; `ignore` = box ids skipped for every link.
        """
        pts, tcp = self.points(q)
        d = box_distance(pts, self.boxes) - self._radius[None]
        d = np.where(self._always_skip | np.array([b in ignore for b in self.box_ids])[:, None], np.inf, d)
        bi, pi = np.unravel_index(np.argmin(d), d.shape)
        hit = None
        if self.keepouts:
            k = box_distance(np.r_[pts, tcp[None]], self.keepouts) - np.r_[self._radius, 0.0][None]
            ki, kp = np.unravel_index(np.argmin(k), k.shape)
            if k[ki, kp] < 0:
                hit = ("tcp" if kp == len(pts) else self.names[self._link_of[kp]]), self.keepouts[ki].id
        return float(d[bi, pi]), (self.names[self._link_of[pi]], self.box_ids[bi]), hit


def clearance(lab: LabSpec, q, state: CellState, door_angle_deg: float = 0.0, ignore=frozenset()) -> tuple[float, tuple[str, str]]:
    d, pair, _ = CollisionScene(lab, state, door_angle_deg).check(q, ignore)
    return d, pair
