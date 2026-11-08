"""CPU twin camera: renders the cell through the spec camera (OpenCV intrinsics + distortion) with PIL.

Painter's algorithm: the room shell (door outside the room, floor, walls) is drawn first, then benches,
then everything resting on/above them (instruments, nests, labware, arm capsules) far-to-near.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import geometry as geo
from ..media import Mp4Writer
from ..plan import CellState, Plan
from ..spec import Camera, LabSpec
from .collision import link_capsules
from .kinematics import ArmModel
from .scene import labware_poses

ZNEAR = 0.05
LIGHT = np.array([-0.35, -0.55, 1.0]) / np.linalg.norm([-0.35, -0.55, 1.0])
COLORS = {"floor": (92, 98, 108), "wall": (210, 213, 218), "ceiling": (230, 230, 230), "door": (176, 122, 74),
          "bench": (226, 228, 230), "instrument": (64, 78, 98), "nest": (222, 180, 40)}
LABWARE_COLORS = [(110, 185, 235), (175, 140, 225), (120, 210, 150), (240, 150, 110)]
LINK_COLORS = {"tube": (200, 204, 210), "joint": (150, 176, 204), "tool": (62, 66, 72), "finger": (32, 34, 38)}
JOINT_LINKS = {"base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"}
# box faces as corner indices (Box.corners order: ix*4 + iy*2 + iz) and the local axis of their outward normal
FACES = [([0, 1, 3, 2], 0, -1), ([4, 6, 7, 5], 0, 1), ([0, 4, 5, 1], 1, -1),
         ([2, 3, 7, 6], 1, 1), ([0, 2, 6, 4], 2, -1), ([1, 5, 7, 3], 2, 1)]


def _distort(pc: np.ndarray, cam: Camera) -> np.ndarray:
    """Camera-frame points (N,3), z > 0 -> pixels (N,2), OpenCV pinhole + k1 k2 p1 p2 k3."""
    i = cam.intrinsics
    k1, k2, p1, p2, k3 = i.dist
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.c_[i.fx * xd + i.cx, i.fy * yd + i.cy]


def _room_to_cam(T_room_cam: np.ndarray, pts_room) -> np.ndarray:
    return (np.asarray(pts_room, float) - T_room_cam[:3, 3]) @ T_room_cam[:3, :3]


def project_points(lab: LabSpec, cam: Camera | str | None, pts_room) -> np.ndarray:
    """Room-frame points (N,3) -> pixels (N,2) for a camera (object or id). Points must be in front of it.
    The renderer uses exactly this path (_room_to_cam + _distort)."""
    cam = cam if isinstance(cam, Camera) else lab.camera(cam)
    return _distort(_room_to_cam(cam.T_room_cam(), np.reshape(pts_room, (-1, 3))), cam)


class TwinRenderer:
    def __init__(self, lab: LabSpec, camera_id: str | None = None):
        self.lab, self.cam, self.arm = lab, lab.camera(camera_id), ArmModel(lab)
        self.W, self.H = self.cam.resolution
        self.T = self.cam.T_room_cam()
        self.c = self.T[:3, 3]
        i = self.cam.intrinsics
        mx, my = 1.5 * max(i.cx, self.W - i.cx) / i.fx, 1.5 * max(i.cy, self.H - i.cy) / i.fy  # frustum + margin
        self.planes = [(np.array([0, 0, 1.0]), -ZNEAR), (np.array([-1.0, 0, mx]), 0.0), (np.array([1.0, 0, mx]), 0.0),
                       (np.array([0, -1.0, my]), 0.0), (np.array([0, 1.0, my]), 0.0)]
        self.lw_color = {t: LABWARE_COLORS[k % len(LABWARE_COLORS)] for k, t in enumerate(lab.labware_types)}
        self.static = [d for b in geo.static_boxes(lab) for d in self._box(b)]
        self.font = ImageFont.load_default(size=18)
        self.small = ImageFont.load_default(size=13)

    # -- geometry -> 2D draw items: (layer, depth, kind, payload) ------------------------
    def _cam(self, p: np.ndarray) -> np.ndarray:
        return _room_to_cam(self.T, p)

    def _clip_poly(self, poly: list[np.ndarray]) -> list[np.ndarray]:
        for n, d in self.planes:
            out = []
            for k, p in enumerate(poly):
                q = poly[(k + 1) % len(poly)]
                fp, fq = n @ p + d, n @ q + d
                if fp >= 0:
                    out.append(p)
                if (fp >= 0) != (fq >= 0):
                    out.append(p + (q - p) * (fp / (fp - fq)))
            poly = out
            if len(poly) < 3:
                return []
        return poly

    def _poly_px(self, poly: list[np.ndarray]) -> list[tuple]:
        """Clipped camera-frame polygon -> distorted pixel outline (edges subdivided so they curve)."""
        pts = []
        for k, p in enumerate(poly):
            q = poly[(k + 1) % len(poly)]
            n = max(1, min(24, int(np.linalg.norm(p[:2] / p[2] - q[:2] / q[2]) * self.cam.intrinsics.fx / 40)))
            pts.extend(p + (q - p) * s for s in np.arange(n) / n)
        return [tuple(v) for v in _distort(np.array(pts), self.cam)]

    def _box(self, b: geo.Box, color=None) -> list[tuple]:
        base = np.array(color or COLORS.get(b.kind, (180, 180, 180)), float)
        corners = self._cam(b.corners())
        cam_local = b.T[:3, :3].T @ (self.c - b.T[:3, 3])
        polys = []
        for idx, axis, sign in FACES:
            if sign * cam_local[axis] <= b.size[axis] / 2:  # back face
                continue
            poly = self._clip_poly([corners[k] for k in idx])
            if poly:
                shade = 0.5 + 0.5 * max(0.0, float(sign * b.T[:3, axis] @ LIGHT))
                polys.append((self._poly_px(poly), tuple(int(v) for v in base * shade)))
        if not polys:
            return []
        layer = {"floor": 0, "wall": 0, "ceiling": 0, "bench": 1}.get(b.kind, 2)
        if b.kind == "door" and all(d.swing == "outward" for d in self.lab.doors):
            layer = -1  # outside the room: behind everything, the wall opening frames it
        depth = float(np.linalg.norm(b.T[:3, 3] - self.c))
        return [(layer, depth, "polys", polys)]

    def _capsule(self, a: np.ndarray, b: np.ndarray, r: float, color) -> list[tuple]:
        a, b = self._cam(a), self._cam(b)
        t0, t1 = 0.0, 1.0
        for n, d in self.planes:
            fa, fb = n @ a + d, n @ b + d
            if fa < 0 and fb < 0:
                return []
            if fa < 0:
                t0 = max(t0, fa / (fa - fb))
            elif fb < 0:
                t1 = min(t1, fa / (fa - fb))
        if t0 > t1:
            return []
        a, b = a + t0 * (b - a), a + t1 * (b - a)
        px = _distort(np.array([a, b]), self.cam)
        rp = self.cam.intrinsics.fx * r / np.array([a[2], b[2]])
        depth = float(np.linalg.norm((a + b) / 2))
        return [(2, depth, "capsule", (px, rp, np.array(color, float)))]

    def _arm(self, state: CellState) -> list[tuple]:
        frames = self.arm.frames(state.q)
        out = []
        for name, a, b, r in link_capsules(frames):
            kind = "tool" if name == "tool" else "joint" if name in JOINT_LINKS else "tube"
            out += self._capsule(a, b, r, LINK_COLORS[kind])
        tcp = frames[-1]
        top = tcp[:3, 3] - 0.04 * tcp[:3, 2]
        for side in (-1, 1):  # fingers close along tool y
            off = side * (state.gripper_mm / 2000 + 0.006) * tcp[:3, 1]
            out += self._capsule(top + off, tcp[:3, 3] + 0.008 * tcp[:3, 2] + off, 0.007, LINK_COLORS["finger"])
        return out

    # -- drawing -----------------------------------------------------------------------
    def render(self, state: CellState, door_angle_deg: float = 0.0, overlay: str | None = None) -> np.ndarray:
        items = list(self.static)
        for d in self.lab.doors:
            items += self._box(geo.door_box(self.lab, d, door_angle_deg))
        for lw, T in labware_poses(self.lab, state).items():
            items += self._box(geo.labware_box(self.lab, lw, T), self.lw_color[self.lab.labware_item(lw).type])
        items += self._arm(state)
        img = Image.new("RGB", (self.W, self.H), (40, 42, 46))
        dr = ImageDraw.Draw(img)
        for _, _, kind, payload in sorted(items, key=lambda it: (it[0], -it[1])):
            if kind == "polys":
                for pts, col in payload:
                    dr.polygon(pts, fill=col, outline=tuple(int(v * 0.8) for v in col))
            else:
                self._draw_capsule(dr, *payload)
        text = f"{self.lab.id}  {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z" + (f"  {overlay}" if overlay else "")
        x0, y0, x1, y1 = dr.textbbox((10, 8), text, font=self.font)
        dr.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(0, 0, 0))
        dr.text((10, 8), text, font=self.font, fill=(255, 255, 255))
        dr.text((self.W - 52, self.H - 24), "TWIN", font=self.small, fill=(255, 210, 60))
        return np.asarray(img)

    @staticmethod
    def _draw_capsule(dr: ImageDraw.ImageDraw, px: np.ndarray, rp: np.ndarray, col: np.ndarray):
        (ax, ay), (bx, by) = px
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln, dx / ln
        # outline, body, highlight (offset toward the light, upper left)
        for scale, c, off in ((1.0, col * 0.55, 0.0), (0.82, col * 0.9, 0.0), (0.35, np.minimum(col * 1.15 + 25, 255), 0.35)):
            ra, rb = rp * scale
            oxa, oya, oxb, oyb = -off * rp[0], -off * rp[0], -off * rp[1], -off * rp[1]
            fill = tuple(int(v) for v in c)
            dr.polygon([(ax + nx * ra + oxa, ay + ny * ra + oya), (bx + nx * rb + oxb, by + ny * rb + oyb),
                        (bx - nx * rb + oxb, by - ny * rb + oyb), (ax - nx * ra + oxa, ay - ny * ra + oya)], fill=fill)
            dr.ellipse((ax + oxa - ra, ay + oya - ra, ax + oxa + ra, ay + oya + ra), fill=fill)
            dr.ellipse((bx + oxb - rb, by + oyb - rb, bx + oxb + rb, by + oyb + rb), fill=fill)


def render_plan_video(lab: LabSpec, plan: Plan, out_path: str | Path, fps: float = 15, camera_id: str | None = None) -> Path:
    """Render the plan's waypoints (sampled at fps) through the twin camera to an MP4."""
    r = TwinRenderer(lab, camera_id)
    state = plan.start_state.model_copy(deep=True)
    moves = [e for e in plan.events if e.kind in ("attach", "detach")]
    starts = [e for e in plan.events if e.kind == "step_start"]
    ev, label = 0, ""
    with Mp4Writer(out_path, r.W, r.H, fps) as video:
        for k in range(int(plan.duration_s * fps) + 1):
            t = k / fps
            while ev < len(moves) and moves[ev].t <= t + 1e-9:
                e = moves[ev]
                if e.kind == "attach":
                    state.held = e.data["labware"]
                    state.labware_at.pop(e.data["labware"], None)
                else:
                    state.held = None
                    state.labware_at[e.data["labware"]] = e.data["location"]
                ev += 1
            for s in starts:
                if s.t <= t + 1e-9:
                    label = f"step {s.step} {s.data.get('op', '')}"
            wp = plan.waypoints[min(round(t / plan.dt), len(plan.waypoints) - 1)]
            state.q, state.gripper_mm = wp.q, wp.gripper_mm
            video.write(r.render(state, overlay=f"{plan.program_name} | t={t:6.2f}s | {label}"))
    return Path(out_path)
