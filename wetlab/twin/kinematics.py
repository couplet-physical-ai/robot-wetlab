"""UR5e forward/inverse kinematics in the room frame (official UR DH parameters, controller 'base' frame)."""
from __future__ import annotations

import math

import numpy as np

from ..spec import LabSpec

# Universal Robots "DH parameters for calculations of kinematics and dynamics", UR5e (standard DH).
D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
ALPHA = np.array([math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0])

# Joint limits. UR5e tech sheet: ±360° working range on every joint. The elbow is limited to ±180°:
# UR's own ROS description (UniversalRobots/Universal_Robots_ROS2_Description config/ur5e/joint_limits.yaml)
# documents that the elbow physically cannot turn past ~±pi because the shoulder-lift link is in the way.
Q_MAX = np.radians([360.0, 360.0, 180.0, 360.0, 360.0, 360.0])
Q_MIN = -Q_MAX
TWO_PI = 2 * math.pi


def _dh(theta: float, i: int) -> np.ndarray:
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(ALPHA[i]), math.sin(ALPHA[i])
    return np.array([[ct, -st * ca, st * sa, A[i] * ct],
                     [st, ct * ca, -ct * sa, A[i] * st],
                     [0.0, sa, ca, D[i]],
                     [0.0, 0.0, 0.0, 1.0]])


def fk_base(q) -> list[np.ndarray]:
    """Frames 0..6 (base .. flange) in the UR base frame; frame i-1's z is joint i's axis."""
    out = [np.eye(4)]
    for i in range(6):
        out.append(out[-1] @ _dh(q[i], i))
    return out


def ik_base(T06: np.ndarray) -> list[np.ndarray]:
    """All analytic solutions (up to 8, angles in (-pi, pi]) for a flange pose in the base frame."""
    d4, d6, a2, a3 = D[3], D[5], A[1], A[2]
    p05 = T06[:3, 3] - d6 * T06[:3, 2]
    T60 = np.linalg.inv(T06)
    r = math.hypot(p05[0], p05[1])
    if r < d4:
        return []
    sols = []
    for s1 in (1, -1):
        q1 = math.atan2(p05[1], p05[0]) + s1 * math.acos(d4 / r) + math.pi / 2
        c1, si1 = math.cos(q1), math.sin(q1)
        c5 = (T06[0, 3] * si1 - T06[1, 3] * c1 - d4) / d6
        if abs(c5) > 1 + 1e-9:
            continue
        for s5 in (1, -1):
            q5 = s5 * math.acos(max(-1.0, min(1.0, c5)))
            sn5 = math.sin(q5)
            if abs(sn5) < 1e-9:
                q6 = 0.0  # wrist singularity: q4 and q6 are coupled, any q6 works
            else:
                q6 = math.atan2((-T60[1, 0] * si1 + T60[1, 1] * c1) / sn5, (T60[0, 0] * si1 - T60[0, 1] * c1) / sn5)
            T14 = np.linalg.inv(_dh(q1, 0)) @ T06 @ np.linalg.inv(_dh(q5, 4) @ _dh(q6, 5))
            p13 = (T14 @ np.array([0.0, -d4, 0.0, 1.0]))[:3]
            n2 = p13 @ p13
            c3 = (n2 - a2 * a2 - a3 * a3) / (2 * a2 * a3)
            if abs(c3) > 1 + 1e-9:
                continue
            for s3 in (1, -1):
                q3 = s3 * math.acos(max(-1.0, min(1.0, c3)))
                q2 = -math.atan2(p13[1], -p13[0]) + math.asin(a3 * math.sin(q3) / math.sqrt(n2))
                T34 = np.linalg.inv(_dh(q2, 1) @ _dh(q3, 2)) @ T14
                q4 = math.atan2(T34[1, 0], T34[0, 0])
                sols.append(np.array([q1, q2, q3, q4, q5, q6]))
    return [(s + math.pi) % TWO_PI - math.pi for s in sols]


class ArmModel:
    """UR5e of a cell: room-frame FK/IK including the tool's TCP offset."""

    def __init__(self, lab: LabSpec):
        arm = lab.arm
        self.T_room_base = arm.base.matrix()
        self.T_base_room = np.linalg.inv(self.T_room_base)
        self.T_flange_tcp = np.eye(4)
        self.T_flange_tcp[2, 3] = arm.tool.tcp_offset_m
        self.T_tcp_flange = np.linalg.inv(self.T_flange_tcp)
        self.home_q = np.radians(arm.home_q_deg)

    def frames(self, q) -> list[np.ndarray]:
        """Room-frame frames: base (joint 1), joints 2..6, flange, tcp (8 frames)."""
        fs = [self.T_room_base @ F for F in fk_base(q)]
        return fs + [fs[-1] @ self.T_flange_tcp]

    def tcp(self, q) -> np.ndarray:
        return self.frames(q)[-1]

    def ik(self, T_tcp_room: np.ndarray, q_seed) -> np.ndarray | None:
        """Analytic solution closest to q_seed (each joint wrapped to its 2pi-equivalent nearest the seed, within limits)."""
        seed = np.asarray(q_seed, float)
        best, best_d = None, math.inf
        for s in ik_base(self.T_base_room @ T_tcp_room @ self.T_tcp_flange):
            q = s + TWO_PI * np.round((seed - s) / TWO_PI)
            q = np.where(q > Q_MAX, q - TWO_PI, np.where(q < Q_MIN, q + TWO_PI, q))
            if np.any(q > Q_MAX + 1e-9) or np.any(q < Q_MIN - 1e-9):
                continue
            d = float(np.linalg.norm(q - seed))
            if d < best_d:
                best, best_d = q, d
        return best


def in_limits(q) -> bool:
    q = np.asarray(q)
    return bool(np.all(q <= Q_MAX + 1e-9) and np.all(q >= Q_MIN - 1e-9))
