"""Dry-run planner: IR program -> fixed-rate (DT) joint trajectory + validation. The dry-run is the run."""
from __future__ import annotations

import math

import numpy as np

from .. import geometry as geo
from ..ir import Program
from ..plan import DT, CellState, Plan, PlanEvent, Violation, Waypoint
from ..spec import LabSpec, matrix_to_quat, quat_to_matrix
from .collision import CollisionScene
from .kinematics import ArmModel, in_limits

FATAL = {"state", "unreachable", "unknown_ref", "schema"}
GRIPPER_S = 0.5     # s per gripper move
TCP_ACCEL = 1.2     # m/s², linear-move ramps (URScript movel default; the spec only caps speed)
LIN_STEP = 0.005    # m (and 0.02 rad) between IK samples when sizing a linear move
MAX_JUMP = 0.3      # rad between those samples = IK branch switch / singularity


class _Fatal(Exception):
    pass


def initial_state(lab: LabSpec) -> CellState:
    return CellState(q=list(np.radians(lab.arm.home_q_deg)), gripper_mm=lab.arm.tool.open_mm,
                     labware_at={lw.id: lw.at for lw in lab.labware})


def _profile(D: float, v: float, a: float) -> np.ndarray:
    """Trapezoidal profile over distance D as fractions s(k*DT), k=1..n; stretched to end on the DT grid."""
    if D <= 1e-12:
        return np.zeros(0)
    vp = min(v, math.sqrt(D * a))
    ta, T = vp / a, D / vp + vp / a
    n = max(1, math.ceil(T / DT - 1e-9))
    t = np.arange(1, n + 1) * (T / n)
    s = np.where(t < ta, 0.5 * a * t**2, np.where(t <= T - ta, 0.5 * a * ta**2 + vp * (t - ta), D - 0.5 * a * (T - t) ** 2))
    return np.clip(s / D, 0.0, 1.0)


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)


def _yaw(T: np.ndarray) -> float:
    return math.degrees(math.atan2(T[1, 0], T[0, 0]))


class _Planner:
    def __init__(self, lab: LabSpec, start: CellState):
        self.lab, self.arm, lim = lab, ArmModel(lab), lab.arm.limits
        self.w_max, self.al_max = math.radians(lim.joint_speed_dps), math.radians(lim.joint_accel_dps2)
        self.q, self.grip = np.array(start.q, float), start.gripper_mm
        self.held, self.at = start.held, dict(start.labware_at)
        self.wps = [Waypoint(t=0.0, q=list(self.q), gripper_mm=self.grip)]
        self.events: list[PlanEvent] = []
        self.violations: list[Violation] = []
        self.step: int | None = None
        self.p_prev = self.arm.tcp(self.q)[:3, 3]
        self.stats = {"max_tcp_speed_mps": 0.0, "min_clearance_m": math.inf, "min_clearance_pair": None, "path_length_m": 0.0}
        self._rescene()
        self._check(self.q, frozenset(), 0.0)

    # -- bookkeeping ---------------------------------------------------------
    @property
    def t(self) -> float:
        return round((len(self.wps) - 1) * DT, 6)

    def state(self) -> CellState:
        return CellState(q=list(self.q), gripper_mm=self.grip, held=self.held, labware_at=dict(self.at))

    def _rescene(self):
        self.scene = CollisionScene(self.lab, self.state(), 0.0, self.arm)  # door closed during runs
        self._last = (None, None)  # (q, ignore) -> check() cache for holds and gripper moves

    def violate(self, code: str, msg: str):
        if not any(v.step == self.step and v.code == code for v in self.violations):  # first per step+code
            self.violations.append(Violation(step=self.step, code=code, message=msg, t=self.t))
        if code in FATAL:
            raise _Fatal

    def event(self, kind: str, **data):
        self.events.append(PlanEvent(t=self.t, step=self.step, kind=kind, data=data))

    def _check(self, q, ignore, v_tcp):
        if not in_limits(q):
            self.violate("joint_limit", f"q={np.round(np.degrees(q), 1).tolist()} deg is outside the UR5e joint limits")
        key = (q.tobytes(), ignore)
        if key != self._last[0]:
            self._last = (key, self.scene.check(q, ignore))
        d, pair, hit = self._last[1]
        if d < self.stats["min_clearance_m"]:
            self.stats["min_clearance_m"], self.stats["min_clearance_pair"] = round(d, 4), list(pair)
        if d < self.lab.safety.min_clearance_m:
            self.violate("collision", f"{pair[0]} vs {pair[1]}: clearance {d * 1000:.1f} mm < {self.lab.safety.min_clearance_m * 1000:.0f} mm")
        if hit:
            self.violate("keepout", f"{hit[0]} enters keep-out {hit[1]}")
        if v_tcp > self.lab.arm.limits.tcp_speed_mps * 1.001:
            self.violate("speed", f"TCP speed {v_tcp:.3f} m/s > {self.lab.arm.limits.tcp_speed_mps} m/s")

    def emit(self, qs, grips=None, ignore=frozenset()):
        """Append waypoints (one per DT) and validate each."""
        for i, q in enumerate(qs):
            g = self.grip if grips is None else float(grips[i])
            p = self.arm.tcp(q)[:3, 3]
            step = float(np.linalg.norm(p - self.p_prev))
            v = step / DT
            if np.abs(q - self.q).max() / DT > self.w_max * 1.001:
                self.violate("speed", f"joint speed {np.abs(q - self.q).max() / DT:.3f} rad/s > {self.w_max:.3f} rad/s")
            self.stats["max_tcp_speed_mps"] = max(self.stats["max_tcp_speed_mps"], v)
            self.stats["path_length_m"] += step
            self.q, self.grip, self.p_prev = np.asarray(q, float), g, p
            self.wps.append(Waypoint(t=round(len(self.wps) * DT, 6), q=list(self.q), gripper_mm=g))
            self._check(self.q, ignore, v)

    def hold(self, seconds: float, ignore=frozenset()):
        self.emit([self.q] * round(seconds / DT), ignore=ignore)

    # -- motions ---------------------------------------------------------------
    def _tcp_speeds(self, qs) -> float:
        ps = np.array([self.arm.tcp(q)[:3, 3] for q in [self.q, *qs]])
        return float(np.linalg.norm(np.diff(ps, axis=0), axis=1).max() / DT)

    def joint_move(self, q1, speed=1.0, ignore=frozenset()):
        """Synchronised trapezoid (joint speed/accel x speed), slowed so the TCP stays under its cap."""
        q0, dq = self.q, np.asarray(q1, float) - self.q
        D = float(np.abs(dq).max())
        v, a = self.w_max * speed, self.al_max * speed
        qs = q0 + _profile(D, v, a)[:, None] * dq
        if len(qs):
            f = self._tcp_speeds(qs) / (0.98 * self.lab.arm.limits.tcp_speed_mps * speed)  # 2% margin: resampling shifts the peak
            if f > 1:  # stretching time by f scales v by 1/f and a by 1/f²
                qs = q0 + _profile(D, v / f, a / f / f)[:, None] * dq
        self.emit(qs, ignore=ignore)

    def linear_move(self, T1, speed=1.0, ignore=frozenset()):
        """Straight-line TCP path, IK per DT sample seeded by the previous one."""
        T0 = self.arm.tcp(self.q)
        p0, p1 = T0[:3, 3], T1[:3, 3]
        r0, r1 = np.array(matrix_to_quat(T0[:3, :3])), np.array(matrix_to_quat(T1[:3, :3]))
        L, ang = float(np.linalg.norm(p1 - p0)), 2 * math.acos(min(1.0, abs(float(r0 @ r1))))

        def pose(u):
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = quat_to_matrix(_slerp(r0, r1, u)), p0 + u * (p1 - p0)
            return T

        def solve(us):
            out, q = [], self.q
            for u in us:
                qn = self.arm.ik(pose(u), q)
                if qn is None:
                    self.violate("unreachable", f"no IK solution along the linear path at {np.round(pose(u)[:3, 3], 3).tolist()}")
                if np.abs(qn - q).max() > MAX_JUMP:
                    self.violate("unreachable", f"linear path crosses a singularity near {np.round(pose(u)[:3, 3], 3).tolist()}")
                out.append(qn)
                q = qn
            return np.array(out)

        if L < 1e-6 and ang < 1e-6:
            return
        n = max(2, math.ceil(L / LIN_STEP), math.ceil(ang / 0.02))
        dqdu = np.abs(np.diff(np.r_[self.q[None], solve(np.linspace(0, 1, n + 1)[1:])], axis=0)).max() * n
        cap = self.lab.arm.limits.tcp_speed_mps * speed
        v = min(cap / L if L > 1e-9 else math.inf, self.w_max * speed / dqdu)
        a = min(TCP_ACCEL * speed / L if L > 1e-9 else math.inf, self.al_max * speed / dqdu)
        for _ in range(4):  # the sizing pass is an estimate: re-time until the sampled joint speeds fit
            qs = solve(_profile(1.0, v, a))
            f = np.abs(np.diff(np.r_[self.q[None], qs], axis=0)).max() / DT / (self.w_max * speed)
            if f <= 1:
                break
            v, a = v / f / 1.02, a / (f * 1.02) ** 2
        self.emit(qs, ignore=ignore)

    def move(self, T, linear: bool, speed=1.0, ignore=frozenset()):
        q = self.arm.ik(T, self.q)
        if q is None:
            self.violate("unreachable", f"no IK solution for TCP target {np.round(T[:3, 3], 3).tolist()}")
        if linear:
            self.linear_move(T, speed, ignore)
        else:
            self.joint_move(q, speed, ignore)

    def gripper_to(self, width, ignore=frozenset()):
        n = round(GRIPPER_S / DT)
        self.emit([self.q] * n, grips=np.linspace(self.grip, width, n + 1)[1:], ignore=ignore)

    # -- lookups ---------------------------------------------------------------
    def ref(self, fn, *args):
        try:
            return fn(*args)
        except (KeyError, ValueError) as e:
            self.violate("unknown_ref", str(e).strip("'\""))

    def support(self, loc_id: str) -> frozenset:
        """Boxes a labware at `loc_id` rests on: the nest/instrument and the bench under it."""
        p = geo.location_frame(self.lab, loc_id)[:3, 3]
        ids = {loc_id}
        for b in self.lab.benches:
            local = np.linalg.inv(b.pose.matrix()) @ np.r_[p, 1.0]
            if abs(local[0]) <= b.size[0] / 2 and abs(local[1]) <= b.size[1] / 2:
                ids.add(b.id)
        return frozenset(ids)

    # -- ops -------------------------------------------------------------------
    def op_home(self, st):
        self.joint_move(self.arm.home_q, st.speed)

    def op_move_joints(self, st):
        q = np.radians(st.q_deg)
        if not in_limits(q):
            self.violate("joint_limit", f"target {st.q_deg} deg is outside the UR5e joint limits")
        self.joint_move(q, st.speed)

    def op_move_to(self, st):
        tg = st.target
        if sum(x is not None for x in (tg.xyz, tg.labware, tg.location)) != 1 or (tg.well and not tg.labware):
            self.violate("schema", "target needs exactly one of xyz | labware | location (well needs labware)")
        if tg.xyz is not None:
            p, yaw = np.array(tg.xyz, float), 0.0
        elif tg.labware is not None:
            self.ref(self.lab.labware_item, tg.labware)
            if tg.labware not in self.at:
                self.violate("state", f"{tg.labware} is in the gripper, not at a location")
            T_lw = geo.location_frame(self.lab, self.at[tg.labware])
            top = T_lw @ [0, 0, self.lab.labware_type(tg.labware).size[2], 1]
            p, yaw = (self.ref(geo.well_point, self.lab, tg.labware, tg.well, T_lw) if tg.well else top[:3]), _yaw(T_lw)
        else:
            T_loc = geo.location_frame(self.lab, self.ref(self.lab.location, tg.location).id)
            p, yaw = T_loc[:3, 3], _yaw(T_loc)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = geo.tool_down(yaw if tg.yaw_deg is None else tg.yaw_deg), p + [0, 0, tg.z_above]
        self.move(T, st.linear, st.speed)

    def op_gripper(self, st):
        if st.width_mm > self.lab.arm.tool.open_mm:
            self.violate("joint_limit", f"gripper width {st.width_mm} mm > max opening {self.lab.arm.tool.open_mm} mm")
        if self.held and st.width_mm > self.lab.labware_type(self.held).grip.width_mm:
            self.violate("state", f"opening the gripper would drop {self.held} (use place)")
        self.gripper_to(min(st.width_mm, self.lab.arm.tool.open_mm))

    def _touch(self, lw: str, loc: str, approach_m: float, width_mm: float, attach: bool):
        """Above loc -> straight down to the grasp -> gripper to width -> (de)attach -> straight up."""
        G = geo.grasp_tcp(self.lab, lw, geo.location_frame(self.lab, loc))
        above = G.copy()
        above[2, 3] += approach_m
        ign = self.support(loc) | {lw}
        self.move(above, linear=False)
        self.linear_move(G, ignore=ign)
        self.gripper_to(width_mm, ign)
        if attach:
            self.held = lw
            del self.at[lw]
        else:
            self.held, self.at[lw] = None, loc
        self._rescene()
        self.event("attach" if attach else "detach", labware=lw, location=loc)
        self.linear_move(above, ignore=ign)

    def op_pick(self, st):
        lw = st.labware
        grip = self.ref(self.lab.labware_type, lw).grip
        if self.held:
            self.violate("state", f"pick {lw}: already holding {self.held}")
        if lw not in self.at:
            self.violate("state", f"pick {lw}: it is not at a location")
        if self.grip <= grip.width_mm:
            self.violate("state", f"pick {lw}: gripper is not open ({self.grip:.0f} mm)")
        self._touch(lw, self.at[lw], st.approach_m, grip.width_mm, attach=True)

    def op_place(self, st):
        lw, dest = st.labware, st.at
        self.ref(self.lab.labware_item, lw)
        self.ref(self.lab.location, dest)
        if self.held != lw:
            self.violate("state", f"place {lw}: the gripper holds {self.held or 'nothing'}")
        other = next((k for k, v in self.at.items() if v == dest), None)
        if other:
            self.violate("state", f"place {lw}: {dest} is occupied by {other}")
        self._touch(lw, dest, st.approach_m, self.lab.arm.tool.open_mm, attach=False)

    def op_wait(self, st):
        self.event("wait", seconds=st.seconds)
        self.hold(st.seconds)

    def op_capture(self, st):
        cam = self.ref(self.lab.camera, st.camera)
        self.event("capture", camera=cam.id, tag=st.tag)

    def op_instrument(self, st):
        loc = self.ref(self.lab.location, st.instrument)
        if not loc.driver:
            self.violate("unknown_ref", f"{loc.id} has no instrument driver")
        secs = st.params.get("seconds", 0)
        if isinstance(secs, bool) or not isinstance(secs, (int, float)) or not 0 <= secs <= 3600:
            self.violate("schema", f"instrument params.seconds must be 0..3600, got {secs!r}")
        self.event("instrument", instrument=loc.id, action=st.action, params=st.params)
        self.hold(float(secs))

    def op_mark(self, st):
        self.event("mark", text=st.text)


def plan_program(lab: LabSpec, program: Program, start: CellState | None = None) -> Plan:
    P = _Planner(lab, start or initial_state(lab))
    start_state = P.state()
    if program.cell not in (None, lab.id):
        P.violations.append(Violation(step=None, code="unknown_ref", message=f"program is for cell {program.cell}, not {lab.id}"))
    try:
        for i, st in enumerate(program.steps):
            P.step = i
            P.event("step_start", op=st.op)
            getattr(P, f"op_{st.op}")(st)
    except _Fatal:
        pass
    stats = {**P.stats, "n_waypoints": len(P.wps), "path_length_m": round(P.stats["path_length_m"], 4),
             "max_tcp_speed_mps": round(P.stats["max_tcp_speed_mps"], 4)}
    return Plan(ok=not P.violations, program_name=program.name, cell=lab.id, lab_sha=lab.sha256(), ir_sha=program.sha256(),
                dt=DT, duration_s=P.wps[-1].t, waypoints=P.wps, events=P.events, violations=P.violations,
                start_state=start_state, final_state=P.state(), stats=stats)
