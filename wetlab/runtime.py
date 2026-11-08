"""Cell runtime: one 50 Hz loop per cell. Interlock -> executor -> ArmDriver.servo, plus recorder + state bus.

Every tick: read door / arm / camera heartbeat, evaluate the interlock (trip = arm.stop + abort
the running job + latch), stream one plan waypoint, fire plan events at their time, publish a
state snapshot. Tests call tick() by hand with a fake clock; the server runs start() (a thread,
`speed` x real time).

After an abort the cell stays where the arm stopped (a held plate stays held) and the next plan
starts from there. shortcut: there is no `recover` op yet; an operator re-homes / re-seats
labware by hand and restarts the service (state resets to the spec). Next: an operator-only
recover job planned from the aborted state.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .ir import Program
from .media import Mp4Writer, encode_jpeg
from .plan import DT, CellState, Plan
from .safety import Interlock
from .spec import LabSpec
from .twin.planner import initial_state, plan_program

log = logging.getLogger("wetlab.runtime")

SLOT_GRACE_S = 30.0    # a run may overrun its slot this long before slot_ended trips
FIT_MARGIN_S = 10.0    # a job starts only if planned duration + this fits before the slot ends
REC_EVERY = 5          # ticks per recorded frame (50 Hz / 5 = 10 fps)


def plan_summary(plan: Plan) -> dict:
    s = json.loads(plan.model_dump_json(include={"ok", "violations", "duration_s", "stats", "lab_sha", "ir_sha"}))
    return {**s, "n_waypoints": len(plan.waypoints), "n_events": len(plan.events)}


def _vkey(violations) -> list:
    return sorted((v["step"] if isinstance(v, dict) else v.step, v["code"] if isinstance(v, dict) else v.code) for v in violations)


@dataclass
class _Run:
    job: dict
    user: dict
    booking: dict
    plan: Plan
    dir: Path
    rec: Mp4Writer
    joints: object
    events: object
    i: int = 0
    ev_i: int = 0
    step: int | None = None
    max_dev: float = 0.0
    n_capture: int = 0
    instrument_results: list = field(default_factory=list)


class CellRuntime:
    def __init__(self, lab: LabSpec, arm, door, camera, instruments: dict, db, billing, artifacts_root, clock=time.time, speed: float = 1.0):
        self.lab, self.cell = lab, lab.id
        self.arm, self.door, self.camera, self.instruments = arm, door, camera, instruments
        self.db, self.billing, self.artifacts_root = db, billing, Path(artifacts_root)
        self.clock, self.speed = clock, speed
        self.interlock = Interlock(lab.safety)
        st = initial_state(lab)
        self.held, self.labware_at = st.held, dict(st.labware_at)
        self.estop_pressed = False
        self.run: _Run | None = None
        self.viewers = 0                     # live stream clients: grab at 10 Hz instead of 1 Hz
        self.lock = threading.RLock()
        self.frame, self.frame_t, self.frame_n = None, clock(), 0
        self._jpeg: tuple[bytes | None, int] = (None, -1)
        self._booking_id: str | None = None
        self.n = 0
        self.snapshot: dict = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._publish(clock(), door.state())

    # -- state ---------------------------------------------------------------------
    def cell_state(self) -> CellState:
        with self.lock:
            a = self.arm.read_state()
            return CellState(q=a.q, gripper_mm=a.gripper_mm, held=self.held, labware_at=dict(self.labware_at))

    def dry_run(self, program: Program) -> Plan:
        return plan_program(self.lab, program, start=self.cell_state())

    def job_owner(self) -> str | None:
        return self.run.job["user_id"] if self.run else None

    def latest_jpeg(self) -> bytes | None:
        with self.lock:
            if self.frame is None:
                self._grab()
            if self.frame is None:
                return None
            if self._jpeg[1] != self.frame_n:
                self._jpeg = (encode_jpeg(self.frame), self.frame_n)
            return self._jpeg[0]

    # -- loop ----------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"runtime-{self.cell}")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        period, nxt = DT / self.speed, time.monotonic()
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                log.exception("runtime tick failed")
                with self.lock:
                    self._abort("runtime_error", self.clock(), status="FAILED")
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            elif delay < -1.0:
                nxt = time.monotonic()   # fell far behind (e.g. a slow render): don't burst

    def tick(self) -> None:
        with self.lock:
            now = self.clock()
            if self.run is None and self.n % (REC_EVERY if self.viewers else 50) == 0:
                self._grab()             # camera heartbeat while idle
            door, arm = self.door.state(), self.arm.read_state()
            reason = self.interlock.evaluate(door, arm, now - self.frame_t, self.estop_pressed, active=self.run is not None)
            if self.run and not reason and now > self.run.booking["end_ts"] + SLOT_GRACE_S:
                reason = "slot_ended"
            if self.run and (reason or self.interlock.tripped):
                self._abort(reason or self.interlock.reason, now)
            elif reason:                  # tripped while idle (E-stop)
                self._abort(reason, now)
            elif self.run:
                self._step(now, door)
            else:
                self._maybe_start(now, door, arm)
            self.n += 1
            self._publish(now, self.door.state())

    # -- operator / user actions (API threads) --------------------------------------
    def door_action(self, action: str, actor: str) -> None:
        with self.lock:
            now = self.clock()
            if action == "unlock" and self.run:
                self._abort("entry_requested", now)      # stop first, then release the lock
            if action in ("open", "close") and not hasattr(self.door, action):
                raise ValueError("this door is opened and closed by a person, not the service")
            getattr(self.door, action)()
            self.db.audit(actor, f"door_{action}", ts=now, cell=self.cell)
            self._publish(now, self.door.state())

    def estop(self, actor: str) -> None:
        with self.lock:
            now = self.clock()
            self.estop_pressed = True
            self._abort("estop", now)
            self.db.audit(actor, "estop", ts=now, cell=self.cell)
            self._publish(now, self.door.state())

    def reset(self, actor: str) -> list[str]:
        """Operator reset: clears the latch only when door/E-stop/camera/arm are all clear. -> remaining faults."""
        with self.lock:
            now = self.clock()
            self._grab()
            door = self.door.state()
            pre = self.interlock.faults(door, None, now - self.frame_t, False)
            if pre:
                return pre
            self.arm.reset()
            remaining = self.interlock.reset(door, self.arm.read_state(), now - self.frame_t, False)
            if not remaining:
                self.estop_pressed = False
                self.db.audit(actor, "interlock_reset", ts=now, cell=self.cell)
            self._publish(now, door)
            return remaining

    def cancel(self, job_id: str, actor: str) -> bool:
        """Stop streaming a running job (the arm holds its last setpoint; no latch). -> True if it was running."""
        with self.lock:
            if not self.run or self.run.job["id"] != job_id:
                return False
            now = self.clock()
            self._finish("CANCELLED", f"cancelled by {actor}", now)
            self._publish(now, self.door.state())
            return True

    # -- internals -----------------------------------------------------------------
    def _grab(self):
        f = self.camera.frame()
        if f is not None:   # stamp after the grab: a slow tick (re-plan, render) must not look like a lost camera
            self.frame, self.frame_t, self.frame_n = f, self.clock(), self.frame_n + 1
        return f

    def _abort(self, reason: str, now: float, status: str = "ABORTED") -> None:
        self.interlock.trip(reason)   # no-op if already latched (tick() latches via evaluate)
        self.arm.stop(reason)
        self.db.audit("runtime", "interlock_trip", ts=now, cell=self.cell, reason=reason, latched=self.interlock.reason,
                      job=self.run.job["id"] if self.run else None)
        if self.run:
            self._finish(status, reason, now)

    def _maybe_start(self, now: float, door, arm) -> None:
        if self.n % 50 == 0:
            self.db.exec("UPDATE jobs SET status='CANCELLED', error='booking ended before the job could run', finished_ts=? "
                         "WHERE cell=? AND status='APPROVED' AND booking_id IN "
                         "(SELECT id FROM bookings WHERE end_ts<=? OR status='cancelled')", now, self.cell, now)
        if self.interlock.tripped:
            return
        rows = self.db.all(
            "SELECT j.* FROM jobs j JOIN bookings b ON b.id=j.booking_id WHERE j.cell=? AND j.status='APPROVED' "
            "AND b.status='confirmed' AND b.start_ts<=? AND b.end_ts>? AND (j.isaac_status IS NULL OR j.isaac_status='ok') "
            "ORDER BY j.created_ts, j.rowid", self.cell, now, now)
        booking = job = None
        for j in rows:
            b = self.db.one("SELECT * FROM bookings WHERE id=?", j["booking_id"])
            if now + json.loads(j["plan_summary_json"])["duration_s"] + FIT_MARGIN_S <= b["end_ts"]:
                job, booking = j, b
                break
        if job is None:
            return
        if door.closed and not door.locked:
            self.door.lock()
            door = self.door.state()
        self.interlock.evaluate(door, arm, now - self.frame_t, self.estop_pressed, active=False)
        if not self.interlock.can_start():
            return   # stays APPROVED until the door is closed+locked etc.
        plan = self.dry_run(Program.model_validate_json(job["ir_json"]))
        approved = json.loads(job["plan_summary_json"])
        if not plan.ok or plan.ok != approved["ok"] or _vkey(plan.violations) != _vkey(approved["violations"]):
            err = "cell state changed since the approved dry-run: " + "; ".join(v.message for v in plan.violations)[:500]
            self.db.update("jobs", job["id"], status="FAILED", error=err, finished_ts=now)
            self.db.audit("runtime", "job_replan_refused", ts=now, job=job["id"])
            return
        d = Path(job["artifacts_dir"])
        d.mkdir(parents=True, exist_ok=True)
        cam = self.camera.info
        rec = Mp4Writer(str(d / "recording.mp4"), cam.width, cam.height, 50 / REC_EVERY)
        rec.__enter__()
        joints = open(d / "joints.csv", "w")
        joints.write("t," + ",".join(f"q_cmd{i}" for i in range(6)) + "," + ",".join(f"q_meas{i}" for i in range(6)) + ",gripper_mm,door_closed\n")
        self.run = _Run(job, self.db.user(job["user_id"]), booking, plan, d, rec, joints, open(d / "events.jsonl", "w"))
        (d / "run_plan.json").write_text(plan.model_dump_json())
        self.db.update("jobs", job["id"], status="RUNNING", started_ts=now)
        self.db.audit("runtime", "job_start", ts=now, job=job["id"], cell=self.cell)
        self._event(self.run, {"t": 0.0, "kind": "run_start", "wall_ts": now})
        self._grab()   # the re-plan above can take a second; refresh the heartbeat before the first servo tick

    def _step(self, now: float, door) -> None:
        r = self.run
        wp = r.plan.waypoints[r.i]
        self.arm.servo(wp.q, wp.gripper_mm, DT)
        q_meas = self.arm.read_state().q
        r.max_dev = max(r.max_dev, max(abs(m - c) for m, c in zip(q_meas, wp.q)))
        r.joints.write(",".join(f"{v:.6f}" for v in [wp.t, *wp.q, *q_meas]) + f",{wp.gripper_mm:.2f},{int(door.closed)}\n")
        evs = r.plan.events
        while r.ev_i < len(evs) and evs[r.ev_i].t <= wp.t + 1e-9:
            self._fire(evs[r.ev_i], now)
            r.ev_i += 1
        if r.i % REC_EVERY == 0:
            f = self._grab()
            if f is not None:
                r.rec.write(f)
        r.i += 1
        if r.i >= len(r.plan.waypoints):
            self._finish("SUCCEEDED", None, now)

    def _fire(self, ev, now: float) -> None:
        r, d = self.run, ev.data
        rec = {"t": ev.t, "step": ev.step, "kind": ev.kind, **d}
        if ev.kind == "step_start":
            r.step = ev.step
        elif ev.kind == "attach":
            self.held = d["labware"]
            self.labware_at.pop(d["labware"], None)
        elif ev.kind == "detach":
            self.held = None
            self.labware_at[d["labware"]] = d["location"]
        elif ev.kind == "capture":
            f = self._grab()
            if f is not None:
                r.n_capture += 1
                rec["file"] = f"capture_{r.n_capture:03d}_{d.get('tag', 'frame')}.jpg"
                (r.dir / rec["file"]).write_bytes(encode_jpeg(f, quality=90))
        elif ev.kind == "instrument":
            inst = self.instruments.get(d["instrument"])
            # shortcut: called on the tick; real instruments must return at once (the plan already holds for the duration)
            rec["result"] = inst.run(d["action"], d.get("params", {})) if inst else {"ok": False, "error": "no driver"}
            r.instrument_results.append(rec["result"])
        self._event(r, rec)

    def _finish(self, status: str, reason: str | None, now: float) -> None:
        r, self.run = self.run, None
        try:
            self._event(r, {"t": round(r.i * DT, 3), "kind": "end", "status": status, "reason": reason})
            r.joints.close()
            r.events.close()
            r.rec.__exit__(None, None, None)
        finally:
            executed = round(r.i * DT, 3)
            report = {"job_id": r.job["id"], "cell": self.cell, "status": status, "reason": reason,
                      "duration_s": executed, "planned_duration_s": r.plan.duration_s,
                      "max_deviation_rad": r.max_dev, "lab_sha": r.plan.lab_sha, "ir_sha": r.plan.ir_sha,
                      "started_ts": r.job.get("started_ts") or now - executed, "finished_ts": now,
                      "captures": r.n_capture, "instrument_results": r.instrument_results,
                      "final_state": self.cell_state().model_dump()}
            (r.dir / "report.json").write_text(json.dumps(report, indent=2))
            zip_artifacts(r.dir)
            self.db.update("jobs", r.job["id"], status=status, error=reason, finished_ts=now, max_deviation_rad=r.max_dev)
            self.billing.meter("arm_seconds", executed, r.user, self.lab.billing, booking_id=r.job["booking_id"], job_id=r.job["id"], ts=now)
            self.db.audit("runtime", "job_end", ts=now, job=r.job["id"], status=status, reason=reason)

    @staticmethod
    def _event(r: _Run, rec: dict) -> None:
        r.events.write(json.dumps(rec, default=str) + "\n")

    def _publish(self, now: float, door) -> None:
        if self.n % 25 == 0:
            b = self.db.active_booking(self.cell, now)
            self._booking_id = b["id"] if b else None
        a, r = self.arm.read_state(), self.run
        self.snapshot = {
            "t": now, "cell": self.cell, "q": a.q, "gripper_mm": a.gripper_mm, "held": self.held,
            "labware_at": dict(self.labware_at), "arm_safety": a.safety,
            "door": {"closed": door.closed, "locked": door.locked, "angle_deg": round(door.angle_deg, 2)},
            "interlock": {"tripped": self.interlock.tripped, "reason": self.interlock.reason,
                          "can_start": self.interlock.can_start(), "faults": list(self.interlock.current)},
            "job": {"id": r.job["id"], "status": "RUNNING", "step": r.step,
                    "progress": round(r.i / len(r.plan.waypoints), 4)} if r else None,
            "booking_active": self._booking_id,
        }


def zip_artifacts(d: Path) -> Path:
    out, tmp = d / "results.zip", d / ".results.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.iterdir()):
            if p.is_file() and not p.name.startswith(".") and p.name != "results.zip":
                z.write(p, p.name)
    tmp.replace(out)
    return out
