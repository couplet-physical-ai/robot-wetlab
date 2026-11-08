import json
from pathlib import Path

import pytest

from wetlab.billing import Billing, LedgerSink
from wetlab.ir import Program
from wetlab.plan import DT
from wetlab.runtime import plan_summary
from wetlab.service.app import build_cell
from wetlab.service.db import DB, new_id
from wetlab.spec import DEFAULT_LAB

LAB = DEFAULT_LAB
DEMO = json.loads((Path(__file__).resolve().parent.parent / "examples" / "demo_protocol.json").read_text())
SMALL = {"name": "small", "steps": [{"op": "home", "speed": 0.5}, {"op": "move_joints", "q_deg": [15, -100, 100, -90, -90, 0], "speed": 0.5},
                                    {"op": "wait", "seconds": 3}, {"op": "capture", "tag": "x"}, {"op": "home", "speed": 0.5}]}
T0 = 1_800_000_000 + 60


class Clock:
    def __init__(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t


class Cell:
    def __init__(self, tmp: Path):
        self.tmp, self.clock = tmp, Clock(T0)
        self.db = DB(tmp / "w.db")
        self.rt = build_cell({"lab_path": str(LAB), "driver": "sim"}, self.db, Billing([LedgerSink(self.db)]), tmp / "art", self.clock, 1.0)
        self.user, _ = self.db.add_user("acme", "alice")
        self.booking = new_id("bk")
        self.db.insert("bookings", id=self.booking, cell="cell-a", user_id=self.user["id"], start_ts=T0 - 60,
                       end_ts=T0 - 60 + 3600, status="confirmed", created_ts=T0)

    def approve(self, program: dict) -> str:
        """What POST /v1/jobs does for mode=run once the dry-run is ok."""
        prog = Program.model_validate(program)
        plan = self.rt.dry_run(prog)
        assert plan.ok, plan.violations
        jid = new_id("job")
        d = self.tmp / "art" / jid
        d.mkdir(parents=True)
        self.db.insert("jobs", id=jid, cell="cell-a", booking_id=self.booking, user_id=self.user["id"], kind="ir", mode="run",
                       source="", ir_json=prog.model_dump_json(), status="APPROVED", plan_summary_json=json.dumps(plan_summary(plan)),
                       created_ts=self.clock(), artifacts_dir=str(d))
        return jid

    def job(self, jid: str) -> dict:
        return self.db.one("SELECT * FROM jobs WHERE id=?", jid)

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.clock.t += DT
            self.rt.tick()

    def until(self, pred, limit: int = 20000) -> None:
        for _ in range(limit):
            if pred():
                return
            self.tick()
        raise AssertionError("condition not reached")


@pytest.fixture
def cell(tmp_path):
    return Cell(tmp_path)


def test_runtime_locks_door_runs_and_records(cell):
    rt = cell.rt
    assert not rt.door.state().locked
    jid = cell.approve(SMALL)
    cell.tick()
    assert cell.job(jid)["status"] == "RUNNING" and rt.door.state().locked
    cell.until(lambda: cell.job(jid)["status"] != "RUNNING")
    j = cell.job(jid)
    assert j["status"] == "SUCCEEDED" and 0 < j["max_deviation_rad"] < 0.01
    d = Path(j["artifacts_dir"])
    rows = (d / "joints.csv").read_text().splitlines()
    assert rows[0].startswith("t,q_cmd0") and len(rows) - 1 == json.loads(j["plan_summary_json"])["n_waypoints"]
    assert (d / "recording.mp4").read_bytes()[4:8] == b"ftyp" and list(d.glob("capture_*_x.jpg"))
    kinds = [json.loads(ln)["kind"] for ln in (d / "events.jsonl").read_text().splitlines()]
    assert kinds[0] == "run_start" and kinds[-1] == "end" and "capture" in kinds
    assert json.loads((d / "report.json").read_text())["status"] == "SUCCEEDED" and (d / "results.zip").is_file()
    m = cell.db.one("SELECT * FROM meter_events WHERE kind='arm_seconds' AND job_id=?", jid)
    assert m["qty"] == pytest.approx(rt.snapshot["t"] - T0, abs=0.1)
    assert rt.interlock.can_start() and rt.snapshot["job"] is None


def test_abort_keeps_held_labware_and_stale_plan_is_refused(cell):
    rt = cell.rt
    first, second = cell.approve(DEMO), cell.approve(DEMO)
    cell.until(lambda: rt.held == "plate-1")           # mid-pick
    cell.tick(5)
    rt.estop("op")
    assert cell.job(first)["status"] == "ABORTED" and cell.job(first)["error"] == "estop"
    assert rt.held == "plate-1" and "plate-1" not in rt.labware_at
    q_stop = rt.arm.read_state().q
    cell.tick(20)
    assert rt.arm.read_state().q == q_stop and rt.interlock.tripped
    assert cell.job(second)["status"] == "APPROVED"     # latched: nothing starts
    assert rt.reset("op") == [] and not rt.estop_pressed
    cell.tick()
    j2 = cell.job(second)   # approved from the initial state; the cell has moved on -> refused, never run
    assert j2["status"] == "FAILED" and "changed" in j2["error"] and rt.arm.read_state().q == q_stop
    plan = rt.dry_run(Program.model_validate(DEMO))
    assert plan.start_state.held == "plate-1" and plan.start_state.q == q_stop and not plan.ok


def test_entry_request_stops_then_unlocks(cell):
    rt = cell.rt
    jid = cell.approve(SMALL)
    cell.tick(30)
    n = rt.arm.n_servo
    rt.door_action("unlock", "op")
    assert cell.job(jid)["status"] == "ABORTED" and cell.job(jid)["error"] == "entry_requested"
    assert not rt.door.state().locked and rt.interlock.reason == "entry_requested"
    cell.tick(10)
    assert rt.arm.n_servo == n and rt.arm.read_state().safety == "protective_stop"


def test_slot_end_grace_and_camera_loss(cell):
    rt = cell.rt
    jid = cell.approve(SMALL)
    cell.tick(10)
    rt.run.booking["end_ts"] = cell.clock.t - 29.9   # the run overran its slot: 29.9 s ago
    cell.tick()
    assert cell.job(jid)["status"] == "RUNNING"     # still inside the 30 s grace
    cell.tick(5)
    assert cell.job(jid)["error"] == "slot_ended" and rt.interlock.reason == "slot_ended"

    c2 = Cell(cell.tmp / "b")
    jid = c2.approve(SMALL)
    c2.tick(10)
    c2.rt.camera.alive = False
    c2.tick(80)
    assert c2.job(jid)["status"] == "RUNNING"
    c2.until(lambda: c2.job(jid)["status"] != "RUNNING", limit=30)
    assert c2.job(jid)["error"] == "camera_lost"
    assert c2.rt.reset("op") == ["camera_lost"]
    c2.rt.camera.alive = True
    assert c2.rt.reset("op") == []


def test_physical_estop_latches_when_idle(cell):
    rt = cell.rt
    rt.door.lock()
    rt.arm.estop = True
    cell.tick()
    assert rt.interlock.tripped and rt.interlock.reason == "estop"
    assert rt.reset("op") == ["estop"]
    rt.arm.estop = False
    assert rt.reset("op") == [] and rt.arm.read_state().safety == "normal"


def test_privacy_frame_when_door_open(cell):
    rt = cell.rt
    lit = rt.camera.frame()
    rt.door.open()
    cell.tick(10)
    dark = rt.camera.frame()
    assert dark.shape == lit.shape and dark.mean() < 30 < lit.mean()
