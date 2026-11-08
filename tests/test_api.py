import io
import json
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from wetlab.plan import DT
from wetlab.service.app import create_app
from wetlab.spec import DEFAULT_LAB

LAB = DEFAULT_LAB
DEMO = json.loads((Path(__file__).resolve().parent.parent / "examples" / "demo_protocol.json").read_text())
SMALL = {"name": "small", "steps": [{"op": "home", "speed": 0.5}, {"op": "move_joints", "q_deg": [15, -100, 100, -90, -90, 0], "speed": 0.5},
                                    {"op": "wait", "seconds": 3}, {"op": "capture", "tag": "x"}, {"op": "home", "speed": 0.5}]}
T0 = 1_800_000_000 + 60          # 60 s into a 15-min slot
SLOT0, H = T0 - 60, 3600


class Clock:
    def __init__(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t


class Env:
    def __init__(self, tmp: Path, render: bool = False):
        self.clock = Clock(T0)
        self.app = create_app({"db_path": tmp / "w.db", "artifacts_root": tmp / "art", "clock": self.clock, "run_threads": False,
                               "render_dry_run": render, "worker_token": "wt", "cells": [{"lab_path": str(LAB), "driver": "sim"}]})
        self.c, self.db, self.rt = TestClient(self.app), self.app.state.db, self.app.state.runtimes["cell-a"]
        hdr = lambda key: {"Authorization": f"Bearer {key}"}  # noqa: E731
        self.A, self.B, self.OP = (hdr(self.db.add_user(org, name, role)[1]) for org, name, role in
                                   (("acme", "alice", "renter"), ("beta", "bob", "renter"), ("exo", "olga", "operator")))

    def book(self, h, start=SLOT0, end=SLOT0 + H):
        return self.c.post("/v1/bookings", headers=h, json={"cell": "cell-a", "start_ts": start, "end_ts": end})

    def submit(self, h, source, mode="run", booking=None, kind="ir", **kw):
        return self.c.post("/v1/jobs", headers=h, json={"booking_id": booking, "kind": kind, "source": source, "mode": mode, **kw})

    def status(self, jid):
        return self.db.one("SELECT status FROM jobs WHERE id=?", jid)["status"]

    def tick(self, n=1):
        for _ in range(n):
            self.clock.t += DT
            self.rt.tick()

    def until(self, pred, limit=20000):
        for _ in range(limit):
            if pred():
                return
            self.tick()
        raise AssertionError("condition not reached")

    def token(self, h, cell="cell-a"):
        return self.c.post("/v1/stream-tokens", headers=h, json={"cell": cell})


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def test_book_submit_run_download(tmp_path):
    e = Env(tmp_path, render=True)
    bk = e.book(e.A).json()
    assert e.book(e.B, SLOT0 + 900, SLOT0 + 1800).status_code == 409       # overlap rejected
    r = e.submit(e.A, DEMO, booking=bk["id"])
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["status"] == "APPROVED" and job["plan"]["ok"] and job["violations"] == [] and job["plan"]["duration_s"] > 10
    jid = job["id"]
    # renter B sees nothing of A's
    assert e.c.get(f"/v1/jobs/{jid}", headers=e.B).status_code == 404 and e.c.get("/v1/jobs", headers=e.B).json() == []
    assert e.c.get(f"/v1/jobs/{jid}/artifacts/plan.json", headers=e.B).status_code == 404
    assert [j["id"] for j in e.c.get("/v1/jobs", headers=e.A).json()] == [jid]

    e.until(lambda: e.status(jid) == "RUNNING")
    e.tick(400)
    st_a = e.c.get("/v1/cells/cell-a/state", headers=e.A).json()
    assert st_a["job"]["id"] == jid and 0 < st_a["job"]["progress"] < 1 and st_a["door"]["locked"] and st_a["booking_active"] == bk["id"]
    assert e.c.get("/v1/cells/cell-a/state", headers=e.B).json()["job"]["id"] is None
    e.until(lambda: e.status(jid) != "RUNNING")

    job = e.c.get(f"/v1/jobs/{jid}", headers=e.A).json()
    assert job["status"] == "SUCCEEDED" and job["max_deviation_rad"] < 0.01
    names = {a["name"] for a in job["artifacts"]}
    assert {"plan.json", "recording.mp4", "joints.csv", "events.jsonl", "report.json", "results.zip"} <= names
    assert any(n.startswith("capture_") and n.endswith("after-shake.jpg") for n in names)
    art = lambda n: e.c.get(f"/v1/jobs/{jid}/artifacts/{n}", headers=e.A)  # noqa: E731
    assert art("recording.mp4").content[4:8] == b"ftyp"
    report = art("report.json").json()
    assert report["status"] == "SUCCEEDED" and report["instrument_results"][0] == {"rpm": 600.0, "seconds": 3.0, "ok": True}
    assert report["final_state"]["labware_at"]["plate-1"] == "nest-A1"
    for bad in ("..%2Fw.db", "%2Fetc%2Fpasswd", ".results.zip.tmp", "nope.txt"):
        assert art(bad).status_code == 404
    z = zipfile.ZipFile(io.BytesIO(e.c.get(f"/v1/jobs/{jid}/results.zip", headers=e.A).content))
    assert {"recording.mp4", "joints.csv", "events.jsonl", "report.json"} <= set(z.namelist())

    usage = e.c.get("/v1/usage", headers=e.A).json()
    assert {"slot_minutes", "sim_dry_run", "arm_seconds"} <= set(usage["totals"]) and usage["total"] == pytest.approx(120.25)
    assert e.c.get("/v1/usage", headers=e.B).json()["events"] == []
    assert e.c.get("/v1/invoice?month=2027-01", headers=e.A).json()["total"] == pytest.approx(120.25)
    for _ in range(300):   # dry-run video is rendered in a background thread
        if (Path(tmp_path / "art" / jid / "dry_run.mp4")).is_file():
            break
        time.sleep(0.1)
    assert (tmp_path / "art" / jid / "dry_run.mp4").read_bytes()[4:8] == b"ftyp"


def test_door_open_mid_run_aborts_and_latches(env):
    e, rt = env, env.rt
    bk = e.book(e.A).json()
    jid = e.submit(e.A, SMALL, booking=bk["id"]).json()["id"]
    e.until(lambda: e.status(jid) == "RUNNING")
    e.tick(30)
    assert e.c.post("/v1/cells/cell-a/door", headers=e.A, json={"action": "open"}).status_code == 403   # renters can't
    n = rt.arm.n_servo
    assert e.c.post("/v1/cells/cell-a/door", headers=e.OP, json={"action": "open"}).status_code == 200
    e.tick()   # one tick later the door is ajar
    j = e.c.get(f"/v1/jobs/{jid}", headers=e.A).json()
    assert j["status"] == "ABORTED" and j["error"] == "door_open"
    assert rt.arm.n_servo == n and rt.arm.read_state().safety == "protective_stop"   # no servo after the trip
    e.tick(50)
    assert rt.arm.n_servo == n
    il = e.c.get("/v1/cells/cell-a/state", headers=e.OP).json()["interlock"]
    assert il["tripped"] and il["reason"] == "door_open" and not il["can_start"]

    j2 = e.submit(e.A, SMALL, booking=bk["id"]).json()
    assert j2["status"] == "APPROVED"
    e.tick(100)
    assert e.status(j2["id"]) == "APPROVED"
    assert e.c.post("/v1/cells/cell-a/reset", headers=e.A).status_code == 403
    r = e.c.post("/v1/cells/cell-a/reset", headers=e.OP)
    assert r.status_code == 409 and "door_open" in r.json()["detail"]["faults"]
    e.c.post("/v1/cells/cell-a/door", headers=e.OP, json={"action": "unlock"})
    e.c.post("/v1/cells/cell-a/door", headers=e.OP, json={"action": "close"})
    e.tick(100)
    assert e.c.post("/v1/cells/cell-a/reset", headers=e.OP).json()["detail"]["faults"] == ["door_unlocked"]
    e.c.post("/v1/cells/cell-a/door", headers=e.OP, json={"action": "lock"})
    e.tick(10)
    assert e.status(j2["id"]) == "APPROVED"   # closed + locked, but still latched until the operator resets
    assert e.c.post("/v1/cells/cell-a/reset", headers=e.OP).status_code == 200
    e.until(lambda: e.status(j2["id"]) not in ("APPROVED", "RUNNING"))
    assert e.status(j2["id"]) == "SUCCEEDED"
    arm_s = e.db.all("SELECT job_id, qty FROM meter_events WHERE kind='arm_seconds'")
    assert {m["job_id"] for m in arm_s} == {jid, j2["id"]} and all(m["qty"] > 0 for m in arm_s)
    actions = {a["action"] for a in e.db.all("SELECT action FROM audit")}
    assert {"interlock_trip", "door_open", "interlock_reset", "job_end"} <= actions


def test_stream_and_ws_need_token_and_active_booking(env):
    e = env
    assert e.token(e.A).status_code == 403                       # no booking yet
    assert e.token(e.OP).status_code == 200                      # operators always
    e.book(e.A)
    tok = e.token(e.A).json()
    assert tok["expires"] > T0 and e.token(e.B).status_code == 403
    url = "/v1/cells/cell-a/stream.mjpg"
    assert e.c.get(url).status_code == 401
    assert e.c.get(url, params={"token": "garbage"}).status_code == 403
    assert e.c.get(url, params={"token": tok["token"][:-2] + "00"}).status_code == 403      # bad signature
    assert e.c.get("/v1/cells/cell-z/stream.mjpg", params={"token": tok["token"]}).status_code == 404
    r = e.c.get(url, params={"token": tok["token"], "frames": 2})
    assert r.status_code == 200 and r.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert r.content.count(b"--frame\r\nContent-Type: image/jpeg") == 2 and b"\xff\xd8" in r.content

    with e.c.websocket_connect(f"/v1/cells/cell-a/ws?token={tok['token']}") as ws:
        snap = ws.receive_json()
    assert {"t", "cell", "q", "gripper_mm", "held", "labware_at", "door", "interlock", "job", "booking_active"} <= set(snap)
    with e.c.websocket_connect("/v1/cells/cell-a/ws", headers=e.OP) as ws:   # API key (Isaac mirror)
        assert ws.receive_json()["cell"] == "cell-a"
    with pytest.raises(WebSocketDisconnect):                                    # renter key, no active booking
        with e.c.websocket_connect("/v1/cells/cell-a/ws", headers=e.B) as ws:
            ws.receive_json()
    for bad in ("/v1/cells/cell-a/ws", "/v1/cells/cell-a/ws?token=x"):
        with pytest.raises(WebSocketDisconnect):
            with e.c.websocket_connect(bad) as ws:
                ws.receive_json()
    e.clock.t += 301
    assert e.c.get(url, params={"token": tok["token"], "frames": 1}).status_code == 403   # expired


def test_isaac_gate_and_worker_queue(env):
    e = env
    bk = e.book(e.A).json()
    j = e.submit(e.A, SMALL, booking=bk["id"], isaac=True).json()
    assert j["status"] == "APPROVED" and j["isaac_status"] == "pending"
    e.tick(20)
    assert e.status(j["id"]) == "APPROVED"            # waits for the Isaac verdict
    assert e.c.get("/v1/internal/sim-jobs/next").status_code == 401
    W = {"Authorization": "Bearer wt"}
    nxt = e.c.get("/v1/internal/sim-jobs/next", headers=W).json()
    assert nxt["job_id"] == j["id"] and nxt["plan"]["waypoints"] and nxt["spec"]["id"] == "cell-a"
    assert e.c.get("/v1/internal/sim-jobs/next", headers=W).status_code == 204
    r = e.c.post(f"/v1/internal/sim-jobs/{j['id']}/result", headers=W,
                 files={"report": ("report.json", json.dumps({"ok": True, "wall_s": 90}), "application/json"),
                        "video": ("video.mp4", b"\x00\x00\x00\x18ftypisom", "video/mp4")})
    assert r.json()["isaac_status"] == "ok"
    names = {a["name"] for a in e.c.get(f"/v1/jobs/{j['id']}", headers=e.A).json()["artifacts"]}
    assert {"isaac_report.json", "isaac.mp4"} <= names
    assert e.db.one("SELECT qty FROM meter_events WHERE kind='isaac_minutes'")["qty"] == 1.5
    e.tick()
    assert e.status(j["id"]) == "RUNNING"


def _docker_ok() -> bool:
    try:
        return bool(shutil.which("docker")) and subprocess.run(["docker", "image", "inspect", "python:3.12-slim"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


def test_compile_failures_dry_runs_and_cancel(env):
    e = env
    r = e.submit(e.A, {"steps": [{"op": "self_destruct"}]}, mode="dry_run", cell="cell-a").json()
    assert r["status"] == "COMPILE_FAILED" and "invalid IR" in r["error"]
    assert e.submit(e.A, SMALL).status_code == 422                                  # run needs a booking
    r = e.submit(e.A, {"steps": [{"op": "pick", "labware": "plate-9"}]}, mode="dry_run").json()
    assert r["status"] == "DRY_RUN_FAILED" and r["violations"][0]["code"] == "unknown_ref"
    r = e.submit(e.A, json.dumps(SMALL), mode="dry_run").json()                       # IR as a JSON string
    assert r["status"] == "DRY_RUN_OK" and e.status(r["id"]) == "DRY_RUN_OK"
    assert "plan.json" in zipfile.ZipFile(io.BytesIO(e.c.get(f"/v1/jobs/{r['id']}/results.zip", headers=e.A).content)).namelist()
    bk = e.book(e.A).json()
    assert e.submit(e.B, SMALL, booking=bk["id"]).status_code == 404                 # not B's booking
    j = e.submit(e.A, SMALL, booking=bk["id"]).json()
    assert e.c.post(f"/v1/jobs/{j['id']}/cancel", headers=e.B).status_code == 404
    assert e.c.post(f"/v1/jobs/{j['id']}/cancel", headers=e.A).json()["status"] == "CANCELLED"
    assert e.c.post(f"/v1/jobs/{j['id']}/cancel", headers=e.A).status_code == 409
    e.tick(10)
    assert e.status(j["id"]) == "CANCELLED"
    j = e.submit(e.A, SMALL, booking=bk["id"]).json()
    e.until(lambda: e.status(j["id"]) == "RUNNING")
    e.tick(20)
    assert e.c.post(f"/v1/jobs/{j['id']}/cancel", headers=e.A).json()["status"] == "CANCELLED"
    assert not e.rt.interlock.tripped and e.rt.snapshot["job"] is None             # a cancel is not a safety trip
    assert e.db.one("SELECT COUNT(*) AS n FROM meter_events WHERE kind='sim_dry_run'")["n"] == 4

    if _docker_ok():
        src = "from wetlab import protocol\nprint('compiling')\np = protocol('py')\np.home(speed=0.5)\np.wait(1)\n"
        r = e.submit(e.A, src, kind="python", mode="dry_run").json()
        assert r["status"] == "DRY_RUN_OK" and "compiling" in r["compile_logs"] and r["ir"]["name"] == "py"
        r = e.submit(e.A, "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)", kind="python", mode="dry_run").json()
        assert r["status"] == "COMPILE_FAILED" and "unreachable" in r["compile_logs"].lower()


def test_cells_spec_me_stage(env):
    e = env
    assert e.c.get("/v1/me", headers=e.A).json()["name"] == "alice" and e.c.get("/v1/me").status_code == 401
    cells = e.c.get("/v1/cells", headers=e.A).json()
    assert cells[0]["id"] == "cell-a" and cells[0]["camera"]["fps"] == 15
    assert e.c.get("/v1/cells/cell-a/spec", headers=e.A).json()["doors"][0]["id"] == "door-1"
    r = e.c.get("/v1/cells/cell-a/stage.usda", headers=e.A)
    assert r.status_code == 503 or (r.status_code == 200 and r.text.startswith("#usda"))
    assert e.c.get("/v1/cells/cell-a/stage.usda", headers={"Authorization": "Bearer wt"}).status_code in (200, 503)
    assert e.c.get("/v1/cells/cell-a/stage.usda", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert e.c.post("/v1/cells/cell-a/estop", headers=e.A).status_code == 403
    assert e.c.post("/v1/cells/cell-a/estop", headers=e.OP).json()["interlock"]["reason"] == "estop"
