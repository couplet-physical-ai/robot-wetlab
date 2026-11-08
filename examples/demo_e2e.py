"""End-to-end demo against a running wetlab server (stdlib only).

    .venv/bin/python -m wetlab.cli serve --port 8765 &
    .venv/bin/python examples/demo_e2e.py --bootstrap            # creates a renter + operator in var/wetlab.db
    .venv/bin/python examples/demo_e2e.py --renter-key wl_… --operator-key wl_…

book the current slot -> Python dry-run -> run -> results.zip -> run again and open the door
mid-run (job ABORTED door_open, interlock latched, arm stopped) -> close + lock + reset -> usage.
Exits non-zero on the first failed expectation.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLOT_S = 900
PROTOCOL = '''from wetlab import protocol

p = protocol("demo: shake plate-1, image it, put it back")
p.home()
p.pick("plate-1")
p.place("plate-1", "shaker-1")
p.shake("shaker-1", rpm=600, seconds=3)
p.capture("after-shake")
p.pick("plate-1")
p.place("plate-1", "nest-A1")
p.move_to(labware="plate-2", well="A1", z_above=0.03)
p.home()
'''


class Fail(Exception):
    pass


def check(cond, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def say(icon: str, msg: str) -> None:
    print(f"{icon} {msg}", flush=True)


class Client:
    def __init__(self, server: str, key: str):
        self.server, self.key = server.rstrip("/"), key

    def req(self, method: str, path: str, body=None, raw: bool = False):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(self.server + path, data=data, method=method,
                                   headers={"Authorization": f"Bearer {self.key}", **({"Content-Type": "application/json"} if data else {})})
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                out = resp.read()
                return resp.status, (out if raw else json.loads(out or b"null"))
        except urllib.error.HTTPError as e:
            txt = e.read()
            try:
                return e.code, json.loads(txt)
            except ValueError:
                return e.code, txt.decode(errors="replace")

    def ok(self, method: str, path: str, body=None, raw: bool = False):
        code, out = self.req(method, path, body, raw)
        check(200 <= code < 300, f"{method} {path} -> {code}: {out if not raw else '…'}")
        return out


def wait(pred, timeout: float, what: str, every: float = 0.25):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(every)
    raise Fail(f"timed out after {timeout:g} s waiting for {what}")


def bootstrap(db_path: str) -> tuple[str, str]:
    sys.path.insert(0, str(ROOT))
    from wetlab.service.db import DB   # local only: writes straight into the server's SQLite file
    db = DB(db_path)
    return db.add_user("acme-bio", "demo-renter", "renter")[1], db.add_user("demo-operator-org", "demo-operator", "operator")[1]


def job_status(c: Client, jid: str) -> dict:
    return c.ok("GET", f"/v1/jobs/{jid}")


def clear_cell(op: Client, cell: str) -> None:
    """Leave the cell closed, locked and reset (idempotent: also fixes a cell left tripped by an earlier run)."""
    st = op.ok("GET", f"/v1/cells/{cell}/state")
    if not st["door"]["closed"]:
        op.ok("POST", f"/v1/cells/{cell}/door", {"action": "close"})
    op.ok("POST", f"/v1/cells/{cell}/door", {"action": "lock"})

    def reset():
        code, out = op.req("POST", f"/v1/cells/{cell}/reset")
        return out if code == 200 else None
    if st["interlock"]["tripped"] or not st["door"]["closed"]:
        wait(reset, 10, "the interlock reset (door closing)")


def book_now(r: Client, cell: str, me: dict) -> dict:
    now = time.time()
    for b in r.ok("GET", "/v1/bookings"):   # reuse my booking if one is already active with time to spare
        if b["status"] == "confirmed" and b["start_ts"] <= now < b["end_ts"] - 300 and b["user_id"] == me["id"]:
            return b
    start = int(now // SLOT_S * SLOT_S)
    end = start + SLOT_S * (2 if start + SLOT_S - now < 300 else 1)   # < 5 min left in this slot: take the next one too
    code, b = r.req("POST", "/v1/bookings", {"cell": cell, "start_ts": start, "end_ts": end})
    check(code == 201, f"booking {start}-{end} -> {code}: {b}")
    return b


def run(a) -> None:
    renter_key, op_key = (bootstrap(a.db) if a.bootstrap else (a.renter_key, a.operator_key))
    check(renter_key and op_key, "pass --renter-key and --operator-key, or --bootstrap")
    r, op = Client(a.server, renter_key), Client(a.server, op_key)
    me, op_me = r.ok("GET", "/v1/me"), op.ok("GET", "/v1/me")
    check(op_me["role"] in ("operator", "admin"), f"operator key has role {op_me['role']}")
    cell = r.ok("GET", "/v1/cells")[0]["id"]
    say("👤", f"renter {me['name']} ({me['org']}), operator {op_me['name']}, cell {cell}")
    clear_cell(op, cell)

    # 1. book
    b = book_now(r, cell, me)
    say("📅", f"booked {time.strftime('%H:%M', time.gmtime(b['start_ts']))}-{time.strftime('%H:%M', time.gmtime(b['end_ts']))} UTC ({b['id']})")
    code, _ = r.req("POST", "/v1/stream-tokens", {"cell": cell})
    check(code == 200, f"renter stream token during own booking -> {code}")

    # 2. dry-run (Python through the sandbox)
    j = r.ok("POST", "/v1/jobs", {"booking_id": b["id"], "kind": "python", "source": PROTOCOL, "mode": "dry_run"})
    check(j["status"] == "DRY_RUN_OK", f"dry-run status {j['status']}: {j.get('error')} {j.get('violations')}")
    say("🧪", f"dry-run {j['id']}: ok, {j['plan']['duration_s']:.1f} s, {len(j['violations'])} violations, "
              f"{j['plan']['n_waypoints']} waypoints")

    # 3. run -> SUCCEEDED -> results.zip
    j = r.ok("POST", "/v1/jobs", {"booking_id": b["id"], "kind": "python", "source": PROTOCOL, "mode": "run"})
    check(j["status"] == "APPROVED", f"run status {j['status']}: {j.get('error')}")
    jid, dur = j["id"], j["plan"]["duration_s"]
    wait(lambda: job_status(r, jid)["status"] in ("RUNNING", "SUCCEEDED"), 60, f"{jid} RUNNING")
    say("🤖", f"run {jid}: RUNNING (planned {dur:.1f} s)")
    j = wait(lambda: (s := job_status(r, jid))["status"] not in ("APPROVED", "RUNNING") and s, dur + 90, f"{jid} to finish")
    check(j["status"] == "SUCCEEDED", f"run ended {j['status']}: {j.get('error')}")
    z = r.ok("GET", f"/v1/jobs/{jid}/results.zip", raw=True)
    names = set(zipfile.ZipFile(io.BytesIO(z)).namelist())
    need = {"recording.mp4", "joints.csv", "events.jsonl", "report.json", "plan.json"}
    check(need <= names, f"results.zip missing {need - names}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{jid}-results.zip").write_bytes(z)
    caps = sorted(n for n in names if n.startswith("capture_"))
    say("✅", f"run {jid}: SUCCEEDED, max deviation {j['max_deviation_rad']:.4f} rad, results.zip {len(z) // 1024} KB "
              f"({len(names)} files, captures: {', '.join(caps) or 'none'}) -> {out / (jid + '-results.zip')}")
    (out / "last_success.txt").write_text(jid + "\n")

    # 4. run again, open the door mid-run
    code, _ = r.req("POST", f"/v1/cells/{cell}/door", {"action": "open"})
    check(code == 403, f"renter door open should be 403, got {code}")
    j = r.ok("POST", "/v1/jobs", {"booking_id": b["id"], "kind": "python", "source": PROTOCOL, "mode": "run"})
    jid2 = j["id"]
    wait(lambda: job_status(r, jid2)["status"] == "RUNNING", 60, f"{jid2} RUNNING")
    q0 = op.ok("GET", f"/v1/cells/{cell}/state")["q"]
    time.sleep(3.0)
    st = op.ok("GET", f"/v1/cells/{cell}/state")
    check(st["job"] and st["job"]["id"] == jid2, f"{jid2} should still be running after 3 s: {st['job']}")
    moved = max(abs(x - y) for x, y in zip(q0, st["q"]))
    say("🤖", f"run {jid2}: RUNNING 3 s, progress {st['job']['progress']:.0%}, joints moved {moved:.3f} rad -> operator opens the door")
    op.ok("POST", f"/v1/cells/{cell}/door", {"action": "open"})
    j = wait(lambda: (s := job_status(r, jid2))["status"] != "RUNNING" and s, 5, f"{jid2} to stop")
    check(j["status"] == "ABORTED" and j["error"] == "door_open", f"expected ABORTED door_open, got {j['status']} {j['error']}")
    st1 = op.ok("GET", f"/v1/cells/{cell}/state")
    il = st1["interlock"]
    check(il["tripped"] and il["reason"] == "door_open" and not il["can_start"], f"interlock not latched: {il}")
    time.sleep(1.5)
    st2 = op.ok("GET", f"/v1/cells/{cell}/state")
    drift = max(abs(x - y) for x, y in zip(st1["q"], st2["q"]))
    check(drift < 1e-9 and st2["job"] is None, f"arm still moving after the trip (drift {drift} rad, job {st2['job']})")
    say("🚪", f"door opened: {jid2} ABORTED (door_open), interlock latched, arm {st2['arm_safety']}, joints frozen (drift {drift:.1e} rad over 1.5 s)")
    if a.leave_tripped:
        say("⏸️", "--leave-tripped: cell left open and latched")
        return

    # 5. close + lock + reset
    code, out = op.req("POST", f"/v1/cells/{cell}/reset")
    check(code == 409, f"reset with the door open should be 409, got {code} {out}")
    clear_cell(op, cell)
    st = op.ok("GET", f"/v1/cells/{cell}/state")
    check(not st["interlock"]["tripped"] and st["door"]["closed"] and st["door"]["locked"], f"cell not clear after reset: {st['interlock']} {st['door']}")
    say("🔓", "door closed + locked, operator reset: interlock clear")

    # 6. usage + invoice
    u, inv = r.ok("GET", "/v1/usage"), r.ok("GET", "/v1/invoice")
    check({"slot_minutes", "sim_dry_run", "arm_seconds"} <= set(u["totals"]), f"usage totals missing kinds: {u['totals']}")
    arm_jobs = {e["job_id"] for e in u["events"] if e["kind"] == "arm_seconds"}
    check({jid, jid2} <= arm_jobs, f"arm_seconds not metered for both runs: {arm_jobs}")
    say("🧾", "usage " + ", ".join(f"{k} ${v:.2f}" for k, v in sorted(u["totals"].items())) + f"; invoice {inv['month']} {inv['org']}: ${inv['total']:.2f}")
    say("🎉", "demo passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--renter-key")
    ap.add_argument("--operator-key")
    ap.add_argument("--bootstrap", action="store_true", help="create a renter + operator directly in --db (server on this machine)")
    ap.add_argument("--db", default="var/wetlab.db", help="the server's DB, for --bootstrap")
    ap.add_argument("--out", default="out/demo_e2e", help="where results.zip is saved")
    ap.add_argument("--leave-tripped", action="store_true", help="stop after the door-open abort (for screenshots)")
    a = ap.parse_args()
    try:
        run(a)
    except Fail as e:
        say("❌", f"FAIL: {e}")
        sys.exit(1)
    except urllib.error.URLError as e:
        say("❌", f"FAIL: cannot reach {a.server}: {e.reason}")
        sys.exit(2)


if __name__ == "__main__":
    main()
