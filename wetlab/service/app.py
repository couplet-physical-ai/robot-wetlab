"""wetlab service: FastAPI over SQLite + one CellRuntime per cell. API table: docs/PLAN.md section 5.

Auth: `Authorization: Bearer <api_key>` (sha256 stored). Renters see only their own bookings,
jobs, artifacts and usage; operators/admins see everything and drive door / E-stop / reset.
The camera stream and state WebSocket take a short-lived signed `?token=` (POST /v1/stream-tokens)
so an <img> tag can use it; renters only get one during their own active booking.
"""
from __future__ import annotations

import asyncio
import base64
import calendar
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, WebSocket
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from ..billing import Billing, LedgerSink, StripeMeterSink, invoice, month_range
from ..hw.sim import SimShaker, TwinCamera, VirtualArm, VirtualDoor
from ..ir import Program
from ..runtime import CellRuntime, plan_summary, zip_artifacts
from ..sandbox import CompileError, compile_python
from ..spec import DEFAULT_LAB, load_lab
from ..twin.planner import initial_state
from .db import DB, new_id

SLOT_S, MIN_S, MAX_S = 900, 900, 4 * 3600
TOKEN_TTL_S = 300
MAX_SOURCE = 256_000
STAFF = ("operator", "admin")
STATIC = Path(__file__).resolve().parent / "static"
DEFAULTS = {"db_path": "var/wetlab.db", "artifacts_root": "var/artifacts",
            "cells": [{"lab_path": str(DEFAULT_LAB), "driver": "sim"}],
            "run_threads": True, "speed": 1.0, "render_dry_run": True}
PENDING = ("SUBMITTED", "COMPILING", "DRY_RUN_OK", "APPROVED")


class BookingIn(BaseModel):
    cell: str
    start_ts: int          # UTC epoch s, 15-min aligned
    end_ts: int


class JobIn(BaseModel):
    booking_id: str | None = None
    cell: str | None = None              # only for dry runs without a booking
    kind: Literal["python", "ir"]
    source: str | dict                   # python text, or IR as a JSON object / string
    mode: Literal["dry_run", "run"] = "dry_run"
    isaac: bool = False


class TokenIn(BaseModel):
    cell: str


class DoorIn(BaseModel):
    action: Literal["open", "close", "lock", "unlock"]


def build_cell(c: dict, db, billing, artifacts_root, clock, speed) -> CellRuntime:
    lab = load_lab(c["lab_path"])
    instruments = {loc.id: SimShaker() for loc in lab.locations if loc.driver == "sim_shaker"}
    if c.get("driver", "sim") == "sim":
        arm = VirtualArm(initial_state(lab).q, lab.arm.tool.open_mm, clock=clock)
        door = VirtualDoor(lab, clock=clock)
        camera = TwinCamera(lab, lambda: rt.cell_state(), door)   # rt is bound below
    elif c["driver"] == "ur":   # UNTESTED ON HARDWARE: real-driver stubs
        from ..hw.door_relay import DoorRelay, URControllerIO
        from ..hw.rtsp import RTSPCamera
        from ..hw.ur_rtde import URRTDEArm
        arm = URRTDEArm(c["ur_host"])
        arm.connect()
        door = DoorRelay(URControllerIO(c["ur_host"]))
        cam = lab.camera()
        camera = RTSPCamera(c["rtsp_url"], cam.id, cam.resolution[0], cam.resolution[1], cam.fps)
    else:
        raise ValueError(f"unknown driver {c['driver']!r}")
    rt = CellRuntime(lab, arm, door, camera, instruments, db, billing, artifacts_root, clock=clock, speed=speed)
    return rt


def create_app(config: dict | None = None) -> FastAPI:
    cfg = {**DEFAULTS, **(config or {})}
    clock = cfg.get("clock") or time.time
    db = DB(cfg["db_path"])
    sinks = [LedgerSink(db)]
    if os.environ.get("STRIPE_API_KEY"):
        sinks.append(StripeMeterSink(lambda uid: (db.user(uid) or {}).get("stripe_customer_id")))
    billing = Billing(sinks)
    art = Path(cfg["artifacts_root"]).resolve()
    runtimes = {rt.cell: rt for rt in (build_cell(c, db, billing, art, clock, cfg["speed"]) for c in cfg["cells"])}
    secret = cfg.get("secret") or secrets.token_bytes(32)
    worker_token = cfg.get("worker_token") or os.environ.get("WETLAB_WORKER_TOKEN")

    @asynccontextmanager
    async def lifespan(_app):
        if cfg["run_threads"]:
            for rt in runtimes.values():
                rt.start()
        yield
        for rt in runtimes.values():
            rt.stop()

    app = FastAPI(title="wetlab", version="0.1", lifespan=lifespan)
    app.state.db, app.state.runtimes, app.state.billing = db, runtimes, billing

    # -- auth helpers ---------------------------------------------------------------
    def user(authorization: str | None = Header(None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing Authorization: Bearer <api_key>")
        u = db.user_by_key(authorization[7:].strip())
        if not u:
            raise HTTPException(401, "invalid API key")
        return u

    def staff(u: dict = Depends(user)) -> dict:
        if u["role"] not in STAFF:
            raise HTTPException(403, "operator only")
        return u

    def cell_rt(cell: str) -> CellRuntime:
        if cell not in runtimes:
            raise HTTPException(404, f"unknown cell {cell}")
        return runtimes[cell]

    def may_watch(u: dict, cell: str) -> bool:
        if u["role"] in STAFF:
            return True
        b = db.active_booking(cell, clock())
        return bool(b and b["user_id"] == u["id"])

    def sign(payload: str) -> str:
        return hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()[:32]

    def token_user(token: str | None, cell: str) -> dict | None:
        try:
            p64, sig = token.rsplit(".", 1)
            payload = base64.urlsafe_b64decode(p64 + "=" * (-len(p64) % 4)).decode()
            uid, c, exp = payload.split("|")
        except Exception:
            return None
        if not hmac.compare_digest(sig, sign(payload)) or c != cell or int(exp) < clock():
            return None
        u = db.user(uid)
        return u if u and may_watch(u, cell) else None

    def public_snapshot(rt: CellRuntime, u: dict) -> dict:
        s = dict(rt.snapshot)
        if s.get("job") and u["role"] not in STAFF and rt.job_owner() != u["id"]:
            s["job"] = {**s["job"], "id": None, "step": None}   # someone else's job: show busy only
        return s

    def visible_job(jid: str, u: dict) -> dict:
        j = db.one("SELECT * FROM jobs WHERE id=?", jid)
        if not j or (u["role"] not in STAFF and j["user_id"] != u["id"]):
            raise HTTPException(404, "job not found")
        return j

    def job_view(j: dict, full: bool = True) -> dict:
        out = {k: j[k] for k in ("id", "cell", "booking_id", "user_id", "kind", "mode", "status", "error", "created_ts",
                                 "started_ts", "finished_ts", "max_deviation_rad", "isaac_status")}
        out["plan"] = json.loads(j["plan_summary_json"]) if j["plan_summary_json"] else None
        out["violations"] = out["plan"]["violations"] if out["plan"] else []
        if full:
            d = Path(j["artifacts_dir"])
            files = sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")) if d.is_dir() else []
            out["artifacts"] = [{"name": p.name, "bytes": p.stat().st_size, "url": f"/v1/jobs/{j['id']}/artifacts/{p.name}"} for p in files]
            out["compile_logs"] = (d / "compile.log").read_text() if (d / "compile.log").is_file() else ""
            out["ir"] = json.loads(j["ir_json"]) if j["ir_json"] else None
        return out

    # -- me / cells -------------------------------------------------------------------
    @app.get("/v1/me")
    def me(u: dict = Depends(user)):
        return u

    @app.get("/v1/cells")
    def cells(u: dict = Depends(user)):
        return [{"id": rt.cell, "name": rt.lab.name, "lab_sha": rt.lab.sha256(),
                 "camera": {"id": rt.camera.info.id, "width": rt.camera.info.width, "height": rt.camera.info.height, "fps": rt.camera.info.fps},
                 "billing": rt.lab.billing.model_dump(), "interlock": rt.snapshot.get("interlock")} for rt in runtimes.values()]

    @app.get("/v1/cells/{cell}/state")
    def cell_state(cell: str, u: dict = Depends(user)):
        return public_snapshot(cell_rt(cell), u)

    @app.get("/v1/cells/{cell}/spec")
    def cell_spec(cell: str, u: dict = Depends(user)):
        return cell_rt(cell).lab.model_dump()

    @app.get("/v1/cells/{cell}/stage.usda")
    def stage(cell: str, authorization: str | None = Header(None)):
        if not (worker_token and hmac.compare_digest(authorization or "", f"Bearer {worker_token}")):
            user(authorization)   # any user key, or the Isaac worker's token
        lab = cell_rt(cell).lab
        path = art / "_stage" / f"{lab.id}-{lab.sha256()[:16]}.usda"   # lab.id, not the URL value
        if not path.is_file():
            try:
                from ..twin.usd import write_stage
            except ImportError as e:
                raise HTTPException(503, f"USD authoring unavailable (pip install usd-core): {e}")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{secrets.token_hex(4)}.usda")
            write_stage(lab, tmp)
            tmp.replace(path)
        return FileResponse(path, media_type="text/plain", filename=f"{lab.id}.usda")

    @app.get("/v1/cells/{cell}/availability")
    def availability(cell: str, date: str = Query(pattern=r"^\d{4}-\d{2}-\d{2}$"), u: dict = Depends(user)):
        cell_rt(cell)
        y, m, d = (int(x) for x in date.split("-"))
        day0 = calendar.timegm((y, m, d, 0, 0, 0))
        rows = db.all("SELECT user_id, start_ts, end_ts FROM bookings WHERE cell=? AND status='confirmed' AND start_ts<? AND end_ts>?",
                      cell, day0 + 86400, day0)
        first = clock() // SLOT_S * SLOT_S
        slots = []
        for t in range(day0, day0 + 86400, SLOT_S):
            b = next((r for r in rows if r["start_ts"] <= t < r["end_ts"]), None)
            slots.append({"start_ts": t, "end_ts": t + SLOT_S, "free": b is None and t >= first, "mine": bool(b and b["user_id"] == u["id"])})
        return {"cell": cell, "date": date, "slot_s": SLOT_S, "min_s": MIN_S, "max_s": MAX_S, "slots": slots}

    # -- bookings ---------------------------------------------------------------------
    @app.post("/v1/bookings", status_code=201)
    def book(b: BookingIn, u: dict = Depends(user)):
        rt, now = cell_rt(b.cell), clock()
        dur = b.end_ts - b.start_ts
        if b.start_ts % SLOT_S or b.end_ts % SLOT_S:
            raise HTTPException(422, "start_ts and end_ts must be aligned to 15-minute slots")
        if not MIN_S <= dur <= MAX_S:
            raise HTTPException(422, "a booking is 15 minutes to 4 hours")
        if b.start_ts < now // SLOT_S * SLOT_S:
            raise HTTPException(422, "bookings must start in the current slot or later")
        bid = new_id("bk")
        with db.tx():
            if db.one("SELECT id FROM bookings WHERE cell=? AND status='confirmed' AND start_ts<? AND end_ts>?", b.cell, b.end_ts, b.start_ts):
                raise HTTPException(409, "overlaps an existing booking")
            db.insert("bookings", id=bid, cell=b.cell, user_id=u["id"], start_ts=b.start_ts, end_ts=b.end_ts, status="confirmed", created_ts=now)
            billing.meter("slot_minutes", dur / 60, u, rt.lab.billing, booking_id=bid, ts=now)
            db.audit(u["id"], "booking_create", ts=now, booking=bid)
        return db.one("SELECT * FROM bookings WHERE id=?", bid)

    @app.get("/v1/bookings")
    def bookings(u: dict = Depends(user)):
        if u["role"] in STAFF:
            return db.all("SELECT * FROM bookings ORDER BY start_ts DESC LIMIT 500")
        return db.all("SELECT * FROM bookings WHERE user_id=? ORDER BY start_ts DESC LIMIT 500", u["id"])

    def visible_booking(bid: str, u: dict) -> dict:
        b = db.one("SELECT * FROM bookings WHERE id=?", bid)
        if not b or (u["role"] not in STAFF and b["user_id"] != u["id"]):
            raise HTTPException(404, "booking not found")
        return b

    @app.get("/v1/bookings/{bid}")
    def get_booking(bid: str, u: dict = Depends(user)):
        return visible_booking(bid, u)

    @app.delete("/v1/bookings/{bid}")
    def cancel_booking(bid: str, u: dict = Depends(user)):
        b, now = visible_booking(bid, u), clock()
        if b["status"] != "confirmed":
            raise HTTPException(409, "booking already cancelled")
        if now >= b["start_ts"]:
            raise HTTPException(409, "booking has already started")
        lab = cell_rt(b["cell"]).lab
        refund = b["start_ts"] - now > lab.billing.cancel_refund_hours * 3600
        with db.tx():
            db.update("bookings", bid, status="cancelled")
            db.exec("UPDATE jobs SET status='CANCELLED', error='booking cancelled', finished_ts=? WHERE booking_id=? AND status='APPROVED'", now, bid)
            if refund:
                billing.meter("slot_refund", -(b["end_ts"] - b["start_ts"]) / 60, db.user(b["user_id"]), lab.billing, booking_id=bid, ts=now)
            db.audit(u["id"], "booking_cancel", ts=now, booking=bid, refund=refund)
        return {**db.one("SELECT * FROM bookings WHERE id=?", bid), "refunded": refund}

    # -- jobs -------------------------------------------------------------------------
    @app.post("/v1/jobs", status_code=201)
    def submit(j: JobIn, u: dict = Depends(user)):
        now = clock()
        b = visible_booking(j.booking_id, u) if j.booking_id else None
        if j.mode == "run" and not b:
            raise HTTPException(422, "mode=run needs a booking_id")
        if b and (b["status"] != "confirmed" or b["end_ts"] <= now):
            raise HTTPException(409, "booking is cancelled or over")
        rt = cell_rt(b["cell"] if b else (j.cell or next(iter(runtimes))))
        source = j.source if isinstance(j.source, str) else json.dumps(j.source)
        if len(source) > MAX_SOURCE:
            raise HTTPException(413, f"source over {MAX_SOURCE} bytes")
        jid = new_id("job")
        d = art / jid
        d.mkdir(parents=True)
        db.insert("jobs", id=jid, cell=rt.cell, booking_id=b and b["id"], user_id=u["id"], kind=j.kind, mode=j.mode, source=source,
                  status="SUBMITTED", created_ts=now, artifacts_dir=str(d))
        db.audit(u["id"], "job_submit", ts=now, job=jid, mode=j.mode, kind=j.kind)
        db.update("jobs", jid, status="COMPILING")
        try:
            if j.kind == "python":
                program, logs = compile_python(source)
            else:
                program, logs = Program.model_validate_json(source), ""
        except (CompileError, ValidationError) as e:
            (d / "compile.log").write_text(getattr(e, "logs", ""))
            msg = str(e) if isinstance(e, CompileError) else f"invalid IR: {e}"
            db.update("jobs", jid, status="COMPILE_FAILED", error=msg[:4000], finished_ts=clock())
            return job_view(db.one("SELECT * FROM jobs WHERE id=?", jid))
        if logs:
            (d / "compile.log").write_text(logs)
        plan = rt.dry_run(program)
        billing.meter("sim_dry_run", 1, u, rt.lab.billing, booking_id=b and b["id"], job_id=jid, ts=clock())
        (d / "plan.json").write_text(plan.model_dump_json())
        if cfg["render_dry_run"]:
            threading.Thread(target=_render_dry_run, args=(rt.lab, plan, d), daemon=True).start()
        status = "DRY_RUN_FAILED" if not plan.ok else ("APPROVED" if j.mode == "run" else "DRY_RUN_OK")
        db.update("jobs", jid, ir_json=program.model_dump_json(), status=status, plan_summary_json=json.dumps(plan_summary(plan)),
                  isaac_status="pending" if j.isaac and plan.ok else None,
                  finished_ts=None if status == "APPROVED" else clock())
        return job_view(db.one("SELECT * FROM jobs WHERE id=?", jid))

    @app.get("/v1/jobs")
    def jobs(u: dict = Depends(user), status: str | None = None):
        sql, args = "SELECT * FROM jobs WHERE 1=1", []
        if u["role"] not in STAFF:
            sql, args = sql + " AND user_id=?", [u["id"]]
        if status:
            sql, args = sql + " AND status=?", [*args, status]
        return [job_view(j, full=False) for j in db.all(sql + " ORDER BY created_ts DESC, rowid DESC LIMIT 200", *args)]

    @app.get("/v1/jobs/{jid}")
    def get_job(jid: str, u: dict = Depends(user)):
        return job_view(visible_job(jid, u))

    @app.get("/v1/jobs/{jid}/artifacts/{name}")
    def artifact(jid: str, name: str, u: dict = Depends(user)):
        d = Path(visible_job(jid, u)["artifacts_dir"])
        # pick from the directory listing, so a crafted name can never become a path
        p = next((f for f in d.iterdir() if f.name == name and not f.name.startswith(".") and f.is_file()), None)
        if p is None:
            raise HTTPException(404, "no such artifact")
        return FileResponse(p)

    @app.get("/v1/jobs/{jid}/results.zip")
    def results(jid: str, u: dict = Depends(user)):
        j = visible_job(jid, u)
        if j["status"] in ("SUBMITTED", "COMPILING", "RUNNING"):
            raise HTTPException(409, f"job is {j['status']}")
        d = Path(j["artifacts_dir"])
        p = d / "results.zip" if (d / "results.zip").is_file() else zip_artifacts(d)
        return FileResponse(p, media_type="application/zip", filename=f"{jid}-results.zip")

    @app.post("/v1/jobs/{jid}/cancel")
    def cancel_job(jid: str, u: dict = Depends(user)):
        j = visible_job(jid, u)
        rt = cell_rt(j["cell"])
        with rt.lock:   # the runtime can't pick it up meanwhile
            j = db.one("SELECT * FROM jobs WHERE id=?", jid)
            if j["status"] == "RUNNING":
                rt.cancel(jid, u["id"])
            elif j["status"] in PENDING:
                db.update("jobs", jid, status="CANCELLED", error=f"cancelled by {u['id']}", finished_ts=clock())
            else:
                raise HTTPException(409, f"job is {j['status']}")
        db.audit(u["id"], "job_cancel", ts=clock(), job=jid)
        return job_view(db.one("SELECT * FROM jobs WHERE id=?", jid))

    # -- live: stream tokens, MJPEG, WebSocket ------------------------------------------
    @app.post("/v1/stream-tokens")
    def stream_token(t: TokenIn, u: dict = Depends(user)):
        cell_rt(t.cell)
        if not may_watch(u, t.cell):
            raise HTTPException(403, "the stream is only available during your active booking")
        exp = int(clock()) + TOKEN_TTL_S
        payload = f"{u['id']}|{t.cell}|{exp}"
        return {"token": base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=") + "." + sign(payload), "expires": exp}

    def check_token(token: str | None, cell: str) -> dict:
        if not token:
            raise HTTPException(401, "stream token required (?token=, POST /v1/stream-tokens)")
        u = token_user(token, cell)
        if not u:
            raise HTTPException(403, "invalid or expired stream token, or no active booking")
        return u

    @app.get("/v1/cells/{cell}/stream.mjpg")
    async def stream(cell: str, token: str | None = None, frames: int = Query(0, ge=0)):
        rt = cell_rt(cell)
        u = check_token(token, cell)
        period = 1.0 / rt.camera.info.fps

        async def gen():
            rt.viewers += 1
            try:
                n, checked = 0, time.monotonic()
                while frames == 0 or n < frames:
                    if time.monotonic() - checked > 1.0:     # booking over -> stream ends
                        if not may_watch(u, cell):
                            break
                        checked = time.monotonic()
                    jpeg = await asyncio.to_thread(rt.latest_jpeg)
                    if jpeg:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n" % (len(jpeg), jpeg)
                        n += 1
                    if frames == 0 or n < frames:
                        await asyncio.sleep(period)
            finally:
                rt.viewers -= 1

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.websocket("/v1/cells/{cell}/ws")
    async def ws(websocket: WebSocket, cell: str, token: str | None = None):
        u = token_user(token, cell) if cell in runtimes and token else None
        auth = websocket.headers.get("authorization", "")
        if not u and cell in runtimes and auth.startswith("Bearer "):   # API-key clients (Isaac mirror)
            k = db.user_by_key(auth[7:].strip())
            u = k if k and may_watch(k, cell) else None
        if not u:
            await websocket.close(code=4403)
            return
        rt = runtimes[cell]
        await websocket.accept()
        checked = time.monotonic()
        try:
            while True:
                if time.monotonic() - checked > 1.0:
                    if not may_watch(u, cell):
                        await websocket.close(code=4403)
                        return
                    checked = time.monotonic()
                await websocket.send_json(public_snapshot(rt, u))
                try:   # 10 Hz, and notice a client close
                    if (await asyncio.wait_for(websocket.receive(), 0.1))["type"] == "websocket.disconnect":
                        return
                except asyncio.TimeoutError:
                    pass
        except Exception:
            return   # client went away

    # -- operator -------------------------------------------------------------------------
    @app.post("/v1/cells/{cell}/door")
    def door(cell: str, body: DoorIn, u: dict = Depends(staff)):
        rt = cell_rt(cell)
        try:
            rt.door_action(body.action, u["id"])
        except ValueError as e:
            raise HTTPException(400, str(e))
        return rt.snapshot

    @app.post("/v1/cells/{cell}/estop")
    def estop(cell: str, u: dict = Depends(staff)):
        rt = cell_rt(cell)
        rt.estop(u["id"])
        return rt.snapshot

    @app.post("/v1/cells/{cell}/reset")
    def reset(cell: str, u: dict = Depends(staff)):
        rt = cell_rt(cell)
        remaining = rt.reset(u["id"])
        if remaining:
            raise HTTPException(409, {"message": "conditions not clear", "faults": remaining})
        return rt.snapshot

    # -- usage / invoice ----------------------------------------------------------------
    def this_month() -> str:
        return time.strftime("%Y-%m", time.gmtime(clock()))

    @app.get("/v1/usage")
    def usage(u: dict = Depends(user), month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$")):
        sql, args = "SELECT * FROM meter_events WHERE 1=1", []
        if u["role"] not in STAFF:
            sql, args = sql + " AND user_id=?", [u["id"]]
        if month:
            a, b = month_range(month)
            sql, args = sql + " AND ts>=? AND ts<?", [*args, a, b]
        events = db.all(sql + " ORDER BY ts DESC, id DESC LIMIT 1000", *args)
        totals: dict[str, float] = {}
        for e in events:
            totals[e["kind"]] = round(totals.get(e["kind"], 0) + e["amount"], 4)
        return {"events": events, "totals": totals, "total": round(sum(totals.values()), 2)}

    @app.get("/v1/invoice")
    def get_invoice(u: dict = Depends(user), month: str | None = Query(None, pattern=r"^\d{4}-\d{2}$"), org: str | None = None):
        return invoice(db, org if (org and u["role"] in STAFF) else u["org"], month or this_month())

    # -- Isaac worker queue ---------------------------------------------------------------
    def worker(authorization: str | None = Header(None)) -> None:
        if not worker_token:
            raise HTTPException(503, "Isaac worker queue disabled (set WETLAB_WORKER_TOKEN)")
        if not hmac.compare_digest(authorization or "", f"Bearer {worker_token}"):
            raise HTTPException(401, "bad worker token")

    @app.get("/v1/internal/sim-jobs/next", dependencies=[Depends(worker)])
    def next_sim_job():
        # shortcut: a job whose worker dies stays isaac_status='running'; requeue by hand (UPDATE ... 'pending')
        with db.tx():
            j = db.one("SELECT * FROM jobs WHERE isaac_status='pending' AND status IN ('DRY_RUN_OK','APPROVED') ORDER BY created_ts, rowid LIMIT 1")
            if not j:
                return Response(status_code=204)
            db.update("jobs", j["id"], isaac_status="running")
        plan = json.loads((Path(j["artifacts_dir"]) / "plan.json").read_text())
        return {"job_id": j["id"], "cell": j["cell"], "lab_sha": plan["lab_sha"], "ir": json.loads(j["ir_json"]), "plan": plan,
                "spec": runtimes[j["cell"]].lab.model_dump(), "result_url": f"/v1/internal/sim-jobs/{j['id']}/result"}

    @app.post("/v1/internal/sim-jobs/{jid}/result", dependencies=[Depends(worker)])
    async def sim_job_result(jid: str, report: UploadFile = File(...), video: UploadFile | None = File(None)):
        j = db.one("SELECT * FROM jobs WHERE id=?", jid)
        if not j:
            raise HTTPException(404, "job not found")
        try:
            rep = json.loads(await report.read())
        except ValueError:
            raise HTTPException(422, "report must be JSON")
        d = Path(j["artifacts_dir"])
        (d / "isaac_report.json").write_text(json.dumps(rep, indent=2))
        if video is not None:
            (d / "isaac.mp4").write_bytes(await video.read())
        ok = rep.get("ok") is True
        fields = {"isaac_status": "ok" if ok else "failed"}
        if not ok and j["status"] == "APPROVED":
            fields.update(status="DRY_RUN_FAILED", error=f"Isaac dry-run failed: {rep.get('error') or 'see isaac_report.json'}"[:4000])
        db.update("jobs", jid, **fields)
        rt = runtimes[j["cell"]]
        billing.meter("isaac_minutes", round(float(rep.get("wall_s") or 0) / 60, 3), db.user(j["user_id"]), rt.lab.billing,
                      booking_id=j["booking_id"], job_id=jid, ts=clock())
        db.audit("worker", "isaac_result", ts=clock(), job=jid, ok=ok)
        return {"job_id": jid, "isaac_status": fields["isaac_status"]}

    if STATIC.is_dir():
        app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def _render_dry_run(lab, plan, d: Path) -> None:
    from ..twin.render import render_plan_video
    tmp = d / ".dry_run.mp4"
    try:
        render_plan_video(lab, plan, tmp)
        tmp.replace(d / "dry_run.mp4")
    except Exception as e:   # the dry-run verdict stands without the video
        (d / "dry_run.error.txt").write_text(f"{type(e).__name__}: {e}")
