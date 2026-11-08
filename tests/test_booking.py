import pytest
from fastapi.testclient import TestClient

from wetlab.service.app import create_app
from wetlab.spec import DEFAULT_LAB

T0 = 1_800_000_000 + 60          # 60 s into a 15-min slot (2027-01-15 08:01 UTC)
SLOT0 = T0 - 60
H = 3600


class Clock:
    def __init__(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    clock = Clock(T0)
    app = create_app({"db_path": tmp_path / "w.db", "artifacts_root": tmp_path / "art", "clock": clock, "run_threads": False,
                      "render_dry_run": False, "cells": [{"lab_path": str(DEFAULT_LAB), "driver": "sim"}]})
    db = app.state.db
    hdr = lambda key: {"Authorization": f"Bearer {key}"}  # noqa: E731
    a, ka = db.add_user("acme", "alice")
    b, kb = db.add_user("beta", "bob")
    o, ko = db.add_user("exo", "olga", "operator")
    return TestClient(app), db, clock, hdr(ka), hdr(kb), hdr(ko)


def book(c, h, start, end, cell="cell-a"):
    return c.post("/v1/bookings", headers=h, json={"cell": cell, "start_ts": start, "end_ts": end})


def test_booking_rules(env):
    c, db, clock, A, B, OP = env
    assert c.post("/v1/bookings", json={"cell": "cell-a", "start_ts": SLOT0, "end_ts": SLOT0 + 900}).status_code == 401
    assert book(c, A, SLOT0 + 60, SLOT0 + 960).status_code == 422          # not aligned
    assert book(c, A, SLOT0, SLOT0).status_code == 422                     # too short
    assert book(c, A, SLOT0, SLOT0 + 4 * H + 900).status_code == 422       # > 4 h
    assert book(c, A, SLOT0 - 900, SLOT0).status_code == 422               # past
    assert book(c, A, SLOT0, SLOT0 + H, cell="cell-z").status_code == 404
    r = book(c, A, SLOT0, SLOT0 + H)                                       # the current slot is bookable
    assert r.status_code == 201 and r.json()["status"] == "confirmed"
    assert book(c, B, SLOT0 + 1800, SLOT0 + 2 * H).status_code == 409      # overlap, any user
    assert book(c, A, SLOT0 + H - 900, SLOT0 + H + 900).status_code == 409
    assert book(c, B, SLOT0 + H, SLOT0 + 2 * H).status_code == 201         # back-to-back is fine

    ev = db.all("SELECT * FROM meter_events WHERE kind='slot_minutes' ORDER BY ts, id")
    assert [(e["qty"], e["amount"]) for e in ev] == [(60, 120.0), (60, 120.0)]

    # renters only see their own bookings; availability marks taken slots
    assert len(c.get("/v1/bookings", headers=A).json()) == 1 and len(c.get("/v1/bookings", headers=OP).json()) == 2
    assert c.get(f"/v1/bookings/{r.json()['id']}", headers=B).status_code == 404
    av = c.get("/v1/cells/cell-a/availability?date=2027-01-15", headers=A).json()
    slots = {s["start_ts"]: s for s in av["slots"]}
    assert len(slots) == 96 and not slots[SLOT0]["free"] and slots[SLOT0]["mine"]
    assert not slots[SLOT0 + H]["free"] and not slots[SLOT0 + H]["mine"]
    assert slots[SLOT0 + 2 * H]["free"] and not slots[SLOT0 - 900]["free"]   # past slot


def test_cancel_refund_window(env):
    c, db, clock, A, B, OP = env
    far = book(c, A, SLOT0 + 48 * H, SLOT0 + 49 * H).json()       # > 24 h ahead: refunded
    near = book(c, A, SLOT0 + 2 * H, SLOT0 + 3 * H).json()        # < 24 h ahead: not refunded
    now = book(c, A, SLOT0, SLOT0 + 900).json()
    assert c.delete(f"/v1/bookings/{far['id']}", headers=B).status_code == 404
    r = c.delete(f"/v1/bookings/{far['id']}", headers=A)
    assert r.status_code == 200 and r.json()["refunded"] and r.json()["status"] == "cancelled"
    assert c.delete(f"/v1/bookings/{far['id']}", headers=A).status_code == 409
    assert c.delete(f"/v1/bookings/{near['id']}", headers=A).json()["refunded"] is False
    assert c.delete(f"/v1/bookings/{now['id']}", headers=A).status_code == 409     # already started
    refunds = db.all("SELECT * FROM meter_events WHERE kind='slot_refund'")
    assert [(e["qty"], e["amount"], e["booking_id"]) for e in refunds] == [(-60, -120.0, far["id"])]
    assert book(c, B, SLOT0 + 48 * H, SLOT0 + 49 * H).status_code == 201   # freed
    inv = c.get("/v1/invoice?month=2027-01", headers=A).json()
    assert inv["org"] == "acme" and inv["total"] == 15 * 2 + 60 * 2    # far refunded, near + now billed
