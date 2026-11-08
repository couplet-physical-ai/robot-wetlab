import calendar
import time

import pytest

from wetlab.billing import Billing, LedgerSink, StripeMeterSink, invoice, month_range
from wetlab.service.db import DB
from wetlab.spec import Billing as Rates

RATES = Rates(slot_minute=2.0, arm_second=0.01, sim_dry_run=0.25, isaac_minute=0.10, cancel_refund_hours=24)
JAN = calendar.timegm((2027, 1, 15, 8, 0, 0))


def test_meter_ledger_and_invoice(tmp_path, monkeypatch):
    monkeypatch.delenv("STRIPE_API_KEY", raising=False)
    db = DB(tmp_path / "w.db")
    a, _ = db.add_user("acme", "alice")
    b, _ = db.add_user("acme", "bob")
    c, _ = db.add_user("other", "carol")
    posted = []
    stripe = StripeMeterSink(lambda uid: "cus_123")
    monkeypatch.setattr(StripeMeterSink, "_post", staticmethod(posted.append))
    bill = Billing([LedgerSink(db), stripe])
    ev = bill.meter("slot_minutes", 60, a, RATES, booking_id="bk_1", ts=JAN)
    assert (ev.unit, ev.unit_price, ev.amount) == ("min", 2.0, 120.0)
    bill.meter("slot_refund", -60, a, RATES, booking_id="bk_1", ts=JAN + 1)
    bill.meter("sim_dry_run", 1, b, RATES, job_id="job_1", ts=JAN + 2)
    bill.meter("arm_seconds", 31.34, b, RATES, job_id="job_1", ts=JAN + 3)
    bill.meter("isaac_minutes", 2.5, b, RATES, job_id="job_1", ts=JAN + 4)
    bill.meter("slot_minutes", 15, c, RATES, booking_id="bk_2", ts=JAN)
    bill.meter("slot_minutes", 15, a, RATES, booking_id="bk_3", ts=calendar.timegm((2027, 2, 1, 0, 0, 0)))  # next month
    with pytest.raises(KeyError):
        bill.meter("free_lunch", 1, a, RATES)
    assert posted == []   # Stripe sink is off without STRIPE_API_KEY

    rows = db.all("SELECT * FROM meter_events")
    assert len(rows) == 7 and {r["kind"] for r in rows} == {"slot_minutes", "slot_refund", "sim_dry_run", "arm_seconds", "isaac_minutes"}
    inv = invoice(db, "acme", "2027-01")
    lines = {ln["kind"]: ln for ln in inv["lines"]}
    assert lines["slot_minutes"]["amount"] == 120.0 and lines["slot_refund"]["amount"] == -120.0
    assert lines["arm_seconds"]["amount"] == pytest.approx(0.3134)
    assert inv["total"] == pytest.approx(0.25 + 0.3134 + 0.25, abs=0.01)
    assert invoice(db, "other", "2027-01")["total"] == 30.0
    assert invoice(db, "acme", "2027-02")["total"] == 30.0


def test_stripe_payload_shape(monkeypatch):
    posted = []
    monkeypatch.setattr(StripeMeterSink, "_post", staticmethod(posted.append))
    sink = StripeMeterSink(lambda uid: "cus_9", api_key="sk_test_x")
    bill = Billing([sink])
    bill.meter("slot_refund", -15, {"id": "u_1", "org": "acme"}, RATES, booking_id="bk_1", ts=JAN)   # negative: skipped
    bill.meter("arm_seconds", 12.4, {"id": "u_1", "org": "acme"}, RATES, job_id="job_7", ts=JAN)
    for _ in range(100):   # posted from a background thread
        if posted:
            break
        time.sleep(0.01)
    assert len(posted) == 1
    req = posted[0]
    assert req.full_url == "https://api.stripe.com/v1/billing/meter_events"
    body = req.data.decode()
    assert "event_name=wetlab_arm_seconds" in body and "payload%5Bstripe_customer_id%5D=cus_9" in body
    assert "payload%5Bvalue%5D=12" in body and "identifier=wetlab%3Aarm_seconds%3Ajob_7" in body
    assert req.headers["Authorization"] == "Bearer sk_test_x"


def test_month_range():
    a, b = month_range("2026-12")
    assert b - a == 31 * 86400 and a == calendar.timegm((2026, 12, 1, 0, 0, 0))
    with pytest.raises(ValueError):
        month_range("2026-13")
