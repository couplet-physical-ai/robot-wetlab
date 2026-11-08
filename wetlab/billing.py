"""Metering. Every billable thing becomes a MeterEvent; the SQLite ledger is the source of truth,
Stripe (Billing Meter Events) is an optional mirror. Rates come from lab.billing (USD)."""
from __future__ import annotations

import calendar
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

from .spec import Billing as Rates

log = logging.getLogger("wetlab.billing")

# kind -> (unit, lab.billing rate field)
KINDS = {
    "slot_minutes": ("min", "slot_minute"),
    "slot_refund": ("min", "slot_minute"),     # qty < 0
    "sim_dry_run": ("run", "sim_dry_run"),
    "isaac_minutes": ("min", "isaac_minute"),
    "arm_seconds": ("s", "arm_second"),
}


@dataclass
class MeterEvent:
    kind: str
    qty: float
    unit: str
    unit_price: float
    amount: float
    user_id: str
    org: str
    booking_id: str | None = None
    job_id: str | None = None
    ts: float = 0.0


class BillingSink(Protocol):
    def emit(self, ev: MeterEvent) -> None: ...


class LedgerSink:
    """Always on: the meter_events table in the service DB."""

    def __init__(self, db):
        self.db = db

    def emit(self, ev: MeterEvent) -> None:
        self.db.insert("meter_events", **asdict(ev))


class StripeMeterSink:
    """Mirror to Stripe Billing Meter Events. Disabled unless STRIPE_API_KEY is set; fire-and-forget
    in a thread (never on the runtime tick). One Stripe meter per kind: event_name = wetlab_<kind>.
    shortcut: negative refunds are skipped (Stripe meters take non-negative values); refund via a
    credit note until meter_event_adjustments are wired."""

    URL = "https://api.stripe.com/v1/billing/meter_events"

    def __init__(self, customer_for: Callable[[str], str | None], api_key: str | None = None):
        self.api_key = api_key or os.environ.get("STRIPE_API_KEY")
        self.customer_for = customer_for

    def emit(self, ev: MeterEvent) -> None:
        cust = self.customer_for(ev.user_id)
        if not self.api_key or not cust or ev.qty <= 0:
            return
        body = urllib.parse.urlencode({
            "event_name": f"wetlab_{ev.kind}",
            "payload[stripe_customer_id]": cust,
            "payload[value]": str(round(ev.qty)),
            "identifier": f"wetlab:{ev.kind}:{ev.job_id or ev.booking_id}",   # idempotency
            "timestamp": int(ev.ts),
        }).encode()
        req = urllib.request.Request(self.URL, body, {"Authorization": f"Bearer {self.api_key}"})
        threading.Thread(target=self._post, args=(req,), daemon=True).start()

    @staticmethod
    def _post(req) -> None:
        try:
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:  # the ledger still has it; reconcile later
            log.warning("stripe meter event failed: %s", e)


class Billing:
    def __init__(self, sinks: list[BillingSink]):
        self.sinks = sinks

    def meter(self, kind: str, qty: float, user: dict, rates: Rates, booking_id: str | None = None,
              job_id: str | None = None, ts: float | None = None) -> MeterEvent:
        unit, field = KINDS[kind]
        price = getattr(rates, field)
        ev = MeterEvent(kind, qty, unit, price, round(qty * price, 4), user["id"], user["org"], booking_id, job_id,
                        time.time() if ts is None else ts)
        for s in self.sinks:
            s.emit(ev)
        return ev


def month_range(month: str) -> tuple[int, int]:
    """'YYYY-MM' -> [start, end) UTC epoch seconds."""
    y, m = (int(x) for x in month.split("-"))
    if not 1 <= m <= 12:
        raise ValueError("month must be YYYY-MM")
    start = calendar.timegm((y, m, 1, 0, 0, 0))
    return start, calendar.timegm((y + m // 12, m % 12 + 1, 1, 0, 0, 0))


def invoice(db, org: str, month: str) -> dict:
    a, b = month_range(month)
    lines = db.all("SELECT kind, unit, SUM(qty) AS qty, SUM(amount) AS amount, COUNT(*) AS events FROM meter_events "
                   "WHERE org=? AND ts>=? AND ts<? GROUP BY kind, unit ORDER BY kind", org, a, b)
    return {"org": org, "month": month, "currency": "usd", "lines": lines,
            "total": round(sum(ln["amount"] for ln in lines), 2)}
