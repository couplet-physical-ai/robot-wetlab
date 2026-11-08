"""SQLite store (stdlib sqlite3, WAL, one file). Times are UTC epoch seconds."""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id TEXT PRIMARY KEY, org TEXT NOT NULL, name TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('renter','operator','admin')),
  api_key_sha256 TEXT NOT NULL UNIQUE, stripe_customer_id TEXT);
CREATE TABLE IF NOT EXISTS bookings(
  id TEXT PRIMARY KEY, cell TEXT NOT NULL, user_id TEXT NOT NULL REFERENCES users(id),
  start_ts REAL NOT NULL, end_ts REAL NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('confirmed','cancelled')), created_ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS bookings_cell_time ON bookings(cell, start_ts);
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, cell TEXT NOT NULL, booking_id TEXT, user_id TEXT NOT NULL REFERENCES users(id),
  kind TEXT NOT NULL, mode TEXT NOT NULL CHECK (mode IN ('dry_run','run')), source TEXT, ir_json TEXT,
  status TEXT NOT NULL, plan_summary_json TEXT, error TEXT, created_ts REAL NOT NULL,
  started_ts REAL, finished_ts REAL, artifacts_dir TEXT, max_deviation_rad REAL, isaac_status TEXT);
CREATE INDEX IF NOT EXISTS jobs_cell_status ON jobs(cell, status);
CREATE TABLE IF NOT EXISTS meter_events(
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, qty REAL NOT NULL, unit TEXT NOT NULL, unit_price REAL NOT NULL,
  amount REAL NOT NULL, user_id TEXT NOT NULL, org TEXT NOT NULL, booking_id TEXT, job_id TEXT, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS meter_org_ts ON meter_events(org, ts);
CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, ts REAL NOT NULL, actor TEXT, action TEXT NOT NULL, data_json TEXT);
"""


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


class DB:
    """One shared connection guarded by an RLock (the runtime thread and API threads share it)."""

    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def all(self, sql: str, *args) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql: str, *args) -> dict | None:
        rows = self.all(sql, *args)
        return rows[0] if rows else None

    def exec(self, sql: str, *args) -> int:
        with self.lock:
            return self.conn.execute(sql, args).lastrowid

    def insert(self, table: str, **row) -> int:
        cols = ", ".join(row)
        return self.exec(f"INSERT INTO {table} ({cols}) VALUES ({', '.join('?' * len(row))})", *row.values())

    def update(self, table: str, id_: str, **fields) -> None:
        self.exec(f"UPDATE {table} SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?", *fields.values(), id_)

    @contextmanager
    def tx(self):
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")

    # -- users / audit -------------------------------------------------------------
    def add_user(self, org: str, name: str, role: str = "renter", stripe_customer_id: str | None = None) -> tuple[dict, str]:
        """-> (user row, api key). Only the key's sha256 is stored; show the key once."""
        key = "wl_" + secrets.token_urlsafe(24)
        uid = new_id("u")
        self.insert("users", id=uid, org=org, name=name, role=role, api_key_sha256=sha256(key), stripe_customer_id=stripe_customer_id)
        return self.user(uid), key

    def user(self, uid: str) -> dict | None:
        return self.one("SELECT id, org, name, role, stripe_customer_id FROM users WHERE id=?", uid)

    def user_by_key(self, key: str) -> dict | None:
        return self.one("SELECT id, org, name, role, stripe_customer_id FROM users WHERE api_key_sha256=?", sha256(key))

    def audit(self, actor: str | None, action: str, ts: float | None = None, **data) -> None:
        self.insert("audit", ts=time.time() if ts is None else ts, actor=actor, action=action, data_json=json.dumps(data, default=str))

    def active_booking(self, cell: str, now: float) -> dict | None:
        return self.one("SELECT * FROM bookings WHERE cell=? AND status='confirmed' AND start_ts<=? AND end_ts>? LIMIT 1", cell, now, now)
