"""Hàng chờ video lưu trong SQLite (1 file cơ sở dữ liệu)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from .paths import DB_PATH

_lock = threading.RLock()


def _conn():
    c = sqlite3.connect(str(DB_PATH), timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("""CREATE TABLE IF NOT EXISTS jobs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input TEXT, title TEXT DEFAULT '', status TEXT DEFAULT 'queued', step TEXT DEFAULT '',
            frac REAL DEFAULT 0, note TEXT DEFAULT '', error TEXT DEFAULT '', attempts INTEGER DEFAULT 0,
            created REAL, updated REAL, started REAL, finished REAL, output TEXT DEFAULT '',
            timings TEXT DEFAULT '{}', issues TEXT DEFAULT '[]', not_before REAL DEFAULT 0,
            mode TEXT DEFAULT 'dub')""")
        cols = [r[1] for r in c.execute("PRAGMA table_info(jobs)").fetchall()]
        if "mode" not in cols:
            c.execute("ALTER TABLE jobs ADD COLUMN mode TEXT DEFAULT 'dub'")


def add(inp: str, mode: str = "dub") -> int:
    now = time.time()
    with _lock, _conn() as c:
        cur = c.execute("INSERT INTO jobs(input, created, updated, mode) VALUES(?,?,?,?)", (inp, now, now, mode))
        return cur.lastrowid


def exists_active(inp: str) -> bool:
    with _lock, _conn() as c:
        r = c.execute("SELECT 1 FROM jobs WHERE input=? AND status IN ('queued','running','ready')", (inp,)).fetchone()
        return r is not None


def update(job_id: int, **kw) -> None:
    if not kw:
        return
    kw["updated"] = time.time()
    for k in ("timings", "issues"):
        if k in kw and not isinstance(kw[k], str):
            kw[k] = json.dumps(kw[k], ensure_ascii=False)
    cols = ", ".join(f"{k}=?" for k in kw)
    with _lock, _conn() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*kw.values(), job_id))


def get(job_id: int) -> dict | None:
    with _lock, _conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(r) if r else None


def listing(limit: int = 300) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def next_job() -> dict | None:
    now = time.time()
    with _lock, _conn() as c:
        r = c.execute("SELECT * FROM jobs WHERE status='running' ORDER BY id LIMIT 1").fetchone()
        if not r:
            r = c.execute("SELECT * FROM jobs WHERE status='queued' AND not_before<=? ORDER BY id LIMIT 1",
                          (now,)).fetchone()
        return dict(r) if r else None


def candidates() -> list[dict]:
    """Video có thể xử lý: đang chạy dở, đang chờ (tới giờ thử lại), hoặc đã chuẩn bị xong."""
    now = time.time()
    with _lock, _conn() as c:
        rows = c.execute("SELECT * FROM jobs WHERE status IN ('running','ready') OR "
                         "(status='queued' AND not_before<=?) ORDER BY id", (now,)).fetchall()
        return [dict(r) for r in rows]


def reset_running() -> None:
    with _lock, _conn() as c:
        c.execute("UPDATE jobs SET status='queued', note='' WHERE status='running'")


def remove(job_id: int) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM jobs WHERE id=?", (job_id,))


def counts() -> dict:
    with _lock, _conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        out = {r["status"]: r["n"] for r in rows}
        day0 = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        r = c.execute("SELECT COUNT(*) n FROM jobs WHERE status IN ('done','review') AND finished>=?",
                      (day0,)).fetchone()
        out["today"] = r["n"]
        return out
