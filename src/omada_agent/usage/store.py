"""A small SQLite ledger: one row per question answered.

SQLite is enough here (a few writes per minute at most), needs no server, and survives restarts, so
budgets keep counting after the portal is restarted. Every call opens its own short-lived connection,
which makes the store safe to use from the web server's threads.

The ledger never stores the text of questions or answers, only sizes and totals.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT NOT NULL,
    day           TEXT NOT NULL,
    month         TEXT NOT NULL,
    user          TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL    NOT NULL DEFAULT 0,
    priced        INTEGER NOT NULL DEFAULT 1,
    tool_calls    INTEGER NOT NULL DEFAULT 0,
    rounds        INTEGER NOT NULL DEFAULT 0,
    elapsed_ms    INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'ok',
    error_code    TEXT,
    request_id    TEXT,
    question_chars INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_day_user ON usage(day, user);
CREATE INDEX IF NOT EXISTS idx_usage_month ON usage(month);
"""


def resolve_timezone(name: str) -> ZoneInfo | timezone:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc


class UsageStore:
    def __init__(self, path: Path, tz_name: str = "UTC"):
        self.path = Path(path)
        self.tz_name = tz_name
        self.tz = resolve_timezone(tz_name)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ time helpers
    def now(self) -> datetime:
        return datetime.now(self.tz)

    def day_key(self, when: datetime | None = None) -> str:
        return (when or self.now()).strftime("%Y-%m-%d")

    def month_key(self, when: datetime | None = None) -> str:
        return (when or self.now()).strftime("%Y-%m")

    # ------------------------------------------------------------------ writes
    def record(
        self,
        *,
        user: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float | None = 0.0,
        tool_calls: int = 0,
        rounds: int = 0,
        elapsed_ms: int = 0,
        status: str = "ok",
        error_code: str | None = None,
        request_id: str | None = None,
        question_chars: int = 0,
        when: datetime | None = None,
    ) -> None:
        moment = when or self.now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO usage (ts_utc, day, month, user, model, input_tokens, output_tokens, cost_usd, priced,"
                " tool_calls, rounds, elapsed_ms, status, error_code, request_id, question_chars)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    moment.astimezone(timezone.utc).isoformat(timespec="seconds"),
                    self.day_key(moment),
                    self.month_key(moment),
                    user,
                    model,
                    int(input_tokens),
                    int(output_tokens),
                    float(cost_usd or 0.0),
                    0 if cost_usd is None else 1,
                    int(tool_calls),
                    int(rounds),
                    int(elapsed_ms),
                    status,
                    error_code,
                    request_id,
                    int(question_chars),
                ),
            )

    # ------------------------------------------------------------------ reads
    def spent(self, *, day: str | None = None, month: str | None = None, user: str | None = None) -> dict:
        """Cost and number of questions, filtered by day and/or month and/or user."""
        where, params = [], []
        for column, value in (("day", day), ("month", month), ("user", user)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(value)
        sql = "SELECT COALESCE(SUM(cost_usd),0) c, COUNT(*) n, COALESCE(SUM(input_tokens),0) i, COALESCE(SUM(output_tokens),0) o FROM usage"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._conn() as conn:
            row = conn.execute(sql, params).fetchone()
        return {"cost": row["c"], "requests": row["n"], "input_tokens": row["i"], "output_tokens": row["o"]}

    def unpriced_requests(self, month: str) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM usage WHERE month = ? AND priced = 0", (month,)).fetchone()[0]

    def by_day(self, days: int = 30, user: str | None = None) -> list[dict]:
        start = self.day_key(self.now() - timedelta(days=days - 1))
        sql = (
            "SELECT day, COUNT(*) requests, SUM(cost_usd) cost, SUM(input_tokens) input_tokens,"
            " SUM(output_tokens) output_tokens FROM usage WHERE day >= ?"
        )
        params: list = [start]
        if user:
            sql += " AND user = ?"
            params.append(user)
        sql += " GROUP BY day ORDER BY day"
        with self._conn() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def top_users(self, days: int = 30, limit: int = 10) -> list[dict]:
        start = self.day_key(self.now() - timedelta(days=days - 1))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT user, COUNT(*) requests, SUM(cost_usd) cost, SUM(input_tokens + output_tokens) tokens,"
                " MAX(ts_utc) last_seen FROM usage WHERE day >= ? GROUP BY user ORDER BY cost DESC, requests DESC LIMIT ?",
                (start, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def by_status(self, days: int = 30) -> list[dict]:
        start = self.day_key(self.now() - timedelta(days=days - 1))
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) requests FROM usage WHERE day >= ? GROUP BY status ORDER BY requests DESC",
                (start,),
            ).fetchall()
        return [dict(r) for r in rows]

    def averages(self, days: int = 30) -> dict:
        start = self.day_key(self.now() - timedelta(days=days - 1))
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) n, AVG(cost_usd) cost, AVG(elapsed_ms) ms, AVG(tool_calls) tools,"
                " AVG(input_tokens) tin, AVG(output_tokens) tout FROM usage WHERE day >= ? AND status = 'ok'",
                (start,),
            ).fetchone()
        return {k: (row[k] or 0) for k in row.keys()}
