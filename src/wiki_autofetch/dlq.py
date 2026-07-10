"""
SQLite-backed dead-letter queue for failed auto-fetch tick attempts.

Per issue #38 AC: "Per-provider checkpoint advanced only on success;
on failure → DLQ entry + backoff."

The DLQ is **not** the cursor store. The cursor store advances only on
success; the DLQ records each failure with timestamp + error class so:

1. The next tick can apply exponential backoff per provider
2. `sbw doctor` can surface "Gmail has 3 failures, last at <ts>"
3. The operator can clear an entry once they know the underlying issue
   is fixed (e.g. provider 503 has resolved)

Schema::

    CREATE TABLE autofetch_dlq (
        source_name   TEXT NOT NULL,
        ts            TEXT NOT NULL,          -- ISO-8601 UTC
        error_class   TEXT NOT NULL,
        error_detail  TEXT NOT NULL,
        attempt       INTEGER NOT NULL,       -- consecutive failure count
        PRIMARY KEY (source_name, ts)
    );

The (source_name, ts) PK lets multiple failures coexist; `attempt` lets
the backoff logic compute "wait 2 ** attempt minutes before retrying".
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS autofetch_dlq (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name   TEXT NOT NULL,
    ts            TEXT NOT NULL,
    error_class   TEXT NOT NULL,
    error_detail  TEXT NOT NULL,
    attempt       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dlq_source ON autofetch_dlq(source_name);
CREATE INDEX IF NOT EXISTS idx_dlq_source_ts ON autofetch_dlq(source_name, ts);
"""

DEFAULT_DB_PATH = Path.home() / ".sbw" / "state" / "autofetch_dlq.db"
"""Where the DLQ lives. Distinct from the cursor store so a corrupted
DLQ doesn't take down the polling loop."""

MAX_BACKOFF_MINUTES = 60
"""Cap exponential backoff at 1 hour — beyond that, polling resumes at
the normal cadence and operator attention is required (`sbw doctor`)."""


@dataclass(frozen=True)
class DLQEntry:
    source_name: str
    ts: str
    error_class: str
    error_detail: str
    attempt: int


class AutoFetchDLQ:
    """Synchronous SQLite DLQ used by the one-shot tick.

    The tick is not async (it's a systemd-driven CLI invocation) so the
    DLQ uses plain `sqlite3` rather than `aiosqlite` — one fewer dep on
    the hot path.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ #
    # CRUD                                                               #
    # ------------------------------------------------------------------ #

    def record_failure(self, source_name: str, error_class: str, detail: str = "") -> int:
        """Append a failure for `source_name` and return the new attempt count.

        ``attempt`` is the count of consecutive failures since the last
        success. ``record_success`` resets it to zero.
        """
        if not source_name:
            raise ValueError("source_name must be non-empty")
        cur_attempt = self.attempt_count(source_name)
        next_attempt = cur_attempt + 1
        self._conn.execute(
            "INSERT INTO autofetch_dlq "
            "(source_name, ts, error_class, error_detail, attempt) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_name, _utcnow_iso(), error_class, detail, next_attempt),
        )
        self._conn.commit()
        return next_attempt

    def record_success(self, source_name: str) -> int:
        """Mark a successful tick. Clears the DLQ rows for this source.

        Returns the number of rows cleared (useful for telemetry).
        """
        if not source_name:
            raise ValueError("source_name must be non-empty")
        cur = self._conn.execute("DELETE FROM autofetch_dlq WHERE source_name=?", (source_name,))
        self._conn.commit()
        return int(cur.rowcount)

    def attempt_count(self, source_name: str) -> int:
        """Number of consecutive failures since the last success.

        Returns 0 if no failures recorded (i.e., either fresh or last
        attempt succeeded).
        """
        row = self._conn.execute(
            "SELECT COALESCE(MAX(attempt), 0) FROM autofetch_dlq WHERE source_name=?",
            (source_name,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def backoff_until(self, source_name: str) -> datetime | None:
        """When the next tick for `source_name` is allowed, given current failures.

        Exponential backoff: ``2 ** attempt`` minutes, capped at
        ``MAX_BACKOFF_MINUTES``. Returns ``None`` if the source has no
        active failure history (poll immediately).
        """
        attempt = self.attempt_count(source_name)
        if attempt <= 0:
            return None
        minutes = min(2**attempt, MAX_BACKOFF_MINUTES)
        last_ts = self._last_failure_ts(source_name)
        if last_ts is None:
            return None
        return last_ts + timedelta(minutes=minutes)

    def is_backed_off(self, source_name: str, now: datetime | None = None) -> bool:
        """``True`` iff backoff is active right now."""
        until = self.backoff_until(source_name)
        if until is None:
            return False
        ref = now if now is not None else datetime.now(UTC)
        return ref < until

    def list_failures(self, source_name: str | None = None) -> list[DLQEntry]:
        """Recent failures (most recent first), optionally filtered by source."""
        if source_name is None:
            rows = self._conn.execute(
                "SELECT source_name, ts, error_class, error_detail, attempt "
                "FROM autofetch_dlq ORDER BY id DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT source_name, ts, error_class, error_detail, attempt "
                "FROM autofetch_dlq WHERE source_name=? ORDER BY id DESC",
                (source_name,),
            ).fetchall()
        return [
            DLQEntry(
                source_name=r[0],
                ts=r[1],
                error_class=r[2],
                error_detail=r[3],
                attempt=r[4],
            )
            for r in rows
        ]

    def clear(self, source_name: str | None = None) -> int:
        """Manual operator clear. Returns number of rows removed."""
        if source_name is None:
            cur = self._conn.execute("DELETE FROM autofetch_dlq")
        else:
            cur = self._conn.execute(
                "DELETE FROM autofetch_dlq WHERE source_name=?", (source_name,)
            )
        self._conn.commit()
        return int(cur.rowcount)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _last_failure_ts(self, source_name: str) -> datetime | None:
        row = self._conn.execute(
            "SELECT ts FROM autofetch_dlq WHERE source_name=? ORDER BY id DESC LIMIT 1",  # latest by insertion
            (source_name,),
        ).fetchone()
        if row is None:
            return None
        try:
            return datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
        except ValueError:
            return None


def _utcnow_iso() -> str:
    # ms precision keeps two failures inside the same second from
    # colliding on the (source_name, ts) primary key.
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


__all__ = ["AutoFetchDLQ", "DEFAULT_DB_PATH", "DLQEntry", "MAX_BACKOFF_MINUTES"]
