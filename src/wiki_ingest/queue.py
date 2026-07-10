from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from wiki_ingest.models import IngestEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    rel_path     TEXT NOT NULL,
    bucket       TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    mtime        TEXT NOT NULL,
    size         INTEGER NOT NULL,
    sha256       TEXT,
    mime         TEXT,
    enqueued_at  TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'pending',
    last_error   TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    page_path    TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_status    ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_enqueued  ON events(enqueued_at);
CREATE INDEX IF NOT EXISTS idx_events_path      ON events(path);

CREATE TABLE IF NOT EXISTS ingested (
    sha256       TEXT PRIMARY KEY,
    rel_path     TEXT NOT NULL,
    page_path    TEXT,
    ingested_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rate_limit (
    key          TEXT PRIMARY KEY,
    window_start TEXT NOT NULL,
    tokens_used  INTEGER NOT NULL DEFAULT 0
);
"""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class EventQueue:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.execute("PRAGMA busy_timeout=5000;")
        await self._db.executescript(_SCHEMA)
        await self._migrate(self._db)
        await self._db.commit()

    @staticmethod
    async def _migrate(db: aiosqlite.Connection) -> None:
        """In-place migrations for legacy databases.

        ADR-035 D1: `events.page_path` records the wiki page each done
        event produced. Legacy DBs (created before the column existed in
        `_SCHEMA`) get it via ALTER TABLE — existing rows keep NULL,
        which the recovery script (`scripts/adr-035-recover-ingest-state.py`)
        reports; new `mark_done` calls must provide a real value.
        """
        async with db.execute("PRAGMA table_info(events)") as cur:
            cols = {row[1] for row in await cur.fetchall()}
        if "page_path" not in cols:
            await db.execute("ALTER TABLE events ADD COLUMN page_path TEXT")

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("EventQueue not initialised; call init() first")
        return self._db

    async def enqueue(self, event: IngestEvent) -> str:
        db = self._conn()
        await db.execute(
            "INSERT OR REPLACE INTO events "
            "(event_id, path, rel_path, bucket, event_type, mtime, size, sha256, mime, "
            " enqueued_at, attempts, status, last_error, started_at, finished_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            event.to_row(),
        )
        await db.commit()
        return event.event_id

    async def claim_next(self) -> IngestEvent | None:
        db = self._conn()
        async with db.execute(
            "SELECT event_id, path, rel_path, bucket, event_type, mtime, size, sha256, mime, "
            "enqueued_at, attempts, status, last_error, started_at, finished_at "
            "FROM events WHERE status='pending' ORDER BY enqueued_at ASC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None

        event = IngestEvent.from_row(row)
        now = _utcnow()
        await db.execute(
            "UPDATE events SET status='processing', started_at=?, attempts=attempts+1 "
            "WHERE event_id=? AND status='pending'",
            (now, event.event_id),
        )
        await db.commit()

        async with db.execute(
            "SELECT status, started_at, attempts FROM events WHERE event_id=?",
            (event.event_id,),
        ) as cur:
            latest = await cur.fetchone()
        if latest is None or latest[0] != "processing":
            return None
        event.status = "processing"
        event.started_at = latest[1]
        event.attempts = latest[2]
        return event

    async def mark_done(self, event_id: str, page_path: str | None) -> None:
        """Mark an event done, recording the wiki page it produced.

        ADR-035 D1: a done event without a page is the original
        `page_path=NULL` bug (the daemon claimed success without writing
        anything). Refuse it loudly instead of persisting a lie.
        """
        if not page_path:
            raise ValueError(f"mark_done requires a non-empty page_path (event_id={event_id})")
        db = self._conn()
        now = _utcnow()
        await db.execute(
            "UPDATE events SET status='done', finished_at=?, last_error=NULL, page_path=? "
            "WHERE event_id=?",
            (now, page_path, event_id),
        )
        await db.commit()

    async def get_page_path(self, event_id: str) -> str | None:
        """Return the recorded page_path for an event (None if unset)."""
        db = self._conn()
        async with db.execute("SELECT page_path FROM events WHERE event_id=?", (event_id,)) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def mark_failed(self, event_id: str, err: str) -> None:
        db = self._conn()
        now = _utcnow()
        await db.execute(
            "UPDATE events SET status='failed', finished_at=?, last_error=? WHERE event_id=?",
            (now, err[:2000], event_id),
        )
        await db.commit()

    async def mark_retry(self, event_id: str, err: str) -> None:
        db = self._conn()
        await db.execute(
            "UPDATE events SET status='pending', last_error=? WHERE event_id=?",
            (err[:2000], event_id),
        )
        await db.commit()

    async def mark_skipped(self, event_id: str, reason: str) -> None:
        db = self._conn()
        now = _utcnow()
        await db.execute(
            "UPDATE events SET status='skipped', finished_at=?, last_error=? WHERE event_id=?",
            (now, reason[:2000], event_id),
        )
        await db.commit()

    async def recover_stuck(self) -> int:
        db = self._conn()
        cur = await db.execute(
            "UPDATE events SET status='pending', started_at=NULL WHERE status='processing'"
        )
        await db.commit()
        return cur.rowcount if cur.rowcount is not None else 0

    async def sha_seen(self, sha256: str) -> bool:
        db = self._conn()
        async with db.execute("SELECT 1 FROM ingested WHERE sha256=?", (sha256,)) as cur:
            return (await cur.fetchone()) is not None

    async def record_ingested(self, sha256: str, rel_path: str, page_path: str | None) -> None:
        if not page_path:
            raise ValueError(f"record_ingested requires a non-empty page_path (sha256={sha256})")
        db = self._conn()
        await db.execute(
            "INSERT OR REPLACE INTO ingested (sha256, rel_path, page_path, ingested_at) "
            "VALUES (?,?,?,?)",
            (sha256, rel_path, page_path, _utcnow()),
        )
        await db.commit()

    async def queue_depth(self) -> int:
        db = self._conn()
        async with db.execute("SELECT COUNT(*) FROM events WHERE status='pending'") as cur:
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def lag_p(self, q: float) -> float:
        if not 0 < q <= 1:
            raise ValueError("q must be in (0, 1]")
        db = self._conn()
        async with db.execute(
            "SELECT enqueued_at, finished_at FROM events "
            "WHERE status='done' AND finished_at IS NOT NULL "
            "ORDER BY finished_at DESC LIMIT 500"
        ) as cur:
            rows = await cur.fetchall()
        if not rows:
            return 0.0

        def _parse(ts: str) -> float:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()

        lags = sorted((_parse(r[1]) - _parse(r[0])) for r in rows)
        idx = min(len(lags) - 1, int(round(q * (len(lags) - 1))))
        return max(0.0, lags[idx])

    async def counts_by_status(self) -> dict[str, int]:
        db = self._conn()
        async with db.execute("SELECT status, COUNT(*) FROM events GROUP BY status") as cur:
            rows = await cur.fetchall()
        return {r[0]: int(r[1]) for r in rows}

    # --- SEC-08: persistent rate-limit bucket ------------------------------

    async def rate_limit_consume(self, key: str, window_start: str, limit: int) -> tuple[bool, int]:
        """Try to consume one token under logical window `window_start`.

        Returns (ok, tokens_used_after). If `ok` is False, the caller must
        wait until the next window. Uses BEGIN IMMEDIATE to avoid races
        across crash-restart cycles (SEC-08).
        """
        db = self._conn()
        await db.execute("BEGIN IMMEDIATE")
        try:
            async with db.execute(
                "SELECT window_start, tokens_used FROM rate_limit WHERE key=?",
                (key,),
            ) as cur:
                row = await cur.fetchone()
            if row is None or row[0] != window_start:
                tokens_after = 1
                await db.execute(
                    "INSERT OR REPLACE INTO rate_limit(key, window_start, tokens_used) "
                    "VALUES (?,?,?)",
                    (key, window_start, tokens_after),
                )
                await db.commit()
                return True, tokens_after
            tokens_used = int(row[1])
            if tokens_used >= limit:
                await db.rollback()
                return False, tokens_used
            tokens_after = tokens_used + 1
            await db.execute(
                "UPDATE rate_limit SET tokens_used=? WHERE key=?",
                (tokens_after, key),
            )
            await db.commit()
            return True, tokens_after
        except Exception:
            await db.rollback()
            raise

    async def rate_limit_peek(self, key: str, window_start: str) -> int:
        db = self._conn()
        async with db.execute(
            "SELECT window_start, tokens_used FROM rate_limit WHERE key=?",
            (key,),
        ) as cur:
            row = await cur.fetchone()
        if row is None or row[0] != window_start:
            return 0
        return int(row[1])


# Backward-compat alias — tests in tests/wiki_ingest/ import IngestQueue.
# Canonical name is EventQueue; this alias is kept until the tests migrate
# (tracked in issue #21 — protocols.py introduction).
IngestQueue = EventQueue
