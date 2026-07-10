"""
SQLite-backed per-source cursor persistence.

Per [PRD-006 FR-3](../../docs/PRD-006-autofetch-worker.md) and the OQ-2
resolution in [SPEC-007](../../docs/SPEC-007-autofetch-scheduler.md), each
`OAuthIntegrationSource` (Gmail, GitHub, …) gets its own opaque cursor
string. The cursor is whatever shape the provider needs to resume the
walk on its next tick — Gmail's ``historyId``, GitHub's
``since=<iso8601>``, etc. The auto-fetch substrate doesn't interpret it;
it just persists it so a daemon restart doesn't replay the entire
window.

The store is intentionally minimal. The decision (OQ-2 in SPEC-007) was
to keep `OAuthIntegrationSource.fetch_batch()`'s return type as
``list[IngestEvent]`` rather than the ``(items, next_cursor, has_more)``
tuple PRD-006 sketched. Cursor management is therefore a **separate
concern**: the source reads its cursor at the start of `fetch_batch`,
calls the upstream API with it, and persists the new cursor at the end.
The auto-fetch worker doesn't see the cursor at all.

Table shape (PRD-006 FR-3)::

    CREATE TABLE autofetch_cursors (
        source_name  TEXT PRIMARY KEY,
        cursor_value TEXT NOT NULL,
        updated_at   TEXT NOT NULL
    );

A single primary key on `source_name` enforces "one cursor per source";
`set` is an `INSERT OR REPLACE`, so writes are idempotent and there's
no race window between read-update-write under the SQLite WAL journal.

Concurrency
-----------
The store uses an internal `asyncio.Lock` around mutating operations so
two concurrent `set()` calls for the same source serialise on the
async side. SQLite's BEGIN IMMEDIATE inside each statement provides
write-side isolation, but the lock keeps the in-process queue ordering
predictable for tests that assert on the final value.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS autofetch_cursors (
    source_name  TEXT PRIMARY KEY,
    cursor_value TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""


def _utcnow_iso() -> str:
    """ISO-8601 UTC stamp suitable for the `updated_at` column."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class CursorStore:
    """SQLite-backed cursor persistence per PRD-006 FR-3.

    Table: ``autofetch_cursors(source_name PK, cursor_value, updated_at)``.

    Parameters
    ----------
    db_path:
        Path to the SQLite file. Parent directory is created on `init`
        so callers can pass a freshly-`tmp_path`'d location without
        pre-creating it.

    Notes
    -----
    `init()` MUST be called before any other method. Mirror the
    `EventQueue` lifecycle: explicit `init` + explicit `close`. Both are
    idempotent.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def init(self) -> None:
        """Open the connection and create the table if missing.

        Idempotent: a second call is a no-op (the existing connection
        is reused and `CREATE TABLE IF NOT EXISTS` is harmless).
        """
        if self._db is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.execute("PRAGMA busy_timeout=5000;")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        """Close the connection. Idempotent."""
        if self._db is None:
            return
        await self._db.close()
        self._db = None

    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("CursorStore not initialised; call init() first")
        return self._db

    # ------------------------------------------------------------------ #
    # CRUD                                                               #
    # ------------------------------------------------------------------ #

    async def get(self, source_name: str) -> str | None:
        """Return the cursor for `source_name`, or None if unset."""
        if not source_name:
            raise ValueError("source_name must be non-empty")
        db = self._conn()
        async with db.execute(
            "SELECT cursor_value FROM autofetch_cursors WHERE source_name=?",
            (source_name,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return str(row[0])

    async def set(self, source_name: str, cursor: str) -> None:
        """Persist `cursor` for `source_name`.

        Insert-or-replace, so a second call simply overwrites. Updates
        `updated_at` so operational queries can spot stale cursors.
        Serialised by an internal asyncio lock to keep concurrent writes
        deterministic.
        """
        if not source_name:
            raise ValueError("source_name must be non-empty")
        if cursor is None:
            raise ValueError("cursor must not be None — use clear() to remove")
        db = self._conn()
        async with self._lock:
            await db.execute(
                "INSERT OR REPLACE INTO autofetch_cursors "
                "(source_name, cursor_value, updated_at) VALUES (?, ?, ?)",
                (source_name, str(cursor), _utcnow_iso()),
            )
            await db.commit()

    async def clear(self, source_name: str) -> None:
        """Remove `source_name`'s cursor.

        No-op if the source has no stored cursor. Serialised through the
        same lock as `set` so a concurrent `set; clear` always lands in
        the order the caller awaited.
        """
        if not source_name:
            raise ValueError("source_name must be non-empty")
        db = self._conn()
        async with self._lock:
            await db.execute(
                "DELETE FROM autofetch_cursors WHERE source_name=?",
                (source_name,),
            )
            await db.commit()


__all__ = ["CursorStore"]
