"""
Protocol bridge between `wiki_ingest` storage records and the
`wiki_core.MemoryStore` domain contract.

`EventQueue` (canonical SQLite WAL queue, src/wiki_ingest/queue.py) operates
on `wiki_ingest.models.IngestEvent` — a row-shaped dataclass with storage
columns (path, rel_path, bucket, event_type, mtime, size, attempts, status,
…). `wiki_core.IngestEvent` is a slimmer domain event with `source`,
`path_or_uri`, `received_at`, and a `metadata` dict.

`SqliteMemoryStore` is the bridge: it wraps `EventQueue` and presents the
`MemoryStore` shape. Translation happens at the I/O boundary, so the rest
of the harness (subagents, middlewares, future Memory Tree consumers per
[PRD-004](../../docs/PRD-004-memory-tree.md)) can write against
`wiki_core.MemoryStore` without knowing about SQLite columns.

This is the first piece of M2 Sprint 1 (issue #21 protocols alignment).
Without it, M2 Memory Tree v1 would land on a wobbly substrate (see the
M1 retro §"Risks for M2"). After it lands, the rebuilt
`tests/wiki_ingest/` suite can re-assert on a stable contract instead of
the brownfield `IngestQueue` alias.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from wiki_core.protocols import IngestEvent as DomainEvent

from wiki_ingest.models import IngestEvent as StorageEvent
from wiki_ingest.queue import EventQueue


def _to_domain(record: StorageEvent) -> DomainEvent:
    """Storage row → domain event. Loses storage-only fields (attempts,
    status, etc.); the caller doesn't need them at the MemoryStore surface."""
    # Local import to keep wiki_ingest free of an import-time dep on wiki_core
    # for code that doesn't use the bridge.
    from wiki_core.protocols import IngestEvent

    received = _parse_iso(record.enqueued_at)
    return IngestEvent(
        event_id=record.event_id,
        source=f"watcher:{record.bucket}",
        path_or_uri=record.path,
        sha256=record.sha256 or "",
        received_at=received,
        metadata={
            "rel_path": record.rel_path,
            "bucket": record.bucket,
            "event_type": record.event_type,
            "mtime": record.mtime,
            "size": record.size,
            "mime": record.mime,
        },
    )


def _to_storage(event: DomainEvent) -> StorageEvent:
    """Domain event → storage row. Required metadata keys must be present;
    raises KeyError otherwise so callers fail loudly rather than silently
    inserting a malformed row.

    Required metadata keys: `rel_path`, `bucket`, `event_type`, `mtime`, `size`.
    Optional: `mime`.
    """
    md = event.metadata
    return StorageEvent(
        event_id=event.event_id,
        path=event.path_or_uri,
        rel_path=str(md["rel_path"]),
        bucket=str(md["bucket"]),
        event_type=str(md["event_type"]),
        mtime=str(md["mtime"]),
        size=int(md["size"]),
        sha256=event.sha256 or None,
        mime=md.get("mime"),  # type: ignore[arg-type]
        enqueued_at=_format_iso(event.received_at),
    )


def _parse_iso(s: str) -> datetime:
    """Parse the daemon's `_utcnow_iso` format (seconds + 'Z')."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _format_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class SqliteMemoryStore:
    """`wiki_core.MemoryStore`-shaped adapter over `EventQueue`.

    Construction is light; `init()` performs the schema migration and
    must be awaited before any other method. Caller owns lifecycle.

    The class deliberately does NOT inherit from `Protocol` — Python
    structural protocols are duck-typed at the call site via
    `isinstance(x, MemoryStore)` when the protocol is `@runtime_checkable`.
    """

    def __init__(self, db_path: Path) -> None:
        self._inner = EventQueue(db_path)
        # event_id → (sha256, rel_path) lookup; needed because mark_done at
        # the protocol surface includes "mark this content as seen" semantics
        # while the storage layer keeps `events` and `ingested` tables apart.
        # Bounded by queue depth; cleared on terminal transitions.
        self._sha_index: dict[str, tuple[str, str]] = {}

    async def init(self) -> None:
        """Run schema migration. Idempotent."""
        await self._inner.init()

    async def close(self) -> None:
        await self._inner.close()

    async def enqueue(self, event: DomainEvent) -> str:
        record = _to_storage(event)
        rid = await self._inner.enqueue(record)
        # Remember enough to satisfy the MemoryStore contract on mark_done.
        if record.sha256:
            self._sha_index[rid] = (record.sha256, record.rel_path)
        return rid

    async def claim_next(self) -> DomainEvent | None:
        record = await self._inner.claim_next()
        if record is None:
            return None
        return _to_domain(record)

    async def mark_done(self, event_id: str, page_path: str | None) -> None:
        await self._inner.mark_done(event_id, page_path)
        # MemoryStore contract: post-mark_done, sha_seen(content_sha) is True.
        # The inner stores sha in a separate `ingested` table that isn't
        # populated by mark_done alone; we backfill it here.
        cached = self._sha_index.pop(event_id, None)
        if cached is not None:
            sha, rel_path = cached
            await self._inner.record_ingested(sha, rel_path, page_path)

    async def mark_failed(self, event_id: str, err: str) -> None:
        await self._inner.mark_failed(event_id, err)
        self._sha_index.pop(event_id, None)

    async def sha_seen(self, sha256: str) -> bool:
        return await self._inner.sha_seen(sha256)

    async def queue_depth(self) -> int:
        return await self._inner.queue_depth()

    async def recover_stuck(self) -> int:
        return await self._inner.recover_stuck()


def store_from_path(db_path: Path | str) -> SqliteMemoryStore:
    """Convenience constructor used by `wiki_ingest.daemon` wiring."""
    return SqliteMemoryStore(Path(db_path))


# Re-export the inner concrete type for callers that need storage-layer
# operations not exposed by the protocol (mark_retry, mark_skipped). These
# are deliberately NOT on the MemoryStore protocol because they're
# implementation choices, not domain concepts.
__all__ = ["SqliteMemoryStore", "store_from_path", "EventQueue", "cast"]
