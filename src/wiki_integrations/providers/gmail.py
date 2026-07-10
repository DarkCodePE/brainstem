"""
Gmail `OAuthIntegrationSource` — per [PRD-005](../../../docs/PRD-005-oauth-integrations-layer.md) MVP scope.

Pulls messages via a `ComposioBridge.walk("gmail")` iterator and translates
each into a `wiki_core.IngestEvent`. Translation rules:

- ``source`` = ``"gmail"``.
- ``path_or_uri`` = ``"mailto:<from>"`` so the event's URI is the
  canonical reply-to handle (Gmail message ids are surfaced in metadata
  for deeper drilldown).
- ``sha256`` = ``sha256(body)`` so dedup at the MemoryStore level catches
  repeated forwards of the same thread.
- ``metadata`` carries the keys required by `wiki_ingest.adapter._to_storage`:
  ``bucket`` (= ``"gmail-inbox"``), ``rel_path`` (the message id, used as
  the on-disk-equivalent locator), ``event_type`` (= ``"created"``),
  ``mtime`` (Gmail's ``internal_date``), ``size`` (``len(body.encode())``),
  optional ``mime`` (= ``"text/plain"``).

The class is **transport-agnostic** at the unit-test seam: pass any object
with ``walk("gmail") -> AsyncIterator[dict]`` and the provider will work.
The default constructor wires a `ComposioBridge`.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from wiki_integrations.base import OAuthIntegrationSource

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from wiki_core.protocols import IngestEvent
    from wiki_integrations.cursor_store import CursorStore

log = logging.getLogger("wiki_integrations.providers.gmail")


class _Walker(Protocol):
    """Structural type for the `walk` surface we depend on.

    Anything with a single async-iterator-returning `walk(provider)` method
    works. Concretely it's a `ComposioBridge`, but tests pin a fake.
    """

    def walk(self, provider: str) -> AsyncIterator[dict[str, Any]]: ...


@dataclass(frozen=True, slots=True)
class GmailItem:
    """Normalised Gmail message used inside the provider.

    Frozen + slotted so the in-batch list can be iterated without surprise
    mutation. The dataclass is internal; the public surface is
    `IngestEvent`.
    """

    message_id: str
    thread_id: str
    sender: str
    subject: str
    body: str
    internal_date: str
    metadata: dict[str, Any] = field(default_factory=dict)


class GmailIntegrationSource(OAuthIntegrationSource):
    """Gmail provider implementation."""

    PROVIDER = "gmail"
    BUCKET = "gmail-inbox"
    DEFAULT_FETCH_WINDOW = timedelta(hours=24)

    def __init__(
        self,
        *,
        on_event: Callable[[IngestEvent], Awaitable[None]],
        walker: _Walker,
        fetch_window: timedelta | None = None,
        cursor_store: CursorStore | None = None,
    ) -> None:
        super().__init__(
            self.PROVIDER,
            fetch_window=fetch_window or self.DEFAULT_FETCH_WINDOW,
            on_event=on_event,
        )
        self._walker = walker
        self._cursor_store = cursor_store
        self._cursor: str | None = None

    async def fetch_batch(self) -> list[IngestEvent]:
        """Pull a window's worth of Gmail messages and emit one event each.

        Returns the list so tests can assert on the shape without
        subscribing to `on_event`. Real callers wire both — the list comes
        back for the polling worker's telemetry, the events flow through
        the callback into the MemoryStore.

        Cursor handling (OQ-2, PRD-006 FR-3): if a `CursorStore` was
        passed at construction time, the provider reads its persisted
        cursor before walking and writes the new cursor after a
        successful walk. The cursor for Gmail is the highest
        ``internal_date`` seen in the batch — Composio's Gmail action
        accepts that as a ``newer_than`` boundary. When no
        `CursorStore` is wired, the provider works exactly as before —
        every tick pulls the configured window with no persistence.
        """
        # Read back the persisted cursor on first call after start().
        if self._cursor_store is not None and self._cursor is None:
            self._cursor = await self._cursor_store.get(self.PROVIDER)

        events: list[IngestEvent] = []
        latest_internal_date: str | None = None
        async for raw in self._walker.walk(self.PROVIDER):
            try:
                item = _normalise(raw)
            except (KeyError, TypeError) as exc:
                # A single malformed payload must not poison the batch.
                # Log and skip; the polling worker's next pass picks it
                # up if Composio's response stabilises.
                log.warning(
                    "gmail.malformed_payload",
                    extra={"extra_fields": {"error_class": type(exc).__name__}},
                )
                continue
            event = _to_event(item)
            events.append(event)
            await self.emit(event)
            if latest_internal_date is None or item.internal_date > latest_internal_date:
                latest_internal_date = item.internal_date

        # Persist the new high-water mark when the walk produced events
        # AND a cursor store is wired.
        if self._cursor_store is not None and latest_internal_date is not None:
            self._cursor = latest_internal_date
            await self._cursor_store.set(self.PROVIDER, latest_internal_date)
        return events

    @property
    def cursor(self) -> str | None:
        """Last persisted cursor value, or None if nothing fetched yet.

        Exposed for tests and operational queries — production callers
        should go through the `CursorStore` directly.
        """
        return self._cursor


# --------------------------------------------------------------------------- #
# Translation helpers                                                         #
# --------------------------------------------------------------------------- #


def _normalise(raw: dict[str, Any]) -> GmailItem:
    """Normalise a raw Composio payload to the internal `GmailItem`.

    Composio's Gmail action returns ``id``, ``thread_id``, ``from``,
    ``subject``, ``snippet``, ``body``, ``internal_date``. We accept those
    and tolerate the slightly-different shapes that appear in tests
    (``message_id`` for ``id``, ``sender`` for ``from``).
    """
    message_id = str(raw.get("message_id") or raw["id"])
    sender = str(raw.get("sender") or raw.get("from", "unknown@unknown"))
    return GmailItem(
        message_id=message_id,
        thread_id=str(raw.get("thread_id", message_id)),
        sender=sender,
        subject=str(raw.get("subject", "")),
        body=str(raw.get("body") or raw.get("snippet", "")),
        internal_date=str(raw.get("internal_date") or _utcnow_iso()),
        metadata={k: v for k, v in raw.items() if k not in {"body"}},
    )


def _to_event(item: GmailItem) -> IngestEvent:
    """Translate a normalised `GmailItem` into a `wiki_core.IngestEvent`.

    All metadata keys required by `wiki_ingest.adapter._to_storage` are
    populated so the event survives the bridge into the SQLite queue.
    """
    # Lazy import keeps the provider module free of a hard `wiki_core`
    # import at class definition time (only fetch_batch needs it).
    from wiki_core.protocols import IngestEvent

    body_bytes = item.body.encode("utf-8")
    sha = hashlib.sha256(body_bytes).hexdigest()
    return IngestEvent(
        event_id=str(uuid4()),
        source=GmailIntegrationSource.PROVIDER,
        path_or_uri=f"mailto:{item.sender}",
        sha256=sha,
        received_at=datetime.now(UTC),
        metadata={
            "bucket": GmailIntegrationSource.BUCKET,
            "rel_path": item.message_id,
            "event_type": "created",
            "mtime": item.internal_date,
            "size": len(body_bytes),
            "mime": "text/plain",
            "thread_id": item.thread_id,
            "subject": item.subject,
        },
    )


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
