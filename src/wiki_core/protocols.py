"""
Typed Protocol contracts for the Second Brain Wiki harness.

These contracts capture the four boundaries the rest of the code crosses:

- `IngestSource` — where raw content arrives from (filesystem watcher today,
  OAuth integrations tomorrow per [PRD-005](../../docs/PRD-005-oauth-integrations-layer.md)).
- `MemoryStore` — durable storage for events, hashes, and (M2+) memory-tree
  nodes per [PRD-004](../../docs/PRD-004-memory-tree.md).
- `Search` — read-side: keyword + semantic over the vault.
- `WriteSink` — the boundary that emits canonical pages back into
  `knowledge-base/` (today through the `wiki-knowledge-engine` MCP server).

These are runtime-checkable structural types (`typing.Protocol`,
`runtime_checkable`). Implementations don't need to inherit; they just need
the right shape. That keeps the dependency arrow pointing inward: domain
modules import these contracts, infrastructure modules implement them.

Note (2026-05-22): The legacy `wiki_ingest.queue.EventQueue` and the
`wiki-knowledge-engine` MCP both predate this file. They will be migrated
to satisfy `MemoryStore` and `WriteSink` respectively as part of the
sprint-2 protocols alignment ([issue #21](https://github.com/DarkCodePE/second-brain-wiki/issues/21)).
This file declares the shape; the bridge work follows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# Value types                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class IngestEvent:
    """A single content arrival on the ingest boundary.

    Frozen + slotted: events flow through queues; immutability prevents
    accidental in-flight mutation by middleware.
    """

    event_id: str
    """ULID/UUIDv7 — chronologically sortable and idempotency-friendly."""

    source: str
    """Origin label: "watcher:raw/", "gmail", "slack", "github", "manual", ..."""

    path_or_uri: str
    """Filesystem path for local sources, URI for remote sources."""

    sha256: str
    """Content fingerprint. The MemoryStore uses this for dedup."""

    received_at: datetime
    """When the event was first observed at the boundary (UTC)."""

    metadata: dict[str, Any]
    """Source-specific extras: gmail message-id, slack channel, etc."""


@dataclass(frozen=True, slots=True)
class PageRef:
    """Locator for a wiki page."""

    page_path: str
    """Path relative to the vault root, e.g. ``sources/foo.md``."""

    category: Literal[
        "sources", "entities", "concepts", "answers", "synthesis", "outputs", "observations"
    ]


@dataclass(frozen=True, slots=True)
class Page:
    """A canonical wiki page: frontmatter + body."""

    ref: PageRef
    frontmatter: dict[str, Any]
    body: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A search result with provenance and scoring breakdown."""

    ref: PageRef
    title: str
    snippet: str
    score: float
    """Final fused score (0..1)."""
    score_components: dict[str, float]
    """e.g. {"keyword": 0.4, "semantic": 0.6} for [[ADR-005]] hybrid search."""


# --------------------------------------------------------------------------- #
# Boundary protocols                                                          #
# --------------------------------------------------------------------------- #


@runtime_checkable
class IngestSource(Protocol):
    """Where raw content enters the wiki.

    The current implementation is `src/wiki_ingest/watcher.py` (filesystem
    watchdog over `knowledge-base/raw/`). M3 (per PRD-005) adds OAuth-backed
    sources such as Gmail, Slack, GitHub. Every source emits `IngestEvent`s
    on the same queue.
    """

    async def start(self) -> None:
        """Begin observing the source. Idempotent."""
        ...

    async def stop(self) -> None:
        """Shut down cleanly; in-flight events are queued before returning."""
        ...

    def name(self) -> str:
        """Stable identifier for telemetry/logging."""
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Durable persistence for ingest state and (M2+) memory-tree projections.

    Today this is the SQLite WAL queue in `src/wiki_ingest/queue.py`
    (canonical name ``EventQueue``). M2 promotes the same store to host the
    `tree_source` / `tree_topic` / `tree_global` tables of PRD-004.
    """

    async def enqueue(self, event: IngestEvent) -> str:
        """Persist `event` and return its server-assigned id (== `event.event_id`)."""
        ...

    async def claim_next(self) -> IngestEvent | None:
        """Atomically pull the next pending event into `processing` state, or None."""
        ...

    async def mark_done(self, event_id: str, page_path: str | None) -> None:
        """Finalize a successful ingest, optionally linking to the produced page."""
        ...

    async def mark_failed(self, event_id: str, err: str) -> None:
        """Record a failure; up to the implementation to schedule retries."""
        ...

    async def sha_seen(self, sha256: str) -> bool:
        """Return True iff a content fingerprint has already been ingested."""
        ...

    async def queue_depth(self) -> int:
        """Number of events currently in `pending`."""
        ...

    async def recover_stuck(self) -> int:
        """Move events stuck in `processing` (post-crash) back to `pending`. Returns count moved."""
        ...


@runtime_checkable
class Search(Protocol):
    """Read-side over the vault.

    Today: hybrid keyword + fastembed semantic via `wiki_agent.tools`. M2:
    swap in PRD-004 memory-tree retrieval at the same surface.
    """

    async def search_index(
        self,
        query: str,
        *,
        limit: int = 10,
        categories: Sequence[str] | None = None,
    ) -> Sequence[SearchHit]: ...

    async def search_text(
        self,
        text: str,
        *,
        limit: int = 10,
        threshold: float | None = None,
    ) -> Sequence[SearchHit]:
        """Semantic-only path; used by `find_cross_references` style tools."""
        ...


@runtime_checkable
class WriteSink(Protocol):
    """Boundary that emits canonical pages back into the vault.

    Today: `wiki-knowledge-engine` MCP server (`write_page`). Direct
    filesystem writes are forbidden; everything goes through this surface
    so middleware (validation, lint, prompt-injection guard per
    [ADR-015](../../docs/ADR-015-prompt-injection-guard.md)) sees every
    edit.
    """

    async def write_page(
        self, page: Page, *, mode: Literal["create", "update", "upsert"] = "upsert"
    ) -> Path:
        """Persist `page` and return the on-disk path actually written."""
        ...

    async def append_to_log(self, entry: str) -> None:
        """Append a line to `log.md`. Operation history; never lossy."""
        ...


# --------------------------------------------------------------------------- #
# Source streaming helper                                                     #
# --------------------------------------------------------------------------- #


@runtime_checkable
class StreamingIngestSource(IngestSource, Protocol):
    """Optional capability: emit events as an async iterator instead of (or
    in addition to) pushing them to a queue. The Composio bridge in
    PRD-005 expects this shape so backpressure stays explicit.
    """

    def events(self) -> AsyncIterator[IngestEvent]: ...


# --------------------------------------------------------------------------- #
# Rate limiting                                                               #
# --------------------------------------------------------------------------- #


@runtime_checkable
class RateLimiter(Protocol):
    """Shared rate-limit contract for the auto-fetch + file-drop pipelines.

    Per [PRD-006 FR-7], the auto-fetch worker and the file-drop ingest
    worker cooperate on the same `mcp.write_page` budget (default 10/min).
    Today two concrete implementations live in the tree:

    - `wiki_autofetch.rate_limiter.TokenBucket` — in-process continuous
      refill bucket; deterministic under an injectable clock. Used by the
      auto-fetch loop where survival across restarts isn't required
      (cursors carry the durable state).
    - `wiki_ingest.worker.PersistentRateLimiter` — SQLite-backed minute
      window bucket. Used by the file-drop worker so a crash-loop restart
      cannot refill the window in memory.

    Both implementations satisfy this protocol structurally.

    The protocol exposes two acquisition paths so callers pick the
    semantics they want without reaching for limiter-specific knobs:

    - `acquire(n)` — async, blocks until `n` tokens are available. Use
      when the caller has nothing useful to do without the tokens (the
      file-drop worker's `write_page` is exactly this shape).
    - `try_acquire(n)` — sync, non-blocking. Returns True iff `n` tokens
      were granted, False otherwise. Use when the caller already has a
      fallback for the throttled case (drop the event, defer until next
      tick, record a `rate_limited` metric, etc — the auto-fetch worker
      is exactly this shape).

    Implementations MAY enforce `n >= 1`. Asking for `n > capacity`
    SHOULD be detected at the boundary: `try_acquire` returns False;
    `acquire` is implementation-defined (the in-memory variant raises,
    the persistent variant clamps).
    """

    async def acquire(self, n: int = 1) -> None:
        """Block until `n` tokens are available, then consume them.

        Implementations may sleep, raise, or grant immediately.
        """
        ...

    def try_acquire(self, n: int = 1) -> bool:
        """Non-blocking variant. Returns True iff `n` tokens were granted.

        Use when the caller already has a fallback (e.g. drop the work,
        defer until next tick).
        """
        ...
