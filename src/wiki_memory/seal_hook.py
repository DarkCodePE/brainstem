"""
Post-ingest seal hook — auto-seal every successfully ingested source.

M3 Sprint 4 wire-in: when the unified ingest daemon writes a new wiki
page, we want the Memory Tree to immediately chunk the body into
`content_store`, register a `tree_source` node, and schedule a seal
job. Today the daemon's worker calls `write_page` over MCP and stops.
This hook bridges that gap.

The hook is **fire-and-forget** by design: a seal call that hangs on a
slow LLM must not block the ingest worker. The post-write callback
schedules the seal as a background task and returns immediately. If
the seal fails the daemon logs it — ingest still succeeds.

## Usage

```python
hook = build_default_seal_hook(content_store, tree_store, write_sink)
await worker.run(post_write_hook=hook)
```

Or compose it with the existing worker wiring:

```python
async def on_page_written(event: IngestEvent, page_path: str) -> None:
    await hook(event, page_path)
```

## Why a hook, not a method on the worker

The worker doesn't know about `wiki_memory`; it only knows about
`wiki_core.MemoryStore` and the MCP write surface. Pushing the seal
into the worker would couple the ingest substrate to the memory tree.
Instead we expose a generic `OnPageWrittenCallback` Protocol that the
worker accepts as a kwarg; this module ships the default
implementation, but tests and alternate deployments can plug in
something else.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from wiki_memory.chunker import chunk_page
from wiki_memory.content_store import ContentStore
from wiki_memory.seal_worker import SealError, SealWorker
from wiki_memory.summariser_factory import build_default_summariser
from wiki_memory.tree_nodes import TreeNodeStore

if TYPE_CHECKING:
    from wiki_core.protocols import IngestEvent, WriteSink

log = logging.getLogger(__name__)


@runtime_checkable
class OnPageWrittenCallback(Protocol):
    """Protocol for the post-write callback the ingest worker invokes
    after a successful `write_page`."""

    async def __call__(self, event: IngestEvent, page_path: str) -> None: ...


def _source_id_from_event(event: IngestEvent) -> str:
    """Derive a stable source_id from an ingest event.

    The event's `sha256` is the canonical content fingerprint, but two
    different ingests of the same content from different sources should
    still be distinguishable. We combine `source:sha256` as the source
    id so the same content under two providers (e.g. a doc that appears
    in Gmail *and* Drive) gets two tree_source nodes; the duplicate sha
    in `content_store` is silently skipped (idempotent inserts).
    """
    payload = f"{event.source}:{event.sha256}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_key_from_event(event: IngestEvent) -> str:
    """Stable *logical* identity of a source for temporal supersession
    (ADR-028 #158).

    Keyed on ``path_or_uri`` — the document's location — NOT ``event.source``
    (which is a coarse provider label like ``"watcher:raw/"`` shared by
    every file from that provider; keying on it would group unrelated
    documents and supersede them all on each ingest). Re-ingesting the
    same path with changed content keeps the same ``source_key`` (so the
    prior version is superseded) while producing a new content-addressed
    ``source_id``.
    """
    return hashlib.sha256(event.path_or_uri.encode("utf-8")).hexdigest()


class SealOnIngestHook:
    """Post-write callback: chunk + index + schedule seal.

    Fire-and-forget: the seal task is spawned via `asyncio.create_task`
    and the callback returns once chunks + the tree node have landed.
    The seal itself runs in the background and logs failures.

    Constructor wiring (DI all the way down):
      - content_store / tree_store / write_sink — passed straight to
        the SealWorker.
      - read_page — async callable `(page_path) -> str` that returns
        the page body to chunk. Defaults to a filesystem-direct reader
        rooted at `vault_root`. Tests inject a stub.
      - summariser_factory — defaults to `build_default_summariser`
        (env-driven: OpenRouter primary per M3-S4, Anthropic secondary,
        Ollama final). Tests inject `lambda: NullSummariser()` for
        determinism.
      - schedule — defaults to `asyncio.create_task`. Tests inject a
        synchronous runner so the seal completes before the test
        assertion runs.
    """

    def __init__(
        self,
        *,
        content_store: ContentStore,
        tree_store: TreeNodeStore,
        write_sink: WriteSink,
        vault_root: Path,
        read_page: Callable[[str], Awaitable[str]] | None = None,
        summariser_factory: Callable[[], object] | None = None,
        schedule: Callable[[Awaitable[None]], object] | None = None,
        enable_seal: bool = True,
    ) -> None:
        self._content = content_store
        self._tree = tree_store
        self._write = write_sink
        self._vault_root = vault_root
        self._read_page = read_page or self._default_read_page
        self._summariser_factory = summariser_factory or build_default_summariser
        self._schedule = schedule or asyncio.create_task
        self._enable_seal = enable_seal

    async def __call__(self, event: IngestEvent, page_path: str) -> None:
        """Worker calls this after a successful `write_page`."""
        source_id = _source_id_from_event(event)
        try:
            body = await self._read_page(page_path)
        except FileNotFoundError:
            log.warning(
                "seal_hook.page_missing event_id=%s page_path=%s",
                event.event_id,
                page_path,
            )
            return
        except OSError as e:
            log.warning(
                "seal_hook.read_failed event_id=%s page_path=%s err=%s",
                event.event_id,
                page_path,
                type(e).__name__,
            )
            return

        chunks = chunk_page(body)
        if not chunks:
            log.info(
                "seal_hook.empty_body event_id=%s page_path=%s — no chunks, skipping seal",
                event.event_id,
                page_path,
            )
            return

        inserted = await self._content.insert_many(source_id=source_id, chunks=chunks)
        source_key = _source_key_from_event(event)
        event_time = event.received_at.isoformat() if event.received_at is not None else None
        await self._tree.create_source_node(
            node_id=source_id,
            source_key=source_key,
            event_time=event_time,
        )
        # ADR-028 #158: a re-ingest of the same logical source (same
        # path_or_uri) with changed content lands a new source_id; mark
        # any prior latest version superseded so stale chunks drop out of
        # default recall. No-op on first ingest of a source.
        superseded = await self._tree.supersede(source_key=source_key, new_node_id=source_id)
        log.info(
            "seal_hook.indexed event_id=%s source_id=%s chunks=%d new=%d superseded=%d",
            event.event_id,
            source_id[:12],
            len(chunks),
            inserted,
            len(superseded),
        )

        if not self._enable_seal:
            return

        # Fire-and-forget the seal. A slow or unavailable LLM must not
        # block the ingest worker.
        self._schedule(self._seal_in_background(source_id))

    async def _seal_in_background(self, source_id: str) -> None:
        worker = SealWorker(
            content_store=self._content,
            tree_store=self._tree,
            write_sink=self._write,
            summariser=self._summariser_factory(),  # type: ignore[arg-type]
        )
        try:
            result = await worker.seal_source(source_id=source_id, node_id=source_id)
            log.info(
                "seal_hook.sealed source_id=%s summary_sha=%s page=%s",
                source_id[:12],
                result.summary_sha256[:12],
                result.page_path,
            )
        except SealError as e:
            # Faithfulness gate refused or the summariser failed. The
            # tree node stays unsealed; a manual `memory.tree.seal_now`
            # can retry later. M3 Sprint 5 may add automatic backoff.
            log.warning(
                "seal_hook.seal_refused source_id=%s err=%s",
                source_id[:12],
                e,
            )
        except Exception:  # noqa: BLE001
            # Anything else (network, programming error) — log with the
            # traceback but never raise into the worker.
            log.exception("seal_hook.seal_crashed source_id=%s", source_id[:12])

    async def _default_read_page(self, page_path: str) -> str:
        """Read a wiki page from the vault root.

        Resolves `page_path` against `vault_root` and validates the
        result stays inside the root (SEC-01 path traversal mitigation
        mirrored from `wiki_ingest.security.validate_safe_path`).
        """
        resolved = (self._vault_root / page_path).resolve()
        try:
            resolved.relative_to(self._vault_root.resolve())
        except ValueError as e:
            raise OSError(f"page_path escapes vault root: {page_path!r}") from e
        return resolved.read_text(encoding="utf-8")


class NoopSealHook:
    """Drop-in replacement that does nothing. Useful for tests and
    deployments that want to disable the seal-on-ingest flow without
    swapping the worker wiring."""

    async def __call__(self, event: IngestEvent, page_path: str) -> None:
        return None


def build_default_seal_hook(
    *,
    content_store: ContentStore,
    tree_store: TreeNodeStore,
    write_sink: WriteSink,
    vault_root: Path,
    enable_seal: bool = True,
) -> SealOnIngestHook:
    """Production wiring: build a SealOnIngestHook with sensible
    defaults (filesystem page reader, env-driven summariser factory,
    asyncio.create_task scheduler).

    Pass `enable_seal=False` if you want the chunker + tree-node
    bookkeeping without spending LLM cycles on seals — useful for
    backfilling the index over an existing vault, or for the autofetch
    path on cheap providers."""
    return SealOnIngestHook(
        content_store=content_store,
        tree_store=tree_store,
        write_sink=write_sink,
        vault_root=vault_root,
        enable_seal=enable_seal,
    )


__all__ = [
    "NoopSealHook",
    "OnPageWrittenCallback",
    "SealOnIngestHook",
    "build_default_seal_hook",
]


# Avoid an unused-import warning while keeping `contextlib` available for
# future expansion (timeout wrappers around the seal task land in M3-S5).
_ = contextlib.suppress
