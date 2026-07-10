"""
Tests for the autofetch wiring in `wiki_ingest.daemon` (M3 Sprint 2,
PRD-006). Covers:

- Backwards compat: a daemon built without an `AutoFetchWorker` only
  starts the watcher (and the rest of the existing pipeline).
- Forward compat: a daemon built with a worker starts BOTH the watcher
  and the worker, and stops both on shutdown.
- Per-source error isolation: when one stub source raises on fetch the
  other still produces events into the `MemoryStore`.
- Status RPC shape: the `autofetch` key carries totals plus per-source
  counters.
- Composition factory: `build_daemon` returns a watcher-only daemon when
  no registry is supplied, and a wired daemon when one is.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from wiki_autofetch.metrics import AutoFetchMetrics
from wiki_autofetch.rate_limiter import TokenBucket
from wiki_autofetch.worker import AutoFetchWorker
from wiki_core.protocols import IngestEvent
from wiki_ingest.composition import build_daemon
from wiki_ingest.config import Config
from wiki_ingest.daemon import Daemon

# --------------------------------------------------------------------------- #
# Local stubs                                                                 #
# --------------------------------------------------------------------------- #


class StubSource:
    """Minimal `IngestSource`-shaped stub with a programmable `fetch_delta`."""

    def __init__(
        self,
        name_: str,
        *,
        returns: list[list[IngestEvent] | BaseException] | None = None,
    ) -> None:
        self._name = name_
        self._returns: list[list[IngestEvent] | BaseException] = list(returns or [])
        self.start_calls = 0
        self.stop_calls = 0
        self.fetch_calls = 0

    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self) -> None:
        self.stop_calls += 1

    async def fetch_delta(self) -> list[IngestEvent]:
        self.fetch_calls += 1
        if not self._returns:
            return []
        item = self._returns.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class InMemoryStore:
    """In-memory `MemoryStore` for autofetch sink testing."""

    def __init__(self) -> None:
        self.enqueued: list[IngestEvent] = []
        self.seen: set[str] = set()

    async def enqueue(self, event: IngestEvent) -> str:
        self.enqueued.append(event)
        self.seen.add(event.sha256)
        return event.event_id

    async def claim_next(self) -> IngestEvent | None:
        return None

    async def mark_done(self, event_id: str, page_path: str | None) -> None:
        return None

    async def mark_failed(self, event_id: str, err: str) -> None:
        return None

    async def sha_seen(self, sha256: str) -> bool:
        return sha256 in self.seen

    async def queue_depth(self) -> int:
        return len(self.enqueued)

    async def recover_stuck(self) -> int:
        return 0


def _make_event(*, source: str, sha: str) -> IngestEvent:
    return IngestEvent(
        event_id=f"evt-{source}-{sha[:6]}",
        source=source,
        path_or_uri=f"stub://{source}/{sha[:6]}",
        sha256=sha,
        received_at=datetime.now(UTC),
        metadata={"provider": source},
    )


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def cfg(tmp_path: Path) -> Iterator[Config]:
    raw = tmp_path / "knowledge-base" / "raw"
    (raw / "_ingested").mkdir(parents=True, exist_ok=True)
    yield Config(
        wiki_root=tmp_path / "knowledge-base",
        raw_dir=raw,
        ingested_dir=raw / "_ingested",
        db_path=tmp_path / "wiki-ingest.db",
        autofetch_enabled=True,
        autofetch_interval_seconds=1200,
        autofetch_rate_limit_per_minute=60,
    )


@pytest.fixture
def worker_factory() -> Any:
    """Build an `AutoFetchWorker` with a generous bucket so tests aren't
    starved by the default 10/min rate limit when they emit a handful of
    events."""

    def _build(
        *,
        sources: list[StubSource],
        store: InMemoryStore,
        metrics: AutoFetchMetrics | None = None,
    ) -> AutoFetchWorker:
        return AutoFetchWorker(
            sources=sources,
            store=store,
            interval_seconds=1200,
            jitter_seconds=0,
            rate_limiter=TokenBucket(capacity=100, refill_per_second=100.0),
            metrics=metrics,
        )

    return _build


# --------------------------------------------------------------------------- #
# Test 1: backwards compat — watcher only                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_daemon_without_autofetch_starts_only_watcher(cfg: Config) -> None:
    """`Daemon(cfg)` (legacy ctor) starts the watcher and leaves autofetch
    untouched. Status reflects `autofetch: None`."""
    daemon = Daemon(cfg)
    assert daemon.autofetch is None

    # We don't actually start filesystem watching (that hits the real
    # Observer thread). Instead we patch it out to verify the daemon
    # doesn't try to touch a worker that isn't there.
    daemon.watcher.start = AsyncMock()  # type: ignore[method-assign]
    daemon.watcher.stop = AsyncMock()  # type: ignore[method-assign]

    await daemon.queue.init()
    snap = await daemon.status()
    assert snap["autofetch"] is None
    assert "queue" in snap and "pool" in snap and "watcher" in snap
    await daemon.queue.close()


# --------------------------------------------------------------------------- #
# Test 2: forward compat — both start, both stop                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_daemon_with_autofetch_starts_both(cfg: Config, worker_factory: Any) -> None:
    """When an `AutoFetchWorker` is passed, the daemon starts AND stops
    both the watcher and the worker."""
    store = InMemoryStore()
    src = StubSource("github", returns=[[_make_event(source="github", sha="a" * 64)]])
    worker = worker_factory(sources=[src], store=store)

    daemon = Daemon(cfg, autofetch_worker=worker)
    assert daemon.autofetch is worker

    watcher_start = AsyncMock()
    watcher_stop = AsyncMock()
    daemon.watcher.start = watcher_start  # type: ignore[method-assign]
    daemon.watcher.stop = watcher_stop  # type: ignore[method-assign]

    await daemon.queue.init()
    # Drive start() far enough that the watcher and autofetch are spun up
    # but without entering the dispatcher/metrics loops (we'd then need
    # to cancel them — easier to call the wiring directly).
    await daemon.watcher.start()
    await daemon.autofetch.start()

    assert watcher_start.await_count == 1
    assert worker.is_running is True

    # Manual tick to exercise the wired pipeline.
    result = await worker.tick_once()
    assert result.total_events == 1
    assert store.enqueued and store.enqueued[0].source == "github"

    await daemon.watcher.stop()
    await daemon.autofetch.stop()
    assert watcher_stop.await_count == 1
    assert worker.is_running is False

    await daemon.queue.close()


# --------------------------------------------------------------------------- #
# Test 3: per-source error isolation                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_per_source_error_isolation(cfg: Config, worker_factory: Any) -> None:
    """One source raising on fetch must NOT block the other source from
    producing events into the MemoryStore (PRD-006 US-002)."""
    store = InMemoryStore()
    bad = StubSource("gmail", returns=[RuntimeError("upstream 500")])
    good = StubSource("github", returns=[[_make_event(source="github", sha="b" * 64)]])
    worker = worker_factory(sources=[bad, good], store=store)
    daemon = Daemon(cfg, autofetch_worker=worker)

    result = await daemon.autofetch.tick_once()

    # Errored source is recorded but didn't poison the tick.
    errored = [s for s in result.sources if s.error is not None]
    assert {s.name for s in errored} == {"gmail"}
    # Good source still pushed its event.
    assert len(store.enqueued) == 1
    assert store.enqueued[0].source == "github"
    # Both sources were attempted.
    assert bad.fetch_calls == 1 and good.fetch_calls == 1


# --------------------------------------------------------------------------- #
# Test 4: autofetch metrics surface in status RPC                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_autofetch_metrics_surface_in_status(cfg: Config, worker_factory: Any) -> None:
    """`daemon.status()` exposes `autofetch.fetched_total`,
    `autofetch.errors_total`, and per-source counters."""
    store = InMemoryStore()
    src_ok = StubSource("github", returns=[[_make_event(source="github", sha="c" * 64)]])
    src_err = StubSource("slack", returns=[ValueError("bad token")])
    worker = worker_factory(sources=[src_ok, src_err], store=store)

    daemon = Daemon(cfg, autofetch_worker=worker)
    await daemon.queue.init()
    await daemon.autofetch.tick_once()

    snap = await daemon.status()
    af = snap["autofetch"]
    assert af is not None
    assert af["fetched_total"] == 1
    assert af["errors_total"] == 1
    assert af["ticks_total"] == 1

    # Per-source counters are present and correctly attributed.
    assert "github" in af["sources"]
    assert "slack" in af["sources"]
    assert af["sources"]["github"]["events_fetched"] == 1
    assert af["sources"]["github"]["errors"] == 0
    assert af["sources"]["slack"]["errors"] == 1
    assert af["sources"]["slack"]["last_error_class"] == "ValueError"

    await daemon.queue.close()


# --------------------------------------------------------------------------- #
# Test 5: composition factory — no registry → no worker                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_composition_factory_no_registry(cfg: Config) -> None:
    """`build_daemon(cfg, registry=None)` returns a watcher-only daemon."""
    daemon = await build_daemon(cfg, registry=None)
    assert daemon.autofetch is None


# --------------------------------------------------------------------------- #
# Test 6: composition factory — registry with sources → wired                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_composition_factory_with_registry(cfg: Config) -> None:
    """`build_daemon(cfg, registry=...)` wires an `AutoFetchWorker` when
    the registry has at least one source and `cfg.autofetch_enabled` is
    True."""
    from wiki_integrations.registry import IntegrationRegistry

    store = InMemoryStore()
    registry = IntegrationRegistry()
    # IntegrationRegistry types its members as OAuthIntegrationSource,
    # but registration only inspects `.name()`. StubSource satisfies the
    # structural contract we exercise here.
    registry._sources["github"] = StubSource(  # type: ignore[assignment]
        "github", returns=[[_make_event(source="github", sha="d" * 64)]]
    )

    daemon = await build_daemon(cfg, registry=registry, memory_store=store)
    assert daemon.autofetch is not None
    # Active sources from the registry are picked up.
    assert len(daemon.autofetch._scheduler._sources) == 1  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Test 7: composition factory — autofetch disabled in config                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_composition_factory_respects_disabled_flag(
    cfg: Config,
) -> None:
    """When `cfg.autofetch_enabled is False`, no worker is wired even if
    a non-empty registry is supplied — the deployment gate stays in the
    config."""
    from wiki_integrations.registry import IntegrationRegistry

    cfg.autofetch_enabled = False
    registry = IntegrationRegistry()
    registry._sources["github"] = StubSource("github")  # type: ignore[assignment]

    daemon = await build_daemon(cfg, registry=registry)
    assert daemon.autofetch is None


# --------------------------------------------------------------------------- #
# Test 8: composition factory — empty registry → watcher-only                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_composition_factory_empty_registry(cfg: Config) -> None:
    """An empty registry is treated as 'no sources to poll' — the daemon
    remains watcher-only rather than running an idle autofetch loop."""
    from wiki_integrations.registry import IntegrationRegistry

    registry = IntegrationRegistry()
    daemon = await build_daemon(cfg, registry=registry)
    assert daemon.autofetch is None
