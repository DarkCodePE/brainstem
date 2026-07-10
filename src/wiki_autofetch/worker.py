"""
End-to-end auto-fetch worker.

Composes:

- `AutoFetchScheduler` — periodic polling of `IngestSource`s.
- `TokenBucket` — rate-limit gate (shared budget with file-drop ingest
  per PRD-006 FR-7; the bucket lives here so multiple sources cooperate).
- `wiki_core.MemoryStore` — sink that receives `IngestEvent`s and
  dedups them via `sha_seen` (PRD-006 FR-6 idempotency).

Per-source errors are isolated (PRD-006 US-002): one Gmail 500 does not
stop GitHub from ticking. Errors are captured into `AutoFetchMetrics`.

Usage
-----

    store = await open_memory_store(db_path)
    worker = AutoFetchWorker(sources=[gmail, github], store=store)
    await worker.start()           # background tick loop
    ...
    snap = worker.metrics.to_dict()  # for Prometheus textfile
    await worker.stop()

A single out-of-band tick is exposed via `worker.tick_once()`
(`wiki-agent autofetch trigger`, PRD-006 US-006).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from wiki_autofetch.metrics import AutoFetchMetrics
from wiki_autofetch.rate_limiter import TokenBucket
from wiki_autofetch.scheduler import (
    AutoFetchResult,
    AutoFetchScheduler,
    FetchCallable,
)
from wiki_core.protocols import IngestEvent, IngestSource, MemoryStore

log = logging.getLogger("wiki_autofetch.worker")


class AutoFetchWorker:
    """Composes scheduler + rate-limiter + MemoryStore sink.

    Parameters
    ----------
    sources:
        Sequence of `IngestSource`s to poll.
    store:
        `wiki_core.MemoryStore` sink. Events are enqueued via `enqueue`;
        duplicates (per `sha_seen`) are silently skipped to honour
        PRD-006 FR-6.
    interval_seconds, jitter_seconds:
        Forwarded to `AutoFetchScheduler`.
    rate_limiter:
        Token bucket guarding `store.enqueue` calls. Optional — if not
        supplied, a generous in-memory bucket is created (60/min).
    fetcher:
        Override the per-source fetch strategy (mainly for tests).
    metrics:
        Optional shared metrics object. One is created if not supplied.
    rate_limit_wait_seconds:
        Maximum time to wait for rate-limiter tokens before recording a
        `rate_limited` outcome and moving on. 0 disables waiting
        entirely (drop on first miss). Default 1.0s.
    """

    def __init__(
        self,
        *,
        sources: Sequence[IngestSource],
        store: MemoryStore,
        interval_seconds: int = 1200,
        jitter_seconds: int = 30,
        rate_limiter: TokenBucket | None = None,
        fetcher: FetchCallable | None = None,
        metrics: AutoFetchMetrics | None = None,
        rate_limit_wait_seconds: float = 1.0,
    ) -> None:
        if rate_limit_wait_seconds < 0:
            raise ValueError("rate_limit_wait_seconds must be >= 0")
        self._store = store
        self._bucket = (
            rate_limiter
            if rate_limiter is not None
            else TokenBucket(capacity=10, refill_per_second=10 / 60)
        )
        self._metrics = metrics if metrics is not None else AutoFetchMetrics()
        self._rate_limit_wait = float(rate_limit_wait_seconds)

        self._scheduler = AutoFetchScheduler(
            sources,
            interval_seconds=interval_seconds,
            jitter_seconds=jitter_seconds,
            fetcher=fetcher,
            metrics=self._metrics,
            on_events=self._on_events,
        )

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #

    @property
    def metrics(self) -> AutoFetchMetrics:
        return self._metrics

    @property
    def rate_limiter(self) -> TokenBucket:
        return self._bucket

    @property
    def is_running(self) -> bool:
        return self._scheduler.is_running

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        await self._scheduler.start()

    async def stop(self) -> None:
        await self._scheduler.stop()

    async def tick_once(self) -> AutoFetchResult:
        """Drive a single tick. Used by manual trigger and tests."""
        return await self._scheduler.tick_once()

    # ------------------------------------------------------------------ #
    # Sink                                                               #
    # ------------------------------------------------------------------ #

    async def _on_events(self, source: IngestSource, events: Sequence[IngestEvent]) -> None:
        """Sink invoked by the scheduler with the events from one source.

        Walks each event, dedups via `sha_seen`, optionally waits on the
        token bucket, and `enqueue`s the survivor. A single failing
        event records to metrics but does not abort the batch for the
        rest of the source's events (and per-source isolation in the
        scheduler ensures other sources keep going either way).
        """
        name = _source_name(source)
        for event in events:
            # PRD-006 FR-6: SHA idempotency at the worker boundary.
            try:
                if await self._store.sha_seen(event.sha256):
                    continue
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "autofetch.sha_seen_failed",
                    extra={
                        "extra_fields": {
                            "source": name,
                            "error_class": type(e).__name__,
                        }
                    },
                )
                # If the dedup check itself failed, fall through and try
                # to enqueue; the store can reject if it has its own
                # uniqueness constraint.

            # PRD-006 FR-7: rate-limit before pushing.
            if not await self._acquire_token():
                self._metrics.record_source_rate_limited(name)
                log.info(
                    "autofetch.rate_limited",
                    extra={"extra_fields": {"source": name, "event_id": event.event_id}},
                )
                continue

            try:
                await self._store.enqueue(event)
            except Exception as e:  # noqa: BLE001
                err_class = type(e).__name__
                self._metrics.record_source_error(
                    name, error_class=f"enqueue:{err_class}", duration_seconds=0.0
                )
                log.warning(
                    "autofetch.enqueue_failed",
                    extra={
                        "extra_fields": {
                            "source": name,
                            "event_id": event.event_id,
                            "error_class": err_class,
                        }
                    },
                )

    async def _acquire_token(self) -> bool:
        """Try to take one token, optionally waiting up to
        ``rate_limit_wait_seconds`` for refill. Returns False if the
        deadline elapses without success.

        Uses the `wiki_core.RateLimiter.try_acquire` non-blocking path
        rather than the blocking `acquire`: the worker prefers to record
        a `rate_limited` outcome and move on (PRD-006 FR-7) rather than
        stall the tick loop on an exhausted bucket.
        """
        if self._bucket.try_acquire(1):
            return True
        if self._rate_limit_wait <= 0:
            return False
        wait = min(self._bucket.time_until_available(1), self._rate_limit_wait)
        if wait <= 0:
            return self._bucket.try_acquire(1)
        await asyncio.sleep(wait)
        return self._bucket.try_acquire(1)


def _source_name(source: IngestSource) -> str:
    name_fn = getattr(source, "name", None)
    if callable(name_fn):
        try:
            return str(name_fn())
        except Exception:  # noqa: BLE001
            return type(source).__name__
    return type(source).__name__
