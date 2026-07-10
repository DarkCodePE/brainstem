"""
Periodic auto-fetch scheduler.

The scheduler walks a sequence of `wiki_core.IngestSource` objects every
``interval_seconds`` (default 1200 = 20 minutes per PRD-006 FR-1) plus a
small random ``jitter_seconds`` so multiple deployments don't synchronise.

Per-source contract
-------------------
Sources expose the `IngestSource` shape (`start`, `stop`, `name`). To
actually emit events on a tick the scheduler needs a `FetchCallable`:

    async def fetch(source: IngestSource) -> Sequence[IngestEvent]:
        ...

The default fetcher tries `source.fetch_delta()` if present (the OAuth
integration base class in `wiki_integrations/` defines that) and falls
back to a no-op so legacy sources (the watchdog) keep working under the
same scheduler harness.

Per [PRD-006 US-002], one source raising an exception must not block the
others. Errors are captured per-source in `AutoFetchMetrics` and recorded
in the `AutoFetchResult` returned by `tick_once()`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from wiki_autofetch.metrics import AutoFetchMetrics
from wiki_core.protocols import IngestEvent, IngestSource

log = logging.getLogger("wiki_autofetch.scheduler")

#: Callable used to pull a batch of new events from a single source. The
#: scheduler awaits this for each active source on every tick.
FetchCallable = Callable[[IngestSource], Awaitable[Sequence[IngestEvent]]]


# Sentinel returned by the default fetcher when a source has nothing to
# pull (or has no `fetch_delta` method). Kept distinct from "empty list
# fetched successfully" so tests can assert on it if needed.
_EMPTY: list[IngestEvent] = []


async def _default_fetcher(source: IngestSource) -> Sequence[IngestEvent]:
    """Default fetch strategy: call `source.fetch_delta()` if exposed.

    The M3 `OAuthIntegrationSource` base class is expected to define
    ``fetch_delta`` (per PRD-006 FR-2). Sources without it — like the
    file-system watcher that pushes events through its own queue — get
    an empty list back, so the scheduler tick is a no-op for them.
    """
    fetch_delta = getattr(source, "fetch_delta", None)
    if fetch_delta is None:
        return _EMPTY
    result = await fetch_delta()
    # Allow either a flat sequence of events or the
    # ``(items, next_cursor, has_more)`` tuple PRD-006 FR-2 sketches.
    if isinstance(result, tuple):
        result = result[0]
    return list(result)


@dataclass(slots=True)
class SourceTickResult:
    """Outcome of running one source through one tick."""

    name: str
    events: int
    duration_seconds: float
    error: str | None = None


@dataclass(slots=True)
class AutoFetchResult:
    """Outcome of a single scheduler tick across all sources."""

    duration_seconds: float
    sources: list[SourceTickResult] = field(default_factory=list)

    @property
    def total_events(self) -> int:
        return sum(s.events for s in self.sources)

    @property
    def errors(self) -> list[SourceTickResult]:
        return [s for s in self.sources if s.error is not None]


class AutoFetchScheduler:
    """Asyncio-only periodic polling scheduler.

    Parameters
    ----------
    sources:
        Sequence of `IngestSource`-shaped objects to poll on each tick.
    interval_seconds:
        Base period between ticks. Default 1200s (20 min) per PRD-006 FR-1.
    jitter_seconds:
        Maximum random padding added to each wait so two deployments
        don't lockstep. Default 30s.
    fetcher:
        Override the per-source fetch strategy (see `FetchCallable`).
    metrics:
        Optional metrics object; one is created if not supplied.
    on_events:
        Optional async callback invoked with the events fetched from a
        source. The `AutoFetchWorker` wires the `MemoryStore.enqueue`
        sink here.
    rng:
        Injection point for the random source so tests can pin jitter.
    """

    def __init__(
        self,
        sources: Sequence[IngestSource],
        *,
        interval_seconds: int = 1200,
        jitter_seconds: int = 30,
        fetcher: FetchCallable | None = None,
        metrics: AutoFetchMetrics | None = None,
        on_events: Callable[[IngestSource, Sequence[IngestEvent]], Awaitable[None]] | None = None,
        rng: random.Random | None = None,
        loop_clock: Callable[[], float] | None = None,
    ) -> None:
        if interval_seconds < 1:
            raise ValueError("interval_seconds must be >= 1")
        if jitter_seconds < 0:
            raise ValueError("jitter_seconds must be >= 0")
        self._sources: list[IngestSource] = list(sources)
        self._interval = int(interval_seconds)
        self._jitter = int(jitter_seconds)
        self._fetcher: FetchCallable = fetcher or _default_fetcher
        self._metrics = metrics if metrics is not None else AutoFetchMetrics()
        self._on_events = on_events
        self._rng = rng if rng is not None else random.Random()
        self._loop_clock = loop_clock if loop_clock is not None else _asyncio_loop_clock

        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._started = False

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def metrics(self) -> AutoFetchMetrics:
        return self._metrics

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Spawn the background poll task. Idempotent."""
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="wiki-autofetch-scheduler")

    async def stop(self) -> None:
        """Cancel the background task and wait for it. Idempotent."""
        if not self._started:
            return
        self._started = False
        self._stop_event.set()
        task = self._task
        self._task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                # Cancellation is the happy path; any exception was logged
                # in `_run`. Don't propagate during shutdown.
                pass

    async def tick_once(self) -> AutoFetchResult:
        """Run a single iteration across all sources.

        Per-source errors are captured into the result; only a programming
        error in this method itself can propagate. Exposed so tests and
        the manual-trigger CLI (PRD-006 US-006) can drive a tick out-of-band.
        """
        tick_start = self._loop_clock()
        results: list[SourceTickResult] = []
        for source in self._sources:
            name = _source_name(source)
            source_start = self._loop_clock()
            try:
                events = await self._fetcher(source)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — PRD-006 US-002 isolation
                duration = self._loop_clock() - source_start
                err_class = type(e).__name__
                self._metrics.record_source_error(
                    name, error_class=err_class, duration_seconds=duration
                )
                log.warning(
                    "autofetch.source_error",
                    extra={"extra_fields": {"source": name, "error_class": err_class}},
                )
                results.append(
                    SourceTickResult(
                        name=name, events=0, duration_seconds=duration, error=err_class
                    )
                )
                continue

            duration = self._loop_clock() - source_start
            count = len(events)
            self._metrics.record_source_success(name, events=count, duration_seconds=duration)
            results.append(SourceTickResult(name=name, events=count, duration_seconds=duration))

            if count and self._on_events is not None:
                # Sink errors are isolated too — a bad sink should not
                # poison the tick for other sources.
                try:
                    await self._on_events(source, events)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    err_class = type(e).__name__
                    self._metrics.record_source_error(
                        name, error_class=f"sink:{err_class}", duration_seconds=duration
                    )
                    # Mark the source result as errored too — the events
                    # didn't make it through.
                    results[-1] = SourceTickResult(
                        name=name,
                        events=count,
                        duration_seconds=duration,
                        error=f"sink:{err_class}",
                    )

        total_duration = self._loop_clock() - tick_start
        self._metrics.record_tick(duration_seconds=total_duration)
        return AutoFetchResult(duration_seconds=total_duration, sources=results)

    def next_wait_seconds(self) -> float:
        """Interval + jittered padding for the next sleep.

        Returned as a float so tests can pin it via a stub `rng`.
        """
        if self._jitter == 0:
            return float(self._interval)
        return float(self._interval) + float(self._rng.uniform(0, self._jitter))

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    async def _run(self) -> None:
        log.info(
            "autofetch.started",
            extra={
                "extra_fields": {
                    "interval_seconds": self._interval,
                    "sources": [_source_name(s) for s in self._sources],
                }
            },
        )
        try:
            while not self._stop_event.is_set():
                try:
                    await self.tick_once()
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 — last-line defence
                    log.exception(
                        "autofetch.tick_unhandled",
                        extra={"extra_fields": {"error_class": type(e).__name__}},
                    )
                wait = self.next_wait_seconds()
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=wait)
                except TimeoutError:
                    continue
                else:
                    return
        except asyncio.CancelledError:
            return
        finally:
            log.info("autofetch.stopped")


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #


def _source_name(source: IngestSource) -> str:
    """Best-effort `source.name()` with a string-typed fallback."""
    name_fn = getattr(source, "name", None)
    if callable(name_fn):
        try:
            value = name_fn()
        except Exception:  # noqa: BLE001
            return type(source).__name__
        return str(value)
    return type(source).__name__


def _asyncio_loop_clock() -> float:
    """Monotonic-ish clock anchored to the running event loop.

    We use the loop's clock so tests using `asyncio.loop.time` (or a
    deterministic loop) stay consistent. Falls back to `asyncio.get_event_loop`
    when not inside a running loop (e.g. unit tests instantiating the
    scheduler without running it).
    """
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        # No running loop — use the default loop's clock. This branch is
        # only hit by tests that touch the scheduler synchronously.
        import time

        return time.monotonic()
