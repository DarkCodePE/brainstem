"""Fixtures for the wiki_autofetch tests.

Deterministic by construction:

- `FakeClock` lets `TokenBucket` advance time without `time.sleep`.
- `StubSource` implements the `IngestSource` shape with a queued list of
  `(events_or_exc)` returns so tests don't touch a network.
- `InMemoryMemoryStore` implements the `MemoryStore` protocol with an
  in-process dict — no SQLite needed for these unit tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from wiki_core.protocols import IngestEvent

# --------------------------------------------------------------------------- #
# Clocks                                                                      #
# --------------------------------------------------------------------------- #


class FakeClock:
    """Manually-advanced clock with `time.monotonic`-compatible signature.

    Tests call `tick(seconds)` to move forward. The instance is callable
    so it slots into `TokenBucket(clock=fake_clock)` directly.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)

    def __call__(self) -> float:
        return self._now

    def tick(self, seconds: float) -> None:
        self._now += float(seconds)

    def set(self, t: float) -> None:
        self._now = float(t)


@pytest.fixture
def fake_clock() -> FakeClock:
    return FakeClock(start=1000.0)


# --------------------------------------------------------------------------- #
# Events                                                                      #
# --------------------------------------------------------------------------- #


def make_event(
    *,
    sha256: str | None = None,
    source: str = "stub",
    event_id: str | None = None,
    path_or_uri: str = "stub://item",
    metadata: dict[str, Any] | None = None,
) -> IngestEvent:
    """Factory for `IngestEvent`s with minimal-but-typed values."""
    return IngestEvent(
        event_id=event_id or f"evt-{id(object())}",
        source=source,
        path_or_uri=path_or_uri,
        sha256=sha256 or ("a" * 64),
        received_at=datetime.now(UTC),
        metadata=metadata or {},
    )


@pytest.fixture
def event_factory() -> Iterator[Any]:
    yield make_event


# --------------------------------------------------------------------------- #
# Stub source                                                                 #
# --------------------------------------------------------------------------- #


class StubSource:
    """In-process IngestSource with a programmable `fetch_delta`.

    Pass a list of "returns" — each entry is either a list of events
    (success) or an `Exception` instance (raise). One entry is consumed
    per `fetch_delta()` call; once exhausted, returns `[]` forever.
    """

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
        # Use BaseException so `CancelledError` (BaseException in 3.8+) is
        # raised rather than returned as data.
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def stub_source_factory() -> Iterator[Any]:
    """Hand back the StubSource class so tests can build their own."""
    yield StubSource


# --------------------------------------------------------------------------- #
# In-memory MemoryStore                                                       #
# --------------------------------------------------------------------------- #


class InMemoryMemoryStore:
    """Minimal `wiki_core.MemoryStore`-shaped store for unit tests."""

    def __init__(self) -> None:
        self.enqueued: list[IngestEvent] = []
        self.seen: set[str] = set()
        # If set, `enqueue` raises this on the next call (one-shot).
        self.fail_next_enqueue: Exception | None = None

    async def enqueue(self, event: IngestEvent) -> str:
        if self.fail_next_enqueue is not None:
            err = self.fail_next_enqueue
            self.fail_next_enqueue = None
            raise err
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


@pytest.fixture
def memory_store() -> InMemoryMemoryStore:
    return InMemoryMemoryStore()
