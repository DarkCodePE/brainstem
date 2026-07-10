"""
Tests for `wiki_integrations.base.OAuthIntegrationSource`.

Coverage matrix:

| Behaviour                                  | Test                                  |
| ------------------------------------------ | ------------------------------------- |
| start() flips `started` to True            | test_start_sets_started_true          |
| start() is idempotent                      | test_start_is_idempotent              |
| stop() flips `started` back to False       | test_stop_clears_started_flag         |
| stop() before start is a no-op             | test_stop_before_start_is_noop        |
| name() returns the constructor argument    | test_name_round_trips                 |
| Empty name raises ValueError               | test_empty_name_rejected              |
| Negative fetch_window raises ValueError    | test_negative_fetch_window_rejected   |
| Zero fetch_window raises ValueError        | test_zero_fetch_window_rejected       |
| Abstract fetch_batch raises NotImplemented | test_abstract_fetch_batch_raises      |
| `wiki_core.IngestSource` structural conform| test_satisfies_ingest_source_protocol |
| `emit` calls the wired callback            | test_emit_forwards_to_callback        |
| Subclass overriding start() chains super   | test_subclass_can_extend_start        |
| fetch_window property surface              | test_fetch_window_exposed             |
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wiki_core.protocols import IngestEvent, IngestSource
from wiki_integrations.base import OAuthIntegrationSource


class _NoOpSource(OAuthIntegrationSource):
    """Concrete subclass that does nothing — used to exercise base behaviour."""

    async def fetch_batch(self) -> list[IngestEvent]:
        return []


class _ExtendingSource(OAuthIntegrationSource):
    """Subclass that records super().start()/stop() invocations."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.started_extras = 0
        self.stopped_extras = 0

    async def start(self) -> None:
        await super().start()
        self.started_extras += 1

    async def stop(self) -> None:
        await super().stop()
        self.stopped_extras += 1

    async def fetch_batch(self) -> list[IngestEvent]:
        return []


def _make_event() -> IngestEvent:
    return IngestEvent(
        event_id="evt-1",
        source="test",
        path_or_uri="mailto:tester@example.com",
        sha256="a" * 64,
        received_at=datetime.now(UTC),
        metadata={
            "bucket": "test-bucket",
            "rel_path": "test-rel",
            "event_type": "created",
            "mtime": "2026-05-22T20:00:00Z",
            "size": 42,
        },
    )


@pytest.mark.asyncio
async def test_start_sets_started_true(recording_callback, fetch_window) -> None:
    src = _NoOpSource("noop", fetch_window=fetch_window, on_event=recording_callback)
    assert src.started is False
    await src.start()
    assert src.started is True


@pytest.mark.asyncio
async def test_start_is_idempotent(recording_callback, fetch_window) -> None:
    src = _NoOpSource("noop", fetch_window=fetch_window, on_event=recording_callback)
    await src.start()
    await src.start()  # second call must not raise
    assert src.started is True


@pytest.mark.asyncio
async def test_stop_clears_started_flag(recording_callback, fetch_window) -> None:
    src = _NoOpSource("noop", fetch_window=fetch_window, on_event=recording_callback)
    await src.start()
    await src.stop()
    assert src.started is False


@pytest.mark.asyncio
async def test_stop_before_start_is_noop(recording_callback, fetch_window) -> None:
    src = _NoOpSource("noop", fetch_window=fetch_window, on_event=recording_callback)
    await src.stop()
    assert src.started is False


def test_name_round_trips(recording_callback, fetch_window) -> None:
    src = _NoOpSource("telemetry-id", fetch_window=fetch_window, on_event=recording_callback)
    assert src.name() == "telemetry-id"


def test_empty_name_rejected(recording_callback, fetch_window) -> None:
    with pytest.raises(ValueError, match="non-empty name"):
        _NoOpSource("", fetch_window=fetch_window, on_event=recording_callback)


def test_negative_fetch_window_rejected(recording_callback) -> None:
    with pytest.raises(ValueError, match="positive"):
        _NoOpSource("noop", fetch_window=timedelta(seconds=-1), on_event=recording_callback)


def test_zero_fetch_window_rejected(recording_callback) -> None:
    with pytest.raises(ValueError, match="positive"):
        _NoOpSource("noop", fetch_window=timedelta(0), on_event=recording_callback)


@pytest.mark.asyncio
async def test_abstract_fetch_batch_raises(recording_callback, fetch_window) -> None:
    # Bare base class — must raise NotImplementedError because the abstract
    # method is the contract subclasses are required to fill.
    src = OAuthIntegrationSource("abstract", fetch_window=fetch_window, on_event=recording_callback)
    with pytest.raises(NotImplementedError, match="fetch_batch"):
        await src.fetch_batch()


def test_satisfies_ingest_source_protocol(recording_callback, fetch_window) -> None:
    src = _NoOpSource("noop", fetch_window=fetch_window, on_event=recording_callback)
    # `IngestSource` is `@runtime_checkable` so isinstance asserts structural shape.
    assert isinstance(src, IngestSource)


@pytest.mark.asyncio
async def test_emit_forwards_to_callback(recording_callback, fetch_window) -> None:
    src = _NoOpSource("noop", fetch_window=fetch_window, on_event=recording_callback)
    ev = _make_event()
    await src.emit(ev)
    assert recording_callback.events == [ev]


@pytest.mark.asyncio
async def test_subclass_can_extend_start(recording_callback, fetch_window) -> None:
    src = _ExtendingSource("ext", fetch_window=fetch_window, on_event=recording_callback)
    await src.start()
    await src.stop()
    assert src.started_extras == 1
    assert src.stopped_extras == 1
    assert src.started is False


def test_fetch_window_exposed(recording_callback) -> None:
    window = timedelta(hours=3)
    src = _NoOpSource("noop", fetch_window=window, on_event=recording_callback)
    assert src.fetch_window == window
