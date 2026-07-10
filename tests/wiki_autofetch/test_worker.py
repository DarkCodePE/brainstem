"""AutoFetchWorker — end-to-end with stub source + in-memory MemoryStore."""

from __future__ import annotations

import pytest

from wiki_autofetch.rate_limiter import TokenBucket
from wiki_autofetch.worker import AutoFetchWorker


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_events_flow_to_memory_store(
        self, stub_source_factory, event_factory, memory_store
    ) -> None:
        evs = [event_factory(sha256="a" * 64), event_factory(sha256="b" * 64)]
        s = stub_source_factory("gmail", returns=[evs])
        worker = AutoFetchWorker(
            sources=[s],
            store=memory_store,
            interval_seconds=3600,
            jitter_seconds=0,
        )
        await worker.tick_once()
        assert len(memory_store.enqueued) == 2
        assert {e.sha256 for e in memory_store.enqueued} == {"a" * 64, "b" * 64}

    @pytest.mark.asyncio
    async def test_duplicate_sha_skipped(
        self, stub_source_factory, event_factory, memory_store
    ) -> None:
        # Pre-load the store with a known SHA.
        sha = "f" * 64
        memory_store.seen.add(sha)
        s = stub_source_factory("gmail", returns=[[event_factory(sha256=sha)]])
        worker = AutoFetchWorker(
            sources=[s],
            store=memory_store,
            interval_seconds=3600,
            jitter_seconds=0,
        )
        await worker.tick_once()
        # Duplicate filtered out — nothing enqueued.
        assert memory_store.enqueued == []

    @pytest.mark.asyncio
    async def test_per_source_error_does_not_block_others(
        self, stub_source_factory, event_factory, memory_store
    ) -> None:
        bad = stub_source_factory("gmail", returns=[RuntimeError("api down")])
        good = stub_source_factory("github", returns=[[event_factory(sha256="1" * 64)]])
        worker = AutoFetchWorker(
            sources=[bad, good],
            store=memory_store,
            interval_seconds=3600,
            jitter_seconds=0,
        )
        result = await worker.tick_once()
        # The good source's event survived.
        assert len(memory_store.enqueued) == 1
        assert memory_store.enqueued[0].source == "stub"
        # Metrics counted the failure.
        assert worker.metrics.total_errors >= 1
        # Result contains one failure and one success.
        names_errored = {r.name for r in result.errors}
        assert "gmail" in names_errored

    @pytest.mark.asyncio
    async def test_enqueue_failure_recorded_in_metrics(
        self, stub_source_factory, event_factory, memory_store
    ) -> None:
        memory_store.fail_next_enqueue = RuntimeError("db locked")
        s = stub_source_factory("gmail", returns=[[event_factory(sha256="2" * 64)]])
        worker = AutoFetchWorker(
            sources=[s],
            store=memory_store,
            interval_seconds=3600,
            jitter_seconds=0,
        )
        await worker.tick_once()
        # Enqueue failed, so nothing in the store yet.
        assert memory_store.enqueued == []
        # And the error was logged on the source's metrics.
        assert worker.metrics.sources["gmail"].errors >= 1


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_rate_limited_events_are_skipped_and_counted(
        self, stub_source_factory, event_factory, memory_store, fake_clock
    ) -> None:
        # Tiny bucket: only 1 token, refill very slowly.
        bucket = TokenBucket(capacity=1, refill_per_second=0.01, clock=fake_clock)
        evs = [
            event_factory(sha256="a" * 64),
            event_factory(sha256="b" * 64),
            event_factory(sha256="c" * 64),
        ]
        s = stub_source_factory("gmail", returns=[evs])
        worker = AutoFetchWorker(
            sources=[s],
            store=memory_store,
            interval_seconds=3600,
            jitter_seconds=0,
            rate_limiter=bucket,
            rate_limit_wait_seconds=0,  # don't wait — drop immediately on miss
        )
        await worker.tick_once()
        # Only one event passed the bucket.
        assert len(memory_store.enqueued) == 1
        # The other two were counted as rate-limited.
        assert worker.metrics.total_rate_limited == 2

    @pytest.mark.asyncio
    async def test_rate_limit_wait_drains_token(
        self, stub_source_factory, event_factory, memory_store
    ) -> None:
        # A generous bucket so no drops happen — happy path with wait enabled.
        bucket = TokenBucket(capacity=10, refill_per_second=10.0)
        s = stub_source_factory("gmail", returns=[[event_factory(sha256="x" * 64)]])
        worker = AutoFetchWorker(
            sources=[s],
            store=memory_store,
            interval_seconds=3600,
            jitter_seconds=0,
            rate_limiter=bucket,
            rate_limit_wait_seconds=0.1,
        )
        await worker.tick_once()
        assert len(memory_store.enqueued) == 1

    @pytest.mark.asyncio
    async def test_invalid_wait_rejected(self, memory_store) -> None:
        with pytest.raises(ValueError):
            AutoFetchWorker(
                sources=[],
                store=memory_store,
                rate_limit_wait_seconds=-1,
            )


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop_round_trip(self, stub_source_factory, memory_store) -> None:
        s = stub_source_factory("gmail")
        worker = AutoFetchWorker(
            sources=[s],
            store=memory_store,
            interval_seconds=3600,
            jitter_seconds=0,
        )
        assert not worker.is_running
        await worker.start()
        assert worker.is_running
        await worker.stop()
        assert not worker.is_running

    @pytest.mark.asyncio
    async def test_metrics_exposed(self, memory_store) -> None:
        worker = AutoFetchWorker(sources=[], store=memory_store)
        snap = worker.metrics.to_dict()
        assert snap["total_ticks"] == 0
        assert "sources" in snap
        assert worker.rate_limiter.capacity == 10  # default 10/min budget
