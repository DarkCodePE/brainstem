"""AutoFetchScheduler — tick semantics, error isolation, lifecycle."""

from __future__ import annotations

import asyncio
import random

import pytest

from wiki_autofetch.scheduler import AutoFetchResult, AutoFetchScheduler


class TestTickOnce:
    @pytest.mark.asyncio
    async def test_iterates_all_sources(self, stub_source_factory, event_factory) -> None:
        s1 = stub_source_factory("gmail", returns=[[event_factory(sha256="a" * 64)]])
        s2 = stub_source_factory("github", returns=[[event_factory(sha256="b" * 64)]])
        sched = AutoFetchScheduler([s1, s2])
        result = await sched.tick_once()
        assert isinstance(result, AutoFetchResult)
        assert s1.fetch_calls == 1
        assert s2.fetch_calls == 1
        assert {r.name for r in result.sources} == {"gmail", "github"}
        assert result.total_events == 2

    @pytest.mark.asyncio
    async def test_empty_source_list_returns_clean_result(self) -> None:
        sched = AutoFetchScheduler([])
        result = await sched.tick_once()
        assert result.sources == []
        assert result.total_events == 0

    @pytest.mark.asyncio
    async def test_source_without_fetch_delta_yields_zero(self, event_factory) -> None:
        class Legacy:
            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            def name(self) -> str:
                return "legacy"

        sched = AutoFetchScheduler([Legacy()])
        result = await sched.tick_once()
        assert result.total_events == 0
        assert result.sources[0].name == "legacy"
        assert result.sources[0].error is None

    @pytest.mark.asyncio
    async def test_metrics_updated_after_tick(self, stub_source_factory, event_factory) -> None:
        s = stub_source_factory("gmail", returns=[[event_factory(), event_factory()]])
        sched = AutoFetchScheduler([s])
        await sched.tick_once()
        m = sched.metrics
        assert m.total_ticks == 1
        assert m.total_events_fetched == 2
        assert m.sources["gmail"].events_fetched == 2


class TestErrorIsolation:
    @pytest.mark.asyncio
    async def test_one_source_raises_others_still_run(
        self, stub_source_factory, event_factory
    ) -> None:
        bad = stub_source_factory("gmail", returns=[RuntimeError("boom")])
        good = stub_source_factory("github", returns=[[event_factory()]])
        sched = AutoFetchScheduler([bad, good])
        result = await sched.tick_once()
        # Both sources executed
        assert bad.fetch_calls == 1
        assert good.fetch_calls == 1
        # Error captured on the bad one
        err_results = result.errors
        assert len(err_results) == 1
        assert err_results[0].name == "gmail"
        assert err_results[0].error == "RuntimeError"
        # Good source still produced its event
        good_result = next(r for r in result.sources if r.name == "github")
        assert good_result.events == 1
        # Metrics reflect both
        assert sched.metrics.total_errors == 1
        assert sched.metrics.total_events_fetched == 1

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self, stub_source_factory) -> None:
        bad = stub_source_factory("gmail", returns=[asyncio.CancelledError()])
        sched = AutoFetchScheduler([bad])
        with pytest.raises(asyncio.CancelledError):
            await sched.tick_once()

    @pytest.mark.asyncio
    async def test_sink_error_does_not_block_other_sources(
        self, stub_source_factory, event_factory
    ) -> None:
        calls: list[str] = []

        async def sink(source, events):
            calls.append(source.name())
            if source.name() == "gmail":
                raise ValueError("sink down")

        s1 = stub_source_factory("gmail", returns=[[event_factory()]])
        s2 = stub_source_factory("github", returns=[[event_factory()]])
        sched = AutoFetchScheduler([s1, s2], on_events=sink)
        result = await sched.tick_once()
        # Both sinks called; second one succeeded.
        assert calls == ["gmail", "github"]
        gmail = next(r for r in result.sources if r.name == "gmail")
        github = next(r for r in result.sources if r.name == "github")
        assert gmail.error == "sink:ValueError"
        assert github.error is None


class TestSinkInvocation:
    @pytest.mark.asyncio
    async def test_on_events_called_with_fetched_events(
        self, stub_source_factory, event_factory
    ) -> None:
        captured: list = []

        async def sink(source, events):
            captured.append((source.name(), list(events)))

        evs = [event_factory(sha256="c" * 64), event_factory(sha256="d" * 64)]
        s = stub_source_factory("gmail", returns=[evs])
        sched = AutoFetchScheduler([s], on_events=sink)
        await sched.tick_once()
        assert len(captured) == 1
        name, items = captured[0]
        assert name == "gmail"
        assert len(items) == 2

    @pytest.mark.asyncio
    async def test_on_events_not_called_when_no_events(self, stub_source_factory) -> None:
        captured: list = []

        async def sink(source, events):
            captured.append(source.name())

        s = stub_source_factory("gmail", returns=[[]])
        sched = AutoFetchScheduler([s], on_events=sink)
        await sched.tick_once()
        assert captured == []


class TestJitter:
    def test_jitter_zero_returns_base_interval(self) -> None:
        sched = AutoFetchScheduler([], interval_seconds=300, jitter_seconds=0)
        assert sched.next_wait_seconds() == 300.0

    def test_jitter_applied_within_bound(self) -> None:
        sched = AutoFetchScheduler(
            [], interval_seconds=300, jitter_seconds=30, rng=random.Random(0)
        )
        for _ in range(50):
            w = sched.next_wait_seconds()
            assert 300.0 <= w <= 330.0

    def test_jitter_uses_injected_rng(self) -> None:
        sched_a = AutoFetchScheduler(
            [], interval_seconds=100, jitter_seconds=10, rng=random.Random(42)
        )
        sched_b = AutoFetchScheduler(
            [], interval_seconds=100, jitter_seconds=10, rng=random.Random(42)
        )
        # Same seed => same sequence.
        for _ in range(20):
            assert sched_a.next_wait_seconds() == sched_b.next_wait_seconds()

    def test_invalid_interval_rejected(self) -> None:
        with pytest.raises(ValueError):
            AutoFetchScheduler([], interval_seconds=0)

    def test_negative_jitter_rejected(self) -> None:
        with pytest.raises(ValueError):
            AutoFetchScheduler([], interval_seconds=10, jitter_seconds=-1)


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_then_stop_is_clean(self, stub_source_factory) -> None:
        s = stub_source_factory("gmail")
        # Use a tiny interval so the background loop will tick if not stopped.
        sched = AutoFetchScheduler([s], interval_seconds=3600, jitter_seconds=0)
        await sched.start()
        assert sched.is_running
        await sched.stop()
        assert not sched.is_running

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, stub_source_factory) -> None:
        sched = AutoFetchScheduler(
            [stub_source_factory("gmail")], interval_seconds=3600, jitter_seconds=0
        )
        await sched.start()
        first = sched._task  # type: ignore[attr-defined]
        await sched.start()  # second call is a no-op
        second = sched._task  # type: ignore[attr-defined]
        assert first is second
        await sched.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, stub_source_factory) -> None:
        sched = AutoFetchScheduler(
            [stub_source_factory("gmail")], interval_seconds=3600, jitter_seconds=0
        )
        await sched.start()
        await sched.stop()
        # Second stop must not raise.
        await sched.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self) -> None:
        sched = AutoFetchScheduler([], interval_seconds=3600, jitter_seconds=0)
        await sched.stop()
        assert not sched.is_running

    @pytest.mark.asyncio
    async def test_background_tick_actually_runs_once(
        self, stub_source_factory, event_factory
    ) -> None:
        # Tiny interval so a tick fires almost immediately, then we stop.
        s = stub_source_factory(
            "gmail",
            returns=[[event_factory()], [event_factory()], [event_factory()]],
        )
        sched = AutoFetchScheduler([s], interval_seconds=1, jitter_seconds=0)
        await sched.start()
        # Give the loop one full cycle.
        await asyncio.sleep(0.05)
        await sched.stop()
        assert s.fetch_calls >= 1
