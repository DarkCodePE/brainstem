"""
`wiki_core.protocols.RateLimiter` — structural conformance + behaviour.

Pins the OQ-1 resolution from M3 Sprint 2: the auto-fetch in-memory
`TokenBucket` and the file-drop SQLite-backed `PersistentRateLimiter`
both satisfy a single Protocol, so daemon wiring can swap them out
without touching either worker.

Coverage matrix:

| Behaviour                                              | Test                                   |
| ------------------------------------------------------ | -------------------------------------- |
| `TokenBucket` satisfies the Protocol                   | test_token_bucket_satisfies_protocol   |
| `PersistentRateLimiter` satisfies the Protocol         | test_persistent_satisfies_protocol     |
| A class missing `try_acquire` does NOT satisfy it      | test_missing_try_acquire_fails         |
| A class missing `acquire` does NOT satisfy it          | test_missing_acquire_fails             |
| `try_acquire` is non-blocking (returns bool)           | test_try_acquire_non_blocking          |
| `acquire` blocks under exhaustion + releases on refill | test_acquire_blocks_until_refill       |
| `acquire(n > capacity)` raises                         | test_acquire_above_capacity_raises     |
| `try_acquire(n > capacity)` returns False, no raise    | test_try_acquire_above_capacity        |
"""

from __future__ import annotations

import asyncio

import pytest

from wiki_autofetch.rate_limiter import TokenBucket
from wiki_core.protocols import RateLimiter


class TestProtocolConformance:
    """Both concrete limiters satisfy the structural Protocol."""

    def test_token_bucket_satisfies_protocol(self) -> None:
        bucket = TokenBucket(capacity=5, refill_per_second=1.0)
        assert isinstance(bucket, RateLimiter)

    @pytest.mark.asyncio
    async def test_persistent_satisfies_protocol(self, tmp_path) -> None:
        from wiki_ingest.queue import EventQueue
        from wiki_ingest.worker import PersistentRateLimiter

        queue = EventQueue(tmp_path / "rate.db")
        await queue.init()
        try:
            limiter = PersistentRateLimiter(queue, limit_per_minute=10)
            assert isinstance(limiter, RateLimiter)
        finally:
            await queue.close()

    def test_missing_try_acquire_fails(self) -> None:
        """A class with only `acquire` does NOT structurally match."""

        class HalfLimiter:
            async def acquire(self, n: int = 1) -> None:
                return None

        assert not isinstance(HalfLimiter(), RateLimiter)

    def test_missing_acquire_fails(self) -> None:
        """A class with only `try_acquire` does NOT structurally match."""

        class HalfLimiter:
            def try_acquire(self, n: int = 1) -> bool:
                return True

        assert not isinstance(HalfLimiter(), RateLimiter)

    def test_full_shape_satisfies(self) -> None:
        """An ad-hoc class with the right shape satisfies the Protocol."""

        class CustomLimiter:
            async def acquire(self, n: int = 1) -> None:
                return None

            def try_acquire(self, n: int = 1) -> bool:
                return True

        assert isinstance(CustomLimiter(), RateLimiter)


class TestTryAcquireBehaviour:
    """Non-blocking semantics."""

    def test_try_acquire_non_blocking_success(self, fake_clock) -> None:
        bucket = TokenBucket(capacity=2, refill_per_second=1.0, clock=fake_clock)
        assert bucket.try_acquire(1) is True

    def test_try_acquire_non_blocking_miss(self, fake_clock) -> None:
        bucket = TokenBucket(capacity=1, refill_per_second=1.0, clock=fake_clock)
        assert bucket.try_acquire(1) is True
        # No clock advance — second call must miss without blocking.
        assert bucket.try_acquire(1) is False

    def test_try_acquire_above_capacity(self, fake_clock) -> None:
        bucket = TokenBucket(capacity=3, refill_per_second=1.0, clock=fake_clock)
        assert bucket.try_acquire(4) is False
        # Bucket state preserved.
        assert bucket.tokens == pytest.approx(3.0)


class TestAcquireBehaviour:
    """Blocking semantics — `acquire` sleeps then returns when tokens refill."""

    @pytest.mark.asyncio
    async def test_acquire_returns_immediately_when_available(self, fake_clock) -> None:
        bucket = TokenBucket(capacity=2, refill_per_second=1.0, clock=fake_clock)
        # No sleep needed: tokens available.
        await asyncio.wait_for(bucket.acquire(1), timeout=0.5)
        # One token consumed.
        assert bucket.tokens == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_acquire_blocks_until_refill(self) -> None:
        """With a real (fast) refill rate, `acquire` waits for the token."""
        bucket = TokenBucket(capacity=1, refill_per_second=20.0)
        # Drain the bucket.
        assert bucket.try_acquire(1) is True
        # Now the next `acquire` must wait ~0.05s for refill.
        loop = asyncio.get_event_loop()
        start = loop.time()
        await asyncio.wait_for(bucket.acquire(1), timeout=1.0)
        elapsed = loop.time() - start
        # Should have slept at least ~0.04s (allow generous slack for CI).
        assert elapsed >= 0.02

    @pytest.mark.asyncio
    async def test_acquire_above_capacity_raises(self) -> None:
        bucket = TokenBucket(capacity=2, refill_per_second=1.0)
        with pytest.raises(ValueError):
            await bucket.acquire(3)

    @pytest.mark.asyncio
    async def test_acquire_zero_raises(self) -> None:
        bucket = TokenBucket(capacity=2, refill_per_second=1.0)
        with pytest.raises(ValueError):
            await bucket.acquire(0)

    @pytest.mark.asyncio
    async def test_acquire_negative_raises(self) -> None:
        bucket = TokenBucket(capacity=2, refill_per_second=1.0)
        with pytest.raises(ValueError):
            await bucket.acquire(-1)


class TestPersistentRateLimiterBehaviour:
    """`PersistentRateLimiter.acquire` honours the minute-window budget."""

    @pytest.mark.asyncio
    async def test_acquire_succeeds_under_budget(self, tmp_path) -> None:
        from wiki_ingest.queue import EventQueue
        from wiki_ingest.worker import PersistentRateLimiter

        queue = EventQueue(tmp_path / "rate.db")
        await queue.init()
        try:
            limiter = PersistentRateLimiter(queue, limit_per_minute=3)
            # Should grant immediately for the first three tokens.
            await asyncio.wait_for(limiter.acquire(1), timeout=1.0)
            await asyncio.wait_for(limiter.acquire(1), timeout=1.0)
            await asyncio.wait_for(limiter.acquire(1), timeout=1.0)
        finally:
            await queue.close()

    @pytest.mark.asyncio
    async def test_try_acquire_returns_bool(self, tmp_path) -> None:
        """The Protocol's non-blocking surface — must not raise, returns bool."""
        from wiki_ingest.queue import EventQueue
        from wiki_ingest.worker import PersistentRateLimiter

        queue = EventQueue(tmp_path / "rate.db")
        await queue.init()
        try:
            limiter = PersistentRateLimiter(queue, limit_per_minute=10)
            result = limiter.try_acquire(1)
            assert isinstance(result, bool)
        finally:
            await queue.close()
