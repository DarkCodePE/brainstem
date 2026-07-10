"""TokenBucket — deterministic behaviour under a fake clock."""

from __future__ import annotations

import pytest

from wiki_autofetch.rate_limiter import TokenBucket


class TestConstruction:
    def test_starts_full(self, fake_clock) -> None:
        b = TokenBucket(capacity=5, refill_per_second=1.0, clock=fake_clock)
        assert b.capacity == 5
        assert b.refill_per_second == 1.0
        assert b.tokens == pytest.approx(5.0)

    def test_rejects_zero_capacity(self, fake_clock) -> None:
        with pytest.raises(ValueError):
            TokenBucket(capacity=0, refill_per_second=1.0, clock=fake_clock)

    def test_rejects_negative_refill(self, fake_clock) -> None:
        with pytest.raises(ValueError):
            TokenBucket(capacity=5, refill_per_second=0.0, clock=fake_clock)

    def test_default_clock_is_time_monotonic(self) -> None:
        # No clock passed: object still constructs and operates.
        b = TokenBucket(capacity=2, refill_per_second=1.0)
        assert b.try_acquire() is True
        assert b.try_acquire() is True
        # Third call may or may not succeed depending on the real clock;
        # what matters is that the construction path doesn't crash.


class TestAcquire:
    def test_drains_until_empty(self, fake_clock) -> None:
        b = TokenBucket(capacity=3, refill_per_second=1.0, clock=fake_clock)
        assert b.try_acquire() is True
        assert b.try_acquire() is True
        assert b.try_acquire() is True
        # Bucket empty at t=0, no time has passed -> fail.
        assert b.try_acquire() is False

    def test_refills_at_rate(self, fake_clock) -> None:
        b = TokenBucket(capacity=2, refill_per_second=1.0, clock=fake_clock)
        assert b.try_acquire() is True
        assert b.try_acquire() is True
        assert b.try_acquire() is False
        fake_clock.tick(1.0)  # one token regenerated
        assert b.try_acquire() is True
        assert b.try_acquire() is False

    def test_refill_caps_at_capacity(self, fake_clock) -> None:
        b = TokenBucket(capacity=2, refill_per_second=1.0, clock=fake_clock)
        # Sit idle for 60s — bucket can never exceed 2 tokens.
        fake_clock.tick(60.0)
        assert b.tokens == pytest.approx(2.0)
        assert b.try_acquire() is True
        assert b.try_acquire() is True
        assert b.try_acquire() is False

    def test_acquire_multiple(self, fake_clock) -> None:
        b = TokenBucket(capacity=10, refill_per_second=1.0, clock=fake_clock)
        assert b.try_acquire(5) is True
        assert b.tokens == pytest.approx(5.0)
        assert b.try_acquire(6) is False  # insufficient
        assert b.try_acquire(5) is True

    def test_acquire_more_than_capacity_always_false(self, fake_clock) -> None:
        b = TokenBucket(capacity=3, refill_per_second=1.0, clock=fake_clock)
        assert b.try_acquire(4) is False
        # Bucket state untouched — original full count remains.
        assert b.tokens == pytest.approx(3.0)

    def test_acquire_zero_raises(self, fake_clock) -> None:
        b = TokenBucket(capacity=3, refill_per_second=1.0, clock=fake_clock)
        with pytest.raises(ValueError):
            b.try_acquire(0)

    def test_acquire_negative_raises(self, fake_clock) -> None:
        b = TokenBucket(capacity=3, refill_per_second=1.0, clock=fake_clock)
        with pytest.raises(ValueError):
            b.try_acquire(-1)


class TestTimeUntilAvailable:
    def test_zero_when_available(self, fake_clock) -> None:
        b = TokenBucket(capacity=5, refill_per_second=2.0, clock=fake_clock)
        assert b.time_until_available(1) == pytest.approx(0.0)
        assert b.time_until_available(5) == pytest.approx(0.0)

    def test_returns_inf_when_above_capacity(self, fake_clock) -> None:
        b = TokenBucket(capacity=3, refill_per_second=1.0, clock=fake_clock)
        assert b.time_until_available(4) == float("inf")

    def test_precise_wait_after_drain(self, fake_clock) -> None:
        b = TokenBucket(capacity=2, refill_per_second=2.0, clock=fake_clock)
        b.try_acquire(2)
        # Empty: need 1 token at 2 tokens/sec = 0.5s
        assert b.time_until_available(1) == pytest.approx(0.5)
        # Need 2 tokens at 2 tokens/sec = 1.0s
        assert b.time_until_available(2) == pytest.approx(1.0)

    def test_decreases_as_time_passes(self, fake_clock) -> None:
        b = TokenBucket(capacity=1, refill_per_second=1.0, clock=fake_clock)
        b.try_acquire()
        assert b.time_until_available(1) == pytest.approx(1.0)
        fake_clock.tick(0.4)
        assert b.time_until_available(1) == pytest.approx(0.6)


class TestRobustness:
    def test_backwards_clock_does_not_crash(self, fake_clock) -> None:
        b = TokenBucket(capacity=2, refill_per_second=1.0, clock=fake_clock)
        b.try_acquire()
        # Simulate clock going backward (manual fake-clock testing only).
        fake_clock.set(-10.0)
        # Should not raise, should not produce phantom tokens.
        tokens_after = b.tokens
        assert 0.0 <= tokens_after <= 2.0

    def test_continuous_refill_is_proportional(self, fake_clock) -> None:
        b = TokenBucket(capacity=10, refill_per_second=4.0, clock=fake_clock)
        b.try_acquire(10)
        fake_clock.tick(0.25)  # 1 token added
        assert b.tokens == pytest.approx(1.0)
        fake_clock.tick(0.25)  # another 1
        assert b.tokens == pytest.approx(2.0)

    def test_acquire_after_partial_refill(self, fake_clock) -> None:
        b = TokenBucket(capacity=10, refill_per_second=10.0, clock=fake_clock)
        b.try_acquire(10)
        fake_clock.tick(0.3)  # 3 tokens refilled
        assert b.try_acquire(3) is True
        assert b.try_acquire(1) is False  # exhausted at 0
