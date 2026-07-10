"""
Deterministic token-bucket rate limiter.

Per [PRD-006 FR-7], auto-fetch and the file-drop ingest worker cooperate
on the same 10/min `write_page` budget. The persistent variant in
`wiki_ingest.worker.PersistentRateLimiter` is keyed on SQLite minute
windows for crash safety; this in-memory variant is for the auto-fetch
loop's own per-tick / per-source budgets where survival across restarts
isn't required (cursors carry the durable state).

Both this bucket and `PersistentRateLimiter` satisfy the shared
`wiki_core.RateLimiter` Protocol (OQ-1 resolution, M3 Sprint 2). That
gives the daemon a single seam to swap in a different limiter without
touching either worker.

Design notes
------------
- **No `time.time()`** in the hot path: the clock is injected via the
  constructor so tests stay deterministic without monkey-patching.
- **Refill is continuous** (not discrete window): each `try_acquire` call
  computes how many tokens have refilled since the last update and
  tops the bucket up to capacity.
- **`try_acquire` is synchronous, `acquire` is async**: per the
  `wiki_core.RateLimiter` Protocol, callers choose between non-blocking
  drop-on-miss (`try_acquire`) and blocking-until-available (`acquire`).
  The async path uses `asyncio.sleep(time_until_available(n))` so the
  bucket itself never busy-polls; the wait is precise because the refill
  rate is known.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class _State:
    tokens: float
    last_refill: float


class TokenBucket:
    """A classic token-bucket limiter with continuous refill.

    Parameters
    ----------
    capacity:
        Maximum number of tokens the bucket can hold.
    refill_per_second:
        Rate at which tokens regenerate. e.g. ``10/60`` for 10 per minute.
    clock:
        Callable returning monotonic seconds. Defaults to `time.monotonic`.
        Pass a fake clock from tests for determinism.

    Notes
    -----
    Constructor validates that ``capacity >= 1`` and ``refill_per_second > 0``.
    A bucket starts **full** — callers get ``capacity`` free tokens before
    any rate limiting kicks in.
    """

    __slots__ = ("_capacity", "_refill", "_clock", "_state")

    def __init__(
        self,
        *,
        capacity: int,
        refill_per_second: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if refill_per_second <= 0:
            raise ValueError("refill_per_second must be > 0")
        self._capacity = int(capacity)
        self._refill = float(refill_per_second)
        self._clock = clock if clock is not None else time.monotonic
        self._state = _State(tokens=float(capacity), last_refill=self._clock())

    # ------------------------------------------------------------------ #
    # Read-only properties (mostly for tests / observability)            #
    # ------------------------------------------------------------------ #

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def refill_per_second(self) -> float:
        return self._refill

    @property
    def tokens(self) -> float:
        """Current token count *after* applying refill at `now()`.

        Reading this property mutates the internal `last_refill` so the
        next mutator call doesn't double-count refill.
        """
        self._refill_now()
        return self._state.tokens

    # ------------------------------------------------------------------ #
    # Core API                                                           #
    # ------------------------------------------------------------------ #

    def try_acquire(self, n: int = 1) -> bool:
        """Non-blocking acquisition — `wiki_core.RateLimiter` shape.

        Try to take `n` tokens. Returns True on success, False if the
        bucket lacks enough tokens at the current clock reading.

        Asking for ``n > capacity`` always returns False — a request
        that big can never be satisfied.

        This was the historical `acquire` method; it was renamed to
        match the shared `RateLimiter` Protocol (OQ-1). The bool-return
        semantics are unchanged.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self._capacity:
            return False
        self._refill_now()
        if self._state.tokens + 1e-9 >= n:
            self._state.tokens -= n
            return True
        return False

    async def acquire(self, n: int = 1) -> None:
        """Blocking acquisition — `wiki_core.RateLimiter` shape.

        Sleeps until `n` tokens are available and then consumes them.
        Uses `time_until_available` to compute the precise wait, then
        retries; under a real `time.monotonic` clock that converges in
        one or two iterations, and under a fake-clock test the retry
        loop is bounded by however far the clock has been advanced.

        Raises `ValueError` if ``n > capacity`` — that request can never
        be satisfied so an unbounded sleep would be a bug, not a feature.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self._capacity:
            raise ValueError(
                f"acquire({n}) exceeds bucket capacity {self._capacity}; "
                "request can never be satisfied"
            )
        while True:
            if self.try_acquire(n):
                return
            wait = self.time_until_available(n)
            # `wait` is finite here because `n <= capacity` was checked
            # above; clamp to a tiny positive value so we always yield.
            await asyncio.sleep(max(wait, 1e-3))

    def time_until_available(self, n: int = 1) -> float:
        """Seconds until `n` tokens are available.

        Returns 0.0 if already available, ``+inf`` if ``n > capacity``
        (unreachable), otherwise the precise wait derived from the
        refill rate.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self._capacity:
            return float("inf")
        self._refill_now()
        deficit = n - self._state.tokens
        if deficit <= 0:
            return 0.0
        return deficit / self._refill

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _refill_now(self) -> None:
        now = self._clock()
        delta = now - self._state.last_refill
        if delta < 0:
            # Clock went backwards (only possible with a manually-set
            # fake clock). Treat as "no refill" and re-anchor.
            self._state.last_refill = now
            return
        if delta == 0:
            return
        added = delta * self._refill
        self._state.tokens = min(self._capacity, self._state.tokens + added)
        self._state.last_refill = now
