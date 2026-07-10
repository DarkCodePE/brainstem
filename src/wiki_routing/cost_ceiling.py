"""
Cost ceiling enforcement per [ADR-013 §"Cost ceiling enforcement"](../../docs/ADR-013-model-router-policy.md).

A ``CostBudget`` enforces two ceilings:

- **per-task** — refuse any single call whose pre-estimated cost
  exceeds ``max_per_task_usd``. Catches "long prompt slipped into the
  Reasoning tier" before the API is dialled.
- **per-day** — running counter; once the cumulative day spend
  crosses ``max_per_day_usd`` every subsequent call is refused.
  Mirrors ADR-013's "hard-cut at 100% — local-only" semantics, but
  this module is **provider-agnostic**: it doesn't know which tier
  qualifies as "local", so it just refuses; the router is responsible
  for surfacing a local-only fallback when configured to do so
  (PRD-008 US-003).

The budget object is **synchronous** because cost checks must be
free of I/O — refusing in <1ms is required for AC-1 (≤ 5ms P95
routing overhead).

The "day" boundary is computed in UTC. Tests that need to advance
time pass a fixed ``now`` to ``charge``/``check``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime


class CostCeilingError(RuntimeError):
    """Raised by ``CostBudget.charge`` (and ``ModelRouter.call`` via the
    pre-call check) when the requested call would exceed either the
    per-task or per-day ceiling.

    The ``kind`` attribute lets callers / telemetry differentiate
    between the two refusal modes without parsing the message text.
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class CostQuote:
    """Pre-call cost estimate. The router feeds this to ``CostBudget``
    before dispatching to a backend.

    ``estimated_usd`` is the **upper bound** the router believes this
    call will cost based on input token count × backend price per 1k
    plus a small headroom for the output. ADR-013 suggests a 10%
    safety margin; the policy applies that margin before constructing
    the quote, so this module compares the quote literally."""

    estimated_usd: float
    backend_label: str
    """Short label like ``"anthropic:claude-sonnet-4.5"``; preserved on
    the resulting refusal exception for debugging only — never logged
    with secrets."""


@dataclass
class CostBudget:
    """Per-day + per-task spend ceiling.

    Mutable on purpose: ``charge`` updates ``_spent_today`` in place
    after a successful call. Callers are expected to construct one
    budget per ``ModelRouter`` instance (i.e. per process). Persistence
    across daemon restarts is **out of scope for M3** — ADR-013 calls
    out a SQLite-backed counter but that lands with PRD-011.

    Parameters
    ----------
    max_per_task_usd:
        Cap for any single call. Tasks above this are refused before
        dispatch with ``kind="per_task"``.
    max_per_day_usd:
        Cap for the cumulative day. Calls that would push
        ``_spent_today`` above this are refused with ``kind="per_day"``.
    """

    max_per_task_usd: float
    max_per_day_usd: float
    _spent_today: float = field(default=0.0, init=False)
    _day: date = field(default_factory=lambda: datetime.now(UTC).date(), init=False)

    def check(self, quote: CostQuote, *, now: datetime | None = None) -> None:
        """Refuse if ``quote`` would breach either ceiling. Pure (no
        state mutation) so the router can pre-flight several backend
        options before picking one.

        Raises
        ------
        CostCeilingError
            ``kind="per_task"`` when the single-call estimate is too high;
            ``kind="per_day"`` when the running counter is exhausted.
        """
        self._rollover_if_needed(now)
        if quote.estimated_usd > self.max_per_task_usd:
            raise CostCeilingError(
                f"per-task ceiling ${self.max_per_task_usd:.4f} exceeded by "
                f"${quote.estimated_usd:.4f} ({quote.backend_label})",
                kind="per_task",
            )
        if self._spent_today + quote.estimated_usd > self.max_per_day_usd:
            raise CostCeilingError(
                f"per-day ceiling ${self.max_per_day_usd:.4f} would be exceeded "
                f"(currently ${self._spent_today:.4f}, +${quote.estimated_usd:.4f})",
                kind="per_day",
            )

    def charge(self, actual_usd: float, *, now: datetime | None = None) -> None:
        """Record an actual post-call cost. Negative values are rejected.

        Called by ``ModelRouter`` after the backend returns; the value
        is the post-call cost computed from real token counts. Note
        that ``charge`` does **not** re-check ceilings — by the time
        the call has run, the spend has happened. The next ``check``
        will fail if the day budget is now exhausted.
        """
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be non-negative, got {actual_usd}")
        self._rollover_if_needed(now)
        self._spent_today += actual_usd

    def spent_today(self, *, now: datetime | None = None) -> float:
        """Current running counter for the UTC day containing ``now``."""
        self._rollover_if_needed(now)
        return self._spent_today

    def reset(self) -> None:
        """Force the counter back to zero (testing / "extend ceiling"
        UX action from ADR-013)."""
        self._spent_today = 0.0
        self._day = datetime.now(UTC).date()

    def _rollover_if_needed(self, now: datetime | None) -> None:
        """If the UTC calendar day has rolled over, reset the counter.

        Implemented inline instead of via a scheduled job so the budget
        behaves correctly even if the daemon was asleep across midnight.
        """
        today = (now or datetime.now(UTC)).astimezone(UTC).date()
        if today != self._day:
            self._day = today
            self._spent_today = 0.0


__all__ = ["CostBudget", "CostCeilingError", "CostQuote"]
