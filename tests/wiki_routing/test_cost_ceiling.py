"""Tests for ``wiki_routing.cost_ceiling`` — per-task + per-day budget."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wiki_routing.cost_ceiling import CostBudget, CostCeilingError, CostQuote


def _quote(usd: float, label: str = "stub") -> CostQuote:
    return CostQuote(estimated_usd=usd, backend_label=label)


class TestPerTaskCeiling:
    def test_allows_call_under_ceiling(self) -> None:
        b = CostBudget(max_per_task_usd=0.10, max_per_day_usd=10.0)
        b.check(_quote(0.05))  # No raise.

    def test_refuses_call_at_ceiling_plus_epsilon(self) -> None:
        b = CostBudget(max_per_task_usd=0.10, max_per_day_usd=10.0)
        with pytest.raises(CostCeilingError) as ei:
            b.check(_quote(0.11))
        assert ei.value.kind == "per_task"

    def test_allows_call_exactly_at_ceiling(self) -> None:
        # The contract is "exceeded" — equality passes.
        b = CostBudget(max_per_task_usd=0.10, max_per_day_usd=10.0)
        b.check(_quote(0.10))  # No raise.

    def test_refusal_message_contains_backend_label(self) -> None:
        b = CostBudget(max_per_task_usd=0.001, max_per_day_usd=1.0)
        with pytest.raises(CostCeilingError) as ei:
            b.check(_quote(0.5, label="anthropic:claude-opus"))
        assert "anthropic:claude-opus" in str(ei.value)


class TestPerDayCeiling:
    def test_running_counter_starts_at_zero(self) -> None:
        b = CostBudget(max_per_task_usd=1.0, max_per_day_usd=5.0)
        assert b.spent_today() == 0.0

    def test_charge_increments_counter(self) -> None:
        b = CostBudget(max_per_task_usd=1.0, max_per_day_usd=5.0)
        b.charge(0.05)
        b.charge(0.20)
        assert b.spent_today() == pytest.approx(0.25)

    def test_refuses_when_quote_would_breach_day_ceiling(self) -> None:
        b = CostBudget(max_per_task_usd=10.0, max_per_day_usd=1.0)
        b.charge(0.95)
        with pytest.raises(CostCeilingError) as ei:
            b.check(_quote(0.10))
        assert ei.value.kind == "per_day"

    def test_allows_when_quote_just_fits_day_ceiling(self) -> None:
        b = CostBudget(max_per_task_usd=10.0, max_per_day_usd=1.0)
        b.charge(0.50)
        b.check(_quote(0.50))  # 0.50 + 0.50 = 1.0; not strictly above.

    def test_per_task_checked_before_per_day(self) -> None:
        # When both ceilings would be breached, per_task wins because
        # it's checked first. This makes refusal kinds deterministic.
        b = CostBudget(max_per_task_usd=0.01, max_per_day_usd=0.01)
        b.charge(0.005)
        with pytest.raises(CostCeilingError) as ei:
            b.check(_quote(0.50))
        assert ei.value.kind == "per_task"

    def test_negative_charge_rejected(self) -> None:
        b = CostBudget(max_per_task_usd=1.0, max_per_day_usd=5.0)
        with pytest.raises(ValueError):
            b.charge(-0.01)


class TestDayRollover:
    def test_rollover_resets_counter(self) -> None:
        b = CostBudget(max_per_task_usd=10.0, max_per_day_usd=1.0)
        # Spend on day N.
        now = datetime(2026, 5, 22, 23, 59, tzinfo=UTC)
        b.charge(0.99, now=now)
        assert b.spent_today(now=now) == pytest.approx(0.99)

        # Cross midnight UTC — counter resets.
        tomorrow = now + timedelta(hours=2)
        assert b.spent_today(now=tomorrow) == 0.0

    def test_rollover_lets_a_previously_refused_call_through(self) -> None:
        b = CostBudget(max_per_task_usd=10.0, max_per_day_usd=1.0)
        now = datetime(2026, 5, 22, 23, 59, tzinfo=UTC)
        b.charge(0.95, now=now)
        with pytest.raises(CostCeilingError):
            b.check(_quote(0.20), now=now)
        tomorrow = now + timedelta(days=1, hours=1)
        # New day, counter zero — same quote passes.
        b.check(_quote(0.20), now=tomorrow)

    def test_explicit_reset(self) -> None:
        b = CostBudget(max_per_task_usd=10.0, max_per_day_usd=1.0)
        b.charge(0.99)
        b.reset()
        assert b.spent_today() == 0.0
        b.check(_quote(0.50))


class TestCostCeilingError:
    def test_has_kind_attribute(self) -> None:
        e = CostCeilingError("x", kind="per_task")
        assert e.kind == "per_task"

    def test_is_runtime_error(self) -> None:
        e = CostCeilingError("x", kind="per_day")
        assert isinstance(e, RuntimeError)
