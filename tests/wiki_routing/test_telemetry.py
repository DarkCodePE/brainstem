"""
Tests for `wiki_routing.telemetry.RouterTelemetry`.
"""

from __future__ import annotations

import pytest

from wiki_routing.telemetry import RouterTelemetry


@pytest.fixture
def tel(tmp_path):
    t = RouterTelemetry(db_path=tmp_path / "tel.db")
    yield t
    t.close()


def test_record_and_rolling_cost(tel):
    tel.record(tier="fast", backend_label="openrouter:deepseek", cost_usd=0.001, success=True)
    tel.record(tier="reasoning", backend_label="anthropic:sonnet", cost_usd=0.05, success=True)
    cost = tel.rolling_cost_usd(window_hours=24)
    assert cost == pytest.approx(0.051, rel=1e-3)


def test_tier_distribution_groups_correctly(tel):
    tel.record(tier="fast", backend_label="a", cost_usd=0.01, success=True)
    tel.record(tier="fast", backend_label="a", cost_usd=0.01, success=True)
    tel.record(tier="reasoning", backend_label="b", cost_usd=0.10, success=False)

    tiers = tel.tier_distribution(window_hours=24)
    by_tier = {s.tier: s for s in tiers}
    assert by_tier["fast"].calls == 2
    assert by_tier["fast"].success_rate == 1.0
    assert by_tier["reasoning"].calls == 1
    assert by_tier["reasoning"].success_rate == 0.0


def test_total_calls_lifetime(tel):
    for _ in range(7):
        tel.record(tier="fast", backend_label="a", cost_usd=0.001, success=True)
    assert tel.total_calls() == 7


def test_clear_removes_all(tel):
    tel.record(tier="fast", backend_label="a", cost_usd=0.001, success=True)
    tel.record(tier="reasoning", backend_label="b", cost_usd=0.01, success=True)
    n = tel.clear()
    assert n == 2
    assert tel.total_calls() == 0


def test_empty_telemetry_is_safe(tel):
    assert tel.rolling_cost_usd() == 0.0
    assert tel.tier_distribution() == []


def test_persists_across_reopen(tmp_path):
    path = tmp_path / "tel.db"
    t1 = RouterTelemetry(db_path=path)
    t1.record(tier="fast", backend_label="a", cost_usd=0.005, success=True)
    t1.close()

    t2 = RouterTelemetry(db_path=path)
    try:
        assert t2.total_calls() == 1
        assert t2.rolling_cost_usd() == pytest.approx(0.005, rel=1e-3)
    finally:
        t2.close()
