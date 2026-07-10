"""
Tests for `wiki_memory.scoring` — node score composition per PRD-004 FR-5.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from wiki_memory.scoring import (
    DEFAULT_RECENCY_HALFLIFE_DAYS,
    ScoreInputs,
    ScoreWeights,
    pagerank_proxy_score,
    recency_score,
    reuse_score,
    score_node,
)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class TestRecency:
    def test_fresh_chunk_scores_near_one(self) -> None:
        now = datetime.now(UTC)
        s = recency_score(_iso(now), now=now)
        assert s == pytest.approx(1.0, abs=0.01)

    def test_halflife_old_scores_half(self) -> None:
        now = datetime.now(UTC)
        old = now - timedelta(days=DEFAULT_RECENCY_HALFLIFE_DAYS)
        s = recency_score(_iso(old), now=now)
        assert s == pytest.approx(0.5, abs=0.01)

    def test_two_halflives_old_scores_quarter(self) -> None:
        now = datetime.now(UTC)
        old = now - timedelta(days=DEFAULT_RECENCY_HALFLIFE_DAYS * 2)
        s = recency_score(_iso(old), now=now)
        assert s == pytest.approx(0.25, abs=0.01)

    def test_future_dates_clamp_to_one(self) -> None:
        now = datetime.now(UTC)
        future = now + timedelta(days=10)
        s = recency_score(_iso(future), now=now)
        # Future ages clamp to age_days=0 → score=1.0
        assert s == pytest.approx(1.0, abs=0.01)


class TestReuse:
    def test_zero_reuse_returns_zero(self) -> None:
        assert reuse_score(0, tree_max=10) == 0.0

    def test_zero_tree_max_returns_zero(self) -> None:
        assert reuse_score(5, tree_max=0) == 0.0

    def test_equal_to_max_scores_one(self) -> None:
        # log1p(10) / log1p(10) = 1.0
        assert reuse_score(10, tree_max=10) == pytest.approx(1.0)

    def test_log_compression_dampens_high_counts(self) -> None:
        # 100 reuse vs 50 reuse should not be 2x — log compresses.
        high = reuse_score(100, tree_max=200)
        low = reuse_score(50, tree_max=200)
        assert high < low * 2

    def test_score_in_unit_interval(self) -> None:
        s = reuse_score(7, tree_max=10)
        assert 0.0 <= s <= 1.0


class TestPagerankProxy:
    def test_zero_in_degree_returns_zero(self) -> None:
        assert pagerank_proxy_score(0, tree_max=5) == 0.0

    def test_zero_tree_max_returns_zero(self) -> None:
        assert pagerank_proxy_score(3, tree_max=0) == 0.0

    def test_equal_to_max_scores_one(self) -> None:
        assert pagerank_proxy_score(5, tree_max=5) == pytest.approx(1.0)

    def test_clamps_overflow(self) -> None:
        # Caller passed a stale tree_max — score still clamps to 1.0.
        assert pagerank_proxy_score(10, tree_max=5) == pytest.approx(1.0)


class TestScoreNode:
    def test_fresh_no_reuse_no_pagerank_scores_recency_share(self) -> None:
        now = datetime.now(UTC)
        inputs = ScoreInputs(
            created_at_iso=_iso(now),
            reuse_count=0,
            in_degree=0,
            tree_max_reuse=1,
            tree_max_in_degree=1,
        )
        s = score_node(inputs, now=now)
        # recency=1, reuse=0, pagerank=0 → weighted=0.5*1=0.5; total_w=1.0 → 0.5
        assert s == pytest.approx(0.5, abs=0.001)

    def test_perfectly_recent_high_reuse_high_pagerank_caps_at_one(self) -> None:
        now = datetime.now(UTC)
        inputs = ScoreInputs(
            created_at_iso=_iso(now),
            reuse_count=10,
            in_degree=10,
            tree_max_reuse=10,
            tree_max_in_degree=10,
        )
        s = score_node(inputs, now=now)
        # All three signals = 1 → weighted_avg of 1.0 → clamp at 1.0
        assert s == pytest.approx(1.0, abs=0.001)

    def test_custom_weights_respected(self) -> None:
        now = datetime.now(UTC)
        inputs = ScoreInputs(
            created_at_iso=_iso(now),
            reuse_count=0,
            in_degree=10,
            tree_max_reuse=1,
            tree_max_in_degree=10,
        )
        # All recency weight → reuse & pagerank shouldn't move the needle
        weights = ScoreWeights(recency=1.0, reuse=0.0, pagerank=0.0)
        s = score_node(inputs, weights=weights, now=now)
        assert s == pytest.approx(1.0, abs=0.001)

    def test_zero_weights_returns_zero(self) -> None:
        inputs = ScoreInputs(created_at_iso=_iso(datetime.now(UTC)))
        s = score_node(inputs, weights=ScoreWeights(0.0, 0.0, 0.0))
        assert s == 0.0
