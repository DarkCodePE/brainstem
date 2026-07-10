"""
M2.5 tests — relevance-ordered recall (ADR-027 #157).

`recall_leaves` must: (a) stay legacy index-order when no scores are
given, and (b) when scores ARE given, select the highest-scored chunks
under the budget while still PRESENTING them in chunk_index order.
`build_chunk_scores` must combine recency × reuse × pagerank correctly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from wiki_memory.content_store import StoredChunk
from wiki_memory.recall import build_chunk_scores, recall_leaves


def _mk_chunk(
    idx: int, tokens: int, *, reuse: int = 0, created: str = "2026-05-22T00:00:00Z"
) -> StoredChunk:
    return StoredChunk(
        sha256=f"{idx:064x}",
        source_id="src-a",
        chunk_index=idx,
        body=f"chunk-{idx}",
        token_count=tokens,
        created_at=created,
        reuse_count=reuse,
    )


class TestLegacyPathUnchanged:
    def test_no_scores_is_index_order(self) -> None:
        chunks = [_mk_chunk(i, 50) for i in (5, 1, 3, 0, 2, 4)]
        bundle = recall_leaves(chunks, token_budget=1000)
        assert [c.chunk_index for c in bundle.chunks] == [0, 1, 2, 3, 4, 5]

    def test_no_scores_truncates_in_index_order(self) -> None:
        chunks = [_mk_chunk(i, 100) for i in range(5)]
        bundle = recall_leaves(chunks, token_budget=250)
        assert [c.chunk_index for c in bundle.chunks] == [0, 1]


class TestScoredSelection:
    def test_high_score_survives_truncation(self) -> None:
        # Four 100-token chunks, budget fits 2. The relevant ones are the
        # LAST two by index — legacy order would drop them; scored keeps them.
        chunks = [_mk_chunk(i, 100) for i in range(4)]
        scores = {
            chunks[0].sha256: 0.1,
            chunks[1].sha256: 0.2,
            chunks[2].sha256: 0.9,
            chunks[3].sha256: 0.8,
        }
        bundle = recall_leaves(chunks, token_budget=200, scores=scores)
        # Selected the two highest-scored (idx 2 and 3)...
        assert {c.chunk_index for c in bundle.chunks} == {2, 3}
        # ...but PRESENTED in chunk_index order.
        assert [c.chunk_index for c in bundle.chunks] == [2, 3]
        assert bundle.truncated is True

    def test_presentation_always_index_order(self) -> None:
        chunks = [_mk_chunk(i, 50) for i in range(5)]
        # Reverse-relevance: idx 4 most relevant, idx 0 least.
        scores = {c.sha256: (5 - c.chunk_index) / 5 for c in chunks}
        bundle = recall_leaves(chunks, token_budget=1000, scores=scores)
        # All fit, presented ascending.
        assert [c.chunk_index for c in bundle.chunks] == [0, 1, 2, 3, 4]

    def test_missing_score_treated_as_zero(self) -> None:
        chunks = [_mk_chunk(i, 100) for i in range(3)]
        # Only idx 2 has a score; the others default to 0.0 and lose.
        scores = {chunks[2].sha256: 0.9}
        bundle = recall_leaves(chunks, token_budget=100, scores=scores)
        assert [c.chunk_index for c in bundle.chunks] == [2]

    def test_tie_breaks_by_index(self) -> None:
        chunks = [_mk_chunk(i, 100) for i in range(3)]
        scores = {c.sha256: 0.5 for c in chunks}  # all equal
        bundle = recall_leaves(chunks, token_budget=100, scores=scores)
        # Deterministic: lowest index wins the single slot.
        assert [c.chunk_index for c in bundle.chunks] == [0]


class TestBuildChunkScores:
    def test_scores_in_unit_interval(self) -> None:
        chunks = [_mk_chunk(i, 10, reuse=i) for i in range(4)]
        scores = build_chunk_scores(chunks, max_reuse=3, max_in_degree=1)
        assert all(0.0 <= v <= 1.0 for v in scores.values())
        assert set(scores) == {c.sha256 for c in chunks}

    def test_more_reuse_scores_higher(self) -> None:
        now = datetime(2026, 5, 23, tzinfo=UTC)
        low = _mk_chunk(0, 10, reuse=0, created="2026-05-22T00:00:00Z")
        high = _mk_chunk(1, 10, reuse=10, created="2026-05-22T00:00:00Z")
        scores = build_chunk_scores([low, high], max_reuse=10, max_in_degree=1, now=now)
        assert scores[high.sha256] > scores[low.sha256]

    def test_newer_scores_higher(self) -> None:
        now = datetime(2026, 5, 23, tzinfo=UTC)
        old = _mk_chunk(0, 10, created=(now - timedelta(days=120)).isoformat())
        fresh = _mk_chunk(1, 10, created=(now - timedelta(hours=1)).isoformat())
        scores = build_chunk_scores([old, fresh], max_reuse=1, max_in_degree=1, now=now)
        assert scores[fresh.sha256] > scores[old.sha256]

    def test_in_degree_lifts_score(self) -> None:
        now = datetime(2026, 5, 23, tzinfo=UTC)
        a = _mk_chunk(0, 10, created="2026-05-22T00:00:00Z")
        b = _mk_chunk(1, 10, created="2026-05-22T00:00:00Z")
        scores = build_chunk_scores(
            [a, b], max_reuse=1, in_degrees={a.sha256: 5}, max_in_degree=5, now=now
        )
        assert scores[a.sha256] > scores[b.sha256]
