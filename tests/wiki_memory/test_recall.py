"""
Tests for `wiki_memory.recall` — token-budgeted retrieval skeleton.
"""

from __future__ import annotations

from wiki_memory.content_store import StoredChunk
from wiki_memory.recall import recall_leaves


def _mk_chunk(idx: int, tokens: int, source_id: str = "src-a") -> StoredChunk:
    return StoredChunk(
        sha256=f"{idx:064x}",
        source_id=source_id,
        chunk_index=idx,
        body=f"chunk-{idx}-body",
        token_count=tokens,
        created_at="2026-05-22T00:00:00Z",
    )


class TestBudget:
    def test_empty_input_returns_empty_bundle(self) -> None:
        bundle = recall_leaves([], token_budget=1000)
        assert bundle.chunks == []
        assert bundle.total_tokens == 0
        assert bundle.truncated is False

    def test_under_budget_returns_all(self) -> None:
        chunks = [_mk_chunk(i, 100) for i in range(5)]
        bundle = recall_leaves(chunks, token_budget=1000)
        assert len(bundle.chunks) == 5
        assert bundle.total_tokens == 500
        assert bundle.truncated is False

    def test_over_budget_truncates(self) -> None:
        chunks = [_mk_chunk(i, 100) for i in range(20)]
        bundle = recall_leaves(chunks, token_budget=350)
        # 3 chunks * 100 = 300, 4th would push to 400 > 350
        assert len(bundle.chunks) == 3
        assert bundle.total_tokens == 300
        assert bundle.truncated is True

    def test_zero_budget_returns_empty_but_marks_truncated_if_input(self) -> None:
        chunks = [_mk_chunk(0, 100)]
        bundle = recall_leaves(chunks, token_budget=0)
        assert bundle.chunks == []
        assert bundle.truncated is True


class TestOrdering:
    def test_order_by_chunk_index_ascending(self) -> None:
        chunks = [_mk_chunk(i, 50) for i in (5, 1, 3, 0, 2, 4)]
        bundle = recall_leaves(chunks, token_budget=1000)
        assert [c.chunk_index for c in bundle.chunks] == [0, 1, 2, 3, 4, 5]

    def test_truncation_preserves_in_order_prefix(self) -> None:
        chunks = [_mk_chunk(i, 100) for i in range(5)]
        bundle = recall_leaves(chunks, token_budget=250)
        # Selects chunks 0 and 1 (sum 200), drops the rest.
        assert [c.chunk_index for c in bundle.chunks] == [0, 1]


class TestSkipOversizedSingle:
    def test_chunk_larger_than_budget_skipped(self) -> None:
        """One huge chunk + several small. The huge one is skipped (it
        alone exceeds the budget), the smalls are selected greedily."""
        huge = _mk_chunk(0, 1000)
        smalls = [_mk_chunk(i, 50) for i in range(1, 5)]
        chunks = [huge, *smalls]
        bundle = recall_leaves(chunks, token_budget=500)
        assert huge not in bundle.chunks
        assert all(c in bundle.chunks for c in smalls)
        assert bundle.total_tokens == 200
        assert bundle.truncated is True
