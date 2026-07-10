"""
Behavioural tests for `HybridSearchAdapter` and `StaticSearchAdapter`.
"""

from __future__ import annotations

import json

import pytest

from wiki_agent.search_adapter import HybridSearchAdapter, StaticSearchAdapter


class TestStaticSearchAdapter:
    @pytest.mark.asyncio
    async def test_substring_match_on_title(self, search_hit_factory) -> None:
        hits = [
            search_hit_factory(title="LangGraph harness"),
            search_hit_factory(title="OpenHuman memory tree"),
        ]
        adapter = StaticSearchAdapter(hits)
        result = await adapter.search_index("memory")
        assert len(result) == 1
        assert "memory" in result[0].title.lower()

    @pytest.mark.asyncio
    async def test_category_filter(self, search_hit_factory) -> None:
        hits = [
            search_hit_factory(page_path="wiki/sources/a.md", category="sources"),
            search_hit_factory(page_path="wiki/concepts/b.md", category="concepts"),
        ]
        adapter = StaticSearchAdapter(hits)
        result = await adapter.search_index("example", categories=["sources"])
        assert len(result) == 1
        assert result[0].ref.category == "sources"

    @pytest.mark.asyncio
    async def test_threshold_filter(self, search_hit_factory) -> None:
        hits = [
            search_hit_factory(score=0.9),
            search_hit_factory(score=0.4),
            search_hit_factory(score=0.6),
        ]
        adapter = StaticSearchAdapter(hits)
        result = await adapter.search_text("example", threshold=0.5)
        assert len(result) == 2
        assert all(h.score >= 0.5 for h in result)

    @pytest.mark.asyncio
    async def test_limit_clamps_result_size(self, search_hit_factory) -> None:
        hits = [search_hit_factory(title=f"Hit {i}") for i in range(20)]
        adapter = StaticSearchAdapter(hits)
        result = await adapter.search_index("hit", limit=5)
        assert len(result) == 5


class TestHybridSearchAdapter:
    @pytest.mark.asyncio
    async def test_translates_legacy_json_to_search_hits(self) -> None:
        rows = [
            {
                "page_path": "wiki/concepts/llm-wiki.md",
                "title": "LLM Wiki",
                "summary": "Pattern for LLM knowledge bases",
                "score": 0.82,
                "score_components": {"keyword": 0.32, "semantic": 0.50},
            },
            {
                "page_path": "wiki/entities/karpathy.md",
                "title": "Karpathy",
                "summary": "ML engineer",
                "score": 0.41,
            },
        ]

        def fake_search(query: str) -> str:
            return json.dumps(rows)

        adapter = HybridSearchAdapter(fake_search)
        result = await adapter.search_index("llm")
        assert len(result) == 2
        assert result[0].title == "LLM Wiki"
        assert result[0].score == 0.82
        assert result[0].score_components == {"keyword": 0.32, "semantic": 0.50}
        assert result[0].ref.category == "concepts"
        assert result[1].ref.category == "entities"

    @pytest.mark.asyncio
    async def test_skips_rows_without_page_path(self) -> None:
        rows = [
            {"page_path": "wiki/x.md", "title": "X", "score": 0.5},
            {"title": "Y", "score": 0.4},  # no page_path → skipped
        ]

        def fake_search(q: str) -> str:
            return json.dumps(rows)

        adapter = HybridSearchAdapter(fake_search)
        result = await adapter.search_index("y")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_category_filter_applied_after_translation(self) -> None:
        rows = [
            {"page_path": "wiki/sources/a.md", "title": "A", "score": 0.5},
            {"page_path": "wiki/concepts/b.md", "title": "B", "score": 0.5},
        ]

        def fake_search(q: str) -> str:
            return json.dumps(rows)

        adapter = HybridSearchAdapter(fake_search)
        result = await adapter.search_index("a", categories=["sources"])
        assert len(result) == 1
        assert result[0].ref.category == "sources"

    @pytest.mark.asyncio
    async def test_search_text_threshold(self) -> None:
        rows = [
            {"page_path": "wiki/x.md", "title": "X", "score": 0.9},
            {"page_path": "wiki/y.md", "title": "Y", "score": 0.3},
        ]

        def fake_search(q: str) -> str:
            return json.dumps(rows)

        adapter = HybridSearchAdapter(fake_search)
        result = await adapter.search_text("anything", threshold=0.5)
        assert len(result) == 1
        assert result[0].title == "X"
