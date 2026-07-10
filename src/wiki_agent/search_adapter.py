"""
Bridge from the canonical `wiki_core.Search` protocol to the existing
hybrid search in `wiki_agent.tools.search_wiki_index` (keyword + fastembed
semantic per [ADR-005](../../docs/ADR-005-deep-agents-harness-migration.md)).

`HybridSearchAdapter` is intentionally thin — it doesn't add new search
behavior, it just normalises the output shape to a sequence of
`wiki_core.SearchHit` value objects so callers can write against the
protocol surface and swap implementations later (M2 Memory Tree retrieval
per [PRD-004](../../docs/PRD-004-memory-tree.md) will satisfy the same
interface).

`StaticSearchAdapter` is a deterministic stub for tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from wiki_core.protocols import SearchHit


_DEFAULT_CATEGORY = "concepts"


def _make_hit(
    page_path: str,
    title: str,
    snippet: str,
    score: float,
    components: dict[str, float] | None = None,
) -> SearchHit:
    from wiki_core.protocols import PageRef, SearchHit

    category = _infer_category(page_path)
    return SearchHit(
        ref=PageRef(page_path=page_path, category=category),
        title=title,
        snippet=snippet,
        score=score,
        score_components=components or {"hybrid": score},
    )


def _infer_category(page_path: str) -> str:
    """Pull `sources` / `entities` / `concepts` / … from a vault path."""
    parts = page_path.split("/")
    if len(parts) >= 2 and parts[0] in {
        "wiki",
        "knowledge-base",
    }:  # tolerate both prefixes
        candidate = (
            parts[1] if parts[0] == "wiki" else parts[2] if len(parts) >= 3 else _DEFAULT_CATEGORY
        )
    else:
        candidate = parts[0] if parts else _DEFAULT_CATEGORY
    allowed = {"sources", "entities", "concepts", "answers", "synthesis", "outputs", "observations"}
    return candidate if candidate in allowed else _DEFAULT_CATEGORY


class HybridSearchAdapter:
    """Wraps the legacy `search_wiki_index` LangChain tool (which returns
    a JSON-shaped string) and translates to `Sequence[SearchHit]`.

    The constructor takes a single sync callable returning a JSON string
    so that the adapter is testable without LangChain or fastembed
    imports. The agent passes the actual tool in via DI in
    `wiki_agent.agent.create_wiki_agent`.
    """

    def __init__(self, search_callable: callable) -> None:  # type: ignore[type-arg]
        self._search = search_callable

    async def search_index(
        self,
        query: str,
        *,
        limit: int = 10,
        categories: Sequence[str] | None = None,
    ) -> Sequence[SearchHit]:
        import json

        raw = self._search(query)
        rows = json.loads(raw) if isinstance(raw, str) else raw
        results: list[SearchHit] = []
        for row in rows[:limit]:
            page_path = str(row.get("page_path") or row.get("path") or "")
            if not page_path:
                continue
            if categories and _infer_category(page_path) not in set(categories):
                continue
            results.append(
                _make_hit(
                    page_path=page_path,
                    title=str(row.get("title") or page_path),
                    snippet=str(row.get("summary") or row.get("snippet") or ""),
                    score=float(row.get("score") or 0.0),
                    components=cast(
                        "dict[str, float] | None",
                        row.get("score_components") or row.get("components"),
                    ),
                )
            )
        return results

    async def search_text(
        self,
        text: str,
        *,
        limit: int = 10,
        threshold: float | None = None,
    ) -> Sequence[SearchHit]:
        # The legacy tool fuses keyword+semantic in one call; we route both
        # surfaces to it. M2 Memory Tree will split these into two backends
        # behind the same façade.
        hits = await self.search_index(text, limit=limit)
        if threshold is None:
            return hits
        return [h for h in hits if h.score >= threshold]


class StaticSearchAdapter:
    """In-memory `Search` for tests. Construct with a list of (PageRef,
    title, snippet, score) tuples; queries are matched by substring on
    title and snippet."""

    def __init__(self, hits: Sequence[SearchHit]) -> None:
        self._hits = list(hits)

    async def search_index(
        self,
        query: str,
        *,
        limit: int = 10,
        categories: Sequence[str] | None = None,
    ) -> Sequence[SearchHit]:
        q = query.lower()
        matched = [h for h in self._hits if q in h.title.lower() or q in h.snippet.lower()]
        if categories:
            cset = set(categories)
            matched = [h for h in matched if h.ref.category in cset]
        return matched[:limit]

    async def search_text(
        self,
        text: str,
        *,
        limit: int = 10,
        threshold: float | None = None,
    ) -> Sequence[SearchHit]:
        hits = await self.search_index(text, limit=limit)
        if threshold is None:
            return hits
        return [h for h in hits if h.score >= threshold]


__all__ = ["HybridSearchAdapter", "StaticSearchAdapter"]
