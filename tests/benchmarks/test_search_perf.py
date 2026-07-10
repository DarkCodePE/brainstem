"""
Performance benchmarks for the retrieval surface (#132).

Establishes baselines for the M3.6 ranker stack (substring / FTS5 /
vector) at 500 and 5000 chunks. The CI workflow in
``.github/workflows/benchmark.yml`` runs these weekly and stores the
history via ``benchmark-action/github-action-benchmark`` so a 2x
regression triggers an automatic PR alert.

These are NOT part of the normal pytest run (``--benchmark-skip`` is
default per ``pyproject.toml``). Run explicitly with::

    pytest tests/benchmarks/ --benchmark-only --benchmark-min-rounds=5
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from wiki_memory.content_store import ContentStore
from wiki_memory.embedder import encode_vector


def _run_search(db_path: Path, mode: str, query: str) -> int:
    """Open store, run one search, close. Wrapper for pytest-benchmark
    which only times sync callables. Returns the hit count so the
    benchmark's "result" stat has a meaningful payload."""

    async def _inner() -> int:
        store = ContentStore(db_path)
        await store.init()
        try:
            if mode == "fts":
                hits = await store.search_fts(query, limit=50)
            elif mode == "substring":
                hits = await store.search_substring(query, limit=50)
            else:
                raise ValueError(mode)
            return len(hits)
        finally:
            await store.close()

    return asyncio.run(_inner())


def _run_vector_search(db_path: Path, query_text: str) -> int:
    """Vector path — uses a synthetic 4-dim query vector matching the
    fixture's embedding scheme. Catches regressions in the cosine
    brute-force ranker without needing Ollama."""

    async def _inner() -> int:
        store = ContentStore(db_path)
        await store.init()
        try:
            # Same scheme as the fixture (source 0)
            vec = [0.0, 0.0, 0.0, 0.0]
            hits = await store.search_vector(
                encode_vector(vec),
                query_dim=4,
                model="bench-stub-4d",
                limit=50,
            )
            return len(hits)
        finally:
            await store.close()

    return asyncio.run(_inner())


class TestSearchAt500:
    def test_bench_fts_at_500_chunks(self, benchmark, populated_500: Path) -> None:
        """Baseline FTS5 at production-scale (500 chunks). Gate target:
        <50ms p95. Anything above 200ms is a regression."""
        result = benchmark(_run_search, populated_500, "fts", "alpha")
        assert result >= 1

    def test_bench_substring_at_500_chunks(self, benchmark, populated_500: Path) -> None:
        """LIKE-based substring at 500 chunks — kept as a comparison
        baseline for the FTS5 win. Substring should always be slower
        per query at scale (no index), but the absolute number sets
        the "good enough" floor."""
        result = benchmark(_run_search, populated_500, "substring", "alpha")
        assert result >= 1


class TestSearchAt5000:
    """10x current scale. ADR-020 claims brute-force is fine up to
    100k chunks; these benchmarks pin the actual numbers so the claim
    survives auditing as the wiki grows."""

    def test_bench_fts_at_5000_chunks(self, benchmark, populated_5000: Path) -> None:
        """FTS5 at 10x current scale. Target: <100ms p95."""
        result = benchmark(_run_search, populated_5000, "fts", "alpha")
        assert result >= 1

    def test_bench_substring_at_5000_chunks(self, benchmark, populated_5000: Path) -> None:
        """Substring at 10x scale — where the FTS5 advantage starts to
        be substantial. Numbers here justify the FTS5 default switch
        (#118)."""
        result = benchmark(_run_search, populated_5000, "substring", "alpha")
        assert result >= 1


class TestVectorSearch:
    def test_bench_vector_at_500_chunks(self, benchmark, populated_500_with_vectors: Path) -> None:
        """Cosine brute-force vector search at 500 chunks. Synthetic
        4-dim embeddings so the test runs without Ollama; production
        uses 1024-dim bge-m3 which is ~256x more arithmetic per
        comparison. Extrapolate cautiously."""
        result = benchmark(_run_vector_search, populated_500_with_vectors, "alpha")
        assert result >= 1
