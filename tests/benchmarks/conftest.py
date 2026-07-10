"""Fixtures for M3.7 performance benchmarks (#132).

The benchmark fixtures populate real SQLite content_stores at different
scales (500 / 5000 chunks) and hold them open across the benchmark
session. The async/sync bridge uses ``asyncio.run`` inside the timed
callable so pytest-benchmark's stats apply to the full end-to-end
call (init handshake amortised across iterations via a module-scoped
fixture).

Run with::

    pytest tests/benchmarks/ --benchmark-only

Skip during normal pytest runs (the default ``--benchmark-skip``
configured in ``pyproject.toml``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path

import pytest

from wiki_memory.chunker import chunk_text
from wiki_memory.content_store import ContentStore
from wiki_memory.embedder import encode_vector


def _populate_store_sync(db_path: Path, *, n_sources: int, with_embeddings: bool) -> None:
    """Synchronous seed helper. Each source gets a deterministic body
    derived from its index so tokens repeat across sources (FTS5 ranker
    has something meaningful to rank)."""

    async def _seed() -> None:
        store = ContentStore(db_path)
        await store.init()
        try:
            for i in range(n_sources):
                # 5 paragraphs per source — produces ~5 chunks.
                paras = [
                    f"source {i} paragraph {p}: topic alpha beta gamma delta number {i}."
                    for p in range(5)
                ]
                chunks = chunk_text("\n\n".join(paras), target_tokens=15, hard_cap_tokens=40)
                await store.insert_many(source_id=f"src-{i:05d}.md", chunks=chunks)
                if with_embeddings:
                    # Synthetic 4-dim embeddings derived from source index
                    # so tests are deterministic without touching Ollama.
                    vec = [float((i * (p + 1)) % 17) for p in range(4)]
                    for c in chunks:
                        await store.set_embedding(
                            c.sha256,
                            vector=encode_vector(vec),
                            model="bench-stub-4d",
                            dim=4,
                        )
        finally:
            await store.close()

    asyncio.run(_seed())


@pytest.fixture(scope="module")
def populated_500(tmp_path_factory) -> Generator[Path, None, None]:
    """500-chunk fixture (~100 sources × 5 chunks each). Reflects SBW's
    current production state (228 files / 477 chunks)."""
    db_path = tmp_path_factory.mktemp("bench500") / "content.db"
    _populate_store_sync(db_path, n_sources=100, with_embeddings=False)
    yield db_path


@pytest.fixture(scope="module")
def populated_5000(tmp_path_factory) -> Generator[Path, None, None]:
    """5000-chunk fixture (~1000 sources × 5). 10x current scale — the
    target for "still snappy" performance per ADR-020."""
    db_path = tmp_path_factory.mktemp("bench5000") / "content.db"
    _populate_store_sync(db_path, n_sources=1000, with_embeddings=False)
    yield db_path


@pytest.fixture(scope="module")
def populated_500_with_vectors(tmp_path_factory) -> Generator[Path, None, None]:
    """500 chunks with synthetic 4-dim embeddings — exercises the cosine
    brute-force path without needing Ollama. The vector dim is small
    on purpose so per-call cost reflects the SQL + Python loop overhead,
    not the cosine math."""
    db_path = tmp_path_factory.mktemp("bench500vec") / "content.db"
    _populate_store_sync(db_path, n_sources=100, with_embeddings=True)
    yield db_path
