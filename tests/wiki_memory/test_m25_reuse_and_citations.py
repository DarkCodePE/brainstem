"""
M2.5 tests — reuse_count tracking (ADR-027 #155) and citation in-degree
(ADR-027 #156) in the content store.

These exercise the two scoring-input signals that were previously
impossible to compute (no reuse column, no citation persistence).
"""

from __future__ import annotations

import aiosqlite
import pytest

from wiki_memory.chunker import Chunk, chunk_text
from wiki_memory.content_store import ContentStore


def _mk_chunks(text: str) -> list[Chunk]:
    return chunk_text(text, target_tokens=5, hard_cap_tokens=20)


def _doc(n: int) -> str:
    """Build a document of exactly `n` chunks. Each paragraph is long
    enough (~10 tokens > the 5-token target) to flush as its own chunk."""
    para = "word word word word word word word word."
    return "\n\n".join(f"p{i} {para}" for i in range(n))


async def _seed(store: ContentStore, source_id: str, text: str) -> list[str]:
    chunks = _mk_chunks(text)
    await store.insert_many(source_id=source_id, chunks=chunks)
    return [c.sha256 for c in chunks]


class TestReuseCount:
    @pytest.mark.asyncio
    async def test_new_chunk_starts_at_zero(self, content_store) -> None:
        shas = await _seed(content_store, "src", "alpha one.\n\nbeta two.")
        chunk = await content_store.get(shas[0])
        assert chunk is not None
        assert chunk.reuse_count == 0
        assert await content_store.max_reuse() == 0

    @pytest.mark.asyncio
    async def test_increment_bumps_only_named_shas(self, content_store) -> None:
        shas = await _seed(content_store, "src", _doc(3))
        changed = await content_store.increment_reuse([shas[0], shas[2]])
        assert changed == 2
        assert (await content_store.get(shas[0])).reuse_count == 1
        assert (await content_store.get(shas[1])).reuse_count == 0
        assert (await content_store.get(shas[2])).reuse_count == 1

    @pytest.mark.asyncio
    async def test_increment_dedups_within_call(self, content_store) -> None:
        shas = await _seed(content_store, "src", "only one.")
        # Same sha passed twice in one call counts once.
        await content_store.increment_reuse([shas[0], shas[0], shas[0]])
        assert (await content_store.get(shas[0])).reuse_count == 1

    @pytest.mark.asyncio
    async def test_repeated_calls_accumulate(self, content_store) -> None:
        shas = await _seed(content_store, "src", "only one.")
        for _ in range(5):
            await content_store.increment_reuse([shas[0]])
        assert (await content_store.get(shas[0])).reuse_count == 5
        assert await content_store.max_reuse() == 5

    @pytest.mark.asyncio
    async def test_increment_empty_is_noop(self, content_store) -> None:
        assert await content_store.increment_reuse([]) == 0

    @pytest.mark.asyncio
    async def test_increment_unknown_sha_changes_nothing(self, content_store) -> None:
        assert await content_store.increment_reuse(["deadbeef" * 8]) == 0

    @pytest.mark.asyncio
    async def test_reuse_survives_list_by_source(self, content_store) -> None:
        shas = await _seed(content_store, "src", _doc(2))
        await content_store.increment_reuse([shas[1]])
        rows = {c.sha256: c.reuse_count for c in await content_store.list_by_source("src")}
        assert rows[shas[1]] == 1


class TestLegacyMigration:
    @pytest.mark.asyncio
    async def test_reuse_column_added_to_legacy_db(self, tmp_path) -> None:
        """A pre-#155 DB has no reuse_count column. init() must ALTER it
        in and default existing rows to 0 (not crash)."""
        db_path = tmp_path / "legacy.db"
        # Hand-build a legacy chunks table WITHOUT reuse_count.
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "CREATE TABLE chunks (sha256 TEXT PRIMARY KEY, source_id TEXT NOT NULL,"
                " chunk_index INTEGER NOT NULL, body TEXT NOT NULL,"
                " token_count INTEGER NOT NULL, created_at TEXT NOT NULL)"
            )
            await db.execute(
                "INSERT INTO chunks VALUES ('s1','src',0,'legacy body',3,'2024-01-01T00:00:00Z')"
            )
            await db.commit()

        store = ContentStore(db_path)
        await store.init()
        try:
            chunk = await store.get("s1")
            assert chunk is not None
            assert chunk.reuse_count == 0  # backfilled
            # And the column is writable.
            await store.increment_reuse(["s1"])
            assert (await store.get("s1")).reuse_count == 1
        finally:
            await store.close()


class TestCitations:
    @pytest.mark.asyncio
    async def test_record_and_count_in_degree(self, content_store) -> None:
        shas = await _seed(content_store, "src", _doc(3))
        # Two summaries both cite chunk shas[0]; only one cites shas[1].
        await content_store.record_citations(summary_sha256="sum-A", cited_shas=[shas[0], shas[1]])
        await content_store.record_citations(summary_sha256="sum-B", cited_shas=[shas[0]])
        degrees = await content_store.in_degrees(shas)
        assert degrees[shas[0]] == 2
        assert degrees[shas[1]] == 1
        assert shas[2] not in degrees  # never cited -> absent (caller treats as 0)

    @pytest.mark.asyncio
    async def test_resealing_replaces_not_doubles(self, content_store) -> None:
        shas = await _seed(content_store, "src", _doc(2))
        await content_store.record_citations(summary_sha256="sum-A", cited_shas=[shas[0]])
        # Re-seal the same summary citing the same chunk — must stay in-degree 1.
        await content_store.record_citations(summary_sha256="sum-A", cited_shas=[shas[0]])
        assert (await content_store.in_degrees([shas[0]]))[shas[0]] == 1

    @pytest.mark.asyncio
    async def test_max_in_degree(self, content_store) -> None:
        shas = await _seed(content_store, "src", _doc(2))
        assert await content_store.max_in_degree() == 0
        await content_store.record_citations(summary_sha256="s1", cited_shas=[shas[0]])
        await content_store.record_citations(summary_sha256="s2", cited_shas=[shas[0]])
        await content_store.record_citations(summary_sha256="s3", cited_shas=[shas[1]])
        assert await content_store.max_in_degree() == 2  # shas[0] cited by 2

    @pytest.mark.asyncio
    async def test_in_degrees_empty_input(self, content_store) -> None:
        assert await content_store.in_degrees([]) == {}

    @pytest.mark.asyncio
    async def test_delete_by_source_cleans_dangling_citations(self, content_store) -> None:
        shas = await _seed(content_store, "src", _doc(2))
        await content_store.record_citations(summary_sha256="sum-A", cited_shas=shas)
        assert await content_store.in_degrees(shas)  # non-empty
        deleted = await content_store.delete_by_source("src")
        assert deleted == len(shas)
        # Citations pointing at now-deleted chunks are gone.
        assert await content_store.in_degrees(shas) == {}
        assert await content_store.max_in_degree() == 0
