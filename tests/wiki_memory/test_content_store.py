"""
Tests for `wiki_memory.content_store`.
"""

from __future__ import annotations

import pytest

from wiki_memory.chunker import Chunk, chunk_text


def _mk_chunks(text: str, *, target: int = 50, cap: int = 100) -> list[Chunk]:
    return chunk_text(text, target_tokens=target, hard_cap_tokens=cap)


class TestInsert:
    @pytest.mark.asyncio
    async def test_insert_single_chunk(self, content_store) -> None:
        chunk = _mk_chunks("hello world")[0]
        ok = await content_store.insert(source_id="src-1", chunk=chunk)
        assert ok is True
        assert await content_store.count() == 1

    @pytest.mark.asyncio
    async def test_duplicate_insert_is_noop(self, content_store) -> None:
        chunk = _mk_chunks("repeated text")[0]
        first = await content_store.insert(source_id="src-1", chunk=chunk)
        second = await content_store.insert(source_id="src-1", chunk=chunk)
        assert first is True
        assert second is False
        assert await content_store.count() == 1

    @pytest.mark.asyncio
    async def test_get_returns_persisted_body(self, content_store) -> None:
        chunk = _mk_chunks("queryable body")[0]
        await content_store.insert(source_id="src-1", chunk=chunk)
        loaded = await content_store.get(chunk.sha256)
        assert loaded is not None
        assert loaded.body == chunk.body
        assert loaded.source_id == "src-1"
        assert loaded.token_count == chunk.token_count

    @pytest.mark.asyncio
    async def test_get_unknown_sha_returns_none(self, content_store) -> None:
        assert await content_store.get("0" * 64) is None


class TestBatchInsert:
    @pytest.mark.asyncio
    async def test_insert_many_atomic(self, content_store) -> None:
        text = "p1.\n\nparagraph two with a few more chars to fill it.\n\np3."
        chunks = _mk_chunks(text, target=10, cap=20)
        assert len(chunks) >= 2
        n = await content_store.insert_many(source_id="src-batch", chunks=chunks)
        assert n == len(chunks)
        assert await content_store.count() == len(chunks)

    @pytest.mark.asyncio
    async def test_insert_many_skips_duplicates(self, content_store) -> None:
        chunks = _mk_chunks("para a.\n\npara b.", target=10, cap=20)
        first = await content_store.insert_many(source_id="src-d", chunks=chunks)
        # Re-insert the same chunks → 0 new.
        second = await content_store.insert_many(source_id="src-d", chunks=chunks)
        assert first == len(chunks)
        assert second == 0


class TestListAndDelete:
    @pytest.mark.asyncio
    async def test_list_by_source_orders_by_chunk_index(self, content_store) -> None:
        chunks = _mk_chunks("paraone.\n\nparatwo.\n\nparathree.", target=5, cap=15)
        await content_store.insert_many(source_id="src-list", chunks=chunks)
        rows = await content_store.list_by_source("src-list")
        assert [r.chunk_index for r in rows] == sorted(r.chunk_index for r in rows)

    @pytest.mark.asyncio
    async def test_delete_by_source_removes_all_rows(self, content_store) -> None:
        chunks = _mk_chunks("a.\n\nb.", target=5, cap=15)
        await content_store.insert_many(source_id="to-delete", chunks=chunks)
        n = await content_store.delete_by_source("to-delete")
        assert n == len(chunks)
        assert await content_store.list_by_source("to-delete") == []

    @pytest.mark.asyncio
    async def test_delete_unknown_source_is_zero(self, content_store) -> None:
        assert await content_store.delete_by_source("never-existed") == 0


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_init_is_idempotent(self, content_store) -> None:
        # The fixture already inited; second init() must be a no-op.
        await content_store.init()
        assert await content_store.count() == 0


class TestSchemaVersionInvalidation:
    """Issue #127 sub-item 1 — Tolaria-inspired auto-invalidate on
    schema bump. Tolaria uses ``CACHE_VERSION = 14`` and bumps trigger
    a full rescan. SBW uses a meta-table marker for the same effect."""

    @pytest.mark.asyncio
    async def test_first_init_stamps_version(self, tmp_path) -> None:
        from wiki_memory.content_store import SCHEMA_VERSION, ContentStore

        store = ContentStore(tmp_path / "fresh.db")
        await store.init()
        try:
            stored = await store._meta_get("schema_version")
            assert stored == str(SCHEMA_VERSION)
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_lower_version_truncates_chunks(self, tmp_path) -> None:
        from wiki_memory.content_store import SCHEMA_VERSION, ContentStore

        store = ContentStore(tmp_path / "bumped.db")
        await store.init()
        chunk = chunk_text("about to be invalidated")[0]
        await store.insert(source_id="s", chunk=chunk)
        assert await store.count() == 1
        # Force the stored version down — simulate a code-side bump.
        await store._meta_set("schema_version", str(SCHEMA_VERSION - 1))
        await store.close()

        # Reopen: invalidation should fire and the chunk is gone.
        store2 = ContentStore(tmp_path / "bumped.db")
        await store2.init()
        try:
            assert await store2.count() == 0
            assert await store2._meta_get("schema_version") == str(SCHEMA_VERSION)
        finally:
            await store2.close()

    @pytest.mark.asyncio
    async def test_equal_or_higher_version_leaves_chunks_alone(self, tmp_path) -> None:
        from wiki_memory.content_store import ContentStore

        store = ContentStore(tmp_path / "stable.db")
        await store.init()
        chunk = chunk_text("safe content")[0]
        await store.insert(source_id="s", chunk=chunk)
        await store.close()

        # Second open with the SAME version — no truncation.
        store2 = ContentStore(tmp_path / "stable.db")
        await store2.init()
        try:
            assert await store2.count() == 1
        finally:
            await store2.close()


class TestVaultRootValidation:
    """Issue #127 sub-item 4 — Tolaria's cross-machine cache invalidation
    pattern. If a user moves their vault (or clones to a new machine
    with a different absolute path) the source_ids no longer match
    disk, so we log a warning to nudge a reseed."""

    @pytest.mark.asyncio
    async def test_first_init_stamps_vault_root(self, tmp_path) -> None:
        from wiki_memory.content_store import ContentStore

        vault = tmp_path / "wiki"
        vault.mkdir()
        store = ContentStore(tmp_path / "store.db")
        await store.init(vault_root=vault)
        try:
            stored = await store._meta_get("vault_root")
            assert stored == str(vault.resolve())
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_vault_root_mismatch_warns(self, tmp_path, caplog) -> None:
        import logging

        from wiki_memory.content_store import ContentStore

        vault_a = tmp_path / "vault_a"
        vault_b = tmp_path / "vault_b"
        vault_a.mkdir()
        vault_b.mkdir()

        store = ContentStore(tmp_path / "store.db")
        await store.init(vault_root=vault_a)
        await store.close()

        store2 = ContentStore(tmp_path / "store.db")
        with caplog.at_level(logging.WARNING, logger="wiki_memory.content_store"):
            await store2.init(vault_root=vault_b)
        try:
            assert any("vault_root changed" in r.message for r in caplog.records)
            # And the stored value has been updated so the warning is
            # one-shot per change.
            assert await store2._meta_get("vault_root") == str(vault_b.resolve())
        finally:
            await store2.close()

    @pytest.mark.asyncio
    async def test_no_vault_root_arg_is_a_noop(self, tmp_path) -> None:
        from wiki_memory.content_store import ContentStore

        store = ContentStore(tmp_path / "store.db")
        # Pre-#127 callers don't pass vault_root — that path must stay
        # working.
        await store.init()
        try:
            assert await store._meta_get("vault_root") is None
        finally:
            await store.close()


class TestSearchFts:
    """FTS5 ranker (issue #118). Replaces substring as the default in
    memory_tree_recall — substring stays as a fallback. Tests pin the
    BM25 ordering and the porter+unicode61 stemming gains."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, content_store) -> None:
        chunk = _mk_chunks("body content")[0]
        await content_store.insert(source_id="src-1", chunk=chunk)
        assert await content_store.search_fts("") == []
        assert await content_store.search_fts("   ") == []

    @pytest.mark.asyncio
    async def test_sanitised_query_dangerous_chars_dropped(self, content_store) -> None:
        # User accidentally passes FTS5 operators — must not crash the parser.
        chunk = _mk_chunks("body content here")[0]
        await content_store.insert(source_id="src-1", chunk=chunk)
        hits = await content_store.search_fts('body "content" *')
        assert len(hits) == 1
        assert hits[0].sha256 == chunk.sha256

    @pytest.mark.asyncio
    async def test_porter_stemming_matches_plural(self, content_store) -> None:
        # Porter stemmer collapses "agents" ↔ "agent".
        chunk = _mk_chunks("autonomous agents are level three")[0]
        await content_store.insert(source_id="src-1", chunk=chunk)
        hits_singular = await content_store.search_fts("agent")
        assert len(hits_singular) == 1
        assert hits_singular[0].sha256 == chunk.sha256

    @pytest.mark.asyncio
    async def test_unicode61_normalises_diacritics(self, content_store) -> None:
        # unicode61 strips accents — Spanish queries hit Spanish bodies.
        chunk = _mk_chunks("La observación es necesaria")[0]
        await content_store.insert(source_id="src-1", chunk=chunk)
        hits = await content_store.search_fts("observacion")
        assert len(hits) == 1
        assert hits[0].sha256 == chunk.sha256

    @pytest.mark.asyncio
    async def test_bm25_orders_by_relevance(self, content_store) -> None:
        # The chunk with the query repeated should rank higher.
        rare = _mk_chunks("alpha appears once.")[0]
        frequent = _mk_chunks("alpha alpha alpha alpha alpha.")[0]
        await content_store.insert(source_id="src-rare", chunk=rare)
        await content_store.insert(source_id="src-frequent", chunk=frequent)
        hits = await content_store.search_fts("alpha")
        assert len(hits) == 2
        # BM25 ranks the body with higher term frequency first.
        assert hits[0].source_id == "src-frequent"
        assert hits[1].source_id == "src-rare"

    @pytest.mark.asyncio
    async def test_delete_propagates_to_fts(self, content_store) -> None:
        chunk = _mk_chunks("to be deleted")[0]
        await content_store.insert(source_id="src-del", chunk=chunk)
        assert len(await content_store.search_fts("deleted")) == 1
        await content_store.delete_by_source("src-del")
        assert await content_store.search_fts("deleted") == []

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, content_store) -> None:
        chunks = _mk_chunks("alpha.\n\nalpha two.\n\nalpha three.", target=5, cap=15)
        await content_store.insert_many(source_id="src-many", chunks=chunks)
        hits = await content_store.search_fts("alpha", limit=2)
        assert len(hits) <= 2


class TestFtsRebuildOnExistingDb:
    """Migration path: a content_store.db that pre-dates FTS5 (created
    before issue #118) has chunks but no fts rows. ``init()`` runs the
    one-shot rebuild so the existing 477-chunk DB in production picks
    up FTS5 automatically when the new code lands."""

    @pytest.mark.asyncio
    async def test_rebuild_populates_fts_from_existing_chunks(self, tmp_path) -> None:
        import aiosqlite

        from wiki_memory.content_store import ContentStore

        db_path = tmp_path / "legacy.db"
        # Simulate a pre-#118 DB: chunks table populated WITHOUT the
        # FTS5 virtual table being created (the old schema).
        async with aiosqlite.connect(db_path) as legacy:
            await legacy.executescript(
                """
                CREATE TABLE chunks (
                    sha256 TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    body TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            await legacy.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                ("sha1", "src", 0, "legacy content searchable", 5, "2026-05-29T00:00:00Z"),
            )
            await legacy.commit()

        # New ContentStore opens it — should auto-rebuild FTS.
        store = ContentStore(db_path)
        await store.init()
        try:
            hits = await store.search_fts("legacy")
            assert len(hits) == 1
            assert hits[0].sha256 == "sha1"
        finally:
            await store.close()


class TestSearchSubstring:
    """Tests for the v1 substring scan that backs the MCP recall surface
    (issue #78). Replaced by a vector index once PRD-005 lands; until
    then this is the deterministic fallback the eval suite gates on."""

    @pytest.mark.asyncio
    async def test_empty_needle_returns_empty(self, content_store) -> None:
        chunk = _mk_chunks("body content")[0]
        await content_store.insert(source_id="src-1", chunk=chunk)
        assert await content_store.search_substring("") == []
        assert await content_store.search_substring("   ") == []

    @pytest.mark.asyncio
    async def test_match_is_case_insensitive(self, content_store) -> None:
        chunk = _mk_chunks("Hello World")[0]
        await content_store.insert(source_id="src-1", chunk=chunk)
        hits = await content_store.search_substring("HELLO")
        assert len(hits) == 1
        assert hits[0].sha256 == chunk.sha256

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, content_store) -> None:
        chunks = _mk_chunks("alpha.\n\nalpha two.\n\nalpha three.", target=5, cap=15)
        await content_store.insert_many(source_id="src-many", chunks=chunks)
        hits = await content_store.search_substring("alpha", limit=2)
        assert len(hits) == 2

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, content_store) -> None:
        chunk = _mk_chunks("present")[0]
        await content_store.insert(source_id="src-1", chunk=chunk)
        assert await content_store.search_substring("absent") == []
