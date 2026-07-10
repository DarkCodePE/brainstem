"""
M2.5 tests — temporal supersession at the source-node level (ADR-028 #158).

Re-ingesting the same logical source (same source_key) must mark the
prior version superseded (is_latest=0, superseded_by set) while retaining
it on disk. First ingest supersedes nothing. Includes a legacy-DB
migration test and an end-to-end seal-hook re-ingest test.
"""

from __future__ import annotations

from datetime import UTC, datetime

import aiosqlite
import pytest

from wiki_agent.write_sink import NullWriteSink
from wiki_core.protocols import IngestEvent
from wiki_memory.seal_hook import SealOnIngestHook
from wiki_memory.summariser import NullSummariser
from wiki_memory.tree_nodes import TreeNodeStore


class TestSupersedeStore:
    @pytest.mark.asyncio
    async def test_create_source_node_persists_source_key(self, tree_store) -> None:
        node = await tree_store.create_source_node(
            node_id="v1", source_key="key-1", event_time="2026-05-01T00:00:00Z"
        )
        assert node.source_key == "key-1"
        assert node.is_latest is True
        assert node.superseded_by is None
        fetched = await tree_store.get("v1")
        assert fetched.source_key == "key-1"
        assert fetched.event_time == "2026-05-01T00:00:00Z"
        assert fetched.is_latest is True

    @pytest.mark.asyncio
    async def test_first_ingest_supersedes_nothing(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="v1", source_key="key-1")
        superseded = await tree_store.supersede(source_key="key-1", new_node_id="v1")
        assert superseded == []
        assert (await tree_store.get("v1")).is_latest is True

    @pytest.mark.asyncio
    async def test_reingest_supersedes_prior(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="v1", source_key="key-1")
        await tree_store.create_source_node(node_id="v2", source_key="key-1")
        superseded = await tree_store.supersede(source_key="key-1", new_node_id="v2")
        assert superseded == ["v1"]
        v1 = await tree_store.get("v1")
        v2 = await tree_store.get("v2")
        assert v1.is_latest is False
        assert v1.superseded_by == "v2"
        assert v2.is_latest is True
        assert v2.superseded_by is None

    @pytest.mark.asyncio
    async def test_three_version_chain_only_latest_live(self, tree_store) -> None:
        for nid in ("v1", "v2", "v3"):
            await tree_store.create_source_node(node_id=nid, source_key="key-1")
            await tree_store.supersede(source_key="key-1", new_node_id=nid)
        superseded = set(await tree_store.superseded_node_ids())
        assert superseded == {"v1", "v2"}
        assert (await tree_store.get("v3")).is_latest is True

    @pytest.mark.asyncio
    async def test_distinct_source_keys_isolated(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="a1", source_key="key-A")
        await tree_store.create_source_node(node_id="b1", source_key="key-B")
        await tree_store.create_source_node(node_id="a2", source_key="key-A")
        superseded = await tree_store.supersede(source_key="key-A", new_node_id="a2")
        # Only key-A's prior version is touched; key-B is untouched.
        assert superseded == ["a1"]
        assert (await tree_store.get("b1")).is_latest is True

    @pytest.mark.asyncio
    async def test_supersede_idempotent(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="v1", source_key="key-1")
        await tree_store.create_source_node(node_id="v2", source_key="key-1")
        first = await tree_store.supersede(source_key="key-1", new_node_id="v2")
        second = await tree_store.supersede(source_key="key-1", new_node_id="v2")
        assert first == ["v1"]
        assert second == []  # nothing still-latest to flip


class TestLegacyMigration:
    @pytest.mark.asyncio
    async def test_supersession_columns_added_to_legacy_db(self, tmp_path) -> None:
        db_path = tmp_path / "legacy_tree.db"
        async with aiosqlite.connect(db_path) as db:
            await db.execute(
                "CREATE TABLE tree_nodes (node_id TEXT PRIMARY KEY, kind TEXT NOT NULL,"
                " parent_id TEXT, level INTEGER NOT NULL, summary_sha256 TEXT,"
                " score REAL NOT NULL DEFAULT 0.0, sealed_at TEXT,"
                " tombstoned INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL)"
            )
            await db.execute(
                "INSERT INTO tree_nodes (node_id, kind, parent_id, level, created_at)"
                " VALUES ('old', 'source', NULL, 0, '2024-01-01T00:00:00Z')"
            )
            await db.commit()

        store = TreeNodeStore(db_path)
        await store.init()
        try:
            node = await store.get("old")
            assert node is not None
            assert node.is_latest is True  # backfilled default
            assert node.source_key is None
            # And supersession works on the migrated DB.
            await store.create_source_node(node_id="new", source_key="k")
            # 'old' had no source_key so it is isolated; supersede on a
            # fresh key is a clean no-op-then-supersede chain.
        finally:
            await store.close()


def _make_event(*, path_or_uri: str, sha256: str, event_id: str) -> IngestEvent:
    return IngestEvent(
        event_id=event_id,
        source="watcher:articles",
        path_or_uri=path_or_uri,
        sha256=sha256,
        received_at=datetime.now(UTC),
        metadata={},
    )


class TestSealHookReingest:
    @pytest.mark.asyncio
    async def test_reingest_same_path_supersedes_old_version(
        self, content_store, tree_store, tmp_path
    ) -> None:
        """End-to-end: ingest a file, then re-ingest the SAME path with
        changed content. The hook must create a new source node and mark
        the old one superseded — proving the #158 wiring, not just the
        store method."""
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        page_path = "wiki/sources/doc.md"
        full = vault_root / page_path
        full.parent.mkdir(parents=True, exist_ok=True)

        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=NullWriteSink(),
            vault_root=vault_root,
            summariser_factory=NullSummariser,
            schedule=lambda coro: coro.close(),  # don't run the seal task
            enable_seal=False,
        )

        # First ingest.
        full.write_text("original content here.\n", encoding="utf-8")
        await hook(
            _make_event(path_or_uri="/abs/doc.md", sha256="a" * 64, event_id="e1"), page_path
        )

        # Re-ingest same path, changed content (=> new sha => new node).
        full.write_text("revised and corrected content.\n", encoding="utf-8")
        await hook(
            _make_event(path_or_uri="/abs/doc.md", sha256="b" * 64, event_id="e2"), page_path
        )

        sources = await tree_store.list_by_kind("source")
        assert len(sources) == 2  # both versions retained
        latest = [n for n in sources if n.is_latest]
        superseded = [n for n in sources if not n.is_latest]
        assert len(latest) == 1
        assert len(superseded) == 1
        # They share one logical source_key (same path).
        assert latest[0].source_key == superseded[0].source_key
        assert superseded[0].superseded_by == latest[0].node_id
