"""
Tests for `wiki_memory.tree_nodes` — Memory Tree v1 SQLite-backed node CRUD.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wiki_memory.tree_nodes import TreeNode


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mk_node(
    node_id: str = "node-1",
    kind: str = "source",
    parent_id: str | None = None,
    level: int = 0,
    summary_sha256: str | None = None,
    score: float = 0.0,
    sealed_at: str | None = None,
    tombstoned: bool = False,
) -> TreeNode:
    return TreeNode(
        node_id=node_id,
        kind=kind,  # type: ignore[arg-type]
        parent_id=parent_id,
        level=level,
        summary_sha256=summary_sha256,
        score=score,
        sealed_at=sealed_at,
        tombstoned=tombstoned,
        created_at=_utcnow_iso(),
    )


class TestUpsert:
    @pytest.mark.asyncio
    async def test_create_source_node(self, tree_store) -> None:
        n = await tree_store.create_source_node(node_id="src-a")
        assert n.kind == "source"
        assert n.level == 0
        assert n.tombstoned is False
        assert n.sealed_at is None
        # Re-fetch to verify persistence
        fetched = await tree_store.get("src-a")
        assert fetched is not None
        assert fetched.node_id == "src-a"

    @pytest.mark.asyncio
    async def test_upsert_overwrites_mutable_fields(self, tree_store) -> None:
        first = _mk_node(node_id="x", score=0.0)
        await tree_store.upsert(first)
        second = _mk_node(node_id="x", score=0.99)
        await tree_store.upsert(second)
        fetched = await tree_store.get("x")
        assert fetched is not None
        assert fetched.score == pytest.approx(0.99)

    @pytest.mark.asyncio
    async def test_count_initial_zero(self, tree_store) -> None:
        assert await tree_store.count() == 0


class TestListAndChildren:
    @pytest.mark.asyncio
    async def test_list_by_kind_segregates(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="s-1")
        await tree_store.upsert(_mk_node(node_id="t-1", kind="topic", level=1))
        await tree_store.upsert(_mk_node(node_id="g-1", kind="global", level=2))
        sources = await tree_store.list_by_kind("source")
        topics = await tree_store.list_by_kind("topic")
        globals_ = await tree_store.list_by_kind("global")
        assert len(sources) == 1 and sources[0].node_id == "s-1"
        assert len(topics) == 1 and topics[0].node_id == "t-1"
        assert len(globals_) == 1 and globals_[0].node_id == "g-1"

    @pytest.mark.asyncio
    async def test_children_of_returns_parents_subtree(self, tree_store) -> None:
        await tree_store.upsert(_mk_node(node_id="t-root", kind="topic", level=1))
        await tree_store.create_source_node(node_id="s-1", parent_id="t-root")
        await tree_store.create_source_node(node_id="s-2", parent_id="t-root")
        await tree_store.create_source_node(node_id="s-3", parent_id=None)  # orphan
        kids = await tree_store.children_of("t-root")
        assert {k.node_id for k in kids} == {"s-1", "s-2"}


class TestTombstone:
    @pytest.mark.asyncio
    async def test_tombstone_marks_node(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="dead")
        changed = await tree_store.tombstone("dead")
        assert changed is True
        n = await tree_store.get("dead")
        assert n is not None
        assert n.tombstoned is True

    @pytest.mark.asyncio
    async def test_tombstone_unknown_returns_false(self, tree_store) -> None:
        assert await tree_store.tombstone("ghost") is False

    @pytest.mark.asyncio
    async def test_tombstone_excluded_from_default_list(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="live")
        await tree_store.create_source_node(node_id="dead")
        await tree_store.tombstone("dead")
        nodes = await tree_store.list_by_kind("source")
        assert {n.node_id for n in nodes} == {"live"}
        # include_tombstoned=True surfaces both
        all_nodes = await tree_store.list_by_kind("source", include_tombstoned=True)
        assert {n.node_id for n in all_nodes} == {"live", "dead"}


class TestMarkSealed:
    @pytest.mark.asyncio
    async def test_mark_sealed_sets_summary_and_timestamp(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="to-seal")
        await tree_store.mark_sealed("to-seal", summary_sha256="s" * 64)
        n = await tree_store.get("to-seal")
        assert n is not None
        assert n.summary_sha256 == "s" * 64
        assert n.sealed_at is not None

    @pytest.mark.asyncio
    async def test_mark_sealed_with_score_updates_both(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="to-score")
        await tree_store.mark_sealed("to-score", summary_sha256="x" * 64, score=0.42)
        n = await tree_store.get("to-score")
        assert n is not None
        assert n.score == pytest.approx(0.42)


class TestCount:
    @pytest.mark.asyncio
    async def test_count_excludes_tombstoned_by_default(self, tree_store) -> None:
        await tree_store.create_source_node(node_id="a")
        await tree_store.create_source_node(node_id="b")
        await tree_store.tombstone("b")
        assert await tree_store.count() == 1
        assert await tree_store.count(include_tombstoned=True) == 2
