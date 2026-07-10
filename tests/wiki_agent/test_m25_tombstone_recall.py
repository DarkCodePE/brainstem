"""
M2.5 integration tests at the MCP handler level:

- ADR-029 #159: memory_tree_tombstone deletes chunks so forgotten content
  leaves the recall surface (the bug: it used to only flag the node).
- ADR-027 #155: a recall increments reuse_count on the chunks it returns.
- ADR-028 #158: recall excludes chunks from superseded source versions.

Mirrors the fixture/unwrap strategy of test_mcp_memory_tree.py.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from wiki_agent import mcp_server
from wiki_memory.chunker import chunk_text

recall_tool = mcp_server.memory_tree_recall
tombstone_tool = mcp_server.memory_tree_tombstone


async def _unwrap(tool, **kwargs):
    if hasattr(tool, "fn"):
        return await tool.fn(**kwargs)
    return await tool(**kwargs)


@pytest_asyncio.fixture
async def memory_env(tmp_path: Path, monkeypatch) -> AsyncIterator[Path]:
    monkeypatch.setenv("WIKI_MEMORY_CONTENT_DB", str(tmp_path / "content_store.db"))
    monkeypatch.setenv("WIKI_MEMORY_TREE_DB", str(tmp_path / "tree_nodes.db"))
    monkeypatch.setenv("WIKI_ROOT", str(tmp_path / "vault"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    mcp_server._memory_stack = None
    mcp_server._memory_init_lock = None
    yield tmp_path
    if mcp_server._memory_stack is not None:
        await mcp_server._memory_stack["content_store"].close()
        await mcp_server._memory_stack["tree_store"].close()
    mcp_server._memory_stack = None
    mcp_server._memory_init_lock = None


async def _seed(stack, source_id: str, text: str, *, source_key: str | None = None) -> list[str]:
    chunks = chunk_text(text, target_tokens=5, hard_cap_tokens=20)
    await stack["content_store"].insert_many(source_id=source_id, chunks=chunks)
    await stack["tree_store"].create_source_node(node_id=source_id, source_key=source_key)
    return [c.sha256 for c in chunks]


class TestTombstoneDeletesFromRecall:
    @pytest.mark.asyncio
    async def test_tombstone_removes_chunks_from_recall(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        await _seed(stack, "src-forget", "unique_token_xyz appears here.\n\nmore xyz body.")

        # Before: recall finds the chunks.
        before = json.loads(await _unwrap(recall_tool, query="xyz", mode="fts", token_budget=4000))
        assert len(before["chunks"]) >= 1

        # Tombstone reports how many chunks it deleted.
        result = json.loads(await _unwrap(tombstone_tool, node_id="src-forget"))
        assert result["changed"] is True
        assert result["chunks_deleted"] >= 1

        # After: recall finds nothing — the forget actually forgot.
        after = json.loads(await _unwrap(recall_tool, query="xyz", mode="fts", token_budget=4000))
        assert after["chunks"] == []

    @pytest.mark.asyncio
    async def test_second_tombstone_deletes_zero(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        await _seed(stack, "src-x", "alpha body here.")
        first = json.loads(await _unwrap(tombstone_tool, node_id="src-x"))
        assert first["chunks_deleted"] >= 1
        second = json.loads(await _unwrap(tombstone_tool, node_id="src-x"))
        assert second["chunks_deleted"] == 0
        assert second["changed"] is False


class _StubContentStore:
    """London-school stub — records the orchestration order."""

    def __init__(self, calls: list[tuple[str, str]], *, chunks_deleted: int = 3) -> None:
        self._calls = calls
        self._chunks_deleted = chunks_deleted

    async def delete_by_source(self, source_id: str) -> int:
        self._calls.append(("delete_by_source", source_id))
        return self._chunks_deleted


class _StubTreeStore:
    def __init__(self, calls: list[tuple[str, str]], *, fail: bool = False) -> None:
        self._calls = calls
        self._fail = fail

    async def tombstone(self, node_id: str) -> bool:
        self._calls.append(("tombstone", node_id))
        if self._fail:
            raise RuntimeError("tree db unavailable")
        return True


@pytest.fixture
def stub_stack(monkeypatch):
    """Install stub stores as the handler's memory stack; monkeypatch
    restores the real (None) stack afterwards."""

    def _install(content_store, tree_store) -> None:
        monkeypatch.setattr(
            mcp_server,
            "_memory_stack",
            {"content_store": content_store, "tree_store": tree_store, "seal_worker": None},
        )

    return _install


class TestTombstoneOrchestration:
    """ADR-029: the MCP handler (the one layer holding BOTH store handles)
    orchestrates chunk deletion + node flag, in that order."""

    @pytest.mark.asyncio
    async def test_deletes_chunks_before_flagging_node(self, stub_stack) -> None:
        calls: list[tuple[str, str]] = []
        stub_stack(_StubContentStore(calls), _StubTreeStore(calls))

        result = json.loads(await _unwrap(tombstone_tool, node_id="n-1"))

        assert calls == [("delete_by_source", "n-1"), ("tombstone", "n-1")]
        assert result["node_tombstoned"] is True
        assert result["changed"] is True  # backwards-compatible mirror
        assert result["chunks_deleted"] == 3

    @pytest.mark.asyncio
    async def test_flag_failure_reports_partial_state_not_silent(self, stub_stack) -> None:
        """Chunk delete succeeds, node flag fails → the partial state is
        REPORTED (chunks_deleted surfaced alongside the error), never a
        silent half-completion or an unhandled exception."""
        calls: list[tuple[str, str]] = []
        stub_stack(_StubContentStore(calls, chunks_deleted=5), _StubTreeStore(calls, fail=True))

        result = json.loads(await _unwrap(tombstone_tool, node_id="n-broken"))

        assert "error" in result
        assert result["chunks_deleted"] == 5  # the delete that DID happen
        assert result["node_tombstoned"] is False
        assert result["error_class"] == "RuntimeError"


class TestRecallIncrementsReuse:
    @pytest.mark.asyncio
    async def test_returned_chunks_get_reuse_incremented(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        shas = await _seed(stack, "src-r", "reuseword present in body text.")

        # Reuse starts at 0.
        assert (await stack["content_store"].get(shas[0])).reuse_count == 0

        out = json.loads(
            await _unwrap(recall_tool, query="reuseword", mode="fts", token_budget=4000)
        )
        returned = {c["sha256"] for c in out["chunks"]}
        assert returned  # something came back

        for sha in returned:
            assert (await stack["content_store"].get(sha)).reuse_count == 1


class TestRecallExcludesSuperseded:
    @pytest.mark.asyncio
    async def test_superseded_source_dropped_by_default(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        tree = stack["tree_store"]
        # Two versions of the same logical source; v2 supersedes v1.
        await _seed(stack, "v1", "sharedkw old stale wording.", source_key="key-1")
        await _seed(stack, "v2", "sharedkw new corrected wording.", source_key="key-1")
        await tree.supersede(source_key="key-1", new_node_id="v2")

        # Default recall: only the latest version's chunks.
        default = json.loads(
            await _unwrap(recall_tool, query="sharedkw", mode="fts", token_budget=4000)
        )
        sources = {c["source_id"] for c in default["chunks"]}
        assert "v1" not in sources
        assert "v2" in sources

        # Opt-in: superseded chunks come back too.
        allv = json.loads(
            await _unwrap(
                recall_tool,
                query="sharedkw",
                mode="fts",
                token_budget=4000,
                include_superseded=True,
            )
        )
        sources_all = {c["source_id"] for c in allv["chunks"]}
        assert {"v1", "v2"} <= sources_all
