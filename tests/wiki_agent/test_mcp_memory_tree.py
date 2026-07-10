"""
Tests for the ``memory.tree.*`` MCP surface added by issue #78.

Hermes, Tauri (M4), and any future MCP client call these four methods to
recall chunks, force a seal, list nodes, and tombstone. They are the
chat-shell-facing surface of the Memory Tree substrate.

Strategy: import the underlying async tool callables (FastMCP exposes
them on the module), point the env vars at tmp paths, reset the
module-level singleton between tests, and exercise the JSON envelope
contract. We deliberately do NOT spin up an MCP server here — the
FastMCP ``tools/call`` plumbing is tested by the upstream mcp package.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from wiki_agent import mcp_server
from wiki_memory.chunker import chunk_text

# Resolve the underlying async callables. FastMCP wraps decorated tools
# but keeps the original function reachable via the module namespace
# because we declared them as plain ``async def`` and decorated in-place.
recall_tool = mcp_server.memory_tree_recall
seal_tool = mcp_server.memory_tree_seal_now
list_tool = mcp_server.memory_tree_list_topics
tombstone_tool = mcp_server.memory_tree_tombstone


async def _unwrap(tool, **kwargs):
    """FastMCP @mcp.tool() returns a FunctionTool wrapper. Call the
    original via .fn when present, else call the tool directly."""
    if hasattr(tool, "fn"):
        return await tool.fn(**kwargs)
    return await tool(**kwargs)


@pytest_asyncio.fixture
async def memory_env(tmp_path: Path, monkeypatch) -> AsyncIterator[Path]:
    """Point the MCP server's env vars at tmp paths and reset the
    lazy-singleton so each test gets a clean stack. Provider keys are
    explicitly unset to force the NullSummariser path; seal flow stays
    deterministic for assertions."""
    monkeypatch.setenv("WIKI_MEMORY_CONTENT_DB", str(tmp_path / "content_store.db"))
    monkeypatch.setenv("WIKI_MEMORY_TREE_DB", str(tmp_path / "tree_nodes.db"))
    monkeypatch.setenv("WIKI_ROOT", str(tmp_path / "vault"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    mcp_server._memory_stack = None
    mcp_server._memory_init_lock = None
    yield tmp_path
    # Best-effort close so the SQLite WAL files don't linger.
    if mcp_server._memory_stack is not None:
        await mcp_server._memory_stack["content_store"].close()
        await mcp_server._memory_stack["tree_store"].close()
    mcp_server._memory_stack = None
    mcp_server._memory_init_lock = None


async def _seed(stack, source_id: str, text: str) -> list[str]:
    """Insert chunks for a source and return their shas. The tree_node
    row is created with the same id as the source so seal_now's
    auto-create path doesn't need to fire."""
    chunks = chunk_text(text, target_tokens=5, hard_cap_tokens=20)
    await stack["content_store"].insert_many(source_id=source_id, chunks=chunks)
    await stack["tree_store"].create_source_node(node_id=source_id)
    return [c.sha256 for c in chunks]


class TestRecall:
    @pytest.mark.asyncio
    async def test_no_scope_returns_error(self, memory_env) -> None:
        result = json.loads(await _unwrap(recall_tool))
        assert "error" in result
        assert "query" in result["error"] or "source_id" in result["error"]

    @pytest.mark.asyncio
    async def test_zero_budget_returns_error(self, memory_env) -> None:
        result = json.loads(await _unwrap(recall_tool, source_id="x", token_budget=0))
        assert "error" in result
        assert "token_budget" in result["error"]

    @pytest.mark.asyncio
    async def test_recall_by_source_id(self, memory_env) -> None:
        # First call lazy-opens the stack; reach in via the module to seed.
        stack = await mcp_server._get_memory_stack()
        await _seed(stack, "src-A", "alpha one.\n\nalpha two.\n\nalpha three.")

        result = json.loads(await _unwrap(recall_tool, source_id="src-A", token_budget=4000))
        assert "chunks" in result
        assert len(result["chunks"]) >= 1
        # Returned in chunk_index order
        indices = [c["chunk_index"] for c in result["chunks"]]
        assert indices == sorted(indices)
        assert result["total_tokens"] <= 4000
        # All chunks belong to the scoped source.
        assert all(c["source_id"] == "src-A" for c in result["chunks"])

    @pytest.mark.asyncio
    async def test_recall_by_query_substring(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        await _seed(stack, "src-A", "alpha pattern here.")
        await _seed(stack, "src-B", "beta unrelated.")

        result = json.loads(await _unwrap(recall_tool, query="alpha", token_budget=4000))
        assert all("alpha" in c["body"].casefold() for c in result["chunks"])
        sources = {c["source_id"] for c in result["chunks"]}
        assert sources == {"src-A"}

    @pytest.mark.asyncio
    async def test_recall_query_and_source_intersect(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        await _seed(stack, "src-A", "alpha one.\n\nbeta two.")
        await _seed(stack, "src-B", "alpha three.")

        result = json.loads(
            await _unwrap(recall_tool, query="alpha", source_id="src-A", token_budget=4000)
        )
        # src-B's "alpha three" is excluded by the source scope; src-A's
        # "beta two" is excluded by the query filter.
        sources = {c["source_id"] for c in result["chunks"]}
        assert sources == {"src-A"}
        bodies = " ".join(c["body"].casefold() for c in result["chunks"])
        assert "alpha" in bodies


class TestSealNow:
    @pytest.mark.asyncio
    async def test_seal_unknown_source_errors(self, memory_env) -> None:
        result = json.loads(await _unwrap(seal_tool, source_id="never-seeded"))
        assert "error" in result
        # SealWorker.seal_source raises SealError on no-chunks; the tool
        # wraps that as a generic seal-failed envelope.
        assert "seal" in result["error"]

    @pytest.mark.asyncio
    async def test_seal_seeded_source_returns_result(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        await _seed(stack, "src-seal", "para one.\n\npara two.")

        result = json.loads(await _unwrap(seal_tool, source_id="src-seal"))
        assert "summary_sha256" in result
        assert result["node_id"] == "src-seal"
        assert result["children_count"] >= 1
        assert result["page_path"].startswith("wiki/trees/")

    @pytest.mark.asyncio
    async def test_seal_explicit_node_id(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        await _seed(stack, "src-other", "fresh chunk.")

        # Pass a different node_id; the tool should create the row and seal.
        result = json.loads(await _unwrap(seal_tool, source_id="src-other", node_id="custom-node"))
        assert result["node_id"] == "custom-node"


class TestListTopics:
    @pytest.mark.asyncio
    async def test_list_empty(self, memory_env) -> None:
        result = json.loads(await _unwrap(list_tool))
        assert result["kind"] == "topic"
        assert result["count"] == 0
        assert result["nodes"] == []

    @pytest.mark.asyncio
    async def test_list_sources(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        await stack["tree_store"].create_source_node(node_id="src-1")
        await stack["tree_store"].create_source_node(node_id="src-2")

        result = json.loads(await _unwrap(list_tool, kind="source"))
        ids = {n["node_id"] for n in result["nodes"]}
        assert ids == {"src-1", "src-2"}
        assert all(n["kind"] == "source" for n in result["nodes"])

    @pytest.mark.asyncio
    async def test_list_invalid_kind_errors(self, memory_env) -> None:
        result = json.loads(await _unwrap(list_tool, kind="bogus"))
        assert "error" in result
        assert result["received"] == "bogus"


class TestTombstone:
    @pytest.mark.asyncio
    async def test_tombstone_existing_node(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        await stack["tree_store"].create_source_node(node_id="to-rm")

        result = json.loads(await _unwrap(tombstone_tool, node_id="to-rm"))
        # ADR-029 #159: envelope carries chunks_deleted + node_tombstoned
        # alongside changed (its backwards-compatible mirror).
        assert result == {
            "node_id": "to-rm",
            "changed": True,
            "node_tombstoned": True,
            "chunks_deleted": 0,
        }

        # Second tombstone is a no-op (idempotent).
        result = json.loads(await _unwrap(tombstone_tool, node_id="to-rm"))
        assert result == {
            "node_id": "to-rm",
            "changed": False,
            "node_tombstoned": False,
            "chunks_deleted": 0,
        }

    @pytest.mark.asyncio
    async def test_tombstone_missing_node_no_change(self, memory_env) -> None:
        result = json.loads(await _unwrap(tombstone_tool, node_id="never-was"))
        assert result == {
            "node_id": "never-was",
            "changed": False,
            "node_tombstoned": False,
            "chunks_deleted": 0,
        }

    @pytest.mark.asyncio
    async def test_tombstoned_hidden_from_list(self, memory_env) -> None:
        stack = await mcp_server._get_memory_stack()
        await stack["tree_store"].create_source_node(node_id="visible")
        await stack["tree_store"].create_source_node(node_id="hidden")
        await _unwrap(tombstone_tool, node_id="hidden")

        default = json.loads(await _unwrap(list_tool, kind="source"))
        ids_default = {n["node_id"] for n in default["nodes"]}
        assert ids_default == {"visible"}

        with_tomb = json.loads(await _unwrap(list_tool, kind="source", include_tombstoned=True))
        ids_all = {n["node_id"] for n in with_tomb["nodes"]}
        assert ids_all == {"visible", "hidden"}
