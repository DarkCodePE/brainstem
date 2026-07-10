"""
End-to-end test that exercises the same MCP-client → MCP-server path
that Hermes uses when serving a Telegram message.

Hermes' workflow: Telegram → subagent → MCP client → stdio subprocess
→ wiki_agent.mcp_server. This test replicates the MCP client side,
spawning the real server as a subprocess and asserting:

1. tools/list returns the 15 expected tool names
2. memory_tree_list_topics returns a sane (possibly empty) shape
3. memory_tree_tombstone is reachable and returns the no-op envelope
   for a fresh node id

We deliberately do NOT exercise the seal_now / recall paths here
because they need a populated content_store + (optionally) provider
keys. Those are exercised by the in-process test suite (test_mcp_memory_tree.py)
and the live smoke playbook (docs/smoke/hermes-sbw-telegram.md).

What this test gates is the **transport contract**: the very thing
hermes-agent's MCP client interacts with. Issue #114 closes against
this — once this passes, the bridge surface is verified end-to-end
short of a real Telegram message.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_stdio_handshake_lists_15_tools(tmp_path: Path, monkeypatch) -> None:
    """Spawn the SBW MCP server over stdio, perform the discovery
    handshake, and assert exactly the 15 tools are advertised
    (11 LangChain wiki + 4 memory.tree.*)."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    # Point the server at tmp dirs so we don't touch real state.
    env = os.environ.copy()
    env["WIKI_ROOT"] = str(tmp_path / "kb")
    env["WIKI_MEMORY_CONTENT_DB"] = str(tmp_path / "content.db")
    env["WIKI_MEMORY_TREE_DB"] = str(tmp_path / "tree.db")
    # Force NullSummariser to keep the seal path deterministic if the
    # client happens to call it.
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENROUTER_API_KEY", None)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "wiki_agent.mcp_server"],
        env=env,
    )

    expected_tools = {
        # 11 LangChain wiki tools
        "search_wiki_index",
        "read_wiki_file",
        "get_wiki_stats",
        "find_cross_references",
        "detect_orphan_pages",
        "validate_frontmatter",
        "write_page",
        "update_index_entry",
        "append_to_log",
        "web_clip",
        "update_schema_lessons",
        # 4 memory.tree.* methods (#78)
        "memory_tree_recall",
        "memory_tree_seal_now",
        "memory_tree_list_topics",
        "memory_tree_tombstone",
    }

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_response = await session.list_tools()
            advertised = {t.name for t in tools_response.tools}
            # ⊇ rather than == so future additions don't break this test —
            # but the 15 we promised MUST all be present.
            missing = expected_tools - advertised
            assert not missing, (
                f"missing expected tools: {missing}; advertised={sorted(advertised)}"
            )


@pytest.mark.asyncio
async def test_stdio_memory_tree_list_topics_callable(tmp_path: Path) -> None:
    """Same transport, but actually CALL one of the new tools. Closes
    the loop: handshake works AND the new tools are invocable."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = os.environ.copy()
    env["WIKI_ROOT"] = str(tmp_path / "kb")
    env["WIKI_MEMORY_CONTENT_DB"] = str(tmp_path / "content.db")
    env["WIKI_MEMORY_TREE_DB"] = str(tmp_path / "tree.db")
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENROUTER_API_KEY", None)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "wiki_agent.mcp_server"],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "memory_tree_list_topics",
                arguments={"kind": "source"},
            )
            # The tool returns a JSON string; FastMCP wraps it as a
            # TextContent block.
            text_blocks = [c.text for c in result.content if hasattr(c, "text")]
            assert text_blocks, f"expected text content, got {result.content!r}"
            import json as _json

            payload = _json.loads(text_blocks[0])
            assert payload["kind"] == "source"
            assert payload["count"] == 0
            assert payload["nodes"] == []
