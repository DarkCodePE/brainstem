"""Tests for WIKI_MCP_READONLY (ADR-034 D5, wiki_agent/mcp_server.py).

The flag must strip every write/publish surface and keep every read surface.
The removal test re-adds nothing: it runs against the live module registry,
so it executes in a subprocess to avoid mutating the shared `mcp` instance
for other tests.
"""

from __future__ import annotations

import json
import subprocess
import sys

from wiki_agent.mcp_server import READONLY_TOOLS, apply_readonly_mode

_LIST_TOOLS_SNIPPET = """
import asyncio, json, os
os.environ["WIKI_MCP_READONLY"] = "1"
from wiki_agent import mcp_server
tools = asyncio.run(mcp_server.mcp.list_tools())
print(json.dumps(sorted(t.name for t in tools)))
"""


def test_flag_unset_removes_nothing() -> None:
    assert apply_readonly_mode(env={}) == []
    assert apply_readonly_mode(env={"WIKI_MCP_READONLY": "0"}) == []
    assert apply_readonly_mode(env={"WIKI_MCP_READONLY": ""}) == []


def test_readonly_registry_is_exactly_the_allowlist() -> None:
    out = subprocess.run(
        [sys.executable, "-c", _LIST_TOOLS_SNIPPET],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    names = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert names == READONLY_TOOLS
    # The surfaces the flag exists to withhold:
    assert "write_page" not in names
    assert "linkedin_publish_draft" not in names
    assert "memory_tree_tombstone" not in names
    # The surfaces the video pipeline needs:
    assert {"search_wiki_index", "read_wiki_file", "memory_tree_recall"} <= names
