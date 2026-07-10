"""MCP shim for ask_repo (ADR-046 Fase 3 — tool surface).

The pack/resolution logic lives (and is tested) in wiki_repos.repo_context;
these tests cover the thin MCP wrapper: WIKI_ROOT/repos wiring, focus
normalisation, max_chars clamping, and the JSON envelope.
"""

from __future__ import annotations

import json

import pytest

from wiki_agent import mcp_server

_GRAPH = {
    "project": {"name": "acme__toolkit"},
    "nodes": [
        {"id": "f1", "type": "file", "name": "core.py", "filePath": "src/core.py"},
        {
            "id": "fn1",
            "type": "function",
            "name": "distill",
            "filePath": "src/core.py",
            "lineStart": 10,
        },
    ],
    "edges": [],
    "layers": [{"name": "src", "nodeIds": ["f1", "fn1"]}],
}


@pytest.fixture
def graph_root(tmp_path, monkeypatch):
    d = tmp_path / "repos" / "acme__toolkit"
    d.mkdir(parents=True)
    (d / "knowledge-graph.json").write_text(json.dumps(_GRAPH))
    monkeypatch.setattr(mcp_server, "WIKI_ROOT", str(tmp_path))
    return tmp_path


def test_ask_repo_returns_pack_json(graph_root):
    res = json.loads(mcp_server.ask_repo("acme/toolkit", focus="distill", max_chars=100))
    assert res["repo"] == "acme__toolkit"
    # max_chars is clamped to >= 500, so the tiny budget still yields a pack.
    assert "# Repo context" in res["pack"]


def test_ask_repo_unknown_repo_lists_available(graph_root):
    res = json.loads(mcp_server.ask_repo("nope"))
    assert "error" in res
    assert res["available"] == ["acme__toolkit"]


def test_ask_repo_blank_focus_is_none(graph_root):
    res = json.loads(mcp_server.ask_repo("acme/toolkit", focus="   "))
    assert "Symbols matching" not in res["pack"]  # no focus section
