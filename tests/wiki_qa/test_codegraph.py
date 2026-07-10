"""Tests for `wiki_qa.codegraph` — context/subgraph/overview queries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wiki_qa.codegraph import (
    context_of,
    list_contexts,
    load_code_graph,
    overview,
    subgraph,
)


def _code_graph() -> dict[str, Any]:
    # Two contexts: wiki_a (orchestrator) imports wiki_b (foundation).
    return {
        "version": "1.0.0",
        "kind": "codebase",
        "project": {"name": "test-src"},
        "nodes": [
            {
                "id": "file:wiki_a/svc.py",
                "type": "file",
                "name": "svc.py",
                "filePath": "wiki_a/svc.py",
                "summary": "",
                "tags": [],
                "complexity": "simple",
            },
            {
                "id": "function:wiki_a/svc.py:run",
                "type": "function",
                "name": "run",
                "filePath": "wiki_a/svc.py",
                "summary": "",
                "tags": [],
                "complexity": "simple",
            },
            {
                "id": "file:wiki_b/core.py",
                "type": "file",
                "name": "core.py",
                "filePath": "wiki_b/core.py",
                "summary": "",
                "tags": [],
                "complexity": "simple",
            },
            {
                "id": "file:wiki_b/util.py",
                "type": "file",
                "name": "util.py",
                "filePath": "wiki_b/util.py",
                "summary": "",
                "tags": [],
                "complexity": "simple",
            },
            {
                "id": "class:wiki_b/core.py:Engine",
                "type": "class",
                "name": "Engine",
                "filePath": "wiki_b/core.py",
                "summary": "",
                "tags": [],
                "complexity": "simple",
            },
        ],
        "edges": [
            {
                "source": "file:wiki_a/svc.py",
                "target": "function:wiki_a/svc.py:run",
                "type": "contains",
                "direction": "forward",
                "weight": 1.0,
            },
            {
                "source": "file:wiki_b/core.py",
                "target": "class:wiki_b/core.py:Engine",
                "type": "contains",
                "direction": "forward",
                "weight": 1.0,
            },
            # cross-context: wiki_a -> wiki_b
            {
                "source": "file:wiki_a/svc.py",
                "target": "file:wiki_b/core.py",
                "type": "imports",
                "direction": "forward",
                "weight": 0.7,
            },
            # intra-context: wiki_b/util -> wiki_b/core
            {
                "source": "file:wiki_b/util.py",
                "target": "file:wiki_b/core.py",
                "type": "imports",
                "direction": "forward",
                "weight": 0.7,
            },
        ],
        "layers": [],
        "tour": [],
    }


@pytest.fixture
def cg() -> dict[str, Any]:
    return _code_graph()


class TestContextOf:
    def test_wiki_prefix(self) -> None:
        assert context_of("wiki_routing/router.py") == "wiki_routing"

    def test_non_context_is_root(self) -> None:
        assert context_of("pyproject.toml") == "root"


class TestListContexts:
    def test_counts_and_ordering(self, cg: dict[str, Any]) -> None:
        ctxs = list_contexts(cg)
        names = [c["name"] for c in ctxs]
        assert names == ["wiki_b", "wiki_a"]  # wiki_b has 3 nodes, wiki_a has 2
        wiki_b = next(c for c in ctxs if c["name"] == "wiki_b")
        assert wiki_b["files"] == 2
        assert wiki_b["classes"] == 1


class TestSubgraph:
    def test_dependencies_direction(self, cg: dict[str, Any]) -> None:
        sg = subgraph(cg, "wiki_a")
        assert sg["depends_on"] == {"wiki_b": 1}
        assert sg["depended_on_by"] == {}

    def test_reverse_dependency(self, cg: dict[str, Any]) -> None:
        sg = subgraph(cg, "wiki_b")
        assert sg["depended_on_by"] == {"wiki_a": 1}
        assert sg["depends_on"] == {}

    def test_internal_edges_and_hubs(self, cg: dict[str, Any]) -> None:
        sg = subgraph(cg, "wiki_b")
        # core.py imported by util.py (intra) + svc.py (cross) = 2
        hub = next(h for h in sg["hubs"] if h["name"] == "core.py")
        assert hub["imported_by"] == 2
        # internal edges: contains(core->Engine) + imports(util->core) = 2
        assert sg["counts"]["internal_edges"] == 2

    def test_unknown_context(self, cg: dict[str, Any]) -> None:
        sg = subgraph(cg, "wiki_nope")
        assert "error" in sg
        assert "wiki_a" in sg["known_contexts"]


class TestOverview:
    def test_encapsulation_and_coupling(self, cg: dict[str, Any]) -> None:
        ov = overview(cg)
        assert ov["totals"]["cross_context_imports"] == 1
        assert ov["totals"]["intra_context_imports"] == 1
        assert ov["totals"]["encapsulation_pct"] == 50.0
        assert ov["cross_context_coupling"] == {"wiki_a -> wiki_b": 1}

    def test_foundation_contexts(self, cg: dict[str, Any]) -> None:
        ov = overview(cg)
        # wiki_b imports nothing cross-context -> foundation
        assert "wiki_b" in ov["foundation_contexts"]
        assert "wiki_a" not in ov["foundation_contexts"]


class TestLoadCodeGraph:
    def test_load_from_path(self, cg: dict[str, Any], tmp_path: Path) -> None:
        p = tmp_path / "kg.json"
        p.write_text(json.dumps(cg), encoding="utf-8")
        loaded = load_code_graph(p)
        assert loaded["project"]["name"] == "test-src"

    def test_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_code_graph(tmp_path / "nope.json")
