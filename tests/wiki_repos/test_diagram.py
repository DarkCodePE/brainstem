"""Tests for `wiki_repos.diagram` — deterministic Mermaid from a code graph."""

from __future__ import annotations

from typing import Any

import pytest

from wiki_repos.diagram import diagram_is_renderable, mermaid_from_graph


def _code_graph() -> dict[str, Any]:
    """Mirror of `tests/wiki_qa/test_codegraph.py::_code_graph`.

    Two contexts: ``wiki_a`` (orchestrator) imports ``wiki_b`` (foundation),
    plus one intra-context ``wiki_b`` import.
    """
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
            },
            {
                "id": "function:wiki_a/svc.py:run",
                "type": "function",
                "name": "run",
                "filePath": "wiki_a/svc.py",
            },
            {
                "id": "file:wiki_b/core.py",
                "type": "file",
                "name": "core.py",
                "filePath": "wiki_b/core.py",
            },
            {
                "id": "file:wiki_b/util.py",
                "type": "file",
                "name": "util.py",
                "filePath": "wiki_b/util.py",
            },
            {
                "id": "class:wiki_b/core.py:Engine",
                "type": "class",
                "name": "Engine",
                "filePath": "wiki_b/core.py",
            },
        ],
        "edges": [
            {
                "source": "file:wiki_a/svc.py",
                "target": "function:wiki_a/svc.py:run",
                "type": "contains",
            },
            {
                "source": "file:wiki_b/core.py",
                "target": "class:wiki_b/core.py:Engine",
                "type": "contains",
            },
            # cross-context: wiki_a -> wiki_b
            {"source": "file:wiki_a/svc.py", "target": "file:wiki_b/core.py", "type": "imports"},
            # intra-context: wiki_b/util -> wiki_b/core
            {"source": "file:wiki_b/util.py", "target": "file:wiki_b/core.py", "type": "imports"},
        ],
        "layers": [],
    }


def _no_cross_graph() -> dict[str, Any]:
    """Two contexts but ZERO cross-context import edges (only intra-context)."""
    return {
        "version": "1.0.0",
        "kind": "codebase",
        "project": {"name": "isolated"},
        "nodes": [
            {"id": "file:wiki_a/a.py", "type": "file", "name": "a.py", "filePath": "wiki_a/a.py"},
            {"id": "file:wiki_a/b.py", "type": "file", "name": "b.py", "filePath": "wiki_a/b.py"},
            {"id": "file:wiki_b/c.py", "type": "file", "name": "c.py", "filePath": "wiki_b/c.py"},
            {"id": "file:wiki_b/d.py", "type": "file", "name": "d.py", "filePath": "wiki_b/d.py"},
        ],
        "edges": [
            # intra wiki_a
            {"source": "file:wiki_a/b.py", "target": "file:wiki_a/a.py", "type": "imports"},
            # intra wiki_b
            {"source": "file:wiki_b/d.py", "target": "file:wiki_b/c.py", "type": "imports"},
        ],
        "layers": [],
    }


@pytest.fixture
def cg() -> dict[str, Any]:
    return _code_graph()


class TestMermaidFromGraph:
    def test_renders_flowchart_with_contexts_and_edge(self, cg: dict[str, Any]) -> None:
        out = mermaid_from_graph(cg)
        assert out.startswith("```mermaid\n")
        assert out.endswith("\n```")
        assert "flowchart" in out
        # context slugs present
        assert "wiki_a" in out
        assert "wiki_b" in out
        # a cross-context edge is drawn
        assert "-->" in out
        assert "wiki_a -->|1| wiki_b" in out
        assert diagram_is_renderable(out) is True

    def test_node_labels_include_counts(self, cg: dict[str, Any]) -> None:
        out = mermaid_from_graph(cg)
        # wiki_b has 3 nodes, wiki_a has 2 (see codegraph test ordering)
        assert 'wiki_b["wiki_b (3 nodes)"]' in out
        assert 'wiki_a["wiki_a (2 nodes)"]' in out

    def test_none_graph_returns_empty(self) -> None:
        assert mermaid_from_graph(None) == ""

    def test_empty_graph_returns_empty(self) -> None:
        assert mermaid_from_graph({"nodes": [], "edges": []}) == ""
        assert mermaid_from_graph({}) == ""

    def test_no_cross_context_still_renders_nodes(self) -> None:
        out = mermaid_from_graph(_no_cross_graph())
        assert out != ""
        assert "flowchart" in out
        assert "wiki_a" in out
        assert "wiki_b" in out
        # no cross-context coupling -> no edges
        assert "-->" not in out
        assert diagram_is_renderable(out) is True

    def test_max_nodes_limits_contexts(self, cg: dict[str, Any]) -> None:
        out = mermaid_from_graph(cg, max_nodes=1)
        # only the largest context (wiki_b) is drawn; cross edge needs both -> dropped
        assert "wiki_b" in out
        assert "-->" not in out
        assert diagram_is_renderable(out) is True

    def test_max_edges_zero_drops_edges(self, cg: dict[str, Any]) -> None:
        out = mermaid_from_graph(cg, max_edges=0)
        assert "-->" not in out
        assert "wiki_a" in out and "wiki_b" in out
        assert diagram_is_renderable(out) is True

    def test_deterministic(self, cg: dict[str, Any]) -> None:
        assert mermaid_from_graph(cg) == mermaid_from_graph(cg)


class TestDiagramIsRenderable:
    def test_empty_is_false(self) -> None:
        assert diagram_is_renderable("") is False
        assert diagram_is_renderable(None) is False  # type: ignore[arg-type]

    def test_missing_fence_is_false(self) -> None:
        assert diagram_is_renderable("flowchart LR\n    a --> b") is False

    def test_missing_flowchart_is_false(self) -> None:
        assert diagram_is_renderable('```mermaid\n    a["x"]\n```') is False

    def test_undeclared_edge_endpoint_is_false(self) -> None:
        bad = '```mermaid\nflowchart LR\n    a["A"]\n    a -->|1| b\n```'
        assert diagram_is_renderable(bad) is False

    def test_well_formed_is_true(self) -> None:
        good = '```mermaid\nflowchart LR\n    a["A"]\n    b["B"]\n    a -->|2| b\n```'
        assert diagram_is_renderable(good) is True
