"""Tests for `wiki_qa.tour` — Markdown rendering of the guided tour."""

from __future__ import annotations

from typing import Any

from wiki_qa.tour import render_tour


class TestRenderTour:
    def test_renders_title_and_steps(self, graph_dict: dict[str, Any]) -> None:
        md = render_tour(graph_dict)
        assert "# Guided Tour — Test Wiki" in md
        assert "## 1. Concepts" in md

    def test_resolves_node_names_and_paths(self, graph_dict: dict[str, Any]) -> None:
        md = render_tour(graph_dict)
        assert "**Foo** — `concepts/foo.md`" in md

    def test_missing_node_is_marked(self) -> None:
        data: dict[str, Any] = {
            "project": {"name": "X"},
            "nodes": [],
            "tour": [{"order": 1, "title": "T", "nodeIds": ["article:gone"]}],
        }
        md = render_tour(data)
        assert "_(missing node)_" in md

    def test_empty_tour_is_handled(self) -> None:
        md = render_tour({"project": {"name": "X"}, "nodes": [], "tour": []})
        assert "No tour steps" in md
