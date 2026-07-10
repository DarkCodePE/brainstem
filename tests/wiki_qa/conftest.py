"""Shared fixtures for `wiki_qa` tests — a small synthetic UA knowledge graph.

The fixture graph is hand-crafted to exercise every health check exactly once:

- ``article:orphans/lonely`` and ``article:other/foo`` have no ``related`` edges
  -> orphans.
- ``concepts/foo`` links ``[[missing-page]]`` which resolves to nothing
  -> one broken wikilink. Its ``[[bar]]`` link resolves via unique basename.
- ``entities/bar`` links ``[[foo]]`` which resolves via the ``/foo`` suffix
  (so it is NOT broken even though ``foo`` is an ambiguous basename).
- ``foo`` appears under both ``concepts/`` and ``other/`` -> one duplicate slug.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wiki_qa.graph import KnowledgeGraph, graph_from_dict


def _graph_dict() -> dict[str, Any]:
    return {
        "version": "1.0.0",
        "kind": "knowledge",
        "project": {"name": "Test Wiki"},
        "nodes": [
            {
                "id": "article:concepts/foo",
                "type": "article",
                "name": "Foo",
                "filePath": "concepts/foo.md",
                "knowledgeMeta": {"wikilinks": ["bar", "missing-page"], "category": "Concepts"},
            },
            {
                "id": "article:entities/bar",
                "type": "article",
                "name": "Bar",
                "filePath": "entities/bar.md",
                "knowledgeMeta": {"wikilinks": ["foo"]},
            },
            {
                "id": "article:orphans/lonely",
                "type": "article",
                "name": "Lonely",
                "filePath": "orphans/lonely.md",
                "knowledgeMeta": {"wikilinks": []},
            },
            {
                "id": "article:other/foo",
                "type": "article",
                "name": "Foo (dup)",
                "filePath": "other/foo.md",
                "knowledgeMeta": {"wikilinks": []},
            },
            {"id": "topic:concepts", "type": "topic", "name": "Concepts"},
            {
                "id": "source:raw/x",
                "type": "source",
                "name": "x.md",
                "filePath": "raw/x.md",
            },
        ],
        "edges": [
            {"source": "article:concepts/foo", "target": "article:entities/bar", "type": "related"},
            {"source": "article:entities/bar", "target": "article:concepts/foo", "type": "related"},
            {
                "source": "article:concepts/foo",
                "target": "topic:concepts",
                "type": "categorized_under",
            },
        ],
        "layers": [],
        "tour": [
            {
                "order": 1,
                "title": "Concepts",
                "description": "Explore the Concepts section (1 article)",
                "nodeIds": ["article:concepts/foo"],
            }
        ],
    }


@pytest.fixture
def graph_dict() -> dict[str, Any]:
    return _graph_dict()


@pytest.fixture
def graph() -> KnowledgeGraph:
    return graph_from_dict(_graph_dict())


@pytest.fixture
def graph_file(tmp_path: Path) -> Path:
    path = tmp_path / "knowledge-graph.json"
    path.write_text(json.dumps(_graph_dict()), encoding="utf-8")
    return path
