"""Typed loader for the Understand-Anything knowledge-graph JSON.

Consumes the graph produced by the UA ``/understand-knowledge`` pipeline
(``parse-knowledge-base.py`` + ``merge-knowledge-graph.py``) run against SBW's
``knowledge-base/`` wiki. See
``docs/swarm-investigation/2026-06-01-understand-anything-vs-sbw/`` (issue UA-3).

The loader is intentionally tolerant: UA may emit either the intermediate
``scan-manifest.json`` (``nodes``/``edges``/``warnings``) or the assembled
``knowledge-graph.json`` (adds ``layers``/``tour``). Only the fields the QA
linter needs are projected into the frozen dataclasses below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTICLE_TYPE = "article"


@dataclass(frozen=True)
class Node:
    """A single graph node (article, topic, source, entity, ...)."""

    id: str
    type: str
    name: str
    file_path: str | None
    wikilinks: tuple[str, ...]
    category: str | None


@dataclass(frozen=True)
class Edge:
    """A directed relationship between two node ids."""

    source: str
    target: str
    type: str


@dataclass(frozen=True)
class KnowledgeGraph:
    """An immutable view over the nodes and edges of a UA knowledge graph."""

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    @property
    def articles(self) -> tuple[Node, ...]:
        """Only the article (wiki page) nodes — the unit the linter checks."""
        return tuple(n for n in self.nodes if n.type == ARTICLE_TYPE)


def graph_from_dict(data: dict[str, Any]) -> KnowledgeGraph:
    """Project a parsed UA graph ``dict`` into a :class:`KnowledgeGraph`."""
    nodes: list[Node] = []
    for raw in data.get("nodes", []):
        meta = raw.get("knowledgeMeta") or {}
        wikilinks = meta.get("wikilinks") or []
        nodes.append(
            Node(
                id=str(raw["id"]),
                type=str(raw.get("type", "")),
                name=str(raw.get("name", "")),
                file_path=raw.get("filePath"),
                wikilinks=tuple(str(w) for w in wikilinks),
                category=meta.get("category"),
            )
        )
    edges: list[Edge] = []
    for raw in data.get("edges", []):
        edges.append(
            Edge(
                source=str(raw["source"]),
                target=str(raw["target"]),
                type=str(raw.get("type", "")),
            )
        )
    return KnowledgeGraph(nodes=tuple(nodes), edges=tuple(edges))


def load_graph(path: Path | str) -> KnowledgeGraph:
    """Load and project a UA knowledge-graph JSON file from disk."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    return graph_from_dict(data)
