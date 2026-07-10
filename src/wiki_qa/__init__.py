"""`wiki_qa` — QA & navigation tooling over the Understand-Anything graph.

Implements the adoption of Understand-Anything (UA) as SBW's read/visualize/QA
layer — milestone *M-UA Wiki Visualization & QA* (#9). See
``docs/swarm-investigation/2026-06-01-understand-anything-vs-sbw/``.

- **UA-3** — :mod:`wiki_qa.linter` flags orphan pages, broken wikilinks, and
  duplicate slugs over the UA ``knowledge-graph.json``;
  :mod:`wiki_qa.baseline` turns that into a CI regression gate.
- **UA-4** — :mod:`wiki_qa.tour` renders the graph's guided tour to Markdown.

UA itself remains the (read-only, TypeScript) graph builder; this package is the
thin Python QA consumer that feeds SBW's ingest pipeline — it never builds or
mutates the wiki.
"""

from __future__ import annotations

from wiki_qa.baseline import (
    Regression,
    compute_regressions,
    load_baseline,
    report_keys,
    save_baseline,
)
from wiki_qa.codegraph import (
    list_contexts,
    load_code_graph,
    overview,
    subgraph,
)
from wiki_qa.graph import Edge, KnowledgeGraph, Node, graph_from_dict, load_graph
from wiki_qa.linter import (
    BrokenWikilink,
    DuplicateSlug,
    HealthReport,
    find_broken_wikilinks,
    find_duplicate_slugs,
    find_orphans,
    lint,
)
from wiki_qa.tour import render_tour

__all__ = [
    "BrokenWikilink",
    "DuplicateSlug",
    "Edge",
    "HealthReport",
    "KnowledgeGraph",
    "Node",
    "Regression",
    "compute_regressions",
    "find_broken_wikilinks",
    "find_duplicate_slugs",
    "find_orphans",
    "graph_from_dict",
    "lint",
    "list_contexts",
    "load_baseline",
    "load_code_graph",
    "load_graph",
    "overview",
    "render_tour",
    "report_keys",
    "save_baseline",
    "subgraph",
]
