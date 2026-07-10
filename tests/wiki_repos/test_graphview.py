"""Tests for ``wiki_repos.graphview.overview_by_layers`` — layer-aware bucketing.

The whole point: an external repo's graph must NOT collapse to a single ``root``
context the way ``wiki_qa.codegraph.overview`` does. Bucketing follows the
graph's own ``layers`` (top-level dirs), with a filePath fallback.
"""

from __future__ import annotations

from wiki_repos.graphview import overview_by_layers


def _graph() -> dict:
    # Three top-level dirs => three contexts; routes imports services + core.
    return {
        "project": {"name": "odysseus"},
        "nodes": [
            {"id": "file:routes/api.py", "type": "file", "filePath": "routes/api.py"},
            {"id": "file:services/llm.py", "type": "file", "filePath": "services/llm.py"},
            {"id": "file:core/db.py", "type": "file", "filePath": "core/db.py"},
            {"id": "fn:core/db.py:q", "type": "function", "filePath": "core/db.py"},
            {"id": "cls:core/db.py:DB", "type": "class", "filePath": "core/db.py"},
        ],
        "edges": [
            {"source": "file:routes/api.py", "target": "file:services/llm.py", "type": "imports"},
            {"source": "file:routes/api.py", "target": "file:core/db.py", "type": "imports"},
            {"source": "file:services/llm.py", "target": "file:core/db.py", "type": "imports"},
            {"source": "file:core/db.py", "target": "fn:core/db.py:q", "type": "contains"},
        ],
        "layers": [
            {"id": "layer:routes", "name": "routes", "nodeIds": ["file:routes/api.py"]},
            {"id": "layer:services", "name": "services", "nodeIds": ["file:services/llm.py"]},
            {"id": "layer:core", "name": "core", "nodeIds": ["file:core/db.py", "fn:core/db.py:q"]},
        ],
    }


def test_buckets_by_layers_not_root():
    ov = overview_by_layers(_graph())
    names = {c["name"] for c in ov["contexts"]}
    assert names == {"routes", "services", "core"}
    assert ov["totals"]["contexts"] == 3
    assert "root" not in names


def test_cross_context_coupling_and_foundation():
    ov = overview_by_layers(_graph())
    # routes -> services and routes -> core and services -> core are cross-context.
    assert ov["totals"]["cross_context_imports"] == 3
    coupling = ov["cross_context_coupling"]
    assert "routes -> core" in coupling
    # core depends on nobody => it is a foundation context.
    assert "core" in ov["foundation_contexts"]
    assert "routes" not in ov["foundation_contexts"]


def test_top_hubs_and_hotspots():
    ov = overview_by_layers(_graph())
    hubs = {h["file"]: h["imported_by"] for h in ov["top_hubs"]}
    assert hubs.get("core/db.py") == 2  # imported by routes + services
    # core/db.py contains a function => a complexity hotspot.
    hotspots = {h["file"] for h in ov["complexity_hotspots"]}
    assert "core/db.py" in hotspots


def test_fallback_to_filepath_when_no_layer():
    g = _graph()
    g["layers"] = []  # force the top-level-dir fallback
    ov = overview_by_layers(g)
    names = {c["name"] for c in ov["contexts"]}
    assert names == {"routes", "services", "core"}
