"""Unit tests for the cbm -> knowledge-graph.json adapter (ADR-046 D2).

Pure mapping over cbm SQLite-shaped rows; no I/O. Verifies label/edge collapse,
endpoint filtering, id synthesis, layer bucketing — and that the mapped graph is
consumable by the existing ``graphview.overview_by_layers`` unchanged (the whole
point of the contract-preserving design).
"""

from __future__ import annotations

from wiki_repos.cbm_adapter import to_knowledge_graph
from wiki_repos.graphview import overview_by_layers


def _node(nid, label, name, file_path, start=None, end=None):
    return {
        "id": nid,
        "label": label,
        "name": name,
        "file_path": file_path,
        "start_line": start,
        "end_line": end,
    }


def _edge(src, tgt, etype):
    return {"source_id": src, "target_id": tgt, "type": etype}


# --------------------------------------------------------------------------- #
# Node label collapse
# --------------------------------------------------------------------------- #


def test_label_collapse_to_three_types() -> None:
    nodes = [
        _node(1, "File", "a.py", "pkg/a.py"),
        _node(2, "Function", "f", "pkg/a.py"),
        _node(3, "Method", "m", "pkg/a.py"),
        _node(4, "Class", "C", "pkg/a.py"),
        _node(5, "Interface", "I", "pkg/a.py"),
        _node(6, "Enum", "E", "pkg/a.py"),
    ]
    g = to_knowledge_graph(nodes, [], project_name="o__p")
    types = {n["name"]: n["type"] for n in g["nodes"]}
    assert types == {
        "a.py": "file",
        "f": "function",
        "m": "function",
        "C": "class",
        "I": "class",
        "E": "class",
    }


def test_unmapped_labels_dropped() -> None:
    nodes = [
        _node(1, "Function", "f", "pkg/a.py"),
        _node(2, "Module", "mod", "pkg/a.py"),
        _node(3, "Variable", "v", "pkg/a.py"),
        _node(4, "Decorator", "d", "pkg/a.py"),
        _node(5, "Folder", "pkg", "pkg"),
        _node(6, "Project", "proj", ""),
    ]
    g = to_knowledge_graph(nodes, [], project_name="o__p")
    assert [n["name"] for n in g["nodes"]] == ["f"]


def test_id_synthesis_ua_form() -> None:
    g = to_knowledge_graph([_node(1, "Function", "run", "src/agent.py")], [], project_name="o__p")
    assert g["nodes"][0]["id"] == "function:src/agent.py:run"


def test_collision_disambiguated_by_line() -> None:
    nodes = [
        _node(1, "Function", "f", "a.py", start=10),
        _node(2, "Function", "f", "a.py", start=20),  # same synth id -> suffixed by line
    ]
    g = to_knowledge_graph(nodes, [], project_name="o__p")
    ids = sorted(n["id"] for n in g["nodes"])
    assert ids == ["function:a.py:f", "function:a.py:f:20"]


def test_line_ranges_preserved() -> None:
    g = to_knowledge_graph([_node(1, "Function", "f", "a.py", 5, 9)], [], project_name="o__p")
    n = g["nodes"][0]
    assert n["lineStart"] == 5 and n["lineEnd"] == 9


# --------------------------------------------------------------------------- #
# Edge type collapse + endpoint filtering
# --------------------------------------------------------------------------- #


def test_edge_type_collapse() -> None:
    nodes = [
        _node(1, "File", "a.py", "a.py"),
        _node(2, "Function", "f", "a.py"),
        _node(3, "Function", "g", "a.py"),
        _node(4, "Class", "C", "a.py"),
    ]
    edges = [
        _edge(1, 2, "DEFINES"),
        _edge(4, 2, "DEFINES_METHOD"),
        _edge(2, 3, "CALLS"),
        _edge(1, 4, "IMPORTS"),
    ]
    g = to_knowledge_graph(nodes, edges, project_name="o__p")
    got = sorted(e["type"] for e in g["edges"])
    assert got == ["calls", "contains", "contains", "imports"]


def test_unmapped_edges_dropped() -> None:
    nodes = [_node(1, "Function", "f", "a.py"), _node(2, "Function", "g", "a.py")]
    edges = [
        _edge(1, 2, "CALLS"),
        _edge(1, 2, "USAGE"),
        _edge(1, 2, "SEMANTICALLY_RELATED"),
        _edge(1, 2, "TESTS"),
    ]
    g = to_knowledge_graph(nodes, edges, project_name="o__p")
    assert [e["type"] for e in g["edges"]] == ["calls"]


def test_edge_to_dropped_node_is_removed() -> None:
    """A CALLS edge whose endpoint is a dropped Module node must not survive."""
    nodes = [_node(1, "Function", "f", "a.py"), _node(2, "Module", "mod", "a.py")]
    g = to_knowledge_graph(nodes, [_edge(2, 1, "CALLS")], project_name="o__p")
    assert g["edges"] == []


def test_duplicate_edges_deduped() -> None:
    nodes = [_node(1, "Function", "f", "a.py"), _node(2, "Function", "g", "a.py")]
    edges = [_edge(1, 2, "CALLS"), _edge(1, 2, "CALLS")]
    g = to_knowledge_graph(nodes, edges, project_name="o__p")
    assert len(g["edges"]) == 1


def test_edges_reference_synthesised_ids() -> None:
    nodes = [_node(7, "Function", "f", "a.py"), _node(8, "Function", "g", "a.py")]
    g = to_knowledge_graph(nodes, [_edge(7, 8, "CALLS")], project_name="o__p")
    e = g["edges"][0]
    assert e["source"] == "function:a.py:f" and e["target"] == "function:a.py:g"


# --------------------------------------------------------------------------- #
# Layers + project + empties
# --------------------------------------------------------------------------- #


def test_layers_by_top_level_dir() -> None:
    nodes = [
        _node(1, "Function", "f", "src/a.py"),
        _node(2, "Function", "g", "src/b.py"),
        _node(3, "Function", "t", "tests/t.py"),
        _node(4, "File", "readme", "README.md"),  # top-level file -> root
    ]
    g = to_knowledge_graph(nodes, [], project_name="o__p")
    layers = {layer["name"]: len(layer["nodeIds"]) for layer in g["layers"]}
    assert layers == {"src": 2, "tests": 1, "root": 1}
    # largest first
    assert g["layers"][0]["name"] == "src"


def test_project_name_passthrough() -> None:
    g = to_knowledge_graph([_node(1, "File", "a", "a.py")], [], project_name="acme__widget")
    assert g["project"]["name"] == "acme__widget"


def test_empty_input_yields_empty_nodes() -> None:
    g = to_knowledge_graph([], [], project_name="o__p")
    assert g["nodes"] == [] and g["edges"] == [] and g["layers"] == []


def test_only_dropped_labels_yields_empty() -> None:
    g = to_knowledge_graph([_node(1, "Module", "m", "a.py")], [], project_name="o__p")
    assert g["nodes"] == []


# --------------------------------------------------------------------------- #
# Contract integration: the mapped graph feeds overview_by_layers unchanged
# --------------------------------------------------------------------------- #


def test_mapped_graph_consumable_by_overview() -> None:
    nodes = [
        _node(1, "File", "a.py", "src/a.py"),
        _node(2, "Function", "f", "src/a.py"),
        _node(3, "Class", "C", "core/c.py"),
        _node(4, "File", "c.py", "core/c.py"),
    ]
    edges = [
        _edge(1, 2, "DEFINES"),
        _edge(2, 3, "IMPORTS"),  # src -> core : a cross-context import
    ]
    g = to_knowledge_graph(nodes, edges, project_name="o__p")
    ov = overview_by_layers(g)
    assert ov["project"] == "o__p"
    assert ov["totals"]["nodes"] == 4
    # the cross-layer import shows up as coupling (UA returned {} here)
    assert ov["totals"]["cross_context_imports"] == 1
    assert "src -> core" in ov["cross_context_coupling"]
