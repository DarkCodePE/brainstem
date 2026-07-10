"""Map a codebase-memory-mcp graph onto the UA knowledge-graph.json contract.

ADR-046 D2. The cbm engine (a pure-C tree-sitter + LSP indexer) persists a far
richer typed graph than SBW's UA tree-sitter builder, but every SBW consumer
(``graphview.overview_by_layers``, ``diagram``, ``wiki_qa.codequery`` impact, the
``code_graph_*`` MCP tools) reads ONE fixed shape::

    {project:{name}, nodes:[{id,type,name,filePath}],
     edges:[{source,target,type}], layers:[{name,nodeIds}]}

This module collapses cbm's taxonomy onto that contract so the producer can be
swapped without touching a single consumer:

- **labels** ``File→file``, ``Function``/``Method``→``function``,
  ``Class``/``Interface``/``Enum``/``Type``→``class``. Every other cbm label
  (``Module``/``Section``/``Variable``/``EnvVar``/``Decorator``/``Folder``/
  ``Branch``/``Project``) is dropped — it has no place in the 3-type contract.
- **edge types** ``IMPORTS``→``imports``, ``CALLS``→``calls``,
  ``DEFINES``/``DEFINES_METHOD``→``contains``. The rest
  (``USAGE``/``WRITES``/``TESTS``/``CONFIGURES``/``SEMANTICALLY_RELATED``/
  ``DECORATES``/``INHERITS``/``HAS_BRANCH``/``RAISES``) are dropped — they are
  non-structural or threshold-based; keeping only the deterministic structural
  subset means the mapped graph is reproducible (ADR-046 "rejected: semantic
  edges break determinism").
- **node ids** are re-synthesised to UA's ``<type>:<filePath>:<name>`` form so
  ``wiki_qa.codequery.resolve_symbol`` (ADR-037) keeps working unchanged.
- **layers** are the top-level directory of each node's ``filePath`` — matching
  ``graphview``'s own fallback bucketing, so ``overview_by_layers`` produces the
  same shape it does for a UA graph.

Pure functions; no I/O. ``cbm_runner`` reads the cbm SQLite and calls in here.
"""

from __future__ import annotations

from typing import Any

#: cbm node label -> contract node type. Unlisted labels are dropped.
LABEL_TO_TYPE: dict[str, str] = {
    "File": "file",
    "Function": "function",
    "Method": "function",
    "Class": "class",
    "Interface": "class",
    "Enum": "class",
    "Type": "class",
}

#: cbm edge type -> contract edge type. Unlisted types are dropped.
EDGE_TYPE_MAP: dict[str, str] = {
    "IMPORTS": "imports",
    "CALLS": "calls",
    "DEFINES": "contains",
    "DEFINES_METHOD": "contains",
}


def _synth_id(node_type: str, file_path: str, name: str) -> str:
    """UA-style node id (``<type>:<filePath>:<name>``) for resolve_symbol parity."""
    return f"{node_type}:{file_path}:{name}"


def _top_dir(file_path: str) -> str:
    """Top-level directory of a relative path; ``root`` for a top-level file."""
    return file_path.split("/", 1)[0] if "/" in file_path else "root"


def to_knowledge_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    project_name: str,
) -> dict[str, Any]:
    """Build a UA-shaped knowledge-graph dict from cbm SQLite rows.

    Args:
        nodes: cbm ``nodes`` rows (``id``/``label``/``name``/``file_path``/
            ``start_line``/``end_line`` — extra keys ignored).
        edges: cbm ``edges`` rows (``source_id``/``target_id``/``type``).
        project_name: value for ``project.name`` (caller passes ``graph_dirname``).

    Returns:
        ``{project, nodes, edges, layers}`` — contract-shaped. ``nodes`` is empty
        when the cbm graph carries nothing the contract recognises (the caller
        treats that as a degrade).
    """
    # 1. Keep contract-typed nodes; map cbm id -> synthesised contract id.
    id_map: dict[Any, str] = {}
    out_nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for n in nodes:
        node_type = LABEL_TO_TYPE.get(n.get("label"))
        if node_type is None:
            continue
        file_path = n.get("file_path") or ""
        name = n.get("name") or ""
        sid = _synth_id(node_type, file_path, name)
        if sid in seen_ids:
            # Disambiguate overloads / same-named symbols by start line.
            sid = f"{sid}:{n.get('start_line', '')}"
            if sid in seen_ids:
                continue
        seen_ids.add(sid)
        id_map[n.get("id")] = sid
        node_out: dict[str, Any] = {
            "id": sid,
            "type": node_type,
            "name": name,
            "filePath": file_path,
        }
        # Line ranges are a bonus cbm carries and UA never did (ADR-037 noted the
        # gap). Harmless extra keys for current consumers; useful for future ones.
        if n.get("start_line") is not None:
            node_out["lineStart"] = n["start_line"]
        if n.get("end_line") is not None:
            node_out["lineEnd"] = n["end_line"]
        out_nodes.append(node_out)

    # 2. Keep mapped edges whose BOTH endpoints survived; dedup.
    out_edges: list[dict[str, Any]] = []
    edge_seen: set[tuple[str, str, str]] = set()
    for e in edges:
        edge_type = EDGE_TYPE_MAP.get(e.get("type"))
        if edge_type is None:
            continue
        source = id_map.get(e.get("source_id"))
        target = id_map.get(e.get("target_id"))
        if source is None or target is None:
            continue
        key = (source, target, edge_type)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        out_edges.append({"source": source, "target": target, "type": edge_type})

    # 3. Layers = top-level dir of each surviving node (largest first).
    by_layer: dict[str, list[str]] = {}
    for node_out in out_nodes:
        by_layer.setdefault(_top_dir(node_out["filePath"]), []).append(node_out["id"])
    layers = [
        {"name": name, "nodeIds": ids}
        for name, ids in sorted(by_layer.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]

    return {
        "project": {"name": project_name},
        "nodes": out_nodes,
        "edges": out_edges,
        "layers": layers,
    }
