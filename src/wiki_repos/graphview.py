"""Layer-aware code-graph overview for *external* repos (PRD-012 FR-4/FR-7).

``wiki_qa.codegraph.overview`` buckets nodes into bounded contexts using SBW's
own convention (``wiki_*`` / ``search-fusion``), so for an arbitrary external
repo it collapses everything into a single ``root`` context. The per-repo graph
built by ``scripts/ua_codegraph_ext.py`` already carries correct generic
``layers`` (one per top-level directory). This module computes the same overview
shape ``synthesize``/``diagram`` consume, but buckets by those ``layers`` (with a
top-level-directory fallback) so the architecture summary reflects the real repo.

The output dict is shape-compatible with ``wiki_qa.codegraph.overview``:
``project, totals{…}, contexts[…], cross_context_coupling, foundation_contexts,
top_hubs, complexity_hotspots``.
"""

from __future__ import annotations

import collections
from typing import Any

IMPORTS_EDGE = "imports"
CONTAINS_EDGE = "contains"


def _id_to_context(graph: dict[str, Any]) -> dict[str, str]:
    """Map every node id to its bounded context, preferring the graph's layers.

    Falls back to the top-level directory of ``filePath`` for any node not
    covered by a layer (and ``root`` for top-level files)."""
    by_layer: dict[str, str] = {}
    for layer in graph.get("layers", []):
        name = layer.get("name", "?")
        for nid in layer.get("nodeIds", []):
            by_layer[nid] = name

    out: dict[str, str] = {}
    for node in graph.get("nodes", []):
        nid = node["id"]
        if nid in by_layer:
            out[nid] = by_layer[nid]
            continue
        fp = node.get("filePath", "")
        out[nid] = fp.split("/", 1)[0] if "/" in fp else "root"
    return out


def _contexts(graph: dict[str, Any], id_ctx: dict[str, str]) -> list[dict[str, Any]]:
    by_ctx: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"nodes": 0, "files": 0, "functions": 0, "classes": 0}
    )
    _plural = {"file": "files", "function": "functions", "class": "classes"}
    for n in graph.get("nodes", []):
        c = id_ctx.get(n["id"], "root")
        by_ctx[c]["nodes"] += 1
        key = _plural.get(n.get("type", ""))
        if key:
            by_ctx[c][key] += 1
    out = [{"name": c, **counts} for c, counts in by_ctx.items()]
    return sorted(out, key=lambda d: -d["nodes"])


def overview_by_layers(graph: dict[str, Any], top: int = 8) -> dict[str, Any]:
    """Whole-repo panorama bucketed by the graph's ``layers`` (external-safe)."""
    id_ctx = _id_to_context(graph)
    imports = [e for e in graph.get("edges", []) if e.get("type") == IMPORTS_EDGE]
    intra = inter = 0
    coupling: collections.Counter[str] = collections.Counter()
    indeg: collections.Counter[str] = collections.Counter()
    for e in imports:
        s_ctx, t_ctx = id_ctx.get(e["source"], "?"), id_ctx.get(e["target"], "?")
        if s_ctx == t_ctx:
            intra += 1
        else:
            inter += 1
            coupling[f"{s_ctx} -> {t_ctx}"] += 1
        indeg[e["target"]] += 1

    contains = collections.Counter(
        e["source"] for e in graph.get("edges", []) if e.get("type") == CONTAINS_EDGE
    )
    path_by_id = {n["id"]: n.get("filePath", n["id"]) for n in graph.get("nodes", [])}
    name_by_id = {n["id"]: n.get("name", n["id"]) for n in graph.get("nodes", [])}

    deps: dict[str, set[str]] = collections.defaultdict(set)
    for pair in coupling:
        s, t = pair.split(" -> ")
        deps[s].add(t)
    all_ctx = set(id_ctx.values())
    foundation = sorted(c for c in all_ctx if c not in deps)

    total_imports = intra + inter
    return {
        "project": graph.get("project", {}).get("name", "?"),
        "totals": {
            "nodes": len(graph.get("nodes", [])),
            "edges": len(graph.get("edges", [])),
            "contexts": len(all_ctx),
            "imports": total_imports,
            "intra_context_imports": intra,
            "cross_context_imports": inter,
            "encapsulation_pct": round(100 * intra / total_imports, 1) if total_imports else 0.0,
        },
        "contexts": _contexts(graph, id_ctx),
        "cross_context_coupling": dict(coupling.most_common()),
        "foundation_contexts": foundation,
        "top_hubs": [
            {"file": path_by_id.get(nid, nid), "imported_by": c}
            for nid, c in indeg.most_common(top)
        ],
        "complexity_hotspots": [
            {"file": path_by_id.get(nid, name_by_id.get(nid, nid)), "symbols": c}
            for nid, c in contains.most_common(top)
        ],
    }
