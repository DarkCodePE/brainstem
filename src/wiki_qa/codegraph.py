"""Code knowledge-graph queries over the UA ``knowledge-graph.json`` (kind=codebase).

Pure functions that answer architecture questions from the deterministic code
graph produced by ``scripts/ua-codegraph.sh``:

- :func:`list_contexts` — the bounded contexts and their sizes
- :func:`subgraph` — one context's internal nodes/edges + cross-context deps
- :func:`overview` — the whole-system panorama (coupling, hubs, hotspots)

Exposed to agents through the SBW MCP server (``wiki_agent.mcp_server``):
``code_graph_contexts`` / ``code_graph_subgraph`` / ``code_graph_overview``.
The graph is a regenerated snapshot, not a live service — callers should treat
results as "as of the last ``ua-codegraph.sh`` run".
"""

from __future__ import annotations

import collections
import json
import os
from pathlib import Path
from typing import Any

ENV_GRAPH_PATH = "SBW_CODE_GRAPH"
IMPORTS_EDGE = "imports"
CONTAINS_EDGE = "contains"


def default_graph_path() -> Path:
    """Locate the code graph: ``$SBW_CODE_GRAPH`` or ``<repo>/src/.understand-anything``."""
    override = os.environ.get(ENV_GRAPH_PATH)
    if override:
        return Path(override)
    # src/wiki_qa/codegraph.py -> repo root is two parents up from src/
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "src" / ".understand-anything" / "knowledge-graph.json"


def load_code_graph(path: Path | str | None = None) -> dict[str, Any]:
    """Load the code knowledge-graph JSON (raises FileNotFoundError if absent)."""
    p = Path(path) if path is not None else default_graph_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Code graph not found at {p}. Generate it with scripts/ua-codegraph.sh."
        )
    data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return data


def context_of(file_path: str) -> str:
    """Map a file path to its bounded context (top-level dir)."""
    top = file_path.split("/", 1)[0] if "/" in file_path else file_path
    if top.startswith("wiki_") or top == "search-fusion":
        return top
    return "root"


def _node_context(node: dict[str, Any]) -> str:
    return context_of(node.get("filePath", node.get("id", "")))


def _id_to_context(graph: dict[str, Any]) -> dict[str, str]:
    return {n["id"]: _node_context(n) for n in graph.get("nodes", [])}


def list_contexts(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """List bounded contexts with node and file counts, largest first."""
    nodes = graph.get("nodes", [])
    by_ctx: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"nodes": 0, "files": 0, "functions": 0, "classes": 0}
    )
    for n in nodes:
        c = _node_context(n)
        by_ctx[c]["nodes"] += 1
        t = n.get("type")
        if t == "file":
            by_ctx[c]["files"] += 1
        elif t == "function":
            by_ctx[c]["functions"] += 1
        elif t == "class":
            by_ctx[c]["classes"] += 1
    out = [{"name": c, **counts} for c, counts in by_ctx.items()]
    return sorted(out, key=lambda d: -d["nodes"])


def subgraph(graph: dict[str, Any], context: str, hub_limit: int = 10) -> dict[str, Any]:
    """Return one context's internal graph plus its cross-context dependencies."""
    id_ctx = _id_to_context(graph)
    ids_in = {nid for nid, c in id_ctx.items() if c == context}
    if not ids_in:
        known = sorted({c for c in id_ctx.values()})
        return {"error": f"Unknown context '{context}'", "known_contexts": known}

    nodes_in = [n for n in graph.get("nodes", []) if n["id"] in ids_in]
    internal_edges: list[dict[str, Any]] = []
    depends_on: collections.Counter[str] = collections.Counter()
    depended_on_by: collections.Counter[str] = collections.Counter()
    indeg: collections.Counter[str] = collections.Counter()

    for e in graph.get("edges", []):
        s, t = e["source"], e["target"]
        s_in, t_in = s in ids_in, t in ids_in
        if s_in and t_in:
            internal_edges.append(e)
        elif s_in and not t_in and e["type"] == IMPORTS_EDGE:
            depends_on[id_ctx.get(t, "?")] += 1
        elif t_in and not s_in and e["type"] == IMPORTS_EDGE:
            depended_on_by[id_ctx.get(s, "?")] += 1
        if t_in and e["type"] == IMPORTS_EDGE:
            indeg[t] += 1

    file_nodes = [n for n in nodes_in if n.get("type") == "file"]
    hubs = sorted(file_nodes, key=lambda n: -indeg.get(n["id"], 0))[:hub_limit]
    type_counts = collections.Counter(n.get("type") for n in nodes_in)

    return {
        "context": context,
        "counts": {
            "nodes": len(nodes_in),
            "files": type_counts.get("file", 0),
            "functions": type_counts.get("function", 0),
            "classes": type_counts.get("class", 0),
            "internal_edges": len(internal_edges),
        },
        "depends_on": dict(depends_on.most_common()),
        "depended_on_by": dict(depended_on_by.most_common()),
        "hubs": [
            {"id": h["id"], "name": h["name"], "imported_by": indeg.get(h["id"], 0)}
            for h in hubs
            if indeg.get(h["id"], 0) > 0
        ],
        "files": sorted(n["filePath"] for n in file_nodes if n.get("filePath")),
    }


def overview(graph: dict[str, Any], top: int = 8) -> dict[str, Any]:
    """Whole-system panorama: contexts, cross-context coupling, hubs, hotspots."""
    id_ctx = _id_to_context(graph)
    imports = [e for e in graph.get("edges", []) if e["type"] == IMPORTS_EDGE]
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
        e["source"] for e in graph.get("edges", []) if e["type"] == CONTAINS_EDGE
    )
    name_by_id = {n["id"]: n.get("name", n["id"]) for n in graph.get("nodes", [])}
    path_by_id = {n["id"]: n.get("filePath", n["id"]) for n in graph.get("nodes", [])}

    deps: dict[str, set[str]] = collections.defaultdict(set)
    for pair, _ in coupling.items():
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
        "contexts": list_contexts(graph),
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
