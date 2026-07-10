#!/usr/bin/env python3
"""Generalized deterministic code knowledge-graph builder (PRD-012 / ADR-022).

Generalized variant of ``scripts/ua_codegraph.py``. Converts Understand-Anything's
``extract-structure`` output + tree-sitter import map into a schema-valid
``knowledge-graph.json`` with bounded-context layers — no LLM.

Differences from the SBW-specific builder:

- :func:`context_of` is GENERIC: the bounded context of a node is simply the
  top-level directory of its ``filePath`` (no ``wiki_*`` / ``search-fusion``
  special-casing). A top-level file (no ``/``) lands in the ``root`` context.
- The project ``name`` comes from an argument, and ``languages`` are DETECTED
  from the scanned files rather than hardcoded.

Used by ``scripts/ua-codegraph-ext.sh`` to analyze arbitrary external repos.

Usage:
    python3 ua_codegraph_ext.py <intermediate-dir> <git-sha> <project-name> [out-path]

If ``out-path`` is omitted, the graph is written to
``<intermediate-dir>/../knowledge-graph.json`` (mirroring the original).
"""

from __future__ import annotations

import collections
import datetime
import json
import sys
from pathlib import Path


def complexity(lines: int) -> str:
    """Bucket a line/symbol count into a coarse complexity band."""
    if lines < 100:
        return "simple"
    if lines < 300:
        return "moderate"
    return "complex"


def basename(path: str) -> str:
    """Final path component (POSIX-style), used as a node display name."""
    return path.rsplit("/", 1)[-1]


def context_of(node: dict) -> str:
    """Map a node to its bounded context: the top-level dir of its filePath.

    Generic scheme (no project-specific special-casing): the first path segment
    is the context; a top-level file with no ``/`` is assigned to ``root``.

    >>> context_of({"filePath": "pkg/sub/mod.py"})
    'pkg'
    >>> context_of({"filePath": "main.py"})
    'root'
    >>> context_of({"filePath": ""})
    'root'
    """
    fp = node.get("filePath", "")
    if "/" in fp:
        return fp.split("/", 1)[0]
    return "root"


def build_batches(int_dir: Path) -> tuple[list[dict], list[dict]]:
    """Walk every batch's extract-structure output into graph nodes + edges.

    Faithfully mirrors the node/edge derivation of the SBW-specific builder:
    files, sufficiently-large/exported functions and multi-method/exported
    classes become nodes; ``contains`` / ``exports`` / ``imports`` / ``calls``
    become edges.
    """
    batches = json.loads((int_dir / "batches.json").read_text())["batches"]
    nodes: list[dict] = []
    edges: list[dict] = []
    node_ids: set[str] = set()

    def add(node: dict) -> None:
        if node["id"] not in node_ids:
            node_ids.add(node["id"])
            nodes.append(node)

    for b in batches:
        idx = b["batchIndex"]
        es = json.loads((int_dir / f"es-output-{idx}.json").read_text())
        import_data = b.get("batchImportData", {})
        for r in es["results"]:
            path = r["path"]
            lang = r.get("language", "")
            cat = r.get("fileCategory", "code")
            file_id = f"file:{path}"
            add(
                {
                    "id": file_id,
                    "type": "file",
                    "name": basename(path),
                    "filePath": path,
                    "summary": f"{cat} file ({lang}, {r.get('totalLines', 0)} lines).",
                    "tags": [t for t in (lang, cat) if t],
                    "complexity": complexity(r.get("totalLines", 0)),
                }
            )
            exported = {e.get("name") for e in r.get("exports", [])}
            fn_names = {f.get("name") for f in r.get("functions", [])}

            for fn in r.get("functions", []):
                name = fn.get("name", "")
                span = fn.get("endLine", 0) - fn.get("startLine", 0)
                if not name or (span < 10 and name not in exported):
                    continue
                fid = f"function:{path}:{name}"
                add(
                    {
                        "id": fid,
                        "type": "function",
                        "name": name,
                        "filePath": path,
                        "summary": f"Function {name}({', '.join(fn.get('params', []))}).",
                        "tags": [lang, "function"],
                        "complexity": complexity(span),
                    }
                )
                edges.append(
                    {
                        "source": file_id,
                        "target": fid,
                        "type": "contains",
                        "direction": "forward",
                        "weight": 1.0,
                    }
                )
                if name in exported:
                    edges.append(
                        {
                            "source": file_id,
                            "target": fid,
                            "type": "exports",
                            "direction": "forward",
                            "weight": 0.8,
                        }
                    )

            for cl in r.get("classes", []):
                name = cl.get("name", "")
                methods = cl.get("methods", [])
                if not name or (len(methods) < 2 and name not in exported):
                    continue
                cid = f"class:{path}:{name}"
                add(
                    {
                        "id": cid,
                        "type": "class",
                        "name": name,
                        "filePath": path,
                        "summary": f"Class {name} ({len(methods)} methods).",
                        "tags": [lang, "class"],
                        "complexity": complexity(len(methods) * 10),
                    }
                )
                edges.append(
                    {
                        "source": file_id,
                        "target": cid,
                        "type": "contains",
                        "direction": "forward",
                        "weight": 1.0,
                    }
                )
                if name in exported:
                    edges.append(
                        {
                            "source": file_id,
                            "target": cid,
                            "type": "exports",
                            "direction": "forward",
                            "weight": 0.8,
                        }
                    )

            for tgt in import_data.get(path, []):
                edges.append(
                    {
                        "source": file_id,
                        "target": f"file:{tgt}",
                        "type": "imports",
                        "direction": "forward",
                        "weight": 0.7,
                    }
                )

            for call in r.get("callGraph", []):
                caller, callee = call.get("caller"), call.get("callee")
                if caller in fn_names and callee in fn_names and caller != callee:
                    edges.append(
                        {
                            "source": f"function:{path}:{caller}",
                            "target": f"function:{path}:{callee}",
                            "type": "calls",
                            "direction": "forward",
                            "weight": 0.8,
                        }
                    )
    return nodes, edges


def detect_languages(nodes: list[dict]) -> list[str]:
    """Distinct file languages present, ordered by frequency (most common first)."""
    counts: collections.Counter[str] = collections.Counter()
    for n in nodes:
        if n.get("type") == "file":
            lang = (n.get("tags") or [""])[0]
            if lang:
                counts[lang] += 1
    return [lang for lang, _ in counts.most_common()]


def finalize(nodes: list[dict], edges: list[dict], *, sha: str, project_name: str) -> dict:
    """Assemble the schema-valid graph: prune dangling edges, layer by context."""
    # Drop dangling edges (endpoints outside the node set) for self-consistency.
    ids = {n["id"] for n in nodes}
    edges = [e for e in edges if e["source"] in ids and e["target"] in ids]

    by_ctx: dict[str, list[str]] = collections.defaultdict(list)
    for n in nodes:
        by_ctx[context_of(n)].append(n["id"])
    order = sorted(by_ctx, key=lambda c: -len(by_ctx[c]))
    layers = [
        {
            "id": f"layer:{c}",
            "name": c,
            "description": f"Bounded context {c} ({len(by_ctx[c])} nodes)",
            "nodeIds": by_ctx[c],
        }
        for c in order
    ]
    outdeg = collections.Counter(e["source"] for e in edges)
    tour = []
    for i, c in enumerate(order):
        files = [nid for nid in by_ctx[c] if nid.startswith("file:")]
        entry = sorted(files, key=lambda f: -outdeg.get(f, 0))[:3] or by_ctx[c][:1]
        tour.append(
            {
                "order": i + 1,
                "title": c,
                "description": f"Explore the {c} bounded context ({len(by_ctx[c])} nodes)",
                "nodeIds": entry,
            }
        )
    languages = detect_languages(nodes)
    return {
        "version": "1.0.0",
        "kind": "codebase",
        "project": {
            "name": project_name,
            "languages": languages,
            "frameworks": [],
            "description": f"Code knowledge-graph for {project_name} "
            f"({len(by_ctx)} bounded contexts).",
            "analyzedAt": datetime.datetime.now(datetime.UTC).isoformat(),
            "gitCommitHash": sha,
        },
        "nodes": nodes,
        "edges": edges,
        "layers": layers,
        "tour": tour,
    }


def main() -> None:
    """CLI entrypoint: build the graph and write knowledge-graph.json."""
    int_dir = Path(sys.argv[1])
    sha = sys.argv[2] if len(sys.argv) > 2 else ""
    project_name = sys.argv[3] if len(sys.argv) > 3 else int_dir.parent.name
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else int_dir.parent / "knowledge-graph.json"

    nodes, edges = build_batches(int_dir)
    graph = finalize(nodes, edges, sha=sha, project_name=project_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(graph, indent=2))
    print(
        f"[ua-codegraph-ext] {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, "
        f"{len(graph['layers'])} layers -> {out}"
    )


if __name__ == "__main__":
    main()
