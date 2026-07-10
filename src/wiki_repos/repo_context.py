"""Token-bounded structural context pack for agent / Q&A retrieval (ADR-046 Phase 3).

The repo-ingest *synthesis* path is already token-lean (its only LLM call sees a
~1.3 K-token assembled page, not the raw digest), so there is nothing to "rewire"
there. The real cbm token win is in **agent retrieval**: instead of an agent
grep/read/cat-ing whole files into its context to answer "what is this repo / where
is X", it can read one compact structural pack built from the deterministic code
graph.

``build_context_pack`` turns a contract-shaped graph (from either backend, but
richest from cbm — calls, real coupling, ``lineStart/lineEnd``) into a small
markdown bundle:

- an architecture overview (contexts, coupling, encapsulation) via
  :func:`wiki_repos.graphview.overview_by_layers`;
- the top import hubs (the files everything depends on);
- when a ``focus`` term is given, the matching symbols with ``file:line`` — and,
  if ``repo_dir`` is provided, a few bounded source slices for them.

Pure and deterministic (no LLM, no network). The whole thing is capped at
``max_chars`` so the caller has a hard token budget. ``mcp_server.ask_repo``
exposes it over MCP; :func:`find_repo_graph` resolves a free-form repo query to
the per-repo graph store (``<wiki_root>/repos/<owner>__<repo>/``, PRD-012 FR-4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wiki_repos.graphview import overview_by_layers

GRAPH_FILENAME = "knowledge-graph.json"

_DEFAULT_MAX_CHARS = 8000
_MAX_CONTEXTS = 8
_MAX_COUPLING = 8
_MAX_HUBS = 10
_MAX_FOCUS_HITS = 12
_MAX_SLICE_LINES = 40


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[TRUNCATED: context pack exceeded budget]"


def _matches(node: dict[str, Any], focus: str) -> bool:
    f = focus.lower()
    return f in (node.get("name") or "").lower() or f in (node.get("filePath") or "").lower()


def _source_slice(repo_dir: Path, node: dict[str, Any]) -> str | None:
    """A few source lines for a node with a line range (best-effort, bounded)."""
    fp = node.get("filePath")
    start, end = node.get("lineStart"), node.get("lineEnd")
    if not fp or start is None:
        return None
    end = end if end is not None else start
    span = min(end, start + _MAX_SLICE_LINES - 1)
    try:
        lines = (repo_dir / fp).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    chunk = lines[start - 1 : span]
    if not chunk:
        return None
    body = "\n".join(chunk)
    return f"`{fp}:{start}-{span}`\n```\n{body}\n```"


def build_context_pack(
    graph: dict[str, Any],
    *,
    repo_dir: Path | None = None,
    focus: str | None = None,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Build a compact, token-bounded structural context pack from a code graph.

    Args:
        graph: A contract-shaped graph (``nodes``/``edges``/``layers``/``project``).
        repo_dir: Optional extracted repo dir; enables source slices for focus hits.
        focus: Optional term — when set, the pack adds matching symbols (and slices).
        max_chars: Hard ceiling for the returned string (the token budget).

    Returns:
        A markdown context pack, never longer than ``max_chars``.
    """
    ov = overview_by_layers(graph)
    t = ov.get("totals", {})
    out: list[str] = []

    name = ov.get("project", "?")
    out.append(f"# Repo context: {name}")
    out.append(
        f"{t.get('nodes', 0)} symbols, {t.get('edges', 0)} edges, "
        f"{t.get('contexts', 0)} contexts, encapsulation {t.get('encapsulation_pct', 0)}%."
    )

    contexts = ov.get("contexts", [])[:_MAX_CONTEXTS]
    if contexts:
        out.append("\n## Contexts (largest first)")
        out += [f"- {c['name']}: {c.get('nodes', 0)} symbols" for c in contexts]

    coupling = list(ov.get("cross_context_coupling", {}).items())[:_MAX_COUPLING]
    if coupling:
        out.append("\n## Cross-context coupling")
        out += [f"- {pair} ({n})" for pair, n in coupling]

    hubs = ov.get("top_hubs", [])[:_MAX_HUBS]
    if hubs:
        out.append("\n## Key modules (most depended on)")
        out += [f"- {h.get('file')} (imported by {h.get('imported_by', 0)})" for h in hubs]

    if focus:
        hits = [n for n in graph.get("nodes", []) if _matches(n, focus)][:_MAX_FOCUS_HITS]
        out.append(f"\n## Symbols matching '{focus}'")
        if not hits:
            out.append("- (none found)")
        for n in hits:
            loc = n.get("filePath", "?")
            if n.get("lineStart") is not None:
                loc = f"{loc}:{n['lineStart']}"
            out.append(f"- {n.get('type', '?')} `{n.get('name', '?')}` — {loc}")
        if repo_dir is not None:
            slices = [s for n in hits if (s := _source_slice(repo_dir, n))]
            if slices:
                out.append("\n## Source slices")
                out += slices

    return _truncate("\n".join(out), max_chars)


# --------------------------------------------------------------------------- #
# Graph-store resolution (the `ask_repo` MCP leg)
# --------------------------------------------------------------------------- #
def list_repo_graphs(repos_root: Path) -> list[str]:
    """Dirnames under ``repos_root`` that carry a ``knowledge-graph.json``.

    These are the repos `ask_repo` can answer about — one ``owner__repo`` dir
    per ingested repo (PRD-012 FR-4), written by the ingest's graph stage."""
    if not repos_root.is_dir():
        return []
    return sorted(p.name for p in repos_root.iterdir() if (p / GRAPH_FILENAME).is_file())


def find_repo_graph(repos_root: Path, repo: str) -> Path | None:
    """Resolve a free-form repo query to its ``knowledge-graph.json`` path.

    Accepts ``owner/repo``, ``owner__repo``, or a bare repo name; matching is
    case-insensitive and only ever against the dir names ``list_repo_graphs``
    returned (never joined raw — no path traversal by construction). Returns
    ``None`` when nothing matches or a bare name is ambiguous."""
    q = repo.strip().strip("/").replace("/", "__").lower()
    if not q:
        return None
    names = list_repo_graphs(repos_root)
    exact = [n for n in names if n.lower() == q]
    if exact:
        return repos_root / exact[0] / GRAPH_FILENAME
    bare = [n for n in names if n.lower().split("__", 1)[-1] == q]
    if len(bare) == 1:
        return repos_root / bare[0] / GRAPH_FILENAME
    sub = [n for n in names if q in n.lower()]
    if len(sub) == 1:
        return repos_root / sub[0] / GRAPH_FILENAME
    return None


def ask_repo_pack(
    repos_root: Path,
    repo: str,
    *,
    focus: str | None = None,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> dict[str, Any]:
    """Resolve + load + pack in one call — the full `ask_repo` tool body.

    Returns ``{repo, graph_path, pack}`` on success or ``{error, available}``
    when the repo is unknown/ambiguous or its graph is unreadable, so the MCP
    layer stays a thin JSON shim."""
    graph_path = find_repo_graph(repos_root, repo)
    if graph_path is None:
        return {
            "error": f"no ingested code graph matches {repo!r} (or the name is ambiguous)",
            "available": list_repo_graphs(repos_root),
            "hint": "Pass owner/repo (or the exact owner__repo dirname); ingest the repo first.",
        }
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "error": f"graph unreadable for {graph_path.parent.name}: {type(exc).__name__}",
            "available": list_repo_graphs(repos_root),
        }
    pack = build_context_pack(graph, focus=focus or None, max_chars=max_chars)
    return {"repo": graph_path.parent.name, "graph_path": str(graph_path), "pack": pack}
