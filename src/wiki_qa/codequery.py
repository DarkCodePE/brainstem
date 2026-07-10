"""Call-graph impact analysis over the UA code graph's ``calls`` edges (ADR-037).

The deterministic code graph produced by ``scripts/ua-codegraph.sh`` already
carries function→function ``calls`` edges, but nothing consumed them — the
existing :mod:`wiki_qa.codegraph` queries only read ``imports``/``contains``.
This module answers the blast-radius question — *"who is affected if I change
this function?"* — with bounded, cycle-guarded reachability over those edges.

Pure functions over an already-loaded graph dict (same shape as
:func:`wiki_qa.codegraph.load_code_graph`). No Cypher engine and no new graph
builder: a query language over a fixed-shape graph would violate the
typed-interface / file-size rules for no gain, and dead-code detection is
explicitly deferred (ADR-037) because zero-caller is a weak signal on a
call-graph blind to dynamic dispatch and framework hooks.

- :func:`resolve_symbol` — map a human symbol to matching node id(s)
- :func:`callers_of` / :func:`callees_of` — direct neighbours over ``calls``
- :func:`reaches` — does ``src`` transitively call ``dst`` within ``max_hops``?
- :func:`impact` — the blast radius: transitive callers of a symbol
"""

from __future__ import annotations

import collections
from typing import Any

CALLS_EDGE = "calls"

DEFAULT_MAX_HOPS = 4
DEFAULT_MAX_RESULTS = 200


# --------------------------------------------------------------------------- #
# Symbol resolution                                                            #
# --------------------------------------------------------------------------- #


def resolve_symbol(graph: dict[str, Any], symbol: str) -> list[str]:
    """Resolve a human-supplied symbol to matching node id(s), sorted.

    Accepts a bare name (``create_wiki_agent``), a ``path/to/file.py:name``
    pair, or a full ``function:...`` node id. A bare name may be ambiguous
    (the same function name in several files) — all matches are returned.
    """
    function_id = f"function:{symbol}"
    suffix = f":{symbol}"
    out: set[str] = set()
    for node in graph.get("nodes", []):
        nid = node.get("id", "")
        if (
            nid == symbol
            or nid == function_id
            or node.get("name") == symbol
            or nid.endswith(suffix)
        ):
            out.add(nid)
    return sorted(out)


# --------------------------------------------------------------------------- #
# Adjacency (over ``calls`` edges only)                                        #
# --------------------------------------------------------------------------- #


def _calls_edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in graph.get("edges", []) if e.get("type") == CALLS_EDGE]


def _forward_adjacency(graph: dict[str, Any]) -> dict[str, set[str]]:
    """``source -> {targets}`` — what each function calls."""
    adj: dict[str, set[str]] = collections.defaultdict(set)
    for e in _calls_edges(graph):
        adj[e["source"]].add(e["target"])
    return adj


def _reverse_adjacency(graph: dict[str, Any]) -> dict[str, set[str]]:
    """``target -> {sources}`` — who calls each function (callers)."""
    adj: dict[str, set[str]] = collections.defaultdict(set)
    for e in _calls_edges(graph):
        adj[e["target"]].add(e["source"])
    return adj


def callers_of(graph: dict[str, Any], symbol: str) -> list[str]:
    """Direct callers of ``symbol`` (node ids, sorted)."""
    ids = resolve_symbol(graph, symbol)
    if not ids:
        return []
    adj = _reverse_adjacency(graph)
    out: set[str] = set()
    for nid in ids:
        out |= adj.get(nid, set())
    return sorted(out)


def callees_of(graph: dict[str, Any], symbol: str) -> list[str]:
    """Direct callees of ``symbol`` (node ids, sorted)."""
    ids = resolve_symbol(graph, symbol)
    if not ids:
        return []
    adj = _forward_adjacency(graph)
    out: set[str] = set()
    for nid in ids:
        out |= adj.get(nid, set())
    return sorted(out)


# --------------------------------------------------------------------------- #
# Reachability + impact                                                        #
# --------------------------------------------------------------------------- #


def reaches(graph: dict[str, Any], src: str, dst: str, *, max_hops: int = DEFAULT_MAX_HOPS) -> bool:
    """Does ``src`` transitively call ``dst`` within ``max_hops``?

    Bounded breadth-first search over ``calls`` edges with a visited-set
    cycle-guard, so cyclic call graphs terminate. Either endpoint resolving
    to several nodes is treated as "any src reaches any dst".
    """
    src_ids = resolve_symbol(graph, src)
    dst_ids = set(resolve_symbol(graph, dst))
    if not src_ids or not dst_ids:
        return False
    visited = set(src_ids)
    if visited & dst_ids:
        return True
    adj = _forward_adjacency(graph)
    frontier = list(src_ids)
    hops = 0
    while frontier and hops < max_hops:
        hops += 1
        nxt: list[str] = []
        for nid in frontier:
            for target in adj.get(nid, set()):
                if target in dst_ids:
                    return True
                if target not in visited:
                    visited.add(target)
                    nxt.append(target)
        frontier = nxt
    return False


def _node_info(graph: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        n.get("id", ""): (n.get("name", ""), n.get("filePath", "")) for n in graph.get("nodes", [])
    }


def _entry(
    info: dict[str, tuple[str, str]], nid: str, *, hops: int | None = None
) -> dict[str, Any]:
    name, file = info.get(nid, (nid.rsplit(":", 1)[-1], ""))
    entry: dict[str, Any] = {"id": nid, "name": name, "file": file}
    if hops is not None:
        entry["hops"] = hops
    return entry


def _candidates(graph: dict[str, Any], symbol: str, *, limit: int = 10) -> list[str]:
    """Best-effort near-matches for an unresolved symbol (substring on name)."""
    needle = symbol.lower()
    out = {
        n.get("id", "")
        for n in graph.get("nodes", [])
        if n.get("type") == "function" and needle and needle in n.get("name", "").lower()
    }
    return sorted(out)[:limit]


def impact(
    graph: dict[str, Any],
    symbol: str,
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    """Blast radius of ``symbol``: its transitive callers (who is affected if
    you change it), plus its direct callers and callees for context.

    Returns a JSON-ready dict. An unresolved symbol returns an ``error`` plus
    near-match ``candidates`` (and empty result lists, so callers can treat the
    shape uniformly). ``transitive_callers`` is capped at ``max_results`` and
    ``truncated`` flags when the full blast radius was larger.
    """
    info = _node_info(graph)
    resolved = resolve_symbol(graph, symbol)
    if not resolved:
        return {
            "symbol": symbol,
            "resolved": [],
            "error": f"Symbol '{symbol}' not found in code graph",
            "candidates": _candidates(graph, symbol),
            "direct_callers": [],
            "direct_callees": [],
            "transitive_callers": [],
            "blast_radius_count": 0,
            "max_hops": max_hops,
            "truncated": False,
        }

    reverse = _reverse_adjacency(graph)
    forward = _forward_adjacency(graph)
    direct_callers = sorted({c for r in resolved for c in reverse.get(r, set())})
    direct_callees = sorted({c for r in resolved for c in forward.get(r, set())})

    # Reverse BFS = transitive callers (the blast radius). Cycle-guarded by the
    # visited set; the resolved symbol(s) are pre-seeded so a cycle back into
    # the symbol itself is excluded from its own blast radius.
    found: dict[str, int] = {}
    visited = set(resolved)
    frontier = list(resolved)
    hop = 0
    while frontier and hop < max_hops:
        hop += 1
        nxt: list[str] = []
        for nid in frontier:
            for caller in reverse.get(nid, set()):
                if caller not in visited:
                    visited.add(caller)
                    found[caller] = hop
                    nxt.append(caller)
        frontier = nxt

    blast = len(found)
    ordered = sorted(found.items(), key=lambda kv: (kv[1], kv[0]))
    transitive = [_entry(info, nid, hops=h) for nid, h in ordered[:max_results]]
    return {
        "symbol": symbol,
        "resolved": resolved,
        "direct_callers": [_entry(info, nid) for nid in direct_callers],
        "direct_callees": [_entry(info, nid) for nid in direct_callees],
        "transitive_callers": transitive,
        "blast_radius_count": blast,
        "max_hops": max_hops,
        "truncated": blast > max_results,
    }


__all__ = [
    "CALLS_EDGE",
    "callees_of",
    "callers_of",
    "impact",
    "reaches",
    "resolve_symbol",
]
