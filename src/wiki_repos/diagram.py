"""Deterministic Mermaid architecture diagrams from a code knowledge-graph.

No LLM, no network — a pure rendering of the bounded-context panorama produced
by :func:`wiki_qa.codegraph.overview`. Given a code graph dict, we draw a
``flowchart LR`` whose **nodes** are the largest bounded contexts and whose
**edges** are the heaviest cross-context import couplings.

Public API:

- :func:`mermaid_from_graph` — graph dict -> fenced ```mermaid block (or "")
- :func:`diagram_is_renderable` — cheap structural sanity check on the output

Designed to *degrade gracefully*: a ``None``/empty graph yields ``""`` so the
caller can write a wiki page without a diagram rather than failing.
"""

from __future__ import annotations

import re
from typing import Any

from wiki_qa.codegraph import overview

__all__ = ["mermaid_from_graph", "diagram_is_renderable"]

_FENCE_OPEN = "```mermaid"
_FENCE_CLOSE = "```"
_SLUG_RE = re.compile(r"[^0-9a-zA-Z]+")
_EDGE_RE = re.compile(r"^\s*([0-9A-Za-z_]+)\s*-->\s*(?:\|[^|]*\|\s*)?([0-9A-Za-z_]+)\s*$")


def _slug(name: str) -> str:
    """Slug a context name into a Mermaid-safe node id (alnum + underscore).

    Mermaid node ids must not contain spaces, dashes, or punctuation, and must
    not start with a digit. Empty results fall back to ``ctx``.
    """
    s = _SLUG_RE.sub("_", str(name)).strip("_")
    if not s:
        s = "ctx"
    if s[0].isdigit():
        s = f"c_{s}"
    return s


def _safe_label(text: str) -> str:
    """Make a label safe inside a Mermaid double-quoted node label.

    Mermaid breaks on raw double quotes and square brackets inside a quoted
    label; we strip quotes, neutralise ``[]`` to ``()``, and flatten newlines.
    Parentheses are safe inside a double-quoted label and are preserved.
    """
    cleaned = (
        str(text).replace('"', "").replace("[", "(").replace("]", ")").replace("\n", " ").strip()
    )
    return cleaned or "?"


def _unique_slug(name: str, taken: dict[str, str]) -> str:
    """Return a slug for ``name`` that is unique within ``taken`` (slug -> name)."""
    base = _slug(name)
    candidate = base
    i = 2
    while candidate in taken and taken[candidate] != name:
        candidate = f"{base}_{i}"
        i += 1
    return candidate


def mermaid_from_graph(
    graph: dict[str, Any] | None,
    *,
    max_nodes: int = 15,
    max_edges: int = 20,
    overview_data: dict[str, Any] | None = None,
) -> str:
    """Render a deterministic ``flowchart LR`` from a code knowledge-graph.

    Nodes are the top ``max_nodes`` bounded contexts (largest first), and edges
    are the top ``max_edges`` cross-context import couplings, rendered as
    ``A -->|N| B`` where ``N`` is the import count.

    :param graph: A code graph dict (``nodes``/``edges``/``layers``) as produced
        by ``scripts/ua-codegraph.sh``, or ``None``.
    :param max_nodes: Maximum number of context nodes to draw.
    :param max_edges: Maximum number of coupling edges to draw.
    :returns: A fenced ```mermaid block (string), or ``""`` when there is
        nothing meaningful to draw (so the caller can degrade gracefully).

    Determinism: contexts are taken in ``overview``'s largest-first order;
    couplings in ``overview``'s most-common order. Both are stable.
    """
    if not graph or not graph.get("nodes") or not graph.get("edges"):
        return ""

    # ``overview_data`` lets a caller pass a context bucketing that honors the
    # graph's own ``layers`` (correct for external repos). Without it we fall
    # back to ``wiki_qa.codegraph.overview`` (tuned to SBW's own source).
    ov = overview_data if overview_data is not None else overview(graph)
    contexts = ov.get("contexts", [])
    if not contexts:
        return ""

    # Select the top-N contexts (largest first, already sorted by overview).
    selected = contexts[: max(0, max_nodes)]
    if not selected:
        return ""

    # Build slug<->context maps with stable, collision-free slugs.
    slug_by_ctx: dict[str, str] = {}
    label_by_slug: dict[str, str] = {}
    name_by_slug: dict[str, str] = {}
    for ctx in selected:
        name = ctx.get("name", "?")
        slug = _unique_slug(name, name_by_slug)
        name_by_slug[slug] = name
        slug_by_ctx[name] = slug
        node_count = ctx.get("nodes", 0)
        label_by_slug[slug] = _safe_label(f"{name} ({node_count} nodes)")

    selected_names = set(slug_by_ctx)

    # Select the top-N cross-context couplings that link two *selected* contexts.
    # overview returns cross_context_coupling already in most-common order.
    coupling: dict[str, int] = ov.get("cross_context_coupling", {})
    edge_cap = max(0, max_edges)
    edges: list[tuple[str, str, int]] = []
    for pair, count in coupling.items():
        if len(edges) >= edge_cap:
            break
        if " -> " not in pair:
            continue
        src, dst = pair.split(" -> ", 1)
        if src in selected_names and dst in selected_names and src != dst:
            edges.append((src, dst, count))

    # Assemble the Mermaid body.
    lines: list[str] = [_FENCE_OPEN, "flowchart LR"]
    for slug in slug_by_ctx.values():
        lines.append(f'    {slug}["{label_by_slug[slug]}"]')
    for src, dst, count in edges:
        lines.append(f"    {slug_by_ctx[src]} -->|{count}| {slug_by_ctx[dst]}")
    body = "\n".join(lines)
    return f"{body}\n{_FENCE_CLOSE}"


def diagram_is_renderable(mermaid: str) -> bool:
    """Cheap structural sanity check on a Mermaid diagram string.

    Verifies the string is non-empty, opens/closes exactly one fenced
    ```mermaid block, declares a ``flowchart`` directive, and that every edge
    endpoint refers to a declared node id. Used by tests and by callers that
    want to confirm a diagram before embedding it in a page.

    :param mermaid: The candidate Mermaid string (as returned by
        :func:`mermaid_from_graph`).
    :returns: ``True`` if structurally renderable, ``False`` otherwise.
    """
    if not mermaid or not isinstance(mermaid, str):
        return False

    stripped = mermaid.strip()
    if not stripped.startswith(_FENCE_OPEN):
        return False
    if not stripped.endswith(_FENCE_CLOSE):
        return False

    # Exactly one open + one close fence (balanced).
    if stripped.count(_FENCE_CLOSE) != 2:  # opening fence also matches ```
        return False

    inner = stripped[len(_FENCE_OPEN) :]
    inner = inner[: -len(_FENCE_CLOSE)] if inner.endswith(_FENCE_CLOSE) else inner

    body_lines = [ln.strip() for ln in inner.splitlines() if ln.strip()]
    if not body_lines:
        return False
    if not any("flowchart" in ln for ln in body_lines):
        return False

    # Collect declared node ids: lines like `slug["label"]` (or bare `slug`).
    declared: set[str] = set()
    for ln in body_lines:
        if "flowchart" in ln:
            continue
        m = re.match(r"^([0-9A-Za-z_]+)\[", ln)
        if m:
            declared.add(m.group(1))

    # Every edge endpoint must be declared.
    for ln in body_lines:
        if "-->" not in ln:
            continue
        em = _EDGE_RE.match(ln)
        if not em:
            return False
        src, dst = em.group(1), em.group(2)
        if src not in declared or dst not in declared:
            return False

    return True
