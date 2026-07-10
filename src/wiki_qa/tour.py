"""Render the UA graph's guided tour into a Markdown onboarding doc (issue UA-4).

The assembled ``knowledge-graph.json`` carries a ``tour`` array — an ordered
list of ``{order, title, description, nodeIds}`` steps derived from the
``index.md`` category ordering. This module turns that machine artifact into a
human-browsable onboarding walkthrough of the second-brain wiki, resolving node
ids to page names and relative paths.
"""

from __future__ import annotations

from typing import Any


def _node_lookup(graph_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(n["id"]): n for n in graph_data.get("nodes", [])}


def render_tour(graph_data: dict[str, Any]) -> str:
    """Render the graph's ``tour`` into a Markdown onboarding document."""
    nodes = _node_lookup(graph_data)
    project = graph_data.get("project", {})
    name = project.get("name", "Wiki")
    tour = graph_data.get("tour", [])

    lines: list[str] = []
    lines.append(f"# Guided Tour — {name}")
    lines.append("")
    lines.append(
        "_Auto-generated from the Understand-Anything knowledge graph "
        "(`tour` ordering follows `index.md` categories). Issue UA-4._"
    )
    lines.append("")

    if not tour:
        lines.append("> No tour steps were generated (the graph has no `tour` array).")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"**{len(tour)} stops** — follow them in order to learn the wiki top-down.")
    lines.append("")

    for step in tour:
        order = step.get("order", "?")
        title = step.get("title", "Untitled")
        description = step.get("description", "")
        lines.append(f"## {order}. {title}")
        lines.append("")
        if description:
            lines.append(description)
            lines.append("")
        node_ids = step.get("nodeIds", [])
        if node_ids:
            lines.append("Start here:")
            for nid in node_ids:
                node = nodes.get(str(nid))
                if node is None:
                    lines.append(f"- `{nid}` _(missing node)_")
                    continue
                label = node.get("name", nid)
                file_path = node.get("filePath")
                if file_path:
                    lines.append(f"- **{label}** — `{file_path}`")
                else:
                    lines.append(f"- **{label}**")
            lines.append("")

    return "\n".join(lines)
