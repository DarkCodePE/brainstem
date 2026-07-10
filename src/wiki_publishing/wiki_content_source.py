"""
``WikiContentSource`` — read-only ``ContentSource`` over ``wiki/index.md``.

Selects synthesised wiki pages relevant to a topic for the LinkedIn draft
generator (ADR-021 Phase 1). Strictly read-only: it parses the wiki index,
keyword-scores entries, and reads matched page bodies. It never writes, and
it sanitises page paths to stay inside the wiki root (no directory traversal).

The scoring is keyword term-overlap only — a deliberately simple, dependency-
free first cut. Upgrading to the hybrid keyword+embedding scoring already used
by ``wiki_agent.tools.search_wiki_index`` is a follow-up; the ``ContentSource``
protocol keeps that swap a one-line wiring change.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from wiki_publishing.linkedin_draft import WikiSnippet

# Markdown link in an index table cell: [Title](path/to/page.md)
_LINK_RE = re.compile(r"\[(.+?)\]\((.+?)\)")


def _category_of(page_path: str) -> str:
    """The wiki category segment of a page path (e.g. ``sources/x.md`` -> ``sources``).

    Tolerates an optional leading ``wiki/`` so ``wiki/sources/x.md`` -> ``sources``."""
    parts = [p for p in page_path.strip("/").split("/") if p]
    if parts and parts[0] == "wiki":
        parts = parts[1:]
    return parts[0].lower() if len(parts) > 1 else ""


class WikiContentSource:
    """Read-only content selection over the wiki index.

    Parameters
    ----------
    wiki_root:
        The vault root (the directory that contains ``wiki/index.md``),
        i.e. ``$WIKI_ROOT`` / ``knowledge-base/``.
    """

    def __init__(self, *, wiki_root: Path) -> None:
        self._wiki_root = Path(wiki_root).resolve()
        self._index_path = self._wiki_root / "wiki" / "index.md"

    def search(
        self, query: str, *, limit: int = 3, categories: Sequence[str] | None = None
    ) -> list[WikiSnippet]:
        if limit <= 0 or not self._index_path.is_file():
            return []

        entries = self._parse_index()
        if categories:
            wanted = {c.strip("/").lower() for c in categories}
            entries = [e for e in entries if _category_of(e["page_path"]) in wanted]
        if not entries:
            return []

        terms = [t.lower() for t in re.split(r"\W+", query) if len(t) > 2]
        scored: list[tuple[int, dict[str, str]]] = []
        for entry in entries:
            haystack = f"{entry['title']} {entry['summary']}".lower()
            score = sum(haystack.count(t) for t in terms)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:limit]

        snippets: list[WikiSnippet] = []
        for _score, entry in top:
            body = self._read_page_body(entry["page_path"]) or entry["summary"]
            snippets.append(
                WikiSnippet(
                    title=entry["title"],
                    page_path=entry["page_path"],
                    body=body,
                )
            )
        return snippets

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _parse_index(self) -> list[dict[str, str]]:
        """Parse ``wiki/index.md`` markdown tables into entries.

        Mirrors the column shapes accepted by ``wiki_agent.tools`` so the two
        readers agree on the index format."""
        content = self._index_path.read_text(encoding="utf-8")
        entries: list[dict[str, str]] = []
        for line in content.splitlines():
            if not line.startswith("|") or line.startswith(("| Page", "|--")):
                continue
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 4:
                continue
            page_cell = cells[0]
            summary = cells[2] if len(cells) >= 5 else cells[1]
            link = _LINK_RE.match(page_cell)
            if not link:
                continue
            entries.append(
                {
                    "title": link.group(1),
                    "page_path": link.group(2),
                    "summary": summary,
                }
            )
        return entries

    def _read_page_body(self, page_path: str) -> str | None:
        """Read a wiki page body if the path resolves safely inside the vault.

        Returns ``None`` when the file is missing or the path escapes the wiki
        root (directory-traversal guard per CLAUDE.md security rules)."""
        for candidate in (self._wiki_root / page_path, self._wiki_root / "wiki" / page_path):
            try:
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                continue
            if self._wiki_root not in resolved.parents and resolved != self._wiki_root:
                continue
            if resolved.is_file():
                return resolved.read_text(encoding="utf-8")
        return None
