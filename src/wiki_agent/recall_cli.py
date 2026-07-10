"""Deterministic recall for non-interactive consumers (ADR-034 D4).

``wiki-agent recall "<query>"`` keyword-scores ``wiki/index.md`` entries,
reads the top page(s), and emits one JSON document carrying everything an
external content pipeline (e.g. auto-publish-social) needs: page metadata
for captions (title/summary/tags/sources) plus the token-budgeted body for
script grounding.

No LLM, no embeddings, no network — safe for cron. Degrades to an empty
result set (exit 0) when the index or page is missing, so callers can fall
back to plain-topic mode.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

#: Crude but stable chars-per-token estimate used across the codebase.
_CHARS_PER_TOKEN = 4

_LINK_RE = re.compile(r"\[(.+?)\]\((.+?)\)")
_WORD_RE = re.compile(r"[a-z0-9]{3,}")


@dataclass(frozen=True, slots=True)
class IndexEntry:
    page_path: str
    title: str
    summary: str
    category: str
    tags: tuple[str, ...]


def parse_index(index_md: str) -> tuple[IndexEntry, ...]:
    """Parse the ``wiki/index.md`` table into entries (same shape the
    ``search_wiki_index`` tool reads; tolerant of 4- and 5-column rows)."""
    entries: list[IndexEntry] = []
    current_section = ""
    for line in index_md.splitlines():
        section = re.match(r"^##\s+(\w+)", line)
        if section:
            current_section = section.group(1).lower()
            continue
        if not line.startswith("|") or line.startswith("| Page") or line.startswith("|--"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 4:
            continue
        if len(cells) >= 5:
            page_cell, category, summary, _sources, _updated = cells[:5]
        else:
            page_cell, summary, _sources, _updated = cells[:4]
            category = current_section or "unknown"
        link = _LINK_RE.match(page_cell)
        if not link:
            continue
        entries.append(
            IndexEntry(
                page_path=link.group(2),
                title=link.group(1),
                summary=summary,
                category=category,
                tags=(),
            )
        )
    return tuple(entries)


def _terms(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def score_entry(query: str, entry: IndexEntry) -> float:
    """Keyword term-overlap in [0, 1] (the deterministic 40% of the hybrid
    search, used alone here so cron runs need no embedding model)."""
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    haystack = _terms(f"{entry.title} {entry.summary} {entry.category}")
    return len(query_terms & haystack) / len(query_terms)


def _split_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                data = yaml.safe_load(parts[1])
            except yaml.YAMLError:
                data = None
            if isinstance(data, dict):
                return data, parts[2].strip()
    return {}, text


def _as_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return []


def build_recall_context(
    wiki_root: str | Path,
    query: str,
    *,
    token_budget: int = 1500,
    limit: int = 1,
) -> dict:
    """The full vault→video payload for one query (ADR-034 D4)."""
    root = Path(wiki_root)
    index_path = root / "wiki" / "index.md"
    out: dict = {"query": query, "results": []}
    if not index_path.is_file():
        return out

    entries = parse_index(index_path.read_text(encoding="utf-8"))
    scored = sorted(
        ((score_entry(query, e), e) for e in entries),
        key=lambda pair: (-pair[0], pair[1].page_path),
    )

    budget_chars = max(token_budget, 1) * _CHARS_PER_TOKEN
    for score, entry in scored[: max(limit, 0)]:
        if score <= 0.0:
            continue
        # Index links are relative to wiki/ (where index.md lives); tolerate
        # legacy rows that already carry the wiki/ prefix by falling back to
        # the vault root.
        page_file = index_path.parent / entry.page_path
        if not page_file.is_file():
            page_file = root / entry.page_path
        if not page_file.is_file():
            print(
                f"recall: entry {entry.page_path!r} scored {score:.2f}"
                " but no file found on disk; skipping",
                file=sys.stderr,
            )
            continue
        front, body = _split_frontmatter(page_file.read_text(encoding="utf-8"))
        truncated = len(body) > budget_chars
        out["results"].append(
            {
                "page_path": entry.page_path,
                "title": str(front.get("title") or entry.title),
                "summary": entry.summary,
                "category": entry.category,
                "tags": _as_list(front.get("tags")),
                "sources": _as_list(front.get("sources")),
                "date": str(front.get("date") or ""),
                "score": round(score, 4),
                "body": body[:budget_chars],
                "truncated": truncated,
            }
        )
    return out


def run_recall_cli(args) -> int:
    """Dispatch target for ``wiki-agent recall`` (mirrors run_fetch_cli style)."""
    context = build_recall_context(
        args.root,
        args.query,
        token_budget=args.token_budget,
        limit=args.limit,
    )
    print(json.dumps(context, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0
