"""Deterministic, LLM-free page accretion (ADR-036).

The synthesis agent renders a *fresh* page per source and would otherwise
``write_page`` it over any existing page at the same slug — destroying the
prior body (entity/concept pages never accreted; ``source_count`` was always
1). This module merges the fresh render with the prior page text instead:

- :func:`accrete_mention_page` — entity/concept pages. Unions the ``sources``
  frontmatter, sets ``source_count`` to the union size, **preserves the prior
  canonical body verbatim** (the legacy-rich-body guard), and upserts a
  per-source bullet under a ``## Mentions`` ledger keyed on the source page
  path (re-ingesting the same source replaces only its own bullet — mirrors
  ADR-028's ``source_key`` idempotency at the page layer).
- :func:`accrete_source_page` — source pages. Keeps the fresh body on top and
  appends a one-line ``## History`` entry of the prior version's summary
  (carrying prior entries forward, newest first). ``source_count`` stays 1.

Both are pure functions over strings. On any parse failure they return the
fresh text untouched (``accreted=False``) — accretion can never corrupt; the
worst case is today's overwrite behaviour. There is NO LLM call here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import yaml

from wiki_synthesis.templates import render_frontmatter

__all__ = ["Accretion", "accrete_mention_page", "accrete_source_page"]

_FM_RE = re.compile(r"\A---\s*\n(.*?)\n---[ \t]*\n?(.*)\Z", re.DOTALL)
_MENTIONS_HEADING = re.compile(r"(?m)^##\s+Mentions\s*$")
_HISTORY_HEADING = re.compile(r"(?m)^##\s+History\s*$")
_BULLET_RE = re.compile(r"^- \[\[(?P<src>[^\]]+)\]\](?: \((?P<date>[^)]*)\))?:\s?(?P<text>.*)$")

_MAX_SUMMARY_CHARS = 140
# Lines to skip when picking a one-line summary: headings, image embeds,
# horizontal rules, table rows, blockquotes. NOTE: we do NOT skip lines
# starting with "[[" — synthesized bodies wikilink their first term, so a
# leading "[[Term]]" is ordinary prose, not a bare embed.
_SKIP_PREFIXES = ("#", "![", "---", "|", ">")


@dataclass(frozen=True)
class Accretion:
    """Result of merging a fresh render with a prior page.

    ``text`` is the page to write; ``source_count`` is the effective number of
    distinct source documents (for the index entry); ``accreted`` is True when
    a prior page was actually merged in (False = fresh passed through).
    """

    text: str
    source_count: int
    accreted: bool


# --------------------------------------------------------------------------- #
# Parsing helpers (tolerant; never raise)                                      #
# --------------------------------------------------------------------------- #


def _split_page(text: str) -> tuple[dict[str, Any], str] | None:
    """Split a rendered page into (frontmatter dict, body-after-frontmatter)."""
    match = _FM_RE.match(text)
    if not match:
        return None
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(frontmatter, dict):
        return None
    return frontmatter, match.group(2)


def _split_heading(rest: str) -> tuple[str, str]:
    """Peel the leading ``# Title`` H1 off the body; return (heading, content)."""
    stripped = rest.lstrip("\n")
    if stripped.startswith("# "):
        nl = stripped.find("\n")
        if nl == -1:
            return stripped, ""
        return stripped[:nl], stripped[nl + 1 :].lstrip("\n")
    return "", stripped


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [value]
    return []


def _union(items: list[str]) -> list[str]:
    """Dedup preserving first-seen order; drop empties."""
    seen: dict[str, None] = {}
    for item in items:
        if item and item not in seen:
            seen[item] = None
    return list(seen)


def _one_line(text: str, *, limit: int = _MAX_SUMMARY_CHARS) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > limit:
        return collapsed[:limit].rsplit(" ", 1)[0] + "…"
    return collapsed


def _first_prose_line(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(_SKIP_PREFIXES):
            continue
        return _one_line(stripped)
    return ""


# --------------------------------------------------------------------------- #
# Entity / concept accretion (D1)                                              #
# --------------------------------------------------------------------------- #


def _split_mentions(content: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Separate the canonical body from a trailing ``## Mentions`` ledger."""
    match = _MENTIONS_HEADING.search(content)
    if not match:
        return content.strip(), []
    canonical = content[: match.start()].strip()
    mentions: list[tuple[str, str, str]] = []
    for line in content[match.end() :].splitlines():
        bullet = _BULLET_RE.match(line.strip())
        if bullet:
            mentions.append(
                (bullet.group("src"), bullet.group("date") or "", bullet.group("text").strip())
            )
    return canonical, mentions


def _upsert(
    mentions: list[tuple[str, str, str]], src: str, date: str, text: str
) -> list[tuple[str, str, str]]:
    out = [(s, d, t) for (s, d, t) in mentions if s != src]
    out.append((src, date, text))
    return out


def accrete_mention_page(prior_text: str, fresh_text: str, *, now: datetime) -> Accretion:
    """Merge a fresh entity/concept render into the prior page by union."""
    prior = _split_page(prior_text)
    fresh = _split_page(fresh_text)
    if prior is None or fresh is None:
        count = _frontmatter_count(fresh) if fresh else 1
        return Accretion(text=fresh_text, source_count=count, accreted=False)

    prior_fm, prior_rest = prior
    fresh_fm, fresh_rest = fresh
    _, prior_content = _split_heading(prior_rest)
    _, fresh_content = _split_heading(fresh_rest)

    new_sources = _as_list(fresh_fm.get("sources"))
    new_source = new_sources[0] if new_sources else ""
    date = now.date().isoformat()
    mention = _one_line(fresh_content) or f"Mentioned in [[{new_source}]]."

    canonical, mentions = _split_mentions(prior_content)
    mentions = _upsert(mentions, new_source, date, mention)

    sources = _union([*_as_list(prior_fm.get("sources")), new_source])
    source_count = len(sources)
    name = str(prior_fm.get("title") or fresh_fm.get("title") or "Untitled")

    frontmatter = render_frontmatter(
        title=name,
        date=date,
        sources=sources,
        tags=_as_list(prior_fm.get("tags")) or _as_list(fresh_fm.get("tags")),
        origin=str(prior_fm.get("origin") or fresh_fm.get("origin") or "synthesized-deterministic"),
        category=str(prior_fm.get("category") or fresh_fm.get("category") or "entities"),
        source_count=source_count,
    )

    ledger = "## Mentions\n" + "\n".join(f"- [[{src}]] ({d}): {txt}" for (src, d, txt) in mentions)
    blocks = [frontmatter, f"# {name}"]
    if canonical:
        blocks.append(canonical)
    blocks.append(ledger)
    return Accretion(text="\n\n".join(blocks) + "\n", source_count=source_count, accreted=True)


# --------------------------------------------------------------------------- #
# Source page accretion (D2)                                                   #
# --------------------------------------------------------------------------- #


def _split_history(content: str) -> tuple[str, list[str]]:
    match = _HISTORY_HEADING.search(content)
    if not match:
        return content.strip(), []
    main = content[: match.start()].strip()
    history = [
        line.strip()
        for line in content[match.end() :].splitlines()
        if line.strip().startswith("- ")
    ]
    return main, history


def accrete_source_page(prior_text: str, fresh_text: str, *, now: datetime) -> Accretion:
    """Append a ``## History`` provenance ledger of the prior summary.

    Guard: in the systemd-worker flow (ADR-035) the worker writes a *mechanical*
    page (raw body inside a SEC-05 ``<ingested_source ...>`` envelope) at the
    source slug BEFORE synthesis runs, so the prior *synthesized* source page is
    not visible here — only the mechanical page is. We must not ledger that
    envelope as a prior version, so a mechanical page is treated as "no prior"
    (no-op). Source-page History therefore engages only when synthesis sees a
    real prior synthesized page (direct/re-synthesis flows); entity/concept
    accretion is unaffected (the worker never writes those paths)."""
    if "<ingested_source" in prior_text:
        return Accretion(text=fresh_text, source_count=1, accreted=False)
    prior = _split_page(prior_text)
    if prior is None:
        return Accretion(text=fresh_text, source_count=1, accreted=False)

    prior_fm, prior_rest = prior
    _, prior_content = _split_heading(prior_rest)
    prior_main, prior_history = _split_history(prior_content)
    prior_summary = _first_prose_line(prior_main)
    if not prior_summary:
        return Accretion(text=fresh_text, source_count=1, accreted=False)

    prior_date = str(prior_fm.get("date") or now.date().isoformat())
    history_lines = [f"- {prior_date}: {prior_summary}", *prior_history]
    body = fresh_text.rstrip("\n")
    block = "## History\n" + "\n".join(history_lines)
    return Accretion(text=f"{body}\n\n{block}\n", source_count=1, accreted=True)


def _frontmatter_count(parsed: tuple[dict[str, Any], str] | None) -> int:
    if parsed is None:
        return 1
    raw = parsed[0].get("source_count", 1)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1
