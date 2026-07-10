"""Render source/entity/concept pages per the wiki contract.

The contract (knowledge-base/schema/wiki-schema.md + the Hermes batch
prompt): every page carries YAML frontmatter with the OKF-mandatory
``type`` (ADR-045 §9.2), plus title, date (ISO), sources, tags, origin;
recommended category and source_count. Entity
pages tag ``["entity", ...]``, concept pages tag ``["concept", ...]``.
The source body preserves all image refs and source URLs, and wikilinks
``[[Term]]`` every extracted entity/concept.
"""

from __future__ import annotations

import re

__all__ = [
    "render_concept_page",
    "render_entity_page",
    "render_frontmatter",
    "render_source_page",
    "slugify",
    "wikilink_terms",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Deterministic, filesystem-safe slug (lowercase, hyphenated)."""
    return _SLUG_RE.sub("-", name.lower()).strip("-") or "untitled"


def _yaml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


# OKF v0.1 §9.2 (ADR-045): every page declares a `type`. Derived from the
# category so synthesis pages are born conformant without relying on the
# write_page injection (also covers writers that bypass it, e.g. the seal path).
_CATEGORY_TO_OKF_TYPE = {
    "sources": "Source",
    "entities": "Entity",
    "concepts": "Concept",
    "answers": "Answer",
    "observations": "Observation",
}


def category_to_okf_type(category: str) -> str:
    return _CATEGORY_TO_OKF_TYPE.get(category, "Note")


def render_frontmatter(
    *,
    title: str,
    date: str,
    sources: list[str],
    tags: list[str],
    origin: str,
    category: str,
    source_count: int,
) -> str:
    """The shared YAML frontmatter block (with surrounding ``---``).

    Emits the OKF-mandatory ``type`` first (ADR-045), derived from ``category``.
    """
    src_items = ", ".join(f'"{_yaml_escape(s)}"' for s in sources)
    tag_items = ", ".join(f'"{_yaml_escape(t)}"' for t in tags)
    return "\n".join(
        [
            "---",
            f"type: {category_to_okf_type(category)}",
            f'title: "{_yaml_escape(title)}"',
            f"date: {date}",
            f"sources: [{src_items}]",
            f"tags: [{tag_items}]",
            f"origin: {origin}",
            f"category: {category}",
            f"source_count: {source_count}",
            "---",
        ]
    )


def render_source_page(
    *,
    title: str,
    date: str,
    sources: list[str],
    tags: list[str],
    origin: str,
    body: str,
    source_count: int = 1,
    relevance: str = "",
) -> str:
    fm = render_frontmatter(
        title=title,
        date=date,
        sources=sources,
        tags=tags,
        origin=origin,
        category="sources",
        source_count=source_count,
    )
    # ADR-036 D4: an AI-first `## For future Claude` preamble heads the body
    # when a relevance note is supplied (the agent always supplies one — model
    # text or a deterministic default). Absent => the prior plain shape.
    if relevance.strip():
        return (
            f"{fm}\n\n# {title}\n\n## For future Claude\n\n{relevance.strip()}\n\n{body.strip()}\n"
        )
    return f"{fm}\n\n# {title}\n\n{body.strip()}\n"


def render_entity_page(
    *,
    name: str,
    date: str,
    source_page_path: str,
    mention: str,
    origin: str,
) -> str:
    fm = render_frontmatter(
        title=name,
        date=date,
        sources=[source_page_path],
        tags=["entity"],
        origin=origin,
        category="entities",
        source_count=1,
    )
    body = mention.strip() or f"Mentioned in [[{source_page_path}]]."
    return f"{fm}\n\n# {name}\n\n{body}\n"


def render_concept_page(
    *,
    name: str,
    date: str,
    source_page_path: str,
    mention: str,
    origin: str,
) -> str:
    fm = render_frontmatter(
        title=name,
        date=date,
        sources=[source_page_path],
        tags=["concept"],
        origin=origin,
        category="concepts",
        source_count=1,
    )
    body = mention.strip() or f"Mentioned in [[{source_page_path}]]."
    return f"{fm}\n\n# {name}\n\n{body}\n"


def wikilink_terms(body: str, terms: list[str]) -> str:
    """Wikilink the first plain-prose occurrence of each term.

    Skips occurrences already inside ``[[...]]``, markdown links/images,
    or inline code (cheap guard: the char before must not be ``[``,
    a backtick, or a word char; the char after must not be ``]`` or a
    word char). Longer terms are linked first so "Claude Code" wins
    over "Claude".
    """
    for term in sorted(terms, key=len, reverse=True):
        if not term:
            continue
        pattern = re.compile(rf"(?<![\[`\w]){re.escape(term)}(?![\]`\w])")
        body = pattern.sub(f"[[{term}]]", body, count=1)
    return body
