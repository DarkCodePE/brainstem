"""Deterministic paper distillation (ADR-048 Fase 3 / D5).

The single biggest quality offender in the vault scan was papers: 7/10 pages
were the **full PDF→markdown extraction shoved in as the page body** (raw_dump,
up to 336 KB of OCR-mangled text). The LLM paper archetype (PRD-015 FR-6)
produces a real body on the happy path — but the synthesis *degrade* path
(`_compose_source_body`) deliberately never summarises, so any router failure
left the raw dump as the page.

This module gives the degrade path a paper-shaped alternative: extract the
sections a paper page's body contract requires (Abstract, Contributions,
Results — ADR-048 D1) from the extraction markdown itself. Grounded by
construction (only the paper's own text, sliced by its own headings), no LLM,
no network, ``$0``. The full extraction stays in the ``raw/`` sidecar — the
distilled sections become the body, per D5: "keep the raw extraction behind a
fold or in the raw/ sidecar — not as the page itself".

Returns ``None`` when the extraction has no recognisable paper shape (no
abstract-like section and no usable opening prose), so the caller can fall
back to the existing behaviour — degrade-first, never worse than before.
"""

from __future__ import annotations

import re

__all__ = ["distill_paper"]

#: Cap per distilled section (chars) — a section is a slice, not the paper.
_MAX_SECTION_CHARS = 4_000
#: Minimum prose (chars) for a distilled body to be worth emitting.
_MIN_DISTILLED_CHARS = 200

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$", re.MULTILINE)

#: Heading synonyms per target section, tried in order (first match wins).
#: Numbered headings ("1 Introduction", "5. Results") are normalised first.
_SECTION_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Abstract", ("abstract",)),
    ("Contributions", ("contributions", "contribution", "introduction")),
    (
        "Results",
        (
            "results",
            "experiments",
            "experimental results",
            "evaluation",
            "conclusion",
            "conclusions",
        ),
    ),
)

_NUMBER_PREFIX_RE = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*\.?|[ivxlc]+\.)\s+", re.IGNORECASE)


def _normalise_heading(text: str) -> str:
    """Lowercased heading with numbering / trailing punctuation stripped."""
    t = _NUMBER_PREFIX_RE.sub("", text.strip())
    return re.sub(r"[:.\s]+$", "", t).lower()


def _split_sections(md: str) -> list[tuple[str, str]]:
    """``[(normalised_heading, section_text), ...]`` in document order.

    The text before the first heading is returned under the pseudo-heading
    ``""`` so a heading-less abstract (common in PDF extractions) is usable.
    """
    matches = list(_HEADING_RE.finditer(md))
    sections: list[tuple[str, str]] = []
    preamble = md[: matches[0].start()] if matches else md
    if preamble.strip():
        sections.append(("", preamble.strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        sections.append((_normalise_heading(m.group(2)), md[m.end() : end].strip()))
    return sections


def _clip(text: str, limit: int = _MAX_SECTION_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Break on a paragraph edge when one exists in the tail third.
    edge = cut.rfind("\n\n")
    if edge > limit // 3:
        cut = cut[:edge]
    return cut.rstrip() + "\n\n[…]"


def _first_prose_block(sections: list[tuple[str, str]]) -> str:
    """Fallback abstract: the first substantial prose paragraph in the doc."""
    for _, text in sections:
        for para in text.split("\n\n"):
            p = para.strip()
            if len(p) >= _MIN_DISTILLED_CHARS and not p.startswith(("|", "![", "```", ">")):
                return p
    return ""


def distill_paper(raw_md: str, *, rel_path: str = "") -> str | None:
    """Distil a paper extraction into an Abstract/Contributions/Results body.

    Purely structural: sections are located by the extraction's own headings
    (with numbering stripped and per-section synonyms), clipped to
    ``_MAX_SECTION_CHARS`` each. Returns ``None`` when no abstract-like text
    can be found — the caller keeps its existing degrade behaviour.
    """
    sections = _split_sections(raw_md)
    if not sections:
        return None

    by_heading: dict[str, str] = {}
    for heading, text in sections:
        by_heading.setdefault(heading, text)

    picked: list[tuple[str, str]] = []
    used: set[str] = set()
    for title, synonyms in _SECTION_SYNONYMS:
        for syn in synonyms:
            text = by_heading.get(syn, "")
            if text and syn not in used:
                picked.append((title, _clip(text)))
                used.add(syn)
                break

    # A paper body without an Abstract is not a distillation — fall back to
    # the first substantial prose block; if even that is absent, give up.
    if not any(title == "Abstract" for title, _ in picked):
        fallback = _first_prose_block(sections)
        if not fallback:
            return None
        picked.insert(0, ("Abstract", _clip(fallback)))

    body = "\n\n".join(f"## {title}\n\n{text}" for title, text in picked)
    if len(body) < _MIN_DISTILLED_CHARS:
        return None

    sidecar = f" (`{rel_path}`)" if rel_path else ""
    note = (
        f"> Distilled deterministically from the paper's own extraction{sidecar}; "
        "the full text stays in the raw sidecar, not in this page (ADR-048 D5)."
    )
    return f"{body}\n\n{note}"
