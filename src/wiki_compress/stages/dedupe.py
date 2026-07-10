"""
Paragraph-level dedupe stage.

Walks paragraphs (blank-line separated) and collapses any paragraph that
exactly matches one already seen in the same payload. The replacement is
``[see above]`` — short, unambiguous, and easy for the model to interpret.

This is the **safe** form of dedupe — exact-body only. PRD-007 R-1 warns
that over-aggressive dedupe can drop signal. Single-line dedupe is
deferred to a future stage; v1 ships exact-paragraph dedupe with a small
minimum size to avoid collapsing one-word headings.

Preserved spans (code blocks etc.) are walked over but never deduped:
two identical code blocks may carry different *contextual* meaning and
the audit cost of mis-collapsing them outweighs the token saving.
"""

from __future__ import annotations

import re

from wiki_compress.stages.preserve import iter_unprotected

#: Paragraphs shorter than this are exempt — collapsing "Yes." or a section
#: heading would do more harm than good. The 4 chars/token convention says
#: 40 chars ≈ 10 tokens, a good floor.
MIN_PARAGRAPH_CHARS = 40

#: Replacement body for collapsed paragraphs.
SEE_ABOVE = "[see above]"

_PARAGRAPH_SPLIT = re.compile(r"(\n\s*\n)")  # split, keeping the separators


def _normalised(paragraph: str) -> str:
    """Canonical form for comparison — strip + collapse internal whitespace.

    Two paragraphs that differ only in whitespace count as duplicates;
    a one-character difference (PRD-007 R-1 example) does not.
    """
    return re.sub(r"\s+", " ", paragraph.strip())


def dedupe_paragraphs(text: str) -> str:
    """Collapse exact-duplicate paragraphs, leaving preserved spans alone.

    Idempotent: the replacement body itself is < ``MIN_PARAGRAPH_CHARS``
    so re-running on collapsed output is a no-op.
    """
    out: list[str] = []
    seen: set[str] = set()

    for is_protected, segment in iter_unprotected(text):
        if is_protected:
            out.append(segment)
            continue
        out.append(_dedupe_plain(segment, seen))

    return "".join(out)


def _dedupe_plain(segment: str, seen: set[str]) -> str:
    parts = _PARAGRAPH_SPLIT.split(segment)
    rebuilt: list[str] = []
    for part in parts:
        if not part or part.isspace():
            rebuilt.append(part)
            continue
        canon = _normalised(part)
        if len(canon) < MIN_PARAGRAPH_CHARS:
            rebuilt.append(part)
            continue
        if canon in seen:
            rebuilt.append(SEE_ABOVE)
        else:
            seen.add(canon)
            rebuilt.append(part)
    return "".join(rebuilt)
