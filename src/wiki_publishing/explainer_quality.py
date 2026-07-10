"""Deterministic quality pre-screen for LinkedIn explainer drafts (ADR-044).

ADR-044's acceptance criteria are probabilistic and **human-rated** (on-archetype
≥0.85, publishable ≥0.85). The one band that is *machine-checkable* is AC-2's
factual consistency — "**0 invented numbers, social-proof counts, or product
claims**". This module pre-screens exactly that and nothing more: every numeric /
stat token the model wrote in the body must be grounded either in the cited
source pages or in the author-supplied CTA trailer values (those are rendered
VERBATIM by [[ADR-044]], so they are facts the author owns, not model fabrication).

It deliberately does **not** try to judge "on-archetype" or "publishable" — those
stay human (the spot-check harness emits a fillable checklist, mirroring
``scripts/ac3_spotcheck_history.py``). This is the cheap, reproducible pre-filter
the harness runs first to catch the one objective failure mode before a human reads.

Pure, deterministic, no LLM, no network. Mirrors the ADR-031 ``QualityScore`` shape.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: A numeric/stat token: a digit run with optional decimal/grouping and a unit
#: suffix the model might fabricate (%, ×/x multipliers, K/M/B/bn scales, "+").
#: The unit must be ATTACHED (no space) — otherwise the case-insensitive class
#: would swallow the first letter of a following word ("1 minute" → "1 m").
_NUMERIC_RE = re.compile(r"\d[\d.,]*(?:%|×|x|bn|B|K|M|\+)?", re.IGNORECASE)

#: Bare integers ≤ this are treated as structural (e.g. "1 minute", "3 steps",
#: numbered list markers) and not held to source-grounding — they are never the
#: invented *statistics* AC-2 targets.
_STRUCTURAL_MAX = 10

_WEIGHTS = {"numbers_grounded": 0.7, "structure": 0.3}


@dataclass(frozen=True, slots=True)
class ExplainerQualityScore:
    """Machine pre-screen of an explainer draft (ADR-044 AC-2 + a structure hint).

    ``verdict`` is ``"grounded"`` only when *every* statistic in the body is found
    in the sources or the CTA trailer; any ungrounded number → ``"ungrounded"``
    (the human reviewer should check those before publishing)."""

    score: float
    components: dict[str, float]
    verdict: str  # "grounded" | "ungrounded"
    ungrounded_numbers: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)


def _digits(token: str) -> str:
    """The bare digit sequence of a token (``"4,000+"`` → ``"4000"``)."""
    return re.sub(r"\D", "", token)


def _structural(token: str) -> bool:
    """A small bare integer with no stat unit — structural, not a claimed statistic."""
    raw = token.strip()
    if re.search(r"[%×xBKM]|\+|bn", raw, re.IGNORECASE):
        return False
    d = _digits(raw)
    return d.isdigit() and int(d) <= _STRUCTURAL_MAX


def _structure_signal(body: str) -> float:
    """Light explainer-skeleton signal: bullets, multiple blocks, a hook question."""
    has_bullets = bool(re.search(r"(?m)^\s*(?:[-•]|➡️|\d+\.)\s", body))
    has_blocks = body.count("\n\n") >= 2
    has_hook = "?" in body[: max(1, len(body) // 3)]
    return round(sum((has_bullets, has_blocks, has_hook)) / 3, 3)


def score_explainer(
    body: str,
    sources: Sequence[Any],
    *,
    exempt_values: Sequence[str] = (),
) -> ExplainerQualityScore:
    """Pre-screen an explainer draft body for AC-2 factual grounding + structure.

    Args:
        body: The generated post body (may include the CTA trailers).
        sources: The cited ``WikiSnippet``-like objects (need a ``.body`` and,
            optionally, ``.title``) the draft was grounded on.
        exempt_values: Author-supplied verbatim strings (the ``newsletter_cta`` /
            ``product_ps`` field values) — numbers inside these are facts the
            author owns, never model fabrication, so they are exempt from grounding.

    Returns:
        An :class:`ExplainerQualityScore`. ``verdict="ungrounded"`` lists every
        body statistic not found in the sources/CTA — the actionable AC-2 signal.
    """
    haystack = " ".join(
        [*(getattr(s, "body", "") or "" for s in sources)]
        + [*(getattr(s, "title", "") or "" for s in sources)]
        + [str(v) for v in exempt_values]
    ).lower()
    haystack_digits = re.sub(r"\D", " ", haystack)

    candidates = [t.strip() for t in _NUMERIC_RE.findall(body) if t.strip()]
    checked = [t for t in candidates if not _structural(t)]
    ungrounded: list[str] = []
    for tok in checked:
        d = _digits(tok)
        if not d:
            continue
        # grounded if the literal token text OR its bare digit run appears in
        # the sources/CTA (digit-run match tolerates "4,000+" vs "4000").
        if tok.lower() in haystack or f" {d} " in f" {haystack_digits} ":
            continue
        ungrounded.append(tok)

    n = len(checked)
    numbers_grounded = 1.0 if n == 0 else round((n - len(ungrounded)) / n, 3)
    structure = _structure_signal(body)
    score = round(
        numbers_grounded * _WEIGHTS["numbers_grounded"] + structure * _WEIGHTS["structure"], 3
    )
    verdict = "grounded" if not ungrounded else "ungrounded"

    notes: list[str] = []
    if n == 0:
        notes.append("no statistics in body — nothing to ground (AC-2 vacuously holds)")
    if ungrounded:
        notes.append(
            f"{len(ungrounded)} ungrounded number(s): {', '.join(dict.fromkeys(ungrounded))}"
        )
    if structure < 0.34:
        notes.append("weak explainer structure signal (few bullets/blocks/hook)")

    return ExplainerQualityScore(
        score=score,
        components={"numbers_grounded": numbers_grounded, "structure": structure},
        verdict=verdict,
        ungrounded_numbers=tuple(dict.fromkeys(ungrounded)),
        notes=tuple(notes),
    )
