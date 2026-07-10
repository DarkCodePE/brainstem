"""Lesson lifecycle: supersession, expiry, decay, contradictions (SPEC-010 FR-4).

Reuses the ADR-028 supersession concept at lesson granularity and ADR-027's
``recency_score`` for decay. The governance rules (ADR-033 D2/D3) are
mechanical, not advisory:

- an INFERRED lesson never supersedes an EXTRACTED one;
- contradictions are *detected and flagged*, never auto-resolved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from wiki_lessons.distill import Lesson
from wiki_lessons.verdict import PROVENANCE_EXTRACTED
from wiki_memory.scoring import recency_score

#: Lessons decay slower than chunks: strategies stay useful for months.
DEFAULT_LESSON_HALFLIFE_DAYS: float = 90.0

#: Retrieval weight multiplier per provenance (ADR-033 D2).
PROVENANCE_FACTOR: dict[str, float] = {"EXTRACTED": 1.0, "INFERRED": 0.7}

#: Minimum term-Jaccard between opposite-kind lessons on the same
#: (repo, domain) before we flag them as a contradiction candidate.
CONTRADICTION_JACCARD: float = 0.3

_WORD_RE = re.compile(r"[a-z0-9]{3,}")


@dataclass(frozen=True, slots=True)
class SupersessionPlan:
    """Outcome of trying to insert ``new`` into an existing lesson set."""

    new: Lesson
    """The incoming lesson, with ``supersedes`` filled in."""

    superseded: tuple[str, ...]
    """lesson_ids the new lesson supersedes."""

    blocked: tuple[str, ...]
    """lesson_ids the new lesson tried but was not allowed to supersede."""

    notes: tuple[str, ...] = ()


def resolve_supersession(
    new: Lesson, existing: tuple[Lesson, ...] | list[Lesson]
) -> SupersessionPlan:
    """Apply the provenance-gated supersession rules (ADR-033 D3).

    A newer EXTRACTED lesson supersedes prior same-``source_key`` lessons of
    any provenance. An INFERRED lesson supersedes only INFERRED ones; its
    attempts against EXTRACTED lessons are blocked and reported.
    """
    already_superseded = {sid for lesson in existing for sid in lesson.supersedes}
    superseded: list[str] = []
    blocked: list[str] = []
    notes: list[str] = []

    for lesson in existing:
        if lesson.source_key != new.source_key or lesson.lesson_id == new.lesson_id:
            continue
        if lesson.lesson_id in already_superseded:
            continue
        if new.provenance == PROVENANCE_EXTRACTED:
            superseded.append(lesson.lesson_id)
        elif lesson.provenance != PROVENANCE_EXTRACTED:
            superseded.append(lesson.lesson_id)
        else:
            blocked.append(lesson.lesson_id)
            notes.append("inferred_cannot_supersede_extracted")

    updated = replace(new, supersedes=tuple(superseded)) if superseded else new
    return SupersessionPlan(
        new=updated,
        superseded=tuple(superseded),
        blocked=tuple(blocked),
        notes=tuple(dict.fromkeys(notes)),
    )


def is_superseded(lesson: Lesson, all_lessons: tuple[Lesson, ...] | list[Lesson]) -> bool:
    return any(lesson.lesson_id in other.supersedes for other in all_lessons)


def is_expired(lesson: Lesson, *, now: datetime | None = None) -> bool:
    if not lesson.expires:
        return False
    now = now or datetime.now(UTC)
    try:
        expires = datetime.fromisoformat(lesson.expires.replace("Z", "+00:00"))
    except ValueError:
        return False
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=UTC)
    return now >= expires


def effective_weight(
    lesson: Lesson,
    *,
    now: datetime | None = None,
    halflife_days: float = DEFAULT_LESSON_HALFLIFE_DAYS,
) -> float:
    """Decayed, provenance-discounted weight in [0, 1] (ADR-033 D2/D3)."""
    recency = recency_score(lesson.created_at, now=now, halflife_days=halflife_days)
    factor = PROVENANCE_FACTOR.get(lesson.provenance, PROVENANCE_FACTOR["INFERRED"])
    return recency * lesson.confidence * factor


def terms_of(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def detect_contradictions(
    lessons: tuple[Lesson, ...] | list[Lesson],
    *,
    threshold: float = CONTRADICTION_JACCARD,
) -> tuple[tuple[str, str], ...]:
    """Flag AMBIGUOUS opposite-kind pairs on the same (repo, domain).

    Detection only — resolution is a human decision (SSGM posture).
    Pairs are returned sorted for determinism.
    """
    flagged: list[tuple[str, str]] = []
    items = sorted(lessons, key=lambda lesson: lesson.lesson_id)
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            if a.kind == b.kind or (a.repo, a.domain) != (b.repo, b.domain):
                continue
            overlap = _jaccard(terms_of(a.strategy), terms_of(b.strategy))
            if overlap >= threshold:
                flagged.append((a.lesson_id, b.lesson_id))
    return tuple(flagged)
