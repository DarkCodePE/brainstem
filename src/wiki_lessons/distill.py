"""Trajectory -> Lesson distillation (SPEC-010 FR-2).

Follows the trajectory-informed memory-generation pattern (arXiv:2603.10600):
successes distill into ``strategy`` lessons, failures into ``pitfall``
lessons — both carry signal, so neither is discarded.

The LLM distiller is an injectable callable (ADR-031 judge pattern). Without
one — or when it fails — a deterministic template still produces a valid,
useful lesson, marked ``distiller_fallback`` (ADR-033 D5).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from wiki_lessons.verdict import PROVENANCE_INFERRED, Verdict

#: INFERRED lessons can never claim more confidence than this (SSGM-style
#: stability cap, ADR-033 D2): an unverified judgment must not outrank a
#: verifier-graded one on confidence alone.
INFERRED_CONFIDENCE_CAP: float = 0.6

KIND_STRATEGY = "strategy"
KIND_PITFALL = "pitfall"


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Summarized agent run over one verifiable task."""

    task_id: str
    """Repo2RLEnv task name, e.g. ``"DarkCodePE/fix-ingest-race"``."""

    repo: str
    """``owner/name`` of the repo the task was mined from."""

    domain: str
    """Coarse area, e.g. ``"bugfix"``, ``"ingest"``, ``"publishing"``."""

    instruction: str
    """Problem statement (caller excerpts; keep it short)."""

    actions: tuple[str, ...] = ()
    """Summarized agent steps, in order."""

    reference: str = ""
    """PR / commit / task URL for the trace."""


@dataclass(frozen=True, slots=True)
class DistillContext:
    """Everything an LLM distiller may look at."""

    trajectory: Trajectory
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class DistilledText:
    """What a distiller returns; any empty field falls back to the template."""

    title: str = ""
    strategy: str = ""
    key_learnings: tuple[str, ...] = ()


Distiller = Callable[[DistillContext], DistilledText]


@dataclass(frozen=True, slots=True)
class Lesson:
    """A distilled, lifecycle-managed unit of verified experience."""

    lesson_id: str
    source_key: str
    """sha256 of ``"{task_id}|{kind}"`` — stable identity for supersession."""

    title: str
    kind: str
    """``strategy`` (verified success) or ``pitfall`` (verified failure)."""

    strategy: str
    key_learnings: tuple[str, ...]
    domain: str
    repo: str
    provenance: str
    confidence: float
    verdict: Verdict
    derived_from: str
    created_at: str
    supersedes: tuple[str, ...] = ()
    expires: str | None = None
    notes: tuple[str, ...] = field(default=())


def source_key_for(task_id: str, kind: str) -> str:
    return hashlib.sha256(f"{task_id}|{kind}".encode()).hexdigest()


def _slugify(text: str, *, max_len: int = 48) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")[:max_len] or "lesson"


def _fallback_text(trajectory: Trajectory, verdict: Verdict, kind: str) -> DistilledText:
    """Deterministic template — always available, never empty (AC-3)."""
    task = trajectory.task_id
    first = trajectory.actions[0] if trajectory.actions else "no recorded action"
    last = trajectory.actions[-1] if trajectory.actions else first
    reward = f"{verdict.reward:.2f}"
    if kind == KIND_STRATEGY:
        title = f"Strategy: {trajectory.instruction[:60].strip() or task}"
        strategy = (
            f"On task {task} ({trajectory.instruction[:160].strip()}), the approach "
            f"'{first}' followed by '{last}' resolved the task with verified "
            f"reward {reward} ({verdict.kind})."
        )
    else:
        title = f"Pitfall: {trajectory.instruction[:60].strip() or task}"
        strategy = (
            f"On task {task} ({trajectory.instruction[:160].strip()}), starting with "
            f"'{first}' failed (reward {reward}, {verdict.kind}). Avoid this opening "
            f"or validate its preconditions before committing to it."
        )
    learnings = tuple(f"{name} = {value:.2f}" for name, value in verdict.components[:4])
    return DistilledText(title=title, strategy=strategy, key_learnings=learnings)


def distill_lesson(
    trajectory: Trajectory,
    verdict: Verdict,
    *,
    distiller: Distiller | None = None,
    now: datetime | None = None,
) -> Lesson:
    """Distill one (trajectory, verdict) pair into a :class:`Lesson`.

    The distiller seam may raise or return empty fields; both degrade to the
    deterministic template with a note in ``Lesson.notes``.
    """
    kind = KIND_STRATEGY if verdict.success else KIND_PITFALL
    notes: list[str] = []

    distilled: DistilledText | None = None
    if distiller is not None:
        try:
            distilled = distiller(DistillContext(trajectory=trajectory, verdict=verdict))
        except Exception:  # noqa: BLE001 — seam must never break the pipeline (D5)
            notes.append("distiller_error")
            distilled = None

    fallback = _fallback_text(trajectory, verdict, kind)
    if distilled is None or not distilled.strategy.strip():
        distilled = fallback
        if "distiller_error" not in notes:
            notes.append("distiller_fallback")

    title = distilled.title.strip() or fallback.title
    strategy = distilled.strategy.strip()
    key_learnings = distilled.key_learnings or fallback.key_learnings

    confidence = verdict.reward
    if verdict.provenance == PROVENANCE_INFERRED:
        confidence = min(confidence, INFERRED_CONFIDENCE_CAP)
    # A pitfall is evidence of failure: its confidence reflects how decisive
    # the failure signal is, not how high the reward was.
    if kind == KIND_PITFALL:
        confidence = max(confidence, 1.0 - verdict.reward)
        if verdict.provenance == PROVENANCE_INFERRED:
            confidence = min(confidence, INFERRED_CONFIDENCE_CAP)

    created = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_key = source_key_for(trajectory.task_id, kind)
    lesson_id = f"{kind}-{source_key[:8]}-{_slugify(trajectory.task_id)}"

    return Lesson(
        lesson_id=lesson_id,
        source_key=source_key,
        title=title,
        kind=kind,
        strategy=strategy,
        key_learnings=tuple(key_learnings),
        domain=trajectory.domain,
        repo=trajectory.repo,
        provenance=verdict.provenance,
        confidence=round(min(max(confidence, 0.0), 1.0), 4),
        verdict=verdict,
        derived_from=trajectory.reference or trajectory.task_id,
        created_at=created,
        notes=tuple(notes),
    )
