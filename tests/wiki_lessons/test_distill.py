"""Tests for trajectory distillation (SPEC-010 FR-2, wiki_lessons/distill.py).

Hermetic: the LLM distiller is an injectable callable (ADR-031 pattern); no
test touches a router or the network.
"""

from __future__ import annotations

from datetime import UTC, datetime

from wiki_lessons.distill import (
    INFERRED_CONFIDENCE_CAP,
    DistillContext,
    DistilledText,
    Trajectory,
    distill_lesson,
)
from wiki_lessons.verdict import Verdict, llm_verdict

NOW = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)


def _trajectory(**overrides: object) -> Trajectory:
    base: dict = {
        "task_id": "DarkCodePE/fix-ingest-race",
        "repo": "DarkCodePE/second-brain-wiki",
        "domain": "ingest",
        "instruction": "Fix the race between the watch daemon and the batch ingest cron",
        "actions": ("read daemon.py", "add lockfile guard", "run pytest tests/wiki_ingest"),
        "reference": "https://github.com/DarkCodePE/second-brain-wiki/pull/166",
    }
    base.update(overrides)
    return Trajectory(**base)


def _verifier_verdict(reward: float, *, success: bool) -> Verdict:
    return Verdict(
        source="verifier",
        reward=reward,
        success=success,
        kind="test_execution",
        components=(("f2p_rate", reward), ("p2p_rate", 1.0)),
    )


def test_success_distills_strategy_and_failure_distills_pitfall() -> None:
    success = distill_lesson(_trajectory(), _verifier_verdict(0.9, success=True), now=NOW)
    failure = distill_lesson(_trajectory(), _verifier_verdict(0.1, success=False), now=NOW)
    assert success.kind == "strategy"
    assert failure.kind == "pitfall"
    assert success.source_key != failure.source_key  # kind is part of identity


def test_fallback_template_references_task_and_action() -> None:
    lesson = distill_lesson(_trajectory(), _verifier_verdict(0.9, success=True), now=NOW)
    assert "distiller_fallback" in lesson.notes
    assert "fix-ingest-race" in lesson.strategy
    assert "read daemon.py" in lesson.strategy
    assert lesson.key_learnings  # from verdict components


def test_injected_distiller_text_is_used() -> None:
    def distiller(ctx: DistillContext) -> DistilledText:
        return DistilledText(
            title="Lock before you watch",
            strategy="Acquire the ingest lockfile before starting any watcher.",
            key_learnings=("inotify and cron must share one lock",),
        )

    lesson = distill_lesson(
        _trajectory(), _verifier_verdict(0.9, success=True), distiller=distiller, now=NOW
    )
    assert lesson.title == "Lock before you watch"
    assert lesson.strategy.startswith("Acquire the ingest lockfile")
    assert lesson.notes == ()


def test_distiller_exception_degrades_to_fallback_with_note() -> None:
    def boom(ctx: DistillContext) -> DistilledText:
        raise RuntimeError("router down")

    lesson = distill_lesson(
        _trajectory(), _verifier_verdict(0.9, success=True), distiller=boom, now=NOW
    )
    assert "distiller_error" in lesson.notes
    assert "fix-ingest-race" in lesson.strategy  # fallback text


def test_empty_distiller_output_falls_back() -> None:
    def empty(ctx: DistillContext) -> DistilledText:
        return DistilledText()

    lesson = distill_lesson(
        _trajectory(), _verifier_verdict(0.9, success=True), distiller=empty, now=NOW
    )
    assert "distiller_fallback" in lesson.notes
    assert lesson.strategy


def test_extracted_confidence_tracks_reward() -> None:
    lesson = distill_lesson(_trajectory(), _verifier_verdict(0.85, success=True), now=NOW)
    assert lesson.provenance == "EXTRACTED"
    assert lesson.confidence == 0.85


def test_inferred_confidence_is_capped() -> None:
    lesson = distill_lesson(_trajectory(), llm_verdict(success=True, confidence=0.95), now=NOW)
    assert lesson.provenance == "INFERRED"
    assert lesson.confidence == INFERRED_CONFIDENCE_CAP


def test_pitfall_confidence_reflects_failure_decisiveness() -> None:
    lesson = distill_lesson(_trajectory(), _verifier_verdict(0.05, success=False), now=NOW)
    assert lesson.kind == "pitfall"
    assert lesson.confidence == 0.95  # decisive failure -> confident pitfall


def test_lesson_id_is_stable_and_slugged() -> None:
    a = distill_lesson(_trajectory(), _verifier_verdict(0.9, success=True), now=NOW)
    b = distill_lesson(_trajectory(), _verifier_verdict(0.9, success=True), now=NOW)
    assert a.lesson_id == b.lesson_id
    assert a.lesson_id.startswith("strategy-")
    assert "darkcodepe-fix-ingest-race" in a.lesson_id
