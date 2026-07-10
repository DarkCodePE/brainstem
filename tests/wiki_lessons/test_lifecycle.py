"""Tests for lesson lifecycle governance (SPEC-010 FR-4, wiki_lessons/lifecycle.py).

The ADR-033 D2/D3 rules are mechanical: INFERRED never supersedes EXTRACTED
(AC-4), contradictions are flagged, decay reuses ADR-027's recency curve.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from wiki_lessons.distill import Trajectory, distill_lesson
from wiki_lessons.lifecycle import (
    detect_contradictions,
    effective_weight,
    is_expired,
    is_superseded,
    resolve_supersession,
)
from wiki_lessons.verdict import Verdict, llm_verdict

NOW = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)


def _verifier(reward: float, *, success: bool) -> Verdict:
    return Verdict(source="verifier", reward=reward, success=success, kind="test_execution")


def _lesson(
    task_id: str = "DarkCodePE/fix-ingest-race",
    *,
    verdict: Verdict | None = None,
    when: datetime = NOW,
    domain: str = "ingest",
    strategy: str | None = None,
):
    trajectory = Trajectory(
        task_id=task_id,
        repo="DarkCodePE/second-brain-wiki",
        domain=domain,
        instruction="Fix the ingest race condition",
        actions=("add lockfile", "run tests"),
    )
    lesson = distill_lesson(trajectory, verdict or _verifier(0.9, success=True), now=when)
    if strategy is not None:
        lesson = replace(lesson, strategy=strategy)
    return lesson


def test_extracted_supersedes_prior_extracted_on_same_source_key() -> None:
    old = _lesson(when=NOW - timedelta(days=30))
    old = replace(old, lesson_id="strategy-old")
    new = _lesson(when=NOW)
    plan = resolve_supersession(new, [old])
    assert plan.superseded == ("strategy-old",)
    assert plan.new.supersedes == ("strategy-old",)
    assert plan.blocked == ()
    assert is_superseded(old, [plan.new, old])


def test_inferred_cannot_supersede_extracted() -> None:
    extracted = _lesson(when=NOW - timedelta(days=30))
    inferred = _lesson(verdict=llm_verdict(success=True, confidence=0.9), when=NOW)
    inferred = replace(inferred, lesson_id="strategy-inferred")
    plan = resolve_supersession(inferred, [extracted])
    assert plan.superseded == ()
    assert plan.blocked == (extracted.lesson_id,)
    assert "inferred_cannot_supersede_extracted" in plan.notes
    assert plan.new.supersedes == ()


def test_inferred_supersedes_inferred() -> None:
    old = _lesson(verdict=llm_verdict(success=True, confidence=0.5), when=NOW - timedelta(days=10))
    old = replace(old, lesson_id="strategy-old-inferred")
    new = _lesson(verdict=llm_verdict(success=True, confidence=0.6), when=NOW)
    plan = resolve_supersession(new, [old])
    assert plan.superseded == ("strategy-old-inferred",)


def test_different_source_key_is_untouched() -> None:
    other_task = _lesson(task_id="DarkCodePE/other-task")
    new = _lesson(when=NOW)
    plan = resolve_supersession(new, [other_task])
    assert plan.superseded == ()
    assert plan.blocked == ()


def test_expiry_and_no_expiry() -> None:
    lesson = _lesson()
    assert is_expired(lesson, now=NOW) is False
    expiring = replace(lesson, expires="2026-06-01")
    assert is_expired(expiring, now=NOW) is True
    future = replace(lesson, expires="2026-12-31")
    assert is_expired(future, now=NOW) is False


def test_effective_weight_orders_by_provenance_and_age() -> None:
    fresh_extracted = _lesson(when=NOW)
    fresh_inferred = _lesson(verdict=llm_verdict(success=True, confidence=0.9), when=NOW)
    old_extracted = _lesson(when=NOW - timedelta(days=180))

    w_fresh = effective_weight(fresh_extracted, now=NOW)
    w_inferred = effective_weight(fresh_inferred, now=NOW)
    w_old = effective_weight(old_extracted, now=NOW)

    assert w_fresh > w_inferred  # provenance factor + confidence cap
    assert w_fresh > w_old  # 180 days = 2 half-lives -> ~0.25x recency
    assert 0.0 <= w_old <= 1.0


def test_contradiction_flagged_for_opposite_kinds_with_shared_terms() -> None:
    strategy = _lesson(strategy="Acquire the ingest lockfile before starting the watcher daemon.")
    pitfall = _lesson(
        verdict=_verifier(0.1, success=False),
        strategy="Acquire the ingest lockfile before starting the watcher daemon.",
    )
    flagged = detect_contradictions([strategy, pitfall])
    assert flagged == (
        (min(strategy.lesson_id, pitfall.lesson_id), max(strategy.lesson_id, pitfall.lesson_id)),
    )


def test_no_contradiction_across_domains_or_same_kind() -> None:
    a = _lesson(strategy="Acquire the lockfile first.")
    b = _lesson(
        verdict=_verifier(0.1, success=False),
        domain="publishing",
        strategy="Acquire the lockfile first.",
    )
    assert detect_contradictions([a, b]) == ()
    c = _lesson(task_id="DarkCodePE/another", strategy="Acquire the lockfile first.")
    assert detect_contradictions([a, c]) == ()
