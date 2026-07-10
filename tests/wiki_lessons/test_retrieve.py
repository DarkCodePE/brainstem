"""Tests for topology-boosted retrieval (SPEC-010 FR-5, wiki_lessons/retrieve.py)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from wiki_lessons.distill import Trajectory, distill_lesson
from wiki_lessons.retrieve import (
    build_lesson_graph,
    label_propagation_communities,
    rank_lessons,
)
from wiki_lessons.verdict import Verdict, llm_verdict

NOW = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)


def _lesson(
    task_id: str,
    *,
    repo: str = "DarkCodePE/second-brain-wiki",
    domain: str = "ingest",
    strategy: str,
    verdict: Verdict | None = None,
    expires: str | None = None,
):
    trajectory = Trajectory(
        task_id=task_id,
        repo=repo,
        domain=domain,
        instruction=strategy[:80],
        actions=("step one", "step two"),
    )
    verdict = verdict or Verdict(source="verifier", reward=0.9, success=True, kind="test_execution")
    lesson = distill_lesson(trajectory, verdict, now=NOW)
    lesson = replace(lesson, strategy=strategy)
    if expires:
        lesson = replace(lesson, expires=expires)
    return lesson


def test_graph_connects_same_repo_domain_and_supersession() -> None:
    a = _lesson("o/t1", strategy="Use the lockfile guard around the watcher.")
    b = _lesson("o/t2", strategy="Batch the embeddings to cut latency.")
    c = _lesson(
        "x/t3",
        repo="other/repo",
        domain="publishing",
        strategy="Schedule LinkedIn drafts after media upload.",
    )
    adjacency = build_lesson_graph([a, b, c])
    assert b.lesson_id in adjacency[a.lesson_id]  # same repo+domain
    assert c.lesson_id not in adjacency[a.lesson_id]

    superseder = replace(a, lesson_id="newer", supersedes=(c.lesson_id,))
    adjacency = build_lesson_graph([superseder, c])
    assert c.lesson_id in adjacency["newer"]  # supersession edge


def test_communities_are_deterministic_and_group_connected_lessons() -> None:
    a = _lesson("o/t1", strategy="Lockfile strategy one.")
    b = _lesson("o/t2", strategy="Lockfile strategy two.")
    c = _lesson(
        "x/t3",
        repo="other/repo",
        domain="publishing",
        strategy="Completely unrelated publishing tactic.",
    )
    adjacency = build_lesson_graph([a, b, c])
    first = label_propagation_communities(adjacency)
    second = label_propagation_communities(adjacency)
    assert first == second
    assert first[a.lesson_id] == first[b.lesson_id]
    assert first[a.lesson_id] != first[c.lesson_id]


def test_superseded_and_expired_lessons_are_excluded() -> None:
    old = _lesson("o/t1", strategy="Old ingest lockfile approach.")
    new = replace(
        _lesson("o/t1", strategy="New ingest lockfile approach."),
        lesson_id="newer",
        supersedes=(old.lesson_id,),
    )
    expired = _lesson("o/t4", strategy="Expired ingest hint.", expires="2026-01-01")
    ranked = rank_lessons("ingest lockfile", [old, new, expired], now=NOW)
    ids = [scored.lesson.lesson_id for scored in ranked]
    assert "newer" in ids
    assert old.lesson_id not in ids
    assert expired.lesson_id not in ids


def test_extracted_outranks_inferred_at_equal_relevance() -> None:
    extracted = _lesson("o/t1", strategy="Guard the ingest daemon with a lockfile.")
    inferred = _lesson(
        "o/t2",
        strategy="Guard the ingest daemon with a lockfile.",
        verdict=llm_verdict(success=True, confidence=0.9),
    )
    ranked = rank_lessons("ingest daemon lockfile", [extracted, inferred], now=NOW)
    assert ranked[0].lesson.lesson_id == extracted.lesson_id
    assert ranked[0].weight > ranked[1].weight


def test_community_bonus_lifts_peer_without_direct_match() -> None:
    hit = _lesson("o/t1", strategy="Tune the dedup regex for spaced filenames.")
    peer = _lesson("o/t2", strategy="Throttle the batch worker queue.")  # same repo+domain
    stranger = _lesson(
        "x/t3",
        repo="other/repo",
        domain="publishing",
        strategy="Unrelated publishing approach.",
    )
    ranked = rank_lessons("dedup regex spaced filenames", [hit, peer, stranger], now=NOW)
    by_id = {scored.lesson.lesson_id: scored for scored in ranked}
    assert by_id[hit.lesson_id].score > by_id[peer.lesson_id].score
    assert by_id[peer.lesson_id].score > by_id[stranger.lesson_id].score  # topology lift


def test_limit_and_empty_inputs() -> None:
    assert rank_lessons("anything", [], now=NOW) == ()
    lessons = [_lesson(f"o/t{i}", strategy=f"Ingest strategy number {i}.") for i in range(8)]
    ranked = rank_lessons("ingest strategy", lessons, now=NOW, limit=3)
    assert len(ranked) == 3
