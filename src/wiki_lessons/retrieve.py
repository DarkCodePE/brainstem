"""Provenance-aware, topology-boosted lesson retrieval (SPEC-010 FR-5).

The lesson graph (same repo+domain, supersession links, term similarity)
gives a topology signal: deterministic label propagation clusters lessons
into communities, and a query that hits one lesson lifts its community
peers — graphify's "topology is the similarity signal", in stdlib
(ADR-033 D4: no networkx).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from wiki_lessons.distill import Lesson
from wiki_lessons.lifecycle import (
    DEFAULT_LESSON_HALFLIFE_DAYS,
    effective_weight,
    is_expired,
    is_superseded,
    terms_of,
)

#: Minimum strategy-term Jaccard for a similarity edge.
SIMILARITY_EDGE_JACCARD: float = 0.25

#: Ranking blend (ADR-033 D4).
W_RELEVANCE = 0.5
W_WEIGHT = 0.3
W_COMMUNITY = 0.2


@dataclass(frozen=True, slots=True)
class ScoredLesson:
    lesson: Lesson
    score: float
    relevance: float
    weight: float
    community: int


def _lesson_terms(lesson: Lesson) -> frozenset[str]:
    blob = " ".join(
        (lesson.title, lesson.strategy, lesson.domain, lesson.repo, *lesson.key_learnings)
    )
    return terms_of(blob)


def build_lesson_graph(
    lessons: tuple[Lesson, ...] | list[Lesson],
    *,
    similarity_threshold: float = SIMILARITY_EDGE_JACCARD,
) -> dict[str, set[str]]:
    """Undirected adjacency over lesson_ids."""
    adjacency: dict[str, set[str]] = {lesson.lesson_id: set() for lesson in lessons}
    by_id = {lesson.lesson_id: lesson for lesson in lessons}
    term_cache = {lesson.lesson_id: _lesson_terms(lesson) for lesson in lessons}

    def connect(a: str, b: str) -> None:
        if a != b and a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)

    items = sorted(by_id)
    for i, id_a in enumerate(items):
        a = by_id[id_a]
        for sid in a.supersedes:
            connect(id_a, sid)
        for id_b in items[i + 1 :]:
            b = by_id[id_b]
            if (a.repo, a.domain) == (b.repo, b.domain):
                connect(id_a, id_b)
                continue
            ta, tb = term_cache[id_a], term_cache[id_b]
            if ta and tb and len(ta & tb) / len(ta | tb) >= similarity_threshold:
                connect(id_a, id_b)
    return adjacency


def label_propagation_communities(
    adjacency: dict[str, set[str]],
    *,
    max_iters: int = 20,
) -> dict[str, int]:
    """Deterministic label propagation: sorted node order, smallest-label
    tiebreak. Returns lesson_id -> community id (0-based, stable)."""
    nodes = sorted(adjacency)
    label = {node: i for i, node in enumerate(nodes)}
    for _ in range(max_iters):
        changed = False
        for node in nodes:
            neighbors = adjacency[node]
            if not neighbors:
                continue
            counts: dict[int, int] = {}
            for peer in neighbors:
                counts[label[peer]] = counts.get(label[peer], 0) + 1
            best = min(
                counts,
                key=lambda candidate: (-counts[candidate], candidate),
            )
            if best != label[node]:
                label[node] = best
                changed = True
        if not changed:
            break
    # Renumber communities densely in first-seen (sorted-node) order.
    renumber: dict[int, int] = {}
    out: dict[str, int] = {}
    for node in nodes:
        raw = label[node]
        if raw not in renumber:
            renumber[raw] = len(renumber)
        out[node] = renumber[raw]
    return out


def _relevance(query_terms: frozenset[str], lesson_terms: frozenset[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & lesson_terms) / len(query_terms)


def rank_lessons(
    query: str,
    lessons: tuple[Lesson, ...] | list[Lesson],
    *,
    now: datetime | None = None,
    limit: int = 5,
    halflife_days: float = DEFAULT_LESSON_HALFLIFE_DAYS,
) -> tuple[ScoredLesson, ...]:
    """Top-k active lessons for a query (ADR-033 D4 blend).

    Superseded and expired lessons are excluded before scoring; provenance
    and decay enter through :func:`effective_weight`; community peers of a
    direct hit get a topology lift.
    """
    all_lessons = list(lessons)
    active = [
        lesson
        for lesson in all_lessons
        if not is_superseded(lesson, all_lessons) and not is_expired(lesson, now=now)
    ]
    if not active:
        return ()

    adjacency = build_lesson_graph(active)
    communities = label_propagation_communities(adjacency)

    query_terms = terms_of(query)
    relevance = {
        lesson.lesson_id: _relevance(query_terms, _lesson_terms(lesson)) for lesson in active
    }
    weight = {
        lesson.lesson_id: effective_weight(lesson, now=now, halflife_days=halflife_days)
        for lesson in active
    }

    by_community: dict[int, list[str]] = {}
    for lesson_id, community in communities.items():
        by_community.setdefault(community, []).append(lesson_id)

    scored: list[ScoredLesson] = []
    for lesson in active:
        lesson_id = lesson.lesson_id
        community = communities[lesson_id]
        peers = [pid for pid in by_community[community] if pid != lesson_id]
        community_bonus = max((relevance[pid] for pid in peers), default=0.0)
        score = (
            W_RELEVANCE * relevance[lesson_id]
            + W_WEIGHT * weight[lesson_id]
            + W_COMMUNITY * community_bonus
        )
        scored.append(
            ScoredLesson(
                lesson=lesson,
                score=round(score, 6),
                relevance=round(relevance[lesson_id], 6),
                weight=round(weight[lesson_id], 6),
                community=community,
            )
        )

    scored.sort(key=lambda item: (-item.score, item.lesson.lesson_id))
    return tuple(scored[: max(limit, 0)])
