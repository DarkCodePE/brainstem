"""Lesson memory bounded context (PRD-016 / SPEC-010 / ADR-033).

Turns (trajectory, objective verdict) pairs into typed, lifecycle-managed
wiki pages — *lessons* — and retrieves them with provenance-aware,
graph-topology-boosted ranking.

Pure functions throughout: no network, no DB, no router imports. LLM seams
are injectable callables (ADR-031 pattern); parsers degrade to ``None`` on
malformed input.
"""

from wiki_lessons.distill import DistillContext, Lesson, Trajectory, distill_lesson
from wiki_lessons.lifecycle import (
    SupersessionPlan,
    detect_contradictions,
    effective_weight,
    is_expired,
    resolve_supersession,
)
from wiki_lessons.page import lesson_page_path, parse_lesson_page, render_lesson_page
from wiki_lessons.retrieve import (
    ScoredLesson,
    build_lesson_graph,
    label_propagation_communities,
    rank_lessons,
)
from wiki_lessons.verdict import Verdict, llm_verdict, parse_reward_json, parse_reward_txt

__all__ = [
    "DistillContext",
    "Lesson",
    "ScoredLesson",
    "SupersessionPlan",
    "Trajectory",
    "Verdict",
    "build_lesson_graph",
    "detect_contradictions",
    "distill_lesson",
    "effective_weight",
    "is_expired",
    "label_propagation_communities",
    "lesson_page_path",
    "llm_verdict",
    "parse_lesson_page",
    "parse_reward_json",
    "parse_reward_txt",
    "rank_lessons",
    "render_lesson_page",
    "resolve_supersession",
]
