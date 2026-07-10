"""Tests for the wiki page substrate (SPEC-010 FR-3, wiki_lessons/page.py).

The frontmatter must satisfy the existing wiki contract: required fields
{title, date, sources, tags, origin} with a valid origin enum value (AC-1).
"""

from __future__ import annotations

from datetime import UTC, datetime

import yaml

from wiki_lessons.distill import Trajectory, distill_lesson
from wiki_lessons.page import lesson_page_path, parse_lesson_page, render_lesson_page
from wiki_lessons.verdict import Verdict

NOW = datetime(2026, 6, 10, 12, 0, tzinfo=UTC)

REQUIRED_FIELDS = {"title", "date", "sources", "tags", "origin"}
VALID_ORIGINS = {"human", "llm-generated", "llm-synthesized", "mcp-ingested"}


def _lesson(**verdict_overrides: object):
    verdict_kwargs: dict = {
        "source": "verifier",
        "reward": 0.88,
        "success": True,
        "kind": "test_execution",
        "components": (("f2p_rate", 1.0), ("p2p_rate", 0.88)),
    }
    verdict_kwargs.update(verdict_overrides)
    trajectory = Trajectory(
        task_id="DarkCodePE/fix-dedup-spaces",
        repo="DarkCodePE/second-brain-wiki",
        domain="ingest",
        instruction="Make the dedup regex match filenames containing spaces",
        actions=("reproduce with spaced filename", "fix regex", "run tests"),
        reference="https://github.com/DarkCodePE/second-brain-wiki/pull/120",
    )
    return distill_lesson(trajectory, Verdict(**verdict_kwargs), now=NOW)


def _frontmatter(page: str) -> dict:
    return yaml.safe_load(page.split("---", 2)[1])


def test_page_carries_all_wiki_required_fields() -> None:
    page = render_lesson_page(_lesson())
    front = _frontmatter(page)
    assert REQUIRED_FIELDS.issubset(front.keys())
    assert front["origin"] in VALID_ORIGINS
    assert isinstance(front["sources"], list) and front["sources"]
    assert isinstance(front["tags"], list) and "lesson" in front["tags"]
    assert front["date"] == "2026-06-10"


def test_page_carries_lesson_typed_fields() -> None:
    front = _frontmatter(render_lesson_page(_lesson()))
    assert front["type"] == "lesson"
    assert front["provenance"] == "EXTRACTED"
    assert front["lesson_kind"] == "strategy"
    assert front["reward"] == 0.88
    assert front["source_key"]


def test_round_trip_preserves_identity_and_verdict() -> None:
    original = _lesson()
    parsed = parse_lesson_page(render_lesson_page(original))
    assert parsed is not None
    assert parsed.lesson_id == original.lesson_id
    assert parsed.source_key == original.source_key
    assert parsed.kind == original.kind
    assert parsed.provenance == original.provenance
    assert parsed.confidence == original.confidence
    assert parsed.verdict.reward == original.verdict.reward
    assert parsed.strategy == original.strategy
    assert parsed.key_learnings == original.key_learnings


def test_non_lesson_and_malformed_pages_parse_to_none() -> None:
    assert parse_lesson_page("# Just a heading\nNo frontmatter here.") is None
    assert parse_lesson_page("---\ntitle: An article\norigin: human\n---\nBody") is None
    assert parse_lesson_page("---\ntype: lesson\n---\nmissing required keys") is None


def test_page_path_lives_under_wiki_lessons() -> None:
    lesson = _lesson()
    path = lesson_page_path(lesson)
    assert path == f"wiki/lessons/{lesson.lesson_id}.md"
