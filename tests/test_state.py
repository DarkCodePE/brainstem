"""Tests for wiki agent Pydantic response schemas (state.py).

Validates all 8 Pydantic models: IngestResult, QueryResult, LintIssue,
LintResult, IndexResult, CaptureResult, ThemeCluster, ReviewResult.
"""

import pytest
from pydantic import ValidationError

from wiki_agent.state import (
    CaptureResult,
    IndexResult,
    IngestResult,
    LintIssue,
    LintResult,
    QueryResult,
    ReviewResult,
    ThemeCluster,
)


class TestIngestResult:
    def test_minimal_valid(self):
        r = IngestResult(summary_page="wiki/sources/foo.md")
        assert r.summary_page == "wiki/sources/foo.md"
        assert r.pages_created == []
        assert r.lessons_learned == []

    def test_full_valid(self):
        r = IngestResult(
            summary_page="wiki/sources/bar.md",
            pages_created=["wiki/entities/x.md"],
            pages_updated=["wiki/concepts/y.md"],
            entities_extracted=["Alice"],
            concepts_extracted=["RAG"],
            lessons_learned=["Always slugify"],
        )
        assert len(r.pages_created) == 1
        assert "Alice" in r.entities_extracted

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            IngestResult()


class TestQueryResult:
    def test_minimal_valid(self):
        r = QueryResult(answer="42")
        assert r.answer == "42"
        assert r.confidence == 0.0
        assert r.filed_path is None

    def test_full_valid(self):
        r = QueryResult(
            answer="The answer",
            citations=["wiki/sources/a.md"],
            confidence=0.95,
            filed_path="wiki/answers/q.md",
        )
        assert r.confidence == 0.95

    def test_missing_answer_raises(self):
        with pytest.raises(ValidationError):
            QueryResult()

    def test_confidence_bounds_low(self):
        with pytest.raises(ValidationError):
            QueryResult(answer="x", confidence=-0.1)

    def test_confidence_bounds_high(self):
        with pytest.raises(ValidationError):
            QueryResult(answer="x", confidence=1.1)


class TestLintIssue:
    def test_minimal_valid(self):
        issue = LintIssue(
            category="orphan",
            severity="high",
            page_path="wiki/concepts/old.md",
            description="No inbound links",
        )
        assert issue.auto_fixed is False

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            LintIssue(category="orphan")


class TestLintResult:
    def test_defaults(self):
        r = LintResult()
        assert r.issues == []
        assert r.pages_scanned == 0
        assert r.issues_fixed == 0

    def test_with_issues(self):
        issue = LintIssue(
            category="orphan",
            severity="low",
            page_path="wiki/x.md",
            description="orphan",
        )
        r = LintResult(issues=[issue], pages_scanned=5, issues_fixed=1)
        assert len(r.issues) == 1
        assert r.pages_scanned == 5


class TestIndexResult:
    def test_defaults(self):
        r = IndexResult()
        assert r.entries_added == 0
        assert r.stale_removed == 0
        assert r.backlinks_added == []

    def test_full_valid(self):
        r = IndexResult(
            entries_added=3,
            entries_updated=1,
            stale_removed=2,
            backlinks_added=["wiki/a.md"],
            broken_links=["wiki/b.md"],
        )
        assert r.entries_added == 3


class TestCaptureResult:
    def test_minimal_valid(self):
        r = CaptureResult(
            obs_id="OBS-2026-04-15-001",
            category="product-gap",
            confidence="high",
            file_path="observations/2026-04-15.md",
        )
        assert r.obs_id == "OBS-2026-04-15-001"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            CaptureResult(obs_id="OBS-1")


class TestThemeCluster:
    def test_minimal_valid(self):
        tc = ThemeCluster(
            theme_name="MCP gaps",
            obs_ids=["OBS-1", "OBS-2"],
            pattern_strength=2,
            proposed_graduation="concept-page",
            rationale="Repeated pattern",
        )
        assert tc.pattern_strength == 2

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ThemeCluster(theme_name="x")


class TestReviewResult:
    def test_defaults(self):
        r = ReviewResult()
        assert r.observations_reviewed == 0
        assert r.themes == []
        assert r.unmatched_count == 0

    def test_with_themes(self):
        tc = ThemeCluster(
            theme_name="T",
            obs_ids=["O1"],
            pattern_strength=1,
            proposed_graduation="schema-rule",
            rationale="R",
        )
        r = ReviewResult(observations_reviewed=5, themes=[tc], unmatched_count=2)
        assert len(r.themes) == 1
        assert r.observations_reviewed == 5
