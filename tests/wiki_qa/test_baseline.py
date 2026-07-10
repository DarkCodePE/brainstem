"""Tests for `wiki_qa.baseline` — the CI regression gate."""

from __future__ import annotations

from pathlib import Path

from wiki_qa.baseline import (
    compute_regressions,
    load_baseline,
    report_keys,
    save_baseline,
)
from wiki_qa.graph import KnowledgeGraph
from wiki_qa.linter import lint


class TestReportKeys:
    def test_keys_are_sorted_and_stable(self, graph: KnowledgeGraph) -> None:
        keys = report_keys(lint(graph))
        assert keys["orphans"] == sorted(keys["orphans"])
        assert "concepts/foo.md -> missing-page" in keys["broken_wikilinks"]
        assert keys["duplicate_slugs"] == ["foo"]


class TestBaselineRoundTrip:
    def test_save_then_load(self, graph: KnowledgeGraph, tmp_path: Path) -> None:
        report = lint(graph)
        path = tmp_path / "baseline.json"
        save_baseline(path, report)
        loaded = load_baseline(path)
        assert loaded == report_keys(report)

    def test_missing_baseline_is_empty(self, tmp_path: Path) -> None:
        loaded = load_baseline(tmp_path / "does-not-exist.json")
        assert loaded == {"orphans": [], "broken_wikilinks": [], "duplicate_slugs": []}


class TestRegressions:
    def test_no_regression_against_self(self, graph: KnowledgeGraph) -> None:
        report = lint(graph)
        baseline = report_keys(report)
        regressions = compute_regressions(report, baseline)
        assert not regressions.has_regressions
        assert regressions.total == 0

    def test_empty_baseline_flags_everything(self, graph: KnowledgeGraph) -> None:
        report = lint(graph)
        empty = {"orphans": [], "broken_wikilinks": [], "duplicate_slugs": []}
        regressions = compute_regressions(report, empty)
        assert regressions.total == report.total_issues

    def test_only_new_issue_is_flagged(self, graph: KnowledgeGraph) -> None:
        report = lint(graph)
        baseline = report_keys(report)
        # Accept everything except one orphan -> exactly that orphan regresses.
        baseline["orphans"] = [o for o in baseline["orphans"] if o != "article:other/foo"]
        regressions = compute_regressions(report, baseline)
        assert regressions.orphans == ("article:other/foo",)
        assert regressions.total == 1
