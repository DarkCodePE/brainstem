"""Tests for `wiki_qa.cli` — exit codes and baseline gating behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wiki_qa.baseline import load_baseline
from wiki_qa.cli import main


class TestHealthCommand:
    def test_json_output(self, graph_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["health", "--graph", str(graph_file), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["summary"]["total"] == 4

    def test_update_baseline_writes_file(self, graph_file: Path, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        rc = main(
            [
                "health",
                "--graph",
                str(graph_file),
                "--baseline",
                str(baseline),
                "--update-baseline",
            ]
        )
        assert rc == 0
        assert baseline.is_file()
        assert load_baseline(baseline)["duplicate_slugs"] == ["foo"]

    def test_update_baseline_requires_path(self, graph_file: Path) -> None:
        rc = main(["health", "--graph", str(graph_file), "--update-baseline"])
        assert rc == 2

    def test_fail_on_regression_passes_against_self(self, graph_file: Path, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        main(
            [
                "health",
                "--graph",
                str(graph_file),
                "--baseline",
                str(baseline),
                "--update-baseline",
            ]
        )
        rc = main(
            [
                "health",
                "--graph",
                str(graph_file),
                "--baseline",
                str(baseline),
                "--fail-on-regression",
            ]
        )
        assert rc == 0

    def test_fail_on_regression_with_empty_baseline(self, graph_file: Path) -> None:
        # No baseline file -> every issue is a regression -> exit 1.
        rc = main(["health", "--graph", str(graph_file), "--fail-on-regression"])
        assert rc == 1


class TestTourCommand:
    def test_tour_writes_markdown(self, graph_file: Path, tmp_path: Path) -> None:
        out = tmp_path / "tour.md"
        rc = main(["tour", "--graph", str(graph_file), "--out", str(out)])
        assert rc == 0
        assert "Guided Tour" in out.read_text(encoding="utf-8")
