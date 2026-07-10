"""Tests for :class:`wiki_publishing.project_ledger.ProjectLedger` (ADR-026).

The ledger is the launch-once state for build-in-public posts: it records when a
project was launched (so a second ``project_launch`` is refused/redirected) and
when it was last posted. Pure stdlib (json + pathlib), no network — these tests
exercise it over a real temp file only (``tmp_path``), no mocking needed."""

from __future__ import annotations

import json
from pathlib import Path

from wiki_publishing.project_ledger import ProjectLedger


def test_was_launched_false_then_record_then_true(tmp_path: Path) -> None:
    ledger = ProjectLedger(tmp_path / "ledger.json")
    assert ledger.was_launched("DarkCodePE/second-brain-wiki") is False

    ledger.record_launch("DarkCodePE/second-brain-wiki", when="2026-06-05T00:00:00+00:00")
    assert ledger.was_launched("DarkCodePE/second-brain-wiki") is True


def test_record_launch_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "ledger.json"  # parent dir does not exist yet
    ledger = ProjectLedger(path)
    ledger.record_launch("acme/widget", when="2026-06-05T12:00:00+00:00")

    # A fresh instance reading the same file sees the launch (creates parent dir).
    assert path.exists()
    reloaded = ProjectLedger(path)
    assert reloaded.was_launched("acme/widget") is True


def test_record_launch_stores_launched_at(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = ProjectLedger(path)
    ledger.record_launch("acme/widget", when="2026-06-05T12:00:00+00:00")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["acme/widget"]["launched_at"] == "2026-06-05T12:00:00+00:00"


def test_record_post_updates_last_post_at_without_launching(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = ProjectLedger(path)
    ledger.record_post("acme/widget", when="2026-06-05T09:00:00+00:00")

    # record_post must NOT flip launched (a weekly post is not a launch).
    assert ledger.was_launched("acme/widget") is False
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["acme/widget"]["last_post_at"] == "2026-06-05T09:00:00+00:00"
    assert data["acme/widget"]["launched_at"] is None


def test_record_post_then_launch_preserves_both(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = ProjectLedger(path)
    ledger.record_post("acme/widget", when="2026-06-01T00:00:00+00:00")
    ledger.record_launch("acme/widget", when="2026-06-05T00:00:00+00:00")
    ledger.record_post("acme/widget", when="2026-06-06T00:00:00+00:00")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["acme/widget"]["launched_at"] == "2026-06-05T00:00:00+00:00"
    assert data["acme/widget"]["last_post_at"] == "2026-06-06T00:00:00+00:00"


def test_missing_file_is_empty_ledger(tmp_path: Path) -> None:
    ledger = ProjectLedger(tmp_path / "does-not-exist.json")
    assert ledger.was_launched("anything") is False


def test_corrupt_file_is_tolerated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    ledger = ProjectLedger(path)
    assert ledger.was_launched("anything") is False
    # And it can still record over the corrupt file.
    ledger.record_launch("acme/widget", when="2026-06-05T00:00:00+00:00")
    assert ledger.was_launched("acme/widget") is True


def test_non_dict_json_is_tolerated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON, wrong shape
    ledger = ProjectLedger(path)
    assert ledger.was_launched("anything") is False


def test_distinct_repos_are_independent(tmp_path: Path) -> None:
    ledger = ProjectLedger(tmp_path / "ledger.json")
    ledger.record_launch("a/one", when="2026-06-05T00:00:00+00:00")
    assert ledger.was_launched("a/one") is True
    assert ledger.was_launched("b/two") is False
