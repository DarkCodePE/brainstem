"""Hermetic tests for ``wiki_repos.history`` (PRD-014 / ADR-030).

The ``opener`` seam is always injected — no real network, no git, no subprocess.
Covers PR/commit mining, conventional-commit classification, the merged-only
filter, truncation accounting, and the degrade-first contract (FR-5 / AC-2 / AC-4).
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from wiki_repos.history import _API_HOST, classify_commit, mine_history
from wiki_repos.types import RepoRef

REF = RepoRef(owner="octocat", repo="hello-world")


def _routed_opener(pulls: list | None, commits: list | None, *, calls: list[str] | None = None):
    """Build a fake ``(url) -> (status, body)`` that routes by endpoint.

    A ``None`` payload for a leg makes that leg return HTTP 500 (degrade).
    """

    def _open(url: str) -> tuple[int, bytes]:
        if calls is not None:
            calls.append(url)
        if "/pulls" in url:
            payload = pulls
        elif "/commits" in url:
            payload = commits
        else:  # pragma: no cover — only the two endpoints are queried
            return 404, b""
        if payload is None:
            return 500, b""
        return 200, json.dumps(payload).encode()

    return _open


def _pr(number: int, title: str, *, merged: bool = True, body: str = "", labels=()) -> dict:
    return {
        "number": number,
        "title": title,
        "merged_at": "2026-06-01T10:00:00Z" if merged else None,
        "user": {"login": "alice"},
        "labels": [{"name": n} for n in labels],
        "body": body,
    }


def _commit(sha: str, subject: str) -> dict:
    return {
        "sha": sha,
        "commit": {"message": subject, "author": {"date": "2026-06-01T10:00:00Z"}},
    }


# --------------------------------------------------------------------------- #
# classify_commit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "subject,expected",
    [
        ("fix: null deref in parser", "fix"),
        ("feat(api): add pagination", "feat"),
        ("refactor!: split module", "refactor"),
        ("perf: cache lookups", "perf"),
        ("security: patch SSRF", "security"),
        ("sec: bump dep", "security"),
        ("docs: update readme", "docs"),
        ("chore(deps): bump", "chore"),
        ("build: tweak CI", "chore"),
        ("Merge pull request #5", "other"),
        ("random subject no prefix", "other"),
    ],
)
def test_classify_commit(subject, expected):
    assert classify_commit(subject) == expected


# --------------------------------------------------------------------------- #
# mine_history — happy path
# --------------------------------------------------------------------------- #
def test_mine_happy_path():
    opener = _routed_opener(
        pulls=[_pr(10, "Add retry logic", body="Fixes flakiness in the worker loop.")],
        commits=[
            _commit("abcdef1234", "fix: handle empty input"),
            _commit("1111111", "docs: tidy"),
        ],
    )
    hist = mine_history(REF, opener=opener)

    assert hist is not None
    assert hist.stats.n_prs == 1
    assert hist.stats.n_commits == 2
    assert hist.merged_prs[0].number == 10
    assert hist.merged_prs[0].author == "alice"
    assert hist.merged_prs[0].body_excerpt.startswith("Fixes flakiness")
    assert hist.commits[0].kind == "fix"
    assert hist.commits[0].sha == "abcdef1"  # truncated to 7
    assert hist.kind_counts == {"fix": 1, "docs": 1}


def test_unmerged_prs_are_filtered():
    opener = _routed_opener(
        pulls=[_pr(1, "merged one"), _pr(2, "still open", merged=False)],
        commits=[],
    )
    hist = mine_history(REF, opener=opener)
    assert hist is not None
    assert [pr.number for pr in hist.merged_prs] == [1]


def test_truncation_flag_when_page_full():
    # max_prs=2 and exactly 2 merged PRs returned → truncated True (FR-4).
    opener = _routed_opener(
        pulls=[_pr(1, "a"), _pr(2, "b")],
        commits=[],
    )
    hist = mine_history(REF, opener=opener, max_prs=2)
    assert hist is not None
    assert hist.stats.truncated is True


def test_only_api_github_host_is_queried():
    """AC-4: every network call targets the allowlisted GitHub API host, no clone."""
    calls: list[str] = []
    opener = _routed_opener(pulls=[_pr(1, "a")], commits=[_commit("a", "fix: x")], calls=calls)
    mine_history(REF, opener=opener)
    assert calls, "expected at least one API call"
    assert all(u.startswith(_API_HOST + "/repos/octocat/hello-world") for u in calls)


# --------------------------------------------------------------------------- #
# mine_history — degrade paths (FR-5 / AC-2)
# --------------------------------------------------------------------------- #
def test_degrade_both_legs_unreachable_returns_none():
    hist = mine_history(REF, opener=_routed_opener(pulls=None, commits=None))
    assert hist is None


def test_degrade_reachable_but_empty_returns_none():
    hist = mine_history(REF, opener=_routed_opener(pulls=[], commits=[]))
    assert hist is None


def test_degrade_one_leg_only_still_returns_history():
    # commits reachable, pulls 500 → still usable (commits leg alone).
    hist = mine_history(REF, opener=_routed_opener(pulls=None, commits=[_commit("a", "feat: y")]))
    assert hist is not None
    assert hist.stats.n_prs == 0
    assert hist.stats.n_commits == 1


def test_opener_raising_httperror_degrades_not_raises():
    def _boom(url: str):
        raise urllib.error.HTTPError(url, 403, "rate limited", hdrs=None, fp=None)

    # 403 on both legs → None, no exception escapes.
    assert mine_history(REF, opener=_boom) is None


def test_opener_raising_generic_exception_degrades():
    def _boom(url: str):
        raise TimeoutError("slow")

    assert mine_history(REF, opener=_boom) is None


# --------------------------------------------------------------------------- #
# AC-4 — no clone / no git subprocess; hostile-owner cannot escape the API path
# --------------------------------------------------------------------------- #
def test_no_git_subprocess_is_spawned(monkeypatch):
    """AC-4: history mining must NEVER shell out to git / any subprocess."""
    import os
    import subprocess

    def _explode(*a, **k):  # any subprocess/system call is a no-clone violation
        raise AssertionError("subprocess/git was invoked during mine_history")

    monkeypatch.setattr(subprocess, "Popen", _explode)
    monkeypatch.setattr(subprocess, "run", _explode)
    monkeypatch.setattr(os, "system", _explode)

    opener = _routed_opener(pulls=[_pr(1, "a")], commits=[_commit("a", "fix: x")])
    hist = mine_history(REF, opener=opener)
    assert hist is not None  # mined purely via the injected opener, no git


def test_hostile_owner_cannot_escape_api_path():
    """AC-4 / SR-1: a traversal-y owner must NOT produce a URL escaping
    /repos/{owner}/{repo} nor reach a host off the allowlist."""
    calls: list[str] = []
    hostile = RepoRef(owner="../../evil", repo="x")
    opener = _routed_opener(pulls=[_pr(1, "a")], commits=[_commit("a", "fix: x")], calls=calls)
    mine_history(hostile, opener=opener)
    assert calls, "expected at least one API call"
    for u in calls:
        # Every call stays on the hardcoded API host (no scheme/host swap).
        assert u.startswith(_API_HOST + "/repos/"), u
        # The path after the host is never allowed to climb above /repos.
        path = u[len(_API_HOST) :]
        assert ".." not in path.split("?", 1)[0].split("/") or path.startswith("/repos/"), u
        assert "://evil" not in u and "@" not in u


# --------------------------------------------------------------------------- #
# AC-5 — call budget: exactly two GitHub calls (1 pulls + 1 commits)
# --------------------------------------------------------------------------- #
def test_history_makes_exactly_two_github_calls():
    """AC-5: history mining is one /pulls page + one /commits page → 2 calls."""
    calls: list[str] = []
    opener = _routed_opener(pulls=[_pr(1, "a")], commits=[_commit("a", "fix: x")], calls=calls)
    mine_history(REF, opener=opener)
    assert len(calls) == 2
    assert sum(1 for u in calls if "/pulls" in u) == 1
    assert sum(1 for u in calls if "/commits" in u) == 1


# --------------------------------------------------------------------------- #
# FR-1 — merged PRs are ordered most-recently-MERGED first (not by updated)
# --------------------------------------------------------------------------- #
def test_merged_prs_sorted_by_merged_at_desc():
    """The /pulls feed is sorted by 'updated'; the kept PRs must be re-sorted by
    merged_at desc so the section's 'newest first' claim is honest (FR-1)."""
    older = _pr(1, "older merge")
    older["merged_at"] = "2026-01-01T00:00:00Z"
    newer = _pr(2, "newer merge")
    newer["merged_at"] = "2026-06-01T00:00:00Z"
    # API returns the older-merged PR first (it was 'updated' more recently).
    hist = mine_history(REF, opener=_routed_opener(pulls=[older, newer], commits=[]))
    assert hist is not None
    assert [pr.number for pr in hist.merged_prs] == [2, 1]


# --------------------------------------------------------------------------- #
# _excerpt — strips PR-template boilerplate, ellipsises on a word boundary
# --------------------------------------------------------------------------- #
def test_excerpt_strips_template_boilerplate():
    from wiki_repos.history import _excerpt

    body = (
        "## Description\n"
        "<!-- describe your change -->\n"
        "- [ ] tests added\n"
        "Fixes a race in the worker loop.\n"
    )
    out = _excerpt(body)
    assert out == "Fixes a race in the worker loop."
    assert "##" not in out and "[ ]" not in out and "<!--" not in out


def test_excerpt_truncates_with_ellipsis_on_word_boundary():
    from wiki_repos.history import _MAX_BODY_CHARS, _excerpt

    out = _excerpt("word " * 200)
    assert out.endswith("…")
    assert len(out) <= _MAX_BODY_CHARS + 1  # +1 for the ellipsis char


def test_commit_date_falls_back_to_committer():
    """Partial author object (no 'date') must fall back to committer.date."""
    item = {
        "sha": "deadbee1234",
        "commit": {
            "message": "fix: x",
            "author": {"name": "alice"},  # no 'date'
            "committer": {"date": "2026-06-02T00:00:00Z"},
        },
    }
    hist = mine_history(REF, opener=_routed_opener(pulls=[], commits=[item]))
    assert hist is not None
    assert hist.commits[0].date == "2026-06-02T00:00:00Z"
