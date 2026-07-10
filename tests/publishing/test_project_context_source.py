"""Tests for :class:`wiki_publishing.project_context_source.ProjectContextSource`
(ADR-026 / PRD-019).

Hermetic by construction: every test injects a fake ``git_runner`` (a
``list[str] -> str`` returning canned ``git log`` stdout) and a fake
``docs_reader`` (a ``() -> list[tuple[relpath, text]]`` returning canned ADR/PRD
text). No real ``git``, no ``subprocess``, no filesystem, no network ever runs.

Coverage:
- ``project_feature``: topic selects the right ADR; body carries the decision
  text + a measured number + a real commit subject.
- ``project_weekly``: lists real commit subjects; empty window → minimal honest
  snippet (no padding).
- ``project_launch``: README vision + an accepted ADR's title/decision.
- SECURITY: the diff-arg guard refuses ``-p`` / ``--patch`` / ``diff`` / ``show``.
- ContentSource conformance: ``search`` ignores ``query``/``categories`` and
  returns the snippet; the instance satisfies the ``ContentSource`` protocol.
"""

from __future__ import annotations

import pytest

from wiki_publishing.linkedin_draft import ContentSource, WikiSnippet
from wiki_publishing.project_context_source import (
    ProjectContextSource,
    _guard_git_args,
    _normalize_remote_url,
)

# --------------------------------------------------------------------------- #
# Fakes / fixtures
# --------------------------------------------------------------------------- #

_README = (
    "# second-brain-wiki\n\n"
    "Turn your bookmarks and notes into a living, queryable knowledge base "
    "that drafts content for you.\n"
)

_ADR_022 = (
    "---\n"
    "id: ADR-022\n"
    'title: "Repo as a knowledge source: ingest your own code into the wiki"\n'
    "status: accepted\n"
    "date: 2026-06-01\n"
    "---\n\n"
    "# ADR-022: Repo as a Knowledge Source\n\n"
    "## Context and problem statement\n\n"
    "Your own code was invisible to the wiki; only bookmarks were ingested.\n\n"
    "## Decision outcome\n\n"
    "We ingest the local repo via a deterministic code-graph builder, so the "
    "wiki answers questions about your own code. The graph build is 26x faster "
    "than the prior prototype.\n"
)

_ADR_025 = (
    "---\n"
    "id: ADR-025\n"
    'title: "Lightweight post-context acquisition"\n'
    "status: accepted\n"
    "date: 2026-06-04\n"
    "---\n\n"
    "# ADR-025: Lightweight Post-Context Acquisition\n\n"
    "## Context and problem statement\n\n"
    "The heavy ingest was overkill for a quick post about a tool.\n\n"
    "## Decision outcome\n\n"
    "We add a live ContentSource that fetches repo metadata, feeding the same "
    "drafter with zero drafter changes.\n"
)

_ADR_PROPOSED = (
    "---\n"
    "id: ADR-099\n"
    'title: "A proposed thing not yet accepted"\n'
    "status: proposed\n"
    "date: 2026-06-05\n"
    "---\n\n"
    "# ADR-099: A Proposed Thing\n\n"
    "## Decision outcome\n\nDo something later.\n"
)


def _docs_reader_full():
    return lambda: [
        ("docs/ADR-022-repo-as-knowledge-source.md", _ADR_022),
        ("docs/ADR-025-lightweight-post-context.md", _ADR_025),
        ("docs/ADR-099-proposed.md", _ADR_PROPOSED),
        ("README.md", _README),
    ]


def _git_runner(mapping: dict[str, str]):
    """Build a fake git_runner that returns canned stdout keyed by a substring
    of the joined args. Falls back to '' (empty stdout)."""

    def runner(args: list[str]) -> str:
        joined = " ".join(args)
        for needle, out in mapping.items():
            if needle in joined:
                return out
        return ""

    return runner


# --------------------------------------------------------------------------- #
# Security: the diff-arg guard
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bad_args",
    [
        ["log", "-p"],
        ["log", "--patch"],
        ["diff", "HEAD~1"],
        ["show", "HEAD"],
        ["log", "-p", "--", "docs/ADR-022.md"],
    ],
)
def test_guard_refuses_diff_args(bad_args: list[str]) -> None:
    with pytest.raises(ValueError):
        _guard_git_args(bad_args)


@pytest.mark.parametrize(
    "ok_args",
    [
        ["log", "--pretty=%s"],
        ["log", "--since=14 days ago", "--pretty=%s"],
        ["log", "--pretty=%s", "--", "docs/ADR-022.md"],
    ],
)
def test_guard_allows_message_level_args(ok_args: list[str]) -> None:
    # Must not raise.
    _guard_git_args(ok_args)


def test_default_git_runner_refuses_diff_args() -> None:
    # The DEFAULT git_runner (the real subprocess one) must also enforce the
    # guard before it would ever shell out — assert the guard fires, no git runs.
    src = ProjectContextSource.from_repo("/tmp/fake-repo", "project_weekly")
    with pytest.raises(ValueError):
        src._git_runner(["show", "HEAD"])  # would emit file content/diff
    with pytest.raises(ValueError):
        src._git_runner(["log", "-p"])


# --------------------------------------------------------------------------- #
# project_feature
# --------------------------------------------------------------------------- #


def test_feature_body_has_decision_number_and_commit() -> None:
    git = _git_runner(
        {
            "docs/ADR-022": (
                "feat(wiki-qa): deterministic code knowledge-graph builder\n"
                "fix(ingest): dedup guard for repo pages\n"
            )
        }
    )
    src = ProjectContextSource.from_repo(
        "/repos/second-brain-wiki",
        "project_feature",
        topic="repo as knowledge source",
        git_runner=git,
        docs_reader=_docs_reader_full(),
    )
    snippets = src.search("ignored")
    assert len(snippets) == 1
    body = snippets[0].body

    # Decision text from the selected ADR.
    assert "deterministic code-graph builder" in body
    # The measured number, verbatim.
    assert "26x faster" in body
    # A real commit subject touching that ADR's file.
    assert "deterministic code knowledge-graph builder" in body


def test_feature_topic_selects_right_adr() -> None:
    # topic="lightweight" should select ADR-025, not ADR-022.
    git = _git_runner({"docs/ADR-025": "feat(post): live content source\n"})
    src = ProjectContextSource.from_repo(
        "/repos/second-brain-wiki",
        "project_feature",
        topic="lightweight post context",
        git_runner=git,
        docs_reader=_docs_reader_full(),
    )
    body = src.search("ignored")[0].body
    assert "live ContentSource" in body
    assert "deterministic code-graph builder" not in body
    # And the commit touching ADR-025 made it in.
    assert "live content source" in body


def test_feature_without_topic_picks_most_recent_adr() -> None:
    # No topic → most recent accepted ADR by date (ADR-025, 2026-06-04).
    git = _git_runner({})
    src = ProjectContextSource.from_repo(
        "/repos/r",
        "project_feature",
        git_runner=git,
        docs_reader=_docs_reader_full(),
    )
    body = src.search("ignored")[0].body
    assert "live ContentSource" in body


# --------------------------------------------------------------------------- #
# project_weekly
# --------------------------------------------------------------------------- #


def test_weekly_lists_real_commit_subjects() -> None:
    git = _git_runner(
        {
            "--since": (
                "feat(wiki-qa): code-graph MCP tools\n"
                "feat(wiki-qa): deterministic code knowledge-graph builder\n"
                "fix(mcp): remove accidentally-merged WIP\n"
            )
        }
    )
    src = ProjectContextSource.from_repo(
        "/repos/r",
        "project_weekly",
        git_runner=git,
        docs_reader=lambda: [],
    )
    body = src.search("ignored")[0].body
    assert "code-graph MCP tools" in body
    assert "deterministic code knowledge-graph builder" in body
    assert "remove accidentally-merged WIP" in body


def test_weekly_empty_window_is_minimal_no_padding() -> None:
    git = _git_runner({"--since": ""})  # no commits in window
    src = ProjectContextSource.from_repo(
        "/repos/r",
        "project_weekly",
        git_runner=git,
        docs_reader=lambda: [],
    )
    snippets = src.search("ignored")
    assert len(snippets) == 1
    body = snippets[0].body
    # Honest minimal snippet — must NOT fabricate commit subjects.
    assert "feat(" not in body
    assert "fix(" not in body
    # It should be short (no padded prose).
    assert len(body) < 400


# --------------------------------------------------------------------------- #
# project_launch
# --------------------------------------------------------------------------- #


def test_launch_body_has_vision_and_accepted_adr() -> None:
    git = _git_runner({})
    src = ProjectContextSource.from_repo(
        "/repos/second-brain-wiki",
        "project_launch",
        git_runner=git,
        docs_reader=_docs_reader_full(),
    )
    body = src.search("ignored")[0].body
    # README vision.
    assert "living, queryable knowledge base" in body
    # Accepted ADR title surfaces.
    assert "Repo as a Knowledge Source" in body
    # Proposed (non-accepted) ADR must NOT appear in a launch.
    assert "A Proposed Thing" not in body
    assert "ADR-099" not in body


def test_launch_title_and_page_path_are_provenance() -> None:
    src = ProjectContextSource.from_repo(
        "/repos/second-brain-wiki",
        "project_launch",
        git_runner=_git_runner({}),
        docs_reader=_docs_reader_full(),
    )
    snippet = src.search("ignored")[0]
    # title = repo basename; page_path = repo path (provenance).
    assert snippet.title == "second-brain-wiki"
    assert snippet.page_path == "/repos/second-brain-wiki"


# --------------------------------------------------------------------------- #
# ContentSource conformance
# --------------------------------------------------------------------------- #


def test_is_content_source_protocol() -> None:
    src = ProjectContextSource.from_repo(
        "/repos/r",
        "project_weekly",
        git_runner=_git_runner({}),
        docs_reader=lambda: [],
    )
    assert isinstance(src, ContentSource)


def test_search_ignores_query_and_categories() -> None:
    git = _git_runner({"--since": "feat(x): a thing\n"})
    src = ProjectContextSource.from_repo(
        "/repos/r",
        "project_weekly",
        git_runner=git,
        docs_reader=lambda: [],
    )
    a = src.search("query one", limit=3, categories=("sources",))
    b = src.search("totally different", limit=1, categories=None)
    assert len(a) == 1 and len(b) == 1
    assert isinstance(a[0], WikiSnippet)
    assert a[0].body == b[0].body


# --------------------------------------------------------------------------- #
# Source URL (issue #193): cite the REAL, verified origin remote
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "https://github.com/DarkCodePE/second-brain-wiki.git\n",
            "https://github.com/DarkCodePE/second-brain-wiki",
        ),
        (
            "https://github.com/DarkCodePE/second-brain-wiki",
            "https://github.com/DarkCodePE/second-brain-wiki",
        ),
        (
            "git@github.com:DarkCodePE/second-brain-wiki.git",
            "https://github.com/DarkCodePE/second-brain-wiki",
        ),
        ("", ""),
        ("not-a-url", ""),
    ],
)
def test_normalize_remote_url(raw: str, expected: str) -> None:
    assert _normalize_remote_url(raw) == expected


def test_feature_body_injects_verified_origin_url() -> None:
    git = _git_runner(
        {
            "remote get-url origin": "https://github.com/DarkCodePE/second-brain-wiki.git\n",
            "docs/ADR-025": "feat(post): live content source\n",
        }
    )
    src = ProjectContextSource.from_repo(
        "/repos/second-brain-wiki",
        "project_feature",
        topic="lightweight",
        git_runner=git,
        docs_reader=_docs_reader_full(),
    )
    body = src.search("ignored")[0].body
    # The real, scheme-qualified repo URL is present (no fabricated author URL).
    assert "https://github.com/DarkCodePE/second-brain-wiki" in body
    assert "CoherenceIA" not in body


def test_no_origin_remote_injects_no_url() -> None:
    # git_runner returns "" for every arg → no origin → no fabricated URL.
    git = _git_runner({})
    src = ProjectContextSource.from_repo(
        "/repos/r",
        "project_weekly",
        git_runner=git,
        docs_reader=lambda: [],
    )
    body = src.search("ignored")[0].body
    assert "http" not in body
    assert "## Fuentes" not in body


def test_invalid_sub_type_raises() -> None:
    with pytest.raises(ValueError):
        ProjectContextSource.from_repo(
            "/repos/r",
            "not_a_project_type",
            git_runner=_git_runner({}),
            docs_reader=lambda: [],
        )
