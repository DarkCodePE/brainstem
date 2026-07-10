"""
Tests for `wiki_integrations.providers.github.GitHubIntegrationSource`.

Coverage matrix:

| Behaviour                                  | Test                                  |
| ------------------------------------------ | ------------------------------------- |
| name() == "github"                         | test_name_is_github                   |
| source field == "github"                   | test_event_source_is_github           |
| path_or_uri is the html_url                | test_event_path_or_uri_is_html_url    |
| sha256 matches title+body                  | test_event_sha256_matches_title_body  |
| Required metadata: bucket = github-issues  | test_event_bucket_is_github_issues    |
| Required metadata: rel_path is kind/number | test_event_rel_path_format            |
| Required metadata: event_type = created    | test_event_type_is_created            |
| Required metadata: mtime is updated_at     | test_event_mtime_is_updated_at        |
| Required metadata: size matches payload    | test_event_size_matches_payload       |
| Optional metadata: mime is text/markdown   | test_event_mime_is_text_markdown      |
| Issues + PRs are both translated           | test_handles_issue_and_pr             |
| Unknown kind logged and skipped            | test_unknown_kind_skipped             |
| Walker called with provider="github"       | test_walker_invoked_with_github       |
| `wiki_core.IngestSource` shape conformance | test_source_satisfies_protocol        |
| fetch_batch returns same list as callback  | test_batch_list_matches_callback      |
| Empty walk yields empty batch              | test_fetch_batch_empty                |
| Lifecycle (start/stop)                     | test_lifecycle                        |
"""

from __future__ import annotations

import hashlib

import pytest

from wiki_core.protocols import IngestSource
from wiki_integrations.providers.github import GitHubIntegrationSource

_ISSUE = {
    "id": "gh-issue-001",
    "kind": "issue",
    "number": 42,
    "title": "Track PRD-005 substrate work",
    "body": "Issue body covering the substrate ship.",
    "html_url": "https://github.com/DarkCodePE/second-brain-wiki/issues/42",
    "updated_at": "2026-05-22T07:00:00Z",
    "state": "open",
}
_PR = {
    "id": "gh-pr-001",
    "kind": "pull_request",
    "number": 124,
    "title": "feat(m3): integrations foundation",
    "body": "PR body explaining the change.",
    "html_url": "https://github.com/DarkCodePE/second-brain-wiki/pull/124",
    "updated_at": "2026-05-22T12:00:00Z",
    "state": "open",
}


@pytest.fixture
def github_source(recording_callback, fake_walker_factory, fetch_window):
    walker = fake_walker_factory({"github": [_ISSUE, _PR]})
    return GitHubIntegrationSource(
        on_event=recording_callback,
        walker=walker,
        fetch_window=fetch_window,
    ), walker


def test_name_is_github(github_source) -> None:
    src, _ = github_source
    assert src.name() == "github"


@pytest.mark.asyncio
async def test_event_source_is_github(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    assert all(e.source == "github" for e in events)


@pytest.mark.asyncio
async def test_event_path_or_uri_is_html_url(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    urls = {e.path_or_uri for e in events}
    assert urls == {_ISSUE["html_url"], _PR["html_url"]}


@pytest.mark.asyncio
async def test_event_sha256_matches_title_body(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    by_url = {e.path_or_uri: e for e in events}
    issue_sha = hashlib.sha256(f"{_ISSUE['title']}\n{_ISSUE['body']}".encode()).hexdigest()
    pr_sha = hashlib.sha256(f"{_PR['title']}\n{_PR['body']}".encode()).hexdigest()
    assert by_url[_ISSUE["html_url"]].sha256 == issue_sha
    assert by_url[_PR["html_url"]].sha256 == pr_sha


@pytest.mark.asyncio
async def test_event_bucket_is_github_issues(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    assert all(e.metadata["bucket"] == "github-issues" for e in events)


@pytest.mark.asyncio
async def test_event_rel_path_format(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    rels = {e.metadata["rel_path"] for e in events}
    assert rels == {"issue/42", "pull_request/124"}


@pytest.mark.asyncio
async def test_event_type_is_created(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    assert all(e.metadata["event_type"] == "created" for e in events)


@pytest.mark.asyncio
async def test_event_mtime_is_updated_at(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    mtimes = {e.metadata["mtime"] for e in events}
    assert mtimes == {_ISSUE["updated_at"], _PR["updated_at"]}


@pytest.mark.asyncio
async def test_event_size_matches_payload(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    sizes = {e.metadata["size"] for e in events}
    expected = {
        len(f"{_ISSUE['title']}\n{_ISSUE['body']}".encode()),
        len(f"{_PR['title']}\n{_PR['body']}".encode()),
    }
    assert sizes == expected


@pytest.mark.asyncio
async def test_event_mime_is_text_markdown(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    assert all(e.metadata["mime"] == "text/markdown" for e in events)


@pytest.mark.asyncio
async def test_handles_issue_and_pr(github_source) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    kinds = {e.metadata["kind"] for e in events}
    assert kinds == {"issue", "pull_request"}


@pytest.mark.asyncio
async def test_handles_repository_shape(
    recording_callback, fake_walker_factory, fetch_window
) -> None:
    # #151: GITHUB_LIST_REPOSITORIES_FOR_AUTHENTICATED_USER returns repo
    # payloads (full_name/description, no number/kind) — inferred as repository.
    repo = {
        "id": "gh-repo-1",
        "full_name": "DarkCodePE/second-brain-wiki",
        "description": "AI-first knowledge engine",
        "html_url": "https://github.com/DarkCodePE/second-brain-wiki",
        "updated_at": "2026-06-01T00:00:00Z",
    }
    src = GitHubIntegrationSource(
        on_event=recording_callback,
        walker=fake_walker_factory({"github": [repo]}),
        fetch_window=fetch_window,
    )
    events = await src.fetch_batch()
    assert len(events) == 1
    e = events[0]
    assert e.metadata["kind"] == "repository"
    assert e.metadata["bucket"] == "github-repos"
    assert e.metadata["rel_path"] == "repo/DarkCodePE--second-brain-wiki"
    assert e.path_or_uri == "https://github.com/DarkCodePE/second-brain-wiki"


@pytest.mark.asyncio
async def test_unknown_kind_skipped(recording_callback, fake_walker_factory, fetch_window) -> None:
    bad = dict(_ISSUE, kind="wiki-page")
    walker = fake_walker_factory({"github": [bad, _PR]})
    src = GitHubIntegrationSource(
        on_event=recording_callback, walker=walker, fetch_window=fetch_window
    )
    events = await src.fetch_batch()
    # Only the PR survives; the unknown kind is skipped, not raised.
    assert len(events) == 1
    assert events[0].metadata["kind"] == "pull_request"


@pytest.mark.asyncio
async def test_walker_invoked_with_github(github_source) -> None:
    src, walker = github_source
    await src.fetch_batch()
    assert walker.walked == ["github"]


def test_source_satisfies_protocol(github_source) -> None:
    src, _ = github_source
    assert isinstance(src, IngestSource)


@pytest.mark.asyncio
async def test_batch_list_matches_callback(github_source, recording_callback) -> None:
    src, _ = github_source
    events = await src.fetch_batch()
    assert recording_callback.events == events


@pytest.mark.asyncio
async def test_fetch_batch_empty(recording_callback, fake_walker_factory, fetch_window) -> None:
    walker = fake_walker_factory({"github": []})
    src = GitHubIntegrationSource(
        on_event=recording_callback, walker=walker, fetch_window=fetch_window
    )
    events = await src.fetch_batch()
    assert events == []
    assert recording_callback.events == []


@pytest.mark.asyncio
async def test_lifecycle(github_source) -> None:
    src, _ = github_source
    assert src.started is False
    await src.start()
    assert src.started is True
    await src.stop()
    assert src.started is False
