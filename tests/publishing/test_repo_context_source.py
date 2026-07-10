"""Tests for the lightweight post-context path (ADR-025).

Hermetic / mock-first per CLAUDE.md: a fake fetcher returns a canned
:class:`~wiki_repos.github_meta.RepoMeta`, and ``fetch_repo_meta`` is exercised
with an injected ``opener`` ``(url) -> (status, bytes)``. NO network access.
"""

from __future__ import annotations

import base64
import json

import pytest

from wiki_publishing import RepoContextSource
from wiki_publishing.linkedin_draft import ContentSource, WikiSnippet
from wiki_repos.errors import InvalidUrl
from wiki_repos.github_meta import RepoMeta, RepoMetaError, fetch_repo_meta

# --------------------------------------------------------------------------- #
# Canned RepoMeta + fake fetcher                                              #
# --------------------------------------------------------------------------- #

_README = (
    "# turbovec\n\nA blazing-fast vector store.\n\n"
    "## Benchmarks\n\n- 12,500x faster search than baseline\n"
)

_META = RepoMeta(
    owner="acme",
    repo="turbovec",
    description="A blazing-fast embedded vector database for RAG.",
    topics=("vector-database", "rag", "embeddings"),
    stars=4242,
    homepage="https://turbovec.dev",
    license="MIT",
    language="Rust",
    readme=_README,
)


def _fake_fetcher(owner: str, repo: str) -> RepoMeta:
    return _META


def _raising_fetcher(owner: str, repo: str) -> RepoMeta:
    raise RepoMetaError(f"repo {owner}/{repo} not found or unreachable")


# --------------------------------------------------------------------------- #
# RepoContextSource — snippet composition                                     #
# --------------------------------------------------------------------------- #


def test_search_returns_single_snippet_with_value_prop_and_facts() -> None:
    src = RepoContextSource("acme", "turbovec", fetcher=_fake_fetcher)
    snippets = src.search("anything")

    assert len(snippets) == 1
    snip = snippets[0]
    assert isinstance(snip, WikiSnippet)
    assert snip.title == "acme/turbovec"
    # page_path is the canonical URL so the drafter's URL-citation picks it up.
    assert snip.page_path == "https://github.com/acme/turbovec"

    body = snip.body
    # Description = the value prop.
    assert "A blazing-fast embedded vector database for RAG." in body
    # Topics.
    assert "vector-database" in body and "rag" in body and "embeddings" in body
    # README excerpt (use-focus posts need features/benchmarks).
    assert "12,500x faster search" in body
    # Canonical URL also in the body.
    assert "https://github.com/acme/turbovec" in body
    # Other facts surfaced.
    assert "4242" in body
    assert "MIT" in body
    assert "Rust" in body
    assert "https://turbovec.dev" in body


def test_search_ignores_query_and_categories() -> None:
    src = RepoContextSource("acme", "turbovec", fetcher=_fake_fetcher)
    a = src.search("totally different query", limit=1, categories=("sources",))
    b = src.search("", categories=None)
    assert a == b
    assert len(a) == 1


def test_fetcher_called_once_and_cached() -> None:
    calls: list[tuple[str, str]] = []

    def counting(owner: str, repo: str) -> RepoMeta:
        calls.append((owner, repo))
        return _META

    src = RepoContextSource("acme", "turbovec", fetcher=counting)
    src.search("x")
    src.search("y")
    assert calls == [("acme", "turbovec")]


def test_satisfies_content_source_protocol() -> None:
    src = RepoContextSource("acme", "turbovec", fetcher=_fake_fetcher)
    assert isinstance(src, ContentSource)


# --------------------------------------------------------------------------- #
# RepoContextSource.from_url                                                   #
# --------------------------------------------------------------------------- #


def test_from_url_parses_owner_repo() -> None:
    src = RepoContextSource.from_url("https://github.com/acme/turbovec", fetcher=_fake_fetcher)
    snip = src.search("x")[0]
    assert snip.title == "acme/turbovec"


def test_from_url_rejects_non_github_url() -> None:
    with pytest.raises(InvalidUrl):
        RepoContextSource.from_url("https://evil.example.com/acme/turbovec")


def test_repo_meta_error_propagates() -> None:
    src = RepoContextSource("ghost", "nope", fetcher=_raising_fetcher)
    with pytest.raises(RepoMetaError):
        src.search("x")


# --------------------------------------------------------------------------- #
# fetch_repo_meta — injected opener (no network)                              #
# --------------------------------------------------------------------------- #


def _make_opener(responses: dict[str, tuple[int, bytes]]):
    """Build an opener that maps a URL suffix to a canned (status, bytes)."""

    def opener(url: str) -> tuple[int, bytes]:
        for suffix, resp in responses.items():
            if url.endswith(suffix):
                return resp
        return (404, b"")

    return opener


def test_fetch_repo_meta_happy_path() -> None:
    repo_json = json.dumps(
        {
            "description": "A blazing-fast embedded vector database for RAG.",
            "topics": ["vector-database", "rag"],
            "stargazers_count": 4242,
            "homepage": "https://turbovec.dev",
            "license": {"spdx_id": "MIT"},
            "language": "Rust",
        }
    ).encode()
    readme_json = json.dumps({"content": base64.b64encode(_README.encode()).decode()}).encode()
    opener = _make_opener(
        {
            "/repos/acme/turbovec/readme": (200, readme_json),
            "/repos/acme/turbovec": (200, repo_json),
        }
    )

    meta = fetch_repo_meta("acme", "turbovec", opener=opener)

    assert meta.owner == "acme"
    assert meta.repo == "turbovec"
    assert meta.description == "A blazing-fast embedded vector database for RAG."
    assert meta.topics == ("vector-database", "rag")
    assert meta.stars == 4242
    assert meta.homepage == "https://turbovec.dev"
    assert meta.license == "MIT"
    assert meta.language == "Rust"
    assert "12,500x faster search" in meta.readme
    assert meta.canonical_url == "https://github.com/acme/turbovec"


def test_fetch_repo_meta_404_raises() -> None:
    opener = _make_opener({"/repos/acme/missing": (404, b"")})
    with pytest.raises(RepoMetaError):
        fetch_repo_meta("acme", "missing", opener=opener)


def test_fetch_repo_meta_missing_readme_yields_empty() -> None:
    repo_json = json.dumps({"description": "desc", "language": "Python"}).encode()
    # README endpoint 404s; repo endpoint OK.
    opener = _make_opener(
        {"/repos/acme/noreadme/readme": (404, b""), "/repos/acme/noreadme": (200, repo_json)}
    )
    meta = fetch_repo_meta("acme", "noreadme", opener=opener)
    assert meta.description == "desc"
    assert meta.language == "Python"
    assert meta.readme == ""
