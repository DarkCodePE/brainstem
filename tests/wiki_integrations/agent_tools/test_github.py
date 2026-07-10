"""
Tests for `GitHubIntegration`: list/get/search shape, normalisation, dedup,
substring search.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wiki_integrations.agent_tools.github import GitHubIntegration

from .conftest import FakeBridge

GITHUB_PAYLOAD = [
    {
        "id": 100,
        "kind": "issue",
        "number": 100,
        "title": "Bug: memory tree fails on empty file",
        "body": "Repro: ingest an empty markdown file…",
        "html_url": "https://github.com/example/repo/issues/100",
        "updated_at": "2026-05-20T10:00:00Z",
        "state": "open",
        "repo": "example/repo",
    },
    {
        "id": 101,
        "kind": "pull_request",
        "number": 101,
        "title": "feat: add token compression",
        "body": "TokenJuice port substrate.",
        "html_url": "https://github.com/example/repo/pull/101",
        "updated_at": "2026-05-22T15:30:00Z",
        "state": "merged",
    },
    {
        "id": 102,
        "kind": "issue",
        "number": 102,
        "title": "RFC: hide internal endpoints",
        "body": "We should consider…",
        "html_url": "https://github.com/example/repo/issues/102",
        "updated_at": "2026-05-25T09:15:00Z",
        "state": "open",
    },
]


def _make(secret_store, audit_jsonl, audit_md):
    bridge = FakeBridge(payloads={"github": GITHUB_PAYLOAD})
    gh = GitHubIntegration(
        bridge=bridge, store=secret_store, audit_jsonl=audit_jsonl, audit_md=audit_md
    )
    return gh, bridge


@pytest.mark.asyncio
async def test_list_returns_items_with_normalised_shape(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    await gh.connect()
    items = await gh.list(limit=10)
    assert len(items) == 3
    assert {i.id for i in items} == {"100", "101", "102"}
    assert all(i.uri.startswith("https://github.com") for i in items)
    # Snippets bounded
    assert all(len(i.snippet) <= 500 for i in items)
    # Metadata exposes kind + state + sha256
    one = next(i for i in items if i.id == "101")
    assert one.metadata["kind"] == "pull_request"
    assert one.metadata["state"] == "merged"
    assert isinstance(one.metadata["sha256"], str)
    assert len(one.metadata["sha256"]) == 64


@pytest.mark.asyncio
async def test_list_respects_since_filter(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    await gh.connect()
    items = await gh.list(since=datetime(2026, 5, 22, tzinfo=UTC))
    assert {i.id for i in items} == {"101", "102"}


@pytest.mark.asyncio
async def test_list_respects_limit(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    await gh.connect()
    items = await gh.list(limit=1)
    assert len(items) == 1


@pytest.mark.asyncio
async def test_get_returns_one_by_id(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    await gh.connect()
    item = await gh.get("101")
    assert item.id == "101"
    assert "compression" in item.title


@pytest.mark.asyncio
async def test_get_raises_keyerror_when_missing(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    await gh.connect()
    with pytest.raises(KeyError):
        await gh.get("99999")


@pytest.mark.asyncio
async def test_search_substring_match(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    await gh.connect()
    result = await gh.search("memory tree")
    assert len(result.items) == 1
    assert result.items[0].id == "100"


@pytest.mark.asyncio
async def test_search_empty_query_rejected(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    await gh.connect()
    with pytest.raises(ValueError, match="non-empty"):
        await gh.search("")


@pytest.mark.asyncio
async def test_search_is_case_insensitive(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    await gh.connect()
    result = await gh.search("MEMORY TREE")
    assert len(result.items) == 1
