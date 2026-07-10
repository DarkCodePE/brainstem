"""
Tests for `ComposioBackedIntegration` lifecycle: connect, disconnect, health,
NotConnectedError on uncalled list/get/search, audit-log writes.
"""

from __future__ import annotations

import json

import pytest

from wiki_core.integrations.protocol import NotConnectedError
from wiki_integrations.agent_tools.github import GitHubIntegration

from .conftest import FakeBridge


def _make(secret_store, audit_jsonl, audit_md, payloads=None, connected=None):
    bridge = FakeBridge(
        payloads=payloads or {"github": []},
        already_connected=connected or [],
    )
    gh = GitHubIntegration(
        bridge=bridge,
        store=secret_store,
        audit_jsonl=audit_jsonl,
        audit_md=audit_md,
    )
    return gh, bridge


def _read_jsonl(audit_log_path):
    return [json.loads(line) for line in audit_log_path.read_text().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_connect_writes_secret_and_returns_connected(secret_store, audit_jsonl, audit_md):
    gh, bridge = _make(secret_store, audit_jsonl, audit_md)
    result = await gh.connect()
    assert result.provider == "github"
    assert result.connection_id == "new-github"
    assert secret_store.get("composio.connection.github") == "new-github"
    meta = secret_store.get_meta("composio.connection.github")
    assert meta is not None
    assert meta.scope == ("repo:status", "read:user", "read:org")


@pytest.mark.asyncio
async def test_connect_idempotent(secret_store, audit_jsonl, audit_md):
    gh, bridge = _make(secret_store, audit_jsonl, audit_md)
    a = await gh.connect()
    b = await gh.connect()
    assert a.connection_id == b.connection_id
    # Bridge should be hit exactly once
    assert bridge.connect_calls == ["github"]


@pytest.mark.asyncio
async def test_list_requires_connection(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    with pytest.raises(NotConnectedError):
        await gh.list()


@pytest.mark.asyncio
async def test_search_requires_connection(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    with pytest.raises(NotConnectedError):
        await gh.search("foo")


@pytest.mark.asyncio
async def test_disconnect_clears_secret_and_logs(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md)
    await gh.connect()
    assert secret_store.get("composio.connection.github") == "new-github"

    await gh.disconnect()

    assert secret_store.get("composio.connection.github") is None
    events = _read_jsonl(audit_jsonl.path)
    assert any(e["event"] == "revoked" and e["provider"] == "github" for e in events)
    md_lines = audit_md.path.read_text(encoding="utf-8")
    assert "disconnect" in md_lines
    assert "ok" in md_lines


@pytest.mark.asyncio
async def test_health_returns_true_when_connected_upstream(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md, connected=["github"])
    assert await gh.health() is True


@pytest.mark.asyncio
async def test_health_returns_false_when_not_in_active_list(secret_store, audit_jsonl, audit_md):
    gh, _ = _make(secret_store, audit_jsonl, audit_md, connected=[])
    assert await gh.health() is False


@pytest.mark.asyncio
async def test_subclass_must_set_provider(secret_store, audit_jsonl, audit_md):
    from wiki_integrations.agent_tools.base import ComposioBackedIntegration

    class _Bad(ComposioBackedIntegration):
        # Deliberately no PROVIDER override
        def _to_item(self, raw):
            raise NotImplementedError

        async def get(self, item_id):
            raise NotImplementedError

        async def search(self, query, *, limit=50, cursor=None):
            raise NotImplementedError

    bridge = FakeBridge()
    with pytest.raises(TypeError, match="must set PROVIDER"):
        _Bad(bridge=bridge, store=secret_store, audit_jsonl=audit_jsonl, audit_md=audit_md)
