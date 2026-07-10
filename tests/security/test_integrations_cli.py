"""
Smoke tests for `sbw integrations {list,revoke,audit}`.

We import the dispatcher directly to avoid spawning subprocesses; the
in-memory keyring backend is wired by the `store` fixture so the OS
keychain is never touched.
"""

from __future__ import annotations

import argparse

import pytest

from wiki_agent.integrations_cli import run_integrations_cli
from wiki_core.secrets import SecretMeta


@pytest.fixture
def populated_store(store):
    store.set(
        "composio.connection.gmail",
        "conn_abc",
        SecretMeta(
            kind="connection_id",
            provider="gmail",
            scope=("https://www.googleapis.com/auth/gmail.readonly",),
        ),
    )
    return store


def test_list_prints_table(populated_store, monkeypatch, capsys):
    # Force `_open_store` to use the fixture-bound singleton.
    from wiki_agent import integrations_cli

    monkeypatch.setattr(integrations_cli, "_open_store", lambda: populated_store)

    # Force the bridge into the "unavailable" branch so the legacy
    # keyring-only display fires. We're asserting the legacy fallback
    # path here; the live-status path has its own focused tests below.
    from wiki_integrations import composio_bridge as _bridge_mod

    def _explode(*args, **kwargs):
        raise RuntimeError("simulated bridge unavailable in unit test")

    monkeypatch.setattr(_bridge_mod, "ComposioBridge", _explode)

    rc = run_integrations_cli(argparse.Namespace(integrations_action="list"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "gmail" in out
    assert "github" in out
    assert "yes" in out  # gmail in keyring + bridge unavailable → legacy "yes"
    assert "no" in out  # others have no keyring entry


def test_list_surfaces_live_upstream_status(populated_store, monkeypatch, capsys):
    """Issue #107: ``sbw integrations list`` must reflect Composio's
    real status, not stale keyring presence. Gmail's keyring row says
    "connected was attempted at some point"; Composio says it's
    expired. The CLI must surface ``expired``."""
    from wiki_agent import integrations_cli
    from wiki_integrations import composio_bridge as _bridge_mod
    from wiki_integrations.composio_bridge import ComposioConnection

    monkeypatch.setattr(integrations_cli, "_open_store", lambda: populated_store)

    class _FakeBridge:
        async def list_connections(self):
            return [
                ComposioConnection(
                    provider="gmail",
                    connection_id="conn_abc",
                    status="expired",
                    metadata={},
                ),
                ComposioConnection(
                    provider="github",
                    connection_id="conn_gh",
                    status="initializing",
                    metadata={},
                ),
            ]

    monkeypatch.setattr(_bridge_mod, "ComposioBridge", lambda *a, **kw: _FakeBridge())

    rc = run_integrations_cli(argparse.Namespace(integrations_action="list"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "expired" in out
    assert "initializing" in out
    # The literal stale "yes" must NOT appear for gmail's row.
    gmail_line = next(line for line in out.splitlines() if line.startswith("gmail"))
    assert "yes" not in gmail_line


def test_list_marks_uninstalled_provider_as_no(populated_store, monkeypatch, capsys):
    """When Composio returns no row for a provider, it's "no" — even
    if the keyring is stale from an old install."""
    from wiki_agent import integrations_cli
    from wiki_integrations import composio_bridge as _bridge_mod

    monkeypatch.setattr(integrations_cli, "_open_store", lambda: populated_store)

    class _EmptyBridge:
        async def list_connections(self):
            return []

    monkeypatch.setattr(_bridge_mod, "ComposioBridge", lambda *a, **kw: _EmptyBridge())

    rc = run_integrations_cli(argparse.Namespace(integrations_action="list"))
    assert rc == 0
    out = capsys.readouterr().out
    gmail_line = next(line for line in out.splitlines() if line.startswith("gmail"))
    # Even though keyring has gmail, upstream says no row → "no".
    assert " no " in gmail_line or gmail_line.split()[1] == "no"


def test_search_empty_query_falls_back_to_list(populated_store, monkeypatch, capsys):
    """Issue #108: empty / whitespace-only query routes to
    ``IIntegration.list()`` so the user gets the latest window
    instead of an error."""
    from datetime import UTC, datetime

    from wiki_agent import integrations_cli
    from wiki_core.integrations.protocol import IntegrationItem

    monkeypatch.setattr(integrations_cli, "_open_store", lambda: populated_store)

    list_calls: list[dict] = []
    search_calls: list[dict] = []

    class _FakeIntegration:
        async def list(self, *, since=None, limit=50):
            list_calls.append({"since": since, "limit": limit})
            return (
                IntegrationItem(
                    id="evt-1",
                    title="Standup",
                    snippet="",
                    uri="https://calendar/evt-1",
                    updated_at=datetime.now(UTC),
                ),
            )

        async def search(self, query, *, limit=50, cursor=None):
            search_calls.append({"query": query, "limit": limit})
            raise AssertionError("search must not be called for empty query")

    monkeypatch.setattr(
        integrations_cli, "_build_integration", lambda provider, store: _FakeIntegration()
    )

    rc = run_integrations_cli(
        argparse.Namespace(
            integrations_action="search",
            provider="calendar",
            query="   ",
            limit=10,
        )
    )
    assert rc == 0
    assert list_calls == [{"since": None, "limit": 10}]
    assert search_calls == []
    out = capsys.readouterr().out
    assert "recent items" in out
    assert "Standup" in out


def test_search_real_query_still_calls_search(populated_store, monkeypatch, capsys):
    """Non-empty query continues to call ``search()`` (regression guard
    on the #108 fix)."""
    from datetime import UTC, datetime

    from wiki_agent import integrations_cli
    from wiki_core.integrations.protocol import IntegrationItem, SearchResult

    monkeypatch.setattr(integrations_cli, "_open_store", lambda: populated_store)

    seen_queries: list[str] = []

    class _FakeIntegration:
        async def list(self, *, since=None, limit=50):
            raise AssertionError("list must not be called for non-empty query")

        async def search(self, query, *, limit=50, cursor=None):
            seen_queries.append(query)
            return SearchResult(
                items=(
                    IntegrationItem(
                        id="i-1",
                        title="found-it",
                        snippet="",
                        uri="https://x/i-1",
                        updated_at=datetime.now(UTC),
                    ),
                ),
            )

    monkeypatch.setattr(
        integrations_cli, "_build_integration", lambda provider, store: _FakeIntegration()
    )

    rc = run_integrations_cli(
        argparse.Namespace(
            integrations_action="search",
            provider="calendar",
            query="standup",
            limit=10,
        )
    )
    assert rc == 0
    assert seen_queries == ["standup"]
    out = capsys.readouterr().out
    assert "found-it" in out


def test_revoke_unknown_provider(capsys):
    rc = run_integrations_cli(argparse.Namespace(integrations_action="revoke", provider="twitter"))
    assert rc == 1
    assert "Unknown provider" in capsys.readouterr().err


def test_revoke_known_provider_clears_local(populated_store, monkeypatch, tmp_path, capsys):
    from wiki_agent import integrations_cli
    from wiki_core.secrets import AuditLog

    monkeypatch.setattr(integrations_cli, "_open_store", lambda: populated_store)
    # Redirect the audit log into tmp_path so we don't pollute ~/.sbw
    audit_path = tmp_path / "integrations.log.jsonl"
    monkeypatch.setattr(
        integrations_cli, "AuditLog", lambda: AuditLog(path=audit_path), raising=False
    )
    # The CLI imports `AuditLog` locally inside `_cmd_revoke`, so we also
    # need to patch the canonical symbol the import resolves to.
    import wiki_core.secrets as secrets_mod

    monkeypatch.setattr(secrets_mod, "AuditLog", lambda: AuditLog(path=audit_path))

    rc = run_integrations_cli(argparse.Namespace(integrations_action="revoke", provider="gmail"))
    assert rc == 0
    assert populated_store.get("composio.connection.gmail") is None
    assert audit_path.exists()
    captured = capsys.readouterr()
    assert "cleared all local secrets for gmail" in captured.out


def test_audit_no_log_yet(monkeypatch, tmp_path, capsys):
    from wiki_core.secrets import AuditLog

    missing = tmp_path / "nothing.jsonl"
    import wiki_core.secrets as secrets_mod

    monkeypatch.setattr(secrets_mod, "AuditLog", lambda: AuditLog(path=missing))
    # Delete the file the AuditLog constructor would touch via mkdir+nothing
    # else; the path doesn't exist so the branch is exercised.
    missing.unlink(missing_ok=True)

    rc = run_integrations_cli(argparse.Namespace(integrations_action="audit"))
    assert rc == 0
    assert "No audit log" in capsys.readouterr().out
