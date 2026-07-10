"""
Tests for `wiki_core.secrets.disconnect` — atomic revocation contract per
ADR-017 §Revocation contract.
"""

from __future__ import annotations

import json

import pytest

from wiki_core.secrets import SecretMeta, disconnect


def _populate_gmail(store):
    """Seed a typical Phase-1 connection + a Phase-3 OAuth pair under gmail."""
    store.set(
        "composio.connection.gmail",
        "conn_abc123",
        SecretMeta(
            kind="connection_id",
            provider="gmail",
            scope=("https://www.googleapis.com/auth/gmail.readonly",),
            granted_at="2026-05-26T10:00:00Z",
        ),
    )
    store.set(
        "oauth.gmail.access_token",
        "ya29.testbearer",
        SecretMeta(
            kind="oauth_access",
            provider="gmail",
            scope=("https://www.googleapis.com/auth/gmail.readonly",),
            expires_at="2026-05-26T11:00:00Z",
            refresh_token_name="oauth.gmail.refresh_token",
        ),
    )
    store.set(
        "oauth.gmail.refresh_token",
        "1//refreshtoken",
        SecretMeta(kind="oauth_refresh", provider="gmail"),
    )


def _read_audit(audit):
    return [json.loads(line) for line in audit.path.read_text().splitlines() if line.strip()]


def test_disconnect_clears_all_provider_entries(store, audit_log_tmp):
    _populate_gmail(store)
    revoked_args = []

    def fake_revoker(provider, connection_id):
        revoked_args.append((provider, connection_id))

    result = disconnect("gmail", store=store, audit=audit_log_tmp, upstream_revoker=fake_revoker)

    assert result.upstream_revoked is True
    assert result.local_cleared is True
    assert result.remaining_names == ()
    assert revoked_args == [("gmail", "conn_abc123")]

    # Everything is gone
    assert store.get("composio.connection.gmail") is None
    assert store.get("oauth.gmail.access_token") is None
    assert store.get("oauth.gmail.refresh_token") is None


def test_disconnect_writes_revoked_event(store, audit_log_tmp):
    _populate_gmail(store)
    disconnect("gmail", store=store, audit=audit_log_tmp, upstream_revoker=lambda *_: None)
    events = _read_audit(audit_log_tmp)
    revoked = [e for e in events if e["event"] == "revoked"]
    assert len(revoked) == 1
    assert revoked[0]["provider"] == "gmail"
    assert revoked[0]["result"] == "ok"
    assert revoked[0]["params_redacted"]["upstream_revoked"] is True
    assert revoked[0]["params_redacted"]["names_cleared"] == 3


def test_upstream_failure_preserves_local_state(store, audit_log_tmp):
    _populate_gmail(store)

    def failing_revoker(provider, connection_id):
        raise RuntimeError("composio 503")

    result = disconnect(
        "gmail",
        store=store,
        audit=audit_log_tmp,
        upstream_revoker=failing_revoker,
    )

    assert result.upstream_revoked is False
    assert result.local_cleared is False
    # Local secrets untouched so user can retry
    assert store.get("composio.connection.gmail") == "conn_abc123"
    assert store.get("oauth.gmail.access_token") == "ya29.testbearer"

    events = _read_audit(audit_log_tmp)
    assert events[-1]["event"] == "revoked"
    assert events[-1]["result"] == "upstream_failed"


def test_disconnect_without_upstream_revoker_clears_locally(store, audit_log_tmp):
    """Phase-3 path: SBW holds the bearer, no upstream call."""
    _populate_gmail(store)
    result = disconnect("gmail", store=store, audit=audit_log_tmp, upstream_revoker=None)
    assert result.upstream_revoked is False  # not applicable
    assert result.local_cleared is True
    assert store.get("oauth.gmail.refresh_token") is None


def test_disconnect_unknown_provider_is_noop(store, audit_log_tmp):
    """Disconnecting a provider with no stored secrets must succeed cleanly."""
    result = disconnect(
        "notion", store=store, audit=audit_log_tmp, upstream_revoker=lambda *_: None
    )
    assert result.local_cleared is True
    assert result.remaining_names == ()
    events = _read_audit(audit_log_tmp)
    assert events[-1]["params_redacted"]["names_cleared"] == 0


def test_disconnect_isolates_other_providers(store, audit_log_tmp):
    """Disconnecting gmail must not touch github entries."""
    _populate_gmail(store)
    store.set(
        "composio.connection.github",
        "conn_github_xyz",
        SecretMeta(kind="connection_id", provider="github"),
    )

    disconnect(
        "gmail",
        store=store,
        audit=audit_log_tmp,
        upstream_revoker=lambda *_: None,
    )

    assert store.get("composio.connection.github") == "conn_github_xyz"


@pytest.mark.parametrize(
    "provider,name,kind",
    [
        ("github", "oauth.github.access_token", "oauth_access"),
        ("github", "oauth.github.refresh_token", "oauth_refresh"),
    ],
)
def test_refresh_token_rotation_replacement(store, provider, name, kind):
    """Setting a new token under the same name replaces the old one atomically.

    Stand-in for ADR-017's 'refresh token rotation' contract: the integration
    code calls `store.set(name, new_value, meta)` on rotation, and the old
    value is unreadable immediately.
    """
    old = SecretMeta(kind=kind, provider=provider)
    store.set(name, "old_token", old)

    new = SecretMeta(kind=kind, provider=provider, expires_at="2026-05-27T00:00:00Z")
    store.set(name, "new_token_rotated", new)

    assert store.get(name) == "new_token_rotated"
    out = store.get_meta(name)
    assert out is not None
    assert out.expires_at == "2026-05-27T00:00:00Z"
