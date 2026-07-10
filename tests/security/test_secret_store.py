"""
Tests for `wiki_core.secrets.KeyringSecretStore` against the in-memory backend.
"""

from __future__ import annotations

import pytest

from wiki_core.secrets import SecretMeta, SecretStoreUnavailable


def test_set_and_get_roundtrip(store):
    meta = SecretMeta(kind="api_key", provider="composio")
    store.set("composio.api_key", "comp_test_xyz", meta)
    assert store.get("composio.api_key") == "comp_test_xyz"


def test_meta_roundtrip_preserves_scope_tuple(store):
    meta = SecretMeta(
        kind="connection_id",
        provider="gmail",
        scope=("https://www.googleapis.com/auth/gmail.readonly",),
        granted_at="2026-05-26T15:00:00Z",
    )
    store.set("composio.connection.gmail", "conn_abc123", meta)

    out = store.get_meta("composio.connection.gmail")
    assert out == meta
    assert isinstance(out.scope, tuple)  # must round-trip as tuple


def test_get_missing_returns_none(store):
    assert store.get("never.set") is None
    assert store.get_meta("never.set") is None


def test_delete_existing_returns_true(store):
    store.set("x", "v", SecretMeta(kind="api_key", provider="composio"))
    assert store.delete("x") is True
    assert store.get("x") is None


def test_delete_missing_returns_false(store):
    assert store.delete("never.set") is False


def test_overwrite_replaces_value_and_meta(store):
    m1 = SecretMeta(kind="oauth_access", provider="github", scope=("repo:status",))
    m2 = SecretMeta(kind="oauth_access", provider="github", scope=("repo:status", "read:user"))
    store.set("oauth.github.access_token", "ghp_old", m1)
    store.set("oauth.github.access_token", "ghp_new", m2)
    assert store.get("oauth.github.access_token") == "ghp_new"
    assert store.get_meta("oauth.github.access_token") == m2


def test_list_names_prefix_filter(store):
    m = SecretMeta(kind="api_key", provider="composio")
    store.set("composio.api_key", "v1", m)
    store.set("composio.connection.gmail", "v2", m)
    store.set("oauth.github.access_token", "v3", m)

    composio_names = sorted(store.list_names(prefix="composio."))
    assert composio_names == ["composio.api_key", "composio.connection.gmail"]

    github_names = sorted(store.list_names(prefix="oauth.github."))
    assert github_names == ["oauth.github.access_token"]


def test_reject_empty_name(store):
    with pytest.raises(ValueError, match="non-empty"):
        store.get("")
    with pytest.raises(ValueError, match="non-empty"):
        store.set("", "v", SecretMeta(kind="api_key", provider="composio"))


def test_reject_empty_value(store):
    with pytest.raises(ValueError, match="empty secret"):
        store.set("composio.api_key", "", SecretMeta(kind="api_key", provider="composio"))


def test_reject_reserved_suffix(store):
    with pytest.raises(ValueError, match="reserved"):
        store.set(
            "anything:meta",
            "v",
            SecretMeta(kind="api_key", provider="composio"),
        )


def test_unavailable_when_no_backend(monkeypatch):
    """If the keyring resolves to the `fail` backend, construction must raise."""
    import keyring  # noqa: PLC0415
    import keyring.backends.fail  # noqa: PLC0415

    from wiki_core.secrets import KeyringSecretStore

    original = keyring.get_keyring()
    keyring.set_keyring(keyring.backends.fail.Keyring())
    try:
        with pytest.raises(SecretStoreUnavailable, match="No usable keyring backend"):
            KeyringSecretStore(service="test.fail")
    finally:
        keyring.set_keyring(original)
