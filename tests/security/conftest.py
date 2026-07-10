"""
Shared fixtures for `tests/security/`.

The in-memory keyring backend lets every test run hermetically — no touching
the real OS keychain, no leftover entries between tests.
"""

from __future__ import annotations

import keyring.backend
import keyring.errors
import pytest


class InMemoryKeyringBackend(keyring.backend.KeyringBackend):
    """Minimal `keyring.backend.KeyringBackend` impl backed by a dict.

    Implements only what `KeyringSecretStore` actually calls. Keyed by
    ``(service, username)`` per `keyring`'s convention.
    """

    priority = 100  # high enough to beat the default backends

    def __init__(self) -> None:
        super().__init__()
        self._store: dict[tuple[str, str], str] = {}

    @property
    def errors(self):
        return keyring.errors

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self._store[(service, username)]
        except KeyError:
            raise keyring.errors.PasswordDeleteError(
                f"no entry for {service!r}/{username!r}"
            ) from None


@pytest.fixture
def memory_backend() -> InMemoryKeyringBackend:
    """Per-test in-memory keyring backend."""
    return InMemoryKeyringBackend()


@pytest.fixture
def store(memory_backend, monkeypatch):
    """A `KeyringSecretStore` wired against the in-memory backend.

    `monkeypatch` here is unused at runtime but kept to mark the fixture as
    test-scoped (auto-resets between tests).
    """
    import keyring  # noqa: PLC0415

    from wiki_core.secrets import KeyringSecretStore

    # Save and restore so we don't poison parallel tests
    original = keyring.get_keyring()
    keyring.set_keyring(memory_backend)
    try:
        yield KeyringSecretStore(service="test.sbw.integrations")
    finally:
        keyring.set_keyring(original)


@pytest.fixture
def audit_log_tmp(tmp_path):
    """An `AuditLog` writing to a fresh tmp file per test."""
    from wiki_core.secrets import AuditLog

    return AuditLog(path=tmp_path / "integrations.log.jsonl")
