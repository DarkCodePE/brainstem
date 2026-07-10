"""
`KeyringSecretStore` — the default `SecretStore` impl.

Wraps the `keyring` package (cross-platform: libsecret on Linux, Keychain
on macOS, DPAPI/Credential Manager on Windows) and persists `SecretMeta`
as a JSON sidecar under a deterministic name.

Per [ADR-017](../../../docs/ADR-017-oauth-token-storage-and-scope-policy.md) §Implementation notes:
the keychain stores ciphertext-equivalent (it's already encrypted by the OS),
no plaintext ever lands on disk, and `name` is the only unique identifier.

Two keys are written per logical entry:

- ``<service>:<name>``       — the secret value
- ``<service>:<name>:meta``  — JSON-serialised `SecretMeta`

This keeps the OS-keychain UI (which shows one row per username) consistent
with the meta-alongside-secret contract.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict
from typing import TYPE_CHECKING

from wiki_core.secrets.protocol import (
    SecretMeta,
    SecretStore,
    SecretStoreUnavailable,
)

if TYPE_CHECKING:
    import keyring.backend

_log = logging.getLogger(__name__)

DEFAULT_SERVICE = "com.sbw.integrations"
"""Service namespace shown in the OS keychain UI. ADR-017 §Decision outcome."""

_META_SUFFIX = ":meta"
_INDEX_NAME = "__index__"
"""Sidecar entry holding the list of stored names — keyring doesn't expose a
list operation, so we maintain our own index."""


class KeyringSecretStore(SecretStore):
    """Production `SecretStore` impl backed by the `keyring` package.

    Pass `backend` to override the default OS keychain (used in tests with
    an in-memory backend).
    """

    def __init__(
        self,
        *,
        service: str = DEFAULT_SERVICE,
        backend: keyring.backend.KeyringBackend | None = None,
    ) -> None:
        try:
            import keyring  # noqa: PLC0415  (lazy: don't crash imports on headless)
        except ImportError as exc:  # pragma: no cover -- declared dep
            raise SecretStoreUnavailable("keyring package not installed") from exc

        self._keyring = keyring
        self._service = service
        if backend is not None:
            keyring.set_keyring(backend)
        # Sanity-check the active backend isn't the broken `fail.Keyring`,
        # which `keyring` falls back to when nothing else works.
        active = keyring.get_keyring()
        if type(active).__name__ == "Keyring" and active.__module__ == "keyring.backends.fail":
            raise SecretStoreUnavailable(
                "No usable keyring backend found. Install libsecret (Linux), "
                "unlock the Login Keychain (macOS), or set $SBW_VAULT_PASSWORD "
                "for headless mode."
            )

    # ------------------------------------------------------------------ #
    # SecretStore surface                                                #
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> str | None:
        _validate_name(name)
        try:
            return self._keyring.get_password(self._service, name)
        except Exception as exc:  # noqa: BLE001 -- keyring raises a soup of errors
            _log.exception("keyring.get_password failed for name=%s", name)
            raise SecretStoreUnavailable(str(exc)) from exc

    def set(self, name: str, value: str, meta: SecretMeta) -> None:
        _validate_name(name)
        if not value:
            raise ValueError("Refusing to set an empty secret value")
        meta_json = json.dumps(asdict(meta), separators=(",", ":"), sort_keys=True)
        try:
            self._keyring.set_password(self._service, name, value)
            self._keyring.set_password(self._service, name + _META_SUFFIX, meta_json)
        except Exception as exc:  # noqa: BLE001
            # Best-effort rollback of the value if the meta write failed,
            # so list_names()'s state stays consistent.
            try:
                self._keyring.delete_password(self._service, name)
            except Exception:  # noqa: BLE001
                pass
            raise SecretStoreUnavailable(f"failed to write secret {name!r}: {exc}") from exc
        self._index_add(name)

    def delete(self, name: str) -> bool:
        _validate_name(name)
        existed = self.get(name) is not None
        for key in (name, name + _META_SUFFIX):
            try:
                self._keyring.delete_password(self._service, key)
            except self._keyring.errors.PasswordDeleteError:
                pass  # already gone
        self._index_remove(name)
        return existed

    def get_meta(self, name: str) -> SecretMeta | None:
        _validate_name(name)
        raw = self._keyring.get_password(self._service, name + _META_SUFFIX)
        if raw is None:
            return None
        data = json.loads(raw)
        return SecretMeta(
            kind=data["kind"],
            provider=data["provider"],
            scope=tuple(data.get("scope", ())),
            granted_at=data.get("granted_at"),
            expires_at=data.get("expires_at"),
            refresh_token_name=data.get("refresh_token_name"),
        )

    def list_names(self, prefix: str = "") -> Iterable[str]:
        return tuple(n for n in self._index_load() if n.startswith(prefix))

    # ------------------------------------------------------------------ #
    # Index helpers                                                      #
    # ------------------------------------------------------------------ #

    def _index_load(self) -> set[str]:
        raw = self._keyring.get_password(self._service, _INDEX_NAME)
        if not raw:
            return set()
        return set(json.loads(raw))

    def _index_save(self, names: set[str]) -> None:
        self._keyring.set_password(self._service, _INDEX_NAME, json.dumps(sorted(names)))

    def _index_add(self, name: str) -> None:
        idx = self._index_load()
        idx.add(name)
        self._index_save(idx)

    def _index_remove(self, name: str) -> None:
        idx = self._index_load()
        idx.discard(name)
        self._index_save(idx)


def _validate_name(name: str) -> None:
    if not name:
        raise ValueError("Secret name must be non-empty")
    if name.endswith(_META_SUFFIX) or name == _INDEX_NAME:
        raise ValueError(f"Secret name {name!r} collides with reserved suffix/name")
