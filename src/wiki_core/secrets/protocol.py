"""
Typed `SecretStore` Protocol per [ADR-017](../../../docs/ADR-017-oauth-token-storage-and-scope-policy.md).

The Protocol is the only thing the rest of the codebase imports — concrete
backends (`KeyringSecretStore`, future `RustVaultSecretStore` per ADR-011)
implement it structurally. This keeps the OAuth integration layer testable
without touching the OS keychain.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SecretMeta:
    """OAuth-token-policy metadata stored alongside the ciphertext.

    Per ADR-017 §Secret storage map. Every field but `kind` and `provider`
    is optional so the same shape works for an API key, a Composio
    connection_id, and a Phase-3 OAuth access/refresh pair.
    """

    kind: str
    """One of: ``api_key``, ``connection_id``, ``oauth_access``,
    ``oauth_refresh``. Drives lookup names and rotation policy."""

    provider: str
    """Stable provider identifier — ``composio``, ``gmail``, ``github`` …
    Always lowercase. Matches the keys in `scope_policy.PROVIDER_SCOPES`."""

    scope: tuple[str, ...] = field(default_factory=tuple)
    """OAuth scopes granted. Empty for API keys. Locked per provider in
    `scope_policy.PROVIDER_SCOPES` — drift triggers `scope_drift_required`."""

    granted_at: str | None = None
    """ISO-8601 timestamp the user first consented. Surfaced in Settings."""

    expires_at: str | None = None
    """ISO-8601 expiry for access tokens; ``None`` for refresh tokens and
    API keys (long-lived)."""

    refresh_token_name: str | None = None
    """For `oauth_access` entries, the `name` of the paired refresh entry.
    Lets `revocation.disconnect` delete both atomically."""


@runtime_checkable
class SecretStore(Protocol):
    """Durable, OS-keychain-backed key-value store for OAuth bearers and API keys.

    Implementations MUST:

    - Never persist plaintext to disk (the `no_plaintext_secrets` CI guard
      will fail if they do).
    - Be safe to call from sync code; OAuth flows are not async-hot.
    - Treat `name` as the only unique identifier — overwriting a name
      replaces the secret and the meta in one atomic operation.
    - Persist `SecretMeta` alongside the value, retrievable via `get_meta`.

    Implementations SHOULD:

    - Surface backend failures (locked keychain, missing libsecret on
      headless Linux) as `SecretStoreUnavailable` rather than raw OS
      exceptions, so the integration layer can degrade gracefully.
    """

    def get(self, name: str) -> str | None:
        """Return the secret value for `name`, or `None` if absent."""

    def set(self, name: str, value: str, meta: SecretMeta) -> None:
        """Store `value` under `name` with policy metadata.

        Overwrites any existing entry under the same `name`. Both the
        secret and the meta are replaced atomically — partial state is
        not observable.
        """

    def delete(self, name: str) -> bool:
        """Remove the entry. Returns ``True`` if it existed, ``False`` otherwise."""

    def get_meta(self, name: str) -> SecretMeta | None:
        """Return the meta for `name`, or ``None`` if no such entry."""

    def list_names(self, prefix: str = "") -> Iterable[str]:
        """Return entry names matching `prefix`. Values are never returned here."""


class SecretStoreError(Exception):
    """Base for secret-store failures. Catch this in integration code."""


class SecretStoreUnavailable(SecretStoreError):  # noqa: N818 -- API surface; `Error` suffix is on the base
    """The keychain backend is not usable (locked, no libsecret, no Login Keychain).

    Surfaced separately from generic errors so callers can prompt the user
    to install/unlock a keyring, rather than crash.
    """


class SecretNotFound(SecretStoreError):  # noqa: N818 -- API surface; `Error` suffix is on the base
    """A name was expected but not present. Used by callers that prefer
    explicit errors over `None`-returns."""
