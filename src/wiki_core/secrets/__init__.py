"""
`wiki_core.secrets` — typed secret storage + OAuth scope policy + audit log
per [ADR-017](../../../docs/ADR-017-oauth-token-storage-and-scope-policy.md)
and issue #39.

Public API:

- `SecretStore` Protocol + `SecretMeta` dataclass — what integration code depends on.
- `KeyringSecretStore` — default OS-keychain-backed impl.
- `PROVIDER_SCOPES` / `policy_for(provider)` — frozen scope table.
- `AuditLog` — append-only JSONL for integration events (redaction baked in).
- `disconnect(...)` — atomic revocation flow.

Integration code should *only* import from this top-level module; the
submodules are implementation detail.
"""

from __future__ import annotations

from wiki_core.secrets.audit_log import (
    AUDIT_SCHEMA_VERSION,
    DEFAULT_LOG_PATH,
    AuditLog,
    redact_params,
)
from wiki_core.secrets.keyring_store import DEFAULT_SERVICE, KeyringSecretStore
from wiki_core.secrets.protocol import (
    SecretMeta,
    SecretNotFound,
    SecretStore,
    SecretStoreError,
    SecretStoreUnavailable,
)
from wiki_core.secrets.revocation import (
    DisconnectResult,
    IntegrationDisconnectedError,
    UpstreamRevoker,
    disconnect,
)
from wiki_core.secrets.scope_policy import (
    PROVIDER_SCOPES,
    SUPPORTED_PROVIDERS,
    ScopePolicy,
    policy_for,
)

__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AuditLog",
    "DEFAULT_LOG_PATH",
    "DEFAULT_SERVICE",
    "DisconnectResult",
    "IntegrationDisconnectedError",
    "KeyringSecretStore",
    "PROVIDER_SCOPES",
    "SUPPORTED_PROVIDERS",
    "ScopePolicy",
    "SecretMeta",
    "SecretNotFound",
    "SecretStore",
    "SecretStoreError",
    "SecretStoreUnavailable",
    "UpstreamRevoker",
    "disconnect",
    "policy_for",
    "redact_params",
]
