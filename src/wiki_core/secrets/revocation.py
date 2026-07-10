"""
Atomic `disconnect(provider)` per [ADR-017](../../../docs/ADR-017-oauth-token-storage-and-scope-policy.md) §Revocation contract.

The contract: clicking Disconnect <Provider> must, in this order:

1. Call the upstream OAuth/Composio revocation endpoint.
2. Delete the local `vault.db` rows for connection_id + access/refresh tokens.
3. Append a `revoked` event to the audit log.
4. Notify in-flight tools (they'll see ``IntegrationDisconnectedError`` next call).

If step 1 fails the local state is left untouched — the secret stays in the
vault, the user can retry. If step 2 fails after step 1 succeeded, the local
state is marked `disconnected_pending_cleanup` (a `meta_json` mutation) so a
follow-up cron run can finish the cleanup. The audit log always reflects the
final terminal state.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from wiki_core.secrets.audit_log import AuditLog
from wiki_core.secrets.protocol import SecretStore

_log = logging.getLogger(__name__)


UpstreamRevoker = Callable[[str, str], None]
"""(provider, connection_id) -> None. Raises on failure."""


class IntegrationDisconnectedError(RuntimeError):
    """Raised by integration code when a provider has been revoked
    mid-session and a cached `connection_id` is no longer valid."""


@dataclass(frozen=True)
class DisconnectResult:
    provider: str
    upstream_revoked: bool
    local_cleared: bool
    remaining_names: tuple[str, ...]
    """Vault entries that still exist after the attempt — empty on a clean run."""


def disconnect(
    provider: str,
    *,
    store: SecretStore,
    audit: AuditLog,
    upstream_revoker: UpstreamRevoker | None = None,
) -> DisconnectResult:
    """Run the full revocation flow for `provider`.

    Parameters
    ----------
    provider:
        Lowercase provider id (``gmail``, ``github``, …).
    store:
        Where Composio `connection_id` + Phase-3 OAuth tokens live.
    audit:
        Where to record the `revoked` event.
    upstream_revoker:
        Callable invoked first to revoke the bearer at Composio (or the
        provider directly in Phase 3). When ``None``, only local state is
        cleared — used by tests and by the Phase-3 path when SBW holds the
        bearer itself.

    Returns
    -------
    DisconnectResult describing what happened. ``remaining_names`` is empty
    on a clean run; populated on partial failure so the caller can surface
    a "needs cleanup" badge in Settings.
    """
    names = _names_for(provider, store)
    connection_id = _connection_id(provider, store) or ""

    upstream_revoked = False
    if upstream_revoker is not None and connection_id:
        try:
            upstream_revoker(provider, connection_id)
            upstream_revoked = True
        except Exception as exc:
            _log.warning(
                "Upstream revoke failed for provider=%s: %s; local state preserved.",
                provider,
                exc,
            )
            audit.write(
                event="revoked",
                provider=provider,
                result="upstream_failed",
                params={"reason": str(exc)},
            )
            return DisconnectResult(
                provider=provider,
                upstream_revoked=False,
                local_cleared=False,
                remaining_names=tuple(names),
            )
    elif upstream_revoker is None:
        # No upstream to call; we treat the local-only flow as "upstream
        # not applicable" — record but don't claim success on the upstream.
        upstream_revoked = False

    # Local cleanup — delete each entry; if any delete throws, the
    # remaining ones are still attempted, and the failure is surfaced.
    remaining: list[str] = []
    cleared_count = 0
    for name in names:
        try:
            existed = store.delete(name)
            if existed:
                cleared_count += 1
        except Exception as exc:  # noqa: BLE001 -- want broad coverage of backend errors
            _log.error("delete failed for %s: %s", name, exc)
            remaining.append(name)

    local_cleared = not remaining

    audit.write(
        event="revoked",
        provider=provider,
        result="ok" if local_cleared else "partial",
        params={
            "upstream_revoked": upstream_revoked,
            "names_cleared": cleared_count,
            "names_remaining": remaining,
        },
    )

    return DisconnectResult(
        provider=provider,
        upstream_revoked=upstream_revoked,
        local_cleared=local_cleared,
        remaining_names=tuple(remaining),
    )


# ----------------------------------------------------------------------- #
# Helpers                                                                 #
# ----------------------------------------------------------------------- #


def _names_for(provider: str, store: SecretStore) -> list[str]:
    """Collect all entry names that belong to `provider`."""
    # Stable conventions per ADR-017 §Secret storage map. Use prefix scan
    # so a Phase-3 access+refresh pair is caught alongside connection_id.
    candidates = [
        f"composio.connection.{provider}",
        f"oauth.{provider}.access_token",
        f"oauth.{provider}.refresh_token",
    ]
    discovered = [n for n in store.list_names(prefix=f"oauth.{provider}.")]
    composio = [n for n in store.list_names(prefix=f"composio.connection.{provider}")]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for n in (*candidates, *discovered, *composio):
        if n in seen:
            continue
        seen.add(n)
        if store.get(n) is not None or store.get_meta(n) is not None:
            out.append(n)
    return out


def _connection_id(provider: str, store: SecretStore) -> str | None:
    return store.get(f"composio.connection.{provider}")
