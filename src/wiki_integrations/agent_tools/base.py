"""
`ComposioBackedIntegration` — abstract base for `IIntegration` providers
that route through the existing `ComposioBridge` (Phase 1 per ADR-009).

Subclasses provide:

- `provider` (class const)
- `scopes` (class const, lifted from `wiki_core.secrets.PROVIDER_SCOPES`
  via `policy_for(provider)`)
- `_to_item(raw: dict) -> IntegrationItem` — provider-shape normalisation
- `_search_action_name` (class const) — Composio action id used by `search`
- `_get_action_name` (class const) — Composio action id used by `get`

Everything else (connect / disconnect / health / list / audit-log writes /
SecretStore interactions) is centralised here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from wiki_core.integrations.protocol import (
    ConnectResult,
    IIntegration,
    IntegrationError,
    IntegrationItem,
    NotConnectedError,
    SearchResult,
)
from wiki_core.secrets import (
    AuditLog,
    SecretMeta,
    SecretStore,
    policy_for,
)
from wiki_core.secrets import (
    disconnect as secrets_disconnect,
)

if TYPE_CHECKING:
    from wiki_integrations.agent_tools.audit_md import ProviderMarkdownLog
    from wiki_integrations.composio_bridge import ComposioBridge

_log = logging.getLogger(__name__)


class ComposioBackedIntegration(IIntegration, ABC):
    """Composio-routed base for an OAuth provider's agent-tool surface.

    Implements all of `IIntegration` except `list`, `get`, `search` —
    those need provider-specific Composio action names and payload shapes.
    """

    PROVIDER: str = ""
    """Override in subclass. Must match a key in `PROVIDER_SCOPES`."""

    def __init__(
        self,
        *,
        bridge: ComposioBridge,
        store: SecretStore,
        audit_jsonl: AuditLog,
        audit_md: ProviderMarkdownLog | None = None,
    ) -> None:
        if not self.PROVIDER:
            raise TypeError(f"{type(self).__name__} must set PROVIDER class const")
        # Will raise KeyError if subclass picked an unsupported provider id;
        # that's the point — scope policy is the source of truth.
        self._policy = policy_for(self.PROVIDER)
        self._bridge = bridge
        self._store = store
        self._audit_jsonl = audit_jsonl
        self._audit_md = audit_md

    @property
    def provider(self) -> str:
        return self.PROVIDER

    @property
    def scopes(self) -> tuple[str, ...]:
        """Default scopes for this provider (no opt-in extras)."""
        return self._policy.default

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def connect(self) -> ConnectResult:
        """Idempotent — returns the existing connection if present.

        On first connect: writes the connection_id to the SecretStore with
        the locked scope set as metadata, records a `connect` event in
        both audit logs.
        """
        existing = self._connection_id()
        if existing is not None:
            self._mark("connect", "already_connected")
            return ConnectResult(
                provider=self.PROVIDER,
                connection_id=existing,
                status="connected",
            )

        try:
            conn = await self._bridge.connect(self.PROVIDER)
        except Exception as exc:  # noqa: BLE001 -- bridge raises a soup of errors
            self._mark("connect", "error", note=type(exc).__name__)
            raise IntegrationError(f"connect failed: {exc}") from exc

        meta = SecretMeta(
            kind="connection_id",
            provider=self.PROVIDER,
            scope=self._policy.default,
            granted_at=self._utcnow_iso(),
        )
        self._store.set(f"composio.connection.{self.PROVIDER}", conn.connection_id, meta)
        self._mark("connect", "ok", note=f"status={conn.status}")
        return ConnectResult(
            provider=self.PROVIDER,
            connection_id=conn.connection_id,
            status=conn.status,
            redirect_url=getattr(conn, "redirect_url", None),
        )

    async def disconnect(self) -> None:
        """Atomic revocation. Delegates to `wiki_core.secrets.disconnect`."""
        # Composio's REST API exposes DELETE /api/v1/connections/{id}; we
        # don't have a thin wrapper on ComposioBridge yet, so the upstream
        # revoker is left as None — local-only clear. Real upstream call
        # lands when bridge.revoke() ships (issue #40 followup).
        result = secrets_disconnect(
            self.PROVIDER,
            store=self._store,
            audit=self._audit_jsonl,
            upstream_revoker=None,
        )
        if not result.local_cleared:
            self._mark("disconnect", "partial")
            raise IntegrationError(
                f"local revoke partial: {len(result.remaining_names)} entries remain"
            )
        self._mark("disconnect", "ok")

    async def health(self) -> bool:
        """Probe the bridge: does it list our provider as connected?

        Returns ``False`` on any backend error rather than raising — the
        caller is usually a status dot in Settings, not a tool call.
        """
        try:
            active = await self._bridge.list_active()
        except Exception as exc:  # noqa: BLE001
            _log.warning("health check error for %s: %s", self.PROVIDER, exc)
            self._mark("health", "error", note=type(exc).__name__)
            return False
        ok = any(c.provider == self.PROVIDER for c in active)
        self._mark("health", "ok" if ok else "not_connected")
        return ok

    # ------------------------------------------------------------------ #
    # CRUD — provider-specific shapes                                    #
    # ------------------------------------------------------------------ #

    async def list(
        self,
        *,
        since: datetime | None = None,
        limit: int = 50,
    ) -> tuple[IntegrationItem, ...]:
        self._require_connected()
        items: list[IntegrationItem] = []
        async for raw in self._bridge.walk(self.PROVIDER):
            try:
                item = self._to_item(raw)
            except (KeyError, TypeError, ValueError):
                continue
            if since is not None and item.updated_at < since:
                continue
            items.append(item)
            if len(items) >= limit:
                break
        self._mark("list", "ok", items=len(items))
        return tuple(items)

    @abstractmethod
    def _to_item(self, raw: dict[str, Any]) -> IntegrationItem:
        """Map a raw Composio payload row into an `IntegrationItem`.

        Subclasses pick the right keys per Composio's per-provider shape.
        Failures are converted into "skip this row" by the caller.
        """

    @abstractmethod
    async def get(self, item_id: str) -> IntegrationItem:
        """Per-provider single-item fetch.

        Subclasses call a Composio action (Composio doesn't expose a
        uniform "get by id"). MUST call `self._require_connected()` first.
        """

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResult:
        """Per-provider search."""

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _connection_id(self) -> str | None:
        return self._store.get(f"composio.connection.{self.PROVIDER}")

    def _require_connected(self) -> None:
        if self._connection_id() is None:
            self._mark("call", "not_connected")
            raise NotConnectedError(
                f"Provider {self.PROVIDER!r} is not connected. Call connect() first."
            )

    def _mark(
        self,
        op: str,
        result: str,
        *,
        items: int | None = None,
        note: str | None = None,
    ) -> None:
        """Write to both the JSONL forensic log and the markdown user-log."""
        self._audit_jsonl.write(
            event="execute_action" if op not in {"connect", "disconnect"} else op,
            provider=self.PROVIDER,
            action=op,
            result=result,
            scope_used=self._policy.default,
        )
        if self._audit_md is not None:
            self._audit_md.append(op=op, result=result, items=items, note=note)

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
