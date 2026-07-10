"""
`IIntegration` — the typed agent-tool surface for OAuth integrations per
[PRD-006](../../../docs/PRD-006-integrations-framework.md) and
[ADR-017](../../../docs/ADR-017-oauth-token-storage-and-scope-policy.md).

This is the *interactive* surface the agent calls when the user says
"search my Gmail for the contract". The polling/ingest surface lives in
`wiki_core.protocols.IngestSource` — a single provider implementation
may expose both.

Every method is `async` because the underlying Composio + provider REST
clients are async. Implementations MUST NOT swallow auth failures —
`NotConnectedError` lets the orchestrator surface "click Connect" UX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class IntegrationItem:
    """One result row returned by `list`, `get`, or `search`.

    Shape is intentionally provider-agnostic — the agent reasons over
    titles, snippets, and URIs, not provider-specific JSON. Per-provider
    detail lives in `metadata`.
    """

    id: str
    """Stable provider-local id (issue number, gmail message id, …)."""

    title: str
    """One-line human-readable identifier (issue title, email subject)."""

    snippet: str
    """Short body excerpt; capped at ~500 chars by convention so the agent
    doesn't blow its context window. Empty for metadata-only listings."""

    uri: str
    """Canonical URL (GitHub html_url, ``mailto:`` for gmail). The agent
    quotes this back to the user so they can click through."""

    updated_at: datetime
    """Last-modified timestamp from the upstream. Used for ordering and
    for cursor-based pagination on the polling side."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-specific bag: ``state`` for issues, ``labels`` for gmail,
    etc. Never contains secrets — that's enforced at the SecretStore /
    audit-log layer."""


@dataclass(frozen=True)
class SearchResult:
    """Search response: the matched items plus pagination hints."""

    items: tuple[IntegrationItem, ...]
    total_estimated: int | None = None
    """Provider's estimate of total matches; ``None`` if unknown."""
    next_cursor: str | None = None
    """Opaque cursor to pass back for the next page; ``None`` ends pagination."""


@dataclass(frozen=True)
class ConnectResult:
    """Returned by `connect`: enough to drive the user through the OAuth flow."""

    provider: str
    connection_id: str
    """Stable id stored in the keychain — same shape Composio surfaces."""
    redirect_url: str | None = None
    """If the OAuth flow is browser-based, the URL the user opens.
    ``None`` for already-connected accounts or stub mode."""
    status: str = "pending"
    """``pending`` until the user completes the flow; ``connected`` once Composio
    confirms; ``failed`` on terminal error."""


class IntegrationError(RuntimeError):
    """Base for IIntegration failures. Callers catch this to surface UX."""


class NotConnectedError(IntegrationError):
    """The provider has no live connection. Caller should prompt the user to connect."""


@runtime_checkable
class IIntegration(Protocol):
    """Agent-tool surface for one OAuth provider.

    Lifecycle is two-phased: ``connect()`` once per user per provider,
    then any number of ``list/get/search`` calls. ``disconnect()`` is the
    revocation endpoint per [[adr-017]] §Revocation contract.

    Health is exposed so the desktop shell's Settings → Integrations can
    show a green/red dot without forcing the agent to issue a probe call
    on every render.
    """

    @property
    def provider(self) -> str:
        """Lowercase provider id (``gmail``, ``github``, …)."""
        ...

    async def connect(self) -> ConnectResult:
        """Start (or resume) the OAuth flow. Idempotent — calling on an
        already-connected provider returns the existing connection."""
        ...

    async def disconnect(self) -> None:
        """Revoke the connection: upstream + local vault + audit log.
        Raises `IntegrationError` if any step fails irrecoverably."""
        ...

    async def health(self) -> bool:
        """Return ``True`` iff the connection is alive and the locked
        scope is still granted upstream. Never raises — returns ``False``
        on any error path."""
        ...

    async def list(
        self,
        *,
        since: datetime | None = None,
        limit: int = 50,
    ) -> tuple[IntegrationItem, ...]:
        """Return up to `limit` items newer than `since`.

        ``since=None`` is "latest window" — provider-defined (24h for Gmail,
        all open issues for GitHub). Raises `NotConnectedError` if the
        provider has not been connected yet.
        """
        ...

    async def get(self, item_id: str) -> IntegrationItem:
        """Fetch a single item by its provider-local id.

        Raises `KeyError` if no such item, `NotConnectedError` if the
        provider isn't connected.
        """
        ...

    async def search(
        self,
        query: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> SearchResult:
        """Full-text search the provider for `query`.

        ``cursor`` is opaque — pass `SearchResult.next_cursor` back to
        paginate. Raises `NotConnectedError` if not connected.
        """
        ...
