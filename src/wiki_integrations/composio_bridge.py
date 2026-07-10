"""
Composio managed-mode bridge for Phase 1 of [ADR-009](../../docs/ADR-009-oauth-integrations-strategy.md).

Composio holds the OAuth tokens server-side and exposes a uniform
``execute_action`` (and friends) RPC. This module wraps the slice that
matters for ingest-side polling:

- ``connect(provider)`` — kick off the OAuth handshake (returns a connection id).
- ``list_active()`` — list the providers the current account has connected.
- ``walk(provider)`` — yield the provider's items as raw JSON dicts. The
  concrete ``OAuthIntegrationSource`` subclasses (see ``providers/``)
  translate these into ``wiki_core.IngestEvent``s.

Because we don't ship with real Composio credentials, the bridge gates on
the ``COMPOSIO_API_KEY`` env var (or an explicit constructor argument):

- If a key is present, we hit the real API via ``httpx.AsyncClient``.
- If no key is present, we fall back to a deterministic stub iterator
  defined inline. This keeps tests hermetic and lets development continue
  without burning real OAuth tokens. The stub is gated by a ``stub_data``
  hook so unit tests can pin it to a known payload.

The real-mode path is structured but **best-effort**: SPEC-006 OQ-1 calls
out that we have not been able to verify the exact JSON schema against a
real Composio account. The endpoints and field names below were drawn
from Composio's public documentation as of 2026-05-24. When a real key
lands, a follow-up PR should validate these against live traffic. See
the comments tagged ``schema-best-effort`` for the specific guesses.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wiki_routing.fallback import BackendError

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import httpx

log = logging.getLogger("wiki_integrations.composio_bridge")


# Deterministic stub payloads — used when no API key is set so the test
# suite (and local development) runs without network. The fixture is
# pinned: items are ordered, ids are stable, bodies have content the
# providers know how to translate.
_DEFAULT_STUB: Mapping[str, tuple[dict[str, Any], ...]] = {
    "gmail": (
        {
            "id": "gmail-msg-001",
            "thread_id": "gmail-thread-001",
            "from": "alice@example.com",
            "to": "okuanb@efectiva.com.pe",
            "subject": "Project kickoff",
            "snippet": "Starting on the OAuth substrate today.",
            "body": "Starting on the OAuth substrate today.\nWill have a PR by EOW.",
            "internal_date": "2026-05-22T10:00:00Z",
        },
        {
            "id": "gmail-msg-002",
            "thread_id": "gmail-thread-002",
            "from": "bob@example.com",
            "to": "okuanb@efectiva.com.pe",
            "subject": "RFC: token vault",
            "snippet": "Phase 2 plan attached.",
            "body": "Phase 2 plan attached. Let me know if the scope matches.",
            "internal_date": "2026-05-22T11:30:00Z",
        },
    ),
    "github": (
        {
            "id": "gh-issue-001",
            "kind": "issue",
            "number": 121,
            "title": "OAuth substrate: scaffold provider registry",
            "body": "Track the PRD-005 substrate ship.",
            "html_url": "https://github.com/DarkCodePE/second-brain-wiki/issues/121",
            "updated_at": "2026-05-22T09:00:00Z",
            "state": "open",
        },
        {
            "id": "gh-pr-001",
            "kind": "pull_request",
            "number": 124,
            "title": "feat(m3): integrations foundation",
            "body": "Implements PRD-005 Phase 1 per ADR-009.",
            "html_url": "https://github.com/DarkCodePE/second-brain-wiki/pull/124",
            "updated_at": "2026-05-22T12:15:00Z",
            "state": "open",
        },
    ),
}


# Per-provider Composio v3 tool slugs. Confirmed against the live v3
# ``/api/v3/tools?toolkit_slug=…&search=…`` enumeration on 2026-05-27.
# v3 slug convention: UPPERCASE_UNDERSCORE (was v1 ``<APP>.<ACTION>``).
_LIST_ACTION_BY_PROVIDER: Mapping[str, str] = {
    "gmail": "GMAIL_FETCH_EMAILS",
    "github": "GITHUB_LIST_REPOSITORIES_FOR_AUTHENTICATED_USER",
    "calendar": "GOOGLECALENDAR_EVENTS_LIST",
}

# Per-provider required arguments for the list/walk action. Composio's
# v3 tool schemas require certain fields that aren't pagination params
# (e.g. ``calendarId`` for Google Calendar). Until we expose a richer
# argument-passing surface (followup), these defaults satisfy the
# minimum-required-fields contract so ``walk()`` doesn't 400.
_DEFAULT_ARGUMENTS_BY_PROVIDER: Mapping[str, Mapping[str, Any]] = {
    "calendar": {"calendarId": "primary"},
}

# Maps SBW's internal provider id → Composio v3 toolkit slug. They
# diverge for Google products: SBW says ``calendar`` / ``drive`` while
# Composio expects ``googlecalendar`` / ``googledrive`` (see
# ``/api/v3/toolkits``). The locked scope policy keeps the SBW-side names
# stable; this map is the only translation point.
_COMPOSIO_TOOLKIT_SLUG: Mapping[str, str] = {
    "calendar": "googlecalendar",
    "drive": "googledrive",
    "gmail": "gmail",
    "github": "github",
    "notion": "notion",
    "slack": "slack",
}


def _composio_slug(provider: str) -> str:
    """SBW-internal provider id → Composio v3 toolkit slug."""
    return _COMPOSIO_TOOLKIT_SLUG.get(provider, provider)


@dataclass(frozen=True, slots=True)
class ComposioConnection:
    """One row in ``list_active()``'s response.

    The real Composio response carries more fields (scopes, expires_at, …);
    we surface only what the registry needs at this layer.
    """

    provider: str
    connection_id: str
    status: str  # "connected" | "revoked" | "pending" | "error"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    redirect_url: str | None = None
    """Populated for ``connect()`` responses where Composio returns a URL the
    user must visit to authorise the integration. ``None`` when the
    handshake completes server-side (or in stub mode)."""


class ComposioBridge:
    """Managed-mode bridge over the Composio REST API.

    Parameters
    ----------
    api_key:
        Composio API key. Defaults to ``os.environ["COMPOSIO_API_KEY"]``.
        If neither is set, the bridge operates in **stub mode** and serves
        deterministic fixtures.
    base_url:
        Override for testing against a mock server. Defaults to
        ``https://backend.composio.dev``.
    stub_data:
        Override the inline stub payload. Tests pass a dict keyed by
        provider id to pin exactly what ``walk`` yields.
    transport:
        Optional ``httpx.BaseTransport`` (typically ``httpx.MockTransport``)
        used by tests to short-circuit real network calls. When set, the
        bridge wires it into every ``httpx.AsyncClient`` it constructs.
    retry_backoff:
        Sleep-seconds between retries on transient failures (429, 5xx).
        Default ``(1.0, 4.0, 16.0)`` mirrors the AnthropicProvider policy.
        Tests pass a tuple of zeros to keep runtime hermetic.
    sleep:
        Async sleep function. Defaults to ``asyncio.sleep``; tests inject
        a no-op so retries don't burn wall-clock.
    """

    DEFAULT_BASE_URL = "https://backend.composio.dev"
    DEFAULT_RETRY_BACKOFF: tuple[float, ...] = (1.0, 4.0, 16.0)
    DEFAULT_TIMEOUT_SECONDS = 10.0
    DEFAULT_USER_ID = "sbw-local"
    """SBW is single-user / local-first per ADR-009, so we don't need a
    per-end-user split. A stable id keeps Composio's per-user
    ``connected_account`` rows tidy across reconnects."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        user_id: str | None = None,
        stub_data: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
        transport: httpx.BaseTransport | None = None,
        retry_backoff: tuple[float, ...] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        env_key = os.environ.get("COMPOSIO_API_KEY", "").strip() or None
        self._api_key = api_key if api_key is not None else env_key
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        # user_id resolution order: explicit constructor arg > COMPOSIO_USER_ID
        # env var > DEFAULT_USER_ID. The env var matters when Composio's
        # dashboard auto-assigns a UUID-shaped user_id (e.g. ``pg-test-…``)
        # and we need the bridge to match the dashboard's connection rows.
        env_user_id = os.environ.get("COMPOSIO_USER_ID", "").strip() or None
        self._user_id = user_id or env_user_id or self.DEFAULT_USER_ID
        self._transport = transport
        self._retry_backoff = (
            retry_backoff if retry_backoff is not None else self.DEFAULT_RETRY_BACKOFF
        )
        self._sleep: Callable[[float], Awaitable[None]] = sleep or asyncio.sleep
        self._stub_data: Mapping[str, tuple[dict[str, Any], ...]] | None
        if stub_data is not None:
            # Defensive copy + freeze ordering so tests can rely on it.
            self._stub_data = {k: tuple(dict(item) for item in v) for k, v in stub_data.items()}
        elif self._api_key is None:
            self._stub_data = {k: tuple(dict(item) for item in v) for k, v in _DEFAULT_STUB.items()}
        else:
            self._stub_data = None
        # Track connections handed out in stub mode so list_active() reflects
        # what connect() was called for. In real mode we round-trip the
        # server and never use this.
        self._stub_connections: dict[str, ComposioConnection] = {}
        # Cache auth_config_id per Composio toolkit slug for the lifetime
        # of this bridge. v3's connect flow needs an auth_config_id *before*
        # the /link call; we list-or-create on first connect and reuse on
        # subsequent connects.
        self._auth_config_cache: dict[str, str] = {}
        # Stub responses keyed by ``(provider, tool_slug)`` for the
        # ``execute()`` escape hatch. Empty default so stub-mode integrations
        # that hit ``execute`` see ``{}`` until a test seeds something.
        self._stub_execute_data: dict[tuple[str, str], Mapping[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Mode introspection                                                 #
    # ------------------------------------------------------------------ #

    @property
    def stub_mode(self) -> bool:
        """True iff no API key was configured."""
        return self._api_key is None

    @property
    def base_url(self) -> str:
        return self._base_url

    # ------------------------------------------------------------------ #
    # Public surface                                                     #
    # ------------------------------------------------------------------ #

    async def connect(self, provider: str) -> ComposioConnection:
        """Kick off the OAuth handshake for ``provider``.

        In stub mode the connection is created synchronously and marked
        ``connected``. In real mode this issues a
        ``POST /api/v1/connections/initiate`` and the returned
        connection_id (plus ``redirect_url``) is the opaque handle the
        user's browser flow will eventually attach.
        """
        if not provider:
            raise ValueError("connect() requires a provider id")
        if self.stub_mode:
            conn = ComposioConnection(
                provider=provider,
                connection_id=f"stub-conn-{provider}",
                status="connected",
                metadata={"mode": "stub"},
            )
            self._stub_connections[provider] = conn
            return conn
        return await self._connect_real(provider)

    async def list_active(self) -> list[ComposioConnection]:
        """Return providers currently in the ``connected`` state."""
        if self.stub_mode:
            return [c for c in self._stub_connections.values() if c.status == "connected"]
        return [c for c in await self._list_active_real() if c.status == "connected"]

    async def list_connections(self) -> list[ComposioConnection]:
        """Return ALL Composio connections regardless of status.

        Used by ``sbw integrations list`` (issue #107) to surface the
        real upstream state — ``initializing`` after a half-completed
        OAuth flow, ``expired`` after token timeout — instead of trusting
        keyring presence (which only proves a connect() call was made,
        not that the connection is healthy today).

        Stub mode returns the in-memory stub connections as-is (any
        status). Real mode hits Composio's ``GET /api/v3/connected_accounts``
        once and returns every row.
        """
        if self.stub_mode:
            return list(self._stub_connections.values())
        return await self._list_active_real()

    async def walk(self, provider: str) -> AsyncIterator[dict[str, Any]]:
        """Yield the provider's items as raw JSON dicts.

        This is the surface the per-provider ``OAuthIntegrationSource``s
        consume. The iterator is async so the real-mode path can stream
        pages without buffering everything in memory.

        IMPORTANT: this is an async generator; callers must use
        ``async for item in bridge.walk(provider): ...``.
        """
        if not provider:
            raise ValueError("walk() requires a provider id")
        if self.stub_mode:
            assert self._stub_data is not None  # narrow for mypy
            items = self._stub_data.get(provider, ())
            for item in items:
                # Defensive copy so a consumer mutating an item doesn't
                # poison the next test.
                yield dict(item)
            return

        async for item in self._walk_real(provider):
            yield item

    async def execute(
        self,
        provider: str,
        tool_slug: str,
        arguments: dict[str, Any],
        *,
        version: str = "latest",
    ) -> dict[str, Any]:
        """Execute a single Composio v3 tool and return the inner ``data``.

        Unlike ``walk`` (which iterates the per-provider default tool with
        pagination), this is the escape hatch for integrations that need
        to fan out across many tools or pass per-call arguments —
        ``SlackIntegration`` uses it to iterate over its allowlist of
        channels.

        Path: ``POST /api/v3/tools/execute/{tool_slug}``.
        Body: ``{"user_id": <user>, "arguments": <arguments>, "version": <version>}``.
        Returns the ``data`` field of the response wrapper. If the
        response carries ``successful: false``, raises ``BackendError``.

        ``version`` pins the Composio *toolkit* version (NOT the SBW arg). It
        defaults to ``"latest"`` because the API's own default is the base
        pin ``00000000_00``, whose LinkedIn toolkit sends a sunset
        ``LinkedIn-Version`` header and gets HTTP 426 NONEXISTENT_VERSION on
        every post. "latest" tracks Composio's current, supported toolkit.

        In stub mode, returns the matching stub from
        ``_stub_execute_data[(provider, tool_slug)]`` if present, or an
        empty dict if not.
        """
        if not provider or not tool_slug:
            raise ValueError("execute() requires both provider and tool_slug")
        if self.stub_mode:
            return dict(self._stub_execute_data.get((provider, tool_slug), {}))
        body: dict[str, Any] = {"user_id": self._user_id, "arguments": arguments}
        if version:
            body["version"] = version
        payload = await self._request(
            method="POST",
            path=f"/api/v3/tools/execute/{tool_slug}",
            json=body,
        )
        if isinstance(payload, Mapping) and payload.get("successful") is False:
            err = payload.get("error") or "unknown error"
            raise BackendError(
                f"execute: tool {tool_slug} returned error: {err}",
                kind="unknown",
            )
        data = payload.get("data") if isinstance(payload, Mapping) else None
        return dict(data) if isinstance(data, Mapping) else {}

    # ------------------------------------------------------------------ #
    # Real-mode implementations                                          #
    # ------------------------------------------------------------------ #
    # NB: every endpoint path / payload field below is marked
    # ``schema-best-effort`` where we are guessing — SPEC-006 OQ-1 tracks
    # validation against a real Composio account.

    async def _connect_real(self, provider: str) -> ComposioConnection:
        """v3 connect: ensure auth_config exists, then POST /link.

        v3 splits what v1 used to do in one call into two:

        1. ``POST /api/v3/auth_configs`` registers a per-org auth bundle
           for the toolkit (using Composio-managed OAuth). One-time per
           toolkit per org — cached locally for the lifetime of the
           bridge.
        2. ``POST /api/v3/connected_accounts/link`` initiates the OAuth
           handshake for the user, returning the redirect URL.

        The returned ``redirect_url`` is what the end-user must open to
        complete consent. ``connection_id`` here is Composio's
        ``connected_account_id`` (``ca_…`` prefix).
        """
        toolkit_slug = _composio_slug(provider)
        auth_config_id = await self._ensure_auth_config_real(toolkit_slug)
        payload = await self._request(
            method="POST",
            path="/api/v3/connected_accounts/link",
            json={"auth_config_id": auth_config_id, "user_id": self._user_id},
        )
        if not isinstance(payload, Mapping):
            raise BackendError(
                f"connect: unexpected response shape: {type(payload).__name__}",
                kind="server",
            )
        connection_id = payload.get("connected_account_id") or payload.get("id")
        if connection_id is None:
            raise BackendError(
                "connect: missing connected_account_id in response",
                kind="server",
            )
        # v3 link responses don't carry a separate status — until the user
        # completes OAuth, Composio holds the connection in INITIALIZING.
        # We surface "pending" so the existing callers' state machine
        # ("pending" → "connected") stays compatible.
        return ComposioConnection(
            provider=provider,
            connection_id=str(connection_id),
            status="pending",
            metadata={
                "auth_config_id": auth_config_id,
                "link_token": payload.get("link_token"),
                "expires_at": payload.get("expires_at"),
            },
            redirect_url=(str(payload["redirect_url"]) if payload.get("redirect_url") else None),
        )

    async def _ensure_auth_config_real(self, toolkit_slug: str) -> str:
        """Return an auth_config_id for ``toolkit_slug``, creating one if needed.

        Hits ``GET /api/v3/auth_configs?toolkit=<slug>`` first; if Composio
        returns one already, reuses it. Otherwise creates a
        ``use_composio_managed_auth`` config and caches the new id.
        """
        cached = self._auth_config_cache.get(toolkit_slug)
        if cached is not None:
            return cached
        # Look for an existing managed auth_config for this toolkit so we
        # don't accumulate duplicate rows on every reconnect.
        listing = await self._request(
            method="GET",
            path="/api/v3/auth_configs",
            params={"toolkit_slug": toolkit_slug, "limit": 5},
        )
        if isinstance(listing, Mapping):
            for item in listing.get("items") or []:
                if not isinstance(item, Mapping):
                    continue
                ac = item.get("auth_config")
                if isinstance(ac, Mapping) and ac.get("id"):
                    auth_config_id = str(ac["id"])
                    self._auth_config_cache[toolkit_slug] = auth_config_id
                    return auth_config_id
        # No existing config — create a managed one. Per Composio v3 docs:
        # ``{"toolkit": {"slug": <slug>}, "auth_config": {"type": "use_composio_managed_auth"}}``
        created = await self._request(
            method="POST",
            path="/api/v3/auth_configs",
            json={
                "toolkit": {"slug": toolkit_slug},
                "auth_config": {"type": "use_composio_managed_auth"},
            },
        )
        if not isinstance(created, Mapping):
            raise BackendError(
                f"auth_config create: unexpected response shape: {type(created).__name__}",
                kind="server",
            )
        ac = created.get("auth_config")
        if not isinstance(ac, Mapping) or not ac.get("id"):
            raise BackendError(
                "auth_config create: missing auth_config.id in response",
                kind="server",
            )
        auth_config_id = str(ac["id"])
        self._auth_config_cache[toolkit_slug] = auth_config_id
        return auth_config_id

    async def _list_active_real(self) -> list[ComposioConnection]:
        """v3 list: ``GET /api/v3/connected_accounts``.

        Response shape: ``{"items": [{id, toolkit: {slug}, status, user_id,
        ...}], "next_cursor": …}``. Composio uses ``ACTIVE`` /
        ``INITIALIZING`` / ``EXPIRED`` for status; we lowercase
        ``ACTIVE`` to ``connected`` to keep the existing callers'
        semantics stable.
        """
        # Reverse map Composio toolkit slug → SBW provider id so the
        # returned connections expose the same provider strings the rest
        # of the codebase already uses.
        slug_to_provider: dict[str, str] = {v: k for k, v in _COMPOSIO_TOOLKIT_SLUG.items()}

        payload = await self._request(
            method="GET",
            path="/api/v3/connected_accounts",
            params={"limit": 100},
        )
        rows: Iterable[Any]
        if isinstance(payload, Mapping):
            rows = payload.get("items") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        out: list[ComposioConnection] = []
        for r in rows:
            if not isinstance(r, Mapping):
                continue
            toolkit = r.get("toolkit") if isinstance(r.get("toolkit"), Mapping) else {}
            toolkit_slug = str(toolkit.get("slug", "")) if toolkit else ""
            provider = slug_to_provider.get(toolkit_slug, toolkit_slug or "unknown")
            raw_status = str(r.get("status", "unknown")).lower()
            # Normalise Composio's ACTIVE → SBW's "connected".
            status = "connected" if raw_status == "active" else raw_status
            out.append(
                ComposioConnection(
                    provider=provider,
                    connection_id=str(r.get("id") or ""),
                    status=status,
                    metadata={
                        "user_id": r.get("user_id"),
                        "auth_config_id": (
                            (r.get("auth_config") or {}).get("id")
                            if isinstance(r.get("auth_config"), Mapping)
                            else None
                        ),
                        "raw_status": raw_status,
                    },
                )
            )
        return out

    async def _walk_real(
        self,
        provider: str,
        *,
        cursor: str | None = None,
        limit: int = 50,
    ) -> AsyncIterator[dict[str, Any]]:
        """Walk the provider's items via v3 tool execution.

        v3 endpoint: ``POST /api/v3/tools/execute/{TOOL_SLUG}``. Body:
        ``{"user_id": <user>, "arguments": {"limit": ..., "cursor": ...}}``.
        Response wrapper: ``{"data": <action-shape>, "error": ..., "successful": bool}``.
        The action's pagination key (``page_token`` vs ``next_cursor`` vs
        ``cursor``) is per-provider; we accept the union.
        """
        action = _LIST_ACTION_BY_PROVIDER.get(provider)
        if action is None:
            raise BackendError(
                f"walk: no Composio action mapping for provider {provider!r}",
                kind="unknown",
            )
        next_cursor: str | None = cursor
        # Safety net: never loop more than 1000 pages. Defensive against a
        # mis-behaving server returning the same cursor forever.
        for _ in range(1000):
            arguments: dict[str, Any] = {"limit": limit}
            # Layer in provider-specific required args (e.g. Calendar's calendarId).
            arguments.update(_DEFAULT_ARGUMENTS_BY_PROVIDER.get(provider, {}))
            if next_cursor is not None:
                arguments["cursor"] = next_cursor
            payload = await self._request(
                method="POST",
                path=f"/api/v3/tools/execute/{action}",
                json={"user_id": self._user_id, "arguments": arguments},
            )
            # v3 wraps every tool result in {data, error, successful}.
            if isinstance(payload, Mapping) and payload.get("successful") is False:
                err = payload.get("error") or "unknown error"
                raise BackendError(
                    f"walk: tool execute {action} returned error: {err}",
                    kind="unknown",
                )
            # Unwrap the inner data envelope before passing to the parser
            # so the existing ``items`` / ``next_cursor`` extraction works
            # whether the action puts them at the top level or under
            # ``data``.
            inner = payload.get("data") if isinstance(payload, Mapping) else None
            items, next_cursor = _parse_walk_page(inner if isinstance(inner, Mapping) else payload)
            for item in items:
                yield dict(item)
            if next_cursor is None:
                return
        log.warning(
            "composio.walk_pagination_limit",
            extra={"extra_fields": {"provider": provider}},
        )

    # ------------------------------------------------------------------ #
    # HTTP plumbing                                                      #
    # ------------------------------------------------------------------ #

    async def _request(
        self,
        *,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
    ) -> Any:
        """Issue a single HTTP request with the bridge's retry policy.

        Retries are scheduled per :attr:`_retry_backoff` on 429 and any
        5xx response. 401 → ``BackendError(kind="auth")`` (not retried).
        Timeouts → ``BackendError(kind="timeout")``. Other transport
        errors → ``BackendError(kind="network")``.
        """
        import httpx

        url = f"{self._base_url}{path}"
        # +1 because the loop covers the initial attempt + len(backoff)
        # follow-up retries.
        max_attempts = len(self._retry_backoff) + 1
        client = await self._http_client()
        try:
            for attempt in range(max_attempts):
                try:
                    resp = await client.request(
                        method,
                        url,
                        params=params,
                        json=json,
                        headers=self._auth_headers(),
                    )
                except httpx.TimeoutException as exc:
                    raise BackendError(f"{method} {path}: timeout", kind="timeout") from exc
                except httpx.HTTPError as exc:
                    raise BackendError(
                        f"{method} {path}: transport error: {exc}",
                        kind="network",
                    ) from exc

                status = resp.status_code
                if status == 401:
                    raise BackendError(
                        f"{method} {path}: 401 unauthorised — check COMPOSIO_API_KEY",
                        kind="auth",
                    )
                if status == 429 or 500 <= status < 600:
                    # Retryable. If we've still got budget, sleep and
                    # retry; otherwise classify and raise.
                    if attempt < max_attempts - 1:
                        await self._sleep(self._retry_backoff[attempt])
                        continue
                    kind = "rate_limit" if status == 429 else "server"
                    raise BackendError(
                        f"{method} {path}: {status} after {max_attempts} attempts",
                        kind=kind,
                    )
                if status >= 400:
                    # 4xx other than 401/429 — not retried; surface so a
                    # follow-up PR can refine classification once we have
                    # a real account to test against.
                    raise BackendError(
                        f"{method} {path}: {status} {resp.text[:200]}",
                        kind="unknown",
                    )
                # 2xx / 3xx — return the decoded body.
                try:
                    return resp.json()
                except ValueError as exc:
                    raise BackendError(
                        f"{method} {path}: non-JSON 2xx response",
                        kind="server",
                    ) from exc
            # Defensive: the loop should always either return or raise.
            raise BackendError(
                f"{method} {path}: exhausted retries without resolution",
                kind="unknown",
            )
        finally:
            await client.aclose()

    def _auth_headers(self) -> dict[str, str]:
        if self._api_key is None:
            # Defensive: should never be hit because the stub-mode branches
            # never call this, but assert anyway so a future refactor fails
            # loud.
            raise RuntimeError("Composio API key required for real-mode call")
        return {
            "x-api-key": self._api_key,
            "User-Agent": "second-brain-wiki/wiki_integrations",
            "Accept": "application/json",
        }

    async def _http_client(self) -> httpx.AsyncClient:
        # Lazy import so the stub mode never needs httpx loaded; this also
        # keeps the import graph shallow for test consumers.
        import httpx

        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.DEFAULT_TIMEOUT_SECONDS, connect=5.0),
            transport=self._transport,
        )


# --------------------------------------------------------------------------- #
# Response-parsing helpers                                                    #
# --------------------------------------------------------------------------- #


def _parse_walk_page(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    """Pull ``items`` + ``next_cursor`` from a walk response.

    schema-best-effort: accepts ``{"data": {"items": [...], "next_cursor": "..."}}``,
    ``{"items": [...], "next_cursor": "..."}``, or a bare list. Returns
    an empty list / ``None`` cursor if the shape is unrecognisable so
    pagination terminates instead of looping forever.
    """
    if isinstance(payload, Mapping):
        data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
        if isinstance(data, Mapping):
            raw_items = data.get("items", [])
            next_cursor_raw = data.get("next_cursor")
        else:
            raw_items = []
            next_cursor_raw = None
    elif isinstance(payload, list):
        raw_items = payload
        next_cursor_raw = None
    else:
        raw_items = []
        next_cursor_raw = None

    items: list[dict[str, Any]] = [dict(item) for item in raw_items if isinstance(item, Mapping)]
    next_cursor = str(next_cursor_raw) if next_cursor_raw else None
    return items, next_cursor


__all__ = ["BackendError", "ComposioBridge", "ComposioConnection"]
