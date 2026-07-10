"""
Tests for ``wiki_integrations.composio_bridge.ComposioBridge``.

The bridge has two modes:

- **stub mode** — no ``COMPOSIO_API_KEY``; deterministic in-memory fixtures.
- **real mode** — API key present; httpx round-trips to Composio.

We test stub mode end-to-end (deterministic, hermetic) and mock the
real-mode HTTP layer with ``httpx.MockTransport`` so we never hit the
network.

Coverage matrix:

| Behaviour                                  | Test                                    |
| ------------------------------------------ | --------------------------------------- |
| stub_mode True when env unset              | test_stub_mode_when_env_unset           |
| stub_mode True when api_key=None passed    | test_stub_mode_when_explicit_none       |
| stub_mode False when key passed            | test_real_mode_with_explicit_key        |
| stub_mode False when env set               | test_real_mode_when_env_set             |
| walk(gmail) yields default fixtures        | test_walk_yields_default_gmail_stub     |
| walk(github) yields default fixtures       | test_walk_yields_default_github_stub    |
| walk(unknown) yields nothing               | test_walk_unknown_provider_empty        |
| Custom stub_data overrides defaults        | test_custom_stub_data_overrides         |
| walk results are dict copies (no aliasing) | test_walk_returns_defensive_copies      |
| walk("") raises ValueError                 | test_walk_empty_provider_rejected       |
| connect() in stub mode marks connected     | test_connect_stub_mode                  |
| connect("") raises ValueError              | test_connect_empty_provider_rejected    |
| list_active() reports stub connections     | test_list_active_includes_connected     |
| list_active() starts empty                 | test_list_active_empty_by_default       |
| Stub-mode deterministic across calls       | test_stub_walk_is_deterministic         |
| Real-mode connect parses ConnectionResult  | test_real_mode_connect                  |
| Real-mode list_active empty                | test_real_mode_list_active_empty        |
| Real-mode list_active populated            | test_real_mode_list_active_populated    |
| Real-mode walk yields items                | test_real_mode_walk_yields_items        |
| Real-mode walk paginates with next_cursor  | test_real_mode_walk_handles_pagination  |
| Real-mode walk retries on 429              | test_real_mode_walk_handles_429_retry   |
| Real-mode walk raises auth on 401          | test_real_mode_walk_handles_401         |
| Real-mode walk raises timeout              | test_real_mode_walk_handles_timeout     |
| Stub mode still works when key unset       | test_stub_mode_still_works_when_key_unset |
| _auth_headers raises without key           | test_auth_headers_requires_key          |
| Explicit base_url is normalised            | test_real_mode_with_explicit_base_url   |
| stub_data does not flip stub_mode flag     | test_explicit_stub_data_keeps_real_mode_flag |
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from wiki_integrations.composio_bridge import (
    BackendError,
    ComposioBridge,
    ComposioConnection,
)


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests start with ``COMPOSIO_API_KEY`` unset unless explicitly set."""
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)


# --------------------------------------------------------------------------- #
# Mode introspection                                                          #
# --------------------------------------------------------------------------- #


def test_stub_mode_when_env_unset() -> None:
    bridge = ComposioBridge()
    assert bridge.stub_mode is True


def test_stub_mode_when_explicit_none() -> None:
    bridge = ComposioBridge(api_key=None)
    assert bridge.stub_mode is True


def test_real_mode_with_explicit_key() -> None:
    bridge = ComposioBridge(api_key="sk-test-123")
    assert bridge.stub_mode is False


def test_real_mode_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPOSIO_API_KEY", "sk-from-env")
    bridge = ComposioBridge()
    assert bridge.stub_mode is False


# --------------------------------------------------------------------------- #
# Stub-mode behaviour                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_walk_yields_default_gmail_stub() -> None:
    bridge = ComposioBridge()
    items: list[dict[str, Any]] = []
    async for item in bridge.walk("gmail"):
        items.append(item)
    assert len(items) == 2
    assert {it["id"] for it in items} == {"gmail-msg-001", "gmail-msg-002"}


@pytest.mark.asyncio
async def test_walk_yields_default_github_stub() -> None:
    bridge = ComposioBridge()
    items: list[dict[str, Any]] = []
    async for item in bridge.walk("github"):
        items.append(item)
    assert len(items) == 2
    kinds = {it["kind"] for it in items}
    assert kinds == {"issue", "pull_request"}


@pytest.mark.asyncio
async def test_walk_unknown_provider_empty() -> None:
    bridge = ComposioBridge()
    items: list[dict[str, Any]] = []
    async for item in bridge.walk("unknown"):
        items.append(item)
    assert items == []


@pytest.mark.asyncio
async def test_custom_stub_data_overrides() -> None:
    custom = {"gmail": [{"id": "custom-1", "from": "x@y", "body": "hi"}]}
    bridge = ComposioBridge(stub_data=custom)
    items: list[dict[str, Any]] = []
    async for item in bridge.walk("gmail"):
        items.append(item)
    assert items == [{"id": "custom-1", "from": "x@y", "body": "hi"}]


@pytest.mark.asyncio
async def test_walk_returns_defensive_copies() -> None:
    bridge = ComposioBridge()
    seen_first: list[dict[str, Any]] = []
    async for item in bridge.walk("gmail"):
        seen_first.append(item)
    # Mutate the dict the caller received.
    seen_first[0]["body"] = "POISONED"
    # Walk again; the upstream payload must not have been tainted.
    seen_second: list[dict[str, Any]] = []
    async for item in bridge.walk("gmail"):
        seen_second.append(item)
    assert seen_second[0]["body"] != "POISONED"


@pytest.mark.asyncio
async def test_walk_empty_provider_rejected() -> None:
    bridge = ComposioBridge()
    with pytest.raises(ValueError, match="provider"):
        async for _ in bridge.walk(""):
            pass


@pytest.mark.asyncio
async def test_connect_stub_mode() -> None:
    bridge = ComposioBridge()
    conn = await bridge.connect("gmail")
    assert isinstance(conn, ComposioConnection)
    assert conn.provider == "gmail"
    assert conn.status == "connected"
    assert conn.connection_id.startswith("stub-conn-")


@pytest.mark.asyncio
async def test_connect_empty_provider_rejected() -> None:
    bridge = ComposioBridge()
    with pytest.raises(ValueError):
        await bridge.connect("")


@pytest.mark.asyncio
async def test_list_active_empty_by_default() -> None:
    bridge = ComposioBridge()
    assert await bridge.list_active() == []


@pytest.mark.asyncio
async def test_list_active_includes_connected() -> None:
    bridge = ComposioBridge()
    await bridge.connect("gmail")
    await bridge.connect("github")
    actives = await bridge.list_active()
    providers = {c.provider for c in actives}
    assert providers == {"gmail", "github"}
    assert all(c.status == "connected" for c in actives)


@pytest.mark.asyncio
async def test_stub_walk_is_deterministic() -> None:
    bridge_a = ComposioBridge()
    bridge_b = ComposioBridge()
    items_a = [it async for it in bridge_a.walk("gmail")]
    items_b = [it async for it in bridge_b.walk("gmail")]
    assert items_a == items_b


# --------------------------------------------------------------------------- #
# Real-mode behaviour (httpx.MockTransport)                                   #
# --------------------------------------------------------------------------- #
#
# All real-mode tests below short-circuit network I/O with
# ``httpx.MockTransport``. The bridge accepts a ``transport=`` kwarg so
# every ``httpx.AsyncClient`` it builds attaches the mock. Retries are
# defanged via ``retry_backoff=(0.0, 0.0, 0.0)`` so test runtime stays sub-ms.


def _build_bridge(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "sk-test-real",
) -> ComposioBridge:
    """Build a real-mode bridge whose transport runs ``handler``."""
    transport = httpx.MockTransport(handler)
    return ComposioBridge(
        api_key=api_key,
        transport=transport,
        retry_backoff=(0.0, 0.0, 0.0),
        sleep=_noop_sleep,
    )


async def _noop_sleep(_seconds: float) -> None:
    """Test-only: skip actual delays so retries don't burn wall-clock."""
    return None


@pytest.mark.asyncio
async def test_real_mode_connect_v3_two_step_flow() -> None:
    """v3 connect issues TWO requests: list/create auth_config + POST /link.

    The bridge must:
    1. GET ``/api/v3/auth_configs?toolkit_slug=gmail`` to look for existing.
    2. If empty, POST ``/api/v3/auth_configs`` with
       ``{toolkit: {slug}, auth_config: {type: use_composio_managed_auth}}``.
    3. POST ``/api/v3/connected_accounts/link`` with
       ``{auth_config_id, user_id}``.
    4. Return ``ComposioConnection`` with the ``connected_account_id``
       Composio assigned + ``redirect_url`` for the user.
    """
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        path = request.url.path
        if path == "/api/v3/auth_configs" and request.method == "GET":
            # No existing auth_config → bridge must create one.
            return httpx.Response(200, json={"items": []})
        if path == "/api/v3/auth_configs" and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "toolkit": {"slug": "gmail"},
                    "auth_config": {
                        "id": "ac_test123",
                        "auth_scheme": "OAUTH2",
                        "is_composio_managed": True,
                    },
                },
            )
        if path == "/api/v3/connected_accounts/link" and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "link_token": "lk_test",
                    "redirect_url": "https://connect.composio.dev/link/lk_test",
                    "expires_at": "2026-05-27T23:00:00Z",
                    "connected_account_id": "ca_test_gmail",
                },
            )
        return httpx.Response(404)

    bridge = _build_bridge(handler)
    conn = await bridge.connect("gmail")

    assert conn.provider == "gmail"
    assert conn.connection_id == "ca_test_gmail"
    assert conn.status == "pending"
    assert conn.redirect_url == "https://connect.composio.dev/link/lk_test"
    assert conn.metadata["auth_config_id"] == "ac_test123"
    # Three calls: GET auth_configs (empty), POST auth_configs (create),
    # POST connected_accounts/link.
    assert [r.url.path for r in requests_seen] == [
        "/api/v3/auth_configs",
        "/api/v3/auth_configs",
        "/api/v3/connected_accounts/link",
    ]
    assert requests_seen[0].headers["x-api-key"] == "sk-test-real"


@pytest.mark.asyncio
async def test_real_mode_connect_reuses_existing_auth_config() -> None:
    """If Composio already has an auth_config for the toolkit, we reuse
    the ``ac_*`` id instead of creating a duplicate row."""
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        path = request.url.path
        if path == "/api/v3/auth_configs" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "toolkit": {"slug": "gmail"},
                            "auth_config": {
                                "id": "ac_existing",
                                "auth_scheme": "OAUTH2",
                                "is_composio_managed": True,
                            },
                        }
                    ]
                },
            )
        if path == "/api/v3/connected_accounts/link":
            return httpx.Response(
                201,
                json={
                    "link_token": "lk_x",
                    "redirect_url": "https://connect.composio.dev/link/lk_x",
                    "connected_account_id": "ca_x",
                },
            )
        return httpx.Response(404)

    bridge = _build_bridge(handler)
    conn = await bridge.connect("gmail")
    assert conn.metadata["auth_config_id"] == "ac_existing"
    # Exactly TWO calls — no auth_config creation.
    assert [r.url.path for r in requests_seen] == [
        "/api/v3/auth_configs",
        "/api/v3/connected_accounts/link",
    ]


@pytest.mark.asyncio
async def test_real_mode_connect_calendar_uses_googlecalendar_slug() -> None:
    """SBW provider ``calendar`` → Composio toolkit slug ``googlecalendar``."""
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"items": []})
        if request.url.path == "/api/v3/auth_configs":
            body = request.content.decode()
            # The request must use Composio's "googlecalendar" slug, not SBW's "calendar".
            assert '"slug": "googlecalendar"' in body or '"slug":"googlecalendar"' in body
            return httpx.Response(
                201,
                json={
                    "toolkit": {"slug": "googlecalendar"},
                    "auth_config": {"id": "ac_cal", "auth_scheme": "OAUTH2"},
                },
            )
        return httpx.Response(
            201,
            json={
                "connected_account_id": "ca_cal",
                "redirect_url": "https://x",
                "link_token": "lk",
            },
        )

    bridge = _build_bridge(handler)
    conn = await bridge.connect("calendar")
    # The returned ``provider`` keeps SBW's internal id.
    assert conn.provider == "calendar"
    assert conn.connection_id == "ca_cal"


@pytest.mark.asyncio
async def test_real_mode_list_active_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/api/v3/connected_accounts"
        return httpx.Response(200, json={"items": []})

    bridge = _build_bridge(handler)
    rows = await bridge.list_active()
    assert rows == []


@pytest.mark.asyncio
async def test_real_mode_list_active_populated() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "ca_gm",
                        "toolkit": {"slug": "gmail"},
                        "status": "ACTIVE",
                        "user_id": "sbw-local",
                        "auth_config": {"id": "ac_gm"},
                    },
                    {
                        "id": "ca_gh",
                        "toolkit": {"slug": "github"},
                        "status": "ACTIVE",
                        "user_id": "sbw-local",
                    },
                    {
                        "id": "ca_cal",
                        "toolkit": {"slug": "googlecalendar"},
                        "status": "INITIALIZING",
                        "user_id": "sbw-local",
                    },
                ]
            },
        )

    bridge = _build_bridge(handler)
    rows = await bridge.list_active()
    # ``list_active`` now filters to status=="connected" — issue #107 fix.
    # The INITIALIZING calendar row is excluded.
    assert len(rows) == 2
    by_provider = {r.provider: r for r in rows}
    assert by_provider["gmail"].status == "connected"
    assert by_provider["github"].status == "connected"
    assert "calendar" not in by_provider
    assert by_provider["gmail"].metadata["auth_config_id"] == "ac_gm"

    # The new ``list_connections`` returns ALL rows including INITIALIZING.
    all_rows = await bridge.list_connections()
    assert len(all_rows) == 3
    by_provider_all = {r.provider: r for r in all_rows}
    assert by_provider_all["calendar"].status == "initializing"


@pytest.mark.asyncio
async def test_real_mode_walk_yields_items_via_v3_tool_execute() -> None:
    """Walk hits ``POST /api/v3/tools/execute/{TOOL_SLUG}`` and unwraps the
    ``{data, error, successful}`` envelope before yielding items."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/v3/tools/execute/GMAIL_FETCH_EMAILS"
        body = request.content.decode()
        # v3 body: {"user_id": "...", "arguments": {...}}
        assert '"user_id"' in body
        assert '"arguments"' in body
        return httpx.Response(
            200,
            json={
                "data": {
                    "items": [
                        {"id": "real-1", "body": "hello"},
                        {"id": "real-2", "body": "world"},
                    ],
                    "next_cursor": None,
                },
                "error": None,
                "successful": True,
            },
        )

    bridge = _build_bridge(handler)
    items = [it async for it in bridge.walk("gmail")]
    assert [it["id"] for it in items] == ["real-1", "real-2"]


@pytest.mark.asyncio
async def test_real_mode_walk_handles_pagination() -> None:
    """Two-page walk: first page returns ``next_cursor``, second uses it
    in the request body and returns ``next_cursor=None``."""
    pages = [
        {
            "data": {
                "items": [{"id": "p1-a"}, {"id": "p1-b"}],
                "next_cursor": "cursor-after-page-1",
            },
            "error": None,
            "successful": True,
        },
        {
            "data": {
                "items": [{"id": "p2-a"}, {"id": "p2-b"}],
                "next_cursor": None,
            },
            "error": None,
            "successful": True,
        },
    ]
    requests_seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests_seen.append(request)
        page_idx = len(requests_seen) - 1
        return httpx.Response(200, json=pages[page_idx])

    bridge = _build_bridge(handler)
    items = [it async for it in bridge.walk("gmail")]

    assert [it["id"] for it in items] == ["p1-a", "p1-b", "p2-a", "p2-b"]
    assert len(requests_seen) == 2

    # Second request body must carry the cursor returned by the first page.
    second_body = requests_seen[1].content.decode()
    assert "cursor-after-page-1" in second_body
    # First request must NOT carry a cursor (initial page).
    first_body = requests_seen[0].content.decode()
    assert "cursor-after-page-1" not in first_body


@pytest.mark.asyncio
async def test_real_mode_walk_raises_when_successful_false() -> None:
    """A tool-execute that returns ``successful: false`` surfaces as
    ``BackendError`` — otherwise an action-level OAuth/perms error would
    silently swallow itself behind an empty items list."""
    from wiki_routing.fallback import BackendError

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": {},
                "error": "Insufficient permissions: missing gmail.readonly scope",
                "successful": False,
            },
        )

    bridge = _build_bridge(handler)
    with pytest.raises(BackendError):
        async for _ in bridge.walk("gmail"):
            pass


@pytest.mark.asyncio
async def test_real_mode_walk_handles_429_retry() -> None:
    """A 429 on the first attempt should be retried; the second 200 wins."""
    call_count = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={
                "data": {"items": [{"id": "after-429"}], "next_cursor": None},
                "error": None,
                "successful": True,
            },
        )

    bridge = _build_bridge(handler)
    items = [it async for it in bridge.walk("gmail")]
    assert items == [{"id": "after-429"}]
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_real_mode_walk_handles_401() -> None:
    """401 surfaces as a BackendError(kind="auth") — not retried."""
    call_count = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(401, json={"error": "invalid api key"})

    bridge = _build_bridge(handler)
    with pytest.raises(BackendError) as excinfo:
        [it async for it in bridge.walk("gmail")]
    assert excinfo.value.kind == "auth"
    # Auth is never retried.
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_real_mode_walk_handles_timeout() -> None:
    """A transport-level ``httpx.TimeoutException`` becomes BackendError(kind="timeout")."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=_request)

    bridge = _build_bridge(handler)
    with pytest.raises(BackendError) as excinfo:
        [it async for it in bridge.walk("gmail")]
    assert excinfo.value.kind == "timeout"


@pytest.mark.asyncio
async def test_real_mode_walk_handles_500_then_success() -> None:
    """5xx is retried like 429; cover the success-after-server-error path."""
    call_count = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(503, text="upstream surge")
        return httpx.Response(
            200,
            json={"data": {"items": [{"id": "after-503"}], "next_cursor": None}},
        )

    bridge = _build_bridge(handler)
    items = [it async for it in bridge.walk("github")]
    assert items == [{"id": "after-503"}]
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_real_mode_walk_exhausts_retries() -> None:
    """Persistent 429s past the retry budget raise BackendError(kind="rate_limit")."""
    call_count = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(429, text="still throttled")

    bridge = _build_bridge(handler)
    with pytest.raises(BackendError) as excinfo:
        [it async for it in bridge.walk("gmail")]
    assert excinfo.value.kind == "rate_limit"
    # 1 initial + 3 retries (retry_backoff=(0,0,0)) = 4 calls.
    assert call_count["n"] == 4


@pytest.mark.asyncio
async def test_stub_mode_still_works_when_key_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backwards-compat: no key + no transport == deterministic stub fixtures."""
    monkeypatch.delenv("COMPOSIO_API_KEY", raising=False)
    bridge = ComposioBridge()
    assert bridge.stub_mode is True

    # walk hits the in-memory fixture, not the network.
    gmail_items = [it async for it in bridge.walk("gmail")]
    assert len(gmail_items) == 2
    assert {it["id"] for it in gmail_items} == {"gmail-msg-001", "gmail-msg-002"}

    github_items = [it async for it in bridge.walk("github")]
    assert len(github_items) == 2

    # connect/list_active operate on the in-memory stub state.
    conn = await bridge.connect("gmail")
    assert conn.connection_id.startswith("stub-conn-")
    assert conn.status == "connected"
    actives = await bridge.list_active()
    assert {c.provider for c in actives} == {"gmail"}


# --------------------------------------------------------------------------- #
# Misc                                                                        #
# --------------------------------------------------------------------------- #


def test_auth_headers_requires_key() -> None:
    bridge = ComposioBridge()
    # Stub mode never invokes _auth_headers; force it to.
    with pytest.raises(RuntimeError, match="API key required"):
        bridge._auth_headers()


def test_real_mode_with_explicit_base_url() -> None:
    bridge = ComposioBridge(api_key="sk-x", base_url="https://staging.composio.dev/")
    # Trailing slash should be stripped so URL joining is unambiguous.
    assert bridge.base_url == "https://staging.composio.dev"


def test_explicit_stub_data_keeps_real_mode_flag() -> None:
    # ``stub_data`` only swaps the in-memory fixture; the bridge's mode
    # is still driven by whether an API key was configured. This guards
    # against accidentally flipping ``stub_mode`` based on the wrong field.
    bridge = ComposioBridge(
        api_key="sk-x",
        stub_data={"gmail": [{"id": "z"}]},
    )
    assert bridge.stub_mode is False


@pytest.mark.asyncio
async def test_execute_pins_latest_toolkit_version() -> None:
    """execute() must send ``version="latest"`` in the body. The API default
    is the base pin ``00000000_00``, whose LinkedIn toolkit sends a sunset
    ``LinkedIn-Version`` and gets HTTP 426 NONEXISTENT_VERSION (verified live
    2026-05-31). "latest" tracks Composio's current, supported toolkit."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"successful": True, "data": {"ok": True}})

    bridge = _build_bridge(handler)
    result = await bridge.execute(
        "linkedin", "LINKEDIN_CREATE_LINKED_IN_POST", {"commentary": "hi"}
    )

    assert result == {"ok": True}
    assert len(seen) == 1
    assert seen[0].url.path == "/api/v3/tools/execute/LINKEDIN_CREATE_LINKED_IN_POST"
    body = json.loads(seen[0].content)
    assert body["version"] == "latest"
    assert body["arguments"] == {"commentary": "hi"}
    assert "user_id" in body


@pytest.mark.asyncio
async def test_execute_accepts_version_override() -> None:
    """A caller may pin a specific toolkit version instead of 'latest'."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"successful": True, "data": {}})

    bridge = _build_bridge(handler)
    await bridge.execute("linkedin", "X", {}, version="20251027_00")
    assert json.loads(seen[0].content)["version"] == "20251027_00"
