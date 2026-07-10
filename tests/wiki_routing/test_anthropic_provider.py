"""Tests for ``wiki_routing.providers.anthropic.AnthropicBackend`` —
real Anthropic Messages API backend per [SPEC-009 §"M3 Sprint 3"](../../docs/SPEC-009-model-router.md).

Every test installs an ``httpx.MockTransport`` so **no real network
calls happen.** Tests that exercise the retry/backoff path patch
``asyncio.sleep`` so the suite stays sub-second.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from wiki_routing.cost_ceiling import CostQuote
from wiki_routing.fallback import BackendError
from wiki_routing.providers.anthropic import (
    AnthropicBackend,
    AnthropicProvider,
)
from wiki_routing.router import BackendResponse, Message, ModelBackend

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _ok_body(
    text: str = "hello",
    input_tokens: int = 5,
    output_tokens: int = 3,
) -> bytes:
    """Build a minimal 200 OK Anthropic Messages API body."""
    return json.dumps(
        {
            "content": [{"type": "text", "text": text}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }
    ).encode("utf-8")


def _make_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.MockTransport:
    """Tiny wrapper so each test reads as intent + handler pair."""
    return httpx.MockTransport(handler)


def _make_backend(
    *,
    transport: httpx.MockTransport,
    max_retries: int = 2,
    model: str = "claude-3-5-sonnet-latest",
    timeout_seconds: float = 60.0,
) -> AnthropicBackend:
    """Construct an ``AnthropicBackend`` wired to a mock transport."""
    return AnthropicBackend(
        api_key="test-key",
        model=model,
        base_url="https://api.anthropic.com",
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        transport=transport,
    )


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``asyncio.sleep`` with an instant no-op so the retry
    schedule (1s / 4s / 16s) doesn't slow the suite. We patch only the
    function the backend module looks up; pytest_asyncio's own usage
    of asyncio.sleep is untouched."""

    async def _fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("wiki_routing.providers.anthropic.asyncio.sleep", _fast_sleep)


# --------------------------------------------------------------------------- #
# Construction / protocol conformance                                         #
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_constructs_with_required_args(self) -> None:
        """``api_key`` + ``model`` is enough; defaults fill the rest."""
        backend = AnthropicBackend(api_key="sk-test", model="claude-3-5-sonnet-latest")
        assert backend.label == "anthropic:claude-3-5-sonnet-latest"

    def test_default_model_is_sonnet_latest(self) -> None:
        backend = AnthropicBackend(api_key="sk-test")
        assert "sonnet-latest" in backend.label

    def test_negative_max_retries_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            AnthropicBackend(api_key="sk-test", max_retries=-1)

    def test_non_positive_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            AnthropicBackend(api_key="sk-test", timeout_seconds=0.0)

    def test_anthropic_provider_alias(self) -> None:
        """``AnthropicProvider`` is an alias for ``AnthropicBackend`` —
        SPEC-009 uses the ``Provider`` name in prose."""
        assert AnthropicProvider is AnthropicBackend

    def test_satisfies_modelbackend_protocol(self) -> None:
        """Runtime isinstance check via ``@runtime_checkable``."""
        backend = AnthropicBackend(api_key="sk-test")
        assert isinstance(backend, ModelBackend)


# --------------------------------------------------------------------------- #
# Happy-path generate()                                                       #
# --------------------------------------------------------------------------- #


class TestGenerateHappyPath:
    @pytest.mark.asyncio
    async def test_returns_text_from_content_array(self) -> None:
        """200 OK with a single text block populates ``BackendResponse``."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_ok_body("hello"))

        backend = _make_backend(transport=_make_transport(handler))
        result = await backend.generate([Message(role="user", content="hi")])
        assert isinstance(result, BackendResponse)
        assert result.text == "hello"
        assert result.tokens_in == 5
        assert result.tokens_out == 3
        # cost_usd should match the sonnet pricing × token counts.
        # 5 in @ $3/M + 3 out @ $15/M = $0.000015 + $0.000045 = $0.00006
        assert result.cost_usd == pytest.approx(0.00006, rel=1e-6)

    @pytest.mark.asyncio
    async def test_sends_required_headers(self) -> None:
        """``x-api-key``, ``anthropic-version``, ``content-type`` are set."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, content=_ok_body())

        backend = _make_backend(transport=_make_transport(handler))
        await backend.generate([Message(role="user", content="hi")])

        headers = captured["headers"]
        assert headers["x-api-key"] == "test-key"
        assert headers["anthropic-version"] == "2023-06-01"
        # httpx lowercases header names; content-type is set by httpx but
        # also explicitly added by the backend.
        assert "application/json" in headers["content-type"]

    @pytest.mark.asyncio
    async def test_posts_to_v1_messages(self) -> None:
        """Path must be ``/v1/messages``."""
        captured_url: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_url.append(request.url.path)
            return httpx.Response(200, content=_ok_body())

        backend = _make_backend(transport=_make_transport(handler))
        await backend.generate([Message(role="user", content="hi")])
        assert captured_url == ["/v1/messages"]

    @pytest.mark.asyncio
    async def test_system_messages_concatenated(self) -> None:
        """Two ``system`` messages are joined into the top-level
        ``system`` field; user/assistant entries are passed through
        unchanged in the ``messages`` list."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=_ok_body())

        backend = _make_backend(transport=_make_transport(handler))
        await backend.generate(
            [
                Message(role="system", content="A"),
                Message(role="system", content="B"),
                Message(role="user", content="hello"),
            ]
        )

        body = captured["body"]
        assert body["system"] == "A\nB"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["model"] == "claude-3-5-sonnet-latest"

    @pytest.mark.asyncio
    async def test_no_system_field_when_no_system_messages(self) -> None:
        """No system messages → no ``system`` key in payload."""
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=_ok_body())

        backend = _make_backend(transport=_make_transport(handler))
        await backend.generate([Message(role="user", content="hi")])

        assert "system" not in captured["body"]

    @pytest.mark.asyncio
    async def test_unsupported_role_rejected(self) -> None:
        """A role outside system/user/assistant raises ``ValueError``
        before any HTTP call."""

        def handler(_: httpx.Request) -> httpx.Response:
            raise AssertionError("HTTP call should not have happened")

        backend = _make_backend(transport=_make_transport(handler))
        with pytest.raises(ValueError, match="unsupported message role"):
            await backend.generate([Message(role="tool", content="bad")])


# --------------------------------------------------------------------------- #
# Retry / backoff                                                             #
# --------------------------------------------------------------------------- #


class TestRetryBehaviour:
    @pytest.mark.asyncio
    async def test_429_retries_with_backoff(self) -> None:
        """One 429 then a 200 OK ⇒ final response is the 200."""
        responses = iter(
            [
                httpx.Response(429, content=b'{"error": "rate"}'),
                httpx.Response(200, content=_ok_body("retried")),
            ]
        )
        call_count = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return next(responses)

        backend = _make_backend(transport=_make_transport(handler), max_retries=2)
        result = await backend.generate([Message(role="user", content="hi")])
        assert result.text == "retried"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_500_retries_then_raises_server_error(self) -> None:
        """Two 500s with max_retries=1 ⇒ exhausted; raises
        ``BackendError(kind="server")``. ``max_retries=1`` allows
        exactly 2 attempts."""
        call_count = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(500, content=b'{"error": "boom"}')

        backend = _make_backend(transport=_make_transport(handler), max_retries=1)
        with pytest.raises(BackendError) as exc_info:
            await backend.generate([Message(role="user", content="hi")])
        assert exc_info.value.kind == "server"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_529_classified_as_overloaded(self) -> None:
        """HTTP 529 ⇒ ``BackendError.kind == "overloaded"``."""
        call_count = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(529, content=b'{"error": "overloaded"}')

        backend = _make_backend(transport=_make_transport(handler), max_retries=0)
        with pytest.raises(BackendError) as exc_info:
            await backend.generate([Message(role="user", content="hi")])
        assert exc_info.value.kind == "overloaded"
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_max_retries_zero_means_single_attempt(self) -> None:
        """``max_retries=0`` ⇒ exactly one HTTP call regardless of 5xx."""
        call_count = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(503, content=b'{"error": "down"}')

        backend = _make_backend(transport=_make_transport(handler), max_retries=0)
        with pytest.raises(BackendError):
            await backend.generate([Message(role="user", content="hi")])
        assert call_count["n"] == 1


# --------------------------------------------------------------------------- #
# Non-retryable errors                                                        #
# --------------------------------------------------------------------------- #


class TestNonRetryableErrors:
    @pytest.mark.asyncio
    async def test_401_raises_auth_error_no_retry(self) -> None:
        """HTTP 401 ⇒ ``BackendError(kind="auth")`` on first attempt."""
        call_count = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(401, content=b'{"error": "bad key"}')

        backend = _make_backend(transport=_make_transport(handler), max_retries=3)
        with pytest.raises(BackendError) as exc_info:
            await backend.generate([Message(role="user", content="hi")])
        assert exc_info.value.kind == "auth"
        # No retries on auth errors — exactly one call.
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_400_raises_client_error_no_retry(self) -> None:
        """Any non-429 4xx ⇒ ``BackendError(kind="client")``, no retry."""
        call_count = {"n": 0}

        def handler(_: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(400, content=b'{"error": "bad req"}')

        backend = _make_backend(transport=_make_transport(handler), max_retries=3)
        with pytest.raises(BackendError) as exc_info:
            await backend.generate([Message(role="user", content="hi")])
        assert exc_info.value.kind == "client"
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_auth_error_does_not_leak_api_key(self) -> None:
        """Refusal messages must not echo the ``api_key`` value."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(401, content=b'{"error": "bad key"}')

        backend = AnthropicBackend(
            api_key="sk-super-secret-xyz",
            transport=_make_transport(handler),
        )
        with pytest.raises(BackendError) as exc_info:
            await backend.generate([Message(role="user", content="hi")])
        assert "sk-super-secret-xyz" not in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Transport-level failures                                                    #
# --------------------------------------------------------------------------- #


class TestTransportFailures:
    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self) -> None:
        """``httpx.TimeoutException`` ⇒ ``BackendError(kind="timeout")``."""

        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("simulated timeout")

        backend = _make_backend(transport=_make_transport(handler), max_retries=0)
        with pytest.raises(BackendError) as exc_info:
            await backend.generate([Message(role="user", content="hi")])
        assert exc_info.value.kind == "timeout"

    @pytest.mark.asyncio
    async def test_timeout_is_retried(self) -> None:
        """Timeout then success ⇒ caller sees the success."""
        calls: list[str] = []

        def handler(_: httpx.Request) -> httpx.Response:
            calls.append("call")
            if len(calls) == 1:
                raise httpx.TimeoutException("simulated timeout")
            return httpx.Response(200, content=_ok_body("recovered"))

        backend = _make_backend(transport=_make_transport(handler), max_retries=2)
        result = await backend.generate([Message(role="user", content="hi")])
        assert result.text == "recovered"
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_network_error_raises_network_kind(self) -> None:
        """``httpx.ConnectError`` (a ``TransportError``) ⇒
        ``BackendError(kind="network")``."""

        def handler(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("conn refused")

        backend = _make_backend(transport=_make_transport(handler), max_retries=0)
        with pytest.raises(BackendError) as exc_info:
            await backend.generate([Message(role="user", content="hi")])
        assert exc_info.value.kind == "network"


# --------------------------------------------------------------------------- #
# Quote / pricing                                                             #
# --------------------------------------------------------------------------- #


class TestQuote:
    def test_quote_uses_price_table_sonnet(self) -> None:
        """4000-char message → ~1000 input tokens at sonnet pricing.

        Math: 1000 tokens * $3/M = $0.003 input + 256 * $15/M = $0.00384
        output. Total $0.00684 × 1.10 safety margin ≈ $0.00752.
        """
        backend = AnthropicBackend(api_key="sk-test", model="claude-3-5-sonnet-latest")
        msg = Message(role="user", content="x" * 4000)
        quote = backend.quote([msg])
        assert isinstance(quote, CostQuote)
        # Estimated input tokens ≈ 1000 (4 chars/token).
        expected = ((1000 * 3.0 / 1_000_000.0) + (256 * 15.0 / 1_000_000.0)) * 1.10
        assert quote.estimated_usd == pytest.approx(expected, rel=1e-4)
        assert quote.backend_label == "anthropic:claude-3-5-sonnet-latest"

    def test_quote_uses_opus_pricing_when_named(self) -> None:
        """Opus is roughly 5× more expensive than sonnet on input."""
        opus = AnthropicBackend(api_key="sk-test", model="claude-opus-4-5-20250929")
        sonnet = AnthropicBackend(api_key="sk-test", model="claude-3-5-sonnet-latest")
        msg = Message(role="user", content="x" * 4000)
        assert opus.quote([msg]).estimated_usd > sonnet.quote([msg]).estimated_usd
        # Opus input is exactly 5× sonnet input; check ratio is in the
        # right neighbourhood (assumed-output adds a constant).
        assert opus.quote([msg]).estimated_usd / sonnet.quote([msg]).estimated_usd > 3.0

    def test_quote_haiku_cheapest(self) -> None:
        """Haiku is the cheapest tier in the table."""
        haiku = AnthropicBackend(api_key="sk-test", model="claude-haiku-3-5-20241022")
        sonnet = AnthropicBackend(api_key="sk-test", model="claude-3-5-sonnet-latest")
        msg = Message(role="user", content="x" * 4000)
        assert haiku.quote([msg]).estimated_usd < sonnet.quote([msg]).estimated_usd

    def test_unknown_model_falls_back_to_sonnet_pricing(self) -> None:
        """An unrecognised model name uses sonnet-equivalent pricing —
        the safe pick (over-quote rather than under-quote)."""
        unknown = AnthropicBackend(api_key="sk-test", model="claude-something-new-2027")
        sonnet = AnthropicBackend(api_key="sk-test", model="claude-3-5-sonnet-latest")
        msg = Message(role="user", content="x" * 4000)
        assert unknown.quote([msg]).estimated_usd == pytest.approx(
            sonnet.quote([msg]).estimated_usd, rel=1e-9
        )


# --------------------------------------------------------------------------- #
# Latency envelope                                                            #
# --------------------------------------------------------------------------- #


class TestLatency:
    @pytest.mark.asyncio
    async def test_latency_envelope_uses_perf_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The backend wraps the HTTP call in ``time.perf_counter()``
        so a future ``BackendResponse.latency_ms`` field can read a
        real wall-clock value. We verify by patching ``perf_counter``
        to return a strictly-increasing sequence and asserting both
        ends are observed."""
        # Supply a generous tick stream — asyncio internals may call
        # perf_counter several times around the HTTP/transport boundary
        # depending on the event-loop implementation. We only care that
        # the *first* tick (envelope start) and at least one later tick
        # (envelope end) are observed.
        ticks = iter([100.0 + (i * 0.001) for i in range(64)])
        observed: list[float] = []

        def fake_perf_counter() -> float:
            value = next(ticks)
            observed.append(value)
            return value

        monkeypatch.setattr(
            "wiki_routing.providers.anthropic.time.perf_counter",
            fake_perf_counter,
        )

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_ok_body())

        backend = _make_backend(transport=_make_transport(handler), max_retries=0)
        result = await backend.generate([Message(role="user", content="hi")])
        # The envelope start at 100.0 was observed; at least one later
        # tick was observed (envelope end after the HTTP call).
        assert observed[0] == 100.0
        assert len(observed) >= 2
        assert result.text == "hello"


# --------------------------------------------------------------------------- #
# Response parsing edge cases                                                 #
# --------------------------------------------------------------------------- #


class TestResponseParsing:
    @pytest.mark.asyncio
    async def test_missing_usage_block_yields_zeros(self) -> None:
        """A 200 OK body without ``usage`` ⇒ tokens default to 0."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode("utf-8"),
            )

        backend = _make_backend(transport=_make_transport(handler))
        result = await backend.generate([Message(role="user", content="hi")])
        assert result.text == "ok"
        assert result.tokens_in == 0
        assert result.tokens_out == 0
        assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_empty_content_array_raises_server_error(self) -> None:
        """200 OK with ``content: []`` is server-side malformed."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=json.dumps({"content": [], "usage": {}}).encode("utf-8"),
            )

        backend = _make_backend(transport=_make_transport(handler))
        with pytest.raises(BackendError) as exc_info:
            await backend.generate([Message(role="user", content="hi")])
        assert exc_info.value.kind == "server"

    @pytest.mark.asyncio
    async def test_multiple_text_blocks_concatenated(self) -> None:
        """Several ``type=text`` blocks ⇒ joined into a single string."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=json.dumps(
                    {
                        "content": [
                            {"type": "text", "text": "hello "},
                            {"type": "text", "text": "world"},
                        ],
                        "usage": {"input_tokens": 1, "output_tokens": 2},
                    }
                ).encode("utf-8"),
            )

        backend = _make_backend(transport=_make_transport(handler))
        result = await backend.generate([Message(role="user", content="hi")])
        assert result.text == "hello world"

    @pytest.mark.asyncio
    async def test_non_text_blocks_ignored(self) -> None:
        """``tool_use`` and other non-text blocks are skipped."""

        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=json.dumps(
                    {
                        "content": [
                            {"type": "tool_use", "id": "x", "name": "y"},
                            {"type": "text", "text": "only text counts"},
                        ],
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ).encode("utf-8"),
            )

        backend = _make_backend(transport=_make_transport(handler))
        result = await backend.generate([Message(role="user", content="hi")])
        assert result.text == "only text counts"
