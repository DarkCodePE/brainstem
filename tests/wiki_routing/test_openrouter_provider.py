"""Tests for ``wiki_routing.providers.openrouter.OpenRouterProvider``.

All network I/O is intercepted by ``httpx.MockTransport`` — no real
HTTP request escapes the test runner. Backoff sleeps are patched to
zero so the retry tests don't blow up CI wall-clock time."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from wiki_routing.fallback import BackendError
from wiki_routing.providers.openrouter import OpenRouterProvider
from wiki_routing.router import Message, ModelBackend


def _msgs(*pairs: tuple[str, str]) -> list[Message]:
    if not pairs:
        pairs = (("user", "hello"),)
    return [Message(role=r, content=c) for r, c in pairs]


def _ok_payload(text: str = "hi back", prompt: int = 12, completion: int = 7) -> dict[str, Any]:
    return {
        "id": "gen-test",
        "model": "openai/gpt-4o",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def _transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the backoff sleep to zero — keeps retry tests fast."""

    async def _instant(*_a: object, **_kw: object) -> None:
        return None

    monkeypatch.setattr("wiki_routing.providers.openrouter.asyncio.sleep", _instant)


class TestConstruction:
    def test_empty_api_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            OpenRouterProvider(api_key="")

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            OpenRouterProvider(api_key="k", max_retries=-1)

    def test_label_includes_model(self) -> None:
        p = OpenRouterProvider(api_key="k", model="google/gemini-1.5-flash")
        assert p.label == "openrouter:google/gemini-1.5-flash"

    def test_protocol_conformance(self) -> None:
        p = OpenRouterProvider(api_key="k")
        assert isinstance(p, ModelBackend)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_happy_path_returns_text_and_usage(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = request.read()
            return httpx.Response(200, json=_ok_payload("hello world", 42, 17))

        p = OpenRouterProvider(api_key="sk-test", transport=_transport(handler))
        resp = await p.generate(_msgs(("user", "ping")))

        assert resp.text == "hello world"
        assert resp.tokens_in == 42
        assert resp.tokens_out == 17
        # Default gpt-4o prices: $5/M input + $15/M output.
        expected_cost = 42 * 5.0 / 1_000_000 + 17 * 15.0 / 1_000_000
        assert resp.cost_usd == pytest.approx(expected_cost)
        # Endpoint + headers.
        assert captured["url"].endswith("/chat/completions")
        assert captured["headers"]["authorization"] == "Bearer sk-test"
        assert "second-brain-wiki" in captured["headers"]["http-referer"]
        assert captured["headers"]["x-title"] == "Second Brain Wiki"

    @pytest.mark.asyncio
    async def test_system_and_user_messages_passed_through(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            seen["payload"] = json.loads(request.read())
            return httpx.Response(200, json=_ok_payload())

        p = OpenRouterProvider(api_key="sk", transport=_transport(handler))
        await p.generate(_msgs(("system", "be terse"), ("user", "hi")))

        msgs = seen["payload"]["messages"]
        assert msgs == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
        assert seen["payload"]["model"] == "openai/gpt-4o"
        assert seen["payload"]["temperature"] == 0.2
        assert seen["payload"]["max_tokens"] == 2048


class TestLocalBilling:
    """A loopback ``base_url`` (local Gemma/llama-server) is billed at $0 —
    ADR-040 D1: local synthesis has zero marginal cost. Cloud calls still
    accrue against the budget."""

    @pytest.mark.asyncio
    async def test_local_base_url_costs_zero(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_payload("local", 999, 999))

        p = OpenRouterProvider(
            api_key="sk-local",
            model="gemma-4-12B-it-qat",
            base_url="http://127.0.0.1:60141/v1",
            transport=_transport(handler),
        )
        resp = await p.generate(_msgs(("user", "ping")))
        # Real tokens reported, but cost zeroed for a local backend.
        assert resp.tokens_in == 999 and resp.tokens_out == 999
        assert resp.cost_usd == 0.0
        # The pre-call quote is $0 too, so the budget never pre-rejects local.
        assert p.quote(_msgs(("user", "x"))).estimated_usd == 0.0

    @pytest.mark.asyncio
    async def test_localhost_and_ipv6_loopback_also_zero(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_payload("x", 100, 100))

        for base in ("http://localhost:8080/v1", "http://[::1]:8080/v1"):
            p = OpenRouterProvider(api_key="k", base_url=base, transport=_transport(handler))
            resp = await p.generate(_msgs())
            assert resp.cost_usd == 0.0, base

    @pytest.mark.asyncio
    async def test_cloud_base_url_still_charges(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_ok_payload("cloud", 42, 17))

        # Default cloud base_url — must NOT be zeroed.
        p = OpenRouterProvider(api_key="sk", transport=_transport(handler))
        resp = await p.generate(_msgs())
        expected = 42 * 5.0 / 1_000_000 + 17 * 15.0 / 1_000_000
        assert resp.cost_usd == pytest.approx(expected)
        assert resp.cost_usd > 0.0


class TestRetries:
    @pytest.mark.asyncio
    async def test_429_retried_then_succeeds(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=_ok_payload("recovered"))

        p = OpenRouterProvider(api_key="k", transport=_transport(handler), max_retries=2)
        resp = await p.generate(_msgs())
        assert resp.text == "recovered"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_500_retried_then_fails(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500, json={"error": "boom"})

        p = OpenRouterProvider(api_key="k", transport=_transport(handler), max_retries=2)
        with pytest.raises(BackendError) as ei:
            await p.generate(_msgs())
        assert ei.value.kind == "server"
        # 1 initial + 2 retries = 3 total attempts.
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_timeout_raises_backend_error_timeout(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow upstream")

        p = OpenRouterProvider(api_key="k", transport=_transport(handler), max_retries=0)
        with pytest.raises(BackendError) as ei:
            await p.generate(_msgs())
        assert ei.value.kind == "timeout"


class TestNonRetryableErrors:
    @pytest.mark.asyncio
    async def test_401_raises_permission_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "bad key"})

        p = OpenRouterProvider(api_key="k", transport=_transport(handler), max_retries=3)
        # Auth errors should NOT be retried and should NOT be BackendError —
        # the fallback chain must not paper over a misconfigured key.
        with pytest.raises(PermissionError):
            await p.generate(_msgs())

    @pytest.mark.asyncio
    async def test_400_raises_runtime_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad body"})

        p = OpenRouterProvider(api_key="k", transport=_transport(handler), max_retries=3)
        with pytest.raises(RuntimeError):
            await p.generate(_msgs())


class TestQuote:
    def test_default_price_table(self) -> None:
        p = OpenRouterProvider(api_key="k")
        # 80 chars / 4 chars-per-token = 20 input tokens.
        msgs = _msgs(("user", "x" * 80))
        q = p.quote(msgs)
        # 20 tokens * $5/M + 256 assumed out * $15/M, * 1.10 safety margin.
        expected = (20 * 5.0 / 1_000_000 + 256 * 15.0 / 1_000_000) * 1.10
        assert q.estimated_usd == pytest.approx(expected)
        assert q.backend_label == "openrouter:openai/gpt-4o"

    def test_custom_price_table_used(self) -> None:
        p = OpenRouterProvider(api_key="k", prices={"input": 1.0, "output": 2.0})
        q = p.quote(_msgs(("user", "x" * 80)))
        # 20 tokens * $1/M + 256 * $2/M, * 1.10.
        expected = (20 * 1.0 / 1_000_000 + 256 * 2.0 / 1_000_000) * 1.10
        assert q.estimated_usd == pytest.approx(expected)
