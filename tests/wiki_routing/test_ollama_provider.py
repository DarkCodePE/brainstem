"""Tests for ``wiki_routing.providers.ollama.OllamaProvider``.

Every test routes through ``httpx.MockTransport`` — the real Ollama
daemon at ``localhost:11434`` is never contacted. Backoff sleeps are
patched to zero so retry tests stay fast."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from wiki_routing.fallback import BackendError
from wiki_routing.providers.ollama import OllamaProvider
from wiki_routing.router import Message, ModelBackend


def _msgs(*pairs: tuple[str, str]) -> list[Message]:
    if not pairs:
        pairs = (("user", "hi"),)
    return [Message(role=r, content=c) for r, c in pairs]


def _ok_payload(text: str = "hello", prompt: int = 10, eval_: int = 5) -> dict[str, Any]:
    return {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": text},
        "prompt_eval_count": prompt,
        "eval_count": eval_,
        "done": True,
    }


def _transport(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the backoff sleep to zero — keeps retry tests fast."""

    async def _instant(*_a: object, **_kw: object) -> None:
        return None

    monkeypatch.setattr("wiki_routing.providers.ollama.asyncio.sleep", _instant)


class TestConstruction:
    def test_label_includes_model(self) -> None:
        p = OllamaProvider(model="qwen2.5:14b")
        assert p.label == "ollama:qwen2.5:14b"

    def test_no_api_key_required(self) -> None:
        # Plain construction without auth — Ollama is local.
        p = OllamaProvider()
        assert p.label == "ollama:llama3.2"

    def test_negative_retries_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            OllamaProvider(max_retries=-1)

    def test_protocol_conformance(self) -> None:
        p = OllamaProvider()
        assert isinstance(p, ModelBackend)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_happy_path_returns_text_and_token_counts(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["url"] = str(request.url)
            captured["payload"] = json.loads(request.read())
            return httpx.Response(200, json=_ok_payload("greetings", 33, 21))

        p = OllamaProvider(transport=_transport(handler))
        resp = await p.generate(_msgs(("user", "ping")))

        assert resp.text == "greetings"
        assert resp.tokens_in == 33
        assert resp.tokens_out == 21
        assert resp.cost_usd == 0.0
        assert captured["url"].endswith("/api/chat")
        assert captured["payload"]["model"] == "llama3.2"
        assert captured["payload"]["stream"] is False
        assert captured["payload"]["options"]["temperature"] == 0.2
        assert captured["payload"]["messages"] == [{"role": "user", "content": "ping"}]

    @pytest.mark.asyncio
    async def test_prompt_eval_count_maps_to_tokens_in(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "ok"},
                    "prompt_eval_count": 99,
                    "eval_count": 1,
                },
            )

        p = OllamaProvider(transport=_transport(handler))
        resp = await p.generate(_msgs())
        assert resp.tokens_in == 99
        assert resp.tokens_out == 1


class TestConnectionRefused:
    @pytest.mark.asyncio
    async def test_connection_refused_promotes_to_unavailable(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectError("Connection refused")

        p = OllamaProvider(transport=_transport(handler), max_retries=1)
        with pytest.raises(BackendError) as ei:
            await p.generate(_msgs())
        # Final attempt promotes network → unavailable for telemetry.
        assert ei.value.kind == "unavailable"
        # Two attempts: 1 initial + 1 retry.
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_connection_refused_first_attempt_then_recovers(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ConnectError("Connection refused")
            return httpx.Response(200, json=_ok_payload("up now"))

        p = OllamaProvider(transport=_transport(handler), max_retries=1)
        resp = await p.generate(_msgs())
        assert resp.text == "up now"
        assert calls["n"] == 2


class TestRetries:
    @pytest.mark.asyncio
    async def test_timeout_retried_then_fails(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ReadTimeout("slow")

        p = OllamaProvider(transport=_transport(handler), max_retries=1)
        with pytest.raises(BackendError) as ei:
            await p.generate(_msgs())
        assert ei.value.kind == "timeout"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_500_retried_then_fails(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        p = OllamaProvider(transport=_transport(handler), max_retries=1)
        with pytest.raises(BackendError) as ei:
            await p.generate(_msgs())
        assert ei.value.kind == "server"


class TestQuote:
    def test_quote_always_zero(self) -> None:
        p = OllamaProvider(model="llava:13b")
        msgs = _msgs(("user", "x" * 10_000))  # arbitrarily large input
        q = p.quote(msgs)
        assert q.estimated_usd == 0.0
        assert q.backend_label == "ollama:llava:13b"
