"""
Ollama (local) backend per [PRD-008 FR-6](../../../docs/PRD-008-model-routing.md)
and [ADR-013 §"Local fallback (Ollama)"](../../../docs/ADR-013-model-router-policy.md).

Ollama is the **last link** in every tier's fallback chain — when the
network is down or all external providers fail, the agent degrades to a
local model running at ``http://localhost:11434``. ADR-013 requires this
fallback never bills cost; ``quote`` always returns ``$0.00`` so the
cost ceiling can never refuse it.

This is the M3-S3 real implementation: a plain ``httpx`` POST to
``/api/chat`` with ``stream: false``. Failure classification:

- ``httpx.ConnectError`` → ``BackendError(kind="server")`` first attempt,
  then ``BackendError(kind="unavailable")`` after retries exhaust.
  ("Connection refused" usually means the Ollama daemon is down; the
  router walks to no further backend because this provider is itself
  the last link, but the kind="unavailable" is preserved for telemetry.)
- ``httpx.TimeoutException`` → ``BackendError(kind="timeout")`` (retried)
- HTTP 5xx → ``BackendError(kind="server")`` (retried)
- HTTP 4xx → non-retryable ``RuntimeError`` (programmer bug, e.g. model
  not pulled; bubbles out so the fallback chain doesn't paper over it)
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from wiki_routing.cost_ceiling import CostQuote
from wiki_routing.fallback import BackendError
from wiki_routing.router import BackendResponse, Message


def _classify_status(status: int) -> BackendError | Exception:
    """Map an HTTP status code to the right exception type.

    Ollama is local, so a 4xx is almost always "model not pulled" or
    "bad payload" — not something the fallback chain should retry."""
    if 500 <= status < 600:
        return BackendError(f"ollama server error (HTTP {status})", kind="server")
    return RuntimeError(f"ollama client error (HTTP {status})")


class OllamaProvider:
    """Real ``ModelBackend`` for local Ollama.

    Parameters
    ----------
    model:
        Ollama model name (e.g. ``"llama3.2"``, ``"qwen2.5:14b"``,
        ``"llava:13b"``). Embedded in ``label``.
    base_url:
        Ollama daemon root. Defaults to ``http://localhost:11434`` —
        the daemon's standard listen address.
    timeout_seconds:
        Per-request timeout. Defaults to 120s (local models on modest
        hardware can be slow; a too-tight timeout makes the offline
        fallback brittle).
    max_retries:
        Number of retries on retryable failures (connection refused,
        timeouts, 5xx). Defaults to 1 (so 2 total attempts) — the
        daemon's local so most retries beyond one are pointless.
    transport:
        Optional ``httpx`` transport. Tests pass an ``httpx.MockTransport``;
        production leaves it ``None``.
    """

    def __init__(
        self,
        *,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 120.0,
        max_retries: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.label = f"ollama:{model}"
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._transport = transport

    # ------------------------------------------------------------------ #
    # Public Protocol surface                                            #
    # ------------------------------------------------------------------ #

    async def generate(self, messages: list[Message]) -> BackendResponse:
        """POST ``/api/chat`` with ``stream: false`` and a short retry
        loop for the connection-refused case.

        Returns a ``BackendResponse`` populated from the daemon's
        ``message.content`` + ``prompt_eval_count`` / ``eval_count``
        token counts. ``cost_usd`` is always ``0.0`` — local is free."""
        body = self._build_body(messages)
        attempts = self._max_retries + 1
        last_err: BackendError | None = None

        for attempt in range(attempts):
            try:
                data = await self._post(body)
            except BackendError as exc:
                last_err = exc
                if attempt >= self._max_retries:
                    # Final attempt: if this was connection refused,
                    # promote to "unavailable" so the router's telemetry
                    # can distinguish a flaky network from a missing
                    # daemon.
                    if exc.kind == "network":
                        raise BackendError(
                            "ollama daemon unavailable after retries",
                            kind="unavailable",
                        ) from exc
                    raise
                await self._sleep_for_backoff(attempt)
                continue
            return self._parse_response(data)

        # Defensive: loop always returns or raises.
        assert last_err is not None
        raise last_err  # pragma: no cover

    def quote(self, messages: list[Message]) -> CostQuote:
        """Local is free. Always returns ``$0.00``."""
        return CostQuote(estimated_usd=0.0, backend_label=self.label)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_body(self, messages: list[Message]) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": 0.2},
        }

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Single HTTP attempt."""
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                resp = await client.post("/api/chat", json=body)
        except httpx.TimeoutException as exc:
            raise BackendError("ollama request timed out", kind="timeout") from exc
        except httpx.ConnectError as exc:
            # Connection refused — kind="network" on the inner attempt;
            # promoted to "unavailable" by ``generate`` after retries.
            raise BackendError("ollama connection refused", kind="network") from exc

        if resp.status_code != 200:
            err = _classify_status(resp.status_code)
            raise err

        return resp.json()

    def _parse_response(self, data: dict[str, Any]) -> BackendResponse:
        """Extract the assistant text + token counts from Ollama's
        ``/api/chat`` JSON shape (with ``stream: false``)."""
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise BackendError(
                "ollama response missing message.content",
                kind="unknown",
            ) from exc

        tokens_in = int(data.get("prompt_eval_count", 0))
        tokens_out = int(data.get("eval_count", 0))
        return BackendResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
        )

    async def _sleep_for_backoff(self, attempt: int) -> None:
        """Constant short backoff — the daemon's local, so exponential
        backoff doesn't buy us much. 0.25s × (attempt + 1)."""
        await asyncio.sleep(0.25 * (attempt + 1))


# Backwards-compatible alias so existing M3-S1 imports keep working
# until the rest of the codebase migrates to the ``Provider`` naming
# that M3-S3 standardises on.
OllamaBackend = OllamaProvider


__all__ = ["OllamaBackend", "OllamaProvider"]
