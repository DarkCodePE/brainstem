"""
OpenRouter backend per [ADR-013 §"Provider fallback chain"](../../../docs/ADR-013-model-router-policy.md)
and [SPEC-009 §"M3 Sprint 3"](../../../docs/SPEC-009-model-router.md).

OpenRouter brokers OpenAI-compatible requests to many upstream providers
(Gemini, Anthropic, Llama, ...). This is the M3-S3 real implementation:
an ``httpx.AsyncClient`` POST to ``/chat/completions`` with bearer auth +
OpenRouter's attribution headers. Failure classification mirrors
``AnthropicBackend``:

- HTTP 429 → ``BackendError(kind="rate_limit")`` (retried with backoff)
- HTTP 5xx → ``BackendError(kind="server")`` (retried)
- ``httpx.TimeoutException`` → ``BackendError(kind="timeout")`` (retried)
- HTTP 401/403 → non-retryable ``PermissionError`` (auth bug, fall through)
- Other 4xx → non-retryable ``RuntimeError`` (programmer bug, fall through)

Authentication errors are deliberately **not** ``BackendError`` — ADR-013
§"Provider health tracking" requires the fallback chain to not paper
over a misconfigured key.

### Pricing

``quote`` uses a small in-memory price table. Defaults track the
midpoint of OpenRouter's gpt-4o pricing (input $5/M, output $15/M); pass
a custom ``prices`` dict to override per-deployment. The post-call
``cost_usd`` is computed from the *real* token counts reported in
``usage`` so pricing drift only affects the pre-call quote, never the
budget charge.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from wiki_routing.cost_ceiling import CostQuote
from wiki_routing.fallback import BackendError
from wiki_routing.router import BackendResponse, Message

# Default per-million-token pricing. OpenRouter's published gpt-4o
# pricing midpoint as of 2026-05; override via the ``prices`` kwarg
# when wiring a non-default model.
_DEFAULT_PRICE_PER_M_INPUT_USD = 5.0
_DEFAULT_PRICE_PER_M_OUTPUT_USD = 15.0
_OUTPUT_TOKENS_ASSUMED = 256
"""How many output tokens ``quote`` assumes pre-call. 10% safety margin
applied at the construction site (ADR-013 §"Cost ceiling enforcement")."""

_DEFAULT_REFERER = "https://github.com/DarkCodePE/second-brain-wiki"
_DEFAULT_TITLE = "Second Brain Wiki"


def _is_loopback_base_url(base_url: str) -> bool:
    """True iff ``base_url`` targets a LOCAL OpenAI-compatible server.

    Mirrors ``factory._is_local_openrouter`` so a local Gemma/llama-server
    primary is billed at **$0** (ADR-040 D1: local synthesis has zero marginal
    cost). The cloud price table over-charges otherwise — real money is $0 but
    the budget/telemetry would still accrue cost for a local call."""
    u = base_url.lower()
    return "127.0.0.1" in u or "localhost" in u or "::1" in u


def _estimate_tokens(messages: list[Message]) -> int:
    """4 chars/token heuristic. The router only uses this for the
    pre-call budget; post-call we always use the provider's count."""
    return max(1, sum(len(m.content) for m in messages) // 4)


def _classify_status(status: int) -> BackendError | Exception:
    """Map an HTTP status code to the right exception type.

    Retryable conditions become ``BackendError`` (the fallback chain
    walks); auth/programmer errors become plain exceptions (bubble out).
    """
    if status == 429:
        return BackendError(f"openrouter rate-limited (HTTP {status})", kind="rate_limit")
    if 500 <= status < 600:
        return BackendError(f"openrouter server error (HTTP {status})", kind="server")
    if status in (401, 403):
        return PermissionError(f"openrouter auth failed (HTTP {status})")
    # Generic 4xx — programmer error (bad model name, malformed body, etc.)
    return RuntimeError(f"openrouter client error (HTTP {status})")


class OpenRouterProvider:
    """Real ``ModelBackend`` for OpenRouter's REST API.

    Parameters
    ----------
    api_key:
        OpenRouter API key. Stored only on ``__init__``; never logged
        or serialised.
    model:
        OpenRouter model slug (e.g. ``"openai/gpt-4o"``,
        ``"google/gemini-1.5-flash"``). Embedded in ``label``.
    base_url:
        API root; defaults to OpenRouter's public endpoint.
    timeout_seconds:
        Per-request timeout. Defaults to 60s (OpenRouter brokers some
        slow upstreams; this is the recommended floor).
    max_retries:
        Number of retries on retryable failures (429/5xx/timeout).
        Defaults to 2 (so 3 total attempts). Set to 0 for tests that
        want deterministic single-shot behaviour.
    prices:
        Optional per-million-token price override. Keys
        ``"input"`` / ``"output"`` map to USD per million tokens.
        Falls back to gpt-4o defaults when omitted.
    transport:
        Optional ``httpx`` transport. Tests pass an ``httpx.MockTransport``
        here; production leaves it ``None`` so the default transport
        speaks to the real network.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "openai/gpt-4o",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        prices: dict[str, float] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        referer: str = _DEFAULT_REFERER,
        title: str = _DEFAULT_TITLE,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouterProvider requires a non-empty api_key")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.label = f"openrouter:{model}"
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        # Local (loopback) servers are billed at $0 — see ADR-040 D1.
        self._is_local = _is_loopback_base_url(self._base_url)
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._prices = {
            "input": _DEFAULT_PRICE_PER_M_INPUT_USD,
            "output": _DEFAULT_PRICE_PER_M_OUTPUT_USD,
        }
        if prices is not None:
            self._prices.update(prices)
        self._transport = transport
        self._referer = referer
        self._title = title

    # ------------------------------------------------------------------ #
    # Public Protocol surface                                            #
    # ------------------------------------------------------------------ #

    async def generate(self, messages: list[Message]) -> BackendResponse:
        """POST ``/chat/completions`` with exponential-backoff retries.

        Returns a ``BackendResponse`` populated from the provider's
        ``choices[0].message.content`` + ``usage`` payload. The
        ``cost_usd`` field uses the **real** token counts × the
        configured price table — even if OpenRouter later reports a
        ``cost`` field of its own, this code stays the source of truth
        for the budget."""
        body = self._build_body(messages)
        headers = self._build_headers()
        attempts = self._max_retries + 1
        last_err: BackendError | None = None

        for attempt in range(attempts):
            try:
                data = await self._post(body, headers)
            except BackendError as exc:
                last_err = exc
                if attempt >= self._max_retries:
                    raise
                await self._sleep_for_backoff(attempt)
                continue
            return self._parse_response(data)

        # Defensive: the loop above always either returns or raises.
        assert last_err is not None
        raise last_err  # pragma: no cover

    def quote(self, messages: list[Message]) -> CostQuote:
        """Pre-call cost estimate with a 10% safety margin (ADR-013)."""
        if self._is_local:
            return CostQuote(estimated_usd=0.0, backend_label=self.label)
        tokens_in = _estimate_tokens(messages)
        usd = (
            tokens_in * self._prices["input"] / 1_000_000.0
            + _OUTPUT_TOKENS_ASSUMED * self._prices["output"] / 1_000_000.0
        )
        return CostQuote(estimated_usd=usd * 1.10, backend_label=self.label)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _build_body(self, messages: list[Message]) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": 2048,
            "temperature": 0.2,
        }

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._referer,
            "X-Title": self._title,
            "Content-Type": "application/json",
        }

    async def _post(self, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """Single HTTP attempt. Raises ``BackendError`` for retryable
        failures and bubbles non-retryable exceptions to the caller."""
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                resp = await client.post("/chat/completions", json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise BackendError("openrouter request timed out", kind="timeout") from exc
        except httpx.ConnectError as exc:
            raise BackendError("openrouter connection refused", kind="network") from exc

        if resp.status_code != 200:
            err = _classify_status(resp.status_code)
            raise err

        return resp.json()

    def _parse_response(self, data: dict[str, Any]) -> BackendResponse:
        """Extract the assistant text + usage from the OpenAI-compatible
        JSON shape OpenRouter returns."""
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(
                "openrouter response missing choices[0].message.content",
                kind="unknown",
            ) from exc

        usage = data.get("usage", {}) or {}
        tokens_in = int(usage.get("prompt_tokens", 0))
        tokens_out = int(usage.get("completion_tokens", 0))
        # Local (loopback) backends have $0 marginal cost (ADR-040 D1); only
        # cloud calls accrue against the budget/telemetry.
        cost_usd = (
            0.0
            if self._is_local
            else (
                tokens_in * self._prices["input"] / 1_000_000.0
                + tokens_out * self._prices["output"] / 1_000_000.0
            )
        )
        return BackendResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )

    async def _sleep_for_backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter: 0.5s, 1s, 2s, ... + [0, 0.25]s."""
        base = 0.5 * (2**attempt)
        await asyncio.sleep(base + random.uniform(0.0, 0.25))


# Backwards-compatible alias so existing M3-S1 imports keep working
# until the rest of the codebase migrates to the ``Provider`` naming
# that M3-S3 standardises on.
OpenRouterBackend = OpenRouterProvider


__all__ = ["OpenRouterBackend", "OpenRouterProvider"]
