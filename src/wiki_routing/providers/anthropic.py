"""
Anthropic backend per [ADR-013 §"Provider fallback chain"](../../../docs/ADR-013-model-router-policy.md)
and [SPEC-009 §"M3 Sprint 3 — real Anthropic backend"](../../../docs/SPEC-009-model-router.md).

This module wraps the Anthropic Messages REST API
(``POST /v1/messages``) behind the ``ModelBackend`` Protocol so the
``ModelRouter`` can dispatch chat completions to Claude (Sonnet / Opus
/ Haiku) without any further coupling.

### Why httpx instead of the ``anthropic`` SDK

We talk to the REST surface directly with ``httpx.AsyncClient``:

- Smallest dependency surface — ``httpx`` is already a transitive dep
  of ``wiki_agent.tools``; the official ``anthropic`` SDK would pin
  pydantic, packaging headers, and a couple of optional extras we
  don't need.
- The Messages API is a thin enough JSON surface that the savings
  from the SDK don't pay for the version-coupling cost (the SDK's
  client shape has rotated three times in 2025 alone).
- Easier to test: ``httpx.MockTransport`` lets the test suite intercept
  every byte without monkeypatching SDK internals.

If a later sprint needs streaming, tool-use, or vision parts, this
choice can be revisited — but the public Protocol shape stays.

### Failure classification

Mapping from HTTP / transport conditions to ``BackendError.kind`` so
the ``FallbackChain`` can step deterministically:

| Condition                         | ``BackendError.kind`` | Retry? |
|-----------------------------------|-----------------------|--------|
| HTTP 429                          | ``"rate_limit"``      | yes    |
| HTTP 529 (overloaded)             | ``"overloaded"``      | yes    |
| HTTP 5xx (other)                  | ``"server"``          | yes    |
| HTTP 401                          | ``"auth"``            | no     |
| HTTP 4xx (other)                  | ``"client"``          | no     |
| ``httpx.TimeoutException``        | ``"timeout"``         | yes    |
| ``httpx.TransportError``          | ``"network"``         | yes    |

Retry policy: exponential backoff at 1s, 4s, 16s, bounded by
``max_retries`` (default 2 — i.e. up to 3 total attempts). Auth and
non-429 4xx errors short-circuit without retry; they're configuration
or input bugs, not transient failures.

### Pricing

Built-in table (USD per 1M tokens) keyed by a substring match on the
model name; an unknown model falls back to Sonnet-equivalent pricing.
The table is co-located here (not in ``cost_ceiling``) because it's
Anthropic-specific — sibling providers carry their own. Update on
the quarterly review trigger called out in ADR-013.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from wiki_routing.cost_ceiling import CostQuote
from wiki_routing.fallback import BackendError
from wiki_routing.router import BackendResponse, Message

# --------------------------------------------------------------------------- #
# Pricing table — USD per *million* tokens                                    #
# --------------------------------------------------------------------------- #

# Anthropic published pricing (2025-Q4 snapshot). The router treats these
# as ground truth for ``BackendResponse.cost_usd`` and for the pre-call
# ``quote``. ADR-013 §"Cost ceiling enforcement" calls out a quarterly
# review of these numbers.
_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    # (input_usd_per_million, output_usd_per_million)
    "sonnet": (3.0, 15.0),
    "opus": (15.0, 75.0),
    "haiku": (0.80, 4.0),
}
"""Model-name substring → (input price, output price) in USD per 1M tokens.

We key on substrings (``"sonnet"`` matches both ``claude-3-5-sonnet-latest``
and ``claude-sonnet-4-5-20250929``) so the table stays stable across
quarterly model SKU rotations.
"""

_DEFAULT_PRICING: tuple[float, float] = _PRICING_PER_MILLION["sonnet"]
"""Fallback pricing for unrecognised model names. Sonnet-equivalent is
the safe pick — quoting too high refuses budget rather than under-billing."""

_OUTPUT_TOKENS_ASSUMED = 256
"""Heuristic output-token count used by ``quote``. ADR-013 recommends a
10% safety margin on top; we apply it on the line that builds the quote."""


def _price_for_model(model: str) -> tuple[float, float]:
    """Substring lookup against ``_PRICING_PER_MILLION``.

    Returns the first matching entry's ``(input, output)`` USD per
    million tokens; falls back to ``_DEFAULT_PRICING`` when nothing
    matches."""
    model_lower = model.lower()
    for needle, prices in _PRICING_PER_MILLION.items():
        if needle in model_lower:
            return prices
    return _DEFAULT_PRICING


def _estimate_tokens(messages: list[Message]) -> int:
    """4 chars/token heuristic — matches ``wiki_memory.chunker`` so the
    cost ceiling's pre-call quote uses the same accounting as the chunker.
    """
    return max(1, sum(len(m.content) for m in messages) // 4)


# --------------------------------------------------------------------------- #
# Anthropic Messages API translation                                          #
# --------------------------------------------------------------------------- #

# Anthropic's Messages API splits ``system`` out of the chat history:
# - top-level ``system: str``
# - ``messages: [{role: "user"|"assistant", content: str}, ...]``
# https://docs.anthropic.com/en/api/messages — version 2023-06-01.
_ANTHROPIC_API_VERSION = "2023-06-01"
_MESSAGES_PATH = "/v1/messages"
_DEFAULT_MAX_TOKENS = 1024
"""Output ceiling for the Messages API. We don't expose this on the
``ModelBackend`` Protocol because the router's cost-ceiling math already
caps spend; the Anthropic API requires *some* value here, so we pick a
conservative default."""

# Retry schedule per SPEC-009 §"M3 Sprint 3 — real Anthropic backend".
# Tuple of seconds to sleep before attempt N+1 (i.e. attempt 0 is the
# initial call, attempt 1 sleeps _RETRY_BACKOFF_SECONDS[0] first, etc.).
_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 4.0, 16.0)


def _build_payload(model: str, messages: list[Message]) -> dict[str, Any]:
    """Translate ``list[Message]`` → Anthropic Messages API JSON body.

    System messages are concatenated (newline-joined) into the top-level
    ``system`` field; user/assistant entries pass through with their
    role + content. The order of user/assistant messages is preserved
    so multi-turn conversations work correctly.
    """
    system_parts: list[str] = []
    chat: list[dict[str, str]] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            # Anthropic only accepts user/assistant in the messages list.
            # Anything else is a programmer bug — we surface it loudly.
            if msg.role not in ("user", "assistant"):
                raise ValueError(
                    f"unsupported message role {msg.role!r}; expected one of system|user|assistant"
                )
            chat.append({"role": msg.role, "content": msg.content})

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": _DEFAULT_MAX_TOKENS,
        "messages": chat,
    }
    if system_parts:
        payload["system"] = "\n".join(system_parts)
    return payload


def _classify_status(status_code: int) -> tuple[str, bool]:
    """Map an HTTP status code → ``(BackendError.kind, retryable)``.

    Retryable conditions are 429, 529, and 5xx. Auth (401) and other
    4xx codes are not retried — they signal configuration / input bugs
    that won't fix themselves with another attempt.
    """
    if status_code == 429:
        return "rate_limit", True
    if status_code == 529:
        return "overloaded", True
    if 500 <= status_code < 600:
        return "server", True
    if status_code == 401:
        return "auth", False
    if 400 <= status_code < 500:
        return "client", False
    # Treat any non-2xx outside the buckets above as a server error so
    # the fallback chain has a chance to recover.
    return "server", True


# --------------------------------------------------------------------------- #
# Backend                                                                     #
# --------------------------------------------------------------------------- #


class AnthropicBackend:
    """``ModelBackend`` for the Anthropic Messages REST API.

    Parameters
    ----------
    api_key:
        Anthropic API key. Stored on ``self`` (private) and sent as the
        ``x-api-key`` header on every request. **Never logged or
        serialised.** ``None`` is accepted to preserve the legacy stub
        signature for tests that don't intend to call ``generate``; a
        real call with ``api_key=None`` will be rejected by the API.
    model:
        Model SKU (e.g. ``"claude-3-5-sonnet-latest"``,
        ``"claude-sonnet-4-5-20250929"``, ``"claude-opus-4-5-20250929"``).
        Used verbatim in the request body and consulted by the pricing
        lookup.
    base_url:
        Override the API host. Tests use this with ``httpx.MockTransport``;
        production leaves the default.
    timeout_seconds:
        Per-request timeout passed to ``httpx.AsyncClient``. The retry
        loop wraps multiple requests so the *total* wall clock can be
        higher than this when backoff kicks in.
    max_retries:
        Maximum number of retry attempts after the first call. Default
        2 means up to 3 total HTTP calls per ``generate``. The chain
        ``_RETRY_BACKOFF_SECONDS`` is truncated to ``max_retries`` so
        the sleep schedule respects this bound.
    transport:
        Optional ``httpx.AsyncBaseTransport``. Tests pass an
        ``httpx.MockTransport`` here to intercept requests without
        touching the network. Production leaves this ``None``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-latest",
        base_url: str = "https://api.anthropic.com",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        self.label = f"anthropic:{model}"
        self._model = model
        # Never log, never expose in error messages.
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._transport = transport
        # Cache pricing once at construction — model is immutable.
        self._price_in_per_million, self._price_out_per_million = _price_for_model(model)

    # --------------------------------------------------------------------- #
    # ModelBackend Protocol                                                 #
    # --------------------------------------------------------------------- #

    async def generate(self, messages: list[Message]) -> BackendResponse:
        """POST the messages payload and return a populated
        ``BackendResponse``.

        Raises
        ------
        BackendError
            On any retryable failure (rate limit / 5xx / timeout /
            network) after the retry budget is exhausted; or on a
            non-retryable failure (auth / 4xx) immediately.
        """
        payload = _build_payload(self._model, messages)
        headers = {
            "x-api-key": self._api_key or "",
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

        attempts_allowed = self._max_retries + 1
        backoff_schedule = _RETRY_BACKOFF_SECONDS[: self._max_retries]
        last_error: BackendError | None = None
        started = time.perf_counter()

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            transport=self._transport,
        ) as client:
            for attempt in range(attempts_allowed):
                try:
                    response = await client.post(
                        _MESSAGES_PATH,
                        headers=headers,
                        json=payload,
                    )
                except httpx.TimeoutException as exc:
                    last_error = BackendError(
                        f"anthropic request timed out after {self._timeout}s",
                        kind="timeout",
                    )
                    last_error.__cause__ = exc
                except httpx.TransportError as exc:
                    last_error = BackendError(
                        f"anthropic network error: {exc.__class__.__name__}",
                        kind="network",
                    )
                    last_error.__cause__ = exc
                else:
                    if 200 <= response.status_code < 300:
                        latency_ms = (time.perf_counter() - started) * 1000.0
                        return self._build_response(response, latency_ms)
                    kind, retryable = _classify_status(response.status_code)
                    err = BackendError(
                        f"anthropic API returned HTTP {response.status_code} ({self.label})",
                        kind=kind,
                    )
                    if not retryable:
                        raise err
                    last_error = err

                # We get here only on retryable failures. Sleep then loop.
                if attempt < len(backoff_schedule):
                    await asyncio.sleep(backoff_schedule[attempt])

        # Retry budget exhausted. Re-raise the final classified error.
        assert last_error is not None, "loop must produce a last_error on exit"
        raise last_error

    def quote(self, messages: list[Message]) -> CostQuote:
        """Pre-call USD estimate using the input-token heuristic + a
        fixed assumed output (256 tokens) + a 10% safety margin (ADR-013).
        """
        tokens_in = _estimate_tokens(messages)
        input_usd = tokens_in * self._price_in_per_million / 1_000_000.0
        output_usd = _OUTPUT_TOKENS_ASSUMED * self._price_out_per_million / 1_000_000.0
        return CostQuote(
            estimated_usd=(input_usd + output_usd) * 1.10,
            backend_label=self.label,
        )

    # --------------------------------------------------------------------- #
    # Internals                                                             #
    # --------------------------------------------------------------------- #

    def _build_response(self, http_response: httpx.Response, latency_ms: float) -> BackendResponse:
        """Translate a 2xx Anthropic JSON body → ``BackendResponse``.

        Latency is supplied by the caller (envelope around the full
        retry loop) so a successful retry's response carries the total
        wall-clock latency the user actually waited.
        """
        # ``latency_ms`` is part of the public ``BackendResponse`` only
        # when the dataclass has the field; today it doesn't (the
        # router computes its own envelope), so we keep the variable
        # local for future extension. Suppress unused warnings.
        _ = latency_ms

        try:
            data = http_response.json()
        except ValueError as exc:
            # 200 OK with un-parseable body is a server-side bug.
            raise BackendError(
                f"anthropic 200 OK but body was not JSON ({self.label})",
                kind="server",
            ) from exc

        text = self._extract_text(data)
        tokens_in, tokens_out = self._extract_usage(data)
        cost_usd = (
            tokens_in * self._price_in_per_million / 1_000_000.0
            + tokens_out * self._price_out_per_million / 1_000_000.0
        )

        return BackendResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
        )

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Pull the first text block out of ``data["content"]``.

        Anthropic returns ``content`` as a list of content blocks; for
        plain text responses there's one entry of ``{"type": "text",
        "text": "..."}``. We concatenate all text-type blocks so future
        tool-use or thinking blocks (which use different ``type`` values)
        don't leak into the summary.
        """
        content = data.get("content")
        if not isinstance(content, list) or not content:
            raise BackendError(
                "anthropic response had empty/missing content array",
                kind="server",
            )
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            # Default to "text" type when missing — the legacy single-
            # block responses sometimes omit it.
            block_type = block.get("type", "text")
            if block_type == "text":
                text_val = block.get("text", "")
                if isinstance(text_val, str):
                    parts.append(text_val)
        if not parts:
            raise BackendError(
                "anthropic response had no text content blocks",
                kind="server",
            )
        return "".join(parts)

    @staticmethod
    def _extract_usage(data: dict[str, Any]) -> tuple[int, int]:
        """Pull ``(input_tokens, output_tokens)`` out of ``data["usage"]``.

        Anthropic always includes ``usage`` on success, but defensively
        we tolerate a missing block by returning zeros so the response
        is still well-formed.
        """
        usage = data.get("usage") or {}
        tokens_in = int(usage.get("input_tokens", 0) or 0)
        tokens_out = int(usage.get("output_tokens", 0) or 0)
        return tokens_in, tokens_out


# ``AnthropicProvider`` is the name SPEC-009 §"M3 Sprint 3" uses in
# prose; the historical class name is ``AnthropicBackend`` because the
# Protocol it satisfies is ``ModelBackend``. Both names refer to the
# same class.
AnthropicProvider = AnthropicBackend


__all__ = ["AnthropicBackend", "AnthropicProvider"]
