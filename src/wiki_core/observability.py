"""
Observability scaffold (issue #133).

Single entry point ``init_observability()`` wires:
- **structlog** for structured logging — JSON in prod, console in dev
- **Sentry** for error tracking — opt-in via ``WIKI_SENTRY_DSN``, no-op when unset
- **OpenTelemetry OTLP exporter** for traces — opt-in via
  ``WIKI_OTEL_ENDPOINT``, no-op when unset

All three are opt-in per the [[ADR-018]] single-tenant + local-first
posture: nothing leaves Orlando's machine unless he sets the env var
explicitly. The setup wizard (``sbw init``) will prompt for opt-in
later; this module is the runtime side.

### Why structlog and not stdlib logging directly

structlog gives us free JSON formatting + context-vars-based
request-id propagation + a processor chain so PII filtering is a
single ``before_emit`` step instead of N callsite changes. The cost
is one extra dep (~150 KiB) which is trivial.

### Privacy: ``before_send`` filter

The Sentry + structlog processors strip chunk bodies and prompt text
before emission so the agent's private content never leaves the
machine even when the user opts in to error tracking. The redaction
is keyed on field names: any field named ``body``, ``content``,
``prompt``, or ``answer`` gets replaced with ``"<redacted len=N>"``.

### Imports

Heavy deps (Sentry SDK, OTLP exporters) are imported INSIDE the
``_maybe_init_*`` helpers so this module imports cleanly on systems
that don't have them installed. Only ``structlog`` is a hard dep.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

try:
    import structlog
except ImportError:  # pragma: no cover - structlog is in pyproject deps
    structlog = None  # type: ignore[assignment]


# Field names whose values get redacted before any log/event leaves
# the process. Conservative — privacy regression here is much worse
# than missing one log field.
_REDACT_FIELDS = frozenset(
    {
        "body",
        "content",
        "prompt",
        "answer",
        "completion",
        "summary",
        "chunk_body",
        "user_message",
        "tool_input",
        "tool_output",
    }
)


# Failure kinds we classify as transient so the on-call/Sentry view
# can suppress noise. Borrowed from OpenHuman's classification in
# ``src/core/observability.rs:1-4926`` — same intent, smaller surface.
_TRANSIENT_ERROR_CLASSES = frozenset(
    {
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ConnectTimeout",
        "RemoteProtocolError",
        "RateLimitError",
        "BackendError",
        "EmbeddingUnavailableError",
    }
)


def redact_sensitive(event_dict: dict[str, Any]) -> dict[str, Any]:
    """Replace values of sensitive-named keys with ``<redacted len=N>``.

    Recursive — dives into nested dicts and lists so a chunk inside a
    ``trace`` field doesn't leak. Used as a structlog processor AND as
    the Sentry ``before_send`` body cleaner."""

    def _walk(value: Any, key_hint: str | None = None) -> Any:
        if isinstance(value, dict):
            return {k: _walk(v, key_hint=k) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v, key_hint=key_hint) for v in value]
        if key_hint in _REDACT_FIELDS and isinstance(value, str) and value:
            return f"<redacted len={len(value)}>"
        return value

    return _walk(event_dict)


def classify_error(exc: BaseException | None) -> str:
    """Return ``"transient"`` for known retryable failures, ``"critical"``
    otherwise. Sentry ``before_send`` uses this to set the ``level``
    so on-call gets paged on critical only."""
    if exc is None:
        return "unknown"
    return "transient" if type(exc).__name__ in _TRANSIENT_ERROR_CLASSES else "critical"


_initialised = False


def init_observability(
    *,
    service_name: str = "sbw",
    json_logs: bool | None = None,
    log_level: str = "INFO",
) -> None:
    """Idempotent one-shot setup. Safe to call from every entry point.

    ``json_logs`` defaults to ``True`` when ``WIKI_LOG_FORMAT=json``
    or when stdout is not a TTY (likely a systemd unit) — otherwise
    pretty console output for dev ergonomics."""
    global _initialised
    if _initialised:
        return
    _initialised = True

    if json_logs is None:
        format_env = os.environ.get("WIKI_LOG_FORMAT", "").lower()
        json_logs = format_env == "json" or not sys.stdout.isatty()

    _configure_structlog(json_logs=json_logs, log_level=log_level)
    _maybe_init_sentry(service_name=service_name)
    _maybe_init_otel(service_name=service_name)


def _configure_structlog(*, json_logs: bool, log_level: str) -> None:
    """Wire structlog's processor chain. The redact step runs first so
    every downstream processor (formatter, Sentry forwarder) sees the
    redacted view."""
    if structlog is None:
        # structlog not importable — degrade to stdlib logging at least.
        logging.basicConfig(level=log_level)
        return

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_processor,
    ]

    if json_logs:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _redact_processor(logger, method_name, event_dict):  # type: ignore[no-untyped-def]
    """structlog processor wrapper for ``redact_sensitive``."""
    return redact_sensitive(event_dict)


def _maybe_init_sentry(*, service_name: str) -> None:
    """No-op when ``WIKI_SENTRY_DSN`` is unset. Imports sentry_sdk
    lazily so this module imports clean without it installed."""
    dsn = os.environ.get("WIKI_SENTRY_DSN", "").strip()
    if not dsn:
        return

    try:
        import sentry_sdk
    except ImportError:
        logging.getLogger(__name__).warning(
            "WIKI_SENTRY_DSN is set but sentry-sdk is not installed; skipping."
        )
        return

    def _before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any] | None:
        # Redact sensitive bodies first
        event = redact_sensitive(event)
        # Classify and demote transient errors to warning level so
        # they don't page on-call.
        exc_info = hint.get("exc_info")
        if exc_info and len(exc_info) >= 2:
            kind = classify_error(exc_info[1])
            if kind == "transient":
                event["level"] = "warning"
                event["tags"] = {**event.get("tags", {}), "failure_kind": "transient"}
        return event

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=float(os.environ.get("WIKI_SENTRY_TRACES_RATE", "0.1")),
        environment=os.environ.get("WIKI_ENV", "dev"),
        release=os.environ.get("WIKI_VERSION", "0.1.0-dev"),
        before_send=_before_send,
        server_name=service_name,
    )


def _maybe_init_otel(*, service_name: str) -> None:
    """No-op when ``WIKI_OTEL_ENDPOINT`` is unset. Imports OTel deps
    lazily so this module imports clean without them installed."""
    endpoint = os.environ.get("WIKI_OTEL_ENDPOINT", "").strip()
    if not endpoint:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logging.getLogger(__name__).warning(
            "WIKI_OTEL_ENDPOINT is set but opentelemetry-sdk + "
            "opentelemetry-exporter-otlp-proto-http are not installed; skipping."
        )
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": os.environ.get("WIKI_VERSION", "0.1.0-dev"),
            "deployment.environment": os.environ.get("WIKI_ENV", "dev"),
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)


def reset_observability_for_testing() -> None:
    """Tests use this to reset the singleton flag between cases."""
    global _initialised
    _initialised = False


__all__ = [
    "classify_error",
    "init_observability",
    "redact_sensitive",
    "reset_observability_for_testing",
]
