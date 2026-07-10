"""
Tests for the observability scaffold (#133).

The contract under test:
- ``init_observability`` is idempotent and no-op without env vars
- Sensitive fields get redacted before structlog emits or Sentry forwards
- Error classification distinguishes transient (retryable) from critical
- Missing optional deps (sentry_sdk, opentelemetry) degrade gracefully
"""

from __future__ import annotations

import pytest

from wiki_core.observability import (
    classify_error,
    init_observability,
    redact_sensitive,
    reset_observability_for_testing,
)


class TestRedactSensitive:
    def test_redacts_body_field(self) -> None:
        out = redact_sensitive({"body": "secret chunk text", "source_id": "x.md"})
        assert out["body"].startswith("<redacted len=")
        assert out["source_id"] == "x.md"

    def test_redacts_nested(self) -> None:
        out = redact_sensitive({"trace": {"prompt": "secret prompt", "tier": "FAST"}})
        assert out["trace"]["prompt"].startswith("<redacted len=")
        assert out["trace"]["tier"] == "FAST"

    def test_redacts_inside_lists(self) -> None:
        out = redact_sensitive({"chunks": [{"body": "a"}, {"body": "bb"}]})
        assert all(c["body"].startswith("<redacted len=") for c in out["chunks"])

    def test_keeps_unmodified_fields(self) -> None:
        original = {
            "timestamp": "2026-05-29T00:00:00Z",
            "level": "info",
            "event": "request_handled",
            "request_id": "abc-123",
        }
        out = redact_sensitive(dict(original))
        assert out == original

    def test_empty_value_not_redacted(self) -> None:
        # Empty string in a sensitive field shouldn't be redacted to
        # avoid noise; it's already a no-op for privacy.
        out = redact_sensitive({"body": ""})
        assert out["body"] == ""


class TestClassifyError:
    def test_connect_error_is_transient(self) -> None:
        import httpx

        assert classify_error(httpx.ConnectError("nope")) == "transient"

    def test_arbitrary_error_is_critical(self) -> None:
        class _MyBugError(Exception):
            pass

        assert classify_error(_MyBugError("boom")) == "critical"

    def test_none_is_unknown(self) -> None:
        assert classify_error(None) == "unknown"


class TestInitObservability:
    def setup_method(self) -> None:
        reset_observability_for_testing()

    def test_no_env_vars_is_no_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default case: nothing configured, init succeeds without
        touching Sentry/OTLP. This is the local-first / single-tenant
        baseline."""
        for v in (
            "WIKI_SENTRY_DSN",
            "WIKI_OTEL_ENDPOINT",
            "WIKI_LOG_FORMAT",
        ):
            monkeypatch.delenv(v, raising=False)

        # Should not raise.
        init_observability(service_name="sbw-test")

        # Calling again is a no-op (idempotent guard).
        init_observability(service_name="sbw-test")

    def test_missing_sentry_sdk_degrades_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting the DSN without sentry-sdk installed should log a
        warning, not crash. Patches sys.modules to simulate the missing
        dep."""
        import sys

        monkeypatch.setenv("WIKI_SENTRY_DSN", "https://fake@sentry.io/1")
        monkeypatch.setitem(sys.modules, "sentry_sdk", None)

        # Should not raise.
        init_observability(service_name="sbw-test")

    def test_missing_otel_degrades_gracefully(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setenv("WIKI_OTEL_ENDPOINT", "http://localhost:4318")
        monkeypatch.setitem(sys.modules, "opentelemetry", None)

        init_observability(service_name="sbw-test")

    def test_json_log_format_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WIKI_LOG_FORMAT", "json")
        # Should not raise.
        init_observability(service_name="sbw-test")
