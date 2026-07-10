"""
Append-only JSONL audit log for integration events per [ADR-017](../../../docs/ADR-017-oauth-token-storage-and-scope-policy.md) §Audit log.

Path is `~/.sbw/logs/integrations.log.jsonl` by default — explicitly NOT
inside `knowledge-base/` so a vault sync never picks it up. One JSON object
per line; weekly rotation is left to a future cron entry (out of scope for #39).

Redaction rules:

- OAuth bearers, refresh tokens, and Composio API keys never appear in `params`.
- Free-form search strings (`q`, `query`, message bodies) are replaced with
  ``"<redacted>"`` unless `verbose_audit=True` is passed when writing.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

_log = logging.getLogger(__name__)


Event = Literal[
    "execute_action",
    "connect",
    "disconnect",
    "refresh",
    "revoked",
    "scope_drift_required",
    "composio_outage",
]

AUDIT_SCHEMA_VERSION: Final = 1
"""Bump when changing the on-disk JSONL shape. Readers MUST verify."""

DEFAULT_LOG_PATH: Final = Path.home() / ".sbw" / "logs" / "integrations.log.jsonl"


# Keys whose values should be redacted unless verbose_audit is true.
# Conservative list; expand if a provider introduces another free-form field.
_REDACT_KEYS: frozenset[str] = frozenset(
    {"q", "query", "body", "snippet", "subject", "text", "content"}
)

# Keys that, if present at any nesting level, indicate a leaked secret and
# MUST always be redacted regardless of `verbose_audit`. The presence of
# any of these in `params` is also a bug worth surfacing in the log.
_ALWAYS_REDACT: frozenset[str] = frozenset(
    {
        "access_token",
        "refresh_token",
        "api_key",
        "client_secret",
        "password",
        "authorization",
        "bearer",
    }
)


class AuditLog:
    """Thread-safe append-only JSONL writer for integration events."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_LOG_PATH
        self._lock = threading.Lock()
        # Create parent dir on first use; permissions 0o700 since the log
        # contains action metadata (params_redacted, agent_turn_id) that
        # is sensitive even after redaction.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._path.parent, 0o700)
        except (OSError, PermissionError):
            # On Windows / weird mounts chmod can fail; not fatal.
            pass

    @property
    def path(self) -> Path:
        return self._path

    def write(
        self,
        *,
        event: Event,
        provider: str,
        action: str | None = None,
        agent_turn_id: str | None = None,
        params: dict | None = None,
        result: str = "ok",
        latency_ms: int | None = None,
        scope_used: tuple[str, ...] | None = None,
        verbose_audit: bool = False,
    ) -> None:
        """Append a single event line.

        Caller is responsible for passing semantically correct fields; the
        only thing this method does to `params` is redaction. The full
        record shape matches ADR-017's schema v1.
        """
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "schema": AUDIT_SCHEMA_VERSION,
            "event": event,
            "provider": provider,
            "result": result,
        }
        if action is not None:
            record["action"] = action
        if agent_turn_id is not None:
            record["agent_turn_id"] = agent_turn_id
        if latency_ms is not None:
            record["latency_ms"] = int(latency_ms)
        if scope_used is not None:
            record["scope_used"] = list(scope_used)
        record["params_redacted"] = redact_params(params or {}, verbose=verbose_audit)

        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        with self._lock, self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def redact_params(params: dict, *, verbose: bool = False) -> dict:
    """Return a copy of `params` with sensitive keys redacted.

    Walks dicts recursively. Lists are walked but not redacted as a unit
    (only their dict elements). Strings on `_ALWAYS_REDACT` keys are
    replaced unconditionally; `_REDACT_KEYS` only when `verbose=False`.
    """
    return _redact_value(params, verbose=verbose)


def _redact_value(value: object, *, verbose: bool) -> object:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key_lc = k.lower()
            if key_lc in _ALWAYS_REDACT:
                out[k] = "<redacted>"
            elif not verbose and key_lc in _REDACT_KEYS:
                out[k] = "<redacted>"
            else:
                out[k] = _redact_value(v, verbose=verbose)
        return out
    if isinstance(value, list):
        return [_redact_value(item, verbose=verbose) for item in value]
    return value
