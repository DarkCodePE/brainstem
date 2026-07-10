"""
Custom Deep Agents middleware that ports the legacy Claude Code hooks
(``hooks/safety-gate.sh``, ``hooks/context-updater.sh``, ``hooks/til-capture.sh``)
into the harness so the same observation/safety behaviour fires when the
wiki agent runs *outside* Claude Code (CLI, MCP SSE, batch runs).

Closes issue #25.

## Mapping

| Original Claude Code hook   | Deep Agents middleware                    | Notes |
| --------------------------- | ----------------------------------------- | --- |
| ``hooks/safety-gate.sh``    | :class:`SafetyGateMiddleware`             | Logs every ``write_page`` invocation; refuses writes to disallowed paths. |
| ``hooks/context-updater.sh``| :class:`ContextUpdaterMiddleware`         | Detects edits to structural files (schema, CLAUDE.md, src/wiki_agent/) and appends a structured log line. |
| ``hooks/til-capture.sh``    | *not ported* — Claude-Code-specific       | Fires on Bash ``git commit``; the Deep Agents harness has no ``Bash`` tool. |

The shell hooks in ``hooks/`` remain in place — they serve Claude Code's
lifecycle (PreToolUse / PostToolUse on its own tools). The middleware
here serves the wiki agent's own lifecycle when running standalone.

## Wiring

The middlewares are added to ``create_deep_agent(..., middleware=[...])``
in ``src/wiki_agent/agent.py``. They are opt-out via the
``WIKI_DISABLE_HOOKS`` environment variable for headless test runs that
don't want the side effects.

## Design notes

- Middleware never mutates the ``ToolMessage`` body. It can:
  (a) record telemetry,
  (b) refuse the call by returning a ``ToolMessage(status='error', ...)``,
  (c) annotate the response with extra metadata.
- All side effects are best-effort — a logging failure must never break
  the underlying tool call.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

log = logging.getLogger(__name__)

# Mirrors hooks/context-updater.sh: file globs whose edits we treat as
# "structural" and therefore worth recording.
STRUCTURAL_GLOBS: tuple[str, ...] = (
    "schema/",
    "wiki/index.md",
    "CLAUDE.md",
    "src/wiki_agent/",
    "docs/ADR-",
    "docs/PRD-",
    "docs/SPEC-",
)


def _is_disabled() -> bool:
    """Opt-out switch for headless test runs."""
    return os.environ.get("WIKI_DISABLE_HOOKS", "").lower() in {"1", "true", "yes"}


def _utcnow_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# SafetyGateMiddleware — port of hooks/safety-gate.sh                         #
# --------------------------------------------------------------------------- #


class SafetyGateMiddleware(AgentMiddleware):
    """Log every ``write_page`` invocation; refuse writes to disallowed paths.

    Original hook (``hooks/safety-gate.sh``) was informational only (echo to
    stderr). This middleware extends it slightly: writes outside the
    allowed root prefix (``wiki/``, ``observations/``) are refused with a
    structured error.
    """

    allowed_prefixes: tuple[str, ...]

    def __init__(self, allowed_prefixes: tuple[str, ...] = ("wiki/", "observations/")) -> None:
        super().__init__()
        self.allowed_prefixes = allowed_prefixes

    def wrap_tool_call(
        self,
        request: Any,  # ToolCallRequest
        handler: Callable[[Any], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        if _is_disabled():
            return handler(request)

        tool_name = getattr(request.tool, "name", None) or request.call.get("name", "")
        if tool_name != "write_page":
            return handler(request)

        args: dict[str, Any] = request.call.get("args", {}) or {}
        page_path = str(args.get("page_path") or args.get("path") or "<unknown>")

        log.info("safety_gate.write_page path=%s ts=%s", page_path, _utcnow_iso())

        if not page_path.startswith(self.allowed_prefixes) and page_path != "<unknown>":
            tool_call_id = request.call.get("id", "")
            return ToolMessage(
                content=(
                    f"SafetyGate refused write_page: path '{page_path}' is "
                    f"outside the allowed prefixes {list(self.allowed_prefixes)}. "
                    "Move the page under wiki/ or observations/ and retry."
                ),
                tool_call_id=tool_call_id,
                status="error",
            )

        return handler(request)


# --------------------------------------------------------------------------- #
# ContextUpdaterMiddleware — port of hooks/context-updater.sh                 #
# --------------------------------------------------------------------------- #


class ContextUpdaterMiddleware(AgentMiddleware):
    """Notify on edits to structural files.

    The original hook appended a line to ``MEMORY.md`` whenever a
    structural file was touched. In the harness context we don't have
    MEMORY.md available, so we instead:

    - emit a structured ``log.info`` event;
    - optionally write to a ring buffer file (``<vault>/log.md``) via the
      append-only ``append_to_log`` MCP tool *if* the harness exposes it
      (deferred to the WriteSink protocol per [PRD-004](../../docs/PRD-004-memory-tree.md));
    - debounce: skip if the same file was logged in the last 60 seconds.
    """

    structural_globs: tuple[str, ...]
    debounce_seconds: int

    def __init__(
        self,
        structural_globs: tuple[str, ...] = STRUCTURAL_GLOBS,
        debounce_seconds: int = 60,
    ) -> None:
        super().__init__()
        self.structural_globs = structural_globs
        self.debounce_seconds = debounce_seconds
        self._last_logged: dict[str, datetime] = {}

    def _is_structural(self, path: str) -> bool:
        return any(g in path for g in self.structural_globs)

    def _should_log(self, path: str) -> bool:
        last = self._last_logged.get(path)
        now = datetime.now(UTC)
        if last is None or (now - last).total_seconds() >= self.debounce_seconds:
            self._last_logged[path] = now
            return True
        return False

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        result = handler(request)

        if _is_disabled():
            return result

        tool_name = getattr(request.tool, "name", None) or request.call.get("name", "")
        if tool_name not in {"write_page", "Edit", "Write", "update_index_entry"}:
            return result

        args: dict[str, Any] = request.call.get("args", {}) or {}
        path = str(args.get("file_path") or args.get("page_path") or args.get("path") or "")

        if not path or not self._is_structural(path):
            return result

        if self._should_log(path):
            log.info(
                "context_updater.structural_edit path=%s tool=%s ts=%s",
                Path(path).name,
                tool_name,
                _utcnow_iso(),
            )

        return result


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def default_hook_middleware() -> list[AgentMiddleware]:
    """Return the default set of hook middleware for ``create_deep_agent``.

    Used by ``wiki_agent.agent.create_wiki_agent``. Order matters — outer
    first. SafetyGate runs outside (so refusals short-circuit before
    ContextUpdater would log them as a successful structural edit).
    """
    if _is_disabled():
        return []
    return [
        SafetyGateMiddleware(),
        ContextUpdaterMiddleware(),
    ]
