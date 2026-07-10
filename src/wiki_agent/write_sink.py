"""
Bridge from the canonical `wiki_core.WriteSink` protocol to the
`wiki-knowledge-engine` MCP `write_page` tool (the historic write path
for the SBW vault).

Two consumer surfaces:

- `LocalWriteSink` — synchronous-style writes that import and call the
  underlying handler module directly (used inside the same process as the
  MCP server, e.g. when the agent and the server share a Python runtime).
- `McpWriteSink` — out-of-process writes via the MCP protocol. Skeleton
  here; M2 polish wires in the actual MCP client. Today the agent runs
  in-process with the MCP server, so `LocalWriteSink` is sufficient.

In both cases the bridge:

- frontmatter-serialises `Page.frontmatter` into the YAML header that
  the `write_page` handler expects;
- forbids writes outside the `wiki/` and `observations/` prefixes
  (mirrors `wiki_agent.middleware.SafetyGateMiddleware`);
- normalises return values to a `pathlib.Path` for the protocol contract.

This is the M2 Sprint 1 bridge for [PRD-004 Memory Tree](../../docs/PRD-004-memory-tree.md):
the Memory Tree write surface will satisfy `WriteSink`, so the agent
talks to the tree through the same interface as it talks to the legacy
flat vault. No call sites need to change when M2 lands.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml

if TYPE_CHECKING:
    from wiki_compress import CompressionResult
    from wiki_core.protocols import Page

#: Optional body-compression hook.
#:
#: Accepts the page body (post-frontmatter) and returns a
#: ``CompressionResult``. The sink consults ``result.body`` for the
#: text that actually goes to the underlying handler. Frontmatter is
#: never compressed — it carries structural metadata (titles, tags,
#: dates) that downstream readers must parse verbatim.
BodyCompressor = Callable[[str], "CompressionResult"]


ALLOWED_PREFIXES: tuple[str, ...] = ("wiki/", "observations/")


class WriteSinkPolicyError(RuntimeError):
    """Raised when a Page violates the configured write policy
    (path prefix, dangerous frontmatter keys, etc.).

    Mirrors the refusal path of
    `wiki_agent.middleware.SafetyGateMiddleware` so that direct callers
    (CLI, batch scripts) see the same boundary checks as the harness.
    """


def _serialise_page(page: Page, *, body_override: str | None = None) -> str:
    """Frontmatter+body markdown string for the `write_page` handler.

    `frontmatter` ordering is preserved by `yaml.safe_dump` with
    `sort_keys=False` so that downstream readers see a stable byte
    layout (helps git diffs).

    When *body_override* is provided, it replaces ``page.body`` in the
    serialised output (the compression hook's product). The frontmatter
    block is never touched — structural metadata stays verbatim."""
    frontmatter = yaml.safe_dump(page.frontmatter, sort_keys=False, allow_unicode=True).strip()
    body = body_override if body_override is not None else page.body
    return f"---\n{frontmatter}\n---\n\n{body.lstrip()}"


def _compressed_body(body: str, compressor: BodyCompressor | None) -> str | None:
    """Run *body* through *compressor* if given, return the new body.

    Returns ``None`` when no compressor was supplied so that callers can
    cheaply tell "no transform applied" from "transform → empty string".
    """
    if compressor is None:
        return None
    return compressor(body).body


def _enforce_path(page_path: str) -> None:
    if not page_path.startswith(ALLOWED_PREFIXES):
        raise WriteSinkPolicyError(
            f"WriteSink refused page_path '{page_path}': must start with "
            f"one of {list(ALLOWED_PREFIXES)}"
        )


class LocalWriteSink:
    """In-process `WriteSink` that calls the wiki-knowledge-engine
    handlers directly (no MCP RPC). Used when the agent and the MCP
    server share a Python runtime, which is the default deployment
    today.

    The constructor accepts the two handler callables — that keeps the
    adapter testable without spinning up an MCP server. Real wiring
    happens in `wiki_agent.agent.create_wiki_agent` (M2 sprint follow-up).
    """

    def __init__(
        self,
        write_page_handler: Callable[[str, str], Awaitable[str] | str],
        append_to_log_handler: Callable[[str, str, str], Awaitable[str] | str],
        *,
        log_entry_type: str = "ingest",
        log_title_prefix: str = "page-write",
        compressor: BodyCompressor | None = None,
    ) -> None:
        self._write_page = write_page_handler
        self._append_to_log = append_to_log_handler
        self._log_entry_type = log_entry_type
        self._log_title_prefix = log_title_prefix
        self._compressor = compressor

    async def write_page(
        self,
        page: Page,
        *,
        mode: Literal["create", "update", "upsert"] = "upsert",
    ) -> Path:
        _enforce_path(page.ref.page_path)
        compressed = _compressed_body(page.body, self._compressor)
        body = _serialise_page(page, body_override=compressed)
        result = self._write_page(page.ref.page_path, body)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]
        return Path(page.ref.page_path)

    async def append_to_log(self, entry: str) -> None:
        # The legacy handler is (entry_type, title, details); flatten the
        # protocol's single-string `entry` into a title + details split.
        title, _, details = entry.partition("\n")
        if not details:
            details = title
            title = f"{self._log_title_prefix}: {title[:60]}"
        result = self._append_to_log(self._log_entry_type, title, details)
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]


class NullWriteSink:
    """Read-only sink for tests and dry-runs. Writes are recorded into
    `calls` instead of touching the filesystem.

    Useful when validating that a subagent or middleware *would* have
    written something, without committing to a real I/O side effect.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Page]] = []
        self.log_entries: list[str] = []

    async def write_page(
        self,
        page: Page,
        *,
        mode: Literal["create", "update", "upsert"] = "upsert",
    ) -> Path:
        _enforce_path(page.ref.page_path)
        self.calls.append((mode, page))
        return Path(page.ref.page_path)

    async def append_to_log(self, entry: str) -> None:
        self.log_entries.append(entry)


class CompressingWriteSink:
    """Decorator sink that compresses page bodies before delegating.

    Wraps any sink satisfying the ``wiki_core.protocols.WriteSink``
    protocol (``LocalWriteSink``, ``NullWriteSink``, an eventual
    ``McpWriteSink``, …). The inner sink sees a `Page` whose ``body``
    has already been passed through *compressor*; frontmatter and path
    are untouched, so the policy checks the inner sink performs (path
    prefix, dangerous keys) remain valid.

    The compressor is invoked **once per write**, never on retries the
    inner sink might do internally — composition keeps the contract
    simple.
    """

    def __init__(self, inner: object, compressor: BodyCompressor) -> None:
        self._inner = inner
        self._compressor = compressor

    async def write_page(
        self,
        page: Page,
        *,
        mode: Literal["create", "update", "upsert"] = "upsert",
    ) -> Path:
        from dataclasses import replace as _replace

        compressed_body = self._compressor(page.body).body
        compressed_page = _replace(page, body=compressed_body)
        return await self._inner.write_page(compressed_page, mode=mode)  # type: ignore[attr-defined]

    async def append_to_log(self, entry: str) -> None:
        await self._inner.append_to_log(entry)  # type: ignore[attr-defined]


__all__ = [
    "ALLOWED_PREFIXES",
    "BodyCompressor",
    "CompressingWriteSink",
    "LocalWriteSink",
    "NullWriteSink",
    "WriteSinkPolicyError",
]
