"""Mechanical-page composition + `write_page` result parsing (ADR-035 D1).

Split out of `worker.py` so the worker module stays focused on the
dispatch/security pipeline. Everything here is pure and synchronous:
deterministic slug + page rendering on the way in, tolerant (but
honest) result parsing on the way out.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wiki_ingest.models import IngestEvent

__all__ = [
    "WritePageError",
    "WriteSkippedError",
    "extract_page_path",
    "extract_skip_reason",
    "page_slug",
    "render_page",
    "result_text",
]

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class WritePageError(RuntimeError):
    """`write_page` failed or returned an unusable response (ADR-035 D1).

    Raised by `WorkerPool._call_write_page` so the failure routes into
    the worker's retry/mark_failed branch instead of being silently
    converted into a fake success — the root cause of the
    `page_path=NULL` incident (events marked done, no page on disk,
    raw file consumed).
    """


class WriteSkippedError(RuntimeError):
    """`write_page` deliberately declined the write (ADR-048 D4 skip tier).

    Not a failure: the quality policy judged the page `no_signal` (zero-value
    stub). The worker must mark the event skipped — NOT retry (the verdict is
    deterministic, a retry can never succeed) and NOT mark it failed.
    """


def page_slug(rel_path: str, fallback: str) -> str:
    """Deterministic wiki slug for a raw file (lowercase, hyphenated)."""
    stem = Path(rel_path).stem
    slug = _SLUG_RE.sub("-", stem.lower()).strip("-")
    return slug or fallback


def _yaml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_page(
    event: IngestEvent,
    sha: str,
    mime: str,
    size: int,
    envelope: dict | None,
    ingested_at: str,
) -> str:
    """Compose the full markdown page (frontmatter + wrapped body).

    SEC-05: the page is stamped trust-untrusted in its own frontmatter
    and carries the sha256 + ingested_at so the downstream agent can
    refuse directives from it. Raw frontmatter is never forwarded —
    only a sanitised key digest.
    """
    title = Path(event.rel_path).stem or event.event_id
    fm_lines = [
        "---",
        f'title: "{_yaml_escape(title)}"',
        f"date: {ingested_at[:10]}",
        f'sources: ["{_yaml_escape(event.rel_path)}"]',
        f"tags: [ingested, {event.bucket}]",
        "origin: ingested-untrusted",
        f"ingested_sha256: {sha}",
        f"ingested_at: {ingested_at}",
    ]
    if envelope is not None and envelope.get("frontmatter_in"):
        keys = ", ".join(sorted(envelope["frontmatter_in"].keys()))
        fm_lines.append(f"frontmatter_keys: [{keys}]")
    fm_lines.append("---")

    if envelope is not None:
        body = envelope["wrapped_body"]
    else:
        body = (
            f"Binary or non-text source (mime: {mime or 'unknown'}, "
            f"{size} bytes); content not inlined."
        )
    return "\n".join(fm_lines) + f"\n\n# {title}\n\n{body}\n"


def result_text(result: dict) -> str:
    """Concatenate the text items of an MCP tool result (for errors)."""
    parts: list[str] = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return " ".join(parts) or json.dumps(result)[:500]


def extract_page_path(result: dict) -> str | None:
    """Pull the confirmed page_path out of a `write_page` tool result.

    Handles `structuredContent.page_path`, `structuredContent.result`
    (FastMCP's wrapper for plain-string tool returns), and JSON text
    content items. A `refused`/`duplicate_source` response means an
    existing page already covers this source — that existing page is
    the truthful page_path.
    """
    candidates: list[dict] = []
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        candidates.append(structured)
        inner = structured.get("result")
        if isinstance(inner, str):
            parsed = _try_json(inner)
            if parsed is not None:
                candidates.append(parsed)
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parsed = _try_json(str(item.get("text", "")))
            if parsed is not None:
                candidates.append(parsed)

    for payload in candidates:
        if payload.get("error"):
            return None
        if payload.get("status") == "refused":
            existing = payload.get("existing_page")
            return str(existing) if existing else None
        if payload.get("page_path"):
            return str(payload["page_path"])
    return None


def extract_skip_reason(result: dict) -> str | None:
    """Return the skip reason when a `write_page` result is a deliberate
    quality skip (ADR-048 D4 `status: skipped`), else ``None``.

    Mirrors :func:`extract_page_path`'s tolerant payload walk so both FastMCP
    result shapes (structuredContent and JSON text items) are recognised.
    """
    candidates: list[dict] = []
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        candidates.append(structured)
        inner = structured.get("result")
        if isinstance(inner, str):
            parsed = _try_json(inner)
            if parsed is not None:
                candidates.append(parsed)
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parsed = _try_json(str(item.get("text", "")))
            if parsed is not None:
                candidates.append(parsed)
    for payload in candidates:
        if payload.get("status") == "skipped":
            return str(payload.get("reason") or "skipped")
    return None


def _try_json(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
