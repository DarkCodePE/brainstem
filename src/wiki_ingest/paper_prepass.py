"""PDF → markdown sidecar pre-pass for ``raw/papers/`` (PRD-015 FR-5, ADR-032 D2).

The daemon-route caller of the ``wiki_papers`` bounded context: when the
worker meets an ``application/pdf`` event under ``raw/papers/``, this
module extracts the paper IN-PROCESS (engine chain per ADR-032 D1) and
writes the markdown sidecar ``raw/papers/<arxiv_id-or-slug>.md`` next to
the PDF. The sidecar then re-enters the normal ingest pipeline as a
plain text event, so the SEC-05 untrusted envelope and the ADR-035
synthesis leg apply to extracted paper text unchanged (SR-2). The
original PDF moves to ``raw/_ingested/papers/`` on success (FR-5).

Degrade-first, never crash (same posture as the code-graph leg of
ADR-022): a missing ``wiki_papers`` install, an engine failure, or a
sidecar write error each mark the event skipped with a log event and
leave the PDF where it was.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from wiki_ingest.pagewrite import page_slug
from wiki_ingest.security import atomic_write_text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from wiki_ingest.models import IngestEvent
    from wiki_ingest.queue import EventQueue

log = logging.getLogger("wiki_ingest.paper_prepass")

__all__ = ["is_paper_pdf", "render_paper_sidecar", "run_paper_pre_pass"]

_PAPERS_BUCKET = "papers"


def is_paper_pdf(rel_path: str) -> bool:
    """True when ``rel_path`` lives under ``raw/papers/`` (any depth).

    Accepts both the watcher's kb-root-relative shape
    (``raw/papers/x.pdf``) and a raw-dir-relative shape
    (``papers/x.pdf``).
    """
    parts = Path(rel_path).parts
    if parts and parts[0] == "raw":
        parts = parts[1:]
    return len(parts) >= 2 and parts[0] == _PAPERS_BUCKET


def render_paper_sidecar(frontmatter: dict, markdown: str) -> str:
    """Compose the sidecar markdown (FR-4 frontmatter + extracted body).

    ``wiki_papers.extract_paper`` returns frontmatter and body
    separately; when the body already opens with its own frontmatter
    block the engine composed the full document itself and it is
    written verbatim.
    """
    body = (markdown or "").strip()
    if body.startswith("---\n"):
        return body + "\n"
    import yaml  # local import — same optional-at-import posture as security.py

    fm = yaml.safe_dump(dict(frontmatter or {}), sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm}\n---\n\n{body}\n"


async def _call_engine(fn: Any, *args: Any) -> Any:
    """Call a ``wiki_papers`` function whether it shipped sync or async
    (the PRD-015 contract does not pin this)."""
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    result = await asyncio.to_thread(fn, *args)
    if inspect.isawaitable(result):
        return await result
    return result


async def run_paper_pre_pass(
    event: IngestEvent,
    src: Path,
    *,
    queue: EventQueue,
    move_to_ingested: Callable[[Path, IngestEvent], Awaitable[None]],
) -> None:
    """Extract one ``raw/papers/`` PDF into its markdown sidecar.

    Success: the sidecar lands next to the PDF (the watcher/oneshot
    picks it up as a normal text event — the *sidecar* event is the one
    that produces the wiki page), the PDF moves to
    ``raw/_ingested/papers/``, and the PDF event is marked skipped with
    reason ``paper-extracted``. Any failure skips with an honest reason
    and leaves the PDF in place; this function NEVER raises into the
    worker.
    """
    try:
        from wiki_papers import extract_paper  # lazy: optional bounded context
    except Exception as e:  # noqa: BLE001 — ImportError or any transitive failure
        await queue.mark_skipped(event.event_id, "paper-extractor-unavailable")
        log.warning(
            "paper_prepass.unavailable",
            extra={
                "extra_fields": {
                    "event_id": event.event_id,
                    "reason": "paper-extractor-unavailable",
                    "error_class": type(e).__name__,
                }
            },
        )
        return

    try:
        paper = await _call_engine(extract_paper, src)
    except Exception as e:  # noqa: BLE001 — engine-chain failure: degrade to skip
        await queue.mark_skipped(event.event_id, f"paper-extract-failed:{type(e).__name__}")
        log.warning(
            "paper_prepass.extract_failed",
            extra={
                "extra_fields": {
                    "event_id": event.event_id,
                    "error_class": type(e).__name__,
                }
            },
        )
        return

    frontmatter = dict(getattr(paper, "frontmatter", None) or {})
    stem = str(frontmatter.get("arxiv_id") or "").strip() or page_slug(
        event.rel_path, event.event_id
    )
    sidecar = src.parent / f"{stem}.md"
    content = render_paper_sidecar(frontmatter, getattr(paper, "markdown", "") or "")
    try:
        await asyncio.to_thread(atomic_write_text, sidecar, content)
    except OSError as e:
        await queue.mark_skipped(event.event_id, f"paper-sidecar-write-failed:{type(e).__name__}")
        log.warning(
            "paper_prepass.sidecar_write_failed",
            extra={
                "extra_fields": {
                    "event_id": event.event_id,
                    "error_class": type(e).__name__,
                }
            },
        )
        return

    await queue.mark_skipped(event.event_id, "paper-extracted")
    await move_to_ingested(src, event)
    stats = getattr(paper, "stats", None)
    # FR-7: surface PaperStats — no silent truncation.
    log.info(
        "paper_prepass.extracted",
        extra={
            "extra_fields": {
                "event_id": event.event_id,
                "sidecar": sidecar.name,
                "engine_used": getattr(stats, "engine_used", None),
                "pages": getattr(stats, "pages", None),
                "extracted_chars": getattr(stats, "extracted_chars", None),
                "sections_found": getattr(stats, "sections_found", None),
                "truncated": getattr(stats, "truncated", None),
            }
        },
    )
