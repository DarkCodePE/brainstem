"""Post-write hooks wiring synthesis into the ingest worker (ADR-035 D3).

``SynthesisOnIngestHook`` satisfies the same
``wiki_memory.seal_hook.OnPageWrittenCallback`` protocol the seal hook
uses: ``async (domain_event, page_path) -> None``, invoked by
``WorkerPool._process`` after a successful ``write_page``, and it
NEVER raises into the ingest worker — any failure is logged and
swallowed (ingest already succeeded; the mechanical page is the
degrade output).

Unlike the seal hook's background task, synthesis is awaited inline by
default: the D2 deployment is an *ephemeral* ``sbw-ingest --once``
process, and a fire-and-forget ``asyncio.create_task`` would be
dropped when the oneshot drains the queue and exits. Inline awaiting
keeps the oneshot honest; the never-raises guarantee is what protects
the worker.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wiki_synthesis.agent import SynthesisAgent

log = logging.getLogger("wiki_synthesis.hooks")

__all__ = ["CompositePostWriteHook", "SynthesisOnIngestHook", "unwrap_envelope"]

_ENVELOPE_RE = re.compile(
    r"<ingested_source[^>]*>\n(?P<body>.*)\n</ingested_source>",
    re.DOTALL,
)


def unwrap_envelope(page_text: str) -> str:
    """Recover the raw body from the worker's SEC-05 trust envelope.

    The mechanical page wraps the raw file's body in an
    ``<ingested_source ...>...</ingested_source>`` envelope; synthesis
    wants the body itself. Falls back to the whole text when no
    envelope is present (e.g. a page written by another producer).
    """
    match = _ENVELOPE_RE.search(page_text)
    return match.group("body") if match else page_text


class SynthesisOnIngestHook:
    """Post-write callback: synthesize source/entity/concept pages.

    Reads the page the worker just wrote (the raw file has already
    moved to ``raw/_ingested/`` by hook time), unwraps the trust
    envelope to recover the raw body, and runs the
    :class:`~wiki_synthesis.agent.SynthesisAgent` sequence. The
    synthesized source page overwrites the mechanical page at the same
    ``wiki/sources/<slug>.md`` path — the mechanical page IS the
    degrade output if anything here fails.
    """

    def __init__(
        self,
        *,
        agent: SynthesisAgent,
        vault_root: Path,
        read_page: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        self._agent = agent
        self._vault_root = vault_root
        self._read_page = read_page or self._default_read_page

    async def __call__(self, event: Any, page_path: str) -> None:
        try:
            page_text = await self._read_page(page_path)
            raw_body = unwrap_envelope(page_text)
            rel_path = str(
                getattr(event, "metadata", {}).get("rel_path")
                or getattr(event, "path_or_uri", page_path)
            )
            outcome = await self._agent.synthesize(raw_text=raw_body, rel_path=rel_path)
            log.info(
                "synthesis_hook.done event_id=%s source=%s entities=%d concepts=%d llm=%s",
                getattr(event, "event_id", "?"),
                outcome.source_page_path,
                len(outcome.entity_page_paths),
                len(outcome.concept_page_paths),
                getattr(outcome, "llm_extracted", False),
            )
        except Exception:  # noqa: BLE001 — NEVER raise into the ingest worker.
            log.exception(
                "synthesis_hook.failed event_id=%s page_path=%s",
                getattr(event, "event_id", "?"),
                page_path,
            )

    async def _default_read_page(self, page_path: str) -> str:
        """Read a wiki page from the vault root with the same
        path-jail check as ``SealOnIngestHook`` (SEC-01 mirror)."""
        resolved = (self._vault_root / page_path).resolve()
        try:
            resolved.relative_to(self._vault_root.resolve())
        except ValueError as e:
            raise OSError(f"page_path escapes vault root: {page_path!r}") from e
        return resolved.read_text(encoding="utf-8")


class CompositePostWriteHook:
    """Run several post-write hooks in order, isolating failures.

    Used when both seal-on-ingest and synthesis-on-ingest are enabled:
    one hook crashing must affect neither its siblings nor the worker.
    """

    def __init__(self, hooks: list[Any]) -> None:
        self._hooks = list(hooks)

    async def __call__(self, event: Any, page_path: str) -> None:
        for hook in self._hooks:
            try:
                await hook(event, page_path)
            except Exception:  # noqa: BLE001 — sibling isolation.
                log.exception(
                    "composite_hook.child_failed hook=%s page_path=%s",
                    type(hook).__name__,
                    page_path,
                )
