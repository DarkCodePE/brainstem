"""``SynthesisAgent`` — the worker's final leg (ADR-035 D3).

Port of the Hermes ``wiki-batch-ingest`` cron's per-file sequence (its
prompt is the specification of record):

1. read the raw source,
2. extract key entities and concepts,
3. write a summary page at ``wiki/sources/{slug}.md`` (frontmatter:
   title, date, sources, tags, origin) — preserving every markdown
   image reference (``![alt](url)`` and ``![[embed]]``) and the
   canonical source URL(s),
4. write entity pages at ``wiki/entities/{slug}.md`` and concept pages
   at ``wiki/concepts/{slug}.md``,
5. register each page in the index,
6. append an ``entry_type=ingest`` log entry recording the processing
   (the raw relative path appears verbatim — the Hermes batch detector
   marks files processed by substring match on log/index).

Extraction (issue #180): when a router is wired, ONE structured call
(REASONING tier) returns summary + entities + concepts as JSON and the
pages are rendered from it (``origin: llm-synthesized``). On ANY
failure the deterministic heuristics of :mod:`wiki_synthesis.extractors`
take over unchanged (``origin: synthesized-deterministic``). Either
way the budget invariant holds: ≤1 LLM call per file, and image refs +
source URLs are preserved by CODE (re-appended if the model drops
them), never trusted to the model.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wiki_synthesis.extractors import (
    extract_concepts,
    extract_entities,
    extract_image_refs,
    extract_urls,
    strip_frontmatter,
)
from wiki_synthesis.paper_distill import distill_paper
from wiki_synthesis.reconcile import accrete_mention_page, accrete_source_page
from wiki_synthesis.structured import ARCHETYPE_PAPER, detect_archetype, extract_structured
from wiki_synthesis.templates import (
    render_concept_page,
    render_entity_page,
    render_source_page,
    slugify,
    wikilink_terms,
)

log = logging.getLogger("wiki_synthesis.agent")

__all__ = ["PageWriteSkippedError", "SynthesisAgent", "SynthesisOutcome"]


class PageWriteSkippedError(RuntimeError):
    """``write_page`` deliberately declined the write (ADR-048 D4 skip tier).

    Raised by the composition-side ``write_page`` adapter when the tool
    returns ``status: skipped`` (quality ``no_signal``). The agent treats it
    as a per-page skip: a declined stub entity/concept must not abort the
    rest of the synthesis, and a declined source page leaves the mechanical
    page as the honest degrade output."""


# DI callables — composition wires these to the wiki_agent tools;
# tests inject stubs (London school).
WritePageFn = Callable[[str, str], Awaitable[str]]
UpdateIndexFn = Callable[[str, str, str, int], Awaitable[None]]
AppendLogFn = Callable[[str, str, str], Awaitable[None]]
# Optional read-back of an existing page for ADR-036 accretion. Returns the
# page text, or None if absent. When this callable is NOT injected the agent
# overwrites exactly as before (byte-identical degrade).
ReadPageFn = Callable[[str], Awaitable[str | None]]

_ORIGIN_LLM = "llm-synthesized"
_ORIGIN_DETERMINISTIC = "synthesized-deterministic"

_MAX_SUMMARY_CHARS = 140
_MAX_MENTION_LINES = 2
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SynthesisOutcome:
    """What one synthesis run produced."""

    source_page_path: str
    entity_page_paths: tuple[str, ...] = ()
    concept_page_paths: tuple[str, ...] = ()
    llm_extracted: bool = False
    log_details: str = ""

    @property
    def page_count(self) -> int:
        return 1 + len(self.entity_page_paths) + len(self.concept_page_paths)


@dataclass
class SynthesisAgent:
    """Router-driven extraction with a deterministic degrade path.

    Parameters
    ----------
    write_page:
        Async ``(page_path, content) -> confirmed_page_path``.
    update_index:
        Async ``(page_path, category, summary, source_count) -> None``.
    append_log:
        Async ``(entry_type, title, details) -> None``.
    router:
        Optional ``ModelRouter``-shaped object (``await router.call(task,
        messages=...)``). When present, ONE structured extraction call is
        made per file; on success ``origin`` is ``llm-synthesized`` and
        entity/concept pages carry the model's descriptions. ANY failure
        degrades to the deterministic heuristics (honest ``origin``).
    read_page:
        Optional ``(page_path) -> page_text | None`` used for ADR-036
        page-level accretion: when present, an existing entity/concept page
        is enriched (sources unioned, per-source ``## Mentions`` ledger)
        instead of overwritten, and a source page gains a ``## History``
        entry. When ABSENT (the default) every page is overwritten exactly
        as before — the accretion seam is opt-in and byte-identical when off.
    clock:
        Injectable ``() -> datetime`` for deterministic dates.
    """

    write_page: WritePageFn
    update_index: UpdateIndexFn
    append_log: AppendLogFn
    router: Any | None = None
    read_page: ReadPageFn | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    max_entities: int = 5
    max_concepts: int = 5

    async def synthesize(self, *, raw_text: str, rel_path: str) -> SynthesisOutcome:
        """Run the full Hermes-parity sequence for one raw file."""
        now = self.clock()
        date = now.date().isoformat()

        title = _derive_title(raw_text, rel_path)
        slug = slugify(title)
        source_page_path = f"wiki/sources/{slug}.md"
        urls = extract_urls(raw_text)

        # PRIMARY path: one structured router call (the file's only LLM
        # call). Falls back to the deterministic heuristics on ANY
        # failure — page bodies pair (name, description-or-mention).
        # Papers (PRD-015 FR-6) get the paper-shaped prompt: problem /
        # method / key results with numbers / limitations / relevance.
        archetype = detect_archetype(raw_text, rel_path)
        extraction = await extract_structured(
            raw_text,
            title=title,
            router=self.router,
            max_entities=self.max_entities,
            max_concepts=self.max_concepts,
            archetype=archetype,
        )
        if extraction is not None:
            origin = _ORIGIN_LLM
            entity_items = [(e.name, e.description) for e in extraction.entities]
            concept_items = [(c.name, c.description) for c in extraction.concepts]
            names = [name for name, _ in [*entity_items, *concept_items]]
            body = _append_missing_refs(wikilink_terms(extraction.summary, names), raw_text)
            # ADR-036 D4: model-written relevance (already non-empty: structured
            # parse falls back to the summary's first sentences if the model
            # omits it). No extra LLM call.
            relevance = extraction.relevance
        else:
            origin = _ORIGIN_DETERMINISTIC
            entities = extract_entities(raw_text, limit=self.max_entities)
            concepts = extract_concepts(raw_text, limit=self.max_concepts)
            entity_items = [(name, _mention_of(raw_text, name)) for name in entities]
            concept_items = [(name, _mention_of(raw_text, name)) for name in concepts]
            # ADR-048 D5: a paper's degrade body is its own distilled
            # Abstract/Contributions/Results — never the full extraction dump
            # (the raw text stays in the sidecar). Falls back to the classic
            # full-prose compose when the extraction has no paper shape.
            distilled = (
                distill_paper(raw_text, rel_path=rel_path) if archetype == ARCHETYPE_PAPER else None
            )
            if distilled is not None:
                body = wikilink_terms(distilled, [*entities, *concepts])
            else:
                body = _compose_source_body(raw_text, entities, concepts)
            # ADR-036 D4: deterministic preamble so EVERY source page carries
            # the AI-first contract, even on the degrade path.
            relevance = _default_preamble(title, date, origin)

        # Sources preserve BOTH the canonical URL(s) and the raw path —
        # deterministic, taken from the raw text, never from the model.
        sources = [*urls[:3], rel_path]
        tags = _derive_tags(rel_path, [name for name, _ in concept_items])
        page_md = render_source_page(
            title=title,
            date=date,
            sources=sources,
            tags=tags,
            origin=origin,
            body=body,
            source_count=1,
            relevance=relevance,
        )

        # ADR-036 D2: keep a ## History ledger when re-synthesizing an
        # existing (synthesized) source page. No-op when read_page is absent
        # or the prior page is the worker's mechanical envelope.
        prior_source = await self._read_prior(source_page_path)
        if prior_source is not None:
            page_md = accrete_source_page(prior_source, page_md, now=now).text

        confirmed_source = await self.write_page(source_page_path, page_md)
        # ADR-036 D4: the relevance note is the purpose-built index 1-liner.
        await self.update_index(
            confirmed_source, "sources", _index_summary(relevance, body, title), 1
        )

        entity_paths: list[str] = []
        for name, description in entity_items:
            path = f"wiki/entities/{slugify(name)}.md"
            content = render_entity_page(
                name=name,
                date=date,
                source_page_path=confirmed_source,
                mention=description,
                origin=origin,
            )
            content, count = await self._accrete_mention(path, content, now=now)
            try:
                confirmed = await self.write_page(path, content)
            except PageWriteSkippedError as e:
                log.warning("synthesis.page_skipped page_path=%s reason=%s", path, e)
                continue
            await self.update_index(confirmed, "entities", f"Entity extracted from {title}", count)
            entity_paths.append(confirmed)

        concept_paths: list[str] = []
        for name, description in concept_items:
            path = f"wiki/concepts/{slugify(name)}.md"
            content = render_concept_page(
                name=name,
                date=date,
                source_page_path=confirmed_source,
                mention=description,
                origin=origin,
            )
            content, count = await self._accrete_mention(path, content, now=now)
            try:
                confirmed = await self.write_page(path, content)
            except PageWriteSkippedError as e:
                log.warning("synthesis.page_skipped page_path=%s reason=%s", path, e)
                continue
            await self.update_index(confirmed, "concepts", f"Concept extracted from {title}", count)
            concept_paths.append(confirmed)

        # The raw rel_path MUST appear verbatim: the Hermes batch
        # detector treats a raw file as processed when its path
        # substring-matches the log or index.
        details = (
            f"Created 1 source, {len(entity_paths)} entities, "
            f"{len(concept_paths)} concepts from {rel_path}. "
            f"Source page: {confirmed_source}."
        )
        await self.append_log("ingest", f"Ingested {title}", details)

        return SynthesisOutcome(
            source_page_path=confirmed_source,
            entity_page_paths=tuple(entity_paths),
            concept_page_paths=tuple(concept_paths),
            llm_extracted=extraction is not None,
            log_details=details,
        )

    async def _read_prior(self, page_path: str) -> str | None:
        """Read an existing page for accretion (ADR-036). Returns None when
        no reader is wired (degrade-first: overwrite as before) or on ANY
        read failure — accretion must never break the ingest hot-path."""
        if self.read_page is None:
            return None
        try:
            return await self.read_page(page_path)
        except Exception:  # noqa: BLE001 — a read failure degrades to overwrite.
            log.warning("synthesis.read_prior_failed page_path=%s", page_path, exc_info=True)
            return None

    async def _accrete_mention(self, path: str, content: str, *, now: datetime) -> tuple[str, int]:
        """Merge a fresh entity/concept render with its prior page if one
        exists (ADR-036 D1). Returns (page_text, effective_source_count).
        Falls back to (content, 1) when there is no prior page."""
        prior = await self._read_prior(path)
        if prior is None:
            return content, 1
        acc = accrete_mention_page(prior, content, now=now)
        return acc.text, acc.source_count


# --------------------------------------------------------------------------- #
# Composition helpers (deterministic)                                          #
# --------------------------------------------------------------------------- #


def _derive_title(raw_text: str, rel_path: str) -> str:
    body = strip_frontmatter(raw_text)
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip()
            if candidate:
                return candidate[:120]
    stem = Path(rel_path).stem.strip()
    return stem[:120] or "Untitled source"


def _derive_tags(rel_path: str, concepts: list[str]) -> list[str]:
    parts = Path(rel_path).parts
    bucket = parts[1] if len(parts) >= 3 and parts[0] == "raw" else ""
    tags = ["ingested"]
    if bucket:
        tags.append(bucket)
    for concept in concepts[:2]:
        tags.append(slugify(concept))
    return tags


def _compose_source_body(raw_text: str, entities: list[str], concepts: list[str]) -> str:
    """Deterministic body: full raw prose (frontmatter stripped) with
    entities/concepts wikilinked. Preserves image refs and URLs by
    construction — the degrade path never summarises."""
    body = strip_frontmatter(raw_text).strip()
    return wikilink_terms(body, [*entities, *concepts])


def _append_missing_refs(body: str, raw_text: str) -> str:
    """Deterministic guarantee: every image ref and URL from the raw
    source survives into the page body even when the model summary
    omits them — re-appended by CODE, never trusted to the model."""
    out = body.rstrip()
    missing_images = [ref for ref in extract_image_refs(raw_text) if ref not in out]
    if missing_images:
        out += "\n\n" + "\n\n".join(missing_images)
    missing_urls = [url for url in extract_urls(raw_text) if url not in out]
    if missing_urls:
        out += "\n\n" + "\n".join(f"Source: {url}" for url in missing_urls)
    return out


def _summary_of(body: str, title: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "![", "[[", "---", "|", ">")):
            continue
        if len(stripped) > _MAX_SUMMARY_CHARS:
            return stripped[:_MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"
        return stripped
    return title


def _default_preamble(title: str, date: str, origin: str) -> str:
    """Deterministic AI-first relevance note for the degrade path (ADR-036 D4),
    so every source page carries the contract without an LLM call."""
    return (
        f"Source captured {date} ({origin}). The body below preserves the "
        f"source with its key entities and concepts wikilinked."
    )


def _index_summary(relevance: str, body: str, title: str) -> str:
    """The index 1-liner: the first sentence of the relevance preamble when
    present (ADR-036 D4), else the first prose line of the body."""
    rel = relevance.strip()
    if rel:
        first = _SENTENCE_RE.split(rel, maxsplit=1)[0].strip()
        if len(first) > _MAX_SUMMARY_CHARS:
            return first[:_MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"
        return first
    return _summary_of(body, title)


def _mention_of(raw_text: str, term: str) -> str:
    """First lines of the raw text that mention ``term`` — keeps the
    degrade-path entity/concept stub factual without any generation."""
    body = strip_frontmatter(raw_text)
    hits: list[str] = []
    for line in body.splitlines():
        if term in line or term.lower() in line.lower():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                hits.append(stripped)
        if len(hits) >= _MAX_MENTION_LINES:
            break
    return "\n\n".join(hits)
