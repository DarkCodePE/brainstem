"""Router-driven structured extraction (issue #180, ADR-035 D3).

The Hermes batch prompt — the specification of record — asked the LLM to
"extract key entities (people, tools, projects) and concepts (ideas,
patterns, frameworks)... the most important ones". The first live batch
of the deterministic port produced junk pages (``llm.md``, ``gpu.md``,
``kv.md``) because heuristics cannot rank by importance. This module
restores LLM extraction as the PRIMARY path: ONE structured router call
returns ``{summary, entities[], concepts[]}`` as JSON, parsed
defensively. ``None`` on ANY failure (no router, transport error, bad
JSON, empty extraction) — the caller then degrades to the heuristic
path, so the budget invariant stays ≤1 LLM call per file.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wiki_synthesis.extractors import strip_frontmatter

log = logging.getLogger("wiki_synthesis.structured")

__all__ = [
    "ARCHETYPE_DEFAULT",
    "ARCHETYPE_PAPER",
    "EXTRACTION_SYSTEM_PROMPT",
    "PAPER_EXTRACTION_SYSTEM_PROMPT",
    "ExtractedConcept",
    "ExtractedEntity",
    "StructuredExtraction",
    "detect_archetype",
    "extract_structured",
]

_ENTITY_TYPES = frozenset({"person", "tool", "project", "org"})
_DEFAULT_ENTITY_TYPE = "tool"
_MAX_BODY_CHARS = 24_000
_MAX_NAME_LEN = 80
_MAX_DESCRIPTION_LEN = 600
_MAX_RELEVANCE_LEN = 600
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

EXTRACTION_SYSTEM_PROMPT = (
    "You are a knowledge-base analyst. Analyze the source document and "
    "return ONE JSON object — no prose, no markdown fences — with exactly "
    "this shape:\n"
    "\n"
    '{"summary": "2-4 sentence summary of the document\'s core value '
    '(claims, numbers, results)",\n'
    ' "relevance": "2-3 sentences written FOR A FUTURE AI READER: what this '
    "document is, why it matters, and any time-sensitivity — so the reader "
    'can decide whether to open the page without reading all of it",\n'
    ' "entities": [{"name": "...", "type": "person|tool|project|org", '
    '"description": "1-2 factual sentences grounded in the document"}],\n'
    ' "concepts": [{"name": "...", "description": "1-2 factual sentences '
    'grounded in the document"}]}\n'
    "\n"
    "Rules:\n"
    "- relevance: write it for retrieval by a future AI, not for a human. "
    "Do NOT invent dates or facts; omit time-sensitivity if the document "
    "states none.\n"
    "- entities: the MOST IMPORTANT named things (people, tools, projects, "
    "organizations). At most 5 — fewer, better items beat many noisy ones.\n"
    "- concepts: the MOST IMPORTANT ideas, patterns, or frameworks. "
    "At most 5.\n"
    "- NEVER include generic technical terms with no standalone knowledge "
    "value (e.g. LLM, GPU, CPU, KV, AI, API, JSON, HTTP).\n"
    "- Use the document's own wording for names; do not invent facts.\n"
    "- Return an empty list when the document has no meaningful entities "
    "or concepts."
)

# PRD-015 FR-6: the paper archetype. Same JSON envelope as the default
# prompt (one router call, one parse path), but the summary is
# paper-shaped — mirrors the ADR-024 repo-archetype lesson of
# 2026-06-04: capture the numbers, or downstream posts have nothing to
# say. Wikilinking of the extracted names into the summary stays in
# CODE (``wikilink_terms``), never trusted to the model.
PAPER_EXTRACTION_SYSTEM_PROMPT = (
    "You are a research-paper analyst for a personal knowledge base. "
    "Analyze the paper and return ONE JSON object — no prose, no markdown "
    "fences — with exactly this shape:\n"
    "\n"
    '{"summary": "markdown with the five sections described below",\n'
    ' "relevance": "2-3 plain-text sentences for a FUTURE AI READER: what '
    "the paper contributes and when it is worth opening — no markdown, no "
    'invented numbers",\n'
    ' "entities": [{"name": "...", "type": "person|tool|project|org", '
    '"description": "1-2 factual sentences grounded in the paper"}],\n'
    ' "concepts": [{"name": "...", "description": "1-2 factual sentences '
    'grounded in the paper"}]}\n'
    "\n"
    "The summary MUST be markdown with exactly these sections:\n"
    "- **Problem** — the gap or question the paper addresses.\n"
    "- **Method** — the approach, in 2-3 sentences.\n"
    "- **Key results** — the main findings WITH CONCRETE NUMBERS (metrics, "
    "dataset sizes, speedups, accuracy deltas) taken verbatim from the "
    "paper. Never invent or round numbers.\n"
    "- **Limitations** — what the authors concede or the evaluation misses.\n"
    "- **Why it matters** — one paragraph relating the paper to the "
    "concepts you extracted, using their exact names so they can be "
    "wikilinked to existing wiki concepts.\n"
    "\n"
    "Rules:\n"
    "- entities: the MOST IMPORTANT named things (authors, models, systems, "
    "datasets, organizations). At most 5 — fewer, better items beat many "
    "noisy ones.\n"
    "- concepts: the MOST IMPORTANT ideas, techniques, or frameworks the "
    "paper contributes to or builds on. At most 5.\n"
    "- NEVER include generic technical terms with no standalone knowledge "
    "value (e.g. LLM, GPU, CPU, KV, AI, API, JSON, HTTP).\n"
    "- Use the paper's own wording for names; do not invent facts.\n"
    "- Return an empty list when the paper has no meaningful entities "
    "or concepts."
)

ARCHETYPE_DEFAULT = "default"
ARCHETYPE_PAPER = "paper"

_SYSTEM_PROMPTS = {
    ARCHETYPE_DEFAULT: EXTRACTION_SYSTEM_PROMPT,
    ARCHETYPE_PAPER: PAPER_EXTRACTION_SYSTEM_PROMPT,
}

_FRONTMATTER_BLOCK_RE = re.compile(r"\A---\s*\n(?P<fm>.*?)\n---\s*(?:\n|\Z)", re.DOTALL)
_PAPER_TYPE_RE = re.compile(r"^type:\s*[\"']?paper[\"']?\s*$", re.MULTILINE)


def detect_archetype(raw_text: str, rel_path: str = "") -> str:
    """Pick the extraction archetype for one raw file (PRD-015 FR-6).

    ``paper`` when the file's own YAML frontmatter carries ``type:
    paper`` (the FR-4 sidecar contract) OR when it arrived under
    ``raw/papers/`` (the worker's SEC-05 envelope strips raw
    frontmatter before synthesis sees it, so the path is the honest
    signal on the daemon route). Everything else stays ``default``.
    """
    match = _FRONTMATTER_BLOCK_RE.match(raw_text)
    if match and _PAPER_TYPE_RE.search(match.group("fm")):
        return ARCHETYPE_PAPER
    parts = Path(rel_path).parts
    if parts and parts[0] == "raw":
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] == "papers":
        return ARCHETYPE_PAPER
    return ARCHETYPE_DEFAULT


@dataclass(frozen=True)
class ExtractedEntity:
    """One named thing the model judged important."""

    name: str
    type: str
    description: str


@dataclass(frozen=True)
class ExtractedConcept:
    """One idea/pattern/framework the model judged important."""

    name: str
    description: str


@dataclass(frozen=True)
class StructuredExtraction:
    """Validated output of the single structured router call."""

    summary: str
    entities: tuple[ExtractedEntity, ...]
    concepts: tuple[ExtractedConcept, ...]
    # ADR-036 D4: an AI-first relevance note (2-3 sentences, written for a
    # future AI reader). Always non-empty after parsing — falls back to the
    # first sentences of the summary when the model omits it. No extra call.
    relevance: str = ""


async def extract_structured(
    raw_text: str,
    *,
    title: str,
    router: Any | None,
    max_entities: int = 5,
    max_concepts: int = 5,
    archetype: str = ARCHETYPE_DEFAULT,
) -> StructuredExtraction | None:
    """ONE structured router call → validated extraction, or ``None``.

    ``None`` on ANY failure: missing router, router exception, empty
    response, unparseable JSON, wrong shape, or an extraction with no
    entities AND no concepts. The caller never retries — this is the
    file's single LLM call.
    """
    if router is None:
        return None
    try:
        text = await _call_router(raw_text, title=title, router=router, archetype=archetype)
    except Exception:  # noqa: BLE001 — degrade on any router/transport failure.
        log.warning("structured.extraction_call_failed title=%r", title, exc_info=True)
        return None
    if not isinstance(text, str) or not text.strip():
        # Distinct from unparseable: the call "succeeded" but yielded no
        # text (e.g. reasoning models returning empty content). Without
        # this line the degrade is invisible in the journal.
        log.warning("structured.extraction_empty_text title=%r archetype=%r", title, archetype)
        return None
    extraction = _parse(text, max_entities=max_entities, max_concepts=max_concepts)
    if extraction is None:
        log.warning("structured.extraction_unparseable title=%r", title)
    return extraction


async def _call_router(
    raw_text: str, *, title: str, router: Any, archetype: str = ARCHETYPE_DEFAULT
) -> str | None:
    from wiki_routing import Message, TaskDescriptor  # local import — optional dep

    system_prompt = _SYSTEM_PROMPTS.get(archetype, EXTRACTION_SYSTEM_PROMPT)
    body = strip_frontmatter(raw_text).strip()[:_MAX_BODY_CHARS]
    user_content = f"Title: {title}\n\nDocument:\n{body}"
    task = TaskDescriptor(
        intent="ingest",
        estimated_input_tokens=max(1, (len(system_prompt) + len(user_content)) // 4),
        has_image=False,
        caller_priority="background",
    )
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_content),
    ]
    caller = getattr(router, "call", None) or getattr(router, "route", None)
    if caller is None:
        return None
    result = await caller(task, messages=messages)
    text = getattr(result, "text", None)
    return text if isinstance(text, str) else None


# --------------------------------------------------------------------------- #
# Defensive parsing (plain checks, no schema library)                          #
# --------------------------------------------------------------------------- #


def _parse(text: str, *, max_entities: int, max_concepts: int) -> StructuredExtraction | None:
    obj = _first_json_object(text)
    if obj is None:
        return None
    summary = obj.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return None
    entities = _parse_entities(obj.get("entities"), max_entities)
    concepts = _parse_concepts(obj.get("concepts"), max_concepts)
    if not entities and not concepts:
        return None  # empty extraction — heuristic may do better.
    summary_clean = summary.strip()
    # ADR-036 D4: relevance is model-provided when present, else the first
    # sentences of the summary — never empty, never invented.
    relevance = _clean_str(obj.get("relevance"), _MAX_RELEVANCE_LEN) or _first_sentences(
        summary_clean
    )
    return StructuredExtraction(
        summary=summary_clean,
        entities=tuple(entities),
        concepts=tuple(concepts),
        relevance=relevance,
    )


def _first_sentences(text: str, n: int = 2) -> str:
    """First ``n`` sentences of ``text`` (whitespace-collapsed). Deterministic
    fallback for the relevance preamble — no model call, no invented content."""
    collapsed = " ".join(text.split())
    parts = _SENTENCE_SPLIT_RE.split(collapsed)
    return " ".join(parts[:n]).strip()[:_MAX_RELEVANCE_LEN]


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the FIRST JSON object embedded anywhere in ``text`` —
    tolerates prose preambles and markdown code fences around it."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_entities(items: Any, limit: int) -> list[ExtractedEntity]:
    out: list[ExtractedEntity] = []
    seen: set[str] = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = _clean_str(item.get("name"), _MAX_NAME_LEN)
        if not name or name.lower() in seen:
            continue
        etype = _clean_str(item.get("type"), _MAX_NAME_LEN).lower()
        if etype not in _ENTITY_TYPES:
            etype = _DEFAULT_ENTITY_TYPE
        seen.add(name.lower())
        out.append(
            ExtractedEntity(
                name=name,
                type=etype,
                description=_clean_str(item.get("description"), _MAX_DESCRIPTION_LEN),
            )
        )
        if len(out) >= limit:  # cap in CODE, not just in the prompt.
            break
    return out


def _parse_concepts(items: Any, limit: int) -> list[ExtractedConcept]:
    out: list[ExtractedConcept] = []
    seen: set[str] = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = _clean_str(item.get("name"), _MAX_NAME_LEN)
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(
            ExtractedConcept(
                name=name,
                description=_clean_str(item.get("description"), _MAX_DESCRIPTION_LEN),
            )
        )
        if len(out) >= limit:  # cap in CODE, not just in the prompt.
            break
    return out


def _clean_str(value: Any, max_len: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:max_len].strip()
