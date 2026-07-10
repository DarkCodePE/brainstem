"""
Deterministic ≤3k-token chunking per [PRD-004 FR-1](../../docs/PRD-004-memory-tree.md).

Token counting uses a 4-chars-per-token heuristic (the approximation used
by OpenAI tokenisers for English+code). This is good enough for chunking
*decisions* (which side of the 3k boundary a paragraph falls on). It is
**not** good enough for billing — billing-grade counting requires the
provider's actual tokeniser. Switch to `tiktoken` (cl100k_base) for that
when PRD-008 model routing lands.

Chunks split at paragraph boundaries (blank-line separated). A single
paragraph larger than the soft target gets pushed into its own chunk;
a single paragraph larger than the hard cap is split mid-paragraph at
sentence boundaries (.!?), then mid-sentence at whitespace if all else
fails. Chunks are keyed by sha256 of the body so a re-ingest of unchanged
text produces zero new chunks (PRD-004 AC-4 idempotency hard guarantee).

The 4-chars-per-token heuristic is documented at the call sites.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final

#: Default soft target. Most chunks land here; the chunker prefers a chunk
#: at the target over splitting a paragraph mid-text.
DEFAULT_TARGET_TOKENS: Final[int] = 2500

#: Hard cap from PRD-004 FR-1. Single paragraphs larger than this are split
#: at sentence boundaries (then whitespace) to fit the cap.
DEFAULT_HARD_CAP_TOKENS: Final[int] = 3000

#: Conservative heuristic used across the codebase for chunking decisions.
#: 4 chars/token matches cl100k_base average for English; for CJK it
#: undercounts (real cost ≈ 1 char/token) — we accept the slack because
#: this is a chunking knob, not a billing one.
CHARS_PER_TOKEN: Final[int] = 4

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")

#: ADR-036 D4: the AI-first relevance preamble heading. ``chunk_page`` isolates
#: this section into its own chunk so it is a clean embedding target for recall.
PREAMBLE_HEADING: Final[str] = "## For future Claude"

#: Matches the preamble section: the heading plus the single relevance paragraph
#: that follows it (the rendered ``relevance`` is always one whitespace-collapsed
#: line — see ``wiki_synthesis.templates.render_source_page``), up to the next
#: blank line or end of text.
_PREAMBLE_SECTION_RE = re.compile(
    r"(?ms)^" + re.escape(PREAMBLE_HEADING) + r"[ \t]*\n\n.+?(?=\n\n|\Z)"
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A canonical Memory Tree chunk."""

    sha256: str
    """Content fingerprint — 64-char hex. Stable across re-ingests."""

    chunk_index: int
    """0-based ordinal within the source. Tie-breaks identical bodies."""

    body: str
    """Verbatim chunk text. The vault-side md mirror writes this."""

    token_count: int
    """Estimated tokens (4-char heuristic). Use for budget calculations only."""


def count_tokens(text: str) -> int:
    """Estimate token count for *text* using the 4-char heuristic.

    Use only for chunking decisions and token-budget retrieval. Real
    LLM token counts will differ — switch to a provider-specific
    tokeniser for billing or model-fit checks.
    """
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_paragraph(
    paragraph: str,
    *,
    hard_cap_tokens: int,
) -> list[str]:
    """Split a paragraph that exceeds the hard cap into smaller pieces.

    Strategy:
    1. Split at sentence boundaries (`.!?` followed by whitespace).
    2. Pack sentences greedily until adding another would exceed hard_cap.
    3. If a single sentence is still over the cap, fall through to a
       whitespace split.
    """
    sentences = _SENTENCE_BOUNDARY.split(paragraph)
    cap_chars = hard_cap_tokens * CHARS_PER_TOKEN

    pieces: list[str] = []
    buf = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(buf) + len(sentence) + 1 <= cap_chars:
            buf = f"{buf} {sentence}".strip() if buf else sentence
        else:
            if buf:
                pieces.append(buf)
            if len(sentence) <= cap_chars:
                buf = sentence
            else:
                # Sentence itself too long — split on whitespace.
                buf = ""
                words = sentence.split()
                wb = ""
                for w in words:
                    if len(wb) + len(w) + 1 <= cap_chars:
                        wb = f"{wb} {w}".strip() if wb else w
                    else:
                        if wb:
                            pieces.append(wb)
                        wb = w
                if wb:
                    buf = wb
    if buf:
        pieces.append(buf)
    return pieces


def chunk_text(
    text: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    hard_cap_tokens: int = DEFAULT_HARD_CAP_TOKENS,
) -> list[Chunk]:
    """Split *text* into deterministic, sha-keyed chunks.

    The chunker never emits a chunk over `hard_cap_tokens`. Most chunks
    land near `target_tokens`. Identical input produces identical output;
    feeding the same chunks back into a content store keyed by `sha256`
    is a zero-op (PRD-004 idempotency hard guarantee).
    """
    if not text.strip():
        return []

    target_chars = target_tokens * CHARS_PER_TOKEN
    hard_cap_chars = hard_cap_tokens * CHARS_PER_TOKEN

    paragraphs = [p.strip() for p in _PARAGRAPH_BOUNDARY.split(text) if p.strip()]

    # Pack paragraphs greedily into chunk buffers.
    raw_chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > hard_cap_chars:
            if buf:
                raw_chunks.append(buf)
                buf = ""
            raw_chunks.extend(_split_paragraph(para, hard_cap_tokens=hard_cap_tokens))
            continue
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) > target_chars:
            if buf:
                raw_chunks.append(buf)
            buf = para
        else:
            buf = candidate
    if buf:
        raw_chunks.append(buf)

    chunks: list[Chunk] = []
    for i, body in enumerate(raw_chunks):
        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        chunks.append(
            Chunk(
                sha256=sha,
                chunk_index=i,
                body=body,
                token_count=count_tokens(body),
            )
        )
    return chunks


def chunk_page(
    text: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    hard_cap_tokens: int = DEFAULT_HARD_CAP_TOKENS,
) -> list[Chunk]:
    """Page-aware chunking (ADR-036 D4).

    Isolates the ``## For future Claude`` relevance preamble into its own chunk
    so it is a clean embedding target — recall retrieves the preamble on
    relevance queries instead of it being diluted inside a greedily-packed
    chunk 0. The text before the preamble (frontmatter + title) and the body
    after it are each chunked normally with :func:`chunk_text`, and the result
    is re-indexed in document order. Content-addressed ``sha256`` values are
    preserved, so re-ingest of unchanged text is still idempotent.

    Falls back to :func:`chunk_text` — byte-for-byte identical chunking — when
    no preamble is present (entity/concept pages, pre-ADR-036 pages, arbitrary
    text), so this is a safe drop-in for ``chunk_text`` at the seal seam.
    """
    match = _PREAMBLE_SECTION_RE.search(text)
    if match is None:
        return chunk_text(text, target_tokens=target_tokens, hard_cap_tokens=hard_cap_tokens)

    regions = (text[: match.start()], match.group(0), text[match.end() :])
    raw: list[Chunk] = []
    for region in regions:
        raw.extend(chunk_text(region, target_tokens=target_tokens, hard_cap_tokens=hard_cap_tokens))
    return [
        Chunk(sha256=c.sha256, chunk_index=i, body=c.body, token_count=c.token_count)
        for i, c in enumerate(raw)
    ]
