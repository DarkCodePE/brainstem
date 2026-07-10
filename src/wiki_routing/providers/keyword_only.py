"""
``KeywordOnlyBackend`` — terminal fallback for the router per issue #37 AC:
"Fallback chain: cloud → Ollama → keyword-only."

When every LLM backend (cloud + local Ollama) is unreachable, this
backend produces a **deterministic, LLM-free** response by extracting
the first sentence + a keyword summary from the prompt. Quality is
intentionally low — this is the "degraded gracefully" branch, not a
real summariser. It exists so the seal worker (PRD-004 FR-4) never
crashes during outages; the user gets a placeholder summary marked
``[keyword-only]`` they can re-seal later when LLMs are reachable.
"""

from __future__ import annotations

import re
from collections import Counter

from wiki_routing.cost_ceiling import CostQuote
from wiki_routing.router import BackendResponse, Message


class KeywordOnlyBackend:
    """Final-fallback ``ModelBackend`` that does NOT call any LLM.

    Always returns a structured "[keyword-only]"-tagged summary built
    from the most-frequent content words in the user message. Cost is
    always 0; latency is microseconds.
    """

    label = "keyword-only:terminal-fallback"
    """Stable label so the router telemetry can spot which calls fell
    through the entire chain."""

    def __init__(self, *, max_keywords: int = 8) -> None:
        if max_keywords < 1:
            raise ValueError("max_keywords must be >= 1")
        self._max_keywords = max_keywords

    async def generate(self, messages: list[Message]) -> BackendResponse:
        text = _last_user_text(messages)
        keywords = _top_keywords(text, k=self._max_keywords)
        first_sentence = _first_sentence(text)
        summary = (
            "[keyword-only] " + first_sentence if first_sentence else "[keyword-only] (no content)"
        )
        if keywords:
            summary += f"\nKeywords: {', '.join(keywords)}"
        return BackendResponse(
            text=summary,
            tokens_in=_rough_token_count(text),
            tokens_out=_rough_token_count(summary),
            cost_usd=0.0,
        )

    def quote(self, messages: list[Message]) -> CostQuote:
        # Free by construction. The budget check is a no-op for this backend.
        return CostQuote(estimated_usd=0.0, backend_label=self.label)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


# Conservative stopword list — kept minimal so per-language summaries
# stay readable. Real NLP belongs upstream of this fallback.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "by",
        "from",
        "as",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "them",
        "their",
        "our",
        "my",
        "el",
        "la",
        "los",
        "las",
        "y",
        "o",
        "pero",
        "si",
        "de",
        "a",
        "en",
        "es",
        "son",
        "fue",
        "ser",
        "este",
        "esta",
        "estos",
        "estas",
    }
)


def _last_user_text(messages: list[Message]) -> str:
    """Pull the most recent user message's text. Falls back to all messages
    joined for non-standard shapes (system-only, etc.)."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return msg.content
    return "\n".join(m.content for m in messages if m.content)


def _first_sentence(text: str) -> str:
    """Conservative sentence split — first hit on ``.``, ``!``, ``?``, ``\\n``.
    Caps at 200 chars so a long-line wall doesn't blow context."""
    if not text:
        return ""
    text = text.strip()
    for sep in (". ", "! ", "? ", "\n"):
        idx = text.find(sep)
        if idx > 0:
            return text[:idx].strip()[:200]
    return text[:200].strip()


_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")


def _top_keywords(text: str, *, k: int) -> list[str]:
    """Most-frequent non-stopword tokens, lowercased, deduped."""
    words = (m.group(0).lower() for m in _WORD_RE.finditer(text))
    counter: Counter[str] = Counter(w for w in words if w not in _STOPWORDS)
    return [w for w, _ in counter.most_common(k)]


def _rough_token_count(text: str) -> int:
    """Crude token estimate. Real tokenisation is the cloud provider's job."""
    return max(1, len(text) // 4)


__all__ = ["KeywordOnlyBackend"]
