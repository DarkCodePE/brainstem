"""
Tier value type for the 3-tier model router per [PRD-008 Model Routing](../../docs/PRD-008-model-routing.md)
and [ADR-013 Model router policy](../../docs/ADR-013-model-router-policy.md).

The router routes every LLM call to one of three logical tiers:

- ``Tier.REASONING`` — slow, expensive, high-quality. Used for the
  Memory Tree seal worker (PRD-004 FR-4), architecture decisions, deep
  query synthesis, and contradiction detection. Default provider per
  ADR-013: Claude Sonnet 4.5 with OpenRouter and Ollama ``qwen2.5:14b``
  fallbacks.
- ``Tier.FAST`` — cheap, low-latency. Used for routine ingest
  classification, orchestrator routing, lint scans, orphan detection,
  and the rest of the "trivial decision" call sites. Default provider:
  Claude Haiku 3.5.
- ``Tier.VISION`` — multimodal-capable. Used when the task payload
  contains images (screenshots, PDFs with figures, Excalidraw
  exports). Default provider: Claude Sonnet 4.5 (vision) with
  Gemini 1.5 Flash and Ollama ``llava:13b`` fallbacks.

This module deliberately keeps the value type **free of provider
choice**. The mapping ``Tier → ModelBackend`` lives in
``wiki_routing.router.ModelRouter.providers`` so a test can wire a
``StubBackend`` against ``Tier.REASONING`` without touching anything
else.

``Tier`` is an ``enum.Enum`` (not ``StrEnum``) so equality is
identity-based and downstream code can use ``match`` exhaustively.
"""

from __future__ import annotations

from enum import Enum


class Tier(Enum):
    """Logical model tier. See module docstring for the per-tier policy."""

    REASONING = "reasoning"
    """High-cost, high-quality. Seal summaries, deep query synthesis."""

    FAST = "fast"
    """Low-cost, low-latency. Ingest classification, orchestrator routing."""

    VISION = "vision"
    """Multimodal. Anything where the prompt contains an image."""

    def __str__(self) -> str:
        return self.value


__all__ = ["Tier"]
