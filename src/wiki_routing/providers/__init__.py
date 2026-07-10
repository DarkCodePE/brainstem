"""
Concrete ``ModelBackend`` provider implementations per [PRD-008
§"Provider abstraction"](../../../docs/PRD-008-model-routing.md).

M3-S3 ships **real** implementations for all three providers
(``AnthropicProvider`` / ``OpenRouterProvider`` / ``OllamaProvider``).
Each provider:

1. Implements the ``ModelBackend`` Protocol shape (``label`` + async
   ``generate`` + sync ``quote``).
2. Speaks to its upstream via ``httpx.AsyncClient`` with a configurable
   transport so tests can inject ``httpx.MockTransport`` and avoid the
   real network.
3. Maps HTTP failures to ``BackendError(kind=...)`` for the retryable
   classes and bubbles non-retryable failures (auth, programmer bugs)
   through the fallback chain.

The ``*Backend`` aliases (``AnthropicBackend``, ``OpenRouterBackend``,
``OllamaBackend``) are kept for backwards compatibility with M3-S1
imports; new code should prefer the ``*Provider`` names that match the
SPEC-009 §"M3 Sprint 3" terminology.
"""

from __future__ import annotations

from wiki_routing.providers.anthropic import AnthropicBackend, AnthropicProvider
from wiki_routing.providers.ollama import OllamaBackend, OllamaProvider
from wiki_routing.providers.openrouter import OpenRouterBackend, OpenRouterProvider

__all__ = [
    "AnthropicBackend",
    "AnthropicProvider",
    "OllamaBackend",
    "OllamaProvider",
    "OpenRouterBackend",
    "OpenRouterProvider",
]
