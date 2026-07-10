"""
``wiki_routing`` — 3-tier LLM router for the Second Brain Wiki.

Implements [PRD-008 Model Routing](../../docs/PRD-008-model-routing.md)
and [ADR-013 Model router policy](../../docs/ADR-013-model-router-policy.md).

Public surface:

- ``Tier`` — three logical tiers (REASONING / FAST / VISION).
- ``RoutingPolicy``, ``TaskDescriptor`` — declared-not-learned routing
  table; given a task, returns a tier.
- ``CostBudget``, ``CostCeilingError`` — per-task + per-day cost
  enforcement.
- ``FallbackChain``, ``BackendError`` — ordered provider walk; tries
  each backend until one succeeds.
- ``ModelRouter`` — wires policy + budget + per-tier provider chains.
  Single seam every LLM call site uses.
- ``Message``, ``BackendResponse``, ``RouterResult``, ``ModelBackend``,
  ``StubBackend`` — the call/response value types and the test
  double.
- ``RouterSummariser`` — ``wiki_memory.summariser.Summariser``
  implementation backed by the router. Plug-in for ``SealWorker``.
- ``providers.{AnthropicBackend, OpenRouterBackend, OllamaBackend}``
  — concrete backends shipped as **stubs** until M3-S2.

Module layout mirrors PRD-008's recommended split (factory / policy
/ provider / etc.) but condensed to a single-package shape suitable
for the SBW codebase (no per-feature subpackages).
"""

from __future__ import annotations

from wiki_routing.cost_ceiling import CostBudget, CostCeilingError, CostQuote
from wiki_routing.fallback import BackendError, FallbackChain
from wiki_routing.policy import CallerPriority, Intent, RoutingPolicy, TaskDescriptor
from wiki_routing.router import (
    BackendResponse,
    Message,
    ModelBackend,
    ModelRouter,
    RouterResult,
    StubBackend,
)
from wiki_routing.router_summariser import RouterSummariser
from wiki_routing.tiers import Tier

__all__ = [
    "BackendError",
    "BackendResponse",
    "CallerPriority",
    "CostBudget",
    "CostCeilingError",
    "CostQuote",
    "FallbackChain",
    "Intent",
    "Message",
    "ModelBackend",
    "ModelRouter",
    "RouterResult",
    "RouterSummariser",
    "RoutingPolicy",
    "StubBackend",
    "TaskDescriptor",
    "Tier",
]
