"""
Routing policy — pick a ``Tier`` from a ``TaskDescriptor`` per [ADR-013
Model router policy](../../docs/ADR-013-model-router-policy.md).

The policy is **declared, not learned** (PRD-008 §Non-goals). The
caller states what it wants done (``intent``), whether the payload
contains an image (``has_image``), and how urgent the call is
(``caller_priority``); the policy returns a single ``Tier``. The
``ModelRouter`` then looks up the backend wired for that tier.

Default decision matrix (from highest priority to lowest):

1. ``has_image=True`` OR ``intent="vision"`` → ``Tier.VISION``
   (a missing vision-capable backend is a configuration error the
   router surfaces at construction time, not a policy bug).
2. ``intent="seal"`` or ``intent="draft"`` → ``Tier.REASONING`` regardless
   of priority (sealing is the most-quality-sensitive call per PRD-004 R-1;
   drafting publishes under the user's own identity per ADR-021 — both want
   the best model).
3. ``intent="query"`` AND ``caller_priority="foreground"`` →
   ``Tier.REASONING`` (interactive query gets the best model).
4. Everything else (``ingest``, ``lint``, background ``query``) →
   ``Tier.FAST``.

This matrix matches ADR-013's per-agent default mapping while
collapsing the agent-name dimension into ``intent``. The mapping is
overridable: pass ``overrides={intent: Tier}`` to the constructor to
pin a specific intent to a tier without modifying the matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wiki_routing.tiers import Tier

Intent = Literal["seal", "ingest", "query", "lint", "vision", "draft"]
"""Intent the call site declares.

Mirrors ADR-013's per-agent mapping condensed to five symbolic
intents. New intents should be added here (and to the policy
matrix below) rather than encoded as free-form strings — making the
policy table exhaustive is what keeps stale routes out of production."""

CallerPriority = Literal["foreground", "background"]
"""How urgent the caller is.

- ``foreground`` — a user is waiting (interactive query, manual seal).
  Policy biases toward ``Tier.REASONING`` for query-class intents.
- ``background`` — daemon work (autofetch, scheduled ingest, periodic
  lint). Policy biases toward ``Tier.FAST`` because latency budget is
  generous and cost ceiling pressure (per ADR-013) dominates."""


@dataclass(frozen=True, slots=True)
class TaskDescriptor:
    """Inputs the policy reads.

    Kept tiny on purpose: every field here must be something the call
    site already knows. Anything the call site can't observe (like
    "actual token count after prompt compression") belongs on the
    backend side, not in the policy.
    """

    intent: Intent
    """What the call site is trying to accomplish."""

    estimated_input_tokens: int
    """Pre-call token estimate. Reserved for the cost ceiling, not
    consulted by the default routing matrix. A future policy revision
    may downgrade very-large inputs from REASONING → FAST when the
    cost ceiling is hot (ADR-013 §"Cost ceiling enforcement")."""

    has_image: bool = False
    """True iff the prompt contains image content. Forces ``VISION``."""

    caller_priority: CallerPriority = "background"
    """Default to ``background`` — most calls in the system are daemon
    work, so ``foreground`` is the opt-in. This avoids accidentally
    burning REASONING quota on bulk-ingest jobs."""


class RoutingPolicy:
    """Decides the tier for a ``TaskDescriptor``.

    The constructor takes an optional ``overrides`` mapping that
    short-circuits the matrix. Overrides are applied **after** the
    vision check (you can't override an image task off VISION because
    a non-vision model can't process pixels). This matches ADR-013's
    "per-call override … still subject to the cost ceiling" guidance.

    Parameters
    ----------
    overrides:
        Mapping ``intent → Tier`` that overrides the default matrix
        for non-vision tasks. Useful when a deployment wants e.g.
        ``"ingest"`` on REASONING (high-quality wiki) or ``"query"``
        permanently on FAST (cost-constrained deployments).
    """

    def __init__(
        self,
        *,
        overrides: dict[Intent, Tier] | None = None,
    ) -> None:
        self._overrides: dict[Intent, Tier] = dict(overrides) if overrides else {}

    def route(self, task: TaskDescriptor) -> Tier:
        """Return the ``Tier`` to use for ``task``.

        The decision order is fixed:

        1. Image content forces VISION (override-proof — cannot be
           bypassed because non-vision backends would fail anyway).
        2. Explicit per-intent override from ``self._overrides``.
        3. Default matrix.
        """
        # 1. Vision is non-negotiable when there's an image in the prompt.
        if task.has_image or task.intent == "vision":
            return Tier.VISION

        # 2. Caller-supplied per-intent override.
        if task.intent in self._overrides:
            return self._overrides[task.intent]

        # 3. Default matrix.
        match task.intent:
            case "seal" | "draft":
                # Seal-time summarisation is the most quality-sensitive
                # call in the system (PRD-004 R-1 faithfulness). Drafting a
                # post composed under the user's own professional identity
                # (ADR-021 Phase 1) is equally quality-sensitive. Both go
                # to REASONING and never downgrade.
                return Tier.REASONING
            case "query":
                # Interactive queries deserve the best model. Background
                # queries (e.g. autofetch follow-ups) are fine on FAST.
                if task.caller_priority == "foreground":
                    return Tier.REASONING
                return Tier.FAST
            case "ingest" | "lint":
                # ADR-013: orchestrator/capture/review/index all on Fast.
                # Ingest classification + lint scans fit the same shape.
                return Tier.FAST
            case _:
                # Defensive default — should be unreachable thanks to
                # the ``Intent`` Literal, but ``match`` doesn't enforce
                # exhaustiveness at runtime so we keep a safe fallback.
                return Tier.FAST


__all__ = ["CallerPriority", "Intent", "RoutingPolicy", "TaskDescriptor"]
