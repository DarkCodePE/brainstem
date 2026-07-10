"""
``summariser_factory`` — assemble the production ``Summariser`` for the
M2 seal worker per [SPEC-009 §"Wire-up surface for M3-S2"](../../docs/SPEC-009-model-router.md)
and [PRD-008 FR-7](../../docs/PRD-008-model-routing.md).

The seal worker (`wiki_memory.seal_worker.SealWorker`) accepts any
``Summariser`` in its constructor. M2-S4 shipped with ``NullSummariser``
as the default because the router didn't exist yet. M3-S1 added the
``RouterSummariser`` adapter so an LLM-driven path is possible. This
module is the **wire-in**: it inspects the environment and returns the
right Summariser instance, defaulting to a router-backed composite when
provider keys are configured and to ``NullSummariser`` when they aren't.

### Why a factory, not a constructor argument

The seal worker can't import ``wiki_routing`` directly (PRD-004 R-1 +
SPEC-009 §"Why this lives in wiki_routing and not wiki_memory" — the
memory substrate must not depend on a specific LLM strategy). The
factory pushes that wiring decision up one level: callers (the agent
wiring, daemon entrypoints, CLI) call ``build_default_summariser()``
and pass the result into ``SealWorker(summariser=...)``. The seal
worker itself stays unchanged.

### Production router source

PR #102 (closing #37) shipped ``wiki_routing.factory.default_router()``
which reads ``~/.sbw/config.toml`` and wires the **production** chain:
DeepSeek V4 Flash on FAST, Anthropic Sonnet on REASONING, cost ceiling
of $10/day, telemetry to ``~/.sbw/state/router_telemetry.db``, and
``KeywordOnlyBackend`` as the terminal link so a fully-offline deployment
still returns something. This factory now delegates to it so the seal
flow honours the user's ``config.toml`` (overrides, model overrides,
budget) instead of building a parallel router with different defaults.

### Fallback chain semantics

When a real provider is available, the returned Summariser is a
``CompositeSummariser`` of:

1. ``RouterSummariser`` wrapping the production ``ModelRouter`` from
   ``default_router()`` — every tier already terminates in
   ``KeywordOnlyBackend`` per the router-factory contract.
2. ``NullSummariser`` as the seal-specific deterministic fallback so a
   ``RouterSummariser`` exception (citation parse failure, etc.) still
   produces a citation-preserving stub summary.

### No network at construction time

``build_default_summariser`` does **not** make any HTTP calls. It only
inspects env vars, picks a tier policy, and assembles objects. The first
real call happens at ``await router.call(...)`` time, deep inside
``RouterSummariser.summarise``. This is important for tests and for the
daemon's startup latency: importing this module is free.

### Env vars consulted (indirectly, via ``default_router``)

- ``OPENROUTER_API_KEY`` — required for the FAST tier (DeepSeek) and
  the REASONING secondary.
- ``ANTHROPIC_API_KEY`` — required for the REASONING and VISION primary.
- See ``wiki_routing.config`` for the full ``~/.sbw/config.toml`` schema
  (per-tier overrides, budget ceilings, model overrides).

The router still terminates each chain in ``KeywordOnlyBackend`` so a
fully-offline run never crashes; this factory only short-circuits to
``NullSummariser`` when no cloud key is present, because preserving
citations matters more for the seal flow than emitting a degraded
keyword-only first sentence.
"""

from __future__ import annotations

import os

from wiki_memory.summariser import CompositeSummariser, NullSummariser, Summariser


def _has_env(name: str) -> bool:
    """Return ``True`` iff ``name`` is set in the environment to a
    non-empty value. We treat whitespace-only values as unset because a
    stray export with no value is almost always a misconfiguration."""
    value = os.environ.get(name)
    return value is not None and value.strip() != ""


def _can_build_router() -> bool:
    """Decide whether the env carries enough to build a useful router.

    The minimum is **one** real cloud provider key — either Anthropic or
    OpenRouter. Without either, the production router would only have
    Ollama + ``KeywordOnlyBackend`` in its chain, and the seal worker is
    better served by ``NullSummariser`` directly: NullSummariser
    preserves chunk citations (``[[chunk:SHA8]]`` markers) which the
    faithfulness gate verifies, while KeywordOnly would emit a degraded
    first-sentence summary with no citation guarantee."""
    return _has_env("ANTHROPIC_API_KEY") or _has_env("OPENROUTER_API_KEY")


def build_default_summariser(*, prefer_llm: bool = True) -> Summariser:
    """Build the Summariser the seal worker uses by default.

    Parameters
    ----------
    prefer_llm:
        When ``True`` (the default), the factory tries to assemble a
        router-backed summariser if at least one real provider key is
        present in the environment. When ``False``, the factory skips
        the router entirely and returns ``NullSummariser`` — useful for
        offline tests, deterministic dry-runs, and the
        "no-LLM-please" deployment knob.

    Returns
    -------
    Summariser
        Either a ``CompositeSummariser(RouterSummariser, NullSummariser)``
        when the environment supports LLM dispatch, or a bare
        ``NullSummariser`` otherwise. Both satisfy the
        ``wiki_memory.summariser.Summariser`` Protocol so the seal
        worker doesn't care which one it gets.

    Notes
    -----
    The factory does **not** make any LLM call at construction time.
    Provider stubs are only **constructed** here; they're invoked when
    ``router.call(...)`` runs deep inside ``RouterSummariser.summarise``.

    The composite wrap is the key invariant: a real call that raises
    (rate-limit, 5xx, network, citation-parse failure on the JSON
    envelope) falls back to the deterministic ``NullSummariser`` instead
    of the seal flow failing outright.
    """
    if not prefer_llm:
        return NullSummariser()

    if not _can_build_router():
        # No cloud provider keys configured. Returning NullSummariser
        # keeps the seal flow working offline (M2-S4 behaviour) and
        # preserves the citation surface the faithfulness gate cares
        # about — which KeywordOnlyBackend wouldn't guarantee.
        return NullSummariser()

    # Defer imports of ``wiki_routing`` until we actually need them —
    # keeps the seal-worker import path clean for deployments that don't
    # have the router installed (none today, but the module boundary
    # makes this safer to refactor).
    from wiki_routing.factory import default_router
    from wiki_routing.router_summariser import RouterSummariser

    router = default_router()

    # The composite is what gives us the "LLM with deterministic
    # fallback" semantic the docstring promises. The RouterSummariser
    # runs first; if it raises (citation-parse failure, anything the
    # router's own KeywordOnly terminal didn't already catch), the
    # NullSummariser kicks in and the seal still succeeds with a
    # citation-preserving stub summary.
    return CompositeSummariser(RouterSummariser(router=router), NullSummariser())


__all__ = ["build_default_summariser"]
