"""
``ModelRouter`` — wires policy + cost ceiling + provider-specific
backends per [PRD-008 §"Provider abstraction"](../../docs/PRD-008-model-routing.md)
and [ADR-013 §"Routing API (Python)"](../../docs/ADR-013-model-router-policy.md).

The router is the single seam every call site uses to talk to an LLM:

    result = await router.call(task, messages=msgs)

Inside ``call`` the router:

1. Asks ``RoutingPolicy`` for a ``Tier``.
2. Looks up the ``ModelBackend`` (or ``FallbackChain[ModelBackend]``)
   registered for that tier.
3. Pre-flights the ``CostBudget`` (if configured) against the
   pre-call cost estimate the backend exposes.
4. Dispatches through the fallback primitive.
5. Charges the budget post-call.
6. Wraps the outcome in a ``RouterResult`` with telemetry fields
   (tier, tokens, cost, latency, fallback steps).

The router is **stateless beyond the budget counter** — multiple
async tasks can call ``router.call`` concurrently without explicit
locking. The ``CostBudget`` rollover code is not atomic across day
boundaries, which is acceptable for a single-user MVP (PRD-008
§"Non-functional requirements" leaves multi-tenant fairness out).

Tests use a ``StubBackend`` (provided here) that returns canned text
and exposes a call log; no real provider calls happen until M3-S2
swaps the stub providers in ``wiki_routing.providers`` for real API
clients.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from wiki_routing.cost_ceiling import CostBudget, CostCeilingError, CostQuote
from wiki_routing.fallback import BackendError, FallbackChain
from wiki_routing.policy import RoutingPolicy, TaskDescriptor
from wiki_routing.tiers import Tier


@dataclass(frozen=True, slots=True)
class Message:
    """One role/content pair. Kept minimal on purpose — the real
    provider layers expand this into provider-native shapes (Anthropic
    Messages, OpenAI ChatCompletion, etc.); the router only needs role
    + body and an optional image flag for telemetry."""

    role: str
    """``"system"`` | ``"user"`` | ``"assistant"`` — provider clients
    are free to coerce to their native vocabulary."""

    content: str
    """Text body. Multimodal payloads carry image bytes via a side
    channel that's part of M3-S2's provider wiring, not this module."""


@dataclass(frozen=True, slots=True)
class BackendResponse:
    """What a ``ModelBackend.generate`` returns when it succeeds.

    The router converts this into a ``RouterResult`` plus a
    ``charge`` against the budget; the backend itself doesn't talk
    to the budget so it stays trivially mockable."""

    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    """Actual cost (not an estimate) of this call. Computed by the
    backend from the provider's pricing table × real token counts."""


@runtime_checkable
class ModelBackend(Protocol):
    """Provider-agnostic backend.

    Real implementations live under ``wiki_routing.providers``; tests
    use ``StubBackend`` from this module. The protocol intentionally
    exposes a pre-call ``quote`` so the budget can refuse before any
    real provider call burns a token."""

    label: str
    """Short identifier like ``"anthropic:claude-sonnet-4.5"``. Used
    in telemetry and error messages. Never contains secret material."""

    async def generate(self, messages: list[Message]) -> BackendResponse:
        """Run the prompt. Raise ``BackendError`` on retryable
        failures (rate limits, 5xx, timeouts) so the fallback chain
        can step. Raise any other ``Exception`` to surface immediately
        (auth errors, programmer mistakes)."""
        ...

    def quote(self, messages: list[Message]) -> CostQuote:
        """Pre-call cost estimate. Should include a small safety margin
        so over-budget calls are caught **before** dispatch."""
        ...


@dataclass(frozen=True, slots=True)
class RouterResult:
    """The router's response. Combines the backend's text with the
    telemetry fields PRD-008 FR-5 wants in the ``routing_calls`` log:
    ``intent, tier, model_used, fallback_steps, prompt_tokens,
    completion_tokens, latency_ms, cost_usd, success``."""

    text: str
    tier: Tier
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: float
    backend_label: str
    """Which backend actually answered (the primary or a fallback).
    Mirrors PRD-008's ``model_used`` telemetry field."""

    fallback_steps: int
    """0 for the primary; 1 for the first fallback; etc."""


@dataclass
class _RegisteredTier:
    """Internal: a tier may be wired to either a single backend or a
    fallback chain. We normalise both shapes to ``FallbackChain`` at
    construction time so the dispatch path is uniform."""

    chain: FallbackChain[ModelBackend]


class ModelRouter:
    """Wires ``RoutingPolicy`` + per-tier ``ModelBackend`` providers
    + optional ``CostBudget``.

    Parameters
    ----------
    policy:
        Decides the ``Tier`` for a given task.
    providers:
        Mapping ``Tier → ModelBackend | FallbackChain[ModelBackend]``.
        Single backends are wrapped in a one-entry chain so the
        dispatch path is uniform.
    budget:
        Optional cost ceiling. When ``None`` (default), no budget
        enforcement happens — used by tests and by the local-only
        deployment that runs on Ollama.

    Notes
    -----
    The router does **not** maintain provider health state in this
    cut — that lands with the telemetry surface in a follow-up sprint
    once the real providers are wired (M3-S2). Until then, a failing
    backend just walks the chain on every call.
    """

    def __init__(
        self,
        *,
        policy: RoutingPolicy,
        providers: Mapping[Tier, ModelBackend | FallbackChain[ModelBackend]],
        budget: CostBudget | None = None,
        telemetry: object | None = None,
    ) -> None:
        """Optional ``telemetry`` is a duck-typed object exposing
        ``record(tier, backend_label, cost_usd, success)``. The factory
        wires a ``wiki_routing.telemetry.RouterTelemetry`` so the
        ``sbw doctor`` aggregates persist across processes per #37 AC.
        Tests inject a no-op object."""
        if not providers:
            raise ValueError("ModelRouter requires at least one tier mapping")
        self._policy = policy
        self._budget = budget
        self._telemetry = telemetry
        self._tiers: dict[Tier, _RegisteredTier] = {}
        for tier, backend in providers.items():
            if isinstance(backend, FallbackChain):
                chain = backend
            else:
                chain = FallbackChain[ModelBackend]([backend])
            self._tiers[tier] = _RegisteredTier(chain=chain)

    @property
    def policy(self) -> RoutingPolicy:
        """Exposed read-only for tests."""
        return self._policy

    @property
    def budget(self) -> CostBudget | None:
        """Exposed read-only for tests / status CLI."""
        return self._budget

    def tier_for(self, task: TaskDescriptor) -> Tier:
        """Short-hand: ``policy.route(task)``. Exposed because callers
        may want to introspect the tier before deciding whether to
        actually dispatch (e.g. preview / dry-run)."""
        return self._policy.route(task)

    async def call(
        self,
        task: TaskDescriptor,
        *,
        messages: list[Message],
    ) -> RouterResult:
        """Run ``task`` and return a populated ``RouterResult``.

        Pre-call ``CostBudget`` rejection raises ``CostCeilingError``.
        A ``FallbackChain`` exhaustion raises ``BackendError``."""
        tier = self._policy.route(task)
        registered = self._tiers.get(tier)
        if registered is None:
            # Misconfiguration: policy returned a tier nobody wired a
            # backend for. Better to fail loudly than silently downgrade.
            raise LookupError(
                f"no backend registered for tier {tier!r} "
                f"(policy chose it for intent={task.intent!r})"
            )

        # Pre-flight the budget against the primary's quote. We pick
        # the primary here because falling all the way through the
        # chain to a cheaper local model on every refused-too-expensive
        # call would mask budget bugs. ADR-013 expects the user to
        # widen the ceiling instead.
        primary = registered.chain.backends[0]
        if self._budget is not None:
            self._budget.check(primary.quote(messages))

        started = time.perf_counter()
        response, used_backend, steps = await registered.chain.run(
            lambda backend: backend.generate(messages),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        if self._budget is not None:
            self._budget.charge(response.cost_usd)

        if self._telemetry is not None:
            try:
                self._telemetry.record(
                    tier=str(tier),
                    backend_label=used_backend.label,
                    cost_usd=response.cost_usd,
                    success=True,
                )
            except Exception:  # noqa: BLE001 -- telemetry errors never block the call
                pass

        return RouterResult(
            text=response.text,
            tier=tier,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            cost_usd=response.cost_usd,
            latency_ms=latency_ms,
            backend_label=used_backend.label,
            fallback_steps=steps,
        )


# --------------------------------------------------------------------------- #
# StubBackend — test double                                                   #
# --------------------------------------------------------------------------- #


@dataclass
class StubCall:
    """One recorded call against ``StubBackend``."""

    messages: list[Message]


class StubBackend:
    """In-memory backend used by tests and the ``router_summariser``
    smoke wiring. Returns a canned response and records every call
    on ``self.calls`` so tests can assert.

    Parameters
    ----------
    label:
        ``ModelBackend.label`` value. Defaults to ``"stub"``.
    response:
        Canned ``BackendResponse``. Defaults to a small, free,
        deterministic response so most tests don't have to specify
        every field.
    quote_usd:
        Pre-call cost estimate returned by ``quote``. Defaults to the
        canned response's ``cost_usd``.
    raise_on_call:
        If set, ``generate`` raises this exception instead of returning
        the canned response. Used by fallback tests to make a backend
        deliberately fail.
    """

    def __init__(
        self,
        *,
        label: str = "stub",
        response: BackendResponse | None = None,
        quote_usd: float | None = None,
        raise_on_call: BaseException | None = None,
    ) -> None:
        self.label = label
        self._response = response or BackendResponse(
            text="stub-response",
            tokens_in=10,
            tokens_out=5,
            cost_usd=0.0001,
        )
        self._quote_usd = quote_usd if quote_usd is not None else self._response.cost_usd
        self._raise_on_call = raise_on_call
        self.calls: list[StubCall] = []

    async def generate(self, messages: list[Message]) -> BackendResponse:
        self.calls.append(StubCall(messages=list(messages)))
        if self._raise_on_call is not None:
            raise self._raise_on_call
        return self._response

    def quote(self, messages: list[Message]) -> CostQuote:
        return CostQuote(estimated_usd=self._quote_usd, backend_label=self.label)


# Re-export for convenience so callers don't have to know which module
# each name lives in.
__all__ = [
    "BackendError",
    "BackendResponse",
    "CostBudget",
    "CostCeilingError",
    "CostQuote",
    "FallbackChain",
    "Message",
    "ModelBackend",
    "ModelRouter",
    "RouterResult",
    "RoutingPolicy",
    "StubBackend",
    "StubCall",
    "TaskDescriptor",
    "Tier",
]


# Generic instantiation helper kept for tests that want to construct a
# typed empty chain (rare; mostly the router does this internally).
def _empty_chain(backends: list[ModelBackend]) -> FallbackChain[ModelBackend]:
    return FallbackChain[ModelBackend](backends)
