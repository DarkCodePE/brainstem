"""Tests for ``wiki_routing.router.ModelRouter`` — the wired router."""

from __future__ import annotations

import asyncio

import pytest

from wiki_routing.cost_ceiling import CostBudget, CostCeilingError
from wiki_routing.fallback import FallbackChain
from wiki_routing.policy import RoutingPolicy, TaskDescriptor
from wiki_routing.router import (
    BackendResponse,
    Message,
    ModelBackend,
    ModelRouter,
    StubBackend,
)
from wiki_routing.tiers import Tier


def _msgs(text: str = "hi") -> list[Message]:
    return [Message(role="user", content=text)]


def _task(intent: str = "ingest", **kw: object) -> TaskDescriptor:
    return TaskDescriptor(
        intent=intent,  # type: ignore[arg-type]
        estimated_input_tokens=kw.get("tokens", 50),  # type: ignore[arg-type]
        has_image=bool(kw.get("has_image", False)),
        caller_priority=kw.get("caller_priority", "background"),  # type: ignore[arg-type]
    )


class TestConstruction:
    def test_empty_providers_rejected(self, policy: RoutingPolicy) -> None:
        with pytest.raises(ValueError):
            ModelRouter(policy=policy, providers={})

    def test_single_backend_wrapped_in_chain(
        self, policy: RoutingPolicy, fast_stub: StubBackend
    ) -> None:
        router = ModelRouter(
            policy=policy,
            providers={Tier.FAST: fast_stub},
        )
        # Internal shape: a one-entry FallbackChain. Test indirectly by
        # confirming the call succeeds against that single backend.
        result = asyncio.run(router.call(_task("ingest"), messages=_msgs()))
        assert result.backend_label == "stub:fast"
        assert result.fallback_steps == 0

    def test_explicit_fallback_chain_preserved(
        self, policy: RoutingPolicy, fast_stub: StubBackend, reasoning_stub: StubBackend
    ) -> None:
        chain = FallbackChain[ModelBackend]([fast_stub, reasoning_stub])
        router = ModelRouter(
            policy=policy,
            providers={Tier.FAST: chain},
        )
        result = asyncio.run(router.call(_task("ingest"), messages=_msgs()))
        assert result.backend_label == "stub:fast"


class TestTierDispatch:
    @pytest.mark.asyncio
    async def test_ingest_goes_to_fast(self, router: ModelRouter) -> None:
        result = await router.call(_task("ingest"), messages=_msgs())
        assert result.tier is Tier.FAST
        assert result.backend_label == "stub:fast"

    @pytest.mark.asyncio
    async def test_seal_goes_to_reasoning(self, router: ModelRouter) -> None:
        result = await router.call(_task("seal"), messages=_msgs())
        assert result.tier is Tier.REASONING

    @pytest.mark.asyncio
    async def test_image_goes_to_vision(self, router: ModelRouter) -> None:
        result = await router.call(_task("ingest", has_image=True), messages=_msgs())
        assert result.tier is Tier.VISION

    @pytest.mark.asyncio
    async def test_query_foreground_goes_to_reasoning(self, router: ModelRouter) -> None:
        result = await router.call(_task("query", caller_priority="foreground"), messages=_msgs())
        assert result.tier is Tier.REASONING

    @pytest.mark.asyncio
    async def test_query_background_goes_to_fast(self, router: ModelRouter) -> None:
        result = await router.call(_task("query", caller_priority="background"), messages=_msgs())
        assert result.tier is Tier.FAST


class TestRouterResultShape:
    @pytest.mark.asyncio
    async def test_result_has_all_telemetry_fields(self, router: ModelRouter) -> None:
        result = await router.call(_task("ingest"), messages=_msgs())
        # PRD-008 FR-5 fields.
        assert result.text == "fast response"
        assert result.tier is Tier.FAST
        assert result.tokens_in == 20
        assert result.tokens_out == 10
        assert result.cost_usd == pytest.approx(0.0002)
        assert result.backend_label == "stub:fast"
        assert result.fallback_steps == 0

    @pytest.mark.asyncio
    async def test_latency_ms_populated_and_nonneg(self, router: ModelRouter) -> None:
        result = await router.call(_task("ingest"), messages=_msgs())
        assert result.latency_ms >= 0.0
        # Sanity: a stub call should be fast (well under a second).
        assert result.latency_ms < 1000.0

    @pytest.mark.asyncio
    async def test_messages_passed_through(
        self, router: ModelRouter, fast_stub: StubBackend
    ) -> None:
        msgs = [Message(role="user", content="hello-marker")]
        await router.call(_task("ingest"), messages=msgs)
        assert len(fast_stub.calls) == 1
        assert fast_stub.calls[0].messages == msgs


class TestMissingTierWiring:
    @pytest.mark.asyncio
    async def test_policy_chose_unconfigured_tier_raises(
        self, policy: RoutingPolicy, fast_stub: StubBackend
    ) -> None:
        # Only FAST wired; seal will route to REASONING and not find a
        # backend. We want a loud failure, not silent downgrade.
        router = ModelRouter(
            policy=policy,
            providers={Tier.FAST: fast_stub},
        )
        with pytest.raises(LookupError):
            await router.call(_task("seal"), messages=_msgs())


class TestBudgetIntegration:
    @pytest.mark.asyncio
    async def test_budget_charges_actual_cost(
        self, policy: RoutingPolicy, fast_stub: StubBackend, generous_budget: CostBudget
    ) -> None:
        router = ModelRouter(
            policy=policy,
            providers={Tier.FAST: fast_stub},
            budget=generous_budget,
        )
        await router.call(_task("ingest"), messages=_msgs())
        assert generous_budget.spent_today() == pytest.approx(0.0002)

    @pytest.mark.asyncio
    async def test_budget_refuses_too_expensive_call(
        self, policy: RoutingPolicy, reasoning_stub: StubBackend
    ) -> None:
        budget = CostBudget(max_per_task_usd=0.001, max_per_day_usd=10.0)
        router = ModelRouter(
            policy=policy,
            providers={Tier.REASONING: reasoning_stub, Tier.FAST: reasoning_stub},
            budget=budget,
        )
        with pytest.raises(CostCeilingError) as ei:
            await router.call(_task("seal"), messages=_msgs())
        assert ei.value.kind == "per_task"
        # Backend was never called.
        assert reasoning_stub.calls == []

    @pytest.mark.asyncio
    async def test_no_budget_no_enforcement(self, router: ModelRouter) -> None:
        # Default fixture wires no budget — even a notional huge spend
        # is fine.
        result = await router.call(_task("seal"), messages=_msgs())
        assert result.tier is Tier.REASONING


class TestTierForHelper:
    def test_tier_for_returns_policy_decision(self, router: ModelRouter) -> None:
        assert router.tier_for(_task("seal")) is Tier.REASONING
        assert router.tier_for(_task("ingest")) is Tier.FAST
        assert router.tier_for(_task("ingest", has_image=True)) is Tier.VISION


class TestBackendResponseImmutability:
    def test_backend_response_is_frozen(self) -> None:
        r = BackendResponse(text="x", tokens_in=1, tokens_out=1, cost_usd=0.0)
        with pytest.raises(Exception):
            r.text = "y"  # type: ignore[misc]
