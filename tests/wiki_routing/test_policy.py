"""Tests for ``wiki_routing.policy.RoutingPolicy`` — the routing matrix."""

from __future__ import annotations

import pytest

from wiki_routing.policy import RoutingPolicy, TaskDescriptor
from wiki_routing.tiers import Tier


def _task(
    *,
    intent: str = "ingest",
    has_image: bool = False,
    caller_priority: str = "background",
    tokens: int = 100,
) -> TaskDescriptor:
    return TaskDescriptor(
        intent=intent,  # type: ignore[arg-type]
        estimated_input_tokens=tokens,
        has_image=has_image,
        caller_priority=caller_priority,  # type: ignore[arg-type]
    )


class TestVisionRouting:
    def test_has_image_forces_vision(self) -> None:
        # has_image always wins, regardless of intent.
        policy = RoutingPolicy()
        for intent in ["seal", "ingest", "query", "lint", "vision", "draft"]:
            assert policy.route(_task(intent=intent, has_image=True)) is Tier.VISION, (
                f"intent={intent!r} should route to VISION when has_image"
            )

    def test_intent_vision_routes_vision(self) -> None:
        policy = RoutingPolicy()
        assert policy.route(_task(intent="vision")) is Tier.VISION

    def test_vision_cannot_be_overridden_to_non_vision(self) -> None:
        # Even with an override map pointing vision → FAST, the
        # image-content branch fires first and forces VISION.
        policy = RoutingPolicy(overrides={"vision": Tier.FAST})  # type: ignore[dict-item]
        assert policy.route(_task(intent="ingest", has_image=True)) is Tier.VISION


class TestSealRouting:
    def test_seal_always_reasoning(self) -> None:
        # Seal is the highest-quality call site. Background or
        # foreground — doesn't matter.
        policy = RoutingPolicy()
        assert policy.route(_task(intent="seal")) is Tier.REASONING
        assert policy.route(_task(intent="seal", caller_priority="foreground")) is Tier.REASONING


class TestDraftRouting:
    def test_draft_always_reasoning(self) -> None:
        # ADR-021 Phase 1: LinkedIn draft generation composes under the
        # user's professional identity — quality-sensitive like seal, so
        # REASONING regardless of priority.
        policy = RoutingPolicy()
        assert policy.route(_task(intent="draft")) is Tier.REASONING
        assert policy.route(_task(intent="draft", caller_priority="background")) is Tier.REASONING


class TestQueryRouting:
    def test_foreground_query_reasoning(self) -> None:
        policy = RoutingPolicy()
        assert policy.route(_task(intent="query", caller_priority="foreground")) is Tier.REASONING

    def test_background_query_fast(self) -> None:
        policy = RoutingPolicy()
        assert policy.route(_task(intent="query", caller_priority="background")) is Tier.FAST


class TestIngestAndLintRouting:
    @pytest.mark.parametrize("intent", ["ingest", "lint"])
    def test_routes_to_fast(self, intent: str) -> None:
        policy = RoutingPolicy()
        assert policy.route(_task(intent=intent)) is Tier.FAST

    @pytest.mark.parametrize("priority", ["foreground", "background"])
    def test_ingest_priority_does_not_promote(self, priority: str) -> None:
        # ingest is cheap by policy; foreground does not promote it.
        policy = RoutingPolicy()
        assert policy.route(_task(intent="ingest", caller_priority=priority)) is Tier.FAST


class TestOverrides:
    def test_override_promotes_ingest_to_reasoning(self) -> None:
        policy = RoutingPolicy(overrides={"ingest": Tier.REASONING})  # type: ignore[dict-item]
        assert policy.route(_task(intent="ingest")) is Tier.REASONING

    def test_override_demotes_query_foreground(self) -> None:
        # Override beats the default matrix.
        policy = RoutingPolicy(overrides={"query": Tier.FAST})  # type: ignore[dict-item]
        assert policy.route(_task(intent="query", caller_priority="foreground")) is Tier.FAST

    def test_override_does_not_affect_vision_decision(self) -> None:
        # Vision still wins on image content.
        policy = RoutingPolicy(overrides={"ingest": Tier.REASONING})  # type: ignore[dict-item]
        assert policy.route(_task(intent="ingest", has_image=True)) is Tier.VISION

    def test_empty_overrides_default_matrix(self) -> None:
        policy = RoutingPolicy(overrides={})
        assert policy.route(_task(intent="ingest")) is Tier.FAST


class TestTaskDescriptor:
    def test_descriptor_is_frozen(self) -> None:
        t = _task()
        with pytest.raises(Exception):  # FrozenInstanceError subclass of AttributeError
            t.intent = "seal"  # type: ignore[misc]

    def test_default_caller_priority_is_background(self) -> None:
        t = TaskDescriptor(intent="ingest", estimated_input_tokens=10)
        assert t.caller_priority == "background"

    def test_default_has_image_false(self) -> None:
        t = TaskDescriptor(intent="ingest", estimated_input_tokens=10)
        assert t.has_image is False
