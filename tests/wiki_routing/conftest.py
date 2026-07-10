"""Shared fixtures for ``wiki_routing`` tests.

Most tests construct their own ``StubBackend`` because each test
case wants to assert on a different canned response. The fixtures
here are the common scaffolding only — a default policy, a router
factory, a budget factory, and storage fixtures for the
``RouterSummariser`` integration tests (which need a live
``ContentStore`` + ``TreeNodeStore`` so the seal worker can write
through them)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from wiki_memory.content_store import ContentStore
from wiki_memory.tree_nodes import TreeNodeStore
from wiki_routing.cost_ceiling import CostBudget
from wiki_routing.policy import RoutingPolicy
from wiki_routing.router import (
    BackendResponse,
    ModelRouter,
    StubBackend,
)
from wiki_routing.tiers import Tier


@pytest_asyncio.fixture
async def content_store(tmp_path: Path) -> AsyncIterator[ContentStore]:
    s = ContentStore(tmp_path / "content_store.db")
    await s.init()
    try:
        yield s
    finally:
        await s.close()


@pytest_asyncio.fixture
async def tree_store(tmp_path: Path) -> AsyncIterator[TreeNodeStore]:
    s = TreeNodeStore(tmp_path / "tree_nodes.db")
    await s.init()
    try:
        yield s
    finally:
        await s.close()


@pytest.fixture
def policy() -> RoutingPolicy:
    """Default policy — no overrides."""
    return RoutingPolicy()


@pytest.fixture
def fast_stub() -> StubBackend:
    return StubBackend(
        label="stub:fast",
        response=BackendResponse(
            text="fast response",
            tokens_in=20,
            tokens_out=10,
            cost_usd=0.0002,
        ),
        quote_usd=0.0002,
    )


@pytest.fixture
def reasoning_stub() -> StubBackend:
    return StubBackend(
        label="stub:reasoning",
        response=BackendResponse(
            text="reasoning response",
            tokens_in=200,
            tokens_out=400,
            cost_usd=0.005,
        ),
        quote_usd=0.005,
    )


@pytest.fixture
def vision_stub() -> StubBackend:
    return StubBackend(
        label="stub:vision",
        response=BackendResponse(
            text="vision response",
            tokens_in=300,
            tokens_out=100,
            cost_usd=0.003,
        ),
        quote_usd=0.003,
    )


@pytest.fixture
def router(
    policy: RoutingPolicy,
    fast_stub: StubBackend,
    reasoning_stub: StubBackend,
    vision_stub: StubBackend,
) -> ModelRouter:
    """Router wired with three stubs, one per tier. No budget."""
    return ModelRouter(
        policy=policy,
        providers={
            Tier.FAST: fast_stub,
            Tier.REASONING: reasoning_stub,
            Tier.VISION: vision_stub,
        },
    )


@pytest.fixture
def generous_budget() -> CostBudget:
    """A budget too large to ever trip in tests."""
    return CostBudget(max_per_task_usd=10.0, max_per_day_usd=100.0)
