"""Tests for ``wiki_routing.fallback.FallbackChain``."""

from __future__ import annotations

import pytest

from wiki_routing.fallback import BackendError, FallbackChain
from wiki_routing.router import BackendResponse, Message, StubBackend


def _msgs() -> list[Message]:
    return [Message(role="user", content="hi")]


class TestConstruction:
    def test_empty_chain_rejected(self) -> None:
        with pytest.raises(ValueError):
            FallbackChain[StubBackend]([])

    def test_backends_exposed_in_order(self) -> None:
        a = StubBackend(label="a")
        b = StubBackend(label="b")
        c = StubBackend(label="c")
        chain = FallbackChain([a, b, c])
        assert chain.backends == (a, b, c)


class TestPrimarySuccess:
    @pytest.mark.asyncio
    async def test_primary_succeeds_returns_result_and_zero_steps(self) -> None:
        a = StubBackend(
            label="primary",
            response=BackendResponse(text="ok", tokens_in=1, tokens_out=1, cost_usd=0.0),
        )
        b = StubBackend(label="secondary")
        chain = FallbackChain([a, b])
        result, used, steps = await chain.run(lambda be: be.generate(_msgs()))
        assert result.text == "ok"
        assert used is a
        assert steps == 0
        # Secondary not called.
        assert b.calls == []


class TestFallback:
    @pytest.mark.asyncio
    async def test_falls_back_when_primary_raises_backend_error(self) -> None:
        a = StubBackend(label="primary", raise_on_call=BackendError("429", kind="rate_limit"))
        b = StubBackend(
            label="secondary",
            response=BackendResponse(text="ok-b", tokens_in=1, tokens_out=1, cost_usd=0.0),
        )
        chain = FallbackChain([a, b])
        result, used, steps = await chain.run(lambda be: be.generate(_msgs()))
        assert result.text == "ok-b"
        assert used is b
        assert steps == 1
        # Both backends saw the call (primary first, then secondary).
        assert len(a.calls) == 1
        assert len(b.calls) == 1

    @pytest.mark.asyncio
    async def test_falls_through_multiple_failures(self) -> None:
        a = StubBackend(label="a", raise_on_call=BackendError("a-fail", kind="server"))
        b = StubBackend(label="b", raise_on_call=BackendError("b-fail", kind="timeout"))
        c = StubBackend(
            label="c",
            response=BackendResponse(text="c-ok", tokens_in=1, tokens_out=1, cost_usd=0.0),
        )
        chain = FallbackChain([a, b, c])
        result, used, steps = await chain.run(lambda be: be.generate(_msgs()))
        assert result.text == "c-ok"
        assert used is c
        assert steps == 2


class TestExhaustion:
    @pytest.mark.asyncio
    async def test_all_fail_raises_backend_error(self) -> None:
        a = StubBackend(label="a", raise_on_call=BackendError("a", kind="rate_limit"))
        b = StubBackend(label="b", raise_on_call=BackendError("b", kind="overloaded"))
        chain = FallbackChain([a, b])
        with pytest.raises(BackendError) as ei:
            await chain.run(lambda be: be.generate(_msgs()))
        # Final error inherits the last attempt's kind.
        assert ei.value.kind == "overloaded"
        # __cause__ chain preserves the inner error.
        assert isinstance(ei.value.__cause__, BackendError)

    @pytest.mark.asyncio
    async def test_non_backend_error_not_swallowed(self) -> None:
        # A programmer error (e.g. ValueError) should bubble immediately
        # — falling back from a coding mistake hides bugs.
        a = StubBackend(label="a", raise_on_call=ValueError("bug"))
        b = StubBackend(label="b")
        chain = FallbackChain([a, b])
        with pytest.raises(ValueError):
            await chain.run(lambda be: be.generate(_msgs()))
        # Second backend never tried.
        assert b.calls == []


class TestBackendError:
    def test_default_kind_is_unknown(self) -> None:
        e = BackendError("oops")
        assert e.kind == "unknown"

    def test_kind_preserved(self) -> None:
        e = BackendError("rate", kind="rate_limit")
        assert e.kind == "rate_limit"
