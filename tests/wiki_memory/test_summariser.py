"""
Tests for `wiki_memory.summariser` — protocol + reference implementations.
"""

from __future__ import annotations

import hashlib

import pytest

from wiki_memory.summariser import (
    CompositeSummariser,
    NullSummariser,
    Summariser,
    SummaryPart,
    SummaryResult,
)


def _mk_parts(n: int = 3) -> list[SummaryPart]:
    parts = []
    for i in range(n):
        body = f"child body {i} with some content"
        sha = hashlib.sha256(body.encode()).hexdigest()
        parts.append(SummaryPart(sha256=sha, body=body, token_count=10))
    return parts


class TestNullSummariser:
    @pytest.mark.asyncio
    async def test_satisfies_summariser_protocol(self) -> None:
        s = NullSummariser()
        assert isinstance(s, Summariser)

    @pytest.mark.asyncio
    async def test_cites_all_input_shas(self) -> None:
        parts = _mk_parts(5)
        result = await NullSummariser().summarise(parts)
        assert set(result.cited_shas) == {p.sha256 for p in parts}

    @pytest.mark.asyncio
    async def test_body_contains_chunk_references(self) -> None:
        parts = _mk_parts(3)
        result = await NullSummariser().summarise(parts)
        for p in parts:
            short = p.sha256[:8]
            assert f"[[chunk:{short}]]" in result.body

    @pytest.mark.asyncio
    async def test_sha_matches_body(self) -> None:
        parts = _mk_parts(2)
        result = await NullSummariser().summarise(parts)
        recomputed = hashlib.sha256(result.body.encode()).hexdigest()
        assert result.sha256 == recomputed

    @pytest.mark.asyncio
    async def test_deterministic_for_same_input(self) -> None:
        parts = _mk_parts(4)
        a = await NullSummariser().summarise(parts)
        b = await NullSummariser().summarise(parts)
        assert a.sha256 == b.sha256
        assert a.body == b.body

    @pytest.mark.asyncio
    async def test_empty_input_still_produces_summary(self) -> None:
        result = await NullSummariser().summarise([])
        assert result.cited_shas == ()
        assert result.body  # non-empty (header only)

    @pytest.mark.asyncio
    async def test_custom_header(self) -> None:
        result = await NullSummariser(header="Topic: AI agents").summarise(_mk_parts(2))
        assert "Topic: AI agents" in result.body


class TestCompositeSummariser:
    @pytest.mark.asyncio
    async def test_first_delegate_wins(self) -> None:
        class Marker:
            def __init__(self, marker: str) -> None:
                self.marker = marker

            async def summarise(self, parts) -> SummaryResult:  # type: ignore[no-untyped-def]
                body = f"by {self.marker}"
                return SummaryResult(
                    body=body,
                    sha256=hashlib.sha256(body.encode()).hexdigest(),
                    parent_token_count=1,
                    cited_shas=(),
                )

        result = await CompositeSummariser(Marker("A"), Marker("B")).summarise(_mk_parts(1))
        assert "by A" in result.body

    @pytest.mark.asyncio
    async def test_falls_back_when_first_raises(self) -> None:
        class Bad:
            async def summarise(self, parts) -> SummaryResult:  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

        class Good:
            async def summarise(self, parts) -> SummaryResult:  # type: ignore[no-untyped-def]
                body = "fallback"
                return SummaryResult(
                    body=body,
                    sha256=hashlib.sha256(body.encode()).hexdigest(),
                    parent_token_count=1,
                    cited_shas=(),
                )

        result = await CompositeSummariser(Bad(), Good()).summarise(_mk_parts(1))
        assert "fallback" in result.body

    @pytest.mark.asyncio
    async def test_raises_when_all_fail(self) -> None:
        class Bad:
            async def summarise(self, parts) -> SummaryResult:  # type: ignore[no-untyped-def]
                raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await CompositeSummariser(Bad(), Bad()).summarise(_mk_parts(1))

    def test_empty_chain_rejected(self) -> None:
        with pytest.raises(ValueError):
            CompositeSummariser()
