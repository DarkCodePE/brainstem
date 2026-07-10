"""
Property-based retrieval scenarios (issue #131).

Inspired by OpenHuman ``src/openhuman/memory_tree/retrieval/benchmarks.rs:1-728``.
Each test builds a synthetic content_store, asserts a behavioural
invariant of the retrieval layer, and cleans up. No real wiki content;
no LLM judge; no network. Pure structural correctness.

These complement ``evals/m3.5/datasets/telegram_e2e.yaml`` (which tests
specific real-wiki questions). Together: dataset golden = "does the
right *answer* come out?", these scenarios = "does the retrieval layer
*behave* correctly under edge cases?".
"""

from __future__ import annotations

import time

import pytest

from wiki_memory.chunker import chunk_text


def _mk_chunks(text: str, *, target: int = 50, cap: int = 200) -> list:
    return chunk_text(text, target_tokens=target, hard_cap_tokens=cap)


# --------------------------------------------------------------------------- #
# Scenario 1 — citation provenance                                           #
# --------------------------------------------------------------------------- #


class TestCitationProvenance:
    """Retrieved chunks must preserve their ``source_id`` and
    ``chunk_index`` intact so the agent can cite back. A retrieval
    that returns the body but loses the address can't ground citations."""

    @pytest.mark.asyncio
    async def test_fts_preserves_source_id_and_index(self, fresh_stack) -> None:
        content, _ = fresh_stack
        # 3 distinct sources, each with a unique-token chunk
        sources = {
            "src-alpha.md": "this chunk talks about apricot fruits",
            "src-beta.md": "this chunk talks about blueberry season",
            "src-gamma.md": "this chunk talks about cranberry harvest",
        }
        for sid, body in sources.items():
            chunks = _mk_chunks(body)
            await content.insert_many(source_id=sid, chunks=chunks)

        hits = await content.search_fts("apricot")
        assert len(hits) >= 1
        # The returned source_id matches the one we inserted under
        assert hits[0].source_id == "src-alpha.md"
        # chunk_index is intact (0 for single-chunk source)
        assert hits[0].chunk_index == 0
        # sha matches the chunker output exactly (no mutation in flight)
        assert hits[0].sha256 == _mk_chunks(sources["src-alpha.md"])[0].sha256


# --------------------------------------------------------------------------- #
# Scenario 2 — stale supersedes fresh                                        #
# --------------------------------------------------------------------------- #


class TestStaleVsFresh:
    """When a source is re-ingested with new content, recall must not
    silently return only the stale chunk. Either both should surface
    (so the agent can resolve) or the freshest only — never stale-only."""

    @pytest.mark.asyncio
    async def test_recall_surfaces_both_versions_when_both_present(self, fresh_stack) -> None:
        content, _ = fresh_stack
        source = "src-evolving.md"
        v1 = _mk_chunks("the meeting is at three pm")
        v2 = _mk_chunks("the meeting is at five pm")
        # Both chunks share source_id but have different shas (different bodies)
        await content.insert_many(source_id=source, chunks=v1)
        await content.insert_many(source_id=source, chunks=v2)

        # Query a token both share
        hits = await content.search_fts("meeting")
        # Both shas appear
        shas = {h.sha256 for h in hits}
        assert v1[0].sha256 in shas
        assert v2[0].sha256 in shas
        # Both belong to the same source
        assert all(h.source_id == source for h in hits)


# --------------------------------------------------------------------------- #
# Scenario 3 — contradiction surfaces both                                   #
# --------------------------------------------------------------------------- #


class TestContradictionSurfacesBoth:
    """Two chunks with contradicting facts about the same entity must
    both appear in recall so the agent can flag the conflict. A recall
    that silently drops one contradiction breaks the agent's ability to
    reason about uncertainty."""

    @pytest.mark.asyncio
    async def test_both_contradicting_chunks_returned(self, fresh_stack) -> None:
        content, _ = fresh_stack
        await content.insert_many(
            source_id="src-claim-a.md",
            chunks=_mk_chunks("the project deadline is october"),
        )
        await content.insert_many(
            source_id="src-claim-b.md",
            chunks=_mk_chunks("the project deadline is november"),
        )

        hits = await content.search_fts("deadline")
        # Both source_ids appear — neither was silently dropped
        sources = {h.source_id for h in hits}
        assert {"src-claim-a.md", "src-claim-b.md"} <= sources


# --------------------------------------------------------------------------- #
# Scenario 4 — long source isolates leaf                                     #
# --------------------------------------------------------------------------- #


class TestLongSourceIsolatesLeaf:
    """A source with many chunks where only one contains the query —
    FTS5 BM25 must rank that one chunk over earlier-index chunks with
    no match. Catches naive ``ORDER BY chunk_index`` regression."""

    @pytest.mark.asyncio
    async def test_relevant_chunk_outranks_earlier_unrelated_chunks(self, fresh_stack) -> None:
        content, _ = fresh_stack
        # Build a body with 5 separate paragraphs. Only paragraph 3
        # contains "needle". The chunker emits one chunk per paragraph
        # if the target is small enough.
        paragraphs = [
            "first paragraph about generic background material here.",
            "second paragraph also generic background material here.",
            "third paragraph contains the unique needle word target.",
            "fourth paragraph generic generic generic generic generic.",
            "fifth paragraph generic generic generic generic generic.",
        ]
        body = "\n\n".join(paragraphs)
        chunks = chunk_text(body, target_tokens=10, hard_cap_tokens=30)
        await content.insert_many(source_id="src-long.md", chunks=chunks)

        hits = await content.search_fts("needle")
        assert len(hits) >= 1
        # The top hit's body contains "needle" — BM25 didn't return a
        # chunk based on its position alone.
        assert "needle" in hits[0].body.lower()


# --------------------------------------------------------------------------- #
# Scenario 5 — drill-down isolates children                                  #
# --------------------------------------------------------------------------- #


class TestDrillDownIsolatesChildren:
    """A query about a specific sub-topic should not return the parent's
    generic chunks. Tests the discriminative power of the ranker — a
    parent saying 'fruits include many kinds' should NOT outrank a
    child saying 'apricot is a stone fruit' for the query 'apricot'."""

    @pytest.mark.asyncio
    async def test_specific_query_prefers_specific_chunk(self, fresh_stack) -> None:
        content, _ = fresh_stack
        await content.insert_many(
            source_id="src-parent.md",
            chunks=_mk_chunks(
                "fruits include many varieties of stone fruits and berries and citrus"
            ),
        )
        await content.insert_many(
            source_id="src-apricot.md",
            chunks=_mk_chunks("apricot is a stone fruit native to china"),
        )

        hits = await content.search_fts("apricot")
        # The apricot-specific source wins the top spot
        assert hits[0].source_id == "src-apricot.md"


# --------------------------------------------------------------------------- #
# Scenario 6 — scale: 50 sources × ~5 chunks                                 #
# --------------------------------------------------------------------------- #


class TestScaleLatency:
    """At 50 sources × 5 chunks each (~250 chunks total), FTS5 search
    latency must stay under 200ms p95. This is a smoke test for the
    "no ANN index needed" claim of ADR-020 at the current scale —
    catches a hypothetical regression where the index gets thrashed."""

    @pytest.mark.asyncio
    async def test_fts_latency_at_250_chunks_under_200ms(self, fresh_stack) -> None:
        content, _ = fresh_stack
        # Seed 50 synthetic sources, each ~5 chunks
        for i in range(50):
            body_paragraphs = [
                f"source {i} paragraph {p} talks about topicx{i % 7} here." for p in range(5)
            ]
            chunks = chunk_text(
                "\n\n".join(body_paragraphs),
                target_tokens=10,
                hard_cap_tokens=30,
            )
            await content.insert_many(source_id=f"src-{i:02d}.md", chunks=chunks)

        # Sanity: roughly 250 chunks
        assert await content.count() >= 200

        # Measure 10 queries; assert p95 < 200ms
        latencies_ms = []
        for k in range(10):
            t0 = time.perf_counter()
            await content.search_fts(f"topicx{k % 7}")
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        latencies_ms.sort()
        p95 = latencies_ms[max(0, int(0.95 * len(latencies_ms)) - 1)]
        # Generous bound — local CI may be slow under load. The real
        # signal is "did this jump from 1ms to seconds".
        assert p95 < 200.0, f"FTS p95 latency = {p95:.1f}ms (gate 200ms)"


# --------------------------------------------------------------------------- #
# Scenario 7 — empty corpus graceful                                         #
# --------------------------------------------------------------------------- #


class TestEmptyCorpusGraceful:
    """Recall against an empty content_store must return an empty list,
    not crash. Catches an init-order regression where someone forgets
    to handle the zero-chunk case."""

    @pytest.mark.asyncio
    async def test_fts_on_empty_store(self, fresh_stack) -> None:
        content, _ = fresh_stack
        assert await content.search_fts("anything") == []

    @pytest.mark.asyncio
    async def test_substring_on_empty_store(self, fresh_stack) -> None:
        content, _ = fresh_stack
        assert await content.search_substring("anything") == []
