"""
Tests for `wiki_memory.seal_worker` — composes chunks into parent summaries
and persists via WriteSink.
"""

from __future__ import annotations

import hashlib

import pytest

from wiki_agent.write_sink import NullWriteSink
from wiki_memory.chunker import chunk_text
from wiki_memory.seal_worker import VAULT_TREES_PREFIX, SealError, SealWorker
from wiki_memory.summariser import (
    NullSummariser,
    SummaryPart,
    SummaryResult,
)


class _FixedSummariser:
    """Summariser stub that returns a pre-canned body / cited shas."""

    def __init__(
        self,
        *,
        body: str = "FAKE SUMMARY",
        cited: tuple[str, ...] = (),
    ) -> None:
        self._body = body
        self._cited = cited

    async def summarise(self, parts) -> SummaryResult:  # type: ignore[no-untyped-def]
        if not self._cited:
            cited = tuple(p.sha256 for p in parts)
        else:
            cited = self._cited
        return SummaryResult(
            body=self._body,
            sha256=hashlib.sha256(self._body.encode()).hexdigest(),
            parent_token_count=max(1, len(self._body) // 4),
            cited_shas=cited,
        )


@pytest.fixture
def write_sink() -> NullWriteSink:
    return NullWriteSink()


class TestSealSource:
    @pytest.mark.asyncio
    async def test_no_chunks_raises_seal_error(self, content_store, tree_store, write_sink) -> None:
        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
        )
        with pytest.raises(SealError):
            await worker.seal_source(source_id="empty", node_id="n-empty")

    @pytest.mark.asyncio
    async def test_seals_source_node_end_to_end(
        self, content_store, tree_store, write_sink
    ) -> None:
        # Seed the content store with two chunks for one source.
        chunks = chunk_text("para one.\n\npara two.", target_tokens=5, hard_cap_tokens=20)
        await content_store.insert_many(source_id="src-1", chunks=chunks)
        await tree_store.create_source_node(node_id="src-1")
        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
        )
        result = await worker.seal_source(source_id="src-1", node_id="src-1")
        # Summary written to vault under wiki/trees/sources/
        assert result.page_path.startswith(VAULT_TREES_PREFIX)
        assert "sources/" in result.page_path
        # tree_nodes row was marked sealed
        node = await tree_store.get("src-1")
        assert node is not None
        assert node.sealed_at is not None
        assert node.summary_sha256 == result.summary_sha256
        # WriteSink saw exactly one write
        assert len(write_sink.calls) == 1
        mode, page = write_sink.calls[0]
        assert mode == "upsert"
        assert page.ref.page_path == result.page_path
        # Citations carried into frontmatter
        assert "cited" in page.frontmatter

    @pytest.mark.asyncio
    async def test_faithfulness_gate_rejects_hallucinated_citations(
        self, content_store, tree_store, write_sink
    ) -> None:
        chunks = chunk_text("real content", target_tokens=10)
        await content_store.insert_many(source_id="src-faith", chunks=chunks)
        await tree_store.create_source_node(node_id="src-faith")
        ghost_sha = "0" * 64
        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            summariser=_FixedSummariser(cited=(ghost_sha,)),
        )
        with pytest.raises(SealError):
            await worker.seal_source(source_id="src-faith", node_id="src-faith")
        # Sink was not called; tree node was not marked sealed
        assert write_sink.calls == []
        node = await tree_store.get("src-faith")
        assert node is not None
        assert node.sealed_at is None


class TestSealTopic:
    @pytest.mark.asyncio
    async def test_topic_seal_persists_summary(self, content_store, tree_store, write_sink) -> None:
        # Pre-create a topic node and three sub-summaries.
        await tree_store.upsert(
            type(await tree_store.get("never"))(  # ick; build via dataclass below
                node_id="t-1",
                kind="topic",
                parent_id=None,
                level=1,
                summary_sha256=None,
                score=0.0,
                sealed_at=None,
                tombstoned=False,
                created_at="2026-05-22T00:00:00Z",
            )
        ) if False else None  # noqa: E501  (suppress IDE warning; we use the safe path below)
        from wiki_memory.tree_nodes import TreeNode

        topic = TreeNode(
            node_id="t-1",
            kind="topic",
            parent_id=None,
            level=1,
            summary_sha256=None,
            score=0.0,
            sealed_at=None,
            tombstoned=False,
            created_at="2026-05-22T00:00:00Z",
        )
        await tree_store.upsert(topic)
        children = [
            SummaryPart(
                sha256=hashlib.sha256(f"c{i}".encode()).hexdigest(), body=f"c{i}", token_count=5
            )
            for i in range(3)
        ]
        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
        )
        result = await worker.seal_topic(topic_node_id="t-1", child_sub_summaries=children)
        assert "topics/" in result.page_path
        node = await tree_store.get("t-1")
        assert node is not None
        assert node.sealed_at is not None
        assert node.summary_sha256 == result.summary_sha256

    @pytest.mark.asyncio
    async def test_topic_seal_empty_children_raises(
        self, content_store, tree_store, write_sink
    ) -> None:
        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
        )
        with pytest.raises(SealError):
            await worker.seal_topic(topic_node_id="t-empty", child_sub_summaries=[])


class TestSealPersistsScore:
    """ADR-027 #157: seal_source persists a live score into
    tree_nodes.score via the `score` param mark_sealed already accepted
    (the row previously kept its 0.0 placeholder forever)."""

    @pytest.mark.asyncio
    async def test_seal_writes_live_score_to_tree_node(
        self, content_store, tree_store, write_sink
    ) -> None:
        chunks = chunk_text("para one.\n\npara two.", target_tokens=5, hard_cap_tokens=20)
        await content_store.insert_many(source_id="src-score", chunks=chunks)
        await tree_store.create_source_node(node_id="src-score")
        assert (await tree_store.get("src-score")).score == 0.0  # placeholder

        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
        )
        await worker.seal_source(source_id="src-score", node_id="src-score")

        node = await tree_store.get("src-score")
        # Fresh chunks => recency ≈ 1.0; NullSummariser cites every chunk
        # so the new summary alone gives in-degree 1 == tree max. The
        # composed score must be live (> recency-only floor) and bounded.
        assert 0.5 < node.score <= 1.0

    @pytest.mark.asyncio
    async def test_reused_source_scores_above_cold_source(
        self, content_store, tree_store, write_sink
    ) -> None:
        hot = chunk_text("hot body alpha.", target_tokens=5, hard_cap_tokens=20)
        cold = chunk_text("cold body beta.", target_tokens=5, hard_cap_tokens=20)
        await content_store.insert_many(source_id="src-hot", chunks=hot)
        await content_store.insert_many(source_id="src-cold", chunks=cold)
        await tree_store.create_source_node(node_id="src-hot")
        await tree_store.create_source_node(node_id="src-cold")
        for _ in range(5):
            await content_store.increment_reuse([c.sha256 for c in hot])

        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
        )
        await worker.seal_source(source_id="src-hot", node_id="src-hot")
        await worker.seal_source(source_id="src-cold", node_id="src-cold")

        hot_score = (await tree_store.get("src-hot")).score
        cold_score = (await tree_store.get("src-cold")).score
        assert hot_score > cold_score


class TestNullSummariserIntegration:
    @pytest.mark.asyncio
    async def test_seal_with_null_summariser_writes_valid_markdown(
        self, content_store, tree_store, write_sink
    ) -> None:
        chunks = chunk_text(
            "alpha alpha alpha.\n\nbeta beta beta.\n\ngamma gamma gamma.",
            target_tokens=5,
            hard_cap_tokens=20,
        )
        await content_store.insert_many(source_id="full", chunks=chunks)
        await tree_store.create_source_node(node_id="full")
        worker = SealWorker(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            summariser=NullSummariser(header="Source roll-up"),
        )
        result = await worker.seal_source(source_id="full", node_id="full")
        # NullSummariser embeds chunk citations
        page = write_sink.calls[0][1]
        assert "[[chunk:" in page.body
        # Frontmatter has the cited list
        assert isinstance(page.frontmatter.get("cited"), list)
        assert len(page.frontmatter["cited"]) == len(chunks)
        # tree_nodes row sealed
        node = await tree_store.get("full")
        assert node is not None
        assert node.summary_sha256 == result.summary_sha256
