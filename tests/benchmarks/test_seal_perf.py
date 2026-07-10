"""
Seal worker latency benchmarks (#132).

Times one ``seal_source`` invocation using NullSummariser so the
benchmark stays deterministic and offline (no LLM round-trip). The
gate-relevant scenario is "small source seal" — 5 chunks composed
into one parent summary. Bigger sources (50+ chunks) are bounded by
LLM time, not by the seal worker's plumbing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from wiki_agent.write_sink import NullWriteSink
from wiki_memory.chunker import chunk_text
from wiki_memory.content_store import ContentStore
from wiki_memory.seal_worker import SealWorker
from wiki_memory.summariser import NullSummariser
from wiki_memory.tree_nodes import TreeNodeStore


def _run_seal(tmp_dir: Path) -> str:
    """One full seal cycle: seed 5 chunks, build worker, seal, return
    the summary sha. tmp_dir lets pytest-benchmark reuse the same
    directory across iterations (each iteration is a fresh DB)."""

    async def _inner() -> str:
        content_path = tmp_dir / f"bench-{id(tmp_dir)}.db"
        tree_path = tmp_dir / f"bench-tree-{id(tmp_dir)}.db"
        content = ContentStore(content_path)
        tree = TreeNodeStore(tree_path)
        await content.init()
        await tree.init()
        try:
            chunks = chunk_text(
                "first para about widgets.\n\nsecond para about gadgets.\n\nthird para about thingamajigs.\n\nfourth para about doohickeys.\n\nfifth para about whatsits.",
                target_tokens=10,
                hard_cap_tokens=30,
            )
            await content.insert_many(source_id="bench-src", chunks=chunks)
            await tree.create_source_node(node_id="bench-src")

            worker = SealWorker(
                content_store=content,
                tree_store=tree,
                write_sink=NullWriteSink(),
                summariser=NullSummariser(),
            )
            result = await worker.seal_source(source_id="bench-src", node_id="bench-src")
            return result.summary_sha256
        finally:
            await content.close()
            await tree.close()
            content_path.unlink(missing_ok=True)
            tree_path.unlink(missing_ok=True)

    return asyncio.run(_inner())


class TestSealLatency:
    def test_bench_seal_5_chunks(self, benchmark, tmp_path: Path) -> None:
        """One seal cycle with NullSummariser. Times the chunker JOIN +
        faithfulness gate + write_sink call + tree_node mark_sealed.
        Target: <50ms p95."""
        result = benchmark(_run_seal, tmp_path)
        assert isinstance(result, str)
        assert len(result) == 64  # sha256 hex length
