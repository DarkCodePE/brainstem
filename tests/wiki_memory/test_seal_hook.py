"""
Tests for `wiki_memory.seal_hook` — post-ingest seal callback.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from wiki_agent.write_sink import NullWriteSink
from wiki_core.protocols import IngestEvent
from wiki_memory.seal_hook import (
    NoopSealHook,
    OnPageWrittenCallback,
    SealOnIngestHook,
    build_default_seal_hook,
)
from wiki_memory.summariser import NullSummariser


def _make_event(
    *,
    event_id: str = "ev-001",
    source: str = "watcher:articles",
    sha256: str = "a" * 64,
    path_or_uri: str = "/tmp/x.md",
) -> IngestEvent:
    return IngestEvent(
        event_id=event_id,
        source=source,
        path_or_uri=path_or_uri,
        sha256=sha256,
        received_at=datetime.now(UTC),
        metadata={
            "rel_path": "x.md",
            "bucket": "articles",
            "event_type": "created",
            "mtime": "2026-05-24T00:00:00Z",
            "size": 1024,
            "mime": "text/markdown",
        },
    )


def _sync_schedule(coro):
    """Schedule = run synchronously. Lets tests assert on seal outcomes
    without waiting for an event-loop tick."""
    return (
        asyncio.get_event_loop().run_until_complete(coro) if not asyncio.iscoroutine(coro) else coro
    )


class _CapturingScheduler:
    """Stores spawned coroutines so tests can await them in-order.

    Closes any un-awaited coroutines on destruction to silence
    "coroutine was never awaited" warnings when a test only checks
    that scheduling happened (not the seal outcome).
    """

    def __init__(self) -> None:
        self.spawned: list = []

    def __call__(self, coro):
        self.spawned.append(coro)
        return coro

    async def drain(self) -> None:
        while self.spawned:
            coro = self.spawned.pop(0)
            await coro

    def __del__(self) -> None:
        for coro in self.spawned:
            try:
                coro.close()
            except Exception:  # noqa: BLE001
                pass
        self.spawned.clear()


@pytest.fixture
def write_sink() -> NullWriteSink:
    return NullWriteSink()


class TestProtocolConformance:
    def test_seal_on_ingest_hook_satisfies_protocol(
        self, content_store, tree_store, write_sink, tmp_path
    ) -> None:
        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=tmp_path,
        )
        assert isinstance(hook, OnPageWrittenCallback)

    def test_noop_hook_satisfies_protocol(self) -> None:
        assert isinstance(NoopSealHook(), OnPageWrittenCallback)


class TestPageIngestPath:
    @pytest.mark.asyncio
    async def test_indexes_chunks_and_creates_tree_node(
        self, content_store, tree_store, write_sink, tmp_path
    ) -> None:
        # Seed a real file in the vault root.
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        page_path = "wiki/sources/sample.md"
        full = vault_root / page_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("para one.\n\npara two.\n", encoding="utf-8")

        scheduler = _CapturingScheduler()
        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=vault_root,
            summariser_factory=NullSummariser,
            schedule=scheduler,
        )
        ev = _make_event()
        await hook(ev, page_path)

        # Chunks landed in content_store
        assert await content_store.count() > 0
        # Source tree node created
        sources = await tree_store.list_by_kind("source")
        assert len(sources) == 1
        # Scheduled exactly one seal task (still pending)
        assert len(scheduler.spawned) == 1

    @pytest.mark.asyncio
    async def test_ai_first_preamble_isolated_as_own_chunk(
        self, content_store, tree_store, write_sink, tmp_path
    ) -> None:
        """ADR-036 D4: the `## For future Claude` preamble lands as its own
        chunk (a clean embedding target) instead of being packed into chunk 0."""
        from wiki_memory.seal_hook import _source_id_from_event

        vault_root = tmp_path / "vault"
        page_path = "wiki/sources/sample.md"
        full = vault_root / page_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(
            '---\ntitle: "T"\ndate: 2026-06-14\nsources: ["raw/x.md"]\n'
            'tags: ["ingested"]\norigin: llm-synthesized\ncategory: sources\n'
            "source_count: 1\n---\n\n# T\n\n## For future Claude\n\n"
            "Relevance note for a future reader.\n\n"
            "The body summary with [[Some Entity]].\n",
            encoding="utf-8",
        )
        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=vault_root,
            summariser_factory=NullSummariser,
            schedule=_CapturingScheduler(),
        )
        ev = _make_event()
        await hook(ev, page_path)

        bodies = [c.body for c in await content_store.list_by_source(_source_id_from_event(ev))]
        assert "## For future Claude\n\nRelevance note for a future reader." in bodies
        # the body summary is in a different chunk (not diluting the preamble)
        assert all("body summary" not in b for b in bodies if b.startswith("## For future Claude"))

    @pytest.mark.asyncio
    async def test_missing_file_logs_and_returns(
        self, content_store, tree_store, write_sink, tmp_path, caplog
    ) -> None:
        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=tmp_path,
        )
        with caplog.at_level(logging.WARNING, logger="wiki_memory.seal_hook"):
            await hook(_make_event(), "wiki/sources/ghost.md")
        assert "page_missing" in caplog.text
        assert await content_store.count() == 0

    @pytest.mark.asyncio
    async def test_empty_body_skips_seal(
        self, content_store, tree_store, write_sink, tmp_path
    ) -> None:
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        page_path = "wiki/sources/empty.md"
        full = vault_root / page_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("   \n\n  \n", encoding="utf-8")  # whitespace-only

        scheduler = _CapturingScheduler()
        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=vault_root,
            schedule=scheduler,
        )
        await hook(_make_event(), page_path)
        # No chunks, no tree node, no seal scheduled
        assert await content_store.count() == 0
        assert await tree_store.count() == 0
        assert scheduler.spawned == []

    @pytest.mark.asyncio
    async def test_path_traversal_refused(
        self, content_store, tree_store, write_sink, tmp_path, caplog
    ) -> None:
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        secret = tmp_path / "secret.md"
        secret.write_text("secret content", encoding="utf-8")

        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=vault_root,
        )
        with caplog.at_level(logging.WARNING, logger="wiki_memory.seal_hook"):
            await hook(_make_event(), "../secret.md")
        assert "read_failed" in caplog.text
        assert await content_store.count() == 0


class TestSealScheduling:
    @pytest.mark.asyncio
    async def test_seal_runs_when_scheduler_drains(
        self, content_store, tree_store, write_sink, tmp_path
    ) -> None:
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        page_path = "wiki/sources/x.md"
        full = vault_root / page_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("para one.\n\npara two.\n", encoding="utf-8")

        scheduler = _CapturingScheduler()
        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=vault_root,
            summariser_factory=NullSummariser,
            schedule=scheduler,
        )
        await hook(_make_event(), page_path)
        # Before drain: seal pending, write_sink not yet called.
        assert write_sink.calls == []
        await scheduler.drain()
        # After drain: vault mirror was written and the tree node sealed.
        assert len(write_sink.calls) == 1
        sources = await tree_store.list_by_kind("source")
        assert len(sources) == 1
        assert sources[0].summary_sha256 is not None
        assert sources[0].sealed_at is not None

    @pytest.mark.asyncio
    async def test_disable_seal_skips_summariser_call(
        self, content_store, tree_store, write_sink, tmp_path
    ) -> None:
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        page_path = "wiki/sources/x.md"
        full = vault_root / page_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("para one.\n\npara two.\n", encoding="utf-8")

        scheduler = _CapturingScheduler()
        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=vault_root,
            schedule=scheduler,
            enable_seal=False,
        )
        await hook(_make_event(), page_path)
        # Chunks + node still get indexed
        assert await content_store.count() > 0
        assert await tree_store.count() == 1
        # But no seal task scheduled
        assert scheduler.spawned == []
        # And the vault mirror was NOT written (no summary)
        assert write_sink.calls == []


class TestDifferentSources:
    @pytest.mark.asyncio
    async def test_same_content_different_source_creates_distinct_nodes(
        self, content_store, tree_store, write_sink, tmp_path
    ) -> None:
        """Same content sha but two different `source`s → two source ids
        (and therefore two distinct tree_source nodes). The dedup happens
        in content_store on sha collision, so chunks are inserted once."""
        vault_root = tmp_path / "vault"
        vault_root.mkdir()
        path1 = "wiki/sources/from-gmail.md"
        path2 = "wiki/sources/from-drive.md"
        for p in (path1, path2):
            full = vault_root / p
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text("identical body content here.\n", encoding="utf-8")

        scheduler = _CapturingScheduler()
        hook = SealOnIngestHook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=vault_root,
            summariser_factory=NullSummariser,
            schedule=scheduler,
            enable_seal=False,
        )
        await hook(_make_event(source="gmail", sha256="b" * 64), path1)
        await hook(_make_event(source="drive", sha256="b" * 64), path2)

        sources = await tree_store.list_by_kind("source")
        assert len(sources) == 2
        # Both source_ids are distinct because we hash `source:sha256`
        assert sources[0].node_id != sources[1].node_id


class TestNoop:
    @pytest.mark.asyncio
    async def test_noop_is_a_silent_no_op(self) -> None:
        await NoopSealHook()(_make_event(), "wiki/x.md")  # should not raise


class TestFactory:
    def test_build_default_seal_hook_returns_real_hook(
        self, content_store, tree_store, write_sink, tmp_path
    ) -> None:
        hook = build_default_seal_hook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=tmp_path,
        )
        assert isinstance(hook, SealOnIngestHook)
        assert isinstance(hook, OnPageWrittenCallback)

    def test_build_default_seal_hook_respects_enable_seal_false(
        self, content_store, tree_store, write_sink, tmp_path
    ) -> None:
        hook = build_default_seal_hook(
            content_store=content_store,
            tree_store=tree_store,
            write_sink=write_sink,
            vault_root=tmp_path,
            enable_seal=False,
        )
        assert hook._enable_seal is False
