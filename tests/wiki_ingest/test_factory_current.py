"""
Tests for the public `wiki_ingest.open_memory_store` factory.

This is the M2 Sprint 2 wire-in: external consumers (Memory Tree workers
per PRD-004, ad-hoc CLI scripts, future test harnesses) get a
`wiki_core.MemoryStore`-shaped handle without having to know about
`SqliteMemoryStore` or `EventQueue`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wiki_core.protocols import IngestEvent, MemoryStore
from wiki_ingest import open_memory_store


@pytest.mark.asyncio
async def test_factory_returns_initialised_memory_store(tmp_path: Path) -> None:
    db = tmp_path / "factory.db"
    store = await open_memory_store(db)
    try:
        assert isinstance(store, MemoryStore)
        assert await store.queue_depth() == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_factory_accepts_str_path(tmp_path: Path) -> None:
    db = str(tmp_path / "factory_str.db")
    store = await open_memory_store(db)
    try:
        assert await store.queue_depth() == 0
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_factory_handle_supports_full_round_trip(tmp_path: Path) -> None:
    store = await open_memory_store(tmp_path / "factory_rt.db")
    try:
        ev = IngestEvent(
            event_id="019e5130-0000-7000-8000-fffffffffff1",
            source="watcher:articles",
            path_or_uri="/tmp/factory.md",
            sha256="d" * 64,
            received_at=datetime.now(UTC),
            metadata={
                "rel_path": "factory.md",
                "bucket": "articles",
                "event_type": "created",
                "mtime": "2026-05-22T20:00:00Z",
                "size": 256,
                "mime": "text/markdown",
            },
        )
        rid = await store.enqueue(ev)
        claimed = await store.claim_next()
        assert claimed is not None
        assert claimed.event_id == rid
        await store.mark_done(rid, "wiki/sources/factory.md")
        assert await store.sha_seen("d" * 64) is True
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_factory_tilde_expansion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    store = await open_memory_store("~/factory_home.db")
    try:
        assert await store.queue_depth() == 0
        # File materialised under the tilde-expanded HOME
        assert (tmp_path / "factory_home.db").exists()
    finally:
        await store.close()
