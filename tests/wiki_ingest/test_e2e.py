"""
End-to-end tests for `wiki_ingest` — wire the watcher + queue + worker
against a mocked MCP client and drive real files through the pipeline.

Validates ADR-006 §"Acceptance criteria":
  - drop → wiki page within ≤ 15s
  - bulk 50 files → pool ≤ 2, no duplicates
  - daemon kill mid-flight → recover_stuck re-enqueues
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

wiki_ingest = pytest.importorskip(
    "wiki_ingest",
    reason="core not implemented yet",
)
daemon_mod = pytest.importorskip(
    "wiki_ingest.daemon",
    reason="core not implemented yet",
)
queue_mod = pytest.importorskip(
    "wiki_ingest.queue",
    reason="core not implemented yet",
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


async def _boot_daemon(root: Path, db_path: Path, mcp_client: MagicMock, **kwargs: Any):
    d = daemon_mod.IngestDaemon(
        root=root,
        db_path=db_path,
        mcp_client=mcp_client,
        pool_size=2,
        rate_limit_per_min=600,  # relax for tests
        debounce_seconds=0.15,
        **kwargs,
    )
    await d.start()
    return d


# --------------------------------------------------------------------------- #
# 1. Drop → wiki                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_drop_file_produces_wiki_page_under_15s(
    tmp_wiki_root: Path,
    ingest_db_path: Path,
    mock_mcp_client: MagicMock,
) -> None:
    daemon = await _boot_daemon(tmp_wiki_root, ingest_db_path, mock_mcp_client)
    try:
        drop = tmp_wiki_root / "raw" / "articles" / "hello.md"
        drop.write_bytes(b"# hello\n")

        async def _wait_for_write():
            while mock_mcp_client.write_page.await_count == 0:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(_wait_for_write(), timeout=15.0)
        assert mock_mcp_client.write_page.await_count == 1

        # Archive side-effect: file moved to _ingested/articles
        async def _wait_for_move():
            moved = tmp_wiki_root / "raw" / "_ingested" / "articles" / "hello.md"
            while not moved.exists():
                await asyncio.sleep(0.05)

        await asyncio.wait_for(_wait_for_move(), timeout=5.0)
    finally:
        await daemon.stop()


# --------------------------------------------------------------------------- #
# 2. Bulk drain with bounded concurrency                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_bulk_upload_bounds_pool_and_produces_no_duplicates(
    tmp_wiki_root: Path,
    ingest_db_path: Path,
    mock_mcp_client: MagicMock,
) -> None:
    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def _tracked_write_page(*_a, **_kw):
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return {"ok": True, "page_path": "wiki/sources/bulk.md"}

    mock_mcp_client.write_page.side_effect = _tracked_write_page

    daemon = await _boot_daemon(tmp_wiki_root, ingest_db_path, mock_mcp_client)
    try:
        for i in range(50):
            (tmp_wiki_root / "raw" / "articles" / f"bulk-{i:02d}.md").write_bytes(
                f"# bulk {i}\n".encode()
            )

        async def _wait_all():
            while mock_mcp_client.write_page.await_count < 50:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(_wait_all(), timeout=30.0)
        assert mock_mcp_client.write_page.await_count == 50
        assert peak <= 2, f"worker pool exceeded 2 (peak={peak})"

        # No duplicate sha entries
        q = queue_mod.IngestQueue(ingest_db_path)
        assert q.unique_sha_count() == 50
    finally:
        await daemon.stop()


# --------------------------------------------------------------------------- #
# 3. Chaos: kill daemon mid-flight, restart, recover                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_chaos_kill_mid_processing_recovers_on_restart(
    tmp_wiki_root: Path,
    ingest_db_path: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    # Pre-seed queue with 3 events stuck in 'processing' (simulates kill -9)
    q = queue_mod.IngestQueue(ingest_db_path)
    for _ in range(3):
        q.enqueue(event_factory())
        q.claim_next()  # flips to processing
        # mimic crash: no mark_done / mark_failed

    # Restart daemon — should call recover_stuck and drain
    daemon = await _boot_daemon(tmp_wiki_root, ingest_db_path, mock_mcp_client)
    try:

        async def _wait_all():
            while mock_mcp_client.write_page.await_count < 3:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(_wait_all(), timeout=15.0)
        assert mock_mcp_client.write_page.await_count == 3
        assert q.queue_depth() == 0
    finally:
        await daemon.stop()
