"""
Tests for `wiki_ingest.worker` — asyncio worker pool that drains the SQLite
queue and calls the `wiki-knowledge-engine` MCP.

Contract (from ADR-006 §Architecture / §Backpressure):
    worker = IngestWorker(queue, mcp_client, filters, root,
                          pool_size=2, rate_limit_per_min=10,
                          max_attempts=3, backoff=(1, 4, 16))
    await worker.process_one(event)
    await worker.run(stop_event)
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
worker_mod = pytest.importorskip(
    "wiki_ingest.worker",
    reason="core not implemented yet",
)
queue_mod = pytest.importorskip(
    "wiki_ingest.queue",
    reason="core not implemented yet",
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_worker(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    **overrides: Any,
):
    q = queue_mod.IngestQueue(ingest_db_path)
    kwargs = {
        "queue": q,
        "mcp_client": mock_mcp_client,
        "root": tmp_wiki_root,
        "pool_size": 2,
        "rate_limit_per_min": 10,
        "max_attempts": 3,
        "backoff": (0, 0, 0),  # zero backoff in tests
    }
    kwargs.update(overrides)
    return worker_mod.IngestWorker(**kwargs), q


# --------------------------------------------------------------------------- #
# Happy path: call ordering                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_happy_path_calls_mcp_tools_in_order(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    w, q = _make_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    ev = event_factory()
    q.enqueue(ev)
    claimed = q.claim_next()
    await w.process_one(claimed)

    # Assert ordering: validate_frontmatter → write_page → update_index_entry → append_to_log
    parent = MagicMock()
    parent.attach_mock(mock_mcp_client.validate_frontmatter, "validate_frontmatter")
    parent.attach_mock(mock_mcp_client.write_page, "write_page")
    parent.attach_mock(mock_mcp_client.update_index_entry, "update_index_entry")
    parent.attach_mock(mock_mcp_client.append_to_log, "append_to_log")
    observed = [c[0] for c in parent.mock_calls]
    assert observed == [
        "validate_frontmatter",
        "write_page",
        "update_index_entry",
        "append_to_log",
    ]
    assert mock_mcp_client.validate_frontmatter.await_count == 1
    assert mock_mcp_client.write_page.await_count == 1
    assert mock_mcp_client.update_index_entry.await_count == 1
    assert mock_mcp_client.append_to_log.await_count == 1


# --------------------------------------------------------------------------- #
# Backpressure: semaphore bounds concurrent MCP calls                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_pool_size_bounds_concurrent_mcp_calls(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    in_flight = 0
    peak = 0
    barrier = asyncio.Event()

    async def _slow_write_page(*_a, **_kw):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await barrier.wait()
        in_flight -= 1
        return {"ok": True, "page_path": "wiki/sources/x.md"}

    mock_mcp_client.write_page.side_effect = _slow_write_page

    w, q = _make_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client, pool_size=2)
    events = [event_factory() for _ in range(5)]
    for e in events:
        q.enqueue(e)

    claims = [q.claim_next() for _ in events]
    tasks = [asyncio.create_task(w.process_one(c)) for c in claims]
    await asyncio.sleep(0.1)  # let workers enter the gate
    assert peak <= 2, f"peak concurrency {peak} exceeded pool_size=2"
    barrier.set()
    await asyncio.gather(*tasks, return_exceptions=True)


# --------------------------------------------------------------------------- #
# Rate-limit: 10 write_page calls per minute                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_rate_limit_throttles_eleventh_call(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = [0.0]

    def fake_monotonic() -> float:
        return t[0]

    sleeps: list[float] = []

    async def fake_sleep(d: float) -> None:
        sleeps.append(d)
        t[0] += d

    monkeypatch.setattr(worker_mod.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(worker_mod.asyncio, "sleep", fake_sleep)

    w, q = _make_worker(
        ingest_db_path,
        tmp_wiki_root,
        mock_mcp_client,
        rate_limit_per_min=10,
        pool_size=1,
    )
    for _ in range(11):
        q.enqueue(event_factory())

    for _ in range(11):
        claimed = q.claim_next()
        await w.process_one(claimed)

    # 11th call must wait at least a non-trivial slice (≥ 1s) before running
    assert any(s >= 1.0 for s in sleeps), f"no throttling observed, sleeps={sleeps}"


# --------------------------------------------------------------------------- #
# Idempotency: sha already ingested → skip                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_duplicate_sha_is_skipped_without_mcp_calls(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    w, q = _make_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    ev = event_factory()
    # Prime `ingested` table directly
    q.mark_ingested(ev["sha256"], rel_path=ev["rel_path"], page_path="wiki/sources/x.md")

    q.enqueue(ev)
    claimed = q.claim_next()
    await w.process_one(claimed)

    assert mock_mcp_client.write_page.await_count == 0
    row = q.get(claimed["event_id"])
    assert row["status"] == "skipped"


# --------------------------------------------------------------------------- #
# Retry + DLQ                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_retry_succeeds_on_third_attempt(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    mock_mcp_client.write_page.side_effect = [
        RuntimeError("transient-1"),
        RuntimeError("transient-2"),
        {"ok": True, "page_path": "wiki/sources/x.md"},
    ]
    w, q = _make_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    ev = event_factory()
    q.enqueue(ev)
    claimed = q.claim_next()
    await w.process_one(claimed)

    row = q.get(claimed["event_id"])
    assert row["status"] == "done"
    assert row["attempts"] == 3


@pytest.mark.asyncio
async def test_retry_exhausted_marks_failed(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    mock_mcp_client.write_page.side_effect = RuntimeError("always-fails")
    w, q = _make_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    ev = event_factory()
    q.enqueue(ev)
    claimed = q.claim_next()
    await w.process_one(claimed)

    row = q.get(claimed["event_id"])
    assert row["status"] == "failed"
    assert row["last_error"] is not None
    assert "always-fails" in row["last_error"]


# --------------------------------------------------------------------------- #
# Post-success side effects                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_successful_ingest_moves_file_to_ingested_bucket(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    w, q = _make_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    src = tmp_wiki_root / "raw" / "articles" / "to-move.md"
    src.write_bytes(b"# move me\n")
    ev = event_factory(path=src, bucket="articles")
    q.enqueue(ev)
    claimed = q.claim_next()
    await w.process_one(claimed)

    assert not src.exists(), "source must be moved out of inbox"
    moved = tmp_wiki_root / "raw" / "_ingested" / "articles" / "to-move.md"
    assert moved.exists(), "file must land in _ingested/articles/"


@pytest.mark.asyncio
async def test_disallowed_mime_is_skipped_with_reason(
    ingest_db_path: Path,
    tmp_wiki_root: Path,
    mock_mcp_client: MagicMock,
    event_factory: Callable[..., dict[str, Any]],
) -> None:
    w, q = _make_worker(ingest_db_path, tmp_wiki_root, mock_mcp_client)
    ev = event_factory()
    ev["mime"] = "image/png"
    q.enqueue(ev)
    claimed = q.claim_next()
    await w.process_one(claimed)

    assert mock_mcp_client.write_page.await_count == 0
    row = q.get(claimed["event_id"])
    assert row["status"] == "skipped"
    assert (row["last_error"] or "").startswith("mime-filtered") or row.get(
        "skip_reason"
    ) == "mime-filtered"
