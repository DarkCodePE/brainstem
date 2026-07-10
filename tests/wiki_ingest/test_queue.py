"""
Tests for `wiki_ingest.queue` — SQLite WAL durable queue per ADR-006.

Contract (inferred from ADR-006 §Architecture and §Idempotency):
    queue = IngestQueue(db_path)
    queue.enqueue(event_dict) -> event_id
    queue.claim_next() -> event_dict | None   # sets status='processing'
    queue.mark_done(event_id, page_path)      # inserts into `ingested`
    queue.mark_failed(event_id, err, retryable=True)
    queue.sha_seen(sha256) -> bool
    queue.recover_stuck() -> int              # processing → pending
    queue.queue_depth() -> int                # status='pending' count
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

wiki_ingest = pytest.importorskip(
    "wiki_ingest",
    reason="core not implemented yet",
)
queue_mod = pytest.importorskip(
    "wiki_ingest.queue",
    reason="core not implemented yet",
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_queue(db_path: Path):
    return queue_mod.IngestQueue(db_path)


# --------------------------------------------------------------------------- #
# Core enqueue / claim / done / fail                                          #
# --------------------------------------------------------------------------- #


def test_enqueue_then_claim_returns_processing(
    ingest_db_path: Path, event_factory: Callable[..., dict[str, Any]]
) -> None:
    q = _make_queue(ingest_db_path)
    ev = event_factory()
    eid = q.enqueue(ev)
    claimed = q.claim_next()
    assert claimed is not None
    assert claimed["event_id"] == eid
    assert claimed["status"] == "processing"


def test_claim_next_on_empty_returns_none(ingest_db_path: Path) -> None:
    q = _make_queue(ingest_db_path)
    assert q.claim_next() is None


def test_mark_done_sets_status_and_inserts_ingested(
    ingest_db_path: Path, event_factory: Callable[..., dict[str, Any]]
) -> None:
    q = _make_queue(ingest_db_path)
    ev = event_factory()
    eid = q.enqueue(ev)
    q.claim_next()
    q.mark_done(eid, page_path="wiki/sources/foo.md")

    # status flips to 'done'
    row = q.get(eid)
    assert row["status"] == "done"
    # sha recorded in `ingested`
    assert q.sha_seen(ev["sha256"]) is True


def test_mark_failed_increments_attempts(
    ingest_db_path: Path, event_factory: Callable[..., dict[str, Any]]
) -> None:
    q = _make_queue(ingest_db_path)
    ev = event_factory()
    eid = q.enqueue(ev)
    q.claim_next()
    q.mark_failed(eid, err="boom", retryable=True)
    row = q.get(eid)
    assert row["attempts"] == 1
    assert row["last_error"] == "boom"


# --------------------------------------------------------------------------- #
# Idempotency                                                                 #
# --------------------------------------------------------------------------- #


def test_sha_seen_returns_true_for_existing(
    ingest_db_path: Path, event_factory: Callable[..., dict[str, Any]]
) -> None:
    q = _make_queue(ingest_db_path)
    ev = event_factory()
    eid = q.enqueue(ev)
    q.claim_next()
    q.mark_done(eid, page_path="wiki/sources/foo.md")
    assert q.sha_seen(ev["sha256"]) is True


def test_sha_seen_returns_false_for_unknown(ingest_db_path: Path) -> None:
    q = _make_queue(ingest_db_path)
    assert q.sha_seen("0" * 64) is False


def test_duplicate_enqueue_produces_distinct_events_but_same_sha(
    ingest_db_path: Path, event_factory: Callable[..., dict[str, Any]]
) -> None:
    q = _make_queue(ingest_db_path)
    ev1 = event_factory()
    ev2 = event_factory(path=ev1["path"])  # same file, same content → same sha
    eid1 = q.enqueue(ev1)
    eid2 = q.enqueue(ev2)
    assert eid1 != eid2
    assert ev1["sha256"] == ev2["sha256"]
    # worker-side guarantee: sha_seen should drive a skip, not the queue itself


# --------------------------------------------------------------------------- #
# Recovery & depth                                                            #
# --------------------------------------------------------------------------- #


def test_recover_stuck_moves_processing_back_to_pending(
    ingest_db_path: Path, event_factory: Callable[..., dict[str, Any]]
) -> None:
    q = _make_queue(ingest_db_path)
    q.enqueue(event_factory())
    q.enqueue(event_factory())
    q.claim_next()
    q.claim_next()
    recovered = q.recover_stuck()
    assert recovered == 2
    assert q.queue_depth() == 2


def test_queue_depth_counts_only_pending(
    ingest_db_path: Path, event_factory: Callable[..., dict[str, Any]]
) -> None:
    q = _make_queue(ingest_db_path)
    eid1 = q.enqueue(event_factory())
    q.enqueue(event_factory())
    q.enqueue(event_factory())
    q.claim_next()
    q.mark_done(eid1, page_path="wiki/sources/x.md")
    # 1 done, 1 processing, 1 pending
    assert q.queue_depth() == 1


# --------------------------------------------------------------------------- #
# Concurrency                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_parallel_enqueues_lose_no_events(
    ingest_db_path: Path, event_factory: Callable[..., dict[str, Any]]
) -> None:
    q = _make_queue(ingest_db_path)
    events = [event_factory() for _ in range(50)]

    async def _enq(e):
        await asyncio.to_thread(q.enqueue, e)

    await asyncio.gather(*[_enq(e) for e in events])
    assert q.queue_depth() == 50
