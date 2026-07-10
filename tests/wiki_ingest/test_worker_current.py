"""Regression tests for the ADR-035 D1 `page_path=NULL` fix.

The original incident (2026-05-30): the worker called `write_page` with
an argument shape the MCP server never accepted, FastMCP reported the
Pydantic failure as `isError: true` inside a *successful* JSON-RPC
response, `_call_write_page` shrugged and returned None, and `_process`
marked the event done with `page_path=NULL` — then moved the raw file
out of raw/ ("ate" it). These tests pin every link of that chain.

Written against the CURRENT async API (unlike the quarantined
`test_worker.py` / `test_e2e.py` listed in conftest.collect_ignore,
which target a never-shipped synchronous API — do not revive those).
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest

import wiki_ingest.worker as worker_mod
from wiki_ingest.config import Config
from wiki_ingest.models import IngestEvent
from wiki_ingest.pagewrite import extract_page_path
from wiki_ingest.queue import EventQueue
from wiki_ingest.worker import WorkerPool, WritePageError

# Verbatim shape of the original production failure: FastMCP wraps the
# Pydantic validation error in a SUCCESSFUL JSON-RPC response.
VALIDATION_ERROR_RESULT = {
    "content": [
        {
            "type": "text",
            "text": (
                "Error executing tool write_page: 2 validation errors for "
                "write_pageArguments\npage_path\n  Field required "
                "[type=missing, input_value={'source_path': ...}, "
                "input_type=dict]\ncontent\n  Field required "
                "[type=missing, input_value={'source_path': ...}, "
                "input_type=dict]"
            ),
        }
    ],
    "isError": True,
}


def make_config(kb_root: Path, db_path: Path, mcp_command=("/bin/true",)) -> Config:
    return Config(
        wiki_root=kb_root,
        raw_dir=kb_root / "raw",
        ingested_dir=kb_root / "raw" / "_ingested",
        db_path=db_path,
        mcp_command=tuple(mcp_command),
        metrics_path=None,
    )


def make_event(raw_file: Path, kb_root: Path) -> IngestEvent:
    stat = raw_file.stat()
    return IngestEvent(
        path=str(raw_file),
        rel_path=str(raw_file.relative_to(kb_root)),
        bucket=raw_file.parent.name,
        event_type="created",
        mtime=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(timespec="seconds"),
        size=stat.st_size,
        mime="text/markdown",
    )


class StubMcp:
    """In-process stand-in for McpStdioClient."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return self._handler(name, arguments)

    async def close(self) -> None:
        return None


async def drain(queue: EventQueue, pool: WorkerPool) -> None:
    """Claim + dispatch until the queue has no claimable events left
    (retries re-enter as pending, so loop until exhaustion)."""
    while True:
        event = await queue.claim_next()
        if event is None:
            return
        await pool.dispatch(event)


@pytest.fixture
def fast_retries(monkeypatch):
    monkeypatch.setattr(worker_mod, "_RETRY_DELAYS", (0.0, 0.0, 0.0))


@pytest.fixture
def kb(tmp_wiki_root: Path) -> Path:
    return tmp_wiki_root


@pytest.fixture
def raw_file(kb: Path) -> Path:
    f = kb / "raw" / "articles" / "My Article (1).md"
    f.write_text(
        "---\ntitle: original\n---\n\nBody text with ![alt](http://img.example/x.png)\n",
        encoding="utf-8",
    )
    return f


# --------------------------------------------------------------------------- #
# 1. Original-failure reproduction                                            #
# --------------------------------------------------------------------------- #


class TestOriginalFailureRepro:
    @pytest.mark.asyncio
    async def test_iserror_response_ends_failed_not_done(
        self, kb, raw_file, ingest_db_path, fast_retries
    ) -> None:
        cfg = make_config(kb, ingest_db_path)
        queue = EventQueue(ingest_db_path)
        await queue.init()
        pool = WorkerPool(cfg, queue)
        pool._mcp = StubMcp(lambda name, args: dict(VALIDATION_ERROR_RESULT))

        event = make_event(raw_file, kb)
        await queue.enqueue(event)
        await drain(queue, pool)

        counts = await queue.counts_by_status()
        assert counts.get("failed") == 1, f"expected failed, got {counts}"
        assert counts.get("done", 0) == 0

        # No `ingested` row was recorded for the file's content.
        async with aiosqlite.connect(ingest_db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM ingested") as cur:
                row = await cur.fetchone()
        assert row[0] == 0

        # The raw file was NOT eaten: still in raw/, not in _ingested/.
        assert raw_file.exists()
        assert not list((kb / "raw" / "_ingested").rglob("*.md"))
        await queue.close()

    @pytest.mark.asyncio
    async def test_call_write_page_raises_on_iserror(self, kb, raw_file, ingest_db_path) -> None:
        cfg = make_config(kb, ingest_db_path)
        queue = EventQueue(ingest_db_path)
        await queue.init()
        pool = WorkerPool(cfg, queue)
        pool._mcp = StubMcp(lambda name, args: dict(VALIDATION_ERROR_RESULT))

        event = make_event(raw_file, kb)
        with pytest.raises(WritePageError, match="validation errors"):
            await pool._call_write_page(event, "deadbeef", "text/markdown", 10, None)
        await queue.close()


# --------------------------------------------------------------------------- #
# 2. Done invariant: done ⇒ non-null page_path AND page on disk               #
# --------------------------------------------------------------------------- #


class TestDoneInvariant:
    @pytest.mark.asyncio
    async def test_done_event_has_page_path_and_page_on_disk(
        self, kb, raw_file, ingest_db_path
    ) -> None:
        cfg = make_config(kb, ingest_db_path)
        queue = EventQueue(ingest_db_path)
        await queue.init()
        pool = WorkerPool(cfg, queue)

        def fake_write_page(name: str, args: dict) -> dict:
            assert name == "write_page"
            # The worker must speak the REAL contract: page_path + content.
            assert set(args) <= {"page_path", "content", "overwrite"}
            assert args["page_path"].startswith("wiki/sources/")
            assert args["page_path"].endswith(".md")
            full = kb / args["page_path"]
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(args["content"], encoding="utf-8")
            payload = json.dumps(
                {"status": "created", "page_path": args["page_path"], "size_bytes": 1}
            )
            return {
                "content": [{"type": "text", "text": payload}],
                "structuredContent": {"result": payload},
                "isError": False,
            }

        pool._mcp = StubMcp(fake_write_page)

        event = make_event(raw_file, kb)
        await queue.enqueue(event)
        await drain(queue, pool)

        counts = await queue.counts_by_status()
        assert counts.get("done") == 1, f"expected done, got {counts}"

        page_path = await queue.get_page_path(event.event_id)
        assert page_path, "done event must carry a non-null page_path"
        page = kb / page_path
        assert page.exists(), f"page {page_path} must exist on disk"

        body = page.read_text(encoding="utf-8")
        assert "origin: ingested-untrusted" in body
        assert "ingested_sha256:" in body
        assert "![alt](http://img.example/x.png)" in body  # body preserved

        # Slug is deterministic + filesystem-safe (spaces/parens collapse).
        assert page_path == "wiki/sources/my-article-1.md"

        # Raw file moved out of raw/ only after real success.
        assert not raw_file.exists()
        assert list((kb / "raw" / "_ingested" / "articles").iterdir())
        await queue.close()

    @pytest.mark.asyncio
    async def test_unparseable_success_response_is_not_done(
        self, kb, raw_file, ingest_db_path, fast_retries
    ) -> None:
        """A 'successful' tool result with no page_path must not be
        treated as success (the old code returned None here)."""
        cfg = make_config(kb, ingest_db_path)
        queue = EventQueue(ingest_db_path)
        await queue.init()
        pool = WorkerPool(cfg, queue)
        pool._mcp = StubMcp(
            lambda name, args: {
                "content": [{"type": "text", "text": "ok!"}],
                "isError": False,
            }
        )

        event = make_event(raw_file, kb)
        await queue.enqueue(event)
        await drain(queue, pool)

        counts = await queue.counts_by_status()
        assert counts.get("done", 0) == 0
        assert counts.get("failed") == 1
        assert raw_file.exists()
        await queue.close()

    def test_extract_page_path_refused_uses_existing_page(self) -> None:
        payload = json.dumps(
            {
                "status": "refused",
                "reason": "duplicate_source",
                "page_path": "wiki/sources/new-slug.md",
                "existing_page": "wiki/sources/old-slug.md",
            }
        )
        result = {"content": [{"type": "text", "text": payload}], "isError": False}
        assert extract_page_path(result) == "wiki/sources/old-slug.md"

    def test_extract_page_path_error_payload_returns_none(self) -> None:
        result = {"structuredContent": {"error": "disk full"}, "isError": False}
        assert extract_page_path(result) is None


# --------------------------------------------------------------------------- #
# 3. Contract/integration: the REAL MCP server over stdio                     #
# --------------------------------------------------------------------------- #


class TestRealMcpServerContract:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_against_real_server(
        self, kb, raw_file, ingest_db_path, monkeypatch
    ) -> None:
        """Kills mock drift: spawn `python -m wiki_agent.mcp_server` over
        stdio with WIKI_ROOT=tmp and ingest a real raw file through the
        WorkerPool. Asserts page on disk + event done with that path."""
        (kb / "wiki").mkdir(exist_ok=True)
        monkeypatch.setenv("WIKI_ROOT", str(kb))
        cfg = make_config(
            kb, ingest_db_path, mcp_command=(sys.executable, "-m", "wiki_agent.mcp_server")
        )
        queue = EventQueue(ingest_db_path)
        await queue.init()
        pool = WorkerPool(cfg, queue)

        event = make_event(raw_file, kb)
        await queue.enqueue(event)
        try:
            # Timeout guard: server import is heavy (langchain); a hang
            # here must fail the test, not the suite.
            await asyncio.wait_for(drain(queue, pool), timeout=120)
        finally:
            await pool.close()

        counts = await queue.counts_by_status()
        assert counts.get("done") == 1, f"expected done, got {counts}"
        page_path = await queue.get_page_path(event.event_id)
        assert page_path
        assert (kb / page_path).exists()
        assert not raw_file.exists()  # moved to _ingested only on real success
        await queue.close()


# --------------------------------------------------------------------------- #
# 4. Queue guards + migration                                                 #
# --------------------------------------------------------------------------- #


class TestQueueGuards:
    @pytest.mark.asyncio
    async def test_mark_done_rejects_null_page_path(self, ingest_db_path) -> None:
        queue = EventQueue(ingest_db_path)
        await queue.init()
        with pytest.raises(ValueError, match="page_path"):
            await queue.mark_done("some-event", None)
        with pytest.raises(ValueError, match="page_path"):
            await queue.mark_done("some-event", "")
        await queue.close()

    @pytest.mark.asyncio
    async def test_record_ingested_rejects_null_page_path(self, ingest_db_path) -> None:
        queue = EventQueue(ingest_db_path)
        await queue.init()
        with pytest.raises(ValueError, match="page_path"):
            await queue.record_ingested("sha", "raw/x.md", None)
        await queue.close()

    @pytest.mark.asyncio
    async def test_mark_done_persists_and_round_trips(self, ingest_db_path) -> None:
        queue = EventQueue(ingest_db_path)
        await queue.init()
        event = IngestEvent(
            path="/tmp/kb/raw/articles/x.md",
            rel_path="raw/articles/x.md",
            bucket="articles",
            event_type="created",
            mtime="2026-06-10T00:00:00Z",
            size=1,
        )
        await queue.enqueue(event)
        await queue.mark_done(event.event_id, "wiki/x.md")
        assert await queue.get_page_path(event.event_id) == "wiki/x.md"
        counts = await queue.counts_by_status()
        assert counts.get("done") == 1
        await queue.close()

    @pytest.mark.asyncio
    async def test_migration_adds_column_to_legacy_db(self, tmp_path: Path) -> None:
        """A pre-ADR-035 DB (15-column events table) gains page_path on
        init() without losing existing rows."""
        db_path = tmp_path / "legacy.db"
        legacy_schema = """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY, path TEXT NOT NULL, rel_path TEXT NOT NULL,
            bucket TEXT NOT NULL, event_type TEXT NOT NULL, mtime TEXT NOT NULL,
            size INTEGER NOT NULL, sha256 TEXT, mime TEXT, enqueued_at TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
            last_error TEXT, started_at TEXT, finished_at TEXT
        );
        CREATE TABLE ingested (
            sha256 TEXT PRIMARY KEY, rel_path TEXT NOT NULL,
            page_path TEXT, ingested_at TEXT NOT NULL
        );
        CREATE TABLE rate_limit (
            key TEXT PRIMARY KEY, window_start TEXT NOT NULL,
            tokens_used INTEGER NOT NULL DEFAULT 0
        );
        """
        async with aiosqlite.connect(db_path) as db:
            await db.executescript(legacy_schema)
            await db.execute(
                "INSERT INTO events (event_id, path, rel_path, bucket, event_type, "
                "mtime, size, enqueued_at, status) "
                "VALUES ('legacy-1', '/p', 'raw/p', 'articles', 'created', "
                "'2026-01-01T00:00:00Z', 1, '2026-01-01T00:00:00Z', 'done')"
            )
            await db.commit()

        queue = EventQueue(db_path)
        await queue.init()  # runs the migration
        assert await queue.get_page_path("legacy-1") is None  # NULL, not lost
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("PRAGMA table_info(events)") as cur:
                cols = {r[1] for r in await cur.fetchall()}
            async with db.execute("SELECT COUNT(*) FROM events") as cur:
                count = (await cur.fetchone())[0]
        assert "page_path" in cols
        assert count == 1
        # New writes use the column.
        await queue.mark_done("legacy-1", "wiki/sources/p.md")
        assert await queue.get_page_path("legacy-1") == "wiki/sources/p.md"
        await queue.close()


# --------------------------------------------------------------------------- #
# ADR-048 Fase 3: a deliberate quality skip is mark_skipped, never retried    #
# --------------------------------------------------------------------------- #

QUALITY_SKIP_PAYLOAD = json.dumps(
    {
        "status": "skipped",
        "reason": "quality-no_signal",
        "page_path": None,
        "quality_verdict": "no_signal",
        "notes": ["prose 12c < 120c floor for source"],
    }
)

QUALITY_SKIP_RESULT = {
    "content": [{"type": "text", "text": QUALITY_SKIP_PAYLOAD}],
    "structuredContent": {"result": QUALITY_SKIP_PAYLOAD},
    "isError": False,
}


class TestQualitySkip:
    @pytest.mark.asyncio
    async def test_quality_skip_marks_skipped_and_consumes_raw(
        self, kb, raw_file, ingest_db_path, fast_retries
    ) -> None:
        cfg = make_config(kb, ingest_db_path)
        queue = EventQueue(ingest_db_path)
        await queue.init()
        pool = WorkerPool(cfg, queue)
        pool._mcp = StubMcp(lambda name, args: dict(QUALITY_SKIP_RESULT))

        event = make_event(raw_file, kb)
        await queue.enqueue(event)
        await drain(queue, pool)

        counts = await queue.counts_by_status()
        # Deterministic verdict: exactly one attempt — no retry, no failed.
        assert counts.get("skipped") == 1, f"expected skipped, got {counts}"
        assert counts.get("failed", 0) == 0
        assert counts.get("done", 0) == 0
        assert len(pool._mcp.calls) == 1, "a quality skip must never be retried"

        # The raw file is consumed (declined-but-processed): moved to
        # _ingested/ so the watcher doesn't re-fire on it forever.
        assert not raw_file.exists()
        assert list((kb / "raw" / "_ingested" / "articles").iterdir())
        await queue.close()

    @pytest.mark.asyncio
    async def test_call_write_page_raises_write_skipped(self, kb, raw_file, ingest_db_path) -> None:
        from wiki_ingest.pagewrite import WriteSkippedError

        cfg = make_config(kb, ingest_db_path)
        queue = EventQueue(ingest_db_path)
        await queue.init()
        pool = WorkerPool(cfg, queue)
        pool._mcp = StubMcp(lambda name, args: dict(QUALITY_SKIP_RESULT))

        event = make_event(raw_file, kb)
        with pytest.raises(WriteSkippedError, match="quality-no_signal"):
            await pool._call_write_page(event, "deadbeef", "text/markdown", 10, None)
        await queue.close()
