"""Worker PDF pre-pass for raw/papers/ (PRD-015 FR-5, ADR-032 D2).

Hermetic: ``wiki_papers`` is replaced with an in-memory module via
``sys.modules`` — the real engine chain (opendataloader/Docling/pypdf)
is never imported; no network, no JVM. Written against the CURRENT
async worker API, same style as ``test_worker_current.py``.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from wiki_ingest.config import Config
from wiki_ingest.models import IngestEvent
from wiki_ingest.paper_prepass import is_paper_pdf, render_paper_sidecar
from wiki_ingest.queue import EventQueue
from wiki_ingest.worker import WorkerPool

PDF_BYTES = b"%PDF-1.4 fake paper body just for hashing"

STATS = SimpleNamespace(
    pages=12,
    extracted_chars=34_567,
    sections_found=6,
    truncated=False,
    engine_used="opendataloader",
)

PAPER = SimpleNamespace(
    markdown="# GEPA: Reflective Prompt Evolution\n\nGEPA outperforms GRPO by 10%.",
    frontmatter={
        "type": "paper",
        "arxiv_id": "2605.23904",
        "arxiv_version": "v2",
        "title": "GEPA: Reflective Prompt Evolution",
    },
    stats=STATS,
)


def make_config(kb_root: Path, db_path: Path) -> Config:
    return Config(
        wiki_root=kb_root,
        raw_dir=kb_root / "raw",
        ingested_dir=kb_root / "raw" / "_ingested",
        db_path=db_path,
        mcp_command=("/bin/true",),
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
        mime="application/pdf",
    )


class StubMcp:
    """In-process stand-in for McpStdioClient; records every call."""

    def __init__(self, handler=None) -> None:
        self._handler = handler
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        if self._handler is None:
            raise AssertionError("MCP must not be called during the paper pre-pass")
        return self._handler(name, arguments)

    async def close(self) -> None:
        return None


async def drain(queue: EventQueue, pool: WorkerPool) -> None:
    while True:
        event = await queue.claim_next()
        if event is None:
            return
        await pool.dispatch(event)


async def event_row(db_path: Path, event_id: str) -> tuple[str, str | None]:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute(
            "SELECT status, last_error FROM events WHERE event_id=?", (event_id,)
        ) as cur:
            row = await cur.fetchone()
    assert row is not None
    return row[0], row[1]


@pytest.fixture
def kb(tmp_wiki_root: Path) -> Path:
    (tmp_wiki_root / "raw" / "papers").mkdir(exist_ok=True)
    return tmp_wiki_root


@pytest.fixture
def paper_pdf(kb: Path) -> Path:
    f = kb / "raw" / "papers" / "2605.23904v2.pdf"
    f.write_bytes(PDF_BYTES)
    return f


@pytest.fixture
def install_wiki_papers(monkeypatch):
    """Inject an in-memory ``wiki_papers`` exposing ``extract_paper``."""

    def install(extract):
        mod = types.ModuleType("wiki_papers")
        mod.extract_paper = extract
        monkeypatch.setitem(sys.modules, "wiki_papers", mod)
        return mod

    return install


async def make_pool(kb: Path, db_path: Path, mcp: StubMcp) -> tuple[EventQueue, WorkerPool]:
    cfg = make_config(kb, db_path)
    queue = EventQueue(db_path)
    await queue.init()
    pool = WorkerPool(cfg, queue)
    pool._mcp = mcp
    return queue, pool


# --------------------------------------------------------------------------- #
# 1. Pre-pass success: sidecar written, PDF moved, no write_page              #
# --------------------------------------------------------------------------- #


class TestPaperPrePassSuccess:
    @pytest.mark.asyncio
    async def test_sidecar_written_pdf_moved_event_skipped(
        self, kb, paper_pdf, ingest_db_path, install_wiki_papers
    ) -> None:
        seen: list[Path] = []

        def fake_extract(pdf_path, meta=None):
            seen.append(Path(pdf_path))
            return PAPER

        install_wiki_papers(fake_extract)
        mcp = StubMcp()  # raises if write_page is attempted
        queue, pool = await make_pool(kb, ingest_db_path, mcp)

        event = make_event(paper_pdf, kb)
        await queue.enqueue(event)
        await drain(queue, pool)

        # Sidecar next to the PDF, named by versionless arXiv ID (FR-5).
        sidecar = kb / "raw" / "papers" / "2605.23904.md"
        assert sidecar.exists()
        text = sidecar.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "type: paper" in text
        assert "2605.23904" in text
        assert "# GEPA: Reflective Prompt Evolution" in text

        # Extraction ran on the real PDF path.
        assert seen == [paper_pdf]

        # PDF moved to raw/_ingested/papers/ (current convention).
        assert not paper_pdf.exists()
        assert (kb / "raw" / "_ingested" / "papers" / "2605.23904v2.pdf").exists()

        # The PDF event is skipped (the sidecar event makes the page);
        # no MCP write_page was attempted for the binary.
        status, reason = await event_row(ingest_db_path, event.event_id)
        assert status == "skipped"
        assert reason == "paper-extracted"
        assert mcp.calls == []
        await queue.close()

    @pytest.mark.asyncio
    async def test_sidecar_name_falls_back_to_slug_without_arxiv_id(
        self, kb, ingest_db_path, install_wiki_papers
    ) -> None:
        pdf = kb / "raw" / "papers" / "My Local Paper (draft).pdf"
        pdf.write_bytes(PDF_BYTES)
        local = SimpleNamespace(
            markdown="# Local Paper\n\nBody.",
            frontmatter={"type": "paper", "metadata_source": "heuristic"},
            stats=STATS,
        )
        install_wiki_papers(lambda pdf_path, meta=None: local)
        queue, pool = await make_pool(kb, ingest_db_path, StubMcp())

        event = make_event(pdf, kb)
        await queue.enqueue(event)
        await drain(queue, pool)

        assert (kb / "raw" / "papers" / "my-local-paper-draft.md").exists()
        assert not pdf.exists()
        await queue.close()


# --------------------------------------------------------------------------- #
# 2. Degrade paths: extract failure / module absent — skip + leave PDF        #
# --------------------------------------------------------------------------- #


class TestPaperPrePassDegrade:
    @pytest.mark.asyncio
    async def test_extract_failure_skips_and_leaves_pdf(
        self, kb, paper_pdf, ingest_db_path, install_wiki_papers
    ) -> None:
        def boom(pdf_path, meta=None):
            raise RuntimeError("engine exploded")

        install_wiki_papers(boom)
        mcp = StubMcp()
        queue, pool = await make_pool(kb, ingest_db_path, mcp)

        event = make_event(paper_pdf, kb)
        await queue.enqueue(event)
        await drain(queue, pool)

        status, reason = await event_row(ingest_db_path, event.event_id)
        assert status == "skipped"
        assert reason == "paper-extract-failed:RuntimeError"
        # The PDF stays put for a later retry; no sidecar, no MCP call.
        assert paper_pdf.exists()
        assert not list((kb / "raw" / "papers").glob("*.md"))
        assert mcp.calls == []
        await queue.close()

    @pytest.mark.asyncio
    async def test_wiki_papers_unimportable_skips_without_crash(
        self, kb, paper_pdf, ingest_db_path, monkeypatch
    ) -> None:
        # sys.modules[name] = None makes `from wiki_papers import ...`
        # raise ImportError even when the real package is installed.
        monkeypatch.setitem(sys.modules, "wiki_papers", None)
        queue, pool = await make_pool(kb, ingest_db_path, StubMcp())

        event = make_event(paper_pdf, kb)
        await queue.enqueue(event)
        await drain(queue, pool)

        status, reason = await event_row(ingest_db_path, event.event_id)
        assert status == "skipped"
        assert reason == "paper-extractor-unavailable"
        assert paper_pdf.exists()
        await queue.close()


# --------------------------------------------------------------------------- #
# 3. PDFs OUTSIDE raw/papers/ keep the current binary-page behaviour          #
# --------------------------------------------------------------------------- #


class TestNonPaperPdfUnchanged:
    @pytest.mark.asyncio
    async def test_pdf_outside_papers_writes_binary_page(
        self, kb, ingest_db_path, install_wiki_papers
    ) -> None:
        pdf = kb / "raw" / "articles" / "scan.pdf"
        pdf.write_bytes(PDF_BYTES)

        def must_not_extract(pdf_path, meta=None):
            raise AssertionError("extract_paper must not run outside raw/papers/")

        install_wiki_papers(must_not_extract)

        def fake_write_page(name: str, args: dict) -> dict:
            full = kb / args["page_path"]
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(args["content"], encoding="utf-8")
            payload = json.dumps({"status": "created", "page_path": args["page_path"]})
            return {"content": [{"type": "text", "text": payload}], "isError": False}

        mcp = StubMcp(fake_write_page)
        queue, pool = await make_pool(kb, ingest_db_path, mcp)

        event = make_event(pdf, kb)
        await queue.enqueue(event)
        await drain(queue, pool)

        counts = await queue.counts_by_status()
        assert counts.get("done") == 1, f"expected done, got {counts}"
        assert [name for name, _ in mcp.calls] == ["write_page"]
        page = kb / "wiki" / "sources" / "scan.md"
        assert page.exists()
        assert "content not inlined" in page.read_text(encoding="utf-8")
        await queue.close()


# --------------------------------------------------------------------------- #
# 4. Pure helpers                                                             #
# --------------------------------------------------------------------------- #


class TestHelpers:
    def test_is_paper_pdf_shapes(self) -> None:
        assert is_paper_pdf("raw/papers/2605.23904v2.pdf")
        assert is_paper_pdf("papers/x.pdf")
        assert is_paper_pdf("raw/papers/nested/x.pdf")
        assert not is_paper_pdf("raw/articles/x.pdf")
        assert not is_paper_pdf("raw/papers")  # the dir itself, not a member
        assert not is_paper_pdf("x.pdf")

    def test_render_paper_sidecar_composes_frontmatter(self) -> None:
        out = render_paper_sidecar({"type": "paper", "title": "T"}, "# T\n\nBody")
        assert out.startswith("---\n")
        assert "type: paper" in out
        assert out.rstrip().endswith("Body")

    def test_render_paper_sidecar_passthrough_when_engine_composed(self) -> None:
        doc = "---\ntype: paper\n---\n\n# T\n\nBody"
        assert render_paper_sidecar({"ignored": True}, doc) == doc + "\n"
