"""``ingest_arxiv_paper`` MCP tool (PRD-015, ADR-032 D2 — agent/Telegram route).

Hermetic: ``wiki_papers`` is replaced with an in-memory module via
``sys.modules`` (no network, no arXiv, no extraction engines), WIKI_ROOT
points at a temp vault, and the ``write_page`` boundary is armed to fail
the test if the tool ever crosses it (SR-2: the sidecar flows through
the normal ingest envelope; the tool itself never writes wiki pages).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from wiki_agent import mcp_server

STATS = SimpleNamespace(
    pages=12,
    extracted_chars=34_567,
    sections_found=6,
    truncated=False,
    engine_used="opendataloader",
)

META = SimpleNamespace(
    arxiv_id="2605.23904",
    version="v2",
    title="GEPA: Reflective Prompt Evolution",
    pdf_url="https://arxiv.org/pdf/2605.23904v2",
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


@pytest.fixture
def wiki_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(mcp_server, "WIKI_ROOT", str(tmp_path))
    monkeypatch.delenv("SBW_PAPER_CACHE", raising=False)

    # SR-2 guard: the tool must never reach write_page (or any wiki tool).
    def _forbidden(name: str, kwargs: dict) -> str:
        raise AssertionError(f"ingest_arxiv_paper must not call wiki tool {name!r}")

    monkeypatch.setattr(mcp_server, "_call", _forbidden)
    return tmp_path


@pytest.fixture
def install_wiki_papers(monkeypatch):
    """Inject an in-memory ``wiki_papers`` and record every call."""

    def install(**overrides):
        calls: dict[str, list] = {"parse": [], "fetch": [], "download": [], "extract": []}

        def parse_arxiv_id(url_or_id):
            calls["parse"].append(url_or_id)
            return ("2605.23904", "v2")

        def fetch_arxiv(url_or_id):
            calls["fetch"].append(url_or_id)
            return META

        def download_pdf(meta, cache_dir):
            calls["download"].append((meta, Path(cache_dir)))
            return Path("/tmp/cache/2605.23904.pdf")

        def extract_paper(pdf_path, meta=None):
            calls["extract"].append((Path(pdf_path), meta))
            return PAPER

        mod = types.ModuleType("wiki_papers")
        mod.parse_arxiv_id = overrides.get("parse_arxiv_id", parse_arxiv_id)
        mod.fetch_arxiv = overrides.get("fetch_arxiv", fetch_arxiv)
        mod.download_pdf = overrides.get("download_pdf", download_pdf)
        mod.extract_paper = overrides.get("extract_paper", extract_paper)
        mod._calls = calls
        monkeypatch.setitem(sys.modules, "wiki_papers", mod)
        return mod

    return install


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_extracts_and_writes_raw_sidecar(self, wiki_root, install_wiki_papers) -> None:
        mod = install_wiki_papers()
        out = json.loads(await mcp_server.ingest_arxiv_paper("https://arxiv.org/abs/2605.23904v2"))

        assert out["status"] == "extracted"
        assert out["raw_path"] == "raw/papers/2605.23904.md"
        assert out["arxiv_id"] == "2605.23904"
        assert out["arxiv_version"] == "v2"
        assert out["title"] == "GEPA: Reflective Prompt Evolution"
        # FR-7: PaperStats surfaced, no silent truncation.
        assert out["stats"] == {
            "pages": 12,
            "extracted_chars": 34_567,
            "sections_found": 6,
            "truncated": False,
            "engine_used": "opendataloader",
        }

        # Sidecar on disk with the FR-4 frontmatter, versionless ID name.
        sidecar = wiki_root / "raw" / "papers" / "2605.23904.md"
        assert sidecar.exists()
        text = sidecar.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "type: paper" in text
        assert "# GEPA: Reflective Prompt Evolution" in text

        # fetch → download → extract, once each; cache OUTSIDE raw/.
        assert mod._calls["fetch"] == ["https://arxiv.org/abs/2605.23904v2"]
        ((_meta, cache_dir),) = mod._calls["download"]
        assert cache_dir == wiki_root / ".paper-cache"
        ((pdf_path, meta),) = mod._calls["extract"]
        assert pdf_path == Path("/tmp/cache/2605.23904.pdf")
        assert meta is META

    @pytest.mark.asyncio
    async def test_async_wiki_papers_contract_also_supported(
        self, wiki_root, install_wiki_papers
    ) -> None:
        """The frozen contract does not pin sync/async — both must work."""

        async def fetch_arxiv(url_or_id):
            return META

        async def download_pdf(meta, cache_dir):
            return Path("/tmp/cache/2605.23904.pdf")

        async def extract_paper(pdf_path, meta=None):
            return PAPER

        install_wiki_papers(
            fetch_arxiv=fetch_arxiv, download_pdf=download_pdf, extract_paper=extract_paper
        )
        out = json.loads(await mcp_server.ingest_arxiv_paper("2605.23904"))
        assert out["status"] == "extracted"
        assert (wiki_root / "raw" / "papers" / "2605.23904.md").exists()


class TestFailures:
    @pytest.mark.asyncio
    async def test_invalid_input_returns_typed_error_before_any_fetch(
        self, wiki_root, install_wiki_papers
    ) -> None:
        def bad_parse(url_or_id):
            raise ValueError("not an arXiv id or URL")

        mod = install_wiki_papers(parse_arxiv_id=bad_parse)
        out = json.loads(await mcp_server.ingest_arxiv_paper("github.com/o/r"))
        assert out["status"] == "failed"
        assert out["error"] == "ValueError"
        assert out["url_or_id"] == "github.com/o/r"
        assert mod._calls["fetch"] == []
        assert mod._calls["download"] == []

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_typed_error_no_sidecar(
        self, wiki_root, install_wiki_papers
    ) -> None:
        def boom(url_or_id):
            raise ConnectionError("arxiv unreachable")

        install_wiki_papers(fetch_arxiv=boom)
        out = json.loads(await mcp_server.ingest_arxiv_paper("2605.23904"))
        assert out["status"] == "failed"
        assert out["error"] == "ConnectionError"
        assert out["arxiv_id"] == "2605.23904"
        assert not (wiki_root / "raw" / "papers").exists()

    @pytest.mark.asyncio
    async def test_wiki_papers_unimportable_is_a_clean_error(self, wiki_root, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "wiki_papers", None)
        out = json.loads(await mcp_server.ingest_arxiv_paper("2605.23904"))
        assert out["status"] == "failed"
        assert out["error"] == "PapersUnavailable"
