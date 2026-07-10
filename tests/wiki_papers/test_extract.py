"""Unit tests for ``wiki_papers.extract`` — engine chain, validation,
frontmatter contract. Hermetic: engines are monkeypatched; the only real
engine exercised is the pypdf floor against a synthetic in-repo PDF.
"""

from __future__ import annotations

import pytest

from wiki_papers import extract as extract_mod
from wiki_papers.extract import (
    MAX_PAGES,
    PaperMarkdown,
    PaperStats,
    extract_paper,
    render_page,
)

from .conftest import make_text_pdf

FR4_KEYS = [
    "type",
    "arxiv_id",
    "arxiv_version",
    "title",
    "authors",
    "published",
    "categories",
    "abstract",
    "pdf_url",
    "extraction",
    "metadata_source",
    "sources",
]

FAKE_MD = "# Attention Is Not Enough\n\n## Method\n\nbody text\n\n## Results\n\n2.4x speedup\n"


@pytest.fixture
def no_engines(monkeypatch):
    """Force the quality engines off so the chain falls through to pypdf."""
    monkeypatch.setattr(extract_mod, "_opendataloader_available", lambda: False)
    monkeypatch.setattr(extract_mod, "_run_docling", lambda _p: None)


# ───────────────────────── pre-parse validation (SR-3) ───────────────────────


def test_not_a_pdf_degrades_to_metadata_only(tmp_path, sample_meta):
    bad = tmp_path / "fake.pdf"
    bad.write_bytes(b"MZ this is not a pdf at all")
    paper = extract_paper(bad, sample_meta)
    assert paper.stats.engine_used == "none"
    assert paper.frontmatter["extraction"] == "unavailable"
    assert paper.stats.extracted_chars == 0
    assert sample_meta.abstract in paper.markdown  # metadata-only page has the abstract


def test_empty_file_degrades(tmp_path, sample_meta):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    paper = extract_paper(empty, sample_meta)
    assert paper.stats.engine_used == "none"
    assert paper.frontmatter["extraction"] == "unavailable"


def test_missing_file_and_none_path_degrade(tmp_path, sample_meta):
    assert extract_paper(tmp_path / "ghost.pdf", sample_meta).stats.engine_used == "none"
    assert extract_paper(None, sample_meta).stats.engine_used == "none"


def test_oversize_pdf_rejected_before_engines(tmp_path, sample_meta, monkeypatch):
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF-1.4" + b"\0" * extract_mod.MAX_PDF_BYTES)

    def must_not_run(*_a):
        raise AssertionError("engine ran on an oversized pdf")

    monkeypatch.setattr(extract_mod, "_run_pypdf", must_not_run)
    paper = extract_paper(big, sample_meta)
    assert paper.stats.engine_used == "none"


# ───────────────────────────── engine chain order ────────────────────────────


def test_opendataloader_wins_when_available(text_pdf, sample_meta, monkeypatch):
    monkeypatch.setattr(extract_mod, "_opendataloader_available", lambda: True)
    monkeypatch.setattr(extract_mod, "_run_opendataloader", lambda _p, _n: FAKE_MD)
    monkeypatch.setattr(extract_mod, "_run_docling", lambda _p: pytest.fail("docling must not run"))
    paper = extract_paper(text_pdf, sample_meta)
    assert paper.stats.engine_used == "opendataloader"
    assert paper.frontmatter["extraction"] == "full"
    assert paper.stats.sections_found == 3  # one H1 + two H2
    assert paper.stats.extracted_chars == len(FAKE_MD)
    assert paper.stats.pages == 1


def test_docling_runs_when_opendataloader_unavailable(text_pdf, sample_meta, monkeypatch):
    monkeypatch.setattr(extract_mod, "_opendataloader_available", lambda: False)
    monkeypatch.setattr(extract_mod, "_run_docling", lambda _p: FAKE_MD)
    paper = extract_paper(text_pdf, sample_meta)
    assert paper.stats.engine_used == "docling"
    assert paper.frontmatter["extraction"] == "full"


def test_engine_exception_degrades_to_next(text_pdf, sample_meta, monkeypatch):
    monkeypatch.setattr(extract_mod, "_opendataloader_available", lambda: True)

    def explode(_p, _n):
        raise RuntimeError("jvm fell over")

    monkeypatch.setattr(extract_mod, "_run_opendataloader", explode)
    monkeypatch.setattr(extract_mod, "_run_docling", lambda _p: None)
    paper = extract_paper(text_pdf, sample_meta)  # real pypdf floor
    assert paper.stats.engine_used == "pypdf"
    assert paper.frontmatter["extraction"] == "degraded"
    assert "Hello arXiv" in paper.markdown


def test_pypdf_floor_extracts_real_text(text_pdf, sample_meta, no_engines):
    paper = extract_paper(text_pdf, sample_meta)
    assert paper.stats.engine_used == "pypdf"
    assert paper.frontmatter["extraction"] == "degraded"
    assert "Hello arXiv" in paper.markdown
    assert paper.stats.extracted_chars > 0


def test_all_engines_unavailable_yields_metadata_page_never_raises(
    text_pdf, sample_meta, no_engines, monkeypatch
):
    """AC-3: full forced-unavailable chain still succeeds with a degrade note."""
    monkeypatch.setattr(extract_mod, "_run_pypdf", lambda _p, _n: None)
    paper = extract_paper(text_pdf, sample_meta)
    assert paper.stats.engine_used == "none"
    assert paper.frontmatter["extraction"] == "unavailable"
    assert "extraction: unavailable" in paper.markdown  # visible degrade note
    assert paper.frontmatter["abstract"] == sample_meta.abstract


def test_truncation_flag_when_pages_exceed_cap(text_pdf, sample_meta, monkeypatch):
    monkeypatch.setattr(extract_mod, "_probe_pdf", lambda _p: MAX_PAGES + 10)
    monkeypatch.setattr(extract_mod, "_opendataloader_available", lambda: True)
    seen_pages: list[int] = []

    def fake_engine(_p, n):
        seen_pages.append(n)
        return FAKE_MD

    monkeypatch.setattr(extract_mod, "_run_opendataloader", fake_engine)
    paper = extract_paper(text_pdf, sample_meta)
    assert paper.stats.truncated is True
    assert paper.stats.pages == MAX_PAGES + 10
    assert seen_pages == [MAX_PAGES + 10]


# ─────────────────────────── frontmatter contract ────────────────────────────


def test_frontmatter_contract_complete_with_meta(text_pdf, sample_meta, no_engines):
    fm = extract_paper(text_pdf, sample_meta).frontmatter
    assert list(fm.keys()) == FR4_KEYS
    assert fm["type"] == "paper"
    assert fm["arxiv_id"] == "2605.23904"
    assert fm["arxiv_version"] == "2"
    assert fm["title"] == sample_meta.title
    assert fm["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert fm["published"] == sample_meta.published
    assert fm["categories"] == ["cs.AI", "cs.CL"]
    assert fm["abstract"] == sample_meta.abstract
    assert fm["pdf_url"] == sample_meta.pdf_url
    assert fm["metadata_source"] == "arxiv-api"
    assert fm["sources"] == ["https://arxiv.org/abs/2605.23904"]


def test_heuristic_mode_without_meta(tmp_path, no_engines):
    pdf = tmp_path / "2605.23904v2.pdf"
    pdf.write_bytes(make_text_pdf())
    fm = extract_paper(pdf).frontmatter
    assert list(fm.keys()) == FR4_KEYS
    assert fm["metadata_source"] == "heuristic"
    assert fm["arxiv_id"] == "2605.23904"  # recovered from the filename
    assert fm["arxiv_version"] == "2"
    assert fm["sources"] == ["2605.23904v2.pdf"]


def test_heuristic_title_falls_back_to_filename(tmp_path, no_engines):
    pdf = tmp_path / "my-great_paper.pdf"
    pdf.write_bytes(make_text_pdf())
    fm = extract_paper(pdf).frontmatter
    assert fm["arxiv_id"] is None
    assert fm["title"] == "my great paper"


# ──────────────────────────────── render_page ────────────────────────────────


def test_render_page_emits_frontmatter_block(text_pdf, sample_meta, no_engines):
    page = render_page(extract_paper(text_pdf, sample_meta))
    assert page.startswith("---\n")
    assert "\n---\n\n" in page
    assert "type: paper" in page
    assert "arxiv_id: '2605.23904'" in page or "arxiv_id: 2605.23904" in page
    assert "Hello arXiv" in page
    assert page.endswith("\n")


def test_paper_markdown_dataclass_shape():
    stats = PaperStats(1, 2, 3, False, "pypdf")
    pm = PaperMarkdown(markdown="x", frontmatter={"type": "paper"}, stats=stats)
    assert (pm.stats.pages, pm.stats.extracted_chars) == (1, 2)
    assert pm.stats.engine_used == "pypdf"
