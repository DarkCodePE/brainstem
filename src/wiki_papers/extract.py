"""PDF → markdown extraction engine chain (PRD-015 FR-3/FR-4/FR-7, ADR-032 D1).

Resolution chain — each step isolated, the chain as a whole NEVER raises out of
:func:`extract_paper` (degrade-first, same posture as the code-graph leg of
ADR-022):

1. **opendataloader-pdf** — deterministic quality engine, used when ``java``
   11+ is on PATH. Markdown output, content-safety filter ON (its hidden-text
   stripping runs *in addition to*, never instead of, the ADR-015 guard).
2. **Docling** — optional ML escalation (``sbw[papers-ml]`` extra), table
   structure on, OCR off, ~60-page cap (course-validated defaults).
3. **pypdf** — pure-Python plain-text floor → ``extraction: degraded``.
4. **Metadata-only** — page built from :class:`~wiki_papers.arxiv.PaperMeta`
   alone → ``extraction: unavailable``.

Pre-parse validation (non-empty, ``%PDF-`` magic, page-count probe, 25 MB cap
— PRD-015 SR-3) runs before any engine touches the file.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pypdf import PdfReader

from wiki_papers.arxiv import MAX_PDF_BYTES, PaperMeta, parse_arxiv_id

__all__ = [
    "MAX_PAGES",
    "EngineName",
    "PaperStats",
    "PaperMarkdown",
    "extract_paper",
    "render_page",
]

logger = logging.getLogger(__name__)

MAX_PAGES = 60
"""Page cap for full structural extraction (ADR-032 D1, course-validated).
Larger PDFs are truncated to the first MAX_PAGES pages — never silently:
``PaperStats.truncated`` flips on (FR-7)."""

MIN_JAVA_MAJOR = 11

EngineName = Literal["opendataloader", "docling", "pypdf", "none"]

_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class PaperStats:
    """Accounting for an extraction run — truncation is never silent (FR-7)."""

    pages: int
    extracted_chars: int
    sections_found: int
    truncated: bool
    engine_used: EngineName


@dataclass(frozen=True, slots=True)
class PaperMarkdown:
    """The extraction product: markdown body + FR-4 frontmatter + stats."""

    markdown: str
    frontmatter: dict = field(default_factory=dict)
    stats: PaperStats = PaperStats(0, 0, 0, False, "none")


# ───────────────────────────── pre-parse validation ──────────────────────────


def _probe_pdf(pdf_path: Path | None) -> int | None:
    """SR-3 pre-parse validation. Returns the page count, or None if the file
    must not be handed to any engine (missing, empty, oversized, not a PDF,
    unreadable structure)."""
    if pdf_path is None:
        return None
    try:
        p = Path(pdf_path)
        if not p.is_file():
            return None
        size = p.stat().st_size
        if size == 0 or size > MAX_PDF_BYTES:
            logger.warning("pdf rejected by size check (%d bytes): %s", size, p)
            return None
        with p.open("rb") as fh:
            if not fh.read(8).startswith(b"%PDF-"):
                logger.warning("missing %%PDF- magic header: %s", p)
                return None
        pages = len(PdfReader(p).pages)
        if pages <= 0:
            return None
        return pages
    except Exception as exc:
        logger.warning("pdf probe failed for %s: %s", pdf_path, type(exc).__name__)
        return None


# ───────────────────────────────── engines ───────────────────────────────────


def _java_major() -> int | None:
    """Major version of the ``java`` on PATH, or None when absent/unparseable."""
    if shutil.which("java") is None:
        return None
    try:
        proc = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=15, check=False
        )
        m = re.search(r'version "(\d+)(?:\.(\d+))?', proc.stderr + proc.stdout)
        if not m:
            return None
        major = int(m.group(1))
        if major == 1 and m.group(2):  # legacy "1.8" scheme
            major = int(m.group(2))
        return major
    except Exception:
        return None


def _opendataloader_available() -> bool:
    """Primary engine gate: importable wrapper + Java 11+ on PATH (ADR-032 D1)."""
    try:
        import opendataloader_pdf  # noqa: F401
    except ImportError:
        return False
    major = _java_major()
    return major is not None and major >= MIN_JAVA_MAJOR


def _run_opendataloader(pdf_path: Path, pages: int) -> str | None:
    """opendataloader-pdf → markdown. Safety filter stays ON (we never pass
    ``content_safety_off``). Returns None on any failure or empty output."""
    import opendataloader_pdf

    with tempfile.TemporaryDirectory(prefix="sbw-odl-") as td:
        kwargs: dict = {}
        if pages > MAX_PAGES:
            kwargs["pages"] = f"1-{MAX_PAGES}"
        opendataloader_pdf.convert(
            input_path=str(pdf_path),
            output_dir=td,
            format="markdown",
            quiet=True,
            **kwargs,
        )
        md_files = sorted(Path(td).glob("**/*.md"))
        if not md_files:
            return None
        text = md_files[0].read_text(encoding="utf-8", errors="replace").strip()
        return text or None


def _run_docling(pdf_path: Path) -> str | None:
    """Docling (optional ``papers-ml`` extra) → markdown, or None when the
    import is unavailable or conversion fails/yields nothing."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        return None

    pipeline_options = PdfPipelineOptions(do_table_structure=True, do_ocr=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    result = converter.convert(str(pdf_path), max_num_pages=MAX_PAGES)
    text = result.document.export_to_markdown().strip()
    return text or None


def _run_pypdf(pdf_path: Path, pages: int) -> str | None:
    """pypdf plain-text floor (first MAX_PAGES pages), or None when no text."""
    reader = PdfReader(pdf_path)
    chunks: list[str] = []
    for page in reader.pages[:MAX_PAGES]:
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text.strip())
    body = "\n\n".join(chunks).strip()
    return body or None


def _run_engine(name: str, runner, *args) -> str | None:
    """Run one engine, swallowing every failure (degrade-first)."""
    try:
        return runner(*args)
    except Exception as exc:
        logger.warning("paper engine %s failed: %s: %s", name, type(exc).__name__, exc)
        return None


# ──────────────────────────── metadata / frontmatter ─────────────────────────


def _heuristic_meta(pdf_path: Path | None) -> dict:
    """Best-effort local-PDF metadata (PRD-015 FR-4 ``metadata_source:
    heuristic``): arXiv ID from the filename, title/authors from the PDF info
    dictionary, title falling back to a prettified filename stem."""
    arxiv_id: str | None = None
    version: str | None = None
    title = ""
    authors: list[str] = []
    if pdf_path is not None:
        stem = Path(pdf_path).stem
        try:
            arxiv_id, version = parse_arxiv_id(stem)
        except ValueError:
            pass
        try:
            info = PdfReader(pdf_path).metadata
            if info:
                title = (info.title or "").strip()
                if info.author and info.author.strip():
                    authors = [a.strip() for a in info.author.split(",") if a.strip()]
        except Exception:
            pass
        if not title:
            title = re.sub(r"[-_]+", " ", stem).strip() or stem
    return {
        "arxiv_id": arxiv_id,
        "arxiv_version": version,
        "title": title,
        "authors": authors,
        "published": None,
        "categories": [],
        "abstract": "",
        "pdf_url": None,
        "sources": [Path(pdf_path).name] if pdf_path is not None else [],
    }


def _build_frontmatter(meta: PaperMeta | None, pdf_path: Path | None, extraction: str) -> dict:
    """FR-4 frontmatter contract — every key always present."""
    if meta is not None:
        base = {
            "arxiv_id": meta.arxiv_id,
            "arxiv_version": meta.version,
            "title": meta.title,
            "authors": list(meta.authors),
            "published": meta.published,
            "categories": list(meta.categories),
            "abstract": meta.abstract,
            "pdf_url": meta.pdf_url,
            "sources": [meta.abs_url],
        }
        source = "arxiv-api"
    else:
        base = _heuristic_meta(pdf_path)
        source = "heuristic"
    return {
        "type": "paper",
        "arxiv_id": base["arxiv_id"],
        "arxiv_version": base["arxiv_version"],
        "title": base["title"],
        "authors": base["authors"],
        "published": base["published"],
        "categories": base["categories"],
        "abstract": base["abstract"],
        "pdf_url": base["pdf_url"],
        "extraction": extraction,
        "metadata_source": source,
        "sources": base["sources"],
    }


def _metadata_only_body(frontmatter: dict) -> str:
    """The engine-less floor: a page from metadata alone (ADR-032 D1 step 4)."""
    title = frontmatter.get("title") or frontmatter.get("arxiv_id") or "Untitled paper"
    lines = [f"# {title}", ""]
    abstract = frontmatter.get("abstract") or ""
    if abstract:
        lines += ["## Abstract", "", abstract, ""]
    lines.append("> extraction: unavailable — page generated from metadata only.")
    return "\n".join(lines)


# ─────────────────────────────── public surface ──────────────────────────────


def extract_paper(pdf_path: Path | None, meta: PaperMeta | None = None) -> PaperMarkdown:
    """Convert a PDF into markdown via the ADR-032 D1 engine chain.

    NEVER raises: every failure degrades to the next engine and ultimately to a
    metadata-only page (``extraction: unavailable``).

    Args:
        pdf_path: Path to the local PDF, or None to skip straight to the
            metadata-only floor (e.g. when the download failed).
        meta: arXiv API metadata when available (``metadata_source:
            arxiv-api``); None switches frontmatter to heuristic mode.
    """
    try:
        return _extract(Path(pdf_path) if pdf_path is not None else None, meta)
    except Exception as exc:  # belt-and-braces: the chain must not fail ingest
        logger.error("extract_paper degraded unexpectedly: %s: %s", type(exc).__name__, exc)
        frontmatter = _build_frontmatter(meta, None, "unavailable")
        body = _metadata_only_body(frontmatter)
        return PaperMarkdown(
            markdown=body,
            frontmatter=frontmatter,
            stats=PaperStats(0, 0, 0, False, "none"),
        )


def _extract(pdf_path: Path | None, meta: PaperMeta | None) -> PaperMarkdown:
    pages = _probe_pdf(pdf_path)

    body: str | None = None
    engine: EngineName = "none"
    if pages is not None:
        assert pdf_path is not None
        if _opendataloader_available():
            body = _run_engine("opendataloader", _run_opendataloader, pdf_path, pages)
            if body is not None:
                engine = "opendataloader"
        if body is None:
            body = _run_engine("docling", _run_docling, pdf_path)
            if body is not None:
                engine = "docling"
        if body is None:
            body = _run_engine("pypdf", _run_pypdf, pdf_path, pages)
            if body is not None:
                engine = "pypdf"

    if engine in ("opendataloader", "docling"):
        extraction = "full"
    elif engine == "pypdf":
        extraction = "degraded"
    else:
        extraction = "unavailable"

    frontmatter = _build_frontmatter(meta, pdf_path, extraction)
    if body is None:
        body = _metadata_only_body(frontmatter)
        extracted_chars = 0
    else:
        extracted_chars = len(body)

    stats = PaperStats(
        pages=pages or 0,
        extracted_chars=extracted_chars,
        sections_found=len(_HEADING_RE.findall(body)),
        truncated=engine != "none" and (pages or 0) > MAX_PAGES,
        engine_used=engine,
    )
    return PaperMarkdown(markdown=body, frontmatter=frontmatter, stats=stats)


def render_page(paper: PaperMarkdown) -> str:
    """Serialise a :class:`PaperMarkdown` into the on-disk ``raw/papers/*.md``
    shape: YAML frontmatter block + markdown body (PRD-015 FR-5)."""
    fm = yaml.safe_dump(paper.frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{fm}\n---\n\n{paper.markdown.strip()}\n"
