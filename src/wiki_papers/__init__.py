"""``wiki_papers`` — papers-as-knowledge-source bounded context (PRD-015 / ADR-032).

Turns an arXiv ID/URL or a local PDF into an extracted markdown page with the
FR-4 frontmatter contract, written to ``raw/papers/`` where the existing ingest
route picks it up unchanged (route-agnostic placement, ADR-032 D2). Extracted
text is ``ingested-untrusted`` and flows through the ADR-006 envelope +
ADR-015 guard downstream.

Public contract (keep stable — callers in the MCP server, the Hermes pre-pass
and the daemon worker code against exactly this surface):

- ``arxiv``   — :func:`parse_arxiv_id`, :func:`fetch_arxiv`,
  :func:`download_pdf`, :class:`PaperMeta` (3s rate limit, 25 MB cap,
  arxiv.org/export.arxiv.org allowlist — SR-1).
- ``extract`` — :func:`extract_paper` (ADR-032 D1 engine chain:
  opendataloader-pdf → Docling → pypdf → metadata-only; never raises),
  :class:`PaperMarkdown`, :class:`PaperStats`, :func:`render_page`.
- ``cli``     — ``sbw-ingest-paper`` console entry point.
"""

from __future__ import annotations

from wiki_papers.arxiv import (
    ALLOWED_HOSTS,
    MAX_PDF_BYTES,
    RATE_LIMIT_SECONDS,
    ArxivError,
    PaperMeta,
    PdfOversize,
    download_pdf,
    fetch_arxiv,
    parse_arxiv_id,
)
from wiki_papers.extract import (
    MAX_PAGES,
    PaperMarkdown,
    PaperStats,
    extract_paper,
    render_page,
)

__all__ = [
    "ALLOWED_HOSTS",
    "MAX_PAGES",
    "MAX_PDF_BYTES",
    "RATE_LIMIT_SECONDS",
    "ArxivError",
    "PaperMeta",
    "PaperMarkdown",
    "PaperStats",
    "PdfOversize",
    "download_pdf",
    "extract_paper",
    "fetch_arxiv",
    "parse_arxiv_id",
    "render_page",
]
