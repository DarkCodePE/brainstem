"""``sbw-ingest-paper`` — CLI caller for the paper pipeline (ADR-032 D2).

One thin, route-agnostic entry point::

    sbw-ingest-paper <arxiv-id|arxiv-url|pdf-path> [--root PATH]

- **arXiv mode** (ID/URL): fetch metadata → download PDF (cached under
  ``$SBW_PAPER_CACHE`` or ``<root>/.paper-cache`` — outside ``raw/`` so the
  ingest watcher never enqueues cached PDFs) → extract → write
  ``<root>/raw/papers/<arxiv_id>.md``.
- **Local-PDF mode** (existing ``.pdf`` path): extract with heuristic metadata
  → write ``<root>/raw/papers/<slugified-filename>.md``.

Prints a one-line JSON result (page path + ``PaperStats``) and exits 0 on any
degrade (degrade-first, PRD-015 FR-3); non-zero only on usage errors or when
arXiv metadata itself is unreachable (no page can exist without it).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from wiki_papers.arxiv import ArxivError, PaperMeta, download_pdf, fetch_arxiv, parse_arxiv_id
from wiki_papers.extract import PaperMarkdown, extract_paper, render_page

__all__ = ["main"]

EXIT_OK = 0
EXIT_FETCH_FAILED = 1
EXIT_USAGE = 2


def _slugify(stem: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return slug or "paper"


def _write_result(papers_dir: Path, filename: str, paper: PaperMarkdown) -> Path:
    papers_dir.mkdir(parents=True, exist_ok=True)
    out_path = papers_dir / filename
    out_path.write_text(render_page(paper), encoding="utf-8")
    return out_path


def _emit(out_path: Path, paper: PaperMarkdown, notes: list[str]) -> None:
    stats = paper.stats
    print(
        json.dumps(
            {
                "page_path": str(out_path),
                "engine_used": stats.engine_used,
                "extraction": paper.frontmatter.get("extraction"),
                "pages": stats.pages,
                "extracted_chars": stats.extracted_chars,
                "sections_found": stats.sections_found,
                "truncated": stats.truncated,
                "notes": notes,
            }
        )
    )


def _cache_dir(root: Path) -> Path:
    # Same convention as the MCP caller: outside raw/, so neither the inotify
    # watch nor the catch-up rglob ever enqueues a cached PDF.
    env = os.environ.get("SBW_PAPER_CACHE")
    return Path(env) if env else root / ".paper-cache"


def _ingest_arxiv(id_or_url: str, papers_dir: Path, cache_dir: Path) -> int:
    try:
        meta: PaperMeta = fetch_arxiv(id_or_url)
    except ArxivError as exc:
        print(json.dumps({"error": "arxiv-fetch-failed", "detail": str(exc)}), file=sys.stderr)
        return EXIT_FETCH_FAILED

    notes: list[str] = []
    pdf_path: Path | None = None
    try:
        pdf_path = download_pdf(meta, cache_dir)
    except ArxivError as exc:  # degrade to metadata-only, never fail the ingest
        notes.append(f"pdf-download-failed: {exc}")

    paper = extract_paper(pdf_path, meta)
    out_path = _write_result(papers_dir, f"{meta.arxiv_id}.md", paper)
    _emit(out_path, paper, notes)
    return EXIT_OK


def _ingest_local(pdf_path: Path, papers_dir: Path) -> int:
    paper = extract_paper(pdf_path)
    out_path = _write_result(papers_dir, f"{_slugify(pdf_path.stem)}.md", paper)
    _emit(out_path, paper, [])
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sbw-ingest-paper",
        description="Ingest an arXiv paper (ID/URL) or a local PDF into raw/papers/.",
    )
    parser.add_argument("input", help="arXiv ID, arXiv URL, or path to a local PDF file")
    parser.add_argument(
        "--root",
        default="knowledge-base",
        help="Knowledge-base root (default: knowledge-base)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    papers_dir = root / "raw" / "papers"

    candidate = Path(args.input)
    if candidate.suffix.lower() == ".pdf" and candidate.is_file():
        return _ingest_local(candidate, papers_dir)

    try:
        parse_arxiv_id(args.input)
    except ValueError as exc:
        print(
            json.dumps({"error": "usage", "detail": f"not an arXiv id/url or PDF path: {exc}"}),
            file=sys.stderr,
        )
        return EXIT_USAGE
    return _ingest_arxiv(args.input, papers_dir, _cache_dir(root))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
