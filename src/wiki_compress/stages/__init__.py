"""
Compression-pipeline stages — one module per stage.

Every stage exports a top-level callable that takes a ``str`` and returns
a ``str``. Stages that need cross-call state (e.g. URL shortener with its
mapping table) ship as an instantiable class whose ``__call__`` is the
stage callable.
"""

from __future__ import annotations

from wiki_compress.stages.dedupe import dedupe_paragraphs
from wiki_compress.stages.email_quotes import (
    QuoteCollapser,
    collapse_email_quotes,
)
from wiki_compress.stages.html_to_markdown import html_to_markdown, is_html_like
from wiki_compress.stages.line_dedupe import (
    DEFAULT_MIN_LINE_LENGTH,
    dedupe_lines,
    make_line_deduper,
)
from wiki_compress.stages.preserve import (
    apply_to_unprotected,
    has_protected_spans,
    iter_unprotected,
    mask_code_blocks,
    release_preserved,
)
from wiki_compress.stages.url_shorten import MIN_URL_LENGTH, UrlShortener
from wiki_compress.stages.whitespace_normalise import normalise_whitespace

__all__ = [
    "DEFAULT_MIN_LINE_LENGTH",
    "MIN_URL_LENGTH",
    "QuoteCollapser",
    "UrlShortener",
    "apply_to_unprotected",
    "collapse_email_quotes",
    "dedupe_lines",
    "dedupe_paragraphs",
    "has_protected_spans",
    "html_to_markdown",
    "is_html_like",
    "iter_unprotected",
    "make_line_deduper",
    "mask_code_blocks",
    "normalise_whitespace",
    "release_preserved",
]
