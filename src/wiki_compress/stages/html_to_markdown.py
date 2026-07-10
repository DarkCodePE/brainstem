"""
HTML → Markdown stage.

Strips ``<script>``, ``<style>``, ``<noscript>`` and HTML comments, then
hands the remainder to ``markdownify.markdownify`` for the actual conversion.
``markdownify`` is already a top-level dependency (see ``pyproject.toml``)
and is a pure-Python wrapper around ``beautifulsoup4``.

Why a stage and not a one-liner over markdownify? Because:

1. The pipeline wants a single ``(name, str -> str)`` callable.
2. We strip blackholes (script/style/noscript) first — markdownify keeps
   ``<style>`` content as plain text otherwise, which is the opposite of
   what we want.
3. We post-process the markdown to collapse the extra blank lines
   markdownify introduces between blocks (whitespace normalisation runs
   later, but we want a tidy hand-off).
4. We **only run if the input is HTML-ish**. A markdown-only string
   passes through untouched so the stage is safe to include in the
   default pipeline.
"""

from __future__ import annotations

import re

from markdownify import markdownify

# Heuristic — if any of these markers appear, we treat the input as HTML.
_HTML_MARKER_RE = re.compile(
    r"<(?:html|head|body|div|p|span|h[1-6]|table|ul|ol|li|a|code|pre|script|style)\b",
    re.IGNORECASE,
)

# Drop entirely — content inside is non-textual noise for an LLM.
_DROP_BLOCKS_RE = re.compile(
    r"<(?P<tag>script|style|noscript)\b[^>]*>.*?</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)

# Drop HTML comments — `<!-- … -->`.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Markdownify emits e.g. `\n\n\n\n` for tightly packed paragraphs — collapse.
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def is_html_like(text: str) -> bool:
    """True if *text* looks like HTML and should be converted."""
    return bool(_HTML_MARKER_RE.search(text))


def html_to_markdown(text: str) -> str:
    """Convert HTML in *text* to Markdown; pass non-HTML through untouched.

    Preserves CJK and emoji via UTF-8 string operations (no byte slicing).
    ``markdownify`` itself walks the BeautifulSoup tree in Unicode.
    """
    if not is_html_like(text):
        return text

    cleaned = _DROP_BLOCKS_RE.sub("", text)
    cleaned = _HTML_COMMENT_RE.sub("", cleaned)

    converted = markdownify(
        cleaned,
        heading_style="ATX",
        bullets="-",
        code_language="",
        # Strip non-semantic anchors that markdownify would otherwise emit.
        strip=["meta", "link", "br"],
    )

    return _MULTI_BLANK_RE.sub("\n\n", converted).strip()
