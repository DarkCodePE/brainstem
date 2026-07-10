"""
Whitespace normalisation stage.

Rules (applied to **unprotected** segments only — preserved spans pass
through untouched so a code block's indentation stays exact):

1. CRLF and CR line endings → LF.
2. Trailing whitespace per line stripped.
3. Runs of 3+ blank lines collapsed to 2 (so paragraph boundaries survive).
4. Tabs (outside preserved code) → 4 spaces.
5. Runs of intra-line spaces (3+) collapsed to a single space — long
   ascii-art padding that the LLM cannot use.

CJK and emoji are not touched: the regexes are codepoint-aware and the
replacement strings are plain ascii/utf-8, no byte slicing anywhere.
"""

from __future__ import annotations

import re

from wiki_compress.stages.preserve import apply_to_unprotected

_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
_TAB_RE = re.compile(r"\t")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_INTRA_SPACE_RE = re.compile(r"  +")


def normalise_whitespace(text: str) -> str:
    """Apply the rules above to *text*; idempotent."""
    return apply_to_unprotected(text, _normalise_segment)


def _normalise_segment(segment: str) -> str:
    # Line-ending normalisation first — everything else assumes LF.
    out = segment.replace("\r\n", "\n").replace("\r", "\n")
    out = _TAB_RE.sub("    ", out)
    out = _TRAILING_WS_RE.sub("", out)
    out = _MULTI_INTRA_SPACE_RE.sub(" ", out)
    out = _MULTI_BLANK_RE.sub("\n\n", out)
    return out
