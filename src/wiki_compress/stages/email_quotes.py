"""
Email quote-chain collapse stage.

Real-world email threads accumulate quote indents (``> ``, ``>> ``, etc.) as
each reply nests above the previous. The visible-text content of those
chains is usually verbatim duplicate signal already living elsewhere in
the wiki (the original message it quotes), so leaving it in the LLM
context burns tokens without adding meaning.

This stage walks the unprotected segments (so fenced code blocks
containing a literal ``> `` line are left alone), finds *contiguous runs*
of quoted lines (one or more ``>`` prefix), and replaces the whole run
with a single ``[quoted:<hash8>]`` marker. The hash is the first 8 hex
chars of ``sha1(joined_quoted_text)``, so:

- The same quote chain anywhere in the payload yields the same marker.
- Distinct chains get distinct markers (collisions are astronomically
  unlikely for the 8-char prefix at the volume the wiki sees).
- Running the stage on already-collapsed text is a no-op (the marker
  body does not start with ``> ``).

Mapping side-channel
--------------------

The full ``hash8 -> quoted_text`` mapping is stored on the module-level
``LAST_RUN_QUOTE_MAP`` for the simple module-callable form, and on the
``QuoteCollapser`` instance for the stateful form (mirrors
``url_shorten.UrlShortener``). Downstream code that needs to expand the
markers (citation rendering, audit reconstruction) reads from there.

PRD-007 R-1 and SPEC-008 OQ-1 motivate this stage. SPEC-008 measured the
default pipeline's median ratio on emails at 1.0 (no compression) because
``dedupe_paragraphs`` only collapses paragraph-level *exact* duplicates
and a quote chain reads as one block.
"""

from __future__ import annotations

import hashlib
import re

from wiki_compress.stages.preserve import apply_to_unprotected

#: Detects a single quoted line: optional indentation, then one or more
#: ``>`` prefixes, then either end-of-line or a space + content. Matches
#: both ``> reply`` and ``>>> deeply nested``.
_QUOTED_LINE_RE = re.compile(r"^[ \t]*>+(?:[ \t].*)?$")

#: Used to recognise already-collapsed markers (so a second run is a
#: no-op even on text that contains them).
_QUOTE_MARKER_RE = re.compile(r"\[quoted:[0-9a-f]{8}\]")


def _hash8(text: str) -> str:
    """8-char hex prefix of ``sha1(text)`` — stable across runs."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _collapse_segment(segment: str, quote_map: dict[str, str]) -> str:
    """Walk lines of *segment*, fold contiguous quoted runs into markers.

    Updates *quote_map* in place. Preserves trailing newline state so the
    rejoined output matches the input's line-boundary count.
    """
    lines = segment.split("\n")
    out: list[str] = []
    run: list[str] = []

    def _flush() -> None:
        if not run:
            return
        joined = "\n".join(run)
        digest = _hash8(joined)
        quote_map[digest] = joined
        out.append(f"[quoted:{digest}]")
        run.clear()

    for line in lines:
        if _QUOTED_LINE_RE.match(line):
            run.append(line)
        else:
            _flush()
            out.append(line)
    _flush()
    return "\n".join(out)


class QuoteCollapser:
    """Stateful email-quote collapser.

    Mirrors the shape of ``url_shorten.UrlShortener`` — call it like a
    function, inspect ``quote_map`` afterwards. The pipeline can use the
    instance directly as a stage.
    """

    __slots__ = ("quote_map",)

    def __init__(self) -> None:
        self.quote_map: dict[str, str] = {}

    def __call__(self, text: str) -> str:
        if not text:
            return text
        return apply_to_unprotected(text, lambda seg: _collapse_segment(seg, self.quote_map))

    def reset(self) -> None:
        """Drop the accumulated mapping (useful between pipeline runs)."""
        self.quote_map.clear()


#: Module-level side channel for the simple ``collapse_email_quotes``
#: function. Refreshed on every call so test runs do not bleed state
#: across each other. Audit code that wants persistent state should hold
#: a ``QuoteCollapser`` instance instead.
LAST_RUN_QUOTE_MAP: dict[str, str] = {}


def collapse_email_quotes(text: str) -> str:
    """Collapse contiguous email quote-line runs in *text*.

    The module-level :data:`LAST_RUN_QUOTE_MAP` is replaced with the
    mapping built during this call. Idempotent — already-collapsed markers
    are not re-hashed because the marker body itself does not start with
    ``>``.
    """
    LAST_RUN_QUOTE_MAP.clear()
    if not text:
        return text
    return apply_to_unprotected(text, lambda seg: _collapse_segment(seg, LAST_RUN_QUOTE_MAP))


__all__ = [
    "LAST_RUN_QUOTE_MAP",
    "QuoteCollapser",
    "collapse_email_quotes",
]
