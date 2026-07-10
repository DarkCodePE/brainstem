"""
Line-level dedupe stage.

The paragraph-level dedupe in ``stages.dedupe`` collapses exact-match
*paragraphs* (blank-line separated) and refuses to touch anything below
40 chars. That is the right safety floor for prose: a one-word heading
must never collapse, and short status lines often carry signal even when
they repeat ("OK", "DONE").

Some payload shapes, though, are mostly *lines*: ``git status`` output,
``ls -la`` listings, tool stdout with long, structured rows. Those have
heavy line-level duplication that the paragraph dedupe cannot reach. This
stage fills that gap with two guardrails:

1. **Configurable minimum-line-length floor.** Default 80 chars (well
   above status-line territory). The opting compressor turns it down
   when the payload kind warrants it (e.g. ``build_tool_output_pipeline``
   uses 40, since ``git status`` rows are typically 50-90 chars).
2. **Per-call opt-in.** The stage is *off by default* in the standard
   pipeline. Compressors that explicitly want it (tool-output preset)
   include it; the general-purpose default pipeline skips it.

On a hit, the second and later occurrences of an exact-line duplicate
are replaced with ``[see-line:<sha8>]`` so a downstream audit reader can
look up which earlier line the marker refers to. The first occurrence is
left intact.

Idempotency: a marker line itself is shorter than the typical floor and
hashes differently from the original anyway, so re-running the stage on
already-deduped output is a no-op.

PRD-007 R-1 / SPEC-008 OQ-1 (tool-stdout payload) motivate this stage.
"""

from __future__ import annotations

import hashlib
import re

from wiki_compress.stages.preserve import apply_to_unprotected

#: Default minimum line length to be a dedupe candidate. PRD-007 R-1
#: warns against collapsing short status lines; 80 chars is the
#: standard terminal width and well above status-line territory.
DEFAULT_MIN_LINE_LENGTH: int = 80

#: Marker shape, with a placeholder sha. Used by the idempotency
#: check so a re-run does not double-hash an existing marker.
_MARKER_RE = re.compile(r"\[see-line:[0-9a-f]{8}\]")


def _sha8(text: str) -> str:
    """8-char hex prefix of ``sha1(text)``."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _dedupe_segment(segment: str, *, min_length: int) -> str:
    seen: dict[str, str] = {}  # canonical line -> sha8
    out: list[str] = []
    for line in segment.split("\n"):
        canonical = line.rstrip()
        if len(canonical) < min_length or _MARKER_RE.fullmatch(canonical.strip()):
            out.append(line)
            continue
        if canonical in seen:
            out.append(f"[see-line:{seen[canonical]}]")
        else:
            digest = _sha8(canonical)
            seen[canonical] = digest
            out.append(line)
    return "\n".join(out)


def dedupe_lines(text: str, *, min_length: int = DEFAULT_MIN_LINE_LENGTH) -> str:
    """Collapse exact-duplicate long lines to ``[see-line:<sha8>]``.

    *min_length* (default 80) is the per-call floor. Lines shorter than
    this are left alone — they may be status flags or section markers.

    The stage walks *unprotected* segments only (preserved spans, such
    as code blocks, are passed through verbatim). Idempotent.
    """
    if not text:
        return text
    return apply_to_unprotected(text, lambda seg: _dedupe_segment(seg, min_length=min_length))


def make_line_deduper(*, min_length: int = DEFAULT_MIN_LINE_LENGTH):
    """Return a pipeline-shaped ``(str) -> str`` callable with *min_length* baked in.

    The pipeline stage signature is ``Callable[[str], str]``, so any
    custom floor has to be captured before the callable is registered.
    """

    def _stage(text: str) -> str:
        return dedupe_lines(text, min_length=min_length)

    _stage.__name__ = f"line_dedupe_min{min_length}"
    return _stage


__all__ = [
    "DEFAULT_MIN_LINE_LENGTH",
    "dedupe_lines",
    "make_line_deduper",
]
