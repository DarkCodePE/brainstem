"""
Composable compression pipeline.

A `CompressionPipeline` is an ordered list of named stages. Each stage is a
plain ``(name, callable)`` pair where the callable accepts a ``str`` and
returns a ``str``. Stages that need cross-call state (e.g. the URL
shortener with its mapping table) carry it on the callable itself (see
``wiki_compress.stages.url_shorten.UrlShortener``).

Public surface
--------------

- :class:`CompressionPipeline` — the composable pipeline
- :class:`CompressionResult` — frozen result with body + metrics
- :func:`build_default_pipeline` — the M3 default stage order

Stage order (default pipeline)
------------------------------

1. ``preserve``       — mask code blocks with sentinels (set up envelope)
2. ``html_to_md``     — strip HTML tags, convert to markdown
3. ``url_shorten``    — long URLs → ``[url:xxxxx]`` (stateful)
4. ``whitespace``     — CRLF→LF, collapse extra blanks/spaces
5. ``dedupe``         — collapse exact-duplicate paragraphs
6. ``release``        — strip sentinels, restore originals

The envelope (preserve → release) protects code blocks from the
intermediate stages so a fenced code block survives the round-trip
unchanged. See ``test_preserve.py`` for the guarantee.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

from wiki_compress.metrics import CompressionMetrics, StageDelta, count_tokens
from wiki_compress.stages import (
    QuoteCollapser,
    UrlShortener,
    dedupe_paragraphs,
    html_to_markdown,
    make_line_deduper,
    mask_code_blocks,
    normalise_whitespace,
    release_preserved,
)

Stage = tuple[str, Callable[[str], str]]


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """Frozen output of a single pipeline run.

    The ``url_map`` and any other stage state can be retrieved by holding
    a reference to the stateful stage (e.g. the ``UrlShortener`` instance
    that the caller passes in or that ``build_default_pipeline`` returns).
    """

    body: str
    """The compressed text."""

    original_tokens: int
    """Token estimate of the original input (4-char heuristic)."""

    compressed_tokens: int
    """Token estimate of ``body`` after the pipeline ran."""

    ratio: float
    """``compressed / original``. < 1.0 means tokens shrank."""

    stages_applied: list[str] = field(default_factory=list)
    """Ordered stage names that ran (mirrors the pipeline definition)."""

    metrics: CompressionMetrics | None = None
    """Per-stage timing + delta. ``None`` when metrics are disabled."""


class CompressionPipeline:
    """Ordered composition of compression stages.

    Use :func:`build_default_pipeline` for the ready-to-go default, or
    construct directly with ``CompressionPipeline(stages=[...])`` to
    swap, reorder, or omit stages (e.g. dropping ``url_shorten`` for
    payloads that already have curated URLs).
    """

    __slots__ = ("stages",)

    def __init__(self, stages: list[Stage]) -> None:
        self.stages = list(stages)

    def compress(self, text: str, *, with_metrics: bool = True) -> CompressionResult:
        """Run *text* through the configured stages, return the result.

        Behaviour for an empty input is to return a zero-token,
        ``ratio=1.0`` result without invoking any stage.
        """
        original_tokens = count_tokens(text) if text else 0
        if not text:
            return CompressionResult(
                body="",
                original_tokens=0,
                compressed_tokens=0,
                ratio=1.0,
                stages_applied=[],
                metrics=CompressionMetrics(original_tokens=0) if with_metrics else None,
            )

        metrics: CompressionMetrics | None
        metrics = CompressionMetrics(original_tokens=original_tokens) if with_metrics else None

        body = text
        applied: list[str] = []
        for name, fn in self.stages:
            tokens_before = count_tokens(body)
            t0 = time.perf_counter()
            body = fn(body)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            tokens_after = count_tokens(body)
            applied.append(name)
            if metrics is not None:
                metrics.record(
                    StageDelta(
                        name=name,
                        tokens_before=tokens_before,
                        tokens_after=tokens_after,
                        elapsed_ms=elapsed_ms,
                    )
                )

        compressed_tokens = count_tokens(body) if body else 0
        ratio = (compressed_tokens / original_tokens) if original_tokens else 1.0

        return CompressionResult(
            body=body,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            ratio=ratio,
            stages_applied=applied,
            metrics=metrics,
        )


#: Default URL-length floor — long enough to leave inline links alone but
#: aggressive enough to catch tracking-heavy URLs.
_DEFAULT_MIN_URL_LENGTH: Final[int] = 40


def build_default_pipeline(
    *, url_shortener: UrlShortener | None = None
) -> tuple[CompressionPipeline, UrlShortener]:
    """Construct the M3 default pipeline + the URL shortener it uses.

    Returns a tuple of ``(pipeline, shortener)`` so the caller can inspect
    ``shortener.url_map`` after the run. Passing in an existing shortener
    re-uses its mapping table across runs (handy for batches of payloads
    that should share handles).
    """
    shortener = url_shortener or UrlShortener(min_length=_DEFAULT_MIN_URL_LENGTH)

    def _preserve(text: str) -> str:
        body, _ = mask_code_blocks(text)
        return body

    stages: list[Stage] = [
        ("preserve", _preserve),
        ("html_to_md", html_to_markdown),
        ("url_shorten", shortener),
        ("whitespace", normalise_whitespace),
        ("dedupe", dedupe_paragraphs),
        ("release", release_preserved),
    ]
    return CompressionPipeline(stages=stages), shortener


def build_email_pipeline(
    *,
    url_shortener: UrlShortener | None = None,
    quote_collapser: QuoteCollapser | None = None,
) -> tuple[CompressionPipeline, UrlShortener, QuoteCollapser]:
    """Construct an email-payload-tuned pipeline.

    Stage order:

    1. ``preserve``       — mask code blocks (envelope start).
    2. ``html_to_md``     — many mail clients send multipart with an
       HTML alternative; the upstream caller normally hands us only the
       text/plain part, but if HTML slips through we collapse it first.
    3. ``email_quotes``   — collapse ``> ``-prefixed reply chains to
       ``[quoted:<hash8>]`` markers. **Key delta vs default.**
    4. ``url_shorten``    — long URLs → ``[url:xxxxx]``.
    5. ``whitespace``     — CRLF→LF, collapse blanks.
    6. ``dedupe``         — paragraph-level dedupe (catches signature
       blocks repeated across the chain).
    7. ``release``        — strip sentinels (envelope end).

    Returns ``(pipeline, shortener, quote_collapser)`` so the caller can
    inspect both side-channel maps after the run.
    """
    shortener = url_shortener or UrlShortener(min_length=_DEFAULT_MIN_URL_LENGTH)
    collapser = quote_collapser or QuoteCollapser()

    def _preserve(text: str) -> str:
        body, _ = mask_code_blocks(text)
        return body

    stages: list[Stage] = [
        ("preserve", _preserve),
        ("html_to_md", html_to_markdown),
        ("email_quotes", collapser),
        ("url_shorten", shortener),
        ("whitespace", normalise_whitespace),
        ("dedupe", dedupe_paragraphs),
        ("release", release_preserved),
    ]
    return CompressionPipeline(stages=stages), shortener, collapser


#: Default minimum line length when line-dedupe is enabled for tool
#: output. Lower than ``DEFAULT_MIN_LINE_LENGTH`` because ``git status``
#: rows are typically 50-90 chars and we still want to collapse the
#: repeated long ones.
_TOOL_OUTPUT_LINE_FLOOR: Final[int] = 40


def build_tool_output_pipeline() -> CompressionPipeline:
    """Construct a tool-stdout-tuned pipeline.

    Tool stdout (``git status``, ``ls -la``, error tracebacks) is line
    oriented, not HTML, and benefits from line-level dedupe more than
    paragraph dedupe. This preset:

    1. ``preserve``       — mask code blocks (envelope start).
    2. ``line_dedupe``    — collapse exact-duplicate long lines.
       Floor set to 40 chars (vs the 80-char general floor) so that
       structured-row outputs like ``git status`` collapse properly.
       **Key delta vs default.**
    3. ``whitespace``     — CRLF→LF, collapse blanks.
    4. ``release``        — strip sentinels (envelope end).

    Skipped on purpose: ``html_to_md`` (stdout is not HTML — running
    markdownify on shell output would corrupt the formatting).
    """

    def _preserve(text: str) -> str:
        body, _ = mask_code_blocks(text)
        return body

    stages: list[Stage] = [
        ("preserve", _preserve),
        ("line_dedupe", make_line_deduper(min_length=_TOOL_OUTPUT_LINE_FLOOR)),
        ("whitespace", normalise_whitespace),
        ("release", release_preserved),
    ]
    return CompressionPipeline(stages=stages)
