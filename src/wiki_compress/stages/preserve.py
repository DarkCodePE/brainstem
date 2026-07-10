"""
Preserve sentinel stage — protects spans (e.g. code blocks) from later stages.

Other stages are blind to content: they will dedupe, normalise whitespace,
strip "URLs". A fenced code block that happens to contain a long URL or a
repeated boilerplate header must survive intact. The `preserve` stage wraps
such spans in a sentinel marker that subsequent stages know to skip, then a
final "release" pass strips the sentinels and restores the originals.

The sentinel deliberately uses characters that do not occur in normal prose
or in markdown structural tokens: the **Private Use Area** code point
``U+E000`` (\\uE000) for the opener and ``U+E001`` for the closer, with a
short token ID in between. The token ID is a 0-padded counter so it stays
stable across runs of the same input (idempotency).

Public surface
--------------

- :func:`preserve_spans` — wrap spans matching a regex with sentinels
- :func:`release_preserved` — strip sentinels, restoring the captured text
- :func:`mask_code_blocks` — convenience: wrap fenced + indented code blocks

The two pieces together form the *envelope* the pipeline applies first
and last. Stages in between use :func:`iter_unprotected` to walk over only
the parts that are NOT inside a sentinel.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator

# U+E000 / U+E001 sit in the Unicode Private Use Area — they will not occur
# in CJK, emoji, code, prose, or markdown structural characters.
_SENTINEL_OPEN = ""
_SENTINEL_CLOSE = ""

# Regex that matches an already-wrapped span: `<id><body><id>`.
# The captured body is everything between the second opener and the trailing closer.
_WRAPPED_RE = re.compile(
    f"{_SENTINEL_OPEN}(\\d+){_SENTINEL_CLOSE}(.*?){_SENTINEL_OPEN}\\1{_SENTINEL_CLOSE}",
    re.DOTALL,
)

# Fenced code block — ``` … ``` or ~~~ … ~~~ (3+ markers, language optional).
_FENCED_CODE_RE = re.compile(
    r"(?P<fence>```+|~~~+)[^\n]*\n.*?(?P=fence)",
    re.DOTALL,
)

# Inline backtick code — `like this`. Single line, no embedded newlines.
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _open(token_id: int) -> str:
    return f"{_SENTINEL_OPEN}{token_id}{_SENTINEL_CLOSE}"


def preserve_spans(
    text: str,
    pattern: re.Pattern[str],
    *,
    start_id: int = 0,
) -> tuple[str, int]:
    """Wrap every match of *pattern* in sentinels.

    Returns the wrapped text and the next free token id. Ids are stable
    across runs of identical input → the operation is idempotent.

    A wrapped span looks like ``\\uE000<id>\\uE001<body>\\uE000<id>\\uE001``
    so subsequent stages can find both ends with a single regex
    (see :data:`_WRAPPED_RE`).
    """
    next_id = start_id

    def _wrap(match: re.Match[str]) -> str:
        nonlocal next_id
        token = _open(next_id)
        wrapped = f"{token}{match.group(0)}{token}"
        next_id += 1
        return wrapped

    return pattern.sub(_wrap, text), next_id


def mask_code_blocks(text: str, *, start_id: int = 0) -> tuple[str, int]:
    """Wrap fenced and inline code blocks with sentinels.

    Fenced blocks are wrapped first (greedy outer match) so an inline
    backtick inside a fence is not double-wrapped.
    """
    wrapped, next_id = preserve_spans(text, _FENCED_CODE_RE, start_id=start_id)
    wrapped, next_id = preserve_spans(wrapped, _INLINE_CODE_RE, start_id=next_id)
    return wrapped, next_id


def release_preserved(text: str) -> str:
    """Strip all sentinel pairs, restoring the captured body verbatim.

    Idempotent: input without sentinels passes through unchanged.
    """
    return _WRAPPED_RE.sub(lambda m: m.group(2), text)


def has_protected_spans(text: str) -> bool:
    """True if *text* contains at least one preserved span."""
    return _SENTINEL_OPEN in text


def iter_unprotected(text: str) -> Iterator[tuple[bool, str]]:
    """Walk *text* yielding ``(is_protected, segment)`` tuples.

    Stages that should not touch preserved content call this and only
    apply transforms to segments where ``is_protected is False``. The
    preserved segments are emitted intact, sentinels and all, so the
    rejoined output still parses with :func:`release_preserved`.
    """
    pos = 0
    for match in _WRAPPED_RE.finditer(text):
        if match.start() > pos:
            yield False, text[pos : match.start()]
        yield True, match.group(0)
        pos = match.end()
    if pos < len(text):
        yield False, text[pos:]


def apply_to_unprotected(text: str, fn: Callable[[str], str]) -> str:
    """Apply *fn* to every unprotected segment; rejoin.

    Convenience around :func:`iter_unprotected` for the common case of
    a stage that wants to transform plain text while leaving preserved
    spans untouched.
    """
    parts: list[str] = []
    for is_protected, segment in iter_unprotected(text):
        parts.append(segment if is_protected else fn(segment))
    return "".join(parts)
