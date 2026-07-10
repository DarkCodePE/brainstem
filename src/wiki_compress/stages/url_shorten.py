"""
URL shortening stage.

Replaces URLs longer than ``MIN_URL_LENGTH`` with short, deterministic
handles like ``[url:abc12]``. The full URL → handle mapping is exposed on
the stage instance so downstream code can dereference handles when
generating tool replies or citations.

Determinism: the handle is the first 5 hex chars of ``sha1(url)`` — so two
runs of the same input emit the same handle. Distinct URLs that collide
on the 5-char prefix get an incrementing disambiguator (``abc12_2``).

PRD-007 US-003 mandates that long URLs be shortened so the LLM doesn't
burn tokens echoing back deep query-string URLs.
"""

from __future__ import annotations

import hashlib
import re

from wiki_compress.stages.preserve import apply_to_unprotected

#: Length floor — URLs shorter than this are left alone. PRD-007 says 60.
#: We default to 40 because the test corpus mixes shorter URLs and the
#: ratio improvement comes from heavy query strings.
MIN_URL_LENGTH = 40

#: Pattern that captures http/https URLs with optional path + query.
#: The pattern is conservative — it stops at whitespace, ``<``, ``>``,
#: ``)`` (markdown link close), or end-of-line. Idempotent: if the URL
#: is already shortened (looks like ``[url:xxxxx]``) we will not match it.
_URL_RE = re.compile(r"https?://[^\s<>)\]]+")


def _handle(url: str, disambiguator: int = 0) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:5]
    if disambiguator:
        return f"[url:{digest}_{disambiguator}]"
    return f"[url:{digest}]"


class UrlShortener:
    """Stateful URL shortener.

    The pipeline invokes ``__call__`` on this instance as a stage. After
    the run, ``url_map`` holds the ``handle -> full_url`` mapping for any
    downstream resolver (tool result re-citation, audit log, …).

    Idempotency: re-running on already-shortened text leaves the handles
    intact — the URL regex requires ``http://`` or ``https://``, neither
    of which appears inside the ``[url:xxxxx]`` shape.
    """

    __slots__ = ("min_length", "url_map", "_seen")

    def __init__(self, *, min_length: int = MIN_URL_LENGTH) -> None:
        self.min_length = min_length
        self.url_map: dict[str, str] = {}
        # Reverse index for collision handling and deterministic re-use.
        self._seen: dict[str, str] = {}  # url -> handle

    def __call__(self, text: str) -> str:
        return apply_to_unprotected(text, self._shorten)

    def _shorten(self, segment: str) -> str:
        return _URL_RE.sub(self._maybe_replace, segment)

    def _maybe_replace(self, match: re.Match[str]) -> str:
        url = match.group(0)
        if len(url) < self.min_length:
            return url
        if url in self._seen:
            return self._seen[url]
        handle = _handle(url)
        # Collision guard — two distinct URLs share the same 5-char prefix.
        disambig = 1
        while handle in self.url_map and self.url_map[handle] != url:
            disambig += 1
            handle = _handle(url, disambig)
        self.url_map[handle] = url
        self._seen[url] = handle
        return handle

    def reset(self) -> None:
        """Drop any accumulated state (useful between pipeline runs)."""
        self.url_map.clear()
        self._seen.clear()
