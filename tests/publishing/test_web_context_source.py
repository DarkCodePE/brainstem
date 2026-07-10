"""Tests for WebContextSource (ADR-025 non-GitHub leg).

Unit tests inject an HTML fetcher (to test the title/markdown/snippet COMPOSITION
deterministically); a LIVE test (marked `network`) hits a real URL to prove the
real httpx+markdownify fetch actually works (no false-green — per Orlando).
"""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from wiki_publishing import WebContextSource, WebFetchError, fetch_url_markdown

_HTML = """
<html><head><title>  Turbovec — vector search  </title></head>
<body><h1>Turbovec</h1><p>10M vectors in 4GB. Faster than FAISS.</p>
<script>ignore()</script></body></html>
"""


def test_fetch_url_markdown_extracts_title_and_strips_scripts():
    title, md = fetch_url_markdown("https://example.com/turbovec", html_fetcher=lambda u: _HTML)
    assert title == "Turbovec — vector search"  # collapsed whitespace
    assert "Turbovec" in md and "10M vectors" in md
    assert "ignore()" not in md  # <script> stripped


def test_fetch_rejects_non_http():
    with pytest.raises(WebFetchError):
        fetch_url_markdown("ftp://x/y", html_fetcher=lambda u: _HTML)


def test_websource_returns_one_snippet_with_url_for_citation():
    src = WebContextSource.from_url(
        "https://example.com/turbovec", fetcher=lambda u: ("Turbovec", "10M vectors in 4GB.")
    )
    snips = src.search("anything", limit=3, categories=("whatever",))
    assert len(snips) == 1
    s = snips[0]
    assert s.page_path == "https://example.com/turbovec"  # URL → drafter cites it
    assert "Turbovec" in s.body and "10M vectors" in s.body
    assert "https://example.com/turbovec" in s.body


def test_websource_caches_fetch():
    calls = {"n": 0}

    def fetcher(u):
        calls["n"] += 1
        return ("T", "body")

    src = WebContextSource.from_url("https://example.com", fetcher=fetcher)
    src.search("a")
    src.search("b")
    assert calls["n"] == 1  # fetched once, cached


@pytest.mark.network
def test_fetch_url_markdown_live():
    """Real httpx+markdownify fetch of a stable page (skips offline)."""
    try:
        urllib.request.urlopen("https://example.com", timeout=5)  # noqa: S310
    except urllib.error.HTTPError:
        pass
    except Exception:
        pytest.skip("no network — live web fetch skipped")
    title, md = fetch_url_markdown("https://example.com")
    assert md.strip()
    assert "example" in (title + md).lower()
