"""``WebContextSource`` — a live web page as draft context (ADR-025, non-GitHub).

The lightweight post path's NON-GitHub leg: fetch an arbitrary web page (a tool's
landing page, a launch blog, an article), convert it to markdown, and feed it to
the EXISTING ADR-024 drafter as a single ``WikiSnippet`` — no ingest, no wiki
page, draft-only.

Fetch uses the same dependency-free mechanism as ``wiki_agent.tools.web_clip``
(``httpx`` + ``markdownify``) — NOT firecrawl/Scrapling, which are not installed
in this Python env (firecrawl is Hermes's own web backend; Scrapling is vendored
but uninstalled). The fetcher is an injectable seam, so a firecrawl/Scrapling
backend can be swapped in later for JS-heavy / anti-bot sites without touching
the drafter. For GitHub repos use ``RepoContextSource`` (the GitHub API is richer
and structured); this is for everything else.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from wiki_publishing.linkedin_draft import WikiSnippet

# (url) -> raw HTML string. Injectable so tests don't hit the network.
HtmlFetcher = Callable[[str], str]

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
# Remove script/style BLOCKS (tag + inner text) before markdownify — `strip=`
# only drops the tag, leaving the inner JS/CSS text in the output.
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_MAX_MARKDOWN_CHARS = 12_000


class WebFetchError(RuntimeError):
    """The web page could not be fetched/converted (network error, non-2xx)."""

    kind = "WebFetchError"


def _default_html_fetcher(url: str) -> str:
    """Fetch ``url`` to HTML with httpx (mirrors ``web_clip``). Raises WebFetchError."""
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - httpx is a project dep
        raise WebFetchError(f"missing dependency: {exc}") from exc
    try:
        resp = httpx.get(
            url,
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "second-brain-wiki/wiki_publishing (+post-context)"},
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — surface any fetch failure uniformly
        raise WebFetchError(f"failed to fetch {url}: {type(exc).__name__}") from exc
    return resp.text


def fetch_url_markdown(
    url: str, *, html_fetcher: HtmlFetcher | None = None, max_chars: int = _MAX_MARKDOWN_CHARS
) -> tuple[str, str]:
    """Fetch ``url`` and return ``(title, markdown)``. Raises :class:`WebFetchError`."""
    if not (url or "").strip().lower().startswith(("http://", "https://")):
        raise WebFetchError(f"not an http(s) URL: {url!r}")
    html = (html_fetcher or _default_html_fetcher)(url)
    if not html:
        raise WebFetchError(f"empty response from {url}")

    title_m = _TITLE_RE.search(html)
    title = re.sub(r"\s+", " ", title_m.group(1)).strip() if title_m else url

    try:
        from markdownify import markdownify as md  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - markdownify is a project dep
        raise WebFetchError(f"missing dependency: {exc}") from exc
    clean_html = _SCRIPT_STYLE_RE.sub("", html)
    markdown = md(clean_html).strip()
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)  # collapse blank runs
    if max_chars and len(markdown) > max_chars:
        markdown = markdown[:max_chars] + "\n…[page truncated]"
    if not markdown:
        raise WebFetchError(f"no readable content at {url}")
    return title, markdown


class WebContextSource:
    """A :class:`ContentSource` backed by a single live web page (ADR-025).

    ``search`` ignores ``query``/``categories`` (the "search" already resolved to
    this one URL) and returns one :class:`WikiSnippet` whose body is the page's
    title + URL + markdown — for the ADR-024 drafter to compose a post from.
    """

    def __init__(self, url: str, *, fetcher: Callable[..., tuple[str, str]] | None = None) -> None:
        self._url = url
        self._fetcher = fetcher or fetch_url_markdown
        self._snippet: WikiSnippet | None = None

    @classmethod
    def from_url(
        cls, url: str, *, fetcher: Callable[..., tuple[str, str]] | None = None
    ) -> WebContextSource:
        return cls(url, fetcher=fetcher)

    def _load(self) -> WikiSnippet:
        if self._snippet is None:
            title, markdown = self._fetcher(self._url)
            body = f"{title}\nURL: {self._url}\n\n{markdown}"
            self._snippet = WikiSnippet(title=title, page_path=self._url, body=body)
        return self._snippet

    def search(self, query: str, *, limit: int = 3, categories=None) -> list[WikiSnippet]:
        return [self._load()]
