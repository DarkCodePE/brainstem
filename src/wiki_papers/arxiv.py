"""arXiv acquisition leg — ID parsing, Atom metadata fetch, cached PDF download.

Front door of the ``wiki_papers`` bounded context (PRD-015 FR-1/FR-2, ADR-032
D2/D3). Everything an untrusted user types arrives here, so this module is
deliberately paranoid, mirroring ``wiki_repos.fetcher``:

- :func:`parse_arxiv_id` — strict shape validation; accepts only modern arXiv
  IDs (``2605.23904``, optionally ``v2``) bare or inside an
  ``arxiv.org``/``export.arxiv.org`` ``/abs/`` or ``/pdf/`` URL.
- :func:`fetch_arxiv` — Atom API metadata via ``export.arxiv.org`` with the
  arXiv-recommended 3-second inter-request rate limit (module-level, shared
  with the PDF download).
- :func:`download_pdf` — retry/backoff download, cache keyed by *versionless*
  arXiv ID, 25 MB hard cap (PRD-015 ingest limit).

SR-1: outbound hosts are limited to :data:`ALLOWED_HOSTS` — the allowlist *is*
the SSRF guard (recorded as the ADR-032 D3 amendment to the ADR-017 rule).

All network I/O is injectable (``fetch`` / ``fetch_bytes``) so the full surface
is unit-testable with zero network access (mock-first per CLAUDE.md).
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import httpx

__all__ = [
    "ALLOWED_HOSTS",
    "MAX_PDF_BYTES",
    "RATE_LIMIT_SECONDS",
    "ArxivError",
    "PdfOversize",
    "PaperMeta",
    "parse_arxiv_id",
    "fetch_arxiv",
    "download_pdf",
]

ALLOWED_HOSTS: tuple[str, ...] = ("arxiv.org", "export.arxiv.org", "www.arxiv.org")
"""Hosts we will fetch from (ADR-032 D3 / PRD-015 SR-1). Adding a host is a
security decision recorded in an ADR amendment, never a convenience."""

API_URL = "https://export.arxiv.org/api/query"
RATE_LIMIT_SECONDS = 3.0
"""arXiv's recommended inter-request delay (PRD-015 FR-1, course-validated)."""

MAX_PDF_BYTES = 25 * 1024 * 1024
"""PDF size cap — reuses the 25 MB ingest limit (PRD-015 FR-2 / SR-3)."""

DOWNLOAD_MAX_RETRIES = 3
DOWNLOAD_RETRY_DELAY_BASE = 2.0
_USER_AGENT = "second-brain-wiki/wiki_papers (+https://github.com/DarkCodePE/second-brain-wiki)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# Modern arXiv identifier: YYMM.NNNNN with an optional explicit version.
_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(?:v(\d+))?$")

# Injectable network type aliases (documented for callers/tests).
TextFetcher = Callable[[str], str]
BytesFetcher = Callable[[str], bytes]

# Module-level rate limiter state. ``_monotonic``/``_sleep`` are indirections
# so tests can monkeypatch the clock without touching ``time`` globally.
_monotonic = time.monotonic
_sleep = time.sleep
_LAST_REQUEST_AT: float | None = None


class ArxivError(RuntimeError):
    """arXiv acquisition failure (API error, paper not found, download failed)."""

    kind = "ArxivError"


class PdfOversize(ArxivError):  # noqa: N818 — short typed-error names, wiki_repos house style
    """The PDF exceeds :data:`MAX_PDF_BYTES`. Never retried."""

    kind = "PdfOversize"


@dataclass(frozen=True, slots=True)
class PaperMeta:
    """Authoritative paper metadata from the arXiv Atom API (PRD-015 FR-1).

    ``arxiv_id`` is always *versionless* — it is the dedup ``source_key``
    (PRD-015 FR-5); the version travels separately in ``version``.
    """

    arxiv_id: str
    version: str | None
    title: str
    authors: tuple[str, ...]
    abstract: str
    categories: tuple[str, ...]
    published: str
    updated: str
    pdf_url: str

    @property
    def abs_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


def parse_arxiv_id(id_or_url: str) -> tuple[str, str | None]:
    """Validate and normalise an arXiv ID or URL.

    Accepted shapes (scheme optional, host case-insensitive)::

        2605.23904
        2605.23904v2
        [https://]arxiv.org/abs/2605.23904[v2]
        [https://]arxiv.org/pdf/2605.23904[v2][.pdf]
        [https://]export.arxiv.org/{abs|pdf}/...
        [https://]www.arxiv.org/{abs|pdf}/...

    Returns:
        ``(versionless_id, version)`` — version is the bare number string
        (``"2"`` for ``v2``) or ``None`` when unversioned.

    Raises:
        ValueError: For any non-conforming input (wrong host, old-style IDs,
            unsupported paths, garbage).
    """
    if not isinstance(id_or_url, str):
        raise ValueError("arxiv id must be a string")
    raw = id_or_url.strip()
    if not raw:
        raise ValueError("empty arxiv id")

    m = _ID_RE.match(raw)
    if m:
        return m.group(1), m.group(2)

    # URL form. Normalise the scheme-less shorthand before splitting.
    candidate = raw if "://" in raw else "https://" + raw
    parts = urlsplit(candidate)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme {parts.scheme!r}")
    host = (parts.hostname or "").lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError(f"host {host!r} is not an arXiv host")

    segments = [s for s in parts.path.split("/") if s]
    if len(segments) != 2 or segments[0] not in ("abs", "pdf"):
        raise ValueError(f"unsupported arXiv path {parts.path!r}")
    tail = segments[1]
    if segments[0] == "pdf" and tail.lower().endswith(".pdf"):
        tail = tail[: -len(".pdf")]
    m = _ID_RE.match(tail)
    if not m:
        raise ValueError(f"unrecognised arXiv identifier {tail!r}")
    return m.group(1), m.group(2)


def _check_allowed_host(url: str) -> None:
    """SR-1 outbound guard: refuse any URL whose host is off the allowlist."""
    host = (urlsplit(url).hostname or "").lower()
    if urlsplit(url).scheme != "https" or host not in ALLOWED_HOSTS:
        raise ArxivError(f"outbound host {host!r} not in arXiv allowlist")


def _respect_rate_limit() -> None:
    """Sleep so consecutive arXiv requests are >= RATE_LIMIT_SECONDS apart."""
    global _LAST_REQUEST_AT
    now = _monotonic()
    if _LAST_REQUEST_AT is not None:
        wait = RATE_LIMIT_SECONDS - (now - _LAST_REQUEST_AT)
        if wait > 0:
            _sleep(wait)
    _LAST_REQUEST_AT = _monotonic()


def _default_fetch_text(url: str) -> str:
    resp = httpx.get(url, timeout=30.0, follow_redirects=True, headers={"User-Agent": _USER_AGENT})
    resp.raise_for_status()
    return resp.text


def _default_fetch_bytes(url: str) -> bytes:
    """Streamed GET with the size cap enforced *while* reading (SR-3)."""
    buf = bytearray()
    with httpx.stream(
        "GET", url, timeout=60.0, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
    ) as resp:
        resp.raise_for_status()
        for chunk in resp.iter_bytes():
            buf.extend(chunk)
            if len(buf) > MAX_PDF_BYTES:
                raise PdfOversize(f"pdf exceeds {MAX_PDF_BYTES} byte cap: {url}")
    return bytes(buf)


def _text(entry: ET.Element, path: str, *, squash: bool = True) -> str:
    elem = entry.find(path, _ATOM_NS)
    if elem is None or elem.text is None:
        return ""
    text = elem.text.strip()
    return " ".join(text.split()) if squash else text


def _parse_entry(entry: ET.Element) -> PaperMeta:
    id_text = _text(entry, "atom:id")
    tail = id_text.split("/")[-1] if id_text else ""
    m = _ID_RE.match(tail)
    if not m:
        raise ArxivError(f"unparseable entry id {id_text!r}")
    arxiv_id, version = m.group(1), m.group(2)

    authors = tuple(
        name
        for author in entry.findall("atom:author", _ATOM_NS)
        if (name := _text(author, "atom:name"))
    )
    categories = tuple(
        term for cat in entry.findall("atom:category", _ATOM_NS) if (term := cat.get("term"))
    )
    pdf_url = ""
    for link in entry.findall("atom:link", _ATOM_NS):
        if link.get("type") == "application/pdf" or link.get("title") == "pdf":
            pdf_url = link.get("href", "")
            break
    if pdf_url.startswith("http://"):
        pdf_url = "https://" + pdf_url[len("http://") :]
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{tail}"

    return PaperMeta(
        arxiv_id=arxiv_id,
        version=version,
        title=_text(entry, "atom:title"),
        authors=authors,
        abstract=_text(entry, "atom:summary"),
        categories=categories,
        published=_text(entry, "atom:published"),
        updated=_text(entry, "atom:updated"),
        pdf_url=pdf_url,
    )


def fetch_arxiv(id_or_url: str, *, fetch: TextFetcher | None = None) -> PaperMeta:
    """Fetch authoritative metadata for one paper from the arXiv Atom API.

    Args:
        id_or_url: arXiv ID or URL in any :func:`parse_arxiv_id` shape.
        fetch: Injectable ``(url) -> response_text``. Defaults to an httpx GET.
            Tests MUST inject a fake here — no real network.

    Raises:
        ValueError: If the input is not a valid arXiv ID/URL.
        ArxivError: If the API is unreachable, returns garbage, or the paper
            does not exist.
    """
    arxiv_id, version = parse_arxiv_id(id_or_url)
    query_id = f"{arxiv_id}v{version}" if version else arxiv_id
    url = f"{API_URL}?{urlencode({'id_list': query_id, 'max_results': 1})}"
    _check_allowed_host(url)

    _respect_rate_limit()
    try:
        xml_text = (fetch or _default_fetch_text)(url)
    except ArxivError:
        raise
    except Exception as exc:
        raise ArxivError(f"arXiv API request failed for {arxiv_id}: {type(exc).__name__}") from exc

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivError(f"unparseable arXiv API response for {arxiv_id}") from exc

    entries = root.findall("atom:entry", _ATOM_NS)
    if not entries:
        raise ArxivError(f"paper {arxiv_id} not found on arXiv")
    meta = _parse_entry(entries[0])
    if not meta.title:
        raise ArxivError(f"paper {arxiv_id} not found on arXiv (empty entry)")
    return meta


def download_pdf(
    meta: PaperMeta,
    cache_dir: Path,
    *,
    fetch_bytes: BytesFetcher | None = None,
) -> Path:
    """Download the paper's PDF into ``cache_dir``, with retry and cache.

    The cache key is the *versionless* arXiv ID (PRD-015 FR-2): a cached
    ``<arxiv_id>.pdf`` short-circuits the download entirely.

    Args:
        meta: Paper metadata carrying the PDF URL.
        cache_dir: Directory for cached PDFs (created if missing).
        fetch_bytes: Injectable ``(url) -> bytes``. Defaults to a streamed,
            size-capped httpx GET. Tests MUST inject a fake here.

    Raises:
        PdfOversize: If the body exceeds :data:`MAX_PDF_BYTES` (no retry).
        ArxivError: On a non-arXiv URL or persistent download failure.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{meta.arxiv_id}.pdf"
    if target.exists() and target.stat().st_size > 0:
        return target

    url = meta.pdf_url or f"https://arxiv.org/pdf/{meta.arxiv_id}"
    _check_allowed_host(url)
    fetch = fetch_bytes or _default_fetch_bytes

    last_error: Exception | None = None
    for attempt in range(DOWNLOAD_MAX_RETRIES):
        _respect_rate_limit()
        try:
            data = fetch(url)
            if not data:
                raise ArxivError(f"empty pdf body from {url}")
            if len(data) > MAX_PDF_BYTES:
                raise PdfOversize(f"pdf is {len(data)} bytes (cap {MAX_PDF_BYTES})")
            tmp = target.with_suffix(".pdf.part")
            tmp.write_bytes(data)
            tmp.replace(target)
            return target
        except PdfOversize:
            raise
        except Exception as exc:  # network/HTTP/IO — retry with backoff
            last_error = exc
            if attempt < DOWNLOAD_MAX_RETRIES - 1:
                _sleep(DOWNLOAD_RETRY_DELAY_BASE * (attempt + 1))

    raise ArxivError(
        f"pdf download failed for {meta.arxiv_id} after {DOWNLOAD_MAX_RETRIES} attempts: "
        f"{type(last_error).__name__}"
    ) from last_error
