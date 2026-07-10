"""Unit tests for ``wiki_papers.arxiv`` — mock-first, zero network access.

Covers: ID/URL parsing variants, Atom XML parsing from a fixture string, the
3-second rate-limit logic (patched clock), the download cache short-circuit,
the 25 MB size cap, retry/backoff, and the SR-1 outbound host allowlist.
"""

from __future__ import annotations

import pytest

from wiki_papers import arxiv as arxiv_mod
from wiki_papers.arxiv import (
    ALLOWED_HOSTS,
    MAX_PDF_BYTES,
    RATE_LIMIT_SECONDS,
    ArxivError,
    PaperMeta,
    PdfOversize,
    download_pdf,
    fetch_arxiv,
    parse_arxiv_id,
)

# ───────────────────────────── parse_arxiv_id: accept ────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2605.23904", ("2605.23904", None)),
        ("2605.23904v2", ("2605.23904", "2")),
        ("2605.23904v11", ("2605.23904", "11")),
        ("1706.03762", ("1706.03762", None)),
        ("  2605.23904v2  ", ("2605.23904", "2")),
        ("https://arxiv.org/abs/2605.23904", ("2605.23904", None)),
        ("https://arxiv.org/abs/2605.23904v2", ("2605.23904", "2")),
        ("http://arxiv.org/abs/2605.23904v2", ("2605.23904", "2")),
        ("arxiv.org/abs/2605.23904", ("2605.23904", None)),
        ("www.arxiv.org/abs/2605.23904", ("2605.23904", None)),
        ("https://arxiv.org/pdf/2605.23904v2.pdf", ("2605.23904", "2")),
        ("https://arxiv.org/pdf/2605.23904.pdf", ("2605.23904", None)),
        ("https://arxiv.org/pdf/2605.23904v2", ("2605.23904", "2")),
        ("arxiv.org/pdf/2605.23904.pdf", ("2605.23904", None)),
        ("https://export.arxiv.org/abs/2605.23904", ("2605.23904", None)),
        ("export.arxiv.org/pdf/2605.23904v3.pdf", ("2605.23904", "3")),
        ("https://arxiv.org/abs/2605.23904/", ("2605.23904", None)),
    ],
)
def test_parse_accepts(raw, expected):
    assert parse_arxiv_id(raw) == expected


# ───────────────────────────── parse_arxiv_id: reject ────────────────────────


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "notanid",
        "10.1234/foo",  # DOI, not arXiv
        "cs/0501001",  # old-style arXiv ID — out of scope
        "2605.23904v",  # dangling version marker
        "26050.23904",  # wrong shape
        "2605.239",  # too few digits
        "https://evil.com/abs/2605.23904",  # non-arXiv host
        "https://arxiv.org.evil.com/abs/2605.23904",  # host-suffix trick
        "https://arxiv.org/find/2605.23904",  # unsupported path
        "https://arxiv.org/abs/2605.23904/extra",  # deep path
        "ftp://arxiv.org/abs/2605.23904",  # unsupported scheme
        "https://github.com/o/r",
    ],
)
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_arxiv_id(bad)


# ───────────────────────────── Atom XML parsing ──────────────────────────────

ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: search_query=&amp;id_list=2605.23904</title>
  <entry>
    <id>http://arxiv.org/abs/2605.23904v2</id>
    <updated>2026-06-02T09:00:00Z</updated>
    <published>2026-05-30T17:59:59Z</published>
    <title>Attention Is Not
 Enough</title>
    <summary>  We show a 2.4x speedup on LongBench.
  Across two lines.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/2605.23904v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2605.23904v2" rel="related" type="application/pdf"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query: id_list=9999.99999</title>
</feed>
"""


def test_fetch_arxiv_parses_atom_fixture():
    seen_urls: list[str] = []

    def fake_fetch(url: str) -> str:
        seen_urls.append(url)
        return ATOM_FIXTURE

    meta = fetch_arxiv("2605.23904", fetch=fake_fetch)
    assert seen_urls and seen_urls[0].startswith("https://export.arxiv.org/api/query?")
    assert "id_list=2605.23904" in seen_urls[0]
    assert meta.arxiv_id == "2605.23904"
    assert meta.version == "2"
    assert meta.title == "Attention Is Not Enough"  # newline squashed
    assert meta.authors == ("Ada Lovelace", "Alan Turing")
    assert "2.4x speedup" in meta.abstract
    assert meta.categories == ("cs.AI", "cs.CL")
    assert meta.published == "2026-05-30T17:59:59Z"
    assert meta.updated == "2026-06-02T09:00:00Z"
    assert meta.pdf_url == "https://arxiv.org/pdf/2605.23904v2"  # http → https
    assert meta.abs_url == "https://arxiv.org/abs/2605.23904"


def test_fetch_arxiv_versioned_input_queries_versioned_id():
    seen: list[str] = []
    fetch_arxiv("2605.23904v2", fetch=lambda u: (seen.append(u), ATOM_FIXTURE)[1])
    assert "id_list=2605.23904v2" in seen[0]


def test_fetch_arxiv_not_found_raises():
    with pytest.raises(ArxivError, match="not found"):
        fetch_arxiv("9999.99999", fetch=lambda _u: EMPTY_FEED)


def test_fetch_arxiv_garbage_response_raises():
    with pytest.raises(ArxivError):
        fetch_arxiv("2605.23904", fetch=lambda _u: "<html>not atom</html")


def test_fetch_arxiv_network_error_wrapped():
    def boom(_url: str) -> str:
        raise OSError("connection refused")

    with pytest.raises(ArxivError, match="request failed"):
        fetch_arxiv("2605.23904", fetch=boom)


def test_fetch_arxiv_invalid_input_is_value_error():
    with pytest.raises(ValueError):
        fetch_arxiv("not-an-id", fetch=lambda _u: ATOM_FIXTURE)


# ───────────────────────────── rate-limit logic ──────────────────────────────


def test_rate_limit_sleeps_between_consecutive_requests(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv_mod, "_monotonic", lambda: 100.0)
    monkeypatch.setattr(arxiv_mod, "_sleep", sleeps.append)
    monkeypatch.setattr(arxiv_mod, "_LAST_REQUEST_AT", None)

    fetch_arxiv("2605.23904", fetch=lambda _u: ATOM_FIXTURE)
    assert sleeps == []  # first request goes straight through

    fetch_arxiv("2605.23904", fetch=lambda _u: ATOM_FIXTURE)
    assert sleeps == [pytest.approx(RATE_LIMIT_SECONDS)]  # zero elapsed → full delay


def test_rate_limit_skips_sleep_after_enough_elapsed(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv_mod, "_sleep", sleeps.append)
    monkeypatch.setattr(arxiv_mod, "_monotonic", lambda: 200.0)
    # Last request was RATE_LIMIT_SECONDS + 1 ago → no sleep needed.
    monkeypatch.setattr(arxiv_mod, "_LAST_REQUEST_AT", 200.0 - RATE_LIMIT_SECONDS - 1.0)
    fetch_arxiv("2605.23904", fetch=lambda _u: ATOM_FIXTURE)
    assert sleeps == []


# ───────────────────────────── download_pdf ──────────────────────────────────


def test_download_pdf_cache_hit_skips_network(tmp_path, sample_meta):
    cache = tmp_path / "cache"
    cache.mkdir()
    cached = cache / "2605.23904.pdf"
    cached.write_bytes(b"%PDF-1.4 cached")

    def boom(_url: str) -> bytes:
        raise AssertionError("network must not be hit on cache hit")

    result = download_pdf(sample_meta, cache, fetch_bytes=boom)
    assert result == cached
    assert result.read_bytes() == b"%PDF-1.4 cached"


def test_download_pdf_writes_cache_keyed_by_versionless_id(tmp_path, sample_meta):
    body = b"%PDF-1.4 fresh bytes"
    result = download_pdf(sample_meta, tmp_path / "c", fetch_bytes=lambda _u: body)
    assert result.name == "2605.23904.pdf"  # versionless key, not ...v2
    assert result.read_bytes() == body


def test_download_pdf_size_cap_rejected_no_retry(tmp_path, sample_meta):
    calls: list[str] = []

    def huge(url: str) -> bytes:
        calls.append(url)
        return b"%" + b"\0" * MAX_PDF_BYTES  # cap + 1 bytes

    with pytest.raises(PdfOversize):
        download_pdf(sample_meta, tmp_path / "c", fetch_bytes=huge)
    assert len(calls) == 1  # oversize is never retried
    assert not (tmp_path / "c" / "2605.23904.pdf").exists()


def test_download_pdf_retries_then_succeeds(tmp_path, sample_meta, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv_mod, "_sleep", sleeps.append)
    attempts: list[int] = []

    def flaky(_url: str) -> bytes:
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("timeout")
        return b"%PDF-1.4 third time lucky"

    result = download_pdf(sample_meta, tmp_path / "c", fetch_bytes=flaky)
    assert len(attempts) == 3
    assert result.read_bytes().endswith(b"lucky")
    assert any(s >= arxiv_mod.DOWNLOAD_RETRY_DELAY_BASE for s in sleeps)  # backoff happened


def test_download_pdf_persistent_failure_raises(tmp_path, sample_meta):
    def always_down(_url: str) -> bytes:
        raise OSError("down")

    with pytest.raises(ArxivError, match="after 3 attempts"):
        download_pdf(sample_meta, tmp_path / "c", fetch_bytes=always_down)


def test_download_pdf_refuses_non_arxiv_host(tmp_path, sample_meta):
    evil = PaperMeta(
        **{
            **{f: getattr(sample_meta, f) for f in PaperMeta.__dataclass_fields__},
            "pdf_url": "https://evil.example.com/x.pdf",
        }
    )
    with pytest.raises(ArxivError, match="allowlist"):
        download_pdf(evil, tmp_path / "c", fetch_bytes=lambda _u: b"%PDF-")


def test_allowlist_is_arxiv_only():
    assert set(ALLOWED_HOSTS) == {"arxiv.org", "export.arxiv.org", "www.arxiv.org"}
