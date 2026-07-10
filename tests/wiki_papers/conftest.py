"""Shared fixtures for ``wiki_papers`` tests — hermetic, zero network access.

``make_text_pdf`` hand-assembles a minimal-but-valid one-page PDF with real
text content and a correct xref table, so the pypdf floor can be exercised
against genuine bytes without any external fixture file or PDF library writer.
"""

from __future__ import annotations

import pytest

from wiki_papers import arxiv as arxiv_mod
from wiki_papers.arxiv import PaperMeta


def make_text_pdf(text: str = "Hello arXiv") -> bytes:
    """Build a valid single-page PDF whose page contains ``text``."""
    content = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objs) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        xref_pos,
    )
    return bytes(out)


@pytest.fixture
def text_pdf(tmp_path):
    """A real, parseable one-page PDF on disk containing 'Hello arXiv'."""
    path = tmp_path / "sample.pdf"
    path.write_bytes(make_text_pdf())
    return path


@pytest.fixture
def sample_meta() -> PaperMeta:
    return PaperMeta(
        arxiv_id="2605.23904",
        version="2",
        title="Attention Is Not Enough",
        authors=("Ada Lovelace", "Alan Turing"),
        abstract="We show a 2.4x speedup on LongBench.",
        categories=("cs.AI", "cs.CL"),
        published="2026-05-30T17:59:59Z",
        updated="2026-06-02T09:00:00Z",
        pdf_url="https://arxiv.org/pdf/2605.23904v2",
    )


@pytest.fixture(autouse=True)
def _quiet_rate_limit(monkeypatch):
    """Reset the module rate-limiter and neutralise real sleeping in every
    test. Tests that assert on sleeps re-patch ``_sleep`` themselves."""
    monkeypatch.setattr(arxiv_mod, "_LAST_REQUEST_AT", None)
    monkeypatch.setattr(arxiv_mod, "_sleep", lambda _s: None)
    yield
