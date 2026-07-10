"""Unit tests for the ``sbw-ingest-paper`` CLI — hermetic, zero network.

arXiv acquisition is monkeypatched at the ``wiki_papers.cli`` seam; the
extraction chain is forced onto the real pypdf floor via the engine gates.
"""

from __future__ import annotations

import json

import pytest

from wiki_papers import cli as cli_mod
from wiki_papers import extract as extract_mod
from wiki_papers.arxiv import ArxivError

from .conftest import make_text_pdf


@pytest.fixture
def pypdf_only(monkeypatch):
    monkeypatch.setattr(extract_mod, "_opendataloader_available", lambda: False)
    monkeypatch.setattr(extract_mod, "_run_docling", lambda _p: None)


def _run(capsys, argv):
    code = cli_mod.main(argv)
    captured = capsys.readouterr()
    return code, captured


# ────────────────────────────── local-PDF mode ───────────────────────────────


def test_local_pdf_writes_slugified_page(tmp_path, capsys, pypdf_only):
    root = tmp_path / "kb"
    pdf = tmp_path / "My Paper (Final v2).pdf"
    pdf.write_bytes(make_text_pdf())

    code, captured = _run(capsys, [str(pdf), "--root", str(root)])
    assert code == 0

    out_path = root / "raw" / "papers" / "my-paper-final-v2.md"
    assert out_path.is_file()
    content = out_path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "type: paper" in content
    assert "metadata_source: heuristic" in content
    assert "Hello arXiv" in content

    result = json.loads(captured.out.strip())
    assert result["page_path"] == str(out_path)
    assert result["engine_used"] == "pypdf"
    assert result["extraction"] == "degraded"
    assert result["extracted_chars"] > 0


# ─────────────────────────────── arXiv mode ──────────────────────────────────


def test_arxiv_mode_fetches_downloads_extracts(
    tmp_path, capsys, sample_meta, monkeypatch, pypdf_only
):
    root = tmp_path / "kb"
    fetched: list[str] = []
    monkeypatch.setattr(cli_mod, "fetch_arxiv", lambda raw: (fetched.append(raw), sample_meta)[1])

    def fake_download(meta, cache_dir, **_kw):
        cache_dir.mkdir(parents=True, exist_ok=True)
        pdf = cache_dir / f"{meta.arxiv_id}.pdf"
        pdf.write_bytes(make_text_pdf())
        return pdf

    monkeypatch.setattr(cli_mod, "download_pdf", fake_download)

    code, captured = _run(capsys, ["2605.23904v2", "--root", str(root)])
    assert code == 0
    assert fetched == ["2605.23904v2"]

    out_path = root / "raw" / "papers" / "2605.23904.md"  # versionless filename
    assert out_path.is_file()
    content = out_path.read_text(encoding="utf-8")
    assert "metadata_source: arxiv-api" in content
    assert "Attention Is Not Enough" in content
    # PDF cached OUTSIDE raw/ (FR-2 cache; same convention as the MCP caller)
    # so the watcher/catch-up rglob never enqueues cached PDFs.
    assert (root / ".paper-cache" / "2605.23904.pdf").is_file()

    result = json.loads(captured.out.strip())
    assert result["page_path"] == str(out_path)
    assert result["engine_used"] == "pypdf"


def test_arxiv_mode_download_failure_degrades_to_metadata_page(
    tmp_path, capsys, sample_meta, monkeypatch
):
    root = tmp_path / "kb"
    monkeypatch.setattr(cli_mod, "fetch_arxiv", lambda _raw: sample_meta)

    def no_pdf(_meta, _cache, **_kw):
        raise ArxivError("network down")

    monkeypatch.setattr(cli_mod, "download_pdf", no_pdf)

    code, captured = _run(capsys, ["2605.23904", "--root", str(root)])
    assert code == 0  # degrade-first: still exit 0

    out_path = root / "raw" / "papers" / "2605.23904.md"
    assert out_path.is_file()
    content = out_path.read_text(encoding="utf-8")
    assert "extraction: unavailable" in content
    assert sample_meta.abstract in content

    result = json.loads(captured.out.strip())
    assert result["engine_used"] == "none"
    assert any("pdf-download-failed" in n for n in result["notes"])


def test_arxiv_mode_metadata_fetch_failure_is_nonzero(tmp_path, capsys, monkeypatch):
    def down(_raw):
        raise ArxivError("api unreachable")

    monkeypatch.setattr(cli_mod, "fetch_arxiv", down)
    code, captured = _run(capsys, ["2605.23904", "--root", str(tmp_path / "kb")])
    assert code == cli_mod.EXIT_FETCH_FAILED
    err = json.loads(captured.err.strip())
    assert err["error"] == "arxiv-fetch-failed"


# ─────────────────────────────── usage errors ────────────────────────────────


def test_garbage_input_is_usage_error(tmp_path, capsys):
    code, captured = _run(capsys, ["definitely not a paper", "--root", str(tmp_path)])
    assert code == cli_mod.EXIT_USAGE
    err = json.loads(captured.err.strip())
    assert err["error"] == "usage"
    assert not (tmp_path / "raw").exists()


def test_missing_pdf_path_falls_through_to_usage_error(tmp_path, capsys):
    code, _ = _run(capsys, [str(tmp_path / "ghost.pdf"), "--root", str(tmp_path)])
    assert code == cli_mod.EXIT_USAGE


def test_slugify():
    assert cli_mod._slugify("My Paper (Final v2)") == "my-paper-final-v2"
    assert cli_mod._slugify("___") == "paper"
