"""
Tests for `GoogleDriveIntegration`: metadata-only contract, mime filter
(Docs + Sheets, PDFs deferred), q-param search with single-quote
escape, scope policy lock (drive.metadata.readonly default, drive.readonly
NEVER granted), per-file sha256 idempotency.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from wiki_integrations.agent_tools.drive import (
    _DOCS_AND_SHEETS_MIMES,
    GoogleDriveIntegration,
    _compose_metadata_snippet,
    _escape_q_value,
    _extract_files,
    _extract_single_file,
    _humanise_size,
    _parse_iso,
)

from .conftest import FakeBridge

_DOC_MIME = "application/vnd.google-apps.document"
_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
_PDF_MIME = "application/pdf"


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _now() -> datetime:
    return datetime.now(UTC)


def _build_files(now: datetime | None = None) -> list[dict[str, Any]]:
    """Mixed payload: docs, sheets, plus a PDF that must be filtered out."""
    now = now or _now()
    return [
        {
            "id": "doc_one",
            "name": "Q3 plan",
            "mimeType": _DOC_MIME,
            "modifiedTime": _iso(now - timedelta(hours=2)),
            "createdTime": _iso(now - timedelta(days=10)),
            "parents": ["folder_a"],
            "webViewLink": "https://docs.google.com/document/d/doc_one/edit",
            "owners": [{"emailAddress": "alice@example.com"}],
        },
        {
            "id": "sheet_one",
            "name": "Budget 2026",
            "mimeType": _SHEET_MIME,
            "modifiedTime": _iso(now - timedelta(days=1)),
            "createdTime": _iso(now - timedelta(days=30)),
            "parents": ["folder_a"],
            "webViewLink": "https://docs.google.com/spreadsheets/d/sheet_one/edit",
            "owners": [{"emailAddress": "alice@example.com"}],
        },
        {
            # Deferred per AC — must be filtered out by list / search.
            "id": "pdf_one",
            "name": "Contract.pdf",
            "mimeType": _PDF_MIME,
            "size": "10485760",
            "modifiedTime": _iso(now - timedelta(hours=5)),
            "createdTime": _iso(now - timedelta(days=2)),
            "parents": ["folder_a"],
            "webViewLink": "https://drive.google.com/file/d/pdf_one/view",
        },
        {
            "id": "doc_old",
            "name": "Old notes",
            "mimeType": _DOC_MIME,
            "modifiedTime": _iso(now - timedelta(days=60)),
            "createdTime": _iso(now - timedelta(days=180)),
            "parents": ["folder_z"],
        },
    ]


def _make(
    secret_store,
    audit_jsonl,
    audit_md_drive,
    *,
    execute_responses: dict[tuple[str, str], dict[str, Any]] | None = None,
    files: list[dict[str, Any]] | None = None,
) -> tuple[GoogleDriveIntegration, FakeBridge]:
    """Build a GoogleDriveIntegration wired to a FakeBridge.

    If ``files`` is provided, registers it as the response for the LIST
    tool. ``execute_responses`` lets a test override per-tool responses
    fully (used for the GET tests)."""
    if execute_responses is None:
        execute_responses = {}
    if files is not None:
        execute_responses = {
            **execute_responses,
            ("drive", "GOOGLEDRIVE_LIST_FILES"): {"files": files},
        }
    bridge = FakeBridge(execute_responses=execute_responses)
    integration = GoogleDriveIntegration(
        bridge=bridge,
        store=secret_store,
        audit_jsonl=audit_jsonl,
        audit_md=audit_md_drive,
    )
    return integration, bridge


# --------------------------------------------------------------------------- #
# list — happy path + mime filter (the #35 AC headline)                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_list_happy_path(secret_store, audit_jsonl, audit_md_drive):
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    await drive.connect()
    items = await drive.list(limit=20)
    ids = {i.id for i in items}
    assert "doc_one" in ids
    assert "sheet_one" in ids


@pytest.mark.asyncio
async def test_list_excludes_pdfs(secret_store, audit_jsonl, audit_md_drive):
    """Mime filter is the locked contract: PDFs are deferred per #35 AC.

    A future PR that wants to add PDFs has to widen
    ``_DOCS_AND_SHEETS_MIMES`` AND update this test. Until then, PDFs
    must never surface from ``list``."""
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    await drive.connect()
    items = await drive.list(limit=50)
    ids = {i.id for i in items}
    assert "pdf_one" not in ids
    # The constant itself must not include PDFs — locked contract.
    assert _PDF_MIME not in _DOCS_AND_SHEETS_MIMES


@pytest.mark.asyncio
async def test_list_since_filter(secret_store, audit_jsonl, audit_md_drive):
    """``since`` drops items older than the cutoff."""
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    await drive.connect()
    # doc_old is 60 days back — should disappear when since = 7 days back.
    items = await drive.list(since=_now() - timedelta(days=7), limit=50)
    ids = {i.id for i in items}
    assert "doc_old" not in ids
    assert "doc_one" in ids


@pytest.mark.asyncio
async def test_list_limit_cap(secret_store, audit_jsonl, audit_md_drive):
    """``limit`` caps the result count even with many matching mime rows."""
    files = [
        {
            "id": f"doc_{i}",
            "name": f"Doc {i}",
            "mimeType": _DOC_MIME,
            "modifiedTime": _iso(_now() - timedelta(hours=i)),
        }
        for i in range(10)
    ]
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=files)
    await drive.connect()
    items = await drive.list(limit=3)
    assert len(items) == 3


@pytest.mark.asyncio
async def test_list_passes_page_size_to_bridge(secret_store, audit_jsonl, audit_md_drive):
    """The bridge.execute call must pass page_size (capped at 100)."""
    drive, bridge = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    await drive.connect()
    await drive.list(limit=200)
    # Exactly one execute call for the list tool.
    list_calls = [c for c in bridge.execute_calls if c[1] == "GOOGLEDRIVE_LIST_FILES"]
    assert len(list_calls) == 1
    _, _, args = list_calls[0]
    # limit=200 → page_size capped at 100.
    assert args["page_size"] == 100


# --------------------------------------------------------------------------- #
# search — q-param + single-quote escape (the security ask)                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_search_uses_q_param(secret_store, audit_jsonl, audit_md_drive):
    """Search must call the LIST tool with a ``q`` argument shaped as
    ``name contains '<value>'``."""
    drive, bridge = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    await drive.connect()
    await drive.search("budget", limit=10)
    list_calls = [c for c in bridge.execute_calls if c[1] == "GOOGLEDRIVE_LIST_FILES"]
    assert len(list_calls) == 1
    _, _, args = list_calls[0]
    assert "q" in args
    assert args["q"] == "name contains 'budget'"


@pytest.mark.asyncio
async def test_search_escapes_single_quotes(secret_store, audit_jsonl, audit_md_drive):
    """A user-controlled string with a ``'`` MUST be backslash-escaped
    so an attacker can't inject extra q clauses (e.g. ``'; trashed = true``)."""
    drive, bridge = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    await drive.connect()
    await drive.search("alice's plan", limit=10)
    list_calls = [c for c in bridge.execute_calls if c[1] == "GOOGLEDRIVE_LIST_FILES"]
    _, _, args = list_calls[-1]
    # The quote inside the user's value is escaped, not closed.
    assert args["q"] == "name contains 'alice\\'s plan'"
    # The outer single quotes that wrap the literal must still be a
    # balanced pair around the escaped value.
    assert args["q"].startswith("name contains '")
    assert args["q"].endswith("'")


@pytest.mark.asyncio
async def test_search_empty_query_rejected(secret_store, audit_jsonl, audit_md_drive):
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    await drive.connect()
    with pytest.raises(ValueError, match="non-empty"):
        await drive.search("")


@pytest.mark.asyncio
async def test_search_applies_mime_filter(secret_store, audit_jsonl, audit_md_drive):
    """Search returns only Docs + Sheets — PDFs filtered out even if the
    upstream search returns them (the LIST tool is unaware of SBW's
    contract; the filter is enforced client-side)."""
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    await drive.connect()
    result = await drive.search("contract", limit=10)
    ids = {i.id for i in result.items}
    assert "pdf_one" not in ids


# --------------------------------------------------------------------------- #
# get — by file id, not-found, parser error                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_by_file_id(secret_store, audit_jsonl, audit_md_drive):
    file_row = {
        "id": "doc_target",
        "name": "Specific doc",
        "mimeType": _DOC_MIME,
        "modifiedTime": _iso(_now() - timedelta(hours=1)),
        "parents": ["folder_b"],
    }
    drive, bridge = _make(
        secret_store,
        audit_jsonl,
        audit_md_drive,
        execute_responses={("drive", "GOOGLEDRIVE_FIND_FILE"): file_row},
    )
    await drive.connect()
    item = await drive.get("doc_target")
    assert item.id == "doc_target"
    assert item.title == "Specific doc"
    # GET tool was called with file_id arg.
    get_calls = [c for c in bridge.execute_calls if c[1] == "GOOGLEDRIVE_FIND_FILE"]
    assert len(get_calls) == 1
    _, _, args = get_calls[0]
    assert args == {"file_id": "doc_target"}


@pytest.mark.asyncio
async def test_get_not_found_raises_key_error(secret_store, audit_jsonl, audit_md_drive):
    """An empty/mismatched response yields KeyError, not a crash."""
    drive, _ = _make(
        secret_store,
        audit_jsonl,
        audit_md_drive,
        execute_responses={("drive", "GOOGLEDRIVE_FIND_FILE"): {}},
    )
    await drive.connect()
    with pytest.raises(KeyError, match="not found"):
        await drive.get("does_not_exist")


@pytest.mark.asyncio
async def test_get_empty_id_raises(secret_store, audit_jsonl, audit_md_drive):
    """An empty file id is rejected before we hit the bridge."""
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    await drive.connect()
    with pytest.raises(KeyError, match="non-empty"):
        await drive.get("")


# --------------------------------------------------------------------------- #
# Scope lock — the lock-in test                                               #
# --------------------------------------------------------------------------- #


def test_drive_scope_is_metadata_only(secret_store, audit_jsonl, audit_md_drive):
    """ADR-017 §scope table locks Drive to metadata-only by default.

    ``drive.readonly`` is listed as ``opt_in_extra`` in the locked scope
    policy but **must never appear** in the default set this integration
    requests. A regression that surfaces body-fetch without the consent
    UI (#44 follow-up) would fail this test."""
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    # Default scope must be metadata-only.
    assert "https://www.googleapis.com/auth/drive.metadata.readonly" in drive.scopes
    # Body-fetch scope must NEVER be granted by default — it's gated
    # behind the deferred #44 Settings UI consent dialog.
    assert "https://www.googleapis.com/auth/drive.readonly" not in drive.scopes


# --------------------------------------------------------------------------- #
# _to_item — metadata snippet has no body content                             #
# --------------------------------------------------------------------------- #


def test_to_item_builds_metadata_snippet(secret_store, audit_jsonl, audit_md_drive):
    """Snippet is mime + size + path; NEVER body content."""
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    raw = {
        "id": "doc_x",
        "name": "Sample",
        "mimeType": _DOC_MIME,
        "modifiedTime": _iso(_now()),
        "parents": ["folder_a", "folder_b"],
    }
    item = drive._to_item(raw)
    assert item.id == "doc_x"
    # Snippet describes the file but doesn't contain body text.
    assert "document" in item.snippet
    # Body content would be a sentence; ours is a metadata description.
    assert "Lorem" not in item.snippet
    # Parents end up in metadata.
    assert item.metadata["parents"] == ["folder_a", "folder_b"]


def test_to_item_per_file_sha256_idempotency(secret_store, audit_jsonl, audit_md_drive):
    """sha256 = file_id + modified_time. Same file with new modified_time
    yields a DIFFERENT sha (so the MemoryStore picks up updates)."""
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    base = {
        "id": "doc_y",
        "name": "y",
        "mimeType": _DOC_MIME,
        "modifiedTime": "2026-05-26T12:00:00Z",
    }
    item_v1 = drive._to_item(dict(base))
    item_v2 = drive._to_item({**base, "modifiedTime": "2026-05-27T12:00:00Z"})
    assert item_v1.metadata["sha256"] != item_v2.metadata["sha256"]
    # Idempotent for the same input.
    item_v1_again = drive._to_item(dict(base))
    assert item_v1.metadata["sha256"] == item_v1_again.metadata["sha256"]
    # All shas are 64-char hex.
    assert len(item_v1.metadata["sha256"]) == 64


def test_to_item_uri_falls_back_to_drive_url(secret_store, audit_jsonl, audit_md_drive):
    """A file without webViewLink still has a usable URI."""
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    raw = {
        "id": "doc_no_link",
        "name": "n",
        "mimeType": _DOC_MIME,
        "modifiedTime": _iso(_now()),
    }
    item = drive._to_item(raw)
    assert item.uri == "https://drive.google.com/file/d/doc_no_link"


def test_to_item_raises_on_missing_id(secret_store, audit_jsonl, audit_md_drive):
    drive, _ = _make(secret_store, audit_jsonl, audit_md_drive, files=_build_files())
    with pytest.raises(KeyError, match="id"):
        drive._to_item({"name": "no id"})


# --------------------------------------------------------------------------- #
# Module-level helper unit tests                                              #
# --------------------------------------------------------------------------- #


def test_escape_q_value_handles_single_quote():
    assert _escape_q_value("alice's plan") == "alice\\'s plan"


def test_escape_q_value_passes_through_plain_text():
    assert _escape_q_value("simple") == "simple"


def test_extract_files_accepts_top_level():
    assert _extract_files({"files": [{"id": "1"}]}) == [{"id": "1"}]


def test_extract_files_accepts_nested_data():
    assert _extract_files({"data": {"files": [{"id": "1"}]}}) == [{"id": "1"}]


def test_extract_files_accepts_items_alias():
    assert _extract_files({"items": [{"id": "1"}]}) == [{"id": "1"}]


def test_extract_files_returns_empty_on_bad_shape():
    assert _extract_files({}) == []
    assert _extract_files("not a dict") == []  # type: ignore[arg-type]


def test_extract_single_file_direct_shape():
    out = _extract_single_file({"id": "x", "name": "n"}, "x")
    assert out == {"id": "x", "name": "n"}


def test_extract_single_file_nested_under_file():
    out = _extract_single_file({"file": {"id": "x"}}, "x")
    assert out == {"id": "x"}


def test_extract_single_file_returns_none_when_no_match():
    assert _extract_single_file({"id": "other"}, "x") is None


def test_parse_iso_handles_zulu():
    dt = _parse_iso("2026-05-26T15:00:00Z")
    assert dt == datetime(2026, 5, 26, 15, 0, 0, tzinfo=UTC)


def test_parse_iso_falls_back_on_garbage():
    dt = _parse_iso("not a date")
    assert dt == datetime.fromtimestamp(0, tz=UTC)


def test_humanise_size_renders_kb():
    assert _humanise_size(2048) == "2.0 KB"


def test_humanise_size_handles_none():
    assert _humanise_size(None) == ""


def test_compose_metadata_snippet_no_body_content():
    """Composed snippet has only metadata; no body excerpts."""
    snippet = _compose_metadata_snippet(
        mime_type=_DOC_MIME,
        size_bytes=2048,
        path_hint="/My Drive/Projects/foo",
        parents=["folder_a"],
    )
    assert "document" in snippet
    assert "/My Drive/Projects/foo" in snippet
    # Parents are NOT in the snippet when path_hint is present.
    assert "parents=" not in snippet
    # No body content sneaks in.
    assert "Lorem" not in snippet


def test_compose_metadata_snippet_falls_back_to_parents():
    """Without a path_hint, the parents folder ids surface."""
    snippet = _compose_metadata_snippet(
        mime_type=_SHEET_MIME,
        size_bytes=None,
        path_hint="",
        parents=["folder_a"],
    )
    assert "spreadsheet" in snippet
    assert "parents=folder_a" in snippet
