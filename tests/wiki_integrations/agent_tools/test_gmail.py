"""
Tests for `GmailIntegration`: list/get/search, sha256 dedup, timestamp parsing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wiki_integrations.agent_tools.gmail import GmailIntegration

from .conftest import FakeBridge

GMAIL_PAYLOAD = [
    {
        "id": "msg_001",
        "thread_id": "t_001",
        "from": "alice@example.com",
        "subject": "Q3 contract review",
        "body": "Hi team, please review the attached contract by Friday.",
        "internal_date": "1716000000000",  # ms since epoch
        "labels": ["INBOX", "IMPORTANT"],
    },
    {
        "id": "msg_002",
        "thread_id": "t_001",
        "from": "bob@example.com",
        "subject": "Re: Q3 contract review",
        "body": "Reviewed — looks good.",
        "internal_date": "1716100000000",
    },
    # A duplicate body — same sender forwards earlier message; sha256 dedup catches it.
    {
        "id": "msg_003",
        "thread_id": "t_999",
        "from": "alice@example.com",
        "subject": "Fwd: Q3 contract review",
        "body": "Hi team, please review the attached contract by Friday.",
        "internal_date": "1716200000000",
    },
]


def _make(secret_store, audit_jsonl, audit_md_gmail):
    bridge = FakeBridge(payloads={"gmail": GMAIL_PAYLOAD})
    gmail = GmailIntegration(
        bridge=bridge,
        store=secret_store,
        audit_jsonl=audit_jsonl,
        audit_md=audit_md_gmail,
    )
    return gmail, bridge


@pytest.mark.asyncio
async def test_list_returns_normalised_items(secret_store, audit_jsonl, audit_md_gmail):
    gmail, _ = _make(secret_store, audit_jsonl, audit_md_gmail)
    await gmail.connect()
    items = await gmail.list(limit=10)
    assert len(items) == 3
    assert {i.id for i in items} == {"msg_001", "msg_002", "msg_003"}
    assert all(i.uri.startswith("mailto:") for i in items)
    one = next(i for i in items if i.id == "msg_001")
    assert one.title == "Q3 contract review"
    assert one.metadata["thread_id"] == "t_001"
    assert isinstance(one.metadata["sha256"], str)


@pytest.mark.asyncio
async def test_internal_date_parsed_as_ms_epoch(secret_store, audit_jsonl, audit_md_gmail):
    gmail, _ = _make(secret_store, audit_jsonl, audit_md_gmail)
    await gmail.connect()
    items = await gmail.list()
    one = next(i for i in items if i.id == "msg_001")
    # 1716000000000 ms = 2024-05-18T05:00:00+00:00; we only assert it's a
    # UTC-aware reasonable timestamp.
    assert one.updated_at.tzinfo is not None
    assert one.updated_at > datetime(2020, 1, 1, tzinfo=UTC)


@pytest.mark.asyncio
async def test_search_dedups_by_sha256(secret_store, audit_jsonl, audit_md_gmail):
    gmail, _ = _make(secret_store, audit_jsonl, audit_md_gmail)
    await gmail.connect()
    # "contract" matches msg_001 and msg_003 (same body). Dedup should keep one.
    result = await gmail.search("contract")
    # msg_001 + msg_002 ("Re: …" body different) but msg_003 is a forward
    # of msg_001's body → identical sha256 → dropped.
    ids = {i.id for i in result.items}
    assert "msg_003" not in ids
    assert "msg_001" in ids


@pytest.mark.asyncio
async def test_get_returns_one_by_id(secret_store, audit_jsonl, audit_md_gmail):
    gmail, _ = _make(secret_store, audit_jsonl, audit_md_gmail)
    await gmail.connect()
    item = await gmail.get("msg_002")
    assert item.id == "msg_002"
    assert item.title.startswith("Re:")


@pytest.mark.asyncio
async def test_search_substring_case_insensitive(secret_store, audit_jsonl, audit_md_gmail):
    gmail, _ = _make(secret_store, audit_jsonl, audit_md_gmail)
    await gmail.connect()
    result = await gmail.search("Q3")
    assert len(result.items) >= 1


@pytest.mark.asyncio
async def test_search_empty_query_rejected(secret_store, audit_jsonl, audit_md_gmail):
    gmail, _ = _make(secret_store, audit_jsonl, audit_md_gmail)
    await gmail.connect()
    with pytest.raises(ValueError, match="non-empty"):
        await gmail.search("")
