"""
Tests for `wiki_integrations.providers.gmail.GmailIntegrationSource`.

Coverage matrix:

| Behaviour                                  | Test                                  |
| ------------------------------------------ | ------------------------------------- |
| name() == "gmail"                          | test_name_is_gmail                    |
| source field == "gmail"                    | test_event_source_is_gmail            |
| path_or_uri uses mailto:                   | test_event_path_or_uri_is_mailto      |
| sha256 matches body                        | test_event_sha256_matches_body        |
| Required metadata: bucket = gmail-inbox    | test_event_bucket_is_gmail_inbox      |
| Required metadata: rel_path                | test_event_rel_path_is_message_id     |
| Required metadata: event_type = created    | test_event_type_is_created            |
| Required metadata: mtime is internal_date  | test_event_mtime_is_internal_date     |
| Required metadata: size matches body bytes | test_event_size_matches_body          |
| Optional metadata: mime is text/plain      | test_event_mime_is_text_plain         |
| Emits one event per fetched message        | test_fetch_batch_emits_per_item       |
| Empty walk returns empty batch             | test_fetch_batch_empty                |
| Malformed payload is skipped, not raised   | test_fetch_batch_skips_malformed      |
| Walker called with provider="gmail"        | test_walker_invoked_with_gmail        |
| `wiki_core.IngestSource` shape conformance | test_source_satisfies_protocol        |
| fetch_batch returns the same list emitted  | test_batch_list_matches_callback      |
| Two messages with same body still ingested | test_distinct_events_for_same_body    |
| `started` flag round trips                 | test_lifecycle                        |
"""

from __future__ import annotations

import hashlib

import pytest

from wiki_core.protocols import IngestSource
from wiki_integrations.providers.gmail import GmailIntegrationSource

_PAYLOAD_1 = {
    "id": "msg-001",
    "thread_id": "thread-001",
    "from": "alice@example.com",
    "subject": "Sample subject",
    "snippet": "snippet",
    "body": "Body text for sha hashing.",
    "internal_date": "2026-05-22T08:00:00Z",
}
_PAYLOAD_2 = {
    "id": "msg-002",
    "thread_id": "thread-002",
    "from": "bob@example.com",
    "subject": "Another",
    "body": "Different body content.",
    "internal_date": "2026-05-22T09:30:00Z",
}


@pytest.fixture
def gmail_source(recording_callback, fake_walker_factory, fetch_window):
    walker = fake_walker_factory({"gmail": [_PAYLOAD_1, _PAYLOAD_2]})
    return GmailIntegrationSource(
        on_event=recording_callback,
        walker=walker,
        fetch_window=fetch_window,
    ), walker


def test_name_is_gmail(gmail_source) -> None:
    src, _ = gmail_source
    assert src.name() == "gmail"


@pytest.mark.asyncio
async def test_event_source_is_gmail(gmail_source) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    assert all(e.source == "gmail" for e in events)


@pytest.mark.asyncio
async def test_event_path_or_uri_is_mailto(gmail_source) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    senders = {"mailto:alice@example.com", "mailto:bob@example.com"}
    assert {e.path_or_uri for e in events} == senders


@pytest.mark.asyncio
async def test_event_sha256_matches_body(gmail_source) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    by_path = {e.path_or_uri: e for e in events}
    sha_alice = hashlib.sha256(_PAYLOAD_1["body"].encode()).hexdigest()
    sha_bob = hashlib.sha256(_PAYLOAD_2["body"].encode()).hexdigest()
    assert by_path["mailto:alice@example.com"].sha256 == sha_alice
    assert by_path["mailto:bob@example.com"].sha256 == sha_bob


@pytest.mark.asyncio
async def test_event_bucket_is_gmail_inbox(gmail_source) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    assert all(e.metadata["bucket"] == "gmail-inbox" for e in events)


@pytest.mark.asyncio
async def test_event_rel_path_is_message_id(gmail_source) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    rels = {e.metadata["rel_path"] for e in events}
    assert rels == {"msg-001", "msg-002"}


@pytest.mark.asyncio
async def test_event_type_is_created(gmail_source) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    assert all(e.metadata["event_type"] == "created" for e in events)


@pytest.mark.asyncio
async def test_event_mtime_is_internal_date(gmail_source) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    mtimes = {e.metadata["mtime"] for e in events}
    assert mtimes == {_PAYLOAD_1["internal_date"], _PAYLOAD_2["internal_date"]}


@pytest.mark.asyncio
async def test_event_size_matches_body(gmail_source) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    sizes = {e.metadata["size"] for e in events}
    expected = {len(_PAYLOAD_1["body"].encode()), len(_PAYLOAD_2["body"].encode())}
    assert sizes == expected


@pytest.mark.asyncio
async def test_event_mime_is_text_plain(gmail_source) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    assert all(e.metadata["mime"] == "text/plain" for e in events)


@pytest.mark.asyncio
async def test_fetch_batch_emits_per_item(gmail_source, recording_callback) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    assert len(events) == 2
    assert len(recording_callback.events) == 2


@pytest.mark.asyncio
async def test_fetch_batch_empty(recording_callback, fake_walker_factory, fetch_window) -> None:
    walker = fake_walker_factory({"gmail": []})
    src = GmailIntegrationSource(
        on_event=recording_callback, walker=walker, fetch_window=fetch_window
    )
    events = await src.fetch_batch()
    assert events == []
    assert recording_callback.events == []


@pytest.mark.asyncio
async def test_fetch_batch_skips_malformed(
    recording_callback, fake_walker_factory, fetch_window
) -> None:
    walker = fake_walker_factory({"gmail": [_PAYLOAD_1, {"no_id_field": True}, _PAYLOAD_2]})
    src = GmailIntegrationSource(
        on_event=recording_callback, walker=walker, fetch_window=fetch_window
    )
    events = await src.fetch_batch()
    # The malformed middle entry is logged-and-skipped; the surrounding
    # two events still flow through.
    assert len(events) == 2


@pytest.mark.asyncio
async def test_walker_invoked_with_gmail(gmail_source) -> None:
    src, walker = gmail_source
    await src.fetch_batch()
    assert walker.walked == ["gmail"]


def test_source_satisfies_protocol(gmail_source) -> None:
    src, _ = gmail_source
    assert isinstance(src, IngestSource)


@pytest.mark.asyncio
async def test_batch_list_matches_callback(gmail_source, recording_callback) -> None:
    src, _ = gmail_source
    events = await src.fetch_batch()
    # The list returned and the events captured by the callback must be
    # the same objects in the same order.
    assert recording_callback.events == events


@pytest.mark.asyncio
async def test_distinct_events_for_same_body(
    recording_callback, fake_walker_factory, fetch_window
) -> None:
    duplicate = dict(_PAYLOAD_1, id="msg-dup")
    walker = fake_walker_factory({"gmail": [_PAYLOAD_1, duplicate]})
    src = GmailIntegrationSource(
        on_event=recording_callback, walker=walker, fetch_window=fetch_window
    )
    events = await src.fetch_batch()
    # Same body → same sha. Different message ids → still two distinct
    # events. Dedup is the MemoryStore's job; the provider must emit
    # one event per upstream item.
    assert len(events) == 2
    assert events[0].sha256 == events[1].sha256
    assert events[0].event_id != events[1].event_id


@pytest.mark.asyncio
async def test_lifecycle(gmail_source) -> None:
    src, _ = gmail_source
    assert src.started is False
    await src.start()
    assert src.started is True
    await src.stop()
    assert src.started is False
