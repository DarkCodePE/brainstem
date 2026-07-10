"""
Provider-level cursor wiring — OAuth integrations + `CursorStore`.

Pins the OQ-2 resolution end-to-end: each provider reads its persisted
cursor on the next fetch and writes the new high-water mark after a
successful walk. Without a `cursor_store` wired the provider still
works (no persistence) — the M3 Sprint 1 substrate stays backwards
compatible.

Coverage matrix:

| Behaviour                                                  | Test                                |
| ---------------------------------------------------------- | ----------------------------------- |
| Gmail with cursor_store persists after fetch               | test_gmail_persists_cursor          |
| Gmail reads back cursor on second fetch                    | test_gmail_reads_cursor_on_restart  |
| Gmail without cursor_store works (no persistence)          | test_gmail_no_store_works           |
| Gmail empty walk does not persist a cursor                 | test_gmail_empty_walk_no_persist    |
| GitHub with cursor_store persists after fetch              | test_github_persists_cursor         |
| GitHub reads back cursor on second fetch                   | test_github_reads_cursor_on_restart |
| GitHub without cursor_store works (no persistence)         | test_github_no_store_works          |
| Two providers don't collide on cursor key                  | test_isolated_cursor_keys           |
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from wiki_integrations.cursor_store import CursorStore
from wiki_integrations.providers.github import GitHubIntegrationSource
from wiki_integrations.providers.gmail import GmailIntegrationSource

# Shared minimal payloads — copies kept local so the test file is
# self-contained (the heavy provider suites already exercise the
# full translation matrix).
_GMAIL_PAYLOAD_1 = {
    "id": "msg-001",
    "thread_id": "thread-001",
    "from": "alice@example.com",
    "subject": "First",
    "snippet": "snippet",
    "body": "Body one.",
    "internal_date": "2026-05-22T08:00:00Z",
}
_GMAIL_PAYLOAD_2 = {
    "id": "msg-002",
    "thread_id": "thread-002",
    "from": "bob@example.com",
    "subject": "Second",
    "snippet": "snippet2",
    "body": "Body two.",
    "internal_date": "2026-05-22T10:00:00Z",
}

_GH_PAYLOAD_1 = {
    "id": "gh-001",
    "kind": "issue",
    "number": 1,
    "title": "First issue",
    "body": "Body one.",
    "html_url": "https://github.com/example/repo/issues/1",
    "updated_at": "2026-05-22T08:00:00Z",
    "state": "open",
}
_GH_PAYLOAD_2 = {
    "id": "gh-002",
    "kind": "issue",
    "number": 2,
    "title": "Second issue",
    "body": "Body two.",
    "html_url": "https://github.com/example/repo/issues/2",
    "updated_at": "2026-05-22T10:00:00Z",
    "state": "open",
}


@pytest.fixture
def cursor_db(tmp_path: Path) -> Path:
    return tmp_path / "cursor.db"


class TestGmailCursorWiring:
    @pytest.mark.asyncio
    async def test_gmail_persists_cursor(
        self,
        cursor_db: Path,
        fake_walker_factory,
        recording_callback,
    ) -> None:
        store = CursorStore(cursor_db)
        await store.init()
        try:
            walker = fake_walker_factory({"gmail": [_GMAIL_PAYLOAD_1, _GMAIL_PAYLOAD_2]})
            src = GmailIntegrationSource(
                on_event=recording_callback,
                walker=walker,
                fetch_window=timedelta(hours=24),
                cursor_store=store,
            )
            events = await src.fetch_batch()
            assert len(events) == 2
            # New high-water cursor is the latest internal_date in the batch.
            assert await store.get("gmail") == _GMAIL_PAYLOAD_2["internal_date"]
            # And the provider exposes it on the public property.
            assert src.cursor == _GMAIL_PAYLOAD_2["internal_date"]
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_gmail_reads_cursor_on_restart(
        self,
        cursor_db: Path,
        fake_walker_factory,
        recording_callback,
    ) -> None:
        """A second source pointed at the same store sees the cursor
        previously written — that's the "daemon restart" scenario.
        """
        store = CursorStore(cursor_db)
        await store.init()
        try:
            # First instance writes the cursor.
            walker_1 = fake_walker_factory({"gmail": [_GMAIL_PAYLOAD_1]})
            src_1 = GmailIntegrationSource(
                on_event=recording_callback,
                walker=walker_1,
                cursor_store=store,
            )
            await src_1.fetch_batch()

            # Fresh instance, same store: cursor must be readable.
            walker_2 = fake_walker_factory({"gmail": []})
            src_2 = GmailIntegrationSource(
                on_event=recording_callback,
                walker=walker_2,
                cursor_store=store,
            )
            # Trigger the lazy read by running an empty fetch.
            await src_2.fetch_batch()
            assert src_2.cursor == _GMAIL_PAYLOAD_1["internal_date"]
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_gmail_no_store_works(
        self,
        fake_walker_factory,
        recording_callback,
    ) -> None:
        """Without a `cursor_store`, the provider still emits events —
        backwards compat with the M3 Sprint 1 surface.
        """
        walker = fake_walker_factory({"gmail": [_GMAIL_PAYLOAD_1, _GMAIL_PAYLOAD_2]})
        src = GmailIntegrationSource(
            on_event=recording_callback,
            walker=walker,
        )
        events = await src.fetch_batch()
        assert len(events) == 2
        assert src.cursor is None

    @pytest.mark.asyncio
    async def test_gmail_empty_walk_no_persist(
        self,
        cursor_db: Path,
        fake_walker_factory,
        recording_callback,
    ) -> None:
        """An empty walk must not overwrite a previously-set cursor."""
        store = CursorStore(cursor_db)
        await store.init()
        try:
            await store.set("gmail", "previous-cursor")
            walker = fake_walker_factory({"gmail": []})
            src = GmailIntegrationSource(
                on_event=recording_callback,
                walker=walker,
                cursor_store=store,
            )
            events = await src.fetch_batch()
            assert events == []
            # Previous cursor is preserved.
            assert await store.get("gmail") == "previous-cursor"
        finally:
            await store.close()


class TestGithubCursorWiring:
    @pytest.mark.asyncio
    async def test_github_persists_cursor(
        self,
        cursor_db: Path,
        fake_walker_factory,
        recording_callback,
    ) -> None:
        store = CursorStore(cursor_db)
        await store.init()
        try:
            walker = fake_walker_factory({"github": [_GH_PAYLOAD_1, _GH_PAYLOAD_2]})
            src = GitHubIntegrationSource(
                on_event=recording_callback,
                walker=walker,
                fetch_window=timedelta(hours=24),
                cursor_store=store,
            )
            events = await src.fetch_batch()
            assert len(events) == 2
            assert await store.get("github") == _GH_PAYLOAD_2["updated_at"]
            assert src.cursor == _GH_PAYLOAD_2["updated_at"]
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_github_reads_cursor_on_restart(
        self,
        cursor_db: Path,
        fake_walker_factory,
        recording_callback,
    ) -> None:
        store = CursorStore(cursor_db)
        await store.init()
        try:
            walker_1 = fake_walker_factory({"github": [_GH_PAYLOAD_1]})
            src_1 = GitHubIntegrationSource(
                on_event=recording_callback,
                walker=walker_1,
                cursor_store=store,
            )
            await src_1.fetch_batch()

            walker_2 = fake_walker_factory({"github": []})
            src_2 = GitHubIntegrationSource(
                on_event=recording_callback,
                walker=walker_2,
                cursor_store=store,
            )
            await src_2.fetch_batch()
            assert src_2.cursor == _GH_PAYLOAD_1["updated_at"]
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_github_no_store_works(
        self,
        fake_walker_factory,
        recording_callback,
    ) -> None:
        walker = fake_walker_factory({"github": [_GH_PAYLOAD_1]})
        src = GitHubIntegrationSource(
            on_event=recording_callback,
            walker=walker,
        )
        events = await src.fetch_batch()
        assert len(events) == 1
        assert src.cursor is None


class TestCrossProviderIsolation:
    @pytest.mark.asyncio
    async def test_isolated_cursor_keys(
        self,
        cursor_db: Path,
        fake_walker_factory,
        recording_callback,
    ) -> None:
        """Gmail and GitHub must each see only their own cursor."""
        store = CursorStore(cursor_db)
        await store.init()
        try:
            gmail = GmailIntegrationSource(
                on_event=recording_callback,
                walker=fake_walker_factory({"gmail": [_GMAIL_PAYLOAD_1]}),
                cursor_store=store,
            )
            github = GitHubIntegrationSource(
                on_event=recording_callback,
                walker=fake_walker_factory({"github": [_GH_PAYLOAD_1]}),
                cursor_store=store,
            )
            await gmail.fetch_batch()
            await github.fetch_batch()
            assert await store.get("gmail") == _GMAIL_PAYLOAD_1["internal_date"]
            assert await store.get("github") == _GH_PAYLOAD_1["updated_at"]
            # Clearing one source must not touch the other.
            await store.clear("gmail")
            assert await store.get("gmail") is None
            assert await store.get("github") == _GH_PAYLOAD_1["updated_at"]
        finally:
            await store.close()
