"""
`wiki_integrations.cursor_store.CursorStore` — SQLite-backed per-source
cursor persistence.

Pins the OQ-2 resolution from M3 Sprint 2: cursors live in a separate
value object, not in `OAuthIntegrationSource.fetch_batch`'s return type.
This file proves the store's CRUD + concurrency semantics so providers
can rely on the contract.

Coverage matrix:

| Behaviour                                  | Test                                  |
| ------------------------------------------ | ------------------------------------- |
| `init()` is idempotent                     | test_init_idempotent                  |
| `set` + `get` round-trips                  | test_set_get_round_trip               |
| `get` on unknown source returns None       | test_get_unknown_returns_none         |
| `clear` removes the entry                  | test_clear_removes_entry              |
| `clear` on unknown source is a no-op       | test_clear_unknown_is_noop            |
| `set` is overwrite (last write wins)       | test_set_overwrites                   |
| Concurrent sets serialise                  | test_concurrent_sets_serialise        |
| Empty source name raises                   | test_empty_source_name_raises         |
| Use without init raises                    | test_use_before_init_raises           |
| Close is idempotent                        | test_close_idempotent                 |
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from wiki_integrations.cursor_store import CursorStore


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "cursors.db"


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_init_idempotent(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        # A second init() should not raise.
        await store.init()
        try:
            await store.set("gmail", "abc")
            assert await store.get("gmail") == "abc"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        await store.close()
        # Second close: no-op.
        await store.close()

    @pytest.mark.asyncio
    async def test_use_before_init_raises(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        with pytest.raises(RuntimeError):
            await store.get("gmail")

    @pytest.mark.asyncio
    async def test_init_creates_parent_dir(self, tmp_path: Path) -> None:
        # Parent does not exist yet; init() should create it.
        db = tmp_path / "deep" / "nested" / "cursors.db"
        store = CursorStore(db)
        try:
            await store.init()
            assert db.parent.exists()
        finally:
            await store.close()


class TestCRUD:
    @pytest.mark.asyncio
    async def test_set_get_round_trip(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        try:
            await store.set("gmail", "history-12345")
            assert await store.get("gmail") == "history-12345"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        try:
            assert await store.get("never-set") is None
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_set_overwrites(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        try:
            await store.set("github", "since-2026-05-22")
            await store.set("github", "since-2026-05-23")
            assert await store.get("github") == "since-2026-05-23"
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_clear_removes_entry(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        try:
            await store.set("gmail", "history-99")
            await store.clear("gmail")
            assert await store.get("gmail") is None
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_clear_unknown_is_noop(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        try:
            # Clearing a source that was never set must not raise.
            await store.clear("never-set")
            assert await store.get("never-set") is None
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_distinct_sources_isolated(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        try:
            await store.set("gmail", "g-1")
            await store.set("github", "h-1")
            assert await store.get("gmail") == "g-1"
            assert await store.get("github") == "h-1"
            await store.clear("gmail")
            assert await store.get("gmail") is None
            assert await store.get("github") == "h-1"
        finally:
            await store.close()


class TestValidation:
    @pytest.mark.asyncio
    async def test_empty_source_name_get_raises(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        try:
            with pytest.raises(ValueError):
                await store.get("")
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_empty_source_name_set_raises(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        try:
            with pytest.raises(ValueError):
                await store.set("", "x")
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_none_cursor_set_raises(self, db_path: Path) -> None:
        store = CursorStore(db_path)
        await store.init()
        try:
            with pytest.raises(ValueError):
                await store.set("gmail", None)  # type: ignore[arg-type]
        finally:
            await store.close()


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_sets_serialise(self, db_path: Path) -> None:
        """Two concurrent `set` calls on the same key must not interleave
        in a way that produces a hybrid row — last write wins, no race.
        """
        store = CursorStore(db_path)
        await store.init()
        try:
            # Fire many concurrent sets with distinguishable values.
            await asyncio.gather(*(store.set("gmail", f"v-{i:03d}") for i in range(50)))
            final = await store.get("gmail")
            assert final is not None
            # Final value must be one of the values we wrote.
            assert final.startswith("v-")
            assert int(final.split("-")[1]) in range(50)
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_set_then_clear_serialise(self, db_path: Path) -> None:
        """`set` then `clear` always lands in awaited order — the lock
        prevents `clear` from sneaking in before the set commits.
        """
        store = CursorStore(db_path)
        await store.init()
        try:
            await store.set("gmail", "v-1")
            await store.clear("gmail")
            assert await store.get("gmail") is None
        finally:
            await store.close()


class TestPersistence:
    @pytest.mark.asyncio
    async def test_value_survives_close_reopen(self, db_path: Path) -> None:
        """A cursor written by store A is visible to store B opened on
        the same file — the whole point of the persistence layer.
        """
        store_a = CursorStore(db_path)
        await store_a.init()
        await store_a.set("gmail", "persisted")
        await store_a.close()

        store_b = CursorStore(db_path)
        await store_b.init()
        try:
            assert await store_b.get("gmail") == "persisted"
        finally:
            await store_b.close()
