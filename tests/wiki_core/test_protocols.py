"""
Protocol-shape tests for `wiki_core.protocols`.

Verifies that the runtime_checkable Protocols accept the adapters we ship
in M2 sprint 1 and reject objects missing required methods. These tests
are the safety net for the M2 Memory Tree bridge work — every time a
new backend is added it should pass these checks before the agent wires
it in.
"""

from __future__ import annotations

import pytest

from wiki_core.protocols import (
    IngestSource,
    MemoryStore,
    PageRef,
    Search,
    WriteSink,
)


class TestValueTypes:
    """The value types should be frozen + slotted dataclasses."""

    def test_ingest_event_is_immutable(self, event_factory) -> None:
        ev = event_factory()
        with pytest.raises((AttributeError, TypeError)):  # frozen dataclass
            ev.event_id = "tampered"  # type: ignore[misc]

    def test_page_is_immutable(self, page_factory) -> None:
        page = page_factory()
        with pytest.raises((AttributeError, TypeError)):
            page.body = "tampered"  # type: ignore[misc]

    def test_search_hit_score_in_range_is_caller_concern(self, search_hit_factory) -> None:
        # No runtime enforcement — the protocol only types this; the
        # docstring says (0..1). Test documents the contract.
        hit = search_hit_factory(score=1.5)
        assert hit.score == 1.5

    def test_page_ref_category_is_typed_literal(self) -> None:
        # The Literal is for the type checker; runtime accepts any str.
        ref = PageRef(page_path="wiki/x.md", category="sources")  # type: ignore[arg-type]
        assert ref.category == "sources"


class TestMemoryStoreProtocol:
    @pytest.mark.asyncio
    async def test_sqlite_store_satisfies_protocol(self, tmp_db_path) -> None:
        from wiki_ingest.adapter import SqliteMemoryStore

        store = SqliteMemoryStore(tmp_db_path)
        await store.init()
        try:
            assert isinstance(store, MemoryStore)
        finally:
            await store.close()

    def test_object_missing_method_fails_isinstance(self) -> None:
        class Incomplete:
            async def enqueue(self, event):  # missing other methods
                return ""

        assert not isinstance(Incomplete(), MemoryStore)


class TestWriteSinkProtocol:
    def test_null_sink_satisfies_protocol(self) -> None:
        from wiki_agent.write_sink import NullWriteSink

        sink = NullWriteSink()
        assert isinstance(sink, WriteSink)

    def test_object_missing_append_to_log_fails(self) -> None:
        class Incomplete:
            async def write_page(self, page, *, mode="upsert"):  # type: ignore[no-untyped-def]
                from pathlib import Path

                return Path(page.ref.page_path)

        assert not isinstance(Incomplete(), WriteSink)


class TestSearchProtocol:
    def test_static_adapter_satisfies_protocol(self, search_hit_factory) -> None:
        from wiki_agent.search_adapter import StaticSearchAdapter

        adapter = StaticSearchAdapter([search_hit_factory()])
        assert isinstance(adapter, Search)


class TestIngestSourceProtocol:
    def test_minimal_implementation_satisfies_protocol(self) -> None:
        class Source:
            async def start(self) -> None:
                pass

            async def stop(self) -> None:
                pass

            def name(self) -> str:
                return "test"

        assert isinstance(Source(), IngestSource)
