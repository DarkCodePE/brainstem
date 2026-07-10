"""
Behavioural tests for the WriteSink adapters (`LocalWriteSink`,
`NullWriteSink`) and the policy enforcement helper.
"""

from __future__ import annotations

import pytest

from wiki_agent.write_sink import (
    ALLOWED_PREFIXES,
    LocalWriteSink,
    NullWriteSink,
    WriteSinkPolicyError,
    _serialise_page,
)


class TestPolicyEnforcement:
    @pytest.mark.parametrize(
        "path",
        [
            "src/x.md",
            "knowledge-base/raw/x.md",
            "../../etc/passwd",
            "/tmp/x.md",
            "wiki",  # exact prefix without slash should be refused
        ],
    )
    @pytest.mark.asyncio
    async def test_refuses_paths_outside_allowed_prefixes(self, path, page_factory) -> None:
        sink = NullWriteSink()
        page = page_factory(page_path=path)
        with pytest.raises(WriteSinkPolicyError):
            await sink.write_page(page)

    @pytest.mark.parametrize(
        "path",
        [
            "wiki/sources/x.md",
            "wiki/concepts/y.md",
            "observations/2026/note.md",
        ],
    )
    @pytest.mark.asyncio
    async def test_accepts_paths_under_allowed_prefixes(self, path, page_factory) -> None:
        sink = NullWriteSink()
        page = page_factory(page_path=path)
        await sink.write_page(page)
        assert sink.calls[0][1].ref.page_path == path

    def test_allowed_prefixes_is_a_tuple_of_strings(self) -> None:
        # The middleware introspects this; assert the shape.
        assert isinstance(ALLOWED_PREFIXES, tuple)
        assert all(isinstance(p, str) for p in ALLOWED_PREFIXES)


class TestSerialisation:
    def test_serialise_page_emits_yaml_frontmatter(self, page_factory) -> None:
        page = page_factory(title="My Page", body="Hello world.\n")
        out = _serialise_page(page)
        assert out.startswith("---\n")
        assert "title: My Page" in out
        assert "Hello world." in out

    def test_serialise_preserves_unicode(self, page_factory) -> None:
        page = page_factory(title="日本語タイトル", body="こんにちは\n")
        out = _serialise_page(page)
        assert "日本語タイトル" in out
        assert "こんにちは" in out

    def test_serialise_orders_frontmatter_keys_stably(self, page_factory) -> None:
        page = page_factory()
        out1 = _serialise_page(page)
        out2 = _serialise_page(page)
        assert out1 == out2


class TestNullSink:
    @pytest.mark.asyncio
    async def test_records_writes(self, page_factory) -> None:
        sink = NullWriteSink()
        page = page_factory()
        path = await sink.write_page(page)
        assert path.as_posix() == "wiki/sources/sample.md"
        assert len(sink.calls) == 1
        assert sink.calls[0][0] == "upsert"

    @pytest.mark.asyncio
    async def test_records_log_appends(self) -> None:
        sink = NullWriteSink()
        await sink.append_to_log("ingest: page-x written")
        assert sink.log_entries == ["ingest: page-x written"]


class TestLocalSink:
    @pytest.mark.asyncio
    async def test_calls_write_page_handler_with_serialised_page(self, page_factory) -> None:
        captured: dict[str, object] = {}

        def fake_write(page_path: str, content: str) -> str:
            captured["page_path"] = page_path
            captured["content"] = content
            return "ok"

        def fake_log(et: str, title: str, details: str) -> str:
            return "ok"

        sink = LocalWriteSink(fake_write, fake_log)
        page = page_factory(body="The body.\n")
        await sink.write_page(page)
        assert captured["page_path"] == "wiki/sources/sample.md"
        assert "The body." in captured["content"]  # type: ignore[operator]
        assert "title: Sample" in captured["content"]  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_propagates_policy_errors_before_handler_call(self, page_factory) -> None:
        called = False

        def fake_write(page_path: str, content: str) -> str:
            nonlocal called
            called = True
            return "ok"

        sink = LocalWriteSink(fake_write, lambda et, t, d: "ok")
        page = page_factory(page_path="src/escape.py")
        with pytest.raises(WriteSinkPolicyError):
            await sink.write_page(page)
        assert called is False
