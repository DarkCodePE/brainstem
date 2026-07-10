"""
Tests for the compression hook on ``wiki_agent.write_sink``.

Coverage matrix:

| Behaviour                                              | Test                                           |
| ------------------------------------------------------ | ---------------------------------------------- |
| Default-constructed ``LocalWriteSink`` (no compressor) | test_local_write_sink_default_unchanged        |
| ``LocalWriteSink(compressor=…)`` calls compressor      | test_local_write_sink_invokes_compressor       |
| Compressor only touches body, not frontmatter          | test_compressor_does_not_touch_frontmatter     |
| Empty input through compressor yields empty body       | test_empty_body_compresses_to_empty            |
| Compressor receives the page body, not the frontmatter | test_compressor_receives_body_only             |
| ``CompressingWriteSink`` wraps ``NullWriteSink``       | test_compressing_wrapper_around_null           |
| ``CompressingWriteSink`` wraps ``LocalWriteSink``      | test_compressing_wrapper_around_local          |
| ``CompressingWriteSink`` forwards ``mode``             | test_compressing_wrapper_forwards_mode         |
| ``CompressingWriteSink`` forwards ``append_to_log``    | test_compressing_wrapper_forwards_log          |
| Default pipeline integration shrinks body              | test_default_pipeline_shrinks_body             |
"""

from __future__ import annotations

import pytest

from wiki_agent.write_sink import (
    CompressingWriteSink,
    LocalWriteSink,
    NullWriteSink,
)
from wiki_compress import (
    CompressionResult,
    build_default_pipeline,
)
from wiki_core.protocols import Page, PageRef

# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_page() -> Page:
    body = (
        "<p>A long enough paragraph that would normally trigger HTML stripping "
        "and other transforms when handed to the default compressor.</p>\n\n"
        "<p>A long enough paragraph that would normally trigger HTML stripping "
        "and other transforms when handed to the default compressor.</p>"
    )
    return Page(
        ref=PageRef(page_path="wiki/test/sample.md", category="sources"),
        frontmatter={"title": "Sample", "tags": ["test"]},
        body=body,
    )


class _Recorder:
    """Captures (path, body) handed to a write_page handler."""

    def __init__(self) -> None:
        self.path: str | None = None
        self.body: str | None = None
        self.log: list[tuple[str, str, str]] = []

    def write_page(self, path: str, body: str) -> str:
        self.path = path
        self.body = body
        return "ok"

    def append_to_log(self, entry_type: str, title: str, details: str) -> str:
        self.log.append((entry_type, title, details))
        return "ok"


def _fake_compressor(text: str) -> CompressionResult:
    """Compressor stub that records the input and returns a known body."""
    _fake_compressor.last_input = text  # type: ignore[attr-defined]
    return CompressionResult(
        body="COMPRESSED",
        original_tokens=10,
        compressed_tokens=1,
        ratio=0.1,
    )


# --------------------------------------------------------------------------- #
# LocalWriteSink — backwards compat & opt-in compression                      #
# --------------------------------------------------------------------------- #


class TestLocalWriteSinkBackwardsCompat:
    @pytest.mark.asyncio
    async def test_local_write_sink_default_unchanged(self, sample_page: Page) -> None:
        recorder = _Recorder()
        sink = LocalWriteSink(recorder.write_page, recorder.append_to_log)
        await sink.write_page(sample_page)
        # Body must still contain the original paragraph text — no compression.
        assert recorder.body is not None
        assert "would normally trigger HTML stripping" in recorder.body


class TestLocalWriteSinkCompressorHook:
    @pytest.mark.asyncio
    async def test_local_write_sink_invokes_compressor(self, sample_page: Page) -> None:
        recorder = _Recorder()
        sink = LocalWriteSink(
            recorder.write_page,
            recorder.append_to_log,
            compressor=_fake_compressor,
        )
        await sink.write_page(sample_page)
        assert recorder.body is not None
        assert "COMPRESSED" in recorder.body
        # The compressor was called with the original body, not the frontmatter.
        assert "A long enough paragraph" in _fake_compressor.last_input  # type: ignore[attr-defined]
        assert "title: Sample" not in _fake_compressor.last_input  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_compressor_does_not_touch_frontmatter(self, sample_page: Page) -> None:
        recorder = _Recorder()
        sink = LocalWriteSink(
            recorder.write_page,
            recorder.append_to_log,
            compressor=_fake_compressor,
        )
        await sink.write_page(sample_page)
        # The serialised body still has the YAML frontmatter intact.
        assert recorder.body is not None
        assert recorder.body.startswith("---\n")
        assert "title: Sample" in recorder.body
        assert "tags:" in recorder.body

    @pytest.mark.asyncio
    async def test_empty_body_compresses_to_empty(self) -> None:
        recorder = _Recorder()
        empty_page = Page(
            ref=PageRef(page_path="wiki/test/empty.md", category="sources"),
            frontmatter={"title": "Empty"},
            body="",
        )
        sink = LocalWriteSink(
            recorder.write_page,
            recorder.append_to_log,
            compressor=build_default_pipeline()[0].compress,
        )
        await sink.write_page(empty_page)
        assert recorder.body is not None
        # Frontmatter still there; body section is empty.
        assert "title: Empty" in recorder.body

    @pytest.mark.asyncio
    async def test_compressor_receives_body_only(self, sample_page: Page) -> None:
        captured: list[str] = []

        def _capturing(text: str) -> CompressionResult:
            captured.append(text)
            return CompressionResult(body=text, original_tokens=1, compressed_tokens=1, ratio=1.0)

        recorder = _Recorder()
        sink = LocalWriteSink(
            recorder.write_page,
            recorder.append_to_log,
            compressor=_capturing,
        )
        await sink.write_page(sample_page)
        assert len(captured) == 1
        # The compressor sees the raw page body — no YAML, no fences.
        assert captured[0] == sample_page.body


# --------------------------------------------------------------------------- #
# CompressingWriteSink wrapper — Protocol composition                         #
# --------------------------------------------------------------------------- #


class TestCompressingWrapper:
    @pytest.mark.asyncio
    async def test_compressing_wrapper_around_null(self, sample_page: Page) -> None:
        inner = NullWriteSink()
        sink = CompressingWriteSink(inner, compressor=_fake_compressor)
        await sink.write_page(sample_page)
        # Inner sink received a Page whose body is the compressed form.
        assert len(inner.calls) == 1
        _mode, recorded_page = inner.calls[0]
        assert recorded_page.body == "COMPRESSED"
        # Frontmatter passed through untouched.
        assert recorded_page.frontmatter == sample_page.frontmatter
        # Path passed through untouched.
        assert recorded_page.ref.page_path == sample_page.ref.page_path

    @pytest.mark.asyncio
    async def test_compressing_wrapper_around_local(self, sample_page: Page) -> None:
        recorder = _Recorder()
        local = LocalWriteSink(recorder.write_page, recorder.append_to_log)
        sink = CompressingWriteSink(local, compressor=_fake_compressor)
        await sink.write_page(sample_page)
        assert recorder.body is not None
        assert "COMPRESSED" in recorder.body

    @pytest.mark.asyncio
    async def test_compressing_wrapper_forwards_mode(self, sample_page: Page) -> None:
        inner = NullWriteSink()
        sink = CompressingWriteSink(inner, compressor=_fake_compressor)
        await sink.write_page(sample_page, mode="create")
        assert inner.calls[0][0] == "create"

    @pytest.mark.asyncio
    async def test_compressing_wrapper_forwards_log(self) -> None:
        inner = NullWriteSink()
        sink = CompressingWriteSink(inner, compressor=_fake_compressor)
        await sink.append_to_log("entry text")
        assert inner.log_entries == ["entry text"]


# --------------------------------------------------------------------------- #
# End-to-end with the real default pipeline                                   #
# --------------------------------------------------------------------------- #


class TestDefaultPipelineIntegration:
    @pytest.mark.asyncio
    async def test_default_pipeline_shrinks_body(self, sample_page: Page) -> None:
        recorder = _Recorder()
        pipe, _ = build_default_pipeline()
        sink = LocalWriteSink(
            recorder.write_page,
            recorder.append_to_log,
            compressor=pipe.compress,
        )
        await sink.write_page(sample_page)
        # The HTML stripped + paragraph-dedupe applied to the recorded body.
        assert recorder.body is not None
        assert "<p>" not in recorder.body  # HTML stripped.
        # The duplicate paragraph was collapsed.
        assert recorder.body.count("would normally trigger HTML stripping") == 1

    @pytest.mark.asyncio
    async def test_compressing_wrapper_with_default_pipeline(self, sample_page: Page) -> None:
        inner = NullWriteSink()
        pipe, _ = build_default_pipeline()
        sink = CompressingWriteSink(inner, compressor=pipe.compress)
        await sink.write_page(sample_page)
        recorded_page = inner.calls[0][1]
        assert "<p>" not in recorded_page.body
        # Original body had two paragraphs; only one should remain.
        assert recorded_page.body.count("would normally trigger HTML stripping") == 1


# --------------------------------------------------------------------------- #
# Type-shape sanity — the wrapper is structural-WriteSink                     #
# --------------------------------------------------------------------------- #


class TestProtocolShape:
    def test_wrapper_has_write_sink_methods(self) -> None:
        wrapped = CompressingWriteSink(NullWriteSink(), compressor=_fake_compressor)
        # Duck-type the protocol surface: both required methods exist.
        assert callable(wrapped.write_page)
        assert callable(wrapped.append_to_log)

    def test_local_sink_accepts_keyword_compressor(self) -> None:
        recorder = _Recorder()
        # Keyword-only — positional must raise. This guards against future
        # accidental positional args mis-aligning with `compressor`.
        sink = LocalWriteSink(
            recorder.write_page,
            recorder.append_to_log,
            compressor=_fake_compressor,
        )
        assert sink is not None


# --------------------------------------------------------------------------- #
# Sanity: importable names                                                    #
# --------------------------------------------------------------------------- #


def test_public_names_exported() -> None:
    from wiki_agent import write_sink

    for name in (
        "ALLOWED_PREFIXES",
        "BodyCompressor",
        "CompressingWriteSink",
        "LocalWriteSink",
        "NullWriteSink",
        "WriteSinkPolicyError",
    ):
        assert name in write_sink.__all__, f"missing export: {name}"
