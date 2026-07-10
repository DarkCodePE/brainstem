"""
Tests for the ``linkedin_draft_from_wiki`` MCP tool — ADR-021 Phase 1.

This is the publishing-facing surface Hermes/Tauri call to turn a topic into
a *draft-only* LinkedIn post. The generator + content source are unit-tested
in ``tests/publishing``; here we cover the thin MCP wrapper: the JSON envelope
contract, the file write under ``<wiki>/outputs/linkedin/``, and the two
error branches (no-content, generic failure).

Strategy mirrors ``test_mcp_memory_tree``: inject a fake generator into the
module-level cache so no real router is built, point ``WIKI_ROOT`` at a temp
vault, and await the tool's underlying ``.fn`` (it runs in FastMCP's loop).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from wiki_agent import mcp_server
from wiki_publishing import EmptyContentError, LinkedInDraft, WikiSnippet

draft_tool = mcp_server.linkedin_draft_from_wiki


async def _unwrap(tool, **kwargs):
    """``@mcp.tool()`` wraps the function in a FunctionTool. The LinkedIn
    tool is async (it runs inside FastMCP's event loop), so await ``.fn``."""
    fn = tool.fn if hasattr(tool, "fn") else tool
    return await fn(**kwargs)


class FakeGenerator:
    """Async ``generate`` returning a canned draft — records its calls.

    Mirrors ``LinkedInDraftGenerator.generate``'s current signature (ADR-024:
    the tool forwards ``post_type`` and ``focus`` archetype kwargs)."""

    def __init__(
        self, draft: LinkedInDraft | None = None, *, raises: Exception | None = None
    ) -> None:
        self._draft = draft
        self._raises = raises
        self.calls: list[tuple[str, int, str | None, str | None]] = []

    async def generate(
        self,
        topic: str,
        *,
        max_sources: int = 3,
        post_type: str | None = None,
        focus: str | None = None,
    ) -> LinkedInDraft:
        self.calls.append((topic, max_sources, post_type, focus))
        if self._raises is not None:
            raise self._raises
        assert self._draft is not None
        return self._draft


def _draft(topic: str = "Contextual Retrieval") -> LinkedInDraft:
    return LinkedInDraft(
        topic=topic,
        body="A thoughtful, original post about RAG.",
        sources=(WikiSnippet(title="Contextual Retrieval", page_path="concepts/cr.md", body="x"),),
        model_label="reasoning-stub",
        created_at="2026-05-30T12:00:00+00:00",
    )


@pytest.fixture
def linkedin_env(tmp_path: Path, monkeypatch) -> Iterator[Path]:
    """Point WIKI_ROOT at a temp vault and reset the generator cache so each
    test injects its own fake. The real router is never constructed."""
    monkeypatch.setattr(mcp_server, "WIKI_ROOT", str(tmp_path))
    monkeypatch.setattr(mcp_server, "_linkedin_generator_cache", None)
    yield tmp_path
    monkeypatch.setattr(mcp_server, "_linkedin_generator_cache", None)


class TestDraftTool:
    @pytest.mark.asyncio
    async def test_success_writes_file_and_returns_envelope(
        self, linkedin_env, monkeypatch
    ) -> None:
        fake = FakeGenerator(_draft())
        monkeypatch.setattr(mcp_server, "_linkedin_generator_cache", fake)

        result = json.loads(await _unwrap(draft_tool, topic="contextual retrieval", max_sources=2))

        # Envelope contract.
        assert result["status"] == "draft (unpublished)"
        assert result["topic"] == "Contextual Retrieval"
        assert result["body"] == "A thoughtful, original post about RAG."
        assert result["model"] == "reasoning-stub"
        assert result["sources"] == [
            {"title": "Contextual Retrieval", "page_path": "concepts/cr.md"}
        ]
        # max_sources + the ADR-024 archetype defaults forwarded to the generator
        # (empty focus is normalised to None → the post type's default lens).
        assert fake.calls == [("contextual retrieval", 2, "repo_deep_dive", None)]

        # The draft was persisted under <wiki>/outputs/linkedin/ and is unpublished.
        path = Path(result["draft_path"])
        assert path.parent == linkedin_env / "outputs" / "linkedin"
        assert path.is_file()
        assert "published: false" in path.read_text(encoding="utf-8")

    def test_real_builder_constructs_and_caches_generator(self, linkedin_env) -> None:
        # Cache starts empty (fixture reset). Building wires a real router +
        # read-only content source; no network happens until .generate() runs.
        from wiki_publishing import LinkedInDraftGenerator

        first = mcp_server._get_linkedin_generator()
        second = mcp_server._get_linkedin_generator()

        assert isinstance(first, LinkedInDraftGenerator)
        assert first is second  # lazy-singleton: built once, then cached

    @pytest.mark.asyncio
    async def test_default_max_sources_is_three(self, linkedin_env, monkeypatch) -> None:
        fake = FakeGenerator(_draft())
        monkeypatch.setattr(mcp_server, "_linkedin_generator_cache", fake)

        await _unwrap(draft_tool, topic="some topic")

        assert fake.calls == [("some topic", 3, "repo_deep_dive", None)]

    @pytest.mark.asyncio
    async def test_empty_content_returns_no_content_envelope(
        self, linkedin_env, monkeypatch
    ) -> None:
        fake = FakeGenerator(raises=EmptyContentError("no wiki pages match 'x'"))
        monkeypatch.setattr(mcp_server, "_linkedin_generator_cache", fake)

        result = json.loads(await _unwrap(draft_tool, topic="x"))

        assert result["status"] == "no-content"
        assert result["topic"] == "x"
        assert "no wiki pages match" in result["error"]
        # Nothing written on the no-content path.
        assert not (linkedin_env / "outputs").exists()

    @pytest.mark.asyncio
    async def test_generic_failure_returns_failed_envelope(self, linkedin_env, monkeypatch) -> None:
        fake = FakeGenerator(raises=RuntimeError("router exploded"))
        monkeypatch.setattr(mcp_server, "_linkedin_generator_cache", fake)

        result = json.loads(await _unwrap(draft_tool, topic="x"))

        assert result["status"] == "failed"
        assert result["topic"] == "x"
        # type + message surfaced for the agent, not a stack trace.
        assert result["error"] == "RuntimeError: router exploded"
