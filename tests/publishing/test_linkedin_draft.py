"""Tests for ``wiki_publishing`` — ADR-021 Phase 1 (draft-only).

Mock-first: the generator is exercised against a fake ``ContentSource`` and a
``StubBackend``-wired ``ModelRouter``, so no network or vault access happens.
The ``WikiContentSource`` adapter is tested against a temp vault.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wiki_publishing import (
    EmptyContentError,
    LinkedInDraft,
    LinkedInDraftGenerator,
    WikiContentSource,
    WikiSnippet,
    write_draft,
)
from wiki_routing.policy import RoutingPolicy
from wiki_routing.router import BackendResponse, ModelRouter, StubBackend
from wiki_routing.tiers import Tier

_FIXED_NOW = datetime(2026, 5, 30, 12, 0, 0, tzinfo=UTC)


class FakeContentSource:
    """Records queries and returns canned snippets — no I/O."""

    def __init__(self, snippets: list[WikiSnippet]) -> None:
        self._snippets = snippets
        self.queries: list[tuple[str, int]] = []
        self.last_categories = None

    def search(self, query: str, *, limit: int = 3, categories=None) -> list[WikiSnippet]:
        self.queries.append((query, limit))
        self.last_categories = categories
        return list(self._snippets[:limit])


def _router(text: str = "Drafted post body.", label: str = "reasoning-stub"):
    reasoning = StubBackend(
        label=label,
        response=BackendResponse(text=text, tokens_in=10, tokens_out=20, cost_usd=0.0),
    )
    fast = StubBackend(
        label="fast-stub",
        response=BackendResponse(
            text="FAST MUST NOT BE USED", tokens_in=1, tokens_out=1, cost_usd=0.0
        ),
    )
    router = ModelRouter(
        policy=RoutingPolicy(),
        providers={Tier.FAST: fast, Tier.REASONING: reasoning},
    )
    return router, fast, reasoning


def _snippet(
    title: str = "Contextual Retrieval",
    path: str = "concepts/cr.md",
    body: str = "RAG preprocessing improves retrieval.",
) -> WikiSnippet:
    return WikiSnippet(title=title, page_path=path, body=body)


class TestDraftGeneration:
    def test_routes_to_reasoning_and_returns_draft(self) -> None:
        router, fast, reasoning = _router(text="A thoughtful post about RAG.")
        source = FakeContentSource([_snippet()])
        gen = LinkedInDraftGenerator(router=router, content_source=source, clock=lambda: _FIXED_NOW)

        draft = asyncio.run(gen.generate("contextual retrieval"))

        assert isinstance(draft, LinkedInDraft)
        assert draft.body == "A thoughtful post about RAG."
        assert draft.model_label == "reasoning-stub"
        assert draft.sources[0].page_path == "concepts/cr.md"
        assert draft.created_at == _FIXED_NOW.isoformat()
        # intent="draft" -> REASONING; FAST tier must be untouched.
        assert len(reasoning.calls) == 1
        assert fast.calls == []
        # The content source was queried with the topic.
        assert source.queries == [("contextual retrieval", 3)]

    def test_system_prompt_guards_plagiarism_and_fabrication(self) -> None:
        router, _fast, reasoning = _router()
        gen = LinkedInDraftGenerator(
            router=router, content_source=FakeContentSource([_snippet()]), clock=lambda: _FIXED_NOW
        )

        asyncio.run(gen.generate("contextual retrieval"))

        messages = reasoning.calls[0].messages
        system = messages[0].content.lower()
        user = messages[1].content
        assert messages[0].role == "system"
        assert "verbatim" in system and "invent" in system  # no copying, no fabrication
        assert "draft" in system
        # Output language + tone are pinned: Spanish, "profesional cercano".
        assert "spanish" in system
        assert "profesional cercano" in system
        # The user prompt carries the topic + the snippet body as source material.
        assert "contextual retrieval" in user
        assert "RAG preprocessing improves retrieval." in user

    def test_empty_content_raises_and_router_not_called(self) -> None:
        router, fast, reasoning = _router()
        gen = LinkedInDraftGenerator(
            router=router, content_source=FakeContentSource([]), clock=lambda: _FIXED_NOW
        )

        with pytest.raises(EmptyContentError):
            asyncio.run(gen.generate("a topic with no wiki coverage"))

        assert reasoning.calls == [] and fast.calls == []

    def test_blank_topic_rejected(self) -> None:
        router, _fast, _reasoning = _router()
        gen = LinkedInDraftGenerator(router=router, content_source=FakeContentSource([_snippet()]))
        with pytest.raises(ValueError):
            asyncio.run(gen.generate("   "))

    def test_max_sources_forwarded(self) -> None:
        router, _fast, _reasoning = _router()
        source = FakeContentSource([_snippet(), _snippet(title="B", path="concepts/b.md")])
        gen = LinkedInDraftGenerator(router=router, content_source=source, clock=lambda: _FIXED_NOW)
        asyncio.run(gen.generate("x topic", max_sources=2))
        assert source.queries == [("x topic", 2)]


class TestSourceUrls:
    """L3 fix: source URLs must reach the composer so they can appear in the post."""

    def test_source_urls_extracted_from_frontmatter_and_prose(self) -> None:
        from wiki_publishing.linkedin_draft import _source_urls

        body = (
            '---\ntitle: "X"\nsources:\n'
            '  - "https://github.com/rowboatlabs/rowboat"\n'
            "---\n\n"
            "El paper esta en https://arxiv.org/abs/2507.19457. "
            "Repite https://github.com/rowboatlabs/rowboat (dup)."
        )
        urls = _source_urls([_snippet(body=body)])
        # deduped, order-preserved, trailing punctuation stripped
        assert urls == [
            "https://github.com/rowboatlabs/rowboat",
            "https://arxiv.org/abs/2507.19457",
        ]

    def test_render_user_prompt_includes_urls_block(self) -> None:
        from wiki_publishing.linkedin_draft import _render_user_prompt

        body = '---\nsources:\n  - "https://example.com/post"\n---\nCuerpo.'
        prompt = _render_user_prompt("tema", [_snippet(body=body)])
        assert "FUENTES (URLs)" in prompt
        assert "https://example.com/post" in prompt

    def test_render_user_prompt_no_block_when_no_urls(self) -> None:
        from wiki_publishing.linkedin_draft import _render_user_prompt

        prompt = _render_user_prompt("tema", [_snippet(body="Sin enlaces aqui.")])
        assert "FUENTES (URLs)" not in prompt

    def test_system_prompt_instructs_url_inclusion(self) -> None:
        from wiki_publishing.linkedin_draft import _SYSTEM_PROMPT

        s = _SYSTEM_PROMPT.lower()
        assert "url" in s and "fuente" in s
        assert "never invent" in s  # must not fabricate URLs

    def test_generic_urls_filtered(self) -> None:
        from wiki_publishing.linkedin_draft import _source_urls

        body = (
            "Ver https://www.linkedin.com/feed/ y https://example.com "
            "y el permalink https://arxiv.org/abs/2507.19457v2 ."
        )
        # bare domain + social feed root dropped; real permalink kept
        assert _source_urls([_snippet(body=body)]) == ["https://arxiv.org/abs/2507.19457v2"]

    def test_generate_appends_url_when_model_omits_it(self) -> None:
        # Model returns prose WITHOUT the link; generate() must append it.
        router, _fast, _reasoning = _router(text="Un post sobre el paper, sin enlace.")
        body = '---\nsources:\n  - "https://arxiv.org/abs/2507.19457v2"\n---\nGEPA paper.'
        gen = LinkedInDraftGenerator(
            router=router, content_source=FakeContentSource([_snippet(body=body)])
        )
        draft = asyncio.run(gen.generate("gepa"))
        assert "Fuente: https://arxiv.org/abs/2507.19457v2" in draft.body

    def test_generate_does_not_duplicate_url_already_present(self) -> None:
        # Model already included the URL → no appended "Fuente:" duplicate.
        url = "https://arxiv.org/abs/2507.19457v2"
        router, _fast, _reasoning = _router(text=f"Un post con el enlace {url} incluido.")
        body = f'---\nsources:\n  - "{url}"\n---\nGEPA paper.'
        gen = LinkedInDraftGenerator(
            router=router, content_source=FakeContentSource([_snippet(body=body)])
        )
        draft = asyncio.run(gen.generate("gepa"))
        assert draft.body.count(url) == 1
        assert "Fuente:" not in draft.body


class TestDraftSerialisation:
    def test_to_markdown_is_unpublished_and_lists_sources(self) -> None:
        draft = LinkedInDraft(
            topic="Contextual Retrieval",
            body="The post body.",
            sources=(_snippet(),),
            model_label="reasoning-stub",
            created_at=_FIXED_NOW.isoformat(),
        )
        md = draft.to_markdown()
        assert "status: draft" in md
        assert "published: false" in md
        assert "published: true" not in md
        assert "The post body." in md
        assert "concepts/cr.md" in md  # source attribution

    def test_write_draft_persists_under_outputs_linkedin(self, tmp_path: Path) -> None:
        draft = LinkedInDraft(
            topic="Contextual Retrieval",
            body="Body.",
            sources=(_snippet(),),
            model_label="reasoning-stub",
            created_at=_FIXED_NOW.isoformat(),
        )
        path = write_draft(draft, wiki_root=tmp_path)
        assert (
            path
            == tmp_path
            / "outputs"
            / "linkedin"
            / "linkedin-draft-contextual-retrieval-2026-05-30.md"
        )
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "published: false" in text and "Body." in text


class TestWikiContentSource:
    def _build_vault(self, root: Path) -> None:
        wiki = root / "wiki"
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "index.md").write_text(
            "## concepts\n\n"
            "| Page | Summary | Sources | Updated |\n"
            "|------|---------|---------|---------|\n"
            "| [Contextual Retrieval](concepts/cr.md) | RAG preprocessing improves retrieval accuracy | s | 2026-05-29 |\n"
            "| [Tomato Gardening](concepts/tomato.md) | Growing tomatoes in summer | s | 2026-05-01 |\n",
            encoding="utf-8",
        )
        (wiki / "concepts" / "cr.md").write_text(
            "# Contextual Retrieval\n\nFull body: prepend context to each chunk before embedding.\n",
            encoding="utf-8",
        )

    def test_search_returns_relevant_snippet_with_body(self, tmp_path: Path) -> None:
        self._build_vault(tmp_path)
        src = WikiContentSource(wiki_root=tmp_path)

        results = src.search("contextual retrieval RAG", limit=3)

        assert len(results) == 1
        assert results[0].title == "Contextual Retrieval"
        assert results[0].page_path == "concepts/cr.md"
        assert "prepend context to each chunk" in results[0].body  # body read from page

    def test_irrelevant_query_returns_nothing(self, tmp_path: Path) -> None:
        self._build_vault(tmp_path)
        src = WikiContentSource(wiki_root=tmp_path)
        assert src.search("quantum chromodynamics", limit=3) == []

    def test_missing_index_returns_empty(self, tmp_path: Path) -> None:
        src = WikiContentSource(wiki_root=tmp_path)
        assert src.search("anything", limit=3) == []


# --- manual image attachment (ADR-023 Phase 1: surface the diagram PNG) ---


class _FakeRouter:
    """Minimal ModelRouter stand-in returning a fixed body."""

    async def call(self, task, messages):
        from types import SimpleNamespace

        return SimpleNamespace(text="Cuerpo del post.", backend_label="fake:model")


class _FakeSource:
    def __init__(self, snippets):
        self._snippets = snippets

    def search(self, query, *, limit=3, categories=None):
        return self._snippets[:limit]


def test_attachments_render_in_draft_markdown():
    from wiki_publishing.linkedin_draft import LinkedInDraft, WikiSnippet

    d = LinkedInDraft(
        topic="odysseus",
        body="Texto del post.",
        sources=(WikiSnippet(title="odysseus", page_path="wiki/sources/o-odysseus.md", body="x"),),
        model_label="m",
        created_at="2026-06-03T00:00:00+00:00",
        attachments=("/vault/assets/diagrams/o-odysseus.png",),
    )
    md = d.to_markdown()
    assert "📎 Imagen para adjuntar manualmente" in md
    assert "/vault/assets/diagrams/o-odysseus.png" in md
    # the attachment is NOT inside the post body section
    assert "diagrams/o-odysseus.png" not in d.body


def test_no_attachments_block_when_empty():
    from wiki_publishing.linkedin_draft import LinkedInDraft

    d = LinkedInDraft(
        topic="t", body="b", sources=(), model_label="m", created_at="2026-06-03T00:00:00+00:00"
    )
    assert "📎" not in d.to_markdown()


async def _run_with_resolver(resolver):
    from wiki_publishing.linkedin_draft import LinkedInDraftGenerator, WikiSnippet

    snip = WikiSnippet(title="odysseus", page_path="wiki/sources/o-odysseus.md", body="repo info")
    gen = LinkedInDraftGenerator(
        router=_FakeRouter(),
        content_source=_FakeSource([snip]),
        clock=lambda: __import__("datetime").datetime(
            2026, 6, 3, tzinfo=__import__("datetime").timezone.utc
        ),
        attachment_resolver=resolver,
    )
    return await gen.generate("odysseus")


def test_resolver_populates_attachments():
    import asyncio

    draft = asyncio.run(
        _run_with_resolver(lambda snippets: ["/vault/assets/diagrams/o-odysseus.png"])
    )
    assert draft.attachments == ("/vault/assets/diagrams/o-odysseus.png",)
    assert "📎" in draft.to_markdown()


def test_resolver_failure_degrades_to_no_attachments():
    import asyncio

    def boom(snippets):
        raise RuntimeError("fs error")

    draft = asyncio.run(_run_with_resolver(boom))
    assert draft.attachments == ()
