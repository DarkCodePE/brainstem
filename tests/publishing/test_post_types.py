"""Tests for ADR-024 post types — registry + per-type generate behavior.

Hermetic: fake ContentSource + fake router; no network/LLM/vault.
"""

from __future__ import annotations

import pytest

from wiki_publishing.linkedin_draft import LinkedInDraftGenerator, WikiSnippet
from wiki_publishing.post_types import (
    ATTACH_DIAGRAM,
    ATTACH_NONE,
    DEFAULT_POST_TYPE,
    PostType,
    coerce_post_type,
    spec_for,
)

# --- registry ---


def test_all_archetypes_have_a_spec():
    for pt in PostType:
        spec = spec_for(pt)
        assert spec.post_type is pt
        assert spec.system_overlay.strip()
        assert spec.max_chars > 0
        assert spec.attachment in (ATTACH_DIAGRAM, ATTACH_NONE)


def test_coerce_is_lenient_and_defaults():
    assert coerce_post_type("showcase") is PostType.SHOWCASE
    assert coerce_post_type("SHOWCASE") is PostType.SHOWCASE
    assert coerce_post_type(PostType.TUTORIAL) is PostType.TUTORIAL
    assert coerce_post_type(None) is DEFAULT_POST_TYPE
    assert coerce_post_type("nonsense") is DEFAULT_POST_TYPE  # no inference, safe default


def test_source_categories_per_type():
    assert spec_for(PostType.REPO_DEEP_DIVE).categories == ("sources",)
    assert spec_for(PostType.SHOWCASE).categories == ("sources",)
    assert "concepts" in spec_for(PostType.TUTORIAL).categories
    assert spec_for(PostType.INFORMATIVO).categories == ("concepts", "entities")
    # informativo must NOT draw from repo source pages
    assert "sources" not in spec_for(PostType.INFORMATIVO).categories


def test_attachment_policy_per_type():
    assert spec_for(PostType.REPO_DEEP_DIVE).attachment == ATTACH_DIAGRAM
    assert spec_for(PostType.SHOWCASE).attachment == ATTACH_DIAGRAM
    assert spec_for(PostType.TUTORIAL).attachment == ATTACH_NONE
    assert spec_for(PostType.INFORMATIVO).attachment == ATTACH_NONE


# --- generate behavior ---


class _RecordingSource:
    def __init__(self, snippets):
        self._snippets = snippets
        self.last_categories = "UNSET"

    def search(self, query, *, limit=3, categories=None):
        self.last_categories = categories
        return list(self._snippets[:limit])


class _SpyRouter:
    def __init__(self):
        self.system = None

    async def call(self, task, messages):
        from types import SimpleNamespace

        self.system = next(m.content for m in messages if m.role == "system")
        return SimpleNamespace(text="Cuerpo del post.", backend_label="spy:model")


def _snip(path="sources/x.md"):
    return WikiSnippet(title="X", page_path=path, body="contenido del repo")


def _clock():
    from datetime import UTC, datetime

    return datetime(2026, 6, 4, tzinfo=UTC)


@pytest.mark.asyncio
async def test_generate_passes_type_categories_to_search():
    src = _RecordingSource([_snip()])
    gen = LinkedInDraftGenerator(router=_SpyRouter(), content_source=src, clock=_clock)
    await gen.generate("turbovec", post_type="showcase")
    assert src.last_categories == ("sources",)

    src2 = _RecordingSource([_snip("concepts/a.md")])
    gen2 = LinkedInDraftGenerator(router=_SpyRouter(), content_source=src2, clock=_clock)
    await gen2.generate("agentes de IA", post_type="informativo")
    assert src2.last_categories == ("concepts", "entities")


@pytest.mark.asyncio
async def test_generate_overlays_type_prompt_and_tags_draft():
    router = _SpyRouter()
    gen = LinkedInDraftGenerator(
        router=router, content_source=_RecordingSource([_snip()]), clock=_clock
    )
    draft = await gen.generate("turbovec", post_type="showcase")
    assert draft.post_type == "showcase"
    assert "spotlight" in router.system.lower()  # the showcase overlay is present
    assert "NUNCA inventes" in router.system  # the no-fabricated-numbers rule
    assert "post_type: showcase" in draft.to_markdown()


@pytest.mark.asyncio
async def test_default_is_repo_deep_dive():
    router = _SpyRouter()
    gen = LinkedInDraftGenerator(
        router=router, content_source=_RecordingSource([_snip()]), clock=_clock
    )
    draft = await gen.generate("mi repo")
    assert draft.post_type == "repo_deep_dive"


@pytest.mark.asyncio
async def test_attachment_resolved_only_for_diagram_types():
    calls = {"n": 0}

    def resolver(snips):
        calls["n"] += 1
        return ["/vault/assets/diagrams/x.png"]

    # showcase → diagram policy → resolver runs
    gen = LinkedInDraftGenerator(
        router=_SpyRouter(),
        content_source=_RecordingSource([_snip()]),
        clock=_clock,
        attachment_resolver=resolver,
    )
    d1 = await gen.generate("x", post_type="showcase")
    assert d1.attachments == ("/vault/assets/diagrams/x.png",)

    # informativo → no attachment policy → resolver NOT called
    gen2 = LinkedInDraftGenerator(
        router=_SpyRouter(),
        content_source=_RecordingSource([_snip("concepts/a.md")]),
        clock=_clock,
        attachment_resolver=resolver,
    )
    before = calls["n"]
    d2 = await gen2.generate("tema", post_type="informativo")
    assert d2.attachments == ()
    assert calls["n"] == before  # resolver was not invoked for informativo


# --- focus axis (ADR-024 A1) ---


def test_focus_defaults_per_type():
    from wiki_publishing.post_types import Focus, spec_for

    assert spec_for(PostType.SHOWCASE).default_focus is Focus.USE
    assert spec_for(PostType.REPO_DEEP_DIVE).default_focus is Focus.CODE


def test_coerce_focus():
    from wiki_publishing.post_types import Focus, coerce_focus

    assert coerce_focus("use") is Focus.USE
    assert coerce_focus("CODE") is Focus.CODE
    assert coerce_focus(None) is None  # → caller uses the type default
    assert coerce_focus("nonsense") is None


@pytest.mark.asyncio
async def test_focus_overlay_applied_and_tagged():
    router = _SpyRouter()
    gen = LinkedInDraftGenerator(
        router=router, content_source=_RecordingSource([_snip()]), clock=_clock
    )
    # explicit use focus on a deep_dive (override its code default)
    d = await gen.generate("headroom", post_type="repo_deep_dive", focus="use")
    assert d.focus == "use"
    assert "ENFOQUE = USO" in router.system
    assert "post_type: repo_deep_dive" in d.to_markdown()
    assert "focus: use" in d.to_markdown()


@pytest.mark.asyncio
async def test_showcase_defaults_to_use_focus():
    router = _SpyRouter()
    gen = LinkedInDraftGenerator(
        router=router, content_source=_RecordingSource([_snip()]), clock=_clock
    )
    d = await gen.generate("turbovec", post_type="showcase")  # no explicit focus
    assert d.focus == "use"
    assert "ENFOQUE = USO" in router.system
