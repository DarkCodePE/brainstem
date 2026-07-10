"""Tests for ``wiki_publishing.linkedin_publish`` — ADR-021 Phase 2b.

Mock-first: a fake ``PostExecutor`` records calls and returns canned Composio
responses — no network, no credentials. Covers body extraction, URN
resolution (``urn:li:person``, OIDC ``sub``, and the ``latest`` bare-``id``
shapes), the live-PUBLISHED default + DRAFT opt-in, and the guard rails
(empty/oversized body, stub-mode refusal).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from wiki_publishing import (
    LinkedInPublisher,
    PublishError,
    PublishResult,
    extract_attachments,
    extract_post_body,
    format_for_linkedin,
)


class FakeExecutor:
    """Records (provider, tool, args) calls and returns scripted responses."""

    def __init__(self, responses: dict[str, dict[str, Any]], *, stub_mode: bool = False) -> None:
        self._responses = responses
        self.stub_mode = stub_mode
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def execute(
        self, provider: str, tool_slug: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append((provider, tool_slug, arguments))
        return dict(self._responses.get(tool_slug, {}))


_DRAFT_MD = """---
title: "LinkedIn draft: LLM wiki"
status: draft
published: false
sources:
  - "wiki/concepts/llm-wiki.md"
---

# LinkedIn draft — LLM wiki

> Unpublished draft. Review, edit, then post manually. SBW does not publish in Phase 1 (ADR-021).

Llevo semanas explorando el patrón LLM Wiki y es una de esas ideas que parecen obvias en retrospectiva.

El conocimiento se acumula con cada fuente. #LLMWiki

## Sources (not part of the post)

- [[wiki/concepts/llm-wiki.md]] — LLM Wiki
"""


def _executor(stub_mode: bool = False) -> FakeExecutor:
    return FakeExecutor(
        {
            "LINKEDIN_GET_MY_INFO": {"response_dict": {"sub": "mWaY2Gabc"}},
            "LINKEDIN_CREATE_LINKED_IN_POST": {"id": "urn:li:share:7123"},
        },
        stub_mode=stub_mode,
    )


_DRAFT_MD_WITH_IMAGE = """---
title: "LinkedIn draft: headroom"
status: draft
---

# LinkedIn draft — headroom

> Unpublished draft. Review, edit, then post manually.

Headroom comprime el contexto de tus agentes. 60-95% menos tokens.

Repo: https://github.com/chopratejas/headroom

## 📎 Imagen para adjuntar manualmente

> Al publicar en LinkedIn, adjunta esta imagen. NO es parte del texto.

- `/home/user/second-brain-wiki/knowledge-base/assets/diagrams/chopratejas-headroom.png`

## Sources (not part of the post)

- [[wiki/sources/chopratejas-headroom.md]] — Headroom
"""


class TestExtractPostBody:
    def test_strips_frontmatter_title_note_and_sources(self) -> None:
        body = extract_post_body(_DRAFT_MD)
        assert body.startswith("Llevo semanas")
        assert "#LLMWiki" in body
        # None of the non-post scaffolding leaks into the published text.
        assert "title:" not in body
        assert "# LinkedIn draft" not in body
        assert "Unpublished draft" not in body
        assert "## Sources" not in body
        assert "llm-wiki.md" not in body

    def test_strips_attachments_block_and_local_path(self) -> None:
        """The '## 📎' attachments block carries a LOCAL file path; it must never
        reach the published text (regression: the headroom live post leaked the
        PNG's absolute path)."""
        body = extract_post_body(_DRAFT_MD_WITH_IMAGE)
        assert body.startswith("Headroom comprime")
        assert "github.com/chopratejas/headroom" in body  # the repo URL stays
        assert "📎" not in body
        assert "Imagen para adjuntar" not in body
        assert ".png" not in body
        assert "/home/user" not in body
        assert "## Sources" not in body


class TestFormatForLinkedIn:
    def test_bold_becomes_unicode_and_no_asterisks(self) -> None:
        from wiki_publishing.linkedin_publish import _to_unicode_bold

        out = format_for_linkedin("Como **librería**: rápido.")
        assert "**" not in out  # no literal markdown asterisks reach LinkedIn
        assert _to_unicode_bold("librer") in out  # ASCII letters rendered bold
        assert "í" in out  # accented char passes through (no bold glyph exists)

    def test_bold_digits_and_percent(self) -> None:
        from wiki_publishing.linkedin_publish import _to_unicode_bold

        out = format_for_linkedin("ahorros de **47% y 92%** de tokens")
        assert "**" not in out
        assert _to_unicode_bold("47") in out and _to_unicode_bold("92") in out
        assert "%" in out  # punctuation untouched

    def test_single_asterisk_italic_becomes_unicode(self) -> None:
        from wiki_publishing.linkedin_publish import _to_unicode_italic

        out = format_for_linkedin("Con el contexto *Lesson Memory* todo se escribe.")
        assert "*" not in out  # no literal single-asterisk markup leaks
        assert _to_unicode_italic("Lesson Memory") in out

    def test_italic_does_not_eat_bold_or_bullets(self) -> None:
        from wiki_publishing.linkedin_publish import _to_unicode_bold

        # Bold survives as bold (not mangled by the italic pass)…
        out = format_for_linkedin("**fuerte** y *suave*")
        assert "*" not in out
        assert _to_unicode_bold("fuerte") in out
        # …and a "* " bullet is still a bullet, not italic.
        bullets = format_for_linkedin("* uno\n* dos")
        assert "• uno" in bullets and "• dos" in bullets

    def test_inline_code_backticks_removed(self) -> None:
        out = format_for_linkedin("Usa `compress(messages)` aquí.")
        assert "`" not in out
        assert "compress(messages)" in out

    def test_bullets_become_dots_and_headings_lose_hashes(self) -> None:
        out = format_for_linkedin("## Título\n- uno\n- dos")
        assert "•" in out
        assert "- uno" not in out
        assert "#" not in out
        assert "Título" in out

    def test_markdown_link_becomes_text_and_url(self) -> None:
        out = format_for_linkedin("Mira [los docs](https://x.dev/benchmarks).")
        assert "[" not in out and "](" not in out
        assert "los docs (https://x.dev/benchmarks)" in out

    def test_scheme_less_url_gets_https(self) -> None:
        # issue #193: a bare host/path is upgraded so LinkedIn links it.
        out = format_for_linkedin("Fuente: github.com/DarkCodePE/second-brain-wiki")
        assert "https://github.com/DarkCodePE/second-brain-wiki" in out

    def test_existing_scheme_not_doubled(self) -> None:
        out = format_for_linkedin("Ver https://github.com/org/repo y lnkd.in/abc123")
        assert "https://https://" not in out
        assert "https://github.com/org/repo" in out
        assert "https://lnkd.in/abc123" in out  # the bare one gets a scheme

    def test_file_paths_and_model_ids_are_not_urls(self) -> None:
        # No dotted TLD + path, so these must NOT be turned into links.
        out = format_for_linkedin("Usa wiki/lessons/ con deepseek/v4-flash")
        assert "https://" not in out


class TestExtractAttachments:
    def test_returns_image_path_from_block(self) -> None:
        paths = extract_attachments(_DRAFT_MD_WITH_IMAGE)
        assert paths == [
            "/home/user/second-brain-wiki/knowledge-base/assets/diagrams/chopratejas-headroom.png"
        ]

    def test_no_block_returns_empty(self) -> None:
        assert extract_attachments(_DRAFT_MD) == []


class TestPublishDraft:
    def test_publishes_live_under_own_urn(self) -> None:
        ex = _executor()
        pub = LinkedInPublisher(executor=ex)

        result = asyncio.run(pub.publish_draft(body="Hola mundo.", draft_path="/v/d.md"))

        assert isinstance(result, PublishResult)
        # Phase 2b default: live PUBLISHED (an API DRAFT is orphaned/invisible).
        assert result.status == "published-on-linkedin"
        assert result.lifecycle_state == "PUBLISHED"
        assert result.author_urn == "urn:li:person:mWaY2Gabc"
        assert result.post_id == "urn:li:share:7123"
        # Two Composio calls: resolve URN, then create the post.
        assert [c[1] for c in ex.calls] == [
            "LINKEDIN_GET_MY_INFO",
            "LINKEDIN_CREATE_LINKED_IN_POST",
        ]
        # The create call posts LIVE under the resolved author.
        _, _, args = ex.calls[1]
        assert args["lifecycleState"] == "PUBLISHED"
        assert args["author"] == "urn:li:person:mWaY2Gabc"
        assert args["commentary"] == "Hola mundo."

    def test_draft_lifecycle_is_opt_in(self) -> None:
        ex = _executor()
        result = asyncio.run(
            LinkedInPublisher(executor=ex).publish_draft(
                body="x", draft_path="/v/d.md", lifecycle_state="DRAFT"
            )
        )
        assert result.status == "draft-created-on-linkedin"
        assert result.lifecycle_state == "DRAFT"
        assert ex.calls[1][2]["lifecycleState"] == "DRAFT"

    def test_urn_from_explicit_urn_field(self) -> None:
        ex = FakeExecutor(
            {
                "LINKEDIN_GET_MY_INFO": {"author": "urn:li:person:ZZZ"},
                "LINKEDIN_CREATE_LINKED_IN_POST": {},
            }
        )
        result = asyncio.run(
            LinkedInPublisher(executor=ex).publish_draft(body="x", draft_path="/v/d.md")
        )
        assert result.author_urn == "urn:li:person:ZZZ"

    def test_urn_from_latest_toolkit_bare_id(self) -> None:
        # Composio toolkit version="latest" returns a bare member id under
        # "id" (no urn: prefix, no sub) — the publisher must build the URN.
        ex = FakeExecutor(
            {
                "LINKEDIN_GET_MY_INFO": {"id": "mWaY2Gm8rQ", "localizedFirstName": "Orlando"},
                "LINKEDIN_CREATE_LINKED_IN_POST": {},
            }
        )
        result = asyncio.run(
            LinkedInPublisher(executor=ex).publish_draft(body="x", draft_path="/v/d.md")
        )
        assert result.author_urn == "urn:li:person:mWaY2Gm8rQ"

    def test_stub_mode_executor_refuses_to_publish(self) -> None:
        ex = _executor(stub_mode=True)
        with pytest.raises(PublishError, match="stub mode"):
            asyncio.run(LinkedInPublisher(executor=ex).publish_draft(body="x", draft_path="/d.md"))
        assert ex.calls == []  # nothing dispatched

    def test_empty_body_rejected(self) -> None:
        ex = _executor()
        with pytest.raises(PublishError, match="empty"):
            asyncio.run(
                LinkedInPublisher(executor=ex).publish_draft(body="   ", draft_path="/d.md")
            )
        assert ex.calls == []

    def test_oversized_body_rejected(self) -> None:
        ex = _executor()
        with pytest.raises(PublishError, match="exceeds"):
            asyncio.run(
                LinkedInPublisher(executor=ex).publish_draft(body="x" * 3001, draft_path="/d.md")
            )
        assert ex.calls == []

    def test_unresolvable_urn_raises_and_skips_post(self) -> None:
        ex = FakeExecutor(
            {"LINKEDIN_GET_MY_INFO": {"nothing": "useful"}, "LINKEDIN_CREATE_LINKED_IN_POST": {}}
        )
        with pytest.raises(PublishError, match="author URN"):
            asyncio.run(LinkedInPublisher(executor=ex).publish_draft(body="x", draft_path="/d.md"))
        # GET_MY_INFO was called, but the post was never attempted.
        assert [c[1] for c in ex.calls] == ["LINKEDIN_GET_MY_INFO"]
