"""
Tests for the ``linkedin_publish_draft`` MCP tool — ADR-021 Phase 2a.

Covers the chat-facing seam: the HITL contract (anything but the typed
``PUBLICAR`` phrase returns ``manual-ready`` — the post-it-yourself path with
the clean body + image, since 2026-06-04 the default outcome), the path
traversal guard, and that the publisher is ONLY invoked after an explicit
typed confirm. The publisher itself is unit-tested in
``tests/publishing/test_linkedin_publish.py``; here we inject a fake publisher
via the module so no Composio/network is touched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from wiki_agent import mcp_server
from wiki_publishing import PublishResult

publish_tool = mcp_server.linkedin_publish_draft


async def _unwrap(tool, **kwargs):
    fn = tool.fn if hasattr(tool, "fn") else tool
    return await fn(**kwargs)


_DRAFT_MD = """---
title: "LinkedIn draft: LLM wiki"
published: false
---

# LinkedIn draft — LLM wiki

> Unpublished draft. Review, edit, then post manually.

Cuerpo del post anclado en el wiki. #LLMWiki

## 📎 Imagen para adjuntar manualmente

> Al publicar, adjunta esta imagen. NO es parte del texto.

- `/vault/assets/diagrams/demo.png`

## Sources (not part of the post)

- [[wiki/concepts/llm-wiki.md]] — LLM Wiki
"""


@pytest.fixture
def vault(tmp_path: Path, monkeypatch) -> Iterator[Path]:
    monkeypatch.setattr(mcp_server, "WIKI_ROOT", str(tmp_path))
    out = tmp_path / "outputs" / "linkedin"
    out.mkdir(parents=True)
    (out / "linkedin-draft-llm-wiki-2026-05-31.md").write_text(_DRAFT_MD, encoding="utf-8")
    yield tmp_path


class TestHITL:
    @pytest.mark.asyncio
    async def test_step1_without_confirm_returns_full_body_and_no_publish(
        self, vault, monkeypatch
    ) -> None:
        called = {"n": 0}

        class BoomPublisher:
            def __init__(self, **_: object) -> None: ...
            async def publish_draft(self, **_: object):  # must NOT run on step 1
                called["n"] += 1
                raise AssertionError("publish must not happen without confirm")

        monkeypatch.setattr(mcp_server, "LinkedInPublisher", BoomPublisher, raising=False)

        res = json.loads(
            await _unwrap(publish_tool, draft_path="linkedin-draft-llm-wiki-2026-05-31.md")
        )
        # Without the typed PUBLICAR phrase the tool hands the post back for
        # manual posting (MANUAL-mode default, 2026-06-04) — it never publishes.
        assert res["status"] == "manual-ready"
        assert "Cuerpo del post" in res["body"]  # full body shown for review
        assert res["char_count"] > 0
        assert "PUBLICAR" in res["instruction"]  # live publish still offered, opt-in
        assert called["n"] == 0  # nothing published

    @pytest.mark.asyncio
    async def test_step2_with_confirm_invokes_publisher(self, vault, monkeypatch) -> None:
        captured = {}

        class FakePublisher:
            def __init__(self, *, executor) -> None:
                captured["built"] = True

            async def publish_draft(self, *, body, draft_path, visibility="PUBLIC"):
                captured["body"] = body
                return PublishResult(
                    status="published-on-linkedin",
                    author_urn="urn:li:person:ME",
                    post_id="urn:li:share:1",
                    lifecycle_state="PUBLISHED",
                    draft_path=draft_path,
                )

        # Avoid building a real ComposioBridge / writing the audit log.
        monkeypatch.setattr(mcp_server, "LinkedInPublisher", FakePublisher, raising=False)
        import wiki_integrations.composio_bridge as cb

        monkeypatch.setattr(cb, "ComposioBridge", lambda *a, **k: object())

        res = json.loads(
            await _unwrap(
                publish_tool,
                draft_path="linkedin-draft-llm-wiki-2026-05-31.md",
                confirm="PUBLICAR",
            )
        )
        assert res["status"] == "published-on-linkedin"
        assert res["lifecycle_state"] == "PUBLISHED"
        assert res["author_urn"] == "urn:li:person:ME"
        assert "Cuerpo del post" in captured["body"]
        assert "## Sources" not in captured["body"]

    @pytest.mark.asyncio
    async def test_manual_returns_body_and_image_without_publishing(
        self, vault, monkeypatch
    ) -> None:
        """confirm='MANUAL' hands back the clean body + image path and never
        touches the publisher — this is the post-it-yourself-with-image path."""

        class BoomPublisher:
            def __init__(self, **_: object) -> None: ...
            async def publish_draft(self, **_: object):
                raise AssertionError("MANUAL must not publish")

        monkeypatch.setattr(mcp_server, "LinkedInPublisher", BoomPublisher, raising=False)
        res = json.loads(
            await _unwrap(
                publish_tool,
                draft_path="linkedin-draft-llm-wiki-2026-05-31.md",
                confirm="MANUAL",
            )
        )
        assert res["status"] == "manual-ready"
        assert "Cuerpo del post" in res["body"]
        assert res["attachments"] == ["/vault/assets/diagrams/demo.png"]
        # The clean body must NOT carry the local image path or scaffolding.
        assert "demo.png" not in res["body"]
        assert "## Sources" not in res["body"]

    @pytest.mark.asyncio
    async def test_manual_ready_step_warns_about_image_when_present(self, vault) -> None:
        res = json.loads(
            await _unwrap(publish_tool, draft_path="linkedin-draft-llm-wiki-2026-05-31.md")
        )
        assert res["status"] == "manual-ready"
        assert res["attachments"] == ["/vault/assets/diagrams/demo.png"]
        assert "MANUAL" in res["instruction"]

    @pytest.mark.asyncio
    async def test_wrong_confirm_phrase_does_not_publish(self, vault, monkeypatch) -> None:
        class BoomPublisher:
            def __init__(self, **_: object) -> None: ...
            async def publish_draft(self, **_: object):
                raise AssertionError("must not publish on wrong confirm")

        monkeypatch.setattr(mcp_server, "LinkedInPublisher", BoomPublisher, raising=False)
        res = json.loads(
            await _unwrap(
                publish_tool, draft_path="linkedin-draft-llm-wiki-2026-05-31.md", confirm="si"
            )
        )
        # A wrong phrase falls back to the safe manual-ready step — had the
        # publisher run, BoomPublisher's AssertionError would surface as "failed".
        assert res["status"] == "manual-ready"


_DRAFT_MD_MARKDOWN = """---
title: "LinkedIn draft: markdown"
published: false
---

# LinkedIn draft — markdown

> Unpublished draft.

Mi **Second Brain Wiki** usa `wiki/lessons/` como sustrato.

### Sección

- punto uno
- punto dos

Fuente: github.com/DarkCodePE/second-brain-wiki
"""


class TestPublishFormatsForLinkedIn:
    """Issue #192: the body must be flattened to LinkedIn-ready text (no raw
    markdown) in BOTH the preview and the live publish — LinkedIn renders no
    markdown, so literal ``**``/``###``/`` ` `` would leak into the post."""

    @pytest.fixture
    def md_vault(self, tmp_path: Path, monkeypatch) -> Path:
        monkeypatch.setattr(mcp_server, "WIKI_ROOT", str(tmp_path))
        out = tmp_path / "outputs" / "linkedin"
        out.mkdir(parents=True)
        (out / "md.md").write_text(_DRAFT_MD_MARKDOWN, encoding="utf-8")
        return tmp_path

    @pytest.mark.asyncio
    async def test_preview_body_has_no_raw_markdown(self, md_vault) -> None:
        res = json.loads(await _unwrap(publish_tool, draft_path="md.md"))
        body = res["body"]
        assert "**" not in body  # bold markup flattened to Unicode glyphs
        assert "𝗦𝗲𝗰𝗼𝗻𝗱 𝗕𝗿𝗮𝗶𝗻 𝗪𝗶𝗸𝗶" in body  # **Second Brain Wiki** → Unicode bold
        assert "### " not in body and "`" not in body  # heading/code markup gone
        assert "• punto uno" in body  # "- " bullet → "• "
        # scheme-less URL is upgraded so LinkedIn renders a verified link
        assert "https://github.com/DarkCodePE/second-brain-wiki" in body

    @pytest.mark.asyncio
    async def test_published_body_matches_formatted_preview(self, md_vault, monkeypatch) -> None:
        captured = {}

        class FakePublisher:
            def __init__(self, *, executor) -> None: ...

            async def publish_draft(self, *, body, draft_path, visibility="PUBLIC"):
                captured["body"] = body
                return PublishResult(
                    status="published-on-linkedin",
                    author_urn="urn:li:person:ME",
                    post_id="urn:li:share:1",
                    lifecycle_state="PUBLISHED",
                    draft_path=draft_path,
                )

        monkeypatch.setattr(mcp_server, "LinkedInPublisher", FakePublisher, raising=False)
        import wiki_integrations.composio_bridge as cb

        monkeypatch.setattr(cb, "ComposioBridge", lambda *a, **k: object())

        res = json.loads(await _unwrap(publish_tool, draft_path="md.md", confirm="PUBLICAR"))
        assert res["status"] == "published-on-linkedin"
        # What was published is the flattened body — never raw markdown.
        assert "**" not in captured["body"]
        assert "𝗦𝗲𝗰𝗼𝗻𝗱 𝗕𝗿𝗮𝗶𝗻 𝗪𝗶𝗸𝗶" in captured["body"]


class TestGuards:
    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, vault) -> None:
        res = json.loads(
            await _unwrap(publish_tool, draft_path="../../etc/passwd", confirm="PUBLICAR")
        )
        assert res["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_missing_draft_rejected(self, vault) -> None:
        res = json.loads(await _unwrap(publish_tool, draft_path="does-not-exist.md"))
        assert res["status"] == "rejected"
        assert "not found" in res["error"]
