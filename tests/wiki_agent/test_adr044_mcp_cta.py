"""ADR-044 — MCP-boundary coverage for the CTA modifiers + draft tool wiring.

The drafter-level behaviour is covered in ``tests/publishing``; here we cover the
thin MCP surface that ``tests/publishing`` cannot reach: ``_build_cta_modifiers``
(the plain-string → dataclass assembler, incl. the "partial product P.S. raises"
acceptance criterion) and that a draft tool forwards the assembled modifiers into
``generate``. Strategy mirrors ``test_mcp_linkedin_draft``: inject a fake
generator, point ``WIKI_ROOT`` at a temp vault, await the tool's ``.fn``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wiki_agent import mcp_server
from wiki_publishing import LinkedInDraft, NewsletterCTA, ProductPS, WikiSnippet

# --------------------------------------------------------------------------- #
# _build_cta_modifiers (pure)
# --------------------------------------------------------------------------- #


def test_newsletter_only_builds_cta_verbatim() -> None:
    mods = mcp_server._build_cta_modifiers(
        newsletter_url="https://news.example/sub", newsletter_proof="leído por 4000+"
    )
    assert "product_ps" not in mods
    cta = mods["newsletter_cta"]
    assert isinstance(cta, NewsletterCTA)
    assert cta.url == "https://news.example/sub"
    assert cta.proof == "leído por 4000+"  # verbatim


def test_product_only_builds_ps_verbatim() -> None:
    mods = mcp_server._build_cta_modifiers(
        product_name="Headroom", product_pitch="context that fits", product_url="https://h.dev"
    )
    assert "newsletter_cta" not in mods
    ps = mods["product_ps"]
    assert isinstance(ps, ProductPS)
    assert (ps.name, ps.pitch, ps.url) == ("Headroom", "context that fits", "https://h.dev")


def test_both_modifiers_compose() -> None:
    mods = mcp_server._build_cta_modifiers(
        newsletter_url="https://n", product_name="P", product_pitch="x"
    )
    assert set(mods) == {"newsletter_cta", "product_ps"}


def test_empty_args_yield_no_modifiers() -> None:
    assert mcp_server._build_cta_modifiers() == {}


@pytest.mark.parametrize("kw", [{"product_name": "P"}, {"product_pitch": "x"}])
def test_partial_product_raises(kw: dict) -> None:
    """ADR-044 AC: a partial product P.S. is a caller error, never silently dropped."""
    with pytest.raises(ValueError):
        mcp_server._build_cta_modifiers(**kw)


# --------------------------------------------------------------------------- #
# Draft tool forwards modifiers into generate (MCP wiring)
# --------------------------------------------------------------------------- #


class _FakeGen:
    def __init__(self, draft: LinkedInDraft) -> None:
        self._draft = draft
        self.kwargs: dict = {}

    async def generate(self, topic: str, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs
        return self._draft


def _draft() -> LinkedInDraft:
    return LinkedInDraft(
        topic="acme/widget",
        body="An explainer post.",
        sources=(WikiSnippet(title="widget", page_path="sources/w.md", body="x"),),
        model_label="stub",
        created_at="2026-06-22T00:00:00+00:00",
    )


async def _unwrap(tool, **kwargs):
    fn = tool.fn if hasattr(tool, "fn") else tool
    return await fn(**kwargs)


@pytest.mark.asyncio
async def test_draft_from_repo_forwards_modifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeGen(_draft())
    monkeypatch.setattr(mcp_server, "_get_repo_context_generator", lambda _url: fake)
    monkeypatch.setattr(
        mcp_server, "_resolve_repo_input", lambda u: "https://github.com/acme/widget"
    )
    monkeypatch.setattr(mcp_server, "WIKI_ROOT", str(tmp_path))

    out = await _unwrap(
        mcp_server.linkedin_draft_from_repo,
        url="acme/widget",
        post_type="explainer",
        newsletter_url="https://news.example/sub",
        newsletter_proof="leído por 4000+",
        product_name="Headroom",
        product_pitch="context that fits",
    )

    # the assembled modifiers reached generate (lines _build_cta_modifiers + generate)
    assert isinstance(fake.kwargs.get("newsletter_cta"), NewsletterCTA)
    assert isinstance(fake.kwargs.get("product_ps"), ProductPS)
    payload = json.loads(out)
    assert "draft" in payload["status"]  # draft-only (unpublished)
    assert (tmp_path / "outputs" / "linkedin").exists()
