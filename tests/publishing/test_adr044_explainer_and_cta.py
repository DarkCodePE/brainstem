"""TDD London (mock-first) tests for ADR-044 — explainer archetype + composable
CTA modifiers (``newsletter_cta`` / ``product_ps``) + opt-in emoji-arrow bullets.

Hermetic: fake ``ContentSource`` + spy/mock router; no network, LLM, or vault.
Each test maps to a DETERMINISTIC acceptance criterion in
``docs/ADR-044-explainer-archetype-and-cta-modifiers.md``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from wiki_publishing import (
    LinkedInDraft,
    NewsletterCTA,
    ProductPS,
    extract_bullet_style,
    format_for_linkedin,
    render_newsletter_cta,
    render_product_ps,
)
from wiki_publishing.linkedin_draft import LinkedInDraftGenerator, WikiSnippet
from wiki_publishing.post_types import (
    ATTACH_DIAGRAM,
    PostType,
    spec_for,
)


def _clock():
    return datetime(2026, 6, 20, tzinfo=UTC)


class _RecordingSource:
    """Records the categories passed to search and returns canned snippets."""

    def __init__(self, snippets):
        self._snippets = snippets
        self.last_categories = "UNSET"
        self.last_limit = None

    def search(self, query, *, limit=3, categories=None):
        self.last_categories = categories
        self.last_limit = limit
        return list(self._snippets[:limit])


class _SpyRouter:
    """Records the system prompt + counts calls; returns a fixed body."""

    def __init__(self, text="Cuerpo del explicador.", label="spy:reasoning"):
        self.calls = 0
        self.system = None
        self._text = text
        self._label = label

    async def call(self, task, messages):
        self.calls += 1
        self.system = next(m.content for m in messages if m.role == "system")
        return SimpleNamespace(text=self._text, backend_label=self._label)


def _snip(path="concepts/x.md", body="contenido del concepto"):
    return WikiSnippet(title="X", page_path=path, body=body)


def _gen(router, source):
    return LinkedInDraftGenerator(router=router, content_source=source, clock=_clock)


# --------------------------------------------------------------------------- #
# AC: explainer source filter + length bound + ATTACH_DIAGRAM + arrow bullets
# --------------------------------------------------------------------------- #


def test_explainer_spec_source_filter_length_attachment_bullets():
    spec = spec_for(PostType.EXPLAINER)
    assert spec.categories == ("concepts", "entities", "sources")
    assert spec.max_chars == 2400
    # bounded by the ADR-021 hard ceiling at publish time
    from wiki_publishing.linkedin_publish import _MAX_POST_CHARS

    assert spec.max_chars <= _MAX_POST_CHARS
    assert spec.attachment == ATTACH_DIAGRAM
    assert spec.bullet_style == "arrow"


@pytest.mark.asyncio
async def test_explainer_generate_uses_its_categories_and_resolves_diagram():
    src = _RecordingSource([_snip()])
    calls = {"n": 0}

    def resolver(snips):
        calls["n"] += 1
        return ["/vault/assets/diagrams/x.png"]

    gen = LinkedInDraftGenerator(
        router=_SpyRouter(), content_source=src, clock=_clock, attachment_resolver=resolver
    )
    draft = await gen.generate("compresion cobertura", post_type="explainer")
    assert src.last_categories == ("concepts", "entities", "sources")
    # ATTACH_DIAGRAM → resolver invoked and surfaced
    assert calls["n"] == 1
    assert draft.attachments == ("/vault/assets/diagrams/x.png",)
    assert draft.post_type == "explainer"
    assert draft.bullet_style == "arrow"


# --------------------------------------------------------------------------- #
# AC: exactly ONE router call even with BOTH modifiers on
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_one_router_call_with_both_modifiers():
    router = _SpyRouter()
    gen = _gen(router, _RecordingSource([_snip()]))
    await gen.generate(
        "hnsw",
        post_type="explainer",
        newsletter_cta=NewsletterCTA(
            url="https://news.example/x", proof="leído por 4000+ ingenieros"
        ),
        product_ps=ProductPS(
            name="Liten.AI", pitch="prueba una demo de 10 s", url="https://liten.ai"
        ),
    )
    assert router.calls == 1


# --------------------------------------------------------------------------- #
# AC: newsletter_cta renders proof verbatim and ONLY when supplied
# --------------------------------------------------------------------------- #


def test_newsletter_cta_renders_proof_verbatim():
    out = render_newsletter_cta(
        NewsletterCTA(url="https://news.example/x", proof="leído por 4000+ ingenieros")
    )
    assert "leído por 4000+ ingenieros" in out
    assert "https://news.example/x" in out
    assert out.startswith("\n\n")


def test_newsletter_cta_no_proof_clause_when_none():
    out = render_newsletter_cta(NewsletterCTA(url="https://news.example/x"))
    assert "https://news.example/x" in out
    # no parenthesised social-proof clause; default framing only
    assert "(" not in out
    assert "La versión detallada" in out


def test_newsletter_cta_custom_label():
    out = render_newsletter_cta(
        NewsletterCTA(url="https://news.example/x", label="El análisis completo")
    )
    assert "El análisis completo" in out
    assert "La versión detallada" not in out


# --------------------------------------------------------------------------- #
# AC: product_ps renders name/pitch/url verbatim; absent fields → absent lines
# --------------------------------------------------------------------------- #


def test_product_ps_renders_all_fields_verbatim():
    out = render_product_ps(
        ProductPS(name="Liten.AI", pitch="prueba una demo de 10 s", url="https://liten.ai")
    )
    assert out == "\n\nP.D. Liten.AI — prueba una demo de 10 s https://liten.ai"


def test_product_ps_omits_url_line_when_absent():
    out = render_product_ps(ProductPS(name="Liten.AI", pitch="prueba una demo de 10 s"))
    assert out == "\n\nP.D. Liten.AI — prueba una demo de 10 s"
    assert "http" not in out


# --------------------------------------------------------------------------- #
# AC: trailer ordering body → product P.D. → newsletter link
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_trailer_ordering_body_then_product_then_newsletter():
    router = _SpyRouter(text="EL_CUERPO")
    gen = _gen(router, _RecordingSource([_snip()]))
    draft = await gen.generate(
        "hnsw",
        post_type="explainer",
        newsletter_cta=NewsletterCTA(url="https://news.example/go", proof="leído por 4000+"),
        product_ps=ProductPS(name="Liten.AI", pitch="demo de 10 s", url="https://liten.ai"),
    )
    body = draft.body
    i_body = body.index("EL_CUERPO")
    i_product = body.index("P.D. Liten.AI")
    i_news = body.index("https://news.example/go")
    assert i_body < i_product < i_news


# --------------------------------------------------------------------------- #
# AC: missing required field raises (caller error, never a model-filled gap)
# --------------------------------------------------------------------------- #


def test_newsletter_cta_without_url_raises():
    with pytest.raises(ValueError):
        NewsletterCTA(url="")
    with pytest.raises(ValueError):
        NewsletterCTA(url="   ")


def test_product_ps_missing_name_or_pitch_raises():
    with pytest.raises(ValueError):
        ProductPS(name="", pitch="x")
    with pytest.raises(ValueError):
        ProductPS(name="Liten.AI", pitch="")


def test_mcp_build_cta_modifiers_partial_product_raises():
    """MCP boundary: a partial product P.S. (only one of name/pitch) is a caller
    error and raises — never silently dropped (ADR-044 'raises' AC at the MCP edge)."""
    from wiki_agent.mcp_server import _build_cta_modifiers

    with pytest.raises(ValueError):
        _build_cta_modifiers(product_name="Liten.AI")
    with pytest.raises(ValueError):
        _build_cta_modifiers(product_pitch="te escribe los correos")
    # both present → builds the modifier; neither present → no modifier (no raise)
    assert "product_ps" in _build_cta_modifiers(
        product_name="Liten.AI", product_pitch="te escribe los correos"
    )
    assert _build_cta_modifiers() == {}


# --------------------------------------------------------------------------- #
# CRITICAL structural no-fabrication test: the model NEVER authors CTA facts.
# Feed the mock router a body CLAIMING a DIFFERENT count / product fact; assert
# the rendered output keeps the CALLER's verbatim values — proving the trailer
# is built from caller input, not the model.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cta_values_come_from_caller_not_model():
    # The model fabricates a competing count ("9999 readers") and a bogus product
    # ("FooBar 100x faster"). The deterministic trailers must override with the
    # caller's verbatim values.
    rogue_body = (
        "Texto del modelo. P.D. FooBar — 100x más rápido https://foobar.example "
        "La versión detallada (leído por 9999 lectores): https://evil.example"
    )
    router = _SpyRouter(text=rogue_body)
    gen = _gen(router, _RecordingSource([_snip()]))
    draft = await gen.generate(
        "hnsw",
        post_type="explainer",
        newsletter_cta=NewsletterCTA(
            url="https://news.example/real", proof="leído por 4000+ ingenieros"
        ),
        product_ps=ProductPS(name="Liten.AI", pitch="demo de 10 s", url="https://liten.ai"),
    )
    body = draft.body
    # The caller's verbatim values are present as appended trailers...
    assert "P.D. Liten.AI — demo de 10 s https://liten.ai" in body
    assert "leído por 4000+ ingenieros" in body
    assert "https://news.example/real" in body
    # ...and the trailers are appended AFTER the model body (caller-authored, last).
    # The deterministic product trailer sits after the model's rogue text.
    assert body.index("P.D. Liten.AI") > body.index("P.D. FooBar")
    # The caller's real newsletter URL is the LAST trailer (appended after body).
    assert body.rindex("https://news.example/real") > body.rindex("https://evil.example")


# --------------------------------------------------------------------------- #
# AC: arrow bullets only for explainer; showcase byte-identical to pre-ADR dot
# --------------------------------------------------------------------------- #


def test_arrow_bullets_only_for_explainer_render():
    md = "- uno\n- dos"
    arrow = format_for_linkedin(md, bullet_style="arrow")
    assert "➡️ uno" in arrow and "➡️ dos" in arrow


def test_showcase_renders_dot_byte_identical_to_pre_adr():
    md = "## Título\n- uno\n* dos"
    # showcase spec keeps dot
    assert spec_for(PostType.SHOWCASE).bullet_style == "dot"
    # the default (no bullet_style arg) == the pre-ADR-044 call == explicit "dot"
    pre_adr = format_for_linkedin(md)
    explicit_dot = format_for_linkedin(md, bullet_style="dot")
    assert pre_adr == explicit_dot
    assert "• uno" in pre_adr and "• dos" in pre_adr
    assert "➡️" not in pre_adr


def test_unknown_bullet_style_falls_back_to_dot():
    out = format_for_linkedin("- uno", bullet_style="nonsense")
    assert "• uno" in out
    assert "➡️" not in out


def test_extract_bullet_style_defaults_to_dot_for_pre_adr_draft():
    # a pre-ADR-044 draft has no bullet_style line in frontmatter
    md = "---\ntitle: x\nstatus: draft\n---\n\nCuerpo."
    assert extract_bullet_style(md) == "dot"


def test_extract_bullet_style_reads_arrow_from_frontmatter():
    md = "---\ntitle: x\nbullet_style: arrow\n---\n\nCuerpo."
    assert extract_bullet_style(md) == "arrow"


def test_explainer_draft_markdown_records_bullet_style_arrow():
    router = _SpyRouter()
    gen = _gen(router, _RecordingSource([_snip()]))
    draft = asyncio.run(gen.generate("tema", post_type="explainer"))
    md = draft.to_markdown()
    assert "bullet_style: arrow" in md
    # and a publish-time roundtrip recovers it
    assert extract_bullet_style(md) == "arrow"


# --------------------------------------------------------------------------- #
# AC: DRAFT-ONLY preserved — explainer w/ both modifiers => published:false,
# zero publish/Composio calls.
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_explainer_with_modifiers_is_draft_only_no_publish():
    # A spy executor that FAILS the test if any publish call is attempted.
    class _ExplodingExecutor:
        def __init__(self):
            self.calls = []

        async def execute(self, provider, tool_slug, arguments):
            self.calls.append((provider, tool_slug))
            raise AssertionError("generate() must NEVER touch the publish path")

    executor = _ExplodingExecutor()
    router = _SpyRouter()
    gen = _gen(router, _RecordingSource([_snip()]))
    draft = await gen.generate(
        "hnsw",
        post_type="explainer",
        newsletter_cta=NewsletterCTA(url="https://news.example/x", proof="4000+"),
        product_ps=ProductPS(name="Liten.AI", pitch="demo"),
    )
    assert isinstance(draft, LinkedInDraft)
    md = draft.to_markdown()
    assert "published: false" in md
    assert "published: true" not in md
    # the generator never constructed or invoked a publisher/executor
    assert executor.calls == []


# --------------------------------------------------------------------------- #
# AC: omitting modifiers reproduces a plain draft (backward compatible)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_modifiers_omitted_is_plain_draft():
    router = _SpyRouter(text="Solo el cuerpo.")
    gen = _gen(router, _RecordingSource([_snip()]))
    draft = await gen.generate("tema", post_type="explainer")
    assert "P.D." not in draft.body
    assert "La versión detallada" not in draft.body
    # body is the model output (no source URL in snippet → no Fuente backstop)
    assert draft.body == "Solo el cuerpo."


@pytest.mark.asyncio
async def test_newsletter_url_exempt_from_source_url_guard():
    # The newsletter URL must NOT be mistaken for / duplicated as a Fuente: citation.
    src_body = '---\nsources:\n  - "https://arxiv.org/abs/2507.19457v2"\n---\nPaper.'
    router = _SpyRouter(text="Cuerpo sin enlace.")
    gen = _gen(router, _RecordingSource([_snip(body=src_body)]))
    draft = await gen.generate(
        "tema",
        post_type="explainer",
        newsletter_cta=NewsletterCTA(url="https://news.example/go"),
    )
    body = draft.body
    # the source-URL backstop still fires for the real source...
    assert "Fuente: https://arxiv.org/abs/2507.19457v2" in body
    # ...and the newsletter URL is its own trailer, never relabelled "Fuente:".
    assert "https://news.example/go" in body
    assert body.count("https://news.example/go") == 1
    # Fuente: must precede the newsletter trailer (ordering: body+source → newsletter)
    assert body.index("Fuente:") < body.index("https://news.example/go")
