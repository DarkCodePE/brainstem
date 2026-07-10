"""
LinkedIn draft generator — Phase 1 of [ADR-021](../../docs/ADR-021-linkedin-publishing-flow.md).

DRAFT-ONLY. This module reads *synthesised* wiki content (read-only) and
composes a LinkedIn-shaped post **draft** via the [ADR-013] model router
(REASONING tier, ``intent="draft"``). The draft is written to a vault
``outputs/linkedin/`` file for a human to review and paste manually.

What this module deliberately does **not** do (Phase 2, separately gated by
an [ADR-017] scope amendment): call the LinkedIn API, hold a ``w_member_social``
scope, or send anything to any third party. The only network egress is the
model-router call that generates the draft text.

Design notes
------------
- The generator depends on a ``ContentSource`` *protocol*, not a concrete
  search — so it unit-tests against a fake and never touches the filesystem
  or network in tests (TDD London / mock-first per CLAUDE.md).
- Drafting is quality-sensitive (it composes under the user's own
  professional identity), so it routes to the REASONING tier via
  ``intent="draft"`` — added to ``wiki_routing.policy`` for this feature.
- The system prompt forbids verbatim re-posting of captured third-party
  content (ADR-021 Risk #5, plagiarism) and forbids fabricated facts/quotes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from wiki_publishing.post_types import (
    ATTACH_NONE,
    LEAVE_ROOM_FOR_CLOSER,
    LEAVE_ROOM_FOR_PS,
    Focus,
    NewsletterCTA,
    PostType,
    ProductPS,
    coerce_focus,
    focus_overlay,
    render_newsletter_cta,
    render_product_ps,
    spec_for,
)
from wiki_routing import Message, ModelRouter, TaskDescriptor

# LinkedIn's hard limit for a member post body is 3000 characters; we aim
# under it and let the human trim. Used only as prompt guidance.
_MAX_POST_CHARS = 2800

_SYSTEM_PROMPT = (
    "You are a writing assistant that drafts a LinkedIn post for the user to "
    "review and publish themselves. You compose in the user's own voice from "
    "the WIKI MATERIAL provided below.\n\n"
    "Language & tone:\n"
    "- Write the ENTIRE post in Spanish (neutral Latin American Spanish). Do NOT "
    "mix English into the prose. Keep only established technical terms that are "
    "normally left in English (e.g. prompt, embeddings, RAG, framework, chunk).\n"
    "- Tone: 'profesional cercano' — first person, an expert sharing what they "
    "learned, conversational but credible. Not academic, not hypey, no clickbait.\n\n"
    "Hard rules:\n"
    "1. This is a DRAFT for human review — never assume it will be posted as-is.\n"
    "2. Do NOT copy any captured third-party post verbatim. Synthesise the ideas "
    "in original wording. If you quote, attribute explicitly.\n"
    "3. Do NOT invent facts, statistics, names, or quotes. Use only what the wiki "
    "material supports; if the material is thin, write a shorter post.\n"
    f"4. Keep the body under {_MAX_POST_CHARS} characters. No hashtag spam "
    "(0-3 relevant hashtags max).\n"
    "5. If a 'FUENTES (URLs)' list is provided below, include the most relevant "
    "source URL(s) in the post so readers can go deeper — e.g. a closing line "
    "'Fuente: <url>' or woven inline. LinkedIn turns a plain URL into a link. "
    "Use the URLs verbatim WITH their https:// scheme; never invent, shorten, or "
    "alter them, and never build a repository URL from a person's or author's "
    "name. Prefer 1-2 links max; if no URLs are listed, don't fabricate any.\n"
    "6. Return ONLY the post body text — no preamble, no surrounding quotes, no "
    "'Here is your post' framing."
)

# URLs in wiki material (frontmatter `sources:` or inline). Trailing
# punctuation is trimmed so a URL at the end of a sentence stays clean.
_URL_RE = re.compile(r"https?://[^\s)\]\"'>]+")

# Generic / non-permalink URLs that are useless as a citation: a bare domain
# (scheme://host with no path) or a social feed root (a bad clip that didn't
# capture the real permalink, e.g. https://www.linkedin.com/feed/).
_GENERIC_URL_RE = re.compile(
    r"(?:linkedin\.com|x\.com|twitter\.com)/feed/?$|^https?://[^/]+/?$", re.I
)


def _is_generic_url(url: str) -> bool:
    return bool(_GENERIC_URL_RE.search(url))


def _source_urls(snippets: Sequence[WikiSnippet]) -> list[str]:
    """Collect unique, citable http(s) URLs from the snippets (order-preserving).

    Pulls from the whole snippet body — frontmatter ``sources:`` and any
    inline links (e.g. an arXiv URL in prose) — so the composer can cite the
    real source instead of paraphrasing it away. Generic/non-permalink URLs
    (bare domains, social feed roots) are filtered out."""
    seen: dict[str, None] = {}
    for s in snippets:
        for raw in _URL_RE.findall(s.body):
            url = raw.rstrip(".,;:!?")
            if url not in seen and not _is_generic_url(url):
                seen[url] = None
    return list(seen)


class EmptyContentError(RuntimeError):
    """Raised when no wiki content matches the topic, so there is nothing to
    draft from. The router is never called — drafting from zero sources would
    invite fabrication (ADR-021 Risk #3/#5)."""


@dataclass(frozen=True, slots=True)
class WikiSnippet:
    """A read-only slice of synthesised wiki content used as draft material."""

    title: str
    page_path: str
    """Path of the source wiki page, relative to the wiki root. Used for
    attribution in the draft's Sources section — never published."""
    body: str


@runtime_checkable
class ContentSource(Protocol):
    """Read-only provider of wiki snippets relevant to a topic.

    Implementations MUST NOT mutate the vault or call any external write
    surface — content selection for drafting is strictly read-only."""

    def search(
        self, query: str, *, limit: int = 3, categories: Sequence[str] | None = None
    ) -> list[WikiSnippet]:
        """Return up to ``limit`` snippets most relevant to ``query``.

        ``categories`` optionally restricts results to wiki page-path category
        segments (e.g. ``("sources",)`` or ``("concepts", "entities")``) so a
        ``post_type`` can bias which content it draws from (ADR-024). ``None`` =
        no restriction (backward-compatible)."""
        ...


@dataclass(frozen=True, slots=True)
class LinkedInDraft:
    """A generated LinkedIn post draft. Holds the body plus provenance so the
    human reviewer (and any audit) can see what it was built from and how."""

    topic: str
    body: str
    sources: tuple[WikiSnippet, ...]
    model_label: str
    """Which router backend produced the text (e.g. ``openrouter:deepseek-...``)."""
    created_at: str
    """ISO-8601 UTC timestamp."""
    attachments: tuple[str, ...] = ()
    """Local image paths (e.g. a rendered architecture diagram PNG) the human
    should ATTACH MANUALLY in LinkedIn's composer at post time. NOT part of the
    post text — LinkedIn renders only plain text, and automated member-image
    upload needs a self-hosted LinkedIn app (ADR-023: deferred). Surfaced in the
    reviewable draft so the 1-click drag-and-drop is obvious."""
    post_type: str = "repo_deep_dive"
    """The ADR-024 archetype this draft was composed as (provenance)."""
    focus: str = "code"
    """The ADR-024 A1 lens this draft used (``use`` or ``code``)."""
    bullet_style: str = "dot"
    """The ADR-044 publish-time bullet render style (``dot`` default → ``• ``;
    ``arrow`` → ``➡️ ``). Recorded here (and in ``to_markdown`` frontmatter) so the
    publish path can honour it; the 7 pre-ADR-044 archetypes are always ``dot``."""

    def to_markdown(self) -> str:
        """Render the draft as a reviewable markdown file.

        Frontmatter is explicit that this is an unpublished draft — there is
        no ``published: true`` path in Phase 1."""
        source_lines = "\n".join(f'  - "{s.page_path}"' for s in self.sources) or "  []"
        source_refs = "\n".join(f"- [[{s.page_path}]] — {s.title}" for s in self.sources)
        return (
            "---\n"
            f'title: "LinkedIn draft: {self.topic}"\n'
            f"date: {self.created_at[:10]}\n"
            f"created_at: {self.created_at}\n"
            "status: draft\n"
            "published: false\n"
            "platform: linkedin\n"
            f"post_type: {self.post_type}\n"
            f"focus: {self.focus}\n"
            f"bullet_style: {self.bullet_style}\n"
            "category: drafts\n"
            "tags: [linkedin-draft, draft, generated]\n"
            f"generated_by: sbw-linkedin-draft (ADR-021 Phase 1)\n"
            f"model: {self.model_label}\n"
            "sources:\n"
            f"{source_lines}\n"
            "---\n\n"
            f"# LinkedIn draft — {self.topic}\n\n"
            "> Unpublished draft. Review, edit, then post manually. "
            "SBW does not publish in Phase 1 (ADR-021).\n\n"
            f"{self.body.strip()}\n\n"
            f"{self._attachments_block()}"
            "## Sources (not part of the post)\n\n"
            f"{source_refs or '_none_'}\n"
        )

    def _attachments_block(self) -> str:
        """A prominent, reviewer-facing block listing images to attach by hand."""
        if not self.attachments:
            return ""
        items = "\n".join(f"- `{a}`" for a in self.attachments)
        return (
            "## 📎 Imagen para adjuntar manualmente\n\n"
            "> Al publicar en LinkedIn, **adjunta esta imagen** arrastrándola al "
            "editor. NO es parte del texto del post.\n\n"
            f"{items}\n\n"
        )


class LinkedInDraftGenerator:
    """Composes a LinkedIn post draft from synthesised wiki content.

    Parameters
    ----------
    router:
        Live ``ModelRouter``. Drafting dispatches ``intent="draft"`` which the
        routing policy maps to the REASONING tier.
    content_source:
        Read-only source of wiki snippets. Injected so tests use a fake.
    clock:
        Returns the current UTC time. Injectable for deterministic tests.
    """

    def __init__(
        self,
        *,
        router: ModelRouter,
        content_source: ContentSource,
        clock: Callable[[], datetime] | None = None,
        attachment_resolver: Callable[[Sequence[WikiSnippet]], list[str]] | None = None,
    ) -> None:
        self._router = router
        self._content_source = content_source
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        # Resolves images (e.g. a repo's rendered diagram PNG) the human should
        # attach by hand. Injected so it stays filesystem-free in unit tests.
        self._attachment_resolver = attachment_resolver

    async def generate(
        self,
        topic: str,
        *,
        max_sources: int = 3,
        post_type: PostType | str | None = None,
        focus: Focus | str | None = None,
        newsletter_cta: NewsletterCTA | None = None,
        product_ps: ProductPS | None = None,
    ) -> LinkedInDraft:
        """Select wiki content for ``topic`` and compose a draft of ``post_type``.

        ``post_type`` (ADR-024) selects the archetype — it biases the source
        search to the type's wiki categories and overlays a per-type structure
        on the shared base voice. Defaults to ``repo_deep_dive``.

        ``focus`` (ADR-024 A1) is the lens: ``use`` (user/value) or ``code``
        (internals). ``None`` falls back to the post type's default focus
        (``showcase``→use, ``repo_deep_dive``→code).

        ``newsletter_cta`` / ``product_ps`` (ADR-044) are optional, composable
        CTA modifiers orthogonal to ``post_type``/``focus``. When set, they add a
        short overlay HINT (the model writes NO CTA copy) and, after the model
        body + the existing source-URL guard, append a DETERMINISTIC trailer built
        from the caller-supplied dataclass values — so a subscriber count or a
        product claim is never model-authored. Still exactly ONE router call. The
        newsletter URL is promotion, not a source citation, and is exempt from the
        generic-URL filter / source-URL guard. Order: body → product P.D. →
        newsletter/go-deeper link.

        Raises ``EmptyContentError`` when no content matches — the router is
        not called, so a no-source topic can never produce a fabricated post.
        """
        topic = topic.strip()
        if not topic:
            raise ValueError("topic must be non-empty")

        spec = spec_for(post_type)
        snippets = self._content_source.search(
            topic, limit=max_sources, categories=spec.categories or None
        )
        if not snippets:
            raise EmptyContentError(
                f"no wiki content matched topic {topic!r} for post_type "
                f"{spec.post_type.value!r}; nothing to draft from"
            )

        user_prompt = _render_user_prompt(topic, snippets)
        estimated_tokens = max(1, len(user_prompt) // 4)
        task = TaskDescriptor(
            intent="draft",
            estimated_input_tokens=estimated_tokens,
            has_image=False,
            caller_priority="foreground",
        )
        resolved_focus = coerce_focus(focus) or spec.default_focus
        # ADR-044: when a CTA modifier is active, hint the model to leave room for
        # a closing block — it writes NO CTA copy; the deterministic trailer below
        # renders the factual values. Still ONE router call.
        modifier_hints = ""
        if product_ps is not None:
            modifier_hints += f"\n\n{LEAVE_ROOM_FOR_PS}"
        if newsletter_cta is not None:
            modifier_hints += f"\n\n{LEAVE_ROOM_FOR_CLOSER}"
        system_prompt = (
            f"{_SYSTEM_PROMPT}\n\n{spec.system_overlay}\n\n"
            f"{focus_overlay(resolved_focus)}\n\n"
            f"Largo objetivo: hasta ~{spec.max_chars} caracteres."
            f"{modifier_hints}"
        )
        result = await self._router.call(
            task,
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
        )
        # Deterministic source-URL guarantee: the system prompt asks the model
        # to cite the source URL, but model compliance varies (it sometimes
        # paraphrases "disponible en arXiv" instead of pasting the link). If no
        # citable source URL made it into the body, append the primary one so
        # the link is never silently dropped (see project_draft_url_image_loss).
        body = result.text.strip()
        urls = _source_urls(snippets)
        if urls and not any(u in body for u in urls):
            body = f"{body}\n\nFuente: {urls[0]}"

        # ADR-044 deterministic CTA trailers — appended AFTER the source-URL guard
        # so the newsletter URL is never seen by _source_urls/_is_generic_url (it
        # is promotion, not a citation). Built from caller-supplied dataclass values
        # ONLY — the model never authors the count/name/pitch/url. Order per ADR:
        # body → product P.D. → newsletter/go-deeper link.
        if product_ps is not None:
            body = f"{body}{render_product_ps(product_ps)}"
        if newsletter_cta is not None:
            body = f"{body}{render_newsletter_cta(newsletter_cta)}"

        # Attachments only for archetypes whose policy wants an image (e.g.
        # deep-dive / showcase → the diagram PNG). Tutorial/informativo: none.
        attachments: tuple[str, ...] = ()
        if spec.attachment != ATTACH_NONE and self._attachment_resolver is not None:
            try:
                attachments = tuple(self._attachment_resolver(snippets))
            except Exception:  # noqa: BLE001 — attachments are best-effort, never block a draft
                attachments = ()

        return LinkedInDraft(
            topic=topic,
            body=body,
            sources=tuple(snippets),
            model_label=result.backend_label,
            created_at=self._clock().isoformat(),
            attachments=attachments,
            post_type=spec.post_type.value,
            focus=resolved_focus.value,
            bullet_style=spec.bullet_style,
        )


def _render_user_prompt(topic: str, snippets: Sequence[WikiSnippet]) -> str:
    blocks = []
    for i, s in enumerate(snippets, start=1):
        blocks.append(f"### Source {i}: {s.title} ({s.page_path})\n{s.body.strip()}")
    material = "\n\n".join(blocks)
    urls = _source_urls(snippets)
    urls_block = ""
    if urls:
        listed = "\n".join(f"- {u}" for u in urls)
        urls_block = (
            "\n\n=== FUENTES (URLs) ===\n"
            "Incluye la(s) URL(s) mas relevante(s) en el post (verbatim):\n"
            f"{listed}\n"
            "=== END FUENTES ==="
        )
    return (
        f"Topic for the LinkedIn post: {topic}\n\n"
        "Compose the post using ONLY the wiki material below. Synthesise — do "
        "not copy any source verbatim.\n\n"
        "=== WIKI MATERIAL ===\n"
        f"{material}\n"
        "=== END WIKI MATERIAL ==="
        f"{urls_block}"
    )


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s or "untitled")[:60]


def write_draft(draft: LinkedInDraft, *, wiki_root: Path) -> Path:
    """Persist ``draft`` to ``<wiki_root>/outputs/linkedin/`` and return the path.

    This is the only filesystem write in Phase 1 — a draft inside the vault,
    never an external publish."""
    out_dir = Path(wiki_root) / "outputs" / "linkedin"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"linkedin-draft-{_slug(draft.topic)}-{draft.created_at[:10]}.md"
    path = out_dir / filename
    path.write_text(draft.to_markdown(), encoding="utf-8")
    return path
