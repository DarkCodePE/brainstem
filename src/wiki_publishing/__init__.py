"""
``wiki_publishing`` — outbound content-repurposing surface for SBW.

Implements **Phase 1** of [ADR-021 LinkedIn publishing flow](../../docs/ADR-021-linkedin-publishing-flow.md):
*draft-only*. SBW reads its own synthesised wiki content (read-only) and
composes a LinkedIn-shaped post **draft** via the [ADR-013] model router;
the draft lands in a vault ``outputs/linkedin/`` file for the human to
review and paste. **Nothing is published** — there is no LinkedIn API call,
no OAuth write scope, and no network egress beyond the model router.

Phase 2a (gated live publish via Composio) lives in ``linkedin_publish``:
the ``LinkedInPublisher`` write path, invoked by the MCP tool only after the
chat HITL typed-confirm gate. See [ADR-021].

Public surface:

- ``WikiSnippet`` — a read-only slice of synthesised wiki content.
- ``ContentSource`` — Protocol: ``search(query, *, limit) -> list[WikiSnippet]``.
- ``LinkedInDraft`` — the generated draft (body + sources + provenance).
- ``LinkedInDraftGenerator`` — composes a draft from wiki content via the router.
- ``EmptyContentError`` — raised when no source content matches the topic.
- ``write_draft`` — persist a draft to ``<wiki_root>/outputs/linkedin/``.
- ``WikiContentSource`` — read-only ``ContentSource`` over ``wiki/index.md``.
- ``RepoContextSource`` — live-fetch ``ContentSource`` over one GitHub repo (ADR-025).
- ``LinkedInPublisher`` — Phase-2a live publish via a ``PostExecutor`` (Composio).
- ``PublishError`` / ``PublishResult`` / ``PostExecutor`` — publish path types.
- ``extract_post_body`` / ``format_for_linkedin`` / ``extract_attachments`` —
  draft-markdown → post-body helpers used by the publish tool.
"""

from __future__ import annotations

from wiki_publishing.linkedin_draft import (
    ContentSource,
    EmptyContentError,
    LinkedInDraft,
    LinkedInDraftGenerator,
    WikiSnippet,
    write_draft,
)
from wiki_publishing.linkedin_publish import (
    LinkedInPublisher,
    PostExecutor,
    PublishError,
    PublishResult,
    extract_attachments,
    extract_bullet_style,
    extract_post_body,
    format_for_linkedin,
)
from wiki_publishing.post_types import (
    Focus,
    NewsletterCTA,
    PostType,
    PostTypeSpec,
    ProductPS,
    render_newsletter_cta,
    render_product_ps,
    spec_for,
)
from wiki_publishing.project_context_source import ProjectContextSource
from wiki_publishing.project_ledger import ProjectLedger
from wiki_publishing.repo_context_source import RepoContextSource
from wiki_publishing.web_context_source import WebContextSource, WebFetchError, fetch_url_markdown
from wiki_publishing.wiki_content_source import WikiContentSource

__all__ = [
    "ContentSource",
    "EmptyContentError",
    "Focus",
    "LinkedInDraft",
    "LinkedInDraftGenerator",
    "LinkedInPublisher",
    "NewsletterCTA",
    "PostExecutor",
    "PostType",
    "PostTypeSpec",
    "ProductPS",
    "ProjectContextSource",
    "ProjectLedger",
    "PublishError",
    "PublishResult",
    "RepoContextSource",
    "WebContextSource",
    "WebFetchError",
    "WikiContentSource",
    "WikiSnippet",
    "extract_attachments",
    "extract_bullet_style",
    "extract_post_body",
    "fetch_url_markdown",
    "format_for_linkedin",
    "render_newsletter_cta",
    "render_product_ps",
    "spec_for",
    "write_draft",
]
