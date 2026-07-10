"""
LinkedIn publisher — Phase 2a of [ADR-021](../../docs/ADR-021-linkedin-publishing-flow.md).

Publishes a *reviewed* post to the user's own LinkedIn via the
Composio-brokered write surface, under the chat HITL typed-confirm gate.

This is the project's first **write** path, so it is deliberately narrow:

- Single-tenant ([ADR-018]): the post author is always the connected user's
  own ``urn:li:person`` (resolved via ``LINKEDIN_GET_MY_INFO``). Org-page
  targets are not supported.
- Scope is read from the locked table (``policy_for("linkedin")``) — the
  ``w_member_social`` puncture is gated by the [ADR-017] amendment + lock test.
- Phase 2b: ``lifecycle_state`` defaults to ``"PUBLISHED"`` (a LIVE post).
  An API ``"DRAFT"`` is orphaned (no post id returned) and invisible in
  LinkedIn's composer UI, so it cannot be promoted to published — the
  reviewable artifact is the local vault draft, and the typed-confirm is the
  publish gate. ``"DRAFT"`` remains available as a non-default option.
- The HITL typed-confirm gate lives in the MCP tool (the chat-facing seam);
  this module is the mechanism it calls only *after* approval.

Mock-first: the executor is a ``PostExecutor`` protocol (the relevant slice
of ``ComposioBridge``), so unit tests inject a fake — no network, no creds.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from wiki_core.secrets import policy_for

_PROVIDER = "linkedin"
_GET_INFO_TOOL = "LINKEDIN_GET_MY_INFO"
_CREATE_POST_TOOL = "LINKEDIN_CREATE_LINKED_IN_POST"
# Mirror the draft generator's cap; LinkedIn's hard member-post limit is 3000.
_MAX_POST_CHARS = 2800
_URN_RE = re.compile(r"urn:li:person:[A-Za-z0-9_-]+")


class PublishError(RuntimeError):
    """Raised when a draft cannot be published — empty/oversized body, an
    unresolvable author URN, a stub-mode executor (no real credentials), or a
    backend error. The caller preserves the local draft and surfaces this."""


@runtime_checkable
class PostExecutor(Protocol):
    """The minimal write surface this module needs — the ``execute`` slice of
    ``ComposioBridge``. Injected so tests use a fake.

    ``stub_mode`` is read to refuse a no-op "success": a bridge with no API
    key returns canned stubs, which for a *publish* path would be a dangerous
    silent no-post. Implementations that cannot be in stub mode may omit it."""

    stub_mode: bool

    async def execute(
        self, provider: str, tool_slug: str, arguments: dict[str, Any]
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class PublishResult:
    """Outcome of a successful native-draft creation on LinkedIn."""

    status: str
    """``"published-on-linkedin"`` (live) or ``"draft-created-on-linkedin"``."""
    author_urn: str
    post_id: str | None
    """LinkedIn URN/id of the created post, if the API returned one."""
    lifecycle_state: str
    """``"PUBLISHED"`` (Phase 2b default, live) or ``"DRAFT"``."""
    draft_path: str
    """The local vault draft this was created from (provenance)."""


# Trailing sections that are scaffolding for the human, never part of the
# published text. The attachments block carries a LOCAL image PATH — cutting it
# is what stops that path from leaking into a live post.
_TRAILER_MARKERS = ("\n## 📎", "\n## Sources")
_ATTACH_HEADING = "## 📎"


def extract_post_body(markdown: str) -> str:
    """Pull the publishable post body out of a saved draft ``.md``.

    Drops the YAML frontmatter, the ``# LinkedIn draft — …`` H1, the
    ``> Unpublished draft`` reviewer note, and every trailing scaffolding
    section: ``## 📎 Imagen para adjuntar manualmente`` (a local file path that
    must never reach a live post) and ``## Sources (not part of the post)``."""
    text = markdown
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    # Cut at the earliest trailing scaffolding section (attachments or sources).
    cut = len(text)
    for marker in _TRAILER_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    text = text[:cut]
    lines = [
        ln
        for ln in text.splitlines()
        if not ln.strip().startswith("# ") and not ln.strip().startswith(">")
    ]
    return "\n".join(lines).strip()


_BULLET_STYLE_RE = re.compile(r"^bullet_style:\s*(\S+)\s*$", re.MULTILINE)


def extract_bullet_style(markdown: str) -> str:
    """Recover the ADR-044 ``bullet_style`` from a saved draft's frontmatter.

    ``extract_post_body`` strips ALL frontmatter, so the publish path reads this
    separately to honour ``arrow`` bullets. Returns ``"dot"`` when the draft has
    no ``bullet_style:`` line (every pre-ADR-044 draft), keeping output unchanged."""
    if markdown.startswith("---"):
        end = markdown.find("\n---", 3)
        if end != -1:
            m = _BULLET_STYLE_RE.search(markdown[: end + 1])
            if m:
                return m.group(1)
    return "dot"


# --------------------------------------------------------------------------- #
# Markdown → LinkedIn-ready text
# --------------------------------------------------------------------------- #
# LinkedIn renders NO markdown, so literal ``**``/`` ` ``/``#`` leak into the
# published post. We map ``**bold**`` to the Unicode "Mathematical Sans-Serif
# Bold" block — glyphs LinkedIn DOES display as bold — and clean the rest.
_BOLD_UPPER_BASE = 0x1D5D4  # 𝗔
_BOLD_LOWER_BASE = 0x1D5EE  # 𝗮
_BOLD_DIGIT_BASE = 0x1D7EC  # 𝟬
# Mathematical Sans-Serif Italic — for single-``*`` emphasis (no digit glyphs).
_ITALIC_UPPER_BASE = 0x1D608  # 𝘈
_ITALIC_LOWER_BASE = 0x1D622  # 𝘢

_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")  # no DOTALL: stay on one line
# Single-``*`` italic, run AFTER the bold pass so ``**x**`` is already consumed.
# Requires non-space just inside the markers so a ``* `` bullet never matches.
_MD_ITALIC_RE = re.compile(r"\*(?=\S)([^*\n]+?)(?<=\S)\*")
_MD_CODE_RE = re.compile(r"`([^`]+)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+")

# A scheme-less URL (e.g. ``github.com/org/repo``) that the composer sometimes
# emits: LinkedIn only auto-links — and shows as a verified link — URLs that
# carry an explicit scheme, so a bare ``host.tld/path`` renders as dead,
# "unverified" plain text. Match a host/path token NOT already part of a URL
# (not preceded by ``//``, ``@``, ``.`` or a word char) and prefix ``https://``.
# Requires a real dotted TLD + a ``/path`` so file paths (``wiki/lessons/``),
# model ids (``deepseek/v4``) and identifiers never match.
_BARE_URL_RE = re.compile(
    r"(?<![\w/@.])((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}/[^\s)]+)",
    re.IGNORECASE,
)


def _ensure_url_scheme(text: str) -> str:
    """Prefix ``https://`` to scheme-less URLs so LinkedIn renders them as
    verified, clickable links instead of dead plain text."""
    return _BARE_URL_RE.sub(lambda m: f"https://{m.group(1)}", text)


def _to_unicode_bold(text: str) -> str:
    """Map ASCII letters/digits in ``text`` to Unicode sans-serif bold glyphs.

    Non-ASCII-alphanumeric characters (spaces, %, accents like ``í``) pass
    through unchanged — they have no glyph in the bold block."""
    out: list[str] = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr(_BOLD_UPPER_BASE + (ord(ch) - ord("A"))))
        elif "a" <= ch <= "z":
            out.append(chr(_BOLD_LOWER_BASE + (ord(ch) - ord("a"))))
        elif "0" <= ch <= "9":
            out.append(chr(_BOLD_DIGIT_BASE + (ord(ch) - ord("0"))))
        else:
            out.append(ch)
    return "".join(out)


def _to_unicode_italic(text: str) -> str:
    """Map ASCII letters in ``text`` to Unicode sans-serif italic glyphs.

    Digits and non-letters (the italic block has no digit glyphs) pass through."""
    out: list[str] = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr(_ITALIC_UPPER_BASE + (ord(ch) - ord("A"))))
        elif "a" <= ch <= "z":
            out.append(chr(_ITALIC_LOWER_BASE + (ord(ch) - ord("a"))))
        else:
            out.append(ch)
    return "".join(out)


# Bullet markers by style (ADR-044). ``dot`` is the default and keeps the exact
# ``• `` of every pre-ADR-044 archetype (byte-identical); ``arrow`` opts the
# ``explainer`` into a friendlier ``➡️ ``. Unknown styles fall back to ``• ``.
_BULLET_MARKERS = {"dot": "• ", "arrow": "➡️ "}


def format_for_linkedin(text: str, bullet_style: str = "dot") -> str:
    """Render a markdown post body as LinkedIn-ready text.

    LinkedIn shows markdown source literally, so this:
    - converts ``**bold**`` / ``__bold__`` to Unicode bold glyphs (visible bold);
    - converts single-``*`` ``*italic*`` to Unicode italic glyphs;
    - unwraps inline ``code`` (drops the backticks LinkedIn would show);
    - turns ``[text](url)`` into ``text (url)``;
    - converts ``- ``/``* `` bullets to ``• `` (or ``➡️ `` when
      ``bullet_style="arrow"``, ADR-044 — default ``dot`` is byte-identical to
      the pre-ADR-044 renderer);
    - strips heading ``#`` markers, keeping the heading text;
    - prefixes ``https://`` to scheme-less URLs so LinkedIn links them.
    """
    bullet = _BULLET_MARKERS.get(bullet_style, _BULLET_MARKERS["dot"])
    # Links first, so any ** inside link text is handled by the bold pass.
    text = _MD_LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    text = _MD_BOLD_RE.sub(lambda m: _to_unicode_bold(m.group(1) or m.group(2)), text)
    # Italic AFTER bold, so ``**x**`` is already consumed and only single ``*`` remain.
    text = _MD_ITALIC_RE.sub(lambda m: _to_unicode_italic(m.group(1)), text)
    text = _MD_CODE_RE.sub(lambda m: m.group(1), text)

    lines: list[str] = []
    for ln in text.split("\n"):
        stripped = ln.lstrip()
        indent = ln[: len(ln) - len(stripped)]
        if stripped.startswith(("- ", "* ")):
            lines.append(f"{indent}{bullet}{stripped[2:]}")
        elif _MD_HEADING_RE.match(stripped):
            lines.append(indent + _MD_HEADING_RE.sub("", stripped))
        else:
            lines.append(ln)
    return _ensure_url_scheme("\n".join(lines))


def extract_attachments(markdown: str) -> list[str]:
    """Return the suggested image attachment path(s) from a draft's
    ``## 📎 Imagen para adjuntar manualmente`` block.

    The block lists each path as a ``- `<path>``` bullet. Returns ``[]`` when
    the draft has no attachments block. Used by the MANUAL (post-it-yourself)
    publish path so the human knows which image to drag into LinkedIn's editor."""
    start = markdown.find(_ATTACH_HEADING)
    if start == -1:
        return []
    section = markdown[start:]
    nxt = section.find("\n## ", len(_ATTACH_HEADING))
    if nxt != -1:
        section = section[:nxt]
    paths: list[str] = []
    for ln in section.splitlines():
        s = ln.strip()
        if s.startswith("-"):
            s = s.lstrip("-").strip().strip("`").strip()
            if s:
                paths.append(s)
    return paths


def _author_urn_from_info(info: dict[str, Any]) -> str:
    """Resolve the member's ``urn:li:person:<id>`` from a GET_MY_INFO response.

    Handles both Composio toolkit shapes:
    - old pin (``00000000_00``): ``{"response_dict": {"author_id":
      "urn:li:person:..", "sub": ".."}}`` — full URN / OIDC ``sub``.
    - ``latest``: ``{"id": "<bare-id>", "localizedFirstName": ..}`` — bare
      member id with no ``urn:`` prefix and no ``sub``.

    Raises ``PublishError`` if no id can be found."""
    blob = json.dumps(info)
    m = _URN_RE.search(blob)
    if m:
        return m.group(0)
    # Fall back to a bare id from `sub` (OIDC) or `id` (latest profile shape),
    # at the top level or nested under `response_dict`.
    rd = info.get("response_dict")
    rd = rd if isinstance(rd, dict) else {}
    member_id = info.get("sub") or info.get("id") or rd.get("sub") or rd.get("id")
    if member_id:
        member_id = str(member_id)
        return member_id if member_id.startswith("urn:li:person:") else f"urn:li:person:{member_id}"
    raise PublishError("could not resolve author URN from LINKEDIN_GET_MY_INFO response")


def _extract_post_id(result: dict[str, Any]) -> str | None:
    """Best-effort pull of the created post's id/URN from the create response."""
    for key in ("id", "post_id", "urn", "activity"):
        val = result.get(key)
        if isinstance(val, str) and val:
            return val
    rd = result.get("response_dict")
    if isinstance(rd, dict):
        for key in ("id", "post_id", "urn"):
            val = rd.get(key)
            if isinstance(val, str) and val:
                return val
    return None


class LinkedInPublisher:
    """Creates a native LinkedIn DRAFT from a reviewed post body.

    Parameters
    ----------
    executor:
        A ``PostExecutor`` (production: a live ``ComposioBridge``). Must not be
        in stub mode — a publish must reach the real API or fail loudly.
    """

    def __init__(self, *, executor: PostExecutor) -> None:
        self._executor = executor
        # Reading the policy asserts `linkedin` exists in the locked scope
        # table — i.e. the ADR-017 amendment is in place. KeyError otherwise.
        self._scope = policy_for(_PROVIDER)

    async def resolve_author_urn(self) -> str:
        """Resolve the connected user's own ``urn:li:person`` (single-tenant)."""
        info = await self._executor.execute(_PROVIDER, _GET_INFO_TOOL, {})
        return _author_urn_from_info(info)

    async def publish_draft(
        self,
        *,
        body: str,
        draft_path: str,
        visibility: str = "PUBLIC",
        lifecycle_state: str = "PUBLISHED",
    ) -> PublishResult:
        """Publish ``body`` to the user's own LinkedIn under their URN.

        ADR-021 Phase 2b: ``lifecycle_state`` defaults to ``"PUBLISHED"`` —
        a LIVE post — because a ``"DRAFT"`` created via the API is orphaned
        (Composio returns no post id) and is not visible/editable in
        LinkedIn's composer UI, so it can never be promoted to published.
        The reviewable artifact is the local vault draft; the chat HITL
        typed-confirm is the publish gate. Pass ``lifecycle_state="DRAFT"``
        only for tests or a deliberate (invisible) draft.

        Raises ``PublishError`` on empty/oversized body or a stub-mode
        executor (no real credentials → would silently not post)."""
        if getattr(self._executor, "stub_mode", False):
            raise PublishError(
                "Composio is in stub mode (no COMPOSIO_API_KEY) — refusing to "
                "report a publish that would not actually reach LinkedIn."
            )
        body = body.strip()
        if not body:
            raise PublishError("post body is empty; nothing to publish")
        if len(body) > _MAX_POST_CHARS:
            raise PublishError(
                f"post body exceeds {_MAX_POST_CHARS} chars ({len(body)}); trim before publishing"
            )

        author = await self.resolve_author_urn()
        result = await self._executor.execute(
            _PROVIDER,
            _CREATE_POST_TOOL,
            {
                "author": author,
                "commentary": body,
                "visibility": visibility,
                "lifecycleState": lifecycle_state,
            },
        )
        published = lifecycle_state == "PUBLISHED"
        return PublishResult(
            status="published-on-linkedin" if published else "draft-created-on-linkedin",
            author_urn=author,
            post_id=_extract_post_id(result),
            lifecycle_state=lifecycle_state,
            draft_path=draft_path,
        )
