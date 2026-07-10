"""``wiki_repos.synthesize`` — compose ONE markdown wiki source page.

This is the deterministic heart of the repo-as-knowledge-source pipeline
(PRD-012 / ADR-022). Given a validated :class:`RepoRef`, a locally-built
:class:`Digest`, and two optional, already-computed artifacts — a code-graph
``overview`` dict (from :mod:`wiki_qa.codegraph`) and a verbatim Mermaid
``diagram`` string — it returns a complete markdown page (YAML frontmatter +
body) ready to flow through ``write_page``'s ADR-006 trust envelope.

Design rules: deterministic by default (synchronous, stdlib-only, no network —
identical inputs + ``clock`` give byte-identical output); no fabrication (every
fact comes from ``digest`` or ``graph_overview``); graceful degradation (LLM
prose-polishing lives behind the *separate* async :func:`refine_prose`, which
returns its input unchanged if a router is absent or fails).

The frontmatter mirrors what ``write_page`` validates (title, date, sources,
tags, origin) plus repo-specific keys (``repo``, ``graph``, ``draft_mode``).
``origin`` is ``llm-synthesized`` only when a router actually refined the prose;
the pure deterministic page is honestly stamped ``repo-digest``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from wiki_repos.types import Digest, DraftMode, RepoHistory, RepoRef

__all__ = [
    "synthesize_page",
    "refine_prose",
    "repo_slug",
    "page_path_for",
]

# Languages we will tag (lowercased) when the digest tree/content reveals them.
# Order matters only for determinism of the tag list.
_LANG_EXTENSIONS: tuple[tuple[str, str], ...] = (
    (".py", "python"),
    (".ts", "typescript"),
    (".tsx", "typescript"),
    (".js", "javascript"),
    (".jsx", "javascript"),
    (".rs", "rust"),
    (".go", "go"),
    (".java", "java"),
    (".rb", "ruby"),
    (".kt", "kotlin"),
    (".swift", "swift"),
    (".c", "c"),
    (".cpp", "cpp"),
    (".cs", "csharp"),
    (".php", "php"),
)

# Matches a file block header so we can pull a README excerpt. Supports both the
# ``===== <path> =====`` form emitted by ``wiki_repos.digest`` AND the classic
# gitingest ``FILE: <path>`` form, so the excerpt/language fallbacks fire on real
# digest output (not only on test fixtures).
_FILE_HEADER_RE = re.compile(
    r"(?:=====\s*(?P<path_eq>.+?)\s*=====|FILE:\s*(?P<path_file>[^\n]+))\n",
    flags=re.IGNORECASE,
)

_MAX_DESC_CHARS = 90
_MAX_README_LINES = 3
_MAX_CONTEXTS = 6
_MAX_HUBS = 8
_MAX_PRS_SHOWN = 8  # cap rendered PRs in the evolution section (PRD-014)
_MAX_FIXES_SHOWN = 6


# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #
def repo_slug(ref: RepoRef) -> str:
    """Return the page slug for ``ref`` (re-export of :attr:`RepoRef.slug`)."""
    return ref.slug


def page_path_for(ref: RepoRef) -> str:
    """Return the wiki-relative path a synthesized page should be written to."""
    return f"wiki/sources/{ref.slug}.md"


# --------------------------------------------------------------------------- #
# Core composition
# --------------------------------------------------------------------------- #
def synthesize_page(
    ref: RepoRef,
    digest: Digest,
    *,
    graph_overview: dict | None = None,
    diagram: str = "",
    history: RepoHistory | None = None,
    mode: DraftMode = "showcase",
    router: Any | None = None,
    clock: Callable[[], datetime] | None = None,
    description_override: str | None = None,
    topics: tuple[str, ...] = (),
) -> str:
    """Compose a single markdown wiki source page from a repo digest.

    Args:
        ref: The validated repository reference.
        digest: The locally-built file-tree + content digest.
        graph_overview: Optional output of :func:`wiki_qa.codegraph.overview`
            (keys ``contexts``, ``top_hubs``, ``totals``). ``None`` triggers the
            degraded architecture note.
        diagram: A pre-rendered Mermaid block (including its ```` ```mermaid ````
            fence). Embedded verbatim under ``## Diagram`` when non-empty.
        mode: Draft angle for the downstream LinkedIn drafter — ``"showcase"``
            (third-person tool presentation) or ``"experiential"`` (first-person
            "I used this in my project").
        router: Presence flips ``origin`` to ``llm-synthesized`` (signalling the
            caller intends to / has run :func:`refine_prose`). This function does
            **not** call the router itself; it stays synchronous and pure.
        clock: Injectable ``() -> datetime`` for deterministic timestamps;
            defaults to ``datetime.now(UTC)``.
        description_override: The GitHub ``description`` one-liner (ADR-025). When
            non-empty it leads the ``## What it is`` section — the repo's own value
            prop is better than the digest summary at saying *what the tool does*.
            The digest summary / README excerpt still follow it. ``None`` or empty
            leaves the original digest-only behaviour unchanged.
        topics: The repo's GitHub ``topics`` (ADR-025) — its own positioning
            keywords (e.g. ``compression``, ``token-optimization``). Merged into
            the frontmatter ``tags`` (deduped against the detected languages) and
            surfaced as a short ``Topics:`` line in the body. Empty by default.

    Returns:
        A complete markdown document: YAML frontmatter block followed by the
        rendered body.
    """
    now = (clock or (lambda: datetime.now(UTC)))()

    languages = _detect_languages(digest)
    topics = _clean_topics(topics)
    description_override = (description_override or "").strip()
    description = description_override or _short_description(ref, digest)
    origin = "llm-synthesized" if router is not None else "repo-digest"

    frontmatter = _render_frontmatter(
        ref=ref,
        description=description,
        now=now,
        origin=origin,
        languages=languages,
        topics=topics,
        graph_overview=graph_overview,
        mode=mode,
    )
    body = _render_body(
        ref=ref,
        digest=digest,
        description=description,
        description_override=description_override,
        topics=topics,
        graph_overview=graph_overview,
        diagram=diagram,
        history=history,
        mode=mode,
    )
    return f"{frontmatter}\n{body}"


# --------------------------------------------------------------------------- #
# Frontmatter
# --------------------------------------------------------------------------- #
def _render_frontmatter(
    *,
    ref: RepoRef,
    description: str,
    now: datetime,
    origin: str,
    languages: list[str],
    topics: tuple[str, ...],
    graph_overview: dict | None,
    mode: DraftMode,
) -> str:
    """Build the YAML frontmatter block (including the surrounding ``---``).

    The ``tags`` line merges the structural tags, the detected ``languages``, and
    the repo's own GitHub ``topics`` (ADR-025), de-duplicated while preserving the
    structural → language → topic order for deterministic output.
    """
    title = f"{ref.repo} — {description}"
    tags = _dedupe(["repo", "code", *languages, *topics])
    tag_list = ", ".join(tags)
    graph_state = "available" if graph_overview else "unavailable"

    lines = [
        "---",
        f'title: "{_yaml_escape(title)}"',
        f"date: {now.date().isoformat()}",
        f"created_at: {now.isoformat()}",
        f"origin: {origin}",
        f'sources: ["{ref.canonical_url}"]',
        f"tags: [{tag_list}]",
        "category: sources",
        f"repo: {ref.owner}/{ref.repo}",
        f"graph: {graph_state}",
        f"draft_mode: {mode}",
        "---",
    ]
    return "\n".join(lines)


def _yaml_escape(value: str) -> str:
    """Escape a string for a double-quoted YAML scalar."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


# --------------------------------------------------------------------------- #
# Body
# --------------------------------------------------------------------------- #
def _render_body(
    *,
    ref: RepoRef,
    digest: Digest,
    description: str,
    description_override: str,
    topics: tuple[str, ...],
    graph_overview: dict | None,
    diagram: str,
    history: RepoHistory | None,
    mode: DraftMode,
) -> str:
    """Assemble the markdown body sections."""
    sections: list[str] = [
        f"# {ref.repo}",
        _what_it_is(digest, description, description_override, topics),
        _capabilities(digest),
        _architecture(graph_overview),
    ]

    diagram = diagram.strip()
    if diagram:
        sections.append(f"## Diagram\n\n{diagram}")

    evolution = _evolution_section(history)
    if evolution:
        sections.append(evolution)

    sections.append(_key_modules(digest, graph_overview))
    sections.append(_draft_angle_note(mode))

    return "\n\n".join(s for s in sections if s).rstrip() + "\n"


# Markdown metacharacters that let mined PR/commit text break out of its bullet:
# a leading '#' (heading), backticks/fences (code), and link/image syntax.
_INERT_RE = re.compile(r"[`#\[\]]")


def _inert(text: str) -> str:
    """Render mined PR/commit text inert as markdown (SR-2 / ADR-006 envelope).

    Mined bodies are untrusted; escaping the structural metacharacters
    (``#`` ``` ` ``` ``[`` ``]``) keeps a hostile heading/fence/link rendering as
    literal text in its bullet rather than as live markdown. Newlines are
    collapsed so a multi-line body cannot inject extra list items.
    """
    collapsed = " ".join(text.split())
    return _INERT_RE.sub(lambda m: "\\" + m.group(0), collapsed)


def _evolution_section(history: RepoHistory | None) -> str:
    """The '## Evolution & decisions' section (PRD-014 / ADR-030).

    Deterministic, fact-only: a digest of recent merged PRs (where design rationale
    lives) plus a Conventional-Commit activity breakdown. When a router is present,
    the separate :func:`refine_prose` pass only *rewords* this assembled section and
    is forbidden from adding facts (the narrative is deterministic-first, reword-
    optional — never independently LLM-generated). With no router the bulleted
    digest stands on its own (the degrade path). Returns ``""`` when no history was
    mined — the section is simply omitted (mirrors ``## Diagram``).

    Mined PR/commit text is untrusted and rendered inert via :func:`_inert` (SR-2).
    """
    if history is None or (not history.merged_prs and not history.commits):
        return ""

    lines = ["## Evolution & decisions", ""]

    if history.commits:
        kinds = history.kind_counts
        # Lead with the change-shaping kinds; chore/docs/other are noise here.
        highlight = [k for k in ("fix", "feat", "refactor", "perf", "security") if k in kinds]
        breakdown = ", ".join(f"{kinds[k]} {k}" for k in highlight) or "mixed"
        lines.append(f"Recent activity across {history.stats.n_commits} commits — {breakdown}.")
        lines.append("")

    if history.merged_prs:
        lines.append("Notable merged pull requests (design rationale lives here):")
        lines.append("")
        for pr in history.merged_prs[:_MAX_PRS_SHOWN]:
            date = pr.merged_at[:10]
            excerpt = _inert(pr.body_excerpt)
            tail = f" — {excerpt}" if excerpt else ""
            lines.append(f"- **#{pr.number}** {_inert(pr.title)} ({date}){tail}")
        lines.append("")

    if history.commits:
        fixes = [c for c in history.commits if c.kind in ("fix", "security")]
        if fixes:
            lines.append("Recent fixes:")
            lines.append("")
            for c in fixes[:_MAX_FIXES_SHOWN]:
                lines.append(f"- `{c.sha}` {_inert(c.summary)}")

    return "\n".join(lines).rstrip()


_CAPABILITY_HEADINGS = (
    # use / how-to
    "feature",
    "usage",
    "use",
    "install",
    "getting started",
    "quick start",
    "quickstart",
    "example",
    "why",
    "benchmark",
    "performance",
    "what you can",
    "para qué",
    "uso",
    "instalación",
    "ejemplo",
    # purpose / value / results (ADR-024 A1 — focus=use needs the *what & why*,
    # not only install steps; headroom's value lived under Purpose / Token
    # Reduction / How It Works / Problem Solved and was being dropped).
    "purpose",
    "propósito",
    "overview",
    "about",
    "acerca",
    "what is",
    "what it does",
    "qué es",
    "qué hace",
    "how it works",
    "cómo funciona",
    "capabilit",
    "capacidad",
    "token",
    "cost",
    "saving",
    "ahorro",
    "result",
    "resultado",
    "problem",
    "problema",
    "benefit",
    "beneficio",
    "highlight",
    # proof / numbers / get-started / alternatives — the headline benchmark
    # numbers often sit under a "Proof"/"Benchmarks" heading in tables, several
    # sections deep (headroom's 17,765→1,408 / 92% lived under "## Proof").
    "proof",
    "prueba",
    "evidence",
    "get started",
    "start",
    "demo",
    "when to",
    "cuándo",
    "compared",
    "compar",
    "vs ",
)
# Budget must reach value sections that come *several* headings deep (e.g. a
# "## Proof" benchmark table after What-it-does / How-it-works / Get-started),
# while still bounding how much README we paste into the page.
_MAX_CAPABILITY_CHARS = 3000


def _capabilities(digest: Digest) -> str:
    """Use/value section drawn from the README's Features/Usage/Install/Benchmark
    headings (ADR-024 A1 — gives `focus=use` posts real material to draw from).
    Returns ``""`` when the README has no such sections."""
    readme = ""
    for path, text in _split_file_blocks(digest.content):
        if "readme" in path.lower():
            readme = text
            break
    if not readme:
        return ""

    lines = readme.splitlines()
    captured: list[str] = []
    keep = False
    for raw in lines:
        stripped = raw.strip()
        is_heading = stripped.startswith("#")
        if is_heading:
            title = stripped.lstrip("#").strip().lower()
            keep = any(h in title for h in _CAPABILITY_HEADINGS)
            if keep:
                captured.append(f"**{stripped.lstrip('#').strip()}**")
            continue
        if keep and stripped and _is_prose_line(stripped):
            captured.append(stripped)
        if sum(len(c) for c in captured) > _MAX_CAPABILITY_CHARS:
            break
    if not captured:
        return ""
    body = "\n".join(captured)[:_MAX_CAPABILITY_CHARS].rstrip()
    return f"## Para qué sirve y cómo empezar\n\n{body}"


def _what_it_is(
    digest: Digest,
    description: str,
    description_override: str = "",
    topics: tuple[str, ...] = (),
) -> str:
    """The 'What it is' section — leads with WHAT THE TOOL DOES.

    When ``description_override`` (the GitHub one-line value prop, ADR-025) is
    present it leads the section, because it states the tool's purpose more
    clearly than the digest summary or code structure. The digest summary and a
    cheap README excerpt still follow it so no detail is lost. Without an
    override the original digest-only behaviour is preserved exactly.

    ``topics`` (the repo's own GitHub positioning keywords) are rendered as a
    short trailing ``Topics: a, b, c`` line so the page also carries the repo's
    self-described focus.
    """
    description_override = (description_override or "").strip()
    parts = ["## What it is", ""]

    summary = digest.summary.strip()
    if description_override:
        parts.append(description_override)
        # Keep the digest summary / fallback after the value prop (avoid an exact
        # duplicate if the override and summary happen to match).
        if summary and summary != description_override:
            parts.extend(["", summary])
    else:
        parts.append(summary or description)

    excerpt = _readme_excerpt(digest.content)
    if excerpt:
        parts.extend(["", excerpt])

    topics = _clean_topics(topics)
    if topics:
        parts.extend(["", f"Topics: {', '.join(topics)}"])

    return "\n".join(parts)


def _architecture(graph_overview: dict | None) -> str:
    """The '## Architecture' section, with a degraded note when no graph."""
    if not graph_overview:
        return (
            "## Architecture\n\n"
            "code-graph unavailable for this repo's language; "
            "summary derived from the digest only."
        )

    lines = ["## Architecture", ""]

    totals = graph_overview.get("totals") or {}
    encapsulation = totals.get("encapsulation_pct")
    if encapsulation is not None:
        n_ctx = totals.get("contexts")
        suffix = f" across {n_ctx} bounded contexts" if n_ctx else ""
        lines.append(
            f"Encapsulation: **{encapsulation}%** of imports stay within their context{suffix}."
        )
        lines.append("")

    contexts = _as_list(graph_overview.get("contexts"))
    if contexts:
        lines.append("Top bounded contexts (by node count):")
        for ctx in contexts[:_MAX_CONTEXTS]:
            name = ctx.get("name", "?")
            nodes = ctx.get("nodes")
            files = ctx.get("files")
            detail = []
            if nodes is not None:
                detail.append(f"{nodes} nodes")
            if files is not None:
                detail.append(f"{files} files")
            suffix = f" ({', '.join(detail)})" if detail else ""
            lines.append(f"- `{name}`{suffix}")
        lines.append("")

    hubs = _as_list(graph_overview.get("top_hubs"))
    if hubs:
        lines.append("Most-imported files (graph hubs):")
        for hub in hubs[:_MAX_HUBS]:
            f = hub.get("file", "?")
            n = hub.get("imported_by")
            suffix = f" — imported by {n}" if n is not None else ""
            lines.append(f"- `{f}`{suffix}")

    return "\n".join(lines).rstrip()


def _key_modules(digest: Digest, graph_overview: dict | None) -> str:
    """A short bullet list — prefer graph hubs, else top digest-tree entries."""
    lines = ["## Key modules", ""]

    hubs = _as_list(graph_overview.get("top_hubs")) if graph_overview else []
    if hubs:
        for hub in hubs[:_MAX_HUBS]:
            f = hub.get("file")
            if f:
                lines.append(f"- `{f}`")
        return "\n".join(lines).rstrip()

    # Fall back to the digest's file tree: list the first few file-looking lines.
    for entry in _tree_top_entries(digest.tree):
        lines.append(f"- `{entry}`")
    if len(lines) == 2:  # nothing usable found
        lines.append("- (see file tree in the digest)")
    return "\n".join(lines).rstrip()


def _draft_angle_note(mode: DraftMode) -> str:
    """Trailing note that tells the LinkedIn drafter how to angle the post."""
    if mode == "experiential":
        return (
            "> Draft angle: experiential — write in the first person, "
            "as a tool I used in my project and what I learned from it."
        )
    return (
        "> Draft angle: showcase — present the tool in the third person: "
        "what it is, who it is for, and why it matters."
    )


# --------------------------------------------------------------------------- #
# Digest mining (cheap, fact-only)
# --------------------------------------------------------------------------- #
def _short_description(ref: RepoRef, digest: Digest) -> str:
    """A short, factual description for the title — never fabricated.

    Prefers the first clause of the digest summary (stripping a leading
    ``owner/repo:`` prefix if present), then a README first line, then a plain
    fallback that states only what we know.
    """
    summary = digest.summary.strip()
    if summary:
        # Drop a leading "owner/repo:" or "repo:" prefix if the digest added one.
        summary = re.sub(r"^[\w./-]+:\s*", "", summary, count=1)
        clause = re.split(r"(?<=[.!?])\s|\n", summary, maxsplit=1)[0].strip()
        if clause:
            return _truncate(clause, _MAX_DESC_CHARS)

    excerpt = _readme_excerpt(digest.content, max_lines=1)
    if excerpt:
        return _truncate(excerpt.lstrip("# ").strip(), _MAX_DESC_CHARS)

    return f"{ref.owner}/{ref.repo} repository"


def _readme_excerpt(content: str, *, max_lines: int = _MAX_README_LINES) -> str:
    """Pull the first prose lines of a README block out of the digest content.

    Looks for a ``FILE: ...README...`` header and returns the first few
    non-empty, non-heading-only lines that follow it, before the next file
    block. Returns ``""`` when no README is present.
    """
    if not content:
        return ""

    blocks = _split_file_blocks(content)
    for path, text in blocks:
        if "readme" not in path.lower():
            continue
        prose: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            # A heading contributes only its text; everything else contributes
            # the whole line. Decorative noise (ASCII-art banners, badges, image
            # and HTML lines, table/HR separators) is skipped so the excerpt is
            # the real one-line description, not the repo's logo.
            candidate = line.lstrip("#").strip() if line.startswith("#") else line
            if not _is_prose_line(candidate):
                continue
            prose.append(candidate)
            if len(prose) >= max_lines:
                break
        if prose:
            return "\n".join(prose)
    return ""


# Box-drawing (U+2500–U+259F) + block-element glyphs used by ASCII-art banners.
_BANNER_CHARS = frozenset("█▀▄▌▐░▒▓") | frozenset(chr(c) for c in range(0x2500, 0x25A0))


def _is_prose_line(line: str) -> bool:
    """True if ``line`` is real README prose, not decoration.

    Rejects markdown images / badges / HTML, horizontal-rule and table-separator
    lines, ASCII-art banners (box-drawing / block glyphs), and lines that are
    mostly non-alphanumeric symbols. Keeps ordinary sentences and table *data*
    rows (which carry benchmark numbers). Used by both the 'What it is' excerpt
    and the capabilities body so neither surfaces a repo's logo banner.
    """
    s = line.strip()
    if not s:
        return False
    if s.startswith(("![", "[![", "<")):  # image / badge / HTML
        return False
    if set(s) <= set("-=*_~|: "):  # horizontal rule or table separator row
        return False
    if any(ch in _BANNER_CHARS for ch in s):  # ASCII-art / box-drawing banner
        return False
    alnum = sum(ch.isalnum() for ch in s)
    if alnum < max(3, len(s) * 0.4):  # mostly symbols → not prose
        return False
    return True


def _split_file_blocks(content: str) -> list[tuple[str, str]]:
    """Split gitingest-style content into ``(path, text)`` blocks."""
    blocks: list[tuple[str, str]] = []
    matches = list(_FILE_HEADER_RE.finditer(content))
    for i, m in enumerate(matches):
        path = (m.group("path_eq") or m.group("path_file") or "").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end]
        # Trim a trailing separator rule of '=' characters if present.
        text = re.sub(r"\n=+\s*$", "", text)
        blocks.append((path, text))
    return blocks


def _detect_languages(digest: Digest) -> list[str]:
    """Return lowercased language tags inferred from file extensions.

    Scans the (cheap) tree text and falls back to file-block headers. Returns a
    deterministic, de-duplicated list ordered by :data:`_LANG_EXTENSIONS`.
    """
    haystack = digest.tree or ""
    if not haystack:
        haystack = "\n".join(p for p, _ in _split_file_blocks(digest.content))

    found: list[str] = []
    lower = haystack.lower()
    for ext, lang in _LANG_EXTENSIONS:
        if ext in lower and lang not in found:
            found.append(lang)
    return found


def _tree_top_entries(tree: str, limit: int = _MAX_HUBS) -> list[str]:
    """Extract up to ``limit`` file-looking entries from a digest tree string."""
    entries: list[str] = []
    for raw in (tree or "").splitlines():
        name = raw.strip().rstrip("/")
        if not name or name.endswith(("/", ":")):
            continue
        # Heuristic: a file entry has an extension dot somewhere in the name.
        leaf = name.split()[-1]
        if "." in leaf and not leaf.startswith("."):
            entries.append(leaf)
        if len(entries) >= limit:
            break
    return entries


def _as_list(value: Any) -> list[Any]:
    """Coerce a possibly-missing value into a list (defensive against bad input)."""
    if isinstance(value, list):
        return value
    return []


def _clean_topics(topics: tuple[str, ...]) -> tuple[str, ...]:
    """Normalise GitHub topics: drop blanks, lowercase, strip, and de-duplicate.

    Defensive against non-tuple / non-string input so the caller can pass
    ``meta.topics`` straight through without pre-validating it.
    """
    if not topics:
        return ()
    cleaned = [t.strip().lower() for t in topics if isinstance(t, str) and t.strip()]
    return tuple(_dedupe(cleaned))


def _dedupe(values: list[str]) -> list[str]:
    """Return ``values`` with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars on a word boundary with an ellipsis."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",.;:")
    return f"{cut}…"


# --------------------------------------------------------------------------- #
# Optional LLM refinement (async, degrades gracefully)
# --------------------------------------------------------------------------- #
async def refine_prose(page_md: str, router: Any | None) -> str:
    """Optionally polish a synthesized page's prose via the model router.

    This is the *only* path that touches an LLM, and it is strictly optional:
    on a missing router or *any* failure it returns ``page_md`` unchanged, so
    the deterministic page is never lost. Callers that want a polished page wrap
    this and use the result; the core :func:`synthesize_page` never depends on
    it.

    The refinement is constrained to rewording: the system prompt forbids adding
    facts, so the no-fabrication guarantee is preserved.

    Args:
        page_md: The deterministic page (frontmatter + body).
        router: A ``ModelRouter``-shaped object exposing an awaitable ``call``
            (or ``route``) coroutine, or ``None``.

    Returns:
        The refined markdown, or ``page_md`` verbatim if refinement was skipped
        or failed.
    """
    if router is None:
        return page_md

    try:
        return await _call_router_for_refine(page_md, router)
    except Exception:  # noqa: BLE001 — degrade on ANY router/transport failure.
        return page_md


_REFINE_SYSTEM_PROMPT = (
    "You are an editor. Improve the readability of the markdown wiki page "
    "below WITHOUT adding, removing, or changing any factual claim, module "
    "name, language, number, or URL. Preserve the YAML frontmatter block and "
    "all headings exactly. Only reword prose for clarity. Return the full "
    "markdown document."
)


async def _call_router_for_refine(page_md: str, router: Any) -> str:
    """Invoke the router with whichever call convention it supports.

    Built dynamically (rather than importing ``wiki_routing`` at module load)
    so the deterministic core has zero runtime dependency on the routing layer.
    Returns the router's text, falling back to ``page_md`` if the response is
    empty or unparseable.
    """
    from wiki_routing import Message, TaskDescriptor  # local import — optional dep

    task = TaskDescriptor(
        intent="synthesize",
        estimated_input_tokens=max(1, len(page_md) // 4),
        has_image=False,
        caller_priority="background",
    )
    messages = [
        Message(role="system", content=_REFINE_SYSTEM_PROMPT),
        Message(role="user", content=page_md),
    ]

    caller = getattr(router, "call", None) or getattr(router, "route", None)
    if caller is None:
        return page_md

    result = await caller(task, messages=messages)
    text = getattr(result, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return page_md
